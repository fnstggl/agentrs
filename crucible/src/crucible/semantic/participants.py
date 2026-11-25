"""Compact semantic generalists (MASTER_SPEC Part 3.2 class 4, Gate H): `crucible.agents.base.Policy` participants that
share one `SemanticService` and act only through ordinary world tools.

Each participant has its own principal, goal, memory and history; the only thing participants share is the model
weights behind the service. A turn is one tool call, decided in two short model steps that are batched across every
participant deciding at the same time (`SemanticGroup.act_batch`):

1. **Choose a tool.** The prompt holds the goal, a short retrieved context and a candidate list of tools. The model's
   log probability of each tool name decides (`SemanticService.choose`). Candidates are the tools the participant
   discovered itself (results of its own ``world.find_tools`` calls), the available tools most relevant to its goal
   and recent experience (BM25 over names and descriptions: retrieval of relevant state, MASTER_SPEC 13.4), the
   discovery tools themselves, and ``wait``. Nothing about which tool achieves which goal is written anywhere.
2. **Write the arguments.** For the chosen tool the model sees its signature, description and argument schema and
   writes one JSON object (`SemanticService.generate`, bounded length). Invalid arguments are not repaired by code:
   the kernel rejects the call and the error becomes part of the participant's history for its next turn.

Context is short by construction (MASTER_SPEC 13.4): the goal, the participant's current state (balances, unread
mail), the result of its last action, and the few memory items (observed events, own action results) that BM25
ranks most relevant to the goal and the latest situation, within a character budget.

All model calls go through `crucible.semantic.recording.record_inference` when the group has a world with the
``semantic_inference`` module, so the world log holds every decision's provenance and replay never reruns the model.

`BatchedRunner` is a `crucible.agents.base.Runner` whose rounds decide all participants of one `SemanticGroup` with
one batched call per step. `AsyncRunner` uses the same `batch_group.act_batch` hook.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

import numpy as np

from ..agents.base import Policy, Runner, Step, _compact
from ..api import render as R
from ..core import schema as S

SYSTEM_PROMPT = (
    "You are a participant in a simulated digital world: a small internet with people, companies, money, mail, "
    "documents, forums, code repositories and cloud services. You can change the world only by calling tools. "
    "Each turn you call exactly one tool; then time passes and you see the result. Money amounts are integer cents "
    "(100 cents = $1). Identifiers such as p:name (people), acct_000123 (bank accounts) or mail_000004 (messages) "
    "must be copied exactly. Use world.find_tools to search for tools you do not see yet. Use only facts that are "
    "written in your goal or your observations; never invent ids. When your goal is done, choose wait.")
WAIT = {"name": "wait", "description": "Do nothing this turn (your goal is done, or you are waiting for someone).",
        "inputSchema": {"type": "object", "properties": {}}}
DISCOVERY = ("world.find_tools",)
THINKING_MODELS = ("qwen3",)      # model families with a native, switchable thinking mode
LETTERS = "ABCDEFGHIJKLMNOP"
_TOKEN = re.compile(r"[a-z0-9]+")


# ====================================================================== retrieval
def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a small fixed corpus (deterministic)."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [{t: d.count(t) for t in set(d)} for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / (self.avg or 1.0))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


def tool_text(t: dict) -> str:
    props = ((t.get("inputSchema") or {}).get("properties") or {})
    name = t.get("name", "")
    return f"{name} {name.replace('.', ' ').replace('_', ' ')} {t.get('description', '')} {' '.join(props)}"


def type_str(s: dict) -> str:
    """Short type of a JSON-schema property: str, int, a|b|c (enums), str[] ..."""
    if "enum" in s:
        return "|".join(str(v) for v in s["enum"])
    t = s.get("type", "any")
    if isinstance(t, list):
        t = "|".join(t)
    if t == "array" and isinstance(s.get("items"), dict):
        return f"{type_str(s['items'])}[]"
    return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj"}.get(t, t)


def tool_line(t: dict, chars: int = 150) -> str:
    return R.render_tool(t, chars, whole_description=True)


def parse_json_object(text: str) -> dict | None:
    """The first JSON object in `text` (code fences and prose around it are ignored)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    frag = text[start:i + 1]
                    for cand in (frag, frag.replace("'", '"')):
                        try:
                            v = json.loads(cand)
                            if isinstance(v, dict):
                                return v
                        except ValueError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


OTHER = "(a value not listed: I will write it)"
OMIT = "(leave this optional argument out)"
TEXT_ARGS = ("body", "text", "message", "description", "title", "subject", "note", "bio", "query", "q", "reason",
             "comment", "name", "memo", "label", "purpose", "path", "target", "url", "handle", "need", "keywords")
_IDS = re.compile(r"\b(?:[a-z]{1,8}:[a-z0-9][a-z0-9_.-]*[a-z0-9]|[a-z]+_\d{3,}|[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*)\b")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_QUOTED = re.compile(r"'([^']{2,80})'|\"([^\"]{2,80})\"")


def coerce_value(text: str, schema: dict, name: str = ""):
    """A written value converted to the type the schema asks for (None if impossible). A leading ``name:`` echo
    of the argument's name, surrounding quotes and empty answers (None, null) are removed."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```\w*\n?|```$", "", t).strip()
    if name:
        t = re.sub(rf"^\s*(?:argument\s+)?{re.escape(name)}\s*[:=]\s*", "", t, flags=re.I).strip()
    if t.lower() in ("none", "null", "n/a", '""', "''"):
        return None
    typ = schema.get("type")
    typ = [typ] if isinstance(typ, str) else (typ or ["string"])
    if "object" in typ or "array" in typ and not isinstance(schema.get("items"), dict):
        v = parse_json_object(t)
        return v
    if "array" in typ:
        inner = coerce_value(t, schema["items"])
        return None if inner is None else [inner]
    if "integer" in typ or "number" in typ:
        m = _NUM.search(t.replace(",", ""))
        if not m:
            return None
        x = float(m.group(0))
        return int(x) if "integer" in typ else x
    if "boolean" in typ:
        return t.lower().startswith(("true", "yes"))
    t = t.strip().strip('"').strip("'")
    if (t.startswith("{") or t.startswith("[")):
        try:
            v = json.loads(t)
            if isinstance(v, str):
                t = v
        except ValueError:
            pass
    mx = int(schema.get("maxLength", 4000))
    return t[:mx] if t else None


# ====================================================================== the participant
class SemanticParticipant(Policy):
    """A semantic generalist with its own goal, memory and history (see the module docstring)."""
    name, kind, version = "semantic", "llm", "5.0"
    choice_method = "think"       # how a tool is chosen from the candidates (see choice_request and `select`)
    think_tokens = 384            # bound on the model's own reasoning before it names a tool (choice_method think)
    think_args = False            # arg_mode json only: reason (thinking mode) before writing the JSON
    arg_mode = "structured"       # "structured": schema-guided, one argument at a time; "json": one JSON object

    def __init__(self, group: "SemanticGroup | None", goal: str, *, profile: str = "", n_candidates: int = 8,
                 memory_k: int = 6, context_chars: int = 1800, arg_tokens: int = 160, tool_chars: int = 150,
                 result_chars: int = 900):
        self.group = group                      # None: the participant only builds prompts (benchmark items)
        self.batch_group = group
        self.goal, self.profile = goal, profile
        self.n_candidates, self.memory_k, self.context_chars = n_candidates, memory_k, context_chars
        self.arg_tokens, self.tool_chars, self.result_chars = arg_tokens, tool_chars, result_chars
        self.history: list[dict] = []           # own actions with results (never shared)
        self.memory: list[dict] = []            # observed facts: {t, kind, text}
        self.discovered: list[str] = []         # tools found by own world.find_tools calls, newest first
        self.state_text = ""
        self.last_result: dict | None = None
        self.turn = 0
        self._read_only: dict[str, bool] = {}    # tool -> readOnlyHint, from the tool lists this participant saw
        self.stats = {"turns": 0, "choose_calls": 0, "generate_calls": 0, "waits": 0, "parse_failures": 0,
                      "actions": 0, "ok_actions": 0, "prompt_tokens": 0, "gen_tokens": 0}

    # ------------------------------------------------------------------ Policy interface
    def reset(self, actor, rng):
        super().reset(actor, rng)
        self.history, self.memory, self.discovered = [], [], []
        self.last_result, self.turn, self.state_text = None, 0, ""

    def act(self, obs, tools):
        return self.group.act_batch([(self, obs, tools)])[0]

    def observe_result(self, action, result):
        self.last_result = result
        rec = {"t": result.get("t"), "tool": action.get("tool"), "args": action.get("args", {}),
               "ok": bool(result.get("ok")), "result": result.get("result") if result.get("ok") else None,
               "error": result.get("error"), "read_only": bool(self._read_only.get(action.get("tool"), False))}
        self.history.append(rec)
        self.stats["actions"] += 1
        self.stats["ok_actions"] += int(rec["ok"])
        self.memory.append({"t": result.get("t", 0), "kind": "result",
                            "text": f"You called {action.get('tool')} {R.compact_json(action.get('args', {}), 160)}: "
                                    + R.render_result(result, self.result_chars)})
        if action.get("tool") == "world.find_tools" and result.get("ok"):
            found = [t.get("name") for t in (result.get("result") or {}).get("results", []) if t.get("name")]
            self.discovered = list(dict.fromkeys(found + self.discovered))[:12]

    def usage(self) -> dict:
        return {"model": self.group.service.model_id if self.group is not None else None, **self.stats}

    def history_digest(self) -> str:
        return hashlib.sha256(json.dumps([[h["tool"], h["args"], h["ok"]] for h in self.history],
                                         sort_keys=True, default=str).encode()).hexdigest()

    # ------------------------------------------------------------------ observation intake
    def ingest(self, obs: dict) -> None:
        t = obs.get("time", 0)
        parts = []
        for k in sorted(obs):
            if k in ("time", "principal", "events", "events_dropped"):
                continue
            parts.append(f"{k}: {R.compact_json(obs[k], 240)}")
        self.state_text = "; ".join(parts)
        new = 0
        for e in obs.get("events") or []:
            if e.get("type") in ("cmd", "cmd.rejected", "ctr", "know") or e.get("actor") == getattr(self, "actor", None):
                continue
            new += 1
            self.memory.append({"t": e.get("t", t), "kind": "event",
                                "text": f"{e.get('type')} from {e.get('actor')}: {R.compact_json(e.get('payload'), 260)}"})
        if new and self.history:
            self.history[-1]["new_after"] = True
        if len(self.memory) > 400:
            self.memory = self.memory[-400:]

    def context(self) -> str:
        """Short context: the newest items plus the items most relevant to the goal and the latest situation."""
        if not self.memory:
            return "(nothing yet)"
        recent = list(range(max(0, len(self.memory) - 2), len(self.memory)))
        q = self.goal + " " + " ".join(self.memory[i]["text"] for i in recent)
        sc = BM25([m["text"] for m in self.memory]).scores(q)
        ranked = sorted(range(len(self.memory)), key=lambda i: (-sc[i], -i))
        pick = list(dict.fromkeys(recent[::-1] + ranked))[: self.memory_k]
        lines, used = [], 0
        for i in sorted(pick):
            s = f"- [{R.fmt_time(self.memory[i]['t'])}] {self.memory[i]['text']}"
            if used + len(s) > self.context_chars:
                s = s[: max(0, self.context_chars - used)]
            if s:
                lines.append(s)
                used += len(s)
            if used >= self.context_chars:
                break
        return "\n".join(lines)

    def _known_answers(self) -> set[str]:
        """Read-only calls whose answer the participant already holds: the same call succeeded and nothing new has
        been observed since (calling it again cannot tell it anything new)."""
        out = set()
        for h in reversed(self.history):
            if h.get("new_after"):
                break
            if h["ok"] and h.get("read_only"):
                out.add(h["tool"] + R.compact_json(h["args"], None))
        return out

    def candidates(self, tools: list[dict]) -> list[dict]:
        by = {t["name"]: t for t in tools if not t["name"].startswith("admin.")}
        for n, t in by.items():
            self._read_only[n] = bool((t.get("annotations") or {}).get("readOnlyHint"))
        names = sorted(by)
        docs = BM25([tool_text(by[n]) for n in names])
        rank = lambda q: [n for s, n in sorted(zip(docs.scores(q), names), key=lambda x: (-x[0], x[1])) if s > 0]
        by_goal = rank(self.goal)
        recent = " ".join(m["text"] for m in self.memory[-2:] if m["kind"] != "result")
        by_news = rank(recent)[:2] if recent else []
        known = self._known_answers()
        stale = {n for n in by if not (by[n].get("inputSchema") or {}).get("properties") and n + "{}" in known}
        failed = [h["tool"] for h in self.history[-4:] if not h["ok"]]
        stale |= {n for n in set(failed) if failed.count(n) >= 2}      # kept failing: try something else for now
        k = self.n_candidates
        pick = [n for n in self.discovered if n in by][:3] + by_goal[:k] + by_news
        pick = [n for n in dict.fromkeys(pick) if n not in DISCOVERY and n not in stale][: k - 1]
        out = [by[n] for n in pick] + [by[n] for n in DISCOVERY if n in by]
        return out + [WAIT]

    # ------------------------------------------------------------------ prompts
    def recent_actions(self, k: int = 4) -> str:
        if not self.history:
            return "none yet"
        lines = []
        for h in self.history[-k:]:
            if h["ok"]:
                lines.append(f"- {h['tool']} {R.compact_json(h['args'], 120)} -> ok")
            else:
                e = h.get("error") or {}
                lines.append(f"- {h['tool']} {R.compact_json(h['args'], 120)} -> FAILED {e.get('code', '')}: "
                             f"{str(e.get('message', ''))[:140]}")
        return "\n".join(lines)

    def header(self) -> str:
        who = f"You are {getattr(self, 'actor', '?')}." + (f" {self.profile}" if self.profile else "")
        return (f"{who}\nGoal: {self.goal}\nYour state: {self.state_text or 'unknown'}\n"
                f"Your recent actions:\n{self.recent_actions()}\n"
                f"What you know (relevant observations, oldest first):\n{self.context()}")

    def choice_request(self, cands: list[dict]) -> dict:
        """The tool-choice request. ``choice_method`` "mean"/"sum": the model's log probability of each tool name
        (per token, or in total); "label": the tools are lettered and the letters are scored; "think": the model
        first reasons in its own thinking mode (bounded by ``think_tokens``) and names a tool (see `select`)."""
        if self.choice_method == "label":
            lines = "\n".join(f"{LETTERS[i]}) {tool_line(t, self.tool_chars)}" for i, t in enumerate(cands))
            ask = "Which tool do you call next to make progress on your goal? Answer with the letter only."
            choices, norm = [LETTERS[i] for i in range(len(cands))], "sum"
        else:
            lines = "\n".join(f"- {tool_line(t, self.tool_chars)}" for t in cands)
            ask = "Which tool do you call next to make progress on your goal? Answer with the tool name only."
            choices, norm = [t["name"] for t in cands], ("mean" if self.choice_method == "think" else
                                                          self.choice_method)
        user = f"{self.header()}\n\nTools you can call now:\n{lines}\n\n{ask}"
        req = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
               "choices": choices, "normalize": norm, "prefix_messages": 1}
        if self.choice_method == "think":
            req["think_tokens"] = self.think_tokens
        return req

    def args_request(self, tool: dict) -> dict:
        """The argument-writing request: the tool's required arguments in full, its optional ones by name only (small
        models otherwise fill them with invented values), the previous failure of this tool if any. With
        ``think_args`` the model reasons first in its thinking mode, bounded by ``think_tokens`` extra tokens."""
        schema = tool.get("inputSchema") or {}
        props = schema.get("properties") or {}
        req = [n for n in (schema.get("required") or []) if n in props]
        fields = []
        for n in req:
            p = props[n]
            d = (p.get("description") or "").strip()
            fields.append(f"  {n} ({type_str(p)}){': ' + d if d else ''}")
        opt = [n for n in sorted(props) if n not in req]
        if opt:
            fields.append("  optional, leave out unless your goal needs them: " +
                          ", ".join(f"{n} ({type_str(props[n])})" for n in opt))
        prev = next((h for h in reversed(self.history[-6:]) if h["tool"] == tool["name"] and not h["ok"]), None)
        fix = ""
        if prev is not None:
            e = prev.get("error") or {}
            fix = (f"Your earlier call {tool['name']} {R.compact_json(prev['args'], 200)} failed: {e.get('code', '')}: "
                   f"{str(e.get('message', ''))[:200]}. Use different values this time.\n")
        user = (f"{self.header()}\n\nYou call {tool['name']}: {' '.join((tool.get('description') or '').split())}\n"
                f"Arguments:\n" + "\n".join(fields) + "\n" + fix + "\n"
                "Write the arguments as one JSON object on one line. Take every value from your goal or from what you "
                "know (copy ids exactly; never use an id that names something else).")
        r = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
             "max_new_tokens": self.arg_tokens, "temperature": 0.0, "seed": 0, "prefix_messages": 1}
        if self.think_args:
            r["thinking"] = True
            r["max_new_tokens"] = self.arg_tokens + self.think_tokens
        return r

    def _context_values(self) -> tuple[list[str], list[str], list[str]]:
        """Identifiers, numbers and quoted phrases in the participant's goal, state and memory (newest first)."""
        texts = [self.goal, self.state_text] + [m["text"] for m in reversed(self.memory[-40:])]
        ids, nums = [], []
        for t in texts:
            ids += _IDS.findall(t)
            nums += _NUM.findall(t)
        quoted = [a or b for a, b in _QUOTED.findall(self.goal)]      # phrases the goal quotes (observations are JSON)
        own = getattr(self, "actor", None)
        ids = [x for x in dict.fromkeys(ids) if x != own and not x.startswith(("http:", "https:"))]
        return ids, list(dict.fromkeys(nums)), list(dict.fromkeys(quoted))

    def arg_plan(self, tool: dict) -> list[dict]:
        """One slot per argument to fill: {name, schema, kind: choose|write, options, values}."""
        schema = tool.get("inputSchema") or {}
        props = schema.get("properties") or {}
        req = [n for n in (schema.get("required") or []) if n in props]
        ids, nums, quoted = self._context_values()
        slots = []
        for n in sorted(props, key=lambda n: (n not in req, n)):
            ps = props[n]
            typ = ps.get("type")
            typ = [typ] if isinstance(typ, str) else (typ or [])
            item = ps.get("items") if isinstance(ps.get("items"), dict) else {}
            required = n in req
            if "enum" in ps or "boolean" in typ:
                vals = list(ps["enum"]) if "enum" in ps else [True, False]
                opts = [str(v).lower() if isinstance(v, bool) else str(v) for v in vals]
                if not required:
                    vals, opts = vals + [None], opts + [OMIT]
                slots.append({"name": n, "schema": ps, "kind": "choose", "options": opts, "values": vals})
                continue
            if not required:
                continue
            if "object" in typ or ("array" in typ and not item.get("type") in ("string", "integer", "number")):
                slots.append({"name": n, "schema": ps, "kind": "write"})
            elif "integer" in typ or "number" in typ:
                cand = [x for x in nums if ps.get("minimum") is None or float(x) >= float(ps["minimum"])][:8]
                vals = [int(float(x)) if "integer" in typ else float(x) for x in cand]
                if vals:
                    slots.append({"name": n, "schema": ps, "kind": "choose", "options": [str(v) for v in vals] + [OTHER],
                                  "values": vals + [OTHER]})
                else:
                    slots.append({"name": n, "schema": ps, "kind": "write"})
            else:
                target = item if "array" in typ else ps
                textual = n.lower() in TEXT_ARGS or int(target.get("maxLength", 4000)) > 300
                cand = (quoted if textual else []) + ([] if textual and n.lower() not in ("name", "handle") else ids)
                cand = [c for c in dict.fromkeys(cand) if len(c) <= int(target.get("maxLength", 4000)) and
                        ("pattern" not in target or re.fullmatch(target["pattern"], c))][:10]
                if not cand:
                    slots.append({"name": n, "schema": ps, "kind": "write"})
                else:
                    vals = [[c] if "array" in typ else c for c in cand]
                    slots.append({"name": n, "schema": ps, "kind": "choose", "options": cand + [OTHER],
                                  "values": vals + [OTHER]})
        return slots

    def _arg_head(self, tool: dict, slot: dict) -> str:
        ps = slot["schema"]
        d = (ps.get("description") or "").strip()
        prev = next((h for h in reversed(self.history[-6:]) if h["tool"] == tool["name"] and not h["ok"]), None)
        fix = ""
        if prev is not None:
            e = prev.get("error") or {}
            fix = (f"\nYour earlier call {tool['name']} {R.compact_json(prev['args'], 200)} failed: "
                   f"{str(e.get('message', ''))[:200]}")
        return (f"{self.header()}\n\nYou call {tool['name']}: {' '.join((tool.get('description') or '').split())}"
                f"{fix}\nArgument {slot['name']} ({type_str(ps)}){': ' + d if d else ''}")

    def value_choice_request(self, tool: dict, slot: dict) -> dict:
        lines = "\n".join(f"- {o}" for o in slot["options"])
        user = (f"{self._arg_head(tool, slot)}\n\nYour goal: {self.goal}\nWhich value of {slot['name']} does this call "
                f"need for your goal?\n{lines}\nAnswer with the value only.")
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "choices": slot["options"], "normalize": "mean", "prefix_messages": 1}

    def value_write_request(self, tool: dict, slot: dict) -> dict:
        ps = slot["schema"]
        typ = ps.get("type")
        if typ == "object" or (isinstance(typ, list) and "object" in typ):
            how = "Reply with one JSON object only."
            n = self.arg_tokens
        elif typ in ("integer", "number"):
            how, n = "Reply with the number only.", 12
        else:
            mx = int(ps.get("maxLength", 4000))
            n = 24 if mx <= 120 else (64 if mx <= 400 else self.arg_tokens)
            how = "Reply with the value only, no quotes and no explanation."
        user = (f"{self._arg_head(tool, slot)}\n\nYour goal: {self.goal}\nWrite the value of {slot['name']} that this "
                f"call needs for your goal. {how}")
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "max_new_tokens": n, "temperature": 0.0, "seed": 0, "prefix_messages": 1}

    def repeats_known_answer(self, action: dict) -> bool:
        """True if `action` is a read-only call that already succeeded with the same arguments and nothing new has
        been observed since: its answer is already in the participant's memory."""
        return bool(self._read_only.get(action["tool"])) and \
            action["tool"] + R.compact_json(action.get("args", {}), None) in self._known_answers()


# ====================================================================== batched decisions
class SemanticGroup:
    """Decides the turns of many participants that share one `SemanticService`, one batched call per step.

    `act_batch([(participant, obs, tools)]) -> [action | None]` is the `AsyncRunner` batch hook. With ``world`` (which
    must include the ``semantic_inference`` module) every model call is recorded in that world."""

    def __init__(self, service, world=None, record: bool = True):
        self.service = service
        self.world = world
        self.record = record and world is not None
        self.calls = {"choose": 0, "generate": 0, "rows": 0, "wall_s": 0.0}
        self.members: list[SemanticParticipant] = []

    def participant(self, goal: str, **kw) -> SemanticParticipant:
        p = SemanticParticipant(self, goal, **kw)
        if not any(f in self.service.model_id.lower() for f in THINKING_MODELS):
            p.choice_method, p.think_args = "mean", False   # the model has no thinking mode: score the tool names
        need = max(p.think_tokens if p.choice_method == "think" else 0,
                   p.arg_tokens + (p.think_tokens if p.think_args else 0))
        if need > self.service.max_new_tokens:   # the service would cut the reasoning short, silently
            raise ValueError(f"the service bounds generation at {self.service.max_new_tokens} tokens; participants "
                             f"need {need} (think_tokens + arg_tokens): raise SemanticService(max_new_tokens=...)")
        self.members.append(p)
        return p

    def infer(self, kind: str, requests: list[dict], principals: list[str], purpose: str) -> list[dict]:
        t0 = time.time()
        if self.record:
            from .recording import record_inference
            out = record_inference(self.world, self.service, kind, requests, principals, purpose)
        elif kind == "choose":
            out = self.service.choose(None, requests)
        else:
            out = self.service.generate(None, requests)
        self.calls[kind] += 1
        self.calls["rows"] += len(requests)
        self.calls["wall_s"] += time.time() - t0
        return out

    def select(self, reqs: list[dict], principals: list[str]) -> tuple[list[int], list[dict]]:
        """Pick one option per tool-choice request (`SemanticParticipant.choice_request`). Requests with
        ``think_tokens`` are answered in the model's thinking mode: it reasons (bounded), then names an option; when
        the text after its reasoning names none of them, the options are scored after the closed reasoning."""
        return select_options(self.infer, reqs, principals)

    def act_batch(self, items: list[tuple]) -> list[dict | None]:
        if not items:
            return []
        parts: list[SemanticParticipant] = [p for p, _, _ in items]
        cands = []
        for p, obs, tools in items:
            p.ingest(obs)
            cands.append(p.candidates(tools))
        reqs = [p.choice_request(c) for p, c in zip(parts, cands)]
        picks, notes = self.select(reqs, [getattr(p, "actor", "") for p in parts])
        chosen = []
        for p, c, k, note in zip(parts, cands, picks, notes):
            p.turn += 1
            p.stats["turns"] += 1
            p.stats["choose_calls"] += 1
            p.stats["prompt_tokens"] += int(note.get("prompt_tokens", 0))
            p.stats["gen_tokens"] += int(note.get("gen_tokens", 0))
            chosen.append(c[k])
        actions: list[dict | None] = [None] * len(items)
        need = []
        for i, (p, t) in enumerate(zip(parts, chosen)):
            if t["name"] == "wait":
                p.stats["waits"] += 1
                p.memory.append({"t": p.memory[-1]["t"] if p.memory else 0, "kind": "note", "text": "You waited."})
                continue
            schema = t.get("inputSchema") or {}
            if not (schema.get("properties") or {}):
                actions[i] = {"tool": t["name"], "args": {}}
                continue
            need.append(i)
        if need:
            if all(parts[i].arg_mode == "structured" for i in need):
                filled = self._fill_structured(parts, chosen, need)
            else:
                filled = self._fill_json(parts, chosen, need)
            for i, args in zip(need, filled):
                p = parts[i]
                if args is None:
                    continue
                act = {"tool": chosen[i]["name"], "args": args}
                if p.repeats_known_answer(act):
                    p.stats["repeats_skipped"] = p.stats.get("repeats_skipped", 0) + 1
                    p.memory.append({"t": p.memory[-1]["t"] if p.memory else 0, "kind": "note",
                                     "text": f"You already called {act['tool']} with these arguments and nothing has "
                                             "changed since; its answer is above. Take a different next step."})
                    continue
                actions[i] = act
        return actions

    def _fill_json(self, parts, chosen, need) -> list[dict | None]:
        """arg_mode "json": the model writes all arguments as one JSON object."""
        reqs = [parts[i].args_request(chosen[i]) for i in need]
        outs = self.infer("generate", reqs, [getattr(parts[i], "actor", "") for i in need], "tool_args")
        res = []
        for i, o in zip(need, outs):
            p = parts[i]
            p.stats["generate_calls"] += 1
            p.stats["prompt_tokens"] += int(o.get("prompt_tokens", 0))
            p.stats["gen_tokens"] += int(o.get("gen_tokens", 0))
            text = o["text"].split("</think>", 1)[1] if "</think>" in o["text"] else o["text"]
            args = parse_json_object(text)
            if args is None:
                p.stats["parse_failures"] += 1
                p.memory.append({"t": p.memory[-1]["t"] if p.memory else 0, "kind": "note",
                                 "text": f"Your arguments for {chosen[i]['name']} were not valid JSON."})
            elif "args" in args and isinstance(args["args"], dict) and len(args) <= 2:
                args = args["args"]          # the model wrapped the call as {"tool":..., "args": {...}}
            res.append(args)
        return res

    def _fill_structured(self, parts, chosen, need) -> list[dict | None]:
        """arg_mode "structured": schema-guided filling, one argument at a time and batched across participants.
        Choice-like arguments (enums, booleans, identifiers, numbers) are chosen by the model among the values
        allowed by the schema or present in the participant's own goal and observations (plus "a value not listed",
        which it then writes); text arguments and objects are written by the model. Optional arguments are left out
        except enums and booleans, where leaving them out is one of the options."""
        plans = {i: parts[i].arg_plan(chosen[i]) for i in need}
        choose_q, write_q = [], []
        for i in need:
            for slot in plans[i]:
                (choose_q if slot["kind"] == "choose" else write_q).append((i, slot))
        values: dict[int, dict] = {i: {} for i in need}
        if choose_q:
            reqs = [parts[i].value_choice_request(chosen[i], slot) for i, slot in choose_q]
            outs = self.infer("choose", reqs, [getattr(parts[i], "actor", "") for i, _ in choose_q], "tool_args")
            for (i, slot), o in zip(choose_q, outs):
                parts[i].stats["choose_calls"] += 1
                parts[i].stats["prompt_tokens"] += int(o.get("prompt_tokens", 0))
                v = slot["options"][int(o["choice"])]
                if v == OTHER:
                    write_q.append((i, dict(slot, kind="write")))
                elif v != OMIT:
                    values[i][slot["name"]] = slot["values"][int(o["choice"])]
        if write_q:
            reqs = [parts[i].value_write_request(chosen[i], slot) for i, slot in write_q]
            outs = self.infer("generate", reqs, [getattr(parts[i], "actor", "") for i, _ in write_q], "tool_args")
            for (i, slot), o in zip(write_q, outs):
                p = parts[i]
                p.stats["generate_calls"] += 1
                p.stats["prompt_tokens"] += int(o.get("prompt_tokens", 0))
                p.stats["gen_tokens"] += int(o.get("gen_tokens", 0))
                v = coerce_value(o["text"], slot["schema"], slot["name"])
                if v is None:
                    p.stats["parse_failures"] += 1
                    p.memory.append({"t": p.memory[-1]["t"] if p.memory else 0, "kind": "note",
                                     "text": f"Your value for {slot['name']} of {chosen[i]['name']} was not usable."})
                else:
                    values[i][slot["name"]] = v
        return [values[i] for i in need]


def _named(text: str, options: list[str]) -> int | None:
    """The option named in a free-text answer (the longest option string it contains), or None."""
    t = text.strip().lower()
    hits = [i for i, o in enumerate(options) if o.lower() in t]
    return max(hits, key=lambda i: (len(options[i]), -i)) if hits else None


def select_options(infer, reqs: list[dict], principals: list[str]) -> tuple[list[int], list[dict]]:
    """Shared by participants and the benchmark: `infer(kind, requests, principals, purpose)` runs the model."""
    picks: list[int | None] = [None] * len(reqs)
    notes: list[dict] = [{} for _ in reqs]
    think = [i for i, r in enumerate(reqs) if r.get("think_tokens")]
    plain = [i for i, r in enumerate(reqs) if not r.get("think_tokens")]
    rescore: list[tuple[int, dict]] = []
    if think:
        gen = [{"messages": reqs[i]["messages"], "max_new_tokens": int(reqs[i]["think_tokens"]), "temperature": 0.0,
                "thinking": True, "prefix_messages": reqs[i].get("prefix_messages")} for i in think]
        outs = infer("generate", gen, [principals[i] for i in think], "tool_choice_thinking")
        for i, o in zip(think, outs):
            txt = o["text"]
            notes[i] = {"prompt_tokens": o.get("prompt_tokens", 0), "gen_tokens": o.get("gen_tokens", 0),
                        "thought_closed": "</think>" in txt}
            if "</think>" in txt:
                k = _named(txt.split("</think>", 1)[1], reqs[i]["choices"])
                if k is not None:
                    picks[i] = k
                    continue
            thought = txt.split("</think>", 1)[0].replace("<think>", "").strip()
            rescore.append((i, {"messages": reqs[i]["messages"], "choices": reqs[i]["choices"], "normalize": "mean",
                                "thinking": True, "assistant_prefix": f"<think>\n{thought}\n</think>\n\n",
                                "prefix_messages": reqs[i].get("prefix_messages")}))
    batch = [(i, {k: v for k, v in reqs[i].items() if k != "think_tokens"}) for i in plain] + rescore
    if batch:
        outs = infer("choose", [r for _, r in batch], [principals[i] for i, _ in batch], "tool_choice")
        for (i, _), o in zip(batch, outs):
            picks[i] = int(o["choice"])
            notes[i]["prompt_tokens"] = notes[i].get("prompt_tokens", 0) + int(o.get("prompt_tokens", 0))
    return [int(k) for k in picks], notes


# ====================================================================== runner
class BatchedRunner(Runner):
    """`Runner` whose rounds decide all participants sharing a batch group with one batched call per step; other
    policies act one by one. Actions are executed in a seeded order, then time advances by `dt`."""

    def round(self) -> None:
        order = [int(i) for i in self.rng.permutation(len(self.agents))]
        t0 = time.perf_counter()
        obs = {i: self.observe(self.agents[i][0]) for i in order}
        tools = {i: self.tools(self.agents[i][0]) for i in order}
        groups: dict[int, list[int]] = {}
        for i in order:
            g = getattr(self.agents[i][1], "batch_group", None)
            if g is not None:
                groups.setdefault(id(g), []).append(i)
        decided: dict[int, Any] = {}
        for idx in groups.values():
            g = self.agents[idx[0]][1].batch_group
            acts = g.act_batch([(self.agents[i][1], obs[i], tools[i]) for i in idx])
            decided.update(zip(idx, acts))
        wall = (time.perf_counter() - t0) * 1000 / max(1, len(order))
        for i in order:
            actor, pol = self.agents[i]
            action = decided[i] if i in decided else pol.act(obs[i], tools[i])
            res, evs = None, []
            if action is not None:
                r = self.world.execute(actor, action)
                res, evs = r.to_dict(), r.events
                pol.observe_result(action, res)
            reward = 0.0
            obj = self.objectives.get(actor)
            if obj is not None:
                reward = obj.step_reward(self.world, actor)
            if self.record:
                self.trajs[actor].steps.append(Step(self.world.now, actor, pol.name, _compact(obs[i]), len(tools[i]),
                                                    action, res, evs, reward, wall))
        self.world.advance_time(self.world.now + self.dt)


def consequential_actions(world, principals: list[str]) -> list[dict]:
    """Successful commands of `principals` whose tool is not read-only and whose transaction changed module state
    (emitted at least one event other than the kernel's own bookkeeping), read from the event log."""
    ps = set(principals)
    out = []
    by_cmd: dict[int, list] = {}
    for e in world.events_range(0):
        if e.cmd is not None:
            by_cmd.setdefault(e.cmd, []).append(e)
    for evs in by_cmd.values():
        head = evs[0]
        if head.type != "cmd" or head.actor not in ps:
            continue
        ep = world.endpoints.get(head.payload.get("tool"))
        if ep is None or ep.read_only:
            continue
        effects = [e.type for e in evs[1:] if e.type not in ("ctr", "know", "idem", "ext", "note")]
        if effects:
            out.append({"actor": head.actor, "tool": head.payload["tool"], "t": head.t, "effects": effects[:6]})
    return out


__all__ = ["SemanticParticipant", "SemanticGroup", "BatchedRunner", "SYSTEM_PROMPT", "BM25", "parse_json_object",
           "consequential_actions", "tool_line", "tool_text", "select_options"]
