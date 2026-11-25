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


# ====================================================================== the participant
class SemanticParticipant(Policy):
    """A semantic generalist with its own goal, memory and history (see the module docstring)."""
    name, kind, version = "semantic", "llm", "5.0"

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
               "error": result.get("error")}
        self.history.append(rec)
        self.stats["actions"] += 1
        self.stats["ok_actions"] += int(rec["ok"])
        self.memory.append({"t": result.get("t", 0), "kind": "result",
                            "text": f"You called {action.get('tool')} {R.compact_json(action.get('args', {}), 160)}: "
                                    + R.render_result(result, self.result_chars)})
        if action.get("tool") == "world.find_tools" and result.get("ok"):
            found = [t.get("name") for t in (result.get("result") or {}).get("tools", []) if t.get("name")]
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
        for e in obs.get("events") or []:
            if e.get("type") in ("cmd", "cmd.rejected", "ctr", "know"):
                continue
            self.memory.append({"t": e.get("t", t), "kind": "event",
                                "text": f"{e.get('type')} from {e.get('actor')}: {R.compact_json(e.get('payload'), 260)}"})
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

    def candidates(self, tools: list[dict]) -> list[dict]:
        by = {t["name"]: t for t in tools if not t["name"].startswith("admin.")}
        q = self.goal + " " + " ".join(m["text"] for m in self.memory[-3:])
        names = list(by)
        sc = BM25([tool_text(by[n]) for n in names]).scores(q)
        ranked = [n for _, n in sorted(zip(sc, names), key=lambda x: (-x[0], x[1])) if _ > 0]
        used_ok = [h["tool"] for h in reversed(self.history) if h["ok"] and h["tool"] in by]
        k = self.n_candidates
        pick = [n for n in self.discovered if n in by][:3] + ranked[: k] + used_ok[:2]
        pick = [n for n in dict.fromkeys(pick) if n not in DISCOVERY][: k - 1]
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
        lines = "\n".join(f"- {tool_line(t, self.tool_chars)}" for t in cands)
        user = (f"{self.header()}\n\nTools you can call now:\n{lines}\n\n"
                "Which tool do you call next to make progress on your goal? Answer with the tool name only.")
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "choices": [t["name"] for t in cands], "prefix_messages": 1}

    def args_request(self, tool: dict) -> dict:
        schema = tool.get("inputSchema") or {}
        props = schema.get("properties") or {}
        req = schema.get("required") or []
        fields = []
        for n in sorted(props, key=lambda n: (n not in req, n)):
            p = props[n]
            d = (p.get("description") or "").strip()
            fields.append(f"  {n} ({type_str(p)}{', required' if n in req else ', optional'}){': ' + d if d else ''}")
        user = (f"{self.header()}\n\nYou call {tool['name']}: {' '.join((tool.get('description') or '').split())}\n"
                f"Arguments:\n" + "\n".join(fields) + "\n\n"
                "Write the arguments as one JSON object on one line. Include every required argument, copy ids "
                "exactly from what you know, and leave out optional arguments you do not need.")
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                "max_new_tokens": self.arg_tokens, "temperature": 0.0, "seed": 0, "prefix_messages": 1}


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

    def act_batch(self, items: list[tuple]) -> list[dict | None]:
        if not items:
            return []
        parts: list[SemanticParticipant] = [p for p, _, _ in items]
        cands = []
        for p, obs, tools in items:
            p.ingest(obs)
            cands.append(p.candidates(tools))
        reqs = [p.choice_request(c) for p, c in zip(parts, cands)]
        outs = self.infer("choose", reqs, [getattr(p, "actor", "") for p in parts], "tool_choice")
        chosen = []
        for p, c, o in zip(parts, cands, outs):
            p.turn += 1
            p.stats["turns"] += 1
            p.stats["choose_calls"] += 1
            p.stats["prompt_tokens"] += int(o.get("prompt_tokens", 0))
            chosen.append(c[int(o["choice"])])
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
            reqs = [parts[i].args_request(chosen[i]) for i in need]
            outs = self.infer("generate", reqs, [getattr(parts[i], "actor", "") for i in need], "tool_args")
            for i, o in zip(need, outs):
                p = parts[i]
                p.stats["generate_calls"] += 1
                p.stats["prompt_tokens"] += int(o.get("prompt_tokens", 0))
                p.stats["gen_tokens"] += int(o.get("gen_tokens", 0))
                args = parse_json_object(o["text"])
                if args is None:
                    p.stats["parse_failures"] += 1
                    p.memory.append({"t": p.memory[-1]["t"] if p.memory else 0, "kind": "note",
                                     "text": f"Your arguments for {chosen[i]['name']} were not valid JSON."})
                    continue
                if "args" in args and isinstance(args["args"], dict) and len(args) <= 2:
                    args = args["args"]          # the model wrapped the call as {"tool":..., "args": {...}}
                actions[i] = {"tool": chosen[i]["name"], "args": args}
        return actions


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
           "consequential_actions", "tool_line", "tool_text"]
