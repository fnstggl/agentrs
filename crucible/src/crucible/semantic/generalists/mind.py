"""One activation of semantic participants: perceive, form an own goal, choose tools, write arguments, act, remember.

Everything a participant knows comes from tool results it received itself (as its own principal, with the kernel's
authorisation) and from its own persistent state; nothing is read from another participant's private state or from
evaluator data. Every world-changing call it makes is chosen by its brain (`brain.py`) - the model picks the tool and
writes or picks every argument; the runtime only formats prompts, retrieves a shortlist and executes the call.

Batched across all participants due in the same population tick (MASTER_SPEC 13.1/13.4): each phase (goal, tool
choice, each argument slot) is one batched request per routed model.

Phases of an activation (`activate`):

1. **perceive** - read-only tools as the participant: its bank accounts, unread mail headers, its published items'
   analytics, its population state (`pop.me`). New facts enter memory; identifiers it saw become known ids.
2. **goal** - a participant without an active goal writes one (one sentence, its own words) from its persona, needs
   (money, unread mail, how its publications fare, time since it last created or talked to anyone) and memory. No
   goal is ever assigned.
3. **act**, up to ``steps`` times: retrieve a shortlist of its available tools for the goal and the latest situation
   (`catalog`), choose one (or ``finish``: the goal is achieved / given up, or ``wait``), fill the tool's argument
   slots one by one (enum and boolean values, numbers and known identifiers are chosen among the values the
   participant has seen; free text is written), execute it with `ctx.call` as the participant, and remember the
   result (an error message is a result too).
4. **persist** - one `semp.update` event per participant with the memory, history, known ids and goals it added.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.errors import CrucibleError
from . import catalog

SYSTEM = (
    "You are a person living in a simulated digital world: a small internet with other people, companies, money, "
    "mail, documents, forums, blogs and social platforms, code repositories and cloud servers. You change the world "
    "only by calling tools, and every call has real consequences: money moves, messages arrive, other people read "
    "what you publish. Money amounts are integer cents (100 cents = $1). Copy identifiers (people such as p123, "
    "accounts such as acct_000123, messages such as mail_000004, content such as c12, artifacts such as sha256:...) "
    "exactly as you saw them. Never invent identifiers or facts.")
FINISH = {"name": "finish", "description": "your goal is achieved, or it cannot be achieved: stop working on it"}
WAIT = {"name": "wait", "description": "do nothing more now (for example while you wait for someone's answer)"}
OTHER = "(another value: I will write it)"
OMIT = "(leave this out)"
COMPOSE = "(new text: I will write it now)"
COMPOSED = "__compose__"          # marker of an argument whose artifact the participant writes in this call
SHORT_TEXT = ("title", "subject", "name", "handle", "board", "q", "query", "label", "role", "slug", "tag", "host",
              "topic", "keywords", "currency", "city", "region")
CODE_FIELDS = ("files", "code", "script", "html", "source", "program", "style", "tests")
NEGOTIATION_HOSTS = ("deal", "commit", "contracts")
PROGRAMMING_TOOLS = ("code.commit", "studio.app", "studio.package", "studio.page", "studio.run", "studio.test",
                     "pkg.register", "pkg.ci", "wf.define")
_NUM = re.compile(r"(?<![\w.$])\d{1,12}(?![\w])")
_ID_PATTERNS = re.compile(r"\b(?:p\d{1,9}|p:[a-z0-9][a-z0-9_.-]{0,40}|o:[a-z0-9][a-z0-9_.-]{0,40}|[a-z]{2,8}_\d{3,9}"
                          r"|c\d{1,12}|sha256:[0-9a-f]{64})\b")
_ID_KEYS = ("id", "ref", "msg", "doc", "repo", "post", "account", "item", "principal", "from", "sender", "to", "owner",
            "org", "instance", "invoice", "listing", "offer", "posting", "question", "thread", "issue", "project",
            "plan", "product", "contract", "proposal", "creator", "author", "url", "host", "channel", "community")


# ============================================================================================== helpers
def clip(s: Any, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def cjson(x: Any, n: int = 300) -> str:
    return clip(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")), n)


def fmt_time(t: int) -> str:
    d, r = divmod(int(t), 86400)
    return f"day {d}, {r // 3600:02d}:{(r % 3600) // 60:02d}"


def money(c: int) -> str:
    return f"${int(c) / 100:,.2f}"


def summarise(x: Any, depth: int = 0, n_items: int = 4, n_chars: int = 160) -> Any:
    """A short view of a tool result for memory and prompts: long strings cut, lists cut to `n_items`."""
    if isinstance(x, dict):
        return {k: summarise(v, depth + 1, n_items, n_chars) for k, v in sorted(x.items())[:14]}
    if isinstance(x, list):
        out = [summarise(v, depth + 1, n_items, n_chars) for v in x[:n_items]]
        if len(x) > n_items:
            out.append(f"... {len(x) - n_items} more")
        return out
    if isinstance(x, str):
        return clip(x, n_chars)
    return x


def harvest_ids(x: Any, label: str, out: dict[str, str], depth: int = 0) -> None:
    """Identifiers in a tool result, with a short label saying where they were seen."""
    if depth > 4 or len(out) > 80:
        return
    if isinstance(x, dict):
        title = next((str(x[k]) for k in ("title", "subject", "name", "kind", "status") if isinstance(x.get(k), str)),
                     "")
        for k, v in sorted(x.items()):
            if isinstance(v, str) and (k in _ID_KEYS or k.endswith("_id") or k.endswith("_account")) and \
                    _ID_PATTERNS.fullmatch(v):
                out.setdefault(v, clip(f"{label} {k}" + (f" ('{title}')" if title and k != "title" else ""), 80))
            elif isinstance(v, (dict, list)):
                harvest_ids(v, label, out, depth + 1)
            elif isinstance(v, int) and k in ("item", "id") and "aud" in label:
                out.setdefault(str(v), clip(f"{label} {k}" + (f" ('{title}')" if title else ""), 80))
    elif isinstance(x, list):
        for v in x[:20]:
            harvest_ids(v, label, out, depth + 1)


OBJECT_ARGS = ("listing", "offer", "posting", "invoice", "product", "contract", "proposal", "plan", "project", "issue",
               "thread", "question", "repo", "post", "reply_to", "parent", "conversation", "instance", "order", "job",
               "package", "paper", "channel", "community", "board", "msg", "in_reply_to")
PRINCIPAL_ARGS = ("to", "recipients", "payer", "payee", "person", "target", "grantee", "member", "principal",
                  "counterparty", "invitee", "user", "seller", "buyer", "assignee", "arbiter", "who", "with")


def compatible(arg: str, schema: dict, value: str) -> bool:
    """Whether a known identifier can be a value of argument `arg` (its kind, read from the argument's name and
    description; patterns and lengths are checked separately)."""
    n, d = arg.lower(), str(schema.get("description") or "").lower()
    if n == "ref" or "artifact" in d:
        return value.startswith("sha256:")
    if "account" in n:
        return value.startswith("acct_")
    if n in ("msg", "in_reply_to"):
        return value.startswith("mail_")
    if n in PRINCIPAL_ARGS or "principal" in d or "handle" in d:
        return bool(re.match(r"^(p\d+$|p:|o:)", value)) or (value.startswith("acct_") and "account" in d)
    if n == "doc":
        return value.startswith("doc_")
    if n == "item":
        return value.isdigit()
    if n in OBJECT_ARGS and re.match(r"^(p\d+$|p:|o:)", value):
        return False                    # a listing, offer, thread ... is not a person (market.buy listing=p10295)
    return not value.startswith(("sha256:", "acct_")) or n in ("id",)


def _specific(arg: str, value: str) -> bool:
    """Optional arguments are offered only when the known identifier is of the arguments's own kind (a message for
    in_reply_to, content for reply_to / post / item ...), not merely compatible in form."""
    n = arg.lower()
    if n in PRINCIPAL_ARGS:
        # an optional addressee (msg.send takes either `to` or `conversation`): a person it knows
        return bool(re.match(r"^(p\d+$|p:|o:)", value))
    if n in ("in_reply_to", "msg"):
        return value.startswith("mail_")
    if n in ("reply_to", "post", "thread", "question", "parent", "item", "id"):
        return bool(re.fullmatch(r"c\d+|post_\d+|[a-z]{2,8}_\d{3,9}", value)) and not value.startswith(("acct_", "mail_"))
    return False


def ids_in_text(s: str) -> list[str]:
    return list(dict.fromkeys(_ID_PATTERNS.findall(s or "")))


# ============================================================================================== the agent's view
@dataclass
class Agent:
    """A participant's persistent state during one activation (row of `semp_state`) and what it adds."""
    i: int
    principal: str
    account: str | None
    persona: dict
    goals: list = field(default_factory=list)
    active: int | None = None
    memory: list = field(default_factory=list)
    history: list = field(default_factory=list)
    known: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    # this activation
    situation: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    add_memory: list = field(default_factory=list)
    add_history: list = field(default_factory=list)
    add_known: dict = field(default_factory=dict)
    goals_changed: set = field(default_factory=set)
    stop: bool = False
    intent: str = ""
    failures: int = 0
    step_log: list = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict) -> "Agent":
        return cls(i=int(row["i"]), principal=row["principal"], account=row.get("account"), persona=row["persona"],
                   goals=[dict(g) for g in row.get("goals", [])], active=row.get("active"),
                   memory=list(row.get("memory", [])), history=list(row.get("history", [])),
                   known=dict(row.get("known", {})), seen=dict(row.get("seen", {})), counts=dict(row.get("counts", {})))

    # ---------------------------------------------------------------- state changes
    def count(self, k: str, n: int = 1) -> None:
        self.counts[k] = int(self.counts.get(k, 0)) + int(n)

    def remember(self, t: int, kind: str, text: str) -> None:
        m = {"t": int(t), "kind": kind, "text": clip(text, 420)}
        self.memory.append(m)
        self.add_memory.append(m)

    def learn_ids(self, ids: dict[str, str]) -> None:
        for k, v in ids.items():
            if k == self.principal:
                continue
            if k not in self.known:
                self.known[k] = v
                self.add_known[k] = v

    def goal(self) -> dict | None:
        if self.active is None:
            return None
        return next((g for g in self.goals if g["id"] == self.active), None)

    def new_goal(self, t: int, text: str) -> dict:
        gid = int(self.counts.get("goals", 0)) + 1
        g = {"id": gid, "text": clip(text, 300), "t": int(t), "status": "active", "steps": 0, "ok": 0,
             "activations": 0, "ended": None}
        self.goals.append(g)
        self.active = gid
        self.goals_changed.add(gid)
        self.count("goals")
        return g

    def end_goal(self, t: int, status: str) -> None:
        g = self.goal()
        if g is not None:
            g["status"], g["ended"] = status, int(t)
            self.goals_changed.add(g["id"])
            self.count(f"goals_{status}")
        self.active = None

    # ---------------------------------------------------------------- prompt parts
    def persona_line(self) -> str:
        p = self.persona
        s = f"You are {self.principal}. " + p.get("about", "")
        if p.get("contacts"):
            s += " People you know: " + ", ".join(p["contacts"]) + "."
        return s

    def recent_actions(self, k: int = 4) -> str:
        hs = self.history[-k:]
        if not hs:
            return "- none yet"
        lines = []
        for h in hs:
            r = "ok" if h["ok"] else f"FAILED ({h.get('code')}): {h.get('error', '')}"
            lines.append(f"- {fmt_time(h['t'])} {h['tool']} {h.get('args', '')} -> {r} {h.get('result', '')}".rstrip())
        return "\n".join(lines)

    def relevant_memory(self, query_vec: np.ndarray | None, k: int = 6) -> str:
        mem = self.memory[-40:]
        if not mem:
            return "- nothing yet"
        recent = list(range(max(0, len(mem) - 3), len(mem)))
        pick = list(recent)
        if query_vec is not None and len(mem) > len(recent):
            rest = [j for j in range(len(mem)) if j not in recent]
            vecs = catalog.embed([mem[j]["text"] for j in rest])
            sc = vecs @ query_vec
            pick += [rest[j] for j in sorted(range(len(rest)), key=lambda j: (-float(np.round(sc[j], 5)), -j))[:k - 3]]
        return "\n".join(f"- {fmt_time(mem[j]['t'])}: {mem[j]['text']}" for j in sorted(set(pick)))

    def header(self, now: int, query_vec=None, with_goal: bool = True) -> str:
        g = self.goal()
        parts = [self.persona_line()]
        if with_goal and g is not None:
            parts.append(f"Your goal (you chose it on {fmt_time(g['t'])}): {g['text']}")
        parts.append(f"Now: {fmt_time(now)}.")
        parts.append("Your situation:\n" + ("\n".join(f"- {s}" for s in self.situation) or "- nothing new"))
        parts.append("What you remember:\n" + self.relevant_memory(query_vec))
        parts.append("Your last actions:\n" + self.recent_actions())
        return "\n".join(parts)


# ============================================================================================== perception
def perceive(ctx, a: Agent) -> None:
    """Read-only look at the participant's own situation (see the module docstring)."""
    now = ctx.now
    a.situation = []
    a.learn_ids({c: "a person you know" for c in a.persona.get("contacts", [])})

    def call(tool: str, args: dict) -> Any:
        try:
            return ctx.call(tool, args, as_principal=a.principal)
        except (CrucibleError, TypeError, ValueError, KeyError, IndexError, AttributeError):
            return None                 # read-only tools: a refusal or fault leaves nothing to undo
    accts = call("bank.accounts", {})
    for r in accts if isinstance(accts, list) else []:
        if isinstance(r, dict) and r.get("account") and r.get("status", "open") == "open":
            a.situation.append(f"money: {money(r.get('balance') or 0)} in your account {r['account']}")
            a.learn_ids({r["account"]: "your own bank account"})
    me = call("pop.me", {})
    if isinstance(me, dict):
        a.situation.append(f"you are {'employed' if me.get('employed') else 'not employed'}; you have "
                           f"{float(me.get('attention_left_min', 0)):.0f} minutes of free time left today")
    inbox = call("mail.inbox", {"unread_only": True, "limit": 5})
    if isinstance(inbox, list):
        seen = set(a.seen.get("mail", []))
        heads = []
        for m in inbox:
            mid = m.get("msg") if isinstance(m, dict) else None
            if not mid:
                continue
            frm = m.get("from") or "?"
            subj = clip(m.get("subject", ""), 60)
            heads.append(f"{mid} from {frm}: '{subj}'")
            a.learn_ids({mid: f"unread mail from {frm} ('{clip(subj, 40)}')"})
            if isinstance(frm, str) and _ID_PATTERNS.fullmatch(frm):
                a.learn_ids({frm: "wrote to you"})
            if mid not in seen:
                a.remember(now, "mail", f"New mail {mid} from {frm}: '{subj}' (unread)")
                seen.add(mid)
        a.seen["mail"] = sorted(seen)[-60:]
        a.situation.append("unread mail: " + ("; ".join(heads) if heads else "none"))
    mine = call("aud.mine", {"limit": 5})
    items = (mine or {}).get("items") if isinstance(mine, dict) else None
    if isinstance(items, list) and items:
        prev = dict(a.seen.get("items", {}))
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            iid = str(it.get("item", it.get("id", "?")))
            an = it.get("analytics") or it
            views = int(an.get("views", 0) or 0)
            lines.append(f"'{clip(it.get('title', iid), 40)}' ({iid}): {views} views, {int(an.get('shares', 0) or 0)} "
                         f"shares")
            if views != int(prev.get(iid, -1)):
                if iid in prev:
                    a.remember(now, "feedback", f"Your item '{clip(it.get('title', iid), 50)}' now has {views} views "
                                                f"(was {prev[iid]})")
                prev[iid] = views
        a.seen["items"] = dict(sorted(prev.items())[-30:])
        a.situation.append("your published items: " + "; ".join(lines))
    a.tools = catalog.usable(ctx.world.available_tools(a.principal))


# ============================================================================================== goals
def needs(a: Agent, now: int) -> list[str]:
    """The participant's needs in words, from its own state (no evaluator data)."""
    out = []
    p = a.persona
    bal = next((s for s in a.situation if s.startswith("money:")), None)
    if bal:
        out.append(bal.replace("money:", "You have") + ".")
    if p.get("income_per_day_cents") is not None:
        out.append(f"Your outside income is about {money(p['income_per_day_cents'])} per day.")
    last = {k: max([h["t"] for h in a.history if h.get("kind") == k and h["ok"]] or [-1]) for k in ("create", "talk")}
    for k, what in (("create", "created or published anything"), ("talk", "written to anyone")):
        if last[k] < 0:
            out.append(f"You have not {what} yet.")
        elif now - last[k] > 2 * 86400:
            out.append(f"You have not {what} for {int((now - last[k]) // 86400)} days.")
    unread = next((s for s in a.situation if s.startswith("unread mail:") and not s.endswith("none")), None)
    if unread:
        out.append("Some people wrote to you and wait for an answer.")
    return out


def goal_request(a: Agent, now: int) -> dict:
    past = [g for g in a.goals[-4:] if g["status"] != "active"]
    hist = "\n".join(f"- '{g['text']}': {g['status']} after {g['steps']} actions" for g in past) or "- none yet"
    user = (f"{a.persona_line()}\nNow: {fmt_time(now)}.\nYour situation:\n" +
            "\n".join(f"- {s}" for s in a.situation) + "\nYour needs: " + " ".join(needs(a, now)) +
            f"\nWhat you remember:\n{a.relevant_memory(None)}\nYour earlier goals:\n{hist}\n\n"
            "Decide what you want to get done next in this world, something you can do yourself with your tools "
            "(for example writing and publishing, answering or asking people, documents, code, payments, buying, "
            "jobs, servers). Write it as one concrete sentence starting with \"I will\".")
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_new_tokens": 48, "temperature": 0.8, "seed": int(a.i * 7919 + len(a.goals)), "prefix_messages": 1}


def clean_goal(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip().strip('"').strip()
    t = t.split("\n")[0].strip()
    m = re.search(r"[Ii] will[^.!?]*[.!?]?", t)
    if m:
        t = m.group(0)
    return clip(t, 240) if len(t) >= 8 else ""


# ============================================================================================== tools and arguments
def type_str(s: dict) -> str:
    if "enum" in s:
        return "|".join(str(v) for v in s["enum"][:8])
    t = s.get("type", "any")
    if isinstance(t, list):
        t = "|".join(map(str, t))
    if t == "array" and isinstance(s.get("items"), dict):
        return f"list of {type_str(s['items'])}"
    return {"string": "text", "integer": "integer", "number": "number", "boolean": "true|false",
            "object": "object"}.get(t, str(t))


def tool_line(t: dict, n: int = 150) -> str:
    sch = t.get("inputSchema") or {}
    props = sch.get("properties") or {}
    req = set(sch.get("required") or [])
    args = ", ".join(f"{k}{'' if k in req else '?'}" for k in props)
    return f"- {t['name']}({args}): {clip(t.get('description', ''), n)}"


def started(a: Agent) -> bool:
    """Whether the participant has already acted on its active goal."""
    g = a.goal()
    return bool(g and int(g.get("steps", 0)) > 0)


def awaiting(a: Agent) -> bool:
    """Whether it has something to wait for: a message, post or offer it sent to others for its active goal
    succeeded (waiting is offered only then; in the first standard-civilisation run the participants, offered "wait"
    after any action, waited out most goals: 17 calls in 72 activations)."""
    g = a.goal()
    if not g:
        return False
    return any(h.get("goal") == g["id"] and h.get("ok") and h.get("kind") == "talk" and not h.get("read_only")
               for h in a.history)


def achieved_something(a: Agent) -> bool:
    """Whether at least one call for its active goal succeeded (only then can it judge the goal achieved: a goal it
    has just set itself, or one on which every call failed, cannot be achieved already)."""
    g = a.goal()
    return bool(g and int(g.get("ok", 0)) > 0)


def intent_request(a: Agent, now: int, qv) -> dict:
    """The participant names its next action in general words (what kind of action, not its details): that phrase
    is what the service directory is searched for. Once something it did for the goal succeeded it may instead say
    that the goal is achieved ("stop"); once it has acted, that it waits for someone ("wait")."""
    alt = ""
    if achieved_something(a):
        alt += "If your last actions above already achieved your goal, answer \"stop\" instead. "
    if awaiting(a):
        alt += "If you are waiting for someone's answer, answer \"wait\" instead."
    # no example phrase: small models copy it (Qwen3-0.6B answered "reply to a message" for unrelated goals)
    user = (f"{a.header(now, qv)}\n\n{alt.strip() + chr(10) if alt else ''}What single action do you take next for "
            "your goal? Describe it in general terms, in two to six words: a verb and what it acts on, without "
            "names, numbers or other details. Answer with the action only.")
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_new_tokens": 16, "temperature": 0.0, "seed": 0, "prefix_messages": 1}


def clean_intent(text: str) -> str:
    """The next-step phrase, or "" when the answer is not one: a tool call written out (``call jobs.offer {...}``,
    seen from Qwen3-1.7B on DEV items) names no action in general words, so the goal itself is then the query."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = t.split("\n")[0].strip().strip('"').strip("'").strip(" .")
    t = re.sub(r"^(next,?\s+)?i\s+will\s+", "", t, flags=re.I)
    if re.search(r"[{}\[\]=]|\b[a-z]+\.[a-z_]+\b|^call\b", t, flags=re.I):
        return ""
    return clip(t, 80)


def endings(a: Agent) -> list[str]:
    """The non-actions offered beside the services: finish once something it did for the goal succeeded, wait once
    it has sent something to others for it (`awaiting`)."""
    return (["finish"] if achieved_something(a) else []) + (["wait"] if awaiting(a) else [])


def service_request(a: Agent, now: int, hosts: list[str], summaries: dict, qv) -> dict:
    lines = "\n".join(catalog.host_line(h, summaries.get(h, ""), a.tools) for h in hosts)
    nxt = f"\nYour next action: {a.intent}" if a.intent else ""
    ends = "".join(f"\n- {e}: {(FINISH if e == 'finish' else WAIT)['description']}" for e in endings(a))
    user = (f"{a.header(now, qv)}{nxt}\n\nServices you can use:\n{lines}{ends}\n\nWhich service do you use for "
            "this step? Answer with the service name only.")
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "choices": list(hosts) + endings(a), "normalize": "mean", "prefix_messages": 1}


def tool_request(a: Agent, now: int, host: str, shortlist: list[dict], qv) -> dict:
    lines = "\n".join(tool_line(t) for t in shortlist)
    nxt = f"\nYour next action: {a.intent}" if a.intent else ""
    user = (f"{a.header(now, qv)}{nxt}\n\nTools of the service {host}:\n{lines}\n\nWhich tool do you call for this "
            "step? Answer with the tool name only.")
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "choices": [t["name"] for t in shortlist], "normalize": "mean", "prefix_messages": 1}


def capability_of(tool: str, arg: str | None = None) -> str:
    host = tool.split(".", 1)[0]
    if tool in PROGRAMMING_TOOLS or (arg or "") in CODE_FIELDS:
        return "programming"
    if host in NEGOTIATION_HOSTS:
        return "negotiation"
    return "communication" if arg else "tool_choice"


def is_text_field(name: str, ps: dict) -> bool:
    return name in SHORT_TEXT or int(ps.get("maxLength", 0) or 0) > 300 or name in (
        "body", "text", "message", "description", "note", "answer", "bio", "reason", "comment", "memo", "summary",
        "content", "html", "script", "style")


@dataclass
class Slot:
    name: str
    schema: dict
    required: bool
    kind: str                     # choose | number | text | json | list | compose
    options: list = field(default_factory=list)
    values: list = field(default_factory=list)
    tokens: int = 24


MAX_OPTIONS = 12


def _recent(a: Agent) -> list[str]:
    """Known identifiers, most recently learned first."""
    return list(reversed(list(a.known)))


def publishes_content(tool: dict, arg: str) -> bool:
    """A publishing or updating tool's artifact argument (``ref`` of *.publish / content.update)."""
    short = tool["name"].split(".", 1)[-1]
    return arg == "ref" and short in ("publish", "update")


def plan_slots(a: Agent, tool: dict, max_slots: int = 6) -> list[Slot]:
    sch = tool.get("inputSchema") or {}
    props = sch.get("properties") or {}
    req = [k for k in (sch.get("required") or []) if k in props]
    out: list[Slot] = []
    ctx_text = " ".join([(a.goal() or {}).get("text", "")] + a.situation + [m["text"] for m in a.memory[-6:]])
    nums = list(dict.fromkeys(_NUM.findall(ctx_text.replace(",", ""))))[:8]
    for k in list(req) + [k for k in props if k not in req]:
        ps = props[k] if isinstance(props[k], dict) else {}
        required = k in req
        typ = ps.get("type")
        typ = [typ] if isinstance(typ, str) else list(typ or [])
        items = ps.get("items") if isinstance(ps.get("items"), dict) else {}
        if "enum" in ps or "boolean" in typ:
            vals = list(ps["enum"]) if "enum" in ps else [True, False]
            opts = [str(v).lower() if isinstance(v, bool) else str(v) for v in vals]
            if not required:
                vals, opts = vals + [None], opts + [OMIT]
            if required or k in ("visibility", "access", "kind", "format", "platform", "role", "sort", "folder"):
                out.append(Slot(k, ps, required, "choose", opts, vals))
            continue
        if not required:
            # an optional reference to something the participant has seen (the message it answers, the post it
            # replies to): offered with the option to leave it out
            if "string" in typ and not ps.get("enum") and not is_text_field(k, ps):
                pat, mx = ps.get("pattern"), int(ps.get("maxLength", 200) or 200)
                ids = [x for x in _recent(a) if len(x) <= mx and (not pat or re.fullmatch(pat, x)) and
                       _ID_PATTERNS.fullmatch(x) and compatible(k, ps, x) and _specific(k, x)][:MAX_OPTIONS]
                if ids:
                    out.append(Slot(k, ps, False, "choose", [f"{x} ({a.known[x]})" for x in ids][:8] + [OMIT],
                                    ids[:8] + [None]))
            continue
        if "integer" in typ or "number" in typ:
            lo = ps.get("minimum")
            cand = [x for x in nums if lo is None or float(x) >= float(lo)]
            out.append(Slot(k, ps, required, "number", cand + [OTHER], [int(x) for x in cand] + [OTHER], 10))
        elif "object" in typ:
            out.append(Slot(k, ps, required, "json", tokens=320 if k in CODE_FIELDS else 160))
        elif "array" in typ:
            pat = items.get("pattern")
            if pat or items.get("type") != "string":
                out.append(Slot(k, ps, required, "list", tokens=24))
            else:
                ids = [x for x in _recent(a) if int(items.get("maxLength", 200)) >= len(x) and
                       compatible(k, items, x)][:MAX_OPTIONS]
                out.append(Slot(k, ps, required, "choose", [f"{x} ({a.known[x]})" for x in ids] + [OTHER],
                                [[x] for x in ids] + [OTHER]) if ids else Slot(k, ps, required, "list", tokens=24))
        elif is_text_field(k, ps):
            n = 24 if (k in SHORT_TEXT or int(ps.get("maxLength", 4000) or 4000) <= 200) else 200
            if k in CODE_FIELDS:
                n = 320
            out.append(Slot(k, ps, required, "text", tokens=n))
        else:
            pat, mx = ps.get("pattern"), int(ps.get("maxLength", 200) or 200)
            ids = [x for x in _recent(a) if len(x) <= mx and (not pat or re.fullmatch(pat, x)) and
                   _ID_PATTERNS.fullmatch(x) and compatible(k, ps, x)][:MAX_OPTIONS]
            if publishes_content(tool, k):
                # the content of a publication: an artifact it already made, or text it writes now (stored as its
                # own artifact by the same call, see `execute`) - as a publishing form offers a text box. Only its own
                # artifacts: a platform publishes only what the publisher owns (the third standard run offered
                # artifacts seen in others' posts and got 17 refusals "you can only publish artifacts you own")
                own = set(a.seen.get("own_refs", []))
                ids = [x for x in ids if x in own]
                out.append(Slot(k, ps, required, "choose", [f"{x} ({a.known[x]})" for x in ids] + [COMPOSE],
                                ids + [COMPOSE]) if ids else Slot(k, ps, required, "compose", tokens=200))
            elif ids:
                out.append(Slot(k, ps, required, "choose", [f"{x} ({a.known[x]})" for x in ids] + [OTHER],
                                ids + [OTHER]))
            else:
                out.append(Slot(k, ps, required, "text", tokens=20))
        if len(out) >= max_slots:
            break
    return out


def slot_request(a: Agent, now: int, tool: dict, slot: Slot, sofar: dict, qv) -> dict:
    d = " ".join(str(slot.schema.get("description") or "").split())
    head = (f"{a.header(now, qv)}\n\nYou are calling {tool['name']}: {clip(tool.get('description', ''), 300)}\n"
            f"Arguments so far: {cjson(sofar, 400) if sofar else 'none'}\n"
            f"Next argument: {slot.name} ({type_str(slot.schema)}){': ' + d if d else ''}")
    if slot.kind in ("choose", "number") and len(slot.options) > 1:
        lines = "\n".join(f"- {o}" for o in slot.options)
        user = f"{head}\nOptions:\n{lines}\nAnswer with one option exactly as written."
        choices = [o.split(" (", 1)[0] if not o.startswith("(") else o for o in slot.options]
        return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "choices": choices, "normalize": "mean", "prefix_messages": 1}
    if slot.kind == "compose":
        ask = ("Write the text you publish now: the article, post or message itself, at least four sentences "
               "with concrete details about its subject, in plain words. Do not describe what you will do; write "
               "the text itself, nothing else.")
    elif slot.kind == "number":
        ask = "Write the number only (digits)."
    elif slot.kind == "json":
        ask = "Write the value as one JSON object on one line, nothing else."
    elif slot.kind == "list":
        ask = "Write the values separated by commas, nothing else."
    elif slot.tokens > 40:
        ask = f"Write the {slot.name} itself now (plain text, a few sentences), nothing else."
    else:
        ask = f"Write the {slot.name} only, on one line."
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{head}\n{ask}"}],
            "max_new_tokens": int(slot.tokens), "temperature": 0.0, "seed": 0, "prefix_messages": 1}


def parse_json_object(text: str) -> dict | None:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = t.find("{")
    while start >= 0:
        depth, ins, esc = 0, False, False
        for j in range(start, len(t)):
            c = t[j]
            if ins:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    ins = False
                continue
            if c == '"':
                ins = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:j + 1])
                        return v if isinstance(v, dict) else None
                    except ValueError:
                        break
        start = t.find("{", start + 1)
    return None


# lines of the prompt's own scaffolding (not the participant's words): a written value that repeats one of them is an
# echo of the instructions ("You are calling blog.comment: ...", seen from SmolLM2-135M in the second standard run)
SCAFFOLD = re.compile(r"\b(you are calling|arguments so far|next argument|answer with (one option|the \w+ name only|"
                      r"the action only)|write the \w+ (only|itself)|your goal \(you chose|your situation:|"
                      r"what you remember:|your last actions:|you are p\d+\. |options:\s*\n\s*- |"
                      r"services you can use:|tools of the service)", re.I)


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def _grams(t: str, n: int = 5) -> set:
    w = _words(t)
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def tool_names(a: Agent, tool: str = "") -> tuple[set, set]:
    """(names of the participant's tools and services, the called tool's own names): a written value that is one of
    them names a service instead of being a title, query or text (the audit's junk values: calendar entries titled
    'cal', mail bodies 'mail.send')."""
    every = {t["name"].lower() for t in a.tools if isinstance(t, dict) and t.get("name")}
    if tool:
        every.add(tool.lower())
    every |= {n.split(".", 1)[0] for n in every}
    called = {tool.lower(), tool.lower().split(".", 1)[0]} if tool else set()
    return every, called | {n for n in every if "." in n}


def restates(value: str, goal: str, n: int = 5, cover: float = 0.6) -> bool:
    """Whether a written value is the participant's goal sentence restated (most of the goal's word n-grams, and
    mostly nothing else) rather than a value of its own."""
    g, gg = _grams(value, n), _grams(goal, n)
    if not g or not gg:
        return False
    both = g & gg
    return len(both) / len(gg) > cover and len(both) / len(g) > cover


def copied_from_prompt(text: str, req: dict, own: str = "", n: int = 5, share: float = 0.3, short: bool = False,
                       goal: str = "", names: tuple[set, set] = (set(), set()), compose: bool = False) -> bool:
    """Whether a written value is not the value itself: the prompt copied back, the goal restated, or a tool name.

    * any value that repeats a line of the prompt's scaffolding (`SCAFFOLD`) is an echo;
    * a value that is the name of a tool or service (`tool_names`: for a short value the called tool, its service or
      any tool; for a longer text any tool or service) names a service instead of saying anything;
    * a value that restates the participant's goal sentence (`restates`) is not a title, query or text of its own;
    * a short value (`short`: a title, subject, query, name ...) of four words or more is a copy when more than
      half of its word 4-grams occur in the prompt outside the participant's own words (`own`: its goal and the
      values it wrote in its earlier calls) - another person's title seen in a result, the tool's description, a
      memory line, its persona, a situation line. A title or query about its goal naturally reuses the goal's words
      (the third standard run discarded 48 such values of one participant as copies, leaving every publication
      without its required title), and so does a rewrite of a value it wrote before;
    * a longer value is one when more than `share` of its word 5-grams occur in the prompt outside its own words,
      and a composed publication (`compose`: an article, post or message written for the call) also when more than
      half of them occur anywhere in the prompt, its goal included (an article that only restates the goal sentence,
      once, twice or with filler, is not an article; verifier probe P1b).
    """
    t = text if isinstance(text, str) else json.dumps(text, default=str)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    if SCAFFOLD.search(t):
        return True
    words = _words(t)
    if not words:
        return False
    flat = " ".join(t.lower().split()).strip(" .,:;!?'\"`")
    every, called = names
    if flat in (called if short else every):
        return True
    if goal and restates(t, goal):
        return True
    prompt = " ".join(m.get("content", "") for m in req.get("messages", []))
    if short:
        if len(words) < 4:
            return False
        g = _grams(t, 4)
        foreign = _grams(prompt, 4) - _grams(own or "", 4)
        return len(g & foreign) / len(g) > 0.5
    g = _grams(t, n)
    if not g:
        return False
    ref = _grams(prompt, n)
    if compose and len(g & ref) / len(g) > 0.5:
        return True
    return len(g & (ref - _grams(own or "", n))) / len(g) > share


def written_value(slot: Slot, text: str):
    """The written text as the slot's JSON type (None when nothing usable was written)."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```\w*\n?|```$", "", t).strip()
    t = re.sub(rf"^\s*{re.escape(slot.name)}\s*[:=]\s*", "", t, flags=re.I).strip()
    typ = slot.schema.get("type")
    typ = [typ] if isinstance(typ, str) else list(typ or [])
    if slot.kind == "number":
        m = _NUM.search(t.replace(",", ""))
        if not m:
            return None
        return int(m.group(0)) if "integer" in typ else float(m.group(0))
    if slot.kind == "json":
        return parse_json_object(t)
    if slot.kind == "compose":
        t = t.strip().strip('"').strip()
        return {COMPOSED: t[:6000]} if len(t.split()) >= 3 else None
    if slot.kind == "list":
        parts = [p.strip().strip('"').strip("'") for p in re.split(r"[,\n]", t) if p.strip()]
        return parts[:5] or None
    if slot.tokens <= 40:
        t = t.split("\n")[0].strip()
    t = t.strip().strip('"').strip("'").strip()
    mx = int(slot.schema.get("maxLength", 20000) or 20000)
    return t[:mx] if t else None


# ============================================================================================== the activation
_SUMMARIES: dict = {}


def host_summaries(world) -> dict:
    key = tuple(sorted(world.modules))
    s = _SUMMARIES.get(key)
    if s is None:
        from ...core import discovery
        s = _SUMMARIES[key] = discovery.host_summaries(world)
    return s


@dataclass
class Config:
    steps: int = 3
    hosts: int = 10
    shortlist: int = 10
    max_slots: int = 6
    patience: int = 4             # activations a goal may stay active
    max_failures: int = 4         # consecutive failed calls before the goal is given up
    memory: int = 40
    history: int = 80
    summaries: dict = field(default_factory=dict)     # host summaries when deciding without a world transaction


def _qtext(a: Agent) -> str:
    g = a.goal()
    last = a.add_memory[-1]["text"] if a.add_memory else (a.memory[-1]["text"] if a.memory else "")
    return clip(((g or {}).get("text", "") + " " + last), 600)


def activate(ctx, agents: list[Agent], brain, cfg: Config) -> None:
    now = ctx.now
    for a in agents:
        perceive(ctx, a)
        a.count("activations")
        g = a.goal()
        if g is not None:
            g["activations"] += 1
            a.goals_changed.add(g["id"])
            if g["activations"] > cfg.patience:
                a.remember(now, "goal", f"You gave up your goal '{g['text']}' (it took too long).")
                a.end_goal(now, "abandoned")
    # ---- goals
    need = [a for a in agents if a.goal() is None]
    if need:
        texts = brain.generate(ctx, "situation_interpretation", [goal_request(a, now) for a in need])
        for a, t in zip(need, texts):
            goal = clean_goal(t)
            if goal:
                a.new_goal(now, goal)
                a.remember(now, "goal", f"You decided: {goal}")
            else:
                a.stop = True
                a.count("goal_unclear")
    # ---- actions
    for step in range(cfg.steps):
        live = [a for a in agents if not a.stop and a.goal() is not None]
        if not live:
            break
        acting = decide(ctx, brain, live, now, cfg, step)
        if not acting:
            continue
        args = fill_arguments(ctx, brain, acting, now, cfg)
        for (a, tool, _), arg in zip(acting, args):
            execute(ctx, a, tool, arg, now, cfg)


def decide(ctx, brain, live: list[Agent], now: int, cfg: Config, step: int = 0) -> list[tuple]:
    """One decision of each live participant: its next step in words, the service, the tool (or finish / wait).
    Returns [(agent, tool, query vector)] of the participants that act."""
    qvecs = catalog.embed([_qtext(a) for a in live])
    # the next-step phrase: the use of situation interpretation the class-4 bench measures (routing.USES)
    intents = brain.generate(ctx, "next_step", [intent_request(a, now, qv) for a, qv in zip(live, qvecs)])
    for a, t in zip(live, intents):
        a.intent = clean_intent(t)
    rest = []
    for a in live:                          # the participant itself said its goal is done / it must wait
        word = a.intent.lower().strip(" .!")
        if (word == "stop" and not achieved_something(a)) or (word == "wait" and not awaiting(a)):
            a.intent = ""                   # not offered (see intent_request): the goal itself is the query
            a.count("premature_stop")
            rest.append(a)
        elif word in ("stop", "wait"):
            a.step_log.append({"step": step, "intent": a.intent, "service": "finish" if word == "stop" else "wait"})
            if word == "stop":
                g = a.goal()
                a.remember(now, "goal", f"You finished your goal '{g['text']}'.")
                a.end_goal(now, "done")
            else:
                a.count("waits")
            a.stop = True
        else:
            rest.append(a)
    live = rest
    if not live:
        return []
    goals = [(a.goal() or {}).get("text", "") for a in live]
    queries = [a.intent or g for a, g in zip(live, goals)]
    qvecs = catalog.embed(queries)
    summaries = host_summaries(ctx.world) if ctx is not None else cfg.summaries
    hsl = catalog.host_shortlists(ctx, queries, [a.tools for a in live], cfg.hosts, goals, summaries)
    hp = brain.choose(ctx, "tool_choice", [service_request(a, now, h, summaries, qv)
                                           for a, h, qv in zip(live, hsl, qvecs)])
    chosen = []
    for a, hs, p in zip(live, hsl, hp):
        names = list(hs) + endings(a)
        name = names[p] if 0 <= p < len(names) else "wait"
        a.step_log.append({"step": step, "intent": a.intent, "services": list(hs), "service": name})
        if name == "finish":
            g = a.goal()
            a.remember(now, "goal", f"You finished your goal '{g['text']}'.")
            a.end_goal(now, "done")
            a.stop = True
        elif name == "wait":
            a.count("waits")
            a.stop = True
        else:
            chosen.append((a, name))
    acting = []
    if chosen:
        q2 = [a.intent or (a.goal() or {}).get("text", "") for a, _ in chosen]
        in_host = [[t for t in a.tools if catalog.host_of(t["name"]) == h] for a, h in chosen]
        tsl = catalog.tool_shortlists(ctx, q2, in_host, cfg.shortlist)
        qv2 = catalog.embed(q2)
        reqs, keep = [], []
        for (a, h), ts, names, qv in zip(chosen, in_host, tsl, qv2):
            by = {t["name"]: t for t in ts}
            short = [by[n] for n in names if n in by]
            if short:
                reqs.append(tool_request(a, now, h, short, qv))
                keep.append((a, short, qv))
        picks = brain.choose(ctx, "tool_choice", reqs)
        for (a, short, qv), p in zip(keep, picks):
            t = short[p] if 0 <= p < len(short) else short[0]
            a.step_log[-1].update({"tools": [x["name"] for x in short], "pick": t["name"]})
            acting.append((a, t, qv))
    return acting


def fill_arguments(ctx, brain, acting: list, now: int, cfg: Config) -> list[dict]:
    plans = [plan_slots(a, t, cfg.max_slots) for a, t, _ in acting]
    vals: list[dict] = [{} for _ in acting]
    rounds = max((len(p) for p in plans), default=0)
    for r in range(rounds):
        todo = [(j, plans[j][r]) for j in range(len(acting)) if r < len(plans[j])]
        groups: dict[tuple[str, str], list] = {}
        for j, slot in todo:
            a, tool, qv = acting[j]
            req = slot_request(a, now, tool, slot, vals[j], qv)
            kind = "choose" if "choices" in req else "generate"
            groups.setdefault((kind, capability_of(tool["name"], slot.name if kind == "generate" else None)),
                              []).append((j, slot, req))
        written: list[tuple[int, Slot]] = []
        retry: list[tuple[int, Slot]] = []
        for (kind, cap), items in sorted(groups.items()):
            reqs = [x[2] for x in items]
            if kind == "choose":
                out = brain.choose(ctx, cap if cap != "tool_choice" else "situation_interpretation", reqs)
                for (j, slot, _), o in zip(items, out):
                    v = slot.values[o] if 0 <= o < len(slot.values) else None
                    if v in (OTHER, COMPOSE):
                        written.append((j, slot if v == OTHER else Slot(slot.name, slot.schema, slot.required,
                                                                        "compose", tokens=200)))
                    elif v is not None:
                        vals[j][slot.name] = v
            else:
                out = brain.generate(ctx, cap, reqs)
                for (j, slot, req), o in zip(items, out):
                    v = accept(acting[j][0], slot, req, o, acting[j][1]["name"])
                    if v is not None:
                        vals[j][slot.name] = v
                    elif slot.required:
                        retry.append((j, slot))
        if written:                     # "another value": write it (one more batched request)
            reqs, caps = [], []
            for j, slot in written:
                a, tool, qv = acting[j]
                s2 = slot if slot.kind == "compose" else \
                    Slot(slot.name, slot.schema, slot.required, "number" if slot.kind == "number" else "text",
                         tokens=10 if slot.kind == "number" else 24)
                reqs.append((j, s2, slot_request(a, now, tool, s2, vals[j], qv)))
            out = brain.generate(ctx, "communication", [x[2] for x in reqs])
            for (j, s2, req), o in zip(reqs, out):
                v = accept(acting[j][0], s2, req, o, acting[j][1]["name"])
                if v is not None:
                    vals[j][s2.name] = [v] if ("array" in str(s2.schema.get("type")) and s2.kind != "compose") else v
                elif s2.required:
                    retry.append((j, s2))
        if retry:
            # a required value the participant did not write usably (nothing, or an echo of the prompt): it writes it
            # once more, sampling (a person rewrites a field it got wrong); still nothing -> the call goes out without
            # it and the world's refusal ("args.title: required") is what the participant learns
            reqs = []
            for j, slot in retry:
                a, tool, qv = acting[j]
                req = slot_request(a, now, tool, slot, vals[j], qv)
                req = dict(req, temperature=0.7, seed=int(a.i * 104729 + now + len(a.history)))
                reqs.append((j, slot, req))
            groups2: dict[str, list] = {}
            for x in reqs:
                groups2.setdefault(capability_of(acting[x[0]][1]["name"], x[1].name), []).append(x)
            for cap, items in sorted(groups2.items()):
                out = brain.generate(ctx, cap, [x[2] for x in items])
                for (j, slot, req), o in zip(items, out):
                    v = accept(acting[j][0], slot, req, o, acting[j][1]["name"])
                    acting[j][0].count("rewritten_values")
                    if v is not None:
                        vals[j][slot.name] = [v] if ("array" in str(slot.schema.get("type")) and
                                                     slot.kind not in ("compose", "list")) else v
    return vals


def own_words(a: Agent) -> str:
    """The participant's own words in its prompts: its goal and the values it wrote in its recent calls."""
    return " ".join([(a.goal() or {}).get("text", "")] + [str(h.get("args", "")) for h in a.history[-8:]])


def accept(a: Agent, slot: Slot, req: dict, text: str, tool: str = ""):
    """The value a participant wrote for a slot, or None when it wrote nothing usable, echoed the prompt, restated
    its goal or wrote a tool's name (`copied_from_prompt`)."""
    v = written_value(slot, text)
    if v is None or slot.kind == "number":
        return v                          # a number is chosen or written, never an echo
    if copied_from_prompt(text, req, own=own_words(a), short=slot.kind in ("text", "list") and slot.tokens <= 40,
                          goal=(a.goal() or {}).get("text", ""), names=tool_names(a, tool),
                          compose=slot.kind == "compose"):
        a.count("copied_values")
        return None
    return v


READING_TOOLS = ("mail.read", "forum.read", "docs.get", "msg.read", "msg.history", "content.get", "blog.view",
                 "posts.view", "kb.question", "dev.question", "search.get", "code.read")
KIND_OF_HOST = {"mail": "talk", "msg": "talk", "forum": "talk", "social": "talk", "deal": "talk", "commit": "talk",
                "blog": "create", "posts": "create", "pics": "create", "tube": "create", "shorts": "create",
                "pods": "create", "content": "create", "studio": "create", "docs": "create", "code": "create",
                "dev": "create", "kb": "create", "launch": "create", "letters": "create", "pkg": "create",
                "bank": "money", "prod": "money", "subs": "money", "market": "money", "xchg": "money",
                "jobs": "work", "labor": "work", "cloud": "build"}


def compose_artifacts(ctx, a: Agent, args: dict, g: dict | None, made: dict) -> dict:
    """Arguments whose content the participant wrote in this call (`COMPOSED`) become artifacts of its own, stored
    as itself with the world's ordinary tools (`studio.write`, else `artifact.put`) inside the same decision, so
    their events belong to the call it chose."""
    out = dict(args)
    for k, v in args.items():
        if not (isinstance(v, dict) and set(v) == {COMPOSED}):
            continue
        body = str(v[COMPOSED])
        title = clip(str(args.get("title") or (g or {}).get("text") or "Notes"), 120)
        if "studio" in ctx.world.modules:
            stored = ctx.call("studio.write", {"format": "text", "title": title, "body": body},
                            as_principal=a.principal)
        else:
            import base64
            stored = ctx.call("artifact.put", {"data_b64": base64.b64encode(body.encode("utf-8")).decode("ascii"),
                                             "mime": "text/plain", "kind": "text"}, as_principal=a.principal)
        out[k] = stored["ref"]
        made["refs"].append(stored["ref"])
        made["n"] = len(ctx.events)
        a.learn_ids({stored["ref"]: f"the text you wrote ('{clip(title, 40)}')"})
        own_ref(a, stored["ref"])
    return out


def own_ref(a: Agent, ref: str) -> None:
    """An artifact the participant made itself (its own record: the text it wrote, the result of its studio call)."""
    own = [x for x in a.seen.get("own_refs", []) if x != ref] + [ref]
    a.seen["own_refs"] = own[-30:]


def made_refs(a: Agent, evs: list, res: Any, read_only: bool) -> list[str]:
    """Artifacts a successful call shows to be the participant's own: those its own non-read-only call stored in its
    name (the `artifact.stored` events of the call, visible to their owner), and a result that names its owner
    (`artifact.info` of one of its own artifacts). A read-only call that returns another participant's published
    artifact (`artifact.info`, `artifact.get`) makes nothing its own (verifier P4: such a ref was offered for a
    publication and refused "you can only publish artifacts you own")."""
    out = []
    if not read_only:
        out += [e.payload["ref"] for e in evs if e.type == "artifact.stored" and
                e.payload.get("owner") == a.principal and isinstance(e.payload.get("ref"), str)]
    if isinstance(res, dict) and isinstance(res.get("ref"), str) and res["ref"].startswith("sha256:") and \
            res.get("owner") == a.principal:
        out.append(res["ref"])
    return list(dict.fromkeys(out))


def forget(a: Agent, refs: list[str]) -> None:
    """Identifiers of artifacts a refused decision stored and that the rollback removed again (the third standard
    run kept offering them: 17 calls "no artifact sha256:...")."""
    for r in refs:
        a.known.pop(r, None)
        a.add_known.pop(r, None)
    if refs and a.seen.get("own_refs"):
        a.seen["own_refs"] = [x for x in a.seen["own_refs"] if x not in refs]


def execute(ctx, a: Agent, tool: dict, args: dict, now: int, cfg: Config) -> None:
    name = tool["name"]
    ro = bool((tool.get("annotations") or {}).get("readOnlyHint"))
    g = a.goal()
    dec = hashlib.sha256(json.dumps([a.i, now, name, args], sort_keys=True, default=str).encode()).hexdigest()[:16]
    rec = {"t": int(now), "tool": name, "args": cjson(args, 240), "ok": False, "read_only": ro,
           "goal": g["id"] if g else None, "kind": KIND_OF_HOST.get(name.split(".", 1)[0], "other"), "decision": dec}
    a.count("calls")
    n0, seq0, mark = len(ctx.events), ctx.world.seq, ctx.state.savepoint()
    made = {"refs": [], "n": n0}       # artifacts stored for this call, and the event count after the last one
    try:
        args = compose_artifacts(ctx, a, args, g, made)
        res = ctx.call(name, args, as_principal=a.principal)
        rec["ok"] = True
        evs = ctx.events[n0:]
        rec["changed"] = any(not e.type.startswith(("know", "ctr")) for e in evs)
        if evs:
            rec["seq"] = [int(evs[0].seq), int(evs[-1].seq)]
        view = summarise(res)
        rec["result"] = cjson(view, 200)
        found: dict[str, str] = {}
        harvest_ids(res, f"from your {name}", found)
        a.learn_ids(found)
        for r in made_refs(a, evs, res, ro):
            own_ref(a, r)               # an artifact it just made with its own tools, or one a result says it owns
        text = f"You called {name} {cjson(args, 200)} -> ok: {cjson(view, 260)}"
        if name in READING_TOOLS and isinstance(res, dict):
            inner = res.get("message") if isinstance(res.get("message"), dict) else res
            body = inner.get("body") or inner.get("text")
            if body:
                text = f"You read via {name}: {cjson(summarise(res, n_chars=400), 420)}"
                for x in ids_in_text(str(body)):
                    a.learn_ids({x: f"mentioned in what you read ({name})"})
        a.remember(now, "act", text)
        a.count("ok")
        if not ro and rec["changed"]:
            a.count("changes")
        a.failures = 0
        if g:
            g["ok"] += 1
    except CrucibleError as e:
        if made["refs"]:
            # the whole decision is undone, including the text it stored for this call (compose_artifacts): Ctx.call
            # undoes only the refused call itself; the facts it keeps about the refusal are kept
            keep = [(ev.type, ev.payload, ev.vis) for ev in ctx.events[made["n"]:]]
            ctx.state.rollback_to(mark)
            del ctx.events[n0:]
            ctx.world.seq = seq0
            for t_, p_, v_ in keep:
                ctx.emit(t_, p_, v_)
        forget(a, made["refs"])
        rec["code"] = type(e).__name__
        rec["error"] = clip(str(e), 200)
        a.remember(now, "act", f"You called {name} {cjson(args, 200)} -> FAILED ({rec['code']}): {rec['error']}")
        a.count("rejected")
        a.failures += 1
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, ZeroDivisionError) as e:
        # a fault inside another module's tool handler (not a refusal): undo the call's partial effects exactly as
        # Ctx.call undoes a refused call, record it in the participant's history (and so in the evidence), and let
        # the population's tick go on instead of failing every individual's session of this tick
        ctx.state.rollback_to(mark)
        del ctx.events[n0:]
        ctx.world.seq = seq0
        forget(a, made["refs"])
        rec["code"] = "ToolFault"
        rec["error"] = clip(f"{type(e).__name__}: {e}", 200)
        a.remember(now, "act", f"You called {name} {cjson(args, 200)} -> FAILED (the service broke): {rec['error']}")
        a.count("tool_faults")
        a.failures += 1
    if g:
        g["steps"] += 1
        a.goals_changed.add(g["id"])
    a.history.append(rec)
    a.add_history.append(rec)
    if a.failures >= cfg.max_failures and a.goal() is not None:
        a.remember(now, "goal", f"You gave up your goal '{a.goal()['text']}' after repeated failures.")
        a.end_goal(now, "abandoned")
        a.stop = True


def delta(a: Agent, next_at: int) -> dict:
    """The `semp.update` payload of one activation."""
    return {"i": a.i, "memory_add": a.add_memory, "history_add": a.add_history, "known_add": a.add_known,
            "goals_set": [g for g in a.goals if g["id"] in a.goals_changed], "active": a.active, "seen": a.seen,
            "counts": a.counts, "next_at": int(next_at), "steps": a.step_log}
