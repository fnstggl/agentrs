"""Shared plumbing for learning agents that act in Crucible worlds (LLM agents, CAM 2.0, the AGENTRS superlearner,
small neural RL learners).

Nothing here encodes a strategy. It provides

* `agent_tools(world, actor)`      - the tool specs an in-world actor may call (operator-only ``admin.*`` removed).
* `make_goal_world(...)`           - goal family x world family x seed x level -> (GoalContract, World, actor), with
                                     an optional documented *scripted start-state prefix* (`PREFIXES`).
* `EntityMemory`                   - what the actor has seen: identifier-like strings and numbers, indexed by the JSON
                                     key they appeared under (``{"listing": "lst_000003"}`` -> key ``listing``).
* `ActionTemplates`                - a factored, structured action space over *primitive* tool calls: tool, then one
                                     slot per argument filled from `EntityMemory` (same-key values first, then recent
                                     identifiers), the schema's enum options, booleans, a fixed log-spaced grid of
                                     integers, and "omit" for optional arguments. Arguments that need free-form
                                     content (objects such as ``code.commit.files``, arrays of objects) cannot be
                                     filled from observations: tools that *require* them are not offered and optional
                                     ones are always omitted. Writing such arguments needs a generative model
                                     (see `crucible.agents.llm` and the CAM 2.0 adapter).
* `obs_features(...)`              - parameter-free hashed numeric features of an observation (for neural policies).

Observations used here are the actor's own view only: `World.observe(actor)` summaries, events visible to the actor,
and the results of its own actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

DAY = 86400
SKIP_OBS_KEYS = ("time", "principal", "events", "events_dropped")

# ---------------------------------------------------------------------------------------------- tools
def agent_tools(world, actor: str) -> list[dict]:
    """MCP-shaped tool specs available to an in-world actor (``admin.*`` is reserved for the operator)."""
    return [t for t in world.available_tools(actor) if not t["name"].startswith("admin.")]


def tool_signature(tools: list[dict]) -> str:
    return hashlib.sha1("\n".join(sorted(t["name"] for t in tools)).encode()).hexdigest()[:12]


class ActorView:
    """The actor's own view of a world, as `agents.base.Runner` builds it: ``World.observe(actor)`` summaries plus the
    events visible to the actor since its previous observation (its own ``cmd`` records omitted; the outcome of its
    last action is returned to it directly)."""

    def __init__(self, world, actor: str, max_events: int = 40):
        self.world, self.actor, self.max_events = world, actor, max_events
        self.cursor = world.seq
        self._tools: tuple | None = None

    def rebind(self, world) -> None:
        self.world, self.cursor, self._tools = world, world.seq, None

    def observe(self) -> dict:
        w = self.world
        obs = w.observe(self.actor)
        evs, dropped = w.visible_events(self.actor, self.cursor, limit=self.max_events)
        self.cursor = w.seq
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload}
                         for e in evs if not (e.type in ("cmd", "cmd.rejected") and e.actor == self.actor)]
        obs["events_dropped"] = dropped
        return obs

    def tools(self) -> list[dict]:
        n = len(self.world.state.find_ids("knowledge", "principal", self.actor))
        if self._tools is None or self._tools[0] != n:
            self._tools = (n, agent_tools(self.world, self.actor))
        return self._tools[1]


# ---------------------------------------------------------------------------------------------- start states
def deployed_service_prefix(world, actor: str, need: str = "route.shortest", price_mult: float = 4.0,
                            capital: int = 400_000, dt: int = 3600, max_rounds: int = 60) -> dict:
    """Scripted start-state prefix (documented in docs/RL_INTEGRATION.md): the actor founds a company, writes the best
    known implementation of `need`, rents one core, deploys it and lists it on the market at
    ``price_mult x`` the cheapest rival's price (no ads, no forum post, no operations loop). It runs through ordinary
    recorded commands, one per simulated `dt`, like any agent. Returns the created entity ids and the prefix's own
    action log (the actor's history, which a learner may remember)."""
    from ..agents.base import Runner
    from ..agents.scripted import ScriptPolicy, call, cheapest_rival, ok, service_files
    from ..economy.needs import NEEDS, best_tier, tier_source

    info: dict = {"need": need, "price_mult": price_mult}

    class _Prefix(ScriptPolicy):
        name = "prefix:deployed_service"

        def script(self):
            r = yield call("market.browse", need=need, sort="price", limit=10)
            price = max(1, int(round(cheapest_rival(r["result"] if ok(r) else [], need, 1.0) * price_mult)))
            org, acct, personal = yield from self.found_company("labs", capital)
            info.update(org=org, account=acct, personal_account=personal, price=price)
            if acct is None:
                return
            r = yield call("code.create_repo", name=need.replace(".", "-"), owner=org, description=NEEDS[need].title)
            repo = r["result"]["repo"]
            info["repo"] = repo
            yield call("code.commit", repo=repo, files=service_files(need, tier_source(need, best_tier(need)), price),
                       message="first version")
            svc = yield from self.launch_service(org, acct, repo, f"{self.handle[:12]}-{need.split('.')[1][:6]}",
                                                 need, price)
            if svc:
                info.update(host=svc["host"], instance=svc["instance"], listing=svc["listing"])

    pol = _Prefix()
    run = Runner(world, [(actor, pol)], dt=dt, record=True)
    for _ in range(max_rounds):
        run.round()
        if pol.finished:
            break
    info["history"] = [(s.action, s.result) for s in run.trajs[actor].steps if s.action]
    info["ok"] = bool(info.get("listing"))
    info["errors"] = list(pol.log)
    return info


PREFIXES = {"none": None, "deployed_service": deployed_service_prefix}


def make_goal_world(goal_family: str, world_family: str = "tiny", seed: int = 0, level: int = 0,
                    split: str | None = None, prefix: str | None = None, prefix_kw: dict | None = None,
                    handle: str = "agent", overrides: dict | None = None, goal=None):
    """Instantiate a goal (``generate_goal`` + ``instantiate``) and optionally run a scripted start-state prefix.
    Returns ``(goal, world, actor, prefix_info)``."""
    import crucible
    crucible.load_all_modules()
    from ..goals.contracts import generate_goal, instantiate
    if goal is None:
        goal = generate_goal(goal_family, world_family, int(seed), int(level), split=split or "dev", overrides=overrides)
    world, actor = instantiate(goal, handle)
    pinfo: dict = {}
    if prefix and prefix != "none":
        pinfo = PREFIXES[prefix](world, actor, **(prefix_kw or {}))
    return goal, world, actor, pinfo


# ---------------------------------------------------------------------------------------------- entity memory
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_./@-]{0,80}$")
_TEXT_KEYS = {"text", "body", "description", "memo", "note", "message", "title", "bio", "doc", "q", "comment",
              "reason", "name", "content", "stdout", "stderr", "error", "balance_fmt", "news"}


def id_kind(v: str) -> str:
    """Coarse lexical type of an identifier (``acct_000017`` -> ``acct``, ``o:agent-labs`` -> ``o``)."""
    if ":" in v and v.split(":", 1)[0] in ("p", "o", "svc", "board"):
        return v.split(":", 1)[0]
    m = re.match(r"^([a-z]+)_\d+$", v)
    if m:
        return m.group(1)
    if "/" in v:
        return "path"
    if "." in v:
        return "dotted"
    return "word"


_GENERIC_LAST = {"cents", "s", "id", "at", "n", "ms", "t", "to", "limit", "days", "only", "in", "of"}


def _last(k: str) -> str:
    """Last ``_`` token of a key (``from_account`` -> ``account``); generic unit tokens do not alias."""
    t = k.rsplit("_", 1)[-1] if k else k
    return k if t in _GENERIC_LAST else t


class EntityMemory:
    """Recency-ordered memory of identifier-like strings and integers the actor has seen, indexed by JSON key.

    ``observe_obj(x)`` walks any JSON value. String leaves that look like identifiers (no spaces, <= 81 chars, not under
    a free-text key) are stored under their key; dictionary keys that look like ids are stored under the parent key
    (``{"balances": {"acct_000018": 398712}}`` -> key ``balances``); integer leaves are stored as numbers per key."""

    def __init__(self, per_key: int = 8, recent: int = 32, numbers_per_key: int = 4):
        self.per_key, self.n_recent, self.numbers_per_key = per_key, recent, numbers_per_key
        self.by_key: dict[str, OrderedDict] = {}
        self.recent: OrderedDict = OrderedDict()
        self.numbers: dict[str, OrderedDict] = {}
        self.tick = 0
        self.version = 0          # bumped on every write; caches below are keyed by it
        self._cache: dict = {}

    def _put_id(self, key: str, v: str) -> None:
        self.version += 1
        d = self.by_key.setdefault(key, OrderedDict())
        d.pop(v, None)
        d[v] = self.tick
        while len(d) > self.per_key:
            d.popitem(last=False)
        self.recent.pop(v, None)
        self.recent[v] = (self.tick, key)
        while len(self.recent) > self.n_recent:
            self.recent.popitem(last=False)

    def _put_num(self, key: str, v: int) -> None:
        self.version += 1
        d = self.numbers.setdefault(key, OrderedDict())
        d.pop(v, None)
        d[v] = self.tick
        while len(d) > self.numbers_per_key:
            d.popitem(last=False)

    def observe_obj(self, x: Any, key: str = "", depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(x, dict):
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0])):
                ks = str(k)
                if _ID_RE.match(ks) and id_kind(ks) != "word" and key:
                    self._put_id(key, ks)
                self.observe_obj(v, ks if not (_ID_RE.match(ks) and id_kind(ks) != "word") else key, depth + 1)
        elif isinstance(x, list):
            for v in x[:40]:
                self.observe_obj(v, key, depth + 1)
        elif isinstance(x, bool) or x is None:
            return
        elif isinstance(x, int):
            if key and abs(x) < 10**9:
                self._put_num(key, int(x))
        elif isinstance(x, str):
            if key in _TEXT_KEYS or not _ID_RE.match(x) or x.isdigit():
                return
            self._put_id(key or "_", x)

    def observe(self, obs: dict | None = None, action: dict | None = None, result: dict | None = None) -> None:
        """Remember an observation (summaries + visible events), the actor's own action and its result."""
        self.tick += 1
        if obs:
            for k in sorted(obs):
                if k in ("time",):
                    continue
                if k == "events":
                    for e in obs["events"][-40:]:
                        self.observe_obj(e.get("payload", {}), e.get("type", ""))
                        if e.get("actor"):
                            self._put_id("actor", e["actor"])
                elif k == "principal":
                    self._put_id("principal", obs[k])
                elif isinstance(obs[k], (dict, list)):
                    self.observe_obj(obs[k], k)
        if action:
            self.observe_obj(action.get("args", {}), "", 0)
        if result and result.get("ok"):
            self.observe_obj(result.get("result"), (action or {}).get("tool", "").split(".")[-1])

    def ids_for(self, param: str, limit: int = 6) -> list[tuple[str, str]]:
        """Candidate identifiers for an argument named `param`: values seen under the same key (or a key with the same
        last ``_`` token, e.g. ``from_account`` ~ ``account``), most recent first. Returns ``[(value, source)]``."""
        ck = ("ids", param, limit, self.version)
        if ck in self._cache:
            return self._cache[ck]
        if len(self._cache) > 4096:
            self._cache = {}
        best: dict[str, int] = {}
        for k, d in self.by_key.items():
            if k == param or _last(k) == _last(param):
                for v, tick in d.items():
                    best[v] = max(best.get(v, -1), tick)
        ranked = sorted(best, key=lambda v: (-best[v], v))
        out = self._cache[ck] = [(v, "key") for v in ranked[:limit]]
        return out

    def recent_ids(self, limit: int = 6, exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
        ex = set(exclude)
        out = []
        for v in reversed(self.recent):
            if v in ex or id_kind(v) == "word":
                continue
            out.append((v, "recent"))
            if len(out) >= limit:
                break
        return out

    def numbers_for(self, param: str, limit: int = 3) -> list[int]:
        ck = ("num", param, limit, self.version)
        if ck in self._cache:
            return self._cache[ck]
        out: list[int] = []
        for k, d in self.numbers.items():
            if k == param or _last(k) == _last(param):
                for v in reversed(d):
                    if v not in out:
                        out.append(v)
        self._cache[ck] = out[:limit]
        return out[:limit]

    def state_dict(self) -> dict:
        return {"by_key": {k: list(d.items()) for k, d in self.by_key.items()}, "recent": list(self.recent.items()),
                "numbers": {k: list(d.items()) for k, d in self.numbers.items()}, "tick": self.tick}

    def load_state_dict(self, s: dict) -> None:
        self.by_key = {k: OrderedDict((a, b) for a, b in v) for k, v in s["by_key"].items()}
        self.recent = OrderedDict((a, tuple(b)) for a, b in s["recent"])
        self.numbers = {k: OrderedDict((int(a), b) for a, b in v) for k, v in s["numbers"].items()}
        self.tick = s["tick"]
        self.version += 1
        self._cache = {}


# ---------------------------------------------------------------------------------------------- factored actions
INT_GRID = (0, 1, 2, 3, 5, 8, 13, 21, 50, 100, 1_000, 10_000, 100_000)
NUM_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 10.0)
OMIT = "<omit>"
WAIT = "<wait>"


@dataclass
class SlotOption:
    label: str            # human-readable label, e.g. "acct_000018" or "13" or "<omit>"
    value: Any            # the JSON value (OMIT sentinel string for omitted optional arguments)
    source: str           # key | recent | enum | bool | grid | seen | self | gen | omit
    rank: int = 0         # recency rank within its source


@dataclass
class Slot:
    param: str
    kind: str             # id | enum | bool | int | num | text | handle | list
    required: bool
    options: list = field(default_factory=list)


def _ptype(p: dict) -> str:
    t = p.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"][0] if [x for x in t if x != "null"] else "null"
    return t or "any"


class ActionTemplates:
    """Structured candidates for primitive tool calls: tool -> argument slots -> options (see module docstring).

    ``slots(tool)`` returns the slots of one tool (None if a required argument cannot be filled); ``build(tool,
    choices)`` turns one option index per slot into ``{"tool", "args"}``. ``max_slots`` bounds the arguments per
    tool (required first, then optional ones in schema order; arguments beyond the bound are omitted)."""

    def __init__(self, memory: EntityMemory, actor: str, max_slots: int = 6, max_ids: int = 6, max_recent: int = 4,
                 handle_prefix: str | None = None):
        self.mem, self.actor, self.max_slots = memory, actor, max_slots
        self.max_ids, self.max_recent = max_ids, max_recent
        self.handle_prefix = handle_prefix or actor.split(":", 1)[-1][:12]
        self._slot_cache: dict = {}

    def _id_options(self, param: str, p: dict) -> list:
        opts = [SlotOption(v, v, src, i) for i, (v, src) in enumerate(self.mem.ids_for(param, self.max_ids))]
        seen = {o.value for o in opts}
        if len(opts) < 2:   # nothing (or one thing) seen under this key: fall back to recent identifiers of any key
            opts += [SlotOption(v, v, src, i) for i, (v, src) in
                     enumerate(self.mem.recent_ids(self.max_recent, exclude=seen))]
        if self.actor not in {o.value for o in opts}:
            opts.append(SlotOption(self.actor, self.actor, "self", 0))
        mx = p.get("maxLength", 4000)
        pat = p.get("pattern")
        out = []
        for o in opts:
            if len(o.value) > mx or (pat and not re.fullmatch(pat, o.value)):
                continue
            out.append(o)
        return out

    def slot_for(self, param: str, p: dict, required: bool) -> Slot | None:
        t = _ptype(p)
        omit = [] if required else [SlotOption(OMIT, OMIT, "omit")]
        if "enum" in p:
            return Slot(param, "enum", required, omit + [SlotOption(str(v), v, "enum", i) for i, v in enumerate(p["enum"])])
        if t == "boolean":
            return Slot(param, "bool", required, omit + [SlotOption("true", True, "bool"), SlotOption("false", False, "bool", 1)])
        if t in ("integer", "number"):
            lo, hi = p.get("minimum", -math.inf), p.get("maximum", math.inf)
            grid = INT_GRID if t == "integer" else NUM_GRID
            vals = [SlotOption(str(v), v, "grid", i) for i, v in enumerate(grid) if lo <= v <= hi]
            for i, v in enumerate(self.mem.numbers_for(param)):
                if lo <= v <= hi and v not in grid:
                    vals.append(SlotOption(str(v), v if t == "integer" else float(v), "seen", i))
            return Slot(param, "int" if t == "integer" else "num", required, omit + vals) if vals else None
        if t == "string":
            pat = p.get("pattern")
            if pat:
                # constrained names (handles, repo / host names): seen values of the same key, plus a fresh handle
                gen = f"{self.handle_prefix}{len(self.mem.by_key.get(param, {})) + 1}"
                ids = [o for o in self._id_options(param, p) if o.source in ("key", "self")]
                opts = ids + ([SlotOption(gen, gen, "gen")] if re.fullmatch(pat, gen) else [])
                return Slot(param, "handle", required, omit + opts) if opts else None
            if p.get("maxLength", 4000) > 200 or param in _TEXT_KEYS:
                # free text: a short neutral token (the policy cannot write prose; see module docstring)
                txt = f"{param} {self.handle_prefix}"[: p.get("maxLength", 4000)]
                ids = [o for o in self._id_options(param, p) if o.source == "key"][:2]
                return Slot(param, "text", required, omit + ids + [SlotOption(txt, txt, "gen")])
            opts = self._id_options(param, p)
            return Slot(param, "id", required, omit + opts) if opts else None
        if t == "array":
            it = p.get("items") or {}
            if _ptype(it) == "string":
                base = self.slot_for(param, it, True)
                if base is None:
                    return None
                return Slot(param, "list", required,
                            omit + [SlotOption(f"[{o.label}]", [o.value], o.source, o.rank) for o in base.options])
            return None
        return None   # object / any: needs generated content

    def slots(self, tool: dict) -> list | None:
        key = (tool["name"], self.mem.version, id(self.mem))
        c = self._slot_cache.get(key, False)
        if c is not False:
            return c
        if len(self._slot_cache) > 2048:
            self._slot_cache = {}
        out = self._slots(tool)
        self._slot_cache[key] = out
        return out

    def _slots(self, tool: dict) -> list | None:
        sc = tool.get("inputSchema") or {}
        props = sc.get("properties") or {}
        req = list(sc.get("required") or [])
        order = req + [k for k in props if k not in req]
        out = []
        for name in order:
            is_req = name in req
            if len(out) >= self.max_slots:
                if is_req:
                    return None
                break
            s = self.slot_for(name, props[name], is_req)
            if s is None:
                if is_req:
                    return None
                continue
            out.append(s)
        return out

    @staticmethod
    def build(tool: dict, slots: list, choices: list) -> dict:
        args = {}
        for s, c in zip(slots, choices):
            o = s.options[c]
            if o.source == "omit":
                continue
            args[s.param] = o.value
        return {"tool": tool["name"], "args": args}


# ---------------------------------------------------------------------------------------------- features
NUM_RE = re.compile(r"(?<![A-Za-z])\d+")


def _norm_key(k: str) -> str:
    """Replace identifiers inside a feature path by their lexical kind (so features generalise across entities)."""
    return k if not (_ID_RE.match(k) and id_kind(k) != "word") else f"<{id_kind(k)}>"


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _h(s: str, dim: int) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16) % dim


def obs_features(obs: dict, last_result: dict | None, dim: int = 512, max_events: int = 60) -> np.ndarray:
    """Hashed features of an actor observation: numeric leaves of module summaries (paths with ids replaced by kinds),
    visible event counts by type / kind and their numeric payload sums, the last action's outcome. Deterministic."""
    v = np.zeros(dim, np.float32)
    acc: dict[str, float] = {}

    def walk(x, path, depth=0):
        if depth > 5:
            return
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(x[k], f"{path}.{_norm_key(str(k))}", depth + 1)
        elif isinstance(x, list):
            acc[f"{path}#n"] = acc.get(f"{path}#n", 0.0) + len(x)
            for y in x[:20]:
                walk(y, f"{path}[]", depth + 1)
        elif isinstance(x, bool):
            acc[path] = acc.get(path, 0.0) + float(x)
        elif isinstance(x, (int, float)):
            acc[path] = acc.get(path, 0.0) + float(x)
        elif isinstance(x, str) and len(x) < 40:
            acc[f"{path}={_norm_key(x)}"] = acc.get(f"{path}={_norm_key(x)}", 0.0) + 1.0

    for k in sorted(obs):
        if k in SKIP_OBS_KEYS:
            continue
        walk(obs[k], "s." + k)
    evs = obs.get("events") or []
    for e in evs[-max_events:]:
        et = e.get("type", "?")
        acc[f"ev:{et}"] = acc.get(f"ev:{et}", 0.0) + 1.0
        pl = e.get("payload") or {}
        if isinstance(pl, dict) and isinstance(pl.get("kind"), str):
            acc[f"ev:{et}:{pl['kind']}"] = acc.get(f"ev:{et}:{pl['kind']}", 0.0) + 1.0
        walk(pl, f"e.{et}")
    acc["ev:#"] = float(len(evs))
    acc["dropped"] = float(obs.get("events_dropped", 0) or 0)
    if last_result is not None:
        tool = last_result.get("tool", "?")
        if last_result.get("ok"):
            acc[f"r:{tool}:ok"] = 1.0
            acc["r:ok"] = 1.0
        else:
            code = (last_result.get("error") or {}).get("code", "error")
            acc[f"r:{tool}:{code}"] = 1.0
            acc[f"r:err:{code}"] = 1.0
    t = int(obs.get("time", 0))
    acc["t:sin_day"] = math.sin(2 * math.pi * (t % DAY) / DAY)
    acc["t:cos_day"] = math.cos(2 * math.pi * (t % DAY) / DAY)
    for k, x in acc.items():
        v[_h(k, dim)] += symlog(x)
    return v


def compact_json(x: Any, n: int = 200) -> str:
    s = json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)
    return s if len(s) <= n else s[: n - 3] + "..."
