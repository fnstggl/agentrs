"""The authoritative world kernel.

A world is an event-sourced state machine:

    command (actor, tool, args)  ──handler──▶  events  ──appliers──▶  state
    scheduled process (time, name, args)  ──handler──▶  events  ──appliers──▶  state

* Handlers validate, read state, draw seeded randomness and **emit events**; they never write state directly.
* Appliers are deterministic functions of (state, event payload). Replaying the event log on an empty world
  therefore reproduces the state exactly; the hash chain over events makes tampering detectable.
* Each command or process run is one transaction: either all of its events are applied, or none.
* The clock only moves through `advance_time`; scheduled processes run in (time, priority, sequence) order.
* Outcomes the kernel cannot recompute (sandboxed program output, measured CPU time, LLM text that arrived as
  arguments) are recorded in events, so replay never regenerates them.
"""
from __future__ import annotations

import heapq
import time as _time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from . import schema as S
from .canonical import canonical_bytes, chain_hash, digest, to_jsonable
from .errors import (CrucibleError, Forbidden, InvalidAction, InvariantViolation, ProtocolError, RateLimited,
                     Unauthenticated, UnknownTool)
from .state import StateStore

KERNEL_VERSION = "crucible-kernel/1"
SYSTEM = "sys"
GENESIS_HASH = "0" * 32


# ====================================================================== data types
@dataclass(slots=True)
class Event:
    seq: int
    t: int
    type: str
    actor: str
    cmd: int | None
    payload: dict
    vis: tuple | str  # tuple of principal ids, "public", or "internal" (evaluator / audit only)
    hash: str = ""

    def body(self) -> dict:
        return {"seq": self.seq, "t": self.t, "type": self.type, "actor": self.actor, "cmd": self.cmd,
                "payload": self.payload, "vis": list(self.vis) if isinstance(self.vis, tuple) else self.vis}

    def to_dict(self) -> dict:
        d = self.body()
        d["hash"] = self.hash
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        vis = d["vis"]
        return cls(d["seq"], d["t"], d["type"], d["actor"], d["cmd"], d["payload"],
                   tuple(vis) if isinstance(vis, list) else vis, d.get("hash", ""))


@dataclass
class ActionResult:
    ok: bool
    result: Any = None
    error: dict | None = None
    events: list[int] = field(default_factory=list)
    cost_s: int = 0
    t: int = 0
    replayed: bool = False
    tool: str = ""

    def to_dict(self) -> dict:
        d = {"ok": self.ok, "t": self.t, "tool": self.tool, "cost_s": self.cost_s}
        if self.ok:
            d["result"] = self.result
        else:
            d["error"] = self.error
        if self.replayed:
            d["replayed"] = True
        return d


@dataclass
class WorldSpec:
    """Everything needed to rebuild a world from its event log: module list with configs, seed, metadata."""
    family: str
    seed: int
    modules: list[dict]  # [{"name": str, "config": dict}]
    params: dict = field(default_factory=dict)
    kernel: str = KERNEL_VERSION

    def to_dict(self) -> dict:
        return {"family": self.family, "seed": self.seed, "modules": self.modules, "params": self.params,
                "kernel": self.kernel}

    @classmethod
    def from_dict(cls, d: dict) -> "WorldSpec":
        return cls(d["family"], int(d["seed"]), list(d["modules"]), dict(d.get("params", {})),
                   d.get("kernel", KERNEL_VERSION))

    def hash(self) -> str:
        return digest(self.to_dict())


# ====================================================================== module system
@dataclass
class Endpoint:
    name: str          # full tool name "<host>.<endpoint>"
    host: str
    params: dict
    handler: Callable[..., Any]
    doc: str = ""
    cost_s: int = 1
    read_only: bool = False
    returns: str = ""
    owner: str = SYSTEM   # principal operating the service (for deployed services: its service principal)
    builtin: bool = True
    version: str = "1"    # interface version of this endpoint (defaults to its module's version)

    def spec(self) -> dict:
        return {"name": self.name, "description": self.doc, "inputSchema": self.params,
                "annotations": {"readOnlyHint": self.read_only, "costSeconds": self.cost_s,
                                "host": self.host, "builtin": self.builtin, "version": self.version}}


def endpoint(name: str, params: dict | None = None, *, doc: str = "", cost_s: int = 1, read_only: bool = False,
             returns: str = "", version: str | None = None):
    def deco(fn):
        fn.__crucible_endpoint__ = dict(name=name, params=params or S.obj(), doc=doc or (fn.__doc__ or "").strip(),
                                        cost_s=cost_s, read_only=read_only, returns=returns, version=version)
        return fn
    return deco


def applier(event_type: str):
    def deco(fn):
        fn.__crucible_applier__ = event_type
        return fn
    return deco


def process(name: str):
    def deco(fn):
        fn.__crucible_process__ = name
        return fn
    return deco


class Module:
    """A self-contained piece of world mechanics (a service, an economic mechanism, a lab).

    Subclasses declare tables in `install`, endpoints/appliers/processes with decorators, initial content in
    `genesis`, optional per-actor observation summaries in `observe`, and invariants in `check_invariants`.
    """
    name: str = ""
    host: str | None = None  # tool prefix; defaults to name
    version: str = "1"
    depends: tuple[str, ...] = ()

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.world: World | None = None

    def install(self, world: "World") -> None:
        pass

    def genesis(self, ctx: "Ctx") -> None:
        pass

    def on_load(self, world: "World") -> None:
        """Rebuild runtime caches after load/replay/fork (never touch state here)."""

    def observe(self, world: "World", actor: str) -> dict | None:
        return None

    def check_invariants(self, world: "World") -> list[str]:
        return []

    def close(self) -> None:
        pass


MODULES: dict[str, type[Module]] = {}


def register_module(cls: type[Module]) -> type[Module]:
    if not cls.name:
        raise ProtocolError("module without name")
    MODULES[cls.name] = cls
    return cls


# ====================================================================== transaction context
class Ctx:
    """Handed to endpoint handlers and processes. All reads go through `state`; all writes through `emit`."""

    def __init__(self, world: "World", actor: str, cmd: int | None, origin: str | None = None, depth: int = 0):
        self.world = world
        self.state = world.state
        self.actor = actor
        self.origin = origin or actor
        self.cmd = cmd
        self.depth = depth
        # key under which this transaction's external outcomes are recorded: the seq of its first event
        self.ext_key = world._tx_seq0 + 1
        self.events: list[Event] = []
        self.ext: list[Any] = []
        self._ext_i = 0
        self._rng_n = 0
        self.tool: str | None = None          # the tool this context is running (for institutional rules)
        self.tool_args: dict | None = None
        self.read_only = False
        self.gov_approved: str | None = None  # set while executing an approved governance proposal

    @property
    def now(self) -> int:
        return self.world.now

    def emit(self, type: str, payload: dict | None = None, vis: Iterable[str] | str | None = None) -> Event:
        return self.world._emit(self, type, payload or {}, vis)

    def rng(self, *keys: Any) -> np.random.Generator:
        self._rng_n += 1
        return self.world.rng_for("cmd", self.world.seq, self._rng_n, *keys)

    def fail(self, exc: type[CrucibleError], msg: str, **details) -> None:
        raise exc(msg, **details)

    def call(self, tool: str, args: dict | None = None, as_principal: str | None = None, brokered: bool = False) -> Any:
        """Nested call inside the current transaction (services calling services, system calling modules).

        If the callee fails, its partial effects are rolled back to a savepoint and the error propagates.
        `brokered=True` is used only by the sandbox broker: a running program always acts as its own service
        principal (fixed by its deployment), whoever called the service.
        """
        principal = as_principal or self.actor
        if principal != self.actor and self.actor != SYSTEM and not brokered and \
                not self.world.may_act_as(self.actor, principal):
            raise Forbidden(f"{self.actor} cannot act as {principal}")
        if self.depth >= 8:
            raise InvalidAction("call depth limit exceeded")
        ep = self.world.resolve(tool, principal)
        if ep.host == "admin" and principal != SYSTEM:
            raise Forbidden("admin tools are reserved for the simulator operator")
        args = _sorted_keys(S.with_defaults(ep.params, args or {}))
        errs = S.validate(ep.params, args)
        if errs:
            raise InvalidAction("; ".join(errs[:5]), tool=tool)
        sub = Ctx(self.world, principal, self.cmd, origin=self.origin, depth=self.depth + 1)
        sub.ext_key = self.ext_key
        sub.events = self.events
        sub.ext = self.ext
        sub.tool, sub.tool_args, sub.read_only, sub.gov_approved = tool, args, ep.read_only, self.gov_approved
        mark = self.state.savepoint()
        n_ev, seq0, n_ext = len(self.events), self.world.seq, len(self.ext)
        try:
            return ep.handler(sub, **args)
        except CrucibleError as e:
            self.state.rollback_to(mark)
            del self.events[n_ev:]          # drop events emitted by the failed sub-call
            self.world.seq = seq0
            # keep external outcomes: they happened and regeneration must see them in the same order
            for t, payload, vis in e.persist:   # facts about the refusal itself survive (once) in the caller
                self.emit(t, payload, vis)
            e.persist = []
            raise
        finally:
            self._rng_n += sub._rng_n + 1000

    def run_as(self, principal: str, fn: Callable[..., Any], *args, **kw) -> Any:
        """Module-internal: run `fn(sub_ctx, ...)` acting as `principal` inside this transaction. If it raises a
        CrucibleError its partial effects are rolled back to a savepoint and the error propagates."""
        sub = Ctx(self.world, principal, self.cmd, origin=self.origin, depth=self.depth + 1)
        sub.ext_key, sub.events, sub.ext = self.ext_key, self.events, self.ext
        sub.tool, sub.tool_args, sub.read_only, sub.gov_approved = self.tool, self.tool_args, self.read_only, \
            self.gov_approved
        mark = self.state.savepoint()
        n_ev, seq0 = len(self.events), self.world.seq
        try:
            return fn(sub, *args, **kw)
        except CrucibleError:
            self.state.rollback_to(mark)
            del self.events[n_ev:]
            self.world.seq = seq0
            raise
        finally:
            self._rng_n += sub._rng_n + 1000

    def reserve(self) -> tuple[int, Any]:
        """Reserve the next external-outcome slot (before running anything that may nest further external calls).
        Returns (slot, recorded outcome during regeneration or None)."""
        i = len(self.ext)
        self.ext.append(None)
        return i, self.world._regen_external(self.ext_key, i)

    def fill(self, slot: int, kind: str, out: Any) -> Any:
        out = to_jsonable(out)
        self.ext[slot] = {"kind": kind, "out": out}
        return out

    def external(self, kind: str, fn: Callable[[], Any]) -> Any:
        """Run a non-deterministic external computation (sandboxed code, measurement) once and record its
        outcome. During regeneration the recorded outcome is returned instead of re-running `fn`."""
        rec = self.world._regen_external(self.ext_key, len(self.ext))
        if rec is not None:
            out = rec
        else:
            out = to_jsonable(fn())
        self.ext.append({"kind": kind, "out": out})
        return out

    def schedule(self, at: int, proc: str, args: dict | None = None, prio: int = 50, period: int | None = None,
                 key: str | None = None) -> str:
        if proc not in self.world.processes:
            raise ProtocolError(f"unknown process {proc}")
        if at < self.now:
            at = self.now
        n = self.state.next_counter("sched")
        sid = key or f"job_{n:08d}"
        if key and self.state.has("sched", key):
            self.emit("sched.del", {"id": key}, vis="internal")
        self.emit("sched.add", {"id": sid, "at": int(at), "prio": prio, "n": n, "proc": proc, "args": args or {},
                                "period": period}, vis="internal")
        return sid

    def unschedule(self, sid: str) -> None:
        if self.state.has("sched", sid):
            self.emit("sched.del", {"id": sid}, vis="internal")


# ====================================================================== the world
class World:
    def __init__(self, spec: WorldSpec, world_id: str, store=None, *, strict: bool = False):
        self.spec = spec
        self.world_id = world_id
        self.store = store
        self.strict = strict
        self.state = StateStore()
        self.seq = 0
        self.last_hash = GENESIS_HASH
        self.log: list[Event] = []          # in-memory event log since load (index = seq - log_base - 1)
        self.log_base = 0
        self.log_retain = int(spec.params.get("log_retain", 2_000_000))
        self._unflushed: list[Event] = []
        self._heap: list[tuple] = []
        self._seq_at_mark: dict[int, int] = {}
        self._tx_seq0 = 0
        self._regen: dict | None = None
        self.modules: dict[str, Module] = {}
        self.endpoints: dict[str, Endpoint] = {}
        self.appliers: dict[str, Callable] = {}
        self.processes: dict[str, Callable] = {}
        self.resolvers: list[Callable[[str, str], Endpoint | None]] = []
        self.listeners: list[Callable[[Event], None]] = []
        self.hooks: dict[str, Callable] = {}
        self.runtime: dict[str, Any] = {}  # non-state runtime resources (sandbox workers, caches)
        self.stats = {"commands": 0, "rejected": 0, "events": 0, "procs": 0}
        self._install_core()
        for m in spec.modules:
            cls = MODULES.get(m["name"])
            if cls is None:
                raise ProtocolError(f"unknown module {m['name']!r} (registered: {sorted(MODULES)})")
            self._add_module(cls(m.get("config")))
        for mod in self.modules.values():
            for dep in mod.depends:
                if dep not in self.modules:
                    raise ProtocolError(f"module {mod.name} requires {dep}")

    # ------------------------------------------------------------------ construction
    def _install_core(self) -> None:
        st = self.state
        st.declare_table("principals", index=("kind",))
        st.declare_table("sched")
        st.declare_table("idem")
        st.declare_table("knowledge", index=("principal",))
        st.meta.setdefault("clock", 0)
        for name, fn in [("cmd", self._ap_cmd), ("cmd.rejected", self._ap_rejected), ("proc", self._ap_proc),
                         ("proc.failed", _ap_noop), ("time", self._ap_time), ("sched.add", self._ap_sched_add),
                         ("sched.del", self._ap_sched_del), ("idem", self._ap_idem), ("ext", _ap_noop),
                         ("principal.add", self._ap_principal_add), ("principal.update", self._ap_principal_update),
                         ("know", self._ap_know), ("note", _ap_noop), ("ctr", self._ap_ctr)]:
            self.appliers[name] = fn
        self.endpoints_core()

    def endpoints_core(self) -> None:
        def whoami(ctx):
            p = ctx.state.get("principals", ctx.actor)
            return {"principal": ctx.actor, "kind": p["kind"] if p else None, "time": ctx.now}

        def describe(ctx, host: str):
            eps = self.endpoints_for_host(host, ctx.actor)
            if not eps:
                raise UnknownTool(f"no service at host {host!r}")
            self.learn(ctx, ctx.actor, "host", host)
            return {"host": host, "endpoints": [e.spec() for e in eps]}

        def clock(ctx):
            return {"time": ctx.now, "day": ctx.now // 86400, "hour": (ctx.now % 86400) // 3600}

        for fn, name, params, doc, ro in [
            (whoami, "whoami", S.obj(), "Your principal id and the current world time.", True),
            (clock, "clock", S.obj(), "Current simulated time.", True),
            (describe, "describe", S.obj({"host": S.s_str("service host name", maxLength=64)}, ["host"]),
             "Fetch the API description of any service host (built-in or agent-deployed).", True),
        ]:
            self.endpoints["world." + name] = Endpoint("world." + name, "world", params, fn, doc, 0, ro)
        if self.spec.params.get("discovery", True) is not False:
            # 4.0: world.find_tools / hosts / validate / help / api (core/discovery.py)
            from . import discovery
            discovery.install(self)

    def _add_module(self, mod: Module) -> None:
        mod.world = self
        host = mod.host or mod.name
        self.modules[mod.name] = mod
        for attr in dir(type(mod)):
            fn = getattr(type(mod), attr, None)
            if fn is None or not callable(fn):
                continue
            bound = getattr(mod, attr)
            if hasattr(fn, "__crucible_endpoint__"):
                e = fn.__crucible_endpoint__
                full = f"{host}.{e['name']}"
                if full in self.endpoints:
                    raise ProtocolError(f"duplicate endpoint {full}")
                self.endpoints[full] = Endpoint(full, host, e["params"], bound, e["doc"], e["cost_s"], e["read_only"],
                                                e["returns"], version=str(e.get("version") or mod.version))
            if hasattr(fn, "__crucible_applier__"):
                et = fn.__crucible_applier__
                if et in self.appliers:
                    raise ProtocolError(f"duplicate applier {et}")
                self.appliers[et] = bound
            if hasattr(fn, "__crucible_process__"):
                self.processes[f"{mod.name}.{fn.__crucible_process__}"] = bound
        mod.install(self)

    def module(self, name: str) -> Module:
        return self.modules[name]

    @classmethod
    def create(cls, spec: WorldSpec, world_id: str | None = None, store=None, strict: bool = False) -> "World":
        world_id = world_id or f"w-{spec.hash()[:12]}"
        w = cls(spec, world_id, store, strict=strict)
        if store is not None:
            store.create_world(world_id, spec.to_dict(), parent=None)
        w._run_system("world.genesis", lambda ctx: w._genesis(ctx))
        for mod in w.modules.values():
            mod.on_load(w)
        w.flush()
        return w

    def _genesis(self, ctx: Ctx) -> None:
        ctx.emit("note", {"what": "genesis", "spec": self.spec.hash()}, vis="internal")
        self._ensure_principal(ctx, SYSTEM, "system", {"name": "Crucible kernel"})
        for mod in self.modules.values():
            mod.genesis(ctx)

    # ------------------------------------------------------------------ randomness
    def rng_for(self, *keys: Any) -> np.random.Generator:
        h = digest([self.spec.seed, *[str(k) for k in keys]], size=16)
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(h, 16))))

    # ------------------------------------------------------------------ time
    @property
    def now(self) -> int:
        return self.state.meta["clock"]

    # ------------------------------------------------------------------ principals & knowledge
    def _ensure_principal(self, ctx: Ctx, pid: str, kind: str, info: dict | None = None) -> None:
        if not self.state.has("principals", pid):
            ctx.emit("principal.add", {"id": pid, "kind": kind, "info": info or {}}, vis="internal")

    def principal(self, pid: str):
        return self.state.get("principals", pid)

    def principals_of(self, actor: str) -> set[str]:
        """Principals whose notifications this actor sees (itself plus organisations it administers)."""
        fn = self.hooks.get("principals_of")
        return fn(self, actor) if fn else {actor}

    def may_act_as(self, actor: str, principal: str) -> bool:
        fn = self.hooks.get("may_act_as")
        return bool(fn and fn(self, actor, principal))

    def learn(self, ctx: Ctx, principal: str, kind: str, value: str) -> None:
        """Record that a principal now knows of something (e.g. a host). Drives dynamic tool lists."""
        kid = f"{principal}|{kind}|{value}"
        if not self.state.has("knowledge", kid):
            ctx.emit("know", {"id": kid, "principal": principal, "kind": kind, "value": value}, vis=(principal,))

    def knows(self, principal: str, kind: str, value: str) -> bool:
        return self.state.has("knowledge", f"{principal}|{kind}|{value}")

    # ------------------------------------------------------------------ tool resolution
    def resolve(self, tool: str, principal: str) -> Endpoint:
        ep = self.endpoints.get(tool)
        if ep is not None:
            return ep
        for r in self.resolvers:
            ep = r(tool, principal)
            if ep is not None:
                return ep
        raise UnknownTool(f"unknown tool {tool!r}")

    def endpoints_for_host(self, host: str, principal: str) -> list[Endpoint]:
        out = [e for n, e in sorted(self.endpoints.items()) if e.host == host]
        if not out:
            fn = self.hooks.get("host_endpoints")
            if fn:
                out = fn(self, host, principal)
        return out

    def available_tools(self, principal: str) -> list[dict]:
        """Built-in well-known tools plus every agent-deployed service the principal has discovered."""
        tools = [e.spec() for _, e in sorted(self.endpoints.items()) if e.host != "admin" or principal == SYSTEM]
        fn = self.hooks.get("host_endpoints")
        if fn:
            for kid in self.state.find_ids("knowledge", "principal", principal):
                rec = self.state.tables["knowledge"][kid]
                if rec["kind"] == "host" and not any(e.host == rec["value"] for e in self.endpoints.values()):
                    tools.extend(e.spec() for e in fn(self, rec["value"], principal))
        return tools

    # ------------------------------------------------------------------ event emission & application
    def _emit(self, ctx: Ctx, type: str, payload: dict, vis) -> Event:
        if not self.state.in_tx:
            raise ProtocolError("emit outside transaction")
        if vis is None:
            vis = (ctx.actor,)
        elif isinstance(vis, (list, set, frozenset)):
            vis = tuple(sorted(vis))
        elif isinstance(vis, str) and vis not in ("public", "internal"):
            vis = (vis,)
        self.seq += 1
        ev = Event(self.seq, self.now, type, ctx.actor, ctx.cmd, payload, vis)
        self._apply(ev)
        ctx.events.append(ev)
        return ev

    def _apply(self, ev: Event) -> None:
        fn = self.appliers.get(ev.type)
        if fn is None:
            raise ProtocolError(f"no applier for event type {ev.type!r}")
        fn(ev)

    def _commit(self, ctx: Ctx) -> None:
        if self.state.ctr_dirty:
            dirty = {k: self.state.meta[k] for k in sorted(self.state.ctr_dirty) if k in self.state.meta}
            self.state.ctr_dirty = {}
            if dirty:
                self._emit(ctx, "ctr", dirty, "internal")
        h = self.last_hash
        for ev in ctx.events:
            h = chain_hash(h, canonical_bytes(ev.body()))
            ev.hash = h
        self.last_hash = h
        self.state.meta["seq"] = self.seq
        self.state.meta["hash"] = h
        self.state.commit()
        self.log.extend(ctx.events)
        if len(self.log) > self.log_retain:
            drop = len(self.log) - self.log_retain // 2
            if self.store is not None:
                self.flush()
            self.log = self.log[drop:]
            self.log_base += drop
        self._unflushed.extend(ctx.events)
        self.stats["events"] += len(ctx.events)
        for ev in ctx.events:
            for fn in self.listeners:
                fn(ev)

    def _begin(self) -> None:
        self.state.begin()
        self._tx_seq0 = self.seq
        self._seq_at_mark = {}

    def _rollback(self) -> None:
        self.state.rollback()
        self.seq = self._tx_seq0

    # ------------------------------------------------------------------ commands
    def execute(self, actor: str, action: dict) -> ActionResult:
        """Execute one action for an actor. Never raises for in-world failures: they become `ok=False`."""
        if self.state.in_tx:
            raise ProtocolError("re-entrant execute")
        tool = action.get("tool") if isinstance(action, dict) else None
        args = action.get("args", {}) if isinstance(action, dict) else None
        key = action.get("idempotency_key") if isinstance(action, dict) else None
        if isinstance(args, dict) and not _json_ok(args):
            args = None  # NaN/Infinity or non-JSON values: a malformed action (and unhashable in an event)
        self.stats["commands"] += 1
        p = self.state.get("principals", actor)
        if p is None or p.get("status") != "active" or p["kind"] not in ("person", "service"):
            return self._reject(actor, tool, args, Unauthenticated(f"{actor} is not an active actor"))
        if key is not None:
            rec = self.state.get("idem", f"{actor}|{key}")
            if rec is not None:
                return ActionResult(rec["ok"], to_jsonable(rec["result"]), rec.get("error"), [], 0, self.now,
                                    replayed=True, tool=tool or "")
        if not isinstance(tool, str) or not isinstance(args, dict):
            return self._hinted(self._reject(actor, tool, args, InvalidAction(_MALFORMED)), actor, tool, args)
        try:
            ep = self.resolve(tool, actor)
            if ep.host == "admin":
                raise Forbidden("admin tools are reserved for the simulator operator")
        except CrucibleError as e:
            return self._hinted(self._reject(actor, tool, args, e), actor, tool, args)
        # canonical argument order: live execution and regeneration (args read back from sorted JSON) must see
        # identical inputs, or anything that iterates over them (validation messages, handlers) could diverge
        args = _sorted_keys(S.with_defaults(ep.params, args))
        errs = S.validate(ep.params, args)
        if errs:
            return self._hinted(self._reject(actor, tool, args, InvalidAction("; ".join(errs[:6]), tool=tool)),
                                actor, tool, args, ep)
        limit = int(self.spec.params.get("max_actions_per_hour", 0))
        if limit:
            hour = self.now // 3600
            used = p.get("hour_actions", 0) if p.get("hour") == hour else 0
            if used >= limit:
                return self._reject(actor, tool, args, RateLimited(f"limit of {limit} actions per simulated hour"))
        cmd_no = self.state.meta_get("cmd_counter", 0) + 1
        self._begin()
        ctx = Ctx(self, actor, cmd_no)
        ctx.tool, ctx.tool_args, ctx.read_only = tool, args, ep.read_only
        try:
            ctx.emit("cmd", {"n": cmd_no, "tool": tool, "args": args, "key": key, "cost_s": ep.cost_s})
            result = ep.handler(ctx, **args)
            result = to_jsonable(result) if result is not None else None
            if ctx.ext:
                ctx.emit("ext", {"outcomes": ctx.ext, "key": ctx.ext_key}, vis="internal")
            if key is not None:
                ctx.emit("idem", {"id": f"{actor}|{key}", "ok": True, "result": result}, vis="internal")
            self._commit(ctx)
            return ActionResult(True, result, None, [e.seq for e in ctx.events], ep.cost_s, self.now, tool=tool)
        except CrucibleError as e:
            ext = list(ctx.ext)
            self._rollback()
            return self._hinted(self._reject(actor, tool, args, e, cmd_no=cmd_no, ext=ext, key=key, cost=ep.cost_s),
                                actor, tool, args)
        except Exception as e:  # a handler bug: never let it corrupt state
            self._rollback()
            if self.strict:
                raise
            return self._reject(actor, tool, args, CrucibleError(f"internal error: {type(e).__name__}: {e}"))

    def _hinted(self, res: ActionResult, actor, tool, args, ep: Endpoint | None = None) -> ActionResult:
        """4.0: structured detail (near-miss tool names, schema paths) on the error RETURNED for a rejected
        action. The recorded `cmd.rejected` event is untouched, so logs stay byte-identical to 3.0."""
        if self.spec.params.get("discovery", True) is False:
            return res
        from . import discovery
        return discovery.with_hints(self, actor, tool, args, res, ep)

    def _reject(self, actor, tool, args, err: CrucibleError, cmd_no=None, ext=None, key=None, cost=0) -> ActionResult:
        self.stats["rejected"] += 1
        self._begin()
        ctx = Ctx(self, actor if self.state.has("principals", actor) else SYSTEM, cmd_no)
        payload = {"tool": tool if isinstance(tool, str) else repr(tool)[:200], "actor": actor if isinstance(actor, str) else None,
                   "args": args if isinstance(args, dict) else None, "error": err.as_dict()}
        if cmd_no is not None:
            payload["n"] = cmd_no
        if ext:
            payload["ext"] = ext
            payload["ext_key"] = ctx.ext_key      # same first-seq as the rolled-back transaction
        ctx.emit("cmd.rejected", payload, vis=(actor,) if isinstance(actor, str) else "internal")
        for t, p_, v_ in getattr(err, "persist", []):
            if t in self.appliers:
                ctx.emit(t, p_, v_)
        if key is not None and self.state.has("principals", actor):
            ctx.emit("idem", {"id": f"{actor}|{key}", "ok": False, "result": None, "error": err.as_dict()},
                     vis="internal")
        self._commit(ctx)
        return ActionResult(False, None, err.as_dict(), [e.seq for e in ctx.events], cost, self.now,
                            tool=tool if isinstance(tool, str) else "")

    def query(self, actor: str, tool: str, args: dict | None = None) -> Any:
        """Evaluate a read-only endpoint for rendering (web pages, dashboards) WITHOUT recording a command: the
        transaction is always rolled back, so the world state and event log are unchanged. Raises CrucibleError."""
        if self.state.in_tx:
            raise ProtocolError("re-entrant query")
        p = self.state.get("principals", actor)
        if p is None or p.get("status") != "active":
            raise Unauthenticated(f"{actor} is not an active principal")
        ep = self.resolve(tool, actor)
        if not ep.read_only or ep.host == "admin":
            raise Forbidden(f"{tool} is not a read-only query")
        args = S.with_defaults(ep.params, args or {})
        errs = S.validate(ep.params, args)
        if errs:
            raise InvalidAction("; ".join(errs[:6]))
        self._begin()
        try:
            ctx = Ctx(self, actor, None)
            return to_jsonable(ep.handler(ctx, **args))
        finally:
            self._rollback()

    def system_call(self, tool: str, args: dict | None = None, as_principal: str = SYSTEM) -> Any:
        """Research/orchestration code acting with system authority (world setup, interventions). Recorded."""
        out = {}

        def run(ctx):
            ctx.emit("cmd", {"n": None, "tool": tool, "args": args or {}, "system": True, "as": as_principal},
                     vis="internal")
            out["r"] = ctx.call(tool, args or {}, as_principal=as_principal)
        self._run_system("sys.call", run)
        return out["r"]

    def _run_system(self, label: str, fn: Callable[[Ctx], None]) -> None:
        self._begin()
        ctx = Ctx(self, SYSTEM, None)
        try:
            fn(ctx)
            if ctx.ext:
                ctx.emit("ext", {"outcomes": ctx.ext, "key": ctx.ext_key}, vis="internal")
            self._commit(ctx)
        except BaseException:
            self._rollback()
            raise

    # ------------------------------------------------------------------ time & processes
    def advance_time(self, until: int, max_jobs: int | None = None) -> int:
        """Run every scheduled process due at or before `until` in order, then set the clock to `until`.
        Returns the number of process runs."""
        if until < self.now:
            raise ProtocolError(f"time cannot go backwards ({until} < {self.now})")
        runs = 0
        while self._heap:
            at, prio, n, sid = self._heap[0]
            if at > until:
                break
            heapq.heappop(self._heap)
            job = self.state.tables["sched"].get(sid)
            if job is None or job["at"] != at or job["n"] != n:
                continue  # stale heap entry
            self._run_job(sid, job)
            runs += 1
            if max_jobs is not None and runs >= max_jobs:
                return runs
        if until > self.now:
            self._run_system("time", lambda ctx: ctx.emit("time", {"to": int(until)}, vis="internal"))
        return runs

    def next_job_time(self) -> int | None:
        while self._heap:
            at, prio, n, sid = self._heap[0]
            job = self.state.tables["sched"].get(sid)
            if job is None or job["at"] != at or job["n"] != n:
                heapq.heappop(self._heap)
                continue
            return at
        return None

    def _run_job(self, sid: str, job: dict) -> None:
        job = dict(job)
        self.stats["procs"] += 1
        fn = self.processes[job["proc"]]
        self._begin()
        ctx = Ctx(self, SYSTEM, None)
        try:
            ctx.emit("proc", {"id": sid, "proc": job["proc"], "args": job["args"], "at": job["at"]}, vis="internal")
            ctx.emit("sched.del", {"id": sid}, vis="internal")
            if job.get("period"):
                ctx.schedule(job["at"] + job["period"], job["proc"], job["args"], job["prio"], job["period"], key=sid)
            fn(ctx, **job["args"])
            if ctx.ext:
                ctx.emit("ext", {"outcomes": ctx.ext, "key": ctx.ext_key}, vis="internal")
            self._commit(ctx)
        except Exception as e:
            ext = list(ctx.ext)
            self._rollback()
            if self.strict and not isinstance(e, CrucibleError):
                heapq.heappush(self._heap, (job["at"], job["prio"], job["n"], sid))  # job is still scheduled
                raise

            def record(c):
                c.emit("proc", {"id": sid, "proc": job["proc"], "args": job["args"], "at": job["at"]}, vis="internal")
                c.emit("proc.failed", {"id": sid, "proc": job["proc"], "error": f"{type(e).__name__}: {e}",
                                       "ext": ext, "ext_key": c.ext_key}, vis="internal")
                c.emit("sched.del", {"id": sid}, vis="internal")
                if job.get("period"):
                    c.schedule(job["at"] + job["period"], job["proc"], job["args"], job["prio"], job["period"], key=sid)
            self._run_system("proc.failed", record)

    # ------------------------------------------------------------------ core appliers
    def _ap_cmd(self, ev: Event) -> None:
        if ev.payload.get("n") is not None:
            self.state.meta_set("cmd_counter", ev.payload["n"])
            p = self.state.tables["principals"].get(ev.actor)
            if p is not None:
                hour = ev.t // 3600
                used = p.get("hour_actions", 0) if p.get("hour") == hour else 0
                self.state.patch("principals", ev.actor, hour=hour, hour_actions=used + 1,
                                 actions=p.get("actions", 0) + 1, time_used=p.get("time_used", 0) + ev.payload["cost_s"])

    def _ap_ctr(self, ev: Event) -> None:
        for k, v in ev.payload.items():
            if self.state.meta.get(k, 0) < v:
                self.state.meta_set(k, v)

    def _ap_rejected(self, ev: Event) -> None:
        if ev.payload.get("n") is not None:
            self.state.meta_set("cmd_counter", ev.payload["n"])

    def _ap_proc(self, ev: Event) -> None:
        self.state.meta_set("clock", ev.payload["at"])
        ev.t = ev.payload["at"]

    def _ap_time(self, ev: Event) -> None:
        self.state.meta_set("clock", ev.payload["to"])

    def _ap_sched_add(self, ev: Event) -> None:
        p = ev.payload
        self.state.insert("sched", p["id"], {"at": p["at"], "prio": p["prio"], "n": p["n"], "proc": p["proc"],
                                            "args": p["args"], "period": p["period"]})
        heapq.heappush(self._heap, (p["at"], p["prio"], p["n"], p["id"]))

    def _ap_sched_del(self, ev: Event) -> None:
        self.state.delete("sched", ev.payload["id"])

    def _ap_idem(self, ev: Event) -> None:
        p = ev.payload
        self.state.put("idem", p["id"], {"ok": p["ok"], "result": p["result"], "error": p.get("error")})

    def _ap_principal_add(self, ev: Event) -> None:
        p = ev.payload
        self.state.insert("principals", p["id"], {"kind": p["kind"], "status": p.get("status", "active"),
                                                 "info": p.get("info", {}), "created": ev.t, "actions": 0})

    def _ap_principal_update(self, ev: Event) -> None:
        p = ev.payload
        self.state.patch("principals", p["id"], **p["changes"])

    def _ap_know(self, ev: Event) -> None:
        p = ev.payload
        self.state.insert("knowledge", p["id"], {"principal": p["principal"], "kind": p["kind"], "value": p["value"],
                                                "t": ev.t})

    # ------------------------------------------------------------------ observation
    def visible(self, ev: Event, principals: set[str]) -> bool:
        if ev.vis == "public":
            return True
        if ev.vis == "internal":
            return False
        return any(p in principals for p in ev.vis)

    def events_range(self, from_seq: int, to_seq: int | None = None) -> list[Event]:
        """Events with from_seq < seq <= to_seq."""
        to_seq = self.seq if to_seq is None else to_seq
        if from_seq >= self.log_base:
            return self.log[from_seq - self.log_base: to_seq - self.log_base]
        if self.store is None:
            raise ProtocolError("events before the in-memory window and no store")
        self.flush()
        return [Event.from_dict(d) for d in self.store.read_events(self.world_id, from_seq, to_seq)]

    def visible_events(self, actor: str, since_seq: int, limit: int = 50) -> tuple[list[Event], int]:
        ps = self.principals_of(actor)
        evs = [e for e in self.events_range(max(since_seq, self.log_base)) if self.visible(e, ps)]
        dropped = max(0, len(evs) - limit)
        return evs[-limit:], dropped

    def observe(self, actor: str) -> dict:
        out = {"time": self.now, "principal": actor}
        for name, mod in self.modules.items():
            o = mod.observe(self, actor)
            if o:
                out[name] = o
        return out

    # ------------------------------------------------------------------ persistence
    def flush(self) -> None:
        if self.store is not None and self._unflushed:
            self.store.append_events(self.world_id, [e.to_dict() for e in self._unflushed])
        self._unflushed = []

    def snapshot_blob(self) -> bytes:
        if self.state.in_tx:
            raise ProtocolError("snapshot inside transaction")
        import pickle
        payload = {"spec": self.spec.to_dict(), "world_id": self.world_id, "seq": self.seq, "hash": self.last_hash,
                   "state": self.state.dumps(), "kernel": KERNEL_VERSION}
        return zlib.compress(pickle.dumps(payload, protocol=5), 1)

    def snapshot(self) -> str:
        """Persist a snapshot in the store and return its id."""
        if self.store is None:
            raise ProtocolError("snapshot() needs a store; use snapshot_blob() for in-memory use")
        self.flush()
        blob = self.snapshot_blob()
        sid = f"s-{digest([self.world_id, self.seq, self.last_hash])[:16]}"
        self.store.put_snapshot(sid, self.world_id, self.seq, self.last_hash, blob)
        return sid

    @classmethod
    def from_blob(cls, blob: bytes, world_id: str | None = None, store=None, fork: bool = False,
                  strict: bool = False) -> "World":
        import pickle
        d = pickle.loads(zlib.decompress(blob))
        spec = WorldSpec.from_dict(d["spec"])
        if fork and store is not None and world_id in (None, d["world_id"]):
            # a fork is another world: under its parent's id the store would record it as its own parent
            world_id = f"{d['world_id']}.f{d['seq']}"
        w = cls(spec, world_id or d["world_id"], store, strict=strict)
        loaded = StateStore.loads(d["state"])
        # keep table declarations from install(), take data from snapshot
        for name, fields in w.state.index_fields.items():
            loaded.declare_table(name, fields)
        w.state = loaded
        w.seq = d["seq"]
        w.last_hash = d["hash"]
        w.log_base = w.seq
        w._rebuild_heap()
        if fork and store is not None:
            store.create_world(w.world_id, spec.to_dict(), parent={"world": d["world_id"], "seq": d["seq"],
                                                                   "hash": d["hash"]})
            store.put_snapshot(f"s-{digest([w.world_id, w.seq, w.last_hash])[:16]}", w.world_id, w.seq, w.last_hash,
                               blob)
        for mod in w.modules.values():
            mod.on_load(w)
        return w

    @classmethod
    def load(cls, store, world_id: str, strict: bool = False, verify: bool = True) -> "World":
        """Crash recovery / resume: latest snapshot of this world (if any) + replay of the events after it."""
        info = store.world_info(world_id)
        if info is None:
            raise ProtocolError(f"no world {world_id} in store")
        spec = WorldSpec.from_dict(info["spec"])
        snap = store.latest_snapshot(world_id)
        base = store.get_snapshot(snap["id"])[1] if snap else None
        if base is None and info.get("parent"):
            raise ProtocolError(f"fork {world_id} has no base snapshot")
        from_seq = snap["seq"] if snap else 0
        w = cls.replay(spec, store.read_events(world_id, from_seq=from_seq), world_id=world_id, verify=verify,
                       base_blob=base)
        w.store = store
        w.strict = strict
        return w

    def fork(self, world_id: str, store=None) -> "World":
        """Copy-on-snapshot fork: an isolated world that shares history up to now and diverges afterwards."""
        return World.from_blob(self.snapshot_blob(), world_id=world_id, store=store if store is not None else None,
                               fork=True, strict=self.strict)

    def _rebuild_heap(self) -> None:
        self._heap = [(j["at"], j["prio"], j["n"], sid) for sid, j in self.state.tables["sched"].items()]
        heapq.heapify(self._heap)

    @classmethod
    def replay(cls, spec: WorldSpec, events: Iterable[dict], world_id: str = "replay", verify: bool = True,
               base_blob: bytes | None = None) -> "World":
        """Rebuild a world by applying recorded events (from genesis, or on top of a snapshot)."""
        if base_blob is not None:
            w = cls.from_blob(base_blob, world_id=world_id)
        else:
            w = cls(spec, world_id)
        h = w.last_hash
        for d in events:
            ev = Event.from_dict(d)
            if ev.seq <= w.seq:
                continue
            if ev.seq != w.seq + 1:
                raise InvariantViolation(f"gap in event log at {w.seq + 1} (got {ev.seq})")
            if verify:
                h = chain_hash(h, canonical_bytes(ev.body()))
                if h != ev.hash:
                    raise InvariantViolation(f"hash chain broken at seq {ev.seq}")
            w.state.begin()
            w._apply(ev)
            w.state.commit()
            w.seq = ev.seq
            w.last_hash = ev.hash or h
            w.state.meta["seq"] = w.seq
            w.state.meta["hash"] = w.last_hash
        w.log_base = w.seq
        w._rebuild_heap()
        for mod in w.modules.values():
            mod.on_load(w)
        return w

    # ------------------------------------------------------------------ regeneration (determinism check)
    def _regen_external(self, cmd: int | None, i: int):
        if self._regen is None:
            return None
        outs = self._regen.get(cmd)
        if outs is None or i >= len(outs) or outs[i] is None:
            return None
        return outs[i]["out"]

    @classmethod
    def regenerate(cls, spec: WorldSpec, events: list[dict], world_id: str = "regen") -> "World":
        """Re-execute the recorded commands and clock advances through the handlers (not the appliers), with
        recorded external outcomes injected. If the kernel and modules are deterministic, the resulting event
        log is identical to the original (same hashes)."""
        evs = [Event.from_dict(d) for d in events]
        ext_by_cmd: dict[Any, list] = {}
        for e in evs:   # recorded external outcomes, keyed by the first seq of their transaction
            if e.type == "ext":
                k = e.payload.get("key", e.cmd)
                if k is not None:
                    ext_by_cmd.setdefault(k, []).extend(e.payload["outcomes"])
            if e.type in ("cmd.rejected", "proc.failed") and e.payload.get("ext"):
                k = e.payload.get("ext_key", e.cmd)
                if k is not None:
                    ext_by_cmd.setdefault(k, []).extend(e.payload["ext"])
        w = cls.create(spec, world_id=world_id)
        w._regen = ext_by_cmd
        i = 0
        n_gen = w.seq
        while i < len(evs) and evs[i].seq <= n_gen:
            i += 1
        for j in range(i, len(evs)):
            e = evs[j]
            if e.type == "cmd" and e.payload.get("n") is not None:
                w.execute(e.actor, {"tool": e.payload["tool"], "args": e.payload["args"],
                                    **({"idempotency_key": e.payload["key"]} if e.payload.get("key") is not None
                                       else {})})
            elif e.type == "cmd" and e.payload.get("system"):
                w.system_call(e.payload["tool"], e.payload["args"], as_principal=e.payload["as"])
            elif e.type == "cmd.rejected":
                # rejected commands leave no "cmd" event (it was rolled back); re-execute them from here
                actor = e.payload.get("actor") or e.actor
                if e.payload.get("n") is None and e.payload["error"].get("message") == _MALFORMED:
                    # the original (non-str tool, non-object args) is not recoverable from the log: re-record it
                    w._reject(actor, e.payload["tool"], e.payload["args"], InvalidAction(_MALFORMED))
                    continue
                act = {"tool": e.payload["tool"], "args": e.payload["args"]}
                nxt = evs[j + 1] if j + 1 < len(evs) else None
                if e.cmd is not None and nxt is not None and nxt.type == "idem" and nxt.cmd == e.cmd:
                    act["idempotency_key"] = nxt.payload["id"][len(actor) + 1:]  # id = f"{actor}|{key}"
                w.execute(actor, act)
            elif e.type == "proc":
                # one scheduled job run (also when advance_time stopped exactly at the job time or at max_jobs)
                w.advance_time(e.payload["at"], max_jobs=1)
            elif e.type == "time":
                w.advance_time(e.payload["to"])
        w._regen = None
        return w

    # ------------------------------------------------------------------ integrity
    def check_invariants(self) -> list[str]:
        errs = []
        for name, mod in self.modules.items():
            errs.extend(f"{name}: {m}" for m in mod.check_invariants(self))
        for sid, j in self.state.tables["sched"].items():
            if j["at"] < self.now:
                errs.append(f"kernel: job {sid} overdue at {j['at']} < {self.now}")
        return errs

    def digest(self) -> str:
        return self.state.digest()

    def close(self) -> None:
        self.flush()
        for mod in self.modules.values():
            mod.close()


def _ap_noop(ev: Event) -> None:
    return None


def _sorted_keys(x):
    if isinstance(x, dict):
        return {k: _sorted_keys(x[k]) for k in sorted(x, key=str)}
    if isinstance(x, list):
        return [_sorted_keys(v) for v in x]
    return x


_MALFORMED = "action must be {'tool': str, 'args': object}"


def _json_ok(x: Any) -> bool:
    try:
        canonical_bytes(x)
        return True
    except (TypeError, ValueError):
        return False
