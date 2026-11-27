"""Communication discipline and determinism of the shared substrate (MASTER_SPEC Part 2.3; requirement SP-2.3b).

Part 2.3: "All systems should communicate through typed events and approved state interfaces ... Preserve
deterministic simulation and replay wherever possible." This module measures both on running worlds; it changes
nothing a world does.

* `trace_writes(world)` (a context manager) attributes every write to the world's state while it runs to the module
  that performs it: the module whose event applier is running (kernel appliers count as the kernel), or the handler
  when no applier runs. Each write is classified against the declaring owner of its target:

  - a table belongs to the module(s) that declared it in `Module.install` (`StateStore.declared_by`; the kernel's own
    tables to the kernel);
  - a dense array to the module that created it at install (`StateStore.array_declared_by`), a column of a column
    table to its declaring module (`crucible.substrate.columns`);
  - a metadata key (`meta`) to the module that declared it (`StateStore.declare_meta`), else to the module whose name
    or host namespaces it (`feed.changes.posts` -> feed, `learn:board` -> learn), else to the module of a reviewed
    4.0 legacy key (`LEGACY_META_OWNERS`), else to the kernel (`meta_owner`).

  Approved interfaces, the only ways a module changes state it does not own: a `cols.batch` / `cols.grow` event of
  the columns module whose recorded writer (`by`, with its interface) the column's ownership rule admits
  (`columns.may_write`: the owner, its package, a granted module, an interface); counter allocation by handlers
  (`ctr:` keys, recorded by the kernel's `ctr` event at commit) and the kernel's side-event count. Everything else
  is a *cross-module state write* (an applier of one module writing another module's table, array, column or
  metadata) or a *handler write* (state changed outside any applier, so replay could not reproduce it). Package
  siblings are reported apart (`same_package`) and counted as cross-module: the requirement is per module.
  A derived-state applier (`kernel.derives`) is attributed to its own module, like an applier.

  Writes that bypass the state store's methods (wave-3b repair, verifier probe p02) are found by content: the trace
  fingerprints every row and array when it starts; a row an applier (or a handler) read and changed without `put`
  (a nested value of the returned row mutated in place) is found when that applier returns (or when its transaction
  commits), a row changed in place by anything else when it is next written or when the trace ends, and an array
  that changed in a transaction that never asked for write access to it (`array_mut` / `array_rows_mut` /
  `set_array`: a write through a cached reference) when the transaction commits. Each is an *untracked write*,
  attributed to the module whose applier read the row (or to the modules whose appliers ran in the transaction),
  and counted with the cross-module writes: such a write is invisible to rollback, and outside an applier to replay.
  `trace_genesis()` traces worlds built inside it from their first event (genesis included).

* `replay_check(world)` (wave 3b) replays the world's own log from genesis: a write that bypasses the state store's
  methods (a nested value of a row mutated in place, `state.tables` changed directly) is invisible to the trace but
  breaks replay, so a digest mismatch counts as state changed outside the appliers.

* `hashes(kind, seed, hours)` builds a world of a registered kind (`KINDS`), runs it and returns its event hash,
  state digest and event count; `python -m crucible.substrate.integrity hashes <kind> <seed> <hours>` prints them,
  so a determinism check runs the same kind and seed in fresh processes and compares.

The world kinds (`KINDS`) are the v5 world families: the integrated civilisation (crucible.civ), the population
world, the social internet, the commerce world (startup economy), a Tier 1 rapid environment (Mode A) and, when a
Tier 2 domain is registered, its first configuration (Mode B).
"""
from __future__ import annotations

import contextlib
import json
import sys
from collections import Counter
from typing import Any, Callable

import numpy as np

KERNEL = "kernel"
APPROVED_META_PREFIXES = ("ctr:",)
# Metadata keys of 4.0 modules that are neither declared nor namespaced by their module (reviewed by workstream A in
# the wave-3b repair: each is written only by the appliers of the module named here). An exact key, or a prefix
# ending in ":" or ".". New keys are namespaced (`<module>.x`) or declared (`StateStore.declare_meta`).
LEGACY_META_OWNERS = {
    "phys": "industry", "vendors": "industry", "own_operators": "industry", "grid_pending": "industry",
    "cons_needs": "consumers", "cons_hour": "consumers", "shop_regions": "shoppers", "entry_paused": "entry",
    "labor_hours:": "inst", "breaches:": "deal", "goods_minted:": "market",
}


# ---------------------------------------------------------------------------------------------------- owners
def table_owners(world, table: str) -> tuple[str, ...]:
    mods = world.state.declared_by.get(table)
    return tuple(mods) if mods else (KERNEL,)


def array_owner(world, name: str) -> tuple[str | None, str | None, str | None]:
    """(owner module, column table, column) of a dense array: a column of a column table, or an array a module
    created at install; (None, None, None) when nobody declared it."""
    from . import columns
    reg = columns._registry(world)
    if "." in name:
        table, col = name.rsplit(".", 1)
        rec = reg.get(table)
        if rec is not None and col in rec.get("columns", {}):
            return rec.get("by", {}).get(col), table, col
    return world.state.array_declared_by.get(name), None, None


def meta_owner(world, key: str) -> str:
    """The module that owns metadata key `key` (see the module docstring), or the kernel."""
    st = world.state
    declared = st.meta_declared_by
    if key in declared:
        return declared[key]
    for k, mod in declared.items():
        if k[-1:] in (".", ":") and key.startswith(k):
            return mod
    for i, ch in enumerate(key):
        if ch in ".:":
            ns = key[:i]
            if ns in world.modules:
                return ns
            owner = _host_owner(world, ns)
            if owner is not None:
                return owner
            break
    if key in LEGACY_META_OWNERS:
        return LEGACY_META_OWNERS[key]
    for k, mod in LEGACY_META_OWNERS.items():
        if k[-1:] in (".", ":") and key.startswith(k):
            return mod
    return KERNEL


def _host_owner(world, host: str) -> str | None:
    hosts = world.runtime.get("_integrity_hosts")
    if hosts is None:
        hosts = world.runtime["_integrity_hosts"] = {m.host: n for n, m in world.modules.items() if m.host}
    return hosts.get(host)


def _package(name: str | None) -> str | None:
    from . import columns
    return columns._package(name)


def _module_of_applier(fn) -> str:
    mod = getattr(fn, "__self__", None)
    if mod is not None and hasattr(mod, "config") and hasattr(mod, "name"):
        return mod.name
    return KERNEL


# ---------------------------------------------------------------------------------------------------- tracing
class WriteTrace:
    """Writes observed while tracing: counters keyed by (writer, target kind, target, owner)."""

    def __init__(self, world):
        self.world = world
        self.writes: Counter = Counter()             # (writer, kind, target, owner) -> writes
        self.cross: Counter = Counter()              # cross-module writes (same keys)
        self.handler: Counter = Counter()            # writes outside any applier (same keys, writer = "handler")
        self.columns_refused: Counter = Counter()    # cols.batch writes whose recorded writer the rule refuses
        self.untracked: Counter = Counter()          # state changed without the store's methods (same keys)
        self.events: Counter = Counter()             # applied events by module
        self._stack: list[tuple[str, Any]] = []
        self._first_writer: dict[str, str] = {}
        # content fingerprints (untracked writes): rows, arrays; reads and write requests of the open frames
        self._row_fp: dict[tuple[str, str], int] = {}
        self._arr_fp: dict[str, bytes] = {}
        self._reads: list[set] = []                  # one set per applier frame on the stack
        self._handler_reads: set = set()
        self._tx_arrays: set = set()                 # arrays the open transaction asked to write
        self._tx_rows: set = set()                   # rows the open transaction wrote through the store
        self._tx_writers: set = set()                # modules whose appliers ran in the open transaction

    # attribution of one write
    def _writer(self) -> tuple[str, Any]:
        return self._stack[-1] if self._stack else ("handler", None)

    def _record(self, kind: str, target: str, owners: tuple, ev=None, col=None, table=None) -> None:
        w, ev = self._writer()
        if w == "cols" and kind == "column":
            # the approved interface: the batch's recorded writer must be admitted by the column's ownership rule
            from . import columns
            p = ev.payload if ev is not None else {}
            by = p.get("by")
            why = columns.may_write(self.world, table, col, by, p.get("iface"))
            key = (f"cols(by={by})", kind, target, owners[0])
            self.writes[key] += 1
            if why:
                self.columns_refused[key] += 1
            return
        key = (w, kind, target, ",".join(o or "?" for o in owners))
        self.writes[key] += 1
        if w == "handler":
            self.handler[key] += 1
        elif w not in owners:
            self.cross[key] += 1

    def table(self, table: str) -> None:
        self._record("table", table, table_owners(self.world, table))

    def array(self, name: str) -> None:
        owner, table, col = array_owner(self.world, name)
        if owner is None:
            owner = self._first_writer.setdefault(name, self._writer()[0])
        self._record("column" if table else "array", name, (owner,), col=col, table=table)

    def meta(self, key: str) -> None:
        if key.startswith(APPROVED_META_PREFIXES):
            return                      # counter allocation: recorded by the kernel's ctr event at commit
        from ..core.kernel import SIDE_KEY
        if key == SIDE_KEY and not self._stack:
            return                      # the kernel's own side-event count (World._apply)
        self._record("meta", key, (meta_owner(self.world, key),))

    # untracked writes (content fingerprints)
    def _untracked(self, writer: str, kind: str, target: str, owners: tuple) -> None:
        self.untracked[(writer, kind, target, ",".join(o or "?" for o in owners))] += 1

    def fingerprint_all(self) -> None:
        st = self.world.state
        self._row_fp = {(t, rid): _row_fp(rec) for t, tb in st.tables.items() for rid, rec in tb.items()}
        self._arr_fp = {n: _arr_fp(a) for n, a in st.arrays.items()}

    def read(self, table: str, rid: str) -> None:
        (self._reads[-1] if self._reads else self._handler_reads).add((table, rid))

    def check_row(self, table: str, rid: str, writer: str) -> None:
        """A row whose content no longer matches its fingerprint was changed without the store's methods."""
        key = (table, rid)
        rec = self.world.state.tables.get(table, {}).get(rid)
        fp = self._row_fp.get(key)
        if rec is None:
            self._row_fp.pop(key, None)
            return
        now = _row_fp(rec)
        if fp is not None and now != fp:
            self._untracked(writer, "table", table, table_owners(self.world, table))
        self._row_fp[key] = now

    def check_reads(self, reads: set, writer: str) -> None:
        for table, rid in sorted(reads):
            self.check_row(table, rid, writer)

    def refresh_row(self, table: str, rid: str) -> None:
        rec = self.world.state.tables.get(table, {}).get(rid)
        if rec is None:
            self._row_fp.pop((table, rid), None)
        else:
            self._row_fp[(table, rid)] = _row_fp(rec)

    def check_arrays(self) -> None:
        """At commit: arrays the transaction never asked to write must be unchanged."""
        st = self.world.state
        writers = sorted(self._tx_writers - {KERNEL}) or sorted(self._tx_writers) or ["handler"]
        for name, a in st.arrays.items():
            fp = _arr_fp(a)
            old = self._arr_fp.get(name)
            if name not in self._tx_arrays and old is not None and fp != old:
                owner, table, _col = array_owner(self.world, name)
                self._untracked("|".join(writers), "column" if table else "array", name,
                                (owner or self._first_writer.get(name, "?"),))
            self._arr_fp[name] = fp
        for name in [n for n in self._arr_fp if n not in st.arrays]:
            del self._arr_fp[name]

    def end_tx(self, rolled_back: bool) -> None:
        if rolled_back:
            st = self.world.state
            for t, rid in self._tx_rows:
                self.refresh_row(t, rid)       # the journal restored them without the store's methods
            for name in self._tx_arrays:
                if name in st.arrays:
                    self._arr_fp[name] = _arr_fp(st.arrays[name])
                else:
                    self._arr_fp.pop(name, None)
        else:
            self.check_reads(self._handler_reads, "handler")
            self.check_arrays()
        self._handler_reads = set()
        self._tx_arrays, self._tx_rows, self._tx_writers = set(), set(), set()

    def finish(self) -> None:
        """At the end of the trace: every row and array must match its last fingerprint."""
        st = self.world.state
        self.check_reads(self._handler_reads, "handler")
        self._handler_reads = set()
        for (t, rid), fp in sorted(self._row_fp.items()):
            rec = st.tables.get(t, {}).get(rid)
            if rec is not None and _row_fp(rec) != fp:
                self._untracked("unattributed", "table", t, table_owners(self.world, t))
        for (t, tb) in st.tables.items():
            for rid in tb:
                if (t, rid) not in self._row_fp:
                    self._untracked("unattributed", "table", t, table_owners(self.world, t))
        self._tx_writers = {"unattributed"}
        self.check_arrays()

    def report(self) -> dict:
        def rows(c: Counter) -> list[dict]:
            return [{"writer": w, "kind": k, "target": t, "owner": o, "writes": n}
                    for (w, k, t, o), n in sorted(c.items())]
        same_pkg = Counter({k: n for k, n in self.cross.items()
                            if _package(k[0]) is not None and _package(k[0]) in {_package(o) for o in k[3].split(",")}})
        return {"cross_module_state_writes": len(self.cross) + len(self.handler) + len(self.columns_refused)
                + len(self.untracked),
                "cross_module_write_count": int(sum(self.cross.values())),
                "handler_writes": int(sum(self.handler.values())),
                "untracked_writes": int(sum(self.untracked.values())),
                "column_writes_refused_by_rule": int(sum(self.columns_refused.values())),
                "writes_traced": int(sum(self.writes.values())),
                "writers": len({k[0] for k in self.writes}),
                "targets": len({k[2] for k in self.writes}),
                "events_applied": int(sum(self.events.values())),
                "cross": rows(self.cross), "handler": rows(self.handler), "refused": rows(self.columns_refused),
                "untracked": rows(self.untracked), "same_package": rows(same_pkg)}


def _row_fp(rec) -> int:
    import pickle
    try:
        return hash(pickle.dumps(rec, protocol=5))
    except Exception:  # noqa: BLE001 - a value pickle cannot serialise still has a stable repr within one run
        return hash(repr(rec))


def _arr_fp(a) -> bytes:
    import hashlib
    b = np.ascontiguousarray(a)
    return hashlib.blake2b(b.view(np.uint8).reshape(-1) if b.size else b"", digest_size=16).digest()


@contextlib.contextmanager
def trace_writes(world, *, untracked: bool = True):
    """Trace every state write of `world` while the block runs (see the module docstring). Yields a WriteTrace.
    `untracked=False` skips the content fingerprints (writes that bypass the store's methods are then not found)."""
    tr = WriteTrace(world)
    st = world.state
    saved_appliers = dict(world.appliers)
    saved_derivers = {et: list(fns) for et, fns in getattr(world, "derivers", {}).items()}
    names = ("put", "delete", "meta_set", "array_mut", "array_rows_mut", "set_array", "get", "scan", "find",
             "find_one", "commit", "rollback", "rollback_to")
    saved = {n: st.__dict__.get(n) for n in names}
    orig = {n: getattr(st, n) for n in names}

    def frame(mod, fn):
        def applier(ev):
            tr._stack.append((mod, ev))
            tr._reads.append(set())
            tr._tx_writers.add(mod)
            tr.events[mod] += 1
            try:
                return fn(ev)
            finally:
                tr._stack.pop()
                reads = tr._reads.pop()
                if untracked:
                    tr.check_reads(reads, mod)
        return applier

    for et, fn in saved_appliers.items():
        world.appliers[et] = frame(_module_of_applier(fn), fn)
    for et, fns in saved_derivers.items():
        world.derivers[et] = [frame(_module_of_applier(fn), fn) for fn in fns]

    def put(table, rid, record):
        tr.table(table)
        if untracked:
            tr.check_row(table, rid, tr._writer()[0])
        r = orig["put"](table, rid, record)
        if untracked:
            tr.refresh_row(table, rid)
            tr._tx_rows.add((table, rid))
        return r

    def delete(table, rid):
        if rid in st.tables.get(table, {}):
            tr.table(table)
            if untracked:
                tr.check_row(table, rid, tr._writer()[0])
                tr._tx_rows.add((table, rid))
        r = orig["delete"](table, rid)
        if untracked:
            tr.refresh_row(table, rid)
        return r

    def meta_set(key, value):
        tr.meta(key)
        return orig["meta_set"](key, value)

    def array_mut(name):
        tr.array(name)
        tr._tx_arrays.add(name)
        return orig["array_mut"](name)

    def array_rows_mut(name, rows):
        tr.array(name)
        tr._tx_arrays.add(name)
        return orig["array_rows_mut"](name, rows)

    def set_array(name, value):
        tr.array(name)
        tr._tx_arrays.add(name)
        return orig["set_array"](name, value)

    def get(table, rid):
        r = orig["get"](table, rid)
        if r is not None:
            tr.read(table, rid)
        return r

    def scan(table):
        for rid, rec in orig["scan"](table):
            tr.read(table, rid)
            yield rid, rec

    def find(table, field, value):
        ids = st.find_ids(table, field, value)
        for rid in ids:
            tr.read(table, rid)
        return orig["find"](table, field, value)

    def find_one(table, field, value):
        r = orig["find_one"](table, field, value)
        if r is not None:
            ids = st.find_ids(table, field, value)
            if ids:
                tr.read(table, ids[0])
        return r

    def commit():
        r = orig["commit"]()
        if untracked:
            tr.end_tx(rolled_back=False)
        return r

    def rollback():
        r = orig["rollback"]()
        if untracked:
            tr.end_tx(rolled_back=True)
        return r

    def rollback_to(mark):
        r = orig["rollback_to"](mark)
        if untracked:
            for t, rid in tr._tx_rows:
                tr.refresh_row(t, rid)
            for name in tr._tx_arrays:
                if name in st.arrays:
                    tr._arr_fp[name] = _arr_fp(st.arrays[name])
        return r

    wrapped = {"put": put, "delete": delete, "meta_set": meta_set, "array_mut": array_mut,
               "array_rows_mut": array_rows_mut, "set_array": set_array}
    if untracked:
        wrapped.update({"get": get, "scan": scan, "find": find, "find_one": find_one, "commit": commit,
                        "rollback": rollback, "rollback_to": rollback_to})
        tr.fingerprint_all()
    for n, f in wrapped.items():
        setattr(st, n, f)
    try:
        yield tr
        if untracked and not st.in_tx:
            tr.finish()
    finally:
        world.appliers.clear()
        world.appliers.update(saved_appliers)
        if saved_derivers or getattr(world, "derivers", None):
            world.derivers.clear()
            world.derivers.update(saved_derivers)
        for n in names:
            if saved[n] is None:
                st.__dict__.pop(n, None)
            else:
                st.__dict__[n] = saved[n]


_GENESIS_TRACES: list = []


@contextlib.contextmanager
def trace_genesis():
    """Worlds created inside the block are traced from their first event: `World._genesis` runs under
    `trace_writes`, and each world's genesis report is appended to the yielded list as (world_id, report). The
    patch is process-local and removed on exit (wave-3b repair: the gate traced only after build, so genesis
    writes - civilisation genesis had 17 offending pairs - were never measured). Genesis is the
    `World._run_system("world.genesis")` transaction, traced with its commit."""
    from ..core.kernel import World
    reports: list = []
    orig = World._run_system

    def run_system(self, label, fn):
        if label != "world.genesis":
            return orig(self, label, fn)
        with trace_writes(self) as tr:          # the whole genesis transaction, its commit included
            r = orig(self, label, fn)
        reports.append((self.world_id, tr))
        return r
    World._run_system = run_system
    try:
        yield reports
    finally:
        World._run_system = orig


# ---------------------------------------------------------------------------------------------------- world kinds
def _torch_one_thread():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001 - torch is optional for worlds without neural participants
        pass


def _civ(seed: int):
    from ..civ.generator import generate
    return generate("balanced", int(seed), "development")


def _population(seed: int):
    from ..core.kernel import World
    from ..population.worlds import population_world
    return World.create(population_world("development", int(seed), n=400, sites=6), world_id=f"pop-{seed}")


def _social(seed: int):
    from ..core.kernel import World
    from ..social.worlds import social_world
    return World.create(social_world("development", int(seed), n=400), world_id=f"social-{seed}")


def _commerce(seed: int):
    from ..commerce.worlds import startup_spec
    from ..core.kernel import World
    return World.create(startup_spec(int(seed), n=600), world_id=f"commerce-{seed}")


def _domain(seed: int):
    from . import modes
    modes.discover()
    for p in modes.providers("B"):
        names = p.names()
        if names:
            return modes._build(p, names[0], int(seed))
    raise LookupError("no Tier 2 domain is registered (mode B lists no environment)")


KINDS: dict[str, Callable[[int], Any]] = {"civ": _civ, "population": _population, "social": _social,
                                          "commerce": _commerce, "domain": _domain}
ENV_KINDS = ("rapid",)


def build(kind: str, seed: int):
    import crucible
    crucible.load_all_modules()
    _torch_one_thread()
    if kind not in KINDS:
        raise KeyError(f"unknown world kind {kind!r} (one of {sorted(KINDS)} or {ENV_KINDS})")
    return KINDS[kind](seed)


def _rapid_hashes(seed: int, steps: int) -> dict:
    """A Tier 1 environment (Mode A) driven by a seeded random policy: a fingerprint of every observation the
    environment returned, every action taken and the evaluator's outcome, and the number of steps it actually ran
    (wave-3b repair: the first version hashed only the final outcome and reported the requested step count)."""
    import hashlib
    from . import modes
    from ..rapid import policies
    modes.discover()
    prov = [p for p in modes.providers("A") if p.name == "rapid"][0]
    fam = prov.names()[0]
    env = modes._build(prov, fam, int(seed))
    h = hashlib.sha256()
    agents = env.agents()
    first = agents[0]
    rng = np.random.default_rng(int(seed))
    obs = {a: env.observe(a) for a in agents}
    h.update(modes._fingerprint(modes._jsonish(obs)).encode())
    ran = 0
    for _ in range(int(steps)):
        if env.done:
            break
        acts = {}
        for a in agents:
            pol = policies.random_policy if a == first else policies.wait
            x = pol(obs[a], env.action_space(a), rng)
            if x is not None:
                acts[a] = x
        h.update(modes._fingerprint(modes._jsonish(acts)).encode())
        r = env.step(acts if len(agents) > 1 else acts.get(first))
        obs = r.get("observations") or {first: r["observation"]}
        h.update(modes._fingerprint(modes._jsonish(obs)).encode())
        ran += 1
    h.update(modes._fingerprint(modes._jsonish(env.outcome())).encode())
    d = h.hexdigest()
    return {"kind": "rapid", "family": fam, "last_hash": d, "digest": d, "events": ran}


def merged_report(world, *traces: "WriteTrace") -> dict:
    """One report over several traces of the same world (genesis and run)."""
    m = WriteTrace(world)
    for t in traces:
        for name in ("writes", "cross", "handler", "columns_refused", "untracked", "events"):
            getattr(m, name).update(getattr(t, name))
    return m.report()


def replay_check(world) -> dict:
    """Whether replaying the world's own log from genesis reproduces its state (wave 3b). The write trace sees every
    write made through the state store's methods; a write that bypasses them (mutating a nested value of a row a read
    returned, or `state.tables` directly) leaves no trace but cannot be reproduced by replay, which re-applies only
    the recorded events: a digest mismatch here means state changed outside the appliers."""
    from ..core.kernel import World
    if world.log_base != 0:
        return {"replay_exact": None, "reason": "the in-memory log was trimmed"}
    rep = World.replay(world.spec, [e.to_dict() for e in world.log], world_id=f"{world.world_id}-replay-check",
                       store=world.store)
    try:
        return {"replay_exact": rep.digest() == world.digest(), "events": len(world.log)}
    finally:
        rep.close()


def hashes(kind: str, seed: int, hours: float, *, trace: bool = False) -> dict:
    """Build a world of `kind` from `seed`, run it `hours` simulated hours and return {kind, last_hash, digest,
    events, modules} (plus the write trace's report when `trace`)."""
    if kind == "rapid":
        return _rapid_hashes(seed, int(max(hours, 1)))
    if trace:
        with trace_genesis() as gen:
            w = build(kind, seed)
    else:
        w = build(kind, seed)
    out: dict = {"kind": kind, "world": w.world_id, "modules": len(w.modules)}
    if trace:
        mine = [t for wid, t in gen if wid == w.world_id]
        if not mine:
            raise RuntimeError(f"genesis of {w.world_id} was not traced")
        with trace_writes(w) as tr:
            w.advance_time(w.now + int(hours * 3600))
        out["genesis_writes"] = mine[-1].report()
        out["run_writes"] = tr.report()
        out["writes"] = merged_report(w, mine[-1], tr)
        out["replay"] = replay_check(w)
    else:
        w.advance_time(w.now + int(hours * 3600))
    out.update({"last_hash": w.last_hash, "digest": w.digest(), "events": int(w.seq),
                "proc_failed": sum(1 for e in w.log if e.type == "proc.failed")})
    w.close()
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 4 and argv[0] in ("hashes", "trace"):
        print(json.dumps(hashes(argv[1], int(argv[2]), float(argv[3]), trace=argv[0] == "trace"), default=str))
        return 0
    print("usage: python -m crucible.substrate.integrity hashes|trace <kind> <seed> <hours>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
