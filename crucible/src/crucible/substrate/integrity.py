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
  - kernel metadata (`meta`) to the kernel.

  Approved interfaces, the only ways a module changes state it does not own: a `cols.batch` / `cols.grow` event of
  the columns module whose recorded writer (`by`, with its interface) the column's ownership rule admits
  (`columns.may_write`: the owner, its package, a granted module, an interface); counter allocation by handlers
  (`ctr:` keys, recorded by the kernel's `ctr` event at commit) and the kernel's side-event count. Everything else
  is a *cross-module state write* (an applier of one module writing another module's table, array, column or
  metadata) or a *handler write* (state changed outside any applier, so replay could not reproduce it). Package
  siblings are reported apart (`same_package`) and counted as cross-module: the requirement is per module.

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
        self.events: Counter = Counter()             # applied events by module
        self._stack: list[tuple[str, Any]] = []
        self._first_writer: dict[str, str] = {}

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
        self._record("meta", key, (KERNEL,))

    def report(self) -> dict:
        def rows(c: Counter) -> list[dict]:
            return [{"writer": w, "kind": k, "target": t, "owner": o, "writes": n}
                    for (w, k, t, o), n in sorted(c.items())]
        same_pkg = Counter({k: n for k, n in self.cross.items()
                            if _package(k[0]) is not None and _package(k[0]) in {_package(o) for o in k[3].split(",")}})
        return {"cross_module_state_writes": len(self.cross) + len(self.handler) + len(self.columns_refused),
                "cross_module_write_count": int(sum(self.cross.values())),
                "handler_writes": int(sum(self.handler.values())),
                "column_writes_refused_by_rule": int(sum(self.columns_refused.values())),
                "writes_traced": int(sum(self.writes.values())),
                "writers": len({k[0] for k in self.writes}),
                "targets": len({k[2] for k in self.writes}),
                "events_applied": int(sum(self.events.values())),
                "cross": rows(self.cross), "handler": rows(self.handler), "refused": rows(self.columns_refused),
                "same_package": rows(same_pkg)}


@contextlib.contextmanager
def trace_writes(world):
    """Trace every state write of `world` while the block runs (see the module docstring). Yields a WriteTrace."""
    tr = WriteTrace(world)
    st = world.state
    saved_appliers = dict(world.appliers)
    names = ("put", "delete", "meta_set", "array_mut", "array_rows_mut", "set_array")
    saved = {n: st.__dict__.get(n) for n in names}
    orig = {n: getattr(st, n) for n in names}

    def wrap_applier(etype, fn):
        mod = _module_of_applier(fn)

        def applier(ev):
            tr._stack.append((mod, ev))
            tr.events[mod] += 1
            try:
                return fn(ev)
            finally:
                tr._stack.pop()
        return applier

    for et, fn in saved_appliers.items():
        world.appliers[et] = wrap_applier(et, fn)

    def put(table, rid, record):
        tr.table(table)
        return orig["put"](table, rid, record)

    def delete(table, rid):
        if rid in st.tables.get(table, {}):
            tr.table(table)
        return orig["delete"](table, rid)

    def meta_set(key, value):
        tr.meta(key)
        return orig["meta_set"](key, value)

    def array_mut(name):
        tr.array(name)
        return orig["array_mut"](name)

    def array_rows_mut(name, rows):
        tr.array(name)
        return orig["array_rows_mut"](name, rows)

    def set_array(name, value):
        tr.array(name)
        return orig["set_array"](name, value)

    for n, f in (("put", put), ("delete", delete), ("meta_set", meta_set), ("array_mut", array_mut),
                 ("array_rows_mut", array_rows_mut), ("set_array", set_array)):
        setattr(st, n, f)
    try:
        yield tr
    finally:
        world.appliers.clear()
        world.appliers.update(saved_appliers)
        for n in names:
            if saved[n] is None:
                st.__dict__.pop(n, None)
            else:
                st.__dict__[n] = saved[n]


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
    """A Tier 1 environment (Mode A) driven by a seeded random policy: fingerprint of its observations."""
    import hashlib
    from . import modes
    from ..rapid import policies
    modes.discover()
    prov = [p for p in modes.providers("A") if p.name == "rapid"][0]
    fam = prov.names()[0]
    env = modes._build(prov, fam, int(seed))
    out = policies.run(env, policies.random_policy, np.random.default_rng(int(seed)), ticks=int(steps))
    h = hashlib.sha256(modes._fingerprint(modes._jsonish(out)).encode()).hexdigest()
    return {"kind": "rapid", "family": fam, "last_hash": h, "digest": h, "events": int(steps)}


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
    w = build(kind, seed)
    out: dict = {"kind": kind, "world": w.world_id, "modules": len(w.modules)}
    if trace:
        with trace_writes(w) as tr:
            w.advance_time(w.now + int(hours * 3600))
        out["writes"] = tr.report()
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
