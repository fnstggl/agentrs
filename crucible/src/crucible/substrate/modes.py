"""The three execution modes of Crucible 5.0 (MASTER_SPEC Part 2.1; docs/v5/ARCHITECTURE.md section 3).

    A  controlled experimental environments   small, seeded, CPU, batched      crucible.rapid (+ 4.0 mechanisms)
    B  standalone semantic digital worlds     one Tier 2 domain, real tools    crucible.domains
    C  integrated persistent civilizations    every domain in one kernel world crucible.civ

Not to be confused with 4.0's `crucible.modes` (synthetic / offline grounding / supervised real world).

A tier package makes its environments available by registering a *provider* (in its `modules.py`, which
`crucible.load_all_modules()` imports):

    from crucible.substrate import modes
    modes.register("B", "domains", package="crucible.domains", names=lambda: [...], make=make_world,
                   kind="world", doc="...")

`make(name, seed, **kw)` returns an environment of that mode: for `kind="world"` a `crucible.core.kernel.World`
(or a `WorldSpec`, which the registry creates); for `kind="env"` an object with `reset()` and `step(action)`; for
`kind="sim"` a 4.0 `MechSim`-like object with `step(settings)`.

A mode is *operational* only when at least one of its providers passes its check, run for real by `status()`:
the provider lists at least one environment, constructs it twice from the same seed, runs it, and gets identical
results (and, for kernel worlds, exact replay of the log). Nothing is assumed from a registration alone, and modes
without a provider are reported with the reason.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.errors import ProtocolError

MODES: dict[str, dict] = {
    "A": {"title": "controlled experimental environments", "package": "crucible.rapid", "tier": 1,
          "spec": "MASTER_SPEC Part 2.1 Mode A; Part 8", "runs_on": "CPU, vectorised, no kernel needed"},
    "B": {"title": "standalone semantic digital worlds", "package": "crucible.domains", "tier": 2,
          "spec": "MASTER_SPEC Part 2.1 Mode B; Part 9", "runs_on": "a kernel world built from one domain"},
    "C": {"title": "integrated persistent civilizations", "package": "crucible.civ", "tier": 3,
          "spec": "MASTER_SPEC Part 2.1 Mode C; Part 10", "runs_on": "one kernel world with every subsystem"},
}
KINDS = ("world", "env", "sim")
TIER_PACKAGES = ("rapid", "domains", "civ")


@dataclass
class Provider:
    mode: str
    name: str
    package: str
    names: Callable[[], list[str]]
    make: Callable[..., Any]
    kind: str = "world"
    doc: str = ""
    check: Callable[["Provider"], dict] | None = None      # custom check; default by kind
    meta: dict = field(default_factory=dict)


_PROVIDERS: dict[tuple[str, str], Provider] = {}


def register(mode: str, name: str, *, package: str, names: Callable[[], list[str]], make: Callable[..., Any],
             kind: str = "world", doc: str = "", check: Callable[[Provider], dict] | None = None, **meta) -> None:
    """Register (or replace) provider `name` of execution mode `mode`."""
    if mode not in MODES:
        raise ProtocolError(f"unknown execution mode {mode!r} (A, B or C)")
    if kind not in KINDS:
        raise ProtocolError(f"provider kind must be one of {KINDS}")
    _PROVIDERS[(mode, name)] = Provider(mode, name, package, names, make, kind, doc, check, dict(meta))


def unregister(mode: str, name: str) -> None:
    _PROVIDERS.pop((mode, name), None)


def discover() -> None:
    """Import every package that may register providers: the v5 tier packages and their `modules.py`
    (crucible.load_all_modules) plus the built-in 4.0 provider."""
    import crucible
    crucible.load_all_modules()
    for pkg in TIER_PACKAGES:
        importlib.import_module(f"crucible.{pkg}")
        if importlib.util.find_spec(f"crucible.{pkg}.modes") is not None:
            importlib.import_module(f"crucible.{pkg}.modes")
    _register_builtin()


def providers(mode: str | None = None) -> list[Provider]:
    return [p for (m, _), p in sorted(_PROVIDERS.items()) if mode is None or m == mode]


def make(mode: str, name: str, seed: int = 0, **kw) -> Any:
    """Build environment `name` of `mode` from the first provider that lists it."""
    for p in providers(mode):
        if name in p.names():
            return _build(p, name, seed, **kw)
    raise ProtocolError(f"no provider of mode {mode} offers {name!r}")


def _build(p: Provider, name: str, seed: int, **kw) -> Any:
    out = p.make(name, seed, **kw)
    if p.kind == "world":
        from ..core.kernel import World, WorldSpec
        if isinstance(out, WorldSpec):
            out = World.create(out, world_id=f"mode{p.mode}-{name}-{seed}")
    return out


# ---------------------------------------------------------------------------------------------------- checks
def _fingerprint(x: Any) -> str:
    h = hashlib.sha256()

    def walk(v):
        if isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode() + str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                walk(v[k])
        elif isinstance(v, (list, tuple)):
            for y in v:
                walk(y)
        else:
            h.update(repr(v).encode())
    walk(x)
    return h.hexdigest()


def _check_sim(p: Provider, name: str, seed: int, ticks: int = 24) -> dict:
    runs = []
    for _ in range(2):
        sim = _build(p, name, seed)
        out = sim.run(None, ticks)
        runs.append(_fingerprint({"revenue": out["revenue"], "cost": out["cost"], "score": out["score"]}))
    # the environment responds to actions: some lever setting changes the outcome
    sim = _build(p, name, seed)
    base = _fingerprint(sim.run(None, ticks)["net"])
    responsive = False
    eng = sim.engine
    for lv in eng.levers(0):
        for v in (lv.lo, lv.hi):
            s2 = _build(p, name, seed)
            if _fingerprint(s2.run({lv.name: v}, ticks)["net"]) != base:
                responsive = True
                break
        if responsive:
            break
    return {"deterministic": runs[0] == runs[1], "responsive": responsive, "ticks": ticks,
            "ok": runs[0] == runs[1] and responsive}


def _check_env(p: Provider, name: str, seed: int, steps: int = 8) -> dict:
    fps = []
    for _ in range(2):
        env = _build(p, name, seed)
        obs = [env.reset()]
        for _k in range(steps):
            obs.append(env.step({}))
        fps.append(_fingerprint(_jsonish(obs)))
    return {"deterministic": fps[0] == fps[1], "steps": steps, "ok": fps[0] == fps[1]}


def _jsonish(x):
    from ..core.canonical import to_jsonable
    try:
        return to_jsonable(x)
    except TypeError:
        return repr(x)


def _check_world(p: Provider, name: str, seed: int, days: int = 1) -> dict:
    from ..core.kernel import World
    hashes, digests = [], []
    w = None
    for _ in range(2):
        w = _build(p, name, seed)
        w.advance_time(w.now + days * 86400)
        hashes.append(w.last_hash)
        digests.append(w.digest())
    evs = [e.to_dict() for e in w.log] if w.log_base == 0 else None
    replay_ok = None
    if evs is not None:
        replay_ok = World.replay(w.spec, evs, world_id="mode-check-replay").digest() == w.digest()
    ok = hashes[0] == hashes[1] and digests[0] == digests[1] and replay_ok is not False
    return {"deterministic": hashes[0] == hashes[1], "replay_exact": replay_ok, "events": w.seq,
            "modules": len(w.modules), "ok": ok}


def check(p: Provider) -> dict:
    """Run provider `p` for real; {ok, n_envs, name, ..., error?}."""
    t0 = time.perf_counter()
    try:
        names = list(p.names())
        if not names:
            return {"ok": False, "n_envs": 0, "error": "provider lists no environments"}
        name = names[0]
        if p.check is not None:
            res = dict(p.check(p))
        elif p.kind == "sim":
            res = _check_sim(p, name, seed=0)
        elif p.kind == "env":
            res = _check_env(p, name, seed=0)
        else:
            res = _check_world(p, name, seed=0)
        res.setdefault("name", name)
        res["n_envs"] = len(names)
    except Exception as e:  # noqa: BLE001 - a provider that cannot run is not operational; say why
        res = {"ok": False, "n_envs": 0, "error": f"{type(e).__name__}: {e}"[:400]}
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    return res


def status(run_checks: bool = True, *, discover_first: bool = True) -> dict:
    """{mode: {title, package, operational, providers: [{name, package, kind, check}], reason}} and the count."""
    if discover_first:
        discover()
    out = {}
    for m, info in MODES.items():
        ps = providers(m)
        rows = []
        for p in ps:
            row = {"name": p.name, "package": p.package, "kind": p.kind, "doc": p.doc}
            if run_checks:
                row["check"] = check(p)
            rows.append(row)
        operational = bool(run_checks and any(r["check"]["ok"] for r in rows))
        reason = "" if operational else (
            f"no provider registered (expected from {info['package']})" if not rows else
            "no provider passed its check" if run_checks else "checks not run")
        out[m] = dict(info, operational=operational, providers=rows, reason=reason)
    return {"modes": out, "operational": [m for m in MODES if out[m]["operational"]],
            "execution_modes": sum(1 for m in MODES if out[m]["operational"])}


def operational_modes() -> list[str]:
    return status()["operational"]


# ---------------------------------------------------------------------------------------------------- built-in
def _register_builtin() -> None:
    """Mode A, 4.0: the hidden-mechanism families (crucible.mechanisms) run kernel-free by MechSim. They are small,
    seeded, CPU-only, scientifically controlled environments (levers in, gauges out, hidden causal structure, oracle
    and validity checks evaluator-side): ARCHITECTURE.md lists them under Mode A next to crucible.rapid."""
    if ("A", "mechanisms-4.0") in _PROVIDERS:
        return

    def names():
        from ..mechanisms import load_engines
        from ..mechanisms.base import mech_families
        load_engines()
        return [f for split in ("train", "val", "test") for f in mech_families(split=split)]

    def make_sim(name, seed, **kw):
        from ..mechanisms import load_engines
        from ..mechanisms.base import make_mechanism
        from ..mechanisms.sim import MechSim
        load_engines()
        return MechSim(make_mechanism(name, int(seed), kw.get("overrides")), int(seed))

    register("A", "mechanisms-4.0", package="crucible.mechanisms", names=names, make=make_sim, kind="sim",
             doc="Crucible 4.0 hidden-mechanism families (scm, abm, techsys engines) as kernel-free MechSim "
                 "environments; train/val/test splits as in 4.0")
