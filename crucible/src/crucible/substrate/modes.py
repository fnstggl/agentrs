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

Mode-specific checks (5.0, D-21):

* Mode A (`crucible.rapid`, registered here): one family per engine, each built twice from a seed and driven by the
  same seeded random policy over its public action space - identical outcomes - and responsive (some policy changes
  the evaluator's outcome).
* Mode C: a world counts as an integrated civilization only if it really composes shared state (`composition`):
  besides determinism and exact replay, the mandatory duplicate-fact check passes; at least `C_MIN_DOMAINS` of the
  17 Part 2.1 domains are *engaged* - another module or a population individual caused events in them after
  genesis (seeding its own tables at genesis does not count); the one population reaches at least
  `C_MIN_POP_DOMAINS` domains (D-15); at least `C_MIN_CROSS_KINDS` shared kinds - including the population and
  money - are written by transactions another module initiated (cross-module events touching shared owners); and at
  least `C_MIN_PAIRS` distinct (initiator, owner module) pairs occur. Loading every module into one world without
  interaction (audit J2) fails it.
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
    days = float(p.meta.get("check_days", days))
    for _ in range(2):
        w = _build(p, name, seed)
        w.advance_time(w.now + int(days * 86400))
        hashes.append(w.last_hash)
        digests.append(w.digest())
    evs = [e.to_dict() for e in w.log] if w.log_base == 0 else None
    replay_ok = None
    if evs is not None:
        replay_ok = World.replay(w.spec, evs, world_id="mode-check-replay").digest() == w.digest()
    ok = hashes[0] == hashes[1] and digests[0] == digests[1] and replay_ok is not False
    out = {"deterministic": hashes[0] == hashes[1], "replay_exact": replay_ok, "events": w.seq,
           "modules": len(w.modules), "ok": ok}
    if p.mode == "C":
        comp = composition(w)
        out["composition"] = comp
        out["integrated"] = comp["integrated"]
        out["ok"] = ok and comp["integrated"]
    return out


# ---------------------------------------------------------------------------------------------------- Mode C
#: MASTER_SPEC Part 2.1 Mode C: the major domains, each with the kernel modules that implement it
DOMAINS: dict[str, tuple[str, ...]] = {
    "digital_economy": ("products", "subs", "market", "ventures", "business", "shoppers", "consumers", "patrons",
                        "ads", "firms"),
    "social_media": ("social", "content", "feed", "aud", "creators", "letters", "studio", "posts", "shorts", "tube",
                     "pics", "pods", "blog", "msg"),
    "search_information": ("search", "net", "popweb", "webgen", "popsites"),
    "scientific_research": ("sci", "lab", "rig"),
    "software_development": ("code", "pkg", "issues", "dev"),
    "technology_invention": ("caps", "needs", "use", "adopt", "launch"),
    "industry": ("industry", "industrygen"),
    "supply_chains": ("freight",),
    "energy": ("event:make.grid", "fleet"),
    "computing_infrastructure": ("cloud", "ops", "infra", "mechinfra", "fleet"),
    "employment": ("labor", "jobs"),
    "education": ("learn", "kb"),
    "finance": ("bank", "contracts", "commit", "npcs"),
    "communities": ("community", "forum"),
    "organizations": ("identity", "inst", "firms"),
    "institutions": ("gov", "inst"),
    "marketplaces": ("xchg", "market", "deal"),
}
C_MIN_DOMAINS = 12
C_MIN_POP_DOMAINS = 6
C_MIN_CROSS_KINDS = 8
C_MIN_PAIRS = 12
_TX_START = ("cmd", "cmd.rejected", "proc", "time", "note")


def _module_of_event(world, etype: str) -> str | None:
    fn = world.appliers.get(etype)
    mod = getattr(fn, "__self__", None)
    return getattr(mod, "name", None) if mod is not None and hasattr(mod, "config") else None


POP_MODULES = ("pop", "popweb", "pop_adaptive", "pop_strategic")


def composition(world, events=None) -> dict:
    """How much a world composes shared state across modules, measured from its event log (default: the in-memory
    log; it must start at genesis). The genesis transaction (world construction seeding its own tables) never
    counts. An event *engages* the domain of the module that applies it when another module, or a population
    individual, initiated its transaction (or a subscriber of another module emitted it). See the module docstring
    for the Mode C thresholds."""
    from . import columns, facts, ids, primitives
    evs = list(world.log if events is None else events)
    host_mod = {m.host or m.name: m.name for m in world.modules.values()}
    owner_kinds: dict[str, list[str]] = {}
    for kind, o in primitives.OWNERS.items():
        if o["module"] != primitives.KERNEL:
            owner_kinds.setdefault(o["module"], []).append(kind)
    col_kind = {o["table"][5:]: k for k, o in primitives.OWNERS.items() if o["table"].startswith("cols:")}
    col_mod = {}
    for t, rec in columns._registry(world).items():
        by = rec.get("by", {})
        if by:
            col_mod[t] = sorted(by.values(), key=lambda m: list(by.values()).index(m))[0]
    ev_kind = {t: k for t, (k, _m, _s) in primitives.PRIMITIVE_EVENTS.items()}
    dom_of: dict[str, list[str]] = {}
    for d, mods in DOMAINS.items():
        for m in mods:
            dom_of.setdefault(m, []).append(d)
    engaged: dict[str, set] = {}
    pop_reach: set[str] = set()
    cross: dict[str, int] = {}
    pairs: set[tuple[str, str]] = set()
    initiator, actor, genesis = None, None, True
    for e in evs:
        if e.type in _TX_START and e.via is None:
            genesis = e.type == "note" and e.payload.get("what") == "genesis"
            actor = e.actor
            if e.type == "proc":
                initiator = e.payload.get("proc", "").split(".", 1)[0]
            elif e.type in ("cmd", "cmd.rejected") and isinstance(e.payload.get("tool"), str):
                h = e.payload["tool"].split(".", 1)[0]
                initiator = host_mod.get(h, h)
            else:
                initiator = "kernel"
        if genesis:
            continue
        if e.type in ("cols.batch", "cols.grow"):
            table = e.payload.get("table")
            mod = col_mod.get(table)
            kinds = [col_kind[table]] if table in col_kind else []
            owner = primitives.OWNERS[kinds[0]]["module"] if kinds else None
        else:
            mod = _module_of_event(world, e.type)
            kinds = [ev_kind[e.type]] if e.type in ev_kind else owner_kinds.get(mod, [])
            owner = mod if kinds else None
        if mod is None:
            continue
        # who caused it: the transaction's initiator and, for a subscriber's event, the subscribing module
        srcs = {x for x in (initiator, e.via) if x and x != "kernel"}
        by_person = ids.index(actor or "") is not None
        src_label = "population" if by_person or srcs & set(POP_MODULES) else (sorted(srcs - {mod}) or [None])[0]
        if srcs - {mod} or by_person:
            for d in dom_of.get(mod, []) + [d for k, ds in dom_of.items() if k.startswith("event:")
                                             and ("event:" + e.type).startswith(k) for d in ds]:
                engaged.setdefault(d, set()).add(src_label)
                if src_label == "population":
                    pop_reach.add(d)
        if owner and (srcs - {owner} or by_person):
            for k in kinds:
                cross[k] = cross.get(k, 0) + 1
            pairs.add((src_label, owner))
    fact_problems = facts.world_problems(world)
    ok = {"domains": len(engaged) >= C_MIN_DOMAINS, "population_reach": len(pop_reach) >= C_MIN_POP_DOMAINS,
          "cross_kinds": len(cross) >= C_MIN_CROSS_KINDS and {"population", "money"} <= set(cross),
          "pairs": len(pairs) >= C_MIN_PAIRS, "facts": not fact_problems}
    return {"domains_engaged": sorted(engaged), "domains_total": len(DOMAINS),
            "domains_not_engaged": sorted(set(DOMAINS) - set(engaged)),
            "population_domains": sorted(pop_reach),
            "cross_module_kinds": dict(sorted(cross.items())), "cross_module_pairs": len(pairs),
            "pairs_sample": sorted(f"{a}->{b}" for a, b in pairs)[:40], "fact_problems": fact_problems[:5],
            "criteria": ok,
            "thresholds": {"domains": C_MIN_DOMAINS, "population_domains": C_MIN_POP_DOMAINS,
                           "cross_kinds": C_MIN_CROSS_KINDS, "pairs": C_MIN_PAIRS},
            "integrated": all(ok.values())}


def check(p: Provider) -> dict:
    """Run provider `p` for real; {ok, n_envs, name, ..., error?}."""
    t0 = time.perf_counter()
    try:
        names = list(p.names())
        if not names:
            return {"ok": False, "n_envs": 0, "error": "provider lists no environments",
                    "wall_s": round(time.perf_counter() - t0, 3)}
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
def _rapid_names() -> list[str]:
    from .. import rapid
    return rapid.families.families(split="train")


def _rapid_make(name, seed, **kw):
    from .. import rapid
    return rapid.make(name, int(seed), **kw)


def _rapid_check(p: Provider, ticks: int = 24) -> dict:
    """One training family per engine: built twice from a seed and driven by the same seeded random policy over its
    public action space - identical outcomes; responsive when some policy changes the evaluator's outcome."""
    from .. import rapid
    from ..rapid import policies
    fams = {}
    for f in p.names():
        fams.setdefault(rapid.FAMILIES[f].engine, f)
    rows = {}
    for eng, fam in sorted(fams.items()):
        outs = []
        for _ in range(2):
            env = _rapid_make(fam, 3)
            outs.append(_fingerprint(_jsonish(policies.run(env, policies.random_policy, np.random.default_rng(7),
                                                           ticks=ticks))))
        alt = {_fingerprint(_jsonish(policies.run(_rapid_make(fam, 3), pol, np.random.default_rng(s), ticks=ticks)))
               for pol, s in ((policies.wait, 0), (policies.random_policy, 11))}
        rows[fam] = {"engine": eng, "deterministic": outs[0] == outs[1], "responsive": bool(alt - {outs[0]})}
    det = all(r["deterministic"] for r in rows.values())
    resp = sum(1 for r in rows.values() if r["responsive"])
    return {"deterministic": det, "responsive": resp >= max(1, len(rows) // 2), "engines": len(rows),
            "responsive_engines": resp, "families_checked": rows, "ticks": ticks,
            "ok": det and resp >= max(1, len(rows) // 2) and len(rows) >= 3}


def _register_builtin() -> None:
    """Mode A, 4.0: the hidden-mechanism families (crucible.mechanisms) run kernel-free by MechSim. They are small,
    seeded, CPU-only, scientifically controlled environments (levers in, gauges out, hidden causal structure, oracle
    and validity checks evaluator-side): ARCHITECTURE.md lists them under Mode A next to crucible.rapid.

    Mode A, 5.0: the Tier 1 rapid families (crucible.rapid, training split) as RapidEnv environments, unless the rapid
    package registers its own provider under the same name."""
    if ("A", "rapid") not in _PROVIDERS:
        register("A", "rapid", package="crucible.rapid", names=_rapid_names, make=_rapid_make, kind="env",
                 check=_rapid_check,
                 doc="Tier 1 rapid scientific environments (crucible.rapid, training families) as RapidEnv: hidden "
                     "causal systems over nine engines, general action interface, evaluator-side outcomes")
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
