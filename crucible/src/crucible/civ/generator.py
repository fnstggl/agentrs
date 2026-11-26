"""The civilisation generator (ONE_WORLD_CONTRACTS section 4, D-17): `recipe`, `spec`, `generate`.

`recipe(profile, seed, population=..., overrides=...)` returns the inspectable 4.0 world-family form - an ordered
module list, each module's config and the world params - plus what a reader needs to check it: the profile's
structural key, the subsystems and their modules, the coupling records, the modules left out and why (`excluded`:
second populations, scripted narrative; `pending`: modules whose population hook is not in this checkout yet).

Everything is composed into ONE kernel world: one identity registry (`identity`), one bank, one population
(`pop` and its activities), one artifact store, one clock and event log. Initial entities are created by the
modules' own genesis and by `civgen` (crucible.civ.genesis) through ordinary APIs; nothing about what happens
afterwards is scripted.

Shared physics is generated once: when a profile binds science to industry, the hidden quality response of an
industrial machine model (drawn exactly as `industry` draws it, from the world seed) is written as the hidden law of
a `sci` phenomenon, and `civlinks.check_invariants` verifies on every check that the two agree.
"""
from __future__ import annotations

import copy

import numpy as np

from ..core.canonical import digest
from ..core.kernel import MODULES, WorldSpec
from . import couplings as C
from . import profiles as P
from . import subsystems as SUB

POPULATION_PROFILES = ("development", "standard", "large", "internet")

# The modules of a civilisation, in the order their genesis must run (dependencies are moved before their users).
ORDER = [
    # substrate, identity, money, the 4.0 internet services
    "identity", "bank", "admin", "artifacts", "cols", "assets", "code", "search", "forum", "docs", "mail", "issues",
    "support", "contracts", "business", "jobs", "ops", "cloud", "net", "appdb", "wf", "caps", "gov", "deal",
    "commit",
    # the one population and the web it uses
    "popsites", "pop", "popweb", "pop_adaptive", "pop_strategic",
    # social internet
    "content", "social_ids", "social_rows", "social", "aud", "feed", "creators", "msg", "studio", "community", "kb",
    # commerce, marketplaces, institutions, labour
    "products", "subs", "ventures", "xchg", "market", "inst", "labor",
    # technology, software ecosystem
    "needs", "pkg",
    # science, industry, energy, supply chains
    "sci", "industry", "freight", "society", "industrygen",
    # computing infrastructure, organisations, education
    "fleet", "firms", "learn",
    # the civilisation's own modules: households' goods, genesis, couplings
    "civgoods", "civgen", "civlinks",
]


def _rng(seed: int, *keys) -> np.random.Generator:
    """The same stream `World.rng_for(*keys)` gives in a world of this seed."""
    h = digest([int(seed), *[str(k) for k in keys]], size=16)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(h, 16))))


def _ensure_registered(names) -> None:
    import crucible
    crucible.load_all_modules()
    missing = [n for n in names if n not in MODULES]
    if missing:
        from ..api.local import ensure_modules
        ensure_modules(missing)


# --------------------------------------------------------------------------------------------- supply chains
def _firm(handle, name, role, op, region, **kw):
    return {"handle": handle, "name": name, "role": role, "op": op, "region": region, **kw}


def chain(kind: str, regions: int) -> list[dict]:
    """Supply-chain graphs (mines -> component makers -> assemblers -> households). Different kinds are different
    graphs, not different numbers."""
    R = max(1, int(regions))
    r = lambda k: k % R  # noqa: E731
    mine = lambda i, mat, reg, **kw: _firm(f"mine{i}", f"Mine {i}", "mining", "extract", r(reg), material=mat,  # noqa
                                           price=70 + 10 * mat, operators=3, rate=700, **kw)
    forge = lambda i, mat, form, reg, **kw: _firm(f"forge{i}", f"{form.capitalize()} Works {i}", "components",  # noqa
                                                  "form", r(reg), material=mat, form=form, product_name=form,
                                                  price=300 + 10 * i, operators=4, **kw)
    assem = lambda i, comps, reg, **kw: _firm(f"assem{i}", f"Assembler {i}", "products", "assemble", r(reg),  # noqa
                                              components=comps, product_name=f"product {i}", price=1500 + 50 * i,
                                              operators=4, **kw)
    if kind == "linear":
        return [mine(1, 0, 1), forge(1, 0, "sheet", 1, machines=2), assem(1, ["forge1", "forge1"], 0)]
    if kind == "standard":
        from ..worlds.open_industry import default_chain
        return default_chain(R)
    if kind == "diamond":
        return [mine(1, 0, 1), mine(2, 1, 2), forge(1, 0, "sheet", 1), forge(2, 1, "rod", 2),
                assem(1, ["forge1", "forge2"], 0), assem(2, ["forge1", "forge2"], 3)]
    if kind == "wide":
        return [mine(1, 0, 1), mine(2, 1, 2), mine(3, 2, 3), forge(1, 0, "sheet", 1, machines=2),
                forge(2, 1, "rod", 2), forge(3, 2, "fiber", 3),
                assem(1, ["forge1", "forge2"], 0), assem(2, ["forge2", "forge3"], 1),
                assem(3, ["forge1", "forge3"], 2), assem(4, ["forge1", "forge2"], 3)]
    raise KeyError(f"unknown supply-chain kind {kind!r}")


def links(kind: str, regions: int, rng) -> list[dict]:
    """Freight network topologies over regions r0..r{R-1}."""
    R = int(regions)
    pairs: list[tuple[int, int]]
    if kind == "ring":
        pairs = [(i, (i + 1) % R) for i in range(R)] if R > 2 else [(0, 1)] if R == 2 else []
    elif kind == "ring_chords":
        pairs = [(i, (i + 1) % R) for i in range(R)] + [(i, (i + 2) % R) for i in range(0, R, 2)]
    elif kind == "line":
        pairs = [(i, i + 1) for i in range(R - 1)]
    elif kind == "star":
        pairs = [(0, i) for i in range(1, R)]
    elif kind == "complete":
        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    else:
        raise KeyError(f"unknown freight topology {kind!r}")
    seen, out = set(), []
    for a, b in pairs:
        a, b = sorted((a, b))
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        low = kind == "line"
        out.append({"a": f"r{a}", "b": f"r{b}", "hours": int(rng.integers(6, 14)) + (6 if low else 0),
                    "cost": int(rng.integers(2, 5)), "capacity": int(rng.integers(150, 300) if low else
                                                                       rng.integers(500, 900))})
    return out


# --------------------------------------------------------------------------------------------- science <-> industry
MODEL_PRIORITY = ("form-A", "form-B", "assemble-A", "assemble-B", "extract-A", "extract-B")


def science(prof: P.Profile, seed: int, ind_cfg: dict) -> tuple[dict, list[dict]]:
    """`sci` config and the phenomenon <-> machine-model bindings. A bound phenomenon's hidden law is the machine
    model's quality response (100 * exp(-d^2 / 2 w^2)) - the same physics `industry` uses."""
    from ..domains import lawlang
    from ..economy.industry import gen_physics
    sc = prof.science
    phys = gen_physics(_rng(seed, "industry-physics"), int(ind_cfg.get("materials", 6)),
                       int(ind_cfg.get("regions", 4)), ind_cfg)
    rng = _rng(seed, "civ-science")
    n, bound = int(sc.get("phenomena", 1)), min(int(sc.get("bound", 0)), int(sc.get("phenomena", 1)))
    phen, apps, binds = [], [], []
    for k in range(n):
        pid = f"ph{k}"
        if k < bound:
            model = MODEL_PRIORITY[k % len(MODEL_PRIORITY)]
            m = phys["models"][model]
            w2 = 2.0 * float(m["width"]) ** 2
            law = (f"100.0 * exp(-((speed - {lawlang.fmt(m['s_opt'])}) ** 2 + (temp - {lawlang.fmt(m['t_opt'])}) ** 2)"
                   f" / {lawlang.fmt(w2)})")
            phen.append({"id": pid, "title": f"Process response of machine model {model}",
                         "doc": f"Quality index (0-100) of parts made on a {model} machine as a function of its "
                                f"operating speed and temperature settings (both 0-1).",
                         "vars": [{"name": "speed", "lo": 0.0, "hi": 1.0}, {"name": "temp", "lo": 0.0, "hi": 1.0}],
                         "response": "quality_index", "unit": "index", "law": law, "noise_rel": 0.0,
                         "standards": [{"id": f"{pid}s1", "at": {"speed": 0.2, "temp": 0.3}},
                                       {"id": f"{pid}s2", "at": {"speed": 0.7, "temp": 0.6}}]})
            binds.append({"phenomenon": pid, "model": model, "speed_var": "speed", "temp_var": "temp"})
            rngs = {"speed": [0.0, 1.0], "temp": [0.0, 1.0]}
        else:
            xs = lawlang.draw_variables(rng, 2)
            fam = sorted(lawlang.LAW_FAMILIES)[k % len(lawlang.LAW_FAMILIES)]
            law = lawlang.LAW_FAMILIES[fam](rng, xs)
            phen.append({"id": pid, "title": f"{xs[0]} and {xs[1]} response", "law": law["law"], "noise_rel": 0.02,
                         "vars": [{"name": xs[0], "lo": 0.0, "hi": 10.0}, {"name": xs[1], "lo": 0.0, "hi": 5.0}],
                         "standards": [{"id": f"{pid}s1", "at": {xs[0]: 2.0, xs[1]: 1.0}},
                                       {"id": f"{pid}s2", "at": {xs[0]: 6.0, xs[1]: 3.0}}]})
            rngs = {xs[0]: [0.0, 10.0], xs[1]: [0.0, 5.0]}
        apps.append({"id": f"app{k}a", "phenomenon": pid, "operator": "instr", "range": rngs, "price_cents": 400,
                     "sd": 1.5 if k < bound else 0.3, "capacity_per_day": 60,
                     "gain": round(float(rng.uniform(0.97, 1.05)), 4), "offset": round(float(rng.normal(0, 0.3)), 4)})
        apps.append({"id": f"app{k}b", "phenomenon": pid, "operator": "instr2", "range": rngs, "price_cents": 650,
                     "sd": 1.0 if k < bound else 0.2, "capacity_per_day": 40})
    cfg = {"phenomena": phen, "apparatus": apps,
           "operators": [{"handle": "instr", "name": "Instrument Works", "cash": 200_000},
                         {"handle": "instr2", "name": "Precision Labs", "cash": 200_000}],
           "rivals": {"n": int(sc.get("rivals", 0))}, "replicators": {"n": int(sc.get("replicators", 0))}}
    return cfg, binds


# --------------------------------------------------------------------------------------------- recipe
def recipe(profile: str | P.Profile, seed: int, *, population: str = "development",
           overrides: dict | None = None) -> dict:
    """The 4.0 world-family form of a civilisation (see the module docstring)."""
    prof = profile if isinstance(profile, P.Profile) else P.get(profile)
    if population not in POPULATION_PROFILES:
        raise ValueError(f"population must be one of {POPULATION_PROFILES}")
    ov = copy.deepcopy(overrides or {})
    seed = int(seed)
    rng = _rng(seed, "civ-recipe", prof.name)
    ready = SUB.readiness()
    R = int(prof.industry.get("regions", 4))
    lo, hi = P.ENERGY_REGIMES[prof.energy.get("regime", "normal")]
    ind_cfg = {"regions": R, "materials": int(prof.industry.get("materials", 6))}
    sci_cfg, binds = science(prof, seed, ind_cfg)
    cpl = [c for c in prof.couplings if c in C.CATALOGUE]
    boards = {"topical": None, "global": ["everything"],
              "regional": [f"region-{i}" for i in range(R)] + ["market"]}[prof.social.get("boards", "topical")]
    cfgs: dict[str, dict] = {
        "pop": {"profile": population, "tick_s": 900},
        "popsites": {"n": int(prof.digital.get("sites", 20)), "shop_share": float(prof.digital.get("shop_share", 0.4))},
        "forum": ({"boards": boards} if boards else {}),
        "industry": ind_cfg,
        "freight": {"links": links(prof.industry.get("links", "ring"), R, rng)},
        "industrygen": {"chain": chain(prof.industry.get("chain", "standard"), R),
                        "grid_kwh": [int(rng.integers(lo, hi)) for _ in range(R)], "kwh_price": 10.0,
                        "grid_expand": bool(prof.energy.get("expand", True))},
        "sci": sci_cfg,
        "learn": {"n": 0},
        "labor": {"workers": 0},
        "civgoods": {"placement": prof.social.get("placement", "mixed")},
        "civlinks": {"couplings": cpl, "bindings": binds},
        "civgen": {"profile": prof.to_dict(), "regions": R,
                   "compute": {"providers": P.COMPUTE_MARKETS[prof.compute.get("market", "duopoly")],
                               "cores": P.COMPUTE_REGIMES[prof.compute.get("regime", "normal")],
                               "datacenters": prof.compute.get("datacenters", "spread")}},
    }
    excluded = dict(SUB.EXCLUDED)
    pending = []
    mods = []
    for m in ORDER:
        if m in excluded:
            continue
        if m in ready and not ready[m]["ready"]:
            pending.append({"module": m, "subsystem": SUB.subsystem_of_module(m), "waits_for": ready[m]["waits_for"]})
            continue
        mods.append(m)
    for m in ov.pop("drop_modules", []):
        if m in mods:
            mods.remove(m)
    for m, c in (ov.pop("module_config", {}) or {}).items():
        cfgs[m] = {**cfgs.get(m, {}), **c}
    # sandbox_max_workers: one warm worker per hosted site (the default pool of 24 is smaller than a civilisation's
    # web, and restarting evicted workers dominated the cost of a simulated day: 26 s -> 9-13 s per day)
    params = {"cost_model": "deterministic", "civ_profile": prof.name, "civ_population": population,
              "log_retain": 300_000, "sandbox_max_workers": 64, **ov.pop("params", {})}
    if ov:
        raise ValueError(f"unknown overrides {sorted(ov)} (drop_modules, module_config, params)")
    subs = {s: [m for m in d["modules"] if m in mods] for s, d in SUB.SUBSYSTEMS.items()}
    return {"family": f"civ_{prof.name}", "profile": prof.name, "seed": seed, "population": population,
            "structure": P.structural_key(prof), "module_list": mods,
            "modules": {m: copy.deepcopy(cfgs.get(m, {})) for m in mods}, "params": params,
            "subsystems": subs, "couplings": [C.CATALOGUE[c].record() for c in cpl],
            "excluded": excluded, "pending": pending}


def spec(profile: str | P.Profile, seed: int, population: str = "development", *,
         overrides: dict | None = None) -> WorldSpec:
    """The WorldSpec of a civilisation (modules in dependency order)."""
    r = recipe(profile, seed, population=population, overrides=overrides)
    names = r["module_list"]
    _ensure_registered(names)
    order: list[str] = []

    def add(m):
        if m in order or m not in names:
            return
        for d in MODULES[m].depends:
            add(d)
        order.append(m)
    for m in names:
        add(m)
    missing = sorted({d for m in order for d in MODULES[m].depends if d not in order})
    if missing:
        raise ValueError(f"civilisation modules miss dependencies {missing}")
    return WorldSpec(r["family"], int(seed), [{"name": m, "config": r["modules"][m]} for m in order], r["params"])


def generate(profile: str | P.Profile, seed: int, population: str = "development", *, overrides: dict | None = None,
             store=None, world_id: str | None = None, strict: bool = False):
    """Create the civilisation world (genesis run, clock at 0)."""
    from ..core.kernel import World
    sp = spec(profile, seed, population, overrides=overrides)
    return World.create(sp, world_id=world_id or f"civ-{sp.hash()[:12]}", store=store, strict=strict)
