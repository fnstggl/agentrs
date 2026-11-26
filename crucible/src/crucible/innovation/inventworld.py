"""Worlds in which technology invention is live (wave 3b, workstream G): the integrated civilisation of
`crucible.civ` plus the population's autonomous developers (`innovation.devs`), and an ECON creation world with
the same addition for fast unit tests.

`civ_spec(profile, seed, ...)` takes the civilisation's own recipe (`crucible.civ.generator.recipe`: every Mode C
subsystem, the one population and its activities, the incumbent SaaS firms, the capability registry and the
software ecosystem) and changes only what invention needs:

* the population's work gains activities that nobody documents or serves at genesis - procedurally generated task
  categories (`usefulness` families, a fresh hidden rule per world) - and, optionally, a two-step pipeline of
  published formats whose second step is a small share of the hand work (so it becomes the bottleneck only once
  the first is automated: adoption creating a need). These are activities of the world's people, drawn like the
  civilisation's own work families; nothing about any product is configured;
* module `devs` (the population's developers) is added after the modules it uses.

Integration note (CIV): once `crucible.civ.generator.ORDER` lists `devs` (after `pkg`) and the profiles draw
generated work categories, this helper reduces to the work families. Until then every Gate J/L/SP-6.3b-6.5b world
of workstream G is built here, on the civilisation generator's own spec.
"""
from __future__ import annotations

import copy

from ..core.kernel import MODULES, WorldSpec

NOVEL = ("allocate",)
PIPELINE = {"need": "route.shortest", "then": "pack.knapsack", "share": 1.0, "tasks_day": [1.0, 2.5],
            "diy_min": [20.0, 45.0], "then_diy_min": [3.0, 7.0], "size": [0.2, 0.5], "standard": [0.9, 0.99],
            "budget_min": [60.0, 200.0]}


def novel_family(family: str, seed: int, k: int, share: float = 1.0) -> dict:
    return {"category": {"family": family, "seed": int(seed) * 100 + k}, "share": float(share),
            "tasks_day": [1.0, 2.5], "diy_min": [15.0, 45.0], "size": [0.2, 0.5], "standard": [0.9, 0.99],
            "budget_min": [60.0, 200.0]}


def _with_devs(mods: list[dict], devs: dict | None) -> list[dict]:
    import crucible
    crucible.load_all_modules()
    names = [m["name"] for m in mods]
    out = [dict(m) for m in mods]
    for extra in ("issues", "pkg"):
        if extra not in names:
            out.append({"name": extra, "config": {}})
            names.append(extra)
    if "devs" not in names:
        out.append({"name": "devs", "config": dict(devs or {})})
    else:
        for m in out:
            if m["name"] == "devs":
                m["config"] = {**dict(m.get("config") or {}), **dict(devs or {})}
    # dependency order (stable): a module after everything it depends on
    by = {m["name"]: m for m in out}
    order: list[str] = []

    def add(n):
        if n in order or n not in by:
            return
        for d in MODULES[n].depends:
            add(d)
        order.append(n)
    for m in out:
        add(m["name"])
    return [by[n] for n in order]


def civ_spec(profile: str, seed: int, *, population: str = "development", novel: tuple = NOVEL,
             pipeline: bool = True, devs: dict | None = None, overrides: dict | None = None,
             params: dict | None = None) -> WorldSpec:
    from ..civ import generator as G
    ov = copy.deepcopy(overrides or {})
    rec = G.recipe(profile, seed, population=population, overrides=copy.deepcopy(ov))
    fams = [dict(f) for f in rec["modules"]["work"]["families"]]
    for k, fam in enumerate(novel):
        fams.append(novel_family(fam, seed, k))
    if pipeline:
        fams.append(dict(PIPELINE))
    mc = dict(ov.get("module_config") or {})
    mc["work"] = {**dict(mc.get("work") or {}), "families": fams}
    ov["module_config"] = mc
    if params:
        ov["params"] = {**dict(ov.get("params") or {}), **params}
    sp = G.spec(profile, seed, population, overrides=ov)
    return WorldSpec(sp.family, sp.seed, _with_devs(list(sp.modules), devs), dict(sp.params))


def generate(profile: str, seed: int, *, store=None, strict: bool = False, **kw):
    from ..core.kernel import World
    sp = civ_spec(profile, seed, **kw)
    return World.create(sp, world_id=f"civinv-{sp.hash()[:12]}", store=store, strict=strict)


def creation_spec(seed: int, *, n: int = 600, published: tuple = ("schedule.jobs",), novel: tuple = NOVEL,
                  pipeline: bool = True, devs: dict | None = None, params: dict | None = None) -> WorldSpec:
    """The ECON creation world (`innovation.popworld.creation_spec`: startup economy with incumbents, published
    formats and generated categories on the population) with the population's developers (unit tests). Compute
    is priced like the civilisation's providers (`civ.genesis`: a few cents per core-hour; ECON's startup worlds
    price it at 1.2-2.6 dollars, which no individual's project could pay)."""
    from .popworld import creation_spec as cs
    sp = cs(seed, n=n, published=published, novel=novel, novel_share=0.8, params=params)
    mods = [dict(m) for m in sp.modules]
    for m in mods:
        if pipeline and m["name"] == "work":
            m["config"] = dict(m["config"], families=list(m["config"]["families"]) + [dict(PIPELINE)])
        if m["name"] == "ventures":
            provs = [dict(x, price=int(2 + k * 2)) for k, x in enumerate(m["config"].get("providers") or [])]
            m["config"] = dict(m["config"], providers=provs)
    return WorldSpec("creation_devs", int(seed), _with_devs(mods, devs), dict(sp.params))
