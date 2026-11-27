"""Worlds in which technology invention is live (wave 3b, workstream G): the integrated civilisation of
`crucible.civ` plus the population's autonomous developers (`innovation.devs`), and an ECON creation world with
the same addition for fast unit tests.

`civ_spec(profile, seed, ...)` takes the civilisation's own recipe (`crucible.civ.generator.recipe`: every Mode C
subsystem, the one population and its activities, the incumbent SaaS firms, the capability registry and the
software ecosystem) and changes only what invention needs:

* the population's work gains activities that nobody documents or serves at genesis - procedurally generated task
  categories (`usefulness` families, a fresh hidden rule per world) -, composed activities from a generic world
  rule (`raw_feed_compositions`: people whose activity consumes a table get part of their tables as raw rows of
  another activity's wire format; no pair is named) and, optionally, a two-step pipeline of published formats whose
  second step is a small share of the hand work (so it becomes the bottleneck only once the first is automated:
  adoption creating a need). The pipeline IS configured by this builder (`PIPELINE`); the evaluator reports needs
  derived from it separately from the SP-6.4b mechanism count. These are activities of the world's people; nothing
  about any product, provider or combination of services is configured;
* module `devs` (the population's developers) is added after the modules it uses.

Integration note (CIV): once `crucible.civ.generator.ORDER` lists `devs` (after `pkg`) and the profiles draw
generated work categories, this helper reduces to the work families. Until then every Gate J/L/SP-6.3b-6.5b world
of workstream G is built here, on the civilisation generator's own spec.
"""
from __future__ import annotations

import copy

from ..core.kernel import MODULES, WorldSpec

NOVEL = ("allocate", "transform")
RAW_FEED_SHARE = 0.8      # world rule: share of a table-consuming activity's people whose tables arrive as raw feeds
PIPELINE = {"need": "route.shortest", "then": "pack.knapsack", "share": 1.0, "tasks_day": [1.0, 2.5],
            "diy_min": [20.0, 45.0], "then_diy_min": [3.0, 7.0], "size": [0.2, 0.5], "standard": [0.9, 0.99],
            "budget_min": [60.0, 200.0]}


def novel_family(family: str, seed: int, k: int, share: float = 1.0) -> dict:
    return {"category": {"family": family, "seed": int(seed) * 100 + k}, "share": float(share),
            "tasks_day": [1.0, 2.5], "diy_min": [15.0, 45.0], "size": [0.2, 0.5], "standard": [0.9, 0.99],
            "budget_min": [60.0, 200.0]}


def raw_feed_compositions(novel: tuple, seed: int) -> list[tuple[str, str]]:
    """The world rule that makes composed activities (no pair is configured by name): people whose activity takes a
    table of records (a generated category with a table input, `table_key`) receive, for a share of their work, that
    table as raw rows in the wire format of another of the world's activities that produces tables (`render_table`:
    the data arrive from partner feeds), so their task is "normalise, then do the work on the records"
    (`usefulness.category.compose`). Every ordered (producer, consumer) pair of the world's generated categories gets
    it; which pairs exist follows from which categories the world has."""
    from ..usefulness import category as C
    from ..usefulness import families as _fams  # noqa: F401  (registers the families)
    cats = {fam: C.instantiate(fam, int(seed) * 100 + k) for k, fam in enumerate(novel)}
    return [(a, b) for a in novel for b in novel
            if a != b and cats[a].render_table is not None and cats[b].table_key is not None]


def rule_compositions(families: list[dict]) -> set[tuple]:
    """The composed activities the raw-feed rule gives for a list of work families (their generated categories):
    {((family, seed), (family, seed))} (evaluator side: `gates.designer_supplied_demand`)."""
    from ..usefulness import category as C
    from ..usefulness import families as _fams  # noqa: F401  (registers the families)
    gens = [(f["category"]["family"], int(f["category"]["seed"])) for f in families if f.get("category")]
    cats = {g: C.instantiate(g[0], g[1]) for g in gens}
    return {(a, b) for a in gens for b in gens if a != b and cats[a].render_table is not None
            and cats[b].table_key is not None}


def composed_family(pair: tuple, novel: tuple, seed: int, share: float = RAW_FEED_SHARE) -> dict:
    """A work activity whose task is a composition of two of the world's generated categories (the same hidden
    rules as the single-stage activities: the same family and seed)."""
    def ref(fam):
        k = list(novel).index(fam)
        return {"family": fam, "seed": int(seed) * 100 + k}
    return {"compose": [ref(pair[0]), ref(pair[1])], "share": float(share), "tasks_day": [1.0, 2.0],
            "diy_min": [30.0, 60.0], "size": [0.2, 0.4], "standard": [0.9, 0.99], "budget_min": [60.0, 200.0]}


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
             compose: tuple | None = None, pipeline: bool = True, devs: dict | None = None,
             overrides: dict | None = None, params: dict | None = None) -> WorldSpec:
    """`compose` None (default): composed activities from the raw-feed world rule (`raw_feed_compositions`); a tuple
    of pairs configures them by name instead (experiments and negative controls only: the evaluator reports such
    activities as designer-supplied demand, `gates.designer_supplied_demand`)."""
    from ..civ import generator as G
    ov = copy.deepcopy(overrides or {})
    rec = G.recipe(profile, seed, population=population, overrides=copy.deepcopy(ov))
    fams = [dict(f) for f in rec["modules"]["work"]["families"]]
    for k, fam in enumerate(novel):
        fams.append(novel_family(fam, seed, k))
    for pair in (raw_feed_compositions(novel, seed) if compose is None else compose):
        fams.append(composed_family(pair, novel, seed))
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
                  compose: tuple | None = None, pipeline: bool = True, devs: dict | None = None,
                  params: dict | None = None) -> WorldSpec:
    """The ECON creation world (`innovation.popworld.creation_spec`: startup economy with incumbents, published
    formats and generated categories on the population) with the population's developers (unit tests). Compute
    is priced like the civilisation's providers (`civ.genesis`: a few cents per core-hour; ECON's startup worlds
    price it at 1.2-2.6 dollars, which no individual's project could pay)."""
    from .popworld import creation_spec as cs
    sp = cs(seed, n=n, published=published, novel=novel, novel_share=0.8, params=params)
    mods = [dict(m) for m in sp.modules]
    for m in mods:
        if m["name"] == "work":
            pairs = raw_feed_compositions(novel, seed) if compose is None else compose
            fams = list(m["config"]["families"]) + [composed_family(pair, novel, seed) for pair in pairs]
            m["config"] = dict(m["config"], families=fams + ([dict(PIPELINE)] if pipeline else []))
        if m["name"] == "ventures":
            provs = [dict(x, price=int(2 + k * 2)) for k, x in enumerate(m["config"].get("providers") or [])]
            m["config"] = dict(m["config"], providers=provs)
    return WorldSpec("creation_devs", int(seed), _with_devs(mods, devs), dict(sp.params))
