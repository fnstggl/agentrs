"""OpenWorld families: economies whose valuable work is drawn from procedural category families.

Importing this module registers the world families (it is not imported by `worldgen/__init__`; see the wiring
note in the work-package report).

    ow_market     train  open digital economy (firm minds, capability substrate) over training category families,
                         incumbents at low tiers
    ow_frontier   train  as ow_market plus latent categories (almost no active customers at first) and hidden
                         derived-demand edges: serving one category well creates need for another
    ow_composite  val    composed categories ("normalise, then ...") next to the categories they are made of
    ow_heldout    test   only held-out category families, one of them latent
    ow_tiny       dev    small and fast, for tests

The concrete rule of every category is drawn from the world seed, so two worlds of one family never share a
hidden rule. Demand for a latent category and the edges that wake it are written into the `consumers` config
(keys `latent` and `derived`, architecture section 2.5); agents can observe them only through their effects.
"""
from __future__ import annotations

import numpy as np

from ..usefulness.category import (NeedCategory, category_families, from_config, load_families, register_category)
from .generator import BASE_MODULES, FAMILIES, Family, _open, _recipe

MAX_SEED = 2**31 - 1
OW_FAMILIES: dict[str, Family] = {}


def family(name: str, split: str, doc: str):
    """`worldgen.generator.family`, remembering the OpenWorld families so they can be (un)registered as a set."""
    def deco(fn):
        OW_FAMILIES[name] = FAMILIES[name] = Family(name, split, fn, doc)
        return fn
    return deco


def register_ow_families() -> list[str]:
    """(Re-)register the OpenWorld families with the world generator. Idempotent."""
    FAMILIES.update(OW_FAMILIES)
    return sorted(OW_FAMILIES)


def unregister_ow_families() -> None:
    """Remove the OpenWorld families from the world generator (used by tests to stay hermetic: code that iterates
    over every train family must not pick them up as a side effect of an import)."""
    for n in OW_FAMILIES:
        FAMILIES.pop(n, None)


def _draw(rng: np.random.Generator, fams: list[str], k: int) -> list[dict]:
    """k categories of distinct families, each with its own seed."""
    names = [fams[int(i)] for i in rng.choice(len(fams), size=min(int(k), len(fams)), replace=False)]
    return [{"family": f, "seed": int(rng.integers(0, MAX_SEED)), "params": {}} for f in names]


def _edges(cats: list[NeedCategory], latent: set[int], volume: dict[str, float]) -> list[dict]:
    """Derived-demand edges of this world, taken from the categories' externalities: an edge exists where the
    source category names a family present in the world as a latent category."""
    out = []
    for i, c in enumerate(cats):
        for e in c.externalities.derived:
            for j, d in enumerate(cats):
                if j != i and j in latent and i not in latent and d.family == e.dst_family:
                    out.append({"src": c.name, "dst": d.name, "weight": e.weight,
                                "scale": round(e.scale * volume[c.name], 1), "lag_days": e.lag_days})
    return out


def _has_source(cats: list[NeedCategory], latent: set[int], j: int) -> bool:
    return any(i != j and i not in latent and any(e.dst_family == cats[j].family for e in c.externalities.derived)
               for i, c in enumerate(cats))


def _pick_latent(rng, cats: list[NeedCategory], k: int) -> set[int]:
    """Up to k latent categories, each with at least one derived-demand edge from a category that stays active."""
    latent: set[int] = set()
    for j in [int(x) for x in rng.permutation(len(cats))]:
        if len(latent) >= k:
            break
        trial = latent | {j}
        if all(_has_source(cats, trial, m) for m in trial):
            latent = trial
    return latent


def _ow(rng, o, *, cat_cfgs, compose=(), n_latent=0, n_consumers=4000, firms_per_need=2, composed_firms=1,
        tier_range=(0, 1), firm_cash=1_500_000, providers=((120, 150, 0.995), (80, 170, 0.995)), workers=12,
        wage_base=250_000, rate=(10.0, 40.0), open_world=True, consumers=None, params=None, segments=None,
        entry=False):
    load_families()
    cat_cfgs = [dict(c) for c in (o.get("categories") or cat_cfgs)]
    compose = [list(p) for p in (o.get("compose") if o.get("compose") is not None else compose)]
    use_cfg = {"categories": cat_cfgs, "compose": compose}
    cats = from_config(use_cfg)
    for c in cats:                      # needed already at generation time (goal feasibility reads tier qualities)
        register_category(c)
    n = int(o.get("n_consumers", n_consumers))
    rate = tuple(o.get("rate", rate))
    r = _recipe(rng, needs=[], n_consumers=n, firms_per_need=0, tier_range=(0, 0), firm_cash=firm_cash,
                providers=list(providers), workers=o.get("workers", workers), wage_base=wage_base, bounties=0,
                rate=rate)
    weights = {c.name: round(float(c.population.share * rng.uniform(0.7, 1.3)), 4) for c in cats}
    total_w = sum(weights.values())
    mean_rate = (rate[0] + rate[1]) / 2.0
    volume = {c.name: n * weights[c.name] / total_w * mean_rate for c in cats}     # nominal requests per day
    latent_idx = _pick_latent(rng, cats, int(o.get("n_latent", n_latent)))
    n_prov = len(r["modules"]["civ"]["providers"])
    per_need = int(o.get("firms_per_need", firms_per_need))
    firms = []
    for i, c in enumerate(cats):
        k = 0 if i in latent_idx else (int(o.get("composed_firms", composed_firms)) if c.parts else per_need)
        stem = ("cx" if c.parts else c.family[:5]) + str(i)
        for j in range(k):
            top = len(c.tiers) - 2                            # incumbents never run the best reference tier
            tier = int(rng.integers(min(tier_range[0], top), min(tier_range[1], top) + 1))
            firms.append({"handle": f"{stem}co{j + 1}", "name": f"{c.family.title()} {i}.{j + 1}", "need": c.name,
                          "tier": tier, "price": max(1, int(round(c.base_price_cents * float(rng.uniform(0.8, 1.6))))),
                          "host": f"{stem}svc{j + 1}", "provider": f"o:cloud{1 + int(rng.integers(0, n_prov))}",
                          "cores": 1, "cash": int(firm_cash * rng.uniform(0.6, 1.4)),
                          "policy": {"price_step": float(rng.uniform(0.04, 0.12)), "rd_days": int(rng.integers(5, 20)),
                                     "outsource": bool(rng.random() < 0.7)}})
    cons = r["modules"]["consumers"]
    cons["needs"] = weights
    mean_price = sum(weights[c.name] * c.base_price_cents for c in cats) / total_w
    cons["income_per_day_cents"] = int(0.7 * n * mean_rate * mean_price * 1.5)      # never the registry-wide default
    if latent_idx:
        cons["latent"] = {cats[j].name: round(float(rng.uniform(0.02, 0.08)), 3) for j in sorted(latent_idx)}
        cons["derived"] = _edges(cats, latent_idx, volume)
    cons.update(consumers or {})
    # who the customers are: a mix of structurally different behaviour models (economy/behavior.py). The mix is a
    # property of the family; `overrides.segments` replaces it (`[]`: the single 3.0 rule set).
    seg = o.get("segments", segments)
    if seg:
        cons["segments"] = [dict(x) for x in seg]
    # customers pay only for answers they can use: a verified score below the floor is refused, not paid
    if o.get("accept_floor", 0.15) is not None:
        cons["accept_floor"] = float(o.get("accept_floor", 0.15))
    civ = r["modules"]["civ"]
    civ["needs"] = [c.name for c in cats]
    civ["firms"] = firms
    r["modules"]["use"] = use_cfg
    with_entry = bool(o.get("entry", entry))
    if with_entry:                      # prospective founders who enter markets they judge profitable (actors/entry.py)
        r["modules"]["entry"] = dict(o.get("entry_config") or _founders(rng, int(o.get("founders", 14))))
    if open_world:
        r = _open(r, extra=["use"] + (["entry"] if with_entry else []),
                  consumers={"diurnal": float(o.get("diurnal", 0.7)), "peak_hour": 14,
                                               "k_samples": int(o.get("k_samples", 2))}, params=params)
    else:
        r["module_list"] = BASE_MODULES + ["use"] + (["entry"] if with_entry else [])
        cons["k_samples"] = int(o.get("k_samples", 2))
        r["params"].update({"cost_model": "deterministic", **(params or {})})
    return r


def _founders(rng, n: int, horizon_days: int = 360) -> dict:
    """A founder pool that does not run dry: six prospective founders look at markets from day 10 on, the others
    become active at seeded times spread over a year (their savings exist from genesis and sit idle until then).
    Traits are drawn over the module's default ranges."""
    specs = []
    for i in range(n):
        first = 10.0 + 3.0 * i / 6 if i < 6 else float(rng.uniform(30, horizon_days))
        specs.append({"capital": int(rng.integers(400_000, 1_600_000)), "risk": round(float(rng.uniform(0.15, 0.9)), 3),
                      "competence": round(float(rng.uniform(0.05, 0.95)), 3),
                      "informed": round(float(rng.uniform(0.1, 0.95)), 3),
                      "patience": round(float(rng.uniform(0.2, 0.95)), 3), "first_scan_day": round(first, 2)})
    return {"founder_specs": specs}


# Population mixes per family. Training worlds are dominated by the rule sets a learner can meet anywhere; the
# held-out family is dominated by social adoption and private experience, which are minorities or absent in
# training, so success there cannot come from having fitted one participant model.
MIX_MARKET = [{"name": "mainstream", "share": 0.55, "model": "legacy"},
              {"name": "price_shoppers", "share": 0.20, "model": "bargain"},
              {"name": "cautious", "share": 0.25, "model": "reputation_gate"}]
MIX_FRONTIER = [{"name": "mainstream", "share": 0.50, "model": "legacy"},
                {"name": "tinkerers", "share": 0.30, "model": "explorer"},
                {"name": "cautious", "share": 0.20, "model": "reputation_gate"}]
MIX_COMPOSITE = [{"name": "cautious", "share": 0.40, "model": "reputation_gate"},
                 {"name": "price_shoppers", "share": 0.30, "model": "bargain"},
                 {"name": "mainstream", "share": 0.30, "model": "legacy"}]
MIX_HELDOUT = [{"name": "followers", "share": 0.45, "model": "contagion"},
               {"name": "tinkerers", "share": 0.35, "model": "explorer"},
               {"name": "mainstream", "share": 0.20, "model": "legacy"}]


@family("ow_market", "train",
        "Open digital economy whose valuable work is three generated categories from training families; firms are "
        "minds running low-tier implementations, so better software is worth building.")
def f_ow_market(rng, o):
    return _ow(rng, o, cat_cfgs=_draw(rng, category_families("train"), int(o.get("n_categories", 3))),
               segments=MIX_MARKET, entry=True)


@family("ow_frontier", "train",
        "Open digital economy with a frontier: some generated categories are latent (almost no active customers) "
        "until other categories are served well; the derived-demand edges are drawn per world and hidden.")
def f_ow_frontier(rng, o):
    return _ow(rng, o, cat_cfgs=_draw(rng, category_families("train"), int(o.get("n_categories", 5))), n_latent=2,
               segments=MIX_FRONTIER, entry=True)


@family("ow_composite", "val",
        "Composed categories: requests that need one category's work and then another's on the result, next to "
        "the categories they are made of (an agent can build both halves or call someone else's service for one).")
def f_ow_composite(rng, o):
    train = [f for f in category_families("train") if f != "transform"]
    cfgs = [{"family": "transform", "seed": int(rng.integers(0, MAX_SEED)), "params": {}}] + _draw(rng, train, 1) + \
        _draw(rng, category_families("val"), 2)
    return _ow(rng, o, cat_cfgs=cfgs, compose=[[0, 1], [0, 2 + int(rng.integers(0, 2))]], segments=MIX_COMPOSITE,
               entry=True)


@family("ow_heldout", "test",
        "HELD-OUT CATEGORIES: only category families never present in training or validation worlds, one of them "
        "latent until another is served well.")
def f_ow_heldout(rng, o):
    return _ow(rng, o, cat_cfgs=_draw(rng, category_families("test"), 3), n_latent=1, segments=MIX_HELDOUT,
               entry=True)


@family("ow_tiny", "dev", "Small generated-category world for tests and smoke runs (no firm minds).")
def f_ow_tiny(rng, o):
    cfgs = [{"family": "transform", "seed": int(rng.integers(0, MAX_SEED)), "params": {}},
            {"family": "allocate", "seed": int(rng.integers(0, MAX_SEED)), "params": {}}]
    return _ow(rng, o, cat_cfgs=cfgs, n_consumers=600, firms_per_need=1, tier_range=(0, 0), firm_cash=500_000,
               providers=((20, 200, 0.99),), workers=6, wage_base=200_000, rate=(5.0, 30.0), open_world=False)


@family("ow_works", "train",
        "An open digital economy (generated need categories, incumbents, customers of several kinds, founders) "
        "that also contains a leasable facility with a hidden mechanism: selling software to customers and "
        "operating the facility are two unrelated ways of earning from independent demand in one world, through "
        "the same tools, accounts and clock.")
def f_ow_works(rng, o):
    from ..mechanisms.base import make_mechanism
    from ..mechanisms.module import rig_entry
    from ..mechanisms.sim import ensure_engines
    cfgs = [{"family": "transform", "seed": int(rng.integers(0, MAX_SEED)), "params": {}},
            {"family": "allocate", "seed": int(rng.integers(0, MAX_SEED)), "params": {}}]
    mech_seed = int(rng.integers(0, MAX_SEED))
    r = _ow(rng, o, cat_cfgs=cfgs, n_consumers=800, firms_per_need=1, tier_range=(0, 0), firm_cash=500_000,
            providers=((40, 200, 0.99),), workers=6, wage_base=200_000, rate=(5.0, 30.0), open_world=False,
            segments=MIX_MARKET, entry=False)
    ensure_engines()
    mech = o.get("mech") or make_mechanism(str(o.get("mech_family", "scm_additive")), int(o.get("mech_seed", mech_seed)))
    r["modules"]["rig"] = {"rigs": [rig_entry(mech, "works", int(o.get("tick_s", 3600)), None,
                                              [None] * int(mech["config"].get("n_seats", 1)),
                                              int(o.get("lease_cents", 20_000)))]}
    r["module_list"] = r["module_list"] + ["rig"]
    return r


def ow_categories(spec) -> list[NeedCategory]:
    """The categories of an OpenWorld spec (evaluator side)."""
    cfg = next((m["config"] for m in spec.modules if m["name"] == "use"), None)
    return from_config(cfg) if cfg else []
