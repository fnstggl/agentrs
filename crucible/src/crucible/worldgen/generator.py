"""Procedural world generation.

`generate(config, seed) -> WorldSpec`. A config names a *family* (a structural recipe) plus overrides. Families
differ causally, not cosmetically: which needs exist, how expensive compute is, how reachable customers are (search
and word-of-mouth rates, social graph rewiring), how loyal customers are, how strong and well-funded incumbents
are, how reliable infrastructure is, how abundant and cheap labour is, how communication behaves (mail latency,
failures), and which shocks arrive when. Within a family, the seed draws the concrete population, firms, prices
and schedules.

Partitions: every family is tagged `train`, `val` or `test`. The `test` families include entire need formats and
mechanism settings never present in `train` (held-out generators, not just held-out seeds). Use
`families(split)` to enumerate; the locked test families are only constructed through `crucible.evaluation`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.canonical import digest
from ..core.kernel import WorldSpec

BASE_MODULES = ["identity", "bank", "admin", "code", "search", "forum", "mail", "docs", "cal", "issues", "support",
                "ads", "contracts", "business", "jobs", "cloud", "market", "consumers", "npcs", "shocks", "goals", "civ"]

TRAIN_NEEDS = ["sort.records", "route.shortest", "schedule.jobs", "pack.knapsack", "text.extract"]
HELDOUT_NEEDS = ["route.tsp", "match.stable", "forecast.series"]


@dataclass
class Family:
    name: str
    split: str
    build: Callable[[np.random.Generator, dict], dict]
    doc: str


FAMILIES: dict[str, Family] = {}


def family(name: str, split: str, doc: str):
    def deco(fn):
        FAMILIES[name] = Family(name, split, fn, doc)
        return fn
    return deco


def families(split: str | None = None) -> list[str]:
    return sorted(n for n, f in FAMILIES.items() if split is None or f.split == split)


# ---------------------------------------------------------------------------- shared recipe
def _recipe(rng, *, needs, n_consumers, firms_per_need, tier_range, firm_cash, providers, workers, wage_base,
            lenders=1, search_rate=0.02, wom_rate=0.01, loyalty=(0.0, 0.9), rewire=0.15, mail_latency=60,
            mail_failure=0.0, shocks=(), params=None, price_mult=(0.8, 1.6), bounties=1, rate=(5.0, 30.0)):
    from ..economy.needs import NEEDS
    prov_cfg = []
    for i, (cores, price, rel) in enumerate(providers):
        prov_cfg.append({"handle": f"cloud{i + 1}", "name": f"Cloud{i + 1}", "cores": int(cores), "price": int(price),
                         "reliability": float(rel), "cash": int(5_000_000 + cores * 50_000)})
    firms = []
    k = 0
    for need in needs:
        for j in range(firms_per_need):
            k += 1
            tier = int(rng.integers(tier_range[0], min(tier_range[1], len(NEEDS[need].tiers) - 1) + 1))
            price = max(1, int(round(NEEDS[need].base_price_cents * float(rng.uniform(*price_mult)))))
            stem = need.split(".")[1][:6]
            firms.append({"handle": f"{stem}co{j + 1}", "name": f"{stem.title()}Co {j + 1}", "need": need,
                          "tier": tier, "price": price, "host": f"{stem}{j + 1}",
                          "provider": f"o:cloud{1 + int(rng.integers(0, len(prov_cfg)))}",
                          "cores": 1, "cash": int(firm_cash * rng.uniform(0.6, 1.4)),
                          "policy": {"price_step": float(rng.uniform(0.04, 0.12)),
                                     "rd_days": int(rng.integers(5, 20)), "outsource": bool(rng.random() < 0.7)}})
    bounty_cfg = []
    for f in firms[:bounties]:
        bounty_cfg.append({"poster": f["handle"], "need": f["need"], "reward": int(rng.integers(100_000, 400_000)),
                           "min_q": 0.97, "deadline_days": 60})
    lend = [{"handle": f"bank{i + 1}", "name": f"Lender {i + 1}", "rate": float(rng.uniform(0.01, 0.03)),
             "max": 3_000_000, "cash": 30_000_000} for i in range(lenders)]
    cons = {"n": int(n_consumers), "needs": {n: float(rng.uniform(0.6, 1.4)) for n in needs},
            "search_rate": search_rate, "wom_rate": wom_rate, "loyalty": list(loyalty), "rate_per_day": list(rate)}
    p = {"mail_latency_s": mail_latency, "mail_failure_p": mail_failure, "cu_per_ms": 50.0,
         "max_actions_per_hour": 240}
    p.update(params or {})
    return {"modules": {"consumers": cons,
                        "civ": {"needs": needs, "providers": prov_cfg, "firms": firms, "workers": workers,
                                "wage_base": wage_base, "lenders": lend, "bounties": bounty_cfg},
                        "shocks": {"events": list(shocks)}},
            "params": p, "rewire": rewire}


# ---------------------------------------------------------------------------- families
@family("bazaar", "train", "Balanced economy: several needs, two modest incumbents each, ordinary compute and reach.")
def f_bazaar(rng, o):
    needs = list(rng.choice(TRAIN_NEEDS, size=4, replace=False))
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=2, tier_range=(0, 1),
                   firm_cash=1_500_000, providers=[(40, 200, 0.97), (30, 260, 0.99)], workers=30, wage_base=300_000,
                   shocks=[{"day": 20, "kind": "pref_drift", "need": needs[0], "toward": [0.7, 0.1, 0.2],
                            "strength": 0.4}])


@family("cheap_compute_closed_markets", "train",
        "Compute is cheap and plentiful but customers are hard to reach (little search, weak word of mouth, loyal).")
def f_cheap_closed(rng, o):
    needs = list(rng.choice(TRAIN_NEEDS, size=3, replace=False))
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=2, tier_range=(0, 1),
                   firm_cash=1_000_000, providers=[(120, 60, 0.98), (80, 80, 0.98)], workers=25, wage_base=280_000,
                   search_rate=0.004, wom_rate=0.003, loyalty=(0.4, 0.95),
                   params={"hardware_cents_per_core": 120_000, "opex_cents_per_core_hour": 20})


@family("expensive_compute_open_distribution", "train",
        "Compute is scarce and expensive (algorithmic efficiency pays), customers are easy to reach and fickle.")
def f_expensive_open(rng, o):
    needs = list(rng.choice(TRAIN_NEEDS, size=4, replace=False))
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=2, tier_range=(0, 1),
                   firm_cash=2_000_000, providers=[(12, 900, 0.97), (10, 1100, 0.99)], workers=30, wage_base=320_000,
                   search_rate=0.05, wom_rate=0.03, loyalty=(0.0, 0.4),
                   params={"cu_per_ms": 120.0, "opex_cents_per_core_hour": 300, "hardware_cents_per_core": 900_000},
                   shocks=[{"day": 25, "kind": "compute_shortage", "capacity_loss": 0.3, "price_mult": 1.6,
                            "news": "Datacenter fire: compute capacity reduced across providers."}])


@family("entrenched_incumbents", "val",
        "Strong, rich incumbents at the best implementation tier; customers are loyal and switching is costly.")
def f_entrenched(rng, o):
    needs = list(rng.choice(TRAIN_NEEDS, size=3, replace=False))
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=2, tier_range=(1, 3),
                   firm_cash=8_000_000, providers=[(40, 220, 0.98), (30, 240, 0.98)], workers=20, wage_base=350_000,
                   loyalty=(0.6, 0.98), price_mult=(1.2, 2.0),
                   shocks=[{"day": 15, "kind": "entrant", "firm": {"handle": "upstart", "name": "Upstart", "need": needs[0],
                                                                  "tier": 3, "price": 2, "host": "upstart",
                                                                  "provider": "o:cloud1", "cash": 3_000_000}}])


@family("fragmented_networks", "val",
        "Communication is slow and lossy, the social graph is clustered, search is little used; many small niches.")
def f_fragmented(rng, o):
    needs = list(TRAIN_NEEDS)
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=1, tier_range=(0, 1),
                   firm_cash=1_000_000, providers=[(30, 220, 0.97), (30, 220, 0.97)], workers=20, wage_base=300_000,
                   search_rate=0.006, wom_rate=0.02, rewire=0.01, mail_latency=6 * 3600, mail_failure=0.15)


@family("labor_rich_fragile_infra", "test",
        "Abundant cheap labour, few incumbents, unreliable infrastructure with a reliability storm.")
def f_labor_fragile(rng, o):
    needs = list(rng.choice(TRAIN_NEEDS, size=3, replace=False))
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=1, tier_range=(0, 0),
                   firm_cash=800_000, providers=[(40, 180, 0.85), (40, 200, 0.88)], workers=80, wage_base=150_000,
                   shocks=[{"day": 10, "kind": "reliability", "reliability": 0.7,
                            "news": "Storms hit regional datacenters."}])


@family("novel_needs", "test",
        "Held-out problem formats never present in training families; one emerges mid-world as a new market.")
def f_novel(rng, o):
    needs = list(HELDOUT_NEEDS[:2])
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 6000), firms_per_need=2, tier_range=(0, 1),
                   firm_cash=1_500_000, providers=[(40, 200, 0.97), (30, 260, 0.99)], workers=30, wage_base=300_000,
                   shocks=[{"day": 12, "kind": "new_need", "need": HELDOUT_NEEDS[2], "fraction": 0.2,
                            "news": "A new kind of demand appears: forecasting."}])


@family("tiny", "dev", "Small world for tests and smoke runs.")
def f_tiny(rng, o):
    needs = o.get("needs") or ["sort.records", "route.shortest"]
    return _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 800), firms_per_need=o.get("firms_per_need", 1),
                   tier_range=(0, 0), firm_cash=500_000, providers=[(20, 200, 0.99)], workers=o.get("workers", 6),
                   wage_base=200_000, bounties=o.get("bounties", 1))


def generate(config: dict | str, seed: int = 0) -> WorldSpec:
    """Build a WorldSpec from {"family": name, "overrides": {...}, "modules": [...] (optional extra modules)}."""
    if isinstance(config, str):
        config = {"family": config}
    name = config.get("family", "bazaar")
    if name not in FAMILIES:
        raise KeyError(f"unknown world family {name!r}; known: {families()}")
    fam = FAMILIES[name]
    ov = dict(config.get("overrides", {}))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(name)[:8], 16)]))
    r = fam.build(rng, ov)
    mods = []
    for m in BASE_MODULES + list(config.get("extra_modules", [])):
        cfg = copy.deepcopy(r["modules"].get(m, {}))
        if m in ov.get("module_config", {}):
            cfg.update(ov["module_config"][m])
        mods.append({"name": m, "config": cfg})
    for m in config.get("drop_modules", []):
        mods = [x for x in mods if x["name"] != m]
    params = dict(r["params"])
    params.update(ov.get("params", {}))
    params["family_split"] = fam.split
    return WorldSpec(family=name, seed=int(seed), modules=mods, params=params)


def describe(name: str) -> dict:
    f = FAMILIES[name]
    return {"family": name, "split": f.split, "doc": f.doc}
