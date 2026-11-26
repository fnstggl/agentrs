"""Generated startup economies on the one population (Gate K, Part 9.1; D-15, D-17).

`startup_spec(seed, ...)` draws a world: two or more public problem formats in which individuals of the population
work (`economy.work` families with drawn workloads, hand-work times and acceptance standards), one or two compute
providers, one to three incumbent SaaS firms per format running a kernel reference solver that is never the best
known tier, at drawn prices and trial policies, that respond to the market by explicit rules (`rivals`), the 4.0
internet services a business needs (search with sponsored results, forum, documents, support, ads, code hosting,
cloud) and the 5.0 commerce stack (`products`, `subs`, `ventures`). Demand is the population's: nobody buys
anything except individuals of `pop` deciding from their own work outcomes.

This is the commerce slice of a civilization recipe: workstream CIV composes the same modules and configs into
`crucible.civ.generate` (integration note), and the Tier 2 startup domain is a configuration of it.
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import WorldSpec

DAY = 86400
FORMAT_POOL = ("schedule.jobs", "pack.knapsack", "route.tsp", "route.shortest", "match.stable", "text.extract")
BASE = ["identity", "bank", "admin", "artifacts", "cols", "code", "search", "docs", "forum", "support", "contracts",
        "business", "cloud", "market", "ads", "products", "subs", "ventures"]


def draw(seed: int, *, n: int = 3000, formats: list[str] | None = None, n_formats: int = 2) -> dict:
    """The world's drawn parameters (inspectable; `startup_spec` turns them into a world spec)."""
    from ..economy.needs import NEEDS
    rng = np.random.default_rng([int(seed), 5150, 2])
    fmts = list(formats) if formats else [str(x) for x in rng.choice(FORMAT_POOL, size=int(n_formats), replace=False)]
    families, incumbents = [], []
    for k, f in enumerate(fmts):
        families.append({"need": f, "share": float(rng.uniform(0.6, 1.4)), "tasks_day": [1.0, float(rng.uniform(2.0, 3.0))],
                         "diy_min": [float(rng.uniform(8, 15)), float(rng.uniform(30, 55))], "size": [0.2, 0.6],
                         "standard": [float(rng.uniform(0.86, 0.9)), float(rng.uniform(0.96, 0.99))],
                         "budget_min": [60.0, 200.0]})
        tiers = len(NEEDS[f].tiers)
        for i in range(int(rng.integers(1, 4))):
            tier = int(rng.integers(0, max(1, tiers - 1)))                # never the best known solver
            month = int(rng.integers(15, 45)) * 100 - 1
            incumbents.append({"handle": f"{f.split('.')[0][:6]}{k}{i}", "name": f"{NEEDS[f].title.split()[0]} Co {k}{i}",
                               "need": f, "tier": tier, "price_month": month,
                               "price_year": month * 10 if rng.random() < 0.5 else None,
                               "trial_days": int(rng.choice([0, 7, 14])), "cores": 2, "cash": 5_000_000})
    providers = [{"handle": "cloudco", "name": "CloudCo", "cores": 128, "price": int(rng.integers(150, 260)),
                  "reliability": 0.995, "cash": 10_000_000}]
    if rng.random() < 0.5:
        providers.append({"handle": "nimbus", "name": "Nimbus", "cores": 64, "price": int(rng.integers(120, 300)),
                          "reliability": 0.98, "cash": 8_000_000})
    return {"seed": int(seed), "n": int(n), "formats": fmts, "families": families, "incumbents": incumbents,
            "providers": providers, "professional_share": float(rng.uniform(0.5, 0.7)),
            "need_end_month_p": float(rng.uniform(0.02, 0.05))}


def startup_spec(seed: int, *, n: int = 3000, formats: list[str] | None = None, n_formats: int = 2,
                 browse: bool = False, work: dict | None = None, rivals: dict | None = None,
                 params: dict | None = None, incumbents: list[dict] | None = None,
                 extra: tuple = ()) -> WorldSpec:
    """`incumbents` replaces the drawn incumbents (experiments that set the competitors, e.g. graded quality);
    `extra` adds modules after the commerce stack (`xchg` exchanges, `inst` institutions, `labor` the labour
    market: the population takes part in each through `economy.work`)."""
    d = draw(seed, n=n, formats=formats, n_formats=n_formats)
    if incumbents is not None:
        d["incumbents"] = [dict(x) for x in incumbents]
    mods = [{"name": m, "config": {}} for m in BASE]
    for m in extra:
        if m not in BASE:
            mods.append({"name": m, "config": {}})
    for m in mods:
        if m["name"] == "ventures":
            m["config"] = {"providers": d["providers"], "incumbents": d["incumbents"]}
        elif m["name"] == "forum":
            m["config"] = {"boards": ["general", "tools", "help", "market", "dev"]}
        elif m["name"] == "caps":
            m["config"] = {"registry": True}
    mods.append({"name": "wf", "config": {}})
    mods.append({"name": "caps", "config": {"registry": True}})
    if browse:
        mods.append({"name": "net", "config": {}})
        mods.append({"name": "appdb", "config": {}})
        mods.append({"name": "popsites", "config": {"n": 12}})
    mods.append({"name": "pop", "config": {"profile": "development", "n": d["n"], "tick_s": 900,
                                           "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}})
    if browse:
        mods.append({"name": "popweb", "config": {}})
    wcfg = {"families": d["families"], "professional_share": d["professional_share"],
            "need_end_month_p": d["need_end_month_p"], "publish_formats": True}
    wcfg.update(work or {})
    mods.append({"name": "work", "config": wcfg})
    mods.append({"name": "rivals", "config": dict(rivals or {})})
    return WorldSpec("startup", int(seed), mods, {"cost_model": "deterministic", "formats": d["formats"],
                                                  **(params or {})})
