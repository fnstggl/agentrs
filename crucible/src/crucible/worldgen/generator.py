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
    economic: bool | None = None    # hint for goal generation (None: decided by building a spec once)


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
            mail_failure=0.0, shocks=(), params=None, price_mult=(0.8, 1.6), bounties=1, rate=(5.0, 30.0),
            provider_choice="random"):
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
                          "provider": "o:cloud1" if provider_choice == "cheapest" else
                          f"o:cloud{1 + int(rng.integers(0, len(prov_cfg)))}",
                          "cores": 1, "cash": int(firm_cash * rng.uniform(0.6, 1.4)),
                          "policy": {"price_step": float(rng.uniform(0.04, 0.12)),
                                     "rd_days": int(rng.integers(5, 20)), "outsource": bool(rng.random() < 0.7)}})
    bounty_cfg = []
    for f in firms[:bounties]:
        # where quality saturates (sorting), the firm also wants speed: a CPU budget per request
        bounty_cfg.append({"poster": f["handle"], "need": f["need"], "reward": int(rng.integers(100_000, 400_000)),
                           "min_q": 0.97, "deadline_days": 60,
                           "max_cpu_ms": 1.0 if f["need"] == "sort.records" else None})
    lend = [{"handle": f"bank{i + 1}", "name": f"Lender {i + 1}", "rate": float(rng.uniform(0.01, 0.03)),
             "max": 3_000_000, "cash": 30_000_000} for i in range(lenders)]
    cons = {"n": int(n_consumers), "needs": {n: float(rng.uniform(0.6, 1.4)) for n in needs},
            "search_rate": search_rate, "wom_rate": wom_rate, "loyalty": list(loyalty), "rate_per_day": list(rate),
            "rewire": rewire}
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


# ---------------------------------------------------------------------------- open-world families
OPEN_MODULES = ["ops", "wf", "caps", "deal", "gov", "net", "appdb", "society", "webgen", "adversary"]


def _open(r: dict, extra: list[str] | None = None, consumers: dict | None = None, params: dict | None = None) -> dict:
    """Turn a recipe into an open-world recipe: firms run by Tier-2 minds, the capability substrate installed."""
    r["module_list"] = BASE_MODULES + OPEN_MODULES + list(extra or [])
    r["modules"]["civ"]["firm_minds"] = True
    r["modules"]["consumers"].update(consumers or {})
    r["params"].update({"cost_model": "deterministic", **(params or {})})
    return r


@family("open_digital", "train",
        "Open digital economy: service firms are run by minds that measure their own books, publish problems and "
        "adopt (or reject) software others invent; demand follows a daily cycle, so fixed capacity is wasteful.")
def f_open_digital(rng, o):
    needs = o.get("needs") or list(rng.choice(["sort.records", "route.shortest", "text.extract"], size=2,
                                              replace=False))
    r = _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 4000), firms_per_need=o.get("firms_per_need", 2),
                tier_range=(0, 0), firm_cash=1_500_000, providers=[(120, 150, 0.995), (80, 170, 0.995)],
                workers=o.get("workers", 12), wage_base=250_000, bounties=0, rate=tuple(o.get("rate", (10.0, 40.0))))
    for f, tr in zip(r["modules"]["civ"]["firms"], o.get("firm_traits") or []):
        f["traits"] = dict(tr)
    return _open(r, consumers={"diurnal": float(o.get("diurnal", 0.7)), "peak_hour": 14,
                               "k_samples": int(o.get("k_samples", 2))})


INDUSTRY_MODULES = ["identity", "bank", "admin", "code", "search", "forum", "mail", "docs", "cal", "issues", "support",
                    "ads", "contracts", "business", "jobs", "cloud", "market", "npcs", "goals", "civ", "ops", "wf", "caps",
                    "deal", "gov", "net", "appdb", "society", "webgen", "adversary", "industry", "freight", "shoppers",
                    "industrygen"]


def _ring_links(rng, n, hours=(6, 14), cost=(2, 5), capacity=(500, 900), extra=0):
    links = [{"a": f"r{i}", "b": f"r{(i + 1) % n}", "hours": int(rng.integers(*hours)), "cost": int(rng.integers(*cost)),
              "capacity": int(rng.integers(*capacity))} for i in range(n)]
    for _ in range(extra):
        a, b = rng.choice(n, 2, replace=False)
        links.append({"a": f"r{a}", "b": f"r{b}", "hours": int(rng.integers(*hours)) + 6, "cost": int(rng.integers(*cost)) + 2,
                      "capacity": int(rng.integers(*capacity)) // 2})
    return links


def _industry_recipe(rng, o, *, regions=4, materials=6, grid_kwh=(6000, 9000), kwh_price=10.0, links=None,
                     shoppers=None, chain=None, params=None, workers=24, providers=((60, 150, 0.995),),
                     industry=None, grid_expand=True):
    r = _recipe(rng, needs=[], n_consumers=0, firms_per_need=0, tier_range=(0, 0), firm_cash=0,
                providers=list(providers), workers=o.get("workers", workers), wage_base=220_000, bounties=0)
    r["modules"]["civ"]["worker_roles"] = ["operator", "operator", "operator", "developer"]
    r["modules"]["civ"]["needs"] = []
    r["modules"]["industry"] = {"regions": regions, "materials": materials, **(industry or {})}
    r["modules"]["freight"] = {"links": links if links is not None else _ring_links(rng, regions)}
    r["modules"]["shoppers"] = {"n": int(o.get("n_shoppers", 20000)), **(shoppers or {})}
    r["modules"]["industrygen"] = {"grid_kwh": [int(rng.integers(*grid_kwh)) for _ in range(regions)],
                                   "kwh_price": kwh_price, "grid_expand": grid_expand,
                                   **({"chain": chain} if chain else {})}
    r["modules"]["civ"]["firm_minds"] = True
    r["module_list"] = list(INDUSTRY_MODULES)
    r["params"].update({"cost_model": "deterministic", **(params or {})})
    return r


def _mixed(rng, o, **kw):
    """A digital economy and an industrial economy in one world, sharing people, money, compute and institutions."""
    d = f_open_digital(rng, o)
    i = _industry_recipe(rng, o, **kw)
    mods = dict(d["modules"])
    for k in ("industry", "freight", "shoppers", "industrygen"):
        mods[k] = i["modules"][k]
    mods["civ"]["worker_roles"] = ["operator", "operator", "developer", "support", "sales"]
    mods["civ"]["workers"] = int(o.get("workers", 30))
    r = {"modules": mods, "params": {**i["params"], **d["params"]}, "rewire": d["rewire"]}
    r["module_list"] = BASE_MODULES + OPEN_MODULES + ["industry", "freight", "shoppers", "industrygen"]
    return r


@family("open_mixed", "train",
        "Mixed open civilisation: service firms with daily demand cycles, a multi-tier industrial supply chain, "
        "households buying both, shared labour, compute, money, negotiation and institutions.")
def f_open_mixed(rng, o):
    return _mixed(rng, o)


@family("open_scarce_energy", "val",
        "Industrial world whose regional grids are small and slow to expand: energy is the binding constraint, so "
        "process efficiency (and where production happens) decides who can produce.")
def f_open_scarce_energy(rng, o):
    return _industry_recipe(rng, o, grid_kwh=(1500, 2600), kwh_price=18.0, grid_expand=False)


@family("open_fragile_suppliers", "val",
        "Industrial world with a long, thinly connected freight network (one link of low capacity) and a single "
        "shared supplier: upstream failures propagate downstream.")
def f_open_fragile(rng, o):
    links = _ring_links(rng, 4, hours=(10, 20), capacity=(150, 300))
    return _industry_recipe(rng, o, links=links)


@family("open_weak_institutions", "test",
        "HELD-OUT MECHANISM: no public reputation and unenforceable claims (breaches leave no public record); trust "
        "can only be built privately. Otherwise a mixed civilisation.")
def f_open_weak(rng, o):
    r = _mixed(rng, o)
    r["params"]["public_reputation"] = False
    return r


@family("open_perishable", "test",
        "HELD-OUT MECHANISM: components and products perish (a share of stock is lost every day) and freight is "
        "slow, so inventory policy and logistics dominate; unseen chain topology (both assemblers share a supplier "
        "set with a third component maker).")
def f_open_perishable(rng, o):
    chain = _perishable_chain()
    return _industry_recipe(rng, o, industry={"decay": 0.06}, chain=chain,
                            links=_ring_links(rng, 4, hours=(16, 30), capacity=(400, 700)))


def _perishable_chain():
    from ..worlds.open_industry import default_chain
    ch = default_chain(4)
    ch.insert(4, {"handle": "forge3", "name": "Fibre Works", "role": "components", "op": "form", "region": 3,
                  "material": 1, "form": "fiber", "product_name": "fibre", "price": 280, "operators": 3})
    for f in ch:
        if f["handle"] == "assem2":
            f["components"] = ["forge1", "forge2", "forge3"]
    return ch


@family("open_industry", "train",
        "Industrial open world: a multi-tier supply chain (mines -> component makers -> assemblers -> households) "
        "with hidden process physics, machine wear, finite energy grids and freight capacity; firms are minds.")
def f_open_industry(rng, o):
    return _industry_recipe(rng, o)


@family("open_tight_compute", "val",
        "Open digital economy whose cheapest compute provider (where every firm starts) is small relative to the "
        "daily peak; a pricier provider has spare capacity. Fixed capacity per firm fits; capacity that every firm "
        "re-acquires at the same peak hour may not.")
def f_open_tight_compute(rng, o):
    needs = o.get("needs") or ["route.shortest", "sort.records"]
    r = _recipe(rng, needs=needs, n_consumers=o.get("n_consumers", 4000), firms_per_need=o.get("firms_per_need", 3),
                tier_range=(0, 0), firm_cash=1_500_000,
                providers=[(int(o.get("cheap_cores", 16)), 150, 0.995), (int(o.get("spare_cores", 80)), 210, 0.995)],
                workers=o.get("workers", 12), wage_base=250_000, bounties=0, rate=tuple(o.get("rate", (10.0, 40.0))),
                provider_choice="cheapest")
    return _open(r, consumers={"diurnal": float(o.get("diurnal", 0.7)), "peak_hour": 14,
                               "k_samples": int(o.get("k_samples", 2))})


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
    for m in r.get("module_list", BASE_MODULES) + list(config.get("extra_modules", [])):
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


# ---------------------------------------------------------------------------- engineering lab (non-economic worlds)
# A computational engineering laboratory (crucible/worlds/engineering_lab.py): improve the routing controller of a
# production network; ground truth is executed flow physics on a hidden traffic process. Own module list (no
# economy modules); the build result carries it as "module_list".
LAB_MODULES = ["identity", "bank", "admin", "code", "search", "forum", "mail", "docs", "lab"]


def _lab_recipe(o, *, lab, params=None):
    cfg = dict(lab)
    cfg["engineer_seat"] = o.get("engineer_seat", "eng")
    cfg["operator"] = dict(o.get("operator", {"handle": "netops", "name": "NetOps", "lead": "netops-lead",
                                              "cash": 5_000_000}))
    for k in ("n_nodes", "n_demands", "shocks", "peak_util"):
        if k in o:
            cfg[k] = o[k]
    p = {"max_actions_per_hour": 240, "free_cpu_ms_per_day": 2_000, "dev_compute_cents_per_cpu_s": 20.0,
         "mail_latency_s": 60}
    p.update(params or {})
    return {"modules": {"lab": cfg}, "params": p, "module_list": list(LAB_MODULES)}


@family("engineering_lab", "train",
        "Engineering lab: improve the routing controller of a moderate grid-with-express-links network under "
        "gravity traffic with one diurnal peak, hotspots and slow drift; measured by executed flow physics.")
def f_engineering_lab(rng, o):
    return _lab_recipe(o, lab={"topology": "grid", "traffic": "gravity", "n_nodes": int(rng.integers(14, 21)),
                               "shortcuts": int(rng.integers(2, 5)), "n_demands": int(rng.integers(30, 46)),
                               "hotspots": int(rng.integers(1, 3)), "hotspot_mult": [2.5, 4.0],
                               "peak_util": round(float(rng.uniform(1.0, 1.15)), 3), "drift_sigma": 0.06,
                               "shock_p": 0.1, "sla_factor": 1.2})


@family("engineering_lab_heldout", "test",
        "Held-out engineering lab: larger scale-free network, datacenter-hub traffic with two daily peaks, "
        "stronger flash-crowd hotspots and more frequent shocks (topology and traffic generators unseen in train).")
def f_engineering_lab_heldout(rng, o):
    return _lab_recipe(o, lab={"topology": "scale_free", "traffic": "hub", "n_nodes": int(rng.integers(28, 41)),
                               "ba_m": 2, "n_hubs": int(rng.integers(2, 4)), "n_demands": int(rng.integers(60, 81)),
                               "hotspots": 2, "hotspot_mult": [3.0, 6.0],
                               "peak_util": round(float(rng.uniform(1.05, 1.25)), 3), "drift_sigma": 0.1,
                               "shock_p": 0.2, "sla_factor": 1.2})
