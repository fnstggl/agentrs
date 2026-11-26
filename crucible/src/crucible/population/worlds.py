"""World specs with a population (used by tests, benchmarks and the tiers that need people).

`population_world(profile, seed, ...)` composes the 4.0 internet (identity, bank, code, search, forum, cloud, net,
appdb), the 5.0 substrate (artifacts, columns), generated web sites (`popsites`) and the population modules
(`pop`, `popweb`, and `pop_adaptive` / `pop_strategic` when the profile has those classes).
"""
from __future__ import annotations

from ..core.kernel import WorldSpec
from . import profiles

BASE = ["identity", "bank", "admin", "artifacts", "cols", "code", "search", "forum", "cloud", "net", "appdb"]


def population_world(profile: str = "development", seed: int = 0, *, n: int | None = None, sites: int = 24,
                     site_specs: list | None = None, mix: dict | None = None, shop_share: float = 0.4,
                     tick_s: int = 900, adaptive: bool | dict = True, strategic: bool | dict = True,
                     classes: dict | None = None, pop: dict | None = None, web: dict | None = None,
                     forum: bool = True, params: dict | None = None, family: str = "population") -> WorldSpec:
    pcfg = {"profile": profile, "tick_s": int(tick_s), **(pop or {})}
    if n is not None:
        pcfg["n"] = int(n)
    if classes:
        pcfg["classes"] = dict(classes)
    counts = profiles.resolve(pcfg)["counts"]
    mods = [{"name": m, "config": {}} for m in BASE if m != "forum" or forum]
    scfg = {"sites": site_specs} if site_specs is not None else {"n": int(sites), "mix": mix, "shop_share": shop_share}
    mods.append({"name": "popsites", "config": scfg})
    mods.append({"name": "pop", "config": pcfg})
    mods.append({"name": "popweb", "config": dict(web or {})})
    if adaptive and counts["adaptive"]:
        mods.append({"name": "pop_adaptive", "config": dict(adaptive) if isinstance(adaptive, dict) else {}})
    if strategic and counts["strategic"]:
        mods.append({"name": "pop_strategic", "config": dict(strategic) if isinstance(strategic, dict) else {}})
    return WorldSpec(family, int(seed), mods, {"cost_model": "deterministic", **(params or {})})
