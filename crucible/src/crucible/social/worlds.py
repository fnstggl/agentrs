"""World specs of the social internet on the ONE population (D-15): `crucible.population` (`pop`) is the only
population; every platform of the social internet (C1 sites, the content catalogue, the follow graph, communities,
messaging, creator subscriptions, newsletters, product discovery, developer and knowledge platforms) and the
distribution and consumption engine (`aud`, `feed`) run on it, with content and interests in the shared semantic
space (`space`, population.semantics). The crowd stand-in (`crowd`) is not used by v5 worlds.

`social_world(profile, seed, ...)` composes: the 4.0 services the platforms need, the substrate, `space`, `pop`
(optionally `popsites` + `popweb`, the population's web browsing, and `pop_adaptive`), every C1 module, `aud` with
the distributed platforms, `feed` with one recommender mixture per platform, and `pop_social` (members create).
"""
from __future__ import annotations

import copy

from ..core.kernel import WorldSpec
from ..population import profiles

BASE = ["identity", "bank", "admin", "artifacts", "cols", "space", "code", "search", "forum", "mail", "cloud", "net",
        "appdb"]
C1 = ["social_ids", "social_rows", "content", "studio", "social", "shorts", "tube", "posts", "pics", "blog", "pods",
      "community", "msg", "creators", "letters", "launch", "dev", "kb"]
#: platforms whose public items are distributed by feeds (aud + feed)
DISTRIBUTED = ["posts", "blog", "pics", "shorts", "tube", "pods", "launch", "dev", "kb", "community", "social"]
#: each platform's own distribution: genuinely different mechanisms (Part 4.5)
MIXTURES = {
    "posts": [{"name": "social", "weight": 1.0}, {"name": "chronological", "weight": 1.0},
              {"name": "interest", "weight": 0.8}, {"name": "explore", "weight": 0.2}],
    "blog": [{"name": "interest", "weight": 1.0}, {"name": "search", "weight": 0.6},
             {"name": "chronological", "weight": 0.6}, {"name": "reputation", "weight": 0.4}],
    "pics": [{"name": "trending", "weight": 1.0}, {"name": "interest", "weight": 0.6},
             {"name": "explore", "weight": 0.3}],
    "shorts": [{"name": "engagement", "weight": 1.0}, {"name": "explore", "weight": 0.4},
               {"name": "trending", "weight": 0.4}],
    "tube": [{"name": "cf", "weight": 1.0}, {"name": "chronological", "weight": 0.8},
             {"name": "interest", "weight": 0.5}],
    "pods": [{"name": "chronological", "weight": 1.0}, {"name": "interest", "weight": 0.5}],
    "launch": [{"name": "fresh", "weight": 1.0}, {"name": "trending", "weight": 0.6}],
    "dev": [{"name": "search", "weight": 1.0}, {"name": "reputation", "weight": 0.6},
            {"name": "chronological", "weight": 0.4}],
    "kb": [{"name": "search", "weight": 1.0}, {"name": "diversity", "weight": 0.5}],
    "community": [{"name": "chronological", "weight": 1.0}, {"name": "social", "weight": 0.5}],
    "social": [{"name": "social", "weight": 1.0}, {"name": "chronological", "weight": 1.0}],
}


def social_world(profile: str = "development", seed: int = 0, *, n: int | None = None,
                 classes: dict | None = None, platforms: list[str] | None = None, web: bool = False,
                 sites: int = 12, adaptive: bool = False, tick_s: int = 900, pop: dict | None = None,
                 feed: dict | None = None, aud: dict | None = None, creators: dict | None = None,
                 extra: list[dict] | None = None, params: dict | None = None,
                 family: str = "social_internet") -> WorldSpec:
    pcfg = {"profile": profile, "tick_s": int(tick_s), **(pop or {})}
    if n is not None:
        pcfg["n"] = int(n)
    pcfg["classes"] = dict(classes) if classes is not None else {"adaptive": 0, "strategic": 0, "semantic": 0}
    counts = profiles.resolve(pcfg)["counts"]
    plats = list(platforms or DISTRIBUTED)
    mods = [{"name": m, "config": {}} for m in BASE]
    if web:
        mods.append({"name": "popsites", "config": {"n": int(sites)}})
    mods.append({"name": "pop", "config": pcfg})
    if web:
        mods.append({"name": "popweb", "config": {}})
        if adaptive and counts["adaptive"]:
            mods.append({"name": "pop_adaptive", "config": {}})
    mods += [{"name": m, "config": {}} for m in C1]
    acfg = {"platforms": plats, "max_items": 4096, "max_creators": 1024, **(aud or {})}
    mods.append({"name": "aud", "config": acfg})
    fcfg = {"platforms": {p: {"recommenders": copy.deepcopy(MIXTURES.get(p, MIXTURES["posts"]))} for p in plats}}
    for k, v in (feed or {}).items():
        if k == "platforms":
            for p, pc in v.items():
                fcfg["platforms"].setdefault(p, {}).update(pc)
        else:
            fcfg[k] = v
    mods.append({"name": "feed", "config": fcfg})
    mods.append({"name": "pop_social", "config": dict(creators or {})})
    mods += list(extra or [])
    return WorldSpec(family, int(seed), mods, {"cost_model": "deterministic", **(params or {})})
