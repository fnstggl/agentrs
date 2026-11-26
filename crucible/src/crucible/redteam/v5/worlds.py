"""Worlds for the Crucible 5.0 probes.

`v5_world_factory(**requirements)` is a drop-in world factory for `crucible.redteam.run_probes`: a probe that sets
`world = {"v5": "<kind>", ...}` gets a small real world with the 5.0 modules of that kind; any other probe (the 4.0
catalogue) is passed to the 4.0 `default_world_factory` unchanged.

Kinds (each is an ordinary `WorldSpec` with the deterministic cost model, so replay and regeneration are exact):

* ``substrate``  - identity, bank, operator tools, `artifacts`, `cols`, `assets` (+ `media` with ``media=True``)
* ``commerce``   - the F1 commerce stack (products, subs, ventures, xchg) on the 4.0 economy services, plus the F2
                   institutions, commitments and labour market (``workers`` simulated workers)
* ``innovation`` - `crucible.innovation.worlds.innovation_spec` (caps registry, needs graph, pkg, adopt, infra)
* ``all``        - every 5.0 kernel module merged so far in one world, with the 4.0 services they build on
* ``experience`` - like ``all``; probes wrap it in `crucible.experience.KernelExperienceEnv` themselves
* ``social``     - the C1 social internet (catalogue, studio, follow graph, sites, communities, messaging, creator
                   memberships, launches, Q&A) with code hosting and the cloud
* ``audience``   - C2: the stand-in population `crowd`, exact audience accounting `aud` and the feeds `feed`
* ``population`` - the B population (`pop`, `popweb`, generated sites) at a small size

Module configuration is overridable with ``configs={module: {...}}``; extra modules with ``extra=[...]``.
"""
from __future__ import annotations

import os
from typing import Iterable

from ..framework import NotApplicable, default_world_factory

BASE = ["identity", "bank", "admin"]
SUBSTRATE = BASE + ["artifacts", "cols", "assets"]
COMMERCE = BASE + ["code", "search", "docs", "support", "contracts", "business", "cloud", "infra", "market", "products", "subs",
                   "ventures", "xchg", "gov", "deal", "inst", "commit", "labor"]
ALL = COMMERCE + ["mail", "cal", "forum", "issues", "wf", "caps", "infra", "pkg", "needs", "artifacts", "cols",
                  "assets", "media", "embed", "exp", "semantic_inference"]
DEFAULT_CONFIGS = {"caps": {"registry": True}, "forum": {"boards": ["general", "dev", "market"]},
                   "labor": {"workers": 0}}


def _have(names: Iterable[str]) -> list[str]:
    import crucible
    crucible.load_all_modules()
    from ...core.kernel import MODULES
    return [n for n in names if n in MODULES]


def v5_spec(kind: str, seed: int = 1, *, configs: dict | None = None, extra: Iterable[str] = (),
            params: dict | None = None, workers: int = 0, media: bool = False, drop: Iterable[str] = ()):
    """The `WorldSpec` of one probe-world kind (see the module docstring)."""
    from ...core.kernel import WorldSpec
    _have([])                                  # every registered kernel module (also for the innovation kind)
    from . import harness  # noqa: F401 - the red team's own harness module (rt_cohort) is registered up front
    if kind == "population":
        from ...population.worlds import population_world
        pc = dict((configs or {}).get("pop", {}))
        spec = population_world("development", int(seed), n=int(pc.pop("n", 300)), sites=int(pc.pop("sites", 6)),
                                 adaptive=False, strategic=False, pop=pc or None, params=params)
        have = {m["name"] for m in spec.modules}
        for m in extra:
            if m not in have:
                spec.modules.append({"name": m, "config": dict((configs or {}).get(m, {}))})
        return spec
    if kind in ("social", "audience"):
        from ...social.modules import social_internet_modules
        more = ["code", "cloud", "net"] if kind == "social" else ["crowd", "aud", "feed"]
        mods = social_internet_modules(more + [m for m in extra if m not in more])
        mods = [m for m in mods if m["name"] not in set(drop)]
        for m in mods:
            m["config"] = dict((configs or {}).get(m["name"], {}))
        p = {"cost_model": "deterministic"}
        p.update(params or {})
        return WorldSpec(f"v5-redteam-{kind}", int(seed), mods, p)
    if kind == "innovation":
        from ...innovation.worlds import innovation_spec
        spec = innovation_spec(int(seed), **dict((configs or {}).get("_innovation", {})))
        have = {m["name"] for m in spec.modules}
        for m in extra:
            if m not in have:
                spec.modules.append({"name": m, "config": dict((configs or {}).get(m, {}))})
        if params:
            spec.params.update(params)
        return spec
    names = {"substrate": SUBSTRATE + (["media"] if media else []), "commerce": COMMERCE, "all": ALL,
             "experience": ALL}.get(kind)
    if names is None:
        raise NotApplicable(f"unknown v5 probe world kind {kind!r}")
    names = list(dict.fromkeys(list(names) + list(extra)))
    names = [n for n in names if n not in set(drop)]
    missing = [n for n in names if n not in _have(names)]
    if missing:
        raise NotApplicable(f"modules not in this build: {missing}")
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIGS.items()}
    if workers:
        cfg["labor"] = {"workers": int(workers)}
    for k, v in (configs or {}).items():
        cfg.setdefault(k, {}).update(v)
    p = {"cost_model": "deterministic"}
    p.update(params or {})
    return WorldSpec(f"v5-redteam-{kind}", int(seed), [{"name": n, "config": dict(cfg.get(n, {}))} for n in names], p)


def v5_world_factory(v5: str | None = None, seed: int = 1, store=None, configs: dict | None = None,
                     extra: Iterable[str] = (), params: dict | None = None, workers: int = 0, media: bool = False,
                     drop: Iterable[str] = (), strict: bool = False, **legacy):
    """World factory for `run_probes`: 5.0 probe worlds for probes that ask for them, the 4.0 factory otherwise."""
    from ...core.kernel import World
    if v5 is None:
        return default_world_factory(seed=seed, store=store, params=params, strict=strict, drop=drop, **legacy)
    spec = v5_spec(v5, seed, configs=configs, extra=extra, params=params, workers=workers, media=media, drop=drop)
    tmp = None
    if store == "sqlite":
        import tempfile
        from ...storage.sqlite import SQLiteStore
        tmp = tempfile.mkdtemp(prefix="crucible-redteam-v5-")
        store = SQLiteStore(os.path.join(tmp, "world.db"))
    elif store == "memory":
        from ...storage.memory import MemoryStore
        store = MemoryStore()
    w = World.create(spec, store=store, strict=strict)
    if tmp is not None:
        w.runtime["redteam_tmpdir"] = tmp
    return w
