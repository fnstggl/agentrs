"""Civilisation worlds with the class-4 runtime bound.

`crucible.civ.generate` composes every subsystem of a civilisation; the class-4 runtime (`pop_semantic`) belongs
right after the population's other class runtimes. Until the civilisation generator lists it in its module order
(integration request of workstream D to CIV), `civ_world` builds the same spec as `civ.spec` and inserts
`pop_semantic` after `pop_strategic`; once the generator includes it, `civ_world` is exactly `civ.generate`.
"""
from __future__ import annotations

from ...core.kernel import World, WorldSpec

MODULE = "pop_semantic"


def ensure_module(spec: WorldSpec, config: dict | None = None) -> WorldSpec:
    """`spec` with `pop_semantic` (and `config` for it) placed after the population's class runtimes."""
    mods = [dict(m) for m in spec.modules]
    names = [m["name"] for m in mods]
    if MODULE in names:
        if config:
            k = names.index(MODULE)
            mods[k] = {"name": MODULE, "config": {**(mods[k].get("config") or {}), **config}}
            return WorldSpec(spec.family, spec.seed, mods, spec.params, spec.kernel)
        return spec
    after = max(names.index(m) for m in ("pop", "pop_adaptive", "pop_strategic") if m in names)
    mods.insert(after + 1, {"name": MODULE, "config": dict(config or {})})
    return WorldSpec(spec.family, spec.seed, mods, spec.params, spec.kernel)


def civ_world(profile: str = "balanced", seed: int = 0, population: str = "standard", *, config: dict | None = None,
              overrides: dict | None = None, store=None, world_id: str | None = None, strict: bool = False) -> World:
    """A civilisation (`crucible.civ`) whose class-4 individuals are semantic participants."""
    from ... import civ
    from . import module  # noqa: F401  (registers the kernel module)
    sp = ensure_module(civ.spec(profile, seed, population, overrides=overrides), config)
    return World.create(sp, world_id=world_id or f"civ-{sp.hash()[:12]}", store=store, strict=strict)
