"""Execution mode B (standalone semantic digital worlds): every registered Tier 2 domain configuration is a world."""
from __future__ import annotations

from ..substrate import modes
from . import base, load_all


def _names() -> list[str]:
    load_all()
    return sorted(f"{d}:{c.name}" for d, dom in base.REGISTRY.items() for c in dom.configs)


def _make(name: str, seed: int = 0, **kw):
    load_all()
    domain, config = name.split(":", 1)
    return base.world(domain, config, seed, kw.get("overrides"))


modes.register("B", "domains", package="crucible.domains", names=_names, make=_make, kind="world",
               doc="Tier 2 domains (crucible.domains): one kernel world per domain configuration")
