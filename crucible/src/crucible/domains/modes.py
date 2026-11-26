"""Execution mode B (standalone semantic digital worlds, SP-2.1): every configuration of every registered Tier 2
domain is a world of this provider (`"<domain>:<config>"`). An empty registry lists no world, so mode B is then not
operational (never faked)."""
from __future__ import annotations

from ..substrate import modes
from . import base, load_all


def _names() -> list[str]:
    load_all()
    return sorted(f"{d}:{c.name}" for d, dom in base.REGISTRY.items() for c in dom.configs if not dom.fixture)


def _make(name: str, seed: int = 0, **kw):
    load_all()
    domain, config = name.split(":", 1)
    return base.spec(domain, config, seed, kw.get("overrides"))


modes.register("B", "domains", package="crucible.domains", names=_names, make=_make, kind="world",
               doc="Tier 2 domains (crucible.domains): one kernel world per domain configuration instance")
