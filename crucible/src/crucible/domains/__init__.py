"""Crucible 5.0 - Tier 2 semantic digital-world domains (15), each with configurations, objectives, evaluators and
hidden reference trajectories (MASTER_SPEC Part 9, INTERFACE_CONTRACTS section 9).

Owned by workstream I (docs/v5/WORKSTREAMS.md). The framework is `crucible.domains.base`; each domain lives in
`crucible/domains/<name>.py` and calls `base.register(DOMAIN)` at import. `load_all()` imports every domain file.
"""
from __future__ import annotations

import importlib
import pkgutil

from .base import REGISTRY, Domain, DomainConfig, register, run_reference, spec, world  # noqa: F401

_FRAMEWORK = {"base", "modes", "modules"}


def load_all() -> dict:
    """Import every domain module in this package (each registers itself) and return the registry."""
    for info in pkgutil.iter_modules(__path__):
        if info.name not in _FRAMEWORK and not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")
    return REGISTRY
