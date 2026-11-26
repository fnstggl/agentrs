"""Crucible 5.0 - Tier 2 semantic digital-world domains (MASTER_SPEC Part 9, Gate N; D-17, D-48).

The framework is `crucible.domains.base` (the contract and its measurements; authoring guide:
docs/v5/components/I.md). Each domain lives in `crucible/domains/<name>.py` and calls `register(DOMAIN)` at import;
`load_all()` imports every domain module. Reachable as `crucible.experience.make(2, "<domain>:<config>", seed)`,
through execution mode B (`crucible.domains.modes`), the research distribution registry and the gate tests.

Public names (stable since 'domains-framework: ready'):

    Domain, DomainConfig, Seat, Mechanic, Actor, Probe          the contract
    register, unregister, validate, REGISTRY, part9_domains, PART9
    civ_recipe, recipe, spec, world, tier3_spec, ordered_spec    building worlds
    ScriptedReference, act_, Idle, RandomTools, drive, make_env, run
    check, structure_report, fingerprint, mechanics_exercised    measurements (Gate N, SP-9.0, SP-9.16)
    section_metrics                                              the shared metrics of a per-section SP-9.<n> test
"""
from __future__ import annotations

import importlib
import pkgutil

from .base import (API_VERSION, PART9, REGISTRY, Actor, Domain, DomainConfig, Idle, Mechanic, Probe,  # noqa: F401
                   RandomTools, ScriptedReference, Seat, act_, check, independent_revenue, civ_recipe, drive, fingerprint, make_env,
                   mechanics_exercised, ordered_spec, part9_domains, recipe, register, run, section_metrics, spec,
                   seat_orgs, structure_report, summary, tier3_spec, unregister, validate, world)

# framework and kernel-module files that define no domain (load_all imports the others)
_FRAMEWORK = {"base", "modes", "modules", "seats", "common", "lawlang", "concepts"}


def run_reference(domain: str, config: str, seed: int = 0, *, days: float | None = None, difficulty: int | None = None,
                  **_kw) -> dict:
    """The domain's scripted reference driven through the learner interface (`base.run(policy="reference")`)."""
    return run(domain, config, seed, policy="reference", days=days, difficulty=difficulty)


def load_all() -> dict:
    """Import every domain module in this package (each registers itself) and return the registry."""
    for info in pkgutil.iter_modules(__path__):
        if info.name not in _FRAMEWORK and not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")
    return REGISTRY
