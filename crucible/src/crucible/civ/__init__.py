"""Crucible 5.0 - Tier 3 integrated civilisations: generator, profiles, cross-domain couplings, causal chains, long
horizons (MASTER_SPEC Part 10, Gates O and P; ONE_WORLD_CONTRACTS section 4; owned by workstream J).

    PROFILES                                   >= 10 structurally different profiles (crucible.civ.profiles)
    generate(profile, seed, population=...)    -> World: one kernel world with every subsystem and the population
    recipe(profile, seed, population=...)      -> dict: the inspectable world-family form
    spec(profile, seed, population=...)        -> WorldSpec
    couplings                                  cross-domain relations (crucible.civ.couplings)
    chains                                     provenance graph, generic chain mining, counterfactual removal
    longrun                                    long-horizon harness (snapshots, resume, replay, accounting)

Kernel modules of this package live in `civ/modules.py` (imported by `crucible.load_all_modules()`).
"""
from __future__ import annotations

from . import couplings  # noqa: F401
from .generator import POPULATION_PROFILES, generate, recipe, spec  # noqa: F401
from .profiles import PROFILES, Profile  # noqa: F401
