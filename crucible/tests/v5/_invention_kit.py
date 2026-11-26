"""Shared invention experiment for Gates J and L, SP-6.3 and SP-6.3b (one run per test session).

`A()` runs the four arms of `crucible.innovation.civexp` (factual, garbage, ban_novel, ban_published) in the
integrated civilisation with the population's autonomous developers (`innovation.inventworld.civ_spec`: profile
balanced, seed 3, development population, 24 simulated days) and caches them for the session, so the gate files
measure the same worlds. Nobody is scripted.
"""
from __future__ import annotations

import crucible

crucible.load_all_modules()
from crucible.innovation import gates as G  # noqa: E402

KIND, SEED, DAYS, PROFILE = "civ", 3, 24, "balanced"


def A() -> dict:
    return G.arms(KIND, SEED, DAYS, PROFILE)
