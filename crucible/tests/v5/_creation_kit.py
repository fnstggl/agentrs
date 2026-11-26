"""Shared counterfactual creation experiment for Gates J, L and SP-6.3 (one run per test session).

`arms()` runs the four arms of `crucible.innovation.experiments` (factual, no_saas, no_novelty, garbage) on one
seed and caches them for the session, so the gate files measure the same worlds. Every agent is LABELLED SCRIPTED.
"""
from __future__ import annotations

import functools

import crucible

crucible.load_all_modules()
from crucible.innovation import experiments as X  # noqa: E402

SEED = 3
N = 600
DAYS = 26
ARMS = ("factual", "no_saas", "no_novelty", "garbage")


@functools.lru_cache(maxsize=2)
def arms(seed: int = SEED, n: int = N, days: int = DAYS) -> dict:
    out = {}
    for a in ARMS:
        r = X.run_arm(seed, a, n=n, days=days)
        r["outcome"] = X.outcome(r)
        r["chains"] = X.mine_chains(r)
        out[a] = r
    return out
