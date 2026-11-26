"""SP-14.2 - Default tier mixture 50/30/20 with all tiers available (MASTER_SPEC Part 14.2; workstream N).

The default mixture (`sampler.DEFAULT_MIXTURE`: 50% of training compute to Tier 1, 30% to Tier 2, 20% to Tier 3)
is executed: episodes drawn by the deficit scheduler are built and run with the architecture-neutral reference
learner, the CPU seconds of every episode are measured, and the realised compute shares must be within 0.05 of the
targets with every tier actually instantiated. A tier without training worlds makes the default sampler refuse to
start (it never trains on fewer tiers silently); the test then records that and fails.
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import sampler as S

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
TOL = 0.05


def test_SP14_2_default_mixture():
    reg = D.build_registry()
    try:
        res = S.run_mixture(reg, S.DEFAULT_MIXTURE, episodes=60)
        why = ""
    except S.TierUnavailable as e:
        res, why = None, str(e)
    if res is None:
        record("SP-14.2", {"default_mixture_matches": False, "tiers_instantiated": 0,
                           "training_families_tier1": reg.tiers()[1], "training_families_tier2": reg.tiers()[2],
                           "training_families_tier3": reg.tiers()[3]},
               notes=f"The default 50/30/20 mixture cannot run in this checkout: {why}")
        pytest.fail(f"default mixture not executable: {why}")
    comp = res["shares"]["compute"]
    err = max(abs(comp[str(t)] - v) for t, v in S.DEFAULT_MIXTURE.tiers.items())
    inst = sum(1 for v in res["built"].values() if v > 0)
    matches = err <= TOL and inst == 3
    record("SP-14.2", {"default_mixture_matches": bool(matches), "tiers_instantiated": inst,
                       "max_compute_share_error": float(err), "compute_share_tier1": comp["1"],
                       "compute_share_tier2": comp["2"], "compute_share_tier3": comp["3"]},
           notes=f"Realised shares {res['shares']}; episodes per tier {res['built']}")
    assert matches
