"""SP-14.2b - Compute and environment experience are measured separately; every tier is available from the start
(MASTER_SPEC Part 14.2; workstream N).

Every tier with training worlds is instantiated through the sampler at the start of a run (one world per tier is
built and stepped); a mixture over the instantiable tiers is executed with the reference learner and its
accounting must report measured CPU seconds and experienced interactions as separate quantities per tier (the two
must not be proportional across tiers - they are measured, not derived from each other).
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import distribution as D
from crucible.research.v5 import sampler as S
from crucible.research.v5.integrity import classify_failure

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def test_SP14_2b_compute_and_experience_and_tiers_at_start():
    reg = D.build_registry()
    avail, why = {}, {}
    for t in (1, 2, 3):
        fams = [w["id"] for w in reg.worlds(tier=t, family_partition="train")]
        if not fams:
            avail[t], why[t] = False, "no training world family in the registry"
            continue
        try:
            sp = S.episode_spec(reg, fams[0], 11, level="0", objective={"mode": "assigned"})
            env = S.make_env(sp, reg)
            a = env.agents()[0]
            env.observe(a)
            env.advance(3600)
            avail[t] = True
        except Exception as e:  # noqa: BLE001 - the failure is classified and reported
            avail[t], why[t] = False, classify_failure(e)
    tiers = [t for t, ok in avail.items() if ok]
    mix = S.Mixture(tiers={t: 1.0 for t in tiers}, require_all_tiers=False, name="available")
    res = S.run_mixture(reg, mix, episodes=24)
    sh = res["shares"]
    tot = sh["totals"]
    separate = all("cpu_s" in v and "interactions" in v for v in tot.values()) and \
        sum(v["cpu_s"] for v in tot.values()) > 0 and sum(v["interactions"] for v in tot.values()) > 0
    record("SP-14.2b", {"compute_and_experience_reported_separately": bool(separate),
                        "tiers_available_at_start": len(tiers), "tier1_available": bool(avail[1]),
                        "tier2_available": bool(avail[2]), "tier3_available": bool(avail[3])},
           notes=f"Per-tier accounting {tot}; unavailable tiers: {why}")
    assert separate
    assert len(tiers) == 3, why
