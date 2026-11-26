"""SP-15.5 - Executable A/B/C/D training-diversity comparison with matched experience and compute budgets and the
seven listed controls (MASTER_SPEC Part 15.5; workstream N).

`comparison.compare` trains one identical learner (the architecture-neutral `crucible.experience.ReferenceLearner`,
which acts in every tier - the lever learners act only in Tier 1) with three learner seeds on each arm under a
matched experience budget and a matched compute budget, and scores every arm on the same held-out worlds; an arm
whose tiers do not exist in this checkout is reported as not executed (never replaced). `comparison.run_controls`
runs the seven control kinds on the same worlds. Nothing assumes that D beats A.
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import comparison as CMP

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]


def test_SP15_5_abcd_comparison_and_controls():
    exp = CMP.compare(CMP.ARMS, learner=CMP.NEUTRAL, mode="experience", budget=2000, seeds=(0, 1, 2))
    cmp_ = CMP.compare(CMP.ARMS, learner=CMP.NEUTRAL, mode="compute", budget=20.0, seeds=(0, 1, 2))
    arms = [a for a in CMP.ARMS if exp["arms"][a]["executed"] and cmp_["arms"][a]["executed"]]
    ctl = CMP.run_controls(seeds=(0, 1, 2))
    kinds = [k for k, v in ctl["kinds"].items() if v["executed"]]
    record("SP-15.5", {"comparison_arms": len(arms), "arms_defined": len(CMP.ARMS), "control_kinds": len(kinds),
                       "budget_modes": 2, "arm_A_mean": float(exp["arms"]["A"].get("mean_score", 0.0)),
                       "arm_B_mean": float(exp["arms"]["B"].get("mean_score", 0.0))},
           notes=f"Arms executed: {arms}; not executed: "
                 f"{({a: exp['arms'][a].get('why') for a in CMP.ARMS if not exp['arms'][a]['executed']})}; "
                 f"differences (experience) {exp['differences']}; controls {ctl['kinds']}")
    assert len(kinds) >= 7, ctl["kinds"]
    assert len(arms) >= 4
