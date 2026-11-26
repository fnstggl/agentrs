"""SP-11.6b - Reports separate algorithmic limitations from environment failures (MASTER_SPEC Part 11.6;
workstream N).

The research harness runs real protocol steps that fail for different reasons and classifies every failure with
`integrity.classify_failure`; a failure the classifier cannot attribute is reported as ``unclassified`` and counts
against this requirement:

    algorithmic_limitation   a cannot-learn control practising an unfamiliar task never reaches the competence
                             threshold (a completed, censored run)
    environment_failure      a Tier 2 or Tier 3 world that this checkout cannot build (`TierUnavailable` from the
                             experience registry; Level 5 itself executes in U.1 - running it here on CPU timed out)
    interface_failure        a lever learner given a world that offers no continuous lever interface
    partition_refusal        a training request for an evaluation item
    infrastructure_failure   a request to an evaluator process after its channel closed
"""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curves as CV
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import learners as LR
from crucible.research.v5 import sampler as S
from crucible.research.v5.episodes import lever_entries, run_episode
from crucible.research.v5.integrity import FAILURE_CAUSES, classify_failure

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def test_SP11_6b_failure_causes_classified():
    reg = D.build_registry()
    reports = []
    fam = lever_entries(reg, "train")[0]
    run = CV.practice_curve(reg, {"entry": fam, "seed": 1_000_961}, LR.make("hold", 0), budgets=(0, 1, 2, 4),
                            rng=np.random.default_rng(0), q=0.3)
    reports.append({"case": "cannot_learn_practice", **classify_failure(
        {"completed": True, "censored": run["censored"], "score": run["curve"][-1]["score"], "q": 0.3})})
    from crucible.experience import make
    try:
        make(2, "no_such_domain:none", 0)
    except Exception as e:  # noqa: BLE001 - the failure under test
        reports.append({"case": "tier2_world", **classify_failure(e)})
    no_levers = [w["id"] for w in reg.worlds(tier=1, family_partition="train", source="crucible.rapid")
                 if "levers" not in w["action_kinds"]][0]
    try:
        run_episode(S.make_env(S.episode_spec(reg, no_levers, 5, level="0", objective={"mode": "assigned"}), reg),
                    LR.make("adapter", 0), learn=True, rng=np.random.default_rng(0))
    except Exception as e:  # noqa: BLE001
        reports.append({"case": "no_lever_interface", **classify_failure(e)})
    ev_entry = reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")[0]["id"]
    try:
        S.make_env(S.episode_spec(reg, ev_entry, 5), reg)
    except Exception as e:  # noqa: BLE001
        reports.append({"case": "evaluation_item_for_training", **classify_failure(e)})
    ev = EV.Evaluator.start()
    ev.close()
    try:
        ev.must("ping")
    except Exception as e:  # noqa: BLE001
        reports.append({"case": "closed_evaluator", **classify_failure(e)})
    try:                                  # a Tier 3 world this checkout cannot build (Level 5 itself runs in U.1)
        make(3, "no_such_profile", 0)
    except Exception as e:  # noqa: BLE001
        reports.append({"case": "tier3_world", **classify_failure(e)})
    causes = {r["case"]: r["cause"] for r in reports}
    unclassified = sum(1 for r in reports if r["cause"] not in FAILURE_CAUSES)
    expected = {"cannot_learn_practice": "algorithmic_limitation", "tier2_world": "environment_failure",
                "tier3_world": "environment_failure",
                "no_lever_interface": "interface_failure", "evaluation_item_for_training": "partition_refusal",
                "closed_evaluator": "infrastructure_failure"}
    right = sum(1 for k, v in expected.items() if causes.get(k) == v)
    record("SP-11.6b", {"failure_reports_with_cause_classification": len(reports) - unclassified,
                        "unclassified_failures": unclassified, "expected_causes_matched": right,
                        "distinct_causes": len(set(causes.values()))},
           notes=f"Failure reports: {reports}")
    assert unclassified == 0 and right == len(expected), causes
