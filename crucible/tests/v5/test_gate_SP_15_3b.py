"""SP-15.3b - Time to competence, total action cost, wall-clock, data usage, retention, transfer and sample
efficiency are recorded (MASTER_SPEC Part 15.3; workstream N).

From the learning-curve runs of `instrument.transfer_curves` every run gets its Part 15.3 record
(`curves.efficiency_record`). A quantity counts as recorded when every run carries a measured value for it (time
to competence: the budget at which competence was reached, or an explicit censoring flag): performance before
adaptation, performance by budget, time to competence (interactions and simulated seconds), total action cost,
wall-clock compute, data usage, retention (score on the first task after practising the others), transfer
(trained minus untrained performance before adaptation on the same task) and sample efficiency (area under the
curve).
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curves as CV
from crucible.research.v5 import instrument as I

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
QUANTITIES = ("performance_before_adaptation", "performance_by_budget", "time_to_competence", "total_action_cost",
              "wall_clock", "data_usage", "retention", "transfer", "sample_efficiency")


def test_SP15_3b_efficiency_quantities():
    res = I.transfer_curves()
    unt = {(r["task"], r["learner_seed"]): r for r in res["runs"]["untrained"]}
    ret = {r["learner_seed"]: r["score_after_other_tasks"] for r in res["retention"]}
    recs = []
    for r in res["runs"]["trained"]:
        u = unt[(r["task"], r["learner_seed"])]
        recs.append(CV.efficiency_record(r, q=res["q"], retention=ret[r["learner_seed"]],
                                         transfer=r["curve"][0]["score"] - u["curve"][0]["score"]))
    have = {
        "performance_before_adaptation": all(isinstance(x["performance_before_adaptation"], float) for x in recs),
        "performance_by_budget": all(len(x["performance_by_budget"]) >= 8 for x in recs),
        "time_to_competence": all(x["time_to_competence_censored"] or x["time_to_competence_interactions"]
                                  is not None for x in recs),
        "total_action_cost": all(x["total_action_cost_s"] >= 0 for x in recs),
        "wall_clock": all(x["wall_clock_s"] > 0 and x["cpu_s"] > 0 for x in recs),
        "data_usage": all(x["data_ticks"] > 0 for x in recs),
        "retention": all(isinstance(x["retention"], float) for x in recs),
        "transfer": all(isinstance(x["transfer"], float) for x in recs),
        "sample_efficiency": all(isinstance(x["sample_efficiency_auc"], float) for x in recs)}
    n = sum(have.values())
    record("SP-15.3b", {"efficiency_quantities_recorded": n, "quantities_defined": len(QUANTITIES), "runs": len(recs),
                        "mean_transfer": float(sum(x["transfer"] for x in recs) / len(recs)),
                        "mean_sample_efficiency": float(sum(x["sample_efficiency_auc"] for x in recs) / len(recs))},
           notes=f"Recorded per run: {have}; first record {recs[0]}")
    assert n >= 7 and all(have.values()), have
