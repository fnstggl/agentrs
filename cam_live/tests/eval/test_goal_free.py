"""X5 driver (Protocol v3.4): a goal-free period is a task without a success criterion on the same computer; its
verifier checks nothing (signed pending verdicts, unobservable measurements); the X5 plan puts it before X1's plan
without shifting the world seeds of X1's objectives."""
from __future__ import annotations

from cam_live.contracts import GoalStatus, OutcomeQuery, OutcomeStatus, SimClock
from cam_live.eval import oe_batch
from cam_live.eval.openended import NoCriterionVerifier, goal_free_task


def test_goal_free_task_has_no_success_criterion_and_its_verifier_checks_nothing():
    t = goal_free_task("obj_gf", max_decisions=20)
    assert t.success == {} and t.limits.max_decisions == 20 and "code" in t.interfaces
    v = NoCriterionVerifier(SimClock())
    o = v.verify(t)
    assert o.status == GoalStatus.PENDING and o.value == 0.0 and v.check_signature(o)
    o.value = 1.0                                                   # a changed verdict no longer verifies
    assert not v.check_signature(o)
    q = OutcomeQuery(query_id="q1", metric="sw.pass_fraction")
    assert [x.status for x in v.measure([q])] == [OutcomeStatus.UNOBSERVABLE]


def test_x5_plan_is_the_goal_free_period_then_x1s_plan():
    jobs = oe_batch.PLANS["v3x5"]()
    assert [j["name"] for j in jobs] == ["x5/goalfree_s0", "x5/goalfree_s1"]
    assert all(j["plan"] == "GOALFREE:sw," + oe_batch.SW_TRAIN for j in jobs)
    x1 = {j["seed"]: j for j in oe_batch.PLANS["v3"]() if j["name"].startswith("x1/")}
    assert all(j["plan"].split(",", 1)[1] == x1[j["seed"]]["plan"] for j in jobs)
