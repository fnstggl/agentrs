"""Budgeted goal-free learning machinery (spec 11.4; AE-04)."""
import pytest

from cam_live.contracts import Cost, GoalStatus, Trust, VerifiedOutcome
from cam_live.reward import integrity as I
from cam_live.reward.selfgoals import GoalFreeBudget, LearningGoalCandidate as C, SelfGoalSelector


def _cands():
    return [C("a", "learn what input a accepts", epistemic=1.0, achievable_progress=0.5, est_cost=Cost(seconds=10)),
            C("b", "learn how panel b responds", epistemic=2.0, achievable_progress=0.5, est_cost=Cost(seconds=10)),
            C("tv", "predict the random banner", epistemic=9.0, achievable_progress=0.0, est_cost=Cost(seconds=1)),
            C("sure", "confirm the known thing", epistemic=1.0, achievable_progress=0.5, p_success=0.999,
              est_cost=Cost(seconds=10)),
            C("big", "long study of c", epistemic=5.0, achievable_progress=1.0, est_cost=Cost(seconds=5000))]


def test_budget_must_be_finite_and_is_enforced():
    with pytest.raises(ValueError):
        GoalFreeBudget(seconds=float("inf"), decisions=10)
    with pytest.raises(ValueError):
        GoalFreeBudget(seconds=10.0, decisions=-1)
    sel = SelfGoalSelector(GoalFreeBudget(seconds=25.0, decisions=10))
    picked = []
    for _ in range(10):
        goals = sel.select(_cands(), k=1)
        if not goals:
            break
        picked.append(goals[0].candidate.key)
        sel.record_result(goals[0], Cost(seconds=10.0), epistemic_after=0.5, realised_progress=0.4)
    assert len(picked) == 2 and sel.budget.spent.seconds == 20.0          # a third 10 s goal does not fit in 25 s
    assert sel.select(_cands(), k=3) == []
    assert "big" not in picked                                            # never affordable
    capped = SelfGoalSelector(GoalFreeBudget(seconds=1e6, decisions=1))
    g = capped.select(_cands())[0]
    capped.record_result(g, Cost(seconds=1.0), epistemic_after=0.1, realised_progress=0.1)
    assert capped.budget.exhausted and capped.select(_cands()) == []
    assert capped.budget.remaining()["decisions"] == 0


def test_goals_are_ranked_by_uncertainty_and_achievable_progress_not_by_unpredictability():
    sel = SelfGoalSelector(GoalFreeBudget(seconds=1000.0, decisions=100))
    goals = sel.select(_cands(), k=5)
    keys = [g.candidate.key for g in goals]
    assert keys[:2] == ["b", "a"]                                # more uncertainty at equal achievable progress
    assert "tv" not in keys                                      # unpredictable but nothing to learn: never chosen
    assert sel.score(_cands()[2]) == 0.0
    assert all(g.origin == Trust.SELF for g in goals)
    sure = [g for g in goals if g.candidate.key == "sure"][0]
    assert sure.score < 0.02 * goals[1].score                    # near-certain success: trivial
    assert sel.select(_cands(), objective_assigned=True) == []   # the mode is off while an objective is assigned


def test_repeating_the_same_self_goal_is_discounted_and_progress_predictions_are_scored():
    sel = SelfGoalSelector(GoalFreeBudget(seconds=1e5, decisions=100))
    one = [C("a", "learn what input a accepts", epistemic=1.0, achievable_progress=0.5)]
    scores = []
    for _ in range(6):
        g = sel.select(one)[0]
        scores.append(g.score)
        sel.record_result(g, Cost(seconds=1.0), epistemic_after=1.0, realised_progress=0.0)
    assert all(b < a for a, b in zip(scores, scores[1:])) and scores[-1] < scores[0] / 30
    assert sel.progress_estimator_error() == pytest.approx(0.5)  # it kept predicting progress that never came
    twin = SelfGoalSelector(GoalFreeBudget(seconds=1e5, decisions=100)); twin.load_state_dict(sel.state_dict())
    assert twin.budget.n_decisions == 6 and twin.score(one[0]) == sel.score(one[0])


def test_usefulness_is_judged_only_on_later_independent_verified_objectives():
    ok = lambda v: v.signature == "signed"
    led = I.OutcomeLedger(ok)
    sel = SelfGoalSelector(GoalFreeBudget(seconds=100.0, decisions=10), min_objectives=3)
    assert sel.usefulness() == {"n": 0, "established": False, "mean_gain": None, "se": None}
    # the outcome of a self-chosen goal cannot enter the ledger, so it cannot count here
    own = VerifiedOutcome(objective_id="self-1", status=GoalStatus.ACHIEVED, value=1.0, signature="signed")
    assert led.admit_verified(own, objective_origin=Trust.SELF).reason == "self_authored_objective"
    gains = [0.3, 0.1, 0.2]
    for i, gain in enumerate(gains):
        # utility = goal-progress level 0.5 * value / target (integrity.objective_utility) = 0.1 + gain
        v = VerifiedOutcome(objective_id=f"later-{i}", status=GoalStatus.PENDING, value=2.0 + 20.0 * gain,
                            target=10.0, evaluated_at=100.0 + i, signature="signed")
        led.register_objective(v.objective_id, origin=Trust.OPERATOR)   # independently assigned objectives
        entry = led.admit_verified(v).entry
        sel.record_later_objective(entry, predicted_without=0.1)
        sel.record_later_objective(entry, predicted_without=0.1)          # duplicate: counted once
        u = sel.usefulness()
        assert u["n"] == i + 1 and u["established"] == (i + 1 >= 3)
    assert u["mean_gain"] == pytest.approx(0.2) and u["se"] == pytest.approx(0.0577, abs=1e-3)
