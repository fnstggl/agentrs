"""Self-generated goals are grounded in observable evidence and verified operator-side (AU-02, AU-06)."""
from superlearner.cam_unified.autonomy.goals import (FAILURE, PENDING, SUCCESS, TRIVIAL, UNVERIFIABLE, GoalCompiler,
                                                     GoalPredicate, SelfGoal, SelfGoalVerifier)
from superlearner.cam_unified.interfaces import ObsKind, Observation


def num(path, v, t=0.0):
    return Observation(ObsKind.NUMERIC, value=v, source=path, t=t)


def test_candidates_come_from_observed_quantities_and_are_not_trivially_true():
    gc = GoalCompiler()
    obs = [num("bank.balances.a1", 1000.0), num("rig.r1.readings.x", 5.0)]
    for k in range(5):
        gc.observe([num("bank.balances.a1", 1000.0 + 10 * k), num("rig.r1.readings.x", 5.0 + 0.1 * k)])
    cands = gc.candidates(obs, 0.0)
    assert cands and {p.path for g in cands for p in g.predicates} == {"bank.balances.a1", "rig.r1.readings.x"}
    v = SelfGoalVerifier()
    assert all(v.register(g, obs, 0.0) == PENDING for g in cands)


def test_trivial_unverifiable_and_duplicate_goals_earn_nothing():
    v = SelfGoalVerifier()
    obs = [num("q", 10.0)]
    triv = SelfGoal("g1", "reach q 5", [GoalPredicate("q", "reach", 5.0)], 1.0)
    assert v.register(triv, obs, 0.0) == TRIVIAL
    gc = GoalCompiler()
    bad = gc.compile_text("make lots of money soon", 0.0)
    assert not bad.verifiable and v.register(bad, obs, 0.0) == UNVERIFIABLE
    g = gc.compile_text("exceed q 20 within 2 days", 0.0)
    assert g.verifiable and v.register(g, obs, 0.0) == PENDING
    g2 = gc.compile_text("exceed q 20 within 2 days", 0.0)
    assert v.register(g2, obs, 0.0).startswith("duplicate")


def test_verifier_resolves_on_world_observations_only_at_or_before_deadline():
    v = SelfGoalVerifier()
    gc = GoalCompiler()
    up = gc.compile_text("exceed q 20 within 2 days", 0.0)
    down = gc.compile_text("reduce q 1 within 1 days", 0.0)
    keep = gc.compile_text("maintain q 5 within 3 days", 0.0)
    for g in (up, down, keep):
        v.register(g, [num("q", 10.0)], 0.0)
    assert v.check([num("q", 15.0)], 0.5) == []
    res = dict((gid, st) for gid, st, _ in v.check([num("q", 25.0)], 1.0))
    assert res[up.goal_id] == SUCCESS and res[down.goal_id] == FAILURE
    assert v.status(keep.goal_id) == PENDING
    v.check([num("q", 6.0)], 2.0)
    res = dict((gid, st) for gid, st, _ in v.check([num("q", 7.0)], 3.0))
    assert res[keep.goal_id] == SUCCESS
