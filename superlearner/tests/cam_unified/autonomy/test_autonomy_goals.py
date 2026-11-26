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


def test_curriculum_policy_learns_to_prefer_goals_with_higher_learning_gain():
    """Candidates differ in a descriptor feature; measured gain is high only for one region. Policy gradient on the
    gain moves probability mass there; completion is not part of the signal."""
    import torch
    from superlearner.cam_unified.autonomy.curriculum import CurriculumPolicy, LearningGainPredictor
    torch.manual_seed(0)
    pol = CurriculumPolicy(16, 64)
    opt = torch.optim.Adam(pol.parameters(), 3e-3)
    B, C = 64, 8
    feats = torch.zeros(B, C, 5)
    feats[:, :, 2] = torch.linspace(0, 1, C)       # 'scale' of the goal
    gain_of = lambda idx: torch.exp(-((feats[0, idx, 2] - 0.6) ** 2) / 0.02)   # informative middle difficulty
    mask = torch.ones(B, C, dtype=torch.bool)
    b = torch.randn(B, 16)
    life = torch.zeros(B, 6)
    p0 = pol.choose(b, life, feats, mask).probs.mean(0)
    for _ in range(300):
        ch = pol.choose(b, life, feats, mask)
        L = pol.loss(b, life, feats, mask, ch.index, gain_of(ch.index))
        opt.zero_grad(); L.backward(); opt.step()
    p1 = pol.choose(b, life, feats, mask, greedy=True).probs.mean(0)
    best = int(torch.argmax(gain_of(torch.arange(C))))
    assert float(p1[best]) > 0.5 > float(p0[best])
    lgp = LearningGainPredictor(16, 32)
    mu, sd = lgp(b, feats[:, 0], torch.zeros(B, 4))
    assert mu.shape == (B,) and bool((sd > 0).all())


def test_hindsight_relabels_verified_partial_successes_without_touching_the_original_failure(tmp_path):
    from superlearner.cam_unified.autonomy.hindsight import relabel_episode
    from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore
    gc = GoalCompiler(scales=(1.0, 4.0), horizons=(10.0,))
    for v in (100.0, 110.0, 90.0, 105.0):
        gc.observe([num("rev", v)])
    snaps = [(0.0, [num("rev", 100.0)]), (2.0, [num("rev", 125.0)]), (5.0, [num("rev", 150.0)])]
    labs = relabel_episode("w#0", 3, snaps, original_goal="reach rev 10000", compiler=gc)
    assert labs and all(l.kind == "hindsight_relabel" and l.status == "success" for l in labs)
    assert all(l.meta["original_goal"] == "reach rev 10000" for l in labs)
    exceeded = [l for l in labs if l.meta["text"].startswith("exceed rev")]
    assert exceeded and min(l.t_resolved_days for l in exceeded) == 2.0
    s = LabelStore(str(tmp_path))
    assert s.append(LabelRecord("w#0", 3, "failure", 0.01, 150.0, 10.0, "crucible", goal_id="orig"))
    for l in labs:
        assert s.append(l)
    orig = [r for r in s.iterate() if r.goal_id == "orig"]
    assert len(orig) == 1 and orig[0].status == "failure"
