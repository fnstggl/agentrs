"""SG (amendment item 7): the expected transferable competence of a goal-free learning goal is learned from later
independently assigned objectives (it was the constant 1.0: AE-04 partial, KF-E21).

The evidence is measured, not assumed: an outcome of a later objective is scored by the goal-free posterior and by
the reference prior that posterior started from, on the same data. Debug model + LabWorld (diagnostic scale)."""
import math

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, Level, Modality, OutcomeKind, OutcomeQuery, new_id
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabWorld, lab_task
from cam_live.nn.model import build_model
from cam_live.reward.selfgoals import GoalFreeBudget, GoalFreeMode, LearningGoalCandidate, SelfGoalSelector, \
    TransferEstimator


def _probe(world, ch):
    tools = {t.name: t for t in world.discover_tools()}
    a = ActionSpec(new_id("act"), "lab.probe", Modality.API, Level.INTERVENTION, arguments={"channel": ch},
                   est_cost=tools["lab.probe"].est_cost)
    return a, OutcomeQuery(new_id("q"), f"lab.probe.{ch}", OutcomeKind.REAL, 600.0, description="probe reading")


def _goal_free(model, world, n=24):
    gf = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e7, decisions=n, money=1e3))
    b = torch.zeros(1, model.d_model)
    task, pol = lab_task(world), AllowAllPolicy()
    for _ in range(n):
        goals = gf.propose(b, [_probe(world, ch) for ch in range(world.cfg.n_channels)])
        if not goals:
            break
        g = goals[0]
        res = world.execute(g.action, pol.authorize(g.action, task, {}))
        gf.resolve(g, res.observations[0].quantities[0].value, res.cost)
    return gf, b


def _later(gf, b, world, n=12):
    """Outcomes of a later assigned objective in ``world`` fed to the transfer evidence -> gains."""
    task, pol = lab_task(world), AllowAllPolicy()
    gains = []
    for i in range(n):
        a, q = _probe(world, i % world.cfg.n_channels)
        y = world.execute(a, pol.authorize(a, task, {})).observations[0].quantities[0].value
        phi = gf._features(b, [(a, q)])[0]
        gains.append(gf.observe_assigned(gf.key_of(a, q), phi, q.kind, y, objective_id="later"))
    return gains


def _model():
    torch.manual_seed(0)
    return build_model(CamLiveConfig.debug())


def test_the_transfer_estimate_is_learned_from_later_objectives_not_a_constant():
    model = _model()
    same = LabWorld(LabConfig(n_channels=3, seed=5))
    gf, b = _goal_free(model, same)
    before = {c.key: c.transfer for c, _, _ in gf.candidates(b, [_probe(same, ch) for ch in range(3)])}
    assert set(before.values()) == {0.5}                                # no evidence: the uninformative prior
    # the same mechanism under a later objective: what goal-free learning found predicts it better than the prior
    gains = _later(gf, b, same)
    assert all(g is not None for g in gains) and sum(g > 0 for g in gains) >= 9
    after = {c.key: c.transfer for c, _, _ in gf.candidates(b, [_probe(same, ch) for ch in range(3)])}
    assert all(v > 0.7 for v in after.values()), after

    # goal-free learning in one world, a later objective in a different one (re-drawn mechanism): no transfer
    model2 = _model()
    gf2, b2 = _goal_free(model2, LabWorld(LabConfig(n_channels=3, seed=5)))
    other = LabWorld(LabConfig(n_channels=3, seed=91, w_mean=(-1.0, 0.5, -0.3)))
    gains2 = _later(gf2, b2, other)
    est2 = {c.key: c.transfer for c, _, _ in gf2.candidates(b2, [_probe(other, ch) for ch in range(3)])}
    assert sum(g < 0 for g in gains2) > sum(g > 0 for g in gains2) and sum(gains2) < 0
    assert sum(est2.values()) / len(est2) < 0.5 and min(after.values()) > max(est2.values())
    rep = gf.report()["transfer"]
    assert rep["points"] == len(gains) and set(rep["by_key"]) == set(after)


def test_transfer_changes_which_learning_goal_is_selected():
    """Two candidates equal in uncertainty, achievable progress and cost: the one whose learning was worth more
    later is selected first (the estimate enters the selection score multiplicatively)."""
    te = TransferEstimator()
    for _ in range(6):
        te.observe("op.a|m", [1.0, 0.0], 0.4)
        te.observe("op.b|m", [0.0, 1.0], -0.4)
    ta, tb = te.estimate("op.a|m", [1.0, 0.0])["mean"], te.estimate("op.b|m", [0.0, 1.0])["mean"]
    assert ta > 0.8 and tb < 0.2
    sel = SelfGoalSelector(GoalFreeBudget(seconds=1e5, decisions=10))
    mk = lambda k, t: LearningGoalCandidate(key=k, description=f"learn {k}", epistemic=1.0, achievable_progress=0.5,
                                            transfer=t)
    assert sel.score(mk("op.a|m", ta)) / sel.score(mk("op.b|m", tb)) == pytest.approx(ta / tb)
    picked = sel.select([mk("op.b|m", tb), mk("op.a|m", ta)], k=1)
    assert picked[0].candidate.key == "op.a|m"
    # similar contexts share evidence; unrelated ones keep the prior
    assert te.estimate("op.c|other", [0.99, 0.141])["mean"] > 0.7
    assert te.estimate("op.z|q", [-1.0, 0.0])["mean"] == pytest.approx(0.5, abs=0.01)
    assert te.estimate("op.a|other")["mean"] == pytest.approx(0.5)              # no phi: only the same key counts


def test_transfer_evidence_persists_and_old_checkpoints_load():
    model = _model()
    world = LabWorld(LabConfig(n_channels=3, seed=5))
    gf, b = _goal_free(model, world, n=10)
    _later(gf, b, world, n=6)
    sd = gf.state_dict()
    twin = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e7, decisions=10, money=1e3))
    twin.load_state_dict(sd)
    twin.fast.load_state_dict(gf.fast.state_dict())
    twin.after_fast_restore()
    k = gf.key_of(*_probe(world, 0))
    assert twin.transfer.estimate(k)["mean"] == pytest.approx(gf.transfer.estimate(k)["mean"])
    a, q = _probe(world, 1)
    phi = gf._features(b, [(a, q)])[0]
    assert twin.reference.nll(phi, 1.0, q.kind) == pytest.approx(float(gf.reference.nll(phi, 1.0, q.kind)))
    # a checkpoint written before SG: no transfer evidence, no reference -> fresh, and said so
    old = {k2: v for k2, v in sd.items() if k2 not in ("transfer", "reference")}
    legacy = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e7, decisions=10, money=1e3))
    legacy.load_state_dict(old)
    legacy.fast.load_state_dict(gf.fast.state_dict())
    legacy.after_fast_restore()
    assert legacy.transfer.points == [] and len(legacy.restore_notes) == 2
    assert legacy.observe_assigned(k, phi, q.kind, 1.0) == pytest.approx(0.0, abs=1e-9)   # reference = restored state


def test_nothing_is_said_without_goal_free_learning_or_for_unmodelled_kinds():
    model = _model()
    gf = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e5, decisions=5))
    world = LabWorld(LabConfig(n_channels=3, seed=5))
    a, q = _probe(world, 0)
    phi = gf._features(torch.zeros(1, model.d_model), [(a, q)])[0]
    assert gf.observe_assigned(gf.key_of(a, q), phi, q.kind, 1.0) is None      # no goal-free learning yet
    assert gf.transfer.points == []
    te = TransferEstimator()
    assert te.observe("k", [1.0], 0.0) is False and te.observe("k", [1.0], float("nan")) is False
    assert te.estimate("k", [1.0]) == {"mean": 0.5, "n_eff": 0.0, "mean_gain": None,
                                       "sd": pytest.approx(math.sqrt(0.25 / 3.0))}
