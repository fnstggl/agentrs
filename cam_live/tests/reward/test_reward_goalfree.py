"""Goal-free autonomous learning on the real (debug) model in LabWorld (spec 11.4)."""
import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, Cost, GoalStatus, Level, Modality, OutcomeKind, OutcomeQuery, Trust, new_id
from cam_live.diagnostics.collect import LabCollector, uniform_policy, verified_return
from cam_live.diagnostics.memstore import MemoryStore
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.nn.model import build_model
from cam_live.reward.integrity import OutcomeLedger
from cam_live.reward.selfgoals import GoalFreeBudget, GoalFreeMode


def _pairs(world):
    tools = {t.name: t for t in world.discover_tools()}
    out = []
    for ch in range(world.cfg.n_channels):
        a = ActionSpec(new_id("act"), "lab.probe", Modality.API, Level.INTERVENTION, arguments={"channel": ch},
                       est_cost=tools["lab.probe"].est_cost)
        out.append((a, OutcomeQuery(new_id("q"), f"lab.probe.{ch}", OutcomeKind.REAL, 600.0,
                                    description="reading returned by the probe")))
    a = ActionSpec(new_id("act"), "lab.noise", Modality.API, Level.INTERVENTION, est_cost=tools["lab.noise"].est_cost)
    out.append((a, OutcomeQuery(new_id("q"), "lab.static", OutcomeKind.REAL, 600.0, description="static gauge reading")))
    return out


def _run(budget, max_steps=120, seed=5):
    torch.manual_seed(0)
    model = build_model(CamLiveConfig.debug())
    world = LabWorld(LabConfig(n_channels=4, seed=seed))
    task, policy = lab_task(world), AllowAllPolicy()
    gf = GoalFreeMode(model, model.new_fast_state(), budget)
    b = torch.zeros(1, model.d_model)
    picks = []
    for _ in range(max_steps):
        goals = gf.propose(b, _pairs(world))
        if not goals:
            break
        g = goals[0]
        res = world.execute(g.action, policy.authorize(g.action, task, {}))     # still authorized like any action
        gf.resolve(g, res.observations[0].quantities[0].value, res.cost)
        picks.append(g)
    return model, world, gf, picks, b


def test_goal_free_mode_is_budgeted_and_stops_when_nothing_is_left_to_learn():
    model, world, gf, picks, b = _run(GoalFreeBudget(seconds=80 * 600.0, decisions=80, money=12.0))
    assert 10 < len(picks) < 80                                   # went idle by itself, budget left over
    assert not gf.budget.exhausted and gf.propose(b, _pairs(world)) == []
    assert gf.budget.n_decisions == len(picks) and gf.budget.spent.money <= 12.0 + 1e-9
    assert all(g.origin == Trust.SELF and g.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED) for g in picks)
    assert sum(g.status == GoalStatus.ACHIEVED for g in picks) >= 5
    gauge = [i for i, g in enumerate(picks) if g.action.operation == "lab.noise"]
    assert len(gauge) <= len(picks) // 3                          # the static gauge does not keep the mode busy
    assert not [i for i in gauge if i >= len(picks) - 15]         # and is never chosen once it stopped teaching
    assert gf.calibration["lab.noise|lab.static"] < 0.6
    rep = gf.report()
    assert rep["resolved"] == len(picks) and rep["usefulness"]["established"] is False
    assert gf.propose(b, _pairs(world), objective_assigned=True) == []       # the mode is off under an objective


def test_fixed_budget_is_never_exceeded():
    with pytest.raises(ValueError):
        GoalFreeBudget(seconds=float("inf"), decisions=10)
    model, world, gf, picks, b = _run(GoalFreeBudget(seconds=7 * 600.0, decisions=100, money=0.7))
    assert gf.budget.spent.money <= 0.7 + 1e-9 and gf.budget.spent.seconds <= 7 * 600.0 + 1e-9
    assert sum(g.action.operation == "lab.probe" for g in picks) <= 2        # 0.3 per probe, 0.7 in total
    assert len(picks) <= 7 and gf.propose(b, _pairs(world)) == []


def test_achievement_is_decided_by_observation_only():
    torch.manual_seed(0)
    model = build_model(CamLiveConfig.debug())
    world = LabWorld(LabConfig(n_channels=4, seed=1))
    gf = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e5, decisions=20, money=5.0))
    b = torch.zeros(1, model.d_model)
    g = gf.propose(b, _pairs(world))[0]
    assert g.status == GoalStatus.PENDING and g.predicted_progress > 0 and g.required_progress > 0
    gf.resolve(g, None, Cost(seconds=600.0, money=0.3))           # executed, nothing observed
    assert g.status == GoalStatus.FAILED and g.realised_progress == 0.0
    assert gf.budget.n_decisions == 1 and gf.budget.spent.money == pytest.approx(0.3)   # charged all the same
    with pytest.raises(ValueError):
        gf.resolve(g, 1.0, Cost())
    g2 = gf.propose(b, _pairs(world))[0]
    gf.resolve(g2, 2.5, Cost(seconds=600.0, money=0.3))
    assert g2.status == GoalStatus.ACHIEVED and g2.realised_progress >= g2.required_progress
    assert g2.epistemic_after < g2.epistemic_before
    sd = gf.state_dict()
    twin = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e5, decisions=20, money=5.0))
    twin.load_state_dict(sd)
    assert twin.budget.n_decisions == 2 and twin.calibration == gf.calibration and twin.achieved == gf.achieved


def test_self_goals_never_supervise_and_usefulness_needs_later_independent_objectives():
    model, world, gf, picks, b = _run(GoalFreeBudget(seconds=20 * 600.0, decisions=20, money=3.0))
    store = MemoryStore()
    worlds = [LabWorld(LabConfig(n_channels=4, seed=50 + i, horizon_s=86400.0, target=10)) for i in range(5)]
    key = LabVerifier(worlds[0])
    ledger = OutcomeLedger(key.check_signature)
    lts = [LabCollector(store).run(w, uniform_policy(i), 8, seed=i) for i, w in enumerate(worlds)]
    as_self = ledger.admit_verified(lts[0].verified, objective_origin=Trust.SELF)
    assert not as_self.accepted and as_self.reason == "self_authored_objective" and len(ledger) == 0
    for i, lt in enumerate(lts):
        ledger.register_objective(lt.task)                            # operator tasks (TaskSpec.origin)
        entry = ledger.admit_verified(lt.verified, utility=verified_return(lt)).entry
        gf.record_later_objective(entry, predicted_without=0.2)
        assert gf.usefulness()["established"] is (i + 1 >= 5)
    u = gf.usefulness()
    assert u["n"] == 5 and u["se"] is not None
    assert u["mean_gain"] == pytest.approx(sum(verified_return(lt) for lt in lts) / 5 - 0.2)
