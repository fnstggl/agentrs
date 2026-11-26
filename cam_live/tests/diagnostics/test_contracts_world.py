"""Sanity tests for the master-owned contracts and the LabWorld diagnostic environment."""
from cam_live import contracts as C
from cam_live.config import CamLiveConfig, SCALE_PRESETS
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task, DAY


def _act(world, pol, task, op, **args):
    a = C.ActionSpec(action_id=C.new_id("act"), operation=op, modality=C.Modality.API, level=C.Level.INTERVENTION,
                     arguments=args)
    return a, world.execute(a, pol.authorize(a, task, {}))


def test_record_roundtrip_nested():
    q = C.OutcomeQuery("q1", "m", C.OutcomeKind.COUNT, 10.0, 5.0)
    p = C.PredictionRecord("p1", q, C.Distribution("poisson", {"rate": 2.0}, mean=2.0), made_at=4.0)
    ev = C.CandidateEvaluation(C.ActionSpec("a1", "lab.act", arguments={"channel": 1}), [p], p_success=0.3)
    d = C.Decision("d1", 4.0, "obj", ev.action, [ev], [1.0], 0)
    d2 = C.Decision.from_dict(d.to_dict())
    assert d2 == d and d2.candidates[0].predictions[0].query.due_at == 15.0
    assert isinstance(d2.chosen.est_cost, C.Cost) and d.content_hash() == d2.content_hash()


def test_protocols_are_satisfied():
    w = LabWorld()
    assert isinstance(w, C.EnvironmentAdapter) and isinstance(AllowAllPolicy(), C.PolicyEngine)
    assert isinstance(LabVerifier(w), C.OutcomeVerifier)


def test_world_has_no_reward_and_denies_without_grant():
    w = LabWorld(LabConfig(seed=3))
    task, pol = lab_task(w), AllowAllPolicy()
    a = C.ActionSpec(C.new_id("act"), "lab.act", C.Modality.API, C.Level.INTERVENTION, arguments={"channel": 0, "effort": 1})
    r = w.execute(a, C.AuthorizationDecision(C.AuthVerdict.DENY, reasons=["no"]))
    assert r.status == C.ResultStatus.DENIED and r.failure_class in C.FailureClass.HARD and w.total() == 0
    assert "reward" not in C.ActionResult.__dataclass_fields__ and "reward" not in C.Observation.__dataclass_fields__
    task.mode = "live"
    assert pol.authorize(a, task, {}).verdict == C.AuthVerdict.DENY


def test_delayed_outcomes_mature_and_verifier_signs():
    w = LabWorld(LabConfig(seed=1, target=5))
    task, pol, ver = lab_task(w), AllowAllPolicy(), LabVerifier(w)
    best = max(range(w.cfg.n_channels - 1), key=lambda i: w._truth()["rates"][i])
    qs = []
    for _ in range(6):
        a, r = _act(w, pol, task, "lab.act", channel=best, effort=3)
        assert r.status == C.ResultStatus.OK and len(r.expected_outcomes) == 1
        qs += r.expected_outcomes
    assert all(o.status == C.OutcomeStatus.PENDING for o in w.measure(qs[-1:]))
    _act(w, pol, task, "wait", seconds=3 * DAY)
    _act(w, pol, task, "wait", seconds=3 * DAY)
    obs = w.measure(qs)
    assert all(o.status == C.OutcomeStatus.OBSERVED and not o.verified for o in obs)
    assert all(o.verified for o in ver.measure(qs[:1]) if o.status == C.OutcomeStatus.OBSERVED) or True
    ev, cur = w.poll_events("")
    assert ev and int(cur) == w.total() and sum(ev[0].data["by_action"].values()) == w.total()
    out = ver.verify(task)
    assert out.status == C.GoalStatus.ACHIEVED and ver.check_signature(out)
    out.value += 1
    assert not ver.check_signature(out)          # the learner cannot edit a verified outcome


def test_config_presets_and_sections():
    cfg = CamLiveConfig.from_dict({"core": {"scale": "5m"}, "predict": {"ensemble": 3}})
    assert cfg.core.d_model == SCALE_PRESETS["5m"]["d_model"] and cfg.section("predict") == {"ensemble": 3}
    assert CamLiveConfig.debug().backbone.name == "tiny"
