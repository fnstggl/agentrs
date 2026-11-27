"""W2: verified labels through `interfaces.LabelSource.take_labels()` (ExperienceOperator, OpenWorldEvaluator and its
process-isolated proxy): emitted only at verification time, exactly once, with the actor-visible opaque env id and
goal id, never pending, never private; the RewardGate gives the actor 0.0 in observation_only while the operator's
legacy record still accumulates (audit actions: observation_only take_rewards == 0 on an ExampleInductor run that
reaches success on day ~4; take_labels contains exactly one terminal_success whose time equals the verifier's)."""
from __future__ import annotations

import json
import os
import sys

import pytest

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

from superlearner.cam_unified.envs import experience as X  # noqa: E402
from superlearner.cam_unified.envs.factory import make_env  # noqa: E402
from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore, drain  # noqa: E402
from superlearner.cam_unified.interfaces import Action, LabelSource  # noqa: E402

DAY = 86400.0


def _inductor_run(reward_mode, allow, days=8, seed=0):
    from crucible.usefulness.agents import ExampleInductor
    from superlearner.cam_unified.envs.agent_protocol import PolicyAgent
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    es = make_env("openworld", tasks=[open_economy_task(seed=1, days=days)], isolated=False, reward_mode=reward_mode,
                  allow_reward_modes=allow, seed=seed, guard=_permissive())
    ad, op, ev = es
    steps = []

    def on_step(k, tr):
        steps.append((list(tr.rewards), float(tr.next_obs.t[0]), ev.take_labels()))
    res = run_lifetime(PolicyAgent(lambda b: ExampleInductor()), ad, ev, think_s=600, on_step=on_step)
    out = dict(res=res, steps=steps, env_id=op.envs[0].env_id, goal_id=op.envs[0].goal.goal_id, t0=op.envs[0].t0,
               success_t=ev.success_t[0], total=ev.total[0], ev=ev, op=op)
    es.close()
    return out


def _permissive():
    from superlearner.cam_unified.envs.factory import default_guard
    return default_guard()


def test_label_sources_implement_the_protocol():
    op = X.ExperienceOperator([X.WorldSpec(1, "rapid:dl_delayed_returns", 0)], compute_references=False)
    assert isinstance(op, LabelSource)
    from superlearner.cam_unified.envs.openworld import RemoteEvaluator
    from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator
    assert hasattr(OpenWorldEvaluator, "take_labels") and hasattr(RemoteEvaluator, "take_labels")


def test_experience_labels_once_after_resolution_with_opaque_ids(tmp_path):
    es = make_env("rapid", ["dl_delayed_returns"])
    ad, op, src = es
    assert src is op and isinstance(src, LabelSource)
    b = ad.reset()
    eid = b.env_ids[0]
    assert "dl_delayed_returns" not in eid and "#" not in eid and eid.startswith("x0-")
    assert src.take_labels() == []                       # pending: nothing (pending is never a label)
    tool = ad.available_tools()[0][0].name
    for _ in range(200):
        ad.execute([Action(tool=tool, args={"kind": "levers", "values": {}})])
        ad.advance_time([1 / 24])
        if op.resolved(0):
            break
    assert op.resolved(0)
    (lab,) = src.take_labels()
    assert src.take_labels() == []                       # exactly once
    assert isinstance(lab, LabelRecord) and lab.status in ("success", "failure") and lab.utility is not None
    assert lab.env_id == eid and lab.goal_id == ad.goals()[0].goal_id and lab.kind == "terminal"
    assert lab.t_end_days is not None and lab.stream == "train"
    assert op.private_verify(0)["name_id"] == "rapid:dl_delayed_returns#0"     # the mapping stays operator-side
    store = LabelStore(str(tmp_path))
    assert store.append(lab) and not store.append(lab)
    assert [r.env_id for r in LabelStore(str(tmp_path)).iterate()] == [eid]


def test_openworld_observation_only_inductor_rewards_zero_and_one_terminal_success_label(tmp_path):
    run = _inductor_run("observation_only", ("observation_only",))
    res = run["res"]
    assert res.final[0]["achieved"] is True and run["success_t"] is not None
    assert all(r == 0.0 for rw, _, _ in run["steps"] for r in rw)        # the actor never receives a reward
    assert res.returns[0] == 0.0
    assert run["total"] == pytest.approx(2.0, abs=1e-6)                    # the operator's legacy record still runs
    labels = [(t, lab) for _, t, labs in run["steps"] for lab in labs]
    assert len(labels) == 1
    t_rel, lab = labels[0]
    assert lab.status == "success" and lab.utility == 1.0 and lab.kind == "terminal"
    assert lab.t_resolved_days == pytest.approx((run["success_t"] - run["t0"]) / DAY)
    assert t_rel >= run["success_t"] / DAY - 1e-9                          # released at / after verification
    assert lab.env_id == run["env_id"] and lab.goal_id == run["goal_id"]
    assert lab.meta["verified"]["independent_revenue"] >= 10_000 and lab.value is not None
    assert lab.t_start_days == pytest.approx(run["t0"] / DAY) and lab.t_end_days >= lab.t_resolved_days
    store = LabelStore(str(tmp_path))
    assert store.append(lab)                                              # no private key in the label
    assert run["ev"].take_labels() == []


def test_openworld_verified_mode_pays_one_once():
    run = _inductor_run("verified", ("verified",), days=6)
    assert run["res"].final[0]["achieved"] is True
    assert sum(r for rw, _, _ in run["steps"] for r in rw) == pytest.approx(1.0)
    assert sum(len(labs) for _, _, labs in run["steps"]) == 1


def test_openworld_failure_label_only_at_the_horizon_and_through_the_process_boundary():
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    es = make_env("openworld", tasks=[open_economy_task(seed=100_003, days=1)], isolated=True)
    ad, rop, rev = es
    try:
        assert rev.reward_mode == "observation_only"
        ad.reset()
        got = []
        for k in range(30):
            ad.execute([Action("wait", {"seconds": 3600})])
            nb = ad.advance_time([3600 / DAY])
            got += [(nb.t[0], lab) for lab in rev.take_labels()]
            assert rev.take_rewards() == [0.0]
            if nb.done[0]:
                break
        assert len(got) == 1
        t, lab = got[0]
        assert lab.status == "failure" and lab.utility == 0.0 and lab.t_resolved_days == pytest.approx(1.0)
        assert t - lab.t_start_days >= 1.0 - 1e-9
        assert lab.meta["resources"]["waits"] >= 23
    finally:
        es.close()


def test_predicate_verification_and_public_result_carry_verdicts_only():
    es = make_env("openworld", tasks=["open_economy#100005"], isolated=True)
    ad, rop, rev = es
    try:
        ad.reset()
        preds = [{"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 1_000_000},
                 {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": 10 ** 12},
                 {"metric": "no_such_metric", "args": {}, "op": ">=", "value": 1}]
        ok, high, bad = rev.verify_predicates(0, preds)
        assert ok["satisfied"] is True and ok["values"]["cash"] == pytest.approx(3_000_000)
        assert high["satisfied"] is False and "error" in bad and not bad["satisfied"]
        pub = rev.public_result(0)
        full = rev.evaluate_goal(0)
        assert pub.details == {} and pub.achieved is False and pub.score == 0.0 and full.details
        assert "independent_revenue" in pub.metrics
        assert "ow_tiny" not in json.dumps(pub.to_dict()) and "ow_tiny" in json.dumps(full.to_dict())
    finally:
        es.close()


def test_eval_stream_labels_are_refused_by_a_training_store(tmp_path):
    es = make_env("rapid", ["srch_distractors"], split="val", compute_references=False)
    ad, op, src = es
    ad.reset()
    tool = ad.available_tools()[0][0].name
    labs = []
    for _ in range(400):
        ad.execute([Action(tool=tool, args={})])
        ad.advance_time([1 / 24])
        labs += src.take_labels()
        if op.resolved(0):
            break
    labs += src.take_labels()
    assert len(labs) == 1 and labs[0].stream == "eval"
    with pytest.raises(ValueError):
        LabelStore(str(tmp_path / "train")).append(labs[0])
    assert LabelStore(str(tmp_path / "eval"), accept_eval=True).append(labs[0])
    with pytest.raises(ValueError):
        drain(_Src(labs), LabelStore(str(tmp_path / "train2")))


class _Src:
    def __init__(self, labs):
        self.labs = labs

    def take_labels(self):
        out, self.labs = self.labs, []
        return out


def test_label_store_rejects_pending_private_unknown_kinds_and_reads_old_lines(tmp_path):
    s = LabelStore(str(tmp_path))
    with pytest.raises(ValueError):
        s.append(LabelRecord("x0-a", 1, "pending", None, None, None, "e"))
    with pytest.raises(ValueError):
        s.append(LabelRecord("x0-a", 1, "success", 1.0, 1.0, 1.0, "e", meta={"verified": {"progress": 0.5}}))
    with pytest.raises(ValueError):
        s.append(LabelRecord("x0-a", 1, "success", 1.0, 1.0, 1.0, "e", kind="whatever"))
    with open(os.path.join(str(tmp_path), "labels.jsonl"), "a") as fh:     # a pre-W2 line (no t_start/t_end)
        fh.write('{"env_id": "w#0", "episode": 2, "status": "failure", "utility": 0.0, "value": 1.0, '
                 '"t_resolved_days": 3.0, "evaluator": "e", "goal_id": "", "actor_id": "", "t_release_wall": 0.0, '
                 '"kind": "terminal", "meta": {}}\n')
    (r,) = list(LabelStore(str(tmp_path)).iterate())
    assert r.env_id == "w#0" and r.t_start_days is None


def test_capability_score_evaluator_gates_rewards_and_releases_episode_labels():
    from superlearner.cam_unified.capabilities.envs import ProbeEnv, ScoreEvaluator
    es = make_env("capability", ["probe"], n_envs=2, seed=1, horizon=2)
    env, _, ev = es
    assert isinstance(ev, LabelSource)
    env.reset()
    env.execute([Action("run_test", {}), Action("play_safe", {})])
    assert ev.take_rewards() == [0.0, 0.0] and ev.take_labels() == []
    env.execute([Action("play_safe", {}), Action("play_safe", {})])
    assert ev.take_rewards() == [0.0, 0.0]
    labs = ev.take_labels()
    assert len(labs) == 2 and ev.take_labels() == []
    null, best = env.utility_bounds()
    for b, lab in enumerate(labs):
        assert lab.value == pytest.approx(env.returns[b])
        assert lab.utility == pytest.approx((env.returns[b] - null) / (best - null))
        assert lab.status in ("success", "failure") and lab.env_id == env.env_ids[b]
    leg_env = ProbeEnv(1, seed=1, horizon=2)
    leg = ScoreEvaluator(leg_env)                                   # legacy: the increments, as before
    leg_env.reset()
    leg_env.execute([Action("play_safe", {})])
    assert leg.take_rewards() == [pytest.approx(leg_env.returns[0])] and leg.take_rewards() == [0.0]
    leg_env.reset()
    assert leg.take_rewards() == [0.0]
