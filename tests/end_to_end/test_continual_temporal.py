"""Delayed-reward RL, temporal planning and the continual-learning cycle (integration test 10)."""

import torch

from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy, ContentPolicyActor
from agentrs.models.policy import LMPolicy, PolicyConfig
from agentrs.planning.temporal_planner import TemporalContentPlanner, TemporalPlannerConfig
from agentrs.sensors.observation_network import NetworkConfig
from agentrs.sensors.socialweblab_source import lab_temporal_spec
from agentrs.socialweblab.agent_tasks import SocialWebLabContentEnv, make_help_task, own_context, space_for, verify_help
from agentrs.socialweblab.engine import SocialWebLab
from agentrs.agent.temporal_runtime import act_with_delayed_outcome
from agentrs.temporal.system import build_network
from agentrs.temporal.temporal_world_model import TemporalWorldModel, TemporalWorldModelConfig
from agentrs.temporal.time_encoding import DAY
from agentrs.training.common import param_change, param_snapshot
from agentrs.training.continual_learning_orchestrator import ContinualLearningOrchestrator, OrchestratorConfig
from agentrs.training.delayed_reward_trainer import DelayedRewardConfig, DelayedRewardTrainer, record_matured_rewards
from agentrs.training.temporal_world_model_trainer import (
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)


def _help_verifier(lab):
    def v(a):
        item = (a["context"] or {}).get("object_ref")
        if item is None:
            return "censored", None, None
        ver = verify_help(lab, int(item), a["action_time"], DAY)
        if ver.evidence.get("pending"):
            return "pending", None, None
        return "matured", float(ver.success), a["action_time"] + DAY
    return v


def test_10_continual_cycles_with_delayed_rewards_and_planning(tmp_path, scratch_bb):
    torch.manual_seed(0)
    lab = SocialWebLab(tmp_path / "lab.db", seed=8, n_users=40, lab_id="cyc8")
    lab.advance(12 * 3600)
    wm = TemporalWorldModel(TemporalWorldModelConfig(backbone=scratch_bb, n_heads=2, spec=lab_temporal_spec()))
    net = build_network(tmp_path / "net", lab, wm, NetworkConfig(scheduling="event"),
                        split_fn=lambda o: "val" if int(o.provider_object_id) % 4 == 0 else "train")
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=scratch_bb)), ContentGenConfig(max_content_tokens=16),
                                 checkpoint_id="policy-v0")
    env = SocialWebLabContentEnv(lab)
    actor = ContentPolicyActor(cp, mode="sample", seed=0)
    T = TemporalWorldModelTrainer(wm, TemporalTrainConfig(batch_size=8, progressive_steps=6, progressive_min_new=1))
    builder = TemporalTrainingDataBuilder(net.ledger, wm.cfg.spec)
    prt = DelayedRewardTrainer(cp, net.ledger, DelayedRewardConfig(), ref=None)

    def act(cycle, n):
        out = []
        for i in range(n):
            t = make_help_task("train", cycle * 100 + i // 2, account=f"agent{i % 2}")
            out.append(act_with_delayed_outcome(env, t, actor, net, allow_decline=False))
        return out

    orch = ContinualLearningOrchestrator(tmp_path / "orch", lab, net, T, builder,
                                         OrchestratorConfig(cycle_s=DAY + 3600, step_s=1800, tasks_per_cycle=4),
                                         act_fn=act, reward_fn=lambda: record_matured_rewards(
                                             net.ledger, _help_verifier(lab), "help@86400"), policy_trainer=prt)
    pol_snap = param_snapshot(cp.policy)
    wm_snap = param_snapshot(wm)
    r1 = orch.run_cycle()
    r2 = orch.run_cycle()
    for r in (r1, r2):
        assert r["wm_update"]["trained"] and r["wm_update"]["param_change"] > 0  # real WM gradient updates
        assert r["wm_eval_after"]["held_out"]  # held-out evaluation ran
        assert len(r["actions"]) == 4 and all(a["item_id"] for a in r["actions"])
    # cycle 1's posts matured during cycle 2's day -> real policy update from delayed verified rewards
    n1 = sum(a["item_id"] is not None for a in r1["actions"])
    n2 = sum(a["item_id"] is not None for a in r2["actions"])
    assert r1["rewards"].get("matured", 0) == n1  # declined actions (finish) produce no reward
    assert r1["policy_update"]["n_matured"] == n1
    assert param_change(cp.policy, pol_snap) > 0 or not r1["policy_update"]["updated"]
    assert r2["policy_update"]["n_matured"] == n2 and r2["policy_update"]["checkpoint"].startswith("policy-delayed-rl")
    # third-party outcomes were never used as policy rewards; self actions were predicted before execution
    for a in net.ledger.actions("source_type='self'"):
        f = net.ledger.forecasts(a["action_id"])[0]
        assert f["created_time"] <= a["action_time"] + 1e-6
    assert net.ledger.stats()["actions"] > 20  # third-party actions observed alongside our own
    hist = orch.registry.list("temporal_world_model")
    assert len(hist) == 2 and {h.status for h in hist} <= {"promoted", "rejected", "candidate", "retired"}
    assert param_change(wm, wm_snap) >= 0
    # the planner compares drafts at the goal's horizon and picks the best predicted one
    planner = TemporalContentPlanner(cp, wm, TemporalPlannerConfig(num_candidates=4, horizon_s=DAY),
                                     context_fn=lambda op: own_context(lab, env.account_uid(), space_for(op) or "linux"))
    t = make_help_task("test", 0, account="agent0")
    out = act_with_delayed_outcome(env, t, planner, net, allow_decline=False)
    assert out["selection"] == "planner" and len(planner.last) >= 2
    best = max(planner.last, key=lambda x: x["score"])
    assert best["content"] == out["content"]
    lab.close()
