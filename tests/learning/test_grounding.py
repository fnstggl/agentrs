"""Grounding: an inaccurate world model that predicts guaranteed success must not create
real success; the verifier rejects it; the failure becomes real negative experience; and the
world model learns from the discrepancy."""

import torch

from agentrs.agent.runtime import run_episode
from agentrs.data.schemas import Source
from agentrs.environments.miniweb.tasks import make_task
from agentrs.evaluation.benchmark import make_env
from agentrs.models.critic import Critic, CriticConfig
from agentrs.models.policy import LMPolicy, PolicyConfig
from agentrs.models.world_model import N_EVENTS, WorldModel, WorldModelConfig
from agentrs.planning.mpc import MPCPlanner, PlannerConfig
from agentrs.training.common import OptimConfig
from agentrs.training.imagination import ImaginationConfig, imagine_rollouts
from agentrs.training.rl import RLConfig, build_rl_steps
from agentrs.training.world_model import WMTrainConfig, train_world_model


def _deluded_wm(scratch_bb):
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=3))
    with torch.no_grad():  # predicts reward=1 and termination=1 for every action
        wm.heads.w2.zero_()
        wm.heads.b2[:, N_EVENTS] = 8.0
        wm.heads.b2[:, N_EVENTS + 1] = 8.0
    return wm


def test_imagined_success_is_not_real_success(scratch_bb):
    torch.manual_seed(0)
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    wm = _deluded_wm(scratch_bb)
    cr = Critic(CriticConfig(backbone=scratch_bb))
    env = make_env("miniweb")
    task = make_task("login", "train", 3)  # finishing immediately does NOT log in
    planner = MPCPlanner(pol, wm, cr, PlannerConfig(num_candidates=6, policy_prior_weight=0.0,
                                                    uncertainty_penalty=0.0), seed=0)
    ep = run_episode(env, task, planner)
    first = ep.transitions[0]
    assert first.wm_prediction is not None and first.wm_prediction.reward_prob > 0.99  # model is sure it succeeds
    assert ep.success is False  # ... but the external verifier rejects it
    assert all(t.reward == 0.0 for t in ep.transitions)
    assert ep.verification["success"] is False
    # the failure is real negative experience for RL
    steps = build_rl_steps([ep.transitions], RLConfig(algo="reinforce", baseline="none", advantage_norm=False))
    assert all(s.ret <= 0.0 for s in steps)
    # imagined rollouts with the same deluded model are labelled IMAGINED and carry no verified outcome
    rolls, stats = imagine_rollouts(pol, wm, cr, ep.transitions, ep.transitions,
                                    ImaginationConfig(enabled=True, rollouts_per_update=3, horizon=2,
                                                      max_epistemic=1.0))
    assert stats["imagined_success_rate"] > 0.9
    for traj in rolls:
        for t in traj:
            assert t.provenance.source == Source.IMAGINED and t.episode_success is None and t.verification is None
            assert not t.is_real

    # the world model learns from the real discrepancy
    wm_prompt_action = first.action
    before = wm.predict_batch_heads(task.goal, first.observation, [wm_prompt_action])[0][0, N_EVENTS].item()
    train_world_model(wm, ep.transitions, [], WMTrainConfig(steps=40, batch_size=8, eval_every=10**9,
                                                            optim=OptimConfig(lr=3e-3, total_steps=40)))
    after = wm.predict_batch_heads(task.goal, first.observation, [wm_prompt_action])[0][0, N_EVENTS].item()
    assert after < before - 0.3, (before, after)
