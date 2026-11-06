"""RL correctness: log-probs, reward attribution, returns/advantages, masking, gradient flow."""

import math

import pytest
import torch

from agentrs.agent.runtime import run_episode
from agentrs.data.schemas import Provenance, Source, Transition
from agentrs.environments.base import Observation
from agentrs.environments.miniweb.tasks import task_set
from agentrs.evaluation.benchmark import make_env
from agentrs.models.critic import Critic, CriticConfig
from agentrs.models.policy import LMPolicy, PolicyActor, PolicyConfig
from agentrs.training.common import param_change, param_snapshot
from agentrs.training.rl import RLConfig, RLTrainer, build_rl_steps, policy_loss


def _tr(ep, step, reward=0.0, terminated=False, truncated=False, valid=True, action="click(e1)",
        cands=("click(e1)", "click(e2)"), beh=-0.69, task="t1"):
    o = Observation(env_id="x")
    return Transition(episode_id=ep, step=step, task_id=task, env_id="x", goal="g", observation=o,
                      belief_prompt=f"GOAL: g {ep} {step}\nACTION:", action=action, candidates=list(cands),
                      chosen_index=list(cands).index(action), behavior_logprob=beh, policy_logprob=beh,
                      next_observation=o, reward=reward, terminated=terminated, truncated=truncated,
                      action_valid=valid, provenance=Provenance(source=Source.AGENT))


def test_recorded_logprobs_match_recomputed(scratch_bb):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    env = make_env("miniweb")
    ep = run_episode(env, task_set("train", ["checkboxes"], 1)[0], PolicyActor(pol, "sample", seed=0))
    for t in ep.transitions:
        lp = pol.log_probs([t.belief_prompt], [t.candidates])[0]
        assert math.isclose(float(lp[t.chosen_index].detach()), t.policy_logprob, abs_tol=1e-4)
        assert t.behavior_logprob == pytest.approx(t.policy_logprob)  # on-policy sampling


def test_reward_attribution_and_episode_boundaries():
    cfg = RLConfig(algo="reinforce", baseline="none", gamma=0.5, advantage_norm=False, invalid_action_penalty=0.0)
    a = [_tr("A", 0), _tr("A", 1), _tr("A", 2, reward=1.0, terminated=True)]
    b = [_tr("B", 0), _tr("B", 1, reward=0.0, terminated=True)]
    steps = build_rl_steps([a, b], cfg)
    rets = [s.ret for s in steps]
    assert rets == pytest.approx([0.25, 0.5, 1.0, 0.0, 0.0])  # B never receives A's reward


def test_truncation_is_bootstrapped_termination_is_not(scratch_bb):
    cfg = RLConfig(algo="reinforce", baseline="critic", gamma=1.0, advantage_norm=False, invalid_action_penalty=0.0)
    cr = Critic(CriticConfig(backbone=scratch_bb))
    trunc = [_tr("A", 0), _tr("A", 1, truncated=True)]
    trunc[-1].next_belief_prompt = "GOAL: g final\nACTION:"
    term = [_tr("B", 0), _tr("B", 1, terminated=True)]
    st = build_rl_steps([trunc], cfg, cr)
    v_final = float(cr.value(["GOAL: g final\nACTION:"])[0].detach())
    assert st[-1].ret == pytest.approx(v_final, abs=1e-5)
    st2 = build_rl_steps([term], cfg, cr)
    assert st2[-1].ret == pytest.approx(0.0)


def test_invalid_actions_never_positively_reinforced():
    cfg = RLConfig(algo="reinforce", baseline="none", advantage_norm=False, invalid_action_penalty=0.1)
    ep = [_tr("A", 0, valid=False), _tr("A", 1, reward=1.0, terminated=True)]
    st = build_rl_steps([ep], cfg)
    assert st[0].advantage <= 0.0 and st[1].advantage > 0.0


def test_grpo_group_relative_advantages():
    cfg = RLConfig(algo="grpo", invalid_action_penalty=0.0)
    eps = [[_tr(f"E{i}", 0, reward=float(i % 2), terminated=True, task="same")] for i in range(4)]
    st = build_rl_steps(eps, cfg)
    assert sum(s.advantage for s in st) == pytest.approx(0.0, abs=1e-5)
    assert all((s.advantage > 0) == (eps[i][0].reward > 0) for i, s in enumerate(st))


def test_gradient_flows_through_policy_only(scratch_bb):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    cfg = RLConfig(algo="ppo", advantage_norm=False, invalid_action_penalty=0.0, entropy_coef=0.0)
    lp = pol.log_probs(["GOAL: g A 0\nACTION:"], [["click(e1)", "click(e2)"]])[0].detach()
    ep = [_tr("A", 0, reward=1.0, terminated=True, beh=float(lp[0]))]
    steps = build_rl_steps([ep], cfg)
    loss, st = policy_loss(pol, steps, cfg)
    assert st["mean_ratio"] == pytest.approx(1.0, abs=1e-4)  # on-policy: ratio 1 before updating
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pol.parameters())
    adv = torch.tensor([s.advantage for s in steps])
    assert not adv.requires_grad  # rewards/advantages are constants: no gradient through the environment


def test_off_policy_importance_weights_truncated(scratch_bb):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    cfg = RLConfig(algo="reinforce", rho_max=1.0, advantage_norm=False)
    # behaviour probability much lower than pi -> ratio > 1 -> truncated to rho_max
    ep = [_tr("A", 0, reward=1.0, terminated=True, beh=math.log(1e-3))]
    _, st = policy_loss(pol, build_rl_steps([ep], cfg), cfg)
    assert st["clipfrac"] == 1.0


def test_rl_update_changes_policy_and_improves_on_easy_task(scratch_bb):
    """A few REINFORCE updates on click_button (1-step tasks) increase the probability of the
    rewarded action on the training states: the policy improves because its weights change."""
    torch.manual_seed(0)
    # length-normalised scoring: an untrained model otherwise puts ~all mass on the shortest action
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb, length_norm=1.0))
    cr = Critic(CriticConfig(backbone=scratch_bb))
    cfg = RLConfig(algo="reinforce", baseline="batch_mean", policy_optim={"lr": 3e-3, "total_steps": 1000},
                   entropy_coef=0.0, minibatch_size=32)
    trainer = RLTrainer(pol, cr, cfg)
    env = make_env("miniweb")
    tasks = task_set("train", ["click_button"], n_per_family=8)
    snap = param_snapshot(pol)

    def success_prob():
        from agentrs.agent.belief import BeliefState

        tot = 0.0
        for t in tasks:
            obs = env.reset(t)
            b = BeliefState(task=t.public())
            b.reset(obs)
            cands, lp = pol.distribution(b, env.available_actions())
            tgt = next(i for i, c in enumerate(cands) if c.type.value == "click"
                       and obs.element(c.target).name == t.private["target"])
            tot += float(lp[tgt].exp())
        return tot / len(tasks)

    p0 = success_prob()
    for it in range(12):
        eps = [run_episode(env, t, PolicyActor(pol, "sample", seed=100 * it + i)) for i, t in enumerate(tasks * 2)]
        trainer.update([e.transitions for e in eps])
    assert param_change(pol, snap) > 0
    assert success_prob() > p0 + 0.05
