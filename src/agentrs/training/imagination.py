"""Imagined experience: bounded world-model rollouts for conservative policy improvement.

Safeguards
* Rollouts start from REAL states sampled from replay and are short (``horizon`` <= 3).
* A rollout is cut off as soon as the world model's ensemble disagreement
  exceeds ``max_epistemic`` (the remaining return is bootstrapped by the critic).
* Imagined transitions carry ``Source.IMAGINED``; the schema forbids them from
  holding verification results or episode success, and the replay buffer keeps
  them in a separate pool. Imagined rewards are *expected* rewards predicted by
  the world model, never verified rewards.
* ``ExploitationMonitor`` compares imagined success with real verified success
  of the same policy; a large gap indicates the policy is exploiting model
  errors and imagination weight is reduced / disabled.
* Real-data anchoring: imagined steps are always mixed with real on-policy steps
  and down-weighted (``RLConfig.imagined_weight``).
"""

from __future__ import annotations

import random
import uuid
from typing import Optional, Sequence

import torch
from pydantic import ConfigDict, BaseModel

from agentrs.actions.candidates import candidates_from_observation
from agentrs.actions.schema import ActionType
from agentrs.agent.belief import BeliefState, HistoryItem
from agentrs.data.diff import StateDiff
from agentrs.data.schemas import Provenance, Source, Transition
from agentrs.environments.base import PublicTask
from agentrs.models.critic import Critic
from agentrs.models.policy import LMPolicy
from agentrs.models.world_model import N_EVENTS, WorldModel
from agentrs.training.world_model import episode_histories


class ImaginationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    horizon: int = 2
    rollouts_per_update: int = 16
    max_epistemic: float = 0.08
    sample_rewards: bool = False  # if True, sample r ~ Bernoulli(p_r) instead of using the expectation
    exploitation_gap: float = 0.3  # disable imagination if imagined_success - real_success exceeds this
    seed: int = 0


def belief_from_transition(t: Transition, hist: list[tuple[str, str]]) -> BeliefState:
    b = BeliefState(task=PublicTask(task_id=t.task_id, env_id=t.env_id, family=t.family, goal=t.goal))
    b.reset(t.observation)
    b.history = [HistoryItem(action=a, outcome=o) for a, o in hist]
    return b


@torch.no_grad()
def imagine_rollouts(policy: LMPolicy, wm: WorldModel, critic: Optional[Critic], start: Sequence[Transition],
                     all_real: Sequence[Transition], cfg: ImaginationConfig) -> tuple[list[list[Transition]], dict]:
    rng = random.Random(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    hists = episode_histories(all_real)
    starts = [t for t in start if t.is_real and not t.terminated]
    if not starts:
        return [], {}
    rollouts: list[list[Transition]] = []
    n_cut, imag_success, lens = 0, 0.0, []
    for _ in range(cfg.rollouts_per_update):
        t0 = rng.choice(starts)
        belief = belief_from_transition(t0, hists.get((t0.episode_id, t0.step), []))
        obs = t0.observation
        ep_id = "imag-" + uuid.uuid4().hex[:12]
        traj: list[Transition] = []
        for h in range(cfg.horizon):
            env_cands = candidates_from_observation(obs, belief.task.goal)
            cands, lp = policy.distribution(belief, belief.augment_candidates(env_cands, obs))
            i = int(torch.multinomial(lp.exp(), 1, generator=gen))
            a = cands[i]
            hs = [(x.action, x.outcome) for x in belief.history]
            mean, std = wm.predict_batch_heads(belief.task.goal, obs, [a.to_str()], hs)
            m, s = mean[0].tolist(), std[0].tolist()
            epi = float(sum(s) / len(s))
            p_r, p_term = m[N_EVENTS], m[N_EVENTS + 1]
            terminal = a.type == ActionType.FINISH or p_term > 0.5
            reward = float(torch.bernoulli(torch.tensor(p_r), generator=gen)) if cfg.sample_rewards else p_r
            diff_text, _ = wm.sample_diffs(wm.prompt(belief.task.goal, obs, a.to_str(), hs), n=1, greedy=False,
                                           generator=gen)[0]
            d = StateDiff.from_text(diff_text)
            nxt = d.apply(obs, id_prefix="im")
            prompt = belief.prompt()
            cut = epi > cfg.max_epistemic
            tr = Transition(
                episode_id=ep_id, step=h, task_id=t0.task_id, env_id=t0.env_id, family=t0.family, split=t0.split,
                goal=t0.goal, observation=obs, belief_prompt=prompt, action=a.to_str(), action_type=a.type.value,
                candidates=[c.to_str() for c in cands], chosen_index=i, policy_logprob=float(lp[i]),
                behavior_logprob=float(lp[i]), selection="imagination", next_observation=nxt, diff_text=diff_text,
                events=m[:N_EVENTS], reward=reward, terminated=terminal, truncated=(not terminal) and (
                    cut or h == cfg.horizon - 1),
                provenance=Provenance(source=Source.IMAGINED, source_id="world_model"))
            belief.update(a, _fake_step_result(nxt, d))
            if tr.truncated:
                tr.next_belief_prompt = belief.prompt()
            traj.append(tr)
            obs = nxt
            if terminal:
                imag_success += p_r
                break
            if cut:
                n_cut += 1
                break
        lens.append(len(traj))
        rollouts.append(traj)
    stats = {"imagined_rollouts": len(rollouts), "imagined_mean_len": sum(lens) / max(1, len(lens)),
             "imagined_cut_by_uncertainty": n_cut / max(1, len(rollouts)),
             "imagined_success_rate": imag_success / max(1, len(rollouts))}
    return rollouts, stats


def _fake_step_result(obs, d: StateDiff):
    """StepResult-like object for updating an imagined belief (never sent to an environment)."""
    from agentrs.environments.base import StepResult

    return StepResult(observation=obs, action_valid=d.error is None, error=d.error)


class ExploitationMonitor:
    """Tracks the gap between imagined and real success of the current policy."""

    def __init__(self, gap: float):
        self.gap = gap
        self.history: list[tuple[float, float]] = []

    def update(self, imagined_success: float, real_success: float) -> bool:
        """Returns True if imagination should be (kept) enabled."""
        self.history.append((imagined_success, real_success))
        return (imagined_success - real_success) <= self.gap
