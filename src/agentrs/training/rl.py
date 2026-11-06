"""Policy-gradient RL from environment-verified rewards.

Algorithms (selected by ``RLConfig.algo``):

* ``reinforce`` - REINFORCE with a learned (critic) or batch-mean baseline.
* ``ppo``       - clipped surrogate objective with GAE, several epochs per batch.
* ``grpo``      - group-relative outcome advantages: each task is attempted
                  ``group_size`` times and advantage = (R - mean_group) / std_group.

All variants differentiate log pi_theta(a_t | s_t) recomputed from the stored
prompt and candidate set; rewards and advantages are constants (no gradient
flows through environment transitions). Off-policy data (planner or epsilon
selection) is corrected with truncated importance weights
rho = min(rho_max, pi_theta(a|s) / mu(a|s)) using the recorded behaviour
log-probability mu. Steps with invalid actions are never positively
reinforced (advantage clipped to <= 0, plus an optional penalty).
Truncation is bootstrapped with V(s_T); termination is not.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from pydantic import ConfigDict, BaseModel, Field

from agentrs.data.schemas import EpisodeRecord, Transition
from agentrs.models.critic import Critic
from agentrs.models.policy import LMPolicy, entropy_from_logp, kl_from_logp
from agentrs.training.common import GradientTrainer, OptimConfig
from agentrs.training.returns import gae


class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algo: str = "reinforce"  # reinforce | ppo | grpo
    gamma: float = 0.95
    gae_lambda: float = 0.95
    baseline: str = "critic"  # critic | batch_mean | none
    advantage_norm: bool = True
    epochs: int = 1  # PPO epochs per batch
    minibatch_size: int = 16
    clip_eps: float = 0.2
    rho_max: float = 1.0  # truncated importance weight for off-policy steps
    entropy_coef: float = 0.01
    kl_coef: float = 0.0  # KL(pi || pi_ref) regulariser over candidate sets
    invalid_action_penalty: float = 0.1
    group_size: int = 4  # GRPO
    distill_coef: float = 0.0  # cross-entropy towards planner-selected actions (search distillation)
    imagined_weight: float = 0.5  # weight of imagined-trajectory steps relative to real steps
    policy_optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=1e-4, total_steps=10_000))
    critic_optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=3e-4, total_steps=10_000))
    critic_steps_per_iter: int = 4
    max_kl_stop: float = 0.5  # early-stop PPO epochs if approx KL exceeds this


@dataclass
class RLStep:
    prompt: str
    candidates: list[str]
    action_index: int
    behavior_logprob: float
    advantage: float
    ret: float
    valid: bool
    weight: float = 1.0
    selection: str = "policy"
    imagined: bool = False


def _valid_index(t: Transition) -> Optional[int]:
    if t.chosen_index is not None and t.chosen_index < len(t.candidates) and t.candidates[t.chosen_index] == t.action:
        return t.chosen_index
    if t.action in t.candidates:
        return t.candidates.index(t.action)
    return None


@torch.no_grad()
def build_rl_steps(episodes: Sequence[Sequence[Transition]], cfg: RLConfig, critic: Optional[Critic] = None,
                   imagined: bool = False) -> list[RLStep]:
    """Compute advantages/returns per episode. Rewards: verified env reward minus invalid-action penalty."""
    steps: list[RLStep] = []
    groups: dict[str, list[float]] = defaultdict(list)
    if cfg.algo == "grpo":
        for ep in episodes:
            groups[ep[0].task_id].append(sum(t.reward for t in ep))
    for ep in episodes:
        ep = sorted(ep, key=lambda t: t.step)
        rewards = [float(t.reward) - (cfg.invalid_action_penalty if not t.action_valid else 0.0) for t in ep]
        terminated = bool(ep[-1].terminated)
        if cfg.algo == "grpo":
            g = np.asarray(groups[ep[0].task_id])
            R = sum(t.reward for t in ep)
            adv_ep = (R - g.mean()) / (g.std() + 1e-6) if len(g) > 1 else 0.0
            pen = [-(cfg.invalid_action_penalty if not t.action_valid else 0.0) for t in ep]
            advs = [adv_ep + p for p in pen]
            rets = [R] * len(ep)
        else:
            if cfg.baseline == "critic" and critic is not None:
                vals = critic.value([t.belief_prompt for t in ep]).tolist()
                last_v = 0.0
                if not terminated and ep[-1].next_belief_prompt:
                    last_v = float(critic.value([ep[-1].next_belief_prompt])[0])
                lam = cfg.gae_lambda if cfg.algo == "ppo" else 1.0
                advs, rets = gae(rewards, vals, cfg.gamma, lam, terminated, last_v)
            else:
                advs, rets = gae(rewards, [0.0] * len(ep), cfg.gamma, 1.0, terminated, 0.0)
        for t, a, r in zip(ep, advs, rets):
            idx = _valid_index(t)
            if idx is None or not t.belief_prompt:
                continue
            beh = t.behavior_logprob if t.behavior_logprob is not None else t.policy_logprob
            steps.append(RLStep(prompt=t.belief_prompt, candidates=list(t.candidates), action_index=idx,
                                behavior_logprob=float(beh if beh is not None else 0.0), advantage=float(a),
                                ret=float(r), valid=t.action_valid, selection=t.selection, imagined=imagined,
                                weight=cfg.imagined_weight if imagined else 1.0))
    if cfg.baseline == "batch_mean" and cfg.algo != "grpo" and steps:
        m = float(np.mean([s.ret for s in steps]))
        for s in steps:
            s.advantage = s.ret - m
    if cfg.advantage_norm and cfg.algo != "grpo" and len(steps) > 1:
        a = np.asarray([s.advantage for s in steps])
        mu, sd = a.mean(), a.std() + 1e-6
        for s in steps:
            s.advantage = float((s.advantage - mu) / sd)
    for s in steps:  # never positively reinforce invalid actions
        if not s.valid:
            s.advantage = min(s.advantage, 0.0)
    return steps


def policy_loss(policy: LMPolicy, batch: Sequence[RLStep], cfg: RLConfig,
                ref_policy: Optional[LMPolicy] = None) -> tuple[torch.Tensor, dict]:
    lps = policy.log_probs([s.prompt for s in batch], [s.candidates for s in batch])
    dev = lps[0].device
    lp_a = torch.stack([lp[s.action_index] for lp, s in zip(lps, batch)])
    beh = torch.tensor([s.behavior_logprob for s in batch], device=dev)
    adv = torch.tensor([s.advantage for s in batch], device=dev)  # constants: no grad through env
    w = torch.tensor([s.weight for s in batch], device=dev)
    log_ratio = lp_a - beh
    ratio = torch.exp(log_ratio)
    if cfg.algo == "ppo":
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        pg = -(w * torch.min(s1, s2)).sum() / w.sum()
        clipfrac = float(((ratio - 1).abs() > cfg.clip_eps).float().mean())
    else:  # reinforce / grpo with truncated importance weights (=1 for on-policy samples)
        rho = torch.clamp(ratio.detach(), max=cfg.rho_max)
        pg = -(w * rho * adv * lp_a).sum() / w.sum()
        clipfrac = float((ratio.detach() > cfg.rho_max).float().mean())
    ent = torch.stack([entropy_from_logp(lp) for lp in lps]).mean()
    loss = pg - cfg.entropy_coef * ent
    stats = {"pg_loss": float(pg.detach()), "entropy": float(ent.detach()), "clipfrac": clipfrac,
             "approx_kl_old": float(((ratio.detach() - 1) - log_ratio.detach()).mean()),
             "mean_adv": float(adv.mean()), "mean_ratio": float(ratio.detach().mean())}
    if cfg.distill_coef > 0:
        planner = [i for i, s in enumerate(batch) if s.selection == "planner"]
        if planner:
            ce = -torch.stack([lp_a[i] for i in planner]).mean()
            loss = loss + cfg.distill_coef * ce
            stats["distill_ce"] = float(ce.detach())
    if cfg.kl_coef > 0 and ref_policy is not None:
        with torch.no_grad():
            ref = ref_policy.log_probs([s.prompt for s in batch], [s.candidates for s in batch])
        kl = torch.stack([kl_from_logp(lp, r) for lp, r in zip(lps, ref)]).mean()
        loss = loss + cfg.kl_coef * kl
        stats["kl_ref"] = float(kl.detach())
    return loss, stats


class RLTrainer:
    """Owns policy + critic optimisers; one ``update`` per batch of collected episodes."""

    def __init__(self, policy: LMPolicy, critic: Optional[Critic], cfg: RLConfig,
                 ref_policy: Optional[LMPolicy] = None, seed: int = 0):
        self.policy, self.critic, self.cfg, self.ref = policy, critic, cfg, ref_policy
        self.pt = GradientTrainer(policy, cfg.policy_optim)
        self.ct = GradientTrainer(critic, cfg.critic_optim) if critic is not None else None
        self.rng = random.Random(seed)
        self.updates = 0

    def update(self, episodes: Sequence[Sequence[Transition]],
               imagined: Sequence[Sequence[Transition]] = ()) -> dict:
        cfg = self.cfg
        steps = build_rl_steps(episodes, cfg, self.critic)
        if imagined:
            steps += build_rl_steps(imagined, cfg, self.critic, imagined=True)
        if not steps:
            return {"skipped": 1.0}
        stats_acc: dict[str, list[float]] = defaultdict(list)
        self.policy.train()
        for epoch in range(cfg.epochs if cfg.algo == "ppo" else 1):
            self.rng.shuffle(steps)
            stop = False
            for i in range(0, len(steps), cfg.minibatch_size):
                mb = steps[i : i + cfg.minibatch_size]
                loss, st = policy_loss(self.policy, mb, cfg, self.ref)
                upd = self.pt.step(loss)
                for k, v in {**st, **upd}.items():
                    stats_acc[k].append(v)
                if cfg.algo == "ppo" and st["approx_kl_old"] > cfg.max_kl_stop:
                    stop = True
                    break
            if stop:
                stats_acc["early_stop"].append(1.0)
                break
        if self.critic is not None and self.ct is not None and cfg.baseline == "critic":
            real = [s for s in steps if not s.imagined]
            for _ in range(cfg.critic_steps_per_iter):
                mb = self.rng.sample(real, min(len(real), 32))
                closs = self.critic.loss([s.prompt for s in mb],
                                         torch.tensor([min(1.0, max(0.0, s.ret)) for s in mb]))
                cu = self.ct.step(closs)
                stats_acc["critic_loss"].append(float(closs.detach()))
                stats_acc["critic_grad_norm"].append(cu["grad_norm"])
        self.updates += 1
        out = {k: float(np.nanmean(v)) for k, v in stats_acc.items() if v}
        out["n_steps"] = len(steps)
        out["n_imagined_steps"] = sum(s.imagined for s in steps)
        return out


def episode_transitions(eps: Sequence[EpisodeRecord]) -> list[list[Transition]]:
    return [ep.transitions for ep in eps if ep.transitions]


Collector = Callable[[int], list[EpisodeRecord]]
_ = math
