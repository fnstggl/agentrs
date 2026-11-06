"""Critic training from real trajectories with verified rewards."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from pydantic import ConfigDict, BaseModel, Field

from agentrs.data.schemas import Transition
from agentrs.evaluation.metrics import auroc, bernoulli_nll, expected_calibration_error
from agentrs.models.critic import Critic
from agentrs.monitoring.tracker import Tracker
from agentrs.training.common import GradientTrainer, OptimConfig, param_change, param_snapshot, save_checkpoint
from agentrs.training.returns import gae


class CriticTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=5e-4, total_steps=300, warmup_steps=20,
                                                                   schedule="cosine"))
    batch_size: int = 32
    steps: int = 300
    eval_every: int = 100
    seed: int = 0


def episodes_of(ts: Sequence[Transition]) -> list[list[Transition]]:
    eps: dict[str, list[Transition]] = defaultdict(list)
    for t in ts:
        if t.is_real and t.belief_prompt:
            eps[t.episode_id].append(t)
    return [sorted(v, key=lambda t: t.step) for v in eps.values()]


@torch.no_grad()
def critic_targets(critic: Optional[Critic], ts: Sequence[Transition], gamma: float, lam: float) -> list[tuple[str, float]]:
    """(belief_prompt, lambda-return target) pairs. Truncated episodes bootstrap with V(s_T)
    when a critic is given (otherwise 0)."""
    out = []
    for ep in episodes_of(ts):
        rewards = [t.reward for t in ep]
        last = ep[-1]
        terminated = bool(last.terminated)
        if critic is not None:
            vals = critic.value([t.belief_prompt for t in ep]).tolist()
            last_v = 0.0
            if not terminated and last.next_belief_prompt:
                last_v = float(critic.value([last.next_belief_prompt])[0])
        else:
            vals, last_v = [0.0] * len(ep), 0.0
        _, rets = gae(rewards, vals, gamma, lam, terminated, last_v)
        out += [(t.belief_prompt, float(np.clip(r, 0, 1))) for t, r in zip(ep, rets)]
    return out


@torch.no_grad()
def evaluate_critic(critic: Critic, ts: Sequence[Transition], batch_size: int = 64) -> dict:
    """Calibration of V(s_0) against verified episode success, and fit to returns."""
    eps = episodes_of(ts)
    if not eps:
        return {}
    critic.eval()
    first = [ep[0].belief_prompt for ep in eps]
    succ = [float(bool(ep[-1].episode_success)) for ep in eps]
    v0 = []
    for i in range(0, len(first), batch_size):
        v0 += critic.value(first[i : i + batch_size]).tolist()
    pairs = critic_targets(None, ts, critic.cfg.gamma, 1.0)
    vs = []
    for i in range(0, len(pairs), batch_size):
        vs += critic.value([p for p, _ in pairs[i : i + batch_size]]).tolist()
    tg = [t for _, t in pairs]
    critic.train()
    return {"v0_success_auroc": auroc(v0, succ), "v0_ece": expected_calibration_error(v0, succ),
            "v0_nll": bernoulli_nll(v0, succ), "mean_v0": float(np.mean(v0)), "success_rate": float(np.mean(succ)),
            "return_mse": float(np.mean((np.asarray(vs) - np.asarray(tg)) ** 2))}


def train_critic(critic: Critic, train: Sequence[Transition], val: Sequence[Transition], cfg: CriticTrainConfig,
                 tracker: Optional[Tracker] = None, out_dir: Optional[str] = None,
                 trainer: Optional[GradientTrainer] = None, log_prefix: str = "critic/") -> dict:
    rng = random.Random(cfg.seed)
    lam = critic.cfg.td_lambda
    data = critic_targets(critic if lam < 1.0 else None, train, critic.cfg.gamma, lam)
    if not data:
        raise ValueError("no critic training data")
    trainer = trainer or GradientTrainer(critic, cfg.optim)
    snap = param_snapshot(critic)
    before = evaluate_critic(critic, val)
    hist = []
    for step in range(cfg.steps):
        batch = rng.sample(data, min(cfg.batch_size, len(data)))
        loss = critic.loss([p for p, _ in batch], torch.tensor([t for _, t in batch]))
        upd = trainer.step(loss)
        rec = {"loss": float(loss.detach()), **upd}
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            rec.update({f"val_{k}": v for k, v in evaluate_critic(critic, val).items()})
            if out_dir:
                save_checkpoint(out_dir, critic, trainer, extra={"stage": "critic", "step": step + 1})
        if tracker:
            tracker.log(rec, step, prefix=log_prefix)
        hist.append(rec)
    return {"n_train": len(data), "before": before, "after": evaluate_critic(critic, val),
            "param_change_l2": param_change(critic, snap), "history": hist, "trainer": trainer}
