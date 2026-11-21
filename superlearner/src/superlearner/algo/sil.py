"""PPO with a replay buffer of the agent's own verified-successful episodes (self-imitation).  [candidate-editable]

After the on-policy PPO update the algorithm takes `sil_steps` extra gradient steps on episodes drawn from a
buffer of past episodes that the evaluator verified as successful. Two forms of the loss:

``advantage``  Oh et al. (2018): -log pi(a|s) * max(R - V(s), 0) with the matching one-sided value loss;
``filtered``   reward-filtered imitation: -log pi(a|s) on every decision of a successful episode whose action the
               world accepted (decisions whose action failed are not imitated).

Episodes are drawn so that every task arm in the buffer is equally likely (rare, newly solved arms are not
drowned by arms that are solved all the time). This is reinforcement learning from the agent's own experience:
the only supervision is the evaluator's verified outcome of episodes the agent itself generated; no demonstration
is involved. Off-policy data is used only through these lower-bound objectives, so no importance weights are needed.
"""
from __future__ import annotations

import numpy as np
import torch

from ..cam.batch import evaluate
from . import rule as R
from .base import register
from .ppo import PPO, grad_norm, make_minibatches


@register("ppo_sil")
class PPOSIL(PPO):
    uses_replay = True

    def __init__(self, model, cfg=None, device="cpu"):
        super().__init__(model, cfg, device)
        self.buffer: list[dict] = []
        self.capacity = int(self.cfg.get("replay_episodes", 1500))
        self.sil_steps = int(self.cfg.get("sil_steps", 2))
        self.sil_coef = float(self.cfg.get("sil_coef", 0.5))
        self.sil_mode = str(self.cfg.get("sil_mode", "advantage"))
        self.sil_batch = int(self.cfg.get("sil_batch", self.c["minibatch_eps"]))

    def _sample(self) -> list[dict]:
        by: dict = {}
        for e in self.buffer:
            by.setdefault(tuple(e["summary"].get("arm", ())), []).append(e)
        arms = sorted(by)
        out = []
        for _ in range(min(self.sil_batch, len(self.buffer))):
            lst = by[arms[int(self.rng.integers(0, len(arms)))]]
            out.append(lst[int(self.rng.integers(0, len(lst)))])
        return out

    def update(self, episodes, store, demos=None):
        m = super().update(episodes, store, demos)
        self.buffer += [e for e in episodes if e["summary"].get("success")]
        self.buffer = self.buffer[-self.capacity:]
        if not self.buffer or self.sil_steps <= 0 or self.n_calls <= self.c["critic_warmup"]:
            m["sil_loss"] = 0.0
            m["replay_episodes"] = len(self.buffer)
            return m
        model, dev = self.model, self.device
        model.train()
        mc = R.RuleConfig(**dict(self.rule.to_dict(), lam=1.0))
        losses = []
        for _ in range(self.sil_steps):
            part = self._sample()
            b = make_minibatches(part, store, dev, len(part), self.rng)[0]
            t = b.t
            out = evaluate(model, b)
            w = t["valid"].float()
            _, ret = R.advantages(t["reward"], torch.zeros_like(t["reward"]), t["done"], t["trunc"], t["valid"], mc)
            v = out["values"].mean(-1)
            if self.sil_mode == "filtered":
                wi = w * t["ok"]
                loss = -(out["logp"] * wi).sum() / wi.sum().clamp(min=1) \
                    + 0.5 * (((ret - v) ** 2) * w).sum() / w.sum().clamp(min=1)
            else:
                gap = (ret - v.detach()).clamp(min=0)
                loss = (-(out["logp"] * gap) * w).sum() / w.sum().clamp(min=1) \
                    + 0.5 * ((((ret - v).clamp(min=0)) ** 2) * w).sum() / w.sum().clamp(min=1)
            if not torch.isfinite(loss):
                continue
            self.opt.zero_grad(set_to_none=True)
            (self.sil_coef * loss).backward()
            grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.c["max_grad_norm"])
            self.opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        m["sil_loss"] = float(np.mean(losses)) if losses else 0.0
        m["sil_steps"] = len(losses)
        m["replay_episodes"] = len(self.buffer)
        m["replay_arms"] = len({tuple(e["summary"].get("arm", ())) for e in self.buffer})
        return m
