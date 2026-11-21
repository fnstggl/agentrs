"""PPO with a replay buffer of the agent's own successful episodes (self-imitation).  [candidate-editable]

An example of an algorithm that keeps experience between iterations: after the on-policy PPO update it takes
extra gradient steps on episodes drawn from a buffer of past *verified-successful* episodes, with the
self-imitation loss of Oh et al. (2018): -log pi(a|s) * max(R - V(s), 0) and the matching one-sided value loss.
Off-policy data is used only through this lower-bound objective (no importance weights are needed for it).
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
        self.capacity = int(self.cfg.get("replay_episodes", 512))
        self.sil_steps = int(self.cfg.get("sil_steps", 2))
        self.sil_coef = float(self.cfg.get("sil_coef", 0.5))

    def update(self, episodes, store, demos=None):
        m = super().update(episodes, store, demos)
        self.buffer += [e for e in episodes if e["summary"].get("success")]
        self.buffer = self.buffer[-self.capacity:]
        if not self.buffer or self.sil_steps <= 0:
            m["sil_loss"] = 0.0
            return m
        model, dev = self.model, self.device
        model.train()
        mc = R.RuleConfig(**dict(self.rule.to_dict(), lam=1.0))
        losses = []
        for _ in range(self.sil_steps):
            idx = self.rng.choice(len(self.buffer), size=min(self.c["minibatch_eps"], len(self.buffer)), replace=False)
            b = make_minibatches([self.buffer[i] for i in idx], store, dev, len(idx), self.rng)[0]
            t = b.t
            out = evaluate(model, b)
            _, ret = R.advantages(t["reward"], torch.zeros_like(t["reward"]), t["done"], t["trunc"], t["valid"], mc)
            v = out["values"].mean(-1)
            gap = (ret - v.detach()).clamp(min=0)
            w = t["valid"].float()
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
        m["replay_episodes"] = len(self.buffer)
        return m
