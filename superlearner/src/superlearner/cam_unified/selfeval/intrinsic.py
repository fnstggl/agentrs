"""Meta-learned intrinsic reward R_psi and lifetime value V_L (meta-reward spec 5.1, 5.4; manifest IR-01, IR-05).

    r_internal_t = R_psi(b_t, e_a, b_next, e_g, uncertainty, outcome_evidence, learning_context)

`IntrinsicReward` receives pooled latent understanding before and after the action, the action embedding, the goal
embedding (or a learned null-goal embedding in autonomous / goal-free mode) and a vector of *legitimately computable*
learning signals assembled by `ExperienceSignals` (prediction errors, belief entropy/ESS before and after, world-model
NLL of the realised outcome, change in predicted prospects, VOI and learning-gain predictions, observed costs, elapsed
lifetime fraction). It learns nonlinear combinations of them; nothing fixes their relative weights. Its output is
bounded, r = scale * tanh(x / scale), and normalised by running statistics of its own outputs on legitimately
available data (never using evaluator data).

Parameters psi are owned by the outer (meta) optimiser only: `meta_parameters()` is the only accessor the trainer
uses, the actor's optimiser never receives them, and a test checks that the actor's policy loss has no gradient path
into psi except through the differentiable inner update (where the meta-gradient is wanted).

`LifetimeValue` V_L(lifetime state) predicts the remaining *verified* utility of the current lifetime (sum of
independently verified objective utilities still to come) and is trained only on those verified targets; it is the
baseline / bootstrap of the outer objective when a lifetime is truncated (Zheng et al. 2020's lifetime value).

Versioning: `version` increments on every meta-update so records can say which reward model generated a reward
(CD-01) and stale rewards can be recomputed from the stored inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..learn.heads import _MLP


@dataclass
class ExperienceSignals:
    """Per-transition scalar learning signals, every one computable by CAM from its own model and observations.
    Shape [B] each; missing signals are 0 with the corresponding mask bit 0."""
    pred_error: Tensor            # world-model NLL of the realised outcome tokens (surprise)
    pred_error_epistemic: Tensor  # part of the surprise explained by particle disagreement (learnable)
    belief_entropy_before: Tensor
    belief_entropy_after: Tensor
    ess_before: Tensor
    ess_after: Tensor
    prospect_before: Tensor       # P(success) before (0.5 when no goal)
    prospect_after: Tensor        # P(success) after
    voi: Tensor                   # predicted decision-relevant value of the information
    learning_gain: Tensor         # predicted persistent learning gain (KV-02 predictor)
    cost: Tensor                  # observed resource cost of the action (symlog cents)
    duration: Tensor              # elapsed days
    lifetime_frac: Tensor         # fraction of the lifetime budget used
    novelty: Tensor               # 1 / sqrt(1 + visit count) of the action's tool within this lifetime
    executed_ok: Tensor           # 1 if the world accepted the action

    @classmethod
    def names(cls) -> list:
        return [f.name for f in fields(cls)]

    @classmethod
    def zeros(cls, B: int, device=None) -> "ExperienceSignals":
        z = torch.zeros(B, device=device)
        return cls(**{n: z.clone() for n in cls.names()})

    def stack(self) -> Tensor:
        return torch.stack([getattr(self, n).float() for n in self.names()], -1)

    @property
    def dim(self) -> int:
        return len(self.names())


N_SIGNALS = len(ExperienceSignals.names())


class IntrinsicReward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512, scale: float = 1.0, n_signals: int = N_SIGNALS):
        super().__init__()
        self.null_goal = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(4 * d_model, d_hidden)
        self.sig = nn.Sequential(nn.Linear(2 * n_signals, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.mlp = _MLP(d_hidden, d_hidden, 1, zero_out=True)
        self.scale = scale
        self.register_buffer("version_buf", torch.zeros((), dtype=torch.long))
        self.register_buffer("run_mean", torch.zeros(()))
        self.register_buffer("run_var", torch.ones(()))
        self.momentum = 0.01

    @property
    def version(self) -> int:
        return int(self.version_buf)

    def bump_version(self) -> None:
        self.version_buf += 1

    def meta_parameters(self):
        return list(self.parameters())

    def forward(self, b_t: Tensor, e_a: Tensor, b_next: Tensor, e_g: Optional[Tensor], signals: Tensor,
                signal_mask: Optional[Tensor] = None, normalise: bool = False) -> Tensor:
        """b_t, e_a, b_next: [B, D] pooled; e_g: [B, D] or None (goal-free: learned null goal); signals [B, n]."""
        B = b_t.shape[0]
        g = e_g if e_g is not None else self.null_goal.expand(B, -1)
        x = self.proj(torch.cat([b_t, e_a, b_next - b_t, g], -1).float())
        m = signal_mask.float() if signal_mask is not None else torch.ones_like(signals)
        s = torch.sign(signals) * torch.log1p(signals.abs())
        x = x + self.sig(torch.cat([s * m, m], -1).float())
        raw = self.mlp(x).squeeze(-1)
        r = self.scale * torch.tanh(raw / self.scale)
        if normalise:
            r = (r - self.run_mean) / self.run_var.clamp_min(1e-6).sqrt()
        return r

    @torch.no_grad()
    def update_stats(self, r: Tensor) -> None:
        self.run_mean.mul_(1 - self.momentum).add_(self.momentum * r.mean())
        self.run_var.mul_(1 - self.momentum).add_(self.momentum * r.var(unbiased=False))


class LifetimeValue(nn.Module):
    """V_L(lifetime feature) -> predicted remaining verified lifetime utility (symlog regression)."""

    def __init__(self, d_model: int, d_hidden: int = 512, n_ctx: int = 4):
        super().__init__()
        self.ctx = nn.Linear(n_ctx, d_model)
        self.mlp = _MLP(d_model, d_hidden, 1, zero_out=True)
        self.n_ctx = n_ctx

    def forward(self, feat: Tensor, ctx: Optional[Tensor] = None) -> Tensor:
        c = ctx if ctx is not None else feat.new_zeros(feat.shape[0], self.n_ctx)
        return self.mlp(feat.float() + self.ctx(c.float())).squeeze(-1)

    def loss(self, pred: Tensor, verified_remaining: Tensor) -> Tensor:
        y = torch.sign(verified_remaining) * torch.log1p(verified_remaining.abs())
        return F.smooth_l1_loss(pred, y.float())

    @staticmethod
    def to_utility(pred: Tensor) -> Tensor:
        return torch.sign(pred) * torch.expm1(pred.abs())
