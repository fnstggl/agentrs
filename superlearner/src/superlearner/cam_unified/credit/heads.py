"""CAM-scale learned credit-assignment components (meta-reward spec 6.2, 7.3, 7.5, 7.6; manifest CR-02, CR-03, CR-05).

* `OutcomeEncoder` u = U(outcome evidence): rewarding-outcome representations. Trained to predict the outcome's
  verified utility contribution (so u keeps what matters for the return) with a small bottleneck so irrelevant
  detail is discarded (COCOA's 'rewarding outcome' representation, 7.5).
* `HindsightModel` h_kappa(a | b_t, u, g, policy_ctx): which of the actions that were available at decision t was
  taken, given the later outcome u. Actions are scored by pointer attention over the candidate/tool embeddings that
  were available at t (a finite, recorded set: the restricted candidate-policy estimator of 7.6). Trained with
  cross-entropy on the executed candidate. Retrospective-only: it may see future information; the online policy
  never calls it.
* contribution coefficients w(b_t, a, u) = h(a | b_t, u) / pi(a | b_t) - 1, from recorded behaviour probabilities.
* `ReturnDecomposer` F_xi(H_t, g): a causal transformer over per-decision summaries that predicts the eventual
  verified return from every prefix; RUDDER redistribution c_t with the discounted / variable-duration conservation
  condition  sum_t gamma^{tau_t} c_t = G_disc  (and the undiscounted condition when gamma = 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..learn.heads import _MLP, symexp, symlog


class OutcomeEncoder(nn.Module):
    def __init__(self, d_in: int, d_u: int = 64, d_hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.RMSNorm(d_in), nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_u))
        self.util = nn.Linear(d_u, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc(x.float())

    def utility(self, u: Tensor) -> Tensor:
        return self.util(u).squeeze(-1)

    def loss(self, x: Tensor, utility: Tensor) -> Tensor:
        u = self(x)
        return F.smooth_l1_loss(self.utility(u), symlog(utility.float()))


class HindsightModel(nn.Module):
    """h_kappa(a | b_t, u, g, policy context, elapsed time). Output softmax(gate * f + (1 - gate) * centred log pi):
    the behaviour policy's log-probabilities enter as a prior through a learned gate computed from detached
    features (COCOA Prop. 5 / App. E.8.1, R-HCA App. F.1: needed to track a moving policy), and the elapsed time
    between decision and outcome (log days and decisions) is an input (time-indexed hindsight)."""

    def __init__(self, d_model: int, d_u: int = 64, d_hidden: int = 512):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(2 * d_model + d_u + 2, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_model))
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.null_goal = nn.Parameter(torch.zeros(d_model))
        self.temp = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.gate = nn.Sequential(nn.Linear(2 * d_model + d_u + 2, 64), nn.SiLU(), nn.Linear(64, 1))

    def logits(self, b_t: Tensor, u: Tensor, e_g: Optional[Tensor], cand: Tensor, cand_mask: Tensor,
               logp_policy: Optional[Tensor] = None, elapsed: Optional[Tensor] = None) -> Tensor:
        """b_t [B, D] (detached state feature at t), u [B, d_u], e_g [B, D] | None, cand [B, A, D], mask [B, A],
        logp_policy [B, A] behaviour log-probabilities at t (None: uniform prior), elapsed [B, 2] (log1p days,
        log1p decisions between t and the outcome; None: zeros)."""
        B = b_t.shape[0]
        g = e_g if e_g is not None else self.null_goal.expand(B, -1)
        el = elapsed.float() if elapsed is not None else b_t.new_zeros(B, 2).float()
        x = torch.cat([b_t.float(), g.float(), u.float(), el], -1)
        q = self.q(x)
        k = self.k(cand.float())
        f = torch.einsum("bd,bad->ba", q, k) / self.temp.exp()
        if logp_policy is not None:
            lp = logp_policy.float().masked_fill(~cand_mask.bool(), 0.0)
            n = cand_mask.float().sum(-1, keepdim=True).clamp_min(1)
            lp = lp - (lp * cand_mask.float()).sum(-1, keepdim=True) / n
            gt = torch.sigmoid(self.gate(x.detach()))
            f = gt * f + (1 - gt) * lp
        return f.masked_fill(~cand_mask.bool(), float("-inf"))

    def loss(self, b_t, u, e_g, cand, cand_mask, taken: Tensor, logp_policy=None, elapsed=None) -> Tensor:
        lg = self.logits(b_t, u, e_g, cand, cand_mask, logp_policy, elapsed)
        return F.cross_entropy(lg, taken.long())


def contribution_coefficients(h_logits: Tensor, logp_policy: Tensor, cand_mask: Tensor, clip: float = 10.0) -> Tensor:
    """w[b, a] = h(a|b,u) / pi(a|b) - 1 over the recorded candidate set; clipped for variance control (the clip is a
    declared bias, measured against exact coefficients in credit/exact tests)."""
    h = torch.softmax(h_logits.float(), -1)
    pi = logp_policy.float().exp()
    w = h / pi.clamp_min(1e-8) - 1.0
    w = w.clamp(-1.0, clip)
    return torch.where(cand_mask.bool(), w, torch.zeros_like(w))


class _CausalBlock(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = _MLP(d, 2 * d, d)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        y = self.n1(x)
        a, _ = self.att(y, y, y, attn_mask=causal, key_padding_mask=~pad.bool(), need_weights=False)
        x = x + a
        return x + self.ff(self.n2(x))


@dataclass
class Redistribution:
    contrib: Tensor        # [B, T] redistributed rewards c_t (in utility units)
    pred: Tensor           # [B, T] predicted eventual return after each prefix (utility units)
    offset: Tensor         # [B] prediction before any decision (the expected return)
    residual: Tensor       # [B] realised minus final prediction, added to the last valid step


class ReturnDecomposer(nn.Module):
    """F_xi(H_{<=t}, g) -> predicted eventual (discounted) verified return, for every prefix.

    Trained by squared error on the linearly scaled return G / `scale` (a conditional-mean regression, as RUDDER's
    r * 10 / g_max scaling), so F estimates E[G | H_t] and differences of predictions are contribution estimates; a
    nonlinear target transform (symlog) would break that even with infinite data (research note §10.5)."""

    def __init__(self, d_in: int, d_model: int = 384, n_layers: int = 3, n_heads: int = 6, max_len: int = 4096,
                 scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.inp = nn.Linear(d_in, d_model)
        self.goal = nn.Linear(d_in, d_model)
        self.time = nn.Linear(2, d_model)
        self.start = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList([_CausalBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, mask: Tensor, e_g: Optional[Tensor] = None, t_days: Optional[Tensor] = None) -> Tensor:
        """x [B, T, d_in], mask [B, T] -> predictions of G / scale [B, T+1] (index 0 = before any decision)."""
        B, T, _ = x.shape
        h = self.inp(x.float())
        if t_days is not None:
            h = h + self.time(torch.stack([torch.log1p(t_days.float().clamp_min(0)), t_days.float() / 365.0], -1))
        s = self.start.expand(B, 1, -1)
        if e_g is not None:
            s = s + self.goal(e_g.float())[:, None]
            h = h + self.goal(e_g.float())[:, None]
        h = torch.cat([s, h], 1)
        pad = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mask.bool()], 1)
        for blk in self.blocks:
            h = blk(h, pad)
        return self.head(self.norm(h)).squeeze(-1)

    def loss(self, x: Tensor, mask: Tensor, G: Tensor, e_g: Optional[Tensor] = None, t_days: Optional[Tensor] = None,
             resolved: Optional[Tensor] = None) -> Tensor:
        """Every prefix regresses the eventual verified return G [B]; unresolved trajectories are excluded."""
        pred = self(x, mask, e_g, t_days)
        tgt = (G.float() / self.scale)[:, None].expand_as(pred)
        valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1).float()
        if resolved is not None:
            valid = valid * resolved.float()[:, None]
        l = (pred - tgt) ** 2
        return (l * valid).sum() / valid.sum().clamp_min(1.0)

    @torch.no_grad()
    def redistribute(self, x: Tensor, mask: Tensor, G: Tensor, e_g: Optional[Tensor] = None,
                     t_days: Optional[Tensor] = None, gamma_tau: Optional[Tensor] = None) -> Redistribution:
        """c_t = (F(H_{<=t}) - F(H_{<t})) / gamma^{tau_t}, residual G - F(H_{<=T}) on the last valid step, offset
        F(empty) on the first; then sum_t gamma^{tau_t} c_t = G (gamma_tau [B, T] = gamma^{tau_t}; None -> 1)."""
        pred = self(x, mask, e_g, t_days) * self.scale        # [B, T+1] in utility units
        m = mask.float()
        # carry the last valid prediction across padding so differences on padded steps are 0
        filled = pred.clone()
        for t in range(1, filled.shape[1]):
            filled[:, t] = torch.where(mask[:, t - 1].bool(), filled[:, t], filled[:, t - 1])
        diff = (filled[:, 1:] - filled[:, :-1]) * m
        g = gamma_tau.float() if gamma_tau is not None else torch.ones_like(diff)
        contrib = diff / g.clamp_min(1e-8)
        last = (mask.long().cumsum(1) == mask.long().sum(1, keepdim=True)) & mask.bool()
        first = (mask.long().cumsum(1) == 1) & mask.bool()
        residual = G.float() - filled[:, -1]
        contrib = contrib + last.float() * (residual / (g * last.float()).sum(1).clamp_min(1e-8))[:, None]
        contrib = contrib + first.float() * (filled[:, 0] / (g * first.float()).sum(1).clamp_min(1e-8))[:, None]
        return Redistribution(contrib, filled[:, 1:], filled[:, 0], residual)
