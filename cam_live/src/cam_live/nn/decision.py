"""cam_live.nn.decision — encoders for actions / goals / outcome queries, the decision trunk and the feature map.

One shared representation, several light heads (spec 1.3): the world model, evaluator, policy, value-of-information
and reward heads built by the other workstreams all read ``z = F_theta(H_t, a_t, g)`` and
``phi = Phi_theta(z, q)``.

Design rules behind this file:

* every input is RMS-normalised on its own before it is mixed, so the semantic side (action text) can never drown
  the belief ``b`` (an earlier system's policy ignored its latent: KL 4e-9 under a permuted belief);
* belief x action and (belief, action) x query interact multiplicatively as well as additively;
* ``phi`` has unit L2 norm with a constant last coordinate: bounded and well-conditioned for the Bayesian last layer.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from ..contracts import Level, Modality, OutcomeKind
from .core import N_TIME_FEATS, RMSNorm, SwiGLU, elapsed_features

MODALITY_ID: Dict[str, int] = {m: i for i, m in enumerate(Modality.ALL)}
LEVEL_ID: Dict[str, int] = {m: i for i, m in enumerate(Level.ALL)}


def _unit_init(lin: nn.Linear, gain: float = 1.0) -> None:
    nn.init.normal_(lin.weight, std=gain / math.sqrt(lin.in_features))
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


class ActionEncoder(nn.Module):
    """item token of the rendered action (content + exact cost quantities) + modality / level / size features."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.modality = nn.Embedding(len(Modality.ALL) + 1, d)
        self.level = nn.Embedding(len(Level.ALL) + 1, d)
        self.extra = nn.Linear(N_TIME_FEATS + 1, d, bias=False)
        self.norm = RMSNorm(d)
        _unit_init(self.proj)
        for e in (self.modality, self.level):
            nn.init.normal_(e.weight, std=0.3)
        nn.init.normal_(self.extra.weight, std=0.3 / math.sqrt(N_TIME_FEATS + 1))

    def forward(self, x: Tensor, modality_ids: Tensor, level_ids: Tensor, timeout_s: Tensor, n_steps: Tensor) -> Tensor:
        extra = torch.cat([elapsed_features(timeout_s), torch.log1p(n_steps.to(torch.float32)).unsqueeze(-1)], -1)
        return self.norm(self.proj(x) + self.modality(modality_ids) + self.level(level_ids) + self.extra(extra))


class GoalEncoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)
        _unit_init(self.proj)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class QueryEncoder(nn.Module):
    """content of "metric: description" + outcome kind + horizon (log-scaled) + exposure + the hashed metric-name
    embedding shared with the tokenizer's value channel (so a query about a metric is tied to its observed values)."""

    def __init__(self, d: int, value_dim: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.kind = nn.Embedding(len(OutcomeKind.ALL), d)
        self.name = nn.Linear(value_dim, d, bias=False)
        self.extra = nn.Linear(N_TIME_FEATS + 3, d, bias=False)
        self.norm = RMSNorm(d)
        _unit_init(self.proj)
        nn.init.normal_(self.kind.weight, std=0.5)
        _unit_init(self.name, 0.5)
        nn.init.normal_(self.extra.weight, std=0.5 / math.sqrt(N_TIME_FEATS + 3))

    def forward(self, content: Tensor, kind_ids: Tensor, name_emb: Tensor, horizon_s: Tensor, exposure: Tensor,
                goal_relevant: Tensor) -> Tensor:
        ex = exposure.to(torch.float64)
        extra = torch.cat([elapsed_features(horizon_s),
                           (torch.log1p(ex.clamp_min(0.0)) / 10.0).to(torch.float32).unsqueeze(-1),
                           (ex != 1.0).to(torch.float32).unsqueeze(-1),
                           goal_relevant.to(torch.float32).unsqueeze(-1)], -1)
        return self.norm(self.proj(content) + self.kind(kind_ids) + self.name(name_emb) + self.extra(extra))


class DecisionTrunk(nn.Module):
    """z [B, N, D] = F(b [B, D], e_a [B, N, D], e_g [B, D])."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.nb, self.na, self.ng = RMSNorm(d, affine=False), RMSNorm(d, affine=False), RMSNorm(d, affine=False)
        self.wb = nn.Linear(d, d, bias=False)
        self.wa = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, d, bias=False)
        self.ia = nn.Linear(d, d, bias=False)       # multiplicative action x situation interaction
        self.isit = nn.Linear(d, d, bias=False)
        self.sit_norm = RMSNorm(d, affine=False)
        self.ff_norm = RMSNorm(d)
        self.ff = SwiGLU(d, d_ff)
        self.out_norm = RMSNorm(d)
        for lin in (self.wb, self.wa, self.wg, self.ia, self.isit):
            _unit_init(lin)
        _unit_init(self.ff.w1)
        _unit_init(self.ff.w2)
        _unit_init(self.ff.w3, 0.5)

    def forward(self, b: Tensor, e_a: Tensor, e_g: Tensor) -> Tensor:
        sit = (self.wb(self.nb(b)) + self.wg(self.ng(e_g))).unsqueeze(1)                 # [B, 1, D]
        a = self.na(e_a)
        u = self.wa(a) + sit + self.ia(a) * self.isit(self.sit_norm(sit))
        return self.out_norm(u + self.ff(self.ff_norm(u)))


class FeatureMap(nn.Module):
    """phi [B, N, Q, F] = Phi(z [B, N, D], e_q [B, Q, D]); ||phi||_2 = 1, phi[..., -1] = 1 / sqrt(F)."""

    def __init__(self, d: int, d_ff: int, n_features: int):
        super().__init__()
        if n_features < 2:
            raise ValueError("n_features must be at least 2 (one constant feature)")
        self.n_features = n_features
        self.nz, self.nq = RMSNorm(d, affine=False), RMSNorm(d, affine=False)
        self.wz = nn.Linear(d, d, bias=False)
        self.wq = nn.Linear(d, d, bias=False)
        self.iz = nn.Linear(d, d, bias=False)
        self.iq = nn.Linear(d, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.ff = SwiGLU(d, d_ff)
        self.pre_out = RMSNorm(d)
        self.out = nn.Linear(d, n_features - 1, bias=False)
        for lin in (self.wz, self.wq, self.iz, self.iq, self.ff.w1, self.ff.w2, self.out):
            _unit_init(lin)
        _unit_init(self.ff.w3, 0.5)

    def forward(self, z: Tensor, e_q: Tensor) -> Tensor:
        zn, qn = self.nz(z).unsqueeze(2), self.nq(e_q).unsqueeze(1)                    # [B,N,1,D], [B,1,Q,D]
        u = self.wz(zn) + self.wq(qn) + self.iz(zn) * self.iq(qn)
        raw = self.out(self.pre_out(u + self.ff(self.ff_norm(u))))
        raw = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
        F = self.n_features
        const = torch.full_like(raw[..., :1], 1.0 / math.sqrt(F))
        return torch.cat([raw / math.sqrt(F), const], -1)


def modality_ids(mods: Sequence[str], device: torch.device) -> Tensor:
    return torch.tensor([MODALITY_ID.get(m, len(Modality.ALL)) for m in mods], dtype=torch.long, device=device)


def level_ids(levels: Sequence[str], device: torch.device) -> Tensor:
    return torch.tensor([LEVEL_ID.get(m, len(Level.ALL)) for m in levels], dtype=torch.long, device=device)
