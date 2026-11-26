"""cam_live.credit.rudder — RUDDER-style return decomposition for delayed verified outcomes (spec 9.2; RC-02, RC-03).

``ReturnDecomposer`` is a causal sequence model F_xi over per-decision latents that predicts the eventual verified
return of a trajectory from every prefix. Its prediction differences redistribute the return over the decisions.

Conventions (all tested):

* **Target.** Squared error on the *linearly* scaled return G / scale, at every prefix, so F estimates the
  conditional mean E[G | H_t]. A nonlinear target transform (symlog, Huber) would make the differences of
  predictions stop being contributions even with infinite data, so there is none.
* **Indexing.** ``pred[:, t]`` is the prediction *before* decision t (it has seen decisions < t only) and
  ``pred[:, t + 1]`` the prediction after it; ``pred[:, 0]`` is the initial offset F(empty).
* **Discounting.** Decision t happens ``tau_t`` seconds after the first decision of the trajectory and
  Gamma_t = exp(-beta * tau_t). G is the return referenced at the trajectory start. Contributions are expressed at
  their own decision time:  c_t = [F(H_<=t) - F(H_<t)] / Gamma_t, plus F(empty) / Gamma_first on the first step and
  the residual [G - F(H_<=T)] / Gamma_last on the last, hence  sum_t Gamma_t c_t = G  exactly, for any F.
* **Learning signal.** The redistributed return-to-go with the predictor's own prefix value as baseline:
  signal_t = [G - F(H_<t)] / Gamma_t = (sum_{k>=t} Gamma_k c_k) / Gamma_t  (without the offset share). It differs
  from the true discounted return-to-go only by a function of the past, so it is an unbiased policy-gradient
  signal for *any* F. Using c_t itself as an immediate reward is only correct for a perfect decomposition (a
  decomposition that explains the return late gives the key decision zero credit).
* **No fabricated return.** Trajectories whose outcome has not matured contribute no loss and receive no
  redistributed credit: their rows are NaN, never 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NAME = "rudder"


def duration_discount(tau: Tensor, beta: float) -> Tensor:
    """Gamma = exp(-beta * tau) for elapsed seconds ``tau`` (float64)."""
    return torch.exp(-float(beta) * tau.double().clamp_min(0.0))


def discounted_return(value: Tensor, outcome_time: Tensor, start_time: Tensor, beta: float) -> Tensor:
    """A verified outcome value observed at ``outcome_time``, referenced at the trajectory start."""
    return value.double() * duration_discount(outcome_time.double() - start_time.double(), beta)


class _CausalBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


@dataclass
class Redistribution:
    """Result of :meth:`ReturnDecomposer.redistribute` (float64). Rows that are not matured are NaN throughout."""
    contrib: Tensor          # [B, T] c_t including the offset (first step) and the residual (last step)
    diff: Tensor             # [B, T] prediction differences alone, [F(H_<=t) - F(H_<t)] / Gamma_t
    signal: Tensor           # [B, T] learning signal: redistributed return-to-go minus the prefix prediction
    return_to_go: Tensor     # [B, T] (sum_{k>=t} Gamma_k c_k) / Gamma_t
    pred: Tensor             # [B, T+1] predicted return after each prefix (return units; a prediction, not an outcome)
    offset: Tensor           # [B] F(empty)
    residual: Tensor         # [B] G - F(H_<=T)
    discount: Tensor         # [B, T] Gamma_t
    mask: Tensor             # [B, T] bool
    matured: Tensor          # [B] bool: rows that received credit

    def conservation_error(self, G: Tensor) -> Tensor:
        """|sum_t Gamma_t c_t - G| per matured row (NaN for the others)."""
        tot = torch.where(self.mask, self.discount * self.contrib, torch.zeros_like(self.contrib)).sum(1)
        err = (tot - G.double()).abs()
        return torch.where(self.matured, err, torch.full_like(err, float("nan")))


class ReturnDecomposer(nn.Module):
    """F_xi(H_<=t, g): causal transformer over per-decision latents predicting the eventual verified return.

    d_in      width of the per-decision latent (e.g. ``DecisionBatch.z_chosen`` along a trajectory)
    d_goal    width of an optional goal embedding added to every position
    beta      rate of the duration-aware discount exp(-beta * seconds)
    """

    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, d_goal: int = 0,
                 beta: float = 0.0, scale: float = 1.0, dropout: float = 0.0, time_unit_s: float = 60.0):
        super().__init__()
        self.beta = float(beta)
        self.time_unit_s = float(time_unit_s)
        self.d_model = d_model
        self.inp = nn.Linear(d_in, d_model)
        self.goal = nn.Linear(d_goal, d_model) if d_goal else None
        self.time = nn.Linear(2, d_model)
        self.start = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList([_CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("scale", torch.tensor(float(scale)))

    # ------------------------------------------------------------------------------------------------ scale
    @torch.no_grad()
    def fit_scale(self, G: Tensor, matured: Optional[Tensor] = None) -> float:
        """Set the linear return scale to the largest matured |G| (RUDDER's r * c / g_max). Linear only."""
        g = G.double() if matured is None else G.double()[matured.bool()]
        if g.numel():
            self.scale.fill_(max(float(g.abs().max()), 1e-8))
        return float(self.scale)

    # ------------------------------------------------------------------------------------------------ forward
    def _positions(self, L: int, device) -> Tensor:
        pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32) *
                        (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(L, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : self.d_model // 2]
        return pe

    def _tau(self, mask: Tensor, times: Optional[Tensor]) -> Tensor:
        """Seconds since the first valid decision of each row [B, T] (0 on padding and without times)."""
        m = mask.bool()
        if times is None:
            return torch.zeros(m.shape, dtype=torch.float64, device=m.device)
        t = times.double()
        first = torch.where(m, t, torch.full_like(t, float("inf"))).min(1, keepdim=True).values
        first = torch.where(torch.isfinite(first), first, torch.zeros_like(first))
        return torch.where(m, (t - first).clamp_min(0.0), torch.zeros_like(t))

    def forward(self, x: Tensor, mask: Tensor, e_g: Optional[Tensor] = None, times: Optional[Tensor] = None) -> Tensor:
        """x [B, T, d_in], mask [B, T] (True = real decision), e_g [B, d_goal], times [B, T] decision times (s)
        -> predictions of G / scale [B, T+1]; index t has seen decisions < t only."""
        B, T, _ = x.shape
        m = mask.bool()
        h = self.inp(x.float())
        tau = self._tau(m, times)
        dt = torch.diff(tau, dim=1, prepend=torch.zeros_like(tau[:, :1])).clamp_min(0.0)
        feats = torch.stack([torch.log1p(tau / self.time_unit_s), torch.log1p(dt / self.time_unit_s)], -1).float()
        h = h + self.time(feats)
        s = self.start.expand(B, 1, -1)
        if self.goal is not None and e_g is not None:
            g = self.goal(e_g.float())[:, None]
            s, h = s + g, h + g
        h = torch.cat([s, h], 1) + self._positions(T + 1, x.device)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), m], 1)          # [B, T+1] keys
        causal = torch.tril(torch.ones(T + 1, T + 1, dtype=torch.bool, device=x.device))
        attn = causal[None, None] & keep[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, attn)
        return self.head(self.norm(h)).squeeze(-1)

    # ------------------------------------------------------------------------------------------------ training
    def loss(self, x: Tensor, mask: Tensor, G: Tensor, matured: Tensor, e_g: Optional[Tensor] = None,
             times: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """Mean squared error between every prefix prediction and the scaled eventual return of *matured*
        trajectories. Returns ``(None, ...)`` when nothing in the batch has matured: no target is invented."""
        mat = matured.bool() & mask.bool().any(1)
        n = int(mat.sum())
        if n == 0:
            return None, {"n_matured": 0.0}
        pred = self(x, mask, e_g, times)
        tgt = torch.where(mat, G.float(), torch.zeros_like(G.float())) / self.scale
        valid = torch.cat([torch.ones_like(mask[:, :1], dtype=torch.bool), mask.bool()], 1) & mat[:, None]
        err2 = (pred - tgt[:, None]) ** 2
        loss = (err2 * valid).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            last = mask.long().sum(1)                                   # index of the final prediction
            final = pred.gather(1, last[:, None]).squeeze(1)
            fe = ((final - tgt) ** 2)[mat].mean() * self.scale ** 2
        return loss, {"n_matured": float(n), "mse_final": float(fe), "mse_prefix": float(loss.detach()) * float(self.scale) ** 2}

    # ------------------------------------------------------------------------------------------------ credit
    @torch.no_grad()
    def redistribute(self, x: Tensor, mask: Tensor, G: Tensor, matured: Tensor, e_g: Optional[Tensor] = None,
                     times: Optional[Tensor] = None) -> Redistribution:
        """Redistribute each matured trajectory's realised return G [B] over its decisions (see module docstring).
        Bookkeeping is done in float64 so conservation holds to numerical precision."""
        was_training = self.training
        self.eval()
        try:
            pred = self(x, mask, e_g, times).double() * float(self.scale)       # [B, T+1]
        finally:
            self.train(was_training)
        m = mask.bool()
        T = m.shape[1]
        filled = pred.clone()                       # carry the last valid prediction across padding
        for t in range(1, T + 1):
            filled[:, t] = torch.where(m[:, t - 1], pred[:, t], filled[:, t - 1])
        gamma = torch.where(m, duration_discount(self._tau(m, times), self.beta).clamp_min(1e-300),
                            torch.ones(m.shape, dtype=torch.float64, device=m.device))
        mf = m.double()
        cnt = m.long().cumsum(1)
        first = (cnt == 1) & m
        last = (cnt == m.long().sum(1, keepdim=True)) & m
        Gd = G.double()
        offset, residual = filled[:, 0], Gd - filled[:, -1]
        raw_diff = (filled[:, 1:] - filled[:, :-1]) * mf                         # start-referenced
        raw = raw_diff + first.double() * offset[:, None] + last.double() * residual[:, None]
        togo = torch.flip(torch.cumsum(torch.flip(raw, [1]), 1), [1])
        mat = matured.bool() & m.any(1)
        nan = torch.full_like(raw, float("nan"))
        ok = mat[:, None] & m

        def out(v: Tensor) -> Tensor:
            return torch.where(ok, v, torch.where(mat[:, None], torch.zeros_like(v), nan))

        return Redistribution(
            contrib=out(raw / gamma), diff=out(raw_diff / gamma), signal=out((Gd[:, None] - filled[:, :-1]) / gamma),
            return_to_go=out(togo / gamma), pred=filled, offset=offset,
            residual=torch.where(mat, residual, torch.full_like(residual, float("nan"))), discount=gamma, mask=m,
            matured=mat)


def write_credit(store, exp_ids: Sequence[Sequence[str]], red: Redistribution, *, name: str = NAME) -> int:
    """Write the redistributed learning signal of matured trajectories to ``record.credit[name]`` (and the raw
    contribution to ``name + '_contribution'``). ``exp_ids[b][t]`` is the record of decision t of row b. Rows that
    have not matured are skipped: nothing is written for them. Returns the number of records written."""
    n = 0
    with store.batch():
        for b, ids in enumerate(exp_ids):
            if not bool(red.matured[b]):
                continue
            steps = torch.nonzero(red.mask[b]).flatten().tolist()
            for t, exp_id in zip(steps, ids):
                store.set_credit(exp_id, name, float(red.signal[b, t]))
                store.set_credit(exp_id, name + "_contribution", float(red.contrib[b, t]))
                n += 1
    return n
