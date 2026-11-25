"""Iterative latent reasoning (workstream B; implements `ReasonerP`).

    R_0 = Z_t,   R_{j+1} = Reason_theta(R_j, [Z_t; B_t; M_t; G_t])

Reason_theta is the last `cfg.reason_layers` blocks of the shared backbone (same weights), applied to
[R_j + IterEmb(j) + KindEmb(REASON) + StatePos; anchor Z_t (kind STATE) ; context tokens], followed by the backbone's
output norm. The context (Z_t anchor + belief / memory / goal tokens) is embedded once per call.

Depth: fixed J (`n_iters`, e.g. 1, 2, 4, 8, 16) or adaptive halting (PonderNet): at iteration j a halting head gives
lambda_j = sigmoid(h(R_j)); the halting distribution p_j = lambda_j prod_{i<j}(1 - lambda_i) (the last computed
iteration absorbs the remaining mass).

* Training (module.training and adaptive): all iterations are computed until the unhalted mass of every row is below
  `halt_eps` (or max_reason_iters); R = sum_j p_j R_j (differentiable in the halting head); E[iters] = sum_j j p_j;
  info["ponder"] = cfg.reason_compute_cost * E[iters] (the compute penalty to add to the loss); info["halt_kl"] =
  KL(p || Geometric(prior_lambda)) (PonderNet regulariser).
* Evaluation: each row stops at the first j where its cumulative halting probability reaches `halt_threshold`
  (or samples its halting step when `sample_halt`); halted rows are removed from the batch, so compute really drops.

Intervention hooks (for ablation / causal tests): `with reasoner.intervention(fn):` applies R_j <- fn(j, R_j) after
each iteration j (1-based); `n_iters=` forces the depth.
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import CamConfig
from ..interfaces import TokenKind, TokenSet
from .backbone import UnifiedCore
from .blocks import RMSNorm, count_parameters


class Reasoner(nn.Module):
    """forward(Z [B,S,D], ctx TokenSet [B,N,D], n_iters=None) -> (R [B,S,D], info).

    info: {"halt_p": [B,J] halting distribution over the computed iterations, "iters": [B] float (expected iterations
    in training, actual iterations in evaluation), "ponder": [B] compute penalty, "halt_kl": [B]}.

    The core is held by reference (not registered as a submodule) so the shared blocks are owned and counted once,
    by the backbone; this module's own parameters are the iteration embedding, the halting head and a context gate.
    """

    def __init__(self, cfg: CamConfig, core: UnifiedCore, default_iters: Optional[int] = None,
                 halt_threshold: float = 0.5, halt_eps: float = 1e-3, prior_lambda: float = 0.25,
                 init_lambda: float = 0.25, sample_halt: bool = False):
        super().__init__()
        self.cfg = cfg
        self._core = (core,)
        D = cfg.d_model
        self.J_max = cfg.max_reason_iters
        self.n_reason_layers = min(cfg.reason_layers, cfg.n_layers)
        self.default_iters = default_iters or cfg.reason_iters[len(cfg.reason_iters) // 2]
        self.halt_threshold, self.halt_eps = halt_threshold, halt_eps
        self.prior_lambda, self.sample_halt = prior_lambda, sample_halt
        self.iter_emb = nn.Parameter(torch.randn(self.J_max + 1, D) * 0.02)
        self.halt_norm = RMSNorm(D)
        self.halt_head = nn.Linear(D, 1)
        nn.init.normal_(self.halt_head.weight, std=0.02)
        nn.init.constant_(self.halt_head.bias, math.log(init_lambda / (1 - init_lambda)))
        self.intervene: Optional[Callable[[int, Tensor], Tensor]] = None

    @property
    def core(self) -> UnifiedCore:
        return self._core[0]

    @property
    def blocks(self):
        return self.core.blocks[len(self.core.blocks) - self.n_reason_layers:]

    @contextlib.contextmanager
    def intervention(self, fn: Callable[[int, Tensor], Tensor]):
        """Within the context, R_j <- fn(j, R_j) after each iteration j (1-based)."""
        prev, self.intervene = self.intervene, fn
        try:
            yield self
        finally:
            self.intervene = prev

    # ------------------------------------------------------------------ pieces
    def _context(self, Z: Tensor, ctx: Optional[TokenSet]) -> tuple:
        """Embedded context [B, S+N', D] = [anchor Z (kind STATE); ctx tokens] and its key mask [B, S+N']."""
        core = self.core
        B, S, _ = Z.shape
        anchor = core.embed_state(Z, TokenKind.STATE)
        if ctx is None or ctx.N == 0:
            return anchor, torch.ones(B, S, dtype=torch.bool, device=Z.device)
        sel, _, _ = core.select_budget(ctx)
        hc = core.embed_ctx(sel)
        return torch.cat([anchor, hc], 1), torch.cat([torch.ones(B, S, dtype=torch.bool, device=Z.device),
                                                      sel.mask], 1)

    def step(self, R: Tensor, j: int, hctx: Tensor, cmask: Tensor) -> Tensor:
        """One reasoning iteration: R_j [b,S,D] -> R_{j+1} [b,S,D] (float32)."""
        core = self.core
        S = R.shape[1]
        hr = core.embed_state(R, TokenKind.REASON) + self.iter_emb[min(j, self.J_max)]
        h = torch.cat([hr, hctx], 1)
        mask = torch.cat([torch.ones(R.shape[0], S, dtype=torch.bool, device=R.device), cmask], 1)
        mask = None if bool(mask.all()) else mask
        h = core.run_blocks(h, mask, self.blocks)
        return core.out_norm(h[:, :S]).float()

    def halt_lambda(self, R: Tensor) -> Tensor:
        """Conditional halting probability lambda(R) [b] from mean-pooled state tokens."""
        return torch.sigmoid(self.halt_head(self.halt_norm(R.mean(1))).squeeze(-1).float())

    def _apply_intervention(self, j: int, R: Tensor) -> Tensor:
        return self.intervene(j, R) if self.intervene is not None else R

    # ------------------------------------------------------------------ forward
    def forward(self, Z: Tensor, ctx: Optional[TokenSet] = None, n_iters: Optional[int] = None) -> tuple:
        B = Z.shape[0]
        dev = Z.device
        if not self.cfg.use_reasoning:
            zero = torch.zeros(B, device=dev)
            return Z, {"halt_p": torch.zeros(B, 0, device=dev), "iters": zero, "ponder": zero, "halt_kl": zero}
        adaptive = self.cfg.adaptive_reasoning and n_iters is None
        with self.core._autocast_ctx(dev):
            hctx, cmask = self._context(Z, ctx)
            if not adaptive:
                return self._fixed(Z, hctx, cmask, n_iters or self.default_iters)
            if self.training:
                return self._ponder_train(Z, hctx, cmask)
            return self._ponder_eval(Z, hctx, cmask)

    def _fixed(self, Z, hctx, cmask, J: int):
        B = Z.shape[0]
        R = Z.float()
        for j in range(1, J + 1):
            R = self._apply_intervention(j, self.step(R, j, hctx, cmask))
        halt_p = torch.zeros(B, J, device=Z.device)
        halt_p[:, -1] = 1.0
        iters = torch.full((B,), float(J), device=Z.device)
        return R, {"halt_p": halt_p, "iters": iters, "ponder": self.cfg.reason_compute_cost * iters,
                   "halt_kl": torch.zeros(B, device=Z.device)}

    def _ponder_train(self, Z, hctx, cmask):
        B = Z.shape[0]
        R = Z.float()
        remain = torch.ones(B, device=Z.device)
        ps, Rs = [], []
        for j in range(1, self.J_max + 1):
            R = self._apply_intervention(j, self.step(R, j, hctx, cmask))
            lam = self.halt_lambda(R)
            last = j == self.J_max
            p = remain if last else remain * lam
            ps.append(p)
            Rs.append(R)
            remain = remain - p
            if last or bool((remain < self.halt_eps).all()):
                ps[-1] = ps[-1] + remain                                    # absorb the leftover mass
                break
        P = torch.stack(ps, 1)                                              # [B,J]
        Rstack = torch.stack(Rs, 1)                                         # [B,J,S,D]
        R_out = (P[:, :, None, None] * Rstack).sum(1)
        J = P.shape[1]
        steps = torch.arange(1, J + 1, device=Z.device, dtype=P.dtype)
        iters = (P * steps).sum(1)
        lp = self.prior_lambda
        prior = lp * (1 - lp) ** (steps - 1)
        prior = prior / prior.sum()
        kl = (P * (torch.log(P.clamp_min(1e-12)) - torch.log(prior))).sum(1)
        return R_out, {"halt_p": P, "iters": iters, "ponder": self.cfg.reason_compute_cost * iters, "halt_kl": kl}

    @torch.no_grad()
    def _halt_now(self, cum: Tensor, lam: Tensor) -> Tensor:
        if self.sample_halt:
            return torch.rand_like(lam) < lam
        return cum >= self.halt_threshold

    def _ponder_eval(self, Z, hctx, cmask):
        B = Z.shape[0]
        dev = Z.device
        R_out = Z.float().clone()
        active = torch.arange(B, device=dev)
        R = Z.float()
        remain = torch.ones(B, device=dev)
        P = torch.zeros(B, self.J_max, device=dev)
        iters = torch.zeros(B, device=dev)
        J_used = 0
        for j in range(1, self.J_max + 1):
            R = self._apply_intervention(j, self.step(R, j, hctx[active], cmask[active]))
            lam = self.halt_lambda(R)
            last = j == self.J_max
            p = remain[active] if last else remain[active] * lam
            P[active, j - 1] = p
            remain[active] = remain[active] - p
            stop = torch.ones_like(lam, dtype=torch.bool) if last else self._halt_now(1 - remain[active], lam)
            J_used = j
            if bool(stop.any()):
                done = active[stop]
                R_out[done] = R[stop]
                iters[done] = float(j)
                P[done, j - 1] = P[done, j - 1] + remain[done]
                remain[done] = 0.0
                keep = ~stop
                active, R = active[keep], R[keep]
            if active.numel() == 0:
                break
        P = P[:, :J_used]
        return R_out, {"halt_p": P, "iters": iters, "ponder": self.cfg.reason_compute_cost * iters,
                       "halt_kl": torch.zeros(B, device=dev)}

    def count_parameters(self) -> dict:
        own = count_parameters(self)
        shared = count_parameters(self.blocks)
        return {"own": own, "shared_blocks_reused": shared, "reason_layers": self.n_reason_layers}
