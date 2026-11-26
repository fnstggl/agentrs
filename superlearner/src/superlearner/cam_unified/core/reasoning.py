"""Iterative latent reasoning (workstream B; implements `ReasonerP`).

    R_0 = Z_t,   R_{j+1} = Reason_theta(R_j, [Z_t; B_t; M_t; G_t])

Reason_theta is the last `cfg.reason_layers` blocks of the shared backbone (same weights), applied to
[R_j + IterEmb(j) + KindEmb(REASON) + StatePos; anchor Z_t (kind STATE) ; context tokens], followed by the backbone's
output norm. The context (Z_t anchor + belief / memory / goal tokens) is embedded once per call and its residual
stream is carried from iteration to iteration (weight-tied looped computation), so later iterations can build on
intermediate results held by context tokens.

Depth
-----
* Fixed J (`n_iters=`, e.g. 1, 2, 4, 8, 16; or `cfg.reason_iters_fixed = J > 0`, the dom presets' J = 2 until
  halting is trained; or `cfg.adaptive_reasoning=False` with the default J): R = R_J. J <= cfg.max_reason_iters
  (32 in the dom presets).
* Adaptive (default): at iteration j a halting head gives lambda_j = sigmoid(h(R_j)); the halting distribution is
  p_j = lambda_j prod_{i<j}(1 - lambda_i). **Bounded unroll, per row:** a row stops after the first iteration at which
  its unhalted mass prod_{i<=j}(1 - lambda_i) falls below `halt_eps` (default 0.05, reachable: at the initial
  lambda = 0.25 — zero-initialised halting weights — that is iteration 11 for every row), or at `max_iters`
  (default cfg.max_reason_iters); its leftover mass is absorbed
  by that last iteration. Finished rows leave the batch, so compute really drops as halting is learned (in training
  and evaluation). In training the unroll is activation-checkpointed per iteration by default
  (`checkpoint_iters=None` = auto), so memory does not grow with the data-dependent depth (each iteration stores only
  its inputs R_j and the context stream).

  Two halting objectives (`halting=`):

  - "act" (default; Graves' ACT output with PonderNet's halting distribution): R = sum_j p_j R_j, in training **and**
    evaluation, with the same truncation rule — the same deterministic function in both modes (no train/eval
    mismatch), and the halting head is trained directly by any loss on R.
  - "pondernet": the PonderNet objective sum_j p_j L(R_j). Training computes the full truncated distribution and
    returns R = R_{j*} with j* ~ p (a sample of the output distribution) plus info["R_all"] [B,J,S,D]; the loss
    must be formed with `Reasoner.expected_loss(loss_fn, info)` (= sum_j p_j loss_fn(R_j)) — that is what trains the
    halting head. Evaluation halts each row at a step drawn from the same distribution (`sample_halt=True`) or at the
    first step whose cumulative halting probability reaches `halt_threshold` (median, default) and returns that R_j.

  info["ponder"] = cfg.reason_compute_cost * E_p[iters] (the compute penalty to add to the loss); info["halt_kl"] =
  KL(p || truncated Geometric(prior_lambda)) (PonderNet regulariser).

Imagination branches
--------------------
`code_rows` [B'] maps batch rows to per-episode entity slot-code rows (as in `UnifiedCore.forward`), so the reasoner
runs on branch batches of any size after `core.new_episode(B)`.

Semantic context (DOM Mechanism C)
----------------------------------
`sem` (a `core.semantic_xattn.SemanticContext`, rows aligned as in `UnifiedCore.project_semantic`: the batch's rows,
or source rows selected by `sem_rows` / `code_rows`) is projected once per call and passed as an explicit input to
every (checkpointed) iteration, so the cross-attention layers among the reused blocks read it; rows that halt leave
the semantic batch with their state. Nothing is held as module state: recomputation in the backward pass uses the
tensors of the original forward even when other decisions were encoded in between.

Interventions (ablations / causal tests)
----------------------------------------
`with reasoner.intervention(fn):` applies R_j <- fn(j, R_j) after each iteration j (1-based);
`with reasoner.intervention(fn, context=True):` applies (R_j, C_j) <- fn(j, R_j, C_j, rows) where C_j [b,L,D] is the
carried context stream after iteration j (anchor Z then context tokens) and rows [b] the batch rows still active.
`n_iters=` forces the depth.
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import CamConfig
from ..interfaces import TokenKind, TokenSet
from .backbone import UnifiedCore
from .blocks import RMSNorm, count_parameters
from .semantic_xattn import SemanticContext, SemanticKV

HALTING_MODES = ("act", "pondernet")


class Reasoner(nn.Module):
    """forward(Z [B,S,D], ctx TokenSet [B,N,D] | None, n_iters=None, code_rows=None, sem=None) -> (R [B,S,D] float32,
    info).

    info (all modes): "halt_p" [B,J] halting distribution over the computed iterations (zero after a row stopped),
    "iters" [B] (training: expected iterations E_p[j], differentiable; evaluation: iterations computed), "ponder" [B]
    = reason_compute_cost * iters, "halt_kl" [B], "expected_iters" [B], "n_computed" [B] (iterations actually computed
    per row). Adaptive training also returns "R_all" [B,J,S,D] (R_j per iteration; rows that stopped repeat their last
    state with zero halting probability).

    The core is held by reference (not registered as a submodule) so the shared blocks are owned and counted once,
    by the backbone; this module's own parameters are the iteration embedding and the halting head.
    """

    def __init__(self, cfg: CamConfig, core: UnifiedCore, default_iters: Optional[int] = None,
                 halting: str = "act", halt_eps: float = 0.05, max_iters: Optional[int] = None,
                 halt_threshold: float = 0.5, sample_halt: bool = False, prior_lambda: float = 0.25,
                 init_lambda: float = 0.25, iter_emb_std: float = 0.5, checkpoint_iters: Optional[bool] = None):
        super().__init__()
        assert halting in HALTING_MODES, halting
        assert 0.0 < halt_eps < 1.0, halt_eps
        self.cfg = cfg
        self._core = (core,)
        D = cfg.d_model
        self.J_max = cfg.max_reason_iters
        self.max_iters = min(max_iters or self.J_max, self.J_max)
        self.n_reason_layers = min(cfg.reason_layers, cfg.n_layers)
        self.default_iters = default_iters or cfg.reason_iters[len(cfg.reason_iters) // 2]
        self.halting = halting
        self.halt_threshold, self.halt_eps = halt_threshold, halt_eps
        self.prior_lambda, self.sample_halt = prior_lambda, sample_halt
        self.checkpoint_iters = checkpoint_iters
        # R is RMS-normalised (unit scale per element): the iteration embedding must start at a comparable scale,
        # or every iteration computes the same function and iterations cannot specialise (measured: with std 0.02 a
        # 2-iteration lookup composition is not learned in 4000 steps; with 0.5 it is learned in < 3000).
        self.iter_emb = nn.Parameter(torch.randn(self.J_max + 1, D) * iter_emb_std)
        self.halt_norm = RMSNorm(D)
        self.halt_head = nn.Linear(D, 1)
        # zero weights: at initialisation lambda_j = init_lambda for every iteration and row, so the initial halting
        # distribution is exactly Geometric(init_lambda) and the initial unroll is deterministic,
        # ceil(log(halt_eps) / log(1 - init_lambda)) = 11 iterations with the defaults, at every scale (with random
        # weights lambda drifts with R_j — measured at the main scale: below 0.25, so every row ran 16 iterations).
        # The weights still receive gradients (d logit / d w = halt_norm(mean R)).
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, math.log(init_lambda / (1 - init_lambda)))
        self.intervene: Optional[tuple] = None             # (fn, context: bool)
        self._ckpt_now = False
        self.last_gates: dict = {}                         # {iteration: {block index: mean semantic gate}} (detached)

    @property
    def fixed_iters(self) -> Optional[int]:
        """J of the fixed-depth mode selected by the config (`cfg.reason_iters_fixed`), None = adaptive/default."""
        j = int(getattr(self.cfg, "reason_iters_fixed", 0) or 0)
        return min(j, self.J_max) if j > 0 else None

    @property
    def core(self) -> UnifiedCore:
        return self._core[0]

    @property
    def blocks(self):
        return self.core.blocks[len(self.core.blocks) - self.n_reason_layers:]

    @contextlib.contextmanager
    def intervention(self, fn: Callable, context: bool = False):
        """Within the context: R_j <- fn(j, R_j) after each iteration j (1-based); with context=True,
        (R_j, C_j) <- fn(j, R_j, C_j, rows) where C_j is the carried context stream and rows the active batch rows."""
        prev, self.intervene = self.intervene, (fn, context)
        try:
            yield self
        finally:
            self.intervene = prev

    # ------------------------------------------------------------------ pieces
    def _context(self, Z: Tensor, ctx: Optional[TokenSet], code_rows: Optional[Tensor] = None) -> tuple:
        """Embedded context [B, S+N', D] = [anchor Z (kind STATE); ctx tokens] and its key mask [B, S+N']."""
        core = self.core
        B, S, _ = Z.shape
        anchor = core.embed_state(Z, TokenKind.STATE)
        if ctx is None or ctx.N == 0:
            return anchor, torch.ones(B, S, dtype=torch.bool, device=Z.device)
        assert ctx.B == B, (ctx.B, B)
        sel, _, _ = core.select_budget(ctx)
        hc = core.embed_ctx(sel, code_rows)
        return torch.cat([anchor, hc], 1), torch.cat([torch.ones(B, S, dtype=torch.bool, device=Z.device),
                                                      sel.mask], 1)

    def _step_impl(self, R: Tensor, hctx: Tensor, mask: Optional[Tensor], j: int,
                   checkpoint_blocks: Optional[bool] = None, sem: Optional[SemanticKV] = None,
                   gate_log: Optional[dict] = None) -> tuple:
        core = self.core
        S = R.shape[1]
        hr = core.embed_state(R, TokenKind.REASON) + self.iter_emb[min(j, self.J_max)]
        h = torch.cat([hr, hctx], 1)
        if sem is None:
            h = core.run_blocks(h, mask, self.blocks, checkpoint_blocks=checkpoint_blocks)
        else:
            h = core.run_blocks(h, mask, self.blocks, checkpoint_blocks=checkpoint_blocks, sem=sem,
                                gate_log=gate_log)
        return core.out_norm(h[:, :S]).float(), h[:, S:]

    def step(self, R: Tensor, j: int, hctx: Tensor, cmask: Tensor, sem: Optional[SemanticKV] = None,
             gate_log: Optional[dict] = None) -> tuple:
        """One reasoning iteration: (R_j [b,S,D], context stream [b,L,D]) -> (R_{j+1} [b,S,D] float32, context
        stream after the reused blocks). The context's residual stream is carried across iterations, so J
        iterations of k reused blocks compose like J*k weight-tied layers over [R; Z; ctx] (context tokens can
        accumulate what earlier iterations computed, e.g. which item matched a query). Checkpointed as one unit
        when iteration checkpointing is active (only R_j and the context stream are stored)."""
        b, S = R.shape[:2]
        mask = torch.cat([torch.ones(b, S, dtype=torch.bool, device=R.device), cmask], 1)
        mask = None if bool(mask.all()) else mask
        if self._ckpt_now:
            # the flag and the semantic memory are arguments (not read from self during recomputation, when forward
            # has already returned): no nested per-block checkpoints inside the iteration checkpoint, and the
            # recomputation reads exactly the semantic tensors of this forward
            if sem is None:
                return checkpoint(self._step_impl, R, hctx, mask, j, False, use_reentrant=False)
            return checkpoint(self._step_impl, R, hctx, mask, j, False, sem, gate_log, use_reentrant=False)
        if sem is None:
            return self._step_impl(R, hctx, mask, j)
        return self._step_impl(R, hctx, mask, j, None, sem, gate_log)

    def _iterate(self, R, j, hctx, cmask, kv: Optional[SemanticKV], log: Optional[dict]) -> tuple:
        # without a semantic context `step` is called exactly as before (spies and subclasses keep working)
        if kv is None:
            return self.step(R, j, hctx, cmask)
        return self.step(R, j, hctx, cmask, kv, log.setdefault(j, {}))

    def halt_lambda(self, R: Tensor) -> Tensor:
        """Conditional halting probability lambda(R) [b] from mean-pooled state tokens."""
        return torch.sigmoid(self.halt_head(self.halt_norm(R.mean(1))).squeeze(-1).float())

    def _apply_intervention(self, j: int, R: Tensor, hctx: Tensor, rows: Tensor) -> tuple:
        if self.intervene is None:
            return R, hctx
        fn, with_ctx = self.intervene
        if with_ctx:
            return fn(j, R, hctx, rows)
        return fn(j, R), hctx

    def _use_iter_checkpoint(self, adaptive: bool) -> bool:
        if not (self.training and torch.is_grad_enabled()):
            return False
        if self.checkpoint_iters is not None:
            return bool(self.checkpoint_iters)
        return adaptive                       # data-dependent depth: keep activation memory independent of it

    # ------------------------------------------------------------------ forward
    def forward(self, Z: Tensor, ctx: Optional[TokenSet] = None, n_iters: Optional[int] = None,
                code_rows: Optional[Tensor] = None, sem: Optional[SemanticContext] = None, *,
                sem_rows: Optional[Tensor] = None) -> tuple:
        B = Z.shape[0]
        dev = Z.device
        if not self.cfg.use_reasoning:
            zero = torch.zeros(B, device=dev)
            return Z, {"halt_p": torch.zeros(B, 0, device=dev), "iters": zero, "ponder": zero, "halt_kl": zero,
                       "expected_iters": zero, "n_computed": zero}
        J_fixed = n_iters if n_iters is not None else self.fixed_iters
        adaptive = self.cfg.adaptive_reasoning and J_fixed is None
        self._ckpt_now = self._use_iter_checkpoint(adaptive)
        log: dict = {}
        try:
            with self.core._autocast_ctx(dev):
                hctx, cmask = self._context(Z, ctx, code_rows)
                kv = self.core.project_semantic(sem, B, code_rows, sem_rows) if sem is not None else None
                if not adaptive:
                    out = self._fixed(Z, hctx, cmask, J_fixed or self.default_iters, kv, log)
                else:
                    out = self._adaptive(Z, hctx, cmask, kv, log)
        finally:
            self._ckpt_now = False
        if kv is not None:
            self.last_gates = log
        return out

    def _fixed(self, Z, hctx, cmask, J: int, kv: Optional[SemanticKV] = None, log: Optional[dict] = None):
        B = Z.shape[0]
        R = Z.float()
        rows = torch.arange(B, device=Z.device)
        for j in range(1, J + 1):
            R, hctx = self._iterate(R, j, hctx, cmask, kv, log)
            R, hctx = self._apply_intervention(j, R, hctx, rows)
        halt_p = torch.zeros(B, J, device=Z.device)
        halt_p[:, -1] = 1.0
        iters = torch.full((B,), float(J), device=Z.device)
        return R, {"halt_p": halt_p, "iters": iters, "ponder": self.cfg.reason_compute_cost * iters,
                   "halt_kl": torch.zeros(B, device=Z.device), "expected_iters": iters, "n_computed": iters.clone()}

    @torch.no_grad()
    def _halt_now(self, cum: Tensor, lam: Tensor) -> Tensor:
        """PonderNet evaluation halting decision of the active rows (given they were not stopped by truncation)."""
        if self.sample_halt:
            return torch.rand_like(lam) < lam
        return cum >= self.halt_threshold

    def _adaptive(self, Z: Tensor, hctx: Tensor, cmask: Tensor, kv: Optional[SemanticKV] = None,
                  log: Optional[dict] = None):
        B, S, D = Z.shape
        dev = Z.device
        train = self.training
        pn_eval = self.halting == "pondernet" and not train
        J = self.max_iters
        R = Z.float()
        active = torch.arange(B, device=dev)
        remain = torch.ones(B, device=dev)                      # unhalted mass per row
        R_acc = torch.zeros(B, S, D, device=dev)                # ACT mixture sum_j p_j R_j
        R_last = Z.float()                                      # latest state per row ([B,S,D], R_all padding)
        R_halt = Z.float()                                      # PonderNet eval: state at the halting step
        n_comp = torch.zeros(B, dtype=torch.long, device=dev)
        ps, R_all = [], []
        for j in range(1, J + 1):
            R, hctx = self._iterate(R, j, hctx, cmask, kv, log)
            R, hctx = self._apply_intervention(j, R, hctx, active)
            lam = self.halt_lambda(R)
            rem_a = remain[active]
            last = j == J
            p = rem_a if last else rem_a * lam
            finish = torch.ones_like(lam, dtype=torch.bool) if last else (rem_a - p).detach() < self.halt_eps
            stop = finish | self._halt_now(1 - (rem_a - p).detach(), lam) if pn_eval else finish
            p = torch.where(stop, rem_a, p)                     # the last computed iteration absorbs the leftover
            ps.append(torch.zeros(B, device=dev, dtype=p.dtype).index_copy(0, active, p))
            remain = remain.index_copy(0, active, rem_a - p)
            R_acc = R_acc.index_add(0, active, p[:, None, None] * R)
            R_last = R_last.index_copy(0, active, R)
            if train:
                R_all.append(R_last)
            if pn_eval and bool(stop.any()):
                R_halt = R_halt.index_copy(0, active[stop], R[stop])
            n_comp[active] = j
            if bool(stop.any()):
                keep = ~stop
                active, R, hctx, cmask = active[keep], R[keep], hctx[keep], cmask[keep]
                if kv is not None:
                    kv = kv.select(keep.nonzero().squeeze(1))
            if active.numel() == 0:
                break
        P = torch.stack(ps, 1)                                  # [B,J_used]
        Ju = P.shape[1]
        steps = torch.arange(1, Ju + 1, device=dev, dtype=P.dtype)
        exp_iters = (P * steps).sum(1)
        kl = self._halt_kl(P, n_comp)
        info = {"halt_p": P, "expected_iters": exp_iters, "n_computed": n_comp.float(), "halt_kl": kl}
        if train:
            info["R_all"] = torch.stack(R_all, 1)               # [B,J_used,S,D]
        if self.halting == "act":
            R_out = R_acc
        elif train:                                             # PonderNet: a sample of the output distribution
            jstar = torch.multinomial(P.detach().clamp_min(0) + 1e-12, 1).squeeze(1)
            R_out = info["R_all"][torch.arange(B, device=dev), jstar]
            info["sampled_iter"] = (jstar + 1).float()
        else:
            R_out = R_halt
        info["iters"] = exp_iters if train else n_comp.float()
        info["ponder"] = self.cfg.reason_compute_cost * info["iters"]
        return R_out, info

    def _halt_kl(self, P: Tensor, n_comp: Tensor) -> Tensor:
        """KL(p || prior) per row, prior = Geometric(prior_lambda) truncated at the row's last computed iteration
        (tail mass on that iteration, as for p)."""
        Ju = P.shape[1]
        j = torch.arange(1, Ju + 1, device=P.device, dtype=P.dtype)
        lp = self.prior_lambda
        geo = lp * (1 - lp) ** (j - 1)
        tail = (1 - lp) ** (j - 1)
        n = n_comp.to(P.dtype).unsqueeze(1)
        prior = torch.where(j < n, geo, torch.where(j == n, tail, torch.zeros_like(geo)))
        inside = j <= n
        terms = P * (torch.log(P.clamp_min(1e-12)) - torch.log(prior.clamp_min(1e-12)))
        return torch.where(inside, terms, torch.zeros_like(terms)).sum(1)

    @staticmethod
    def expected_loss(loss_fn: Callable[[Tensor], Tensor], info: dict) -> Tensor:
        """PonderNet objective per row: sum_j p_j loss_fn(R_j) [B], with loss_fn(R [B,S,D]) -> [B]. Needs the
        "R_all" of an adaptive training call; gradients reach the halting head through p and every R_j."""
        R_all, P = info["R_all"], info["halt_p"]
        L = torch.stack([loss_fn(R_all[:, j]) for j in range(P.shape[1])], 1)
        return (P * L).sum(1)

    def count_parameters(self) -> dict:
        own = count_parameters(self)
        shared = count_parameters(self.blocks)
        return {"own": own, "shared_blocks_reused": shared, "reason_layers": self.n_reason_layers}
