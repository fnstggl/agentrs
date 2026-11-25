"""The shared trainable backbone F_theta of CAM Unified (workstream B; implements `UnifiedCoreP`).

    Z_t = F_theta(Z_{t-1}, [O_t; A_{t-1}; B_t; M_t; G_t])

One computation: joint self-attention over the S persistent state tokens concatenated with the context tokens, for
`cfg.n_layers` pre-norm blocks (RMSNorm -> SDPA attention -> RMSNorm -> SwiGLU). The outputs at the state positions
are Z_t; the outputs at the context positions are returned as `ctx_out` (aligned with the input context).

Context embedding (per token n):  RMSNorm(x_n) + KindEmb(kind_n) + TimeEnc(time_n) + SlotCode(entity_n)
                                    + SlotCode2(entity2_n).
State embedding (per slot s):     Z_prev[s] + StatePos[s] + KindEmb(STATE).
Context tokens get **no positional index**: the context is a set, so permuting it permutes `ctx_out` and leaves Z
unchanged (up to floating-point summation order). State tokens are ordered memory slots and do carry a learned slot
embedding.

The per-step working budget `cfg.max_ctx_tokens` is enforced by kind priority, then recency (token time closer to
now first); every dropped token is counted in `budget_stats` and the kept mask of the last call is `last_kept`. The
budget is a working-memory budget per step, never a lifetime limit (the archive keeps everything).

The world model's prior pass calls the same `forward` with ACTION / BELIEF / TIME / NOISE context tokens (one shared
transition computation); the Reasoner reuses the last `cfg.reason_layers` entries of `.blocks`.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import CamConfig
from ..interfaces import TokenKind, TokenSet
from .blocks import (Block, EntitySlotCodes, KindEmbedding, RMSNorm, TimeEncoding, count_parameters,
                     init_transformer_weights)

#: kind priority for the context budget (higher is kept first; ties broken by recency).
DEFAULT_KIND_PRIORITY = {
    TokenKind.GOAL: 10, TokenKind.ACTION: 9, TokenKind.TIME: 9, TokenKind.NOISE: 9, TokenKind.BUDGET: 8,
    TokenKind.OPTION: 8, TokenKind.BELIEF: 8, TokenKind.QUERY: 7, TokenKind.EVENT: 6, TokenKind.ENTITY: 5,
    TokenKind.RELATION: 5, TokenKind.OBS_NUM: 4, TokenKind.OBS_SEM: 4, TokenKind.MEMORY: 3, TokenKind.REASON: 2,
    TokenKind.STATE: 2, TokenKind.TOOL: 1, TokenKind.PAD: 0,
}


class UnifiedCore(nn.Module):
    """F_theta. forward(Z_prev [B,S,D], ctx TokenSet [B,N,D]) -> (Z [B,S,D] float32, ctx_out [B,N,D] float32).

    Args:
        cfg: CamConfig (n_layers, d_model, n_heads, n_state, d_ff, dropout, max_ctx_tokens, grad_checkpoint,
             max_entities, n_fourier).
        autocast: run the blocks under bf16 autocast (default: True on CUDA, False on CPU).
        kind_priority: override of DEFAULT_KIND_PRIORITY for the context budget.
        qk_norm: per-head RMS normalisation of queries/keys.
    """

    def __init__(self, cfg: CamConfig, autocast: Optional[bool] = None, kind_priority: Optional[dict] = None,
                 qk_norm: bool = True, slot_seed: int = 1234):
        super().__init__()
        self.cfg = cfg
        D, S = cfg.d_model, cfg.n_state
        self.autocast = autocast
        self.grad_checkpoint = cfg.grad_checkpoint
        self.z0 = nn.Parameter(torch.randn(S, D) * 0.02)                    # learned initial state tokens
        self.state_pos = nn.Parameter(torch.randn(S, D) * 0.02)             # identity of each state slot
        self.kind_emb = KindEmbedding(D)
        self.time_enc = TimeEncoding(D, n_freq=cfg.n_fourier)
        self.slot_codes = EntitySlotCodes(D, cfg.max_entities, seed=slot_seed)
        self.ctx_in_norm = RMSNorm(D)
        self.blocks = nn.ModuleList([Block(D, cfg.n_heads, cfg.d_ff, cfg.dropout, qk_norm=qk_norm)
                                     for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(D)
        init_transformer_weights(self.blocks, cfg.n_layers)
        init_transformer_weights(self.time_enc, cfg.n_layers)
        init_transformer_weights(self.slot_codes, cfg.n_layers)
        prio = dict(DEFAULT_KIND_PRIORITY)
        if kind_priority:
            prio.update({TokenKind(k): v for k, v in kind_priority.items()})
        self.register_buffer("kind_priority", torch.tensor([float(prio.get(TokenKind(k), 0))
                                                            for k in range(int(TokenKind.N_KINDS))],
                                                           dtype=torch.float64), persistent=False)
        self.budget_stats = {"calls": 0, "calls_over_budget": 0, "rows_over_budget": 0, "dropped_tokens": 0,
                             "dropped_by_kind": {}}
        self.last_kept: Optional[Tensor] = None

    # ------------------------------------------------------------------ state
    def init_state(self, B: int, device=None) -> Tensor:
        """Learned initial state tokens Z_0 [B,S,D]."""
        z = self.z0.unsqueeze(0).expand(B, -1, -1)
        return (z.to(device) if device is not None else z).clone()

    def new_episode(self, B: int, rows=None, generator: Optional[torch.Generator] = None) -> Tensor:
        """Fresh random per-episode entity-slot code permutations for all rows or the given rows."""
        return self.slot_codes.new_episode(B, rows=rows, generator=generator)

    def _autocast_ctx(self, device: torch.device):
        use = self.autocast if self.autocast is not None else device.type == "cuda"
        if not use:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ context budget
    def select_budget(self, ctx: TokenSet, budget: Optional[int] = None) -> tuple:
        """Enforce the working budget. Returns (ctx' TokenSet [B,k,D], idx [B,k] long | None, kept [B,N] bool).
        idx is None when no row exceeds the budget (ctx' is ctx). Selection: valid tokens by kind priority, then
        recency (larger `time` = more recent)."""
        budget = self.cfg.max_ctx_tokens if budget is None else budget
        B, N = ctx.mask.shape
        n_valid = ctx.mask.sum(1)
        self.budget_stats["calls"] += 1
        if N <= budget:
            return ctx, None, ctx.mask
        # over budget, or only padding beyond it: compact to k <= budget positions (dropping only if a row needs it)
        k = min(budget, int(n_valid.max()))
        prio = self.kind_priority.to(ctx.kind.device)[ctx.kind.clamp(0, int(TokenKind.N_KINDS) - 1)]
        score = prio * 1e7 + ctx.time.double().clamp(-1e6, 1e6)
        score = score.masked_fill(~ctx.mask, float("-inf"))
        idx = torch.topk(score, k, dim=1).indices                          # [B,k]
        kept = torch.zeros_like(ctx.mask).scatter_(1, idx, True) & ctx.mask
        over = n_valid > budget
        n_drop = int((n_valid - kept.sum(1)).sum())
        if n_drop:
            self.budget_stats["calls_over_budget"] += 1
            self.budget_stats["rows_over_budget"] += int(over.sum())
            self.budget_stats["dropped_tokens"] += n_drop
            dropped = ctx.mask & ~kept
            kinds, counts = torch.unique(ctx.kind[dropped], return_counts=True)
            for kd, c in zip(kinds.tolist(), counts.tolist()):
                name = TokenKind(kd).name
                self.budget_stats["dropped_by_kind"][name] = self.budget_stats["dropped_by_kind"].get(name, 0) + c

        def g(t):
            return torch.gather(t, 1, idx)
        x = torch.gather(ctx.x, 1, idx.unsqueeze(-1).expand(-1, -1, ctx.x.shape[-1]))
        refs = None if ctx.refs is None else [[ctx.refs[b][i] for i in idx[b].tolist()] for b in range(B)]
        sel = TokenSet(x, g(ctx.mask), g(ctx.kind), g(ctx.time), g(ctx.entity), refs,
                       None if ctx.entity2 is None else g(ctx.entity2))
        return sel, idx, kept

    # ------------------------------------------------------------------ embeddings and blocks
    def embed_ctx(self, ctx: TokenSet, code_rows: Optional[Tensor] = None) -> Tensor:
        """Context token embeddings h [B,N,D] (float32). No positional index (set semantics)."""
        h = self.ctx_in_norm(ctx.x.float())
        h = h + self.kind_emb(ctx.kind) + self.time_enc(ctx.time)
        h = h + self.slot_codes(ctx.entity, 0, code_rows)
        if ctx.entity2 is not None:
            h = h + self.slot_codes(ctx.entity2, 1, code_rows)
        return h * ctx.mask.unsqueeze(-1).to(h.dtype)

    def embed_state(self, Z: Tensor, kind: int = TokenKind.STATE) -> Tensor:
        """State token embeddings [B,S,D]: Z + StatePos + KindEmb(kind)."""
        return Z.float() + self.state_pos + self.kind_emb.weight[int(kind)]

    def run_blocks(self, h: Tensor, key_mask: Optional[Tensor], blocks=None,
                   attn_bias: Optional[Tensor] = None) -> Tensor:
        """Apply `blocks` (default: all) to h [B,L,D] with key-padding mask [B,L] | None. Activation checkpointing
        per block when `grad_checkpoint` is set and gradients are being recorded."""
        blocks = self.blocks if blocks is None else blocks
        ckpt = self.grad_checkpoint and torch.is_grad_enabled() and self.training
        for blk in blocks:
            if ckpt:
                h = checkpoint(blk, h, key_mask, attn_bias, use_reentrant=False)
            else:
                h = blk(h, key_mask, attn_bias)
        return h

    # ------------------------------------------------------------------ forward
    def forward(self, Z_prev: Tensor, ctx: TokenSet, code_rows: Optional[Tensor] = None) -> tuple:
        """Z_prev [B,S,D], ctx [B,N,D] -> (Z [B,S,D] float32, ctx_out [B,N,D] float32; zero at padded/dropped
        positions). `code_rows` [B] maps batch rows to per-episode slot-code rows (imagination branches)."""
        B, S, D = Z_prev.shape
        assert S == self.cfg.n_state and D == self.cfg.d_model, (Z_prev.shape, self.cfg.n_state, self.cfg.d_model)
        assert ctx.B == B, (ctx.B, B)
        N = ctx.N
        sel, idx, kept = self.select_budget(ctx)
        self.last_kept = kept
        with self._autocast_ctx(Z_prev.device):
            hs = self.embed_state(Z_prev)
            hc = self.embed_ctx(sel, code_rows)
            h = torch.cat([hs, hc], 1)
            key_mask = torch.cat([torch.ones(B, S, dtype=torch.bool, device=h.device), sel.mask], 1)
            if bool(key_mask.all()):
                key_mask = None                                    # nothing masked: flash kernel eligible
            h = self.run_blocks(h, key_mask)
            h = self.out_norm(h).float()
        Z = h[:, :S]
        out_sel = h[:, S:] * sel.mask.unsqueeze(-1).float()
        if idx is None:
            return Z, out_sel
        ctx_out = torch.zeros(B, N, D, device=h.device, dtype=out_sel.dtype)
        ctx_out = ctx_out.scatter(1, idx.unsqueeze(-1).expand(-1, -1, D), out_sel)
        return Z, ctx_out

    # ------------------------------------------------------------------ accounting
    def count_parameters(self) -> dict:
        """Trainable parameters per component of the backbone."""
        out = {
            "state_init": self.z0.numel() + self.state_pos.numel(),
            "kind_embedding": count_parameters(self.kind_emb),
            "time_encoding": count_parameters(self.time_enc),
            "entity_slot_codes": count_parameters(self.slot_codes),
            "ctx_in_norm": count_parameters(self.ctx_in_norm),
            "blocks": count_parameters(self.blocks),
            "per_block": count_parameters(self.blocks[0]) if len(self.blocks) else 0,
            "out_norm": count_parameters(self.out_norm),
        }
        out["total"] = count_parameters(self)
        return out
