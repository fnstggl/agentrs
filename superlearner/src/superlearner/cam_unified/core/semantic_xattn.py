"""CAM-controlled gated semantic cross-attention inside the core (DOM amendment Mechanism C; manifest DOM-05).

At selected CAM layers l (every `cfg.sem_xattn_every` blocks):

    R_l     = F_theta,l(Z_l)                                   (the ordinary pre-norm block: self-attention + SwiGLU)
    C_l     = CrossAttention_theta,l(Z_l, S_t, S_t)            (queries: CAM's tokens; keys/values: semantic tokens)
    g_l     = sigmoid(Gate_theta,l(Z_l, C_l, B_t, G_t))        (per token and per feature)
    Z_{l+1} = Z_l + R_l + g_l * C_l

S_t are the token-level hidden states of the pretrained semantic model for the current step's semantic content,
projected once to CAM's width by a trainable `SemanticProjection` (RMS-normalised, so the semantic pathway cannot
dominate by scale). The gate bias starts negative (g ~ 0.05): CAM begins mostly self-driven and learns how much
semantic information each token needs; gate statistics are recorded every call (`last_gates`) so experiments can
report how much CAM reads from the semantic pathway. There is no fixed activation ratio.

`GatedUnifiedCore` is `UnifiedCore` with these modules interleaved; with no semantic input (or
`sem_xattn_every == 0`) it is exactly `UnifiedCore` (the cross-attention is skipped, not zero-filled). Gradients flow
into the projection, the cross-attention, the gates and, through the attended states, to every block; a frozen
semantic model is a constant input (its parameters receive nothing) but CAM's conditioning of it is trainable
(`semantic/decoder_xattn.py` for the opposite direction, Mechanism A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import CamConfig
from ..interfaces import TokenSet
from .backbone import UnifiedCore
from .blocks import MultiHeadAttention, RMSNorm


@dataclass
class SemanticContext:
    """Token-level semantic states for one step: h [B, Ns, sem_dim] (pretrained model hidden states, any dtype),
    mask [B, Ns] bool. `cond` [B, D] optional pooled belief+goal vector for the gates."""
    h: Tensor
    mask: Tensor
    cond: Optional[Tensor] = None


class SemanticProjection(nn.Module):
    def __init__(self, sem_dim: int, d: int):
        super().__init__()
        self.norm_in = RMSNorm(sem_dim)
        self.proj = nn.Linear(sem_dim, d, bias=False)
        self.norm_out = RMSNorm(d)
        nn.init.normal_(self.proj.weight, std=sem_dim ** -0.5)

    def forward(self, h: Tensor) -> Tensor:
        return self.norm_out(self.proj(self.norm_in(h.to(self.proj.weight.dtype))))


class GatedSemanticXAttn(nn.Module):
    def __init__(self, d: int, n_heads: int, gate_rank: int = 128, gate_bias: float = -3.0):
        super().__init__()
        self.norm_q = RMSNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, qk_norm=True, null_kv=True)
        self.gate_in = nn.Linear(3 * d, gate_rank)
        self.gate_out = nn.Linear(gate_rank, d)
        nn.init.normal_(self.gate_in.weight, std=(3 * d) ** -0.5)
        nn.init.normal_(self.gate_out.weight, std=1e-3)      # small, not zero: gate_in learns from step 0
        nn.init.constant_(self.gate_out.bias, gate_bias)

    def forward(self, h: Tensor, S: Tensor, S_mask: Tensor, cond: Optional[Tensor]) -> tuple:
        C = self.attn(self.norm_q(h), S, key_mask=S_mask)                        # [B, L, D]
        c = cond if cond is not None else h.new_zeros(h.shape[0], h.shape[-1])
        gx = torch.cat([h, C, c[:, None, :].expand_as(h).to(h.dtype)], -1)
        g = torch.sigmoid(self.gate_out(torch.nn.functional.silu(self.gate_in(gx))))
        return g * C, g


class GatedUnifiedCore(UnifiedCore):
    """UnifiedCore with CAM-controlled gated semantic cross-attention after every `sem_xattn_every`-th block."""

    def __init__(self, cfg: CamConfig, **kw):
        super().__init__(cfg, **kw)
        every = int(getattr(cfg, "sem_xattn_every", 0) or 0)
        sem_dim = int(getattr(cfg, "sem_xattn_dim", 0) or 0)
        self.xattn_layers = [i for i in range(cfg.n_layers) if every and (i + 1) % every == 0]
        self.sem_proj = SemanticProjection(sem_dim, cfg.d_model) if (every and sem_dim) else None
        self.xattn = nn.ModuleDict({str(i): GatedSemanticXAttn(cfg.d_model, cfg.n_heads,
                                                               gate_bias=getattr(cfg, "sem_gate_bias", -3.0))
                                    for i in self.xattn_layers}) if self.sem_proj is not None else nn.ModuleDict()
        self.cond_proj = nn.Linear(cfg.d_model, cfg.d_model) if self.sem_proj is not None else None
        self._sem: Optional[SemanticContext] = None
        self.last_gates: dict = {}

    def set_semantic(self, sem: Optional[SemanticContext]) -> None:
        """Semantic context for the next forward call(s); None disables the pathway (pure UnifiedCore)."""
        self._sem = sem

    def run_blocks(self, h: Tensor, key_mask: Optional[Tensor], blocks=None, attn_bias: Optional[Tensor] = None,
                   checkpoint_blocks: Optional[bool] = None) -> Tensor:
        sem = self._sem
        if sem is None or self.sem_proj is None or blocks is not None and blocks is not self.blocks:
            # reasoning re-uses sub-lists of blocks: semantic cross-attention there too when the block indices are
            # known (handled by `run_block_range`); plain path otherwise
            if sem is not None and self.sem_proj is not None and blocks is not None:
                return self._run_with_xattn(h, key_mask, list(blocks), attn_bias, checkpoint_blocks, sem)
            return super().run_blocks(h, key_mask, blocks, attn_bias, checkpoint_blocks)
        return self._run_with_xattn(h, key_mask, list(self.blocks), attn_bias, checkpoint_blocks, sem)

    def _run_with_xattn(self, h, key_mask, blocks, attn_bias, checkpoint_blocks, sem: SemanticContext) -> Tensor:
        want = self.grad_checkpoint if checkpoint_blocks is None else checkpoint_blocks
        ckpt = want and torch.is_grad_enabled() and self.training
        S = self.sem_proj(sem.h)
        if S.dtype != h.dtype:
            S = S.to(h.dtype)
        cond = self.cond_proj(sem.cond.to(self.cond_proj.weight.dtype)) if sem.cond is not None else None
        index = {id(b): i for i, b in enumerate(self.blocks)}
        gates = {}
        for blk in blocks:
            h = checkpoint(blk, h, key_mask, attn_bias, use_reentrant=False) if ckpt else blk(h, key_mask, attn_bias)
            li = index.get(id(blk))
            if li is not None and str(li) in self.xattn:
                mod = self.xattn[str(li)]
                if ckpt:
                    delta, g = checkpoint(mod, h, S, sem.mask, cond, use_reentrant=False)
                else:
                    delta, g = mod(h, S, sem.mask, cond)
                h = h + delta
                gates[li] = float(g.detach().float().mean())
        self.last_gates = gates
        return h

    def count_parameters(self) -> dict:
        out = super().count_parameters()
        from .blocks import count_parameters as cp
        out["semantic_projection"] = cp(self.sem_proj) if self.sem_proj is not None else 0
        out["semantic_xattn"] = cp(self.xattn)
        out["semantic_cond"] = cp(self.cond_proj) if self.cond_proj is not None else 0
        return out
