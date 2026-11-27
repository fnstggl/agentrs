"""CAM-controlled gated semantic cross-attention inside the core (DOM amendment Mechanism C; manifest DOM-05).

At selected CAM layers l (after every `cfg.sem_xattn_every`-th block of the shared backbone):

    R_l     = F_theta,l(Z_l)                                   (the ordinary pre-norm block: self-attention + SwiGLU)
    C_l     = CrossAttention_theta,l(Z_l, S_t, S_t)            (queries: CAM's tokens; keys/values: semantic tokens)
    g_l     = sigmoid(Gate_theta,l(Z_l, C_l, cond_t))          (per token and per feature; cond_t = pooled B_t + G_t)
    Z_{l+1} = Z_l + R_l + g_l * C_l

S_t are the token-level hidden states of the pretrained semantic model for the current step's semantic content,
projected once per call to CAM's width by a trainable `SemanticProjection` (RMS-normalised, so the semantic pathway
cannot dominate by scale). The gate bias starts negative (`cfg.sem_gate_bias`, g ~ 0.05): CAM begins mostly
self-driven and learns how much semantic information each token needs. There is no fixed activation ratio.

The semantic context is an **explicit argument** (repair of audit 2026-11-26, h-core / Mechanism C):

    UnifiedCore.forward(Z_prev, ctx, code_rows=None, sem=None, *, sem_rows=None, blocks=None)
    UnifiedCore.run_blocks(h, key_mask, blocks=None, attn_bias=None, checkpoint_blocks=None, sem=None, block_ids=None)
    Reasoner.forward(Z, ctx, n_iters=None, code_rows=None, sem=None, *, sem_rows=None)

There is no module state: the same call with the same arguments always gives the same result, a deep copy of the
core (an EMA target) given the same `sem` computes the same function, activation / iteration checkpointing recomputes
with exactly the tensors of the original forward (they are checkpoint inputs, not attributes read at recomputation),
and batches of another size (imagination / evaluator rows) select their semantic rows explicitly: `sem` either has the
batch's rows, or it has the source-environment rows and `sem_rows` (default: `code_rows`) maps each batch row to one.

The cross-attention modules live in `UnifiedCore` itself (built when `cfg.sem_xattn_every > 0` and
`cfg.sem_xattn_dim > 0`; the dom presets switch them on), so the registry's core class carries Mechanism C.
`GatedUnifiedCore` is kept as a name for the same class. With no semantic input (or `sem_xattn_every == 0`) the core
is exactly the plain backbone: the cross-attention is skipped, not zero-filled. Gradients flow into the projection,
the cross-attention, the gates and, through the attended states, to every block; a frozen semantic model is a
constant input (its parameters receive nothing). Gate statistics of the last call are kept as detached 0-d tensors in
`core.last_gates` (no host synchronisation; `core.gate_stats()` converts them).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .blocks import MultiHeadAttention, RMSNorm, init_transformer_weights


def _rows_tensor(rows, device) -> Tensor:
    if isinstance(rows, Tensor):
        return rows.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(rows), dtype=torch.long, device=device)


@dataclass
class SemanticContext:
    """Token-level semantic states for one step (the producer is the tokenizer's `semantic_context`, W4).

    h [B, Ns, sem_dim]: pretrained-model hidden states (any dtype, no gradient needed); mask [B, Ns] bool (True =
    real token); cond [B, D] optional pooled belief + goal vector for the gates; refs optional [B][Ns] source ids of
    the tokens (observation refs) for traces."""
    h: Tensor
    mask: Tensor
    cond: Optional[Tensor] = None
    refs: Optional[list] = None

    @property
    def B(self) -> int:
        return int(self.h.shape[0])

    @property
    def N(self) -> int:
        return int(self.h.shape[1])

    def select(self, rows: Sequence[int] | Tensor) -> "SemanticContext":
        """Rows `rows` (with repetition: imagination branches -> their environment's semantic context)."""
        idx = _rows_tensor(rows, self.h.device)
        refs = None if self.refs is None else [self.refs[int(r)] for r in idx.tolist()]
        return SemanticContext(self.h[idx], self.mask[idx.to(self.mask.device)],
                               None if self.cond is None else self.cond[idx.to(self.cond.device)], refs)

    def to(self, device) -> "SemanticContext":
        return SemanticContext(self.h.to(device), self.mask.to(device),
                               None if self.cond is None else self.cond.to(device), self.refs)


@dataclass(frozen=True)
class SemanticKV:
    """Projected semantic memory, ready for the cross-attention layers: S [B, Ns, D] (CAM width), mask [B, Ns] bool,
    cond [B, D] | None (projected gate conditioning). Produced by `UnifiedCore.project_semantic`; immutable, so a
    checkpointed region that receives it recomputes with the same tensors."""
    S: Tensor
    mask: Tensor
    cond: Optional[Tensor] = None

    @property
    def B(self) -> int:
        return int(self.S.shape[0])

    def select(self, rows: Sequence[int] | Tensor) -> "SemanticKV":
        idx = _rows_tensor(rows, self.S.device)
        return SemanticKV(self.S[idx], self.mask[idx.to(self.mask.device)],
                          None if self.cond is None else self.cond[idx.to(self.cond.device)])


class SemanticProjection(nn.Module):
    """h [..., sem_dim] (any dtype) -> RMSNorm(W RMSNorm(h)) [..., D]."""

    def __init__(self, sem_dim: int, d: int):
        super().__init__()
        self.norm_in = RMSNorm(sem_dim)
        self.proj = nn.Linear(sem_dim, d, bias=False)
        self.norm_out = RMSNorm(d)
        nn.init.normal_(self.proj.weight, std=sem_dim ** -0.5)

    def forward(self, h: Tensor) -> Tensor:
        return self.norm_out(self.proj(self.norm_in(h.to(self.proj.weight.dtype))))


class GatedSemanticXAttn(nn.Module):
    """One gated cross-attention layer: forward(h [B,L,D], S [B,Ns,D], S_mask [B,Ns], cond [B,D] | None) ->
    (g * C [B,L,D], g [B,L,D]). The attention uses the backbone's GPT-style initialisation (residual output projection
    scaled by 1/sqrt(2 n_layers)); a learned null key keeps rows without semantic tokens finite."""

    def __init__(self, d: int, n_heads: int, gate_rank: int = 128, gate_bias: float = -3.0, n_layers: int = 1):
        super().__init__()
        self.norm_q = RMSNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, qk_norm=True, null_kv=True)
        init_transformer_weights(self.attn, n_layers)
        self.gate_in = nn.Linear(3 * d, gate_rank)
        self.gate_out = nn.Linear(gate_rank, d)
        nn.init.normal_(self.gate_in.weight, std=(3 * d) ** -0.5)
        nn.init.zeros_(self.gate_in.bias)
        nn.init.normal_(self.gate_out.weight, std=1e-3)      # small, not zero: gate_in learns from step 0
        nn.init.constant_(self.gate_out.bias, gate_bias)

    def forward(self, h: Tensor, S: Tensor, S_mask: Tensor, cond: Optional[Tensor]) -> tuple:
        C = self.attn(self.norm_q(h), S, key_mask=S_mask)                        # [B, L, D]
        c = cond if cond is not None else h.new_zeros(h.shape[0], h.shape[-1])
        gx = torch.cat([h, C.to(h.dtype), c[:, None, :].expand_as(h).to(h.dtype)], -1)
        g = torch.sigmoid(self.gate_out(torch.nn.functional.silu(self.gate_in(gx))))
        return g * C, g


def build_semantic_modules(cfg, d: int, n_heads: int, n_layers: int) -> tuple:
    """(layers [int], SemanticProjection | None, ModuleDict, cond Linear | None) for a backbone of `n_layers` blocks:
    cross-attention after every `cfg.sem_xattn_every`-th block when `cfg.sem_xattn_dim` is known, else nothing."""
    every = int(getattr(cfg, "sem_xattn_every", 0) or 0)
    sem_dim = int(getattr(cfg, "sem_xattn_dim", 0) or 0)
    if not (every > 0 and sem_dim > 0):
        return [], None, nn.ModuleDict(), None
    layers = [i for i in range(n_layers) if (i + 1) % every == 0]
    proj = SemanticProjection(sem_dim, d)
    gate_bias = float(getattr(cfg, "sem_gate_bias", -3.0))
    xattn = nn.ModuleDict({str(i): GatedSemanticXAttn(d, n_heads, gate_bias=gate_bias, n_layers=n_layers)
                           for i in layers})
    cond = nn.Linear(d, d)
    return layers, proj, xattn, cond


def __getattr__(name):
    # GatedUnifiedCore is the backbone class itself (Mechanism C is built from the config); resolved lazily because
    # the backbone imports this module's layers.
    if name == "GatedUnifiedCore":
        from .backbone import GatedUnifiedCore
        return GatedUnifiedCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
