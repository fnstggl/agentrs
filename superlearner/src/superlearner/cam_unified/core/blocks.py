"""Transformer building blocks of the CAM Unified backbone (workstream B).

Shapes: B batch, L sequence length (state + context tokens), D d_model, H heads, Dh = D / H.

* `RMSNorm` — root-mean-square normalisation computed in float32 (bf16-safe), learned gain.
* `SwiGLU` — gated feed-forward  W2 (silu(x W1) * (x W3)), hidden `cfg.d_ff`.
* `MultiHeadAttention` — self- or cross-attention through `torch.nn.functional.scaled_dot_product_attention`
  (flash kernel when nothing is masked, memory-efficient kernel with a boolean key-padding mask otherwise), optional
  per-head RMS normalisation of queries and keys (stabilises bf16 training), optional learned null key/value so a
  query whose keys are all padding still has a well-defined output.
* `Block` — pre-norm residual block  x + Attn(norm(x)); x + FFN(norm(x)).
* `TimeEncoding` — continuous-time encoding of `TokenSet.time` (token time minus now, simulated days): sinusoids of the
  signed log-scaled time  u = sign(t) log1p(|t| / tau)  (tau = 1 minute) at geometric frequencies, plus u itself.
* `KindEmbedding` — `TokenKind` embeddings.
* `EntitySlotCodes` — binding codes for episode-local entity slots: a fixed random (non-learned) table of unit vectors,
  re-indexed by a random per-episode permutation per batch row, through a learned projection. A slot code therefore
  carries no meaning across episodes; tokens about the same entity share a code within an episode (binding only).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import TokenKind


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g. Computed in float32, returned in the input dtype. x [..., D]."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        # fused kernel, float32 accumulation (x.float() is a no-op for the float32 residual stream)
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: x [..., D] -> W2(silu(W1 x) * W3 x) [..., D] with hidden width d_ff."""

    def __init__(self, d: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w13 = nn.Linear(d, 2 * d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w2._residual_out = True
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w13(x).chunk(2, -1)
        return self.drop(self.w2(F.silu(a) * b))


def expand_key_mask(key_mask: Optional[Tensor], n_q: int) -> Optional[Tensor]:
    """Boolean key-padding mask [B, Lk] (True = real) -> SDPA mask [B, 1, 1, Lk] (broadcast over heads/queries).
    None stays None (no masking: flash kernel eligible)."""
    if key_mask is None:
        return None
    return key_mask[:, None, None, :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention via SDPA.

    forward(x [B,Lq,D], kv [B,Lk,D] | None (= self-attention), key_mask [B,Lk] bool | None (True = attend),
            attn_bias [B,1|H,Lq,Lk] float | None (additive)) -> [B,Lq,D].
    With `null_kv=True` a learned key/value pair is prepended to every key set and always attended to, so rows whose
    keys are all padding produce a learned default instead of NaN."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0, qk_norm: bool = True, null_kv: bool = False,
                 kv_dim: Optional[int] = None):
        super().__init__()
        assert d % n_heads == 0, (d, n_heads)
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        kv_dim = kv_dim or d
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(kv_dim, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.o._residual_out = True
        self.dropout = dropout
        self.q_norm = RMSNorm(self.dh) if qk_norm else None
        self.k_norm = RMSNorm(self.dh) if qk_norm else None
        self.null_kv = nn.Parameter(torch.randn(2, n_heads, 1, self.dh) * 0.02) if null_kv else None

    def _heads(self, t: Tensor) -> Tensor:
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.dh).transpose(1, 2)        # [B,H,L,Dh]

    def forward(self, x: Tensor, kv: Optional[Tensor] = None, key_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None) -> Tensor:
        B, Lq, _ = x.shape
        src = x if kv is None else kv
        q = self._heads(self.q(x))
        k, v = self.kv(src).chunk(2, -1)
        k, v = self._heads(k), self._heads(v)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.null_kv is not None:
            nk = self.null_kv[0].unsqueeze(0).expand(B, -1, -1, -1).to(k.dtype)
            nv = self.null_kv[1].unsqueeze(0).expand(B, -1, -1, -1).to(v.dtype)
            k, v = torch.cat([nk, k], 2), torch.cat([nv, v], 2)
            if key_mask is not None:
                key_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=key_mask.device), key_mask], 1)
            if attn_bias is not None:
                attn_bias = F.pad(attn_bias, (1, 0))
        mask = None
        if attn_bias is not None:
            mask = attn_bias.to(q.dtype)
            if key_mask is not None:
                mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        elif key_mask is not None:
            mask = expand_key_mask(key_mask, Lq)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Lq, self.d))


class Block(nn.Module):
    """Pre-norm residual transformer block. forward(h [B,L,D], key_mask [B,L] | None, attn_bias | None) -> [B,L,D]."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float = 0.0, qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff, dropout)

    def forward(self, h: Tensor, key_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None) -> Tensor:
        h = h + self.attn(self.norm1(h), key_mask=key_mask, attn_bias=attn_bias)
        return h + self.ffn(self.norm2(h))


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + FFN block: queries x [B,Lq,D] read from kv [B,Lk,D] (masked, null key always
    available). Used for action-conditioning and attention pooling."""

    def __init__(self, d: int, n_heads: int, d_ff: int, kv_dim: Optional[int] = None):
        super().__init__()
        self.norm_q = RMSNorm(d)
        self.norm_kv = RMSNorm(kv_dim or d)
        self.attn = MultiHeadAttention(d, n_heads, qk_norm=True, null_kv=True, kv_dim=kv_dim)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x: Tensor, kv: Tensor, key_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm_q(x), self.norm_kv(kv), key_mask=key_mask, attn_bias=attn_bias)
        return x + self.ffn(self.norm2(x))


def signed_log_time(t: Tensor, tau_days: float = 1.0 / 1440.0) -> Tensor:
    """u = sign(t) * log1p(|t| / tau): resolves minutes near `now` and years far away. t [...] days."""
    return torch.sign(t) * torch.log1p(t.abs() / tau_days)


class TimeEncoding(nn.Module):
    """Continuous-time encoding: t [B,N] (days relative to now) -> [B,N,D].
    Features: sin/cos(omega_k * u) for geometric omega_k in [w_min, w_max], plus u / 10, with u = signed_log_time(t)."""

    def __init__(self, d: int, n_freq: int = 16, w_min: float = 0.05, w_max: float = 20.0,
                 tau_days: float = 1.0 / 1440.0):
        super().__init__()
        self.tau = tau_days
        self.register_buffer("omega", torch.logspace(math.log10(w_min), math.log10(w_max), n_freq),
                             persistent=False)
        self.proj = nn.Linear(2 * n_freq + 1, d)

    def features(self, t: Tensor) -> Tensor:
        u = signed_log_time(t.float(), self.tau)
        a = u.unsqueeze(-1) * self.omega
        return torch.cat([torch.sin(a), torch.cos(a), (u / 10.0).unsqueeze(-1)], -1)

    def forward(self, t: Tensor) -> Tensor:
        return self.proj(self.features(t))


class KindEmbedding(nn.Embedding):
    """TokenKind -> [D]. kind [B,N] long -> [B,N,D]."""

    def __init__(self, d: int):
        super().__init__(int(TokenKind.N_KINDS), d)
        nn.init.normal_(self.weight, std=0.02)


class EntitySlotCodes(nn.Module):
    """Binding codes for episode-local entity slots.

    table [E, Dc]: fixed random unit vectors (buffer; saved with the model so a slot index maps to the same raw code
    in every process). Per-episode randomisation: `new_episode(B)` draws a random permutation of the slots per batch
    row (perm [B, E]); the code of slot s in row b is table[perm[b, s]]. Until `new_episode` is called the permutation
    is the identity — then per-episode randomness comes from `EntityRegistry`'s random slot binding (seeded per
    episode). Rows can be remapped with `code_rows` [B'] (e.g. imagination branches -> their real environment's row).

    forward(slot [B,N] long, -1 = none, code_rows) -> [B,N,D] (zero where slot < 0), through a learned projection
    with a learned gain. Two projections exist (`which` = 0 primary entity, 1 second entity of a relation) so the
    direction of a relation is visible."""

    def __init__(self, d: int, max_entities: int, d_code: Optional[int] = None, seed: int = 1234):
        super().__init__()
        self.E = max_entities
        dc = d_code or d
        g = torch.Generator().manual_seed(seed)
        table = torch.randn(max_entities, dc, generator=g)
        self.register_buffer("table", table / table.norm(dim=-1, keepdim=True))
        self.proj = nn.ModuleList([nn.Linear(dc, d, bias=False) for _ in range(2)])
        self.gain = nn.Parameter(torch.ones(2))
        self.perm: Optional[Tensor] = None
        self.episodes_started = 0

    @torch.no_grad()
    def new_episode(self, B: int, rows: Optional[Tensor] = None, generator: Optional[torch.Generator] = None,
                    device=None) -> Tensor:
        """Draw fresh per-episode permutations for all B rows (rows=None) or only for `rows` (environments that just
        reset). Returns perm [B, E]."""
        dev = device or self.table.device
        if self.perm is None or self.perm.shape[0] != B:
            self.perm = torch.arange(self.E, device=dev).repeat(B, 1)
            rows = None                                          # a new batch layout: every row is a new episode
        rows = list(range(B)) if rows is None else [int(r) for r in (rows.tolist() if isinstance(rows, Tensor) else rows)]
        for r in rows:
            self.perm[r] = torch.randperm(self.E, generator=generator).to(self.perm.device)
        self.episodes_started += len(rows)
        return self.perm

    def forward(self, slot: Tensor, which: int = 0, code_rows: Optional[Tensor] = None) -> Tensor:
        B, N = slot.shape
        valid = (slot >= 0) & (slot < self.E)
        s = slot.clamp(0, self.E - 1)
        if self.perm is not None:
            if self.perm.device != slot.device:
                self.perm = self.perm.to(slot.device)
            perm = self.perm if code_rows is None else self.perm[code_rows.to(slot.device)]
            if perm.shape[0] != B:
                raise ValueError(f"EntitySlotCodes: {perm.shape[0]} per-episode permutations for a batch of {B}; "
                                 "pass code_rows (batch row -> episode row) or call new_episode(B)")
            s = torch.gather(perm, 1, s)
        codes = self.table[s]                                    # [B,N,Dc]
        out = self.proj[which](codes) * self.gain[which]
        return out * valid.unsqueeze(-1).to(out.dtype)


def init_transformer_weights(module: nn.Module, n_layers: int, std: float = 0.02) -> None:
    """GPT-style init: Linear ~ N(0, std) (residual output projections N(0, std / sqrt(2 n_layers))), zero biases."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            s = std / math.sqrt(2 * max(1, n_layers)) if getattr(m, "_residual_out", False) else std
            nn.init.normal_(m.weight, std=s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters, shared tensors counted once."""
    seen, n = set(), 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        n += p.numel()
    return n
