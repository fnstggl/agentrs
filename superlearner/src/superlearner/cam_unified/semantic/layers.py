"""Small neural building blocks shared by the semantic components (tokenizer, bridge, action generator)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def swiglu_hidden(d: int, mult: float = 8 / 3) -> int:
    return max(64, int(round(mult * d / 64)) * 64)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or swiglu_hidden(d)
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class CrossAttention(nn.Module):
    """Multi-head attention of queries q [B, Q, D] onto keys/values kv [B, N, Dk] with key mask [B, N] (True = real).
    Rows whose keys are all masked get a zero output (no NaN)."""

    def __init__(self, d: int, n_heads: int, d_kv: Optional[int] = None):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        dk = d_kv or d
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(dk, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Q, D = q.shape
        N = kv.shape[1]
        qh = self.q(q).view(B, Q, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(kv).view(B, N, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if mask is not None:
            any_key = mask.any(1)
            m = mask[:, None, None, :].expand(B, 1, Q, N)
            m = m | ~any_key[:, None, None, None]                   # avoid all-masked rows (output zeroed below)
            out = F.scaled_dot_product_attention(qh, k, v, attn_mask=m)
            out = out * any_key[:, None, None, None].to(out.dtype)
        else:
            out = F.scaled_dot_product_attention(qh, k, v)
        return self.o(out.transpose(1, 2).reshape(B, Q, D))


class ContextReader(nn.Module):
    """Pre-norm cross-attention + SwiGLU blocks: decision queries read the decision context [R; G; B]."""

    def __init__(self, d: int, n_heads: int, n_layers: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({"n1": RMSNorm(d), "nk": RMSNorm(d), "att": CrossAttention(d, n_heads),
                                              "n2": RMSNorm(d), "ff": SwiGLU(d)}))
        self.out_norm = RMSNorm(d)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        for b in self.blocks:
            q = q + b["att"](b["n1"](q), b["nk"](ctx), mask)
            q = q + b["ff"](b["n2"](q))
        return self.out_norm(q)


class SegmentAttentionPool(nn.Module):
    """Attention pooling of many variable-length token sequences without padding.

    tokens [L, Din] (all segments concatenated), seg [L] long (segment id of each token, 0..S-1), queries [S, Q, D]
    (per-segment query sets, usually one learned set broadcast) -> pooled [S, Q, D]. A numerically stable segment
    softmax (per-segment max subtraction) is computed with scatter_reduce/index_add, so a 12k-token observation and a
    5-token one share a batch at no padding cost.
    """

    def __init__(self, d_in: int, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.norm = RMSNorm(d_in)
        self.kv = nn.Linear(d_in, 2 * d, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, tokens: torch.Tensor, seg: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        S, Q, D = queries.shape
        L = tokens.shape[0]
        k, v = self.kv(self.norm(tokens)).view(L, 2, self.h, self.dh).unbind(1)       # [L, h, dh]
        q = self.q(queries).view(S, Q, self.h, self.dh)                                # [S, Q, h, dh]
        qs = q[seg]                                                                    # [L, Q, h, dh]
        scores = (qs.float() * k.float().unsqueeze(1)).sum(-1) / math.sqrt(self.dh)    # [L, Q, h]
        idx = seg.view(L, 1, 1).expand_as(scores)
        mx = torch.full((S, Q, self.h), -torch.inf, device=scores.device, dtype=scores.dtype)
        mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
        w = torch.exp(scores - mx[seg].detach())
        den = torch.zeros(S, Q, self.h, device=w.device, dtype=w.dtype).index_add(0, seg, w)
        w = w / den[seg].clamp_min(1e-30)
        out = torch.zeros(S, Q, self.h, self.dh, device=w.device, dtype=torch.float32)
        out = out.index_add(0, seg, w.unsqueeze(-1) * v.float().unsqueeze(1))           # [S, Q, h, dh]
        return self.o(out.reshape(S, Q, D).to(queries.dtype))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class NumericFeatures(nn.Module):
    """Scale-free features of raw numbers v [...] -> [..., 2*n_fourier + 7]: symlog value, sign, zero/integer flags,
    Fourier features sin/cos(w_k * symlog(v)) at geometrically spaced frequencies w_k in [0.05, 50], and nan / +inf /
    -inf flags. A non-finite value sets its flag and contributes 0 to every other feature (never NaN)."""

    def __init__(self, n_fourier: int):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(math.log(0.05), math.log(50.0), n_fourier)),
                             persistent=False)
        self.dim = 2 * n_fourier + 7

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.double()
        finite = torch.isfinite(v)
        flags = torch.stack([torch.isnan(v), torch.isposinf(v), torch.isneginf(v)], -1).double()
        v = torch.where(finite, v, torch.zeros_like(v))
        u = symlog(v)
        ang = u.unsqueeze(-1) * self.freq.double()
        is_int = (torch.abs(v - torch.round(v)) < 1e-9).double()
        feats = torch.cat([torch.sin(ang), torch.cos(ang), (u / 10.0).unsqueeze(-1), torch.sign(v).unsqueeze(-1),
                           (v == 0).double().unsqueeze(-1), is_int.unsqueeze(-1)], -1)
        feats = feats * finite.double().unsqueeze(-1)
        return torch.cat([feats, flags], -1).float()


class ValueFeatures(nn.Module):
    """Sign-separated features of raw numbers for the NUMERIC VALUE channel: v [...] -> [..., 2*(2K+1) + 5].

    m(v) = [sin(w_k u), cos(w_k u), u/10] with u = log1p(|v|) at K geometrically spaced frequencies w_k in [lo, hi],
    then [m, sign(v) * m, zero flag, integer flag, nan, +inf, -inf]. The sign multiplies a full copy of the magnitude
    block, so v and -v have opposite halves (the old NumericFeatures had a single sign entry among 39: cos(+1, -1) of
    the final token 0.985 at init). High frequencies (hi = 200) resolve nearby magnitudes, low ones keep the encoding
    smooth across scales. Non-finite values set their flag and give 0 elsewhere (never NaN)."""

    def __init__(self, n_freq: int = 24, lo: float = 0.02, hi: float = 200.0):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(math.log(lo), math.log(hi), n_freq)), persistent=False)
        self.dim = 2 * (2 * n_freq + 1) + 5

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.double()
        finite = torch.isfinite(v)
        flags = torch.stack([torch.isnan(v), torch.isposinf(v), torch.isneginf(v)], -1).double()
        v = torch.where(finite, v, torch.zeros_like(v))
        u = torch.log1p(torch.abs(v))
        ang = u.unsqueeze(-1) * self.freq.double()
        m = torch.cat([torch.sin(ang), torch.cos(ang), (u / 10.0).unsqueeze(-1)], -1)
        is_int = (torch.abs(v - torch.round(v)) < 1e-9).double().unsqueeze(-1)
        feats = torch.cat([m, torch.sign(v).unsqueeze(-1) * m, (v == 0).double().unsqueeze(-1), is_int], -1)
        feats = feats * finite.double().unsqueeze(-1)
        return torch.cat([feats, flags], -1).float()


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, temperature=1.0) -> torch.Tensor:
    """log_softmax over allowed entries (mask True); disallowed -> -inf. Rows must have >= 1 allowed entry.
    `temperature` is a float or a tensor broadcastable to logits (per-row temperatures)."""
    if isinstance(temperature, torch.Tensor):
        z = logits.float() / temperature.float().clamp_min(1e-6)
    else:
        z = logits.float() / max(float(temperature), 1e-6)
    z = z.masked_fill(~mask, -torch.inf)
    return torch.log_softmax(z, -1)


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return -(p * logp.masked_fill(~mask, 0.0)).sum(-1)
