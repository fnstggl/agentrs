"""Small transformer building blocks used by the memory encoders (pre-norm RMSNorm -> SDPA attention -> SwiGLU).

Same block design as the shared backbone (architecture §1), kept local so the memory package has no import-time
dependency on workstream B's modules.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden(d: int, mult: float = 8 / 3) -> int:
    return max(64, int(round(mult * d / 64)) * 64)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or swiglu_hidden(d)
        self.w1 = nn.Linear(d, h, bias=False)     # gate
        self.w3 = nn.Linear(d, h, bias=False)     # up
        self.w2 = nn.Linear(h, d, bias=False)     # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention over SDPA. ``mask`` [B, Lk] bool, True = attend."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, (d, n_heads)
        self.h = n_heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, Lq, D = x.shape
        Lk = ctx.shape[1]
        hd = D // self.h
        q = self.q(x).view(B, Lq, self.h, hd).transpose(1, 2)
        k, v = self.kv(ctx).view(B, Lk, 2, self.h, hd).permute(2, 0, 3, 1, 4)
        am = None if mask is None else mask[:, None, None, :].expand(B, 1, Lq, Lk)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(y.transpose(1, 2).reshape(B, Lq, D))


class SelfBlock(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.att = Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d)

    def forward(self, x, mask=None):
        x = x + self.att(self.n1(x), mask=mask)
        return x + self.ffn(self.n2(x))


class CrossBlock(nn.Module):
    """Queries ``x`` [B,Lq,D] attend to ``ctx`` [B,Lk,D] (pre-norm), then a SwiGLU."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.nq = RMSNorm(d)
        self.nc = RMSNorm(d)
        self.att = Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d)

    def forward(self, x, ctx, mask=None):
        x = x + self.att(self.nq(x), self.nc(ctx), mask=mask)
        return x + self.ffn(self.n2(x))


class TimeFeatures(nn.Module):
    """Continuous elapsed-time encoding: dt (days, any sign) -> sign*log1p|dt| -> Fourier features -> Linear(d)."""

    def __init__(self, d: int, n_fourier: int = 8):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(math.log(0.25), math.log(32.0), n_fourier)),
                             persistent=False)
        self.lin = nn.Linear(2 * n_fourier + 1, d)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        s = torch.sign(dt) * torch.log1p(dt.abs())
        a = s.unsqueeze(-1).float() * self.freq
        return self.lin(torch.cat([s.unsqueeze(-1).float(), torch.sin(a), torch.cos(a)], -1))


def signed_log_fourier(v: torch.Tensor, n: int) -> torch.Tensor:
    """[...] -> [..., 2n+1] features of sign*log1p|v| (scale-free numeric encoding)."""
    s = torch.sign(v) * torch.log1p(v.abs())
    freq = torch.exp(torch.linspace(math.log(0.25), math.log(16.0), n, device=v.device))
    a = s.unsqueeze(-1) * freq
    return torch.cat([s.unsqueeze(-1), torch.sin(a), torch.cos(a)], -1)


def autocast_for(t: torch.Tensor):
    """bf16 autocast on CUDA tensors, disabled on CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.is_cuda)
