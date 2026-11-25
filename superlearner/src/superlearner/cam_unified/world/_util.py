"""Shared numerical helpers for the world-model package (workstream C).

Nothing here is a model component on its own; these are the small, tested building blocks the components use:
symlog/two-hot regression, continuous-time features, attention pooling, padding/stacking of variable-length
``TokenSet`` sequences, masked reductions and generator-controlled sampling.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import OutcomeBatch, TokenKind, TokenSet

SECONDS_PER_DAY = 86400.0


# ----------------------------------------------------------------------------------------------- scalar transforms

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x).clamp(max=60.0))


class TwoHot(nn.Module):
    """Two-hot categorical regression in symlog space (DreamerV3). ``logits [..., n_bins]``."""

    def __init__(self, n_bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.register_buffer("bins", torch.linspace(low, high, n_bins), persistent=False)

    @property
    def n(self) -> int:
        return self.bins.numel()

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """y [...] (raw scale) -> two-hot weights [..., n_bins]."""
        bins = self.bins.to(y.device)
        v = symlog(y.float()).clamp(bins[0], bins[-1])
        above = torch.searchsorted(bins, v.contiguous(), right=False).clamp(1, bins.numel() - 1)
        below = above - 1
        w_above = (v - bins[below]) / (bins[above] - bins[below])
        out = torch.zeros(*v.shape, bins.numel(), device=y.device, dtype=torch.float32)
        out.scatter_(-1, below.unsqueeze(-1), (1 - w_above).unsqueeze(-1))
        out.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
        return out

    def log_lik(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Two-hot log-likelihood (negative cross-entropy) [...]."""
        return (self.target(y) * F.log_softmax(logits.float(), -1)).sum(-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value in the raw scale (symexp of the expected symlog value)."""
        p = F.softmax(logits.float(), -1)
        return symexp((p * self.bins.to(p.device)).sum(-1))


# ----------------------------------------------------------------------------------------------- time features

def time_features(dt: torch.Tensor, n_fourier: int) -> torch.Tensor:
    """Continuous elapsed-time features for irregular intervals.

    dt [...] in simulated days (>= 0; negative values are treated by magnitude with a sign channel) ->
    [..., 2 * n_fourier + 3]: Fourier features of log-seconds (scale-free: 1 s .. years), the log itself, a
    saturating linear term and the sign.
    """
    dt = dt.float()
    sign = torch.sign(dt)
    sec = dt.abs() * SECONDS_PER_DAY
    ls = torch.log1p(sec) / math.log1p(SECONDS_PER_DAY * 365.0)          # ~[0, 1] for 0 .. 1 year
    freqs = torch.exp(torch.linspace(0.0, math.log(64.0), n_fourier, device=dt.device)) * math.pi
    ang = ls.unsqueeze(-1) * freqs
    lin = torch.tanh(dt.abs() / 30.0)
    return torch.cat([torch.sin(ang), torch.cos(ang), ls.unsqueeze(-1), lin.unsqueeze(-1), sign.unsqueeze(-1)], -1)


class TimeEncoder(nn.Module):
    """EncTime(dt): [...] days -> [..., D]."""

    def __init__(self, d_model: int, n_fourier: int = 16):
        super().__init__()
        self.n_fourier = n_fourier
        self.net = nn.Sequential(nn.Linear(2 * n_fourier + 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return self.net(time_features(dt, self.n_fourier))


# ----------------------------------------------------------------------------------------------- modules

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, zero_out: bool = False):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_in)
        if zero_out:
            nn.init.zeros_(self.l2.weight)
            nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        return self.l2(F.silu(self.l1(self.norm(x))))


class AttnPool(nn.Module):
    """Learned-query attention pooling over a masked token set: x [B,N,D], mask [B,N] -> [B,n_q,D]."""

    def __init__(self, d_model: int, n_heads: int, n_queries: int = 1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.att = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, query: Optional[torch.Tensor] = None):
        B = x.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1) if query is None else query
        kpm = None if mask is None else safe_key_padding(mask)
        out, _ = self.att(q, self.norm(x), self.norm(x), key_padding_mask=kpm, need_weights=False)
        return out


class StateNorm(nn.Module):
    """Running per-position standardisation of latent states [..., S, D] (FeatureNorm; never same-batch stats).

    Latent states of a residual backbone are dominated by a component shared by all samples; prediction heads
    read (Z - mean) / std with running statistics so sample-specific content is visible at unit scale. Inputs whose
    trailing shape is not (S, D) pass through unchanged. ``update`` is called by the owner with detached states."""

    def __init__(self, n_state: int, d_model: int, momentum: float = 0.02):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(n_state, d_model))
        self.register_buffer("var", torch.ones(n_state, d_model))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, Z: torch.Tensor) -> None:
        """Z [N,S,D] (detached valid states)."""
        if Z.shape[0] < 2 or Z.shape[-2:] != self.mean.shape:
            return
        Z = Z.float()
        m, v = Z.mean(0), Z.var(0)
        mom = 1.0 if int(self.count) == 0 else self.momentum
        self.mean.lerp_(m, mom)
        self.var.lerp_(v, mom)
        self.count += 1

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.shape[-2:] != self.mean.shape or int(self.count) == 0:
            return Z
        floor = 0.01 * self.var.mean() + 1e-6
        return (Z.float() - self.mean) * torch.rsqrt(self.var + floor)


def safe_key_padding(mask: torch.Tensor) -> torch.Tensor:
    """key_padding_mask (True = ignore) from a real-token mask, never fully masking a row (avoids NaN)."""
    kpm = ~mask.bool()
    full = kpm.all(-1)
    if full.any():
        kpm = kpm.clone()
        kpm[full, 0] = False
    return kpm


# ----------------------------------------------------------------------------------------------- reductions

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dims=None) -> torch.Tensor:
    if mask is None:
        return x.mean() if dims is None else x.mean(dims)
    m = mask.to(x.dtype)
    x = torch.where(mask.bool(), x, torch.zeros_like(x))
    if dims is None:
        return x.sum() / m.sum().clamp(min=1.0)
    return x.sum(dims) / m.sum(dims).clamp(min=1.0)


def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim) - math.log(x.shape[dim])


# ----------------------------------------------------------------------------------------------- token helpers

def make_tokens(x: torch.Tensor, kind: int, time: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> TokenSet:
    """Dense [B,N,D] -> TokenSet of one kind (entity -1, refs None)."""
    return TokenSet.from_dense(x, int(kind), time=time, mask=mask)


def tokens_nonempty(ts: Optional[TokenSet]) -> bool:
    return ts is not None and ts.N > 0


def cat_tokens(*sets: Optional[TokenSet]) -> TokenSet:
    sets = [s for s in sets if s is not None and s.N > 0]
    if not sets:
        raise ValueError("cat_tokens needs at least one non-empty TokenSet")
    if len(sets) == 1:
        return sets[0]
    return TokenSet.cat(*sets)


def pad_tokens(ts: TokenSet, N: int) -> TokenSet:
    """Right-pad a TokenSet to N tokens (PAD kind, mask False)."""
    if ts.N == N:
        return ts
    if ts.N > N:
        raise ValueError("cannot pad to fewer tokens")
    B, n, D = ts.x.shape
    p = N - n
    dev = ts.x.device
    refs = None if ts.refs is None else [list(r) + [None] * p for r in ts.refs]
    e2 = None if ts.entity2 is None else torch.cat([ts.entity2, torch.full((B, p), -1, dtype=torch.long, device=dev)], 1)
    return TokenSet(torch.cat([ts.x, ts.x.new_zeros(B, p, D)], 1),
                    torch.cat([ts.mask, torch.zeros(B, p, dtype=torch.bool, device=dev)], 1),
                    torch.cat([ts.kind, torch.full((B, p), int(TokenKind.PAD), dtype=torch.long, device=dev)], 1),
                    torch.cat([ts.time, ts.time.new_zeros(B, p)], 1),
                    torch.cat([ts.entity, torch.full((B, p), -1, dtype=torch.long, device=dev)], 1), refs, e2)


def stack_tokens(sets: Sequence[TokenSet]) -> TokenSet:
    """list of T TokenSets [B,N_t,D] -> TokenSet with leading time axis: x [T,B,N,D] (padded to max N).
    refs are dropped (not needed for tensor computations)."""
    N = max(s.N for s in sets)
    ps = [pad_tokens(s, N) for s in sets]
    e2 = None
    if any(s.entity2 is not None for s in ps):
        e2 = torch.stack([s._e2() for s in ps])
    return TokenSet(torch.stack([s.x for s in ps]), torch.stack([s.mask for s in ps]),
                    torch.stack([s.kind for s in ps]), torch.stack([s.time.float() for s in ps]),
                    torch.stack([s.entity for s in ps]), None, e2)


def gather_tokens(stacked: TokenSet, t_idx: torch.Tensor, b_idx: torch.Tensor) -> TokenSet:
    """Rows (t_idx[i], b_idx[i]) of a time-stacked TokenSet -> TokenSet [len(idx), N, D]."""
    return TokenSet(stacked.x[t_idx, b_idx], stacked.mask[t_idx, b_idx], stacked.kind[t_idx, b_idx],
                    stacked.time[t_idx, b_idx], stacked.entity[t_idx, b_idx], None,
                    None if stacked.entity2 is None else stacked.entity2[t_idx, b_idx])


def repeat_tokens(ts: TokenSet, n: int) -> TokenSet:
    """Each batch row repeated n times consecutively (row b -> rows b*n .. b*n+n-1)."""
    idx = torch.arange(ts.B, device=ts.x.device).repeat_interleave(n)
    return ts.select(idx)


def outcome_select(ob: OutcomeBatch, idx) -> OutcomeBatch:
    """Batch rows of an OutcomeBatch."""
    idx_l = [int(i) for i in idx]
    return OutcomeBatch(ob.q.select(idx), ob.lik[idx], ob.n_classes[idx], ob.y[idx], ob.censored[idx],
                        ob.y_mask[idx], ob.channel[idx], None if ob.keys is None else [ob.keys[i] for i in idx_l])


def outcome_pad(ob: OutcomeBatch, Q: int) -> OutcomeBatch:
    """Right-pad the query axis to Q (padding queries are unobserved and masked)."""
    q0 = ob.q.N
    if q0 == Q:
        return ob
    B = ob.q.B
    p = Q - q0
    dev = ob.lik.device

    def padc(t, val, dtype):
        return torch.cat([t.to(dtype), torch.full((B, p), val, dtype=dtype, device=dev)], 1)

    keys = None if ob.keys is None else [list(k) + [None] * p for k in ob.keys]
    return OutcomeBatch(pad_tokens(ob.q, Q), padc(ob.lik, 0, torch.long), padc(ob.n_classes, 0, torch.long),
                        padc(ob.y, 0.0, torch.float32), padc(ob.censored, False, torch.bool),
                        padc(ob.y_mask, False, torch.bool), padc(ob.channel, -1, torch.long), keys)


@contextlib.contextmanager
def seeded_from(generator: Optional[torch.Generator], device=None):
    """Run torch sampling ops that do not take a generator (e.g. Gamma) reproducibly from ``generator``.
    Without a generator this is a no-op (global RNG)."""
    if generator is None:
        yield
        return
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator, device=generator.device).item())
    devices = []
    if device is not None and torch.device(device).type == "cuda":
        devices = [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield
