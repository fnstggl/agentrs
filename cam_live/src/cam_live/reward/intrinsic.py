"""cam_live.reward.intrinsic — the learned intrinsic reward generator R_psi (spec 7.2).

    r = R_psi(b_t, e_a, b_next, e_g, signals, life_ctx)        r in (-bound, bound)

A learned nonlinear function of the belief before and after the action, the action and goal embeddings, the named
signal vector of ``reward.signals`` and a recurrent summary of the whole lifetime. Nothing fixes the relative
weight of its inputs. It has no input for the verified return: verified outcomes reach psi only through the outer
meta-gradient (``reward.meta``).

* Output is bounded by a scaled tanh.
* Inputs are normalised by running statistics that can be frozen inside a meta window, so the reward that is
  differentiated is the reward the inner update used.
* ``version`` is a hash of the parameters and normalisation statistics: it changes whenever they change.
* ``LifetimeContext`` is a GRU over the per-step encodings of the lifetime. Its state is a buffer (so it is in the
  checkpoint) and nothing resets it at objective boundaries; only :meth:`LifetimeContext.new_lifetime` does.
* ``meta_parameters()`` is exactly the reward group psi (GRU included). No other optimizer may hold them.

Layout adapted from superlearner/cam_unified/selfeval/intrinsic.py at commit 3968583 (frozen statistics, lifetime
state, hash version and the non-zero output initialisation are new: a zero-initialised output layer blocks the
meta-gradient to every earlier layer).
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..nn.api import GROUP_REWARD
from .integrity import bound_reward
from .signals import N_SIGNALS, SIGNAL_SCHEMA_VERSION

REWARD_ARCH_VERSION = "1"

__all__ = ["RunningNorm", "LifetimeContext", "IntrinsicReward", "REWARD_ARCH_VERSION", "GROUP_REWARD"]


class RunningNorm(nn.Module):
    """Per-feature running mean / variance. ``forward`` never updates the statistics; :meth:`update` does, unless
    frozen. Masked entries (mask 0) neither contribute to the statistics nor produce output."""

    def __init__(self, dim: int, clip: float = 8.0, eps: float = 1e-6, max_count: float = 1e5):
        super().__init__()
        self.clip, self.eps, self.max_count = clip, eps, max_count
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: Tensor, mask: Optional[Tensor] = None) -> None:
        if bool(self.frozen):
            return
        x = x.detach().to(torch.float64).reshape(-1, x.shape[-1])
        m = torch.ones_like(x) if mask is None else mask.detach().to(torch.float64).reshape(-1, x.shape[-1])
        n = m.sum(0)
        has = n > 0
        bm = (x * m).sum(0) / n.clamp_min(1.0)
        bv = (((x - bm) ** 2) * m).sum(0) / n.clamp_min(1.0)
        tot = self.count + n
        delta = bm - self.mean
        new_mean = self.mean + delta * n / tot.clamp_min(1.0)
        new_var = (self.var * self.count + bv * n + delta ** 2 * self.count * n / tot.clamp_min(1.0)) / tot.clamp_min(1.0)
        first = has & (self.count == 0)
        new_var = torch.where(first, bv.clamp_min(self.eps) if x.shape[0] > 1 else self.var, new_var)
        self.mean.copy_(torch.where(has, new_mean, self.mean))
        self.var.copy_(torch.where(has, new_var, self.var))
        self.count.copy_(torch.where(has, tot.clamp(max=self.max_count), self.count))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z = (x - self.mean.to(x.dtype)) / torch.sqrt(self.var.to(x.dtype) + self.eps)
        z = z.clamp(-self.clip, self.clip)
        return z if mask is None else z * mask.to(x.dtype)

    def freeze(self, on: bool = True) -> None:
        self.frozen.fill_(bool(on))


class LifetimeContext(nn.Module):
    """Recurrent summary of the lifetime: h_{t+1} = GRU(x_t, h_t). The context a step's reward sees is h_t, the
    summary of everything *before* it. ``state`` persists across objectives and is saved with the module."""

    def __init__(self, d_in: int, d_life: int = 64):
        super().__init__()
        self.d_in, self.d_life = d_in, d_life
        self.cell = nn.GRUCell(d_in, d_life)
        self.register_buffer("state", torch.zeros(d_life))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def encode(self, x_seq: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """x_seq [T, d_in] in lifetime order -> (ctx [T, d_life] with ctx[t] = state before step t, h_T).
        Differentiable in the GRU parameters; ``h0`` defaults to the stored state (a constant)."""
        h = (self.state.detach() if h0 is None else h0).to(x_seq.dtype)
        out = []
        for t in range(x_seq.shape[0]):
            out.append(h)
            h = self.cell(x_seq[t][None], h[None])[0]
        ctx = torch.stack(out) if out else x_seq.new_zeros(0, self.d_life)
        return ctx, h

    @torch.no_grad()
    def advance(self, x_seq: Tensor) -> Tensor:
        """Consume new steps of the lifetime and store the new state. Returns the contexts those steps saw."""
        ctx, h = self.encode(x_seq.detach())
        self.state.copy_(h.to(self.state.dtype))
        self.steps += x_seq.shape[0]
        return ctx

    @torch.no_grad()
    def new_lifetime(self) -> None:
        """The only reset. Called when a new lifetime starts — never at an objective boundary."""
        self.state.zero_()
        self.steps.zero_()


class IntrinsicReward(nn.Module):
    def __init__(self, d_model: int, n_signals: int = N_SIGNALS, d_hidden: int = 256, d_life: int = 64,
                 bound: float = 1.0, use_action: bool = True, out_init_std: float = 0.02):
        super().__init__()
        self.d_model, self.n_signals, self.d_hidden, self.bound = d_model, n_signals, d_hidden, float(bound)
        self.use_action = use_action                  # ablation: reward state changes only
        self.null_goal = nn.Parameter(torch.zeros(d_model))          # goal-free mode
        self.norm_b, self.norm_a, self.norm_g = RunningNorm(d_model), RunningNorm(d_model), RunningNorm(d_model)
        self.norm_s = RunningNorm(n_signals)
        self.latent_in = nn.Linear(4 * d_model, d_hidden)
        self.signal_in = nn.Linear(2 * n_signals, d_hidden)
        self.lifetime = LifetimeContext(d_hidden, d_life)
        self.life_in = nn.Linear(d_life, d_hidden)
        self.body = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden), nn.SiLU())
        self.out = nn.Linear(d_hidden, 1)
        nn.init.normal_(self.out.weight, std=out_init_std)           # not zero: every layer must receive gradient
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.null_goal, std=0.02)
        self._version_key: Optional[tuple] = None
        self._version: str = ""

    # ---------------------------------------------------------------------------------------- parameters
    def meta_parameters(self) -> List[nn.Parameter]:
        """Exactly the reward group psi: every parameter of this module and nothing else."""
        return list(self.parameters())

    def psi(self) -> Dict[str, nn.Parameter]:
        """Named psi for ``torch.func.functional_call`` / ``reward.meta``."""
        return dict(self.named_parameters())

    def _norms(self) -> List[RunningNorm]:
        return [self.norm_b, self.norm_a, self.norm_g, self.norm_s]

    @property
    def version(self) -> str:
        """Hash of parameters and normalisation statistics (not of the lifetime state). Changes iff they change."""
        tensors = list(self.parameters()) + [t for n in self._norms() for t in (n.mean, n.var)]
        key = tuple((id(t), t._version) for t in tensors)
        if key != self._version_key:
            h = hashlib.sha256()
            for t in tensors:
                c = t.detach().cpu().contiguous()
                h.update(f"{tuple(c.shape)}|{c.dtype}|".encode())
                h.update(c.view(-1).view(torch.uint8).numpy().tobytes())
            self._version = f"rpsi{REWARD_ARCH_VERSION}-s{SIGNAL_SCHEMA_VERSION}-{h.hexdigest()[:16]}"
            self._version_key = key
        return self._version

    # ---------------------------------------------------------------------------------------- statistics
    @torch.no_grad()
    def observe(self, b_t: Tensor, e_a: Tensor, b_next: Tensor, e_g: Optional[Tensor], signals: Tensor,
                signal_mask: Optional[Tensor] = None) -> None:
        """Update the input statistics from real experience (no-op while frozen)."""
        self.norm_b.update(torch.cat([b_t, b_next], 0))
        self.norm_a.update(e_a)
        if e_g is not None:
            self.norm_g.update(e_g)
        self.norm_s.update(signals, signal_mask)

    def freeze_stats(self, on: bool = True) -> None:
        for n in self._norms():
            n.freeze(on)

    @property
    def stats_frozen(self) -> bool:
        return all(bool(n.frozen) for n in self._norms())

    @contextlib.contextmanager
    def meta_window(self) -> Iterator[None]:
        """Freeze the normalisation statistics for the duration of a meta window."""
        before = [bool(n.frozen) for n in self._norms()]
        self.freeze_stats(True)
        try:
            yield
        finally:
            for n, b in zip(self._norms(), before):
                n.freeze(b)

    # ---------------------------------------------------------------------------------------- forward
    def step_features(self, b_t: Tensor, e_a: Tensor, b_next: Tensor, e_g: Optional[Tensor], signals: Tensor,
                      signal_mask: Optional[Tensor] = None) -> Tensor:
        """[B, d_hidden] encoding of one step (also the input of the lifetime recurrence)."""
        B = b_t.shape[0]
        g = self.null_goal.expand(B, -1) if e_g is None else e_g
        nb, nn_ = self.norm_b(b_t), self.norm_b(b_next)
        na = self.norm_a(e_a) if self.use_action else torch.zeros_like(e_a)
        lat = torch.cat([nb, na, nn_ - nb, self.norm_g(g)], -1)
        m = torch.ones_like(signals) if signal_mask is None else signal_mask.to(signals.dtype)
        sig = torch.cat([self.norm_s(signals, m), m], -1)
        return self.latent_in(lat) + self.signal_in(sig)

    def head(self, x: Tensor, life_ctx: Optional[Tensor] = None) -> Tensor:
        if life_ctx is None:
            life_ctx = self.lifetime.state.detach().to(x.dtype).expand(x.shape[0], -1)
        raw = self.out(self.body(x + self.life_in(life_ctx))).squeeze(-1)
        return bound_reward(raw, self.bound)

    def forward(self, b_t: Tensor, e_a: Tensor, b_next: Tensor, e_g: Optional[Tensor], signals: Tensor,
                life_ctx: Optional[Tensor] = None, signal_mask: Optional[Tensor] = None) -> Tensor:
        """r [B] in (-bound, bound). b_t, e_a, b_next: [B, D]; e_g: [B, D] or None (goal-free: learned null goal);
        signals [B, n_signals] (``RewardSignals.to_tensor``); life_ctx [B, d_life] or None (= stored state)."""
        return self.head(self.step_features(b_t, e_a, b_next, e_g, signals, signal_mask), life_ctx)

    def reward_sequence(self, b_t: Tensor, e_a: Tensor, b_next: Tensor, e_g: Optional[Tensor], signals: Tensor,
                        signal_mask: Optional[Tensor] = None, h0: Optional[Tensor] = None,
                        advance: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Rewards of T consecutive steps of the lifetime (rows in lifetime order), with the lifetime context
        recomputed in-graph from ``h0`` (default: the stored state), so the meta-gradient also reaches the
        recurrence. Returns (r [T], ctx [T, d_life], h_T). ``advance=True`` stores h_T as the new lifetime state
        (use it once per step of real experience, never inside a meta window)."""
        x = self.step_features(b_t, e_a, b_next, e_g, signals, signal_mask)
        ctx, hT = self.lifetime.encode(x, h0)
        r = self.head(x, ctx)
        if advance:
            with torch.no_grad():
                self.lifetime.state.copy_(hT.detach().to(self.lifetime.state.dtype))
                self.lifetime.steps += x.shape[0]
        return r, ctx, hT
