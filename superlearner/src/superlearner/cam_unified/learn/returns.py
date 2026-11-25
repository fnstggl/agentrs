"""Returns, advantages and off-policy targets for CAM Unified decisions on a semi-Markov clock.

Copy-adapted from ``superlearner/src/superlearner/unified/learning/returns.py`` (CAM 6.1: ``lambda_returns``,
``vtrace`` after Espeholt et al. 2018, ``ReturnNormalizer``; tested in ``superlearner/tests/unified/test_returns.py``).
Changes for CAM Unified:

* discounts come from the SMDP clock, ``gamma_t = exp(-kappa * dt_days)`` (architecture decision 5: one decision = one
  tool call, the harness then advances the world by ``cost_s + think_s``), so a long action or option is discounted by
  its real elapsed simulated time;
* every recursion takes a validity mask for right-padded batches (``mask[b, t] = False`` after row ``b`` ended); the
  bootstrap value of a row is ``values[b, L_b]``, the first position after its last valid step;
* GAE (Schulman et al. 2016) for fresh on-policy data; V-trace also returns the policy-gradient advantages
  ``rho_t (r_t + gamma_t vs_{t+1} - V_t)``;
* option-level SMDP targets ``R_option + gamma_Delta V(s_next)`` with ``gamma_Delta = exp(-kappa * Delta_t)``.

Shapes: B rows (environments / actors), T decisions. ``rewards, gammas, continues, mask`` are ``[B, T]``;
``values`` is ``[B, T + 1]``. ``continues[b, t] = 0`` where the transition t -> t+1 reached a true terminal state;
a truncated row keeps 1 and bootstraps from ``values``.
"""
from __future__ import annotations

import torch
from torch import Tensor


def semi_markov_discount(dt_days, kappa: float) -> Tensor:
    """``gamma = exp(-kappa * dt)`` for elapsed simulated days ``dt >= 0`` (any shape). Zero elapsed time is not
    discounted; an option or action that takes ``n`` days is discounted exactly like ``n`` one-day steps."""
    dt = torch.as_tensor(dt_days, dtype=torch.float32) if not torch.is_tensor(dt_days) else dt_days
    if not torch.isfinite(dt).all():
        raise ValueError("elapsed time must be finite")
    if (dt < 0).any():
        raise ValueError("elapsed time must be non-negative")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return torch.exp(-float(kappa) * dt)


def _prep(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, mask: Tensor | None):
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [B, T], got {tuple(rewards.shape)}")
    B, T = rewards.shape
    if values.shape != (B, T + 1):
        raise ValueError(f"values must be [B, T+1] = {(B, T + 1)}, got {tuple(values.shape)}")
    if gammas.shape != (B, T) or continues.shape != (B, T):
        raise ValueError("rewards, gammas and continues must have the same [B, T] shape")
    m = torch.ones(B, T, dtype=torch.bool, device=rewards.device) if mask is None else mask.bool()
    if m.shape != (B, T):
        raise ValueError("mask must be [B, T]")
    return B, T, m, continues.to(rewards.dtype)


def lambda_returns(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, lam: float,
                   mask: Tensor | None = None) -> Tensor:
    """``G_t = r_t + gamma_t c_t ((1 - lam) V_{t+1} + lam G_{t+1})`` with ``G`` at a padded position equal to the value
    there (so the last valid step bootstraps from ``values[b, L_b]``). Returns ``[B, T]`` (0 at padded steps)."""
    B, T, m, c = _prep(rewards, values, gammas, continues, mask)
    out = torch.zeros_like(rewards)
    nxt = values[:, T]
    for t in reversed(range(T)):
        g = rewards[:, t] + gammas[:, t] * c[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        g = torch.where(m[:, t], g, values[:, t])
        out[:, t] = torch.where(m[:, t], g, torch.zeros_like(g))
        nxt = g
    return out


def gae(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, lam: float,
        mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Generalised advantage estimation with per-step (semi-Markov) discounts.

    ``delta_t = r_t + gamma_t c_t V_{t+1} - V_t``; ``A_t = delta_t + gamma_t c_t lam A_{t+1}``; ``A = 0`` at padding.
    Returns ``(advantages [B, T], returns [B, T])`` with ``returns = A + V`` (the lambda-return); both 0 at padding.
    """
    B, T, m, c = _prep(rewards, values, gammas, continues, mask)
    adv = torch.zeros_like(rewards)
    acc = torch.zeros_like(values[:, 0])
    mf = m.to(rewards.dtype)
    for t in reversed(range(T)):
        disc = gammas[:, t] * c[:, t]
        delta = rewards[:, t] + disc * values[:, t + 1] - values[:, t]
        acc = mf[:, t] * (delta + disc * lam * acc)
        adv[:, t] = acc
    ret = (adv + values[:, :T]) * mf
    return adv, ret


def vtrace(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, log_ratio: Tensor, lam: float = 1.0,
           rho_clip: float = 1.0, c_clip: float = 1.0, pg_rho_clip: float | None = None,
           mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """V-trace off-policy targets (Espeholt et al. 2018) with semi-Markov discounts.

    ``log_ratio [B, T] = log pi(a_t | s_t) - log mu(a_t | s_t)`` for the recorded action under the current policy pi
    and the behaviour policy mu (detached by the caller). ``vs_t = V_t + sum_{s>=t} (prod_{u<s} gamma_u c_u)
    rho_s delta_s``, ``rho = min(rho_clip, ratio)``, ``c = lam min(c_clip, ratio)``.

    Returns ``(vs [B, T + 1], pg_adv [B, T], rho [B, T])``: ``vs[:, L_b]`` equals the bootstrap value;
    ``pg_adv_t = min(pg_rho_clip, ratio_t) (r_t + gamma_t c_t vs_{t+1} - V_t)`` (0 at padding). With zero log-ratio,
    ``rho_clip = c_clip = 1`` the targets equal the lambda-returns.
    """
    B, T, m, c_ = _prep(rewards, values, gammas, continues, mask)
    if log_ratio.shape != (B, T):
        raise ValueError("log_ratio must be [B, T]")
    ratio = log_ratio.clamp(max=30.0).exp()
    rho = ratio.clamp(max=rho_clip)
    cc = lam * ratio.clamp(max=c_clip)
    disc = gammas * c_
    delta = rho * (rewards + disc * values[:, 1:] - values[:, :-1])
    mf = m.to(rewards.dtype)
    vs = values.clone()
    acc = torch.zeros_like(values[:, 0])
    for t in reversed(range(T)):
        acc = mf[:, t] * (delta[:, t] + disc[:, t] * cc[:, t] * acc)
        vs[:, t] = values[:, t] + acc
    pg_rho = ratio.clamp(max=rho_clip if pg_rho_clip is None else pg_rho_clip)
    pg_adv = pg_rho * (rewards + disc * vs[:, 1:] - values[:, :-1]) * mf
    return vs, pg_adv, rho * mf


def option_targets(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, option_start: Tensor,
                   mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Semi-Markov targets for temporally extended options (spec 9.4).

    An option starts at every ``t`` with ``option_start[b, t]`` and runs until the next start, a terminal transition
    or the end of the row's valid data at ``e``. Its target is

        target_s = R_option + Gamma * V(s_e),   R_option = sum_{t=s}^{e-1} (prod_{u=s}^{t-1} gamma_u) r_t,
        Gamma = prod_{t=s}^{e-1} gamma_t c_t   (= exp(-kappa * Delta_t) for gamma_t = exp(-kappa dt_t))

    ``values[b, e]`` is the value of the state where the next option begins (or the bootstrap at the row end).
    Returns ``(target [B, T], gamma_option [B, T])``, both meaningful only where ``option_start`` (0 elsewhere).
    """
    B, T, m, c = _prep(rewards, values, gammas, continues, mask)
    start = option_start.bool()
    G = torch.zeros_like(rewards)
    Gam = torch.zeros_like(rewards)
    nxt_G = values[:, T]
    nxt_Gam = torch.ones_like(values[:, T])
    nxt_boundary = torch.ones(B, dtype=torch.bool, device=rewards.device)   # position T is always a boundary
    for t in reversed(range(T)):
        disc = gammas[:, t] * c[:, t]
        boot = torch.where(nxt_boundary, values[:, t + 1], nxt_G)
        gboot = torch.where(nxt_boundary, torch.ones_like(nxt_Gam), nxt_Gam)
        g = rewards[:, t] + disc * boot
        gm = disc * gboot
        G[:, t] = torch.where(m[:, t], g, torch.zeros_like(g))
        Gam[:, t] = torch.where(m[:, t], gm, torch.zeros_like(gm))
        nxt_G, nxt_Gam = g, gm
        nxt_boundary = start[:, t] | ~m[:, t]
    keep = (start & m).to(rewards.dtype)
    return G * keep, Gam * keep


class ReturnNormalizer:
    """Per-family EMA of the 5th-95th percentile range of returns, floored at ``limit`` so small returns are not
    inflated (DreamerV3 percentile scaling; copy-adapted from CAM 6.1). Families are kept separate so one family's
    reward scale (e.g. cents of revenue vs. a 0/1 success bonus) cannot erase differences inside another."""

    def __init__(self, rate: float = 0.01, limit: float = 1.0, low: float = 0.05, high: float = 0.95):
        self.rate, self.limit, self.low, self.high = rate, limit, low, high
        self.ranges: dict[str, tuple[float, float]] = {}

    def update(self, returns: Tensor, family: str = "default", mask: Tensor | None = None) -> float:
        flat = returns.detach().float()
        flat = flat[mask.bool()] if mask is not None else flat.flatten()
        if flat.numel() == 0:
            return self.scale(family)
        q = torch.quantile(flat, torch.tensor([self.low, self.high], device=flat.device))
        lo, hi = float(q[0]), float(q[1])
        if family in self.ranges:
            old_lo, old_hi = self.ranges[family]
            lo = (1 - self.rate) * old_lo + self.rate * lo
            hi = (1 - self.rate) * old_hi + self.rate * hi
        self.ranges[family] = (lo, hi)
        return self.scale(family)

    def scale(self, family: str = "default") -> float:
        if family not in self.ranges:
            return self.limit
        lo, hi = self.ranges[family]
        return max(self.limit, hi - lo)

    def state_dict(self) -> dict:
        return {"ranges": dict(self.ranges), "rate": self.rate, "limit": self.limit, "low": self.low,
                "high": self.high}

    def load_state_dict(self, state: dict) -> None:
        self.ranges = {k: tuple(v) for k, v in state["ranges"].items()}
        for k in ("rate", "limit", "low", "high"):
            if k in state:
                setattr(self, k, state[k])
