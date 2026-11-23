"""Duration-aware discounting, lambda returns and return-scale normalisation."""
from __future__ import annotations

import torch
from torch import Tensor


def duration_discount(dt_seconds: Tensor, gamma_day: float, seconds_per_day: float = 86400.0) -> Tensor:
    """``gamma_t = gamma_day ** (dt_t / one_day)``. A zero-duration action is not discounted."""
    if (dt_seconds < 0).any():
        raise ValueError("elapsed time must be non-negative")
    return torch.pow(torch.as_tensor(gamma_day, dtype=dt_seconds.dtype, device=dt_seconds.device),
                     dt_seconds / seconds_per_day)


def lambda_returns(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, lam: float) -> Tensor:
    """Compute ``G_t = r_t + gamma_t * c_t * ((1 - lam) * V_{t+1} + lam * G_{t+1})``.

    Args:
        rewards: ``[B, T]``, reward received on the transition from ``t`` to ``t + 1``.
        values: ``[B, T + 1]``, value estimates; ``values[:, T]`` bootstraps the final step.
        gammas: ``[B, T]``, duration-aware discounts for each transition.
        continues: ``[B, T]``, 0 where the transition reached a true terminal state, else 1.
            A truncated sequence keeps 1 so it bootstraps; a terminal step never does.
        lam: the lambda mixing coefficient.

    Returns:
        ``[B, T]`` returns.
    """
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError(f"values must have T+1 steps: {values.shape} vs {rewards.shape}")
    if rewards.shape != gammas.shape or rewards.shape != continues.shape:
        raise ValueError("rewards, gammas and continues must have the same shape")
    T = rewards.shape[1]
    out = torch.zeros_like(rewards)
    next_return = values[:, T]
    for t in reversed(range(T)):
        boot = (1 - lam) * values[:, t + 1] + lam * next_return
        next_return = rewards[:, t] + gammas[:, t] * continues[:, t] * boot
        out[:, t] = next_return
    return out


def vtrace(rewards: Tensor, values: Tensor, gammas: Tensor, continues: Tensor, log_ratio: Tensor, lam: float,
           rho_clip: float = 1.0, c_clip: float = 1.0) -> tuple[Tensor, Tensor]:
    """Off-policy value targets (V-trace, Espeholt et al. 2018) with duration-aware discounts.

    ``log_ratio[B, T] = log pi(a_t) - log mu(a_t)`` for the replayed action. With a zero ratio and ``lam = 1`` this
    is the Monte Carlo return bootstrapped from ``values[:, T]``.

    Returns ``(vs [B, T + 1], rho [B, T])``; ``vs[:, T]`` equals the bootstrap value.
    """
    if values.shape[1] != rewards.shape[1] + 1:
        raise ValueError("values must have T+1 steps")
    ratio = log_ratio.exp()
    rho = ratio.clamp(max=rho_clip)
    c = lam * ratio.clamp(max=c_clip)
    disc = gammas * continues
    delta = rho * (rewards + disc * values[:, 1:] - values[:, :-1])
    T = rewards.shape[1]
    vs = values.clone()
    acc = torch.zeros_like(values[:, 0])
    for t in reversed(range(T)):
        acc = delta[:, t] + disc[:, t] * c[:, t] * acc
        vs[:, t] = values[:, t] + acc
    return vs, rho


class ReturnNormalizer:
    """Per-task-family EMA of the 5th-95th percentile return range, floored so small returns are not inflated.

    Families are kept separate so that one family's reward scale cannot erase differences in another.
    """

    def __init__(self, rate: float = 0.01, limit: float = 1.0, low: float = 0.05, high: float = 0.95):
        self.rate, self.limit, self.low, self.high = rate, limit, low, high
        self.ranges: dict[str, tuple[float, float]] = {}

    def update(self, returns: Tensor, family: str = "default") -> float:
        flat = returns.detach().flatten().float()
        lo, hi = torch.quantile(flat, torch.tensor([self.low, self.high], device=flat.device)).tolist()
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
        return {"ranges": dict(self.ranges)}

    def load_state_dict(self, state: dict) -> None:
        self.ranges = {k: tuple(v) for k, v in state["ranges"].items()}
