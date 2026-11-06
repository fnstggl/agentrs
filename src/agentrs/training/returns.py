"""Returns and advantages for variable-length episodes.

Conventions: ``rewards[t]`` is the reward received after action ``a_t``.
``terminated`` means the final state is a true terminal state (value 0 beyond
it). If the episode was *truncated* (budget/timeout), the return is
bootstrapped with ``last_value`` = V(s_T), because the task was not decided.
"""

from __future__ import annotations

from typing import Sequence


def discounted_returns(rewards: Sequence[float], gamma: float, terminated: bool, last_value: float = 0.0) -> list[float]:
    g = 0.0 if terminated else float(last_value)
    out = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def gae(rewards: Sequence[float], values: Sequence[float], gamma: float, lam: float, terminated: bool,
        last_value: float = 0.0) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation. Returns (advantages, lambda-returns)."""
    T = len(rewards)
    adv = [0.0] * T
    nxt_v = 0.0 if terminated else float(last_value)
    a = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nxt_v - values[t]
        a = delta + gamma * lam * a
        adv[t] = a
        nxt_v = values[t]
    rets = [adv[t] + values[t] for t in range(T)]
    return adv, rets
