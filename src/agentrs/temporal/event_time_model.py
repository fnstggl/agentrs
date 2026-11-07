"""Time-to-event and cumulative-count distributions over continuous elapsed time.

Event times use a **mixture cure model with a piecewise-linear hazard in log time**::

    P(event ever happens)          = pi                       (pi = sigmoid(cure logit))
    lambda(tau)                    = piecewise linear between G grid nodes (rates >= 0)
    Lambda(tau)                    = integral_0^tau lambda    (closed form per segment)
    P(T <= h)                      = pi * (1 - exp(-Lambda(tau(h))))

so "not yet" (right-censored, the event can still happen) and "never" (1 - pi) are different
quantities, the CDF is monotone in h by construction, and every duration — not only grid nodes or
configured horizons — has a probability.

Likelihoods (all in tau space; the Jacobian of tau(h) does not depend on parameters):

* exact event time t (the environment reported when it happened):   log pi + log lambda(t) - Lambda(t)
* interval-censored (seen absent at a, present at b):              log[F(b) - F(a)]
* right-censored at c (pending, observation stopped, or a competing
  event such as removal made further observation meaningless):     log[1 - F(c)]

A pending outcome is therefore never trained as a failure. Competing risks are handled
cause-specifically: a competing terminal event censors the others at its time.

Cumulative counts (replies, reactions, views) use the same integral for a monotone mean
``mu(tau) = integral of rate`` and a negative-binomial likelihood on *increments* between
consecutive observations, so observing the same action at many horizons never counts the same
reply twice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from agentrs.temporal.time_encoding import DAY, log_time

EPS = 1e-6
TAU_MAX = float(log_time(60 * DAY))


def make_grid(n: int = 24, tau_max: float = TAU_MAX) -> torch.Tensor:
    return torch.linspace(0.0, tau_max, n)


def piecewise_linear_integral(rates: torch.Tensor, tau: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rates (..., G) >= 0 at equally spaced nodes (spacing dt); tau (..., Q) query points.

    Returns (Lambda(tau), lambda(tau)), each (..., Q). Beyond the last node the last rate is
    extrapolated as a constant."""
    G = rates.shape[-1]
    seg = 0.5 * (rates[..., 1:] + rates[..., :-1]) * dt  # (..., G-1)
    cum = torch.cat([torch.zeros_like(rates[..., :1]), torch.cumsum(seg, -1)], -1)  # (..., G)
    tau = tau.to(rates.dtype)
    tau_max = dt * (G - 1)
    t_in = tau.clamp(0.0, tau_max)
    idx = torch.clamp((t_in / dt).floor().long(), 0, G - 2)
    d = t_in - idx.to(tau.dtype) * dt
    r0 = torch.gather(rates, -1, idx)
    r1 = torch.gather(rates, -1, idx + 1)
    c0 = torch.gather(cum, -1, idx)
    slope = (r1 - r0) / dt
    Lam = c0 + r0 * d + 0.5 * slope * d * d
    lam = r0 + slope * d
    extra = (tau - tau_max).clamp(min=0.0)
    Lam = Lam + rates[..., -1:].expand_as(extra) * extra
    lam = torch.where(tau > tau_max, rates[..., -1:].expand_as(lam), lam)
    return Lam, lam


def cure_cdf(pi_logit: torch.Tensor, Lam: torch.Tensor) -> torch.Tensor:
    """P(T <= h) for a cure model; pi_logit broadcastable to Lam."""
    return torch.sigmoid(pi_logit) * (-torch.expm1(-Lam))


def conditional_cdf(pi_logit, Lam_h, Lam_c):
    """P(T <= h | T > c) for h >= c."""
    Fh, Fc = cure_cdf(pi_logit, Lam_h), cure_cdf(pi_logit, Lam_c)
    return ((Fh - Fc) / (1.0 - Fc).clamp(min=EPS)).clamp(0.0, 1.0)


# label kinds
NONE, EXACT, INTERVAL, RIGHT = 0, 1, 2, 3
KIND_NAMES = {NONE: "none", EXACT: "exact", INTERVAL: "interval", RIGHT: "right_censored"}


def event_log_likelihood(pi_logit: torch.Tensor, Lam: torch.Tensor, lam: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """Per-label log-likelihood.

    pi_logit: (...,)      cure logit
    Lam, lam: (..., 2)    integral / rate at the label's two time points (a, b):
                           exact -> (t, t), interval -> (a, b), right -> (c, c)
    kind:     (...,)      NONE / EXACT / INTERVAL / RIGHT (NONE gives 0)
    """
    log_pi = F.logsigmoid(pi_logit)
    log_1m_pi = F.logsigmoid(-pi_logit)
    La, Lb = Lam[..., 0], Lam[..., 1]
    exact = log_pi + torch.log(lam[..., 1] + EPS) - Lb
    diff = (Lb - La).clamp(min=1e-8)
    interval = log_pi - La + torch.log(-torch.expm1(-diff) + 1e-12)
    right = torch.logaddexp(log_1m_pi, log_pi - Lb)
    out = torch.zeros_like(La)
    out = torch.where(kind == EXACT, exact, out)
    out = torch.where(kind == INTERVAL, interval, out)
    out = torch.where(kind == RIGHT, right, out)
    return out


def negbin_log_prob(k: torch.Tensor, mean: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    mean = mean.clamp(min=EPS)
    r = r.clamp(min=EPS)
    return (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1.0) + r * (torch.log(r) - torch.log(r + mean))
            + k * (torch.log(mean) - torch.log(r + mean)))


def time_quantile(pi: float, Lam_grid: Sequence[float], tau_grid: Sequence[float], q: float) -> float:
    """Smallest tau with P(T <= tau | ever) >= q, i.e. Lambda(tau) >= -log(1-q); inf if never reached."""
    target = -math.log(1.0 - q)
    prev_t, prev_L = 0.0, 0.0
    for t, L in zip(tau_grid, Lam_grid):
        if L >= target:
            if L == prev_L:
                return t
            return prev_t + (target - prev_L) / (L - prev_L) * (t - prev_t)
        prev_t, prev_L = t, L
    return float("inf")
