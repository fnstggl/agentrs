"""Decision-relevant value of information (meta-reward spec 8.2; manifest KV-01).

    VOI(a) = E_{y ~ p(y | b, a)} [ max_{a'} Q(b_updated(a, y), a', g) ] - max_{a'} Q(b, a', g) - cost(a)

`voi_monte_carlo` estimates it with CAM's own components passed as callables: `sample_outcomes(a, n)` draws n
predicted outcomes of experiment a from the world model / belief predictive, `update(a, y)` returns the updated
belief (an independent clone of the fast state, never the real one), and `q_values(belief)` returns the values of the
continuation candidates under a belief. Elapsed time, resource cost and risk enter through `cost(a)` and a discount
of the post-information value (`gamma(a)`): an experiment that takes a long time or burns funds is worth less even if
it is informative. Repeated irreducible noise has VOI ~ 0 (it does not move max Q), unlike prediction-error
curiosity: `voi_monte_carlo` on a pure-noise channel is tested to be ~0 while surprise is large.

`knowledge_gradient_gaussian` is the closed form for independent Gaussian arms (Frazier et al. 2008), used as the
exact reference (and as CAM 7.0's programmed KG baseline).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


def voi_monte_carlo(candidates: Sequence, sample_outcomes: Callable, update: Callable, q_values: Callable,
                    belief0, n: int = 64, cost: Optional[Callable] = None, gamma: Optional[Callable] = None) -> Tensor:
    """VOI for each candidate experiment [len(candidates)] (common random numbers are the caller's choice)."""
    base = float(torch.as_tensor(q_values(belief0)).max())
    out = []
    for a in candidates:
        ys = sample_outcomes(a, n)
        vals = torch.stack([torch.as_tensor(q_values(update(a, y))).max() for y in ys]).float()
        g = float(gamma(a)) if gamma is not None else 1.0
        c = float(cost(a)) if cost is not None else 0.0
        out.append(g * vals.mean() - base - c)
    return torch.stack(out)


def knowledge_gradient_gaussian(mu: Tensor, sigma2: Tensor, noise2: float) -> Tensor:
    """KG of measuring each arm once (independent normal beliefs, known noise variance)."""
    mu = mu.double()
    s2 = sigma2.double()
    sig_tilde = s2 / torch.sqrt(s2 + noise2)
    K = mu.shape[0]
    kg = torch.zeros(K, dtype=torch.float64)
    for i in range(K):
        others = torch.cat([mu[:i], mu[i + 1:]]).max() if K > 1 else torch.tensor(-math.inf, dtype=torch.float64)
        z = -(mu[i] - others).abs() / sig_tilde[i].clamp_min(1e-300)
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        kg[i] = sig_tilde[i] * (z * Phi + phi)
    return kg


def gaussian_arm_callables(mu: Tensor, sigma2: Tensor, noise2: float, gen: torch.Generator):
    """CAM-style callables for independent Gaussian arms (for tests and as a baseline adapter)."""
    def sample(a, n):
        pred_sd = math.sqrt(float(sigma2[a]) + noise2)
        return list(float(mu[a]) + pred_sd * torch.randn(n, generator=gen, dtype=torch.float64))

    def update(a, y):
        m, s2 = mu.clone().double(), sigma2.clone().double()
        k = s2[a] / (s2[a] + noise2)
        m[a] = m[a] + k * (y - m[a])
        s2[a] = (1 - k) * s2[a]
        return (m, s2)

    def q(b):
        return b[0] if isinstance(b, tuple) else b
    return sample, update, q
