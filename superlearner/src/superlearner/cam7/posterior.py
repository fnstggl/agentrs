"""Fast Bayesian inference for CAM 7.0: conjugate regression in learned features, and the value of information.

Model, for each outcome k (revenue, cost) of one environment:

    y_t = w_k^T phi_t + eps_t,           eps_t ~ N(0, s2_k)
    w_k | s2_k ~ N(mu0_k, s2_k Lambda0_k^{-1}),   s2_k ~ InvGamma(a0_k, b0_k)

Rows carry weights omega_t in [0, 1] (exponential forgetting, change-point cuts, padding). The posterior is exact:

    Lambda_n = Lambda0 + Phi^T Omega Phi
    mu_n     = Lambda_n^{-1} (Lambda0 mu0 + Phi^T Omega y)
    a_n      = a0 + sum(omega) / 2
    b_n      = b0 + [ (y - Phi mu_n)^T Omega (y - Phi mu_n) + (mu_n - mu0)^T Lambda0 (mu_n - mu0) ] / 2

and the predictive distribution of an outcome at features phi is Student-t with 2 a_n degrees of freedom, location
phi^T mu_n and squared scale (b_n / a_n) (1 + phi^T Lambda_n^{-1} phi).

With the noise variance known, this is exactly the recursion in the specification (innovation, gain K_t, mean and
covariance updates) applied to all rows at once. Computing it from sufficient statistics through a Cholesky factor of
Lambda_n never forms an explicit inverse, keeps Lambda_n positive definite by construction (diagonal positive prior
plus a positive semi-definite Gram matrix, plus a scale-relative jitter), and lets the features that enter Phi change
as the learned representation of the history changes, which a running recursion could not. Linear algebra is float64.

The noise variance is not known: it is inferred jointly (normal-inverse-gamma), so predictions widen when outcomes are
noisier than the prior expected and the scale of each task's outcomes is learned inside the episode instead of being
fixed by the training distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

F64 = torch.float64


@dataclass
class Posterior:
    """Batched posterior over B environments and K outcomes with D features."""
    mu: Tensor        # [B, K, D]
    chol: Tensor      # [B, K, D, D] lower Cholesky factor of Lambda_n
    a: Tensor         # [B, K]
    b: Tensor         # [B, K]
    n: Tensor         # [B, K] effective number of rows (sum of omega)

    @property
    def s2(self) -> Tensor:
        """Posterior mean-like estimate of the noise variance, b_n / a_n."""
        return self.b / self.a

    @property
    def dof(self) -> Tensor:
        return 2.0 * self.a


def fit(phi: Tensor, y: Tensor, omega: Tensor, mu0: Tensor, lam0: Tensor, a0: Tensor, b0: Tensor) -> Posterior:
    """phi [B, T, D]; y [B, T, K]; omega [B, T] (0 = row absent); mu0, lam0 [B, K, D] or [K, D] (lam0 = diagonal of
    the prior precision); a0, b0 [B, K] or [K]. Returns the posterior. Differentiable in every argument."""
    B, T, D = phi.shape
    K = y.shape[-1]
    phi = phi.to(F64)
    y = y.to(F64)
    w = omega.to(F64)
    mu0 = mu0.to(F64).expand(B, K, D)
    lam0 = lam0.to(F64).expand(B, K, D)
    a0 = a0.to(F64).expand(B, K)
    b0 = b0.to(F64).expand(B, K)
    gram = torch.einsum("btd,bt,bte->bde", phi, w, phi)                       # [B, D, D], shared by the outcomes
    prec = gram.unsqueeze(1) + torch.diag_embed(lam0)                           # [B, K, D, D]
    jitter = 1e-9 * prec.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12)
    prec = prec + torch.diag_embed(jitter.expand(B, K, D))
    chol = torch.linalg.cholesky(prec)
    rhs = lam0 * mu0 + torch.einsum("btd,bt,btk->bkd", phi, w, y)             # [B, K, D]
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    resid = y - torch.einsum("btd,bkd->btk", phi, mu)                          # [B, T, K]
    sse = torch.einsum("bt,btk->bk", w, resid * resid)
    dev = mu - mu0
    reg = (dev * lam0 * dev).sum(-1)
    n = w.sum(1, keepdim=True).expand(B, K)
    a = a0 + 0.5 * n
    b = b0 + 0.5 * (sse + reg)
    return Posterior(mu, chol, a, b, n)


def _solve_l(post: Posterior, phi: Tensor) -> Tensor:
    """L^{-1} phi^T for query features phi [B, Q, D] -> [B, K, D, Q]."""
    B, Q, D = phi.shape
    K = post.mu.shape[1]
    rhs = phi.to(F64).transpose(1, 2).unsqueeze(1).expand(B, K, D, Q)
    return torch.linalg.solve_triangular(post.chol, rhs, upper=False)


def predict(post: Posterior, phi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Predictive Student-t at query features phi [B, Q, D]: location [B, Q, K], squared scale [B, Q, K], dof [B, K]."""
    loc = torch.einsum("bqd,bkd->bqk", phi.to(F64), post.mu)
    v = _solve_l(post, phi)                                                    # [B, K, D, Q]
    quad = (v * v).sum(2).transpose(1, 2)                                      # [B, Q, K]
    scale2 = post.s2.unsqueeze(1) * (1.0 + quad)
    return loc, scale2, post.dof


def mean_cov(post: Posterior, phi: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean [B, Q, K] and covariance [B, K, Q, Q] of the noise-free outcome phi^T w at query features
    (the Student-t marginal's covariance with the plug-in scale b_n / a_n)."""
    loc = torch.einsum("bqd,bkd->bqk", phi.to(F64), post.mu)
    v = _solve_l(post, phi)
    cov = post.s2[..., None, None] * torch.einsum("bkdq,bkdr->bkqr", v, v)
    return loc, cov


def t_nll(y: Tensor, loc: Tensor, scale2: Tensor, dof: Tensor) -> Tensor:
    """Negative log density of y under Student-t(dof, loc, scale2); y, loc, scale2 [B, Q, K], dof [B, K]."""
    y = y.to(F64)
    nu = dof.unsqueeze(1).expand_as(loc)
    z2 = (y - loc) ** 2 / scale2
    return (-torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2) + 0.5 * torch.log(nu * math.pi * scale2)
            + (nu + 1) / 2 * torch.log1p(z2 / nu))


def sample_weights(post: Posterior, n: int, generator: torch.Generator | None = None, np_rng=None) -> Tensor:
    """n joint posterior draws of (s2, w) for each environment and outcome; returns w [B, n, K, D]. With `generator`
    and `np_rng` given, the draws are reproducible."""
    B, K, D = post.mu.shape
    dev = post.mu.device
    # s2 ~ InvGamma(a, b): b / Gamma(a, 1)
    shape = post.a.unsqueeze(1).expand(B, n, K)
    if np_rng is not None:
        g = torch.from_numpy(np_rng.standard_gamma(shape.detach().cpu().numpy())).to(dev, F64)
    else:
        g = torch._standard_gamma(shape.contiguous())
    s2 = post.b.unsqueeze(1) / g.clamp_min(1e-12)                              # [B, n, K]
    z = torch.randn(B, n, K, D, 1, dtype=F64, device=dev, generator=generator)
    # w = mu + sqrt(s2) L^{-T} z
    lt = post.chol.unsqueeze(1).expand(B, n, K, D, D).transpose(-1, -2)
    eps = torch.linalg.solve_triangular(lt, z, upper=True).squeeze(-1)
    return post.mu.unsqueeze(1) + s2.sqrt().unsqueeze(-1) * eps


GH_POINTS = 32


def _normal_quantiles(n: int, device, dtype=F64) -> Tensor:
    """Deterministic quadrature for E[f(Z)], Z ~ N(0, 1): the midpoints of n equal-probability bins."""
    p = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return math.sqrt(2.0) * torch.erfinv(2 * p - 1)


def knowledge_gradient(mean: Tensor, cov: Tensor, obs_var: Tensor, valid: Tensor | None = None,
                       n_points: int = GH_POINTS) -> Tensor:
    """Decision-relevant value of one observation of each candidate.

    mean [B, C]: current expected value of each candidate; cov [B, C, C]: posterior covariance of the candidates'
    values; obs_var [B, C]: noise variance of the observation that trying candidate a would produce. For candidate a,
    observing y_a moves every candidate's posterior mean linearly: mean'(a') = mean(a') + cov(a', a) / sd_a * Z with
    sd_a^2 = cov(a, a) + obs_var(a) and Z standard normal. The knowledge gradient is

        KG(a) = E_Z[ max_a' mean'(a') ] - max_a' mean(a')  >= 0

    i.e. VOI(a) = E_y[max_a' Q(updated belief(a, y), a')] - max_a' Q(current belief, a') with Q the posterior mean
    value of holding a candidate. Invalid candidates (valid False) are excluded from both the max and the result (0).
    """
    B, C = mean.shape
    mean = mean.to(F64)
    cov = cov.to(F64)
    big = torch.finfo(F64).max / 4
    if valid is None:
        valid = torch.ones(B, C, dtype=torch.bool, device=mean.device)
    m = torch.where(valid, mean, torch.full_like(mean, -big))
    sd = (cov.diagonal(dim1=-2, dim2=-1) + obs_var.to(F64)).clamp_min(1e-30).sqrt()   # [B, C]
    beta = cov / sd.unsqueeze(1)                                               # [B, C(a'), C(a)]
    z = _normal_quantiles(n_points, mean.device)
    vals = m[:, :, None, None] + beta[..., None] * z                           # [B, C', C, S]
    vals = torch.where(valid[:, :, None, None], vals, torch.full_like(vals, -big))
    best_after = vals.max(1).values.mean(-1)                                   # [B, C]
    kg = (best_after - m.max(1, keepdim=True).values).clamp_min(0.0)
    return torch.where(valid, kg, torch.zeros_like(kg))


def info_gain(cov: Tensor, obs_var: Tensor) -> Tensor:
    """Expected information gain (nats) about the candidate's own value from one observation: 0.5 log(1 + var/noise).
    The decision-blind comparison method for the value of information."""
    var = cov.diagonal(dim1=-2, dim2=-1).to(F64)
    return 0.5 * torch.log1p(var / obs_var.to(F64).clamp_min(1e-30))
