"""cam_live.nn.fast — the Bayesian fast adaptive state (spec 2.1): immediate numerical learning, no gradient step.

For each outcome kind a Bayesian last layer ``w ~ N(mu, Sigma)`` over the learned features ``phi`` (unit-norm, from
``CamLiveModel.features``) models the outcome through a link:

    real            y        = phi^T w + eps,  eps ~ N(0, s2)      exact conjugate update (known noise, or the noise
                                                                     inferred jointly: normal-inverse-gamma, Student-t)
    count           y        ~ Poisson(exposure * exp(phi^T w))     assumed-density (Laplace) update
    binary          y        ~ Binomial(n, sigmoid(phi^T w))        assumed-density (Laplace) update
    time_to_event   t        ~ Exponential(rate exp(phi^T w)), right-censoring handled by the Poisson-exposure
                               identity: an event at t (delta=1) or censoring at t (delta=0) has log-likelihood
                               delta * eta - t * exp(eta), i.e. a Poisson count ``delta`` with exposure ``t``.

Everything is float64. Gaussian updates use the Joseph form (symmetric positive semi-definite by construction, also
after thousands of collinear updates). A GLM observation only moves the posterior along ``Sigma phi``, so its Laplace
approximation is found by a safeguarded Newton iteration on the scalar linear predictor and applied as a Joseph
update with pseudo-noise ``1 / h``.

Two forms are provided:

* **functional** (pure tensor functions, batched over leading dimensions, differentiable in every argument) — used to
  train the features and the learned prior ALPaCA-style through online updates (:func:`online_nll`);
* **stateful** :class:`FastBeliefState`, the numerical state ``m_t`` the agent carries, with stabilised forgetting
  (the posterior is flattened towards the learned prior, so unexcited directions never blow up) and a CUSUM change
  detector on standardised surprise.

The learned prior is :class:`FastPrior` (an ``nn.Module`` in the ``experiential`` parameter group).

Provenance: recursion and normal-inverse-gamma algebra follow ``superlearner/cam_unified/core/bayes.py`` and
``superlearner/cam7/posterior.py`` (commit 3968583), re-derived and extended here (GLM likelihoods, censoring,
stabilised forgetting, change detection, closed-form belief-change KL); no code is imported from those packages.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from ..contracts import OutcomeKind

F64 = torch.float64

#: outcome kinds the fast layer models (categorical outcomes are handled by the world-model head only)
FAST_KINDS: Tuple[str, ...] = (OutcomeKind.REAL, OutcomeKind.COUNT, OutcomeKind.BINARY, OutcomeKind.TIME_TO_EVENT)
GLM_KINDS: Tuple[str, ...] = (OutcomeKind.COUNT, OutcomeKind.BINARY, OutcomeKind.TIME_TO_EVENT)

_GH_N = 24
_gh_x, _gh_w = np.polynomial.hermite.hermgauss(_GH_N)
_GH_X = torch.tensor(_gh_x, dtype=F64)
_GH_LOGW = torch.tensor(np.log(_gh_w) - 0.5 * math.log(math.pi), dtype=F64)
_ETA_MAX = 40.0


def _f64(x: Any, like: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=F64, device=like.device)


def _sym(a: Tensor) -> Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _quad(cov: Tensor, phi: Tensor) -> Tuple[Tensor, Tensor]:
    """(Sigma phi [..., F], phi^T Sigma phi [...])."""
    pf = (cov @ phi.unsqueeze(-1)).squeeze(-1)
    return pf, (phi * pf).sum(-1).clamp_min(0.0)


def _joseph(cov: Tensor, phi: Tensor, gain: Tensor, rkk: Tensor) -> Tensor:
    """(I - K phi^T) Sigma (I - K phi^T)^T + rkk, with ``rkk`` the (already formed) R K K^T term."""
    eye = torch.eye(cov.shape[-1], dtype=F64, device=cov.device)
    a = eye - gain.unsqueeze(-1) * phi.unsqueeze(-2)
    return _sym(a @ cov @ a.transpose(-1, -2) + rkk)


# ------------------------------------------------------------------------------------------------------------------
# Gaussian: exact conjugate updates
# ------------------------------------------------------------------------------------------------------------------
def gaussian_update(mu: Tensor, cov: Tensor, phi: Tensor, y: Any, noise: Any
                    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """One exact update with known noise variance. mu [..., F], cov [..., F, F], phi [..., F], y [...], noise [...].

    Returns (mu', cov', info) with info = {surprise (predictive NLL, nats), kl (KL(posterior || prior), nats),
    z2 (squared standardised innovation), err}.
    """
    mu, cov, phi = mu.to(F64), cov.to(F64), phi.to(F64)
    y = _f64(y, mu)
    pf, s = _quad(cov, phi)
    noise = _f64(noise, mu).expand_as(s)
    S = s + noise
    gain = pf / S.unsqueeze(-1)
    err = y - (phi * mu).sum(-1)
    mu_n = mu + gain * err.unsqueeze(-1)
    cov_n = _joseph(cov, phi, gain, noise[..., None, None] * gain.unsqueeze(-1) * gain.unsqueeze(-2))
    z2 = err * err / S
    info = {"surprise": 0.5 * (torch.log(2 * math.pi * S) + z2),
            "kl": 0.5 * (-s / S + z2 * s / S + torch.log(S / noise)),
            "z2": z2, "err": err}
    return mu_n, cov_n, info


def nig_update(mu: Tensor, V: Tensor, a: Tensor, b: Tensor, phi: Tensor, y: Any
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """One exact normal-inverse-gamma update: w | s2 ~ N(mu, s2 V), s2 ~ InvGamma(a, b). Predictive is Student-t."""
    mu, V, phi = mu.to(F64), V.to(F64), phi.to(F64)
    a, b = _f64(a, mu), _f64(b, mu)
    y = _f64(y, mu)
    pf, q = _quad(V, phi)
    one_q = 1.0 + q
    gain = pf / one_q.unsqueeze(-1)
    err = y - (phi * mu).sum(-1)
    mu_n = mu + gain * err.unsqueeze(-1)
    V_n = _joseph(V, phi, gain, gain.unsqueeze(-1) * gain.unsqueeze(-2))
    a_n = a + 0.5
    b_n = b + 0.5 * err * err / one_q
    s2, s2_n = b / a, b_n / a_n
    scale2 = s2 * one_q
    F = mu.shape[-1]
    c2q = (err / one_q) ** 2 * q
    kl = 0.5 * ((s2_n / s2) * (F - q / one_q) - F + c2q / s2 + F * torch.log(s2 / s2_n) + torch.log(one_q))
    info = {"surprise": nll_student(y, (phi * mu).sum(-1), scale2, 2.0 * a), "kl": kl,
            "z2": err * err / scale2, "err": err}
    return mu_n, V_n, a_n, b_n, info


def _prec_from_cov(cov: Tensor) -> Tuple[Tensor, Tensor]:
    chol = torch.linalg.cholesky(_sym(cov))
    eye = torch.eye(cov.shape[-1], dtype=F64, device=cov.device).expand_as(cov)
    return torch.cholesky_solve(eye, chol), chol


def gaussian_fit(Phi: Tensor, y: Tensor, mask: Optional[Tensor], mu0: Tensor, cov0: Tensor, noise: Any
                 ) -> Tuple[Tensor, Tensor]:
    """Closed-form posterior after all rows at once (precision form through Cholesky factors; equals applying
    :func:`gaussian_update` row by row). Phi [..., T, F], y [..., T], mask [..., T] (1 = row present)."""
    Phi, y, mu0, cov0 = Phi.to(F64), y.to(F64), mu0.to(F64), cov0.to(F64)
    w = torch.ones_like(y) if mask is None else mask.to(F64)
    noise = _f64(noise, y)
    w = w / (noise.unsqueeze(-1) if noise.dim() > 0 else noise)
    lam0, _ = _prec_from_cov(cov0)
    lam = lam0 + torch.einsum("...tf,...t,...tg->...fg", Phi, w, Phi)
    rhs = (lam0 @ mu0.unsqueeze(-1)).squeeze(-1) + torch.einsum("...tf,...t->...f", Phi, w * y)
    chol = torch.linalg.cholesky(_sym(lam))
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    eye = torch.eye(lam.shape[-1], dtype=F64, device=lam.device).expand_as(lam)
    return mu, _sym(torch.cholesky_solve(eye, chol))


def nig_fit(Phi: Tensor, y: Tensor, mask: Optional[Tensor], mu0: Tensor, V0: Tensor, a0: Any, b0: Any
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closed-form normal-inverse-gamma posterior after all rows at once."""
    Phi, y, mu0, V0 = Phi.to(F64), y.to(F64), mu0.to(F64), V0.to(F64)
    w = torch.ones_like(y) if mask is None else mask.to(F64)
    lam0, _ = _prec_from_cov(V0)
    lam = lam0 + torch.einsum("...tf,...t,...tg->...fg", Phi, w, Phi)
    rhs = (lam0 @ mu0.unsqueeze(-1)).squeeze(-1) + torch.einsum("...tf,...t->...f", Phi, w * y)
    chol = torch.linalg.cholesky(_sym(lam))
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    eye = torch.eye(lam.shape[-1], dtype=F64, device=lam.device).expand_as(lam)
    V = _sym(torch.cholesky_solve(eye, chol))
    resid = y - torch.einsum("...tf,...f->...t", Phi, mu)
    dev = mu - mu0
    reg = (dev.unsqueeze(-2) @ lam0 @ dev.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    a = _f64(a0, y) + 0.5 * w.sum(-1)
    b = _f64(b0, y) + 0.5 * ((w * resid * resid).sum(-1) + reg)
    return mu, V, a, b


def predict_gaussian(mu: Tensor, cov: Tensor, phi: Tensor, noise: Any) -> Tuple[Tensor, Tensor, Tensor]:
    """(mean, epistemic variance phi^T Sigma phi, noise variance) at phi [..., F] (mu/cov broadcast)."""
    phi = phi.to(F64)
    mean = (phi * mu).sum(-1)
    epi = torch.einsum("...f,...fg,...g->...", phi, cov.to(F64), phi).clamp_min(0.0)
    return mean, epi, _f64(noise, mean).expand_as(mean)


def predict_nig(mu: Tensor, V: Tensor, a: Any, b: Any, phi: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Student-t predictive (loc, squared scale, degrees of freedom)."""
    phi = phi.to(F64)
    loc = (phi * mu).sum(-1)
    q = torch.einsum("...f,...fg,...g->...", phi, V.to(F64), phi).clamp_min(0.0)
    a, b = _f64(a, loc), _f64(b, loc)
    return loc, (b / a) * (1.0 + q), (2.0 * a).expand_as(loc)


def nll_gaussian(y: Any, mean: Tensor, var: Tensor) -> Tensor:
    y = _f64(y, mean)
    return 0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)


def nll_student(y: Any, loc: Tensor, scale2: Tensor, dof: Any) -> Tensor:
    y = _f64(y, loc)
    nu = _f64(dof, loc)
    z2 = (y - loc) ** 2 / scale2
    return (-torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2) + 0.5 * torch.log(nu * math.pi * scale2)
            + (nu + 1) / 2 * torch.log1p(z2 / nu))


# ------------------------------------------------------------------------------------------------------------------
# GLM likelihoods: assumed-density (Laplace) updates
# ------------------------------------------------------------------------------------------------------------------
def _glm_terms(kind: str, y: Tensor, exposure: Optional[Any], censored: Optional[Any]
               ) -> Tuple[str, Tensor, Tensor, Tensor]:
    """-> (family, c, e, scale): Poisson family: count ``c`` with exposure ``e``; binomial: ``c`` successes of ``e``
    trials. ``scale`` is the time unit of a time-to-event outcome (1 otherwise)."""
    one = torch.ones_like(y)
    if kind == OutcomeKind.COUNT:
        e = one if exposure is None else _f64(exposure, y).expand_as(y)
        return "poisson", y, e.clamp_min(0.0), one
    if kind == OutcomeKind.TIME_TO_EVENT:
        scale = one if exposure is None else _f64(exposure, y).expand_as(y).clamp_min(1e-300)
        delta = one if censored is None else 1.0 - _f64(censored, y).expand_as(y)
        return "poisson", delta, (y / scale).clamp_min(0.0), scale
    if kind == OutcomeKind.BINARY:
        n = one if exposure is None else _f64(exposure, y).expand_as(y)
        return "binomial", y, n.clamp_min(0.0), one
    raise ValueError(f"outcome kind {kind!r} has no GLM fast update (supported: {GLM_KINDS})")


def _glm_grad_hess(family: str, eta: Tensor, c: Tensor, e: Tensor) -> Tuple[Tensor, Tensor]:
    """(d loglik / d eta, -d2 loglik / d eta2 >= 0)."""
    if family == "poisson":
        m = e * torch.exp(eta.clamp(max=_ETA_MAX))
        return c - m, m
    p = torch.sigmoid(eta)
    return c - e * p, e * p * (1.0 - p)


def _glm_loglik(family: str, eta: Tensor, c: Tensor, e: Tensor, kind: str) -> Tensor:
    """Normalised log pmf / pdf at linear predictor ``eta`` (broadcasts over a trailing quadrature dimension)."""
    if family == "poisson":
        ll = c * eta - e * torch.exp(eta.clamp(max=_ETA_MAX))
        if kind == OutcomeKind.COUNT:
            ll = ll + c * torch.log(e.clamp_min(1e-300)) - torch.lgamma(c + 1.0)
        return ll
    return (c * eta - e * Fn.softplus(eta)
            + torch.lgamma(e + 1.0) - torch.lgamma(c + 1.0) - torch.lgamma((e - c).clamp_min(0.0) + 1.0))


def _gh_eta(eta0: Tensor, s: Tensor) -> Tensor:
    return eta0.unsqueeze(-1) + torch.sqrt(2.0 * s.clamp_min(0.0)).unsqueeze(-1) * _GH_X.to(eta0.device)


def _laplace_1d(family: str, eta0: Tensor, s_safe: Tensor, c: Tensor, e: Tensor, disp: Tensor, n_newton: int
                ) -> Tuple[Tensor, Tensor]:
    """Mode eta* of loglik(eta) / disp - (eta - eta0)^2 / (2 s) by safeguarded Newton steps, and the curvature
    h = -loglik''(eta*) / disp."""
    eta = eta0
    for _ in range(int(n_newton)):
        g, h = _glm_grad_hess(family, eta, c, e)
        step = (g / disp - (eta - eta0) / s_safe) / (h / disp + 1.0 / s_safe)
        eta = eta + step.clamp(-3.0, 3.0)
    _, h = _glm_grad_hess(family, eta, c, e)
    return eta, h / disp


def _adaptive_gh_nll(family: str, kind: str, eta0: Tensor, s_safe: Tensor, eta_star: Tensor, h: Tensor, c: Tensor,
                     e: Tensor) -> Tensor:
    """-log of the integral of p(y | eta) N(eta; eta0, s) over eta by Gauss-Hermite quadrature centred on the
    Laplace approximation N(eta*, s / (1 + h s)) of the integrand (accurate also when the likelihood is much
    sharper than the prior, where quadrature centred on the prior fails)."""
    s_star = s_safe / (1.0 + h * s_safe)
    x = _GH_X.to(eta0.device)
    eta = eta_star.unsqueeze(-1) + torch.sqrt(2.0 * s_star).unsqueeze(-1) * x
    ll = _glm_loglik(family, eta, c.unsqueeze(-1), e.unsqueeze(-1), kind)
    log_prior = -0.5 * torch.log(2 * math.pi * s_safe).unsqueeze(-1) - (eta - eta0.unsqueeze(-1)) ** 2 / (2.0 * s_safe.unsqueeze(-1))
    logw = _GH_LOGW.to(eta0.device) + 0.5 * math.log(math.pi) + x * x
    return -(0.5 * torch.log(2.0 * s_star) + torch.logsumexp(logw + ll + log_prior, -1))


def nll_glm(mu: Tensor, cov: Tensor, phi: Tensor, y: Any, kind: str, exposure: Optional[Any] = None,
            censored: Optional[Any] = None, n_newton: int = 8) -> Tensor:
    """Predictive negative log-likelihood -log E_{w ~ N(mu, Sigma)} p(y | phi^T w) by adaptive Gauss-Hermite
    quadrature. For time-to-event the density is taken w.r.t. time in units of ``exposure`` (survival probability
    if censored)."""
    mu, cov, phi = mu.to(F64), cov.to(F64), phi.to(F64)
    eta0 = (phi * mu).sum(-1)
    y = _f64(y, eta0).expand_as(eta0)
    _, s = _quad(cov, phi)
    s_safe = s.clamp_min(1e-12)
    family, c, e, _ = _glm_terms(kind, y, exposure, censored)
    eta, h = _laplace_1d(family, eta0, s_safe, c, e, torch.ones_like(eta0), n_newton)
    return _adaptive_gh_nll(family, kind, eta0, s_safe, eta, h, c, e)


def glm_update(mu: Tensor, cov: Tensor, phi: Tensor, y: Any, kind: str, exposure: Optional[Any] = None,
               censored: Optional[Any] = None, dispersion: Any = 1.0, n_newton: int = 8
               ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """One assumed-density update for a count / binary / time-to-event observation.

    The exact posterior mode lies on the line mu + Sigma phi * c, so the Laplace approximation is found on the scalar
    linear predictor eta = phi^T w: maximise loglik(eta) / dispersion - (eta - eta0)^2 / (2 s), s = phi^T Sigma phi,
    by ``n_newton`` safeguarded Newton steps (concave problem), then applied as a Joseph update with curvature
    h = -loglik''(eta*) / dispersion. ``dispersion`` >= 1 tempers the likelihood (quasi-likelihood for over-dispersed
    counts). Differentiable through the unrolled iterations.
    """
    mu, cov, phi = mu.to(F64), cov.to(F64), phi.to(F64)
    eta0 = (phi * mu).sum(-1)
    y = _f64(y, eta0).expand_as(eta0)
    pf, s = _quad(cov, phi)
    s_safe = s.clamp_min(1e-12)
    disp = _f64(dispersion, eta0).expand_as(eta0)
    family, c, e, _ = _glm_terms(kind, y, exposure, censored)
    eta, h = _laplace_1d(family, eta0, s_safe, c, e, disp, n_newton)
    delta = eta - eta0
    denom = 1.0 + h * s
    gain = pf * (h / denom).unsqueeze(-1)
    mu_n = mu + pf * (delta / s_safe).unsqueeze(-1)
    cov_n = _joseph(cov, phi, gain, (h / denom ** 2)[..., None, None] * pf.unsqueeze(-1) * pf.unsqueeze(-2))
    # Pearson-type standardised residual from the predictive moments (for change detection)
    if family == "poisson":
        m = e * torch.exp((eta0 + 0.5 * s).clamp(max=_ETA_MAX))
        var = disp * m + m * m * torch.expm1(s.clamp(max=_ETA_MAX))
    else:
        p = torch.sigmoid(eta0 / torch.sqrt(1.0 + math.pi * s / 8.0))
        m = e * p
        var = e * p * (1.0 - p)
    z2 = (c - m) ** 2 / var.clamp_min(1e-12)
    info = {"surprise": _adaptive_gh_nll(family, kind, eta0, s_safe, eta, h * disp, c, e),
            "kl": 0.5 * (-(h * s) / denom + delta * delta / s_safe + torch.log(denom)),
            "z2": z2, "err": c - m, "eta": eta}
    return mu_n, cov_n, info


def predict_glm(mu: Tensor, cov: Tensor, phi: Tensor, kind: str, exposure: Optional[Any] = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
    """(predictive mean of the outcome, epistemic variance of that mean, aleatoric variance) in outcome units:
    expected count for ``exposure``; success probability times ``exposure`` trials; expected time to the event in
    units of ``exposure``."""
    phi = phi.to(F64)
    eta0 = (phi * mu.to(F64)).sum(-1)
    s = torch.einsum("...f,...fg,...g->...", phi, cov.to(F64), phi).clamp_min(0.0)
    e = torch.ones_like(eta0) if exposure is None else _f64(exposure, eta0).expand_as(eta0)
    sc = s.clamp(max=_ETA_MAX)
    if kind == OutcomeKind.COUNT:
        m = e * torch.exp((eta0 + 0.5 * sc).clamp(max=_ETA_MAX))
        return m, m * m * torch.expm1(sc), m
    if kind == OutcomeKind.TIME_TO_EVENT:
        m = e * torch.exp((-eta0 + 0.5 * sc).clamp(max=_ETA_MAX))
        return m, m * m * torch.expm1(sc), m * m * torch.exp(sc)
    if kind == OutcomeKind.BINARY:
        p = torch.sigmoid(_gh_eta(eta0, s))
        w = torch.exp(_GH_LOGW.to(p.device))
        pm = (p * w).sum(-1)
        epi = ((p * p * w).sum(-1) - pm * pm).clamp_min(0.0)
        return e * pm, e * e * epi, e * (pm * (1.0 - pm) - epi).clamp_min(0.0)
    raise ValueError(f"outcome kind {kind!r} has no GLM fast prediction (supported: {GLM_KINDS})")


# ------------------------------------------------------------------------------------------------------------------
# Changing mechanisms
# ------------------------------------------------------------------------------------------------------------------
def forget(mu: Tensor, cov: Tensor, mu0: Tensor, cov0: Tensor, gamma: Any) -> Tuple[Tensor, Tensor]:
    """Stabilised exponential forgetting: p <- p^gamma * prior^(1 - gamma) (geometric mixture of Gaussians), i.e.
    precision Lambda <- gamma Lambda + (1 - gamma) Lambda0 and the same for the natural mean. gamma = 1 is a no-op;
    gamma = 0 returns the prior. Unlike covariance inflation it is bounded by the prior in every direction."""
    lam, _ = _prec_from_cov(cov.to(F64))
    lam0, _ = _prec_from_cov(cov0.to(F64))
    g = _f64(gamma, mu)
    g2, g1 = g[..., None, None] if g.dim() else g, g[..., None] if g.dim() else g
    lam_n = g2 * lam + (1.0 - g2) * lam0
    eta = g1 * (lam @ mu.to(F64).unsqueeze(-1)).squeeze(-1) + (1.0 - g1) * (lam0 @ mu0.to(F64).unsqueeze(-1)).squeeze(-1)
    chol = torch.linalg.cholesky(_sym(lam_n))
    eye = torch.eye(lam_n.shape[-1], dtype=F64, device=lam_n.device).expand_as(lam_n)
    return torch.cholesky_solve(eta.unsqueeze(-1), chol).squeeze(-1), _sym(torch.cholesky_solve(eye, chol))


# ------------------------------------------------------------------------------------------------------------------
# The learned prior
# ------------------------------------------------------------------------------------------------------------------
def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class FastPrior(nn.Module):
    """Learned prior of the Bayesian fast layer: per outcome kind a mean ``mu0`` and a full covariance
    ``L0 L0^T`` (Cholesky-parameterised, positive diagonal), the Gaussian noise variance with its inverse-gamma
    shape, and the count dispersion. Trained through :func:`online_nll` (ALPaCA-style). ``experiential`` group."""

    def __init__(self, n_features: int, *, prior_var: float = 1.0, noise: float = 0.5, a0: float = 2.0,
                 noise_mode: str = "nig", kinds: Sequence[str] = FAST_KINDS):
        super().__init__()
        if noise_mode not in ("nig", "fixed"):
            raise ValueError("noise_mode must be 'nig' or 'fixed'")
        self.n_features = int(n_features)
        self.kinds = tuple(kinds)
        self.noise_mode = noise_mode
        K, F = len(self.kinds), self.n_features
        self.mu0 = nn.Parameter(torch.zeros(K, F))
        self.chol_diag_raw = nn.Parameter(torch.full((K, F), _inv_softplus(math.sqrt(prior_var))))
        self.chol_lower = nn.Parameter(torch.zeros(K, F * (F - 1) // 2))
        self.log_noise = nn.Parameter(torch.tensor(math.log(noise)))
        self.a0_raw = nn.Parameter(torch.tensor(_inv_softplus(a0 - 1.0)))
        self.disp_raw = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("_tril", torch.tril_indices(F, F, -1), persistent=False)

    def index(self, kind: str) -> int:
        if kind not in self.kinds:
            raise ValueError(f"outcome kind {kind!r} is not modelled by the fast layer (supported: {self.kinds})")
        return self.kinds.index(kind)

    def chol(self, kind: str) -> Tensor:
        k = self.index(kind)
        F = self.n_features
        L = torch.zeros(F, F, dtype=F64, device=self.mu0.device)
        L = L.index_put((self._tril[0], self._tril[1]), self.chol_lower[k].to(F64))
        return L + torch.diag(Fn.softplus(self.chol_diag_raw[k].to(F64)) + 1e-6)

    def noise(self) -> Tensor:
        return self.log_noise.to(F64).exp()

    def a0(self) -> Tensor:
        return 1.0 + Fn.softplus(self.a0_raw.to(F64))

    def dispersion(self) -> Tensor:
        return 1.0 + Fn.softplus(self.disp_raw.to(F64))

    def prior(self, kind: str) -> Dict[str, Tensor]:
        """Differentiable float64 prior for ``kind``: ``mu`` [F], ``cov`` [F, F] (weights covariance; for the
        normal-inverse-gamma mode the scale-free ``V = cov / noise`` is under ``"V"`` with ``a``, ``b``),
        ``noise``, ``dispersion``."""
        L = self.chol(kind)
        cov = L @ L.T
        noise, a0 = self.noise(), self.a0()
        return {"mu": self.mu0[self.index(kind)].to(F64), "cov": cov, "V": cov / noise, "a": a0, "b": a0 * noise,
                "noise": noise, "dispersion": self.dispersion()}


def online_nll(prior: Dict[str, Tensor], phi: Tensor, y: Tensor, kind: str, *, mask: Optional[Tensor] = None,
               exposure: Optional[Tensor] = None, censored: Optional[Tensor] = None, noise_mode: str = "nig",
               n_newton: int = 6, update_mask: Optional[Tensor] = None) -> Tensor:
    """Prequential (predict, then update) negative log-likelihood of a sequence — the ALPaCA training signal.

    phi [..., T, F], y [..., T]; ``mask`` [..., T] marks rows that exist; ``update_mask`` (default = mask) marks rows
    the posterior may learn from (set it to the support rows to score later held-out rows without learning from
    them). Returns per-row NLL [..., T] (0 where masked), differentiable w.r.t. ``phi`` and every prior tensor.
    """
    phi, y = phi.to(F64), y.to(F64)
    lead, T = phi.shape[:-2], phi.shape[-2]
    Fd = phi.shape[-1]
    mask = torch.ones_like(y, dtype=torch.bool) if mask is None else mask.bool()
    upd = mask if update_mask is None else (update_mask.bool() & mask)
    mu = prior["mu"].expand(*lead, Fd)
    gauss = kind == OutcomeKind.REAL
    nig = gauss and noise_mode == "nig"
    cov = (prior["V"] if nig else prior["cov"]).expand(*lead, Fd, Fd)
    a = prior["a"].expand(lead) if nig else None
    b = prior["b"].expand(lead) if nig else None
    out = []
    for t in range(T):
        f, yt = phi[..., t, :], y[..., t]
        yt = torch.where(mask[..., t], yt, torch.zeros_like(yt))
        ex = None if exposure is None else exposure[..., t]
        ce = None if censored is None else censored[..., t]
        if nig:
            mu_n, cov_n, a_n, b_n, info = nig_update(mu, cov, a, b, f, yt)
        elif gauss:
            mu_n, cov_n, info = gaussian_update(mu, cov, f, yt, prior["noise"])
        else:
            mu_n, cov_n, info = glm_update(mu, cov, f, yt, kind, ex, ce, prior["dispersion"], n_newton)
        out.append(torch.where(mask[..., t], info["surprise"], torch.zeros_like(info["surprise"])))
        u = upd[..., t]
        mu = torch.where(u[..., None], mu_n, mu)
        cov = torch.where(u[..., None, None], cov_n, cov)
        if nig:
            a, b = torch.where(u, a_n, a), torch.where(u, b_n, b)
    return torch.stack(out, -1)


# ------------------------------------------------------------------------------------------------------------------
# The stateful fast belief state
# ------------------------------------------------------------------------------------------------------------------
class FastBeliefState:
    """The numerical fast state ``m_t``: one last-layer posterior per outcome kind, updated immediately on evidence.

    Args:
        prior: the learned :class:`FastPrior` (snapshotted, detached, at construction and at :meth:`reset`).
        forgetting: per-update stabilised forgetting factor gamma in (0, 1]; 1 disables it.
        change_k / change_h / change_clip: CUSUM on the squared standardised surprise ``z2`` — the statistic
            ``c <- max(0, c + min(z2, clip) - 1 - k)`` raises a change when ``c > h`` (``h <= 0`` disables it).
        change_forget: gamma applied once when a change is detected (0.1 = keep 10 % of the evidence).
        n_newton: Newton iterations of the GLM updates.
    """

    def __init__(self, prior: FastPrior, *, forgetting: float = 1.0, change_k: float = 1.0, change_h: float = 12.0,
                 change_clip: float = 9.0, change_forget: float = 0.1, n_newton: int = 12,
                 device: Optional[torch.device] = None):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        self.kinds = tuple(prior.kinds)
        self.n_features = prior.n_features
        self.noise_mode = prior.noise_mode
        self.forgetting = float(forgetting)
        self.change_k, self.change_h = float(change_k), float(change_h)
        self.change_clip, self.change_forget = float(change_clip), float(change_forget)
        self.n_newton = int(n_newton)
        self.device = torch.device(device or "cpu")
        self._prior: Dict[str, Dict[str, Tensor]] = {}
        self._post: Dict[str, Dict[str, Any]] = {}
        self.reset(prior=prior)

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    def supports(self, kind: str) -> bool:
        return kind in self.kinds

    def reset(self, kind: Optional[str] = None, prior: Optional[FastPrior] = None) -> None:
        """Back to the prior (all kinds, or one). With ``prior`` given, re-snapshot the learned prior first."""
        if prior is not None:
            with torch.no_grad():
                self._prior = {k: {n: v.detach().to(self.device, F64).clone() for n, v in prior.prior(k).items()}
                               for k in self.kinds}
        for k in (self.kinds if kind is None else (kind,)):
            self._check(k)
            p = self._prior[k]
            nig = self._nig(k)
            self._post[k] = {"mu": p["mu"].clone(), "cov": (p["V"] if nig else p["cov"]).clone(),
                             "a": p["a"].clone(), "b": p["b"].clone(), "n": 0, "cusum": 0.0, "n_changes": 0,
                             "last_change_n": -1}

    def clone(self) -> "FastBeliefState":
        new = copy.copy(self)
        new._prior = {k: {n: v.clone() for n, v in p.items()} for k, p in self._prior.items()}
        new._post = {k: {n: (v.clone() if isinstance(v, Tensor) else v) for n, v in p.items()}
                     for k, p in self._post.items()}
        return new

    def state_dict(self) -> Dict[str, Any]:
        return {"version": 1, "n_features": self.n_features, "noise_mode": self.noise_mode,
                "config": {"forgetting": self.forgetting, "change_k": self.change_k, "change_h": self.change_h,
                           "change_clip": self.change_clip, "change_forget": self.change_forget,
                           "n_newton": self.n_newton},
                "prior": {k: {n: v.clone() for n, v in p.items()} for k, p in self._prior.items()},
                "post": {k: {n: (v.clone() if isinstance(v, Tensor) else v) for n, v in p.items()}
                         for k, p in self._post.items()}}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if int(sd["n_features"]) != self.n_features or sd["noise_mode"] != self.noise_mode:
            raise ValueError("fast state does not match this model (n_features / noise_mode)")
        for name, v in sd["config"].items():
            setattr(self, name, v)
        self.kinds = tuple(sd["post"].keys())
        self._prior = {k: {n: v.to(self.device, F64).clone() for n, v in p.items()} for k, p in sd["prior"].items()}
        self._post = {k: {n: (v.to(self.device, F64).clone() if isinstance(v, Tensor) else v) for n, v in p.items()}
                      for k, p in sd["post"].items()}

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _check(self, kind: str) -> None:
        if kind not in self.kinds:
            raise ValueError(f"outcome kind {kind!r} is not modelled by the fast layer (supported: {self.kinds})")

    def _nig(self, kind: str) -> bool:
        return kind == OutcomeKind.REAL and self.noise_mode == "nig"

    def _phi(self, phi: Tensor) -> Tensor:
        phi = phi.detach().to(self.device, F64)
        if phi.shape[-1] != self.n_features:
            raise ValueError(f"phi has {phi.shape[-1]} features, the fast state has {self.n_features}")
        return phi

    def posterior(self, kind: str) -> Dict[str, Any]:
        """Current posterior: ``mu`` [F], ``cov`` [F, F] (covariance of the weights; for the inferred-noise Gaussian
        this is ``(b / a) V``), ``n`` updates so far, ``noise`` plug-in noise variance, change-detector state."""
        self._check(kind)
        s = self._post[kind]
        nig = self._nig(kind)
        noise = (s["b"] / s["a"]) if nig else self._prior[kind]["noise"]
        return {"mu": s["mu"].clone(), "cov": (s["cov"] * noise if nig else s["cov"]).clone(), "n": s["n"],
                "noise": noise.clone(), "a": s["a"].clone(), "b": s["b"].clone(), "cusum": s["cusum"],
                "n_changes": s["n_changes"], "last_change_n": s["last_change_n"]}

    def n_updates(self, kind: str) -> int:
        self._check(kind)
        return int(self._post[kind]["n"])

    def linear(self, phi: Tensor, kind: str) -> Tuple[Tensor, Tensor]:
        """(mean, variance) of the linear predictor phi^T w under the current posterior, shaped like phi[..., 0]."""
        post = self.posterior(kind)
        f = self._phi(phi)
        mean = (f * post["mu"]).sum(-1)
        var = torch.einsum("...f,fg,...g->...", f, post["cov"], f).clamp_min(0.0)
        return mean.to(phi.device, phi.dtype), var.to(phi.device, phi.dtype)

    def kl_from_prior(self, kind: str) -> float:
        """KL(current weight posterior || prior) in nats: total information absorbed since the last reset."""
        post, p = self.posterior(kind), self._prior[kind]
        lam0, chol0 = _prec_from_cov(p["cov"])
        chol = torch.linalg.cholesky(_sym(post["cov"]) + 1e-300 * torch.eye(self.n_features, dtype=F64))
        d = post["mu"] - p["mu"]
        logdet0 = 2.0 * torch.log(torch.diagonal(chol0)).sum()
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        return float(0.5 * ((lam0 * post["cov"]).sum() - self.n_features + d @ lam0 @ d + logdet0 - logdet))

    # ---- prediction ---------------------------------------------------------------------------------------------
    def predict(self, phi: Tensor, kind: str, exposure: Optional[Any] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """(mean, epistemic variance, noise) for features phi [..., F], each shaped like phi[..., 0], on phi's device
        and dtype. ``mean`` is in outcome units (see :func:`predict_glm` for the non-Gaussian kinds); ``epistemic
        variance`` is the part that more evidence removes; ``noise`` is the aleatoric variance of one observation
        (real: the noise variance — plug-in ``b / a`` when inferred; count: the dispersion-scaled Poisson variance;
        binary: p(1-p); time-to-event: the exponential variance)."""
        self._check(kind)
        f = self._phi(phi)
        s = self._post[kind]
        if kind == OutcomeKind.REAL:
            if self._nig(kind):
                loc, scale2, _ = predict_nig(s["mu"], s["cov"], s["a"], s["b"], f)
                noise = (s["b"] / s["a"]).expand_as(loc)
                out = (loc, (scale2 - noise).clamp_min(0.0), noise)
            else:
                out = predict_gaussian(s["mu"], s["cov"], f, self._prior[kind]["noise"])
        else:
            ex = None if exposure is None else torch.as_tensor(exposure, dtype=F64, device=self.device)
            mean, epi, alea = predict_glm(s["mu"], s["cov"], f, kind, ex)
            if kind == OutcomeKind.COUNT:
                alea = alea * self._prior[kind]["dispersion"]
            out = (mean, epi, alea)
        return tuple(o.to(phi.device, phi.dtype) for o in out)   # type: ignore[return-value]

    def nll(self, phi: Tensor, y: Any, kind: str, exposure: Optional[Any] = None,
            censored: Optional[Any] = None) -> Tensor:
        """Predictive negative log-likelihood of ``y`` under the current posterior (no update)."""
        self._check(kind)
        f = self._phi(phi)
        s = self._post[kind]
        yt = torch.as_tensor(y, dtype=F64, device=self.device)
        if kind == OutcomeKind.REAL:
            if self._nig(kind):
                loc, scale2, dof = predict_nig(s["mu"], s["cov"], s["a"], s["b"], f)
                out = nll_student(yt, loc, scale2, dof)
            else:
                mean, epi, noise = predict_gaussian(s["mu"], s["cov"], f, self._prior[kind]["noise"])
                out = nll_gaussian(yt, mean, epi + noise)
        else:
            out = nll_glm(s["mu"], s["cov"], f, yt, kind, exposure, censored)
        return out.to(phi.device, phi.dtype)

    def sample_weights(self, n: int, kind: str, generator: Optional[torch.Generator] = None) -> Tensor:
        """[n, F] float64 posterior draws of the last-layer weights (Thompson sampling). For the inferred-noise
        Gaussian the noise variance is drawn jointly (so the marginal is the multivariate Student-t)."""
        self._check(kind)
        s = self._post[kind]
        cov = _sym(s["cov"])
        jitter = 1e-12 * float(torch.diagonal(cov).mean().clamp_min(1e-30))
        L = torch.linalg.cholesky(cov + jitter * torch.eye(self.n_features, dtype=F64, device=self.device))
        z = torch.randn(n, self.n_features, dtype=F64, device=self.device, generator=generator)
        w = z @ L.T
        if self._nig(kind):
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            g = np.random.default_rng(seed).standard_gamma(float(s["a"]), size=n)
            s2 = float(s["b"]) / torch.tensor(np.maximum(g, 1e-300), dtype=F64, device=self.device)
            w = w * s2.sqrt().unsqueeze(-1)
        return s["mu"] + w

    # ---- learning ------------------------------------------------------------------------------------------------
    def _flatten(self, kind: str, gamma: float) -> None:
        s, p = self._post[kind], self._prior[kind]
        nig = self._nig(kind)
        s["mu"], s["cov"] = forget(s["mu"], s["cov"], p["mu"], p["V"] if nig else p["cov"], gamma)
        if nig:
            s["a"] = gamma * s["a"] + (1.0 - gamma) * p["a"]
            s["b"] = gamma * s["b"] + (1.0 - gamma) * p["b"]

    def _apply(self, kind: str, f: Tensor, y: Tensor, ex: Optional[Tensor], ce: Optional[Tensor]
               ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """The posterior after one observation (state untouched) and the update's info."""
        s, p = self._post[kind], self._prior[kind]
        if kind == OutcomeKind.REAL:
            if self._nig(kind):
                mu, cov, a, b, info = nig_update(s["mu"], s["cov"], s["a"], s["b"], f, y)
                return {"mu": mu, "cov": cov, "a": a, "b": b}, info
            mu, cov, info = gaussian_update(s["mu"], s["cov"], f, y, p["noise"])
        else:
            mu, cov, info = glm_update(s["mu"], s["cov"], f, y, kind, ex, ce, p["dispersion"], self.n_newton)
        return {"mu": mu, "cov": cov}, info

    def update(self, phi: Tensor, y: Any, kind: str, exposure: Optional[Any] = None,
               censored: Optional[Any] = None) -> Dict[str, Any]:
        """Absorb observation(s): phi [F] with scalar y, or phi [M, F] with y [M] (applied in order).

        ``exposure``: counts — the exposure the count refers to; binary — number of trials (y = successes);
        time-to-event — the time unit ``y`` is expressed in. ``censored`` (time-to-event): True where the event
        had not happened by time ``y`` (right-censoring). Returns {"surprise": predictive NLL summed over the
        observations (nats), "kl": belief change KL(posterior || previous posterior) summed (nats), "z2": last
        squared standardised residual, "change_detected": bool, "n": updates of this kind so far, and the per-
        observation lists "surprises" / "kls"}.
        """
        self._check(kind)
        f = self._phi(phi).reshape(-1, self.n_features)
        M = f.shape[0]
        yt = torch.as_tensor(y, dtype=F64, device=self.device).reshape(-1).expand(M)
        ex = None if exposure is None else torch.as_tensor(exposure, dtype=F64, device=self.device).reshape(-1).expand(M)
        ce = None if censored is None else torch.as_tensor(censored, dtype=F64, device=self.device).reshape(-1).expand(M)
        if not torch.isfinite(yt).all() or not torch.isfinite(f).all():
            raise ValueError("fast update needs finite features and outcomes (mask unobservable outcomes instead)")
        s, p = self._post[kind], self._prior[kind]
        surprises: List[float] = []
        kls: List[float] = []
        changed, z2 = False, 0.0
        for i in range(M):
            fi, yi = f[i], yt[i]
            exi = None if ex is None else ex[i]
            cei = None if ce is None else ce[i]
            if self.forgetting < 1.0 and s["n"] > 0:
                self._flatten(kind, self.forgetting)
            new, info = self._apply(kind, fi, yi, exi, cei)
            z2 = float(info["z2"])                       # standardised surprise under the pre-update belief
            if self.change_h > 0:
                s["cusum"] = max(0.0, s["cusum"] + min(z2, self.change_clip) - 1.0 - self.change_k)
                if s["cusum"] > self.change_h:           # mechanism change: flatten first, then absorb the datum
                    self._flatten(kind, self.change_forget)
                    s["cusum"] = 0.0
                    s["n_changes"] += 1
                    s["last_change_n"] = s["n"]
                    changed = True
                    new, info = self._apply(kind, fi, yi, exi, cei)
            s.update(new)
            s["n"] += 1
            surprises.append(float(info["surprise"]))
            kls.append(float(info["kl"]))
        out = {"surprise": float(sum(surprises)), "kl": float(sum(kls)), "z2": z2, "change_detected": changed,
               "n": int(s["n"]), "surprises": surprises, "kls": kls}
        return out
