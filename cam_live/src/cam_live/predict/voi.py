"""cam_live.predict.voi — decision-relevant value of information (spec Part 8).

    VOI(a) = E_y[ max_a' Q(b_updated(a, y), a') ] - max_a' Q(b, a')

Information is valued by how much it is expected to improve the *next decision*, never by how surprising it is.
An unpredictable observation that cannot change which candidate is best has VOI exactly 0, however large its
prediction error (the "noisy TV").

Computed estimators (the reference):

* :func:`voi_linear_gaussian` — Bayesian linear-Gaussian last layer: the posterior mean of every decision candidate
  moves linearly in one standard normal variable, so the expectation of the maximum is evaluated *exactly*
  (:func:`expected_max_affine`, Frazier, Powell and Dayanik 2009).
* :func:`knowledge_gradient_gaussian` — closed form for independent arms (Frazier et al. 2008), used as a check.
* :func:`voi_glm` — Bernoulli / Poisson outcomes: outcomes are sampled from the posterior predictive, the
  assumed-density update of :class:`GLMPosterior` is applied, and the improvement of the best candidate averaged.
* :func:`voi_sampling` — the same definition for arbitrary callables (world model + fast-state update).

:class:`VOICosts` nets out resource cost, elapsed time (discounting), opportunity cost and the action's own direct
effect. :class:`DownstreamValueEstimator` is a small learned predictor of *realised* downstream usefulness;
:func:`score_downstream` compares it with what actually happened and with the computed VOI, and the learned
estimator is preferred only when it demonstrably ranks better (learned exploration never beat computed VOI on new
worlds in earlier projects).

``voi_sampling`` and ``knowledge_gradient_gaussian`` are adapted from
``superlearner/cam_unified/selfeval/voi.py`` (agentrs commit 3968583).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import dists as D

F64 = torch.float64
Num = Union[Tensor, float]


def _f64(x: Any) -> Tensor:
    return torch.as_tensor(x, dtype=F64) if not isinstance(x, Tensor) else x.detach().to(F64)


# ----------------------------------------------------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class VOICosts:
    """What acquiring the information costs, in the same units as Q.

    net = exp(-discount_rate * elapsed_s) * E[max Q after] - max Q before
          + direct_value - cost - opportunity_rate * elapsed_s
    """
    cost: Num = 0.0                  # resources spent by the action
    elapsed_s: Num = 0.0             # how long the action takes before the next decision can use its result
    opportunity_rate: float = 0.0    # value forgone per second of not exploiting the current best option
    discount_rate: float = 0.0       # per-second discount of value obtained later
    direct_value: Num = 0.0          # expected direct effect of the action on the objective (not information)

    def net(self, after: Tensor, base: Tensor) -> Tensor:
        t = _f64(self.elapsed_s)
        return (torch.exp(-self.discount_rate * t) * after - base + _f64(self.direct_value) - _f64(self.cost)
                - self.opportunity_rate * t)


# ----------------------------------------------------------------------------------------------------------------
# exact E[max of affine functions of a standard normal]
# ----------------------------------------------------------------------------------------------------------------
def expected_max_affine(a: Any, b: Any) -> float:
    """E[max_i (a_i + b_i Z)], Z ~ N(0, 1), exactly: the upper envelope of the lines is found by a sort and a
    stack, then each segment integrates in closed form."""
    a = np.asarray(_f64(a).cpu().numpy(), dtype=np.float64).reshape(-1)
    b = np.asarray(_f64(b).cpu().numpy(), dtype=np.float64).reshape(-1)
    order = np.lexsort((a, b))                                    # by slope, then intercept
    a, b = a[order], b[order]
    keep = np.ones(len(a), dtype=bool)
    keep[:-1] = b[1:] != b[:-1]                                   # equal slopes: keep the largest intercept
    a, b = a[keep], b[keep]
    idx: List[int] = []                                           # lines on the envelope, by increasing slope
    cut: List[float] = []                                         # cut[j]: where idx[j] takes over from idx[j-1]
    for i in range(len(a)):
        while idx:
            j = idx[-1]
            c = (a[j] - a[i]) / (b[i] - b[j])
            if cut and c <= cut[-1]:
                idx.pop()
                cut.pop()
            else:
                break
        if idx:
            cut.append((a[idx[-1]] - a[i]) / (b[i] - b[idx[-1]]))
        else:
            cut.append(-math.inf)
        idx.append(i)
    edges = cut + [math.inf]
    total = 0.0
    for j, i in enumerate(idx):
        lo, hi = edges[j], edges[j + 1]
        cdf_lo = 0.0 if lo == -math.inf else 0.5 * math.erfc(-lo / math.sqrt(2))
        cdf_hi = 1.0 if hi == math.inf else 0.5 * math.erfc(-hi / math.sqrt(2))
        pdf_lo = 0.0 if lo == -math.inf else math.exp(-0.5 * lo * lo) / math.sqrt(2 * math.pi)
        pdf_hi = 0.0 if hi == math.inf else math.exp(-0.5 * hi * hi) / math.sqrt(2 * math.pi)
        total += a[i] * (cdf_hi - cdf_lo) + b[i] * (pdf_lo - pdf_hi)
    return float(total)


def knowledge_gradient_gaussian(mu: Tensor, sigma2: Tensor, noise2: Num) -> Tensor:
    """Knowledge gradient of measuring each arm once: independent normal beliefs N(mu_i, sigma2_i), known
    observation noise variance (scalar or per arm). Closed form, [K] float64."""
    mu, s2 = _f64(mu), _f64(sigma2)
    n2 = _f64(noise2) + torch.zeros_like(mu)
    sig_tilde = s2 / torch.sqrt(s2 + n2)
    k = mu.shape[0]
    kg = torch.zeros(k, dtype=F64)
    for i in range(k):
        if k == 1 or float(sig_tilde[i]) <= 0:
            continue
        others = torch.cat([mu[:i], mu[i + 1:]]).max()
        z = -(mu[i] - others).abs() / sig_tilde[i]
        pdf = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
        kg[i] = sig_tilde[i] * (z * cdf + pdf)
    return kg


# ----------------------------------------------------------------------------------------------------------------
# Bayesian linear-Gaussian posterior
# ----------------------------------------------------------------------------------------------------------------
class LinearGaussianPosterior:
    """w ~ N(mean, cov); an observation of features phi is y = phi . w + N(0, noise_var). Float64."""

    def __init__(self, mean: Tensor, cov: Tensor, noise_var: float = 1.0):
        self.mean, self.cov, self.noise_var = _f64(mean), _f64(cov), float(noise_var)

    @classmethod
    def prior(cls, n_features: int, scale: float = 1.0, noise_var: float = 1.0) -> "LinearGaussianPosterior":
        return cls(torch.zeros(n_features, dtype=F64), torch.eye(n_features, dtype=F64) * scale ** 2, noise_var)

    def predict(self, phi: Tensor) -> Tuple[Tensor, Tensor]:
        """(mean, epistemic variance) of phi . w for phi [..., F]."""
        phi = _f64(phi)
        return phi @ self.mean, torch.einsum("...i,ij,...j->...", phi, self.cov, phi)

    def update(self, phi: Tensor, y: Num, noise_var: Optional[float] = None) -> "LinearGaussianPosterior":
        """Exact conjugate update for one observation (Joseph form: the covariance stays symmetric PSD)."""
        phi, nv = _f64(phi), self.noise_var if noise_var is None else float(noise_var)
        s2 = float(phi @ self.cov @ phi) + nv
        gain = self.cov @ phi / s2
        mean = self.mean + gain * (float(y) - float(phi @ self.mean))
        eye = torch.eye(len(phi), dtype=F64)
        a = eye - torch.outer(gain, phi)
        cov = a @ self.cov @ a.t() + nv * torch.outer(gain, gain)
        return LinearGaussianPosterior(mean, 0.5 * (cov + cov.t()), self.noise_var)

    def sample_weights(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        chol = torch.linalg.cholesky(self.cov + 1e-12 * torch.eye(len(self.mean), dtype=F64))
        return self.mean + torch.randn(n, len(self.mean), generator=generator, dtype=F64) @ chol.t()


def info_gain_nats(post: LinearGaussianPosterior, phi: Tensor, noise_var: Num) -> Tensor:
    """Decision-blind expected information gain about the weights from one observation: 0.5 log(1 + var/noise)."""
    _, var = post.predict(phi)
    return 0.5 * torch.log1p(var / _f64(noise_var))


def expected_surprise(post: LinearGaussianPosterior, phi: Tensor, noise_var: Num) -> Tensor:
    """Expected negative log-likelihood of the observation under the current predictive (its entropy, in nats):
    how unpredictable the outcome is. Large for pure noise; deliberately *not* a measure of value."""
    _, var = post.predict(phi)
    return 0.5 * torch.log(2 * math.pi * math.e * (var + _f64(noise_var)))


def voi_linear_gaussian(post: LinearGaussianPosterior, phi_probe: Tensor, phi_dec: Tensor,
                        noise_var: Optional[Num] = None, costs: Optional[VOICosts] = None,
                        dec_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Exact VOI of observing each probe once.

    ``phi_probe`` [P, F] features of the candidate observations; ``phi_dec`` [A, F] features of the candidates
    of the *next decision*, whose value is Q(b, a) = phi_dec[a] . E[w]. After observing probe p the decision
    values become  m + beta_p Z  with  beta_p = phi_dec Sigma phi_p / sqrt(phi_p' Sigma phi_p + noise_p).
    Returns ``gross`` (>= 0), ``net`` (after :class:`VOICosts`), ``base``, ``info_gain_nats``,
    ``expected_surprise`` — all [P] float64 except ``base``.
    """
    phi_probe, phi_dec = _f64(phi_probe), _f64(phi_dec)
    p = phi_probe.shape[0]
    nv = (torch.full((p,), post.noise_var, dtype=F64) if noise_var is None
          else _f64(noise_var) + torch.zeros(p, dtype=F64))
    m = phi_dec @ post.mean
    if dec_mask is not None:
        m, phi_dec = m[dec_mask], phi_dec[dec_mask]
    base = m.max()
    var = torch.einsum("pi,ij,pj->p", phi_probe, post.cov, phi_probe)
    beta = (phi_dec @ post.cov @ phi_probe.t()) / torch.sqrt(var + nv)             # [A, P]
    after = torch.tensor([expected_max_affine(m, beta[:, j]) for j in range(p)], dtype=F64)
    gross = (after - base).clamp_min(0.0)
    net = (costs or VOICosts()).net(base + gross, base)
    return {"gross": gross, "net": net, "base": base, "info_gain_nats": 0.5 * torch.log1p(var / nv),
            "expected_surprise": 0.5 * torch.log(2 * math.pi * math.e * (var + nv))}


# ----------------------------------------------------------------------------------------------------------------
# generic sampling estimator
# ----------------------------------------------------------------------------------------------------------------
def voi_sampling(candidates: Sequence[Any], sample_outcomes: Callable[[Any, int], Sequence[Any]],
                 update: Callable[[Any, Any], Any], q_values: Callable[[Any], Tensor], belief0: Any, n: int = 64,
                 costs: Optional[Sequence[VOICosts]] = None) -> Dict[str, Tensor]:
    """VOI by Monte Carlo with CAM's own components as callables.

    ``sample_outcomes(a, n)`` draws n outcomes of candidate ``a`` from the current predictive;
    ``update(a, y)`` returns the belief after observing y (a copy — never the live fast state);
    ``q_values(belief)`` returns the values of the next decision's candidates under a belief.
    Returns ``gross`` (may be slightly negative from sampling noise), ``net``, ``stderr`` and ``base``."""
    base = _f64(q_values(belief0)).max()
    gross, net, se = [], [], []
    for i, a in enumerate(candidates):
        ys = sample_outcomes(a, n)
        vals = torch.stack([_f64(q_values(update(a, y))).max() for y in ys])
        after = vals.mean()
        gross.append(after - base)
        se.append(vals.std(unbiased=True) / math.sqrt(len(vals)) if len(vals) > 1 else torch.zeros((), dtype=F64))
        net.append((costs[i] if costs is not None else VOICosts()).net(after, base))
    return {"gross": torch.stack(gross), "net": torch.stack(net), "stderr": torch.stack(se), "base": base}


# ----------------------------------------------------------------------------------------------------------------
# GLM posterior (assumed-density / Laplace rank-one updates)
# ----------------------------------------------------------------------------------------------------------------
class GLMPosterior:
    """w ~ N(mean, cov); eta = phi . w;  y ~ Bernoulli(sigmoid(eta))  or  y ~ Poisson(exposure * exp(eta)).

    One observation informs w only through eta, so the update is a one-dimensional Laplace approximation in eta
    (Newton's method on a concave log-posterior) mapped back to a rank-one change of (mean, cov)."""
    FAMILIES = ("bernoulli", "poisson")

    def __init__(self, mean: Tensor, cov: Tensor, family: str):
        if family not in self.FAMILIES:
            raise ValueError(f"family must be one of {self.FAMILIES}")
        self.mean, self.cov, self.family = _f64(mean), _f64(cov), family

    def eta(self, phi: Tensor) -> Tuple[Tensor, Tensor]:
        phi = _f64(phi)
        return phi @ self.mean, torch.einsum("...i,ij,...j->...", phi, self.cov, phi)

    def predict_mean(self, phi: Tensor, exposure: Num = 1.0) -> Tensor:
        """Posterior-mean response: sigmoid with the probit correction, or exposure * exp(m + v / 2)."""
        m, v = self.eta(phi)
        if self.family == "bernoulli":
            return torch.sigmoid(m / torch.sqrt(1.0 + math.pi * v / 8.0))
        return _f64(exposure) * torch.exp((m + 0.5 * v).clamp(max=50.0))

    def update(self, phi: Tensor, y: Num, exposure: float = 1.0) -> "GLMPosterior":
        phi = _f64(phi)
        m, v = float(phi @ self.mean), float(phi @ self.cov @ phi)
        if v <= 0:
            return GLMPosterior(self.mean, self.cov, self.family)
        y, eta = float(y), m
        for _ in range(50):
            if self.family == "bernoulli":
                s = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, eta))))
                g, h = y - s, -s * (1.0 - s)
            else:
                lam = exposure * math.exp(min(eta, 50.0))
                g, h = y - lam, -lam
            step = (g - (eta - m) / v) / (h - 1.0 / v)
            eta -= max(-5.0, min(5.0, step))
            if abs(step) < 1e-12:
                break
        if self.family == "bernoulli":
            s = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, eta))))
            h = -s * (1.0 - s)
        else:
            h = -exposure * math.exp(min(eta, 50.0))
        v_new = 1.0 / (1.0 / v - h)
        k = self.cov @ phi / v
        mean = self.mean + k * (eta - m)
        cov = self.cov - torch.outer(k, k) * (v - v_new)
        return GLMPosterior(mean, 0.5 * (cov + cov.t()), self.family)

    def sample_outcomes(self, phi: Tensor, n: int, generator: Optional[torch.Generator] = None,
                        exposure: float = 1.0) -> Tensor:
        """n draws from the posterior predictive of an observation with features ``phi`` [F]."""
        m, v = self.eta(phi)
        eta = m + torch.sqrt(v.clamp_min(0.0)) * torch.randn(n, generator=generator, dtype=F64)
        if self.family == "bernoulli":
            return torch.bernoulli(torch.sigmoid(eta), generator=generator)
        return torch.poisson(exposure * torch.exp(eta.clamp(max=30.0)), generator=generator)


def voi_glm(post: GLMPosterior, phi_probe: Tensor, phi_dec: Tensor, n_samples: int = 256,
            generator: Optional[torch.Generator] = None, exposure: float = 1.0,
            costs: Optional[VOICosts] = None) -> Dict[str, Tensor]:
    """VOI of each probe [P] for a Bernoulli / Poisson GLM by sampling outcomes from the posterior predictive.

    Q(b, a) is the posterior-mean response of decision candidate a. Identical sampled outcomes share one update
    (binary outcomes need two, counts a handful), so the cost is far below n_samples updates per probe."""
    phi_probe, phi_dec = _f64(phi_probe), _f64(phi_dec)
    base = post.predict_mean(phi_dec, exposure).max()
    gross, se = [], []
    for j in range(phi_probe.shape[0]):
        ys = post.sample_outcomes(phi_probe[j], n_samples, generator, exposure)
        uniq, inv = torch.unique(ys, return_inverse=True)
        best = torch.stack([post.update(phi_probe[j], float(y), exposure).predict_mean(phi_dec, exposure).max()
                            for y in uniq])
        vals = best[inv]
        gross.append(vals.mean() - base)
        se.append(vals.std(unbiased=True) / math.sqrt(n_samples))
    gross_t = torch.stack(gross)
    return {"gross": gross_t, "net": (costs or VOICosts()).net(base + gross_t, base), "stderr": torch.stack(se),
            "base": base}


# ----------------------------------------------------------------------------------------------------------------
# learned estimator of realised downstream usefulness
# ----------------------------------------------------------------------------------------------------------------
class DownstreamOutput:
    def __init__(self, loc: Tensor, scale: Tensor, df: Tensor):
        self.loc, self.scale, self.df = loc, scale, df

    @property
    def mean(self) -> Tensor:
        return self.loc

    def dist(self) -> D.StudentT:
        return D.StudentT(self.loc, self.scale, self.df)


class DownstreamValueEstimator(nn.Module):
    """Lightweight predictor of how useful the knowledge acquired by an action will turn out to be for later
    decisions. Inputs: the candidate's decision latent ``z`` and a vector of computed ``signals`` (for example
    the computed VOI, information gain, epistemic uncertainty, cost). Target: the *realised* downstream
    usefulness measured afterwards (e.g. reduction of later regret or of later prediction error on decisions
    that mattered). Output is a Student-t, so its own uncertainty is explicit."""

    def __init__(self, d_model: int, n_signals: int = 0, hidden: int = 96, out_std: float = 0.05):
        super().__init__()
        self.n_signals = n_signals
        self.in_z = nn.Linear(d_model, hidden)
        self.in_s = nn.Linear(n_signals, hidden) if n_signals > 0 else None
        self.norm = nn.LayerNorm(hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 3)
        nn.init.normal_(self.out.weight, std=out_std)                 # small random, never zero
        nn.init.zeros_(self.out.bias)

    def forward(self, z: Tensor, signals: Optional[Tensor] = None) -> DownstreamOutput:
        x = self.in_z(z)
        if self.in_s is not None:
            if signals is None:
                raise ValueError("this estimator was built with n_signals > 0 and needs signals")
            x = x + self.in_s(signals)
        o = self.out(F.silu(self.mid(F.silu(self.norm(x)))))
        return DownstreamOutput(o[..., 0], F.softplus(o[..., 1] + 0.5413) + 1e-4, 2.0 + F.softplus(o[..., 2] + 3.0))


def downstream_loss(out: DownstreamOutput, realised: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """NLL of the realised downstream usefulness; entries whose usefulness has not been measured are masked."""
    if mask is None:
        mask = torch.ones_like(realised, dtype=torch.bool)
    y = D.safe_targets(realised.to(out.loc.dtype), mask)
    return D.masked_mean(-D.student_t_log_prob(y, out.loc, out.scale, out.df), mask)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / cnt)[inv]


def spearman(x: Any, y: Any) -> float:
    x, y = np.asarray(_f64(x).numpy()).reshape(-1), np.asarray(_f64(y).numpy()).reshape(-1)
    if len(x) < 2:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def score_downstream(estimate: Any, realised: Any, reference: Optional[Any] = None, n_boot: int = 1000,
                     level: float = 0.95, seed: int = 0, margin: float = 0.0) -> Dict[str, Any]:
    """Compare estimates of downstream usefulness with what was realised.

    Reports the rank correlation and mean squared error of ``estimate`` against ``realised``; with a
    ``reference`` (the computed VOI) also its rank correlation and a paired-bootstrap interval of the difference.
    ``learned_beats_reference`` is True only when the whole interval of (learned - reference) lies above
    ``margin``: the learned estimator has to prove itself; otherwise the computed VOI stays in use."""
    e, r = np.asarray(_f64(estimate).numpy()).reshape(-1), np.asarray(_f64(realised).numpy()).reshape(-1)
    n = len(e)
    out: Dict[str, Any] = {"n": n, "spearman": spearman(e, r), "mse": float(np.mean((e - r) ** 2)),
                           "sign_agreement": float(np.mean(np.sign(e) == np.sign(r)))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n)) if n > 2 else np.zeros((0, n), dtype=np.int64)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    boots = np.array([spearman(e[i], r[i]) for i in idx])
    boots = boots[~np.isnan(boots)]
    if len(boots):
        out["spearman_lo"], out["spearman_hi"] = (float(q) for q in np.quantile(boots, [lo_q, hi_q]))
    if reference is not None:
        ref = np.asarray(_f64(reference).numpy()).reshape(-1)
        out["reference_spearman"] = spearman(ref, r)
        diffs = np.array([spearman(e[i], r[i]) - spearman(ref[i], r[i]) for i in idx])
        diffs = diffs[~np.isnan(diffs)]
        out["delta"] = out["spearman"] - out["reference_spearman"]
        if len(diffs):
            out["delta_lo"], out["delta_hi"] = (float(q) for q in np.quantile(diffs, [lo_q, hi_q]))
        out["learned_beats_reference"] = bool(len(diffs) > 0 and out.get("delta_lo", -1.0) > margin)
    return out
