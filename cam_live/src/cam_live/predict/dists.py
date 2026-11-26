"""cam_live.predict.dists — predictive families with exact, mask-aware log-likelihoods (spec 4.2, 6.2).

Families (names match ``contracts.Distribution.family``):

    normal, student_t                  real-valued quantities
    poisson, neg_binomial              counts over an exposure (mean = rate * exposure)
    bernoulli                          binary outcomes, parameterised by a logit
    discrete_hazard                    time-to-event on K bins with right-censoring
    categorical                        one of C classes

Every family has a functional log-likelihood (``*_log_prob``) used in training losses and a :class:`Dist` wrapper
with ``log_prob / mean / std / cdf / quantile / sample`` and conversion to and from ``contracts.Distribution``.
Nothing here depends on scipy at run time (the tests compare against scipy): the regularised incomplete beta
function needed for Student-t and negative-binomial CDFs is implemented below.

Masks: a likelihood is never evaluated on an invented target. :func:`masked_mean` and :func:`safe_targets` give
exactly zero loss *and* exactly zero gradient for masked entries, even when the masked target is NaN.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ..contracts import Distribution, OutcomeKind, OutcomeStatus

F64 = torch.float64
LOG_2PI = math.log(2.0 * math.pi)

FAMILY_NORMAL = "normal"
FAMILY_STUDENT_T = "student_t"
FAMILY_NEG_BINOMIAL = "neg_binomial"
FAMILY_POISSON = "poisson"
FAMILY_BERNOULLI = "bernoulli"
FAMILY_HAZARD = "discrete_hazard"
FAMILY_CATEGORICAL = "categorical"
FAMILIES = (FAMILY_NORMAL, FAMILY_STUDENT_T, FAMILY_NEG_BINOMIAL, FAMILY_POISSON, FAMILY_BERNOULLI, FAMILY_HAZARD,
            FAMILY_CATEGORICAL)

#: which outcome kind each family predicts
FAMILY_KIND = {FAMILY_NORMAL: OutcomeKind.REAL, FAMILY_STUDENT_T: OutcomeKind.REAL,
               FAMILY_NEG_BINOMIAL: OutcomeKind.COUNT, FAMILY_POISSON: OutcomeKind.COUNT,
               FAMILY_BERNOULLI: OutcomeKind.BINARY, FAMILY_HAZARD: OutcomeKind.TIME_TO_EVENT,
               FAMILY_CATEGORICAL: OutcomeKind.CATEGORICAL}

TensorLike = Union[Tensor, float, int, Sequence[float]]


def _t(x: TensorLike, like: Optional[Tensor] = None, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        return x if dtype is None else x.to(dtype)
    if like is not None:
        return torch.as_tensor(x, dtype=dtype or like.dtype, device=like.device)
    return torch.as_tensor(x, dtype=dtype or F64)


# ----------------------------------------------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------------------------------------------
def safe_targets(y: Tensor, mask: Tensor, fill: float = 0.0) -> Tensor:
    """Replace masked (pending / unobservable) targets by a harmless constant so no NaN can enter the graph."""
    return torch.where(mask, y, torch.full_like(y, fill))


def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, values, torch.zeros_like(values)).sum()


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over ``mask``; exactly 0 (with a zero gradient) when nothing is unmasked."""
    n = mask.sum().clamp_min(1).to(values.dtype)
    return masked_sum(values, mask) / n


# ----------------------------------------------------------------------------------------------------------------
# functional log-likelihoods (broadcasting; keep the dtype of the parameters)
# ----------------------------------------------------------------------------------------------------------------
def normal_log_prob(y: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    z = (y - loc) / scale
    return -0.5 * z * z - torch.log(scale) - 0.5 * LOG_2PI


def student_t_log_prob(y: Tensor, loc: Tensor, scale: Tensor, df: Tensor) -> Tensor:
    df = _t(df, like=scale)
    z = (y - loc) / scale
    return (torch.lgamma(0.5 * (df + 1.0)) - torch.lgamma(0.5 * df) - 0.5 * torch.log(df * math.pi)
            - torch.log(scale) - 0.5 * (df + 1.0) * torch.log1p(z * z / df))


def poisson_log_prob(y: Tensor, rate: Tensor, exposure: TensorLike = 1.0) -> Tensor:
    """log P(Y = y), Y ~ Poisson(rate * exposure)."""
    mu = rate * _t(exposure, like=rate)
    return torch.xlogy(y, mu) - mu - torch.lgamma(y + 1.0)


def neg_binomial_log_prob_from_log(y: Tensor, log_mu: Tensor, log_r: Tensor) -> Tensor:
    """Negative binomial with mean ``exp(log_mu)`` and concentration ``r = exp(log_r)`` (variance mu + mu^2 / r),
    evaluated in log space so very small or very large means stay finite."""
    r = torch.exp(log_r)
    log_r_mu = torch.logaddexp(log_r, log_mu)
    return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
            + r * (log_r - log_r_mu) + y * (log_mu - log_r_mu))


def neg_binomial_log_prob(y: Tensor, rate: Tensor, concentration: Tensor, exposure: TensorLike = 1.0) -> Tensor:
    """log P(Y = y) for mean ``rate * exposure`` and concentration ``r`` (r -> inf recovers the Poisson)."""
    mu = rate * _t(exposure, like=rate)
    conc = _t(concentration, like=rate)
    return neg_binomial_log_prob_from_log(y, torch.log(mu), torch.log(conc))


def bernoulli_log_prob(y: Tensor, logits: Tensor) -> Tensor:
    """y * log sigmoid(l) + (1 - y) * log sigmoid(-l); stable for any logit, accepts soft labels."""
    return y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)


def categorical_log_prob(y: Tensor, logits: Tensor, n_categories: Optional[Tensor] = None) -> Tensor:
    """``y`` long class index [...], ``logits`` [..., C]; classes >= ``n_categories`` [...] get zero probability."""
    logp = masked_log_softmax(logits, n_categories)
    idx = y.long().clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
    return logp.gather(-1, idx).squeeze(-1)


def masked_log_softmax(logits: Tensor, n_categories: Optional[Tensor] = None) -> Tensor:
    if n_categories is None:
        return torch.log_softmax(logits, dim=-1)
    c = logits.shape[-1]
    valid = torch.arange(c, device=logits.device) < n_categories.long().clamp(1, c).unsqueeze(-1)
    neg = torch.finfo(logits.dtype).min
    return torch.log_softmax(torch.where(valid, logits, torch.full_like(logits, neg)), dim=-1)


# ---- discrete-time hazard --------------------------------------------------------------------------------------
def hazard_bin(t: Tensor, edges: Tensor) -> Tensor:
    """Index of the bin ``[edges[k], edges[k+1])`` containing ``t``; K (one past the last bin) when
    ``t >= edges[-1]``. ``edges`` is [K+1] or [..., K+1] broadcastable against ``t`` [...]."""
    upper = edges[..., 1:]
    return (t.unsqueeze(-1) >= upper).sum(-1)


def _hazard_cum(hazard_logits: Tensor) -> Tensor:
    """[..., K+1]: log-probability of surviving the first k bins, k = 0..K."""
    cs = torch.cumsum(F.logsigmoid(-hazard_logits), dim=-1)
    return torch.cat([torch.zeros_like(cs[..., :1]), cs], dim=-1)


def hazard_log_prob(t: Tensor, event: Tensor, hazard_logits: Tensor, edges: Tensor) -> Tensor:
    """Censoring-aware log-likelihood of a time-to-event observation.

    ``hazard_logits`` [..., K]: logit of h_k = P(event in bin k | no event before bin k).
    ``event`` True  — the event happened at elapsed time ``t``: log h_k + sum_{j<k} log(1 - h_j); an event at or
                      beyond the last edge only tells us that every bin was survived.
    ``event`` False — right-censored at ``t`` (we only know T > t): sum over the bins *completed* by ``t`` of
                      log(1 - h_j). A censoring time inside a bin gives no information about that bin.
    """
    k_bins = hazard_logits.shape[-1]
    cum = _hazard_cum(hazard_logits)
    k = hazard_bin(t, edges).clamp(0, k_bins)
    surv = cum.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    log_h = F.logsigmoid(hazard_logits).gather(-1, k.clamp(max=k_bins - 1).unsqueeze(-1)).squeeze(-1)
    in_range = k < k_bins
    ev = event.bool()
    return torch.where(ev & in_range, surv + log_h, surv)


def hazard_pmf(hazard_logits: Tensor) -> Tensor:
    """[..., K+1]: P(event in bin k) for k < K, and the tail mass P(no event within the last edge) at index K."""
    cum = _hazard_cum(hazard_logits)
    pmf = torch.exp(cum[..., :-1] + F.logsigmoid(hazard_logits))
    return torch.cat([pmf, torch.exp(cum[..., -1:])], dim=-1)


def hazard_survival(hazard_logits: Tensor) -> Tensor:
    """[..., K]: survival function S(edges[k+1]) = prod_{j<=k} (1 - h_j) at the upper edge of each bin."""
    return torch.exp(_hazard_cum(hazard_logits)[..., 1:])


def hazard_expected_time(hazard_logits: Tensor, edges: Tensor) -> Tensor:
    """Restricted mean event time E[min(T, edges[-1])], events uniform inside their bin."""
    pmf = hazard_pmf(hazard_logits)
    mid = 0.5 * (edges[..., :-1] + edges[..., 1:])
    return (pmf[..., :-1] * mid).sum(-1) + pmf[..., -1] * edges[..., -1]


# ----------------------------------------------------------------------------------------------------------------
# special functions (float64)
# ----------------------------------------------------------------------------------------------------------------
def betainc(a: TensorLike, b: TensorLike, x: TensorLike, iters: int = 300) -> Tensor:
    """Regularised incomplete beta function I_x(a, b), continued fraction (modified Lentz), float64."""
    a, b, x = torch.broadcast_tensors(_t(a, dtype=F64), _t(b, dtype=F64), _t(x, dtype=F64))
    x = x.clamp(0.0, 1.0)
    swap = x > (a + 1.0) / (a + b + 2.0)
    a_, b_, x_ = torch.where(swap, b, a), torch.where(swap, a, b), torch.where(swap, 1.0 - x, x)
    lbeta = torch.lgamma(a_) + torch.lgamma(b_) - torch.lgamma(a_ + b_)
    front = torch.exp(torch.xlogy(a_, x_) + b_ * torch.log1p(-x_.clamp(max=1.0 - 1e-300)) - lbeta) / a_
    tiny = 1e-300

    def fix(v: Tensor) -> Tensor:
        return torch.where(v.abs() < tiny, torch.full_like(v, tiny), v)

    c = torch.ones_like(x_)
    d = 1.0 / fix(1.0 - (a_ + b_) * x_ / (a_ + 1.0))
    h = d.clone()
    for m in range(1, iters + 1):
        m2 = 2.0 * m
        aa = m * (b_ - m) * x_ / ((a_ + m2 - 1.0) * (a_ + m2))
        d = 1.0 / fix(1.0 + aa * d)
        c = fix(1.0 + aa / c)
        h = h * d * c
        aa = -(a_ + m) * (a_ + b_ + m) * x_ / ((a_ + m2) * (a_ + m2 + 1.0))
        d = 1.0 / fix(1.0 + aa * d)
        c = fix(1.0 + aa / c)
        delta = d * c
        h = h * delta
        if bool(((delta - 1.0).abs() < 1e-15).all()):
            break
    res = (front * h).clamp(0.0, 1.0)
    return torch.where(swap, 1.0 - res, res)


def betainc_scalar(a: float, b: float, x: float, iters: int = 400) -> float:
    """I_x(a, b) for Python floats (same continued fraction as :func:`betainc`, ~100x faster for one value)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - betainc_scalar(b, a, 1.0 - x, iters)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))) / a
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, max(0.0, front * h))


def student_t_cdf_scalar(z: float, df: float) -> float:
    tail = 0.5 * betainc_scalar(0.5 * df, 0.5, df / (df + z * z))
    return 1.0 - tail if z > 0 else tail


def student_t_icdf_scalar(q: float, df: float) -> float:
    lo, hi = -1.0, 1.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        z = mid / max(1.0 - abs(mid), 1e-300)
        if student_t_cdf_scalar(z, df) < q:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid / max(1.0 - abs(mid), 1e-300)


def normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * torch.erfc(-z / math.sqrt(2.0))


def normal_icdf(q: Tensor) -> Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


def student_t_cdf(z: TensorLike, df: TensorLike) -> Tensor:
    """CDF of the standard Student-t with ``df`` degrees of freedom."""
    z, df = torch.broadcast_tensors(_t(z, dtype=F64), _t(df, dtype=F64))
    tail = 0.5 * betainc(0.5 * df, torch.full_like(df, 0.5), df / (df + z * z))
    return torch.where(z > 0, 1.0 - tail, tail)


def student_t_icdf(q: TensorLike, df: TensorLike, iters: int = 90) -> Tensor:
    """Quantile of the standard Student-t by bisection on u in (-1, 1), z = u / (1 - |u|)."""
    q, df = torch.broadcast_tensors(_t(q, dtype=F64), _t(df, dtype=F64))
    lo, hi = torch.full_like(q, -1.0), torch.full_like(q, 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        z = mid / (1.0 - mid.abs()).clamp_min(1e-300)
        below = student_t_cdf(z, df) < q
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    mid = 0.5 * (lo + hi)
    return mid / (1.0 - mid.abs()).clamp_min(1e-300)


def poisson_cdf(k: Tensor, mu: Tensor) -> Tensor:
    k, mu = torch.broadcast_tensors(_t(k, dtype=F64), _t(mu, dtype=F64))
    out = torch.special.gammaincc(torch.floor(k).clamp_min(0.0) + 1.0, mu)
    return torch.where(k < 0, torch.zeros_like(out), out)


def neg_binomial_cdf(k: Tensor, mu: Tensor, concentration: Tensor) -> Tensor:
    k, mu, r = torch.broadcast_tensors(_t(k, dtype=F64), _t(mu, dtype=F64), _t(concentration, dtype=F64))
    out = betainc(r, torch.floor(k).clamp_min(0.0) + 1.0, r / (r + mu))
    return torch.where(k < 0, torch.zeros_like(out), out)


def _count_quantile(q: Tensor, cdf, mean: Tensor, std: Tensor) -> Tensor:
    """Smallest integer k with CDF(k) >= q (bisection over integers, vectorised)."""
    q = q.to(F64) * (1.0 - 1e-12)                        # CDFs are accurate to ~1e-13: ties resolve downwards
    hi = torch.ceil(mean + 12.0 * std + 20.0).to(F64) + torch.zeros_like(q)
    for _ in range(60):
        need = cdf(hi) < q
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0 + 1.0, hi)
    lo = torch.full_like(hi, -1.0)                       # invariant: CDF(lo) < q <= CDF(hi)
    for _ in range(200):
        if bool(((hi - lo) <= 1.0).all()):
            break
        mid = torch.floor(0.5 * (lo + hi))
        ok = cdf(mid) >= q
        hi, lo = torch.where(ok, mid, hi), torch.where(ok, lo, mid)
    return hi


def _gamma(shape: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    return torch._standard_gamma(shape, generator=generator)


# ----------------------------------------------------------------------------------------------------------------
# Dist wrappers
# ----------------------------------------------------------------------------------------------------------------
class Dist:
    """A batch of predictive distributions of one family. ``batch_shape`` is the shape of ``mean``."""
    family: str = ""

    @property
    def kind(self) -> str:
        return FAMILY_KIND[self.family]

    def param_dict(self) -> Dict[str, Tensor]:
        raise TypeError(f"{type(self).__name__} does not define parameters")

    def log_prob(self, y: Tensor, **kw: Any) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define a likelihood")

    @property
    def mean(self) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define a mean")

    @property
    def variance(self) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define a variance")

    @property
    def std(self) -> Tensor:
        return self.variance.clamp_min(0.0).sqrt()

    @property
    def batch_shape(self) -> torch.Size:
        return self.mean.shape

    def cdf(self, y: Tensor) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define a CDF")

    def quantile(self, q: TensorLike) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define quantiles")

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        raise TypeError(f"{type(self).__name__} does not define sampling")

    def interval(self, level: float = 0.9) -> Tuple[Tensor, Tensor]:
        a = 0.5 * (1.0 - level)
        return self.quantile(a), self.quantile(1.0 - a)

    # ---- contracts.Distribution ----
    def _select(self, v: Tensor, index: Optional[Tuple[int, ...]], event_dims: int = 0) -> Any:
        v = v.detach().to("cpu", F64)
        if index is not None:
            v = v[tuple(index)]
        batch_dims = v.dim() - event_dims
        if batch_dims > 0:
            if any(n != 1 for n in v.shape[:batch_dims]):
                raise ValueError("to_record needs an index selecting exactly one distribution of the batch")
            v = v.reshape(v.shape[batch_dims:])
        return v.tolist() if event_dims else float(v)

    _EVENT_DIMS: Dict[str, int] = {}

    def to_record(self, index: Optional[Tuple[int, ...]] = None, epistemic_std: Optional[float] = None
                  ) -> Distribution:
        params = {k: self._select(v, index, self._EVENT_DIMS.get(k, 0)) for k, v in self.param_dict().items()}
        mean = self._select(self.mean, index)
        std = self._select(self.std, index)
        return Distribution(family=self.family, params=params,
                            mean=mean if math.isfinite(mean) else None, std=std if math.isfinite(std) else None,
                            epistemic_std=None if epistemic_std is None else float(epistemic_std))


class Normal(Dist):
    family = FAMILY_NORMAL

    def __init__(self, loc: TensorLike, scale: TensorLike):
        self.loc = _t(loc)
        self.scale = _t(scale, like=self.loc)

    def param_dict(self):
        loc, scale = torch.broadcast_tensors(self.loc, self.scale)
        return {"loc": loc, "scale": scale}

    def log_prob(self, y, **kw):
        return normal_log_prob(y, self.loc, self.scale)

    @property
    def mean(self):
        return torch.broadcast_tensors(self.loc, self.scale)[0]

    @property
    def variance(self):
        return torch.broadcast_tensors(self.scale, self.loc)[0] ** 2

    def cdf(self, y):
        return normal_cdf(((y - self.loc) / self.scale).to(F64))

    def quantile(self, q):
        q = _t(q, dtype=F64)
        return self.loc.to(F64) + self.scale.to(F64) * normal_icdf(q)

    def sample(self, n, generator=None):
        loc, scale = torch.broadcast_tensors(self.loc, self.scale)
        eps = torch.randn((n,) + loc.shape, generator=generator, dtype=loc.dtype, device=loc.device)
        return loc + scale * eps


class StudentT(Dist):
    family = FAMILY_STUDENT_T

    def __init__(self, loc: TensorLike, scale: TensorLike, df: TensorLike):
        self.loc = _t(loc)
        self.scale = _t(scale, like=self.loc)
        self.df = _t(df, like=self.loc)

    def param_dict(self):
        loc, scale, df = torch.broadcast_tensors(self.loc, self.scale, self.df)
        return {"loc": loc, "scale": scale, "df": df}

    def log_prob(self, y, **kw):
        return student_t_log_prob(y, self.loc, self.scale, self.df)

    @property
    def mean(self):
        loc, _, df = torch.broadcast_tensors(self.loc, self.scale, self.df)
        return torch.where(df > 1.0, loc, torch.full_like(loc, float("nan")))

    @property
    def variance(self):
        _, scale, df = torch.broadcast_tensors(self.loc, self.scale, self.df)
        var = scale ** 2 * df / (df - 2.0).clamp_min(1e-12)
        return torch.where(df > 2.0, var, torch.full_like(var, float("inf")))

    def cdf(self, y):
        return student_t_cdf(((y - self.loc) / self.scale).to(F64), self.df)

    def quantile(self, q):
        loc, scale, df = torch.broadcast_tensors(self.loc.to(F64), self.scale.to(F64), self.df.to(F64))
        return loc + scale * student_t_icdf(_t(q, dtype=F64), df)

    def sample(self, n, generator=None):
        loc, scale, df = torch.broadcast_tensors(self.loc, self.scale, self.df)
        eps = torch.randn((n,) + loc.shape, generator=generator, dtype=loc.dtype, device=loc.device)
        chi2 = 2.0 * _gamma((0.5 * df).expand((n,) + loc.shape).contiguous(), generator)
        return loc + scale * eps * torch.sqrt(df / chi2.clamp_min(1e-30))


class Poisson(Dist):
    family = FAMILY_POISSON

    def __init__(self, rate: TensorLike, exposure: TensorLike = 1.0):
        self.rate = _t(rate)
        self.exposure = _t(exposure, like=self.rate)

    def param_dict(self):
        rate, exposure = torch.broadcast_tensors(self.rate, self.exposure)
        return {"rate": rate, "exposure": exposure}

    def log_prob(self, y, exposure: Optional[TensorLike] = None, **kw):
        return poisson_log_prob(y, self.rate, self.exposure if exposure is None else exposure)

    @property
    def mean(self):
        rate, exposure = torch.broadcast_tensors(self.rate, self.exposure)
        return rate * exposure

    @property
    def variance(self):
        return self.mean

    def cdf(self, y):
        return poisson_cdf(y, self.mean)

    def quantile(self, q):
        mean = self.mean.to(F64)
        q = _t(q, dtype=F64) + torch.zeros_like(mean)
        return _count_quantile(q, lambda k: poisson_cdf(k, mean), mean, mean.sqrt())

    def sample(self, n, generator=None):
        mean = self.mean
        return torch.poisson(mean.expand((n,) + mean.shape).contiguous(), generator=generator)


class NegBinomial(Dist):
    family = FAMILY_NEG_BINOMIAL

    def __init__(self, rate: TensorLike, concentration: TensorLike, exposure: TensorLike = 1.0):
        self.rate = _t(rate)
        self.concentration = _t(concentration, like=self.rate)
        self.exposure = _t(exposure, like=self.rate)

    def param_dict(self):
        rate, conc, exposure = torch.broadcast_tensors(self.rate, self.concentration, self.exposure)
        return {"rate": rate, "concentration": conc, "exposure": exposure}

    def log_prob(self, y, exposure: Optional[TensorLike] = None, **kw):
        return neg_binomial_log_prob(y, self.rate, self.concentration,
                                     self.exposure if exposure is None else exposure)

    @property
    def mean(self):
        rate, _, exposure = torch.broadcast_tensors(self.rate, self.concentration, self.exposure)
        return rate * exposure

    @property
    def variance(self):
        mu = self.mean
        return mu + mu * mu / torch.broadcast_tensors(self.concentration, mu)[0]

    def cdf(self, y):
        return neg_binomial_cdf(y, self.mean, self.concentration)

    def quantile(self, q):
        mean = self.mean.to(F64)
        conc = torch.broadcast_tensors(self.concentration.to(F64), mean)[0]
        q = _t(q, dtype=F64) + torch.zeros_like(mean)
        return _count_quantile(q, lambda k: neg_binomial_cdf(k, mean, conc), mean, self.std.to(F64))

    def sample(self, n, generator=None):
        mu = self.mean
        conc = torch.broadcast_tensors(self.concentration, mu)[0]
        lam = _gamma(conc.expand((n,) + mu.shape).contiguous(), generator) * (mu / conc)
        return torch.poisson(lam, generator=generator)


class Bernoulli(Dist):
    family = FAMILY_BERNOULLI

    def __init__(self, logits: TensorLike):
        self.logits = _t(logits)

    @classmethod
    def from_prob(cls, p: TensorLike, eps: float = 1e-12) -> "Bernoulli":
        p = _t(p).clamp(eps, 1.0 - eps)
        return cls(torch.log(p) - torch.log1p(-p))

    def param_dict(self):
        return {"logit": self.logits, "p": torch.sigmoid(self.logits)}

    def log_prob(self, y, **kw):
        return bernoulli_log_prob(y, self.logits)

    @property
    def mean(self):
        return torch.sigmoid(self.logits)

    @property
    def variance(self):
        p = self.mean
        return p * (1.0 - p)

    def cdf(self, y):
        p = self.mean.to(F64)
        y = _t(y, dtype=F64)
        return torch.where(y < 0, torch.zeros_like(p), torch.where(y < 1, 1.0 - p, torch.ones_like(p)))

    def quantile(self, q):
        p = self.mean.to(F64)
        return (_t(q, dtype=F64) > 1.0 - p).to(F64)

    def sample(self, n, generator=None):
        p = self.mean
        return torch.bernoulli(p.expand((n,) + p.shape).contiguous(), generator=generator)


class Categorical(Dist):
    family = FAMILY_CATEGORICAL
    _EVENT_DIMS = {"logits": 1, "probs": 1}

    def __init__(self, logits: TensorLike, n_categories: Optional[Tensor] = None):
        self.logits = _t(logits)
        self.n_categories = n_categories
        self.log_probs = masked_log_softmax(self.logits, n_categories)

    @property
    def probs(self) -> Tensor:
        return torch.exp(self.log_probs)

    def param_dict(self):
        return {"logits": self.log_probs, "probs": self.probs}

    def log_prob(self, y, **kw):
        idx = y.long().clamp(0, self.logits.shape[-1] - 1).unsqueeze(-1)
        return self.log_probs.gather(-1, idx).squeeze(-1)

    @property
    def mean(self):
        """Expected class index (meaningful only for ordered categories; reported for completeness)."""
        idx = torch.arange(self.logits.shape[-1], dtype=self.logits.dtype, device=self.logits.device)
        return (self.probs * idx).sum(-1)

    @property
    def variance(self):
        idx = torch.arange(self.logits.shape[-1], dtype=self.logits.dtype, device=self.logits.device)
        return (self.probs * idx ** 2).sum(-1) - self.mean ** 2

    def entropy(self) -> Tensor:
        return -(self.probs * torch.where(self.probs > 0, self.log_probs, torch.zeros_like(self.probs))).sum(-1)

    def cdf(self, y):
        c = torch.cumsum(self.probs.to(F64), -1)
        idx = torch.floor(_t(y, dtype=F64)).long().clamp(-1, self.logits.shape[-1] - 1)
        out = c.gather(-1, idx.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return torch.where(idx < 0, torch.zeros_like(out), out)

    def quantile(self, q):
        c = torch.cumsum(self.probs.to(F64), -1)
        q = _t(q, dtype=F64) + torch.zeros_like(c[..., 0])
        return (c < q.unsqueeze(-1) - 1e-12).sum(-1).clamp(max=self.logits.shape[-1] - 1).to(F64)

    def sample(self, n, generator=None):
        p = self.probs
        flat = p.reshape(-1, p.shape[-1])
        s = torch.multinomial(flat, n, replacement=True, generator=generator)       # [prod, n]
        return s.t().reshape((n,) + p.shape[:-1])


class DiscreteHazard(Dist):
    """Time-to-event on bins ``[edges[k], edges[k+1])`` with hazards ``sigmoid(hazard_logits[k])``.

    ``mean`` is the restricted mean ``E[min(T, edges[-1])]``; ``sample`` returns ``inf`` when the event does not
    occur within the last edge (the caller sees a censored observation at ``edges[-1]``)."""
    family = FAMILY_HAZARD
    _EVENT_DIMS = {"hazard_logits": 1, "edges": 1}

    def __init__(self, hazard_logits: TensorLike, edges: TensorLike):
        self.hazard_logits = _t(hazard_logits)
        self.edges = _t(edges, like=self.hazard_logits)
        if self.edges.shape[-1] != self.hazard_logits.shape[-1] + 1:
            raise ValueError("edges must have one more entry than hazard_logits")

    def param_dict(self):
        edges = self.edges.expand(self.hazard_logits.shape[:-1] + self.edges.shape[-1:])
        return {"hazard_logits": self.hazard_logits, "edges": edges}

    def log_prob(self, y, event: Optional[Tensor] = None, **kw):
        ev = torch.ones_like(y, dtype=torch.bool) if event is None else event
        return hazard_log_prob(y, ev, self.hazard_logits, self.edges)

    def pmf(self) -> Tensor:
        return hazard_pmf(self.hazard_logits)

    def survival(self) -> Tensor:
        return hazard_survival(self.hazard_logits)

    def p_event(self) -> Tensor:
        """Probability that the event happens within the last edge."""
        return 1.0 - self.survival()[..., -1]

    @property
    def mean(self):
        return hazard_expected_time(self.hazard_logits, self.edges)

    @property
    def variance(self):
        pmf = self.pmf()
        lo, hi = self.edges[..., :-1], self.edges[..., 1:]
        second = (pmf[..., :-1] * (lo * lo + lo * hi + hi * hi) / 3.0).sum(-1) + pmf[..., -1] * self.edges[..., -1] ** 2
        return second - self.mean ** 2

    def cdf(self, y):
        """P(T <= y), linear inside a bin; equals ``p_event`` for y >= edges[-1]."""
        y = _t(y, like=self.hazard_logits)
        pmf = self.pmf()[..., :-1]
        lo, hi = self.edges[..., :-1], self.edges[..., 1:]
        frac = ((y.unsqueeze(-1) - lo) / (hi - lo)).clamp(0.0, 1.0)
        return (pmf * frac).sum(-1)

    def quantile(self, q):
        """Smallest t with P(T <= t) >= q; ``edges[-1]`` when the event mass within the horizon is below q."""
        pmf = self.pmf()[..., :-1].to(F64)
        edges = self.edges.to(F64).expand(pmf.shape[:-1] + self.edges.shape[-1:])
        cum = torch.cumsum(pmf, -1)
        q = (_t(q, dtype=F64) + torch.zeros_like(cum[..., 0])).unsqueeze(-1)
        k = (cum < q).sum(-1, keepdim=True)
        kk = k.clamp(max=pmf.shape[-1] - 1)
        prev = torch.where(kk > 0, cum.gather(-1, (kk - 1).clamp_min(0)), torch.zeros_like(q))
        mass = pmf.gather(-1, kk).clamp_min(1e-300)
        lo, hi = edges.gather(-1, kk), edges.gather(-1, kk + 1)
        t = lo + (hi - lo) * ((q - prev) / mass).clamp(0.0, 1.0)
        return torch.where(k >= pmf.shape[-1], edges[..., -1:], t).squeeze(-1)

    def sample(self, n, generator=None):
        pmf = self.pmf()
        flat = pmf.reshape(-1, pmf.shape[-1])
        k = torch.multinomial(flat, n, replacement=True, generator=generator).t().reshape((n,) + pmf.shape[:-1])
        edges = self.edges.expand(pmf.shape[:-1] + self.edges.shape[-1:])
        edges_n = edges.unsqueeze(0).expand((n,) + edges.shape)
        kk = k.clamp(max=pmf.shape[-1] - 2).unsqueeze(-1)
        lo, hi = edges_n.gather(-1, kk).squeeze(-1), edges_n.gather(-1, kk + 1).squeeze(-1)
        u = torch.rand(k.shape, generator=generator, dtype=pmf.dtype, device=pmf.device)
        t = lo + (hi - lo) * u
        return torch.where(k >= pmf.shape[-1] - 1, torch.full_like(t, float("inf")), t)


# ----------------------------------------------------------------------------------------------------------------
# contracts.Distribution <-> Dist
# ----------------------------------------------------------------------------------------------------------------
def from_record(rec: Distribution) -> Dist:
    """Rebuild a scalar :class:`Dist` (float64) from a stored ``contracts.Distribution``."""
    p, fam = rec.params, rec.family
    if fam == FAMILY_NORMAL:
        return Normal(_t(p["loc"]), _t(p["scale"]))
    if fam == FAMILY_STUDENT_T:
        return StudentT(_t(p["loc"]), _t(p["scale"]), _t(p["df"]))
    if fam == FAMILY_POISSON:
        return Poisson(_t(p["rate"]), _t(p.get("exposure", 1.0)))
    if fam == FAMILY_NEG_BINOMIAL:
        return NegBinomial(_t(p["rate"]), _t(p["concentration"]), _t(p.get("exposure", 1.0)))
    if fam == FAMILY_BERNOULLI:
        if "logit" in p:
            return Bernoulli(_t(p["logit"]))
        return Bernoulli.from_prob(_t(p["p"]))
    if fam == FAMILY_HAZARD:
        return DiscreteHazard(_t(p["hazard_logits"]), _t(p["edges"]))
    if fam == FAMILY_CATEGORICAL:
        if "logits" in p:
            return Categorical(_t(p["logits"]))
        return Categorical(torch.log(_t(p["probs"]).clamp_min(1e-300)))
    raise ValueError(f"unknown distribution family {fam!r}")


def record_log_prob(rec: Distribution, value: float, status: str = OutcomeStatus.OBSERVED,
                    exposure: Optional[float] = None) -> float:
    """Log-likelihood of a matured outcome under a stored prediction. ``status`` CENSORED is only meaningful for
    time-to-event predictions (the value is the censoring time). Pending / unobservable outcomes have no
    likelihood and raise."""
    if status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
        raise ValueError(f"no likelihood for an outcome with status {status!r}")
    d = from_record(rec)
    y = torch.as_tensor(float(value), dtype=F64)
    if isinstance(d, DiscreteHazard):
        return float(d.log_prob(y, event=torch.tensor(status == OutcomeStatus.OBSERVED)))
    if status == OutcomeStatus.CENSORED:
        raise ValueError("a censored observation can only be scored by a discrete_hazard prediction")
    if exposure is not None and isinstance(d, (Poisson, NegBinomial)):
        return float(d.log_prob(y, exposure=float(exposure)))
    return float(d.log_prob(y))


def default_family(kind: str) -> str:
    return {OutcomeKind.REAL: FAMILY_STUDENT_T, OutcomeKind.COUNT: FAMILY_NEG_BINOMIAL,
            OutcomeKind.BINARY: FAMILY_BERNOULLI, OutcomeKind.TIME_TO_EVENT: FAMILY_HAZARD,
            OutcomeKind.CATEGORICAL: FAMILY_CATEGORICAL}[kind]


def uniform_edges(horizon: TensorLike, n_bins: int, like: Optional[Tensor] = None) -> Tensor:
    """[..., K+1] edges dividing ``[0, horizon]`` into K equal bins (``horizon`` may be a batch)."""
    h = _t(horizon, like=like)
    frac = torch.linspace(0.0, 1.0, n_bins + 1, dtype=h.dtype, device=h.device)
    return h.unsqueeze(-1) * frac
