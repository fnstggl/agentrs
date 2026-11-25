"""CAM 7.0-style mathematical value of information: optional policy inputs ('math-assist') and baselines only.

Spec 8: "Retain mathematical value of information as a baseline and optional initial training signal, not the sole
permanent exploration mechanism. The learned exploration strategy must be capable of operating without the old
programmed knowledge-gradient action score."

The NIG regression, Student-t predictive, Thompson draws, knowledge gradient and information gain are ported from
``superlearner/src/superlearner/cam7/posterior.py`` (CAM 7.0, tested in ``superlearner/tests/cam7``); the math is
unchanged. Added here: a streaming sufficient-statistics form (O(D^2) per observation instead of refitting the whole
history), per-candidate score tensors, a guarded encoder that turns the scores into BELIEF-kind tokens **only** in the
declared math-assist condition (``CamConfig.math_explore_assist``), and the baseline exploration policies of spec 8.5
(random, greedy, Thompson, knowledge gradient, expected information gain) over discrete candidates.

Two candidate models back the baselines:

* `CandidateValueModel` — CAM 7.0's model: an independent NIG value per candidate (one-hot features). It cannot
  represent that one candidate's observation is informative about another, so one-step VOI computed on it can never
  value an information-only experiment (the probe of the probe bandit) — by construction, not by optimisation.
* `HypothesisValueModel` — a *structured prior*: a discrete hypothesis space with per-hypothesis outcome likelihoods
  for every candidate (given by the task family, e.g. `diagnostics.ProbeBanditAdapter.structured_prior`). Any
  observation updates the posterior over hypotheses, so the knowledge gradient of an experiment includes what it
  reveals about every other candidate (exact, finite outcome sets). This is the strong programmed baseline: it needs
  the hand-specified hypothesis model the learned policy never receives.

Model per environment b and outcome k: y = w_k^T phi + eps, eps ~ N(0, s2_k), w_k | s2_k ~ N(mu0, s2_k Lambda0^-1),
s2_k ~ InvGamma(a0, b0). Linear algebra in float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..interfaces import TokenKind, TokenSet

F64 = torch.float64


@dataclass
class Posterior:
    mu: Tensor        # [B,K,D]
    chol: Tensor      # [B,K,D,D] lower Cholesky factor of Lambda_n
    a: Tensor         # [B,K]
    b: Tensor         # [B,K]
    n: Tensor         # [B,K]

    @property
    def s2(self) -> Tensor:
        return self.b / self.a

    @property
    def dof(self) -> Tensor:
        return 2.0 * self.a


def _expand_prior(B, K, D, mu0, lam0, a0, b0):
    return (mu0.to(F64).expand(B, K, D), lam0.to(F64).expand(B, K, D), a0.to(F64).expand(B, K),
            b0.to(F64).expand(B, K))


def _solve_post(gram, rhs_y, yy, n, mu0, lam0, a0, b0) -> Posterior:
    """Exact NIG posterior from sufficient statistics: gram [B,D,D], rhs_y = Phi^T Omega y [B,K,D],
    yy = y^T Omega y [B,K], n = sum omega [B,K]."""
    B, D, _ = gram.shape
    K = rhs_y.shape[1]
    prec = gram.unsqueeze(1) + torch.diag_embed(lam0)
    jitter = 1e-9 * prec.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12)
    prec = prec + torch.diag_embed(jitter.expand(B, K, D))
    chol = torch.linalg.cholesky(prec)
    rhs = lam0 * mu0 + rhs_y
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    # sse = sum_t w (y - phi mu)^2 = yy - 2 mu.rhs_y + mu^T G mu
    sse = (yy - 2 * (mu * rhs_y).sum(-1) + torch.einsum("bkd,bde,bke->bk", mu, gram, mu)).clamp_min(0)
    dev = mu - mu0
    reg = (dev * lam0 * dev).sum(-1)
    return Posterior(mu, chol, a0 + 0.5 * n, b0 + 0.5 * (sse + reg), n)


def fit(phi: Tensor, y: Tensor, omega: Tensor, mu0: Tensor, lam0: Tensor, a0: Tensor, b0: Tensor) -> Posterior:
    """phi [B,T,D]; y [B,T,K]; omega [B,T] row weights (0 = absent); priors [B,K,D]/[K,D] (lam0 = diagonal precision),
    a0, b0 [B,K]/[K]. Exact NIG posterior (CAM 7.0 posterior.fit), differentiable."""
    B, T, D = phi.shape
    K = y.shape[-1]
    phi, y, w = phi.to(F64), y.to(F64), omega.to(F64)
    mu0, lam0, a0, b0 = _expand_prior(B, K, D, mu0, lam0, a0, b0)
    gram = torch.einsum("btd,bt,bte->bde", phi, w, phi)
    prec = gram.unsqueeze(1) + torch.diag_embed(lam0)
    jitter = 1e-9 * prec.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12)
    prec = prec + torch.diag_embed(jitter.expand(B, K, D))
    chol = torch.linalg.cholesky(prec)
    rhs = lam0 * mu0 + torch.einsum("btd,bt,btk->bkd", phi, w, y)
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    resid = y - torch.einsum("btd,bkd->btk", phi, mu)
    sse = torch.einsum("bt,btk->bk", w, resid * resid)
    dev = mu - mu0
    reg = (dev * lam0 * dev).sum(-1)
    n = w.sum(1, keepdim=True).expand(B, K)
    return Posterior(mu, chol, a0 + 0.5 * n, b0 + 0.5 * (sse + reg), n)


class NIGStats:
    """Streaming sufficient statistics of the NIG regression (decayed Gram, Phi^T y, y^T y, n). ``posterior()``
    equals `fit` on the same weighted rows; each ``add`` costs O(N D^2) for N new rows."""

    def __init__(self, B: int, D: int, K: int = 1, device=None):
        self.gram = torch.zeros(B, D, D, dtype=F64, device=device)
        self.xy = torch.zeros(B, K, D, dtype=F64, device=device)
        self.yy = torch.zeros(B, K, dtype=F64, device=device)
        self.n = torch.zeros(B, K, dtype=F64, device=device)

    def add(self, phi: Tensor, y: Tensor, omega: Optional[Tensor] = None) -> "NIGStats":
        """phi [B,D] or [B,N,D]; y [B,K] or [B,N,K]; omega [B] or [B,N] (default 1)."""
        if phi.dim() == 2:
            phi, y = phi.unsqueeze(1), y.unsqueeze(1)
            omega = None if omega is None else omega.unsqueeze(1)
        phi, y = phi.to(F64), y.to(F64)
        w = torch.ones(phi.shape[:2], dtype=F64, device=phi.device) if omega is None else omega.to(F64)
        self.gram = self.gram + torch.einsum("bnd,bn,bne->bde", phi, w, phi)
        self.xy = self.xy + torch.einsum("bnd,bn,bnk->bkd", phi, w, y)
        self.yy = self.yy + torch.einsum("bn,bnk->bk", w, y * y)
        self.n = self.n + w.sum(1, keepdim=True)
        return self

    def decay(self, factor: float) -> "NIGStats":
        """Exponential forgetting of every past row (non-stationary mechanisms)."""
        self.gram, self.xy, self.yy, self.n = (self.gram * factor, self.xy * factor, self.yy * factor,
                                               self.n * factor)
        return self

    def posterior(self, mu0: Tensor, lam0: Tensor, a0: Tensor, b0: Tensor) -> Posterior:
        B, K, D = self.xy.shape
        mu0, lam0, a0, b0 = _expand_prior(B, K, D, mu0, lam0, a0, b0)
        return _solve_post(self.gram, self.xy, self.yy, self.n, mu0, lam0, a0, b0)


def _solve_l(post: Posterior, phi: Tensor) -> Tensor:
    B, Q, D = phi.shape
    K = post.mu.shape[1]
    rhs = phi.to(F64).transpose(1, 2).unsqueeze(1).expand(B, K, D, Q)
    return torch.linalg.solve_triangular(post.chol, rhs, upper=False)


def predict(post: Posterior, phi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Student-t predictive at phi [B,Q,D]: (loc [B,Q,K], squared scale [B,Q,K], dof [B,K])."""
    loc = torch.einsum("bqd,bkd->bqk", phi.to(F64), post.mu)
    v = _solve_l(post, phi)
    quad = (v * v).sum(2).transpose(1, 2)
    return loc, post.s2.unsqueeze(1) * (1.0 + quad), post.dof


def mean_cov(post: Posterior, phi: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean [B,Q,K] and covariance [B,K,Q,Q] of the noise-free outcome phi^T w."""
    loc = torch.einsum("bqd,bkd->bqk", phi.to(F64), post.mu)
    v = _solve_l(post, phi)
    return loc, post.s2[..., None, None] * torch.einsum("bkdq,bkdr->bkqr", v, v)


def t_nll(y: Tensor, loc: Tensor, scale2: Tensor, dof: Tensor) -> Tensor:
    y = y.to(F64)
    nu = dof.unsqueeze(1).expand_as(loc)
    z2 = (y - loc) ** 2 / scale2
    return (-torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2) + 0.5 * torch.log(nu * math.pi * scale2)
            + (nu + 1) / 2 * torch.log1p(z2 / nu))


def sample_weights(post: Posterior, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
    """n joint posterior draws (s2, w); returns w [B,n,K,D]. Reproducible with a CPU ``generator``."""
    B, K, D = post.mu.shape
    dev = post.mu.device
    shape = post.a.unsqueeze(1).expand(B, n, K).contiguous()
    if generator is not None:
        # Gamma(a, 1) via torch.distributions with a seeded uniform source is not available; use the
        # Marsaglia-Tsang free form through the standard normal / uniform of the generator.
        g = _gamma_mt(shape, generator)
    else:
        g = torch._standard_gamma(shape)
    s2 = post.b.unsqueeze(1) / g.clamp_min(1e-12)
    z = _draw((B, n, K, D, 1), dev, generator, normal=True)
    lt = post.chol.unsqueeze(1).expand(B, n, K, D, D).transpose(-1, -2)
    eps = torch.linalg.solve_triangular(lt, z, upper=True).squeeze(-1)
    return post.mu.unsqueeze(1) + s2.sqrt().unsqueeze(-1) * eps


def _draw(shape, device, generator: Optional[torch.Generator], normal: bool = False) -> Tensor:
    """float64 uniform / normal draws on ``device`` from a generator that may live on another device."""
    fn = torch.randn if normal else torch.rand
    if generator is None:
        return fn(shape, dtype=F64, device=device)
    return fn(shape, dtype=F64, device=generator.device, generator=generator).to(device)


def _gamma_mt(alpha: Tensor, generator: torch.Generator, max_iter: int = 64) -> Tensor:
    """Marsaglia-Tsang Gamma(alpha, 1) sampler driven by ``generator`` (alpha > 0; boost for alpha < 1).
    The generator may live on another device than ``alpha`` (draws are moved)."""
    a = alpha.to(F64)
    boost = a < 1
    a1 = torch.where(boost, a + 1, a)
    d = a1 - 1.0 / 3
    c = 1.0 / torch.sqrt(9 * d)
    out = torch.zeros_like(a)
    done = torch.zeros_like(a, dtype=torch.bool)
    for _ in range(max_iter):
        x = _draw(a.shape, a.device, generator, normal=True)
        u = _draw(a.shape, a.device, generator)
        v = (1 + c * x) ** 3
        ok = (v > 0) & (torch.log(u.clamp_min(1e-300)) < 0.5 * x * x + d - d * v + d * torch.log(v.clamp_min(1e-300)))
        new = ok & ~done
        out = torch.where(new, d * v, out)
        done = done | ok
        if bool(done.all()):
            break
    out = torch.where(done, out, d)        # vanishingly rare fallback: the mode
    u = _draw(a.shape, a.device, generator)
    return torch.where(boost, out * u.clamp_min(1e-300) ** (1.0 / a), out)


def _normal_quantiles(n: int, device, dtype=F64) -> Tensor:
    p = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return math.sqrt(2.0) * torch.erfinv(2 * p - 1)


def knowledge_gradient(mean: Tensor, cov: Tensor, obs_var: Tensor, valid: Optional[Tensor] = None,
                       n_points: int = 32) -> Tensor:
    """KG(a) = E_y[max_a' mean'(a')] - max_a' mean(a') >= 0 for one observation of candidate a (CAM 7.0).
    mean [B,C], cov [B,C,C], obs_var [B,C]; invalid candidates excluded (KG 0)."""
    B, C = mean.shape
    mean, cov = mean.to(F64), cov.to(F64)
    big = torch.finfo(F64).max / 4
    if valid is None:
        valid = torch.ones(B, C, dtype=torch.bool, device=mean.device)
    m = torch.where(valid, mean, torch.full_like(mean, -big))
    sd = (cov.diagonal(dim1=-2, dim2=-1) + obs_var.to(F64)).clamp_min(1e-30).sqrt()
    beta = cov / sd.unsqueeze(1)
    z = _normal_quantiles(n_points, mean.device)
    vals = m[:, :, None, None] + beta[..., None] * z
    vals = torch.where(valid[:, :, None, None], vals, torch.full_like(vals, -big))
    best_after = vals.max(1).values.mean(-1)
    kg = (best_after - m.max(1, keepdim=True).values).clamp_min(0.0)
    return torch.where(valid, kg, torch.zeros_like(kg))


def info_gain(cov: Tensor, obs_var: Tensor) -> Tensor:
    """0.5 log(1 + var / noise): expected information (nats) about a candidate's own value (decision-blind)."""
    var = cov.diagonal(dim1=-2, dim2=-1).to(F64)
    return 0.5 * torch.log1p(var / obs_var.to(F64).clamp_min(1e-30))


# ----------------------------------------------------------------------------------------------- candidate scores

SCORE_NAMES = ("mean", "sd", "kg", "eig", "thompson")


class CandidateValueModel:
    """NIG model of each discrete candidate's immediate outcome (one-hot features, K = 1) for B environments.

    This is the per-candidate value model CAM 7.0's KG acted on. It has no representation of what a candidate's
    *observation* reveals about other candidates (e.g. a costly probe), which is exactly why one-step VOI baselines
    cannot value such experiments (spec 7.4 / 8.4)."""

    def __init__(self, B: int, n_candidates: int, prior_mean=0.0, prior_prec: float = 1.0, a0: float = 2.0,
                 b0: float = 1.0, device=None):
        self.B, self.C = B, n_candidates
        self.stats = NIGStats(B, n_candidates, 1, device)
        pm = torch.as_tensor(prior_mean, dtype=F64, device=device).reshape(-1)
        self.mu0 = (pm.expand(n_candidates) if pm.numel() == 1 else pm).reshape(1, n_candidates).clone()
        self.lam0 = torch.full((1, n_candidates), float(prior_prec), dtype=F64, device=device)
        self.a0 = torch.full((1,), float(a0), dtype=F64, device=device)
        self.b0 = torch.full((1,), float(b0), dtype=F64, device=device)
        self.eye = torch.eye(n_candidates, dtype=F64, device=device)

    def update(self, choice: Tensor, outcome: Tensor, active: Optional[Tensor] = None) -> None:
        phi = self.eye[choice.long()]                                  # [B,C]
        w = None if active is None else active.to(F64)
        self.stats.add(phi, outcome.to(F64).unsqueeze(-1), w)

    def posterior(self) -> Posterior:
        return self.stats.posterior(self.mu0, self.lam0, self.a0, self.b0)

    def scores(self, valid: Optional[Tensor] = None, generator: Optional[torch.Generator] = None) -> Tensor:
        """[B, C, 5] float32 scores per candidate: posterior mean, sd, KG, EIG, one Thompson draw."""
        post = self.posterior()
        phi = self.eye.unsqueeze(0).expand(self.B, -1, -1)
        mean, cov = mean_cov(post, phi)                                # [B,C,1], [B,1,C,C]
        mean, cov = mean[..., 0], cov[:, 0]
        obs_var = post.s2.expand(self.B, self.C)
        kg = knowledge_gradient(mean, cov, obs_var, valid)
        eig = info_gain(cov, obs_var)
        w = sample_weights(post, 1, generator)[:, 0, 0]                # [B,C]
        sd = cov.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
        return torch.stack([mean, sd, kg, eig, w], -1).float()


class MathAssistEncoder(nn.Module):
    """Per-candidate math scores [B,C,F] -> BELIEF-kind tokens [B,C,D] (refs ``math_assist:<c>``).

    Only usable in the declared math-assist condition: constructed with ``enabled=cfg.math_explore_assist`` and
    raising otherwise, so the learned-exploration condition cannot silently receive programmed VOI scores."""

    def __init__(self, d_model: int, n_scores: int = len(SCORE_NAMES), enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        self.proj = nn.Sequential(nn.Linear(2 * n_scores, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    @staticmethod
    def from_config(cfg, d_model: Optional[int] = None, n_scores: int = len(SCORE_NAMES)) -> "MathAssistEncoder":
        """Enabled exactly when the run declares the math-assist condition (``cfg.math_explore_assist``)."""
        return MathAssistEncoder(cfg.d_model if d_model is None else d_model, n_scores,
                                 enabled=bool(cfg.math_explore_assist))

    def features(self, scores: Tensor) -> Tensor:
        s = scores.float()
        return torch.cat([torch.sign(s) * torch.log1p(s.abs()), (s - s.mean(1, keepdim=True))], -1)

    def forward(self, scores: Tensor, mask: Optional[Tensor] = None) -> TokenSet:
        if not self.enabled:
            raise RuntimeError("math-assist inputs are only allowed in the declared math-assist condition "
                               "(CamConfig.math_explore_assist=True)")
        x = self.proj(self.features(scores))
        B, C, _ = x.shape
        dev = x.device
        return TokenSet(x, torch.ones(B, C, dtype=torch.bool, device=dev) if mask is None else mask.bool(),
                        torch.full((B, C), int(TokenKind.BELIEF), dtype=torch.long, device=dev),
                        torch.zeros(B, C, device=dev), torch.full((B, C), -1, dtype=torch.long, device=dev),
                        [[f"math_assist:{c}" for c in range(C)] for _ in range(B)])


@dataclass
class StructuredPrior:
    """A task family's discrete hypothesis model over C candidates (input of `HypothesisValueModel`).

    lik [C, H, O]: p(outcome o | candidate c, hypothesis h), rows sum to 1 over o (pad unused outcomes with 0);
    values [C, O]: immediate value of outcome o of candidate c (money unit, including the candidate's cost);
    prior_logw [H]; names: [C] (tool, args) of the candidates in `DiscreteToolPolicy.candidates` order;
    outcome_fn(c, observations) -> int: the outcome index of candidate c from the observations it produced."""
    lik: Tensor
    values: Tensor
    prior_logw: Tensor
    names: list
    outcome_fn: object


class HypothesisValueModel:
    """Exact Bayesian decision model over discrete hypotheses with correlated candidates (structured prior), for B
    environments. ``logw [B,H]`` is updated by every observed outcome; ``M [H,C]`` is the expected immediate value
    of each candidate under each hypothesis.

    * ``mean()`` [B,C] = sum_h w_h M[h, c];
    * ``knowledge_gradient()`` [B,C] = E_{o ~ p(o | c)}[max_c' mean'(c' | o)] - max_c' mean(c') >= 0, where mean' uses
      the posterior after observing o from c (exact sum over the finite outcome set);
    * ``info_gain()`` [B,C] = I(o; h | c) in nats (BALD);
    * ``thompson(generator)`` [B,C] = M[h*, c] for one posterior draw h*."""

    def __init__(self, B: int, prior: StructuredPrior, device=None):
        self.B = B
        self.lik = prior.lik.to(device=device, dtype=F64)              # [C,H,O]
        self.values = prior.values.to(device=device, dtype=F64)        # [C,O]
        self.C, self.H, self.O = self.lik.shape
        if self.values.shape != (self.C, self.O):
            raise ValueError("values must be [C, O] matching lik [C, H, O]")
        s = self.lik.sum(-1)
        if not torch.allclose(s, torch.ones_like(s), atol=1e-6):
            raise ValueError("lik[c, h, :] must sum to 1")
        self.M = torch.einsum("cho,co->hc", self.lik, self.values)     # [H,C]
        self.logw = torch.log_softmax(prior.prior_logw.to(device=device, dtype=F64), -1).expand(B, -1).clone()

    def update(self, choice: Tensor, outcome: Tensor, active: Optional[Tensor] = None) -> None:
        """choice, outcome [B] long; rows with ``active`` False are left unchanged."""
        ll = torch.log(self.lik[choice.long(), :, outcome.long()].clamp_min(1e-300))   # [B,H]
        new = torch.log_softmax(self.logw + ll, -1)
        if active is not None:
            new = torch.where(active.bool().to(new.device).unsqueeze(-1), new, self.logw)
        self.logw = new

    def mean(self) -> Tensor:
        return self.logw.exp() @ self.M

    def _outcome_posteriors(self):
        w = self.logw.exp()                                            # [B,H]
        joint = w[:, None, :, None] * self.lik.unsqueeze(0)            # [B,C,H,O]
        p_o = joint.sum(2)                                             # [B,C,O]
        post = joint / p_o.unsqueeze(2).clamp_min(1e-300)              # [B,C,H,O]
        return w, p_o, post

    def knowledge_gradient(self, valid: Optional[Tensor] = None) -> Tensor:
        big = torch.finfo(F64).max / 4
        valid = torch.ones(self.B, self.C, dtype=torch.bool, device=self.logw.device) if valid is None else valid.bool()
        w, p_o, post = self._outcome_posteriors()
        mean_after = torch.einsum("bcho,hd->bcod", post, self.M)       # [B,C,O,C']
        mean_after = torch.where(valid[:, None, None, :], mean_after, torch.full_like(mean_after, -big))
        best_after = (p_o * mean_after.max(-1).values).sum(-1)         # [B,C] (p_o = 0 outcomes contribute 0)
        cur = torch.where(valid, w @ self.M, torch.full((self.B, self.C), -big, dtype=F64, device=w.device))
        kg = (best_after - cur.max(-1, keepdim=True).values).clamp_min(0)
        return torch.where(valid, kg, torch.zeros_like(kg))

    def info_gain(self) -> Tensor:
        w, p_o, post = self._outcome_posteriors()
        h = lambda q, dim: -(q * torch.log(q.clamp_min(1e-300))).sum(dim)   # noqa: E731
        h_prior = h(w, -1).unsqueeze(1)                                # [B,1]
        h_post = (p_o * h(post, 2)).sum(-1)                            # [B,C]
        return (h_prior - h_post).clamp_min(0)

    def thompson(self, generator: Optional[torch.Generator] = None) -> Tensor:
        w = self.logw.exp()
        if generator is None or generator.device == w.device:
            hs = torch.multinomial(w, 1, generator=generator).squeeze(-1)
        else:
            hs = torch.multinomial(w.to(generator.device), 1, generator=generator).squeeze(-1).to(w.device)
        return self.M[hs]                                               # [B,C]


class BanditBaseline:
    """Baseline exploration policies of spec 8.5 over C discrete candidates (no learning across tasks).

    Independent-candidate model (`CandidateValueModel`, CAM 7.0): ``random`` | ``greedy`` (posterior mean) |
    ``thompson`` (posterior draw) | ``kg`` (online knowledge gradient, argmax mean + (remaining - 1) KG,
    Ryzhov-Powell-Frazier) | ``eig`` (argmax mean + (remaining - 1) sd_noise EIG).
    Structured prior (`HypothesisValueModel`, needs ``structured``): ``greedy_structured`` | ``thompson_structured`` |
    ``kg_structured`` (online KG with correlated hypotheses: an information-only experiment has KG > 0).
    For the structured kinds `update` takes outcome *indices* (`StructuredPrior.outcome_fn`)."""

    KINDS = ("random", "greedy", "thompson", "kg", "eig")
    STRUCTURED_KINDS = ("greedy_structured", "thompson_structured", "kg_structured")

    def __init__(self, kind: str, n_candidates: int, seed: int = 0, scale: float = 1.0,
                 structured: Optional[StructuredPrior] = None, **prior):
        if kind not in self.KINDS + self.STRUCTURED_KINDS:
            raise ValueError(f"unknown baseline {kind}")
        if kind in self.STRUCTURED_KINDS:
            if structured is None:
                raise ValueError(f"{kind} needs a StructuredPrior")
            if structured.lik.shape[0] != n_candidates:
                raise ValueError("the structured prior must cover every candidate")
        prior.setdefault("b0", float(scale) ** 2)          # prior noise / value scale in the outcome's unit
        self.kind, self.C, self.prior, self.structured = kind, n_candidates, prior, structured
        self.gen = torch.Generator().manual_seed(seed)
        self.model = None

    @property
    def is_structured(self) -> bool:
        return self.kind in self.STRUCTURED_KINDS

    def reset(self, B: int) -> None:
        self.model = HypothesisValueModel(B, self.structured) if self.is_structured else \
            CandidateValueModel(B, self.C, **self.prior)

    def act(self, remaining: Tensor, valid: Optional[Tensor] = None) -> Tensor:
        assert self.model is not None, "call reset(B) first"
        B, C = self.model.B, self.C
        valid = torch.ones(B, C, dtype=torch.bool) if valid is None else valid.bool()
        if self.kind == "random":
            p = valid.double()
            return torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=self.gen).squeeze(1)
        rem = (remaining.double().clamp_min(1) - 1).unsqueeze(1)
        noise = torch.rand(B, C, generator=self.gen, dtype=torch.float64) * 1e-9      # random tie-breaking
        if self.is_structured:
            mdl: HypothesisValueModel = self.model
            if self.kind == "greedy_structured":
                score = mdl.mean()
            elif self.kind == "thompson_structured":
                score = mdl.thompson(self.gen)
            else:
                score = mdl.mean() + rem * mdl.knowledge_gradient(valid)
        else:
            s = self.model.scores(valid, self.gen).double()
            if self.kind == "greedy":
                score = s[..., 0]
            elif self.kind == "thompson":
                score = s[..., 4]
            elif self.kind == "kg":
                score = s[..., 0] + rem * s[..., 2]
            else:
                post = self.model.posterior()
                sd_noise = post.s2.sqrt()[:, :1]
                score = s[..., 0] + rem * sd_noise * s[..., 3]
        score = torch.where(valid, score + noise, torch.full_like(score, -1e300))
        return score.argmax(1)

    def update(self, choice: Tensor, outcome: Tensor, active: Optional[Tensor] = None) -> None:
        assert self.model is not None
        self.model.update(choice, outcome, active)


class DiscreteToolPolicy:
    """`meta.EpisodePolicy` that runs a `BanditBaseline` over the discrete candidates of an adapter's tools.

    Candidates: every tool without parameters, and every tool with a single bounded integer parameter expanded to one
    candidate per value (enough for the diagnostic tasks; continuous arguments are out of scope for these baselines).
    The immediate outcome of a candidate is read from the observations delivered after acting: the sum of NUMERIC
    values in ``money_unit`` reported by the chosen tool, minus its ``cost_hint``. The prior mean of a candidate is
    ``prior_mean.get(tool, 0) - cost_hint`` (e.g. the known price, plus the known expected payout of an untried arm).
    The structured kinds take a `StructuredPrior` and read each decision's outcome index with its ``outcome_fn``.
    Actions are tagged ``ActionSource.BASELINE``."""

    def __init__(self, kind: str, seed: int = 0, money_unit: str = "cents", scale: float = 100.0,
                 prior_mean: Optional[dict] = None, structured: Optional[StructuredPrior] = None):
        self.kind, self.seed, self.money_unit, self.scale = kind, seed, money_unit, scale
        self.prior_mean = dict(prior_mean or {})       # tool name -> prior expected outcome (default -cost_hint)
        self.structured = structured
        self.version = 0

    @staticmethod
    def candidates(tools: list) -> list:
        out = []
        for t in tools:
            props = (t.parameters or {}).get("properties", {}) or {}
            if not props:
                out.append((t.name, {}, float(t.cost_hint or 0.0)))
            elif len(props) == 1:
                (k, sch), = props.items()
                if sch.get("type") == "integer" and "minimum" in sch and "maximum" in sch:
                    for v in range(int(sch["minimum"]), int(sch["maximum"]) + 1):
                        out.append((t.name, {k: v}, float(t.cost_hint or 0.0)))
        if not out:
            raise ValueError("no discrete candidates in the tool list")
        return out

    def initial_state(self, B: int) -> dict:
        return {"cands": None, "prev": None, "base": None}

    def act(self, state, obs, tools, budget, goals, active, greedy=False):
        from ..interfaces import Action, ActionSource, ObsKind
        B = len(obs)
        if state["cands"] is None:
            state["cands"] = self.candidates(tools[0])
            if self.structured is not None and [(n, a) for n, a, _ in state["cands"]] != \
                    [(n, dict(a)) for n, a in self.structured.names]:
                raise ValueError("the structured prior's candidates do not match the tool list")
            base = BanditBaseline(self.kind, len(state["cands"]), seed=self.seed, scale=self.scale,
                                  structured=self.structured,
                                  prior_mean=[self.prior_mean.get(n, 0.0) - c for n, _, c in state["cands"]])
            base.reset(B)
            state["base"] = base
        cands, base = state["cands"], state["base"]
        act_now = active.detach().cpu().bool()
        if state["prev"] is None:
            prev, pend = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.bool)
        else:
            prev, pend = state["prev"]
            upd = pend & act_now           # the outcome of a row's last decision is delivered when it acts again
            if bool(upd.any()):
                if base.is_structured:
                    out = torch.tensor([int(self.structured.outcome_fn(int(prev[b]), obs[b])) if bool(upd[b])
                                        else 0 for b in range(B)], dtype=torch.long)
                else:
                    out = torch.zeros(B, dtype=F64)
                    for b in range(B):
                        name, _, cost = cands[int(prev[b])]
                        val = sum(float(o.value) for o in obs[b] if o.kind == ObsKind.NUMERIC
                                  and o.unit == self.money_unit and o.source == name and o.value is not None)
                        out[b] = val - cost
                base.update(prev, out, upd)
            pend = pend & ~upd
        rem = torch.nan_to_num(budget.decisions_left.detach().cpu().double(), nan=1.0)
        choice = base.act(rem)
        state["prev"] = (torch.where(act_now, choice, prev), pend | act_now)
        actions = [Action(cands[int(c)][0], dict(cands[int(c)][1]), source=ActionSource.BASELINE)
                   for c in choice.tolist()]
        z = torch.zeros(B)
        return actions, z, z, state, {}
