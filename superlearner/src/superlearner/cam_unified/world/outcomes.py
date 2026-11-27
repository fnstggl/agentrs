"""Typed outcome prediction (spec 6.2): ``OutcomeHead`` implements ``OutcomeHeadP``; ``MixedOutcomeDist`` implements
``OutcomeDist`` for every ``Likelihood``.

Queries are QUERY tokens (they encode outcome type, unit, entity and horizon; built by workstream A's tokenizer).
``features(Z, outcomes)`` lets every query cross-attend onto the latent state (or any context token set) and returns
one feature vector per query. ``dist(feat, outcomes, cond)`` maps features to the parameters of the query's own
likelihood family; ``cond`` (belief particles ``[B,K,Dc]``) modulates the features by FiLM, giving a batch of
``[B,K,Q]`` particle-conditioned predictive distributions (used by the belief module's importance weights).

Likelihood families and parameterisations (all numerically stable, float32):

* GAUSSIAN   mean mu = symexp(r0) (reaches large raw values, e.g. cents), std = (softplus(r1)+min_std)(1+|mu|^)
             where |mu|^ is the detached magnitude (relative-precision parameterisation).
* STUDENT_T  loc/scale as Gaussian, df = min_df + softplus(r2 + 2) (min_df > 2 keeps the variance finite).
* BERNOULLI  logit r0.
* CATEGORICAL ``max_classes`` logits, classes >= n_classes masked (n_classes per query). A categorical query with
             n_classes > max_classes (or < 1) cannot be represented and raises ValueError; so does an observed class
             index outside [0, n_classes) (or a non-integer one) -- nothing is silently merged or clipped.
* COUNT      negative binomial with mean m = exp(lm), lm = softbound(r0; -12, count_log_max) and total_count
             (dispersion) r = exp(softbound(r1 + 1; -6, 12)); var = m + m^2/r. softbound(x; lo, hi) =
             c + h tanh((x - c) / h) (c, h = centre, half-width) has slope 1 at the centre and a nonzero gradient
             everywhere (no dead hard clamp). The log-probability is evaluated in float64 (lgamma of large counts).
             Observed counts must be non-negative integers <= exp(count_log_max) (ValueError otherwise).
* SURVIVAL   discrete hazard over M horizon bins with lower edges ``edges`` (days, edges[0] = 0); bin j covers
             [e_j, e_{j+1}), the last bin [e_{M-1}, inf) has hazard 1 (no event within the horizon).
             Uncensored event time y in bin j: log P_j = log h_j + sum_{i<j} log(1-h_i).
             Censored at c in bin j (known T > c): log S_j = sum_{i<j} log(1-h_i) (at risk through the start of the
             censoring bin; the censoring bin itself is not counted as survived).
             ``log_prob`` is the probability mass of the bin (a discrete-time likelihood), not a density.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CamConfig
from ..interfaces import Likelihood, OutcomeBatch, TokenSet
from ._util import MLP, TimeEncoder, masked_mean, safe_key_padding, seeded_from, symexp

_LOG_2PI = math.log(2 * math.pi)
_NEG = -1e4      # finite "minus infinity" for masked logits (keeps gradients finite)


def softbound(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth bound into (lo, hi): c + h * tanh((x - c) / h), slope 1 at the centre c, gradient nonzero everywhere."""
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return c + h * torch.tanh((x - c) / h)


def default_survival_edges(n_bins: int = 16, first: float = 1.0 / 24.0, last: float = 180.0) -> torch.Tensor:
    """Lower bin edges in days: 0, then geometric from ``first`` to ``last`` (n_bins edges in total)."""
    return torch.cat([torch.zeros(1), torch.logspace(math.log10(first), math.log10(last), n_bins - 1)])


class MixedOutcomeDist:
    """A batch of typed predictive distributions, one per (row, [particle,] query).

    raw [..., P] unconstrained head outputs; lik [...] long; n_classes [...] long. Batch shape ``raw.shape[:-1]``
    ([B,Q] or [B,K,Q]). Only the families present in ``lik`` are evaluated (parameters are computed lazily);
    within a family all entries are evaluated on sanitised inputs and each entry's own family is selected, so
    gradients stay finite everywhere.
    """

    _SLICES = ("g", 2), ("t", 3), ("b", 1), ("c", None), ("n", 2), ("s", None)

    def __init__(self, raw: torch.Tensor, lik: torch.Tensor, n_classes: torch.Tensor, edges: torch.Tensor,
                 max_classes: int, min_df: float = 2.1, min_std: float = 1e-3, nb_support: int = 256,
                 count_log_max: float = 20.0, validate: bool = True):
        raw = raw.float()
        self.raw = raw
        self.batch_shape = raw.shape[:-1]
        self.lik = lik.expand(self.batch_shape).long()
        self.n_classes = n_classes.expand(self.batch_shape).long()
        self.edges = edges.to(raw.device).float()
        self.M = self.edges.numel()
        self.Cm = max_classes
        self.min_df, self.min_std, self.nb_support = min_df, min_std, nb_support
        self.count_log_max, self.validate = float(count_log_max), validate
        self.present = sorted(int(v) for v in torch.unique(self.lik).tolist())
        if validate and int(Likelihood.CATEGORICAL) in self.present:
            nc = self.n_classes[self.lik == int(Likelihood.CATEGORICAL)]
            if bool((nc > max_classes).any()) or bool((nc < 1).any()):
                raise ValueError(f"categorical queries need 1 <= n_classes <= max_classes={max_classes}; got "
                                 f"n_classes in [{int(nc.min())}, {int(nc.max())}] (raise the head's max_classes)")
        sizes = {"g": 2, "t": 3, "b": 1, "c": max_classes, "n": 2, "s": self.M}
        self._off, i = {}, 0
        for k in ("g", "t", "b", "c", "n", "s"):
            self._off[k] = (i, i + sizes[k])
            i += sizes[k]
        if i != raw.shape[-1]:
            raise ValueError(f"expected {i} raw parameters, got {raw.shape[-1]}")

    def _raw(self, k):
        a, b = self._off[k]
        return self.raw[..., a:b]

    def _has(self, lk) -> bool:
        return int(lk) in self.present

    # ------------------------------------------------------------------ constrained parameters (lazy)
    @cached_property
    def g_mu(self):
        return symexp(self._raw("g")[..., 0])

    @cached_property
    def g_std(self):
        return (F.softplus(self._raw("g")[..., 1]) + self.min_std) * (1.0 + self.g_mu.detach().abs())

    @cached_property
    def t_loc(self):
        return symexp(self._raw("t")[..., 0])

    @cached_property
    def t_scale(self):
        return (F.softplus(self._raw("t")[..., 1]) + self.min_std) * (1.0 + self.t_loc.detach().abs())

    @cached_property
    def t_df(self):
        return self.min_df + F.softplus(self._raw("t")[..., 2] + 2.0)

    @cached_property
    def b_logit(self):
        return self._raw("b")[..., 0]

    @cached_property
    def n_eff(self):
        """Number of classes per entry (>= 1 so non-categorical entries stay finite)."""
        return self.n_classes.clamp(min=1, max=self.Cm)

    @cached_property
    def c_valid(self):
        return torch.arange(self.Cm, device=self.raw.device) < self.n_eff.unsqueeze(-1)

    @cached_property
    def c_logp(self):
        c = self._raw("c")
        return F.log_softmax(torch.where(self.c_valid, c, torch.full_like(c, _NEG)), -1)

    @cached_property
    def n_logm(self):
        return softbound(self._raw("n")[..., 0], -12.0, self.count_log_max)

    @cached_property
    def n_logr(self):
        return softbound(self._raw("n")[..., 1] + 1.0, -6.0, 12.0)

    @cached_property
    def _surv(self):
        s = self._raw("s")
        last = torch.zeros_like(s, dtype=torch.bool)
        last[..., -1] = True
        log_h = torch.where(last, torch.zeros_like(s), -F.softplus(-s))          # log h_j (last bin: h = 1)
        log_1mh = torch.where(last, torch.full_like(s, _NEG), -F.softplus(s))    # log(1 - h_j)
        cum = torch.cumsum(log_1mh, -1)
        logS = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], -1)    # log P(T >= e_j)
        return logS, log_h + logS                                                # (log S_j, log P_j)

    @property
    def s_logS(self):
        return self._surv[0]

    @property
    def s_logP(self):
        return self._surv[1]

    @property
    def bern_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.b_logit)

    @property
    def nb_mean(self) -> torch.Tensor:
        return self.n_logm.exp()

    @property
    def nb_total_count(self) -> torch.Tensor:
        return self.n_logr.exp()

    @property
    def cat_probs(self) -> torch.Tensor:
        """[..., max_classes] (zero beyond n_classes)."""
        return self.c_logp.exp() * self.c_valid

    @property
    def surv_bin_probs(self) -> torch.Tensor:
        """[..., M] probability mass of each horizon bin."""
        return self.s_logP.exp()

    def survival_bin(self, y: torch.Tensor) -> torch.Tensor:
        """Bin index of event/censoring time y (days)."""
        yv = y.float().clamp(min=0.0)
        return (torch.searchsorted(self.edges, yv.contiguous(), right=True) - 1).clamp(0, self.M - 1)

    def _bc(self, t, dtype=None) -> torch.Tensor:
        """Broadcast a [B,Q] tensor to the batch shape ([B,K,Q] inserts the particle axis)."""
        t = torch.as_tensor(t, device=self.raw.device)
        if dtype is not None:
            t = t.to(dtype)
        if t.dim() == len(self.batch_shape) - 1 and len(self.batch_shape) >= 3:
            t = t.unsqueeze(1)
        return t.expand(self.batch_shape)

    def _sel(self, fns: dict) -> torch.Tensor:
        """Evaluate only the present families (dict Likelihood -> zero-arg callable) and select per entry."""
        out = None
        for lk in self.present:
            v = fns[Likelihood(lk)]()
            out = v if out is None else torch.where(self.lik == lk, v, out)
        return out

    def _bin_mid(self) -> torch.Tensor:
        e = self.edges
        mid = torch.empty_like(e)
        mid[:-1] = 0.5 * (e[:-1] + e[1:])
        mid[-1] = e[-1]
        return mid

    # ------------------------------------------------------------------ OutcomeDist protocol
    def log_prob(self, y, censored=None, mask=None) -> torch.Tensor:
        """y broadcastable to the batch shape (value / class index / count / event time). ``censored`` (SURVIVAL)
        broadcastable bool. ``mask`` (optional) marks the observed entries: the others get 0 and are not validated
        (without a mask every entry is treated as observed)."""
        y = self._bc(y, torch.float32)

        def gauss():
            z = (y - self.g_mu) / self.g_std
            return -0.5 * z * z - torch.log(self.g_std) - 0.5 * _LOG_2PI

        def student():
            df = self.t_df
            u = (y - self.t_loc) / self.t_scale
            return (torch.lgamma(0.5 * (df + 1)) - torch.lgamma(0.5 * df) - 0.5 * torch.log(df * math.pi)
                    - torch.log(self.t_scale) - 0.5 * (df + 1) * torch.log1p(u * u / df))

        def bern():
            yb = y.clamp(0.0, 1.0)
            return yb * self.b_logit - F.softplus(self.b_logit)

        obs = None if mask is None else self._bc(mask, torch.bool)

        def checked(lk) -> torch.Tensor:
            sel = self.lik == int(lk)
            return sel if obs is None else sel & obs

        def cat():
            if self.validate:
                chk = checked(Likelihood.CATEGORICAL)
                bad = chk & ((y < 0) | (y >= self.n_classes) | ((y - y.round()).abs() > 1e-3))
                if bool(bad.any()):
                    raise ValueError(f"observed categorical class {float(y[bad][0])} outside [0, "
                                     f"{int(self.n_classes[bad][0])}) or not an integer")
            yc = torch.minimum(y.round().clamp(min=0).long(), self.n_eff - 1)
            return self.c_logp.gather(-1, yc.unsqueeze(-1)).squeeze(-1)

        def count():
            if self.validate:
                chk = checked(Likelihood.COUNT)
                bad = chk & ((y < 0) | (y > math.exp(self.count_log_max)) | ((y - y.round()).abs() > 1e-3))
                if bool(bad.any()):
                    raise ValueError(f"observed count {float(y[bad][0])} is not an integer in [0, "
                                     f"exp({self.count_log_max:g})] (outside the representable range)")
            k = y.clamp(min=0.0).round().double()
            lm, lr = self.n_logm.double(), self.n_logr.double()
            r = lr.exp()
            log_rm = torch.logaddexp(lr, lm)
            lp = torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1) + r * (lr - log_rm) + k * (lm - log_rm)
            return lp.float()

        def surv():
            j = self.survival_bin(y).unsqueeze(-1)
            lp_ev = self.s_logP.gather(-1, j).squeeze(-1)
            if censored is None:
                return lp_ev
            lp_ce = self.s_logS.gather(-1, j).squeeze(-1)
            return torch.where(self._bc(censored, torch.bool), lp_ce, lp_ev)

        lp = self._sel({Likelihood.GAUSSIAN: gauss, Likelihood.STUDENT_T: student, Likelihood.BERNOULLI: bern,
                        Likelihood.CATEGORICAL: cat, Likelihood.COUNT: count, Likelihood.SURVIVAL: surv})
        if obs is not None:
            lp = torch.where(obs, lp, torch.zeros_like(lp))
        return lp

    def mean(self) -> torch.Tensor:
        return self._sel({
            Likelihood.GAUSSIAN: lambda: self.g_mu, Likelihood.STUDENT_T: lambda: self.t_loc,
            Likelihood.BERNOULLI: lambda: self.bern_probs,
            Likelihood.CATEGORICAL: lambda: (self.cat_probs * torch.arange(self.Cm, device=self.raw.device)).sum(-1),
            Likelihood.COUNT: lambda: self.nb_mean,
            Likelihood.SURVIVAL: lambda: (self.surv_bin_probs * self._bin_mid()).sum(-1)})

    def variance(self) -> torch.Tensor:
        def cat():
            cls = torch.arange(self.Cm, device=self.raw.device).float()
            pc = self.cat_probs
            cm = (pc * cls).sum(-1)
            return ((pc * cls * cls).sum(-1) - cm * cm).clamp(min=0.0)

        def surv():
            mid = self._bin_mid()
            ps = self.surv_bin_probs
            sm = (ps * mid).sum(-1)
            return ((ps * mid * mid).sum(-1) - sm * sm).clamp(min=0.0)

        return self._sel({
            Likelihood.GAUSSIAN: lambda: self.g_std ** 2,
            Likelihood.STUDENT_T: lambda: self.t_scale ** 2 * self.t_df / (self.t_df - 2.0),
            Likelihood.BERNOULLI: lambda: self.bern_probs * (1 - self.bern_probs), Likelihood.CATEGORICAL: cat,
            Likelihood.COUNT: lambda: self.nb_mean + self.nb_mean ** 2 / self.nb_total_count,
            Likelihood.SURVIVAL: surv})

    def entropy(self) -> torch.Tensor:
        def student():
            df = self.t_df
            lbeta = torch.lgamma(0.5 * df) + math.lgamma(0.5) - torch.lgamma(0.5 * (df + 1))
            return (0.5 * (df + 1) * (torch.digamma(0.5 * (df + 1)) - torch.digamma(0.5 * df))
                    + 0.5 * torch.log(df) + lbeta + torch.log(self.t_scale))

        def bern():
            p = self.bern_probs
            return p * F.softplus(-self.b_logit) + (1 - p) * F.softplus(self.b_logit)

        return self._sel({
            Likelihood.GAUSSIAN: lambda: 0.5 * (1.0 + _LOG_2PI) + torch.log(self.g_std),
            Likelihood.STUDENT_T: student, Likelihood.BERNOULLI: bern,
            Likelihood.CATEGORICAL: lambda: -(self.cat_probs * self.c_logp).sum(-1),
            Likelihood.COUNT: self._nb_entropy,
            Likelihood.SURVIVAL: lambda: -(self.surv_bin_probs * self.s_logP).sum(-1)})

    def _nb_entropy(self) -> torch.Tensor:
        """Exact truncated sum over k < nb_support; Gaussian approximation where the mass extends beyond it."""
        K = self.nb_support
        k = torch.arange(K, device=self.raw.device).float()
        r = self.nb_total_count.unsqueeze(-1)
        lm, lr = self.n_logm.unsqueeze(-1), self.n_logr.unsqueeze(-1)
        log_rm = torch.logaddexp(lr, lm)
        lp = torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1) + r * (lr - log_rm) + k * (lm - log_rm)
        exact = -(lp.exp() * lp).sum(-1)
        m = self.nb_mean
        var = m + m * m / self.nb_total_count
        approx = 0.5 * torch.log(2 * math.pi * math.e * var.clamp(min=1e-6))
        heavy = (m + 6 * var.sqrt()) > (K - 1)
        return torch.where(heavy, approx, exact)

    @torch.no_grad()
    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        dev = self.raw.device
        shp = self.batch_shape

        def gauss():
            return self.g_mu + self.g_std * torch.randn(shp, generator=generator, device=dev)

        def student():
            eps = torch.randn(shp, generator=generator, device=dev)
            with seeded_from(generator, dev):
                chi2 = torch.distributions.Chi2(self.t_df).sample()
            return self.t_loc + self.t_scale * eps / torch.sqrt(chi2 / self.t_df).clamp(min=1e-6)

        def bern():
            return (torch.rand(shp, generator=generator, device=dev) < self.bern_probs).float()

        def cat():
            pc = self.cat_probs.reshape(-1, self.Cm).clamp(min=0)
            return torch.multinomial(pc, 1, generator=generator).reshape(shp).float()

        def count():
            m, r = self.nb_mean, self.nb_total_count
            with seeded_from(generator, dev):
                lam = torch.distributions.Gamma(r, r / m.clamp(min=1e-8)).sample()
            # the gamma mixing draw of a very over-dispersed NB can overflow; draws are truncated to the representable
            # range [0, exp(count_log_max)] the log-probability accepts (a simulated count above it -- e.g. the
            # surprise simulation of beliefs.surprise_z on an early, very over-dispersed head -- otherwise raised)
            cmax = float(math.floor(math.exp(self.count_log_max)))
            lam = torch.nan_to_num(lam, nan=0.0, posinf=cmax).clamp(max=cmax)
            return torch.poisson(lam, generator=generator).clamp(max=cmax)

        def surv():
            ps = self.surv_bin_probs.reshape(-1, self.M).clamp(min=0)
            jb = torch.multinomial(ps, 1, generator=generator).reshape(shp)
            e = self.edges
            lo = e[jb]
            hi = torch.cat([e[1:], e[-1:]])[jb]
            return lo + torch.rand(shp, generator=generator, device=dev) * (hi - lo)

        return self._sel({Likelihood.GAUSSIAN: gauss, Likelihood.STUDENT_T: student, Likelihood.BERNOULLI: bern,
                          Likelihood.CATEGORICAL: cat, Likelihood.COUNT: count, Likelihood.SURVIVAL: surv})

    # ------------------------------------------------------------------ helpers
    def nll(self, outcomes: OutcomeBatch) -> torch.Tensor:
        """Masked mean negative log-likelihood of the observed outcomes (scalar)."""
        m = self._bc(outcomes.y_mask, torch.bool)
        lp = self.log_prob(outcomes.y, outcomes.censored, m)
        return -masked_mean(lp, m)


class OutcomeHead(nn.Module):
    """Type-conditioned distributional heads over variable outcome queries (implements ``OutcomeHeadP``).

    features(Z [B,N,D] | TokenSet, outcomes) -> feat [B,Q,D]  (latent-state inputs [B,S,D] are standardised by the
        owning WorldModel's running StateNorm when attached; other contexts pass through)
    dist(feat [B,Q,D], outcomes, cond=None | [B,K,Dc]) -> MixedOutcomeDist with batch [B,Q] | [B,K,Q]
    """

    def __init__(self, cfg: CamConfig, cond_dim: Optional[int] = None, n_layers: int = 1, max_classes: int = 32,
                 survival_edges: Optional[torch.Tensor] = None, min_df: float = 2.1, min_std: float = 1e-3,
                 nb_support: int = 256, count_log_max: float = 20.0, validate: bool = True,
                 film_init_std: Optional[float] = None):
        """film_init_std: initial std of the particle-conditioning FiLM weights (0 = exact zero init: at
        initialisation every particle predicts the same outcome distribution, and the mixture's first-order gradient
        moves all particles' predictions together - the symmetric start does not break by itself; a small positive
        value starts the hypotheses slightly apart). None = 1/sqrt(cond_dim) for the DOM presets (plan 2026-11-26
        §1 BeliefModule: break the symmetric dead start, measured act_cond gradient 3e-7 at 0), 0 otherwise."""
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.max_classes = max_classes
        self.min_df, self.min_std, self.nb_support = min_df, min_std, nb_support
        self.count_log_max, self.validate = count_log_max, validate
        edges = default_survival_edges() if survival_edges is None else torch.as_tensor(survival_edges).float()
        if edges[0] != 0 or bool((edges[1:] <= edges[:-1]).any()):
            raise ValueError("survival edges must start at 0 and increase strictly")
        self.register_buffer("edges", edges)
        self.lik_emb = nn.Embedding(len(Likelihood), D)
        self.q_time = TimeEncoder(D, cfg.n_fourier)
        self.q_in = nn.Linear(D, D)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "nq": nn.LayerNorm(D), "nk": nn.LayerNorm(D),
                "att": nn.MultiheadAttention(D, cfg.n_heads, batch_first=True),
                "ff": MLP(D, 2 * D, D)}))
        self.out_norm = nn.LayerNorm(D)
        self.cond_dim = cond_dim or cfg.d_particle
        self.film = nn.Linear(self.cond_dim, 2 * D)
        if film_init_std is None:
            film_init_std = self.cond_dim ** -0.5 if str(getattr(cfg, "preset", "") or "").startswith("dom") else 0.0
        if film_init_std > 0:
            nn.init.normal_(self.film.weight, std=film_init_std)
        else:
            nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.cond_mix = MLP(D, 2 * D, D, zero_out=True)
        self.state_norm = None          # set by the WorldModel: running standardisation of latent-state inputs
        self.n_params = 2 + 3 + 1 + max_classes + 2 + edges.numel()
        self.param = nn.Linear(D, self.n_params)
        with torch.no_grad():
            self.param.weight.mul_(0.1)
            self.param.bias.zero_()

    def features(self, Z, outcomes: OutcomeBatch, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Z: latent state [B,S,D] (or any context: tensor [B,N,D] with optional ``mask`` [B,N], or a TokenSet).
        Returns per-query features [B,Q,D] (float32)."""
        if isinstance(Z, TokenSet):
            mask = Z.mask if mask is None else mask
            Z = Z.x
        Z = Z.float()
        if self.state_norm is not None:
            Z = self.state_norm(Z)
        q = outcomes.q
        h = self.q_in(q.x.float()) + self.lik_emb(outcomes.lik.long().clamp(0, len(Likelihood) - 1)) \
            + self.q_time(q.time.float())
        kpm = None if mask is None else safe_key_padding(mask)
        for blk in self.layers:
            kv = blk["nk"](Z)
            a, _ = blk["att"](blk["nq"](h), kv, kv, key_padding_mask=kpm, need_weights=False)
            h = h + a
            h = h + blk["ff"](h)
        return self.out_norm(h)

    def dist(self, feat, outcomes: OutcomeBatch, cond=None) -> MixedOutcomeDist:
        lik, ncls = outcomes.lik.long(), outcomes.n_classes.long()
        if cond is not None:
            gam, bet = self.film(cond.float()).chunk(2, -1)                         # [B,K,D]
            feat = feat.unsqueeze(1) * (1 + gam.unsqueeze(2)) + bet.unsqueeze(2)    # [B,K,Q,D]
            feat = feat + self.cond_mix(feat)
            lik, ncls = lik.unsqueeze(1), ncls.unsqueeze(1)
        raw = self.param(feat)
        return MixedOutcomeDist(raw, lik, ncls, self.edges, self.max_classes, self.min_df, self.min_std,
                                self.nb_support, self.count_log_max, self.validate)

    def nll(self, Z, outcomes: OutcomeBatch, cond=None, mask=None) -> torch.Tensor:
        d = self.dist(self.features(Z, outcomes, mask), outcomes, cond)
        return d.nll(outcomes)
