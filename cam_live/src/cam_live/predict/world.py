"""cam_live.predict.world — action-conditioned world model heads  p(y | z, q, horizon)  (spec Part 4).

``WorldHeads`` maps the decision latent ``z`` [B, N, D] of every candidate action and the outcome-query embeddings
``e_q`` to distribution parameters of the query's *kind* (real -> Student-t / normal, count -> negative binomial
with exposure, binary -> Bernoulli, time-to-event -> discrete hazard with right-censoring, categorical). The
horizon is an input, so one head serves every horizon.

Uncertainty. The heads are a small **bootstrapped ensemble with randomised prior functions** (Osband et al. 2018):
M members with independent initialisation, each added to its own frozen random prior network and trained on its
own bootstrap resample (:func:`bootstrap_weights`). ``epistemic`` = spread of the members' predictive means
(reducible: it shrinks with data and stays large away from the data); ``aleatoric`` = mean of the members'
predictive variances. The Bayesian fast layer of Workstream A adds a second, within-context epistemic estimate; the
two are combined by the caller (stage 2), never silently merged here.

Training is :func:`world_nll`: exact negative log-likelihood of *observed* outcomes only. Pending and unobservable
targets are masked and contribute exactly zero loss and zero gradient; each kind is averaged separately and divided
by a running scale so that no kind (e.g. large counts) dominates the shared trunk.

``LatentTransition`` predicts the next belief ``b'`` from ``(z, dt)`` with its own ensemble and a diagonal
aleatoric variance; it is the one-step model used by :mod:`cam_live.predict.imagination`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import Distribution, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus
from . import dists as D

#: same order as ``cam_live.nn.api.KIND_INDEX`` (asserted in the tests; not imported so this module stays light)
KIND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(OutcomeKind.ALL)}
K_REAL, K_COUNT, K_BINARY, K_TTE, K_CAT = (KIND_INDEX[k] for k in OutcomeKind.ALL)
N_KINDS = len(OutcomeKind.ALL)


def time_features(dt_s: Tensor, dim: int) -> Tensor:
    """[..., dim] smooth features of an elapsed time in seconds (log-time, sinusoids of log-time)."""
    x = torch.log1p(dt_s.clamp_min(0.0).to(torch.float64))            # 0 .. ~20 for a year
    n_freq = max(1, (dim - 2) // 2)
    freqs = torch.exp(torch.linspace(math.log(0.15), math.log(3.0), n_freq, dtype=torch.float64, device=x.device))
    ang = x.unsqueeze(-1) * freqs
    feats = torch.cat([(x / 10.0).unsqueeze(-1), ((x / 10.0) ** 2).unsqueeze(-1), torch.sin(ang), torch.cos(ang)], -1)
    if feats.shape[-1] < dim:
        feats = F.pad(feats, (0, dim - feats.shape[-1]))
    return feats[..., :dim].to(torch.float32)


class EnsembleLinear(nn.Module):
    """M independent linear maps evaluated in one einsum: x [M, ..., in] -> [M, ..., out]."""

    def __init__(self, n_members: int, d_in: int, d_out: int, std: Optional[float] = None, trainable: bool = True,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        std = (1.0 / math.sqrt(d_in)) if std is None else std
        w = torch.randn(n_members, d_in, d_out, generator=generator) * std
        b = torch.zeros(n_members, d_out)
        if trainable:
            self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)
        else:
            self.register_buffer("weight", w)
            self.register_buffer("bias", b)

    def forward(self, x: Tensor) -> Tensor:
        m = self.weight.shape[0]
        flat = x.reshape(m, -1, x.shape[-1])
        out = torch.baddbmm(self.bias.unsqueeze(1), flat, self.weight)
        return out.reshape(x.shape[:-1] + (self.weight.shape[-1],))


class _EnsembleMLP(nn.Module):
    def __init__(self, n_members: int, d_in: int, hidden: int, d_out: int, out_std: float, trainable: bool = True,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.l1 = EnsembleLinear(n_members, d_in, hidden, trainable=trainable, generator=generator)
        self.l2 = EnsembleLinear(n_members, hidden, hidden, trainable=trainable, generator=generator)
        self.out = EnsembleLinear(n_members, hidden, d_out, std=out_std, trainable=trainable, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(F.silu(self.l2(F.silu(self.l1(x)))))


@dataclass
class WorldConfig:
    d_model: int
    n_features: int = 0              # width of phi; 0 = phi is not an input
    hidden: int = 192
    n_members: int = 5
    n_hazard_bins: int = 12
    max_categories: int = 16
    d_time: int = 16
    real_family: str = D.FAMILY_STUDENT_T      # or "normal"
    prior_scale: float = 1.0         # weight of the frozen randomised prior functions (0 disables them)
    prior_hidden: int = 64
    min_scale: float = 1e-3
    max_log_rate: float = 20.0
    seed: int = 0


@dataclass
class WorldTargets:
    """Matured outcomes for one row of queries per decision. Only ``observed`` / ``censored`` entries are ever
    used; everything else (pending, unobservable, padding) is masked."""
    y: Tensor                        # [B, Q] float64 value (count, 0/1, elapsed seconds, class index, real)
    observed: Tensor                 # [B, Q] bool — OutcomeStatus.OBSERVED
    censored: Tensor                 # [B, Q] bool — OutcomeStatus.CENSORED (time-to-event only)

    def mask(self, kind_ids: Tensor) -> Tensor:
        return self.observed | (self.censored & (kind_ids == K_TTE))


@dataclass
class QueryMeta:
    kind_ids: Tensor                 # [B, Q] long
    horizon_s: Tensor                # [B, Q] float64
    exposure: Tensor                 # [B, Q] float64
    n_categories: Tensor             # [B, Q] long
    q_mask: Tensor                   # [B, Q] bool


def build_targets(rows: Sequence[Sequence[Tuple[OutcomeQuery, Optional[OutcomeObservation]]]],
                  device: Optional[torch.device] = None) -> Tuple[QueryMeta, WorldTargets]:
    """Tensors from contract records. A query with no observation, a pending one or an unobservable one is
    masked — never filled in. Imagined values are refused. Values and exposures are kept in float64 (exact for
    counts up to 2**53 and for precise real measurements); the likelihoods cast them to the head's dtype."""
    from .imagination import refuse_imagined
    b, q = len(rows), max((len(r) for r in rows), default=0)
    q = max(q, 1)
    kind = torch.zeros(b, q, dtype=torch.long)
    hor = torch.zeros(b, q, dtype=torch.float64)
    expo = torch.ones(b, q, dtype=torch.float64)
    ncat = torch.ones(b, q, dtype=torch.long)
    qm = torch.zeros(b, q, dtype=torch.bool)
    y = torch.zeros(b, q, dtype=torch.float64)
    obs = torch.zeros(b, q, dtype=torch.bool)
    cen = torch.zeros(b, q, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, (query, o) in enumerate(row):
            kind[i, j], hor[i, j], qm[i, j] = KIND_INDEX[query.kind], query.horizon_s, True
            expo[i, j] = max(float(query.exposure), 1e-12)
            ncat[i, j] = max(len(query.categories), 1)
            if o is None:
                continue
            refuse_imagined(o)
            if o.query_id != query.query_id:
                raise ValueError(f"observation {o.query_id!r} does not answer query {query.query_id!r}")
            if o.value is None or o.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                continue
            if o.status == OutcomeStatus.CENSORED and query.kind != OutcomeKind.TIME_TO_EVENT:
                continue
            y[i, j] = float(o.value)
            obs[i, j] = o.status == OutcomeStatus.OBSERVED
            cen[i, j] = o.status == OutcomeStatus.CENSORED
    meta = QueryMeta(kind, hor, expo, ncat, qm)
    tgt = WorldTargets(y, obs, cen)
    if device is not None:
        meta = QueryMeta(*(t.to(device) for t in (kind, hor, expo, ncat, qm)))
        tgt = WorldTargets(y.to(device), obs.to(device), cen.to(device))
    return meta, tgt


def bootstrap_weights(keys: Sequence[str], n_members: int, seed: int = 0) -> Tensor:
    """[M, B] Poisson(1) bootstrap multiplicities, a deterministic function of (key, member, seed): a stored
    record keeps the same membership every time it is replayed, which is what makes the members differ."""
    cdf = [math.exp(-1.0) * sum(1.0 / math.factorial(j) for j in range(k + 1)) for k in range(12)]
    out = torch.zeros(n_members, len(keys))
    for bi, key in enumerate(keys):
        for m in range(n_members):
            h = hashlib.sha256(f"{seed}|{m}|{key}".encode()).digest()
            u = int.from_bytes(h[:8], "big") / 2.0 ** 64
            out[m, bi] = float(sum(1 for c in cdf if u > c))
    return out


class WorldOutput:
    """Per-member distribution parameters for every (candidate, query). Shapes are [M, B, N, Q, ...]."""

    def __init__(self, raw: Tensor, cfg: WorldConfig, kind_ids: Tensor, horizon_s: Tensor, exposure: Tensor,
                 n_categories: Tensor, real_loc: Tensor, real_scale: Tensor):
        self.cfg, self.raw = cfg, raw
        self.kind_ids, self.horizon_s, self.exposure, self.n_categories = kind_ids, horizon_s, exposure, n_categories
        self.real_loc, self.real_scale = real_loc, real_scale
        k, c = cfg.n_hazard_bins, cfg.max_categories
        o = raw
        # real (in normalised units; ``real_loc`` / ``real_scale`` map back to the metric's units)
        self.loc = o[..., 0]
        self.scale = F.softplus(o[..., 1] + 0.5413) + cfg.min_scale          # softplus(0.5413) = 1
        self.df = 2.0 + F.softplus(o[..., 2] + 3.0)                           # > 2: finite variance, ~5 at init
        # count
        self.log_rate = o[..., 3].clamp(-cfg.max_log_rate, cfg.max_log_rate)
        self.log_conc = (o[..., 4] + 2.0).clamp(-6.0, 9.0)
        # binary
        self.logit = o[..., 5]
        # time-to-event: hazards on K equal bins of [0, horizon]
        self.hazard_logits = o[..., 6:6 + k] - 2.0
        # categorical
        self.cat_logits = o[..., 6 + k:6 + k + c]

    # ---- helpers ----------------------------------------------------------------------------------------
    @property
    def n_members(self) -> int:
        return self.raw.shape[0]

    def _bq(self, t: Tensor) -> Tensor:
        """[B, Q] -> broadcastable against [M, B, N, Q]."""
        return t[None, :, None, :]

    def _edges(self) -> Tensor:
        h = self.horizon_s.to(self.raw.dtype).clamp_min(1e-6)
        return D.uniform_edges(h, self.cfg.n_hazard_bins)[None, :, None, :, :]      # [1, B, 1, Q, K+1]

    def _kind(self) -> Tensor:
        return self._bq(self.kind_ids)

    def _select(self, real: Tensor, count: Tensor, binary: Tensor, tte: Tensor, cat: Tensor) -> Tensor:
        kind = self._kind()
        out = torch.where(kind == K_REAL, real, torch.zeros_like(real))
        for kid, v in ((K_COUNT, count), (K_BINARY, binary), (K_TTE, tte), (K_CAT, cat)):
            out = torch.where(kind == kid, v, out)
        return out

    # ---- likelihood -------------------------------------------------------------------------------------
    def log_prob(self, y: Tensor, event: Optional[Tensor] = None) -> Tensor:
        """[M, B, N, Q] log-likelihood of ``y`` [B, Q] (or [B, N, Q]) under every member and candidate. Real
        targets are scored in the query's normalised units. Entries of the wrong kind never produce NaN."""
        kind = self._kind()
        yb = y[None, :, None, :] if y.dim() == 2 else y[None]
        yb = yb.to(self.raw.dtype)
        ev = torch.ones_like(yb, dtype=torch.bool) if event is None else (
            event[None, :, None, :] if event.dim() == 2 else event[None])
        zeros = torch.zeros_like(yb)
        y_real = (torch.where(kind == K_REAL, yb, zeros) - self._bq(self.real_loc)) / self._bq(self.real_scale)
        y_real = torch.where(kind == K_REAL, y_real, zeros)
        if self.cfg.real_family == D.FAMILY_NORMAL:
            lp_real = D.normal_log_prob(y_real, self.loc, self.scale)
        else:
            lp_real = D.student_t_log_prob(y_real, self.loc, self.scale, self.df)
        y_count = torch.where(kind == K_COUNT, yb, zeros).clamp_min(0.0)
        log_mu = self.log_rate + torch.log(self._bq(self.exposure).to(self.raw.dtype).clamp_min(1e-12))
        lp_count = D.neg_binomial_log_prob_from_log(y_count, log_mu, self.log_conc)
        y_bin = torch.where(kind == K_BINARY, yb, zeros).clamp(0.0, 1.0)
        lp_bin = D.bernoulli_log_prob(y_bin, self.logit)
        y_tte = torch.where(kind == K_TTE, yb, zeros).clamp_min(0.0)
        lp_tte = D.hazard_log_prob(y_tte.expand(self.logit.shape), ev.expand(self.logit.shape),
                                   self.hazard_logits, self._edges())
        y_cat = torch.where(kind == K_CAT, yb, zeros).long().expand(self.logit.shape)
        ncat = self._bq(self.n_categories).expand(self.logit.shape)
        lp_cat = D.categorical_log_prob(y_cat, self.cat_logits, ncat)
        return self._select(lp_real, lp_count, lp_bin, lp_tte, lp_cat)

    # ---- predictive moments -----------------------------------------------------------------------------
    def member_means(self) -> Tensor:
        """[M, B, N, Q] predictive mean of each member in the query's own units (real: metric units; count:
        expected count; binary: probability; time-to-event: restricted mean seconds; categorical: probability
        of the class that is most likely under the ensemble mixture)."""
        real = self.loc * self._bq(self.real_scale) + self._bq(self.real_loc)
        count = torch.exp(self.log_rate) * self._bq(self.exposure)
        binary = torch.sigmoid(self.logit)
        tte = D.hazard_expected_time(self.hazard_logits, self._edges())
        return self._select(real, count, binary, tte, self._cat_mode_prob())

    def _cat_probs(self) -> Tensor:
        ncat = self._bq(self.n_categories).expand(self.logit.shape)
        return torch.exp(D.masked_log_softmax(self.cat_logits, ncat))

    def _cat_mode_prob(self) -> Tensor:
        p = self._cat_probs()
        mode = p.mean(0, keepdim=True).argmax(-1, keepdim=True).expand(p.shape[:-1] + (1,))
        return p.gather(-1, mode).squeeze(-1)

    def member_variances(self) -> Tensor:
        """[M, B, N, Q] aleatoric variance of each member's predictive distribution."""
        s = self._bq(self.real_scale)
        if self.cfg.real_family == D.FAMILY_NORMAL:
            real = (self.scale * s) ** 2
        else:
            real = (self.scale * s) ** 2 * self.df / (self.df - 2.0)
        mu = torch.exp(self.log_rate) * self._bq(self.exposure)
        count = mu + mu * mu * torch.exp(-self.log_conc)
        p = torch.sigmoid(self.logit)
        binary = p * (1.0 - p)
        tte = D.DiscreteHazard(self.hazard_logits, self._edges()).variance
        pm = self._cat_mode_prob()
        return self._select(real, count, binary, tte, pm * (1.0 - pm))

    def predictive(self) -> Dict[str, Tensor]:
        """Ensemble summary, each [B, N, Q]: ``mean``, ``epistemic_std`` (between-member, reducible),
        ``aleatoric_std`` (within-member, irreducible under the model) and ``total_std``."""
        mm, mv = self.member_means(), self.member_variances()
        mean = mm.mean(0)
        epi = mm.var(0, unbiased=True) if mm.shape[0] > 1 else torch.zeros_like(mean)
        alea = mv.mean(0)
        return {"mean": mean, "epistemic_std": epi.sqrt(), "aleatoric_std": alea.clamp_min(0).sqrt(),
                "total_std": (epi + alea).clamp_min(0).sqrt()}

    def thompson_means(self, generator: Optional[torch.Generator] = None) -> Tensor:
        """[B, N, Q]: predictive means of one ensemble member drawn uniformly per batch row (a posterior sample
        of the model, shared by all candidates of a decision so their comparison is coherent)."""
        mm = self.member_means()
        idx = torch.randint(0, mm.shape[0], (mm.shape[1],), generator=generator).to(mm.device)
        return mm[idx, torch.arange(mm.shape[1], device=mm.device)]

    # ---- contracts.Distribution -------------------------------------------------------------------------
    def to_distribution(self, b: int, n: int, q: int) -> Distribution:
        """One ``contracts.Distribution`` for candidate ``n`` and query ``q`` of row ``b``. Bernoulli, hazard and
        categorical records are the exact equal-weight mixture of the members; real and count records are
        moment-matched (location / mean averaged, spread = aleatoric + epistemic)."""
        kind = int(self.kind_ids[b, q])
        pred = {k: v[b, n, q].detach().double() for k, v in self.predictive().items()}
        epi = float(pred["epistemic_std"])
        sel = (slice(None), b, n, q)
        if kind == K_REAL:
            s, l0 = float(self.real_scale[b, q]), float(self.real_loc[b, q])
            loc = self.loc[sel].detach().double()
            scale2 = (self.scale[sel].detach().double() ** 2).mean() + (loc.var(unbiased=True) if len(loc) > 1 else 0.0)
            if self.cfg.real_family == D.FAMILY_NORMAL:
                d: D.Dist = D.Normal(loc.mean() * s + l0, scale2.sqrt() * s)
            else:
                d = D.StudentT(loc.mean() * s + l0, scale2.sqrt() * s, self.df[sel].detach().double().mean())
        elif kind == K_COUNT:
            expo = float(self.exposure[b, q])
            mu, var = pred["mean"], pred["total_std"] ** 2
            conc = (mu * mu / (var - mu).clamp_min(1e-9)).clamp(1e-3, 1e6)
            d = D.NegBinomial(mu / expo, conc, expo)
        elif kind == K_BINARY:
            d = D.Bernoulli.from_prob(torch.sigmoid(self.logit[sel].detach().double()).mean())
        elif kind == K_TTE:
            pmf = D.hazard_pmf(self.hazard_logits[sel].detach().double()).mean(0)          # exact mixture
            surv_before = 1.0 - torch.cumsum(pmf[:-1], 0) + pmf[:-1]
            h = (pmf[:-1] / surv_before.clamp_min(1e-300)).clamp(1e-12, 1 - 1e-12)
            edges = D.uniform_edges(self.horizon_s[b, q].double().clamp_min(1e-6), self.cfg.n_hazard_bins)
            d = D.DiscreteHazard(torch.log(h) - torch.log1p(-h), edges)
        else:
            ncat = int(self.n_categories[b, q])
            p = self._cat_probs()[sel].detach().double().mean(0)[:ncat]
            d = D.Categorical(torch.log(p.clamp_min(1e-300)))
        return d.to_record(epistemic_std=epi)


class WorldHeads(nn.Module):
    """Bootstrapped-ensemble world-model heads over the shared decision latent."""

    def __init__(self, cfg: WorldConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        h, m = cfg.hidden, cfg.n_members
        self.n_out = 6 + cfg.n_hazard_bins + cfg.max_categories
        self.in_z = nn.Linear(cfg.d_model, h)
        self.in_q = nn.Linear(cfg.d_model, h)
        self.in_zg = nn.Linear(cfg.d_model, h, bias=False)
        self.in_qg = nn.Linear(cfg.d_model, h, bias=False)
        self.in_t = nn.Linear(cfg.d_time, h)
        self.kind_emb = nn.Embedding(N_KINDS, h)
        self.in_phi = nn.Linear(cfg.n_features, h) if cfg.n_features > 0 else None
        self.norm = nn.LayerNorm(h)
        # small random output layer (a zero-initialised output blocked learning in an earlier project)
        self.members = _EnsembleMLP(m, h, h, self.n_out, out_std=0.1 / math.sqrt(h), generator=g)
        d_prior_in = 2 * cfg.d_model + cfg.d_time + N_KINDS
        self.prior = _EnsembleMLP(m, d_prior_in, cfg.prior_hidden, self.n_out, out_std=1.0 / math.sqrt(cfg.prior_hidden),
                                  trainable=False, generator=g) if cfg.prior_scale > 0 else None

    def forward(self, z: Tensor, e_q: Tensor, kind_ids: Tensor, horizon_s: Tensor, phi: Optional[Tensor] = None, *,
                exposure: Optional[Tensor] = None, n_categories: Optional[Tensor] = None,
                real_loc: Optional[Tensor] = None, real_scale: Optional[Tensor] = None) -> WorldOutput:
        """z [B, N, D]; e_q [B, Q, D] or [B, N, Q, D]; kind_ids [B, Q] long; horizon_s [B, Q] seconds;
        phi [B, N, Q, F] optional. Returns a :class:`WorldOutput` holding every member's parameters."""
        cfg = self.cfg
        b, n, _ = z.shape
        eq = e_q if e_q.dim() == 4 else e_q[:, None].expand(b, n, e_q.shape[1], e_q.shape[2])
        q = eq.shape[2]
        tf = time_features(horizon_s, cfg.d_time).to(z.dtype)                       # [B, Q, d_time]
        zz = z[:, :, None, :]
        x = (self.in_z(zz) + self.in_q(eq) + self.in_zg(zz) * self.in_qg(eq) / math.sqrt(cfg.hidden)
             + (self.in_t(tf) + self.kind_emb(kind_ids))[:, None])
        if self.in_phi is not None:
            if phi is None:
                raise ValueError("this WorldHeads was built with n_features > 0 and needs phi")
            x = x + self.in_phi(phi)
        x = F.silu(self.norm(x))
        raw = self.members(x.unsqueeze(0).expand((cfg.n_members,) + x.shape))       # [M, B, N, Q, P]
        if self.prior is not None:
            kind_1h = F.one_hot(kind_ids, N_KINDS).to(z.dtype)
            u = torch.cat([zz.expand(b, n, q, z.shape[-1]), eq,
                           torch.cat([tf, kind_1h], -1)[:, None].expand(b, n, q, cfg.d_time + N_KINDS)], -1).detach()
            raw = raw + cfg.prior_scale * self.prior(u.unsqueeze(0).expand((cfg.n_members,) + u.shape))
        dev = z.device
        expo = torch.ones(b, q, device=dev) if exposure is None else exposure.to(dev, z.dtype)
        ncat = torch.full((b, q), cfg.max_categories, device=dev, dtype=torch.long) if n_categories is None \
            else n_categories.to(dev).clamp(1, cfg.max_categories)
        rl = torch.zeros(b, q, device=dev) if real_loc is None else real_loc.to(dev, z.dtype)
        rs = torch.ones(b, q, device=dev) if real_scale is None else real_scale.to(dev, z.dtype).clamp_min(1e-12)
        return WorldOutput(raw, cfg, kind_ids, horizon_s, expo, ncat, rl, rs)


class KindNormaliser(nn.Module):
    """Running scale of each kind's mean NLL. Dividing a kind's loss by ``max(1, EMA |NLL|)`` keeps a kind whose
    likelihood is numerically large (big counts, many hazard bins) from owning the shared trunk's gradient."""

    def __init__(self, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("scale", torch.zeros(N_KINDS))
        self.register_buffer("seen", torch.zeros(N_KINDS, dtype=torch.bool))

    @torch.no_grad()
    def update(self, kind: int, value: float) -> None:
        v = abs(float(value))
        if not bool(self.seen[kind]):
            self.scale[kind], self.seen[kind] = v, True
        else:
            self.scale[kind] = self.momentum * self.scale[kind] + (1 - self.momentum) * v

    def denominator(self, kind: int) -> float:
        return max(1.0, float(self.scale[kind]))


def world_nll(out: WorldOutput, targets: WorldTargets, *, chosen: Optional[Tensor] = None,
              q_mask: Optional[Tensor] = None, member_weights: Optional[Tensor] = None,
              kind_weights: Optional[Dict[str, float]] = None, normaliser: Optional[KindNormaliser] = None
              ) -> Tuple[Tensor, Dict[str, float]]:
    """Masked negative log-likelihood of matured outcomes.

    ``chosen`` [B] selects the executed candidate (outcomes exist only for the action that was actually taken);
    omit it when ``out`` was computed for N = 1. ``member_weights`` [M, B] are the bootstrap multiplicities.
    Returns ``(loss, metrics)``; the loss is 0 with a zero gradient when nothing has matured.
    """
    kind_ids = out.kind_ids
    mask = targets.mask(kind_ids)
    if q_mask is not None:
        mask = mask & q_mask
    y = D.safe_targets(targets.y.to(out.raw.dtype), mask)
    lp = out.log_prob(y, event=targets.observed)                                   # [M, B, N, Q]
    m, b, n, q = lp.shape
    if chosen is not None:
        idx = chosen.view(1, b, 1, 1).expand(m, b, 1, q)
        lp = lp.gather(2, idx).squeeze(2)
    elif n == 1:
        lp = lp.squeeze(2)
    else:
        raise ValueError("world_nll needs `chosen` when the output covers more than one candidate")
    nll = -lp                                                                      # [M, B, Q]
    w = torch.ones(m, b, device=nll.device, dtype=nll.dtype) if member_weights is None \
        else member_weights.to(nll.device, nll.dtype)
    wm = w[:, :, None] * mask[None].to(nll.dtype)                                  # [M, B, Q]
    nll = torch.where(mask[None].expand_as(nll), nll, torch.zeros_like(nll))
    total = nll.sum() * 0.0                              # attached to the graph: an all-pending batch backpropagates zeros
    metrics: Dict[str, float] = {"n_targets": float(mask.sum()), "n_masked": float((~mask).sum())}
    for name, kid in KIND_INDEX.items():
        km = (kind_ids == kid) & mask
        cnt = int(km.sum())
        if cnt == 0:
            continue
        kw = wm * km[None].to(nll.dtype)
        per_member = (nll * kw).sum((1, 2)) / kw.sum((1, 2)).clamp_min(1e-12)       # [M]
        has = (kw.sum((1, 2)) > 0).to(nll.dtype)
        l_k = (per_member * has).sum() / has.sum().clamp_min(1.0)
        plain = float((nll.detach() * km[None]).sum() / (m * cnt))
        metrics[f"nll_{name}"] = plain
        metrics[f"n_{name}"] = float(cnt)
        denom = 1.0
        if normaliser is not None:
            if out.raw.requires_grad:
                normaliser.update(kid, plain)
            denom = normaliser.denominator(kid)
        total = total + (kind_weights or {}).get(name, 1.0) * l_k / denom
    metrics["loss"] = float(total.detach())
    return total, metrics


# ----------------------------------------------------------------------------------------------------------------
# latent transition (one-step model for imagination)
# ----------------------------------------------------------------------------------------------------------------
class TransitionOutput:
    """Predicted next belief. ``imagined`` is always True: this is a model output, never an observation."""
    imagined = True

    def __init__(self, mean: Tensor, log_var: Tensor):
        self.member_mean, self.member_log_var = mean, log_var                      # [M, ..., D]

    @property
    def mean(self) -> Tensor:
        return self.member_mean.mean(0)

    @property
    def epistemic_var(self) -> Tensor:
        mm = self.member_mean
        return mm.var(0, unbiased=True) if mm.shape[0] > 1 else torch.zeros_like(mm[0])

    @property
    def aleatoric_var(self) -> Tensor:
        return torch.exp(self.member_log_var).mean(0)

    @property
    def total_std(self) -> Tensor:
        return (self.epistemic_var + self.aleatoric_var).sqrt()

    def sample(self, generator: Optional[torch.Generator] = None, member: Optional[Tensor] = None) -> Tensor:
        """One draw of the next belief per leading row: a random member plus its aleatoric noise."""
        mm, lv = self.member_mean, self.member_log_var
        rows = mm.shape[1]
        if member is None:
            member = torch.randint(0, mm.shape[0], (rows,), generator=generator)
        member = member.to(mm.device)
        ar = torch.arange(rows, device=mm.device)
        mu, var = mm[member, ar], torch.exp(lv[member, ar])
        eps = torch.randn(mu.shape, generator=generator).to(mu.device, mu.dtype)
        return mu + var.sqrt() * eps


class LatentTransition(nn.Module):
    """(z, dt) -> distribution over the next belief b' (ensemble mean + diagonal variance). With ``b`` given the
    members predict the change ``b' - b`` (residual), which is the easier target for short steps."""

    def __init__(self, d_model: int, hidden: int = 256, n_members: int = 3, d_time: int = 16, seed: int = 0,
                 min_log_var: float = -10.0, max_log_var: float = 4.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_model, self.d_time, self.n_members = d_model, d_time, n_members
        self.min_log_var, self.max_log_var = min_log_var, max_log_var
        self.in_z = nn.Linear(d_model, hidden)
        self.in_b = nn.Linear(d_model, hidden)
        self.in_t = nn.Linear(d_time, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.members = _EnsembleMLP(n_members, hidden, hidden, 2 * d_model, out_std=0.1 / math.sqrt(hidden),
                                    generator=g)

    def forward(self, z: Tensor, dt_s: Tensor, b: Optional[Tensor] = None) -> TransitionOutput:
        """z [..., D]; dt_s [...] seconds; b [..., D] optional current belief (residual prediction)."""
        x = self.in_z(z) + self.in_t(time_features(dt_s, self.d_time).to(z.dtype))
        if b is not None:
            x = x + self.in_b(b)
        x = F.silu(self.norm(x))
        o = self.members(x.unsqueeze(0).expand((self.n_members,) + x.shape))
        mean, log_var = o[..., :self.d_model], o[..., self.d_model:] - 2.0
        if b is not None:
            mean = mean + b.unsqueeze(0)
        return TransitionOutput(mean, log_var.clamp(self.min_log_var, self.max_log_var))


def transition_nll(out: TransitionOutput, b_next: Tensor, mask: Optional[Tensor] = None,
                   member_weights: Optional[Tensor] = None, detach_target: bool = True
                   ) -> Tuple[Tensor, Dict[str, float]]:
    """Latent-consistency loss: Gaussian NLL of the *real* next belief under each member, averaged over
    dimensions. ``mask`` [...] is False where no next belief exists (those rows contribute exactly zero)."""
    tgt = b_next.detach() if detach_target else b_next
    mm, lv = out.member_mean, out.member_log_var
    if mask is None:
        mask = torch.ones(mm.shape[1:-1], dtype=torch.bool, device=mm.device)
    tgt = torch.where(mask.unsqueeze(-1), tgt, torch.zeros_like(tgt))             # masked rows: harmless target
    nll = 0.5 * ((tgt.unsqueeze(0) - mm) ** 2 * torch.exp(-lv) + lv + D.LOG_2PI).mean(-1)      # [M, ...]
    w = mask.unsqueeze(0).to(nll.dtype).expand_as(nll)
    if member_weights is not None:
        w = w * member_weights.to(nll.device, nll.dtype).reshape(member_weights.shape + (1,) * (nll.dim() - 2))
    nll = torch.where(mask.unsqueeze(0).expand_as(nll), nll, torch.zeros_like(nll))
    loss = (nll * w).sum() / w.sum().clamp_min(1e-12) if bool(mask.any()) else nll.sum() * 0.0
    with torch.no_grad():
        err = ((out.mean - tgt) ** 2).mean(-1)
        scale = tgt.reshape(-1, tgt.shape[-1]).var(0).mean().clamp_min(1e-12) if tgt.numel() > tgt.shape[-1] \
            else tgt.new_ones(())
        mse = float(D.masked_mean(err, mask))
    return loss, {"transition_nll": float(loss.detach()), "transition_mse": mse,
                  "transition_nmse": mse / float(scale), "n": float(mask.sum())}
