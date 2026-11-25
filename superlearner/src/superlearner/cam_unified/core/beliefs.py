"""Multiple mechanism hypotheses: a neural particle belief (workstream B; implements `BeliefModuleP`).

B_t = {(m^k, w^k, Sigma^k)}_{k=1..K}: particles m^k in R^{Dp} (`cfg.d_particle`), normalised log-weights, diagonal
scales. Nothing assigns particles to named concepts; what a particle means is whatever the outcome head conditioned on
it predicts.

Prior        m^k ~ N(mu_0(Z), diag sigma_0(Z)^2) (learned mean/scale, optionally conditioned on pooled Z).
Weighting    log w^k += log p_theta(o_t | Z_{t-1}, a_{t-1}, m^k), using any OutcomeHeadP:
             feat = head.features(Z_a, outcomes) with Z_a = Z_{t-1} conditioned on the action tokens by a learned
             cross-attention block (identity at initialisation), then head.dist(feat, outcomes, cond=particles)
             .log_prob(y) [B,K,Q], masked by y_mask and summed over queries; weights renormalised.
Proposal     m^k <- m^k + g_theta(m^k, event summary, relative fit of k) + jitter * Sigma^k eps, with a learned per-
             dimension contraction of Sigma^k per informative step (so uncertainty shrinks as evidence accrues).
Resampling   ESS = 1 / sum_k w_k^2 < cfg.ess_frac * K -> systematic resampling + rejuvenation noise rejuv * Sigma^k eps,
             weights reset to uniform.
Misspecification  per informative step the observed mixture predictive log-likelihood ll_obs is compared with its
             own predictive distribution, estimated by Monte Carlo from the mixture itself (y ~ sum_k w_k p(y | m^k),
             `n_mc` draws of ll(y); censored SURVIVAL queries are simulated through the same observation process:
             y_sim = min(T_sim, c), censored iff T_sim > c, with c the observed censoring time); the mean of those
             draws is the predicted entropy -H_pred. Per-step z-scores: "rank" (default): z = Phi^-1(1 - u), u the
             rank of ll_obs among the draws with random tie-breaking (probability-integral transform; discrete-
             uniform under a correctly specified belief for any outcome type, ties included, so z ~ N(0,1) truncated
             at Phi^-1(1 - 0.5/(n_mc+1))); "moment": z = (E[ll] - ll_obs) / sd[ll] (heavy upper tail for Gaussian
             outcomes: the log-likelihood is a shifted chi-square — no false-alarm guarantee).
             Sequential test: one-sided CUSUM over the surprise window, S = max(0, max over suffixes of the window of
             sum_j (z_j - k)), alarm when S > h (with >= `min_steps` entries). k defaults to min(misspec_z, z_max)/2
             (half the sustained shift to detect); h is set from a target in-control average run length `misspec_arl`
             (informative steps per row between false alarms; Siegmund's approximation for N(0,1) increments,
             conservative for the truncated rank z). Under sustained surprise near z_max the alarm fires after about
             h / (z_max - k) steps (5 with the defaults).
             Response: inject fresh particles from the proposal conditioned on the latest event (replacing the
             lowest-weight `inject_frac` of particles; prior scale inflated by `inflate`), inflate every Sigma^k by
             `inflate`, reset weights to uniform and the surprise window (cooldown: the CUSUM must re-accumulate).
             Sigma^k is capped at `max_scale_mult` x the prior scale. Persistent misspecification: an alarm while
             the row's uncertainty is already saturated (median Sigma^k >= cap / inflate, i.e. recent expansions did
             not help) does not inject or reset weights again — Sigma^k is pinned at the cap and the row is flagged
             (`info["persistent"]`, counters, log). Every expansion / persistent episode is counted and logged
             (`counters`, `misspec_log`, python logging).

All operations are batched over environments; the FastState tensors stay in the autograd graph (the caller detaches
between truncation windows), so the prior, proposal and token encoders train through `predictive_log_prob`
support/query objectives. Resampling indices and the misspecification decision are discrete (no gradient).
"""
from __future__ import annotations

import collections
import contextlib
import inspect
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import CamConfig
from ..interfaces import FastState, Likelihood, OutcomeBatch, TokenKind, TokenSet
from .blocks import CrossBlock, RMSNorm, SwiGLU, count_parameters, init_transformer_weights

log = logging.getLogger(__name__)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def dist_log_prob(dist, y: Tensor, censored: Optional[Tensor] = None) -> Tensor:
    """OutcomeDist.log_prob(y, censored=...) when the distribution accepts censoring (CAM outcome heads), plain
    log_prob(y) otherwise (torch.distributions)."""
    fn = dist.log_prob
    try:
        accepts = "censored" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return fn(y, censored=censored)
    return fn(y)


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def _rand(fn, shape, generator: Optional[torch.Generator], device) -> Tensor:
    """torch.rand / torch.randn on `device`, drawing from `generator` (which may live on another device)."""
    if generator is None:
        return fn(*shape, device=device)
    return fn(*shape, generator=generator, device=generator.device).to(device)


def randn(shape, generator: Optional[torch.Generator], device) -> Tensor:
    return _rand(torch.randn, shape, generator, device)


def systematic_resample(w: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Systematic resampling indices. w [B,K] normalised weights -> idx [B,K] long (ancestor of each new particle)."""
    B, K = w.shape
    u0 = _rand(torch.rand, (B, 1), generator, w.device)
    u = (u0 + torch.arange(K, device=w.device, dtype=w.dtype)) / K
    cdf = torch.cumsum(w, -1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-30)
    return torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(max=K - 1)


class BeliefModule(nn.Module):
    """Particle belief over environment mechanisms.

    Args:
        cfg: CamConfig (n_particles, d_particle, d_model, n_heads, d_ff, ess_frac, misspec_z, misspec_window,
             bayes_feat, bayes_prior_var_init, bayes_obs_var_init).
        n_channels: Bayesian last-layer channel slots C allocated in FastState.
        n_mc: Monte Carlo draws for the predictive log-likelihood distribution (misspecification z-score).
        surprise_stat: "rank" (calibrated probability-integral-transform z) or "moment" ((E[ll] - ll_obs) / sd).
        min_steps: minimum informative steps in the window before expansion can trigger.
        z_clip: per-step |z| clip (one outlier cannot trigger an expansion on its own).
        inject_frac, inflate: fraction of particles replaced and scale inflation on expansion.
        jitter, rejuv: proposal noise and post-resampling rejuvenation noise (multiples of Sigma^k).
        min_scale: lower bound of Sigma^k.
        misspec_enabled, resample_enabled: ablation switches.
    """

    def __init__(self, cfg: CamConfig, n_channels: int = 16, n_mc: int = 32, surprise_stat: str = "rank",
                 min_steps: int = 4, z_clip: float = 8.0, misspec_arl: float = 1e5, cusum_k: Optional[float] = None,
                 cusum_h: Optional[float] = None, max_scale_mult: float = 10.0,
                 inject_frac: float = 0.5, inflate: float = 3.0, jitter: float = 0.1, rejuv: float = 0.5,
                 min_scale: float = 1e-3, prior_scale: float = 1.0, contraction: float = 0.95,
                 misspec_enabled: bool = True, resample_enabled: bool = True, autocast: Optional[bool] = None,
                 log_capacity: int = 1000):
        super().__init__()
        self.cfg = cfg
        K, Dp, D = cfg.n_particles, cfg.d_particle, cfg.d_model
        self.K, self.Dp, self.C = K, Dp, n_channels
        assert surprise_stat in ("rank", "moment"), surprise_stat
        assert inflate > 1.0 and max_scale_mult > inflate, (inflate, max_scale_mult)
        self.n_mc, self.surprise_stat, self.min_steps, self.z_clip = n_mc, surprise_stat, min_steps, z_clip
        self.inject_frac, self.inflate, self.jitter, self.rejuv = inject_frac, inflate, jitter, rejuv
        self.min_scale, self.max_scale_mult = min_scale, max_scale_mult
        # ---- sequential misspecification test (one-sided windowed CUSUM)
        self.z_max = self.rank_z_max(n_mc) if surprise_stat == "rank" else z_clip
        self.cusum_k = cusum_k if cusum_k is not None else 0.5 * min(cfg.misspec_z, self.z_max)
        self.misspec_arl = misspec_arl
        self.cusum_h = cusum_h if cusum_h is not None else self.cusum_threshold(self.cusum_k, misspec_arl)
        self.misspec_enabled, self.resample_enabled = misspec_enabled, resample_enabled
        self.autocast = autocast
        # ---- prior
        self.prior_mu = nn.Parameter(torch.zeros(Dp))
        self.prior_raw_scale = nn.Parameter(torch.full((Dp,), _inv_softplus(prior_scale)))
        self.prior_cond = nn.Sequential(RMSNorm(D), nn.Linear(D, 2 * Dp))
        # ---- action conditioning of Z_{t-1} (identity at init)
        self.act_cond = CrossBlock(D, cfg.n_heads, cfg.d_ff)
        # ---- event summary
        De = Dp
        self.ev_feat = nn.Linear(D, De)
        self.ev_y = nn.Linear(2 + int(Likelihood.SURVIVAL) + 1, De)
        self.ev_mlp = nn.Sequential(nn.SiLU(), nn.Linear(De, De))
        # ---- proposal g_theta(m, s, fit)
        self.prop = nn.Sequential(nn.Linear(Dp + De + 2, 2 * Dp), nn.SiLU(), nn.Linear(2 * Dp, Dp))
        self.contract_raw = nn.Parameter(torch.full((Dp,), math.log(contraction / (1 - contraction))))
        # ---- belief tokens
        self.tok_m = nn.Linear(Dp, D)
        self.tok_s = nn.Linear(Dp, D)
        self.tok_w = nn.Linear(4, D)
        self.tok_norm = RMSNorm(D)
        self.tok_ffn = SwiGLU(D, cfg.d_ff)
        self.sum_feat = nn.Linear(7, D)
        self.sum_emb = nn.Parameter(torch.randn(D) * 0.02)
        self.out_norm = RMSNorm(D)
        init_transformer_weights(self, 2)
        for lin in (self.prior_cond[1], self.prop[2], self.act_cond.attn.o, self.act_cond.ffn.w2):
            nn.init.zeros_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        self.counters = {"updates": 0, "resampled_rows": 0, "expanded_rows": 0, "nonfinite_ll": 0,
                         "alarm_rows": 0, "persistent_row_steps": 0, "persistent_episodes": 0}
        self.misspec_log: collections.deque = collections.deque(maxlen=log_capacity)

    def _autocast_ctx(self, device):
        use = self.autocast if self.autocast is not None else device.type == "cuda"
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use else contextlib.nullcontext()

    # ------------------------------------------------------------------ prior / init
    def prior(self, B: int, Z: Optional[Tensor] = None, device=None) -> tuple:
        """(mu_0 [B,Dp], sigma_0 [B,Dp]) — learned, optionally conditioned on pooled Z [B,S,D]."""
        dev = device or self.prior_mu.device
        mu = self.prior_mu.unsqueeze(0).expand(B, -1).to(dev)
        raw = self.prior_raw_scale.unsqueeze(0).expand(B, -1).to(dev)
        if Z is not None:
            dmu, dls = self.prior_cond(Z.float().mean(1)).chunk(2, -1)
            mu, raw = mu + dmu, raw + dls
        return mu, F.softplus(raw) + self.min_scale

    def init_fast(self, B: int, device=None, Z: Optional[Tensor] = None,
                  generator: Optional[torch.Generator] = None) -> FastState:
        """Fresh FastState: K particles from the (optionally Z-conditioned) prior, uniform weights, Bayesian channels at
        the config prior, empty surprise window (NaN = no entry)."""
        dev = device if device is not None else (Z.device if Z is not None else self.prior_mu.device)
        cfg, K, Dp, C, Fb = self.cfg, self.K, self.Dp, self.C, self.cfg.bayes_feat
        mu, sc = self.prior(B, Z, dev)
        eps = randn((B, K, Dp), generator, dev)
        return FastState(
            particles=mu.unsqueeze(1) + sc.unsqueeze(1) * eps,
            logw=torch.full((B, K), -math.log(K), device=dev),
            pscale=sc.unsqueeze(1).expand(B, K, Dp).clone(),
            bayes_mu=torch.zeros(B, C, Fb, device=dev, dtype=torch.float64),
            bayes_P=torch.eye(Fb, device=dev, dtype=torch.float64).expand(B, C, Fb, Fb).clone()
            * cfg.bayes_prior_var_init,
            bayes_noise=torch.full((B, C), cfg.bayes_obs_var_init, device=dev, dtype=torch.float64),
            bayes_n=torch.zeros(B, C, device=dev),
            surprise=torch.full((B, cfg.misspec_window), float("nan"), device=dev),
            steps=torch.zeros(B, dtype=torch.long, device=dev),
            channels=[{} for _ in range(B)])

    # ------------------------------------------------------------------ summaries
    @staticmethod
    def weight_stats(logw: Tensor) -> tuple:
        """(w [B,K], ESS [B], entropy [B]) from normalised log-weights."""
        w = logw.exp()
        ess = 1.0 / (w * w).sum(-1).clamp_min(1e-30)
        ent = -(w * logw.clamp_min(-1e4)).sum(-1)
        return w, ess, ent

    @staticmethod
    def rank_z_max(n_mc: int) -> float:
        """Largest per-step rank z: Phi^-1(1 - 0.5 / (n_mc + 1))."""
        return -math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2 * (0.5 / (n_mc + 1)) - 1, dtype=torch.float64)))

    @staticmethod
    def cusum_arl0(k: float, h: float) -> float:
        """Siegmund's approximation of the in-control average run length of a one-sided CUSUM with reference k and
        threshold h for N(0,1) increments: (exp(2kb) - 2kb - 1) / (2k^2), b = h + 1.166."""
        a = 2.0 * k * (h + 1.166)
        return (math.exp(min(a, 700.0)) - a - 1.0) / (2.0 * k * k)

    @classmethod
    def cusum_threshold(cls, k: float, arl0: float) -> float:
        """Threshold h with cusum_arl0(k, h) = arl0 (bisection)."""
        lo, hi = 0.0, 200.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cls.cusum_arl0(k, mid) < arl0:
                lo = mid
            else:
                hi = mid
        return hi

    def cusum_stat(self, surprise: Tensor) -> tuple:
        """Windowed one-sided CUSUM of the surprise z-scores (oldest first, NaN = no entry): S [B] = max(0, max over
        suffixes of sum_j (z_j - k)), and the number of valid entries n [B]."""
        valid = ~torch.isnan(surprise)
        inc = torch.where(valid, surprise.clamp(-self.z_clip, self.z_clip) - self.cusum_k, torch.zeros_like(surprise))
        suffix = inc.flip(-1).cumsum(-1)                                     # sums of the most recent 1, 2, ... entries
        return suffix.max(-1).values.clamp_min(0.0), valid.sum(-1)

    def window_stat(self, surprise: Tensor) -> tuple:
        """(sum(z) / sqrt(n) [B], n [B]) over the valid (non-NaN) entries of the surprise window."""
        valid = ~torch.isnan(surprise)
        z = torch.where(valid, surprise, torch.zeros_like(surprise)).clamp(-self.z_clip, self.z_clip)
        n = valid.sum(-1)
        return z.sum(-1) / n.clamp_min(1).float().sqrt(), n

    def tokens(self, fast: FastState) -> TokenSet:
        """Belief tokens [B,K+1,D]: one token per particle (mean, log scale, weight, ESS, weight entropy) and a summary
        token (weighted particle tokens + ESS / entropy / surprise statistics). Particles carry no index (set)."""
        B, K, _ = fast.particles.shape
        dev = fast.particles.device
        w, ess, ent = self.weight_stats(fast.logw)
        logK = math.log(K)
        pf = torch.stack([w * K, (fast.logw + logK).clamp(-20, 5) / 5, (ess / K).unsqueeze(-1).expand(B, K),
                          (ent / logK).unsqueeze(-1).expand(B, K)], -1)
        with self._autocast_ctx(dev):
            x = self.tok_m(fast.particles.float()) + self.tok_s(torch.log(fast.pscale.float().clamp_min(1e-8))) \
                + self.tok_w(pf)
            x = x.float()
            x = x + self.tok_ffn(self.tok_norm(x)).float()
            wz, n = self.window_stat(fast.surprise)
            valid = ~torch.isnan(fast.surprise)
            last_mean = torch.where(valid, fast.surprise, torch.zeros_like(fast.surprise)).sum(-1) / n.clamp_min(1)
            cs, _ = self.cusum_stat(fast.surprise)
            sf = torch.stack([ess / K, ent / logK, last_mean.clamp(-self.z_clip, self.z_clip) / self.z_clip,
                              wz.clamp(-self.z_clip, self.z_clip) / self.z_clip, torch.log1p(fast.steps.float()) / 5,
                              n.float() / max(1, fast.surprise.shape[1]), (cs / self.cusum_h).clamp(max=4.0)], -1)
            summ = (w.unsqueeze(-1) * x).sum(1) + self.sum_feat(sf).float() + self.sum_emb
            h = self.out_norm(torch.cat([x, summ.unsqueeze(1)], 1)).float()
        kind = torch.full((B, K + 1), int(TokenKind.BELIEF), dtype=torch.long, device=dev)
        refs = [[f"particle:{k}" for k in range(K)] + ["belief:summary"] for _ in range(B)]
        return TokenSet(h, torch.ones(B, K + 1, dtype=torch.bool, device=dev), kind, torch.zeros(B, K + 1, device=dev),
                        torch.full((B, K + 1), -1, dtype=torch.long, device=dev), refs)

    # ------------------------------------------------------------------ likelihood pieces
    def condition_on_action(self, Z_prev: Tensor, act: Optional[TokenSet]) -> Tensor:
        """Z_a [B,S,D]: Z_{t-1} conditioned on the executed action's tokens (cross-attention, null key for empty)."""
        if act is None or act.N == 0:
            return Z_prev
        with self._autocast_ctx(Z_prev.device):
            return self.act_cond(Z_prev.float(), act.x.float(), key_mask=act.mask).float()

    def particle_loglik(self, fast: FastState, feat: Tensor, outcomes: OutcomeBatch, outcome_head) -> tuple:
        """Per-particle log-likelihood of the observed outcomes. Returns (loglik [B,K], dist) with dist the
        particle-conditioned OutcomeDist (batch [B,K,Q])."""
        B, K, _ = fast.particles.shape
        Q = outcomes.y.shape[1]
        dist = outcome_head.dist(feat, outcomes, cond=fast.particles)
        m = outcomes.y_mask
        y = torch.where(m, outcomes.y, torch.zeros_like(outcomes.y)).unsqueeze(1).expand(B, K, Q)
        cen = None if outcomes.censored is None else outcomes.censored.unsqueeze(1).expand(B, K, Q)
        ll = dist_log_prob(dist, y, cen)
        bad = m.unsqueeze(1) & ~torch.isfinite(ll)
        if bool(bad.any()):
            self.counters["nonfinite_ll"] += int(bad.sum())
            ll = torch.nan_to_num(ll, nan=-1e4, neginf=-1e4, posinf=1e4)
        ll = torch.where(m.unsqueeze(1), ll, torch.zeros_like(ll))
        return ll.sum(-1), dist

    def predictive_log_prob(self, fast: FastState, feat: Tensor, outcomes: OutcomeBatch, outcome_head) -> Tensor:
        """log sum_k w_k prod_q p(y_q | m^k) for the observed (y_mask) queries -> [B] (0 for rows with none).
        Differentiable in the particles, weights, features and head — the support/query adaptation objective."""
        loglik, _ = self.particle_loglik(fast, feat, outcomes, outcome_head)
        lp = torch.logsumexp(fast.logw + loglik, -1)
        return torch.where(outcomes.y_mask.any(1), lp, torch.zeros_like(lp))

    def mixture_moments(self, fast: FastState, feat: Tensor, outcomes: OutcomeBatch, outcome_head) -> dict:
        """Mixture predictive moments per query: {"mean", "var", "epistemic"} each [B,Q]; epistemic = weighted
        variance of the particle means (particle disagreement), var = E_k[var_k] + epistemic."""
        dist = outcome_head.dist(feat, outcomes, cond=fast.particles)
        mu, var = dist.mean, dist.variance
        mu = mu() if callable(mu) else mu
        var = var() if callable(var) else var
        w = fast.logw.exp().unsqueeze(-1)
        mean = (w * mu).sum(1)
        epi = (w * (mu - mean.unsqueeze(1)) ** 2).sum(1)
        return {"mean": mean, "var": (w * var).sum(1) + epi, "epistemic": epi}

    @torch.no_grad()
    def surprise_z(self, dist, logw: Tensor, y_mask: Tensor, pred_ll: Tensor,
                   generator: Optional[torch.Generator] = None, y_obs: Optional[Tensor] = None,
                   censored: Optional[Tensor] = None, lik: Optional[Tensor] = None) -> tuple:
        """Per-step surprise of the observation under the mixture predictive. Returns (z [B], predicted entropy [B]):
        "rank": z = Phi^-1(1 - u), u = (r - 1/2) / (n_mc + 1) with r the rank of ll_obs among the n_mc draws
        ll(y), y ~ mixture, ties broken uniformly at random (exactly discrete-uniform under the model);
        "moment": z = (E[ll] - ll_obs) / sd[ll]. Positive z = more surprising than the belief expects.
        With `censored` [B,Q] (and the observed y_obs [B,Q]) censored queries are simulated through the observation
        process: y_sim = min(T_sim, c), censored_sim = T_sim > c, c = y_obs (restricted to SURVIVAL when `lik`)."""
        B, K = logw.shape
        w = logw.exp()
        gen = generator if generator is not None and generator.device == w.device else None
        cen_obs = None
        if censored is not None and y_obs is not None:
            cen_obs = censored.bool() & y_mask
            if lik is not None:
                cen_obs = cen_obs & (lik == int(Likelihood.SURVIVAL))
            if not bool(cen_obs.any()):
                cen_obs = None
        lls = []
        for _ in range(self.n_mc):
            ys = dist.sample().detach()                                       # [B,K,Q]
            k = torch.multinomial(w.float(), 1, generator=gen) if gen is not None else torch.multinomial(w.float(), 1)
            Q = ys.shape[-1]
            y = torch.gather(ys, 1, k.unsqueeze(-1).expand(B, 1, Q)).squeeze(1)
            cen_sim = None
            if cen_obs is not None:
                c = y_obs.to(y.dtype)
                cen_sim = cen_obs & (y > c)
                y = torch.where(cen_obs, torch.minimum(y, c), y)
                cen_sim = cen_sim.unsqueeze(1).expand(B, K, Q)
            ll = dist_log_prob(dist, y.unsqueeze(1).expand(B, K, Q), cen_sim)
            ll = torch.nan_to_num(ll, nan=-1e4, neginf=-1e4, posinf=1e4)
            ll = torch.where(y_mask.unsqueeze(1), ll, torch.zeros_like(ll)).sum(-1)
            lls.append(torch.logsumexp(logw + ll, -1))
        L = torch.stack(lls, 0)                                                # [n_mc,B]
        if self.surprise_stat == "moment":
            z = (L.mean(0) - pred_ll) / L.std(0).clamp_min(1e-3)
        else:
            below = (L < pred_ll.unsqueeze(0)).float().sum(0)
            ties = (L == pred_ll.unsqueeze(0)).float().sum(0)
            v = torch.rand(B, generator=gen, device=w.device) if gen is not None else torch.rand(B, device=w.device)
            r = below + 1 + torch.floor(v * (ties + 1)).clamp(max=ties)       # rank among n_mc + 1, random ties
            u = (r - 0.5) / (self.n_mc + 1)
            z = -math.sqrt(2.0) * torch.erfinv(2 * u - 1)
        return z, -L.mean(0)

    def event_summary(self, feat: Tensor, outcomes: OutcomeBatch) -> Tensor:
        """s [B,De]: masked mean over observed queries of (feature, symlog value, likelihood type, censoring)."""
        m = outcomes.y_mask
        y = torch.where(m, outcomes.y, torch.zeros_like(outcomes.y))
        onehot = F.one_hot(outcomes.lik.clamp(0, int(Likelihood.SURVIVAL)), int(Likelihood.SURVIVAL) + 1).float()
        cen = outcomes.censored.float() if outcomes.censored is not None else torch.zeros_like(y)
        yf = torch.cat([symlog(y).unsqueeze(-1), cen.unsqueeze(-1), onehot], -1)
        e = self.ev_feat(feat.float()) + self.ev_y(yf)
        mf = m.float().unsqueeze(-1)
        s = (e * mf).sum(1) / mf.sum(1).clamp_min(1)
        return self.ev_mlp(s)

    def propose(self, m: Tensor, s: Tensor, fit: Tensor, logw: Tensor) -> Tensor:
        """Learned proposal displacement g_theta(m^k, s, fit_k, w_k) [B,K,Dp] (zero at initialisation)."""
        B, K, _ = m.shape
        inp = torch.cat([m, s.unsqueeze(1).expand(B, K, -1), fit.unsqueeze(-1),
                         (logw + math.log(K)).clamp(-20, 5).unsqueeze(-1) / 5], -1)
        return self.prop(inp)

    # ------------------------------------------------------------------ update
    def update(self, fast: FastState, Z_prev: Tensor, act: Optional[TokenSet], outcomes: OutcomeBatch,
               outcome_head, feat: Optional[Tensor] = None, generator: Optional[torch.Generator] = None) -> tuple:
        """One Bayesian-style belief update with the outcomes of the executed action.

        Returns (fast', info) with info {"ess": [B] (after reweighting, before resampling), "loglik": [B,K],
        "surprise_z": [B] (0 where nothing was observed), "resampled": [B] bool, "expanded": [B] bool (particles
        injected, scales inflated, weights reset), "persistent": [B] bool (alarm while uncertainty is saturated: no
        reset, scale pinned at the cap), "alarm": [B] bool (CUSUM test fired), "cusum": [B] CUSUM statistic
        (threshold `cusum_h`), "pred_ll": [B] mixture predictive log-lik of the observation, "pred_entropy": [B] its
        predicted entropy (MC), "window_z": [B] sum(z)/sqrt(n) over the window (diagnostic), "feat": [B,Q,D]}.
        `feat` overrides the outcome features (e.g. computed by the world model from its prior state)."""
        B, K, Dp = fast.particles.shape
        dev = fast.particles.device
        self.counters["updates"] += 1
        has_obs = outcomes.y_mask.any(1)                                       # [B]
        if feat is None:
            feat = outcome_head.features(self.condition_on_action(Z_prev, act), outcomes)
        loglik, dist = self.particle_loglik(fast, feat, outcomes, outcome_head)
        joint = fast.logw + loglik
        pred_ll = torch.logsumexp(joint, -1)
        logw = joint - pred_ll.unsqueeze(-1)
        logw = torch.where(has_obs.unsqueeze(-1), logw, fast.logw)
        pred_ll = torch.where(has_obs, pred_ll, torch.zeros_like(pred_ll))

        # ---- misspecification statistic
        z, h_pred = self.surprise_z(dist, fast.logw.detach(), outcomes.y_mask, pred_ll.detach(), generator,
                                    y_obs=outcomes.y, censored=outcomes.censored, lik=outcomes.lik)
        z = torch.where(has_obs, z, torch.zeros_like(z))
        sur = torch.cat([fast.surprise[:, 1:], z.unsqueeze(-1).to(fast.surprise.dtype)], 1)
        sur = torch.where(has_obs.unsqueeze(-1), sur, fast.surprise)
        wz, nz = self.window_stat(sur)
        cusum, _ = self.cusum_stat(sur)
        cusum_prev, _ = self.cusum_stat(fast.surprise)

        # ---- learned proposal move + scale contraction
        s = self.event_summary(feat, outcomes)
        fit = (loglik - loglik.mean(-1, keepdim=True)) / (loglik.std(-1, keepdim=True, unbiased=False) + 1.0)
        delta = self.propose(fast.particles, s, fit, logw)
        pscale = torch.maximum(fast.pscale * torch.sigmoid(self.contract_raw), torch.full_like(fast.pscale,
                                                                                              self.min_scale))
        eps = randn((B, K, Dp), generator, dev)
        moved = fast.particles + delta + self.jitter * pscale * eps
        upd = has_obs.view(B, 1, 1)
        particles = torch.where(upd, moved, fast.particles)
        pscale = torch.where(upd, pscale, fast.pscale)

        # ---- ESS-triggered systematic resampling + rejuvenation
        w, ess, _ = self.weight_stats(logw)
        resampled = has_obs & (ess < self.cfg.ess_frac * K) if self.resample_enabled else torch.zeros_like(has_obs)
        if bool(resampled.any()):
            idx = systematic_resample(w.detach(), generator)
            g = idx.unsqueeze(-1).expand(B, K, Dp)
            p_rs = torch.gather(particles, 1, g)
            s_rs = torch.gather(pscale, 1, g)
            p_rs = p_rs + self.rejuv * s_rs * randn((B, K, Dp), generator, dev)
            r3 = resampled.view(B, 1, 1)
            particles = torch.where(r3, p_rs, particles)
            pscale = torch.where(r3, s_rs, pscale)
            logw = torch.where(resampled.unsqueeze(-1), torch.full_like(logw, -math.log(K)), logw)
            self.counters["resampled_rows"] += int(resampled.sum())

        # ---- misspecification -> uncertainty expansion (or, when already saturated, a persistent flag)
        mu0, sc0 = self.prior(B, Z_prev, dev)
        cap = (self.max_scale_mult * sc0).unsqueeze(1).detach()                     # [B,1,Dp]
        alarm = torch.zeros_like(has_obs)
        if self.misspec_enabled:
            alarm = has_obs & (nz >= self.min_steps) & (cusum > self.cusum_h)
        level = (pscale.detach() / cap).flatten(1).median(1).values                   # [B] in (0, 1]
        saturated = level >= 1.0 / self.inflate
        expanded = alarm & ~saturated
        persistent = alarm & saturated
        if bool(expanded.any()):
            n_inj = max(1, int(round(self.inject_frac * K)))
            anchor = mu0.unsqueeze(1) + self.inflate * sc0.unsqueeze(1) * randn((B, n_inj, Dp), generator, dev)
            fresh = anchor + self.propose(anchor, s, torch.zeros(B, n_inj, device=dev),
                                          torch.full((B, n_inj), -math.log(K), device=dev))
            low = torch.topk(-logw.detach(), n_inj, dim=1).indices                # lowest-weight particles
            gi = low.unsqueeze(-1).expand(B, n_inj, Dp)
            p_ex = particles.scatter(1, gi, fresh)
            s_ex = (pscale * self.inflate).scatter(1, gi, (self.inflate * sc0).unsqueeze(1).expand(B, n_inj, Dp))
            e3 = expanded.view(B, 1, 1)
            particles = torch.where(e3, p_ex, particles)
            pscale = torch.where(e3, s_ex, pscale)
            logw = torch.where(expanded.unsqueeze(-1), torch.full_like(logw, -math.log(K)), logw)
            sur = torch.where(expanded.unsqueeze(-1), torch.full_like(sur, float("nan")), sur)
            rows = expanded.nonzero().flatten().tolist()
            self.counters["expanded_rows"] += len(rows)
            for r in rows:
                rec = {"kind": "expansion", "row": r, "step": int(fast.steps[r]) + 1, "cusum": float(cusum[r]),
                       "window_z": float(wz[r]), "n_injected": n_inj, "ess_before": float(ess[r]),
                       "scale_level": float(level[r])}
                self.misspec_log.append(rec)
                log.info("belief misspecification: row %d step %d CUSUM %.2f > %.2f -> injected %d particles, "
                         "scale x%.1f", r, rec["step"], rec["cusum"], self.cusum_h, n_inj, self.inflate)
        if bool(persistent.any()):
            # recent expansions did not explain the data: keep the weights, pin the uncertainty at the cap
            pscale = torch.where(persistent.view(B, 1, 1), torch.maximum(pscale, cap.expand_as(pscale)), pscale)
            rows = persistent.nonzero().flatten().tolist()
            self.counters["persistent_row_steps"] += len(rows)
            onset = (cusum_prev <= self.cusum_h).tolist()       # the alarm starts now (not carried from last step)
            for r in rows:
                if onset[r]:
                    self.counters["persistent_episodes"] += 1
                    rec = {"kind": "persistent", "row": r, "step": int(fast.steps[r]) + 1, "cusum": float(cusum[r]),
                           "window_z": float(wz[r]), "scale_level": float(level[r])}
                    self.misspec_log.append(rec)
                    log.warning("belief misspecification persists: row %d step %d CUSUM %.2f with uncertainty "
                                "saturated -> no further resets; scale pinned at %.1f x prior", r, rec["step"],
                                rec["cusum"], self.max_scale_mult)
        self.counters["alarm_rows"] += int(alarm.sum())
        pscale = torch.minimum(pscale, cap.expand_as(pscale))

        new = FastState(particles, logw, pscale, fast.bayes_mu, fast.bayes_P, fast.bayes_noise, fast.bayes_n, sur,
                        fast.steps + has_obs.long(), [dict(c) for c in fast.channels])
        info = {"ess": ess, "loglik": loglik, "surprise_z": z, "resampled": resampled, "expanded": expanded,
                "persistent": persistent, "alarm": alarm, "cusum": cusum,
                "pred_ll": pred_ll, "pred_entropy": torch.where(has_obs, h_pred, torch.zeros_like(h_pred)),
                "window_z": wz, "feat": feat}
        return new, info

    def count_parameters(self) -> int:
        return count_parameters(self)


def detach_fast(fast: FastState) -> FastState:
    """Cut the autograd graph through a FastState (truncated backpropagation through adaptation)."""
    return FastState(*(t.detach() for t in (fast.particles, fast.logw, fast.pscale, fast.bayes_mu, fast.bayes_P,
                                            fast.bayes_noise, fast.bayes_n, fast.surprise, fast.steps)),
                     [dict(c) for c in fast.channels])
