"""Belief / Bayes-last-layer runtime of CamUnifiedModel (moved out of model.py by the seam commit, plan 2026-11-26 §3;
owner W6).

Outcome predictions mixed over belief particles and combined with the Kalman/NIG last layer, the belief updates from
realised outcomes, and the support/query adaptation objective:

    base_moments, predict_outcomes, particle_dist          predictive moments [B,Q] of the action-conditioned latent
    kalman_on, kalman_bound, kalman_predict, combine,      the Kalman/NIG residual layer on bound continuous outcomes
    kalman_log_prob, kalman_update, residual_batch,
    channels_of, continuous_mask, bayes_prior
    mixture_ll, mixture_loglik                              particle-mixture log-likelihoods
    belief_view                                             what the belief updates condition on (reward_in_beliefs)
    observe_outcomes                                        realised outcomes -> particle + Kalman updates (no grad)
    adapt_loss                                              L_adapt = -log p(Q | B_S, query actions)

DOM path (plan 2026-11-26 §1 Beliefs / §3 W6; `dom_beliefs(model)`: the `dom*` presets):

    fused_predictive / FusedPredictive                     gated fusion p = g p_kal + (1 - g) p_mix of the particle
                                                            mixture and the Kalman/NIG predictive (no variance
                                                            replacement; the Kalman base is the particle-free head mean)
    predictive_log_prob, predictive_moments                 the full predictive actually used (decide, LP, adapt)
    belief_change(fast_prev, fast_new, info=None)           differentiable belief-change / information-gain features
                                                            {kl_weights, kalman_ig, d_entropy, info_gain} [B]
    adapt_loss_all_k                                        separation-correct support/query loss over every support
                                                            size in one pass (support-end world-model rollout, reset
                                                            at done, Kalman dropout, mixture auxiliary term)
    observation-only outcomes                               in reward_mode "observation_only" the realised reward is an
                                                            unobserved outcome (never a belief / Kalman / WM target)
    world-model likelihood                                  particle weights from feat = outcome_head.features(Z_hat)

Every function takes the model first; `CamUnifiedModel` keeps one-line delegating methods of the same names (helpers
that were private keep their leading underscore there: `_particle_dist`, `_continuous`, `_kalman_on`,
`_bayes_prior`, `_padded`). Calls between these functions go through the model's methods (`model.base_moments`,
`model.kalman_update`, ...), so instance-level overrides (e.g. capabilities/imagination.py patches
`model.kalman_update`) keep applying. `belief_change` and the `*_modules` helpers take modules / states only.

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers are imported inside the functions that use them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from ..interfaces import FastState, OutcomeBatch, TokenSet
from ..outcome_spec import CONTINUOUS, bind_channels, build_outcome_batch, lookup_channels, make_room, with_targets

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamState, CamUnifiedModel, Decision, FilterResult

F64 = torch.float64


# ============================================================================ DOM switches

def dom_beliefs(model) -> bool:
    """The DOM belief path (WM-prior likelihood features, full-predictive LP, all-k adaptation, belief-change
    features). On for the `dom*` presets; a future config field `belief_dom` overrides (change request to W1)."""
    cfg = model.cfg
    flag = getattr(cfg, "belief_dom", None)
    if flag is not None:
        return bool(flag)
    return str(getattr(cfg, "preset", "") or "").startswith("dom")


def fusion_on(model) -> bool:
    """Gated fusion of the particle mixture and the Kalman/NIG predictive (BayesLastLayer(fusion=True))."""
    return kalman_on(model) and bool(getattr(model.bayes, "fusion", False)) and \
        getattr(model.bayes, "gate", None) is not None


def reward_observed(model) -> bool:
    """False in reward_mode "observation_only": the realised verified return is not an actor-side outcome."""
    return str(getattr(model.cfg, "reward_mode", "legacy")) != "observation_only"


# ============================================================================ outcome predictions (particles + Kalman)

def particle_dist(model: "CamUnifiedModel", feat: torch.Tensor, ob: OutcomeBatch, fast: Optional[FastState]):
    if model.cfg.use_fast_adaptation and fast is not None:
        return model.outcome_head.dist(feat, ob, cond=fast.particles)
    return model.outcome_head.dist(feat, ob)


def base_moments(model: "CamUnifiedModel", Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch
                 ) -> tuple:
    """Outcome-head predictive mean/var [B,Q] from the action-conditioned latent Z_hat: the mixture over belief
    particles (weights w) with fast adaptation, the unconditioned head otherwise. Also returns the features."""
    from ..model import check_shape, dist_mean, dist_var
    B, Q = ob.lik.shape
    feat = model.outcome_head.features(Z_hat, ob)
    check_shape(feat, (B, Q, model.cfg.d_model), "outcome_head.features")
    d = model._particle_dist(feat, ob, fast)
    if model.cfg.use_fast_adaptation and fast is not None:
        mk, vk = dist_mean(d).float(), dist_var(d).float()
        check_shape(mk, (B, model.cfg.n_particles, Q), "outcome_head.dist(cond) mean")
        w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
        mean = (w * mk).sum(1)
        var = (w * (vk + mk * mk)).sum(1) - mean * mean
    else:
        mean, var = dist_mean(d).float(), dist_var(d).float()
        check_shape(mean, (B, Q), "outcome_head.dist mean")
    return mean, var.clamp_min(0.0), feat


def continuous_mask(ob: OutcomeBatch) -> torch.Tensor:
    """[B,Q] bool: queries with a continuous likelihood (the Kalman layer's domain)."""
    c = torch.zeros_like(ob.y_mask)
    for k in CONTINUOUS:
        c = c | (ob.lik == k)
    return c


def kalman_on(model: "CamUnifiedModel") -> bool:
    return model.cfg.use_fast_adaptation and model.bayes is not None


def channels_of(model: "CamUnifiedModel", fast: FastState, ob: OutcomeBatch) -> OutcomeBatch:
    """The same queries with `channel` looked up in fast.channels by key (continuous keys only)."""
    B, Q = ob.lik.shape
    cont = model._continuous(ob).cpu()
    ch = torch.full((B, Q), -1, dtype=torch.long)
    for b in range(B):
        table = fast.channels[b]
        keys = ob.keys[b] if ob.keys is not None else []
        for q, k in enumerate(keys):
            if k is not None and bool(cont[b, q]):
                ch[b, q] = int(table.get(k, -1))
    return OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y, ob.censored, ob.y_mask, ch.to(ob.lik.device), ob.keys)


def kalman_bound(model: "CamUnifiedModel", ob: OutcomeBatch) -> torch.Tensor:
    return model._continuous(ob) & (ob.channel >= 0) & ob.q.mask


def kalman_predict(model: "CamUnifiedModel", fast: FastState, feat: torch.Tensor, ob: OutcomeBatch) -> tuple:
    """Residual correction of the Kalman/NIG last layer: (corr_mean [B,Q], corr_var [B,Q], phi [B,Q,F],
    bound [B,Q])."""
    from ..model import check_shape
    B, Q = ob.lik.shape
    phi = model.bayes.features(feat, ob)
    cm, cv = model.bayes.predict(fast, phi, ob)
    check_shape(cm, (B, Q), "bayes.predict mean")
    return cm.float(), cv.float().clamp_min(0.0), phi, model.kalman_bound(ob)


def combine(model: "CamUnifiedModel", head_mean, head_var, corr_mean, corr_var, bound) -> tuple:
    """Bound continuous queries: (head + correction mean, the Kalman predictive variance); others: the head.
    (B's BayesLastLayer.combine when present: same semantics.)"""
    if hasattr(model.bayes, "combine"):
        m, v = model.bayes.combine(head_mean, head_var, corr_mean, corr_var, bound)
        return m.float(), v.float()
    return torch.where(bound, head_mean + corr_mean, head_mean), torch.where(bound, corr_var, head_var)


def kalman_log_prob(model: "CamUnifiedModel", fast: FastState, phi: torch.Tensor, ob: OutcomeBatch,
                    head_mean: torch.Tensor) -> torch.Tensor:
    """log p(y | head mean + Kalman residual predictive) [B,Q] (0 where unbound/unobserved). Differentiable in
    phi, head_mean and the Kalman statistics (hence the learned prior / noise)."""
    if hasattr(model.bayes, "log_prob"):
        return model.bayes.log_prob(fast, phi, ob, head_mean).float()
    m, v = model.bayes.predict(fast, phi, ob)
    v = v.float().clamp_min(1e-8)
    r = ob.y.float() - head_mean.float()
    lp = -0.5 * (math.log(2 * math.pi) + torch.log(v) + (r - m.float()) ** 2 / v)
    return torch.where(model.kalman_bound(ob) & ob.y_mask, lp, torch.zeros_like(lp))


def residual_batch(model: "CamUnifiedModel", ob: OutcomeBatch, base_mean: torch.Tensor) -> OutcomeBatch:
    """Targets r = y - head mean on observed continuous queries (the Kalman layer's regression target)."""
    cont = model._continuous(ob) & ob.y_mask
    return with_targets(ob, torch.where(cont, ob.y.float() - base_mean.float(), torch.zeros_like(ob.y.float())),
                        cont)


def kalman_update(model: "CamUnifiedModel", fast: FastState, feat: torch.Tensor, ob: OutcomeBatch,
                  base_mean: torch.Tensor) -> FastState:
    """Kalman/NIG update with the residuals of `ob` against `base_mean` (differentiable closed form when the
    Bayes layer is). The layer allocates slots for new observed keys when it can (`assign_channels`)."""
    resid = model.residual_batch(ob, base_mean)
    phi = model.bayes.features(feat, resid)
    return model.bayes.update(fast, phi, resid)


def predict_outcomes(model: "CamUnifiedModel", Z_hat: torch.Tensor, fast: Optional[FastState], ob: OutcomeBatch
                     ) -> tuple:
    """Predictive mean/var [B,Q]: base_moments combined, on continuous outcomes with a bound channel, with the
    Kalman/NIG last layer (mean head + phi^T mu_c, variance = Kalman predictive variance). With gated fusion
    (`fusion_on`) the moments of the fused two-expert predictive instead (`fused_predictive`)."""
    if fusion_on(model) and fast is not None:
        feat = model.outcome_head.features(Z_hat, ob)
        fp = fused_predictive(model, fast, feat, ob)
        return fp.mean(), fp.var(), feat
    mean, var, feat = model.base_moments(Z_hat, fast, ob)
    if model._kalman_on() and fast is not None:
        cm, cv, _, bound = model.kalman_predict(fast, feat, ob)
        mean, var = model.combine(mean, var, cm, cv, bound)
    return mean, var, feat


def belief_view(model: "CamUnifiedModel", ob: OutcomeBatch) -> OutcomeBatch:
    """The outcomes the belief / Kalman updates condition on (the realised reward is masked unless
    reward_in_beliefs)."""
    if model.reward_in_beliefs or ob.keys is None:
        return ob
    m = ob.y_mask.clone()
    for b, keys in enumerate(ob.keys):
        for q, k in enumerate(keys):
            if k == "reward":
                m[b, q] = False
    return OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y, ob.censored, m, ob.channel, ob.keys)


def mixture_ll(model: "CamUnifiedModel", fast: FastState, feat: torch.Tensor, ob: OutcomeBatch,
               use: torch.Tensor) -> torch.Tensor:
    """log sum_k w_k prod_{q in use} p(y_q | feat, m_k) [B] (0 where nothing is used)."""
    from ..model import dist_log_prob
    B, Q = ob.lik.shape
    d = model._particle_dist(feat, ob, fast)
    K = fast.logw.shape[1]
    y = ob.y.unsqueeze(1).expand(B, K, Q)
    cens = None if ob.censored is None else ob.censored.unsqueeze(1).expand(B, K, Q)
    lp = dist_log_prob(d, y, cens).float()
    ll = torch.where(use.unsqueeze(1), lp, torch.zeros_like(lp)).sum(-1)              # [B,K]
    return torch.logsumexp(torch.log_softmax(fast.logw.float(), -1) + ll, -1)


def mixture_loglik(model: "CamUnifiedModel", fast: FastState, Z_hat: torch.Tensor, ob: OutcomeBatch
                   ) -> torch.Tensor:
    """Particle-mixture log p(y | B) per env [B] divided by the number of observed outcomes (per-outcome nats)."""
    feat = model.outcome_head.features(Z_hat, ob)
    return model.mixture_ll(fast, feat, ob, ob.y_mask) / ob.y_mask.float().sum(-1).clamp_min(1.0)


# ============================================================================ DOM: gated fusion, full predictive

@dataclass
class FusedPredictive:
    """The full outcome predictive of the DOM belief path for one OutcomeBatch [B,Q].

    Particle mixture: `dist` (the particle-conditioned head distribution, batch [B,K,Q]) with normalised log-weights
    `logw` [B,K]; `mix_mean`/`mix_var` [B,Q] its moments. Kalman/NIG expert (absent when the Kalman layer is off or
    dropped): `kal_mean` (particle-free head mean + phi^T mu_c) and `kal_var` [B,Q] of y itself, `kal_lp_fn(ob) ->
    [B,Q]` its log density (0 where unbound) and `bound` [B,Q] (continuous query on a bound channel); `gate` [B,Q]
    the Kalman expert's weight. RBPF extension point: per-particle Kalman statistics would make the kal_* tensors
    [B,K,Q] and the log density would mix them over particles with `logw` (the per-query gate then becomes
    unnecessary); every consumer goes through `mean`, `var`, `query_log_prob` and `joint_log_prob`."""
    dist: object
    logw: torch.Tensor
    mix_mean: torch.Tensor
    mix_var: torch.Tensor
    bound: torch.Tensor
    gate: Optional[torch.Tensor] = None
    kal_mean: Optional[torch.Tensor] = None
    kal_var: Optional[torch.Tensor] = None
    kal_lp_fn: Optional[object] = None

    @property
    def fused(self) -> torch.Tensor:
        """[B,Q] queries predicted by the two-expert fusion (bound continuous queries with the Kalman expert on)."""
        if self.gate is None:
            return torch.zeros_like(self.bound)
        return self.bound

    def mean(self) -> torch.Tensor:
        if self.gate is None:
            return self.mix_mean
        g = self.gate
        m = g * self.kal_mean + (1 - g) * self.mix_mean
        return torch.where(self.fused, m, self.mix_mean)

    def var(self) -> torch.Tensor:
        if self.gate is None:
            return self.mix_var
        g = self.gate
        m = self.mean()
        second = g * (self.kal_var + self.kal_mean ** 2) + (1 - g) * (self.mix_var + self.mix_mean ** 2)
        return torch.where(self.fused, (second - m * m).clamp_min(0.0), self.mix_var)

    def particle_log_prob(self, ob: OutcomeBatch) -> torch.Tensor:
        """Per-particle log-likelihood of every query [B,K,Q] (0 where unobserved)."""
        from ..model import dist_log_prob
        B, Q = ob.lik.shape
        K = self.logw.shape[1]
        y = torch.where(ob.y_mask, ob.y, torch.zeros_like(ob.y)).unsqueeze(1).expand(B, K, Q)
        cens = None if ob.censored is None else ob.censored.unsqueeze(1).expand(B, K, Q)
        lp = dist_log_prob(self.dist, y, cens).float()
        lp = torch.nan_to_num(lp, nan=-1e4, neginf=-1e4, posinf=1e4)
        return torch.where(ob.y_mask.unsqueeze(1), lp, torch.zeros_like(lp))

    def query_log_prob(self, ob: OutcomeBatch) -> torch.Tensor:
        """Marginal log p(y_q | belief) per query [B,Q] (0 where unobserved)."""
        lpk = self.particle_log_prob(ob)
        lw = torch.log_softmax(self.logw.float(), -1).unsqueeze(-1)
        mix = torch.logsumexp(lw + lpk, 1)
        out = mix
        if self.gate is not None:
            lk = self.kal_lp_fn(ob).float()
            g = self.gate.clamp(1e-6, 1 - 1e-6)
            fz = torch.logaddexp(torch.log(g) + lk, torch.log1p(-g) + mix)
            out = torch.where(self.fused & ob.y_mask, fz, mix)
        return torch.where(ob.y_mask, out, torch.zeros_like(out))

    def joint_log_prob(self, ob: OutcomeBatch) -> torch.Tensor:
        """log p(y | belief) of a step's observed outcomes [B]: the particle mixture of the joint likelihood of the
        non-fused queries times the fused per-query predictives of the fused ones (the Kalman layer treats channels
        independently)."""
        lpk = self.particle_log_prob(ob)
        fz = self.fused & ob.y_mask
        rest = (ob.y_mask & ~fz).unsqueeze(1)
        lw = torch.log_softmax(self.logw.float(), -1)
        joint_rest = torch.logsumexp(lw + torch.where(rest, lpk, torch.zeros_like(lpk)).sum(-1), -1)
        if self.gate is None or not bool(fz.any()):
            return joint_rest
        q = self.query_log_prob(ob)
        return joint_rest + torch.where(fz, q, torch.zeros_like(q)).sum(-1)


def fusion_features(bayes, fast: FastState, ob: OutcomeBatch, kal_scale2: torch.Tensor, mix_var: torch.Tensor
                    ) -> torch.Tensor:
    """Detached gate inputs [B,Q,7] (BayesLastLayer FusionGate): log1p(n_c), log Kalman scale^2, log mixture variance,
    their difference, the particle filter's mean recent surprise z (clipped, /8), log(noise estimate / prior noise)
    and the Kalman expert's realised per-observation log evidence plus the mixture's predictive entropy (clipped)."""
    with torch.no_grad():
        st = bayes.channel_stats(fast, ob)
        ls = torch.log(kal_scale2.float().clamp_min(1e-8))
        lm = torch.log(mix_var.float().clamp_min(1e-8))
        sur = fast.surprise.float()
        valid = ~torch.isnan(sur)
        zbar = torch.where(valid, sur, torch.zeros_like(sur)).sum(-1) / valid.sum(-1).clamp_min(1)
        zbar = (zbar.clamp(-8.0, 8.0) / 8.0).unsqueeze(-1).expand_as(ls)
        ln = torch.log(st["noise"].float().clamp_min(1e-8)) - torch.log(bayes.noise().detach().float().clamp_min(1e-8))
        ev = bayes.log_evidence(fast, ob) if hasattr(bayes, "log_evidence") else torch.zeros_like(ls)
        h_mix = 0.5 * (math.log(2 * math.pi * math.e) + lm)
        evf = torch.where(st["n"] > 0, (ev + h_mix).clamp(-10.0, 10.0), torch.zeros_like(ev))
        return torch.stack([torch.log1p(st["n"].float()), ls, lm, ls - lm, zbar, ln, evf], -1)


def kalman_base_modules(head, feat: torch.Tensor, ob: OutcomeBatch) -> torch.Tensor:
    """The Kalman expert's base mean [B,Q] on the fusion path: the particle-free head mean (outcome_head.dist without
    particle conditioning, trained by the world loss). It depends on the action-conditioned latent only, so the
    residual the Kalman layer is updated with at observation time and the one it predicts later have the same
    decomposition, and the particles are not cancelled by the Kalman correction."""
    from ..model import dist_mean
    return dist_mean(head.dist(feat, ob)).float()


def fused_predictive_modules(head, bayes, fast: FastState, feat: torch.Tensor, ob: OutcomeBatch,
                             kalman: bool = True) -> FusedPredictive:
    """`FusedPredictive` from the modules: head = OutcomeHead, bayes = BayesLastLayer (or None), feat [B,Q,D] the
    outcome features of the action-conditioned latent, ob with channels looked up (keys) for the Kalman part.
    ``kalman=False`` (no Bayes layer, or the adaptation objective's Kalman dropout): the particle mixture only."""
    from ..model import dist_mean, dist_var
    d = head.dist(feat, ob, cond=fast.particles)
    mk, vk = dist_mean(d).float(), dist_var(d).float()
    w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
    m_mix = (w * mk).sum(1)
    v_mix = ((w * (vk + mk * mk)).sum(1) - m_mix * m_mix).clamp_min(0.0)
    bound0 = torch.zeros_like(ob.y_mask)
    if not kalman or bayes is None:
        return FusedPredictive(d, fast.logw, m_mix, v_mix, bound0)
    base0 = kalman_base_modules(head, feat, ob)
    phi = bayes.features(feat, ob)
    p = bayes.predictive(fast, phi, ob)
    bound = p["bound"] & continuous_mask(ob) & ob.q.mask
    if not bool(bound.any()):
        return FusedPredictive(d, fast.logw, m_mix, v_mix, bound0)
    loc = base0 + p["loc"].float()
    s2 = p["scale2"]
    if getattr(bayes, "noise_mode", "nig") == "nig":
        nu = p["dof"]
        kvar = (s2 * torch.where(nu > 2.05, nu / (nu - 2), torch.full_like(nu, 41.0))).float()
    else:
        kvar = s2.float()
    gate = bayes.gate(fusion_features(bayes, fast, ob, s2, v_mix)) if getattr(bayes, "gate", None) is not None \
        else torch.ones_like(m_mix)

    def kal_lp(o: OutcomeBatch) -> torch.Tensor:
        return bayes.log_prob(fast, phi, o, base0).float()
    return FusedPredictive(d, fast.logw, m_mix, v_mix, bound, gate, loc, kvar, kal_lp)


def fused_predictive(model: "CamUnifiedModel", fast: FastState, feat: torch.Tensor, ob: OutcomeBatch,
                     kalman: Optional[bool] = None) -> FusedPredictive:
    """Model-level `fused_predictive_modules` (channels looked up in fast.channels by key)."""
    use_k = model._kalman_on() if kalman is None else (bool(kalman) and model._kalman_on())
    obc = model.channels_of(fast, ob) if use_k else ob
    return fused_predictive_modules(model.outcome_head, model.bayes if use_k else None, fast, feat, obc, use_k)


def kalman_base(model: "CamUnifiedModel", feat: torch.Tensor, ob: OutcomeBatch, fast: Optional[FastState] = None
                ) -> torch.Tensor:
    """Base mean the Kalman residuals are taken against: the particle-free head mean with fusion, the particle-
    mixture mean (legacy composition) otherwise."""
    if fusion_on(model):
        return kalman_base_modules(model.outcome_head, feat, ob)
    from ..model import dist_mean
    d = model._particle_dist(feat, ob, fast)
    if model.cfg.use_fast_adaptation and fast is not None:
        w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
        return (w * dist_mean(d).float()).sum(1)
    return dist_mean(d).float()


def predictive_moments(model: "CamUnifiedModel", fast: Optional[FastState], feat: torch.Tensor, ob: OutcomeBatch
                       ) -> tuple:
    """(mean [B,Q], var [B,Q]) of the predictive actually used: fused (gated fusion), or the legacy composition
    (mixture moments, Kalman correction + variance replacement on bound continuous queries)."""
    if fusion_on(model) and fast is not None:
        fp = fused_predictive(model, fast, feat, ob)
        return fp.mean(), fp.var()
    from ..model import dist_mean, dist_var
    d = model._particle_dist(feat, ob, fast)
    if model.cfg.use_fast_adaptation and fast is not None:
        mk, vk = dist_mean(d).float(), dist_var(d).float()
        w = torch.softmax(fast.logw.float(), -1).unsqueeze(-1)
        mean = (w * mk).sum(1)
        var = ((w * (vk + mk * mk)).sum(1) - mean * mean).clamp_min(0.0)
    else:
        mean, var = dist_mean(d).float(), dist_var(d).float()
    if model._kalman_on() and fast is not None:
        cm, cv, _, bound = model.kalman_predict(fast, feat, ob)
        mean, var = model.combine(mean, var, cm, cv, bound)
    return mean, var


def predictive_log_prob(model: "CamUnifiedModel", fast: FastState, Z_hat: torch.Tensor, ob: OutcomeBatch,
                        per_outcome: bool = True, feat: Optional[torch.Tensor] = None,
                        kalman: Optional[bool] = None) -> torch.Tensor:
    """log p(o | B, Z_hat) of the observed outcomes under the full predictive actually used [B]: with gated fusion
    the fused predictive (`FusedPredictive.joint_log_prob`); with the legacy composition the Kalman-combined
    predictive on bound continuous queries plus the particle mixture on the others (the adaptation objective's
    scoring); without a Kalman layer the particle mixture. ``per_outcome`` divides by the number of observed
    outcomes (nats per outcome; 0 for rows with none). Differentiable."""
    feat = model.outcome_head.features(Z_hat, ob) if feat is None else feat
    use_k = model._kalman_on() if kalman is None else (bool(kalman) and model._kalman_on())
    if not model.cfg.use_fast_adaptation or fast is None:
        from ..model import dist_log_prob
        d = model.outcome_head.dist(feat, ob)
        lp = dist_log_prob(d, torch.where(ob.y_mask, ob.y, torch.zeros_like(ob.y)), ob.censored).float()
        ll = torch.where(ob.y_mask, lp, torch.zeros_like(lp)).sum(-1)
    elif fusion_on(model) or not use_k:
        ll = fused_predictive(model, fast, feat, ob, kalman=use_k).joint_log_prob(ob)
    else:
        obc = model.channels_of(fast, ob)
        base, _, _ = model.base_moments(Z_hat, fast, obc)
        kb = model.kalman_bound(obc) & obc.y_mask
        ll = model.mixture_ll(fast, feat, obc, obc.y_mask & ~kb)
        if bool(kb.any()):
            phi = model.bayes.features(feat, obc)
            lpk = model.kalman_log_prob(fast, phi, obc, base)
            ll = ll + torch.where(kb, lpk, torch.zeros_like(lpk)).sum(-1)
    ll = torch.where(ob.y_mask.any(-1), ll, torch.zeros_like(ll))
    if per_outcome:
        ll = ll / ob.y_mask.float().sum(-1).clamp_min(1.0)
    return ll


# ============================================================================ DOM: belief change / information gain

def _logdet_spd(P: torch.Tensor) -> torch.Tensor:
    """log det of symmetric positive definite matrices [..., F, F] (float64 Cholesky; eigenvalue fallback)."""
    P = P.to(F64)
    L, info = torch.linalg.cholesky_ex(P)
    ld = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-300)).sum(-1)
    bad = info != 0
    if bool(bad.any()):
        ev = torch.linalg.eigvalsh(0.5 * (P + P.transpose(-1, -2))).clamp_min(1e-300)
        ld = torch.where(bad, torch.log(ev).sum(-1), ld)
    return ld


def belief_change(fast_prev: FastState, fast_new: FastState, info: Optional[dict] = None) -> dict:
    """Belief-change / information-gain features of one belief update (differentiable in both states; plan §2: the
    IG_t term of R_psi and the VOI / curriculum signals). All values [B] float32:

    * ``kl_weights``  KL(w_post || w_prev) over the particle index: the information the step gave about which
      hypothesis holds. w_post are the reweighted weights before resampling (``info["logw_post"]`` from
      `BeliefModule.update`, indexed like the previous particles); without ``info`` the new state's weights are used
      (exact when no row was resampled or expanded);
    * ``d_entropy``   H(w_prev) - H(w_post): the drop of the weight entropy (can be negative: a surprising
      observation can flatten the weights);
    * ``kalman_ig``   sum over channels updated in this step of 0.5 (log det V_prev - log det V_new), V = Sigma /
      s2_hat the noise-scaled posterior covariance: the exact Gaussian information gain about each channel's
      weights given the noise (0.5 log(1 + phi^T V phi) per scalar observation); channels whose count did not grow
      (unobserved, or reset to the prior) contribute 0;
    * ``info_gain``   kl_weights + kalman_ig (the actor-visible IG_t of R_psi)."""
    lw_prev = torch.log_softmax(fast_prev.logw.float(), -1)
    post = info.get("logw_post") if isinstance(info, dict) else None
    lw_post = torch.log_softmax((fast_new.logw if post is None else post).float(), -1)
    w_post = lw_post.exp()
    kl = (w_post * (lw_post - lw_prev)).sum(-1).clamp_min(0.0)
    h_prev = -(lw_prev.exp() * lw_prev).sum(-1)
    h_post = -(w_post * lw_post).sum(-1)
    B = fast_prev.logw.shape[0]
    dev = fast_prev.logw.device
    kig = torch.zeros(B, device=dev, dtype=torch.float32)
    if fast_prev.bayes_P is not None and fast_new.bayes_P is not None and fast_prev.bayes_P.shape[1] > 0:
        upd = fast_new.bayes_n.float() > fast_prev.bayes_n.float()                        # [B,C]
        if bool(upd.any()):
            s2p = fast_prev.bayes_noise.to(F64).clamp_min(1e-300)
            s2n = fast_new.bayes_noise.to(F64).clamp_min(1e-300)
            Fd = fast_prev.bayes_P.shape[-1]
            ldp = _logdet_spd(fast_prev.bayes_P) - Fd * torch.log(s2p)
            ldn = _logdet_spd(fast_new.bayes_P) - Fd * torch.log(s2n)
            kig = torch.where(upd, 0.5 * (ldp - ldn), torch.zeros_like(ldp)).sum(-1).float()
    return {"kl_weights": kl, "d_entropy": h_prev - h_post, "kalman_ig": kig, "info_gain": kl + kig}


# ============================================================================ outcomes -> beliefs

def observe_outcomes(model: "CamUnifiedModel", state: "CamState", decision: "Decision", results: list,
                     next_obs: list, rewards: Sequence[float]) -> dict:
    """Realise the outcomes of `decision` (OutcomeBatch with y), compute the hindsight learning progress of the
    previous belief update, then update the particle beliefs and the Kalman/NIG last layer (use_fast_adaptation).
    Runs without gradients in acting (eval) mode, whatever mode the caller left the model in.
    Kalman channels: slots are freed for the realised continuous keys by least-recently-realised eviction
    (`make_room`), then allocated by the Bayes layer (B's `assign_channels`) or by the model; evictions and keys
    that do not fit are counted. Returns {"outcomes", "specs", "belief_info", "lp" [B] or None, "belief_after",
    "loglik" [B], "channels" [B] {"evicted", "unbound"}, "dropped" [B], "belief_change"}.
    DOM path (`dom_beliefs`): the particle weights use the world-model likelihood (feat of decision.Z_hat), 'loglik'
    and 'lp' are computed with the full predictive actually used (`predictive_log_prob`), the Kalman residual base is
    `kalman_base` and "belief_change" holds the detached `belief_change` features [B] of the update. In
    reward_mode "observation_only" (or with ``rewards`` None) the realised reward is recorded as unobserved."""
    from ..model import acting_mode
    with torch.no_grad(), acting_mode(model):
        return _observe_outcomes(model, state, decision, results, next_obs, rewards)


def _observe_outcomes(model, state, decision, results, next_obs, rewards) -> dict:
    from ..model import belief_summary, check_shape, fast_detach
    cfg = model.cfg
    B = state.B
    dom = dom_beliefs(model)
    # observation_only (or no reward stream): the realised reward is an unobserved outcome for every consumer
    r_obs = reward_observed(model) and rewards is not None
    rew = [(float(rewards[b]) if (r_obs and rewards[b] is not None) else None) for b in range(B)]
    specs = [state.trackers[b].realize(decision.inputs.obs_lists[b], decision.actions[b], results[b], next_obs[b],
                                       rew[b]) for b in range(B)]
    dropped = [dict(state.trackers[b].last_dropped) for b in range(B)]
    ch_stats = [{"evicted": 0, "unbound": 0} for _ in range(B)]
    if model._kalman_on() and state.fast is not None:
        pv, nv = model._bayes_prior()
        chans = []
        for b in range(B):
            if model._bayes_allocates:
                make_room(state.fast, b, specs[b], state.trackers[b].last_seen, pv, nv, ch_stats[b])
                chans.append(lookup_channels(state.fast, b, specs[b]))
            else:
                chans.append(bind_channels(state.fast, b, specs[b], state.trackers[b].last_seen, pv, nv,
                                           ch_stats[b]))
    else:
        chans = [[-1] * len(s) for s in specs]
    for d in dropped:
        for k, n in d.items():
            model._count(f"outcome_dropped/{k}", n)
    with model._autocast():
        ob = build_outcome_batch(specs, model.query_encoder, state.registries, chans, model.device)
        out = {"outcomes": ob, "specs": specs, "belief_info": {}, "lp": None, "loglik": None,
               "channels": ch_stats, "dropped": dropped, "belief_change": None}
        if cfg.use_fast_adaptation and state.fast is not None:
            obb = model.belief_view(ob)
            before = state.fast.clone()
            if dom:
                # the full predictive actually used (fused particles + Kalman) on the world-model prior latent
                feat = model.outcome_head.features(decision.Z_hat, obb)
                ll_now = predictive_log_prob(model, state.fast, decision.Z_hat, obb, feat=feat)
                out["loglik"] = ll_now
                if cfg.use_learned_exploration and state.fast_prev is not None:
                    out["lp"] = (ll_now - predictive_log_prob(model, state.fast_prev, decision.Z_hat, obb,
                                                              feat=feat)).float()
                base_mean = kalman_base(model, feat, obb, before)
                model._count("beliefs.update")
                # particle weights from the world-model likelihood (feat of Z_hat = WM.prior(R_t, A_t, B_t, dt))
                fast1, binfo = model.beliefs.update(state.fast, decision.Z, decision.act_tokens, obb,
                                                    model.outcome_head, feat=feat)
            else:
                ll_now = model.mixture_loglik(state.fast, decision.Z_hat, obb)
                out["loglik"] = ll_now
                if cfg.use_learned_exploration and state.fast_prev is not None:
                    # LP_{t-1} = log p(o_t | B_t) - log p(o_t | B_{t-1}): o_t is disjoint from the support o_{t-1}
                    out["lp"] = (ll_now - model.mixture_loglik(state.fast_prev, decision.Z_hat, obb)).float()
                # Kalman target: residual of y against the same base prediction the residual is added to
                base_mean, _, feat = model.base_moments(decision.Z_hat, before, obb)
                model._count("beliefs.update")
                fast1, binfo = model.beliefs.update(state.fast, decision.Z, decision.act_tokens, obb,
                                                    model.outcome_head)
            check_shape(fast1.logw, (B, cfg.n_particles), "beliefs.update logw")
            model._count("bayes.update")
            fast2 = model.kalman_update(fast1, feat, obb, base_mean) if model._kalman_on() else fast1
            if dom:
                bc = belief_change(before, fast2, binfo)
                out["belief_change"] = {k: v.detach().float() for k, v in bc.items()}
            state.fast_prev = before
            state.fast = fast_detach(fast2)
            out["belief_info"] = {k: v for k, v in (binfo or {}).items() if k not in ("feat", "logw_post")}
            for b in range(B):
                table = state.fast.channels[b]
                ch_stats[b]["unbound"] = sum(1 for s in specs[b] if s.observed and int(s.lik) in CONTINUOUS
                                             and (s.key != "reward" or model.reward_in_beliefs)
                                             and s.key not in table)
    for st in ch_stats:
        model._count("channels.evicted", st["evicted"])
        model._count("channels.unbound", st["unbound"])
    out["belief_after"] = belief_summary(None if state.fast is None else state.fast.logw) or [{} for _ in range(B)]
    return out


def bayes_prior(model: "CamUnifiedModel") -> tuple:
    """(prior variance [F] or scalar, noise) a freed slot is reset to: the Bayes layer's learned values when it
    exposes them, the config initial values otherwise."""
    pv, nv = model.cfg.bayes_prior_var_init, model.cfg.bayes_obs_var_init
    if callable(getattr(model.bayes, "prior_var", None)):
        pv = model.bayes.prior_var().detach().double()
    if callable(getattr(model.bayes, "noise", None)):
        nv = float(model.bayes.noise().detach())
    return pv, nv


# ============================================================================ support / query adaptation

def adapt_loss(model: "CamUnifiedModel", seq, filt: "FilterResult", k: Optional[int] = None) -> tuple:
    """L_adapt. DOM path (`dom_beliefs`) or ``k`` None: `adapt_loss_all_k` with its defaults (every support size in
    one pass, separation-correct; ``k`` is then ignored). Otherwise the legacy single-k objective
    (`adapt_loss_single_k`)."""
    if dom_beliefs(model) or k is None:
        return adapt_loss_all_k(model, seq, filt)
    return adapt_loss_single_k(model, seq, filt, k)


def _outcome_rows(steps, t_idx: Sequence[int], b_idx: Sequence[int]) -> OutcomeBatch:
    """Rows (t_idx[i], b_idx[i]) of the per-step OutcomeBatches (queries right-padded to the longest), keys kept."""
    from ..world._util import gather_tokens, outcome_pad, stack_tokens
    Qm = max(s.outcomes.q.N for s in steps)
    outs = [outcome_pad(s.outcomes, Qm) for s in steps]
    dev = outs[0].lik.device
    ti = torch.as_tensor(list(t_idx), dtype=torch.long, device=dev)
    bi = torch.as_tensor(list(b_idx), dtype=torch.long, device=dev)
    q = gather_tokens(stack_tokens([o.q for o in outs]), ti, bi)
    st = lambda name, dt: torch.stack([getattr(o, name).to(dt) for o in outs])[ti, bi]
    keys = None
    if all(o.keys is not None for o in outs):
        keys = [list(outs[t].keys[b]) for t, b in zip(t_idx, b_idx)]
    return OutcomeBatch(q, st("lik", torch.long), st("n_classes", torch.long), st("y", torch.float32),
                        st("censored", torch.bool), st("y_mask", torch.bool), st("channel", torch.long), keys)


def adapt_loss_all_k(model: "CamUnifiedModel", seq, filt: "FilterResult", ks: Optional[Sequence[int]] = None,
                     max_query: int = 8, kalman_dropout: float = 0.5, mix_aux: float = 1.0,
                     generator: Optional[torch.Generator] = None, support_feat: str = "prior",
                     use_kalman: Optional[bool] = None) -> tuple:
    """Separation-correct support/query objective over every support size in one pass (spec 12.2, plan §2 L_adapt):

        L = mean_k [ -sum_{episodes, j in query_k} log p(o_j | B_k, a_{start+k..j}) / #outcomes ]
            + mix_aux * the same with the particle-mixture predictive only.

    Episodes are delimited by ``done`` (and the window start); position pos(t) counts decisions since the episode
    start. One support filter runs over the window (a fresh belief at every episode start): B_k is the belief after
    absorbing the episode's first k decisions (particles with the world-model likelihood, ``support_feat="prior"``,
    or the belief module's own action conditioning, "module"; plus the Kalman/NIG layer). For every episode longer
    than k a query job scores the next ``max_query`` decisions of the same episode from the support-end state: the
    prior input at decision start+k (R_t on the DOM path, else Z_t; it has seen O_{<=start+k}, i.e. support
    consequences only) is rolled open loop through the recorded query actions with the world-model prior (mode
    latent; belief tokens of B_k; reasoner on the imagined state at later steps), and outcome j is scored from the
    predicted state after its own action under B_k (`FusedPredictive.joint_log_prob`). Query outcomes never enter
    B_k or any query state. Gradients reach the belief module, the Bayes layer and its fusion gate, the outcome head,
    the world-model prior and the core / reasoner through the support-end state only.
    ``kalman_dropout``: probability that the whole call runs without the Kalman layer (support and scoring), so the
    particle path is trained on continuous outcomes too (plan §1 Beliefs interim). Returns (loss, metrics) with
    per-k NLLs (full and mixture), query counts, skipped support sizes and whether the Kalman layer was used."""
    from ..learn.meta import SUPPORT_SIZES
    from ..model import fast_cat, fast_where
    from ..world._util import gather_tokens, stack_tokens
    from ..world.train_seq import prior_on_R, reason_fn_of
    steps = seq.steps
    T = len(steps)
    B = steps[0].obs.B
    dev = filt.Zs[0].device
    ks = tuple(sorted(set(int(k) for k in (SUPPORT_SIZES if ks is None else ks))))
    kal_avail = model._kalman_on()
    if use_kalman is None:
        u = float(torch.rand((), generator=generator, device=generator.device)) if generator is not None else \
            float(torch.rand(()))
        use_k = kal_avail and not (kalman_dropout > 0 and u < kalman_dropout)
    else:
        use_k = bool(use_kalman) and kal_avail
    Rs = getattr(filt, "Rs", None)
    use_R = prior_on_R(model) and Rs is not None and len(Rs) == T and all(r is not None for r in Rs)
    Zin = [(Rs[t] if use_R else filt.Zs[t]).float() for t in range(T)]
    reason = reason_fn_of(model) if use_R else None
    # ---- episode structure
    padded = torch.stack([padded_rows(s).to(dev) for s in steps], 1)                 # [B,T]
    valid = ~padded
    done = torch.stack([torch.as_tensor(s.done, device=dev).bool().reshape(B) for s in steps], 1) & valid
    start = torch.ones(B, T, dtype=torch.bool, device=dev)
    start[:, 1:] = done[:, :-1]
    ep = torch.zeros(B, T, dtype=torch.long, device=dev)
    if T > 1:
        ep[:, 1:] = torch.cumsum(done[:, :-1].long(), 1)
    pos = torch.zeros(B, T, dtype=torch.long, device=dev)
    for t in range(1, T):
        pos[:, t] = torch.where(start[:, t], torch.zeros_like(pos[:, t]), pos[:, t - 1] + 1)
    ep_len = torch.zeros(B, T + 1, dtype=torch.long, device=dev).scatter_add_(1, ep, valid.long()).gather(1, ep)
    k_max = max(ks)
    # ---- one support filter over the window; snapshot B_k at the decision where pos == k
    jobs = []                       # (k, t, rows, FastState snapshot of those rows)
    fast = model.beliefs.init_fast(B, device=dev)
    fast.channels = [{} for _ in range(B)]
    with model._autocast():
        for t in range(T):
            if t > 0 and bool(start[:, t].any()):
                fresh = model.beliefs.init_fast(B, device=dev)
                fresh.channels = [{} for _ in range(B)]
                fast = fast_where(start[:, t], fresh, fast)
            for k in ks:
                rows = (valid[:, t] & (pos[:, t] == k) & (ep_len[:, t] > k)).nonzero().flatten()
                if rows.numel():
                    jobs.append((k, t, rows, fast.select(rows)))
            sup = valid[:, t] & (pos[:, t] < k_max)
            st = steps[t]
            ob = model.belief_view(st.outcomes)
            sup = sup & ob.y_mask.any(-1)
            if not bool(sup.any()) or t == T - 1:
                continue
            Bt = model.beliefs.tokens(fast)
            ob = model.channels_of(fast, ob)
            Z_hat, _ = model.world_model.prior(Zin[t], st.act, Bt, st.dt, sample=False)
            feat = model.outcome_head.features(Z_hat, ob)
            if support_feat == "prior":
                fast_n, _ = model.beliefs.update(fast, filt.Zs[t], st.act, ob, model.outcome_head, feat=feat)
            else:
                fast_n, _ = model.beliefs.update(fast, filt.Zs[t], st.act, ob, model.outcome_head)
            if use_k:
                fast_n = model.kalman_update(fast_n, feat, ob, kalman_base(model, feat, ob, fast))
            fast = fast_n if bool(sup.all()) else fast_where(sup, fast_n, fast)
        # ---- batched query rollouts of every (k, episode) job
        tot = {k: Zin[0].new_zeros(()) for k in ks}
        tot_mix = {k: Zin[0].new_zeros(()) for k in ks}
        cnt = {k: 0.0 for k in ks}
        if jobs:
            jk = torch.cat([torch.full((len(j[2]),), j[0], dtype=torch.long, device=dev) for j in jobs])
            jt = torch.cat([torch.full((len(j[2]),), j[1], dtype=torch.long, device=dev) for j in jobs])
            jb = torch.cat([j[2] for j in jobs])
            jf = fast_cat([j[3] for j in jobs])
            J = jb.numel()
            Zst = torch.stack(Zin)                                                    # [T,B,S,D]
            Z = Zst[jt, jb]
            btok = model.beliefs.tokens(jf)
            acts = stack_tokens([s.act for s in steps])
            goals = None
            if all(s.goal is not None for s in steps):
                goals = stack_tokens([s.goal for s in steps])
            g_rows = None if goals is None else gather_tokens(goals, jt, jb)
            dts = torch.stack([torch.as_tensor(s.dt, device=dev).float().reshape(B) for s in steps])    # [T,B]
            alive = torch.ones(J, dtype=torch.bool, device=dev)
            ep_j = ep[jb, jt]
            for i in range(max_query):
                ti = jt + i
                alive = alive & (ti < T)
                if not bool(alive.any()):
                    break
                tc = ti.clamp(max=T - 1)
                alive = alive & valid[jb, tc] & (ep[jb, tc] == ep_j)
                if not bool(alive.any()):
                    break
                if i > 0 and reason is not None:
                    Z = reason(Z, btok, g_rows)
                a_i = gather_tokens(acts, tc, jb)
                Z_hat, _ = model.world_model.prior(Z, a_i, btok, dts[tc, jb], sample=False)
                ob = _outcome_rows(steps, tc.tolist(), jb.tolist())
                ob = OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y, ob.censored, ob.y_mask & alive.unsqueeze(-1),
                                  ob.channel, ob.keys)
                n_o = ob.y_mask.float().sum(-1)
                if bool((n_o > 0).any()):
                    feat = model.outcome_head.features(Z_hat, ob)
                    lp = fused_predictive(model, jf, feat, ob, kalman=use_k).joint_log_prob(ob)
                    lpm = fused_predictive(model, jf, feat, ob, kalman=False).joint_log_prob(ob) \
                        if (mix_aux > 0 and use_k) else lp
                    for k in ks:
                        m = (jk == k) & (n_o > 0)
                        if bool(m.any()):
                            tot[k] = tot[k] + torch.where(m, lp, torch.zeros_like(lp)).sum()
                            tot_mix[k] = tot_mix[k] + torch.where(m, lpm, torch.zeros_like(lpm)).sum()
                            cnt[k] += float(n_o[m].sum())
                Z = Z_hat
    used = [k for k in ks if cnt[k] > 0]
    skipped = [k for k in ks if cnt[k] == 0]
    z0 = Zin[0].sum() * 0
    if not used:
        return z0, {"adapt_nll": float("nan"), "adapt_ks": [], "adapt_skipped_k": skipped, "adapt_kalman": use_k,
                    "adapt_queries": 0}
    nll = {k: -tot[k] / cnt[k] for k in used}
    nll_mix = {k: -tot_mix[k] / cnt[k] for k in used}
    main = torch.stack([nll[k] for k in used]).mean()
    aux = torch.stack([nll_mix[k] for k in used]).mean()
    loss = main + (mix_aux * aux if mix_aux > 0 else 0.0)
    mets = {"adapt_nll": float(main.detach()), "adapt_nll_mix": float(aux.detach()), "adapt_ks": used,
            "adapt_skipped_k": skipped, "adapt_kalman": bool(use_k), "adapt_jobs": len(jobs), "adapt_T": T,
            "adapt_queries": int(sum(cnt.values()))}
    for k in used:
        mets[f"adapt_nll_k{k}"] = float(nll[k].detach())
        mets[f"adapt_nll_mix_k{k}"] = float(nll_mix[k].detach())
        mets[f"adapt_n_query_k{k}"] = int(cnt[k])
    return loss, mets


def adapt_loss_single_k(model: "CamUnifiedModel", seq, filt: "FilterResult", k: int) -> tuple:
    """L_adapt = -log p(Q | B_S, query actions) (spec 12.2). The support belief B_S is inferred from the first k
    steps by the belief module (particles) *and* the Kalman/NIG last layer (residuals of the observed continuous
    outcomes against the particle-mixture head mean, differentiable closed form); the later steps' observed
    outcomes (queries, disjoint from the support) are then scored under B_S without further updates: outcomes on
    a bound Kalman channel by the combined head + Kalman predictive (`kalman_log_prob`), all others by the
    particle mixture. Gradients reach the core (via Z), outcome head, world-model prior, belief module, the
    Kalman feature map Phi_theta and the Bayes layer's learned prior / noise. Returns (loss, metrics)."""
    steps = seq.steps
    T = len(steps)
    B = steps[0].obs.B
    Zs = filt.Zs
    dev = Zs[0].device
    fast = model.beliefs.init_fast(B, device=dev)
    fast.channels = [{} for _ in range(B)]
    kal = model._kalman_on()
    n_kal = n_mix = 0
    with model._autocast():
        for t in range(k):
            st = steps[t]
            ob = model.belief_view(st.outcomes)
            if not bool(ob.y_mask.any()):
                continue
            Bt = model.beliefs.tokens(fast)
            Z_hat, _ = model.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
            ob = model.channels_of(fast, ob)
            base_mean, _, feat = model.base_moments(Z_hat, fast, ob)
            fast_n, _ = model.beliefs.update(fast, Zs[t], st.act, ob, model.outcome_head)
            if kal:
                fast_n = model.kalman_update(fast_n, feat, ob, base_mean)
            fast = fast_n
        Bt = model.beliefs.tokens(fast)
        nll, n = Zs[0].new_zeros(()).float(), 0
        for t in range(k, T):
            st = steps[t]
            valid = st.outcomes.y_mask.any(-1) & ~model._padded(st)
            if not bool(valid.any()):
                continue
            Z_hat, _ = model.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
            ob = model.channels_of(fast, st.outcomes)
            base_mean, _, feat = model.base_moments(Z_hat, fast, ob)
            kb = model.kalman_bound(ob) & ob.y_mask if kal else torch.zeros_like(ob.y_mask)
            ll = model.mixture_ll(fast, feat, ob, ob.y_mask & ~kb)
            if bool(kb.any()):
                phi = model.bayes.features(feat, ob)
                lpk = model.kalman_log_prob(fast, phi, ob, base_mean)
                ll = ll + torch.where(kb, lpk, torch.zeros_like(lpk)).sum(-1)
            n_kal += int(kb.sum())
            n_mix += int((ob.y_mask & ~kb).sum())
            ll = ll / ob.y_mask.float().sum(-1).clamp_min(1.0)
            nll = nll - (ll * valid.float()).sum() / valid.float().sum()
            n += 1
    loss = nll / max(1, n)
    return loss, {"adapt_k": k, "adapt_T": T, "adapt_queries": n, "adapt_nll": float(loss.detach()),
                  "adapt_kalman_outcomes": n_kal, "adapt_mixture_outcomes": n_mix}


def padded_rows(st) -> torch.Tensor:
    """[B] bool: rows of a replay StepBatch that are padding (no observation and no action token)."""
    return ~st.obs.mask.any(-1) & ~st.act.mask.any(-1)
