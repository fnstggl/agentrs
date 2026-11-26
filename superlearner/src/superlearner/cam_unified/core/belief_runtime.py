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

Every function takes the model first; `CamUnifiedModel` keeps one-line delegating methods of the same names (helpers
that were private keep their leading underscore there: `_particle_dist`, `_continuous`, `_kalman_on`,
`_bayes_prior`, `_padded`). Calls between these functions go through the model's methods (`model.base_moments`,
`model.kalman_update`, ...), so instance-level overrides (e.g. capabilities/imagination.py patches
`model.kalman_update`) keep applying.

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers are imported inside the functions that use them.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from ..interfaces import FastState, OutcomeBatch
from ..outcome_spec import CONTINUOUS, bind_channels, build_outcome_batch, lookup_channels, make_room, with_targets

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamState, CamUnifiedModel, Decision, FilterResult


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
    Kalman/NIG last layer (mean head + phi^T mu_c, variance = Kalman predictive variance)."""
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


# ============================================================================ outcomes -> beliefs

def observe_outcomes(model: "CamUnifiedModel", state: "CamState", decision: "Decision", results: list,
                     next_obs: list, rewards: Sequence[float]) -> dict:
    """Realise the outcomes of `decision` (OutcomeBatch with y), compute the hindsight learning progress of the
    previous belief update, then update the particle beliefs and the Kalman/NIG last layer (use_fast_adaptation).
    Runs without gradients in acting (eval) mode, whatever mode the caller left the model in.
    Kalman channels: slots are freed for the realised continuous keys by least-recently-realised eviction
    (`make_room`), then allocated by the Bayes layer (B's `assign_channels`) or by the model; evictions and keys
    that do not fit are counted. Returns {"outcomes", "specs", "belief_info", "lp" [B] or None, "belief_after",
    "loglik" [B], "channels" [B] {"evicted", "unbound"}, "dropped" [B]}."""
    from ..model import acting_mode
    with torch.no_grad(), acting_mode(model):
        return _observe_outcomes(model, state, decision, results, next_obs, rewards)


def _observe_outcomes(model, state, decision, results, next_obs, rewards) -> dict:
    from ..model import belief_summary, check_shape, fast_detach
    cfg = model.cfg
    B = state.B
    specs = [state.trackers[b].realize(decision.inputs.obs_lists[b], decision.actions[b], results[b], next_obs[b],
                                       float(rewards[b])) for b in range(B)]
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
               "channels": ch_stats, "dropped": dropped}
        if cfg.use_fast_adaptation and state.fast is not None:
            obb = model.belief_view(ob)
            ll_now = model.mixture_loglik(state.fast, decision.Z_hat, obb)
            out["loglik"] = ll_now
            if cfg.use_learned_exploration and state.fast_prev is not None:
                # LP_{t-1} = log p(o_t | B_t) - log p(o_t | B_{t-1}): o_t is disjoint from the support o_{t-1}
                out["lp"] = (ll_now - model.mixture_loglik(state.fast_prev, decision.Z_hat, obb)).float()
            before = state.fast.clone()
            # Kalman target: residual of y against the same base prediction the residual is added to
            base_mean, _, feat = model.base_moments(decision.Z_hat, before, obb)
            model._count("beliefs.update")
            fast1, binfo = model.beliefs.update(state.fast, decision.Z, decision.act_tokens, obb, model.outcome_head)
            check_shape(fast1.logw, (B, cfg.n_particles), "beliefs.update logw")
            model._count("bayes.update")
            fast2 = model.kalman_update(fast1, feat, obb, base_mean)
            state.fast_prev = before
            state.fast = fast_detach(fast2)
            out["belief_info"] = {k: v for k, v in (binfo or {}).items() if k != "feat"}
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

def adapt_loss(model: "CamUnifiedModel", seq, filt: "FilterResult", k: int) -> tuple:
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
