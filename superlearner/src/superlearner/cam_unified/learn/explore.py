"""Learned exploration (spec Part 8): learning-progress rewards that separate information from noise.

The exploration *policy* is the ordinary policy (meta-RL): the same action generator, conditioned on the reasoned
state, the belief tokens and the BUDGET token, chooses experiments; this module supplies what it is trained on.

1. **Learning progress in hindsight** (spec 8.3). For decision t with observed outcome o_t (the *support*), and the
   later real observations o_{t+1..t+H} of the same episode (the *queries*, disjoint from the support):

       LP_t = sum_{j=t+1}^{t+H} [ log p(o_j | B_t^+) - log p(o_j | B_t^-) ],   B_t^+ = Update(B_t^-, o_t)

   B_t^- is the belief before o_t. Queries are scored, never used to update; an observation is never used both to
   update the learner and to claim improved prediction.

2. **Epistemic / aleatoric separation** (spec 8.1). The particle belief gives per-particle predictive moments of the
   support outcome. Law of total variance over particles with weights w_k:

       total = sum_k w_k var_k + sum_k w_k (mu_k - mu_bar)^2 = aleatoric + epistemic

   The *signal fraction* rho = epistemic / total (the Kalman-gain-like share of the innovation that is attributable to
   the unknown mechanism) multiplies LP: an outcome on which all hypotheses agree (pure noise, however large) earns
   exactly nothing even if a learned update reacts to it, while an outcome that discriminates hypotheses keeps its
   measured LP. For discrete outcomes the same split uses entropies (mutual information / predictive entropy).
   The ex-ante expected information gain 0.5 log(1 + epistemic / aleatoric) is also returned (optional policy input /
   baseline score).

3. `surprise_reward` (raw prediction error, -log p(o | B^-)) is provided only as the contrast baseline the spec
   forbids using as an exploration reward; tests show it prefers a noisy TV.

Long-term credit of an experiment comes from putting ``lp_coef * LP_corr`` into the return stream
(`Trajectory.intrinsic`, see `meta.MetaRLObjective`), i.e. return-based credit assignment through the same policy.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor


# ----------------------------------------------------------------------------------------------- belief predictives

def mixture_log_prob(particle_logp: Tensor, logw: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Predictive log-likelihood of a step's observed outcomes under a particle belief.

    particle_logp [B,K,Q]: log p(y_q | particle k); logw [B,K] normalised log-weights; mask [B,Q] observed outcomes.
    Returns [B] = log sum_k w_k prod_q p(y_q | k) (outcomes of one step are jointly scored per particle)."""
    lp = particle_logp.float()
    if mask is not None:
        lp = lp * mask.unsqueeze(1).to(lp.dtype)
    return torch.logsumexp(logw.float() + lp.sum(-1), dim=1)


def gaussian_decomposition(mean: Tensor, var: Tensor, logw: Tensor, eps: float = 1e-12) -> dict:
    """Moment decomposition of a particle mixture predictive.

    mean, var [B,K,Q] per-particle predictive moments; logw [B,K]. Returns dict of [B,Q] tensors: ``total``,
    ``aleatoric`` (E_k var_k), ``epistemic`` (Var_k mu_k), ``signal_frac`` = epistemic / total,
    ``info_gain`` = 0.5 log(1 + epistemic / aleatoric) (nats; Gaussian expected information about the mechanism)."""
    w = logw.float().softmax(-1).unsqueeze(-1)                       # [B,K,1]
    mean, var = mean.float(), var.float().clamp_min(0)
    mu_bar = (w * mean).sum(1)
    alea = (w * var).sum(1)
    epi = (w * (mean - mu_bar.unsqueeze(1)) ** 2).sum(1)
    total = alea + epi
    return {"total": total, "aleatoric": alea, "epistemic": epi,
            "signal_frac": epi / total.clamp_min(eps),
            "info_gain": 0.5 * torch.log1p(epi / alea.clamp_min(eps))}


def categorical_decomposition(probs: Tensor, logw: Tensor, eps: float = 1e-12) -> dict:
    """Entropy decomposition (BALD) of a particle mixture over C discrete outcomes.

    probs [B,K,Q,C]; logw [B,K]. Returns [B,Q] tensors ``total`` H[sum_k w_k p_k], ``aleatoric`` sum_k w_k H[p_k],
    ``epistemic`` (mutual information, nats), ``signal_frac`` = MI / total, ``info_gain`` = MI."""
    w = logw.float().softmax(-1)[:, :, None, None]
    p = probs.float().clamp_min(0)
    p = p / p.sum(-1, keepdim=True).clamp_min(eps)
    pbar = (w * p).sum(1)
    H = lambda q: -(q * torch.log(q.clamp_min(eps))).sum(-1)          # noqa: E731
    total = H(pbar)
    alea = (w[..., 0] * H(p)).sum(1)
    mi = (total - alea).clamp_min(0)
    return {"total": total, "aleatoric": alea, "epistemic": mi, "signal_frac": mi / total.clamp_min(eps),
            "info_gain": mi}


def decompose_outcome_dist(dist, logw: Tensor) -> dict:
    """Moment decomposition for any particle-conditioned `OutcomeDist` with batch shape [B,K,Q] (contract)."""
    mean = dist.mean() if callable(getattr(dist, "mean", None)) else dist.mean
    var = dist.variance() if callable(getattr(dist, "variance", None)) else dist.variance
    return gaussian_decomposition(mean, var, logw)


def combine_signal_frac(rho: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Per-step signal fraction from per-outcome fractions rho [B,Q]: 1 - prod_q (1 - rho_q) over observed outcomes.
    0 iff every observed outcome is pure noise; 1 if any outcome is fully attributable to the mechanism."""
    r = rho.float().clamp(0, 1)
    if mask is not None:
        r = r * mask.to(r.dtype)
    return 1.0 - torch.prod(1.0 - r, dim=-1)


# ----------------------------------------------------------------------------------------------- learning progress

def hindsight_learning_progress(beliefs: Sequence[Any], steps: Sequence[Any],
                                update_fn: Callable[[Any, Any, int], Any],
                                log_prob_fn: Callable[[Any, Any, int], Tensor], horizon: int,
                                mask: Optional[Tensor] = None, episode: Optional[Tensor] = None,
                                reduce: str = "sum", no_grad: bool = True) -> dict:
    """Hindsight LP of every decision of a trajectory.

    beliefs[t]: belief *before* step t's outcome (B_t^-, as produced by the agent's filter); steps[t]: the step record
    accepted by the callables; ``update_fn(belief, step, t) -> belief`` incorporates step t's outcome;
    ``log_prob_fn(belief, step, t) -> [B]`` is the predictive log-likelihood of step t's observed outcome.
    mask [B,T] valid steps; episode [B,T] long episode index (queries never cross an episode boundary).

    Returns {"lp": [B,T], "n_query": [B,T]}: LP_t summed (or averaged, ``reduce="mean"``) over up to ``horizon``
    later valid steps of the same episode. Support (step t) and queries (t+1..t+H) are disjoint by construction; if the
    steps carry ``event_ids`` they are checked too."""
    T = len(steps)
    if len(beliefs) != T:
        raise ValueError("need one prior belief per step")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    _check_disjoint_ids(steps)
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        lp0 = log_prob_fn(beliefs[0], steps[0], 0)
        B = lp0.shape[0]
        dev = lp0.device
        m = torch.ones(B, T, dtype=torch.bool, device=dev) if mask is None else mask.bool().to(dev)
        ep = torch.zeros(B, T, dtype=torch.long, device=dev) if episode is None else episode.long().to(dev)
        lp = torch.zeros(B, T, device=dev)
        nq = torch.zeros(B, T, device=dev)
        for t in range(T - 1):
            post = update_fn(beliefs[t], steps[t], t)
            for j in range(t + 1, min(T, t + 1 + horizon)):
                valid = m[:, t] & m[:, j] & (ep[:, j] == ep[:, t])
                if not bool(valid.any()):
                    continue
                d = log_prob_fn(post, steps[j], j).float() - log_prob_fn(beliefs[t], steps[j], j).float()
                lp[:, t] = lp[:, t] + torch.where(valid, d, torch.zeros_like(d))
                nq[:, t] = nq[:, t] + valid.float()
        if reduce == "mean":
            lp = lp / nq.clamp_min(1.0)
        elif reduce != "sum":
            raise ValueError("reduce must be 'sum' or 'mean'")
    return {"lp": lp, "n_query": nq}


def _check_disjoint_ids(steps: Sequence[Any]) -> None:
    seen: set = set()
    for s in steps:
        ids = getattr(s, "event_ids", None)
        if ids is None:
            continue
        flat = {e for e in (ids if isinstance(ids, (list, tuple)) else [ids]) if e is not None}
        if flat & seen:
            raise ValueError("an event appears in two steps: support and query would not be disjoint")
        seen |= flat


def noise_corrected_lp(lp: Tensor, signal_frac: Tensor) -> Tensor:
    """LP_corr = LP * rho, rho in [0, 1] the support outcome's epistemic share (0 for pure noise)."""
    return lp * signal_frac.float().clamp(0, 1)


def exploration_reward(lp: Tensor, signal_frac: Optional[Tensor], coef: float, clip: Optional[float] = 10.0,
                       enabled: bool = True) -> Tensor:
    """Intrinsic reward ``coef * clip(LP * rho)`` for `Trajectory.intrinsic` (0 when learned exploration is off)."""
    if not enabled:
        return torch.zeros_like(lp)
    r = lp.float() if signal_frac is None else noise_corrected_lp(lp, signal_frac)
    if clip is not None:
        r = r.clamp(-clip, clip)
    return float(coef) * r


def surprise_reward(log_prob_before: Tensor) -> Tensor:
    """Raw prediction error -log p(o_t | B_t^-). NOT a valid exploration reward (spec 8.1); contrast baseline only."""
    return -log_prob_before.float()


def lp_reward_for_trajectory(beliefs: Sequence[Any], steps: Sequence[Any], update_fn, log_prob_fn,
                             decompose_fn: Optional[Callable[[Any, Any, int], Tensor]], horizon: int, coef: float,
                             mask: Optional[Tensor] = None, episode: Optional[Tensor] = None,
                             enabled: bool = True) -> dict:
    """Convenience: hindsight LP, the per-step signal fraction ``decompose_fn(belief, step, t) -> [B]`` (rho of the
    support outcome under B_t^-), and the intrinsic reward. Returns {"lp", "signal_frac", "intrinsic", "n_query"}."""
    res = hindsight_learning_progress(beliefs, steps, update_fn, log_prob_fn, horizon, mask, episode)
    rho = None
    if decompose_fn is not None:
        with torch.no_grad():
            rho = torch.stack([decompose_fn(beliefs[t], steps[t], t).float() for t in range(len(steps))], 1)
    res["signal_frac"] = rho
    res["intrinsic"] = exploration_reward(res["lp"], rho, coef, enabled=enabled)
    return res

