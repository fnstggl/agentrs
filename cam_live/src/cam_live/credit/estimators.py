"""cam_live.credit.estimators — exact finite-action reference estimators of the policy gradient on tabular MDPs
(spec 9.2, 9.3; manifest RC-02, RC-04).

Provenance: adapted from superlearner/src/superlearner/cam_unified/credit/estimators.py at commit 3968583, where each
estimator was verified against enumerated gradients; the MDP library is ``cam_live.diagnostics.tabular`` (same
commit). ``explaining_away_redistribution``, ``return_conditioned_hindsight`` and ``unbiasedness`` were added here.

Each ``*_grad`` returns per-trajectory gradient estimates [N, S, A] for a tabular softmax policy ``theta`` [S, A]:

  reinforce    sum_t grad log pi(A_t|S_t) (G_t - b(S_t))
  all_actions  sum_t sum_a grad pi(a|S_t) Q_t(S_t, a)                                   (exact-critic reference)
  cocoa        sum_t [grad log pi(A_t|S_t) R_t + sum_a grad pi(a|S_t) sum_{k>=1} w_t(S_t, a, U_{t+k}) R_{t+k}]
               with w(s, a, u) = p(a | s, u) / pi(a | s) - 1                             (Meulemans et al. 2023, Eq. 4)
  rudder       sum_t grad log pi(A_t|S_t) (sum_{k>=t} rr_k - b), rr a return-equivalent redistribution
               (Arjona-Medina et al. 2019); ``immediate_only`` uses rr_t alone, which is only unbiased for an optimal
               decomposition — the redistributed *return-to-go* is unbiased for any decomposition.
  hca_return   return-conditioned hindsight (Harutyunyan et al. 2019): biased in general, kept as the counterexample.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from ..diagnostics.tabular import Batch, TabularMDP, policy, q_values


def _per_traj_grads(theta: torch.Tensor, coef_logp: torch.Tensor, coef_pi: Optional[torch.Tensor],
                    batch: Batch) -> torch.Tensor:
    """Closed-form per-trajectory gradients of the surrogate
        sum_t coef_logp[n,t] * log pi(A_t|S_t) + sum_t sum_a coef_pi[n,t,a] * pi(a|S_t)
    for a tabular softmax: d log pi(a|s)/d theta[s,:] = e_a - pi(.|s); d pi(a|s)/d theta[s,:] = pi(a|s)(e_a - pi(.|s))."""
    pi = policy(theta).detach()
    N, T = batch.s.shape
    S, A = pi.shape
    G = torch.zeros(N, S, A, dtype=pi.dtype)
    eye = torch.eye(A, dtype=pi.dtype)
    p_s = pi[batch.s]                                                   # [N, T, A]
    contrib = coef_logp[..., None] * (eye[batch.a] - p_s)
    if coef_pi is not None:
        cp = coef_pi * p_s                                              # sum_a c_a pi_a (e_a - pi)
        contrib = contrib + cp - p_s * cp.sum(-1, keepdim=True)
    G.index_put_((torch.arange(N)[:, None].expand(N, T), batch.s), contrib, accumulate=True)
    return G


def reinforce_grad(theta: torch.Tensor, batch: Batch, baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """REINFORCE with the return-to-go; ``baseline``: optional V[t, s]."""
    G = batch.returns_to_go()
    if baseline is not None:
        G = G - baseline[torch.arange(batch.s.shape[1])[None, :], batch.s]
    return _per_traj_grads(theta, G, None, batch)


def all_actions_grad(theta: torch.Tensor, batch: Batch, Q: torch.Tensor) -> torch.Tensor:
    """All-actions estimator with a critic Q[t, s, a]."""
    T = batch.s.shape[1]
    q = Q[torch.arange(T)[None, :], batch.s]                            # [N, T, A]
    return _per_traj_grads(theta, torch.zeros_like(batch.r), q, batch)


def cocoa_grad(theta: torch.Tensor, batch: Batch, w: torch.Tensor) -> torch.Tensor:
    """COCOA with contribution coefficients w[t, s, a, u] (exact: ``tabular.contribution_coefficients``, or
    learned). The immediate reward goes through the score function; every later reward R_{t+k} is credited to every
    alternative action a through w_t(S_t, a, U_{t+k})."""
    N, T = batch.s.shape
    tt = torch.arange(T)
    wt = w[tt[None, :], batch.s]                                        # [N, T, A, U]
    Wk = torch.gather(wt, 3, batch.u[:, None, None, :].expand(N, T, wt.shape[2], T))   # [N, T, A, K]
    future = (torch.arange(T)[None, :] > torch.arange(T)[:, None]).to(batch.r.dtype)    # k > t
    coef_pi = torch.einsum("ntak,tk,nk->nta", Wk, future, batch.r)
    return _per_traj_grads(theta, batch.r, coef_pi, batch)


def hca_return_grad(theta: torch.Tensor, batch: Batch,
                    h_taken: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Return-conditioned HCA: sum_t grad log pi(A_t|S_t) (1 - pi(A_t|S_t)/h(A_t|S_t,Z_t)) Z_t. Biased in general
    (COCOA App. L). h_taken(t, s [N], a [N], z [N]) -> [N] hindsight probability of the taken action."""
    N, T = batch.s.shape
    pi = policy(theta).detach()
    Z = batch.returns_to_go()
    coef = torch.zeros(N, T, dtype=pi.dtype)
    for t in range(T):
        h = h_taken(t, batch.s[:, t], batch.a[:, t], Z[:, t]).clamp_min(1e-12)
        coef[:, t] = (1 - pi[batch.s[:, t], batch.a[:, t]] / h) * Z[:, t]
    return _per_traj_grads(theta, coef, None, batch)


def return_conditioned_hindsight(batch: Batch, probs: torch.Tensor, decimals: int = 9):
    """Exact h(a | S_t, Z_t) from an enumerated batch and its trajectory probabilities (for ``hca_return_grad``)."""
    Z = batch.returns_to_go()
    table: Dict[tuple, float] = {}
    for n in range(batch.N):
        for t in range(batch.s.shape[1]):
            base = (t, int(batch.s[n, t]), round(float(Z[n, t]), decimals))
            a = int(batch.a[n, t])
            table[base + (a,)] = table.get(base + (a,), 0.0) + float(probs[n])
            table[base] = table.get(base, 0.0) + float(probs[n])

    def h_taken(t, s, a, z):
        return torch.tensor([table[(t, int(si), round(float(zi), decimals), int(ai))] /
                             table[(t, int(si), round(float(zi), decimals))] for si, ai, zi in zip(s, a, z)],
                            dtype=batch.r.dtype)
    return h_taken


def rudder_exact_redistribution(mdp: TabularMDP, theta: torch.Tensor, batch: Batch,
                                include_offset: bool = True) -> torch.Tensor:
    """Optimal return-equivalent redistribution rr [N, T] from the exact return predictor g(H_t) = E[G | H_t, A_t] =
    sum_{k<t} r_k + Q_t(S_t, A_t): rr_t = g(H_t) - g(H_{t-1}), with the initial offset E[G | S_0] on the first step
    and the residual G - g(H_{T-1}) on the last, so that sum_t rr_t equals the realised return on every trajectory.
    ``include_offset=False`` leaves the offset out (sum_t rr_t = G - E[G | S_0]): the offset is the prediction before
    any decision, i.e. a baseline, not a contribution of the first decision."""
    Q, V = q_values(mdp, theta)
    N, T = batch.s.shape
    tt = torch.arange(T)
    q_sa = Q[tt[None, :], batch.s, batch.a]                            # [N, T]
    past = torch.cumsum(batch.r, 1) - batch.r                          # sum_{k<t} r_k
    g = past + q_sa
    g0 = V[0][batch.s[:, 0]]                                           # prediction before any decision
    rr = torch.empty_like(g)
    rr[:, 0] = g[:, 0] - g0
    rr[:, 1:] = g[:, 1:] - g[:, :-1]
    rr[:, -1] = rr[:, -1] + (batch.r.sum(1) - g[:, -1])                # residual correction
    if include_offset:
        rr[:, 0] = rr[:, 0] + g0                                       # initial offset
    return rr


def explaining_away_redistribution(batch: Batch) -> torch.Tensor:
    """A return-equivalent but useless decomposition: the predictor says nothing until the end, so the whole return
    lands on the last step. Used to show what goes wrong when redistributed rewards are used as immediate credit."""
    rr = torch.zeros_like(batch.r)
    rr[:, -1] = batch.r.sum(1)
    return rr


def rudder_grad(theta: torch.Tensor, batch: Batch, redistributed: torch.Tensor,
                baseline: Optional[torch.Tensor] = None, immediate_only: bool = False) -> torch.Tensor:
    """REINFORCE on return-equivalent redistributed rewards rr [N, T]. Default: the redistributed return-to-go
    (unbiased for any decomposition). ``immediate_only``: rr_t alone as the credit of A_t."""
    G = redistributed if immediate_only else torch.flip(torch.cumsum(torch.flip(redistributed, [1]), 1), [1])
    if baseline is not None:
        G = G - baseline[torch.arange(batch.s.shape[1])[None, :], batch.s]
    return _per_traj_grads(theta, G, None, batch)


# ============================================================================================ statistics

def estimator_stats(per_traj: torch.Tensor, exact: torch.Tensor) -> dict:
    """Monte-Carlo statistics of an estimator against the exact gradient."""
    N = per_traj.shape[0]
    flat = per_traj.reshape(N, -1)
    mean = flat.mean(0)
    ex = exact.reshape(-1)
    var = flat.var(0, unbiased=True).sum()
    return {"bias_norm": float((mean - ex).norm()), "var_trace": float(var), "se_of_mean": float((var / N).sqrt()),
            "snr": float(ex.norm() ** 2 / var.clamp_min(1e-30)), "n": N, "exact_norm": float(ex.norm()),
            "cos_to_exact": float(torch.nn.functional.cosine_similarity(mean, ex, dim=0))}


def unbiasedness(per_traj: torch.Tensor, exact: torch.Tensor) -> dict:
    """z-scores and relative error of the Monte-Carlo mean against the exact gradient. An unbiased estimator has
    |z| of order 1 on every component (cosine similarity alone cannot detect a scale bias)."""
    N = per_traj.shape[0]
    flat = per_traj.reshape(N, -1)
    mean, ex = flat.mean(0), exact.reshape(-1)
    se = flat.std(0, unbiased=True) / N ** 0.5
    live = se > 1e-14
    z = (mean[live] - ex[live]) / se[live]
    dead_err = float((mean[~live] - ex[~live]).abs().max()) if (~live).any() else 0.0
    return {"max_abs_z": float(z.abs().max()) if live.any() else 0.0, "rms_z": float((z ** 2).mean().sqrt()) if live.any() else 0.0,
            "rel_error": float((mean - ex).norm() / ex.norm().clamp_min(1e-30)), "n_components": int(live.sum()),
            "max_err_zero_variance_components": dead_err}
