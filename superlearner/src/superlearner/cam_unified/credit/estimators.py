"""Policy-gradient estimators for tabular softmax policies, written as surrogate losses whose autograd gradient is
the estimator (meta-reward spec 7.4, 7.8, Exp C). Each `*_grad` returns the per-batch mean estimate [S, A] and the
per-trajectory estimates [N, S, A] (for bias / variance / SNR).

  reinforce   sum_t grad log pi(A_t|S_t) (G_t - b(S_t))                        (Monte Carlo, optional baseline)
  all_actions sum_t sum_a grad pi(a|S_t) Q_t(S_t, a)                          (exact-critic reference)
  cocoa       sum_t [grad log pi(A_t|S_t) R_t + sum_a grad pi(a|S_t) sum_{k>=1} w_t(S_t,a,U_{t+k}) R_{t+k}]
                                                                             (Meulemans et al. 2023, eq. 4)
  hca_return  sum_t sum_a grad pi(a|S_t) (h(a|S_t,G_t)/pi(a|S_t)) G_t ... see `hca_return_grad`
  rudder      sum_t grad log pi(A_t|S_t) (sum_{k>=t} rr_k - b) with rr the return-equivalent redistributed reward

`w`, `h`, `Q` may be exact (from `credit.exact`) or learned; the comparison of learned vs exact coefficients is the
bias measurement the spec asks for (7.5, 7.8).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .exact import Batch, TabularMDP, policy


def _per_traj_grads(theta: torch.Tensor, surrogate: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """surrogate(theta) -> [N] per-trajectory surrogate values; returns [N, S, A] gradients (functional jacobian)."""
    th = theta.detach().clone().requires_grad_(True)
    vals = surrogate(th)
    N = vals.shape[0]
    grads = []
    for i in range(N):
        (g,) = torch.autograd.grad(vals[i], th, retain_graph=i < N - 1)
        grads.append(g)
    return torch.stack(grads)


def _per_traj_grads_fast(theta: torch.Tensor, coef_logp: torch.Tensor, coef_pi: Optional[torch.Tensor],
                         batch: Batch) -> torch.Tensor:
    """Closed-form per-trajectory gradients for surrogate
        sum_t coef_logp[n,t] * log pi(A_t|S_t) + sum_t sum_a coef_pi[n,t,a] * pi(a|S_t)
    with tabular softmax: d log pi(a|s)/d theta[s,:] = e_a - pi(.|s); d pi(a|s)/d theta[s,:] = pi(a|s)(e_a - pi(.|s))."""
    pi = policy(theta).detach()
    N, T = batch.s.shape
    S, A = pi.shape
    G = torch.zeros(N, S, A, dtype=pi.dtype)
    eye = torch.eye(A, dtype=pi.dtype)
    p_s = pi[batch.s]                                                   # [N, T, A]
    glogp = eye[batch.a] - p_s                                          # [N, T, A]
    contrib = coef_logp[..., None] * glogp
    if coef_pi is not None:
        # sum_a c_a pi_a (e_a - pi) = (c * pi) - pi * sum_a c_a pi_a
        cp = coef_pi * p_s
        contrib = contrib + cp - p_s * cp.sum(-1, keepdim=True)
    G.index_put_((torch.arange(N)[:, None].expand(N, T), batch.s), contrib, accumulate=True)
    return G


def reinforce_grad(theta: torch.Tensor, batch: Batch, baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """baseline: optional V[t, s] (exact or learned)."""
    G = batch.returns_to_go()
    if baseline is not None:
        G = G - baseline[torch.arange(batch.s.shape[1])[None, :], batch.s]
    return _per_traj_grads_fast(theta, G, None, batch)


def all_actions_grad(theta: torch.Tensor, batch: Batch, Q: torch.Tensor) -> torch.Tensor:
    T = batch.s.shape[1]
    q = Q[torch.arange(T)[None, :], batch.s]                            # [N, T, A]
    return _per_traj_grads_fast(theta, torch.zeros_like(batch.r), q, batch)


def cocoa_grad(theta: torch.Tensor, batch: Batch, w: torch.Tensor) -> torch.Tensor:
    """w: [T, S, A, U] contribution coefficients (exact or learned).

    Immediate reward through the score function; future rewards credited to every alternative action a through
    w_t(S_t, a, U_{t+k}) R_{t+k} (k >= 1)."""
    N, T = batch.s.shape
    tt = torch.arange(T)
    # W[n, t, k, a] = w[t, S_t, a, U_k]
    wt = w[tt[None, :], batch.s]                                        # [N, T, A, U]
    Wk = torch.gather(wt, 3, batch.u[:, None, None, :].expand(N, T, wt.shape[2], T))   # [N, T, A, K]
    future = (torch.arange(T)[None, :] > torch.arange(T)[:, None]).to(batch.r.dtype)    # [T(t), T(k)]  k > t
    coef_pi = torch.einsum("ntak,tk,nk->nta", Wk, future, batch.r)
    return _per_traj_grads_fast(theta, batch.r, coef_pi, batch)


def hca_return_grad(theta: torch.Tensor, batch: Batch, h_ret: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]
                    ) -> torch.Tensor:
    """Return-conditioned HCA (Harutyunyan et al. 2019): sum_t sum_a grad pi(a|S_t) (h(a|S_t, Z_t)/pi(a|S_t)) Z_t with
    Z_t the realised return-to-go. Equivalent all-actions form: coefficient on pi(a|S_t) is h/pi * Z.
    h_ret(t, s [N], z [N]) -> [N, A] hindsight probabilities."""
    N, T = batch.s.shape
    pi = policy(theta).detach()
    Z = batch.returns_to_go()
    coef = torch.zeros(N, T, pi.shape[1], dtype=pi.dtype)
    for t in range(T):
        h = h_ret(t, batch.s[:, t], Z[:, t])
        coef[:, t] = h / pi[batch.s[:, t]] * Z[:, t:t + 1]
    return _per_traj_grads_fast(theta, torch.zeros_like(batch.r), coef, batch)


def rudder_grad(theta: torch.Tensor, batch: Batch, redistributed: torch.Tensor,
                baseline: Optional[torch.Tensor] = None, immediate_only: bool = False) -> torch.Tensor:
    """REINFORCE on return-equivalent redistributed rewards rr [N, T] (sum_t rr = sum_t r per trajectory).

    immediate_only: use rr_t alone as the credit of A_t (RUDDER's goal: with an optimal redistribution the expected
    future redistributed reward is zero, so the immediate redistributed reward carries the whole credit)."""
    G = redistributed if immediate_only else torch.flip(torch.cumsum(torch.flip(redistributed, [1]), 1), [1])
    if baseline is not None:
        G = G - baseline[torch.arange(batch.s.shape[1])[None, :], batch.s]
    return _per_traj_grads_fast(theta, G, None, batch)


def rudder_exact_redistribution(mdp: TabularMDP, theta: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Optimal return-equivalent redistribution for a Markov state: rr_t = Q_t(S_t,A_t) - V_t(S_t) + [V offsets] such
    that sum_t rr_t = G. Uses g(H_t) = E[G | H_t] which, with the state carrying all reward-relevant history and
    the accumulated reward, is sum_{k<t} r_k + Q_t(S_t, A_t). The difference g(H_{t}) - g(H_{t-1}) plus the final
    residual G - g(H_{T-1}) gives a decomposition whose per-trajectory sum equals the realised return exactly."""
    from .exact import q_values
    Q, V = q_values(mdp, theta)
    N, T = batch.s.shape
    tt = torch.arange(T)
    q_sa = Q[tt[None, :], batch.s, batch.a]                            # [N, T]
    past = torch.cumsum(batch.r, 1) - batch.r                          # sum_{k<t} r_k
    g = past + q_sa                                                    # E[G | H_t, A_t]
    g0 = V[0][batch.s[:, 0]]                                           # E[G] at the initial state (offset)
    rr = torch.empty_like(g)
    rr[:, 0] = g[:, 0] - g0
    rr[:, 1:] = g[:, 1:] - g[:, :-1]
    rr[:, -1] = rr[:, -1] + (batch.r.sum(1) - g[:, -1])                # residual correction (noise of the last step)
    rr[:, 0] = rr[:, 0] + g0                                           # keep the offset so sum rr = G
    return rr


# ============================================================================================ statistics

def estimator_stats(per_traj: torch.Tensor, exact: torch.Tensor) -> dict:
    """bias (norm of mean - exact), variance (trace of covariance per sample), SNR, cosine to exact."""
    N = per_traj.shape[0]
    flat = per_traj.reshape(N, -1)
    mean = flat.mean(0)
    ex = exact.reshape(-1)
    var = flat.var(0, unbiased=True).sum()
    bias = (mean - ex).norm()
    se = (var / N).sqrt()                      # standard error of the mean estimate (norm scale)
    cos = torch.nn.functional.cosine_similarity(mean, ex, dim=0)
    return {"bias_norm": float(bias), "var_trace": float(var), "se_of_mean": float(se),
            "snr": float(ex.norm() ** 2 / var.clamp_min(1e-30)), "cos_to_exact": float(cos), "n": N,
            "exact_norm": float(ex.norm())}
