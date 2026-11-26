"""cam_live.reward.exact — exact tabular reference for the lifetime meta-gradient of a learned intrinsic reward.

Setting (Zheng et al. 2020, "What Can Learned Intrinsic Rewards Capture?", adapted so that the outer objective is
the independently *verified* extrinsic return and the reward generator never sees it):

    intrinsic reward       r_psi[t, s, a]        (a table, a linear map of features, optionally tanh-bounded)
    inner update           theta_{k+1} = theta_k + alpha * grad_theta J_int(theta_k; psi)      k = 0 .. K-1
                           J_int(theta; psi) = E_{pi_theta}[ sum_t r_psi[t, S_t, A_t] ]
    outer objective        J(psi) = J_ext(theta_K(psi))            expected EXTRINSIC return of the updated learner
    meta-gradient          dJ/dpsi by automatic differentiation through the K-step unroll

Three independent routes to the same number, so that each can check the others:

    exact_meta_gradient                autograd through K exact inner updates and the exact outer objective
    finite_difference_gradient         central differences of the exact objective
    sampled_meta_gradient              K sampled REINFORCE inner updates on intrinsic rewards (create_graph), a
                                       score-function surrogate on fresh query trajectories from theta_K, plus the
                                       inner-data score term; leave-one-out baselines. This is the estimator used at
                                       scale (``cam_live.reward.meta``).

The sampled estimator is unbiased for the gradient of the objective *with sampled inner updates*,
J_n(psi) = E_{D_0..D_{K-1}}[ J_ext(theta_K(psi; D)) ], which differs from the exact-update objective by O(1/n_inner).
``enumerated_sampled_objective`` computes J_n exactly (every inner batch enumerated with its probability) so that
unbiasedness can be tested with z-scores and relative error instead of cosine similarity (cosine hid a 60 %
magnitude bias in earlier work).

Provenance: the estimator structure follows superlearner/cam_unified/credit/meta_exact.py at commit 3968583
(rewritten here: batched over repetitions, leave-one-out baselines kept in the graph, enumerated reference added).
Diagnostics only: nothing here is used as evidence of live learning.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..diagnostics.tabular import DT, TabularMDP, enumerate_batch

__all__ = [
    "reward_table", "extrinsic_reward_table", "occupancy_return", "extrinsic_return", "unroll_exact",
    "outer_objective_exact", "exact_meta_gradient", "finite_difference_gradient", "finite_difference_directional",
    "lifetime_meta_gradient", "truncation_bias", "sample_batched", "sampled_meta_gradient",
    "enumerated_sampled_objective", "enumerated_sampled_gradient", "mc_summary", "meta_train",
    "learning_curve_exact", "learning_curve_sampled", "reachable_mask", "scale_matched",
]


# ====================================================================================================== rewards
def reward_table(psi: torch.Tensor, features: Optional[torch.Tensor] = None,
                 bound: Optional[float] = None) -> torch.Tensor:
    """r_psi[..., t, s, a]. ``psi`` is a [T, S, A] table, or weights [F] for ``features`` [T, S, A, F]; a leading
    repetition dimension on ``psi`` is carried through. ``bound`` applies ``bound * tanh(r / bound)``."""
    if features is None:
        r = psi
    else:
        r = torch.einsum("tsaf,...f->...tsa", features.to(psi.dtype), psi)
    if bound is not None:
        r = bound * torch.tanh(r / bound)
    return r


def extrinsic_reward_table(mdp: TabularMDP) -> torch.Tensor:
    """E[R_ext | t, s, a] as a [T, S, A] table (the reward a learner trained on the extrinsic signal follows)."""
    return (mdp.P * mdp.reward_mean[mdp.outcome]).sum(-1)


def occupancy_return(mdp: TabularMDP, theta: torch.Tensor, r_tsa: torch.Tensor) -> torch.Tensor:
    """E_{pi_theta}[sum_t r[t, S_t, A_t]] for ``theta`` [..., S, A] and ``r_tsa`` [T, S, A] or [..., T, S, A]."""
    pi = torch.softmax(theta, -1)
    d = mdp.init.to(theta.dtype).expand(theta.shape[:-1])
    J = theta.new_zeros(theta.shape[:-2])
    for t in range(mdp.T):
        sa = d[..., None] * pi
        J = J + (sa * r_tsa[..., t, :, :]).sum((-1, -2))
        d = torch.einsum("...sa,sap->...p", sa, mdp.P[t])
    return J


def extrinsic_return(mdp: TabularMDP, theta: torch.Tensor) -> torch.Tensor:
    """Exact expected extrinsic (verified) return J_ext(theta); batched over leading dimensions of ``theta``."""
    return occupancy_return(mdp, theta, extrinsic_reward_table(mdp).to(theta.dtype))


def reachable_mask(mdp: TabularMDP) -> torch.Tensor:
    """[T, S] bool: states that some policy reaches at time t (only these rows of a reward table matter)."""
    d = mdp.init > 0
    out = []
    for t in range(mdp.T):
        out.append(d.clone())
        d = ((mdp.P[t] > 0) & d[:, None, None]).any(0).any(0)
    return torch.stack(out)


def scale_matched(mdp: TabularMDP, r: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale ``r`` so its largest reachable magnitude equals that of ``reference`` (scale-fair comparisons)."""
    m = reachable_mask(mdp)[:, :, None].expand_as(r)
    return r * (reference[m].abs().max() / r[m].abs().max().clamp_min(1e-300))


# ============================================================================================= exact inner / outer
def unroll_exact(mdp: TabularMDP, theta0: torch.Tensor, r_int: torch.Tensor, K: int, alpha: float,
                 detach_reward: bool = False) -> List[torch.Tensor]:
    """[theta_0, ..., theta_K] with exact inner policy-gradient updates on the intrinsic return (graph kept)."""
    th = theta0 if theta0.requires_grad else theta0.detach().clone().requires_grad_(True)
    r = r_int.detach() if detach_reward else r_int
    out = [th]
    with torch.enable_grad():
        for _ in range(K):
            (g,) = torch.autograd.grad(occupancy_return(mdp, th, r).sum(), th, create_graph=True)
            th = th + alpha * g
            out.append(th)
    return out


def outer_objective_exact(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float, *,
                          features: Optional[torch.Tensor] = None, bound: Optional[float] = None,
                          detach_reward: bool = False, stop_unroll: bool = False) -> torch.Tensor:
    """J(psi) = J_ext(theta_K(psi)), differentiable in ``psi``. ``detach_reward`` cuts psi out of the inner update;
    ``stop_unroll`` puts a stop-gradient on theta_K. Either must give a meta-gradient of exactly zero."""
    thK = unroll_exact(mdp, theta0, reward_table(psi, features, bound), K, alpha, detach_reward)[-1]
    return extrinsic_return(mdp, thK.detach() if stop_unroll else thK)


def exact_meta_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float, *,
                        features: Optional[torch.Tensor] = None, bound: Optional[float] = None,
                        detach_reward: bool = False, stop_unroll: bool = False) -> Tuple[torch.Tensor, float]:
    """(dJ/dpsi, J) by automatic differentiation through the K-step unroll."""
    ps = psi.detach().clone().requires_grad_(True)
    J = outer_objective_exact(mdp, ps, theta0.detach(), K, alpha, features=features, bound=bound,
                              detach_reward=detach_reward, stop_unroll=stop_unroll)
    g = torch.autograd.grad(J, ps, allow_unused=True)[0] if J.requires_grad else None
    return (torch.zeros_like(ps) if g is None else g), float(J.detach())


def _j(mdp, psi, theta0, K, alpha, features, bound) -> float:
    return float(outer_objective_exact(mdp, psi.detach(), theta0.detach(), K, alpha, features=features,
                                       bound=bound).detach())


def finite_difference_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float, *,
                               eps: float = 1e-5, features: Optional[torch.Tensor] = None,
                               bound: Optional[float] = None, coords: Optional[Sequence[int]] = None
                               ) -> torch.Tensor:
    """Central-difference dJ/dpsi, coordinate by coordinate (``coords``: flat indices; others are left at 0)."""
    flat = psi.detach().reshape(-1).clone()
    g = torch.zeros_like(flat)
    for i in (range(flat.numel()) if coords is None else coords):
        old = float(flat[i])
        flat[i] = old + eps
        jp = _j(mdp, flat.view_as(psi), theta0, K, alpha, features, bound)
        flat[i] = old - eps
        jm = _j(mdp, flat.view_as(psi), theta0, K, alpha, features, bound)
        flat[i] = old
        g[i] = (jp - jm) / (2 * eps)
    return g.view_as(psi)


def finite_difference_directional(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                                  direction: torch.Tensor, *, eps: float = 1e-5,
                                  features: Optional[torch.Tensor] = None, bound: Optional[float] = None) -> float:
    """Central-difference derivative of J along ``direction`` (normalised)."""
    d = direction / direction.norm()
    return (_j(mdp, psi + eps * d, theta0, K, alpha, features, bound)
            - _j(mdp, psi - eps * d, theta0, K, alpha, features, bound)) / (2 * eps)


# ================================================================================== lifetimes and truncation bias
def lifetime_meta_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, L: int, alpha: float, *,
                           truncate: Optional[int] = None, objective: str = "sum",
                           features: Optional[torch.Tensor] = None, bound: Optional[float] = None
                           ) -> Tuple[torch.Tensor, float]:
    """Exact meta-gradient of a lifetime of ``L`` inner updates.

    objective="sum":   J = sum_{j=1..L} J_ext(theta_j)  (lifetime return: every updated policy is evaluated)
    objective="final": J = J_ext(theta_L)
    truncate=K: the derivative of each evaluated theta_j is propagated through its last K inner updates only
    (theta_{j-K} is treated as a constant), which is what a K-step meta window computes. ``truncate=None`` is exact.
    """
    if objective not in ("sum", "final"):
        raise ValueError(objective)
    ps = psi.detach().clone().requires_grad_(True)
    r = reward_table(ps, features, bound)
    targets = range(1, L + 1) if objective == "sum" else (L,)
    if truncate is None or truncate >= L:
        ths = unroll_exact(mdp, theta0.detach(), r, L, alpha)
        J = sum(extrinsic_return(mdp, ths[j]) for j in targets)
        (g,) = torch.autograd.grad(J, ps)
        return g, float(J.detach())
    values = [t.detach() for t in unroll_exact(mdp, theta0.detach(), r.detach(), L, alpha)]
    g = torch.zeros_like(ps)
    Jtot = 0.0
    for j in targets:
        start = max(0, j - truncate)
        thj = unroll_exact(mdp, values[start].clone().requires_grad_(True), r, j - start, alpha)[-1]
        J = extrinsic_return(mdp, thj)
        g = g + torch.autograd.grad(J, ps, retain_graph=True)[0]
        Jtot += float(J.detach())
    return g, Jtot


def truncation_bias(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, L: int, alpha: float,
                    Ks: Sequence[int] = (1, 2, 4, 8), objective: str = "sum", **kw) -> Dict[int, Dict[str, float]]:
    """Exact relative error, cosine and norm ratio of the K-truncated meta-gradient against the full lifetime."""
    full, _ = lifetime_meta_gradient(mdp, psi, theta0, L, alpha, objective=objective, **kw)
    out = {}
    for K in Ks:
        g, _ = lifetime_meta_gradient(mdp, psi, theta0, L, alpha, truncate=K, objective=objective, **kw)
        out[int(K)] = {"rel_err": float((g - full).norm() / full.norm()),
                       "cos": float((g * full).sum() / (g.norm() * full.norm()).clamp_min(1e-300)),
                       "norm_ratio": float(g.norm() / full.norm())}
    return out


# ============================================================================================ sampled estimator
def sample_batched(mdp: TabularMDP, pi: torch.Tensor, n: int, generator: Optional[torch.Generator] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``n`` trajectories from each of the R policies ``pi`` [R, S, A].
    Returns (s [R, n, T] long, a [R, n, T] long, r_ext [R, n, T] float64)."""
    R = pi.shape[0]
    rep = torch.arange(R).repeat_interleave(n)
    s = torch.multinomial(mdp.init.expand(R * n, -1), 1, generator=generator).squeeze(1)
    S_, A_, R_ = [], [], []
    for t in range(mdp.T):
        a = torch.multinomial(pi[rep, s], 1, generator=generator).squeeze(1)
        s2 = torch.multinomial(mdp.P[t, s, a], 1, generator=generator).squeeze(1)
        r = mdp.reward_mean[mdp.outcome[t, s, a, s2]].clone()
        if mdp.reward_noise > 0:
            r = r + mdp.reward_noise * torch.randn(R * n, dtype=DT, generator=generator)
        S_.append(s); A_.append(a); R_.append(r)
        s = s2
    st = lambda xs: torch.stack(xs, 1).view(R, n, mdp.T)
    return st(S_), st(A_), st(R_)


def _rtg(r: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(r, [-1]), -1), [-1])


def _loo(G: torch.Tensor, dim: int) -> torch.Tensor:
    """Leave-one-out mean along ``dim`` (a self-inclusive mean scales the gradient by 1 - 1/N)."""
    n = G.shape[dim]
    return (G.sum(dim, keepdim=True) - G) / max(n - 1, 1)


def _gather_sa(x: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """x [R, S, A] (per repetition) or [R, T, S, A] (per repetition and time) at (s, a) [R, n, T] -> [R, n, T]."""
    R, n, T = s.shape
    rep = torch.arange(R)[:, None, None]
    if x.dim() == 3:
        return x[rep, s, a]
    return x[rep, torch.arange(T)[None, None, :], s, a]


def _inner_step_sampled(th: torch.Tensor, r_int: torch.Tensor, s: torch.Tensor, a: torch.Tensor, alpha: float,
                        baseline: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """One REINFORCE inner update per repetition on intrinsic returns of the batch (s, a).
    th [R, S, A]; r_int [R, T, S, A]. Returns (theta', log-probabilities [R, n, T] of the batch under th).
    The leave-one-out baseline stays in the graph: it is a function of psi and part of the real update."""
    lp = _gather_sa(torch.log_softmax(th, -1), s, a)
    G = _rtg(_gather_sa(r_int, s, a))
    if baseline:
        G = G - _loo(G, 1)
    surr = (lp * G).sum(2).mean(1).sum()
    (g,) = torch.autograd.grad(surr, th, create_graph=True)
    return th + alpha * g, lp


def sampled_meta_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                          n_inner: int, n_query: int, generator: torch.Generator, *, reps: int = 1,
                          features: Optional[torch.Tensor] = None, bound: Optional[float] = None,
                          inner_baseline: bool = True, outer_baseline: str = "loo", inner_score: bool = True,
                          score_baseline: float = 0.0, exact_query: bool = False) -> torch.Tensor:
    """``reps`` independent samples of the K-step (truncated-window) sampled meta-gradient, shape [reps, *psi.shape].

    Per repetition: for k < K draw ``n_inner`` trajectories from pi_{theta_k}, take one REINFORCE step on their
    intrinsic returns (graph kept), then draw ``n_query`` trajectories from pi_{theta_K} and differentiate

        mean_i sum_t log pi_{theta_K}(a_t|s_t) (G_ext_t - baseline)                         query surrogate
        + [ sum_{k>=1} sum_{i,t} log pi_{theta_k}(a_t|s_t) ] (J_hat_query - score_baseline)   inner-data score term

    into psi. Batches k >= 1 are drawn from policies that depend on psi; without the score term the estimator is
    biased for K >= 2. ``outer_baseline``: "loo" (unbiased), "none", or "mean" (self-inclusive; biased by the factor
    1 - 1/n_query, kept only so tests can show that z-scores and relative error detect it while cosine does not).
    ``exact_query`` replaces the query sample by the exact J_ext(theta_K) (Rao-Blackwellised outer step)."""
    if outer_baseline not in ("loo", "mean", "none"):
        raise ValueError(outer_baseline)
    R = int(reps)
    ps = psi.detach().expand(R, *psi.shape).clone().requires_grad_(True)
    r_int = reward_table(ps, features, bound)                       # [R, T, S, A]
    th = theta0.detach().expand(R, *theta0.shape).clone().requires_grad_(True)
    score = torch.zeros(R, dtype=th.dtype)
    for k in range(K):
        s, a, _ = sample_batched(mdp, torch.softmax(th.detach(), -1), n_inner, generator)
        th, lp = _inner_step_sampled(th, r_int, s, a, alpha, inner_baseline)
        if k >= 1 and inner_score:
            score = score + lp.sum((1, 2))
    if exact_query:
        outer = extrinsic_return(mdp, th)
        Jhat = outer.detach()
    else:
        s, a, rx = sample_batched(mdp, torch.softmax(th.detach(), -1), n_query, generator)
        Gx = _rtg(rx)
        if outer_baseline == "loo":
            Gx = Gx - _loo(Gx, 1)
        elif outer_baseline == "mean":
            Gx = Gx - Gx.mean(1, keepdim=True)
        outer = (_gather_sa(torch.log_softmax(th, -1), s, a) * Gx).sum(2).mean(1)
        Jhat = rx.sum(2).mean(1)
    if inner_score and K >= 2:
        outer = outer + score * (Jhat - score_baseline)
    (g,) = torch.autograd.grad(outer.sum(), ps)
    return g


# ======================================================================== exact reference for sampled inner updates
def _trajectories(mdp: TabularMDP) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All (s, a) sequences with non-zero environment probability: (s [M, T], a [M, T], log env-probability [M])."""
    b, probs = enumerate_batch(mdp, torch.zeros(mdp.S, mdp.A, dtype=DT))
    return b.s, b.a, torch.log(probs) - b.logp_b.sum(1)


def enumerated_sampled_objective(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                                 n_inner: int, *, features: Optional[torch.Tensor] = None,
                                 bound: Optional[float] = None, inner_baseline: bool = True,
                                 max_paths: int = 2_000_000) -> torch.Tensor:
    """J_n(psi) = E_{D_0..D_{K-1}}[ J_ext(theta_K(psi; D)) ] computed exactly: every inner batch (a multiset of
    ``n_inner`` trajectories) is enumerated with its probability under the policy that generated it. Differentiable
    in ``psi``; its gradient contains both the pathwise part and the score of the psi-dependent inner data."""
    S_tr, A_tr, log_env = _trajectories(mdp)
    M = S_tr.shape[0]
    combos = list(itertools.combinations_with_replacement(range(M), n_inner))
    C = len(combos)
    if C ** K > max_paths:
        raise ValueError(f"{C}^{K} inner-data paths exceed max_paths={max_paths}")
    idx = torch.tensor(combos, dtype=torch.long)                                   # [C, n]
    log_coef = torch.tensor([math.lgamma(n_inner + 1) - sum(math.lgamma(c + 1) for c in
                                                            torch.bincount(torch.tensor(cb), minlength=M).tolist())
                             for cb in combos], dtype=DT)
    s_c, a_c, env_c = S_tr[idx], A_tr[idx], log_env[idx].sum(1)                    # [C, n, T], [C, n, T], [C]
    r_int = reward_table(psi, features, bound)
    th = theta0.detach().clone().requires_grad_(True)[None]                        # [1, S, A]
    logw = torch.zeros(1, dtype=DT)
    for _ in range(K):
        B = th.shape[0]
        th_e = th.repeat_interleave(C, 0)                                          # [B*C, S, A]
        s, a = s_c.repeat(B, 1, 1), a_c.repeat(B, 1, 1)
        th, lp = _inner_step_sampled(th_e, r_int.expand(B * C, *r_int.shape), s, a, alpha, inner_baseline)
        logw = logw.repeat_interleave(C) + (log_coef + env_c).repeat(B) + lp.sum((1, 2))
    return (torch.exp(logw) * extrinsic_return(mdp, th)).sum()


def enumerated_sampled_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                                n_inner: int, **kw) -> Tuple[torch.Tensor, float]:
    ps = psi.detach().clone().requires_grad_(True)
    J = enumerated_sampled_objective(mdp, ps, theta0, K, alpha, n_inner, **kw)
    (g,) = torch.autograd.grad(J, ps)
    return g, float(J.detach())


def mc_summary(samples: torch.Tensor, exact: torch.Tensor) -> Dict[str, float]:
    """Unbiasedness statistics of Monte Carlo ``samples`` [R, ...] against ``exact``: relative error of the mean,
    cosine, the largest |z| over coordinates with sampling noise, and the largest absolute error over coordinates
    without any (those must match exactly)."""
    R = samples.shape[0]
    flat, ex = samples.reshape(R, -1), exact.reshape(-1)
    mean = flat.mean(0)
    se = flat.std(0, unbiased=True) / math.sqrt(R)
    noisy = se > 0
    z = ((mean - ex)[noisy] / se[noisy]).abs()
    return {"n": float(R),
            "rel_err": float((mean - ex).norm() / ex.norm()),
            "cos": float((mean * ex).sum() / (mean.norm() * ex.norm()).clamp_min(1e-300)),
            "max_abs_z": float(z.max()) if z.numel() else 0.0,
            "n_coords": float(noisy.sum()),
            "noise_free_max_abs_err": float((mean - ex)[~noisy].abs().max()) if (~noisy).any() else 0.0,
            "rel_se": float(se.norm() / ex.norm()),
            "norm_ratio": float(mean.norm() / ex.norm())}


# ================================================================================================ meta-training
def meta_train(mdps: Sequence[TabularMDP], K: int, alpha: float, steps: int, *, lr: float = 0.1,
               bound: Optional[float] = None, theta_init_scale: float = 0.0, seed: int = 0,
               estimator: str = "exact", n_inner: int = 64, n_query: int = 256,
               features: Optional[torch.Tensor] = None, psi0: Optional[torch.Tensor] = None) -> Dict[str, object]:
    """Meta-train psi across lifetimes. Every lifetime draws one MDP from ``mdps`` (all sharing T, S, A) and a fresh
    policy theta_0 ~ N(0, theta_init_scale^2); only psi persists. Adam ascent on the exact or sampled meta-gradient.
    Returns {"psi", "curve"} where curve[i] is the exact J_ext(theta_K) of lifetime i *before* its meta-update."""
    g = torch.Generator().manual_seed(seed)
    m0 = mdps[0]
    if psi0 is not None:
        psi = psi0.detach().clone()
    elif features is not None:
        psi = torch.zeros(features.shape[-1], dtype=DT)
    else:
        psi = torch.zeros(m0.T, m0.S, m0.A, dtype=DT)
    psi.requires_grad_(True)
    opt = torch.optim.Adam([psi], lr=lr, maximize=True)
    curve = []
    for _ in range(steps):
        mdp = mdps[int(torch.randint(len(mdps), (1,), generator=g))]
        th0 = theta_init_scale * torch.randn(mdp.S, mdp.A, dtype=DT, generator=g)
        if estimator == "exact":
            grad, J = exact_meta_gradient(mdp, psi, th0, K, alpha, features=features, bound=bound)
        elif estimator == "sampled":
            J = _j(mdp, psi, th0, K, alpha, features, bound)
            grad = sampled_meta_gradient(mdp, psi, th0, K, alpha, n_inner, n_query, g, features=features,
                                         bound=bound, score_baseline=J)[0]
        else:
            raise ValueError(estimator)
        curve.append(J)
        opt.zero_grad(set_to_none=True)
        psi.grad = grad
        opt.step()
    return {"psi": psi.detach(), "curve": curve}


def learning_curve_exact(mdp: TabularMDP, r_table: torch.Tensor, theta0: torch.Tensor, steps: int, alpha: float
                         ) -> List[float]:
    """J_ext(theta_k), k = 0..steps, of a learner taking exact policy-gradient steps on the reward ``r_table``."""
    ths = unroll_exact(mdp, theta0.detach(), r_table.detach(), steps, alpha)
    return [float(extrinsic_return(mdp, t.detach())) for t in ths]


def learning_curve_sampled(mdp: TabularMDP, theta0: torch.Tensor, steps: int, alpha: float, n_inner: int,
                           reps: int, generator: torch.Generator, r_table: Optional[torch.Tensor] = None
                           ) -> Tuple[List[float], List[float]]:
    """Mean and standard error over ``reps`` learners of the exact J_ext(theta_k), k = 0..steps, when each learner
    takes sampled REINFORCE steps (leave-one-out baseline, ``n_inner`` trajectories per step) on ``r_table``, or on
    the sampled extrinsic rewards themselves when ``r_table`` is None (the extrinsic-reward-only learner)."""
    th = theta0.detach().expand(reps, *theta0.shape).clone()
    mean, se = [], []

    def rec(t):
        J = extrinsic_return(mdp, t)
        mean.append(float(J.mean()))
        se.append(float(J.std() / math.sqrt(reps)) if reps > 1 else 0.0)
    rec(th)
    for _ in range(steps):
        s, a, rx = sample_batched(mdp, torch.softmax(th, -1), n_inner, generator)
        thg = th.clone().requires_grad_(True)
        lp = _gather_sa(torch.log_softmax(thg, -1), s, a)
        G = _rtg(rx if r_table is None else _gather_sa(r_table.detach().expand(reps, *r_table.shape), s, a))
        G = G - _loo(G, 1)
        (g,) = torch.autograd.grad((lp * G).sum(2).mean(1).sum(), thg)
        th = th + alpha * g
        rec(th)
    return mean, se
