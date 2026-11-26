"""Exact and sampled meta-gradients of a learned intrinsic reward on tabular MDPs (meta-reward spec 5.2-5.5, Exp J).

Setting (Zheng et al. 2020, adapted to verified outcomes): an intrinsic reward r_psi(t, s, a) (a table, or a linear
function of features) drives K inner policy-gradient updates of a freshly initialised softmax policy,

    theta_{k+1} = theta_k + alpha * grad_theta J_int(theta_k; psi),       J_int = E_pi[sum_t r_psi(t, S_t, A_t)]

and the outer objective is the verified extrinsic utility of the updated learner on fresh query experience,

    J_outer(psi) = J_ext(theta_K(psi)),          psi <- psi + beta * dJ_outer/dpsi.

Three meta-gradients are provided:

  exact_meta_gradient      autograd through K exact inner updates and the exact outer objective (ground truth)
  sampled_meta_gradient    K inner REINFORCE updates on *sampled* trajectories rewarded by r_psi (create_graph), then
                           a score-function surrogate of the verified extrinsic return on fresh trajectories sampled
                           from theta_K, differentiated back into psi: the estimator CAM uses at scale
  finite_difference        central differences of the exact J_outer (direction check)

The meta-gradient path is psi -> r_psi -> inner update -> theta_K -> verified query performance -> psi. Detaching the
intrinsic reward inside the inner update (`detach_inner=True`) removes the path and must give a zero meta-gradient.
"""
from __future__ import annotations

from typing import Optional

import torch

from .exact import DT, Batch, TabularMDP, policy, sample


def intrinsic_table(mdp: TabularMDP, psi: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
    """r_psi[t, s, a]: psi is a [T, S, A] table, or weights [F] for features [T, S, A, F]."""
    if features is None:
        return psi
    return features @ psi


def expected_intrinsic_return(mdp: TabularMDP, theta: torch.Tensor, r_int: torch.Tensor) -> torch.Tensor:
    pi = policy(theta)
    d = mdp.init.to(theta.dtype)
    J = theta.new_zeros(())
    for t in range(mdp.T):
        sa = d[:, None] * pi
        J = J + (sa * r_int[t]).sum()
        d = torch.einsum("sa,sap->p", sa, mdp.P[t])
    return J


def expected_extrinsic_return(mdp: TabularMDP, theta: torch.Tensor) -> torch.Tensor:
    from .exact import expected_return
    return expected_return(mdp, theta)


def inner_updates_exact(mdp: TabularMDP, theta0: torch.Tensor, r_int: torch.Tensor, K: int, alpha: float,
                        detach_inner: bool = False) -> torch.Tensor:
    th = theta0
    for _ in range(K):
        r = r_int.detach() if detach_inner else r_int
        J = expected_intrinsic_return(mdp, th, r)
        (g,) = torch.autograd.grad(J, th, create_graph=True)
        th = th + alpha * g
    return th


def exact_meta_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                        features: Optional[torch.Tensor] = None, detach_inner: bool = False) -> tuple[torch.Tensor, float]:
    ps = psi.detach().clone().requires_grad_(True)
    th0 = theta0.detach().clone().requires_grad_(True)
    r_int = intrinsic_table(mdp, ps, features)
    thK = inner_updates_exact(mdp, th0, r_int, K, alpha, detach_inner)
    J = expected_extrinsic_return(mdp, thK)
    (g,) = torch.autograd.grad(J, ps, allow_unused=True)
    if g is None:
        g = torch.zeros_like(ps)
    return g, float(J.detach())


def j_outer_exact(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                  features: Optional[torch.Tensor] = None) -> float:
    th0 = theta0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        thK = inner_updates_exact(mdp, th0, intrinsic_table(mdp, psi.detach(), features), K, alpha)
    return float(expected_extrinsic_return(mdp, thK.detach()))


def finite_difference(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                      direction: torch.Tensor, eps: float = 1e-5, features: Optional[torch.Tensor] = None) -> float:
    d = direction / direction.norm()
    jp = j_outer_exact(mdp, psi + eps * d, theta0, K, alpha, features)
    jm = j_outer_exact(mdp, psi - eps * d, theta0, K, alpha, features)
    return (jp - jm) / (2 * eps)


def _logp(theta: torch.Tensor, b: Batch) -> torch.Tensor:
    return torch.log_softmax(theta, -1)[b.s, b.a]          # [N, T]


def sampled_meta_gradient(mdp: TabularMDP, psi: torch.Tensor, theta0: torch.Tensor, K: int, alpha: float,
                          n_inner: int, n_query: int, generator: torch.Generator,
                          features: Optional[torch.Tensor] = None, baseline: bool = True) -> torch.Tensor:
    """Truncated sampled meta-gradient: inner REINFORCE on intrinsic rewards of sampled trajectories (the trajectory
    distribution's own dependence on theta is the score function), outer score-function surrogate of verified
    extrinsic returns of fresh query trajectories from theta_K."""
    ps = psi.detach().clone().requires_grad_(True)
    r_int = intrinsic_table(mdp, ps, features)
    th = theta0.detach().clone().requires_grad_(True)
    for _ in range(K):
        b = sample(mdp, th.detach(), n_inner, generator)
        ri = r_int[torch.arange(mdp.T)[None, :], b.s, b.a]          # [N, T] intrinsic rewards (depend on psi)
        G = torch.flip(torch.cumsum(torch.flip(ri, [1]), 1), [1])
        if baseline:
            G = G - G.mean(0, keepdim=True).detach()
        surr = (_logp(th, b) * G).sum(1).mean()
        (g,) = torch.autograd.grad(surr, th, create_graph=True)
        th = th + alpha * g
    q = sample(mdp, th.detach(), n_query, generator)
    Gx = torch.flip(torch.cumsum(torch.flip(q.r, [1]), 1), [1])     # verified extrinsic returns (no psi dependence)
    if baseline:
        Gx = Gx - Gx.mean(0, keepdim=True)
    outer = (_logp(th, q) * Gx).sum(1).mean()
    (gpsi,) = torch.autograd.grad(outer, ps)
    return gpsi


def meta_train(mdp: TabularMDP, K: int, alpha: float, beta: float, steps: int, theta_init_scale: float = 0.0,
               exact: bool = True, n_inner: int = 64, n_query: int = 256, seed: int = 0,
               features: Optional[torch.Tensor] = None, psi0: Optional[torch.Tensor] = None) -> dict:
    """Meta-train psi across lifetimes. Each lifetime starts from a fresh policy (theta0 ~ N(0, scale^2)); psi persists.
    Returns the learning curve of J_outer and the final psi."""
    g = torch.Generator().manual_seed(seed)
    psi = psi0.clone() if psi0 is not None else (torch.zeros(features.shape[-1], dtype=DT) if features is not None
                                                 else torch.zeros(mdp.T, mdp.S, mdp.A, dtype=DT))
    curve = []
    for i in range(steps):
        th0 = theta_init_scale * torch.randn(mdp.S, mdp.A, dtype=DT, generator=g)
        if exact:
            gpsi, J = exact_meta_gradient(mdp, psi, th0, K, alpha, features)
        else:
            gpsi = sampled_meta_gradient(mdp, psi, th0, K, alpha, n_inner, n_query, g, features)
            J = j_outer_exact(mdp, psi, th0, K, alpha, features)
        curve.append(J)
        psi = psi + beta * gpsi
    return {"psi": psi, "curve": curve}
