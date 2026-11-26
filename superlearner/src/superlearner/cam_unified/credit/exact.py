"""Exact finite-horizon tabular environments for verifying credit-assignment estimators (meta-reward spec 7.4, 7.8,
Exp C, Exp J).

A `TabularMDP` has time-indexed transitions P[t, s, a, s'], a rewarding-outcome function u = outcome[t, s, a, s'] and
an immediate reward R_t = reward_mean[U_t] + noise (Gaussian, std `reward_noise`). Because every quantity can be
enumerated, the exact expected return J(theta), its exact policy gradient (autograd through dynamic programming), the
exact Q-values and the exact COCOA contribution coefficients

    w_t(s, a, u) = sum_{k>=1} P(U_{t+k}=u | S_t=s, A_t=a) / sum_{k>=1} P(U_{t+k}=u | S_t=s) - 1
                 = p_pi(A_t=a | S_t=s, U_future=u) / pi(a|s) - 1                     (Meulemans et al. 2023, eq. 5-6)

are available as ground truth. Policies are tabular softmax pi(a|s) = softmax(theta[s]). States are unique per time
step in the provided builders (the state index encodes t), so a stationary table is fully general here.

Builders: `key_to_door` (an early action enables a delayed reward behind distractor rewards; the COCOA paper's
canonical long-term credit task) and `random_tree` (a random tree MDP for unbiasedness checks).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

DT = torch.float64


@dataclass
class TabularMDP:
    P: torch.Tensor            # [T, S, A, S] transition probabilities (rows sum to 1 over s')
    outcome: torch.Tensor      # [T, S, A, S] long: rewarding outcome id in [0, n_outcomes)
    reward_mean: torch.Tensor  # [n_outcomes] immediate reward mean of each outcome
    init: torch.Tensor         # [S] initial state distribution
    reward_noise: float = 0.0  # std of Gaussian noise added to every immediate reward
    name: str = "mdp"

    @property
    def T(self) -> int:
        return self.P.shape[0]

    @property
    def S(self) -> int:
        return self.P.shape[1]

    @property
    def A(self) -> int:
        return self.P.shape[2]

    @property
    def U(self) -> int:
        return self.reward_mean.shape[0]

    def validate(self) -> None:
        assert torch.allclose(self.P.sum(-1), torch.ones_like(self.P.sum(-1)), atol=1e-9), "transition rows"
        assert abs(float(self.init.sum()) - 1.0) < 1e-9
        assert int(self.outcome.min()) >= 0 and int(self.outcome.max()) < self.U


def policy(theta: torch.Tensor) -> torch.Tensor:
    return torch.softmax(theta, dim=-1)


# ============================================================================================ exact quantities

def expected_return(mdp: TabularMDP, theta: torch.Tensor) -> torch.Tensor:
    """J(theta) by forward dynamic programming (differentiable: autograd gives the exact policy gradient)."""
    pi = policy(theta)                                        # [S, A]
    r_sas = mdp.reward_mean[mdp.outcome]                      # [T, S, A, S]
    d = mdp.init.to(theta.dtype)
    J = theta.new_zeros(())
    for t in range(mdp.T):
        sa = d[:, None] * pi                                  # [S, A]
        J = J + (sa[:, :, None] * mdp.P[t] * r_sas[t]).sum()
        d = torch.einsum("sa,sap->p", sa, mdp.P[t])
    return J


def exact_gradient(mdp: TabularMDP, theta: torch.Tensor) -> torch.Tensor:
    th = theta.detach().clone().requires_grad_(True)
    J = expected_return(mdp, th)
    (g,) = torch.autograd.grad(J, th)
    return g


def state_occupancy(mdp: TabularMDP, theta: torch.Tensor) -> torch.Tensor:
    """d[t, s] = P(S_t = s)."""
    pi = policy(theta)
    d = mdp.init.to(theta.dtype)
    out = []
    for t in range(mdp.T):
        out.append(d)
        d = torch.einsum("s,sa,sap->p", d, pi, mdp.P[t])
    return torch.stack(out)


def q_values(mdp: TabularMDP, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Q[t, s, a] (expected return from (S_t=s, A_t=a), including R_t) and V[t, s]."""
    pi = policy(theta).detach()
    r_sas = mdp.reward_mean[mdp.outcome]
    T, S, A = mdp.T, mdp.S, mdp.A
    Q = torch.zeros(T, S, A, dtype=DT)
    V = torch.zeros(T + 1, S, dtype=DT)
    for t in reversed(range(T)):
        Q[t] = (mdp.P[t] * (r_sas[t] + V[t + 1][None, None, :])).sum(-1)
        V[t] = (pi * Q[t]).sum(-1)
    return Q, V[:T]


def future_outcome_counts(mdp: TabularMDP, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D[t, s, a, u] = sum_{k>=1} P(U_{t+k}=u | S_t=s, A_t=a) and its policy average Dbar[t, s, u]."""
    pi = policy(theta).detach()
    T, S, A, U = mdp.T, mdp.S, mdp.A, mdp.U
    onehot = torch.nn.functional.one_hot(mdp.outcome, U).to(DT)       # [T, S, A, S', U]
    C_next = torch.zeros(S, U, dtype=DT)                              # C_{t+1}(s', u): occurrences at times >= t+1
    D = torch.zeros(T, S, A, U, dtype=DT)
    for t in reversed(range(T)):
        D[t] = torch.einsum("sap,pu->sau", mdp.P[t], C_next)          # future k>=1 after (s, a) at t
        now = torch.einsum("sap,sapu->sau", mdp.P[t], onehot[t])       # occurrence at time t itself
        C_next = torch.einsum("sa,sau->su", pi, now + D[t])
    Dbar = torch.einsum("sa,tsau->tsu", pi, D)
    return D, Dbar


def contribution_coefficients(mdp: TabularMDP, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact COCOA coefficients w[t, s, a, u] (0 where the outcome cannot occur in the future of s)."""
    D, Dbar = future_outcome_counts(mdp, theta)
    ok = Dbar > eps
    w = torch.where(ok[:, :, None, :], D / Dbar.clamp_min(eps)[:, :, None, :] - 1.0, torch.zeros_like(D))
    return w


def hindsight_action_probs(mdp: TabularMDP, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact hindsight distribution h[t, s, u, a] = p_pi(A_t = a | S_t = s, U_future = u) (outcome occurrence
    weighted over k >= 1), i.e. the target a learned hindsight model must approximate."""
    D, Dbar = future_outcome_counts(mdp, theta)
    pi = policy(theta).detach()
    h = pi[None, :, :, None] * D / Dbar.clamp_min(eps)[:, :, None, :]    # [T, S, A, U]
    return h.permute(0, 1, 3, 2)


# ============================================================================================ sampling

@dataclass
class Batch:
    s: torch.Tensor   # [N, T] long
    a: torch.Tensor   # [N, T] long
    u: torch.Tensor   # [N, T] long (outcome at t)
    r: torch.Tensor   # [N, T] float64 (noisy immediate reward)
    logp_b: torch.Tensor  # [N, T] behaviour log-probabilities

    @property
    def N(self) -> int:
        return self.s.shape[0]

    def returns_to_go(self) -> torch.Tensor:
        return torch.flip(torch.cumsum(torch.flip(self.r, [1]), 1), [1])


def sample(mdp: TabularMDP, theta: torch.Tensor, n: int, generator: Optional[torch.Generator] = None) -> Batch:
    pi = policy(theta).detach()
    T = mdp.T
    s = torch.multinomial(mdp.init.expand(n, -1), 1, generator=generator).squeeze(1)
    S_, A_, U_, R_, L_ = [], [], [], [], []
    for t in range(T):
        a = torch.multinomial(pi[s], 1, generator=generator).squeeze(1)
        s2 = torch.multinomial(mdp.P[t, s, a], 1, generator=generator).squeeze(1)
        u = mdp.outcome[t, s, a, s2]
        r = mdp.reward_mean[u].clone()
        if mdp.reward_noise > 0:
            r = r + mdp.reward_noise * torch.randn(n, dtype=DT, generator=generator)
        S_.append(s); A_.append(a); U_.append(u); R_.append(r); L_.append(torch.log(pi[s, a]))
        s = s2
    st = lambda xs: torch.stack(xs, 1)
    return Batch(st(S_), st(A_), st(U_), st(R_), st(L_))


# ============================================================================================ builders

def key_to_door(n_distract: int = 8, n_actions: int = 2, apple_prob: float = 0.5, apple_reward: float = 1.0,
                door_reward: float = 1.0, key_cost: float = 0.0, reward_noise: float = 0.0,
                apple_action_bias: float = 0.3) -> TabularMDP:
    """Key-to-door: at t=0 the agent picks up a key (action 0) or not; then `n_distract` distractor steps where each
    action yields an apple (reward `apple_reward`) with an action-dependent probability; at the last step the door
    pays `door_reward` iff the key was taken. States encode (t, has_key, apple-flag of the step just finished).

    Outcomes: 0 nothing, 1 key picked (reward -key_cost), 2 apple, 3 door opened.
    """
    T = n_distract + 2
    S = 4 * T + 4                       # index(t, key, flag) = 4t + 2key + flag  (+4 spare for terminal t=T)
    A = n_actions
    idx = lambda t, key, flag: 4 * t + 2 * key + flag
    P = torch.zeros(T, S, A, S, dtype=DT)
    out = torch.zeros(T, S, A, S, dtype=torch.long)
    for t in range(T):
        for key in (0, 1):
            for flag in (0, 1):
                s = idx(t, key, flag)
                for a in range(A):
                    if t == 0:
                        k2 = 1 if a == 0 else 0
                        P[t, s, a, idx(1, k2, 0)] = 1.0
                        out[t, s, a, idx(1, k2, 0)] = 1 if k2 else 0
                    elif t < T - 1:
                        p = min(max(apple_prob + apple_action_bias * (1 if a == 0 else -1) / max(1, A - 1), 0.0), 1.0)
                        P[t, s, a, idx(t + 1, key, 1)] = p
                        P[t, s, a, idx(t + 1, key, 0)] = 1 - p
                        out[t, s, a, idx(t + 1, key, 1)] = 2
                        out[t, s, a, idx(t + 1, key, 0)] = 0
                    else:
                        P[t, s, a, idx(t + 1, key, 0)] = 1.0
                        out[t, s, a, idx(t + 1, key, 0)] = 3 if key else 0
    # unreachable rows: self-loop to keep rows stochastic
    rowsum = P.sum(-1)
    for t in range(T):
        for s in range(S):
            for a in range(A):
                if rowsum[t, s, a] == 0:
                    P[t, s, a, s] = 1.0
    init = torch.zeros(S, dtype=DT)
    init[idx(0, 0, 0)] = 1.0
    rm = torch.tensor([0.0, -key_cost, apple_reward, door_reward], dtype=DT)
    m = TabularMDP(P, out, rm, init, reward_noise=reward_noise, name=f"key_to_door_d{n_distract}")
    m.validate()
    return m


def random_tree(depth: int = 3, branching: int = 2, n_actions: int = 2, n_outcomes: int = 4, seed: int = 0,
                reward_noise: float = 0.0) -> TabularMDP:
    """A random tree MDP: each node has `n_actions` actions, each leading stochastically to one of `branching`
    children; each transition emits a random outcome with a random reward mean."""
    g = torch.Generator().manual_seed(seed)
    levels = [1]
    for _ in range(depth):
        levels.append(levels[-1] * branching * n_actions)
    offs = [0]
    for n in levels:
        offs.append(offs[-1] + n)
    S = offs[-1]
    T = depth
    P = torch.zeros(T, S, n_actions, S, dtype=DT)
    out = torch.zeros(T, S, n_actions, S, dtype=torch.long)
    for t in range(T):
        for i in range(levels[t]):
            s = offs[t] + i
            for a in range(n_actions):
                probs = torch.distributions.Dirichlet(torch.ones(branching, dtype=DT)).sample() if False else \
                    torch.rand(branching, generator=g, dtype=DT) + 0.2
                probs = probs / probs.sum()
                for b in range(branching):
                    child = offs[t + 1] + (i * n_actions + a) * branching + b
                    P[t, s, a, child] = probs[b]
                    out[t, s, a, child] = int(torch.randint(0, n_outcomes, (1,), generator=g))
    rowsum = P.sum(-1)
    for t in range(T):
        for s in range(S):
            for a in range(n_actions):
                if rowsum[t, s, a] == 0:
                    P[t, s, a, s] = 1.0
    init = torch.zeros(S, dtype=DT)
    init[0] = 1.0
    rm = torch.randn(n_outcomes, generator=g, dtype=DT)
    m = TabularMDP(P, out, rm, init, reward_noise=reward_noise, name=f"tree_d{depth}_b{branching}_s{seed}")
    m.validate()
    return m
