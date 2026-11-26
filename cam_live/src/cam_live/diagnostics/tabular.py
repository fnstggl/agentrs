"""cam_live.diagnostics.tabular — exact tabular MDPs for credit-assignment and meta-gradient reference tests.

MASTER-OWNED. Copied unchanged (apart from this header) from superlearner/cam_unified/credit/exact.py at commit
3968583, where REINFORCE / all-actions / COCOA / HCA / RUDDER estimators were verified unbiased against enumerated
gradients. Shared by cam_live.credit and cam_live.reward so both test against the same ground truth.
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


def rhca_ratios(mdp: TabularMDP, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact R-HCA ratios rho[t, s, a, u] = h_{k>=0}(a | S_t=s, U=u) / pi(a|s) with occurrence counts that INCLUDE
    the same step (k = 0) (Ramesh et al. 2026, Eq. 4 with the time-abstracted, undiscounted hindsight of Prop. 5).
    R-HCA uses the ratio itself (not ratio - 1) on every k >= 0, so the immediate reward is credited through
    hindsight too. 1 where u cannot occur from s."""
    D, _ = future_outcome_counts(mdp, theta)
    pi = policy(theta).detach()
    onehot = torch.nn.functional.one_hot(mdp.outcome, mdp.U).to(DT)
    now = torch.einsum("tsap,tsapu->tsau", mdp.P, onehot)
    inc = now + D
    den = torch.einsum("sa,tsau->tsu", pi, inc)
    ok = den > eps
    return torch.where(ok[:, :, None, :], inc / den.clamp_min(eps)[:, :, None, :], torch.ones_like(inc))


def hindsight_action_probs(mdp: TabularMDP, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exact hindsight distribution h[t, s, u, a] = p_pi(A_t = a | S_t = s, U_future = u) (outcome occurrence
    weighted over k >= 1), i.e. the target a learned hindsight model must approximate."""
    D, Dbar = future_outcome_counts(mdp, theta)
    pi = policy(theta).detach()
    h = pi[None, :, :, None] * D / Dbar.clamp_min(eps)[:, :, None, :]    # [T, S, A, U]
    return h.permute(0, 1, 3, 2)


# ============================================================================================ enumeration

def enumerate_batch(mdp: TabularMDP, theta: torch.Tensor, max_traj: int = 200_000) -> tuple:
    """Every trajectory with its probability (exact expectations). Reward noise must be encoded in the outcomes
    (reward_noise == 0), e.g. `noisy_key_to_door`. Returns (Batch, probs [N])."""
    assert mdp.reward_noise == 0.0, "enumeration needs reward noise encoded in discrete outcomes"
    pi = policy(theta).detach()
    paths = [([], [], [], [], int(s0), float(p0)) for s0, p0 in enumerate(mdp.init.tolist()) if p0 > 0]
    for t in range(mdp.T):
        new = []
        for S_, A_, U_, R_, s, p in paths:
            for a in range(mdp.A):
                pa = float(pi[s, a])
                if pa == 0:
                    continue
                for s2 in torch.nonzero(mdp.P[t, s, a] > 0).flatten().tolist():
                    ps = float(mdp.P[t, s, a, s2])
                    u = int(mdp.outcome[t, s, a, s2])
                    new.append((S_ + [s], A_ + [a], U_ + [u], R_ + [float(mdp.reward_mean[u])], s2, p * pa * ps))
        paths = new
        if len(paths) > max_traj:
            raise ValueError(f"{len(paths)} trajectories exceed max_traj")
    tl = lambda i, dt=torch.long: torch.tensor([pth[i] for pth in paths], dtype=dt)
    b = Batch(tl(0), tl(1), tl(2), tl(3, DT), torch.log(pi[tl(0), tl(1)]))
    return b, torch.tensor([pth[5] for pth in paths], dtype=DT)


def exact_stats(per_traj: torch.Tensor, probs: torch.Tensor, exact: torch.Tensor, block=None) -> dict:
    """Exact relative bias ||E g - g||^2/||g||^2, relative variance E||g - E g||^2/||g||^2 and SNR (dB) of an
    estimator given its per-trajectory values and the trajectory probabilities (block: index into theta's rows)."""
    G = per_traj if block is None else per_traj[:, block]
    ex = exact if block is None else exact[block]
    P = probs[:, None]
    flat = G.reshape(G.shape[0], -1)
    m = (P * flat).sum(0)
    e = ex.reshape(-1)
    gn = float((e ** 2).sum())
    var = float((probs * ((flat - m) ** 2).sum(1)).sum())
    mse = float((probs * ((flat - e) ** 2).sum(1)).sum())
    bias = float(((m - e) ** 2).sum())
    import math as _m
    return {"rel_bias": bias / gn, "rel_var": var / gn, "snr_db": 10 * _m.log10(gn / max(mse, 1e-300)),
            "grad_norm2": gn}


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


def noisy_key_to_door(sigma0: float = 0.6, c: float = 0.3, sigma: float = 0.8, treasure: float = 1.0,
                      n_distract: int = 2) -> TabularMDP:
    """NoisyKeyDoor (research note E1): t=0 a=0 takes the key, R_0 = +-sigma0 (pure two-point noise);
    t=1..n_distract: R_t = c*[a=0] +- sigma (an immediate-only effect); last step: R = treasure iff key.
    Noise is part of the outcome (the next state carries the noise sign), so every trajectory is enumerable."""
    T = n_distract + 2
    # state (t, key, noise-bit of the transition that led here): index 4t + 2key + bit
    S = 4 * (T + 1)
    A = 2
    idx = lambda t, key, bit: 4 * t + 2 * key + bit
    vals = []

    def outc(v):
        v = round(float(v), 12)
        if v not in vals:
            vals.append(v)
        return vals.index(v)
    P = torch.zeros(T, S, A, S, dtype=DT)
    out = torch.zeros(T, S, A, S, dtype=torch.long)
    for t in range(T):
        for key in (0, 1):
            for bit in (0, 1):
                s = idx(t, key, bit)
                for a in range(A):
                    if t == 0:
                        k2 = 1 if a == 0 else 0
                        for nb, r in ((0, -sigma0), (1, sigma0)):
                            P[t, s, a, idx(1, k2, nb)] = 0.5
                            out[t, s, a, idx(1, k2, nb)] = outc(r)
                    elif t < T - 1:
                        for nb, r in ((0, c * (a == 0) - sigma), (1, c * (a == 0) + sigma)):
                            P[t, s, a, idx(t + 1, key, nb)] = 0.5
                            out[t, s, a, idx(t + 1, key, nb)] = outc(r)
                    else:
                        P[t, s, a, idx(t + 1, key, 0)] = 1.0
                        out[t, s, a, idx(t + 1, key, 0)] = outc(treasure if key else 0.0)
    rs = P.sum(-1)
    for t in range(T):
        for s in range(S):
            for a in range(A):
                if rs[t, s, a] == 0:
                    P[t, s, a, s] = 1.0
                    out[t, s, a, s] = outc(0.0)
    init = torch.zeros(S, dtype=DT)
    init[idx(0, 0, 0)] = 1.0
    m = TabularMDP(P, out, torch.tensor(vals, dtype=DT), init, 0.0, name=f"noisy_key_to_door_s{sigma0}")
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
