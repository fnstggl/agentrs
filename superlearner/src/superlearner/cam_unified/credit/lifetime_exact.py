"""Exact lifetime meta-gradient on a bandit lifetime (Zheng et al. 2020, Eq. 4; research note §8.5, E5).

A lifetime is L one-step episodes of a 3-armed bandit whose extrinsic arm rewards (the task) are drawn per lifetime.
The intrinsic reward is a table eta[arm, sign(r_ext)] (it sees the current outcome, as Zheng's r_eta(tau_{t+1})
does). A fresh softmax learner (theta_0 = 0) updates after every episode:
    theta_{t+1} = theta_t + alpha * eta[a_t, sign(r_t)] * grad log pi_{theta_t}(a_t).
Outer objective J(eta) = E_task E[ sum_t r_ext(a_t) ] (lifetime return). Exact J and grad J by enumeration of 3^L
action sequences x tasks; Monte Carlo estimators: Zheng Eq. 4 with full backprop of d theta_t / d eta through the
lifetime (unbiased), truncated backprop through the last N updates (biased), and the myopic episodic objective.
"""
from __future__ import annotations

import itertools
import math

import torch

DT = torch.float64
TASKS = (torch.tensor([1.0, -0.5, 0.2], dtype=DT), torch.tensor([-1.0, -0.5, 0.2], dtype=DT))


def _bucket(r):
    return torch.where(r < -1e-9, 0, torch.where(r > 1e-9, 2, 1))


def exact_J(eta: torch.Tensor, L: int = 5, alpha: float = 1.0, tasks=TASKS) -> torch.Tensor:
    NA = eta.shape[0]
    J = eta.new_zeros(())
    for task in tasks:
        for seq in itertools.product(range(NA), repeat=L):
            th = eta.new_zeros(NA)
            logp, ret = eta.new_zeros(()), 0.0
            for a in seq:
                p = torch.softmax(th, 0)
                logp = logp + torch.log(p[a])
                r = float(task[a])
                ret += r
                b = 0 if r < -1e-9 else (2 if r > 1e-9 else 1)
                th = th + alpha * eta[a, b] * (torch.nn.functional.one_hot(torch.tensor(a), NA).to(DT) - p)
            J = J + torch.exp(logp) * ret / len(tasks)
    return J


def exact_grad(eta: torch.Tensor, **kw) -> tuple:
    e = eta.detach().clone().requires_grad_(True)
    J = exact_J(e, **kw)
    (g,) = torch.autograd.grad(J, e)
    return g, float(J.detach())


def mc_estimate(eta: torch.Tensor, n: int, L: int = 5, alpha: float = 1.0, objective: str = "lifetime",
                trunc=None, seed: int = 0, chunks: int = 20, tasks=TASKS) -> tuple:
    """Vectorised Zheng Eq. 4 over n sampled lifetimes; trunc=N rebuilds theta_t from theta_{t-N} detached."""
    e = eta.detach().clone().requires_grad_(True)
    NA = e.shape[0]
    g = torch.Generator().manual_seed(seed)
    R = torch.stack(tasks)[torch.randint(len(tasks), (n,), generator=g)]
    th_hist = [torch.zeros(n, NA, dtype=DT)]
    acts, rews, logps = [], [], []
    oh = lambda a: torch.nn.functional.one_hot(a, NA).to(DT)
    for t in range(L):
        th = th_hist[-1]
        if trunc is not None and t - trunc >= 0:
            th = th_hist[t - trunc].detach()
            for k in range(t - trunc, t):
                th = th + alpha * e[acts[k], _bucket(rews[k])][:, None] * (oh(acts[k]) - torch.softmax(th, 1))
        p = torch.softmax(th, 1)
        a = torch.multinomial(p.detach(), 1, generator=g).squeeze(1)
        r = R[torch.arange(n), a]
        logps.append(torch.log(p[torch.arange(n), a]))
        acts.append(a)
        rews.append(r)
        th_hist.append(th_hist[-1] + alpha * e[a, _bucket(r)][:, None] * (oh(a) - torch.softmax(th_hist[-1], 1)))
    rt = torch.stack(rews, 1)
    G = rt.flip(1).cumsum(1).flip(1) if objective == "lifetime" else rt
    sur = (torch.stack(logps, 1) * G.detach()).sum(1)
    grads = []
    for c in range(chunks):
        sl = slice(c * n // chunks, (c + 1) * n // chunks)
        grads.append(torch.autograd.grad(sur[sl].mean(), e, retain_graph=True)[0])
    Gs = torch.stack(grads)
    return Gs.mean(0), Gs.std(0) / math.sqrt(chunks)


def compare(est: torch.Tensor, se: torch.Tensor, exact: torch.Tensor) -> dict:
    return {"rel_err": float((est - exact).norm() / exact.norm()),
            "cos": float((est * exact).sum() / (est.norm() * exact.norm())),
            "max_abs_z": float(((est - exact) / (se + 1e-12)).abs().max())}
