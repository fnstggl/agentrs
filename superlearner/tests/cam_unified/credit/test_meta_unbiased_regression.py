"""Regression: the sampled meta-gradient (tabular reference AND the CAM-scale MetaRewardLearner the meta step uses) is
unbiased against the exact meta-gradient, checked by 2000 independent repetitions with per-coordinate z-scores
(max |z| < 4) and the relative error of the mean — never by cosine alone (research note §8.5, §10.3; plan §3 W8).

Setting (the audit's): key_to_door with 5 distractors, key cost 0.3, apples 0.2, door 1.0; psi ~ 0.1 N(0, 1) table;
theta_0 = 0; K = 4 on-policy inner REINFORCE steps on r_psi (alpha = 1) with 64 inner trajectories; 256 query
trajectories. Without the inner-data score term the committed estimator had rel. error 0.015 and max |z| 10.2.
"""
import math

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.credit import exact as X
from superlearner.cam_unified.credit import meta_exact as M
from superlearner.cam_unified.selfeval.metagrad import InnerBatch, MetaRewardLearner, QueryBatch

REP = 2000
K = 4
N_INNER, N_QUERY = 64, 256


def _setup():
    mdp = X.key_to_door(n_distract=5, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=X.DT)
    psi = 0.1 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    ge, J = M.exact_meta_gradient(mdp, psi, th0, K, 1.0)
    return mdp, th0, psi, ge, J


def _z_report(G: torch.Tensor, exact: torch.Tensor) -> dict:
    m, se = G.mean(0), G.std(0) / math.sqrt(G.shape[0])
    ex = exact.reshape(-1)
    mask = (ex.abs() > 1e-6) | (m.abs() > 1e-6)          # coordinates with exact or estimated mass
    z = ((m - ex)[mask] / se[mask].clamp_min(1e-12)).abs()
    return {"max_abs_z": float(z.max()), "rel_err": float((m - ex).norm() / ex.norm()),
            "rel_se": float(se.norm() / ex.norm()), "n_coords": int(mask.sum())}


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(4)
    return _setup()


def test_tabular_sampled_meta_gradient_is_unbiased_2000_repetitions(setup):
    mdp, th0, psi, ge, J = setup
    gen = torch.Generator().manual_seed(7)
    G = torch.stack([M.sampled_meta_gradient(mdp, psi, th0, K, 1.0, N_INNER, N_QUERY, gen, score_baseline=J)
                     for _ in range(REP)]).reshape(REP, -1)
    rep = _z_report(G, ge)
    assert rep["max_abs_z"] < 4.0, rep
    assert rep["rel_err"] < 0.03, rep


def test_tabular_estimator_without_inner_score_term_is_detectably_biased(setup):
    """The same 2000-repetition test has the power to see the omitted term (guards the test itself)."""
    mdp, th0, psi, ge, J = setup
    gen = torch.Generator().manual_seed(7)
    G = torch.stack([M.sampled_meta_gradient(mdp, psi, th0, K, 1.0, N_INNER, N_QUERY, gen, inner_score=False)
                     for _ in range(REP)]).reshape(REP, -1)
    rep = _z_report(G, ge)
    assert rep["max_abs_z"] > 6.0, rep


class TabPolicy(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(S, A, dtype=torch.float64))

    def forward(self, s):
        return self.theta[s]


class TabReward(nn.Module):
    def __init__(self, psi):
        super().__init__()
        self.psi = nn.Parameter(psi.clone())

    def forward(self, t, s, a):
        return self.psi[t, s, a]


def _cam_estimate(mdp, psi, gen, J, policy, reward) -> torch.Tensor:
    T = mdp.T
    # inner lr alpha * T: the class minimises a per-step mean, the exact inner step ascends the sum over T steps
    mrl = MetaRewardLearner(policy, reward, inner_lr=1.0 * T, K=K, normalize_inner_adv=False)
    with torch.no_grad():
        policy.theta.zero_()
    fast = mrl.start()
    for _ in range(K):
        b = X.sample(mdp, fast["theta"].detach(), N_INNER, gen)
        tt = torch.arange(T)[None, :].expand_as(b.s)
        fast, _ = mrl.inner_step(fast, InnerBatch({"s": b.s}, b.a, torch.ones_like(b.s, dtype=torch.bool),
                                                  {"t": tt, "s": b.s, "a": b.a}))
    q = X.sample(mdp, fast["theta"].detach(), N_QUERY, gen)
    rtg = torch.flip(torch.cumsum(torch.flip(q.r, [1]), 1), [1])        # verified return-to-go [N, T]
    qb = QueryBatch({"s": q.s}, q.a, torch.ones_like(q.s, dtype=torch.bool), utility=rtg)
    (g,) = mrl.meta_gradient(fast, qb, score_baseline=J)
    return -g * T                                                       # outer loss = -(per-step mean surrogate)


def test_cam_meta_reward_learner_is_unbiased_2000_repetitions(setup):
    mdp, th0, psi, ge, J = setup
    gen = torch.Generator().manual_seed(11)
    pol, rew = TabPolicy(mdp.S, mdp.A), TabReward(psi)
    G = torch.stack([_cam_estimate(mdp, psi, gen, J, pol, rew) for _ in range(REP)]).reshape(REP, -1)
    rep = _z_report(G, ge)
    assert rep["max_abs_z"] < 4.0, rep
    assert rep["rel_err"] < 0.03, rep
