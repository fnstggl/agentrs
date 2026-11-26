"""Exact checks of the intrinsic-reward meta-gradient (meta-reward spec 5.5, Exp J)."""
import torch

from superlearner.cam_unified.credit import exact as X
from superlearner.cam_unified.credit import meta_exact as M


def _setup():
    mdp = X.key_to_door(n_distract=5, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=X.DT)
    psi = 0.1 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(0))
    return mdp, th0, psi


def test_exact_meta_gradient_matches_finite_difference_and_detaching_removes_it():
    mdp, th0, psi = _setup()
    for K in (1, 2, 4):
        g, _ = M.exact_meta_gradient(mdp, psi, th0, K, 1.0)
        fd = M.finite_difference(mdp, psi, th0, K, 1.0, g)
        assert abs(fd - float(g.norm())) < 1e-6 * max(1.0, float(g.norm()))
        gd, _ = M.exact_meta_gradient(mdp, psi, th0, K, 1.0, detach_inner=True)
        assert float(gd.norm()) == 0.0


def test_sampled_truncated_meta_gradient_agrees_with_exact():
    mdp, th0, psi = _setup()
    ge, _ = M.exact_meta_gradient(mdp, psi, th0, 2, 1.0)
    g = torch.Generator().manual_seed(1)
    gs = torch.stack([M.sampled_meta_gradient(mdp, psi, th0, 2, 1.0, 256, 512, g) for _ in range(20)]).mean(0)
    assert float(torch.nn.functional.cosine_similarity(gs.flatten(), ge.flatten(), 0)) > 0.95


def test_meta_learned_reward_values_costly_exploration_and_beats_extrinsic_reward():
    mdp = X.key_to_door(n_distract=5, key_cost=0.3, apple_reward=0.2, door_reward=1.0)
    out = M.meta_train(mdp, K=3, alpha=1.0, beta=2.0, steps=40, exact=True)
    assert out["curve"][-1] > out["curve"][0] + 0.3
    psi = out["psi"]
    assert float(psi[0, 0, 0]) > 0.5 > float(psi[0, 0, 1])     # rewards the costly key pickup at t=0
    th0 = torch.zeros(mdp.S, mdp.A, dtype=X.DT, requires_grad=True)
    r_ext = (mdp.P * mdp.reward_mean[mdp.outcome]).sum(-1)
    thK = M.inner_updates_exact(mdp, th0, r_ext, 3, 1.0)
    assert out["curve"][-1] > float(M.expected_extrinsic_return(mdp, thK.detach())) + 0.1
