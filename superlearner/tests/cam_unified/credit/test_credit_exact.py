"""Exact references for credit assignment: every estimator is checked against the enumerated policy gradient."""
import pytest
import torch

from superlearner.cam_unified.credit import estimators as E
from superlearner.cam_unified.credit import exact as X

MDPS = [lambda: X.key_to_door(n_distract=5), lambda: X.key_to_door(n_distract=5, reward_noise=1.0),
        lambda: X.random_tree(depth=3, seed=2)]


def _theta(mdp, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(mdp.S, mdp.A, dtype=X.DT, generator=g)


@pytest.mark.parametrize("make", MDPS)
def test_exact_gradient_matches_finite_differences(make):
    mdp = make()
    th = _theta(mdp)
    g = X.exact_gradient(mdp, th)
    eps = 1e-6
    for (s, a) in [(0, 0), (0, 1), (min(5, mdp.S - 1), 1)]:
        d = torch.zeros_like(th); d[s, a] = eps
        fd = (X.expected_return(mdp, th + d) - X.expected_return(mdp, th - d)) / (2 * eps)
        assert abs(float(fd) - float(g[s, a])) < 1e-7


@pytest.mark.parametrize("make", MDPS)
def test_contribution_coefficients_have_zero_policy_mean_and_hindsight_normalises(make):
    mdp = make()
    th = _theta(mdp, 1)
    pi = X.policy(th)
    w = X.contribution_coefficients(mdp, th)
    _, Dbar = X.future_outcome_counts(mdp, th)
    m = torch.einsum("sa,tsau->tsu", pi, w)
    assert float(m[Dbar > 1e-9].abs().max()) < 1e-9
    h = X.hindsight_action_probs(mdp, th)
    tot = h.sum(-1)
    assert float((tot[Dbar > 1e-9] - 1).abs().max()) < 1e-9


@pytest.mark.parametrize("make", MDPS)
def test_estimators_are_unbiased_and_rudder_conserves_return(make):
    mdp = make()
    th = _theta(mdp, 2)
    ex = X.exact_gradient(mdp, th)
    Q, V = X.q_values(mdp, th)
    w = X.contribution_coefficients(mdp, th)
    b = X.sample(mdp, th, 20000, torch.Generator().manual_seed(3))
    rr = E.rudder_exact_redistribution(mdp, th, b)
    assert float((rr.sum(1) - b.r.sum(1)).abs().max()) < 1e-9
    for pt in [E.reinforce_grad(th, b), E.reinforce_grad(th, b, V), E.all_actions_grad(th, b, Q),
               E.cocoa_grad(th, b, w), E.rudder_grad(th, b, rr), E.rudder_grad(th, b, rr, immediate_only=True)]:
        st = E.estimator_stats(pt, ex)
        # componentwise z-test on the mean estimate
        flat = pt.reshape(pt.shape[0], -1)
        z = (flat.mean(0) - ex.reshape(-1)) / (flat.std(0) / flat.shape[0] ** 0.5).clamp_min(1e-12)
        assert float(z.abs().max()) < 5.0, st
        assert st["cos_to_exact"] > 0.99


def test_cocoa_reduces_variance_on_key_to_door():
    mdp = X.key_to_door(n_distract=8)
    th = _theta(mdp, 4)
    ex = X.exact_gradient(mdp, th)
    b = X.sample(mdp, th, 4000, torch.Generator().manual_seed(5))
    _, V = X.q_values(mdp, th)
    v_rf = E.estimator_stats(E.reinforce_grad(th, b, V), ex)["var_trace"]
    v_co = E.estimator_stats(E.cocoa_grad(th, b, X.contribution_coefficients(mdp, th)), ex)["var_trace"]
    assert v_co < 0.6 * v_rf
