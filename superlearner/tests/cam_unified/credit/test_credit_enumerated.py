"""Exact (enumerated, not sampled) properties of the credit estimators (research note §8.1-8.2; CR-04, CR-07)."""
import pytest
import torch

from superlearner.cam_unified.credit import estimators as E
from superlearner.cam_unified.credit import exact as X


def _setup(s0, seed=0):
    mdp = X.noisy_key_to_door(sigma0=s0)
    th = 0.5 * torch.randn(mdp.S, mdp.A, dtype=X.DT, generator=torch.Generator().manual_seed(seed))
    b, P = X.enumerate_batch(mdp, th)
    return mdp, th, b, P, X.exact_gradient(mdp, th)


@pytest.mark.parametrize("s0", [0.0, 0.6, 1.5])
def test_exact_expectations_equal_the_policy_gradient(s0):
    mdp, th, b, P, ex = _setup(s0)
    Q, V = X.q_values(mdp, th)
    assert abs(float(P.sum()) - 1) < 1e-12
    ests = {"reinforce": E.reinforce_grad(th, b), "advantage": E.reinforce_grad(th, b, V),
            "q_critic": E.all_actions_grad(th, b, Q), "cocoa": E.cocoa_grad(th, b, X.contribution_coefficients(mdp, th)),
            "rhca": E.rhca_grad(th, b, X.rhca_ratios(mdp, th)),
            "rudder_imm": E.rudder_grad(th, b, E.rudder_exact_redistribution(mdp, th, b), immediate_only=True)}
    for k, pt in ests.items():
        assert X.exact_stats(pt, P, ex)["rel_bias"] < 1e-20, k


def test_rhca_key_variance_is_invariant_to_immediate_noise_while_cocoa_grows():
    v_c, v_r = [], []
    for s0 in (0.0, 0.6, 1.5):
        mdp, th, b, P, ex = _setup(s0)
        v_c.append(X.exact_stats(E.cocoa_grad(th, b, X.contribution_coefficients(mdp, th)), P, ex, block=[0])["rel_var"])
        v_r.append(X.exact_stats(E.rhca_grad(th, b, X.rhca_ratios(mdp, th)), P, ex, block=[0])["rel_var"])
    assert max(v_r) - min(v_r) < 1e-9 * max(v_r)
    assert v_c[0] < v_c[1] < v_c[2]
    assert abs(v_c[0] - v_r[0]) < 1e-9 * v_r[0]


def test_hca_return_is_biased_in_the_cocoa_bandit():
    """COCOA App. L: rewards (1, -2), pi = (2/3, 1/3): HCA-return advantages (1/3, -4/3), true (1, -2)."""
    pa = torch.tensor([2 / 3, 1 / 3], dtype=X.DT)
    theta = torch.log(pa)[None, :]
    b = X.Batch(torch.zeros(2, 1, dtype=torch.long), torch.tensor([[0], [1]]), torch.zeros(2, 1, dtype=torch.long),
                torch.tensor([[1.0], [-2.0]], dtype=X.DT), torch.log(pa)[:, None])
    h = lambda t, s, a, z: torch.ones_like(z)          # each return identifies its arm
    per = E.hca_return_grad(theta, b, h)                # [2, 1, 2]
    # implied advantage of the taken arm = coef = (1 - pi/h) * Z
    coef = (1 - pa) * torch.tensor([1.0, -2.0], dtype=X.DT)
    assert torch.allclose(coef, torch.tensor([1 / 3, -4 / 3], dtype=X.DT))
    exact = (pa * (pa.new_tensor([1.0, -2.0]) - (pa * pa.new_tensor([1.0, -2.0])).sum()))   # pi (r - V)
    mean = (pa[:, None] * per[:, 0, :]).sum(0)
    true_grad = exact - pa * exact.sum()
    assert float((mean - true_grad).abs().max()) > 0.1
