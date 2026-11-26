"""Decision-relevant VOI matches the Gaussian knowledge gradient and ignores irreducible noise (KV-01, KV-03)."""
import torch

from superlearner.cam_unified.selfeval.voi import gaussian_arm_callables, knowledge_gradient_gaussian, voi_monte_carlo


def test_monte_carlo_voi_matches_closed_form_knowledge_gradient():
    mu = torch.tensor([0.0, 0.3, -0.2, 0.25], dtype=torch.float64)
    s2 = torch.tensor([1.0, 0.05, 2.0, 0.5], dtype=torch.float64)
    kg = knowledge_gradient_gaussian(mu, s2, 0.5)
    gen = torch.Generator().manual_seed(0)
    sample, update, q = gaussian_arm_callables(mu, s2, 0.5, gen)
    est = voi_monte_carlo(range(4), sample, update, q, (mu, s2), n=4000)
    assert torch.allclose(est.double(), kg, atol=0.02), (est, kg)
    assert int(est.argmax()) == int(kg.argmax())


def test_irreducible_noise_has_no_decision_value_even_when_surprise_is_large():
    mu = torch.tensor([1.0, 0.0], dtype=torch.float64)
    s2 = torch.tensor([1e-8, 1e-8], dtype=torch.float64)        # nothing left to learn
    gen = torch.Generator().manual_seed(1)
    sample, update, q = gaussian_arm_callables(mu, s2, 100.0, gen)   # very noisy observations: big surprise
    est = voi_monte_carlo(range(2), sample, update, q, (mu, s2), n=2000)
    assert float(est.abs().max()) < 1e-3
    ys = sample(1, 1000)
    surprise = torch.tensor([abs(y - 0.0) for y in ys]).mean()
    assert float(surprise) > 5.0


def test_cost_and_delay_reduce_value():
    mu = torch.tensor([0.0, 0.0], dtype=torch.float64)
    s2 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(2)
    sample, update, q = gaussian_arm_callables(mu, s2, 0.1, gen)
    free = voi_monte_carlo([0], sample, update, q, (mu, s2), n=2000)
    gen.manual_seed(2)
    costly = voi_monte_carlo([0], sample, update, q, (mu, s2), n=2000, cost=lambda a: 0.1, gamma=lambda a: 0.8)
    assert float(costly) < float(free) - 0.1
