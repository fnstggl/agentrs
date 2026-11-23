"""Mandatory Test 8 and the numeric foundations, checked against hand-computed values."""
import math

import pytest
import torch

from superlearner.unified.dist import (balanced_kl, categorical_kl, sample_categorical, symexp, symlog,
                                       twohot_encode, twohot_loss, twohot_mean, make_bins, unimix_logits)
from superlearner.unified.learning.returns import ReturnNormalizer, duration_discount, lambda_returns


def test_symlog_inverse_and_values():
    x = torch.tensor([-1e6, -3.0, 0.0, 2.5, 1e6], dtype=torch.float64)
    assert torch.allclose(symexp(symlog(x)), x)
    assert symlog(torch.tensor(math.e - 1)).item() == pytest.approx(1.0, abs=1e-6)


def test_twohot_interpolates_and_preserves_mean():
    bins = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    t = twohot_encode(torch.tensor([0.25, 1.0, -5.0, 9.0]), bins)
    assert torch.allclose(t[0], torch.tensor([0.0, 0.75, 0.25, 0.0]))
    assert torch.allclose(t[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(t[2], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(t[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(t.sum(-1), torch.ones(4))


def test_twohot_mean_zero_for_uniform_and_recovers_target():
    bins = make_bins(255, -20, 20)
    assert twohot_mean(torch.zeros(255), bins).item() == pytest.approx(0.0, abs=1e-5)
    for v in [-250.0, -1.5, 0.0, 0.7, 4000.0]:
        target = twohot_encode(symlog(torch.tensor(v)), bins)
        logits = torch.log(target + 1e-12)
        assert twohot_mean(logits, bins).item() == pytest.approx(v, rel=1e-3, abs=1e-3)


def test_twohot_loss_minimised_at_target():
    bins = make_bins(41, -5, 5)
    logits = torch.zeros(41, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        twohot_loss(logits, torch.tensor(3.0), bins).backward()
        opt.step()
    assert twohot_mean(logits.detach(), bins).item() == pytest.approx(3.0, abs=0.05)


def test_kl_gradient_directions_are_separate():
    post = torch.randn(2, 4, 8, requires_grad=True)
    prior = torch.randn(2, 4, 8, requires_grad=True)
    dyn, rep = balanced_kl(post, prior, free_nats=0.0)
    g_post, g_prior = torch.autograd.grad(dyn.sum(), [post, prior], allow_unused=True)
    assert g_post is None and g_prior.abs().sum() > 0
    dyn, rep = balanced_kl(post, prior, free_nats=0.0)
    g_post, g_prior = torch.autograd.grad(rep.sum(), [post, prior], allow_unused=True)
    assert g_prior is None and g_post.abs().sum() > 0


def test_free_nats_apply_to_summed_kl():
    post = torch.zeros(1, 4, 8)
    prior = torch.zeros(1, 4, 8)
    dyn, rep = balanced_kl(post, prior, free_nats=1.0)
    assert dyn.item() == pytest.approx(1.0) and rep.item() == pytest.approx(1.0)
    assert categorical_kl(post, prior).item() == pytest.approx(0.0)


def test_unimix_floor_and_straight_through():
    logits = torch.tensor([[50.0, -50.0, -50.0, -50.0]], requires_grad=True)
    mixed = unimix_logits(logits, 0.01)
    assert torch.softmax(mixed, -1).min().item() == pytest.approx(0.0025, rel=1e-3)
    s = sample_categorical(mixed)
    assert torch.allclose(s.detach().sum(-1), torch.ones(1))
    assert set(s.detach().flatten().round().tolist()) <= {0.0, 1.0}
    s[0, 0].backward()
    assert logits.grad.abs().sum() > 0


def test_duration_discount():
    g = duration_discount(torch.tensor([0.0, 86400.0, 43200.0, 864000.0]), 0.9)
    assert g.tolist() == pytest.approx([1.0, 0.9, 0.9 ** 0.5, 0.9 ** 10], rel=1e-6)
    with pytest.raises(ValueError):
        duration_discount(torch.tensor([-1.0]), 0.9)


def test_lambda_return_hand_computed_varied_durations():
    r = torch.tensor([[1.0, 2.0]])
    v = torch.tensor([[0.0, 10.0, 20.0]])
    gam = torch.tensor([[0.5, 0.25]])
    c = torch.ones(1, 2)
    lam = 0.8
    # G_1 = 2 + .25 * (.2*20 + .8*20) = 7;  G_0 = 1 + .5 * (.2*10 + .8*7) = 4.8
    G = lambda_returns(r, v, gam, c, lam)
    assert G[0].tolist() == pytest.approx([4.8, 7.0])


def test_lambda_return_does_not_bootstrap_through_terminal():
    r = torch.tensor([[1.0, 2.0, 100.0]])
    v = torch.tensor([[0.0, 10.0, 1000.0, 1000.0]])
    gam = torch.full((1, 3), 0.5)
    c = torch.tensor([[1.0, 0.0, 1.0]])
    G = lambda_returns(r, v, gam, c, 0.8)
    # Step 1 is terminal: G_1 = 2. G_0 = 1 + .5 * (.2*10 + .8*2) = 2.8
    assert G[0, 1].item() == pytest.approx(2.0)
    assert G[0, 0].item() == pytest.approx(2.8)


def test_lambda_extremes():
    r = torch.tensor([[1.0, 1.0]])
    v = torch.tensor([[0.0, 5.0, 7.0]])
    g = torch.full((1, 2), 0.5)
    c = torch.ones(1, 2)
    assert lambda_returns(r, v, g, c, 0.0)[0].tolist() == pytest.approx([3.5, 4.5])   # one-step TD
    assert lambda_returns(r, v, g, c, 1.0)[0].tolist() == pytest.approx([3.25, 4.5])  # Monte Carlo + bootstrap


def test_return_normalizer_is_per_family_and_floored():
    n = ReturnNormalizer(rate=1.0)
    assert n.update(torch.linspace(0, 0.1, 100), "tiny") == 1.0
    assert n.update(torch.linspace(0, 1000, 101), "huge") == pytest.approx(900.0, rel=1e-3)
    assert n.scale("tiny") == 1.0
