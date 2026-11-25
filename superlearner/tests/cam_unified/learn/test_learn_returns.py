"""Returns on the SMDP clock: GAE / lambda-returns / V-trace vs independent scalar reference implementations,
semi-Markov discount, option-level targets and percentile normalisation."""
import math
import random

import pytest
import torch

from superlearner.cam_unified.learn.returns import (ReturnNormalizer, gae, lambda_returns, option_targets,
                                                    semi_markov_discount, vtrace)


# ---------------------------------------------------------------- scalar references (written independently, per row)

def ref_gae_row(r, v, g, c, lam, L):
    """r, g, c: lists length >= L; v: list length >= L+1 (v[L] = bootstrap). Returns advantages for t < L."""
    adv = [0.0] * L
    nxt = 0.0
    for t in range(L - 1, -1, -1):
        delta = r[t] + g[t] * c[t] * v[t + 1] - v[t]
        nxt = delta + g[t] * c[t] * lam * nxt
        adv[t] = nxt
    return adv


def ref_vtrace_row(r, v, g, c, log_ratio, lam, rho_bar, c_bar, L):
    """Direct (non-recursive) definition: vs_t = V_t + sum_{s=t}^{L-1} (prod_{i=t}^{s-1} gamma_i c_i lam min(c_bar, r_i))
    * min(rho_bar, r_s) * (r_s + gamma_s c_s V_{s+1} - V_s)."""
    ratio = [math.exp(x) for x in log_ratio]
    vs = []
    for t in range(L):
        acc, prod = 0.0, 1.0
        for s in range(t, L):
            delta = min(rho_bar, ratio[s]) * (r[s] + g[s] * c[s] * v[s + 1] - v[s])
            acc += prod * delta
            prod *= g[s] * c[s] * lam * min(c_bar, ratio[s])
        vs.append(v[t] + acc)
    vs.append(v[L])
    pg = [min(rho_bar, ratio[t]) * (r[t] + g[t] * c[t] * vs[t + 1] - v[t]) for t in range(L)]
    return vs, pg


def _random_batch(B=5, T=9, seed=0):
    rng = random.Random(seed)
    lengths = [rng.randint(1, T) for _ in range(B)]
    lengths[0] = T
    r = torch.tensor([[rng.gauss(0, 1) for _ in range(T)] for _ in range(B)])
    v = torch.tensor([[rng.gauss(0, 2) for _ in range(T + 1)] for _ in range(B)])
    dt = torch.tensor([[rng.uniform(0, 3) for _ in range(T)] for _ in range(B)])
    g = semi_markov_discount(dt, 0.1)
    c = torch.tensor([[0.0 if rng.random() < 0.15 else 1.0 for _ in range(T)] for _ in range(B)])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b, L in enumerate(lengths):
        mask[b, :L] = True
    lr = torch.tensor([[rng.gauss(0, 0.7) for _ in range(T)] for _ in range(B)])
    return r, v, g, c, mask, lengths, lr


def test_semi_markov_discount_exact_and_compositional():
    k = 0.05
    g = semi_markov_discount(torch.tensor([0.0, 1.0, 2.5, 10.0]), k)
    assert g.tolist() == pytest.approx([1.0, math.exp(-0.05), math.exp(-0.125), math.exp(-0.5)], rel=1e-6)
    # a 3-day option is discounted exactly like three 1-day steps
    assert float(semi_markov_discount(torch.tensor(3.0), k)) == pytest.approx(
        float(semi_markov_discount(torch.tensor(1.0), k)) ** 3, rel=1e-6)
    with pytest.raises(ValueError):
        semi_markov_discount(torch.tensor([-0.1]), k)
    with pytest.raises(ValueError):
        semi_markov_discount(torch.tensor([float("nan")]), k)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_gae_matches_reference_with_padding_terminals_and_smdp_discounts(lam):
    r, v, g, c, mask, lengths, _ = _random_batch(seed=int(lam * 100))
    adv, ret = gae(r, v, g, c, lam, mask)
    for b, L in enumerate(lengths):
        ref = ref_gae_row(r[b].tolist(), v[b].tolist(), g[b].tolist(), c[b].tolist(), lam, L)
        assert adv[b, :L].tolist() == pytest.approx(ref, abs=1e-5)
        assert adv[b, L:].abs().sum() == 0
        assert ret[b, :L].tolist() == pytest.approx([a + v[b, t].item() for t, a in enumerate(ref)], abs=1e-5)


@pytest.mark.parametrize("lam", [0.0, 0.7, 1.0])
def test_gae_returns_equal_lambda_returns(lam):
    r, v, g, c, mask, _, _ = _random_batch(seed=3)
    _, ret = gae(r, v, g, c, lam, mask)
    G = lambda_returns(r, v, g, c, lam, mask)
    assert torch.allclose(ret, G, atol=1e-5)


def test_lambda_return_hand_computed_from_cam61_suite():
    # same numbers as superlearner/tests/unified/test_returns.py (the copied implementation must agree)
    r = torch.tensor([[1.0, 2.0]])
    v = torch.tensor([[0.0, 10.0, 20.0]])
    gam = torch.tensor([[0.5, 0.25]])
    assert lambda_returns(r, v, gam, torch.ones(1, 2), 0.8)[0].tolist() == pytest.approx([4.8, 7.0])
    r = torch.tensor([[1.0, 2.0, 100.0]])
    v = torch.tensor([[0.0, 10.0, 1000.0, 1000.0]])
    G = lambda_returns(r, v, torch.full((1, 3), 0.5), torch.tensor([[1.0, 0.0, 1.0]]), 0.8)
    assert G[0, :2].tolist() == pytest.approx([2.8, 2.0])


@pytest.mark.parametrize("lam,rho_bar,c_bar", [(1.0, 1.0, 1.0), (0.9, 1.0, 1.0), (1.0, 2.0, 0.5), (0.8, 0.7, 1.3)])
def test_vtrace_matches_direct_definition(lam, rho_bar, c_bar):
    r, v, g, c, mask, lengths, lr = _random_batch(seed=11)
    vs, pg, rho = vtrace(r, v, g, c, lr, lam, rho_bar, c_bar, mask=mask)
    for b, L in enumerate(lengths):
        ref_vs, ref_pg = ref_vtrace_row(r[b].tolist(), v[b].tolist(), g[b].tolist(), c[b].tolist(), lr[b].tolist(),
                                        lam, rho_bar, c_bar, L)
        assert vs[b, :L + 1].tolist() == pytest.approx(ref_vs, abs=1e-4)
        assert pg[b, :L].tolist() == pytest.approx(ref_pg, abs=1e-4)
        assert pg[b, L:].abs().sum() == 0


def test_vtrace_on_policy_equals_lambda_returns():
    r, v, g, c, mask, _, _ = _random_batch(seed=5)
    for lam in (0.6, 1.0):
        vs, _, _ = vtrace(r, v, g, c, torch.zeros_like(r), lam, mask=mask)
        G = lambda_returns(r, v, g, c, lam, mask)
        assert torch.allclose(vs[:, :-1] * mask, G, atol=1e-5)


def test_vtrace_truncation_bounds_off_policy_influence():
    """With a very improbable action under pi (log ratio -> -inf), the target falls back to V_t at that step."""
    r = torch.tensor([[5.0, 5.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    g = torch.ones(1, 2)
    c = torch.ones(1, 2)
    vs, pg, rho = vtrace(r, v, g, c, torch.tensor([[-50.0, 0.0]]), 1.0)
    assert vs[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert rho[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_option_targets_semi_markov_hand_computed():
    kappa = 0.1
    dt = torch.tensor([[0.5, 1.5, 2.0, 1.0, 0.25]])
    gam = semi_markov_discount(dt, kappa)
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    v = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    start = torch.tensor([[True, False, False, True, False]])
    c = torch.ones(1, 5)
    tgt, G = option_targets(r, v, gam, c, start)
    # option 1: steps 0..2, Delta = 4 days -> R = 1 + e^{-.05}*2 + e^{-.2}*3 ; target = R + e^{-.4} * V[3]
    R1 = 1 + math.exp(-0.05) * 2 + math.exp(-0.2) * 3
    assert tgt[0, 0].item() == pytest.approx(R1 + math.exp(-0.4) * 13.0, rel=1e-5)
    assert G[0, 0].item() == pytest.approx(math.exp(-kappa * 4.0), rel=1e-5)
    # option 2: steps 3..4, Delta = 1.25 -> R = 4 + e^{-.1}*5 ; target = R + e^{-.125} * V[5]
    assert tgt[0, 3].item() == pytest.approx(4 + math.exp(-0.1) * 5 + math.exp(-0.125) * 15.0, rel=1e-5)
    assert tgt[0, [1, 2, 4]].abs().sum() == 0
    # a terminal inside the option truncates its return (no bootstrap)
    c2 = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]])
    start2 = torch.tensor([[True, False, True, False, False]])
    tgt2, G2 = option_targets(r, v, gam, c2, start2)
    assert tgt2[0, 0].item() == pytest.approx(1 + math.exp(-0.05) * 2, rel=1e-5)
    assert G2[0, 0].item() == 0.0


def test_return_normalizer_percentile_scaling_per_family():
    n = ReturnNormalizer(rate=1.0)
    assert n.update(torch.linspace(0, 0.1, 100), "tiny") == 1.0        # floored: small returns are not inflated
    assert n.update(torch.linspace(0, 1000, 101), "cents") == pytest.approx(900.0, rel=1e-3)
    assert n.scale("tiny") == 1.0
    m = torch.zeros(101, dtype=torch.bool)
    m[:11] = True                                                      # only the masked returns count
    n2 = ReturnNormalizer(rate=1.0)
    assert n2.update(torch.linspace(0, 1000, 101), "x", m) == pytest.approx(90.0, rel=1e-3)
    s = n.state_dict()
    n3 = ReturnNormalizer()
    n3.load_state_dict(s)
    assert n3.scale("cents") == pytest.approx(900.0, rel=1e-3)
