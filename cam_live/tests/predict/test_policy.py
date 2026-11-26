"""CandidatePolicy and exploration rules: exact propensities, latent dependence, differentiable updates."""
import math

import numpy as np
import pytest
import scipy.stats as st
import torch

from cam_live.predict import policy as P
from cam_live.predict.policy import CandidatePolicy, PolicyConfig, apply_floor, latent_dependence, select, \
    selection_probs

DM = 16


def _policy(seed=0, hidden=48):
    torch.manual_seed(seed)
    return CandidatePolicy(PolicyConfig(d_model=DM, hidden=hidden))


def _batch(gen, b=8, n=6):
    z = torch.randn(b, n, DM, generator=gen)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 3:] = False
    mask[1, 1:] = False                                                # a decision with a single feasible candidate
    return z, mask, torch.randn(b, DM, generator=gen)


def test_masked_softmax_value_and_q_shapes(gen):
    pol = _policy()
    z, mask, b = _batch(gen)
    out = pol(z, mask, b)
    assert out.probs.shape == (8, 6) and out.value.shape == (8,) and out.q.shape == (8, 6)
    np.testing.assert_allclose(out.probs.sum(-1).detach(), 1.0, rtol=1e-6)
    assert float(out.probs[0, 3:].detach().abs().max()) == 0.0 and float(out.probs[1, 0].detach()) == 1.0
    assert bool(torch.isinf(out.logits[0, 3:]).all()) and float(out.q[0, 3:].detach().abs().max()) == 0.0
    np.testing.assert_allclose(out.log_prob(torch.zeros(8, dtype=torch.long)).detach(), out.log_probs[:, 0].detach())
    assert float(out.entropy()[1].detach()) == 0.0 and float(out.entropy()[2].detach()) > 0.5
    # output layers are small-random, not zero: scores differ between candidates and every parameter gets gradient
    assert float(out.raw_logits.detach().std()) > 1e-4 and float(out.q.detach().std()) > 1e-4
    (out.log_prob(torch.zeros(8, dtype=torch.long)).sum() + out.value.sum() + out.q.sum()).backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pol.parameters())
    with pytest.raises(ValueError):
        pol(z, torch.zeros(8, 6, dtype=torch.bool), b)


def test_variable_candidate_sets_padding_and_permutation(gen):
    pol = _policy()
    z, _, b = _batch(gen, b=4, n=5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    base = pol(z, mask, b)
    # padding with masked candidates changes nothing for the real ones
    zp = torch.cat([z, torch.randn(4, 3, DM, generator=gen) * 50], 1)
    mp = torch.cat([mask, torch.zeros(4, 3, dtype=torch.bool)], 1)
    padded = pol(zp, mp, b)
    np.testing.assert_allclose(padded.probs[:, :5].detach(), base.probs.detach(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(padded.value.detach(), base.value.detach(), rtol=1e-5, atol=1e-6)
    # re-ordering the candidates re-orders the probabilities
    perm = torch.tensor([3, 0, 4, 1, 2])
    np.testing.assert_allclose(pol(z[:, perm], mask, b).probs.detach(), base.probs[:, perm].detach(), rtol=1e-5,
                               atol=1e-7)


# ----------------------------------------------------------------------------------------------------------------
# exploration rules
# ----------------------------------------------------------------------------------------------------------------
def _rule_inputs(gen, b=5, n=6):
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 4:] = False
    q_mean = torch.randn(b, n, generator=gen)
    q_std = torch.rand(b, n, generator=gen) + 0.1
    return dict(a_mask=mask, logits=torch.randn(b, n, generator=gen), q_mean=q_mean, q_std=q_std,
                q_samples=q_mean + q_std * torch.randn(4000, b, n, generator=gen),
                info_gain=torch.rand(b, n, generator=gen))


@pytest.mark.parametrize("rule", [P.RULE_POLICY, P.RULE_THOMPSON, P.RULE_UCB, P.RULE_GREEDY, P.RULE_INFO_GAIN])
def test_every_rule_returns_exact_propensities(rule, gen):
    kw = _rule_inputs(gen)
    eps = 0.06
    probs = selection_probs(rule, eps_floor=eps, **kw)
    mask = kw["a_mask"]
    assert probs.dtype == torch.float64
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-12)
    assert float(probs[~mask].abs().max()) == 0.0                          # infeasible candidates: exactly zero
    floor = eps / mask.sum(-1, keepdim=True)
    assert bool((probs[mask] >= floor.expand_as(probs)[mask] - 1e-15).all())   # every feasible one keeps support
    # the recorded vector is the distribution actually sampled: empirical frequencies agree (chi-square)
    reps = 30000
    big = {k: (v.repeat_interleave(reps, 1 if k == "q_samples" else 0) if k != "q_samples" else v)
           for k, v in kw.items()}
    if rule == P.RULE_THOMPSON:
        sel_probs = probs.repeat_interleave(reps, 0)
        idx = torch.multinomial(sel_probs, 1, generator=gen).squeeze(-1)
    else:
        big.pop("q_samples")
        sel = select(rule, eps_floor=eps, generator=gen, **big)
        idx, sel_probs = sel.index, sel.probs
        np.testing.assert_allclose(sel.propensity, sel_probs[torch.arange(len(idx)), idx])
    for row in range(5):
        counts = torch.bincount(idx[row * reps:(row + 1) * reps], minlength=6).double()
        expect = probs[row] * reps
        keep = expect > 0
        assert float(counts[~keep].sum()) == 0.0
        assert st.chisquare(counts[keep].numpy(), expect[keep].numpy()).pvalue > 1e-4


def test_select_is_seeded_and_reports_what_it_used(gen):
    kw = _rule_inputs(gen)
    s1 = select(P.RULE_THOMPSON, generator=torch.Generator().manual_seed(3), eps_floor=0.05, **kw)
    s2 = select(P.RULE_THOMPSON, generator=torch.Generator().manual_seed(3), eps_floor=0.05, **kw)
    assert torch.equal(s1.index, s2.index) and torch.equal(s1.probs, s2.probs) and s1.rule == "thompson"
    assert bool((s1.propensity > 0).all()) and s1.eps_floor == 0.05
    lst = s1.probabilities(0, n=4)
    assert len(lst) == 4 and abs(sum(lst) - 1.0) < 1e-12                  # row 0 has four feasible candidates
    assert int(s1.diagnostics["n_support"][0]) == 4
    with pytest.raises(ValueError):
        select("softmax_magic", a_mask=kw["a_mask"])
    for rule in (P.RULE_POLICY, P.RULE_THOMPSON, P.RULE_UCB, P.RULE_GREEDY, P.RULE_INFO_GAIN, P.RULE_RANDOMIZED):
        with pytest.raises(ValueError):
            selection_probs(rule, a_mask=kw["a_mask"])                    # required inputs missing
    with pytest.raises(ValueError):
        apply_floor(torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool), 1.0)


def test_rule_semantics(gen):
    mask = torch.ones(1, 3, dtype=torch.bool)
    q_mean = torch.tensor([[1.0, 0.9, 0.0]])
    q_std = torch.tensor([[0.0, 0.5, 0.1]])
    greedy = selection_probs(P.RULE_GREEDY, a_mask=mask, q_mean=q_mean, eps_floor=0.0)
    ucb = selection_probs(P.RULE_UCB, a_mask=mask, q_mean=q_mean, q_std=q_std, eps_floor=0.0, ucb_c=1.0)
    assert greedy.tolist() == [[1.0, 0.0, 0.0]] and ucb.tolist() == [[0.0, 1.0, 0.0]]   # optimism prefers the uncertain arm
    assert selection_probs(P.RULE_UCB, a_mask=mask, q_mean=q_mean, q_std=q_std, eps_floor=0.0, ucb_c=0.1
                           ).tolist() == [[1.0, 0.0, 0.0]]
    g2 = selection_probs(P.RULE_GREEDY, a_mask=mask, q_mean=q_mean, eps_floor=0.3)
    np.testing.assert_allclose(g2, [[0.8, 0.1, 0.1]])
    # Thompson: share of posterior draws in which each arm is best == P(arm best) for two normal arms
    mu, sd = torch.tensor([0.3, 0.0]), torch.tensor([0.5, 1.0])
    samples = mu + sd * torch.randn(200000, 1, 2, generator=gen)
    th = selection_probs(P.RULE_THOMPSON, a_mask=torch.ones(1, 2, dtype=torch.bool), q_samples=samples, eps_floor=0.0)
    np.testing.assert_allclose(th[0, 0], st.norm.cdf(0.3 / math.sqrt(0.25 + 1.0)), atol=0.005)
    # information-gain weighting: with equal values the more informative candidate is preferred, smoothly
    ig = selection_probs(P.RULE_INFO_GAIN, a_mask=mask, info_gain=torch.tensor([[0.0, 1.0, 2.0]]), eps_floor=0.0)
    np.testing.assert_allclose(ig, torch.softmax(torch.tensor([[0.0, 1.0, 2.0]]), -1).double(), rtol=1e-6)
    tilted = selection_probs(P.RULE_INFO_GAIN, a_mask=mask, q_mean=torch.tensor([[5.0, 0.0, 0.0]]),
                             info_gain=torch.tensor([[0.0, 1.0, 2.0]]), eps_floor=0.0, ig_weight=0.5)
    assert float(tilted[0, 0]) > 0.9
    # policy temperature
    lg = torch.tensor([[2.0, 0.0, -2.0]])
    cold = selection_probs(P.RULE_POLICY, a_mask=mask, logits=lg, temperature=0.25, eps_floor=0.0)
    hot = selection_probs(P.RULE_POLICY, a_mask=mask, logits=lg, temperature=4.0, eps_floor=0.0)
    assert float(cold[0, 0]) > 0.99 and float(hot[0, 0]) < 0.6


def test_randomised_assignment_over_arms(gen):
    mask = torch.tensor([[True, True, True, True, False]])
    arms = torch.tensor([[True, False, True, False, False]])
    p = selection_probs(P.RULE_RANDOMIZED, a_mask=mask, arms=arms, eps_floor=0.0)
    assert p.tolist() == [[0.5, 0.0, 0.5, 0.0, 0.0]]                       # non-arms are never assigned
    p2 = selection_probs(P.RULE_RANDOMIZED, a_mask=mask, arms=arms, arm_probs=torch.tensor([[0.8, 0, 0.2, 0, 0]]),
                         eps_floor=0.1)
    np.testing.assert_allclose(p2, [[0.77, 0.0, 0.23, 0.0, 0.0]])
    sel = select(P.RULE_RANDOMIZED, a_mask=mask.expand(20000, 5), arms=arms.expand(20000, 5), eps_floor=0.0,
                 generator=gen)
    counts = torch.bincount(sel.index, minlength=5)
    assert counts[[1, 3, 4]].sum() == 0 and abs(int(counts[0]) - 10000) < 300
    assert bool((sel.propensity == 0.5).all())
    with pytest.raises(ValueError):
        selection_probs(P.RULE_RANDOMIZED, a_mask=mask, arms=torch.tensor([[True, False, False, False, True]]))
    with pytest.raises(ValueError):
        selection_probs(P.RULE_RANDOMIZED, a_mask=mask, arms=arms, arm_probs=torch.tensor([[0.5, 0.5, 0, 0, 0]]))
    with pytest.raises(ValueError):
        selection_probs(P.RULE_RANDOMIZED, a_mask=mask, arms=torch.zeros(1, 5, dtype=torch.bool))


# ----------------------------------------------------------------------------------------------------------------
# dependence on the latent
# ----------------------------------------------------------------------------------------------------------------
def test_permuting_belief_changes_the_action_distribution(gen):
    """The candidates are identical for every decision; only the belief differs. A policy that ignored b would
    give the same distribution for every row (KL = 0)."""
    n_arm, bsz = 5, 64
    e_a = torch.randn(n_arm, DM, generator=gen)
    z_const = e_a[None].expand(bsz, n_arm, DM).contiguous()                # z without any belief information
    mask = torch.ones(bsz, n_arm, dtype=torch.bool)
    b = torch.randn(bsz, DM, generator=gen)
    pol = _policy()
    at_init = latent_dependence(pol, z_const, mask, b)
    assert at_init["kl"] > 1e-5                                           # the direct b path is alive at init
    probs = pol(z_const, mask, b).probs
    assert float((probs - probs.mean(0, keepdim=True)).detach().abs().max()) > 1e-3   # not a constant function of candidates

    # train on a task whose best action depends on the belief: arm i is best when b is most aligned with u_i
    u = torch.randn(n_arm, DM, generator=gen)

    def best(bb):
        return (bb @ u.t()).argmax(-1)

    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    for _ in range(400):
        bb = torch.randn(128, DM, generator=gen)
        out = pol(e_a[None].expand(128, n_arm, DM), torch.ones(128, n_arm, dtype=torch.bool), bb)
        loss = -out.log_prob(best(bb)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = pol(z_const, mask, b)
        acc = float((out.probs.argmax(-1) == best(b)).float().mean())
        perm = torch.roll(torch.arange(bsz), 1)
        acc_perm = float((pol(z_const, mask, b[perm]).probs.argmax(-1) == best(b)).float().mean())
    dep = latent_dependence(pol, z_const, mask, b)
    assert acc > 0.8 and acc_perm < 0.45
    assert dep["kl"] > 0.5 and dep["tv"] > 0.4 and dep["argmax_changed"] > 0.5
    # with z recomputed from the permuted belief (the model's decision_latent) dependence is at least as visible
    dep_z = latent_dependence(pol, z_const, mask, b, z_fn=lambda bb: e_a[None] + 0.3 * bb[:, None])
    assert dep_z["kl"] > 0.1


# ----------------------------------------------------------------------------------------------------------------
# functional forward and differentiation through an update
# ----------------------------------------------------------------------------------------------------------------
def test_functional_forward_matches_module_and_supports_meta_gradients(gen):
    pol = _policy(hidden=24).double()
    z, mask, b = (t.double() if t.dtype.is_floating_point else t for t in _batch(gen, b=6, n=4))
    params = pol.named_param_dict()
    np.testing.assert_allclose(pol.functional_forward(params, z, mask, b).log_probs.detach(),
                               pol(z, mask, b).log_probs.detach(), rtol=1e-12)
    # replaced parameters are really used and the module's own weights stay untouched
    shifted = {k: v + 0.1 for k, v in params.items()}
    before = {k: v.detach().clone() for k, v in params.items()}
    assert not torch.allclose(pol.functional_forward(shifted, z, mask, b).log_probs, pol(z, mask, b).log_probs)
    assert all(torch.equal(before[k], v) for k, v in pol.named_parameters())
    # the actor subset is exactly what log pi depends on
    lp = pol(z, mask, b).log_prob(torch.zeros(6, dtype=torch.long)).sum()
    used = {k for k, v in params.items() if torch.autograd.grad(lp, v, retain_graph=True, allow_unused=True)[0] is not None}
    assert used == set(pol.actor_param_dict())

    # inner step on a parametric per-decision weight w(psi); outer objective on other decisions; d outer / d psi
    feat = torch.randn(6, 3, generator=gen, dtype=torch.float64)
    a_sup = torch.tensor([0, 1, 2, 3, 0, 1])
    zq, mq, bq = (t.double() if t.dtype.is_floating_point else t for t in _batch(gen, b=6, n=4))
    a_q = torch.tensor([1, 0, 3, 2, 1, 0])
    alpha = 0.2

    def outer(psi, detach_inner=False):
        w = torch.tanh(feat @ psi)
        if detach_inner:
            w = w.detach()
        inner = -(pol.functional_forward(params, z, mask, b).log_prob(a_sup) * w).mean()
        actor = pol.actor_param_dict()
        grads = torch.autograd.grad(inner, list(actor.values()), create_graph=True)
        new = {k: v - alpha * g for (k, v), g in zip(actor.items(), grads)}
        return pol.functional_forward({**params, **new}, zq, mq, bq).log_prob(a_q).mean()

    psi = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64, requires_grad=True)
    (g_auto,) = torch.autograd.grad(outer(psi), psi)
    g_fd = torch.zeros(3, dtype=torch.float64)
    h = 1e-5
    for i in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[i] = h
        g_fd[i] = ((outer(psi.detach() + d) - outer(psi.detach() - d)) / (2 * h)).detach()
    assert float(g_auto.norm()) > 1e-6
    np.testing.assert_allclose(g_auto, g_fd, rtol=1e-5, atol=1e-10)
    # cutting the inner signal removes the dependence entirely
    out_cut = outer(psi, detach_inner=True)
    assert torch.autograd.grad(out_cut, psi, allow_unused=True)[0] is None
