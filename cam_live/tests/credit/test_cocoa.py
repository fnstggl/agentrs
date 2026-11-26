"""HindsightModel / contribution coefficients against exact tabular coefficients; support refusal; the three
separate kinds of statement (RC-04, RC-05)."""
import json
import math

import pytest
import torch

from cam_live.credit import estimators as E
from cam_live.credit.cocoa import (R_ESS, R_INVALID, R_MISSING, R_NEAR_ZERO, R_OPEN, SUPPORTED, UNSUPPORTED,
                                   HindsightModel, candidate_ess, check_support, coefficients_table,
                                   effective_sample_size)
from cam_live.credit.simple import arm_difference
from cam_live.diagnostics import tabular as tb

DT = torch.float64


def _theta(mdp, seed=0, scale=0.5):
    return scale * torch.randn(mdp.S, mdp.A, generator=torch.Generator().manual_seed(seed), dtype=DT)


def _pairs(batch):
    """All legitimate retrospective pairs (decision t, outcome at t + k, k >= 1) of sampled trajectories."""
    N, T = batch.s.shape
    tt, kk = torch.triu_indices(T, T, offset=1)
    s = batch.s[:, tt].reshape(-1)
    a = batch.a[:, tt].reshape(-1)
    u = batch.u[:, kk].reshape(-1)
    el = (kk - tt)[None, :].expand(N, -1).reshape(-1)
    return s, a, u, el


def _fit(mdp, theta, n_traj=6000, steps=700, n_elapsed=0, seed=0, lr=3e-3, n_members=4):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    batch = tb.sample(mdp, theta, n_traj, gen)
    s, a, u, el = _pairs(batch)
    logpi = torch.log_softmax(theta, -1).float()
    eyeS, eyeU, eyeA = torch.eye(mdp.S), torch.eye(mdp.U), torch.eye(mdp.A)
    model = HindsightModel(mdp.S, mdp.U, mdp.A, d_hidden=64, d_key=32, n_members=n_members, n_elapsed=n_elapsed,
                           elapsed_units=(1.0,), seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mask = torch.ones(1024, mdp.A, dtype=torch.bool)
    for _ in range(steps):
        i = torch.randint(0, s.shape[0], (1024,), generator=gen)
        loss, info = model.loss(eyeS[s[i]], eyeU[u[i]], eyeA[None].expand(1024, -1, -1), mask, a[i], logpi[s[i]],
                                el[i, None].float() if n_elapsed else None, generator=gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, batch, info


def test_untrained_model_equals_the_policy_and_claims_no_credit():
    torch.manual_seed(0)
    m = HindsightModel(6, 3, 5, n_members=3)
    b, u, cand = torch.randn(4, 6), torch.randn(4, 3), torch.randn(4, 7, 5)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[1, 4:] = False
    logp = torch.log_softmax(torch.randn(4, 7).masked_fill(~mask, float("-inf")), -1)
    h = m(b, u, cand, mask, logp, torch.tensor([[60.0, 3.0]] * 4))
    assert torch.allclose(h, logp.exp(), atol=1e-6) and float(h.detach()[1, 4:].abs().max()) == 0.0
    out = m.contribution(b, u, cand, mask, logp, torch.tensor([[60.0, 3.0]] * 4), outcome_value=torch.ones(4) * 5.0)
    assert out.status == [SUPPORTED] * 4
    assert float(out.w[mask].abs().max()) < 1e-5 and float(out.counterfactual_estimate[mask].abs().max()) < 1e-4
    assert torch.isnan(out.w[1, 4:]).all() and out.experimental_effect == {}


def test_learned_coefficients_match_exact_on_key_to_door():
    mdp = tb.key_to_door(n_distract=3)
    theta = _theta(mdp, seed=3)
    model, batch, info = _fit(mdp, theta, n_traj=12000)
    assert info["ce"] < info["ce_policy_prior"]                 # hindsight beats the prior it started from
    pi = tb.policy(theta)
    w_exact = tb.contribution_coefficients(mdp, theta)          # [T, S, A, U]
    _, Dbar = tb.future_outcome_counts(mdp, theta)
    occ = tb.state_occupancy(mdp, theta)                        # [T, S]
    W = coefficients_table(model, torch.eye(mdp.S), torch.eye(mdp.U), torch.eye(mdp.A), torch.log(pi).float())
    weight = occ[:, :, None] * Dbar                             # probability mass of each (t, s, u) pair
    err = ((W[None] - w_exact).abs() * pi[None, :, :, None]).sum(2)
    mean_err = float((err * weight).sum() / weight.sum())
    assert mean_err < 0.05, mean_err
    # the key decision: w(key, door) = 1 / pi(key) - 1 and w(no key, door) = -1
    assert float(W[0, 0, 3]) == pytest.approx(1 / float(pi[0, 0]) - 1, abs=0.08)
    assert float(W[0, 1, 3]) == pytest.approx(-1.0, abs=0.05)
    print("COCOA_KTD", json.dumps({"weighted_abs_err": mean_err, "w_key_door": float(W[0, 0, 3]),
                                   "w_key_door_exact": 1 / float(pi[0, 0]) - 1, "w_nokey_door": float(W[0, 1, 3])}))


def test_learned_coefficients_give_a_nearly_unbiased_gradient_and_the_bias_is_measured():
    """Enumerated NoisyKeyDoor: the exact expectation of COCOA with learned coefficients, against the true gradient.
    With w = 0 (nothing learned) only immediate rewards are credited and the key decision is missed."""
    mdp = tb.noisy_key_to_door(sigma0=0.6)
    theta = _theta(mdp)
    model, _, _ = _fit(mdp, theta, n_traj=8000, steps=900)
    pi = tb.policy(theta)
    W = coefficients_table(model, torch.eye(mdp.S), torch.eye(mdp.U), torch.eye(mdp.A), torch.log(pi).float())
    w_learned = W[None].expand(mdp.T, -1, -1, -1)
    exact = tb.exact_gradient(mdp, theta)
    batch, probs = tb.enumerate_batch(mdp, theta)
    key = [0]
    learned = tb.exact_stats(E.cocoa_grad(theta, batch, w_learned), probs, exact)
    learned_key = tb.exact_stats(E.cocoa_grad(theta, batch, w_learned), probs, exact, block=key)
    nothing = tb.exact_stats(E.cocoa_grad(theta, batch, torch.zeros_like(w_learned)), probs, exact, block=key)
    truth = tb.exact_stats(E.cocoa_grad(theta, batch, tb.contribution_coefficients(mdp, theta)), probs, exact)
    assert truth["rel_bias"] < 1e-20
    assert nothing["rel_bias"] > 0.9                             # the delayed treasure is not credited at all
    assert learned["rel_bias"] < 0.02 and learned_key["rel_bias"] < 0.02, (learned, learned_key)
    print("COCOA_LEARNED_BIAS", json.dumps({"rel_bias": learned["rel_bias"], "key_rel_bias": learned_key["rel_bias"],
                                            "key_rel_bias_w0": nothing["rel_bias"], "rel_var": learned["rel_var"],
                                            "rel_var_exact_w": truth["rel_var"]}))


def _per_delay_hindsight(mdp, theta):
    """Exact time-indexed hindsight h[t, k, s, u, a] = p(A_t = a | S_t = s, U_{t+k} = u) and the pair mass."""
    pi = tb.policy(theta)
    T, S, A, U = mdp.T, mdp.S, mdp.A, mdp.U
    onehot = torch.nn.functional.one_hot(mdp.outcome, U).to(DT)
    now = torch.einsum("sa,tsap,tsapu->tsu", pi, mdp.P, onehot)          # outcome distribution of a step from s
    Tpi = torch.einsum("sa,tsap->tsp", pi, mdp.P)
    occ = tb.state_occupancy(mdp, theta)
    h = torch.zeros(T, T, S, U, A, dtype=DT)
    mass = torch.zeros(T, T, S, U, dtype=DT)
    for t in range(T):
        M = mdp.P[t].clone()                                             # [S, A, S'] state after the step at t
        for j in range(t + 1, T):
            D = torch.einsum("sap,pu->sau", M, now[j])                   # P(U_j = u | s, a)
            den = torch.einsum("sa,sau->su", pi, D)
            h[t, j - t] = (pi[:, :, None] * D / den.clamp_min(1e-300)[:, None, :]).permute(0, 2, 1)
            mass[t, j - t] = occ[t][:, None] * den
            M = torch.einsum("sap,pq->saq", M, Tpi[j])
    return h, mass


def test_elapsed_time_input_gives_time_indexed_hindsight():
    """Outcome ids are reused at several delays in a random tree. With the elapsed-time input the model matches the
    exact per-delay hindsight; without it the model can only learn the delay-averaged one."""
    mdp = tb.random_tree(depth=3, branching=2, n_actions=2, n_outcomes=3, seed=5)
    theta = _theta(mdp, seed=6, scale=0.8)
    h_exact, mass = _per_delay_hindsight(mdp, theta)
    pi = tb.policy(theta)
    eyeS, eyeU, eyeA = torch.eye(mdp.S), torch.eye(mdp.U), torch.eye(mdp.A)

    def error(model, with_elapsed):
        tot = num = 0.0
        for k in (1, 2):
            S, U = mdp.S, mdp.U
            b = eyeS[:, None, :].expand(S, U, -1).reshape(S * U, -1)
            u = eyeU[None].expand(S, U, -1).reshape(S * U, -1)
            lp = torch.log(pi).float()[:, None, :].expand(S, U, -1).reshape(S * U, -1)
            el = torch.full((S * U, 1), float(k)) if with_elapsed else None
            h = model(b, u, eyeA[None].expand(S * U, -1, -1), torch.ones(S * U, mdp.A, dtype=torch.bool), lp, el)
            h = h.view(S, U, mdp.A).double()
            for t in range(mdp.T - k):
                e = (h - h_exact[t, k]).abs().sum(-1) / 2          # total variation per (s, u)
                tot += float((e * mass[t, k]).sum())
                num += float(mass[t, k].sum())
        return tot / num

    timed, _, _ = _fit(mdp, theta, n_traj=12000, steps=600, n_elapsed=1, seed=1, n_members=2)
    flat, _, _ = _fit(mdp, theta, n_traj=12000, steps=600, n_elapsed=0, seed=1, n_members=2)
    with torch.no_grad():
        timed_err, flat_err = error(timed, True), error(flat, False)
    assert timed_err < 0.03 and flat_err > 2 * timed_err, (timed_err, flat_err)
    print("COCOA_ELAPSED", json.dumps({"tv_with_elapsed": timed_err, "tv_without_elapsed": flat_err}))


def test_support_check_refuses_instead_of_returning_a_number():
    torch.manual_seed(0)
    m = HindsightModel(4, 2, 3, n_members=2)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)
    B, N = 5, 4
    b, u, cand = torch.randn(B, 4), torch.randn(B, 2), torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1],
                          [0.9995, 0.0003, 0.0001, 0.0001],        # candidates with near-zero logged probability
                          [0.4, 0.3, 0.2, 0.1],                    # open-ended, proposals not tracked
                          [0.2, 0.2, 0.2, 0.2],                    # does not sum to one: not the logged distribution
                          [0.25, 0.25, 0.25, 0.25]])
    out = m.contribution(b, u, cand, mask, probs.log(), outcome_value=torch.ones(B),
                         open_ended=[False, False, True, False, True], tracked_proposals=[True, True, False, True, True])
    assert out.status == [SUPPORTED, UNSUPPORTED, UNSUPPORTED, UNSUPPORTED, SUPPORTED]
    assert out.reasons[1] == [R_NEAR_ZERO] and out.reasons[2] == [R_OPEN] and out.reasons[3] == [R_INVALID]
    assert torch.isfinite(out.counterfactual_estimate[[0, 4]]).all() and torch.isfinite(out.w_std[[0, 4]]).all()
    for row in (1, 2, 3):
        assert torch.isnan(out.counterfactual_estimate[row]).all() and torch.isnan(out.w[row]).all()
    assert out.candidate_supported[1].tolist() == [True, False, False, False]
    # an association is still an association (flagged as such) where the logged distribution is valid
    assert torch.isfinite(out.learned_association[[0, 1, 2, 4]]).all() and torch.isnan(out.learned_association[3]).all()
    assert float(out.learned_association[0].sum()) == pytest.approx(0.0, abs=1e-6)       # h and pi both sum to 1
    # propensities not recorded at all
    none = m.contribution(b, u, cand, mask, None)
    assert none.status == [UNSUPPORTED] * B and all(r == [R_MISSING] for r in none.reasons)
    assert torch.isnan(none.counterfactual_estimate).all() and torch.isnan(none.learned_association).all()
    assert none.w is None
    # support by effective sample size
    n_eff = torch.full((B, N), 50.0)
    n_eff[4, 2] = 1.0
    ess = m.contribution(b, u, cand, mask, probs.log(), n_eff=n_eff, min_ess=10.0)
    assert ess.status[0] == SUPPORTED and ess.status[4] == UNSUPPORTED and ess.reasons[4] == [R_ESS]
    plain = m.contribution(b, u, cand, mask, probs.log())
    assert bool((ess.w_std[0] > plain.w_std[0]).all())             # support adds to the ensemble spread
    # clipping is reported
    clipped = m.contribution(b, u, cand, mask, probs.log(), clip=0.01)
    assert bool(clipped.clipped[0].any()) and float(clipped.w[0].max()) <= 0.01 + 1e-12
    rep = check_support(probs.log(), mask, min_prob=0.15)
    assert rep.status[0] == UNSUPPORTED and rep.candidate_ok[0].tolist() == [True, True, True, False]


def test_three_kinds_of_statement_stay_separate():
    torch.manual_seed(0)
    m = HindsightModel(4, 2, 3, n_members=2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    logp = torch.log(torch.tensor([[0.5, 0.3, 0.2]] * 2))
    out = m.contribution(torch.randn(2, 4), torch.randn(2, 2), torch.randn(2, 3, 3), mask, logp)
    assert out.experimental_effect == {}                           # the model never fills the experimental field
    observational = arm_difference([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], randomized=False)
    with pytest.raises(ValueError):
        out.attach_experiment(0, observational)
    randomized = arm_difference([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], randomized=True, experiment_id="x1")
    out.attach_experiment(0, randomized)
    assert set(out.experimental_effect) == {0} and out.experimental_effect[0].diff == pytest.approx(1.5)
    assert out.learned_association is not out.counterfactual_estimate


def test_effective_sample_size_per_candidate():
    assert effective_sample_size(torch.ones(10)) == pytest.approx(10.0)
    assert effective_sample_size(torch.tensor([1.0, 1.0, 100.0])) < 1.1
    keys = [["a", "b"], ["a", "b"], ["a", "c"], ["a", "b"]]
    ess = candidate_ess(keys, chosen=[0, 0, 1, 0], probs=[[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.01, 0.99]])
    assert ess["b"] == 0.0 and ess["c"] == pytest.approx(1.0)
    assert ess["a"] == pytest.approx((2 + 2 + 100) ** 2 / (4 + 4 + 10000))     # one rare take dominates: ESS ~ 1
