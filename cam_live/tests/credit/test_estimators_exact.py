"""Exact finite-action reference estimators against enumerated / dynamic-programming gradients (RC-02, RC-04)."""
import json

import pytest
import torch

from cam_live.credit import estimators as E
from cam_live.credit.longdelay import LongDelay
from cam_live.diagnostics import tabular as tb

DT = torch.float64


def _theta(mdp, seed=0, scale=0.5):
    return scale * torch.randn(mdp.S, mdp.A, generator=torch.Generator().manual_seed(seed), dtype=DT)


def _all(mdp, theta, batch):
    Q, V = tb.q_values(mdp, theta)
    w = tb.contribution_coefficients(mdp, theta)
    rr = E.rudder_exact_redistribution(mdp, theta, batch)
    rr_no_offset = E.rudder_exact_redistribution(mdp, theta, batch, include_offset=False)
    return {"reinforce": E.reinforce_grad(theta, batch),
            "reinforce_baseline": E.reinforce_grad(theta, batch, V),
            "all_actions": E.all_actions_grad(theta, batch, Q),
            "cocoa": E.cocoa_grad(theta, batch, w),
            "rudder_return_to_go": E.rudder_grad(theta, batch, rr),
            "rudder_immediate_exact": E.rudder_grad(theta, batch, rr_no_offset, immediate_only=True)}, rr


def test_exact_expectations_on_enumerated_noisy_key_to_door():
    """Every trajectory is enumerated, so the estimator means are exact: bias at round-off, not at Monte-Carlo noise."""
    mdp = tb.noisy_key_to_door(sigma0=0.6)
    theta = _theta(mdp)
    exact = tb.exact_gradient(mdp, theta)
    batch, probs = tb.enumerate_batch(mdp, theta)
    assert abs(float(probs.sum()) - 1.0) < 1e-12 and batch.N == 128
    grads, rr = _all(mdp, theta, batch)
    stats = {k: tb.exact_stats(g, probs, exact) for k, g in grads.items()}
    for k, s in stats.items():
        assert s["rel_bias"] < 1e-20, (k, s)
    # the redistribution conserves the realised return on every trajectory
    assert float((rr.sum(1) - batch.r.sum(1)).abs().max()) < 1e-12
    # variance ordering
    assert stats["all_actions"]["rel_var"] < stats["cocoa"]["rel_var"] < stats["reinforce"]["rel_var"]
    assert stats["rudder_immediate_exact"]["rel_var"] < 0.5 * stats["reinforce"]["rel_var"]
    # return-conditioned hindsight is biased (the reason the outcome, not the return, is the hindsight variable)
    hca = E.hca_return_grad(theta, batch, E.return_conditioned_hindsight(batch, probs))
    hca_bias = tb.exact_stats(hca, probs, exact)["rel_bias"]
    assert hca_bias > 1e-2                                       # 18 orders of magnitude above round-off
    stats["hca_return"] = {"rel_bias": hca_bias}
    print("EXACT_NKD", json.dumps({k: {m: float(f"{v:.4g}") for m, v in s.items()} for k, s in stats.items()}))


def test_return_conditioned_hindsight_bias_closed_form():
    """COCOA App. L bandit: rewards (1, -2), pi = (2/3, 1/3). Return-conditioned HCA assigns advantages (1/3, -4/3);
    the true advantages are (1, -2). COCOA on the same problem is exact."""
    P = torch.zeros(1, 3, 2, 3, dtype=DT)
    P[0, 0, 0, 1] = P[0, 0, 1, 2] = 1.0
    P[0, 1, :, 1] = P[0, 2, :, 2] = 1.0
    out = torch.zeros(1, 3, 2, 3, dtype=torch.long)
    out[0, 0, 1, 2] = 1
    out[0, 1:, :, :] = 2
    mdp = tb.TabularMDP(P, out, torch.tensor([1.0, -2.0, 0.0], dtype=DT), torch.tensor([1.0, 0.0, 0.0], dtype=DT))
    mdp.validate()
    theta = torch.zeros(3, 2, dtype=DT)
    theta[0] = torch.tensor([2.0, 1.0], dtype=DT).log()
    pi = tb.policy(theta)[0]
    assert pi.tolist() == pytest.approx([2 / 3, 1 / 3])
    batch, probs = tb.enumerate_batch(mdp, theta)
    exact = tb.exact_gradient(mdp, theta)
    eye = torch.eye(2, dtype=DT)
    from_adv = lambda adv: sum(pi[a] * (eye[a] - pi) * adv[a] for a in range(2))
    assert torch.allclose(exact[0], from_adv([1.0, -2.0]), atol=1e-12)
    hca = E.hca_return_grad(theta, batch, E.return_conditioned_hindsight(batch, probs))
    assert torch.allclose((probs[:, None, None] * hca).sum(0)[0], from_adv([1 / 3, -4 / 3]), atol=1e-12)
    assert tb.exact_stats(hca, probs, exact)["rel_bias"] == pytest.approx(16 / 81, abs=1e-9)
    assert tb.exact_stats(E.reinforce_grad(theta, batch), probs, exact)["rel_bias"] < 1e-25


def test_redistribution_as_immediate_reward_starves_the_key_decision_but_return_to_go_does_not():
    """The pitfall: a return-equivalent decomposition that explains the return only at the end gives the key decision
    zero credit when the redistributed reward is used as an immediate reward; its return-to-go stays unbiased."""
    mdp = tb.noisy_key_to_door(sigma0=0.6)
    theta = _theta(mdp)
    exact = tb.exact_gradient(mdp, theta)
    batch, probs = tb.enumerate_batch(mdp, theta)
    rr = E.explaining_away_redistribution(batch)
    assert float((rr.sum(1) - batch.r.sum(1)).abs().max()) < 1e-12
    key_rows = [0]                                               # the t = 0 state (the key decision)
    imm = tb.exact_stats(E.rudder_grad(theta, batch, rr, immediate_only=True), probs, exact, block=key_rows)
    rtg = tb.exact_stats(E.rudder_grad(theta, batch, rr), probs, exact, block=key_rows)
    assert imm["grad_norm2"] > 1e-3
    assert abs(imm["rel_bias"] - 1.0) < 1e-12                    # the estimate of the key gradient is exactly zero
    assert rtg["rel_bias"] < 1e-20
    full = tb.exact_stats(E.rudder_grad(theta, batch, rr), probs, exact)
    assert full["rel_bias"] < 1e-20


def test_contribution_coefficients_are_hindsight_over_policy_minus_one():
    for mdp in (tb.key_to_door(n_distract=3), tb.random_tree(depth=3, seed=1)):
        theta = _theta(mdp, seed=2)
        pi = tb.policy(theta)
        w = tb.contribution_coefficients(mdp, theta)                     # [T, S, A, U]
        h = tb.hindsight_action_probs(mdp, theta)                        # [T, S, U, A]
        _, Dbar = tb.future_outcome_counts(mdp, theta)
        ok = (Dbar > 1e-12)[:, :, None, :].expand_as(w)
        ratio = h.permute(0, 1, 3, 2) / pi[None, :, :, None] - 1.0
        assert float((w - ratio)[ok].abs().max()) < 1e-10
        assert float((h.sum(-1) - 1.0)[Dbar > 1e-12].abs().max()) < 1e-10
        # sum_a pi(a|s) w(s, a, u) = 0: coefficients redistribute, they do not create credit
        assert float(torch.einsum("sa,tsau->tsu", pi, w).abs().max()) < 1e-10
    # key-to-door, closed form at the key decision: w(key, door) = 1/pi(key) - 1, w(no key, door) = -1, apples 0
    # (the distractor policy is made independent of the key so that apples really do not depend on it)
    mdp = tb.key_to_door(n_distract=3)
    theta = torch.zeros(mdp.S, mdp.A, dtype=DT)
    theta[0] = torch.tensor([0.4, -0.3], dtype=DT)
    pi0 = tb.policy(theta)[0]
    w = tb.contribution_coefficients(mdp, theta)
    assert float(w[0, 0, 0, 3]) == pytest.approx(1 / float(pi0[0]) - 1) and float(w[0, 0, 1, 3]) == pytest.approx(-1.0)
    assert float(w[0, 0, :, 2].abs().max()) < 1e-12
    # and the hindsight probabilities agree with frequencies in sampled trajectories
    b = tb.sample(mdp, theta, 100_000, torch.Generator().manual_seed(0))
    door = b.u[:, -1] == 3
    emp = float((b.a[door, 0] == 0).double().mean())
    assert emp == pytest.approx(float(tb.hindsight_action_probs(mdp, theta)[0, 0, 3, 0]), abs=1e-9)   # only key opens


@pytest.mark.parametrize("name", ["random_tree", "key_to_door"])
def test_monte_carlo_unbiasedness_by_z_score_and_relative_error(name):
    mdp = tb.random_tree(depth=3, seed=3, reward_noise=0.3) if name == "random_tree" else \
        tb.key_to_door(n_distract=4, reward_noise=0.2)
    theta = _theta(mdp, seed=1)
    exact = tb.exact_gradient(mdp, theta)
    batch = tb.sample(mdp, theta, 120_000, torch.Generator().manual_seed(11))
    grads, rr = _all(mdp, theta, batch)
    assert float((rr.sum(1) - batch.r.sum(1)).abs().max()) < 1e-9
    out = {}
    for k, g in grads.items():
        u = E.unbiasedness(g, exact)
        out[k] = u
        assert u["max_abs_z"] < 4.5, (k, u)
        assert 0.5 < u["rms_z"] < 1.6, (k, u)                       # z-scores look like N(0, 1), not all zero
        assert u["rel_error"] < 0.08, (k, u)
        assert u["max_err_zero_variance_components"] < 1e-12
    print("MC_UNBIASED", name, json.dumps({k: {m: float(f"{v:.3g}") for m, v in u.items()} for k, u in out.items()}))


def test_variance_comparison_on_key_to_door():
    """Key decision (t = 0 logits): COCOA and the exact RUDDER redistribution remove the distractor variance that
    REINFORCE carries; the all-actions exact critic is the lower bound."""
    mdp = tb.key_to_door(n_distract=8, reward_noise=0.3)
    theta = _theta(mdp, seed=4)
    exact = tb.exact_gradient(mdp, theta)
    batch = tb.sample(mdp, theta, 60_000, torch.Generator().manual_seed(5))
    grads, _ = _all(mdp, theta, batch)
    var = {k: float(g[:, 0].var(0, unbiased=True).sum()) for k, g in grads.items()}        # row 0 = key state
    snr = {k: float((exact[0] ** 2).sum()) / v if v > 0 else float("inf") for k, v in var.items()}
    assert var["cocoa"] < 0.2 * var["reinforce"], var
    assert var["rudder_immediate_exact"] < 0.2 * var["reinforce"], var
    assert var["all_actions"] < 1e-20
    assert abs(var["rudder_return_to_go"] / var["reinforce"] - 1.0) < 0.15      # unbiased for any g, no variance gain
    print("KTD_VARIANCE", json.dumps({"var": var, "snr": snr}))


def test_long_delay_key_decision_snr():
    """Delays of 16 ... 1024 decisions: REINFORCE's signal-to-noise falls with the delay, COCOA's and the exact
    RUDDER decomposition's do not; all are unbiased (bias within 4 standard errors)."""
    table = {}
    for T in (16, 64, 256, 1024):
        st = LongDelay(T=T).stats(n=20_000, seed=T)
        table[T] = {k: v["snr"] for k, v in st.items()}
        for k, v in st.items():
            assert v["bias_norm"] < 4.0 * v["se"] + 1e-12, (T, k, v)
    assert table[1024]["reinforce"] < table[16]["reinforce"] / 20
    for k in ("cocoa", "rudder"):
        assert 0.7 < table[1024][k] / table[16][k] < 1.4
        assert table[1024][k] > 500 * table[1024]["reinforce"]
    # an imperfect learned coefficient biases COCOA by a known amount (bias measurement, not hidden)
    biased = LongDelay(T=64).stats(n=40_000, seed=1, w_door_key=0.5)["cocoa"]
    assert biased["bias_norm"] > 10 * biased["se"]
    print("LONG_DELAY_SNR", json.dumps({str(T): {k: float(f"{min(v, 1e9):.4g}") for k, v in d.items()} for T, d in table.items()}))
