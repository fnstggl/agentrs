"""Exact tabular meta-gradient reference (spec 7.4, 7.7, 16.5; MR-03, MR-04, MR-05, MR-07). All float64."""
import pytest
import torch

from cam_live.diagnostics import tabular as X
from cam_live.reward import exact as E

DT = X.DT


def _door():
    mdp = X.key_to_door(n_distract=2, key_cost=0.3, apple_reward=0.2)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=DT)
    psi = 0.3 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=DT, generator=torch.Generator().manual_seed(0))
    return mdp, th0, psi


def _tree(branching=1, seed=3):
    mdp = X.random_tree(depth=2, branching=branching, n_actions=2, seed=seed)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=DT)
    psi = 0.5 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=DT, generator=torch.Generator().manual_seed(1))
    return mdp, th0, psi


def test_extrinsic_return_matches_reference_dynamic_programme():
    mdp, th0, _ = _door()
    th = torch.randn(mdp.S, mdp.A, dtype=DT, generator=torch.Generator().manual_seed(2))
    assert float(E.extrinsic_return(mdp, th)) == pytest.approx(float(X.expected_return(mdp, th)), abs=1e-13)
    both = E.extrinsic_return(mdp, torch.stack([th0, th]))
    assert both.shape == (2,) and float(both[1]) == pytest.approx(float(X.expected_return(mdp, th)), abs=1e-13)


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_exact_meta_gradient_equals_finite_differences(K):
    mdp, th0, psi = _door()
    g, J = E.exact_meta_gradient(mdp, psi, th0, K, 1.0)
    fd = E.finite_difference_gradient(mdp, psi, th0, K, 1.0)
    assert float(g.norm()) > 1e-2
    assert float((g - fd).norm() / g.norm()) < 1e-7
    assert float((g - fd).abs().max()) < 1e-8
    # unreachable rows of the table cannot matter
    unreachable = ~E.reachable_mask(mdp)[:, :, None].expand_as(g)
    assert float(g[unreachable].abs().max()) == 0.0


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_cutting_the_pathway_gives_exactly_zero(K):
    mdp, th0, psi = _door()
    g_detached, _ = E.exact_meta_gradient(mdp, psi, th0, K, 1.0, detach_reward=True)
    g_stopped, _ = E.exact_meta_gradient(mdp, psi, th0, K, 1.0, stop_unroll=True)
    assert float(g_detached.abs().max()) == 0.0
    assert float(g_stopped.abs().max()) == 0.0
    assert float(E.exact_meta_gradient(mdp, psi, th0, K, 1.0)[0].abs().max()) > 0.0


def test_bounded_and_feature_rewards_match_finite_differences():
    mdp, th0, psi = _door()
    g, _ = E.exact_meta_gradient(mdp, 3 * psi, th0, 4, 1.0, bound=1.0)
    fd = E.finite_difference_gradient(mdp, 3 * psi, th0, 4, 1.0, bound=1.0)
    assert float((g - fd).norm() / g.norm()) < 1e-7
    assert float(E.reward_table(100 * psi, bound=1.0).abs().max()) <= 1.0
    feats = torch.randn(mdp.T, mdp.S, mdp.A, 5, dtype=DT, generator=torch.Generator().manual_seed(4))
    w = torch.randn(5, dtype=DT, generator=torch.Generator().manual_seed(5))
    g, _ = E.exact_meta_gradient(mdp, w, th0, 2, 1.0, features=feats)
    fd = E.finite_difference_gradient(mdp, w, th0, 2, 1.0, features=feats)
    assert float((g - fd).norm() / g.norm()) < 1e-7


def test_enumerated_sampled_objective_gradient_matches_finite_differences():
    mdp, th0, psi = _tree(branching=2)
    g, _ = E.enumerated_sampled_gradient(mdp, psi, th0, 2, 0.7, 1)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        d = torch.randn(psi.shape, dtype=DT, generator=gen) * E.reachable_mask(mdp)[:, :, None]
        d = d / d.norm()
        eps = 1e-5
        jp = float(E.enumerated_sampled_objective(mdp, psi + eps * d, th0, 2, 0.7, 1).detach())
        jm = float(E.enumerated_sampled_objective(mdp, psi - eps * d, th0, 2, 0.7, 1).detach())
        assert (jp - jm) / (2 * eps) == pytest.approx(float((g * d).sum()), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("K,n_inner,branching", [(1, 2, 1), (2, 2, 1), (4, 2, 1), (8, 1, 1), (2, 1, 2)])
def test_sampled_estimator_is_unbiased_by_z_score_and_relative_error(K, n_inner, branching):
    """Against the exact gradient of the sampled-inner-update objective: z-scores AND relative error, never cosine
    alone. n_query = 2 keeps the leave-one-out outer baseline in play."""
    mdp, th0, psi = _tree(branching)
    exact, J = E.enumerated_sampled_gradient(mdp, psi, th0, K, 0.7, n_inner)
    gen = torch.Generator().manual_seed(100 + K)
    stats = None
    for reps in (20_000, 160_000):                                    # error must fall as the sample grows
        smp = E.sampled_meta_gradient(mdp, psi, th0, K, 0.7, n_inner, 2, gen, reps=reps, score_baseline=J)
        prev, stats = stats, E.mc_summary(smp, exact)
        assert stats["max_abs_z"] < 4.5, stats
        assert stats["noise_free_max_abs_err"] == 0.0
    assert stats["rel_err"] < 0.04, stats                             # largest sample size
    assert stats["rel_se"] < prev["rel_se"] * 0.5
    assert abs(stats["norm_ratio"] - 1.0) < 0.04


def test_omitting_inner_score_term_is_biased_and_cosine_does_not_see_it():
    mdp, th0, psi = _tree()
    exact, J = E.enumerated_sampled_gradient(mdp, psi, th0, 4, 0.7, 2)
    gen = torch.Generator().manual_seed(7)
    smp = E.sampled_meta_gradient(mdp, psi, th0, 4, 0.7, 2, 2, gen, reps=100_000, inner_score=False)
    st = E.mc_summary(smp, exact)
    assert st["cos"] > 0.99                       # looks fine by cosine ...
    assert st["max_abs_z"] > 10 and st["rel_err"] > 0.1, st   # ... and is not


def test_self_inclusive_outer_baseline_bias_is_detected():
    mdp, th0, psi = _tree()
    exact, _ = E.enumerated_sampled_gradient(mdp, psi, th0, 1, 0.7, 2)
    gen = torch.Generator().manual_seed(8)
    smp = E.sampled_meta_gradient(mdp, psi, th0, 1, 0.7, 2, 2, gen, reps=100_000, outer_baseline="mean")
    st = E.mc_summary(smp, exact)
    assert st["cos"] > 0.99 and abs(st["norm_ratio"] - 0.5) < 0.05     # scaled by 1 - 1/n_query
    assert st["max_abs_z"] > 10 and st["rel_err"] > 0.4


def test_sampled_inner_objective_approaches_exact_update_objective():
    mdp, th0, psi = _tree()
    exact, _ = E.exact_meta_gradient(mdp, psi, th0, 2, 0.7)
    gaps = [float((E.enumerated_sampled_gradient(mdp, psi, th0, 2, 0.7, n)[0] - exact).norm() / exact.norm())
            for n in (2, 4, 8, 16)]
    assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] < 0.03 and gaps[-1] < gaps[0] / 4


def test_truncation_bias_is_measured():
    mdp = X.key_to_door(n_distract=3, key_cost=0.3, apple_reward=0.2)
    th0 = torch.zeros(mdp.S, mdp.A, dtype=DT)
    psi = 0.3 * torch.randn(mdp.T, mdp.S, mdp.A, dtype=DT, generator=torch.Generator().manual_seed(0))
    for objective in ("sum", "final"):
        tb = E.truncation_bias(mdp, psi, th0, 8, 0.5, Ks=(1, 2, 4, 8), objective=objective)
        assert tb[8]["rel_err"] < 1e-12                           # K = L is the full gradient
        assert tb[1]["rel_err"] > tb[2]["rel_err"] > tb[4]["rel_err"] > 0.1
        assert tb[1]["rel_err"] > 0.5 and tb[1]["cos"] > 0.99     # large magnitude bias, invisible to cosine
    # the truncated gradient is itself exact for its own window: finite differences of J_ext(theta_L) with the
    # earlier updates held fixed
    g_trunc, _ = E.lifetime_meta_gradient(mdp, psi, th0, 3, 0.5, truncate=1, objective="final")
    th2 = E.unroll_exact(mdp, th0, psi, 2, 0.5)[-1].detach()
    g_win, _ = E.exact_meta_gradient(mdp, psi, th2, 1, 0.5)
    assert torch.allclose(g_trunc, g_win, atol=1e-14)


def _family():
    train = [X.key_to_door(n_distract=3, key_cost=kc, apple_reward=ar) for kc in (0.1, 0.3, 0.5)
             for ar in (0.1, 0.3)]
    held_out = X.key_to_door(n_distract=3, key_cost=0.4, apple_reward=0.2)
    return train, held_out


def test_meta_trained_reward_values_delayed_payoff_action_and_speeds_up_learning():
    train, held = _family()
    K, alpha = 3, 1.0
    out = E.meta_train(train, K, alpha, 150, lr=0.1, bound=1.0, theta_init_scale=0.3, seed=0)
    r_learn = E.reward_table(out["psi"], bound=1.0)
    assert float(r_learn.abs().max()) <= 1.0
    # the key pickup at t=0 pays only at the door, costs 0.1-0.5 immediately; the learned reward values it
    assert float(r_learn[0, 0, 0]) > 0.5 > 0.0 > float(r_learn[0, 0, 1])
    # held-out MDP (key cost / distractor reward never seen in meta-training), fresh learner
    th0 = torch.zeros(held.S, held.A, dtype=DT)
    r_ext = E.extrinsic_reward_table(held)
    ext = E.learning_curve_exact(held, r_ext, th0, K, alpha)
    lrn = E.learning_curve_exact(held, E.scale_matched(held, r_learn, r_ext), th0, K, alpha)
    assert lrn[-1] > ext[-1] + 0.15 and sum(lrn) > sum(ext)
    # sampled learners at the same step size and the same largest reward magnitude
    gen = torch.Generator().manual_seed(0)
    me, se = E.learning_curve_sampled(held, th0, K, alpha, 4, 2000, gen)
    ml, sl = E.learning_curve_sampled(held, th0, K, alpha, 4, 2000, gen,
                                      r_table=E.scale_matched(held, r_learn, r_ext))
    assert ml[-1] - me[-1] > 10 * (se[-1] + sl[-1])


def test_meta_training_with_the_sampled_estimator_also_learns_the_key():
    train, _ = _family()
    out = E.meta_train(train, 2, 1.0, 300, lr=0.05, bound=1.0, theta_init_scale=0.3, seed=1, estimator="sampled",
                       n_inner=32, n_query=64)
    r = E.reward_table(out["psi"], bound=1.0)
    assert float(r[0, 0, 0] - r[0, 0, 1]) > 0.5
    assert sum(out["curve"][-30:]) / 30 > sum(out["curve"][:30]) / 30 + 0.1
