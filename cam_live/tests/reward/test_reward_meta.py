"""The general meta-gradient engine on a small neural policy (MLP over random features) in a bandit lifetime
(spec 7.3, 7.4, 7.7, 16.5; MR-03, MR-04, MR-05, MR-06, MR-07). float64 throughout."""
import math

import pytest
import torch

from cam_live.reward import meta as M
from cam_live.reward.bandit_lifetime import BanditLifetime
from cam_live.reward.integrity import params_hash

ALPHA = 0.5


def _problem(seed=0, n_rows=12, **kw):
    P = BanditLifetime(seed=seed, **kw)
    g = torch.Generator().manual_seed(seed + 1)
    sup = [P.batch(torch.randint(P.n_ctx, (n_rows,), generator=g), torch.randint(P.n_arms, (n_rows,), generator=g))
           for _ in range(8)]
    return P, sup


def _engine(P, normalise=True, **kw):
    return M.MetaGradient(M.policy_gradient_inner_loss(P.logp_fn, P.reward_fn, normalise=normalise), P.logp_fn,
                          inner_lr=ALPHA, **kw)


def _objective(P, mg, sup, K):
    def J(psi):
        return float(P.exact_utility(mg.inner_update(P.theta(), psi, sup, K=K, create_graph=False).detached()))
    return J


def _grad(P, mg, sup, K, mode="unroll"):
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=K, mode=mode)
    return tr, mg.meta_gradient(tr, P.all_actions_query(tr.theta_K))


@pytest.mark.parametrize("K", [1, 2, 4, 8])
@pytest.mark.parametrize("normalise", [True, False])
def test_meta_gradient_matches_finite_differences(K, normalise):
    P, sup = _problem()
    mg = _engine(P, normalise)
    _, res = _grad(P, mg, sup, K)
    assert res.supported and res.estimator == M.ESTIMATOR_ON_POLICY and res.K == K
    g = M.flat(res.grads)
    fd = M.finite_difference(_objective(P, mg, sup, K), {n: v.detach() for n, v in P.psi().items()})
    assert float(g.norm()) > 1e-3
    assert float((g - fd).norm() / fd.norm()) < 1e-6          # requirement: a few percent; measured ~1e-9
    assert res.objective == pytest.approx(float(P.exact_utility(mg.inner_update(
        P.theta(), P.psi(), sup, K=K, create_graph=False).detached())), abs=1e-12)


@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_checkpointed_unroll_gives_the_same_gradient_without_keeping_the_graph(K):
    P, sup = _problem()
    mg = _engine(P)
    _, full = _grad(P, mg, sup, K)
    tr, ck = _grad(P, mg, sup, K, mode="checkpoint")
    assert not tr.has_graph and all(v.grad_fn is None for th in tr.thetas for v in th.values())
    assert ck.supported and ck.mode == "checkpoint"
    assert torch.allclose(M.flat(ck.grads), M.flat(full.grads), rtol=1e-10, atol=1e-13)


def test_first_order_approximation_is_exact_for_one_step_and_biased_beyond():
    P, sup = _problem()
    mg = _engine(P, normalise=False)
    rel = {}
    for K in (1, 2, 4, 8):
        _, full = _grad(P, mg, sup, K)
        _, fo = _grad(P, mg, sup, K, mode="first_order")
        rel[K] = float((M.flat(fo.grads) - M.flat(full.grads)).norm() / M.flat(full.grads).norm())
    assert rel[1] < 1e-12
    assert rel[2] > 1e-3 and rel[8] > rel[2]
    small = M.MetaGradient(mg.inner_loss, P.logp_fn, inner_lr=ALPHA / 10)       # the bias is O(alpha)
    _, full = _grad(P, small, sup, 4)
    _, fo = _grad(P, small, sup, 4, mode="first_order")
    assert float((M.flat(fo.grads) - M.flat(full.grads)).norm() / M.flat(full.grads).norm()) < rel[4] / 3


def test_nonzero_meta_gradient_reaches_every_reward_parameter():
    P, sup = _problem()
    for K in (1, 2, 4, 8):
        _, res = _grad(P, _engine(P), sup, K)
        assert set(res.grads) == set(P.psi())
        for name, g in res.grads.items():
            assert bool(torch.isfinite(g).all()) and bool((g != 0).all()), (K, name)


def test_meta_gradient_is_zero_when_the_pathway_is_cut():
    P, sup = _problem()
    # (a) the reward is detached inside the inner update
    cut = M.policy_gradient_inner_loss(P.logp_fn, lambda psi, b: P.reward_fn(psi, b).detach())
    mg = M.MetaGradient(cut, P.logp_fn, inner_lr=ALPHA)
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=4)
    res = mg.meta_gradient(tr, P.all_actions_query(tr.theta_K))
    assert res.supported and res.grad_norm == 0.0 and all(float(g.abs().max()) == 0.0 for g in res.grads.values())
    # (b) stop-gradient on theta_K (the unroll is removed)
    mg = _engine(P)
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=4)
    stopped = M.InnerTrace(thetas=tr.thetas[:-1] + [tr.detached()], psi=tr.psi, batches=tr.batches,
                           on_policy=tr.on_policy, inner_lr=tr.inner_lr, mode=tr.mode, has_graph=True)
    res = mg.meta_gradient(stopped, P.all_actions_query(tr.theta_K))
    assert res.grad_norm == 0.0
    # (c) an update taken without a graph cannot be differentiated and says so
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=4, create_graph=False)
    res = mg.meta_gradient(tr, P.all_actions_query(tr.theta_K))
    assert not res.supported and res.reason == "inner_update_has_no_graph" and res.grads == {}
    # intact pathway for contrast
    assert _grad(P, mg, sup, 4)[1].grad_norm > 1e-3


def test_inner_update_changes_behaviour_depends_on_psi_and_never_touches_psi():
    P, sup = _problem()
    mg = _engine(P)
    h0 = params_hash(P.psi())
    p0 = P.probs(P.theta())
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=2)
    p1 = P.probs(tr.detached())
    assert float((p1 - p0).abs().max()) > 1e-2                       # the inner update changes the policy
    other = {n: v.detach() + 0.3 * torch.randn(v.shape, dtype=v.dtype, generator=torch.Generator().manual_seed(9))
             for n, v in P.psi().items()}
    p2 = P.probs(mg.inner_update(P.theta(), other, sup, K=2).detached())
    assert float((p2 - p1).abs().max()) > 1e-3                       # ... and how depends on the reward parameters
    assert params_hash(P.psi()) == h0 and all(p.grad is None for p in P.psi().values())
    assert all(p.grad is None for p in P.policy.parameters())        # functional: the live policy is untouched


def test_importance_weighted_mode_matches_on_policy_gradient_when_supported():
    P, sup = _problem()
    mg = _engine(P)
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=2)
    on = mg.meta_gradient(tr, P.all_actions_query(tr.theta_K))
    q = P.all_actions_query(tr.theta_K)
    c, a = q.batch["c"], q.batch["a"]
    mu = torch.softmax(torch.randn(P.n_ctx, P.n_arms, dtype=P.dtype, generator=torch.Generator().manual_seed(3)), -1)
    q.weights, q.behaviour_logp = P.p_ctx[c] * mu[c, a], torch.log(mu[c, a])     # rows as logged under mu
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=2)
    iw = M.MetaGradient(mg.inner_loss, P.logp_fn, inner_lr=ALPHA, weight_clip=1e9).meta_gradient(tr, q, "importance")
    assert iw.supported and iw.estimator == M.ESTIMATOR_IMPORTANCE and iw.max_weight > 1.0
    assert torch.allclose(M.flat(iw.grads), M.flat(on.grads), rtol=1e-9, atol=1e-12)
    assert iw.objective == pytest.approx(on.objective, abs=1e-12)


def test_importance_mode_refuses_instead_of_estimating_without_support():
    P, sup = _problem()
    mg = _engine(P, min_ess=4.0)
    g = torch.Generator().manual_seed(5)

    def run(blp, **kw):
        tr = mg.inner_update(P.theta(), P.psi(), sup, K=2)
        batch, U = P.sample(tr.detached(), 24, g)
        return mg.meta_gradient(tr, M.QueryData(batch=batch, utility=U, behaviour_logp=blp, **kw), "importance")
    uniform = torch.full((24,), math.log(1.0 / P.n_arms), dtype=P.dtype)
    ok = run(uniform)
    assert ok.supported and ok.ess > 4.0
    cases = {"missing_propensities": None,
             "invalid_propensities": torch.full((24,), 0.3, dtype=P.dtype),                # log-prob > 0
             "propensity_below_floor": torch.full((24,), math.log(1e-9), dtype=P.dtype)}
    for reason, blp in cases.items():
        r = run(blp)
        assert not r.supported and r.reason == reason and r.grads == {} and r.grad_norm == 0.0
    nan = uniform.clone(); nan[3] = float("nan")
    assert run(nan).reason == "invalid_propensities"
    # logged policy concentrated where the target puts little mass: one huge weight, tiny effective sample size
    skew = torch.full((24,), math.log(0.9), dtype=P.dtype); skew[0] = math.log(1e-5)
    r = run(skew)
    assert not r.supported and r.reason.startswith("effective_sample_size_too_small") and r.grads == {}
    assert not run(uniform, verified=torch.tensor([True] * 23 + [False])).supported
    # an unsupported result changes nothing
    opt = torch.optim.SGD(P.reward.parameters(), lr=1.0)
    h = params_hash(P.reward)
    assert M.MetaGradient.apply(opt, P.reward, r)["applied"] == 0.0 and params_hash(P.reward) == h


def test_importance_weights_are_clipped_and_reported():
    P, sup = _problem()
    mg = _engine(P, weight_clip=1.5, min_ess=1.0, min_ess_fraction=0.0)
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=1)
    batch, U = P.sample(tr.detached(), 64, torch.Generator().manual_seed(2))
    est = mg.outer_surrogate(tr.theta_K, M.QueryData(batch=batch, utility=U, behaviour_logp=torch.full(
        (64,), math.log(1.0 / P.n_arms), dtype=P.dtype)), "importance")
    assert est.supported and est.max_weight > 1.5 and 0.0 < est.clipped_fraction < 1.0


def test_query_groups_count_independent_outcomes_and_leave_one_group_out():
    P, sup = _problem()
    mg = _engine(P, min_independent=4)
    tr = mg.inner_update(P.theta(), P.psi(), sup, K=1)
    batch, _ = P.sample(tr.detached(), 12, torch.Generator().manual_seed(2))
    groups = torch.arange(12) // 4                                   # 12 rows, 3 independent outcomes
    U = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, -2.0, -2.0, -2.0, -2.0], dtype=P.dtype)
    est = mg.outer_surrogate(tr.theta_K, M.QueryData(batch=batch, utility=U, groups=groups))
    assert not est.supported and est.reason == "insufficient_independent_outcomes:3<4" and est.n_groups == 3
    mg3 = _engine(P, min_independent=3)
    est = mg3.outer_surrogate(tr.theta_K, M.QueryData(batch=batch, utility=U, groups=groups))
    assert est.supported and est.n == 12 and est.n_groups == 3
    lp = P.logp_fn(tr.theta_K, batch)
    base = torch.tensor([1.5] * 4 + [-0.5] * 4 + [3.0] * 4, dtype=P.dtype)      # mean of the *other* groups
    assert float(est.surrogate.detach()) == pytest.approx(float((lp * (U - base)).sum().detach() / 12), abs=1e-12)


def _enumerated_two_step_gradient(P, mg, n_ctx, n_arms):
    """Exact gradient of E_{D_0, D_1}[J(theta_2)] with one on-policy support row per step: all 16 data paths."""
    cs = torch.arange(n_ctx).repeat_interleave(n_arms)
    as_ = torch.arange(n_arms).repeat(n_ctx)
    C = cs.shape[0]
    psi = P.psi()
    th0 = {n: v.clone().requires_grad_(True) for n, v in P.theta(C).items()}
    p0 = (P.p_ctx[cs] * P.probs(P.theta())[cs, as_]).detach()                     # no psi dependence
    th1, _ = mg.inner_step(th0, psi, P.batch(cs[:, None], as_[:, None]))
    th1e = {n: v.repeat_interleave(C, 0) for n, v in th1.items()}
    b1 = P.batch(cs.repeat(C)[:, None], as_.repeat(C)[:, None])
    lp1 = P.logp_fn(th1e, b1)
    th2, _ = mg.inner_step(th1e, psi, b1)
    J = (p0.repeat_interleave(C) * P.p_ctx[cs.repeat(C)] * torch.exp(lp1) * P.exact_utility(th2)).sum()
    return M.flat(dict(zip(psi, torch.autograd.grad(J, list(psi.values()))))), float(J.detach())


@pytest.mark.parametrize("mode", ["unroll", "checkpoint"])
def test_sampled_on_policy_estimator_with_score_term_is_unbiased(mode):
    """On-policy support, on-policy query, leave-one-out baselines, parallel lifetimes: z-scores and relative
    error against the enumerated truth; leaving the score term out is biased."""
    P = BanditLifetime(n_ctx=2, n_arms=2, d_x=3, hidden=4, reward_hidden=4, seed=4)
    loss = M.policy_gradient_inner_loss(P.logp_fn, P.reward_fn, normalise=False, loo_baseline=False)
    mg = M.MetaGradient(loss, P.logp_fn, inner_lr=2.0, mode=mode)
    exact, _ = _enumerated_two_step_gradient(P, mg, 2, 2)
    R, calls = 4000, 24
    gen = torch.Generator().manual_seed(11)

    def estimate(on_policy):
        out = []
        for _ in range(calls):
            tr = mg.inner_update(P.theta(R), P.psi(), lambda k, th: P.sample(th, 1, gen)[0], K=2,
                                 on_policy=on_policy, n_lifetimes=R)
            batch, U = P.sample(tr.detached(), 2, gen)
            res = mg.meta_gradient(tr, M.QueryData(batch=batch, utility=U, lifetime=batch["lifetime"]))
            assert res.supported and res.score_term_used == on_policy
            out.append(M.flat(res.grads) / R)
        s = torch.stack(out)
        mean, se = s.mean(0), s.std(0) / math.sqrt(calls)
        return float(((mean - exact) / se).abs().max()), float((mean - exact).norm() / exact.norm())
    z, rel = estimate(True)
    assert z < 4.5 and rel < 0.05, (z, rel)
    if mode == "unroll":
        z_no, rel_no = estimate(False)
        assert z_no > 5.0 and rel_no > 2 * rel, (z_no, rel_no)


def test_apply_steps_only_the_reward_optimizer_and_ascends_the_verified_objective():
    P, sup = _problem()
    mg = _engine(P)
    J = _objective(P, mg, sup, 2)
    before = J({n: v.detach() for n, v in P.psi().items()})
    opt = torch.optim.SGD(P.reward.parameters(), lr=0.05)
    _, res = _grad(P, mg, sup, 2)
    h_policy = params_hash(P.policy)
    info = M.MetaGradient.apply(opt, P.reward, res, clip=0.5)
    assert info["applied"] == 1.0 and info["clip_scale"] <= 1.0
    assert J({n: v.detach() for n, v in P.psi().items()}) > before
    assert params_hash(P.policy) == h_policy
    both = torch.optim.SGD(list(P.reward.parameters()) + list(P.policy.parameters()), lr=0.05)
    with pytest.raises(ValueError):
        M.MetaGradient.apply(both, P.reward, res)
    rec = res.record_fields()
    assert rec["estimator"] == "on_policy_query" and rec["inner_steps"] == 2 and rec["skipped_reason"] == ""


def test_meta_training_improves_verified_utility_of_fresh_learners_on_held_out_support():
    """Scale-normalised inner loss: whatever is learned is the reward's shape, not its magnitude."""
    P, sup = _problem(seed=3, n_rows=16)
    mg = _engine(P, normalise=True)
    opt = torch.optim.Adam(P.reward.parameters(), lr=0.02)
    g = torch.Generator().manual_seed(77)
    fresh = lambda: [P.batch(torch.randint(P.n_ctx, (16,), generator=g), torch.randint(P.n_arms, (16,), generator=g))
                     for _ in range(2)]
    held_out = [fresh() for _ in range(20)]
    evaluate = lambda: sum(float(P.exact_utility(mg.inner_update(P.theta(), P.psi(), b, K=2, create_graph=False)
                                                 .detached())) for b in held_out) / len(held_out)
    start = evaluate()
    for _ in range(150):
        tr = mg.inner_update(P.theta(), P.psi(), fresh(), K=2)
        M.MetaGradient.apply(opt, P.reward, mg.meta_gradient(tr, P.all_actions_query(tr.theta_K)), clip=1.0)
    end = evaluate()
    untrained = float(P.exact_utility(P.theta()))
    assert end > start + 0.1 and end > untrained + 0.1, (start, end, untrained)


def test_discounted_returns_and_dtype_paths():
    r = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    assert M.discounted_returns(r).tolist() == [[6.0, 5.0, 3.0]]
    gam = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    assert M.discounted_returns(r, gamma=gam).tolist() == [[1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3.0]]
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64)
    assert M.discounted_returns(r, mask).tolist() == [[3.0, 2.0, 0.0]]
    P32, sup = _problem(dtype=torch.float32)
    _, res = _grad(P32, _engine(P32), sup, 2)
    P64, sup64 = _problem()
    _, ref = _grad(P64, _engine(P64), sup64, 2)
    assert res.supported and all(g.dtype == torch.float32 for g in res.grads.values())
    assert float((M.flat(res.grads).double() - M.flat(ref.grads)).norm() / M.flat(ref.grads).norm()) < 1e-3
