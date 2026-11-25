"""CAM 7.0 math (NIG / Student-t / KG / EIG / Thompson) ported unchanged; streaming statistics; guarded math-assist
inputs; spec 8.5 baselines on the probe bandit."""
import math

import pytest
import torch

from superlearner.cam7 import posterior as cam7
from superlearner.cam_unified.interfaces import Action, TokenKind
from superlearner.cam_unified.learn import math_assist as M
from superlearner.cam_unified.learn.diagnostics import ProbeBanditAdapter
from superlearner.cam_unified.learn.meta import TrialRunner


def _data(B=3, T=20, D=4, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(B, T, D, generator=g, dtype=torch.float64)
    w = torch.randn(B, K, D, generator=g, dtype=torch.float64)
    y = torch.einsum("btd,bkd->btk", phi, w) + 0.3 * torch.randn(B, T, K, generator=g, dtype=torch.float64)
    om = (torch.rand(B, T, generator=g) > 0.2).double()
    pri = (torch.zeros(K, D), torch.full((K, D), 0.5), torch.full((K,), 2.0), torch.full((K,), 1.0))
    return phi, y, om, pri


def test_ported_nig_and_voi_math_equals_cam7():
    phi, y, om, pri = _data()
    a, b = M.fit(phi, y, om, *pri), cam7.fit(phi, y, om, *pri)
    for f in ("mu", "chol", "a", "b", "n"):
        assert torch.allclose(getattr(a, f), getattr(b, f))
    q = torch.randn(3, 5, 4, dtype=torch.float64)
    for x, z in zip(M.predict(a, q), cam7.predict(b, q)):
        assert torch.allclose(x, z)
    mean, cov = M.mean_cov(a, q)
    mean2, cov2 = cam7.mean_cov(b, q)
    assert torch.allclose(mean, mean2) and torch.allclose(cov, cov2)
    obs = a.s2[:, :1].expand(3, 5)
    kg = M.knowledge_gradient(mean[..., 0], cov[:, 0], obs)
    assert torch.allclose(kg, cam7.knowledge_gradient(mean[..., 0], cov[:, 0], obs))
    assert torch.all(kg >= 0)
    assert torch.allclose(M.info_gain(cov[:, 0], obs), cam7.info_gain(cov[:, 0], obs))
    loc, s2, dof = M.predict(a, q)
    yq = torch.randn(3, 5, 2, dtype=torch.float64)
    assert torch.allclose(M.t_nll(yq, loc, s2, dof), cam7.t_nll(yq, loc, s2, dof))


def test_streaming_statistics_equal_full_refit_and_decay():
    phi, y, om, pri = _data(T=30)
    st = M.NIGStats(3, 4, 2)
    for t in range(0, 30, 7):                 # arbitrary chunks
        st.add(phi[:, t:t + 7], y[:, t:t + 7], om[:, t:t + 7])
    p1, p2 = st.posterior(*pri), M.fit(phi, y, om, *pri)
    for f in ("mu", "a", "b", "n"):
        assert torch.allclose(getattr(p1, f), getattr(p2, f), atol=1e-9)
    st2 = M.NIGStats(3, 4, 2).add(phi[:, :10], y[:, :10]).decay(0.5).add(phi[:, 10:20], y[:, 10:20])
    w = torch.cat([torch.full((3, 10), 0.5), torch.ones(3, 10)], 1).double()
    p3, p4 = st2.posterior(*pri), M.fit(phi[:, :20], y[:, :20], w, *pri)
    assert torch.allclose(p3.mu, p4.mu, atol=1e-9) and torch.allclose(p3.b, p4.b, atol=1e-9)


def test_thompson_draws_reproducible_and_consistent():
    phi, y, om, pri = _data(T=50)
    post = M.fit(phi, y, om, *pri)
    w1 = M.sample_weights(post, 4000, torch.Generator().manual_seed(3))
    w2 = M.sample_weights(post, 4000, torch.Generator().manual_seed(3))
    assert torch.equal(w1, w2)
    assert torch.allclose(w1.mean(1), post.mu, atol=0.05)
    g = M._gamma_mt(torch.full((20000,), 0.7, dtype=torch.float64), torch.Generator().manual_seed(1))
    assert g.mean().item() == pytest.approx(0.7, rel=0.05) and g.var().item() == pytest.approx(0.7, rel=0.1)


def test_math_assist_inputs_only_in_the_declared_condition():
    scores = torch.randn(2, 7, 5)
    off = M.MathAssistEncoder(32, enabled=False)
    with pytest.raises(RuntimeError):
        off(scores)
    on = M.MathAssistEncoder(32, enabled=True)
    tok = on(scores)
    assert tok.x.shape == (2, 7, 32) and int(tok.kind[0, 0]) == TokenKind.BELIEF
    assert tok.refs[0][3] == "math_assist:3"
    cvm = M.CandidateValueModel(2, 7, prior_mean=[-30.0] + [0.0] * 6, b0=1e4)
    s = cvm.scores()
    assert s.shape == (2, 7, 5) and s[0, 0, 0].item() == pytest.approx(-30.0)
    cvm.update(torch.tensor([3, 3]), torch.tensor([100.0, 0.0]))
    s2 = cvm.scores()
    assert s2[0, 3, 0].item() > 10.0 and abs(s2[1, 3, 0].item()) < 1e-6     # 100 observed vs 0 observed
    assert s2[0, 3, 1] < s[0, 3, 1]                                         # observing shrinks the posterior sd
    assert torch.all(s2[..., 2] >= 0)


def test_discrete_tool_baselines_run_on_the_probe_bandit():
    ad = ProbeBanditAdapter(64, horizon=6)
    ad.reset()
    cands = M.DiscreteToolPolicy.candidates(ad.available_tools()[0])
    assert [c[0] for c in cands] == ["probe"] + ["pull"] * 6 and cands[0][2] == 30.0
    rets = {}
    for kind in M.BanditBaseline.KINDS:
        pol = M.DiscreteToolPolicy(kind, prior_mean={"pull": 100 / 6}, seed=1)
        res = TrialRunner(ad, ad.evaluator).run(pol, list(range(500, 564)))
        rets[kind] = float(res.per_trial_return.sum(1).mean())
        assert all(len(a) == 6 for a in res.actions)
    assert rets["greedy"] > rets["random"] + 100            # posterior-mean elimination vs uniform random
    assert rets["greedy"] <= ad.elimination_value(6) + 60   # no baseline exploits the probe's information


def test_structured_prior_kg_values_the_information_only_probe():
    """With correlated hypotheses the probe's report updates every arm's value: KG(probe) > 0 although the probe pays
    nothing; the independent-candidate model gives it no value of information."""
    ad = ProbeBanditAdapter(4, horizon=6)
    sp = ad.structured_prior()
    m = M.HypothesisValueModel(4, sp)
    mean, kg = m.mean(), m.knowledge_gradient()
    assert mean[0, 0].item() == pytest.approx(-30.0) and mean[0, 1].item() == pytest.approx(100 / 6)
    assert kg[0, 0].item() == pytest.approx(100.0 - 100 / 6)                 # after the probe the best arm pays 100
    assert kg[0, 1].item() == pytest.approx((1 / 6) * (100 - 100 / 6) + (5 / 6) * (20 - 100 / 6))
    assert m.info_gain()[0, 0].item() == pytest.approx(math.log(6))          # the probe reveals the hypothesis
    m.update(torch.zeros(4, dtype=torch.long), torch.tensor([2, 2, 3, 5]))  # probe outcomes: arms 2, 2, 3, 5 signal
    assert torch.allclose(m.mean()[:, 1 + 2], torch.tensor([100.0, 100.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(m.knowledge_gradient(), torch.zeros(4, 7, dtype=torch.float64), atol=1e-9)
    # the outcome reader maps observations to outcome indices
    ad.reset([0, 1, 2, 3])
    res = ad.execute([Action("probe")] * 4)
    for b in range(4):
        g = ad.evaluator.evaluate_goal(b).details["hidden"]["good_arm"]
        assert sp.outcome_fn(0, res[b].observations) == g
    # one-step VOI on the independent model: the probe has no value
    cvm = M.CandidateValueModel(1, 7, prior_mean=[-30.0] + [100 / 6] * 6, b0=100.0 ** 2)
    assert cvm.scores()[0, 0, 2].item() < cvm.scores()[0, 1:, 2].min().item()


def test_structured_kg_baseline_probes_first_and_is_near_bayes_optimal():
    rets, first = {}, {}
    for kind in ("kg", "kg_structured", "greedy_structured"):
        for H in (2, 4, 6):
            ad = ProbeBanditAdapter(64, horizon=H)
            pol = M.DiscreteToolPolicy(kind, prior_mean={"pull": 100 / 6}, seed=1, structured=ad.structured_prior())
            res = TrialRunner(ad, ad.evaluator).run(pol, list(range(900, 964)))
            rets[(kind, H)] = float(res.per_trial_return.sum(1).mean())
            first[(kind, H)] = sum(a[0] == "probe" for a in res.actions) / 64
    for H in (2, 4, 6):
        assert first[("kg_structured", H)] == 1.0 and first[("kg", H)] == 0.0
        assert rets[("kg_structured", H)] == pytest.approx(ad.bayes_optimal_value(H))   # noise-free: exact
        assert rets[("kg_structured", H)] > rets[("kg", H)] + 15
        assert first[("greedy_structured", H)] == 0.0                                  # myopic: never probes
    with pytest.raises(ValueError):
        M.BanditBaseline("kg_structured", 7)


def test_math_assist_from_config_and_thompson_draws_with_a_generator_on_another_device():
    from superlearner.cam_unified.config import make_config
    assert M.MathAssistEncoder.from_config(make_config("dev")).enabled is False
    on = M.MathAssistEncoder.from_config(make_config("dev", math_explore_assist=True), d_model=16)
    assert on.enabled and on(torch.randn(1, 3, 5)).x.shape == (1, 3, 16)
    phi, y, om, pri = _data(T=30)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    post = M.fit(phi.to(dev), y.to(dev), om.to(dev), *(p.to(dev) for p in pri))
    w1 = M.sample_weights(post, 8, torch.Generator().manual_seed(5))
    w2 = M.sample_weights(post, 8, torch.Generator().manual_seed(5))
    assert torch.equal(w1, w2) and w1.device.type == torch.device(dev).type
