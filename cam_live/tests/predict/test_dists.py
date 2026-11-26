"""dists: exact likelihoods, CDFs, quantiles and sampling against scipy; masks; record round trips."""
import math

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import torch

from cam_live.contracts import Distribution, OutcomeStatus
from cam_live.predict import dists as D

F64 = torch.float64


def t64(x):
    return torch.as_tensor(x, dtype=F64)


def test_normal_and_student_t_match_scipy():
    y = t64([-3.0, -0.2, 0.0, 1.7, 40.0])
    loc, scale, df = t64([0.5, -1.0, 0.0, 2.0, 3.0]), t64([0.1, 2.0, 1.0, 0.7, 5.0]), t64([1.0, 2.5, 4.0, 30.0, 0.7])
    np.testing.assert_allclose(D.normal_log_prob(y, loc, scale), st.norm.logpdf(y, loc, scale), rtol=1e-12)
    np.testing.assert_allclose(D.student_t_log_prob(y, loc, scale, df), st.t.logpdf(y, df, loc, scale), rtol=1e-12)
    n, t = D.Normal(loc, scale), D.StudentT(loc, scale, df)
    np.testing.assert_allclose(n.cdf(y), st.norm.cdf(y, loc, scale), rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(t.cdf(y), st.t.cdf(y, df, loc, scale), rtol=1e-9)
    for q in (0.001, 0.05, 0.5, 0.9, 0.999):
        np.testing.assert_allclose(n.quantile(q), st.norm.ppf(q, loc, scale), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(t.quantile(q), st.t.ppf(q, df, loc, scale), rtol=1e-7, atol=1e-7)
    assert math.isnan(float(t.mean[-1])) and math.isinf(float(t.variance[0]))
    np.testing.assert_allclose(t.variance[3], st.t.var(30.0, 2.0, 0.7), rtol=1e-12)


def test_betainc_matches_scipy():
    rng = np.random.default_rng(0)
    a, b, x = rng.uniform(0.05, 60, 400), rng.uniform(0.05, 60, 400), rng.uniform(0, 1, 400)
    np.testing.assert_allclose(D.betainc(a, b, x), sp.betainc(a, b, x), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(D.betainc([2.0, 2.0], [3.0, 3.0], [0.0, 1.0]), [0.0, 1.0])
    for ai, bi, xi in zip(a[:60], b[:60], x[:60]):
        np.testing.assert_allclose(D.betainc_scalar(ai, bi, xi), sp.betainc(ai, bi, xi), rtol=1e-9, atol=1e-13)
    for df in (1.0, 3.5, 40.0):
        for q in (0.001, 0.2, 0.975):
            np.testing.assert_allclose(D.student_t_icdf_scalar(q, df), st.t.ppf(q, df), rtol=1e-8)
            np.testing.assert_allclose(D.student_t_cdf_scalar(st.t.ppf(q, df), df), q, rtol=1e-9)


def test_count_families_match_scipy():
    y = t64([0, 1, 2, 7, 40, 300])
    rate, expo, conc = t64([0.3, 2.0, 0.01, 5.0, 4.0, 2.5]), t64([1.0, 0.5, 10.0, 2.0, 12.0, 100.0]), \
        t64([0.2, 1.0, 3.0, 10.0, 0.7, 50.0])
    mu = rate * expo
    np.testing.assert_allclose(D.poisson_log_prob(y, rate, expo), st.poisson.logpmf(y, mu), rtol=1e-11)
    ref = st.nbinom.logpmf(y, conc, conc / (conc + mu))
    np.testing.assert_allclose(D.neg_binomial_log_prob(y, rate, conc, expo), ref, rtol=1e-10)
    po, nb = D.Poisson(rate, expo), D.NegBinomial(rate, conc, expo)
    np.testing.assert_allclose(po.mean, mu)
    np.testing.assert_allclose(nb.variance, st.nbinom.var(conc, conc / (conc + mu)), rtol=1e-12)
    np.testing.assert_allclose(po.cdf(y), st.poisson.cdf(y, mu), rtol=1e-7, atol=1e-300)   # torch gammaincc precision
    np.testing.assert_allclose(nb.cdf(y), st.nbinom.cdf(y, conc, conc / (conc + mu)), rtol=1e-8)
    for q in (0.01, 0.25, 0.5, 0.9, 0.999):
        np.testing.assert_array_equal(po.quantile(q), st.poisson.ppf(q, mu))
        np.testing.assert_array_equal(nb.quantile(q), st.nbinom.ppf(q, conc, conc / (conc + mu)))
    # exposure passed at evaluation time overrides the stored one
    np.testing.assert_allclose(po.log_prob(y, exposure=3.0), st.poisson.logpmf(y, rate * 3.0), rtol=1e-11)
    # the negative binomial tends to the Poisson as the concentration grows
    big = D.neg_binomial_log_prob(y, rate, t64(1e9), expo)
    np.testing.assert_allclose(big, st.poisson.logpmf(y, mu), rtol=1e-5, atol=1e-5)


def test_neg_binomial_log_space_is_finite_at_extremes():
    lp = D.neg_binomial_log_prob_from_log(t64([0.0, 5.0, 1e6]), t64([-60.0, 40.0, 13.0]), t64([-30.0, 30.0, 0.0]))
    assert torch.isfinite(lp).all()


def test_bernoulli_and_categorical_match_scipy():
    logit = t64([-40.0, -1.2, 0.0, 3.0, 60.0])
    for yv in (0.0, 1.0):
        y = torch.full_like(logit, yv)
        ref = st.bernoulli.logpmf(yv, sp.expit(logit.numpy()))
        got = D.bernoulli_log_prob(y, logit)
        assert torch.isfinite(got).all()                       # scipy underflows to -inf at |logit| = 60
        np.testing.assert_allclose(got[1:4], ref[1:4], rtol=1e-12)
    logits = t64(np.random.default_rng(1).normal(size=(4, 6)))
    yk = torch.tensor([0, 5, 2, 3])
    p = sp.softmax(logits.numpy(), axis=-1)
    np.testing.assert_allclose(D.categorical_log_prob(yk, logits), np.log(p[np.arange(4), yk]), rtol=1e-12)
    ncat = torch.tensor([6, 6, 3, 4])
    lp = D.masked_log_softmax(logits, ncat)
    np.testing.assert_allclose(torch.exp(lp).sum(-1), np.ones(4), rtol=1e-12)
    assert float(torch.exp(lp[2, 3:]).sum()) < 1e-300
    c = D.Categorical(logits, ncat)
    np.testing.assert_allclose(c.entropy()[0], st.entropy(p[0]), rtol=1e-10)
    np.testing.assert_allclose(c.cdf(t64([1, 1, 1, 1]))[0], p[0, :2].sum(), rtol=1e-12)
    assert c.quantile(0.999999)[2] <= 2


def test_discrete_hazard_matches_geometric_and_brute_force():
    k_bins, h = 12, 0.2
    logits = t64(np.full(k_bins, sp.logit(h)))
    edges = t64(np.arange(k_bins + 1))                         # unit bins: T in bin k  <=>  geometric = k + 1
    d = D.DiscreteHazard(logits, edges)
    for k in range(k_bins):
        np.testing.assert_allclose(d.log_prob(t64(k + 0.5)), st.geom.logpmf(k + 1, h), rtol=1e-12)
        np.testing.assert_allclose(d.log_prob(t64(k + 0.5), event=torch.tensor(False)), st.geom.logsf(k, h),
                                   rtol=1e-12, atol=1e-14)   # censored inside bin k: survived bins < k only
    np.testing.assert_allclose(d.survival(), st.geom.sf(np.arange(1, k_bins + 1), h), rtol=1e-12)
    np.testing.assert_allclose(d.pmf().sum(), 1.0, rtol=1e-12)
    # an event at / beyond the last edge and a censoring at the last edge carry the same information
    np.testing.assert_allclose(d.log_prob(t64(99.0)), d.log_prob(t64(12.0), event=torch.tensor(False)))
    np.testing.assert_allclose(d.log_prob(t64(12.0), event=torch.tensor(False)), st.geom.logsf(k_bins, h), rtol=1e-12)
    # non-uniform hazards and batched edges, brute force
    rng = np.random.default_rng(3)
    lg = t64(rng.normal(size=(5, 7)))
    hz = sp.expit(lg.numpy())
    eg = t64(np.cumsum(np.concatenate([np.zeros((5, 1)), rng.uniform(0.5, 3, size=(5, 7))], 1), 1))
    dd = D.DiscreteHazard(lg, eg)
    pmf = np.array([[hz[i, k] * np.prod(1 - hz[i, :k]) for k in range(7)] for i in range(5)])
    tail = np.prod(1 - hz, axis=1)
    np.testing.assert_allclose(dd.pmf(), np.concatenate([pmf, tail[:, None]], 1), rtol=1e-12)
    mid = 0.5 * (eg[:, :-1] + eg[:, 1:]).numpy()
    np.testing.assert_allclose(dd.mean, (pmf * mid).sum(1) + tail * eg[:, -1].numpy(), rtol=1e-12)
    t_ev = 0.5 * (eg[:, 3] + eg[:, 4])
    np.testing.assert_allclose(dd.log_prob(t_ev), np.log(pmf[:, 3]), rtol=1e-12)
    np.testing.assert_allclose(dd.log_prob(t_ev, event=torch.zeros(5, dtype=torch.bool)),
                               np.log(np.prod(1 - hz[:, :3], axis=1)), rtol=1e-12)
    # cdf and quantile are inverse on the event mass
    q = 0.6 * dd.p_event()
    np.testing.assert_allclose(dd.cdf(dd.quantile(q)), q, rtol=1e-9)
    assert torch.equal(dd.quantile(0.999999999), eg[:, -1])


def test_restricted_mean_equals_area_under_survival():
    lg = t64(np.random.default_rng(5).normal(size=9))
    edges = t64(np.linspace(0, 18, 10))
    d = D.DiscreteHazard(lg, edges)
    s = np.concatenate([[1.0], d.survival().numpy()])
    area = np.sum(np.diff(edges.numpy()) * 0.5 * (s[:-1] + s[1:]))
    np.testing.assert_allclose(d.mean, area, rtol=1e-12)


@pytest.mark.parametrize("dist, ref_cdf", [
    (D.Normal(t64(1.5), t64(2.0)), lambda x: st.norm.cdf(x, 1.5, 2.0)),
    (D.StudentT(t64(-1.0), t64(0.5), t64(3.0)), lambda x: st.t.cdf(x, 3.0, -1.0, 0.5)),
])
def test_continuous_sampling_passes_ks(dist, ref_cdf, gen):
    x = dist.sample(20000, generator=gen).numpy()
    assert st.kstest(x, ref_cdf).pvalue > 1e-3


def test_discrete_sampling_matches_moments(gen):
    n = 60000
    po = D.Poisson(t64([0.5, 7.0]), t64([2.0, 1.0])).sample(n, gen)
    nb = D.NegBinomial(t64([3.0]), t64([1.5]), t64([2.0]))
    xs = nb.sample(n, gen)
    np.testing.assert_allclose(po.mean(0), [1.0, 7.0], rtol=0.03)
    np.testing.assert_allclose(xs.mean(0), nb.mean, rtol=0.03)
    np.testing.assert_allclose(xs.var(0), nb.variance, rtol=0.08)
    be = D.Bernoulli(t64([-1.0, 2.0])).sample(n, gen)
    np.testing.assert_allclose(be.mean(0), sp.expit([-1.0, 2.0]), atol=0.01)
    ca = D.Categorical(t64([[0.0, 1.0, -1.0], [2.0, 0.0, 0.0]]))
    s = ca.sample(n, gen)
    assert s.shape == (n, 2)
    freq = torch.stack([(s == k).double().mean(0) for k in range(3)], -1)
    np.testing.assert_allclose(freq, ca.probs, atol=0.01)
    hz = D.DiscreteHazard(t64([-1.0, 0.0, 1.0]), t64([0.0, 1.0, 3.0, 4.0]))
    th = hz.sample(n, gen)
    np.testing.assert_allclose(torch.isinf(th).double().mean(), hz.pmf()[-1], atol=0.01)
    np.testing.assert_allclose(th.clamp(max=4.0).mean(), hz.mean, rtol=0.02)


def test_record_round_trip_for_every_family():
    batch = {
        "normal": D.Normal(t64([[0.0, 1.0]]), t64([[1.0, 2.0]])),
        "student_t": D.StudentT(t64([[0.0, 1.0]]), t64([[1.0, 2.0]]), t64([[3.0, 9.0]])),
        "poisson": D.Poisson(t64([[0.4, 2.0]]), t64([[1.0, 5.0]])),
        "neg_binomial": D.NegBinomial(t64([[0.4, 2.0]]), t64([[1.0, 4.0]]), t64([[1.0, 5.0]])),
        "bernoulli": D.Bernoulli(t64([[0.3, -2.0]])),
        "discrete_hazard": D.DiscreteHazard(t64([[[0.1, 0.2, -1.0], [1.0, 0.0, 0.5]]]), t64([0.0, 1.0, 2.0, 5.0])),
        "categorical": D.Categorical(t64([[[0.1, 0.2, -1.0], [1.0, 0.0, 0.5]]])),
    }
    y = {"normal": 0.7, "student_t": 0.7, "poisson": 3.0, "neg_binomial": 3.0, "bernoulli": 1.0,
         "discrete_hazard": 1.5, "categorical": 2.0}
    for fam, d in batch.items():
        rec = d.to_record(index=(0, 1), epistemic_std=0.25)
        assert rec.family == fam and rec.epistemic_std == 0.25
        back = D.from_record(Distribution.from_dict(rec.to_dict()))          # through JSON-native form
        yv = t64(y[fam])
        full = d.log_prob(yv.expand(1, 2) if fam != "categorical" else yv.expand(1, 2).long())[0, 1]
        np.testing.assert_allclose(back.log_prob(yv if fam != "categorical" else yv.long()), full, rtol=1e-12)
        np.testing.assert_allclose(D.record_log_prob(rec, y[fam]), full, rtol=1e-12)
        if fam != "categorical":
            np.testing.assert_allclose(rec.mean, d.mean[0, 1], rtol=1e-12)
    with pytest.raises(ValueError):
        batch["normal"].to_record()                                           # ambiguous without an index
    with pytest.raises(ValueError):
        D.from_record(Distribution("cauchy", {}))


def test_record_log_prob_statuses():
    hz = D.DiscreteHazard(t64([0.0, 0.0]), t64([0.0, 10.0, 20.0])).to_record()
    np.testing.assert_allclose(D.record_log_prob(hz, 20.0, OutcomeStatus.CENSORED), 2 * math.log(0.5))
    np.testing.assert_allclose(D.record_log_prob(hz, 12.0, OutcomeStatus.OBSERVED), 2 * math.log(0.5))
    with pytest.raises(ValueError):
        D.record_log_prob(hz, 1.0, OutcomeStatus.PENDING)
    with pytest.raises(ValueError):
        D.record_log_prob(hz, 1.0, OutcomeStatus.UNOBSERVABLE)
    with pytest.raises(ValueError):
        D.record_log_prob(D.Normal(0.0, 1.0).to_record(), 1.0, OutcomeStatus.CENSORED)
    po = D.Poisson(2.0, 1.0).to_record()
    np.testing.assert_allclose(D.record_log_prob(po, 3.0, exposure=4.0), st.poisson.logpmf(3, 8.0), rtol=1e-12)


def test_masked_targets_give_exactly_zero_loss_and_gradient():
    loc = torch.zeros(4, requires_grad=True)
    y = torch.tensor([1.0, float("nan"), -2.0, float("inf")])
    mask = torch.tensor([True, False, True, False])
    nll = -D.normal_log_prob(D.safe_targets(y, mask), loc, torch.ones(4))
    loss = D.masked_mean(nll, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.equal(loc.grad[~mask], torch.zeros(2)) and (loc.grad[mask] != 0).all()
    none = D.masked_mean(-D.normal_log_prob(D.safe_targets(y, ~mask | mask & False), loc, torch.ones(4)),
                         torch.zeros(4, dtype=torch.bool))
    assert float(none.detach()) == 0.0


def test_uniform_edges_batched():
    e = D.uniform_edges(torch.tensor([10.0, 100.0]), 4)
    assert e.shape == (2, 5) and torch.allclose(e[1], torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0]))
    assert D.default_family("count") == "neg_binomial"
