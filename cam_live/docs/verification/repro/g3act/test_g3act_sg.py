"""G3-act numerical property checks of SG's learned components (CPU, no browser).

* ApplicabilityModel.update is the assumed-density-filtering (moment-matching) update of a probit likelihood: after one
  observation the factorised posterior's mean / variance equal the moments of prior x Phi(y w x / beta) computed by
  quadrature (one feature, and two features where the exact posterior is no longer factorised: the marginal means /
  variances of moment matching are exact for the projection onto the joint Gaussian's marginals).
* ApplicabilityModel.p_below(tau) equals a Monte-Carlo estimate of P(Phi(w.x/beta) < tau) under the posterior.
* TransferEstimator: no evidence -> the prior 1/2; positive measured gains raise the estimate, negative ones lower it;
  the gain enters as (1 + clip(gain / scale)) / 2.

Run: PYTHONPATH=<tree>/cam_live/src .venv/bin/python -m pytest <this file> -q -p no:cacheprovider
"""
import math
import random

from cam_live.harness.skills import ApplicabilityModel
from cam_live.reward.selfgoals import TransferEstimator


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moments_1d(prior_var, beta, y, n=20001, lim=8.0):
    sd = math.sqrt(prior_var)
    xs = [(-lim + 2 * lim * i / (n - 1)) * sd for i in range(n)]
    w = [math.exp(-x * x / (2 * prior_var)) * _Phi(y * x / beta) for x in xs]
    z = sum(w)
    m = sum(a * b for a, b in zip(w, xs)) / z
    v = sum(a * (b - m) ** 2 for a, b in zip(w, xs)) / z
    return m, v


def test_adf_update_matches_quadrature_one_feature():
    for ok in (True, False):
        for prior_var in (0.25, 1.0, 4.0):
            am = ApplicabilityModel(prior_var=prior_var, beta=1.0)
            am.update(["f"], ok)
            m, v = _moments_1d(prior_var, 1.0, 1.0 if ok else -1.0)
            assert abs(am.mu["f"] - m) < 1e-4, (ok, prior_var, am.mu["f"], m)
            assert abs(am.var["f"] - v) < 1e-4, (ok, prior_var, am.var["f"], v)


def test_adf_update_matches_quadrature_two_features_marginals():
    # w1, w2 ~ N(0, s2) iid; likelihood Phi(y (w1 + w2) / beta). The sum u = w1 + w2 ~ N(0, 2 s2); by symmetry
    # E[w1 | data] = E[u | data] / 2 and Var[w1 | data] = s2/2 + Var[u | data] / 4.
    s2 = 0.25
    am = ApplicabilityModel(prior_var=s2, beta=1.0)
    am.update(["a", "b"], True)
    mu, vu = _moments_1d(2 * s2, 1.0, 1.0)
    assert abs(am.mu["a"] - mu / 2) < 1e-4 and abs(am.mu["b"] - mu / 2) < 1e-4
    assert abs(am.var["a"] - (s2 / 2 + vu / 4)) < 1e-4, (am.var["a"], s2 / 2 + vu / 4)


def test_p_below_matches_monte_carlo():
    rng = random.Random(0)
    am = ApplicabilityModel(prior_var=0.25, beta=1.0)
    for ok in (True, False, False, True, False):
        am.update(["x", "y"], ok)
    feats = ["x", "y"]
    for tau in (0.15, 0.5, 0.8):
        n, below = 40000, 0
        for _ in range(n):
            s = sum(rng.gauss(am.mu[f], math.sqrt(am.var[f])) for f in feats)
            below += _Phi(s / am.beta) < tau
        assert abs(below / n - am.p_below(feats, tau)) < 0.01, (tau, below / n, am.p_below(feats, tau))


def test_transfer_estimator_prior_and_direction():
    te = TransferEstimator()
    assert te.estimate("op|m", [1.0, 0.0])["mean"] == 0.5
    for g in (0.2, 0.3, 0.1):
        te.observe("op|m", [1.0, 0.0], g)
    up = te.estimate("op|m", [1.0, 0.0])["mean"]
    te2 = TransferEstimator()
    for g in (-0.2, -0.3, -0.1):
        te2.observe("op|m", [1.0, 0.0], g)
    down = te2.estimate("op|m", [1.0, 0.0])["mean"]
    assert up > 0.5 > down
    # exact value: Beta(1 + sum s_i, 1 + sum (1 - s_i)), s = (1 + clip(g / 0.25)) / 2
    s = [0.5 * (1 + max(-1, min(1, g / 0.25))) for g in (0.2, 0.3, 0.1)]
    assert abs(up - (1 + sum(s)) / (2 + 3)) < 1e-12
