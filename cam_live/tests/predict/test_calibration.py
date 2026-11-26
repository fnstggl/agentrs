"""Calibration metrics against closed forms / scipy, interval coverage, grouping, and the prospective guard."""
import math

import numpy as np
import pytest
import scipy.optimize as so
import scipy.stats as st

from cam_live.contracts import Distribution, OutcomeQuery, OutcomeStatus, PredictionRecord
from cam_live.predict import calibration as C
from cam_live.predict import dists as D


def _calibrated(n, rng, slope=1.0, intercept=0.0):
    """Forecasts p whose true success probability is sigmoid(intercept + slope * logit(p))."""
    p = rng.uniform(0.03, 0.97, n)
    true = 1 / (1 + np.exp(-(intercept + slope * C.logit(p))))
    return p, (rng.uniform(size=n) < true).astype(float)


def test_per_item_scores():
    np.testing.assert_allclose(C.brier([0.8, 0.2], [1, 1]), [0.04, 0.64])
    np.testing.assert_allclose(C.binary_nll([0.8, 0.2], [1, 0]), [-math.log(0.8)] * 2)
    assert np.isfinite(C.binary_nll([0.0, 1.0], [1, 0])).all()


def test_studentised_ci_matches_scipy_and_respects_clusters():
    rng = np.random.default_rng(0)
    v = rng.normal(2.0, 3.0, 37)
    ci = C.studentised_ci(v, 0.9)
    lo, hi = st.t.interval(0.9, 36, loc=v.mean(), scale=st.sem(v))
    np.testing.assert_allclose([ci["lo"], ci["hi"]], [lo, hi], rtol=1e-8)
    assert ci["n"] == 37
    # 5 clusters of 20 perfectly correlated rows are 5 observations, not 100
    base = rng.normal(size=5)
    rows = np.repeat(base, 20)
    labels = np.repeat(np.arange(5), 20)
    naive, clustered = C.studentised_ci(rows), C.studentised_ci(rows, clusters=labels)
    assert clustered["n"] == 5 and clustered["hi"] - clustered["lo"] > 3 * (naive["hi"] - naive["lo"])
    lo5, hi5 = st.t.interval(0.95, 4, loc=base.mean(), scale=st.sem(base))
    np.testing.assert_allclose([clustered["lo"], clustered["hi"]], [lo5, hi5], rtol=1e-8)
    assert C.studentised_ci([])["n"] == 0 and math.isinf(C.studentised_ci([1.0])["hi"])


def test_t_interval_coverage_of_expected_brier():
    rng = np.random.default_rng(1)
    hits, reps = 0, 400
    grid = np.linspace(0.03, 0.97, 200001)
    true = float(np.mean(grid * (1 - grid)))                        # E[p (1 - p)] for p ~ U(0.03, 0.97)
    for _ in range(reps):
        p, y = _calibrated(80, rng)
        ci = C.studentised_ci(C.brier(p, y))
        hits += ci["lo"] <= true <= ci["hi"]
    assert 0.91 <= hits / reps <= 0.985


def test_logistic_recalibration_matches_direct_optimisation():
    rng = np.random.default_rng(2)
    p, y = _calibrated(3000, rng, slope=0.6, intercept=-0.4)
    x = C.logit(p)

    def nll(beta):
        eta = beta[0] + beta[1] * x
        return np.sum(np.logaddexp(0, eta) - y * eta)

    ref = so.minimize(nll, np.zeros(2), method="BFGS").x
    fit = C.logistic_recalibration(p, y)
    np.testing.assert_allclose([fit["intercept"], fit["slope"]], ref, atol=2e-4)
    assert fit["identifiable"] and fit["slope_lo"] < 0.6 < fit["slope_hi"]
    assert fit["intercept_lo"] < -0.4 < fit["intercept_hi"]


def test_slope_and_intercept_diagnose_miscalibration():
    rng = np.random.default_rng(3)
    p, y = _calibrated(6000, rng)
    good = C.logistic_recalibration(p, y)
    assert good["slope_lo"] < 1 < good["slope_hi"] and good["intercept_lo"] < 0 < good["intercept_hi"]
    over = 1 / (1 + np.exp(-2.0 * C.logit(p)))                    # forecasts twice as extreme as warranted
    f_over = C.logistic_recalibration(over, y)
    assert abs(f_over["slope"] - 0.5) < 0.05 and f_over["slope_hi"] < 1.0
    high = 1 / (1 + np.exp(-(C.logit(p) + 1.0)))                  # uniformly too optimistic
    f_high = C.logistic_recalibration(high, y)
    assert abs(f_high["intercept"] + 1.0) < 0.12 and f_high["intercept_hi"] < 0.0
    assert C.ece(p, y) < 0.03 < C.ece(over, y) and C.ece(high, y) > 0.1
    assert C.brier(p, y).mean() < C.brier(over, y).mean() and C.binary_nll(p, y).mean() < C.binary_nll(high, y).mean()
    # not identifiable: a single class, or a constant forecast
    assert not C.logistic_recalibration(p[:50], np.ones(50))["identifiable"]
    assert not C.logistic_recalibration(np.full(50, 0.8), y[:50])["identifiable"]


def test_reliability_bins_and_ece_by_hand():
    p = [0.05, 0.15, 0.15, 0.85, 0.95, 0.95]
    y = [0, 0, 1, 1, 1, 0]
    bins = C.reliability_bins(p, y, n_bins=10)
    assert sum(b["n"] for b in bins) == 6 and [b["n"] for b in bins] == [1, 2, 0, 0, 0, 0, 0, 0, 1, 2]
    assert math.isnan(bins[2]["mean_p"]) and bins[1]["frac_pos"] == 0.5
    expect = (1 * 0.05 + 2 * abs(0.15 - 0.5) + 1 * abs(0.85 - 1) + 2 * abs(0.95 - 0.5)) / 6
    np.testing.assert_allclose(C.ece(p, y), expect, rtol=1e-12)
    qb = C.reliability_bins(np.linspace(0.01, 0.99, 100), np.zeros(100), n_bins=4, strategy="quantile")
    assert [b["n"] for b in qb] == [25, 25, 25, 25]
    assert C.reliability_bins([1.0, 0.0], [1, 0], n_bins=5)[-1]["n"] == 1      # p = 1 lands in the last bin


def test_ece_interval_separates_calibrated_from_overconfident():
    rng = np.random.default_rng(4)
    p, y = _calibrated(1500, rng)
    over = 1 / (1 + np.exp(-2.5 * C.logit(p)))
    ci_good, ci_bad = C.ece_ci(p, y, seed=1), C.ece_ci(over, y, seed=1)
    assert ci_good["lo"] <= ci_good["ece"] <= ci_good["hi"] and ci_good["hi"] < ci_bad["lo"]
    assert ci_bad["ece"] > 0.08


def test_grouped_report_by_family_and_horizon():
    rng = np.random.default_rng(5)
    p1, y1 = _calibrated(1500, rng)
    p2, y2 = _calibrated(1500, rng)
    p2 = 1 / (1 + np.exp(-3.0 * C.logit(p2)))                     # the second family is over-confident
    fam = ["adoption"] * 1500 + ["build"] * 1500
    hor = list(rng.choice([600.0, 7200.0, 3 * 86400.0], 3000))
    rep = C.grouped_binary_report(np.concatenate([p1, p2]), np.concatenate([y1, y2]), fam, hor)
    assert set(rep["by_family"]) == {"adoption", "build"} and set(rep["by_horizon"]) == {"<=1h", "<=1d", "<=1w"}
    assert len(rep["by_family_horizon"]) == 6 and rep["overall"]["n"] == 3000
    a, b = rep["by_family"]["adoption"], rep["by_family"]["build"]
    assert a["calibration"]["slope_lo"] < 1 < a["calibration"]["slope_hi"]
    assert b["calibration"]["slope_hi"] < 0.5 and b["ece"]["ece"] > 3 * a["ece"]["ece"]
    assert b["brier"]["lo"] > a["brier"]["hi"]
    assert sum(v["n"] for v in rep["by_family_horizon"].values()) == 3000
    assert C.horizon_bucket(3600.0) == "<=1h" and C.horizon_bucket(1e9) == ">30d"
    assert C.horizon_bucket(50.0, edges=(10.0, 100.0)) == "<=100s"


def _record(i, dist, made_at, horizon=3600.0, resolved_at=None, value=None, status=OutcomeStatus.PENDING, exp=""):
    q = OutcomeQuery(f"q{i}", "metric", horizon_s=horizon, issued_at=made_at, experiment_id=exp)
    return PredictionRecord(f"p{i}", q, dist, made_at, resolved_at=resolved_at, resolved_value=value,
                            resolved_status=status)


def test_prospective_guard_and_record_scoring():
    C.prospective_guard(10.0, 10.5)
    for made, out in ((10.0, 10.0), (11.0, 10.0), (10.0, None)):
        with pytest.raises(C.RetrospectivePredictionError):
            C.prospective_guard(made, out)
    with pytest.raises(C.RetrospectivePredictionError):
        C.prospective_guard(1.0, 5.0, registered_at=5.0)          # registered only when the outcome arrived

    rng = np.random.default_rng(6)
    recs = []
    for i in range(400):
        p = float(rng.uniform(0.05, 0.95))
        y = float(rng.uniform() < p)
        recs.append(_record(i, D.Bernoulli.from_prob(p).to_record(), 100.0 + i, resolved_at=100.0 + i + 3600,
                            value=y, status=OutcomeStatus.OBSERVED, exp=f"e{i // 4}"))
    for i in range(400, 700):
        mu = float(rng.normal())
        recs.append(_record(i, D.Normal(mu, 2.0).to_record(), 50.0, horizon=86400.0, resolved_at=90000.0,
                            value=float(mu + 2.0 * rng.normal()), status=OutcomeStatus.OBSERVED))
    recs.append(_record(900, D.Bernoulli.from_prob(0.5).to_record(), 5.0))                              # pending
    recs.append(_record(901, D.Normal(0.0, 1.0).to_record(), 5.0, resolved_at=9.0, status=OutcomeStatus.UNOBSERVABLE))
    hz = D.DiscreteHazard([0.0, 0.0], [0.0, 10.0, 20.0]).to_record()
    recs.append(_record(902, hz, 5.0, horizon=20.0, resolved_at=25.0, value=20.0, status=OutcomeStatus.CENSORED))
    rep = C.score_predictions(recs, family_of=lambda r: "fam_a")
    assert rep["n_scored"] == 701 and rep["skipped"] == 2 and rep["refused"] == 0
    cal = rep["binary"]["overall"]
    assert cal["n"] == 400 and cal["calibration"]["slope_lo"] < 1 < cal["calibration"]["slope_hi"]
    assert cal["brier"]["n"] == 100                                 # intervals count experiments, not rows
    key = "normal|fam_a|<=1d"
    assert abs(rep["coverage"][key]["cover90"] - 0.9) < 0.06 and abs(rep["coverage"][key]["cover50"] - 0.5) < 0.09
    assert abs(rep["nll"][key]["mean"] - (0.5 * math.log(2 * math.pi * math.e * 4.0))) < 0.12
    np.testing.assert_allclose(rep["nll"]["discrete_hazard|fam_a|<=1h"]["mean"], -2 * math.log(0.5))

    # a prediction "made" at or after its outcome is refused: strictly by default, counted otherwise; never scored
    late = _record(950, D.Bernoulli.from_prob(0.99).to_record(), 200.0, resolved_at=200.0, value=1.0,
                   status=OutcomeStatus.OBSERVED)
    with pytest.raises(C.RetrospectivePredictionError):
        C.score_predictions(recs + [late])
    soft = C.score_predictions(recs + [late], strict=False, family_of=lambda r: "fam_a")
    assert soft["refused"] == 1 and soft["n_scored"] == 701 and soft["binary"]["overall"]["n"] == 400
    with pytest.raises(C.RetrospectivePredictionError):
        C.score_predictions(recs[:5], registered_at=lambda r: r.resolved_at)
