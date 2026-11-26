"""E5 advertising-response redesign: the instrumental-variable estimator on a synthetic randomised trial with a known
display effect, the predictor kind, and the runtime decision model ad_response@2 (workstream E)."""
import numpy as np
import pytest

from crucible.calibration import fit_e5, predictors_e5 as P5


def synthetic_trial(n=1_500_000, shift=1.2, seed=3, lab_p=(0.05, 0.8, 0.15), e0=0.1, e1=0.8):
    """A Criteo-like trial: 85% eligible; display probability rising with the organic propensity (a bidder that
    targets likely visitors); displayed eligible users visit with logit(p) + shift, everyone else with p."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 12)).astype(np.float32)
    lin = -3.0 + 0.9 * X[:, 0] - 0.5 * X[:, 3]
    p = 1 / (1 + np.exp(-lin))
    z = rng.random(n) < 0.85
    e = np.clip(e0 + e1 * p, 0, 1)
    D = z & (rng.random(n) < e)
    q = np.where(D, 1 / (1 + np.exp(-(lin + shift))), p)
    y = rng.random(n) < q
    conv = y & (rng.random(n) < 0.06)
    lab = rng.choice(3, size=n, p=list(lab_p)).astype(np.uint8)
    d = {"X": X, "treatment": z.astype(np.int8), "exposure": D.astype(np.int8), "visit": y.astype(np.int8),
         "conversion": conv.astype(np.int8)}
    return d, lab


def _true_propensity(model, dd, rows, sub="visit_control"):
    """The synthetic trial's organic propensity (a pre-treatment function of the features): the score the plan
    estimates with commercial_response@1 on real data; the treated sub-model is not used by the estimator."""
    Xr = np.asarray(dd["X"][rows], float)
    p = 1 / (1 + np.exp(-(-3.0 + 0.9 * Xr[:, 0] - 0.5 * Xr[:, 3])))
    return p if sub == "visit_control" else np.zeros_like(p)


def test_iv_estimator_recovers_a_known_display_shift():
    """Joint three-arm likelihood with profiled nuisances on a synthetic trial whose displayed users visit with
    logit(p) + 1.2 (bidder-style targeting: display probability rises with p)."""
    from crucible.calibration import fit_e5
    d, lab = synthetic_trial()
    S = fit_e5.prepare(d=d, lab=lab, score_fn=_true_propensity, splits=("fit", "test"))
    thetas, preds, center = fit_e5.fit_candidates(S["nz"]["fit"], S["cells"]["fit"])
    D = float(thetas["M1_constant_log_odds"][0])
    assert abs(D - 1.2) < 0.2, D
    ct = S["cells"]["test"]
    ex = {k: fit_e5.excess_log_loss(k, thetas[k], ct) for k in fit_e5.CANDS}
    assert ex["M1_constant_log_odds"] < ex["B0_no_effect"] and ex["M1_constant_log_odds"] < ex["B1_constant_difference"]
    assert min(ex.values()) >= -1e-9                     # never better than the saturated fit
    # the naive displayed-minus-not-displayed contrast is confounded by targeting; the IV estimate is not
    fr = S["fr"]["test"]
    naive = fr["y"][fr["z"] & fr["D"]].mean() - fr["y"][fr["z"] & ~fr["D"]].mean()
    late = (fr["y"][fr["z"]].mean() - fr["y"][~fr["z"]].mean()) / fr["D"][fr["z"]].mean()
    assert naive > 1.1 * late > 0
    # conversion after an incremental visit happens at the organic rate in this trial
    rho = fit_e5.conversion_ratio(fr)
    assert rho["ci95"][0] < 1.0 < rho["ci95"][1]


def test_estimator_finds_no_effect_when_there_is_none():
    from crucible.calibration import fit_e5
    d, lab = synthetic_trial(shift=0.0, seed=5)
    S = fit_e5.prepare(d=d, lab=lab, score_fn=_true_propensity, splits=("fit",))
    thetas, _, _ = fit_e5.fit_candidates(S["nz"]["fit"], S["cells"]["fit"])
    assert float(thetas["M1_constant_log_odds"][0]) < 0.2


def test_predictor_is_rowwise_floored_and_round_trips():
    m = P5.AdExposure("linear_logit", a=0.5, b=-2.0, c=-3.0)
    l0 = P5.logit(np.array([0.001, 0.01, 0.05, 0.3, 0.6]))
    q = m.predict(l0[:, None])
    assert np.all(q >= 1 / (1 + np.exp(-l0)) - 1e-15)            # never below the organic probability
    assert m.shift(l0)[-1] == 0.0                                  # floored where a + b (l0 - c) < 0
    # row independence: each row alone gives the same value
    assert np.allclose([m.predict(l0[i:i + 1, None])[0] for i in range(l0.size)], q)
    m2 = P5.AdExposure.from_params(m.params())
    assert np.allclose(m2.predict(l0[:, None]), q)
    st = P5.AdExposure("step_logit", knots=[-6.0, -3.0], shifts=[2.0, -1.0])
    assert np.allclose(st.shift(np.array([-7.0, -2.0])), [2.0, 0.0])
    with pytest.raises(ValueError):
        P5.AdExposure("bogus")
