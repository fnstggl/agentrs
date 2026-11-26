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


# ---------------------------------------------------------------- runtime ad_response@2 (the fitted checkpoint)
def test_ad_response_2_serves_the_fitted_shift_rowwise_with_attention():
    from crucible.calibration import runtime as RT
    from crucible.calibration.runtime import ContractError
    dm = RT.get("ad_response@2")
    assert dm.model_id == "ad_exposure_response@1" and dm.served == "K"
    D = float(dm.fitted.predictor.a)
    assert 1.0 < D < 2.0                                   # the kept log-odds shift of the card
    p0 = np.array([0.02, 0.05, 0.1, 0.2, 0.35])
    full = dm.evaluate({"organic_visit_probability": p0, "attended_probability": np.ones(5)})
    expect = 1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + D)))
    np.testing.assert_allclose(full.values[:, 0], expect, rtol=1e-12)
    assert not full.extrapolated.any()                     # inside the data's organic-propensity support
    # individual state: the absolute increment grows with the individual's own organic propensity here
    assert np.all(np.diff(full.values[:, 2]) > 0)
    # attention scales the increment; no attention, no effect; never below the organic probability
    half = dm.evaluate({"organic_visit_probability": p0, "attended_probability": np.full(5, 0.5)})
    np.testing.assert_allclose(half.values[:, 2], 0.5 * full.values[:, 2], rtol=1e-12)
    none = dm.evaluate({"organic_visit_probability": p0, "attended_probability": np.zeros(5)})
    np.testing.assert_allclose(none.values[:, 0], p0)
    # row independence: a row alone gives the same value
    one = dm.evaluate({"organic_visit_probability": p0[2:3], "attended_probability": np.ones(1)})
    assert one.values[0, 0] == full.values[2, 0]
    # coupled draws: incremental visits are visits that would not have happened organically
    rng = np.random.default_rng(0)
    n = 200_000
    dr = dm.sample({"organic_visit_probability": np.full(n, 0.1), "attended_probability": np.ones(n)}, rng)
    visit, inc = dr[:, 0], dr[:, 1]
    assert abs(visit.mean() - full.values[2, 0]) < 0.005 and abs(inc.mean() - full.values[2, 2]) < 0.005
    assert np.all(visit[inc])
    # provenance and scope
    prov = full.provenance()
    assert prov["decision"] == "ad_response@2" and prov["model"] == "ad_exposure_response@1"
    dm.check_scope(use="feed_campaign")
    with pytest.raises(ContractError):
        dm.check_scope(use="search_result")
    with pytest.raises(ContractError):
        dm.evaluate({"organic_visit_probability": p0})
    assert dm.spec.params["incremental_conversion_ratio"] > 1.0


def test_at_least_one_attended_display_is_per_individual():
    from crucible.calibration import runtime as RT
    a = RT.at_least_one(np.array([0.2, 0.2, 0.5]), np.array([0, 0, 2]), 3)
    np.testing.assert_allclose(a, [1 - 0.8 ** 2, 0.0, 0.5])
    # repetition saturates: more displays raise the probability with diminishing increments
    steps = [RT.at_least_one(np.full(k, 0.3), np.zeros(k, int), 1)[0] for k in range(1, 6)]
    assert np.all(np.diff(steps) > 0) and np.all(np.diff(np.diff(steps)) < 0)


# ---------------------------------------------------------------- in-world validation harness (model-only driver)
def test_model_only_driver_compares_fitted_behaviour_with_heldout_users():
    from crucible.calibration import inworld as IW
    r = IW.model_only_reports(beyond_days=30)
    assert r["_meta"]["individuals"] > 1000
    for k in ("activity_distribution", "retention_curve", "heterogeneity", "long_horizon_drift"):
        assert isinstance(r[k]["within_tolerance"], bool)
    # the statistics are measured, not constant: total variation and KS are inside (0, 1)
    assert 0 < r["activity_distribution"]["active_days_tv"] < 1
    assert 0 < r["heterogeneity"]["user_active_share_ks"] < 1
    assert r["long_horizon_drift"]["beyond_data"]["days"] == 37


def test_compare_detects_a_distorted_simulation():
    """A simulation that is the reference with one in three active days removed is outside the tolerances."""
    from crucible.calibration import inworld as IW
    rng = np.random.default_rng(0)
    A = rng.random((3000, 21)) < 0.7
    S = np.where(A, 1 + rng.poisson(1.0, A.shape), 0)
    ref = IW.activity_stats(A, S, 14, 7)
    B = A & (rng.random(A.shape) > 0.33)
    sim = IW.activity_stats(B, np.where(B, S, 0), 14, 7)
    r = IW.compare(sim, ref)
    assert not r["activity_distribution"]["within_tolerance"] and not r["long_horizon_drift"]["within_tolerance"]
    same = IW.compare(ref, ref)
    assert same["activity_distribution"]["within_tolerance"] and same["heterogeneity"]["within_tolerance"]


def test_world_without_wired_models_is_reported_not_validated():
    from crucible.calibration import inworld as IW
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World, WorldSpec
    w = World.create(WorldSpec("inworld-test", 1, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]),
                     "w-inworld")
    w.advance_time(3600)
    try:
        with pytest.raises(IW.NotWired):
            IW.world_activity(w)
        m = IW.provenance_metrics(w)
        assert m["fitted_models_used_in_world"] == 0
    finally:
        w.close()


def test_out_of_scope_check_counts_direct_checkpoint_loads():
    from crucible.calibration import inworld as IW
    r = IW.out_of_scope_uses()
    assert set(r["direct_checkpoint_loads"]) <= {"population/models.py"}     # the population's silent plug-in
    assert r["out_of_card_scope_uses"] == len(r["direct_checkpoint_loads"])
