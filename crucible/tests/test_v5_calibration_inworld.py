"""In-world validation harness (crucible.calibration.inworld; SP-5.0 / SP-5.5b measurements; workstream E).

The population workstream wires the runtime decision models into the default world (D-15). These tests exercise the
harness on real kernel worlds with the real population module: a test-only activity module whose individuals decide
through the runtime decision models and record each decision with provenance exactly as RUNTIME.md's wiring contract
says (the "wired" fixture), and adversarial worlds that stamp fitted-model provenance on decisions the runtime never
evaluated, or inject them through a system call. Only the wired world's decisions may count."""
import numpy as np
import pytest

import crucible
from crucible import population as P
from crucible.calibration import inworld as IW, runtime
from crucible.core.kernel import Module, World, WorldSpec, register_module

DAY = 86400


def _inputs(ctx, idx, now):
    """daily_return@1 inputs from the individual's own population state (test fixture)."""
    from crucible.substrate import columns
    w = ctx.world
    ns = columns.col(w, "pop", "n_sessions")[idx].astype(float)
    last = columns.col(w, "pop", "last_session_at")[idx].astype(float)
    since = np.where(ns > 0, np.minimum(14.0, np.floor(np.maximum(now - last, 0) / DAY)), 14.0)
    s7 = np.minimum(ns, 40.0)
    return {"active_days_7d": np.minimum(ns, 7.0), "days_since_last_active": since,
            "feed_exposures_7d": 12.0 * s7, "sessions_7d": s7, "active_yesterday": (since <= 1).astype(float),
            "weekend": np.full(idx.size, float(((now // DAY) + 4) % 7 >= 5))}


def _declare(world, cols, owner):
    """The decision columns the harness reads, declared on the population table by the fixture (the population
    declares its own when it wires the models); the population's tick may write them."""
    from crucible.substrate import columns
    columns.declare(world, "pop", columns.rows(world, "pop"), cols)
    columns.grant(world, "pop", list(cols), "pop")


@register_module
class _Wired(Module):
    """Test fixture: a population activity whose decisions go through the runtime decision models (daily return,
    sessions per active day, a campaign with randomised eligibility through ad_response@2) and are recorded with
    population.record_decisions inside the population's own session."""
    name = "calib_wired_test"
    host = "calib_wired_test"
    version = "1"
    depends = ("pop",)

    COLS = {"active": ("bool", (), False), "sessions": ("int32", (), 0), "visit": ("bool", (), False),
            "displayed": ("bool", (), False), "organic_visit_probability": ("float32", (), 0.0),
            "attended_probability": ("float32", (), 0.0)}

    def install(self, world):
        _declare(world, self.COLS, self.name)
        P.register_activity(world, "calibwired", self.act, owner=self.name, prior=4.0)

    def act(self, ctx, idx, now):
        from crucible.substrate import columns
        rng = ctx.rng("calibwired")
        res = runtime.get("daily_return@1").evaluate(_inputs(ctx, idx, now))
        active = rng.random(idx.size) < res.values
        pv = res.provenance()
        P.record_decisions(ctx, "calibwired.daily_active", idx, {"active": active}, model_id=pv["model"],
                           model_version=str(pv["version"]), fidelity_id=pv["fidelity"])
        ns = columns.col(ctx.world, "pop", "n_sessions")[idx].astype(float)
        ses = runtime.get("extra_sessions@1")
        extra = ses.sample({"extra_sessions_observed": np.zeros(idx.size), "active_days_observed": np.minimum(ns, 7)},
                           rng)
        P.record_decisions(ctx, "calibwired.sessions", idx, {"sessions": (1 + extra).astype(np.int32)},
                           model_id="extra_sessions", model_version="1", fidelity_id=ses.fidelity_id)
        # a campaign: eligibility randomised per individual, displays to 40% of the eligible, attention 0.3
        prefs = columns.col(ctx.world, "pop", "prefs")[idx].astype(float)
        p0 = np.clip(0.03 + 0.25 * prefs[:, 0], 0.02, 0.3)
        z = rng.random(idx.size) < 0.5
        disp = z & (rng.random(idx.size) < 0.4)
        att = np.where(disp, 0.3, 0.0)
        ad = runtime.get("ad_response@2").scoped(use="display_campaign")
        dr = ad.sample({"organic_visit_probability": p0, "attended_probability": att}, rng)
        for arm, m in (("eligible", z), ("holdout", ~z)):
            if m.any():
                P.record_decisions(ctx, f"calibwired.ad.{arm}", idx[m],
                                   {"visit": dr[m, 0], "displayed": disp[m], "organic_visit_probability": p0[m],
                                    "attended_probability": att[m]},
                                   model_id="ad_response", model_version="2", fidelity_id=ad.fidelity_id,
                                   extra={"experiment": {"id": "ad_holdout", "arm": arm}})
        return np.full(idx.size, 0.2)


@register_module
class _Stamp(Module):
    """Adversarial fixture: decisions stamped with daily_return@1 provenance inside the population's session, values
    fabricated (everyone active), the runtime model never evaluated."""
    name = "calib_stamp_test"
    host = "calib_stamp_test"
    version = "1"
    depends = ("pop",)

    def install(self, world):
        _declare(world, {"active": ("bool", (), False)}, self.name)
        P.register_activity(world, "calibstamp", self.act, owner=self.name, prior=4.0)

    def act(self, ctx, idx, now):
        P.record_decisions(ctx, "calibstamp.daily_active", idx, {"active": np.ones(idx.size, bool)},
                           model_id="daily_return", model_version="1", fidelity_id="runtime:daily_return@1")
        return np.full(idx.size, 0.2)


def _world(extra_module: str | None, n=240, seed=5):
    crucible.load_all_modules()
    mods = [{"name": m, "config": {}} for m in ["identity", "bank", "admin", "artifacts", "cols"]]
    mods.append({"name": "pop", "config": {"profile": "development", "n": n, "tick_s": 3600}})
    if extra_module:
        mods.append({"name": extra_module, "config": {}})
    return World.create(WorldSpec("population", seed, mods, {"cost_model": "deterministic"}))


@pytest.fixture(scope="module")
def wired():
    runtime.reset_usage()
    w = _world("calib_wired_test")
    w.advance_time(w.now + 16 * DAY)
    au = IW.audit(w)
    yield w, au, runtime.usage(), runtime.scope_usage()
    w.close()


def test_wired_world_decisions_count_and_agree_with_check_provenance(wired):
    w, au, usage, _ = wired
    m = IW.provenance_metrics(w, au)
    assert set(m["_used"]) >= {"return_daily@1", "session_rate@1", "ad_exposure_response@1"}, m["_used"]
    assert m["fitted_models_used_in_world"] >= 3
    assert m["_forced_with_fitted_provenance"] == {}
    assert all(v["ok"] for v in m["_runtime_backing"].values()), m["_runtime_backing"]
    # the population's own provenance check sees the same organic rows (Audit raises when they diverge)
    cp = P.check_provenance(w, events=w.events_range(0), prefixes=("calibwired.",))
    assert cp["organic_population_rows"] > 0 and cp["forced_rows"] == 0
    # the population's own session and activation decisions carry no provenance yet: counted, by kind
    assert m["decision_events_without_model_provenance"] > 0
    assert {"pop.session", "pop.activation"} & set(m["_without_by_kind"])
    assert not any(k.startswith(("attention.", "pop.init")) for k in m["_without_by_kind"])


def test_wired_world_gives_four_reports_and_the_ad_holdout_matches(wired):
    w, au, _, _ = wired
    r = IW.in_world_reports(w, au=au)
    for k in ("activity_distribution", "retention_curve", "heterogeneity", "long_horizon_drift"):
        assert isinstance(r[k]["within_tolerance"], bool)
    from crucible.substrate import columns
    assert r["_meta"]["individuals"] == columns.rows(w, "pop")
    ad = IW.ad_holdout_experiment(w, au=au)
    assert ad["protocol_violations"] == {"displayed_in_holdout": 0, "attention_without_display": 0}
    assert ad["mean_attention_displayed"] == pytest.approx([0.3] * ad["strata"])
    assert ad["matching"], ad          # the world's display effect per attended display is Criteo's
    with pytest.raises(IW.NotWired):
        IW.random_exposure_experiment(w, au=au)


def test_scoped_ad_use_is_in_the_scope_ledger(wired):
    _, _, _, led = wired
    assert led["used"]["ad_response@2"] == {"content_kind=None;use=display_campaign": led["used"]["ad_response@2"][
        "content_kind=None;use=display_campaign"]}
    r = IW.out_of_scope_uses(led)
    assert r["scoped_models_evaluated_without_declared_use"] == {} and r["refused_out_of_scope_attempts"] == {}


def test_stamped_decisions_without_runtime_evaluation_do_not_count():
    """Probe p2 of the verifier: organic batches stamped daily_return@1, values fabricated, runtime never called."""
    runtime.reset_usage()
    w = _world("calib_stamp_test", n=120)
    try:
        w.advance_time(w.now + 16 * DAY)
        au = IW.audit(w)
        m = IW.provenance_metrics(w, au)
        assert m["fitted_models_used_in_world"] == 0, m["_used"]
        assert m["_runtime_backing"] and not any(v["ok"] for v in m["_runtime_backing"].values())
        with pytest.raises(IW.NotWired):
            IW.in_world_reports(w, au=au)
    finally:
        w.close()


def test_system_injected_provenance_is_forced_and_does_not_count():
    """Probe p2d of the verifier: six batches injected through a system transaction, stamped with six fitted
    models' provenance."""
    runtime.reset_usage()
    w = _world(None, n=60)
    try:
        names = ["return_daily", "session_rate", "content_consumption", "watch_duration", "session_length",
                 "news_click"]
        for nm in names:
            w._run_system("inject-" + nm, lambda ctx, nm=nm: P.record_decisions(
                ctx, "test.stamp", np.arange(10), {"last_utility": np.ones(10, np.float32)},
                model_id=nm, model_version="1", fidelity_id=f"model:{nm}@1"))
        # the runtime evaluating the models elsewhere in the process does not make the stamps count either
        runtime.get("daily_return@1").evaluate({k: np.zeros(20) for k in ("active_days_7d", "feed_exposures_7d",
                                                                           "sessions_7d", "active_yesterday",
                                                                           "weekend")} |
                                               {"days_since_last_active": np.full(20, 14.0)})
        m = IW.provenance_metrics(w)
        assert m["fitted_models_used_in_world"] == 0
        assert m["_provenance_classes"] == {"forced": 6}
        assert sum(m["_forced_with_fitted_provenance"].values()) == 6
    finally:
        w.close()


# ---------------------------------------------------------------- the ad-holdout analysis on draws of ad_response@2
def _campaign(attention, shift_world=True, n=500_000, seed=1):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1.2, 12, n), 0.013, 0.39)          # inside the served support of organic propensities
    z = rng.random(n) < 0.5
    D = z & (rng.random(n) < 0.3)
    A = np.where(D, attention, 0.0)
    if shift_world:
        y = runtime.get("ad_response@2").scoped(use="display_campaign").sample(
            {"organic_visit_probability": p, "attended_probability": A}, rng)[:, 0]
    else:
        y = rng.random(n) < p
    return IW.ad_holdout_estimate(z, D, y.astype(float), p, A)


@pytest.mark.parametrize("attention", [1.0, 0.5, 0.19])
def test_attention_adjusted_shift_matches_criteo_at_any_attention(attention):
    """Probe p4 of the verifier: with A_ref = 1 a world display attended with probability A has effect
    A (q - p0); the attention-adjusted estimator targets the runtime's estimand (the shift of an attended display),
    so a world wired as RUNTIME.md says matches Criteo whatever its attention level; the unadjusted shift does not
    at low attention."""
    est = _campaign(attention)
    m = IW.match_criteo(est)
    assert m["matching"], (est, m)
    assert est["shift_ci95"][0] < m["criteo_shift"] < est["shift_ci95"][1]
    if attention < 0.5:
        assert est["shift_unadjusted"] < 0.8 * m["criteo_shift"]


def test_an_uninformative_campaign_never_matches():
    """Wave 3b: a campaign too small to tell the display effect from none has a wide interval, which a z test alone
    would accept; a match also requires the world's interval to exclude a null effect."""
    est = _campaign(0.19, n=3000, seed=3)
    m = IW.match_criteo(est)
    assert est["shift_ci95"][0] == 0.0 and not m["powered"] and not m["matching"], (est, m)
    big = IW.match_criteo(_campaign(0.19))
    assert big["powered"] and big["matching"]


# KuaiRand's random-exposure reference as measured (open rate 0.186, own-topic log-odds ratio 0.40 on 89,752 rows)
KR_REF = {"open_rate": 0.18645, "open_rate_se": 0.0013, "own_topic_log_odds_ratio": 0.4015,
          "own_topic_log_odds_ratio_se": 0.0181, "rows": 89752}


def _inserted(n, lor, seed=0):
    rng = np.random.default_rng(seed)
    own = rng.random(n) < 0.45
    base = 0.155                                            # other-topic open rate so that the mixture is ~0.186
    logit = np.log(base / (1 - base)) + np.where(own, lor, 0.0)
    return rng.random(n) < 1 / (1 + np.exp(-logit)), own


@pytest.mark.parametrize("n,lor,match", [(60_000, 0.40, True), (60_000, 0.0, False)])
def test_random_exposure_matches_only_an_informative_agreeing_world(n, lor, match):
    """Wave 3b: KuaiRand-like random insertions match; a world without the own-topic preference does not."""
    y, own = _inserted(n, lor)
    r = IW.random_exposure_estimate(y, own, ref=KR_REF)
    assert r["matching"] is match and r["powered"] is match, r


def test_a_random_exposure_experiment_too_small_to_detect_anything_never_matches():
    """Wave 3b: with 300 insertions the z tests accept KuaiRand's values for most draws (the interval is wide); the
    world's estimate does not detect the own-topic preference there, so it is not a match."""
    accepted = []
    for seed in range(40):
        r = IW.random_exposure_estimate(*_inserted(300, 0.40, seed), ref=KR_REF)
        if abs(r["z_open_rate"]) < 1.96 and abs(r.get("z_log_odds_ratio", 99.0)) < 1.96:
            accepted.append(r)
    assert len(accepted) >= 20                      # a z test alone would call these matches
    weak = [r for r in accepted if not r["powered"]]
    assert weak and not any(r["matching"] for r in weak)


def test_a_world_without_display_effect_does_not_match():
    est = _campaign(0.5, shift_world=False)
    assert not IW.match_criteo(est)["matching"]
    assert est["shift_ci95"][0] < 0.3


def test_out_of_scope_ad_use_is_counted():
    """Probe p3 of the verifier: a search placement is refused, and an unscoped evaluation of a scoped model is
    counted (its scope cannot be verified)."""
    runtime.reset_usage()
    dm = runtime.get("ad_response@2")
    with pytest.raises(runtime.ContractError):
        dm.scoped(use="search_result")
    dm.evaluate({"organic_visit_probability": np.array([0.1]), "attended_probability": np.array([1.0])})
    r = IW.out_of_scope_uses()
    assert r["scoped_models_evaluated_without_declared_use"] == {"ad_response@2": 1}
    assert r["refused_out_of_scope_attempts"] == {"ad_response@2:content_kind=None;use=search_result": 1}
    assert r["out_of_card_scope_uses"] == len(r["static_uses"]) + 2
    runtime.reset_usage()


BYPASSES = {
    "relative calibration.load": "from .. import calibration\nm = calibration.load('news_click@1')",
    "relative from-import load": "from ..calibration import load\nm = load('news_click@1')",
    "absolute from-import load": "from crucible.calibration import load\nm = load('news_click@1')",
    "absolute registry from-import": "from crucible.calibration.registry import load\nm = load('x')",
    "alias registry import": "from ..calibration import registry as reg\nm = reg.load('news_click@1')",
    "aliased package": "import crucible.calibration as c\nc.load('x')",
    "package attribute": "import crucible\ncrucible.calibration.load('x')",
    "getattr": "from .. import calibration as cal\nf = getattr(cal, 'load')",
    "decision_for checkpoint": "from ..calibration import runtime\n"
                               "runtime.decision_for('news_click@1').evaluate_features(X, checkpoint=True)",
    "params.npz": "import numpy as np\nP = np.load('docs/v5/calibration/models/news_click/params.npz')",
    "importlib": "import importlib\nimportlib.import_module('crucible.calibration').load('news_click@1')",
    "fitted predict": "from ..calibration import runtime\nruntime.get('content_open@1').fitted.predict(X)",
}


@pytest.mark.parametrize("name", sorted(BYPASSES))
def test_static_scan_detects_every_bypass(name):
    assert IW.scan_source(BYPASSES[name], "crucible.population"), name


def test_static_scan_ignores_docstrings_comments_and_the_runtime():
    clean = ('"""Uses crucible.calibration.load(name) and params.npz."""\n'
             "# from crucible.calibration import load\n"
             "from ..calibration import runtime\n"
             "from crucible.calibration.runtime import ContractError\n"
             "dm = runtime.get('content_open@1').scoped(content_kind='video_short')\n")
    assert IW.scan_source(clean, "crucible.population") == []


def test_static_scan_of_the_tree():
    r = IW.out_of_scope_uses({"used": {}, "refused": {}})
    # population/models.py's plug-in loads checkpoints directly (getattr(cal, "load")) and predicts from them; its
    # module docstring mentions calibration.load and is not counted
    assert set(r["static_uses"]) <= {"population/models.py"}
    assert all(not s.startswith("10:") for s in r["static_uses"].get("population/models.py", []))
    assert r["out_of_card_scope_uses"] == len(r["static_uses"])
