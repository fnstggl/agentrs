"""Unit and integration tests of the calibration package: metrics, portable predictors, the model registry,
dataset acquisition and verification, prior-day feature counts, the fidelity register parser and the gate audit.

These tests need no calibration dataset; they use the committed checkpoints and small generated inputs (generated
inputs exercise code paths only; they never count as calibration)."""
import hashlib
import importlib
import json
import os

import numpy as np
import pytest

from crucible.calibration import audit, fidelity as FI, fitlib as FL, metrics as M
from crucible.calibration import predictors as P, registry as R
from crucible.calibration.kuairand import PriorCounts

DS = importlib.import_module("crucible.calibration.datasets")      # the module (the package exports the function)


# ---------------------------------------------------------------- metrics
def test_binary_metrics_match_reference_implementations():
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    p[::7] = np.round(p[::7], 1)                         # ties
    assert M.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert M.log_loss(y, p) == pytest.approx(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), abs=1e-9)
    assert M.brier(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)


def test_ece_and_reliability():
    y = np.array([0, 0, 1, 1, 1, 1], float)
    p = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    # bins: 0.1 -> obs 1/3, 0.9 -> obs 1.0
    assert M.ece(y, p) == pytest.approx(0.5 * abs(0.1 - 1 / 3) + 0.5 * abs(0.9 - 1.0))
    assert M.ece(np.array([0, 1.0]), np.array([0.0, 1.0])) == 0.0


def test_group_ranking_known_case():
    y = np.array([1, 0, 0, 0, 1, 0], float)
    p = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.8])
    g = np.array([0, 0, 0, 1, 1, 1])
    r = M.group_ranking(y, p, g)
    assert r["impressions"] == 2
    assert r["impression_auc"] == pytest.approx((1.0 + 0.0) / 2)
    assert r["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_quantile_metrics_and_sampling_are_consistent():
    lv = np.array([0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98])
    rng = np.random.default_rng(2)
    Q = np.sort(rng.normal(size=(400, len(lv))), axis=1)
    y = rng.normal(size=400)
    assert np.allclose(M.pit_fast(y, Q, lv), M.pit(y, Q, lv))
    u = rng.random(400)
    s = P.interpolate_quantiles(Q, lv, u)
    assert np.allclose(M.pit_fast(s, Q, lv), u, atol=1e-9)          # sampling inverts the PIT convention
    # a perfectly specified quantile model has a uniform PIT
    z = rng.normal(size=20000)
    from scipy.stats import norm
    Qn = np.tile(norm.ppf(lv), (len(z), 1))
    assert M.ks_uniform(M.pit_fast(z, Qn, lv)) < 0.02
    assert M.coverage(z, Qn, lv, 0.1, 0.9) == pytest.approx(0.8, abs=0.02)
    assert M.pinball(np.zeros(3), np.zeros((3, len(lv))), lv) == 0.0


def test_cluster_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(3)
    v = rng.normal(size=3000)
    g = rng.integers(0, 300, 3000)
    b = M.cluster_bootstrap(v, g, n_boot=100, seed=0)
    assert b["lo"] < b["mean"] < b["hi"] and b["clusters"] == 300
    assert b["mean"] == pytest.approx(v.mean())


def test_poisson_deviance_and_distances():
    assert M.poisson_deviance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert M.poisson_deviance([0, 4], [1, 1]) > 0
    a = np.arange(100.0)
    assert M.wasserstein1(a, a + 2) == pytest.approx(2.0, abs=1e-9)
    assert M.ks_two_sample(a, a) == 0.0
    assert M.total_variation([1, 1], [1, 1]) == 0.0


# ---------------------------------------------------------------- predictors
def test_tree_export_reproduces_sklearn_with_missing_values():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6000, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) * np.nan_to_num(X[:, 1]) + rng.normal(size=6000) > 0).astype(int)
    m = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X, y)
    te = P.TreeEnsemble.from_sklearn(m)
    assert np.max(np.abs(te.predict(X) - m.predict_proba(X)[:, 1])) < 1e-9
    te2 = P.TreeEnsemble.from_params({k: np.asarray(v) for k, v in te.params().items()})
    assert np.array_equal(te2.predict(X), te.predict(X))


def test_logistic_fit_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 3)) * [1, 10, 0.1] + [0, 5, 1]
    y = (X[:, 0] + 0.1 * X[:, 1] + rng.normal(size=3000) > 0.5).astype(int)
    lg = FL.fit_logistic(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    ref = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y).predict_proba(Z)[:, 1]
    assert np.max(np.abs(lg.predict(X) - ref)) < 1e-6


def test_quantile_network_is_monotone_and_matches_torch():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(4000, 2))
    y = X[:, 0] + np.exp(0.5 * X[:, 1]) * rng.normal(size=4000)
    lv = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qn, hist = FL.fit_quantile_net(X[:3000], y[:3000], X[3000:], y[3000:], lv, hidden=(8,), epochs=5, batch=512,
                                   threads=1)
    Q = qn.predict(X[3000:])
    assert np.all(np.diff(Q, axis=1) >= 0)
    assert hist["epochs"] >= 1
    # heteroscedasticity is learned: the interval widens with X[:, 1]
    hi, lo = X[3000:, 1] > 1, X[3000:, 1] < -1
    width = Q[:, -1] - Q[:, 0]
    assert width[hi].mean() > width[lo].mean()


def test_binned_quantiles_and_gamma_poisson():
    lv = np.array([0.1, 0.5, 0.9])
    X = np.column_stack([np.r_[np.zeros(500), np.ones(500)], np.r_[np.zeros(250), np.ones(250), np.zeros(250),
                                                                  np.ones(250)]])
    y = X[:, 0] * 10 + X[:, 1] * 100 + np.tile(np.arange(250.0), 4) / 250
    b = FL.fit_binned_quantiles(X, y, 0, lv, n_bins=2, group_col=1)
    out = b.predict(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float))
    assert out[:, 1] == pytest.approx([0.5, 10.5, 100.5, 110.5], abs=0.01)
    g = P.GammaPoisson(2.0, 4.0)
    assert g.predict(np.array([[3.0, 6.0]]))[0] == pytest.approx(5 / 10)
    assert g.draw_rates(200000, np.random.default_rng(0)).mean() == pytest.approx(0.5, rel=0.01)


# ---------------------------------------------------------------- registry
def test_registry_round_trip_and_tamper_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MODELS_DIR", str(tmp_path))
    pred = P.Logistic([1.0, -2.0], 0.5, [0, 0], [1, 1])
    card = {"domain": "content_selection", "features": ["a", "b"], "target": "t",
            "provenance": {"identification": "observational", "datasets": ["kuairand_pure"]}, "limits": "x"}
    R.save_model("toy", 3, pred, card)
    f = R.load("toy@3")
    X = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert np.allclose(f.predict(X), pred.predict(X))
    assert R.list_models() == ["toy@3"]
    with pytest.raises(KeyError):
        R.load("toy@2")
    with pytest.raises(ValueError):
        f.predict(np.ones((2, 3)))
    ck = tmp_path / "toy" / "params.npz"
    data = bytearray(ck.read_bytes())
    data[-10] ^= 1
    ck.write_bytes(bytes(data))
    with pytest.raises(IOError):
        R.load("toy")
    with pytest.raises(ValueError):
        R.save_model("bad", 1, pred, dict(card, domain="vibes"))
    with pytest.raises(ValueError):
        R.save_model("bad", 1, pred, dict(card, provenance={"identification": "guess"}))
    big = P.Logistic(np.random.default_rng(0).random(1_500_000), 0.0, np.zeros(1_500_000), np.ones(1_500_000))
    with pytest.raises(ValueError):
        R.save_model("big", 1, big, card)
    assert not (tmp_path / "big" / "params.npz").exists()


def test_committed_models_load_and_predict():
    names = R.list_models()
    assert len(names) >= 9
    for n in names:
        f = R.load(n)
        assert isinstance(f, R.FittedModel) and f.id == n
        assert f.provenance["datasets"] == f.datasets and f.datasets and f.limits and f.features
        assert f.domain in R.DOMAINS and f.identification in R.IDENTIFICATION
        assert set(f.features) <= set(f.feature_map)                    # every input has a simulator mapping
        assert R.card_problems(f.card) == [], (n, R.card_problems(f.card))
        q = f.card.get("feature_quantiles")
        if q:
            X = np.array([[q[c][i] for c in f.features] for i in range(len(q["levels"]))])
            y = f.predict(X)
            assert np.all(np.isfinite(y))
            if f.kind in ("gbdt_binary", "logistic"):
                assert np.all((y > 0) & (y < 1))


def test_one_registry_serves_both_card_schemas():
    import crucible.calibration as C
    from crucible.calibration import registry_e2
    from crucible.calibration.commercial import CommercialResponseModel
    from crucible.calibration.software import DeepEngagementModel, RepoGrowthModel
    names = C.list_models()
    assert {"content_consumption@1", "content_consumption_feed@1", "watch_duration@1", "return_daily@1",
            "session_rate@1", "news_click@1", "commercial_response@1", "software_deep_engagement@1",
            "software_repo_growth@1"} <= set(names)
    schemas = {R.card_view(R.read_card(n))["schema"] for n in names}
    assert schemas == set(R.CARD_SCHEMAS)
    # loader-schema cards load through the same entry point as their own classes
    for n, cls in (("commercial_response@1", CommercialResponseModel), ("software_deep_engagement@1",
                                                                        DeepEngagementModel),
                   ("software_repo_growth@1", RepoGrowthModel)):
        f = C.load(n)
        assert type(f) is cls and isinstance(f, C.FittedModel)
        g = registry_e2.load(n)                                       # the compatibility layer is the same registry
        assert type(g) is cls and np.array_equal(g.predict(f.arrays["probe/X"]), f.predict(f.arrays["probe/X"]))
    ad = C.load("commercial_response@1")
    assert ad.domain == "advertising_response" and ad.view["domain_label"] == "commercial_intervention_response"
    assert ad.identification == "randomized_controlled_trial"
    assert ad.view["identification_label"] == "randomized_experiment"
    # both sampling argument orders give the same draws
    growth = C.load("software_repo_growth@1")
    X = growth.arrays["probe/X"][:50]
    assert np.array_equal(growth.sample(X, np.random.default_rng(1)), growth.sample(np.random.default_rng(1), X))
    # every dataset a card names is served by the one datasets()
    ds = C.datasets()
    assert {d for n in names for d in R.card_view(R.read_card(n))["datasets"]} <= set(ds)
    with pytest.raises(R.CalibrationError):
        C.load("commercial_response@2")
    with pytest.raises(R.CalibrationError):
        registry_e2.load("commercial_response")                       # E2's strict name form is kept


def test_card_views_agree_with_both_schemas():
    for n in R.list_models():
        card = R.read_card(n)
        v = R.card_view(card)
        sel = v["selection"]
        assert v["files"] == R.card_files(card) and R.checkpoint_file(card) in v["files"]
        assert sel["kept"] in sel["test"] and sel["primary_metric"] in card["metrics"]
        assert card["metrics"][sel["primary_metric"]] == pytest.approx(sel["test"][sel["kept"]], abs=1e-12)
        assert v["plans"] and all(p["path"] and p["commit"] and p["sha256"] for p in v["plans"])
        assert v["mechanisms"] and v["report"]
    e2 = R.card_view(R.read_card("software_repo_growth"))
    assert [p["role"] for p in e2["plans"]] == ["plan", "amendment"]
    assert e2["selection"]["primary_metric"] == "mean_poisson_deviance" and e2["selection"]["baselines"] == ["proportional"]
    assert R.card_view(R.read_card("commercial_response"))["selection"]["primary_metric"] == \
        "primary_pooled_logloss_visit_plus_conversion"
    with pytest.raises(R.CardError):
        R.card_schema({"schema": "something/else", "name": "x"})


def test_watch_model_samples_plausible_watch_times():
    f = R.load("watch_duration")
    q = f.card["feature_quantiles"]
    X = np.tile([[q[c][5] for c in f.features]], (5000, 1))
    X[:, f.features.index("consumed")] = 1
    w1 = np.expm1(np.maximum(f.sample(X, np.random.default_rng(0)), 0))
    X[:, f.features.index("consumed")] = 0
    w0 = np.expm1(np.maximum(f.sample(X, np.random.default_rng(0)), 0))
    assert np.median(w1) > 3 * np.median(w0)               # consumed exposures are watched much longer than skips


# ---------------------------------------------------------------- datasets
def test_dataset_entries_are_complete():
    ds = DS.datasets()
    assert {"kuairand_pure", "mind_small", "criteo_uplift_v2.1", "gharchive_2024-10-07_96h"} <= set(ds)
    for n, e in ds.items():
        assert all(len(f["sha256"]) == 64 and f["url"].startswith("https://") and f["path"] for f in e["files"]), n
        assert audit.provenance_complete(e), n
        assert e["synthetic"] is False
        assert e["sha256"] and e["size"] > 0 and e["schema"]
    # manifest-backed entries keep the manifest's own fields next to the normalised ones
    cr = ds["criteo_uplift_v2.1"]
    assert cr["manifest"] == "docs/v5/calibration/datasets/criteo_uplift_v2.1.json"
    assert cr["license"]["spdx"] == cr["license"]["name"] == "CC-BY-NC-SA-4.0" and cr["license"]["text_location"]
    assert cr["files"][0]["file"] == cr["files"][0]["path"] and cr["files"][0]["bytes"] == cr["files"][0]["size"]
    assert cr["acquirer"] == "crucible.calibration.criteo" and cr["extracted"]
    assert len(ds["gharchive_2024-10-07_96h"]["files"]) == 96


def test_acquire_verifies_checksums(tmp_path, monkeypatch):
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n3,4\n")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    entry = {"title": "toy", "files": [{"path": "download/x.csv", "url": src.as_uri(), "sha256": sha}],
             "schema": {"download/x.csv": {"format": "csv", "header": True, "columns": ["a", "b"], "rows": 2}},
             "license": {"name": "n", "text_location": "t"}, "provenance": {"publisher": "p"}}
    monkeypatch.setitem(DS._DATASETS, "toy_ok", dict(entry))
    d = DS.acquire("toy_ok", dest=str(tmp_path / "ok"))
    assert DS.verify("toy_ok", d) == {"download/x.csv": "ok"}
    bad = dict(entry, files=[dict(entry["files"][0], sha256="0" * 64)])
    monkeypatch.setitem(DS._DATASETS, "toy_bad", bad)
    with pytest.raises(IOError):
        DS.acquire("toy_bad", dest=str(tmp_path / "bad"))
    assert not (tmp_path / "bad" / "download" / "x.csv").exists()
    wrong = dict(entry, schema={"download/x.csv": {"format": "csv", "header": True, "columns": ["a", "b"],
                                                   "rows": 3}})
    monkeypatch.setitem(DS._DATASETS, "toy_schema", wrong)
    with pytest.raises(IOError):
        DS.acquire("toy_schema", dest=str(tmp_path / "ok"))
    with pytest.raises(ValueError):
        DS.register("toy_nosha", dict(entry, files=[{"path": "x", "url": "https://e"}]))


# ---------------------------------------------------------------- features
def test_prior_counts_match_brute_force():
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 30, 4000)
    days = rng.integers(0, 31, 4000)
    w = rng.random(4000)
    pc = PriorCounts(keys, days, w)
    qk = rng.integers(0, 32, 500)
    qd = rng.integers(0, 32, 500)
    exp_all = np.array([w[(keys == k) & (days < d)].sum() for k, d in zip(qk, qd)])
    exp_7 = np.array([w[(keys == k) & (days < d) & (days >= d - 7)].sum() for k, d in zip(qk, qd)])
    assert np.allclose(pc.before(qk, qd), exp_all)
    assert np.allclose(pc.window(qk, qd, 7), exp_7)


# ---------------------------------------------------------------- fidelity register and audit
def test_fidelity_register_parser(tmp_path):
    md = tmp_path / "r.md"
    md.write_text("intro\n\n| mechanism | module | category | evidence (dataset, period) | held-out result | limits |\n"
                  "|---|---|---|---|---|---|\n| a.b | pop | plausible, unvalidated | none | | assumed |\n\nafter\n"
                  "| other | table |\n")
    rows = FI.v5_rows(str(md))
    assert len(rows) == 1 and rows[0]["mechanism"] == "a.b" and rows[0]["limits"] == "assumed"
    assert FI.record_problems(rows[0]) == []


def test_v40_register_maps_to_v5_categories():
    recs = FI.v40_records()
    assert len(recs) == 55
    assert {r["category"] for r in recs} <= set(FI.CATEGORIES)
    beh = [r for r in recs if "behavioural" in r["kinds"]]
    assert beh and all(FI.record_problems(r) == [] for r in beh)


def test_audit_rejects_models_without_empirical_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MODELS_DIR", str(tmp_path))
    pred = P.Constant(0.3)
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"heldout_splits": ["test"], "splits": {"test": {"rows": 10}},
                               "calibration": {"ece": 0.1}, "extrapolation_limits": "x" * 300,
                               "baseline_comparison": {}}))
    base = {"domain": "content_selection", "features": ["a"], "target": "t", "limits": "x" * 300,
            "report": str(rep), "selection": {}}
    R.save_model("synthetic", 1, pred, dict(base, provenance={"identification": "observational",
                                                               "datasets": ["made_up_fixture"]}))
    R.save_model("real", 1, pred, dict(base, provenance={"identification": "observational",
                                                          "datasets": ["kuairand_pure"]}))
    recs = {r["name"]: r for r in audit.all_records()}
    assert not recs["synthetic"]["qualifies"] and recs["real"]["qualifies"]
    m = audit.gate_f_metrics(list(recs.values()))
    assert m["empirical_domains_fitted"] == 1 and m["synthetic_only_domains"] == 0
    only_synth = audit.gate_f_metrics([recs["synthetic"]])
    assert only_synth["synthetic_only_domains"] == 1 and only_synth["empirical_domains_fitted"] == 0
    assert m["baseline_comparisons"] == 0                     # no baseline comparison in the report


def test_audit_reads_loader_schema_reports():
    rec = audit.examine("software_repo_growth")
    rep = rec["report"]
    assert rec["schema"] == R.LOADER_SCHEMA and rec["qualifies"], rec["problems"]
    assert rep["heldout_splits"] == ["test", "test_new_repos"] and rep["splits"]["test"]["rows"] == 321303
    bc = rep["baseline_comparison"]
    card = rec["card"]
    assert bc["baseline"] == "proportional" and bc["kept"] == card["selection"]["kept"]
    assert bc["kept_test"] == card["metrics"]["mean_poisson_deviance"]
    assert bc["baseline_test"] == card["baseline_metrics"]["mean_poisson_deviance"]
    assert bc["ci95"][1] < 0 and bc["difference"] == pytest.approx(bc["kept_test"] - bc["baseline_test"])
    assert audit.calibration_tables(rep) == {"calibration_by_decile": 10} and rec["calibration_reported"]
    com = audit.examine("commercial_response")
    assert com["identification"] == "randomized_controlled_trial" and com["domain"] in audit.COMMERCIAL_DOMAINS
    assert com["report"]["splits"]["test"]["rows"] == 2795919
    assert {"visit_ece", "conversion_ece"} <= set(audit.calibration_values(com["report"]))
    assert audit.gate_f_metrics([rec, com], datasets_scope="cited")["datasets_with_checksums"] == 2
