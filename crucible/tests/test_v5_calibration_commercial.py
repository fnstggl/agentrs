"""Workstream E2: the randomized commercial-response model (Criteo Uplift v2.1), loaded through the registry.

These tests use the committed checkpoint, card and report, and (where stated) the dataset under $CRUCIBLE_DATA. A
missing dataset is a failure, not a skip: run `crucible.calibration.acquire("criteo_uplift_v2.1")`. Models load
through the unified registry (`crucible.calibration.registry`, the module behind `crucible.calibration.load`);
`registry_e2` is only the compatibility layer whose E2-restricted gate metrics are checked at the end.
"""
import importlib
import os

import numpy as np
import pytest

from crucible.calibration import audit, registry_e2
from crucible.calibration import datasets as all_datasets
from crucible.calibration import registry as R
from crucible.calibration import scoring_e2 as S
from crucible.calibration.commercial import ARMS, OUTCOMES, CommercialResponseModel

MID = "commercial_response@1"


@pytest.fixture(scope="module")
def model():
    return R.load(MID)


@pytest.fixture(scope="module")
def report():
    return R.report(MID)


def _criteo():
    from crucible.calibration import criteo
    d = criteo.dataset_dir()
    if not os.path.exists(os.path.join(d, "features.npy")):
        pytest.fail(f"Criteo dataset not acquired at {d}: run crucible.calibration.acquire('criteo_uplift_v2.1')")
    return criteo


def test_registry_loads_checkpoint_with_verified_hash(model):
    assert isinstance(model, CommercialResponseModel)
    v = R.verify_checkpoint(MID)
    assert v["ok"] and v["bytes"] < 5_000_000
    # canonical identification vocabulary; the card's own term is kept in the view
    assert model.identification == "randomized_controlled_trial"
    assert model.view["identification_label"] == "randomized_experiment"
    assert model.domain == "advertising_response"
    assert model.features == [f"f{i}" for i in range(12)]
    assert model.outputs == ["p_visit_control", "p_visit_treated", "p_conversion_control", "p_conversion_treated"]
    assert R.card_problems(model.card) == []
    assert model.limits and len(model.card["extrapolation_limits"]) >= 5
    assert MID in R.list_models()


def test_tampered_checkpoint_is_refused(model, tmp_path, monkeypatch):
    import shutil
    d = tmp_path / "models" / "commercial_response"
    shutil.copytree(R.model_dir(MID), d)
    with open(d / model.card["checkpoint"]["file"], "ab") as fh:
        fh.write(b"x")
    monkeypatch.setattr(R, "MODELS_DIR", str(tmp_path / "models"))
    with pytest.raises(R.CalibrationError):
        R.load(MID)


def test_portable_predictions_equal_the_fitted_estimator(model):
    X = model.arrays["probe/X"]; ref = model.arrays["probe/pred"]
    P = model.predict(X)
    assert P.shape == (len(X), 4)
    assert np.max(np.abs(P - ref)) < 1e-9
    assert np.all((P >= 0) & (P <= 1))


def test_selection_follows_the_prespecified_rule(model, report):
    sel = model.card["selection"]
    cands = model.card["candidates"]
    primary = "primary_pooled_logloss_visit_plus_conversion"
    best_val = min(cands, key=lambda c: (cands[c]["validation"][primary], cands[c]["complexity"]))
    assert sel["kept"] == best_val
    assert {c["other"] for c in sel["comparisons"]} == set(cands) - {sel["kept"]}
    assert sel["worse_model_kept"] == 0
    for c in sel["comparisons"]:
        assert c["kept_significantly_worse"] == (c["kept_minus_other"]["ci95"][0] > 0)
        assert c["kept_minus_other"]["n_boot"] == 200
    # the kept model beats the constant-effect baseline on the held-out test split, outside bootstrap noise
    base = next(c for c in sel["comparisons"] if c["other"] == "constant_effect") if sel["kept"] != "constant_effect" else None
    assert base is not None and base["kept_minus_other"]["ci95"][1] < 0
    assert model.metrics[primary] < model.baseline_metrics[primary]
    assert report["heldout"]["test"][sel["kept"]]["primary"] == pytest.approx(model.metrics[primary], abs=1e-12)


def test_plan_was_committed_before_the_fit():
    pre = audit.plan_check(R.read_card(MID))
    assert pre["ok"], pre
    assert pre["source"] == "git"
    assert registry_e2.prespecified_before_fit(MID)["ok"]


def test_heldout_audit_metrics_reproduce_from_the_data(model, report):
    criteo = _criteo()
    lab = criteo.splits()
    assert criteo.split_digest(lab) == model.card["fit"]["split"]["digest"]
    idx = criteo.audit_indices(lab)
    from crucible.calibration.fitmeta_e2 import digest
    assert digest(idx.astype(np.int64)) == report["audit"]["indices_digest"]
    data = criteo.load()
    X = np.asarray(data["X"][idx], dtype=np.float64)
    t = np.asarray(data["treatment"][idx]).astype(bool)
    y = {"visit": np.asarray(data["visit"][idx]), "conversion": np.asarray(data["conversion"][idx])}
    got = {}
    for cand, saved in report["audit"]["candidates"].items():
        P = model.candidate_predict(cand, X)
        cols = {f"{o}_{a}": P[:, i] for i, (o, a) in enumerate([(o, a) for o in OUTCOMES for a in ARMS])}
        primary = 0.0
        for o in OUTCOMES:
            own = np.where(t, cols[f"{o}_treated"], cols[f"{o}_control"])
            ll = S.log_loss(y[o], own)
            assert ll == pytest.approx(saved[f"{o}_pooled_log_loss"], abs=1e-9)
            assert S.brier(y[o], own) == pytest.approx(saved[f"{o}_pooled_brier"], abs=1e-9)
            assert S.ece(y[o], own) == pytest.approx(saved[f"{o}_pooled_ece"], abs=1e-9)
            q = S.qini_coefficient(y[o], t, cols[f"{o}_treated"] - cols[f"{o}_control"])["qini_coefficient"]
            assert q == pytest.approx(saved[f"{o}_qini_coefficient"], abs=1e-9)
            primary += ll
        got[cand] = primary
        assert primary == pytest.approx(saved["primary"], abs=1e-9)
    kept = model.card["selection"]["kept"]
    assert got[kept] < got["constant_effect"]
    # the ranking of uplift is informative on held-out users (constant effect has Qini exactly 0)
    assert report["audit"]["candidates"]["constant_effect"]["visit_qini_coefficient"] == 0.0
    assert report["uplift_uncertainty"]["qini_bootstrap_test"]["visit"]["ci95"][0] > 0


def test_intervention_response_is_reported_with_uncertainty(model, report):
    for o in OUTCOMES:
        ir = report["intervention_response"][o]
        itt = ir["itt_difference_in_means"]
        assert itt["ci95"][0] < itt["ate"] < itt["ci95"][1]
        assert itt["ci95"][0] > 0                     # randomized: eligibility for targeting raises visits and conversions
        cace = ir["exposure_complier_effect_wald"]
        assert cace["estimate"] == pytest.approx(itt["ate"] / cace["exposure_rate_treated"], rel=1e-9)
        std = ir["standardised_estimates"]
        assert std["propensity_stratified"]["ci95"][0] > 0
        assert std["model_implied_standardised"] == pytest.approx(ir["model_implied_ate"], abs=1e-12)
    assert model.card["identification"]["kind"] == "randomized_experiment"
    assert model.card["identification"]["conditional_on_features"] is True
    diag = report["post_fit_diagnostics"]
    assert diag["share_without_overlap"] == 0.0 and 0.5 <= diag["propensity_auc_validation"] < 0.6


def test_response_type_table_is_valid_and_heldout_validated(model, report):
    tab = model.response_types()
    w = tab["weight"]
    assert w.size == 64 and abs(w.sum() - 1) < 1e-9 and (w > 0).sum() >= 32
    for k in ("p0_visit", "p1_visit", "p0_conversion", "p1_conversion"):
        assert np.all((tab[k][w > 0] >= 0) & (tab[k][w > 0] <= 1))
    rt = model.card["response_types"]
    tv = report["response_types"]["test_validation"]["visit"]
    assert rt["usable"] == (tv["weighted_corr_ci95"][0] > 0)
    assert rt["usable"] is True
    assert tv["types_compared"] >= 32
    # the table's average effect agrees with the design-based, composition-adjusted estimate on held-out users
    diag = report["post_fit_diagnostics"]["average_effects_test"]
    w = tab["weight"]
    for o in OUTCOMES:
        table_ate = float(np.dot(w, tab[f"p1_{o}"] - tab[f"p0_{o}"]))
        strat = diag[o]["propensity_stratified"]
        assert abs(table_ate - strat["estimate"]) < 3 * strat["se"], (o, table_ate, strat)


def test_sampling_and_response_reproduce_the_table(model):
    tab = model.response_types()
    rng = np.random.default_rng(123)
    n = 400_000
    types = model.sample_types(rng, n)
    assert np.array_equal(types, model.sample_types(np.random.default_rng(123), n))
    for treated in (False, True):
        r = model.respond(np.random.default_rng(7), types, np.full(n, treated))
        assert not np.any(r["conversion"] & ~r["visit"])
        key = "p1" if treated else "p0"
        for o in OUTCOMES:
            expect = float(np.mean(tab[f"{key}_{o}"][types]))
            se = np.sqrt(expect * (1 - expect) / n)
            assert abs(r[o].mean() - expect) < 5 * se + 1e-6


def test_transport_rebases_rates_and_keeps_odds_ratios(model):
    tab = model.response_types()
    new = model.transport(control_visit_rate=0.10, control_conversion_rate=0.01)
    w = tab["weight"]
    assert float(np.dot(w, new["p0_visit"])) == pytest.approx(0.10, abs=1e-9)
    assert float(np.dot(w, new["p0_conversion"])) == pytest.approx(0.01, abs=1e-9)
    lo = lambda p: np.log(p) - np.log1p(-p)
    m = w > 0
    for o in OUTCOMES:
        np.testing.assert_allclose((lo(new[f"p1_{o}"]) - lo(new[f"p0_{o}"]))[m], (lo(tab[f"p1_{o}"]) - lo(tab[f"p0_{o}"]))[m], atol=1e-7)
    assert np.array_equal(np.argsort(new["p0_visit"][m]), np.argsort(tab["p0_visit"][m]))


def test_scoring_matches_reference_implementations():
    from sklearn import metrics as M
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.2
    p = np.clip(0.2 + 0.3 * (y - 0.2) + 0.2 * rng.standard_normal(5000), 0.001, 0.999)
    p[:200] = 0.5                                     # ties
    assert S.log_loss(y, p) == pytest.approx(M.log_loss(y, p), abs=1e-12)
    assert S.brier(y, p) == pytest.approx(M.brier_score_loss(y, p), abs=1e-12)
    assert S.auc(y, p) == pytest.approx(M.roc_auc_score(y, p), abs=1e-12)
    assert S.average_precision(y, p) == pytest.approx(M.average_precision_score(y, p), abs=1e-12)
    t = rng.random(5000) < 0.5
    yf = y.astype(float)
    perfect = yf * t - yf * (~t)
    assert S.qini_coefficient(y, t, perfect)["qini_coefficient"] == pytest.approx(1.0)
    assert S.qini_coefficient(y, t, np.zeros(5000))["qini_coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert S.auuc_normalised(y, t, perfect) == pytest.approx(1.0)


def test_dataset_manifest_and_local_checksums():
    criteo = _criteo()
    man = all_datasets()["criteo_uplift_v2.1"]
    assert man["synthetic"] is False and man["license"]["spdx"] == "CC-BY-NC-SA-4.0"
    assert man["license"]["text_locations"] and man["provenance"]["design"]
    assert audit.provenance_complete(man)
    f = man["files"][0]
    assert f["sha256"] == criteo.SHA256 and f["bytes"] == criteo.BYTES
    path = os.path.join(criteo.dataset_dir(), f["file"])
    assert R.sha256_file(path) == f["sha256"]
    for d in man["derived_files"]:
        assert R.sha256_file(os.path.join(criteo.dataset_dir(), d["file"])) == d["sha256"]
    # the one dataset registry verifies and validates it through the acquirer module's hooks
    DS = importlib.import_module("crucible.calibration.datasets")     # the module (the package exports the function)
    res = DS.verify("criteo_uplift_v2.1")
    assert set(res) == {f["file"]} | {d["file"] for d in man["derived_files"]} and set(res.values()) == {"ok"}
    assert DS.validate_schema("criteo_uplift_v2.1") == []


def test_registry_gate_metrics_for_e2_models():
    g = registry_e2.gate_metrics()
    assert g["checkpoint_hash_mismatches"] == 0 and g["checkpoints_saved"] == 3
    assert g["heldout_reports"] == 3 and g["baseline_comparisons"] == 3 and g["calibration_metric_reports"] == 3
    assert g["datasets_with_checksums"] == 2 and g["provenance_records"] == 2
    assert g["extrapolation_limit_sections"] == 3 and g["synthetic_only_domains"] == 0
    assert g["metrics_prespecified_before_fit"] is True and g["worse_model_kept"] == 0
    # canonical registry.DOMAINS (the cards say commercial_intervention_response, software_collaboration_and_adoption)
    assert set(g["empirical_domains"]) == {"advertising_response", "software_adoption"}
    assert g["card_problems"] == {}
