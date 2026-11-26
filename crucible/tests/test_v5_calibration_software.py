"""Workstream E2: software collaboration and adoption models (GH Archive), loaded through the registry.

Uses the committed checkpoints, cards and reports and the extracted GH Archive window under $CRUCIBLE_DATA. A missing
dataset is a failure, not a skip: run `crucible.calibration.acquire("gharchive_2024-10-07_96h")` and
`python -m crucible.calibration.fit_software` (which also writes the evaluated tables). Models load through the
unified registry (`crucible.calibration.registry`, the module behind `crucible.calibration.load`).
"""
import importlib
import json
import os

import numpy as np
import pytest

from crucible.calibration import audit
from crucible.calibration import datasets as all_datasets
from crucible.calibration import registry as R
from crucible.calibration import scoring_e2 as S
from crucible.calibration.software import DEEP_FEATURES, GROWTH_FEATURES, DeepEngagementModel, RepoGrowthModel

DEEP = "software_deep_engagement@1"
GROWTH = "software_repo_growth@1"


@pytest.fixture(scope="module")
def deep():
    return R.load(DEEP)


@pytest.fixture(scope="module")
def growth():
    return R.load(GROWTH)


@pytest.fixture(scope="module")
def tables():
    from crucible.calibration import fit_software as F
    p = F.tables_path()
    if not os.path.exists(p):
        pytest.fail(f"evaluated tables missing at {p}: acquire the GH Archive window and run fit_software")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def test_models_load_with_verified_hashes(deep, growth):
    assert isinstance(deep, DeepEngagementModel) and isinstance(growth, RepoGrowthModel)
    for m in (deep, growth):
        v = R.verify_checkpoint(m.id)
        assert v["ok"] and v["bytes"] < 5_000_000
        assert m.identification == "observational" and m.domain == "software_adoption"
        assert R.card_problems(m.card) == []
        assert "2024-10-07" in m.card["population"] and "public GitHub" in m.card["population"]
        assert len(m.card["extrapolation_limits"]) >= 6
    assert deep.features == [n for n, _, _ in DEEP_FEATURES]
    assert growth.features == [n for n, _, _ in GROWTH_FEATURES]
    assert growth.dispersion > 0


def test_portable_predictions_equal_the_fitted_estimators(deep, growth):
    for m in (deep, growth):
        X = m.arrays["probe/X"]; ref = m.arrays["probe/pred"]
        got = m.predict(X)
        assert np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))) < 1e-9


# Tables rebuilt from the pseudonymised events (gharchive extractor 2, tables v2): identity columns hold pseudonym
# ranks instead of raw-id ranks, and repository-indexed tables follow the pseudonym order, so their digests differ
# from the fit's (raw-id) tables by construction; every other table is byte-identical to what the fit used.
ORDER_DEPENDENT = ("deep/actor", "growth/", "curve/stars_24h", "curve/stars_48h")


def test_tables_match_report_digests(deep, growth, tables):
    from crucible.calibration.fitmeta_e2 import digest
    checked = 0
    for mid in (DEEP, GROWTH):
        for k, d in R.report(mid)["tables"]["digests"].items():
            if k.startswith(ORDER_DEPENDENT):
                assert k in tables
                continue
            assert digest(np.ascontiguousarray(tables[k])) == d, k
            checked += 1
    assert checked >= 5
    # repository-indexed tables: the same rows as the fit's, in pseudonym order (bucket split preserved)
    assert int(tables["growth/d2/y"].size) == 321303 and int(tables["growth/d1/train_bucket"].sum()) > 0


def _deep_splits(tables):
    from crucible.calibration.fit_software import SPLIT_HOURS
    t = tables["deep/t"]
    return {s: (t >= a * 3600) & (t < b * 3600) for s, (a, b) in SPLIT_HOURS.items()}


def test_deep_engagement_heldout_metrics_reproduce(deep, tables):
    rep = R.report(DEEP)
    sp = _deep_splits(tables)
    X = tables["deep/X"][sp["test"]]; y = tables["deep/y"][sp["test"]].astype(float)
    kept = deep.card["selection"]["kept"]
    for cand, saved in rep["heldout"]["test"].items():
        p = deep.candidate_predict(cand, X)
        for k, fn in (("log_loss", S.log_loss), ("brier", S.brier), ("ece", S.ece), ("auc", S.auc),
                      ("average_precision", S.average_precision)):
            if cand == "constant_rate" and k in ("auc", "average_precision"):
                continue
            assert fn(y, p) == pytest.approx(saved[k], abs=1e-9), (cand, k)
    assert np.array_equal(deep.predict(X), deep.candidate_predict(kept, X))
    base = rep["heldout"]["test"]["constant_rate"]["log_loss"]
    assert rep["heldout"]["test"][kept]["log_loss"] < base
    # time-based holdout and user-level holdout both improve on the base rate
    assert rep["strict_time_check"]["test"]["log_loss"] < base
    nu = rep["heldout"]["test_new_users"]
    assert nu[kept]["n"] > 1000 and nu[kept]["log_loss"] < nu["constant_rate"]["log_loss"]
    assert rep["heldout"]["test"][kept]["auc"] > 0.6


def test_repo_growth_heldout_metrics_reproduce(growth, tables):
    rep = R.report(GROWTH)
    X = tables["growth/d2/X"]; y = tables["growth/d2/y"]
    kept = growth.card["selection"]["kept"]
    for cand, saved in rep["heldout"]["test"].items():
        mu = growth.candidate_predict(cand, X)
        assert S.poisson_deviance(y, mu) == pytest.approx(saved["mean_poisson_deviance"], abs=1e-9), cand
        assert float(np.mean(np.abs(y - mu))) == pytest.approx(saved["mae"], abs=1e-9)
    assert rep["heldout"]["test"][kept]["mean_poisson_deviance"] < rep["heldout"]["test"]["proportional"]["mean_poisson_deviance"]
    assert rep["heldout"]["test"][kept]["spearman"] > rep["heldout"]["test"]["proportional"]["spearman"] - 0.05
    nr = rep["heldout"]["test_new_repos"]
    assert nr[kept]["n"] > 1000 and nr[kept]["mean_poisson_deviance"] < nr["proportional"]["mean_poisson_deviance"]


def test_selection_rule_and_prespecification(deep, growth):
    for m, primary in ((deep, "log_loss"), (growth, "mean_poisson_deviance")):
        cands = m.card["candidates"]
        best = min(cands, key=lambda c: (cands[c]["validation"][primary], cands[c]["complexity"]))
        assert m.card["selection"]["kept"] == best
        assert m.card["selection"]["worse_model_kept"] == 0
        assert {c["other"] for c in m.card["selection"]["comparisons"]} == set(cands) - {best}
        for c in m.card["selection"]["comparisons"]:
            assert c["kept_significantly_worse"] == (c["kept_minus_other"]["ci95"][0] > 0)
        pre = audit.plan_check(m.card)
        assert pre["ok"], pre
        assert [p["role"] for p in pre["plans"]] == ["plan", "amendment"]


def test_negative_binomial_sampling_matches_mean_and_dispersion(growth, tables):
    X = tables["growth/d2/X"][:2000]
    rng = np.random.default_rng(3)
    mu = growth.predict(X)
    draws = np.stack([growth.sample(X, rng) for _ in range(400)])
    m = draws.mean(0); v = draws.var(0)
    big = mu > 1
    assert np.mean(np.abs(m - mu) / np.sqrt((mu + growth.dispersion * mu ** 2) / 400)) < 1.5
    ratio = v[big].sum() / (mu[big] + growth.dispersion * mu[big] ** 2).sum()
    assert big.sum() > 50 and 0.85 < ratio < 1.15
    dist = R.report(GROWTH)["distribution"]
    assert dist["ks_log1p_nb"] <= dist["ks_log1p_poisson"] + 1e-9 or growth.dispersion < 0.05


def test_raw_count_mapping_round_trips(deep, growth, tables):
    for m, spec, X in ((deep, DEEP_FEATURES, tables["deep/X"][:500]), (growth, GROWTH_FEATURES, tables["growth/d2/X"][:500])):
        raw = {n: (np.expm1(X[:, i]) if tf == "log1p" else X[:, i]) for i, (n, tf, _) in enumerate(spec)}
        np.testing.assert_allclose(m.matrix_from_counts(raw), X, rtol=1e-12, atol=1e-12)
        with pytest.raises(KeyError):
            m.matrix_from_counts({k: v for k, v in list(raw.items())[1:]})


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_tables_rebuild_deterministically_from_extracted_events(tables):
    from crucible.calibration import fit_software as F, gharchive as G
    ev = G.load_events()
    rebuilt = F.build_tables(ev)
    assert set(rebuilt) == set(tables)
    for k in tables:
        assert np.array_equal(rebuilt[k], tables[k]), k


def test_dataset_manifest_matches_extracted_window():
    from crucible.calibration import gharchive as G
    man = all_datasets()[G.DATASET]
    assert man["synthetic"] is False and len(man["files"]) == G.HOURS
    assert all(len(f["sha256"]) == 64 for f in man["files"])
    assert man["license"]["text_locations"] and man["provenance"]["privacy"]
    assert audit.provenance_complete(man)
    # the one dataset registry verifies the window through gharchive's hooks (raw hours are deleted by design)
    DS = importlib.import_module("crucible.calibration.datasets")     # the module (the package exports the function)
    res = DS.verify(G.DATASET)
    assert len(res) == G.HOURS and set(res.values()) == {"ok"}
    assert DS.validate_schema(G.DATASET) == []
    with open(os.path.join(G.dataset_dir(), "acquisition.json")) as fh:
        acq = json.load(fh)
    by_file = {f["file"]: f for f in acq["files"]}
    for f in man["files"]:
        assert by_file[f["file"]]["raw_sha256"] == f["sha256"]
        stem = f["file"][:-len(".json.gz")]
        with np.load(os.path.join(G.dataset_dir(), "extracted", f"{stem}.npz")) as z:
            assert G.content_digest({k: z[k] for k in z.files}) == f["extract_digest"]
