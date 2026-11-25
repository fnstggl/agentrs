"""Gate F: empirical behavioural grounding (MASTER_SPEC Part 17, Gate F; owner workstream E).

Every metric is computed at test time from what is registered: the model cards and checkpoints under
`docs/v5/calibration/models/`, their held-out reports, and the dataset registry (`crucible.calibration.datasets()`),
so models registered by any calibration workstream count once merged. F.3 also recomputes the held-out test metric
of each E1 model from the verified datasets and the saved checkpoint, so a report cannot claim what the saved model
does not achieve. The full module needs the KuaiRand-Pure and MIND-small data (`acquire` downloads and verifies them
when absent) and takes a few minutes; run it through the slot scheduler with 2 CPUs.
"""
import importlib
import os
import shutil

import numpy as np
import pytest

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import audit, registry as R, reproduce

DS = importlib.import_module("crucible.calibration.datasets")      # the module (the package exports the function)

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate F"


def _criteria(req: str) -> list[dict]:
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


def _assert_met(req: str, metrics: dict) -> None:
    fails = [msg for c in _criteria(req) for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, f"{req}: {fails}"


def _artifacts(recs) -> list[str]:
    out = []
    for r in recs:
        if r.get("card"):
            out.append(os.path.join(R.model_dir(r["name"]), "model_card.json"))
            rp = r["card"].get("report")
            if rp and os.path.isfile(os.path.join(R.CRUCIBLE_ROOT, rp)):
                out.append(os.path.join(R.CRUCIBLE_ROOT, rp))
    return out


@pytest.fixture(scope="module")
def state():
    recs = audit.all_records()
    return recs, audit.gate_f_metrics(recs)


def _context(m: dict) -> dict:
    return {k: m[k] for k in ("registered_models", "qualifying_models")}


def test_F_1_three_domains_fitted_on_empirical_data(state):
    recs, m = state
    domains = sorted({r["domain"] for r in recs if r.get("qualifies")})
    metrics = {"empirical_domains_fitted": m["empirical_domains_fitted"], **_context(m),
               "domains": ",".join(domains)}
    record("F.1", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="distinct registry.DOMAINS among models that load with verified hashes, cite a checksummed "
                 "non-synthetic dataset and have a held-out report")
    problems = {r["name"]: r["problems"] for r in recs if not r.get("qualifies")}
    assert not problems, f"registered models that do not qualify: {problems}"
    _assert_met("F.1", metrics)


def test_F_2_content_consumption_and_randomized_commercial_response(state):
    recs, m = state
    cc = [r["name"] for r in recs if r.get("qualifies") and r["domain"] == "content_selection"]
    rc = [r["name"] for r in recs if r.get("qualifies") and r["domain"] in audit.COMMERCIAL_DOMAINS
          and any(x in r["identification"] for x in audit.RANDOMIZED)]
    metrics = {"content_consumption_fitted": m["content_consumption_fitted"],
               "randomized_commercial_response_fitted": m["randomized_commercial_response_fitted"],
               "content_consumption_models": ",".join(cc), "randomized_commercial_models": ",".join(rc)}
    # the randomised-exposure choice model must beat its simple baseline on held-out data
    card = R.read_card("content_consumption")
    assert card["provenance"]["identification"] == "randomized_exposure"
    assert not card["selection"]["kept_is_baseline"]
    metrics["content_consumption_test_log_loss"] = card["metrics"]["log_loss"]
    metrics["content_consumption_baseline_log_loss"] = card["baseline_metrics"]["log_loss"]
    record("F.2", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="commercial response counts a qualifying model in an advertising/purchase domain whose "
                 "identification is randomised (registered by workstream E2)")
    _assert_met("F.2", metrics)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_F_3_heldout_reports_reproduce_from_checkpoints(state):
    recs, m = state
    rep = reproduce.reproduce_kuairand()
    rep.update(reproduce.reproduce_news())
    worst = max(r["abs_diff"] / max(abs(r["reported"]), 1e-9) for r in rep.values())
    metrics = {"heldout_reports": m["heldout_reports"], "e1_reports_reproduced": len(rep),
               "max_relative_difference": float(worst), **_context(m)}
    for name, r in rep.items():
        metrics[f"{name}.{r['metric']}"] = r["recomputed"]
    record("F.3", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="held-out test metrics recomputed with registry.load on the verified datasets")
    assert set(rep) == set(reproduce.E1_MODELS)
    assert worst < 1e-4, rep
    for r in recs:
        if r.get("qualifies"):
            sp = r["report"]["splits"]
            assert all((sp[s].get("rows") or sp[s].get("users")) for s in r["heldout"])
    _assert_met("F.3", metrics)


def test_F_4_baseline_comparisons(state):
    recs, m = state
    wins = [r["name"] for r in recs if r.get("qualifies") and r["baseline_compared"]
            and not r["card"]["selection"].get("kept_is_baseline", True)]
    metrics = {"baseline_comparisons": m["baseline_comparisons"], "learned_models_beating_baseline": len(wins),
               **_context(m)}
    for r in recs:
        if r.get("qualifies"):
            bc = r["report"]["baseline_comparison"]
            metrics[f"{r['name']}.difference_vs_{bc['baseline']}"] = float(bc["difference"])
    record("F.4", metrics, command=CMD, artifacts=_artifacts(recs))
    _assert_met("F.4", metrics)


def test_F_5_calibration_metrics(state):
    recs, m = state
    metrics = {"calibration_metric_reports": m["calibration_metric_reports"], **_context(m)}
    for r in recs:
        if r.get("qualifies"):
            cal = r["report"]["calibration"]
            for k in audit.CALIBRATION_KEYS:
                if isinstance(cal.get(k), (int, float)):
                    metrics[f"{r['name']}.{k}"] = float(cal[k])
    record("F.5", metrics, command=CMD, artifacts=_artifacts(recs))
    _assert_met("F.5", metrics)


def test_F_6_checkpoints_saved_and_hash_verified(state, tmp_path, monkeypatch):
    recs, m = state
    # every checkpoint loads and predicts finite values of the documented shape
    for r in recs:
        f = R.load(r["name"])
        q = f.card.get("feature_quantiles")
        X = np.array([[q[n][5] for n in f.features]]) if q else np.ones((1, len(f.features)))
        y = f.predict(X)
        assert np.all(np.isfinite(y))
    # tampering is detected: a copy with one flipped byte fails load and is counted as a mismatch
    name = recs[0]["name"]
    fake = tmp_path / "models"
    shutil.copytree(R.model_dir(name), fake / name)
    ck = fake / name / "params.npz"
    b = bytearray(ck.read_bytes())
    b[len(b) // 2] ^= 0xFF
    ck.write_bytes(bytes(b))
    monkeypatch.setattr(R, "MODELS_DIR", str(fake))
    with pytest.raises(IOError):
        R.load(name)
    tampered = audit.examine(name)
    monkeypatch.undo()
    assert tampered["hash_mismatches"] == 1 and not tampered["qualifies"]
    metrics = {"checkpoints_saved": m["checkpoints_saved"], "checkpoint_hash_mismatches":
               m["checkpoint_hash_mismatches"], "tamper_detected": True,
               "largest_checkpoint_bytes": max(os.path.getsize(os.path.join(R.model_dir(r["name"]), f))
                                               for r in recs for f in r["card"]["files"])}
    record("F.6", metrics, command=CMD, artifacts=_artifacts(recs))
    assert metrics["largest_checkpoint_bytes"] <= R.MAX_CHECKPOINT_BYTES
    _assert_met("F.6", metrics)


def test_F_7_dataset_provenance_and_checksums(state):
    recs, m = state
    ds = DS.datasets()
    verified = 0
    for n in ds:
        root = DS.dataset_dir(n)
        if os.path.isdir(root):
            res = DS.verify(n)
            if all(v == "ok" for v in res.values()):
                verified += 1
    # the datasets E1's models rest on are present and verify (acquire downloads them if absent)
    for n in ("kuairand_pure", "mind_small"):
        DS.acquire(n)
        assert all(v == "ok" for v in DS.verify(n).values())
        assert DS.validate_schema(n) == []
    metrics = {"datasets_with_checksums": m["datasets_with_checksums"], "provenance_records": m["provenance_records"],
               "datasets_registered": len(ds), "datasets_verified_locally": verified,
               "datasets": ",".join(sorted(ds))}
    record("F.7", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="checksums: SHA-256 for every downloaded file (and every extracted file models use); provenance: "
                 "publisher, source, citation, population, platform, period, retrieval date, licence and the "
                 "location of the licence text")
    _assert_met("F.7", metrics)


def test_F_8_extrapolation_limits_documented(state):
    recs, m = state
    metrics = {"extrapolation_limit_sections": m["extrapolation_limit_sections"], **_context(m)}
    for r in recs:
        if r.get("qualifies"):
            md = os.path.join(R.CRUCIBLE_ROOT, r["card"]["report"][:-5] + ".md")
            if os.path.isfile(md):
                assert "## Extrapolation limits" in open(md, encoding="utf-8").read()
    record("F.8", metrics, command=CMD, artifacts=_artifacts(recs))
    _assert_met("F.8", metrics)


def test_F_n1_no_synthetic_only_domain(state):
    recs, m = state
    metrics = {"synthetic_only_domains": m["synthetic_only_domains"],
               "synthetic_datasets_registered": sum(1 for e in DS.datasets().values() if e.get("synthetic")),
               **_context(m)}
    record("F.n1", metrics, command=CMD, artifacts=_artifacts(recs))
    _assert_met("F.n1", metrics)
