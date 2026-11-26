"""Gate F: empirical behavioural grounding (MASTER_SPEC Part 17, Gate F; owner workstream E).

Every metric is computed at test time from what is registered: the model cards and checkpoints under
`docs/v5/calibration/models/` (both card schemas, read through `registry.card_view`), their held-out reports, and the
dataset registry (`crucible.calibration.datasets()`), so models registered by any calibration workstream count once
merged. F.3 also recomputes the held-out test metric of every qualifying model from the verified datasets and the
saved checkpoint through the runtime decision-model API (`crucible.calibration.runtime`), so a report cannot claim
what the saved model does not achieve and the runtime is shown to load every kept checkpoint and evaluate held-out
rows. F.3 is part of every run of this module (not `slow`-only); the module needs every dataset a registered model
cites (`acquire` downloads and verifies them when absent) and takes about 15-25 minutes on one CPU: run it through
the slot scheduler.
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


@pytest.fixture(scope="module")
def reproduction():
    """Every qualifying model's held-out test metric recomputed from the verified datasets through the runtime."""
    return reproduce.reproduce_all()


def _context(m: dict) -> dict:
    return {k: m[k] for k in ("registered_models", "qualifying_models")}


def test_F_1_three_domains_fitted_on_empirical_data(state):
    recs, m = state
    domains = sorted({r["domain"] for r in recs if r.get("qualifies")})
    from crucible.calibration import fidelity as FI
    claims = FI.inworld_claims()
    metrics = {"empirical_domains_fitted": m["empirical_domains_fitted"], **_context(m),
               "domains": ",".join(domains),
               # honesty: how many fitted models a default world actually uses (wave 2 wires them, D-15)
               "fitted_models_used_by_default_worlds": sum(1 for c in claims.values() if c["ok"] and
                                                           c["kind"] == "default"),
               "fitted_models_with_verified_opt_in_use": len({c["model"] for c in claims.values()
                                                              if c["ok"] and c["kind"] == "opt-in"})}
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
    # so must the randomised commercial-response models
    for name in rc:
        sel = R.card_view(R.read_card(name))["selection"]
        sign = 1 if sel["lower_is_better"] else -1
        assert not sel["kept_is_baseline"]
        assert sign * sel["test"][sel["kept"]] < min(sign * sel["test"][b] for b in sel["baselines"])
    metrics["content_consumption_test_log_loss"] = card["metrics"]["log_loss"]
    metrics["content_consumption_baseline_log_loss"] = card["baseline_metrics"]["log_loss"]
    record("F.2", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="commercial response counts a qualifying model in an advertising/purchase domain whose "
                 "identification is randomised (registered by workstream E2)")
    _assert_met("F.2", metrics)


@pytest.mark.timeout(3600)
def test_F_3_heldout_reports_reproduce_from_checkpoints(state, reproduction):
    recs, m = state
    rep = reproduction
    worst = max(r["abs_diff"] / max(abs(r["reported"]), 1e-9) for r in rep.values())
    metrics = {"heldout_reports": m["heldout_reports"], "reports_reproduced": len(rep),
               "max_relative_difference": float(worst), **_context(m)}
    for name, r in rep.items():
        metrics[f"{name}.{r['metric']}"] = r["recomputed"]
    record("F.3", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="held-out test metrics recomputed through crucible.calibration.runtime (registry.load, hash-verified) "
                 "on the verified datasets (commercial_response: on the report's seeded 100,000-user test audit "
                 "subsample)")
    # every qualifying model's held-out metric is recomputed, whichever workstream registered it
    assert set(rep) == {r["name"] for r in recs if r.get("qualifies")}
    assert set(reproduce.E1_MODELS) | set(reproduce.E2_MODELS) <= set(rep)
    assert worst < 1e-4, rep
    for r in recs:
        if r.get("qualifies"):
            sp = r["report"]["splits"]
            assert all((sp[s].get("rows") or sp[s].get("users")) for s in r["heldout"])
    _assert_met("F.3", metrics)


def test_F_4_baseline_comparisons(state):
    recs, m = state
    wins = [r["name"] for r in recs if r.get("qualifies") and r["baseline_compared"] and not r["kept_is_baseline"]]
    metrics = {"baseline_comparisons": m["baseline_comparisons"], "learned_models_beating_baseline": len(wins),
               **_context(m)}
    for r in recs:
        if r.get("qualifies"):
            bc = r["report"]["baseline_comparison"]
            metrics[f"{r['name']}.difference_vs_{bc['baseline']}"] = float(bc["difference"])
    metrics["kept_models_losing_to_their_baseline"] = m["kept_models_losing_to_their_baseline"]
    record("F.4", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="a comparison counts only when the report agrees with the card (audit.report_consistency); the "
                 "simulator's rule-based behaviour is compared in the E3 reviews and the E4 reports")
    # every learned model that is kept beats its best simple baseline on the held-out primary metric
    assert m["kept_models_losing_to_their_baseline"] == 0
    assert all(r["kept_beats_baseline"] for r in recs if r.get("qualifies"))
    _assert_met("F.4", metrics)


def test_F_5_calibration_metrics(state):
    recs, m = state
    metrics = {"calibration_metric_reports": m["calibration_metric_reports"], **_context(m)}
    for r in recs:
        if r.get("qualifies"):
            for k, v in audit.calibration_values(r["report"]).items():
                metrics[f"{r['name']}.{k}"] = v
            for k, bins in audit.calibration_tables(r["report"]).items():
                metrics[f"{r['name']}.{k}.bins"] = bins
    record("F.5", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="held-out calibration of the kept model: scalar measures (ECE, Brier, PIT KS, coverage, count "
                 "calibration error, per outcome where there are several) and reliability tables (bins of mean "
                 "predicted vs mean observed)")
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
    # tampering is detected: for one model of each card schema, a copy with one flipped byte fails load and is
    # counted as a mismatch
    schemas = {}
    for r in recs:
        schemas.setdefault(r["schema"], r["name"])
    assert set(schemas) == set(R.CARD_SCHEMAS), schemas
    fake = tmp_path / "models"
    for name in schemas.values():
        shutil.copytree(R.model_dir(name), fake / name)
        ck = fake / name / R.checkpoint_file(R.read_card(name))
        b = bytearray(ck.read_bytes())
        b[len(b) // 2] ^= 0xFF
        ck.write_bytes(bytes(b))
    monkeypatch.setattr(R, "MODELS_DIR", str(fake))
    tampered = []
    for name in schemas.values():
        with pytest.raises(IOError):
            R.load(name)
        tampered.append(audit.examine(name))
    # the behavioural card fields are pinned too (registry index outside the model directories): a copy whose card
    # changes feature_quantiles / prior_rate / selection, with an intact checkpoint, is refused
    card_fake = tmp_path / "card_models"
    for name in schemas.values():
        shutil.copytree(R.model_dir(name), card_fake / name)
        cp = card_fake / name / "model_card.json"
        c = __import__("json").loads(cp.read_text())
        c["selection"]["kept"] = "tampered"
        c["prior_rate"] = 0.4
        cp.write_text(__import__("json").dumps(c))
    monkeypatch.setattr(R, "MODELS_DIR", str(card_fake))
    for name in schemas.values():
        with pytest.raises(IOError):
            R.load(name)
        tampered.append(audit.examine(name))
    monkeypatch.undo()
    assert all(t["hash_mismatches"] >= 1 and not t["qualifies"] for t in tampered)
    metrics = {"checkpoints_saved": m["checkpoints_saved"], "checkpoint_hash_mismatches":
               m["checkpoint_hash_mismatches"], "tamper_detected": True,
               "largest_checkpoint_bytes": max(R.verify_checkpoint(r["name"])["largest_bytes"] for r in recs)}
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
    # every dataset a qualifying model rests on is present and verifies (acquire downloads it if absent)
    cited = sorted({n for r in recs if r.get("qualifies") for n in r["real_datasets"]})
    assert {"kuairand_pure", "mind_small", "criteo_uplift_v2.1", "gharchive_2024-10-07_96h"} <= set(cited)
    for n in cited:
        DS.acquire(n)
        res = DS.verify(n)
        assert res and all(v == "ok" for v in res.values()), (n, {k: v for k, v in res.items() if v != "ok"})
        assert DS.validate_schema(n) == [], n
    metrics = {"datasets_with_checksums": m["datasets_with_checksums"], "provenance_records": m["provenance_records"],
               "datasets_registered": len(ds), "datasets_verified_locally": verified,
               "datasets_cited_by_qualifying_models": len(cited), "datasets": ",".join(sorted(ds))}
    record("F.7", metrics, command=CMD, artifacts=_artifacts(recs),
           notes="checksums: SHA-256 for every downloaded file (and every extracted file models use); provenance: "
                 "publisher, source, citation, population, platform, period, retrieval date, licence and the "
                 "location of the licence text")
    _assert_met("F.7", metrics)


def test_F_8_extrapolation_limits_documented(state):
    recs, m = state
    from crucible.calibration import runtime as RT
    metrics = {"extrapolation_limit_sections": m["extrapolation_limit_sections"], **_context(m)}
    # limits are machine-readable where a use can be checked: a decision model refuses content outside its scope
    refused = 0
    for sid, sp in RT.specs().items():
        assert len(sp["limits"]) >= 80, sid
        kinds = sp["scope"].get("content_kinds")
        if kinds:
            use = (sp["scope"].get("use") or [None])[0]
            with pytest.raises(RT.ContractError):
                RT.get(sid).check_scope("not-a-" + kinds[0], use)
            if sp["scope"].get("use"):
                with pytest.raises(RT.ContractError):
                    RT.get(sid).check_scope(kinds[0], "individual_choice")   # a recommender target is not a choice rule
            RT.get(sid).check_scope(kinds[0], use)
            refused += 1
    metrics["decision_models_with_enforced_scope"] = refused
    assert refused >= 4
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


@pytest.mark.timeout(3600)
def test_F_runtime_serves_every_kept_model_on_heldout_rows(state, reproduction):
    """The runtime decision-model API loads every kept checkpoint (hash-verified) and its held-out evaluation equals
    the report; where the E3 review keeps another contender, that contender is what the runtime serves."""
    from crucible.calibration import runtime as RT
    recs, _ = state
    for r in recs:
        if not r.get("qualifies"):
            continue
        mid = f"{r['name']}@{r['card']['version']}"
        dm = RT.decision_for(mid)
        assert dm.model_id == mid and dm.fitted.view["files"] == r["view"]["files"]
        row = reproduction[r["name"]]
        assert row["abs_diff"] <= 1e-4 * max(abs(row["reported"]), 1e-9), (mid, row)
        rv = RT.review_of(mid)
        assert dm.served == ((rv or {}).get("kept_after_review") or "K")
    assert RT.usage()                                      # the reproduction ran through the runtime
