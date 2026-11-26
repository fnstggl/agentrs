"""SP-5.1 (MASTER_SPEC Parts 5.1 / 5.2; owner workstream E): calibration data acquired legitimately with licences
recorded, no re-identifiable identifiers, user-level and time-based holdouts, uncertainty, and every effect labelled
with how it is identified (observational, randomised exposure, randomised controlled trial).

Measured on what is registered: the dataset registry (licence records), the derived GH Archive tables on disk (every
actor / repository identifier value is checked: keyed pseudonyms in the event tables, dense codes in the feature
tables, never a raw GitHub id), the held-out reports (bootstrap intervals, holdout definitions) and the model cards
(identification vocabulary)."""
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import audit_sp5, registry as R

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.1"


def _criteria(req):
    return next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]


def test_SP_5_1_data_governance_holdouts_uncertainty_identification():
    m = audit_sp5.sp51_metrics()
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    record("SP-5.1", metrics, command=CMD,
           artifacts=[R.os.path.join(R.CAL_DOCS, "DATASETS.md"), R.os.path.join(R.CAL_DOCS, "models", "LICENSE_NOTICE.md")],
           notes=f"holdout kinds per model: {m['_holdout']}; GH Archive id check: {m['_gharchive']}")
    # the identifier check really ran over the derived data (not vacuously over nothing)
    assert m["_gharchive"]["present"] and m["_gharchive"]["values"] > 1_000_000, m["_gharchive"]
    assert m["datasets_with_licence_record"] == m["datasets_registered"]
    fails = [msg for c in _criteria("SP-5.1") for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, fails


def test_SP_5_1_raw_identifier_check_detects_raw_ids(tmp_path):
    """The identifier check is not vacuous: an event table that keeps raw GitHub ids is counted."""
    import json
    import numpy as np
    d = tmp_path / "gharchive_2024-10-07_96h" / "extracted"
    d.mkdir(parents=True)
    np.savez(d / "2024-10-07-0.npz", actor=np.array([583231, 9919], np.int64),
             repo=np.array([1 << 62, 1 << 61], np.int64))
    (d / "2024-10-07-0.meta.json").write_text(json.dumps({"extractor_version": "1"}))
    out = audit_sp5.gharchive_raw_id_values(str(tmp_path))
    assert out["raw_like_values"] == 2 and out["raw_extractor_files"] == 1
