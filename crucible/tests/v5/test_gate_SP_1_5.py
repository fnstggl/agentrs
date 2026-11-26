"""Gate SP-1.5 (MASTER_SPEC Part 1.5): milestone state is persisted in the machine-readable ledger docs/v5/PROGRESS.json
with every Part 1.5 field in every milestone, and the latest milestone lists every requirement that was not verified
at its commit. SP-1.5b: early-terminated sub-agents are recorded with their reassignment."""
from __future__ import annotations

from crucible.acceptance.v5 import progress
from crucible.acceptance.v5.checker import unmet
from crucible.acceptance.v5.evidence import record

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-1.5 --agent <name>"


def test_SP_1_5_milestone_state_persisted():
    m, probs = progress.metrics()
    keep = {k: m[k] for k in ("milestones_recorded", "ledger_valid", "milestones_missing_fields",
                              "latest_milestone_omits_unverified_requirements")}
    record("SP-1.5", keep, command=CMD, notes=f"validator problems: {probs[:20]}")
    bad = unmet("SP-1.5", keep)
    assert not bad, (bad, probs[:10])


def test_SP_1_5b_interrupted_work_reassigned():
    m, probs = progress.metrics()
    keep = {k: m[k] for k in ("interrupted_workstreams_recorded", "interrupted_workstreams_unassigned")}
    unassigned = [w["workstream"] for w in progress.load().get("interrupted_workstreams", []) if not w.get("reassigned_to")]
    record("SP-1.5b", keep, command=CMD, notes=f"unassigned: {unassigned}")
    bad = unmet("SP-1.5b", keep)
    assert not bad, (bad, unassigned)
