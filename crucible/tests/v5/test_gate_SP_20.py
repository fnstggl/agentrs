"""Gate SP-20 (MASTER_SPEC Part 20): all 21 final deliverables are present and verified by the checker, and the final
report separates the nine Part 20 categories. SP-20b: the report states actual hardware, model sizes, population
sizes, world counts and test results, every number traceable to evidence (qa.claims_audit)."""
from __future__ import annotations

import os

from crucible.acceptance.v5 import manifest as M
from crucible.acceptance.v5 import qa, report
from crucible.acceptance.v5.checker import unmet, verify_all
from crucible.acceptance.v5.evidence import record

ROOT = M.crucible_root()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-20 --agent <name>"
ACTUALS = ("## Actual hardware and compute", "## Actual model sizes", "## Actual population sizes",
           "## Actual world counts", "## Actual test results")


def _report_text() -> str:
    p = os.path.join(M.docs_dir(ROOT), "FINAL_REPORT.md")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def test_SP_20_final_deliverables_present():
    res = verify_all(ROOT)
    dl = report.deliverables(res)
    text = _report_text()
    cats = sum(1 for c in report.CATEGORIES if f"* **{c}**:" in text)
    m = {"deliverables_present": sum(1 for d in dl if d["present"]), "deliverables_verified": sum(1 for d in dl if d["verified"]),
         "report_separation_categories": cats}
    record("SP-20", m, command=CMD, notes=f"not verified: {[d['deliverable'] for d in dl if not d['verified']]}")
    bad = unmet("SP-20", m)
    assert not bad, (bad, [d["deliverable"] for d in dl if not d["verified"]][:5])


def test_SP_20b_report_separation_and_actuals():
    text = _report_text()
    claims = qa.claims_audit(ROOT)
    m = {"report_separation_categories": sum(1 for c in report.CATEGORIES if f"* **{c}**:" in text),
         "report_actuals_sections": sum(1 for s in ACTUALS if s in text),
         "report_numbers_without_evidence": claims["metrics"]["report_claims_without_evidence"]
         + claims["metrics"]["report_claims_mismatching_evidence"] + (0 if text else 1)}
    record("SP-20b", m, command=CMD, notes=f"claims problems: {claims['problems'][:20]}")
    bad = unmet("SP-20b", m)
    assert not bad, (bad, claims["problems"][:10])
