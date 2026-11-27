"""Gate X (MASTER_SPEC Part 17): the final integrated quality audit, run by the independent QA workstream (P) after
every implementation workstream has been integrated (X.n1).

X.1 counts unverified requirements with the checker (fixed point: everything except X, SP-18 and SP-20, which depend on
X). X.2 and X.4 share one recorded run of the whole suite under tests/ (gate tests included, except this file and the
Gate A file, whose own evidence covers them), set CRUCIBLE_GATE_X_WORKERS to split it. X.3 clones the committed tree
and runs the documented commands and the reproduction of every VERIFIED requirement. X.5-X.9 use
crucible.acceptance.v5.qa. Every test records its measurement before asserting on it.
"""
from __future__ import annotations

import functools
import glob
import os
import re
import subprocess
import sys
import tempfile

import pytest

from crucible.acceptance.v5 import baseline as B
from crucible.acceptance.v5 import manifest as M
from crucible.acceptance.v5 import qa
from crucible.acceptance.v5.checker import X_EXCLUDED, unmet, verify_all
from crucible.acceptance.v5.evidence import record

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate X --agent <qa agent>"
# X.2/X.4 wait for the whole suite, submitted job by job to the shared slot queue (baseline.suite_plan; no deadline
# while queued): the measuring test must outlive that wait. 8 h - the earlier limit - is less than the queue's own
# waits on a busy day, so the test was killed before the suite finished (wave 3b repair pass). Default 96 h.
SUITE_TEST_TIMEOUT_S = int(os.environ.get("CRUCIBLE_GATE_X_SUITE_TIMEOUT_S", str(96 * 3600)))
ROOT = M.crucible_root()
SELF = ("tests/v5/test_gate_X.py", "tests/v5/test_gate_A.py")


@functools.lru_cache(maxsize=1)
def _checker() -> dict:
    return verify_all(ROOT)


@functools.lru_cache(maxsize=1)
def _suite() -> dict:
    """One run of every test file under tests/ (except SELF), with collection first."""
    files = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "tests", "**", "test_*.py"),
                                                                recursive=True))
    files = [f for f in files if f not in SELF]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                          *files], cwd=ROOT, env=env, capture_output=True, text=True)
    collected = {ln.strip() for ln in col.stdout.splitlines() if "::" in ln and not ln.startswith(" ")}
    out = tempfile.mkdtemp(prefix="gateX-suite-", dir=B.scratch_dir())
    # through the slot scheduler by default (baseline.suite_plan: one slot job per gate file and about an hour of
    # unit-test files per job, each with its class's CPU/GPU and its measured RAM and wall time, at most
    # CRUCIBLE_GATE_X_INFLIGHT jobs submitted at once); CRUCIBLE_GATE_X_VIA_SLOTS=0 runs the groups in this process's
    # own slot
    via = None if os.environ.get("CRUCIBLE_GATE_X_VIA_SLOTS", "1") == "0" else "P: Gate X.2/X.4 suite"
    res = B.run_pytest_groups(ROOT, files, out, int(os.environ.get("CRUCIBLE_GATE_X_WORKERS", "1")),
                              timeout=30 * 3600, isolate=("tests/test_realworld_gate.py",), via_slots=via)
    ran = {(c["classname"], c["name"]): c["outcome"] for c in res["cases"]}

    def key(node: str) -> tuple[str, str]:
        f, rest = node.split("::", 1)
        parts = rest.split("::")
        return ".".join([f[:-3].replace("/", ".")] + parts[:-1]), parts[-1]
    not_run = [n for n in collected if key(n) not in ran]
    return {"files": files, "collected": collected, "collect_rc": col.returncode, "cases": res["cases"],
            "outcomes": [c["outcome"] for c in res["cases"]], "not_run": not_run, "rcs": res["rcs"]}


def test_X_1_all_manifest_requirements_verified():
    res = _checker()
    counted = [r for r in res["rows"] if not (r["id"].startswith("X.") or r["id"] in X_EXCLUDED)]
    unverified = [r["id"] for r in counted if r["status"] != "VERIFIED"]
    m = {"unverified_requirements": len(unverified), "requirements_counted": len(counted),
         "manifest_anchored": not [p for p in res["global_problems"] if "uncommitted" not in p]}
    record("X.1", m, command=CMD, notes=f"unverified: {unverified[:60]}; project problems: {res['global_problems'][:5]}")
    bad = unmet("X.1", m)
    assert not bad, (bad, unverified[:20])


@pytest.mark.slow
@pytest.mark.timeout(SUITE_TEST_TIMEOUT_S)
def test_X_2_all_required_tests_executed():
    s = _suite()
    man = M.load(ROOT)
    listed = sorted({t for e in man["requirements"] for t in e["tests"]})
    missing = [t for t in listed if not os.path.exists(os.path.join(ROOT, t))]
    m = {"required_tests_collected": len(s["collected"]), "required_tests_not_executed": len(s["not_run"]),
         "required_test_files_missing": len(missing), "required_tests_skipped": s["outcomes"].count("skipped")}
    record("X.2", m, command=CMD, notes=f"not executed: {sorted(s['not_run'])[:20]}; manifest test files missing: "
                                        f"{missing[:20]}")
    bad = unmet("X.2", m)
    assert not bad, (bad, missing[:10], sorted(s["not_run"])[:10])


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_X_3_clean_checkout_reproduction():
    res = qa.clean_checkout(ROOT)
    m = res["metrics"]
    record("X.3", m, command=CMD, notes=f"commit {res['commit']}; failing commands: "
                                        f"{[c['command'] for c in res['commands'] if c['rc']]}; reproduction failed: "
                                        f"{res['reproduction_failed'][:20]}; clone status after: {res['clone_status_after']}")
    bad = unmet("X.3", m)
    assert not bad, (bad, [c for c in res["commands"] if c["rc"]][:3], res["reproduction_failed"][:5])


@pytest.mark.slow
@pytest.mark.timeout(SUITE_TEST_TIMEOUT_S)
def test_X_4_working_integrated_branch():
    s = _suite()
    o = s["outcomes"]
    m = {"integration_branch_tests_run": len(o), "integration_branch_test_failures": o.count("failure"),
         "integration_branch_test_errors": o.count("error"), "integration_branch_tests_skipped": o.count("skipped")}
    failed = [f"{c['classname']}::{c['name']}" for c in s["cases"] if c["outcome"] in ("failure", "error")]
    record("X.4", m, command=CMD, notes=f"failing: {failed[:30]}; group exit codes {s['rcs']}")
    bad = unmet("X.4", m)
    assert not bad, (bad, failed[:10])


def test_X_5_complete_documentation():
    res = qa.documents(ROOT)
    m = res["metrics"]
    record("X.5", m, command=CMD, notes=f"missing: {res['missing']}; lacking: {res['lacking']}; stale: {res['stale']}")
    bad = unmet("X.5", m)
    assert not bad, (bad, res["missing"], res["lacking"])


def test_X_6_no_open_critical_or_high_defects():
    res = qa.defects(ROOT)
    rv = qa.defect_revalidation(ROOT)          # the register's re-validation duty (D-83), reported with X.6
    m = {**res["metrics"], **rv["metrics"]}
    record("X.6", m, command=CMD, notes=f"open critical/high: {res['open_critical_high'][:40]}; unregistered audit "
                                        f"findings: {res['unregistered'][:20]}; malformed: {res['malformed'][:10]}; "
                                        f"re-validation due by owner since {rv['since']}: "
                                        f"{ {k: len(v) for k, v in rv['due_by_owner'].items()} }; owners report fixed: "
                                        f"{rv['to_close'][:10]}; fixed without regression tests: "
                                        f"{rv['fixed_without_regression_test'][:10]}")
    bad = unmet("X.6", m)
    assert not bad, (bad, res["open_critical_high"][:10], res["unregistered"][:5])


def test_X_7_no_fake_skipped_or_placeholder_acceptance_tests():
    res = qa.scan_acceptance_tests(ROOT)
    m = res["metrics"]
    record("X.7", m, command=CMD, notes=f"flagged: {res['flagged'][:30]}; manifest ids no test records: "
                                        f"{res['unrecorded_ids'][:60]}")
    bad = unmet("X.7", m)
    assert not bad, (bad, res["flagged"][:10], res["unrecorded_ids"][:10])


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_X_8_no_disconnected_workstreams():
    res = qa.integration_audit(ROOT)
    m = res["metrics"]
    record("X.8", m, command=CMD, notes=f"disconnected: {res['disconnected']}; errors: {res['errors'][:5]}; "
                                        f"per workstream: {res['per_workstream']}")
    bad = unmet("X.8", m)
    assert not bad, (bad, res["disconnected"], res["errors"][:3])


def test_X_9_accurate_final_evidence_report():
    res = qa.claims_audit(ROOT)
    m = res["metrics"]
    record("X.9", m, command=CMD, notes=f"problems: {res['problems'][:30]}")
    bad = unmet("X.9", m)
    assert not bad, (bad, res["problems"][:10])


def test_X_n1_run_after_full_integration():
    """Every workstream branch (c5c/*, recovery/*, and the branches named in docs/v5/workstreams/*.json) is merged
    into the checked commit, and the checked commit is the head of the integration branch (its own evidence and
    review commits aside). In an official clone the branches are read from the repository it was cloned from."""
    res = qa.integration_state(ROOT)
    m = res["metrics"]
    record("X.n1", m, command=CMD, notes=f"unmerged: {res['unmerged']}; head {res['head'][:9]}; integration branch "
                                         f"{res['integration_tip'][:9]} ({res['record_commits_skipped']} record "
                                         f"commits skipped; {res['repository']})")
    bad = unmet("X.n1", m)
    assert not bad, (bad, res["unmerged"])
