"""The v5 acceptance checker verifies evidence itself: fabricated, stale, unreviewed or self-reviewed evidence fails."""
import json
import os
import subprocess

import pytest

from crucible.acceptance.v5 import checker, evidence, manifest as M

CRIT = [{"metric": "widgets", "op": ">=", "value": 10}, {"metric": "leaks", "op": "==", "value": 0}]


def git(root, *a):
    subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)


def entry(**kw):
    e = {"id": "Z.1", "gate": "Z", "requirement": "make widgets", "workstream": "A", "dependencies": [],
         "implementation_files": ["src/widgets.py"], "tests": ["tests/v5/test_gate_Z.py"],
         "verify_command": "x", "criteria": CRIT,
         "criteria_history": [{"criteria": CRIT, "decision": None, "note": "initial"}],
         "status": "IMPLEMENTED_UNVERIFIED", "evidence": [], "review": None, "commit": None, "blocker": None,
         "source": {}}
    e.update(kw)
    return e


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A repository with one requirement, its implementation, a passing JUnit record, evidence and a review."""
    root = str(tmp_path)
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    os.makedirs(f"{root}/src")
    os.makedirs(f"{root}/tests/v5")
    os.makedirs(f"{root}/docs/v5")
    open(f"{root}/src/widgets.py", "w").write("N = 12\n")
    open(f"{root}/tests/v5/test_gate_Z.py", "w").write("def test_z():\n    pass\n")
    open(f"{root}/docs/v5/DECISION_LOG.md", "w").write("# log\n\n## D-1 raise the bar\n")
    open(f"{root}/docs/v5/BLOCKERS.md", "w").write("# blockers\n")
    M.save({"schema": "x", "requirements": [entry()]}, root)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "impl")

    def produce(metrics=None, reviewer="qa", author="builder"):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/v5/test_gate_Z.py::test_z (call)")
        monkeypatch.setenv("CRUCIBLE_V5_AUTHOR", author)
        p = evidence.record("Z.1", metrics or {"widgets": 12, "leaks": 0}, root=root)
        jd = f"{root}/docs/v5/evidence/gate_Z"
        open(f"{jd}/junit_1.xml", "w").write(
            '<testsuites><testsuite><testcase classname="tests.v5.test_gate_Z" name="test_z"/></testsuite></testsuites>')
        ev = json.load(open(p))
        ev["junit"] = {"path": "docs/v5/evidence/gate_Z/junit_1.xml", "sha256": evidence.sha256_file(f"{jd}/junit_1.xml")}
        json.dump(ev, open(p, "w"))
        os.makedirs(f"{root}/docs/v5/reviews", exist_ok=True)
        json.dump({"requirement": "Z.1", "evidence_sha256": evidence.sha256_file(p), "reviewer": reviewer,
                   "verdict": "approved"}, open(f"{root}/docs/v5/reviews/Z.1.json", "w"))
        return p
    return root, produce


def status(root):
    res = checker.verify_all(root)
    return res, res["rows"][0]


def test_complete_evidence_is_verified(world):
    root, produce = world
    produce()
    res, row = status(root)
    assert row["status"] == "VERIFIED", row["problems"]
    assert res["ok"] and res["project_status"] == "CRUCIBLE_5_COMPLETE"


def test_a_criterion_is_evaluated_on_the_metrics_not_trusted(world):
    root, produce = world
    produce({"widgets": 9, "leaks": 0})
    res, row = status(root)
    assert row["status"] == "FAILED" and not res["ok"]


def test_a_missing_metric_fails(world):
    root, produce = world
    produce({"widgets": 12})
    assert status(root)[1]["status"] == "FAILED"


def test_a_self_review_is_rejected(world):
    root, produce = world
    produce(reviewer="builder", author="builder")
    assert status(root)[1]["status"] == "IMPLEMENTED_UNVERIFIED"


def test_editing_evidence_after_review_is_detected(world):
    root, produce = world
    p = produce()
    ev = json.load(open(p))
    ev["metrics"]["widgets"] = 13
    json.dump(ev, open(p, "w"))
    _, row = status(root)
    assert row["status"] != "VERIFIED" and any("different version" in x for x in row["problems"])


def test_stale_evidence_is_rejected_after_the_implementation_changes(world):
    root, produce = world
    produce()
    open(f"{root}/src/widgets.py", "w").write("N = 3\n")
    git(root, "add", "src/widgets.py")
    git(root, "commit", "-qm", "change")
    _, row = status(root)
    assert row["status"] != "VERIFIED" and any("stale" in x for x in row["problems"])


def test_evidence_from_an_uncommitted_tree_is_rejected(world):
    root, produce = world
    open(f"{root}/src/widgets.py", "w").write("N = 99\n")
    produce()
    _, row = status(root)
    assert any("uncommitted" in x for x in row["problems"])


def test_a_failed_or_skipped_test_case_blocks_verification(world):
    root, produce = world
    p = produce()
    jp = f"{root}/docs/v5/evidence/gate_Z/junit_1.xml"
    open(jp, "w").write('<testsuites><testsuite><testcase classname="tests.v5.test_gate_Z" name="test_z">'
                        '<skipped/></testcase></testsuite></testsuites>')
    ev = json.load(open(p))
    ev["junit"]["sha256"] = evidence.sha256_file(jp)
    json.dump(ev, open(p, "w"))
    json.dump({"requirement": "Z.1", "evidence_sha256": evidence.sha256_file(p), "reviewer": "qa",
               "verdict": "approved"}, open(f"{root}/docs/v5/reviews/Z.1.json", "w"))
    _, row = status(root)
    assert any("not passed" in x for x in row["problems"])


def test_a_claimed_status_is_not_trusted(world):
    root, _ = world
    M.save({"schema": "x", "requirements": [entry(status="VERIFIED")]}, root)
    res, row = status(root)
    assert row["status"] != "VERIFIED" and not res["ok"]
    assert "manifest claims VERIFIED" in row["problems"][0]


def test_criteria_changes_need_a_recorded_decision(world):
    root, produce = world
    produce()
    weaker = [{"metric": "widgets", "op": ">=", "value": 1}, {"metric": "leaks", "op": "==", "value": 0}]
    hist = [{"criteria": CRIT, "decision": None}, {"criteria": weaker, "decision": "D-99"}]
    M.save({"schema": "x", "requirements": [entry(criteria=weaker, criteria_history=hist)]}, root)
    assert any("no recorded decision" in x for x in status(root)[1]["problems"])
    hist[1]["decision"] = "D-1"
    M.save({"schema": "x", "requirements": [entry(criteria=weaker, criteria_history=hist)]}, root)
    assert status(root)[1]["status"] == "VERIFIED"


def test_the_real_manifest_covers_every_spec_gate_item():
    gates = M.spec_gates()
    m = M.load()
    ids = {e["id"] for e in m["requirements"]}
    for g, info in gates.items():
        for i in range(1, len(info["bullets"]) + 1):
            assert f"{g}.{i}" in ids
    s1 = next(e for e in m["requirements"] if e["id"] == "S.1")
    assert "CAM" not in s1["requirement"] and s1["source"]["clarification"]


def test_the_real_checker_exits_nonzero_while_anything_is_unverified():
    from crucible.acceptance.v5.__main__ import main
    assert main(["--verify-all"]) == 1
