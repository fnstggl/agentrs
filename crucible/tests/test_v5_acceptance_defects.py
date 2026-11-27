"""The defect register's duties (qa.defect_revalidation, revalidate_defect, triage_defect, close_defect) on a throwaway
git repository: an open defect needs a re-validation by one of its owners at or after the wave base; QA's re-triage is
recorded but never counts as the owner's; a defect is closed only with a fix commit in HEAD's history and regression
tests that exist."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from crucible.acceptance.v5 import qa


def _git(root: str, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _entry(i: str, owner: str, status: str = "open") -> str:
    d = {"id": i, "severity": "critical", "status": status, "area": "Z_area", "owner": owner, "requirements": ["Z.1"],
         "summary": f"defect {i}", "source": {"audit": "probe", "finding": None}}
    return f"### {i} (critical, {status}): defect {i}\n\n```json\n{json.dumps(d, indent=1, sort_keys=True)}\n```\n"


@pytest.fixture
def repo(tmp_path):
    root = str(tmp_path / "repo" / "crucible")
    os.makedirs(os.path.join(root, "docs", "v5"))
    os.makedirs(os.path.join(root, "tests"))
    top = os.path.dirname(root)
    _git(top, "init", "-q")
    _git(top, "config", "user.email", "t@t")
    _git(top, "config", "user.name", "t")
    with open(os.path.join(root, "tests", "test_z.py"), "w") as fh:
        fh.write("def test_regression():\n    assert True\n")
    with open(os.path.join(root, "README"), "w") as fh:
        fh.write("base\n")
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", "base")
    base = _git(top, "rev-parse", "HEAD")
    header = f"# register\n\n<!-- revalidate-since: {base} -->\n\n"
    with open(os.path.join(root, "docs", "v5", "DEFECTS.md"), "w") as fh:
        fh.write(header + _entry("DEF-Z-1", "B") + "\n" + _entry("DEF-Z-2", "C / orchestrator"))
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", "register")
    return root, top


def test_open_defects_are_due_until_an_owner_revalidates_them(repo):
    root, top = repo
    r = qa.defect_revalidation(root)
    assert r["metrics"]["open_defects_not_revalidated"] == 2 and r["metrics"]["revalidation_base_set"]
    assert r["due_by_owner"] == {"B": ["DEF-Z-1"], "C": ["DEF-Z-2"], "orchestrator": ["DEF-Z-2"]}
    # a non-owner's re-validation does not count; QA's triage is recorded but is not a re-validation
    qa.revalidate_defect(root, "DEF-Z-1", by="Q", verdict="still_open", reason="looked at it")
    qa.triage_defect(root, "DEF-Z-1", verdict="likely_fixed", reason="the test it names passes now", evidence="t")
    _git(top, "commit", "-qam", "reval")
    r = qa.defect_revalidation(root)
    assert "DEF-Z-1" in r["not_revalidated"] and r["invalid_revalidations"]
    assert r["metrics"]["open_defects_qa_triaged"] == 1 and r["metrics"]["open_defects_qa_likely_fixed"] == 1
    # the owner re-validates: still open -> no longer due; reported fixed -> listed for closing
    qa.revalidate_defect(root, "DEF-Z-1", by="B", verdict="fixed", reason="fixed by the population merge")
    qa.revalidate_defect(root, "DEF-Z-2", by="orchestrator", verdict="still_open", reason="needs a decision")
    _git(top, "commit", "-qam", "owner reval")
    r = qa.defect_revalidation(root)
    assert r["metrics"]["open_defects_not_revalidated"] == 0
    assert r["to_close"] == ["DEF-Z-1 (fixed per B)"]
    for bad in (dict(verdict="done", reason="x"), dict(verdict="fixed", reason=" ")):
        with pytest.raises(ValueError):
            qa.revalidate_defect(root, "DEF-Z-2", by="C", **bad)
    with pytest.raises(ValueError):
        qa.triage_defect(root, "DEF-Z-2", verdict="closed", reason="x")


def test_a_defect_is_closed_only_with_a_fix_in_history_and_existing_regression_tests(repo):
    root, top = repo
    head = _git(top, "rev-parse", "HEAD")
    with pytest.raises(ValueError):
        qa.close_defect(root, "DEF-Z-1", fixed_by="0123456789abcdef0123 not a commit",
                        regression_tests=["tests/test_z.py::test_regression"])
    with pytest.raises(ValueError):
        qa.close_defect(root, "DEF-Z-1", fixed_by=f"{head} the fix", regression_tests=[])
    with pytest.raises(ValueError):
        qa.close_defect(root, "DEF-Z-1", fixed_by=f"{head} the fix", regression_tests=["tests/test_z.py::test_nope"])
    d = qa.close_defect(root, "DEF-Z-1", fixed_by=f"{head} the fix", regression_tests=["tests/test_z.py::test_regression"])
    assert d["status"] == "fixed"
    with pytest.raises(ValueError):                 # QA triages open defects only
        qa.triage_defect(root, "DEF-Z-1", verdict="still_open", reason="x")
    r = qa.defect_revalidation(root)
    assert r["metrics"]["fixed_defects_without_regression_test"] == 0 and r["not_revalidated"] == ["DEF-Z-2"]
    # a fixed defect whose regression test disappears is reported
    os.remove(os.path.join(root, "tests", "test_z.py"))
    assert qa.defect_revalidation(root)["fixed_without_regression_test"] == ["DEF-Z-1"]
