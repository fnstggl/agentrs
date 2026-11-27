"""Gate SP-21 (MASTER_SPEC Part 21): `python -m crucible.acceptance.v5 --verify-all` validates the manifest and the
evidence and exits nonzero whenever a requirement is not verified.

Measured on throwaway repositories that contain the real acceptance package (tests/acceptance_kit.py):
* the fabrication suite of the recovery audit (probes P1-P14 and variants in tests/test_v5_acceptance_checker.py) runs
  in a subprocess; every probe test asserts that its attack does not reach VERIFIED, so a passing probe test is a
  rejected attack;
* the real command line exits nonzero on a repository with an unverified requirement and zero on a verified one;
* the three project statuses (IN_PROGRESS, INCOMPLETE_BLOCKED, CRUCIBLE_5_COMPLETE) are produced;
* the real manifest equals its derivation and its ledger.
SP-19.7 (no removal, no weakening) is measured on the real ledger; SP-21b (the release check covers the committed
tree, the regression and clean-checkout reproduction) on a throwaway repository and the real manifest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acceptance_kit import CRIT, GOOD_TEST, Repo  # noqa: E402

from crucible.acceptance.v5 import ledger as L  # noqa: E402
from crucible.acceptance.v5 import manifest as M  # noqa: E402
from crucible.acceptance.v5.checker import unmet  # noqa: E402
from crucible.acceptance.v5.evidence import record  # noqa: E402

ROOT = M.crucible_root()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-21 --agent <name>"


def _cli(root: str, *extra: str) -> int:
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), CRUCIBLE_V5_SESSION="s-release-cli",
               CRUCIBLE_V5_SCRATCH=os.path.join(root, "..", "..", "scratch"))
    env.pop("CRUCIBLE_V5_RUN_ID", None)
    env.pop("CRUCIBLE_V5_EVIDENCE_OUT", None)
    return subprocess.run([sys.executable, "-m", "crucible.acceptance.v5", "--verify-all", *extra, "--out",
                           os.path.join(root, "..", "..", "report.json")], cwd=root, env=env,
                          capture_output=True, text=True).returncode


class _Env:
    """monkeypatch-like setenv for Repo, restoring the environment afterwards."""
    def __init__(self):
        self.saved = {}

    def setenv(self, k, v):
        self.saved.setdefault(k, os.environ.get(k))
        os.environ[k] = v

    def restore(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.timeout(3600)
def test_SP_21_checker_rejects_fabrications_and_exits_nonzero(tmp_path):
    junit = str(tmp_path / "probes.xml")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), TMPDIR=str(tmp_path))
    for k in ("CRUCIBLE_V5_RUN_ID", "CRUCIBLE_V5_EVIDENCE_OUT"):
        env.pop(k, None)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                    f"--junitxml={junit}", "tests/test_v5_acceptance_checker.py", "tests/test_v5_acceptance_probes.py",
                    "-k", "test_P"], cwd=ROOT, env=env, capture_output=True, text=True)
    cases = [(tc.get("name"), tc.find("failure") is None and tc.find("error") is None and tc.find("skipped") is None)
             for tc in ET.parse(junit).getroot().iter("testcase")]
    probes_run = len(cases)
    reaching = [n for n, ok in cases if not ok]
    e = _Env()
    try:
        items = [("Z.1", "Z", "make widgets", "A", [], CRIT, 3),
                 ("Z.2", "Z", "a hard one", "A", [], [{"metric": "hard", "op": ">=", "value": 1}], 3)]
        r = Repo(str(tmp_path / "statuses"), e, items=items)
        os.makedirs(r.scratch, exist_ok=True)
        r.honest()
        e.setenv("CRUCIBLE_V5_SESSION", "s-release")
        release = dict(reproduce=True, agent="release-check", scratch=r.scratch)
        statuses = {r.verify(**release)[0]["project_status"]}
        rc_unverified = _cli(r.root, "--reproduce")
        m = r.manifest()
        next(x for x in m["requirements"] if x["id"] == "Z.2").update(status="BLOCKED", blocker="BLK-1")
        r.save_manifest(m)
        r.commit("blocked")
        statuses.add(r.verify(**release)[0]["project_status"])
        one = Repo(str(tmp_path / "complete"), e)
        one.honest()
        e.setenv("CRUCIBLE_V5_SESSION", "s-release")
        statuses.add(one.verify(reproduce=True, agent="release-check", scratch=one.scratch)[0]["project_status"])
        rc_static = _cli(one.root)                    # static checks alone never complete
        rc_verified = _cli(one.root, "--reproduce")
    finally:
        e.restore()
    cons = M.consistency(ROOT)
    anchored = not cons["per_id"] and all("uncommitted" in p for p in cons["global"])
    metrics = {"fabrication_probes_run": probes_run, "fabrication_probes_reaching_verified": len(reaching),
               "checker_rejects_fabricated_evidence": probes_run >= 14 and not reaching,
               "checker_nonzero_when_unverified": rc_unverified != 0 and rc_static != 0 and rc_verified == 0,
               "project_statuses_supported": len(statuses), "manifest_anchored_to_derivation": anchored}
    record("SP-21", metrics, command=CMD, notes=f"probes reaching VERIFIED: {reaching}; statuses {sorted(statuses)}; "
                                                f"exit codes unverified {rc_unverified}, verified without the release check "
                                                f"{rc_static}, verified with it {rc_verified}")
    bad = unmet("SP-21", metrics)
    assert not bad, (bad, reaching, sorted(statuses))


def test_SP_19_7_no_requirement_removed_and_no_criteria_weakened():
    entries = L.read(ROOT)
    st = L.state(entries)
    derived = {d["id"] for d in M.derive(ROOT)}
    removed = sorted(i for i in st if not i.startswith("__") and i not in derived)
    rules = L.rule_problems(st, L.decision_sections(M.docs_dir(ROOT)))
    weaker = sorted(rid for rid, ps in rules.items() if any("not provably" in p for p in ps))
    git_probs = [p for p in L.git_problems(ROOT) if "uncommitted" not in p]
    m = {"requirements_removed_from_ledger": len(removed), "criteria_changes_not_stricter": len(weaker),
         "ledger_append_only_violations": len(git_probs) + len(L.chain_problems(entries)),
         "ledger_entries_checked": len(entries)}
    record("SP-19.7", m, command=CMD, notes=f"removed: {removed}; not stricter: {weaker}; ledger: {git_probs[:5]}")
    bad = unmet("SP-19.7", m)
    assert not bad, (bad, removed, weaker, git_probs[:3])


@pytest.mark.timeout(1800)
def test_SP_21b_release_check_covers_tree_regression_and_reproduction(tmp_path):
    ext = tmp_path / "external_widgets.txt"
    ext.write_text("12")
    src = GOOD_TEST.replace("    n = widgets.count()\n", f"    n = int(open({str(ext)!r}).read())\n    widgets.count()\n")
    e = _Env()
    try:
        r = Repo(str(tmp_path / "repo"), e, test_src=src)
        r.honest()
        e.setenv("CRUCIBLE_V5_SESSION", "s-release")
        release = dict(reproduce=True, agent="release-check", scratch=r.scratch)
        before = r.verify(**release)[0]["project_status"]
        r.write("src/widgets/__init__.py", r.read("src/widgets/__init__.py") + "\n# uncommitted edit\n")
        dirty = r.verify(**release)[0]["project_status"]
        subprocess.run(["git", "checkout", "--", "."], cwd=r.root, check=True)
        ext.write_text("11")                          # the world outside the tree changed: only reproduction sees it
        static_res, static_rows = r.verify()
        static = static_rows["Z.1"]["status"]
        repro, rows = r.verify(**release)
    finally:
        e.restore()
    ids = {x["id"]: x for x in M.load(ROOT)["requirements"]}
    regression_required = all(i in ids for i in ("A.1", "X.2", "X.4")) and \
        any(c["metric"] == "integration_branch_test_failures" for c in ids["X.4"]["criteria"])
    m = {"verify_all_checks_committed_tree": before == "CRUCIBLE_5_COMPLETE" and dirty != "CRUCIBLE_5_COMPLETE",
         "verify_all_checks_regression": regression_required,
         "verify_all_reproduces_in_clean_checkout": static == "VERIFIED" and
         static_res["project_status"] != "CRUCIBLE_5_COMPLETE" and
         repro["project_status"] != "CRUCIBLE_5_COMPLETE" and rows["Z.1"]["status"] == "FAILED" and
         rows["Z.1"]["reproduction"]["commit"] == static_res["head"]}
    record("SP-21b", m, command=CMD, notes=f"release statuses: clean {before}, dirty {dirty}; after an external change "
                                           f"the static row is {static} (project {static_res['project_status']}), the "
                                           f"release check (re-execution at HEAD) gives {repro['project_status']}")
    bad = unmet("SP-21b", m)
    assert not bad, (bad, before, dirty, static, repro["project_status"])
