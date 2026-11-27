"""Tests of the completion checker (cam_live.verify_completion) on synthetic projects, plus checks of the real
manifest against the specification. Every synthetic project is a real git repository whose commit times, JUnit run
times and verdict times are set explicitly, so each freshness rule is exercised deterministically."""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from cam_live import verify_completion as vc

T0 = 1_790_000_000          # 2026-09-21: all synthetic times are offsets from here (well before "now")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def _iso(t: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


class Proj:
    """A synthetic CAM-Live-shaped project in a fresh git repository."""

    def __init__(self, root: str):
        self.root = root
        self.pkg = "vcsynth_" + uuid.uuid4().hex[:10]
        self.reqs: List[Dict[str, Any]] = []
        os.makedirs(root, exist_ok=True)
        self.git("init", "-q")
        pk = self.pkg
        self.write("pyproject.toml", f'[project]\nname = "x"\n\n[project.scripts]\nsynth = "{pk}.runtime.cli:main"\n')
        self.write(f"src/{pk}/__init__.py", '"""synthetic package"""\n')
        self.write(f"src/{pk}/runtime/__init__.py", '"""runtime"""\n')
        self.write(f"src/{pk}/runtime/cli.py",
                   "import argparse\n\n\ndef build_parser():\n    ap = argparse.ArgumentParser()\n"
                   "    sub = ap.add_subparsers(dest='cmd')\n    sub.add_parser('doctor')\n    sub.add_parser('run')\n"
                   "    return ap\n\n\ndef main(argv=None):\n    return build_parser().parse_args(argv) and 0\n")
        self.write(f"src/{pk}/agent.py", "from .core import act\n\n\ndef build_agent():\n    return act()\n")
        self.write(f"src/{pk}/core.py", "def act():\n    return 1\n")
        self.write(f"src/{pk}/lonely.py", "def unused():\n    return 2\n")
        self.write(f"src/{pk}/diag.py", "def check():\n    return 3\n\n\nif __name__ == '__main__':\n    check()\n")
        self.write(f"src/{pk}/factory_mod.py", "def source(task, run_dir):\n    return 4\n")
        self.write(f"src/{pk}/stub.py", "def later():\n    pass\n")
        self.write(f"src/{pk}/doc_only.py", "X = 1\n")
        self.write("tasks/t.yaml", f"success:\n  evidence_source: {pk}.factory_mod:source\n")
        self.write("tests/test_core.py", "def test_act():\n    assert True\n\n\ndef test_other():\n    assert True\n")
        self.write("docs/GUIDE.md", "# guide\n")
        for d in ("docs/status", "docs/evidence/test_results", "docs/verification"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.commit("initial", T0)
        self.paths = vc.Paths(root, package=self.pkg)

    # -- files and git ------------------------------------------------------------------------------------------
    def write(self, rel: str, text: str) -> None:
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(text)

    def git(self, *args: str, at: Optional[float] = None) -> str:
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@example.invalid")
        if at is not None:
            env.update(GIT_AUTHOR_DATE=f"@{int(at)} +0000", GIT_COMMITTER_DATE=f"@{int(at)} +0000")
        p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                           cwd=self.root, capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stderr
        return p.stdout

    def commit(self, msg: str, at: float, paths: Sequence[str] = ()) -> None:
        self.git("add", *(paths or ["-A"]))
        self.git("commit", "-q", "--allow-empty", "-m", msg, at=at)

    # -- records ------------------------------------------------------------------------------------------------
    def junit(self, name: str, at: float, cases: Sequence[Tuple[str, str]]) -> None:
        """cases: (test name in tests/test_core.py, outcome 'passed' | 'failed' | 'skipped')."""
        body = ""
        for n, oc in cases:
            inner = {"passed": "", "failed": "<failure message='x'/>", "skipped": "<skipped message='gpu'/>"}[oc]
            body += f'<testcase classname="tests.test_core" name="{n}" time="0.1">{inner}</testcase>'
        self.write(f"docs/evidence/test_results/{name}.xml",
                   f'<?xml version="1.0"?><testsuites><testsuite name="pytest" timestamp="{_iso(at)}" '
                   f'tests="{len(cases)}">{body}</testsuite></testsuites>')

    def status(self, ws: str, reqs: Dict[str, Dict[str, Any]], commands: Sequence[str] = ()) -> None:
        for c in reqs.values():
            c.setdefault("commands", list(commands))
        self.write(f"docs/status/{ws}.json", json.dumps({"workstream": ws, "requirements": reqs}))

    def verdicts(self, lens: str, entries: Dict[str, Dict[str, Any]], at: float) -> None:
        self.write(f"docs/verification/verdicts_{lens}.json", json.dumps(entries))
        self.commit(f"verdicts {lens}", at, [f"docs/verification/verdicts_{lens}.json"])

    def req(self, rid: str, ws: str = "A", kind: str = "runtime", level: int = 1, evidence: bool = False,
            declared: Optional[Dict[str, Any]] = None) -> None:
        self.reqs.append({"id": rid, "title": f"requirement {rid}", "spec_ref": "x", "workstream": ws, "kind": kind,
                          "level": level, "needs_executed_evidence": evidence, "declared": declared or {}})

    def run(self, md: bool = False) -> Dict[str, Any]:
        self.write("docs/CAM_LIVE_COMPLETION_MANIFEST.json",
                   json.dumps({"requirements": self.reqs, "coverage": {"x": [{"item": "i", "ids": [r["id"] for r in self.reqs]}]}}))
        for m in [m for m in sys.modules if m == self.pkg or m.startswith(self.pkg + ".")]:
            del sys.modules[m]
        importlib.invalidate_caches()
        out = os.path.join(self.root, "docs", "COMPLETION.md") if md else None
        try:
            return vc.run(paths=self.paths, completion_md=out)
        finally:
            if self.paths.src in sys.path:
                sys.path.remove(self.paths.src)

    def row(self, rep: Dict[str, Any], rid: str) -> Dict[str, Any]:
        return next(r for r in rep["rows"] if r["id"] == rid)

    def claim(self, files: Sequence[str] = (), tests: Sequence[str] = (), ips: Sequence[str] = (),
              state: str = "implemented", evidence: Sequence[str] = ()) -> Dict[str, Any]:
        return {"state": state, "files": list(files), "tests": list(tests), "integration_points": list(ips),
                "evidence": list(evidence)}


@pytest.fixture
def proj(tmp_path) -> Proj:
    return Proj(str(tmp_path / "p"))


def _core_req(p: Proj, rid: str = "R-01", ws: str = "A", **kw: Any) -> Dict[str, Any]:
    p.req(rid, ws, **kw)
    return p.claim([f"src/{p.pkg}/core.py"], ["tests/test_core.py::test_act"], [f"{p.pkg}.core.act"])


VERIFIED = {"status": "verified", "by": "G-a", "notes": "checked"}


# ---------------------------------------------------------------------------------------------------------------
def test_complete_needs_a_fresh_pass_and_a_verdict_issued_after_the_last_change(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "complete", r
    assert r["tests"] == {"named": 1, "pass_fresh": 1, "pass_any": 1, "detail": r["tests"]["detail"]}
    assert r["verification"]["state"] == "fresh-verified"


def test_a_commit_after_the_run_and_the_verdict_makes_both_stale(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 10\n")
    proj.commit("fix", T0 + 300)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "implemented-needs-reverification", r
    assert "predates the last change" in r["reason"] and "1 of 1 test results stale" in r["reason"]
    assert r["tests"]["pass_fresh"] == 0 and r["tests"]["pass_any"] == 1
    assert any("stale" in s for s in r["stale"])
    # a new run and a new verdict after the fix make it complete again
    proj.junit("A2", T0 + 400, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": dict(VERIFIED, notes="re-checked")}, T0 + 500)
    assert proj.row(proj.run(), "R-01")["group"] == "complete"


def test_a_change_to_the_test_file_after_the_run_makes_the_result_stale(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    proj.write("tests/test_core.py", "def test_act():\n    assert 1 == 1\n\n\ndef test_other():\n    assert True\n")
    proj.commit("test edit", T0 + 300)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "implemented-needs-reverification"
    assert "tests/test_core.py" in r["tests"]["detail"][0]["changed_by"]


def test_an_uncommitted_edit_counts_as_a_change_made_now(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 11\n")        # not committed
    rep = proj.run()
    r = proj.row(rep, "R-01")
    assert r["group"] == "implemented-needs-reverification"
    assert "uncommitted" in r["tests"]["detail"][0]["changed_by"]
    assert f"src/{proj.pkg}/core.py" in rep["dirty_requirement_files"]


def test_a_newer_skipped_result_never_hides_an_earlier_pass(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A_gpu", T0 + 100, [("test_act", "passed")])
    proj.junit("A_default", T0 + 150, [("test_act", "skipped")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "complete"
    assert r["tests"]["detail"][0]["result_file"] == "A_gpu.xml"


def test_only_skipped_results_are_not_a_pass(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "skipped")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "partial" and any("only skipped results" in p for p in r["problems"])


def test_a_fresh_failure_is_failed_and_a_failure_before_a_fix_needs_a_rerun(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "failed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "failed" and "FAILED" in r["reason"]
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 12\n")
    proj.commit("fix", T0 + 300)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "partial" and "re-run" in r["reason"]
    proj.junit("A2", T0 + 400, [("test_act", "failed"), ("test_act", "passed")])  # newest of one run: failure wins
    assert proj.row(proj.run(), "R-01")["group"] == "failed"


def test_verdict_states_fresh_partial_failed_and_stale_failed(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": {"status": "partial", "by": "G-a", "notes": "gap"}}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "partial" and r["reason"].startswith("fresh verdict 'partial'")
    proj.verdicts("a", {"R-01": {"status": "failed", "by": "G-a", "notes": "defect"}}, T0 + 250)
    assert proj.row(proj.run(), "R-01")["group"] == "failed"
    # the fix lands after the failed verdict and is tested: re-verification, not failure
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 13\n")
    proj.commit("fix", T0 + 300)
    proj.junit("A2", T0 + 400, [("test_act", "passed")])
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "implemented-needs-reverification" and "verdict 'failed' predates" in r["reason"]


def test_a_verdict_time_is_when_its_entry_last_changed_not_when_the_file_did(proj):
    proj.req("R-02", "A")
    proj.status("A", {"R-01": _core_req(proj), "R-02": proj.claim([f"src/{proj.pkg}/agent.py"],
                                                                   ["tests/test_core.py::test_other"],
                                                                   [f"{proj.pkg}.agent.build_agent"])})
    proj.junit("A", T0 + 100, [("test_act", "passed"), ("test_other", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 14\n")
    proj.commit("fix", T0 + 300)
    proj.junit("A2", T0 + 400, [("test_act", "passed"), ("test_other", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED, "R-02": VERIFIED}, T0 + 500)   # R-01's entry unchanged since T0+200
    rep = proj.run()
    assert proj.row(rep, "R-01")["group"] == "implemented-needs-reverification"
    assert "issued" in proj.row(rep, "R-01")["verification"]["verdicts"][0]["basis"]
    assert proj.row(rep, "R-02")["group"] == "complete"


def test_a_newer_verdict_supersedes_an_older_lens_only_when_it_says_so(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.verdicts("a", {"R-01": {"status": "partial", "by": "G-a", "notes": "pre-fix"}}, T0 + 50)
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 15\n")
    proj.commit("fix", T0 + 100)
    proj.junit("A", T0 + 150, [("test_act", "passed")])
    proj.verdicts("b", {"R-01": {"status": "verified", "by": "G-b", "notes": "new lens"}}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "implemented-needs-reverification"          # G-a's verdict is stale and not superseded
    proj.verdicts("b", {"R-01": {"status": "verified", "by": "G-b", "notes": "new lens", "supersedes": ["G-a"]}},
                  T0 + 300)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "complete" and [v["by"] for v in r["verification"]["verdicts"]] == ["G-b"]


def test_a_verdict_naming_its_commit_is_fresh_while_the_files_are_unchanged(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    head = proj.git("rev-parse", "HEAD").strip()
    proj.write(f"src/{proj.pkg}/lonely.py", "def unused():\n    return 3\n")   # an unrelated file changes later
    proj.commit("unrelated", T0 + 300)
    proj.verdicts("a", {"R-01": dict(VERIFIED, commit=head)}, T0 + 50)       # issued "before", but for that commit
    assert proj.row(proj.run(), "R-01")["group"] == "complete"
    proj.write(f"src/{proj.pkg}/core.py", "def act():\n    return 16\n")
    proj.commit("change", T0 + 400)
    proj.junit("A2", T0 + 500, [("test_act", "passed")])
    assert proj.row(proj.run(), "R-01")["group"] == "implemented-needs-reverification"


def test_missing_files_placeholders_and_test_directories(proj):
    proj.req("R-01")
    proj.req("R-02")
    proj.req("R-03")
    proj.status("A", {"R-01": proj.claim([], ["tests/test_core.py::test_act"], [f"{proj.pkg}.core.act"]),
                      "R-02": proj.claim([f"src/{proj.pkg}/stub.py"], ["tests/test_core.py::test_act"],
                                         [f"{proj.pkg}.stub.later"]),
                      "R-03": proj.claim([f"src/{proj.pkg}/core.py"], ["tests/"], [f"{proj.pkg}.core.act"])})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {k: VERIFIED for k in ("R-01", "R-02", "R-03")}, T0 + 200)
    rep = proj.run()
    assert proj.row(rep, "R-01")["group"] == "missing"
    r2 = proj.row(rep, "R-02")
    assert r2["group"] == "partial" and any("empty function later()" in p for p in r2["problems"])
    r3 = proj.row(rep, "R-03")
    assert r3["group"] == "partial" and any("not a test function" in p for p in r3["problems"])


def test_integration_points_must_be_reachable_for_the_requirements_kind(proj):
    pk = proj.pkg
    proj.req("R-run-lonely")
    proj.req("R-run-factory")
    proj.req("R-diag", kind="diagnostic")
    proj.req("R-run-diag")
    t = ["tests/test_core.py::test_act"]
    proj.status("A", {"R-run-lonely": proj.claim([f"src/{pk}/lonely.py"], t, [f"{pk}.lonely.unused"]),
                      "R-run-factory": proj.claim([f"src/{pk}/factory_mod.py"], t, [f"{pk}.factory_mod.source"]),
                      "R-diag": proj.claim([f"src/{pk}/diag.py"], t, [f"python -m {pk}.diag"]),
                      "R-run-diag": proj.claim([f"src/{pk}/diag.py"], t, [f"{pk}.diag.check"])},
                commands=[f"PYTHONPATH=src python -m {pk}.diag --seed 0"])
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {k: VERIFIED for k in ("R-run-lonely", "R-run-factory", "R-diag", "R-run-diag")}, T0 + 200)
    rep = proj.run()
    assert f"{pk}.lonely" in rep["unreachable_modules"]
    assert f"{pk}.factory_mod" in rep["entry_points"]["factory"]          # named in tasks/t.yaml
    assert f"{pk}.diag" in rep["entry_points"]["diagnostic"]              # its recorded command produced evidence
    lonely = proj.row(rep, "R-run-lonely")
    assert lonely["group"] == "partial" and any("import-reachable" in p for p in lonely["problems"])
    assert proj.row(rep, "R-run-factory")["group"] == "complete"
    assert proj.row(rep, "R-diag")["group"] == "complete"
    rd = proj.row(rep, "R-run-diag")                                       # a runtime claim on a diagnostic module
    assert rd["group"] == "partial" and any("from the runtime entry points" in p for p in rd["problems"])


def test_console_script_commands_programs_and_symbols_resolve_or_are_reported(proj):
    pk = proj.pkg
    proj.req("R-cli")
    proj.req("R-bad")
    t = ["tests/test_core.py::test_act"]
    proj.status("A", {"R-cli": proj.claim([f"src/{pk}/runtime/cli.py"], t, ["synth run --task x.yaml"]),
                      "R-bad": proj.claim([f"src/{pk}/runtime/cli.py"], t,
                                          ["synth nosuch", f"python -m {pk}.core", f"{pk}.core.missing_name",
                                           "<run_dir>/auth/STOP"])})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-cli": VERIFIED, "R-bad": VERIFIED}, T0 + 200)
    rep = proj.run()
    assert proj.row(rep, "R-cli")["group"] == "complete"
    probs = " | ".join(proj.row(rep, "R-bad")["problems"])
    assert "no subcommand 'nosuch'" in probs and "not runnable as a program" in probs
    assert "missing_name" in probs and "not a dotted name" in probs


def test_executed_evidence_must_exist_succeed_and_be_newer_than_the_code(proj):
    pk = proj.pkg
    c = _core_req(proj, evidence=True, level=2)
    proj.status("A", {"R-01": c})
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED}, T0 + 200)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "partial" and any("needs executed evidence" in p for p in r["problems"])
    head = proj.git("rev-parse", "HEAD").strip()
    proj.write("docs/evidence/A/run.json", json.dumps({"executed": True, "ok": False, "requirements": ["R-01"],
                                                       "git_head": head}))
    r = proj.row(proj.run(), "R-01")
    assert any("reports a failed run" in p for p in r["problems"])
    proj.write("docs/evidence/A/run.json", json.dumps({"executed": True, "ok": True, "requirements": ["R-01"],
                                                       "git_head": head}))
    rep = proj.run()
    assert proj.row(rep, "R-01")["group"] == "complete"
    assert rep["level_detail"][vc.LEVEL_NAMES[2]]["own_criteria_hold"] is True
    proj.write(f"src/{pk}/core.py", "def act():\n    return 17\n")
    proj.commit("change after the experiment", T0 + 300)
    proj.junit("A2", T0 + 400, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": dict(VERIFIED, notes="again")}, T0 + 500)
    r = proj.row(proj.run(), "R-01")
    assert r["group"] == "implemented-needs-reverification" and "executed evidence stale" in r["reason"]
    # evidence named by the owner with a recorded command counts too (project convention)
    proj.write("docs/evidence/A/m.json", json.dumps({"command": "python -m x", "finished_unix": T0 + 600}))
    c["evidence"] = ["docs/evidence/A/m.json"]
    proj.status("A", {"R-01": c})
    assert proj.row(proj.run(), "R-01")["group"] == "complete"


def test_document_requirements_need_files_a_claim_and_a_current_verdict_only(proj):
    proj.req("D-01", kind="none")
    proj.req("D-02", kind="none", declared={"files": ["docs/GUIDE.md"]})
    proj.req("D-03", kind="none", declared={"files": ["docs/NOT_WRITTEN.md"]})
    proj.status("A", {"D-01": proj.claim(["docs/GUIDE.md"])})
    proj.verdicts("a", {"D-01": VERIFIED, "D-02": VERIFIED}, T0 + 200)
    rep = proj.run()
    assert proj.row(rep, "D-01")["group"] == "complete"
    d2 = proj.row(rep, "D-02")                     # declared facts are not a claim
    assert d2["group"] == "partial" and d2["claimed_state"] == "unclaimed"
    assert proj.row(rep, "D-03")["group"] == "missing"


def test_the_owners_claim_decides_and_suffixed_status_files_belong_to_their_owner(proj):
    proj.req("R-01", "M")
    proj.req("R-02", "M")
    c = proj.claim([f"src/{proj.pkg}/core.py"], ["tests/test_core.py::test_act"], [f"{proj.pkg}.core.act"])
    proj.status("M-int", {"R-01": dict(c, state="partial"), "R-02": c})
    proj.status("F", {"R-01": c})                  # another workstream's "implemented" does not override the owner
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"R-01": VERIFIED, "R-02": VERIFIED}, T0 + 200)
    rep = proj.run()
    assert proj.row(rep, "R-01")["claimed_state"] == "partial" and proj.row(rep, "R-01")["group"] == "partial"
    assert proj.row(rep, "R-02")["group"] == "complete"
    assert proj.row(rep, "R-01")["claimed_by"] == ["F", "M-int"]


def test_status_levels_are_computed_separately(proj):
    proj.req("L1-a", level=1)
    proj.req("L2-a", level=2, evidence=True)
    proj.req("L3-a", level=3)
    t = ["tests/test_core.py::test_act"]
    proj.status("A", {"L1-a": proj.claim([f"src/{proj.pkg}/core.py"], t, [f"{proj.pkg}.core.act"], state="partial"),
                      "L2-a": proj.claim([f"src/{proj.pkg}/core.py"], t, [f"{proj.pkg}.core.act"]),
                      "L3-a": proj.claim([f"src/{proj.pkg}/core.py"], t, [f"{proj.pkg}.core.act"])})
    proj.write("docs/evidence/A/run.json", json.dumps({"executed": True, "requirements": ["L2-a"],
                                                       "finished_unix": T0 + 150}))
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    proj.verdicts("a", {"L1-a": VERIFIED, "L2-a": VERIFIED, "L3-a": VERIFIED}, T0 + 200)
    rep = proj.run()
    d = rep["level_detail"]
    assert d[vc.LEVEL_NAMES[1]]["own_criteria_hold"] is False
    assert d[vc.LEVEL_NAMES[2]]["own_criteria_hold"] is True and d[vc.LEVEL_NAMES[2]]["cumulative"] is False
    assert d[vc.LEVEL_NAMES[3]]["own_criteria_hold"] is False          # no preflight evidence recorded
    assert rep["live"]["blockers"]
    proj.write("docs/evidence/preflight.json", json.dumps({"live": {"ready": True, "blocked": False, "missing": []}}))
    rep = proj.run()
    assert rep["level_detail"][vc.LEVEL_NAMES[3]]["own_criteria_hold"] is True
    assert not any(rep["levels"].values())                             # level 1 does not hold, so nothing above


def test_completion_md_has_every_group_counts_and_the_regeneration_command(proj):
    proj.status("A", {"R-01": _core_req(proj)})
    proj.req("R-02")
    proj.junit("A", T0 + 100, [("test_act", "passed")])
    rep = proj.run(md=True)
    text = open(os.path.join(proj.root, "docs", "COMPLETION.md")).read()
    for g in vc.GROUPS:
        assert f"## {g} (" in text
    assert "python -m cam_live.verify_completion --merge-status" in text
    assert f"Requirements: **{rep['counts']['total']}**" in text
    assert "| R-02 |" in text and "## Status levels" in text and "## Durability" in text


def test_the_standalone_scan_flags_imports_of_unfinished_projects(proj):
    proj.write(f"src/{proj.pkg}/bad.py", "import superlearner.core\nfrom crucible import x\n")
    found = vc.standalone_findings(proj.paths)
    assert len(found) == 2 and "superlearner.core" in found[0] and "crucible" in found[1]


def test_durability_reports_unpushed_commits_and_uncommitted_files(proj):
    proj.write("notes.txt", "x\n")
    rep = proj.run()
    d = rep["durability"]
    assert d["git"] is True and d["remote_head"] == "" and d["remote_contains_head"] is False
    assert "notes.txt" in d["untracked"]
    assert d["state_root_in_git"] is False


# ---------------------------------------------------------------------------------------------------------------
# the real project
def test_the_cam_live_source_is_standalone():
    assert vc.standalone_findings(vc.Paths()) == []


def _spec() -> str:
    return open(os.path.join(PROJECT, "docs", "CAM_LIVE_SPEC.md"), encoding="utf-8").read()


def _between(t: str, a: str, b: str) -> str:
    return t[t.index(a):t.index(b)]


def test_the_manifest_covers_every_enumerated_mandatory_item_of_the_specification():
    m = json.load(open(os.path.join(PROJECT, "docs", "CAM_LIVE_COMPLETION_MANIFEST.json")))
    ids = [r["id"] for r in m["requirements"]]
    assert len(ids) == len(set(ids))
    for r in m["requirements"]:
        assert r["workstream"] and r["spec_ref"] and r["title"] and r["kind"] in ("runtime", "evaluation",
                                                                                     "diagnostic", "none")
    cov = m["coverage"]
    t = _spec()
    parsed = {
        "18": re.findall(r"^- \[ \] (.+)$", _between(t, "# PART 18", "# PART 19"), re.M),
        "12A.17": re.findall(r"^\* (.+)$", _between(t, "17. Mandatory acceptance tests",
                                                    "Test general-purpose interaction across"), re.M),
        "12A.17 controlled tests": re.findall(r"^\* (.+)$", _between(t, "Include controlled tests with:",
                                                                     "Compare primitive-only operation"), re.M),
        "17": re.findall(r"^`(cam-live[^`]*)`$", _between(t, "# PART 17", "# PART 18"), re.M),
        "20": re.findall(r"^\d+\. (.+)$", _between(t, "# PART 20", "# FINAL DIRECTIVE"), re.M),
        "obstacle resolution": re.findall(r"^\d+\. (.+)$", _between(t, "CAM must independently:",
                                                                    "CAM should be capable of abandoning"), re.M),
        "15.2": re.findall(r"^- (.+)$", _between(t, "## 15.2", "Support restart after"), re.M),
        "12.6": re.findall(r"^- (.+)$", _between(t, "## 12.6", "## 12.7"), re.M),
        "13.2": re.findall(r"^- (.+)$", _between(t, "## 13.2", "## 13.3"), re.M),
    }
    assert len(parsed["18"]) == 94 and len(parsed["12A.17"]) == 30 and len(parsed["20"]) == 20
    for sec, items in parsed.items():
        covered = {c["item"]: c["ids"] for c in cov[sec]}
        missing = [i for i in items if i not in covered]
        assert not missing, (sec, missing)
        for i in items:
            assert covered[i] and all(x in ids for x in covered[i]), (sec, i, covered[i])
    s134 = re.findall(r"^- (.+)$", _between(t, "## 13.4", "# PART 14"), re.M)
    assert all(any(c["item"] == i for c in cov["13.4"]) for i in s134)
    for sec, items in cov.items():
        for c in items:
            assert all(x in ids for x in c["ids"]), (sec, c)


def test_the_operator_amendments_have_manifest_entries():
    m = json.load(open(os.path.join(PROJECT, "docs", "CAM_LIVE_COMPLETION_MANIFEST.json")))
    by = {r["id"]: r for r in m["requirements"]}
    for rid, words in (("OP-01", "free-space"), ("OP-02", "retention"), ("OP-03", "monitoring"),
                       ("OP-04", "designated server"), ("OP-05", "experience pipeline"), ("OP-06", "operator"),
                       ("OP-07", "Credential"), ("OP-08", "no durable backup"), ("OP-09", "CAPTCHA"),
                       ("OP-10", "full-size"), ("OP-11", "verbatim")):
        assert words in by[rid]["title"], (rid, by[rid]["title"])
        assert by[rid]["spec_ref"].startswith("operator amendment")


def test_an_overridden_base_method_is_not_a_placeholder_but_an_unoverridden_one_is(proj):
    pk = proj.pkg
    proj.write(f"src/{pk}/spec.py",
               "class Base:\n    def check(self, v):\n        raise NotImplementedError  # every subclass overrides\n\n\n"
               "class A(Base):\n    def check(self, v):\n        return 1\n\n\n"
               "class Half:\n    def run(self):\n        raise NotImplementedError\n\n\n"
               "class B(Half):\n    def other(self):\n        return 2\n")
    found = vc.placeholder_findings(os.path.join(proj.root, "src", pk, "spec.py"), proj.paths)
    assert len(found) == 1 and ":13:" in found[0]


def test_a_no_op_override_of_a_library_hook_is_not_a_placeholder(proj):
    pk = proj.pkg
    proj.write(f"src/{pk}/srv.py",
               "from http.server import BaseHTTPRequestHandler\n\n\n"
               "def serve():\n    class H(BaseHTTPRequestHandler):\n        def log_message(self, *a):\n            pass\n\n"
               "        def not_a_hook(self):\n            pass\n    return H\n")
    found = vc.placeholder_findings(os.path.join(proj.root, "src", pk, "srv.py"), proj.paths)
    assert len(found) == 1 and "not_a_hook" in found[0]


def test_markers_count_in_comments_docstrings_and_raises_not_in_string_data(proj):
    pk = proj.pkg
    proj.write(f"src/{pk}/marks.py",
               'MARKERS = ("TODO", "FIXME")      # what the scanner looks for\n\n\n'
               'def a():\n    return 1  # TODO handle the other case\n\n\n'
               'def b():\n    """FIXME: wrong for negative input."""\n    return 2\n\n\n'
               'def c():\n    raise NotImplementedError("later")\n')
    found = vc.placeholder_findings(os.path.join(proj.root, "src", pk, "marks.py"), proj.paths)
    assert [f.split(":")[1] for f in found] == ["5", "9", "14"], found
