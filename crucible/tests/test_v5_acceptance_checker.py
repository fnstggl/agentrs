"""The v5 acceptance checker verifies evidence itself.

The honest pipeline (official run in a fresh clone, commit, independent review that re-executes the tests, commit)
reaches VERIFIED. Every fabrication probe of the recovery audit (docs/v5/audit/P_acceptance.md, probes P1-P14) and
the further attacks below must fail to produce VERIFIED. Each test builds a throwaway git repository
(`acceptance_kit.Repo`) containing the real acceptance package and runs everything for real."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from acceptance_kit import CRIT, GOOD_TEST, Repo, git
from crucible.acceptance.v5 import checker, evidence, ledger as L, manifest as M, testscan

pytestmark = pytest.mark.timeout(300)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return Repo(str(tmp_path), monkeypatch)


def not_verified(r: Repo, rid: str = "Z.1"):
    res, rows = r.verify()
    assert rows[rid]["status"] != "VERIFIED", rows[rid]
    assert not res["ok"] and res["project_status"] != "CRUCIBLE_5_COMPLETE"
    return res, rows[rid]


def handwrite(r: Repo, *, metrics=None, reviewer="ws-P") -> None:
    """The audit's P1 forgery: evidence, JUnit and review written by hand; no test is run."""
    jd = os.path.join(r.root, "docs/v5/evidence/gate_Z")
    os.makedirs(jd, exist_ok=True)
    open(f"{jd}/junit_fake.xml", "w").write(
        '<testsuites><testsuite name="x"><testcase classname="tests.v5.test_gate_Z" name="test_z"/></testsuite>'
        '</testsuites>')
    ev = {"schema": evidence.SCHEMA, "requirement": "Z.1", "gate": "Z", "official": True,
          "produced_by": "tests/v5/test_gate_Z.py::test_z", "author": "builder", "run_id": "fake",
          "tree": {"commit": git(r.top, "rev-parse", "HEAD"), "clean": True, "dirty_paths": []},
          "metrics": metrics or {"widgets": 10 ** 9, "leaks": 0}, "artifacts": [], "modules": {},
          "junit": {"path": "docs/v5/evidence/gate_Z/junit_fake.xml",
                    "sha256": evidence.sha256_file(f"{jd}/junit_fake.xml")}}
    r.dump("docs/v5/evidence/gate_Z/Z.1.json", ev)
    r.dump("docs/v5/reviews/Z.1.json", {"schema": evidence.REVIEW_SCHEMA, "requirement": "Z.1", "verdict": "approved",
                                        "evidence_sha256": evidence.sha256_file(f"{jd}/Z.1.json"),
                                        "reviewer": {"agent": reviewer}})
    r.commit("fabricated evidence")


# ------------------------------------------------------------------------------------------------- honest path
def test_the_honest_pipeline_verifies_and_completes(repo):
    res, rows = repo.verify()
    assert rows["Z.1"]["status"] == "IN_PROGRESS" and res["project_status"] == "IN_PROGRESS"
    repo.official_run()
    repo.commit("evidence")
    res, rows = repo.verify()
    assert rows["Z.1"]["status"] == "IMPLEMENTED_UNVERIFIED" and rows["Z.1"]["problems"] == ["no independent review"]
    repo.review()
    repo.commit("review")
    res, rows = repo.verify()
    assert rows["Z.1"]["status"] == "VERIFIED", rows["Z.1"]["problems"]
    # static checks never decide the project status: only the release check (re-execution at HEAD) does
    assert not res["ok"] and res["project_status"] == "IN_PROGRESS" and res["global_problems"] == [checker.RELEASE_NOTE]
    repo.mp.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = repo.verify(reproduce=True, agent="release-check", scratch=repo.scratch)
    assert res["ok"] and res["project_status"] == "CRUCIBLE_5_COMPLETE" and not res["global_problems"]


def test_uncommitted_evidence_and_reviews_do_not_count(repo):
    repo.official_run()
    _, row = not_verified(repo)
    assert any("not committed" in p for p in row["problems"])


def test_reproduce_mode_reexecutes_verified_requirements(repo, monkeypatch):
    repo.honest()
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-checker")
    res, rows = repo.verify(reproduce=True, agent="release-check", scratch=repo.scratch)
    assert rows["Z.1"]["status"] == "VERIFIED" and rows["Z.1"]["reproduction"]["mismatches"] == []
    assert res["project_status"] == "CRUCIBLE_5_COMPLETE"


# ------------------------------------------------------------------------------------------------- audit probes
def test_P1_handwritten_evidence_junit_and_review_are_rejected(repo):
    handwrite(repo)
    _, row = not_verified(repo)
    assert any("no run record" in p for p in row["problems"])
    assert any("imported-module hashes" in p for p in row["problems"])


def test_P2_an_empty_manifest_never_completes(repo):
    m = repo.manifest()
    m["requirements"] = []
    repo.save_manifest(m)
    repo.commit("empty manifest")
    res, rows = repo.verify()
    assert not res["ok"] and res["project_status"] == "IN_PROGRESS"
    assert any("no requirements" in p for p in res["global_problems"])
    assert rows["Z.1"]["status"] == "NOT_STARTED" and "missing from the manifest" in rows["Z.1"]["problems"]


def test_P3_P13_deleting_a_requirement_is_detected(tmp_path, monkeypatch):
    items = [("Z.1", "Z", "make widgets", "A", [], CRIT, 3),
             ("Z.2", "Z", "a hard one", "A", [], [{"metric": "hard", "op": ">=", "value": 1}], 3)]
    r = Repo(str(tmp_path), monkeypatch, items=items)
    r.honest()
    m = r.manifest()
    m["requirements"] = [e for e in m["requirements"] if e["id"] != "Z.2"]
    r.save_manifest(m)
    r.commit("drop Z.2 from the manifest")
    res, rows = r.verify()
    assert not res["ok"] and any("missing from the manifest" in p for p in res["global_problems"])
    # removing it from requirements.py too is still detected: the ledger remembers it
    r.items = items[:1]
    r.write_requirements()
    r.commit("drop Z.2 from requirements.py")
    res, _ = r.verify()
    assert not res["ok"] and any("were removed" in p for p in res["global_problems"])
    with pytest.raises(ValueError, match="never be removed"):
        M.build(r.root)


def test_P4_weakening_criteria_by_rewriting_history_is_detected(repo):
    repo.honest()
    weak = [{"metric": "widgets", "op": ">=", "value": 0}]
    repo.items = [("Z.1", "Z", "make widgets", "A", [], weak, 3)]
    repo.write_requirements()
    m = repo.manifest()
    m["requirements"][0]["criteria"] = weak
    m["requirements"][0]["criteria_history"][0]["criteria"] = weak
    repo.save_manifest(m)
    repo.commit("weaken")
    _, row = not_verified(repo)
    assert any("differ from the ledger" in p or "differs from the ledger" in p for p in row["problems"])


def test_P4b_rewriting_the_ledger_itself_is_detected(repo):
    repo.honest()
    entries = L.read(repo.root)
    i = next(k for k, e in enumerate(entries) if e["kind"] == "criteria")
    entries[i]["value"] = [{"metric": "widgets", "op": ">=", "value": 0}]
    prev = L.GENESIS
    for e in entries:                                   # a careful forger re-chains every hash
        e["prev"] = prev
        e["hash"] = L.entry_hash(e)
        prev = e["hash"]
    with open(L.path(repo.root), "w") as fh:
        fh.write("".join(L.canonical(e) + "\n" for e in entries))
    repo.commit("rewrite the ledger")
    res, _ = repo.verify()
    assert not res["ok"] and any("rewrites the ledger" in p for p in res["global_problems"])


def test_a_weaker_criteria_change_is_refused_even_with_a_decision(repo):
    weak = [{"metric": "widgets", "op": ">=", "value": 5}, {"metric": "leaks", "op": "==", "value": 0}]
    repo.items = [("Z.1", "Z", "make widgets", "A", [], weak, 3)]
    repo.write_requirements()
    with pytest.raises(ValueError, match="not provably at least as strict"):
        M.build(repo.root, decision="D-1")


def test_a_stricter_criteria_change_needs_a_decision_naming_the_requirement(repo):
    strict = [{"metric": "widgets", "op": ">=", "value": 11}, {"metric": "leaks", "op": "==", "value": 0}]
    repo.items = [("Z.1", "Z", "make widgets", "A", [], strict, 3)]
    repo.write_requirements()
    with pytest.raises(ValueError, match="cite a decision"):
        M.build(repo.root)
    repo.write("docs/v5/DECISION_LOG.md", repo.read("docs/v5/DECISION_LOG.md") + "\n## D-2 unrelated\nnothing\n")
    with pytest.raises(ValueError, match="does not name Z.1"):
        M.build(repo.root, decision="D-2")
    M.build(repo.root, decision="D-1")
    repo.commit("stricter")
    repo.honest()
    res, rows = repo.verify()
    assert rows["Z.1"]["status"] == "VERIFIED", rows["Z.1"]["problems"]
    hist = repo.manifest()["requirements"][0]["criteria_history"]
    assert [h["decision"] for h in hist] == [None, "D-1"]


def test_P5_a_dirty_tree_is_never_measured_and_never_verifies(repo):
    repo.write("src/widgets/__init__.py", "def count():\n    return 99\n\n\ndef leaks():\n    return 0\n")
    out = repo.official_run()                           # measures the committed tree, not the dirty one
    rec = repo.json(out["installed"]["Z.1"])
    assert rec["metrics"]["widgets"] == 12 and rec["tree"]["clean"]
    repo.commit("evidence")       # commits the dirty change too: the evidence is now stale
    repo.review()
    repo.commit("review")
    _, row = not_verified(repo)
    assert any("stale" in p for p in row["problems"])


def test_P5b_forging_clean_and_module_hashes_by_hand_fails(repo):
    repo.honest()
    ev = repo.json("docs/v5/evidence/gate_Z/Z.1.json")
    ev["modules"]["src/widgets/__init__.py"] = evidence.git_blob_id(b"def count():\n    return 99\n")
    repo.dump("docs/v5/evidence/gate_Z/Z.1.json", ev)
    repo.commit("forge")
    _, row = not_verified(repo)
    assert any("do not match the commit" in p for p in row["problems"])
    assert any("run record does not hold this record's hash" in p for p in row["problems"])


def test_P6_failing_or_uncollected_cases_block_verification(tmp_path, monkeypatch):
    src = GOOD_TEST + "\n\ndef test_hard():\n    assert 1 + 1 == 3, 'fails'\n"
    r = Repo(str(tmp_path), monkeypatch, test_src=src)
    r.official_run()
    r.commit("evidence")
    _, row = not_verified(r)
    assert any("test_hard" in p for p in row["problems"]) and any("pytest exit code" in p for p in row["problems"])
    # a JUnit edited down to the passing case (hash and run record fixed up) is still incomplete
    ev = r.json("docs/v5/evidence/gate_Z/Z.1.json")
    jp = os.path.join(r.root, ev["junit"]["path"])
    xml = open(jp).read()
    start = xml.index('<testcase classname="tests.v5.test_gate_Z" name="test_hard"')
    end = xml.index("</testcase>", start) + len("</testcase>")
    open(jp, "w").write(xml[:start] + xml[end:])
    ev["junit"]["sha256"] = evidence.sha256_file(jp)
    r.dump("docs/v5/evidence/gate_Z/Z.1.json", ev)
    run = r.json(ev["run"]["path"])
    run["evidence"]["Z.1"] = evidence.sha256_file(os.path.join(r.root, "docs/v5/evidence/gate_Z/Z.1.json"))
    run["pytest_rc"] = 0
    run["collected"] = ["tests/v5/test_gate_Z.py::test_z"]
    r.dump(ev["run"]["path"], run)
    r.commit("forge a complete-looking JUnit")
    _, row = not_verified(r)
    assert any("test_hard exists at the commit but was not collected" in p for p in row["problems"])


def test_P7_placeholder_tests_do_not_verify(tmp_path, monkeypatch):
    src = ('from crucible.acceptance.v5 import evidence\n\n\ndef test_z():\n    try:\n        import widgets\n'
           '    except Exception:\n        return\n    evidence.record("Z.1", {"widgets": widgets.count(), "leaks": 0})\n')
    r = Repo(str(tmp_path), monkeypatch, test_src=src)
    r.honest()
    _, row = not_verified(r)
    assert any("not a genuine measurement" in p and "early_return" in p for p in row["problems"])
    assert any("no assertion on measured values" in p for p in row["problems"])


def test_P8_a_producing_test_outside_the_requirement_is_rejected(repo):
    repo.honest()
    ev = repo.json("docs/v5/evidence/gate_Z/Z.1.json")
    ev["produced_by"] = "tests/unrelated.py::test_other"
    repo.dump("docs/v5/evidence/gate_Z/Z.1.json", ev)
    run = repo.json(ev["run"]["path"])
    run["evidence"]["Z.1"] = evidence.sha256_file(os.path.join(repo.root, "docs/v5/evidence/gate_Z/Z.1.json"))
    repo.dump(ev["run"]["path"], run)
    repo.commit("forge produced_by")
    _, row = not_verified(repo)
    assert any("not one of the requirement's tests" in p for p in row["problems"])


def test_P9_non_finite_metrics_fail(repo):
    with pytest.raises(TypeError):
        evidence.record("Z.1", {"widgets": float("inf"), "leaks": 0})
    handwrite(repo, metrics={"widgets": float("inf"), "leaks": 0})
    res, row = not_verified(repo)
    assert row["status"] == "FAILED" and any("not a finite" in p or "non-finite" in p for p in row["problems"])


def test_P10_swapping_the_test_list_is_detected(repo):
    repo.write("tests/v5/test_trivial.py", "def test_t():\n    pass\n")
    m = repo.manifest()
    m["requirements"][0]["tests"] = ["tests/v5/test_trivial.py"]
    repo.save_manifest(m)
    repo.commit("swap tests")
    _, row = not_verified(repo)
    assert any("change-controlled" in p for p in row["problems"])
    assert any("canonical gate test" in p for p in row["problems"])
    # appending the swap to the ledger without a decision is also caught
    st = L.state(L.read(repo.root))
    L.append(repo.root, [{"id": "Z.1", "kind": "tests", "value": ["tests/v5/test_trivial.py"]}])
    M.sync(repo.root)
    repo.commit("swap through the ledger")
    _, row = not_verified(repo)
    assert any("removal from tests" in p and "no recorded decision" in p for p in row["problems"]), st


def test_P11_a_changed_unlisted_imported_module_makes_evidence_stale(tmp_path, monkeypatch):
    test_src = GOOD_TEST.replace("import widgets\n", "import widgets\nimport helper\n")
    r = Repo(str(tmp_path), monkeypatch, test_src=test_src, extra_files={"src/helper.py": "X = 1\n"})
    r.honest()
    assert r.verify()[1]["Z.1"]["status"] == "VERIFIED"
    r.write("src/helper.py", "X = 2  # changed behaviour\n")
    r.commit("change a dependency the test imports")
    _, row = not_verified(r)
    assert any("stale evidence" in p and "src/helper.py" in p for p in row["problems"])


def test_P12_a_test_that_records_constants_and_asserts_nothing_fails(tmp_path, monkeypatch):
    src = ('from crucible.acceptance.v5 import evidence\n\n\ndef test_z():\n'
           '    evidence.record("Z.1", {"widgets": 25, "leaks": 0})\n')
    r = Repo(str(tmp_path), monkeypatch, test_src=src)
    r.honest()
    _, row = not_verified(r)
    assert any("constant_record" in p for p in row["problems"])
    assert any("imported none of the listed implementation modules" in p for p in row["problems"])


def test_P12b_the_cli_refuses_an_anonymous_official_run(repo, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_V5_AGENT", raising=False)
    with pytest.raises(SystemExit):
        repo.official_run(agent="")


def test_P14_weakening_a_real_criterion_in_memory_is_reported():
    m = M.load()
    e = next(x for x in m["requirements"] if x["id"] == "B.1")
    e["criteria"] = [{"metric": "distinct_participants", "op": ">=", "value": 1}]
    for h in e["criteria_history"]:
        h["criteria"] = e["criteria"]
    cons = M.consistency(m=m)
    assert any("differs from derive()" in p for p in cons["per_id"]["B.1"])
    assert any("criteria_history differs from the ledger" in p for p in cons["per_id"]["B.1"])


# ------------------------------------------------------------------------------------------------- reviews
def test_self_review_is_refused_and_a_forged_self_review_is_rejected(repo):
    repo.official_run()
    repo.commit("evidence")
    with pytest.raises(SystemExit, match="same agent"):
        repo.review(agent="builder", session="s-other")
    with pytest.raises(SystemExit, match="same session"):
        repo.review(agent="qa", session="s-builder")
    repo.review()
    rv = repo.json("docs/v5/reviews/Z.1.json")
    rv["reviewer"]["agent"] = "builder"
    repo.dump("docs/v5/reviews/Z.1.json", rv)
    repo.commit("review")
    _, row = not_verified(repo)
    assert any("same agent name" in p for p in row["problems"])


def test_a_nonreproducible_measurement_is_rejected_by_review(tmp_path, monkeypatch):
    src = GOOD_TEST.replace('{"widgets": n, "leaks": leaks}', '{"widgets": n, "leaks": leaks, "noise": os.urandom(4).hex()}')
    src = "import os\n" + src
    r = Repo(str(tmp_path), monkeypatch, test_src=src)
    r.official_run()
    r.commit("evidence")
    rv = r.review()
    assert rv["verdict"] == "rejected" and any("noise" in x for x in rv["reproduction"]["mismatches"])
    r.commit("review")
    _, row = not_verified(r)
    assert any("review verdict 'rejected'" in p for p in row["problems"])


def test_a_review_is_bound_to_the_criteria_version(repo):
    repo.honest()
    strict = [{"metric": "widgets", "op": ">=", "value": 11}, {"metric": "leaks", "op": "==", "value": 0}]
    repo.items = [("Z.1", "Z", "make widgets", "A", [], strict, 3)]
    repo.write_requirements()
    M.build(repo.root, decision="D-1")
    repo.commit("stricter criteria")
    _, row = not_verified(repo)
    assert any("different criteria version" in p for p in row["problems"])


def test_editing_evidence_after_review_is_detected(repo):
    repo.honest()
    ev = repo.json("docs/v5/evidence/gate_Z/Z.1.json")
    ev["metrics"]["widgets"] = 13
    repo.dump("docs/v5/evidence/gate_Z/Z.1.json", ev)
    repo.commit("edit")
    _, row = not_verified(repo)
    assert any("different version of the evidence" in p for p in row["problems"])


def test_stale_evidence_is_rejected_after_the_implementation_changes(repo):
    repo.honest()
    repo.write("src/widgets/__init__.py", "def count():\n    return 3\n\n\ndef leaks():\n    return 0\n")
    repo.commit("change")
    _, row = not_verified(repo)
    assert any("stale" in p for p in row["problems"])


def test_a_criterion_is_evaluated_on_the_metrics_not_trusted(tmp_path, monkeypatch):
    r = Repo(str(tmp_path), monkeypatch, widgets="def count():\n    return 9\n\n\ndef leaks():\n    return 0\n",
             test_src=GOOD_TEST.replace("assert n >= 10 and leaks == 0", "assert n >= 0 and leaks == 0"))
    r.official_run()
    r.commit("evidence")
    res, rows = r.verify()
    assert rows["Z.1"]["status"] == "FAILED" and not res["ok"]


def test_a_claimed_status_is_not_trusted(repo):
    m = repo.manifest()
    m["requirements"][0]["status"] = "VERIFIED"
    repo.save_manifest(m)
    repo.commit("claim")
    _, row = not_verified(repo)
    assert "manifest claims VERIFIED" in row["problems"][0]


# ------------------------------------------------------------------------------------------------- status vocabulary
def test_incomplete_blocked_only_when_every_unverified_requirement_cites_an_open_blocker(tmp_path, monkeypatch):
    items = [("Z.1", "Z", "make widgets", "A", [], CRIT, 3),
             ("Z.2", "Z", "a hard one", "A", [], [{"metric": "hard", "op": ">=", "value": 1}], 3)]
    r = Repo(str(tmp_path), monkeypatch, items=items)
    r.honest()
    res, _ = r.verify()
    assert res["project_status"] == "IN_PROGRESS"
    m = r.manifest()
    z2 = next(e for e in m["requirements"] if e["id"] == "Z.2")
    z2.update(status="BLOCKED", blocker="BLK-9")          # not an open blocker
    r.save_manifest(m)
    r.commit("blocked by an unknown blocker")
    assert r.verify()[0]["project_status"] == "IN_PROGRESS"
    z2["blocker"] = "BLK-1"
    r.save_manifest(m)
    r.commit("blocked by an open blocker")
    res, rows = r.verify()
    assert rows["Z.2"]["status"] == "BLOCKED" and res["project_status"] == "IN_PROGRESS"   # terminal only on release
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = r.verify(reproduce=True, agent="release-check", scratch=r.scratch)
    assert rows["Z.2"]["status"] == "BLOCKED" and res["project_status"] == "INCOMPLETE_BLOCKED" and not res["ok"]


def test_dependencies_must_be_verified(tmp_path, monkeypatch):
    items = [("Z.1", "Z", "make widgets", "A", ["Y.1"], CRIT, 3),
             ("Y.1", "Y", "a prerequisite", "A", [], [{"metric": "hard", "op": ">=", "value": 1}], 3)]
    r = Repo(str(tmp_path), monkeypatch, items=items)
    r.honest()
    _, rows = r.verify()
    assert rows["Z.1"]["status"] == "IMPLEMENTED_UNVERIFIED"
    assert any("dependency Y.1 not verified" in p for p in rows["Z.1"]["problems"])


# ------------------------------------------------------------------------------------------------- units
def test_strictness_relation():
    ge = lambda v: {"metric": "m", "op": ">=", "value": v}
    assert L.implies(ge(10), ge(5)) and not L.implies(ge(5), ge(10))
    assert L.implies({"metric": "m", "op": "==", "value": 0}, {"metric": "m", "op": "<=", "value": 2})
    assert not L.implies({"metric": "m", "op": "<=", "value": 2}, {"metric": "m", "op": "==", "value": 0})
    assert L.implies({"metric": "m", "op": "==", "value": True}, {"metric": "m", "op": "==", "value": True})
    assert not L.implies({"metric": "m", "op": "==", "value": 1}, {"metric": "m", "op": "==", "value": True})
    assert L.not_implied([ge(10), {"metric": "n", "op": "==", "value": 0}], [ge(5)]) == []
    assert L.not_implied([{"metric": "n", "op": "==", "value": 0}], [ge(5)]) == [ge(5)]


def test_testscan_classifies_placeholders():
    src = '''import pytest
from crucible.acceptance.v5.evidence import record

def test_ok(x):
    v = x.measure()
    record("A", {"m": v})
    assert v > 0

def test_const():
    record("A", {"m": 1})

def test_const_then_fail():
    record("A", {"m": 0})
    pytest.fail("not measurable yet")

def test_trivial():
    assert True

@pytest.mark.skip(reason="later")
def test_skipped():
    assert f() == 1

def test_swallow():
    try:
        g()
    except Exception:
        pass
    assert h()

def test_todo():
    # TODO: measure
    assert k()
'''
    f = {x["name"]: x["findings"] for x in testscan.scan_source(src)}
    assert f["test_ok"] == []
    assert "constant_record" in f["test_const"] and "no_assert" in f["test_const"]
    assert f["test_const_then_fail"] == ["constant_record_then_fail"]
    assert "trivial_assert" in f["test_trivial"]
    assert "skip" in f["test_skipped"] and "swallow" in f["test_swallow"] and "placeholder_text" in f["test_todo"]


def test_compare_metrics_tolerance():
    from crucible.acceptance.v5.official import compare_metrics
    assert compare_metrics({"a": 1, "b": 0.5, "c": True}, {"a": 1, "b": 0.5 + 1e-12, "c": True}) == []
    assert compare_metrics({"a": 1}, {"a": 2})
    assert compare_metrics({"b": 0.5}, {"b": 0.6})
    assert compare_metrics({"b": 0.5}, {"b": 0.6}, {"b": {"abs": 0.2}}) == []
    assert compare_metrics({"c": True}, {"c": 1})


def test_unofficial_records_never_land_in_the_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_V5_RUN_ID", "made-up")
    monkeypatch.setenv("CRUCIBLE_V5_EVIDENCE_OUT", str(tmp_path / "out"))
    p = evidence.record("Z.9", {"m": 1})                # no clone marker in this checkout's git dir
    assert "unofficial" in p and not json.load(open(p))["official"]


# ------------------------------------------------------------------------------------------------- the real tree
def test_the_real_manifest_covers_every_spec_gate_item():
    gates = M.spec_gates()
    m = M.load()
    ids = {e["id"] for e in m["requirements"]}
    for g, info in gates.items():
        for i in range(1, len(info["bullets"]) + 1):
            assert f"{g}.{i}" in ids
    s1 = next(e for e in m["requirements"] if e["id"] == "S.1")
    assert "CAM" not in s1["requirement"] and s1["source"]["clarification"]


def test_the_real_manifest_is_anchored():
    cons = M.consistency()
    assert cons["global"] == [] or all("uncommitted" in p for p in cons["global"]), cons["global"]
    assert not cons["per_id"], dict(list(cons["per_id"].items())[:5])


def test_the_real_checker_exit_code_follows_the_project_status():
    root = M.crucible_root()
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "crucible-v5-verify-test.json")
    p = subprocess.run([sys.executable, "-m", "crucible.acceptance.v5", "--verify-all", "--out", out],
                       cwd=root, env=dict(os.environ, PYTHONPATH=os.path.join(root, "src")),
                       capture_output=True, text=True)
    res = json.load(open(out))
    assert p.returncode == {"CRUCIBLE_5_COMPLETE": 0, "IN_PROGRESS": 1, "INCOMPLETE_BLOCKED": 3}[res["project_status"]]
    assert p.returncode != 0 or all(r["status"] == "VERIFIED" for r in res["rows"])
    assert res["project_status"] == "IN_PROGRESS" or res["reproduced"]       # static checks never complete


def test_a_merge_that_drops_ledger_entries_of_one_parent_is_detected(tmp_path):
    root = str(tmp_path)
    run = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    os.makedirs(os.path.join(root, "docs", "v5"))
    p = os.path.join(root, L.LEDGER_REL)
    open(p, "w").write("a\n")
    run("add", "-A")
    run("commit", "-qm", "base")
    run("checkout", "-qb", "side")
    open(p, "w").write("a\nb\n")
    run("commit", "-qam", "side appends")
    run("checkout", "-q", "main")
    open(os.path.join(root, "other"), "w").write("x\n")
    run("add", "other")
    run("commit", "-qm", "main moves on")
    run("merge", "-q", "-s", "ours", "side", "-m", "merge that keeps only main's ledger")
    assert any("rewrites the ledger" in x for x in L.git_problems(root))


def test_progress_ledger_records_and_updates_interrupted_workstreams(tmp_path):
    from crucible.acceptance.v5 import progress as P
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "docs", "v5"))
    P.save({"schema": P.SCHEMA, "milestones": [], "interrupted_workstreams": []}, root)
    P.add_interrupted(workstream="Q", why="usage limit", recovery_branch="c5c/w1-q", reassigned_to="Q2", root=root)
    e = P.add_interrupted(workstream="Q", why="usage limit", recovery_branch="c5c/w1-q", reassigned_to="Q2",
                          status="completed", root=root)
    d = P.load(root)
    assert len(d["interrupted_workstreams"]) == 1 and e["status"] == "completed"
    assert [h["status"] for h in e["history"]] == ["reassigned"]
    with pytest.raises(ValueError):
        P.add_interrupted(workstream="Q", why="x", recovery_branch="b", reassigned_to="", root=root)
    with pytest.raises(ValueError):
        P.add_interrupted(workstream="Q", why="x", recovery_branch="b", reassigned_to="R", status="vanished", root=root)
