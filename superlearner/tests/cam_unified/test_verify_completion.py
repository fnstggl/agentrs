"""verify_completion hardening (plan 2026-11-26 §3 W9; audit 'h' findings on the completion checker).

Every failure mode the audit demonstrated, plus the new protocol checks, on synthetic manifests inside a scratch git
repository: the audit's probe (X01 EMPIRICALLY_DEMONSTRATED with verdict 'NOT_DEMONSTRATED (primary CI includes 0)',
a markdown file as implementation, a governance test as acceptance + integration test and integration_entry,
verified_by='anyone') must be rejected on every count; JUnit records must be bound to a commit whose cited files are
unchanged; integration records must be clean of test doubles and must have executed the integration entry; empirical
results must carry an exact, recomputable verdict, a pre-registration committed before the first confirm row, raw
files with matching hashes and exactly one confirm evaluation per (learner, seed). The pytest plugin itself is run
in a subprocess on a scratch test file."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import pytest

import superlearner.verify_completion as vc

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
GIT_ID = ["-c", "user.name=vc-test", "-c", "user.email=vc-test@example.invalid", "-c", "commit.gpgsign=false"]


def git(root, *args, date=None):
    env = dict(os.environ)
    if date is not None:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = f"@{int(date)} +0000"
    r = subprocess.run(["git", *GIT_ID, *args], cwd=root, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)
    return rel


SRC_MOD = "superlearner/src/pkg/feature.py"
TEST_MOD = "superlearner/tests/unit/test_feature.py"
INTEG_MOD = "superlearner/tests/integration/test_feature_real.py"


@pytest.fixture
def repo(tmp_path):
    root = str(tmp_path / "repo")
    os.makedirs(root)
    git(root, "init", "-q")
    write(root, SRC_MOD, "def run(x):\n    return 2 * x\n")
    write(root, TEST_MOD, "def test_feature_doubles():\n    assert True\n")
    write(root, INTEG_MOD, "def test_feature_end_to_end():\n    assert True\n")
    write(root, "superlearner/tests/cam_unified/test_manifest.py", "def test_manifest_schema_and_statuses():\n    pass\n")
    write(root, "superlearner/docs/notes.md", "# a design document\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base", date=1_700_000_000)
    vc.clear_caches()
    return root


def manifest(*reqs):
    return {"allowed_statuses": vc.STATUSES, "workstreams": {"A": "a", "B": "b", "H": "h"},
            "requirements": [dict({"workstream": "A", "dependencies": [], "files": [], "acceptance_tests": [],
                                   "integration_tests": [], "evidence": []}, **r) for r in reqs]}


def record(commit, passed=True, dirty=(), integration=None, hits=None):
    rec = {"passed": passed, "skipped": False, "file": "x.xml",
           "props": {"vc_commit": commit, "vc_dirty": json.dumps(list(dirty))}, "entry_hits": hits or {},
           "integration": integration}
    return rec


def problems(rep, rid):
    return rep["problems"].get(rid, [])


def test_the_audit_probe_is_rejected_on_every_count(repo):
    gov = "superlearner/tests/cam_unified/test_manifest.py::test_manifest_schema_and_statuses"
    write(repo, "superlearner/docs/result.json", json.dumps({"verdict": "NOT_DEMONSTRATED (primary CI includes 0)"}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "probe")
    head = git(repo, "rev-parse", "HEAD")
    m = manifest({"id": "X01", "status": "EMPIRICALLY_DEMONSTRATED", "files": ["superlearner/docs/notes.md"],
                  "acceptance_tests": [gov], "integration_tests": [gov],
                  "integration_entry": "superlearner/tests/cam_unified/test_manifest.py:test_manifest_schema_and_statuses",
                  "verified_by": "anyone", "experiment_result": "superlearner/docs/result.json",
                  "evidence": ["it works"]})
    results = {tuple(gov.split("::")): record(head, integration="clean",
                                              hits={m["requirements"][0]["integration_entry"]: 1})}
    rep = vc.check(m, results, root=repo, global_checks=False)
    ps = "\n".join(problems(rep, "X01"))
    assert "X01" not in rep["scientific"]["demonstrated"] and not rep["claims_consistent"]
    for needle in ("without a source file under */src/", "a governance test is not evidence",
                   "integration_entry must be a function of the runtime under */src/",
                   "verified_by must be a verification record", "is not one of", "evidence entry is not a resolvable"):
        assert needle in ps, (needle, ps)


def test_substring_verdicts_no_longer_pass(repo):
    for bad in ("NOT_DEMONSTRATED", "UNDEMONSTRATED", "demonstrated", "DEMONSTRATED (mostly)"):
        write(repo, "superlearner/docs/r.json", json.dumps({"verdict": bad}))
        assert any("is not one of" in p for p in vc.check_experiment_result("superlearner/docs/r.json",
                                                                             "EMPIRICALLY_DEMONSTRATED", repo))
    write(repo, "superlearner/docs/r.json", json.dumps({"verdict": "FAILED"}))
    ps = vc.check_experiment_result("superlearner/docs/r.json", "EMPIRICALLY_DEMONSTRATED", repo)
    assert any("does not support EMPIRICALLY_DEMONSTRATED" in p for p in ps)
    assert vc.check_experiment_result("../../etc/passwd", "FAILED_EMPIRICALLY", repo) == \
        ["experiment_result outside the repository: ../../etc/passwd"]


def test_implemented_claim_with_an_unexecuted_test_fails(repo):
    m = manifest({"id": "A01", "status": "IMPLEMENTED", "files": [SRC_MOD],
                  "acceptance_tests": [f"{TEST_MOD}::test_feature_doubles"]})
    rep = vc.check(m, {}, root=repo, global_checks=False)
    assert any("never executed" in p for p in problems(rep, "A01"))


def test_junit_records_are_bound_to_a_commit_and_to_unchanged_files(repo):
    t = f"{TEST_MOD}::test_feature_doubles"
    m = manifest({"id": "A01", "status": "IMPLEMENTED", "files": [SRC_MOD], "acceptance_tests": [t]})
    key = tuple(t.split("::"))
    c0 = git(repo, "rev-parse", "HEAD")
    assert problems(vc.check(m, {key: record(c0)}, root=repo, global_checks=False), "A01") == []
    # no binding (a record written without the plugin, e.g. the pre-2026-11-26 XML files)
    unbound = {"passed": True, "skipped": False, "file": "x.xml", "props": {}, "entry_hits": {}, "integration": None}
    assert any("not bound to a commit" in p for p in problems(vc.check(m, {key: unbound}, root=repo,
                                                                      global_checks=False), "A01"))
    # recorded while the test file had uncommitted changes
    assert any("uncommitted changes" in p for p in
               problems(vc.check(m, {key: record(c0, dirty=[TEST_MOD])}, root=repo, global_checks=False), "A01"))
    # an unrelated later commit keeps the record valid; a change to the source file invalidates it
    write(repo, "superlearner/docs/other.md", "x\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "unrelated")
    assert problems(vc.check(m, {key: record(c0)}, root=repo, global_checks=False), "A01") == []
    write(repo, SRC_MOD, "def run(x):\n    return 3 * x\n")
    ps = problems(vc.check(m, {key: record(c0)}, root=repo, global_checks=False), "A01")
    assert any("modified in the working tree" in p for p in ps)
    git(repo, "commit", "-qam", "change the feature")
    ps = problems(vc.check(m, {key: record(c0)}, root=repo, global_checks=False), "A01")
    assert any("changed between the recorded commit" in p for p in ps)
    # a record from a commit that is not in HEAD's history (another branch / a rewritten history)
    git(repo, "checkout", "-q", "-b", "side", c0)
    write(repo, "superlearner/docs/side.md", "y\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "side")
    side = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-")
    ps = problems(vc.check(m, {key: record(side)}, root=repo, global_checks=False), "A01")
    assert any("not an ancestor of HEAD" in p for p in ps)


def _verification(repo, rid, workstream="H", commit=None):
    rel = f"superlearner/docs/evidence/verification/{rid}.json"
    write(repo, rel, json.dumps({"requirement": rid, "verifier": "independent session", "workstream": workstream,
                                 "commit": commit or git(repo, "rev-parse", "HEAD"),
                                 "commands": ["pytest ..."], "measured": {"x": 1}}))
    return rel


def test_integration_claims_need_clean_records_that_executed_the_entry_and_an_independent_record(repo):
    entry = f"{SRC_MOD}:run"
    acc, integ = f"{TEST_MOD}::test_feature_doubles", f"{INTEG_MOD}::test_feature_end_to_end"
    ver = _verification(repo, "A02")
    write(repo, "superlearner/docs/evidence/a02.json", "{}")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "verification")
    head = git(repo, "rev-parse", "HEAD")
    base = {"id": "A02", "status": "INTEGRATION_TESTED", "files": [SRC_MOD], "acceptance_tests": [acc],
            "integration_tests": [integ], "integration_entry": entry, "verified_by": ver,
            "evidence": ["superlearner/docs/evidence/a02.json#numbers", "sha256:" + "a" * 64, acc]}
    good = {tuple(acc.split("::")): record(head),
            tuple(integ.split("::")): record(head, integration="clean", hits={entry: 3})}
    rep = vc.check(manifest(base), good, root=repo, global_checks=False)
    assert problems(rep, "A02") == [] and rep["engineering_complete"] == ["A02"]
    cases = {
        "test doubles loaded": record(head, integration="superlearner.cam_unified.integration_doubles", hits={entry: 3}),
        "did not execute": record(head, integration="clean", hits={entry: 0}),
        "not run as an integration test": record(head),
    }
    for needle, rec in cases.items():
        rep = vc.check(manifest(base), {**good, tuple(integ.split("::")): rec}, root=repo, global_checks=False)
        assert any(needle in p for p in problems(rep, "A02")), (needle, problems(rep, "A02"))
    for bad_ver, needle in (("H", "missing"), (_verification(repo, "A02", workstream="A"), "not independent")):
        if bad_ver == "H":
            r = dict(base, verified_by="superlearner/docs/evidence/verification/NOPE.json")
        else:
            r = dict(base, verified_by=bad_ver)
        rep = vc.check(manifest(r), good, root=repo, global_checks=False)
        assert any(needle in p for p in problems(rep, "A02")), problems(rep, "A02")
    rep = vc.check(manifest(dict(base, evidence=["it works on my machine"])), good, root=repo, global_checks=False)
    assert any("evidence entry is not a resolvable" in p for p in problems(rep, "A02"))


def test_run_tests_separates_integration_tests_and_their_entries(repo):
    acc, integ = f"{TEST_MOD}::test_feature_doubles", f"{INTEG_MOD}::test_feature_end_to_end"
    m = manifest({"id": "A01", "status": "IMPLEMENTED", "files": [SRC_MOD], "acceptance_tests": [acc]},
                 {"id": "A02", "status": "INTEGRATION_TESTED", "files": [SRC_MOD], "acceptance_tests": [acc],
                  "integration_tests": [integ], "integration_entry": f"{SRC_MOD}:run"},
                 {"id": "A03", "status": "IN_PROGRESS", "files": [SRC_MOD],
                  "acceptance_tests": [f"{INTEG_MOD}::test_feature_end_to_end"]})
    unit, integ_map = vc.referenced_tests(m, root=repo)
    assert unit == [acc] and integ_map == {integ: [f"{SRC_MOD}:run"]}


# ------------------------------------------------------------------------------------------ experiment results
def _confirm_rows(t0, learners=("cam_full", "frozen"), seeds=(0, 1, 2), dup=None):
    rows = []
    for L in learners:
        for s in seeds:
            for w in range(4):
                rows.append({"set": "confirm", "learner": L, "seed": s, "run_id": f"{L}-{s}", "world": w,
                             "t_wall": t0 + 10 * s + w, "value": 0.5})
    if dup:
        rows.append({"set": "confirm", "learner": dup[0], "seed": dup[1], "run_id": "again", "world": 0,
                     "t_wall": t0 + 999, "value": 0.9})
    return rows


def _result(repo, prereg_commit, rows, verdict="DEMONSTRATED", lo=0.1, hi=0.4, seeds=(0, 1, 2), learner=None,
            tamper=False):
    raw = write(repo, "superlearner/docs/evidence/dom/W9/raw.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    sha = hashlib.sha256(open(os.path.join(repo, raw), "rb").read()).hexdigest()
    if tamper:
        sha = "0" * 64
    L = learner or {"tag": "cam_full", "kind": "cam", "ckpt_sha256": "c" * 64, "cfg_hash": "h", "scale": "dom_dev",
                    "trainable_params": 51_799_312, "semantic_model": "Qwen/Qwen3-0.6B",
                    "reward_mode": "observation_only"}
    d = {"experiment_id": "U-T-1", "prereg_file": "superlearner/docs/experiments/U-T-1.md",
         "prereg_commit": prereg_commit, "set": "confirm", "code_commit": git(repo, "rev-parse", "HEAD"),
         "learners": [L], "seeds": list(seeds), "n_worlds": 4,
         "primary": {"name": "U", "direction": ">", "estimate": (lo + hi) / 2, "lo": lo, "hi": hi, "method": "t"},
         "raw_files": [{"path": raw, "sha256": sha}], "compute": {"cam_full": {"decisions": 100}}, "verdict": verdict}
    rel = write(repo, "superlearner/docs/evidence/dom/W9/result.json", json.dumps(d))
    return rel


@pytest.fixture
def prereg(repo):
    write(repo, "superlearner/docs/experiments/U-T-1.md", "# U-T-1 pre-registration\nprimary: U > 0\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "prereg U-T-1", date=1_700_000_100)
    vc.clear_caches()
    return git(repo, "rev-parse", "HEAD")


def test_a_valid_confirm_result_passes_the_protocol_checks(repo, prereg):
    rel = _result(repo, prereg, _confirm_rows(1_700_001_000))
    assert vc.check_experiment_result(rel, "EMPIRICALLY_DEMONSTRATED", repo) == []


@pytest.mark.parametrize("case,needle", [
    ("verdict_mismatch", "disagrees with the primary interval"),
    ("prereg_after_confirm", "after the first confirm row"),
    ("double_evaluation", "evaluated on the confirm set 2 times"),
    ("hash", "does not match its recorded sha256"),
    ("seeds", "needs >= 3 training seeds"),
    ("reward_mode", "not observation_only"),
    ("dom_scale", "claims scale 'dom'"),
    ("dev_set", "empirical claims rest on the confirm set"),
])
def test_protocol_violations_are_rejected(repo, prereg, case, needle):
    t0 = 1_700_001_000
    kw = {}
    rows = _confirm_rows(t0)
    if case == "verdict_mismatch":
        kw = dict(lo=-0.1, hi=0.4)                         # straddles 0 -> INCONCLUSIVE, recorded DEMONSTRATED
    elif case == "prereg_after_confirm":
        rows = _confirm_rows(1_699_999_000)
    elif case == "double_evaluation":
        rows = _confirm_rows(t0, dup=("cam_full", 1))
    elif case == "hash":
        kw = dict(tamper=True)
    elif case == "seeds":
        kw = dict(seeds=(0,))
    elif case == "reward_mode":
        kw = dict(learner={"tag": "cam_full", "kind": "cam", "ckpt_sha256": "c", "cfg_hash": "h", "scale": "dom_dev",
                           "trainable_params": 1, "semantic_model": "q", "reward_mode": "legacy"})
    elif case == "dom_scale":
        kw = dict(learner={"tag": "cam_1b", "kind": "cam", "ckpt_sha256": "c", "cfg_hash": "h", "scale": "dom",
                           "trainable_params": 1, "semantic_model": "Qwen/Qwen3-0.6B", "reward_mode": "observation_only",
                           "cfg": {"n_layers": 8, "d_model": 384}, "trainable_core_params": 14_503_170})
    rel = _result(repo, prereg, rows, **kw)
    if case == "dev_set":
        d = json.load(open(os.path.join(repo, rel)))
        d["set"] = "dev"
        write(repo, rel, json.dumps(d))
    ps = vc.check_experiment_result(rel, "EMPIRICALLY_DEMONSTRATED", repo)
    assert any(needle in p for p in ps), (case, ps)


def test_a_preregistration_edited_after_the_confirm_run_started_is_flagged(repo, prereg):
    rel = _result(repo, prereg, _confirm_rows(1_700_001_000))
    write(repo, "superlearner/docs/experiments/U-T-1.md", "# U-T-1\nprimary: U > 0.2 (changed)\n")
    git(repo, "commit", "-qam", "edit prereg", date=1_700_002_000)
    ps = vc.check_experiment_result(rel, "EMPIRICALLY_DEMONSTRATED", repo)
    assert any("modified after the confirm evaluation started" in p for p in ps)


def test_failed_and_inconclusive_results_support_only_a_failed_claim(repo, prereg):
    rel = _result(repo, prereg, _confirm_rows(1_700_001_000), verdict="INCONCLUSIVE", lo=-0.1, hi=0.3)
    assert vc.check_experiment_result(rel, "FAILED_EMPIRICALLY", repo) == []
    assert any("does not support" in p for p in vc.check_experiment_result(rel, "EMPIRICALLY_DEMONSTRATED", repo))
    rel = _result(repo, prereg, _confirm_rows(1_700_001_000), verdict="FAILED", lo=-0.4, hi=-0.1, seeds=(0,))
    assert vc.check_experiment_result(rel, "FAILED_EMPIRICALLY", repo) == []


# ------------------------------------------------------------------------------------------ global checks
REG_HEAD = "| ID | Experiment | Status | Declared in commit | Result |\n|---|---|---|---|---|\n"


def test_register_rows_must_match_the_preregistrations(repo, prereg):
    write(repo, vc.REGISTER, REG_HEAD + f"| U-T-1 | test | declared | {prereg[:10]} | |\n")
    assert vc.check_register(repo) == []
    write(repo, vc.REGISTER, REG_HEAD + "| U-T-1 | test | not declared | | |\n")
    assert any("its pre-registration U-T-1.md exists" in p for p in vc.check_register(repo))
    write(repo, vc.REGISTER, REG_HEAD + f"| U-T-1 | test | declared | {prereg[:10]} | |\n"
                                        "| U-T-2 | other | declared | abcdef1 | |\n")
    assert any("U-T-2" in p and "does not exist" in p for p in vc.check_register(repo))
    first = git(repo, "rev-list", "--max-parents=0", "HEAD")
    write(repo, vc.REGISTER, REG_HEAD + f"| U-T-1 | test | declared | {first[:10]} | |\n")
    assert any("does not exist at its declaring commit" in p for p in vc.check_register(repo))
    write(repo, vc.REGISTER, REG_HEAD)
    assert any("has no row" in p for p in vc.check_register(repo))


def test_static_reward_integrity_scan_flags_actor_side_label_imports(repo):
    assert vc.check_reward_integrity(repo) == []
    write(repo, "superlearner/src/superlearner/cam_unified/core/leaky.py",
          "from ..envs.labels import LabelStore\n")
    write(repo, "superlearner/src/superlearner/cam_unified/agent.py", "def act(op):\n    return op.private_verify(0)\n")
    ps = vc.check_reward_integrity(repo)
    assert any("core/leaky.py" in p for p in ps) and any("agent.py" in p for p in ps)


def test_the_real_tree_has_no_actor_side_label_imports():
    assert vc.check_reward_integrity(vc.ROOT) == []


# ------------------------------------------------------------------------------------------ the pytest plugin
PLUGIN_CLEAN = '''
import superlearner.cam_unified.experiments.stats as ST

def test_clean_integration_runs_the_entry():
    assert ST.two_level([[1.0, 2.0], [2.0, 3.0]])["mean"] == 2.0

def test_unit_level():
    assert True
'''
PLUGIN_DIRTY = '''
def test_dirty_integration_loads_a_double():
    import superlearner.cam_unified.integration_doubles  # noqa: F401
    assert True
'''


def test_plugin_records_commit_binding_rejects_doubles_and_counts_entry_calls(tmp_path):
    d = tmp_path / "plug"
    d.mkdir()
    (d / "test_plug.py").write_text(textwrap.dedent(PLUGIN_CLEAN))
    (d / "test_plug_dirty.py").write_text(textwrap.dedent(PLUGIN_DIRTY))
    entry = "superlearner/src/superlearner/cam_unified/experiments/stats.py:two_level"
    (d / "integ.txt").write_text(f"test_plug.py::test_clean_integration_runs_the_entry\t{entry}\n"
                                 f"test_plug_dirty.py::test_dirty_integration_loads_a_double\t{entry}\n")
    out = d / "out.xml"
    env = dict(os.environ, PYTHONPATH=SRC, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p",
                        "superlearner.verify_completion", f"--vc-integration={d / 'integ.txt'}",
                        f"--junitxml={out}", "-o", "junit_family=xunit1", "test_plug.py", "test_plug_dirty.py"],
                       cwd=str(d), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr[-2000:]       # the dirty integration test fails
    cases = {}
    for tc in ET.parse(out).iter("testcase"):
        props = {p.get("name"): p.get("value") for p in tc.iter("property")}
        cases[tc.get("name")] = (props, any(c.tag in ("failure", "error") for c in tc))
    head = vc.git_head(vc.ROOT)
    for name, (props, _) in cases.items():
        assert props["vc_commit"] == head and isinstance(json.loads(props["vc_dirty"]), list)
        assert len(props["vc_test_sha256"]) == 64
    clean, failed = cases["test_clean_integration_runs_the_entry"]
    assert not failed and clean["vc_integration"] == "clean" and json.loads(clean["vc_entry_hits"])[entry] >= 1
    dirty, failed = cases["test_dirty_integration_loads_a_double"]
    assert failed and "integration_doubles" in dirty["vc_integration"]
    assert json.loads(dirty["vc_entry_hits"])[entry] == 0
    unit, failed = cases["test_unit_level"]
    assert not failed and "vc_integration" not in unit
