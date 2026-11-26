"""Independent adversarial verification of the acceptance checker (workstream P, wave 3).

The wave-1 suite (tests/test_v5_acceptance_checker.py) covers the recovery audit's probes P1-P14. The probes here are
new: each one is an attack a cheating producer could mount with the official pipeline itself or with a little file
editing - environment injection (the P skeptic's S1 attack and its variants), measurements that do not depend on the
implementation, patches hidden in fixtures, conftest hooks that rewrite outcomes, tamper code that hides itself from
the module list and is reverted after the run, stale data files and datasets, hidden working-tree changes, ledger and
blocker abuse, self-review without session identities, fully consistent forgeries, and weakening through manifest
history that predates the ledger. Every probe runs for real in a throwaway git repository with the real acceptance
package (`acceptance_kit.Repo`): official runs in fresh clones, reviews that re-execute and ablate, the checker.

A passing probe test is a rejected attack. Each asserts that the attack does not reach VERIFIED (or, for the stated
residual - a forger who rewrites every file consistently - that it cannot reach a terminal project status without the
release reproduction, which then fails it)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap

import pytest

from acceptance_kit import CRIT, GOOD_TEST, WIDGETS, Repo, git
from crucible.acceptance.v5 import checker, evidence, ledger as L, manifest as M, testscan

pytestmark = pytest.mark.timeout(600)

DEFECT = WIDGETS.replace("def leaks():\n    return 0", "def leaks():\n    return 1")
RAISING = WIDGETS.replace("def leaks():\n    return 0", "def leaks():\n    raise RuntimeError('leak detector broken')")

# a meta-path hook that patches `widgets.leaks` to return 0 right after the module is executed (the S1 injection)
PATCHER = textwrap.dedent('''
    import importlib.machinery
    import sys


    class _Patch:
        def find_spec(self, name, path=None, target=None):
            if name != "widgets":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            run = spec.loader.exec_module

            def exec_module(mod):
                run(mod)
                mod.leaks = lambda: 0
            spec.loader.exec_module = exec_module
            return spec


    sys.meta_path.insert(0, _Patch())
''')


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return Repo(str(tmp_path), monkeypatch)


def attempt(r: Repo, *, agent: str = "builder", reviewer: str = "qa") -> tuple[dict, dict]:
    """A producer's full attempt: official run, commit, review by another name and session, commit, check."""
    r.official_run(agent=agent)
    r.commit("evidence")
    r.review(agent=reviewer)
    r.commit("review")
    res, rows = r.verify()
    return res, rows["Z.1"]


def rejected(r: Repo, rid: str = "Z.1", **kw) -> tuple[dict, dict]:
    res, rows = r.verify(**kw)
    assert rows[rid]["status"] != "VERIFIED", rows[rid]
    assert not res["ok"] and res["project_status"] != "CRUCIBLE_5_COMPLETE"
    return res, rows[rid]


def has(row: dict, *words: str) -> bool:
    return any(all(w in p for w in words) for p in row["problems"])


def user_site(base: str) -> str:
    return sysconfig.get_path("purelib", f"{os.name}_user", vars={"userbase": base})


# ------------------------------------------------------------------------------------------------- environment
def test_P15_S1_pythonpath_sitecustomize_injection_is_inert(tmp_path, monkeypatch):
    """S1 exactly: a sitecustomize outside the repository on PYTHONPATH repairs a committed defect at start-up."""
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT)
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "sitecustomize.py").write_text(PATCHER)
    monkeypatch.setenv("PYTHONPATH", f"{evil}{os.pathsep}{os.path.join(r.root, 'src')}")
    out = r.official_run()
    assert out["pytest_rc"] != 0                       # the injection never reached the measured process
    r.commit("evidence")
    r.review()
    r.commit("review")
    _, row = rejected(r)
    assert has(row, "criterion", "leaks")


def test_P15b_user_site_usercustomize_injection_is_inert(tmp_path, monkeypatch):
    """PYTHONUSERBASE points at a user site with a usercustomize (user site-packages are enabled in /venv/main)."""
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT)
    base = tmp_path / "userbase"
    site_dir = user_site(str(base))
    os.makedirs(site_dir)
    open(os.path.join(site_dir, "usercustomize.py"), "w").write(PATCHER)
    open(os.path.join(site_dir, "zz_evil.pth"), "w").write("import usercustomize\n")
    monkeypatch.setenv("PYTHONUSERBASE", str(base))
    out = r.official_run()
    assert out["pytest_rc"] != 0
    r.commit("evidence")
    r.review()
    r.commit("review")
    rejected(r)


def test_P15c_a_poisoned_interpreter_is_detected_in_the_records(tmp_path, monkeypatch):
    """The producer and a colluding reviewer both run with an interpreter whose site directory carries a .pth hook
    that patches the defect away. The records must expose it to the checker, which runs with the honest one."""
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT)
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    sp = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], capture_output=True,
                        text=True, check=True).stdout.strip()
    host_sp = sysconfig.get_path("purelib")
    open(os.path.join(sp, "aa_host.pth"), "w").write(host_sp + "\n")
    open(os.path.join(sp, "evilhook.py"), "w").write(PATCHER)
    open(os.path.join(sp, "zz_evil.pth"), "w").write("import evilhook\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(r.root, "src"))

    def cli(*args, session):
        return subprocess.run([py, "-m", "crucible.acceptance.v5", *args], cwd=r.root, capture_output=True, text=True,
                              env=dict(env, CRUCIBLE_V5_SESSION=session, CRUCIBLE_V5_SCRATCH=r.scratch))
    p = cli("run", "--requirement", "Z.1", "--agent", "builder", session="s-b")
    assert "pytest exit 0" in p.stdout, p.stdout + p.stderr            # the poisoned run itself passes
    r.commit("evidence")
    p = cli("review", "--requirement", "Z.1", "--agent", "qa", "--verdict", "approved", session="s-q")
    r.commit("review")
    _, row = rejected(r)
    assert has(row, "startup hook") or has(row, "foreign"), row["problems"]


def test_P15d_a_function_patched_from_outside_the_tree_is_detected(tmp_path, monkeypatch):
    """Whatever the injection vector, a crucible or implementation function whose code lives in a file outside the
    checkout and outside the interpreter is a patch, and the record says so (unit check of the detector)."""
    r = Repo(str(tmp_path), monkeypatch)
    patch_dir = tmp_path / "patchsrc"
    patch_dir.mkdir()
    (patch_dir / "patchmod.py").write_text("def zero():\n    return 0\n")
    script = textwrap.dedent(f'''
        import sys, json
        sys.path.insert(0, {str(patch_dir)!r})
        import widgets, patchmod
        widgets.leaks = patchmod.zero
        from crucible.acceptance.v5 import evidence
        print(json.dumps(evidence.patched_functions({r.root!r})))
    ''')
    out = subprocess.run([sys.executable, "-c", script], cwd=r.root, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(r.root, "src")))
    found = json.loads(out.stdout.strip().splitlines()[-1])
    assert any("widgets.leaks" in f and "patchmod.py" in f for f in found), (found, out.stderr[-500:])


# ------------------------------------------------------------------------------------------------- measurement
HELPER_CONST = '''from crucible.acceptance.v5 import evidence
import widgets


def _measure():
    return {"widgets": 12, "leaks": 0}


def test_z():
    m = _measure()
    evidence.record("Z.1", m)
    assert m["widgets"] >= 10 and m["leaks"] == 0
'''

MODULE_CONST = '''from crucible.acceptance.v5 import evidence
import widgets

METRICS = {"widgets": 12, "leaks": 0}


def test_z():
    evidence.record("Z.1", METRICS)
    assert METRICS["widgets"] >= 10 and widgets.__name__
'''

DICT_CALL = '''from crucible.acceptance.v5 import evidence
import widgets


def test_z():
    evidence.record("Z.1", dict(widgets=12, leaks=0))
    assert len(widgets.__name__) > 0
'''


# the implementation is called first, so the ablated implementation raises before anything is recorded and the
# ablation alone would count the measurement as dependent: only the static detector can see the laundering
CALL_FIRST_CONST = HELPER_CONST.replace("    m = _measure()\n", "    widgets.count()\n    m = _measure()\n")
BUILTIN_CONST = DICT_CALL.replace("dict(widgets=12, leaks=0)", '{"widgets": int("12"), "leaks": len([])}')


@pytest.mark.parametrize("src", [HELPER_CONST, MODULE_CONST, DICT_CALL, CALL_FIRST_CONST, BUILTIN_CONST],
                         ids=["helper", "module_dict", "dict_call", "call_first", "builtins"])
def test_P16_constant_metrics_that_evade_the_static_scan_are_caught_by_ablation(tmp_path, monkeypatch, src):
    """The implementation is defective; the metrics are constants laundered through a helper, a module-level dict, a
    dict() call or pure builtins. The review's ablation run (every implementation function raises) reproduces them
    when the implementation is never called; when it is called first (call_first), the static detector names the
    constant record."""
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT, test_src=src)
    _, row = attempt(r)
    assert row["status"] != "VERIFIED"
    assert has(row, "ablation") or has(row, "constant_record"), row["problems"]


def test_P16d_an_honest_measurement_survives_ablation(repo):
    repo.honest()
    rv = repo.json("docs/v5/reviews/Z.1.json")
    assert rv["ablation"]["sensitive"] is True and rv["ablation"]["ablated_files"] == ["src/widgets/__init__.py"]
    assert repo.verify()[1]["Z.1"]["status"] == "VERIFIED"


AUTOUSE_PATCH = '''import pytest

from crucible.acceptance.v5 import evidence
import widgets


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(widgets, "leaks", lambda: 0)


def test_z():
    n, leaks = widgets.count(), widgets.leaks()
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''

ASSIGN_PATCH = '''from crucible.acceptance.v5 import evidence
import widgets


def test_z():
    widgets.leaks = lambda: 0
    n, leaks = widgets.count(), widgets.leaks()
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''

SETATTR_HELPER = '''from crucible.acceptance.v5 import evidence
import widgets


def _prepare():
    setattr(widgets, "leaks", lambda: 0)


def test_z():
    _prepare()
    n, leaks = widgets.count(), widgets.leaks()
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''


@pytest.mark.parametrize("src", [AUTOUSE_PATCH, ASSIGN_PATCH, SETATTR_HELPER], ids=["autouse", "assign", "setattr"])
def test_P17_patching_the_implementation_outside_the_test_body_is_rejected(tmp_path, monkeypatch, src):
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT, test_src=src)
    _, row = attempt(r)
    assert row["status"] != "VERIFIED" and has(row, "patch"), row["problems"]


def test_P17d_a_conftest_fixture_patch_is_rejected(tmp_path, monkeypatch):
    conftest = ('import pytest\nimport widgets\n\n\n@pytest.fixture(autouse=True)\ndef _fix(monkeypatch):\n'
                '    monkeypatch.setattr(widgets, "leaks", lambda: 0)\n')
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT, extra_files={"tests/v5/conftest.py": conftest})
    _, row = attempt(r)
    assert row["status"] != "VERIFIED" and has(row, "conftest"), row["problems"]


AUDITED_TEST = GOOD_TEST + "    assert widgets.audit() == 'ok'\n"
AUDITED_WIDGETS = WIDGETS + "\n\ndef audit():\n    return 'broken'\n"
FLIP_CONFTEST = '''import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    rep = out.get_result()
    if rep.failed:
        rep.outcome = "passed"
'''
SKIP_BODY_CONFTEST = '''def pytest_pyfunc_call(pyfuncitem):
    from crucible.acceptance.v5 import evidence
    evidence.record("Z.1", {"widgets": 12, "leaks": 0})
    return True
'''


@pytest.mark.parametrize("conftest", [FLIP_CONFTEST, SKIP_BODY_CONFTEST], ids=["makereport", "pyfunc_call"])
def test_P18_a_conftest_hook_that_rewrites_outcomes_is_rejected(tmp_path, monkeypatch, conftest):
    """The JUnit and the exit code are produced by pytest, so a committed hook can make a failing test 'pass' in the
    evidence run and in every reproduction alike."""
    r = Repo(str(tmp_path), monkeypatch, widgets=AUDITED_WIDGETS, test_src=AUDITED_TEST,
             extra_files={"tests/v5/conftest.py": conftest})
    out = r.official_run()
    assert out["pytest_rc"] == 0                        # the hook really does hide the failure from pytest
    r.commit("evidence")
    r.review()
    r.commit("review")
    _, row = rejected(r)
    assert has(row, "conftest"), row["problems"]


SUPPRESS_TEST = '''import contextlib

from crucible.acceptance.v5 import evidence
import widgets


def test_z():
    n = widgets.count()
    leaks = 0
    with contextlib.suppress(Exception):
        leaks = widgets.leaks()
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''
FALLBACK_TEST = '''from crucible.acceptance.v5 import evidence
import widgets


def test_z():
    n = widgets.count()
    try:
        leaks = widgets.leaks()
    except Exception:
        leaks = 0
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''


@pytest.mark.parametrize("src", [SUPPRESS_TEST, FALLBACK_TEST], ids=["suppress", "fallback"])
def test_P19_swallowing_a_broken_implementation_is_rejected(tmp_path, monkeypatch, src):
    r = Repo(str(tmp_path), monkeypatch, widgets=RAISING, test_src=src)
    _, row = attempt(r)
    assert row["status"] != "VERIFIED" and has(row, "swallow"), row["problems"]


# ------------------------------------------------------------------------------------------------- staleness
DATA_WIDGETS = ('import json\nimport os\n\n\ndef count():\n'
                '    return json.load(open(os.path.join(os.path.dirname(__file__), "data.json")))["count"]\n\n\n'
                'def leaks():\n    return 0\n')


def test_P20_a_changed_data_file_makes_evidence_stale(tmp_path, monkeypatch):
    r = Repo(str(tmp_path), monkeypatch, widgets=DATA_WIDGETS, extra_files={"src/widgets/data.json": '{"count": 12}'})
    r.honest()
    assert r.verify()[1]["Z.1"]["status"] == "VERIFIED"
    r.write("src/widgets/data.json", '{"count": 3}')
    r.commit("the data the implementation reads changed")
    _, row = rejected(r)
    assert has(row, "stale", "src/widgets/data.json"), row["problems"]


def test_P20b_the_release_check_reproduces_at_head_not_at_the_evidence_commit(tmp_path, monkeypatch):
    """Even when a change escapes the static staleness check (here: a file only opened through a subprocess), the
    release reproduction runs the tree being released, so it cannot certify stale evidence."""
    w = ('import json\nimport os\nimport subprocess\nimport sys\n\n\ndef count():\n'
         '    p = os.path.join(os.path.dirname(__file__), "data.json")\n'
         '    out = subprocess.run([sys.executable, "-c", "import json,sys; print(json.load(open(sys.argv[1]))[\'count\'])",'
         ' p], capture_output=True, text=True, check=True).stdout\n'
         '    return int(out)\n\n\ndef leaks():\n    return 0\n')
    r = Repo(str(tmp_path), monkeypatch, widgets=w, extra_files={"src/widgets/data.json": '{"count": 12}'})
    r.honest()
    r.write("src/widgets/data.json", '{"count": 3}')
    r.commit("data changed behind a subprocess")
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = r.verify(reproduce=True, agent="release-check", scratch=r.scratch)
    assert rows["Z.1"]["status"] == "FAILED" and res["project_status"] != "CRUCIBLE_5_COMPLETE", rows["Z.1"]
    assert rows["Z.1"]["reproduction"]["commit"] == git(r.top, "rev-parse", "HEAD")


SELF_HIDING = PATCHER + textwrap.dedent('''

    class _Hide:
        def find_spec(self, name, path=None, target=None):
            if name != "crucible.acceptance.v5.evidence":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            run = spec.loader.exec_module

            def exec_module(mod):
                run(mod)
                real = mod.imported_modules

                def hidden(root):
                    inside, outside = real(root)
                    return {k: v for k, v in inside.items() if "customize" not in k and "conftest" not in k}, outside
                mod.imported_modules = hidden
            spec.loader.exec_module = exec_module
            return spec


    sys.meta_path.insert(0, _Hide())
''')


@pytest.mark.parametrize("where", ["src/sitecustomize.py", "tests/v5/conftest.py"])
def test_P21_tamper_code_that_hides_itself_and_is_reverted_is_detected(tmp_path, monkeypatch, where):
    """Commit C carries a defect plus tamper code (a sitecustomize on the clone's PYTHONPATH, or a conftest) that
    repairs it at run time and removes itself from the record's module list; the evidence and the review run at C;
    the tamper code is then deleted. HEAD is defective."""
    tamper = SELF_HIDING if where.endswith("sitecustomize.py") else \
        "import sys\nsys.path.insert(0, __import__('os').path.dirname(__file__))\n" + SELF_HIDING
    r = Repo(str(tmp_path), monkeypatch, widgets=DEFECT, extra_files={where: tamper})
    out = r.official_run()
    assert out["pytest_rc"] == 0
    r.commit("evidence")
    r.review()
    r.commit("review")
    git(r.top, "rm", "-q", os.path.join("crucible", where))
    r.commit("remove the tamper code")
    rejected(r)
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    rejected(r, reproduce=True, agent="release-check", scratch=r.scratch)


# ------------------------------------------------------------------------------------------------- working tree
@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_P22_index_flags_that_hide_working_tree_changes_block_completion(repo, flag):
    repo.honest()
    git(repo.top, "update-index", flag, "crucible/src/widgets/__init__.py")
    repo.write("src/widgets/__init__.py", DEFECT)
    assert git(repo.top, "status", "--porcelain") == ""          # git status no longer shows the change
    res, _ = repo.verify()
    assert res["project_status"] != "CRUCIBLE_5_COMPLETE"
    assert any("hide" in p for p in res["global_problems"]), res["global_problems"]


def test_P22b_an_ignored_startup_hook_in_the_tree_blocks_completion(repo):
    repo.honest()
    with open(os.path.join(repo.top, ".git", "info", "exclude"), "a") as fh:
        fh.write("sitecustomize.py\n")
    repo.write("src/sitecustomize.py", PATCHER)
    assert git(repo.top, "status", "--porcelain") == ""
    res, _ = repo.verify()
    assert res["project_status"] != "CRUCIBLE_5_COMPLETE"
    assert any("sitecustomize" in p for p in res["global_problems"]), res["global_problems"]


# ------------------------------------------------------------------------------------------------- forgery
def _restamp(r: Repo, old: str, new: str) -> None:
    """Re-stamp every genuine record made at commit `old` onto commit `new` and fix every hash: the residual case of
    a forger with write access who rewrites all files consistently."""
    blob = evidence.git_blob_id(open(os.path.join(r.root, "src/widgets/__init__.py"), "rb").read())
    recs = []
    for d in ("docs/v5/evidence", "docs/v5/reviews"):
        for dp, _, fs in os.walk(os.path.join(r.root, d)):
            recs += [os.path.relpath(os.path.join(dp, f), r.root) for f in fs if f.endswith(".json")]
    for rel in recs:
        obj = json.loads(open(os.path.join(r.root, rel)).read().replace(old, new))
        if "src/widgets/__init__.py" in (obj.get("modules") or {}):
            obj["modules"]["src/widgets/__init__.py"] = blob
        r.dump(rel, obj)
    for rel in recs:                                  # run records hold the hashes of the records they produced
        obj = r.json(rel)
        if obj.get("schema", "").startswith("crucible.v5.run/"):
            base = os.path.dirname(rel)
            for rid in obj.get("evidence", {}):
                ev_rel = evidence.evidence_relpath(rid) if "official_runs" in rel else \
                    os.path.join(base, obj["run_id"], f"{rid}.json")
                obj["evidence"][rid] = evidence.sha256_file(os.path.join(r.root, ev_rel))
            r.dump(rel, obj)
    rv = r.json("docs/v5/reviews/Z.1.json")
    rv["evidence_sha256"] = evidence.sha256_file(os.path.join(r.root, evidence.evidence_relpath("Z.1")))
    r.dump("docs/v5/reviews/Z.1.json", rv)


def test_P23_a_fully_consistent_forgery_cannot_complete_without_the_release_reproduction(repo, monkeypatch):
    repo.honest()
    good = repo.json(evidence.evidence_relpath("Z.1"))["tree"]["commit"]
    repo.write("src/widgets/__init__.py", DEFECT)
    bad = repo.commit("a defect")
    _restamp(repo, good, bad)
    repo.commit("re-stamped records")
    res, rows = repo.verify()
    assert res["project_status"] != "CRUCIBLE_5_COMPLETE"            # static checks alone never complete
    assert any("--reproduce" in p for p in res["global_problems"]), res["global_problems"]
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = repo.verify(reproduce=True, agent="release-check", scratch=repo.scratch)
    assert rows["Z.1"]["status"] == "FAILED" and res["project_status"] != "CRUCIBLE_5_COMPLETE"


def test_P23b_the_honest_project_completes_only_with_the_release_reproduction(repo, monkeypatch):
    repo.honest()
    res, rows = repo.verify()
    assert rows["Z.1"]["status"] == "VERIFIED" and res["project_status"] == "IN_PROGRESS"
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = repo.verify(reproduce=True, agent="release-check", scratch=repo.scratch)
    assert res["project_status"] == "CRUCIBLE_5_COMPLETE", (res["global_problems"], rows["Z.1"]["problems"])


def test_P24_self_review_without_session_identities_is_rejected(repo, monkeypatch):
    """Two agent names from one shell with no session identity at all: the runner refuses, and a review whose
    session identity was blanked by hand does not count."""
    from crucible.acceptance.v5 import official
    for k in ("CRUCIBLE_V5_SESSION", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="session"):
        official.run(repo.root, requirement="Z.1", agent="builder", scratch=repo.scratch)
    repo.official_run(agent="builder", session="s-b")
    repo.commit("evidence")
    repo.review(agent="qa", session="s-q")
    rv = repo.json("docs/v5/reviews/Z.1.json")
    rv["reviewer"]["session_sha256"] = ""
    repo.dump("docs/v5/reviews/Z.1.json", rv)
    repo.commit("review")
    _, row = rejected(repo)
    assert has(row, "session"), row["problems"]


# ------------------------------------------------------------------------------------------------- ledger, blockers
def test_P25_re_anchoring_the_specification_in_the_ledger_is_detected(tmp_path, monkeypatch):
    r = Repo(str(tmp_path), monkeypatch, extra_files={"docs/v5/MASTER_SPEC.md": "# spec\nThe verbatim text.\n"})
    assert L.state(L.read(r.root))["__anchors__"].get("MASTER_SPEC.md") == M.spec_sha256(r.root)
    r.honest()
    r.write("docs/v5/MASTER_SPEC.md", "# spec\nAn edited text.\n")
    L.append(r.root, [{"id": "MASTER_SPEC.md", "kind": "anchor", "value": M.spec_sha256(r.root), "note": "edited"}])
    r.commit("edit and re-anchor the specification")
    res, _ = r.verify()
    assert any("anchor" in p for p in res["global_problems"]), res["global_problems"]


def test_P26_a_blocker_that_does_not_name_the_requirement_does_not_block_it(tmp_path, monkeypatch):
    items = [("Z.1", "Z", "make widgets", "A", [], CRIT, 3),
             ("Z.2", "Z", "a hard one", "A", [], [{"metric": "hard", "op": ">=", "value": 1}], 3)]
    r = Repo(str(tmp_path), monkeypatch, items=items)
    r.honest()
    r.write("docs/v5/BLOCKERS.md", "# blockers\n\n## Open blockers\n\n### BLK-7 general difficulty\nIt is hard.\n")
    m = r.manifest()
    next(x for x in m["requirements"] if x["id"] == "Z.2").update(status="BLOCKED", blocker="BLK-7")
    r.save_manifest(m)
    r.commit("blocked by a generic blocker")
    monkeypatch.setenv("CRUCIBLE_V5_SESSION", "s-release")
    res, rows = r.verify(reproduce=True, agent="release-check", scratch=r.scratch)
    assert rows["Z.2"]["status"] != "BLOCKED" and res["project_status"] == "IN_PROGRESS", rows["Z.2"]
    r.write("docs/v5/BLOCKERS.md", "# blockers\n\n## Open blockers\n\n### BLK-7 the mine is closed\nBlocks Z.2.\n")
    r.commit("the blocker names what it blocks")
    res, rows = r.verify(reproduce=True, agent="release-check", scratch=r.scratch)
    assert rows["Z.2"]["status"] == "BLOCKED" and res["project_status"] == "INCOMPLETE_BLOCKED", rows["Z.2"]


def test_P27_weakening_in_manifest_history_before_the_ledger_existed_is_detected(tmp_path, monkeypatch):
    """The ledger starts from the manifest's state when it was first built. A requirement weakened (or removed) in
    committed manifest history before that point is found by walking every committed manifest version."""
    strict = Repo(str(tmp_path / "strict"), monkeypatch)
    weak_crit = [{"metric": "widgets", "op": ">=", "value": 1}, {"metric": "leaks", "op": "==", "value": 0}]
    weak = Repo(str(tmp_path / "weak"), monkeypatch, items=[("Z.1", "Z", "make widgets", "A", [], weak_crit, 3)])
    top = weak.top
    orig = git(top, "rev-parse", "HEAD")
    git(top, "checkout", "-q", "--orphan", "rewritten")
    git(top, "rm", "-q", "-r", "--cached", ".")
    for f in os.listdir(top):
        if f != ".git":
            shutil.rmtree(os.path.join(top, f)) if os.path.isdir(os.path.join(top, f)) else os.remove(os.path.join(top, f))
    os.makedirs(os.path.join(weak.root, "docs", "v5"))
    shutil.copy(M.manifest_path(strict.root), M.manifest_path(weak.root))        # the pre-ledger, strict manifest
    weak.commit("pre-ledger manifest")
    git(top, "checkout", "-q", orig, "--", ".")
    weak.commit("the weakened requirements, with a fresh ledger")
    res, rows = weak.verify()
    assert any("history" in p for p in res["global_problems"] + rows["Z.1"]["problems"]), \
        (res["global_problems"], rows["Z.1"]["problems"])


def test_P28_a_changed_dataset_makes_evidence_stale(tmp_path, monkeypatch):
    ds = tmp_path / "dataset.txt"
    ds.write_text("12")
    src = GOOD_TEST.replace("    n = widgets.count()\n", f"    n = int(open({str(ds)!r}).read()) + 0 * widgets.count()\n") \
        .replace('evidence.record("Z.1", {"widgets": n, "leaks": leaks})',
                 f'evidence.record("Z.1", {{"widgets": n, "leaks": leaks}}, datasets={{"widgets": {str(ds)!r}}})')
    r = Repo(str(tmp_path / "r"), monkeypatch, test_src=src)
    r.honest()
    assert r.verify()[1]["Z.1"]["status"] == "VERIFIED"
    ds.write_text("13")
    _, row = rejected(r)
    assert has(row, "dataset"), row["problems"]


# ------------------------------------------------------------------------------------------------- skips and exits
@pytest.mark.parametrize("trick", ["helper_skip", "pytest_exit", "os_exit", "collect_ignore"])
def test_P29_skips_and_early_exits_hidden_outside_the_test_are_rejected(tmp_path, monkeypatch, trick):
    widgets, test_src, extra = WIDGETS, GOOD_TEST, {}
    if trick == "helper_skip":
        widgets = WIDGETS.replace("def leaks():\n    return 0", "def leaks():\n    import pytest\n    pytest.skip('x')")
    elif trick == "pytest_exit":
        test_src = GOOD_TEST + "\n\ndef test_zz():\n    import pytest\n    pytest.exit('done', returncode=0)\n\n\n" \
                               "def test_zzz():\n    assert 1 + 1 == 3\n"
    elif trick == "os_exit":
        test_src = GOOD_TEST.replace("    assert n >= 10", "    import os\n    os._exit(0)\n    assert n >= 10")
    else:                                               # the defect must stay hidden from collection
        widgets = DEFECT
        extra = {"tests/v5/conftest.py": "collect_ignore = ['test_gate_Z.py']\n"}
    r = Repo(str(tmp_path), monkeypatch, widgets=widgets, test_src=test_src, extra_files=extra)
    r.official_run()
    r.commit("evidence")
    try:
        r.review()
    except (SystemExit, KeyError, TypeError):           # nothing to review is also a rejection
        pass
    r.commit("review")
    rejected(r)


def test_P30_an_extra_test_file_cannot_stand_in_for_the_canonical_gate_test(tmp_path, monkeypatch):
    """Adding a test file to a requirement's list needs no decision. A producer adds one that measures an easier
    function (count_v2) while the canonical gate test records nothing about the defective count()."""
    widgets = DEFECT.replace("def count():\n    return 12", "def count():\n    return 3") + \
        "\n\ndef count_v2():\n    return 12\n"
    canonical = "import widgets\n\n\ndef test_z():\n    assert widgets.leaks() >= 0\n"
    extra = GOOD_TEST.replace("widgets.count()", "widgets.count_v2()").replace("widgets.leaks()", "0 * widgets.leaks()")
    r = Repo(str(tmp_path), monkeypatch, widgets=widgets, test_src=canonical,
             extra_files={"tests/v5/test_gate_Z_extra.py": extra})
    M.add_files(r.root, "Z.1", tests=["tests/v5/test_gate_Z_extra.py"])
    M.sync(r.root)
    r.commit("an extra test file for Z.1")
    _, row = attempt(r)
    assert row["status"] != "VERIFIED" and has(row, "canonical gate test file"), row["problems"]


# ------------------------------------------------------------------------------------------------- static detector
def test_testscan_flags_the_new_evasions():
    for src, finding in ((HELPER_CONST, "constant_record"), (MODULE_CONST, "constant_record"),
                         (DICT_CALL, "constant_record"), (CALL_FIRST_CONST, "constant_record"), (BUILTIN_CONST, "constant_record"),
                         (AUTOUSE_PATCH, "patch"), (ASSIGN_PATCH, "patch"), (SETATTR_HELPER, "patch"),
                         (SUPPRESS_TEST, "swallow"), (FALLBACK_TEST, "swallow")):
        f = next(x for x in testscan.scan_source(src) if x["name"] == "test_z")
        if finding:
            assert finding in f["findings"], (finding, f, src)
    assert testscan.scan_source(GOOD_TEST)[0]["findings"] == []
    for conf in (FLIP_CONFTEST, SKIP_BODY_CONFTEST):
        assert testscan.harness_findings(conf), conf
    assert testscan.harness_findings("import os, sys\nsys.path.insert(0, os.path.dirname(__file__))\n") == []
