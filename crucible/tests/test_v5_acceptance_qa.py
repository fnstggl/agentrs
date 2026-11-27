"""QA measurements that read the repository's branches (Gate X.n1, RP.5) measured where official evidence is made:
in a fresh clone of the repository (official.clone), whose own refs are not the repository's (wave 3b repair pass)."""
from __future__ import annotations

import os
import subprocess

from crucible.acceptance.v5 import official, qa


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo(tmp):
    top = os.path.join(tmp, "repo")
    root = os.path.join(top, "crucible")
    os.makedirs(os.path.join(root, "docs", "v5", "workstreams"))
    _git(tmp, "init", "-q", "-b", qa.INTEGRATION_BRANCH, top)
    _git(top, "config", "user.email", "t@t")
    _git(top, "config", "user.name", "t")
    open(os.path.join(root, "docs", "v5", "workstreams", "P.json"), "w").write('{"branch": "c5c/w9-p"}\n')
    open(os.path.join(root, "a.txt"), "w").write("a\n")
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", "base")
    for b in ("c5c/w9-p", "c5c/w9-q"):
        _git(top, "branch", b)
    return top, root


def _commit(top, rel, text, msg):
    p = os.path.join(top, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", msg)
    return _git(top, "rev-parse", "HEAD")


def test_integration_state_reads_the_source_repository_in_an_official_clone(tmp_path):
    """In the clone the workstream branches exist only as origin/*, so X.n1 saw no branch and no integration branch
    (run_after_full_integration False whatever the state). Measured in the clone it now reads the source's branches."""
    top, root = _repo(str(tmp_path))
    _git(top, "checkout", "-q", "c5c/w9-q")
    _commit(top, "crucible/q.txt", "q\n", "q work")                      # an unmerged workstream branch
    _git(top, "checkout", "-q", qa.INTEGRATION_BRANCH)
    head = _git(top, "rev-parse", "HEAD")
    croot = official.clone(root, "HEAD", str(tmp_path / "clone"), "r1", "evidence")
    st = qa.integration_state(croot)
    assert st["repository"] == "source of the official clone"
    assert st["head"] == head and st["integration_tip"] == head
    assert st["metrics"] == {"run_after_full_integration": False, "unmerged_workstream_branches": 1,
                             "workstream_branches_checked": 2}, st
    assert st["unmerged"] == ["c5c/w9-q"]
    _git(top, "merge", "-q", "--no-edit", "c5c/w9-q")
    head = _git(top, "rev-parse", "HEAD")
    croot = official.clone(root, "HEAD", str(tmp_path / "clone2"), "r2", "evidence")
    assert qa.integration_state(croot)["metrics"]["run_after_full_integration"] is True


def test_the_checked_commits_own_records_do_not_make_it_stale_but_later_work_does(tmp_path):
    """The evidence and review of the checked commit are committed after it; the integration head is the checked
    commit once those record-only commits are skipped. Any other later commit means the check ran too early."""
    top, root = _repo(str(tmp_path))
    checked = _git(top, "rev-parse", "HEAD")
    croot = official.clone(root, checked, str(tmp_path / "clone"), "r1", "evidence")
    _commit(top, "crucible/docs/v5/evidence/gate_X/X.n1.json", "{}\n", "evidence")
    _commit(top, "crucible/docs/v5/reviews/X.n1.json", "{}\n", "review")
    st = qa.integration_state(croot)
    assert st["metrics"]["run_after_full_integration"] is True and st["record_commits_skipped"] == 2, st
    _commit(top, "crucible/src/late.py", "x = 1\n", "late work")
    assert qa.integration_state(croot)["metrics"]["run_after_full_integration"] is False


def test_ref_repository_is_the_same_directory_of_the_source_and_the_checkout_itself_elsewhere(tmp_path):
    top, root = _repo(str(tmp_path))
    assert qa.ref_repository(root) == root
    croot = official.clone(root, "HEAD", str(tmp_path / "clone"), "r1", "evidence")
    assert os.path.realpath(qa.ref_repository(croot)) == os.path.realpath(root)
    plain = str(tmp_path / "plain")
    _git(str(tmp_path), "clone", "-q", top, plain)               # not an official clone: no marker
    assert qa.ref_repository(os.path.join(plain, "crucible")) == os.path.join(plain, "crucible")
