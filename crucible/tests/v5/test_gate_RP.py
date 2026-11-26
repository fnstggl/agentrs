"""Gate RP (Clarification 3, recovery preamble), measured on the real repository.

RP.1 static scan of every .py file under crucible/ for imports of CAM / superlearner / agentrs learner code, and no
gate whose requirement depends on CAM. RP.2 the audit reports of all 15 areas with skeptic reviews, committed. RP.3 the
progress ledger lists the exact remaining requirements, the integration head and every milestone commit are on GitHub
(`git ls-remote`, a network read), and no milestone work exists only uncommitted: no worktree whose branch is merged
into the integration branch has uncommitted changes, and the worktree that has the integration branch checked out is
clean. RP.4 every artifact of docs/v5/ARTIFACTS.json is present, hash-matching and durable (GitHub or a public source;
/workspace is not a persistent volume, so local copies do not count). RP.5 the old branch is still at the recovered
state, locally and on GitHub. RP.6 no credential pattern in the tree or in any reachable commit, no credentials in
remote URLs, and the remote-tracking integration branch never recorded a forced update.
"""
from __future__ import annotations

import subprocess

import pytest

from crucible.acceptance.v5 import manifest as M
from crucible.acceptance.v5 import progress, qa
from crucible.acceptance.v5.checker import unmet
from crucible.acceptance.v5.evidence import record

ROOT = M.crucible_root()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate RP --agent <name>"


def test_RP_1_crucible_independent_of_cam():
    res = qa.cam_scan(ROOT)
    m = res["metrics"]
    record("RP.1", m, command=CMD, notes=f"imports: {res['imports'][:30]}; gates naming CAM as a dependency: {res['gates']}")
    bad = unmet("RP.1", m)
    assert not bad, (bad, res["imports"][:10], res["gates"])


def test_RP_2_independent_audit_with_skeptic_reviews():
    res = qa.audit_reports(ROOT)
    m = res["metrics"]
    record("RP.2", m, command=CMD, notes=f"missing areas: {res['missing']}")
    bad = unmet("RP.2", m)
    assert not bad, (bad, res["missing"])


def _integration_worktree_dirty() -> list[str]:
    out, cur = [], {}
    for line in subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True).stdout.splitlines() + [""]:
        if not line.strip():
            if cur.get("branch") == f"refs/heads/{qa.INTEGRATION_BRANCH}":
                st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cur["worktree"],
                                    capture_output=True, text=True).stdout.splitlines()
                out += [f"{cur['worktree']}: {s.strip()}" for s in st if s.strip()]
            cur = {}
            continue
        k, _, v = line.partition(" ")
        cur[k] = v
    return out


@pytest.mark.timeout(600)
def test_RP_3_ledger_pushed_and_nothing_only_uncommitted():
    pm, probs = progress.metrics(ROOT)
    ms = progress.load(ROOT)["milestones"]
    rs = qa.remote_state(ROOT, milestones=ms)
    dirty_integration = _integration_worktree_dirty()
    m = {"milestones_recorded": pm["milestones_recorded"], "ledger_lists_remaining_requirements":
         pm["ledger_lists_remaining_requirements"], "integration_head_pushed": rs["metrics"]["integration_head_pushed"],
         "unpushed_milestone_commits": rs["metrics"]["unpushed_milestone_commits"],
         "merged_worktrees_with_uncommitted_changes": rs["metrics"]["merged_worktrees_with_uncommitted_changes"],
         "uncommitted_milestone_work": len(dirty_integration) + rs["metrics"]["merged_worktrees_with_uncommitted_changes"]}
    record("RP.3", m, command=CMD, notes=f"remote head {rs['remote_head']}, local {rs['local_head']}; unpushed "
                                         f"milestones {rs['unpushed']}; dirty merged worktrees {rs['dirty_merged_worktrees']}; "
                                         f"uncommitted in the integration worktree: {dirty_integration[:10]}; ledger "
                                         f"problems {probs[:5]}")
    bad = unmet("RP.3", m)
    assert not bad, (bad, rs["unpushed"], dirty_integration[:5])


@pytest.mark.timeout(600)
def test_RP_4_important_artifacts_durably_preserved():
    res = qa.artifacts(ROOT)
    m = res["metrics"]
    record("RP.4", m, command=CMD, notes=f"not durable: {res['unpreserved'][:30]}; hash mismatches: {res['mismatched']}; "
                                         f"problems: {res['problems'][:10]}")
    bad = unmet("RP.4", m)
    assert not bad, (bad, res["unpreserved"][:10], res["mismatched"][:5])


@pytest.mark.timeout(600)
def test_RP_5_old_branch_immutable():
    rs = qa.remote_state(ROOT)
    m = {k: rs["metrics"][k] for k in ("old_branch_tip_matches_anchor", "old_branch_remote_matches_anchor")}
    record("RP.5", m, command=CMD, notes=f"old branch {rs['old_branch']}")
    bad = unmet("RP.5", m)
    assert not bad, (bad, rs["old_branch"])


@pytest.mark.timeout(1800)
def test_RP_6_no_force_push_and_no_exposed_secrets():
    sec = qa.secrets_scan(ROOT)
    fwd = qa.remote_forward_only(ROOT)
    m = {**sec["metrics"], **fwd["metrics"]}
    record("RP.6", m, command=CMD, notes=f"secret patterns at: {sec['where']}; forced updates: {fwd['forced']}")
    bad = unmet("RP.6", m)
    assert not bad, (bad, sec["where"][:5], fwd["forced"])
