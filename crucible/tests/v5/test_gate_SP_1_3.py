"""Gate SP-1.3 (MASTER_SPEC Part 1.3): the persistent engineering documents exist, MASTER_SPEC.md is the verbatim
specification (SHA-256 anchored in requirements.MASTER_SPEC_SHA256, equal to the file at 4a7591e and at the recovered
state a881d18), and no document still names the old integration branch as the merge target. SP-1.3b: every manifest
entry lists implementation files, tests and the implementation commit, and every workstream report is merged."""
from __future__ import annotations

import hashlib
import os
import subprocess

from crucible.acceptance.v5 import manifest as M
from crucible.acceptance.v5 import qa
from crucible.acceptance.v5 import requirements as R
from crucible.acceptance.v5.checker import unmet
from crucible.acceptance.v5.evidence import record

ROOT = M.crucible_root()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-1.3 --agent <name>"


def _blob_sha(commit: str, rel: str) -> str | None:
    p = subprocess.run(["git", "show", f"{commit}:./{rel}"], cwd=ROOT, capture_output=True)
    return hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None


def test_SP_1_3_persistent_engineering_documents():
    d = M.docs_dir(ROOT)
    present = [f for f in qa.PART_1_3 if os.path.exists(os.path.join(d, f))]
    docs = qa.documents(ROOT, final=False)
    have = M.spec_sha256(ROOT)
    at_first, at_recovery = _blob_sha("4a7591e", M.SPEC_REL), _blob_sha("a881d18", M.SPEC_REL)
    m = {"v5_documents_present": len(present), "v5_documents_missing": len([x for x in docs["missing"]
                                                                            if not x.startswith("components/")]),
         "master_spec_unchanged": have == at_first == at_recovery,
         "master_spec_sha256_matches_anchor": have == R.MASTER_SPEC_SHA256,
         "stale_integration_branch_references": docs["metrics"]["stale_integration_branch_references"]}
    record("SP-1.3", m, command=CMD, notes=f"missing: {docs['missing']}; stale: {docs['stale']}; spec sha {have}")
    bad = unmet("SP-1.3", m)
    assert not bad, (bad, docs["missing"], docs["stale"])


def test_SP_1_3b_manifest_entries_complete_and_reports_merged():
    man = M.load(ROOT)
    no_impl = [e["id"] for e in man["requirements"] if not e["implementation_files"]]
    no_commit = [e["id"] for e in man["requirements"] if not e.get("commit")]
    ws = M.workstream_files(ROOT)
    by = {e["id"]: e for e in man["requirements"]}
    unmerged = sorted({rid for rid, info in ws.items() if rid in by and (
        set(info["tests"]) - set(by[rid]["tests"]) or set(info["implementation_files"]) - set(by[rid]["implementation_files"]))})
    m = {"entries_without_implementation_files": len(no_impl), "entries_without_commit": len(no_commit),
         "workstream_reports_unmerged": len(unmerged)}
    record("SP-1.3b", m, command=CMD, notes=f"no implementation files: {no_impl[:40]}; no commit: {no_commit[:20]}; "
                                            f"unmerged report entries: {unmerged[:20]}")
    bad = unmet("SP-1.3b", m)
    assert not bad, (bad, no_impl[:10], unmerged[:10])
