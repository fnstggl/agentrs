"""Governance check for superlearner/docs/CAM_UNIFIED_COMPLETION.json.

A requirement may only claim a status its evidence supports:
  IMPLEMENTED and above  -> implementation files exist and at least one acceptance test id resolves to a real test
  INTEGRATION_TESTED and above -> at least one resolvable integration test and a non-null `verified_by` (workstream H)
  EMPIRICALLY_DEMONSTRATED / FAILED_EMPIRICALLY -> at least one evidence entry
Test ids have the form "relative/path/test_file.py::test_name" (relative to the repository root).
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "superlearner", "docs", "CAM_UNIFIED_COMPLETION.json")
ORDER = ["NOT_STARTED", "IN_PROGRESS", "BLOCKED", "IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY",
         "EMPIRICALLY_DEMONSTRATED"]
AT_LEAST_IMPLEMENTED = {"IMPLEMENTED", "INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}
AT_LEAST_INTEGRATED = {"INTEGRATION_TESTED", "FAILED_EMPIRICALLY", "EMPIRICALLY_DEMONSTRATED"}


def _load() -> dict:
    with open(MANIFEST) as fh:
        return json.load(fh)


def _resolves(test_id: str) -> bool:
    if "::" not in test_id:
        return False
    path, name = test_id.split("::", 1)
    name = name.split("[", 1)[0].split("::")[-1]
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return False
    with open(full) as fh:
        src = fh.read()
    return re.search(rf"^\s*(async\s+)?def {re.escape(name)}\(", src, re.M) is not None


def test_manifest_schema_and_statuses():
    m = _load()
    allowed = set(m["allowed_statuses"])
    ids = [r["id"] for r in m["requirements"]]
    assert len(ids) == len(set(ids)), "duplicate requirement ids"
    for r in m["requirements"]:
        assert r["status"] in allowed, (r["id"], r["status"])
        assert r["workstream"] in m["workstreams"], r["id"]
        for d in r["dependencies"]:
            assert d in ids, (r["id"], "unknown dependency", d)


def test_claims_are_backed_by_files_and_tests():
    m = _load()
    problems = []
    for r in m["requirements"]:
        st = r["status"]
        if st in AT_LEAST_IMPLEMENTED:
            if not r["files"]:
                problems.append(f"{r['id']}: {st} without implementation files")
            for f in r["files"]:
                if not os.path.exists(os.path.join(ROOT, f)):
                    problems.append(f"{r['id']}: missing file {f}")
            if not any(_resolves(t) for t in r["acceptance_tests"]):
                problems.append(f"{r['id']}: {st} without a resolvable acceptance test")
        if st in AT_LEAST_INTEGRATED:
            if not any(_resolves(t) for t in r["integration_tests"]):
                problems.append(f"{r['id']}: {st} without a resolvable integration test")
            if not r.get("verified_by"):
                problems.append(f"{r['id']}: {st} not confirmed by independent verification (verified_by)")
        if st in {"EMPIRICALLY_DEMONSTRATED", "FAILED_EMPIRICALLY"} and not r["evidence"]:
            problems.append(f"{r['id']}: {st} without evidence")
    assert not problems, "\n".join(problems)
