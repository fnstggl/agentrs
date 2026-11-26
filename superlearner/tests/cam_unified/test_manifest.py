"""Governance check for superlearner/docs/CAM_UNIFIED_COMPLETION.json (rebuilt 2026-11-26, plan §3 W9).

The structural checks of the earlier version (files exist, a test id resolves to a `def`) are subsumed by the
executable checker `superlearner.verify_completion`: a claim must be backed by commit-bound JUnit PASS records of
its tests, clean integration runs that executed its integration entry, an independent verification record and, for
empirical claims, an experiment result that passes the protocol checks. This test runs that checker on the real
manifest and the recorded results, so a claim the evidence does not support FAILS here, visibly.

To make it pass, the manifest owner either records results (``python -m superlearner.verify_completion
--run-tests``, which runs every cited test with the binding plugin) or lowers the claims to what is recorded.
"""
from __future__ import annotations

import superlearner.verify_completion as vc


def test_manifest_schema_and_statuses():
    m = vc.load_manifest()
    allowed = set(m["allowed_statuses"])
    assert allowed <= set(vc.STATUSES), allowed - set(vc.STATUSES)
    ids = [r["id"] for r in m["requirements"]]
    assert len(ids) == len(set(ids)), "duplicate requirement ids"
    for r in m["requirements"]:
        assert r["status"] in allowed, (r["id"], r["status"])
        assert r["workstream"] in m["workstreams"], r["id"]
        for d in r["dependencies"]:
            assert d in ids, (r["id"], "unknown dependency", d)


def test_claims_are_backed_by_recorded_evidence():
    rep = vc.check(vc.load_manifest(), vc.load_results())
    lines = list(rep["global_problems"]) + [f"{rid}: {p}" for rid, ps in rep["problems"].items() for p in ps]
    assert rep["claims_consistent"], f"{len(lines)} problems, e.g.\n" + "\n".join(lines[:40])


def test_a_synthetic_claim_with_an_unexecuted_test_is_rejected():
    m = {"allowed_statuses": vc.STATUSES, "workstreams": {"E": "memory"},
         "requirements": [{"id": "Z01", "status": "IMPLEMENTED", "workstream": "E", "dependencies": [],
                           "files": ["superlearner/src/superlearner/cam_unified/memory/checkpoint.py"],
                           "acceptance_tests": ["superlearner/tests/cam_unified/memory/test_mem_checkpoint.py::"
                                                "test_full_lifetime_roundtrip_restores_everything"],
                           "integration_tests": [], "evidence": []}]}
    rep = vc.check(m, {}, global_checks=False)
    assert not rep["claims_consistent"]
    assert any("never executed" in p for p in rep["problems"]["Z01"])
