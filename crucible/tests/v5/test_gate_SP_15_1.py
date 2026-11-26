"""SP-15.1 - No claim of real-world generalisation without executed, authorised Level 6 experiments
(MASTER_SPEC Part 15.1; workstream N).

The whole documentation corpus (`integrity.corpus`: README, status and architecture documents, docs/**, evidence
notes and every module docstring) is scanned clause by clause for claims of real-world or open-internet
generalisation (`integrity.scan_claims("real_world")`). Before its verdict counts, the scanner must classify its
planted claims and disclaimers correctly. Executed authorised Level 6 experiments are counted from Level 6 logs in
the repository (none exist), so every claim found is unsupported.
"""
from __future__ import annotations

import glob
import os

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import integrity as G
from crucible.research.v5 import levels as LV


def test_SP15_1_no_unsupported_real_world_claims():
    files = G.corpus()
    executed = len(glob.glob(os.path.join(G.ROOT, "docs", "v5", "evidence", "level6", "*.jsonl")))
    r = G.scan_claims("real_world", files, supported=executed >= LV.MIN_EXPERIMENTS_FOR_CLAIM)
    record("SP-15.1", {"unsupported_real_world_claims": r["unsupported_claims"], "files_scanned": r["files_scanned"],
                       "statements_checked": r["statements_checked"], "planted_controls_correct": r["controls"]["ok"],
                       "executed_level6_experiments": executed},
           notes=f"Claims found: {r['claims'][:10]}; scanner controls {r['controls']}")
    assert r["controls"]["ok"], r["controls"]
    assert r["files_scanned"] > 100 and r["statements_checked"] >= 1
    assert r["unsupported_claims"] == 0, r["claims"]
