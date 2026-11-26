"""Gate SP-18 (MASTER_SPEC Part 18): the final report's completion table has exactly the 18 Part 18 capabilities, is
populated only from the checker's verdicts (a row is VERIFIED only when every mapped requirement is), uses only the
Part 18 status vocabulary, and the committed docs/v5/FINAL_REPORT.md carries the table the checker gives now."""
from __future__ import annotations

import os
import re

from crucible.acceptance.v5 import manifest as M
from crucible.acceptance.v5 import report
from crucible.acceptance.v5.checker import unmet, verify_all
from crucible.acceptance.v5.evidence import record

ROOT = M.crucible_root()
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-18 --agent <name>"


def _table_rows(text: str) -> list[list[str]]:
    sec = text.split("## Completion table (Part 18)", 1)[1].split("\n## ", 1)[0]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in sec.splitlines() if ln.startswith("| ")]
    return [r for r in rows if r and r[0] not in ("Capability",) and not set(r[0]) <= set("-")]


def test_SP_18_completion_table_from_verified_evidence():
    res = verify_all(ROOT)
    m = report.table_metrics(res)
    path = os.path.join(M.docs_dir(ROOT), "FINAL_REPORT.md")
    committed = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    have = [(r[0], r[-1]) for r in _table_rows(committed)] if committed else []
    want = [(r["capability"], r["status"]) for r in report.table(res)]
    m["committed_table_matches_checker"] = have == want
    record("SP-18", m, command=CMD, notes=f"rows: {want}; committed: {have}")
    bad = unmet("SP-18", m)
    assert not bad and m["committed_table_matches_checker"], (bad, have, want)
