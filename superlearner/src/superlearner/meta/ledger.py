"""The discovery ledger: an append-only record of every candidate experiment - successful, failed or inconclusive."""
from __future__ import annotations

import json
import os
import time


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def append(self, rec: dict) -> dict:
        rec = dict(rec, t=time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def entries(self, kind: str | None = None) -> list[dict]:
        out = []
        if not os.path.exists(self.path):
            return out
        for line in open(self.path):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if kind is None or r.get("kind") == kind:
                    out.append(r)
        return out

    def experiments(self) -> list[dict]:
        """One final record per candidate (the last one written for it)."""
        last: dict = {}
        for r in self.entries("experiment"):
            last[r["candidate"]] = r
        return list(last.values())

    def summary_text(self, n: int = 12) -> str:
        """Short history for the proposer's prompt: what was tried and what happened."""
        lines = []
        for r in self.experiments()[-n:]:
            d = r.get("comparison", {}).get("aulc_eval", {})
            lines.append(f"- [{r.get('decision')}] {r.get('source')} / {r.get('target')}: {str(r.get('hypothesis'))[:160]} "
                         f"-> AULC difference {d.get('diff', 0):+.3f} (n={d.get('n_cand', 0)}), "
                         f"stage {r.get('stage_reached')}{'; ' + str(r.get('reject_reason'))[:80] if r.get('reject_reason') else ''}")
        return "\n".join(lines)

    def table(self) -> list[dict]:
        rows = []
        for r in self.experiments():
            d = r.get("comparison", {}).get("aulc_eval", {})
            rows.append({"candidate": r["candidate"], "source": r.get("source"), "target": r.get("target"),
                         "decision": r.get("decision"), "stage": r.get("stage_reached"), "diff_aulc": d.get("diff"),
                         "ci": d.get("ci"), "n": d.get("n_cand"), "hypothesis": str(r.get("hypothesis"))[:100]})
        return rows
