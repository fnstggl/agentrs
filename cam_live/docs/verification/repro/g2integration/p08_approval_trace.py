"""G2-integration probe p08 — what happened to the approvals of a staging validation run? (LX-11, OB-01, IN-07)

Reads a finished `cam-live validate --mode staging` run directory (its computer phase) and joins three records:
the authorization service's audit log (`computer/auth_state/audit.jsonl`), its approval table
(`computer/auth_state/auth/state.sqlite`, opened read-only) and the runtime's `action_result` records
(`computer/runtime/events.jsonl`). Checks, for every approval the operator resolved:
  * an approval the operator APPROVED is never reported to the learner as rejected by the operator;
  * an approved action is either executed or reported with the true reason it was not.
Also prints the external-action ledger of the task (what used the budget).

Usage (from /workspace/camlive):
    .venv/bin/python -I cam_live/docs/verification/repro/g2integration/p08_approval_trace.py <validation out dir>
e.g. the run B directory of G2_integration.md: <scratch>/staging/run2
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main() -> int:
    root = os.path.abspath(sys.argv[1])
    comp = os.path.join(root, "computer")
    db = sqlite3.connect(f"file:{os.path.join(comp, 'auth_state', 'auth', 'state.sqlite')}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    approvals = {r["approval_id"]: dict(r) for r in db.execute("SELECT * FROM approvals")}
    usage = {r["k"] if "k" in r.keys() else r[1]: r[2] for r in db.execute("SELECT * FROM usage")}
    grants = [dict(r) for r in db.execute("SELECT status, approval_id, reserve_json, issued_at FROM grants")]
    audit = jsonl(os.path.join(comp, "auth_state", "audit.jsonl"))
    results = [r["data"] for r in jsonl(os.path.join(comp, "runtime", "events.jsonl")) if r.get("type") == "action_result"]
    print(f"approvals at the authorization service: {[(a, v['status']) for a, v in approvals.items()]}")
    print(f"task usage at the authorization service: {usage}")
    print(f"grants: {[(g['status'], g['approval_id'], g['reserve_json']) for g in grants]}")
    print("authorization trail:")
    for r in audit:
        d = r.get("data", r)
        ev = r.get("event") or r.get("type") or d.get("event")
        if ev in ("authorize", "approval_requested", "approval_resolved", "grant_use"):
            print(f"  {ev:18s} {d.get('operation', ''):18s} {d.get('verdict', d.get('status', '')):15s} "
                  f"{(d.get('approval_id') or ''):22s} {'; '.join(d.get('reasons') or [])[:110]}")
    print("runtime action results:")
    for d in results:
        print(f"  {d.get('operation', ''):18s} {d.get('kind', ''):9s} {d.get('status', ''):14s} "
              f"{d.get('failure_class', ''):20s} {(d.get('approval_id') or ''):22s} {str(d.get('failure_detail'))[:90]}")
    for aid, a in approvals.items():
        mine = [d for d in results if d.get("approval_id") == aid and d.get("kind") != "queued"]
        if a["status"] == "approved":
            told = [d for d in mine if "rejected" in str(d.get("failure_detail")) or d.get("kind") == "rejected"]
            check(f"approval {aid} (approved by {a.get('resolved_by')}) is not reported to the learner as rejected by "
                  f"the operator", not told, "; ".join(str(d.get("failure_detail")) for d in told)[:160])
            ran = [d for d in mine if d.get("kind") == "execute"]
            why = [d for d in mine if d.get("kind") in ("over_limit", "cancelled", "denied")]
            check(f"approval {aid}: the approved action was executed, or not executed with the true reason",
                  bool(ran) or bool(why), f"executed {len(ran)}, reported {[d.get('kind') for d in mine]}")
    print(f"== p08_approval_trace: {broken} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
