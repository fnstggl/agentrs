#!/usr/bin/env python
"""Merge docs/verification/verdicts_<lens>.json into docs/verification/verdicts.json (worst status wins)."""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "verification")
RANK = {"failed": 0, "unverified": 1, "partial": 2, "verified": 3}


def main() -> int:
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "verdicts_*.json"))):
        lens = os.path.basename(path)[len("verdicts_"):-len(".json")]
        for rid, v in json.load(open(path)).items():
            v = dict(v, by=v.get("by") or f"G-{lens}")
            st = v.get("status", "unverified")
            if st not in RANK:
                v["status"] = st = "unverified"
            cur = out.get(rid)
            if cur is None:
                out[rid] = v
            else:
                worst = v if RANK[st] < RANK[cur["status"]] else cur
                other = cur if worst is v else v
                out[rid] = dict(worst, by=f"{worst['by']}+{other['by']}",
                                notes=f"{worst.get('notes', '')} || [{other['by']}: {other['status']}] {other.get('notes', '')}",
                                evidence=list(worst.get("evidence", [])) + list(other.get("evidence", [])))
    with open(os.path.join(ROOT, "verdicts.json"), "w") as f:
        json.dump(dict(sorted(out.items())), f, indent=1)
    counts = {}
    for v in out.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(json.dumps({"requirements": len(out), "by_status": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
