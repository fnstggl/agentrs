"""G2-learn round 2: classify the slow-step guard's rejections in learner logs (ContinualLearner log files).

    .venv/bin/python cam_live/docs/verification/repro/g2learn/guard_vetoes.py LOG [LOG ...]

For every judged slow step: accepted / rejected; for a rejection, whether every provider that failed it rose by less
than 1e-4 of its own held-out loss ("micro veto": a rise of that size is far below anything that matters for a loss,
and is only rejected because the provider's tolerance collapsed to the float floor, 1e-7 x |loss|, when its chunks
moved together, SE ~ 0, or it had a single chunk), and the largest improvement another provider would have kept.
"""
import json
import sys

tot = {"judged": 0, "accepted": 0, "rejected": 0, "micro": 0}
for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        d = json.loads(line)
        d = d.get("data", d)
        if d.get("timescale") != "slow_adapters":
            continue
        n = d.get("notes") or "{}"
        n = json.loads(n) if isinstance(n, str) else n
        g = n.get("guard") or {}
        bp = g.get("by_provider") or {}
        if not bp:
            continue
        failing = {k: v for k, v in bp.items() if not v.get("accepted")}
        rel = {k: v["paired_change"] / max(abs(v["validation_before"]), 1e-12) for k, v in failing.items()}
        drift_only = all(all("summed change" in r for r in v.get("reasons", [])) for v in failing.values()) if failing else False
        micro = bool(failing) and all(0 <= r < 1e-4 for r in rel.values()) and not drift_only
        best = min(((v["paired_change"] / max(abs(v["validation_before"]), 1e-12), k) for k, v in bp.items()), default=(0, ""))
        rows.append((bool(g.get("accepted")), micro, {k: f"{r:+.1e}" for k, r in rel.items()}, best))
    acc = sum(1 for r in rows if r[0])
    rej = [r for r in rows if not r[0]]
    mic = sum(1 for r in rej if r[1])
    print(f"{path}\n  judged {len(rows)}, accepted {acc}, rejected {len(rej)}, of which micro vetoes {mic}")
    for r in rej:
        print(f"    rejected: failing providers (rel. rise) {r[2]}; best relative change elsewhere {r[3][0]:+.2e} ({r[3][1]})"
              + ("  [micro veto]" if r[1] else ""))
    tot["judged"] += len(rows)
    tot["accepted"] += acc
    tot["rejected"] += len(rej)
    tot["micro"] += mic
print("total", tot)
