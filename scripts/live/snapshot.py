"""Write an aggregate-only snapshot of a running live learner's scores (no post text, no user ids).

    python scripts/live/snapshot.py --workdir runs/live --out results/live/snapshot.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentrs.live.prequential import PrequentialScorer, PrequentialStore
from agentrs.live.spec import live_spec
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/live")
    ap.add_argument("--out", default="results/live/snapshot.json")
    a = ap.parse_args()
    wd = Path(a.workdir)
    st = json.loads((wd / "status.json").read_text())
    led = DelayedOutcomeLedger(wd / "ledger.sqlite")
    sc = PrequentialScorer(led, live_spec(), PrequentialStore(wd / "prequential.sqlite"))
    first = led._q("SELECT MIN(created_wall) t FROM actions")[0]["t"]
    out = {
        "snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_post_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(first)) if first else None,
        "uptime_s": st["uptime_s"], "checkpoint": st["checkpoint"], "optimizer_steps": st["optimizer_steps"],
        "weight_updates": st["updates"], "learn_cpu_seconds": st["learn_seconds"], "posts": st["posts"],
        "observations": st["observations"], "jobs": st["jobs"],
        "sources": {k: {kk: vv for kk, vv in v.items() if kk != "errors"} for k, v in st["sources"].items()},
        "deletions_propagated": st["governance"], "targets": st["targets"],
        "scores": sc.report(ci=True), "trend_hourly": sc.trend(3600.0),
        "trend_15min": sc.trend(900.0),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"overall": out["scores"]["overall"], "posts": out["posts"], "steps": out["optimizer_steps"]},
                     indent=1))


if __name__ == "__main__":
    main()
