"""Command line: run a protocol's feasibility calibration and append the record.

    python -m superlearner.cam_unified.envs.protocols.calibrate ow1 --split train --n 1 \
        --record superlearner/src/superlearner/cam_unified/envs/protocols/records/ow1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records")


def main(argv=None) -> int:
    from . import load
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--controls", default=None, help="comma-separated control names (default: all)")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--think-s", type=float, default=600.0)
    ap.add_argument("--record", default=None)
    a = ap.parse_args(argv)
    p = load(a.protocol)
    rec_path = a.record or os.path.join(RECORDS, f"{a.protocol}.jsonl")
    t0 = time.time()
    rec = p.calibrate(a.split, n_tasks=a.n, controls=a.controls.split(",") if a.controls else None, days=a.days,
                      think_s=a.think_s, record=rec_path)
    for t in rec["tasks"]:
        print(f"{t['world_family']}:{t['seed']} feasible={t['feasible']} fair={t['fairly_feasible']} "
              f"shortcuts={t['shortcuts_succeeded']} wall={t['wall_s']}s")
        for c in t["controls"]:
            print(f"   {c['control']:22s} {c['role']:9s} achieved={c['achieved']} day={c['success_day']} "
                  f"score={c['score']:.3f} metrics={json.dumps(c['metrics'])}")
    print(f"total {time.time() - t0:.1f}s -> {rec_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
