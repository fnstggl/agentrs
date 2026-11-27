"""Measure the runtime's own overhead per decision on LabWorld (CPU; small test agent), with and without fsync.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/F/measure_loop.py
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests", "runtime"))

from cam_live.diagnostics.world import DAY, LabConfig  # noqa: E402
from cam_live.runtime.loop import RuntimeConfig  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

out = {}
for fsync in (True, False):
    d = tempfile.mkdtemp(prefix="f_measure_", dir=os.path.join(ROOT, ".."))
    lab = make_lab(os.path.join(d, "run"), lab_cfg=LabConfig(seed=3, target=100000, horizon_s=30 * DAY, budget=5000.0),
                   agent_kw={"durable": fsync}, rt_cfg=RuntimeConfig(fsync=fsync))
    t0 = time.perf_counter()
    for _ in range(300):
        lab.rt.step()
    dt = time.perf_counter() - t0
    n = lab.rt.counters["decisions"]
    size = sum(os.path.getsize(os.path.join(lab.run_dir, f)) for f in os.listdir(lab.run_dir)
               if f.endswith(".jsonl"))
    out["fsync_on" if fsync else "fsync_off"] = {
        "decisions": n, "seconds": round(dt, 3), "ms_per_decision": round(1000 * dt / n, 2),
        "log_bytes_per_decision": int(size / n), "agent_compute_s": round(lab.rt.counters["compute_s"], 3)}
    import shutil
    shutil.rmtree(d, ignore_errors=True)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "loop_overhead.json"), "w"), indent=1)
