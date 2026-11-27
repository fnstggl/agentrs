"""G-rt probe 12 — reproduce measured numbers of docs/status/F.json.

    .venv/bin/python cam_live/docs/verification/repro/rt/p12_numbers.py

Reproduced here: runtime ms/decision with and without fsync and log bytes/decision (F's method: 300 loop iterations
on LabWorld with F's own test agent, as the status file defines the number), the count of REQUIRED items in the
Discord task, and the runtime source line count. The suite pass counts are reproduced by running pytest (see G_rt.md).
fsync timings depend on the file system, so the run directory is created on the repository's file system, inside
docs/verification/repro/rt/.num_<pid> (removed at the end). Nothing of F's evidence is overwritten.
"""
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report  # noqa: E402

from cam_live.config import PROJECT_ROOT  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests", "runtime"))
from cam_live.diagnostics.world import DAY, LabConfig  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.loop import RuntimeConfig  # noqa: E402
from rt_helpers import make_lab  # noqa: E402  (F's builder and F's test agent: the number is defined with them)

R = Report("p12_numbers")
claim = json.load(open(os.path.join(PROJECT_ROOT, "docs", "status", "F.json")))["measured"]
HERE = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(HERE, f".num_{os.getpid()}")
got = {}
try:
    for fsync in (True, False):
        d = os.path.join(base, "on" if fsync else "off")
        os.makedirs(d)
        lab = make_lab(os.path.join(d, "run"), lab_cfg=LabConfig(seed=3, target=100000, horizon_s=30 * DAY, budget=5000.0),
                       agent_kw={"durable": fsync}, rt_cfg=RuntimeConfig(fsync=fsync))
        t0 = time.perf_counter()
        for _ in range(300):
            lab.rt.step()
        dt = time.perf_counter() - t0
        n = lab.rt.counters["decisions"]
        size = sum(os.path.getsize(os.path.join(lab.run_dir, f)) for f in os.listdir(lab.run_dir) if f.endswith(".jsonl"))
        got["on" if fsync else "off"] = {"decisions": n, "ms_per_decision": round(1000 * dt / n, 2), "log_bytes_per_decision": int(size / n)}
        lab.rt.close()
finally:
    shutil.rmtree(base, ignore_errors=True)
R.note("measured now", got)
for key, mine in (("runtime_ms_per_decision_fsync_on", got["on"]["ms_per_decision"]),
                  ("runtime_ms_per_decision_fsync_off", got["off"]["ms_per_decision"])):
    R.check(f"{key}: claimed {claim[key]}, measured {mine} (same order: within a factor of 2; the machine is shared)",
            0.5 <= mine / claim[key] <= 2.0, {"ratio": round(mine / claim[key], 2)})
b = got["on"]["log_bytes_per_decision"]
R.check(f"log_bytes_per_decision_16_candidates: claimed {claim['log_bytes_per_decision_16_candidates']}, measured {b} (within 2 %)",
        abs(b - claim["log_bytes_per_decision_16_candidates"]) / claim["log_bytes_per_decision_16_candidates"] < 0.02)
n_req = len(T.unset_paths(T.load_task_raw(os.path.join(PROJECT_ROOT, "tasks", "discord_100.yaml"))))
R.check(f"discord_task_required_items_unset: claimed {claim['discord_task_required_items_unset']}, counted {n_req}",
        n_req == claim["discord_task_required_items_unset"])
lines = sum(sum(1 for _ in open(p)) for p in glob.glob(os.path.join(PROJECT_ROOT, "src", "cam_live", "runtime", "*.py")))
R.check(f"runtime_source_lines: claimed {claim['runtime_source_lines']}, counted {lines}", lines == claim["runtime_source_lines"])
sys.exit(R.finish())
