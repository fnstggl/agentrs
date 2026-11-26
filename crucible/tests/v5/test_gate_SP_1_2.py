"""SP-1.2 (MASTER_SPEC Part 1.2): actual compute is coordinated through the resource scheduler (crucible.scale.slots);
expensive GPU jobs are not started outside it.

The scheduler's usage log exists and holds records of governed jobs; over a two-minute census (12 samples) every
process of this container that holds a GPU context runs inside a governed job granted GPU memory (its environment
carries the grant the scheduler sets; nvidia-smi reports host pids, so processes of other tenants on the host are
counted separately as foreign and are not this project's). A GPU job launched through the scheduler is seen as
governed in the census (positive control)."""
import os
import sys
import time

import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import slots

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 600 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-1.2")


def test_SP_1_2_compute_through_the_scheduler():
    import subprocess
    root = slots.slots_root()
    log = slots.usage(root)
    v2 = [r for r in log if r.get("v") == 2]
    code = "import torch, time\nx = torch.ones(1024, device='cuda'); torch.cuda.synchronize(); time.sleep(40)\n"
    ctl = subprocess.Popen([sys.executable, "-m", "crucible.scale.slots", "--cpu", "1", "--gpu-mb", "400",
                            "--gpu-s", "120", "--label", "SP-1.2 positive control", "--", sys.executable, "-c", code],
                           env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    samples, ungoverned, governed, foreign = [], set(), set(), 0
    for _ in range(12):
        c = slots.gpu_census()
        samples.append(c)
        ungoverned |= {(p["pid"], p.get("cmd", "")[:80]) for p in c["ungoverned"]}
        governed |= {p["pid"] for p in c["governed"]}
        foreign = max(foreign, c["foreign_gpu_apps"])
        time.sleep(10)
    ctl.wait(timeout=300)
    seen_ctl = any("SP-1.2 positive control" in r.get("label", "") for r in slots.usage(root, since=time.time() - 600))
    m = {"scheduler_usage_log_present": len(log) > 0 and len(v2) > 0, "usage_records": len(log),
         "governor_records": len(v2), "gpu_jobs_outside_scheduler_observed": len(ungoverned),
         "governed_gpu_processes_seen": len(governed), "census_samples": len(samples),
         "foreign_host_gpu_apps_max": foreign, "positive_control_logged": seen_ctl}
    evidence.record("SP-1.2", m, command=CMD, notes=f"ungoverned GPU processes: {sorted(ungoverned)}")
    assert m["scheduler_usage_log_present"] and seen_ctl and m["governed_gpu_processes_seen"] >= 1
    assert m["gpu_jobs_outside_scheduler_observed"] == 0, sorted(ungoverned)
