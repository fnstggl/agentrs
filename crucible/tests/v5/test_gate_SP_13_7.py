"""SP-13.7 (MASTER_SPEC Part 13.7): the resource scheduler enforces CPU workers, RAM, GPU memory, GPU time, disk usage,
inference concurrency and active world count.

Every dimension is attacked by a real job that tries to exceed its grant (crucible.scale.govprobe): busy processes
that widen their own CPU affinity, a job allocating past its RAM grant, a 1 GB GPU allocation under a 300 MB grant
(and a job that lifts its own allocator cap), a CUDA context held past its GPU-time grant, files written past a disk
grant (and one file larger than it), 6 threads in 2 processes contending for 2 inference tokens, and worlds created
beyond an active-world grant in two processes of one job. The scheduling rules are probed too: FIFO admission,
zombies, a detached grandchild, a self-declared budget, an oversized request, a nested ledger, a kill that must not
reach a process outside the job, backfilling behind a blocked head and deadline backfilling (a short job in the
head's share ends before the head's planned start; the head is not delayed).

scheduler_limits_enforced = dimensions whose attack was stopped or contained (by measurement of what the job used);
scheduler_limit_violations = attacks (dimensions and rules) that got through. The GPU probes need the GPU: run this
test through the scheduler with GPU memory, e.g.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --gpu-mb 1500 --ram-mb 8000 --label "M: SP-13.7" -- \\
        python -m crucible.acceptance.v5 run --requirement SP-13.7
"""
import os
import tempfile

import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import govprobe, slots

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1200)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --gpu-mb 1500 --ram-mb 8000 --timeout 1200 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.7")


def test_SP_13_7_resource_scheduler_enforces_seven_dimensions():
    base = tempfile.mkdtemp(prefix="sp137-")
    res = govprobe.run_all(base)
    dims = {k: res[k] for k in govprobe.DIMENSION_PROBES}
    rules = {k: res[k] for k in govprobe.RULE_PROBES}
    enforced = [k for k, v in dims.items() if v.get("contained")]
    failed = [k for k, v in {**dims, **rules}.items() if not v.get("contained")]
    m = {"scheduler_limits_enforced": len(enforced), "scheduler_limit_violations": len(failed),
         "dimensions_enforced": ",".join(enforced), "attacks_not_stopped": ",".join(failed) or "none",
         "rules_held": sum(1 for v in rules.values() if v.get("contained")), "rules_probed": len(rules)}
    for k, v in dims.items():
        m[f"{k}_limit"] = float(v.get("limit", 0))
        m[f"{k}_used"] = float(v.get("used") or 0)
    m["gpu_lifted_cap_used_mb"] = float(dims["gpu_mem"].get("lifted_cap_used") or 0)
    m["cpu_persistent_widening_wall_s"] = float(dims["cpu"].get("persistent_wall_s") or 0)
    m["disk_single_file_mb"] = float(dims["disk"].get("big_file_mb") or 0)
    m["inference_calls"] = int(dims["inference"].get("calls") or 0)
    m["worlds_refused"] = int(dims["worlds"].get("refused") or 0)
    hw = slots.hardware()
    m["gpu_visible"] = bool(hw["gpus"]) and os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
    evidence.record("SP-13.7", m, command=CMD,
                    notes="adversarial probes (crucible.scale.govprobe): "
                          + "; ".join(f"{k}: {'stopped' if v.get('contained') else 'GOT THROUGH'} "
                                      f"{({kk: vv for kk, vv in v.items() if not kk.startswith('record')})}"
                                      for k, v in {**dims, **rules}.items()))
    assert m["gpu_visible"], "the GPU dimensions can only be probed with the GPU visible (grant --gpu-mb)"
    assert len(enforced) == 7, f"not enforced: {[k for k in dims if k not in enforced]}: {failed}"
    assert not failed, f"attacks that got through: {failed}"
    # what the jobs used stayed within (or was stopped at) the grant, by measurement
    assert dims["ram"]["record"]["killed_by"] == "ram" and dims["gpu_time"]["record"]["killed_by"] == "gpu_time"
    assert dims["gpu_mem"]["used"] <= 300 and dims["inference"]["used"] <= 2 and dims["worlds"]["used"] <= 2
    assert dims["cpu"]["used"] <= dims["cpu"]["allowed_ratio"] and dims["disk"]["big_file_mb"] <= 20.5
    assert rules["foreign"]["sibling_alive"] and rules["escape"]["grandchild_alive_after"] is False
