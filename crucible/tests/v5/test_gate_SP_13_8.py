"""SP-13.8 (MASTER_SPEC Part 13.8): the performance report - the twelve listed quantities for every population profile
of the benchmark suite (crucible.scale.bench5, 3 seeds x 30 simulated days, the integrated civilisation world and
the population-only world) - and a comparison with Crucible 4.0 on equivalent workloads: the 4.0 `bazaar` family
(4.0's marketplace with aggregate consumers) with as many consumers as the profile has individuals and the same
simulated days, run from the 4.0 source tree (c07098b, git archive) under the same governor.

performance_quantities_reported = the quantities present (measured, not null) in every profile's report, counting
approximation error from the estimated-mode validation; v4_comparison_rows = equivalent-workload rows.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 -- \\
        python -m crucible.acceptance.v5 run --requirement SP-13.8
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import bench5

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.8")
PER_RUN = ("transitions_per_s", "decisions_per_s", "sim_days_per_wall_hour", "cpu_utilisation", "peak_ram_mb",
           "peak_vram_mb", "model_parameter_counts", "inference_token_counts", "event_storage_growth",
           "snapshot_size", "world_generation_cost")


def test_SP_13_8_performance_report_and_v4_comparison():
    s = K.suite()
    runs = s["runs"]
    a = K.approximation()["result"]
    present = {}
    for q in PER_RUN:
        ok = []
        for prof in bench5.PROFILES:
            rs = [r for r in runs if r["world"] == "civ" and r["profile"] == prof]
            ok.append(bool(rs) and all(r.get(q) is not None for r in rs))
        present[q] = all(ok)
    present["approximation_error"] = bool(a and a.get("errors"))
    v4 = K.v4_rows()
    rows = bench5.comparison_rows([r for r in runs if r["seed"] == 0], v4.get("rows") or [])
    m = {"performance_quantities_reported": sum(present.values()), "v4_comparison_rows": len(rows),
         "missing_quantities": ",".join(q for q, v in present.items() if not v) or "none", "runs": len(runs)}
    for r in rows:
        key = f"{r['world']}_{r['profile']}"
        m[f"{key}_v5_sim_days_per_cpu_hour"] = float(r["v5_sim_days_per_cpu_hour"])
        m[f"{key}_v4_sim_days_per_cpu_hour"] = float(r["v4_sim_days_per_cpu_hour"])
        m[f"{key}_v5_peak_ram_mb"] = float(r["v5_peak_ram_mb"])
        m[f"{key}_v4_peak_ram_mb"] = float(r["v4_peak_ram_mb"])
    evidence.record("SP-13.8", m, command=CMD,
                    notes=f"aggregate {s['aggregate']}; comparison rows {rows}; 4.0 runs {v4.get('rows')}")
    assert m["performance_quantities_reported"] >= 12, present
    assert m["v4_comparison_rows"] >= 1 and all(r["v4_sim_days_per_cpu_hour"] > 0 for r in rows)
