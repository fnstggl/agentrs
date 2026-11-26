"""SP-0.10 (MASTER_SPEC Part 0 item 10): Crucible 4.0 throughput, memory consumption and world-generation cost are
measured on the 4.0 commit (c07098b, extracted with git archive, run from that tree's own source) and reported
against v5 (the development-profile benchmark of the same population size and days)."""
import os

import pytest

from crucible.acceptance.v5 import baseline, evidence
from crucible.scale import bench5

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = "PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --ram-mb 16000 -- python -m crucible.acceptance.v5 run --requirement SP-0.10"


def test_SP_0_10_v4_baseline_reported_against_v5():
    v4 = K.v4_rows()
    rows = v4.get("rows") or []
    src = {r["source"]["commit"] for r in rows}
    tree = rows[0]["source"]["tree"] if rows else ""
    q = {"throughput": all(r["sim_days_per_cpu_hour"] > 0 and r["transitions_per_s"] > 0 for r in rows),
         "memory": all(r["peak_ram_mb"] > 0 for r in rows),
         "world_generation_cost": all(r["generation"]["cpu_s"] > 0 for r in rows)}
    on_v4 = bool(rows) and src == {baseline.V4_COMMIT} and os.path.isfile(os.path.join(tree, "pyproject.toml"))
    s = K.suite()
    cmp = bench5.comparison_rows([r for r in s["runs"] if r["seed"] == 0], rows)
    m = {"baseline_quantities_recorded": sum(q.values()) if rows else 0, "baseline_measured_on_v4_commit": on_v4,
         "v4_rows": len(rows), "compared_with_v5_rows": len(cmp)}
    for r in rows:
        m[f"v4_bazaar_{r['consumers']}_sim_days_per_cpu_hour"] = round(float(r["sim_days_per_cpu_hour"]), 3)
        m[f"v4_bazaar_{r['consumers']}_peak_ram_mb"] = round(float(r["peak_ram_mb"]), 1)
        m[f"v4_bazaar_{r['consumers']}_generation_cpu_s"] = round(float(r["generation"]["cpu_s"]), 3)
    evidence.record("SP-0.10", m, command=CMD, notes=f"4.0 rows {rows}; against v5 {cmp}")
    assert m["baseline_quantities_recorded"] >= 3 and m["baseline_measured_on_v4_commit"]
    assert m["compared_with_v5_rows"] >= 1
