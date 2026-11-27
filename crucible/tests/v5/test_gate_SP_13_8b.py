"""SP-13.8b (MASTER_SPEC Part 13 preamble / 13.8): meaningful, diverse learner experience per unit of compute.

For every benchmark run (crucible.scale.bench5): events a learner can observe (public events), their diversity
(distinct public event kinds, distinct acting principals) and the individual decisions behind them, per CPU-second of
the world; compared with the 4.0 equivalent workload (public events per CPU-second of 4.0 bazaar)."""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import bench5

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 "
       "--timeout 172800 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.8b")


def test_SP_13_8b_experience_per_compute():
    s = K.suite()
    rows = {}
    for r in s["runs"]:
        e = r["experience"]
        rows.setdefault(f"{r['world']}_{r['profile']}", []).append(e)
    m = {}
    for k, es in sorted(rows.items()):
        m[f"{k}_public_events_per_cpu_s"] = round(sum(e["public_events_per_cpu_s"] for e in es) / len(es), 4)
        m[f"{k}_decisions_per_cpu_s"] = round(sum(e["decisions_per_cpu_s"] for e in es) / len(es), 4)
        m[f"{k}_distinct_public_kinds"] = min(e["distinct_public_kinds"] for e in es)
        m[f"{k}_acting_principals"] = min(e["acting_principals"] for e in es)
    cmp = bench5.comparison_rows([r for r in s["runs"] if r["seed"] == 0], K.v4_rows().get("rows") or [])
    for c in cmp:
        m[f"v4_{c['profile']}_public_events_per_cpu_s"] = float(c["v4_public_events_per_cpu_s"])
    civ = [k for k in rows if k.startswith("civ_")]
    m["experience_per_compute_reported"] = len(civ) == 3 and all(m[f"{k}_public_events_per_cpu_s"] > 0 for k in civ)
    evidence.record("SP-13.8b", m, command=CMD, notes=f"comparison {cmp}")
    assert m["experience_per_compute_reported"], m
