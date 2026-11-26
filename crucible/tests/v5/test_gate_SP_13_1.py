"""SP-13.1 (MASTER_SPEC Part 13.1): compatible decisions are grouped for batched inference across every neural and
semantic participant class: class 2 (shared adaptive policies), class 3 (strategic policy groups) and class 4
(semantic generalists). A class counts when, in the benchmark runs of the integrated world, its recorded inference
calls decided more than one participant per call on average."""
import pytest

from crucible.acceptance.v5 import evidence

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.1")


def test_SP_13_1_decisions_batched_in_every_learning_class():
    s = K.suite()
    tot = {c: {"calls": 0, "decisions": 0} for c in ("class2", "class3", "class4")}
    for r in s["runs"]:
        if r["world"] != "civ":
            continue
        for c in tot:
            b = r.get("batching", {}).get(c, {})
            tot[c]["calls"] += int(b.get("calls", 0))
            tot[c]["decisions"] += int(b.get("decisions", 0))
    batched = [c for c, v in tot.items() if v["calls"] > 0 and v["decisions"] / v["calls"] > 1.0]
    m = {"participant_classes_batched": len(batched), "classes": ",".join(batched)}
    for c, v in tot.items():
        m[f"{c}_calls"] = v["calls"]
        m[f"{c}_decisions"] = v["decisions"]
    svc = (K.backend_matrix().get("result") or {}).get("service") or {}
    for case, v in svc.get("cases", {}).items():      # the shared service grouping several call sites of one model
        m[f"service_{case}_calls_separate"] = int(v["calls_separate"])
        m[f"service_{case}_speedup"] = float(v["speedup_wall"])
    evidence.record("SP-13.1", m, command=CMD, notes=f"service benchmark {svc}")
    assert m["participant_classes_batched"] >= 3, tot
