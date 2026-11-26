"""SP-13.6 (MASTER_SPEC Part 13.6 / Phase 5): the ten large-population techniques investigated by measurement on the
in-tree mechanisms (crucible.scale.techniques); the estimated large-population mode validated against individual
simulation with documented limits (crucible.scale.approx); and no outcome without a recorded decision event (a
population world's state rebuilds exactly from its event log, and the estimate writes nothing into a world)."""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.scale import techniques

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 2000 --ram-mb 64000 --disk-mb 12000 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-13.6")


def test_SP_13_6_techniques_approximation_and_no_fabricated_outcomes():
    s = K.suite()
    a = K.approximation()["result"]
    civ = [r for r in s["runs"] if r["world"] == "civ"]
    cpu_ind = (sum(r["cpu_s"] for r in civ) / max(1.0, sum(r["individuals"] * r["sim_days"] for r in civ))) if civ else None
    inv = techniques.investigate(civ, a, cpu_ind)
    done = [t for t in techniques.TECHNIQUES if inv.get(t, {}).get("measured")]
    validated = int(bool(a) and len([q for q in K.approx.QUANTITIES if q in a["errors"]]) >= 7 and bool(a["limits"]))
    rp = techniques.replay_exact()
    m = {"techniques_investigated": len(done), "approximate_modes_validated": validated,
         "outcomes_without_decision_events": int(rp["outcomes_without_decision_events"]),
         "replay_digest_equal": bool(rp["replay_digest_equal"]), "events_added_by_estimate": int(rp["events_added_by_estimate"]),
         "not_measured": ",".join(t for t in techniques.TECHNIQUES if t not in done) or "none"}
    for t in done:
        for k, v in inv[t].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                m[f"{t}_{k}"] = float(v)
    evidence.record("SP-13.6", m, command=CMD, labels={"estimated": True},
                    notes=f"investigations {inv}; approximation limits {a.get('limits')}")
    assert m["techniques_investigated"] >= 10, m["not_measured"]
    assert m["approximate_modes_validated"] >= 1 and m["outcomes_without_decision_events"] == 0
