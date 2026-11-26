"""DOM decision-bottleneck fields of decision traces (audit 'h' Decision traces finding; plan 2026-11-26 §1
Experiments "Traces: add intent / evaluation / r_psi fields"): intent I_t, the evaluation of the executed action,
r_psi with components, later credit and rejected candidates are patched onto traces (recorded in
trace.reasoning["dom"], schema unchanged) and survive lifetime partitioning for the dashboard."""
from __future__ import annotations

from superlearner.cam_unified.interfaces import DecisionTrace
from superlearner.cam_unified.observe.trace import (DOM_FIELDS, TraceRecorder, dom_fields, load_partition,
                                                    partition_lifetimes)


def _tr(i, env):
    return DecisionTrace(decision_id=f"d{i}", event_id=f"e{i}", env_id=env, policy_version=0, t_sim=float(i),
                         objective="o", reasoning={"iters": 2})


def test_dom_fields_are_patched_and_partitioned(tmp_path):
    root = str(tmp_path / "traces")
    r = TraceRecorder(root, name="a")
    for i, env in enumerate(["w0", "w0", "w1"]):
        r.record(_tr(i, env))
    r.set_field("intent", {"norm": 1.5, "id": 3}, event_id="e0")
    r.set_field("action_evaluation", {"q": 0.2, "sd": 0.1}, event_id="e0", env_id="w0")
    r.set_field("r_psi", {"total": 0.03, "ig": 0.02, "cost": -0.01}, decision_id="d2")
    r.set_field("credit", {"rudder": 0.7}, event_id="e1")
    r.set_field("candidates_rejected", [{"tool": "noop", "q": -1.0}], event_id="e1")
    r.close()
    ts = {t.decision_id: t for t in r.load()}
    assert dom_fields(ts["d0"]) == {"intent": {"norm": 1.5, "id": 3}, "action_evaluation": {"q": 0.2, "sd": 0.1}}
    assert ts["d0"].reasoning["iters"] == 2                         # existing reasoning content kept
    assert dom_fields(ts["d2"])["r_psi"]["total"] == 0.03 and set(dom_fields(ts["d1"])) == {"credit",
                                                                                           "candidates_rejected"}
    parts = partition_lifetimes(root, str(tmp_path / "parts"))
    unt = str(tmp_path / "parts" / "_untagged.jsonl")
    w0 = {t.decision_id: t for t in load_partition(parts["w0"], unt)}
    assert dom_fields(w0["d0"])["action_evaluation"]["q"] == 0.2 and dom_fields(w0["d1"])["credit"]["rudder"] == 0.7
    assert set(DOM_FIELDS) == {"intent", "action_evaluation", "r_psi", "credit", "candidates_rejected"}
    assert dom_fields(_tr(9, "w9")) == {}
