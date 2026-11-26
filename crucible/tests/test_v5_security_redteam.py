"""Workstream O unit tests of the 5.0 red-team package itself: the catalogue imports and is well formed, the ratchet
entries name real probes, and gate_metrics counts outcomes as documented (errors and findings never count as held)."""
import fnmatch

from crucible.redteam.v5 import GATES, KNOWN_OPEN, MUST_HOLD, catalogue, gate_metrics, probes, ratchet
from crucible.redteam.v5.suite import FIXED


def test_catalogue_is_complete_and_well_formed():
    ps = probes()
    names = [p.name for p in ps]
    assert len(names) == len(set(names)) and len(ps) >= 50
    assert all(p.gates and set(p.gates) <= set(GATES) and p.owner and p.description for p in ps)
    per_gate = {g: sum(g in p.gates for p in ps) for g in GATES}
    assert per_gate["W.3"] >= 20 and per_gate["W.4"] >= 5 and min(per_gate.values()) >= 4, per_gate
    mods = {p.__class__.__module__.rsplit(".", 1)[1] for p in ps}
    assert {"substrate", "population", "social", "commerce", "innovation", "experience", "rapid", "sandbox",
            "network", "isolation", "crossworld", "private"} <= mods


def test_ratchet_entries_name_real_probes():
    names = set(catalogue())
    assert set(KNOWN_OPEN) <= names and set(MUST_HOLD) <= names and not set(MUST_HOLD) & set(KNOWN_OPEN)
    assert all(k["severity"] in ("critical", "high", "medium", "low") and k["owner"] for k in KNOWN_OPEN.values())
    for pat in FIXED.values():
        assert any(fnmatch.fnmatch(n, pat) for n in names), pat


def test_gate_metrics_counts_findings_and_errors():
    cat = {"a": {"gates": ("W.3", "W.4")}, "b": {"gates": ("W.3",)}, "c": {"gates": ("W.3",)},
           "v5.sandbox.escape_matrix": {"gates": ("W.1",)}}
    rep = {"probes": [{"name": "a", "result": "false_success", "severity": "high"},
                      {"name": "b", "result": "error", "severity": ""},
                      {"name": "c", "result": "held", "severity": ""},
                      {"name": "v5.sandbox.escape_matrix", "result": "held",
                       "evidence": {"sandbox_tests_passed": 31, "sandbox_escapes": 0}}]}
    m = gate_metrics(rep, cat)
    assert m["W.3"]["reward_hack_probes"] == 2 and m["W.3"]["open_critical_high_reward_hacks"] == 2
    assert m["W.4"]["open_metric_manipulations"] == 1 and m["W.1"]["sandbox_tests_passed"] == 31
    r = ratchet(rep)
    assert "a" in r["unexpected_findings"] and "b" in r["unexpected_findings"]
