"""SP-10.2 (MASTER_SPEC Part 10.2): cross-domain consequences arise from reusable interactions over shared state
(couplings that can be removed for counterfactuals), not scripted ones. Workstream J.

Measured on generated civilisations (crucible.civ.gates.couplings_battery):

    reusable_couplings            executable coupling records of crucible.civ.couplings.CATALOGUE that are present in
                                  at least two structurally different profiles and carry only numeric parameters
                                  (no entity named anywhere in a coupling)
    couplings_removable_fraction  share of executable couplings whose mechanism's own trace (the events it records
                                  when it acts) is present in a control fork and absent after `civ.couplings.remove`
                                  at the same fork (same footprint: one operator command in each arm)
    scripted_consequence_paths    scripted modules (scheduled shocks) in any profile + one-off (non-periodic) jobs of
                                  the coupling engine + events of scripted modules in the battery's runs

Couplings that are recorded but not executable in this checkout (`pending`, e.g. education_skills while the
population's study and employment are not live) are listed in the evidence and are not counted as couplings.
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_10_2.py"


@pytest.mark.slow
@pytest.mark.timeout(3 * 3600)
def test_SP_10_2_reusable_removable_couplings():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.couplings_battery()
    m = {"reusable_couplings": r["reusable_couplings"],
         "couplings_removable_fraction": r["couplings_removable_fraction"],
         "scripted_consequence_paths": r["scripted_consequence_paths"],
         "executable_couplings": len(r["executable"]), "pending_couplings": len(r["pending"])}
    rows = [f"{c} ({a['profile']}): trace control {a['trace_control']} removed {a['trace_removed']}; downstream "
            f"control {a['downstream_control']} removed {a['downstream_removed']}" for c, a in r["arms"].items()]
    evidence.record("SP-10.2", m, command=CMD, notes=f"pending (not counted): {r['pending']}; uses {r['uses']}; "
                    f"scripted detail {r['scripted_detail']}; " + "; ".join(rows))
    assert all(a["proc_failed"] == 0 for a in r["arms"].values()), r["arms"]
    assert m["reusable_couplings"] >= 10, r["reusable"]
    assert m["couplings_removable_fraction"] == 1.0, {c: a for c, a in r["arms"].items() if not a["removable"]}
    assert m["scripted_consequence_paths"] == 0, r["scripted_detail"]
