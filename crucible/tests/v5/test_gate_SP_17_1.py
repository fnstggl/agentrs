"""SP-17.1 (MASTER_SPEC Part 17 preamble, Clarification 3): every world gate's capability is re-demonstrated inside
an integrated Tier 3 civilisation with the shared population (gates B, C, D, E, G, H, J, K, L, Q), not only in
purpose-built partial worlds. Workstream J.

crucible.civ.redemo runs each gate's own measurement code on worlds built by crucible.civ.generate and applies the
manifest's current criteria of every requirement of the gate; a gate counts only when every one of its requirements
was measured in a civilisation and meets its criteria (a requirement without a civilisation measurement is reported
`not_measured` with the reason and fails its gate):

    world_gates_redemonstrated_in_civilization   gates (of the ten) that hold entirely in civilisations
    partial_world_only_gates                     the others (their capability is shown only in partial worlds)

J and L: the counterfactual creation experiment of Gates J/L (LABELLED SCRIPTED creators) in a civilisation;
B, C, K: one autonomous civilisation run measured with Gates B/C's population and audience functions and the
commerce revenue/integrity functions; D, E, G, H, Q: no civilisation measurement in this checkout (reasons recorded).
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "CRUCIBLE_CIV_WORKERS=4 PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_17_1.py"


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_17_1_world_gates_in_a_civilisation():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.redemonstration()
    m = {"world_gates_redemonstrated_in_civilization": r["world_gates_redemonstrated_in_civilization"],
         "partial_world_only_gates": r["partial_world_only_gates"],
         **{f"gate_{g}_held": bool(v["held"]) for g, v in r["gates"].items()},
         **{f"gate_{g}_requirements_met": sum(1 for x in v["requirements"].values() if x["status"] == "met")
            for g, v in r["gates"].items()}}
    rows = []
    for g, v in r["gates"].items():
        for rid, x in v["requirements"].items():
            rows.append(f"{rid}: {x['status']} {x.get('metrics') or x.get('reason')} {x.get('failed') or ''}")
    evidence.record("SP-17.1", m, command=CMD, notes="; ".join(rows) + f"; creation runs {r['creation_runs']}; "
                    f"autonomous run {r['autonomous_run']}")
    assert all(x["proc_failed"] == 0 for x in r["creation_runs"].values()), r["creation_runs"]
    assert m["world_gates_redemonstrated_in_civilization"] == 10 and m["partial_world_only_gates"] == 0, \
        {g: [k for k, x in v["requirements"].items() if x["status"] != "met"] for g, v in r["gates"].items()}
