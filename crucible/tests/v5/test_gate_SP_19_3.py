"""SP-19.3 (MASTER_SPEC Part 19, Phase 3): the seven vertical slices executed and recorded, with an
architecture-independent learner in place of CAM (Clarification 1). Workstream J.

crucible.civ.slices.run_all executes them end to end on generated worlds and records one traced example each:

    1 discover_consume   a population individual discovers a content item through a platform channel and consumes it
                         (Gate C's own measurement: the recorded impression and consumption batches)
    2 agent_software     an agent's deployed software (a LABELLED SCRIPTED reference inventor) is used by individuals of
                         the population for their own work
    3 social_purchase    word of mouth from a contact's purchase leads the learner of it to buy that listing with its
                         own money
    4 neural_adapts      a population policy group's network parameters change by learning from experience
    5 learner_world      crucible.experience.ReferenceLearner acts in a Tier 3 civilisation through the public
                         experience interface; its actions are executed and change world state
    6 learner_obs_only   in the default observation-only mode the environment's reward is 0.0 at every step and the
                         learner still updates its parameters from its own signal
    7 causal_mechanism   a generated Tier 1 world's causal mechanism is validated by controlled interventions on copies
                         of the world (crucible.rapid.claims)

    vertical_slices_executed == 7
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_19_3.py"


@pytest.mark.slow
@pytest.mark.timeout(3 * 3600)
def test_SP_19_3_seven_vertical_slices():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates, slices
    r = gates.vertical_slices()
    m = {"vertical_slices_executed": r["vertical_slices_executed"],
         **{f"slice_{k}": bool(r.get(k, {}).get("ok")) for k in slices.SLICES}}
    evidence.record("SP-19.3", m, command=CMD, notes="; ".join(f"{k}: {r.get(k)}" for k in slices.SLICES) +
                    f"; civ run {r.get('_run')}")
    assert r["_run"]["proc_failed"] == 0 and not r["_run"]["invariants"], r["_run"]
    assert m["vertical_slices_executed"] == 7, {k: r.get(k) for k in slices.SLICES if not r.get(k, {}).get("ok")}
