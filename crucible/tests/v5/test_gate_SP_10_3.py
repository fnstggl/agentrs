"""SP-10.3 (MASTER_SPEC Part 10.3): multi-order chains with an agent-created artifact, with social activity and
with technical or scientific activity, each confirmed by counterfactual removal. Workstream J.

Chains are mined from event provenance in the reference arm (crucible.civ.chains.mine) and confirmed by removing
their root at the fork across seeds against two placebo arms (crucible.civ.gates.chain_results):

    agent_artifact  a scripted reference agent's web shop (repository, deployment) -> population visits ->
                    purchases -> compute its deployment serves on its provider -> datacenter energy
    social          the same agent's forum posts -> population visits -> purchases (-> compute, energy)
    technical       a scripted reference researcher's paper -> independent replication -> producers' R&D trial ->
                    process settings, yields and energy per good unit -> goods

The roots are scripted reference policies and are labelled as such in the evidence (Part 10.3); every later stage
is the autonomous world's (population, organisational minds, labs, physics).
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "CRUCIBLE_CIV_WORKERS=4 PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_10_3.py"


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_10_3_chain_classes_confirmed_by_removal():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.chain_results()
    m = {"chains_with_agent_artifact": r["chains_with_agent_artifact"],
         "chains_with_social_activity": r["chains_with_social_activity"],
         "chains_with_technical_activity": r["chains_with_technical_activity"],
         "scripted_roots": True}
    evidence.record("SP-10.3", m, command=CMD, notes="; ".join(
        f"{c['class']}: {c['experiment']}/{c['arm']} removed {c['removed']}; confirmed stages {c['confirmed_stages']};"
        f" effects {c['effects']}" for c in r["chains"]))
    assert m["chains_with_agent_artifact"] >= 1 and m["chains_with_social_activity"] >= 1 and \
        m["chains_with_technical_activity"] >= 1, r["chains"]
