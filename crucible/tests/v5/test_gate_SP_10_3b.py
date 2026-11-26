"""SP-10.3b (MASTER_SPEC Part 10.3): scripted narrative events do not count toward multi-order chains; chains occur
in the integrated Tier 3 civilisation and replicate across seeds; scripted reference policies are labelled as such.
Workstream J.

From the chain experiments of crucible.civ.gates.chain_results (the same runs as O.6 / SP-10.3: civilisations of
profiles balanced and research_intensive, three seeds each, roots by LABELLED SCRIPTED reference agents):

    scripted_chain_stages          stages after the root of the provenance-mined chains in which every event belongs
                                   to a transaction of a scripted actor or a scheduled narrative module
                                   (crucible.civ.chains.stage_scripted) - consequences a script produced
    chains_replicated_across_seeds chains confirmed by counterfactual removal (>= 3 stages, sign-consistent in every
                                   seed) that are also mined from provenance in every seed's reference arm
    unlabelled_scripted_results    mined agent-rooted chains whose root actor is not labelled scripted
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "CRUCIBLE_CIV_WORKERS=3 PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_10_3b.py"


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_10_3b_no_scripted_stages_replicated_labelled():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.chain_results()
    m = {"scripted_chain_stages": r["scripted_chain_stages"],
         "chains_replicated_across_seeds": r["chains_replicated_across_seeds"],
         "unlabelled_scripted_results": r["unlabelled_scripted_results"], "chains_confirmed": r["multi_stage_chains"]}
    evidence.record("SP-10.3b", m, command=CMD, notes="; ".join(
        f"{c['class']} ({c['experiment']}/{c['arm']}): confirmed stages {c['confirmed_stages']}, mined in "
        f"{c['seeds_with_mined_chain']}/{c['seeds']} seeds, scripted stages after the root "
        f"{c['scripted_stages_after_root']}, unlabelled scripted roots {c['unlabelled_scripted_roots']}"
        for c in r["chains"]))
    assert m["scripted_chain_stages"] == 0 and m["unlabelled_scripted_results"] == 0, r["chains"]
    assert m["chains_replicated_across_seeds"] >= 3, r["chains"]
