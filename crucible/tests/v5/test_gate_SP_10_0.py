"""SP-10.0 (MASTER_SPEC Parts 10 / 10.4): standalone worlds are preserved for experiments and training is not
exclusive to one permanent civilisation; world forks, historical queries and independent replication are supported;
participants, organisations, relationships, products, technologies and institutions persist and evolve. Workstream J.

Measured on one store-backed civilisation (crucible.civ.gates.persistence, profile balanced, 8 simulated days):

    fork_divergence_supported          an unchanged fork reproduces its parent exactly (hash and digest), a fork with an
                                       intervention (a coupling removed) diverges, and the parent is untouched
    historical_query_kinds             kinds of crucible.civ.history queries whose answers were verified against an
                                       independent source (state at a past time vs the snapshot recorded then, an
                                       entity's timeline, per-day activity vs the record, a past balance from postings
                                       vs the rebuilt state, provenance back from an event)
    independent_replications           fresh processes that rebuild the same spec and seed and reach the same event
                                       hash and state digest after one simulated day
    entity_kinds_persisting_and_evolving  kinds of entity with entities present at day 1 and at the end and changed or
                                       added in between (state records compared, not events counted)

Also recorded: the environments of the standalone execution modes A and B (crucible.substrate.modes) and the number
of Tier 3 profiles - what training can draw from besides any one civilisation.
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_10_0.py"


@pytest.mark.slow
@pytest.mark.timeout(3 * 3600)
def test_SP_10_0_forks_history_replication_persistence():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    r = gates.persistence()
    m = {"fork_divergence_supported": bool(r["fork_divergence_supported"]),
         "historical_query_kinds": r["historical_query_kinds"],
         "independent_replications": r["independent_replications"],
         "entity_kinds_persisting_and_evolving": r["entity_kinds_persisting_and_evolving"],
         "standalone_mode_A_environments": r["standalone_environments"].get("A", 0),
         "standalone_mode_B_environments": r["standalone_environments"].get("B", 0),
         "tier3_profiles": r["tier3_profiles"]}
    evidence.record("SP-10.0", m, command=CMD, notes=f"run {r['params']}; forks {r['forks']}; history "
                    f"{ {k: v for k, v in r['history'].items() if k != 'historical_query_kinds'} }; replications "
                    f"{r['replications']} vs day-1 {r['day1']}; entity kinds {r['entity_kinds']}")
    assert m["fork_divergence_supported"] is True, r["forks"]
    assert m["historical_query_kinds"] >= 3, r["history"]
    assert m["independent_replications"] >= 3, r["replications"]
    assert m["entity_kinds_persisting_and_evolving"] >= 6, r["entity_kinds"]
    assert m["standalone_mode_A_environments"] > 0 and m["tier3_profiles"] >= 10
