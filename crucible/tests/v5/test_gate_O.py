"""Gate O (MASTER_SPEC Part 17): integrated civilisation. Workstream J.

Every metric is measured on a generated civilisation (crucible.civ.generate) with the one population, on data flow
(event provenance), not on which modules are loaded (crucible.civ.integration, crucible.civ.gates):

    O.1   subsystems_in_one_world      Mode C subsystems present AND active AND connected to another subsystem by
                                       provenance in one run (raw present/active counts are recorded too, and per
                                       subsystem whether the population's own decisions carry activity in it)
    O.2   duplicate_shared_facts       data-level scan of the world's tables for the same kind of fact in two places
          identity_consistency_violations  dangling principals/accounts, population principals without a row
    O.3   ledger_invariant_violations  bank invariants plus every balance recomputed from its postings
    O.4   cross_system_event_types     event types with a provenance parent in another subsystem
    O.5   cross_domain_effect_types    Part 10.2 categories whose effect is confirmed by counterfactual intervention
                                       against two placebo arms across seeds (crucible.civ.experiments)
    O.6   multi_stage_chains / min_chain_stages  chains mined from provenance whose stages (>= 3 subsystems) are
                                       confirmed by removing their root (crucible.civ.gates.chain_results)
    O.7   structurally_distinct_profiles  distinct discrete fingerprints of BUILT worlds of the profiles
    O.n1  modules_without_shared_state modules (inert ones included) with no provenance edge to another subsystem

The heavy tests (O.5, O.6) run counterfactual experiments; give them CPUs with
`python -m crucible.scale.slots --cpu 4 -- env CRUCIBLE_CIV_WORKERS=4 python -m pytest ...`.
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_O.py"


@pytest.fixture(scope="module")
def run():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    return gates.integration()


def _note(run):
    pend = "; ".join(f"{p['module']} ({p['subsystem']}): {p['waits_for']}" for p in run["pending"]) or "none"
    return f"civilisation {run['params']} ({run['events']} events, {len(run['modules'])} modules); pending: {pend}"


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_1_all_subsystems_in_one_world(run):
    s = run["subsystems"]
    m = {"subsystems_in_one_world": s["strict"], "subsystems_present": s["present"], "subsystems_active": s["active"],
         "not_active_or_not_connected": ",".join(s["missing"]) or "none",
         "subsystems_with_population_activity": len(s["with_population_activity"]),
         "present_without_population_activity": ",".join(s["without_population_activity"]) or "none"}
    evidence.record("O.1", m, command=CMD, notes=_note(run) + "; per subsystem: " + "; ".join(
        f"{k}: present={v['present']} events={v['events']} cross_edges={v['cross_edges']} "
        f"population_events={v['population_events']}" for k, v in s["subsystems"].items()))
    assert s["strict"] >= 17, s["missing"]


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_2_shared_state_and_identities(run):
    d, i = run["duplicates"], run["identity"]
    m = {"duplicate_shared_facts": d["duplicate_shared_facts"],
         "identity_consistency_violations": i["identity_consistency_violations"],
         "persons_outside_population": i["persons_outside_population"]}
    kc = d.get("kernel_check", {})
    evidence.record("O.2", m, command=CMD, notes=_note(run) + f"; duplicate kinds: {d.get('duplicate_kinds')}; "
                    f"kinds held in several tables: {d['kinds_in_several_tables']}; facts duplicated: "
                    f"{d['duplicate_facts']}; kernel check (crucible.substrate.facts): second copies holding facts "
                    f"{[(x['table'], x['rows'], x['owner'], x['owner_rows']) for x in kc.get('known_duplicates_holding_facts', [])]}"
                    f", problems {kc.get('problems')}; identity detail: {i['detail'][:5]}")
    assert d["duplicate_shared_facts"] == 0, (d.get("duplicate_kinds"), d["kinds_in_several_tables"], kc)
    assert i["identity_consistency_violations"] == 0, i["detail"]


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_3_consistent_financial_accounting(run):
    led = run["ledger"]
    m = {"ledger_invariant_violations": led["ledger_invariant_violations"],
         "population_undeposited_cents": int(led["population_undeposited_cents"])}
    evidence.record("O.3", m, command=CMD, notes=_note(run) + f"; money supply {led['money_supply']}; detail "
                    f"{led['detail'][:5]}")
    assert led["ledger_invariant_violations"] == 0, led["detail"]


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_4_cross_system_events(run):
    c = run["cross"]
    m = {"cross_system_event_types": c["cross_system_event_types"],
         "subsystem_pairs": len(c["subsystem_pairs"])}
    evidence.record("O.4", m, command=CMD, notes=_note(run) + f"; types {list(c['types'])[:40]}; pairs "
                    f"{dict(list(c['subsystem_pairs'].items())[:30])}")
    assert c["cross_system_event_types"] >= 10


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_O_5_cross_domain_effect_types():
    from crucible.civ import gates
    r = gates.effects()
    rows = []
    for n, e in r["experiments"].items():
        arm = next(iter(e["effects"]))
        p = e["effects"][arm][e["primary"]]
        rows.append(f"{n} (cat {e['category']}, {arm}, {e['primary']}): effect {p['effect_mean']:.4g} "
                    f"CI {[round(x, 4) for x in p['ci95']]} noise {p['reshuffle_noise_rms']:.4g} "
                    f"confirmed={p['confirmed']}")
    m = {"cross_domain_effect_types": r["cross_domain_effect_types"],
         "confirmed_categories": ",".join(map(str, r["confirmed_categories"])) or "none",
         "pending_categories": ",".join(map(str, r["pending_categories"])) or "none",
         "seeds": len(gates.EFFECT_SEEDS), "failed_process_runs_in_arms": r.get("proc_failed")}
    evidence.record("O.5", m, command=CMD, notes="; ".join(rows))
    assert r["cross_domain_effect_types"] >= 6, rows


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_O_6_multi_stage_chains():
    from crucible.civ import gates
    r = gates.chain_results()
    m = {"multi_stage_chains": r["multi_stage_chains"], "min_chain_stages": r["min_chain_stages"],
         "chains_evaluated": len(r["chains"]), "failed_process_runs_in_arms": r.get("proc_failed")}
    evidence.record("O.6", m, command=CMD, notes="; ".join(
        f"{c['experiment']}/{c['arm']} ({c['class']}, removed {c['removed']}): confirmed stages "
        f"{c['confirmed_stages']}, mined with provenance in {c['mined_in_reference_arm']} reference runs "
        f"(max {c['max_mined_stages']} stages); scripted root" for c in r["chains"]))
    assert r["multi_stage_chains"] >= 3 and r["min_chain_stages"] >= 3, r["chains"]


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_7_structurally_different_profiles():
    from crucible.civ import gates
    r = gates.structure()
    m = {"structurally_distinct_profiles": r["structurally_distinct_profiles"], "profiles": len(r["profiles"]),
         "identical_pairs": len(r["pairs_identical"]),
         "min_structural_distance": min(r["distances"].values()) if r["distances"] else 0.0}
    evidence.record("O.7", m, command=CMD, notes=f"fingerprints of built worlds: {r['fingerprints']}")
    assert r["structurally_distinct_profiles"] >= 3


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_O_n1_no_disconnected_mini_environments(run):
    s = run["shared"]
    m = {"modules_without_shared_state": s["modules_without_shared_state"], "modules": s["modules"],
         "inert_modules": len(s["inert"])}
    evidence.record("O.n1", m, command=CMD, notes=_note(run) + f"; without shared state: {s['without']}; inert: "
                    f"{s['inert']}")
    assert s["modules_without_shared_state"] == 0, s["without"]
