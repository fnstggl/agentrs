"""Gate Q: civilisation-wide self-play (MASTER_SPEC Part 12, Part 17 Gate Q; requirements Q.1-Q.8, Q.n1).

One campaign (`tests/v5/_selfplay_kit.campaign`, `crucible.multiagent.campaign`): neural tool-level policy groups
pre-trained by behaviour cloning on SCRIPTED demonstrations in CPU training worlds, then trained by self-play in a
Tier 3 civilisation (`crucible.civ`, profile balanced, development population) in observation-only mode through
`crucible.experience` (`run_learners`), each identity computing its own training signal from its own observations:

    alpha  3 identities, one shared REINFORCE policy (shared training system), drive: profit
    beta   2 identities, one PPO policy, drive: profit (competes with alpha)
    gamma  1 identity, A2C, drive: reach            delta  1 identity, A2C, drive: adoption
    alpha_hist  a frozen checkpoint of alpha admitted at half-time (historical)
    2 semantic generalists (Qwen3-0.6B, class 4), and the civilisation's own population (classes 1-3, firms)

Each identity is admitted with its controller (alpha's three are one party). Outcomes are the evaluator's
(`crucible.commerce.integrity`, ledger); cooperation, competition and consequences are read from the world's state
and event log (`crucible.multiagent.analysis`). Held-out evaluation: alpha's final and initial versions, frozen,
against a held-out GRU/PPO family (never in the training civilisation) on sealed seeds. Numeric facility-only
self-play (`crucible.selfplay`) is classified and refused (Q.n1). A failure to build Tier 3 fails the tests."""
import json
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.multiagent import analysis as AN
from crucible.multiagent import campaign as C
from crucible.multiagent import integrity as I
from crucible.multiagent import league as L
from crucible.multiagent.harness import is_civilisation, population_activity

from _selfplay_kit import PROFILE, campaign

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate Q"
LEARNING = ("alpha", "beta", "gamma", "delta")
LABELS = {"learned": True, "scripted_demonstrations_for_pretraining": True}


@pytest.fixture(scope="module")
def q(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("gate_q"))
    run = campaign(root, "full")
    m = run["match"]
    ctrl = {p: mm.controller for p, mm in m.members.items()}
    run["ctrl"] = ctrl
    run["evidence"] = C.learning_evidence(run)
    run["inter"] = AN.interactions(m.world, ctrl, m.t0)
    run["cons"] = AN.consequences(m.world, ctrl, m.seq0)
    run["indep"] = I.independence(m.groups, m.agents)
    run["outcomes"] = m.finish()
    reg = run["registry"]
    run["heldout_eval"] = L.heldout_evaluation(
        reg, {"final": reg.vid(run["groups"]["alpha"]), "initial": reg.vid(run["initial"]["alpha"])},
        [reg.vid(run["heldout"])], world=f"civ:{PROFILE}", seeds=L.SEALED_SEEDS[:2], hours=12)
    run["battery"] = I.collusion_battery(m.world, groups_for_hidden_state=m.groups, clean_party_of=ctrl)
    yield run
    m.close()


def _acting_groups(run) -> dict[str, int]:
    m = run["match"]
    out = {}
    for p, a in m.agents.items():
        g = getattr(a, "group", None)
        if g is None:
            continue
        n = sum(1 for h in a.history if h.get("ok") and not h.get("read_only"))
        out[g.gid] = out.get(g.gid, 0) + n
    return out


def test_Q_1_multiple_learned_policies_in_a_shared_civilisation(q):
    m = q["match"]
    acting = _acting_groups(q)
    ev = q["evidence"]
    learned = [g for g in LEARNING if acting.get(g, 0) >= 1 and ev[g]["updates_in_run"] >= 1]
    assert is_civilisation(m.world)
    record("Q.1", {"learned_policies_in_civilization": len(learned),
                   "world_family": m.world.spec.family, "population_individuals": int(
                       sum(v[0] for v in population_activity(m.world, 0, m.world.now).values())),
                   "state_changing_actions_by_group": json.dumps(acting), "identities": len(m.agents),
                   "simulated_hours": int((m.world.now - m.t0) / 3600)},
           command=CMD, labels=LABELS,
           notes="learned policies = neural policy groups updated by their own training signal in the civilisation "
                 "whose identities carried out accepted state-changing tool actions there")
    assert len(learned) >= 2


def test_Q_2_independently_updated_competing_policy_groups(q):
    m = q["match"]
    ev, comp = q["evidence"], q["inter"]["competitive"]
    party_of_group = {g: f"ctrl:{g}" for g in LEARNING}
    competing = {g for g in LEARNING for r in comp if party_of_group[g] in r["parties"]
                 and any(x != party_of_group[g] and x in party_of_group.values() for x in r["parties"])}
    chains = {g: q["registry"].lineage(q["registry"].vid(m.groups[g])) for g in LEARNING}
    disjoint = all(not set(chains[a]) & set(chains[b]) for a in LEARNING for b in LEARNING if a < b)
    indep = [g for g in LEARNING if ev[g]["updates_in_run"] >= 1 and g in competing]
    record("Q.2", {"independent_policy_groups": len(indep) if disjoint and not q["indep"]["violations"] else 0,
                   "groups_updated": sum(1 for g in LEARNING if ev[g]["updates_in_run"] >= 1),
                   "groups_in_competition": len(competing), "shared_object_violations": len(q["indep"]["violations"]),
                   "hash_chains_disjoint": disjoint,
                   "competition_kinds": json.dumps(sorted({r["kind"] for r in comp}))},
           command=CMD, labels=LABELS,
           notes="independent: own parameters, optimiser, generators and buffer (storage-pointer audit), disjoint "
                 "version chains, updated in the civilisation, and in at least one competitive interaction "
                 "(same market, scarce compute, contested customer, competing technology) with another learning "
                 "group's party, read from world state")
    assert disjoint and not q["indep"]["violations"]
    assert len(indep) >= 2


def test_Q_3_cooperation_and_competition(q):
    co = [r for r in q["inter"]["cooperative"] if r["both_learners"]]
    cm = [r for r in q["inter"]["competitive"] if r["both_learners"]]
    record("Q.3", {"cooperative_interactions": len(co), "competitive_interactions": len(cm),
                   "cooperative_kinds": json.dumps(dict(sorted({r["kind"]: sum(1 for x in co if x["kind"] == r["kind"])
                                                                for r in co}.items()))),
                   "competitive_kinds": json.dumps(dict(sorted({r["kind"]: sum(1 for x in cm if x["kind"] == r["kind"])
                                                                for r in cm}.items()))),
                   "cooperation_with_world_parties": sum(1 for r in q["inter"]["cooperative"]
                                                         if not r["both_learners"]),
                   "shared_reward_objectives": 0},
           command=CMD, labels=LABELS,
           notes="between parties of different controllers that are both admitted agents (learners, the "
                 "historical checkpoint or semantic generalists); no participant has a shared or team reward "
                 "(see SP-12.3b); cooperation exists only as world actions (negotiation, agreements, membership, "
                 "contributions, dependencies, grants, trade, service calls, shared ventures)")
    assert len(co) >= 1 and len(cm) >= 1


def test_Q_4_shared_world_consequences(q):
    c = q["cons"]
    record("Q.4", {"shared_world_consequence_events": c["events_visible_to_other_parties"],
                   "response_chains": len(c["response_chains"]),
                   "party_pairs": len(c["by_pair"]), "examples": json.dumps(c["examples"][:6])},
           command=CMD, labels=LABELS,
           notes="events caused (directly or through scheduled processes) by one party's command and visible to an "
                 "identity of another party; response chains: a later command of another party naming an id the "
                 "first party's action brought into being")
    assert c["events_visible_to_other_parties"] >= 1


def test_Q_5_policy_versioning(q):
    reg, m = q["registry"], q["match"]
    v = [x for x in reg.versions.values() if x["group"] in LEARNING and x["origin"] == "update"]
    ok_chain = all(x["parent"] in reg.versions and x["provenance"]["decisions"] > 0 for x in v)
    final = {g: reg.vid(m.groups[g]) for g in LEARNING}
    reload_ok = all(reg.load(final[g], f"{g}_reload").hash == m.groups[g].hash for g in LEARNING)
    record("Q.5", {"policy_versions_recorded": len(v), "versions_with_provenance": sum(1 for x in v if x["provenance"]),
                   "parent_chain_complete": ok_chain, "final_checkpoints_reload_to_hash": reload_ok,
                   "versions_total_including_bc_and_frozen": len(reg.versions)},
           command=CMD, labels=LABELS)
    assert ok_chain and reload_ok
    assert len(v) >= 4


def test_Q_6_evaluation_against_unfamiliar_opponents(q):
    ho = q["heldout_eval"]
    reg = q["registry"]
    fam = reg.versions[reg.vid(q["heldout"])]
    record("Q.6", {"heldout_opponent_evaluations": ho["matches"], "sealed_seeds": json.dumps(ho["seeds"]),
                   "heldout_family": f"{fam['arch']}+{fam['algo']}",
                   "training_families": json.dumps(sorted({f"{reg.versions[reg.vid(q['groups'][g])]['arch']}+"
                                                           f"{q['groups'][g].algo}" for g in LEARNING})),
                   "final_score_mean": ho["summary"]["final"]["score_mean"],
                   "initial_score_mean": ho["summary"]["initial"]["score_mean"],
                   "final_minus_initial_score": ho["paired"]["diff_mean"],
                   "final_minus_initial_rejected": ho["paired"]["rejected_diff_mean"]},
           command=CMD, labels=LABELS,
           notes="frozen candidates vs frozen held-out GRU/PPO policies trained outside the league, in civilisations "
                 "on sealed seeds no training run used; scores are evaluator outcomes (cash change + independent "
                 "revenue). The difference final - initial is reported, not required to be positive.")
    assert ho["matches"] >= 1
    assert fam["arch"] == "gru" and all(q["groups"][g].arch != "gru" for g in LEARNING)


def test_Q_7_evidence_of_neural_learning(q):
    ev = q["evidence"]
    upd = sum(ev[g]["updates_in_run"] for g in LEARNING)
    changed = all(ev[g]["weight_hash_changed"] for g in LEARNING)
    ctl = [ev[g]["control_hash_changed"] for g in LEARNING]
    record("Q.7", {"weight_updates": upd, "weight_hash_changed": changed,
                   "no_signal_controls_changed": sum(1 for c in ctl if c),
                   "behaviour_tv_mean": float(np.mean([ev[g]["behaviour_tv_mean"] for g in LEARNING])),
                   "control_behaviour_tv_mean": float(np.mean([ev[g]["control_behaviour_tv_mean"] for g in LEARNING])),
                   "early_late": json.dumps({g: {"early": ev[g]["early"], "late": ev[g]["late"]} for g in LEARNING})},
           command=CMD, labels=LABELS,
           notes="updates = parameter-changing reinforcement updates in the civilisation from each group's own "
                 "observation-derived signal (behaviour-cloning pre-training not counted); each group's no-signal "
                 "twin received the same batches with signal 0 and must not change; behaviour change = mean total "
                 "variation between the initial and final tool distributions on recorded situations of the run")
    assert changed and not any(ctl)
    assert upd >= 100
    assert all(ev[g]["behaviour_tv_mean"] > ev[g]["control_behaviour_tv_mean"] for g in LEARNING)


def test_Q_8_collusion_checks(q):
    b = q["battery"]
    record("Q.8", {"collusion_checks_run": b["planted"], "planted_collusion_undetected": b["undetected"],
                   "clean_controls_flagged": b["clean_controls_flagged"],
                   "schemes": json.dumps({k: [v["detected"], v["clean_flagged"]] for k, v in b["schemes"].items()}),
                   "real_run_flags": json.dumps({k: len(v) for k, v in (b.get("real_run_flags") or {}).items()})},
           command=CMD, labels={"scripted": True},
           notes="schemes planted by scripted identities in the self-play civilisation after the run: wash trading "
                 "between identities of one controller, circular funding, mutual subscriptions, a price-fixing "
                 "cartel with a communication channel, a reciprocal rating ring, hidden shared state between two "
                 "policy groups; each with a clean control that must not be flagged; detectors take no "
                 "analyst-chosen reference")
    assert b["planted"] >= 5 and b["undetected"] == 0 and b["clean_controls_flagged"] == 0


def test_Q_n1_numeric_facility_only_selfplay_does_not_pass(q):
    from crucible.selfplay.arena import KernelSelfPlayEnv, arena_task
    from crucible.research.learner import EnvSpec
    rig = KernelSelfPlayEnv(arena_task("abm_duopoly", 3), EnvSpec())
    refused = not is_civilisation(rig.world)
    m = q["match"]
    civ_ok = is_civilisation(m.world) and len([g for g, n in _acting_groups(q).items() if n and g in LEARNING]) >= 2
    tools_used = {h["tool"].split(".", 1)[0] for a in m.agents.values() for h in getattr(a, "history", [])
                  if h.get("tool") and h.get("ok")}
    record("Q.n1", {"civilization_selfplay_runs": int(civ_ok), "numeric_facility_run_refused": refused,
                    "hosts_used": len(tools_used), "rig_tool_calls": sum(1 for a in m.agents.values()
                                                                        for h in getattr(a, "history", [])
                                                                        if str(h.get("tool", "")).startswith("rig."))},
           command=CMD, labels=LABELS,
           notes="a run counts only in a Tier 3 civilisation (crucible.civ, with its population) with >= 2 learned "
                 "groups acting through ordinary tools; a 4.0 numeric facility self-play world is not one")
    assert refused and civ_ok
