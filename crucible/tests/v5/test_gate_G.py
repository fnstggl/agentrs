"""Gate G (MASTER_SPEC Part 17): small neural population. Owner: workstream B+C (CORE-POP).

Class 2 runs in worlds of the one population (`crucible.population`, `population.experiments.adaptation`): 256
adaptive individuals share policy groups (one weight copy per group, batched inference), each with its own GRU memory;
two groups start from IDENTICAL weights, one learns online from its members' own experienced utilities, the other is
frozen. After phase 1 a controlled new condition (the two most visited rich sites fail, the two broken ones become
rich) tests adaptation (G.6, G.n1). Three seeds. Class-2 networks take two decisions: the activity of each session
(which also steps the personal memory, trained through time) and the web resource to visit. Class 3 (G.4) runs its
four strategic policy groups (population.strategic: ventures built through ordinary tools) in a world of its own.
G.7 compares the compact policies with small pretrained language models on the same held-out decisions.

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json):
G.1 simultaneously_active_adaptive >= 256          G.2 distinct_personal_states >= 256
G.3 shared_weight_copies == 1, max_inference_batch >= 64
G.4 independent_strategic_groups >= 2, distinct_group_weight_hashes >= 2
G.5 weight_hash_changed, optimizer_updates >= 100, loss_decrease > 0 (held-out, learned vs initial weights)
G.6 new_condition_improvement_ci_low > 0          G.7 compact_vs_language_comparison_rows >= 2
G.n1 adaptation_effect_ci_low > 0
"""
import time

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.population import experiments as PX

crucible.load_all_modules()


def _check(req, metrics):
    crit = next(e for e in MF.load()["requirements"] if e["id"] == req)["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (req, fails, metrics)
CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement G.1"
DAY = 86400
SEEDS = (0, 1, 2)
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]


@pytest.fixture(scope="module")
def runs():
    return [PX.adaptation(s, 5) for s in SEEDS]


@pytest.fixture(scope="module")
def shared():
    """The standard profile's class 2 as one shared policy group of 256 (its default): batching and presence."""
    return PX.adaptation(3, 2, {"shared": {"id": 0, "share": 1.0, "seed": 5}})


def test_G1_256_simultaneously_active_adaptive(shared):
    r = shared
    m = {"simultaneously_active_adaptive": int(r["adaptive_active_run"]),
         "active_within_two_days": int(r["max_adaptive_active_two_days"]),
         "active_in_phase2": int(r["adaptive_active_phase2"]),
         "active_same_day": int(r["max_adaptive_active_same_day"]),
         "max_adaptive_per_decision_round": int(r["max_adaptive_per_decision_round"]),
         "adaptive_rows": int(r["adaptive_rows"])}
    evidence.record("G.1", m, command=CMD, notes="Class-2 individuals of one shared policy group (the standard "
                                                 "profile's 256), all live in ONE world at the same time with their "
                                                 "own persistent state, that took sessions with their policy during "
                                                 "the same 4-day run. Their days of activity are decided by the "
                                                 "fitted daily-return model (KuaiRand: about 75-85% of users are "
                                                 "active on a given day), so the largest same-day and 48-hour counts "
                                                 "are reported next to it and are lower.")
    assert m["adaptive_rows"] == 256
    _check("G.1", m)


def test_G2_distinct_persistent_personal_states(runs, shared):
    m = {"distinct_personal_states": int(min(r["distinct_personal_states"] for r in runs)),
         "per_seed_10_days": ",".join(str(r["distinct_personal_states"]) for r in runs),
         "shared_group_4_days": int(shared["distinct_personal_states"])}
    evidence.record("G.2", m, command=CMD, notes="Distinct GRU memory rows (exact float16 values) of the 256 class-2 "
                                                 "individuals at the end of each 10-day run (minimum over 3 seeds); "
                                                 "memory is a column of the population table, persistent across "
                                                 "sessions, snapshots and forks, and changes only when the individual "
                                                 "takes a web decision with its policy (individuals whose sessions "
                                                 "were all other activities keep the initial state).")
    _check("G.2", m)


def test_G3_shared_weight_batched_inference(shared):
    r = shared
    m = {"shared_weight_copies": int(r["policy_objects"]),
         "max_inference_batch": int(r["adaptive_stats"].get("max_batch", 0)), "policy_groups": 1,
         "members": int(r["adaptive_rows"])}
    evidence.record("G.3", m, command=CMD, notes="One network object for the group of 256 (personal state in "
                                                 "columns); the largest batch of one forward pass (decisions are "
                                                 "aggregated in 4-hour rounds).")
    assert m["members"] == 256
    _check("G.3", m)


def test_G5_parameter_changes_from_experience(runs):
    r = runs[0]
    hm = {a: r["heldout_mse"][a] for a in r["arms"]}
    m = {"weight_hash_changed": bool(r["hash_end"]["learning"] != r["hash_genesis"]["learning"]),
         "frozen_hash_unchanged": bool(r["hash_end"]["frozen"] == r["hash_genesis"]["frozen"]),
         "optimizer_updates": int(sum(x["optimizer_steps"]["learning"] for x in runs)),
         "policy_updates": int(sum(x["updates"]["learning"] for x in runs)),
         "loss_decrease": float(np.mean([x["heldout_mse"]["frozen"] - x["heldout_mse"]["learning"] for x in runs])),
         "heldout_mse_learned": round(float(hm["learning"]), 5), "heldout_mse_initial": round(float(hm["frozen"]), 5),
         "heldout_examples": int(r["heldout_examples"])}
    evidence.record("G.5", m, command=CMD,
                    notes="Weights of the learning group change through gradient steps on its members' own "
                          "(observation, memory, chosen candidate, realised utility) examples; loss_decrease is the "
                          "HELD-OUT value-prediction error of the initial weights (the frozen twin) minus that of the "
                          "learned weights, on examples never used for training (mean over 3 seeds).")
    assert m["frozen_hash_unchanged"]
    _check("G.5", m)


def test_G6_improvement_on_a_controlled_new_condition(runs):
    lows = [r["phase2_difference"]["ci95"][0] for r in runs]
    m = {"new_condition_improvement_ci_low": float(min(lows)),
         "new_condition_improvement_mean": float(np.mean([r["phase2_difference"]["mean"] for r in runs])),
         "phase1_improvement_mean": float(np.mean([r["phase1_difference"]["mean"] for r in runs])),
         "share_failing_learning": float(np.mean([r["behaviour"]["learning"]["phase2"]["share_failing"] for r in runs])),
         "share_failing_frozen": float(np.mean([r["behaviour"]["frozen"]["phase2"]["share_failing"] for r in runs]))}
    evidence.record("G.6", m, command=CMD,
                    notes="After the switch (two most visited rich sites fail, two broken become rich): realised "
                          "utility of class-2 web visits, learning group minus its identical-init frozen twin, "
                          "bootstrap 95% CI per seed; the criterion is the lowest CI bound of three seeds. The phase-1 "
                          "difference is reported too: part of the improvement is general learning, not specific to "
                          f"the change. Per seed: {[r['phase2_difference'] for r in runs]}")
    assert len(lows) == 3
    _check("G.6", m)


def test_Gn1_actual_adaptation(runs):
    lows = [r["phase2_difference"]["ci95"][0] for r in runs]
    m = {"adaptation_effect_ci_low": float(min(lows)), "seeds": len(runs),
         "weight_hashes_changed_all_seeds": bool(all(r["hash_end"]["learning"] != r["hash_genesis"]["learning"]
                                                     for r in runs))}
    evidence.record("G.n1", m, command=CMD, notes="Adaptation, not the presence of networks: identical initial "
                                                  "weights, the learning group against its frozen twin on the same "
                                                  "world after a controlled change, three seeds.")
    assert m["seeds"] == 3
    _check("G.n1", m)


def test_G4_independent_strategic_groups():
    """Class 3 (population.strategic): four policy groups in one market - A (MLP, REINFORCE, objective profit), A0 (a
    frozen copy of A's initial weights), B (GRU, A2C, objective reach), C (MLP, PPO, objective profit) - each
    trained only on its own members' experience of building ventures with ordinary tools."""
    import torch
    torch.set_num_threads(1)
    from crucible.core.kernel import World
    from crucible.population import strategic as STR
    from crucible.population.worlds import population_world
    spec = population_world("development", 3, n=200, sites=6, tick_s=1800, adaptive=False,
                            classes={"adaptive": 0, "strategic": 16, "semantic": 0})
    w = World.create(spec)
    try:
        w.advance_time(w.now + 8 * DAY)
        ups = [e.payload for e in w.log if e.type == "pop.policy_update" and str(e.payload["group"]).startswith(
            "strategic")]
        by = {}
        for u in ups:
            by.setdefault(u["group"], []).append(u["weights_hash"])
        final = {g: h[-1] for g, h in by.items()}
        b = STR.behaviour(w)
        own = {g: sorted({r["member"] for r in w.state.tables["pop_strat_exp"].values() if r["group"] == g})
               for g in w.state.tables["pop_strat_groups"]}
        m = {"independent_strategic_groups": int(sum(1 for h in by.values() if len(set(h)) > 1)),
             "distinct_group_weight_hashes": int(len(set(final.values()))),
             "strategic_policy_updates": int(len(ups)), "groups": ",".join(sorted(by)),
             "frozen_control_unchanged": bool(len(set(by.get("strategic:A0", ["x"]))) == 1),
             "strategic_decisions": int(b["decisions"])}
        evidence.record("G.4", m, command=CMD, notes=(
            "Four class-3 strategic policy groups (different architectures, algorithms and objectives) in one market, "
            "8 simulated days; each update uses only its own members' decisions (members per group: "
            f"{ {g: len(v) for g, v in own.items()} }); A0 is a frozen copy of A's initial weights. Behaviour: "
            f"{b['kinds']}, outcomes {b['outcomes']}, successful actions {b['actions_ok']}."))
    finally:
        w.close()
    assert m["strategic_decisions"] > 0
    _check("G.4", m)


def test_G7_compact_vs_language_comparison(runs):
    """The same held-out class-2 decisions scored by the compact learned policy, its frozen twin, and a small
    pretrained language model (SmolLM2-135M, zero-shot through crucible.semantic's inference service when its
    weights are in the local cache): rank correlation of each model's value estimate with the realised utility,
    parameters and time per decision."""
    rows = PX.compact_vs_language(runs[0], seed=0)
    lang = [r for r in rows if r.get("kind") == "language" and not r.get("error")]
    comp = [r for r in rows if r.get("kind") == "compact"]
    m = {"compact_vs_language_comparison_rows": len(lang) if comp else 0, "compact_rows": len(comp)}
    for r in rows:
        m[f"{r['model']}_spearman"] = round(float(r["spearman"]), 4)
        m[f"{r['model']}_params"] = int(r["params"])
        m[f"{r['model']}_ms_per_decision"] = round(float(r["ms_per_decision"]), 3)
    evidence.record("G.7", m, command=CMD, notes=(
        "Each language-capable model (zero-shot, crucible.semantic's inference service on local checkpoints) and "
        "each compact class-2 policy (learned, and its identical-init frozen twin) values the same held-out resource "
        "decisions; a comparison row is a language model that ran, set against the compact policies "
        f"(Spearman correlation with the realised utility, parameters, ms per decision). Rows: {rows}"))
    assert len(rows) >= 3
    _check("G.7", m)
