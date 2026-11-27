"""Gate G (MASTER_SPEC Part 17): small neural population. Owner: workstream B+C (CORE-POP).

Class 2 runs in worlds of the one population (`crucible.population`, `population.experiments.adaptation`): 256
adaptive individuals share policy groups (one weight copy per group, batched inference), each with its own GRU memory;
two groups start from IDENTICAL weights, one learns online from its members' own experienced utilities, the other is
frozen. After phase 1 a controlled new condition (the two most visited rich sites fail, the two broken ones become
rich): G.6 measures the change-specific improvement against a fork of the same world in which the group stops
learning at the switch (paired by member); G.n1 measures adaptation against the identical-init frozen twin, with the
other decisions the networks take reported next to it (the activity of each session, class 3's policy change). Three
seeds. Class-2 networks take two decisions: the activity of each session (which also steps the personal memory, trained
through time) and the web resource to visit. Class 3 (G.4, G.n1) runs its four strategic policy groups
(population.strategic: ventures built through ordinary tools) in a world of its own.
G.7 compares the compact policies with small pretrained language models on the same held-out decisions.

Metrics (docs/v5/ACCEPTANCE_MANIFEST.json):
G.1 simultaneously_active_adaptive >= 256          G.2 distinct_personal_states >= 256
G.3 shared_weight_copies == 1, max_inference_batch >= 64
G.4 independent_strategic_groups >= 2, distinct_group_weight_hashes >= 2
G.5 weight_hash_changed, optimizer_updates >= 100, loss_decrease > 0 (held-out, learned vs initial weights)
G.6 new_condition_improvement_ci_low > 0          G.7 compact_vs_language_comparison_rows >= 2
G.n1 adaptation_effect_ci_low > 0
"""

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
    return [PX.adaptation(s, 5, counterfactual=True) for s in SEEDS]


@pytest.fixture(scope="module")
def shared():
    """The standard profile's class 2 as one shared policy group of 256 (its default): batching and presence."""
    return PX.adaptation(3, 2, {"shared": {"id": 0, "share": 1.0, "seed": 5}})


def test_G1_256_simultaneously_active_adaptive(shared):
    r = shared
    m = {"simultaneously_active_adaptive": int(r["max_adaptive_policy_active_same_day"]),
         "policy_active_per_day": ",".join(str(v) for v in r["adaptive_policy_active_days"].values()),
         "took_sessions_during_run": int(r["adaptive_active_run"]),
         "active_within_two_days": int(r["max_adaptive_active_two_days"]),
         "active_in_phase2": int(r["adaptive_active_phase2"]),
         "active_same_day": int(r["max_adaptive_active_same_day"]),
         "max_adaptive_per_decision_round": int(r["max_adaptive_per_decision_round"]),
         "adaptive_rows": int(r["adaptive_rows"])}
    evidence.record("G.1", m, command=CMD, notes="Class-2 individuals of one shared policy group (the standard "
                                                 "profile's 256), all live in ONE world at the same time with their "
                                                 "own persistent state: the largest number of them for whom their "
                                                 "policy network took a decision on the same simulated day (the day "
                                                 "review that steps each member's memory at its first activation of "
                                                 "the day, and the activity decision of each session; recorded "
                                                 "pop_adaptive.memory batches). Whether an individual browses on a "
                                                 "day is decided by the fitted daily-return model (KuaiRand: about "
                                                 "75-85% of users are active on a given day), so the numbers that "
                                                 "took sessions on one day, within 48 hours and during the run are "
                                                 "reported next to it and are lower.")
    assert m["adaptive_rows"] == 256
    _check("G.1", m)


def test_G2_distinct_persistent_personal_states(runs, shared):
    m = {"distinct_personal_states": int(min(r["distinct_personal_states"] for r in runs)),
         "per_seed_10_days": ",".join(str(r["distinct_personal_states"]) for r in runs),
         "changed_from_initial_state_min": int(min(r["personal_states_changed"] for r in runs)),
         "shared_group_4_days": int(shared["distinct_personal_states"])}
    evidence.record("G.2", m, command=CMD, notes="Distinct GRU memory rows (exact float16 values) of the 256 class-2 "
                                                 "individuals at the end of each 10-day run (minimum over 3 seeds); "
                                                 "memory is a column of the population table, persistent across "
                                                 "sessions, snapshots and forks, and changes only through the "
                                                 "member's own policy steps (the day review on its own morning "
                                                 "observation, the activity decision of each of its sessions); every "
                                                 "member starts from the same zero state, so distinct rows are states "
                                                 "that diverged through each individual's own days.")
    assert m["changed_from_initial_state_min"] == 256
    _check("G.2", m)


def test_G3_shared_weight_batched_inference(shared):
    r = shared
    st = r["adaptive_stats"]
    by = {"resource_choice": int(st.get("max_batch", 0)), "activity_choice": int(st.get("max_batch_act", 0)),
          "day_review": int(st.get("max_batch_day", 0))}
    m = {"shared_weight_copies": int(r["policy_objects"]), "max_inference_batch": int(max(by.values())),
         **{f"max_batch_{k}": v for k, v in by.items()}, "policy_groups": 1, "members": int(r["adaptive_rows"])}
    evidence.record("G.3", m, command=CMD, notes="One network object for the group of 256 (personal state in "
                                                 "columns); the largest batch of one forward pass of the shared "
                                                 "network, per decision it takes (activations of class-2 members "
                                                 "are aggregated in 4-hour decision rounds).")
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
    """Change-specific improvement (pre-registered, D-32 addendum 2): at the switch each seed's world is forked; in
    the fork the learning group stops learning (same weights, memories, world state and random streams at the switch).
    The improvement is each learning-group member's phase-2 realised utility in the world where its policy keeps
    learning minus its utility in the fork (paired by member); the criterion is the lower bound of the 95% bootstrap
    CI over the members of the three seeds (resampled within each seed). This separates what learning under the new
    condition did from the general advantage learned before it, which the identical-init comparison (G.n1) contains
    too (audit B, G.6)."""
    ad = [r["adaptation_after_change"] for r in runs]
    d = np.concatenate([np.asarray(a["member_utility_diffs"], np.float64) for a in ad])
    strata = np.concatenate([np.full(len(a["member_utility_diffs"]), k) for k, a in enumerate(ad)])
    pooled = PX._paired_boot(d, strata, seed=7)
    m = {"new_condition_improvement_ci_low": float(pooled["ci95"][0]),
         "new_condition_improvement_mean": float(pooled["mean"]),
         "new_condition_improvement_ci_high": float(pooled["ci95"][1]), "members_compared": int(pooled["units"]),
         "seeds": len(runs),
         "per_seed_ci_low": ",".join(f"{a['ci95'][0]:.4f}" for a in ad),
         "per_seed_mean": ",".join(f"{a['mean']:.4f}" for a in ad),
         "policy_updates_after_switch_learning": int(sum(a["policy_updates_after_switch"] for a in ad)),
         "policy_updates_after_switch_frozen": int(sum(a["policy_updates_after_switch_frozen"] for a in ad)),
         "share_failing_learning": float(np.mean([a["share_failing_learning"] for a in ad])),
         "share_failing_frozen_at_switch": float(np.mean([a["share_failing_frozen_at_switch"] for a in ad])),
         "weights_changed_after_switch_all_seeds": bool(all(a["hash_end_learning"] != a["hash_switch"] for a in ad)),
         "frozen_at_switch_unchanged_all_seeds": bool(all(a["hash_end_frozen_at_switch"] == a["hash_switch"]
                                                          for a in ad))}
    evidence.record("G.6", m, command=CMD, notes=(
        "Controlled new condition: after phase 1 the two most visited rich sites fail and the two broken ones become "
        "rich (their operators' ordinary tools; scripted by the experimenter). Each seed's world is forked at the "
        "switch; in the fork the learning group stops learning (the main world records the same system call with "
        "learning on, so both worlds keep identical random streams until the first post-switch update). "
        "Improvement = phase-2 mean realised utility of each learning-group member's web visits, main world minus "
        "fork, paired by member; pooled bootstrap 95% CI over the members of three seeds (stratified by seed). "
        "Post-switch updates also use the examples pending at the switch (at most one learning interval); the billed "
        "compute of sandboxed sites may differ between the two worlds (D-20 metering tolerance)."))
    assert m["seeds"] == 3 and m["members_compared"] > 0
    assert m["frozen_at_switch_unchanged_all_seeds"] and m["policy_updates_after_switch_frozen"] == 0
    assert m["weights_changed_after_switch_all_seeds"]
    _check("G.6", m)


@pytest.fixture(scope="module")
def strategic():
    """Class 3 (population.strategic): four policy groups in one market - A (MLP, REINFORCE, objective profit), A0 (a
    frozen copy of A's initial weights), B (GRU, A2C, objective reach), C (MLP, PPO, objective profit) - each
    trained only on its own members' experience of building ventures with ordinary tools; 8 simulated days."""
    import torch
    torch.set_num_threads(1)
    from crucible.core.kernel import World
    from crucible.population import strategic as STR
    from crucible.population.worlds import population_world
    spec = population_world("development", 3, n=200, sites=6, tick_s=1800, adaptive=False,
                            classes={"adaptive": 0, "strategic": 16, "semantic": 0})
    w = World.create(spec, strict=True)
    try:
        w.advance_time(w.now + 8 * DAY)
        ups = [e.payload for e in w.log if e.type == "pop.policy_update" and str(e.payload["group"]).startswith(
            "strategic")]
        own = {g: sorted({r["member"] for r in w.state.tables["pop_strat_exp"].values() if r["group"] == g})
               for g in w.state.tables["pop_strat_groups"]}
        yield {"updates": ups, "behaviour": STR.behaviour(w), "own": own}
    finally:
        w.close()


def test_Gn1_actual_adaptation(runs, strategic):
    """Adaptation, not the presence of networks: identical initial weights, the learning group against its frozen twin
    in the same world after the controlled change (three seeds; criterion: the lowest per-seed CI bound of the web
    decisions' realised utility). Coverage of the other decisions the networks take is measured next to it: the
    realised utility of the sessions whose activity the class-2 policy chose (learning vs frozen twin), and class 3's
    lifetime adaptation (each learning strategic group's change of policy on its own recorded observations, first
    checkpoint vs now; the frozen identical-init group A0 is the control)."""
    lows = [r["phase2_difference"]["ci95"][0] for r in runs]
    sb = strategic["behaviour"]
    ad3 = sb["adapt"]
    learned3 = {g: v for g, v in ad3.items() if v["learning"]}
    m = {"adaptation_effect_ci_low": float(min(lows)), "seeds": len(runs),
         "adaptation_effect_mean": float(np.mean([r["phase2_difference"]["mean"] for r in runs])),
         "per_seed_ci_low": ",".join(f"{x:.4f}" for x in lows),
         "phase1_difference_mean": float(np.mean([r["phase1_difference"]["mean"] for r in runs])),
         "weight_hashes_changed_all_seeds": bool(all(r["hash_end"]["learning"] != r["hash_genesis"]["learning"]
                                                     for r in runs)),
         "activity_sessions_phase2_difference_mean": float(np.mean([r["session_phase2_difference"]["mean"]
                                                                   for r in runs])),
         "activity_sessions_phase2_ci_low_min": float(min(r["session_phase2_difference"]["ci95"][0] for r in runs)),
         "strategic_learning_groups_policy_shift_min": float(min((v["policy_shift_tv"] or 0.0)
                                                                 for v in learned3.values())) if learned3 else 0.0,
         "strategic_frozen_policy_shift": float(max((v["policy_shift_tv"] or 0.0) for g, v in ad3.items()
                                                    if not v["learning"])) if ad3 else 0.0,
         "strategic_adapt_over_lifetime": bool(sb["kinds"]["adapt_over_lifetime"])}
    shifts = {g: v["policy_shift_tv"] for g, v in ad3.items()}
    evidence.record("G.n1", m, command=CMD, notes=(
        "Actual adaptation, not the presence of networks: what experience did to the networks' decisions. Web "
        "decisions (criterion): learning group minus its identical-init frozen twin after the controlled change, "
        "bootstrap CI per seed, lowest of three. Activity decisions: realised session utility, learning vs frozen "
        "twin (reported). Class 3: total-variation distance between each strategic group's action distributions on "
        f"its own recorded observations under its first checkpoint and its current weights: {shifts}. "
        f"Per seed (web): {[r['phase2_difference'] for r in runs]}; activity sessions: "
        f"{[r['session_phase2_difference'] for r in runs]}."))
    assert m["seeds"] == 3 and m["weight_hashes_changed_all_seeds"]
    assert m["strategic_adapt_over_lifetime"] and m["strategic_frozen_policy_shift"] == 0.0, ad3
    assert any(not v["learning"] for v in ad3.values()), ad3          # the frozen control was measured too
    _check("G.n1", m)


def test_G4_independent_strategic_groups(strategic):
    """Class 3's policy groups update independently: each group's weights change only through updates on its own
    members' decisions; the frozen copy A0 never changes."""
    ups, b, own = strategic["updates"], strategic["behaviour"], strategic["own"]
    by = {}
    for u in ups:
        by.setdefault(u["group"], []).append(u["weights_hash"])
    final = {g: h[-1] for g, h in by.items()}
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
