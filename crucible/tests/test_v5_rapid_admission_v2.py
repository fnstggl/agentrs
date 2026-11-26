"""The redesigned Tier 1 admission machinery: the fixed-budget dominance search, structural fingerprints that ignore
parameter/size/horizon/distractor variants, intervention-verified distinctness, the D-9 recreation check, the
independent scalar twin of the primitive-graph engine, and independently generated goals."""
from __future__ import annotations

import numpy as np
import pytest

from crucible.rapid import distinct as DI
from crucible.rapid import dominance as D
from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP
from crucible.rapid import goals as G
from crucible.rapid import goals_eval as GE
from crucible.rapid import twin as TW
from crucible.rapid.references import play


# ====================================================================== dominance
def test_the_search_covers_the_full_action_space_with_a_fixed_budget():
    env = F.make("mk_elastic", 0)
    res = D.search(env, seed=0)
    assert res["episodes"] <= 1 + sum(D.BUDGET.values()) + 2
    assert {"grid", "random", "coordinate", "reactive", "repeat_best"} <= set(res["by_arm"])
    joint = D.Joint(env)
    kinds = {a["kind"] for s in joint.canonical()[:12] for a in joint.coords["a0"].actions(s["a0"])}
    assert {"levers", "discrete"} <= kinds                     # settings combine every world-changing kind


def test_a_world_solved_by_a_constant_is_rejected_and_an_adaptive_one_is_not():
    # ch_silent_reversal: the effect's sign flips at a hidden tick; no constant or memoryless rule tracks it
    env = F.make("ch_silent_reversal", 0)
    simple = D.search(env, seed=0)
    v_ref = play(F.make("ch_silent_reversal", 0))["value"]
    assert D.check(simple["default"], v_ref, simple)["ok"]
    # a stationary lever world (overdriving wears a machine past a fixed hidden level: the best policy is a constant
    # drive below it): the hindsight-best constant beats the generic adaptive learner (hc_confounded and cx_overdose,
    # the earlier examples, became prediction tasks in their version 2)
    env = F.make("dl_short_term_damage", 0)
    simple = D.search(env, seed=0)
    v_ref = play(F.make("dl_short_term_damage", 0))["value"]
    c = D.check(simple["default"], v_ref, simple)
    assert not c["ok"] and c["best_simple"].split(":")[0] in ("grid", "random", "coordinate", "reactive",
                                                               "repeat_best")


def test_prediction_worlds_are_compared_with_naive_predictors():
    env = F.make("sci_interaction_hypotheses", 1)
    res = D.search(env, seed=1)
    assert "naive_prediction" in res["by_arm"]


# ====================================================================== fingerprints
@pytest.mark.parametrize("family", ["hc_confounded", "net_routing", "posm_sign_switch", "team_assembly",
                                    "pe_contention_sched", "sci_hidden_conservation"])
def test_fingerprints_ignore_axes_and_are_purely_categorical(family):
    fp = FP.family_fingerprint(family)
    fam = F.FAMILIES[family]
    for ax, v in fam.axes.items():
        nv = (int(v) * 2 if isinstance(v, int) and not isinstance(v, bool) else (v * 2 if v else 1.0))
        assert FP.distance(fp, FP.family_fingerprint(family, **{ax: nv})) == 0.0, (family, ax)

    def numeric(x):
        if isinstance(x, bool):
            return False
        if isinstance(x, (int, float)):
            return True
        if isinstance(x, dict):
            return any(numeric(v) for v in x.values())
        if isinstance(x, list):
            return any(numeric(v) for v in x)
        return False
    assert not any(numeric(fp[k]) for k in FP.WEIGHTS)


def test_d9_recreation_of_a_sealed_mechanism_is_detected():
    hits = {n: FP.legacy_recreation(F.REGISTRY[n]["fingerprint"])["recreates"] for n in F.FAMILIES}
    assert "scm_regime" in hits["ch_sign_reversal"]            # a timed strength/sign switch of one lever
    assert "scm_invu" in hits["ch_wandering_taste"]            # a 4.0 validation mechanism counts too (H-D2)
    assert not hits["ch_silent_reversal"]                      # ... differs from it in its mechanism motifs
    # D-9: such families are evaluation families; no training family recreates a 4.0 held-out mechanism
    assert F.FAMILIES["ch_sign_reversal"].split == "val" and F.FAMILIES["ch_wandering_taste"].split == "val"
    assert not any(hits[n] for n in F.families(split="train"))


# ====================================================================== distinctness by intervention
def test_claimed_differences_are_verified_by_intervention():
    assert DI.verify("mk_elastic", "mk_signal")["ok"]                         # look-alike price protocol
    assert DI.verify("srch_field", "srch_distractors")["method"] in ("observation_structure", "interface",
                                                                     "behaviour")
    v = DI.verify("sci_competing_hypotheses", "sci_interaction_hypotheses")
    assert v["ok"] and v["method"] == "behaviour"


# ====================================================================== the independent scalar twin
@pytest.mark.parametrize("family", ["hc_conditional", "ch_silent_reversal", "sci_hidden_conservation", "mk_drift"])
def test_scalar_twin_agrees_with_the_vectorised_engine(family):
    assert TW.supports(F.build_config(family, 0))
    r = TW.compare(family, 3, ticks=40)
    assert r["ok"] and r["ticks"] == 40, r


def test_lagged_gauges_report_the_previous_tick():
    env = F.make("hc_conditional", 0)
    g = next(x for x in env._config["gauges"] if x["lag"] == 1)
    src = env._engine.idx[g["src"][0]]
    prev = None
    for _ in range(5):
        env.step(None)
        h = env._state["h"][0]
        prev = float(h[-2, src])
    sd = env._config["gauges"][env._engine.gauge_names.index(g["name"])]["chain"][0]["params"]["sd"]
    assert abs(env.observe()["numbers"][g["name"]] - prev) < 6 * sd + 1e-9


# ====================================================================== generated goals
def test_goals_are_generated_from_the_public_face_only():
    a = G.generate(F.make("sci_functional_law", 0), 0.5, 3)
    b = G.generate(F.make("sci_functional_law", 0), 0.5, 3)
    assert a == b and a["type"] in G.GOAL_TYPES and a["text"]
    env = F.make("sci_functional_law", 0, objective=a)
    assert env.observe()["objective"] == a["text"] and env.spec["objective_mode"] == "generated"
    import json
    cfg_text = json.dumps(F.build_config("sci_functional_law", 0))
    assert "claim" in cfg_text and "claim" not in json.dumps(a)          # nothing of the hidden config


@pytest.mark.parametrize("gt", G.GOAL_TYPES)
def test_goal_feasibility_is_validated_without_guaranteeing_success(gt):
    fam = "sci_functional_law" if gt in ("predict_heldout", "identify_effect") else "ch_silent_reversal"
    g = G.generate(F.make(fam, 0), 0.5, 2, goal_type=gt)
    v = GE.validate(lambda: F.make(fam, 0), g)
    assert isinstance(v["feasible"], bool)
    env = F.make(fam, 0, objective=g)
    while not env.done:
        env.step(None)
    ev = GE.evaluate(env, g)
    assert "value" in ev and np.isfinite(ev["value"]) or ev["value"] == -np.inf


def test_identify_goals_are_scored_against_the_counterfactual():
    env = F.make("sci_functional_law", 0)
    g = G.generate(env, 0.5, 2, goal_type="identify_effect")
    tr = GE.truth(F.make("sci_functional_law", 0), g)
    e = F.make("sci_functional_law", 0, objective=g)
    e.step({"kind": "message", "to": "journal", "content": {"goal": g["id"], "sign": int(tr["value"])}})
    assert GE.evaluate(e, g)["success"]
    e2 = F.make("sci_functional_law", 0, objective=g)
    e2.step({"kind": "message", "to": "journal", "content": {"goal": g["id"], "sign": -int(tr["value"]) or 1}})
    assert not GE.evaluate(e2, g)["success"]


# ====================================================================== the mechanism check (DEF-H-4)
def test_a_family_whose_own_claims_fail_is_not_admitted():
    """Admission's eleventh check runs the family's own controlled-intervention claims: a world built like
    ch_silent_reversal but without its reversal keeps the family's claim, the claim fails and the mechanism check
    rejects the world on every seed, while the real family passes it."""
    from crucible.rapid import admission as AD
    from crucible.rapid import claims as CL
    from crucible.rapid.families_v2 import ch_silent_reversal
    assert "mechanism" in AD.CHECKS
    f = F.FAMILIES["ch_silent_reversal"]

    def no_reversal(rng, d):
        cfg = ch_silent_reversal(rng, dict(d))
        for nd in cfg["nodes"]:
            if nd.get("prim", "").startswith("regime_change"):
                nd["params"]["after"] = nd["params"]["before"]
        return cfg
    name = "ctl_claim_without_mechanism"
    F.FAMILIES[name] = F.Family(name, f.engine, f.category, f.item, "train", "negative control", "negative control",
                                no_reversal, dict(f.axes), f.objective, (), f.claims, f.identify, f.reference,
                                f.oracle)
    try:
        for seed in (0, 1):
            bad = CL.run(name, seed)
            good = CL.run("ch_silent_reversal", seed)
            assert bad and not all(r.get("ok") for r in bad.values())
            assert good and all(r.get("ok") for r in good.values())
    finally:
        F.FAMILIES.pop(name, None)
