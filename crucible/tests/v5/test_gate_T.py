"""Gate T - Automatic world generation and curriculum (MASTER_SPEC Parts 7, 14.3, Gate T; workstream N).

Generated worlds come from `crucible.research.v5.compose` (structures of the generator grammar built with the
rapid `GraphBuilder`, economics calibrated with an interior optimum drawn continuously and defaults on a side
independent of it). Counts are measured on built worlds; validation is the full Part 7.4 admission of
`crucible.rapid.admission` (eleven checks, including mechanism claims and the dominance search) run on generated
structures registered for the duration of the run. Heavy: run through the slot scheduler, e.g.

    python -m crucible.scale.slots --cpu 6 -- env CRUCIBLE_RESEARCH_JOBS=6 python -m pytest tests/v5/test_gate_T.py
"""
from __future__ import annotations

import collections
import os

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import compose as C
from crucible.research.v5 import curriculum as CU
from crucible.research.v5 import curriculum_lab as CL
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import proposal as P
from crucible.research.v5 import sampler as S
from crucible.research.v5.episodes import references, run_episode
from crucible.research.v5.sampler import build_env, rapid_world

pytestmark = [pytest.mark.slow, pytest.mark.timeout(14400)]
JOBS = int(os.environ.get("CRUCIBLE_RESEARCH_JOBS", "1"))


def _structures(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    cats = sorted(CU.TEMPLATES)
    while len(out) < n:
        s = CU.targeted_structure(cats[len(out) % len(cats)], rng) if len(out) % 2 else CU.random_structure(rng)
        sid = C.structure_id(s)
        if sid not in seen:
            seen.add(sid)
            out.append(s)
    return out


@pytest.fixture(scope="module")
def generated():
    """Calibrated generated structures with their measured fingerprints."""
    from crucible.rapid import fingerprints as FP
    rows = []
    for s in _structures(110, 7):
        e = D.generated_entry(s)
        cfg = rapid_world(e, 0, {}, None)._config
        if not cfg.get("calibrated"):
            continue
        rows.append({"structure": s, "entry": e, "fp": D.entry_fingerprint(e)})
    return rows


def test_T1_procedural_structural_composition(generated):
    from crucible.rapid import fingerprints as FP
    reg = D.build_registry()
    fams = {w["name"]: D.entry_fingerprint(w) for w in reg.worlds(tier=1, source="crucible.rapid")}
    distinct = []
    for r in generated:
        if any(FP.distance(r["fp"], f) <= 0.0 for f in fams.values()):
            continue
        if any(FP.distance(r["fp"], q["fp"]) <= 0.0 for q in distinct):
            continue
        distinct.append(r)
    topo = collections.Counter(r["structure"]["topology"] for r in distinct)
    record("T.1", {"distinct_generated_structures": len(distinct), "calibrated_structures": len(generated),
                   "topologies_used": len(topo), "registered_rapid_families": len(fams)},
           notes=f"Fingerprint-distinct (distance > 0 to every other generated structure and every registered rapid "
                 f"family), calibrated generated structures; topologies {dict(topo)}")
    assert len(distinct) >= 50


def test_T2_configurable_participant_populations():
    """Each population configuration of the grammar is built on the same base structures and seeds; it counts when
    the participant mechanisms present in the built world (the primitives the composer placed in participant roles:
    adaptive rivals, learning customers, endogenous demand - read from the built configuration) differ from the
    population-free world and the outcome of the same adapter differs (paired by structure and seed)."""
    roles = ("rival", "customers", "demand", "share")

    def participants_of(env):
        return sorted({f"{n.get('role')}:{n.get('prim')}" for n in env.env._config["nodes"] if n.get("role") in roles})
    rng = np.random.default_rng(22)
    bases = [{"topology": t, "motifs": ["saturation"], "n_levers": 1, "horizon": 60} for t in ("chain", "fork", "hub")]
    outcome = {}
    parts = {}
    for kind in C.POPULATIONS:
        vals, fps = [], set()
        for b in bases:
            s = dict(b, population={"kind": kind})
            e = D.generated_entry(s)
            for sd in (1, 2):
                env = build_env(e, sd, {}, {"mode": "assigned"})
                fps.add(str(participants_of(env)))
                rec = run_episode(env, CL.PlantedAdapter(0), learn=False, rng=rng)
                vals.append(rec["value"])
        outcome[kind] = np.array(vals)
        parts[kind] = fps
    tested = ["none"] + [k for k in C.POPULATIONS if k != "none" and parts[k] != parts["none"] and
                         float(np.mean(np.abs(outcome[k] - outcome["none"]))) > 1e-6]
    record("T.2", {"population_configurations_tested": len(tested), "population_configurations_defined":
                   len(C.POPULATIONS), "worlds_built": int(sum(len(v) for v in outcome.values()))},
           notes=f"Configurations with measured participant and outcome differences: {tested}; participant "
                 f"mechanisms {({k: sorted(v) for k, v in parts.items()})}. These are equation primitives inside one "
                 "causal graph (Tier 1), not autonomous policies of crucible.population.")
    assert len(tested) >= 3


def test_T3_independently_varied_objectives(generated):
    """Goals of every type of the independent goal generator (`crucible.rapid.goals`) are drawn from the public face
    of worlds of different structures - generated structures and the registry's Tier 1 rapid training families (the
    prediction and identification types need a journal, which generated worlds do not offer); a type counts when it
    was drawn and validated feasible on at least three distinct structures. Independence: on the generated worlds,
    the goal-type distribution does not depend on the structure's category (chi-square test of independence)."""
    from crucible.rapid import goals as G
    from crucible.rapid import goals_eval as GE
    feasible = collections.defaultdict(set)
    table = collections.Counter()
    reg = D.build_registry()
    rapid = [dict(D.thaw(w)) for w in reg.worlds(tier=1, family_partition="train", source="crucible.rapid")
             if "message" in w["action_kinds"]][:12]
    for e in rapid:
        for t in ("predict_heldout", "identify_effect"):
            env = rapid_world(e, 3, {}, None)
            if not CU.goal_possible(env, t):
                continue
            g = G.generate(env, 0.4, 5, t)
            if GE.validate(lambda: rapid_world(e, 3, {}, None), g).get("feasible"):
                feasible[t].add(e["name"])
    for r in generated[:24]:
        e = r["entry"]
        for t in G.GOAL_TYPES:
            env = rapid_world(e, 3, {}, None)
            possible = CU.goal_possible(env, t)
            if not possible:
                continue
            g = G.generate(env, 0.4, 5, t)
            v = GE.validate(lambda: rapid_world(e, 3, {}, None), g)
            if v.get("feasible"):
                feasible[t].add(e["name"])
        rng = np.random.default_rng(int(e["name"][4:12], 16) % 1000)
        g2 = G.generate(rapid_world(e, 4, {}, None), 0.5, int(rng.integers(1000)))
        table[(CU.category_of(r["structure"]), g2["type"])] += 1
    types = [t for t in G.GOAL_TYPES if len(feasible[t]) >= 3]
    cats = sorted({c for c, _ in table})
    gts = sorted({g for _, g in table})
    obs = np.array([[table[(c, g)] for g in gts] for c in cats], dtype=float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    chi2 = float(((obs - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    dof = max(1, (len(cats) - 1) * (len(gts) - 1))
    record("T.3", {"objective_types": len(types), "objective_types_defined": len(G.GOAL_TYPES),
                   "independence_chi2": chi2, "independence_dof": dof},
           notes=f"Feasible structures per goal type: {({t: len(v) for t, v in feasible.items()})}; type x category "
                 f"table {dict(table)}")
    assert len(types) >= 5
    assert chi2 < 3 * dof


# T.4 sample, fixed before the gate run (D-69 addendum): the first 200 structures of the generator stream used by
# T.1 (alternating curriculum-targeted and uniformly random structures, seed 7), those calibrated at seed 0, each
# at five world seeds. Every world of the sample is admitted; none is selected by its outcome.
T4_SAMPLE = {"structures": 200, "structure_seed": 7, "world_seeds": (0, 1, 2, 3, 4)}
# D-69 addendum 3 (fixed before the run): the stream continues past the addendum-2 sample (its first 200 structures,
# always admitted in full and reported on their own) and stops at the first structure at which 50 worlds are validated,
# or after 1000 structures
T4_STREAM = {"structures": 1000, "target": 50, "prefix": T4_SAMPLE["structures"]}


def test_T4_validated_generated_worlds():
    """Full Part 7.4 admission (`crucible.rapid.admission.seed_report`, eleven checks) of generated worlds; a world
    (structure, seed) is validated when every check passes on it. Static generated worlds (a fixed best setting)
    fail no_dominant_strategy by construction of that check - the best constant in hindsight is then near-optimal;
    worlds whose optimum keeps moving (driver wandering_optimum) are the ones in which adaptation pays. Sequential
    generate-then-admit design (`genadmit.admit_until`, D-69 addendum 3): the structure stream of T.1 (seed 7) is
    walked in order (calibrated structures only, five world seeds each) until 50 worlds are validated or 1000
    structures are walked; its first 200 structures are the fixed sample of D-69 addendum 2, reported on their own.
    The validated share (the generator's yield) is reported with the count."""
    from crucible.research.v5.genadmit import admit_until
    structs = _structures(T4_STREAM["structures"], T4_SAMPLE["structure_seed"])
    res = admit_until(structs, seeds=T4_SAMPLE["world_seeds"], target=T4_STREAM["target"], jobs=JOBS,
                      at_least=T4_STREAM["prefix"])
    by_id = {C.structure_id(s): s for s in structs}
    worlds = res["worlds"]
    fails = collections.Counter(c for w in worlds for c, ok in w["checks"].items() if not ok)
    kind = collections.Counter()
    checked = collections.Counter()
    for w in worlds:
        k = "wandering_optimum" if "wandering_optimum" in by_id[w["structure_id"]]["drivers"] else "other"
        checked[k] += 1
        kind[k] += int(w["validated"])
    errors = sum(1 for w in worlds if w.get("error"))
    pre = T4_STREAM["prefix"]
    pre_ids = {C.structure_id(s) for s in structs[:pre]}
    pre_worlds = [w for w in worlds if w["structure_id"] in pre_ids]
    record("T.4", {"generated_worlds_validated": res["validated"], "generated_worlds_checked": len(worlds),
                   "validated_share": res["validated"] / max(1, len(worlds)),
                   "structures_with_a_validated_world": res["structures_validated"],
                   "structures_walked": res["structures_walked"], "structures_admitted": res["structures_calibrated"],
                   "target_reached": bool(res["reached"]), "worlds_with_admission_errors": errors,
                   "fixed_sample_worlds_validated": sum(1 for w in pre_worlds if w["validated"]),
                   "fixed_sample_worlds_checked": len(pre_worlds),
                   "validated_beyond_stop_not_counted": res["validated_beyond_stop"]},
           notes=f"Stream {T4_SAMPLE['structure_seed']}, design {T4_STREAM}, world seeds {T4_SAMPLE['world_seeds']}; "
                 f"stopped after {res['structures_walked']} structures ({res['structures_calibrated']} calibrated); "
                 f"validated / checked by dynamics: {({k: f'{kind[k]}/{checked[k]}' for k in checked})}; failed "
                 f"checks over all worlds: {dict(fails)}. Fixed sample of D-69 addendum 2 (first {pre} structures): "
                 f"{sum(1 for w in pre_worlds if w['validated'])} of {len(pre_worlds)} worlds validated. The "
                 "no_dominant_strategy reference of wandering-optimum worlds is the scripted public scan-then-track "
                 "reference research.v5.genref (labelled scripted); other generated worlds use "
                 "rapid.refs_v2.adaptive. The wandering lever's public default is independent of the hidden centre "
                 "(DEF-N-8).")
    assert res["validated"] >= 50, dict(fails)


@pytest.fixture(scope="module")
def reference():
    return CL.reference_profile()


def test_T5_curriculum_selection_from_measured_behaviour():
    """Design `curriculum_lab.T5_DESIGN` (wave 3b repair, D-69 addendum, fixed before this run; seeds never used in
    development): does the curriculum's targeting follow the learner it measures? Eight frozen learners; the truth
    is each learner's paired excess regret over the reference adapter per template category on independent probe
    worlds; the statistic is the mean Spearman correlation between a run's targeted weights and its own learner's
    truth profile, tested against all 8! relabellings of learners to truths (a curriculum biased to some categories
    whatever the learner does gains nothing from the relabelling null). The rank-of-weakest-category version of the
    same test is reported as secondary."""
    r = CL.selection_experiment_v2(CL.T5_DESIGN, jobs=JOBS)
    record("T.5", {"weakness_selection_pvalue": r["p_value"], "mean_spearman_own_learner": r["mean_spearman"],
                   "mean_spearman_other_learners": r["mean_spearman_other_learners"],
                   "rank_test_pvalue": r["rank_p_value"], "mean_rank_of_measured_weakness": r["mean_rank_of_weakest"],
                   "mean_rank_of_other_learners_weakness": r["mean_rank_of_other_learners_weakest"],
                   "runs": len(r["runs"]), "learners": len(r["design"]["learners"]),
                   "distinct_weakest_categories": r["distinct_weakest_categories"],
                   "categories_compared": len(r["design"]["categories"]), "permutations": r["permutations"]},
           notes=f"Design {r['design']}. Measured weakest category per learner {r['weakest']}. Per run (learner, "
                 f"seed, Spearman with own truth, rank of own weakest among 8): "
                 f"{[(x['learner'], x['seed'], round(x['spearman'], 3), x['rank_of_weakest']) for x in r['runs']]}. "
                 f"Truth profiles {({k: {c: round(v, 3) for c, v in t.items()} for k, t in r['truth'].items()})}. "
                 "Forgetting and generalization are excluded: a frozen learner has no truth there.")
    assert r["p_value"] <= 0.05


def test_T6_random_exploration_preserved(reference):
    """A learner with one strong weakness (lag-blind) drives the targeted weights to one category; the realised
    share of uniformly random draws and the category entropy are measured over 400 draws."""
    cur = CU.Curriculum(seed=6, reference=reference)
    CL.measure(cur, CL.PlantedAdapter.planted("lag_blind"), 40, seed=6)
    for _ in range(400):
        cur.next()
    st = cur.stats()
    w = cur.weights()
    tail = cur.history[-400:]
    share = sum(1 for h in tail if h["kind"] == "random") / len(tail)
    record("T.6", {"random_component_share": float(share), "configured_random_share": cur.random_share,
                   "normalised_category_entropy": float(st["normalised_entropy"]),
                   "max_targeted_weight": float(max(w.values()))},
           notes=f"Targeted weights {w}; category counts {st['category_counts']}")
    assert share >= 0.2 and st["normalised_entropy"] >= 0.5


def test_T7_evaluation_partitions_protected(tmp_path):
    reg = D.build_registry()
    leaks = {}
    smp = S.Sampler(reg, S.Mixture(tiers={1: 1.0}, name="t1"), seed=71)
    leaks["sampler_stream"] = sum(1 for _ in range(5000) if reg.partition_of((sp := smp.next())["entry"],
                                                                                  sp["seed"]) != "train")
    ev_id = reg.worlds(tier=1, family_partition="evaluation", source="crucible.rapid")[0]["id"]
    try:
        S.make_env(dict(S.episode_spec(reg, ev_id, 3), partition="train"), reg)
        leaks["forged_spec"] = 1
    except D.PartitionError:
        leaks["forged_spec"] = 0
    cur = CU.Curriculum(seed=1)
    stored = sum(int(cur.observe({"partition": p, "category": "memory", "score": 0.1}))
                 for p in ("evaluation", "sealed", None))
    leaks["curriculum_accepts_eval_results"] = stored
    v = P.validate_proposal(reg, {"topology": "chain", "partition": "evaluation"})
    leaks["proposal_relabels_partition"] = int(v["accepted"])
    aud = reg.audit([{"entry": ev_id, "seed": 4}])
    leaks["training_log_audit_missed"] = int(not aud["violations"])
    ev = EV.Evaluator.start(sealed_dir=str(tmp_path / "sealed"))
    try:
        r1 = ev.part.call("open_episode", ev_id, 3)
        r2 = ev.call("open_episode", ev_id, 3, purpose="dev")
        r3 = ev.call("open_episode", ev_id, 2_100_000_000, purpose="dev")
        leaks["participant_opens_eval"] = int(bool(r1.get("ok")))
        leaks["eval_item_without_prereg"] = int(bool(r2.get("ok")))
        leaks["sealed_seed_by_hand"] = int(bool(r3.get("ok")))
        leaks["curriculum_view_served_sealed"] = ev.must("sealed_status")["curriculum_sealed_served"]
    finally:
        ev.close()
    leaks["salt_in_learner_env"] = int(any("SEALED" in k for k in os.environ))
    n = sum(leaks.values())
    record("T.7", {"evaluation_partition_leaks": n, "leak_vectors_probed": len(leaks)},
           notes=f"Leaks per vector: {leaks}")
    assert n == 0 and len(leaks) >= 9, leaks


def test_T8_generated_worlds_target_distinct_weaknesses(reference):
    """Design `curriculum_lab.T8_DESIGN` (fixed before this run, D-69 addendum): four planted weaknesses x five
    learner seeds; each pair trains two identical `TargetableMeta` learners for the same number of episodes on worlds
    generated for the curriculum's measured target or on uniformly drawn worlds, and compares their improvement on
    held-out probes of the target (common random numbers)."""
    r = CL.targeted_training_experiment(reference=reference, jobs=JOBS)
    record("T.8", {"targeted_weakness_improvement_pvalue": r["p_value"], "mean_improvement_difference":
                   r["mean_difference"], "distinct_targeted_weaknesses": len(r["targets"]), "pairs": len(r["pairs"]),
                   "pairs_targeted_better": sum(1 for x in r["pairs"] if x["difference"] > 0),
                   "pairs_untargeted_better": sum(1 for x in r["pairs"] if x["difference"] < 0)},
           notes=f"Design {r['design']}; pairs (planted, seed, target, improvement targeted / untargeted, arm after "
                 f"training targeted / untargeted): "
                 f"{[(x['planted'], x['seed'], x['target'], round(x['targeted'], 3), round(x['untargeted'], 3), x['arm_after']) for x in r['pairs']]}. "
                 "Learners are scripted strategy bandits over within-episode search strategies (controls).")
    assert r["p_value"] <= 0.05
    assert len(r["targets"]) >= 3
