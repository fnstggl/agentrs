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
    its measured participant fingerprint differs from the population-free world and the outcome of the same scripted
    constant policy differs (paired by structure and seed)."""
    from crucible.rapid.fingerprints import instance_fingerprint
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
                fps.add(str(instance_fingerprint(env.env).get("participants")))
                rec = run_episode(env, CL.PlantedAdapter(0), learn=False, rng=rng)
                vals.append(rec["value"])
        outcome[kind] = np.array(vals)
        parts[kind] = fps
    tested = ["none"] + [k for k in C.POPULATIONS if k != "none" and parts[k] != parts["none"] and
                         float(np.mean(np.abs(outcome[k] - outcome["none"]))) > 1e-6]
    record("T.2", {"population_configurations_tested": len(tested), "population_configurations_defined":
                   len(C.POPULATIONS), "worlds_built": int(sum(len(v) for v in outcome.values()))},
           notes=f"Configurations with measured participant and outcome differences: {tested}; participant "
                 f"fingerprints {({k: sorted(v) for k, v in parts.items()})}")
    assert len(tested) >= 3


def test_T3_independently_varied_objectives(generated):
    """Goals of every type of the independent goal generator (`crucible.rapid.goals`) are drawn from the public face
    of generated worlds of different structures; a type counts when it was drawn and validated feasible on at least
    three distinct structures. Independence: the goal-type distribution does not depend on the structure's category
    (chi-square test of independence)."""
    from crucible.rapid import goals as G
    from crucible.rapid import goals_eval as GE
    feasible = collections.defaultdict(set)
    table = collections.Counter()
    for r in generated[:24]:
        e = r["entry"]
        for t in G.GOAL_TYPES:
            env = rapid_world(e, 3, {}, None)
            try:
                g = G.generate(env, 0.4, 5, t)
            except ValueError:
                continue
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


def test_T4_validated_generated_worlds(generated):
    """Full Part 7.4 admission (`crucible.rapid.admission.seed_report`, eleven checks) of generated worlds; a world
    (structure, seed) is validated when every check passes on it."""
    from crucible.research.v5.genadmit import admit_generated
    res = admit_generated([r["structure"] for r in generated], seeds=(0, 1, 2), jobs=JOBS)
    fails = collections.Counter(c for w in res["worlds"] for c, ok in w["checks"].items() if not ok)
    record("T.4", {"generated_worlds_validated": res["validated"], "generated_worlds_checked": len(res["worlds"]),
                   "structures_with_a_validated_world": res["structures_validated"]},
           notes=f"Failed checks over all worlds: {dict(fails)}")
    assert res["validated"] >= 50, dict(fails)


@pytest.fixture(scope="module")
def reference():
    return CL.reference_profile()


def test_T5_curriculum_selection_from_measured_behaviour(reference):
    r = CL.selection_experiment(seeds=(0, 1, 2), n_items=40, reference=reference)
    record("T.5", {"weakness_selection_pvalue": r["p_value"], "mean_rank_of_measured_weakness": r["mean_rank"],
                   "runs": len(r["runs"]), "categories": len(CU.WEAKNESSES)},
           notes=f"Per run (planted, seed, measured weakest category, its rank in the curriculum weights): "
                 f"{[(x['planted'], x['seed'], x['weakest_measured'], x['rank']) for x in r['runs']]}")
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
    r = CL.targeted_training_experiment(seeds=(0, 1, 2), reference=reference)
    record("T.8", {"targeted_weakness_improvement_pvalue": r["p_value"], "mean_improvement_difference":
                   r["mean_difference"], "distinct_targeted_weaknesses": len(r["targets"]), "pairs": len(r["pairs"])},
           notes=f"Pairs (planted, seed, target, improvement targeted / untargeted): "
                 f"{[(x['planted'], x['seed'], x['target'], round(x['targeted'], 3), round(x['untargeted'], 3)) for x in r['pairs']]}")
    assert r["p_value"] <= 0.05
    assert len(r["targets"]) >= 3
