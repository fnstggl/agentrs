"""Gate M - Rapid scientific diversity (workstream H).

Every count is measured from the family registry and from admission runs on generated worlds
(`crucible.rapid.report.robust_report`: the eleven Part 7.4 checks of `crucible.rapid.admission` - including the
fixed-budget dominance search against an adaptive public reference and the family's own mechanism claims - on the
admission seeds 0-2 and, separately, on the held-out seeds 10-12). A family counts only if it is admitted on both
seed sets and its structural fingerprint is not a duplicate. Nothing is declared; the counts are recorded as they
are, including where they are below the gate's thresholds.

Heavy: run through the slot scheduler, e.g.
    python -m crucible.scale.slots --cpu 4 -- env CRUCIBLE_RAPID_JOBS=4 python -m pytest tests/v5/test_gate_M.py
(the admission reports are cached under $CRUCIBLE_CACHE/rapid by source digest)."""
from __future__ import annotations

import ast
import itertools
import os

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP
from crucible.rapid import report as RP
from crucible.rapid.actions import KINDS
from crucible.rapid.engines import ENGINE_KINDS, SPEC_ENGINE_KINDS

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS = int(os.environ.get("CRUCIBLE_RAPID_JOBS", "1"))


@pytest.fixture(scope="module")
def rob():
    return RP.robust_report(seeds=(0, 1, 2), holdout=(10, 11, 12), jobs=JOBS)


@pytest.fixture(scope="module")
def adm(rob):
    return rob["admission"]


def test_M1_admitted_structural_training_families(rob, adm):
    train = rob["robust_train"]
    registered = F.families(split="train")
    assert set(train) <= set(registered)
    for n in train:
        assert adm["reports"][n]["admitted"] and rob["holdout"]["reports"][n]["admitted"] and n in adm["counted"]
    seeds_only = sorted(n for n in registered if adm["reports"][n]["admitted"])
    verified = sum(1 for n in train if (adm["distinctness"][n].get("verification") or {}).get("ok"))
    record("M.1", {"admitted_train_families": len(train), "registered_train_families": len(registered),
                   "admitted_train_families_seeds_0_2_only": len(seeds_only),
                   "admitted_with_intervention_verified_difference": verified,
                   "admission_seeds": "0,1,2", "holdout_seeds": "10,11,12", "admission_checks": 11},
           notes="Admitted = all eleven Part 7.4 checks (with the fixed-budget dominance search against an adaptive "
                 "public reference and the family's own intervention claims) passed on seeds 0-2 AND on held-out "
                 f"seeds 10-12, fingerprint not a duplicate. Admitted: {train}.")


def test_M2_independent_evaluation_families(rob, adm):
    ev = rob["robust_eval"]
    for n in ev:
        assert F.FAMILIES[n].split in ("val", "test") and F.REGISTRY[n]["partition"] == "v5-eval-1"
    # separately partitioned: no training family shares a fingerprint with an evaluation family
    for n in ev:
        for t in F.families(split="train"):
            assert FP.distance(adm["fingerprints"][n], adm["fingerprints"][t]) > 0
    from crucible.mechanisms.base import mech_families
    from crucible.mechanisms.sim import ensure_engines
    ensure_engines()
    legacy = mech_families(split="val") + mech_families(split="test")
    assert not set(legacy) & set(F.FAMILIES)
    record("M.2", {"eval_families": len(ev), "registered_eval_families": len(F.families(split="val")) +
                   len(F.families(split="test")), "legacy_4_0_val_test_families_untouched": len(legacy)},
           notes=f"v5 evaluation partition v5-eval-1, admitted on seeds 0-2 and 10-12: {ev}. The {len(legacy)} 4.0 "
                 "val/test mechanism families are preserved in crucible.mechanisms and are not counted here.")


def test_M3_capability_categories(rob, adm):
    cats = sorted({adm["reports"][n]["category"] for n in rob["robust"]})
    assert set(cats) <= set(F.CATEGORIES)
    reg = sorted({f.category for f in F.FAMILIES.values()})
    record("M.3", {"capability_categories_covered": len(cats), "categories": ",".join(cats),
                   "categories_registered": len(reg)},
           notes="Categories (Part 8.1-8.10) of robustly admitted families. Missing: "
                 f"{sorted(set(F.CATEGORIES) - set(cats))}.")


def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_M4_independently_implemented_engines(rob, adm):
    edir = os.path.join(ROOT, "src", "crucible", "rapid", "engines")
    mods = {n: os.path.join(edir, f"{n}.py") for n in ENGINE_KINDS}
    for name, path in mods.items():
        imp = _imports(path)
        others = {f".{o}" for o in mods if o != name} | {"..engines", "...mechanisms.scm", "...mechanisms.abm",
                                                          "...mechanisms.techsys"}
        assert not (imp & others), (name, imp & others)
    with_families = sorted({adm["reports"][n]["engine"] for n in rob["robust"]})
    from crucible.mechanisms.base import mech_families
    from crucible.mechanisms.sim import ensure_engines
    legacy = sorted(e for e in ensure_engines() if mech_families(engine=e))
    n = len(with_families) + len(legacy)
    record("M.4", {"independent_engines": n, "v5_engines_with_admitted_families": len(with_families),
                   "legacy_4_0_engines_preserved": len(legacy), "v5_engines": ",".join(with_families),
                   "legacy_engines": ",".join(legacy), "v5_engines_registered": len(ENGINE_KINDS)},
           notes="v5 engine modules import no other engine module (checked from their source); the 4.0 scm, abm "
                 "and techsys engines are preserved with their families. Counted v5 engines have a robustly "
                 "admitted family.")


def _has_number(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, dict):
        return any(_has_number(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_number(v) for v in x)
    return isinstance(x, str) and (x.startswith("2^") or x.startswith("<="))


def test_M5_structural_fingerprints():
    missing, numeric = [], []
    for n in F.REGISTRY:
        fp = F.REGISTRY[n]["fingerprint"]
        if not fp or any(k not in fp for k in FP.WEIGHTS) or not fp["motifs"] or not fp["action_interface"]:
            missing.append(n)
        if any(_has_number(fp[k]) for k in FP.WEIGHTS):
            numeric.append(n)
    record("M.5", {"families_missing_fingerprint": len(missing), "families": len(F.REGISTRY),
                   "fingerprint_components": len(FP.WEIGHTS), "fingerprints_with_numeric_components": len(numeric),
                   "fingerprint_version": FP.FP_VERSION},
           notes="Fingerprints are measured from instantiated worlds (core causal graph without distractors, motif "
                 "census, temporal and observation structure, action interface, objective, participants, engine "
                 "and scenario) over three seeds; every component is categorical - no counts, sizes, horizons or "
                 "buckets.")
    assert not missing and not numeric


def test_M6_action_modalities(rob):
    kinds = set()
    for n in rob["robust"]:
        kinds |= set(F.REGISTRY[n]["action_kinds"])
    assert kinds <= set(KINDS)
    record("M.6", {"action_modalities": len(kinds), "kinds": ",".join(sorted(kinds))},
           notes="Action kinds offered by robustly admitted families.")


def test_M7_feasibility_validation(rob, adm):
    reps, hold = adm["reports"], rob["holdout"]["reports"]
    validated = sorted(n for n, r in reps.items() if r["feasible"] and hold[n]["feasible"])
    bad_admitted = sorted(n for n, r in reps.items() if r["admitted"] and not r["feasible"])
    for n in validated:
        c = reps[n]["checks"]
        assert c["consistency"]["ok"] and c["conservation"]["ok"] and c["achievable"]["ok"]
    record("M.7", {"families_feasibility_validated": len(validated),
                   "feasibility_failures_admitted": len(bad_admitted), "families_checked": len(reps)},
           notes="Feasibility = consistency, conservation and achievability (an evaluator-side upper reference "
                 "clearly improves on doing nothing) on every seed of both seed sets.")
    assert not bad_admitted


def test_M8_mechanism_level_intervention_tests():
    from crucible.rapid import claims as CL
    with_tests, failures, ran = [], [], 0
    for n in sorted(F.FAMILIES):
        ok = True
        for seed in (0, 1, 2, 10, 11, 12):
            for name, r in CL.run(n, seed).items():
                ran += 1
                if not r.get("ok"):
                    ok = False
                    failures.append(f"{n}:{seed}:{name}")
        if ok and F.FAMILIES[n].claims:
            with_tests.append(n)
    lk = [CL.lookalike_matrix(s) for s in (0, 1, 2)]
    look_ok = all(m["all_own"] and m["distinct"] and m["same_interface"] for m in lk)
    if not look_ok:
        failures.append("lookalike_matrix")
    record("M.8", {"families_with_intervention_tests": len(with_tests), "intervention_test_failures": len(failures),
                   "intervention_tests_run": ran, "families_registered": len(F.FAMILIES),
                   "lookalike_rules_distinguished": look_ok},
           notes="Each family's claims are controlled interventions on copies of its worlds (common random "
                 "numbers), seeds 0-2 and 10-12; a family whose claims fail is not admitted (admission's mechanism "
                 f"check). Failures: {failures[:12]}.")


AXIS_MULT = {"horizon": 2, "graph_size": 2, "n_variables": 3, "hidden_state_complexity": 2,
             "action_complexity": 2, "resources": 2, "observation_noise": 3, "experiment_cost": 3,
             "opponent_capability": 2}
STRUCTURAL_INTS = ("horizon", "engine", "warmup", "at", "lag", "n", "endowment")


def _raised(v, ax):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return max(v + 1, int(v * AXIS_MULT[ax]))
    return (v * AXIS_MULT[ax]) if v else 1.0


def _scale_numbers(x, f):
    """A parameter variant of a config: every float scaled (recursively); integers (sizes, lags, ticks) kept."""
    if isinstance(x, dict):
        return {k: (v if k in STRUCTURAL_INTS else _scale_numbers(v, f)) for k, v in x.items()}
    if isinstance(x, list):
        return [_scale_numbers(v, f) for v in x]
    if isinstance(x, float):
        return x * f
    return x


def _variant(name: str, kind: str, axis: str | None = None):
    fam = F.FAMILIES[name]
    raised = {axis: _raised(fam.axes[axis], axis)} if axis else {}

    def build(rng, d, _b=fam.build):
        cfg = _b(rng, dict(d, **raised))
        if kind == "param":
            cfg = _scale_numbers(cfg, 1.37)
        return cfg
    vname = f"{name}__{kind}" + (f"__{axis}" if axis else "")
    return F.Family(vname, fam.engine, fam.category, fam.item, fam.split, fam.doc, fam.distinct_because, build,
                    dict(fam.axes), fam.objective, fam.participants, fam.claims, fam.identify)


def test_Mn1_renamed_or_parameter_variants_do_not_count():
    """A renamed copy, a parameter variant (every float of the configuration scaled by 1.37) and a variant at each
    raised difficulty axis (horizon, size, distractors, noise, cost, hidden states, options, resources) of every
    registered family: each has fingerprint distance exactly 0 to its family and is not counted."""
    originals = sorted(F.FAMILIES)
    added, crashed = [], []
    try:
        for n in originals:
            kinds = [("renamed", None), ("param", None)] + [("axis", ax) for ax in F.FAMILIES[n].axes]
            for kind, ax in kinds:
                v = _variant(n, kind, ax)
                F.FAMILIES[v.name] = v
                try:
                    FP.family_fingerprint(v.name)
                    added.append(v.name)
                except Exception as e:  # noqa: BLE001 - a variant the builder cannot make is reported
                    crashed.append(f"{v.name}: {type(e).__name__}: {e}")
                    F.FAMILIES.pop(v.name)
        fps = {n: FP.family_fingerprint(n) for n in sorted(F.FAMILIES)}
        order = sorted(n for n in fps if n not in added) + added
        kept, dup = FP.counted(fps, order)
        counted_variants = [v for v in added if v in kept]
        nonzero = [v for v in added if FP.distance(fps[v], fps[v.split("__")[0]]) != 0.0]
    finally:
        for v in added:
            F.FAMILIES.pop(v, None)
    real = {n: F.REGISTRY[n]["fingerprint"] for n in F.REGISTRY}
    dmin, pair = min((FP.distance(real[a], real[b]), (a, b)) for a, b in itertools.combinations(sorted(real), 2))
    record("M.n1", {"cosmetic_variants_counted": len(counted_variants), "variants_tried": len(added),
                    "variants_at_nonzero_distance": len(nonzero), "variants_not_buildable": len(crashed),
                    "min_fingerprint_distance": float(dmin), "closest_pair": f"{pair[0]}~{pair[1]}"},
           notes="Variants: renamed copy, parameter variant (all floats x1.37) and one variant per raised difficulty "
                 f"axis of every registered family. Nonzero: {nonzero[:5]}. Not buildable: {crashed[:5]}. "
                 "Distance-0 variants are dropped by `fingerprints.counted`; real families are all at distance > 0.")
    assert not counted_variants and not nonzero and dmin > 0


def test_Mx_distinctness_registry_is_machine_readable(adm):
    """Every family states why it is distinct; the registry carries the measured nearest family, the fingerprint
    components that differ and how, and the intervention that verified the difference (or that it could not)."""
    for n, d in adm["distinctness"].items():
        assert d["distinct_because"] and d["nearest"] and d["distance"] > 0 and d["differs_in"]
        assert "verification" in d and "recreates_4_0_sealed" in d
    assert adm["static_isolation"]["ok"]
    from crucible.rapid.admission import EVALUATOR_MODULES
    assert "crucible.rapid.dominance" in EVALUATOR_MODULES and "crucible.rapid.distinct" in EVALUATOR_MODULES
    assert set(SPEC_ENGINE_KINDS.values()) <= set(ENGINE_KINDS)
