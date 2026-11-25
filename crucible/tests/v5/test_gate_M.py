"""Gate M - Rapid scientific diversity (workstream H).

Every count is measured from the family registry and from admission runs on generated worlds
(`crucible.rapid.report.admission_report`, the ten Part 7.4 checks on seeds 0-2); nothing is declared. The counts
are recorded as they are, including where they are below the gate's thresholds."""
from __future__ import annotations

import ast
import os

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.rapid import families as F
from crucible.rapid import fingerprints as FP
from crucible.rapid import report as RP
from crucible.rapid.actions import KINDS
from crucible.rapid.engines import ENGINE_KINDS

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def adm():
    return RP.admission_report(seeds=(0, 1, 2))


@pytest.fixture(scope="module")
def holdout():
    return RP.admission_report(seeds=(10, 11, 12))


def _admitted(adm, split=None):
    return sorted(n for n, r in adm["reports"].items() if r["admitted"] and n in adm["counted"]
                  and (split is None or (r["split"] == "train") == (split == "train")))


def test_M1_admitted_structural_training_families(adm, holdout):
    train = _admitted(adm, "train")
    registered = F.families(split="train")
    assert set(train) <= set(registered)
    assert all(adm["reports"][n]["admitted"] for n in train)
    hold = [n for n in _admitted(holdout, "train")]
    record("M.1", {"admitted_train_families": len(train), "registered_train_families": len(registered),
                   "admitted_train_families_holdout_seeds": len(hold), "admission_seeds": "0,1,2",
                   "holdout_seeds": "10,11,12"},
           notes="Admitted = all ten Part 7.4 checks passed on seeds 0-2 and the fingerprint is distinct (not a "
                 f"cosmetic variant). Admitted: {train}. On held-out seeds 10-12, {len(hold)} of the training "
                 "families pass. The count is below the gate's 60; more families come in the next wave.")


def test_M2_independent_evaluation_families(adm):
    ev = _admitted(adm, "eval")
    for n in ev:
        assert F.FAMILIES[n].split in ("val", "test") and F.REGISTRY[n]["partition"] == "v5-eval-1"
    # evaluation families are separately partitioned: no training family shares a fingerprint with one
    for n in ev:
        for t in F.families(split="train"):
            assert FP.distance(adm["fingerprints"][n], adm["fingerprints"][t]) > 0
    from crucible.mechanisms.base import mech_families
    legacy = mech_families(split="val") + mech_families(split="test")
    record("M.2", {"eval_families": len(ev), "registered_eval_families": len(F.families(split="val")) +
                   len(F.families(split="test")), "legacy_4_0_val_test_families_untouched": len(legacy)},
           notes=f"v5 evaluation partition v5-eval-1, admitted: {ev}. 4.0's {len(legacy)} val/test mechanism "
                 "families are preserved where they were and are not counted here. Below the gate's 20.")


def test_M3_capability_categories(adm):
    cats = sorted({adm["reports"][n]["category"] for n in _admitted(adm)})
    assert set(cats) <= set(F.CATEGORIES)
    record("M.3", {"capability_categories_covered": len(cats), "categories": ",".join(cats)},
           notes="Categories (Part 8.1-8.10) of admitted families.")
    assert len(cats) == 10


def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_M4_independently_implemented_engines(adm):
    edir = os.path.join(ROOT, "src", "crucible", "rapid", "engines")
    mods = {n: os.path.join(edir, f"{n}.py") for n in ENGINE_KINDS}
    independent = []
    for name, path in mods.items():
        imp = _imports(path)
        others = {f".{o}" for o in mods if o != name} | {"..engines", "...mechanisms.scm", "...mechanisms.abm",
                                                          "...mechanisms.techsys"}
        assert not (imp & others), (name, imp & others)
        independent.append(name)
    with_families = sorted({adm["reports"][n]["engine"] for n in _admitted(adm)})
    from crucible.mechanisms.base import mech_families
    from crucible.mechanisms.sim import ensure_engines
    legacy = sorted(e for e in ensure_engines() if mech_families(engine=e))
    n = len(set(with_families) & set(independent)) + len(legacy)
    record("M.4", {"independent_engines": n, "v5_engines_with_admitted_families": len(with_families),
                   "legacy_4_0_engines_preserved": len(legacy), "v5_engines": ",".join(with_families),
                   "legacy_engines": ",".join(legacy)},
           notes="v5 engine modules import no other engine module (checked from their source); the 4.0 scm, abm "
                 "and techsys engines are preserved with their families.")
    assert n >= 6


def test_M5_structural_fingerprints():
    missing = []
    for n in F.REGISTRY:
        fp = F.REGISTRY[n]["fingerprint"]
        if not fp or any(k not in fp for k in FP.WEIGHTS) or not fp["motifs"] or not fp["action_interface"]:
            missing.append(n)
    record("M.5", {"families_missing_fingerprint": len(missing), "families": len(F.REGISTRY),
                   "fingerprint_components": len(FP.WEIGHTS)},
           notes="Fingerprints are measured from instantiated worlds (truth graph, motif census, public "
                 "observation and action structure) over three seeds.")
    assert not missing


def test_M6_action_modalities(adm):
    kinds = set()
    for n in _admitted(adm):
        kinds |= set(F.REGISTRY[n]["action_kinds"])
    assert kinds <= set(KINDS)
    record("M.6", {"action_modalities": len(kinds), "kinds": ",".join(sorted(kinds))},
           notes="Action kinds offered by admitted families.")
    assert len(kinds) >= 5


def test_M7_feasibility_validation(adm):
    reps = adm["reports"]
    validated = sorted(n for n, r in reps.items() if r["feasible"])
    bad_admitted = sorted(n for n, r in reps.items() if r["admitted"] and not r["feasible"])
    for n in validated:
        c = reps[n]["checks"]
        assert c["consistency"]["ok"] and c["conservation"]["ok"] and c["achievable"]["ok"]
    record("M.7", {"families_feasibility_validated": len(validated),
                   "feasibility_failures_admitted": len(bad_admitted), "families_checked": len(reps)},
           notes="Feasibility = consistency, conservation and achievability (oracle) checks passed on every "
                 "admission seed. Below the gate's 80 because there are fewer families than that so far.")
    assert not bad_admitted


def test_M8_mechanism_level_intervention_tests():
    from crucible.rapid import claims as CL
    with_tests, failures, ran = [], [], 0
    for n in sorted(F.FAMILIES):
        ok = True
        for seed in (0, 1, 2):
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
                   "intervention_tests_run": ran, "lookalike_rules_distinguished": look_ok},
           notes="Each family's claims are controlled interventions on copies of its worlds (common random "
                 "numbers), seeds 0-2. The four look-alike markets (identical interfaces) each show only their own "
                 f"price-response signature. Failures: {failures[:10]}. Below 80 families so far.")
    assert not failures


def _variant(name: str, kind: str):
    fam = F.FAMILIES[name]

    def build(rng, d, _b=fam.build):
        cfg = _b(rng, d)
        if kind == "param":
            _scale_numbers(cfg, 1.37)
        return cfg
    return F.Family(f"{name}__{kind}", fam.engine, fam.category, fam.item, fam.split, fam.doc, fam.distinct_because,
                    build, dict(fam.axes), fam.objective, fam.participants, fam.claims, fam.identify)


def _scale_numbers(cfg, f):
    """A parameter variant: every lever cost, price, capacity-like and rate-like number scaled (structure kept)."""
    for k in ("price", "toll", "c1", "c2", "unit_cost", "service_cost", "repair_cost", "probe_cost", "payoff",
              "lookup_cost", "action_cost", "lever_cost", "demand", "base"):
        if k in cfg:
            cfg[k] = [x * f for x in cfg[k]] if isinstance(cfg[k], list) else cfg[k] * f
    for nd in cfg.get("nodes", []):
        if "lever" in nd:
            nd["lever"]["cost"] *= f
        for k, v in list(nd.get("params", {}).items()):
            if isinstance(v, float) and k not in ("rho",):
                nd["params"][k] = v * f
    if "money" in cfg:
        cfg["money"]["price"] *= f


def test_Mn1_renamed_or_parameter_variants_do_not_count():
    originals = ["hc_confounded", "des_inventory", "net_routing", "posm_fault_diagnosis", "srch_field",
                 "coop_split_info"]
    added = []
    try:
        for n in originals:
            for kind in ("renamed", "param"):
                v = _variant(n, kind)
                F.FAMILIES[v.name] = v
                added.append(v.name)
        fps = {n: FP.family_fingerprint(n) for n in sorted(F.FAMILIES)}
        order = sorted(n for n in fps if n not in added) + added
        kept, dup = FP.counted(fps, order)
        counted_variants = [v for v in added if v in kept]
        assert all(dup.get(v) == v.split("__")[0] for v in added)
    finally:
        for v in added:
            F.FAMILIES.pop(v, None)
    import itertools
    real = {n: F.REGISTRY[n]["fingerprint"] for n in F.REGISTRY}
    dmin, pair = min((FP.distance(real[a], real[b]), (a, b)) for a, b in itertools.combinations(sorted(real), 2))
    record("M.n1", {"cosmetic_variants_counted": len(counted_variants), "variants_tried": len(added),
                    "min_fingerprint_distance": float(dmin), "closest_pair": f"{pair[0]}~{pair[1]}"},
           notes="Renamed copies and parameter variants (all costs, prices, capacities and coefficients scaled by "
                 "1.37) of six families on all five engines get distance 0 to their originals and are not counted.")
    assert not counted_variants and dmin > 0


def test_Mx_distinctness_registry_is_machine_readable(adm):
    """Every family states why it is distinct, and the registry carries the measured nearest family and the
    fingerprint components that differ from it."""
    for n, d in adm["distinctness"].items():
        assert d["distinct_because"] and d["nearest"] and d["distance"] > 0 and d["differs_in"]
    assert adm["static_isolation"]["ok"]
