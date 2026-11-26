"""Gate N (MASTER_SPEC Part 17): all fifteen Tier 2 digital domains (Part 9), measured on built and run worlds.

Every registered Part 9 domain (`crucible.domains.base.part9_domains()`; test fixtures excluded) is checked by
`crucible.domains.base.check`: its configurations are built on three seeds (structure measured on the built world,
magnitudes reduced to presence/absence), its tool-only scripted reference, its null baseline and its degenerate
probes run through the learner's own path `crucible.experience.make(2, ...)` (observation_only, reward 0.0) for the
domain's evaluation horizon, and the evaluator is re-run on a world rebuilt from the event log alone. A Part 9
section counts when one of its domains satisfies the criterion; a section without a domain counts as failing. While
fewer than fifteen sections have a domain, every test here FAILS (it never skips).

Each test also runs a positive control on the framework fixture (tests/v5/_domain_fixture.py, a civilisation-backed
household-retail domain that is never counted): the measurement must detect a planted defect, so a criterion cannot
hold vacuously.

    N.1   domains_implemented >= 15                  every N.2-N.n1 criterion holds for the section's domain
    N.2   min_train_configs_per_domain >= 3          structurally distinct training configurations (built worlds)
    N.3   domains_with_executable_mechanisms >= 15, missing_required_mechanics == 0   recorded events per mechanic
    N.4   domains_with_independent_evaluator >= 15   evaluator-side, replay-equal, separates reference from null
    N.5   domains_with_successful_reference >= 15    tool-only reference succeeds on a majority of instances
    N.6   min_seeds_per_config >= 3                  distinct procedural instances of every configuration
    N.7   domains_with_compositional_eval / domains_with_structural_eval >= 15   unfamiliarity measured on atoms
    N.n1  reference_paths_exposed == 0               over all 15 sections (and the scanner detects plants)
"""
import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _domain_fixture as F  # noqa: E402
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.domains import base  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_N.py"
LABELS = {"scripted": True, "reference": "scripted tool-only reference policies (crucible.domains.base.drive)"}


def _metrics():
    reps = base.gate_reports()
    m = base.gate_metrics(reps)
    per = {n: base.verdicts(r) for n, r in reps.items()}
    return reps, m, per


def _context(m, per, keys):
    out = {"domains_registered": m["domains_registered"], "sections_with_domain": m["sections_with_domain"],
           "sections_missing": m["sections_missing"]}
    for k in keys:
        out[f"per_domain_{k}"] = ";".join(f"{n}={v[k]}" for n, v in sorted(per.items())) or "none"
    return out


@pytest.fixture(scope="module")
def fixture():
    d = F.register()
    yield d
    base.unregister(F.NAME)


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_1_all_fifteen_domains_implemented(fixture):
    reps, m, per = _metrics()
    # positive control: the fixture passes every per-domain criterion, and a planted defect (its evaluator never
    # reports success) makes it not implemented
    good = base.verdicts(base.check(F.DOMAIN, tier3=False))
    broken = replace(F.DOMAIN, name="fixture_broken",
                     evaluate=lambda w, p, o: {"success": False, "score": 0.0, "details": {}})
    base.register(broken)
    try:
        bad = base.verdicts(base.check(broken, tier3=False, structure_seeds=(0, 1, 2)))
    finally:
        base.unregister("fixture_broken")
    failing = {n: ",".join(k for k, v in vv.items() if v is False) for n, vv in per.items() if not vv["implemented"]}
    metrics = {"domains_implemented": m["domains_implemented"], **_context(m, per, ["implemented"]),
               "failing_criteria": ";".join(f"{n}:{v}" for n, v in sorted(failing.items())) or "none",
               "control_fixture_implemented": good["implemented"], "control_broken_evaluator_implemented":
               bad["implemented"]}
    evidence.record("N.1", metrics, command=CMD, labels=LABELS)
    assert good["implemented"] is True and bad["implemented"] is False
    assert m["domains_implemented"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_2_multiple_structural_configurations(fixture):
    reps, m, per = _metrics()
    ctl = base.structure_report(F.DOMAIN)
    F.SHAPES["ring_standard6"] = {"chain": "standard", "regions": 6, "links": "ring"}
    try:
        planted = replace(F.DOMAIN, name="fixture_numbers", configs=F.DOMAIN.configs + (
            base.DomainConfig("ring_standard6", "train", dict(F.SHAPES["ring_standard6"]), "only more regions"),))
        base.register(planted)
        ctl2 = base.structure_report(planted, seeds=(0,))
    finally:
        base.unregister("fixture_numbers")
        F.SHAPES.pop("ring_standard6", None)
    metrics = {"min_train_configs_per_domain": m["min_train_configs_per_domain"],
               **_context(m, per, ["train_structures"]),
               "control_fixture_train_structures": ctl["train_structures"],
               "control_numbers_only_config_detected": "ring_standard=ring_standard6" in ctl2["same_structure_pairs"],
               "control_numbers_only_train_structures": ctl2["train_structures"]}
    evidence.record("N.2", metrics, command=CMD, labels=LABELS)
    assert ctl["train_structures"] == 3 and metrics["control_numbers_only_config_detected"]
    assert ctl2["train_structures"] == 3 and ctl2["train_configs"] == 4
    assert m["min_train_configs_per_domain"] >= 3, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_3_executable_mechanisms(fixture):
    reps, m, per = _metrics()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference")
    # positive control: a probe for an event type the world never records is reported missing
    planted = replace(F.DOMAIN, mechanics=F.MECHANICS[:-1] + (base.Mechanic("Production", ("make.never_recorded",)),))
    env = base.make_env(F.NAME, "ring_standard", 0)
    env.advance(2 * 86400)
    orig = base.mechanics_exercised(env.world, F.DOMAIN, 0)
    miss = base.mechanics_exercised(env.world, planted, 0)
    env.close()
    metrics = {"domains_with_executable_mechanisms": m["domains_with_executable_mechanisms"],
               "missing_required_mechanics": m["missing_required_mechanics"],
               **_context(m, per, ["mechanics_ok", "mechanics_missing"]),
               "missing_by_domain": ";".join(f"{n}:{'|'.join(r['mechanics']['missing'])}" for n, r in reps.items()
                                             if r["mechanics"]["missing"]) or "none",
               "control_fixture_mechanics_exercised": ref["mechanics"]["exercised"],
               "control_planted_missing": ",".join(miss["missing"]) or "none"}
    evidence.record("N.3", metrics, command=CMD, labels=LABELS)
    assert ref["mechanics"]["missing"] == [] and "Production" in miss["missing"]
    assert "Production" not in orig["missing"] and orig["counts"]["Production"] > 0
    assert m["domains_with_executable_mechanisms"] >= 15 and m["missing_required_mechanics"] == 0, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_4_independent_outcome_evaluation(fixture):
    reps, m, per = _metrics()
    ev = base.evaluator_report(F.DOMAIN, "ring_standard", 0)
    metrics = {"domains_with_independent_evaluator": m["domains_with_independent_evaluator"],
               **_context(m, per, ["evaluator_ok"]),
               "evaluator_replay_equal": ";".join(f"{n}={(r.get('evaluator') or {}).get('replay_equal')}"
                                                  for n, r in sorted(reps.items())) or "none",
               "control_fixture_evaluator_ok": ev["ok"], "control_fixture_replay_equal": ev["replay_equal"]}
    evidence.record("N.4", metrics, command=CMD, labels=LABELS)
    assert ev["ok"] and ev["replay_equal"] and ev["evaluator_hidden"]
    assert m["domains_with_independent_evaluator"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_5_successful_feasible_reference(fixture):
    reps, m, per = _metrics()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference")
    null = base.cached_run(F.NAME, "ring_standard", 0, "null")

    class Peeking(F.RetailReference):
        def plan(self, obs):
            return [] if self.world.state is None else []
    flagged = base.reads_world(Peeking("p:trader", {}))
    metrics = {"domains_with_successful_reference": m["domains_with_successful_reference"],
               **_context(m, per, ["reference_ok"]),
               "reference_success_rates": ";".join(f"{n}={r['reference_successes']}/{r['reference_runs']}"
                                                   for n, r in sorted(reps.items())) or "none",
               "control_fixture_reference_success": ref["success"], "control_fixture_null_success": null["success"],
               "control_world_reading_reference_flagged": bool(flagged)}
    evidence.record("N.5", metrics, command=CMD, labels=LABELS)
    assert ref["success"] is True and null["success"] is False and flagged
    assert m["domains_with_successful_reference"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_6_procedural_instances(fixture):
    reps, m, per = _metrics()
    ctl = base.structure_report(F.DOMAIN)
    metrics = {"min_seeds_per_config": m["min_seeds_per_config"], **_context(m, per, ["min_instances"]),
               "control_fixture_min_instances": min(ctl["instances"].values())}
    evidence.record("N.6", metrics, command=CMD, labels=LABELS)
    assert min(ctl["instances"].values()) == 3
    assert m["min_seeds_per_config"] >= 3, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_7_unfamiliar_evaluation_configurations(fixture):
    reps, m, per = _metrics()
    ctl = base.structure_report(F.DOMAIN)
    # positive control: an "evaluation" configuration equal in structure to a training one is neither
    t = ctl["fingerprints"]["ring_standard"]
    same = base.unfamiliarity([ctl["fingerprints"][c] for c in ("ring_standard", "line_linear", "star_diamond")], t)
    metrics = {"domains_with_compositional_eval": m["domains_with_compositional_eval"],
               "domains_with_structural_eval": m["domains_with_structural_eval"],
               **_context(m, per, ["compositional_eval", "structural_eval"]),
               "control_fixture_compositional": ctl["compositional_eval"],
               "control_fixture_structural": ctl["structural_eval"],
               "control_copy_of_training_unfamiliar": bool(same["compositional"] or same["structural"])}
    evidence.record("N.7", metrics, command=CMD, labels=LABELS)
    assert ctl["compositional_eval"] == 1 and ctl["structural_eval"] == 1
    assert not same["compositional"] and not same["structural"] and same["equal_to_training"]
    assert m["domains_with_compositional_eval"] >= 15 and m["domains_with_structural_eval"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_n1_reference_not_exposed(fixture):
    reps, m, per = _metrics()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference")
    traj = ref["drive"]["agents"][ref["seats"][0]]["tools"]
    pol = F.RetailReference("p:trader", {})
    canaries = base.reference_canaries(F.DOMAIN, pol)
    clean = base.exposure([F.objective(None, "p:trader", 1)["text"]], canaries, traj)
    # positive controls: an objective that spells out the reference's path, a tool doc naming the reference
    seq = []
    for t in traj:
        if t not in seq:
            seq.append(t)
    spelled = base.exposure([f"Do this: {', then '.join(seq[:4])}."], canaries, traj)
    named = base.exposure([f'{{"name": "help.solve", "description": "runs {canaries[0]}"}}'], canaries, traj)
    metrics = {"reference_paths_exposed": m["reference_paths_exposed"],
               "domains_scanned_for_exposure": m["domains_scanned_for_exposure"], **_context(m, per, ["exposed"]),
               "sandbox_imports_crucible": base.sandbox_can_import_crucible(),
               "control_fixture_exposed": clean["exposed"], "control_spelled_out_path_detected": spelled["exposed"],
               "control_named_reference_detected": named["exposed"]}
    evidence.record("N.n1", metrics, command=CMD, labels=LABELS)
    assert clean["exposed"] == 0 and spelled["exposed"] >= 1 and named["exposed"] >= 1
    assert metrics["sandbox_imports_crucible"] is False
    assert m["domains_scanned_for_exposure"] >= 15, metrics
    assert m["reference_paths_exposed"] == 0, metrics
