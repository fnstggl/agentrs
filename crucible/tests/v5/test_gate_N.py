"""Gate N (MASTER_SPEC Part 17): all fifteen Tier 2 digital domains (Part 9), measured on built and run worlds.

Every registered Part 9 domain (`crucible.domains.base.part9_domains()`; test fixtures excluded) is checked by
`crucible.domains.base.check`: its configurations are built on three seeds (structure measured on the built world,
magnitudes reduced to presence/absence, genesis content compared for instances), its tool-only scripted reference
(in its own process, with the objective's public text only), its null baseline and its degenerate probes run through
the learner's own path `crucible.experience.make(2, ...)` (observation_only, reward 0.0) for the domain's evaluation
horizon on EVERY configuration, and the evaluator is re-run on a world rebuilt from the event log alone and with the
autonomous participants' decision processes frozen. A Part 9 section counts when one of its domains satisfies the
criterion; a section without a domain counts as failing. While fewer than fifteen sections have a domain, every test
here FAILS (it never skips).

Each test also runs positive controls on the framework fixture (tests/v5/_domain_fixture.py, a civilisation-backed
household-retail domain that is never counted): the measurement must detect planted defects (the verifier's probes
of the wave 3b repair pass among them), so a criterion cannot hold vacuously.

    N.1   domains_implemented >= 15                  every N.2-N.n1 criterion + generated objectives, clean runs
    N.2   min_train_configs_per_domain >= 3          structurally distinct training configurations (built worlds)
    N.3   domains_with_executable_mechanisms >= 15, missing_required_mechanics == 0   verified recorded evidence
    N.4   domains_with_independent_evaluator >= 15   evaluator-side, replay-equal, needs the autonomous participants
    N.5   domains_with_successful_reference >= 15    isolated tool-only reference succeeds on a majority of instances
    N.6   min_seeds_per_config >= 3                  distinct generated instances (genesis content) per configuration
    N.7   domains_with_compositional_eval / domains_with_structural_eval >= 15   unfamiliar, clean and feasible
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
LABELS = {"scripted": True, "reference": "scripted tool-only reference policies in their own process "
                                        "(crucible.domains.base.IsolatedPolicy, driven by base.drive)"}
_CTL: dict = {}
CTL_DAYS = 7.0          # the fixture's controls run 7 of its 21 horizon days (its reference succeeds within 3)


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


def _control():
    """The fixture's full check (once per test process)."""
    if "rep" not in _CTL:
        _CTL["rep"] = base.check(F.DOMAIN, tier3=False, days=CTL_DAYS)
    return _CTL["rep"]


@pytest.fixture(scope="module")
def fixture():
    d = F.register()
    yield d
    base.unregister(F.NAME)


def _register(d):
    base.register(d)
    return d


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_1_all_fifteen_domains_implemented(fixture):
    reps, m, per = _metrics()
    # positive control: the fixture passes every per-domain criterion; a planted defect (its evaluator never reports
    # success: infeasible) makes it not implemented
    good = base.verdicts(_control())
    broken = _register(replace(F.DOMAIN, name="fixture_broken", goal=None, objective=F.objective,
                               evaluate=F.never_evaluator))
    try:
        bad = base.verdicts(base.check(broken, tier3=False, seeds=(0,), all_configs=False, days=CTL_DAYS))
    finally:
        base.unregister("fixture_broken")
    failing = {n: ",".join(k for k, v in vv.items() if v is False) for n, vv in per.items() if not vv["implemented"]}
    metrics = {"domains_implemented": m["domains_implemented"], **_context(m, per, ["implemented"]),
               "domains_with_generated_objectives": m["domains_with_generated_objectives"],
               "failing_criteria": ";".join(f"{n}:{v}" for n, v in sorted(failing.items())) or "none",
               "control_fixture_implemented": good["implemented"], "control_broken_evaluator_implemented":
               bad["implemented"]}
    evidence.record("N.1", metrics, command=CMD, labels=LABELS)
    assert good["implemented"] is True, {k: v for k, v in good.items() if v is False}
    assert bad["implemented"] is False
    assert m["domains_implemented"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_2_multiple_structural_configurations(fixture):
    reps, m, per = _metrics()
    ctl = base.structure_report(F.DOMAIN)
    F.SHAPES["ring_standard6"] = {"chain": "standard", "regions": 6, "links": "ring"}
    try:
        planted = _register(replace(F.DOMAIN, name="fixture_numbers", configs=F.DOMAIN.configs + (
            base.DomainConfig("ring_standard6", "train", dict(F.SHAPES["ring_standard6"]), "only more regions"),)))
        ctl2 = base.structure_report(planted, seeds=(0,))
    finally:
        base.unregister("fixture_numbers")
        F.SHAPES.pop("ring_standard6", None)
    # verifier probes p1 (b) and (c): one world declared five ways through a hook reading the spec; inert modules
    declared = _register(replace(F.DOMAIN, name="fixture_declared", build=_same_world,
                                 structure=_hook_reads_declaration))
    inert = _register(replace(F.DOMAIN, name="fixture_inert_mods", build=_inert_modules, structure=None))
    try:
        ctl3 = base.structure_report(declared, seeds=(0,))
        ctl4 = base.structure_report(inert, seeds=(0,))
    finally:
        base.unregister("fixture_declared")
        base.unregister("fixture_inert_mods")
    metrics = {"min_train_configs_per_domain": m["min_train_configs_per_domain"],
               **_context(m, per, ["train_structures"]),
               "control_fixture_train_structures": ctl["train_structures"],
               "control_numbers_only_config_detected": "ring_standard=ring_standard6" in ctl2["same_structure_pairs"],
               "control_numbers_only_train_structures": ctl2["train_structures"],
               "control_declared_hook_train_structures": ctl3["train_structures"],
               "control_inert_modules_train_structures": ctl4["train_structures"]}
    evidence.record("N.2", metrics, command=CMD, labels=LABELS)
    assert ctl["train_structures"] == 3 and metrics["control_numbers_only_config_detected"]
    assert ctl2["train_structures"] == 3 and ctl2["train_configs"] == 4
    assert ctl3["train_structures"] == 0 and "structure_hook" in ctl3["errors"]
    assert ctl4["train_structures"] == 1
    assert m["min_train_configs_per_domain"] >= 3, metrics


def _same_world(config, seed, overrides):
    return F.build(replace(config, name="ring_standard"), seed, overrides)


_FLAGS = {"ring_standard": "x", "line_linear": "y", "star_diamond": "z", "ring_linear": "y", "complete_wide": "w"}


def _hook_reads_declaration(world):
    return {"a": _FLAGS[world.spec.params["domain_config"]]}


def _inert_modules(config, seed, overrides):
    extra = {"ring_standard": [], "line_linear": ["calendar"], "star_diamond": ["support"],
             "ring_linear": ["calendar", "support"], "complete_wide": ["studio"]}[config.name]
    r = F.build(replace(config, name="ring_standard"), seed, overrides)
    for x in extra:
        r["module_list"].append(x)
        r["modules"].setdefault(x, {})
    return r


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_3_executable_mechanisms(fixture):
    reps, m, per = _metrics()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference", CTL_DAYS)
    # positive controls on a NULL run (the seat idles): a probe for an event type the world never records is
    # missing; every mechanic mapped to bank.txn or to the scheduler's events, or evidenced by a constant, counts
    # nothing (verifier probes p9, p4 'count')
    r = base.run(F.NAME, "ring_standard", 0, policy="null", days=2, keep_env=True)
    env = r.pop("env")
    try:
        w, s0 = env.world, r["since_seq"]
        planted = replace(F.DOMAIN, mechanics=F.MECHANICS[:-1] + (base.Mechanic("Production",
                                                                                ("make.never_recorded",)),))
        miss = base.mechanics_exercised(w, planted, s0)
        orig = base.mechanics_exercised(w, F.DOMAIN, s0)
        generic = base.mechanics_exercised(w, replace(F.DOMAIN, mechanics=tuple(
            base.Mechanic(x.name, ("bank.txn",)) for x in F.MECHANICS)), s0)
        sched = base.mechanics_exercised(w, replace(F.DOMAIN, mechanics=tuple(
            base.Mechanic(x.name, ("sched.*",)) for x in F.MECHANICS)), s0)
        const = base.mechanics_exercised(w, replace(F.DOMAIN, mechanics=tuple(
            base.Mechanic(x.name, (), count=lambda w_, s: 1) for x in F.MECHANICS)), s0)
    finally:
        env.close()
    metrics = {"domains_with_executable_mechanisms": m["domains_with_executable_mechanisms"],
               "missing_required_mechanics": m["missing_required_mechanics"],
               **_context(m, per, ["mechanics_ok", "mechanics_missing"]),
               "missing_by_domain": ";".join(f"{n}:{'|'.join(r_['mechanics']['missing'])}" for n, r_ in reps.items()
                                             if r_["mechanics"]["missing"]) or "none",
               "unverified_by_domain": ";".join(f"{n}:{'|'.join(r_['mechanics'].get('unverified', []))}"
                                                for n, r_ in reps.items() if r_["mechanics"].get("unverified"))
               or "none",
               "control_fixture_mechanics_exercised": ref["mechanics"]["exercised"],
               "control_planted_missing": ",".join(miss["missing"]) or "none",
               "control_generic_bank_txn_exercised": generic["exercised"],
               "control_scheduler_events_exercised": sched["exercised"],
               "control_constant_counts_exercised": const["exercised"]}
    evidence.record("N.3", metrics, command=CMD, labels=LABELS)
    assert ref["mechanics"]["missing"] == [] and "Production" in miss["missing"]
    assert "Production" not in orig["missing"] and orig["counts"]["Production"] > 0
    assert generic["exercised"] == 0 and sched["exercised"] == 0 and const["exercised"] == 0
    assert m["domains_with_executable_mechanisms"] >= 15 and m["missing_required_mechanics"] == 0, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_4_independent_outcome_evaluation(fixture):
    reps, m, per = _metrics()
    ev = _control()["evaluator"]
    # verifier probe p4 'selfreport': a verdict on a participant-controlled fact survives the autonomy ablation
    selfrep = _register(replace(F.DOMAIN, name="fixture_selfreport", goal=None, objective=F.objective,
                                evaluate=F.org_exists_evaluator))
    try:
        bad = base.evaluator_report(selfrep, "ring_standard", 0, days=4)
    finally:
        base.unregister("fixture_selfreport")
    # verifier probe p3 'expose': a participant tool named like the evaluator
    named = base.evaluator_like_tools([{"name": "shop.score", "description": "your current objective score"},
                                       {"name": "caps.verify", "description": "run a capability's test suite"}])
    metrics = {"domains_with_independent_evaluator": m["domains_with_independent_evaluator"],
               **_context(m, per, ["evaluator_ok"]),
               "evaluator_replay_equal": ";".join(f"{n}={(r.get('evaluator') or {}).get('replay_equal')}"
                                                  for n, r in sorted(reps.items())) or "none",
               "evaluator_needs_autonomous_participants": ";".join(
                   f"{n}={(r.get('evaluator') or {}).get('needs_autonomous_participants')}"
                   for n, r in sorted(reps.items())) or "none",
               "control_fixture_evaluator_ok": ev["ok"], "control_fixture_replay_equal": ev["replay_equal"],
               "control_fixture_ablation_success": ev.get("ablation_success"),
               "control_self_report_evaluator_ok": bad["ok"],
               "control_self_report_ablation_success": bad.get("ablation_success"),
               "control_evaluator_named_tools": ",".join(named) or "none"}
    evidence.record("N.4", metrics, command=CMD, labels=LABELS)
    assert ev["ok"] and ev["replay_equal"] and ev["evaluator_hidden"] and ev["needs_autonomous_participants"]
    assert bad["ok"] is False and bad["ablation_success"] is True and named == ["shop.score"]
    assert m["domains_with_independent_evaluator"] >= 15, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_5_successful_feasible_reference(fixture):
    reps, m, per = _metrics()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference", CTL_DAYS)
    null = base.cached_run(F.NAME, "ring_standard", 0, "null", CTL_DAYS)

    class Peeking(F.RetailReference):
        def plan(self, obs):
            return [] if self.world.state is None else []
    flagged = base.reads_world(Peeking("p:trader", {}))
    # dynamic isolation: a reference that finds a live World through the garbage collector does so in the parent's
    # process and not in its own
    peek = _register(replace(F.DOMAIN, name="fixture_peek", reference=F.PeekingReference))
    try:
        inproc = base.run(peek.name, "ring_standard", 0, policy="reference", days=0.5, isolate=False)
        iso = base.run(peek.name, "ring_standard", 0, policy="reference", days=0.5)
    finally:
        base.unregister("fixture_peek")
    metrics = {"domains_with_successful_reference": m["domains_with_successful_reference"],
               **_context(m, per, ["reference_ok"]),
               "reference_success_rates": ";".join(f"{n}={r['reference_successes']}/{r['reference_runs']}"
                                                   for n, r in sorted(reps.items())) or "none",
               "control_fixture_reference_success": ref["success"], "control_fixture_null_success": null["success"],
               "control_fixture_reference_isolated": ref["isolated"],
               "control_world_reading_reference_flagged": bool(flagged),
               "control_gc_peek_first_tool_in_process": inproc["drive"]["agents"]["p:trader"]["tools"][0],
               "control_gc_peek_first_tool_isolated": iso["drive"]["agents"]["p:trader"]["tools"][0]}
    evidence.record("N.5", metrics, command=CMD, labels=LABELS)
    assert ref["success"] is True and null["success"] is False and flagged and ref["isolated"]
    assert metrics["control_gc_peek_first_tool_in_process"] == "id.me"
    assert metrics["control_gc_peek_first_tool_isolated"] == "bank.accounts"
    assert m["domains_with_successful_reference"] >= 15, metrics


def _static_build(config, seed, overrides):
    return {"module_list": ["identity", "bank", "admin"], "modules": {}, "params": {}}


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_6_procedural_instances(fixture):
    reps, m, per = _metrics()
    ctl = base.structure_report(F.DOMAIN)
    # verifier probes p5/p6 'static': a world without procedural content has ONE instance however many seeds
    static = _register(replace(F.DOMAIN, name="fixture_static", build=_static_build, structure=None))
    try:
        st = base.structure_report(static)
    finally:
        base.unregister("fixture_static")
    metrics = {"min_seeds_per_config": m["min_seeds_per_config"], **_context(m, per, ["min_instances"]),
               "control_fixture_min_instances": min(ctl["instances"].values()),
               "control_fixture_varying_tables": ",".join(ctl["varying_tables"]["ring_standard"][:12]),
               "control_static_world_max_instances": max(st["instances"].values())}
    evidence.record("N.6", metrics, command=CMD, labels=LABELS)
    assert min(ctl["instances"].values()) == 3 and max(st["instances"].values()) == 1
    assert m["min_seeds_per_config"] >= 3, metrics


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_7_unfamiliar_evaluation_configurations(fixture):
    reps, m, per = _metrics()
    rep = _control()
    v = base.verdicts(rep)
    ctl = rep["structure"]
    # positive control: an "evaluation" configuration equal in structure to a training one is neither
    t = ctl["fingerprints"]["ring_standard"]
    same = base.unfamiliarity([ctl["fingerprints"][c] for c in ("ring_standard", "line_linear", "star_diamond")], t)
    # verifier probe p3 'evalcfg': the structural evaluation configuration broken at runtime (a module it needs
    # dropped) is measured unfamiliar but does not count
    broken = _register(replace(F.DOMAIN, name="fixture_evalcfg", build=_broken_eval_build))
    try:
        brep = base.check(broken, tier3=False, seeds=(0,), structure_seeds=(0,), days=CTL_DAYS)
    finally:
        base.unregister("fixture_evalcfg")
    bv = base.verdicts(brep)
    metrics = {"domains_with_compositional_eval": m["domains_with_compositional_eval"],
               "domains_with_structural_eval": m["domains_with_structural_eval"],
               **_context(m, per, ["compositional_eval", "structural_eval"]),
               "control_fixture_compositional": v["compositional_eval"],
               "control_fixture_structural": v["structural_eval"],
               "control_copy_of_training_unfamiliar": bool(same["compositional"] or same["structural"]),
               "control_broken_eval_config_measured_unfamiliar": brep["structure"]["structural_eval"],
               "control_broken_eval_config_proc_failed": brep["per_config"]["complete_wide"]["proc_failed"],
               "control_broken_eval_config_counted": bv["structural_eval"]}
    evidence.record("N.7", metrics, command=CMD, labels=LABELS)
    assert v["compositional_eval"] and v["structural_eval"]
    assert not same["compositional"] and not same["structural"] and same["equal_to_training"]
    assert brep["per_config"]["complete_wide"]["proc_failed"] > 0 and bv["structural_eval"] is False
    assert m["domains_with_compositional_eval"] >= 15 and m["domains_with_structural_eval"] >= 15, metrics


def _broken_eval_build(config, seed, overrides):
    r = F.build(config, seed, overrides)
    if config.name == "complete_wide":
        r["module_list"] = [x for x in r["module_list"] if x != "market"]
        r["modules"].pop("market", None)
    return r


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_N_n1_reference_not_exposed(fixture):
    reps, m, per = _metrics()
    rep = _control()
    ref = base.cached_run(F.NAME, "ring_standard", 0, "reference", CTL_DAYS)
    traj = ref["drive"]["agents"][ref["seats"][0]]["tools"]
    pol = F.RetailReference("p:trader", {})
    canaries = base.reference_canaries(F.DOMAIN, pol)
    clean = base.exposure([rep["goal"]["texts"][1]], canaries, traj)
    # positive controls: an objective that spells out the reference's path, a tool doc naming the reference, and
    # (verifier probe p3 'expose') the path planted in an observation field the earlier scan never read
    seq = base._first_distinct(traj)
    spelled = base.exposure([f"Do this: {', then '.join(seq[:4])}."], canaries, traj)
    named = base.exposure([f'{{"name": "help.solve", "description": "runs {canaries[0]}"}}'], canaries, traj)
    from crucible.experience import kernel as K
    orig = K.KernelExperienceEnv.observe
    notice = "Tip of the day: call " + ", then ".join(seq[:4]) + "."

    def observe(self, agent):
        o = orig(self, agent)
        o["notice"] = notice
        return o
    K.KernelExperienceEnv.observe = observe
    try:
        planted = base.run(F.NAME, "ring_standard", 0, policy="null", days=0.5)
    finally:
        K.KernelExperienceEnv.observe = orig
    in_obs = base.trajectory_in_sequences(planted["exposure"]["sequences"], traj)
    metrics = {"reference_paths_exposed": m["reference_paths_exposed"],
               "domains_scanned_for_exposure": m["domains_scanned_for_exposure"], **_context(m, per, ["exposed"]),
               "sandbox_imports_crucible": base.sandbox_can_import_crucible(),
               "control_fixture_exposed": rep["reference_paths_exposed"],
               "control_fixture_texts_scanned": rep["isolation"]["texts_scanned"],
               "control_fixture_tools_queried": len(rep["isolation"]["queried_tools"]),
               "control_objective_exposed": clean["exposed"], "control_spelled_out_path_detected": spelled["exposed"],
               "control_named_reference_detected": named["exposed"],
               "control_observation_notice_detected": len(in_obs)}
    evidence.record("N.n1", metrics, command=CMD, labels=LABELS)
    assert rep["reference_paths_exposed"] == 0 and clean["exposed"] == 0, rep["isolation"]["exposure_hits"]
    assert spelled["exposed"] >= 1 and named["exposed"] >= 1 and len(in_obs) >= 1
    assert metrics["sandbox_imports_crucible"] is False and rep["isolation"]["queried_tools"]
    assert m["domains_scanned_for_exposure"] >= 15, metrics
    assert m["reference_paths_exposed"] == 0, metrics
