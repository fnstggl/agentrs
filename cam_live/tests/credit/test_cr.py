"""CR (amendment item 4): estimand-matched held-out selection and credit units inside one long open objective.

* The per-experiment gate scores an estimator against each experiment's own measured effect. When experiments differ
  by a mechanism the decision's information state does not reveal, even the best estimator given that information
  fails it; the pooled unit-level rule tests the estimand the estimator targets and keeps per-experiment calibration
  wherever an experiment has enough units to test it.
* One-unit experiments (the integrated agent's pattern) contribute units to the pooled rule.
* Inside one open objective, complete segments of K decisions are credit units: RUDDER / COCOA credit appears for
  matured segments before the objective closes, earlier credit is revised when the lifetime closes, and the learner's
  segment split holds out whole segments so their randomised decisions are held-out units.
"""
import math
import random

import pytest

from cam_live.config import CamLiveConfig
from cam_live.credit.labkit import Arms, LabRun, build_lab, door_candidates, door_weight, mk_action, weighted_policy
from cam_live.credit.learner import ContinualLearner, LearnerSettings
from cam_live.credit.providers import COCOA, KEY_CONTRIB, RUDDER, trajectory_return
from cam_live.credit.segments import CreditUnits, is_segment, parse_unit, segment_id
from cam_live.credit.select import (PER_DECISION, PER_UNIT_CONTRAST, EstimatorSelector, ExperimentEvidence,
                                    UnitContrast, chi2_cdf, chi2_ppf, paired_comparison, pooled_unit_score)
from cam_live.contracts import ExperimentRecord, Modality
from cam_live.diagnostics.world import LabConfig, LabWorld
from cam_live.learning_api import UpdatePlan

BETA = 2e-6


# ------------------------------------------------------------------------------------------------ statistics
def test_chi2_quantiles_match_tables():
    for k, q99 in ((1, 6.635), (2, 9.210), (5, 15.086), (6, 16.812), (10, 23.209)):
        assert abs(chi2_ppf(0.99, k) - q99) < 2e-3
    assert abs(chi2_cdf(3.841, 1) - 0.95) < 1e-3


def _hidden_mechanism_units(rng, effects, n_per_arm, est_fn, sd=5.0):
    """Units of len(effects) experiments whose true effects differ by a hidden mechanism; every unit has the same
    information state, so the best estimate at that state is the mean effect."""
    units, per_exp = [], {}
    for k, eff in enumerate(effects):
        rows = []
        for arm, mu in (("T", eff), ("C", 0.0)):
            for _ in range(n_per_arm):
                y = mu + rng.gauss(0.0, sd)
                e = est_fn(k) if arm == "T" else 0.0
                units.append(UnitContrast(f"x{k}", f"u{len(units)}", f"o{len(units)}", "T", "C", arm, 0.5, y,
                                          {"est": e}, {"est": PER_DECISION}))
                rows.append((arm, e, y))
        per_exp[f"x{k}"] = rows
    return units, per_exp


def test_the_per_experiment_gate_rejects_the_best_estimate_at_the_information_state_the_pooled_rule_does_not():
    rng = random.Random(3)
    effects = [3.0, 6.0, 2.0, -1.0, 6.0, 12.0]
    marginal = sum(effects) / len(effects)
    units, per_exp = _hidden_mechanism_units(rng, effects, 60, lambda k: marginal)
    z2 = []
    for rows in per_exp.values():
        d, se, _ = paired_comparison(rows, "T", "C")
        z2.append((d / se) ** 2)
    assert sum(z2) / len(z2) > 3.0                           # the old gate fails the information-state optimum
    ps = pooled_unit_score(units, "est", min_units=8, margin_sd=0.5)
    assert ps.passes, ps.reasons                             # the estimand-matched rule accepts it
    assert abs(ps.bias) < 2.5 * ps.se
    biased, _ = _hidden_mechanism_units(random.Random(3), effects, 60, lambda k: marginal + 5.0)
    pb = pooled_unit_score(biased, "est", min_units=8, margin_sd=0.5)
    assert not pb.passes and pb.bias > 3.0                   # a biased estimator is still refused


def test_pooled_rule_is_orientation_invariant_and_handles_per_unit_contrasts():
    rng = random.Random(5)
    units = []
    for i in range(3000):
        arm = "T" if rng.random() < 0.3 else "C"
        y = (2.0 if arm == "T" else 0.0) + rng.gauss(0, 1)
        x = 2.0 + rng.gauss(0, 0.5)                          # an unbiased per-unit contrast estimate
        units.append(UnitContrast("x", f"u{i}", f"o{i}", "T", "C", arm, 0.3, y,
                                  {"cocoa_like": x, "credit": 2.0 if arm == "T" else 0.0,
                                   "biased": 3.5 if arm == "T" else 0.0},
                                  {"cocoa_like": PER_UNIT_CONTRAST, "credit": PER_DECISION, "biased": PER_DECISION}))
    a = pooled_unit_score(units, "cocoa_like", margin_sd=0.5)
    b = pooled_unit_score(units, "credit", margin_sd=0.5)
    assert a.passes and b.passes, (a.reasons, b.reasons)
    assert abs(a.bias) < 3 * a.se and abs(b.bias) < 3 * b.se
    bad = pooled_unit_score(units, "biased", margin_sd=0.5)
    assert not bad.passes and abs(bad.bias - 1.5) < 4 * bad.se
    # the labels of the arms are arbitrary: swapping them (p -> 1 - p) only flips the sign of the bias
    flipped = [UnitContrast(u.experiment_id, u.exp_id, u.unit, "C", "T", u.arm, 1 - u.p_treatment, u.y,
                            dict(u.values), dict(u.kinds)) for u in units]
    c = pooled_unit_score(flipped, "credit", margin_sd=0.5)
    assert abs(c.bias + b.bias) < 1e-9 and abs(c.se - b.se) < 1e-9 and c.passes == b.passes


def test_one_unit_experiments_reach_the_pooled_selection_and_too_few_units_never_promote():
    rng = random.Random(7)
    sel = EstimatorSelector(min_units=8)
    few = []
    for i in range(5):
        arm = "T" if rng.random() < 0.5 else "C"
        y = (1.0 if arm == "T" else 0.0) + rng.gauss(0, 1)
        few.append(UnitContrast(f"x{i}", f"u{i}", f"o{i}", "T", "C", arm, 0.5, y,
                                {"rudder": 1.0 if arm == "T" else 0.0}, {"rudder": PER_DECISION}))
    sel.add_units(few)
    rec = sel.select()
    assert rec.mode == "pooled_units" and rec.default_after == "simple"
    assert any("only 5 held-out randomised units" in r for r in rec.reasons)
    many = []
    for i in range(400):
        arm = "T" if rng.random() < 0.5 else "C"
        y = (1.0 if arm == "T" else 0.0) + rng.gauss(0, 1)
        many.append(UnitContrast(f"y{i}", f"v{i}", f"p{i}", "T", "C", arm, 0.5, y,
                                 {"rudder": 1.0 if arm == "T" else 0.0}, {"rudder": PER_DECISION}))
    sel.add_units(many)
    rec = sel.select()
    assert rec.n_units == 405 and rec.default_after == "rudder", rec.reasons
    st = sel.state_dict()
    sel2 = EstimatorSelector()
    sel2.load_state_dict(st)
    assert len(sel2.units) == 405 and sel2.default == "rudder"
    old = {k: v for k, v in st.items() if k != "units"}        # a selector state written before CR
    sel3 = EstimatorSelector()
    sel3.load_state_dict(old)
    assert sel3.units == {} and sel3.default == "rudder"


def test_per_experiment_calibration_is_kept_where_experiments_can_test_it():
    rng = random.Random(11)
    sel = EstimatorSelector(min_units=8, calib_min_per_arm=5)
    effects = [0.0, 8.0]                                      # identifiable per experiment, estimator ignores it
    for k, eff in enumerate(effects):
        rows, us = [], []
        for arm, mu in (("T", eff), ("C", 0.0)):
            for i in range(40):
                y = mu + rng.gauss(0, 1.0)
                e = 4.0 if arm == "T" else 0.0
                rows.append((arm, e, y))
                us.append(UnitContrast(f"x{k}", f"u{k}_{arm}_{i}", f"o{k}_{arm}_{i}", "T", "C", arm, 0.5, y,
                                       {"rudder": e}, {"rudder": PER_DECISION}))
        d, se, n = paired_comparison(rows, "T", "C")
        ev = ExperimentEvidence(f"x{k}", "T", "C", eff, se, eff - 2 * se, eff + 2 * se, 40, 40, True,
                                {"rudder": 4.0}, {}, held_out=True, paired_diff={"rudder": d}, paired_se={"rudder": se},
                                paired_n={"rudder": n})
        sel.add_evidence(ev)
        sel.add_units(us)
    sc = sel.pooled_scores()["rudder"]
    assert sc["calibration"]["experiments"] == 2 and sc["calibration"]["rejected"]
    assert not sc["passes"]
    assert sel.select().default_after == "simple"


# ------------------------------------------------------------------------------------------------ segments
def _long_world(seed=0):
    return LabWorld(LabConfig(n_channels=3, noisy_tv=False, key_door=True, mean_delay_s=(3600.0, 6 * 3600.0),
                              outcome_horizon_s=12 * 3600.0, horizon_s=400 * 86400.0, budget=1e7,
                              family="lab/door-long", seed=seed))


def _long_lab(tmp_path, *, K=6, val_split="segment", val_fraction=0.25, name="long.sqlite"):
    cfg = CamLiveConfig.debug()
    lab = build_lab(str(tmp_path / name), cfg, world=False, policy=False, credit=True, beta=BETA,
                    settings=LearnerSettings.from_cfg(cfg, batch_size=16, val_fraction=val_fraction,
                                                      val_split=val_split, holdout_experiments=False))
    lab.assigner.units.decisions = K
    return lab


def _run_open_objective(lab, n_decisions, *, experiment_every=3, seed=0):
    world = _long_world(seed)
    world.clock = lab.clock
    run = LabRun(lab.store, lab.linker, world, seed=seed)
    pol = weighted_policy(door_weight, seed=seed + 1)
    xid = "x-long"
    lab.store.put_experiment(ExperimentRecord(experiment_id=xid, created_at=lab.clock.now(), question="q",
                                              hypothesis="h", arms=["unlock", "wait"], assignment="randomized",
                                              arm_probs=[0.5, 0.5], status="running"))
    arms = Arms(xid, {"unlock": lambda: mk_action("lab.unlock"),
                      "wait": lambda: mk_action("wait", Modality.INTERNAL, seconds=3600.0)}, {"unlock": 0.5, "wait": 0.5})
    for t in range(n_decisions):
        run.decide(pol, candidates_fn=door_candidates, arms=arms if t % experiment_every == 0 else None)
        lab.clock.advance(1800.0)
        run.reconcile()
    return run


def test_learned_credit_appears_for_matured_segments_of_an_open_objective_and_is_revised_at_closure(tmp_path):
    lab = _long_lab(tmp_path, K=6, val_split="record")
    try:
        run = _run_open_objective(lab, 20)
        oid = run.task.objective_id
        assert lab.store.objective_outcome(oid) is None                  # the objective stays open
        run.mature()                                                     # every declared measurement answered
        u = lab.assigner.units
        recs = lab.store.trajectory(oid)
        assert [u.unit_of(r) for r in recs[:6]] == [segment_id(oid, 0)] * 6
        assert u.unit_of(recs[19]) == segment_id(oid, 3)                 # 20 decisions: segments 0-2 complete
        assert u.complete(segment_id(oid, 2)) and not u.complete(segment_id(oid, 3))
        up = lab.assigner.assign([r.exp_id for r in recs])
        assert sorted(up.matured) == [segment_id(oid, k) for k in range(3)]
        got = {r.exp_id: r.credit for r in lab.store.trajectory(oid)}
        for r in recs[:18]:
            assert RUDDER in got[r.exp_id] and KEY_CONTRIB in got[r.exp_id]
        for r in recs[18:]:
            assert RUDDER not in got[r.exp_id]                          # the incomplete segment: no learned credit
        # conservation inside every segment: sum_t Gamma_t c_t = G of the segment
        for k in range(3):
            tr = lab.rudder.returns.get(segment_id(oid, k))
            assert tr.matured and tr.objective_id == segment_id(oid, k)
            assert up.conservation_error < 1e-6
        # training on matured segments of the open objective: RUDDER's loss has data before the objective closes
        lab.learner.refresh_eligibility(full=True)
        assert sum(1 for r in lab.store.trajectory(oid) if r.eligible.get(RUDDER)) >= 12
        lab.learner.learn(UpdatePlan(trigger="replay", max_steps=4, budget_s=1e9))
        assert any(is_segment(x) for x in lab.rudder.trained_objectives)
        assert oid in lab.rudder.trained_objectives                     # its lifetime is no longer held out
        v_open = {r.exp_id: dict(r.credit) for r in lab.store.trajectory(oid)}
        run.close()                                                      # the verifier closes the objective
        assert u.unit_of(recs[0]) == oid
        up2 = lab.assigner.assign([r.exp_id for r in recs], reason="objective_closed")
        assert up2.matured == [oid]
        hist = lab.store.credit_history(recs[0].exp_id)
        rud = [h for h in hist if h["estimator"] == RUDDER]
        assert len(rud) >= 2 and rud[-1]["reason"] == "objective_closed"   # revised over the whole lifetime
        assert all(RUDDER in r.credit for r in lab.store.trajectory(oid))
        assert v_open[recs[19].exp_id].get(RUDDER) is None
    finally:
        lab.close()


def test_segment_units_are_off_with_zero_decisions_and_lifetimes_are_unchanged(tmp_path):
    lab = _long_lab(tmp_path, K=0, val_split="record")
    try:
        run = _run_open_objective(lab, 8)
        run.mature()
        recs = lab.store.trajectory(run.task.objective_id)
        assert all(lab.assigner.unit_of(r) == run.task.objective_id for r in recs)
        up = lab.assigner.assign([r.exp_id for r in recs])
        assert up.matured == [] and all(RUDDER not in r.credit for r in lab.store.trajectory(run.task.objective_id))
    finally:
        lab.close()


def test_segment_split_holds_out_whole_segments_and_their_randomised_decisions_are_held_out_units(tmp_path):
    lab = _long_lab(tmp_path, K=4, val_split="segment", val_fraction=0.25)
    try:
        run = _run_open_objective(lab, 40, experiment_every=2)
        run.mature()
        oid = run.task.objective_id
        recs = lab.store.trajectory(oid)
        u = lab.assigner.units
        by_seg = {}
        for r in recs:
            by_seg.setdefault(u.segment_index(r), set()).add(lab.learner.is_validation(r))
        assert by_seg[0] == {True, False}                              # segment 0 keeps the record split
        assert all(len(v) == 1 for k, v in by_seg.items() if k)        # later segments: one side each
        held = sorted(k for k, v in by_seg.items() if k and True in v)
        assert len(held) in (2, 3)                                     # every 4th of segments 1-9
        lab.learner.refresh_eligibility(full=True)
        lab.assigner.assign([r.exp_id for r in recs])
        for _ in range(3):
            lab.learner.learn(UpdatePlan(trigger="replay", max_steps=4, budget_s=1e9))
        trained = lab.assigner.trained_objectives()
        for k in held:
            assert segment_id(oid, k) not in trained                    # nothing trained on a held-out segment
        units, counts = lab.assigner.experiment_units("x-long", treatment="unlock", control="wait")
        assert units and all(parse_unit(x.unit)[1] in held for x in units)
        assert counts["lifetime_trained_on"] > 0                         # units of training segments are excluded
        assert all(x.p_treatment == 0.5 for x in units)
        ev = lab.assigner.evaluate_experiments(force_log=True)
        assert ev.n_units == len(units) and ev.ran
        assert lab.assigner.selector.history[-1].mode == "pooled_units"
    finally:
        lab.close()


def test_switching_to_the_segment_split_keeps_older_records_on_the_record_split(tmp_path):
    lab = _long_lab(tmp_path, K=4, val_split="record", val_fraction=0.25, name="sw.sqlite")
    try:
        run = _run_open_objective(lab, 12)
        recs = lab.store.trajectory(run.task.objective_id)
        old_sides = {r.exp_id: lab.learner.is_validation(r) for r in recs}
        st = lab.learner.state_dict()
        st.pop("trained_units"), st.pop("segment_split_from_seq")     # a checkpoint written before CR
        seg = ContinualLearner(lab.model, lab.store, lab.cfg, clock=lab.clock, credit=lab.assigner,
                               settings=LearnerSettings.from_cfg(lab.cfg, val_fraction=0.25, val_split="segment",
                                                                 holdout_experiments=False))
        seg.load_state_dict(st)
        assert seg.segment_split_from_seq == lab.store.head_seq() + 1
        assert {r.exp_id: seg.is_validation(r) for r in recs} == old_sides
        for r in recs:                                                 # their segments cannot be held-out units
            assert lab.assigner.unit_of(r) in seg.trained_units
    finally:
        lab.close()


def test_revised_credit_is_queued_for_replay_and_survives_a_checkpoint(tmp_path):
    lab = _long_lab(tmp_path, K=6, val_split="record", name="rev.sqlite")
    try:
        run = _run_open_objective(lab, 14)
        recs = lab.store.trajectory(run.task.objective_id)
        lab.assigner.assign([r.exp_id for r in recs])
        early = lab.assigner.pop_revised(100, RUDDER)                  # segments already matured in the run
        assert all(lab.store.get(e).credit.get(RUDDER) is not None for e in early)
        run.mature()
        lab.assigner.assign([r.exp_id for r in recs])
        got = lab.assigner.pop_revised(100, RUDDER)
        ids12 = [r.exp_id for r in recs[:12]]
        assert set(early) | set(got) == set(ids12)                      # the two complete segments
        assert len(got) == len(set(got))                                # each revised record once
        assert lab.assigner.pop_revised(100, RUDDER) == []            # popped once per revision
        simple = lab.assigner.pop_revised(100, "simple")
        assert simple and set(simple) <= {r.exp_id for r in recs}       # simple credit revisions stay queued separately
        run.close()
        lab.assigner.assign([r.exp_id for r in recs], reason="objective_closed")
        st = lab.assigner.state_dict()
        assert st["revised"] and st["segment_decisions"] == 6
        lab.assigner._revised = {}
        lab.assigner.load_state_dict(st)
        again = lab.assigner.pop_revised(100, RUDDER)
        assert set(again) == {r.exp_id for r in recs}                 # lifetime credit revised every decision
        lab.assigner.load_state_dict({k: v for k, v in st.items() if k != "revised"})   # pre-CR state: empty queue
        assert lab.assigner.pop_revised(100, RUDDER) == []
    finally:
        lab.close()
