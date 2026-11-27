"""Regression tests for the defects found by the independent verification of workstream D (docs/verification/G_cl.md):
M1 (estimator selection validated in-sample), M4 / m4 (slow-update guard), m1 (open-ended candidate sets), m2 (COCOA
pairs), m3 (long trajectories), m6 (promotion filter by provenance). M2 / M3 are pinned in test_replay.py and
test_links.py. Each test fails on the code the verifier audited (5a133f7)."""
import hashlib
import json
import math

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import Modality
from cam_live.credit import checkpoint as ckpt
from cam_live.credit.labkit import (LabRun, door_arms, door_config, mk_action, register_experiment,
                                    run_door_experiments, run_door_training, weighted_policy)
from cam_live.credit.learner import LearnerSettings
from cam_live.credit.providers import (COCOA, KEY_RTG, RUDDER, candidate_set_open_ended, return_to_go_at,
                                       trajectory_return)
from cam_live.credit.select import EstimatorSelector, ExperimentEvidence
from cam_live.diagnostics.world import LabWorld
from cam_live.learning_api import TIMESCALE_SLOW, UpdatePlan
from cam_live.nn.api import GROUP_ADAPTERS

BETA = 2e-5


def act_last(world, rng):
    return [mk_action("lab.act", channel=world.cfg.n_channels - 1, effort=3), mk_action("lab.act", channel=0, effort=1),
            mk_action("wait", Modality.INTERNAL, seconds=3600.0)]


def door_run(lab, seed=0, n_later=5, gen=None):
    world = LabWorld(door_config(seed=5), lab.clock)
    register_experiment(lab.store, "door-x", ["unlock", "wait"], lab.clock.now())
    run = LabRun(lab.store, lab.linker, world, seed=seed)
    r0 = run.decide(None, arms=door_arms("door-x"))
    pol = weighted_policy(lambda a, w: 5.0 if a.arguments.get("channel") == w.cfg.n_channels - 1 else 0.5, seed=1)
    later = [run.decide(pol, candidates_fn=gen or act_last) for _ in range(n_later)]
    run.mature()
    run.close()
    return run, r0, later


# ------------------------------------------------------------------------------------------------ M1
def test_experiment_lifetimes_are_held_out_and_the_comparison_excludes_whatever_was_trained_on(lab_factory):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(collect=False, cfg=cfg, world=False, policy=False, credit=True, beta=BETA,
                      settings=LearnerSettings.from_cfg(cfg, batch_size=16, val_fraction=0.15))
    learner, store, assigner = lab.learner, lab.store, lab.assigner
    train = run_door_training(lab, 2, 6, 4, seed=0)
    runs = run_door_experiments(lab, 2, 6, 4, seed=0)
    learner.refresh_eligibility(full=True)
    x_recs = [r for xid in runs for r in store.experiment_records(xid)]
    x_objs = {r.objective_id for r in x_recs}
    # every record of an experiment lifetime is validation data, not only the experiment decision
    x_all = [r for o in x_objs for r in store.trajectory(o)]
    assert x_all and all(learner.is_validation(r) for r in x_all)
    t_objs = {lt.task.objective_id for lts in train.values() for lt in lts}
    assert any(not learner.is_validation(r) for o in t_objs for r in store.trajectory(o))
    for _ in range(3):
        learner.learn(UpdatePlan(trigger="replay", max_steps=4, budget_s=1e9))
    assert not (x_objs & learner.trained_objectives) and not (x_objs & lab.rudder.trained_objectives)
    assert lab.rudder.trained_objectives and lab.rudder.trained_objectives <= t_objs
    assigner.assign(reason="test")
    xid = next(iter(runs))
    ev = assigner.compare_with_experiment(xid, treatment="unlock", control="wait")
    assert ev is not None and ev.held_out is True and ev.n_excluded_trained == 0
    assert RUDDER in ev.paired_se and ev.paired_n[RUDDER] == ev.n_treatment + ev.n_control
    # a lifetime an estimator trained on is left out of the measured difference and of every estimate
    one = store.experiment_records(xid)[0].objective_id
    lab.rudder.trained_objectives.add(one)
    ev2 = assigner.compare_with_experiment(xid, treatment="unlock", control="wait")
    n_units = sum(1 for r in store.experiment_records(xid) if r.objective_id != one)
    assert ev2 is None or (ev2.n_excluded_trained == 1 and ev2.n_treatment + ev2.n_control == n_units)
    # all of them trained on: nothing left to validate with
    lab.cocoa.trained_objectives |= {r.objective_id for r in store.experiment_records(xid)}
    assert assigner.compare_with_experiment(xid, treatment="unlock", control="wait") is None
    # the training registries travel with the checkpoint
    assert lab.rudder.state_dict()["trained_objectives"] == sorted(lab.rudder.trained_objectives)
    assert learner.state_dict()["trained_objectives"] == sorted(learner.trained_objectives)


def _ev(i, measured, est, est_se=None, held_out=True, **kw):
    return ExperimentEvidence(f"x{i}", "B", "A", measured, 0.2, measured - 0.4, measured + 0.4, 30, 30, True,
                              {"rudder": est}, {} if est_se is None else {"rudder": est_se}, held_out=held_out, **kw)


def test_zero_standard_error_and_in_sample_evidence_cannot_promote_an_estimator():
    effects = [1.0, -0.8, 0.6, 1.5, -1.2, 0.9]
    # (a) the estimate's uncertainty is zero / not derived: the old z-score used the measured se alone and promoted
    sel = EstimatorSelector(min_experiments=5)
    for i, e in enumerate(effects):
        sel.add_evidence(_ev(i, e, e + 0.05, est_se=0.0))
    r = sel.select()
    assert r.default_after == "simple" and r.scores["rudder"]["n_refused"] == len(effects)
    assert any("zero or undefined standard error" in x for x in r.scores["rudder"]["reasons"])
    # (b) paired standard error of exactly zero (the estimate *is* the measurement): refused, never passed
    sel = EstimatorSelector(min_experiments=5)
    for i, e in enumerate(effects):
        sel.add_evidence(_ev(i, e, e, paired_diff={"rudder": 0.0}, paired_se={"rudder": 0.0}))
    assert sel.select().default_after == "simple"
    # (c) in-sample or unknown split: kept for the record, never scored
    for held in (False, None):
        sel = EstimatorSelector(min_experiments=5)
        for i, e in enumerate(effects):
            sel.add_evidence(_ev(i, e, e + 0.01, est_se=0.05, held_out=held))
        r = sel.select()
        assert r.default_after == "simple" and len(r.excluded_not_held_out) == len(effects)
    # (d) held out with a derived paired standard error: an accurate estimator is promoted
    sel = EstimatorSelector(min_experiments=5)
    for i, e in enumerate(effects):
        sel.add_evidence(_ev(i, e, e + 0.05, paired_diff={"rudder": 0.05}, paired_se={"rudder": 0.1}))
    r = sel.select()
    assert r.default_after == "rudder" and r.scores["rudder"]["n_paired"] == len(effects)
    assert set(r.holdout) == {f"x{i}:B-A" for i in range(len(effects))}


# ------------------------------------------------------------------------------------------------ m1, m2, m3
def test_long_trajectories_get_credit_everywhere_cocoa_pairs_follow_eq4_and_open_ended_sets_are_refused(lab_factory):
    cfg = CamLiveConfig.debug()
    lab = lab_factory(collect=False, cfg=cfg, world=False, policy=False, credit=True, beta=BETA)
    lab.rudder.encoder.max_len = 3
    lab.cocoa.encoder.max_len = 3
    run, r0, later = door_run(lab)
    st, oid = lab.store, run.task.objective_id
    lab.assigner.assign(run.exp_ids, reason="objective_closed")
    recs = st.trajectory(oid)
    tr = trajectory_return(st, recs, beta=BETA)
    # m3: every decision of the 6-decision objective gets RUDDER credit, referenced at the trajectory start
    assert all(RUDDER in st.get(r.exp_id).credit for r in recs)
    assert st.get(recs[0].exp_id).credit[KEY_RTG] == pytest.approx(tr.G, rel=1e-6, abs=1e-9)
    # ... and a training window of the newest decisions regresses its own return-to-go, referenced at its start
    win = lab.rudder.encoder.encode(oid)
    assert len(win) == 3
    assert lab.rudder.window_target(win, tr) == pytest.approx(return_to_go_at(tr, float(win.ts[0]), BETA))
    assert lab.rudder.window_target(lab.rudder.encoder.encode(oid, full=True), tr) == tr.G
    # m2: an outcome declared by an earlier decision and observed after t is one of t's later outcomes
    enc = lab.cocoa.encoder.encode(oid, full=True)
    pairs = set(lab.cocoa._pairs(enc, tr))
    full = {e: i for i, e in enumerate(tr.exp_ids)}
    want = {(t, k) for k, ev in enumerate(tr.events) for t, e in enumerate(enc.exp_ids)
            if ev.index >= 0 and full[e] != ev.index and float(enc.ts[t]) < ev.observed_at}
    assert want and want <= pairs
    assert any(full[enc.exp_ids[t]] > tr.events[k].index for t, k in pairs)     # earlier-declared ones are included
    assert all(full[enc.exp_ids[t]] != tr.events[k].index for t, k in pairs)    # never the decision's own outcome
    # m1: candidates generated by the backbone without tracked proposal probabilities -> COCOA refuses

    def backbone(world, rng):
        acts = act_last(world, rng)
        for a in acts:
            a.provenance = "backbone"
        return acts
    run2, _, later2 = door_run(lab, seed=3, n_later=2, gen=backbone)
    assert candidate_set_open_ended(later2[0]) and not candidate_set_open_ended(r0)
    cc = lab.cocoa.credit([run2.task.objective_id])
    assert all(cc[r.exp_id].status == "unsupported" and "open_ended_without_tracked_proposals" in cc[r.exp_id].reasons
               for r in later2)
    merged = later2[0]
    merged.decision.candidates[0].action.provenance = "backbone+schema_walker"    # also enumerated: still open
    assert candidate_set_open_ended(merged)
    for c in merged.decision.candidates:
        c.action.provenance = "backbone+schema_walker"
    assert not candidate_set_open_ended(merged)                                   # every one enumerated: closed


# ------------------------------------------------------------------------------------------------ M4, m4
class _AdapterProbe:
    """A slow-timescale loss with an exactly known effect of a training step on its *held-out* value: training
    (model.train()) minimises X = 10 + s * mean(adapters); validation (model.eval()) reports ``eval_sign * X`` (-1: the
    step makes held-out loss worse by about 100 * lr, +1: better). ``spread`` makes the effect differ between
    validation chunks (a non-zero noise estimate)."""
    timescale = TIMESCALE_SLOW
    needs_latents = False

    def __init__(self, model, s=100.0, spread=0.0, name="probe", eval_sign=-1.0):
        self.model, self.s, self.spread, self.name, self.weight = model, s, spread, name, 1.0
        self.eval_sign = eval_sign

    def eligible(self, record):
        return True

    def loss(self, model, batch, store):
        ps = model.param_groups()[GROUP_ADAPTERS]
        n = sum(p.numel() for p in ps)
        x = 10.0 + self.s * sum(p.sum() for p in ps) / n
        if self.spread:
            f = sum(int(hashlib.sha1(r.exp_id.encode()).hexdigest()[:4], 16) / 65536.0 for r in batch.records)
            x = 10.0 + (x - 10.0) * (1.0 + self.spread * (f / max(len(batch.records), 1) - 0.5))
        return (x if model.training else self.eval_sign * x), {}


def _guard_lab(lab_factory, **settings):
    cfg = CamLiveConfig.debug()
    cfg.optim.lr_adapters = 1e-4
    st = dict(slow_guard=True, val_fraction=0.4, val_batch=32, slow_batch_size=4)
    st.update(settings)
    lab = lab_factory(n_worlds=8, n_decisions=6, cfg=cfg, world=False, policy=False, credit=False,
                      settings=LearnerSettings.from_cfg(cfg, **st))
    return lab


def _opt_state(opt):
    return {id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state.get(p, {}).items()}
            for g in opt.param_groups for p in g["params"]}


def _same_state(a, b):
    return a.keys() == b.keys() and all(
        a[i].keys() == b[i].keys() and all(torch.equal(a[i][k], b[i][k]) if torch.is_tensor(a[i][k]) else a[i][k] == b[i][k]
                                           for k in a[i]) for i in a)


def test_a_slow_step_that_degrades_held_out_loss_by_a_small_relative_amount_is_undone_completely(lab_factory):
    lab = _guard_lab(lab_factory)
    L, m = lab.learner, lab.model
    probe = L.register(_AdapterProbe(m, eval_sign=+1.0))
    L.refresh_eligibility(full=True)
    opt = L.optimizers[GROUP_ADAPTERS]
    # an improving step is kept (and leaves non-empty Adam moments behind)
    rec = L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[-1]
    g = json.loads(rec.notes)["guard"]
    assert g["accepted"] and g["paired_change"] < 0 and rec.param_groups == [GROUP_ADAPTERS]
    assert any(opt.state.get(p) for p in m.param_groups()[GROUP_ADAPTERS])
    # a step that raises held-out loss by ~0.1 % (inside the old +5 % rule, far above the noise) is undone exactly
    probe.eval_sign = -1.0
    w0 = [p.detach().clone() for p in m.param_groups()[GROUP_ADAPTERS]]
    o0, s0, c0, v0 = _opt_state(opt), dict(L.sched_step), dict(L.counters), L.model_version
    lr0 = [pg["lr"] for pg in opt.param_groups]
    rec = L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[-1]
    g = json.loads(rec.notes)["guard"]
    assert 0 < g["paired_change"] < 0.05 * abs(g["validation_before"])
    assert g["reverted"] is True and not g["accepted"] and g["tolerance"] < g["paired_change"]
    assert all(torch.equal(a, b) for a, b in zip(w0, m.param_groups()[GROUP_ADAPTERS]))
    assert _same_state(o0, _opt_state(opt)) and [pg["lr"] for pg in opt.param_groups] == lr0
    assert L.sched_step == s0 and L.counters["slow"] == c0["slow"] and L.model_version == v0
    assert L.counters["slow_reverted"] == c0["slow_reverted"] + 1 and rec.param_groups == []


def test_cumulative_drift_of_accepted_slow_steps_is_bounded_until_a_promotion(lab_factory, tmp_path):
    lab = _guard_lab(lab_factory, slow_guard_abs_tol=1.0, slow_guard_tol=1.0, slow_guard_z=1000.0, warmup_steps=0)
    L, m = lab.learner, lab.model
    L.register(_AdapterProbe(m, s=100.0, spread=1.0))
    L.refresh_eligibility(full=True)
    step = lambda: json.loads(L.learn(UpdatePlan(trigger="test", timescales=(TIMESCALE_SLOW,)))[-1].notes)["guard"]
    v1 = step()
    assert v1["accepted"] and v1["paired_change"] > 0 and v1["paired_se"] > 0
    L.settings.slow_guard_max_drift = 2.5 * v1["paired_change"]          # each step is inside its own tolerance ...
    v2, v3 = step(), step()
    assert v2["accepted"] and v2["noise_tolerance"] > v2["paired_change"]
    assert not v3["accepted"] and v3["noise_tolerance"] > v3["paired_change"]     # ... but the sum is not
    assert v3["drift_after"] > L.settings.slow_guard_max_drift and v3["reverted"]
    assert L.slow_guard_state["drift"] == pytest.approx(v2["drift_after"])
    # a promoted checkpoint is the new reference
    mgr = ckpt.CheckpointManager(str(tmp_path / "ckpt"))
    dec, _, _ = ckpt.promote_learner(mgr, L, retention={"lab/base/neg_nll": -1.0})
    assert dec.promoted and L.slow_guard_state["drift"] == 0.0
    assert step()["accepted"]


# ------------------------------------------------------------------------------------------------ m6
def test_promotion_ignores_internal_reward_metrics_by_provenance_whatever_their_name(lab_factory, tmp_path):
    lab = lab_factory(n_worlds=6, collect=True, credit=False)
    L = lab.learner
    mgr = ckpt.CheckpointManager(str(tmp_path / "ckpt"))
    info = ckpt.save_learner(mgr, L)
    ret = {"lab/base/neg_nll": -1.0}
    d = mgr.promote(info.checkpoint_id, validation={"heldout/neg_reward_return": (99.0, "internal_reward")},
                    retention=ret)
    assert not d.promoted and d.ignored == ["heldout/neg_reward_return"] and "no independent validation" in d.reasons[0]
    d = mgr.promote(info.checkpoint_id, validation={"heldout/neg_world": -1.0, "heldout/score": 3.0},
                    retention=ret, provenance={"heldout/score": "reward_model"})
    assert d.promoted and d.ignored == ["heldout/score"] and set(d.validation) == {"heldout/neg_world"}
    log = mgr.promotion_log()[-1]
    assert log["provenance"] == {"heldout/neg_world": "declared_validation", "heldout/score": "reward_model",
                                 "lab/base/neg_nll": "declared_retention"}
    # an unknown provenance is not independent evidence either
    d = mgr.promote(info.checkpoint_id, validation={"heldout/neg_world": {"value": -1.0, "provenance": "vibes"}},
                    retention=ret)
    assert not d.promoted and d.ignored == ["heldout/neg_world"]
    # promote_learner states the provenance of the learner's own held-out losses (a provider whose target is the
    # agent's reward declares it)
    L.providers[0].metric_provenance = "internal_reward"
    dec, _, _ = ckpt.promote_learner(mgr, L, retention=ret)
    assert f"heldout/neg_{L.providers[0].name}" in dec.ignored
