"""MetaLearner on the real (debug) model in LabWorld: windows, verified utility, outer step, promotion, integrity."""
import dataclasses

import pytest
import torch

from cam_live.contracts import GoalStatus, MetaUpdateRecord, Trust
from cam_live.diagnostics.collect import LabCollector, verified_return
from cam_live.diagnostics.world import LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.reward import diagnose as dg
from cam_live.reward.assemble import STORED_KEY
from cam_live.reward.integrity import IntegrityError, noisy_tv_audit, params_hash
from cam_live.reward.meta import ESTIMATOR_IMPORTANCE, ESTIMATOR_ON_POLICY
from cam_live.reward.metalearner import MetaConfig, MetaLearner


def _stack(seed=0, **kw):
    ec = dg.ExperimentConfig(min_independent=kw.pop("min_independent", 3), inner_steps=kw.pop("inner_steps", 2))
    return (ec,) + dg.make_stack(seed, ec, **kw)


def _lifetimes(ml, store, clock, ec, n, seed0, admit=True, **kw):
    out = []
    for i in range(n):
        ml.new_lifetime()
        lt, world, ups = dg.lab_lifetime(ml, store, clock, ec, seed=seed0 + i, **kw)
        if admit:
            assert ml.admit_objective(lt.verified, utility=verified_return(lt)).accepted
        out.append((lt, world, ups))
    return out


def test_single_lifetime_windows_and_on_policy_meta_update():
    """One lifetime, verifier measurements authenticated by the verifier: support windows precede query windows,
    the update is on-policy (the query was collected by the policy the unroll reproduces) and psi moves."""
    auth = dg.LabMeasurementAuth()
    ec, model, store, clock, ml = _stack(0, mix=0.0, check_measurement=auth)
    ec.lab.update(horizon_s=10 * 86400.0, budget=300.0)           # room for all 18 decisions
    lt, world, ups = dg.lab_lifetime(ml, store, clock, ec, seed=11, n_decisions=18, update_every=6, auth=auth)
    assert len(lt.records) == 18 and len(ups) == 3 and len(ml.events) == 3
    for u in ups:                                                # the real inner update is logged
        assert u.timescale == "frequent" and u.hash_before["reward"] == u.hash_after["reward"]
        assert u.hash_before["adaptable"] != u.hash_after["adaptable"]
        assert u.prediction_change["policy_prob_mean_abs_change"] > 0
    by_id = {r.exp_id: r for r in lt.records}
    for ev in ml.events[:-1]:
        assert ev.steps == ec.inner_steps and ev.inner_lr == ec.inner_lr and len(ev.query_ids) == 6
        assert max(by_id[i].decision.ts for i in ev.exp_ids) < min(by_id[i].decision.ts for i in ev.query_ids)
    h0, v0 = params_hash(ml.reward), ml.reward.version
    theta = params_hash(ml.subset.live())
    rec = ml.meta_update()
    assert isinstance(rec, MetaUpdateRecord) and rec.applied and rec.skipped_reason == ""
    assert rec.estimator == ESTIMATOR_ON_POLICY and rec.max_importance_weight == 1.0
    assert rec.n_independent_outcomes >= 3 and rec.n_query >= rec.n_independent_outcomes
    assert rec.inner_steps == ec.inner_steps and rec.inner_lr == ec.inner_lr and rec.meta_grad_norm > 0
    assert rec.reward_version_before == v0 and rec.reward_version_after == ml.reward.version != v0
    assert params_hash(ml.reward) != h0 and params_hash(ml.subset.live()) == theta      # only psi moved
    assert rec.query_window.endswith("@" + ml.ledger.head) and rec.bootstrap_mix == 0.0
    d = rec.diagnostics
    assert d["psi_tensors_with_gradient"] == d["psi_tensors"] - 1    # all but the null goal (no goal-free rows)
    assert d["policy_prob_change_on_query"] > 0 and d["clip_scale"] <= 1.0
    assert all(isinstance(v, float) for v in d.values())
    assert MetaUpdateRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()
    again = ml.meta_update()                                      # each verified outcome supervises one update
    assert not again.applied and again.skipped_reason.startswith("insufficient_independent_outcomes")
    assert again.reward_version_after == rec.reward_version_after


def test_skipped_update_reports_the_reason_and_leaves_psi_bit_identical():
    ec, model, store, clock, ml = _stack(1, mix=0.0, min_independent=50)
    none = ml.meta_update()
    assert not none.applied and none.skipped_reason == "no_policy_update_recorded"
    _lifetimes(ml, store, clock, ec, 3, 20)
    h = params_hash(ml.reward)
    rec = ml.meta_update()
    assert not rec.applied and rec.skipped_reason == "insufficient_independent_outcomes:3<50"
    assert rec.n_independent_outcomes == 3 and params_hash(ml.reward) == h
    assert rec.reward_version_before == rec.reward_version_after
    assert ml.ready() == (False, "insufficient_independent_outcomes:3<50")
    ml.mc.min_independent = 3                                     # nothing was consumed by the skipped attempt
    assert ml.ready() == (True, "")
    assert ml.meta_update().applied and params_hash(ml.reward) != h


def test_unauthenticated_measurements_never_supervise():
    """LabWorld's verifier measurements carry ``verified=True`` but no signature (change request C-1): without an
    authentication callable they are rejected and cannot drive a meta-update."""
    ec, model, store, clock, ml = _stack(2, mix=0.0, min_independent=1)
    lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=31)
    assert any(o.verified for r in lt.records for o in r.outcomes)
    h = params_hash(ml.reward)
    rec = ml.meta_update()
    assert not rec.applied and rec.n_independent_outcomes == 0 and params_hash(ml.reward) == h
    assert len(ml.ledger) == 0 and {r for r, _ in ml.ledger.rejected} == {"no_measurement_authentication"}
    n_rej = len(ml.ledger.rejected)
    ml.meta_update()
    assert len(ml.ledger.rejected) == n_rej                       # each observation is offered once


def test_fabricated_unsigned_duplicated_imagined_outcomes_never_enter_the_outer_objective():
    r = dg.integrity_check(seed=2)
    assert r["ledger_entries_after_invalid_objectives"] == 0 and r["all_invalid_rejected"]
    assert r["rejections"] == {
        "tampered_value_with_old_signature": "bad_signature", "unsigned": "unsigned",
        "fabricated_signature": "bad_signature", "self_authored_objective": "self_authored_objective",
        "duplicate": "duplicate", "measurement_with_unknown_query": "bad_signature",
        "measurement_value_altered": "bad_signature", "measurement_unverified_flag": "not_verified",
        "measurement_imagined": "imagined", "measurement_self_sourced": "self_authored"}
    assert r["genuine_objective_accepted"] and not r["fabricated_value_in_any_query_utility"]
    assert r["independent_groups"] == 3 and r["ledger_chain_ok"]
    assert r["meta_update"]["applied"] and r["meta_update"]["n_independent"] == 3 and r["psi_changed_by_meta_update"]
    s = r["reward_scaling"]
    for c in ("output_x10", "output_x1000"):                      # r -> c * r changes nothing
        assert s[c]["max_abs_diff"] < 1e-6 and abs(s[c]["update_norm_ratio"] - 1) < 1e-4
    for c in ("pre_bound_x10", "pre_bound_x1000"):                # inflating the pre-bound output is capped
        assert s[c]["max_abs_reward"] <= s[c]["bound"] and s[c]["cosine"] > 0.9
        assert s[c]["update_norm_ratio"] * r["update_norm_unscaled"] <= r["update_norm_bound_per_step"] + 1e-6


def test_ledger_tampering_stops_meta_learning():
    ec, model, store, clock, ml = _stack(3, mix=0.0)
    _lifetimes(ml, store, clock, ec, 3, 40)
    e = ml.ledger._entries[1]
    ml.ledger._entries[1] = dataclasses.replace(e, utility=e.utility + 100.0)
    with pytest.raises(IntegrityError):
        ml.meta_update()


def test_shadow_update_with_importance_weights_while_bootstrap_drives_the_policy():
    """Cold start: the live policy is updated with the bootstrap reward (mix 1); the differentiated update uses the
    learned reward from the same snapshot and is evaluated on the logged decisions with importance weights."""
    ec, model, store, clock, ml = _stack(4, mix=1.0, on_policy_tol=0.0, min_ess=1.0)
    _lifetimes(ml, store, clock, ec, 4, 50)
    assert all(ev.mix == 1.0 for ev in ml.events)
    h = params_hash(ml.reward)
    rec = ml.meta_update()
    assert rec.applied and rec.estimator == ESTIMATOR_IMPORTANCE and params_hash(ml.reward) != h
    assert 0.5 < rec.max_importance_weight < ml.mc.weight_clip and rec.diagnostics["ess"] >= 1.0
    assert rec.bootstrap_mix == 1.0 and ml.mixer.mix == 1.0       # no held-out evidence, no promotion
    assert all(r.reward_version.startswith("bootstrap:") for lt in [store.all()] for r in lt)


def test_promotion_moves_the_mix_only_on_held_out_evidence():
    ec, model, store, clock, ml = _stack(5, mix=1.0, mix_step=0.25, heldout=0.5, mixer_tolerance=1.0,
                                         min_independent=2)
    ml.mixer.min_independent = 2
    recs = []
    for i in range(30):                                           # until the held-out buffer allowed a decision
        _lifetimes(ml, store, clock, ec, 1, 70 + i)
        m = ml.meta_update()
        if m.applied:
            recs.append(m)
        if ml.mixer.decisions:
            break
    assert recs and ml.mixer.decisions
    dec = ml.mixer.decisions[0]
    assert dec.n_independent >= 2 and dec.action == "lower" and dec.mix_after == 0.75   # margin 1.0: no harm shown
    assert any(m.diagnostics["promotion"] == 1.0 for m in recs)
    assert all(m.diagnostics["n_heldout_groups"] >= 0 for m in recs)
    held = {e.group for e in ml.ledger.entries() if ml._is_held(e.group)}
    assert held and len(held) < len(ml.ledger)                    # some groups were reserved, not all
    # demanding superiority by an impossible margin moves the mix back up
    ml.mixer.tolerance = -1e9
    before, n0 = ml.mixer.mix, len(ml.mixer.decisions)
    for i in range(30):
        _lifetimes(ml, store, clock, ec, 1, 200 + i)
        ml.meta_update()
        if len(ml.mixer.decisions) > n0:
            break
    assert len(ml.mixer.decisions) > n0
    assert ml.mixer.decisions[-1].action == "raise" and ml.mixer.mix == min(1.0, before + 0.25)


def test_external_optimizer_path_and_unroll_across_two_events():
    """A learner that steps theta with its own optimizer announces the update first; the unroll can cover two
    consecutive update events (K = their steps together)."""
    ec, model, store, clock, ml = _stack(6, mix=0.0, min_independent=2, unroll_events=2, inner_steps=1, min_ess=1.0)
    opt = torch.optim.SGD(list(ml.subset.live().values()), lr=0.05)
    ml.check_learner_optimizers([opt])
    with pytest.raises(IntegrityError):
        ml.check_learner_optimizers([torch.optim.SGD(list(ml.reward.parameters()), lr=0.1)])
    from cam_live.learning_api import encode_decisions
    for i in range(3):
        ml.new_lifetime()
        lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=120 + i, update_every=0)
        for block in (lt.records[:8], lt.records[8:]):
            ev = ml.note_policy_update(block)
            assert ev.external and ev.steps == 1
            loss, _ = ml.inner.loss(model, encode_decisions(model, store, block, with_next=True, grad=True), store)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ml.admit_objective(lt.verified, utility=verified_return(lt))
    rec = ml.meta_update()
    assert rec.applied and rec.inner_steps == 2 and rec.n_independent_outcomes >= 2
    assert rec.estimator in (ESTIMATOR_ON_POLICY, ESTIMATOR_IMPORTANCE, "mixed_on_policy_and_importance")


def test_state_round_trip_rebuilds_windows_from_the_store():
    ec, model, store, clock, ml = _stack(7, mix=0.5)
    _lifetimes(ml, store, clock, ec, 3, 140)
    sd = ml.state_dict()
    twin = MetaLearner(model, store, ml.cfg, policy=ml.policy, reward=ml.reward, ledger=ml.ledger, clock=clock,
                       meta_cfg=ml.mc, inner_cfg=ml.inner.cfg)
    twin.load_state_dict(sd)
    assert [e.exp_ids for e in twin.events] == [e.exp_ids for e in ml.events]
    assert [e.query_ids for e in twin.events] == [e.query_ids for e in ml.events]
    assert all(e.ib is None for e in twin.events) and twin.mixer.mix == 0.5
    assert twin.ready() == ml.ready() == (True, "")
    rec = twin.meta_update()
    assert rec.applied and rec.n_independent_outcomes == 3
    bad = dict(sd, ledger_head="f" * 64)
    with pytest.raises(IntegrityError):
        twin.load_state_dict(bad)


def test_repeated_evaluations_of_one_objective_credit_progress_once():
    ec, model, store, clock, ml = _stack(8, mix=0.0)
    world = LabWorld(LabConfig(seed=5, **ec.lab), clock=clock)
    verifier, task = LabVerifier(world), lab_task(world)
    col = LabCollector(store, max_context=ec.max_context)
    act = dg.model_policy(ml, store, torch.Generator().manual_seed(0))
    first = col.run(world, act, 8, task=task, seed=1, finish=False, on_step=lambda r: ml.internal_rewards([r]))
    a = ml.admit_objective(first.verified)
    second = col.run(world, act, 8, task=task, seed=2, finish=True, on_step=lambda r: ml.internal_rewards([r]))
    b = ml.admit_objective(second.verified)
    assert a.accepted and b.accepted and second.verified.value >= first.verified.value
    assert a.entry.utility == first.verified.value
    assert b.entry.utility == second.verified.value - first.verified.value        # the change, not the total again
    assert set(a.entry.exp_ids) == {r.exp_id for r in first.records}
    assert set(b.entry.exp_ids) == {r.exp_id for r in second.records}
    assert a.entry.group != b.entry.group
    assert not ml.admit_objective(second.verified).accepted


def test_noisy_gauge_earns_no_sustained_bootstrap_reward():
    """A context whose epistemic uncertainty never comes down across steps (the static gauge) loses its
    curiosity-type reward: the gate advances once per real step, and replay does not advance it."""
    ec, model, store, clock, ml = _stack(9, mix=1.0)
    world = LabWorld(LabConfig(seed=3, **{**ec.lab, "horizon_s": 10 * 86400.0}), clock=clock)

    def only_gauge(cands, info):
        i = next(k for k, c in enumerate(cands) if c.operation == "lab.noise")
        return i, [1.0 / len(cands)] * len(cands)
    rewards, comps = [], []

    def on_step(rec):
        rec.decision.diagnostics[STORED_KEY] = {"epistemic_before": 1.0, "epistemic_after": 0.9}
        store.update(rec)
        out = ml.internal_rewards([rec])
        rewards.append(out["bootstrap"][0])
        comps.append(float(ml.mixer.log[-1]["bootstrap_components"]["exploration"][0]))
        ml.inner.build_batch(store, [rec])                        # replay: must not move the gate
    LabCollector(store, max_context=ec.max_context).run(world, only_gauge, 24, seed=0, on_step=on_step)
    assert comps[0] == pytest.approx(0.05) and comps[-1] < 0.01 * comps[0]
    assert all(b <= a + 1e-12 for a, b in zip(comps, comps[1:]))
    audit = noisy_tv_audit(rewards, [0.9] * len(rewards))
    assert audit["epistemic_flat_in_tail"] and audit["passed"] and audit["tail_ratio"] < 0.05
