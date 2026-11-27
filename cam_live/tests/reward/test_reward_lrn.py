"""Regression tests for the learning-mechanism fixes of change request S-5 (workstream LRN; spec 7.3, 7.6, 7.8).

1. The bootstrap's delta V compares the same function at both ends (V(b') - V(b)), so it has no action-dependent
   offset (V(b') - Q(b, a) paid the actions the evaluator rated low).
2. Verified progress in a decision's reward is only *its own* attributed outcome, never another decision's reading
   that arrived during the step (waits and probes were paid for earlier interventions).
3. The inner update is invariant to the scale of R_psi at every mix (integrity), follows R_psi's *shape* with a
   weight that grows as the mix moves toward the learned reward, and ignores R_psi completely at mix 1.
4. The policy-improvement loss uses the verified progress credited to each decision (and, while pending, the
   calibrated prediction), so a policy update moves the policy toward the decisions that produced progress.
5. The live policy step is sized by a trust region (target KL) and recorded, so it changes behaviour and the
   meta-gradient still differentiates exactly the update that was applied.
6. The selection rule's calibration is fed by the learner with final verified outcomes, once each, and persists.
"""
import dataclasses
import math

import pytest
import torch

from cam_live.contracts import CandidateEvaluation, ExperienceItem, ItemKind, OutcomeQuery, Quantity, Trust
from cam_live.predict.evaluator import EvaluatorConfig, GoalEvaluator
from cam_live.predict.policy import SelectionCalibration, selection_calibration
from cam_live.reward import diagnose as dg
from cam_live.reward.assemble import STORED_KEY, SignalAssembler, record_signals
from cam_live.reward.inner import InnerConfig, predicted_goal_value, verified_progress
from cam_live.reward.integrity import params_hash
from cam_live.reward.meta import flat


def _stack(seed=0, inner_lr=0.1, **kw):
    ec = dg.ExperimentConfig(inner_lr=inner_lr)
    model, store, clock, ml = dg.make_stack(seed, ec, **kw)
    lt, world, _ = dg.lab_lifetime(ml, store, clock, ec, seed=4242 + seed, update_every=0)
    return ec, model, store, clock, ml, lt


def _annotate(store, recs, *, y_of=None, q_of=None):
    """Give each record's candidates a goal prediction and (y_of) a final verified progress annotation."""
    out = []
    for k, r in enumerate(recs):
        d = r.decision
        cands = [dataclasses.replace(c, extras={**(c.extras or {}), "q_goal": float(q_of(c.action) if q_of else 1.0),
                                                "net_cost": 0.0, "policy_logit": 0.0, "v_state": 0.5})
                 for c in d.candidates]
        diag = dict(d.diagnostics or {})
        if y_of is not None:
            diag[STORED_KEY] = {"verified_progress_delta": float(y_of(d.chosen))}
        r.decision = dataclasses.replace(d, candidates=cands, diagnostics=diag)
        store.update(r)
        out.append(r)
    return out


def _act_value(a):
    return 5.0 * float(a.arguments.get("effort", 0)) if a.operation == "lab.act" else 0.0


# ------------------------------------------------------------------------------------------------ 1. delta V
def test_delta_v_has_no_action_dependent_offset():
    ec, model, store, clock, ml, lt = _stack(0)
    recs = _annotate(store, lt.records[:8])
    # candidates whose Q(b, a) (p_success) differs strongly by action: the old source of success_before
    for k, r in enumerate(recs):
        c = r.decision.candidates[r.decision.chosen_index]
        cands = list(r.decision.candidates)
        cands[r.decision.chosen_index] = dataclasses.replace(c, p_success=0.05 if k % 2 else 0.95)
        r.decision = dataclasses.replace(r.decision, candidates=cands)
    sig, _, _ = record_signals(recs[0], store)
    assert sig["success_before"] == 0.5                            # the recorded state value, not Q = 0.95
    ev = GoalEvaluator(EvaluatorConfig(d_model=model.d_model, hidden=16))
    asm = SignalAssembler(evaluator=ev)
    b = torch.randn(len(recs), model.d_model)
    e_g = torch.randn(len(recs), model.d_model)
    out = asm.assemble(recs, store, b=b, b_next=b.clone(), e_g=e_g,
                       has_next=torch.ones(len(recs), dtype=torch.bool))
    dv = out.signals.delta_v
    # nothing changed between b and b' (only the remaining horizon, by the step's duration): delta V ~ 0 whatever the
    # action's Q was; the old pairing V(b') - Q(b, a) would have been ~ +-0.45 here, alternating with the action
    old = out.signals.success_after - torch.tensor([r.decision.candidates[r.decision.chosen_index].p_success
                                                    for r in recs], dtype=torch.float64)
    assert dv is not None and float(dv.abs().max()) < 1e-2 and float(old.abs().min()) > 0.3
    b2 = b + 0.5 * torch.randn_like(b)
    out2 = asm.assemble(recs, store, b=b, b_next=b2, e_g=e_g, has_next=torch.ones(len(recs), dtype=torch.bool))
    with torch.no_grad():
        want = ev(b2, e_g, torch.full((len(recs),), 86400.0 - 0.0)).p_success.double() - \
            ev(b, e_g, torch.full((len(recs),), 86400.0)).p_success.double()
    got = out2.signals.delta_v
    assert float(got.abs().max()) > 0
    # the same function at both ends (horizons differ only by the step's duration)
    assert torch.allclose(got, want, atol=0.05)


# --------------------------------------------------------------------------------- 2. attributed verified delta
def test_verified_progress_counts_only_this_decisions_own_outcomes():
    ec, model, store, clock, ml, lt = _stack(1)
    rec = next(r for r in lt.records if r.post_item_refs)
    other = ExperienceItem(kind=ItemKind.OUTCOME, text="another decision's outcome", ts=rec.decision.ts + 1.0,
                           quantities=[Quantity("lab.adopters.from.someone_else", 9.0)], ref="q_not_mine:out",
                           trust=Trust.VERIFIER)
    store.put_item(other)
    gq = OutcomeQuery(query_id="q_mine", metric="lab.adopters.from.me", kind="count", horizon_s=3600.0,
                      issued_at=rec.decision.ts, goal_relevant=True)
    mine = ExperienceItem(kind=ItemKind.OUTCOME, text="this decision's outcome", ts=rec.decision.ts + 2.0,
                          quantities=[Quantity("lab.adopters.from.me", 3.0)], ref="q_mine:out", trust=Trust.VERIFIER)
    store.put_item(mine)
    r2 = dataclasses.replace(rec, post_item_refs=list(rec.post_item_refs) + [other.ref], queries=list(rec.queries))
    sig, _, _ = record_signals(r2, store)
    base = sig["verified_progress_delta"]
    assert base == 0.0                                            # a wait is not paid for someone else's adopters
    r3 = dataclasses.replace(r2, post_item_refs=r2.post_item_refs + [mine.ref], queries=list(rec.queries) + [gq])
    assert record_signals(r3, store)[0]["verified_progress_delta"] == 3.0
    # the agent's signed annotation takes precedence
    r4 = dataclasses.replace(r3, decision=dataclasses.replace(
        r3.decision, diagnostics={**r3.decision.diagnostics, STORED_KEY: {"verified_progress_delta": 2.0}}))
    assert record_signals(r4, store)[0]["verified_progress_delta"] == 2.0 and verified_progress(r4) == 2.0
    assert verified_progress(rec) is None


# ------------------------------------------------------------------------------- 3. scale vs shape of R_psi
def _update(ml, ib, psi, lr=0.1):
    theta0 = ml.subset.snapshot()
    th = ml.engine.inner_update(theta0, psi, [ib], lr, 1, create_graph=False).detached()
    return flat({n: th[n] - theta0[n] for n in th})


def test_inner_update_is_scale_invariant_at_every_mix_and_follows_the_shape_of_the_learned_reward():
    ec, model, store, clock, ml, lt = _stack(2, mix=1.0)
    recs = _annotate(store, lt.records[:10], y_of=_act_value)      # a bootstrap reward that varies across rows
    ib0 = ml.inner.build_batch(store, recs)
    assert float(ib0.r_boot.std()) > 0
    psi = {k: v.detach().clone() for k, v in ml.reward.psi().items()}
    g = torch.Generator().manual_seed(0)
    reshaped = {k: (v + 0.5 * v.std(unbiased=False) * torch.randn(v.shape, generator=g, dtype=v.dtype) if k.startswith("out.")
                    else v) for k, v in psi.items()}
    bound = ml.reward.bound
    shape_effect = {}
    for mix in (1.0, 0.75, 0.5, 0.0):
        ib = ib0.with_mix(mix)
        d1 = _update(ml, ib, psi)
        for c in (10.0, 1000.0):                                  # R_psi -> c * R_psi exactly
            ml.reward.bound = bound * c
            dc = _update(ml, ib, {k: (v * c if k.startswith("out.") else v) for k, v in psi.items()})
            ml.reward.bound = bound
            assert float((dc - d1).abs().max()) <= 1e-5 * max(1.0, float(d1.abs().max())), (mix, c)
        shape_effect[mix] = float((_update(ml, ib, reshaped) - d1).norm())
    assert shape_effect[1.0] == 0.0                               # pure bootstrap: R_psi has no say at all
    assert 0 < shape_effect[0.75] < shape_effect[0.5] < shape_effect[0.0]


# ----------------------------------------------------------------------------- 4. goal term in the inner loss
def test_policy_update_moves_toward_decisions_with_verified_progress():
    out = {}
    for coef in (0.0, 1.0):
        ec, model, store, clock, ml, lt = _stack(3, mix=1.0, target_kl=0.0, goal_coef=coef)
        recs = _annotate(store, lt.records[:16], y_of=_act_value, q_of=_act_value)
        ib = ml.inner.build_batch(store, recs)
        assert ib.goal_source == ["verified"] * len(recs)
        by_id = {r.exp_id: r for r in recs}
        y = torch.tensor([verified_progress(by_id[e]) for e in ib.exp_ids])
        assert float(y.std()) > 0
        lp0 = ml.inner.logp(ml.subset.snapshot(), ib)
        ml.policy_update(recs, inner_lr=0.5)
        lp1 = ml.inner.logp(ml.subset.snapshot(), ib)
        sign = torch.sign(y - y.mean())
        out[coef] = float(((lp1 - lp0) * sign).mean())
    # with the goal term the update raises log pi of the decisions that brought verified progress, lowers the rest
    assert out[1.0] > 0.0 and out[1.0] > out[0.0] + 0.01


def test_pending_decisions_use_the_calibrated_prediction_labelled_as_such():
    ec, model, store, clock, ml, lt = _stack(4, mix=1.0, goal_coef=1.0)
    recs = _annotate(store, lt.records[:6], q_of=_act_value)                  # nothing verified yet
    ib = ml.inner.build_batch(store, recs)
    assert set(ib.goal_source) == {"predicted"}
    by_id = {r.exp_id: r for r in recs}
    assert [float(x) for x in ib.r_goal] == [predicted_goal_value(by_id[i]) for i in ib.exp_ids]
    # with verified evidence the prediction is calibrated: a world model whose ranking predicts nothing gives
    # every pending decision the same expectation
    cal = selection_calibration(ml.policy)
    for k in range(60):
        cal.observe(f"x{k}", float((k % 7) - 3), 0.0, 2.0 + (k % 3))
    ib2 = ml.inner.build_batch(store, recs)
    assert set(ib2.goal_source) == {"predicted"} and float(ib2.r_goal.std()) < 0.5


def test_all_candidate_term_moves_policy_mass_away_from_candidates_predicted_worse():
    """The executed-action policy gradient cannot lower never-chosen candidates; the model term does: after an update
    on decisions that all executed the same kind of action, the policy's mass on the candidates the world model
    predicts worse (here: every non-intervention) goes down, and goes down further with the term than without."""
    res = {}
    for coef in (0.0, 1.0):
        ec, model, store, clock, ml, lt = _stack(7, mix=1.0, target_kl=0.0, model_coef=coef)
        recs = _annotate(store, lt.records[:16], q_of=_act_value)
        ib = ml.inner.build_batch(store, recs)
        z = torch.tensor([SelectionCalibration.candidate_values(r.decision) for r in
                          sorted(recs, key=lambda r: ib.exp_ids.index(r.exp_id))], dtype=torch.float32)
        if coef:
            assert bool(ib.z_ok.all()) and ib.model_weight == pytest.approx(1.0)    # prior beta_q before evidence
            assert torch.allclose(ib.z_pred, z)
        else:
            assert ib.z_pred is None
        worse = (z < 0) & ib.a_mask
        p0 = ml.inner.probs(ml.subset.snapshot(), ib)
        ml.policy_update(recs, inner_lr=0.5)
        p1 = ml.inner.probs(ml.subset.snapshot(), ib)
        res[coef] = float(((p1 - p0) * worse).sum(-1).mean())
    assert res[1.0] < 0.0 and res[1.0] < res[0.0] - 0.01


# ------------------------------------------------------------------------------------------- 5. trust region
def test_live_policy_step_is_sized_by_a_trust_region_and_replays_exactly():
    ec, model, store, clock, ml, lt = _stack(5, inner_lr=0.01, mix=1.0, target_kl=0.05, max_inner_lr=5.0,
                                             goal_coef=1.0)
    recs = _annotate(store, lt.records[:12], y_of=_act_value, q_of=_act_value)
    _, _, fstore, _, fixed, flt = _stack(5, inner_lr=0.01, mix=1.0, target_kl=0.0, goal_coef=1.0)
    frecs = _annotate(fstore, flt.records[:12], y_of=_act_value, q_of=_act_value)
    small = fixed.policy_update(frecs)
    up = ml.policy_update(recs)
    kl_small, kl = small.prediction_change["policy_kl"], up.prediction_change["policy_kl"]
    assert kl_small < 1e-3                                         # the old fixed step: behaviour does not change
    ev = ml.events[-1]
    assert 0.05 / 1.5 <= kl <= 0.05 * 1.5 or ev.inner_lr == 5.0, (kl, up.diagnostics["step_search"])
    assert ev.inner_lr > 0.01 and f"lr={ev.inner_lr:g}" in up.notes
    assert up.diagnostics["step_search"][-1][0] == pytest.approx(ev.inner_lr)
    # the meta-gradient differentiates exactly the applied update: replaying it from the snapshot is bit-identical
    psi = {k: v.detach() for k, v in ml.reward.psi().items()}
    replay = ml.engine.inner_update(ev.theta0, psi, [ev.ib], ev.inner_lr, ev.steps, create_graph=False).detached()
    assert ml.logged_policy_is_current([ev], replay)


# ------------------------------------------------------------------------------------ 6. calibration feed
def test_learner_feeds_the_selection_calibration_once_per_final_outcome_and_persists_it():
    ec, model, store, clock, ml, lt = _stack(6, mix=1.0)
    assert ml.selection is selection_calibration(ml.policy)
    n0 = len(ml.selection)
    recs = _annotate(store, lt.records[:10], q_of=_act_value)               # predictions, nothing verified yet
    assert ml.refresh_selection() == 0
    recs = _annotate(store, recs, y_of=_act_value, q_of=_act_value)
    assert ml.refresh_selection() == len(recs) and ml.refresh_selection() == 0
    assert len(ml.selection) == n0 + len(recs)
    z, c = SelectionCalibration.features(recs[0].decision)
    assert (z, c, _act_value(recs[0].decision.chosen)) in ml.selection.obs
    sd = ml.state_dict()
    ec2, model2, store2, clock2, ml2, _ = _stack(6, mix=1.0)
    ml2.load_state_dict(sd)
    assert ml2.selection.posterior() == ml.selection.posterior() and ml2.selection is selection_calibration(ml2.policy)


# ----------------------------------------------------------------------------------------- 7. INT2-8 restore
def test_restored_update_events_keep_the_support_batch_the_live_update_used():
    """INT2-8: a support batch rebuilt from the store after a restore is not the one the live update used (the encoder
    and the reward's input statistics moved since). The events that will still be differentiated are checkpointed
    with their batch, bit-exactly; the meta-gradient of a restored learner differentiates the same update."""
    from cam_live.reward.metalearner import MetaLearner
    ec, model, store, clock, ml, lt = _stack(8, mix=1.0)
    lt2, _, ups = dg.lab_lifetime(ml, store, clock, ec, seed=9000, update_every=4)
    assert len(ml.events) >= 2 and all(e.ib is not None for e in ml.events)
    sd = ml.state_dict()
    saved = [e["ib"] is not None for e in sd["events"]]
    assert saved[-1] and any(saved)
    # the continual learner moves the encoder after the save: a batch rebuilt from the store is then different
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not n.startswith(("heads.reward", "reward")) and p.dtype.is_floating_point:
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    old = ml.events[-1]
    rebuilt = ml.inner.build_batch(store, [r for r in (store.get(i) for i in old.exp_ids) if r is not None])
    assert not torch.equal(old.ib.b, rebuilt.b)
    twin = MetaLearner(model, store, ml.cfg, meta_cfg=ml.mc, inner_cfg=ml.inner.cfg, ledger=ml.ledger, clock=clock,
                       check_signature=ml.ledger.check_signature)
    twin.load_state_dict(sd)
    for e, f, kept in zip(ml.events, twin.events, saved):
        if not kept:
            assert f.ib is None
            continue
        for name in ("b", "e_a", "a_mask", "e_g", "chosen", "behaviour_logp", "ts", "b_next", "signals",
                     "signal_mask", "r_boot", "gamma", "r_goal"):
            x, y = getattr(e.ib, name), getattr(f.ib, name)
            assert (x is None) == (y is None) and (x is None or torch.equal(x, y.to(x.device))), name
        assert f.ib.goal_source == e.ib.goal_source and f.ib.segments == e.ib.segments and f.ib.mix == e.ib.mix
        assert all(torch.equal(e.ib.norm_state[k], f.ib.norm_state[k].to(e.ib.norm_state[k].device))
                   for k in e.ib.norm_state)
        # the restored event replays the live update bit-identically from its snapshot
        psi = {k: v.detach() for k, v in ml.reward.psi().items()}
        if not e.external and e.theta1_hash:
            th = twin.engine.inner_update(f.theta0, psi, [f.ib], f.inner_lr, f.steps, create_graph=False).detached()
            assert params_hash(th) == e.theta1_hash
