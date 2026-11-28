"""LRN-3 (INT3A): the bootstrap reward's learning-progress and exploration terms get their inputs in a runtime run.

Before: the agent wrote neither ``epistemic_after`` nor ``learning_progress``, so both bootstrap terms were 0 for
every decision (measured: all ``learning_progress`` / ``exploration`` components 0.0 in
``docs/evidence/validation/diagnosis/debug_f1_20261127/``). Now, as decision annotations (an added
``decision.diagnostics`` key ``reward_signals.<name>``, written once; D-14 permits adding, never changing):

* ``learning_progress`` = the world model's mean NLL on the decision's matured outcomes before minus after the next
  frequent update, on the same targets (``predict.providers.record_world_nll``, ``reward.bootstrap.
  LearningProgressTracker``);
* ``epistemic_after`` = the executed candidate's goal-value epistemic std recomputed on the decision-time head view
  with the fast belief state after it absorbed every goal-relevant outcome of that decision
  (``Predictor.goal_epistemic_after``), the same quantity as its decision-time ``uncertainty``.

The assembler reads them (``reward.assemble.stored_signals``) and the inner update's bootstrap components become
non-zero. Fails on the old code (no annotation is ever written; every component is 0).
"""
from __future__ import annotations

import math

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
from cam_live.reward.assemble import annotation_key, record_signals, stored_signals
from int_helpers import DriverState, agent_config, drive, lab_config


def _run(tmp_path, n=14):
    clock = SimClock()
    world = LabWorld(lab_config(3, outcome_horizon_s=1800.0, mean_delay_s=(300.0, 900.0)), clock)
    task = lab_task(world, "obj_lrn3")
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False, policy_update_every=4),
                         str(tmp_path / "a"), clock=clock, task=task, verifier=verifier)
    st = DriverState()
    drive(agent, world, clock, verifier, st, n)
    # let every pending outcome mature, measure it, and give the agent one more cycle of belief update + learning
    clock.advance(4 * 3600.0)
    outs = [o for o in verifier.measure(st.pending) if o.status == "observed"]
    agent.reconcile(outs, None)
    agent.update_beliefs()
    agent.learn(1e9)
    return agent, task


def test_learning_progress_and_epistemic_after_are_recorded_and_reach_the_bootstrap(tmp_path):
    agent, task = _run(tmp_path)
    try:
        recs = [r for r in agent.store.trajectory(task.objective_id) if r.result is not None]
        assert len(recs) >= 10
        lp = [r for r in recs if annotation_key("learning_progress") in r.decision.diagnostics]
        ep = [r for r in recs if annotation_key("epistemic_after") in r.decision.diagnostics]
        assert len(lp) >= len(recs) // 2, (len(lp), len(recs), agent.last.get("learning_progress_error"))
        assert ep, agent.counters
        assert all(math.isfinite(float(r.decision.diagnostics[annotation_key("learning_progress")])) for r in lp)
        assert any(abs(float(r.decision.diagnostics[annotation_key("learning_progress")])) > 1e-9 for r in lp)
        for r in ep:
            ev = r.decision.candidates[r.decision.chosen_index]
            after = float(r.decision.diagnostics[annotation_key("epistemic_after")])
            assert ev.uncertainty is not None and math.isfinite(after) and after >= 0.0
            # (it can rise in outcome units: a count's epistemic variance scales with the fast layer's mean
            # multiplier, which moves with the observation; the bootstrap rewards only a drop)
        assert any(float(r.decision.diagnostics[annotation_key("epistemic_after")]) <
                   float(r.decision.candidates[r.decision.chosen_index].uncertainty) - 1e-12 for r in ep)
        # the assembler reads both annotations
        sig, extras, _ = record_signals(lp[0], agent.store)
        assert extras["learning_progress"] == float(lp[0].decision.diagnostics[annotation_key("learning_progress")])
        sig, _, _ = record_signals(ep[0], agent.store)
        assert sig["epistemic_after"] == float(ep[0].decision.diagnostics[annotation_key("epistemic_after")])
        # ... and the inner update's bootstrap terms are no longer identically zero
        sup = [r for r in recs if agent.meta.inner.eligible(r)]
        ib = agent.meta.inner.build_batch(agent.store, sup)
        comp = ib.boot_components
        assert float(comp["learning_progress"].abs().sum()) > 0.0
        assert float(comp["exploration"].abs().sum()) > 0.0
        # every annotation is written once and never changed (the store refuses changed history)
        assert agent.last.get("annotation_error") is None
        assert agent.counters["annotated_learning_progress"] == len(lp)
        assert agent.counters["annotated_epistemic_after"] == len(ep)
    finally:
        agent.close()


def test_a_later_verified_annotation_and_the_per_signal_keys_coexist(tmp_path):
    """The verified-progress annotation (the ``reward_signals`` dict) and the per-signal keys are independent added
    keys; the dict wins where both name a signal."""
    d = {"reward_signals": {"verified_progress_delta": 2.0}, annotation_key("epistemic_after"): 0.5,
         annotation_key("verified_progress_delta"): 9.0, annotation_key("learning_progress"): 0.1}
    assert stored_signals(d) == {"verified_progress_delta": 2.0, "epistemic_after": 0.5, "learning_progress": 0.1}
    assert stored_signals({annotation_key("learning_progress"): 0.25}) == {"learning_progress": 0.25}
    assert stored_signals(None) == {}


def test_learning_progress_is_the_loss_change_on_the_same_targets(tmp_path):
    """The value written equals the world model's NLL on the record's matured outcomes before minus after the
    frequent update (measured independently here with the weights before / after)."""
    from cam_live.predict.providers import record_world_nll
    clock = SimClock()
    world = LabWorld(lab_config(5), clock)
    task = lab_task(world, "obj_lp")
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "b"), clock=clock,
                         task=task, verifier=verifier)
    try:
        st = DriverState()
        drive(agent, world, clock, verifier, st, 4)
        captured = {}
        real = agent.learner.learn

        def spy(plan):
            recs = agent._lp_records()
            agent.model.eval()
            captured["before"] = (recs, record_world_nll(agent.model, agent.heads, agent.store, recs))
            ups = real(plan)
            captured["after"] = record_world_nll(agent.model, agent.heads, agent.store, recs)
            agent.model.train()
            captured["frequent"] = any(u.timescale == "frequent" for u in ups)
            return ups
        agent.learner.learn = spy
        for _ in range(6):
            drive(agent, world, clock, verifier, st, 1)
            if captured.get("frequent") and any(b is not None for b in captured["before"][1]):
                break
        assert captured.get("frequent")
        recs, before = captured["before"]
        after = captured["after"]
        got = 0
        for r, b, a in zip(recs, before, after):
            rec = agent.store.get(r.exp_id)
            v = rec.decision.diagnostics.get(annotation_key("learning_progress"))
            if b is None or a is None:
                continue
            assert v is not None and abs(float(v) - (b - a)) < 1e-6 * max(1.0, abs(b)), (v, b, a)
            got += 1
        assert got >= 1
    finally:
        agent.close()


def test_pending_goal_views_survive_a_checkpoint(tmp_path):
    """A restored agent can still write ``epistemic_after`` for decisions made before the checkpoint (the goal views
    of executed decisions are checkpointed with the predictor state), with the same value."""
    clock = SimClock()
    world = LabWorld(lab_config(6), clock)
    task = lab_task(world, "obj_gv")
    verifier = LabVerifier(world)
    cfg = agent_config(use_backbone=False, planner=False)
    agent = CamLiveAgent(cfg, str(tmp_path / "a"), clock=clock, task=task, verifier=verifier)
    other = None
    try:
        drive(agent, world, clock, verifier, DriverState(), 5)
        views = {k: v for k, v in agent.predictor._goal_views.items() if "rows" not in v}
        assert views
        path = str(tmp_path / "ck")
        agent.save(path)
        other = CamLiveAgent(cfg, str(tmp_path / "b"), clock=clock, task=task, verifier=verifier)
        other.load(path)
        got = {k: v for k, v in other.predictor._goal_views.items() if "rows" not in v}
        assert set(got) == set(views)
        for did in views:
            for p in (agent.predictor, other.predictor):           # every query absorbed: compare the recomputation
                p._absorbed.update(f"query:{q}" for q in views[did]["queries"])
            a, b = agent.predictor.goal_epistemic_after(did), other.predictor.goal_epistemic_after(did)
            assert a is not None and a == b
    finally:
        agent.close()
        if other is not None:
            other.close()
