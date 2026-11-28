"""Documentation-audit items given to INT3A (docs/KNOWN_FAILURES.md, docs/MATH_NOTES.md sections 13-14).

* KF-E01 "credit trains nothing": the stored retrospective credit (``ExperienceRecord.credit``, written by
  ``credit.providers.CreditAssigner``) now reaches the policy-improvement update (spec 9.2: "use redistributed credit
  during eligible training updates"): the evidence-selected estimator's credit is a per-decision goal reward
  (``simple`` where the environment attributes outcomes, ``rudder``: its redistributed contribution) or the row's goal
  return (``cocoa``: its counterfactual credit replaces the discounted sum) - CR round 2 semantics
  (``credit.providers.LEARNING_KEY`` / ``RETURN_LIKE``, change request CR-4). A change of the stored credit changes
  the training target (returns) and the gradient of the inner loss; a row the selected estimator has not credited
  keeps its simple per-decision credit; without any credit the update is unchanged.
* KF-E02: the reward signal ``info_gain`` was always masked (the assembler read ``extras["info_gain"]``, the
  predictor writes ``info_gain_nats``); it is now present for decisions whose chosen candidate had a VOI estimate.
* KF-E05: R_psi's lifetime context was zeroed at every objective change; the architecture (section 4) says it is a
  summary of the lifetime that is not reset at objective boundaries. It now carries on; the objective is a new
  segment of the meta-learner's returns.

Each test fails on the code before the fix.
"""
from __future__ import annotations

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
from cam_live.reward.assemble import record_signals
from cam_live.reward.inner import verified_progress
from cam_live.reward.signals import SCALAR_NAMES
from int_helpers import DriverState, agent_config, drive, lab_config


def _agent_run(tmp_path, n=8, seed=7):
    clock = SimClock()
    world = LabWorld(lab_config(seed), clock)
    task = lab_task(world, f"obj_audit_{seed}")
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / f"a{seed}"), clock=clock,
                         task=task, verifier=verifier)
    st = DriverState()
    drive(agent, world, clock, verifier, st, n)
    return agent, task, world, clock, verifier, st


def _loss_and_grad(agent, ib):
    inner = agent.meta.inner
    theta = {k: v.requires_grad_(True) for k, v in inner.subset.snapshot().items()}
    psi = {k: v.detach() for k, v in agent.meta.reward.psi().items()}
    loss, aux = inner.functional(theta, psi, ib.with_mix(1.0), return_aux=True)
    grads = torch.autograd.grad(loss, list(theta.values()), allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return float(loss.detach()), aux["returns"].detach().clone(), flat


def _set_credit(agent, exp_id, key, value):
    agent.store.set_credits(exp_id, {key: (float(value), 0.0)}, version=agent.assigner.version + 1, reason="test",
                            ts=agent.clock.now())


def test_stored_credit_changes_the_policy_update_target_and_gradient(tmp_path):
    """KF-E01 under CR round 2's semantics (credit.providers.LEARNING_KEY / RETURN_LIKE, CR-4): simple and RUDDER give
    per-decision credit (RUDDER's learning signal is its redistributed contribution), COCOA a return-like credit that
    replaces the row's discounted goal return. In each case a change of the stored credit changes the training target
    (returns) and the gradient of the real inner loss."""
    agent, task, *_ = _agent_run(tmp_path)
    try:
        assert task.objective_id in agent.attributing                    # LabWorld attributes adopters
        recs = [r for r in agent.store.trajectory(task.objective_id) if agent.meta.inner.eligible(r)]
        assert len(recs) >= 6
        open_rows = [k for k, r in enumerate(recs) if verified_progress(r) is None]
        assert open_rows
        # (1) per-decision credit (estimator simple, the default): a row without a final verified annotation
        assert agent.assigner.default == "simple"
        i = open_rows[0]
        out = []
        for v in (3.0, -3.0):
            _set_credit(agent, recs[i].exp_id, "simple", v)
            fresh = [agent.store.get(r.exp_id) for r in recs]
            ib = agent.meta.inner.build_batch(agent.store, fresh)
            assert ib.goal_source[i] == "credit:simple" and float(ib.r_goal[i]) == v
            out.append(_loss_and_grad(agent, ib))
        assert not torch.equal(out[0][1], out[1][1])                    # the returns (training target) changed
        assert not torch.allclose(out[0][2], out[1][2])                  # ... and the gradient
        # (2) RUDDER selected: its learning signal is the stored contribution, a per-decision goal reward
        agent.selector.default = "rudder"
        assert agent.assigner.default == "rudder"
        rows = []
        for scale in (1.0, -2.0):
            for k, r in enumerate(recs):
                _set_credit(agent, r.exp_id, "rudder_contribution", scale * (k + 1))
                _set_credit(agent, r.exp_id, "rudder", 100.0 + k)         # the segment signal is not the learning key
            fresh = [agent.store.get(r.exp_id) for r in recs]
            ib = agent.meta.inner.build_batch(agent.store, fresh)
            assert ib.g_credit_ok is None                                # per decision: no return-like row
            for k in open_rows:
                assert ib.goal_source[k] == "credit:rudder" and float(ib.r_goal[k]) == scale * (k + 1)
            rows.append(_loss_and_grad(agent, ib))
        assert not torch.equal(rows[0][1], rows[1][1]) and not torch.allclose(rows[0][2], rows[1][2])
        # (3) COCOA selected: return-like - the credited goal return replaces the discounted goal rewards (not added
        # to them): with the intrinsic reward fixed, the return is the intrinsic return plus c_g * C / rms(C)
        agent.selector.default = "cocoa"
        cocoa = []
        for scale in (1.0, -2.0):
            for k, r in enumerate(recs):
                _set_credit(agent, r.exp_id, "cocoa", scale * (k + 1))
            fresh = [agent.store.get(r.exp_id) for r in recs]
            ib = agent.meta.inner.build_batch(agent.store, fresh)
            assert ib.g_credit_ok is not None and bool(ib.g_credit_ok.all())
            assert all(s == "credit:cocoa" for s in ib.goal_source)
            cocoa.append(_loss_and_grad(agent, ib))
        assert not torch.equal(cocoa[0][1], cocoa[1][1]) and not torch.allclose(cocoa[0][2], cocoa[1][2])
        inner = agent.meta.inner
        gc = inner.credited_goal_return(ib, torch.float32)
        c = ib.g_credit
        assert torch.allclose(gc, inner.cfg.goal_coef * c / torch.sqrt((c ** 2).mean()), atol=1e-6)
    finally:
        agent.close()


def test_without_credit_the_update_is_unchanged(tmp_path):
    """(a) CR-4 / G3-core D-CR1: a learned default that has not credited a row leaves the row its simple per-decision
    credit - the update is exactly the one under the default ``simple`` (changing the default removes no credit).
    (b) KF-E01: records without any stored credit give exactly the update before KF-E01 (no credit is invented)."""
    import dataclasses
    agent, task, *_ = _agent_run(tmp_path, seed=8)
    try:
        recs = [r for r in agent.store.trajectory(task.objective_id) if agent.meta.inner.eligible(r)]
        assert any("simple" in r.credit for r in recs)                    # LabWorld attributes adopters
        assert not any(k in r.credit for r in recs for k in ("rudder_contribution", "cocoa"))
        # (a) RUDDER selected, but no record carries RUDDER credit yet: the simple credit stays, labelled simple
        ib_simple = agent.meta.inner.build_batch(agent.store, recs)
        agent.selector.default = "rudder"
        ib = agent.meta.inner.build_batch(agent.store, recs)
        assert ib.goal_source == ib_simple.goal_source and any(s == "credit:simple" for s in ib.goal_source)
        a, b = _loss_and_grad(agent, ib), _loss_and_grad(agent, ib_simple)
        assert a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        # (b) without any stored credit the update equals the update without the credit hook
        bare = [dataclasses.replace(r, credit={}, credit_uncertainty={}) for r in recs]
        ib_bare = agent.meta.inner.build_batch(agent.store, bare)
        assert ib_bare.g_credit is None and not any(s.startswith("credit:") for s in ib_bare.goal_source)
        agent.meta.inner.credit_fn = None                     # the update as before KF-E01
        ib0 = agent.meta.inner.build_batch(agent.store, bare)
        c, d = _loss_and_grad(agent, ib_bare), _loss_and_grad(agent, ib0)
        assert c[0] == d[0] and torch.equal(c[1], d[1]) and torch.equal(c[2], d[2])
    finally:
        agent.close()


def test_info_gain_reaches_the_reward_signals(tmp_path):
    agent, task, *_ = _agent_run(tmp_path, n=5, seed=9)
    try:
        recs = [r for r in agent.store.trajectory(task.objective_id) if r.result is not None]
        with_voi = [r for r in recs if "info_gain_nats" in r.decision.candidates[r.decision.chosen_index].extras]
        assert with_voi, "the predictor computed VOI for the chosen candidates"
        for r in with_voi:
            sig, _, _ = record_signals(r, agent.store)
            ev = r.decision.candidates[r.decision.chosen_index]
            assert sig["info_gain"] == float(ev.extras["info_gain_nats"])
        asm = agent.meta.inner.assembler.assemble(with_voi, agent.store)
        col = SCALAR_NAMES.index("info_gain")
        assert bool((asm.mask[:, col] > 0).all())                         # before: masked for every decision
    finally:
        agent.close()


def test_the_lifetime_context_of_r_psi_is_not_reset_at_an_objective_change(tmp_path):
    agent, task, world, clock, verifier, st = _agent_run(tmp_path, n=6, seed=10)
    try:
        agent.internal_rewards()
        lc = agent.meta.reward.lifetime
        state, steps = lc.state.detach().clone(), int(lc.steps)
        assert steps > 0 and float(state.abs().sum()) > 0
        seg = agent.meta._life
        w2 = LabWorld(lab_config(11), clock)
        t2 = lab_task(w2, "obj_audit_next")
        agent.attach_verifier(LabVerifier(w2))
        agent.set_objective(t2)
        assert torch.equal(lc.state, state) and int(lc.steps) == steps     # before: zeroed
        assert agent.meta._life == seg + 1                                  # a new segment of the returns
        st2 = DriverState()
        drive(agent, w2, clock, LabVerifier(w2), st2, 3)
        agent.internal_rewards()
        recs = [r for r in agent.store.trajectory(t2.objective_id) if r.result is not None]
        first = recs[0]
        h = agent.meta._life_states.get(first.exp_id)
        assert h is not None and torch.allclose(h.cpu(), state.cpu())       # the new objective's first step sees it
    finally:
        agent.close()
