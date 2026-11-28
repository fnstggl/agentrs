"""G2L-8 (INT3A): the prediction providers' eligibility does not read the learner's own eligibility flags.

``ContinualLearner.refresh_eligibility`` writes ``record.eligible[<provider>]`` (trainable) and ``val:<provider>``
(held out) from ``provider.eligible(record)``. The providers' ``_permitted`` read ``record.eligible[<provider>]`` back,
so (1) a held-out record — train flag False by construction — lost its ``val:`` flag at the next refresh, emptying
the world model's held-out set (the first promotion of a short objective then had no validation metric: G2L-8), and
(2) a record checked before its outcomes / verdict existed kept a False flag for ever (the evaluator never trained on
such records). Both fail on the code before the fix.
"""
from __future__ import annotations

import os
import sys

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
from cam_live.predict.providers import WorldModelLoss


def _helpers():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "integration")
    sys.path.insert(0, here)
    try:
        import int_helpers
    finally:
        sys.path.remove(here)
    return int_helpers


def test_the_learners_own_flags_do_not_decide_eligibility(tmp_path):
    h = _helpers()
    clock = SimClock()
    world = LabWorld(h.lab_config(2), clock)
    task = lab_task(world, "obj_elig")
    verifier = LabVerifier(world)
    agent = CamLiveAgent(h.agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock,
                         task=task, verifier=verifier)
    try:
        h.drive(agent, world, clock, verifier, h.DriverState(), 6)
        wm = next(p for p in agent.learner.providers if isinstance(p, WorldModelLoss))
        recs = [r for r in agent.store.trajectory(task.objective_id) if r.result is not None]
        for r in recs:
            r2 = agent.store.get(r.exp_id)
            r2.eligible.update({wm.name: False, "val:" + wm.name: True})       # what the learner wrote for a held-out
            assert wm.eligible(r2) == wm.eligible(r)                          # record: no feedback into eligibility
        agent.learner.refresh_eligibility(full=True)
        agent.learner.refresh_eligibility(full=True)                          # twice: flags must be a fixed point
        for r in (agent.store.get(x.exp_id) for x in recs):
            ok = wm.eligible(r)
            val = agent.learner.is_validation(r)
            assert r.eligible.get(wm.name) is (ok and not val)
            assert r.eligible.get("val:" + wm.name) is (ok and val)
    finally:
        agent.close()


def test_a_record_checked_before_its_verdict_becomes_eligible_for_the_evaluator(tmp_path):
    h = _helpers()
    agent = CamLiveAgent(h.agent_config(use_backbone=False, planner=False, promote_every=10_000), str(tmp_path / "b"),
                         clock=SimClock())
    try:
        lt = h.run_lifetime(h.Recorder(agent), agent.clock, str(tmp_path), 31, max_steps=30, max_decisions=14)
        agent.learner.refresh_eligibility(full=True)
        recs = [r for r in agent.store.trajectory(lt.task.objective_id) if r.result is not None]
        ev = next(p for p in agent.learner.providers if p.name == "evaluator")
        assert ev.objective(lt.task.objective_id) is not None, lt.status      # the lifetime ends with a signed verdict
        flags = [bool(r.eligible.get("evaluator")) or bool(r.eligible.get("val:evaluator")) for r in recs]
        assert sum(flags) >= len(recs) // 2, (sum(flags), len(recs))           # before: 0 (checked before the verdict)
    finally:
        agent.close()
