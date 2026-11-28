"""SG change request SG-2 / SG-3 (agent.py, owned by ST): outcomes of a later assigned objective reach goal-free mode's
learned transfer estimate through the real agent API, and the evidence survives save / load.

Skipped where the hook (``CamLiveAgent._transfer_evidence``) is absent; ST integrated it in dde7d2d. Debug model on
CPU, LabWorld (diagnostic)."""
import os
import sys

import pytest

from cam_live.agent import CamLiveAgent
from cam_live.contracts import Limits, SimClock, TaskSpec, Trust

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "integration"))

pytestmark = [pytest.mark.skipif(not hasattr(CamLiveAgent, "_transfer_evidence"),
                                 reason="change request SG-2 not applied to agent.py yet"),
              pytest.mark.timeout(2400)]                         # a debug-model lifetime with learning on a loaded host


def test_assigned_objective_outcomes_become_transfer_evidence_and_persist(tmp_path):
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
    from int_helpers import DriverState, agent_config, drive, lab_config
    clock = SimClock()
    world = LabWorld(lab_config(71), clock)
    verifier = LabVerifier(world)
    agent = CamLiveAgent(agent_config(seed=9), str(tmp_path / "agent"), clock=clock, verifier=verifier)
    try:
        agent.set_objective(TaskSpec(objective_id="free_71", objective="", success={}, family="lab/integration",
                                     limits=Limits(duration_s=2 * 86400.0, money=40.0, max_decisions=40),
                                     mode="diagnostic", origin=Trust.SELF))
        st = DriverState()
        drive(agent, world, clock, verifier, st, 6)
        assert agent.goal_free.selector.history, "no goal-free learning goal was resolved"
        assert agent.counters["transfer_evidence"] == 0                 # nothing during the goal-free period
        agent.set_objective(lab_task(world, "assigned_71"))
        assert not agent._goal_free_active()
        drive(agent, world, clock, verifier, st, 6)
        clock.advance(12 * 3600.0)                                     # the world's outcomes are delayed: let them mature
        drive(agent, world, clock, verifier, st, 2)
        n = agent.counters["transfer_evidence"]
        assert n > 0 and len(agent.goal_free.transfer.points) > 0
        assert all(p["objective_id"] == "assigned_71" for p in agent.goal_free.transfer.points)
        keys = {p["key"] for p in agent.goal_free.transfer.points}
        assert any(agent.goal_free.transfer.estimate(k)["mean"] != 0.5 for k in keys)
        agent.save(str(tmp_path / "ck"))
        k = sorted(keys)[0]
        before = (len(agent.goal_free.transfer.points), agent.goal_free.transfer.estimate(k)["mean"])
    finally:
        agent.close()
    twin = CamLiveAgent(agent_config(seed=9), str(tmp_path / "agent"), clock=clock, verifier=verifier)
    try:                                               # a restart: same run directory, state from the checkpoint
        twin.load(str(tmp_path / "ck"))
        assert (len(twin.goal_free.transfer.points), twin.goal_free.transfer.estimate(k)["mean"]) == \
            (before[0], pytest.approx(before[1]))
        assert twin.goal_free.restore_notes == []                       # reference prior restored, not re-taken
    finally:
        twin.close()
