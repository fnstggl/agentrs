"""Storage guard with the integrated agent (INT2): two runtime checkpoints, optimizer state only in the newest; the
learner's promotions check the guard and a refusal pauses the run."""
from __future__ import annotations

import json
import os

import pytest

from cam_live.contracts import RunFiles
from cam_live.runtime.loop import RuntimeConfig
from cam_live.storage import DiskGuard, DiskSpaceLow


def _events(run_dir, t):
    p = os.path.join(run_dir, RunFiles.EVENTS)
    return [e for e in (json.loads(l) for l in open(p)) if e["type"] == t]


def test_two_checkpoints_kept_and_optimizer_state_only_in_the_newest(tmp_path):
    from cam_live.agent import CamLiveAgent
    from int_helpers import agent_config, lab_config
    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task, AllowAllPolicy
    from cam_live.runtime.loop import LiveRuntime
    assert RuntimeConfig().keep_checkpoints == 2
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock)
    world = LabWorld(lab_config(4), clock)
    rt = LiveRuntime(agent, world, AllowAllPolicy(), LabVerifier(world), lab_task(world, "obj_k"),
                     str(tmp_path / "run"), clock, RuntimeConfig(fsync=False, candidates_k=6))
    try:
        for _ in range(6):
            rt.step()
        root = os.path.join(rt.run_dir, RunFiles.CHECKPOINTS)
        ck = sorted(n for n in os.listdir(root) if n.startswith("ckpt_"))
        assert len(ck) == 2, ck
        assert not os.path.isdir(os.path.join(root, ck[0], "optim"))
        assert os.path.isdir(os.path.join(root, ck[1], "optim"))
    finally:
        rt.close()
        agent.close()


def test_agent_promotion_checks_the_guard_and_the_runtime_pauses(tmp_path):
    from cam_live.agent import CamLiveAgent
    from int_helpers import Recorder, agent_config, run_lifetime
    from cam_live.contracts import SimClock
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False, retention_records=6),
                         str(tmp_path / "a"), clock=clock)
    try:
        rec = Recorder(agent)
        run_lifetime(rec, clock, str(tmp_path), 41, max_steps=12, max_decisions=8)
        n_promo = len(agent.promotions)
        agent.guard = DiskGuard(agent.dir, 1e9, 2e9)
        with pytest.raises(DiskSpaceLow):
            agent.promote("test")
        assert len(agent.promotions) == n_promo                     # nothing written, nothing decided
        # the refusal at the objective boundary reaches the runtime through learn(): the run pauses
        lt2 = run_lifetime(rec, clock, str(tmp_path), 42, max_steps=4, max_decisions=4,
                           rt_cfg=RuntimeConfig(fsync=False, candidates_k=6, min_free_gb=0.0, warn_free_gb=0.0))
        assert lt2.status == "paused", lt2.status
        assert any(e["data"]["what"].startswith("learner.") for e in _events(lt2.run_dir, "storage_low"))
    finally:
        agent.close()


