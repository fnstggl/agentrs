"""G2L-10 (CL-10 / DX-2): a restored agent keeps the authorization configuration in force and records a change."""
import os
import shutil

from int_helpers import Recorder, agent_config, run_lifetime

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock, TaskSpec
from cam_live.credit.checkpoint import auth_config_hash


def _checkpoint(tmp_path):
    cfg = agent_config(policy_update_every=6)
    agent = CamLiveAgent(cfg, str(tmp_path / "run" / "agent"), clock=SimClock())
    rec = Recorder(agent)
    run_lifetime(rec, agent.clock, str(tmp_path / "run"), 11, max_steps=16, max_decisions=8)
    ck = str(tmp_path / "ckpt")
    agent.save(ck)
    task = agent.task
    agent.close()
    return cfg, ck, task


def test_a_changed_authorization_is_recorded_and_the_current_limits_are_kept(tmp_path):
    cfg, ck, task = _checkpoint(tmp_path)
    changed = TaskSpec.from_dict(task.to_dict())
    changed.limits.max_decisions = int(changed.limits.max_decisions or 0) + 999
    assert auth_config_hash(changed) != auth_config_hash(task)
    shutil.copytree(str(tmp_path / "run"), str(tmp_path / "run2"))
    a = CamLiveAgent(cfg, str(tmp_path / "run2" / "agent"), clock=SimClock())
    try:
        a.set_objective(changed)
        a.load(ck)
        assert a.restore_report.get("auth_config_changed") is True
        assert a.counters["auth_config_changed_on_load"] == 1
        assert a.task.limits.max_decisions == changed.limits.max_decisions      # not the checkpoint's old limit
    finally:
        a.close()


def test_a_restart_without_a_task_compares_when_the_runtime_sets_one(tmp_path):
    cfg, ck, task = _checkpoint(tmp_path)
    shutil.copytree(str(tmp_path / "run"), str(tmp_path / "run3"))
    a = CamLiveAgent(cfg, str(tmp_path / "run3" / "agent"), clock=SimClock())
    try:
        a.load(ck)                                   # restart: the checkpoint's task comes back
        assert a.task is not None and a.counters["auth_config_changed_on_load"] == 0
        same = TaskSpec.from_dict(task.to_dict())
        a.set_objective(same)                        # same authorization: nothing recorded
        assert a.counters["auth_config_changed_on_load"] == 0
    finally:
        a.close()
    shutil.copytree(str(tmp_path / "run"), str(tmp_path / "run4"))
    b = CamLiveAgent(cfg, str(tmp_path / "run4" / "agent"), clock=SimClock())
    try:
        b.load(ck)
        changed = TaskSpec.from_dict(task.to_dict())
        changed.limits.max_decisions = int(changed.limits.max_decisions or 0) + 7
        b.set_objective(changed)                     # the operator's task differs from the checkpoint's
        assert b.counters["auth_config_changed_on_load"] == 1
        assert b.restore_report.get("auth_config_changed") is True
        assert b.task.limits.max_decisions == changed.limits.max_decisions
    finally:
        b.close()
