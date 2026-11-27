"""Outcomes still pending when an objective ends (INT3B item 1; INTEGRATION.md §8: the drain covered only an exhausted
decision budget, so outcomes pending at a verified success, a verified failure or another exhausted limit were never
measured).

The runtime now *settles* an ended objective: outcomes of its decisions that can mature within
``RuntimeConfig.settle_after_finish_s`` (and before the deadline) are measured, reconciled and learned from, and
nothing is proposed or executed meanwhile; then the objective is closed. What cannot mature in that window, and
everything pending when the operator stops the run, is *carried*: it stays in the persisted pending-outcome queue.
"""
from __future__ import annotations

import json
import os

from cam_live.contracts import GoalStatus, RunFiles
from cam_live.diagnostics.world import DAY, LabConfig
from cam_live.runtime.logs import RunReader, read_json
from cam_live.runtime.loop import COMPLETED, EXHAUSTED, STOPPED, LiveRuntime, RuntimeConfig
from rt_helpers import make_lab

#: a small target that is reached while the last interventions' attribution windows (2 days) are still open
LAB = dict(target=3, outcome_horizon_s=2 * DAY)


def _run(lab, n=600):
    for _ in range(n):
        lab.rt.step()
        if lab.rt.status != "running":
            break
    return lab.rt


def _first_achieved_seq(events):
    return min(e["seq"] for e in events if e["type"] == "verified_outcome"
               and e["data"]["outcome"]["status"] == GoalStatus.ACHIEVED)


def test_outcomes_pending_at_verified_success_are_measured_before_the_objective_closes(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, **LAB), rt_cfg=RuntimeConfig(fsync=False))
    rt = _run(lab)
    assert rt.status == COMPLETED
    r = RunReader(lab.run_dir)
    events = r.records(RunFiles.EVENTS)
    won = _first_achieved_seq(events)
    after = [e for e in events if e["type"] == "outcome" and e["seq"] > won]
    settling = [e for e in events if e["type"] == "settling"]
    fin = [e for e in events if e["type"] == "objective_finished"][-1]
    # the substance (fails on the code before INT3B: 3 outcomes stayed pending, none measured after the success)
    assert after and len(rt.pending) == 0
    # at the verified success outcomes of the objective's own decisions were still pending ...
    assert settling and settling[0]["seq"] > won and settling[0]["data"]["pending_outcomes"] > 0
    assert settling[0]["data"]["status"] == COMPLETED
    # ... and they were measured afterwards (by the verifier: they are goal-relevant), not dropped
    assert len(after) == settling[0]["data"]["pending_outcomes"]
    assert all(e["data"]["observation"]["verified"] is True for e in after)
    assert fin["seq"] > max(e["seq"] for e in after)
    assert fin["data"]["status"] == COMPLETED and fin["data"]["pending_outcomes_carried"] == 0
    assert fin["data"]["settle"]["measured"] == len(after)
    assert len(rt.pending) == 0
    # the learner received each of them through reconcile, in the same objective (no new objective in between)
    qids = {e["data"]["query"]["query_id"] for e in after}
    assert qids <= set(lab.agent.reconciled)
    # nothing was decided or executed after the verdict
    assert not [e for e in events if e["type"] in ("action_start", "decision") and e["seq"] > won]
    assert not [d for d in r.records(RunFiles.DECISIONS) if d["seq"] > won]
    # resolutions of the registered predictions exist for them
    res = {x["data"]["query_id"] for x in r.records(RunFiles.PREDICTIONS, types=["resolution"])}
    assert qids <= res
    st = r.state()
    assert st["status"] == COMPLETED and st["settling"] is None
    assert st["lifetime"]["objectives"][-1]["pending_outcomes_carried"] == 0
    rt.close()


def test_the_settle_window_is_bounded_and_what_cannot_mature_is_carried(tmp_path):
    # a 1-hour window: the 2-day attribution windows cannot mature in it
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, **LAB),
                   rt_cfg=RuntimeConfig(fsync=False, settle_after_finish_s=3600.0))
    rt = _run(lab)
    assert rt.status == COMPLETED
    events = RunReader(lab.run_dir).records(RunFiles.EVENTS)
    fin = [e for e in events if e["type"] == "objective_finished"][-1]
    assert not [e for e in events if e["type"] == "settling"]       # nothing could mature: closed at once
    carried = fin["data"]["pending_outcomes_carried"]
    assert carried > 0 and carried == len(rt.pending)
    # carried = kept in the committed, persisted queue (measured if the run continues)
    st = read_json(os.path.join(lab.run_dir, RunFiles.STATE))
    assert len(st["live"]["pending"]["items"]) == carried
    rt.close()
    # 0 turns settling off: the earlier behaviour, but the carried count is reported
    lab0 = make_lab(str(tmp_path / "off"), lab_cfg=LabConfig(seed=3, **LAB),
                    rt_cfg=RuntimeConfig(fsync=False, settle_after_finish_s=0.0))
    rt0 = _run(lab0)
    fin0 = [e for e in RunReader(lab0.run_dir).records(RunFiles.EVENTS) if e["type"] == "objective_finished"][-1]
    assert rt0.status == COMPLETED and fin0["data"]["pending_outcomes_carried"] == len(rt0.pending) > 0
    rt0.close()


def test_settling_never_runs_past_the_deadline(tmp_path):
    def edit(task):
        task.limits.max_decisions = 3

    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000, outcome_horizon_s=2 * DAY,
                                                            horizon_s=1 * DAY), task_edit=edit,
                   rt_cfg=RuntimeConfig(fsync=False))
    rt = _run(lab)
    assert rt.status == EXHAUSTED and rt.status_reason == "limit max_decisions exhausted"
    assert lab.clock.now() <= rt._deadline() + 1e-6
    fin = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS) if e["type"] == "objective_finished"][-1]
    # outcomes due after the deadline are carried, not waited for
    assert fin["data"]["pending_outcomes_carried"] == len(rt.pending) > 0
    assert all(i.ready_at > rt._deadline() for i in rt.pending.items())
    assert rt.counters["decisions"] == 3
    rt.close()


def test_outcomes_pending_when_another_limit_is_used_up_are_measured(tmp_path):
    def edit(task):
        task.limits.max_external_actions = 3

    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=5, target=10000, outcome_horizon_s=6 * 3600.0),
                   task_edit=edit, rt_cfg=RuntimeConfig(fsync=False))
    rt = _run(lab)
    assert rt.status == EXHAUSTED and "max_external_actions" in rt.status_reason
    events = RunReader(lab.run_dir).records(RunFiles.EVENTS)
    s = [e for e in events if e["type"] == "settling"]
    fin = [e for e in events if e["type"] == "objective_finished"][-1]
    assert len(rt.pending) == 0                  # fails on the code before INT3B (outcomes left pending)
    assert s and s[0]["data"]["status"] == EXHAUSTED
    assert fin["data"]["settle"]["measured"] >= s[0]["data"]["maturing_in_window"] > 0
    assert fin["data"]["pending_outcomes_carried"] == 0 == len(rt.pending)
    assert not [e for e in events if e["type"] == "action_start" and e["seq"] > s[0]["seq"]]
    rt.close()


def test_a_stopped_run_carries_its_pending_outcomes(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000, outcome_horizon_s=2 * DAY),
                   rt_cfg=RuntimeConfig(fsync=False))
    rt = lab.rt
    for _ in range(40):
        rt.step()
        if len(rt.pending) >= 2:
            break
    n = len(rt.pending)
    assert n >= 2
    rt.request_stop("operator stop (test)")
    rt.step()
    assert rt.status == STOPPED
    halted = [e for e in RunReader(lab.run_dir).records(RunFiles.EVENTS) if e["type"] == "halted"][-1]
    assert halted["data"]["pending_outcomes_carried"] == n
    st = read_json(os.path.join(lab.run_dir, RunFiles.STATE))
    assert len(st["live"]["pending"]["items"]) == n
    rt.close()


def test_settling_resumes_after_a_restart(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, **LAB), rt_cfg=RuntimeConfig(fsync=False))
    rt = lab.rt
    for _ in range(600):
        rt.step()
        if rt._settle is not None or rt.status != "running":
            break
    assert rt._settle is not None and rt.status == "running"
    pending = len(rt.pending)
    assert pending > 0
    st = read_json(os.path.join(lab.run_dir, RunFiles.STATE))
    assert st["settling"]["status"] == COMPLETED and st["settling"]["pending_at_end"] == pending
    del rt                                       # the process is gone (no close, no further commit)
    lab.rt.log.close()
    rt2 = LiveRuntime(lab.agent, lab.world, lab.policy, lab.verifier, lab.task, lab.run_dir, lab.clock,
                      RuntimeConfig(fsync=False))
    rt2.start()
    assert rt2._settle is not None and rt2._settle["status"] == COMPLETED
    for _ in range(600):
        rt2.step()
        if rt2.status != "running":
            break
    assert rt2.status == COMPLETED and len(rt2.pending) == 0
    events = RunReader(lab.run_dir).records(RunFiles.EVENTS)
    assert [e for e in events if e["type"] == "run_resumed"]
    fin = [e for e in events if e["type"] == "objective_finished"]
    assert len(fin) == 1 and fin[0]["data"]["pending_outcomes_carried"] == 0
    assert not [e for e in events if e["type"] == "action_start" and e["seq"] > _first_achieved_seq(events)]
    rt2.close()
