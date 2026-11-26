"""Operator control, approvals, objective assignment and verified-outcome integrity."""
import os

import pytest
import yaml

from cam_live.contracts import GoalStatus, RunFiles
from cam_live.diagnostics.world import DAY, AllowAllPolicy, LabConfig, LabVerifier, LabWorld
from cam_live.runtime.control import OperatorControl
from cam_live.runtime.logs import RunReader
from cam_live.runtime.loop import COMPLETED, PAUSED, RUNNING, STOPPED, LiveRuntime, RuntimeConfig
from cam_live.runtime.tasks import TaskValidationError
from rt_helpers import ApprovalPolicy, make_lab
from rt_lab_agent import LabAgent


def _queue_one_approval(tmp_path, wait_ok=True):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000),
                   agent_kw={"only_ops": ["lab.unlock", "lab.probe"]}, policy=ApprovalPolicy(("lab.unlock",)))
    for _ in range(200):
        lab.rt.step()
        if lab.rt.approvals:
            break
    assert len(lab.rt.approvals) == 1
    (aid, entry), = lab.rt.approvals.items()
    return lab, aid, entry["decision"]["chosen"]["action_id"]


def test_approval_required_action_queues_and_is_not_reproposed(tmp_path):
    lab, aid, action_id = _queue_one_approval(tmp_path)
    r = RunReader(lab.run_dir)
    # the learner was told once that the action is queued; nothing was executed
    assert lab.agent.results.count((action_id, "needs_approval")) == 1
    assert lab.world._unlocked_at is None
    st = r.state()
    assert st["approvals"][aid]["status"] == "pending" and "lab.unlock" in st["approvals"][aid]["preview"]
    # the loop continues with other work, and the queued action is never proposed again while it waits
    probes_before = sum(1 for e in lab.agent.experience if e["operation"] == "lab.probe")
    for _ in range(25):
        lab.rt.step()
    assert sum(1 for e in lab.agent.experience if e["operation"] == "lab.probe") > probes_before
    unlock_decisions = [d for d in r.records(RunFiles.DECISIONS)
                        if d["data"]["decision"]["chosen"]["operation"] == "lab.unlock"]
    assert len(unlock_decisions) == 1
    assert r.records(RunFiles.EVENTS, types=["candidates_withheld"])
    assert len(lab.rt.approvals) == 1 and len(lab.policy.queue) == 1          # no second approval request
    assert lab.world._unlocked_at is None
    # a queued action is not an obstacle
    assert all(o["data"]["obstacle"]["action_id"] != action_id for o in r.obstacles())


def test_approved_action_executes_exactly_once(tmp_path):
    lab, aid, action_id = _queue_one_approval(tmp_path)
    lab.agent.only_ops = {"lab.probe"}
    cid = OperatorControl(lab.run_dir).approve(aid, note="ok")
    for _ in range(10):
        lab.rt.step()
    r = RunReader(lab.run_dir)
    st = r.state()
    assert st["control_results"][cid]["ok"] is True
    assert st["approvals"][aid]["status"] == "executed" and "auth" not in st["approvals"][aid]
    starts = [e for e in r.records(RunFiles.EVENTS, types=["action_start"]) if e["data"]["action_id"] == action_id]
    assert len(starts) == 1 and starts[0]["data"]["approval_id"] == aid
    assert lab.world._unlocked_at is not None
    assert [s for a, s in lab.agent.results if a == action_id] == ["needs_approval", "ok"]
    # the approval executed an earlier decision; it is not counted as a new one
    assert st["counters"]["decisions"] == len(r.records(RunFiles.DECISIONS)) == lab.agent.calls["select"]


def test_rejected_action_is_reported_once_as_denied(tmp_path):
    lab, aid, action_id = _queue_one_approval(tmp_path)
    lab.agent.only_ops = {"lab.probe"}
    OperatorControl(lab.run_dir).reject(aid, note="not now")
    for _ in range(15):
        lab.rt.step()
    r = RunReader(lab.run_dir)
    assert r.state()["approvals"][aid]["status"] == "reported"
    assert [s for a, s in lab.agent.results if a == action_id] == ["needs_approval", "denied"]
    assert lab.world._unlocked_at is None
    assert not [e for e in r.records(RunFiles.EVENTS, types=["action_start"]) if e["data"]["action_id"] == action_id]
    obs = [o["data"]["obstacle"] for o in r.obstacles() if o["data"]["obstacle"]["action_id"] == action_id]
    assert len(obs) == 1 and obs[0]["category"] == "permission" and obs[0]["failure_class"] == "missing_permission"
    # the obstacle is linked to what followed: the alternatives considered and the one chosen
    assert obs[0]["chosen_alternative"] and obs[0]["alternatives"] and obs[0]["outcome"]


def test_approval_resolved_at_the_authorization_service_by_the_operator(tmp_path):
    lab, aid, action_id = _queue_one_approval(tmp_path)
    lab.agent.only_ops = {"lab.probe"}
    # the operator's own handle resolves it at the policy; the runtime only receives a notice
    OperatorControl(lab.run_dir, policy=lab.policy).approve(aid)
    assert lab.policy.queue[aid]["status"] == "approved"
    for _ in range(10):
        lab.rt.step()
    st = RunReader(lab.run_dir).state()
    assert st["approvals"][aid]["status"] == "executed"
    assert [s for a, s in lab.agent.results if a == action_id] == ["needs_approval", "ok"]
    assert lab.world._n_exec >= 1 and lab.world._unlocked_at is not None


def test_pause_resume_stop(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000))
    op = OperatorControl(lab.run_dir)
    for _ in range(5):
        lab.rt.step()
    cid = op.pause("inspection")
    assert lab.rt.run(max_iterations=50) == PAUSED
    st = RunReader(lab.run_dir).state()
    assert st["status"] == PAUSED and st["control_results"][cid]["ok"]
    assert os.path.exists(os.path.join(st["committed"]["ckpt"]["path"], "agent.json"))
    assert st["committed"]["ckpt"]["id"] == lab.agent.checkpoint_id
    calls, n_exec, t = dict(lab.agent.calls), lab.world._n_exec, lab.clock.now()
    lab.rt.run(max_iterations=20)                                   # paused: nothing happens
    assert lab.agent.calls == calls and lab.world._n_exec == n_exec and lab.clock.now() == t
    op.resume()
    for _ in range(10):
        lab.rt.step()
    assert lab.rt.status == RUNNING and lab.agent.calls["select"] > calls["select"]
    op.stop("done for today")
    assert lab.rt.run(max_iterations=50) == STOPPED
    st = RunReader(lab.run_dir).state()
    assert st["status"] == STOPPED and st["status_reason"] == "done for today"
    cmds = [e["data"]["cmd"] for e in RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["operator_command"])]
    assert cmds == ["pause", "resume", "stop"]
    # a stopped run stays stopped when the process is started again on the same directory
    rt2 = LiveRuntime(LabAgent(lab.clock), lab.world, lab.policy, lab.verifier, lab.task, lab.run_dir, lab.clock,
                      RuntimeConfig(fsync=False))
    n_exec = lab.world._n_exec
    assert rt2.run(max_iterations=5) == STOPPED and lab.world._n_exec == n_exec


def test_emergency_stop_through_the_policy_engine(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000))
    for _ in range(5):
        lab.rt.step()
    OperatorControl(lab.run_dir, policy=lab.policy).stop("halt", emergency=True)
    assert lab.policy.stopped()
    n_exec = lab.world._n_exec
    assert lab.rt.run(max_iterations=20) == STOPPED
    assert lab.world._n_exec == n_exec
    assert os.path.exists(os.path.join(RunReader(lab.run_dir).state()["committed"]["ckpt"]["path"], "agent.json"))


def _lab_yaml(path, objective_id, target, extra=None):
    raw = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "lab_staging.yaml")))
    raw["objective_id"] = objective_id
    raw["success"]["target"] = target
    raw["objective"] = f"Reach {target} adopters in the lab world before time runs out."
    raw.update(extra or {})
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return str(path)


def test_assign_subsequent_objective(tmp_path):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=3, budget=1000.0))
    assert lab.rt.run(max_iterations=2000) == COMPLETED
    op = OperatorControl(lab.run_dir)
    # a task file carrying a way of achieving the objective is refused before it reaches the runtime ...
    bad = _lab_yaml(tmp_path / "bad.yaml", "lab-second", 6, {"strategy": "anything"})
    with pytest.raises(TaskValidationError):
        op.assign_objective(bad)
    # ... and by the runtime itself if it is put on the control channel directly
    cid = op.channel.send("assign_objective", path=bad)
    lab.rt.step()
    res = RunReader(lab.run_dir).state()["control_results"][cid]
    assert res["ok"] is False and "strategy" in res["detail"]
    assert lab.rt.status == COMPLETED and lab.rt.task.objective_id == "obj_lab"
    # a clean task file starts the next objective; the learner object (and its weights) carries over
    good = _lab_yaml(tmp_path / "good.yaml", "lab-second", lab.world.total() + 400)
    before = dict(lab.rt.lifetime)
    version = lab.agent.version
    op.assign_objective(good)
    lab.rt.step()
    assert lab.rt.status == RUNNING and lab.rt.task.objective_id == "lab-second"
    assert lab.agent.objectives == ["obj_lab", "lab-second"] and lab.agent.version == version
    assert lab.rt.counters["decisions"] <= 1 and lab.rt.lifetime["decisions"] >= before["decisions"]
    assert [o["objective_id"] for o in lab.rt.lifetime["objectives"]] == ["obj_lab"]
    ev = RunReader(lab.run_dir).records(RunFiles.EVENTS, types=["objective_assigned"])
    assert ev and ev[-1]["trust"] == "operator" and ev[-1]["data"]["task"]["objective_id"] == "lab-second"


class _Tampering:
    """A channel that alters outcomes after the verifier signed them (or strips the signature)."""

    def __init__(self, inner, mode):
        self.inner, self.mode = inner, mode

    def verify(self, task):
        o = self.inner.verify(task)
        if self.mode == "tamper":
            o.status, o.value = GoalStatus.ACHIEVED, o.target        # signature no longer matches
        elif self.mode == "unsigned":
            o.status, o.value, o.signature = GoalStatus.ACHIEVED, o.target, ""
        elif self.mode == "forged":
            o.status, o.value, o.signature = GoalStatus.ACHIEVED, o.target, "0" * 64
        return o

    def check_signature(self, outcome):
        return self.inner.check_signature(outcome)

    def measure(self, queries):
        return self.inner.measure(queries)


@pytest.mark.parametrize("mode", ["tamper", "unsigned", "forged"])
def test_unsigned_or_tampered_verified_outcomes_are_refused(tmp_path, mode):
    lab = make_lab(str(tmp_path / "run"), lab_cfg=LabConfig(seed=3, target=10000, horizon_s=2 * DAY),
                   verifier=None)
    lab.rt.verifier = _Tampering(LabVerifier(lab.world), mode)
    status = lab.rt.run(max_iterations=300)
    r = RunReader(lab.run_dir)
    refused = r.records(RunFiles.EVENTS, types=["verified_outcome_refused"])
    assert refused and not r.records(RunFiles.EVENTS, types=["verified_outcome"])
    assert lab.agent.verified_seen == []                       # nothing unverified reached the learner as verified
    assert status != COMPLETED and r.state()["refused_verified"] == len(refused)
    assert r.state()["last_verified"] == {}


def test_environment_measurements_are_never_marked_verified(tmp_path):
    class Claiming(LabWorld):
        def measure(self, queries, *, _verified=False):
            return super().measure(queries, _verified=True)    # the environment claims verification

    class NoMeasure(LabVerifier):
        def measure(self, queries):
            return []                                          # the verifier cannot answer these queries

    from cam_live.contracts import SimClock
    from cam_live.diagnostics.world import lab_task
    clock = SimClock()
    world = Claiming(LabConfig(seed=5, target=10000, horizon_s=5 * DAY), clock)
    agent = LabAgent(clock, only_ops=["lab.act", "wait"], wait_seconds=DAY)
    rt = LiveRuntime(agent, world, AllowAllPolicy(), NoMeasure(world), lab_task(world, "obj"),
                     str(tmp_path / "run"), clock, RuntimeConfig(fsync=False))
    rt.run(max_iterations=300)
    outs = RunReader(str(tmp_path / "run")).records(RunFiles.EVENTS, types=["outcome"])
    assert outs and all(o["data"]["observation"]["verified"] is False for o in outs)
    assert all(o["trust"] == "untrusted" for o in outs)
