"""The runtime against the real computer environment and authorization service (in-process, no browser).

Skipped when those packages are not importable. The assertions are about the runtime's side of the contract.
"""
import json
import os

import pytest

from cam_live.contracts import RunFiles, SimClock
from cam_live.runtime import tasks as T
from cam_live.runtime.logs import LOG_POINTER, RunLog, RunReader
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
from cam_live.runtime.verifier import JsonMetricSource, build_verifier
from rt_lab_agent import LabAgent

TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tasks")


def _task(tmp_path, max_decisions=14):
    raw = T.load_task_raw(os.path.join(TASKS, "utility_staging.yaml"))
    metrics = str(tmp_path / "metrics.json")
    json.dump({"utility.distinct_successful_sessions": 0}, open(metrics, "w"))
    raw["success"]["evidence_source"] = "json:" + metrics
    raw["resources"]["staging_site"] = "http://127.0.0.1:9"
    raw["limits"]["max_decisions"] = max_decisions
    return T.task_from_dict(raw), metrics


def test_runtime_on_the_real_environment_and_authorization_service(tmp_path):
    policy_mod = pytest.importorskip("cam_live.harness.auth.policy")
    service_mod = pytest.importorskip("cam_live.harness.service")
    run_dir = str(tmp_path / "run")
    clock = SimClock()
    task, metrics = _task(tmp_path)
    policy = policy_mod.PolicyService(run_dir, clock=clock)
    policy.register_task(task, "operator")
    env = service_mod.build_environment(os.path.join(run_dir, "exec"), policy, provenance="staging", clock=clock,
                                        browser=False)
    verifier = build_verifier(task, os.path.join(run_dir, "verifier"), clock, JsonMetricSource(metrics),
                              log_path=os.path.join(run_dir, RunFiles.VERIFIER))
    agent = LabAgent(clock, seed=2)
    rt = LiveRuntime(agent, env, policy, verifier, task, run_dir, clock, RuntimeConfig(fsync=False))
    status = rt.run(max_iterations=200)
    r = RunReader(run_dir)
    st = r.state()
    assert status in ("exhausted", "failed") and st["counters"]["decisions"] >= 1
    results = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])]
    assert results and len(results) == len(agent.results)
    executed = [d for d in results if d["kind"] == "execute"]
    if executed:                                   # executed through the real environment under a real grant
        auths = {e["data"]["action_id"]: e["data"] for e in r.records(RunFiles.EVENTS, types=["authorization"])}
        assert all(auths[d["action_id"]]["verdict"] == "allow" for d in executed if d["status"] == "ok")
        assert st["counters"]["actions_executed"] == sum(1 for d in executed if d["status"] not in ("denied",))
    assert st["last_verified"]["outcome"]["target"] == 20
    assert env.get_capabilities()["features"]["at_most_once_by_action_id"] is True


def test_runtime_logs_move_aside_when_the_run_directory_names_are_taken(tmp_path):
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    foreign = os.path.join(run_dir, RunFiles.EVENTS)
    with open(foreign, "w") as f:
        f.write(json.dumps({"seq": 1, "src": "another component", "type": "started"}) + "\n")
    clock = SimClock()
    log = RunLog(run_dir, clock, durable=False)
    rid = log.event("note", {"k": 1})
    log.close()
    assert open(foreign).read().count("\n") == 1                       # the other writer's file is left alone
    ptr = json.load(open(os.path.join(run_dir, LOG_POINTER)))
    assert ptr["log_dir"] == "learner" and "events.jsonl" in ptr["reason"]
    r = RunReader(run_dir)
    assert [x["rid"] for x in r.records(RunFiles.EVENTS)] == [rid]     # every reader follows the pointer
    assert RunLog(run_dir, clock, durable=False).next_seq == 1         # and a restart continues the same files
    # foreign lines inside a shared file are ignored by readers and by the sequence counter
    shared = str(tmp_path / "shared")
    log = RunLog(shared, clock, durable=False)
    log.event("a", {})
    with open(os.path.join(shared, RunFiles.EVENTS), "a") as f:
        f.write(json.dumps({"seq": 999, "type": "foreign"}) + "\n")
    log.event("b", {})
    log.close()
    assert [x["type"] for x in RunReader(shared).records(RunFiles.EVENTS)] == ["a", "b"]
    assert RunLog(shared, clock, durable=False).next_seq == 2
