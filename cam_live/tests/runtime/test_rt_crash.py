"""kill -9 in the middle of a run, then resume: no action runs twice, no matured outcome is lost."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from cam_live.contracts import RunFiles
from cam_live.diagnostics.world import DAY, LabConfig, lab_task
from cam_live.harness.rpc import RpcServer
from cam_live.runtime.logs import RunReader, iter_jsonl
from rt_helpers import LabHost

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_child.py")


@pytest.fixture
def host(tmp_path):
    h = LabHost(LabConfig(seed=9, target=100000, horizon_s=9 * DAY, budget=60.0))
    sock = str(tmp_path / "lab.sock")
    server = RpcServer(sock, h, LabHost.METHODS, name="labhost").start()
    task = lab_task(h.world, "obj_crash")
    task.limits.max_decisions = 60
    json.dump(task.to_dict(), open(tmp_path / "task.json", "w"))
    yield h, sock, str(tmp_path / "run"), str(tmp_path / "summary.json")
    server.stop()


def _child(sock, run_dir, summary, max_iter=4000, **env):
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    return subprocess.Popen([sys.executable, CHILD, sock, run_dir, str(max_iter), summary], env=e,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish(p, timeout=120):
    out, err = p.communicate(timeout=timeout)
    return p.returncode, err.decode()[-3000:]


def _check_invariants(h, run_dir, summary):
    s = json.load(open(summary))
    r = RunReader(run_dir)
    st = r.state()
    # 1. the world ran every action id exactly once, however often the runtime asked
    assert h.real_executions and all(n == 1 for n in h.real_executions.values())
    assert h.world._n_exec == len(h.real_executions)
    # 2. the learner (restored from its checkpoint + the write-ahead journal) holds each result exactly once
    delivered = [a for a, status in s["results"] if status == "ok" and a in h.real_executions]
    assert sorted(delivered) == sorted(h.real_executions)
    # 3. counters agree with the world and with the decision log across the crash
    assert st["counters"]["actions_executed"] == h.world._n_exec
    assert abs(st["counters"]["money"] - h.world._spent) < 1e-9
    decisions = r.records(RunFiles.DECISIONS)
    assert st["counters"]["decisions"] == len({d["data"]["decision"]["decision_id"] for d in decisions})
    # 4. no outcome was lost: every query the world issued is reconciled by the learner or still pending
    pending = {i["query_id"] for i in st["live"]["pending"]["items"]}       # redacted summaries in state.json
    reconciled = set(s["reconciled"])
    assert set(h.issued_queries) <= pending | reconciled
    assert reconciled and len(s["reconciled"]) == len(reconciled)          # and none was reconciled twice
    matured = [q for q in h.issued_queries if q in reconciled]
    assert matured
    return s, st


def test_kill_with_an_action_in_flight_then_resume(host):
    h, sock, run_dir, summary = host
    code, err = _finish(_child(sock, run_dir, summary, RT_KILL_AFTER_EXECUTE=3))
    assert code == -signal.SIGKILL, err
    assert h.world._n_exec == 3
    # what the dead process left behind: a journal entry for an action that has no result yet
    wal = list(iter_jsonl(os.path.join(run_dir, "wal.jsonl")))
    gen = RunReader(run_dir).state()["committed"]["generation"]
    live = [e for e in wal if e["g"] == gen]
    begun = {e["action_id"] for e in live if e["t"] == "begin"} - {e["action_id"] for e in live if e["t"] == "end"}
    assert len(begun) == 1
    (inflight,) = begun
    assert h.real_executions[inflight] == 1 and h.execute_calls[inflight] == 1

    code, err = _finish(_child(sock, run_dir, summary))
    assert code == 0, err
    s, st = _check_invariants(h, run_dir, summary)
    # the in-flight action was asked for again under the same id and answered from the stored result
    assert h.execute_calls[inflight] == 2 and h.real_executions[inflight] == 1
    r = RunReader(run_dir)
    rec = r.records(RunFiles.EVENTS, types=["inflight_recovered"])
    assert len(rec) == 1 and rec[0]["data"]["action_id"] == inflight and rec[0]["data"]["status"] == "ok"
    resumed = r.records(RunFiles.EVENTS, types=["run_resumed"])
    assert len(resumed) == 1 and resumed[0]["data"]["inflight_recovered"] == [inflight]
    assert s["loads"] == 1
    # its delayed outcome was not lost: the query issued by the pre-crash execution was reconciled after resume
    q_inflight = [q.query_id for q in h.results[inflight].expected_outcomes]
    assert q_inflight and set(q_inflight) <= set(s["reconciled"])
    assert s["status"] in ("exhausted", "failed")


def test_kill_between_results_and_commit_then_resume(host):
    h, sock, run_dir, summary = host
    code, err = _finish(_child(sock, run_dir, summary, RT_KILL_BEFORE_SAVE=4))
    assert code == -signal.SIGKILL, err
    executed_before = dict(h.real_executions)
    assert executed_before
    code, err = _finish(_child(sock, run_dir, summary))
    assert code == 0, err
    s, st = _check_invariants(h, run_dir, summary)
    r = RunReader(run_dir)
    resumed = r.records(RunFiles.EVENTS, types=["run_resumed"])
    assert len(resumed) == 1 and resumed[0]["data"]["inflight_recovered"] == []
    assert resumed[0]["data"]["replayed_results"]                 # journalled results were delivered again
    assert all(h.execute_calls[a] == 1 for a in executed_before)  # nothing was re-issued, let alone re-run


def test_external_kill_at_an_arbitrary_moment_then_resume(host):
    h, sock, run_dir, summary = host
    p = _child(sock, run_dir, summary)
    deadline = time.time() + 60
    while h.world._n_exec < 5 and time.time() < deadline and p.poll() is None:
        time.sleep(0.002)
    os.kill(p.pid, signal.SIGKILL)                                  # a real kill -9 from outside
    code, _ = _finish(p)
    assert code == -signal.SIGKILL
    code, err = _finish(_child(sock, run_dir, summary))
    assert code == 0, err
    _check_invariants(h, run_dir, summary)
    assert len(RunReader(run_dir).records(RunFiles.EVENTS, types=["run_resumed"])) == 1
