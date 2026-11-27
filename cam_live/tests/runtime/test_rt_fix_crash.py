"""Regression tests for the crash-recovery defects of G_rt.md (D2, D3, D14, D15, D18): kill -9 of the runtime process
at the points the verifier found, then a clean restart; the world, verifier and policy live in this test process
(served over the project RPC), so what really executed can be compared with what the runtime logged and delivered."""
import json
import os
import signal
import subprocess
import sys

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
    task.limits.max_decisions = 14
    json.dump(task.to_dict(), open(tmp_path / "task.json", "w"))
    yield h, sock, str(tmp_path / "run"), str(tmp_path / "summary.json")
    server.stop()


def _run(sock, run_dir, summary, timeout=180, **env):
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    p = subprocess.run([sys.executable, CHILD, sock, run_dir, "4000", summary], env=e, capture_output=True,
                       timeout=timeout)
    return p.returncode, p.stderr.decode()[-3000:]


def _killed_then_resumed(host, **kill):
    h, sock, run_dir, summary = host
    code, err = _run(sock, run_dir, summary, **kill)
    assert code == -signal.SIGKILL, err
    code, err = _run(sock, run_dir, summary, **{k: v for k, v in kill.items() if k == "RT_NO_DEDUP"})
    assert code == 0, err
    return h, run_dir, json.load(open(summary))


def _invariants(h, run_dir, s, dedup=True):
    r = RunReader(run_dir)
    st = r.state()
    ev = r.records(RunFiles.EVENTS)
    results = {}
    for e in ev:
        if e["type"] == "action_result":
            results.setdefault(e["data"]["action_id"], []).append(e)
    # every action the world executed has exactly one action_result record and reached the learner once
    assert all(a in results for a in h.real_executions), "an executed action has no action_result record"
    assert all(len(v) == 1 for v in results.values()), "an action has two action_result records"
    delivered = [a for a, _status in s["results"]]
    assert len(delivered) == len(set(delivered))
    assert all(a in delivered for a in h.real_executions)
    assert all(n == 1 for n in h.real_executions.values())
    # every decision record is closed by one result; the counter equals what the learner holds
    decisions = [d["data"]["decision"]["decision_id"] for d in r.records(RunFiles.DECISIONS)]
    closed = {e["data"]["decision_id"] for e in ev if e["type"] == "action_result"}
    assert set(decisions) <= closed, "a decision record has no result record"
    assert st["counters"]["decisions"] == len(delivered) == len(set(decisions))
    # outcomes: one record per query, delivered once; one resolution per prediction
    oq = [e["data"]["query"]["query_id"] for e in ev if e["type"] == "outcome"]
    assert len(oq) == len(set(oq)), "two outcome records for one query"
    assert len(s["reconciled"]) == len(set(s["reconciled"])) and set(s["reconciled"]) == set(oq)
    res = [p["data"]["prediction_id"] for p in r.records(RunFiles.PREDICTIONS, types=["resolution"])]
    assert len(res) == len(set(res))
    if dedup:
        assert abs(st["counters"]["money"] - h.world._spent) < 1e-9
        assert st["counters"]["external_actions"] == h.world._n_exec
    # the journal never holds a glued (unparseable) line
    raw = open(os.path.join(run_dir, "wal.jsonl"), "rb").read().splitlines()
    assert all(json.loads(x) for x in raw if x.strip())
    return r, st


def test_D2_kill_during_a_wait_loses_no_decision(host):
    h, run_dir, s = _killed_then_resumed(host, RT_KILL_AT="in_wait_sleep:2")
    r, st = _invariants(h, run_dir, s)
    waits = [e for e in r.records(RunFiles.EVENTS, types=["action_result"]) if e["data"]["kind"] == "wait"]
    assert waits and st["counters"]["waits"] == len(waits)


def test_D3_a_journalled_result_reaches_events_after_restart(host):
    h, run_dir, s = _killed_then_resumed(host, RT_KILL_AT="after_wal_end:2")
    r, _ = _invariants(h, run_dir, s)
    late = [e for e in r.records(RunFiles.EVENTS, types=["action_result"]) if e["data"].get("written_after_restart")]
    assert len(late) == 1 and late[0]["data"]["action_id"] in h.real_executions


def test_D14_a_torn_journal_line_does_not_swallow_the_next_entry(host):
    h, run_dir, s = _killed_then_resumed(host, RT_KILL_AT="torn_wal_begin:2")
    r, _ = _invariants(h, run_dir, s)
    # the decision whose execution entry was torn was never executed and is closed as such
    closed = [e for e in r.records(RunFiles.EVENTS, types=["action_result"]) if e["data"]["kind"] == "interrupted"]
    assert len(closed) == 1 and closed[0]["data"]["action_id"] not in h.real_executions


def test_D15_kill_between_decision_record_and_journal_closes_the_decision(host):
    h, run_dir, s = _killed_then_resumed(host, RT_KILL_AT="after_decision_log:3")
    r, _ = _invariants(h, run_dir, s)
    resumed = r.records(RunFiles.EVENTS, types=["run_resumed"])[0]["data"]
    assert len(resumed["decisions_closed_unhandled"]) == 1


def test_D15_kill_after_an_outcome_record_gives_no_second_outcome(host):
    h, run_dir, s = _killed_then_resumed(host, RT_KILL_AT="after_outcome_log:1")
    r, _ = _invariants(h, run_dir, s)
    resumed = r.records(RunFiles.EVENTS, types=["run_resumed"])[0]["data"]
    assert resumed["replayed_outcomes"] >= 1


def test_D18_inflight_action_without_dedup_is_booked_at_its_estimate(host):
    h, run_dir, s = _killed_then_resumed(host, RT_NO_DEDUP=1, RT_KILL_AT="after_wal_end:0", RT_KILL_AFTER_EXECUTE=3)
    r, st = _invariants(h, run_dir, s, dedup=False)
    rec = r.records(RunFiles.EVENTS, types=["inflight_recovered"])
    assert len(rec) == 1 and rec[0]["data"]["status"] == "cancelled"
    res = [e for e in r.records(RunFiles.EVENTS, types=["action_result"])
           if e["data"]["action_id"] == rec[0]["data"]["action_id"]][0]
    assert res["data"]["cost"]["money"] > 0 and "estimated cost" in res["data"]["failure_detail"]
    # the world spent it; the runtime booked the estimate (LabWorld's estimate is its price): never under-counted
    assert st["counters"]["money"] >= h.world._spent - 1e-9
    assert st["counters"]["external_actions"] == h.world._n_exec


def test_D7_journal_lines_are_sealed(host):
    h, sock, run_dir, summary = host
    code, err = _run(sock, run_dir, summary, RT_KILL_AFTER_EXECUTE=3)
    assert code == -signal.SIGKILL, err
    lines = list(iter_jsonl(os.path.join(run_dir, "wal.jsonl")))
    assert lines and all(set(e) <= {"g", "t", "kind", "action_id", "x"} and e["x"].startswith("v1:") for e in lines)
    st = json.load(open(os.path.join(run_dir, RunFiles.STATE)))
    assert st["sealed"].startswith("v1:") and "decision" not in json.dumps(st.get("approvals"))
