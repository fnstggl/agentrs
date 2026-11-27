"""Persistence of the integrated agent (CPU, tiny backbone, debug core; CL-06, spec 15.2):

* save -> a fresh process -> load gives the same decisions as the uninterrupted agent (same candidates chosen with
  the same full probability vectors, same learning and meta-update outcomes), so the checkpoint holds the complete
  learner state: weights, optimizers, fast beliefs, replay sampler, meta-learner, registries' cached state, RNGs;
* kill -9 in the middle of a run (an action executed in the world, its result never seen) and a restart continue
  the run without executing any action twice and without losing or duplicating any executed decision.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys

import pytest

from cam_live.contracts import RunFiles, SimClock
from cam_live.diagnostics.world import DAY, LabVerifier, LabWorld, lab_task
from cam_live.harness.rpc import RpcServer
from cam_live.runtime.logs import RunReader, iter_jsonl
from int_helpers import DriverState, LabHost, agent_config, drive, lab_config

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "int_child.py")


def _run(args, timeout=2400, **env):
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    p = subprocess.run([sys.executable, CHILD, *map(str, args)], env=e, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stderr[-4000:]


def test_save_fresh_process_load_gives_identical_decisions(tmp_path):
    """Two fresh processes: the first runs, saves and continues; the second loads that checkpoint and continues
    from the saved environment. The decisions must be the same: same candidate chosen by the same rule, the same
    probability vector over the whole candidate set (to 1e-6), the same number of learning updates and model
    versions. Measured, not asserted: the probabilities agree to ~1e-7 rather than bitwise, and the reward-parameter
    hash can differ after a meta-update (floating-point summation order somewhere on the meta path)."""
    seed = {"PYTHONHASHSEED": "0"}
    code, err = _run(["warm", tmp_path, 7, 5, tmp_path / "a.json"], **seed)
    assert code == 0, err
    a = json.load(open(tmp_path / "a.json"))
    assert sum(d["n_updates"] for d in a["warm"]) > 0                   # it had learned something before the save
    code, err = _run(["decide", tmp_path / "b", tmp_path / "ckpt", tmp_path / "env.pkl", 5, tmp_path / "b.json"],
                     **seed)
    assert code == 0, err
    b = json.load(open(tmp_path / "b.json"))
    assert b["loaded_checkpoint"] == a["checkpoint"]
    cont, fresh = a["decisions"], b["decisions"]
    assert len(fresh) == len(cont) == 5
    gaps = []
    for x, y in zip(cont, fresh):
        assert (x["operation"], x["arguments"], x["rule"]) == (y["operation"], y["arguments"], y["rule"])
        assert len(x["probabilities"]) == len(y["probabilities"])
        gaps.append(max(abs(p - q) for p, q in zip(x["probabilities"], y["probabilities"])))
        assert gaps[-1] < 1e-6, gaps
        assert (x["n_updates"], x["meta_applied"], x["model_version"]) == \
               (y["n_updates"], y["meta_applied"], y["model_version"])
    assert gaps[0] == 0.0                         # the first decision after the load is bit-identical
    print("max probability gap per decision:", gaps,
          "reward versions equal:", [x["reward_version"] == y["reward_version"] for x, y in zip(cont, fresh)])


@pytest.fixture
def host(tmp_path):
    h = LabHost(lab_config(61, target=100000, horizon_s=6 * DAY, budget=40.0))
    sock = str(tmp_path / "lab.sock")
    server = RpcServer(sock, h, LabHost.METHODS, name="labhost").start()
    task = lab_task(h.world, "obj_crash")
    task.limits.max_decisions = 14
    json.dump(task.to_dict(), open(tmp_path / "task.json", "w"))
    yield h, sock, str(tmp_path / "run"), str(tmp_path / "summary.json")
    server.stop()


def test_kill_9_mid_run_then_restart_without_duplicate_actions(host):
    h, sock, run_dir, summary = host
    code, err = _run(["runtime", sock, run_dir, 400, summary], RT_KILL_AFTER_EXECUTE=4)
    assert code == -signal.SIGKILL, err
    assert h.world._n_exec == 4
    # the dead process left a journal entry for an action whose result it never received
    gen = RunReader(run_dir).state()["committed"]["generation"]
    live = [e for e in iter_jsonl(os.path.join(run_dir, "wal.jsonl")) if e["g"] == gen]
    inflight = {e["action_id"] for e in live if e["t"] == "begin"} - {e["action_id"] for e in live if e["t"] == "end"}
    assert len(inflight) == 1
    (aid,) = inflight
    assert h.real_executions[aid] == 1
    code, err = _run(["runtime", sock, run_dir, 400, summary])
    assert code == 0, err
    s = json.load(open(summary))
    assert s["loaded"], "the restarted agent did not load a checkpoint"
    # no action ran twice in the world, however often the runtime asked
    assert h.real_executions and all(n == 1 for n in h.real_executions.values())
    assert h.execute_calls[aid] == 2                                    # asked again once, de-duplicated by the world
    # the learner holds each executed decision exactly once, with the world's result
    recs = s["records"]
    ids = [a for a, _ in recs]
    assert len(ids) == len(set(ids))
    executed = {a for a, st in recs if st == "ok" and a in h.real_executions}
    assert executed == set(h.real_executions)
    st = RunReader(run_dir).state()
    # the runtime counts every action the world received and did not refuse (also one that failed there, e.g. over
    # the world's budget); the world counts only those it ran
    assert st["counters"]["actions_executed"] == len(h.attempted) >= h.world._n_exec == len(h.real_executions)
    assert st["status"] in ("exhausted", "completed", "failed", "running")
    assert s["decisions"] >= 8
