"""Scale layer (crucible.scale): process-parallel worlds, asynchronous rollout workers, the 4.0 benchmark suite.

These tests start real worker processes (at most 4 at a time) and run real `tiny` worlds. The helper functions
and classes at the top of this file are referenced from job specs as ``"test_scale:<name>"`` and are therefore
imported and executed *inside* worker processes (fault injection, policies whose behaviour depends on the
parameter blob).
"""
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time

import pytest

from crucible.scale import RolloutFarm, WorldPool, pool as pool_mod, run_world_job, world_job
from crucible.scale import bench
from crucible.scale.pool import DAY, Supervisor, _children, ensure_registered, normalise_job
from crucible.scale.rollout import WorldEnv

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
# a small real world: one cheap need, the deterministic cost model (so digests are reproducible across processes)
OV = {"needs": ["route.shortest"], "n_consumers": 150, "params": {"cost_model": "deterministic"}}
AGENTS = [("ann", "crucible.agents.scripted:RandomPolicy", {}), ("bob", "crucible.scale.bench:SweepPolicy", {})]


def spec(seed, **more):
    return {"family": "tiny", "seed": seed, "overrides": {**OV, **more}}


# ---------------------------------------------------------------------------- code that runs inside workers
def facts_summary(world, scale=1):
    import numpy  # noqa: F401 - loaded long before this point; the limits below were in place before that
    return {"principals": world.state.count("principals") * scale, "now": world.now, "pid": os.getpid(),
            "blas": sorted({os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
            "threads": len(os.listdir("/proc/self/task"))}


def boom_summary(world):
    raise RuntimeError("summary exploded on purpose")


def exit_summary(world, code=13):
    sys.stderr.write("last words before dying\n")
    sys.stderr.flush()
    os._exit(code)


def kill_summary(world):
    os.kill(os.getpid(), signal.SIGKILL)


def sleep_summary(world, seconds=120):
    time.sleep(seconds)
    return {}


def unpicklable_summary(world):
    return lambda: None


def traj_summary(world, trajectories, policies, label=""):
    return {"steps": {a: len(t.steps) for a, t in sorted(trajectories.items())}, "policies": sorted(policies),
            "label": label, "tools": sorted({s.action["tool"] for t in trajectories.values() for s in t.steps})[:3]}


def spawn_child_summary(world, pidfile, then="hang"):
    """Leave a child process in its own session (as the sandbox does), then hang or die without cleaning up."""
    p = subprocess.Popen(["sleep", "600"], start_new_session=True)
    with open(pidfile + ".tmp", "w") as f:
        f.write(str(p.pid))
    os.replace(pidfile + ".tmp", pidfile)
    if then == "exit":
        time.sleep(3.0)          # long enough for the supervisor to have seen the child
        os._exit(9)
    time.sleep(600)


class TagPolicy:
    """A policy whose behaviour is a function of its parameter blob: proof that a worker really loaded it."""
    name, kind = "tag", "scripted"

    def __init__(self, params: bytes):
        self.tool = {b"A": "world.clock", b"B": "world.whoami"}.get(params, "bank.accounts")
        self.n = 0

    def reset(self, actor, rng):
        self.actor, self.rng = actor, rng

    def act(self, obs, tools):
        self.n += 1
        return None if self.n % 4 == 0 else {"tool": self.tool, "args": {}}

    def observe_result(self, action, result):
        assert result["tool"] == action["tool"]


def make_tag_policy(params: bytes):
    if params == b"BAD":
        raise ValueError("this parameter blob cannot be loaded")
    return TagPolicy(params)


class FaultyEnv(WorldEnv):
    def __init__(self, ref, **kw):
        super().__init__(ref, **kw)
        self._fault = (ref.get("exit_after"), ref.get("raise_after"))
        self._n = 0

    def step(self, action):
        self._n += 1
        if self._fault[0] is not None and self._n > self._fault[0]:
            os._exit(17)
        if self._fault[1] is not None and self._n > self._fault[1]:
            raise RuntimeError("env exploded on purpose")
        return super().step(action)


def make_faulty_env(ref, **kw):
    return FaultyEnv(ref, **kw)


# ---------------------------------------------------------------------------- helpers
def dead(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return True
    return data[data.rindex(b")") + 2:data.rindex(b")") + 3] == b"Z"


def wait_dead(pids, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not all(dead(p) for p in pids):
        time.sleep(0.1)
    return [p for p in pids if not dead(p)]


def my_children() -> set:
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()        # the stdlib's own helper process: started once, lives with us
    return {p for p in _children(os.getpid()) if not dead(p)}


@pytest.fixture(scope="module")
def reference():
    """Single-process results for seeds 0..3 (and one world with agents): what every pool must reproduce."""
    out = {s: run_world_job(world_job(spec(s), until_day=1)) for s in range(4)}
    out["agents"] = run_world_job(world_job(spec(7), until_day=1, agents=AGENTS))
    assert all(r["ok"] for r in out.values()), [r["error"] for r in out.values() if not r["ok"]]
    return out


# ============================================================================ job specs
def test_job_specs_are_validated_before_anything_runs():
    j = world_job({"family": "tiny", "seed": 3}, days=2, agents=[("a", "m:f", {"x": 1})])
    assert j["world"] == {"family": "tiny", "seed": 3} and j["agents"] == [("a", "m:f", {"x": 1}, {})]
    assert pickle.loads(pickle.dumps(j)) == j                                  # declarative: survives pickling
    assert normalise_job({"family": "tiny", "until_day": 1})["world"]["seed"] == 0   # flat form, default seed
    for bad, msg in [({"family": "tiny"}, "exactly one horizon"),
                     ({"family": "tiny", "days": 1, "until_day": 2}, "exactly one horizon"),
                     ({"days": 1}, "needs a world"),
                     ({"family": "tiny", "days": 1, "hroizon": 3}, "unknown job keys"),
                     ({"world": {"family": "tiny", "sed": 1}, "days": 1}, "unknown world keys"),
                     ({"family": "tiny", "days": 1, "agents": [("a", "not-a-factory")]}, "an agent is"),
                     ({"family": "tiny", "days": 1, "runner": "threads"}, "runner must be"),
                     ({"family": "tiny", "days": 1, "resume": "/x.blob"}, "either a world or 'resume'"),
                     ({"family": "tiny", "days": 1, "summarise": "nomodule"}, "summarise must be")]:
        with pytest.raises(ValueError, match=msg):
            normalise_job(bad)
    with pytest.raises(ValueError, match="exceed the cap"):
        Supervisor(pool_mod.MAX_WORKERS + 1, "crucible.scale.pool:_WorldJobHandler")
    with pytest.raises(ValueError, match="start_method"):
        Supervisor(1, "crucible.scale.pool:_WorldJobHandler", start_method="teleport")
    # a failing job in this process is a result, not an exception
    r = run_world_job({"family": "no_such_family", "days": 1})
    assert not r["ok"] and r["kind"] == "exception" and "unknown world family" in r["error"]["message"]
    assert r["error"]["stage"] == "setup" and "Traceback" in r["error"]["traceback"]


# ============================================================================ determinism
def test_same_spec_same_digest_for_any_worker_count(reference):
    jobs = [world_job(spec(s), until_day=1, tag=s, summarise="test_scale:facts_summary") for s in range(4)]
    jobs.append(world_job(spec(7), until_day=1, agents=AGENTS, tag="agents"))
    with WorldPool(1) as one:
        r1 = one.map(jobs)
        s1 = one.stats()
    with WorldPool(4) as four:
        # the same jobs in another order, next to unrelated worlds that keep the other workers busy
        noise = [four.submit(world_job(spec(100 + i, n_consumers=300), until_day=1)) for i in range(3)]
        ids = [four.submit(j) for j in reversed(jobs)]
        got = {r["job_id"]: r for r in four.as_completed()}
        r4 = [got[i] for i in reversed(ids)]
        assert all(got[i]["ok"] for i in noise)
    assert all(r["ok"] for r in r1 + r4), [r["error"] for r in r1 + r4 if not r["ok"]]
    for a, b in zip(r1, r4):
        ref = reference[a["tag"]]
        assert a["digest"] == b["digest"] == ref["digest"]
        assert a["last_hash"] == b["last_hash"] == ref["last_hash"]
        assert a["counts"] == b["counts"] == ref["counts"] and a["seq"] == ref["seq"]
    assert len({r["digest"] for r in r1}) == len(r1)                 # different seeds are different worlds
    assert len({r["worker_pid"] for r in r1}) == 1 and [r["worker_job_index"] for r in r1] == [1, 2, 3, 4, 5]
    assert len({r["worker_pid"] for r in r4}) >= 2                   # really spread over processes
    assert s1["spawned"] == 1 and s1["completed"] == 5 and s1["failed"] == 0
    # the record is complete and plausible
    r = r1[0]
    assert r["family"] == "tiny" and r["seed"] == 0 and r["sim_time_s"] == DAY and r["sim_days"] == 1.0
    c = r["counts"]
    assert c["events"] == r["seq"] > 500 and c["process_runs"] > 50 and c["process_failures"] == 0
    assert c["commands"] == 0 and c["agent_steps"] == 0
    assert 0 < r["run_cpu_s"] <= r["cpu_s"] + r["cpu_children_s"] and 0 < r["run_wall_s"] <= r["wall_s"]
    assert 20 < r["peak_rss_mb"] < 4000 and r["peak_rss_scope"] in ("job", "process")
    assert r["rss_start_mb"] <= r["peak_rss_mb"] and r["snapshot"] is None and r["event_log"] is None
    assert r["summary"]["principals"] > 5 and r["summary"]["now"] == DAY and r["summary"]["pid"] == r["worker_pid"]
    assert r["timeline"] == [{"t": DAY, "seq": r["seq"], "wall_s": r["timeline"][0]["wall_s"], "store_bytes": None}]
    ag = r1[4]["counts"]
    assert ag["agent_steps"] == 48 and ag["rounds"] == 24 and ag["commands"] == 48 and ag["rejected"] > 0


def test_fork_start_method_gives_the_same_world(reference):
    # "spawn" is the default; "fork" must not change what a world computes, and a forked worker must be unable to
    # stop its siblings through the copy of the supervisor it inherited
    with WorldPool(2, start_method="fork") as pool:
        res = pool.map([world_job(spec(s), until_day=1) for s in (0, 1, 2)])
        assert pool.stats()["start_method"] == "fork"
    assert [r["digest"] for r in res] == [reference[s]["digest"] for s in (0, 1, 2)]
    assert [r["last_hash"] for r in res] == [reference[s]["last_hash"] for s in (0, 1, 2)]


# ============================================================================ isolation
def test_failures_stay_inside_their_job(reference):
    with WorldPool(2, job_timeout_s=300) as pool:
        ids = {
            "good_a": pool.submit(world_job(spec(0), until_day=1)),
            "no_family": pool.submit(world_job({"family": "no_such_family", "seed": 1}, days=1)),
            "bad_agent": pool.submit(world_job(spec(1), days=1, agents=[("zed", "crucible.agents.scripted:NoSuchPolicy", {})])),
            "boom": pool.submit(world_job(spec(1), until_day=1, summarise="test_scale:boom_summary")),
            "unpicklable": pool.submit(world_job(spec(2), days=0.1, summarise="test_scale:unpicklable_summary")),
            "exit": pool.submit(world_job(spec(2), days=0.1, summarise="test_scale:exit_summary")),
            "sigkill": pool.submit(world_job(spec(3), days=0.1, summarise="test_scale:kill_summary")),
            "hang": pool.submit(world_job(spec(3), days=0.1, summarise="test_scale:sleep_summary", timeout_s=6)),
            "good_b": pool.submit(world_job(spec(2), until_day=1)),
        }
        got = {r["job_id"]: r for r in pool.as_completed()}
        res = {k: got[i] for k, i in ids.items()}
        assert len(got) == len(ids)                                   # every job reported exactly once
        # the pool is still usable after two dead workers and a killed one
        again = pool.run(world_job(spec(3), until_day=1))
        st = pool.stats()
    for k, s in (("good_a", 0), ("good_b", 2)):
        assert res[k]["ok"] and res[k]["digest"] == reference[s]["digest"]
    assert again["ok"] and again["digest"] == reference[3]["digest"]
    r = res["no_family"]
    assert not r["ok"] and r["kind"] == "exception" and r["error"]["type"] == "KeyError" and r["error"]["stage"] == "setup"
    r = res["bad_agent"]
    assert not r["ok"] and r["error"]["type"] == "AttributeError" and "NoSuchPolicy" in r["error"]["message"]
    r = res["boom"]
    assert not r["ok"] and r["error"]["stage"] == "summarise" and "exploded on purpose" in r["error"]["message"]
    assert "boom_summary" in r["error"]["traceback"]
    assert r["digest"] == reference[1]["digest"]                      # the world itself ran; only the summary failed
    r = res["unpicklable"]
    assert not r["ok"] and r["error"]["stage"] == "summarise" and r["summary"] is None
    r = res["exit"]
    assert not r["ok"] and r["kind"] == "worker_died" and r["evidence"]["exitcode"] == 13
    assert "last words before dying" in r["evidence"]["stderr_tail"]   # the evidence
    assert f"pid {r['evidence']['pid']}" in r["error"]["message"] and r["digest"] is None
    r = res["sigkill"]
    assert not r["ok"] and r["kind"] == "worker_died" and r["evidence"]["signal"] == "SIGKILL"
    r = res["hang"]
    assert not r["ok"] and r["kind"] == "timeout" and r["evidence"]["ran_s"] >= 6
    assert st["died"] == 3 and st["spawned"] == 5 and st["failed"] == 7 and st["completed"] == 3
    assert st["pending"] == 0 and st["in_flight"] == 0


# ============================================================================ checkpoints
def test_checkpoint_and_resume_in_another_process(tmp_path):
    from crucible.core.kernel import World
    from crucible.storage.sqlite import SQLiteStore
    ensure_registered()                       # this process replays the recorded worlds below
    store = str(tmp_path / "store")
    with WorldPool(2, store_dir=store, max_jobs_per_worker=1) as pool:      # a fresh process for every job
        a, b, c, e1, e_full = pool.map([
            world_job(spec(0), until_day=6, name="straight"),
            world_job(spec(0), until_day=6, snapshot_every_days=3, name="with-checkpoints"),
            world_job(spec(0), until_day=3, name="first-half"),
            world_job(spec(5), until_day=1, agents=AGENTS, name="agents-first-day"),
            world_job(spec(5), until_day=2, agents=AGENTS, name="agents-straight"),
        ])
        assert all(r["ok"] for r in (a, b, c, e1, e_full)), [r["error"] for r in (a, b, c, e1, e_full)]
        d, e, e2, clash = pool.map([
            world_job(resume=c["snapshot"], until_day=6, name="second-half"),
            world_job(resume=b["checkpoints"][0]["path"], until_day=6, name="from-checkpoint"),
            world_job(resume=e1["snapshot"], until_day=2, agents=AGENTS, name="agents-second-day"),
            world_job(spec(0), until_day=1, name="straight"),            # same store name: must not overwrite
        ])
        st = pool.stats()
    assert d["ok"] and e["ok"] and e2["ok"], [r["error"] for r in (d, e, e2)]
    # run-to-day-6 == run-to-day-3, snapshot, resume in another process, run-to-day-6 (state and event hash chain)
    assert c["sim_time_s"] == 3 * DAY and c["digest"] != a["digest"]
    for r in (b, d, e):
        assert r["digest"] == a["digest"] and r["last_hash"] == a["last_hash"] and r["seq"] == a["seq"]
        assert r["sim_time_s"] == 6 * DAY
    assert len({r["worker_pid"] for r in (a, b, c, d, e, e1, e2, e_full)}) == 8 and st["recycled"] >= 8
    assert d["resumed_from"]["seq"] == c["seq"] and d["counts"]["events"] == a["seq"] - c["seq"]
    assert d["world_id"] != c["world_id"] == a["world_id"]
    # the same with agents: the checkpoint carries the runner's and the policies' state
    assert e2["digest"] == e_full["digest"] and e2["last_hash"] == e_full["last_hash"]
    assert e2["agents_restored"] == ["p:bob"] and e2["counts"]["agent_steps"] == 48
    assert e1["counts"]["commands"] + e2["counts"]["commands"] == e_full["counts"]["commands"] == 96
    # checkpoints: every 3 simulated days, not at the horizon (that is the final snapshot)
    assert [k["day"] for k in b["checkpoints"]] == [3.0] and b["checkpoints"][0]["seq"] == c["seq"]
    assert b["checkpoints"][0]["hash"] == c["last_hash"] and a["checkpoints"] == []
    for r in (a, b, c, d):
        assert os.path.getsize(r["snapshot"]) == r["snapshot_bytes"] > 1000 and r["snapshot"].endswith("final.blob")
        assert os.path.dirname(r["event_log"]) == os.path.join(store, r["name"]) and r["store_bytes"] > 10_000
    assert os.path.exists(e1["snapshot"] + ".agents") and not os.path.exists(a["snapshot"] + ".agents")
    # storage grows monotonically, one timeline row per simulated day
    tl = a["timeline"]
    assert [x["t"] for x in tl] == [k * DAY for k in range(1, 7)]
    assert all(x["seq"] < y["seq"] and x["store_bytes"] <= y["store_bytes"] for x, y in zip(tl, tl[1:]))
    assert tl[-1]["store_bytes"] > tl[0]["store_bytes"] and tl[-1]["seq"] == a["seq"]
    # the event log on disk replays to the same world; the resumed one loads from its base snapshot + its events
    s = SQLiteStore(a["event_log"])
    events = s.read_events(a["world_id"])
    spec_a = World.from_blob(open(a["snapshot"], "rb").read()).spec
    assert len(events) == a["seq"] and World.replay(spec_a, events).digest() == a["digest"]
    s.close()
    s = SQLiteStore(d["event_log"])
    assert s.world_info(d["world_id"])["parent"] == {"world": c["world_id"], "seq": c["seq"], "hash": c["last_hash"]}
    assert s.last_seq(d["world_id"]) == a["seq"] and World.load(s, d["world_id"]).digest() == a["digest"]
    s.close()
    # the snapshot file restores the exact final state
    assert World.from_blob(open(d["snapshot"], "rb").read()).digest() == a["digest"]
    # recorded events are never overwritten
    assert not clash["ok"] and clash["error"]["type"] == "FileExistsError" and clash["error"]["stage"] == "store"
    assert os.path.isdir(os.path.join(store, "logs"))


def test_async_runner_jobs_resume_exactly_and_summarisers_get_trajectories(tmp_path):
    # the same job code without a pool (run_world_job is what workers call), with agents on independent clocks
    store = str(tmp_path)
    kw = dict(agents=AGENTS, runner="async", think_mean_s=1800)
    full = run_world_job(world_job(spec(3), until_day=2, name="full", **kw), store, job_id=1)
    half = run_world_job(world_job(spec(3), until_day=1, name="half", **kw), store, job_id=2)
    rest = run_world_job(world_job(resume=half["snapshot"], until_day=2, name="rest", **kw), store, job_id=3)
    assert full["ok"] and half["ok"] and rest["ok"], [r["error"] for r in (full, half, rest)]
    assert rest["digest"] == full["digest"] and rest["last_hash"] == full["last_hash"] and rest["seq"] == full["seq"]
    assert half["counts"]["agent_steps"] + rest["counts"]["agent_steps"] == full["counts"]["agent_steps"] > 40
    assert full["counts"]["rounds"] == 0 and full["counts"]["commands"] == full["counts"]["agent_steps"]
    assert [x["t"] for x in full["timeline"]] == [DAY, 2 * DAY] and rest["job_id"] == 3
    # a job without a store keeps nothing on disk; recorded trajectories reach the summariser
    rec = run_world_job(world_job(spec(3), until_day=0.5, agents=AGENTS, record=True, dt_s=3600,
                                  summarise="test_scale:traj_summary", summarise_kwargs={"label": "x"}))
    assert rec["ok"] and rec["event_log"] is None and rec["snapshot"] is None
    assert rec["summary"]["steps"] == {"p:ann": 12, "p:bob": 12} and rec["summary"]["policies"] == ["p:ann", "p:bob"]
    assert rec["summary"]["label"] == "x" and len(rec["summary"]["tools"]) == 3
    bad = run_world_job(world_job(spec(3), until_day=1, snapshot_every_days=1))
    assert not bad["ok"] and "needs a store directory" in bad["error"]["message"]
    behind = run_world_job(world_job(resume=half["snapshot"], until_day=0.5, name="behind"), store)
    assert not behind["ok"] and "before the world's time" in behind["error"]["message"]


# ============================================================================ recycling
def test_workers_are_recycled_after_a_fixed_number_of_jobs(monkeypatch):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(k, "7")                                    # whatever the parent's environment says ...
    with WorldPool(1, max_jobs_per_worker=2) as pool:
        assert os.environ["OMP_NUM_THREADS"] == "7"                   # ... and without changing it for the parent
        with pytest.raises(Exception):                                # a job must be declarative
            pool.submit(world_job(spec(0), days=0.1, summarise_kwargs={"f": lambda: 1}, summarise="test_scale:facts_summary"))
        res = pool.map([world_job(spec(s), days=0.1, summarise="test_scale:facts_summary",
                                  summarise_kwargs={"scale": 2}) for s in range(5)])
        st = pool.stats()
    assert all(r["ok"] for r in res)
    pids = [r["worker_pid"] for r in res]
    assert pids[0] == pids[1] != pids[2] == pids[3] != pids[4] and len(set(pids)) == 3
    assert [r["worker_job_index"] for r in res] == [1, 2, 1, 2, 1]
    assert st["recycled"] == 2 and st["spawned"] == 3 and st["died"] == 0 and st["submitted"] == 5
    assert all(r["summary"]["principals"] % 2 == 0 for r in res)      # summarise_kwargs reached the summariser
    # ... every worker, replacements included, runs BLAS/OpenMP single-threaded and is one thread
    assert all(r["summary"]["blas"] == ["1"] and r["summary"]["threads"] == 1 for r in res)
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert not wait_dead(pids, 10)                                    # recycled and closed workers are gone


# ============================================================================ shutdown
def test_keyboard_interrupt_leaves_no_process_behind(tmp_path):
    before = my_children()
    pidfile = str(tmp_path / "grandchild.pid")
    pids = []
    with pytest.raises(KeyboardInterrupt):
        with WorldPool(3) as pool:
            pool.wait_ready()
            pool.submit(world_job(spec(0), days=0.05, summarise="test_scale:spawn_child_summary",
                                  summarise_kwargs={"pidfile": pidfile}))
            for s in range(1, 4):
                pool.submit(world_job(spec(s), days=2000))             # far longer than the test
            while not os.path.exists(pidfile):
                assert pool.poll(0.2) == []
            pids = pool.stats()["worker_pids"]
            assert len(pids) == 3 and not any(dead(p) for p in pids)
            assert my_children() - before == set(pids)
            threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()
            pool.join()                                               # Ctrl-C arrives while we wait here
    t = time.monotonic()
    grandchild = int(open(pidfile).read())
    assert wait_dead(pids + [grandchild], 15) == []                   # workers and what they left behind
    assert time.monotonic() - t < 10
    assert my_children() <= before                                    # no child of this process is left alive
    left = pool.results()
    assert len(left) == 4 and {r["kind"] for r in left} == {"terminated"} and not any(r["ok"] for r in left)
    assert pool.stats()["worker_pids"] == [] and pool.stats()["orphans_killed"] >= 1
    with pytest.raises(RuntimeError):
        pool.submit(world_job(spec(0), days=1))
    pool.terminate()                                                  # idempotent


def test_dead_worker_does_not_leave_its_children_behind(tmp_path):
    before = my_children()
    pidfile = str(tmp_path / "grandchild.pid")
    with WorldPool(1) as pool:
        r = pool.run(world_job(spec(0), days=0.05, summarise="test_scale:spawn_child_summary",
                               summarise_kwargs={"pidfile": pidfile, "then": "exit"}))
        grandchild = int(open(pidfile).read())
        assert not r["ok"] and r["kind"] == "worker_died" and r["evidence"]["exitcode"] == 9
        assert r["evidence"]["orphans_killed"] == [grandchild]
        assert wait_dead([grandchild], 10) == []
        assert pool.run(world_job(spec(1), days=0.05))["ok"]            # replaced
    assert my_children() <= before


def test_killed_parent_leaves_no_orphaned_workers(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(f"""
import sys
sys.path.insert(0, {SRC!r})
from crucible.scale import WorldPool, world_job
if __name__ == "__main__":
    pool = WorldPool(2)
    pool.wait_ready()
    for s in range(2):
        pool.submit(world_job({{"family": "tiny", "seed": s, "overrides": {OV!r}}}, days=2000))
    pool.poll(1.0)
    print("PIDS", *pool.stats()["worker_pids"], flush=True)
    while True:
        pool.poll(1.0)
""")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "TMPDIR": str(tmp)})
    try:
        line = p.stdout.readline().split()
        assert line[0] == "PIDS" and len(line) == 3, line
        pids = [int(x) for x in line[1:]]
        assert not any(dead(x) for x in pids)
        p.kill()                                                      # SIGKILL: no chance to clean up
        p.wait(10)
        assert os.listdir(tmp) != []                                  # the pool's temporary worker logs
        assert wait_dead(pids, 20) == []                              # the workers notice and exit by themselves
        end = time.monotonic() + 10
        while os.listdir(tmp) and time.monotonic() < end:
            time.sleep(0.1)
        assert os.listdir(tmp) == []                                  # ... and take their temporary files along
    finally:
        if p.poll() is None:
            p.kill()
        p.stdout.close()


# ============================================================================ rollout farm
def h(blob: bytes) -> str:
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def test_rollout_versions_propagate_and_accounting_is_exact():
    before = my_children()
    think = 600
    with RolloutFarm(2, make_policy="test_scale:make_tag_policy", params=b"A", world_kwargs={"think_s": think},
                     queue_size=3) as farm:
        assert farm.version == 0
        first = [farm.submit(dict(spec(k), key=f"w{k}"), steps=5, tag=("first", k)) for k in range(2)]
        got = {r.task_id: r for r in farm}
        assert sorted(got) == sorted(first) and farm.outstanding == 0
        with pytest.raises(LookupError):
            farm.get(timeout=0.1)
        for r in got.values():
            assert r.ok and r.param_version == 0 and r.params_hash == h(b"A") and r.staleness == 0 and r.fresh_world
            assert [a["tool"] if a else None for a in r.actions] == ["world.clock"] * 3 + [None, "world.clock"]
        # a long task is running when new parameters are published: it finishes with the old ones, and says so
        slow = farm.submit(dict(spec(0), key="w0"), steps=60, tag="slow")
        while farm.stats()["dispatched"] < 3:
            time.sleep(0.02)
        assert farm.publish(b"B") == 1 and farm.version == 1
        after = [farm.submit(dict(spec(k), key=f"w{k}"), steps=5, tag=("after", k)) for k in range(2)]
        rest = {r.task_id: r for r in farm}
        assert sorted(rest) == sorted([slow] + after)
        r = rest[slow]
        assert r.ok and r.param_version == 0 and r.latest_version == 1 and r.staleness == 1 and not r.fresh_world
        assert {a["tool"] for a in r.actions if a} == {"world.clock"} and r.steps == 60
        for t in after:
            r = rest[t]
            assert r.ok and r.param_version == 1 and r.params_hash == h(b"B") and r.staleness == 0
            assert {a["tool"] for a in r.actions if a} == {"world.whoami"}          # the new blob really acts
            assert not r.fresh_world                                               # same world, continued
        # persistent worlds: each key stays on one worker and simulated time is continuous across tasks
        for k in range(2):
            seq = sorted((r for r in list(got.values()) + list(rest.values()) if r.world_key == f"w{k}"),
                         key=lambda r: r.sim_t[0])
            assert len({r.worker_pid for r in seq}) == 1
            for a, b in zip(seq, seq[1:]):
                assert b.sim_t[0] == a.sim_t[-1] + a.sim_dt[-1]
        # variable-duration transitions: every step records the simulated time it consumed
        everything = list(got.values()) + list(rest.values())
        for r in everything:
            assert len(r.obs) == len(r.actions) == len(r.rewards) == len(r.results) == len(r.sim_dt) == r.steps
            for a, res, dt in zip(r.actions, r.results, r.sim_dt):
                assert dt == (think if a is None else think + max(1, res["cost_s"]))
                assert (res is None) == (a is None) and (a is None or res["ok"])
            assert r.sim_seconds == sum(r.sim_dt) and {think, think + 1} == set(r.sim_dt)
            assert r.final_obs["time"] == r.sim_t[-1] + r.sim_dt[-1] and r.obs[0]["principal"] == "p:learner"
            assert r.wall_s > 0 and r.env_s > 0 and len(list(r.transitions())) == r.steps
        # an episode: a fresh world, run to its horizon, closed afterwards
        farm.submit(dict(spec(9), env={"horizon_days": 0.05}), tag="episode")
        ep = farm.get(timeout=120)
        assert ep.ok and ep.done and ep.fresh_world and ep.sim_seconds == int(0.05 * DAY) and ep.param_version == 1
        st = farm.stats()
        total = sum(r.steps for r in everything) + ep.steps
        assert st["submitted"] == st["ok"] == st["delivered"] == 6 and st["failed"] == 0
        assert st["steps"] == st["steps_executed"] == total == 5 * 4 + 60 + ep.steps
        assert st["by_version"] == {0: 3, 1: 3} and st["staleness_max"] == 1 and st["queued"] == st["in_flight"] == 0
        assert st["sim_seconds"] == sum(r.sim_seconds for r in everything) + ep.sim_seconds
        # parameters a worker cannot load fail that task only; the next good version is used again
        assert farm.publish(b"BAD") == 2
        farm.submit(dict(spec(0), key="w0"), steps=2)
        bad = farm.get(timeout=120)
        assert not bad.ok and bad.kind == "exception" and "cannot be loaded" in bad.error["message"]
        assert bad.param_version == 2 and bad.steps == 0
        assert farm.publish(b"A") == 3
        farm.submit(dict(spec(0), key="w0"), steps=2)
        good = farm.get(timeout=120)
        assert good.ok and good.param_version == 3 and good.params_hash == h(b"A") and good.fresh_world
        assert farm.stats()["failed"] == 1 and farm.stats()["died"] == 0
    assert farm.abandoned == [] and my_children() <= before
    with pytest.raises(RuntimeError):
        farm.submit(spec(0), steps=1)


def test_rollout_accounting_survives_a_worker_crash_and_a_world_exception():
    with RolloutFarm(2, make_world="test_scale:make_faulty_env", world_kwargs={"think_s": 600},
                     policy_kwargs={"cls": "crucible.agents.scripted:RandomPolicy"}) as farm:
        tasks = {}
        for k in range(6):
            ref = dict(spec(k), key=f"w{k}")
            if k == 2:
                ref["exit_after"] = 3                     # the worker process dies inside the 4th step
            if k == 4:
                ref["raise_after"] = 2                    # the world raises inside the 3rd step
            tasks[farm.submit(ref, steps=6, tag=k)] = k
        got = [r for r in farm]
        assert sorted(r.task_id for r in got) == sorted(tasks)           # each task exactly once
        by = {tasks[r.task_id]: r for r in got}
        crashed, raised = by[2], by[4]
        assert not crashed.ok and crashed.kind == "worker_died" and crashed.evidence["exitcode"] == 17
        assert crashed.steps == 0 and crashed.actions == [] and "exit code 17" in crashed.error["message"]
        assert not raised.ok and raised.kind == "exception" and "exploded on purpose" in raised.error["message"]
        ok = [r for k, r in by.items() if k not in (2, 4)]
        assert all(r.ok and r.steps == 6 for r in ok)
        st = farm.stats()
        assert st["submitted"] == 6 and st["ok"] == 4 and st["failed"] == 2 and st["delivered"] == 6
        assert st["steps"] == sum(r.steps for r in got) == 24
        assert st["died"] == 1 and st["spawned"] == 3
        # the farm keeps working; the lost worlds are rebuilt from their references and say so
        more = {farm.submit(dict(spec(k), key=f"w{k}"), steps=2, tag=k): k for k in (0, 2, 4)}
        again = {more[r.task_id]: r for r in farm}
        assert all(r.ok and r.steps == 2 for r in again.values())
        assert again[2].fresh_world and again[4].fresh_world
        st = farm.stats()
        assert st["submitted"] == 9 == st["ok"] + st["failed"] and st["steps"] == 30
        workers_alive = [w.pid for w in farm._sup.workers]
    assert wait_dead(workers_alive, 15) == []


def test_rollout_backpressure_blocks_workers_instead_of_growing():
    n_tasks, qsize, n_workers = 14, 2, 2
    with RolloutFarm(n_workers, world_kwargs={"think_s": 600}, queue_size=qsize,
                     policy_kwargs={"cls": "crucible.agents.scripted:IdlePolicy"}, policy_scope="worker") as farm:
        farm.wait_ready()
        for i in range(n_tasks):
            farm.submit(dict(spec(i % 2), key=f"w{i % 2}"), steps=1)
        last, stable_since, give_up = -1, time.monotonic(), time.monotonic() + 180
        while True:                                                      # nobody consumes: wait until it stalls
            st = farm.stats()
            if st["dispatched"] != last:
                last, stable_since = st["dispatched"], time.monotonic()
            if st["buffered"] == qsize and time.monotonic() - stable_since > 3.0:
                break
            assert time.monotonic() < give_up, st
            time.sleep(0.1)
        st = farm.stats()
        # at most: the queue, one result in the farm's hand, one task per worker
        assert st["dispatched"] <= qsize + 1 + n_workers < n_tasks
        assert st["buffered"] == qsize and st["delivered"] == 0 and st["queued"] >= n_tasks - (qsize + 1 + n_workers)
        got = [farm.get(timeout=120) for _ in range(n_tasks)]            # consuming releases the workers
        assert all(r.ok and r.steps == 1 and r.actions == [None] and r.sim_dt == [600] for r in got)
        assert len({r.task_id for r in got}) == n_tasks
        st = farm.stats()
        assert st["ok"] == st["delivered"] == n_tasks and st["steps"] == n_tasks


def test_rollout_throughput_grows_with_workers(capsys):
    # CPU-bound: every step advances a full `tiny` world (two needs, 800 consumers) by a simulated hour
    r = bench.bench_rollout((1, 4), tasks_per_worker=3, steps=8, think_s=3600, spec="tiny")
    one, four = r["rows"]
    assert one["workers"] == 1 and four["workers"] == 4
    assert one["status"] == four["status"] == "ok" and one["accounting_exact"] and four["accounting_exact"]
    assert one["steps"] == 24 and four["steps"] == 96 and one["failed"] == four["failed"] == 0
    assert one["steps_per_wall_s"]["median"] > 0 and four["sim_days_per_wall_s"]["median"] > 0
    speedup = four["speedup_vs_1"]
    with capsys.disabled():
        print(f"\n[rollout throughput] 1 worker {one['steps_per_wall_s']['median']} steps/s, 4 workers "
              f"{four['steps_per_wall_s']['median']} steps/s, speedup {speedup} (load {four['loadavg']}, "
              f"throttled periods {four['cgroup_throttled_fraction']})")
    assert speedup > 1.3                      # modest on purpose: the machine is shared and CPU-throttled


# ============================================================================ benchmarks
def test_bench_kernel_records_are_plausible():
    r = bench.bench_kernel(n_actors=20, n_cmds=300, reps=2)
    rows = {x["measure"].split(" (")[0]: x for x in r["rows"]}
    assert set(rows) == {"bank.transfer", "mixed tool calls", "bank.accounts", "replay from genesis"}
    for x in r["rows"]:
        assert x["per_wall_s"]["n"] == x["reps"] == 2 and len(x["per_wall_s"]["samples"]) == 2
        assert 0 < x["per_wall_s"]["min"] <= x["per_wall_s"]["median"] <= x["per_wall_s"]["max"] < 1e7
        assert x["per_cpu_s"]["median"] >= 0.5 * x["per_wall_s"]["median"]      # CPU time cannot exceed wall by much
        assert x["ok_fraction"] == 1.0 and x["workers"] == 1 and len(x["loadavg"]) == 3
    assert 2.5 <= rows["bank.transfer"]["events_per_command"] <= 5
    mix = rows["mixed tool calls"]["tool_mix"]
    assert len(mix) == 9 and all(v["ok"] == v["calls"] > 0 for v in mix.values())
    assert rows["replay from genesis"]["invariants_ok"] is True
    assert rows["bank.transfer"]["world"]["modules"] == ["identity", "bank", "admin"]      # the 3.0 benchmark world
    assert len(rows["mixed tool calls"]["world"]["modules"]) == 7


def test_bench_families_storage_and_episode_cost():
    tiny = {"family": "tiny", "overrides": OV}
    r = bench.bench_families([tiny, "no_such_family"], idle_days=1, agent_days=1, reps=2, workers=2)
    ok, bad = r["rows"]
    assert bad["family"] == "no_such_family" and bad["status"] == "error" and "build failed" in bad["error"]
    assert bad["idle"] is None and bad["agents"] is None
    assert ok["status"] == "ok" and ok["group"] == "legacy" and ok["split"] == "dev" and ok["workers"] == 2
    assert ok["world"]["consumers"] == 150 and ok["world"]["cost_model"] == "deterministic" and ok["world"]["principals"] > 5
    i, a = ok["idle"], ok["agents"]
    assert i["reps"] == 2 and i["sim_days_per_wall_s"]["n"] == 2 and i["same_digest_across_reps"] is True
    assert 0 < i["sim_days_per_wall_s"]["median"] < 1000 and i["sim_days_per_cpu_s"]["median"] >= 0.5 * i["sim_days_per_wall_s"]["median"]
    assert i["events_per_sim_day"]["median"] > 500 and i["process_failures"] == 0 and i["peak_rss_mb"]["median"] > 20
    assert a["mix"] == ["RandomPolicy", "RandomPolicy", "SweepPolicy", "Builder"] and a["agent_steps"] == 96
    assert a["commands"] > 48 and a["cpu_s"] > 0 and a["sim_days_per_wall_s"]["median"] > 0
    assert a["events_per_sim_day"]["median"] > i["events_per_sim_day"]["median"]     # agents add events
    # cost per episode, with and without a price
    free, priced = bench.episode_cost_rows(r["rows"]), bench.episode_cost_rows(r["rows"], 0.05)
    assert free[0]["usd"] is None and free[0]["cpu_s"] == pytest.approx(a["cpu_s"] + a["cpu_children_s"], abs=1e-3)
    assert priced[0]["usd"] == pytest.approx(priced[0]["cpu_s"] / 3600 * 0.05, abs=1e-6) and priced[0]["usd"] > 0
    assert free[0]["episode_days"] == 1 and free[0]["standard_30_day_episode"] is False
    assert free[0]["peak_rss_mb"] == a["peak_rss_mb"] and free[1]["status"] == "error" and free[1]["cpu_s"] is None
    # storage: measured after every simulated day
    s = bench.bench_storage([tiny], days=3, workers=1)["rows"][0]
    assert s["status"] == "ok", s["error"]
    cum = s["cumulative"]
    assert [c["day"] for c in cum] == [1.0, 2.0, 3.0]
    assert all(x["events"] < y["events"] and x["store_bytes"] <= y["store_bytes"] for x, y in zip(cum, cum[1:]))
    assert cum[-1]["store_bytes"] > cum[0]["store_bytes"] > 10_000 and s["events_total"] == cum[-1]["events"]
    assert s["events_per_sim_day"]["n"] == 2 and s["events_per_sim_day"]["median"] > 300
    assert 50 < s["bytes_per_event"] < 5000 and s["store_bytes_per_sim_day"]["median"] > 10_000
    assert s["snapshot_bytes"] > 1000 and s["snapshot_ms"]["n"] == 3 and s["snapshot_ms"]["median"] > 0
    assert s["fork_ms"]["median"] > 0 and s["fork_digest_equal"] is True
    assert s["replay_digest_equal"] is True and s["replay_events"] == s["events_total"] and s["replay_events_per_s"] > 100


def test_bench_memory_grows_with_population_and_scaling_curve():
    m = bench.bench_memory([{"family": "tiny", "overrides": {**OV, "n_consumers": 200}},
                            {"family": "tiny", "overrides": {**OV, "n_consumers": 200_000}}], days=0.05, workers=2)
    small, big = m["rows"]
    assert small["status"] == big["status"] == "ok"
    assert small["population"] == 206 and big["population"] == 200_006
    assert big["state_array_mb"] > 50 * small["state_array_mb"] and big["snapshot_mb"] > small["snapshot_mb"]
    assert big["world_rss_mb"] > small["world_rss_mb"] + 5 and big["peak_rss_mb"] > big["rss_before_world_mb"]
    sc = bench.bench_scaling((1, 2), spec={"family": "tiny", "overrides": OV}, days=0.5, reps=1)
    one, two = sc["rows"]
    assert (one["parallel_worlds"], two["parallel_worlds"]) == (1, 2) and one["workers"] == two["workers"] == 2
    assert one["failed_jobs"] == two["failed_jobs"] == 0 and one["speedup_vs_1"] == 1.0
    assert one["aggregate_sim_days_per_wall_s"]["median"] > 0 and two["aggregate_sim_days_per_wall_s"]["median"] > 0
    assert two["efficiency"] == pytest.approx(two["speedup_vs_1"] / 2, abs=1e-3) and sc["params"]["pool_start_s"] > 0


def test_suite_document_schema_is_stable(tmp_path):
    assert bench.SCHEMA == "crucible-bench-openworld/1"
    assert bench.SECTIONS == ("kernel", "families", "storage", "memory", "scaling", "rollout", "browser", "gpu",
                              "episode_cost", "llm")
    assert bench.SECTION_KEYS == ("status", "reason", "generated", "mode", "wall_s", "conditions", "params", "rows")
    assert bench.TOP_KEYS == ("schema", "generated", "mode", "argv", "machine", "caveat", "discovery", "run", "sections")
    out = str(tmp_path / "openworld.json")
    doc = bench.run_suite(mode="quick", only=["kernel", "gpu", "llm"], out=out, log=None, argv=["--test"],
                          section_kwargs={"kernel": {"n_actors": 20, "n_cmds": 200, "reps": 2}})
    assert bench.validate(doc) == [] and json.load(open(out)) == json.loads(json.dumps(doc, default=str))
    assert tuple(doc) == bench.TOP_KEYS and tuple(doc["sections"]) == bench.SECTIONS
    assert all(tuple(k for k in s if k != "traceback") == bench.SECTION_KEYS for s in doc["sections"].values())
    k = doc["sections"]["kernel"]
    assert k["status"] == "ok" and len(k["rows"]) == 4 and k["mode"] == "quick" and k["wall_s"] > 0
    assert set(k["rows"][0]) == {"measure", "unit", "per_wall_s", "per_cpu_s", "ok_fraction", "commands_per_rep",
                                 "reps", "workers", "world", "loadavg", "events_per_command"}
    assert set(k["rows"][0]["per_wall_s"]) == {"n", "median", "min", "max", "mean", "stdev", "samples"}
    c = k["conditions"]                                                # every section carries its conditions
    assert set(c) == {"loadavg_start", "loadavg_end", "cpu_count", "cgroup_cpu_quota_cores",
                      "cgroup_cores_used_all_tenants", "cgroup_throttled_fraction", "bench_cpu_s", "bench_cores_used",
                      "wall_s"}
    assert len(c["loadavg_start"]) == 3 and c["cpu_count"] == os.cpu_count() and c["bench_cpu_s"] > 0
    assert doc["sections"]["llm"]["status"] == "not_measured"
    assert doc["sections"]["llm"]["reason"] == "not measured in this run"
    assert doc["sections"]["gpu"]["status"] in ("ok", "skipped")
    if doc["sections"]["gpu"]["status"] == "ok":
        g = doc["sections"]["gpu"]["rows"][0]
        assert "other tenants" in g["label"] and 0 <= g["util_pct"]["median"] <= 100 and g["mem_total_mb"] > 0
    for name in ("families", "storage", "memory", "scaling", "rollout", "browser", "episode_cost"):
        assert doc["sections"][name]["status"] == "not_run" and doc["sections"][name]["rows"] == []
    assert doc["mode"] == "quick (partial)" and doc["run"]["sections_run"] == ["kernel", "gpu", "llm"]
    assert doc["machine"]["cpu_count"] == os.cpu_count() and "pessimistic" in doc["caveat"]
    d = doc["discovery"]
    assert "tiny" in d["families"] and "tiny" in d["groups"]["legacy"] and "open_digital" in d["groups"]["open"]
    assert set(d["selected"]) >= {"tiny", "bazaar", "open_digital", "open_industry"}
    assert all(n.startswith(("mech_", "ow_")) or n in bench.QUICK_BASE_FAMILIES for n in d["selected"])
    assert all(n in d["selected"] or n in d["not_run"] for n in d["families"])
    table = bench.format_table(doc)
    assert "kernel transactions per second" in table and "bank.transfer" in table and "not measured in this run" in table
    # a later partial run is merged: sections it did not run are kept, with their own time stamp
    llm = tmp_path / "llm.json"
    llm.write_text(json.dumps({"rows": [{"model": "example", "tokens_per_s": 123.0}]}))
    doc2 = bench.run_suite(mode="quick", only=["llm"], out=out, llm_file=str(llm), log=None, argv=[])
    assert doc2["sections"]["kernel"] == json.loads(json.dumps(k, default=str))
    assert doc2["sections"]["llm"]["status"] == "ok" and doc2["sections"]["llm"]["rows"] == [
        {"model": "example", "tokens_per_s": 123.0}]
    assert bench.validate(json.load(open(out))) == []
    # schema violations are detected, and the recorded 3.0 results cannot be overwritten
    broken = json.loads(json.dumps(doc2))
    del broken["sections"]["kernel"]["conditions"]["loadavg_start"]
    broken["sections"].pop("llm")
    assert len(bench.validate(broken)) == 2
    for name in ("bench.json", "internet.json"):
        with pytest.raises(ValueError, match="never overwritten"):
            bench.run_suite(mode="quick", only=["llm"], out=str(tmp_path / name), log=None)
    with pytest.raises(ValueError, match="unknown sections"):
        bench.run_suite(mode="quick", only=["kernle"], log=None)


def test_family_selection_covers_new_families():
    disc = {"families": ["bazaar", "mech_a", "mech_b", "mech_c", "novel_needs", "open_digital", "ow_x", "tiny"]}
    full, why = bench.select_families(disc, "full")
    assert full == disc["families"] and why == {}
    quick, why = bench.select_families(disc, "quick", max_per_group=2)
    assert quick == ["bazaar", "mech_a", "mech_b", "open_digital", "ow_x", "tiny"]
    assert set(why) == {"mech_c", "novel_needs"} and "--max-per-group" in why["mech_c"]
    many = {"families": ["mech_abm_x", "mech_abm_x_tools", "mech_abm_y", "mech_scm_z", "mech_tech_w", "tiny"]}
    spread, why = bench.select_families(many, "quick", max_per_group=3)      # one per engine before a second of any
    assert spread == ["mech_abm_x", "mech_scm_z", "mech_tech_w", "tiny"] and set(why) == {"mech_abm_x_tools", "mech_abm_y"}
    assert bench.select_families(many, "quick", max_per_group=4)[0][:3] == ["mech_abm_x", "mech_abm_x_tools", "mech_scm_z"]
    assert bench.select_families(many, "quick")[0] == many["families"]       # no cap: every new family
    only, why = bench.select_families(disc, "quick", wanted=["mech_c"])
    assert only == ["mech_c"] and "mech_a" in why
    assert [bench.family_group(n) for n in ("mech_scm_x", "ow_trade", "open_mixed", "engineering_lab", "bazaar")] == \
        ["mech", "ow", "open", "lab", "legacy"]
    assert bench.agent_mix(["identity", "bank", "admin", "rig"])[-1][1].endswith("SweepPolicy")
    assert bench.stat([3.0, 1.0, 2.0]) == {"n": 3, "median": 2.0, "min": 1.0, "max": 3.0, "mean": 2.0, "stdev": 1.0,
                                           "samples": [3.0, 1.0, 2.0]}
    assert bench.stat([])["n"] == 0 and bench.stat([5])["stdev"] == 0.0
