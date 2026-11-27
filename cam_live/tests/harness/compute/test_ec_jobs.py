"""Background jobs: purpose, limits, deadline, status, cancellation, persisted records, concurrency cap, events,
and interrupted jobs reported after a restart."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from cam_live import contracts as c
from cam_live.harness import jobs as J
from cam_live.harness import sandbox as sbx


@pytest.fixture
def manager(sandbox, tmp_path):
    made = []

    def make(run_dir=None, sb=None, **kw):
        sink = []
        m = J.JobManager(str(run_dir or tmp_path / "run"), sandbox=sb or sandbox(), event_sink=sink.append, **kw)
        m.sink = sink
        made.append(m)
        return m
    yield make
    for m in made:
        m.close()


def wait_state(m, job_id, states, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rec = m.status(job_id)
        if rec.state in states:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stayed {m.status(job_id).state}")


def test_job_needs_purpose_and_deadline(manager):
    m = manager()
    with pytest.raises(ValueError):
        m.submit("", "shell", "true", deadline_s=5)
    with pytest.raises(ValueError):
        m.submit("   ", "shell", "true", deadline_s=5)
    with pytest.raises(ValueError):
        m.submit("count lines", "shell", "true", deadline_s=0)
    with pytest.raises(ValueError):
        m.submit("count lines", "perl", "true", deadline_s=5)
    with pytest.raises(ValueError):
        m.submit("fetch", "shell", "true", deadline_s=5, network="scoped")      # no broker, no network
    assert m.list() == []


def test_job_runs_persists_its_record_and_is_delivered_as_an_event(manager):
    m = manager()
    rec = m.submit("sum the first hundred integers", "python", "print(sum(range(101)))\nopen('sum.txt','w').write('x')",
                   deadline_s=20)
    assert rec.state in (J.JobState.RUNNING, J.JobState.QUEUED) and rec.purpose == "sum the first hundred integers"
    done = m.wait(rec.job_id, 20)
    assert done.state == J.JobState.SUCCEEDED and done.exit_code == 0 and done.stdout_tail == "5050\n"
    assert done.started >= done.created and done.ended >= done.started and done.wall_s > 0 and done.leftover_processes == 0
    assert done.limits["wall_s"] == 20 and done.limits["mem_mb"] > 0
    on_disk = json.load(open(os.path.join(m.dir, rec.job_id, "record.json")))
    assert on_disk["state"] == "succeeded" and on_disk["purpose"] == rec.purpose and on_disk["exit_code"] == 0
    assert open(os.path.join(m.dir, rec.job_id, "stdout.log")).read() == "5050\n"
    assert m.result(rec.job_id)["stdout_tail"] == "5050\n"
    assert m.sandbox.read_file("sum.txt")["data"] == b"x"                         # jobs share the workspace
    assert len(m.sink) == 1
    ev = m.sink[0]
    assert ev.kind == "event" and ev.data["event"] == "job_finished" and ev.data["job"]["job_id"] == rec.job_id
    assert ev.provenance.trust == c.Trust.UNTRUSTED and "5050" in ev.text and "succeeded" in ev.text
    events, cursor = m.events()
    assert [e.obs_id for e in events] == [ev.obs_id] and m.events(cursor) == ([], cursor)
    assert "reward" not in json.dumps(on_disk) and "reward" not in ev.to_json()


def test_failed_job_reports_exit_code_and_error_output(manager):
    m = manager()
    rec = m.submit("fail on purpose", "shell", "echo before; echo problem >&2; exit 4", deadline_s=20)
    done = m.wait(rec.job_id, 20)
    assert done.state == J.JobState.FAILED and done.exit_code == 4 and done.error == "exit code 4"
    assert done.stdout_tail == "before\n" and done.stderr_tail == "problem\n"


def test_deadline_is_enforced_and_nothing_is_left_running(manager, procs_matching):
    m = manager()
    marker = f"jobdeadline-{uuid.uuid4().hex}"
    t0 = time.monotonic()
    rec = m.submit("sleep past the deadline", "shell", f"(exec -a '{marker}' sleep 300) &\nexec -a '{marker}' sleep 300", deadline_s=1.0)
    done = m.wait(rec.job_id, 20)
    assert done.state == J.JobState.TIMEOUT and done.killed_for == "timeout" and "deadline" in done.error
    assert time.monotonic() - t0 < 8 and done.leftover_processes == 0 and procs_matching(marker) == []
    assert m.sink[-1].data["job"]["state"] == "timeout"


def test_resource_limits_apply_to_jobs(manager):
    m = manager()
    lim = sbx.SandboxLimits(mem_mb=128, cpu_s=1, fsize_mb=1)
    mem = m.submit("allocate too much", "python", "b = b'x' * (1 << 30)", deadline_s=20, limits=lim)
    cpu = m.submit("spin", "python", "while True: pass", deadline_s=20, limits=lim)
    big = m.submit("write too much", "shell", "dd if=/dev/zero of=big.bin bs=1M count=50 2>/dev/null", deadline_s=20,
                   limits=lim)
    r_mem, r_cpu, r_big = (m.wait(j.job_id, 30) for j in (mem, cpu, big))
    assert r_mem.state == J.JobState.FAILED and r_mem.limit_hit == "memory"
    assert r_cpu.state == J.JobState.FAILED and r_cpu.limit_hit == "cpu" and r_cpu.signal == signal.SIGXCPU
    assert r_big.state == J.JobState.FAILED and r_big.limit_hit == "fsize"
    assert r_cpu.limits["cpu_s"] == 1 and r_mem.limits["mem_mb"] == 128


def test_status_is_observable_while_running_and_cancel_stops_the_job(manager, procs_matching):
    m = manager()
    marker = f"jobcancel-{uuid.uuid4().hex}"
    rec = m.submit("print then wait", "python",
                   f"import subprocess, sys, time\nprint('phase one', flush=True)\n"
                   f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', '{marker}'])\n"
                   f"time.sleep(300)", deadline_s=120)
    end = time.monotonic() + 15
    while time.monotonic() < end and "phase one" not in m.status(rec.job_id).stdout_tail:
        time.sleep(0.02)
    live = m.status(rec.job_id)
    assert live.state == J.JobState.RUNNING and live.stdout_tail == "phase one\n" and live.pgid > 0
    # the job prints before it starts its child: on a loaded machine the child may appear a moment later
    while time.monotonic() < end and not procs_matching(marker):
        time.sleep(0.05)
    assert live.wall_s > 0 and not live.done and len(procs_matching(marker)) == 1
    t0 = time.monotonic()
    m.cancel(rec.job_id)
    done = m.wait(rec.job_id, 15)
    assert done.state == J.JobState.CANCELLED and done.cancel_requested and time.monotonic() - t0 < 5
    assert procs_matching(marker) == [] and done.leftover_processes == 0
    assert m.cancel(rec.job_id).state == J.JobState.CANCELLED                     # idempotent
    assert [e.data["job"]["state"] for e in m.sink] == ["cancelled"]
    with pytest.raises(KeyError):
        m.status("job_unknown")


def test_concurrency_cap_and_bounded_queue(manager):
    m = manager(max_concurrent=2, max_queued=3)
    code = "import time\ntime.sleep(0.6)\nprint('ok')"
    recs = [m.submit(f"sleeper {i}", "python", code, deadline_s=30) for i in range(5)]
    states = [m.status(r.job_id).state for r in recs]
    assert states.count(J.JobState.RUNNING) == 2 and states.count(J.JobState.QUEUED) == 3
    with pytest.raises(J.JobLimitError):
        m.submit("one too many", "python", code, deadline_s=30)
    assert len(m.list()) == 5
    m.cancel(recs[4].job_id)                                                      # a queued job can be cancelled
    seen_running = 0
    end = time.monotonic() + 30
    while time.monotonic() < end and not all(m.status(r.job_id).done for r in recs):
        seen_running = max(seen_running, m.counts().get(J.JobState.RUNNING, 0))
        time.sleep(0.02)
    final = [m.wait(r.job_id, 10) for r in recs]                 # returns once the job's event has been delivered
    assert [r.state for r in final] == ["succeeded"] * 4 + ["cancelled"]
    assert seen_running == 2 and m.peak_running == 2
    assert final[4].started == 0 and final[4].exit_code is None                   # it never ran
    assert final[2].started >= min(final[0].ended, final[1].ended) - 0.05         # queued ones waited for a slot
    assert sorted(e.data["job"]["job_id"] for e in m.sink) == sorted(r.job_id for r in recs)


def test_callable_jobs_have_cooperative_deadline_and_cancel(manager):
    m = manager(max_concurrent=3)
    quick = m.submit_callable("add numbers", lambda stop: {"sum": 2 + 3}, deadline_s=5)
    polite = m.submit_callable("wait until told to stop", lambda stop: stop.wait(60) and "stopped", deadline_s=0.3)
    broken = m.submit_callable("raise", lambda stop: 1 / 0, deadline_s=5)
    r1, r2, r3 = (m.wait(j.job_id, 10) for j in (quick, polite, broken))
    assert r1.state == J.JobState.SUCCEEDED and r1.result == {"sum": 5} and r1.limits["enforcement"] == "cooperative"
    assert r2.state == J.JobState.TIMEOUT and r2.result is None                   # a late result is not used
    assert r3.state == J.JobState.FAILED and "ZeroDivisionError" in r3.error
    held = m.submit_callable("cancel me", lambda stop: stop.wait(60), deadline_s=30)
    wait_state(m, held.job_id, (J.JobState.RUNNING,))
    m.cancel(held.job_id)
    assert m.wait(held.job_id, 10).state == J.JobState.CANCELLED
    assert json.load(open(os.path.join(m.dir, quick.job_id, "record.json")))["result"] == {"sum": 5}


def test_records_survive_and_a_new_manager_reads_them(manager, sandbox, tmp_path):
    sb = sandbox()
    m1 = manager(run_dir=tmp_path / "persist", sb=sb)
    rec = m1.submit("leave a record", "shell", "echo kept", deadline_s=20)
    m1.wait(rec.job_id, 20)
    m1.close()
    m2 = manager(run_dir=tmp_path / "persist", sb=sb)
    again = m2.status(rec.job_id)
    assert again.state == J.JobState.SUCCEEDED and again.stdout_tail == "kept\n" and m2.interrupted_on_open == []
    assert [e.data["job"]["job_id"] for e in m2.events()[0]] == [rec.job_id]       # events are on disk too


HARNESS = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from cam_live.harness import jobs as J, sandbox as sbx
sb = sbx.Sandbox(sys.argv[3], sbx.SandboxConfig(base_dir=sys.argv[4], allow_same_uid=os.geteuid() != 0))
m = J.JobManager(sys.argv[2], sandbox=sb, max_concurrent=1)
a = m.submit("long computation", "shell", "echo started; exec -a '%s' sleep 300" % os.environ["CAM_TEST_MARKER"],
             deadline_s=600)
b = m.submit("waiting behind it", "shell", "echo never", deadline_s=600)
while "started" not in m.status(a.job_id).stdout_tail or not m.status(a.job_id).pgid:
    time.sleep(0.02)
print(a.job_id, b.job_id, m.status(a.job_id).pgid, flush=True)
time.sleep(600)
"""


def test_restart_reports_interrupted_jobs(sandbox, sbx_base, tmp_path, procs_matching):
    """Kill the process that owns a running and a queued job. Nothing may keep running, and the next manager on
    the same run directory must report both as interrupted (record + event), not lose them."""
    sb = sandbox()
    run_dir = str(tmp_path / "crash")
    marker = f"jobcrash-{uuid.uuid4().hex}"
    src = os.path.abspath(os.path.join(os.path.dirname(sbx.__file__), "..", ".."))
    p = subprocess.Popen([sys.executable, "-c", HARNESS, src, run_dir, sb.workspace, sbx_base],
                         stdout=subprocess.PIPE, text=True, env=dict(os.environ, CAM_TEST_MARKER=marker))
    try:
        running_id, queued_id, pgid = p.stdout.readline().split()
        assert len(procs_matching(marker)) == 1
        p.send_signal(signal.SIGKILL)                              # the harness dies without any cleanup
        p.wait(10)
        end = time.monotonic() + 10
        while time.monotonic() < end and procs_matching(marker):
            time.sleep(0.05)
        assert procs_matching(marker) == []                        # the job's monitor noticed and killed the group
        on_disk = json.load(open(os.path.join(run_dir, "jobs", running_id, "record.json")))
        assert on_disk["state"] == "running" and on_disk["pgid"] == int(pgid)      # what the dead process left
        sink = []
        m = J.JobManager(run_dir, sandbox=sb, event_sink=sink.append)
        try:
            assert sorted(m.interrupted_on_open) == sorted([running_id, queued_id])
            r, q = m.status(running_id), m.status(queued_id)
            assert r.state == q.state == J.JobState.INTERRUPTED and r.done
            assert "stopped while it was running" in r.error and "stopped while it was queued" in q.error
            assert r.stdout_tail == "started\n" and r.ended > 0
            assert json.load(open(os.path.join(run_dir, "jobs", queued_id, "record.json")))["state"] == "interrupted"
            assert sorted(e.data["job"]["job_id"] for e in sink) == sorted([running_id, queued_id])
            assert all(e.data["job"]["state"] == "interrupted" for e in sink)
            fresh = m.submit("work after the restart", "shell", "echo fine", deadline_s=20)
            assert m.wait(fresh.job_id, 20).state == J.JobState.SUCCEEDED
        finally:
            m.close()
        m2 = J.JobManager(run_dir, sandbox=sb)                     # a second restart does not report them again
        assert m2.interrupted_on_open == [] and len(m2.events()[0]) == 3
        m2.close()
    finally:
        if p.poll() is None:
            p.kill()


def test_closing_the_manager_cancels_and_records_live_jobs(manager, procs_matching):
    m = manager(max_concurrent=1)
    marker = f"jobclose-{uuid.uuid4().hex}"
    a = m.submit("runs until closed", "shell", f"exec -a '{marker}' sleep 300", deadline_s=600)
    b = m.submit("queued until closed", "shell", "echo never", deadline_s=600)
    wait_state(m, a.job_id, (J.JobState.RUNNING,))
    m.close()
    assert m.status(a.job_id).state == J.JobState.CANCELLED and m.status(b.job_id).state == J.JobState.CANCELLED
    assert procs_matching(marker) == []
    with pytest.raises(RuntimeError):
        m.submit("after close", "shell", "true", deadline_s=5)


# ---------------------------------------------------------------------------------------------------------- tools
def test_job_tools_through_the_executor(sandbox, make_ctx, act, policy_cls):
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    policy = policy_cls()
    ctx = make_ctx(policy)
    try:
        bad = ex.execute(act("sandbox.job_start", purpose=" ", kind="shell", code="true", deadline_s=5), ctx)
        assert bad.status == c.ResultStatus.FAILED and bad.failure_class == c.FailureClass.INVALID_ARGUMENTS
        start = ex.execute(act("sandbox.job_start", purpose="build the index", kind="python",
                               code="import time\ntime.sleep(0.3)\nprint('built')", deadline_s=30,
                               limits={"mem_mb": 256}), ctx)
        assert start.status == c.ResultStatus.OK
        job_id = start.artifacts["job"]
        d = start.observations[0].data
        assert d["state"] in ("running", "queued") and d["purpose"] == "build the index" and d["limits"]["mem_mb"] == 256
        status = ex.execute(act("sandbox.job_status", job_id=job_id), ctx)
        assert status.observations[0].data["state"] in ("running", "succeeded")
        mgr = ex.jobs(ctx)
        assert mgr.run_dir == os.path.abspath(ctx.run_dir)
        mgr.wait(job_id, 20)
        result = ex.execute(act("sandbox.job_result", job_id=job_id), ctx)
        rd = result.observations[0].data
        assert rd["state"] == "succeeded" and rd["stdout_tail"] == "built\n" and os.path.isfile(rd["record_path"])
        assert any(o.kind == "event" and o.data["job"]["job_id"] == job_id for o in ctx.sink)   # delivered as an event
        everything = ex.execute(act("sandbox.job_status"), ctx)
        assert everything.observations[0].data["counts"] == {"succeeded": 1}
        long = ex.execute(act("sandbox.job_start", purpose="wait", kind="shell", code="sleep 300", deadline_s=600), ctx)
        cancelled = ex.execute(act("sandbox.job_cancel", job_id=long.artifacts["job"]), ctx)
        assert cancelled.status == c.ResultStatus.OK
        assert mgr.wait(long.artifacts["job"], 15).state == J.JobState.CANCELLED
        missing = ex.execute(act("sandbox.job_result", job_id="job_nope"), ctx)
        assert missing.status == c.ResultStatus.FAILED and missing.failure_class == c.FailureClass.INVALID_ARGUMENTS
        assert [x[0] for x in policy.calls].count("sandbox.job_start") == 2
    finally:
        ex.close()
