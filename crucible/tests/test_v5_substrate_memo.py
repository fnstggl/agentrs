"""Same spec and seed, same event hashes - across processes - although sandboxed work is billed by parent-measured
CPU (workstream A: audit K3 after D-20).

D-20 moved sandbox metering out of the measured process: the parent reads the worker's CPU clock. Real CPU time
jitters run to run, and the deterministic cost model's memo (bill the FIRST measurement of each computation of a
world lineage for every later identical computation) lived in one process only, so after the merge a new process
measured everything afresh: two processes running the same population world for half a day recorded different
`cpu_ms`/`cu` in 157 of 1,795 events, and a cross-process resume could not continue with the uninterrupted run's
hashes. `crucible.sandbox.metering.DiskMemo` shares the anchors between the processes of a machine (first writer
wins); the tolerance rule (a measurement above 3x the anchor + 1 ms is billed fresh) is unchanged, and lineages
never share anchors.
"""
import json
import os
import subprocess
import sys

from crucible.sandbox import metering as M
from crucible.sandbox.runner import capabilities

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LOOP = "def f(n, c):\n    s = 0\n    for i in range(n):\n        s += i\n    return s\n"

CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from crucible.sandbox.runner import SandboxConfig, run_program
memo = sys.argv[2] if sys.argv[2] != "-" else None
r = run_program({"m.py": %r}, "m:f", 60000, cfg=SandboxConfig(count_lines=True, meter_scope=sys.argv[3],
                                                                memo_path=memo))
print(json.dumps({"ok": r["ok"], "cpu_ms": r["cpu_ms"], "work_lines": r["work_lines"], "memo": r["meter"]["memo"],
                  "cpu_ns": r["meter"]["cpu_ns"]}))
""" % LOOP


def _child(memo, lineage) -> dict:
    p = subprocess.run([sys.executable, "-c", CHILD, SRC, memo or "-", lineage], capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_disk_memo_keeps_the_first_measurement_and_the_tolerance_rule(tmp_path):
    path = str(tmp_path / "memo.sqlite")
    a, b, c = M.Memo(), M.Memo(), M.Memo()          # three processes' in-memory memos, one machine-level file
    assert a.bill("k", 1000, disk=path) == (1000, "first")
    assert b.bill("k", 1200, disk=path) == (1000, "remembered")      # another process bills the first anchor
    assert b.bill("k", 900, disk=path) == (1000, "remembered")
    assert c.bill("k", 10 ** 9, disk=path) == (10 ** 9, "exceeded")  # far more work is still billed fresh
    assert M.Memo().bill("k", 1100, disk=path) == (1000, "remembered")
    assert M.Memo().bill("k", 1100) == (1100, "first")               # without the file: per process, as before
    # an unusable memo file never fails a job: the process falls back to its own memo
    broken = str(tmp_path / "a_directory")
    os.makedirs(broken)
    d = M.Memo()
    assert d.bill("k", 7, disk=broken) == (7, "first") and d.disk_errors == 1
    assert d.bill("k", 8, disk=broken) == (7, "remembered")


def test_new_processes_bill_the_anchor_another_process_measured(tmp_path):
    caps = capabilities()
    assert caps["landlock_abi"] >= 1 and caps["seccomp"], f"this test needs the sandbox: {caps}"
    memo = str(tmp_path / "memo.sqlite")
    runs = [_child(memo, "lineage-1") for _ in range(3)]
    other = _child(memo, "lineage-2")
    assert all(r["ok"] for r in runs) and other["ok"]
    assert runs[0]["memo"] == "first" and [r["memo"] for r in runs[1:]] == ["remembered", "remembered"]
    assert len({(r["cpu_ms"], r["work_lines"]) for r in runs}) == 1 and runs[0]["work_lines"] > 0
    assert other["memo"] == "first"                   # another lineage never reuses this lineage's anchor
    # ...while what the parent measured did differ from process to process: the noise the memo removes
    fresh = [_child(None, "lineage-3") for _ in range(3)]
    assert all(r["memo"] == "first" for r in fresh)
    assert len({r["cpu_ns"] for r in runs + fresh}) > 1


def test_the_same_world_in_two_new_processes_gives_the_same_event_hashes(tmp_path):
    """A population world with sandboxed web sites (every page request a sandbox job billed by measured CPU), run in
    two NEW processes: identical last hash and digest with the machine-level memo; with per-process memos only
    (`CRUCIBLE_METER_MEMO=off`) the recorded CPU differs."""
    code = (
        "import json, sys\nsys.path.insert(0, sys.argv[1])\nimport crucible\ncrucible.load_all_modules()\n"
        "from crucible.core.kernel import World\nfrom crucible.population.worlds import population_world\n"
        "sp = population_world('development', 13, n=200, sites=5, adaptive=False, strategic=False)\n"
        "w = World.create(sp, world_id='xproc')\nw.advance_time(43200)\n"
        "print(json.dumps({'seq': w.seq, 'hash': w.last_hash, 'digest': w.digest(),\n"
        "                  'jobs': sum(1 for e in w.log if e.type == 'cloud.usage')}))\n")

    def run(memo):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CRUCIBLE_METER_MEMO=memo)
        p = subprocess.run([sys.executable, "-c", code, SRC], capture_output=True, text=True, timeout=900, env=env)
        assert p.returncode == 0, p.stderr[-3000:]
        return json.loads(p.stdout.strip().splitlines()[-1])
    memo = str(tmp_path / "memo.sqlite")
    a, b = run(memo), run(memo)
    assert a["jobs"] > 10, "the world must run sandboxed requests for this test to mean anything"
    assert (a["seq"], a["hash"], a["digest"]) == (b["seq"], b["hash"], b["digest"])
    c, d = run("off"), run("off")
    assert c["hash"] != d["hash"]


def test_a_noisy_program_load_is_billed_by_its_anchor_like_a_job(monkeypatch):
    """K2/K1 after the merge (wave 3b): every page request of a population world loads its site program in a fresh
    worker first; the load CPU above LOAD_FREE_NS is billed with the request. On a loaded machine the same load
    measured 2.9 ms in one run and 5.8 ms in the next, so the same request cost 0.1 ms (load free) or 2.9 ms, the
    job memo's tolerance check billed the noisy figure ("exceeded") and two runs of one spec and seed differed in
    `cloud.usage`. The load is now anchored like a job (first measurement per lineage, program and target; same
    tolerance rule) and kept out of the job's own anchor. Load noise is injected here by inflating every CPU-clock
    reading from the end of the load on (job CPU is a difference of two later readings, so only the load grows)."""
    from crucible.sandbox import runner
    caps = capabilities()
    assert caps["landlock_abi"] >= 1 and caps["seccomp"], f"this test needs the sandbox: {caps}"
    files = {"m.py": "s = 0\nfor i in range(2000):\n    s += i\n\ndef f(n, c):\n    return n + s\n"}
    real_cpu, real_settle, extra = M.cpu_ns, M.settle, {"ns": 0, "on": False}

    def noisy(pid):                   # every reading from the end of the load on is offset by the noise
        v = real_cpu(pid)
        return v + extra["ns"] if (v is not None and extra["on"]) else v

    def settle(pid, max_s=0.005):     # first called when the worker reports the program loaded
        extra["on"] = True
        return real_settle(pid, max_s)
    monkeypatch.setattr(M, "cpu_ns", noisy)
    monkeypatch.setattr(M, "settle", settle)
    cfg = runner.SandboxConfig(count_lines=True, meter_scope=f"load-anchor-{os.getpid()}", memo_path=None)

    def run(noise_ns):
        extra.update(ns=noise_ns, on=False)
        r = runner.run_program(files, "m:f", 7, cfg=cfg)
        assert r["ok"] and r["result"] == 7 + sum(range(2000)), r
        return r
    first = run(0)
    assert first["meter"]["load_ns"] < M.LOAD_FREE_NS          # a small program: its load is free
    noisy_runs = [run(int(M.LOAD_FREE_NS * 0.9)) for _ in range(3)]   # +2.7 ms of load noise each time
    assert all(r["meter"]["load_ns"] > M.LOAD_FREE_NS for r in noisy_runs)   # measured above the free allowance
    # ...yet billed exactly like the first run: the anchor, not the noise
    assert {(r["cpu_ms"], r["work_lines"]) for r in noisy_runs} == {(first["cpu_ms"], first["work_lines"])}
    assert all(r["meter"]["memo"] == "remembered" for r in noisy_runs)
    # import-time work that really grows (far beyond the tolerance) is still billed fresh
    grown = run(60_000_000)
    assert grown["cpu_ms"] >= first["cpu_ms"] + 50.0


def test_a_platform_cold_start_bills_a_request_as_the_warm_run_did(monkeypatch):
    """K1 after the merge (wave 3b, D-80): a world resumed in a NEW process (or forked onto a fresh pool) starts
    every sandboxed site cold where the uninterrupted run had warm workers; billing that load (anchored, 2.9-5.8 ms
    measured, above LOAD_FREE_NS) gave the resumed run other `cloud.usage` events than the uninterrupted one. A
    platform cold start's load now joins the job's FIRST measurement (its anchor) and never its tolerance check, so a
    request costs what its first measurement cost whether this host process started cold or warm there; a restart
    after the worker died (or was evicted) still bills its load on top. Load noise is injected as in the test above
    (only while the program loads)."""
    import sys as _sys
    from crucible.sandbox import runner
    caps = capabilities()
    assert caps["landlock_abi"] >= 1 and caps["seccomp"], f"this test needs the sandbox: {caps}"
    files = {"m.py": "s = 0\nfor i in range(2000):\n    s += i\n\ndef f(n, c):\n    return n + s\n"}
    real_cpu, real_settle, extra = M.cpu_ns, M.settle, {"on": False}
    noise = int(M.LOAD_FREE_NS * 1.5)                    # every load measures ~4.5 ms more: billable above 3 ms

    def noisy(pid):
        v = real_cpu(pid)
        return v + noise if (v is not None and extra["on"]) else v

    def settle(pid, max_s=0.005):
        if "jid" not in _sys._getframe(1).f_locals:     # the settle that ends a program load (no job yet)
            extra["on"] = True
        return real_settle(pid, max_s)
    real_run = runner.Worker.run

    def run(self, *a, **kw):
        extra["on"] = False
        return real_run(self, *a, **kw)
    monkeypatch.setattr(M, "cpu_ns", noisy)
    monkeypatch.setattr(M, "settle", settle)
    monkeypatch.setattr(runner.Worker, "run", run)
    cfg = runner.SandboxConfig(count_lines=True, meter_scope=f"cold-start-{os.getpid()}", memo_path=None)
    X = {"kind": "call", "target": "m:f", "args": 1, "caller": None}
    Y = {"kind": "call", "target": "m:f", "args": 2, "caller": None}
    first = runner.WorkerPool(4, cfg)                     # the uninterrupted run's host process
    try:
        w = first.get("site", lambda: files)
        assert w.cold == "process"
        x1 = w.run(dict(X))                               # cold: its load joins X's anchor
        y1 = w.run(dict(Y))                               # warm: Y's anchor is the job alone
        assert x1["ok"] and y1["ok"] and x1["meter"]["load_ns"] > M.LOAD_FREE_NS and y1["meter"]["load_ns"] == 0
        assert x1["cpu_ms"] > y1["cpu_ms"] + 1.0          # the first cold measurement carries the load
    finally:
        first.close()
    resumed = runner.WorkerPool(4, cfg)                   # a NEW host process resumes the world before Y
    try:
        w2 = resumed.get("site", lambda: files)
        y2 = w2.run(dict(Y))                              # cold here, warm in the uninterrupted run
        x2 = w2.run(dict(X))                              # warm here, cold in the uninterrupted run
        assert y2["meter"]["load_ns"] > M.LOAD_FREE_NS and x2["meter"]["load_ns"] == 0
        assert (y2["cpu_ms"], y2["work_lines"]) == (y1["cpu_ms"], y1["work_lines"])
        assert (x2["cpu_ms"], x2["work_lines"]) == (x1["cpu_ms"], x1["work_lines"])
        # a restart after the worker died is the program's or the pool's doing: its load is billed on top
        w2._kill()
        w3 = resumed.get("site", lambda: files)
        assert w3 is not w2 and w3.cold == "restart"
        y3 = w3.run(dict(Y))
        assert y3["ok"] and y3["cpu_ms"] >= y1["cpu_ms"] + 1.0
    finally:
        resumed.close()


def test_the_first_request_after_a_platform_cold_start_bills_its_anchor_despite_warm_up():
    """K1 repair (wave-3b verifier: the cross-process resume test failed on the loaded machine, 3 of 6 diagnosis runs). The
    first job after its program's load in a fresh interpreter does warm-up work the uninterrupted world's worker had
    done long before (lazily built state, interpreter caches): on a population world resumed in a new process the
    first `service:home` request of a site measured 5.0 ms against an anchor below 1.3 ms, the tolerance rule billed
    it fresh ("exceeded") and the resumed run left the uninterrupted run's hashes. That warm-up is a platform cold
    start's, not the program's: the job bills its anchor. A restart after the worker died keeps the tolerance check,
    so a program cannot make its own workers restart to hide work."""
    from crucible.sandbox import runner
    caps = capabilities()
    assert caps["landlock_abi"] >= 1 and caps["seccomp"], f"this test needs the sandbox: {caps}"
    # the first call in a process builds a table (~20 ms of CPU), later calls only read it
    files = {"m.py": "_t = None\n\ndef f(n, c):\n    global _t\n    if _t is None:\n"
                     "        _t = [i * i for i in range(400000)]\n    return n + _t[n]\n"}
    cfg = runner.SandboxConfig(count_lines=True, meter_scope=f"warm-up-{os.getpid()}", memo_path=None)
    X = {"kind": "call", "target": "m:f", "args": 1, "caller": None}
    Y = {"kind": "call", "target": "m:f", "args": 2, "caller": None}
    first = runner.WorkerPool(4, cfg)                     # the uninterrupted run's host process
    try:
        w = first.get("site", lambda: files)
        x1 = w.run(dict(X))                               # cold: X's anchor carries the warm-up
        y1 = w.run(dict(Y))                               # warm: Y's anchor is the read alone
        assert x1["ok"] and y1["ok"] and x1["meter"]["memo"] == y1["meter"]["memo"] == "first"
        assert x1["meter"]["cpu_ns"] > M.MEMO_TOL * y1["meter"]["cpu_ns"] + M.MEMO_SLACK_NS   # the warm-up shows
    finally:
        first.close()
    resumed = runner.WorkerPool(4, cfg)                   # a NEW host process resumes the world before Y
    try:
        w2 = resumed.get("site", lambda: files)
        assert w2.cold == "process"
        y2 = w2.run(dict(Y))                              # cold here (warm-up runs), warm in the uninterrupted run
        assert y2["meter"]["cpu_ns"] > M.MEMO_TOL * y1["meter"]["cpu_ns"] + M.MEMO_SLACK_NS
        assert y2["meter"]["memo"] == "remembered" and (y2["cpu_ms"], y2["work_lines"]) == (y1["cpu_ms"],
                                                                                           y1["work_lines"])
        x2 = w2.run(dict(X))                              # warm here: its anchor (with the warm-up) as before
        assert (x2["cpu_ms"], x2["work_lines"]) == (x1["cpu_ms"], x1["work_lines"])
        # a restart after the worker died keeps the tolerance check: the warm-up above Y's anchor is billed
        w2._kill()
        w3 = resumed.get("site", lambda: files)
        assert w3.cold == "restart"
        y3 = w3.run(dict(Y))
        assert y3["ok"] and y3["meter"]["memo"] == "exceeded" and y3["cpu_ms"] > y1["cpu_ms"]
    finally:
        resumed.close()
