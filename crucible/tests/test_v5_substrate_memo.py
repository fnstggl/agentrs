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
