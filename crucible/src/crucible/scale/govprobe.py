"""Adversarial probes of the resource governor (crucible.scale.slots, SP-13.7).

Each probe runs a real job on a private ledger that tries to go beyond one granted limit - CPU, RAM, GPU memory, GPU
time, disk, inference concurrency, active worlds - and reports what the job managed to use and whether the governor
stopped or contained it. Further probes check the scheduling rules: FIFO admission, zombies holding nothing, a
detached grandchild not escaping its slot, a self-declared budget not widening the stored one, oversized requests
clamped, nested ledgers inside a job, and that a kill reaches only the job's own processes.

    PYTHONPATH=src python -m crucible.scale.govprobe            # every probe, JSON report

Every probe returns {"dim", "limit", "used", "contained", "record": <usage record of the job>, ...}; `contained`
is the probe's verdict computed from the measured values only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

from . import slots

PY = sys.executable
SAMPLE = 0.25
SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../src


def probe_budget() -> slots.Budget:
    return slots.Budget(cpus=2, gpu_mb=1200, jobs=8, ram_mb=6000, disk_mb=600, inference=8, worlds=16, gpu_jobs=2,
                        default_ram_mb=3000, disk_floor_mb=512)


def _env(**extra) -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = SRC + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def _run(d: str, code: str, **kw) -> dict:
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    kw.setdefault("sample_s", SAMPLE)
    kw.setdefault("poll_s", 0.05)
    return slots.run_job([PY, "-c", code], budget=probe_budget(), env=_env(), **kw)


# ====================================================================== the seven dimensions
BUSY = "import time\nt=time.time()\nwhile time.time()-t<{s}: pass\n"


def cpu(d: str) -> dict:
    """Grant 1 CPU. (a) Three busy processes that first widen their affinity to every CPU: re-pinned, the job's CPU
    rate stays at the grant. (b) A task that keeps widening its affinity: killed."""
    a = ("import os, subprocess, sys\n"
         "procs=[subprocess.Popen([sys.executable,'-c','import os\\nos.sched_setaffinity(0, range(os.cpu_count()))\\n"
         + BUSY.format(s=5).replace("\n", "\\n") + "']) for _ in range(3)]\n"
         "[p.wait() for p in procs]\n")
    ra = _run(d, a, cpu=1, label="probe cpu widen-once")
    ratio = ra["cpu_s"] / max(ra["wall_s"], 1e-6)
    b = ("import os, threading, time\n"
         "def busy():\n    t=time.time()\n    while time.time()-t<8: pass\n"
         "ts=[threading.Thread(target=busy) for _ in range(3)]\n[t.start() for t in ts]\n"
         "t=time.time()\nwhile time.time()-t<8:\n"
         "    for tid in os.listdir('/proc/self/task'):\n"
         "        try: os.sched_setaffinity(int(tid), range(os.cpu_count()))\n"
         "        except OSError: pass\n"
         "    time.sleep(0.05)\n")
    rb = _run(d, b, cpu=1, label="probe cpu keeps widening")
    allowed = 1 * slots.CPU_TOLERANCE + 0.2 + 3 * SAMPLE / max(ra["wall_s"], 1e-6) * 3
    contained = ratio <= allowed and rb["killed_by"] == "cpu" and rb["wall_s"] < 7
    return {"dim": "cpu", "limit": 1, "used": round(ratio, 3), "allowed_ratio": round(allowed, 3),
            "persistent_widening_killed": rb["killed_by"] == "cpu", "persistent_wall_s": rb["wall_s"],
            "contained": bool(contained), "record": ra, "record_b": rb}


def ram(d: str) -> dict:
    """Grant 300 MB; the job allocates 50 MB every 50 ms up to 1.5 GB: killed."""
    code = ("import time\nb=[]\nfor i in range(30):\n    b.append(bytearray(50<<20)); time.sleep(0.05)\n"
            "print('FINISHED')\n")
    r = _run(d, code, cpu=1, ram_mb=300, label="probe ram")
    contained = r["killed_by"] == "ram" and r["rc"] != 0
    return {"dim": "ram", "limit": 300, "used": r["peak_rss_mb"], "contained": bool(contained), "record": r}


def gpu_mem(d: str) -> dict:
    """Grant 300 MB of GPU memory. (a) A 1 GB allocation fails inside the job (allocator cap). (b) A job that lifts
    its own cap and allocates 1 GB is killed."""
    a = ("import torch, sys\n"
         "try:\n    x=torch.empty(1<<30, dtype=torch.uint8, device='cuda'); sys.exit(3)\n"
         "except torch.OutOfMemoryError: pass\n"
         "y=torch.empty(100<<20, dtype=torch.uint8, device='cuda'); torch.cuda.synchronize()\n"
         "import time; time.sleep(1.0)\n")
    ra = _run(d, a, cpu=1, gpu_mb=300, label="probe gpu memory capped")
    b = ("import torch, time\ntorch.cuda.init(); torch.cuda.set_per_process_memory_fraction(1.0)\n"
         "x=torch.empty(1<<30, dtype=torch.uint8, device='cuda'); torch.cuda.synchronize()\ntime.sleep(6)\n"
         "print('FINISHED')\n")
    rb = _run(d, b, cpu=1, gpu_mb=300, label="probe gpu memory cap lifted")
    contained = ra["rc"] == 0 and 0 < ra["peak_gpu_mb"] <= 300 and rb["killed_by"] == "gpu_mem"
    return {"dim": "gpu_mem", "limit": 300, "used": ra["peak_gpu_mb"], "lifted_cap_used": rb["peak_gpu_mb"],
            "capped_alloc_refused": ra["rc"] == 0, "lifted_cap_killed": rb["killed_by"] == "gpu_mem",
            "contained": bool(contained), "record": ra, "record_b": rb}


def gpu_time(d: str) -> dict:
    """Grant 3 GPU-seconds; the job holds a CUDA context for 20 s: killed after about 3 s."""
    code = "import torch, time\nx=torch.ones(16, device='cuda'); torch.cuda.synchronize()\ntime.sleep(20)\n"
    r = _run(d, code, cpu=1, gpu_mb=300, gpu_s=3, label="probe gpu time")
    contained = r["killed_by"] == "gpu_time" and r["gpu_s"] <= 3 + 2 * SAMPLE + 0.5
    return {"dim": "gpu_time", "limit": 3, "used": r["gpu_s"], "contained": bool(contained), "record": r}


def disk(d: str) -> dict:
    """Grant 20 MB over a directory. (a) The job writes 15 files of 4 MB: killed. (b) One 40 MB file: stopped by the
    file-size limit."""
    out = tempfile.mkdtemp(prefix="disk-", dir=d)
    a = (f"import os, time\nfor i in range(15):\n    open(os.path.join({out!r}, f'f{{i}}'), 'wb').write(os.urandom(4<<20))"
         "\n    time.sleep(0.3)\nprint('FINISHED')\n")
    ra = _run(d, a, cpu=1, disk_mb=20, disk_dirs=[out], label="probe disk many files")
    written_a = slots._dir_mb(out)
    for f in os.listdir(out):
        os.unlink(os.path.join(out, f))
    b = f"import os\nf=open(os.path.join({out!r}, 'big'), 'wb')\nfor i in range(40): f.write(os.urandom(1<<20))\n"
    rb = _run(d, b, cpu=1, disk_mb=20, disk_dirs=[out], label="probe disk one big file")
    written_b = slots._dir_mb(out)
    for f in os.listdir(out):
        os.unlink(os.path.join(out, f))
    os.rmdir(out)
    contained = (ra["killed_by"] == "disk" and written_a <= 20 + 4 * (2 / SAMPLE) * SAMPLE / 0.3 + 4
                 and rb["rc"] != 0 and written_b <= 20.5)
    return {"dim": "disk", "limit": 20, "used": round(written_a, 2), "big_file_mb": round(written_b, 2),
            "big_file_rc": rb["rc"], "contained": bool(contained), "record": ra, "record_b": rb}


def inference(d: str) -> dict:
    """Grant 2 concurrent inference calls; 2 processes x 3 threads each try 4 calls of 0.2 s at once: never more
    than 2 in flight job-wide."""
    log = tempfile.mktemp(prefix="inf-", suffix=".log", dir=d)
    worker = ("import threading, time, os, json\nfrom crucible.scale import governor\n"
              f"LOG={log!r}\n"
              "def w():\n    for _ in range(4):\n        with governor.inference():\n"
              "            t0=time.time(); time.sleep(0.2); t1=time.time()\n"
              "        with open(LOG,'a') as f: f.write(json.dumps([t0,t1])+'\\n')\n"
              "ts=[threading.Thread(target=w) for _ in range(3)]\n[t.start() for t in ts]\n[t.join() for t in ts]\n")
    code = (f"import subprocess, sys\nW={worker!r}\n"
            "ps=[subprocess.Popen([sys.executable,'-c',W]) for _ in range(2)]\nsys.exit(max(p.wait() for p in ps))\n")
    r = _run(d, code, cpu=1, inference=2, label="probe inference concurrency")
    iv = [json.loads(x) for x in open(log)] if os.path.exists(log) else []
    os.path.exists(log) and os.unlink(log)
    ev = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, k in ev:
        cur += k
        peak = max(peak, cur)
    contained = r["rc"] == 0 and len(iv) == 24 and peak <= 2
    return {"dim": "inference", "limit": 2, "used": peak, "calls": len(iv), "contained": bool(contained), "record": r}


WORLDS_JOB = r'''
import json, subprocess, sys
import crucible
from crucible.core.kernel import World, WorldSpec
from crucible.scale.governor import ResourceLimitExceeded
crucible.load_all_modules()
sp = lambda s: WorldSpec("tiny", s, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}}], {})
CHILD = """
import crucible
from crucible.core.kernel import World, WorldSpec
from crucible.scale.governor import ResourceLimitExceeded
crucible.load_all_modules()
try:
    World.create(WorldSpec("tiny", 9, [{"name": "identity", "config": {}}], {})); print("admitted")
except ResourceLimitExceeded:
    print("refused")
"""
out = {}
w1 = World.create(sp(1)); w2 = World.create(sp(2))
try:
    World.create(sp(3)); out["third"] = "admitted"
except ResourceLimitExceeded:
    out["third"] = "refused"
out["other_process"] = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True).stdout.strip()
w1.close()
w4 = World.create(sp(4)); out["after_close"] = "admitted"
open(sys.argv[1], "w").write(json.dumps(out))
'''


def worlds(d: str) -> dict:
    """Grant 2 active worlds. Two worlds are created; a third (in the process, and in a second process of the job)
    is refused; after one is closed a new one is admitted."""
    outf = tempfile.mktemp(prefix="worlds-", dir=d)
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    r = slots.run_job([PY, "-c", WORLDS_JOB, outf], cpu=1, worlds=2, budget=probe_budget(), env=_env(),
                      sample_s=SAMPLE, poll_s=0.05, label="probe active worlds")
    try:
        out = json.loads(open(outf).read())
    except (OSError, ValueError):
        out = {}
    os.path.exists(outf) and os.unlink(outf)
    contained = (r["rc"] == 0 and out.get("third") == "refused" and out.get("other_process") == "refused"
                 and out.get("after_close") == "admitted" and r["peak_worlds"] <= 2)
    return {"dim": "worlds", "limit": 2, "used": r["peak_worlds"], "refused": r["worlds_refused"], "outcome": out,
            "contained": bool(contained), "record": r}


# ====================================================================== scheduling rules
def fifo(d: str) -> dict:
    """Budget 2 CPUs. A holds 1; B asks for 2, then C asks for 1 (which would fit next to A): C must not be admitted
    before B."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=2, gpu_mb=0, jobs=8)
    sleeper = [PY, "-c", "import time; time.sleep(60)"]
    a = subprocess.Popen(sleeper)
    slots.acquire(1, 0, "A", b, pid=a.pid, d=d)
    order, holders = [], {}

    def ask(name, n, delay):
        time.sleep(delay)
        k = subprocess.Popen(sleeper)
        holders[name] = k
        slots.acquire(n, 0, name, b, poll_s=0.05, pid=k.pid, d=d)
        order.append(name)
    ts = [threading.Thread(target=ask, args=("B", 2, 0.0)), threading.Thread(target=ask, args=("C", 1, 0.3))]
    for t in ts:
        t.start()
    time.sleep(1.2)
    while_a = list(order)
    a.kill()
    a.wait()
    t0 = time.time()
    while "B" not in order and time.time() - t0 < 10:
        time.sleep(0.05)
    time.sleep(0.6)
    after_a = list(order)
    if "B" in holders:
        holders["B"].kill()
        holders["B"].wait()
    for t in ts:
        t.join(10)
    for k in holders.values():
        k.kill()
        k.wait()
    names = list(order)
    contained = while_a == [] and after_a == ["B"] and names == ["B", "C"]
    return {"rule": "fifo", "while_A_held": while_a, "after_A": after_a, "order": names, "contained": bool(contained)}


def zombie(d: str) -> dict:
    """An unreaped (zombie) holder holds nothing."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    p = subprocess.Popen([PY, "-c", "pass"])
    t = time.time()
    while slots._stat(p.pid) and slots._stat(p.pid)[0] != "Z" and time.time() - t < 10:
        time.sleep(0.02)
    is_zombie = bool(slots._stat(p.pid)) and slots._stat(p.pid)[0] == "Z"
    slots.acquire(1, 0, "zombie", slots.Budget(4, 0, 4), pid=p.pid, d=d)
    held = [h for h in slots.status(d)["held"] if h["pid"] == p.pid]
    p.wait()
    return {"rule": "zombie", "was_zombie": is_zombie, "held_after": len(held), "contained": is_zombie and not held}


def escape(d: str) -> dict:
    """The command backgrounds a detached grandchild (new session) and exits: the slot stays held while it runs and
    the grandchild is stopped after the orphan grace."""
    pidf = tempfile.mktemp(prefix="esc-", dir=d)
    code = ("import subprocess, sys\n"
            f"p=subprocess.Popen([sys.executable,'-c','import time\\nt=time.time()\\nwhile time.time()-t<30: pass'],"
            f" start_new_session=True)\nopen({pidf!r},'w').write(str(p.pid))\n")
    r = _run(d, code, cpu=1, orphan_grace_s=1.0, label="probe escape")
    gpid = int(open(pidf).read()) if os.path.exists(pidf) else -1
    os.path.exists(pidf) and os.unlink(pidf)
    alive = slots._alive(gpid) if gpid > 0 else True
    contained = r["killed_by"] == "orphans" and not alive and r["wall_s"] < 10 and r["cpu_s"] > 0.5
    return {"rule": "escape", "grandchild_alive_after": alive, "wall_s": r["wall_s"], "cpu_s": r["cpu_s"],
            "contained": bool(contained), "record": r}


def foreign(d: str) -> dict:
    """A process the governor did not start (a sibling child of the supervising process) survives the kill of a job
    that exceeds its RAM; no signal was sent outside the job."""
    sib = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    code = "import time\nb=[]\nfor i in range(30):\n    b.append(bytearray(50<<20)); time.sleep(0.05)\n"
    r = _run(d, code, cpu=1, ram_mb=200, label="probe foreign")
    alive = slots._alive(sib.pid)
    signalled = [p for p, _ in r["signals"]]
    sib.kill()
    sib.wait()
    contained = r["killed_by"] == "ram" and alive and sib.pid not in signalled
    return {"rule": "foreign", "sibling_alive": alive, "signalled": signalled, "contained": bool(contained)}


def self_declared(d: str) -> dict:
    """With the stored budget full, a caller that sets CRUCIBLE_SLOT_CPUS=1000 in its environment still waits."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=2, gpu_mb=0, jobs=8)
    holder = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    slots.acquire(2, 0, "full", b, pid=holder.pid, d=d)
    old = os.environ.get("CRUCIBLE_SLOT_CPUS")
    os.environ["CRUCIBLE_SLOT_CPUS"] = "1000"
    try:
        slots.acquire(4, 0, "greedy", None, poll_s=0.05, timeout_s=0.5, d=d)
        admitted = True
    except TimeoutError:
        admitted = False
    finally:
        if old is None:
            os.environ.pop("CRUCIBLE_SLOT_CPUS", None)
        else:
            os.environ["CRUCIBLE_SLOT_CPUS"] = old
        holder.kill()
        holder.wait()
    return {"rule": "self_declared_budget", "admitted": admitted, "contained": not admitted}


def oversize(d: str) -> dict:
    """A request for 64 CPUs and 999,999 MB of GPU memory is clamped to the stored budget."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=2, gpu_mb=1000, jobs=8)
    holder = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    e = slots.acquire(64, 999_999, "huge", b, pid=holder.pid, d=d)
    holder.kill()
    holder.wait()
    return {"rule": "oversize", "granted_cpu": e["cpu"], "granted_gpu_mb": e["gpu_mb"], "clamped": e["clamped"],
            "contained": e["cpu"] <= 2 and e["gpu_mb"] <= 1000 and len(e["clamped"]) == 2}


def nested(d: str) -> dict:
    """Inside a job granted 1 CPU, a sub-job asking for 4 is clamped to the parent's grant and pinned inside it."""
    code = ("import json, os, sys\nfrom crucible.scale import slots\n"
            "rec=slots.run_job([sys.executable,'-c','import os; print(sorted(os.sched_getaffinity(0)))'],"
            " cpu=4, sample_s=0.2, poll_s=0.05)\n"
            "print(json.dumps({'cpu': rec['cpu'], 'cpus': rec['cpus'], 'nested': rec['nested'],"
            " 'parent': sorted(os.sched_getaffinity(0))}))\n")
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    outf = tempfile.mktemp(prefix="nest-", dir=d)
    r = slots.run_job(["bash", "-c", f"{PY} -c {_sh(code)} > {outf}"], cpu=1, budget=probe_budget(),
                      env=_env(CRUCIBLE_SLOTS_DIR=d), sample_s=SAMPLE, poll_s=0.05, label="probe nested")
    try:
        out = json.loads(open(outf).read().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        out = {}
    os.path.exists(outf) and os.unlink(outf)
    contained = (r["rc"] == 0 and out.get("nested") is True and out.get("cpu") == 1
                 and set(out.get("cpus", [-1])) <= set(out.get("parent", [])))
    return {"rule": "nested", "outcome": out, "contained": bool(contained)}


def _sh(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def reservation(d: str) -> dict:
    """Budget 2 CPUs. A holds 1; B (2 CPUs) waits at the head of the queue. A client that does not read the queue
    (the first version of this module: it admits whatever fits next to the held entries) must not be able to take
    the CPU that B is waiting for."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=2, gpu_mb=0, jobs=8)
    sleeper = [PY, "-c", "import time; time.sleep(60)"]
    a = subprocess.Popen(sleeper)
    slots.acquire(1, 0, "A", b, pid=a.pid, d=d)
    kb = subprocess.Popen(sleeper)
    got = []
    t = threading.Thread(target=lambda: got.append(slots.acquire(2, 0, "B", b, poll_s=0.05, pid=kb.pid, d=d)))
    t.start()
    time.sleep(0.5)

    def old_client_fits(held):                     # the first version's rule: CPU, GPU and job count of held entries
        return len(held) < b.jobs and sum(int(h["cpu"]) for h in held) + 1 <= b.cpus
    with slots._ledger(d) as box:
        blocked = not old_client_fits(box["held"])
        reserved = sum(int(h["cpu"]) for h in box["held"] if h.get("reserving"))
    a.kill()
    a.wait()
    t.join(10)
    kb.kill()
    kb.wait()
    return {"rule": "reservation", "old_client_blocked": blocked, "reserved_cpus": reserved,
            "head_admitted": bool(got), "contained": bool(blocked and got and reserved == 1)}


def backfill(d: str) -> dict:
    """Budget 3 CPUs, one GPU slot. A holds 1 CPU and the GPU slot; B (1 CPU + GPU) waits at the head; C (1 CPU,
    no GPU) queues behind B and D (1 CPU) behind C. C uses nothing B waits for and leaves B's CPU free, so it starts
    at once (backfill); D would take B's CPU, so it must wait while A holds. When A ends, B starts immediately
    (the backfilled C did not delay it)."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=3, gpu_mb=1000, jobs=8, gpu_jobs=1)
    sleeper = [PY, "-c", "import time; time.sleep(60)"]
    a = subprocess.Popen(sleeper)
    slots.acquire(1, 500, "A", b, pid=a.pid, d=d)
    order, kids, when = [], {}, {}

    def ask(name, cpu, gpu, delay):
        time.sleep(delay)
        k = kids[name] = subprocess.Popen(sleeper)
        slots.acquire(cpu, gpu, name, b, poll_s=0.05, pid=k.pid, d=d)
        order.append(name)
        when[name] = time.time()
    ts = [threading.Thread(target=ask, args=x) for x in (("B", 1, 500, 0.0), ("C", 1, 0, 0.3), ("D", 1, 0, 0.6))]
    for t in ts:
        t.start()
    time.sleep(1.5)
    while_a = list(order)
    a.kill()
    a.wait()
    t_free = time.time()
    for t in ts:
        t.join(10)
    for k in kids.values():
        k.kill()
        k.wait()
    b_delay = round(when.get("B", 1e18) - t_free, 3)
    contained = while_a == ["C"] and "B" in order and b_delay < 2.0
    return {"rule": "backfill", "while_A_held": while_a, "order": list(order), "head_start_delay_s": b_delay,
            "contained": bool(contained)}


def deadline_backfill(d: str) -> dict:
    """Budget 2 CPUs, one GPU slot. A holds 1 CPU and the GPU slot with a 1-hour time limit; B (1 CPU + GPU) waits
    at the head and keeps the other CPU. C (1 CPU, 2-minute time limit) ends before B could start (A's deadline), so
    it may use B's CPU meanwhile; D (1 CPU, no time limit) and E (1 CPU, 2-hour limit) could outlive A, so they wait.
    When A ends, B starts at once next to C: the short job did not delay the head."""
    os.environ["CRUCIBLE_SLOTS_DIR"] = d
    b = slots.Budget(cpus=2, gpu_mb=1000, jobs=8, gpu_jobs=1)
    sleeper = [PY, "-c", "import time; time.sleep(60)"]
    a = subprocess.Popen(sleeper)
    slots.acquire(1, 500, "A", b, pid=a.pid, d=d, run_s=3600)
    order, kids, when = [], {}, {}

    def ask(name, cpu, gpu, delay, run_s):
        time.sleep(delay)
        k = kids[name] = subprocess.Popen(sleeper)
        try:
            slots.acquire(cpu, gpu, name, b, poll_s=0.05, pid=k.pid, d=d, run_s=run_s, timeout_s=8)
        except TimeoutError:
            return
        order.append(name)
        when[name] = time.time()
    ts = [threading.Thread(target=ask, args=x) for x in (("B", 1, 500, 0.0, 3600), ("C", 1, 0, 0.3, 120),
                                                         ("D", 1, 0, 0.6, None), ("E", 1, 0, 0.9, 7200))]
    for t in ts:
        t.start()
    time.sleep(1.8)
    while_a = list(order)
    held_cpu = slots.status(d)["cpu_used"]
    a.kill()
    a.wait()
    t_free = time.time()
    time.sleep(1.0)
    after_a = list(order)
    for t in ts:
        t.join(12)
    for k in kids.values():
        k.kill()
        k.wait()
    b_delay = round(when.get("B", 1e18) - t_free, 3)
    contained = while_a == ["C"] and held_cpu <= 2 and after_a[:2] == ["C", "B"] and b_delay < 2.0
    return {"rule": "deadline_backfill", "while_A_held": while_a, "after_A": after_a, "order": list(order),
            "held_cpu_while_A": held_cpu, "head_start_delay_s": b_delay, "contained": bool(contained)}


DIMENSION_PROBES = {"cpu": cpu, "ram": ram, "gpu_mem": gpu_mem, "gpu_time": gpu_time, "disk": disk,
                    "inference": inference, "worlds": worlds}
RULE_PROBES = {"fifo": fifo, "reservation": reservation, "zombie": zombie, "escape": escape, "foreign": foreign,
               "self_declared_budget": self_declared, "oversize": oversize, "nested": nested, "backfill": backfill,
               "deadline_backfill": deadline_backfill}


def run_all(base: str | None = None, only: list[str] | None = None) -> dict:
    """Run the probes, each on a fresh private ledger under `base` (default: a temporary directory)."""
    base = base or tempfile.mkdtemp(prefix="govprobe-")
    old = os.environ.get("CRUCIBLE_SLOTS_DIR")
    out = {}
    try:
        for name, fn in {**DIMENSION_PROBES, **RULE_PROBES}.items():
            if only and name not in only:
                continue
            d = os.path.join(base, name)
            os.makedirs(d, exist_ok=True)
            t = time.time()
            try:
                out[name] = fn(d)
            except Exception as exc:                      # a probe that cannot run counts as not contained
                out[name] = {"contained": False, "error": f"{type(exc).__name__}: {exc}"}
            out[name]["probe_s"] = round(time.time() - t, 2)
    finally:
        if old is None:
            os.environ.pop("CRUCIBLE_SLOTS_DIR", None)
        else:
            os.environ["CRUCIBLE_SLOTS_DIR"] = old
    return out


if __name__ == "__main__":
    res = run_all(only=sys.argv[1:] or None)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("record")} for k, v in res.items()},
                     indent=1))
