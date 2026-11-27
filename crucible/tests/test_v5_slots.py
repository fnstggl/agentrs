"""Resource governor (`crucible.scale.slots`, `crucible.scale.governor`): admission in every budgeted dimension, FIFO,
stored budget, zombie reclaim, CPU choice, environment and usage records, active-world and inference grants.
The adversarial enforcement probes (a job exceeding each limit) are in tests/v5/test_gate_SP_13_7.py."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from crucible.scale import governor, slots


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_SLOTS_DIR", str(tmp_path))
    return tmp_path


def test_a_request_fits_until_the_budget_is_full_in_every_dimension(ledger_dir):
    b = slots.Budget(cpus=4, gpu_mb=1000, jobs=3, ram_mb=8100, disk_mb=100, inference=4, worlds=10)
    held = [{"pid": 1, "cpu": 3, "gpu_mb": 600, "ram_mb": 6000, "disk_mb": 60, "inference": 3, "worlds": 8}]
    assert slots.fits(held, 1, 400, b)
    assert not slots.fits(held, 2, 0, b)                              # cpu
    assert not slots.fits(held, 1, 500, b)                            # gpu memory
    assert not slots.fits(held, 1, 0, b, ram_mb=2500)                 # ram
    assert not slots.fits(held, 1, 0, b, disk_mb=50)                  # disk
    assert not slots.fits(held, 1, 0, b, inference=2)                 # inference concurrency
    assert not slots.fits(held, 1, 0, b, worlds=3)                    # active worlds
    b1 = slots.Budget(cpus=4, gpu_mb=1000, jobs=3, gpu_jobs=1)
    assert not slots.fits(held, 1, 100, b1)                           # GPU jobs at once (the holder uses the GPU)
    assert not slots.fits([], 1, 0, b, disk_mb=10, free_disk_mb=b.disk_floor_mb + 5)   # free disk
    assert not slots.fits([], 99, 0, b)                               # no oversized request runs, not even alone
    req, cut = slots.clamp({"cpu": 99, "gpu_mb": 99999, "ram_mb": None}, b)
    assert req["cpu"] == 4 and req["gpu_mb"] == 1000 and len(cut) == 2 and slots.fits([], req["cpu"], req["gpu_mb"], b, ram_mb=1000)


def test_undeclared_ram_counts_a_default_reservation():
    b = slots.Budget(cpus=8, gpu_mb=0, jobs=8, ram_mb=3 * slots.UNDECLARED_RAM_PER_CPU_MB)
    held = [{"pid": 1, "cpu": 2, "gpu_mb": 0}]                        # an entry written by an older client
    assert slots.fits(held, 1, 0, b)
    assert not slots.fits(held, 2, 0, b)


def test_dead_and_zombie_holders_are_reclaimed(ledger_dir):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    slots.acquire(2, 0, "ghost", slots.Budget(4, 0, 4), pid=p.pid)
    assert slots.status()["held"] == []                               # the pid no longer exists
    z = subprocess.Popen([sys.executable, "-c", "pass"])
    t = time.time()
    while slots._stat(z.pid)[0] != "Z" and time.time() - t < 10:
        time.sleep(0.02)
    assert slots._stat(z.pid)[0] == "Z" and not slots._alive(z.pid)
    slots.acquire(1, 0, "zombie", slots.Budget(4, 0, 4), pid=z.pid)
    assert slots.status()["held"] == []
    z.wait()


def test_a_second_request_waits_until_the_first_releases(ledger_dir):
    b = slots.Budget(cpus=2, gpu_mb=0, jobs=4)
    slots.acquire(2, 0, "first", b)
    got = []

    def second():
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        slots.acquire(1, 0, "second", b, poll_s=0.05, pid=child.pid)
        got.append(time.time())
        child.kill()
        child.wait()

    t = threading.Thread(target=second)
    t.start()
    time.sleep(0.4)
    assert not got
    assert [q["label"] for q in slots.status()["queue"]] == ["second"]
    released = time.time()
    slots.release(os.getpid())
    t.join(5)
    assert got and got[0] >= released


def test_the_stored_budget_ignores_the_callers_environment(ledger_dir, monkeypatch):
    stored = slots.stored_budget(str(ledger_dir), default=slots.Budget(cpus=2, gpu_mb=0, jobs=4))
    monkeypatch.setenv("CRUCIBLE_SLOT_CPUS", "1000")
    assert slots.stored_budget(str(ledger_dir)).cpus == stored.cpus == 2
    assert slots._tighter(stored, slots.Budget(cpus=50, gpu_mb=0, jobs=4)).cpus == 2     # cannot widen
    assert slots._tighter(stored, slots.Budget(cpus=1, gpu_mb=0, jobs=4)).cpus == 1      # can narrow
    new = slots.set_budget({"cpus": 3}, str(ledger_dir), reason="test")
    assert new.cpus == 3 and json.load(open(ledger_dir / "budget.json"))["history"][-1]["reason"] == "test"


def test_cpus_are_chosen_disjoint_and_inside_the_affinity():
    allowed = sorted(os.sched_getaffinity(0))
    a = slots.choose_cpus([], 2, allowed)
    b = slots.choose_cpus([{"cpus": a}], 2, allowed)
    assert set(a) <= set(allowed) and set(b) <= set(allowed)
    if len(allowed) >= 4:
        assert not set(a) & set(b)
    assert len(slots.choose_cpus([], 10_000, allowed)) == len(allowed)


def test_run_pins_sets_threads_hides_the_gpu_and_logs_usage(ledger_dir):
    rec = slots.run_job([sys.executable, "-c",
                         "import os,sys; sys.exit(0 if os.environ['OMP_NUM_THREADS']=='1' "
                         "and os.environ['NUMBA_NUM_THREADS']=='1' and os.environ['CUDA_VISIBLE_DEVICES']=='' "
                         "and os.environ['CRUCIBLE_SLOT_RAM_MB']=='500' and len(os.sched_getaffinity(0))==1 else 7)"],
                        cpu=1, gpu_mb=0, label="env-check", budget=slots.Budget(4, 0, 2, ram_mb=4000), ram_mb=500,
                        sample_s=0.1, poll_s=0.05)
    assert rec["rc"] == 0 and rec["cpu"] == 1 and len(rec["cpus"]) == 1
    logged = [json.loads(x) for x in open(ledger_dir / "usage.jsonl")]
    assert logged[-1]["label"] == "env-check" and logged[-1]["rc"] == 0 and logged[-1]["job"] == rec["job"]
    assert logged[-1]["declared"] == ["ram_mb"] and logged[-1]["violations"] == []
    assert slots.status()["held"] == [] and slots.status()["queue"] == []
    assert slots.run([sys.executable, "-c", "import sys; sys.exit(3)"], cpu=1, budget=slots.Budget(4, 0, 2),
                     sample_s=0.1, poll_s=0.05) == 3


def test_world_registration_outside_a_job_only_counts():
    import crucible
    from crucible.core.kernel import World, WorldSpec
    crucible.load_all_modules()
    n0 = governor.active_worlds()
    w = World.create(WorldSpec("tiny", 1, [{"name": "identity", "config": {}}], {}))
    assert governor.active_worlds() == n0 + 1
    w.close()
    w.close()
    assert governor.active_worlds() == n0
    with governor.inference():
        assert governor.inflight() == 1
    assert governor.inflight() == 0


def test_the_cgroup_quota_and_hardware_are_read():
    assert slots.cgroup_cpus() > 0 and slots.cgroup_mem_mb() > 0
    hw = slots.hardware()
    assert hw["logical_cpus"] >= len(hw["affinity"]) >= 1 and hw["mem_mb"] > 0


def _sleeper(s=60):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({s})"])


def test_backfill_admits_a_job_that_cannot_delay_the_waiting_head(ledger_dir):
    """A GPU job waits for the only GPU slot; a CPU-only job behind it starts at once (it uses nothing the head waits
    for and leaves the head's CPU free); a job that would take the head's CPU waits; the head starts as soon as the
    GPU slot is free, before the job that queued after it."""
    b = slots.Budget(cpus=3, gpu_mb=1000, jobs=8, gpu_jobs=1)
    a = _sleeper()
    slots.acquire(1, 500, "A gpu", b, pid=a.pid)
    order, kids = [], {}

    def ask(name, cpu, gpu, delay):
        time.sleep(delay)
        k = kids[name] = _sleeper()
        e = slots.acquire(cpu, gpu, name, b, poll_s=0.03, pid=k.pid)
        order.append((name, e["backfilled"]))
    ts = [threading.Thread(target=ask, args=a) for a in (("B gpu", 1, 500, 0.0), ("C cpu", 1, 0, 0.25),
                                                          ("D cpu", 1, 0, 0.5))]
    for t in ts:
        t.start()
    time.sleep(1.2)
    assert order == [("C cpu", True)]                       # C backfilled; D would take B's CPU: it waits
    a.kill()
    a.wait()
    t0 = time.time()
    while len(order) < 3 and time.time() - t0 < 10:
        time.sleep(0.02)
    for t in ts:
        t.join(10)
    # the GPU slot is free: B (head) starts at once; D fits next to B and C now, so it starts too (it delays nobody)
    assert {o[0] for o in order[1:]} == {"B gpu", "D cpu"} and time.time() - t0 < 5
    assert ("B gpu", False) in order
    for k in kids.values():
        k.kill()
        k.wait()


def test_a_cpu_job_starts_behind_a_long_queue_of_gpu_jobs(ledger_dir):
    """The situation of the shared machine: GPU slots full, several GPU jobs queued, CPUs idle. A CPU-only job behind
    them starts (only the head's share is reserved for it; the other GPU jobs could not start now anyway), and a
    request that could start now is never overtaken."""
    b = slots.Budget(cpus=4, gpu_mb=4000, jobs=8, gpu_jobs=1)
    a = _sleeper()
    slots.acquire(1, 500, "A gpu", b, pid=a.pid)
    held = [h for h in slots.status()["held"]]
    ahead = [{"cpu": 1, "gpu_mb": 500, "ram_mb": None}] * 3
    assert slots.fits_behind(held, ahead, 1, 0, b)            # E (CPU only) behind B, C, D (GPU)
    assert not slots.fits_behind(held, ahead, 1, 500, b)       # another GPU job waits its turn
    assert not slots.fits_behind(held, ahead, 3, 0, b)         # would take the head's CPU
    ready = [{"cpu": 2, "gpu_mb": 0, "ram_mb": None}, {"cpu": 1, "gpu_mb": 0, "ram_mb": None}]
    assert not slots.fits_behind(held, ready, 1, 0, b)         # both ahead could start now: nobody overtakes them
    a.kill()
    a.wait()


def test_eighteen_concurrent_clients_never_exceed_the_budget_and_all_get_through(ledger_dir):
    """18 clients (the number of engineers sharing the machine) ask repeatedly for mixed CPU/GPU slots of a 9-CPU,
    4-GPU-job budget, some with time limits (deadline backfilling): the held slots never exceed the budget in any
    dimension and nobody starves (every request is admitted)."""
    import random
    b = slots.Budget(cpus=9, gpu_mb=24000, jobs=12, ram_mb=60000, gpu_jobs=4)
    stop, problems, done = threading.Event(), [], []

    def monitor():
        while not stop.is_set():
            st = slots.status()
            real = [h for h in st["held"] if not h.get("reserving")]
            if sum(int(h["cpu"]) for h in st["held"]) > b.cpus or len(st["held"]) > b.jobs \
                    or sum(1 for h in real if int(h["gpu_mb"]) > 0) > b.gpu_jobs \
                    or sum(int(h["gpu_mb"]) for h in st["held"]) > b.gpu_mb \
                    or sum(slots._ram_of(h) for h in real) > b.ram_mb:
                problems.append(st["held"])
            time.sleep(0.01)

    def client(i):
        rnd = random.Random(i)
        for k in range(3):
            cpu, gpu = rnd.choice(((1, 0), (1, 3000), (2, 0), (3, 4000), (1, 0)))
            t = time.time()
            e = slots.acquire(cpu, gpu, f"c{i}.{k}", b, poll_s=0.02, ram_mb=rnd.choice((4000, 8000, 12000)),
                              job=f"c{i}.{k}", run_s=rnd.choice((None, 0.5, 30.0, 600.0)))   # deadlines: backfill
            done.append((f"c{i}.{k}", time.time() - t, e["queue_pos"]))
            time.sleep(rnd.uniform(0.02, 0.15))
            slots.release_job(f"c{i}.{k}")
    m = threading.Thread(target=monitor)
    m.start()
    cs = [threading.Thread(target=client, args=(i,)) for i in range(18)]
    for c in cs:
        c.start()
    for c in cs:
        c.join(120)
    stop.set()
    m.join(5)
    assert not problems, problems[:2]
    assert len(done) == 18 * 3                                # nobody starved
    assert slots.status()["held"] == [] and slots.status()["queue"] == []


def test_a_reused_pid_does_not_keep_a_dead_holders_slot(ledger_dir):
    b = slots.Budget(cpus=4, gpu_mb=0, jobs=4)
    k = _sleeper()
    slots.acquire(2, 0, "holder", b, pid=k.pid)
    assert len(slots.status()["held"]) == 1
    with slots._ledger() as box:                              # as if the holder died and its pid was reused
        box["held"][0]["start"] = str(int(box["held"][0]["start"]) - 1)
    assert slots.status()["held"] == []
    k.kill()
    k.wait()


def test_a_member_whose_identity_changed_is_never_signalled():
    k = _sleeper()
    job = slots.Job(job="j", grant={}, cpus=[], jobdir="/nonexistent", disk_dirs=[], cli=False, sample_s=1.0)
    job.start[k.pid] = "1"                                    # what was seen as a member is another process now
    assert slots._signal_member(job, k.pid, 15) is False and k.poll() is None
    job.start[k.pid] = slots._starttime(k.pid)
    assert slots._signal_member(job, k.pid, 15) is True
    k.wait(10)
    assert k.returncode == -15


def test_a_head_waiting_only_for_free_disk_gives_up(ledger_dir, monkeypatch):
    monkeypatch.setattr(slots, "DISK_WAIT_S", 0.3)
    b = slots.Budget(cpus=4, gpu_mb=0, jobs=4, disk_mb=10 ** 9)
    free = __import__("shutil").disk_usage(str(ledger_dir)).free // slots.MB
    with pytest.raises(slots.AdmissionError):
        slots.acquire(1, 0, "too much disk", b, poll_s=0.05, disk_mb=int(free + 10 ** 6))
    assert slots.status()["queue"] == [] and slots.status()["held"] == []


def test_a_process_that_only_asks_whether_a_gpu_exists_is_not_a_gpu_job():
    """SP-1.2 census: `torch.cuda.is_available()` opens the GPU device (every process importing the shared semantic
    space evaluates it), but creates no CUDA context and uses no GPU memory; the census counts such a process apart,
    never as an ungoverned GPU job. (The positive control, a real context, is in tests/v5/test_gate_SP_1_2.py.)"""
    code = ("import sys, time, torch\ntorch.cuda.is_available()\nprint('ready', flush=True)\n"
            "sys.stdin.read()\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK")}
    p = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env=env)
    try:
        assert p.stdout.readline().strip() == "ready"
        opened = slots._gpu_device_open(p.pid)
        assert not slots._context_mapped(p.pid) and not slots._gpu_context(p.pid)
        c = slots.gpu_census()
        assert p.pid not in {x["pid"] for x in c["ungoverned"] + c["governed"]}
        assert c["device_queries_without_context"] >= (1 if opened else 0)
    finally:
        p.stdin.close()
        p.wait(timeout=30)


def test_status_shows_what_each_held_job_uses_next_to_its_grant(ledger_dir, monkeypatch):
    """Each supervisor publishes its job's current and peak use (jobdir/live.json); `--status` shows it beside the
    grant, so engineers can size requests from what their running jobs really use."""
    monkeypatch.setenv("CRUCIBLE_SLOTS_LIVE_S", "0.2")          # the supervisor process reads it at start
    recs = []
    code = "import time\nx = bytearray(150 * 1024 * 1024)\nx[::4096] = b'1' * len(x[::4096])\ntime.sleep(4)\n"
    t = threading.Thread(target=lambda: recs.append(slots.run_job(
        [sys.executable, "-c", code], cpu=1, gpu_mb=0, label="live-use", budget=slots.Budget(4, 0, 2, ram_mb=4000),
        ram_mb=1500, sample_s=0.2, poll_s=0.05)))
    t.start()
    live, deadline = None, time.time() + 30
    while time.time() < deadline and live is None:
        st = slots.status()
        held = [h for h in st["held"] if h["label"] == "live-use" and "live" in h]
        if held and held[0]["live"]["peak_ram_mb"] > 100:
            live, total = held[0]["live"], st
        time.sleep(0.2)
    t.join(30)
    assert live is not None, "no live usage published for a running job"
    assert 100 < live["peak_ram_mb"] < 1500 and live["ram_mb"] <= live["peak_ram_mb"] + 1e-6
    assert total["jobs_reporting_use"] >= 1 and total["ram_mb_in_use_reported"] > 0
    assert recs and recs[0]["rc"] == 0 and slots.status()["held"] == []


def test_the_default_gpu_time_covers_every_worker_for_the_whole_time_limit(ledger_dir):
    """A fixed 4-hour default stopped a two-run GPU benchmark with an 18-hour timeout after 2 hours of wall time
    (GPU time sums the context-holding seconds of every process). The default now covers the time limit per worker."""
    assert slots.default_gpu_s(1, None) == slots.DEFAULT_GPU_S
    assert slots.default_gpu_s(2, None) == 2 * slots.DEFAULT_GPU_S
    assert slots.default_gpu_s(1, 600) == slots.DEFAULT_GPU_S           # a short job keeps the 4-hour floor
    assert slots.default_gpu_s(2, 18 * 3600) == 36 * 3600
    rec = slots.run_job([sys.executable, "-c",
                         "import os,sys; sys.exit(0 if os.environ['CRUCIBLE_SLOT_GPU_S']=='72000' else 9)"],
                        cpu=2, gpu_mb=100, label="gpu-s-default", budget=slots.Budget(4, 1000, 2, ram_mb=4000),
                        ram_mb=500, timeout_s=36000, sample_s=0.1, poll_s=0.05)
    assert rec["rc"] == 0 and rec["gpu_s_grant"] == 72000
    rec = slots.run_job([sys.executable, "-c", "import os,sys; sys.exit(0 if os.environ['CRUCIBLE_SLOT_GPU_S']=='60' "
                         "else 9)"], cpu=2, gpu_mb=100, label="gpu-s-explicit", gpu_s=60,
                        budget=slots.Budget(4, 1000, 2, ram_mb=4000), ram_mb=500, timeout_s=36000, sample_s=0.1,
                        poll_s=0.05)
    assert rec["rc"] == 0 and rec["gpu_s_grant"] == 60


def test_a_short_job_uses_the_heads_share_until_the_heads_planned_start(ledger_dir):
    """Deadline (shadow-time) backfilling: the head waits for the only GPU slot, held by a job whose deadline is an
    hour away, and keeps one of the two CPUs for itself. A CPU job whose own deadline comes before that hour starts
    in the head's share; a longer one, or one without a time limit, waits; nothing changes while the GPU holder has
    no deadline (its end is unknown). The head's reservation is trimmed so the held CPUs stay within the budget."""
    b = slots.Budget(cpus=2, gpu_mb=1000, jobs=8, gpu_jobs=1)
    now = 1000.0
    gpu = {"pid": 1, "cpu": 1, "gpu_mb": 500, "ram_mb": None, "until": now + 3600}
    head = {"cpu": 1, "gpu_mb": 500, "ram_mb": None}
    assert slots.shadow_time([gpu], head, b, now) == now + 3600
    assert slots.shadow_time([{**gpu, "until": None}], head, b, now) is None
    assert slots.fits_behind([gpu], [head], 1, 0, b, run_s=600, now=now)
    assert not slots.fits_behind([gpu], [head], 1, 0, b, run_s=3600, now=now)       # would outlive the head's wait
    assert not slots.fits_behind([gpu], [head], 1, 0, b, now=now)                    # no time limit
    assert not slots.fits_behind([{**gpu, "until": None}], [head], 1, 0, b, run_s=60, now=now)
    assert not slots.fits_behind([gpu], [head], 1, 500, b, run_s=60, now=now)        # GPU slot: still the head's turn
    # a head that can start now is never overtaken, however short the request
    assert not slots.fits_behind([], [{"cpu": 2, "gpu_mb": 0, "ram_mb": None}], 1, 0, b, run_s=1, now=now)

    # through the ledger: a GPU holder with a deadline, a waiting GPU head, a short CPU job behind it
    a = _sleeper()
    slots.acquire(1, 500, "A gpu", b, pid=a.pid, run_s=3600)
    held0 = slots.status()["held"]
    assert held0[0]["until"] >= time.time() + 3600 and "until" in held0[0]
    kids, got = {}, {}

    def ask(name, gpu, run_s):
        k = kids[name] = _sleeper()
        got[name] = slots.acquire(1, gpu, name, b, poll_s=0.03, pid=k.pid, run_s=run_s)
    th = threading.Thread(target=ask, args=("B gpu", 500, 3600))
    th.start()
    t0 = time.time()
    while not any(h.get("reserving") and h["cpu"] for h in slots.status()["held"]) and time.time() - t0 < 5:
        time.sleep(0.02)
    tc = threading.Thread(target=ask, args=("C short", 0, 120))
    tc.start()
    tc.join(5)
    assert got["C short"]["backfilled"] and got["C short"]["until"] <= time.time() + 120 + slots.STOP_GRACE_S
    st = slots.status()
    assert st["cpu_used"] <= b.cpus and "B gpu" not in got
    a.kill()                                       # the GPU holder ends: the head starts next to the short job
    a.wait()
    th.join(10)
    assert "B gpu" in got and not got["B gpu"]["backfilled"]
    assert slots.status()["cpu_used"] == 2 and slots.status()["queue"] == []
    for k in kids.values():
        k.kill()
        k.wait()


def test_status_says_what_each_waiting_request_waits_for(ledger_dir):
    """`--status` names the dimensions each queued request is short of, and the head's planned start when every
    holder it waits for has a deadline."""
    b = slots.Budget(cpus=2, gpu_mb=1000, jobs=8, gpu_jobs=1)
    assert slots.short_of([], {"cpu": 1}, b) == []
    assert slots.short_of([{"cpu": 2, "gpu_mb": 0}], {"cpu": 1, "gpu_mb": 100}, b) == ["cpu"]
    assert slots.short_of([{"cpu": 1, "gpu_mb": 500}], {"cpu": 1, "gpu_mb": 600}, b) == ["gpu_mb", "gpu_jobs"]
    a = _sleeper()
    slots.acquire(1, 500, "A gpu", b, pid=a.pid, run_s=3600)
    kids = []

    def ask(name, cpu, gpu):
        k = _sleeper()
        kids.append(k)
        try:
            slots.acquire(cpu, gpu, name, b, poll_s=0.03, pid=k.pid, timeout_s=3)
        except TimeoutError:
            pass
    ts = [threading.Thread(target=ask, args=x) for x in (("B gpu", 1, 500), ("C cpu", 2, 0))]
    for t in ts:
        t.start()
        time.sleep(0.3)
    q = slots.status()["queue"]
    assert [x["label"] for x in q] == ["B gpu", "C cpu"]
    assert q[0]["waiting_for"] == ["gpu_jobs"] and q[0]["planned_start"] >= time.time() + 3600
    assert q[1]["waiting_for"] == ["cpu"]
    for t in ts:
        t.join(10)
    for k in kids + [a]:
        k.kill()
        k.wait()
