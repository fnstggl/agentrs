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
