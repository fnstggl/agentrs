"""Compute slots (`crucible.scale.slots`): requests wait for budget, dead holders are reclaimed, usage is logged."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from crucible.scale import slots


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_SLOTS_DIR", str(tmp_path))
    return tmp_path


def test_a_request_fits_until_the_budget_is_full(ledger_dir):
    b = slots.Budget(cpus=4, gpu_mb=1000, jobs=3)
    held = [{"pid": 1, "cpu": 3, "gpu_mb": 600}]
    assert slots.fits(held, 1, 400, b)
    assert not slots.fits(held, 2, 0, b)
    assert not slots.fits(held, 1, 500, b)
    assert slots.fits([], 99, 99999, b)                      # an oversized request runs alone rather than never


def test_dead_holders_are_reclaimed(ledger_dir):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    slots.acquire(2, 0, "ghost", slots.Budget(4, 0, 4), pid=p.pid)
    assert slots.status()["held"] == []                       # the pid no longer exists


def test_a_second_request_waits_until_the_first_releases(ledger_dir):
    b = slots.Budget(cpus=2, gpu_mb=0, jobs=4)
    slots.acquire(2, 0, "first", b)
    got = []

    def second():
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        slots.acquire(1, 0, "second", b, poll_s=0.05, pid=child.pid)
        got.append(time.time())
        child.kill()

    t = threading.Thread(target=second)
    t.start()
    time.sleep(0.4)
    assert not got
    released = time.time()
    slots.release(os.getpid())
    t.join(5)
    assert got and got[0] >= released


def test_run_sets_threads_hides_the_gpu_and_logs_usage(ledger_dir):
    rc = slots.run([sys.executable, "-c", "import os,sys; sys.exit(0 if os.environ['OMP_NUM_THREADS']=='3' and "
                    "os.environ['CUDA_VISIBLE_DEVICES']=='' else 7)"], cpu=3, gpu_mb=0, label="env-check",
                   budget=slots.Budget(4, 0, 2))
    assert rc == 0
    rec = [json.loads(x) for x in open(ledger_dir / "usage.jsonl")]
    assert rec[-1]["label"] == "env-check" and rec[-1]["rc"] == 0 and rec[-1]["cpu"] == 3
    assert slots.status()["held"] == []


def test_the_cgroup_quota_is_read():
    assert slots.cgroup_cpus() > 0
