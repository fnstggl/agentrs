"""nn.gpu.gpu_lock queues a blocking flock (a non-blocking poller starved behind holders that re-queue at once)."""
import fcntl
import os
import threading
import time

import pytest


def _module(lock, monkeypatch):
    import cam_live.nn.gpu as gpu
    monkeypatch.setattr(gpu, "GPU_LOCK_PATH", lock)     # no module reload (it would replace GpuUnavailable)
    return gpu


def test_a_waiter_gets_the_lock_between_holders_that_requeue_immediately(tmp_path, monkeypatch):
    lock = str(tmp_path / "gpu.lock")
    open(lock, "a").close()
    gpu = _module(lock, monkeypatch)
    stop = threading.Event()

    def churn():                       # another tenant: release and re-queue a blocking flock at once, repeatedly
        while not stop.is_set():
            with open(lock, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                time.sleep(0.2)
                fcntl.flock(f, fcntl.LOCK_UN)
    ts = [threading.Thread(target=churn, daemon=True) for _ in range(2)]
    for t in ts:
        t.start()
    time.sleep(0.3)
    try:
        t0 = time.time()
        with gpu.gpu_lock(timeout_s=20.0):
            got = time.time() - t0
        assert got < 10.0
    finally:
        stop.set()
        for t in ts:
            t.join(5)


def test_timeout_abandons_the_wait_and_releases_a_late_grant(tmp_path, monkeypatch):
    lock = str(tmp_path / "gpu.lock")
    open(lock, "a").close()
    gpu = _module(lock, monkeypatch)
    holder = open(lock, "r+")
    fcntl.flock(holder, fcntl.LOCK_EX)
    with pytest.raises(gpu.GpuUnavailable):
        with gpu.gpu_lock(timeout_s=0.5):
            pass
    fcntl.flock(holder, fcntl.LOCK_UN)          # the abandoned waiter is granted now and must release at once
    time.sleep(0.5)
    probe = open(lock, "r+")
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises BlockingIOError if the abandoned wait kept it
    fcntl.flock(probe, fcntl.LOCK_UN)
    holder.close()
    probe.close()
