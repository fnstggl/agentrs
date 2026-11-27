"""Shared test setup for CAM Live. MASTER-OWNED (add per-directory conftest.py files for local fixtures)."""
import contextlib
import fcntl
import os
import sys
import time

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")


def pytest_collection_modifyitems(config, items):
    """Opt-in tests run only when asked for with -m (keeps the default suite fast, CPU-only and offline).
    ``gpu`` / ``slow``: every such mark on a test must be named. ``browser`` / ``privileged`` / ``network``: a test
    carrying several of them runs when any one is named (``-m privileged`` runs privileged browser tests)."""
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in ("gpu", "slow"):
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
        env_marks = [m for m in ("browser", "privileged", "network") if m in item.keywords]
        if env_marks and not any(m in expr for m in env_marks):
            item.add_marker(pytest.mark.skip(reason=f"run with -m {env_marks[0]}"))


@contextlib.contextmanager
def gpu_lock(timeout_s: float = 1800.0, min_free_gb: float = 0.0):
    """Serialise heavy GPU work with every other tenant of this GPU. Hold for <= ~15 minutes. Never wrap pytest
    itself in ``flock`` on this file: tests take the lock here and would deadlock."""
    from cam_live.config import GPU_LOCK_PATH
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    f = open(GPU_LOCK_PATH, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > timeout_s:
                f.close()
                pytest.skip("shared GPU lock not acquired in time")
            time.sleep(5)
    try:
        if min_free_gb:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < min_free_gb:
                pytest.skip(f"only {free:.1f} GB GPU memory free, need {min_free_gb}")
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


@pytest.fixture
def gpu_heavy():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    with gpu_lock(min_free_gb=14.0):
        yield
        torch.cuda.empty_cache()
