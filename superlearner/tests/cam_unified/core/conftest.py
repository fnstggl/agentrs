"""Fixtures for workstream B tests: the shared-GPU lock for heavy tests and opt-in slow tests.

Slow tests (`@pytest.mark.slow`) in this directory run only with `-m slow` or CAM_RUN_SLOW=1, so the default run of
this directory stays short. Heavy GPU tests take /workspace/agentrs/runs/cam_unified/gpu_heavy.lock (shared by all
agents on the machine); do not run pytest itself under `flock` on that file (the fixture would wait on itself) unless
CAM_GPU_LOCK_HELD=1 is set.
"""
from __future__ import annotations

import fcntl
import os
import time

import pytest
import torch

LOCK = "/workspace/agentrs/runs/cam_unified/gpu_heavy.lock"
HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1" or "slow" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow or CAM_RUN_SLOW=1")
    for it in items:
        if str(it.fspath).startswith(HERE) and "slow" in it.keywords:
            it.add_marker(skip)


@pytest.fixture
def gpu_heavy_lock():
    """Exclusive use of the shared GPU for heavy (main/large-scale) tests; waits up to 30 minutes."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if os.environ.get("CAM_GPU_LOCK_HELD") == "1":
        yield
        return
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    fh = open(LOCK, "a")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > 1800:
                fh.close()
                pytest.fail("could not acquire the shared GPU lock within 30 minutes")
            time.sleep(2.0)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        yield
    finally:
        torch.cuda.empty_cache()
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
