"""cam_live.nn.gpu — the shared-GPU lock for scripts (the same ``flock`` that ``tests/conftest.py::gpu_lock`` takes).

Any job that loads the real backbone or needs more than 4 GB of GPU memory holds this lock, for at most about
15 minutes, and frees its memory before releasing it. Never wrap pytest in it.
"""
from __future__ import annotations

import contextlib
import fcntl
import gc
import os
import time
from typing import Iterator

from ..config import GPU_LOCK_PATH


class GpuUnavailable(RuntimeError):
    """The lock was not acquired in time, or too little GPU memory is free."""


def open_lock_file(path: str):
    """The machine's shared lock file. It is never created here: a missing file means this process would lock a
    private file and coordinate with nobody (G2-integration G2I-18: the default path is shared only through a
    symlink in this worktree; a git archive or another checkout has none) — so it fails closed with
    :class:`GpuUnavailable` naming ``CAM_LIVE_GPU_LOCK``. A service uid that may not write the file (the machine's
    lock is root:root 0664; the separated learner ``camlearn`` is neither) opens it read-only: ``flock`` works on
    any open file, so every process takes the same lock (measured 2026-11-28: ``open(path, "a+")`` as camlearn
    raised PermissionError)."""
    if not os.path.exists(path):
        raise GpuUnavailable(f"the shared GPU lock file {path} does not exist; set CAM_LIVE_GPU_LOCK to the "
                             "machine's lock file (on this machine /workspace/agentrs/runs/cam_unified/gpu_heavy.lock)"
                             " — a private lock would coordinate with no other GPU user")
    try:
        return open(path, "r+")
    except PermissionError:
        return open(path, "r")


@contextlib.contextmanager
def gpu_lock(timeout_s: float = 1800.0, min_free_gb: float = 0.0, poll_s: float = 5.0) -> Iterator[float]:
    """Hold the shared lock; yields the time it was acquired. Raises :class:`GpuUnavailable` on timeout or when
    less than ``min_free_gb`` of GPU memory is free once the lock is held."""
    f = open_lock_file(GPU_LOCK_PATH)
    t0 = time.time()
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > timeout_s:
                    raise GpuUnavailable(f"shared GPU lock not acquired within {timeout_s:.0f} s")
                time.sleep(poll_s)
        try:
            if min_free_gb:
                import torch
                free = torch.cuda.mem_get_info()[0] / 2 ** 30
                if free < min_free_gb:
                    raise GpuUnavailable(f"only {free:.1f} GB GPU memory free, need {min_free_gb}")
            yield time.time()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
