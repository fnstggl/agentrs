"""In-process side of the resource governor (MASTER_SPEC 13.7, SP-13.7): the limits a governed job enforces inside
its own processes.

`crucible.scale.slots` admits a job, pins it and watches it from outside. Two of the seven dimensions can only be
enforced at the point of use, so the job's processes enforce them here, against the grant the supervisor put in the
environment:

* **Active worlds** (`CRUCIBLE_SLOT_WORLDS`): every `crucible.core.kernel.World` registers on construction
  (`register_world`) and leaves on `close()` or garbage collection. With a grant, construction beyond it raises
  `ResourceLimitExceeded`. The count is job-wide: processes of one job share a registry file in the job directory
  (pid -> live worlds, under `flock`, dead pids pruned), so a `WorldPool` with 8 workers cannot hold 8x the grant.
* **Inference concurrency** (`CRUCIBLE_SLOT_INFERENCE`): `with inference(): ...` holds one of the job's N tokens
  (one `flock`ed token file each, released by the kernel when a holder dies) for the duration of one inference call.
  `crucible.substrate.inference.recorded` wraps every recorded neural inference in it, so in-world inference of the
  whole job never has more than N calls in flight. Without a grant the gate only counts.

GPU memory and GPU time are capped in-process by the interpreter hook `crucible/scale/_slotsite/sitecustomize.py`
(torch's caching-allocator limit at CUDA initialisation, plus a reporter the supervisor reads); this module exposes
the same grant for code that wants to check it (`gpu_grant_mb`). Everything here is a no-op costing a few
microseconds when the process is not inside a governed job.

Peaks are written to the job directory so the supervisor's usage record has them (`stats()`).
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import weakref
from contextlib import contextmanager

ENV_JOB = "CRUCIBLE_SLOT_JOB"
ENV_JOBDIR = "CRUCIBLE_SLOT_JOBDIR"
ENV_WORLDS = "CRUCIBLE_SLOT_WORLDS"
ENV_INFER = "CRUCIBLE_SLOT_INFERENCE"
ENV_GPU_MB = "CRUCIBLE_SLOT_GPU_MB"
ENV_GPU_S = "CRUCIBLE_SLOT_GPU_S"
ENV_RAM = "CRUCIBLE_SLOT_RAM_MB"
ENV_CPU = "CRUCIBLE_SLOT_CPU"


class ResourceLimitExceeded(RuntimeError):
    """A governed job tried to go beyond one of its granted limits (the dimension is in `.dimension`)."""

    def __init__(self, dimension: str, msg: str):
        super().__init__(msg)
        self.dimension = dimension


def _int_env(name: str) -> int | None:
    v = os.environ.get(name)
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def jobdir() -> str | None:
    d = os.environ.get(ENV_JOBDIR)
    return d if d and os.path.isdir(d) else None


def grants() -> dict:
    """The job's grant as the supervisor exported it (None: not declared)."""
    return {"job": os.environ.get(ENV_JOB), "cpu": _int_env(ENV_CPU), "ram_mb": _int_env(ENV_RAM),
            "gpu_mb": _int_env(ENV_GPU_MB), "gpu_s": _int_env(ENV_GPU_S), "inference": _int_env(ENV_INFER),
            "worlds": _int_env(ENV_WORLDS)}


def gpu_grant_mb() -> int | None:
    return _int_env(ENV_GPU_MB)


# ====================================================================== shared job files
@contextmanager
def _locked_json(path: str):
    """Read-modify-write a small JSON dict under an exclusive flock (its own lock file)."""
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        box = {"data": data}
        try:
            yield box
        finally:
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(box["data"], fh)
            os.replace(tmp, path)
            fcntl.flock(lk, fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            st = fh.read()
        return st[st.rindex(")") + 2] not in "ZX"
    except (OSError, ValueError):
        return False


# ====================================================================== active worlds
_LOCK = threading.Lock()
_LIVE: dict[int, weakref.finalize] = {}
_PEAK = {"worlds": 0, "inference": 0}
_INFLIGHT = [0]


def active_worlds() -> int:
    """Live (not closed, not collected) worlds of this process."""
    with _LOCK:
        return len(_LIVE)


def _job_worlds(delta: int, limit: int | None) -> int:
    """Apply `delta` to this process's entry in the job's world registry; return the job-wide count afterwards.
    Raises ResourceLimitExceeded (and applies nothing) when the job-wide count would exceed `limit`."""
    d = jobdir()
    if d is None:
        return active_worlds()
    path = os.path.join(d, "worlds.json")
    with _locked_json(path) as box:
        counts = {p: int(n) for p, n in box["data"].get("pids", {}).items() if _pid_alive(int(p))}
        me = str(os.getpid())
        new = max(0, counts.get(me, 0) + delta)
        total = sum(n for p, n in counts.items() if p != me) + new
        if delta > 0 and limit is not None and total > limit:
            box["data"]["pids"] = counts
            box["data"]["refused"] = int(box["data"].get("refused", 0)) + 1
            raise ResourceLimitExceeded("worlds", f"active-world grant of this job is {limit}; {total - delta} live")
        counts[me] = new
        box["data"]["pids"] = counts
        box["data"]["peak"] = max(int(box["data"].get("peak", 0)), total)
        return total


def _release_key(key: int) -> None:
    with _LOCK:
        if _LIVE.pop(key, None) is None:
            return
    if _int_env(ENV_WORLDS) is not None:
        try:
            _job_worlds(-1, None)
        except OSError:
            pass


def register_world(world) -> None:
    """Called by World.__init__. Raises ResourceLimitExceeded beyond the job's active-world grant."""
    limit = _int_env(ENV_WORLDS)
    key = id(world)
    if limit is not None:
        try:
            _job_worlds(+1, limit)
        except ResourceLimitExceeded:
            import gc
            gc.collect()                      # worlds that are unreachable but not yet collected are not active
            _job_worlds(+1, limit)
    with _LOCK:
        _LIVE[key] = weakref.finalize(world, _release_key, key)
        _PEAK["worlds"] = max(_PEAK["worlds"], len(_LIVE))


def release_world(world) -> None:
    """Called by World.close(): the world no longer counts as active."""
    key = id(world)
    with _LOCK:
        fin = _LIVE.get(key)
    if fin is not None:
        fin()                                   # runs _release_key once and detaches the finalizer


# ====================================================================== inference concurrency
@contextmanager
def inference():
    """Hold one of the job's inference tokens while the body runs (no-op counter without a grant)."""
    limit = _int_env(ENV_INFER)
    d = jobdir()
    fh = None
    if limit is not None and limit > 0 and d is not None:
        tdir = os.path.join(d, "inference")
        os.makedirs(tdir, exist_ok=True)
        k = 0
        while fh is None:
            f = open(os.path.join(tdir, f"tok.{k % limit}"), "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if k < limit * 4 else 0))
                fh = f
            except BlockingIOError:
                f.close()
                k += 1
                if k % limit == 0:
                    time.sleep(0.002)
    with _LOCK:
        _INFLIGHT[0] += 1
        _PEAK["inference"] = max(_PEAK["inference"], _INFLIGHT[0])
    try:
        yield
    finally:
        with _LOCK:
            _INFLIGHT[0] -= 1
        if fh is not None:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()


def inflight() -> int:
    with _LOCK:
        return _INFLIGHT[0]


def stats() -> dict:
    """Peaks of this process (and, inside a job, the job-wide world peak)."""
    out = {"pid": os.getpid(), "active_worlds": active_worlds(), "peak_worlds": _PEAK["worlds"],
           "peak_inference": _PEAK["inference"]}
    d = jobdir()
    if d is not None:
        try:
            out["job_peak_worlds"] = int(json.load(open(os.path.join(d, "worlds.json"))).get("peak", 0))
        except (OSError, ValueError):
            pass
    return out


def flush_stats() -> None:
    """Write this process's peaks into the job directory (the supervisor folds them into the usage record)."""
    d = jobdir()
    if d is None:
        return
    try:
        os.makedirs(os.path.join(d, "proc"), exist_ok=True)
        p = os.path.join(d, "proc", f"{os.getpid()}.json")
        with open(p + ".tmp", "w") as fh:
            json.dump(stats(), fh)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


if os.environ.get(ENV_JOBDIR):
    import atexit
    atexit.register(flush_stats)
