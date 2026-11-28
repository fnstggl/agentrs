"""Shared setup for the G2-learn verification probes (lens ``g2learn``). Run every probe from /workspace/camlive:

    .venv/bin/python cam_live/docs/verification/repro/g2learn/<probe>.py

Probes only read the code under test; they print ``[PASS]`` / ``[FAIL]`` per check (FAIL = a claim refuted or a
defect reproduced), ``[INFO]`` lines for measured values, and exit 1 if any check failed. Work directories go to
$G2_WORK (default: the session scratchpad of G2-learn).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
TESTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "tests"))
for p in (SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
WORK = os.environ.get("G2_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G2LEARN/work")
os.makedirs(WORK, exist_ok=True)

RESULTS = []


def check(name, ok, detail=""):
    """Record one check. ok=True: the claim held; ok=False: refuted / defect reproduced."""
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(ok)


def info(name, value):
    RESULTS.append({"info": name, "value": value})
    print(f"[INFO] {name}: {value}", flush=True)


def finish(argv=None):
    argv = sys.argv if argv is None else argv
    if "--json" in argv:
        path = argv[argv.index("--json") + 1]
        with open(path, "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
    n_fail = sum(1 for r in RESULTS if r.get("ok") is False)
    print(f"\n{sum(1 for r in RESULTS if 'check' in r)} checks, {n_fail} failed")
    return 1 if n_fail else 0


def queued_gpu_lock(min_free_gb=16.0):
    """The shared GPU lock taken as a *queued* (blocking) waiter through the project's ``runtime.gpu_cycle.GpuLease``
    when it exists (other tenants wait that way, so a polling waiter never gets a turn); else ``nn.gpu.gpu_lock``.
    Yields the time it was acquired; the caller keeps the hold short (< 15 min)."""
    import contextlib as _cl
    import time as _t
    try:
        from cam_live.runtime.gpu_cycle import GpuLease
    except ImportError:
        from cam_live.nn.gpu import gpu_lock
        return gpu_lock(timeout_s=7200.0, min_free_gb=min_free_gb, poll_s=0.2)

    @_cl.contextmanager
    def _cm():
        lease = GpuLease(hold_s=0.0)
        lease.acquire()
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free < min_free_gb:
                raise RuntimeError(f"only {free:.1f} GB GPU memory free, need {min_free_gb}")
            yield _t.time()
        finally:
            lease.release()
    return _cm()
