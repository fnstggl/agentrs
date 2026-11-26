"""Machine-wide compute slots: run a command only when its CPU, GPU-memory and job budget is free.

Several agents and experiments share one machine. Each heavy command asks for a number of CPU threads and an amount
of GPU memory; it waits until the shared budget can hold it, runs with its thread count and GPU visibility set to
what it asked for, and releases the budget when it exits (even if it is killed: entries of dead processes are
reclaimed). Every run is appended to a usage log, which is how compute use is reported.

    python -m crucible.scale.slots --cpu 2 --gpu-mb 0 --label tests -- python -m pytest -q tests/test_x.py
    python -m crucible.scale.slots --status

The budget comes from the environment: `CRUCIBLE_SLOT_CPUS` (default: the cgroup CPU quota minus 2, at least 1),
`CRUCIBLE_SLOT_GPU_MB` (default 24000), `CRUCIBLE_SLOT_JOBS` (default 8) and `CRUCIBLE_SLOTS_DIR` (default
`/tmp/crucible-slots`). The ledger is a JSON file guarded by an exclusive `flock`, so the scheduler is correct across
processes and across worktrees that share the directory.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass


def cgroup_cpus() -> float:
    """CPUs this container may use: the cgroup quota if one is set, else the visible CPU count."""
    for quota_path, period_path in (("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),):
        try:
            q, p = int(open(quota_path).read()), int(open(period_path).read())
            if q > 0 and p > 0:
                return q / p
        except (OSError, ValueError):
            pass
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return int(q) / int(p)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


@dataclass(frozen=True)
class Budget:
    cpus: int
    gpu_mb: int
    jobs: int

    @staticmethod
    def from_env() -> "Budget":
        cpus = int(os.environ.get("CRUCIBLE_SLOT_CPUS", max(1, math.floor(cgroup_cpus()) - 2)))
        return Budget(cpus=cpus, gpu_mb=int(os.environ.get("CRUCIBLE_SLOT_GPU_MB", 24000)),
                      jobs=int(os.environ.get("CRUCIBLE_SLOT_JOBS", 8)))


def slots_dir() -> str:
    d = os.environ.get("CRUCIBLE_SLOTS_DIR", "/tmp/crucible-slots")
    os.makedirs(d, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def _ledger():
    d = slots_dir()
    with open(os.path.join(d, "lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        path = os.path.join(d, "ledger.json")
        try:
            with open(path) as fh:
                held = json.load(fh)
        except (OSError, ValueError):
            held = []
        held = [h for h in held if _alive(int(h["pid"]))]          # reclaim entries of dead holders
        box = {"held": held}
        try:
            yield box
        finally:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(box["held"], fh)
            os.replace(tmp, path)
            fcntl.flock(lk, fcntl.LOCK_UN)


def fits(held: list[dict], cpu: int, gpu_mb: int, budget: Budget) -> bool:
    """True if a request fits next to what is held. A request larger than the whole budget fits only when the
    machine is otherwise idle, so it can still run (alone) instead of waiting for ever."""
    used_cpu = sum(int(h["cpu"]) for h in held)
    used_gpu = sum(int(h["gpu_mb"]) for h in held)
    if not held:
        return True
    return (len(held) < budget.jobs and used_cpu + cpu <= budget.cpus and used_gpu + gpu_mb <= budget.gpu_mb)


def acquire(cpu: int, gpu_mb: int, label: str, budget: Budget | None = None, poll_s: float = 2.0,
            timeout_s: float | None = None, pid: int | None = None) -> dict:
    """Block until the request fits, then record it. Returns the ledger entry."""
    budget = budget or Budget.from_env()
    pid = int(pid or os.getpid())
    t0 = time.time()
    while True:
        with _ledger() as box:
            if fits(box["held"], cpu, gpu_mb, budget):
                entry = {"pid": pid, "cpu": int(cpu), "gpu_mb": int(gpu_mb), "label": label, "since": time.time()}
                box["held"].append(entry)
                return entry
        if timeout_s is not None and time.time() - t0 > timeout_s:
            raise TimeoutError(f"no slot for cpu={cpu} gpu_mb={gpu_mb} within {timeout_s}s")
        time.sleep(poll_s)


def release(pid: int) -> None:
    with _ledger() as box:
        box["held"] = [h for h in box["held"] if int(h["pid"]) != int(pid)]


def status() -> dict:
    with _ledger() as box:
        held = list(box["held"])
    b = Budget.from_env()
    return {"budget": b.__dict__, "held": held, "cpu_used": sum(h["cpu"] for h in held),
            "gpu_mb_used": sum(h["gpu_mb"] for h in held)}


def _log_usage(rec: dict) -> None:
    with open(os.path.join(slots_dir(), "usage.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run(cmd: list[str], cpu: int = 1, gpu_mb: int = 0, label: str = "", budget: Budget | None = None,
        timeout_s: float | None = None) -> int:
    """Run `cmd` inside a slot. Thread counts follow `cpu`; the GPU is hidden when `gpu_mb` is 0."""
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(max(1, cpu))
    if gpu_mb <= 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["CRUCIBLE_SLOT_GPU_MB"] = str(max(0, int(gpu_mb)))      # the grant, for in-process allocator caps
    waited0 = time.time()
    acquire(cpu, gpu_mb, label or " ".join(cmd)[:120], budget)
    waited = time.time() - waited0
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env)
    with _ledger() as box:                                   # the slot belongs to the child from now on
        for h in box["held"]:
            if int(h["pid"]) == os.getpid():
                h["pid"] = proc.pid
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    finally:
        release(proc.pid)
        release(os.getpid())
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    _log_usage({"label": label, "cmd": cmd[:8], "cpu": cpu, "gpu_mb": gpu_mb, "rc": rc, "waited_s": round(waited, 2),
                "wall_s": round(time.time() - t0, 2), "cpu_s": round(ru.ru_utime + ru.ru_stime, 2),
                "peak_rss_mb": round(ru.ru_maxrss / 1024, 1), "end": time.time()})
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", type=int, default=1)
    ap.add_argument("--gpu-mb", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=None, help="seconds before the command is stopped")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.status:
        print(json.dumps(status(), indent=1))
        return 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given (put it after --)")
    return run(cmd, a.cpu, a.gpu_mb, a.label, timeout_s=a.timeout)


if __name__ == "__main__":
    sys.exit(main())
