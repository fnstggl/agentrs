"""Machine-wide resource governor: admit a command only when its budget is free, then enforce that budget on it.

MASTER_SPEC 13.7 asks that the simulation respect explicit limits on CPU workers, RAM, GPU memory, GPU time, disk
usage, inference concurrency and active world count, on a machine shared with other tenants. This module is the
scheduler that does it (SP-13.7). Each heavy command declares what it needs; it waits in a FIFO queue until the
shared budget can hold it, runs inside that grant, and is stopped if it goes beyond it:

    python -m crucible.scale.slots --cpu 2 --ram-mb 4000 --label "M: tests" -- python -m pytest -q tests/test_x.py
    python -m crucible.scale.slots --cpu 1 --gpu-mb 2000 --gpu-s 600 --inference 4 --worlds 8 \\
        --disk-mb 500 --disk-dir /path/to/store -- python bench.py
    python -m crucible.scale.slots --status          # budget, held slots, queue
    python -m crucible.scale.slots --hardware        # the inspected hardware

| dimension | grant | enforcement |
|---|---|---|
| CPU workers | `--cpu N` | the job is pinned (`sched_setaffinity`) to N CPUs chosen disjoint from other governed jobs; thread pools are capped by environment (OMP, BLAS, MKL, NumExpr, Numba, Rayon); the watchdog re-pins every task of every process of the job each sample, so a process that widens its own affinity is put back; the CPU rate is measured per window |
| RAM | `--ram-mb M` (undeclared: the budget's `default_ram_mb` ceiling) | the watchdog sums the proportional set size (PSS) of every process of the job and kills the job when the sum exceeds the grant |
| GPU memory | `--gpu-mb M` | `--gpu-mb 0` hides the GPU (`CUDA_VISIBLE_DEVICES=""`; a job process that holds a GPU context anyway is killed). Otherwise every Python process of the job caps torch's allocator at the grant when it initialises CUDA (interpreter hook `_slotsite/sitecustomize.py`) and reports its usage; the watchdog kills the job when the sum over its processes exceeds the grant |
| GPU time | `--gpu-s S` (default 4 h when GPU memory is granted) | seconds each job process holds a CUDA context, summed; the job is killed beyond the grant; at most `gpu_jobs` GPU jobs run at once |
| disk | `--disk-mb M` over `--disk-dir D` (repeatable) and the job's scratch directory | growth of those directories (allocated blocks) is measured and the job is killed beyond the grant; no single file may exceed the grant (`RLIMIT_FSIZE`); admission requires that much free disk above a floor |
| inference concurrency | `--inference N` | N token files per job; `crucible.scale.governor.inference()` (used by `substrate.inference.recorded`) holds one per call, job-wide across processes |
| active worlds | `--worlds N` | every `World` registers on construction (`governor.register_world`); job-wide count; construction beyond N raises `ResourceLimitExceeded` |

Admission is FIFO: a request is admitted only when it is at the head of the queue and fits, so small requests cannot
starve a large one. The budget is stored with the ledger (`budget.json`, created from the inspected hardware the first
time; widen it only with `--set-budget`, which keeps a history), so a caller cannot enlarge it through its own
environment. Requests larger than the budget are clamped to it (and the record says so). Liveness reads the process
state, so a zombie holds nothing. The supervisor is a child subreaper: descendants that detach from the command (a
background grandchild) stay in the job, are accounted, and are stopped `--orphan-grace` seconds after the command
itself exited, so a job cannot leave work running outside its slot. The governor signals only processes of the job it
started (its process tree, checked immediately before each signal), never any other process.

A command inside a governed job that calls the scheduler again (a benchmark harness launching runs) gets a nested
ledger whose budget is the parent's grant: sub-jobs are scheduled FIFO inside it and pinned to subsets of the
parent's CPUs.

Every job appends one record to `usage.jsonl`: what it was granted, what it used (CPU seconds and peak window rate,
peak summed PSS, peak GPU memory, GPU seconds, disk growth, bytes written, peak worlds and inference concurrency),
every violation with the action taken, and every signal sent. `CRUCIBLE_SLOTS_DIR` (default `/tmp/crucible-slots`)
selects the ledger; the ledger, queue and budget files are guarded by an exclusive `flock`, so the scheduler is correct
across processes and across worktrees that share the directory. Ledger entries keep the fields of the first version
of this module (pid, cpu, gpu_mb, label, since), so older copies that share the directory still see what is held.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields

from . import governor as G

MB = 1 << 20
DIMENSIONS = ("cpu", "ram", "gpu_mem", "gpu_time", "disk", "inference", "worlds")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS",
               "RAYON_NUM_THREADS", "POLARS_MAX_THREADS")
HOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_slotsite")
DEFAULT_GPU_S = 4 * 3600
UNDECLARED_RAM_PER_CPU_MB = 2048         # what a job that declares no RAM counts against the RAM budget at admission
CPU_TOLERANCE = 1.10                     # a window may use up to 10% (+0.2 CPU) above the grant (sampling jitter)
CPU_STRIKES = 3                          # consecutive windows over the grant, or re-pins of one task, before a kill
GPU_TOOLS = ("nvidia-smi",)              # device queries, not CUDA contexts
PR_SET_CHILD_SUBREAPER, PR_GET_CHILD_SUBREAPER = 36, 37


# ====================================================================== hardware
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


def cgroup_mem_mb() -> int:
    """Memory this container may use (cgroup v1 or v2 limit, else physical memory)."""
    for p in ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.max"):
        try:
            v = open(p).read().strip()
            if v != "max" and int(v) < (1 << 60):
                return int(v) // MB
        except (OSError, ValueError):
            pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // MB)


def gpu_devices() -> list[dict]:
    """[{index, name, total_mb, used_mb, util}] of the visible GPUs (nvidia-smi); [] without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devs = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            try:
                devs.append({"index": int(p[0]), "name": p[1], "total_mb": int(p[2]), "used_mb": int(p[3]),
                             "util": float(p[4]) if p[4] not in ("[N/A]", "") else None})
            except ValueError:
                pass
    return devs


def _topology() -> dict[int, tuple[int, int]]:
    """cpu -> (NUMA node, thread rank in its core: 0 for the first hardware thread, 1 for a hyperthread sibling)."""
    out = {}
    for c in range(os.cpu_count() or 1):
        node = 0
        try:
            node = int(next(n[4:] for n in os.listdir(f"/sys/devices/system/cpu/cpu{c}") if n.startswith("node")))
        except (OSError, StopIteration, ValueError):
            pass
        rank = 0
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
            rank = 0 if int(sib.replace("-", ",").split(",")[0]) == c else 1
        except (OSError, ValueError):
            pass
        out[c] = (node, rank)
    return out


def hardware() -> dict:
    """What this machine offers (MASTER_SPEC 13.7 "inspect available hardware")."""
    try:
        du = shutil.disk_usage(slots_root())
        disk = {"free_mb": du.free // MB, "total_mb": du.total // MB}
    except OSError:
        disk = {}
    topo = _topology()
    return {"cgroup_cpus": cgroup_cpus(), "logical_cpus": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
            "numa_nodes": len({n for n, _ in topo.values()}), "mem_mb": cgroup_mem_mb(), "gpus": gpu_devices(),
            "disk": disk}


# ====================================================================== budget
@dataclass(frozen=True)
class Budget:
    cpus: int
    gpu_mb: int
    jobs: int
    ram_mb: int | None = None
    disk_mb: int | None = None
    inference: int | None = None
    worlds: int | None = None
    gpu_jobs: int | None = None
    default_ram_mb: int | None = None      # RAM ceiling enforced on jobs that declare none
    disk_floor_mb: int = 1024              # free disk that admission keeps untouched

    @staticmethod
    def from_env() -> "Budget":
        """The budget a new ledger starts with: environment overrides, else the inspected hardware."""
        e = os.environ.get
        cpus = int(e("CRUCIBLE_SLOT_CPUS", max(1, math.floor(cgroup_cpus()) - 2)))
        mem = cgroup_mem_mb()
        return Budget(cpus=cpus, gpu_mb=int(e("CRUCIBLE_SLOT_GPU_MB_BUDGET", 24000)), jobs=int(e("CRUCIBLE_SLOT_JOBS", 8)),
                      ram_mb=int(e("CRUCIBLE_SLOT_RAM_MB_BUDGET", int(mem * 0.75))),
                      disk_mb=int(e("CRUCIBLE_SLOT_DISK_MB_BUDGET", 40_000)),
                      inference=int(e("CRUCIBLE_SLOT_INFERENCE_BUDGET", 16)),
                      worlds=int(e("CRUCIBLE_SLOT_WORLDS_BUDGET", 512)), gpu_jobs=int(e("CRUCIBLE_SLOT_GPU_JOBS", 4)),
                      default_ram_mb=int(e("CRUCIBLE_SLOT_DEFAULT_RAM_MB", min(49152, int(mem * 0.5)))))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Budget":
        names = {f.name for f in fields(Budget)}
        return Budget(**{k: v for k, v in d.items() if k in names})


def _min_opt(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _tighter(stored: Budget, given: Budget | None) -> Budget:
    """A caller may narrow the stored budget, never widen it."""
    if given is None:
        return stored
    return Budget(**{f.name: (_min_opt(getattr(stored, f.name), getattr(given, f.name)) if f.name != "disk_floor_mb"
                              else max(stored.disk_floor_mb, given.disk_floor_mb)) for f in fields(Budget)})


# ====================================================================== ledger location
def slots_root() -> str:
    d = os.environ.get("CRUCIBLE_SLOTS_DIR", "/tmp/crucible-slots")
    os.makedirs(d, exist_ok=True)
    return d


def _nested() -> str | None:
    """The parent's job directory when this process runs inside a governed job and schedules against it."""
    jd = os.environ.get(G.ENV_JOBDIR)
    if not jd or not os.path.isdir(jd):
        return None
    root = os.environ.get("CRUCIBLE_SLOTS_DIR")
    if root and os.path.realpath(root) != os.path.realpath(os.environ.get("CRUCIBLE_SLOT_ROOT", "")):
        return None                       # an explicitly different ledger (e.g. a private test ledger)
    return jd


def slots_dir() -> str:
    """The ledger this process schedules against: the nested ledger inside a governed job, else the root."""
    jd = _nested()
    if jd is not None:
        d = os.path.join(jd, "sub")
        os.makedirs(d, exist_ok=True)
        return d
    return slots_root()


def _parent_budget() -> Budget:
    g = G.grants()
    return Budget(cpus=int(g["cpu"] or len(os.sched_getaffinity(0))), gpu_mb=int(g["gpu_mb"] or 0), jobs=64,
                  ram_mb=g["ram_mb"], disk_mb=G._int_env("CRUCIBLE_SLOT_DISK_MB"), inference=g["inference"],
                  worlds=g["worlds"], gpu_jobs=None, default_ram_mb=g["ram_mb"], disk_floor_mb=0)


def stored_budget(d: str | None = None, default: Budget | None = None) -> Budget:
    """The ledger's budget, created on first use (from `default`, the parent grant when nested, else hardware)."""
    d = d or slots_dir()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "budget.json")
    try:
        return Budget.from_dict(json.load(open(p))["budget"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if default is None:
        nested = _nested() is not None and os.path.realpath(d) == os.path.realpath(os.path.join(_nested(), "sub"))
        default = _parent_budget() if nested else Budget.from_env()
    hw = hardware()
    with open(os.path.join(d, "lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not os.path.exists(p):
                rec = {"budget": default.to_dict(), "created": time.time(), "by_pid": os.getpid(), "hardware": hw,
                       "history": []}
                with open(p + ".tmp", "w") as fh:
                    json.dump(rec, fh, indent=1)
                os.replace(p + ".tmp", p)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return Budget.from_dict(json.load(open(p))["budget"])


def set_budget(changes: dict, d: str | None = None, reason: str = "") -> Budget:
    """Change the stored budget (the only way to widen it); the change is kept in the budget file's history."""
    d = d or slots_dir()
    cur = stored_budget(d)
    new = Budget.from_dict({**cur.to_dict(), **changes})
    p = os.path.join(d, "budget.json")
    with open(os.path.join(d, "lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            rec = json.load(open(p))
            rec["history"].append({"t": time.time(), "from": rec["budget"], "to": new.to_dict(), "reason": reason,
                                   "by_pid": os.getpid()})
            rec["budget"] = new.to_dict()
            with open(p + ".tmp", "w") as fh:
                json.dump(rec, fh, indent=1)
            os.replace(p + ".tmp", p)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return new


# ====================================================================== processes
def _stat(pid: int) -> list[str] | None:
    """Fields of /proc/<pid>/stat after the command name: [0] state, [1] ppid, [3] session, [11] utime,
    [12] stime, [19] starttime, [21] rss pages. None if the process does not exist."""
    try:
        with open(f"/proc/{int(pid)}/stat") as fh:
            s = fh.read()
    except (OSError, ValueError):
        return None
    return s[s.rindex(")") + 2:].split()


def _alive(pid: int) -> bool:
    """True while the process exists and is not a zombie (a zombie holds no resources and no slot)."""
    st = _stat(pid)
    if st is None:
        try:
            os.kill(int(pid), 0)          # /proc unreadable: fall back to existence
            return True
        except (ProcessLookupError, ValueError, OverflowError):
            return False
        except PermissionError:
            return True
    return st[0] not in "ZX"


def _children(pid: int) -> list[int]:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as fh:
                    out.extend(int(x) for x in fh.read().split())
            except OSError:
                pass
    except OSError:
        pass
    return out


def _pss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    st = _stat(pid)
    return int(st[21]) * os.sysconf("SC_PAGE_SIZE") / MB if st else 0.0


def _write_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/io") as fh:
            for line in fh:
                if line.startswith("write_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _gpu_context(pid: int) -> bool:
    """True when the process holds a GPU device (/dev/nvidia<N>) open and is not a device query tool."""
    try:
        with open(f"/proc/{pid}/comm") as fh:
            if fh.read().strip() in GPU_TOOLS:
                return False
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                t = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if t.startswith("/dev/nvidia") and t[len("/dev/nvidia"):].isdigit():
                return True
    except OSError:
        pass
    return False


def _dir_mb(path: str) -> float:
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.lstat(os.path.join(dp, f)).st_blocks * 512
            except OSError:
                pass
    return total / MB


def gpu_census() -> dict:
    """Processes of this container holding a GPU context, split by whether each runs inside a governed job that was
    granted GPU memory (SP-1.2). Processes of other tenants on the host are not visible from here (nvidia-smi reports
    host pids); their number is `foreign_gpu_apps` = device compute apps minus the ones found here."""
    inside, outside = [], []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid() or not _gpu_context(pid):
            continue
        env = {}
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            env = dict(x.decode(errors="replace").split("=", 1) for x in raw if b"=" in x)
        except OSError:
            pass
        rec = {"pid": pid, "job": env.get(G.ENV_JOB), "gpu_grant_mb": env.get(G.ENV_GPU_MB)}
        try:
            rec["cmd"] = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:160]
        except OSError:
            pass
        try:
            governed = bool(env.get(G.ENV_JOB)) and int(float(env.get(G.ENV_GPU_MB) or 0)) > 0
        except ValueError:
            governed = False
        (inside if governed else outside).append(rec)
    apps = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        apps = [ln for ln in out.strip().splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"governed": inside, "ungoverned": outside, "device_compute_apps": len(apps),
            "foreign_gpu_apps": max(0, len(apps) - len(inside) - len(outside)), "t": time.time()}


# ====================================================================== ledger
@contextmanager
def _ledger(d: str | None = None):
    d = d or slots_dir()
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        path, qpath = os.path.join(d, "ledger.json"), os.path.join(d, "queue.json")
        try:
            with open(path) as fh:
                held = json.load(fh)
        except (OSError, ValueError):
            held = []
        try:
            with open(qpath) as fh:
                queue = json.load(fh)
        except (OSError, ValueError):
            queue = []
        held = [h for h in held if _alive(int(h["pid"]))]          # reclaim entries of dead or zombie holders
        queue = [q for q in queue if _alive(int(q["pid"]))]
        box = {"held": held, "queue": queue}
        try:
            yield box
        finally:
            for p, v in ((path, box["held"]), (qpath, box["queue"])):
                tmp = f"{p}.{os.getpid()}.tmp"
                with open(tmp, "w") as fh:
                    json.dump(v, fh)
                os.replace(tmp, p)
            fcntl.flock(lk, fcntl.LOCK_UN)


def _ram_of(h: dict) -> int:
    r = h.get("ram_mb")
    return int(r) if r is not None else UNDECLARED_RAM_PER_CPU_MB * max(1, int(h.get("cpu") or 1))


def fits(held: list[dict], cpu: int, gpu_mb: int, budget: Budget, *, ram_mb: int | None = None, disk_mb: int = 0,
         inference: int = 0, worlds: int = 0, free_disk_mb: float | None = None) -> bool:
    """True if a request fits next to what is held, in every budgeted dimension (an empty ledger is no exception:
    oversized requests are clamped before they get here)."""
    if len(held) >= budget.jobs:
        return False
    if sum(int(h["cpu"]) for h in held) + cpu > budget.cpus:
        return False
    if sum(int(h["gpu_mb"]) for h in held) + gpu_mb > budget.gpu_mb:
        return False
    if budget.ram_mb is not None and sum(_ram_of(h) for h in held) + _ram_of({"cpu": cpu, "ram_mb": ram_mb}) \
            > budget.ram_mb:
        return False
    if gpu_mb > 0 and budget.gpu_jobs is not None and sum(1 for h in held if int(h["gpu_mb"]) > 0) >= budget.gpu_jobs:
        return False
    for key, want in (("disk_mb", disk_mb), ("inference", inference), ("worlds", worlds)):
        cap = getattr(budget, key)
        if cap is not None and want and sum(int(h.get(key) or 0) for h in held) + want > cap:
            return False
    if disk_mb and free_disk_mb is not None and free_disk_mb - budget.disk_floor_mb < disk_mb:
        return False
    return True


def clamp(req: dict, budget: Budget) -> tuple[dict, list[str]]:
    """The request cut to the budget in every dimension where it is larger (the cuts are listed)."""
    out, cut = dict(req), []
    for key, cap in (("cpu", budget.cpus), ("gpu_mb", budget.gpu_mb), ("ram_mb", budget.ram_mb),
                     ("disk_mb", budget.disk_mb), ("inference", budget.inference), ("worlds", budget.worlds)):
        v = out.get(key)
        if v is not None and cap is not None and v > cap:
            out[key] = int(cap)
            cut.append(f"{key} {v}->{cap}")
    out["cpu"] = max(1, int(out.get("cpu") or 1))
    return out, cut


def choose_cpus(held: list[dict], n: int, allowed: list[int] | None = None) -> list[int]:
    """n CPUs for a new job: CPUs of `allowed` (default: this process's affinity) not pinned by held jobs, first
    hardware threads first, on one NUMA node when it has room; when too few are free, the least shared ones."""
    allowed = sorted(allowed if allowed is not None else os.sched_getaffinity(0))
    n = max(1, min(int(n), len(allowed)))
    topo = _topology()
    use = {c: 0 for c in allowed}
    for h in held:
        for c in h.get("cpus") or ():
            if c in use:
                use[c] += 1
    free = [c for c in allowed if use[c] == 0]

    def key(c):
        node, rank = topo.get(c, (0, 0))
        return rank, node, c
    if len(free) >= n:
        by_node: dict[int, list[int]] = {}
        for c in sorted(free, key=key):
            by_node.setdefault(topo.get(c, (0, 0))[0], []).append(c)
        first = {nd: [c for c in cs if topo.get(c, (0, 0))[1] == 0] for nd, cs in by_node.items()}
        for nd in sorted(by_node, key=lambda x: (-len(first[x]), x)):
            if len(first[nd]) >= n:
                return sorted(first[nd][:n])
        return sorted(sorted(free, key=key)[:n])
    return sorted(sorted(allowed, key=lambda c: (use[c], *key(c)))[:n])


def acquire(cpu: int, gpu_mb: int, label: str, budget: Budget | None = None, poll_s: float = 2.0,
            timeout_s: float | None = None, pid: int | None = None, *, ram_mb: int | None = None,
            gpu_s: int | None = None, disk_mb: int | None = None, inference: int | None = None,
            worlds: int | None = None, job: str | None = None, d: str | None = None) -> dict:
    """Wait (FIFO) until the request fits, then record it. Returns the ledger entry (pinned CPUs, clamped grant)."""
    d = d or slots_dir()
    eff = _tighter(stored_budget(d, default=budget), budget)
    req, cut = clamp({"cpu": int(cpu), "gpu_mb": max(0, int(gpu_mb)), "ram_mb": ram_mb, "disk_mb": disk_mb,
                      "inference": inference, "worlds": worlds}, eff)
    pid = int(pid or os.getpid())
    ticket = uuid.uuid4().hex[:12]
    t0 = time.time()
    qentry = {"ticket": ticket, "pid": os.getpid(), "since": t0, "label": label[:80], **req}
    with _ledger(d) as box:
        box["queue"].append(qentry)
    try:
        while True:
            free = None
            if req.get("disk_mb"):
                try:
                    free = shutil.disk_usage(d).free / MB
                except OSError:
                    free = 0.0
            with _ledger(d) as box:
                q = box["queue"]
                if not any(x["ticket"] == ticket for x in q):
                    q.append(qentry)
                if q[0]["ticket"] == ticket and fits(box["held"], req["cpu"], req["gpu_mb"], eff, ram_mb=req["ram_mb"],
                                                     disk_mb=int(req["disk_mb"] or 0),
                                                     inference=int(req["inference"] or 0),
                                                     worlds=int(req["worlds"] or 0), free_disk_mb=free):
                    box["queue"] = [x for x in q if x["ticket"] != ticket]
                    entry = {"pid": pid, "cpu": req["cpu"], "gpu_mb": req["gpu_mb"], "label": label,
                             "since": time.time(), "v": 2, "job": job or ticket,
                             "cpus": choose_cpus(box["held"], req["cpu"]), "ram_mb": req["ram_mb"], "gpu_s": gpu_s,
                             "disk_mb": req["disk_mb"], "inference": req["inference"], "worlds": req["worlds"],
                             "clamped": cut, "waited_s": round(time.time() - t0, 3)}
                    box["held"].append(entry)
                    return entry
            if timeout_s is not None and time.time() - t0 > timeout_s:
                raise TimeoutError(f"no slot for cpu={cpu} gpu_mb={gpu_mb} within {timeout_s}s")
            time.sleep(poll_s)
    except BaseException:
        with _ledger(d) as box:
            box["queue"] = [x for x in box["queue"] if x["ticket"] != ticket]
        raise


def release(pid: int, d: str | None = None) -> None:
    with _ledger(d) as box:
        box["held"] = [h for h in box["held"] if int(h["pid"]) != int(pid)]


def release_job(job: str, d: str | None = None) -> None:
    with _ledger(d) as box:
        box["held"] = [h for h in box["held"] if h.get("job") != job]


def status(d: str | None = None) -> dict:
    d = d or slots_dir()
    with _ledger(d) as box:
        held, queue = list(box["held"]), list(box["queue"])
    return {"ledger": d, "budget": stored_budget(d).to_dict(), "held": held, "queue": queue,
            "cpu_used": sum(int(h["cpu"]) for h in held), "gpu_mb_used": sum(int(h["gpu_mb"]) for h in held),
            "ram_mb_reserved": sum(_ram_of(h) for h in held)}


def _log_usage(rec: dict, d: str | None = None) -> None:
    with open(os.path.join(d or slots_dir(), "usage.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def usage(d: str | None = None, since: float | None = None) -> list[dict]:
    """Records of the usage log (those that ended at or after `since`, when given)."""
    out = []
    try:
        with open(os.path.join(d or slots_dir(), "usage.jsonl")) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if since is None or float(r.get("end", 0)) >= since:
                    out.append(r)
    except OSError:
        pass
    return out


def budget_violations(rec: dict) -> list[dict]:
    """Violations of a usage record in the seven budget dimensions (plus timeouts and orphans)."""
    return [v for v in rec.get("violations") or () if v.get("dim") in DIMENSIONS + ("timeout", "orphans")]


# ====================================================================== supervision
def _prctl(op: int, arg: int = 0) -> int:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if op == PR_GET_CHILD_SUBREAPER:
            v = ctypes.c_int(0)
            libc.prctl(op, ctypes.byref(v), 0, 0, 0)
            return v.value
        return libc.prctl(op, arg, 0, 0, 0)
    except (OSError, AttributeError):
        return -1


@dataclass
class Job:
    """What the supervisor knows about a running job."""
    job: str
    grant: dict
    cpus: list[int]
    jobdir: str
    disk_dirs: list[str]
    cli: bool
    sample_s: float
    leader: subprocess.Popen | None = None
    sid: int = -1
    known: set = field(default_factory=set)
    violations: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    killed_by: str | None = None
    peak: dict = field(default_factory=lambda: {"rss_mb": 0.0, "cpu": 0.0, "gpu_mb": 0.0, "disk_mb": 0.0,
                                                "procs": 0, "write_mb": 0.0})
    gpu_s: dict = field(default_factory=dict)
    gpu_unmetered: set = field(default_factory=set)
    disk0: dict = field(default_factory=dict)
    cpu_prev: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    repinned: dict = field(default_factory=dict)
    over: int = 0
    samples: int = 0


def _tree(job: Job) -> list[int]:
    """Live processes of the job: the command's descendants plus everything reparented to the supervisor, which is
    a child subreaper running nothing but this job (so all its children belong to the job)."""
    roots = []
    if job.leader is not None and _alive(job.leader.pid):
        roots.append(job.leader.pid)
    for c in _children(os.getpid()):
        st = _stat(c)
        if st and st[0] not in "ZX" and (job.cli or int(st[3]) == job.sid or c in job.known):
            roots.append(c)
    seen, stack = set(), list(roots)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        st = _stat(p)
        if st is None or st[0] in "ZX":
            continue
        seen.add(p)
        stack.extend(_children(p))
    seen.discard(os.getpid())
    job.known |= seen
    return sorted(seen)


def _signal_tree(job: Job, sig: int) -> None:
    """Signal the job's processes - only those, re-checked as members right before the signal."""
    for p in _tree(job):
        try:
            os.kill(p, sig)
            job.signals.append([p, int(sig)])
        except (ProcessLookupError, PermissionError):
            pass


def _reap(job: Job) -> None:
    """Reap finished children: every child in CLI mode (the supervisor has no others), else the job's own."""
    if job.cli:
        while True:
            try:
                p, st = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if p == 0:
                return
            if job.leader is not None and p == job.leader.pid and job.leader.returncode is None:
                job.leader.returncode = -os.WTERMSIG(st) if os.WIFSIGNALED(st) else os.WEXITSTATUS(st)
    else:
        if job.leader is not None and job.leader.returncode is None:
            job.leader.poll()
        for c in _children(os.getpid()):
            if c in job.known and (job.leader is None or c != job.leader.pid):
                try:
                    os.waitpid(c, os.WNOHANG)
                except ChildProcessError:
                    pass


def _stop(job: Job, why: str, grace_s: float = 5.0) -> None:
    if job.killed_by is None:
        job.killed_by = why
    _signal_tree(job, signal.SIGTERM)
    t = time.time()
    while time.time() - t < grace_s and _tree(job):
        _reap(job)
        time.sleep(0.05)
    if _tree(job):
        _signal_tree(job, signal.SIGKILL)
    _reap(job)


def _violation(job: Job, dim: str, value: float, limit: float, action: str, **extra) -> None:
    job.violations.append({"dim": dim, "value": round(float(value), 3), "limit": float(limit), "action": action,
                           "t": round(time.time(), 3), **extra})


def _gpu_reports(job: Job) -> dict[int, dict]:
    reports = {}
    rep_dir = os.path.join(job.jobdir, "gpu")
    if os.path.isdir(rep_dir):
        for f in os.listdir(rep_dir):
            if f.endswith(".json"):
                try:
                    r = json.load(open(os.path.join(rep_dir, f)))
                    reports[int(r["pid"])] = r
                except (OSError, ValueError, KeyError):
                    pass
    return reports


def _sample(job: Job, now: float, dt: float) -> None:
    pids = _tree(job)
    job.samples += 1
    job.peak["procs"] = max(job.peak["procs"], len(pids))
    g = job.grant
    allowed = set(job.cpus)
    tick = os.sysconf("SC_CLK_TCK")
    cpu_now, rss = {}, 0.0
    for p in pids:
        st = _stat(p)
        if st is None:
            continue
        k = (p, st[19])
        cpu_now[k] = (int(st[11]) + int(st[12])) / tick
        rss += _pss_mb(p)
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            tids = []
        for tid in tids:                              # re-pin every task that widened its affinity
            try:
                cur = set(os.sched_getaffinity(int(tid)))
                if not cur <= allowed:
                    os.sched_setaffinity(int(tid), allowed)
                    n = job.repinned[int(tid)] = job.repinned.get(int(tid), 0) + 1
                    _violation(job, "cpu", len(cur), len(allowed), "repinned", tid=int(tid))
                    if n >= CPU_STRIKES:
                        _violation(job, "cpu", n, CPU_STRIKES, "killed", why="task keeps widening its affinity")
                        _stop(job, "cpu")
                        return
            except OSError:
                pass
        wb = _write_bytes(p)
        if wb:
            job.writes[k] = wb
    if dt > 0:
        rate = sum(max(0.0, v - job.cpu_prev.get(k, 0.0)) for k, v in cpu_now.items()) / dt
        job.peak["cpu"] = max(job.peak["cpu"], rate)
        if rate > g["cpu"] * CPU_TOLERANCE + 0.2 and job.samples > 1:
            job.over += 1
            _violation(job, "cpu", rate, g["cpu"], "killed" if job.over >= CPU_STRIKES else "observed")
            if job.over >= CPU_STRIKES:
                _stop(job, "cpu")
                return
        else:
            job.over = 0
    job.cpu_prev = cpu_now
    job.peak["rss_mb"] = max(job.peak["rss_mb"], rss)
    job.peak["write_mb"] = max(job.peak["write_mb"], sum(job.writes.values()) / MB)
    ram_limit = g["ram_mb"] if g.get("ram_mb") is not None else g.get("default_ram_mb")
    if ram_limit is not None and rss > ram_limit:
        _violation(job, "ram", rss, ram_limit, "killed")
        _stop(job, "ram")
        return
    # GPU memory and GPU time from the interpreter hook's reports; GPU contexts without a report
    reports = _gpu_reports(job)
    live = set(pids)
    gpu_mb = 0.0
    for p, r in reports.items():
        if r.get("t0"):
            t_end = now if (p in live and not r.get("exited")) else float(r.get("t", now))
            job.gpu_s[p] = max(job.gpu_s.get(p, 0.0), t_end - float(r["t0"]))
        if p in live and not r.get("exited"):
            gpu_mb += max(float(r.get("reserved_mb") or 0), float(r.get("max_reserved_mb") or 0))
    if job.samples % 3 == 1:
        for p in pids:
            if p not in reports and p not in job.gpu_unmetered and _gpu_context(p):
                job.gpu_unmetered.add(p)
                if g["gpu_mb"] <= 0:
                    _violation(job, "gpu_mem", 1, 0, "killed", pid=p, why="GPU context in a job granted no GPU")
                    _stop(job, "gpu_mem")
                    return
    job.peak["gpu_mb"] = max(job.peak["gpu_mb"], gpu_mb)
    if g["gpu_mb"] > 0 and gpu_mb > g["gpu_mb"]:
        _violation(job, "gpu_mem", gpu_mb, g["gpu_mb"], "killed")
        _stop(job, "gpu_mem")
        return
    gs = sum(job.gpu_s.values())
    if g.get("gpu_s") is not None and gs > g["gpu_s"]:
        _violation(job, "gpu_time", gs, g["gpu_s"], "killed")
        _stop(job, "gpu_time")
        return
    # disk: growth of the declared directories (every ~2 s)
    every = max(1, int(round(2.0 / max(job.sample_s, 1e-3))))
    if job.disk_dirs and job.samples % every == 1 % every:
        grown = sum(max(0.0, _dir_mb(dd) - job.disk0.get(dd, 0.0)) for dd in job.disk_dirs if os.path.isdir(dd))
        job.peak["disk_mb"] = max(job.peak["disk_mb"], grown)
        if g.get("disk_mb") is not None and grown > g["disk_mb"]:
            _violation(job, "disk", grown, g["disk_mb"], "killed")
            _stop(job, "disk")
            return


def _proc_stats(jobdir: str) -> dict:
    out = {"peak_worlds": 0, "peak_inference": 0, "worlds_refused": 0}
    try:
        w = json.load(open(os.path.join(jobdir, "worlds.json")))
        out["peak_worlds"], out["worlds_refused"] = int(w.get("peak", 0)), int(w.get("refused", 0))
    except (OSError, ValueError):
        pass
    pdir = os.path.join(jobdir, "proc")
    if os.path.isdir(pdir):
        for f in os.listdir(pdir):
            try:
                r = json.load(open(os.path.join(pdir, f)))
            except (OSError, ValueError):
                continue
            out["peak_worlds"] = max(out["peak_worlds"], int(r.get("job_peak_worlds", r.get("peak_worlds", 0))))
            out["peak_inference"] = max(out["peak_inference"], int(r.get("peak_inference", 0)))
    return out


def run_job(cmd: list[str], cpu: int = 1, gpu_mb: int = 0, label: str = "", budget: Budget | None = None,
            timeout_s: float | None = None, *, ram_mb: int | None = None, gpu_s: int | None = None,
            disk_mb: int | None = None, disk_dirs=(), inference: int | None = None, worlds: int | None = None,
            sample_s: float = 1.0, orphan_grace_s: float = 10.0, env: dict | None = None, cwd: str | None = None,
            poll_s: float = 2.0, cli: bool = False) -> dict:
    """Run `cmd` inside a governed slot; returns its usage record (`rc`: the command's exit status, negative for a
    signal). The job is supervised by a dedicated process (this module's CLI, a child subreaper with no other
    children), so every process the job leaves behind is attributed to it."""
    if not cli:
        return _run_via_cli(cmd, cpu, gpu_mb, label, budget, timeout_s, ram_mb=ram_mb, gpu_s=gpu_s, disk_mb=disk_mb,
                            disk_dirs=disk_dirs, inference=inference, worlds=worlds, sample_s=sample_s,
                            orphan_grace_s=orphan_grace_s, env=env, cwd=cwd, poll_s=poll_s)
    d = slots_dir()
    nested = _nested() is not None
    job_id = f"{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    if gpu_mb > 0 and gpu_s is None:
        gpu_s = DEFAULT_GPU_S
    waited0 = time.time()
    entry = acquire(cpu, gpu_mb, label or " ".join(cmd)[:120], budget, poll_s=poll_s, ram_mb=ram_mb, gpu_s=gpu_s,
                    disk_mb=disk_mb, inference=inference, worlds=worlds, job=job_id, d=d)
    waited = time.time() - waited0
    eff = _tighter(stored_budget(d), budget)
    grant = {"cpu": entry["cpu"], "gpu_mb": entry["gpu_mb"], "ram_mb": entry["ram_mb"], "gpu_s": gpu_s,
             "disk_mb": entry["disk_mb"], "inference": entry["inference"], "worlds": entry["worlds"],
             "default_ram_mb": eff.default_ram_mb}
    jobdir = os.path.join(d, "jobs", job_id)
    os.makedirs(jobdir, exist_ok=True)
    scratch = os.path.join(jobdir, "scratch")
    dirs = [os.path.abspath(x) for x in disk_dirs]
    if disk_mb is not None:
        os.makedirs(scratch, exist_ok=True)
        dirs.append(scratch)
    job = Job(job=job_id, grant=grant, cpus=list(entry["cpus"]), jobdir=jobdir, disk_dirs=dirs, cli=cli,
              sample_s=float(sample_s))
    for dd in dirs:
        job.disk0[dd] = _dir_mb(dd) if os.path.isdir(dd) else 0.0
    e = dict(os.environ if env is None else env)
    for v in THREAD_VARS:
        e[v] = str(max(1, entry["cpu"]))
    e["TOKENIZERS_PARALLELISM"] = "false"
    if entry["gpu_mb"] <= 0:
        e["CUDA_VISIBLE_DEVICES"] = ""
    else:
        e["PYTHONPATH"] = HOOK_DIR + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    chain = [x for x in (e.get("CRUCIBLE_SLOT_JOBDIR_CHAIN") or e.get(G.ENV_JOBDIR) or "").split(os.pathsep) if x]
    e.update({G.ENV_JOB: job_id, G.ENV_JOBDIR: jobdir, G.ENV_CPU: str(entry["cpu"]), G.ENV_GPU_MB: str(entry["gpu_mb"]),
              "CRUCIBLE_SLOT_CPUSET": ",".join(map(str, job.cpus)), "CRUCIBLE_SLOT_ROOT": d,
              "CRUCIBLE_SLOT_JOBDIR_CHAIN": os.pathsep.join(chain + [jobdir])})   # GPU reports reach enclosing jobs
    for k, v in ((G.ENV_RAM, entry["ram_mb"]), (G.ENV_GPU_S, gpu_s if entry["gpu_mb"] > 0 else None),
                 (G.ENV_INFER, entry["inference"]), (G.ENV_WORLDS, entry["worlds"]),
                 ("CRUCIBLE_SLOT_DISK_MB", entry["disk_mb"]),
                 ("CRUCIBLE_SLOT_SCRATCH", scratch if disk_mb is not None else None)):
        if v is not None:
            e[k] = str(int(v)) if not isinstance(v, str) else v
        else:
            e.pop(k, None)
    prev_reaper = _prctl(PR_GET_CHILD_SUBREAPER)
    _prctl(PR_SET_CHILD_SUBREAPER, 1)
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    rc = None
    try:
        try:
            job.leader = subprocess.Popen(cmd, env=e, cwd=cwd, start_new_session=True)
        except OSError as exc:
            rc = 127
            job.killed_by = f"spawn failed: {exc}"
        if job.leader is not None:
            job.sid = job.leader.pid
            job.known.add(job.leader.pid)
            try:
                os.sched_setaffinity(job.leader.pid, job.cpus)
            except OSError:
                pass
            if entry["disk_mb"] is not None:
                try:
                    lim = int(entry["disk_mb"]) * MB
                    resource.prlimit(job.leader.pid, resource.RLIMIT_FSIZE, (lim, lim))
                except (OSError, ValueError):
                    pass
            last = time.time()
            leader_done_at = None
            while True:
                if job.leader.returncode is None:
                    try:
                        job.leader.wait(timeout=job.sample_s)
                    except subprocess.TimeoutExpired:
                        pass
                else:
                    time.sleep(job.sample_s)
                now = time.time()
                _reap(job)
                if job.killed_by is None:
                    _sample(job, now, now - last)
                last = now
                done = job.leader.returncode is not None
                if done and leader_done_at is None:
                    leader_done_at = now
                if timeout_s is not None and now - t0 > timeout_s and job.killed_by is None:
                    _violation(job, "timeout", now - t0, timeout_s, "killed")
                    _stop(job, "timeout")
                members = _tree(job)
                if done and not members:
                    break
                if done and members and now - leader_done_at > orphan_grace_s and job.killed_by is None:
                    _violation(job, "orphans", len(members), 0, "killed")
                    _stop(job, "orphans")
            rc = job.leader.returncode
    except BaseException:
        if job.leader is not None:
            _stop(job, "supervisor interrupted", grace_s=2.0)
        raise
    finally:
        _reap(job)
        if not cli:
            _prctl(PR_SET_CHILD_SUBREAPER, prev_reaper if prev_reaper >= 0 else 0)
        release_job(job_id, d)
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        ps = _proc_stats(jobdir)
        for dim in ("worlds", "inference"):
            lim, got = grant.get(dim), ps.get(f"peak_{dim}", 0)
            if lim is not None and got > lim:
                _violation(job, dim, got, lim, "recorded")
        wall = time.time() - t0
        cpu_s = (ru.ru_utime + ru.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
        rec = {"v": 2, "job": job_id, "label": label, "cmd": list(cmd)[:8], "cpu": entry["cpu"],
               "gpu_mb": entry["gpu_mb"], "ram_mb": entry["ram_mb"], "gpu_s_grant": gpu_s, "disk_mb": entry["disk_mb"],
               "inference": entry["inference"], "worlds": entry["worlds"], "cpus": job.cpus,
               "clamped": entry.get("clamped", []), "nested": nested, "ledger": d,
               "declared": [k for k in ("ram_mb", "disk_mb", "inference", "worlds") if entry.get(k) is not None]
               + (["gpu_mb", "gpu_s"] if entry["gpu_mb"] > 0 else []),
               "rc": rc, "waited_s": round(waited, 2), "wall_s": round(wall, 2), "cpu_s": round(cpu_s, 2),
               "mean_cpu": round(cpu_s / wall, 3) if wall > 0 else 0.0, "peak_cpu": round(job.peak["cpu"], 3),
               "peak_rss_mb": round(job.peak["rss_mb"], 1), "max_child_rss_mb": round(ru.ru_maxrss / 1024, 1),
               "peak_gpu_mb": round(job.peak["gpu_mb"], 1), "gpu_s": round(sum(job.gpu_s.values()), 2),
               "gpu_unmetered_procs": len(job.gpu_unmetered), "disk_peak_mb": round(job.peak["disk_mb"], 2),
               "write_mb": round(job.peak["write_mb"], 2), "peak_procs": job.peak["procs"],
               "peak_worlds": ps["peak_worlds"], "worlds_refused": ps["worlds_refused"],
               "peak_inference": ps["peak_inference"], "samples": job.samples, "violations": job.violations,
               "killed_by": job.killed_by, "signals": job.signals[:200], "end": time.time()}
        _log_usage(rec, d)
        shutil.rmtree(jobdir, ignore_errors=True)
    return rec


def _run_via_cli(cmd, cpu, gpu_mb, label, budget, timeout_s, *, ram_mb, gpu_s, disk_mb, disk_dirs, inference,
                 worlds, sample_s, orphan_grace_s, env, cwd, poll_s) -> dict:
    import tempfile
    e = dict(os.environ if env is None else env)
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    e["PYTHONPATH"] = src + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    fd, out = tempfile.mkstemp(prefix="slot-record-", suffix=".json")
    os.close(fd)
    argv = [sys.executable, "-m", "crucible.scale.slots", "--record-out", out, "--cpu", str(int(cpu)),
            "--gpu-mb", str(int(gpu_mb)), "--label", label or " ".join(map(str, cmd))[:120], "--sample", str(sample_s),
            "--poll", str(poll_s), "--orphan-grace", str(orphan_grace_s)]
    for flag, v in (("--ram-mb", ram_mb), ("--gpu-s", gpu_s), ("--disk-mb", disk_mb), ("--inference", inference),
                    ("--worlds", worlds), ("--timeout", timeout_s)):
        if v is not None:
            argv += [flag, str(v)]
    for dd in disk_dirs:
        argv += ["--disk-dir", os.path.abspath(dd)]
    if budget is not None:
        argv += ["--budget-json", json.dumps(budget.to_dict())]
    try:
        p = subprocess.run([*argv, "--", *cmd], env=e, cwd=cwd)
        try:
            with open(out) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {"rc": p.returncode, "killed_by": "supervisor failed", "violations": [], "signals": []}
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def run(cmd: list[str], cpu: int = 1, gpu_mb: int = 0, label: str = "", budget: Budget | None = None,
        timeout_s: float | None = None, **kw) -> int:
    """Run `cmd` inside a governed slot; returns its exit status (`run_job` returns the whole usage record)."""
    return run_job(cmd, cpu, gpu_mb, label, budget, timeout_s, **kw)["rc"]


# ====================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", type=int, default=1, help="CPU workers (the job is pinned to this many CPUs)")
    ap.add_argument("--gpu-mb", type=int, default=0, help="GPU memory; 0 hides the GPU")
    ap.add_argument("--gpu-s", type=int, default=None, help="GPU time: seconds of held CUDA contexts")
    ap.add_argument("--ram-mb", type=int, default=None, help="RAM: PSS summed over the job's processes")
    ap.add_argument("--disk-mb", type=int, default=None, help="disk growth of the --disk-dir directories + scratch")
    ap.add_argument("--disk-dir", action="append", default=[], help="directory counted against --disk-mb")
    ap.add_argument("--inference", type=int, default=None, help="concurrent inference calls of the job")
    ap.add_argument("--worlds", type=int, default=None, help="active worlds of the job")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=None, help="seconds before the command is stopped")
    ap.add_argument("--sample", type=float, default=1.0, help="watchdog sampling interval (s)")
    ap.add_argument("--orphan-grace", type=float, default=10.0,
                    help="seconds detached descendants may outlive the command before they are stopped")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--hardware", action="store_true", help="print the inspected hardware")
    ap.add_argument("--set-budget", action="append", default=[], metavar="KEY=VALUE",
                    help="change the stored budget (e.g. cpus=9); kept in the budget history")
    ap.add_argument("--reason", default="", help="why the budget changes")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between admission attempts")
    ap.add_argument("--budget-json", default=None, help="narrow the stored budget for this job (JSON)")
    ap.add_argument("--record-out", default=None, help="also write the usage record to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.hardware:
        print(json.dumps(hardware(), indent=1))
        return 0
    if a.set_budget:
        ch = {}
        for kv in a.set_budget:
            k, v = kv.split("=", 1)
            ch[k] = None if v.lower() in ("none", "null") else int(v)
        print(json.dumps(set_budget(ch, reason=a.reason).to_dict(), indent=1))
        return 0
    if a.status:
        print(json.dumps(status(), indent=1))
        return 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given (put it after --)")
    def _term(signum, frame):              # stop the job (and only it) when the supervisor is asked to stop
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _term)
    b = Budget.from_dict(json.loads(a.budget_json)) if a.budget_json else None
    rec = run_job(cmd, a.cpu, a.gpu_mb, a.label, b, timeout_s=a.timeout, ram_mb=a.ram_mb, gpu_s=a.gpu_s,
                  disk_mb=a.disk_mb, disk_dirs=a.disk_dir, inference=a.inference, worlds=a.worlds,
                  sample_s=a.sample, orphan_grace_s=a.orphan_grace, poll_s=a.poll, cli=True)
    if a.record_out:
        with open(a.record_out, "w") as fh:
            json.dump(rec, fh)
    if rec["killed_by"]:
        print(f"[crucible.scale.slots] job {rec['job']} stopped: {rec['killed_by']} "
              f"{[v for v in rec['violations'] if v['action'] == 'killed']}", file=sys.stderr)
    rc = rec["rc"]
    return 128 - rc if isinstance(rc, int) and rc < 0 else int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
