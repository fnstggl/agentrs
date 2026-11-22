"""Process-parallel worlds.

One Crucible world is single-threaded by design (the kernel serialises commands). Research needs many independent
worlds at once, so `WorldPool` runs each world inside a worker *process*:

    with WorldPool(4, store_dir="runs/exp1") as pool:
        jobs = [world_job({"family": "tiny", "seed": s}, until_day=6, snapshot_every_days=3) for s in range(16)]
        for res in pool.map(jobs):
            print(res["digest"], res["counts"], res["wall_s"], res["peak_rss_mb"])

Properties (each one is tested in tests/test_scale.py):

* **Declarative jobs.** A job is a picklable dict: a world reference (``family``, ``seed``, ``overrides``,
  ``extra_modules``) or a snapshot to resume, a horizon, agents as ``(handle, "module:callable", kwargs)``
  factories that are resolved *inside* the worker, and an optional ``"module:function"`` summariser. No world,
  policy or model object crosses the process boundary.
* **Isolation.** An exception in a world is that job's result (``ok=False``, with the traceback). A worker that
  dies (segfault, ``os._exit``, OOM kill, timeout) is replaced; the job it was running is reported failed with the
  evidence (exit code or signal, the tail of the worker's stderr). Other jobs are unaffected.
* **Determinism.** A job's digest and event hash depend only on its spec, never on the worker, the number of
  workers or what else runs. (Worlds must use ``params.cost_model = "deterministic"`` for this to hold when they
  execute code: the legacy cost model records measured CPU time.)
* **Checkpoints.** ``snapshot_every_days=k`` writes a snapshot file every k simulated days; ``resume=<file>``
  starts a job from one, in any process. The stepping grid is absolute, so run-to-day-6 has the same event hash
  as run-to-day-3 + snapshot + resume + run-to-day-6.
* **Bounded memory.** Workers are recycled after ``max_jobs_per_worker`` jobs.
* **Clean shutdown.** `close()` and `terminate()` leave no process behind, also after KeyboardInterrupt; workers
  additionally ask the kernel to signal them when the parent dies, so a killed parent leaves no orphans either.

The container this was built in has a hard limit of about 2,800 threads: the pool defaults to at most 8 workers,
pins BLAS/OpenMP to one thread per worker before numpy is imported there, and refuses to start when the pid
budget of the cgroup is nearly used up.

`Supervisor` (the process-management core: one duplex pipe and at most one item in flight per worker) is shared
with `crucible.scale.rollout.RolloutFarm`.

This module imports neither numpy nor the kernel at import time (worker processes import it before the thread
limits are in place); everything heavy is imported inside functions.
"""
from __future__ import annotations

import atexit
import collections
import importlib
import os
import pickle
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import weakref
from typing import Any, Iterable, Iterator

__all__ = ["WorldPool", "world_job", "run_world_job", "Supervisor", "PoolBroken", "resolve", "ensure_registered",
           "pid_budget", "cpu_quota_cores", "default_workers", "DAY", "MAX_WORKERS"]

DAY = 86400
BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
#: Hard cap on workers per pool. Raise it deliberately with the environment variable, never by accident.
MAX_WORKERS = int(os.environ.get("CRUCIBLE_SCALE_MAX_WORKERS", "8"))
_ENV_LOCK = threading.Lock()


class PoolBroken(RuntimeError):
    """Workers cannot be started (they die before becoming ready). Carries the evidence of the last death."""

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


# ====================================================================== small helpers
def resolve(ref: str) -> Any:
    """``"package.module:attr"`` (dots allowed in attr) -> the object. Used for every factory named in a spec."""
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(f"expected 'module:callable', got {ref!r}")
    mod, _, attr = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def cpu_quota_cores() -> float | None:
    """CPU quota of this container's cgroup in cores (v1 or v2), or None when unlimited or unknown. On a shared
    machine this, not `os.cpu_count()`, is what the processes of the container can use together."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            return q / int(f.read()) if q > 0 else None
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, per = f.read().split()[:2]
        return None if q == "max" else int(q) / int(per)
    except (OSError, ValueError, ZeroDivisionError):
        return None


def default_workers() -> int:
    """Workers when the caller does not say: at most 8, at most the cap, at most the cores we may use."""
    quota = cpu_quota_cores()
    cores = os.cpu_count() or 1
    if quota:
        cores = min(cores, max(1, int(quota + 0.5)))
    return max(1, min(MAX_WORKERS, 8, cores))


def pid_budget() -> dict | None:
    """Thread budget of this container's pids cgroup: ``{"current", "max", "free"}``, or None if unlimited or
    unknown. Reads cgroup v1 and v2 locations."""
    for base in ("/sys/fs/cgroup/pids", "/sys/fs/cgroup"):
        try:
            with open(os.path.join(base, "pids.max")) as f:
                mx = f.read().strip()
            with open(os.path.join(base, "pids.current")) as f:
                cur = int(f.read().strip())
        except (OSError, ValueError):
            continue
        if mx == "max":
            return None
        try:
            return {"current": cur, "max": int(mx), "free": int(mx) - cur}
        except ValueError:
            return None
    return None


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """(ppid, start time in clock ticks) of a live process, or None."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return int(rest[1]), int(rest[19])
    except (ValueError, IndexError):
        return None


def _children(pid: int) -> list[int]:
    out: list[int] = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    readable = False
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/children") as f:
                out.extend(int(x) for x in f.read().split())
            readable = True
        except (OSError, ValueError):
            continue
    if not readable:  # kernels without CONFIG_PROC_CHILDREN: scan
        for name in os.listdir("/proc"):
            if name.isdigit():
                st = _proc_stat(int(name))
                if st is not None and st[0] == pid:
                    out.append(int(name))
    return out


def _descendants(pid: int, limit: int = 4096) -> dict[int, int]:
    """Every live descendant of `pid` as {pid: start time}. The start time identifies the process: a recycled pid
    has a different one, so a later kill can never hit an unrelated process."""
    out: dict[int, int] = {}
    todo = [pid]
    while todo and len(out) < limit:
        for c in _children(todo.pop()):
            if c not in out:
                st = _proc_stat(c)
                if st is not None:
                    out[c] = st[1]
                    todo.append(c)
    return out


def _kill_tracked(tracked: dict[int, int]) -> list[int]:
    """SIGKILL the tracked processes that are still the same processes (same start time). Returns those killed."""
    killed = []
    for pid, start in tracked.items():
        st = _proc_stat(pid)
        if st is None or st[1] != start:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except OSError:
            pass
    return killed


def _children_cpu() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _reset_peak_rss() -> bool:
    """Reset this process's peak-RSS counter so that the next reading is the peak of one job, not of the worker's
    life. Linux only (``/proc/self/clear_refs``); returns False when unavailable."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _peak_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _tail(path: str | None, n: int = 4000) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def _err(e: BaseException, stage: str = "") -> dict:
    d = {"type": type(e).__name__, "message": str(e)[:2000],
         "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-6000:]}
    if stage:
        d["stage"] = stage
    return d


# ====================================================================== registration of world mechanics
#: Packages scanned at run time for files that register kernel modules or world families. This is how families
#: added after this file was written (``mech_*``, ``ow_*``) reach workers and benchmarks without any wiring here.
DISCOVER_PACKAGES = ("worldgen", "mechanisms", "usefulness", "economy", "actors", "worlds")
_REGISTERS = re.compile(r"register_module|(?<![A-Za-z_])family\(|mech_family\(")
_REG: dict[str, Any] = {"done": False, "imported": [], "errors": {}}


def ensure_registered(imports: Iterable[str] = (), discover: bool = True) -> dict:
    """Make every world family and kernel module importable in this process known to the registries.

    1. ``crucible.load_all_modules()`` (the 3.0 set, plus whatever the integrator added to it).
    2. Run-time discovery: every file of the packages in `DISCOVER_PACKAGES` whose source registers a kernel
       module or a world family is imported. A file that fails to import is recorded, never fatal, so one broken
       package cannot take down unrelated worlds.
    3. The explicit `imports` (errors here are raised: the caller asked for them).

    Returns ``{"imported": [...], "errors": {module: "Type: message"}}`` (accumulated for this process)."""
    import crucible
    if not _REG["done"]:
        crucible.load_all_modules()
        _REG["done"] = True
    if discover and not _REG.get("discovered") and os.environ.get("CRUCIBLE_SCALE_DISCOVER", "1") != "0":
        _REG["discovered"] = True
        root = os.path.dirname(os.path.abspath(crucible.__file__))
        for pkg in DISCOVER_PACKAGES:
            d = os.path.join(root, pkg)
            if not os.path.isfile(os.path.join(d, "__init__.py")):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".py") or fn == "__init__.py":
                    continue
                name = f"crucible.{pkg}.{fn[:-3]}"
                if name in sys.modules:
                    continue
                try:
                    with open(os.path.join(d, fn), encoding="utf-8") as f:
                        if not _REGISTERS.search(f.read()):
                            continue
                except OSError:
                    continue
                try:
                    importlib.import_module(name)
                    _REG["imported"].append(name)
                except Exception as e:  # noqa: BLE001 - someone else's package; report, do not die
                    _REG["errors"][name] = f"{type(e).__name__}: {str(e)[:300]}"
    for name in imports or ():
        if name not in sys.modules:
            importlib.import_module(name)
            _REG["imported"].append(name)
    return {"imported": list(_REG["imported"]), "errors": dict(_REG["errors"])}


# ====================================================================== worker process
def _on_term(signum, frame):  # noqa: ARG001
    raise SystemExit(128 + signum)


def _set_pdeathsig(sig: int = signal.SIGTERM) -> bool:
    """Ask the kernel to send `sig` to this process when its parent dies (Linux)."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(1, int(sig), 0, 0, 0) == 0  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        return False


def _worker_main(wid: int, conn, factory: str, cfg: dict, log_path: str | None, parent_pid: int | None,
                 blas_threads: int, temp_logs: bool = False) -> None:
    """Entry point of a worker process: build the handler, then serve one item at a time over the pipe."""
    for k in BLAS_ENV:
        os.environ[k] = str(blas_threads)
    _LIVE.clear()                                    # (fork) the parent's supervisors are not ours to stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C goes to the whole terminal group; the parent decides
    signal.signal(signal.SIGTERM, _on_term)          # terminate = unwind, close worlds (and their sandboxes), exit
    _set_pdeathsig()
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(0)                                  # the parent died before we got here
    if log_path:
        try:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)
            sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
            sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
        except OSError:
            pass
    try:
        import faulthandler
        faulthandler.enable()                        # a segfault leaves a Python traceback in the log (evidence)
    except Exception:  # noqa: BLE001
        pass
    mem = cfg.get("mem_limit_mb")
    if mem:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (int(mem) * 2**20, int(mem) * 2**20))
    t0 = time.perf_counter()
    handler = None
    try:
        try:
            handler = resolve(factory)(dict(cfg, worker=wid))
        except Exception as e:  # noqa: BLE001
            conn.send(("fatal", _err(e, "worker_init")))
            return
        conn.send(("ready", {"pid": os.getpid(), "init_s": round(time.perf_counter() - t0, 3),
                             "rss_mb": round(_rss_mb(), 1)}))
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break                                # the parent is gone
            if msg[0] == "stop":
                break
            _, item_id, payload = msg
            try:
                reply = ("done", item_id, True, handler.handle(payload))
            except Exception as e:  # noqa: BLE001 - the item failed; the worker lives on
                reply = ("done", item_id, False, _err(e))
            try:
                conn.send(reply)
            except (pickle.PicklingError, TypeError, AttributeError, ValueError) as e:
                conn.send(("done", item_id, False, _err(e, "result_not_picklable")))
    finally:
        if handler is not None:
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        if temp_logs and log_path and parent_pid is not None and os.getppid() != parent_pid:
            try:                                     # the parent was killed: nobody else will remove our temp log
                os.unlink(log_path)
                os.rmdir(os.path.dirname(log_path))  # succeeds for the last worker to leave
            except OSError:
                pass


class _Worker:
    __slots__ = ("slot", "gen", "proc", "conn", "pid", "state", "item", "t_assign", "deadline", "jobs", "log_path",
                 "tracked", "t_track", "info", "t_spawn", "fatal", "meta", "__weakref__")

    def __init__(self, slot, gen, proc, conn, log_path):
        self.slot, self.gen, self.proc, self.conn, self.log_path = slot, gen, proc, conn, log_path
        self.pid = proc.pid
        self.state = "starting"      # starting | idle | busy | retiring | dead
        self.item: Any = None
        self.t_assign = 0.0
        self.deadline: float | None = None
        self.jobs = 0
        self.tracked: dict[int, int] = {}
        self.t_track = 0.0
        self.info: dict = {}
        self.t_spawn = time.monotonic()
        self.fatal: dict | None = None
        self.meta: dict = {}         # free for the owner (e.g. the parameter version a rollout worker holds)

    def __repr__(self) -> str:
        return f"<worker {self.slot}.{self.gen} pid={self.pid} {self.state}>"


_LIVE: "weakref.WeakSet[Supervisor]" = weakref.WeakSet()


@atexit.register
def _terminate_all() -> None:
    for s in list(_LIVE):
        try:
            s.terminate()
        except Exception:  # noqa: BLE001
            pass


class Supervisor:
    """N worker processes, each serving one item at a time through its own duplex pipe.

    The owner drives it: `assign(worker, item_id, payload)` on an idle worker, then `poll(timeout)` which returns
    events: ``("ready", worker, info)``, ``("done", worker, item_id, ok, payload)`` and
    ``("died", worker, item_id_or_None, evidence)``. A worker that dies is replaced (unless the supervisor is
    closing). There is no shared queue and no lock shared between processes, so a worker killed at any instant
    cannot corrupt the others, and the item it held is known exactly.

    Not thread-safe: all calls must come from one thread at a time. Workers ask the kernel for SIGTERM when the
    thread that started them exits (that is how a killed parent leaves no orphans), so start the supervisor from
    a thread that lives as long as the workers should. A forked copy of a supervisor (in a child process) is
    inert: only the process that created it can stop its workers.
    """

    def __init__(self, n_workers: int | None, factory: str, cfg: dict | None = None, *, start_method: str = "spawn",
                 max_jobs_per_worker: int | None = None, name: str = "crucible-scale", log_dir: str | None = None,
                 blas_threads: int = 1, check_pids: bool = True, term_grace_s: float = 5.0):
        import multiprocessing as mp
        n = default_workers() if n_workers is None else int(n_workers)
        if n < 1:
            raise ValueError("n_workers must be >= 1")
        if n > MAX_WORKERS:
            raise ValueError(f"{n} workers exceed the cap of {MAX_WORKERS} (this container has a hard thread "
                             f"limit); set CRUCIBLE_SCALE_MAX_WORKERS to raise the cap deliberately")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"start_method {start_method!r} not available here ({mp.get_all_start_methods()})")
        resolve(factory)                             # fail in the caller, not in N children
        self.n_workers = n
        self.factory = factory
        self.cfg = dict(cfg or {})
        self.start_method = start_method
        self.max_jobs = max_jobs_per_worker
        self.name = name
        self.blas_threads = int(blas_threads)
        self.term_grace_s = term_grace_s
        self._ctx = mp.get_context(start_method)
        self._owner_pid = os.getpid()
        self._own_log_dir = log_dir is None
        self.log_dir = log_dir or tempfile.mkdtemp(prefix="crucible-scale-")
        os.makedirs(self.log_dir, exist_ok=True)
        self.workers: list[_Worker] = []             # active, one per slot
        self._retiring: list[_Worker] = []
        self._gen = 0
        self._closing = False
        self._closed = False
        self._bad_starts = 0
        self._last_evidence: dict = {}
        self.spawned = 0
        self.recycled = 0
        self.died = 0
        self.orphans_killed = 0
        if check_pids:
            b = pid_budget()
            need = n * 8 + 64
            if b is not None and b["free"] < need:
                raise RuntimeError(f"pid budget nearly exhausted ({b['current']}/{b['max']} threads in use, "
                                   f"{need} needed for {n} workers): refusing to start")
        _LIVE.add(self)
        try:
            for slot in range(n):
                self.workers.append(self._spawn(slot))
        except BaseException:
            self.terminate()
            raise

    # ------------------------------------------------------------------ spawning
    def _spawn(self, slot: int) -> _Worker:
        self._gen += 1
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        log_path = os.path.join(self.log_dir, f"{self.name}-w{slot}-g{self._gen}.log")
        parent_pid = os.getpid() if self.start_method != "forkserver" else None
        proc = self._ctx.Process(target=_worker_main, name=f"{self.name}-{slot}", daemon=True,
                                 args=(slot, child_conn, self.factory, self.cfg, log_path, parent_pid,
                                       self.blas_threads, self._own_log_dir))
        with _ENV_LOCK:                              # the child must see the thread limits before it imports numpy
            saved = {k: os.environ.get(k) for k in BLAS_ENV}
            try:
                for k in BLAS_ENV:
                    os.environ[k] = str(self.blas_threads)
                proc.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        child_conn.close()
        self.spawned += 1
        return _Worker(slot, self._gen, proc, parent_conn, log_path)

    # ------------------------------------------------------------------ owner API
    def idle(self) -> list[_Worker]:
        return [w for w in self.workers if w.state == "idle"]

    def busy(self) -> list[_Worker]:
        return [w for w in self.workers if w.state == "busy"]

    def assign(self, w: _Worker, item_id: Any, payload: Any, timeout_s: float | None = None) -> bool:
        """Send one item to an idle worker. False if the worker turned out to be dead (the item was not started;
        the death is reported by the next `poll`)."""
        if w.state != "idle":
            raise RuntimeError(f"{w} is not idle")
        try:
            w.conn.send(("job", item_id, payload))
        except (BrokenPipeError, ConnectionError, OSError):
            return False
        w.state = "busy"
        w.item = item_id
        w.t_assign = time.monotonic()
        w.deadline = None if not timeout_s else w.t_assign + float(timeout_s)
        return True

    def poll(self, timeout: float = 0.0, extra: Iterable[Any] = ()) -> list[tuple]:
        """Wait up to `timeout` seconds for something to happen; return the events (possibly none). `extra`
        waitables (e.g. a wake-up pipe) end the wait early."""
        from multiprocessing.connection import wait
        if self._closed:
            return []
        if self._bad_starts >= 3:
            raise PoolBroken(f"{self.name}: workers die before becoming ready", self._last_evidence)
        now = time.monotonic()
        deadlines = [w.deadline for w in self.workers if w.state == "busy" and w.deadline is not None]
        if deadlines:
            timeout = max(0.0, min(timeout, min(deadlines) - now))
        every = self.workers + self._retiring
        objs: list[Any] = list(extra)
        for w in every:
            if w.state != "dead":
                objs.append(w.conn)
                objs.append(w.proc.sentinel)
        try:
            wait(objs, timeout)
        except OSError:
            pass
        events: list[tuple] = []
        now = time.monotonic()
        for w in list(every):
            if w.state == "dead":
                continue
            self._drain(w, events)
            if w.state == "busy":
                if now - w.t_track > 1.0:            # remember its children while they can still be found
                    w.t_track = now
                    w.tracked.update(_descendants(w.pid))
                if w.deadline is not None and now > w.deadline:
                    w.tracked.update(_descendants(w.pid))
                    try:
                        w.proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                    w.proc.join(5)
                    self._on_death(w, events, kind="timeout")
                    continue
            elif w.state == "retiring" and w.deadline is not None and now > w.deadline:
                try:
                    w.proc.kill()                    # it was asked to stop and did not
                except Exception:  # noqa: BLE001
                    pass
                w.proc.join(5)
            if not w.proc.is_alive():
                self._drain(w, events)               # a last message may have been written before it died
                self._on_death(w, events)
        return events

    def _drain(self, w: _Worker, events: list[tuple]) -> None:
        while w.state != "dead":
            try:
                if not w.conn.poll(0):
                    return
                msg = w.conn.recv()
            except (EOFError, OSError, pickle.UnpicklingError):
                return                               # the process is gone or going; is_alive() settles it
            kind = msg[0]
            if kind == "ready":
                w.info = msg[1]
                if w.state == "starting":
                    w.state = "idle"
                self._bad_starts = 0
                events.append(("ready", w, msg[1]))
            elif kind == "done":
                _, item_id, ok, payload = msg
                w.jobs += 1
                w.item = None
                w.deadline = None
                w.state = "idle"
                events.append(("done", w, item_id, ok, payload))
                if self.max_jobs and w.jobs >= self.max_jobs and not self._closing:
                    self.retire(w)
            elif kind == "fatal":
                w.fatal = msg[1]

    def retire(self, w: _Worker) -> None:
        """Stop an idle worker after its current state is drained and start a fresh one in its slot."""
        if w.state != "idle" or w not in self.workers:
            return
        try:
            w.conn.send(("stop",))
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        w.state = "retiring"
        w.deadline = time.monotonic() + 30.0
        self._retiring.append(w)
        self.recycled += 1
        self.workers[self.workers.index(w)] = self._spawn(w.slot)

    def _on_death(self, w: _Worker, events: list[tuple], kind: str = "worker_died") -> None:
        w.proc.join(5)
        code = w.proc.exitcode
        was = w.state
        item = w.item
        w.state = "dead"
        w.item = None
        killed = _kill_tracked(w.tracked)
        self.orphans_killed += len(killed)
        try:
            w.conn.close()
        except Exception:  # noqa: BLE001
            pass
        if w in self._retiring:
            self._retiring.remove(w)
            return                                   # an expected exit
        if self._closing:
            return
        evidence = {"kind": kind, "pid": w.pid, "worker": w.slot, "exitcode": code,
                    "signal": signal.Signals(-code).name if code is not None and code < 0 and -code in
                    signal.Signals._value2member_map_ else None,
                    "state": was, "jobs_done": w.jobs, "ran_s": round(time.monotonic() - (w.t_assign or w.t_spawn), 3),
                    "orphans_killed": killed, "stderr_tail": _tail(w.log_path)}
        if w.fatal is not None:
            evidence["init_error"] = w.fatal
        self._last_evidence = evidence
        self.died += 1
        _cleanup_sandbox_dirs(w.pid)
        if was == "starting":
            self._bad_starts += 1
        events.append(("died", w, item, evidence))
        if w in self.workers and self._bad_starts < 3:
            self.workers[self.workers.index(w)] = self._spawn(w.slot)

    def wait_ready(self, timeout: float = 120.0) -> list[tuple]:
        """Block until every worker is ready (or raise). Returns the events seen meanwhile (e.g. early deaths)."""
        events: list[tuple] = []
        end = time.monotonic() + timeout
        while any(w.state == "starting" for w in self.workers):
            left = end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{self.name}: workers not ready after {timeout}s")
            events.extend(self.poll(min(0.5, left)))
        return events

    # ------------------------------------------------------------------ shutdown
    def close(self, timeout: float = 30.0) -> None:
        """Graceful: ask every worker to stop after the item it is running, wait, then force what is left."""
        if self._closed or os.getpid() != self._owner_pid:
            return
        self._closing = True
        every = [w for w in self.workers + self._retiring if w.state != "dead"]
        for w in every:
            w.tracked.update(_descendants(w.pid))
            try:
                w.conn.send(("stop",))
            except (BrokenPipeError, ConnectionError, OSError):
                pass
        self._reap(every, timeout)
        self.terminate()

    def terminate(self) -> None:
        """Forceful and idempotent: SIGTERM (workers unwind and close their worlds), SIGKILL after the grace
        period, then kill any process a worker left behind."""
        if self._closed or os.getpid() != getattr(self, "_owner_pid", None):
            return
        self._closing = True
        every = [w for w in self.workers + self._retiring if w.state != "dead"]
        for w in every:
            w.tracked.update(_descendants(w.pid))
            if w.proc.is_alive():
                try:
                    w.proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
        self._reap(every, self.term_grace_s)
        for w in every:
            if w.proc.is_alive():
                try:
                    w.proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        for w in every:
            w.proc.join(10)
            self.orphans_killed += len(_kill_tracked(w.tracked))
            try:
                w.conn.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                w.proc.close()
            except Exception:  # noqa: BLE001
                pass
            w.state = "dead"
        self.workers, self._retiring = [], []
        self._closed = True
        _LIVE.discard(self)
        if self._own_log_dir:
            shutil.rmtree(self.log_dir, ignore_errors=True)

    @staticmethod
    def _reap(workers: list[_Worker], timeout: float) -> None:
        from multiprocessing.connection import wait
        end = time.monotonic() + timeout
        while True:
            alive = [w for w in workers if w.proc.is_alive()]
            left = end - time.monotonic()
            if not alive or left <= 0:
                return
            for w in alive:                          # a worker blocked sending a large result would never stop
                try:
                    while w.conn.poll(0):
                        w.conn.recv()
                except Exception:  # noqa: BLE001
                    pass
            try:
                wait([w.proc.sentinel for w in alive], min(left, 0.2))
            except OSError:
                return

    def __del__(self):
        try:
            self.terminate()
        except Exception:  # noqa: BLE001
            pass


def _cleanup_sandbox_dirs(pid: int) -> None:
    """Remove the sandbox work directories a dead worker could not clean up itself (they are named after its
    pid; sandboxed programs themselves exit when their pipe to the worker closes)."""
    try:
        import glob
        if os.path.exists(f"/proc/{pid}"):
            return
        base = os.environ.get("CRUCIBLE_SANDBOX_DIR", "/var/tmp/crucible-sbx")
        for d in glob.glob(os.path.join(base, f"{pid}-*")):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


# ====================================================================== world jobs
_WORLD_KEYS = ("family", "seed", "overrides", "extra_modules", "drop_modules")
_JOB_DEFAULTS: dict[str, Any] = {
    "world": None,                # {"family", "seed", "overrides", "extra_modules", "drop_modules"}
    "resume": None,               # path of a snapshot file written by an earlier job (instead of "world")
    "days": None,                 # horizon: simulated days from the world's time at job start, or
    "until_day": None,            # an absolute simulated day, or
    "until_s": None,              # an absolute simulated second (exactly one of the three)
    "agents": (),                 # [(handle, "module:callable", kwargs[, {"endowment": cents}])]
    "endowment": 3_000_000,       # cents granted to each agent admitted by this job
    "dt_s": 3600,                 # simulated seconds per round (agents act once per round)
    "chunk_s": DAY,               # stepping grid of an agent-free world; also the timeline/checkpoint grid
    "runner": "sync",             # "sync" (agents.base.Runner) or "async" (AsyncRunner, independent clocks)
    "think_mean_s": 600.0,        # AsyncRunner only
    "agent_seed": 0,
    "record": False,              # keep agent trajectories in memory (passed to the summariser as `trajectories`)
    "snapshot_every_days": None,  # write a checkpoint every k simulated days (needs a store directory)
    "final_snapshot": True,       # write a snapshot at the horizon (needs a store directory)
    "summarise": None,            # "module:function", called as fn(world, **summarise_kwargs) in the worker
    "summarise_kwargs": None,
    "imports": (),                # extra python modules to import in the worker before the world is built
    "discover": True,             # run-time discovery of family/module files (see ensure_registered)
    "tag": None,                  # returned unchanged
    "name": None,                 # sub-directory of the store directory (default: job-<id>)
    "world_id": None,
    "store": True,                # False: keep this job in memory even if the pool has a store directory
    "strict": False,              # kernel strict mode: handler bugs raise instead of becoming rejected commands
    "timeout_s": None,            # wall-clock limit; the worker is killed and the job reported failed
    "check_invariants": False,
    "timeline": True,             # one row per chunk: simulated time, events, wall seconds, store bytes
}


def world_job(spec: dict | str | None = None, days: float | None = None, *, agents: Iterable = (), **options) -> dict:
    """Build a job dict. `spec` is a family name or ``{"family", "seed", "overrides", "extra_modules"}``; omit it
    when ``resume=<snapshot file>`` is given. Every other key of `_JOB_DEFAULTS` may be passed as an option."""
    job: dict[str, Any] = {"agents": [tuple(a) for a in agents], **options}
    if spec is not None:
        job["world"] = {"family": spec} if isinstance(spec, str) else dict(spec)
    if days is not None:
        job["days"] = days
    return normalise_job(job)


def normalise_job(job: dict) -> dict:
    """Validate a job and fill in defaults. Raises ValueError with the offending key."""
    if not isinstance(job, dict):
        raise ValueError(f"a job is a dict, got {type(job).__name__}")
    j = dict(job)
    top = {k: j.pop(k) for k in _WORLD_KEYS if k in j}
    unknown = sorted(set(j) - set(_JOB_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown job keys {unknown}; known: {sorted(_JOB_DEFAULTS) + list(_WORLD_KEYS)}")
    out = {**_JOB_DEFAULTS, **j}
    world = dict(out["world"] or {})
    if isinstance(job.get("world"), str):
        world = {"family": job["world"]}
    world.update(top)
    if out["resume"] is None:
        if "family" not in world:
            raise ValueError("a job needs a world ('family', 'seed', ...) or 'resume'")
        bad = sorted(set(world) - set(_WORLD_KEYS))
        if bad:
            raise ValueError(f"unknown world keys {bad}; known: {list(_WORLD_KEYS)}")
        world.setdefault("seed", 0)
        world["seed"] = int(world["seed"])
        out["world"] = world
    else:
        if world:
            raise ValueError("give either a world or 'resume', not both")
        out["world"] = None
        out["resume"] = os.fspath(out["resume"])
    if sum(out[k] is not None for k in ("days", "until_day", "until_s")) != 1:
        raise ValueError("give exactly one horizon: 'days', 'until_day' or 'until_s'")
    agents = []
    for a in out["agents"] or ():
        a = tuple(a)
        if len(a) not in (2, 3, 4) or not isinstance(a[0], str) or not isinstance(a[1], str) or ":" not in a[1]:
            raise ValueError(f"an agent is (handle, 'module:callable', kwargs[, options]), got {a!r}")
        agents.append((a[0], a[1], dict(a[2]) if len(a) > 2 and a[2] else {}, dict(a[3]) if len(a) > 3 and a[3] else {}))
    out["agents"] = agents
    if out["runner"] not in ("sync", "async"):
        raise ValueError("runner must be 'sync' or 'async'")
    if int(out["dt_s"]) < 1 or int(out["chunk_s"]) < 1:
        raise ValueError("dt_s and chunk_s must be positive")
    if out["summarise"] is not None and ":" not in str(out["summarise"]):
        raise ValueError("summarise must be 'module:function'")
    out["imports"] = list(out["imports"] or ())
    out["summarise_kwargs"] = dict(out["summarise_kwargs"] or {})
    return out


def _store_bytes(store, path: str) -> int:
    """Size of a SQLite event log on disk, with the write-ahead log folded in first when possible."""
    try:
        store._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()   # noqa: SLF001 - same package tree
    except Exception:  # noqa: BLE001
        pass
    n = 0
    for p in (path, path + "-wal"):
        try:
            n += os.path.getsize(p)
        except OSError:
            pass
    return n


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _agent_state(runner, agents, kind: str) -> dict:
    """What a checkpoint needs beyond the world: runner randomness and cursors, each policy's generator and, if
    the policy offers `state_dict()`, its own state. Policies without it restart from scratch on resume."""
    side: dict[str, Any] = {"kind": kind, "runner_rng": runner.rng.bit_generator.state, "cursor": dict(runner.cursor),
                            "agents": {}}
    if kind == "async":
        side["heap"] = list(runner.heap)
        side["n"] = runner._n                        # noqa: SLF001
    for pid, pol in agents:
        rng = getattr(pol, "rng", None)
        st = pol.state_dict() if hasattr(pol, "state_dict") else None
        side["agents"][pid] = {"rng": rng.bit_generator.state if hasattr(rng, "bit_generator") else None, "state": st}
    return side


def _restore_agent_state(side: dict, runner, agents, kind: str) -> list[str]:
    restored = []
    if side.get("kind") != kind:
        return restored
    runner.rng.bit_generator.state = side["runner_rng"]
    runner.cursor.update({a: c for a, c in side["cursor"].items() if a in runner.cursor})
    if kind == "async":
        import heapq
        runner.heap = [tuple(x) for x in side["heap"]]
        heapq.heapify(runner.heap)
        runner._n = side["n"]                        # noqa: SLF001
    for pid, pol in agents:
        s = side["agents"].get(pid)
        if not s:
            continue
        if s["rng"] is not None and hasattr(getattr(pol, "rng", None), "bit_generator"):
            pol.rng.bit_generator.state = s["rng"]
        if s["state"] is not None and hasattr(pol, "load_state_dict"):
            pol.load_state_dict(s["state"])
            restored.append(pid)
    return restored


def run_world_job(job: dict, store_dir: str | None = None, *, job_id: int = 0, worker: dict | None = None) -> dict:
    """Run one world job in *this* process and return its result record. `WorldPool` calls this inside workers;
    calling it directly gives the single-process reference (same digest, by the determinism property).

    The record: ``ok``, ``job_id``, ``tag``, ``family``, ``seed``, ``world_id``, ``spec_hash``, ``digest``,
    ``last_hash``, ``seq``, ``sim_time_s``, ``counts`` (commands, rejected, events, process_runs,
    process_failures, agent_steps, rounds), ``wall_s`` / ``setup_wall_s`` / ``run_wall_s``, ``cpu_s`` (this
    process, whole job) and ``cpu_children_s`` (sandboxed programs it ran), ``run_cpu_s`` (both, for the
    simulation loop only: what ``run_wall_s`` should be compared with), ``peak_rss_mb`` with ``peak_rss_scope``
    (``"job"`` when the kernel let us reset the peak counter, else the worker's lifetime peak), ``rss_start_mb``
    and ``rss_end_mb`` (resident memory before the world was built and at the horizon),
    ``snapshot`` / ``checkpoints`` / ``event_log`` / ``store_bytes`` when a store directory is given,
    ``timeline``, ``summary``, and on failure ``kind`` and ``error`` (type, message, traceback, stage)."""
    t_wall = time.perf_counter()
    cpu0, kids0 = time.process_time(), _children_cpu()
    scope = "job" if _reset_peak_rss() else "process"
    out: dict[str, Any] = {"ok": False, "job_id": job_id, "tag": None, "name": None, "family": None, "seed": None,
                           "world_id": None, "spec_hash": None, "digest": None, "last_hash": None, "seq": None,
                           "sim_time_s": None, "sim_days": None, "counts": None, "snapshot": None, "checkpoints": [],
                           "event_log": None, "store_bytes": None, "timeline": [], "summary": None, "kind": None,
                           "error": None, "resumed_from": None, "registration_errors": {}}
    if worker:
        out.update(worker)
    out["rss_start_mb"] = round(_rss_mb(), 1)        # before the world exists: the job's own memory is peak - start
    world = store = None
    stage = "spec"
    try:
        job = normalise_job(job)
        out["tag"] = job["tag"]
        name = job["name"] or f"job-{job_id:06d}"
        out["name"] = name
        reg = ensure_registered(job["imports"], discover=job["discover"])
        out["registration_errors"] = reg["errors"]
        import zlib

        from ..agents.base import AsyncRunner, Runner
        from ..api.local import ensure_modules
        from ..core.kernel import World
        from ..worldgen import generate

        # ---------------------------------------------------------------- store
        stage = "store"
        job_dir = db_path = None
        if store_dir and job["store"]:
            job_dir = os.path.join(store_dir, name)
            db_path = os.path.join(job_dir, "events.sqlite")
            if os.path.exists(db_path):
                raise FileExistsError(f"{db_path} exists: a job never overwrites recorded events; "
                                      f"give the job another 'name' or the pool another store_dir")
            os.makedirs(job_dir, exist_ok=True)
            from ..storage.sqlite import SQLiteStore
            store = SQLiteStore(db_path)
            out["event_log"] = db_path
        elif job["snapshot_every_days"]:
            raise ValueError("snapshot_every_days needs a store directory")

        # ---------------------------------------------------------------- world
        stage = "setup"
        side = None
        if job["resume"]:
            with open(job["resume"], "rb") as f:
                blob = f.read()
            head = pickle.loads(zlib.decompress(blob))
            ensure_modules([m["name"] for m in head["spec"]["modules"]])
            wid = job["world_id"] or f"{head['world_id']}.r{head['seq']}"
            world = World.from_blob(blob, world_id=wid, store=store, fork=True, strict=job["strict"])
            out["resumed_from"] = {"snapshot": job["resume"], "world_id": head["world_id"], "seq": head["seq"],
                                   "hash": head["hash"]}
            if os.path.exists(job["resume"] + ".agents"):
                with open(job["resume"] + ".agents", "rb") as f:
                    side = pickle.load(f)
            del blob, head
        else:
            spec = generate(dict(job["world"]), job["world"]["seed"])
            ensure_modules([m["name"] for m in spec.modules])
            world = World.create(spec, job["world_id"], store, strict=job["strict"])
        out.update(family=world.spec.family, seed=world.spec.seed, world_id=world.world_id,
                   spec_hash=world.spec.hash())
        n_failed = [0]

        def on_event(ev, _n=n_failed):
            if ev.type == "proc.failed":
                _n[0] += 1
        world.listeners.append(on_event)

        agents = []
        for handle, factory, kwargs, opts in job["agents"]:
            pid = f"p:{handle}"
            if world.principal(pid) is None:
                world.system_call("admin.join", {"handle": handle,
                                                 "endowment": int(opts.get("endowment", job["endowment"]))})
            agents.append((pid, resolve(factory)(**kwargs)))
        kind = job["runner"] if agents else "sync"
        chunk = int(job["chunk_s"])
        step = int(job["dt_s"]) if agents else chunk
        if kind == "async":
            runner = AsyncRunner(world, agents, seed=job["agent_seed"], think_mean_s=job["think_mean_s"])
        else:
            runner = Runner(world, agents, dt=step, seed=job["agent_seed"], record=bool(job["record"]))
        out["agents_restored"] = _restore_agent_state(side, runner, agents, kind) if side else []

        t0_sim = world.now
        if job["until_s"] is not None:
            t_end = int(job["until_s"])
        elif job["until_day"] is not None:
            t_end = int(round(float(job["until_day"]) * DAY))
        else:
            t_end = t0_sim + int(round(float(job["days"]) * DAY))
        if t_end < t0_sim:
            raise ValueError(f"horizon {t_end}s is before the world's time {t0_sim}s")
        ckpt_s = int(round(float(job["snapshot_every_days"]) * DAY)) if job["snapshot_every_days"] else 0
        next_mark = (t0_sim // chunk + 1) * chunk
        next_ckpt = (t0_sim // ckpt_s + 1) * ckpt_s if ckpt_s else None
        rounds = actions = 0
        setup_s = time.perf_counter() - t_wall
        out["setup_wall_s"] = round(setup_s, 4)

        def write_snapshot(path: str) -> dict:
            t = time.perf_counter()
            world.flush()
            blob_ = world.snapshot_blob()
            _write_atomic(path, blob_)
            if agents:
                _write_atomic(path + ".agents", pickle.dumps(_agent_state(runner, agents, kind), protocol=5))
            return {"t": world.now, "day": round(world.now / DAY, 6), "path": path, "seq": world.seq,
                    "hash": world.last_hash, "bytes": len(blob_), "write_ms": round((time.perf_counter() - t) * 1e3, 2)}

        # ---------------------------------------------------------------- run
        stage = "run"
        t_run = time.perf_counter()
        cpu_run, kids_run = time.process_time(), _children_cpu()
        while world.now < t_end:
            now = world.now
            if kind == "async":
                actions += runner.run_until(min(t_end, (now // chunk + 1) * chunk))
            elif now + step <= t_end:
                runner.round()
                rounds += 1
            else:
                world.advance_time(t_end)
            if world.now >= next_mark or world.now >= t_end:
                world.flush()
                if job["timeline"]:
                    out["timeline"].append({"t": world.now, "seq": world.seq,
                                            "wall_s": round(time.perf_counter() - t_run, 4),
                                            "store_bytes": _store_bytes(store, db_path) if db_path else None})
                next_mark = (world.now // chunk + 1) * chunk
                if next_ckpt is not None and world.now >= next_ckpt and world.now < t_end:
                    stage = "checkpoint"
                    out["checkpoints"].append(write_snapshot(os.path.join(job_dir, f"snap-t{world.now:012d}.blob")))
                    next_ckpt = (world.now // ckpt_s + 1) * ckpt_s
                    stage = "run"
        run_s = time.perf_counter() - t_run
        out["run_wall_s"] = round(run_s, 4)
        out["run_cpu_s"] = round(time.process_time() - cpu_run + _children_cpu() - kids_run, 4)

        # ---------------------------------------------------------------- record
        stage = "finalise"
        world.flush()
        out["rss_end_mb"] = round(_rss_mb(), 1)
        st = world.stats
        out.update(digest=world.digest(), last_hash=world.last_hash, seq=world.seq, sim_time_s=world.now,
                   sim_days=round(world.now / DAY, 6), sim_days_run=round((world.now - t0_sim) / DAY, 6),
                   counts={"commands": st["commands"], "rejected": st["rejected"], "events": st["events"],
                           "process_runs": st["procs"], "process_failures": n_failed[0],
                           "agent_steps": rounds * len(agents) if kind == "sync" else actions, "rounds": rounds})
        if job["check_invariants"]:
            out["invariant_violations"] = world.check_invariants()
        if job_dir and job["final_snapshot"]:
            snap = write_snapshot(os.path.join(job_dir, "final.blob"))
            out["snapshot"] = snap["path"]
            out["snapshot_bytes"] = snap["bytes"]
            out["snapshot_write_ms"] = snap["write_ms"]
        if db_path:
            out["store_bytes"] = _store_bytes(store, db_path)
        if job["summarise"]:
            stage = "summarise"
            fn = resolve(job["summarise"])
            kw = dict(job["summarise_kwargs"])
            import inspect
            params = inspect.signature(fn).parameters
            if "trajectories" in params and kind == "sync":
                kw.setdefault("trajectories", runner.trajs)
            if "policies" in params:
                kw.setdefault("policies", dict(agents))
            out["summary"] = fn(world, **kw)
            pickle.dumps(out["summary"], protocol=5)  # must be able to leave the process
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - the world failed; the process and the other worlds live on
        out["kind"] = "exception"
        out["error"] = _err(e, stage)
        out["summary"] = None
        if world is not None:
            try:
                out.update(seq=world.seq, sim_time_s=world.now)
            except Exception:  # noqa: BLE001
                pass
    finally:
        if world is not None:
            try:
                world.close()
            except Exception:  # noqa: BLE001
                pass
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        world = store = None
        import gc
        gc.collect()
        out["wall_s"] = round(time.perf_counter() - t_wall, 4)
        out["cpu_s"] = round(time.process_time() - cpu0, 4)
        out["cpu_children_s"] = round(_children_cpu() - kids0, 4)
        out["peak_rss_mb"] = round(_peak_rss_mb(), 1)
        out["peak_rss_scope"] = scope
    return out


class _WorldJobHandler:
    """Worker-side handler of `WorldPool`."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n = 0
        ensure_registered(cfg.get("imports") or (), discover=cfg.get("discover", True))

    def handle(self, payload: tuple) -> dict:
        job_id, job = payload
        self.n += 1
        return run_world_job(job, self.cfg.get("store_dir"), job_id=job_id,
                             worker={"worker": self.cfg.get("worker"), "worker_pid": os.getpid(),
                                     "worker_job_index": self.n})

    def close(self) -> None:
        pass


def _failure(job_id: int, job: dict, kind: str, error: dict, evidence: dict | None = None, **more) -> dict:
    w = job.get("world") or {}
    return {"ok": False, "job_id": job_id, "tag": job.get("tag"), "name": job.get("name") or f"job-{job_id:06d}",
            "family": w.get("family"), "seed": w.get("seed"), "kind": kind, "error": error,
            "evidence": evidence or {}, "digest": None, "last_hash": None, "summary": None, **more}


class WorldPool:
    """A pool of worker processes that run world jobs (see the module docstring).

    Parameters: `n_workers` (default: up to 8); `store_dir` (each job gets ``<store_dir>/<name>/`` holding
    ``events.sqlite``, checkpoints ``snap-t<sim seconds>.blob`` and ``final.blob``; worker logs go to
    ``<store_dir>/logs``); `start_method` ("spawn" is the default and the only one that guarantees the BLAS
    thread limits and a clean interpreter; "fork" and "forkserver" also work); `max_jobs_per_worker` (a worker
    is replaced by a fresh process after this many jobs: bounded memory); `job_timeout_s` (default wall-clock
    limit per job); `imports` (python modules every worker imports first); `mem_limit_mb` (address-space limit
    per worker).

    The pool has no background thread: it makes progress whenever `poll`, `as_completed`, `map`, `run` or `join`
    is called. Use it from one thread."""

    def __init__(self, n_workers: int | None = None, store_dir: str | None = None, start_method: str = "spawn", *,
                 max_jobs_per_worker: int | None = 32, job_timeout_s: float | None = None,
                 imports: Iterable[str] = (), discover: bool = True, mem_limit_mb: int | None = None,
                 name: str = "worldpool"):
        self.store_dir = os.path.abspath(store_dir) if store_dir else None
        log_dir = None
        if self.store_dir:
            os.makedirs(self.store_dir, exist_ok=True)
            log_dir = os.path.join(self.store_dir, "logs")
        self.job_timeout_s = job_timeout_s
        self._jobs: dict[int, dict] = {}
        self._pending: collections.deque[int] = collections.deque()
        self._finished: dict[int, dict] = {}
        self._next_id = 1
        self.submitted = self.completed = self.failed = 0
        self._sup = Supervisor(n_workers, "crucible.scale.pool:_WorldJobHandler",
                               {"store_dir": self.store_dir, "imports": list(imports), "discover": discover,
                                "mem_limit_mb": mem_limit_mb},
                               start_method=start_method, max_jobs_per_worker=max_jobs_per_worker, name=name,
                               log_dir=log_dir)
        self.n_workers = self._sup.n_workers

    # ------------------------------------------------------------------ submitting
    def submit(self, job: dict | None = None, **kw) -> int:
        """Queue a job (a dict, see `world_job`; or keyword arguments forming one). Returns its id."""
        if self._sup._closing:                       # noqa: SLF001
            raise RuntimeError("pool is closed")
        job = normalise_job({**(job or {}), **kw})
        pickle.dumps(job, protocol=5)                # a job must be declarative: fail here, not in a worker
        jid = self._next_id
        self._next_id += 1
        self._jobs[jid] = job
        self._pending.append(jid)
        self.submitted += 1
        self._dispatch()
        return jid

    def _dispatch(self) -> None:
        for w in self._sup.idle():
            while self._pending:
                jid = self._pending.popleft()
                job = self._jobs[jid]
                if self._sup.assign(w, jid, (jid, job), job["timeout_s"] or self.job_timeout_s):
                    break
                self._pending.appendleft(jid)        # that worker is dead; poll() will replace it
                break

    def _finish(self, jid: int, res: dict) -> None:
        self._jobs.pop(jid, None)
        self._finished[jid] = res
        if res.get("ok"):
            self.completed += 1
        else:
            self.failed += 1

    def _pump(self, timeout: float) -> None:
        self._dispatch()
        try:
            events = self._sup.poll(timeout)
        except PoolBroken as e:
            for jid in list(self._pending):
                self._finish(jid, _failure(jid, self._jobs[jid], "pool_broken", {"type": "PoolBroken",
                                                                                 "message": str(e)}, e.evidence))
            self._pending.clear()
            raise
        for ev in events:
            if ev[0] == "done":
                _, w, jid, ok, payload = ev
                if jid not in self._jobs:
                    continue
                if ok:
                    self._finish(jid, payload)
                else:                                # an exception outside run_world_job's own handling
                    self._finish(jid, _failure(jid, self._jobs[jid], "exception", payload, worker=w.slot,
                                               worker_pid=w.pid))
            elif ev[0] == "died":
                _, w, jid, evidence = ev
                if jid is not None and jid in self._jobs:
                    what = "exceeded its time limit" if evidence["kind"] == "timeout" else \
                        f"died (exit code {evidence['exitcode']}, signal {evidence['signal']})"
                    self._finish(jid, _failure(jid, self._jobs[jid], evidence["kind"],
                                               {"type": "WorkerDied", "message": f"worker {w.slot} (pid {w.pid}) {what} "
                                                f"after {evidence['ran_s']}s", "stage": "run"},
                                               evidence, worker=w.slot, worker_pid=w.pid))
        self._dispatch()

    # ------------------------------------------------------------------ collecting
    @property
    def outstanding(self) -> int:
        """Jobs submitted and not yet finished (queued or running)."""
        return len(self._jobs)

    def poll(self, timeout: float = 0.0) -> list[dict]:
        """Make progress for up to `timeout` seconds; return (and forget) the results finished so far."""
        self._pump(timeout)
        out = list(self._finished.values())
        self._finished.clear()
        return out

    def as_completed(self, ids: Iterable[int] | None = None, timeout: float | None = None) -> Iterator[dict]:
        """Yield results as jobs finish (any order): the given job ids, or everything outstanding. Raises
        TimeoutError if `timeout` seconds pass without all of them finishing. KeyboardInterrupt terminates the
        pool before propagating."""
        want = set(ids) if ids is not None else None
        end = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                ready = [j for j in self._finished if want is None or j in want]
                for j in ready:
                    if want is not None:
                        want.discard(j)
                    yield self._finished.pop(j)
                if (want is not None and not want) or (want is None and not self._jobs and not self._finished):
                    return
                if want is not None and not any(j in self._jobs or j in self._finished for j in want):
                    raise KeyError(f"unknown or already collected job ids {sorted(want)}")
                if end is not None and time.monotonic() > end:
                    raise TimeoutError(f"{len(self._jobs)} jobs still running after {timeout}s")
                self._pump(0.25)
        except (KeyboardInterrupt, SystemExit):
            self.terminate()
            raise

    def map(self, jobs: Iterable[dict], timeout: float | None = None) -> list[dict]:
        """Submit the jobs and return their results in the order given."""
        ids = [self.submit(j) for j in jobs]
        got = {r["job_id"]: r for r in self.as_completed(ids, timeout)}
        return [got[i] for i in ids]

    def run(self, job: dict | None = None, **kw) -> dict:
        jid = self.submit(job, **kw)
        return next(iter(self.as_completed([jid])))

    def join(self, timeout: float | None = None) -> list[dict]:
        """Wait for everything outstanding; return the results in completion order."""
        return list(self.as_completed(timeout=timeout))

    def wait_ready(self, timeout: float = 120.0) -> float:
        """Block until every worker has finished importing; returns the seconds waited (for benchmarks)."""
        t = time.perf_counter()
        while any(w.state == "starting" for w in self._sup.workers):
            if time.perf_counter() - t > timeout:
                raise TimeoutError(f"workers not ready after {timeout}s")
            self._pump(0.25)
        return time.perf_counter() - t

    def stats(self) -> dict:
        s = self._sup
        return {"workers": self.n_workers, "spawned": s.spawned, "recycled": s.recycled, "died": s.died,
                "orphans_killed": s.orphans_killed, "submitted": self.submitted, "completed": self.completed,
                "failed": self.failed, "pending": len(self._pending),
                "in_flight": len(self._jobs) - len(self._pending), "start_method": s.start_method,
                "worker_pids": [w.pid for w in s.workers if w.state != "dead"]}

    # ------------------------------------------------------------------ shutdown
    def close(self, wait: bool = True, timeout: float | None = None) -> None:
        """Finish what was submitted, then stop the workers. Results not collected yet remain available from
        `results()`. With ``wait=False`` (or when `timeout` seconds pass) this is `terminate()`."""
        if self._sup._closed:                        # noqa: SLF001
            return
        try:
            if wait:
                end = None if timeout is None else time.monotonic() + timeout
                while self._jobs and (end is None or time.monotonic() < end):
                    self._pump(0.25)
        except (KeyboardInterrupt, SystemExit, PoolBroken):
            self.terminate()
            raise
        if self._jobs:
            self.terminate()
        else:
            self._sup.close(timeout=15.0)

    def terminate(self) -> None:
        """Stop now. Running and queued jobs are reported failed (kind ``terminated``)."""
        self._abandon("pool terminated")
        self._sup.terminate()

    def _abandon(self, why: str) -> None:
        for jid, job in list(self._jobs.items()):
            self._finish(jid, _failure(jid, job, "terminated", {"type": "Terminated", "message": why}))
        self._pending.clear()

    def results(self) -> list[dict]:
        """Results finished and not yet collected (does not make progress)."""
        out = list(self._finished.values())
        self._finished.clear()
        return out

    def __enter__(self) -> "WorldPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()
