"""Parent-side metering of sandboxed work (DECISION_LOG D-20).

Nothing a sandboxed process reports about itself is trusted for billing, capacity, program cost or adoption: the
process runs agent code, and agent code can rewrite anything in its own address space (counters, monitoring hooks,
the protocol functions; with `ctypes` even raw memory). The parent therefore measures from outside:

* **CPU time** of the whole worker process (every thread, live or exited) through the kernel's per-process CPU
  clock (``clock_gettime`` on ``MAKE_PROCESS_CPUCLOCK(pid, CPUCLOCK_SCHED)``, nanosecond resolution), read by the
  parent before the job is sent, when its result arrives and before the process is killed;
* **wall time** of the job (parent monotonic clock, time spent in brokered world calls excluded);
* **memory high-water** (``VmHWM`` of ``/proc/<pid>/status``), **threads alive** (``/proc/<pid>/task``) and the
  **bytes and entries in the job's working directory** (scanned by the parent during and after the job);
* **background CPU**: CPU the process consumed while it was not serving a job (a thread or code left running after
  the job's result was sent) is detected at the next contact and billed there.

All figures are clamped to be non-negative. The job's recorded outcome (``ctx.reserve/fill`` in
`crucible.sandbox.bridge`, ``ctx.external`` elsewhere) carries them, so replay and regeneration reuse the recorded
measurement exactly and never re-measure.

Deterministic cost model (``params.cost_model = "deterministic"``, ``SandboxConfig.count_lines``). Measured CPU time
jitters by a few per cent between runs, which would make forks and repeated runs of the same world drift apart. The
deterministic model therefore bills *the first parent-side measurement of a computation* for every later identical
computation in the same world lineage (`Memo`): the key is the world lineage (`SandboxConfig.meter_scope`), the
program's files, the job, every brokered world call with its reply, and the outcome. A later measurement may differ
from the remembered one by measurement noise; when it exceeds ``MEMO_TOL`` times the remembered value plus
``MEMO_SLACK_NS`` the fresh (higher) measurement is billed instead and the job is flagged - so remembering never
lets a program do much more work than it is billed for. Programs that behave identically are billed identically
across forks, replays from snapshots taken in the same process and repeated runs. With a machine-level memo file
(`DiskMemo`, ``SandboxConfig.memo_path``; worlds use `default_memo_path()` unless ``params.meter_memo`` is
"process") the anchors are shared by every process of the machine (first writer wins), so a rerun, a resumed run or
a replication of the same world in a NEW process bills exactly the same and the same spec and seed give the same
event hashes; without it, or on another machine with an empty memo, a new process measures afresh (agreement only up
to measurement noise - the honest limit of measuring real work). Worlds that must bill alike although their specs
differ (an experiment's arms that add or remove a module) share a lineage with ``params.meter_lineage``.

Units. ``work_units`` are CPU nanoseconds divided by ``NS_PER_WORK_UNIT`` (calibrated on the reference need solvers:
about one of their executed CPython lines, including the C work it triggers, on the build host - so magnitudes of
realistic service code stay those of the retired executed-line model), after subtracting the protocol work of the job
(``JOB_OVERHEAD_NS`` plus ``PROTO_NS_PER_BYTE`` per byte of the job message and of its result message, both sizes
known to the parent). Unlike the retired line count they include work done in C (builtins, numpy), which is real
compute; simple arithmetic loops cost fewer units than lines, C-heavy calls many more.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

NS_PER_WORK_UNIT = 350.0         # CPU ns per work unit: ~ one executed line of the reference need solvers, build host
JOB_OVERHEAD_NS = 30_000         # fixed protocol overhead of one job (dispatch, output capture), never billed
PROTO_NS_PER_BYTE = 20           # decoding the job and encoding the result (CPython json ~30-40 ns/byte), never billed
LOAD_FREE_NS = 3_000_000         # load-time CPU never billed: the import system finding and compiling a small
                                 # program (~2 ms measured). Trusted imports the program names (stdlib modules of
                                 # runner.PRELOAD_STDLIB, approved packages such as numpy, ~65 ms) are preloaded by
                                 # the boot module before the first job and cost nothing; the program's own
                                 # module-level work beyond this allowance is billed (it was free up to 250 ms per
                                 # cold start before: a loop moved to import time ran unbilled)
BACKGROUND_SLACK_NS = 2_000_000  # CPU an idle worker may use between jobs (reading the next request)
MEMO_TOL = 3.0
MEMO_SLACK_NS = 1_000_000
MEMO_MAX = 200_000
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _clockid(pid: int) -> int:
    return ((~int(pid)) << 3) | 2          # MAKE_PROCESS_CPUCLOCK(pid, CPUCLOCK_SCHED)


def cpu_ns(pid: int | None) -> int | None:
    """CPU time (ns) consumed so far by the whole process `pid` (all threads), or None if it cannot be read."""
    if pid is None:
        return None
    try:
        return int(time.clock_gettime_ns(_clockid(pid)))
    except (OSError, ValueError, OverflowError):
        pass
    try:                                   # a process that has exited but is not reaped yet: /proc stat ticks
        with open(f"/proc/{pid}/stat") as fh:
            f = fh.read().rsplit(")", 1)[1].split()
        return int((int(f[11]) + int(f[12])) * 1_000_000_000 // CLK_TCK)
    except (OSError, IndexError, ValueError):
        return None


def state(pid: int | None) -> str:
    """Scheduler state of the process's main thread ('R' running, 'S' sleeping, ...; '' if unknown)."""
    if pid is None:
        return ""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return ""


def settle(pid: int | None, max_s: float = 0.005) -> bool:
    """Wait (briefly) until the process is blocked. The kernel folds a running task's CPU into the process clock only
    at scheduler ticks and context switches, so a reading taken while the worker still runs - it has just written
    its result and is on its way back to reading the next request - can miss up to one tick of work; once it blocks
    the reading is exact. A worker that keeps running after its result (forged completion) is not waited for: the
    work it does afterwards is billed as background work at the next contact."""
    end = time.monotonic() + max_s
    while True:
        st = state(pid)
        if st != "R":
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.00005)


def threads(pid: int | None) -> int:
    if pid is None:
        return 0
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return 0


def hwm_kb(pid: int | None) -> int:
    if pid is None:
        return 0
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def dir_usage(path: str | None, max_entries: int) -> tuple[int, int]:
    """(bytes, entries) under `path`, without following symlinks. Stops counting entries at max_entries + 1."""
    if not path:
        return 0, 0
    total, n = 0, 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                n += 1
                if n > max_entries:
                    return total, n
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                else:
                    total += int(getattr(st, "st_blocks", 0)) * 512 or int(st.st_size)
    return total, n


def work_units(ns: int, proto_bytes: int = 0) -> int:
    return int(max(0, int(ns) - JOB_OVERHEAD_NS - PROTO_NS_PER_BYTE * max(0, int(proto_bytes))) // NS_PER_WORK_UNIT)


def digest(x: Any) -> str:
    return hashlib.blake2b(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode(),
                           digest_size=16).hexdigest()


MEMO_DISK_MAX = 2_000_000       # anchors kept in a machine-level memo file (the oldest are dropped beyond)


def default_memo_path() -> str | None:
    """The machine-level memo file worlds share (``$CRUCIBLE_METER_MEMO``; ``off`` keeps memos per process), by default
    ``$CRUCIBLE_CACHE/meter/memo-v1.sqlite`` (``~/.cache/crucible`` when unset, like the shared-space cache)."""
    v = os.environ.get("CRUCIBLE_METER_MEMO")
    if v is not None:
        return None if v.strip().lower() in ("", "0", "off", "none", "process") else v
    base = os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")
    return os.path.join(base, "meter", "memo-v1.sqlite")


class DiskMemo:
    """The first measurement of each computation, shared by every process of this machine (workstream A, D-21 / K3
    after D-20): an SQLite table ``memo(key, ns)`` where the first writer wins (``INSERT OR IGNORE``), so concurrent
    processes agree on one anchor. Without it a NEW process measured every computation afresh and the same spec and
    seed gave different event hashes run to run (cross-process resume, replication); with it a rerun, a resumed run
    or an independent replication on this machine bills the anchor exactly as the in-process memo does. Keys already
    contain the world lineage (`computation_key`), so worlds never share anchors unless they share a lineage. The
    file is written by the host process only (sandboxed workers cannot reach it)."""

    def __init__(self, path: str):
        self.path = path
        self._con = None
        self._pid = None
        self._n = 0

    def _conn(self):
        import sqlite3
        if self._con is None or self._pid != os.getpid():      # a connection never crosses a fork
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, ns INTEGER NOT NULL)")
            self._con, self._pid = con, os.getpid()
        return self._con

    def first(self, key: str, fresh_ns: int) -> tuple[int, bool]:
        """(the computation's anchor, whether `fresh_ns` just became it)."""
        con = self._conn()
        if con.execute("INSERT OR IGNORE INTO memo(key, ns) VALUES (?, ?)", (key, int(fresh_ns))).rowcount == 1:
            self._n += 1
            if self._n % 10_000 == 0:
                hi = con.execute("SELECT max(rowid) FROM memo").fetchone()[0] or 0
                con.execute("DELETE FROM memo WHERE rowid <= ?", (hi - MEMO_DISK_MAX,))
            return int(fresh_ns), True
        row = con.execute("SELECT ns FROM memo WHERE key = ?", (key,)).fetchone()
        return (int(row[0]), False) if row is not None else (int(fresh_ns), True)


_DISKS: dict[str, DiskMemo | None] = {}


def _disk(path: str | None) -> DiskMemo | None:
    """The memo file at `path` (None when no path, or after it failed once in this process)."""
    if not path:
        return None
    if path not in _DISKS:
        _DISKS[path] = DiskMemo(path)
    return _DISKS[path]


class Memo:
    """First measurement of each computation per world lineage (bounded, least-recently-used eviction), optionally
    backed by a machine-level `DiskMemo` (``disk=`` path) so that other processes bill the same anchors."""

    def __init__(self, max_entries: int = MEMO_MAX):
        self.max = max_entries
        self._d: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.exceeded = 0
        self.disk_errors = 0

    def bill(self, key: str, fresh_ns: int, disk: str | None = None) -> tuple[int, str]:
        fresh_ns = max(0, int(fresh_ns))
        with self._lock:
            rec = self._d.get(key)
            dm = _disk(disk) if rec is None else None
            if dm is not None:
                try:
                    anchor, mine = dm.first(key, fresh_ns)
                except Exception:  # noqa: BLE001 - an unusable memo file degrades to the in-process memo
                    self.disk_errors += 1
                    _DISKS[disk] = None
                    anchor, mine = None, True
                if anchor is not None:
                    self._d[key] = anchor
                    if len(self._d) > self.max:
                        self._d.popitem(last=False)
                    if mine:
                        self.misses += 1
                        return fresh_ns, "first"
                    rec = anchor          # another process measured it first: the same tolerance rule applies
            if rec is None:
                self._d[key] = fresh_ns
                if len(self._d) > self.max:
                    self._d.popitem(last=False)
                self.misses += 1
                return fresh_ns, "first"
            self._d.move_to_end(key)
            if fresh_ns <= rec * MEMO_TOL + MEMO_SLACK_NS:
                self.hits += 1
                return rec, "remembered"
            self.exceeded += 1
            return fresh_ns, "exceeded"

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


MEMO = Memo()


def computation_key(scope: str | None, files_digest: str, job: dict, transcript: list, outcome: dict) -> str:
    return digest([scope or "", files_digest, {k: job.get(k) for k in ("kind", "target", "args", "caller", "pattern")},
                   transcript, outcome])
