"""cam_live.harness.jobs — bounded background jobs (workstream EC, spec 12A.11).

Every job has an explicit purpose, resource limits, an execution deadline, an observable status, cancellation and
a persisted result record under ``<run_dir>/jobs/<job_id>/``. At most ``max_concurrent`` jobs run at once; further
submissions wait in a bounded queue. A finished job is delivered as an *event* (an ``Observation`` of kind
``event``), appended to ``<run_dir>/jobs/events.jsonl`` and handed to the event sink. Nothing ends silently: a job
that was queued or running when the process stopped is reported as ``interrupted`` by the next manager that opens
the run directory (and whatever is left of its process group is killed).

Two kinds of job:

* ``python`` / ``shell`` — generated code in the sandbox (:mod:`cam_live.harness.sandbox`); limits and the deadline
  are enforced by the sandbox monitor and the kernel.
* ``callable`` — a function of the harness itself (for example a read-only research session of another executor),
  run on a thread. Its deadline and cancellation are cooperative: the function receives a cancel event that is set
  at the deadline, and a result that arrives after the deadline is discarded and recorded as ``timeout``.

Jobs report facts (exit status, output, resource use). They never return a reward.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .. import contracts as c
from .sandbox import EgressBroker, RunResult, Sandbox, SandboxLimits, _authorize, _scrub

__all__ = ["JobState", "JobRecord", "JobManager", "JobLimitError", "execute_job_tool"]


class JobState:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"      # the owning process stopped before the job ended
    LIVE = (QUEUED, RUNNING)
    FINAL = (SUCCEEDED, FAILED, TIMEOUT, CANCELLED, INTERRUPTED)


class JobLimitError(RuntimeError):
    """The job queue is full."""


@dataclass
class JobRecord(c.Record):
    job_id: str
    purpose: str
    kind: str                                    # "python" | "shell" | "callable"
    state: str = JobState.QUEUED
    created: float = 0.0
    started: float = 0.0
    ended: float = 0.0
    deadline_s: float = 0.0                      # execution deadline, counted from ``started``
    limits: Dict[str, Any] = field(default_factory=dict)
    network: str = "none"
    action_id: str = ""
    manager_id: str = ""                         # the manager instance that owns the live job
    pgid: int = 0
    pgid_starttime: int = 0
    source_file: str = ""
    source_sha256: str = ""
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    killed_for: str = ""
    limit_hit: str = ""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    max_rss_kb: int = 0
    peak_tasks: int = 0
    leftover_processes: int = 0
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    connections: List[Dict[str, Any]] = field(default_factory=list)
    result: Any = None                           # JSON value returned by a callable job
    error: str = ""
    cancel_requested: bool = False

    @property
    def done(self) -> bool:
        return self.state in JobState.FINAL


def _starttime(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return int(f.read().rsplit(b")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


def _tail(path: str, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


class _Live:
    """Runtime-only state of one job."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.start: Optional[Callable[[], None]] = None


class JobManager:
    def __init__(self, run_dir: str, sandbox: Optional[Sandbox] = None, clock: Optional[c.Clock] = None,
                 event_sink: Optional[Callable[[c.Observation], None]] = None, max_concurrent: int = 4,
                 max_queued: int = 32, max_deadline_s: float = 3600.0, tail_bytes: int = 4096):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.run_dir = os.path.abspath(run_dir)
        self.dir = os.path.join(self.run_dir, "jobs")
        os.makedirs(self.dir, exist_ok=True)
        self.sandbox = sandbox
        self.clock = clock or c.WallClock()
        self.event_sink = event_sink
        self.max_concurrent, self.max_queued = int(max_concurrent), int(max_queued)
        self.max_deadline_s = float(max_deadline_s)
        self.tail_bytes = int(tail_bytes)
        self.manager_id = c.new_id("jm")
        self._lock = threading.RLock()
        self._records: Dict[str, JobRecord] = {}
        self._live: Dict[str, _Live] = {}
        self._queue: Deque[str] = collections.deque()
        self._running = 0
        self.peak_running = 0
        self._closed = False
        self._events_path = os.path.join(self.dir, "events.jsonl")
        self._events_lock = threading.Lock()
        self.interrupted_on_open: List[str] = []
        self._recover()

    # ------------------------------------------------------------------------------------------ persistence
    def _path(self, job_id: str, name: str = "record.json") -> str:
        return os.path.join(self.dir, job_id, name)

    def _save(self, rec: JobRecord) -> None:
        d = os.path.join(self.dir, rec.job_id)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".record.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(rec.to_dict(), sort_keys=True, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(d, "record.json"))

    def _load(self, job_id: str) -> Optional[JobRecord]:
        try:
            with open(self._path(job_id)) as f:
                return JobRecord.from_dict(json.load(f))
        except (OSError, ValueError, TypeError):
            return None

    def _recover(self) -> None:
        """Load every record of the run directory; jobs that were live under another manager are interrupted."""
        for name in sorted(os.listdir(self.dir)):
            if not os.path.isdir(os.path.join(self.dir, name)):
                continue
            rec = self._load(name)
            if rec is None:
                continue
            if rec.state in JobState.LIVE and rec.manager_id != self.manager_id:
                was = rec.state
                killed = False
                if rec.pgid and rec.pgid_starttime and _starttime(rec.pgid) == rec.pgid_starttime:
                    try:
                        os.killpg(rec.pgid, signal.SIGKILL)       # the recorded leader is still the same process
                        killed = True
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                rec.state = JobState.INTERRUPTED
                rec.ended = self.clock.now()
                rec.error = (f"the process that owned this job stopped while it was {was}; "
                             + ("its remaining processes were killed" if killed else "no process of it was left"))
                rec.stdout_tail = _tail(self._path(name, "stdout.log"), self.tail_bytes)
                rec.stderr_tail = _tail(self._path(name, "stderr.log"), self.tail_bytes)
                self._save(rec)
                self.interrupted_on_open.append(rec.job_id)
                self._emit(rec)
            self._records[rec.job_id] = rec

    # ------------------------------------------------------------------------------------------ events
    def _emit(self, rec: JobRecord) -> None:
        now = self.clock.now()
        text = (f"background job {rec.job_id} {rec.state}; purpose: {rec.purpose}; kind={rec.kind} "
                f"exit_code={rec.exit_code} wall_s={rec.wall_s:.3f}"
                + (f" limit={rec.limit_hit}" if rec.limit_hit else "") + (f" error={rec.error}" if rec.error else ""))
        if rec.stdout_tail:
            text += f"\n--- stdout (tail) ---\n{rec.stdout_tail}"
        if rec.stderr_tail:
            text += f"\n--- stderr (tail) ---\n{rec.stderr_tail}"
        q = [c.Quantity("job.wall_s", float(rec.wall_s), "s"), c.Quantity("job.cpu_s", float(rec.cpu_s), "s")]
        if rec.exit_code is not None:
            q.append(c.Quantity("job.exit_code", float(rec.exit_code), "", c.OutcomeKind.COUNT))
        obs = c.Observation(
            obs_id=c.new_id("obs"), ts=now, kind="event", text=text[:8000], quantities=q,
            refs={"job": rec.job_id, "record": self._path(rec.job_id)},
            provenance=c.Provenance(source="sandbox.jobs", uri=f"job:{rec.job_id}", retrieved_at=now,
                                    trust=c.Trust.UNTRUSTED,
                                    content_sha256=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()),
            parent_action_id=rec.action_id, data={"event": "job_finished", "job": rec.to_dict()})
        with self._events_lock:
            with open(self._events_path, "a") as f:
                f.write(obs.to_json() + "\n")
                f.flush()
                os.fsync(f.fileno())
        sink = self.event_sink
        if callable(sink):
            try:
                sink(obs)
            except Exception:  # noqa: BLE001 - a failing consumer must not lose the job; the event is on disk
                pass

    def events(self, cursor: str = "") -> Tuple[List[c.Observation], str]:
        """Finished-job events after ``cursor`` (a position returned by an earlier call; '' = from the start)."""
        start = int(cursor) if cursor else 0
        out: List[c.Observation] = []
        with self._events_lock:
            try:
                with open(self._events_path) as f:
                    lines = f.readlines()
            except OSError:
                lines = []
        for line in lines[start:]:
            try:
                out.append(c.Observation.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                continue
        return out, str(len(lines))

    # ------------------------------------------------------------------------------------------ submission
    def _admit(self, rec: JobRecord, live: _Live) -> JobRecord:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is closed")
            self._records[rec.job_id] = rec
            self._live[rec.job_id] = live
            if self._running < self.max_concurrent:
                self._start(rec, live)
            else:
                if len(self._queue) >= self.max_queued:
                    del self._records[rec.job_id], self._live[rec.job_id]
                    raise JobLimitError(f"{self._running} jobs running and {len(self._queue)} queued: queue is full")
                self._queue.append(rec.job_id)
                self._save(rec)
            return replace(rec)

    def _start(self, rec: JobRecord, live: _Live) -> None:
        self._running += 1
        self.peak_running = max(self.peak_running, self._running)
        rec.state, rec.started = JobState.RUNNING, self.clock.now()
        self._save(rec)
        live.thread = threading.Thread(target=self._guard, args=(rec, live), name=f"cam-job-{rec.job_id}", daemon=True)
        live.thread.start()

    def _guard(self, rec: JobRecord, live: _Live) -> None:
        try:
            live.start()
        except BaseException as e:  # noqa: BLE001 - whatever happened is recorded on the job
            with self._lock:
                rec.state, rec.error = JobState.FAILED, f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                if rec.state in JobState.LIVE:
                    rec.state, rec.error = JobState.FAILED, rec.error or "job ended without a result"
                rec.ended = self.clock.now()
                self._save(rec)
                self._running -= 1
                nxt: Optional[str] = None
                while self._queue and not self._closed:
                    cand = self._queue.popleft()
                    if self._records[cand].state == JobState.QUEUED:
                        nxt = cand
                        break
                if nxt is not None:
                    self._start(self._records[nxt], self._live[nxt])
            self._emit(rec)
            live.done.set()

    def _new(self, purpose: str, kind: str, deadline_s: float, action_id: str) -> JobRecord:
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("a job needs a purpose")
        if not deadline_s or deadline_s <= 0:
            raise ValueError("a job needs a positive deadline")
        return JobRecord(job_id=c.new_id("job"), purpose=purpose.strip()[:2000], kind=kind,
                         created=self.clock.now(), deadline_s=min(float(deadline_s), self.max_deadline_s),
                         action_id=action_id, manager_id=self.manager_id)

    def submit(self, purpose: str, kind: str, code: str, *, deadline_s: float,
               limits: Optional[SandboxLimits] = None, network: str = "none", args: Sequence[str] = (),
               cwd: str = ".", env: Optional[Dict[str, str]] = None, action_id: str = "",
               broker_factory: Optional[Callable[[], EgressBroker]] = None) -> JobRecord:
        """Queue generated code as a sandboxed background job. Returns a snapshot of its record at once."""
        if self.sandbox is None or not self.sandbox.available:
            raise RuntimeError("no sandbox available for background jobs"
                               + (f": {self.sandbox.unavailable_reason}" if self.sandbox is not None else ""))
        if kind not in ("python", "shell"):
            raise ValueError("kind must be 'python' or 'shell'")
        if network == "scoped" and broker_factory is None:
            raise ValueError("a scoped-network job needs an egress broker")
        rec = self._new(purpose, kind, deadline_s, action_id)
        lim = replace(limits or self.sandbox.cfg.limits, wall_s=rec.deadline_s)
        rec.limits, rec.network = asdict(lim), network
        rec.source_sha256 = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
        sb = self.sandbox
        if kind == "python":
            argv, rec.source_file = sb.python_argv(code, args, label=rec.job_id)
        else:
            argv, rec.source_file = sb.shell_argv(code, label=rec.job_id)
        live = _Live()
        job_dir = os.path.join(self.dir, rec.job_id)
        os.makedirs(job_dir, exist_ok=True)

        def start() -> None:
            if live.cancel.is_set():
                rec.state = JobState.CANCELLED
                return
            broker = broker_factory() if network == "scoped" else None
            try:
                proc = sb.spawn(argv, limits=lim, network=network, broker=broker, env=env, cwd=cwd,
                                stdout_path=os.path.join(job_dir, "stdout.log"),
                                stderr_path=os.path.join(job_dir, "stderr.log"))
            except BaseException:
                if broker is not None:
                    broker.close()
                raise
            waited = time.monotonic() + 15.0               # the monitor reports the process group at once
            while not proc.pgid and proc.running() and time.monotonic() < waited:
                time.sleep(0.01)
            with self._lock:
                rec.pgid = int(proc.pgid or 0)
                rec.pgid_starttime = _starttime(rec.pgid) if rec.pgid else 0
                self._save(rec)
            res = proc.wait(cancel=live.cancel)
            with self._lock:
                self._finish_sandboxed(rec, res, broker, job_dir)

        live.start = start
        return self._admit(rec, live)

    def _finish_sandboxed(self, rec: JobRecord, res: RunResult, broker: Optional[EgressBroker], job_dir: str) -> None:
        rec.exit_code, rec.signal = res.exit_code, res.signal
        rec.killed_for, rec.limit_hit = res.killed_for or "", res.limit_hit
        rec.wall_s, rec.cpu_s, rec.max_rss_kb = float(res.wall_s), float(res.cpu_s), int(res.max_rss_kb)
        rec.peak_tasks, rec.leftover_processes = int(res.peak_tasks), int(res.leftover)
        rec.stdout_bytes, rec.stderr_bytes = res.stdout_bytes, res.stderr_bytes
        rec.stdout_tail = _tail(os.path.join(job_dir, "stdout.log"), self.tail_bytes)
        rec.stderr_tail = _tail(os.path.join(job_dir, "stderr.log"), self.tail_bytes)
        rec.connections = [dict(e) for e in broker.log] if broker is not None else []
        if res.setup_error:
            rec.state, rec.error = JobState.FAILED, f"sandbox setup: {res.setup_error}"
        elif res.cancelled:
            rec.state = JobState.CANCELLED
        elif res.timed_out:
            rec.state, rec.error = JobState.TIMEOUT, f"deadline of {rec.deadline_s:g}s reached"
        elif res.exit_code == 0:
            rec.state = JobState.SUCCEEDED
        else:
            rec.state = JobState.FAILED
            rec.error = (f"{rec.limit_hit} limit reached" if rec.limit_hit else
                         f"exit code {res.exit_code}" if res.exit_code is not None else f"signal {res.signal}")

    def submit_callable(self, purpose: str, fn: Callable[[threading.Event], Any], *, deadline_s: float,
                        action_id: str = "") -> JobRecord:
        """Run ``fn(cancel_event)`` of the harness itself as a job. ``cancel_event`` is set on cancellation and at
        the deadline; the function must return a JSON-serialisable value. Limits are cooperative (a thread
        cannot be killed): a late result is discarded and the job is recorded as ``timeout``."""
        if not callable(fn):
            raise ValueError("fn must be callable")
        rec = self._new(purpose, "callable", deadline_s, action_id)
        rec.limits = {"wall_s": rec.deadline_s, "enforcement": "cooperative"}
        live = _Live()

        def start() -> None:
            if live.cancel.is_set():
                rec.state = JobState.CANCELLED
                return
            box: Dict[str, Any] = {}
            stop = threading.Event()                       # what the function sees: cancellation or deadline

            def work() -> None:
                try:
                    box["value"] = fn(stop)
                except BaseException as e:  # noqa: BLE001 - recorded on the job
                    box["error"] = f"{type(e).__name__}: {e}"

            t0 = time.monotonic()
            worker = threading.Thread(target=work, name=f"cam-job-fn-{rec.job_id}", daemon=True)
            worker.start()
            timed_out = False
            while worker.is_alive():
                worker.join(0.02)
                if live.cancel.is_set():
                    stop.set()
                if time.monotonic() - t0 >= rec.deadline_s:
                    timed_out = True
                    stop.set()
                    break
            if timed_out or live.cancel.is_set():
                worker.join(2.0)                           # grace for a cooperative stop; the result is not used
            with self._lock:
                rec.wall_s = time.monotonic() - t0
                if timed_out:
                    rec.state, rec.error = JobState.TIMEOUT, f"deadline of {rec.deadline_s:g}s reached"
                    if worker.is_alive():
                        rec.error += "; the function had not returned when it was abandoned"
                elif live.cancel.is_set():
                    rec.state = JobState.CANCELLED
                elif "error" in box:
                    rec.state, rec.error = JobState.FAILED, box["error"]
                else:
                    try:
                        json.dumps(box.get("value"))
                        rec.result, rec.state = box.get("value"), JobState.SUCCEEDED
                    except (TypeError, ValueError):
                        rec.state, rec.error = JobState.FAILED, "the job returned a value that is not JSON"

        live.start = start
        return self._admit(rec, live)

    # ------------------------------------------------------------------------------------------ observation
    def status(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError(f"unknown job {job_id!r}")
            snap = replace(rec)
        if snap.state == JobState.RUNNING:
            snap.wall_s = max(0.0, self.clock.now() - snap.started)
            snap.stdout_tail = _tail(self._path(job_id, "stdout.log"), self.tail_bytes)
            snap.stderr_tail = _tail(self._path(job_id, "stderr.log"), self.tail_bytes)
        return snap

    def list(self) -> List[JobRecord]:
        with self._lock:
            return [replace(r) for r in sorted(self._records.values(), key=lambda r: (r.created, r.job_id))]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.list():
            out[r.state] = out.get(r.state, 0) + 1
        return out

    def result(self, job_id: str, tail_bytes: Optional[int] = None) -> Dict[str, Any]:
        """The persisted record (read back from disk) with output tails."""
        self.status(job_id)
        rec = self._load(job_id)
        if rec is None:
            raise KeyError(f"no record on disk for job {job_id!r}")
        n = int(tail_bytes or self.tail_bytes)
        d = rec.to_dict()
        d["stdout_tail"] = _tail(self._path(job_id, "stdout.log"), n) or rec.stdout_tail
        d["stderr_tail"] = _tail(self._path(job_id, "stderr.log"), n) or rec.stderr_tail
        d["record_path"] = self._path(job_id)
        return d

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self._records.get(job_id)
            if rec is None:
                raise KeyError(f"unknown job {job_id!r}")
            live = self._live.get(job_id)
            if rec.done or live is None:
                return replace(rec)
            rec.cancel_requested = True
            live.cancel.set()
            if rec.state == JobState.QUEUED:
                rec.state, rec.ended = JobState.CANCELLED, self.clock.now()
                self._save(rec)
                emit = True
            else:
                self._save(rec)
                emit = False
        if emit:
            self._emit(rec)
            live.done.set()
        return self.status(job_id)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> JobRecord:
        with self._lock:
            live = self._live.get(job_id)
        if live is not None:
            live.done.wait(timeout)
        return self.status(job_id)

    def close(self, timeout: float = 30.0) -> None:
        """Cancel everything that is live and wait for it (each cancelled job is recorded and reported)."""
        with self._lock:
            self._closed = True
            ids = [j for j, r in self._records.items() if r.state in JobState.LIVE and j in self._live]
        for j in ids:
            self.cancel(j)
        end = time.monotonic() + timeout
        for j in ids:
            self._live[j].done.wait(max(0.0, end - time.monotonic()))


# ==============================================================================================================
# Tool handlers used by SandboxExecutor (sandbox.job_*)
# ==============================================================================================================
def _job_text(d: Dict[str, Any]) -> str:
    return (f"job {d['job_id']} state={d['state']} kind={d['kind']} purpose={d['purpose']!r} "
            f"deadline_s={d['deadline_s']:g} exit_code={d['exit_code']} wall_s={d['wall_s']:.3f}"
            + (f" error={d['error']}" if d.get("error") else ""))


def execute_job_tool(executor: Any, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float) -> c.ActionResult:
    a, name = action.arguments, action.operation
    mgr: JobManager = executor.jobs(ctx)
    op = c.PrimitiveOp(op=name, modality=executor.modality, started=t0,
                       arguments={k: a[k] for k in ("job_id", "purpose", "kind", "deadline_s", "network") if k in a})
    if "code" in a:
        op.arguments["sha256"] = hashlib.sha256(a["code"].encode("utf-8", "replace")).hexdigest()
    verdict, reason = _authorize(ctx, op, action)
    if verdict != c.AuthVerdict.ALLOW:
        return executor._denied(action, ctx, t0, m0, verdict, reason, op)

    def done(d: Dict[str, Any], text: str, artifacts: Optional[Dict[str, str]] = None) -> c.ActionResult:
        op.ended = ctx.clock.now()
        obs = executor._obs(action, ctx, "tool_result", _scrub(ctx, text), d, uri=f"job:{d.get('job_id', '')}")
        return executor._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=obs, ops=[op], artifacts=artifacts)

    def fail(cls: str, detail: str) -> c.ActionResult:
        op.ended, op.status, op.failure_class, op.note = ctx.clock.now(), c.ResultStatus.FAILED, cls, detail
        return executor._result(action, ctx, t0, m0, c.ResultStatus.FAILED, cls, detail, ops=[op])

    try:
        if name == "sandbox.job_start":
            limits = executor._limits(a, float(a["deadline_s"]), mgr.max_deadline_s)
            ops: List[c.PrimitiveOp] = []
            lock = threading.Lock()
            network = a.get("network", "none")
            factory = (lambda: executor.make_broker(action, ctx, ops, lock)) if network == "scoped" else None
            executor.sandbox._check_quota()
            rec = mgr.submit(a["purpose"], a["kind"], a["code"], deadline_s=float(a["deadline_s"]), limits=limits,
                             network=network, args=a.get("args", ()), cwd=a.get("cwd", "."), env=a.get("env"),
                             action_id=action.action_id, broker_factory=factory)
            d = rec.to_dict()
            return done(d, _job_text(d), {"job": rec.job_id})
        if name == "sandbox.job_status":
            if a.get("job_id"):
                d = mgr.status(a["job_id"]).to_dict()
                return done(d, _job_text(d))
            jobs = [r.to_dict() for r in mgr.list()]
            d = {"jobs": jobs, "counts": mgr.counts(), "max_concurrent": mgr.max_concurrent}
            return done(d, "\n".join(_job_text(j) for j in jobs) or "no jobs")
        if name == "sandbox.job_result":
            d = mgr.result(a["job_id"], a.get("tail_bytes"))
            text = _job_text(d)
            if d.get("stdout_tail"):
                text += f"\n--- stdout (tail) ---\n{d['stdout_tail']}"
            if d.get("stderr_tail"):
                text += f"\n--- stderr (tail) ---\n{d['stderr_tail']}"
            return done(d, text)
        if name == "sandbox.job_cancel":
            d = mgr.cancel(a["job_id"]).to_dict()
            return done(d, _job_text(d))
    except KeyError as e:
        return fail(c.FailureClass.INVALID_ARGUMENTS, str(e).strip("'\""))
    except JobLimitError as e:
        return fail(c.FailureClass.RESOURCE_LIMIT, str(e))
    return fail(c.FailureClass.INVALID_ARGUMENTS, f"unknown job operation {name!r}")
