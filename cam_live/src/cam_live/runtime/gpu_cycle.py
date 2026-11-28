"""cam_live.runtime.gpu_cycle — a long run on a GPU shared with another tenant (INT3B, master item C).

The machine's heavy-GPU convention is one flock'ed lock file (``config.GPU_LOCK_PATH``, also used by
``nn.gpu.gpu_lock``) held for at most ~15 minutes at a time. A live run lasts hours, so the learner process does not
hold the lock for its lifetime; it *leases* it:

* the lock is taken (blocking) before the agent is built, and again before every cycle with work, if it is not held;
* a cycle never starts once the hold would exceed ``runtime.gpu_hold_s`` (default 840 s when a pretrained backbone
  runs on CUDA): at that decision boundary the runtime commits its checkpoint, releases the lock, yields for
  ``runtime.gpu_yield_s`` (so that tenants polling the lock get their turn), re-acquires it blocking and continues —
  no exit, no manual resume. The model stays resident in GPU memory meanwhile (its cache is emptied);
* the lock is released whenever the runtime goes to sleep (waiting for outcomes, paused, finished): sleeping never
  holds the GPU;
* while it waits for the lock the runtime's activity is ``waiting_for_gpu_lock`` (``state.json`` / ``cam-live
  status`` / dashboard), operator commands are still applied (a stop or pause ends the wait), and the waiting time is
  not charged to the task's compute limit (only learner calls are);
* every release / acquisition is a ``gpu_lock`` event in ``events.jsonl``; totals are in ``state.json["gpu"]``.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

DEFAULT_HOLD_S = 840.0
DEFAULT_YIELD_S = 15.0


class GpuLease:
    """The shared lock file, held at most ``hold_s`` seconds at a time (0 = unbounded)."""

    def __init__(self, path: str = "", hold_s: float = DEFAULT_HOLD_S, *, yield_s: float = DEFAULT_YIELD_S,
                 poll_s: float = 2.0, release_sleep_s: float = 5.0):
        from ..config import GPU_LOCK_PATH
        self.path = path or GPU_LOCK_PATH
        self.hold_s, self.yield_s, self.poll_s = float(hold_s or 0.0), float(yield_s), max(0.05, float(poll_s))
        self.release_sleep_s = float(release_sleep_s)
        self._fh: Any = None
        self.held_since: Optional[float] = None
        self.waiting_since: Optional[float] = None
        self.last_cycle_s = 0.0
        self.stats: Dict[str, Any] = {"acquisitions": 0, "releases": 0, "waited_s_total": 0.0, "held_s_total": 0.0,
                                      "longest_hold_s": 0.0, "yields": 0}

    @property
    def held(self) -> bool:
        return self._fh is not None

    def held_for(self) -> float:
        return 0.0 if self.held_since is None else time.time() - self.held_since

    def acquire(self, should_abort: Optional[Callable[[], bool]] = None) -> Optional[float]:
        """Block until the lock is held (returns the seconds waited) or ``should_abort()`` says to stop waiting
        (returns None). The wait is a *blocking* ``flock`` in a helper thread — a queued waiter the kernel wakes when
        the holder releases, so a tenant that releases and immediately re-polls does not keep the lock forever —
        while this thread stays responsive (``should_abort`` every ``poll_s``)."""
        if self.held:
            return 0.0
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        from ..nn.gpu import open_lock_file
        fh = open_lock_file(self.path)
        t0 = time.time()
        got = threading.Event()
        state = {"abandoned": False}
        lock = threading.Lock()

        def wait() -> None:
            fcntl.flock(fh, fcntl.LOCK_EX)
            with lock:
                if state["abandoned"]:                  # nobody wants it any more: give it straight back
                    fcntl.flock(fh, fcntl.LOCK_UN)
                    fh.close()
                    return
                got.set()

        threading.Thread(target=wait, name="gpu-lock-wait", daemon=True).start()
        self.waiting_since = t0
        try:
            while not got.wait(self.poll_s):
                if should_abort is not None and should_abort():
                    with lock:
                        if not got.is_set():
                            state["abandoned"] = True
                            return None
                    break
        finally:
            self.waiting_since = None
        self._fh, self.held_since = fh, time.time()
        waited = self.held_since - t0
        self.stats["acquisitions"] += 1
        self.stats["waited_s_total"] += waited
        return waited

    def release(self) -> float:
        """Release the lock (no-op when not held); returns how long it was held."""
        if not self.held:
            return 0.0
        held = self.held_for()
        try:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh, self.held_since = None, None
        self.stats["releases"] += 1
        self.stats["held_s_total"] += held
        self.stats["longest_hold_s"] = max(self.stats["longest_hold_s"], held)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()        # the model stays resident; cached blocks are given back
        except Exception:  # noqa: BLE001
            pass
        return held

    def would_overrun(self) -> bool:
        return self.held and self.hold_s > 0 and self.held_for() + self.last_cycle_s > self.hold_s

    def status(self) -> Dict[str, Any]:
        return {"lock": self.path, "held": self.held, "held_for_s": round(self.held_for(), 1),
                "hold_s": self.hold_s, "yield_s": self.yield_s,
                "waiting_for_s": None if self.waiting_since is None else round(time.time() - self.waiting_since, 1),
                **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()}}


class _LeasedSleeper:
    """The runtime's sleeper, releasing the GPU lock before any sleep longer than ``lease.release_sleep_s``."""

    def __init__(self, inner: Any, lease: GpuLease, rt: Any):
        self.inner, self.lease, self.rt = inner, lease, rt
        self.blocking = inner.blocking
        self.clock = getattr(inner, "clock", None)

    def sleep_until(self, t: Optional[float], interrupted: Callable[[], bool]) -> str:
        if self.lease.held and (t is None or t - self.rt.clock.now() >= self.lease.release_sleep_s):
            held = self.lease.release()
            _event(self.rt, "released", {"why": "the runtime sleeps", "held_s": round(held, 1), "until": t})
        return self.inner.sleep_until(t, interrupted)


def _event(rt: Any, what: str, data: Dict[str, Any]) -> None:
    lease = getattr(rt, "gpu_lease", None)
    rt.gpu = lease.status() if lease is not None else None
    try:
        rt.log.event("gpu_lock", {"what": what, **data, "decisions": rt.counters.get("decisions")})
    except Exception:  # noqa: BLE001 — logging must never stop the run
        pass


def _wait_for_lock(rt: Any, lease: GpuLease, why: str) -> bool:
    """Acquire for ``rt`` (visible as activity ``waiting_for_gpu_lock``); operator commands are applied while
    waiting. False when a command ended the wait (stop / pause): the cycle is not started."""
    from .loop import RUNNING
    prev = dict(rt._activity)
    rt._set_activity("waiting_for_gpu_lock", why)
    rt.gpu = {**lease.status(), "waiting_for_s": 0.0}
    rt._save_state()

    def abort() -> bool:
        # an operator command or a stop ends the wait; it is applied by the next step (its checkpoint then waits for
        # the lock in the learner-call guard: nothing here may itself need the lock)
        rt.gpu = lease.status()
        return bool(rt._stop_requested) or rt._interrupted() or rt.status != RUNNING

    waited = lease.acquire(abort)
    if waited is None:
        _event(rt, "wait_abandoned", {"why": f"run is {rt.status}"})
        rt._save_state()
        return False
    rt._activity = prev
    _event(rt, "acquired", {"why": why, "waited_s": round(waited, 1)})
    return True


def attach(rt: Any, lease: GpuLease) -> None:
    """Install the lease on a ``LiveRuntime``: before every cycle with work the lock is (re)acquired, or released
    and re-acquired at this decision boundary when the hold would exceed ``hold_s``; the sleeper releases it."""
    from .loop import RUNNING
    rt.gpu_lease = lease
    rt.gpu = lease.status()
    rt.sleeper = _LeasedSleeper(rt.sleeper, lease, rt)
    marks: Dict[str, float] = {}

    def before_cycle(r: Any) -> bool:
        if lease.would_overrun():
            r._commit(force=True)                        # the checkpoint first, then the GPU is given back
            held = lease.release()
            lease.stats["yields"] += 1
            _event(r, "released", {"why": "the bounded hold would be exceeded by the next cycle",
                                   "held_s": round(held, 1), "hold_s": lease.hold_s})
            end = time.time() + lease.yield_s
            while time.time() < end and not r._interrupted():
                time.sleep(min(0.5, max(0.0, end - time.time())))
        if not lease.held and not _wait_for_lock(r, lease, "before the next cycle"):
            return False
        if r.status != RUNNING:
            return False
        marks["t0"] = time.time()
        r.gpu = lease.status()
        return True

    def after_cycle(r: Any) -> None:
        if "t0" in marks:
            lease.last_cycle_s = time.time() - marks.pop("t0")
        r.gpu = lease.status()

    rt.hooks.setdefault("before_cycle", []).append(before_cycle)
    rt.hooks.setdefault("after_cycle", []).append(after_cycle)
    longest: Dict[str, float] = {}

    def guard(name: str) -> None:
        """G2R-9: before every learner call (also the checkpoint of a pause / stop / export applied after a sleep,
        which used to run without the lock): hold the lock, and yield first when this call — at its longest
        measured duration — would take the hold past ``hold_s``. A hold is therefore bounded by ``hold_s`` plus at
        most one learner call longer than any of its kind before (each call is bounded by its own budget)."""
        if lease.held and lease.hold_s > 0 and lease.held_for() + longest.get(name, 0.0) > lease.hold_s:
            held = lease.release()
            lease.stats["yields"] += 1
            _event(rt, "released", {"why": f"the bounded hold would be exceeded by the next learner call ({name})",
                                    "held_s": round(held, 1), "hold_s": lease.hold_s})
            time.sleep(max(0.0, lease.yield_s))
        if not lease.held:
            prev = dict(getattr(rt, "_activity", {}) or {})
            try:
                rt._set_activity("waiting_for_gpu_lock", f"before the learner call {name}")
                rt.gpu = {**lease.status(), "waiting_for_s": 0.0}
                rt._save_state()
            except Exception:  # noqa: BLE001 — before the runtime has a state: just wait
                pass
            waited = lease.acquire()          # not abandoned: this learner work (e.g. a stop's checkpoint) needs it
            if prev:
                rt._activity = prev
            _event(rt, "acquired", {"why": f"before the learner call {name}", "waited_s": round(waited or 0.0, 1)})

    def timing(name: str, dt: float) -> None:
        longest[name] = max(longest.get(name, 0.0) * 0.98, dt)       # slowly forgets an outlier

    rt.learner_guard = guard
    rt.learner_timing = timing


def lease_settings(cfg: Any) -> Dict[str, float]:
    sec = cfg.section("runtime") if hasattr(cfg, "section") else dict(cfg or {})
    hold = sec.get("gpu_hold_s")
    return {"hold_s": DEFAULT_HOLD_S if hold is None else float(hold),
            "yield_s": float(sec.get("gpu_yield_s", DEFAULT_YIELD_S)),
            "poll_s": float(sec.get("gpu_poll_s", 2.0))}

