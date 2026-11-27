"""Compute accounting for every CAM Unified process (spec Part 22, manifest X06).

A `ComputeMeter` lives in each process (actor, learner, semantic server, launcher) and records what that process
actually consumed:

* ``wall_s``   — wall-clock seconds since the meter started;
* ``cpu_s``    — user + system CPU seconds of this process and its reaped children (``resource.getrusage``);
* ``gpu_s``    — device time of the sections wrapped in ``with meter.gpu():``, measured with CUDA events on the
                 current stream (elapsed device time between the section's start and end, so idle gaps inside a
                 section count). Timing starts once the process has initialised CUDA; the meter never initialises
                 CUDA itself, so CPU-only processes report 0;
* ``peak_rss_mb`` — peak resident set size of the process (``ru_maxrss``);
* ``peak_gpu_mb`` — ``torch.cuda.max_memory_allocated`` of this process (0 when CUDA was never initialised here);
* counters    — ``env_steps``, ``decisions``, ``tokens_prompt``, ``tokens_generated``, ``sim_days``, ``records``,
                ``updates``, ``samples_trained``, ``env_cost`` (the environment's own money unit, e.g. cents) and any
                other name passed to `add`.

Monetary cost is only reported where a price is known: ``gpu_usd_per_hour`` / ``cpu_usd_per_core_hour`` constructor
arguments turn GPU and CPU seconds into an estimate (``cost_usd``); without a price the field is ``None`` (never a
made-up number).

Snapshots are plain dicts; `ComputeMeter.write` stores one atomically (tmp → fsync → rename) so a dashboard or the
launcher can read the files of live processes at any time, and `merge_snapshots` aggregates them (sums for
consumption, max for per-process peaks, plus the sum of per-process peaks as the host-level upper bound).
"""
from __future__ import annotations

import contextlib
import json
import os
import resource
import socket
import sys
import time
from typing import Iterable, Optional

COUNTERS = ("env_steps", "decisions", "tokens_prompt", "tokens_generated", "sim_days", "records", "updates",
            "samples_trained", "env_cost")


def atomic_write_json(path: str, obj, fsync: bool = True) -> None:
    """Write JSON to ``path`` atomically: ``path.tmp.<pid>`` → flush → fsync → ``os.replace``."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), default=str)
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())
    os.replace(tmp, path)


def _rss_peak_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cpu_s(children: bool = True) -> float:
    s = resource.getrusage(resource.RUSAGE_SELF)
    t = s.ru_utime + s.ru_stime
    if children:
        c = resource.getrusage(resource.RUSAGE_CHILDREN)
        t += c.ru_utime + c.ru_stime
    return t


class ComputeMeter:
    """Per-process consumption meter. Cheap enough to update on every decision (dict increments only)."""

    def __init__(self, name: str, role: str = "", gpu_usd_per_hour: Optional[float] = None,
                 cpu_usd_per_core_hour: Optional[float] = None, include_children: bool = True):
        """include_children: count reaped child processes' CPU (False for a supervisor whose children carry their
        own meters, so merged totals do not double count)."""
        self.include_children = include_children
        self.name = name
        self.role = role
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.t0 = time.time()
        self._wall0 = time.perf_counter()
        self._cpu0 = _cpu_s(include_children)
        self.gpu_usd_per_hour = gpu_usd_per_hour
        self.cpu_usd_per_core_hour = cpu_usd_per_core_hour
        self.counters: dict[str, float] = {k: 0 for k in COUNTERS}
        self.gpu_s = 0.0
        self._gpu_sections = 0
        self._pending: list = []          # (start_event, end_event) not yet synchronised

    # ---------------------------------------------------------------- counting
    def add(self, **counts: float) -> None:
        for k, v in counts.items():
            self.counters[k] = self.counters.get(k, 0) + v

    @contextlib.contextmanager
    def gpu(self):
        """Time the enclosed GPU work with CUDA events (no host synchronisation inside the section; events are
        resolved lazily at the next `snapshot`). Falls back to a no-op when CUDA is unavailable."""
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            # never initialise CUDA just to time a section: CPU-only processes stay CPU-only (gpu_s = 0)
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((start, end))
            self._gpu_sections += 1
            if len(self._pending) > 256:
                self._resolve()

    def _resolve(self) -> None:
        if not self._pending:
            return
        for s, e in self._pending:
            e.synchronize()
            self.gpu_s += s.elapsed_time(e) / 1000.0
        self._pending.clear()

    # ---------------------------------------------------------------- reading
    def snapshot(self) -> dict:
        self._resolve()
        wall = time.perf_counter() - self._wall0
        cpu = _cpu_s(self.include_children) - self._cpu0
        peak_gpu = 0.0
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            peak_gpu = torch.cuda.max_memory_allocated() / 2**20
        cost = None
        if self.gpu_usd_per_hour is not None or self.cpu_usd_per_core_hour is not None:
            cost = (self.gpu_s / 3600.0) * (self.gpu_usd_per_hour or 0.0) + \
                   (cpu / 3600.0) * (self.cpu_usd_per_core_hour or 0.0)
        snap = {"name": self.name, "role": self.role, "pid": self.pid, "host": self.host, "t_start": self.t0,
                "t_snapshot": time.time(), "wall_s": wall, "cpu_s": cpu, "gpu_s": self.gpu_s,
                "gpu_sections": self._gpu_sections, "peak_rss_mb": _rss_peak_mb(), "peak_gpu_mb": peak_gpu,
                "cost_usd": cost}
        snap.update({k: v for k, v in self.counters.items()})
        rates = {}
        if wall > 0:
            for k in ("env_steps", "decisions", "records", "tokens_prompt", "tokens_generated", "sim_days"):
                rates[f"{k}_per_s"] = self.counters.get(k, 0) / wall
        snap["rates"] = rates
        return snap

    def write(self, path: str) -> dict:
        snap = self.snapshot()
        atomic_write_json(path, snap)
        return snap


def read_snapshots(compute_dir: str) -> list:
    """All snapshot files in a run's ``compute/`` directory (a process restarted under the same name keeps one file
    per incarnation: ``<name>.<pid>.json``)."""
    out = []
    if not os.path.isdir(compute_dir):
        return out
    for fn in sorted(os.listdir(compute_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(compute_dir, fn)) as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue          # a file being replaced right now; the next read sees the complete one
    return out


def merge_snapshots(snaps: Iterable[dict]) -> dict:
    """Aggregate per-process snapshots: consumption is summed, peaks are reported per process (max) and summed
    (upper bound of concurrent use). ``wall_s`` is the span from the earliest start to the latest snapshot."""
    snaps = list(snaps)
    tot: dict = {"n_processes": len(snaps), "by_role": {}}
    if not snaps:
        return tot
    keys = set(COUNTERS)
    for s in snaps:
        keys |= {k for k, v in s.items() if isinstance(v, (int, float)) and k not in
                 ("pid", "t_start", "t_snapshot", "wall_s", "peak_rss_mb", "peak_gpu_mb", "gpu_sections")}
    for k in sorted(keys):
        tot[k] = float(sum(float(s.get(k, 0) or 0) for s in snaps))
    costs = [s.get("cost_usd") for s in snaps if s.get("cost_usd") is not None]
    tot["cost_usd"] = float(sum(costs)) if costs else None
    tot["peak_rss_mb_max"] = max(float(s.get("peak_rss_mb", 0)) for s in snaps)
    tot["peak_rss_mb_sum"] = sum(float(s.get("peak_rss_mb", 0)) for s in snaps)
    tot["peak_gpu_mb_max"] = max(float(s.get("peak_gpu_mb", 0)) for s in snaps)
    tot["peak_gpu_mb_sum"] = sum(float(s.get("peak_gpu_mb", 0)) for s in snaps)
    t0 = min(float(s.get("t_start", 0)) for s in snaps)
    t1 = max(float(s.get("t_snapshot", 0)) for s in snaps)
    tot["wall_s"] = max(0.0, t1 - t0)
    for s in snaps:
        r = tot["by_role"].setdefault(s.get("role") or "other", {"n": 0, "cpu_s": 0.0, "gpu_s": 0.0})
        r["n"] += 1
        r["cpu_s"] += float(s.get("cpu_s", 0))
        r["gpu_s"] += float(s.get("gpu_s", 0))
    if tot["wall_s"] > 0:
        tot["decisions_per_s"] = tot.get("decisions", 0.0) / tot["wall_s"]
        tot["env_steps_per_s"] = tot.get("env_steps", 0.0) / tot["wall_s"]
    return tot


# ============================================================================ busy GPU time and semantic tokens
class GpuBusy:
    """Summed CUDA kernel time (seconds) of the code inside ``with GpuBusy() as g:`` from ``torch.profiler``: busy
    device time of THIS process only (``g.busy_s``), unlike a CUDA-event span, which includes idle gaps (plan
    2026-11-26 §5: GPU-busy seconds; audit 'h' X06: gpu_s was the wall span). ``g.span_s`` is the wall span for
    comparison. Without CUDA (or with ``enabled=False``) busy_s stays None. The profiler keeps one record per kernel:
    use it around bounded sections (evaluation runs, smoke runs, sampled windows), not a 12-hour training run."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.busy_s: Optional[float] = None
        self.kernels: Optional[int] = None
        self.span_s = 0.0
        self._prof = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        try:
            import torch
            if self.enabled and torch.cuda.is_available():
                from torch.profiler import ProfilerActivity, profile
                torch.cuda.synchronize()
                self._prof = profile(activities=[ProfilerActivity.CUDA])
                self._prof.__enter__()
        except Exception:                        # profiling unavailable: report None, never a made-up number
            self._prof = None
        return self

    def __exit__(self, *exc):
        if self._prof is not None:
            import torch
            torch.cuda.synchronize()
            self._prof.__exit__(None, None, None)
            kern = [e for e in self._prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")]
            self.busy_s = sum(float(getattr(e, "device_time", 0.0) or 0.0) for e in kern) / 1e6
            self.kernels = len(kern)
        self.span_s = time.perf_counter() - self._t0
        return False


class TokenMeter:
    """Counts the token positions a pretrained semantic model processes (prefill + every decode step) with a forward
    pre-hook on its token-embedding table(s): ``tokens`` = total positions embedded, ``calls`` = forward calls.
    Padding positions are included (they cost compute); the count is of the model's work, not of text length.
    ``attach(module)`` finds every ``nn.Embedding`` with at least ``min_vocab`` rows inside it (Qwen3: 151,936)."""

    def __init__(self, min_vocab: int = 32_000):
        self.min_vocab = int(min_vocab)
        self.tokens = 0
        self.calls = 0
        self._handles: list = []

    def attach(self, module) -> int:
        import torch.nn as nn
        n = 0
        for m in (module.modules() if hasattr(module, "modules") else []):
            if isinstance(m, nn.Embedding) and m.num_embeddings >= self.min_vocab:
                self._handles.append(m.register_forward_pre_hook(self._hook))
                n += 1
        return n

    def _hook(self, mod, args):
        if args:
            self.tokens += int(args[0].numel())
            self.calls += 1

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def snapshot(self) -> dict:
        return {"tokens": self.tokens, "calls": self.calls, "attached": len(self._handles)}
