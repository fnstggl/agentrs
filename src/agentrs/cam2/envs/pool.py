"""Asynchronous environment workers: one adapter per subprocess, stepped concurrently.

The learner process holds the GPU models; each worker owns one environment (e.g. one headless Chromium). Commands
are sent to many workers at once and results are collected as they arrive, so slow environments do not block
fast ones. Workers are started with the ``spawn`` method (no CUDA state is forked).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait
from typing import Any, Optional


def make_adapter(kind: str, kwargs: dict):
    if kind == "miniwob":
        from agentrs.cam2.envs.miniwob import MiniWoBAdapter

        return MiniWoBAdapter(**kwargs)
    if kind == "appworld":
        from agentrs.cam2.envs.appworld import AppWorldAdapter

        return AppWorldAdapter(**kwargs)
    if kind == "fixture":
        from agentrs.cam2.envs.fixture import LookupFixtureAdapter

        return LookupFixtureAdapter(**kwargs)
    if kind == "crucible":
        from agentrs.cam2.envs.crucible_env import CrucibleAdapter

        return CrucibleAdapter(**kwargs)
    from agentrs.cam2.envs.unavailable import make_unavailable

    return make_unavailable(kind, kwargs)


def _worker(conn, kind: str, kwargs: dict, wid: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.setpgrp()  # own process group: a hung worker is killed together with its browser processes
    try:
        env = make_adapter(kind, kwargs)
        conn.send(("ready", env.describe()))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            cmd, arg = conn.recv()
        except EOFError:
            break
        try:
            if cmd == "reset":
                obs = env.reset(arg)
                conn.send(("ok", (obs, env.available_tools())))
            elif cmd == "step":
                conn.send(("ok", env.step(arg)))
            elif cmd == "evaluate":
                conn.send(("ok", env.evaluate()))
            elif cmd == "snapshot":
                conn.send(("ok", env.snapshot()))
            elif cmd == "restore":
                conn.send(("ok", env.restore(arg)))
            elif cmd == "close":
                env.close()
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command {cmd}"))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))


class EnvPool:
    """``n`` workers. Every command has a deadline (``cmd_timeout``); a worker that misses it is killed and replaced
    by a fresh one, and the command returns ``("error", "timeout ...")`` so the caller can drop that episode."""

    def __init__(self, kind: str, n: int, kwargs: Optional[dict] = None, start_timeout: float = 180.0,
                 cmd_timeout: float = 90.0):
        self.ctx = mp.get_context("spawn")
        self.kind, self.n, self.kwargs = kind, n, kwargs or {}
        self.start_timeout, self.cmd_timeout = start_timeout, cmd_timeout
        self.conns: list = [None] * n
        self.procs: list = [None] * n
        self.desc: list = [None] * n
        self.restarts = 0
        for i in range(n):
            self._spawn(i)
        for i in range(n):
            self._await_ready(i)
        self.pending: dict = {}
        self.sent_at: dict = {}

    THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

    def _spawn(self, i: int) -> None:
        # Workers import numpy/torch while unpickling; without these caps every worker starts one BLAS/OpenMP thread
        # per core (48 here) that it never uses - ~1000 wasted pids for 22 workers under a 2816-pid container limit.
        # The variables must be set before the child starts, so they are set around ``start()`` and then restored.
        saved = {k: os.environ.get(k) for k in self.THREAD_VARS}
        for k in self.THREAD_VARS:
            os.environ[k] = "1"
        try:
            a, b = self.ctx.Pipe()
            p = self.ctx.Process(target=_worker, args=(b, self.kind, self.kwargs, i), daemon=True)
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.conns[i], self.procs[i] = a, p

    def _await_ready(self, i: int) -> None:
        c = self.conns[i]
        if not c.poll(self.start_timeout):
            self.close()
            raise RuntimeError(f"{self.kind} worker did not start within {self.start_timeout}s")
        st, v = c.recv()
        if st != "ready":
            self.close()
            from agentrs.cam2.envs.base import DependencyUnavailable

            raise DependencyUnavailable(v)
        self.desc[i] = v

    def _restart(self, i: int) -> None:
        import signal

        try:
            os.killpg(self.procs[i].pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.procs[i].kill()
            self.procs[i].join(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self.restarts += 1
        self._spawn(i)
        self._await_ready(i)

    def send(self, i: int, cmd: str, arg: Any = None) -> None:
        assert i not in self.pending, f"worker {i} busy"
        self.conns[i].send((cmd, arg))
        self.pending[i] = cmd
        self.sent_at[i] = time.time()

    def poll(self, timeout: Optional[float] = None) -> list:
        """Results that are ready: [(worker, cmd, status, value)]; blocks up to ``timeout`` for at least one."""
        if not self.pending:
            return []
        busy = {self.conns[i]: i for i in self.pending}
        now = time.time()
        first_deadline = min(self.sent_at[i] + self.cmd_timeout for i in self.pending)
        t = max(0.0, first_deadline - now) if timeout is None else max(0.0, min(timeout, first_deadline - now))
        ready = wait(list(busy), t)
        out = []
        for c in ready:
            i = busy[c]
            try:
                st, v = c.recv()
            except (EOFError, OSError) as e:
                st, v = "error", f"worker died: {e!r}"
                self._restart(i)
            out.append((i, self.pending.pop(i), st, v))
            self.sent_at.pop(i, None)
        now = time.time()
        for i in [i for i in self.pending if now - self.sent_at[i] > self.cmd_timeout]:
            cmd = self.pending.pop(i)
            self.sent_at.pop(i, None)
            self._restart(i)
            out.append((i, cmd, "error", f"timeout after {self.cmd_timeout}s (worker restarted)"))
        return out

    def call(self, i: int, cmd: str, arg: Any = None, timeout: float = 300.0):
        self.send(i, cmd, arg)
        t0 = time.time()
        while True:
            for j, _c, st, v in self.poll(timeout=max(0.1, timeout - (time.time() - t0))):
                if j == i:
                    if st != "ok":
                        raise RuntimeError(v)
                    return v
            if time.time() - t0 > timeout:
                raise TimeoutError(f"worker {i} {cmd}")

    def close(self) -> None:
        import signal

        pending = getattr(self, "pending", {})
        for i, c in enumerate(self.conns):
            try:
                if c is not None and i not in pending:
                    c.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
        deadline = time.time() + 15
        for i, p in enumerate(self.procs):
            if p is None:
                continue
            if i not in pending:
                p.join(timeout=max(0.1, deadline - time.time()))
            if p.is_alive():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
                p.kill()
                p.join(timeout=5)
