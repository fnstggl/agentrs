"""Persistent asynchronous self-play workers (see package docstring)."""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from agentrs.econ.core import Task
from agentrs.econ.rollout import Episode, Step, finish, make_env, render
from agentrs.selfplay.engine import InferenceEngine


@dataclass
class WorkUnit:
    task: Task
    group: Optional[str] = None


class Collector:
    """Runs work units on ``n_workers`` logical workers that share one engine.

    Isolation: every episode builds its own environment instance (own SQLite DB / repo / app) and
    keeps its own history; nothing mutable is shared between workers. Failures are contained: an
    environment that fails to build, a step that raises or exceeds ``step_timeout_s`` and an episode
    over ``episode_timeout_s`` end only that episode (recorded with an error, no reward), and the
    worker continues with the next unit. Generation errors are retried ``gen_retries`` times.
    """

    def __init__(self, engine: InferenceEngine, n_workers: int,
                 step_timeout_s: float = 120.0, episode_timeout_s: float = 1800.0, gen_retries: int = 2,
                 greedy: bool = False, provenance: str = "agent_policy",
                 sink: Optional[Path] = None, on_episode: Optional[Callable[[Episode], None]] = None):
        self.engine = engine
        self.n_workers = n_workers
        # one dedicated environment thread per worker: environment objects (e.g. SQLite connections) are
        # bound to the thread that created them, and a hung step only ever blocks its own worker
        self._env_threads: dict[int, cf.ThreadPoolExecutor] = {}
        self.step_timeout_s = step_timeout_s
        self.episode_timeout_s = episode_timeout_s
        self.gen_retries = gen_retries
        self.greedy = greedy
        self.provenance = provenance
        self.sink = sink
        self.on_episode = on_episode
        self.env_time_s = 0.0
        self.actions = 0
        self.active_peak = 0
        self._active = 0
        self.active_trace: list = []  # (time, active workers)

    # ------------------------------------------------------------------ durable sink
    def _record(self, ep: Episode) -> None:
        if self.sink is None:
            return
        rec = ep.to_record()
        for s in rec["steps"]:
            s.pop("messages", None)
        line = json.dumps(rec, default=str) + "\n"
        with open(self.sink, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------ one episode
    def _pool(self, wid: int) -> cf.ThreadPoolExecutor:
        if wid not in self._env_threads:
            self._env_threads[wid] = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"env-w{wid}")
        return self._env_threads[wid]

    async def _env_call(self, wid: int, fn, *args, timeout: Optional[float] = None):
        loop = asyncio.get_running_loop()
        t0 = time.time()
        try:
            return await asyncio.wait_for(loop.run_in_executor(self._pool(wid), fn, *args), timeout=timeout)
        except asyncio.TimeoutError:
            # the stuck call keeps its thread; give this worker a fresh one for its next episode
            self._env_threads.pop(wid).shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            self.env_time_s += time.time() - t0

    async def episode(self, unit: WorkUnit, wid: int) -> Episode:
        t = unit.task
        version = self.engine.version
        ep = Episode(t.id, t.domain, t.family, t.split, t.seed, t.goal, self.provenance, version, group=unit.group)
        t0 = time.time()
        try:
            env = await self._env_call(wid, make_env, t, timeout=self.step_timeout_s)
        except Exception as e:  # noqa: BLE001
            ep.verification = {"error": f"env setup failed: {type(e).__name__}: {e}"[:300]}
            ep.wall_s = time.time() - t0
            return ep
        hist: list[tuple[str, str]] = []
        try:
            while not env.done and len(ep.steps) < t.max_steps:
                if time.time() - t0 > self.episode_timeout_s:
                    ep.verification = {"error": "episode timeout"}
                    break
                # rendering reads the environment (e.g. its DB schema): same thread as the environment
                msgs, prefill = await self._env_call(wid, render, env, t, list(hist), timeout=self.step_timeout_s)
                o = None
                for attempt in range(self.gen_retries + 1):
                    try:
                        o = await self.engine.submit(msgs, prefill, greedy=self.greedy)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == self.gen_retries:
                            raise RuntimeError(f"generation failed: {e}") from e
                        await asyncio.sleep(1.0 * (attempt + 1))
                if o["policy_version"] != version:  # the served weights changed mid-episode: not on-policy
                    ep.verification = {"error": "stale: policy version changed during the episode"}
                    break
                action = prefill + o["text"]
                t1 = time.time()
                res = await self._env_call(wid, env.step, action, timeout=self.step_timeout_s)
                self.actions += 1
                ep.steps.append(Step(prompt=o["prompt"], prefill=prefill, action=action, token_ids=o["ids"],
                                     behavior_logprobs=o["logprobs"], result=res.observation, valid=res.valid,
                                     error=res.error, elapsed_s=time.time() - t1, finished_generation=o["finished"],
                                     t_wall=t1))
                hist.append((action, res.observation))
        except asyncio.TimeoutError:
            ep.verification = {"error": f"step timeout (> {self.step_timeout_s}s)"}
        except Exception as e:  # noqa: BLE001 - an environment bug ends this episode only
            ep.verification = {"error": f"step crashed: {type(e).__name__}: {e}"[:300]}
        if "error" not in ep.verification:
            try:
                await self._env_call(wid, finish, ep, env, t0, timeout=self.step_timeout_s)
            except Exception as e:  # noqa: BLE001
                ep.verification = {"error": f"verifier failed: {type(e).__name__}: {e}"[:300]}
                ep.success, ep.reward = False, 0.0
        ep.wall_s = time.time() - t0
        return ep

    # ------------------------------------------------------------------ pool
    async def run(self, units: list[WorkUnit], wall_budget_s: Optional[float] = None,
                  unit_source: Optional[Callable[[], Optional[WorkUnit]]] = None) -> list[Episode]:
        """Run ``units`` (then pull more from ``unit_source`` until it returns None or the wall budget
        is used up). With a wall budget, units that have not finished by then are cancelled and not
        returned (they would be partial episodes)."""
        q: asyncio.Queue = asyncio.Queue()
        for u in units:
            q.put_nowait(u)
        done: list[Episode] = []
        t_end = time.time() + wall_budget_s if wall_budget_s else None

        async def worker(wid: int):
            while True:
                if t_end and time.time() > t_end:
                    return
                try:
                    u = q.get_nowait()
                except asyncio.QueueEmpty:
                    u = unit_source() if unit_source else None
                    if u is None:
                        return
                self._active += 1
                self.active_peak = max(self.active_peak, self._active)
                self.active_trace.append((time.time(), self._active))
                try:
                    ep = await self.episode(u, wid)
                finally:
                    self._active -= 1
                    self.active_trace.append((time.time(), self._active))
                done.append(ep)
                self._record(ep)
                if self.on_episode:
                    self.on_episode(ep)

        tasks = [asyncio.create_task(worker(i)) for i in range(self.n_workers)]
        if t_end:
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, t_end - time.time()))
            for p in pending:
                p.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        else:
            await asyncio.gather(*tasks)
        return done

    def close(self) -> None:
        for ex in self._env_threads.values():
            ex.shutdown(wait=False, cancel_futures=True)


def collect(act_fn: Callable, version: str, units: list[WorkUnit], n_workers: int, max_batch: int = 32,
            max_wait_ms: float = 15.0, greedy: bool = False, sink: Optional[Path] = None,
            provenance: str = "agent_policy", wall_budget_s: Optional[float] = None,
            unit_source: Optional[Callable] = None) -> tuple[list[Episode], dict]:
    """Synchronous entry point used by the trainer: run units on async workers, return episodes + stats."""

    async def main():
        eng = InferenceEngine(act_fn, version, max_batch=max_batch, max_wait_ms=max_wait_ms)
        await eng.start()
        col = Collector(eng, n_workers, greedy=greedy, sink=sink, provenance=provenance)
        t0 = time.time()
        try:
            eps = await col.run(units, wall_budget_s=wall_budget_s, unit_source=unit_source)
        finally:
            await eng.stop()
            col.close()
        wall = time.time() - t0
        st = eng.stats.summary() | {"wall_s": wall, "workers": n_workers, "episodes": len(eps),
                                    "actions": col.actions, "env_time_s": col.env_time_s,
                                    "active_peak": col.active_peak, "successes": sum(e.success for e in eps)}
        return eps, st

    return asyncio.run(main())
