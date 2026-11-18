"""Shared, dynamically batched inference engine (see package docstring)."""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EngineStats:
    batches: int = 0
    requests: int = 0
    gen_tokens: int = 0
    busy_s: float = 0.0
    batch_sizes: list = field(default_factory=list)
    queue_wait_s: list = field(default_factory=list)   # submit -> batch start
    latency_s: list = field(default_factory=list)      # submit -> result
    started: float = field(default_factory=time.time)

    def summary(self) -> dict:
        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs)
            return s[min(len(s) - 1, int(q * len(s)))]
        wall = max(1e-9, time.time() - self.started)
        return {"batches": self.batches, "requests": self.requests, "gen_tokens": self.gen_tokens,
                "mean_batch": self.requests / self.batches if self.batches else 0.0,
                "max_batch": max(self.batch_sizes) if self.batch_sizes else 0,
                "gen_tok_per_s": self.gen_tokens / wall, "engine_busy_frac": self.busy_s / wall,
                "queue_wait_p50_s": pct(self.queue_wait_s, 0.5), "queue_wait_p95_s": pct(self.queue_wait_s, 0.95),
                "latency_mean_s": sum(self.latency_s) / len(self.latency_s) if self.latency_s else None,
                "latency_p95_s": pct(self.latency_s, 0.95)}


@dataclass
class _Req:
    messages: list
    prefill: str
    greedy: bool
    fut: asyncio.Future
    t_submit: float


class InferenceEngine:
    """``act_fn(reqs: list[(messages, prefill)], greedy: bool) -> list[dict]`` must return, per request,
    ``text``, ``ids`` (exact sampled token ids), ``logprobs`` (behaviour log-probs of those ids under the
    sampling distribution), ``finished`` and ``prompt``. It runs on one dedicated thread (the GPU)."""

    def __init__(self, act_fn: Callable, version: str, max_batch: int = 32, max_wait_ms: float = 15.0,
                 max_queue: int = 512, temperature: float = 1.0):
        self.act_fn = act_fn
        self.version = version
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.temperature = temperature
        self.queue: Optional[asyncio.Queue] = None
        self.max_queue = max_queue
        self.stats = EngineStats()
        self._gpu = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-gpu")
        self._task: Optional[asyncio.Task] = None
        self._inflight = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.max_queue)
        self.stats = EngineStats()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def set_version(self, version: str) -> None:
        """Weights changed (e.g. after a learner update / promotion): only allowed between batches."""
        if self._inflight:
            raise RuntimeError("cannot change the served policy version while a batch is in flight")
        self.version = version

    # ------------------------------------------------------------------ API
    async def submit(self, messages, prefill: str, greedy: bool = False) -> dict:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put(_Req(messages, prefill, greedy, fut, time.time()))  # blocks when full: backpressure
        return await fut

    # ------------------------------------------------------------------ scheduler
    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.time() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                left = deadline - time.time()
                if left <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=left))
                except asyncio.TimeoutError:
                    break
            for greedy in (False, True):
                part = [r for r in batch if r.greedy == greedy]
                if part:
                    await self._run(loop, part, greedy)

    async def _run(self, loop, part: list[_Req], greedy: bool) -> None:
        t0 = time.time()
        version = self.version
        self._inflight += 1
        try:
            outs = await loop.run_in_executor(self._gpu, self.act_fn, [(r.messages, r.prefill) for r in part], greedy)
        except Exception as e:  # noqa: BLE001 - deliver the failure to every waiting worker
            for r in part:
                if not r.fut.done():
                    r.fut.set_exception(e)
            return
        finally:
            self._inflight -= 1
        t1 = time.time()
        st = self.stats
        st.batches += 1
        st.requests += len(part)
        st.batch_sizes.append(len(part))
        st.busy_s += t1 - t0
        for r, o in zip(part, outs):
            st.queue_wait_s.append(t0 - r.t_submit)
            st.latency_s.append(t1 - r.t_submit)
            st.gen_tokens += len(o["ids"])
            o["policy_version"] = version
            o["sampling"] = {"greedy": greedy, "temperature": 1.0 if greedy else self.temperature, "top_p": 1.0,
                             "top_k": 0, "constraints": "assistant prefill + stop at newline (no logit masks)"}
            if not r.fut.done():
                r.fut.set_result(o)
