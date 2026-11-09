"""Model inference gateway shared by all agents.

* One server process holds the model(s); agents send requests over HTTP (or in-process). Requests
  are micro-batched (``max_batch`` / ``max_wait_ms``) and the queue is bounded (back-pressure instead
  of unbounded memory). There is never one model copy per agent or per browser.
* Checkpoints are **pinned per request**: a task keeps using the checkpoint it started with, so
  weights never change in the middle of a task; a newly promoted checkpoint is used by new tasks.
* Backends: ``MockBackend`` (deterministic, for infrastructure tests), ``LocalPolicyBackend`` (the
  repository's trainable content-generating LM policy on CPU/GPU; returns exact token ids and
  behaviour log-probs so the RL pipeline can learn from the result), ``OpenAICompatibleBackend``
  (a vLLM / SGLang server's ``/v1/completions`` with log-probs; endpoint-gated, not exercised here).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_id: str = ""
    task_id: str = ""
    checkpoint: Optional[str] = None  # None = current production
    prompt: str
    op_candidates: list[str]
    sample: bool = True
    temperature: float = 1.0
    max_tokens: int = 64
    num_samples: int = 1


class Draft(BaseModel):
    op_index: int
    op_logprob: float
    content: str
    token_ids: list[int]
    token_logprobs: list[float]
    gen_prompt: str
    terminated: bool = True


class DecideResponse(BaseModel):
    request_id: str
    checkpoint: str
    drafts: list[Draft]
    batch_size: int
    queue_ms: float
    compute_ms: float


class Backend:
    def checkpoints(self) -> list[str]: ...
    def batch(self, reqs: list[DecideRequest]) -> list[list[Draft]]: ...


class MockBackend(Backend):
    """Deterministic stand-in: picks the first candidate and returns a fixed template."""

    name = "mock"

    def __init__(self, checkpoint: str = "mock-v0"):
        self.current = checkpoint

    def checkpoints(self):
        return [self.current]

    def batch(self, reqs):
        return [[Draft(op_index=0, op_logprob=0.0, content=f"[mock content for {r.task_id or 'task'}]", token_ids=[],
                       token_logprobs=[], gen_prompt=r.prompt) for _ in range(r.num_samples)] for r in reqs]


class LocalPolicyBackend(Backend):
    """The repository's ContentGeneratingPolicy. ``models`` maps checkpoint id -> policy."""

    name = "local-policy"

    def __init__(self, models: dict[str, Any], production: str, seed: int = 0):
        import torch

        self.models = dict(models)
        self.production = production
        self.gen = torch.Generator().manual_seed(seed)

    def checkpoints(self):
        return sorted(self.models)

    def load(self, checkpoint_id: str, path: str) -> None:
        from agentrs.content.policy import ContentGeneratingPolicy
        from agentrs.training.common import load_model

        base = next(iter(self.models.values()))
        self.models[checkpoint_id] = ContentGeneratingPolicy(load_model(path), base.cfg, checkpoint_id=checkpoint_id)

    def add(self, checkpoint_id: str, cp) -> None:
        cp.checkpoint_id = checkpoint_id
        self.models[checkpoint_id] = cp

    def batch(self, reqs):
        import torch

        from agentrs.content.policy import generation_prompt

        out: list[list[Draft]] = [[] for _ in reqs]
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(reqs):
            groups.setdefault(r.checkpoint or self.production, []).append(i)
        with torch.no_grad():
            for ck, idx in groups.items():
                cp = self.models[ck]
                rs = [reqs[i] for i in idx]
                from agentrs.actions.schema import Action

                cands = [[Action.parse(c) for c in r.op_candidates] for r in rs]
                lps = cp.op_log_probs([r.prompt for r in rs], cands)
                flat_gp, owner, ops = [], [], []
                for k, (r, lp) in enumerate(zip(rs, lps)):
                    for _ in range(r.num_samples):
                        j = int(torch.multinomial(lp.exp(), 1, generator=self.gen)) if r.sample else int(torch.argmax(lp))
                        ops.append((j, float(lp[j])))
                        flat_gp.append(generation_prompt(r.prompt, cands[k][j].name or ""))
                        owner.append(k)
                samples = cp.sample_content(flat_gp, temperature=1.0, generator=self.gen,
                                            greedy=not all(r.sample for r in rs))
                for (j, olp), gp, k, s in zip(ops, flat_gp, owner, samples):
                    out[idx[k]].append(Draft(op_index=j, op_logprob=olp, content=s["content"], token_ids=list(s["ids"]),
                                             token_logprobs=list(s["logprobs"]), gen_prompt=gp,
                                             terminated=bool(s["terminated"])))
        return out


class OpenAICompatibleBackend(Backend):
    """vLLM / SGLang OpenAI-compatible completions endpoint (configured, not exercised in this sandbox)."""

    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key_ref: Optional[str], resolve):
        import httpx

        self.base_url, self.model, self.key_ref, self.resolve = base_url.rstrip("/"), model, api_key_ref, resolve
        self.http = httpx.Client(timeout=60)

    def checkpoints(self):
        return [self.model]

    def batch(self, reqs):
        hdr = {"Authorization": f"Bearer {self.resolve(self.key_ref)}"} if self.key_ref else {}
        out = []
        for r in reqs:
            body = {"model": r.checkpoint or self.model, "prompt": r.prompt + "\nACTION:", "max_tokens": r.max_tokens,
                    "temperature": r.temperature if r.sample else 0.0, "logprobs": 1, "n": r.num_samples}
            j = self.http.post(f"{self.base_url}/v1/completions", json=body, headers=hdr).json()
            drafts = []
            for ch in j["choices"]:
                text = ch["text"]
                idx = next((i for i, c in enumerate(r.op_candidates) if c.split("(")[0] in text), 0)
                lp = ch.get("logprobs") or {}
                drafts.append(Draft(op_index=idx, op_logprob=0.0, content=text.strip(), token_ids=[],
                                    token_logprobs=list(lp.get("token_logprobs") or []), gen_prompt=body["prompt"]))
            out.append(drafts)
        return out


class QueueFull(RuntimeError):
    pass


class BatchingInferenceServer:
    def __init__(self, backend: Backend, max_batch: int = 16, max_wait_ms: float = 10.0, max_queue: int = 1024):
        self.backend, self.max_batch, self.max_wait, self.max_queue = backend, max_batch, max_wait_ms / 1000, max_queue
        self.q: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._loop = None
        self.stats = {"requests": 0, "batches": 0, "max_batch_seen": 0, "compute_s": 0.0, "rejected": 0}

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or self._loop is not loop or self._task.done():
            self.q = asyncio.Queue(maxsize=self.max_queue)  # queue + batcher belong to the serving event loop
            self._loop = loop
            self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def decide(self, req: DecideRequest) -> DecideResponse:
        self.start()
        fut = asyncio.get_running_loop().create_future()
        try:
            self.q.put_nowait((req, fut, time.perf_counter()))
        except asyncio.QueueFull:
            self.stats["rejected"] += 1
            raise QueueFull("inference queue is full")
        return await fut

    async def _run(self) -> None:
        while True:
            items = [await self.q.get()]
            t_end = time.perf_counter() + self.max_wait
            while len(items) < self.max_batch:
                try:
                    items.append(await asyncio.wait_for(self.q.get(), max(0.0, t_end - time.perf_counter())))
                except asyncio.TimeoutError:
                    break
            reqs = [r for r, _, _ in items]
            t0 = time.perf_counter()
            try:
                res = await asyncio.to_thread(self.backend.batch, reqs)
            except Exception as e:  # noqa: BLE001 - fail the batch's callers, keep serving
                for _, f, _ in items:
                    if not f.done():
                        f.set_exception(e)
                continue
            dt = time.perf_counter() - t0
            self.stats["requests"] += len(items)
            self.stats["batches"] += 1
            self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(items))
            self.stats["compute_s"] += dt
            for (r, f, tq), drafts in zip(items, res):
                ck = r.checkpoint or getattr(self.backend, "production", None) or self.backend.checkpoints()[0]
                if not f.done():
                    f.set_result(DecideResponse(request_id=r.request_id, checkpoint=ck, drafts=drafts,
                                                batch_size=len(items), queue_ms=1000 * (t0 - tq), compute_ms=1000 * dt))


def create_inference_app(server: BatchingInferenceServer, admin_token: Optional[str] = None):
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title="agentrs inference gateway")

    @app.get("/health")
    async def health():
        return {"ok": True, "backend": getattr(server.backend, "name", "?"), "checkpoints": server.backend.checkpoints()}

    @app.post("/v1/decide")
    async def decide(req: DecideRequest):
        try:
            return await server.decide(req)
        except QueueFull:
            raise HTTPException(503, "inference queue full") from None

    @app.get("/v1/stats")
    async def stats():
        return server.stats

    @app.post("/v1/checkpoints/production")
    async def set_production(body: dict, authorization: str = Header("")):
        if not admin_token or authorization != f"Bearer {admin_token}":
            raise HTTPException(403, "admin only")
        ck = body["checkpoint"]
        if ck not in server.backend.checkpoints():
            if "path" not in body or not hasattr(server.backend, "load"):
                raise HTTPException(404, "unknown checkpoint")
            server.backend.load(ck, body["path"])
        server.backend.production = ck
        return {"production": ck}

    return app


class InferenceClient:
    """Client used by agent runtimes: in-process (shared server object) or HTTP."""

    def __init__(self, server: Optional[BatchingInferenceServer] = None, base_url: Optional[str] = None):
        self.server, self.base_url = server, base_url
        self._http = None

    async def decide(self, req: DecideRequest) -> DecideResponse:
        if self.server is not None:
            return await self.server.decide(req)
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=120, trust_env=False)
        r = await self._http.post("/v1/decide", json=req.model_dump())
        r.raise_for_status()
        return DecideResponse(**r.json())
