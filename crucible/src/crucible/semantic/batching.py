"""Cross-caller continuous batching for the shared small-model service (MASTER_SPEC 13.4).

`ContinuousBatcher(service)` accepts generation requests from any number of callers - threads of a world pool,
several worlds, a population activity and semantic participants at once - and decodes them in one running batch:

* `submit(request) -> Future` is thread-safe and returns immediately;
* a single worker thread owns the model; at every decode step it first *admits* waiting requests (up to
  `max_batch` live rows): their prompts are prefilled together (left padded), and their key/value caches are merged
  into the running batch by left-padding along the time axis (masked positions), so rows of different callers and
  different lengths decode side by side with their own position ids;
* a row leaves the batch when it emits end-of-turn or reaches its own token bound (bounded generation, capped by the
  service's `max_new_tokens`), and its future is resolved; new rows can join the next step.

The worker never blocks a caller and needs no coordination between callers. Results are the service's reply dicts
(text, tokens, finish). Greedy decoding gives the same tokens as the service's own batched `generate` up to the
numerics of the batch composition (identical in float32 on CPU in the tests; bfloat16 on GPU may differ in the
last bits, which is why world decisions record their outputs and never recompute them).

Within a world, recording works as before: a caller collects its futures and records the replies with
`crucible.substrate.inference.recorded` (see `record_replies`).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

STOP = object()


@dataclass
class _Row:
    req: dict
    ids: list[int]
    fut: Future
    caller: str
    toks: list = field(default_factory=list)
    finish: str = "length"
    gen: object = None
    t_submit: float = 0.0


class ContinuousBatcher:
    def __init__(self, service, max_batch: int | None = None, idle_s: float = 0.002):
        self.svc = service
        self.max_batch = int(max_batch or service.max_batch)
        self.idle_s = float(idle_s)
        self._q: list[_Row] = []
        self._lock = threading.Condition()
        self._stop = False
        self.stats = {"submitted": 0, "completed": 0, "steps": 0, "admissions": 0, "max_live": 0,
                      "rows_by_caller": {}, "mixed_caller_steps": 0, "row_steps": 0}
        self._thread = threading.Thread(target=self._loop, name="crucible-batcher", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def submit(self, request: dict, caller: str = "") -> Future:
        r = self.svc._norm(request)
        ids, _, trunc = self.svc._encode(r, r["max_new_tokens"])
        f: Future = Future()
        row = _Row(r, ids, f, str(caller), t_submit=time.time())
        with self._lock:
            if self._stop:
                raise RuntimeError("batcher is closed")
            self._q.append(row)
            self.stats["submitted"] += 1
            self.stats["rows_by_caller"][row.caller] = self.stats["rows_by_caller"].get(row.caller, 0) + 1
            self._lock.notify()
        return f

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify()
        self._thread.join(timeout=60)

    # ------------------------------------------------------------------ worker
    def _take(self, room: int, block: bool) -> list[_Row]:
        with self._lock:
            while block and not self._q and not self._stop:
                self._lock.wait(timeout=0.5)
            out, self._q = self._q[:room], self._q[room:]
            return out

    def _loop(self) -> None:
        import torch
        svc = self.svc
        w = svc.weights
        live: list[_Row] = []
        cache = mask = pos = logits = None
        while True:
            new = self._take(self.max_batch - len(live), block=not live)
            if not new and not live:
                if self._stop:
                    return
                continue
            try:
                if new:
                    self.stats["admissions"] += 1
                    lg, c2, m2, last2 = svc._prefill([r.ids for r in new], None)
                    for r in new:
                        if r.req["temperature"] > 0:
                            r.gen = torch.Generator(device=svc.device).manual_seed(int(r.req["seed"] or 0))
                    if live:
                        cache, mask = _merge(cache, mask, c2, m2, w.model.config)
                        pos = torch.cat([pos, last2])
                        logits = torch.cat([logits, lg])
                    else:
                        cache, mask, pos, logits = c2, m2, last2, lg
                    live += new
                self.stats["max_live"] = max(self.stats["max_live"], len(live))
                if len({r.caller for r in live}) > 1:
                    self.stats["mixed_caller_steps"] += 1
                with torch.inference_mode():
                    nxt = torch.argmax(logits, dim=-1)
                    for j, r in enumerate(live):
                        if r.req["temperature"] > 0:
                            pr = torch.softmax(logits[j] / r.req["temperature"], dim=-1)
                            nxt[j] = torch.multinomial(pr, 1, generator=r.gen)[0]
                    keep = []
                    for j, (r, t) in enumerate(zip(live, nxt.tolist())):
                        if t in w.eos_ids:
                            r.finish = "stop"
                            self._done(r)
                            continue
                        r.toks.append(t)
                        if len(r.toks) >= r.req["max_new_tokens"]:
                            self._done(r)
                            continue
                        keep.append(j)
                    self.stats["steps"] += 1
                    self.stats["row_steps"] += len(live)
                    if not keep:
                        live, cache = [], None
                        continue
                    if len(keep) < len(live):
                        sel = torch.tensor(keep, device=svc.device)
                        cache.batch_select_indices(sel)
                        mask, pos, nxt = mask[sel], pos[sel], nxt[sel]
                        live = [live[j] for j in keep]
                    pos = pos + 1
                    mask = torch.cat([mask, torch.ones((len(live), 1), dtype=mask.dtype, device=svc.device)], dim=1)
                    o = w.model(input_ids=nxt.unsqueeze(1), attention_mask=mask, position_ids=pos.unsqueeze(1),
                                past_key_values=cache, use_cache=True)
                    cache = o.past_key_values
                    logits = o.logits[:, -1, :].float()
            except Exception as e:  # noqa: BLE001 - a failure is delivered to every waiting caller
                for r in live + new:
                    if not r.fut.done():
                        r.fut.set_exception(e)
                live, cache = [], None

    def _done(self, r: _Row) -> None:
        tok = self.svc.weights.tok
        gen = len(r.toks) + (1 if r.finish == "stop" else 0)
        self.svc.stats["gen_tokens"] += gen
        self.stats["completed"] += 1
        r.fut.set_result({"text": tok.decode(r.toks, skip_special_tokens=True), "prompt_tokens": len(r.ids),
                          "gen_tokens": gen, "finish": r.finish, "truncated": False, "prefix_tokens": 0,
                          "latency_s": round(time.time() - r.t_submit, 4)})


def _merge(c1, m1, c2, m2, config):
    """Concatenate two batched KV caches along the batch axis, left-padding the shorter one in time (masked)."""
    import torch
    from transformers import DynamicCache
    from .service import _cache_tensors
    t1, t2 = m1.shape[1], m2.shape[1]
    T = max(t1, t2)
    data = []
    for (k1, v1), (k2, v2) in zip(_cache_tensors(c1), _cache_tensors(c2)):
        k1, v1 = _lpad(k1, T), _lpad(v1, T)
        k2, v2 = _lpad(k2, T), _lpad(v2, T)
        data.append((torch.cat([k1, k2]), torch.cat([v1, v2])))
    mask = torch.cat([_lpad_mask(m1, T), _lpad_mask(m2, T)])
    return DynamicCache(data, config=config), mask


def _lpad(x, T: int):
    import torch
    if x.shape[2] == T:
        return x
    pad = torch.zeros((x.shape[0], x.shape[1], T - x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=2)


def _lpad_mask(m, T: int):
    import torch
    if m.shape[1] == T:
        return m
    return torch.cat([torch.zeros((m.shape[0], T - m.shape[1]), dtype=m.dtype, device=m.device), m], dim=1)


def record_replies(ctx, service, requests: list[dict], replies: list[dict]) -> list[dict]:
    """Record replies obtained from the batcher in the world (replay returns them without the model)."""
    from .service import recorded
    reqs = [service._norm(r) for r in requests]
    return recorded(ctx, "semantic.generate", service.model_id, service.model_hash(), service.descriptors(reqs),
                    lambda: [{k: v for k, v in r.items() if k != "latency_s"} for r in replies])
