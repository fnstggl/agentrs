"""Shared batched inference across participant classes (MASTER_SPEC 13.1 "Group compatible decisions for batched
inference", 13.3, 13.4; SP-13.1).

Every neural or semantic decision of a v5 world is a recorded inference (`crucible.substrate.inference.recorded`).
This module is the one place where such decisions are *grouped*: requests from any number of call sites - class-2
adaptive policies, class-3 strategic policy groups, class-4 semantic generalists, any other recorded model call -
that are compatible are evaluated in ONE batched call of the shared model and recorded ONCE.

**Compatible** means: the same operation (`Op`, the computation), the same model identity (model id and weights
hash, i.e. the same shared weights), the same static parameters (e.g. the scoring mode, the memory setting of a
policy group) and rows of the same shape and dtype (array ops; list ops take any rows). Rows of compatible requests
are concatenated along axis 0, the operation runs once on the merged batch, its rows-first output is split back to
the requests in submission order. Operations must be row-independent (a row's output depends only on that row and
the model): every neural forward pass of the population and the semantic service is. Floating-point results of a
row can still differ in the last bits with the batch composition (GEMM blocking on CPU, bfloat16 on GPU); this is
why the merged call is recorded and replay never recomputes it.

Three ways to use it, all inside a kernel transaction (`ctx`):

* `decide(ctx, req)` - one request, one recorded call (the drop-in replacement for `inference.recorded` that is
  accounted per class);
* `decide_many(ctx, reqs)` - a list of requests from several callers, grouped now;
* `gather(ctx, tasks)` - cooperative decision rounds: each task is a generator (an activity handler or a class
  runtime written as `out = yield request`); `gather` advances every task to its next `yield`, groups all requests
  of that step across tasks, runs each group once and sends the outputs back, until every task has returned. A
  population tick that drives the due cohorts of classes 2-4 with `gather` gets one forward pass per shared model
  per step, whichever activity or class asked. Non-generator tasks are passed through unchanged.

Determinism: tasks are driven in the order given, requests are grouped in the order they were made and the merged
call's recorded inputs digest covers the merged rows, the static parameters and the row segments, so regeneration
re-forms exactly the same groups (and a different grouping is a recorded divergence, not a silent change).

Accounting (`stats(world)`, process-wide `STATS`): per participant class (the `cls` of each request) the merged
calls, requests, rows (decisions) and the largest batch; per operation the same; calls served from the record
(regeneration) are counted apart from calls that ran the model. SP-13.1 counts a class as batched when its rows per
call exceed one in the integrated world.

Standard operations for the models in the tree (duck-typed; nothing is imported until called):

| op | model object | rows |
|---|---|---|
| `ADAPTIVE_SCORES` | `population.adaptive.AdaptivePolicy` (`scores(obs, mem, cand)`) | arrays obs/mem/cand |
| `STRATEGIC_LOGITS` | `population.strategic.StrategicGroup` (`logits(obs, hist)`) | arrays obs/hist |
| `SEMANTIC_CHOOSE`, `SEMANTIC_GENERATE` | `semantic.service.SemanticService` (`run(kind, normalised requests)`) | request dicts |
| `TORCH_FORWARD` | any `torch.nn.Module` with one array input | array x |
| `method_op(method, args, outputs)` | any model with a row-independent method (`method_request`) | named arrays |

Helpers `adaptive_request`, `strategic_request` and `semantic_request` build the requests (model identity and the
static parameters that make two requests compatible).
"""
from __future__ import annotations

import inspect
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ..core.canonical import canonical_bytes, to_jsonable
from ..core.errors import ProtocolError
from . import inference

KIND_PREFIX = "batch."


# ====================================================================== operations and requests
@dataclass(frozen=True)
class Op:
    """A batched computation of a shared model.

    run(model, inputs, static) -> outputs, rows first (array, dict of arrays, or a list with one JSON item per row).
    rows: "array" (inputs: an array or a dict of arrays, rows on axis 0) or "list" (inputs: a list of JSON rows).
    describe(model, inputs, static) -> what is digested in the record instead of the raw inputs (e.g. request
    digests instead of prompt text); default: the inputs themselves.
    max_rows: largest merged batch (compatible requests beyond it run in further calls; a single larger request is
    never split)."""
    name: str
    run: Callable[[Any, Any, dict], Any]
    rows: str = "array"
    describe: Callable[[Any, Any, dict], Any] | None = None
    max_rows: int | None = None

    def __post_init__(self):
        if self.rows not in ("array", "list"):
            raise ProtocolError(f"Op {self.name}: rows must be 'array' or 'list'")
        if not isinstance(self.name, str) or not self.name:
            raise ProtocolError("Op needs a name")


@dataclass
class Request:
    """One caller's decision request: `inputs` rows evaluated by `op` with the shared `model` whose identity is
    (`model_id`, `model_hash`). `cls` is the participant class the rows belong to (1-4; None for other callers),
    `static` the JSON parameters that must be equal for two requests to share a call."""
    op: Op
    model: Any
    model_id: str
    model_hash: str
    inputs: Any
    cls: int | None = None
    static: dict = field(default_factory=dict)
    label: str = ""

    def n_rows(self) -> int:
        x = self.inputs
        if self.op.rows == "list":
            if not isinstance(x, (list, tuple)):
                raise ProtocolError(f"{self.op.name}: list rows need a list of inputs")
            return len(x)
        if isinstance(x, np.ndarray):
            return int(x.shape[0]) if x.ndim else 1
        if isinstance(x, dict) and x and all(isinstance(v, np.ndarray) and v.ndim >= 1 for v in x.values()):
            ns = {int(v.shape[0]) for v in x.values()}
            if len(ns) != 1:
                raise ProtocolError(f"{self.op.name}: input arrays disagree on the number of rows {sorted(ns)}")
            return ns.pop()
        raise ProtocolError(f"{self.op.name}: array rows need an array or a dict of arrays")


def _signature(req: Request) -> tuple:
    if req.op.rows == "list":
        return ("list",)
    x = req.inputs
    if isinstance(x, np.ndarray):
        return (("", x.dtype.str, tuple(x.shape[1:])),)
    return tuple((k, x[k].dtype.str, tuple(x[k].shape[1:])) for k in sorted(x))


def compat_key(req: Request) -> tuple:
    """Requests with equal keys share one call."""
    if not isinstance(req.model_id, str) or not isinstance(req.model_hash, str) or not req.model_hash:
        raise ProtocolError("a batched request needs model_id and model_hash strings")
    return (req.op.name, req.model_id, req.model_hash, canonical_bytes(to_jsonable(req.static or {})),
            _signature(req))


def _merge(op: Op, parts: list) -> Any:
    if op.rows == "list":
        out = []
        for p in parts:
            out.extend(p)
        return out
    if isinstance(parts[0], np.ndarray):
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    return {k: (parts[0][k] if len(parts) == 1 else np.concatenate([p[k] for p in parts], axis=0))
            for k in sorted(parts[0])}


def _split(op: Op, out: Any, seg: list[int]) -> list:
    total = sum(seg)
    bounds = np.cumsum([0] + seg).tolist()
    if isinstance(out, np.ndarray):
        if out.shape[0] != total:
            raise ProtocolError(f"{op.name}: output has {out.shape[0]} rows for {total} input rows")
        return [out[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    if isinstance(out, dict) and out and all(isinstance(v, np.ndarray) for v in out.values()):
        for k, v in out.items():
            if v.shape[0] != total:
                raise ProtocolError(f"{op.name}: output {k} has {v.shape[0]} rows for {total} input rows")
        return [{k: v[a:b] for k, v in out.items()} for a, b in zip(bounds[:-1], bounds[1:])]
    if isinstance(out, (list, tuple)):
        if len(out) != total:
            raise ProtocolError(f"{op.name}: output has {len(out)} rows for {total} input rows")
        return [list(out[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    raise ProtocolError(f"{op.name}: output must be rows first (array, dict of arrays or list)")


# ====================================================================== accounting
_LOCK = threading.Lock()


def _new_stats() -> dict:
    return {"calls": 0, "computed_calls": 0, "served_calls": 0, "requests": 0, "rows": 0, "merged_calls": 0,
            "mixed_class_calls": 0, "max_rows": 0, "by_class": {}, "by_op": {}, "batch_sizes": Counter()}


STATS = _new_stats()          # process-wide (every world of this process)


def _bump(box: dict, key: str, requests: int, rows: int) -> None:
    s = box.setdefault(key, {"calls": 0, "requests": 0, "rows": 0, "merged_calls": 0, "max_rows": 0})
    s["calls"] += 1
    s["requests"] += requests
    s["rows"] += rows
    s["merged_calls"] += int(requests > 1)
    s["max_rows"] = max(s["max_rows"], rows)


def _account(st: dict, reqs: list[Request], computed: bool) -> None:
    rows = sum(r.n_rows() for r in reqs)
    st["calls"] += 1
    st["computed_calls" if computed else "served_calls"] += 1
    st["requests"] += len(reqs)
    st["rows"] += rows
    st["merged_calls"] += int(len(reqs) > 1)
    st["max_rows"] = max(st["max_rows"], rows)
    st["batch_sizes"][1 << max(0, int(rows - 1).bit_length())] += 1       # power-of-two buckets
    by_cls: dict[str, list[Request]] = {}
    for r in reqs:
        by_cls.setdefault("-" if r.cls is None else str(int(r.cls)), []).append(r)
    st["mixed_class_calls"] += int(len(by_cls) > 1)
    for c, rs in by_cls.items():
        _bump(st["by_class"], c, len(rs), sum(r.n_rows() for r in rs))
    _bump(st["by_op"], reqs[0].op.name, len(reqs), rows)


def stats(world) -> dict:
    """Accounting of the batched calls this world made (in this process; observers only, never world state)."""
    return world.runtime.setdefault("inference_batch.stats", _new_stats())


def report(st: dict | None = None) -> dict:
    """JSON-native summary: per class and per op calls, requests, rows, rows per call and the largest batch."""
    st = STATS if st is None else st

    def one(s):
        return {**{k: int(v) for k, v in s.items()}, "rows_per_call": round(s["rows"] / max(1, s["calls"]), 3),
                "requests_per_call": round(s["requests"] / max(1, s["calls"]), 3)}
    return {"calls": int(st["calls"]), "computed_calls": int(st["computed_calls"]),
            "served_calls": int(st["served_calls"]), "requests": int(st["requests"]), "rows": int(st["rows"]),
            "merged_calls": int(st["merged_calls"]), "mixed_class_calls": int(st["mixed_class_calls"]),
            "max_rows": int(st["max_rows"]), "rows_per_call": round(st["rows"] / max(1, st["calls"]), 3),
            "by_class": {c: one(s) for c, s in sorted(st["by_class"].items())},
            "by_op": {o: one(s) for o, s in sorted(st["by_op"].items())},
            "batch_sizes": {str(k): int(v) for k, v in sorted(st["batch_sizes"].items())}}


def reset_process_stats() -> None:
    with _LOCK:
        fresh = _new_stats()
        STATS.clear()
        STATS.update(fresh)


# ====================================================================== running groups
def _run_group(ctx, reqs: list[Request]) -> list:
    r0 = reqs[0]
    op = r0.op
    seg = [r.n_rows() for r in reqs]
    merged = _merge(op, [r.inputs for r in reqs])
    static = to_jsonable(r0.static or {})
    desc = op.describe(r0.model, merged, static) if op.describe is not None else merged
    ran = [False]

    def fn():
        ran[0] = True
        return op.run(r0.model, merged, static)
    out = inference.recorded(ctx, KIND_PREFIX + op.name, r0.model_id, r0.model_hash,
                             {"x": desc, "static": static, "seg": seg}, fn)
    parts = _split(op, out, seg)
    _account(stats(ctx.world), reqs, ran[0])
    with _LOCK:
        _account(STATS, reqs, ran[0])
    return parts


def _chunks(js: list[int], reqs: list[Request], max_rows: int | None) -> list[list[int]]:
    if not max_rows:
        return [js]
    out, cur, n = [], [], 0
    for j in js:
        k = reqs[j].n_rows()
        if cur and n + k > max_rows:
            out.append(cur)
            cur, n = [], 0
        cur.append(j)
        n += k
    if cur:
        out.append(cur)
    return out


def decide_many(ctx, reqs: list[Request]) -> list:
    """Group `reqs` by compatibility (first appearance order), run each group once, return one output per request
    (in the order given). Requests with no rows get an empty output without a call."""
    reqs = list(reqs)
    outs: list = [None] * len(reqs)
    groups: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for j, r in enumerate(reqs):
        if not isinstance(r, Request):
            raise ProtocolError(f"decide_many takes Request objects, got {type(r).__name__}")
        if r.n_rows() == 0:
            outs[j] = [] if r.op.rows == "list" else None
            continue
        groups.setdefault(compat_key(r), []).append(j)
    for js in groups.values():
        for chunk in _chunks(js, reqs, reqs[js[0]].op.max_rows):
            for j, part in zip(chunk, _run_group(ctx, [reqs[j] for j in chunk])):
                outs[j] = part
    return outs


def decide(ctx, req: Request) -> Any:
    """One request, one recorded call (accounted per class)."""
    return decide_many(ctx, [req])[0]


@dataclass
class Task:
    """A generator task with a context entered around each of its steps (e.g. the population's
    `running_as(ctx, owner)`, so an activity handler writes its own module's columns in every step)."""
    gen: Any
    context: Callable[[], Any] | None = None


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def gather(ctx, tasks: Iterable) -> list:
    """Drive generator tasks together; see the module docstring. A task yields a Request (and receives its output)
    or a list of Requests (and receives the list of outputs); its return value is its result. A `Task` wraps a
    generator with a context entered around each of its steps. Non-generator tasks are results already. Returns the
    results in task order."""
    tasks = list(tasks)
    results: list = [None] * len(tasks)
    live: dict[int, Any] = {}
    ctxs: dict[int, Callable[[], Any]] = {}
    send: dict[int, Any] = {}
    for i, t in enumerate(tasks):
        g, cm = (t.gen, t.context) if isinstance(t, Task) else (t, None)
        if inspect.isgenerator(g):
            live[i] = g
            ctxs[i] = cm or _Null
            send[i] = None
        else:
            results[i] = g
    try:
        while live:
            asked = []
            for i in sorted(live):
                try:
                    with ctxs[i]():
                        y = live[i].send(send[i])
                except StopIteration as e:
                    results[i] = e.value
                    del live[i]
                    continue
                if isinstance(y, Request):
                    asked.append((i, False, [y]))
                elif isinstance(y, (list, tuple)) and all(isinstance(r, Request) for r in y):
                    asked.append((i, True, list(y)))
                else:
                    raise ProtocolError(f"a gathered task must yield a Request or a list of Requests, not "
                                        f"{type(y).__name__}")
            if not asked:
                break
            outs = decide_many(ctx, [r for _, _, rs in asked for r in rs])
            k = 0
            for i, many, rs in asked:
                o = outs[k:k + len(rs)]
                k += len(rs)
                send[i] = o if many else o[0]
    finally:
        for g in live.values():
            g.close()
    return results


class Round:
    """Collect requests and run them together: `with batched(ctx) as r: h = r.submit(req)` ... after the block (or
    `r.flush()`), `h.value` holds the output."""

    class Handle:
        __slots__ = ("value", "done")

        def __init__(self):
            self.value = None
            self.done = False

    def __init__(self, ctx):
        self.ctx = ctx
        self._reqs: list[Request] = []
        self._handles: list[Round.Handle] = []

    def submit(self, req: Request) -> "Round.Handle":
        h = Round.Handle()
        self._reqs.append(req)
        self._handles.append(h)
        return h

    def flush(self) -> None:
        reqs, hs = self._reqs, self._handles
        self._reqs, self._handles = [], []
        for h, o in zip(hs, decide_many(self.ctx, reqs)):
            h.value, h.done = o, True

    def __enter__(self) -> "Round":
        return self

    def __exit__(self, et, ev, tb) -> None:
        if et is None:
            self.flush()


def batched(ctx) -> Round:
    return Round(ctx)


# ====================================================================== standard operations
def _adaptive_run(pol, x, static):
    if bool(static.get("use_memory", True)) != bool(getattr(pol, "use_memory", True)):
        raise ProtocolError("adaptive request's memory setting differs from its policy's")
    s, m = pol.scores(x["obs"], x["mem"], x["cand"])
    return {"scores": np.asarray(s, np.float32), "memory": np.asarray(m, np.float32)}


def _strategic_run(grp, x, static):
    out = grp.logits(x["obs"], x["hist"])
    return {k: np.asarray(v) for k, v in out.items()}


def _semantic_run(kind):
    def run(svc, reqs, static):
        return svc.run(kind, list(reqs))
    return run


def _semantic_describe(svc, reqs, static):
    return svc.descriptors(list(reqs))


def _torch_run(module, x, static):
    import torch
    with torch.no_grad():
        y = module(torch.from_numpy(np.ascontiguousarray(x)))
    return y.detach().cpu().numpy()


ADAPTIVE_SCORES = Op("adaptive.scores", _adaptive_run)
STRATEGIC_LOGITS = Op("strategic.logits", _strategic_run)
SEMANTIC_CHOOSE = Op("semantic.choose", _semantic_run("choose"), rows="list", describe=_semantic_describe)
SEMANTIC_GENERATE = Op("semantic.generate", _semantic_run("generate"), rows="list", describe=_semantic_describe)
TORCH_FORWARD = Op("torch.forward", _torch_run)


def adaptive_request(pol, obs: np.ndarray, mem: np.ndarray, cand: np.ndarray, *, cls: int = 2,
                     model_id: str | None = None, label: str = "") -> Request:
    """A class-2 decision request for an `AdaptivePolicy` (shared weights of one policy group)."""
    return Request(ADAPTIVE_SCORES, pol, model_id or f"adaptive:{getattr(pol, 'group', '?')}", pol.weights_hash(),
                   {"obs": np.ascontiguousarray(obs, np.float32), "mem": np.ascontiguousarray(mem, np.float32),
                    "cand": np.ascontiguousarray(cand, np.float32)},
                   cls=cls, static={"use_memory": bool(getattr(pol, "use_memory", True))}, label=label)


def strategic_request(grp, obs: np.ndarray, hist: np.ndarray, *, cls: int = 3, model_id: str | None = None,
                      label: str = "") -> Request:
    """A class-3 decision request for a `StrategicGroup` (shared weights of one strategic policy group)."""
    name = getattr(grp, "name", None) or (getattr(grp, "rec", {}) or {}).get("name", "?")
    return Request(STRATEGIC_LOGITS, grp, model_id or f"strategic:{name}", grp.weights_hash(),
                   {"obs": np.ascontiguousarray(obs, np.float32), "hist": np.ascontiguousarray(hist, np.float32)},
                   cls=cls, label=label)


def semantic_request(svc, kind: str, requests: list[dict], *, cls: int = 4, label: str = "") -> Request:
    """A class-4 (or any semantic) request for the shared `SemanticService`: `kind` "choose" or "generate"; the
    requests are normalised by the service (bounded generation, option lists) before they are grouped."""
    if kind not in ("choose", "generate"):
        raise ProtocolError("semantic kind must be 'choose' or 'generate'")
    norm = getattr(svc, "normalize", None) or svc._norm
    reqs = [norm(r, choices=True) if kind == "choose" else norm(r) for r in requests]
    return Request(SEMANTIC_CHOOSE if kind == "choose" else SEMANTIC_GENERATE, svc, svc.model_id, svc.model_hash(),
                   reqs, cls=cls, label=label)


_METHOD_OPS: dict[str, Op] = {}


def method_op(method: str, args: tuple, outputs: tuple | None = None) -> Op:
    """The Op that calls `model.<method>(*[inputs[a] for a in args])` on the merged rows: for any row-independent
    model method without a dedicated Op (e.g. a policy's `act_scores(obs, mem, acand) -> (scores, memory)`). The
    output is an array, a dict of arrays, or a tuple of arrays named by `outputs`. One Op per (method, args,
    outputs), so requests of different methods never share a call."""
    args = tuple(args)
    outputs = tuple(outputs) if outputs else None
    if not args or not all(isinstance(a, str) and a for a in args) or len(set(args)) != len(args):
        raise ProtocolError("method_op needs distinct input names")
    name = f"method.{method}({','.join(args)})" + (f"->{','.join(outputs)}" if outputs else "")
    with _LOCK:
        op = _METHOD_OPS.get(name)
        if op is None:
            def run(model, x, static, method=method, args=args, outputs=outputs):
                out = getattr(model, method)(*[x[a] for a in args])
                if outputs:
                    if not isinstance(out, (tuple, list)) or len(out) != len(outputs):
                        raise ProtocolError(f"{name}: expected {len(outputs)} outputs")
                    return {k: np.asarray(v) for k, v in zip(outputs, out)}
                if isinstance(out, dict):
                    return {k: np.asarray(v) for k, v in out.items()}
                if isinstance(out, (tuple, list)):
                    raise ProtocolError(f"{name}: a tuple output needs `outputs` names")
                return np.asarray(out)
            op = _METHOD_OPS[name] = Op(name, run)
    return op


def method_request(model, method: str, inputs: dict, *, model_id: str, args: tuple | None = None,
                   outputs: tuple | None = None, model_hash: str | None = None, cls: int | None = None,
                   static: dict | None = None, label: str = "") -> Request:
    """A request for `model.<method>` (see `method_op`): `inputs` maps argument names to row-first arrays, passed
    positionally in `args` order (default: the order of `inputs`). `model_hash` defaults to the model's
    `weights_hash()`; `static` holds any setting outside the weights that changes the output (two requests share a
    call only if it is equal). Class-2 example (population/adaptive.py `choose_activity`):
    `method_request(pol, "act_scores", {"obs": o, "mem": m, "acand": a}, outputs=("scores", "memory"),
    model_id=f"adaptive:{gid}", cls=2, static={"use_memory": pol.use_memory})`."""
    args = tuple(args or inputs)
    if set(args) != set(inputs):
        raise ProtocolError("method_request: args must name every input exactly once")
    h = model_hash if model_hash is not None else model.weights_hash()
    return Request(method_op(method, args, outputs), model, model_id, h,
                   {a: np.ascontiguousarray(inputs[a]) for a in args}, cls=cls, static=dict(static or {}),
                   label=label)


def torch_request(module, x: np.ndarray, *, model_id: str, cls: int | None = None, label: str = "") -> Request:
    return Request(TORCH_FORWARD, module, model_id, inference.weights_hash(module), np.ascontiguousarray(x),
                   cls=cls, label=label)


__all__ = ["Op", "Request", "Task", "decide", "decide_many", "gather", "batched", "Round", "stats", "report", "STATS",
           "compat_key", "adaptive_request", "strategic_request", "semantic_request", "torch_request",
           "ADAPTIVE_SCORES", "STRATEGIC_LOGITS", "SEMANTIC_CHOOSE", "SEMANTIC_GENERATE", "TORCH_FORWARD",
           "method_op", "method_request",
           "reset_process_stats", "KIND_PREFIX"]
