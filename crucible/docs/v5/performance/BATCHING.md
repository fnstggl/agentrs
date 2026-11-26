# Shared batched inference (MASTER_SPEC 13.1, SP-13.1)

Module: `crucible.substrate.inference_batch` (workstream M). Tests: `tests/test_v5_perf_batching.py`.

## What it does

Every neural or semantic decision of a v5 world is a recorded inference (`substrate.inference.recorded`). The batching
service is the one place where decisions are **grouped**: requests from any number of call sites (class-2 adaptive
policies, class-3 strategic policy groups, class-4 semantic generalists, any other recorded model call) that are
compatible are evaluated in one batched call of the shared model and recorded once.

Compatible = same operation (`Op`), same model identity (model id + weights hash, i.e. the same shared weights),
same static parameters (e.g. the memory setting of an adaptive group, the semantic call kind), and rows of the same
shape and dtype. Rows are concatenated along axis 0, the operation runs once, the rows-first output is split back in
submission order. The merged call is recorded with kind `batch.<op>` and an inputs digest over the merged rows, the
static parameters and the row segments, so regeneration re-forms exactly the same groups and never runs a model.

Operations must be row-independent (true for every forward pass in the tree). A row's floating-point output can
differ in the last bits with the batch composition (CPU GEMM blocking, bfloat16 on the GPU; the skeptic measured 4
of 16 semantic replies changing with the batch's composition): this is recorded, so replay is exact, and it is a
documented cross-participant numerical coupling, not a behavioural one (`test_merged_outputs_equal_separate_evaluation`:
merged class-2 scores equal separate ones to 1e-5 and every argmax agrees).

## API

```python
from crucible.substrate import inference_batch as IB

# requests (model identity and compatibility parameters filled in)
r2 = IB.adaptive_request(policy, obs, mem, cand, model_id=f"adaptive:{gid}")     # class 2 (AdaptivePolicy)
r3 = IB.strategic_request(group, obs, hist, model_id=f"strategic:{name}")         # class 3 (StrategicGroup)
r4 = IB.semantic_request(service, "choose", requests)                             # class 4 (SemanticService)
rx = IB.Request(op, model, model_id, model_hash, inputs, cls=..., static={...})   # anything else (custom IB.Op)

out = IB.decide(ctx, r2)                  # one request, one recorded call (drop-in for inference.recorded)
outs = IB.decide_many(ctx, [r2, r3, r4])  # several callers' requests, grouped now
res = IB.gather(ctx, tasks)               # cooperative decision rounds (below)
with IB.batched(ctx) as rnd:              # submit/flush form
    h = rnd.submit(r2)
use(h.value)

IB.report(IB.stats(world))                # per class / per op: calls, requests, rows, rows per call, max batch
```

`gather(ctx, tasks)`: each task is a generator written as straight-line code that yields a request (or a list of
requests) where it needs a model output:

```python
def class4_turn(ctx, idx, now):
    reqs = [prompt_for(i) for i in idx]
    out = yield IB.semantic_request(svc, "choose", reqs)       # suspended until every task has yielded
    act_on(ctx, idx, out)
    return next_active_at
```

`gather` advances every task to its next `yield`, groups all requests of that step across tasks, runs each group
once and sends the outputs back, until every task has returned. Non-generator tasks are passed through, so plain
handlers keep working unchanged.

## How workstream B (population) uses it

1. `population/adaptive.py` `AdaptivePolicy`-based `choose()`: replace the per-group
   `inference.recorded(ctx, f"pop.adaptive.{gid}", ...)` by
   `IB.decide(ctx, IB.adaptive_request(pol, inp["obs"], inp["mem"], inp["cand"], model_id=f"adaptive:{gid}"))`
   (masking and Gumbel sampling stay with the caller).
2. `population/strategic.py` `pr_decide()`: replace `inference.recorded(ctx, f"pop.strategic.{gname}", ...)` by
   `IB.decide(ctx, IB.strategic_request(grp, O, H, model_id=f"strategic:{gname}"))`.
3. `population/core.py` `pop.tick` (`pr_tick` calls each class runtime, then `run_cohort` calls each activity
   handler, one after the other): when a handler returns a generator, wrap it as
   `IB.Task(gen, lambda owner=owner: running_as(ctx, owner))` (each step then runs as its owner module, as the
   direct call does today), collect the tasks of all class runtimes and activity handlers of the tick and drive
   them with one `IB.gather(ctx, tasks)`; plain return values are passed through unchanged. Then every class-2 group,
   class-3 group and the class-4 runtime that need the same shared model in one tick share one call, whichever
   activity asked.

## How workstream D (semantic) uses it

1. Bind class 4 (`population.register_class_runtime(world, 4, handler, owner=...)`, ONE_WORLD_CONTRACTS section 1)
   with a handler that decides the due participants' turns through `IB.semantic_request(svc, "choose" | "generate",
   reqs)`, as `IB.decide` inside the tick's transaction or as a generator task for `IB.gather`. The tick's ctx records
   the call, so the `admin.semantic_infer` staging path is not needed for in-tick decisions.
2. Several `SemanticGroup`s (different goals, one shared service) then merge into one service call per step.
3. Please expose `SemanticService.normalize(request, choices=False)` (the service's `_norm`); the batching service
   uses `normalize` when it exists and falls back to `_norm`.

## Measurement

`crucible.scale.bench5.batching(world, events)` reports per class the recorded calls and decisions: from the
service's accounting where callers use it, otherwise from the callers' own records (class 2: adaptive counters;
class 3: `pop.strategic.*` calls and `pop_strategic.exp` decisions; class 4: `semantic.inference` batches). SP-13.1
counts a class as batched when its decisions per call exceed one in the integrated civilisation world's benchmark
runs.
