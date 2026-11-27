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
rm = IB.method_request(model, "act_scores", {"obs": o, "mem": m, "acand": a},     # any row-independent method
                       outputs=("scores", "memory"), model_id=f"adaptive:{gid}", cls=2)
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

Call sites as they stand on c5c/w3b-pop (head 7f7387c, 2026-11-27 01:50; first written against the head of
2026-11-26 19:30); every change keeps the recorded outcome of a decision the same function of its inputs (masking,
Gumbel sampling and the action draw stay with the caller):

1. `population/adaptive.py` `choose()`: replace
   `inference.recorded(ctx, f"pop.adaptive.{gid}", f"adaptive:{gid}", pol.weights_hash(), inp, lambda: {...})` by
   `out = IB.decide(ctx, IB.method_request(pol, "scores", {"obs": o, "mem": m, "cand": c}, outputs=("scores",
   "memory"), keep=("scores",), model_id=f"adaptive:{gid}", cls=2, static={"use_memory": pol.use_memory}))`
   (records only the scores, as today; `IB.adaptive_request` records scores and memory; the `mask` stays with the
   caller).
1b. `population/adaptive.py` `review_day()` (new on c5c/w3b-pop: one GRU memory step per policy group at a member's
   first activation of a day, recorded as `pop.adaptive_day.{gid}`): replace it by
   `out = IB.decide(ctx, IB.method_request(pol, "step_memory", {"obs": o, "mem": m}, outputs=("memory",),
   model_id=f"adaptive:{gid}", cls=2, static={"use_memory": pol.use_memory}))` (a method returning one array with
   one output name is returned as `{"memory": ...}`, 2026-11-27). bench5 already counts `pop.adaptive_day.*` calls in
   the class-2 batch histograms.
2. `population/adaptive.py` `choose_activity()`: replace the `pop.adaptive_act.{gid}` recorded call by
   `IB.decide(ctx, IB.method_request(pol, "act_scores", {"obs": o, "mem": m, "acand": a}, outputs=("scores",
   "memory"), model_id=f"adaptive:{gid}", cls=2, static={"use_memory": pol.use_memory}))` (generic Op for any
   row-independent method, no dedicated Op needed).
3. `population/strategic.py` `PopStrategic.decide()` (class-3 runtime registered with
   `register_class_runtime(world, 3, ...)`): replace `inference.recorded(ctx, f"pop.strategic.{gname}", ...)` by
   `out = IB.decide(ctx, IB.strategic_request(grp, O, H, model_id=f"strategic:{gname}"))`. Groups have their own
   weights, so a call can only hold members of one group: class 3 is batched only when several members of a group
   are due in the same tick. On 55433e7 every member had its own decision time (120 calls for 120 decisions in 30
   development days). Done on c5c/w3b-pop: `decide()` now puts every member's next decision on the group's common
   grid (multiples of `decide_every_s`), so from their second decision on the due members of a group share one
   forward pass; the merged tree's suite measures the resulting batch sizes (`run['batching']['class3']['sizes']`).
4. `population/core.py` `pr_tick` / `run_cohort` (class runtimes, then the activity handlers, one after the other):
   to share calls across activities and classes, let a handler return a generator (`out = yield request`) and
   drive the tick's handlers with one `IB.gather(ctx, [IB.Task(gen, lambda o=owner: running_as(ctx, o)), ...])`
   (each step then runs as its owner module, as the direct call does today); plain return values pass through.

## How workstream D (semantic) uses it

Call sites as they stand on c5c/w3b-sem (2026-11-26 19:30; unchanged at head 7417855, 2026-11-27 01:50): `semantic/generalists/module.py` binds class 4 with
`population.register_class_runtime(world, 4, self._activate, owner=NAME)`; `mind.activate` makes the tick's due
participants decide together (`brain.generate(ctx, "situation_interpretation", [...])`, `brain.choose(ctx,
"tool_choice", [...])`), and `brain.LMBrain` routes each capability to its model's shared `SemanticService`, whose
`choose`/`generate` record one batch per call. So class 4 is already batched per (tick, capability, model); its
calls carry more than one participant only when several class-4 rows are due in the same tick.

1. Align class-4 activations: `_next_times` draws each participant's gap independently (`gap_s`), so with a few
   class-4 rows per world almost every call holds one participant. A decision grid (e.g. gaps rounded to the
   population tick's `adaptive_window_s`) makes the due rows of a tick share each step's calls.
2. To merge class 4 with other callers of the same model in one tick (e.g. several `SemanticGroup`s, or the shared
   space's LM), make `LMBrain.choose/generate` build `IB.semantic_request(svc, "choose" | "generate", reqs)` and
   either `IB.decide` it (same records as today, accounted per class) or yield it from a generator step driven by
   `IB.gather`.
3. Please expose `SemanticService.normalize(request, choices=False)` (the service's `_norm`); the batching service
   uses `normalize` when it exists and falls back to `_norm`.
4. Class 4 acts in a civilisation only when the world installs the runtime: no civ profile on c5c/w3b-civ or
   c5c/w3b-sem installs module `pop_semantic` (crucible.semantic.generalists) on 2026-11-27, so the merged civ world's
   class-4 rows stay inert and V.1 / SP-13.1 fail on class 4 however the calls are batched. J (civ.generate) and D
   must add it to the civ module list.

## Measurement

`crucible.scale.bench5.batching(world, events)` reports per class the recorded calls and decisions: from the
service's accounting where callers use it, otherwise from the callers' own records (class 2: adaptive counters;
class 3: `pop.strategic.*` calls and `pop_strategic.exp` decisions; class 4: `semantic.inference` batches). SP-13.1
counts a class as batched when its decisions per call exceed one in the integrated civilisation world's benchmark
runs.
