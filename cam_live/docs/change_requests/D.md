# Change requests — Workstream D (stage 1: D-1 … D-6, stage 2: D-7 … D-12)

Nothing below blocks D: each has a workaround in D's own code.

## D-1 `contracts.ExperienceStore`: outcome de-duplication, provenance and sampler state are not in the protocol

**What.** The protocol has no way to attach an outcome with de-duplication, to say which provenance stratum a sample
comes from, or to checkpoint the sampler. `SqliteExperienceStore` has all three as extra methods / keyword.
**Why.** Spec 10.1 (provenance never mixed), 7.8 (outcomes counted once), 15.2 (replay indices in checkpoints). Code
written against the protocol alone cannot use them, and `diagnostics.memstore.MemoryStore` lacks them.
**Proposed diff.**
```diff
 class ExperienceStore(Protocol):
     def pending(self, now: float) -> List[Tuple[ExperienceRecord, OutcomeQuery]]: ...
     def sample(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None,
-               seed: Optional[int] = None) -> List[ExperienceRecord]: ...
+               seed: Optional[int] = None, provenance: Optional[Sequence[str]] = None) -> List[ExperienceRecord]: ...
+    def add_outcome(self, obs: OutcomeObservation, exp_id: str = "") -> Any: ...   # .accepted / .duplicate
+    def next_due(self) -> Optional[float]: ...
+    def state_dict(self) -> Dict[str, Any]: ...
+    def load_state_dict(self, state: Dict[str, Any]) -> Dict[str, Any]: ...
```
**Workaround.** Callers that hold a `SqliteExperienceStore` use the extra methods directly.

## D-2 `contracts.ExperienceRecord.eligible`: the meaning of a missing entry is not stated

**What / why.** `MemoryStore.sample` treats a missing `eligible[loss]` as eligible; a fail-closed reading is equally
plausible and safer. D's store follows the reference (`eligible_default=True`) and offers
`eligible_default=False`.
**Proposed diff.**
```diff
-    eligible: Dict[str, bool] = field(default_factory=dict)      # loss name -> may be trained on
+    eligible: Dict[str, bool] = field(default_factory=dict)      # loss name -> may be trained on (no entry = eligible)
```

## D-3 `diagnostics.memstore.MemoryStore.put_item` overwrites a stored item

**What / why.** Overwriting an item under an existing ref changes the context `H_t` of every earlier decision that
references it (future leakage, RC-07). D's store keeps the first version and warns. The two stores therefore differ
when the same ref is put twice with different content.
**Proposed diff.**
```diff
     def put_item(self, item: ExperienceItem) -> None:
-        self._items[item.ref] = item
+        self._items.setdefault(item.ref, item)      # history is immutable; use a new ref for a new version
```

## D-4 `contracts.PrimitiveOp`: an operation cannot say which step of a compositional action it belongs to

**What / why.** `ActionResult.ops` is flat and `ActionSpec.steps` is nested; spec 12A.9 asks for the underlying
operations of each procedure. `credit.smdp.build_trace` can only assign operations to steps when the caller supplies
per-step results with time intervals; otherwise every operation stays with the root.
**Proposed diff.**
```diff
 class PrimitiveOp(Record):
     ...
     note: str = ""
+    step_id: str = ""                # action_id of the ActionSpec.steps entry that issued this operation ("" = top)
```

## D-5 `contracts.Record.from_dict` resolves type hints on every call

**What / why.** `typing.get_type_hints(cls)` is recomputed for every nested record: about 6 ms per
`ExperienceRecord` with eight candidates (measured: sampling 32 records took 184 ms, 10.7 ms with a cached plan).
D decodes with its own cached `credit.replay.record_from_dict` (tested equal to `from_dict`).
**Proposed diff.**
```diff
+@functools.lru_cache(maxsize=None)
+def _hints(cls):
+    return typing.get_type_hints(cls), frozenset(f.name for f in dataclasses.fields(cls))
+
     @classmethod
     def from_dict(cls, d: Dict[str, Any]):
-        hints = typing.get_type_hints(cls)
-        known = {f.name for f in dataclasses.fields(cls)}
+        hints, known = _hints(cls)
```

## D-6 `runtime/cli.py` expects `cam_live.credit.checkpoint.export_weights(checkpoint_path, dest)`

Provided with exactly that signature (plus optional `config=`). `AgentAPI.save(path)` can use
`credit.checkpoint.save_checkpoint_at(path, state)`; `export_weights` reads the directory it wrote. No change
needed; recorded so the integration point is explicit.


## D-7 `diagnostics.collect.LabCollector`: sequence numbers, no linker, no experiment arms

**What.** (a) `LabCollector(seq_start=0)` writes `seq=0, 1, 2, …`; a store that assigns `seq` when it is 0 (D's
does: lifetime head + 1) gives the first record `seq=1`, and the collector's second record (`seq=1`) is then refused
(`ValueError: seq 1 is not after the lifetime head 1`). (b) Outcomes are written straight into the record with
`store.update`, so no outcome link, no de-duplication through `add_outcome` and no re-training queue entry exists
for collected experience. (c) A decision cannot carry a randomised arm (`exploration="randomized"`,
`experiment_id`, `diagnostics["arm"]`), which the credit comparison needs.
**Why.** RC-01 / RC-06 / RC-08 evidence has to go through the linker and through registered experiments.
**Workaround.** `credit.labkit.LabRun` / `collect_lifetime` (reuses the master's item builders and `lab_candidates`).
`collect_family(store, …, seq_start=store.head_seq() + 1)` also works for (a).
**Proposed diff** (`diagnostics/collect.py`), minimal fix for (a):
```diff
-                                   seq=self.seq)
-            self.seq += 1
+                                   seq=self.seq if self.seq > 0 else 0)      # 0 = the store assigns the position
+            self.seq += 1 if self.seq > 0 else 0
```

## D-8 `learning_api.encode_decisions`: no way to bound the adapter graph (same as A-3)

**What / why.** `with_grad_semantic=True` builds a backbone graph for every context item of every record. The slow
adapter update needs `tokenize(..., grad_last=k)` (provided by `CamLiveModel`).
**Workaround.** `credit.learner.encode_with_adapter_grad(model, store, records, grad_last=k)` (a copy of
`encode_decisions` that passes `grad_last` when the model accepts it).
**Proposed diff.**
```diff
 def encode_decisions(model, store, records, *, with_next=False, with_grad_semantic=False,
-                     grad=True) -> DecisionBatch:
+                     grad=True, grad_last: Optional[int] = None) -> DecisionBatch:
 ...
-        tokens = model.tokenize(contexts, with_grad=with_grad_semantic)
+        kw = {"grad_last": grad_last} if (with_grad_semantic and grad_last is not None) else {}
+        tokens = model.tokenize(contexts, with_grad=with_grad_semantic, **kw)
```

## D-9 `learning_api.LossProvider`: optional attributes the learner honours

**What.** `ContinualLearner` reads four optional, duck-typed attributes that the protocol does not mention:
`uses_batch = False` (the provider reads its own data and is always called: retention anchors),
`needs_latents = False` (only `batch.records` is used; the learner then skips encoding: trajectory providers),
`adapter_grad = False` (a frequent provider that should not take part in slow adapter updates), and
`state_dict()` / `load_state_dict()` (tensor state saved in checkpoints).
**Why.** Without them every provider costs one batch encoding per update and retention cannot be a provider.
**Proposed diff** (`learning_api.py`, docstring of `LossProvider`):
```diff
+    Optional attributes (defaults in brackets) read by ``credit.learner.ContinualLearner``:
+    ``uses_batch`` [True], ``needs_latents`` [True], ``adapter_grad`` [True], ``state_dict`` / ``load_state_dict``.
```

## D-10 `contracts.ExperienceStore`: a closed objective and its verified outcome have no place

**What / why.** A trajectory's return is final only when the objective is closed. The protocol stores decisions,
not objective outcomes, so "matured verified return" could not be decided from the store. D's store adds
`put_objective_outcome(objective_id, *, status, value, evaluated_at, verified, body)` / `objective_outcome(id)`.
**Integration.** `AgentAPI.reconcile(outcomes, verified)` must call `store.put_objective_outcome(...)` with
`verified=verifier.check_signature(v)` when the objective ends, and queue its records for re-training.
**Proposed diff** (`contracts.py`):
```diff
 class ExperienceStore(Protocol):
+    def put_objective_outcome(self, objective_id: str, *, status: str, value: float = 0.0, evaluated_at: float = 0.0,
+                              verified: bool = False, body: Optional[Dict[str, Any]] = None) -> None: ...
+    def objective_outcome(self, objective_id: str) -> Optional[Dict[str, Any]]: ...
```

## D-11 `contracts.LearningUpdateRecord`: no structured place for loss shares and the behaviour flag

**What / why.** The learner must log each loss's share of the experiential gradient, the balance scale, the replay
strata and whether behaviour changed. They are packed as JSON into `notes` today.
**Proposed diff.**
```diff
     model_version: str = ""
     notes: str = ""
+    diagnostics: Dict[str, Any] = field(default_factory=dict)   # loss shares, balance scale, strata, flags
```

## D-12 `diagnostics.world.LabWorld`: the query metric contains the action id

**What / why.** `OutcomeQuery.metric = f"lab.adopters.from.{action_id}"` makes every query a new metric name: the
tokenizer's per-metric-name embedding and the query text embedding then see a different random string per decision.
`diagnostics.collect.outcome_item` already strips it for items; query encoders have to do the same
(`credit.labkit._canonical`). Putting the action id into `OutcomeQuery.decision_id` / `description` instead would
remove the trap.
**Proposed diff** (`diagnostics/world.py`, with the matching change in `measure`):
```diff
-            q = OutcomeQuery(query_id=new_id("q"), metric=f"lab.adopters.from.{action.action_id}",
+            q = OutcomeQuery(query_id=new_id("q"), metric="lab.adopters.from", decision_id=action.action_id,
```

## Note for Workstream B

`credit.labkit.WorldPredictionProvider` wraps `predict.world.WorldHeads` / `world_nll` (stage 1 API) for D's own
evidence. It is not registered by anything outside `credit.labkit`; the agent should register B's providers.
