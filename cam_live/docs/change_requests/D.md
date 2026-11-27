# Change requests — Workstream D (stage 1)

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
