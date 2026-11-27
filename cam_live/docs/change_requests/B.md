# Change requests — Workstream B (stage 1: B-CR-1..5, stage 2: B-CR-6..11)

None of these blocks B; each has a local workaround in `src/cam_live/predict/`.

## B-CR-1 `contracts.py`: name the operation of a multi-step program

**What.** `ActionSpec.steps` "makes it a program", but the contract does not say what `operation` such an action
carries, so the candidate parser, the environment's `validate_action` and the executors could disagree.
**Why.** `predict.candidates.parse_actions` must emit something every environment recognises.
**Workaround.** `predict.candidates.PROGRAM_OPERATION = "program"`; a program gets `level=PROCEDURE`, the modality
of its first step and `auth_scope` = comma-joined union of its steps' scopes; each step keeps its own tool, scope and
modality for per-step authorization.
**Proposed diff.**
```diff
 class Level:
     ...
+
+#: ``ActionSpec.operation`` of a multi-step program: its content is ``steps``, executed in order, every step
+#: authorized individually (a program never has more permission than its steps issued one by one)
+PROGRAM_OPERATION = "program"
```

## B-CR-2 `contracts.ExperimentRecord`: fields for assignment probabilities and the test used

**What / why.** Spec 11.1 and the architecture (§3 Experiments) require randomised assignment "with recorded
probabilities" and a conclusion that states the test used. `ExperimentRecord` has no field for either.
**Workaround.** `ExperimentRegistry.meta(experiment_id)` returns `arm_probs`, `assignments` (unit, arm, probability,
time), `outcome_arms`, `started_at`, `test`; all of it is in the hash-chained `experiments.jsonl`.
**Proposed diff.**
```diff
     assignment: str = "none"         # "randomized" | "sequential" | "none"
+    arm_probs: List[float] = field(default_factory=list)               # assignment probability of each arm
+    assignments: List[Dict[str, Any]] = field(default_factory=list)    # [{"unit_id", "arm", "probability", "ts"}]
+    test: Dict[str, Any] = field(default_factory=dict)                 # the test used for ``supported``
```

## B-CR-3 `contracts.PredictionRecord`: `registered_at`

**What / why.** "Only prospectively registered predictions are scored." `made_at` is written by the predictor;
the tamper-evident fact is when the registry received the record. Scoring from records alone (e.g. the evaluation
workstream reading `predictions.jsonl`) needs that time.
**Workaround.** `PredictionRegistry.registered_at(prediction_id)`; `PredictionRegistry.score()` passes it to
`calibration.score_predictions(registered_at=...)`; resolution requires `observed_at > max(made_at, registered_at)`.
**Proposed diff.**
```diff
     made_at: float
+    registered_at: Optional[float] = None    # set by the prediction registry; outcomes must be strictly later
```

## B-CR-4 `contracts.Decision.exploration`: list the rule names actually produced

**What.** The comment lists `"thompson" | "ucb" | "greedy" | "voi" | "randomized"`; `predict.policy.RULES` also has
`"policy"` (sampling the learned policy's own distribution).
**Proposed diff.**
```diff
-    exploration: str = ""            # rule that produced the choice: "thompson" | "ucb" | "greedy" | "voi" | "randomized" | ...
+    exploration: str = ""            # predict.policy.RULES: "policy" | "thompson" | "ucb" | "greedy" | "voi" | "randomized"
```

## B-CR-5 `nn/api.py`: `encode_queries` and per-query side information (note, no change required)

`WorldHeads.forward` takes `exposure [B,Q]` and `n_categories [B,Q]`, which `encode_queries` does not return.
B derives them from the records with `predict.world.build_targets` (`QueryMeta`). If A prefers a single source,
returning them from `encode_queries` would be the alternative; B works either way.

---

# Stage 2

## B-CR-6 `learning_api.matured_outcomes` / `contracts.ExperienceRecord`: a home for the questions the agent asked

**What.** `matured_outcomes` finds a query only in the chosen candidate's `predictions`, in `pending_queries`, or in
`result.expected_outcomes`. A question the *agent* adds about its own action (the same metric at a shorter horizon;
an immediate reading reported by the tool result) leaves `pending_queries` when it matures and is then no longer
findable unless a `PredictionRecord` for it exists on the decision. Experience collected without the predictor in
the loop (`diagnostics.collect`) therefore cannot carry multi-horizon or immediate targets.
**Why.** Spec 4.1 asks for several horizons and for immediate tool outcomes.
**Workaround.** `predict.labdata.add_horizons` / `add_immediate` (collector `on_step` hooks) append such queries to
`result.expected_outcomes`, which was meant for the environment's own queries.
**Proposed diff.**
```diff
 class ExperienceRecord(Record):
     pending_queries: List[OutcomeQuery] = field(default_factory=list)
+    queries: List[OutcomeQuery] = field(default_factory=list)   # every question asked about this decision
```
```diff
 def matured_outcomes(record):
     for q in record.pending_queries:
         queries.setdefault(q.query_id, q)
+    for q in record.queries:
+        queries.setdefault(q.query_id, q)
```

## B-CR-7 `learning_api`: let a provider say how its batch must be encoded

**What / why.** The slow-timescale providers need a graph into the semantic adapters, bounded in memory
(`grad_last`, cf. A-3), and the world-model provider needs `with_next=True`. `LossProvider` has no way to state
this and `DecisionBatch` does not record how it was encoded.
**Workaround.** Every B provider exposes `batch_kwargs` (keyword arguments for `encode_decisions`).
`WorldModelLoss` computes the next belief itself when `batch.b_next is None`. The adapter providers re-encode at
most `adapter_max_records` records with `providers.encode_with_adapters(..., grad_last=k)` unless the batch carries
the attribute `with_grad_semantic = True`.
**Proposed diff.**
```diff
 class DecisionBatch:
+    with_grad_semantic: bool = False       # the graph reaches the semantic adapters
 class LossProvider(Protocol):
+    batch_kwargs: Dict[str, Any]           # encode_decisions keyword arguments this provider needs
 def encode_decisions(..., with_grad_semantic: bool = False, grad: bool = True,
+                     grad_last: Optional[int] = None) -> DecisionBatch:
```

## B-CR-8 `contracts.OutcomeObservation`: an `imagined` flag

**What / why.** The contract docstring says imagined values carry `imagined=True` and are refused by outcome APIs,
but `OutcomeObservation` has no such field: a model-generated value wrapped in one is indistinguishable from a
measurement except by its free-text `source`.
**Workaround.** B never builds an `OutcomeObservation` from a model output (the planner returns frozen
`ImaginedEvaluation`s; imagined forecasts are `PredictionRecord`s with `imagined_horizon_steps >= 1`), every outcome
API of B calls `refuse_imagined`, and `Predictor.observe_outcomes` also refuses a `source` that starts with "imagin".
**Proposed diff.**
```diff
 class OutcomeObservation(Record):
     dedup_key: str = ""
+    imagined: bool = False           # never True for anything that supervises; outcome APIs must refuse it
```

## B-CR-9 `config.CamLiveConfig.debug()`: hash vocabulary of the tiny backbone

**What.** With `tiny_vocab=512` distinct argument values collide (measured: the rendered actions for channel 0 and
channel 2 have cosine 1.0000), so no head can tell those actions apart in debug-scale tests.
**Workaround.** `predict.labdata.debug_config()` sets `backbone.tiny_vocab = 4096` (no collision among the digits).
**Proposed diff.** `BackboneConfig.tiny_vocab: int = 4096`.

## B-CR-10 (to A) `nn.fast.FastBeliefState`: offsets and cross-kind information

1. `update` / `predict` take an `exposure` for counts (which B uses as the head's rate offset) but there is no
   offset for the binary logit. B therefore adds the *shift* of the binary posterior from its prior to the head's
   logit — an approximation. Proposed: `update(..., offset=None)` / `predict(..., offset=None)` on the linear
   predictor for every GLM kind.
2. Posteriors are per outcome kind and independent. A real-valued reading that measures the same latent quantity
   as a later count (LabWorld's probe) cannot inform the count posterior, so the fast-layer value of information of
   such a measurement is exactly zero and its value can only be learned through the core. Recorded as a limitation;
   no diff proposed.
3. Categorical outcomes (A-5) remain head-only in B.

## B-CR-11 `diagnostics.collect.lab_candidates`: estimated cost of candidates

`lab_candidates` builds `ActionSpec`s without `est_cost`, so cost-aware selection, the planner's step durations and
the predictor's `issued_at = now + est_cost.seconds` see zero. Proposed: copy `ToolDescriptor.est_cost` (as
`predict.candidates.schema_candidates` does). B's own tests use `schema_candidates`.
