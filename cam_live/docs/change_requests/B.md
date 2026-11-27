# Change requests — Workstream B (stage 1)

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
