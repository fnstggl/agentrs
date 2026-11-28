# Change requests of DX (fixes of G2-learn's findings in nn/fast.py and credit/)

## DX-1 (src/cam_live/agent.py, owner INT3A): the agent hands its experiment registry to the credit assigner (RC-08, G2L-6)

**What / why.** G2L-6: nothing in the integrated agent ever compares the credit estimators with its randomised
experiments. The agent records experiments only in `predict.registry.ExperimentRegistry`; the experience store has
none, so `CreditAssigner.compare_with_experiment` cannot see them, and nothing calls the comparison or
`EstimatorSelector.select`. The credit side now provides what is needed (`credit.providers.CreditAssigner.
attach_experiments` / `record_experiments` / `evaluate_experiments`, called by `ContinualLearner.learn`): once a
source is attached, every learning update first copies the registry's experiments into the experience store (so
randomised lifetimes are known to the held-out split and travel with checkpoints, CL-10), and afterwards compares
the estimators with every randomised experiment that has at least two matured held-out units per arm and re-runs the
selection when that evidence changes. When the selection cannot run, the reason is written to
`<run>/learner/credit_selection.jsonl` (per experiment: decisions, units per arm, how many had no final return,
were in a lifetime that was trained on, or were not randomised). The agent needs one line.

**Exact diff** (against `3e29595`, line 717):
```diff
         self.assigner = CreditAssigner(self.model, self.store, rudder=self.rudder, cocoa=self.cocoa,
                                        selector=self.selector, beta=s.credit_beta, clock=self.clock)
+        self.assigner.attach_experiments(self.experiments)   # RC-08 (DX-1): experiments reach the store; selection runs
```
Nothing else is needed; no setting changes.

**What to expect after applying it (measured with the line applied at construction,
`tests/credit/test_dx_selection.py::test_integrated_agent_with_the_change_request_line_records_and_evaluates_its_experiments`,
`-m slow`, and G2-learn's `probe_agent_credit.py` run through a wrapper that applies the line).** The registry's experiments are in the store
(equal counts), each learning update evaluates them, the selection log has rows. The selection itself still
**does not run in the agent as configured today**, and the log says why:
1. each agent experiment has one randomised decision (a running experiment is reused only for the identical arm
   pair, which LabWorld rarely repeats), while an arm difference needs >= 2 matured units per arm;
2. the agent sets `holdout_experiments=False` with `val_split="record"`, so every lifetime with an experiment is
   one the learner trained on, and the comparison is held out by lifetime (G-cl M1): such units are excluded.
These are agent-side choices (INT3A), not changed by this request. Options, for the owner to weigh: reuse a running
experiment across decisions for the same *kind* of contrast (so units accumulate), and in staging / evaluation runs
with several lifetimes set `credit.learner.holdout_experiments: true` (experiment lifetimes become validation-only:
the estimators get held-out evidence, the learner loses those lifetimes as training data). The credit side refuses
in-sample comparisons either way.

**Meanwhile.** Nothing breaks without the line (no source attached: behaviour unchanged, test
`test_without_an_attached_source_nothing_changes`). D's end-to-end test with a test agent that keeps its
experiments in a real `ExperimentRegistry` (`test_the_agents_experiments_reach_the_store_and_the_selection_runs_on_held_out_lifetimes`)
shows the selection running on held-out lifetimes.

## DX-2 (note to INT3A / INT3B, no diff required): behaviour changes the integrated agent will see from DX

* **Eligibility flags** (`credit/learner.py`): providers are now asked `eligible(record)` without the flags the
  learner itself wrote (`eligible[name]`, `eligible["val:" + name]`). `predict.providers._Base._permitted` reads
  `record.eligible[self.name]` as an exclusion, so before the fix a held-out record lost its `val:` flag at the next
  refresh and a record judged ineligible once (e.g. evaluator records before the objective closed) never became
  trainable. Measured in G2-learn's agent probe: `learner.validate()` at the end had 3 providers (rudder, cocoa,
  bridge_grounding), now 8 (world_model, fast_prior, evaluator, world_model_adapters, evaluator_adapters, rudder, cocoa,
  bridge_grounding). The evaluator providers therefore now train on records of closed objectives, and promotion
  has more independent validation metrics.
* **Slow-step guard** fails closed and judges each provider: slow steps are deferred while no provider has held-out
  records, a loss without held-out records does not move the adapters, and with one validation chunk per provider
  (short lifetimes) the tolerance is the float floor, so more slow steps are reverted (agent probe: 8 of 11 reverted,
  0 without a verdict; before: 0 of 15 reverted, 2 without a verdict).
* **Store**: `SqliteExperienceStore.append` / `update` raise `credit.replay.OutcomeTimingError` (a `ValueError`) for a
  newly attached matured outcome dated before its query was issued or before its decision; `add_outcome` returns
  `accepted=False, reason="observed before the decision was made"` for the latter. The agent's own path checks the
  timing first (G2L-2), so no change is expected there.
* **Checkpoints** now also contain the store's experiment records and the pending outcome queries (`state.json`
  `experiments`, `pending_outcomes`); old checkpoints still load.
* `credit.checkpoint.capture_learner` is called by `agent.save` without `auth_config_hash`, so agent checkpoints
  record `""` for the authorization configuration (the task's `auth_scope` is inside `current_task`). If the runtime
  has a hash of the authorization configuration in force, passing it (`capture_learner(..., auth_config_hash=h)`)
  and checking it on load (`restore_learner(..., expected_auth_hash=h)`) makes CL-10's last item complete for the
  agent; DX cannot do it from credit/.

## INT3A-5 (requested of D / DX by INT3A): done in credit/

The stratified record split and the explicit "no held-out evidence" promotion reason are implemented in
`credit/learner.py` / `credit/checkpoint.py` (docs/DX.md, section INT3A-5). No change is needed in agent.py.

**Master decision on DX-1 (2026-11-28 ~02:30 UTC): not applied at the freeze.** With the line, the integration test
`tests/integration/test_int2_wiring.py::test_runtime_executes_in_the_computer_environment_without_grant_refusals`
failed 5 of 28 runs (the agent's choices change: in failing runs no `api` action succeeded within 16 steps); without it
0 of 10 (snapshot of HEAD with its own tests). DX measured that even with the line the selection does not run inside the
agent (every agent experiment has one unit; `holdout_experiments=False`), so the line would change behaviour without
making RC-08 hold. RC-08 stays partial; the credit-side API (`attach_experiments` / `evaluate_experiments`) is in place.

## Round 2 (2026-11-28): the exact hooks to apply (coordinator applies them)

### DX-1 hook (unchanged, now root-caused as safe)

agent.py, directly after `self.assigner = CreditAssigner(...)` (line 717 at 3e29595; same place at f54d8cc):
```diff
         self.assigner = CreditAssigner(self.model, self.store, rudder=self.rudder, cocoa=self.cocoa,
                                        selector=self.selector, beta=s.credit_beta, clock=self.clock)
+        self.assigner.attach_experiments(self.experiments)   # RC-08 (DX-1): experiments reach the store; selection runs
```

**Root cause of the 5 / 28 failures of `tests/integration/test_int2_wiring.py::test_runtime_executes_in_the_computer_environment_without_grant_refusals`
with the hook (0 / 10 without).** Measured with a driver that replicates the test exactly (its `_env_and_task`,
`agent_config(use_backbone=False, planner=False)`, `LiveRuntime`, 16 steps) and applies the hook by patching
`CamLiveAgent.__init__` in the driver process (agent.py untouched;
`/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DX/r2/dx1_driver.py`), 3 torch threads per run:
* **with the hook 1 failure in 30 runs, without it 1 failure in 22** (same code, same machine, interleaved);
* in every run of both arms the runtime **pauses after 5 consecutive verifier errors** (`_NoVerifier.verify` raises
  "not used" at every verification), so the test always gets exactly 5 decisions; it fails when none of the 5 is
  an `api` action that succeeded (the failing runs chose `internal.inspect` x2-3 and `wait`, or an `http.request`
  that failed). That is a property of the test, not of the hook: the agent's ids carry `os.urandom` bytes and the
  clock is `WallClock`, so runs differ, and a few percent of them choose no successful api action in 5 decisions;
* the hook path raised nothing in 30 runs (no exception in `learn()`, `record_experiments`, `evaluate_experiments`,
  no runtime error other than the verifier's), took 0.06 s per run in `evaluate_experiments` (median; max 0.14 s)
  and 0.009 s in `record_experiments` (max 0.14 s); it consumes no random numbers and nothing outside `credit/`
  reads the copied experiment records (grep: no reader of `store.experiments()` / `experiment_objectives` in the
  agent, predict/, reward/); the only effect on choices is through timing (wall clock).
The 5 / 28 vs 0 / 10 difference is not significant (Fisher exact p = 0.30); pooled with DX's runs, 6 / 58 with the
hook vs 1 / 32 without, p = 0.41; DX's runs alone 1 / 30 vs 1 / 22, p = 1.0. To make the hook unconditionally safe anyway, `ContinualLearner.learn` and `capture_learner` now catch and
record any failure of the experiment copy (`learner.credit_selection["error"]`, a warning) instead of letting it fail
a learning update or a checkpoint (commit 9a47d7d).
*Suggestion for the test's owner:* the assertion "one successful api action in the 5 decisions before the runtime
pauses" is flaky by construction (a few percent); a `_NoVerifier.verify` that returns "not yet verifiable" instead of
raising, or more decisions before the pause, would remove it.

### DX-2: authorization configuration in agent checkpoints (CL-10)

`credit.checkpoint.auth_config_hash(task)` (sha256 of the task's authorization-relevant fields as the learner receives
them: resources, interfaces, limits, data_permissions, stop_conditions, mode, origin). Exact diff for agent.py
(at f54d8cc, `save` line 2325 and `load` line 2345):
```diff
-from .credit.checkpoint import capture_learner, load_checkpoint, restore_learner, save_checkpoint_at
+from .credit.checkpoint import auth_config_hash, capture_learner, load_checkpoint, restore_learner, save_checkpoint_at
@@ def save(self, path: str) -> str:
         state = capture_learner(self.learner, extra_optimizers={"reward_meta": self.meta.optimizer},
                                 generators={"agent": self.gen, "decider": self.decider.generator},
                                 lifetime_context=self.meta.reward.lifetime.state.detach().clone(),
                                 current_task=self.task.to_dict() if self.task is not None else None,
+                                auth_config_hash=auth_config_hash(self.task),
                                 reward_version=self.meta.reward.version,
@@ def load(self, path: str) -> None:
         restore_learner(self.learner, st, extra_optimizers={"reward_meta": self.meta.optimizer},
-                        generators={"agent": self.gen, "decider": self.decider.generator})
+                        generators={"agent": self.gen, "decider": self.decider.generator},
+                        expected_auth_hash=auth_config_hash(self.task) if self.task is not None else None)
```
`restore_learner` then reports `auth_config_changed` when the task's authorization changed since the checkpoint
(the report is returned; the agent may log it). Test: `tests/credit/test_dx_checkpoint.py::test_auth_config_hash_is_the_tasks_authorization_fields_and_a_change_is_reported`.
With DX-1 applied, the agent's experiment records are in the store before every `save` (capture_learner copies the
attached registry first), which closes CL-10's other gap (G2-learn R2.4: `experiments: []`).

### Round-2 behaviour changes the agent will see (no diff needed)

* Slow steps: the guard no longer sees the world model / COCOA "degrade" through its own measurement (running
  statistics restored after the slow step's loss evaluation, deterministic COCOA validation): in the debug agent 94 %
  of slow steps are kept (was ~26 %), on the real configuration 4 of 8 (was 0-1 of 6-7). The DX-2 round-1 note "more
  slow steps are reverted" is superseded. Validation no longer consumes the global torch generator (COCOA).
* `CheckpointManager` (the agent's promotion checkpoints) keeps optimizer state only in the newest kept checkpoint; a
  rollback to a promoted checkpoint without it reports `optimizer_state_reset` and the learner's optimizers start
  fresh (the agent keeps its `reward_meta` optimizer as before).

**Master (2026-11-28 ~05:00 UTC): DX-1 and DX-2 applied** after DX's root cause (the wiring test pauses after five
decisions by construction; failure rates 6/58 with the hook vs 1/32 without, Fisher p 0.41; the hook path draws no
random numbers and cannot fail an update). Agent tests: predict/test_selection_agent, integration agent_runtime /
agent_recovery / int2_checkpoints / int2_queries / int2_selection, credit/test_dx_checkpoint — 26 passed
(`docs/evidence/test_results/M_dx12_agent.xml`).
