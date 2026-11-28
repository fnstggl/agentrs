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
