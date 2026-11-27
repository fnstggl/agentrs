# Workstream D — public API of `cam_live.credit` (stage 1 + stage 2)

Everything is importable from its submodule. Importing `cam_live.credit` (or `.replay`, `.links`, `.simple`, `.select`,
`.smdp`) does not import torch. Tests: `tests/credit/`. Measured numbers: `docs/evidence/D/measured.json` (stage 1,
tensor level), `docs/evidence/D/diagnostics.json` (stage 2, real model at debug scale in LabWorld) and
`docs/evidence/D/real_update.json` (Qwen3-4B + 50m core). Sections 1-8 are the model-independent parts; 9-13 run on
`nn.model.CamLiveModel`. Integration notes for the agent are at the end.

## 1. `credit.replay` — durable experience store

```python
SqliteExperienceStore(path, *, provenance=None, seed=0, recent_window=256, recent_fraction=0.25, priority_alpha=0.6,
                      uniform_floor=0.1, delayed_min_s=1.0, blob_threshold=65536, durable=True, readonly=False,
                      eligible_default=True, item_conflict="keep_first", strict_items=False, strata=None)
```
Implements `contracts.ExperienceStore` (`append, update, get, by_decision, put_item, items, pending, sample, window,
count, flush`) on SQLite WAL + `<path>.blobs/`. One writer (`flock` on `<path>.lock`; a second writer raises
`StoreLockedError`); readers use `readonly=True`. Every public write is one committed transaction
(`synchronous=FULL`): a record is durable when the call returns. `with store.batch():` groups writes into one
transaction.

| call | meaning |
|---|---|
| `append(rec)` | new record; assigns `rec.seq` (lifetime head + 1) when it is 0; duplicate id raises `KeyError` |
| `update(rec)` | replace; only the changed part (static / result / dynamic) is rewritten |
| `add_outcome(obs, exp_id="") -> AddOutcomeResult(accepted, exp_id, duplicate, upgraded, final, reason, dedup_key)` | attach a measurement to the record that registered `obs.query_id`; same (dedup key, query) twice = dropped; a verified copy replaces an unverified one; `PENDING` is not an outcome; an outcome dated before its query is refused |
| `pending(now)`, `pending_all()`, `next_due()`, `pending_pointer()`, `query(query_id)` | unresolved queries (a query stays in `rec.pending_queries`; "pending" = no final outcome yet) |
| `sample(n, *, loss, strata=None, seed=None, provenance=None)` / `sample_ids(...)` | up to `n` **distinct** records eligible for `loss` (`loss="*"`: no filter). One provenance stratum unless several are named: `provenance` argument > constructor default > provenance of the newest record. `strata` weights over `("recent","informative","success","failure","delayed","family")`, default `DEFAULT_STRATA`. Deterministic for a given `seed`; with `seed=None` a counter-based stream that `state_dict()` checkpoints. `store.last_sample_info` says which stratum each draw came from. |
| `set_eligible(exp_ids, loss, flag)`, `refresh_eligibility(loss, predicate, since_seq=0, exp_ids=None)` | per-loss eligibility (`predicate` = `LossProvider.eligible`) |
| `set_success(exp_id, bool|None)`, `set_surprise(exp_id, x)`, `set_credit(exp_id, estimator, value|None, uncertainty=None)`, `labels(exp_id)` | sampling labels. Without an explicit label, success/failure is derived: failed or unverified execution = failure; goal-relevant BINARY (>= 0.5) / COUNT (> 0) outcomes = success; otherwise unknown. Surprise defaults to the largest resolved prediction NLL of the executed candidate. |
| `assign_window(exp_ids, window_id, kind="", ...)`, `window(id)`, `windows()` | meta-learning windows |
| `versions(kind="")`, `provenance_counts()`, `outcome_events(verified_only=False)` | model / reward versions with record counts; distinct outcome events (one per dedup key) |
| `iter_records(objective_id="", provenance=None, since_seq=0, until_seq=None, limit=None)`, `trajectory(objective_id)`, `get_many(ids)`, `by_action(action_id)`, `all()`, `head_seq()` | reads |
| `put_prediction / predictions(...)`, `put_experiment / experiment / experiments / experiment_records(id)` | prediction and experiment records |
| `put_blob(bytes) -> "blob:<sha256>"`, `get_blob(ref)` | large artifacts by reference |
| `mark_for_retraining(exp_ids, reason, outcome_key="", now=0)`, `retrain_queue(limit=None)`, `retrain_count()`, `pop_retrain(exp_ids)` | retrospective update queue |
| `state_dict()`, `load_state_dict(st)` | replay sampler indices for checkpoints (experience itself is never rolled back) |

Defaults that differ from a naive store, deliberately: items are immutable (`put_item` of different content under a
stored ref keeps the first version, warns once and counts in `store.stats`); `items()` skips unknown refs with a
warning (`strict_items=True` raises); `sample` returns fewer than `n` rather than repeating records.

## 2. `credit.links` — outcome linking

```python
OutcomeLinker(store, *, clock=None, retro_predecessors=0, window_half_life_s=None)
  .reconcile(outcomes: Sequence[OutcomeObservation] = (), events: Sequence[Observation] = ()) -> ReconcileReport
  .link_outcome(obs) -> LinkResult          .link_event(event: Observation) -> LinkResult
  .available_at_decision(rec) -> dict       .audit_leakage(exp_ids=None) -> list   # [] = no future information in any H_t
OutcomeLink(link_id, outcome_key, exp_id, decision_id, weight, uncertainty, evidence, outcome_ts, decision_ts,
            value, metric, verified, goal_relevant, created_at, detail, available_at_decision)   # .amount = weight*value
record_arm(rec) -> Optional[str]            # decision.diagnostics["arm"] or chosen.arguments["arm"]
```
Evidence, strongest first: declared query; `event.data["by_action" | "by_decision"]` maps or `"action_id" |
"decision_id"`; `event.parent_action_id`; artifacts (`data["artifact_id(s)"]`, `event.refs`) against
`result.artifacts`; `data["experiment_id"]` + `data["arm"]`; attribution windows declared by queries on the event's
metric. Equal support = equal fractional weights with uncertainty `sqrt(w(1-w))`; no evidence = no link; a decision
made after the outcome is refused and its share stays unattributed. `reconcile` stores outcomes (de-duplicated),
persists links (`store.links(outcome_key=..., exp_id=...)`) and queues the linked records (`"outcome_matured"` /
`"event_linked"`) and optionally their predecessors (`"return_changed"`).

## 3. `credit.estimators` / `credit.longdelay` — exact references (tabular, float64)

```python
reinforce_grad(theta, batch, baseline=None)            all_actions_grad(theta, batch, Q)
cocoa_grad(theta, batch, w)                            # w[t,s,a,u] = p(a|s,u)/pi(a|s) - 1
rudder_exact_redistribution(mdp, theta, batch, include_offset=True) -> rr[N,T]
rudder_grad(theta, batch, redistributed, baseline=None, immediate_only=False)
hca_return_grad(theta, batch, h_taken); return_conditioned_hindsight(batch, probs); explaining_away_redistribution(batch)
estimator_stats(per_traj, exact) -> dict               unbiasedness(per_traj, exact) -> {max_abs_z, rms_z, rel_error, ...}
LongDelay(T=64, ...).stats(n, seed, w_door_key=None)   # key-decision bias / variance / SNR for delays up to 1024+
```
All `*_grad` return per-trajectory gradients `[N, S, A]`; MDPs come from `cam_live.diagnostics.tabular`.

## 4. `credit.rudder` — return decomposition

```python
ReturnDecomposer(d_in, d_model=128, n_layers=2, n_heads=4, d_goal=0, beta=0.0, scale=1.0, dropout=0.0, time_unit_s=60.0)
  .forward(x[B,T,d_in], mask[B,T], e_g=None, times[B,T]=None) -> pred[B,T+1]   # scaled; index t has seen decisions < t
  .loss(x, mask, G[B], matured[B], e_g=None, times=None) -> (Tensor | None, metrics)   # None: nothing matured
  .redistribute(x, mask, G, matured, e_g=None, times=None) -> Redistribution
  .fit_scale(G, matured=None) -> float
Redistribution: contrib, diff, signal, return_to_go [B,T]; pred [B,T+1]; offset, residual [B]; discount [B,T]; mask; matured
  .conservation_error(G) -> [B]            # |sum_t Gamma_t c_t - G|
write_credit(store, exp_ids: Sequence[Sequence[str]], red, name="rudder") -> int
duration_discount(tau, beta); discounted_return(value, outcome_time, start_time, beta)
```
`Gamma_t = exp(-beta * seconds since the first decision)`; `sum_t Gamma_t * contrib_t = G` exactly. **Use `signal`
(= `(G - pred_before_t) / Gamma_t`, the redistributed return-to-go with the prefix prediction as baseline) as the
learning signal**; `contrib`/`diff` used as an immediate reward are only right for a well-trained decomposer. Rows
that have not matured are NaN and `write_credit` skips them.

## 5. `credit.cocoa` — hindsight model and contribution coefficients

```python
HindsightModel(d_b, d_u, d_a, d_hidden=128, d_key=64, n_members=4, n_elapsed=2, elapsed_units=(60.0, 1.0), seed=0)
  .forward(b, u, cand, cand_mask, logp_policy, elapsed=None) -> h[B,N]
  .loss(b, u, cand, cand_mask, taken, logp_policy, elapsed=None, generator=None, bootstrap=True) -> (Tensor, metrics)
  .contribution(b, u, cand, cand_mask, logp_policy | None, elapsed=None, *, outcome_value=None, clip=10.0, min_prob=1e-3,
                open_ended=False, tracked_proposals=True, n_eff=None, min_ess=0.0) -> CounterfactualCredit
CounterfactualCredit: status[B] ("supported" | "unsupported"), reasons, learned_association[B,N],
  counterfactual_estimate[B,N] (NaN where unsupported), experimental_effect {row: ArmDifference} (only via
  .attach_experiment(row, randomized ArmDifference)), w, w_std, h, clipped, candidate_supported
check_support(logp_policy, cand_mask, *, min_prob=1e-3, open_ended=False, tracked_proposals=True, norm_tol=1e-3,
              n_eff=None, min_ess=0.0) -> SupportReport(status, reasons, candidate_ok)
candidate_ess(keys, chosen, probs) -> {key: Kish ESS};  effective_sample_size(weights) -> float
arm_contrast(w[B,N], outcome_value[B], treatment, control) -> (estimate, se, n) | None
coefficients_table(model, b[S,.], u[U,.], cand[A,.], logp[S,A], elapsed=None, clip=1e9) -> w[S,A,U]
```
`elapsed` is raw `[B, n_elapsed]` (seconds, decisions). Refusal reasons: `propensities_missing`,
`propensities_invalid`, `near_zero_propensity`, `open_ended_without_tracked_proposals`, `insufficient_support`,
`no_candidates`.

## 6. `credit.simple` and `credit.select`

```python
SimpleAttribution(store, *, beta=0.0, goal_relevant_only=True, value_fn=None, combine="declared_first")
  .credit(exp_ids=None) -> {exp_id: CreditValue(value, uncertainty, n_links, by_evidence, provisional)}
  .apply(exp_ids=None) -> same, and writes record.credit["simple"] / credit_uncertainty["simple"]
credit_from_links(links, *, beta=0.0, goal_relevant_only=True, value_fn=None, combine="declared_first")
time_decayed_attribution(outcome_ts, [(exp_id, decision_ts)], *, window_s, half_life_s, background=0.0) -> {exp_id: (w, sd)}
arm_difference(treatment, control, *, confidence=0.95, randomized=False, ...) -> ArmDifference | None   # Welch
experiment_effects(store, experiment_id, *, control=None, confidence=0.95, metric_prefix="", value_fn=None) -> [ArmDifference]
student_t_cdf(t, df); student_t_ppf(p, df); discount(dt, beta)

EstimatorSelector(*, baseline="simple", min_experiments=5, max_mean_sq_z=3.0, min_sign_agreement=0.8,
                  min_improvement=0.1, log_path=None, clock=None)
  .add_evidence(ExperimentEvidence)   .add_from_store(store, experiment_id, estimators, control=None, extra=None)
  .select() -> SelectionRecord        .default  .scores()  .history  .state_dict()  .load_state_dict()
```
`combine="declared_first"`: once a decision's declared queries are answered they are its credit; event links before
that give a *provisional* value and are never added on top. Only `randomized=True` evidence is scored; the
selection and its evidence are appended to `log_path` (JSON lines).

## 7. `credit.smdp` — temporally extended actions

```python
discount(dt, beta); smdp_return(rewards, times, beta, t0=None); smdp_target(reward, duration_s, next_value, beta)
build_trace(action, result, step_results=None) -> TraceNode;  attach_trace(rec, step_results=None) -> TraceNode
get_trace(rec); validate_intervention_record(rec) -> [problems]; smdp_transition(rec) -> SMDPTransition
CreditLedger(store, max_depth=8)
  .assign(links) -> [LedgerEntry]     .assign_direct(outcome_key, exp_id, weight, counted=True)   # DoubleCreditError
  .counted_weight(key)  .by_level(key)  .independent_experiences(keys=None)  .apply(entries, name="smdp")
```
Counted credit goes to the top decision of the `parent_decision_id` chain; lower levels hold uncounted *inherited*
entries (`credit["smdp_inherited"]`). The store refuses counted weights above 1 per outcome.

## 8. `credit.checkpoint`

```python
CheckpointState(weights, optimizers, fast_state, lifetime_context, current_task, replay_state, pending_pointer,
                schedules, rng, auth_config_hash, model_version, reward_version, arch_signature, extra)
capture_state(model=None, optimizers=None, *, groups=None, store=None, generators=None, **fields) -> CheckpointState
restore_state(state, model=None, optimizers=None, *, strict=False, store=None, generators=None, rng=True,
              expected_auth_hash=None) -> report
save_checkpoint(root, state, *, parent="", notes="", clock=None) -> CheckpointInfo      # <root>/ckpt_<seq>_<digest>
save_checkpoint_at(path, state, *, parent="", notes="", clock=None) -> CheckpointInfo   # caller-chosen directory
load_checkpoint(path_or_root, checkpoint_id=None, *, map_location="cpu", verify=True) -> CheckpointState
verify_checkpoint(path) -> [problems]; list_checkpoints(root); latest_valid(root, verify=True); recover(root) -> report
collect_weights(model, groups=None, *, include_buffers=True) -> {group: {name: Tensor}}
load_weights(model, weights, *, strict=False) -> LoadReport(loaded, missing, unexpected, mismatched)   # CheckpointMismatch if strict
optimizer_state(optimizer, model) -> dict;  load_optimizer_state(optimizer, model, saved) -> report
capture_rng(generators=None); restore_rng(state, generators=None); tensor_hash(tensors); signature_differences(a, b)
export_weights_only(state, out_dir, *, config=None) -> config;  export_weights(checkpoint_path, out_dir, *, config=None)
load_weights_only(path, *, map_location="cpu") -> (weights by group, config);  export_keys(path)
CheckpointManager(root, *, keep_last=3, keep_promoted=5, gates=PromotionGates(...), clock=None)
  .save(state, notes="", parent=None)  .load(checkpoint_id=None)  .list()  .latest()  .get(id)
  .promote(checkpoint_id, *, validation: dict, retention: dict, notes="") -> PromotionDecision
  .promoted()  .rollback(reason="") -> (CheckpointInfo, CheckpointState)  .prune()  .promotion_log()
PromotionGates(primary="", min_validation_gain=0.0, max_validation_drop=0.0, max_retention_drop=0.02, min_validation={})
```
A model with `param_groups()` / `trainable_state()` / `arch_signature()` (`nn.api.LatentModelAPI`) is grouped
automatically; the frozen backbone is never written. Loading matches tensors and optimizer state **by parameter
name**: added heads are reported in `missing` / `fresh`, removed ones in `unexpected` / `dropped`. Metrics are
oriented higher = better; metrics named like the agent's own reward are ignored by `promote`.

Store additions (stage 2): `put_objective_outcome(objective_id, *, status, value=0.0, evaluated_at=0.0, verified=False,
body=None)`, `objective_outcome(id)`, `objective_ids(closed=None, family="", provenance=None)`, `objective_of(exp_ids)`,
`set_credits(exp_id, {estimator: (value, uncertainty) | None}, *, version=0, reason="", ts=0.0) -> {estimator: change}`,
`credit_history(exp_id="", estimator="")`, `kv_get(key, default=None)`, `kv_put(key, value)`.
Checkpoint additions: `CheckpointState.aux_state` (provider tensors, `aux.pt`); `load_optimizer_state` clears the state
of parameters that had none when the checkpoint was written (report key `unstepped`), so a rollback is exact.

## 9. `credit.learner` — the continual learner

```python
LearnerSettings(batch_size=16, slow_batch_size=4, grad_last=8, adapter_every=8, max_share=0.9,
                min_independent_outcomes=4, min_surprise=0.0, steps_per_trigger=1, max_steps=8, fresh_fraction=0.25,
                val_fraction=0.15, val_split="objective", val_batch=32, probe_size=8, probe_tol=1e-7, strata=None,
                provenance=None, warmup_steps=0, decay_steps=0, min_lr_factor=0.1, with_next=False, slow_guard=True,
                slow_guard_tol=0.05, betas=(0.9, 0.999), seed=0)
LearnerSettings.from_cfg(cfg, **over)        # cfg.optim.batch_size / adapter_every + cfg.section("credit")["learner"]

ContinualLearner(model: LatentModelAPI, store: SqliteExperienceStore, cfg: CamLiveConfig, *, clock=None, settings=None,
                 log_path=None, credit: CreditAssigner | None = None)
  .register(provider: LossProvider) -> provider         .add_probe(name, fn(model, batch, store) -> {key: Tensor})
  .poll() -> UpdatePlan | None                           # trigger: independent matured outcomes in store.retrain_queue()
  .learn(plan: UpdatePlan) -> List[LearningUpdateRecord] # frequent updates (+ slow when due), or a forced slow update
  .fast_update(phi, y, kind, *, exposure=None, censored=None, exp_id="", trigger="outcome_observed") -> LearningUpdateRecord
  .reset_fast(kind=None)                                 # new objective: fast beliefs back to the learned prior
  .validate(providers=None, n=None) -> {provider: held-out loss}
  .refresh_eligibility(exp_ids=None, *, full=False)      .is_validation(record) -> bool
  .set_probe(exp_ids=None)   .probe() -> {key: Tensor}   .group_hashes()   .status()   .state_dict() / .load_state_dict()
  .optimizers {group: AdamW}   .fast_state   .counters   .model_version ("m<frequent>.<slow>")   .history   .share_log
take(batch, rows) -> DecisionBatch;  records_only_batch(model, records) -> DecisionBatch
encode_with_adapter_grad(model, store, records, *, grad_last=None, with_next=False) -> DecisionBatch
```
* One AdamW per group `experiential`, `heads`, `semantic_adapters` (lr / clip from `cfg.optim`; weight decay on
  tensors with >= 2 dims). The `reward` group is never in an optimizer and gradients are taken with
  `torch.autograd.grad` w.r.t. the learner's own parameters, so `reward` tensors get no `.grad` and no step.
* `poll()` returns a plan only when at least `min_independent_outcomes` *units* matured (unit = the top-level decision
  an outcome belongs to; correlated rows of the same decision count once; time is not an input). `learn(plan)` first
  re-assigns credit for `plan.exp_ids` (when `credit` is given), refreshes eligibility, trains, then pops the queue.
* A frequent update draws one replay batch (a quota per provider from the store's strata + the triggering
  experience), encodes it once, gives each provider its eligible rows, measures each provider's gradient on the
  experiential group, scales the dominant one down so its share is at most `max_share`, clips per group, steps.
* A slow update runs every `adapter_every` applied frequent updates (or when the plan lists `TIMESCALE_SLOW`):
  the batch is encoded with an adapter graph for the newest `grad_last` context items and all candidate / goal
  texts; providers with `timescale == TIMESCALE_SLOW` and frequent providers that do not declare
  `adapter_grad = False` contribute; only the adapter optimizer steps; with `slow_guard` a step that raises the mean
  held-out loss by more than `slow_guard_tol` (relative) is undone.
* `LearningUpdateRecord.notes` is JSON: `experiential_share`, `experiential_share_applied`, `balance_scale`,
  `strata`, `rows`, `metrics`, `lr`, `behaviour_changed`, `flag` (weights changed, probe did not), `skipped`,
  `adapter_grad_norm` / `adapter_delta_norm` (`lora`, `bridge`), `guard`.
* Optional provider attributes: `uses_batch = False` (reads its own data, always called), `needs_latents = False`
  (only `batch.records` is used: no encoding), `adapter_grad = False` (not part of slow updates),
  `state_dict()` / `load_state_dict()` (saved in checkpoints).

## 10. `credit.providers` — losses and the credit assigner

```python
ReturnDecompositionProvider(model, store, *, beta=0.0, weight=1.0, d_model=128, n_layers=2, n_heads=4,
                            max_trajectories=8, max_len=64, use_belief=True, detach_latents=True, require_closed=True,
                            verified_only=False, head_name="credit_rudder", name="rudder", seed=0, encoder=None)
  .loss(model, batch, store)  .eligible(record)  .redistribute(objective_ids) -> {objective_id: RudderCredit}
HindsightProvider(model, store, *, beta=0.0, weight=1.0, d_hidden=128, d_key=64, n_members=4, max_trajectories=8,
                  max_pairs=512, max_len=64, min_prob=1e-3, clip=10.0, require_closed=True, verified_only=False,
                  head_name="credit_cocoa", name="cocoa", seed=0, encoder=None)
  .loss(model, batch, store)  .eligible(record)  .credit(objective_ids) -> {exp_id: CocoaCredit}
BridgeGroundingProvider(model, *, weight=1.0, max_rows=4, max_chars=240, name="bridge_grounding")   # TIMESCALE_SLOW
CreditAssigner(model, store, *, rudder=None, cocoa=None, selector: EstimatorSelector = None, beta=0.0, clock=None,
               goal_relevant_only=True)
  .assign(exp_ids=None, *, objective_ids=None, reason="outcome_matured") -> CreditUpdate
  .on_reconcile(report: ReconcileReport) -> CreditUpdate
  .credited_returns(records, estimator=None) -> CreditedReturns(estimator, key, values [B] f64, mask [B], uncertainty [B])
  .compare_with_experiment(experiment_id, *, treatment, control, confidence=0.95) -> ExperimentEvidence | None
  .select() -> SelectionRecord   .default   .version   .state_dict() / .load_state_dict()
trajectory_return(store, records, *, beta=0.0, require_closed=True, verified_only=False, goal_relevant_only=True)
    -> TrajectoryReturn(objective_id, exp_ids, times, events, G, closed, n_pending, matured, reason)
TrajectoryEncoder(model, store, *, max_len=64, chunk=32).encode(objective_id, records=None, *, grad=False)
outcome_features(values) -> [K, 10];  consequence_text(record);  is_top_level(record);  top_index(records)
```
* Return convention: an outcome of value v observed at tau is worth `v * exp(-beta * (tau - t))` at a decision made
  at t; a trajectory return is referenced at its first decision; an outcome event (dedup key) enters once.
* A return is *matured* when the objective was closed (`store.put_objective_outcome`) and no goal-relevant query of
  its decisions is outstanding. Until then `rudder*` and `cocoa` are withheld (absent from `record.credit`, never
  zero); `simple` appears as soon as a linked outcome or event exists.
* `record.credit` keys: `simple`; `rudder` (learning signal: redistributed return-to-go minus the prefix
  prediction), `rudder_rtg`, `rudder_contribution`; `cocoa` (own declared outcomes + sum over later outcomes of
  `w * v`). `credit_uncertainty`: link uncertainty (`simple`), `|G - F(whole trajectory)|` (`rudder`), ensemble
  spread (`cocoa`). COCOA is `unsupported` (no entry, reasons in `CreditUpdate.unsupported`) without usable logged
  propensities. Every changed value gets a row in `store.credit_history` under a new credit version.
* `credited_returns` reads the key of the estimator the selector currently trusts (`assigner.default`, initially
  `simple`); records without that credit are masked.
* `compare_with_experiment` compares the matured return-to-go of the randomised decisions of a registered
  experiment with each estimator's estimate (`rudder`: difference of mean `rudder_contribution`; `cocoa`: mean
  contrast of the per-candidate credit; `simple`: difference of mean credit where it exists). Arms are read from
  `decision.diagnostics["arm"]` and `["arms"] = {arm: candidate index}`.

## 11. `credit.retention`

```python
FunctionSpec(name, fn(model, batch, store) -> values | (values, mask), kind="mse" | "kl", weight=1.0)
latent_functions() -> [latent_b, latent_z]
AnchorSet(functions=None, *, max_per_family=256)
  .add(model, store, records, *, family=None, batch_size=32) -> n_added     .penalty(model, store, exp_ids) -> (loss, metrics)
  .ids(family=None)  .families()  .state_dict() / .load_state_dict()
RetentionProvider(anchors, *, weight=1.0, n_anchors=8, name="retention", seed=0)        # uses_batch = False
evaluate_families(model, store, {family: exp_ids}, eval_fn, *, batch_size=32) -> {family: FamilyEval}
measure_forgetting(model, store, before: {family: FamilyEval}, eval_fn, *, batch_size=32) -> {family: {
    nll_before, nll_after, nll_change, nll_change_se, calibration_error_before/after, calibration_change,
    mean_sq_z_before/after, coverage90_before/after, decision_agreement, action_tv, n, n_scored}}
retention_metrics(report) -> {"<family>/neg_nll", "<family>/decision_agreement"}        # higher = better
validation_metrics({provider: held-out loss}) -> {"heldout/neg_<provider>": ...}
```
`eval_fn(model, batch, store) -> {"nll": [B], "z": [B], "action_probs": [B, N]}` (NaN where a record has no matured
outcome). Evaluate on records the learner holds out (`learner.is_validation`).

## 12. `credit.checkpoint` — complete learner state, promotion, transfer

```python
capture_learner(learner, *, extra_optimizers=None, generators=None, lifetime_context=None, current_task=None,
                auth_config_hash="", reward_version="", extra=None) -> CheckpointState
restore_learner(learner, state, *, strict=False, extra_optimizers=None, generators=None, rng=True,
                expected_auth_hash=None) -> report
save_learner(root | CheckpointManager, learner, *, notes="", **capture_kw) -> CheckpointInfo
load_learner(root | CheckpointManager, learner, *, checkpoint_id=None, strict=False, map_location="cpu", **restore_kw) -> report
promote_learner(manager, learner, *, retention: dict, validation: dict = None, notes="", rollback=True, **capture_kw)
    -> (PromotionDecision, CheckpointInfo, rollback report | None)
export_learner_weights(learner, out_dir) -> config
load_exported_model(export_dir, *, cfg=None, register_heads: Callable[[model], Any] = None, strict=True, device=None)
    -> ExportedModel(model, fast_state, report, config)
```
A learner checkpoint holds: all trainable groups (including `reward`), the learner's optimizers (+ `extra_optimizers`,
e.g. the meta optimizer), fast belief state, lifetime context, replay sampler indices, pending-outcome pointer,
schedules / counters / probe ids, RNG states, versions, architecture signature, `extra["config"]`,
`extra["credit"]` (selector evidence) and provider tensors. `restore_learner` expects a learner built from the same
configuration with the same providers registered. `load_exported_model` builds `nn.model.build_model(cfg)` in the
calling process, calls `register_heads(model)`, loads the exported tensors and returns a fresh fast state at the
exported learned prior; nothing else exists in that process.

## 13. `credit.labkit`, `credit.diagnose` — LabWorld evidence

```python
build_lab(store_path, cfg=None, *, world=True, policy=True, credit=True, bridge=False, beta=0.0, settings=None,
          clock=None, log_path=None, durable=False, selector=None, model=None, **store_kw) -> Lab
register_lab_heads(model, store=None, *, world=True, policy=True, credit=True, beta=0.0, seed=0) -> {name: provider}
LabRun(store, linker, world, *, task=None, seed=0, max_context=40).decide(policy_fn, *, candidates_fn=None, arms=None)
  .reconcile()  .mature()  .close() -> VerifiedOutcome
collect_lifetime(store, linker, world, policy_fn, n_decisions, *, seed=0, candidates_fn=None, arms=None, mature=True,
                 close=True, tag="") -> LabLifetime;  collect_family(store, linker, base, n_worlds, n_decisions, ...)
WorldPredictionProvider(model, ...)   # local wrapper of predict.world (B, stage 1); evidence only
python -m cam_live.credit.diagnose [--only learning,credit,retention,restart,export] [--out PATH] [--work DIR]
python -m cam_live.credit.diagnose --real          # Qwen3-4B + 50m core, shared GPU lock
```

## Integration notes (for `agent.py`)

* Build once: `store = SqliteExperienceStore(run/STORE)`, `linker = OutcomeLinker(store, clock=clock,
  retro_predecessors=64)`, the providers of B / C / D (each registers its heads on the model), `assigner =
  CreditAssigner(model, store, rudder=..., cocoa=..., beta=...)`, `learner = ContinualLearner(model, store, cfg,
  clock=clock, credit=assigner, log_path=run/LEARNING)`, `learner.register(p)` for every provider, probes with
  `learner.add_probe`.
* `AgentAPI.reconcile(outcomes, verified)`: `linker.reconcile(outcomes=..., events=...)`; for each outcome
  `learner.fast_update(phi, y, kind, ...)`; when `verified` closes the objective and its signature was checked,
  `store.put_objective_outcome(...)` and queue its records (`store.mark_for_retraining(ids, "outcome_matured")`).
* `AgentAPI.learn(budget_s)`: `plan = learner.poll()`; `learner.learn(dataclasses.replace(plan, budget_s=budget_s))`
  if a plan was returned, else nothing (no evidence, no learning).
* `AgentAPI.save(path)` / `load(path)`: `save_checkpoint_at(path, capture_learner(learner, extra_optimizers={"reward":
  meta_opt}, lifetime_context=..., current_task=..., auth_config_hash=..., reward_version=...))` and
  `restore_learner(learner, load_checkpoint(path), extra_optimizers=...)`.
* New objective: `learner.reset_fast()`; weights, optimizers, reward parameters, lifetime context and experience stay.
* The policy loss of the reward workstream reads `assigner.credited_returns(batch.records)`; `values` are measured
  outcome units, `mask` says where they exist. D never passes them to the learner as a reward.
