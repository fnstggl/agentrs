# Workstream D (stage 1) — public API of `cam_live.credit`

Everything is importable from its submodule. Importing `cam_live.credit` (or `.replay`, `.links`, `.simple`, `.select`,
`.smdp`) does not import torch. Tests: `tests/credit/`. Measured numbers: `docs/evidence/D/measured.json`.

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

## For stage 2 (continual learner)

* `LossProvider`s for `ReturnDecomposer` and `HindsightModel` on the real model's latents are not written yet;
  both modules take plain tensors (`DecisionBatch.z_chosen` along a trajectory; `b`, candidate embeddings `e_a`,
  `behaviour_probs.log()`), register them with `model.register_head(name, module)` so they are checkpointed.
* After each reconciliation: `store.retrain_queue()` -> train -> `store.pop_retrain(ids)`.
* `AgentAPI.save(path)`: `save_checkpoint_at(path, capture_state(model, optimizers, store=store, ...))`.
