# Workstream C — public API of `cam_live.reward` (stage 1 + stage 2)

Part I (first sections): model-independent mathematics and machinery — tensors, parameter dictionaries, contract
records. Part II (last section, "On the real model"): `PolicyImprovementLoss`, `MetaLearner`, `SignalAssembler`,
`GoalFreeMode` and the diagnostics, wired onto `CamLiveModel`, the experience store and LabWorld.

Measured numbers: `docs/evidence/C/exact_meta_gradient.json`, `docs/evidence/C/meta_cost.json`
(`PYTHONPATH=cam_live/src .venv/bin/python -m cam_live.reward.evidence --out cam_live/docs/evidence/C`),
`docs/evidence/C/diagnostics.json` (`... -m cam_live.reward.diagnose --out cam_live/docs/evidence/C`) and
`docs/evidence/C/meta_cost_50m.json` (`... -m cam_live.reward.diagnose --gpu-cost`).

## reward.meta — the engine

```python
Params = Dict[str, Tensor]
InnerLoss = Callable[[Params, Params, Any], Tensor]     # (theta, psi, support_batch) -> scalar
LogProbFn = Callable[[Params, Any], Tensor]             # (theta, batch) -> [n] log pi_theta(logged action)

class MetaGradient:
    def __init__(self, inner_loss: InnerLoss, logp_fn: LogProbFn, *, inner_lr=1e-2, K=2, mode="unroll",
                 weight_clip=10.0, min_ess=4.0, min_ess_fraction=0.05, min_propensity=1e-6,
                 min_independent=1, self_normalise=False, ema=0.1, grad_transform=None)
    def inner_step(self, theta, psi, batch, inner_lr=None, *, create_graph=True, first_order=False,
                   theta_sg=None) -> Tuple[Params, Tensor]
    def inner_update(self, theta: Params, psi_params: Params, support_batches, inner_lr=None, K=None,
                     create_graph=True, on_policy: bool | Sequence[bool] = False, mode=None, n_lifetimes=1
                     ) -> InnerTrace
    def outer_surrogate(self, theta_K: Params, query: QueryData, estimator="on_policy") -> OuterEstimate
    def meta_gradient(self, trace: InnerTrace, query: QueryData, estimator="on_policy", *,
                      score_baseline=None) -> MetaGradResult
    @staticmethod
    def apply(optimizer, reward_params: nn.Module | Mapping[str, Tensor], result: MetaGradResult,
              clip: float | None = None) -> Dict[str, float]

@dataclass class QueryData:  batch; utility [n]; baseline=None; behaviour_logp=None; weights=None;
                             groups=None; lifetime=None; verified=None
@dataclass class InnerTrace: thetas; psi; batches; on_policy; inner_lr; mode; has_graph; score; inner_losses
                             .theta_K  .K  .detached() -> Params
@dataclass class MetaGradResult: supported; reason; estimator; grads: Params (= dJ/dpsi, ascent); grad_norm;
                             objective; n; n_groups; ess; max_weight; clipped_fraction; K; inner_lr; mode;
                             inner_losses; score_term_used;  .record_fields() -> dict for MetaUpdateRecord

policy_gradient_inner_loss(logp_fn, reward_fn, *, value_fn=None, value_coef=0.5, entropy_fn=None,
                           entropy_coef=0.0, normalise=True, loo_baseline=True) -> InnerLoss
functional_logp(policy: nn.Module, inputs="x", actions="a", mask="a_mask") -> LogProbFn
functional_reward(reward: nn.Module, inputs="reward_inputs") -> Callable[[Params, Any], Tensor]
discounted_returns(r, mask=None, gamma=None) -> Tensor
finite_difference(objective: Callable[[Params], float], psi: Params, *, eps=1e-5, coords=None) -> Tensor
```

Semantics the caller must respect

* `theta` is the adaptable subset only (policy head + decision trunk); closures run on **cached latents**. The
  window is truncated: `inner_update` detaches `theta` first.
* `support_batches` is a list (step k uses `[k % len]`) or a callable `(k, detached theta_k) -> batch` for
  on-policy collection between steps. Set `on_policy[k]=True` only when batch k's actions were sampled from
  `pi_{theta_k}`; then the inner-data score term is added (needed for unbiasedness when K >= 2).
* `estimator="on_policy"`: query actions were sampled from `trace.detached()`.
  `estimator="importance"`: query rows were logged by another policy; needs `behaviour_logp`. The result is
  `supported=False` with `reason` in {`missing_propensities`, `invalid_propensities`, `propensity_below_floor`,
  `effective_sample_size_too_small:<ess>`, `insufficient_independent_outcomes:<n><<min>`,
  `unverified_outcome_in_query`, `non_finite_utility`, `empty_or_misaligned_query`, `inner_update_has_no_graph`}
  and **no gradient**; `apply` then changes nothing.
* `QueryData.groups` = independent-outcome group per row (from `windows`); the default baseline leaves the
  row's whole group out. `QueryData.utility` must come from `integrity.OutcomeLedger` entries.
* Modes: `"unroll"` (exact for the window), `"checkpoint"` (same gradient, adjoint sweep: 2.6x less memory at
  2048 rows/step, K=8, but ~2x time and no memory gain when parameter tensors dominate), `"first_order"`
  (biased; no measured cost advantage — comparison only).
* A parallel-lifetime dimension is supported: theta tensors with a leading `[R, ...]`, batches carrying
  `"lifetime"` [n] and `n_lifetimes=R`.
* While `bootstrap_mix > 0` the live policy was not updated by the pure learned reward, so the meta window's
  `theta'(psi)` is a shadow update and the query must use `estimator="importance"`.
* **Stop-gradient inside the inner loss.** A quantity the inner loss holds fixed for the inner derivative (the
  value baseline of the advantage) is still a function of `theta_k(psi)`; writing it as `v.detach()` silently drops
  that path from the unroll for K >= 2 (measured bias on the real model: 1.7 % at K=2, 4 % at K=4). An inner loss
  with attribute `stopgrad_aware = True` is called as `inner_loss(theta, psi, batch, theta_sg=...)` and must
  evaluate such quantities on `theta_sg` (same values, different graph nodes). `policy_gradient_inner_loss` with a
  `value_fn` and `PolicyImprovementLoss.inner_loss` do this; the adjoint ("checkpoint") sweep handles it too.
* `grad_transform(names, grads)` is applied to the inner gradient inside the graph (`inner.GroupClip`: per-group
  norm clip) and is part of the differentiated function.

## reward.intrinsic

```python
class IntrinsicReward(nn.Module):
    def __init__(self, d_model, n_signals=N_SIGNALS, d_hidden=256, d_life=64, bound=1.0, use_action=True,
                 out_init_std=0.02)
    def forward(self, b_t, e_a, b_next, e_g, signals, life_ctx=None, signal_mask=None) -> Tensor   # [B]
    def reward_sequence(self, b_t, e_a, b_next, e_g, signals, signal_mask=None, h0=None, advance=False
                        ) -> Tuple[Tensor, Tensor, Tensor]          # (r [T], ctx [T, d_life], h_T)
    def step_features(...) -> Tensor;  def head(x, life_ctx=None) -> Tensor
    def functional(self, psi, *, sequence=False, **inputs) -> Tensor
    def as_reward_fn(self, inputs="reward_inputs", sequence=False) -> Callable[[psi, batch], Tensor]
    def meta_parameters(self) -> List[nn.Parameter]      # exactly the reward group
    def psi(self) -> Dict[str, nn.Parameter]
    def observe(self, b_t, e_a, b_next, e_g, signals, signal_mask=None) -> None    # update input statistics
    def freeze_stats(on=True);  stats_frozen;  with reward.meta_window(): ...
    version: str                                         # "rpsi1-s1.0-<sha256[:16]>"
class LifetimeContext(nn.Module):  encode(x_seq, h0=None) -> (ctx, h_T);  advance(x_seq) -> ctx;
                                   new_lifetime();  buffers: state [d_life], steps
class RunningNorm(nn.Module):      update(x, mask=None);  forward(x, mask=None);  freeze(on=True)
```

`e_g=None` selects the learned null goal (goal-free mode). `life_ctx=None` uses the stored lifetime state.
`reward_sequence(..., advance=True)` is called once per step of real experience; inside a meta window call it
with `advance=False` (or `as_reward_fn(sequence=True)`) and the window's `h0`. Register the module with
`model.register_head("reward", reward, group=GROUP_REWARD)`.

## reward.signals

`SIGNAL_SCHEMA_VERSION = "1.0"`, `N_SIGNALS = 33`, `SIGNAL_NAMES`, `SIGNAL_TRANSFORMS`, `schema()`,
`schema_hash()`.

```python
@dataclass class RewardSignals:   # 15 scalar fields ([B] tensors or None) + status / failure_class (str lists)
    @classmethod from_step(*, cost: Cost = None, result: ActionResult = None, **scalars) -> RewardSignals
    @classmethod stack(items) -> RewardSignals
    def to_tensor(dtype=torch.float32, device=None) -> Tensor     # [B, 33], fixed transforms, missing = 0
    def mask(dtype=torch.float32, device=None) -> Tensor          # [B, 33]
    def to_dict() / from_dict(d)
```
Scalars, in order: success_before, success_after, delta_v, action_quality, info_gain, voi, cost_money,
cost_seconds, cost_compute_s, elapsed_s, prediction_error_std, epistemic_before, epistemic_after,
verified_progress_delta, context_novelty; then 6 `status.*` and 12 `failure.*` one-hots.

## reward.bootstrap

```python
learning_progress(loss_before, loss_after) -> Tensor
class LearningProgressTracker:  record_before(key, loss);  progress(key, loss_after) -> float | None
class BootstrapReward:  __call__(*, learning_progress=None, delta_v=None, epistemic_before=None,
        epistemic_after=None, verified_delta=None, verified_mask=None, context_keys=None) -> BootstrapOutput
        # BootstrapOutput: total [B], components {name: [B]}, label="bootstrap", version="bootstrap-1"
class RewardMixer:  __init__(mix=1.0, step=0.1, min_mix=0.0, min_independent=8, tolerance=0.0)
    combine(r_bootstrap, r_learned, *, reward_version, bootstrap_components=None, exp_ids=None) -> MixedReward
    update(diffs: Sequence[float]) -> PromotionDecision        # one number per independent held-out group
    log (deque of dicts);  decisions;  state_dict() / load_state_dict()
paired_policy_value_diffs(logp_learned, logp_bootstrap, behaviour_logp, utility, groups, weight_clip=10.0)
```
`MixedReward.record_fields(i)` gives `intrinsic_reward`, `bootstrap_reward`, `reward_version` for
`ExperienceRecord`; `RewardMixer.mix` is `MetaUpdateRecord.bootstrap_mix`.

## reward.windows

```python
independence_key(rec: ExperienceRecord) -> str
verified_outcomes(rec, outcome_filter=None) -> List[OutcomeObservation]
class WindowBuilder:  __init__(min_independent_outcomes=8, min_gap_s=0.0, provenance="live",
                               require_propensities=False, outcome_filter=None)
    cut(records, boundary_ts, start_ts=-inf, end_ts=inf) -> WindowPair
    blocks(records) -> List[List[ExperienceRecord]];  pairs(records) -> List[WindowPair]
@dataclass WindowPair: pair_id; support: MetaWindow; query: MetaWindow; eligible; reason; gap_s; dropped
@dataclass MetaWindow: window_id; role; exp_ids; t_start; t_end; groups; row_groups; .n_rows; .n_independent
held_out_groups(groups, fraction=0.25, salt="") -> Set[str]
```
`pair.query.row_groups` maps to `QueryData.groups` (via `torch.unique(..., return_inverse=True)`);
`pair.reason` goes to `MetaUpdateRecord.skipped_reason`.

## reward.integrity

```python
scale_normalise(r, group=None, eps=1e-12);  bound_reward(raw, bound);  params_hash(params)
assert_optimizers_disjoint(inner_optimizers, reward_params);  assert_reward_optimizer_exclusive(opt, reward_params)
depends_on(value, params) -> bool;  assert_no_grad_path(value, params, what="loss")
with PsiGuard(reward_params, name): ...         # around every learner step that is not the meta-update
dedup_outcomes(outcomes) -> (kept, dropped)
screen_observation(o, *, imagined=False, check_measurement=None) -> (ok, reason)
screen_verified(v, check_signature, *, objective_origin=Trust.OPERATOR) -> (ok, reason)
is_prospective(pred: PredictionRecord, obs: OutcomeObservation) -> bool
class OutcomeLedger:  __init__(check_signature, check_measurement=None, path=None)
    admit_verified(v, *, utility=None, group="", exp_ids=(), objective_origin=Trust.OPERATOR) -> AdmitResult
    admit_measurement(o, *, objective_id="", utility=None, group="", exp_ids=(), imagined=False, signature="")
    head;  entries();  verify_chain() -> (ok, bad_index);  independent_groups(since_index=0);  rejected
class SelfGoalDiscount:  weight(goal_text, p_success_before, *, register=True) -> float
class NoisyTVGate:  update(key, epistemic) -> float;  gate(key) -> float
noisy_tv_audit(rewards, epistemic, *, tail=0.25, max_tail_ratio=0.2, flat_rel_drop_per_step=0.01) -> dict
```

## reward.selfgoals (goal-free mode machinery)

```python
GoalFreeBudget(seconds, decisions, money=0.0, compute_s=0.0)      # finite or ValueError
LearningGoalCandidate(key, description, epistemic, achievable_progress, p_success=0.5, transfer=1.0, est_cost)
class SelfGoalSelector:  __init__(budget, discount=None, gate=None, min_objectives=5)
    select(candidates, k=1, *, objective_assigned=False) -> List[SelectedGoal]
    record_result(goal, actual_cost, *, epistemic_after, realised_progress)
    record_later_objective(entry: LedgerEntry, predicted_without: float);  usefulness() -> dict
```

## reward.exact (diagnostic reference)

```python
exact_meta_gradient(mdp, psi, theta0, K, alpha, *, features=None, bound=None, detach_reward=False,
                    stop_unroll=False) -> (grad, J)
finite_difference_gradient(mdp, psi, theta0, K, alpha, *, eps=1e-5, features=None, bound=None, coords=None)
sampled_meta_gradient(mdp, psi, theta0, K, alpha, n_inner, n_query, generator, *, reps=1, features=None,
                      bound=None, inner_baseline=True, outer_baseline="loo", inner_score=True,
                      score_baseline=0.0, exact_query=False) -> Tensor [reps, *psi.shape]
enumerated_sampled_gradient(mdp, psi, theta0, K, alpha, n_inner, **kw) -> (grad, J)
lifetime_meta_gradient(mdp, psi, theta0, L, alpha, *, truncate=None, objective="sum") -> (grad, J)
truncation_bias(mdp, psi, theta0, L, alpha, Ks=(1,2,4,8), objective="sum") -> {K: {rel_err, cos, norm_ratio}}
mc_summary(samples, exact) -> {rel_err, max_abs_z, cos, norm_ratio, rel_se, ...}
meta_train(mdps, K, alpha, steps, *, lr=0.1, bound=None, theta_init_scale=0.0, seed=0, estimator="exact", ...)
learning_curve_exact(mdp, r_table, theta0, steps, alpha);  learning_curve_sampled(mdp, theta0, steps, alpha,
                      n_inner, reps, generator, r_table=None)
```
`reward.bandit_lifetime.BanditLifetime` is the small neural test problem with an exact outer objective.

## On the real model (stage 2)

### reward.inner — the policy-improvement loss

```python
InnerConfig(value_coef=0.5, entropy_coef=0.01, half_life_s=86400.0, normalise=True, sequence=True,
            adapt_trunk=True, weight=1.0);  InnerConfig.from_cfg(cfg)      # cfg.section("reward")["inner"]
class AdaptableSubset(model, policy, adapt_trunk=True):      # theta = "policy.<n>" + "trunk.<n>" (model.exp.trunk)
    live() -> Dict[str, nn.Parameter];  snapshot(dtype=None) -> Params;  write(theta);  n_params()
    forward(theta, lat: InnerBatch) -> predict.policy.PolicyOutput
class GroupClip({prefix: max_norm})                          # __call__(names, grads) -> grads, in-graph
@dataclass class InnerBatch: exp_ids, b, e_a, a_mask, e_g, chosen, behaviour_logp, ts, z, b_next, e_a_chosen,
    has_goal, signals, signal_mask, r_boot, gamma, segments, h0, norm_state, mix, boot_components
    .to(dtype=None, device=None)  .with_mix(mix)  .select(idx)  .n
class PolicyImprovementLoss:                                 # learning_api.LossProvider, timescale "frequent"
    name = "policy_improvement"
    def __init__(self, model, policy, reward: IntrinsicReward, *, mixer=None, bootstrap=None, assembler=None,
                 cfg: InnerConfig = None, lifetime_of=None, life_states=None)
    def eligible(self, record) -> bool
    def loss(self, model, batch: DecisionBatch, store) -> Tuple[Optional[Tensor], Dict[str, float]]
    def build_batch(self, store, records, *, batch=None, policy_only=False, gate_update=False) -> InnerBatch
    def functional(self, theta, psi, ib, *, detach_reward=False, return_aux=False, theta_sg=None)
    def inner_loss(self, theta, psi, ib, theta_sg=None) -> Tensor      # reward.meta.InnerLoss (stopgrad_aware)
    def logp(self, theta, ib) -> Tensor;  def probs(self, theta, ib) -> Tensor
    def learned_reward(self, psi, ib) -> Tensor;  def intrinsic(self, psi, ib, *, detach_reward=False)
    def reward_step(self, store, records, *, write=True) -> List[MixedReward]
```

`L = -mean(log pi_theta(a|b,C) * A) + c_v * mean(1/2 (V_theta - G)^2) - c_e * mean(H)`, with
`r = mix * r_bootstrap + (1 - mix) * R_psi`, `G_i = r_i + 0.5 ** (dt_i / half_life_s) * G_{i+1}` inside one lifetime,
`A = standardise(G) - V` (baseline held fixed for the inner derivative). `loss()` evaluates R_psi under `no_grad`
(it cannot move psi); `inner_loss()` is the same function with psi in the graph. Rows are put in lifetime order;
a row with `goal_text == ""` uses the learned null goal. `build_batch` snapshots R_psi's input statistics into the
batch so a later recomputation uses the reward the update used. The model's decision trunk is reached as
`model.exp.trunk` (change request C-6); without it only the policy head is adapted.

### reward.assemble — signal sources

`SignalAssembler(evaluator=None, horizon_fn=None, default_horizon_s=86400.0).assemble(records, store, *, b=None,
b_next=None, e_g=None, has_next=None, device=None) -> AssembledSignals(signals, tensor, mask, learning_progress,
reward_time, keys)`; `record_signals(rec, store) -> (signals, extras, reward_time)`; `SIGNAL_SOURCES`.

| signal | source (first available) |
|---|---|
| any | `decision.diagnostics["reward_signals"][name]` written by the agent at decision / reconciliation time |
| success_before | chosen `CandidateEvaluation.p_success`; else evaluator head on `b` |
| success_after, delta_v | evaluator head on `b_next`; difference |
| action_quality | `predict.evaluator.action_quality` on recorded `CandidateEvaluation.value` and logged probabilities |
| info_gain, epistemic_after | chosen `CandidateEvaluation.extras["info_gain" / "epistemic_after"]` |
| voi, epistemic_before | chosen `CandidateEvaluation.voi` / `.uncertainty` |
| cost_* | `ActionResult.cost` |
| elapsed_s | decision time minus the newest ACTION item in H_t |
| prediction_error_std | prospective predictions of the chosen candidate vs outcomes observed by reward time |
| verified_progress_delta | verifier-trust OUTCOME items in H_{t+1} that are not in H_t |
| context_novelty | stored only |
| status.*, failure.* | `ActionResult` |

Nothing observed after `reward_time` (max of decision end and newest item of H_{t+1}) is read; a signal without a
source is masked. `learning_progress` (bootstrap only) is read from the stored dictionary.

### reward.metalearner

```python
MetaConfig(inner_lr, inner_steps, lr_reward, clip_reward, clip_policy, clip_trunk, min_independent, mode="unroll",
           unroll_events=1, on_policy_tol=1e-3, weight_clip=10.0, min_ess=2.0, min_ess_fraction=0.05,
           min_propensity=1e-6, heldout_fraction=0.25, meta_mix=0.0, utility="return_to_go", max_events=32,
           provenance=None, min_gap_s=0.0, mix=1.0, mixer_step=0.1, mixer_min_independent=8, mixer_tolerance=0.0,
           reward_hidden=256, reward_life=64, reward_bound=1.0, policy_hidden=192)
MetaConfig.from_cfg(cfg, **over)          # cfg.optim (inner_lr, inner_steps, lr_reward, clip_*, meta_min_...) +
                                          # cfg.section("reward")["meta"]
class MetaLearner:
    def __init__(self, model, store, cfg, *, policy=None, reward=None, check_signature=None,
                 check_measurement=None, ledger=None, ledger_path=None, clock=None, assembler=None, bootstrap=None,
                 mixer=None, inner_cfg=None, meta_cfg=None, measurement_utility=None,
                 policy_head="predict_policy", reward_head="reward", evaluator_head="predict_evaluator")
    inner: PolicyImprovementLoss;  subset: AdaptableSubset;  engine: MetaGradient;  optimizer;  ledger;  mixer
    events: List[UpdateEvent];  records: List[MetaUpdateRecord];  stats: Dict[str, float]
    def check_learner_optimizers(self, optimizers) -> None           # raises IntegrityError on overlap with psi
    def new_lifetime(self) -> None
    def observe(self, records) -> List[ExperienceRecord]
    def internal_rewards(self, records, *, write=True) -> Dict[str, Any]          # AgentAPI step 13
    def policy_update(self, records, *, steps=None, inner_lr=None, trigger="policy_improvement")
        -> Optional[contracts.LearningUpdateRecord]                               # the real inner update
    def note_policy_update(self, records, *, inner_lr=None, steps=None) -> Optional[UpdateEvent]
    def admit_objective(self, verified: VerifiedOutcome, *, exp_ids=None, utility=None, group="",
                        objective_origin=Trust.OPERATOR) -> AdmitResult
    def admit_measurements(self, records) -> int
    def ready(self) -> Tuple[bool, str]
    def meta_update(self, now=None) -> contracts.MetaUpdateRecord                 # AgentAPI step 15
    def state_dict(self) -> Dict[str, Any];  def load_state_dict(self, sd) -> None
```

Use by the integrated agent
1. `ml = MetaLearner(model, store, cfg, check_signature=verifier.check_signature, clock=clock,
   ledger_path=<run>/outcome_ledger.jsonl)`; it takes the policy head `predict_policy` (creates a `CandidatePolicy`
   when absent) and registers `IntrinsicReward` as head `reward` in group `reward`.
   `ml.check_learner_optimizers(learner_optimizers)` once.
2. After `record_result`: `ml.internal_rewards([record])` (once per decision, in order).
3. Policy improvement: either `ml.policy_update(records)` (functional SGD; exactly what the unroll reproduces), or
   register `ml.inner` as a `LossProvider` with the continual learner **and** call `ml.note_policy_update(records)`
   before that learner steps (the meta-update then evaluates a shadow SGD update by importance weighting).
4. On a signed outcome: `ml.admit_objective(verified)` (default utility: change of the verified value since the
   previous admitted evaluation of that objective, credited to the decisions since then). Verified measurements
   are admitted automatically, but only with a `check_measurement` (change request C-1).
5. `ml.meta_update()` whenever convenient: it returns a record with `applied=False` and `skipped_reason`
   (`no_policy_update_recorded`, `insufficient_independent_outcomes:n<min`, `no_supported_episode:<why>`) until the
   evidence suffices. A broken ledger chain raises `IntegrityError`.
6. `ml.new_lifetime()` only when a genuinely new lifetime starts. `state_dict()` goes into the checkpoint
   (`CheckpointState.extra`); the ledger persists in its own append-only file.

Semantics: query rows of an update event are the decisions registered after it and before the next event; rows
whose goal-relevant measurements are unresolved wait; every row supervises at most one meta-update. Independent
groups: the objective-outcome group when a signed objective outcome credits the row, else the `WindowBuilder`
group. `MetaUpdateRecord.query_window` is `"<window id>@<ledger head>"`; `diagnostics` holds floats only (ess,
clip_scale, n_episodes, n_heldout_groups, promotion in {-1, 0, 1}, psi_tensors_with_gradient, ...).

### reward.selfgoals.GoalFreeMode

```python
class GoalFreeMode:
    def __init__(self, model, fast: FastBeliefState, budget: GoalFreeBudget, *, selector=None,
                 achieve_fraction=0.25, calibration_rate=0.3, replay_per_key=8, min_progress=0.02, clock=None)
    def active(self, objective_assigned: bool) -> bool
    def candidates(self, b, pairs: Sequence[Tuple[ActionSpec, OutcomeQuery]]) -> List[(candidate, phi, raw)]
    def propose(self, b, pairs, *, k=1, objective_assigned=False) -> List[SelfGoal]
    def resolve(self, goal: SelfGoal, value: Optional[float], cost: Cost, *, censored=None) -> SelfGoal
    def record_later_objective(self, entry: LedgerEntry, predicted_without: float);  usefulness();  report()
    def state_dict();  load_state_dict(d)
@dataclass class SelfGoal: goal_id, key, description, action, query, phi, epistemic_before, predicted_progress,
    required_progress, score, discount, created_at, status, realised_progress, epistemic_after, origin=Trust.SELF
```
The chosen `goal.action` goes through authorization and execution like any other action; `resolve` is called with
what was observed. A goal's outcome is never admitted to the outcome ledger (`objective_origin=Trust.SELF`).

### reward.diagnose

`make_stack(seed, ec, *, mix, mix_step, heldout, meta_mix, check_measurement, **meta_over) -> (model, store, clock,
ml)`, `model_policy(ml, store, generator, theta=None) -> PolicyFn` (exact propensities), `lab_lifetime(...)`,
`LabMeasurementAuth`, `run_learner(kind, seed, ec)`, `labworld_experiment(seeds, ec, kinds, workers)`,
`meta_gradient_check(Ks)`, `psi_isolation_check()`, `integrity_check()`, `meta_cost_50m(Ks)`, `reward_profile(...)`.
