# Workstream B (stage 1) — public API of `cam_live.predict`

Tensor-level and model-independent. Every module is a pure function or `nn.Module` of the tensors of
`cam_live/nn/api.py` (`b [B,D]`, `e_a [B,N,D]`, `a_mask [B,N]`, `e_g [B,D]`, `e_q [B,Q,D]`, `q_mask`,
`kind_ids [B,Q]`, `horizon_s [B,Q]` float64 seconds, `z [B,N,D]`, `phi [B,N,Q,F]`). Nothing here loads a model,
decodes text inside imagination, or returns a reward. Heads are plain modules: register them with
`model.register_head(name, module, GROUP_HEADS)`.

Stage 2 wires these onto `CamLiveModel` (LossProviders, agent steps). Notes for that wiring are marked **S2**.

## dists.py — predictive families

Functional log-likelihoods (broadcasting, dtype of the parameters):
`normal_log_prob(y, loc, scale)`, `student_t_log_prob(y, loc, scale, df)`, `poisson_log_prob(y, rate, exposure=1.0)`,
`neg_binomial_log_prob(y, rate, concentration, exposure=1.0)`, `neg_binomial_log_prob_from_log(y, log_mu, log_r)`,
`bernoulli_log_prob(y, logits)`, `categorical_log_prob(y, logits, n_categories=None)`,
`hazard_log_prob(t, event, hazard_logits, edges)` (event False = right-censored at `t`), `hazard_pmf`,
`hazard_survival`, `hazard_expected_time(hazard_logits, edges)`, `uniform_edges(horizon, n_bins)`.

Masks: `safe_targets(y, mask, fill=0.0)`, `masked_mean(values, mask)` — exactly zero loss and gradient when masked.

Wrappers `Normal, StudentT, Poisson, NegBinomial, Bernoulli (.from_prob), Categorical, DiscreteHazard`, all with
`.log_prob(y, **kw)`, `.mean`, `.variance`, `.std`, `.cdf(y)`, `.quantile(q)`, `.interval(level=0.9)`,
`.sample(n, generator=None)`, `.to_record(index=None, epistemic_std=None) -> contracts.Distribution`.
`from_record(rec) -> Dist`; `record_log_prob(rec, value, status="observed", exposure=None) -> float`.
Family names equal `contracts.Distribution.family`. No scipy at run time (`betainc`, `student_t_cdf/icdf`).

## world.py — world-model heads

```python
WorldConfig(d_model, n_features=0, hidden=192, n_members=5, n_hazard_bins=12, max_categories=16, d_time=16,
            real_family="student_t", prior_scale=1.0, prior_hidden=64, min_scale=1e-3, max_log_rate=20.0, seed=0)
WorldHeads(cfg).forward(z, e_q, kind_ids, horizon_s, phi=None, *, exposure=None, n_categories=None,
                        real_loc=None, real_scale=None) -> WorldOutput
```
`e_q` is `[B,Q,D]` or `[B,N,Q,D]`; `phi` is required iff `n_features > 0`. `exposure [B,Q]` scales count means;
`real_loc / real_scale [B,Q]` are the affine normaliser of each real-valued metric (prediction in metric units,
loss in normalised units). Kind indices are `nn.api.KIND_INDEX` (asserted equal in the tests).

`WorldOutput`: `.log_prob(y [B,Q], event=None) -> [M,B,N,Q]`, `.member_means()`, `.member_variances()`,
`.predictive() -> {"mean","epistemic_std","aleatoric_std","total_std"}` each `[B,N,Q]`,
`.thompson_means(generator) -> [B,N,Q]` (one whole ensemble member per row),
`.to_distribution(b, n, q) -> contracts.Distribution` (carries `epistemic_std`).
Epistemic = spread of the M bootstrapped members with randomised prior functions; aleatoric = mean member variance.

```python
WorldTargets(y [B,Q], observed [B,Q] bool, censored [B,Q] bool).mask(kind_ids)
build_targets(rows: Sequence[Sequence[Tuple[OutcomeQuery, Optional[OutcomeObservation]]]], device=None)
    -> (QueryMeta(kind_ids, horizon_s, exposure, n_categories, q_mask), WorldTargets)
bootstrap_weights(keys: Sequence[str], n_members, seed=0) -> Tensor [M,B]     # deterministic per record key
world_nll(out, targets, *, chosen=None, q_mask=None, member_weights=None, kind_weights=None,
          normaliser: KindNormaliser = None) -> (loss, metrics)
LatentTransition(d_model, hidden=256, n_members=3, d_time=16, seed=0).forward(z [...,D], dt_s [...], b=None)
    -> TransitionOutput(.mean, .epistemic_var, .aleatoric_var, .total_std, .sample(generator), imagined=True)
transition_nll(out, b_next, mask=None, member_weights=None, detach_target=True) -> (loss, metrics)
```
**S2**: `encode_queries` does not return exposure / category counts; take them from `build_targets(...)[0]`.
Use `bootstrap_weights([r.exp_id ...], cfg.n_members)` as `member_weights`, keep one `KindNormaliser` per learner.

## imagination.py, errors.py

```python
rollout(transition, decision_latent, b0, e_a_seq [B,S,D], e_g, dt_s [B,S], *, max_steps=None, n_particles=8,
        generator=None) -> ImaginedRollout            # decision_latent = model.decision_latent
imagine_candidates(transition, z [B,N,D], dt_s [B,N], b, value_fn, *, a_mask=None, n_particles=8, generator=None)
    -> ImaginedValues(value_mean [B,N], value_std, b_next_mean, b_next_std, horizon_steps=1, imagined=True)
multistep_error(transition, decision_latent, b_seq [B,T+1,D], e_a_seq [B,T,D], e_g, dt_s [B,T], max_k) -> [max_k]
HorizonController(max_horizon=8, threshold=0.5, min_count=3, momentum=0.8, min_horizon=1)
    .update(errors) -> int ; .allowed() -> int ; .state() ; .load_state(s)
is_imagined(obj) ; refuse_imagined(obj, what="outcome")  # raises ImaginedDataError
imagined_prediction(query, dist, made_at, horizon_steps, *, decision_id="", action_id="", model_version="")
ChangeDetector(drift=0.5, threshold=8.0, clip=6.0).update(residual) -> bool ; .changed ; .score ; .reset()
classify_error(status, value, mean, aleatoric_std, epistemic_std, change=None, z_noise=2.0) -> ErrorAttribution
```
Labels: `missing_observation | within_noise | model_uncertain | changed_conditions | model_wrong`.

## evaluator.py — goal-conditioned self-evaluation

```python
EvaluatorConfig(d_model, d_ctx=0, hidden=192, n_time_bins=12, n_resources=3, d_time=16, out_std=0.05)
GoalEvaluator(cfg).forward(b [B,D]|[B,T,D], e_g [B,D], horizon_s [B]|[B,T], policy_ctx=None, scale=None)   # V
GoalEvaluator(cfg).forward_candidates(z [B,N,D], e_g, horizon_s [B], policy_ctx=None, scale=None)          # Q
EvalOutput: .success_logit, .p_success, .time_dist(), .p_achieved_in_time(), .expected_time(),
            .magnitude_count() -> NegBinomial, .magnitude_real() -> StudentT, .resources_expected(res_scale=None)
EvalTargets(success, success_mask, magnitude, magnitude_mask, magnitude_is_count, time, time_event, time_mask,
            resources [...,R], resources_mask, weight=None) ; .cat(items) ; .stack(items) ; .index(idx)
evaluator_loss(out, tgt, weights=None, res_scale=None) -> (loss, metrics)
prefix_targets(prefix_ts, *, status, evaluated_at, deadline, signature_checked, value=0.0, achieved_at=None,
               magnitude_is_count=True, resources_final=None, resources_at_prefix=None, n_resources=3)
    -> (horizon_s [T], EvalTargets [T])
prefix_targets_from_outcome(outcome: VerifiedOutcome, prefix_ts, *, deadline, signature_checked, ...)
delta_v(evaluator, b_prev, b_next, e_g, horizon_prev, horizon_next, policy_ctx=None, detach=True)
    -> {"p_prev","p_next","dv_prob","dv_logit"}       # a signal for R_psi, not a reward
action_quality(q [B,N], probs [B,N], chosen [B], a_mask=None) -> {"q_chosen","baseline","quality","regret"}
outcome_decomposition(y, q_chosen, baseline) -> {"quality","luck","surprise"}     # quality + luck = surprise
```
`signature_checked=False` raises `UnverifiedOutcomeError`; a pending objective yields all-False masks.
**S2**: `policy_ctx` is free-form (`d_ctx` features, e.g. exploration rule one-hot and policy entropy);
`scale` is the natural size of the magnitude (e.g. `VerifiedOutcome.target`).

## calibration.py

`brier(p, y)`, `binary_nll(p, y)`, `studentised_ci(values, level=0.95, clusters=None)`,
`logistic_recalibration(p, y, level=0.95)` (slope / intercept with intervals, `identifiable`),
`reliability_bins(p, y, n_bins=10, strategy="uniform"|"quantile")`, `ece(p, y, n_bins=10)`, `ece_ci(...)`,
`binary_report(p, y, n_bins=10, level=0.95, clusters=None)`,
`grouped_binary_report(p, y, families, horizons_s, *, horizon_edges=..., clusters=None)`,
`prospective_guard(made_at, outcome_at, registered_at=None)` (raises `RetrospectivePredictionError`),
`score_predictions(records, *, family_of=None, registered_at=None, horizon_edges=..., strict=True)`.

## policy.py

```python
PolicyConfig(d_model, hidden=192, out_std=0.05)
CandidatePolicy(cfg).forward(z [B,N,D], a_mask [B,N], b [B,D]) -> PolicyOutput
    PolicyOutput: .log_probs [B,N], .probs, .logits, .value [B], .q [B,N], .log_prob(chosen), .entropy()
CandidatePolicy.functional_forward(params: Dict[str, Tensor], z, a_mask, b) -> PolicyOutput   # torch.func
CandidatePolicy.named_param_dict() ; .actor_param_dict()        # the subset log pi depends on
selection_probs(rule, *, a_mask, logits=None, q_mean=None, q_std=None, q_samples=None [S,B,N], info_gain=None,
                arms=None, arm_probs=None, eps_floor=0.02, temperature=1.0, ucb_c=1.0, ig_weight=1.0) -> [B,N] f64
select(rule, *, a_mask, generator=None, **same) -> Selection(index [B], probs [B,N], rule, eps_floor, diagnostics)
    Selection.propensity [B] ; Selection.probabilities(row=0, n=None) -> List[float]   # Decision.probabilities
latent_dependence(policy, z, a_mask, b, *, z_fn=None, perm=None) -> {"kl","tv","argmax_changed"}
```
Rules (`policy.RULES`): `"policy" | "thompson" | "ucb" | "greedy" | "voi" | "randomized"`. The returned vector is
the distribution the index was sampled from; every feasible candidate keeps at least `eps_floor / n_feasible`.
**S2/C**: for the inner update differentiate `actor_param_dict()` and call
`functional_forward({**policy.named_param_dict(), **updated}, z, a_mask, b)`.

## voi.py

```python
LinearGaussianPosterior(mean [F], cov [F,F], noise_var=1.0) ; .prior(n_features, scale=1.0, noise_var=1.0)
    .predict(phi) -> (mean, var) ; .update(phi, y, noise_var=None) ; .sample_weights(n, generator=None)
voi_linear_gaussian(post, phi_probe [P,F], phi_dec [A,F], noise_var=None, costs=None, dec_mask=None)
    -> {"gross" [P], "net" [P], "base", "info_gain_nats" [P], "expected_surprise" [P]}       # exact
expected_max_affine(a, b) -> float ; knowledge_gradient_gaussian(mu, sigma2, noise2) -> [K]
GLMPosterior(mean, cov, family="bernoulli"|"poisson") ; .update(phi, y, exposure=1.0) ; .predict_mean(phi)
voi_glm(post, phi_probe, phi_dec, n_samples=256, generator=None, exposure=1.0, costs=None) -> {"gross","net","stderr","base"}
voi_sampling(candidates, sample_outcomes, update, q_values, belief0, n=64, costs=None) -> {"gross","net","stderr","base"}
VOICosts(cost=0.0, elapsed_s=0.0, opportunity_rate=0.0, discount_rate=0.0, direct_value=0.0)
DownstreamValueEstimator(d_model, n_signals=0, hidden=96).forward(z, signals=None) -> DownstreamOutput(.mean, .dist())
downstream_loss(out, realised, mask=None) ; spearman(x, y)
score_downstream(estimate, realised, reference=None, n_boot=1000, level=0.95, seed=0, margin=0.0)
    -> {..., "reference_spearman", "delta", "delta_lo", "delta_hi", "learned_beats_reference": bool}
```
**S2**: `phi_probe` / `phi_dec` are `model.features(z, e_q)` rows of the fast layer; use `voi_sampling` with the
real `FastBeliefState` copy for non-conjugate cases. The computed VOI is the reference; switch to the learned
estimator only while `score_downstream(...)["learned_beats_reference"]` is True.

## registry.py

```python
PredictionRegistry(path, clock=None).register(rec) ; .resolve(prediction_id, outcome: OutcomeObservation)
    .get(id) ; .records() ; .pending(now=None) ; .scoreable() ; .by_decision(id) ; .by_query(id)
    .registered_at(id) ; .score(family_of=None, **kw) ; .verify_chain()
ExperimentRegistry(path, predictions, clock=None, seed=0)
    .register(*, question, hypothesis, predictions, arms=(), assignment="none", arm_probs=None, intervention="",
              comparison="", horizon_s=0.0, cost=None, evidence_refs=(), experiment_id=None) -> ExperimentRecord
    .start(id) ; .assign(id, unit_id) -> Assignment(arm, arm_index, probability, probabilities)
    .link_decision(id, decision_id) ; .add_outcome(id, outcome, arm="") ; .mature(id, now=None) -> bool
    .analyse(id, arm_a=None, arm_b=None, query_ids=None, level=0.95) -> Welch result dict
    .conclude(id, *, supported, test, alternative_explanations, measurement_uncertainty="", next_action="")
    .get(id) ; .meta(id) ; .records() ; .open() ; .verify_chain()
welch_test(a, b, level=0.95) ; verify_log(path)
```
Errors: `RegistryError`, `DuplicatePredictionError`, `AlreadyResolvedError`, `ResolutionBeforePredictionError`,
`UnknownRecordError`, `TamperedRegistryError`. Files: `RunFiles.PREDICTIONS`, `RunFiles.EXPERIMENTS`.

## candidates.py

```python
parse_actions(text, tools, *, provenance="backbone", max_candidates=64, max_steps=16, max_chars=60000,
              id_prefix="act", parent_decision_id="", experiment_id="") -> ParseResult(actions, rejections, n_items)
validate_schema(value, schema, path="$") -> List[str] ; unsupported_keywords(schema) -> List[str]
values_for(schema, *, vocab=(), name="*", k=5) ; enumerate_arguments(schema, *, vocab=(), max_per_field=5, max_total=64)
sample_arguments(schema, rng: random.Random, *, vocab=(), n=1)
schema_candidates(tools, *, vocab=(), per_tool=8, n_sampled=0, rng=None, max_per_field=5, id_prefix="act",
                  parent_decision_id="", experiment_id="") -> List[ActionSpec]
merge_candidates(*sources, max_n=None) ; content_key(action)
build_proposal_prompt(tools, observation_digest, *, objective="", k=8, max_observation_chars=6000, max_tools=64)
    -> ProposalPrompt(text, instructions, objective_line, tools_line, observation_line, truncated_chars)
PROGRAM_OPERATION = "program"
```
`vocab` is a list of strings or `{property name: [strings], "*": fallback}` (e.g. element refs and URLs from the
current observation). Action ids are deterministic in (content, provenance, `id_prefix`): pass a per-decision
`id_prefix` when the same action may be executed more than once.
**S2**: `model.generate([prompt.text], latent=b)` → `parse_actions` → `merge_candidates(lm, schema, skills)` →
`EnvironmentAdapter.validate_action`.
