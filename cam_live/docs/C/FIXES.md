# Fixes for the independent verification G-pr (workstreams B and C)

Agent `BC` (fixer), 2026-11-27. Report under fix: `docs/verification/G_pr.md`; probes `docs/verification/repro/pr/`.
Every probe was run against the working tree (`.venv/bin/python cam_live/docs/verification/repro/pr/<probe>.py`)
before (commit 7c8b6e7 + other agents' uncommitted edits) and after the fixes. Probe verdict lines are quoted
verbatim (detail text shortened with "...").

## Summary

| id | sev. | fix | regression tests (fail on the audited code) | probe after |
|---|---|---|---|---|
| D1 | major | baseline group = the whole return-to-go chain; prefix-only fallback | `test_reward_fixes.py::test_return_to_go_baseline_keeps_the_outer_policy_gradient_unbiased_exactly`, `::test_meta_learner_baseline_of_a_row_does_not_depend_on_its_own_measured_outcome` | no counter-example left (see D1) |
| D2 | major | utility derived in the ledger from (status, value, target); disagreeing caller utility / foreign `exp_ids` refused and chained | `test_reward_integrity.py::test_utility_is_derived_from_the_signed_outcome_and_a_disagreeing_caller_utility_is_refused`, `test_reward_fixes.py::test_signed_outcome_utility_and_credit_cannot_be_chosen_by_the_caller` | PASS |
| D3 | major | only objectives with a *recorded* operator origin (`TaskSpec.origin`) supervise; unknown -> excluded | `test_reward_integrity.py::test_objective_origin_is_recorded_and_unknown_origin_is_excluded`, `test_reward_fixes.py::test_verifier_signed_outcome_of_a_self_generated_goal_never_supervises_by_default` | PASS |
| D4 | major | selected goals reserve budget until resolved / abandoned | `test_reward_fixes.py::test_goal_free_budget_is_enforced_including_pending_goals`, `test_reward_goalfree.py::test_pending_goal_free_goals_hold_budget_until_resolved_or_abandoned` | PASS (2 checks) |
| D5 | major | cluster-robust (CR1) slope / intercept when clusters are given, HAC in row order otherwise; ECE cluster bootstrap | `test_calibration.py::test_slope_interval_accounts_for_clustering_under_a_clustered_null`, `::test_independent_rows_keep_their_coverage_under_the_robust_intervals` | PASS |
| D6 | minor | goal key folds number words, endings, articles, attainment verbs; optional embedding near-duplicates | `test_reward_integrity.py::test_reworded_self_generated_goals_count_as_repeats`, `::test_semantic_near_duplicates_on_the_goal_embedding_are_discounted`; gpu: `test_reward_semantic_goals.py` | PASS |
| D7 | minor | on-policy only if replaying the live update gives bit-identical parameters | `test_reward_fixes.py::test_cold_start_shadow_update_never_uses_the_on_policy_estimator_by_tolerance`, `::test_on_policy_only_while_psi_is_the_one_the_live_update_used` | info line: `importance_weighted` |
| D8 | minor | head anchor file `<ledger>.head`; truncated / unanchored file fails `verify_chain`, blocks appends and meta-updates | `test_reward_integrity.py::test_ledger_tampering_is_detected` (cases 4, 5), `test_reward_fixes.py::test_truncated_ledger_file_stops_meta_learning` | info line: `(False, 1) n=1` |
| D9 | minor | documented (exact no-op at init, by design); test that a bridge update changes and conditions proposal log-probs | `test_proposer.py::test_bridge_is_an_exact_no_op_at_init_and_conditions_proposals_after_an_update` | unchanged by design (see D9) |
| D10 | minor | `score_predictions` NLL intervals clustered by experiment / decision / prediction | `test_calibration.py::test_score_predictions_clusters_the_nll_intervals` | (code reading) |
| D11 | minor | `build_targets` keeps y and exposure in float64 | `test_world.py::test_build_targets_keeps_values_in_float64` | info line: `torch.float64` |
| D12 | minor | label `bootstrap` until a meta-update was applied **and** promoted | `test_reward_bootstrap.py::test_mixer_starts_as_pure_bootstrap...`, `::test_promotion_before_any_meta_update_does_not_earn_the_learned_label`, `test_reward_fixes.py::test_reward_is_labelled_bootstrap_until_a_meta_update_was_applied_and_promoted` | (code reading) |

Probe totals, working tree, before -> after:
`probe_metalearner_path` 12 / 13 -> 11 / 11 after D1-D12 (two checks became vacuous, D1), then 10 / 11 after the
zero-meta-gradient rule requested later (its "R_psi detached: every *applied* meta-gradient is exactly 0" finds no
applied record any more, see the last section), `probe_integrity` 18 / 21 -> 21 / 21,
`probe_extras` 2 / 4 -> 4 / 4, `probe_predict_numerics` 30 / 31 -> 31 / 31. The remaining probes: see "Other probes".

## D1 — return-to-go baseline (major)

Before (`probe_metalearner_path.py`, last checks):
```
[PASS] re-derived return-to-go utilities match MetaLearner's episode utilities  0.0e+00
[FAIL] the leave-group-out baseline of a row does not depend on that row's own measured outcome  d baseline / d own outcome = 0.284 (row 2 of 8; d U_own = 1)
```
After:
```
[INFO] return-to-go check: no episode with measured rows (nothing to check)
11 checks, 0 failed
```
**What changed.** `_Episode.group` is now the row's *baseline group*: with `utility="return_to_go"` every row of the
same chain (lifetime, held-out partition) — exactly the rows whose return-to-go can contain this row's measured
outcome — shares one group `rtg:life=<l>:held=<h>`; with `"direct"` it is the independent-outcome group as before.
The independent-outcome group moved to `_Episode.indep` (independence count, held-out split, promotion,
`QueryData.groups`: unchanged semantics). The baseline (`MetaLearner.outer_baselines`) is the mean utility of the
outer rows outside the baseline group, else the *prefix* baseline: the running mean verified utility as it was when
the decision was registered (`_Episode.prefix_base`), a function of outcomes observed before the decision only.

**Why the probe line is an INFO now.** The probe looks for a measured row with an earlier row of a *different*
`e.group` and applies the audited formula (mean over other `e.group` rows of the return-to-go). With the fix no such
row exists — all rows that could carry this row's outcome are in its group — so the counter-example search comes up
empty and the two checks are skipped. Whenever the probe's search succeeds the formula depends on the own outcome by
construction, so a PASS line is impossible for *any* valid baseline; the property is pinned by my tests and by an
executed measurement on the probe's own scenario (`make_stack(7)`, `lab_lifetime(seed=4242, 24 decisions, update
every 8)`, authenticated measurements), `docs/evidence/C/fix_D1_baseline.py` -> `fix_D1_baseline.json`:

| | max over measured outer rows of d b_a / d m_a | exact relative bias of E[outer estimator] (4-row lifetime, all 16 action sequences) |
|---|---|---|
| audited rule (leave own independent group out) | 0.678 | 0.966 |
| fixed (`outer_baselines`) | 0.0 | 9.6e-16 |
| no baseline (reference) | — | 1.4e-15 |

The exact test enumerates every action sequence of a lifetime whose row outcomes depend on their own actions and
compares E[sum_t 1/N grad log pi(a_t) (U_t - b_t)] with grad (1/N) sum_t E[m_t]: equal to 1e-12 with the fix.

## D2 — the utility that supervises psi is derived from the signed outcome (major)

Before (`probe_integrity.py`):
```
[FAIL] a validly signed outcome cannot be admitted with a utility the verifier never stated  accepted=True ledger utility=1000000.0 signed value=14.0
[INFO] validly signed outcome credited to arbitrary exp_ids: accepted=True reason=''
```
After:
```
[PASS] a validly signed outcome cannot be admitted with a utility the verifier never stated  accepted=False ledger utility=None signed value=14.0
[INFO] validly signed outcome credited to arbitrary exp_ids: accepted=False reason='exp_ids_not_decisions_of_objective'
```
`integrity.objective_utility(v)` (mapping `goal_progress_v1`): 1 if `status == ACHIEVED`, else
0.5 * clip(value / max(target, 1), 0, 1) — the LabWorld utility C's experiments always used
(`diagnostics.collect.verified_return`). The ledger stores the change of this level since the previous admitted
evaluation of the objective (levels are rebuilt from the chain on load); measurements store their measured value
(`measured_value_v1`). `utility=` is the caller's expectation: if it disagrees the outcome is refused
(`utility_mismatch`), logged in `rejected` / `rejected_details` and — because the signature was valid — chained as a
`"refused"` entry (utility 0, nothing credited, reason and claimed value in `note`). `MetaLearner.admit_objective`
also refuses `exp_ids` that are not decisions registered for that objective. API change: the default utility of
`OutcomeLedger.admit_verified` was `v.value` and of `MetaLearner.admit_objective` the change of the raw value;
both are now the derived goal-progress utility (bounded in [-1, 1] per evaluation).

## D3 — self-generated goals never supervise; origin is recorded, unknown is excluded (major)

Before:
```
[FAIL] a verifier-signed outcome of a self-generated goal does not supervise the outer objective by default  accepted=True reason='' (...)
```
After:
```
[PASS] a verifier-signed outcome of a self-generated goal does not supervise the outer objective by default  accepted=False reason='objective_origin_unknown' (...)
[PASS] with objective_origin=SELF it is rejected
```
`OutcomeLedger.register_objective(task)` records `TaskSpec.origin` (or an explicit origin for an id: operator
registration). `admit_verified` admits only objectives recorded as OPERATOR / VERIFIER; never registered ->
`objective_origin_unknown`, recorded SELF -> `self_authored_objective` (SELF is sticky against later
registrations); the `objective_origin` argument can only exclude. `MetaLearner.register_objective(task)`,
`admit_objective(..., task=task)`; origins are part of `MetaLearner.state_dict()`. The diagnostic agent
(`reward.diagnose.model_policy`) registers the task it acts on, like the integrated agent must (change request
C-10); `lab_lifetime` and `run_learner` register their tasks. Signing the origin by the verifier is proposed as C-12.

## D4 — goal-free budget including pending goals (major)

Before (`probe_extras.py`):
```
[FAIL] one select(k=5) call never returns more goals than the decision budget (2)  returned 5 goals (...)
[FAIL] repeated selection while earlier goals are pending stays within the decision budget (2)  6 goals selected; (...)
```
After:
```
[PASS] one select(k=5) call never returns more goals than the decision budget (2)  returned 2 goals (...)
[PASS] repeated selection while earlier goals are pending stays within the decision budget (2)  2 goals selected; (...)
```
`GoalFreeBudget.reserve/release`, `reserved`, `n_reserved`, `committed`; `SelfGoalSelector.select` reserves each
goal (one decision + estimated cost) and checks them one after another; `record_result` releases and charges the
actual cost; `SelfGoalSelector.abandon` / `GoalFreeMode.abandon` release without charging. Reservations are
checkpointed; `GoalFreeMode.abandon_pending()` releases those that no live goal holds after a restart. The agent must abandon proposed goals it does not execute (change request C-11); until then
goal-free mode fails closed (stops selecting) instead of overspending.

## D5 — calibration slope / intercept intervals under clustering (major)

Before (`probe_predict_numerics.py`):
```
[FAIL] clustered null (true slope 0): the reported 95% slope interval excludes 0 in <= 10% of runs  excluded 0 in 184/300 = 0.61 of runs (...)
```
After:
```
[PASS] clustered null (true slope 0): the reported 95% slope interval excludes 0 in <= 10% of runs  excluded 0 in 21/300 = 0.07 of runs (...)
```
`logistic_recalibration(..., *, clusters=None, cov="auto")`: with clusters the CR1 cluster-robust sandwich with G - 1
degrees of freedom; without them (the probe's call) a HAC sandwich in row order (Bartlett kernel, Andrews AR(1)
bandwidth); `cov="iid"` is the audited model-based interval. `binary_report` passes `clusters` to the slope /
intercept and to the ECE interval (cluster bootstrap). On the verifier's simulation (40 lifetimes x 16 forecasts,
300 runs, nominal 5 %): exclusion of the true slope 0 is 3.3 % with the lifetime clusters, 7.0 % with HAC, 61 % with
the audited interval; on 200 independent data sets (n = 1500, true slope 0.7) both robust intervals miss in <= 9 %.
B's reported evaluator calibration is re-computed below (B.json corrected).

## D6 — reworded self-goals count as repeats (minor)

Before: `[FAIL] a paraphrased repeat of the same self-goal is discounted`; weights `[1.0, 0.5, 1.0, 1.0]`.
After: `[PASS] a paraphrased repeat of the same self-goal is discounted`; weights `[1.0, 0.5, 0.25, 0.125]`.
`SelfGoalDiscount.key` folds digits *and number words*, plural / tense endings, articles and generic attainment verbs
("reach / get / obtain / achieve ... N members" are one goal; "lose 10 members" is not). `SelfGoalDiscount(embed=,
similar_above=)` adds semantic near-duplicates on a goal-text embedding (cosine >= threshold, also for paraphrases
that share no content word). Measured on the real backbone (gpu test `test_reward_semantic_goals.py`,
`docs/evidence/C/semantic_goal_repeats.json`, Qwen3-4B `embed_texts`, 9 same-goal and 46 different-goal pairs):
rewordings are closer on average (cosine 0.54 vs 0.41) but **no threshold separates them** (least similar rewording
0.35 < most similar different goal 0.57). So the semantic path is available but not reliable with this embedding;
it is not switched on anywhere by default, and the effective protection is the lexical fold (which catches the
verifier's examples and not, e.g., "Grow the community to ten people" vs "Recruit 10 members"). `GoalFreeMode` keeps
the lexical key: its goal texts are generated from the structured key (operation, metric), not written freely.

## D7 — cold start never switches to the on-policy estimator by tolerance (minor)

Before: `[INFO] cold start (mix 1): estimator of the shadow meta-update (...): mixed_on_policy_and_importance`.
After: `[INFO] cold start (mix 1): estimator of the shadow meta-update (...): importance_weighted`.
`MetaLearner.logged_policy_is_current(events, theta_K)`: on-policy only if every event of the unroll was applied by
`policy_update` and the same functional SGD replayed without a graph from the event's snapshot (current psi, input
statistics and `meta_mix`) gives parameters bit-identical to those the live update left (`UpdateEvent.theta1_hash`);
the tolerance check stays as an additional consistency check. (The differentiable unroll is not compared directly:
building the graph changes float rounding.) Rows collected under an older psi, the cold-start shadow update and
external updates use importance weighting.

## D8 — truncated outcome ledger fails verification (minor)

Before: `[INFO] truncated ledger (last 2 entries deleted) loads with a valid chain: (True, -1) n=1 (...)`.
After: `[INFO] truncated ledger (last 2 entries deleted) loads with a valid chain: (False, 1) n=1 (...)`.
With a `path`, `<path>.head` (`{"n", "head"}`, atomic replace + fsync) is written after every append. A file that
does not reach its anchor, has none, or lacks `expected_head` loads but fails `verify_chain()`, refuses appends
(`IntegrityError`) and stops `meta_update`. Someone who rewrites both files is caught only by a head kept elsewhere
(`expected_head`, the checkpoint head that `MetaLearner.load_state_dict` checks). A ledger file written before
anchors existed fails as `missing_anchor` until `confirm_head(<checkpoint head>)` adopts it
(`MetaLearner.load_state_dict` does so) — for such legacy files the protection is what it was before the fix.

## D9 — proposer conditioning on the belief (minor)

Unchanged by design: the bridge's cross-attention outputs are zero-initialised so the bridge is an *exact* no-op at
initialisation (architecture requirement). The probe line `at initialisation the backbone's distribution depends on
the belief latent` therefore stays FAIL; it is documented in `BackboneProposer` and docs/B/API.md, and the new test
shows that one real gradient step on the bridge parameters changes the proposals' log-probabilities and makes them
depend on the belief (tiny backbone, CPU).

## D10, D11, D12 (minor)

* D10: `score_predictions` NLL intervals use the same clusters as the binary report (experiment, else decision,
  else prediction id).
* D11: before `[INFO] build_targets stores y as: torch.float32`, after `[INFO] build_targets stores y as:
  torch.float64` (exposure too); the likelihoods cast to the head dtype.
* D12: `RewardMixer.label` is `bootstrap` until `meta_updates_applied > 0` and a later promotion ("lower")
  decision; then `learned` / `mixed`. The mix value is still in every `reward_version` string.

## Untrue claims corrected (G-pr section 4)

1. C.json MR-06 "Utilities come only from the hash-chained ledger: signed VerifiedOutcome records" — was true only
   for the presence of a signature; now the number and the credited decisions are derived / checked and the origin
   is the recorded one (D2, D3). Rewritten in C.json.
2. C.json AE-04 "budgeted" — refuted for pending goals; now true with reservations (D4). Rewritten in C.json.
3. B.json "calibration slope 254.16 (CI 122.43..385.90)" — the interval treated 640 forecasts of 40 lifetimes as
   independent. Re-computed with lifetime clusters (see B.json `measured.evaluator_calibration` and
   `docs/evidence/B/diagnostics.json` experiment 3).

## C's LabWorld experiment re-run (evidence regenerated)

The stored runs of `docs/evidence/C/diagnostics.json` no longer reproduced on the committed tree even *without* my
changes: on a `git archive` of the commit before my first fix (2c89aa0) the reduced reproduction probe gave
`bootstrap seed 11: 55/80 equal`, `bootstrap seed 14: 61/80`, `learned seed 11: 56/80`, `learned seed 14: 59/80`
(other workstreams changed `nn/`, `config.py`, `diagnostics/` since 9ba1e45). With my fixes the bootstrap arm is
bit-identical to that pre-fix tree (55/80 and 61/80 again, the same lifetimes); the learned arm differs (D7). So
the whole experiment (4 arms x 10 seeds x 320 lifetimes) was re-run on a `git archive` snapshot of commit 0536a74
(1472 s wall, 40 workers). That run still *applied* 167 of 1391 meta-updates with an exactly zero gradient (every
lifetime of the window had the same verified utility), which the zero-meta-gradient rule of the last section
forbids; the learned and mixed arms were therefore re-run once more on a snapshot of 8f5339b
(`--rerun-kinds learned,mixed`, 920 s; bootstrap and verified_pg never call the meta-update, their 0536a74 runs are
kept). Final evidence, last quarter of 320 lifetimes, 10 seeds:

| | before (stored, superseded) | run on 0536a74 (superseded) | final (8f5339b for learned / mixed) |
|---|---|---|---|
| bootstrap only | 0.472 [0.405, 0.540] | 0.486 [0.397, 0.576] | 0.486 [0.397, 0.576] |
| learned from scratch (mix 0) | 0.263 [0.079, 0.446] | 0.195 [0.072, 0.318] | 0.136 [0.074, 0.198] |
| cold-start path (mixed) | 0.445 [0.293, 0.597] | 0.425 [0.329, 0.521] | 0.449 [0.351, 0.547] |
| verified-outcome-only PG | 0.471 [0.416, 0.527] | 0.423 [0.358, 0.487] | 0.422 [0.358, 0.487] |
| learned - bootstrap | -0.210 [-0.379, -0.041] hurt | -0.291 [-0.438, -0.144] hurt | -0.350 [-0.430, -0.270] hurt |
| mixed - bootstrap | -0.028 [-0.209, +0.154] none | -0.061 [-0.213, +0.090] none | -0.038 [-0.186, +0.111] none |
| learned seeds collapsed on a non-intervention | 5 | 5 | 3 ([11, 14, 17]) |
| learned seeds with the gauge paid and chosen > 25 % | 3 | 0 | 0 |
| applied meta-updates with a zero gradient | (not recorded) | 167 of 1391 | 0 of 1261 |

The qualitative result stands (the learned reward from scratch hurts; the cold start is indistinguishable from
bootstrap-only); the noisy-gauge failure of the learned arm did not recur. Estimators of the applied updates in the
final run: learned 658 mixed on-policy / importance, 30 on-policy; mixed 511 importance-weighted, 60 mixed, 2
on-policy (on-policy episodes are those whose live update replays bit-identically with the learned reward alone).
`probe_labworld_reduced.py` against the snapshot reproduces the stored runs (see "Other probes").
`docs/status/C.json` (summary, measured, scientific_failures) carries these numbers.

## B's evaluator calibration re-computed (D5, untrue claim 3)

Experiment 3 of `python -m cam_live.predict.diagnose` (40 held-out lifetimes x 16 prospectively registered success
forecasts) was re-run on the same `git archive` snapshot (0536a74) with every interval clustered by lifetime and the
raw (p, y, lifetime) rows stored in `docs/evidence/B/diagnostics.json` (2427 s):

| trained evaluator | first run (superseded) | re-run, clustered |
|---|---|---|
| calibration slope | 254.16, "CI 122.43..385.90" (rows as independent) | 303.98, CR1 CI -27.95..635.90 (39 df) |
| same rows, independent-rows interval | — | 146.15..461.80 (shown for comparison only) |
| Brier (CI over lifetimes) | 0.240 (0.215..0.266) | 0.240 (0.215..0.265) |
| ECE | 0.019 | 0.021 (cluster bootstrap 0.0..0.037) |

The untrained evaluator reproduces exactly (slope -56.60; clustered CI -120.71..7.51 instead of -101.57..-11.63).
Conclusion corrected in B.json: the trained evaluator is a near-constant, calibrated-in-the-large forecast whose
slope is not determined by 40 lifetimes; no slope claim can be made.

## Added during the fix: a zero meta-gradient was recorded as applied (M-int staging run 2)

Report (docs/status/M-int.json, docs/evidence/validation/staging_20261127T053639Z.json, code before my fixes,
c9f3400): one meta-update was recorded `applied` with `meta_grad_norm` 0.0, `outer_objective` 0.0, importance
weighting, 3 query rows, 4 independent groups, psi unchanged.

**Why it was exactly zero.** The agent admits a signed evaluation of its objective after almost every decision,
each crediting the one decision since the previous evaluation, each its own independent group; its utility is the
progress since the previous evaluation. In that run's ledger (25 entries) 17 of the 23 objective evaluations carry
utility 0.0 (no new verified progress in between); non-zero progress appears at 6 evaluations only. The applied
update's query rows all had U = 0 (with positive importance weights, an estimated objective of exactly 0 means every
U_i = 0), so the leave-group-out baselines were 0 as well and every advantage U - b was 0: the policy-gradient
surrogate is identically zero. (It is not caused by the D1 baseline, which was not in that code; with D1 the
baseline of such rows is the prefix mean, also 0 at that point.) `MetaGradient.apply` then made an Adam step on a
zero gradient: psi unchanged on Adam's first step — but on later steps such a step moves psi through the moments.

**Fix.** `MetaLearner.meta_update` refuses before touching the optimizer: `non_finite_meta_gradient`;
`zero_advantage: all n query utilities equal their baselines (k distinct utility value(s), utility in [a, b])`;
`zero_meta_gradient:<norm><=<floor>` (`MetaConfig.min_meta_grad_norm`, default 0: strictly positive required);
after the step, `psi_unchanged_by_the_step` (the optimizer state is restored). `MetaGradient.apply` itself refuses
zero / non-finite gradients. Skipped records carry `meta_grad_norm`, `estimator`, `outer_objective` and diagnostics
(`max_abs_advantage`, `n_distinct_utilities`, `utility_min`, `utility_max`, `n_query_rows`). Zero-advantage rows are
not consumed, so they can be contrasted with later outcomes. Regression tests:
`test_reward_fixes.py::test_equal_query_utilities_give_zero_advantage_and_are_skipped_with_the_reason`,
`::test_zero_meta_gradient_never_moves_psi_through_optimizer_momentum` (fails on the audited code: the update was
applied and Adam moved psi), `::test_meta_gradient_apply_refuses_zero_and_non_finite_gradients`.

**Query window needed for a non-zero meta-gradient on the real model.** (1) At least `min_independent` independent
verified outcome groups among non-held-out query rows, with logged behaviour propensities and enough effective
sample size (staging run 2 also had an `effective_sample_size_too_small` skip); (2) contrast: at least one supported
query row whose verified utility differs from its baseline — with per-evaluation objective credit, the window must
contain both evaluations with verified progress and evaluations without (or different amounts); with return-to-go
in a single chain, a utility different from the prefix mean (0 before the first applied update); (3) an inner update
that depends on psi — R_psi non-constant across the support batch (scale-normalised rewards that are constant give
a zero inner gradient) and a policy step that changes the query rows' log-probabilities (non-zero inner lr, not
clipped to zero). In the agent, (2) is the binding condition: most evaluations report no progress, so a window of
3-5 rows is often all-zero; rows now wait until progress arrives. Note also C-10: until the agent registers its task,
its objective outcomes are refused (unknown origin) and no meta-update can happen at all.

**Effect on the G-pr probe.** `probe_metalearner_path.py` collects `meta_grad_norm` of *applied* records for its
check "real path with R_psi detached: every applied meta-gradient is exactly 0"; with this rule a detached R_psi
gives no applied record (it is skipped as `zero_meta_gradient` with `meta_grad_norm` 0.0), so that line reports FAIL
with an empty list. The property itself (detached reward -> exactly zero meta-gradient, psi untouched) is pinned by
`test_zero_meta_gradient_never_moves_psi_through_optimizer_momentum`.

## Other probes (no defect; run after the fixes)

| probe | source tree | result |
|---|---|---|
| `probe_decisions.py` | working tree | 28 / 28 (before: 28 / 28) |
| `probe_metagrad_small.py` | working tree | 40 / 40 |
| `probe_metagrad_real.py` | working tree | 22 / 22 |
| `probe_reproduce_numbers.py` | snapshot 0536a74 (`G_PR_SRC`), against the regenerated C evidence | 14 / 14 (FD rel. error 7.4e-6 at K = 1, 2, 4; the 8f5339b re-run stored the same meta-gradient numbers) |
| `probe_labworld_reduced.py --lifetimes 80` | snapshot 8f5339b, against the final stored runs | 6 / 6: 80 / 80 lifetimes bit-identical in all four arms; seed 11 collapse reproduced |
| `probe_real_model.py` (Qwen3-4B, under `gpu_lock`) | snapshot 0536a74 | 12 / 13 — the one FAIL is D9's "at initialisation the backbone's distribution depends on the belief latent", unchanged by design (bridge exact no-op at init) |

As in G's report, bit-reproduction of stored LabWorld runs holds on the committed snapshot; other agents'
uncommitted edits in the working tree can change model initialisation and therefore every number.

## Tests (final run, after every change)

`.venv/bin/python -m pytest cam_live/tests/predict cam_live/tests/reward -m "(gpu or slow or not (gpu or slow)) and
not network and not browser and not privileged" --rootdir=cam_live` (default + slow + gpu; the root conftest runs
gpu / slow tests only when the -m expression names them) -> `docs/evidence/test_results/BC_fix.xml`: 293 passed, 1
error, 2539 s wall on a shared, loaded machine. predict/: 157 passed (incl. slow learning evidence and the Qwen3-4B
gpu tests). reward/: 136 passed; the error is `test_reward_semantic_goals.py` (gpu), which timed out at setup after
900 s waiting for the shared GPU lock held by another tenant; re-run alone right after: passed
(`docs/evidence/test_results/BC_fix_gpu_retry.xml`). The gpu tests regenerated `docs/evidence/B/
proposals_real_backbone.json`, `docs/evidence/B/real_config.json`, `docs/evidence/C/meta_cost_50m.json` (end-to-end
50m meta-updates applied with non-zero meta-gradients 0.078 / 0.061 at K = 2 / 4) and
`docs/evidence/C/semantic_goal_repeats.json` (same conclusion: least similar rewording 0.35 < most similar
different goal 0.57).
