# DX — fixes of the independent verifier G2-learn in nn/fast.py and credit/

Input: `docs/verification/G2_learn.md` (defects G2L-1 … G2L-8, probes in `docs/verification/repro/g2learn/`).
Scope of DX: G2L-1 … G2L-6 and CL-10 (claimed here). G2L-7 was already closed by INT3A's KF-E01; G2L-8 is an
agent-side regression (not in DX's paths; see "Side effects" below for what DX changed there).
Every fix below is pinned by a regression test of DX that fails on the code before the fix (run on a
`git archive` export of `6719595` — `nn/` and `credit/` there equal those G2-learn verified — with the new test files copied in) and by the verifier's own probe,
re-run unchanged on the fixed tree. All commands from `/workspace/camlive`.

## Results per defect

| Defect | Fix (file) | Verifier probe after the fix | DX regression tests (fail on old code) |
|---|---|---|---|
| G2L-1 count mode "converged" at an absurd mode | `nn/fast.py::_mode_poisson`: the convex G-step (log form) is taken from the left of the root too (both Newton steps land right of the root, the smaller is kept), and `converged` requires the root condition at the returned point (relative residual <= 1e-8 + float64 rounding of k (eta0 - x), or log-form residual <= 1e-8), not only the relative step rule | `probe_fast_stress.py`: **46/46** (before 42/46); count grid 2625/2625 equal to bisection, both stateful counterexamples reach 48.3543 with `converged=True` in 2 iterations; time-to-event 700/700, binary 425/425 unchanged | `tests/nn/test_dx_fast_mode.py` 8 tests (8 fail on old code: modes 3.3e11 / 1e15, `converged=True` at a non-root) |
| G2L-2 `update()` accepts outcomes dated before their query / decision | `credit/replay.py`: `append` / `update` raise `OutcomeTimingError` (ValueError) for a *newly attached* matured outcome observed before its query was issued or before the decision (stored outcomes are not re-judged; observation time 0 = unknown, as in `add_outcome`); `add_outcome` also refuses an outcome before the decision when the query is unknown | `probe_credit_gaps.py --only store`: **9/9** (before 7/9) | `tests/credit/test_dx_fixes.py` 3 tests (fail on old code) |
| G2L-3 overflowing attribution amounts drop the outcome | `credit/links.py`: when the sum of valid amounts overflows, shares are taken relative to the largest amount (finite sums are computed exactly as before, bit-identical); with no stated outcome value an infinite sum is refused explicitly (`reason "the outcome value is not finite"`, `refused: non_finite_outcome_value`) | `probe_credit_gaps.py --only links`: **6/6** (before 5/6) | `test_dx_fixes.py::test_amounts_that_overflow_when_summed_are_shares_not_a_dropped_outcome` (fails on old code); the explicit-refusal test pins behaviour that was already right |
| G2L-4 slow-step guard fails open without held-out cells | `credit/learner.py::_slow_update`: no provider with held-out cells -> the step is **deferred** (not taken; `counters["slow_deferred"]`, note `deferred`); a loss without held-out records of its own is removed from the adapter gradient (`excluded_without_held_out`), and if no validated loss remains the step is deferred; a held-out loss that cannot be computed after the step rejects and reverts it; a missing verdict is never "accepted" | `probe_credit_gaps.py --only guard`: **3/3** (before 1/3); `probe_agent_credit.py` (integrated agent): "every applied slow step carries a held-out guard verdict" **PASS** (0 of 11 without a verdict; before 2 of 15 / 8 of 61) | `tests/credit/test_dx_guard.py` 3 tests (fail on old code: adapters change) |
| G2L-5 guard judges the mean over providers | `credit/learner.py::_guard_verdict`: every provider with held-out cells gets its own paired change, standard error, tolerance min(abs_tol, rel_tol |loss|, max(z SE, float floor)) and drift since the reference; the step is kept only if every provider passes; the verdict lists `by_provider`, `judged`, `unjudged`, the binding provider and the old cell mean for information | `probe_credit_gaps.py --only guard`: masking check **PASS** (step refused: `degrades` +0.0006 over its tolerance 4.8e-5 although the mean improves) | `test_dx_guard.py::test_one_providers_degradation_is_not_hidden_by_anothers_gain` (fails on old code) |
| (found with G2L-4/5) held-out flags lost; late eligibility never picked up | `credit/learner.py::refresh_eligibility` asks `provider.eligible(record)` without the two flags the learner itself wrote; held-out losses are evaluated on in-memory copies whose `eligible[name]` is the provider's own eligibility. `predict.providers._Base._permitted` reads `eligible[name]` as an exclusion, so before this a held-out record lost `val:<name>` at the next refresh and a record judged ineligible once (e.g. evaluator records before the objective closed) never became trainable | `probe_agent_credit.py`: `learner.validate()` at the end has 8 providers (before: rudder, cocoa, bridge_grounding only) | `test_dx_guard.py::test_held_out_flags_survive_a_refresh_and_late_eligibility_is_picked_up` (fails on old code: 11 val flags -> 0) |
| G2L-6 estimator selection never runs in the agent (RC-08) | `credit/providers.py::CreditAssigner.attach_experiments / record_experiments / evaluate_experiments` (+ `ExperimentEvaluation`), `credit/select.py::EstimatorSelector.select(context=)` / `log_row`, `credit/learner.py::learn` (records the attached source's experiments in the store before the held-out split is refreshed; evaluates after the update). Comparisons use `compare_with_experiment` unchanged (held out by lifetime, paired SE); per experiment and arm the report counts assigned units, units without a final return, in trained-on lifetimes, not randomised, held out; the selection re-runs when the held-out measurement changes, and every pass where it cannot run is logged with the reason. agent.py: change request **DX-1** (one line) | `probe_agent_credit.py` through a wrapper that applies DX-1: see "RC-08 in the integrated agent" | `tests/credit/test_dx_selection.py` 4 default tests + 1 slow test with the real `CamLiveAgent` (all fail on old code: no API) |
| CL-10 (unclaimed) checkpoints persist every 15.2 item | `credit/checkpoint.py`: `CheckpointState.experiments` (store's experiment records) and `pending_outcomes` (unresolved queries) written in `state.json` atomically with the rest; restore writes back experiment records the store lacks (never replaces one it holds) and reports pending queries missing / resolved since; `capture_learner` first records the attached experiment source; old checkpoints load | — (claimed now; not yet verified independently) | `tests/credit/test_dx_checkpoint.py` 3 tests (fail on old code) |

## Change request INT3A-5 (added by the coordinator): a held-out split that cannot be empty (G2L-8 residual)

* `credit/learner.py::ContinualLearner.is_validation` / `_record_held_out` (+ `credit/replay.py::
  SqliteExperienceStore.position_in_objective`): with `val_split="record"` (the agent's setting) the k-th stored record
  of an objective is held out iff floor(u + (k+1) f) > floor(u + k f), u = seeded hash of the objective, f =
  `val_fraction`: exactly every 1/f-th record of each objective (fraction f up to one record), so an objective with
  >= ceil(1/f) = 7 records always has held-out records. Deterministic given the seed and the append-only history
  (a stored record's position never changes; decisions are cached), one side per record. The objective split
  (`val_split="objective"`, D's lab and evidence) is unchanged.
* `credit/checkpoint.py::promote_learner` / `CheckpointManager.promote`: the decision carries `held_out`
  (`evidence`, held-out records per provider, providers with a held-out loss, record count; also in
  `promotions.jsonl`), and a refusal for lack of it adds the reason `no held-out evidence: no provider has a held-out
  loss (...)` next to the existing "no independent validation metric supplied" (fail closed, as CL-03's guard).
* Tests `tests/credit/test_dx_split.py` (4; 3 fail on the code before: 9 of 60 objectives of 7-18 records had no
  held-out record, `PromotionDecision` had no `held_out`; the fourth pins determinism / one-sidedness, which held).
* G2-learn's `probe_promotion_evidence.py --reps 5` (own work directory in DX's scratch), same tree except the
  change: before **3 of 5** first promotions granted (runs 1 and 4 refused: `validate()` empty after the 14-decision
  first lifetime), after **5 of 5** (every first lifetime had held-out records for 6-8 providers; the probe check "granted on held-out evidence in every run" PASS).

## RC-08 in the integrated agent (with DX-1 applied at construction)

A wrapper in DX's scratch directory (not in the repository:
`/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DX/scripts/probe_agent_credit_with_cr.py`)
runs G2-learn's `probe_agent_credit.py` unchanged after patching `CamLiveAgent.__init__` to also call
`self.assigner.attach_experiments(self.experiments)` (= DX-1) and prints the assigner's last evaluation at close; run on
the working tree with DX's code as committed in `715417e`; 3 lifetimes, tiny CPU configuration. The same wiring is the
`-m slow` test `test_dx_selection.py::test_integrated_agent_with_the_change_request_line_records_and_evaluates_its_experiments`
(1 passed, 39 s).

* probe: **13/13** checks; "randomised experiments: registered in the experience store / experiment ids on the
  agent's decisions" **(8, 8)** (before DX-1: (0, 9)); "estimator selection log rows" **13** (before: 0);
* last evaluation: `ran False`, `n_randomized 8`, `n_evidence 0`, default `simple`; reason "no randomised experiment
  has 2 or more matured held-out units in each arm (held out by lifetime …)"; per experiment: `decisions 1`, e.g.
  arm A "0 matured held-out units (< 2; assigned 1, return not final 0, lifetime trained on 1, not randomised 0)",
  arm B "assigned 0";
* "objectives with randomised decisions / of them trained on by the learner (holdout_experiments=False)": (3, 3).

So with DX-1 the evaluation is connected and honest, and the selection still does not run in the agent as
configured — by construction of the agent's experiments (one unit each) and its `holdout_experiments=False`; see
DX-1 for the options. In D's test agent with held-out experiment lifetimes it runs (`test_dx_selection.py`: 2
experiments x 6 units, both compared on held-out lifetimes, paired n = 6, `n_excluded_trained` 0, one selection row,
no re-selection while the measurement is unchanged).

## Side effects (measured with G2-learn's `probe_agent_credit.py`, unchanged agent, tiny CPU configuration)

Before DX (G2-learn, 7 clean runs): 8 of 61 slow steps without a guard verdict; `learner.validate()` at the end had
3 providers (rudder, cocoa, bridge_grounding).

* On a clean export of `e4186c8` (DX's guard + eligibility commit, agent.py and predict/ as at that commit):
  13/13 checks; slow updates attempted 7 / applied-or-reverted 5 (2 deferred: no held-out evidence yet) / **0 without a guard verdict** / 3 reverted; `validate()` at the end has **8 providers** (world_model, fast_prior, evaluator, world_model_adapters, evaluator_adapters, rudder, cocoa, bridge_grounding), so the eligibility fix alone restores them; promotions `[(False, ['no independent validation metric supplied', 'no retention metric supplied']), (True, [])]` (G2L-8 is not DX's; the first promotion of a short run can still lack held-out evidence).
* On the working tree at `e4186c8`-era code (run started right after that commit; INT3A's uncommitted edits, if any, were in the tree): 13/13; slow updates 11, 0 without a verdict, 8 reverted; `validate()` 8 providers; promotions `[(False, ['no retention metric supplied']), (True, [])]`.
* On the working tree at `715417e` with DX-1 applied (above, also contains INT3A's own provider-side eligibility fix
  `914b10a`): 13/13; slow updates 14, **0 without a guard verdict**, 6 reverted; `validate()` 8 providers; promotions
  `[(False, ['no retention metric supplied']), (True, []), (True, [])]`.

The per-provider rule with the float-floor tolerance of a single validation chunk reverts more slow steps than the
averaged rule did (before: 0 reverted). INT3A fixed the same eligibility feedback loop from the provider side in
`914b10a` (predict/providers.py no longer reads the learner's flags); the two fixes are independent and compatible
(DX's also covers any other provider that reads its own flag, and the evaluation copies used for held-out losses).

## Test runs

* Final run on the committed tree (`c9fd652`), default marks:
  `.venv/bin/python -m pytest cam_live/tests/credit cam_live/tests/nn -q --rootdir=cam_live --junitxml=cam_live/docs/evidence/test_results/DX_fix.xml`
  -> **242 passed, 21 skipped, 0 failed, 208 s** (30 DX test cases, one of them the `-m slow` integrated-agent test,
  skipped by the default marks). G2-learn's run before DX: 114 passed / 7 skipped (credit), 99 / 13 (nn).
* `-m slow`: the DX integrated-agent test (`test_dx_selection.py::test_integrated_agent_with_the_change_request_line_records_and_evaluates_its_experiments`)
  1 passed (50 s) on `c9fd652`. `tests/credit -m slow` on `78139c9` (460 s): D's `test_evidence.py` 5 of 5 passed
  with DX's guard / eligibility / selection changes; the DX slow test failed in that run (it read
  `learner.trigger_state`, which a later `poll()` rewrites) and was fixed in `c9fd652` (the last selection pass is
  kept in `learner.credit_selection` / `status()`).
* Old-code runs (export of `6719595` + DX's test files): `test_dx_fast_mode.py` 8 of 8 fail; `test_dx_fixes.py` 4 of 5
  fail (the fifth pins the explicit refusal that was already right); `test_dx_guard.py` 5 of 5 fail on behaviour
  (adapters changed / step accepted / val flags 11 -> 0); `test_dx_selection.py` 4 of 4 and `test_dx_checkpoint.py`
  3 of 3 fail (API absent: new functionality); `test_dx_split.py` 3 of 4 fail (export of `78139c9`).
* Verifier probes on the final tree: `probe_fast_stress.py` 46/46 (before 42/46), `probe_credit_gaps.py` 18/18
  (before 13/18), `probe_agent_credit.py` 13/13 (clean export of `e4186c8`; before: 12/13 or 13/13 depending on the
  run, the failing check being G2L-4's), G-nn's `repro/nn/probe_fast.py` 43/44 (unchanged: the one failure is the
  NIG calibration check that A's D4 fix invalidated, as G2-learn notes).

## Not done / limitations

* DX-1 (agent.py) is a change request; until INT3A applies it the agent's experiments stay outside the store and no
  evaluation runs in the agent. Even with it, the selection does not run in the agent as configured (one unit per
  experiment; every experiment lifetime trained on, `holdout_experiments=False`): that is reported, not changed.
* Held-out evidence recorded earlier stays in the selector after its lifetimes are trained on later (it was held out
  when measured); estimates of a still-training estimator are re-read at the next change of the measurement only.
* The per-provider guard rejects more steps: with one validation chunk per provider the tolerance is the float floor
  (no noise estimate), so in short lifetimes most slow steps are reverted (agent probe: 8 of 11).
* CL-10: the authorization configuration is recorded as the hash the caller passes (`auth_config_hash`, checked on
  load with `expected_auth_hash`) plus the task's `auth_scope` in `current_task`; the agent passes no hash (DX-2).
  The configuration itself stays with the authorization service (a learner checkpoint cannot restore authorization).
  Pending outcomes and experiments live durably in the experience store; the checkpoint carries a copy for
  consistency checks and repair, not a second source of truth. Power-loss durability is not tested.
* The evidence numbers of D (`docs/evidence/D/*.json`, from `e6f51c7`) were not regenerated (G2-learn: conclusions
  reproduce on the current tree, numbers partly differ); D's `-m slow` tests were not re-run by DX.

# Round 2 (after G2-learn round 2 at f54d8cc): slow learning, G2L-9, DX-1 root cause, retention

G2-learn round 2 confirmed G2L-1 … G2L-5 fixed and measured slow (adapter) learning nearly frozen in the integrated
agent (1 of 13 slow steps kept on the real configuration); 37 of 44 rejections looked like real held-out degradations
of other providers (world model +0.5-4 %, COCOA +1-22 %), 7 were micro vetoes at the float-floor tolerance (G2L-9).

## Finding: the "degradations" were mostly the guard's own measurement

Logging the first-order change of every provider's loss predicted for the applied update next to the realized
held-out change showed realized world-model changes of 1-3 % (median) against predicted ~1e-5, with either sign
(debug agent, 8 lifetimes: 23 of 34 world-model rises above tolerance were predicted to be decreases). Two causes,
both in what the guard compares, not in what the adapters did:
1. **Running statistics moved by the step's own loss evaluation.** The slow step computes the providers' losses
   with gradients; `predict.world.world_nll` then updates the world model's `KindNormaliser` buffers (EMA of each
   kind's NLL, the loss's denominator) and `predict.heads.world_batch` updates the real-valued target scaler
   (`heads.scaler`). The held-out loss "after" the step was measured in other units than "before", and a refused
   step did not restore them. Fix (`ContinualLearner._running_state` / `_restore_running_state`): the slow step
   snapshots every model buffer and every provider's `heads.scaler` before its loss evaluation and restores them
   afterwards — the slow timescale changes the adapters only (running statistics stay the frequent timescale's
   business). Test `test_the_slow_step_does_not_move_running_statistics_so_the_guard_measures_the_adapter_change`
   (paired change 1.025 on the previous code, exactly 0 now).
2. **COCOA's held-out loss was random.** `HindsightModel.loss` draws Poisson bootstrap weights from the global
   generator; the provider used them also when evaluating held-out loss, so "before" and "after" were differently
   weighted losses (and every validation consumed global random numbers). Now bootstrap weights are drawn only for a
   training step (`bootstrap=_training_call()`).

After these two fixes, in the same debug-agent run type the realized per-step held-out change is world model median
5e-6 relative (was 2e-2), COCOA 6e-7 (was 4e-2).

## Changes (credit/ only)

1. **G2L-9 — minimum tolerance.** `slow_guard_min_rel_tol = 1e-4`: a provider's tolerance is never below 1e-4 of its
   own held-out loss (was 1e-7). Basis (measured): micro vetoes were rises of 1e-7 … 5.4e-5 relative (G2-learn's 7
   cases); the smallest substantive degradation in G2-learn's runs was 4.4e-4; providers whose chunks agree have a
   paired SE of 3e-7 … 3e-5 relative (DX runs: evaluator, fast_prior, rudder, bridge, and the world model after the
   measurement fix), while every provider's held-out loss moves by a median of 4e-4 … 0.8 relative between two
   consecutive slow steps (frequent updates and validation refresh in between). 1e-4 lies above the first group and
   below the second. Per-provider drift bound max(0.02, 2 % of the provider's loss) (`slow_guard_max_drift_rel`).
   No held-out evidence still defers the step.
2. **Measurement fixes** above (running statistics, deterministic COCOA validation).
3. **Multi-provider combination** (kept; secondary after the measurement fix): PCGrad projection between conflicting
   adapter gradients plus the `max_share` cap 0.9 (the grounding loss had > 0.9 of the adapter gradient); the applied
   AdamW update projected to be first-order non-increasing for every provider with an adapter gradient and for
   COCOA's hindsight model through `HindsightProvider.adapter_sensitivity_loss` (attached latents, its own training
   records, no training step, no random draw); a trust region driven by the verdicts (a refused step halves the next
   adapter step down to 1/16, an accepted one grows it by 1.5 up to 1, checkpointed).
4. **Retention (G2-integration p03):** `CheckpointManager(keep_optim=1)` keeps optimizer state only in the newest kept
   checkpoint (manifest rewritten atomically; every kept checkpoint verifies and loads); a rollback to a checkpoint
   without optimizer state starts the learner's optimizers fresh (`optimizer_state_reset` in the report).
5. **CL-10:** `credit.checkpoint.auth_config_hash(task)`; agent diff in docs/change_requests/DX.md (DX-2).
6. **DX-1:** root cause and the exact hook in docs/change_requests/DX.md; the experiment copy can no longer fail
   `learn()` or a checkpoint.

## Measurements (G2-learn's `probe_slow_acceptance.py`, unchanged, own work directories in DX's scratch)

Debug integrated agent (`--mode agent`, 8 lifetimes, tiny CPU configuration; runs are not reproducible across
processes because ids carry `os.urandom` bytes, so two seed sets each), slow steps judged / kept:

| Code | seeds 11-18 | seeds 21-28 | pooled kept | binding providers of refusals | net adapter change (L2) |
|---|---|---|---|---|---|
| before (DX round 1, = f54d8cc's credit/) | 5 / 22 | 9 / 31 | 14 / 53 (26 %) | cocoa 20, world_model 14, others 5 | 0.026 / 0.011 |
| + G2L-9 tolerance, PCGrad + update projection | 4 / 27 | — | 4 / 27 (15 %) | cocoa 15, world_model 8 | 0.018 |
| + COCOA protection | 14 / 25 | 7 / 34 | 21 / 59 (36 %) | cocoa 21, world_model 16 | 0.056 / 0.023 |
| + trust region | 6 / 17 | 11 / 25 | 17 / 42 (40 %) | cocoa 15, world_model 10 | 0.020 / 0.017 |
| **+ measurement fix (running statistics, deterministic COCOA validation) = final** | **26 / 30** | **32 / 32** | **58 / 62 (94 %)** | rudder 4 (rises 1.2e-4 … 2.8e-4 relative) | 0.037 / 0.018 |

(G2-learn's run at f54d8cc: 7 / 26.) With the final code the realized per-step held-out change of the world model is
median 5e-6 relative (before 2e-2), COCOA 6e-7 (before 4e-2); every applied step still carries a verdict. The net
adapter change stays modest because refused *promotions* roll the learner back (retention gate: decision agreement on
earlier objectives dropped by more than 0.1 in several promotions of these runs) — that is CL-05's gate, not the
slow-step guard. Statistical note: before vs the variants without the measurement fix is not significant (Fisher
p 0.19-0.32); before vs final is (14 / 53 vs 58 / 62, p = 2e-14).

D's lab (`--mode lab --rounds 20`, deterministic): before 20 / 27 kept (all 7 refusals COCOA, the bootstrap-noise
artifact), final 27 / 27; held-out losses at the end world 2.7752 → 2.7738, bridge_grounding 12.335 → 12.306, rudder
0.0644 → 0.0644 (COCOA's held-out value is now the deterministic loss, not comparable with the bootstrap-weighted
one).

Real configuration (Qwen3-4B + 50m core, CUDA, queued GPU lease; `--mode agent --real --seeds 11 12 13`, as G2-learn
ran it): **4 of 8 slow steps kept** (G2-learn at f54d8cc: 0 / 6 and 1 / 7; DX with COCOA protection but without the
measurement fix: 1 / 5); 3 lifetimes / 38 records, GPU lease held 407 s. In every one of the 8 steps the restore
found both running statistics changed by the step's loss evaluation (`heads.predict_world.normaliser.scale` and
the `RealScaler`), confirming the artifact on the real configuration; the world model's per-step held-out change is
now -4e-5 … +2e-5 relative (G2-learn: +0.5 … +4 %), COCOA's ≤ 1.3e-5, RUDDER's ≤ 5e-6. All 4 refusals are by the
grounding loss itself: its held-out NLL rose by 0.09-0.28 % while the step lowered it on the 4-record slow batch,
i.e. the grounding objective did not generalise in those steps, and the guard refused them correctly. The trust
region took the step to 0.19-0.5 of its size after refusals. Net adapter change 0.0043 (one promotion was refused
for forgetting, decision agreement -0.11 on an earlier objective, which rolled the learner back).

Before / after held-out losses per provider are only comparable within a run (different runs see different
experience); the per-step paired changes above are the within-run comparison.

## DX-1 root cause (summary; details and the exact hooks in docs/change_requests/DX.md)

With the hook applied by patching in a driver that replicates the integration test: 1 failure in 30 runs with the
hook, 1 in 22 without (pooled with the coordinator's runs 6 / 58 vs 1 / 32, Fisher p = 0.41). Every run of both arms
pauses after 5 consecutive verifier errors (the test's `_NoVerifier` raises), so the test always has exactly 5
decisions and fails when none is a successful api action — a property of the test. The hook path raised nothing,
cost 0.06 s (evaluation) + 0.009 s (copy) per run, draws no random numbers, and nothing outside `credit/` reads the
copied records. It is now also wrapped so it can never fail `learn()` or a checkpoint.

## Test runs (round 2)

* `.venv/bin/python -m pytest cam_live/tests/credit cam_live/tests/nn -q --rootdir=cam_live --junitxml=cam_live/docs/evidence/test_results/DX_fix2.xml`
  on `6159def`: **251 passed, 21 skipped, 0 failed, 233 s**.
* New tests: `tests/credit/test_dx_slow_r2.py` (7: micro veto, single chunk, real degradation still refused and no
  evidence still defers, dominant loss, COCOA protection, trust region, running statistics),
  `tests/credit/test_dx_checkpoint.py` (+2: auth hash, optimizer retention). On the code before each change: the
  micro-veto and single-chunk tests fail (accepted False), the dominance test fails (unknown setting / world rises),
  the running-statistics test fails (paired change 1.025 instead of 0), the retention test fails (optimizer state in
  every checkpoint).
* Adjusted: `test_dx_guard.py::test_one_providers_degradation_is_not_hidden_by_anothers_gain` uses lr 1e-3 so its
  degradation (6e-4 relative) is above the new minimum tolerance; `test_gcl_fixes.py::test_cumulative_drift_...` sets
  `slow_guard_max_drift_rel = 0` to test the absolute bound it was written for.

## Still open (round 2)

* RC-08 in the agent needs DX-1 applied; CL-10's authorization hash needs DX-2 applied (both agent.py, coordinator).
* The record-level split still lets RUDDER / COCOA train on whole trajectories that contain held-out records of
  their own (they read trajectories, the split is per record); pre-existing, not changed.
* The first-order update projection is a heuristic; on the real configuration the predicted changes were 2-3 orders
  below the realized ones before the measurement fix, so its contribution is not separately established.

