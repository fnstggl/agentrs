# Workstream D — fixes of the G-cl verification (docs/verification/G_cl.md)

Fixer: `D-fix`. Code audited by G: `credit/` at 5a133f7 (= cc9c1c2). Fix commits: `2eafa26` (M2, M3), `c5122be` (M1,
M4, m1-m6), then the evidence / status commit. Every "before" line below was produced by G's unmodified probe on the
*old* `credit/` (`git archive 607fcf0 cam_live/src/cam_live/credit` over the current tree, scratch copy), every
"after" line by the same probe on the fixed tree; commands are run from `/workspace/camlive`. Raw probe outputs: `docs/evidence/D/probes/` (`*_before.txt` on the old
`credit/`, `*_after*.txt` / `real_*2.txt` on the fixed tree).

| defect | probe (G's, unmodified) | before | after | regression test (fails on the old code) |
|---|---|---|---|---|
| M1 | `probe_providers.py` check C | `[FAIL] C: ... 0.71 of the 24 experiment records are in the learner's training split` | `[PASS] C: ... 0.00 of the 24 experiment records are in the learner's training split` | `test_gcl_fixes.py::test_experiment_lifetimes_are_held_out_and_the_comparison_excludes_whatever_was_trained_on`, `::test_zero_standard_error_and_in_sample_evidence_cannot_promote_an_estimator`, `test_simple_select.py::test_rudder_and_cocoa_are_exercised_and_compared_with_randomised_differences` (held-out lifetimes), `::test_experiment_effects_need_real_randomisation` |
| M2 | `probe_store_links.py --only immut` | `[FAIL] update() refuses to rewrite ... (accepted=True; stored probabilities now [0.9, 0.05, 0.05], ts 5.0, context ['it1', 'late'], provenance staging)`; `[FAIL] sample(provenance='live') never returns ... (ids ['e1'] have provenance ['staging'])` | `[PASS] update() refuses ...`; `[PASS] sample(provenance='live') ... (ids ['e1'] have provenance ['live'])` | `test_replay.py::test_update_refuses_to_rewrite_history_and_keeps_the_index_consistent`, `::test_a_result_queued_for_approval_may_be_replaced_once_by_the_executed_one`, `::test_reads_refuse_a_record_whose_index_disagrees_with_its_content` |
| M3 | `probe_store_links.py --only links` | `[FAIL] the weights of one outcome never sum to more than 1 (negative attribution amounts) (links {'e4': (1.5, 3.0)}: total weight 1.5; credited 3.0 for an event of value 2.0)` | `[PASS] ... (links {'e4': (0.5, 1.0), 'e5': (0.5, 1.0)}: total weight 1.0; credited 2.0 for an event of value 2.0)` | `test_links.py::test_malformed_attribution_amounts_are_ambiguous_and_never_create_credit` (4 cases), `::test_valid_amounts_are_shares_and_weights_never_exceed_one` — both fail behaviourally on the old code (weight 1.5, NaN weight crashes the insert, `'many'` raises) |
| M4 | `probe_real.py --part write` (`slow.guard`), real model | G: `3.437 -> 3.579 (+4.1 %)` kept (and D's own `3.341 -> 3.485`) | see section M4 below | `test_gcl_fixes.py::test_a_slow_step_that_degrades_held_out_loss_by_a_small_relative_amount_is_undone_completely`, `::test_cumulative_drift_of_accepted_slow_steps_is_bounded_until_a_promotion` |
| m1 | `probe_providers.py` part D (prints only) | `COCOA status {'..._0': 'supported', '..._1': 'supported'}` | `COCOA status {'..._0': 'unsupported', '..._1': 'unsupported'}` | `test_gcl_fixes.py::test_long_trajectories_get_credit_everywhere_cocoa_pairs_follow_eq4_and_open_ended_sets_are_refused` |
| m2 | `probe_providers.py` check B | `[FAIL] B: every outcome observed after decision t enters t's COCOA sum (Eq. 4) (9 ... pairs left out ...)` | `[PASS] B: ... (0 ... pairs left out: [])` | same test as m1 |
| m3 | `probe_providers.py` check A | `[FAIL] A: with max_len=3 every decision ... gets RUDDER credit (present [False, False, False, True, True, True])`; `[FAIL] A: first window decision's rudder_rtg ... (rudder_rtg 0.127257, ... G at its time 0.157939 (10800 s later))` | `[PASS] A: ... (present [True, True, True, True, True, True])`; `[PASS] A: ... (rudder_rtg 0.127257, G at t0 0.127257, G at its time 0.127257 (0 s later))` | same test as m1 |
| m4 | `probe_learner.py --only learn` | `[FAIL] revert also restores the adapter optimizer state (Adam moments / step) (20 adapter parameters carry Adam state from the rejected step)` | `[PASS] ... (0 adapter parameters carry Adam state from the rejected step)` | `test_gcl_fixes.py::test_a_slow_step_that_degrades_held_out_loss_...` (weights, Adam moments, lr, `sched_step`, `counters["slow"]`, `model_version`) |
| m5 | re-run of D's commands | numbers in D.json not reproducible, `D.json.commit` not their code | see section m5 | — (process fix: `docs/evidence/D/regenerate.py`) |
| m6 | `probe_learner.py --only promote` (prints only) | name filter only | provenance filter (see below) | `test_gcl_fixes.py::test_promotion_ignores_internal_reward_metrics_by_provenance_whatever_their_name` |
| m7 | `probe_real.py --part write / read` | max \|db\| 0.053 after reload | see section m7 | — |

The new tests in `test_gcl_fixes.py` and the three new `test_replay.py` tests use names the old code does not have,
so on the old code they fail at import (`run_door_training`, `ImmutableHistoryError`, ...); the behavioural "before"
for those defects is the probe line in the table.

## What changed

**M1 (RC-08) estimator selection on held-out lifetimes.**
`ContinualLearner.is_validation` holds out every record of a lifetime that contains a decision of a registered
randomised experiment (`LearnerSettings.holdout_experiments=True`; lifetimes that become experiment lifetimes later
are re-evaluated). RUDDER, COCOA and the learner record the objectives they trained on (`trained_objectives`, only
losses computed with gradients count; persisted in checkpoints). `CreditAssigner.compare_with_experiment` leaves out
every unit whose lifetime anything trained on — from the measured difference *and* from the estimates — and marks the
evidence `held_out=True` with `n_excluded_trained`. The comparison is paired: estimate and measurement are computed on
the same units and the z-score uses the standard error of their difference (`select.paired_comparison`: a sum of
independent per-unit terms, variance estimated within each arm). `EstimatorSelector` scores only held-out evidence,
refuses comparisons with a zero or undefined standard error (`n_refused`; such an estimator cannot pass) and records
the split in `SelectionRecord.holdout` / `excluded_not_held_out`. `exp_credit` now trains the estimators on separate,
unregistered lifetimes of the same mechanisms (`labkit.run_door_training`) and validates on the registered
experiments. The synthetic selection test now trains on one set of lifetimes (seed 0) and validates on another
(seed 1).

**M2 (CL-01, RC-07) append-only history.** `update()` compares the stored record with the new one (both through the
current contract) and raises `ImmutableHistoryError` for any change of the decision (incl. probabilities, time, chosen
index and the decision-time diagnostics keys), context refs, goal, family, provenance, seq, lifetime, an executed
result, a stored final outcome or a query definition (`replay.history_changes`). Outcomes, queries, annotations and
derived data (credit, eligibility, labels, intrinsic reward, window) may change; a `needs_approval` placeholder result
may be replaced once. Every read decodes through `_row`, which checks ten index columns against the body
(`StoreIntegrityError`); `verify_index()` lists inconsistencies.

**M3 (RC-01) validated attribution amounts.** `links.attribution_amounts` accepts only finite numbers >= 0; a map with
a negative / non-finite / non-numeric amount is treated as ambiguous (equal shares of the named decisions, with
uncertainty, `detail["malformed_amounts"]`). `_make` refuses negative or non-finite weights and renormalises weights
above a total of 1; a non-finite event value creates no link.

**M4 / m4 (CL-03, CL-08) slow-update guard.** Held-out loss of every provider is measured on its fixed validation
sample in 8 chunks before and after the step; the step is kept only if the mean paired change is at most
min(0.01 absolute, 5 % relative, max(2 x its standard error over chunks, float floor)) and the summed change of
accepted steps since the reference (last promoted checkpoint, `set_slow_reference`) is at most 0.02. A rejected step
restores the adapter weights, their AdamW state (moments, step counts; parameters without state lose it again), the
learning rates and `sched_step`; `counters["slow"]` / `model_version` do not advance. Gradients are made contiguous
before clipping (an expanded gradient made `clip_grad_norm_` raise).

**m1 (RC-05)** the candidate set is open-ended when a candidate's `ActionSpec.provenance` names only open-ended
generators (backbone, schema_sampler, mutation, ...) or the decision says so; it is COCOA-supported only with tracked
proposals (`tracked_proposals` / `proposal_probs`), default not tracked. **m2 (RC-05)** COCOA pairs a decision with
every later-observed outcome it did not declare, including those declared by earlier decisions; the elapsed-steps input
is the number of decisions between t and the observation. **m3 (RC-02, RC-03)** credit assignment encodes the whole
trajectory (every decision gets RUDDER and COCOA credit; G and Gamma referenced at the first decision); a training
window of the newest `max_len` decisions regresses the return-to-go at its first decision, referenced there.
**m6 (CL-05)** promotion uses each metric's provenance (`(value, provenance)`, `{"value", "provenance"}` or
`provenance=`); internal-reward and unknown provenances never count; `promote_learner` declares the provenance of the
learner's held-out losses per provider; the name rule survives only for undeclared metrics.

## M4 on the real configuration (Qwen3-4B + 50m core, CUDA, shared GPU lock)

G's `probe_real.py` stops with `AttributeError: 'SemanticBackbone' object has no attribute '_fingerprint'` after its
update checks: workstream A's e00d28f renamed that private attribute (change request D-13). D ran a copy with only that
diagnostic line adapted (`docs/evidence/D/probe_real_adapted.py`; `diff` against G's file = line 126 only; run it as G's). After (fixed code):

```
[PASS] real slow update: LoRA and bridge tensors both changed  (lora 144/144, bridge 26/28)
[PASS] real slow update: experiential / heads / reward unchanged
[PASS] frozen backbone hash unchanged by both updates
guard: validation_before 3.31438, validation_after 3.31330, paired_change -0.00108, paired_se 0.00090,
       tolerance 0.00179, accepted True
```
and in the regenerated `real_update.json` (commit c9ec615): held-out 3.27348 -> 3.27352 as a mean over (provider,
chunk) cells, paired change over chunks -0.00008, se 0.00197, tolerance 0.00395, accepted. In both real runs the slow
step did not degrade held-out loss beyond noise, so it was kept; neither run exercises a rejection on the real model
(the rejection path, including the optimizer state, is exercised at debug scale by the regression tests and by G's
`probe_learner.py --only learn`). Note: `paired_change` is the mean over chunks of the per-chunk mean difference;
`validation_before` / `_after` are means over (provider, chunk) cells; with providers of unequal sample size they can
differ in sign by less than one standard error (as here: +0.00004 vs -0.00008, se 0.002).

## m7 — real-model reload

With A's batch-invariant embeddings and content-checked adapter caches (e00d28f), the adapted probe reports:
```
[PASS] writer: predictions after learning do not depend on inference caches  ({'b': 0.0, 'z': 0.0, 'nll': 0.0, 'probs': 0.0})
[PASS] real model: fresh-process weights-only reload has identical group hashes
[PASS] real model: fresh-process reload reproduces predictions bit-exactly (CUDA)  ({'b': 0.0, 'z': 0.0, 'nll': 0.0, 'probs': 0.0})
```
(G: max |db| 0.053 before). The cause G identified (which batch a cached bf16 embedding was computed in) is removed by
A's change; D's export / `load_exported_model` needed no change. This is 4 records of one run, not a proof for every
input.

## m5 — evidence regenerated from committed code

`cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/D/regenerate.py` exports a commit with `git archive`
(no uncommitted edit of any workstream can enter), runs the default suite with `-s` (tagged blocks -> `measured.json`),
`bench_store.py`, `python -m cam_live.credit.diagnose` (-> `diagnostics.json`, written fresh) and, with `--only real`,
`diagnose --real` (-> `real_update.json`), and rebuilds `D.json["measured"]` from those files and the JUnit results
(`--only status`). Every file records `_code` (commit, source, threads, hash seed). `measured.json` and
`diagnostics.json` come from e6f51c7, `real_update.json` from c9ec615; the two commits differ only in
`runtime/cli.py` and `runtime/loop_service.py`, which no credit experiment imports.

Reproducibility check (the verifier found retention depending on what ran earlier in the process): the debug-scale
numbers are now bit-identical across (a) the full `diagnose` run in the exported tree (one process, hash seed 0),
(b) the slow test suite in the worktree (each experiment its own test) and (c) `diagnose --only retention` alone with
hash seed 1: e.g. NLL increase on A naive 1.7675209045410156, with retention 0.35748597979545593; held-out NLL
2.980973720550537 -> 2.2905991077423096; RUDDER mean z^2 6.6077, COCOA 13.8415. (The earlier dependence did not
reproduce on the current tree, which includes A's batch-invariant embeddings; D did not isolate its cause.)

## Re-run of the estimator selection (M1) — reported as measured

Estimators trained on 288 unregistered LabWorld lifetimes (6 mechanisms x 48), validated on the 288 lifetimes of the 6
registered randomised experiments, none of which any estimator or the learner trained on (`credit.holdout` in
diagnostics.json: 0 experiment records in the training split, 0 experiment lifetimes trained on):

| estimator | RMSE vs measured | mean z^2 (paired se) | sign agreement | passes |
|---|---|---|---|---|
| RUDDER (contribution of the randomised decision) | 3.40 | 6.61 | 1.0 | no |
| COCOA (per-unit counterfactual contrast) | 4.23 | 13.84 | 1.0 | no |
| simple | — (no linked outcome for the randomised decision) | — | — | — |

**No estimator is promoted; the training default stays `simple`.** The earlier "rudder promoted" (mean z^2 1.90) was an
in-sample result and is withdrawn. Reason visible in the evidence: RUDDER's estimate is nearly the same for all six
experiments (5.7 - 7.4) because the randomised decision is always the first one with the same initial context, while
the measured effects differ by mechanism (-0.55 ... 11.85). In the synthetic selection test (held-out lifetimes)
RUDDER passes (mean z^2 1.00) and COCOA fails (28.3).

## D.json corrections (G-cl section 4)

* RC-08 "rudder: validated on 6 experiments / promoted rudder" — replaced by the held-out re-run above (nothing
  promoted).
* CL-01 / RC-07 history integrity — now true for records too (M2), stated with the rules and the throughput cost.
* RC-01 "weights of one outcome never sum to more than 1" — now enforced in `_make` and by amount validation (M3).
* `measured.*` — the whole block is rebuilt by `regenerate.py --only status` from evidence files that record their
  commit; the hand-maintained `retention_*`, `debug_heldout_nll_*`, `credit_rmse_*`, `credit_mean_sq_z_*` keys are
  gone (their regenerated values are under `retention`, `debug_heldout_nll`, `credit_scores`). `D.json.commit` is the
  code commit of measured.json / diagnostics.json (e6f51c7).
* CL-03 "held-out guard" — the guard is now the noise-measured absolute rule with drift bound; real-model numbers are
  the regenerated ones; the two degrading steps of the old rule are named as such.
* Scientific failures with numbers that no committed command produces are marked "exploratory, not regenerated".

## Final runs (fixed tree)

* `pytest cam_live/tests/credit -q --rootdir=cam_live` — 114 passed, 7 skipped (`docs/evidence/test_results/D_fix.xml`)
* `... -m slow` — 5 passed, 1273 s on a heavily loaded machine (`D_fix_slow.xml`)
* `... -m gpu` — 2 passed, 49 s (`D_fix_gpu.xml`; GPU tests take the shared lock through the `gpu_heavy` fixture)
* G's probes on the fixed tree: `probe_store_links.py --kills 12` 37/37 hold; `probe_providers.py` 4/4 hold (part D
  prints `unsupported`); `probe_learner.py` 26/26 hold; `probe_rudder.py` 63/63; `probe_cocoa.py` 47/47;
  `probe_smdp.py` 8/8; `probe_real.py` (adapted copy, CR D-13) 8/8 write + 2/2 read.

## Still open / limitations

* RUDDER and COCOA are not validated by evidence on LabWorld; the agent's credited returns stay on `simple`.
* The paired standard error treats the fitted estimator as fixed (its training variance is not included): the gate is
  conservative, not exact.
* Unknown candidate generator names are treated as enumerating (COCOA-supported) — only the listed open-ended
  generators trigger the refusal.
* The slow guard was not observed rejecting a step on the real model (both real slow steps improved held-out loss
  within noise); rejection is exercised at debug scale.
* `update()` immutability costs about 37 % of durable `add_outcome` throughput.
* Promotion trusts a plain number passed as `validation` as the caller's declaration of independent validation.
* Integration tests of M-int (`tests/integration`) had 4 failures in a run during this work; none involved
  `ImmutableHistoryError` / `StoreIntegrityError` (searched the output); not investigated further (not D's files).
