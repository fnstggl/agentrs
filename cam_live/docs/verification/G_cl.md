# G-cl — independent verification of retrospective credit and continual learning (workstream D)

Verifier: G, lens `cl`. Scope: NA-08, RC-01..RC-08, CL-01..CL-06, CL-08, CU-20. Code under test:
`cam_live/src/cam_live/credit/` on branch `feat/cam-live` (no change under `credit/` since D's evidence commit,
`cc9c1c2` = rewritten `5a133f7`, see `docs/evidence/commit_map.tsv`). No source file and no existing test was
modified. Verdicts: `verdicts_cl.json`. Probes: `repro/cl/`. All commands run from `/workspace/camlive`; every probe
prints `[PASS]` / `[FAIL]` per check and exits 1 when a check is refuted. Probes use D's code only as the object
under test; ground truth (enumeration, exact gradients, coefficients, discounts, parameter snapshots) is computed
independently.

## 1. What was executed

| What | Command | Result |
|---|---|---|
| D's default tests | `.venv/bin/python -m pytest cam_live/tests/credit -q --rootdir=cam_live` | 100 passed, 7 skipped, 121 s (D: 100 / 7, 123 s) |
| D's slow tests | `... -m slow` | 5 passed, 343 s |
| D's GPU tests (Qwen3-4B) | `... -m gpu` | 2 passed, 243 s (incl. lock wait) |
| COCOA reference, refusals | `.venv/bin/python cam_live/docs/verification/repro/cl/probe_cocoa.py --seeds 200 --n 400` | 47 / 47 hold |
| RUDDER decomposer, discounting, returns | `.../probe_rudder.py` | 63 / 63 hold |
| Store and linking | `.../probe_store_links.py --kills 12` | 34 hold, 3 refuted (M2 x2, M3) |
| Providers on the real model class (debug) | `.../probe_providers.py` | 4 refuted (M1, m1, m2, m3) |
| Learner, promotion, kill -9, export (debug) | `.../probe_learner.py` (`CL_THREADS=1`) | 25 hold, 1 refuted (m4) |
| Temporally extended actions | `.../probe_smdp.py` | 8 / 8 hold |
| Real configuration (Qwen3-4B + 50m, CUDA) | `.../probe_real.py --part write|read|within --dir DIR` | update checks hold; reload not bit-exact (m7) |
| D's experiments re-run on the current tree | `.venv/bin/python -m cam_live.credit.diagnose --out OUT --work DIR`, `--only retention`, `.../repro_retention.py --warmup 0` | conclusions reproduce, recorded numbers do not (m5) |

GPU probes held `cam_live.nn.gpu.gpu_lock` for one part each (longest hold about 4 min, peak 9.35 GB).

## 2. Claims that survived the attack

**COCOA reference (RC-04).** Ground truth built from scratch: brute-force trajectory enumeration, the exact policy
gradient as the autograd gradient of the enumerated J(theta), the hindsight coefficient
w(t,s,a,u) = p(A_t=a | S_t=s, U_{t+k}=u for some k>=1, counted with multiplicity) / pi(a|s) - 1 (Meulemans et al.
2023, Def. 3), and Eq. 4 re-implemented per trajectory. On key_to_door (d3, key cost), noisy_key_to_door and two
random trees (3 actions / 2 actions): D's `contribution_coefficients` equal the own ones to 1.1e-15, D's
`cocoa_grad` equals the own estimator per trajectory to 4.4e-16, the exact relative bias of both is <= 1.2e-30.
Monte Carlo with an own sampler (200 seeds x 400 trajectories), z-scores against the *exact* per-component variance
(so E z^2 = 1 + n bias^2 / var exactly): mean z^2 = 0.989 +- 0.030, 0.972 +- 0.030, 0.999 +- 0.019, 1.013 +- 0.025;
pooled means (80 000 trajectories) inside the Bonferroni 1 % band; with Gaussian reward noise (key_to_door d4,
sigma 0.5, not enumerable) pooled max |z| 2.43 over 38 components. The test is able to fail: w + 0.3 on one action
gives |z| = 201. Variance: COCOA's exact variance trace is below REINFORCE's on every MDP (0.74 vs 4.65, 1.04 vs
2.72, 1.77 vs 4.36, 0.57 vs 1.14; noisy d4: 1.96 vs 13.36).

**RUDDER decomposer (RC-02, RC-03).** With random non-zero weights (D's head is zero-initialised, which would make
every check trivial), padding, irregular decision times and beta in {0, 1e-4, 2e-3, 0.05}: sum_t Gamma_t c_t = G
with an independently computed Gamma_t = exp(-beta (t_t - t_first)) — error exactly 0.0 in float64; signal =
(G - F(H_<t)) / Gamma_t and the return-to-go identity hold to 1e-9; unmatured rows are NaN in every output, give
`loss() is None`, have no influence on the loss and receive no gradient; padding carries exactly 0. Prefix
causality: replacing the latents of decisions >= j by 5x noise and shifting their times by up to 1e5 s leaves
`pred[:, :j+1]`, `signal[:, :j]` and the contributions of earlier decisions bit-identical (G held fixed), for
j in {1, 4, 8, 12}. `trajectory_return` counts one event per dedup key, discounts to the first decision (own value
to 1e-12), and stays unmatured while the objective is open or a goal-relevant query is pending.

**Refusals of the CAM extension (RC-05).** `check_support` refuses missing, NaN, > 1, unnormalised, near-zero and
-inf propensities and open-ended-untracked sets; refused rows are NaN, not numbers; an untrained hindsight model
claims w = 0 (5e-8).

**Store (CL-01).** kill -9 twelve times at random moments of a writer that appends records (a third blob-backed)
and attaches outcomes: 2198 acknowledged appends and 1098 acknowledged outcomes, none lost, `integrity_check` ok,
seq strictly increasing. One writer in-process, across processes and through a symlinked directory; read-only
readers see committed rows and cannot write. Sampling (3000 seeded batches of 32 over 1000 live + 200 diagnostic
records): no repeated record; per-stratum draw counts equal the configured quotas exactly; failures (22 % vs 14 %
of the pool) and delayed consequences (27 % vs 11 %) are over-represented, recent records are not (35 %); the
family stratum is balanced over families of size 600/250/100/50 (chi2 = 5.2, 3 dof); single draws of the
informative stratum follow the configured priority (chi2 = 6.2 over 10 bins); pure strata contain only members;
provenances mix only when named.

**Linking (RC-01, RC-07).** Exact links for declared queries; equal window evidence gives 0.5 / 0.5 with standard
deviation 0.5 carried into the simple credit; no evidence -> no link; a decision after the outcome is refused and
its share stays unattributed; duplicates dropped; a verified copy replaces an unverified one; an outcome dated
before its query is refused; `audit_leakage` reports exactly a planted future-dated context item.

**Learner (NA-08, CL-02, CL-03, CL-08).** Debug scale: `poll()` is silent without matured outcomes and counts
independent units; a frequent update changes experiential + heads only (reward tensors without `.grad`), with the
experiential clip binding the optimizer sees a gradient of norm 0.0009999 for clip 1e-3 (pre-clip 0.071); slow
updates run exactly after every `adapter_every` applied frequent updates (F F S F F F S) and change only
`semantic_adapters`; `fast_update` changes the fast belief and no parameter. **Real configuration** (Qwen3-4B +
50m core, own parameter snapshots): one frequent update changed 127 experiential and 56 head tensors, 0 adapter /
0 reward tensors, moved predictions (max |db| 1.32, |dNLL| 0.064, |dprobs| 0.0014); one slow update changed
144 / 144 LoRA and 26 / 28 bridge tensors and nothing else; the frozen backbone hash is unchanged (checked against a
freshly loaded model). Grad / delta numbers agree with `real_update.json`: experiential weight delta 0.2540
(D 0.2529), adapter delta norms lora 0.015132 / bridge 0.008757 (D 0.015134 / 0.008758).

**Promotion / rollback (CL-05).** A checkpoint whose retention metric fell by 0.5 is refused and `promote_learner`
rolls weights *and* optimizer state back to the promoted checkpoint; missing retention or internal-reward-only
validation is refused.

**Restart / export (CL-06, debug).** kill -9 at random moments of 196 checkpoint writes (10 kills): the newest
intact checkpoint is never older than the last acknowledged one, verifies and loads consistently. A weights-only
export reloaded in a fresh process reproduces belief, decision latent, world NLL and action probabilities
bit-exactly (with equal thread counts; 2 vs 1 threads differ by 1e-6, float reduction order). Optimizer state
restores exactly into a fresh learner by parameter name.

**CU-20.** The duration-aware discount composes across levels (to 1e-15); one event reported by the primitive that
produced it and as the answer to the intervention's declared query (same dedup key) is credited once (1.0 on the
intervention), enters the return once, and the five-record hierarchy is one learning unit; `CreditLedger` refuses a
second counted assignment.

**Reproduced measured numbers.** exact_rel_var_reinforce 23.458 (D 23.46); exact_rel_var_cocoa 10.517 (D 10.52);
key-decision variance REINFORCE 8.03 / COCOA 0.0609 with an own sampler (D 8.09 / 0.0593); test counts 100 / 5 / 2;
real-configuration weight deltas (above); credit experiment: measured arm differences, conservation error
1.78e-15, 0 links to later decisions and 0 context leaks identical to D's; selected default 'rudder' again.

## 3. Defects

Severity: critical — a core claim is false in a way that corrupts learning; major — a stated property is refuted or
the evidence for it does not show what is claimed; minor — a narrower gap.

### M1 — major — estimator selection is validated in-sample (RC-08)
`credit/providers.py:779-835` (`compare_with_experiment`) compares each estimator with the measured arm difference
over **all** matured lifetimes of the experiment; `credit/diagnose.py:208-233` trains RUDDER and COCOA on the same
lifetimes first. 71 % of the experiment records are in the learner's training split (probe C; the held-out split
is 15-20 % of objectives). In D's recorded run, and again in my re-run, RUDDER's estimate has standard error 0.0 in
all six experiments: the first decision of every lifetime in an arm has the same latent, so the "estimate" is one
fitted value F(first decision | arm) per arm, regressed by squared error on the very returns whose arm means define
the measured difference. Agreement is therefore largely by construction and the shared noise is invisible to the
z-score. The synthetic selection test (`tests/credit/test_simple_select.py:163-`) also trains and validates on one
data set. Consequence: "promoted rudder to training default" is not independent evidence (re-run: RUDDER mean
z^2 2.94 against a gate of 3.0). Repro: `.venv/bin/python cam_live/docs/verification/repro/cl/probe_providers.py`
(check C) and `cam_live/docs/evidence/D/diagnostics.json` → `credit.experiments[*].estimate_se.rudder`.

### M2 — major — stored history is mutable and `update()` desynchronises the index (CL-01, RC-07)
`credit/replay.py:516-528`: `update()` rewrites the static part (decision, behaviour probabilities, decision time,
context_item_refs, provenance) of an existing record without refusal or audit trail, and only the body columns are
replaced — `provenance`, `ts`, `objective_id`, `decision_id`, `action_id` keep the old values. After relabelling
a live record as `staging`, `sample(provenance='live')` returns it with stored provenance `staging` (provenance
strata broken); its probabilities and context were rewritten after the fact. Only context *items* are immutable.
Repro: `.venv/bin/python cam_live/docs/verification/repro/cl/probe_store_links.py --only immut`.

### M3 — major — negative attribution amounts create credit beyond the outcome (RC-01)
`credit/links.py:213-221`: shares are `amount / sum(amounts)`. With `data["by_action"] = {a: 3, b: -1}` and event
value 2, the link to `a` has weight 1.5 (> 1, contrary to the module's invariant) and the simple credit is 3.0 for
an outcome of 2.0; `b`'s negative share is skipped (`_make` drops w <= 0). Net-change events (joins and leaves)
are a natural source of such maps. Repro: `.../probe_store_links.py --only links`.

### M4 — major — the slow-update guard admits degrading steps (CL-03)
`credit/learner.py:681-695`: a slow step is kept if held-out loss rises by at most 5 % relative to the value just
before that step; there is no reference to the last good state, so successive steps may compound. On Qwen3-4B both
observed slow updates raised held-out loss and were kept: 3.341 -> 3.485 (+4.3 %, D) and 3.437 -> 3.579 (+4.1 %,
this probe). Repro: `.venv/bin/python cam_live/docs/verification/repro/cl/probe_real.py --part write --dir DIR`
(`slow.guard`).

### m1 — minor — open-ended candidate sets are "supported" unless a flag nobody sets is present (RC-05)
`credit/providers.py:252-253` reads `decision.diagnostics["open_ended"]` (default False) and
`["tracked_proposals"]` (default True); no producer in `src/` sets either, and `ActionSpec.provenance` ("backbone")
is ignored. Decisions whose candidates were generated by the backbone get status `supported` (probe D). Fail-open.

### m2 — minor — the CAM COCOA sum omits later rewards declared by earlier decisions (RC-05)
`credit/providers.py:470-480` (`_pairs`) pairs decision t only with outcomes declared by *strictly later*
decisions. An outcome declared by an earlier decision but observed after t — which Eq. 4 includes as R_{t+k} — never
enters t's training pairs or credit (9 of 12 such (decision, event) pairs in one 6-decision lifetime). If decision t
influences that outcome, the credit is biased; the deviation from the reference estimator is not stated.

### m3 — minor — long trajectories: lost credit and a mis-referenced discount (RC-02, RC-03)
`credit/providers.py:224` keeps the last `max_len` decisions; `rudder.py:132-140` measures Gamma from the window's
first decision while `providers.py:142` references G at the trajectory's first decision. Decisions before the
window get no RUDDER credit (disclosed), and the window's credit is expressed at the wrong time: with max_len 3,
beta 2e-5, the first window decision's `rudder_rtg` is 0.1273 = G at t0 instead of 0.1579 (G at its own time,
10800 s later). Repro: `.../probe_providers.py` (check A).

### m4 — minor — a reverted slow step is not fully reverted (CL-03, CL-08)
`credit/learner.py:670, 689-697`: the guard restores the adapter weights but not the AdamW state (moments and
step of 20 adapter tensors keep the rejected gradient), `sched_step` and `counters["slow"]` advance and
`model_version` changes. Repro: `.../probe_learner.py --only learn`.

### m5 — minor — recorded debug-scale evidence cannot be regenerated from the committed code (CL-02, CL-04, RC-08)
`docs/evidence/D/diagnostics.json` was committed together with a change of `LearnerSettings.warmup_steps` 0 -> 8
(`85905d6`). Re-running D's own commands on the current tree (no change under `credit/` since):
retention `--only retention`: naive +2.772 +- 0.655, with retention -0.023 +- 0.077 (two identical runs; D:
+2.085 +- 0.580 / +0.150 +- 0.153); with warm-up 0: +2.537 / +0.154; learning: held-out mean NLL 2.863 -> 2.300
(D 2.954 -> 2.284; even the pre-training value differs); credit: RUDDER RMSE 1.99, mean z^2 2.94 (D 1.52 / 1.90),
COCOA 3.47 / 4.22 (D 3.06 / 3.84). The conclusions hold (forgetting strongly reduced, mean but not median NLL
improves, 'rudder' selected), the numbers in D.json do not reproduce and `D.json.commit` is not the code they came
from.

### m6 — minor — the internal-reward filter of promotion is name-based (CL-05)
`credit/checkpoint.py:739-741` ignores only names containing "intrinsic" / "internal_reward" or equal to "reward" /
"r_psi"; a validation metric `heldout/neg_reward_return` is accepted. Retention metrics are still required, so a
reward metric alone cannot promote.

### m7 — minor (unexplained) — real-model weights-only reload is not prediction-exact (CL-06)
Fresh process, CUDA, `load_exported_model`: every group hash (experiential, heads, adapters, reward, frozen backbone)
equals the writer's, the load report is clean, yet predictions on the same 4 records differ from the writer's
post-learning predictions by max |db| 0.053, |dz| 0.045, |dNLL| 0.0014, |dprobs| 6.7e-5. Inside the fresh
process the predictions are exact under cache reset and reversed batch order (0.0) and differ by 1.4e-6 for
one-record batches, so bf16 batch composition does not explain it. See section 4 for the in-writer cache check.
D lists GPU / real-backbone restart as not verified; this is the first measurement.

### Observations (not counted as defects)
* RUDDER and COCOA providers default to `verified_only=False`: unverified measurements enter G (spec 9.2 asks for
  "independently grounded" outcomes; D does not claim otherwise).
* The redistributed credit is not consumed by any training update yet (spec 9.2 "use redistributed credit during
  eligible training updates"; disclosed as integration work).
* `CreditLedger` is not wired into `CreditAssigner` (disclosed); dedup keys and top-level aggregation give the
  credit-once property on the paths probed.
* The retention arm combines family-balanced replay with the regulariser against a baseline that replays only the
  new family (disclosed).

## 4. Untrue or unsupported claims in docs/status/D.json

* RC-08 "Default credit estimator is selected by evidence" / "rudder: validated on 6 experiments" — the validation is
  in-sample (M1).
* CL-01 limitations imply history integrity ("stored context items cannot be rewritten"); records themselves can be
  rewritten and the index then contradicts the body (M2).
* RC-01 "the weights of one outcome never sum to more than 1" (links.py docstring) — false for maps with negative
  amounts (M3).
* `measured.retention_nll_increase_on_A_*`, `debug_heldout_nll_*`, `credit_rmse_*`, `credit_mean_sq_z_*` — not
  reproducible from the cited commit (m5).
* CL-03 "held-out guard" presented as protection; on the real model it kept both observed degrading steps (M4).

Everything else checked in D.json held, including the limitations D states itself.
