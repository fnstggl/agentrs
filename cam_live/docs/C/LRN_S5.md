# Learning-mechanism fixes for change request S-5 (workstream LRN, 2026-11-27)

Owner of `predict/` and `reward/` for this round. Diagnosis being answered: `docs/S/DIAGNOSIS.md`,
`docs/change_requests/S.md` S-5, INT2-1 / INT2-6 / INT2-7 (`docs/change_requests/INT2.md`), INT2's measured
diagnosis `docs/evidence/validation/diagnosis/debug_f1_20261127/`. Engineering diagnostics at debug scale (tiny random
backbone, debug core, LabWorld F1), plus real-configuration pathway runs; none of this is a scientific result.

## 1. What was wrong — probes on the code before the fix

`docs/evidence/C/lrn_probes.py` measures each defect with APIs that exist before and after the fix (old code = the
snapshot of commit 6e83bcf, the HEAD when this round started; the same script, `PYTHONPATH` switched).
Results: `docs/evidence/C/lrn_probes_before.json`, `docs/evidence/C/lrn_probes_after.json`.

| probe | before (6e83bcf) | after | property |
|---|---|---|---|
| delta V after a step that changed nothing (b' = b), max over 10 decisions | 0.451 | 0.0002 | no action-dependent offset — HOLDS after |
| verified progress credited for *another* decision's verifier reading in H_{t+1} | 9.0 | 0.0 | only own outcomes — HOLDS after |
| relative change of one inner update when R_psi x 10, mix 0.5 | 0.065 | 2.8e-6 | scale-invariant at every mix — HOLDS after |
| change of the update by R_psi's *shape*, mix 1.0 / 0.5 / 0.0 | 0 / 9.1e-5 / 3.8e-3 | 0 / 1.0e-2 / 1.2e-2 | shape matters below mix 1, never at 1 — HOLDS (both) |
| KL of one live policy update, default configuration | 1.8e-6 (step 0.01) | 0.050 (step 0.49, chosen) | the policy moves — HOLDS after |
| relative predicted value of the choice, debug model after 2 lifetimes (rule `policy` / `thompson_policy`) | 0.23 / — | 0.23 / 0.93 | the rule tracks the predictions — HOLDS after |

## 2. What changed

### 2.1 Selection uses the predicted consequences and their uncertainty (`predict/policy.py`)

New rule **`thompson_policy`** (in `RULES`; INT2 sets it as the agent default, LRN-1):

    p_i = (1/S) sum_s [ i = argmax_j  z_s,j + lambda_s * c_j ]      then the epsilon floor (exact propensities)

* `z_s,j` — the world model's *joint* posterior sample s of candidate j's net goal value (ensemble member x fast-layer
  weight draw, the `ts_<k>` extras the Predictor already computed), standardised by the mean and spread of the
  posterior means over the decision's feasible candidates (units of the prediction drop out; uncertainty is kept).
  This is Thompson sampling: a candidate is chosen with its posterior probability of being the best.
* `c_j` — the trainable policy's preference *beyond the world model's ranking*: centred log pi minus its least-squares
  projection on the standardised predicted values of the same candidates, divided by max(1 nat, its spread). An
  untrained, near-uniform policy, or one that merely repeats the world model, has c ~ 0 and no say; a very peaked one
  is bounded like z. (Without the projection a policy trained toward the world model's choices was collinear with it
  and the calibration drove beta_q negative — measured in an intermediate run.)
* `lambda_s = clip(beta_pi^+ / max(beta_q, 0.05), 0, 10)` with `(beta_q, beta_pi)` drawn per sample from
  **`SelectionCalibration`**: a Bayesian linear regression (exponential forgetting 0.99, prior beta_q ~ N(1, 0.5^2),
  beta_pi ~ N(0, 0.5^2), noise variance plug-in) of the *verified progress* of each executed decision on the chosen
  candidate's (z, c) as recorded before acting. The policy's influence grows only as its preferences prove
  predictive of verified progress beyond the world model's; a policy whose preference is noise or harmful gets
  lambda 0 (never inverted).
* Fed by `MetaLearner.refresh_selection` (called from `internal_rewards` and `meta_update`) with the agent's signed
  per-decision `reward_signals.verified_progress_delta`, once per decision; shared with the `DecisionMaker` through
  the policy head (`selection_calibration(policy)`); persisted in `MetaLearner.state_dict()["selection"]`.
* `Decision.diagnostics["selection"]` records the calibration, lambda, the policy's share of the decision (total
  variation to the same rule without the policy) and the value-only distribution.

### 2.2 The policy head is trained toward better choices (`reward/inner.py`, `reward/metalearner.py`)

    L_inner = - mean_i log pi(a_i) A_i + c_v * value loss - c_e * entropy - c_m * beta_q^+ * mean_i sum_j pi(j|i) z_ij
    r_i     = mix * n(r_bootstrap)_i + (1 - mix) * n(R_psi)_i + c_g * u_i / rms(u)

* **Goal term** `u_i` (labelled, never mixed, not meta-learned): the verified progress attributed to decision i when
  final; through the return-to-go G it credits earlier decisions with the verified progress that followed them.
  While pending: the world model's decision-time prediction for the chosen candidate, calibrated by
  `SelectionCalibration.predict` (labelled `predicted`; never stored or counted as an outcome).
* **All-candidate model term**: the expected standardised predicted value under pi of *every* candidate of each
  decision, weighted by the calibrated beta_q^+. Without it the executed-action policy gradient pushed probability
  onto never-chosen candidates (measured in the first version: policy quality fell to 0.03-0.2 of the best while the
  executed choices were good).
* **Trust-region live step** (INT2-7): `MetaConfig.target_kl = 0.05`; the step is searched (bracketed, step in
  [inner_lr, max_inner_lr = 2]) so the mean KL of the policy before/after on the support batch is ~0.05, recorded as
  the event's `inner_lr` (`LearningUpdateRecord.notes`, `.diagnostics["step_search"]`); the meta-gradient
  differentiates exactly that SGD update (replay is bit-identical, test).
* **Replay of matured decisions**: up to `replay_matured = 16` earlier decisions whose verified progress is final are
  added to each support batch (verified outcomes arrive cycles after the decisions that produced them).

### 2.3 The learned reward reaches the live update; promotion and reversion (`reward/bootstrap.py`, `inner.py`)

* **Normalisation** (the "4e-8" observation): the inner update standardises each reward component within the batch
  before mixing, so the update is exactly invariant to `R_psi -> c R_psi + d` (c > 0) at *every* mix (spec 7.8: R_psi
  cannot buy weight by magnitude; before, at 0 < mix < 1 a x10 output changed the update by 6.5 %). Reward *scale*
  therefore carries no influence by design; what psi controls is R_psi's *shape* (which decisions it rewards more),
  entering with weight `1 - mix`. S measured ~4e-8 because the mix was 1: R_psi had weight 0 (now: left out exactly).
* **Promotion** (spec 7.6): after every applied meta-update the held-out groups' paired importance-weighted verified
  utilities of the learned-reward update vs the bootstrap update (same snapshot, same support) are buffered; with 4
  independent groups the mixer decides by a one-sided 95 % Student-t bound: **lower** the mix by 0.25 when the
  learned reward is non-inferior (lower bound >= -0.05 x RMS of the held-out utilities), **raise** it (revert) when
  harm is shown (upper bound < 0), **hold** otherwise. Before: 8 groups, step 0.1, and any inconclusive evidence
  raised the mix — it never left 1. Every decision is in `meta.mixer.decisions`, `MetaUpdateRecord.diagnostics`
  (`promotion*`, `mix_after_promotion`) and `bootstrap_mix`; E_mix0 (mix 0, step 0) and D (meta off) unchanged.

### 2.4 Bootstrap signals no longer pay non-progress (`reward/assemble.py`, `bootstrap.py`)

* delta V = V(b') - V(b) from the same evaluator head at both ends (before: V(b') - Q(b, a) of the chosen candidate,
  an action-dependent offset that paid the actions the evaluator rated low).
* `verified_progress_delta` from H_{t+1} counts only this decision's own goal-relevant questions (before: any
  verifier reading arriving during the step — a wait was paid for earlier interventions' adopters); the agent's
  signed attribution annotation (INT2-1 workaround) still takes precedence.
* The verified component is no longer flattened by the bound (bootstrap bound 4; INT2-6) and the goal term carries
  verified progress linearly.
* `action_quality` (an R_psi input) is computed on the world model's predicted goal values, not on policy logits.
* Not changed (LRN-3): learning progress and epistemic reduction have no input in runtime runs (the agent writes
  neither); uncertainty-aware exploration is carried by Thompson selection.

## 3. Measured on the diagnostic (debug scale)

`docs/evidence/B/choice_diagnostic.py` (one agent built exactly like evaluation condition E / D / E_mix0 of
`eval.conditions`, LabWorld F1 through the real runtime with the ground-truth tap and close-out, 8 lifetimes x 16
decisions per seed, world seeds 10000 + 100 x seed + i, so arms see the same worlds), aggregated by
`summarise_choice.py` into `docs/evidence/B/choice_summary.json` (raw reports: `choice_runs_20261127.tar.xz`).
before = condition E on the snapshot of 6e83bcf (rule `policy` with INT2's interim value prior, S-5 workaround);
after = the LRN code of f8f3b5d with `selection_rule = thompson_policy`. Intervals: bootstrap 95 % over lifetimes.

| E, 3 seeds x 8 lifetimes | before (6e83bcf) | after (f8f3b5d) |
|---|---|---|
| expected choice quality (sum_i p_i truth_i / max truth over C_t) | 0.311 [0.276, 0.344] | **0.375 [0.321, 0.428]** |
| realised choice quality (truth chosen / max) | 0.299 [0.256, 0.344] | 0.384 [0.317, 0.449] |
| same set, uniform choice / greedy on the agent's own predicted goal value | 0.194 / 0.336 | 0.216 / 0.392 |
| expected adopters per decision | 1.85 | **2.70** |
| verified progress (fraction of target) | 0.58 | 0.67 |
| policy head alone (its own softmax), same measure | 0.196 [0.174, 0.218] (= uniform) | **0.308 [0.263, 0.356]** |
| Spearman(policy probabilities, truth) over C_t | 0.27 | **0.69** |
| choice quality by how well the predictions ranked that decision's candidates (rho < 0.5 / 0.5-0.8 / >= 0.8) | 0.18 / 0.27 / 0.36 | 0.23 / 0.33 / 0.48 |
| correlation over lifetimes: greedy-on-prediction quality vs choice quality | 0.76 | 0.70 |
| zero-value decisions (probe, gauge, wait) | 0.086 | 0.077 |
| share of positive *stored* internal reward paid to zero-value decisions | 0.16 | 0.045 |
| share of positive reward *of the inner update* paid to zero-value decisions; mean reward zero-value / valued | 0.0007; 0.001 / 0.44 | 0.0; -0.88 / +0.74 |
| bootstrap mix at the end of each run; promotion decisions | 1, 1, 1; raise x3 (old rule: inconclusive -> raise) | 0.75, 1, 1; lower x1, hold x5 |

Reading:
* **Choice now tracks the agent's own predictions** (S-5 finding 2): 0.96 of greedy-on-prediction quality, and the
  better the predictions ranked a decision's candidates, the better the choice (0.23 -> 0.48 across the three bins,
  steeper than before). S's v2 snapshot (00cb5a4) had the behaviour at uniform (1.42 adopters per decision vs uniform
  1.43; greedy-on-prediction 5.05 on the same contexts); INT2's interim value prior (in "before") already recovered
  part of it.
* **The policy head is trained toward better choices**: from uniform (0.196 vs 0.194) to 0.308, rank correlation with
  the truth 0.27 -> 0.69 (goal term on verified / calibrated-predicted progress, all-candidate model term, trust-
  region step). Its influence on selection stays small: lambda is learned from verified outcomes and the policy's
  preference counts only beyond the world model's ranking.
* **Not shown: improvement over lifetimes.** At this scale neither the predictions nor the choices improve from
  lifetime to lifetime in any arm (slopes of greedy-on-prediction quality and of choice quality ~0); within a
  lifetime the greedy-on-prediction quality by thirds is 0.23 / 0.38 / 0.39 before and 0.41 / 0.41 / 0.33 after. "Choice improves as predictions
  improve" is shown across decisions and lifetimes of differing prediction quality, not as a learning curve.
* **Other arms** (2 seeds each, after): D (meta-learning off, mix 1) 0.332 [0.265, 0.403], E_mix0 (learned reward
  only, mix 0) 0.358 [0.284, 0.435]; intervals overlap E's — no claim about the learned reward's benefit. In E_mix0
  the learned reward pays zero-value decisions *more* than valued ones (0.70 vs 0.54; share 0.17): R_psi has not
  learned what deserves reinforcement at this scale, and the promotion rule correspondingly held E's mix at 1 in
  two of three runs.
* An intermediate version (no policy projection, no model term; `intermediate_v1_E`) had 0.418 [0.346, 0.487]
  expected choice quality but a policy head *worse* than uniform (0.173): the executed-action policy gradient pushed
  mass onto never-chosen candidates. The final version trades nothing measurable in choice quality (intervals
  overlap) for a policy head that learned.

## 4. Real configuration: the reward meta-learning pathway (task item 4)

`docs/evidence/C/real_pathway.py`: Qwen3-4B (frozen, bf16) + LoRA + bridge + 50m core on the GPU under the shared lock,
one `CamLiveAgent` (rule `thompson_policy`, schema-walker candidates as in the evaluation conditions) through the real
runtime on LabWorld F1 lifetimes (12 decisions, close-out). `meta_update` and `policy_update` are wrapped to record,
for every applied outer step: psi delta, R_psi on the support decisions before / after, the inner update at the live
mix and at the differentiated mix recomputed from its snapshot with psi before vs after, the query evidence; and for
the *next live* policy update: whether the psi it used is the one after the meta-step and how much the meta-step
changed it.

**Run 1, default configuration** (`real_pathway_20261127T200546Z.json`; 5 lifetimes, 60 decisions; peak 10.5 GB;
lock held 1234 s — over the 15-min guideline because a lifetime slowed by CPU load started at 723 s; the script now
never starts a lifetime that would overrun):

| step of the pathway | measured |
|---|---|
| meta-update calls / applied | 63 / 1 (58 insufficient independent outcomes, 2 no policy update yet, 2 ESS too small) |
| reward params psi -> internal reward on the 24 support decisions | mean abs change 0.023 (std of R_psi there 0.028) |
| -> inner update (theta_0 -> theta_K) | mean TV 0.13 (differentiated, mix 0) / 0.11 (live, mix 1) on the support |
| -> changed policy -> query decisions afterwards with signed verified outcomes | 3 query rows, 17 independent groups, utility mean 35.2, ESS 2.8, importance-weighted |
| -> outer meta-gradient -> psi delta | gradient norm 12.8 (14 psi tensors non-zero), psi delta 0.097, reward version changed |
| -> the inner update recomputed with psi after vs before | differentiated (mix 0): mean TV 0.066, max 0.037; **live (mix 1): 0** |
| -> next live policy update | **0** change: the mix was 1 (the one promotion decision: hold, mean diff +8.6, 95 % bounds [-15.0, 32.2], 5 groups), and the psi it used was not the psi after the meta-step although no other step was applied — 3 checkpoint rollbacks happened (inferred cause; LRN-6) |

So on the real configuration psi -> R_psi -> (differentiated) inner update -> verified query outcomes -> meta-gradient
-> psi -> a changed (differentiated) next update holds; the *live* update was not changed, because in a run this
short the learned reward was never promoted and a rollback reverted psi.

**Run 2, pathway-check configuration** (`real_pathway_check_20261127T223812Z.json`; the same real model; the mix
fixed at 0.5 so the live update depends on R_psi as it would after two promotions, and `agent.promotion = false` so
no rollback interferes (LRN-6); 4 lifetimes, 48 decisions, lock held 657 s, peak 10.4 GB):

| applied meta-update | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| meta-gradient norm (14 psi tensors non-zero; importance-weighted, ESS 2.0-2.4) | 0.0095 | 0.23 | 0.80 | 0.073 |
| psi delta (L2) | 0.095 | 0.073 | 0.062 | 0.053 |
| R_psi on the support decisions, mean abs change (its std there) | 0.026 (0.024) | 0.015 (0.029) | 0.014 (0.032) | 0.012 (0.037) |
| that event's live update (mix 0.5) recomputed with psi after vs before, mean TV | 0.0053 | 0.0078 | 0.041 | 0.0067 |
| **the next live policy update** (it used exactly the psi after the step), changed by the step, mean TV / max prob | **0.0090 / 0.0036** | **0.0042 / 0.0031** | (run ended) | (run ended) |

Every live update of this run depended on R_psi's shape (mean TV 0.0016-0.055 when R_psi's output layer is
perturbed). The full chain — reward parameters -> internal reward -> inner update -> changed policy -> later
verified outcomes -> outer meta-gradient -> reward-parameter delta -> the next live inner update changed by it —
holds on the real configuration once R_psi has weight in the live update. In the default configuration that weight
comes only from promotion, which needs more held-out evidence than a 5-lifetime run provides (one "hold"), and a
refused checkpoint promotion currently reverts psi (LRN-6). The effects are small (TV <= 0.04); whether the learned
reward makes later behaviour *better* is not established (debug-scale E vs D vs E_mix0 overlap, section 3).
Debug-scale replica of run 2: `pathway_debug_mix05_20261127.json` (4 applied, next live update changed by TV
0.0085 / 0.0052 / 0.0033).

## 5. Tests (regression tests fail on the code before the fix)

* `tests/predict/test_selection_value.py` — rule registered; an untrained policy follows the world model's predictions
  with exact propensities (the old default rule is uniform on the same evaluations); Thompson spreads by posterior
  overlap; infeasible = 0 and ties not broken by position; the policy's weight grows only when its preference
  predicts verified progress (noise -> no influence, harmful -> lambda 0, a policy that repeats the world model ->
  no influence); recorded features and persistence; on the debug model after two LabWorld lifetimes the rule's
  choice has 0.92 of the predicted value range vs 0.23 for rule `policy`.
* `tests/predict/test_selection_agent.py` — the integrated agent through the runtime: every decision records the
  rule's inputs, the close-out's signed outcomes reach the calibration, a checkpoint restores it exactly.
* `tests/reward/test_reward_lrn.py` — delta V has no action-dependent offset; verified progress only from the
  decision's own outcomes; the inner update is scale-invariant at every mix and follows R_psi's shape with weight
  growing as the mix falls (exactly 0 at mix 1); a policy update moves toward the decisions with verified progress
  (goal term) and away from candidates predicted worse (model term); pending decisions use the calibrated prediction,
  labelled; the trust-region step reaches the target KL (the old fixed step: KL < 1e-3) and replays bit-identically;
  the learner feeds the calibration once per final outcome and persists it.
* Changed expectations (behaviour changes, documented in the tests): promotion holds on inconclusive evidence instead
  of raising (`test_reward_bootstrap.py`, `test_reward_metalearner.py`); success_before is the recorded state value
  (`test_reward_inner.py`); full saturation of R_psi is a shape change (`test_reward_metalearner.py`, cosine bound
  for x1000 0.75 with the measured 0.85; x10 now > 0.99).

## 6. Limitations

* Debug scale only for the choice diagnostics (tiny random backbone, LabWorld). No learning curve over lifetimes was
  observed in any arm; the improvement is in how choices use the predictions, not in the predictions.
* The calibration's beta_q is small and noisy (~0.1-0.3 in units of the verified RMS per prediction spread): the
  verified progress of one decision is very noisy, so Thompson keeps exploring among well-predicted candidates
  (expected quality 0.96 of greedy-on-prediction). Its priors, forgetting (0.99), lambda cap (10) and the projection
  are design choices, not tuned.
* The calibration and the goal term use the agent's per-decision annotation `reward_signals.verified_progress_delta`
  (INT2-1 workaround): an environment that attributes nothing gets the level change, which can still credit a
  decision with another decision's consequences.
* Delayed effects without attribution (LabWorld's `lab.unlock`) are credited only through the return-to-go of the
  goal term; the selection's calibration regresses on immediate attributed progress and so cannot learn their value.
* The learned reward is not shown to help: E vs D vs E_mix0 intervals overlap; in E_mix0 R_psi pays zero-value
  decisions more than valued ones; promotions in E were mostly "hold" (inconclusive). The promotion margin (0.05 x
  utility RMS), step (0.25) and 4-group minimum are set by hand.
* Learning progress and epistemic reduction still have no inputs in runtime runs (LRN-3).
* Restore determinism (INT2-8, also fixed here): the recovery tests passed only when nothing diverged. Two causes were
  in LRN's files — experiment assignment hashed the random experiment id, and restored update events rebuilt their
  support batch from the store — both fixed (5ad3777, tests `test_registry.py::test_assignment_is_reproduced_by_a_
  restored_registry...`, `test_reward_lrn.py::test_restored_update_events_keep_the_support_batch...`);
  `tests/integration/test_agent_recovery.py` passes (2 of 2 runs). R_psi's version hash still differs bitwise after
  the first post-load reward step (input statistics from re-computed embeddings, float order) without changing a
  decision.

## 7. Requests to INT2

`docs/change_requests/LRN.md`: LRN-1 default `selection_rule = "thompson_policy"` (applied by INT2 in d55986d), LRN-3
record `epistemic_after` / `learning_progress`; LRN-5 withdrawn (fixed in LRN's files together with INT2-8).
No other agent wiring is needed (calibration shared through the policy head and checkpointed by `meta.state_dict()`).
