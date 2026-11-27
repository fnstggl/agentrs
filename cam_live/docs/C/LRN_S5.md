# Learning-mechanism fixes for change request S-5 (workstream LRN, 2026-11-27)

Owner of `predict/` and `reward/` for this round. Diagnosis being answered: `docs/S/DIAGNOSIS.md`,
`docs/change_requests/S.md` S-5, INT2-1 / INT2-6 / INT2-7 (`docs/change_requests/INT2.md`), INT2's measured
diagnosis `docs/evidence/validation/diagnosis/debug_f1_20261127/`. Engineering diagnostics at debug scale (tiny random
backbone, debug core, LabWorld F1), plus one real-configuration pathway run; none of this is a scientific result.

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
* `c_j` — the trainable policy's preference: centred log pi divided by max(1 nat, its spread). An untrained,
  near-uniform policy has c ~ 0 and no say at all; a very peaked one is bounded like z.
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
