# G3-core — independent verification of the core-intelligence amendment (items 1-4)

Verifier G3-core (workstream G, lens `g3core`). I wrote none of the verified code. Freeze commit
**6cba496** (`6cba496359da28a9742bcd04fa4a558fdec19160`). Everything below was executed on a `git archive` of
that commit
(`scratchpad/G3CORE/freeze`, `PYTHONPATH=<freeze>/cam_live/src`). The worktree has one later commit (a52dead, `[S]`,
`src/cam_live/eval/oe_*.py` and `eval/v3_report.py` only); no verified module is touched by it. CPU rule observed:
`OMP_NUM_THREADS<=4`, `nice -n 10`, one CPU-heavy job at a time. **No GPU hold was used** (the GPU was at 100 %
utilisation by other holders during the window, and no finding needed the real model; see "Not checked").
Probes and their outputs: `docs/verification/repro/g3core/` (`out/` holds every number quoted here).

Levels (operator item 8): (a) implementation correctness, (b) integration into the actual runtime, (c) causal
influence on behaviour, (d) measured improvement, or an honest negative result with a correct diagnosis.

## 0. Summary

| capability | (a) correctness | (b) runtime integration | (c) causal influence | (d) improvement / diagnosis |
|---|---|---|---|---|
| RW: evidence-based promotion of R_psi (validity e-processes) | **holds for its own null**. Type-I error of the whole procedure with continuous monitoring is 1.0 % (iid) and 0.5 % (tied bootstrap) at a nominal 2.5 %. RW's 1.75 % reproduces bit for bit. | **holds**. Defaults are `validity` / `auto` / `direct`, nothing in `configs/` overrides them, and the RW-1 / RW-2 hooks are present. | **not reached in the agent**. The mix stayed at 1 in every integrated or real run, so R_psi had TV 0.0 influence. | **not established**, as RW says. Two defects RW missed: **D-RW1** (an *unlearned* R_psi is promoted: 3 of 12 vs 2 of 6 learned at the agent's step) and **D-RW2** (the mix cannot go below 0.75 where the bootstrap ties at reward time), §1. |
| RW: pooled first-order outer estimator; per-decision verified credit; level->delta fix | **holds**. FD tests re-run and pass. The pool excludes the support and its groups. The value comes only from signed readings (`agent.py:1237`). The A5 fix is tested. | **holds**. The real run applied 4 of 36 meta-updates through the query fallback; the pooled estimator never had a pool there. | n/a | Diagnosed limits (return half-life, evidence rate): I agree with them. |
| IMP: multistep planner, value-consistency horizon, ImaginationCalibration, transition priors | **holds**, with a latent calibration weakness **D-IMP3** (overlapping windows). | **holds**. `agent.py:1959-1969`: plan -> extras -> `thompson_policy`. | **holds within a decision** (share_tv). The cross-arm claim is **refuted**: the B2 arms are not paired, **D-IMP1**. | **none**, and IMP reports this honestly. Imagination cannot dominate under the prior: mean weight 0.17, and when nothing supports it the posterior pulls the weight below the prior. |
| IMP: content target `predict/content.py` | **partly refuted**. The target is **not fixed**: it moves with the LoRA adapters, **D-IMP2**. No gradient path into it, so it cannot collapse. | trained and recorded | none: no selection rule reads it, which IMP states | not claimed |
| PS: persistent core-state chain | **holds**. nn suite 123 passed: decision-time equals replay to 1e-6, and the FD gradient crosses K = 2. | **holds when enabled**. The default is off (`persist.py:76`), so the default runtime has **no** persistent state. | small, measured by PS (TV 0.004 at full size) | **none**. "Default off" is consistent with that evidence. No defect found. |
| CR: segment credit inside one objective; estimand-matched selection; CR-3 replay | **holds**. credit suite 155 passed; the IPW bias identity was checked by derivation. | **holds**. CR-1 / CR-3 are in `agent.py:805, 2486`. | Selection never had 8 units, so no promotion is possible yet. | The forced-RUDDER result of -66.8 adopters is **confounded**, **D-CR1**. Validation and training use different quantities, **D-CR2**. |

Binding prohibitions: I found no hardcoded strategy, tool name, niche or plan-instructing prompt in the new code. I
grepped `reward/validity.py`, `reward/metalearner.py`, `predict/planner.py`, `predict/policy.py`,
`predict/content.py`, `nn/persist.py`, `credit/segments.py`, `credit/select.py` and `credit/providers.py`. No mix is
forced (`configs/` sets no `mix`, `promotion_rule`, `imagination_select` or `segment_decisions`), and no LLM planner
is used: the planner is a latent search over the learned heads, with no `generate` call.

Re-run test suites (freeze copy, CPU): `tests/reward` **169 passed, 3 skipped** (191 s). `tests/credit` **155 passed,
9 skipped** (181 s). `tests/nn` **123 passed, 13 skipped** (102 s). `tests/predict` **181 passed, 1 failed, 10
skipped** (228 s). The failure is `test_int3a_eligibility.py::test_a_record_checked_before_its_verdict_becomes_eligible_for_the_evaluator`
(0 of 12 records eligible for the evaluator). It also fails when run alone, but passes when its whole file runs (2 passed): an order- or RNG-state-dependent test, not attributable to the amendment by these runs (§5). The 4 integration acceptance tests pass, and so do 8 slow acceptance tests (§5). JUnit and logs are in
`scratchpad/G3CORE/tests/`.

## 1. RW — genuinely learned rewards (amendment item 1)

### 1.1 What I re-ran and checked

* **(a) Tests.** `tests/reward` 169 passed and 3 skipped on the freeze. This covers `test_reward_rw.py`: sequential
  statistics, validity wiring, FD of the pooled estimator, the `auto` fallback, level readings and checkpoint
  compatibility.
* **(a) RW's power simulation** (`docs/evidence/RW/promotion_power_sim.py`) re-run on the freeze is **bit-identical**
  to `promotion_power.json`: 1.75 % ever promoted at p = 0.5, and median 129 / 52 / 24 / 18 pairs at 70 / 80 / 90 / 100 %
  (`repro/g3core/out/rw_promotion_power_rerun.json`). Its bootstrap orders every pair at random, but in RW's own
  LabWorld data the bootstrap **ties** at reward time: `concordance_bootstrap` is exactly 0.000 in all 24 diagnostic
  runs. The power numbers therefore describe a different score distribution than the one realised. With ties the
  simulation is conservative (x in {+-0.5}, smaller variance).
* **(a) The whole procedure under optional stopping** (`probe_rw_validity_procedure.py`). I ran note -> resolve ->
  disjoint pairing -> `decide()` after every scored pair -> mix walk, 200 runs x 1000 pairs.
  * iid null: lowered in **1.0 %** of runs.
  * null with a tied bootstrap: **0.5 %**.
  * Both are within the nominal alpha/2 = 2.5 %. The fixed-sequence / Ville argument
    (`validity.py:391-412`) is correct: it needs E[x | past] <= 0, not independence, so dependent consecutive pairs
    do not invalidate it.
* **(a) Pooled estimator.** Read `metalearner.py:915-1084`.
  * The pool excludes the support rows and every independent-outcome group of the support (`_pool`).
  * The IS weights use logged propensities: self-normalised, clipped, with an ESS floor and leave-group-out baselines.
  * The surrogate is `<g, theta_K(psi)>`, matching RW §2.2. Its FD test passes.
* **(a) Per-decision verified credit.** `agent.py:1207-1262` writes `verified_progress_delta` once and only from
  readings that pass `_check_measurement`. `inner.verified_progress` reads only that annotation. ST-3's information
  value lives in its own key, `reward_signals.information_value` (`assemble.annotation_key`), so it cannot clobber
  the verified annotation.
* **(a) Level->delta (A5).** The level key is `"<objective>|<metric>"` (`metalearner.py:616-619`), so a new objective
  re-anchors and does not inherit the previous one's level. The regression tests pass.
* **(b)** `MetaConfig` defaults are `promotion_rule="validity"`, `estimator="auto"` and `utility="direct"`, and
  `configs/cam_live.yaml` overrides none of them. The RW-1 / RW-2 hooks are in `agent.py:2709, 2891`.

### 1.2 Defects and refutations

**D-RW1 — an *unlearned* R_psi is promoted as often as a learned one at the agent's reward step. The claims "an
uninformative one (random init, frozen psi) is not promoted" (`reward/validity.py:53`, RW.md §2.1) and "only learning
lets R_psi act / no frozen arm ever moved the mix" (RW.md §3.4 (c)) are refuted.**

A random initialisation is not uninformative. It is a fixed random function of the signals and latents. RW's own
frozen arms (R_psi never trained) show prospective concordances whose anytime 95 % CS **excludes 0** in 3 of 9
distinct runs: -0.20 [-0.35, -0.07], -0.21 [-0.37, -0.07] and -0.22 [-0.42, -0.02]
(`docs/evidence/RW/diag_summary_{300,600}.json`). R_psi's output layer is N(0, 0.02) with zero bias and an odd bound
(`bound * tanh`), so the **negated output layer is an equally likely initialisation**. RW's frozen control used
only one sign of each initialisation, and 7 of its 9 runs had concordance <= 0.

`probe_rw_flipped_frozen.py` re-runs RW's controlled 600-decision configuration (`rw_meta_diag.run`, the same seeds
and worlds, reward step 0) with the output layer negated. Its frozen arms reproduce RW's numbers exactly; seed 0, for
example, is -0.0669 [-0.2595, 0.0743]. Learned columns are from RW's `diag_summary_600.json`
(`out/rw_learned_vs_random_init.txt`):

| seed | unlearned R_psi: init / negated init | negated init promoted? | true adopters per decision after promotion, negated minus bootstrap-only (same decisions before) | learned, lr 3e-3 | learned, lr 1e-4 (agent's) |
|---|---|---|---|---|---|
| 0 | -0.067 / +0.074 | no | - | +0.388 promoted | +0.173 promoted |
| 1 | +0.040 / -0.003 | no | - | +0.191 promoted | +0.185 promoted |
| 2 | -0.204 / **+0.248** [0.088, 0.392] | **yes, decision 151** (e-value 3.0e5) | -0.77 [-2.10, +0.60], 449 decisions | +0.145 promoted | +0.029 |
| 3 | -0.057 / +0.057 | no | - | - | +0.055 |
| 4 | -0.207 / **+0.222** [0.093, 0.396] | **yes, decision 99** (e-value 1.4e4) | +1.04 [+0.71, +1.39], 501 decisions | - | -0.052 |
| 5 | -0.154 / +0.182 | yes, decision 566 (e-value 293) | 0.00 (34 decisions, not yet diverged) | - | -0.190 |

* **Unlearned rewards, both signs (n = 12):** mean concordance +0.011, mean |concordance| 0.126, **promoted 3 of 12**.
* **Learned at the agent's reward step 1e-4 (n = 6):** mean concordance +0.033, **promoted 2 of 6**.
* Only at the 30x larger reference step (3e-3) do learned rewards clearly exceed the random-init range: 3 of 3
  promoted, and +0.39 in seed 0, beyond any unlearned value.

Consequences:

1. At the agent's own reward step, the promotion evidence does not distinguish a meta-learned R_psi from a random
   one. RW's "gain over the frozen R_psi on the same pairs" (17 of 18 positive) is measured against a baseline that
   was mostly negative by chance. Moving a negatively concordant initialisation towards 0 is what any perturbation of
   psi can do (regression to the mean), so the positive gains alone do not show learning.
2. After promotion, an unlearned reward produced both a clear gain (seed 4, +1.04 [0.71, 1.39]) and a loss pointing
   the other way (seed 2). That is the same range as RW's learned promotions (+2.14, -0.14, -2.89, +0.70, +0.87), so
   RW's one clear post-promotion gain is not attributable to meta-learning.
   * Promoting seed 4's lucky random reward was *useful*, so this is not a statistical error of the test.
   * What is refuted is that the evidence certifies a *learned* reward.
   * The promotion in seed 2 brought no measurable benefit.
3. The procedure is statistically valid for its own H0. My checks give 1.0 % and 0.5 % false lowering under exact
   nulls. That H0 is "R_psi's concordance over consecutive matured units is no better than the mixture's", not
   "R_psi learned something useful". The first-order argument (`validity.py:7-14`: concordance = within-state
   covariance) assumes the two units of a pair share their state. A reward that ranks the states it passes through
   (`success_before`, the lifetime context) the way outcomes happen to vary between consecutive decisions earns
   concordance without a useful within-state ranking (seed 2).
4. The same holds relative to the bootstrap. Against an anti-concordant bootstrap, an R_psi independent of outcomes
   is lowered in **100 %** of runs (`rw_validity_procedure.json`, `boot_anti`). That is correct under the rule's H0,
   but it contradicts the documented claim.

Severity: **moderate-high** for item 1's central claim ("an empirically grounded mechanism for determining when
learned rewards are useful"). The mechanism decides when R_psi's *ranking agrees*, whether or not anything was
learned. Fix directions (not mine to make):

* control with a sign-symmetric population of unlearned rewards;
* require the learned reward to beat the agreement of its own frozen initialisation on the same pairs (a paired
  "learning helped" test);
* score within-state comparisons (candidates of the same decision, e.g. via the logged propensities) instead of
  consecutive units.

Reproduce:

    OMP_NUM_THREADS=4 nice -n 10 env PYTHONPATH=<freeze>/cam_live/src .venv/bin/python \
      cam_live/docs/verification/repro/g3core/probe_rw_flipped_frozen.py \
      --rw-dir <freeze>/cam_live/docs/evidence/RW --seeds 0 1 2 3 4 5 --out out/rw_flipped_frozen.json   # ~22 min CPU
    python analyse_rw_learned_vs_random_init.py --rw-summary <freeze>/cam_live/docs/evidence/RW/diag_summary_600.json \
      --flipped out/rw_flipped_frozen.json

Note: decisions are identical in both arms until the mix moves (`identical_before_first_change` true in all three
promoted seeds). The *pairing* of units can differ between processes in some seeds (unit ids are random), so a
negated arm's concordance is not always the exact negative of its frozen arm's. That reproducibility weakness has no
validity consequence, because the pairing never reads outcomes.

**D-RW2 — where the bootstrap ties at reward time, R_psi's weight can never exceed 0.25, whatever its quality.**

With `r_bootstrap` equal for every decision at reward time, `pair_scores` gives c_m = c_0 (R_psi alone) for every
m < 1. The lowering increment at every m_k < 1 is then `x+ = (c_0 - c_mk)/2 = 0` for every pair
(`validity.py:374`), so its e-value stays exactly 1. The raising increment is `-c_0/2 < 0` for a concordant R_psi.

Probe (`tied_boot_perfect_R_psi_300_pairs`): a **perfect** R_psi with a tied bootstrap ends at mix 0.75 in 50 of 50
runs, with e-value 1.0 at 0.75. In RW's diagnostics the bootstrap ties in every run: `concordance_bootstrap` is 0.000 in all 24 summaries, and every reward-time bootstrap value was exactly 0.0 in a traced run of the same configuration, and **all 7 promotions ended at
0.75** (RW.md §3.4 table). RW reports the 0.75 as a result and does not recognise it as structural.

The underlying design gap: the score is computed on the *reward-time* bootstrap. The mix weighs the *update-time*
bootstrap, which contains components that mature later, e.g. ST-3's information value (RW.md §2.4: "absent from the
bootstrap reward at reward time ... present in the live updates"). The evidence therefore cannot speak about what
lowering the mix further does to the live update. At full size the bootstrap does not tie (real run: bootstrap
concordance -0.06 over 17 pairs), so the cap is environment-dependent.

Severity: **moderate** (limits item 1's "learned rewards influence ordinary updates" to 25 % in such worlds).

### 1.3 Levels

* (a) holds (statistics, estimator, annotation, level fix).
* (b) holds.
* (c) In the integrated agent R_psi never had weight: mix 1 throughout the debug 120-decision run and the real
  35-decision run, with TV 0.0 on 11 real live updates, as RW reports. In the controlled diagnostic a promotion
  changes decisions 7-31 decisions later. For the promoted unlearned rewards the delay was 27 and 68 decisions
  (seeds 2 and 4: promotion at 151 / 99, first different decision 178 / 167). That meta-learning
  improved R_psi's agreement is shown only at the 30x reference step (D-RW1).
* (d) Improvement is not established (RW says so). Beyond RW's own caveats, D-RW1 shows that at the agent's step
  neither promotion nor post-promotion gains can be attributed to learning: an equally likely unlearned R_psi was
  promoted in 3 of 12 cases, with post-promotion effects of -0.77 to +1.04.

RW's negative-result diagnoses (return half-life smearing credit, evidence rate of tens of informative pairs) are
measured and plausible. I did not re-run `rw_credit_probe.py`.

## 2. IMP — predictive imagination and multistep planning (amendment item 2)

### 2.1 Checked

* **(a)** `tests/predict` 181 passed; all 16 IMP tests pass on the freeze.
* **(a) Transition priors** (`world.py:532-562`).
  * Fixed, seeded generator (`seed*7919+104729`, independent of the global RNG), zero-mean over members, added only
    to the residual mean, not trainable (non-persistent buffers).
  * Regenerated identically after a restore, so the members' learned cancellation stays valid.
  * Minor overstatement: "a transition trained without them predicts the same mean" holds at initialisation and on
    the training data, not off-data after training (the members learned to cancel their own offsets).
* **(a) Value-consistency horizon** (`imagination.HorizonController`, `planner.measure`). The criterion compares the
  candidates' values at the imagined and at the real belief, relative to how much the values differ between
  candidates. If the world model's values barely depend on the belief, the criterion admits the full horizon even
  though imagination carries nothing. That is what happened at full size: value error 0.040 against a persistence
  baseline of 0.038, giving horizon 4. The criterion therefore certifies that imagination is harmless, not that it
  is accurate about consequences. IMP states this ("harmless ... not informative") and records the baseline, so this
  is not a defect, but "the horizon grows only as measured accuracy supports it" overstates it.
* **(b)** `agent.py:1959-1969`: `planner.plan(..., predictor, evaluations)` -> `ev.extras.update(im.extras())` ->
  `select` -> `DecisionMaker.thompson_policy`. Calibration is fed by `_measure_planner` -> `refresh_calibration`.
* **(c) Can poorly calibrated imagination dominate?** Prior draws give a mean weight of 0.174. Over 20,000 draws,
  lambda_im = beta_im^+ / max(beta_q, 1) with beta_im ~ N(0, 0.5^2) and m winsorised at +-4, so under the prior an
  imagined preference can move a score by at most ~0.7 one-step sd on average, in half the Thompson samples.
  In B2 this changed the argmax in 9 % of decisions: exploratory, not dominant.
  With evidence that m carries nothing, the posterior pulls the weight *below* the prior (0.06-0.13 in my
  simulation), so discredited imagination loses influence as designed. The exception is D-IMP3.
* **(d)** No improvement. IMP reports this honestly (B2 +0.237 in lifetime 0, not surviving lifetime 1; §3.6 none;
  full-size pair: on worse, n = 1).

### 2.2 Defects and refutations

**D-IMP1 — the "paired" B2 ablation is not paired; the inference "the arms diverge, therefore imagination causally
changes decisions" (IMP.md §3.3) is refuted.**

`predict/imp_eval.py:10-15` says the arms "draw the same random numbers, so with the same seed they are identical
until behaviour differs". In the raw B2 rows (`docs/evidence/IMP/ablation_B2_final_code_raw.tar.xz`, lifetime 0,
**decision 0**: identical context, no learning yet) the arms' recorded inputs already differ. The one-step values
are up to 0.16 apart and the imagined continuations up to 4.69 apart (seed 7). The executed action differs at
decision 0 in **5 of 10** pairs.

If only imagination caused divergence, the expected count would be sum share_tv = 0.53, and
P(>= 5) = 5e-5 (Poisson-binomial). Seed 1 diverged at a decision where the on arm's distributions with and without
imagination were identical (share_tv = 0.000), which no coupling can attribute to imagination. Seed 7, IMP's example
("unlocked at its first decision ... achieved the objective in 7 decisions"), diverged where the two arms' imagined
continuations already differed by 4.69.

`repro/g3core/out/imp_b2_pairing.txt` (regenerate with `analyse_imp_b2_pairing.py`, §6). Consequences:

* The within-decision counterfactual (share_tv 0.053, argmax changed 9 %) remains valid causal evidence.
* The cross-arm divergence is not evidence.
* The B2 outcome CIs are effectively unpaired comparisons. The direction of IMP's (d) conclusion (no reliable
  improvement) is unaffected.

**Cause, partly isolated** (`probe_imp_same_arm_twice.sh`, `out/imp_same_arm_twice.txt`, `out/imp_same_seed_off.json`).
I ran on the freeze code, on a lightly loaded CPU, with `PYTHONHASHSEED=0`, seed 1, 1 lifetime of 3 decisions. Two
runs of the on arm and one run of the off arm all executed **identical actions in all 3 decisions**. At decision 0
their inputs differ by at most 9e-6 (one-step value) and 1.7e-4 (imagined continuation), and the differences grow
to about 1e-2 by decision 2. The on-vs-off differences are the same size as the on-vs-on ones, so the arm switch
does not perturb anything before selection, and the design *can* be paired. The large decision-0 differences in
the recorded B2 runs therefore come from the run environment, not from imagination. The B2 lifetimes took
277-450 s on the loaded machine. The agent and world run on a wall clock (`validation.OffsetClock`,
`validation.py:47-57`, used at `predict/imp_eval.py:117`), so timing enters elapsed-time features and deadlines.
The B2 runs also did not record `PYTHONHASHSEED`. Small float differences also grow with learning, so long paired
runs diverge in any case. That growth is PS's "trajectory noise" point, and it applies here too.

**D-IMP2**. No gradient path into it, so it cannot collapse. | trained and recorded | none: no selection rule reads it, which IMP states | not claimed |
| PS: persistent core-state chain | **holds**. nn suite 123 passed: decision-time equals replay to 1e-6, and the FD gradient crosses K = 2. | **holds when enabled**. The default is off (`persist.py:76`), so the default runtime has **no** persistent state. | small, measured by PS (TV 0.004 at full size) | **none**. "Default off" is consistent with that evidence. No defect found. |
| CR: segment credit inside one objective; estimand-matched selection; CR-3 replay | **holds**. credit suite 155 passed; the IPW bias identity was checked by derivation. | **holds**. CR-1 / CR-3 are in `agent.py:805, 2486`. | Selection never had 8 units, so no promotion is possible yet. | The forced-RUDDER result of -66.8 adopters is **confounded**, **D-CR1**. Validation and training use different quantities, **D-CR2**. |

Binding prohibitions: I found no hardcoded strategy, tool name, niche or plan-instructing prompt in the new code. I
grepped `reward/validity.py`, `reward/metalearner.py`, `predict/planner.py`, `predict/policy.py`,
`predict/content.py`, `nn/persist.py`, `credit/segments.py`, `credit/select.py` and `credit/providers.py`. No mix is
forced (`configs/` sets no `mix`, `promotion_rule`, `imagination_select` or `segment_decisions`), and no LLM planner
is used: the planner is a latent search over the learned heads, with no `generate` call.

Re-run test suites (freeze copy, CPU): `tests/reward` **169 passed, 3 skipped** (191 s). `tests/credit` **155 passed,
9 skipped** (181 s). `tests/nn` **123 passed, 13 skipped** (102 s). `tests/predict` **181 passed, 1 failed, 10
skipped** (228 s). The failure is `test_int3a_eligibility.py::test_a_record_checked_before_its_verdict_becomes_eligible_for_the_evaluator`
(0 of 12 records eligible for the evaluator). It also fails when run alone, but passes when its whole file runs (2 passed): an order- or RNG-state-dependent test, not attributable to the amendment by these runs (§5). The 4 integration acceptance tests pass, and so do 8 slow acceptance tests (§5). JUnit and logs are in
`scratchpad/G3CORE/tests/`.

## 1. RW — genuinely learned rewards (amendment item 1)

### 1.1 What I re-ran and checked

* **(a) Tests.** `tests/reward` 169 passed and 3 skipped on the freeze. This covers `test_reward_rw.py`: sequential
  statistics, validity wiring, FD of the pooled estimator, the `auto` fallback, level readings and checkpoint
  compatibility.
* **(a) RW's power simulation** (`docs/evidence/RW/promotion_power_sim.py`) re-run on the freeze is **bit-identical**
  to `promotion_power.json`: 1.75 % ever promoted at p = 0.5, and median 129 / 52 / 24 / 18 pairs at 70 / 80 / 90 / 100 %
  (`repro/g3core/out/rw_promotion_power_rerun.json`). Its bootstrap orders every pair at random, but in RW's own
  LabWorld data the bootstrap **ties** at reward time: `concordance_bootstrap` is exactly 0.000 in all 24 diagnostic
  runs. The power numbers therefore describe a different score distribution than the one realised. With ties the
  simulation is conservative (x in {+-0.5}, smaller variance).
* **(a) The whole procedure under optional stopping** (`probe_rw_validity_procedure.py`). I ran note -> resolve ->
  disjoint pairing -> `decide()` after every scored pair -> mix walk, 200 runs x 1000 pairs.
  * iid null: lowered in **1.0 %** of runs.
  * null with a tied bootstrap: **0.5 %**.
  * Both are within the nominal alpha/2 = 2.5 %. The fixed-sequence / Ville argument
    (`validity.py:391-412`) is correct: it needs E[x | past] <= 0, not independence, so dependent consecutive pairs
    do not invalidate it.
* **(a) Pooled estimator.** Read `metalearner.py:915-1084`.
  * The pool excludes the support rows and every independent-outcome group of the support (`_pool`).
  * The IS weights use logged propensities: self-normalised, clipped, with an ESS floor and leave-group-out baselines.
  * The surrogate is `<g, theta_K(psi)>`, matching RW §2.2. Its FD test passes.
* **(a) Per-decision verified credit.** `agent.py:1207-1262` writes `verified_progress_delta` once and only from
  readings that pass `_check_measurement`. `inner.verified_progress` reads only that annotation. ST-3's information
  value lives in its own key, `reward_signals.information_value` (`assemble.annotation_key`), so it cannot clobber
  the verified annotation.
* **(a) Level->delta (A5).** The level key is `"<objective>|<metric>"` (`metalearner.py:616-619`), so a new objective
  re-anchors and does not inherit the previous one's level. The regression tests pass.
* **(b)** `MetaConfig` defaults are `promotion_rule="validity"`, `estimator="auto"` and `utility="direct"`, and
  `configs/cam_live.yaml` overrides none of them. The RW-1 / RW-2 hooks are in `agent.py:2709, 2891`.

### 1.2 Defects and refutations

**D-RW1 — an *unlearned* R_psi is promoted as often as a learned one at the agent's reward step. The claims "an
uninformative one (random init, frozen psi) is not promoted" (`reward/validity.py:53`, RW.md §2.1) and "only learning
lets R_psi act / no frozen arm ever moved the mix" (RW.md §3.4 (c)) are refuted.**

A random initialisation is not uninformative. It is a fixed random function of the signals and latents. RW's own
frozen arms (R_psi never trained) show prospective concordances whose anytime 95 % CS **excludes 0** in 3 of 9
distinct runs: -0.20 [-0.35, -0.07], -0.21 [-0.37, -0.07] and -0.22 [-0.42, -0.02]
(`docs/evidence/RW/diag_summary_{300,600}.json`). R_psi's output layer is N(0, 0.02) with zero bias and an odd bound
(`bound * tanh`), so the **negated output layer is an equally likely initialisation**. RW's frozen control used
only one sign of each initialisation, and 7 of its 9 runs had concordance <= 0.

`probe_rw_flipped_frozen.py` re-runs RW's controlled 600-decision configuration (`rw_meta_diag.run`, the same seeds
and worlds, reward step 0) with the output layer negated. Its frozen arms reproduce RW's numbers exactly; seed 0, for
example, is -0.0669 [-0.2595, 0.0743]. Learned columns are from RW's `diag_summary_600.json`
(`out/rw_learned_vs_random_init.txt`):

| seed | unlearned R_psi: init / negated init | negated init promoted? | true adopters per decision after promotion, negated minus bootstrap-only (same decisions before) | learned, lr 3e-3 | learned, lr 1e-4 (agent's) |
|---|---|---|---|---|---|
| 0 | -0.067 / +0.074 | no | - | +0.388 promoted | +0.173 promoted |
| 1 | +0.040 / -0.003 | no | - | +0.191 promoted | +0.185 promoted |
| 2 | -0.204 / **+0.248** [0.088, 0.392] | **yes, decision 151** (e-value 3.0e5) | -0.77 [-2.10, +0.60], 449 decisions | +0.145 promoted | +0.029 |
| 3 | -0.057 / +0.057 | no | - | - | +0.055 |
| 4 | -0.207 / **+0.222** [0.093, 0.396] | **yes, decision 99** (e-value 1.4e4) | +1.04 [+0.71, +1.39], 501 decisions | - | -0.052 |
| 5 | -0.154 / +0.182 | yes, decision 566 (e-value 293) | 0.00 (34 decisions, not yet diverged) | - | -0.190 |

* **Unlearned rewards, both signs (n = 12):** mean concordance +0.011, mean |concordance| 0.126, **promoted 3 of 12**.
* **Learned at the agent's reward step 1e-4 (n = 6):** mean concordance +0.033, **promoted 2 of 6**.
* Only at the 30x larger reference step (3e-3) do learned rewards clearly exceed the random-init range: 3 of 3
  promoted, and +0.39 in seed 0, beyond any unlearned value.

Consequences:

1. At the agent's own reward step, the promotion evidence does not distinguish a meta-learned R_psi from a random
   one. RW's "gain over the frozen R_psi on the same pairs" (17 of 18 positive) is measured against a baseline that
   was mostly negative by chance. Moving a negatively concordant initialisation towards 0 is what any perturbation of
   psi can do (regression to the mean), so the positive gains alone do not show learning.
2. After promotion, an unlearned reward produced both a clear gain (seed 4, +1.04 [0.71, 1.39]) and a loss pointing
   the other way (seed 2). That is the same range as RW's learned promotions (+2.14, -0.14, -2.89, +0.70, +0.87), so
   RW's one clear post-promotion gain is not attributable to meta-learning.
   * Promoting seed 4's lucky random reward was *useful*, so this is not a statistical error of the test.
   * What is refuted is that the evidence certifies a *learned* reward.
   * The promotion in seed 2 brought no measurable benefit.
3. The procedure is statistically valid for its own H0. My checks give 1.0 % and 0.5 % false lowering under exact
   nulls. That H0 is "R_psi's concordance over consecutive matured units is no better than the mixture's", not
   "R_psi learned something useful". The first-order argument (`validity.py:7-14`: concordance = within-state
   covariance) assumes the two units of a pair share their state. A reward that ranks the states it passes through
   (`success_before`, the lifetime context) the way outcomes happen to vary between consecutive decisions earns
   concordance without a useful within-state ranking (seed 2).
4. The same holds relative to the bootstrap. Against an anti-concordant bootstrap, an R_psi independent of outcomes
   is lowered in **100 %** of runs (`rw_validity_procedure.json`, `boot_anti`). That is correct under the rule's H0,
   but it contradicts the documented claim.

Severity: **moderate-high** for item 1's central claim ("an empirically grounded mechanism for determining when
learned rewards are useful"). The mechanism decides when R_psi's *ranking agrees*, whether or not anything was
learned. Fix directions (not mine to make):

* control with a sign-symmetric population of unlearned rewards;
* require the learned reward to beat the agreement of its own frozen initialisation on the same pairs (a paired
  "learning helped" test);
* score within-state comparisons (candidates of the same decision, e.g. via the logged propensities) instead of
  consecutive units.

Reproduce:

    OMP_NUM_THREADS=4 nice -n 10 env PYTHONPATH=<freeze>/cam_live/src .venv/bin/python \
      cam_live/docs/verification/repro/g3core/probe_rw_flipped_frozen.py \
      --rw-dir <freeze>/cam_live/docs/evidence/RW --seeds 0 1 2 3 4 5 --out out/rw_flipped_frozen.json   # ~22 min CPU
    python analyse_rw_learned_vs_random_init.py --rw-summary <freeze>/cam_live/docs/evidence/RW/diag_summary_600.json \
      --flipped out/rw_flipped_frozen.json

Note: decisions are identical in both arms until the mix moves (`identical_before_first_change` true in all three
promoted seeds). The *pairing* of units can differ between processes in some seeds (unit ids are random), so a
negated arm's concordance is not always the exact negative of its frozen arm's. That reproducibility weakness has no
validity consequence, because the pairing never reads outcomes.

**D-RW2 — where the bootstrap ties at reward time, R_psi's weight can never exceed 0.25, whatever its quality.**

With `r_bootstrap` equal for every decision at reward time, `pair_scores` gives c_m = c_0 (R_psi alone) for every
m < 1. The lowering increment at every m_k < 1 is then `x+ = (c_0 - c_mk)/2 = 0` for every pair
(`validity.py:374`), so its e-value stays exactly 1. The raising increment is `-c_0/2 < 0` for a concordant R_psi.

Probe (`tied_boot_perfect_R_psi_300_pairs`): a **perfect** R_psi with a tied bootstrap ends at mix 0.75 in 50 of 50
runs, with e-value 1.0 at 0.75. In RW's diagnostics the bootstrap ties in every run: `concordance_bootstrap` is 0.000 in all 24 summaries, and every reward-time bootstrap value was exactly 0.0 in a traced run of the same configuration, and **all 7 promotions ended at
0.75** (RW.md §3.4 table). RW reports the 0.75 as a result and does not recognise it as structural.

The underlying design gap: the score is computed on the *reward-time* bootstrap. The mix weighs the *update-time*
bootstrap, which contains components that mature later, e.g. ST-3's information value (RW.md §2.4: "absent from the
bootstrap reward at reward time ... present in the live updates"). The evidence therefore cannot speak about what
lowering the mix further does to the live update. At full size the bootstrap does not tie (real run: bootstrap
concordance -0.06 over 17 pairs), so the cap is environment-dependent.

Severity: **moderate** (limits item 1's "learned rewards influence ordinary updates" to 25 % in such worlds).

### 1.3 Levels

* (a) holds (statistics, estimator, annotation, level fix).
* (b) holds.
* (c) In the integrated agent R_psi never had weight: mix 1 throughout the debug 120-decision run and the real
  35-decision run, with TV 0.0 on 11 real live updates, as RW reports. In the controlled diagnostic a promotion
  changes decisions 7-31 decisions later. For the promoted unlearned rewards the delay was 27 and 68 decisions
  (seeds 2 and 4: promotion at 151 / 99, first different decision 178 / 167). That meta-learning
  improved R_psi's agreement is shown only at the 30x reference step (D-RW1).
* (d) Improvement is not established (RW says so). Beyond RW's own caveats, D-RW1 shows that at the agent's step
  neither promotion nor post-promotion gains can be attributed to learning: an equally likely unlearned R_psi was
  promoted in 3 of 12 cases, with post-promotion effects of -0.77 to +1.04.

RW's negative-result diagnoses (return half-life smearing credit, evidence rate of tens of informative pairs) are
measured and plausible. I did not re-run `rw_credit_probe.py`.

## 2. IMP — predictive imagination and multistep planning (amendment item 2)

### 2.1 Checked

* **(a)** `tests/predict` 181 passed; all 16 IMP tests pass on the freeze.
* **(a) Transition priors** (`world.py:532-562`).
  * Fixed, seeded generator (`seed*7919+104729`, independent of the global RNG), zero-mean over members, added only
    to the residual mean, not trainable (non-persistent buffers).
  * Regenerated identically after a restore, so the members' learned cancellation stays valid.
  * Minor overstatement: "a transition trained without them predicts the same mean" holds at initialisation and on
    the training data, not off-data after training (the members learned to cancel their own offsets).
* **(a) Value-consistency horizon** (`imagination.HorizonController`, `planner.measure`). The criterion compares the
  candidates' values at the imagined and at the real belief, relative to how much the values differ between
  candidates. If the world model's values barely depend on the belief, the criterion admits the full horizon even
  though imagination carries nothing. That is what happened at full size: value error 0.040 against a persistence
  baseline of 0.038, giving horizon 4. The criterion therefore certifies that imagination is harmless, not that it
  is accurate about consequences. IMP states this ("harmless ... not informative") and records the baseline, so this
  is not a defect, but "the horizon grows only as measured accuracy supports it" overstates it.
* **(b)** `agent.py:1959-1969`: `planner.plan(..., predictor, evaluations)` -> `ev.extras.update(im.extras())` ->
  `select` -> `DecisionMaker.thompson_policy`. Calibration is fed by `_measure_planner` -> `refresh_calibration`.
* **(c) Can poorly calibrated imagination dominate?** Prior draws give a mean weight of 0.174. Over 20,000 draws,
  lambda_im = beta_im^+ / max(beta_q, 1) with beta_im ~ N(0, 0.5^2) and m winsorised at +-4, so under the prior an
  imagined preference can move a score by at most ~0.7 one-step sd on average, in half the Thompson samples.
  In B2 this changed the argmax in 9 % of decisions: exploratory, not dominant.
  With evidence that m carries nothing, the posterior pulls the weight *below* the prior (0.06-0.13 in my
  simulation), so discredited imagination loses influence as designed. The exception is D-IMP3.
* **(d)** No improvement. IMP reports this honestly (B2 +0.237 in lifetime 0, not surviving lifetime 1; §3.6 none;
  full-size pair: on worse, n = 1).

### 2.2 Defects and refutations

**D-IMP1 — the "paired" B2 ablation is not paired; the inference "the arms diverge, therefore imagination causally
changes decisions" (IMP.md §3.3) is refuted.**

`predict/imp_eval.py:10-15` says the arms "draw the same random numbers, so with the same seed they are identical
until behaviour differs". In the raw B2 rows (`docs/evidence/IMP/ablation_B2_final_code_raw.tar.xz`, lifetime 0,
**decision 0**: identical context, no learning yet) the arms' recorded inputs already differ. The one-step values
are up to 0.16 apart and the imagined continuations up to 4.69 apart (seed 7). The executed action differs at
decision 0 in **5 of 10** pairs.

If only imagination caused divergence, the expected count would be sum share_tv = 0.53, and
P(>= 5) = 5e-5 (Poisson-binomial). Seed 1 diverged at a decision where the on arm's distributions with and without
imagination were identical (share_tv = 0.000), which no coupling can attribute to imagination. Seed 7, IMP's example
("unlocked at its first decision ... achieved the objective in 7 decisions"), diverged where the two arms' imagined
continuations already differed by 4.69.

`repro/g3core/out/imp_b2_pairing.txt` (regenerate with `analyse_imp_b2_pairing.py`, §6). Consequences:

* The within-decision counterfactual (share_tv 0.053, argmax changed 9 %) remains valid causal evidence.
* The cross-arm divergence is not evidence.
* The B2 outcome CIs are effectively unpaired comparisons. The direction of IMP's (d) conclusion (no reliable
  improvement) is unaffected.

Likely cause: `imp_eval.run` builds the agent on `validation.OffsetClock` (`predict/imp_eval.py:117`;
`validation.py:47-57`: `time.time() + offset`), i.e. **wall-clock time**. Decision timestamps, elapsed-time features
(core, transition `time_features(dt)`, the `elapsed_s` signal) and deadline-dependent imagined sequences therefore
differ between the two processes. I did not run a same-arm-twice test to confirm this.

**D-IMP2 — the content target is not fixed: it moves with the LoRA adapters (refutes `predict/content.py:11-13` "a
fixed target, not one the learner can move" and IMP.md §6).**

`ContentTarget.__call__` (`content.py:79`) uses `model.embed_texts` = `SemanticBackbone.embed`
(`nn/backbone.py:605`). That embedding is computed through the LoRA adapters: its cache is invalidated when the
adapter content changes, `backbone.py:424-459`. The continual learner's slow step trains those adapters.

Probe `probe_imp_content_target.py` (debug model, rank-2 LoRA): adapter steps of absolute size 1e-3 / 1e-2 change the
targets of fixed texts by **2.2 % / 21.9 %** relative. There is no gradient path from the content loss into the
target (`requires_grad` False), so the learner cannot collapse it directly. However, every accepted slow step shifts
all targets, past and future. Content NLL / information-gain histories are therefore not comparable across slow steps
(relevant to IMP's observation that "IG rises with experience").

Fix direction (not mine to make): embed with the adapters disabled, or freeze a copy of the targets at record time.
Severity: **moderate** (documentation is wrong; non-stationary target).

**D-IMP3 (latent) — ImaginationCalibration treats overlapping windows as independent.**

`refresh_calibration` adds one observation per decision with target sum_{j=0..h} y(t+j) (`planner.py:285-290`).
Consecutive targets share h of their h + 1 terms, but `SelectionCalibration.posterior` (`policy.py:403-428`) is an
i.i.d. Bayesian regression.

Simulation with the real class (`probe_imp_overlapping_windows.py`, 200 runs, T = 100, true beta_im = 0): with h = 4
and an autocorrelated imagined preference (AR coefficient 0.9), the 95 % posterior interval excludes 0 in **18 %** of
runs, and the posterior is **confidently positive in 9.5 %** (nominal 2.5 %). The posterior sd is 1.5x too small.
With non-overlapping windows the shares are 3.5 % / 2.0 %. With an i.i.d. preference it is well calibrated
(conservative).

Measured on B2, the lag-1 autocorrelation of the chosen candidate's imagined preference is **0.06** (221 pairs), so
the weakness is **not active on the evaluated experience**. It would become active where imagined preferences
persist across decisions (long objectives with a slowly changing belief). Fix direction: one observation per
non-overlapping window, or an HAC / effective-sample-size correction.

## 3. PS — persistent internal representations (amendment item 3)

* **(a)** `tests/nn` 123 passed, 13 skipped (gpu / slow), including `test_ps_persist.py`:
  * decision-time belief equals the replayed belief to 1e-6, alone and batched;
  * finite differences through K = 2 recomputed transitions for the gate bias, rho and a write-attention weight;
  * the anchor is bit-constant beyond K;
  * the legacy restore sets rho = 0 and is bit-identical to the old model.
  I read `persist.py` resolve / `state_for` and agree with the definition: one function at decision and replay time,
  given equal weights.
* **(b)** The integration test passes (in `tests/nn`). `agent.py:784-790`: the chain is attached only when
  `persistent_state.enabled`, and **the default is `False`** (`nn/persist.py:76`). In the default runtime every core
  call is therefore still `state=None`, so the spec-1.3 requirement "recurrent or persistent latent state" is **not
  met by default**. It is one configuration line away.
* **(c)** As measured by PS: TV 0.0094 (debug) and 0.0038 (full size, 13 decisions) between the policy's
  distributions with and without the carried state, with b changed by 14-27 %. PS's methodological point is correct:
  on-line on / off outcome differences measure trajectory noise (`on_rho0_s0` vs `off_s0`).
* **(d)** No improvement: fixed-stream replay -0.005 +- 0.003 (28 debug pairs) and -0.006 +- 0.006 (9 full-size
  pairs). The one nominal effect (codec block, p = 0.05 after Bonferroni) is tiny.
  * "Default off with its evidence" is consistent with the amendment ("its contribution must be meaningful").
  * The diagnosis (an empirical limitation: the needed information is held by the window and the fast posterior) is
    plausible and is backed by the positive synthetic control (out-of-window accuracy 0.29 -> 0.80).
  * I found **no defect** in the mechanism.
  * Limitation: a run switched on from a pre-PS checkpoint starts at rho = 0 and opens slowly (0.004 in 40 debug
    decisions). This is documented.

## 4. CR — retrospective credit assignment (amendment item 4)

### 4.1 Checked

* **(a)** `tests/credit` 155 passed, 9 skipped, including `test_cr.py` (segment credit in an open objective,
  conservation < 1e-6 per segment, revision at closure, the revised queue surviving a checkpoint, the pooled rule).
* **(a) Pooled rule identity** (`select.py:189-252`). For a per-decision credit, d_i = ((e_i - y_i) - c_i) w_i with
  w = 1/p or -1/(1-p) and c_i independent of unit i's assignment gives
  E[d_i] = (e(T) - e(C)) - (y(T) - y(C)): the contrast bias at the unit's information state. The per-unit form is
  likewise correct. (CR.md's rendering "(e_i - y_i - c_i) w_i" is the same thing.) TOST at 90 % with >= 8 units.
* **(b)** CR-1 (`agent.py:805`, `segment_decisions` from config, default 16) and CR-3 (`agent.py:2483-2488`,
  `pop_revised` replay into `meta.policy_update`) are present.
* **(c)/(d)** The selection never had the 8 held-out units it needs: 0-2 per 64 debug decisions, and none at full size
  (no randomised experiment occurred). CR states this.

### 4.2 Defects and refutations

**D-CR1 — the forced-RUDDER ablation (-66.8 +- 26.0 adopters, CR.md §3.1) does not isolate RUDDER's credit. Part
of the effect comes from what the agent loses when the default changes. That loss would also occur under an
*evidence-based* promotion, so it is an integration defect.**

`CamLiveAgent._credit_for` (`agent.py:2272-2302`) returns the credit of the *default* estimator only. For a row
without that estimator's credit it returns None, and `PolicyImprovementLoss._goal_term` (`reward/inner.py:391-418`)
then falls back to the calibrated prediction. Simple credit (exact attributed per-decision outcomes) is not used as a
fallback.

In CR's own raw runs (`scratchpad/CR/agent/abl1/*_s1.json`, `policy_updates`):

* The `simple` arm uses simple credit on 2, 3, 3 and 6 rows of policy updates 0-3, and on 5-10 of 12 rows thereafter.
* The `rudder` arm has **0 credited rows in updates 0-3**. After that only rows inside matured segments are
  credited, mostly CR-3 replays (6 of 6-11). The batch also grows from 12 to 18 rows.

The arms therefore differ in three things: RUDDER credit on matured rows, removal of per-decision credit from every
not-yet-matured row, and batch composition. Paired by seed (n = 9, `out/cr_forced_rudder_by_quarter.txt`):

* quarter 1 (decisions 0-15; RUDDER credit first reaches the 5th policy update, at about decision 15): rudder - simple
  **-5.6 +- 3.6** (rudder better in 2 of 9). This is weak evidence on its own (1.6 SE), but its direction is the
  confound's;
* quarters 2-4: -9.0 +- 5.4, -24.6 +- 11.6, -27.6 +- 14.7.

The diagnosis "an empirical limitation of the amount of experience ... not a defect found in the credit code"
(CR.md §3.1 / §4.4) is therefore incomplete. If RUDDER were promoted by the selection, the agent would lose the
per-decision credit of all pending rows exactly as in the forced arm. Fix direction: fall back to the per-decision
credit for rows the default estimator has not credited yet, then re-run the ablation.

Caveat (CR's own): runs of one seed are not reproducible (random ids), so the pairing reduces little variance.

**D-CR2 — the selection validates a different quantity than the one the agent trains on.**

`credit/providers.py:53-56`: `VALIDATION_KEY[rudder] = "rudder_contribution"`, while
`LEARNING_KEY[rudder] = "rudder"`. The `rudder` value is the signal (G_segment - V(prefix)) / gamma
(`rudder.py:220`): the whole segment's return minus the prefix value, unbiased for any decomposer but with the
variance of every other decision's outcome. A promotion therefore installs a learning signal whose quality the
selection never tested. CR notes that the contribution "cannot be used today" but does not flag that the
"estimand-matched" selection is mismatched to the training signal. Severity: **moderate** (latent: nothing has been
promoted).

**Not a defect:** COCOA refusing open-ended backbone candidates at full size (spec 9.3) is correct behaviour, but it
means COCOA contributes nothing on the real configuration.

### 4.3 Levels

* (a) holds.
* (b) holds: segment credit on 28 of 39 real-config records in an open objective; CR-3 replayed 41 rows.
* (c) Revised credit reaches the policy update (measured by CR). Selection never acted.
* (d) None in the agent. The negative result is real, but its diagnosis misses D-CR1. The forced-COCOA arm
  (-38.2, n = 6) has the same confound.
* The controlled diagnostic (§3.2: return-based credit learns a delayed-only consequence that simple attribution
  cannot) is sound.

## 5. Test runs (freeze copy, CPU)

| run | result | wall |
|---|---|---|
| `tests/reward` | 169 passed, 3 skipped | 191 s |
| `tests/predict` | 181 passed, **1 failed**, 10 skipped | 228 s |
| `tests/credit` | 155 passed, 9 skipped | 181 s |
| `tests/nn` | 123 passed, 13 skipped | 102 s |
| `test_int3a_eligibility.py::test_a_record_checked_before_its_verdict_becomes_eligible_for_the_evaluator` alone | **1 failed** (0 of 12 records eligible) | 32 s |
| `test_int3a_eligibility.py` (whole file) | 2 passed | 52 s |
| `tests/integration/test_agent_runtime.py` (the 4 acceptance tests: `test_behaviour_changes_and_lives_in_the_weights`, `test_delayed_outcomes_link_to_their_decisions`, `test_internal_rewards_are_stored_per_record`, `test_meta_update_runs_with_a_nonzero_meta_gradient`) | 4 passed | 294 s |
| `-m slow`: `credit/test_evidence.py` (5 tests, incl. delayed unlock credited retrospectively, stored experience improves held-out prediction, forgetting measured / reduced), `reward/test_reward_experiment.py::test_three_learners_...`, `predict/test_learning_evidence.py::test_value_of_information_fast_path_...`, `credit/test_cr_agent.py` (CR's real agent + runtime test) | 8 passed | 654 s |
| `-m gpu` acceptance tests | see §7 | - |

The eligibility test fails deterministically when run alone and passes when its file runs whole, so its outcome
depends on what ran before it in the process (RNG or global state). It was in `tests/predict` before the amendment
(INT3A, G2L-8). These runs do not show whether the amendment changed it. It is outside items 1-4 (evaluator
eligibility, INT3A), and I report it rather than diagnose it.

## 5b. Verdicts (`docs/verification/verdicts_g3core.json`, commit 6cba496, supersedes all)

| status | ids |
|---|---|
| verified | MR-01, MR-02, MR-04, MR-05, MR-06, MR-07, MR-08, MR-09, MR-10, MR-11, PI-11, NA-06, RC-01..RC-07, CL-01, CL-02, CL-04, CL-05 |
| partial | MR-03 (R_psi has zero weight in every integrated or real run; D-RW1, D-RW2), PI-07 (D-IMP1, no improvement, imagination does not distinguish real candidates), SE-06 (D-IMP2, content IG unused), RC-08 (no evidence-based selection ever ran; D-CR1, D-CR2), CL-03 and CL-10 (their gpu acceptance tests were not re-run; see §7) |

The verdicts judge the manifest requirement as titled, under the amended evidence standard. "verified" means the
requirement's mechanism holds on the freeze (acceptance tests re-run, plus my own checks). It does not mean the
amendment's improvement goal for that capability is met: none of items 1-4 shows a measured improvement.

## 6. Reproduction

All with `S=<scratch>/G3CORE`, `F=$S/freeze` (= `git archive 6cba496`), `OMP_NUM_THREADS<=4 nice -n 10`,
`CUDA_VISIBLE_DEVICES=`, `PYTHONPATH=$F/cam_live/src`, from `cam_live/docs/verification/repro/g3core`:

```
python probe_rw_validity_procedure.py --reps 200 --pairs 1000 --out out/rw_validity_procedure.json
python probe_rw_flipped_frozen.py --rw-dir $F/cam_live/docs/evidence/RW --seeds 0 1 2 3 4 5 --out out/rw_flipped_frozen.json
python analyse_rw_learned_vs_random_init.py --rw-summary $F/cam_live/docs/evidence/RW/diag_summary_600.json --flipped out/rw_flipped_frozen.json
python -I $F/cam_live/docs/evidence/RW/promotion_power_sim.py > out/rw_promotion_power_rerun.json
python probe_imp_overlapping_windows.py --reps 200 --out out/imp_overlapping_windows.json
python probe_imp_content_target.py --out out/imp_content_target.json
python analyse_imp_b2_pairing.py --raw <dir with B2/*.pjson extracted from docs/evidence/IMP/ablation_B2_final_code_raw.tar.xz>
python analyse_cr_forced_rudder.py --cr-agent <scratch>/CR/agent
```

## 7. Not checked / limits of this verification

* **No GPU hold.** I did not re-run the gpu-marked acceptance tests (`test_meta_update_cost_at_the_50m_preset_on_the_gpu`,
  `test_real_configuration_one_frequent_and_one_slow_adapter_update`, `test_checkpoint_of_a_gpu_model_is_stored_on_cpu`,
  `test_reworded_goals_...`) or any real-model claim: RW §5, IMP §3.2 / §3.5, PS §4.4 full size and CR §5 are taken
  from the implementers' evidence files. D-IMP2's size at full size (Qwen3-4B rank-r LoRA) was not measured.
* I did not re-run the long ablations (CR's integrated runs take ~40 min each, IMP's B2 batch, PS's replays). My
  findings on them come from their raw outputs.
* The exact cause of the B2 decision-0 differences (D-IMP1) is not pinned down: load and the wall clock, or an
  unrecorded `PYTHONHASHSEED`. The arm switch itself was shown not to cause them. The cause of the process-dependent validity pairing noted under D-RW1 was not found.
* RW's `rw_credit_probe.py` (return-horizon diagnosis) and IMP's content prequential diagnostic were not re-run.
