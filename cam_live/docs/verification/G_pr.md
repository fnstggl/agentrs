# G-pr — independent verification of prediction / decisions (B) and the learned intrinsic reward (C)

Verifier: G, lens `pr`. Scope: PI-01..PI-10, SE-01..SE-06, AE-01, CU-12 (workstream B, `src/cam_live/predict/`) and
MR-01..MR-11, AE-04 (workstream C, `src/cam_live/reward/`). Verdicts: `verdicts_pr.json`. Probes:
`repro/pr/*.py` (each prints `[PASS]` when the claimed property holds, `[FAIL]` when it is refuted, exits 1 on any
refutation). All commands from `/workspace/camlive`. No source file or test of another workstream was modified.

**Code under test.** `predict/` and `reward/` are unchanged in the working tree relative to HEAD (`9ba1e45`), but other
agents had uncommitted edits elsewhere (`config.py` changed the debug/50m `d_ff`, `contracts.py`, `diagnostics/*`,
`learning_api.py`, `nn/*`). Those edits change every numeric result (they change model initialisation), so every
number below was re-run on a read-only snapshot of the committed tree:
`git archive HEAD cam_live/src cam_live/tests cam_live/pyproject.toml cam_live/configs | tar -x -C $SNAP` and
`G_PR_SRC=$SNAP/cam_live/src .venv/bin/python cam_live/docs/verification/repro/pr/<probe>.py`. The properties below
hold on both trees; the numbers in B's and C's evidence reproduce exactly only on the committed tree.

## 1. What was executed

| What | Command | Result |
|---|---|---|
| B default suite | `.venv/bin/python -m pytest cam_live/tests/predict -m "not gpu and not slow" -q --rootdir=cam_live` | 142 passed (81 s); same on the HEAD snapshot |
| B measured numbers | the same on the snapshot with `CAM_LIVE_B_MEASURE=<scratch>/B_measured_mine.json` | **86 / 86 values equal** `docs/evidence/B/measured.json` |
| C default suite | `.venv/bin/python -m pytest cam_live/tests/reward -q --rootdir=cam_live` | 115 passed, 2 skipped (slow, gpu), 84 s |
| Independent meta-gradient, small configuration | `repro/pr/probe_metagrad_small.py` | 40 / 40 |
| Independent meta-gradient, real adaptable subset, K = 1, 2, 4, 8 | `repro/pr/probe_metagrad_real.py` | 22 / 22 |
| MetaLearner end to end (applied step, detached reward, cold start, missing propensities, return-to-go) | `repro/pr/probe_metalearner_path.py` | 12 / 13 — **D1** |
| Estimator support, integrity attacks, ledger | `repro/pr/probe_integrity.py` | 18 / 21 — **D2, D3, D6** (+ D8 info) |
| Likelihoods vs scipy, masks, calibration, registry, evaluator | `repro/pr/probe_predict_numerics.py` | 30 / 31 — **D5** |
| Decision rules vs empirical frequencies, experiments, VOI references | `repro/pr/probe_decisions.py` | 28 / 28 |
| Action quality, goal-free budget | `repro/pr/probe_extras.py` | 2 / 4 — **D4** |
| Real Qwen3-4B (under `gpu_lock`, model load 9 s, < 15 min hold) | `repro/pr/probe_real_model.py` | 12 / 13 — D9 |
| C numbers re-run (real-subset FD check, tabular exact meta-gradient) | `G_PR_SRC=$SNAP/... repro/pr/probe_reproduce_numbers.py` | 14 / 14, numbers equal |
| C LabWorld experiment, reduced (learned + bootstrap arms, seeds 11 and 14, first 80 of 320 lifetimes) | `G_PR_SRC=$SNAP/... repro/pr/probe_labworld_reduced.py --lifetimes 80` | 6 / 6: 80/80 lifetimes bit-identical to the stored runs in all four arms; seed-11 collapse reproduced |

## 2. Claims that survived the attack

**Meta-gradient (spec Part 7) — the core claim holds.** My own implementation (nested `torch.func.grad`; inner loss
written from the documented formula: discounted standardised returns of R_psi inside one lifetime, advantage against
V evaluated on theta_k held fixed only for the inner derivative, value loss, entropy, per-prefix norm clip; outer J =
exact expected utility of pi_{theta_K(psi)} over *all* feasible candidates of the query decisions) agrees with central
finite differences on all 73 psi coordinates of a small configuration (rel. error <= 3.3e-9, K = 1, 2, 4, 8) and on 43
sampled psi coordinates plus a random direction of the real debug-model adaptable subset (62,211 theta, 30,049 psi;
rel. error <= 1.6e-7, directional agreement to 7 digits, **K = 8 included**, which C had not checked on the real
model). C's engine with C's own `PolicyImprovementLoss`/`policy_gradient_inner_loss` equals my gradient to ~1e-15 in
`unroll` and `checkpoint` mode, through both the importance estimator (enumerated query, logged uniform behaviour) and
the on-policy estimator; its reported objective equals the exact J to 1e-12. The inner-data score term is exact: the
expectation of C's estimator over all 9 on-policy support samples equals the gradient of the enumerated expected
objective (rel. 2e-15), and dropping the term is biased (13 %). The path is never detached: every psi tensor receives
gradient; with R_psi detached inside the inner loss the meta-gradient is exactly 0 (small, real subset, and on the
real `MetaLearner.meta_update` path); a detached theta_K carries no graph. C's stage-1 bug (plain `v.detach()`
baseline) is real: it is exact at K = 1 and biased by 5-13 % (small) / 1.0 % and 3.3 % (real, K = 2 / 4) beyond.
`MetaLearner.policy_update` equals my K = 2 SGD on the documented loss (|diff| 1.1e-7 vs update 1.7e-2) and leaves psi
bit-identical; the step `meta_update` applies equals my recomputation from its episodes (rel. 1.8e-7); skipped updates
leave psi bit-identical and are recorded with a reason; query decisions without logged propensities never supervise.

**Importance weighting.** Unbiased with logged behaviour propensities (Monte Carlo, 4000 reps, max |z| 2.2), visibly
biased when wrong propensities are supplied (sensitivity), and refuses — no gradient, psi bit-identical after
`apply` — for missing, invalid, below-floor propensities, too small an ESS and an unverified row. On the real path the
behaviour log-propensities are log(`decision.probabilities[chosen]`) of the stored decisions (float32 precision).

**Integrity that holds.** Bounded output (inflating the pre-bound output x1000 stays within the bound); r -> 50 r
leaves the policy update unchanged (|diff| 3e-8); replayed signed outcomes rejected as duplicates; tampered values,
unsigned, invented and prediction-sourced outcomes rejected; editing the ledger file is detected on load.

**B numerics.** Every family's log-likelihood equals scipy (normal, Student-t, Poisson with exposure, negative binomial
incl. the Poisson limit, Bernoulli incl. logit 800, categorical with per-row category counts) to <= 4e-14; the discrete
hazard equals the geometric law for events and the survival of completed bins under censoring; pending, unobservable,
missing and censored-count outcomes are masked (never filled), imagined observations raise. Calibration metrics are
correct on synthetic data with known calibration (true slope 0.7 / intercept 0.3 inside the intervals, Brier and NLL
equal my numpy, reliability within 0.013 of the known curve). The registry refuses edits, future-dated predictions,
resolutions at or before registration and second resolutions, and detects log edits. Evaluator: success loss = BCE
with logits over the prefixes before evaluation; pending -> zero loss and gradient; unsigned refused.

**Decisions.** For policy, greedy, UCB, VOI-weighted and Thompson (posterior samples and normal marginals) the recorded
vector is the one sampled from (chi-square over 20,000 draws, p = 0.055-0.93), the infeasible candidate gets exactly 0,
the eps floor holds, the mean Thompson propensity from normal marginals is the true P(best) within 0.01. Randomised
experiments are registered (question, hypothesis, prediction copies) before the first assignment and the assignment
frequency matches the recorded 0.7 within 4 se. VOI references: exact E[max affine] and the knowledge gradient match
Monte Carlo; a decision-irrelevant observation with large surprise has VOI exactly 0.

**Not a frozen-LLM wrapper (real Qwen3-4B, committed code).** Perturbing only the experiential group (50,953,765
parameters; adapters, heads and reward hashes unchanged) changes the registered predictions (0.15 % / 7.6 % for 5 % /
50 % of each tensor's std) and the decisions (Thompson TV 0.061 at 5 %; greedy and UCB argmax flip, TV 0.98, at 50 %;
the policy-rule TV is only 8e-5 / 8.5e-4 because the policy head is near-uniform at initialisation). Predictions are
registered with made_at == clock before `evaluate_candidates` returns. The proposer returned 4 schema-valid backbone
actions merged with 13 schema-walker candidates, each with its provenance.

**Reproduced numbers.** (1) C real-subset FD check: rel. error 9.52e-6 / 9.56e-6 / 9.63e-6 and |g| 4.3141e-3 /
8.4536e-3 / 1.6380e-2 for K = 1 / 2 / 4, plain-detach bias 0.99 % / 3.31 % — identical to `docs/evidence/C/
diagnostics.json`. (2) C tabular exact meta-gradient: J and FD agreement for K = 1, 2, 4, 8 identical to
`exact_meta_gradient.json`. (3) B: all 86 measured values of `docs/evidence/B/measured.json`. (4) C's LabWorld
experiment: the first 80 lifetimes of learned and bootstrap on seeds 11 and 14 are bit-identical to the stored runs;
seed 11 collapses (learned 0.05 vs bootstrap 0.60 in the last quarter, policy on the probe 96 %); C's re-summarised
last-quarter means from the stored runs (0.472 / 0.263 / 0.445 / 0.471) equal the report. C's negative result is
reported honestly.

## 3. Defects

| id | sev. | where | defect | repro |
|---|---|---|---|---|
| D1 | major | `reward/metalearner.py:419-424` with `:497-499` | With `utility="return_to_go"` (the default) and verifier measurements as utilities, U_j of an earlier row j of the same lifetime contains the discounted measured outcome of row a, and the leave-group-out baseline of row a averages those U_j: the baseline depends on the row's own outcome (d b_a / d m_a = 0.28 at row 2 of 8; about (#earlier rows)/(N-1), i.e. the last decision of an 8-row lifetime loses almost all credit for its own outcome). The outer policy-gradient is biased in exactly the single-lifetime live setting. Not exercised by C's LabWorld experiment (objective utilities, one group per lifetime). | `.venv/bin/python cam_live/docs/verification/repro/pr/probe_metalearner_path.py` (last check) |
| D2 | major | `reward/metalearner.py:318-331`, `reward/integrity.py:291-301` | `admit_objective` / `admit_verified` accept a caller-chosen `utility` (and `exp_ids`) for any validly signed outcome; the ledger stores the number unauthenticated. A signed value of 14 was admitted with utility 1e6. "Utilities come only from signed outcomes" holds for the presence of a signature, not for the number that supervises psi. | `probe_integrity.py` ("validly signed outcome ... utility") |
| D3 | major | `reward/integrity.py:187-194`, `reward/metalearner.py:320` | Exclusion of self-generated goals depends on the caller argument `objective_origin` (default OPERATOR); `VerifiedOutcome` carries no origin and `LabVerifier.verify` signs any task, so a verifier-signed outcome of a self-generated goal supervises the outer objective by default. | `probe_integrity.py` ("self-generated goal") |
| D4 | major | `reward/selfgoals.py:52-55, 114-125` | Goal-free budget not enforced for selected-but-unresolved goals: `select(k=5)` returns 5 goals under a 2-decision budget; six selections while goals are pending return 6 (unbounded). The budget is charged only in `record_result`/`resolve`. | `probe_extras.py` |
| D5 | major | `predict/calibration.py:203` (`binary_report` -> `logistic_recalibration`, `:95-133`) | Calibration slope/intercept intervals ignore clustering while Brier/NLL use clusters. Clustered null (40 lifetimes x 16 forecasts sharing one outcome, true slope 0): the 95 % interval excludes 0 in 61 % of runs. B's reported "slope 254.16 (CI 122.43..385.90)" is not a valid interval. | `probe_predict_numerics.py` ("clustered null") |
| D6 | minor | `reward/integrity.py:374-377` | `SelfGoalDiscount.key` folds only case, punctuation and digits: "Get ten members" / "obtain 10 members" are not discounted as repeats of "Reach 10 members". | `probe_integrity.py` |
| D7 | minor | `reward/metalearner.py:524-533` | Cold start (mix 1): the shadow meta-update switches to the on-policy estimator whenever |pi_K - mu| <= `on_policy_tol` (1e-3) although the shadow policy did not act (docs/C/API.md says the query must use importance); estimator observed: `mixed_on_policy_and_importance`. Bias bounded by 1e-3/mu per row. | `probe_metalearner_path.py` (info line) |
| D8 | minor | `reward/integrity.py:326-349` | A ledger file truncated at an entry boundary reloads with a valid chain; removal of the newest outcomes is visible only against a head stored elsewhere (checkpoint). | `probe_integrity.py` (info line) |
| D9 | minor | `predict/candidates.py:778` via `nn/backbone.py` bridge | At initialisation the proposer is not conditioned on the belief in effect: log p(completion | prompt, b) is bit-identical for two beliefs (zero bridge output); the path exists (|d log p| = 0.87 after a bridge perturbation) and only D's consequence provider (`credit/providers.py:632`) can train it. Documented by B; "conditioned on the latent" is structural until then. | `probe_real_model.py` |
| D10 | minor | `predict/calibration.py:306` | `score_predictions` NLL intervals are not clustered. | code reading |
| D11 | minor | `predict/world.py:138` | `build_targets` stores targets in float32 (precision loss for large counts / precise real values). | `probe_predict_numerics.py` (info) |
| D12 | minor | `reward/bootstrap.py:182-183` | Label "learned" is attached at mix 0 before any meta-update, i.e. to a randomly initialised R_psi. | code reading |

## 4. Claims not supported or not shown

* C.json MR-06 "Utilities come only from the hash-chained ledger: signed VerifiedOutcome records" — the signed record
  must exist, the utility value is free (D2); self-authored objectives are rejected only when the caller says so (D3).
* C.json AE-04 "budgeted" — refuted for pending goals (D4).
* B.json / measured: "calibration slope 254.16 (CI 122.43..385.90)" — interval invalid under clustering (D5).
* No claim of B or C about *learning* on the real Qwen3-4B configuration exists; none was made, none was checked.
  Everything learned is debug scale in LabWorld. Neither package is called by the runtime for rewards/meta-learning.
* Not independently re-measured: B's imagination accuracy (PI-07), VOI closed loop, evaluator calibration numbers
  beyond the bit-reproduced measured values; C's 50m GPU cost numbers.
