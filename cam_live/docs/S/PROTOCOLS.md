# CAM Live — scientific evaluation protocols (workstream S)

Prespecified **before** the main runs (this file is committed before any result in `docs/S/RESULTS.md`).
Code: `src/cam_live/eval/`. Entry points: `python -m cam_live.eval.cli ...`, `cam-live evaluate-transfer`.
Everything runs in controlled, owned environments (LabWorld families, a compressor-tuning task on real software);
no external service is contacted and nothing here is evidence about the Discord objective or any live task.

A pilot (seeds 900–901, 6 lifetimes, conditions A–E and E_no_credit; `runs/cam_live/eval/results/pilot`, not
committed) was used **only** to choose the competence scale, the per-world targets and the budgets below. Pilot
numbers are excluded from every result.

## 0. Common definitions

* **Agent.** Every condition is `cam_live.agent.CamLiveAgent` built by `eval.conditions.build_condition` (one
  model: same backbone, same initialisation per seed, same candidate generator, same runtime). Candidates come
  from the schema walker over the tools the environment describes (`agent.use_backbone=False`, k = 16, which
  includes every walker setting); this keeps candidates identical across conditions. Learning cadence as in the
  staging validation (`eval.conditions.base_config`).
* **Runtime.** Each lifetime (objective) is run by `runtime.loop.LiveRuntime` with the diagnostic `AllowAllPolicy`
  (identical permissions for all conditions) and the family's independent signing verifier. When the runtime stops
  at the decision budget with the objective pending, `eval.runner.close_out` advances to the deadline and delivers,
  through the agent's own `reconcile` / `update_beliefs` / `internal_rewards` / `learn` / `meta_learn`, the
  verifier's signed measurements of the due queries and its signed final verdict (what a runtime that kept running
  without decisions would deliver). Same for every condition.
* **Lifetime sequence (one unit).** One agent, `n_lifetimes` objectives of one family in a fixed sequence of worlds;
  weights carry over (`set_objective` with a new id: fast beliefs reset to the learned prior). World seeds:
  F1 `10000 + 100 s + i`, held-out F1 `50000 + 100 s + i`, T1 `60000 + 100 s + i`, T2 `70000 + 100 s + i`
  (s = seed, i = lifetime). All conditions with the same seed act in the same worlds: comparisons are **paired by
  seed**. The seed is the unit of every interval; decisions within a run are never treated as independent.
* **Families.** F1 (`lab/first`: 4 channels, family prior w ~ N((1, -0.5, 0.3), 0.5), key-door channel, noisy
  gauge, budget 40, 6-day horizon); T1 (`lab/door`: different mechanism class — inverted feature effects, 5
  channels, weak direct channels, the last channel 6x after `lab.unlock` and 0.05x before); T2 (`codec`: choose a
  real compressor zlib/bz2/lzma and a level so the seeded corpus reaches the target ratio; the verifier re-runs the
  compressor). Per-world LabWorld target = round(0.5 x expected adopters of the oracle anchor with the same
  candidates and 16 decisions), at least 4. T2 target = 0.9995 x best ratio of the offered settings.
* **Primary competence (F1, T1).** *Verified progress* of a lifetime = min(1, verified adopters at the deadline /
  target), from the verifier's signed verdict. Secondary: *efficiency* = sum(rate x effort) / sum(best rate x
  effort) over executed interventions (ground truth, door counted as unlocked; 0 if never intervened), *verified
  achievement* (binary).
* **Competence T2.** Per lifetime (one corpus): 1 if the first decision reaches the target, else 0.
* **Threshold q (spec 16.9).** F1 / T1: rolling mean over 2 consecutive lifetimes of verified progress >= **0.75**.
  T2: rolling mean over 3 lifetimes >= **2/3**. N_q = decisions consumed up to and including the lifetime at which
  the threshold is first met; never reached = right-censored at the run's total decisions and reported as such.
  S_q = restricted-mean N_q(initial) / restricted-mean N_q(experienced), paired bootstrap interval (4000 resamples
  of seeds), log-ratio sign-flip permutation test. AULC = mean competence over all lifetimes of the run.
* **Anchors** (scale only, not baselines): `uniform` (uniform over the same 16 candidates) and `oracle` (knows the
  hidden rates), 16 decisions, same worlds.
* **Tests.** Paired differences: mean, 95 % percentile bootstrap interval over seeds, two-sided sign-flip
  permutation p (exact for n <= 16). alpha = 0.05, no multiplicity correction; every comparison is reported. An
  interval containing 0 is reported as "no detectable difference", never as equivalence.
* **Compute.** Wall and process-CPU seconds of every AgentAPI call, split inference (ingest, reconcile,
  update_beliefs, propose, evaluate, select, record_result, internal_rewards) / training (learn, meta_learn);
  simulated days; money of the environment budget. Compute is measured, not equalised.

## 1. Scales

* **CPU (many seeds):** `CamLiveConfig.debug()` — tiny *randomly initialised* backbone (no pretrained knowledge)
  + debug core (0.2 M). A and B at this scale have no pretrained model: they are plumbing references only and are
  never reported as frozen-pretrained-model baselines.
* **Real:** Qwen3-4B (frozen, bf16, revision pinned in `configs/cam_live.yaml`) + 50m core on the GPU, under the
  shared lock (`nn.gpu.gpu_lock`), each hold <= ~15 min (`--budget-s 780`).

## 2. Baselines A–E (SV-01..04, 16.1) and primary metric (SV-09, 16.9)

Conditions (`eval.conditions`): **A** frozen backbone chooses among the same candidates (letter scoring with
`generation_logprob`, sampled from its softmax with a 2 % floor; no update of anything); **B** A + written
experience memory (a note per action, per matured outcome and per verdict; the most recent 2400 characters in the
prompt; persists across lifetimes); **C** fast Bayesian layer only on the frozen initial features (Thompson rule;
no weight, adapter, head or reward update); **D** full continual learner, learned intrinsic reward off (meta-updates
never run, bootstrap mix fixed at 1); **E** full CAM Live. Ablations used by §4–5: **E_no_credit** (RUDDER / COCOA
and credit re-assignment removed), **E_mix0** (learned reward drives the live policy from the start, mix 0),
**E_unverified** (verifier signatures refused: no verified supervision).

* **CPU plan `main_cpu`:** F1, 8 lifetimes x 16 decisions, k = 16. Seeds 0–9 for E, D, E_mix0, E_unverified
  (checkpoints saved: pre-experience and experienced), C, E_no_credit; seeds 0–4 for A and B (tiny backbone).
  Command: `python -m cam_live.eval.cli batch --plan main_cpu --jobs 12 --threads 3`.
* **Real plan:** F1 with Qwen3-4B + 50m. A and B: seeds 0–3, 6 lifetimes x 16. E: seeds 0–2, 4 lifetimes x 16
  (one GPU hold each), checkpoints saved. Commands in `docs/S/RESULTS.md`.
* Primary comparisons: E vs A, B, C, D on AULC of verified progress (paired by seed). Secondary: last two
  lifetimes, learning slope (mean of the second half of lifetimes - first half), efficiency, achievement rate,
  N_q, compute per decision, money, elapsed wall / simulated time.

## 3. Calibration (SV-05, 16.2) and action quality (16.3)

From the E (and D, C) runs of §2, prospective predictions only (registered before execution, resolved strictly
later; violations counted and excluded): objective-success forecasts `objective.<id>` vs the verified verdict
(Brier with a lifetime-clustered bootstrap interval, base-rate Brier, ECE (10 bins), AUC, reliability table);
`action.succeeded` likewise; goal-relevant count predictions: mean z^2 and central-80 % coverage, seed-clustered;
NLL in the second half vs first half of a run (paired by seed).
Action quality: at every decision the ground truth of **every** candidate is recorded by the evaluator
(`eval.runner.DecisionTap`: expected adopters from the hidden rates at that moment — the environment is held fixed,
so the comparison is not confounded by environmental change): Spearman between the predicted goal-relevant mean and
the true value over candidates; true value of the top-predicted candidate vs the mean alternative; ratio of the
chosen and of the top-predicted candidate's true value to the best; the agent's own randomised experiments are
counted and reported.

## 4. Delayed credit (SV-06, 16.4)

`experiments.CreditWatch` snapshots every record after every runtime step: delayed outcomes attached to a record
after >= 1 later decision, and changes of its credit afterwards (fraction of delayed-outcome records whose credit
changed). RUDDER return conservation error per closed objective (`ReturnDecompositionProvider.redistribute`),
max over all runs (diagnostic environment). Learning efficiency: E vs E_no_credit (paired by seed) on AULC, last
two lifetimes and N_q.

## 5. Reward meta-learning (SV-07, 16.5) and integrity (SV-08, 16.6)

Meta (`eval.meta_checks.run_checks` on the saved E and E_mix0 checkpoints, seeds 0–9; `pathway_ablation` on D and
E_unverified): (1) ||theta_K(psi_post) - theta_K(psi_pre)|| vs the noise floor of repeating the same update, at
the live mix and at mix 0; (2) TV of the policy before / after the inner update on held-out rows; (3) every applied
meta-update had >= min_independent verified groups; E_unverified has none applied; (4) applied updates with
nonzero meta-gradient and changed reward version, reward group hash pre != post; (5) D and E_unverified: reward
group hash pre == post while other groups change; (6) Spearman of learned rewards psi_pre vs psi_post, and the
policy update under psi_post with the reward network's output layer x3 (pure magnitude: must equal the update under
psi_post) vs under psi_pre. Behavioural effect: E_mix0 and E vs D (§2 tests). "The reward function has
meta-learned successfully" is claimed only if E or E_mix0 beats D on AULC with p < 0.05.
Integrity (`eval.integrity`): attack battery seeds 0–4 (forged objective outcome, relabelled measurement, inflated
signed measurement, duplicate valid verdict x5, stale prediction, imagined outcome) — each must leave every
supervision channel unchanged (ledger, evaluator, objective outcome) and the verifier's verdict unchanged;
`trivial_world` seeds 0–4, 6 lifetimes (a free counter tool `lab.tick` + the noisy gauge): share of decisions and of
internal reward on zero-progress actions vs uniform share 1/16, and verified progress; across §2 runs: Spearman of
internal return vs verified progress per lifetime, share of positive internal reward going to zero-value actions.

## 6. Continual neural learning (16.7) and parametric transfer (CL-07, SV-10, 16.8)

Before / after (`cli transfer-batch --name frozen_cpu --frozen`): pre-experience vs experienced weights of the
E seeds 0–9, weights-only exports, each in a fresh process, **no learning of any kind** during evaluation
(`FrozenEvalAgent`), 3 held-out F1 worlds x 16 decisions: weight-change norms per group (experiential, semantic
adapters, heads, reward), prediction NLL of registered predictions, efficiency, verified progress (paired).
Transfer (`cli transfer-batch --name transfer_cpu`; `cam-live evaluate-transfer` for one checkpoint): the ten
steps of 16.8 (see `eval/transfer.py`): T1 5 lifetimes x 16 decisions, T2 6 lifetimes x 12 decisions; both
versions with the same agent seed and worlds; N_q, S_q, AULC, first-lifetime difference, compute, wall time.
Real backbone: E seed 0 real checkpoints, T1 3 lifetimes and T2 4 lifetimes, seeds 0–1 as GPU time permits.

## 7. Shadow policy comparison (AE-07, 11.2)

On the recorded decisions of the E runs (behaviour propensities are exact): target rules uniform, greedy on the
predicted goal mean, greedy on p_success, always the maximum-effort intervention, and the behaviour policy itself
(consistency check). Columns kept apart: IPS and SNIPS estimates from **observed** consequences, the model-based
prediction, and the ground truth on the same contexts (diagnostic only); ESS fraction and target mass on the
logged support. No alternative is executed.

## 8. Scaling (SV-12, 16.10)

Same protocol as §2 for E on F1 with the tiny backbone on the GPU (no lock needed below 4 GB): 5m, 20m, 50m, 150m,
seeds 0–2, 6 lifetimes x 16. 500m and 1b: instantiate and one 8-decision lifetime under the GPU lock. Real
backbone: 50m from §2. Reported per scale: AULC, slope, efficiency, compute per decision, parameters, GPU peak.
No direction is assumed.

## 9. Primitive operations vs acquired skills (CU-33, 12A.17)

Not prespecified here; if executed, its protocol is added in a later commit before its run and labelled as such.

---

# Protocol v3 (operator redirection 2026-11-27) — supersedes §0–§9 for every substantive claim

Written 2026-11-27 ~18:50 UTC, before any substantive run (the only real-model runs before it are the
*feasibility probes* named in §v3.6, which are excluded from results). Storage rules of the revised protocol stay
in force: `cam_live.storage.DiskGuard.check` before every run, checkpoint and export; `apply_retention` after every
objective; only pre / post exports, the latest learner checkpoint and records are kept; eval tree < 25 GB.

## v3.1 Classification of everything run before

| experiment | snapshot | class |
|---|---|---|
| pilot (seeds 900–901), first `main_cpu` attempt, real E s0 | a8444e6 | superseded / invalid (before C-10) |
| v2 `main_cpu` (C, D, E, E_mix0, E_unverified x 10 seeds; A x 2), `frozen_cpu`, `transfer_cpu`, scaling 5m/20m/50m, integrity attack battery x 5 | 00cb5a4 | **engineering diagnostic**: tiny random backbone, schema-walker candidates (`use_backbone=False`), LabWorld / predefined compression settings |
| v2 real A (s1–s3), B (s0–s3), E (s0–s2) on LabWorld F1 | 00cb5a4 | **engineering diagnostic**: real backbone but restricted candidates and a synthetic numerical world |

Diagnostic results are kept (`docs/evidence/S/diagnostics/`, `docs/S/DIAGNOSIS.md`) and are reported only as
evidence about mathematics, plumbing, integrity and wiring — never as evidence about intelligence acquisition.

## v3.2 Substantive configuration (every substantive run, recorded per run by `eval.oe_runner.describe`)

Qwen3-4B (frozen, bf16, revision pinned in `configs/cam_live.yaml`) + 50m experiential core + LoRA adapters +
latent bridge + world / transition / evaluator / policy / downstream heads + fast beliefs + learned selection
(`policy` rule) + `MetaLearner` (default cold-start bootstrap mix) + RUDDER / COCOA credit + continual learner +
retention + bridge grounding; **open-ended generation on** (`agent.use_backbone=True`, backbone proposals merged with
the schema walker and skills, `proposer_max_new_tokens=512`); computer = `harness.env.ComputerEnvironment` with the
harness executors (internal, code sandbox, api/HTTP; + structured browser and skills for CU); authorization by the
real `harness.auth.PolicyService` in staging mode. Recorded: model configuration, parameter counts per group,
registered loss providers, agent settings, active pathways, bootstrap mix, snapshot commit, GPU peak.

## v3.3 Families (executable, independent signing verifiers; `eval/openended.py`, `validation.StagingComputer`)

* **SW — building software.** Natural-language specification of an unfamiliar function to be implemented in
  `solution.py` of the sandbox workspace; `HiddenTestVerifier` runs 60 hidden tests (inputs from the verifier's own
  seed, expected outputs from a hidden reference) in an isolated process; value = pass fraction; achieved at
  >= 0.9. Classes: training sequence `rle, rpn, merge, topk`; held out `unique, base`.
* **MECH — discovering a mechanism.** An owned local HTTP service answers `GET /measure?x=&y=` with a hidden
  two-input function (family drawn per seed: polynomial, trigonometric, rational); write `model.py` with
  `predict(x, y)`; `HeldOutVerifier` scores 80 held-out points (R^2 clipped to [0, 1]; achieved at >= 0.9).
* **CU — computer use on staging interfaces.** `validation.StagingComputer` objectives (`store_entry`,
  `record_item`, `archive_item`: increase a count kept by a staging application through its web interface);
  `MetricThresholdVerifier` reads the application's own state.

No solution, demonstration, strategy template or task-specific rule is given. Per objective: <= 20 decisions
(budget), one objective per GPU-lock hold (<= 15 min, `--hold-s 870`), learner state carried by its own
checkpoint between holds.

## v3.4 Experiments (prespecified; executed in this order as GPU time allows)

* **X0 current capability** (E, seeds 0–2, plan `SW:rle, SW:rpn, MECH:7`): documents what the integrated agent
  does with open-ended generation as it stands (change request S-6 is open). Reported per objective: verified
  status / value, decisions to achievement (censored at 20), executable fraction of chosen actions,
  backbone-written fraction of chosen actions and of candidates, repeats of failed actions, distinct strategies,
  prediction NLL, learning / meta-update records (applied / skipped with reasons, gradient norms, reward-version
  change), compute, wall time. **This is run regardless of S-6 and reported as the state of the agent.**
* **X1 learning within a family** (after S-6 is fixed): E, seeds 0–2, plan `SW:rle, SW:rpn, SW:merge, SW:topk`;
  per-objective metrics as X0; primary: decisions to verified achievement, N_q with q = achieved (pass >= 0.9).
* **X2 transfer (16.8 literal)**: initial vs experienced (X1) weights-only exports, each in a fresh process
  (`eval.transfer` steps, no store / memory / registries / fast state), on held-out `SW:unique`, `SW:base`,
  `MECH:11`, same seed and budget; S_q = mean N_q(initial) / mean N_q(experienced), censored at 20.
* **X3 baselines on the same architecture and action generation**: A (frozen backbone chooses among the same
  open-ended candidates), B (A + written memory), D (no learned reward), C (fast beliefs only) — seeds 0–1, X1 plan.
* **X4 CU-33 primitive-only vs skills, learner in the loop**: arms `skills` / `primitive` (skill executor excluded),
  E, seeds 0–1, 6 objectives `CU:store_entry` then 2 objectives `CU:record_item` (unfamiliar interface);
  success, decisions, primitive operations, compute, latency per objective.
* **X5 goal-free (11.4)**: E with a goal-free phase of 20 decisions (objective with no success criterion) before X1's
  first objective vs E without; seeds 0–1. Run only if X1 shows any achievement (otherwise uninformative).
* **X6 meta-learning (16.5) and integrity (16.6) on the current code**: `eval.meta_checks` on X1 checkpoints and on
  the v2 diagnostic checkpoints; integrity attack battery and trivial-reward world on the current snapshot (CPU,
  engineering diagnostic).

## v3.5 Statistics and power (honest statement)

Unit = seed (an agent and its sequence of objectives); comparisons paired by seed; intervals = bootstrap over seeds;
tests = exact sign-flip permutation. With 3 paired seeds the smallest attainable two-sided p is 0.25 and with 2
seeds 0.5: **no comparison in this protocol can reach p < 0.05**. Results are reported as estimates with their
range across seeds and as counts (objectives achieved / attempted, never-reached reported), not as significance
claims. Measured real-model cost: ~11.5 s per decision with 512-token generation (probe), so one 20-decision
objective ≈ 4–6 min; X0 ≈ 1 h of GPU, X1 + X2 ≈ 2 h, X3 ≈ 3 h, X4 ≈ 4 h, X5 ≈ 1 h; the GPU is shared with the
integration agent and another tenant, so X3–X5 may not all run before the deadline; what does not run is reported
as not run.

## v3.6 Feasibility probes (excluded from results)

`probe_E_s99` (SW:rle, 12 decisions) and `proposal_probe` (SW, MECH) on HEAD 8e666ad — used only to measure cost per
decision and to identify the blocker S-6 (`docs/evidence/S/v3/`).
