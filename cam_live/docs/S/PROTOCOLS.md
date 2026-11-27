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
