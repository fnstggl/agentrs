# IMP — predictive imagination and multistep planning in decision-making (amendment 2026-11-28, item 2)

Agent IMP, worktree `/workspace/camlive` (branch `feat/cam-live`), start HEAD `d1d06dc`. Owned paths:
`src/cam_live/predict/**` except `predict/candidates.py`, `tests/predict/**`, this file, `docs/status/IMP.json`,
`docs/change_requests/IMP.md`. `agent.py` is ST's: the agent-side hook is change request IMP (exact diff, tested on a
patched copy of the tree).

## 1. Audit: what was there

* `CamLiveAgent.evaluate` called `LatentPlanner.plan` in every decision (the controller always allows >= 1 step) and
  wrote `imagined_value_mean / _std / _horizon_steps` into the candidates' extras. **No selection rule read them**
  (`AgentSettings.planner` comment: "recorded; no rule selects on them"; KF-E06; MODEL_ARCHITECTURE §7).
* `plan` imagined *one* rollout per first action: the candidate, then the policy's argmax at every imagined belief,
  read out with the goal evaluator's success probability at the end. No alternative continuations, no value of the
  imagined sequence in the units the decision uses, no information value.
* The horizon grew only while the normalised latent open-loop error of the belief stayed under 0.5 (KF-X10: 1.46 at
  debug scale; in the live planning run's checkpoint `ckpt_00000036`: 2.30 / 1.26 / 2.30 / 1.56 at k = 1..4 after 4
  measurements). Diagnosis of why it can never pass: the belief `b` is the core's re-encoding of the recent context
  (KF-E04, `state=None`); its change from one decision to the next is dominated by the encoding of the *new
  observation text* (stochastic: what the world answered), and the error is normalised by the within-sequence variance
  of `b`. Persistence (`b_{t+1} = b_t`) is itself worse than the sequence mean (untrained residual transition: 1.80),
  i.e. consecutive beliefs fluctuate around their mean. Predicting the full latent is neither possible nor what a
  decision needs.
* Nothing learned whether imagined values are worth anything in verified outcomes.

## 2. What changed (design)

All in `src/cam_live/predict/` (commits listed in §6). Nothing names an action, a tool, a task or a strategy; every
number a decision uses comes from CAM's learned heads, its fast posterior and its verified outcomes.

### 2.1 Multi-step imagination of candidate sequences (`planner.LatentPlanner.plan`)

For every feasible first action `a_i` and imagined particle `p` the first imagined transition applies `a_i` at the
real belief (`LatentTransition`), then a **search over continuations** runs in the latent world model:

* at every imagined belief `b_k` the world model values **every** candidate (`Predictor.goal_values`: the same
  goal-relevant questions, query encodings, scalers, exposures and fast posterior as the real evaluation, read at the
  imagined decision latents `z(b_k, a_j, g)`, net of the same weighted cost; when the decision asks no goal-relevant
  question, the evaluator's success probabilities, read at the end only);
* below the last level every feasible action is expanded as a child while the row budget allows
  (`imagination_max_rows` world-model rows per level, see below), otherwise the best-valued one plus a random
  subset; the value of a node is `max over children (value of the child's action + value of the child's subtree)`;
  an imagined action that would end after the objective's deadline adds nothing;
* the continuation value `C_i^p` is that of the best imagined sequence after `a_i`; the planner reports its mean,
  spread and per-particle values, the imagined next action and the share of particles that agree on it, the
  transition's epistemic spread along the path and the evaluator's success probability at the final imagined belief.

A particle is a coherent sampled model: transition ensemble member `p mod M_T`, world-model ensemble member
`p mod M_W` and the `p`-th draw of the fast posterior's last-layer weights throughout (without the transition's
aleatoric latent noise by default, `imagination_aleatoric`; that noise in a 768-dimensional re-encoding is not
outcome noise). Uncertainty is the spread over particles. Only the goal-relevant columns of the world batch are read
(`Predictor._goal_columns`), with the fast layer's weight draws instead of its posterior-variance path (the first
full-size replay measured 30 s per plan with the variance path on every query; tested: at the real latents the
mean path reproduces `q_goal - net_cost` exactly). Row budget per imagined level: 32768 on a GPU, 8192 on the CPU
(`imagination_max_rows = 0`; measured at debug scale on one thread: 4.1 s vs 0.57 s for a 2-level search of 16
candidates).

**Epistemic spread of the transition (architectural defect found and fixed).** The transition ensemble's members
were initialised with near-zero outputs and no randomised prior functions, so they agreed everywhere: measured on
stored debug-scale transitions, member spread (norm) 0.12 against real belief changes of norm 9.2 — imagination was
*confidently* "nothing changes" for every action never taken, so it could never point at an untried enabling
action. `LatentTransition(prior_scale=...)` now adds randomised prior functions (Osband et al. 2018, as the world
heads already had) that are **zero-mean over the members**: the ensemble mean is exactly that of the trainable
members (a transition trained without them predicts the same mean; checkpoints unchanged, older ones load), each
member learns to cancel its offset where there are data, and members disagree about untried actions
(`predict.transition_prior_scale = 1.0` per dimension of the RMS-normalised belief, the world heads' prior scale).
Test: spread 0.016 -> 0.57 untrained; after training 0.12 on seen actions vs 0.88 on a never-taken one.

"greedy" and "policy" continuations remain available (`imagination_continuation`); the default is "search" because a
greedy continuation cannot find an enabling chain whose intermediate step has no immediate value (test
`test_planner_looks_ahead_through_learned_dynamics_and_needs_the_horizon_for_longer_chains`: KEY1 -> KEY2 -> DOOR is
found with two imagined transitions and not with one).

### 2.2 Information value of experiments

Per candidate, `info_value` = E over the joint particles of `max(0, V_i - V_incumbent)` (for the incumbent:
`max(0, best other - V_incumbent)`), `V = one-step value + continuation`: what settling the candidate's imagined
long-term value would be worth to the decision (an expected-improvement / knowledge-gradient estimate under the
coherent particles). It is recorded on every candidate (`im_info_value`). Exploration over imagined uncertainty
itself happens through Thompson sampling over the particles (§2.4).

### 2.3 Horizon grows only as measured accuracy supports it (`imagination.HorizonController`)

`LatentPlanner.measure` (called by the agent every `planner_measure_every` decisions on its latest real sequence)
now measures, besides the latent error, the **value error** of imagination (`value_consistency_error`): for every
start `s` and `k`, the belief is imagined `k` steps ahead through the actually executed actions, and the candidates of
the real decision `s + k` are valued at the imagined and at the real belief; error = sum of squared differences /
sum of squared deviations of the real values from their mean over the candidates (1.0: the imagination error is as
large as the differences between the actions the decision was about). The persistence baseline (values read at the
start belief) is measured alongside. `imagination_criterion = "value"` (default) decides; `"latent"` keeps the old
rule. Both are recorded (`allowed_by`). Horizon `k` needs >= 3 measurements and a smoothed error <= 0.5 for every
`j <= k`; it shrinks when the error rises.

### 2.4 Imagined consequences in the selection rule, gated by verified outcomes

Rule `thompson_policy` (`DecisionMaker.thompson_policy`) adds per Thompson sample `s`

    score_i[s] = z_i[s] + lambda_pi[s] c_i + lambda_im[s] m_i[s mod P]

* `m` = imagined preference (`policy.imagined_preference`): particle continuation values centred over the feasible
  candidates and divided by the spread of the *one-step* means — the numeraire of `z`; with `lambda_im = 1` the rule
  ranks by the standardised imagined long-term value; equal continuations give exactly 0; winsorised at +-4.
* `lambda_im = beta_im^+ / max(beta_q, 1)` (`policy.imagination_weight`) drawn from the **`ImaginationCalibration`**
  of the planned horizon (the denominator is floored at the prior mean of `beta_q`, unlike the policy's weight whose
  floor is 0.05: in the first ablation with transition priors, `beta_im ~ 0` but `beta_q` near 0.05 gave mean weights
  0.8-1.2, i.e. unsupported imagined differences were amplified because the one-step model predicted the verified
  multi-step progress weakly too; test: when neither predicts, weight < 0.1 vs > 0.3 under the ratio rule):
  Bayesian linear regression (prior `N((0, 1, 0), diag(1, 0.5^2, 0.5^2))`, forgetting 0.99) of the *verified
  multi-step progress* `y_h = sum_{j=0..h} verified_progress(t + j)` of executed decisions on what they recorded
  before acting (`z`, `m` of the chosen candidate). Imagination starts with no systematic weight (prior mean 0) and
  gains weight only where imagined differences predicted verified multi-step progress beyond the one-step model; a
  new horizon starts from the prior. Windows enter only when every decision in them has its final signed value
  (`LatentPlanner.refresh_calibration`, called from `measure`); a decision without goal-relevant questions contributes
  0; a window that runs past the end of an objective is final once a later objective is seen; each decision once.
* `predict.imagination_select = false` is the ablation "imagination weight 0": the same numbers are drawn, the
  distribution without imagination is used. In both arms the decision records `selection.imagination`: both
  distributions, their total variation (`share_tv`: the probability that imagination changes the executed action
  under the maximal coupling), the weight drawn, the preference, the calibration summary and the horizon.

### 2.5 Persistence and compatibility

`LatentPlanner.state()` = the controller's old keys + value-criterion statistics + the imagination calibration;
`load_state` also reads the bare controller state of older checkpoints (calibration from its prior, recorded in
`planner.restore_info`). No new parameters (the transition's prior functions are non-persistent buffers regenerated
from the seed): the live run's checkpoints load unchanged (§3.2).

### 2.6 Agent hook (change request IMP -> ST)

`docs/change_requests/IMP.md`: `plan(..., predictor=self.predictor, evaluations=evals)`, `ev.extras.update(
im.extras())`, `planner.state()` / `planner.load_state()` in save / load. No change to `select()`.

## 3. Evidence

### 3.1 (a) Implementation correctness

`tests/predict/test_imp_planning.py` (10 tests) and `tests/predict/test_imp_agent.py` (2):

* imagined preference: exactly 0 when every candidate continues equally well; one-step-spread units, infeasible 0,
  winsorised; with weight 1 the score ranks by the standardised imagined long-term value;
* imagination calibration: prior weight only exploratory (mean lambda < 0.5, half the draws > 0); after 200 verified
  windows with no imagined signal |beta_im| < 0.1 (lambda ~ 0); with signal 1.5 the learned lambda = 1.5 +- 0.15;
  a new horizon starts at the prior; each decision once; state round trip;
* rule: trusted imagination moves the choice to the enabling candidate (p 0.04 -> > 0.5), discredited imagination
  has no say (lambda < 0.05) however large the imagined differences, the ablation draws the same numbers (identical
  share_tv) and uses the distribution without imagination; infeasible candidates exactly 0;
* planner on a learned world (a real `LatentTransition` trained on KEY1 -> KEY2 -> DOOR dynamics): with one imagined
  transition nothing is found from the start, with two the chain starting with KEY1 is (continuation > 8 vs < 3 for
  every other first action, all particles agree on KEY2 next); after KEY1 one transition suffices; the deadline ends
  imagined sequences; the planner's numbers reach the rule (p(KEY1) > 0.8 with a trusting calibration, < 0.05 with
  imagination off);
* value criterion: zero for exact dynamics, equal to the persistence baseline for a transition that predicts no
  change, horizon needs min_count and shrinks when the error rises; an old controller state loads;
* verified windows: pending until final, sums of signed progress, 0 for decisions with nothing attributable,
  truncated windows final once a later objective is seen, never twice;
* goal values at the real latents equal `q_goal - net_cost` (to 1e-6); per-particle path finite and varied;
* transition priors: ensemble mean and state-dict keys unchanged; spread 0.016 -> 0.57 untrained; trained 0.12 on
  seen vs 0.88 on a never-taken action;
* agent (patched copy, real runtime path, debug scale, T1 2 x 6 decisions): every candidate carries the planner's
  numbers, every thompson_policy decision with imagination records both distributions (sum 1), share_tv in [0, 1],
  the horizon equals `planner_horizon`; >= 3 verified windows reached the calibration; a checkpoint restores it
  exactly; the live run's old planner state loads with a fresh calibration (reported).

### 3.2 (b) Integration into the actual runtime

* Debug scale, patched copy (HEAD + change request IMP): `run_lifetime` -> `LiveRuntime` -> `CamLiveAgent.evaluate`
  -> `Predictor.evaluate_candidates` -> `LatentPlanner.plan` -> extras -> `CamLiveAgent.select` ->
  `DecisionMaker.thompson_policy`; `learn` -> `_measure_planner` -> `LatentPlanner.measure` ->
  `refresh_calibration`; `save` / `load` -> `planner.state()` / `load_state()`. Exercised by the agent test and by
  every run of §3.3-3.4.
* **Full size, CAM's own live experience** (`python -m cam_live.predict.imp_eval replay`, evidence
  `docs/evidence/IMP/replay_live_ckpt38_fullsize.json`): read-only copy of the live planning run
  (`discord-100-planning-20261128T083454Z`, checkpoint `ckpt_00000038`, 36 executed decisions), Qwen3-4B + 50m core on
  CUDA, code ff5faca + change request IMP (IMP's planner, value criterion, calibration and rule; before the
  transition priors 88a9f68, the goal-column path 404a366 and the weight floor 00f8913). The checkpoint loads with the new code (arch-signature differences: none;
  the planner state of the old format loads, calibration and value-criterion statistics start fresh, reported in
  `planner.restore_info`; one optimizer entry `exp.core.persist_rho` fresh — a parameter of another workstream).
  On 3 real sequences: latent errors 1.81 / 1.84 / 2.62 / 3.02 (k = 1..4: worse than predicting the mean belief, as
  KF-X10), value errors 0.040 / 0.047 / 0.066 / 0.069 against the persistence baseline 0.038 / 0.041 / 0.055 /
  0.055 — the horizon by the value criterion is 4, by the latent criterion 1. 24 of CAM's real decisions re-evaluated
  in their recorded contexts with the planner (16 candidates each, Discord / browser / http / sandbox / internal
  tools): imagined continuation values differ between first actions by sd 0.0012 (median) against a particle spread
  of 0.26; the imagined preference |m| <= 0.020 one-step units; information value 0 in every decision; with the
  prior weight (no verified window yet) share_tv mean 0.004, max 0.031, argmax changed in 0 of 24 decisions.
  Plan time 30 s per decision with the code of that run (posterior-variance path on every query; fixed afterwards,
  §2.1). GPU hold ~16 min (over the ~15 min guideline: the plan cost was not known in advance; the process ignored
  SIGINT because it ran in the background).

### 3.3 (c) Causal influence on actual decisions

Measured two ways on the real runtime path: within a decision, the total variation between the rule's distribution
with and without imagination (`share_tv`: the probability that imagination changes the executed action under the
maximal coupling of the two) and whether the argmax differs; across runs, paired arms `imagination_select` on / off
with the same seeds (same code, same random numbers until behaviour differs).

| configuration (debug scale, T1 lab/door, 16 decisions / lifetime) | decisions with imagination | mean share_tv (95% CI over seeds) | argmax changed |
|---|---|---|---|
| A: no transition priors, ratio weight (code c5a81ed + CR; 6 seeds, 47 lifetimes, stopped) | 249 | 0.010 (0.007-0.014) | 7 (2.8 %) |
| B: transition priors, ratio weight (6415b45 + CR; 6 seeds, 24 lifetimes, stopped) | 118 | 0.068 (0.056-0.081) | 21 (18 %) |
| **B2: final code** (00f8913 + CR; 10 seeds, 42 lifetimes) | 244 | **0.053 (0.046-0.059)** | **22 (9.0 %)** |
| full size, CAM's own live decisions (replay, prior weight, no transition priors in that code) | 24 | 0.004 (max 0.031) | 0 |

In every one of the 10 B2 seed pairs the arms' executed actions diverge in lifetime 0 (first divergence at decision
0-5). Imagination therefore causally changes actual decisions on the runtime path; without the transition's
randomised priors that influence was nearly nil (A, and the full-size replay), because imagined continuations hardly
differed between first actions. In B2 the planner ran at horizons 1 (176 decisions), 4 (85), 3 (10), 2 (2); the
chosen candidate's |imagined preference| had median 0.10 and 90th percentile 0.81 one-step units; information value
> 0 for 66 % of chosen candidates (median 0.019 goal units); plan 0.56 s per decision (CPU, debug scale).

What gated it: in B2 the drawn imagination weight stayed at its prior level (mean 0.17 in lifetimes 0, 1 and 2)
while the calibration's `beta_im` stayed at 0 (-0.017, +0.011, +0.074 by lifetime; 2-7 verified windows per seed and
lifetime), i.e. the verified multi-step outcomes gave imagination no systematic weight — as designed, it remained an
exploratory perturbation. Under the earlier ratio weight (A, B) the drawn weight *rose* with experience (A: 0.35 ->
0.45 -> 0.97 -> 1.28 over lifetimes 0-3) while `beta_im` stayed ~0: the defect fixed in 00f8913.

### 3.4 (d) Measured improvement in learning or competence

**Debug scale, B2 (final code), paired on - off over 10 seeds** (evidence
`docs/evidence/IMP/ablation_B2_final_code_summary.json`, raw rows `ablation_B2_final_code_raw.tar.xz`):

* verified progress per lifetime, lifetime 0: +0.237 (bootstrap 95 % CI 0.014-0.488; 5 seeds better, 2 worse,
  3 equal; sign test p = 0.45; driven by three seeds, +0.62 / +0.71 / +1.00, two of them pairs where the off arm got
  stuck waiting / probing for 16 decisions); lifetime 1: -0.011 (CI -0.187-0.180; 2 better, 5 worse). Pooled over the
  lifetimes each pair completed: +0.164 (CI -0.043-0.413). Objectives achieved 8 of 23 lifetimes (on) vs 5 of 19
  (off); ground-truth expected adopters on - off +2.7 per lifetime (CI -2.0-8.5).
* the delayed dependency of T1 (the last channel yields 6x after `lab.unlock`): unlock executed in 3 (on) vs 2 (off)
  lifetimes; interventions on that channel after an unlock 7 (on) vs 0 (off) — one on-arm pair (seed 7) unlocked at
  its first decision (the decisions of the two arms differed from decision 0) and achieved the objective in 7
  decisions where the off arm did not (0.29).
* **Verdict: no reliable improvement.** The lifetime-0 difference does not survive the second lifetime, is carried
  by three seeds, and the per-decision record shows that the on arm's changed choices came from the exploratory
  imagination weight (beta_im ~ 0), not from a learned trust in imagination. The batch was stopped at 42 of 120
  planned lifetimes by the CPU rule (master, 10:50 UTC); A and B, with the earlier designs, show on - off -0.064
  (CI -0.159-0.024) and -0.014 (CI -0.107-0.099).

**Full size**: §3.5. **From one experienced checkpoint**: §3.6 (no improvement either).

### 3.5 Full-size evidence

* Integration, causal influence and accuracy on CAM's own live decisions (replay, §3.2): imagination changes the
  rule's distribution by share_tv 0.004 on average (max 0.031) and no argmax in 24 decisions — with the code of that
  run (before the transition priors, 88a9f68). Its imagined continuations do not distinguish CAM's real candidates
  (sd 0.0012 against particle sd 0.26), the value error of imagination equals that of not imagining (0.040 vs 0.038 at
  k = 1), and there is no verified multi-step window yet for the calibration (the live checkpoint predates it).

### 3.6 Diagnostic: both arms continuing from one experienced checkpoint

To ask whether imagination helps once the world model has experience, one agent first explored 4 T1 lifetimes with
`predict.eps_floor = 0.9` (near-uniform choice among the offered candidates, exact propensities recorded; world seeds
81000-81003), then both arms continued from its checkpoint (identical weights, experience and imagination calibration
of 63 verified windows) on 3 held-out worlds each, 6 replicates (world seeds 82000 + 10 r), code 00f8913 + CR
(evidence `docs/evidence/IMP/ablation_P_from_exploration_checkpoint_summary.json`).

* The exploration did **not** give the world model the delayed dependency: `lab.unlock` was executed once in 64
  exploratory decisions (the offered candidate set rarely contained it) and never followed by an intervention on the
  last channel. The diagnostic therefore tests imagination with an experienced-but-ignorant world model, not one that
  knows the mechanism.
* Causal: share_tv 0.048 (CI 0.035-0.062), argmax changed in 22 of 187 decisions (11.8 %).
* Calibration: after 88-101 verified windows per replicate, `beta_im` = -0.18 .. +0.31 (posterior sd 0.18-0.40):
  no evidence that imagined differences predict verified multi-step progress; the drawn weight rose only from 0.19 to
  0.24 over the three lifetimes.
* Outcome: verified progress on - off -0.013 per lifetime (CI -0.10-0.07; 5 of 18 lifetime pairs better, 8 worse),
  objectives achieved 8 / 18 in both arms, ground-truth adopters -0.2 per lifetime (CI -5.3-4.9); unlock executed in
  6 (on) vs 8 (off) lifetimes, interventions on the last channel after an unlock 20 vs 18. **No improvement.**

## 4. Negative results and their diagnosis

1. **Imagined continuations barely differ between first actions** (full size: sd over candidates 0.0012 against a
   particle spread of 0.26, |m| <= 0.02; debug scale without transition priors: share_tv 0.010). Diagnosis, measured:
   the world model's values of the candidates respond hardly at all to the belief changes on CAM's real sequences —
   the value error of imagination equals that of *not imagining* (full size 0.040 vs 0.038 at k = 1; debug 0.015 vs
   0.015), both small relative to the differences between candidates. The learned world model predicts an action's
   value mostly from the action and the slowly varying context, not from what the previous action changed.
   **Empirical limitation** of the world model at this amount of experience (36 live decisions; <= 6 debug
   lifetimes), compounded by the re-encoded, stateless belief (KF-E04, PS's item) whose step-to-step change is
   dominated by the new observation text. It is why the value criterion admits horizon 4 at both scales: imagination is
   *harmless* to the decision there, not informative — the persistence baseline is recorded next to it for that reason.
2. **The transition ensemble was overconfident** (member spread 0.12 vs real belief changes of norm 9.2): imagination
   was certain that untried actions change nothing, so it could never point at an untried enabling action and its
   information value was 0. **Architectural defect — fixed** (zero-mean randomised priors, 88a9f68); afterwards the
   imagined preference is non-trivial (B2 median 0.10, p90 0.81 one-step units) and information value > 0 for 66 % of
   chosen candidates.
3. **The policy-style ratio weight amplified unsupported imagination** when the one-step coefficient was weak (A:
   drawn weight 0.35 -> 1.28 over four lifetimes with beta_im ~ 0). **Architectural defect — fixed** (00f8913).
4. **No measured improvement of decisions** (§3.4). With (2) and (3) fixed, verified multi-step windows did not
   support imagination (beta_im ~ 0 after <= 7 windows per seed and lifetime), so its influence stayed exploratory.
   The delayed dependency of T1 that imagination is meant for (unlock -> 6x yield) was experienced in only 5 of 42
   lifetimes across both arms; the world model cannot predict, and imagination therefore cannot imagine, a
   consequence that was almost never observed. **Empirical limitation** (exploration of enabling actions and the
   amount of experience), not demonstrated to be architectural. The exploration-checkpoint diagnostic (§3.6) did not
   separate the two either: near-uniform exploration executed `lab.unlock` once in 64 decisions, because the
   candidate set offered per decision rarely contained it (candidate generation is ST's / B's machinery); with 88-101
   verified windows the calibration still found no predictive value in imagination (|beta_im| <= 0.31, sd 0.18-0.40).
5. Selection-calibration data at full size: none yet (the live run's checkpoint predates the imagination calibration;
   its decisions carry no imagination numbers), so the full-size weight is the prior's.

## 5. What is not done

* agent.py hook: change request IMP (exact diff, applies to the current worktree agent.py with offsets) — **not yet
  applied by ST**; until then the worktree's agent records the planner's old three numbers only and its
  thompson_policy sees no imagination (the agent test is skipped with that reason). CAM's live planning run runs from a
  frozen snapshot (`runs/cam_live/live/code_a0f3411`) and does not get any of this.
* Randomised experiments are still designed and measured on one-step outcomes (`DecisionMaker.propose_experiment`);
  the planner's information value is recorded per candidate and enters exploration only through Thompson sampling
  over the imagined particles.
* The transition is still trained on the one-step latent NLL only; a value-consistency objective for it (train the
  imagined belief to preserve the candidates' values) is the next step if, with more experience, the value error rises
  above the persistence baseline in decision-relevant ways — not needed for the sequences measured so far.
* The continuation set at imagined beliefs is the current candidate set (actions that only become available later
  are not imagined); the search is exhaustive at the first imagined level within the row budget and greedy plus a
  random subset deeper.
* Substantive full-size evidence is limited to the replay of CAM's live decisions (§3.2) and one paired T1 lifetime
  (§3.5); the debug batch was stopped at 42 of 120 lifetimes by the CPU rule.

## 6. Commits (prefix `cam live [IMP]`)

`bb24f96` planner search, goal values at imagined beliefs, value criterion, ImaginationCalibration, rule integration ·
`f4c44e6` paired ablation driver · `f259f44` change request IMP, agent-path / compatibility tests, replay ·
`a88f5fe` device row budget · `c5a81ed` partial results · `88a9f68` transition randomised priors · `404a366` goal
columns + per-particle fast draws (30 s -> sub-second plans), full-size replay evidence · `6415b45` lifetime wall cap
· `6e06869` summary · `00f8913` imagination weight floored at the prior mean · `cb1daf7` ablation evidence; later
commits: §3.5-3.6 evidence, this document, status, JUnit.

Reproduce: `scratchpad/IMP/export.sh <dir>` = `git archive HEAD cam_live/src` + `patch agent.py <
docs/change_requests/IMP.md's diff`; `PYTHONPATH=<dir> python -m cam_live.predict.imp_eval run --arm on|off --seed S
--family T1 --lifetimes 6 --decisions 16 --threads 1 --workdir W --out O.json` and `... summarize O*.json --out S.json`;
full size: `--real --backbone --lifetime-wall-s 600` with `CAM_LIVE_GPU_LOCK` set (the run holds the shared lock);
replay: `... replay --run-dir <live run> --ckpt <live run>/checkpoints/LATEST --work W --n-last 24 --out R.json`
(read-only copy of the run; the GPU part inside the lock).
