# CAM 7.0 research ledger

Read this first in every session. Newest entries at the top of each section. Results are only listed as
"demonstrated" when a pre-declared comparison on untouched tasks supports them.

## Where things are

| Item | Location |
|---|---|
| CAM 7.0 branch | `feat/cam-7.0` (pushed to `origin` at `eaf488c` on 2026-11-24, public repository; from `feat/cam-6.1` `feccc86` + crucible-4 `c07098b` + superlearner-4 `58c18ce`) |
| CAM 6.1 code | `superlearner/src/superlearner/unified/` |
| CAM 6.1 report | `superlearner/docs/CAM61_RESULTS.md` |
| Raw run outputs | `runs/cam61/` (git-ignored, ephemeral disk; 4.3 GB incl. checkpoints, 5.9 MB of `lifetime.json`) |
| Hardware | 1× RTX A6000 48 GB, 48 vCPU, 125 GB RAM, 300 GB disk, **no persistent volume** |
| Instance | Vast.ai 62037028, $0.48/h, rental ends 2026-11-30 03:00 UTC |
| Old GPU backup | incomplete: 2,461 of 23,308 files at `/workspace/.agentrs.retry`; resume with `/workspace/.agentrs-sync/resume-from-mac.sh` from the Mac |

## Bars to beat (held-out mechanism families, score 0 = hold defaults, 1 = reference)

| Learner | Held-out score | Source |
|---|---:|---|
| Exact Bayesian regression on move and squared move (`meta_np` untrained) | 0.241 | `crucible/docs/experiments/ow_meta_np_result.json`, 48 tasks |
| `model_based` (meta-learned Bayesian last layer) | 0.170–0.242 | CAM 6.1 decisive run; meta_np report |
| `meta_np` trained (neural process correction) | 0.176 | failed its criteria: trained < untrained on held-out families |
| PPO-GRU | −0.139 | CAM 6.1 decisive run |
| CAM 6.1 (first round) | −0.066 | CAM 6.1 decisive run |

Lesson from `meta_np`: in this benchmark, features learned across worlds have so far *reduced* held-out performance
relative to exact regression. Any CAM 7.0 claim of learned priors must beat the exact regression on held-out families.

## CAM 7.0 integrated learner (Phase 3, started 2026-11-24)

Code: `superlearner/src/superlearner/cam7/` (`posterior.py` fast layer, `model.py` network and tensor interfaces,
`agent.py` learner and objectives, `worlds.py` Experiment 2 families, `experiments/{train,deveval,exp1b,exp2}.py`).
Tests: `superlearner/tests/cam7/` (9 pass). Runs: `runs/cam7/` (git-ignored). A 240k-step training run takes about
10 minutes (CAM 6.1: 3.7 hours).

Design: one transformer over lever, gauge, global and event tokens (per-episode random identities, nothing tied to a
slot). Pass 1 gives each lever's context, from which the adaptive features of every row are computed (fixed d, d², e, e²
+ learned + learned pairwise interactions). An exact normal-inverse-gamma regression on those features (revenue and cost
separately, Student-t predictive, learned priors and forgetting) is the fast belief. Pass 2 of the same transformer reads
the posterior as tokens. Decision score over candidate interventions = β [posterior mean value + remaining decisions ×
knowledge gradient] / noise sd + learned residual. Objectives: support/query meta-NLL (features, priors, pass 1), world
auxiliaries, clipped actor-critic on actual outcomes with the posterior as a fixed input.

### Development results (development seeds only; nothing here is confirmed)

| Date | Experiment | Result | Reading |
|---|---|---|---|
| 2026-11-24 | Ablation: attention learner without explicit Bayesian adaptation (`explore_rule=learned_only`, `use_bayes_inputs=false`: no posterior input, no posterior-generated candidate; trained 240k like the others; 1 seed; dev 48 tasks per split) | Held-out −0.195 (untrained −0.284), familiar −0.282 (−0.281); vs CAM 7.0 small (3 seeds) −0.368 [−0.512, −0.217] and −0.568 [−0.704, −0.449]. A first version that still received posterior-generated candidates reached about +0.18 by learning to pick them (`runs/cam7/dev_cam7_small_learned_only_contaminated`) | Removing the adaptive posterior destroys control: the same network trained by RL alone stays below holding the defaults, like PPO-GRU and CAM 6.1. The explicit fast inference is what works |
| 2026-11-24 | Diagnostic after C-2/C-3: weight on the value of information (`kg_weight`), 3 seeds, dev sets (`runs/cam7/exp1b_dev`, `runs/cam7/exp2_dev3`), no residual | Familiar: weight 0.25 +0.045 [+0.007, +0.084] vs 1.0; held-out: 0.25 −0.018 [−0.058, +0.023], 0.5 −0.003; Exp 2: 0.5 −0.040 [−0.108, +0.026], 0.25 −0.043; weight 0 (greedy over the same candidates) worst everywhere | Hypothesis "KG × remaining decisions over-explores" not supported across sets; textbook weight 1 kept. The mid-episode lag behind exact regression on held-out families (k=16: about 0.02 vs 0.11) is present at every weight, so it is not the exploration weight |
| 2026-11-24 | Exp 2 dev (`runs/cam7/exp2_dev`, x2_confound_gate + x2_regime_mix, 12 tasks each, VAL offset 60,000, one trained small model, budget 24 ticks) | Episode score: knowledge gradient **0.505**, full (KG + learned residual) 0.447, Thompson 0.437, EIG 0.433, greedy 0.351, `model_based` trained 0.344, fixed design 0.276, random 0.166. Commit score at tick 24: `model_based` 0.334, greedy 0.272, Thompson 0.252, KG 0.213. Probe rank correlation: Thompson 0.58, `model_based` 0.55, KG 0.47 | Decision-relevant VOI gives the best verified outcome; it does not give the best information at a fixed early budget. The learned residual hurts on these unseen worlds |
| 2026-11-24 | Exp 1B dev (`runs/cam7/exp1b_dev`, 48 tasks per split at VAL offset 60,000, frozen, fresh context) | Held-out: hold 0.000, exact BLR 0.132, `model_based` untrained 0.155 / trained 0.159 (3 seeds), CAM 7.0 base untrained 0.168, trained 0.165, small trained 0.183 (+0.028 [−0.035, +0.091] vs `model_based` untrained). Familiar: `model_based` untrained 0.246 / trained 0.263, exact BLR 0.244, CAM 7.0 small trained **0.308** (+0.062 [+0.025, +0.101] vs `model_based` untrained) | Training helps on familiar families; no measurable transfer to held-out families. Training mostly cut early exploration losses (held-out window k=4: −0.285 untrained, −0.138 trained) without raising the final level |
| 2026-11-24 | Untrained exploration rules, dev sets (32 tasks per split) | `model_based` untrained 0.202 / 0.240 (familiar / held-out), CAM 7.0 KG 0.130 / 0.156: losses concentrated in `abm_congestion` (−0.53; greedy +0.07) and `tech_pipeline` (−0.43) | Exploration that the one-period value of information prices as cheap does lasting damage in worlds with slow feedback (churn); the learned residual is meant to learn this |

## Experiments

### Completed

| Date | Experiment | Result | Verdict |
|---|---|---|---|
| 2026-11-24 | **Delayed-reward cue task** (`cue_final`: only the last of 8 ticks is rewarded, by the first tick's cue; 3 seeds, 250 rounds; `runs/cam7/memory_cue_final.jsonl`) | window 16: 0.031, −0.156, −0.141; window 1: 0.031, −0.156, −0.188 (scale: +1 always right, 0 chance) | **Failed: agent-level long memory is not demonstrated.** The earlier cue-task "success" was largely win-stay from the previous action and reward inside the current event. Representation holds the cue (probes); the failure is credit assignment from a delayed reward |
| 2026-11-24 | Mandatory test 4 after scoping the conditioning changes to the attention core (CAM 6.1 path restored), 64 held-out episodes | before −1.004, after **0.273** (threshold 0.30) | Narrow fail at the larger, less noisy evaluation; the earlier pass used 16 held-out episodes. Threshold unchanged |
| 2026-11-24 | Held-out reward forecasting (mandatory test 4, RSSM core) after the conditioning changes, 2 seeds | after R²: current −0.016 / 0.202; without FeatureNorm 0.283 / 0.387; zero-init + FeatureNorm 0.015 / −0.563; original 0.635 / −0.155 | FeatureNorm hurts the RSSM's imagined forecasts (statistics of observed beliefs applied to imagined ones): **removed from the RSSM, kept in the attention core**. Metric is noisy with 16 held-out episodes; test now uses about 64 |
| 2026-11-24 | **Agent cue task with all fixes** (event core direct path, lr 3e-4, FeatureNorm, small-init heads, V-trace actor; 5 seeds, 150 rounds; `runs/cam7/memory_agent_allfix_*.json`) | window 16: 0.368, 0.124, 0.375, 0.375, 0.308 (mean **0.310**, optimum 0.375); window 1: mean 0.042 (one seed 0.214, plausibly "repeat what just paid" from the previous action and reward inside the current event); twin-Q actor: 0.002 | **Promising but not reliable**: 4 of 5 seeds near optimal vs control, but the mandatory test run of the same configuration on seed 3 then gave 0.041 vs control 0.256 (GPU runs are not bit-reproducible; successful runs learned only in their last third). Twin-Q actor still fails; V-trace retained. Retesting with 250 rounds |
| 2026-11-24 | Critic isolation on the cue task (supervised target: reward of the chosen action; 3–5 seeds, 600 steps) | Cue and action sign each linearly decodable (1.00) at training rows. Raw belief varies only sd 0.022 across samples, action embedding 0.091. Heads on raw features: 0.50–0.57. Batch-whitened inputs + plain MLP: 0.99–1.00 (3/3). With running feature normalisation, head designs: zero-init LayerNorm 1/5, **small-init LayerNorm 3/5 (+0.90)**, bilinear 1/5 (+0.86) | **Two more causes found**: badly conditioned belief/action features, and exactly-zero output init slows critics. Fixed with `FeatureNorm` (running statistics, applied after use, no within-batch leakage; tested) on belief, action and goal embeddings, and small random output init for all distributional heads |
| 2026-11-24 | CAM 7.0 dev run v2 (fixed event core, before FeatureNorm and init fix), familiar families, seed 0 | 0 → −0.259, 42k +0.085, 81k +0.110, **121k −0.166**; adaptive-layer held-out R² 0.60–0.75, effect R² rising to 0.29 | Unstable: collapse late in training. To revisit with the conditioning fixes |
| 2026-11-24 | Agent cue task with all-candidate twin-Q actor (5 seeds) | window 16: mean −0.005; window 1: −0.004 | Failed: the critic itself did not learn the interaction (see critic isolation) |
| 2026-11-24 | Agent cue task, fixed event core, lr 3e-4, V-trace actor (5 seeds) | window 16: −0.051, −0.012, 0.053, 0.042, **0.31** (mean 0.068); window 1: mean 0.017 | **Still failing at agent level** (1 of 5). Representation and reward head now learn; the reward of the last action is readable from the current observation, so only the critic and multi-horizon heads require the cue. Testing the all-candidate twin-Q actor next |
| 2026-11-24 | Decisive rerun of the CAM 6.1 adaptive layer (3 seeds, 240k; `runs/cam61/decisive_adaptive_*`; intervals use the older seed-averaged task bootstrap) | Held-out trained: full −0.021, no planner −0.015, no hand evidence −0.001; model-based +0.170, PPO −0.139. full − model-based −0.191 [−0.273, −0.107]; full − CAM 6.1 round 1 +0.045 [−0.010, +0.103]; full − PPO +0.118 [+0.018, +0.226]; planner and hand evidence: no measurable effect. No-world-model ablation paused (resumable) | **Q2 answered: the ALPaCA-style layer did not solve the problem.** The dev +0.221 did not replicate. This run predates the float64 / task-scale fixes (C-1) and the event-core fix |
| 2026-11-24 | Agent-level reward isolation on the cue task (`runs/cam7/reward_isolation.jsonl`, 2–3 seeds, 20 evaluation batches) | Reward visible in the observation: pooled observation → any head: sign accuracy 1.00 (all seeds). Through the event core: 0.48–0.53 regardless of head, LayerNorm or zero-init. Event token alone: 1.00; after the attention body: 1 of 2 seeds; full pathway at lr 1e-3: 0 of 3 (one divergence). With a **direct current-event path into the belief and lr 3e-4: 3 of 3 seeds solved**; lr 1e-3 still fails | **Bug found and fixed**: the event core could not reliably read the current step through attention-mixed history, and is unstable at lr 1e-3. Earlier single-batch measurements were too noisy and briefly blamed the head initialisation; that was wrong |
| 2026-11-24 | Agent cue task, attention core with banded mask (5 seeds, 150 rounds; `runs/cam7/memory_agent_*.json`) | window 16: mean 0.008 (0.053, −0.059, 0.053, 0.001, −0.007); window 16 + value gradients into the belief: 0.013; window 1 control: −0.003 | **Failed**: the earlier 1-of-3 success did not replicate. The representation can hold the cue (probe), so the gap is in the agent's learning signal or actor. Probing the trained agent next |
| 2026-11-24 | Exp 1 condition A, round 3: training loss in each task's full-data spread (`blr_meta_taskscale`, fixed+learned features) | Held-out (third look at this set): effect error vs fixed k=8 −0.183 [−0.278, −0.100], k=16 −0.220 [−0.312, −0.141]; rank k=16 +0.032 [+0.003, +0.062], k=1 −0.097 [−0.196, −0.001]. Familiar: effect error better for k ≥ 4, rank neutral to slightly positive | **Supports** the variance-inflation hypothesis. Development evidence only, because this held-out set was inspected in rounds 1–3. Confirmation C-1 declared |
| 2026-11-24 | Agent-level cue task with the attention core (window 16, 3 seeds, 150 rounds) | final 0.035, **0.375 (optimal)**, 0.053 | First CAM agent to solve the memory task in any seed; 2 of 3 seeds stay at chance, so the remaining failure is RL optimisation (the probe shows the representation keeps the cue). The window-1 control crashed (window shorter than training sequences); fixed with a banded mask and no absolute positions |
| 2026-11-24 | Exp 1 condition A, round 2 (float64; held-out and familiar families; learned-only and fixed+learned features; 3 seeds; `runs/cam7/fewshot/condition_a_*_round2.json`) | Meta minus fixed, rank at k=8: held-out −0.092 [−0.146, −0.039], **familiar −0.090 [−0.139, −0.044]**; effect error k=8: held-out +1.32, familiar +0.89. Fixed+learned identical to learned-only. Converge by k=64 | **Not a transfer failure**: meta-training gives worse beliefs even in-distribution. Next hypothesis: the support-estimated spread in the training loss rewards variance inflation over mean accuracy |
| 2026-11-24 | Memory pathway, focused (3 seeds, 1,500 steps, delays 0/4/16; `runs/cam7/memory2/`) | Task accuracy, world-model training: attention 1.00/1.00/1.00, GRU 1.00/0.67/0.48, RSSM sampled 0.53/0.48/0.49, RSSM probs 0.68/0.48/0.50. Supervised: attention and GRU 1.00 at all delays; RSSM 1.00/0.65/0.49; RSSM probs 1.00/0.64/0.48 (its latent keeps the cue, 0.97–1.00, but the belief loses it within 4 steps) | **Cause of CAM 6.1 memory failure found, in two parts**: the sampled latent drops evidence, and the RSSM transition loses it within a few steps even under direct supervision. **Decision**: CAM 7.0 memory = causal attention over event tokens |
| 2026-11-24 | Memory pathway probe, screening (1 seed, 600 steps; `runs/cam7/memory/`) | Under world-model training the cue is 100% decodable from the encoder and posterior logits but 56% from the **sampled** latent and 52% from the belief at the step it appears. GRU and attention keep it (100%). Supervised retention: attention to delay 32, GRU to 4, RSSM to 1. 5 runs died of GPU OOM (shared GPU) | **Located**: the straight-through sampled categorical latent discards evidence the posterior holds. Next: posterior-probability latent (`latent_mode="probs"`) |
| 2026-11-24 | Exp 1 condition A, round 1: identical support sets, 120 held-out-family tasks, 3 seeds (`runs/cam7/fewshot/condition_a_heldout_round1.json`) | Query ranking, k=8: tuned fixed BLR 0.543 vs meta-trained BLR 0.451; k=64: 0.766 vs 0.758. Effect error k=8: 0.91 vs 2.23. NLL: meta better only at k≤2 by inflating variance, worse at k≥16 | **Meta-trained features did not beat generic regression.** NLL at k≤2 is dominated by the support-only scale estimate and is not decision-relevant |
| 2026-11-24 | Memory fixes on the cue task (2 seeds): shared policy gradients | memory 0.026, −0.059; none 0.053, −0.059 | **Failed**. More updates / both variants timed out (no result) |
| 2026-11-24 | Adaptive layer dev run, 3 debugging families, seed 0, 120k steps | full +0.221, no planner +0.115, no hand evidence +0.103 | Development only (families overlap debugging) |
| 2026-11-23 | CAM 6.1 decisive, 3 seeds, 240k steps | full −0.066 held-out vs model-based +0.170 | **Failed** criterion |
| 2026-11-23 | Memory cue task, 3 seeds, 3 configs | all ≈ −0.03 (chance) | **Failed** |
| 2026-11-23 | Probe: cue decodable from belief | recurrent agent 56% even at the step the cue is shown; memoryless 100% at steps 0–1 | Information lost before or at the belief |
| 2026-11-23 | Effect-head planner | held-out effect R² ≤ 0.155 | Failed; off by default |
| 2026-11-23 | Actor objectives | all-candidate twin-Q chases critic error (−0.66); V-trace stable but only learns to hold | V-trace kept |
| 2026-11-23 | Signal check: within-episode regression on settings | explains 61–97% of reward variance across 6 training families | Signal exists |

### Pre-declared confirmations (written and committed before the run)

**C-2, Experiment 1 condition B (each learner chooses its own interventions; declared 2026-11-24, before any look at
the confirmation set).** Set: VAL offset 70,000 (seeds 1,070,000–1,070,011) of the four held-out families (48 tasks)
and of the six training families (72 tasks, "familiar": unseen instances of the training families); never used by any
run. Every learner frozen, fresh context, `explore=False`, `superlearner.cam7.experiments.exp1b`. CAM 7.0 = the `small`
configuration, full decision rule (knowledge gradient + learned residual), the three already-trained checkpoints
`runs/cam7/dev_cam7_small/s{0,1,2}/ckpt/point_0004.ckpt` (240k steps on the six training families); "untrained" =
`point_0000` of the same runs. Statistic: whole-episode score (area under the adaptation curve), paired per task,
two-level bootstrap over training seeds and tasks, 95%. Each claim stands alone and passes only if the lower bound > 0:
- C-2a familiar, parametric learning: CAM 7.0 trained − CAM 7.0 untrained (dev: +0.111 [+0.031, +0.217]);
- C-2b familiar, against the strongest baseline: CAM 7.0 − `model_based` untrained (dev: +0.040 [+0.001, +0.081]);
- C-2c held-out families, against the strongest baseline: CAM 7.0 − `model_based` untrained (dev: +0.018 [−0.029, +0.065]);
- C-2d held-out families, transfer of what was learned: CAM 7.0 trained − untrained (dev: +0.030 [−0.015, +0.077]).
Also reported without a criterion: `model_based` trained (3 seeds), exact regression (`meta_np` untrained), hold, CAM 6.1
adaptive (decisive checkpoints) if it loads, CAM 7.0 without the residual, and the adaptation curves at k = 0…64.

**C-2 result (run once, 2026-11-24, `runs/cam7/confirm/exp1b_{heldout,familiar}`):** C-2a **PASSED** (familiar,
trained − untrained +0.172 [+0.094, +0.261]); C-2b failed (familiar, − `model_based` untrained +0.024 [−0.018, +0.067]);
C-2c failed (held-out, − `model_based` untrained −0.042 [−0.100, +0.010]); C-2d failed (held-out, trained − untrained
+0.021 [−0.032, +0.075]). Scores, held-out / familiar: CAM 7.0 0.180 / 0.285, without residual 0.179 / 0.251, untrained
0.159 / 0.114, `model_based` untrained 0.222 / 0.261, trained 0.175 / 0.287, exact regression 0.164 / 0.262, CAM 6.1
adaptive (decisive checkpoints, evaluated at `feccc86` in a separate worktree) −0.054 / 0.013, hold −0.001 / −0.004.
CAM 7.0 − CAM 6.1 adaptive: about +0.23 / +0.27. Adaptation curves (familiar): trained CAM 7.0 loses much less while
experimenting (k=2: −0.13 vs −0.30 `model_based`, −0.47 exact regression) and ends highest (k=64: 0.397), but exact
regression is ahead in the middle (k=16: 0.326 vs 0.191).

**C-3 result (run once, 2026-11-24, `runs/cam7/confirm/exp2`):** all three failed. C-3a KG − Thompson +0.039
[−0.024, +0.104]; C-3b KG − `model_based` trained +0.047 [−0.047, +0.140]; C-3c full − `model_based` trained −0.004
[−0.102, +0.085]. Reported without a criterion (same model, only the choice of experiments differs): KG − greedy +0.211
[+0.114, +0.303], KG − EIG +0.099 [+0.011, +0.188], KG − fixed design +0.192 [+0.098, +0.287], KG − random +0.228
[+0.133, +0.328]; learned residual on top of KG −0.051 [−0.107, +0.005]. Episode scores: KG 0.410, Thompson 0.371,
`model_based` 0.363, full 0.359, EIG 0.311, fixed design 0.217, greedy 0.198, random 0.181.

**C-4, Experiment 3, parametric transfer (declared 2026-11-24, before the run).** Crucible's `protocols.transfer`
unchanged (`superlearner.cam7.experiments.exp3`), CAM 7.0 `small` full, seeds 0, 1, 2 (fresh trainings): source = the six
training families, 240k steps; adaptation budget 120k on each target's training stream, from the source checkpoint
("pretrained") and from scratch (target-only control), same stream, same evaluation tasks; evaluation tasks at VAL offset
80,000, 8 per family (untouched). Targets: within (new seeds of the training families), compositional (`scm_delay_gate`),
structural (`abm_trust_gate`, `scm_invu`), cross-implementation (`tech_pipeline`). Claim per target, standing alone:
the area under the adaptation curve, pretrained − scratch, paired over (seed, task), has its 95% lower bound > 0
(the protocol's own interval). C-4a within, C-4b compositional, C-4c structural, C-4d cross-implementation. Zero-shot
gains and steps to threshold reported without a criterion; `model_based` through the same protocol as the baseline's
own transfer, without a criterion.

**C-3, Experiment 2, active causal discovery (declared 2026-11-24, before any look at the confirmation set).** Set:
`x2_confound_gate` and `x2_regime_mix` at VAL offset 70,000, 12 tasks each (24), never used. Same three CAM 7.0
checkpoints; budget 24 ticks; `superlearner.cam7.experiments.exp2`. Statistic: whole-episode score, paired per task,
two-level bootstrap, 95%; lower bound > 0 to pass:
- C-3a decision-relevant value of information beats posterior sampling with the same model: KG rule (no residual) −
  Thompson (dev: +0.068 [−0.010, +0.149]);
- C-3b CAM 7.0 (KG rule) − `model_based` trained, 3 seeds (dev: +0.162 [+0.076, +0.246]);
- C-3c full CAM 7.0 (KG + learned residual) − `model_based` trained (dev: +0.103 [+0.031, +0.180]).
Also reported without a criterion: greedy, EIG, random, fixed design; commit score, probe rank correlation and regret,
experimentation cost and the number of distinct settings tried.

**C-1, learned priors for few-shot effect prediction (declared 2026-11-24, before any look at the confirmation set).**
Set: the four held-out families (`abm_trust_gate`, `scm_delay_gate`, `scm_invu`, `tech_pipeline`) at seeds
1,050,000–1,050,029 (120 tasks), never used by any experiment. Learner: `blr_meta_taskscale` (fixed move features plus
learned features, meta-trained on the six training families with each task's full-data spread in the loss), 3 seeds,
3,000 steps, retrained from scratch. Baseline: `blr_fixed` with the settings tuned on the training families. Interval:
two-level bootstrap over seeds and tasks, 95%.
Passes only if all three hold: (1) effect error, learned minus fixed, has an upper bound below 0 at k = 8;
(2) the same at k = 16; (3) ranking, learned minus fixed, has a lower bound above −0.02 at k = 16 (not worse).
Otherwise the claim "meta-learned priors improve few-shot effect prediction on unseen mechanisms" is recorded as failed.
**Result (run once, 2026-11-24, `runs/cam7/fewshot/condition_a_confirm.json`): PASSED.** Effect error k=8 −0.135
[−0.206, −0.070], k=16 −0.189 [−0.285, −0.107]; ranking k=16 +0.018 [−0.003, +0.039]. Not part of the criterion: ranking
at k=4 −0.033 and k=8 −0.014 (point estimates), so learned priors size effects better but do not yet order
interventions better. Scope: identical support sets (condition A), prediction only, not control.

### Active

None. (Both jobs previously listed here finished; their results are in the Completed table.)

## Coverage of the CAM 7.0 specification (audit, 2026-11-24)

**CAM 7.0 as specified has not been built.** The branch is CAM 6.1's code with a new memory core and a better
standalone few-shot learner. CAM 7.0's own six commits changed about 1,070 lines of code and tests, about 600 of them
diagnostics and benchmark harnesses, against about 3,400 for CAM 6.1. Work so far covers the spec's Phases 0–2;
Phase 3 (the integrated backbone) has not started.

| Spec part | State |
|---|---|
| 0 Recover CAM 6.1 state | Done |
| 1 Diagnose CAM 6.1 failures, memory by layer | Done for memory, conditioning and the adaptive layer's loss; delays 0/4/16 (focused) and up to 32 (screening), not 64 |
| 3.2 Event tokens | Minimal: one token per step from the pooled observation, previous action and elapsed time. No per-observation, entity, prediction, outcome, cost, reliability or provenance tokens. Width 128 in `cam7_small`; the 256-wide `cam7` preset is untrained |
| 3.3 Shared adaptive features φ = F(H, a, Δt) | Not done: the adaptive layer's features are a small separate network on lever settings, not computed by the backbone |
| 3.4 Fast Bayesian inference | Done as a separate layer (float64, Cholesky, learned prior, precision and noise). Single reward outcome only; no multiple outcome heads or non-Gaussian likelihoods |
| 3.5 Posterior integrated into the backbone | **Not done**: the posterior still acts mainly as a planner bonus (`runtime/learner.py` `_plan_bonus`), which the spec rules out; only three per-candidate summaries enter the action encoder, and no posterior tokens enter the backbone |
| 3.6 Nonlinear, interacting, nonstationary mechanisms; change points; latent context | Not done |
| 4.1 Support/query meta-training | Done for the standalone few-shot learner (C-1 passed); not applied to the agent's backbone |
| 4.2 Direct training of evidence retention; 4.3 | Not done |
| 5 Intervention-consequence prediction | Effect heads inherited from CAM 6.1 (held-out effect R² ≤ 0.29); no evidence-quality or identification work |
| 6 Value-of-information exploration, coherent uncertainty, imagination | Not done (Thompson bonus from CAM 6.1 only; imagination off in the `cam7` presets) |
| 7 Credit assignment | V-trace from CAM 6.1; delayed-reward credit assignment fails (`cue_final`) |
| 8 General-purpose OpenWorld actions | Not done |
| 9 Continual parametric learning | Not done |
| 10 Experiments 1–4 | Only Experiment 1 condition A |
| 13 Deliverables and the 14 questions | Not written; Q1, Q2 and part of Q3/Q4/Q6 answered in this ledger |

## Hypotheses (status)

| Hypothesis | Status | Evidence |
|---|---|---|
| Learner lacked an efficient way to fold in new evidence | Supported (development) | adaptive layer +0.221 vs +0.036 |
| Reward scale hid small effects | Supported | per-tick reward sd 0.002–0.06; effect R² rose after standardisation |
| Policy chased critic errors | Supported | expected-Q objective collapsed on abm_utility |
| Information lost before the policy belief (memory) | **Supported** | sampled latent 0.50 vs posterior logits 0.88–1.00 at the step the cue appears; RSSM transition fails by delay 4 even with direct supervision; attention keeps it to 16 |
| Stochastic latent bottleneck discarded evidence | **Supported** | as above; probabilities instead of samples recover the latent (0.97) but not the transition |
| Stop-gradient kept RL from shaping the belief | Weakened | shared gradients did not fix the cue task |

## Next steps

1. Agent-level credit assignment under delayed reward (`cue_final`): check whether the critic's λ-return reaches the
   first decisions, whether the value head learns the terminal reward from the cue, and whether an auxiliary
   "predict the episode's final reward" target makes the belief keep the cue (prediction worked for retention in the
   probe; the RL signal alone does not).
2. CAM 7.0 development run collapsed late (+0.110 → −0.166 at 121k); rerun with FeatureNorm and small-init heads,
   then diagnose the collapse (actor entropy, critic scale, adaptive gate) before any held-out comparison.
3. Experiment 1 condition B (each learner chooses its interventions), then Experiment 2 (active discovery), with
   pre-declared criteria and untouched seeds, against exact regression and `model_based`.
4. Experiments 3 (parametric transfer and continual learning) and 4 (OpenWorld) are not started.

## Status of the four mandatory experiments

| Experiment | Status |
|---|---|
| 1 Few-shot adaptation | Condition A done: learned priors beat generic regression on effect size (C-1 passed); ranking not better. Condition B not run |
| 2 Active causal discovery | Not run |
| 3 Parametric transfer and continual learning | Not run |
| 4 OpenWorld long-horizon transfer | Not run |
