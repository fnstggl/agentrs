# CAM 7.0 research ledger

Read this first in every session. Newest entries at the top of each section. Results are only listed as
"demonstrated" when a pre-declared comparison on untouched tasks supports them.

## Where things are

| Item | Location |
|---|---|
| CAM 7.0 branch | `feat/cam-7.0` (local; from `feat/cam-6.1` `feccc86` + crucible-4 `c07098b` + superlearner-4 `58c18ce`) |
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

## Experiments

### Completed

| Date | Experiment | Result | Verdict |
|---|---|---|---|
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

| Started | Job | Output |
|---|---|---|
| 2026-11-24 | CAM 7.0 development run, `cam7_small` with fixed event core, seed 0, familiar families only, 120k | `runs/cam61/cam7_dev_small_v2/`, log `runs/cam61/logs/cam7_dev_small_v2.log` |
| 2026-11-24 | Agent cue task with fixed event core (lr 3e-4), window 16 vs window 1, 5 seeds | `runs/cam7/memory_agent_fixed_w*.json` |

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

1. Collect the adaptive decisive rerun and record it here.
2. Memory: layer-by-layer probes at delays 0, 1, 4, 16, 32, 64 for GRU, attention, RSSM, RSSM + slots, under supervised
   and world-model-only training (`superlearner/src/superlearner/unified/diagnostics/memory_probe.py`).
3. Experiment 1, condition A: identical support sets for exact regression, model_based, meta-trained regression, CAM.
