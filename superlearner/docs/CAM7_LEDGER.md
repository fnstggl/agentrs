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

### Active

| Started | Job | Output |
|---|---|---|
| 2026-11-24 | Decisive rerun: adaptive full / no planner / no hand evidence / no world model, 3 seeds, 240k | `runs/cam61/decisive_adaptive_*`, report via `python -m superlearner.unified.experiments.decisive report` |

## Hypotheses (status)

| Hypothesis | Status | Evidence |
|---|---|---|
| Learner lacked an efficient way to fold in new evidence | Supported (development) | adaptive layer +0.221 vs +0.036 |
| Reward scale hid small effects | Supported | per-tick reward sd 0.002–0.06; effect R² rose after standardisation |
| Policy chased critic errors | Supported | expected-Q objective collapsed on abm_utility |
| Information lost before the policy belief (memory) | Open | probe at step 0 fails in recurrent agent |
| Stop-gradient kept RL from shaping the belief | Weakened | shared gradients did not fix the cue task |

## Next steps

1. Collect the adaptive decisive rerun and record it here.
2. Memory: layer-by-layer probes at delays 0, 1, 4, 16, 32, 64 for GRU, attention, RSSM, RSSM + slots, under supervised
   and world-model-only training (`superlearner/src/superlearner/unified/diagnostics/memory_probe.py`).
3. Experiment 1, condition A: identical support sets for exact regression, model_based, meta-trained regression, CAM.
