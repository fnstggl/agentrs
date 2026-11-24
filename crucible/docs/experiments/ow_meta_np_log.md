# meta_np configurations tried on development tasks (validation seeds 1,000,100-111 only)
a: pure NP, dwell 4, 60k: untrained -0.077, trained 0.228 (36 tasks), 0.231 (72 tasks)
sweep of 11 acting settings on a: 0.210-0.248, no setting beyond noise, defaults kept
b: pure NP, dwell 2, 60k: -0.009 -> 0.260
c: pure NP, dwell 4, 200k: -0.040 -> 0.234 (0.234, 0.272, 0.234 at 67k, 133k, 200k)
d: ridge + correction, dwell 2, 60k: 0.167 -> 0.241
e: ridge + correction, dwell 4, 60k: 0.154 -> 0.243
d, refined search: 0.285 -> 0.316
e, refined search: 0.200 -> 0.308
baselines on the same 72 tasks: probe_exploit 0.191, model_based 0.273
f: + previous-settings input, dwell 2, 60k: 0.285 -> 0.267
g: + previous-settings input, dwell 4, 60k: 0.200 -> 0.237
correction variants on f / g: mean+spread 0.267/0.237, mean only 0.344/0.276, spread only 0.186/0.111, half both 0.232/0.213, none (trained regression settings) 0.230/0.206
decision from development tasks: dwell 2, act on corrected mean with regression spread, regression settings frozen
h: frozen regression settings, corrected mean with regression spread, dwell 2, 120k: untrained 0.285; trained 0.293, 0.321, 0.272 at 40k, 80k, 120k. No reliable training gain; the 0.344 of f/mean-only did not replicate.
prediction check on h (development seeds, 18 family/context cells): corrected z-RMSE below regression in 18/18, rank correlation above in 18/18, pick regret below in 18/18. Diagnosis: the decision rule, not the model.
decision rule: predicted improvement over best arm, per-episode calibrated spread (h checkpoint, 72 tasks): untrained 0.258, trained 0.317 (spread off), 0.317 (spread on); paired gain +0.059 +/- 0.064
sweep2 (10 acting settings, 144 development tasks, h checkpoint, adoption rule fixed in advance): none adopted. Default: untrained 0.248, trained 0.296, paired gain +0.048 +/- 0.039. Baselines on the same tasks: probe_exploit 0.185, model_based 0.290.
DESIGN FROZEN at this point (2026-11-23 ~11:00 UTC). No acting or architecture change after this line.
limitation found after the freeze (synthetic diagnostic): a pure two-lever product with no marginal effect is not learned (corrected/regression MSE 0.99-1.36 after 1,000 updates, both network sizes). Architecture not changed in response.
limitation (synthetic): a saturating curve whose size differs per lever and per world is not learned from 12 arms (ratio 0.96-1.39 after 1,200 updates).
naive-predictor check on h: see pilot_h_final/ckpt.pt.predcheck.json.
longer synthetic check (default network, 3,000 episodes, ratio of corrected to regression error at 1k/2k/4k/8k updates): shared curve 0.98/0.98/0.03/0.02; pure two-lever product 1.01/0.99/0.87/0.16; per-world saturation 1.00/1.03/1.01/1.01. CORRECTION: the product is learned at realistic budgets, it is not an architecture limit. Only per-world saturation remains unlearned at 8,000 updates.

# PRE-DECLARED CRITERIA (written 2026-11-23 after training ended, before any validation or sealed score of meta_np was seen)
Estimate: per-task mean over the 3 trained seeds. Inference: paired over tasks, 95% interval.
C1 (the stated bar): trained meta_np beats probe_exploit, interval excludes 0.
C2 (gain is learned): trained meta_np beats its own untrained weights, interval excludes 0.
C3 (stronger): trained meta_np beats model_based (both its untrained and trained rows), interval excludes 0.
The positive control PASSES only if C1 and C2 both hold on validation AND on the new sealed set (Holm-corrected over C1, C2 on the sealed set). C3 is reported either way.
Nothing about the learner changes after this point. The sealed set is opened once, whatever validation shows, unless validation fails C1 or C2, in which case it is not opened and the result is reported as failed.

# OUTCOME (2026-11-23)
Validation, trained (mean of 3 seeds) minus reference, paired 95% interval:
C1 scripted experimenter +0.184 [+0.129, +0.240]: holds.
C2 own untrained weights +0.048 [-0.006, +0.101]: does not hold.
C3 model_based untrained +0.050 [-0.002, +0.102], trained +0.052 [+0.003, +0.106]: does not hold (both are required).
Verdict: FAILED. Under the rule above, no sealed set was opened. `ow-final` was not touched.
Held-out families: trained minus own untrained -0.065 [-0.118, -0.014].
Full record: ow_meta_np_result.json.
