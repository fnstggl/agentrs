# Why every LabWorld condition sits near uniform — engineering diagnosis (workstream S)

**Status: engineering diagnostic, not a scientific result** (Protocol v3, operator redirection 2026-11-27). Data:
`v2` main CPU batch, snapshot `00cb5a4` (includes C-10), debug scale (tiny random backbone), LabWorld F1,
schema-walker candidates (`use_backbone=False`), 10 seeds x 8 lifetimes x 16 decisions per condition (A: 2 seeds).
Evidence: `docs/evidence/S/diagnostics/v2_main_cpu_*.json` (bootstrap intervals over seeds). Raw records:
`docs/evidence/S/raw/eval_results_20261127.tar.xz`.

## Measured

| quantity (mean [95 % CI over seeds]) | E | D | C | uniform anchor | oracle anchor |
|---|---|---|---|---|---|
| verified progress, mean over 8 lifetimes | 0.372 [0.304, 0.440] | 0.363 [0.307, 0.420] | 0.314 [0.219, 0.412] | 0.277 [0.220, 0.341] | 1.000 |
| learning slope (2nd half - 1st half) | -0.009 [-0.073, 0.060] | 0.035 [-0.054, 0.143] | 0.060 [-0.086, 0.190] | — | — |
| true value of the chosen candidate / best candidate | 0.123 [0.107, 0.138] | 0.128 [0.117, 0.140] | 0.197 [0.141, 0.266] | — | — |
| true value of the candidate the agent's *own world model* ranks first / best | 0.295 [0.216, 0.391] | 0.386 [0.309, 0.458] | 0.197 [0.132, 0.277] | — | — |
| Spearman(predicted goal mean, true value) over candidates | 0.170 [0.075, 0.274] | 0.244 [0.135, 0.360] | 0.163 [0.056, 0.278] | — | — |
| expected adopters per decision on the recorded contexts, behaviour policy | 1.42 [1.20, 1.67] | 1.42 [1.25, 1.61] | 1.55 | uniform rule: 1.43 / 1.46 | — |
| same contexts, greedy on the agent's predicted goal mean (shadow, ground truth) | 5.05 [2.61, 7.94] | 6.83 [4.82, 8.77] | 1.74 | — | — |
| share of decisions on zero-value operations (probe, gauge, wait) | 0.64 [0.60, 0.67] | 0.64 [0.61, 0.66] | — | 9/14 of the set | — |
| share of positive internal reward paid to zero-value operations | 0.73 [0.60, 0.87] | 0.70 [0.57, 0.82] | — | — | — |
| Spearman(internal return of a lifetime, verified progress) | -0.17 [-0.32, -0.05] | -0.09 [-0.35, 0.17] | — | — | — |

## Findings

1. **The task is learnable and the candidate sets contain good actions**: the oracle anchor (same candidates,
   same budget) reaches every target (80/80 worlds); a rule as simple as "always the maximum-effort intervention"
   would yield 10.2 expected adopters per decision on E's recorded contexts against the behaviour's 1.42.
2. **The world model knows something; the selection rule does not use it.** Greedy selection on the agent's own
   registered goal predictions would have chosen candidates worth 3.6x (E) to 4.8x (D) the behaviour's choices
   on the same recorded contexts (ground truth, diagnostic world). The default rule `policy` samples from the
   trainable policy head, whose choices are indistinguishable from uniform (1.42 vs 1.43).
3. **The policy is trained toward zero-value actions.** The policy head is trained by the internal reward (the
   bootstrap mix stays 1.0 in every run: learning progress, epistemic-uncertainty reduction, grounded delta V).
   70–73 % of the positive internal reward is paid to probes, the noisy gauge and waiting, which never produce
   progress, and the internal return of a lifetime is *negatively* correlated with its verified progress in E.
   The pathway works mechanically (non-zero updates, meta-updates, conservation 0.0) but its objective is not
   aligned with verified progress at this scale and budget.
4. **No condition improves over lifetimes** (all slopes' intervals contain 0); learned intrinsic reward (E vs D,
   E_mix0) and verified supervision (E vs E_unverified) make no detectable difference — consistent with 2–3:
   whatever is learned does not reach the choices.

Filed as change request S-5 (`docs/change_requests/S.md`). These are debug-scale observations with a random
backbone and restricted candidates; they diagnose the selection / reward wiring, not the deployment intelligence.
