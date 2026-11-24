# CAM 7.0: results

Branch `feat/cam-7.0`. Research ledger with every experiment, configuration and next step:
[`CAM7_LEDGER.md`](CAM7_LEDGER.md). Code: `superlearner/src/superlearner/cam7/`. Raw results: `runs/cam7/`
(git-ignored, ephemeral disk). Scores: 0 = holding the default settings, 1 = the task's private reference policy.
Intervals: 95%, two-level bootstrap over training seeds and tasks, paired per task.

Status words: **implemented** (code exists and runs), **tested** (unit or development test), **demonstrated** (a
criterion written down before the run passed on untouched tasks), **inconclusive**, **failed** (a pre-declared
criterion failed), **not attempted**.

## The learner

One shared transformer (the `small` configuration: width 128, 2 layers, 4 heads, 0.53M parameters) reads structured
tokens: one per lever, one per gauge, one global token and up to 64 recent event tokens. Each lever and gauge carries a
random per-episode identity, so nothing is tied to a slot.

1. **Pass 1** produces a context for every lever.
2. **Adaptive features** for every observed row and candidate intervention are computed from that context:
   `phi = F_theta(H, a, dt)`. They consist of fixed `[d, d^2, e, e^2]`, learned features, and learned pairwise interactions.
3. **Fast belief**: an exact normal-inverse-gamma regression on those features, one per outcome (revenue and cost). It
   is float64, uses Cholesky factors, gives a Student-t predictive, and has meta-learned priors, noise and forgetting.
   With the noise known, it equals the Kalman recursion in the specification (tested).
4. **Pass 2** of the same transformer reads the posterior as tokens: each lever's expected effect and its uncertainty,
   noise level, evidence and recent innovations.
5. **Candidate interventions**:
   - hold;
   - posterior-mean optimum and posterior-sample optima;
   - single-lever probes;
   - random moves.
6. **Decision score**:
   `beta * [posterior mean value + remaining decisions * knowledge gradient] / noise sd + learned residual`.
   The knowledge gradient is the decision-relevant value of information: `E_y[max_a' Q(updated belief)] - max_a' Q(current belief)`.

**Objectives:**
- support/query meta-NLL, which trains the features, priors and pass 1;
- world auxiliaries;
- a clipped actor-critic on actual outcomes, with the posterior as a fixed input (a deliberate gradient boundary).

**Replay:** episodes are stored with their decisions, candidate sets, behaviour probabilities and model version, and
saved beside the weights.

**Cost:** a 240k-step training run takes about 10 minutes, against 3.7 hours for CAM 6.1.

## Pre-declared confirmations (each written into the ledger and committed before its set was touched)

| ID | Claim | Result | Verdict |
|---|---|---|---|
| C-1 | Learned priors improve few-shot effect prediction (condition A, standalone regression learner) | k=8 −0.135 [−0.206, −0.070], k=16 −0.189 [−0.285, −0.107] | **demonstrated** |
| C-2a | Familiar families: trained CAM 7.0 beats its untrained self | +0.172 [+0.094, +0.261] | **demonstrated** |
| C-2b | Familiar: CAM 7.0 beats `model_based` | +0.024 [−0.018, +0.067] | failed |
| C-2c | Held-out families: CAM 7.0 beats `model_based` | −0.042 [−0.100, +0.010] | failed |
| C-2d | Held-out: trained beats untrained (transfer) | +0.021 [−0.032, +0.075] | failed |
| C-3a | Value of information beats Thompson sampling with the same model | +0.039 [−0.024, +0.104] | failed |
| C-3b | CAM 7.0 (KG) beats trained `model_based` on active discovery | +0.047 [−0.047, +0.140] | failed |
| C-3c | Full CAM 7.0 beats trained `model_based` on active discovery | −0.004 [−0.102, +0.085] | failed |
| C-4a | Experiment 3 within families: pretrained adapts faster than scratch (area) | +0.085 [+0.054, +0.120] | **demonstrated** |
| C-4b | Compositional (`scm_delay_gate`) | −0.099 [−0.156, −0.043] | failed (negative transfer) |
| C-4c | Structural (`abm_trust_gate`, `scm_invu`) | −0.023 [−0.047, −0.002] | failed |
| C-4d | Cross-implementation (`tech_pipeline`) | −0.043 [−0.102, +0.018] | failed |

## The fourteen questions

**1. What caused CAM 6.1's poor performance?**
Several causes, each diagnosed one hypothesis at a time (ledger):
- *Memory:* the sampled stochastic latent discarded evidence the posterior held, and the RSSM transition lost it within
  about four steps.
- *Adaptive layer:* its meta-training loss rewarded inflating variance.
- *Conditioning:* belief and action features were badly conditioned, and zero-initialised critics learned slowly.
- *Event core:* it could not read the current step.

Beneath these, CAM 6.1 had no exact way to fold new evidence into decisions, and no decision-relevant exploration. Its
policy learned from replay against an inaccurate critic.

Replacing that with exact Bayesian inference, value-of-information experimentation and the integrated network raised
scores on untouched tasks:

| | CAM 6.1 adaptive (its decisive checkpoints) | CAM 7.0 |
|---|---:|---:|
| Held-out families | −0.054 | 0.180 |
| Familiar families | +0.013 | 0.285 |

That comparison was declared as reported, not as a pass/fail criterion. *Status: tested; the CAM 6.1 → CAM 7.0 gap is
large and consistent.*

The same diagnosis from the other side: the CAM 7.0 network with the explicit posterior removed, trained the same way by RL, scores −0.195 held-out and −0.282 familiar on the development sets (one seed). That is −0.37 and −0.57 below CAM 7.0, and below holding the defaults. Removing the adaptive posterior degrades control decisively (specification 3.5).

**2. Did its latest ALPaCA-inspired changes solve the problem?** No.
- Decisive run: −0.021 on held-out families, against +0.170 for `model_based`.
- On the untouched C-2 set, its checkpoints score −0.054 held-out and +0.013 familiar.

*Status: failed.*

**3. Does CAM 7.0 incorporate new information more efficiently than the existing Bayesian baseline?** Not demonstrated.
On untouched tasks it is level with trained `model_based` (+0.005 held-out, −0.002 familiar) and with exact regression
(+0.016 / +0.023). C-2b and C-2c failed; on held-out families untrained `model_based` is ahead (−0.042).

The adaptation curves differ in shape. On familiar families:
- **Early experimentation (after 2 observations):** trained CAM 7.0 scores −0.13, against −0.30 for `model_based` and
  −0.47 for exact regression.
- **End (after 64 observations):** CAM 7.0 is highest, at 0.397.
- **Middle (after 16 observations):** exact regression is ahead, 0.326 against 0.191.

*Status: inconclusive, leaning no.*

**4. Does learned representation improve few-shot adaptation?**
- *Unseen instances of the training families:* yes. Trained beats untrained by +0.172 (C-2a), with the largest gains in
  early windows (after 4 observations: −0.13 against −0.44).
- *Prediction alone (condition A):* learned priors size effects better (C-1).
- *Unseen families:* not demonstrated (C-2d +0.021 [−0.032, +0.075]).
- *Learned against fixed features:* one development seed only, held-out 0.155 vs 0.173. Inconclusive.

*Status: demonstrated within families; inconclusive across families.*

**5. Does the model learn when experiments are valuable?** It *computes* when experiments are valuable; it has not
*learned* to. Against Thompson sampling, the pre-declared criterion failed: +0.039 [−0.024, +0.104].

With the same model and posterior, choosing experiments by decision-relevant value of information beat the other rules
on the untouched set (reported comparisons, not criteria):

| Value of information compared with | Difference |
|---|---|
| Greedy | +0.211 [+0.114, +0.303] |
| Decision-blind information gain | +0.099 [+0.011, +0.188] |
| Fixed one-lever-at-a-time design | +0.192 [+0.098, +0.287] |
| Random experiments | +0.228 [+0.133, +0.328] |

Information value itself didn't predict information quality early on. At 24 ticks, the probe rank correlation was
0.48–0.50 for every rule except random (0.495). What the rule did was turn information into outcomes.

The learned residual, which is meant to learn when exploration is too costly, hurt: −0.051 [−0.107, +0.005].

*Status: decision-relevant VOI tested and better than decision-blind alternatives; learned exploration failed.*

**6. Does it remember consequential evidence over long horizons?**
- *Mechanism evidence:* the posterior's sufficient statistics keep all of it exactly (with learned forgetting); the
  attention window covers 64 events.
- *Arbitrary delayed cues:* the agent-level delayed-reward cue task failed in the CAM 6.1 code path (ledger), and was
  not rerun with the CAM 7.0 learner.

*Status: by construction for mechanism evidence; not demonstrated for arbitrary cues.*

**7. Do its core weights acquire persistent transferable capabilities?** Persistent within families: yes. C-2a used
checkpoints evaluated in a fresh process, frozen, with fresh context and no replay. Experiment 3: within families, a source-trained network adapts faster than a fresh one on the same target stream (+0.085 [+0.054, +0.120], C-4a). *Status: demonstrated within families.*

**8. Does experience in one environment improve learning in an unfamiliar one?** No.
- *Zero-shot:* C-2d failed.
- *Adaptation speed (Experiment 3):* pretraining on the six source families did not speed up learning on any unfamiliar category. It slowed learning on the compositional family `scm_delay_gate`: area −0.099 [−0.156, −0.043]; trained from scratch on that family CAM 7.0 reaches 0.437, from the source checkpoint 0.337.
- *Baseline:* `model_based` shows no transfer either.

*Status: failed (negative transfer on one category).*

**9. Does model-based imagination improve actual outcomes?**
- *One-step imagination:* the value-of-information rule imagines an experiment's outcome, updates the belief and
  re-decides. It beats decision-blind and greedy rules (question 5).
- *Multi-step latent imagination:* not attempted in CAM 7.0.

*Status: one-step tested; multi-step not attempted.*

**10. Does the learner outperform frozen language-model agents with matched tools and budgets?** *Not attempted.*

**11. Can it operate in the full open-ended Crucible world?** *Not attempted.* There is no OpenWorld adapter for CAM
7.0 yet; Experiment 4 was not run.

**12. Does prior learning improve consequential long-horizon success?** *Not attempted* (Experiment 4).

**13. What is the actual sample and compute efficiency?**

*Training (240k environment steps, one A6000 shared by several jobs):*

| Learner | Parameters | Updates | Training compute | Inference compute |
|---|---:|---:|---|---|
| CAM 7.0 small | 0.53M | 992 | 208–280 GPU-s | 160–226 GPU-s |
| `model_based` | 1,426 | 1,984 | 575 CPU-s | 92 CPU-s |
| CAM 6.1 adaptive | 3.6M | 6,312 | 7,379 GPU-s | 4,213 GPU-s |

*Evaluation (48 held-out tasks, one pass):*

| Learner | Search evaluations per task | Inference compute |
|---|---:|---|
| CAM 7.0 | 37k | 28 GPU-s |
| `model_based` | 3.5k | 2.4 CPU-s |
| Exact regression | 16k | 167 CPU-s |

CAM 7.0 uses about ten times `model_based`'s search evaluations for equal outcomes. Extra search is not sample efficiency.

**14. What remains unproven?**
- An advantage over the strongest baselines: `model_based` and exact regression.
- Transfer to unseen mechanism families.
- Learned (rather than computed) exploration.
- Long-horizon memory of arbitrary evidence.
- Multi-step imagination.
- OpenWorld operation (Experiment 4).
- Comparisons with frozen language-model agents.
- Distributed actors.
- Sequential multi-domain retention.

## Experiment 3 (parametric transfer)

Crucible's `protocols.transfer`, three seeds, source 240k steps on the six training families, then 120k steps on each target stream from the source checkpoint and from scratch. Evaluation tasks are untouched (VAL offset 80,000).

| Target | CAM 7.0 zero-shot gain | CAM 7.0 area, pretrained − scratch | `model_based` area, pretrained − scratch |
|---|---|---|---|
| Within | +0.163 [+0.109, +0.222] | +0.085 [+0.054, +0.120] | −0.017 [−0.042, +0.006] |
| Compositional | −0.032 [−0.113, +0.046] | −0.099 [−0.156, −0.043] | +0.004 [−0.045, +0.056] |
| Structural | +0.007 [−0.036, +0.047] | −0.023 [−0.047, −0.002] | −0.006 [−0.029, +0.017] |
| Cross-implementation | −0.028 [−0.188, +0.130] | −0.043 [−0.102, +0.018] | −0.023 [−0.087, +0.042] |

No arm reached the goal threshold, so steps-to-threshold is undefined.

The specification's decisive metric is how much less experience is needed in unfamiliar worlds after experience elsewhere. On that metric CAM 7.0 has **not** succeeded: its learning stays inside the families it learned from.
