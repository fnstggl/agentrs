# Causal Agency Model (CAM) — v1 vertical slice

CAM is a new, fully trainable goal-directed core. It does **not** use Qwen or LoRA: every parameter
is trained from scratch by the losses below. Qwen-LoRA (`agentrs.econ`) remains the baseline system.

## What exists (code: `src/agentrs/cam/`)

| module | what it computes | params (default config) |
|---|---|---|
| token embedding (shared, tied decoder output) | vocabulary of all environment grammars + numeric bins | 36 k |
| `obs_enc` (3-layer transformer) | observation tokens → 256-d | 2.47 M |
| `goal_enc`, `act_enc` (1-layer transformers) | goal / action tokens → 256-d | 0.87 M + 0.86 M |
| belief (GRU 512 + elapsed-time features) | b_t = f(obs_t, previous action, goal, Δt, b_{t-1}) | 2.02 M |
| dynamics (2×1024 MLP + heads) | (b_t, action, horizon, elapsed time) → next latent, Gaussian over observable features, event probabilities, Gaussian over verified progress, termination, success | 3.54 M |
| goal-value | expected return, P(verified success), expected remaining steps (Gaussian) | 0.66 M |
| strategy / skills | 8 learned options: π(option \| b, goal), termination β(b, option), option embedding | 0.40 M |
| action decoder (GRU) | autoregressive complete action (tool + every argument) under the env grammar | 1.84 M |
| aux reconstruction | current observable features from b_t (anti-collapse) | 0.14 M |
| **total** | **all trainable** | **12.82 M** |

Interfaces: `update_belief`, `predict_effects`, `option_dists` (choose/terminate skill),
`generate_action`, `action_logprob`, `estimate_value`; learning in `learner.update` (transitions and
rewards), checkpoints in `train.Trainer.save/load`.

## Environments (`src/agentrs/cam/envs/`) — executable, hidden parameters, environment-verified outcomes

* **negotiation** — 4-issue procurement deal; counterparty with hidden weights, reservation utility,
  patience, a concern that the right argument addresses, and one of 6 concession policies (train:
  boulware, conceder, linear, tit-for-tat; **test only**: deadline-driven, argument-anchored; test
  preferences are more peaked and reservations higher). Actions: ASK issue (information), PROPOSE
  four levels + argument, ACCEPT, WALK. Success: agreement with agent utility ≥ target (target is
  feasible by construction).
* **product** — company simulator: hidden segment sizes/needs/price sensitivity/channel affinity;
  RESEARCH (noisy interviews), BUILD (multi-week delay), SETPRICE, CAMPAIGN (4-week spend, lagged
  signups), ONBOARD, WAIT; churn depends on product–need fit; revenue funds the company. Success:
  retained active users ≥ target at week 26 without going bankrupt (target from a hidden-information
  expert × 0.45–0.7). **Test**: B2B-like markets.
* **bizdev** — secure 3 partner commitments within a revenue-share cap; hidden incentives,
  anchors/rivals, delayed replies and typed objections. **Never trained** in the main run: zero-shot
  test and learning-speed transfer target.

The agent sees only goal/observation tokens; tests assert that changing hidden parameters leaves the
agent's inputs unchanged. Rewards: +1 verified success, potential-based shaping 0.5·(γΦ' − Φ) on the
environment-measured progress Φ, −0.005 per action.

## Learning (all losses logged separately, `runs/cam/iters.jsonl`)

1. **Predictive**: matured multi-horizon targets (+1 action, within 3 actions, end of task) with real
   elapsed time; Gaussian NLL (features, progress), BCE (events, termination, success), cosine loss to
   the next belief (stop-grad), feature reconstruction. Held-out evaluation on validation episodes vs
   base-rate / constant baselines, calibration (ECE) and belief variance (collapse check).
2. **RL**: PPO + GAE over the factored decision π(term)·π(option)·π(action tokens); the grammar
   mask is part of the sampling distribution and is re-applied when re-scoring (measured mismatch
   ≈ 2e-7 nats); only episodes of the current version are used.
3. **Goal-value**: return regression, P(success) BCE, remaining-steps NLL.

## Runs and ablations

* `runs/cam` — full CAM (supervisor `agentrs_cam`), negotiation + product, validation gate every 10
  iterations, sealed test at the end on held-out distributions + bizdev zero-shot + planning-assisted
  acting (evaluation only).
* `runs/cam_nowm` — matched ablation C (identical seeds/config, `wm_coef: 0`).
* `runs/cam_transfer/{scratch,pretrained}` — bizdev learning speed from scratch vs from the main run's
  best checkpoint (supervisor `agentrs_cam_transfer`, starts at main-run iteration 150).
* Checkpoints: `runs/cam/ckpt/{initial,latest,best,snap_*,final}` (weights, AdamW, RNG incl. worker
  generator + seed counters, config, run state) → GitHub Releases via `agentrs_cam_backup`.

## Honest limitations of v1

* Hierarchy is two levels (learned options with termination → action decoder), not four.
* Information gathering is learned only through RL on its downstream payoff; there is no explicit
  value-of-information estimator yet.
* Planning (re-ranking candidates by imagined end-of-task success) is evaluation-only; it is not used
  to generate training data.
* Actions are structured token sequences from environment grammars; CAM has no open-vocabulary
  natural-language channel and no Qwen semantic bootstrap yet — Qwen plays no role in CAM v1.
* Simulators are hand-built; success in them is not evidence of real-world adoption or negotiation skill.

## Product-family fix (branch `feat/cam-product-fix`, run `runs/cam_pfix`)

**Diagnosis of the main run** (`scripts/cam_product_diag.py`, `scripts/cam_product_policy.py`). Product
validation success stayed at 0% in both the full run and the no-world-model ablation. At iteration 160 the
policy opens with max-budget campaigns ($20k/week) and goes bankrupt in 100% of validation episodes, after
6.6 weeks when sampling and 3.5 weeks when greedy. Training episodes shrank from 12.8 to about 5 steps by
iteration 16.

Cause:
- With Φ(terminal) = 0, every failed episode has the same shaped return, so the per-action step cost was
  the only signal among failures. It paid the policy to end episodes early, and bankruptcy ends them early.
- Uniform random play reaches the full target while staying solvent in only 0.2% of episodes, too rare to
  counter that signal.

The environment is solvable, measured over 300 validation seeds:
- A policy that knows the hidden market but must act through the real interface: 80%.
- A heuristic with no hidden information (research every segment, build the top 3 features, $3k/week ads,
  2 onboarding upgrades): 60%.
- The oracle policy with $20k/week campaigns: 6%, because 94% of episodes go bankrupt.

**Fix** (`configs/cam/product_fix.yaml`; otherwise identical to `a6000.yaml`):
1. `fail_horizon_cost`: a failed episode pays the step cost of its unused steps. Ending early is then never
   cheaper than failing at the deadline, and the step cost only rewards faster successes.
2. `curriculum.product`: the training target is scaled by a difficulty starting at 0.1. Difficulty rises
   ×1.2 when training success is above 50% and falls ×0.85 when it is below 15%. Validation and test targets
   are bit-identical to the main run (tested), so the reported numbers stay comparable.

The run starts when the ablation finishes (`deploy/cam/install_pfix.sh`). It is backed up as
`ckpt-cam_pfix-*` releases and reported by `scripts/cam_report.py`.
