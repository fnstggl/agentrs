# U-X1 — Experiment 1: one-shot and few-shot adaptation (pre-registration)

Spec Part 17 Experiment 1, Part 19 rigor; manifest X01 (and X05 baselines 4, 9-13 where applicable). Written and
committed **before** any confirmation-offset evaluation (offsets 200,000+). Code:
`superlearner/src/superlearner/cam_unified/experiments/` (`exp1.py`, `predict.py`, `baselines.py`, `mechrun.py`,
`stack.py`, `train.py`). Raw rows: `/workspace/agentrs/runs/cam_unified/wave2/X/exp1/<set>/`.

## System under test

The integrated CAM Unified stack at dev scale (8 layers x 384, 32 state tokens; ~80.7 M CAM parameters, Qwen3-0.6B
frozen semantic encoder counted separately) built by `model.build_model` over the real components of workstreams
A-E, wrapped in `agent.CamUnifiedAgent`, trained by `experiments.train` (D's `CombinedTrainer`, stage "continual":
world + support/query adaptation + memory + actor/value objectives, every update from E's archive/replay and the
fresh on-policy buffer) on a SplitGuard-checked stream of real Crucible mechanism worlds (domain `scm`: Crucible's 3
scm training families + the catalogue's 34 scm training families) through G's `MechWorldAdapter`. Declared
configuration choices are listed in `stack.py` (Qwen3-0.6B, decoder LoRA off, batched learner path, fixed J = 4
reasoning iterations, decoder acting during training). Training: seeds 0, 1, 2, each two locked chunks of 22 min
(dev scale; the decision count reached is reported), episodes presented plain (50%), with predecessor notes (25%)
and with hidden gauges + delayed money (25%). The final checkpoint of each seed is evaluated **weights-only in a
fresh process** (`load_weights_only`, `reset_experience_access`: no optimiser, archive, replay, memory index, fast
state or counters).

## Conditions

* **A identical support** - every learner conditions on the same stream: a seeded design policy sets one random
  lever to a random value per decision (one tick per decision); after k in {0,1,2,4,8,16,32,64} support steps the
  world is forked (evaluator side) for 12 query interventions (hold + 11 random single-lever moves) and the true
  next-tick reward of each is recorded. Queries never enter any learner's input.
* **B agent-selected experiments** - each learner acts itself for whole episodes; decisions are scored per window of
  ticks [0,1), [1,2), [2,4), ..., [64,128) (normalised score: 0 = defaults held, 1 = the task's reference policy),
  plus the episode score and goal success (`Task.success`). CAM additionally gets the same forked queries along its
  own trajectory (prediction after k self-chosen observations).
* **C semantic observations** - as A, plus 4 natural-language notes of a previous operator delivered at reset (real
  single-lever experiments run on a separate earlier instance of the same facility; only public quantities). Only
  text-reading learners can use them; `nig_exact_parsed_notes` (the notes' numbers added to an exact regression) is a
  numeric reference of their information content.
* **D delayed and partially observable** - as A through `DelayedPartial`: half the gauges never reported, money and
  the verified reward of each tick delivered 3 ticks late (numeric learners get the same masking).

## Sets and seeds

Families: familiar = {scm_additive, scm_delay, scm_gate, cw1_scm_confounder, cw1_scm_interaction,
cw1_scm_delay_stock, cw1_scm_feedback_gate, cw1_scm_regime_threshold} (training families, new instances); held-out
= {scm_delay_gate, scm_invu, cw1_scm_delay_gate, cw1_scm_feedback_threshold, cw1_scm_interaction_stock,
cw1_scm_invu, cw1_scm_invu_delay, cw1_scm_gate_regime_threshold} (validation families: compositional and
structural holdouts). Validated tasks at Crucible `VAL_SEED0 + offset`:

* development: offset **100,000**, 2 tasks per family (32 tasks), design seed 11 - inspected freely;
* confirmation: offset **200,000**, 3 tasks per family (48 tasks), design seed 23 - evaluated once, after this file
  is committed, with the checkpoints and baselines fixed below.

Training streams use training seeds only (CAM: 100,000 x (seed+1) [+50,000 for abm]; baseline design episodes
900,000 + 10,000 x seed; Crucible numeric learners 600,000 + 10,000 x seed).

## Learners and baselines

Prediction (A, C, D): `cam` (trained, seeds 0-2), `cam_untrained` (fresh weights, seeds 0-2; spec baseline "without
learning"), `nig_exact` (exact Bayesian NIG regression on per-lever [d, d^2, e, e^2] features, generic prior; spec
baseline 4), `nig_meta` (same with a meta-learned prior = strong Bayesian / ALPaCA, seeds 0-2), `seq_gru` (recurrent
meta-learner), `seq_attn` (attention meta-learner / transformer neural process); the three learned baselines are
fitted on 320 training-family design episodes per seed (plain + condition D presentations).
Decisions (B): `cam`, `cam_untrained`, Crucible `model_based` (meta-learned Bayesian planner) trained on the same
families, `ppo_gru` (recurrent meta-RL), CAM 7.0 (`cam7`, KG rule), `probe_exploit` (scripted), `random`, `hold` -
as far as compute permits; numeric learners train for 20,000 ticks each on the training stream.
Not run here: the frozen semantic agent with tools (G's `FrozenSemanticAgent` drives OpenWorld record conversion;
see results for the reason), CAM 6.1.

## Measures

Per (learner, training seed, task, condition, k): mean log predictive density of the 12 query rewards (proper
scoring rule, primary), squared error, 90% interval coverage, and decision regret from the same queries (true reward
lost by acting on the predictive's best candidate, ties broken uniformly, normalised by the task's (reference -
default) net). Learning-curve area = mean over the 8 grid points k. Condition B: window scores, episode score,
success, area over windows.

## Primary criteria (confirmation set)

1. **X1-P1 (prediction improves with training)**: log-score area, condition A, familiar families:
   `cam` - `cam_untrained` > 0, two-level 95% interval excluding 0.
2. **X1-P2 (against the strong Bayesian baseline)**: same quantity, `cam` - `nig_meta` > 0, interval excluding 0.
3. **X1-P3 (control improves with training)**: condition B episode score on the familiar families,
   `cam` - `cam_untrained` > 0, interval excluding 0.

Everything else (held-out families, every k, regret, C, D, the other baselines) is reported descriptively with
intervals. Prediction improvement and control improvement are reported separately (spec: "do not confuse
prediction improvement with control improvement").

## Statistics

Two-level bootstrap (4,000 draws): resample training seeds, then evaluation worlds (paired: the same worlds for
every learner), mean over cells; 95% percentile intervals; differences formed per (seed, world) when both learners
have seeds 0-2, else against the other learner's seed mean.

## Compute budget

CAM training: 3 seeds x 2 locked chunks x 22 min (two processes per chunk, B = 8 environments each). Evaluation:
prediction (filter-based, ~1-3 min per checkpoint and condition), decisions (~1-1.6 s per round of 16 environments,
96-240 rounds per episode). Every locked GPU job stays under 25 minutes. Compute (wall, CPU, GPU-section seconds,
peak memory, environment steps, decisions, simulated days, updates) is recorded per run (`compute.json`,
`runs.jsonl`).
