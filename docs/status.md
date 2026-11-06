# Phase status

For each phase: what was implemented, what was executed, which tests cover it, and what remains
unverified. "Executed" means run in the development sandbox (4 CPU cores, no GPU).

Test suite: `python -m pytest -q` — unit (actions, environments, diffs, storage, replay, returns,
metrics, registry, curriculum, memory, config, Gymnasium), learning (gradients, parameter updates,
loss decrease, checkpoint/resume, world model, RL, grounding, IDM, skills), safety, integration
(full loops, continual with separate processes, smoke experiment + report) and browser (real
Chromium). See the bottom of this file for the latest run.

## Phase 1 — Environment and experience harness: complete
* Implemented: `Environment` interface; MiniWeb (14 families, server-side verifiers, stale ids,
  scrolling, async loading, validation and stochastic errors); `LocalWebEnv` (real HTML +
  Playwright, 0 mismatches vs MiniWeb over 541 random steps); generic `PlaywrightEnv`;
  `BrowserGymEnv` (MiniWoB++); Gymnasium adapter; typed actions; `GuardedEnvironment`; belief
  state; episode runner; Parquet store with partitions, migration and blobs; replay buffer with
  leakage guard.
* Executed: thousands of episodes (teacher, random, policy, planner); trajectories written and
  reloaded losslessly; teacher solves all 14 families from public information.
* Exit criterion met: complete, reloadable trajectories with genuine observations and
  independently verified results.

## Phase 2 — Observational learning: complete
* Implemented: scripted teacher (public info only, ε-noise for imperfect demos); terminal and
  browser human recorders (consent mandatory, private by default); demonstration quality weights;
  BC trainer; checkpoints; held-out imitation and task evaluation; IDM training/labelling with
  confidence and abstention.
* Executed: BC on 40 demos/family (v2): weights change, imitation accuracy and task success on
  held-out tasks change measurably (numbers in `docs/experiments.md`).
* Exit criterion met. Unverified: real human demonstrations (no human in the loop here; the
  recorders are tested with programmatic input); IDM on real video (only structured observation
  sequences are supported — video-only action inference is not claimed).

## Phase 3 — World-model learning: complete
* Implemented: action-conditioned LM world model (diff NLL + bootstrap ensemble heads for events,
  reward, termination); training on demos + own experience; held-out evaluation with marginal /
  action-type / persistence baselines, calibration (ECE, reliability), uncertainty-error AUROC,
  multi-step drift.
* Executed: on held-out MiniWeb transitions the world model beats both frequency baselines on
  event NLL/AUROC and the persistence baseline on next-observation F1 (see experiments doc).
* Not implemented: a learned latent dynamics model behind the structured interface; multi-step
  training objective (multi-step is evaluated only); screenshot/visual modality.

## Phase 4 — Model-based planning: complete
* Implemented: MPC with top-K proposals, batched world-model rollouts, branching, critic leaf
  values, uncertainty penalty, action cost, compute budget, softmax/argmax/ε selection, behaviour
  log-probs, policy-only mode.
* Executed: planner evaluation on val/test/held-out families; evaluation-only sweeps over horizon
  and uncertainty penalty.
* Exit criterion met: the agent consults learned predictions (recorded per step in
  `Transition.wm_prediction`); planning can be toggled.

## Phase 5 — Real reinforcement learning: complete
* Implemented: REINFORCE with baseline, PPO (clip + GAE + KL early stop), GRPO; off-policy
  truncated importance weights; search distillation; critic training; invalid-action handling;
  KL-to-reference; divergence guards; optimizer-state checkpoints; standalone `train-rl` with
  resume.
* Executed: PPO from BC initialisation for 25 iterations × 32 episodes per seed (v2).
* Exit criterion met: policy weights update from verified rewards; the loop runs repeatedly
  without manual dataset reconstruction.

## Phase 6 — Imagined experience: implemented, measured (no benefit found)
* Implemented: bounded rollouts from real states, uncertainty cut-off, IMAGINED provenance
  (schema-enforced), separate partition, down-weighting, exploitation monitor.
* Executed: D+imagination for 2 seeds vs D without imagination; grounding test.
* Result: no measurable effect on real held-out success. Imagined 2-step rollouts almost never
  reached predicted success and the uncertainty threshold (0.08) was never reached by the actual
  ensemble spread (≈0.01) — the filter needs calibrating to the model's uncertainty scale.

## Phase 7 — Continual learning: complete at small scale
* Implemented: actor and evaluator as separate processes, learner with persistent ingestion
  state, SQLite registry, gated promotion, rollback, curriculum, historical replay, forgetting
  metric.
* Executed: 6-cycle run with Pythia-14M from the seed-0 BC policy (`results/continual/`), plus a
  2-cycle test with spawned processes. Every candidate tied production on the frontier evaluation
  and was promoted under the tie-permissive gate; the curriculum never advanced. The mechanics
  (versioned actors, ingestion, gating, rollback, forgetting tracking) work; one PPO update per
  32-episode cycle is too small a learning signal to change greedy behaviour.

## Phase 8 — Transfer and generalization research: protocol complete, evidence limited
* Implemented: held-out instances/layouts/templates/vocabulary and four held-out families;
  baselines A–D; ablations; skills; memory; multiple seeds; bootstrap CIs; compute accounting.
* Executed: A/B/C/D (2 seeds), D+imagination (2 seeds), ablations (no demos, no replay, horizon,
  uncertainty penalty, policy prior, world-model size), world-model transfer study, real-browser
  transfer. Result: imitation is the only component with a measurable effect; no configuration
  transfers to unseen families. At this scale the experiment cannot detect differences below
  ≈15 points and says nothing about large models.

## Known gaps (honest list)
* No GPU runs; the Qwen3-4B research profile is configured and its code path is the same as the
  executed one, but it was never run.
* WebArena / WorkArena adapters not implemented (BrowserGym adapter covers MiniWoB++ only).
* No latent recurrent belief state; no visual observations.
* Distributed (scaled) actor/learner not implemented.
* Human demonstrations were not collected from real people.
* C seed 1 was not run separately (its weights are identical to D seed 1's by construction;
  identity verified on seed 0).

## Latest full test run

`python -m pytest -q` → **104 passed** in 160 s (4 CPU cores shared with running experiments;
includes 25 real-Chromium browser tests). `bash scripts/verify_cli_workflows.sh` executed all
documented CLI workflows successfully.
