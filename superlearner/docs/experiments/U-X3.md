# U-X3 — Experiment 3: parametric transfer, retention and the corrected negative-transfer diagnosis (pre-registration)

Spec Part 17 Experiment 3, Part 13.7, Part 11.3/11.5; manifest X03 (uses A14 plasticity instruments, A15
weights-only evaluation). Committed before any confirmation-offset evaluation. Code: `experiments/exp3.py`,
`experiments/train.py` (`--init/--condition`), `experiments/stack.py` (`load_condition`). Raw rows:
`/workspace/agentrs/runs/cam_unified/wave2/X/exp3/<set>/`.

## Why a rerun

CAM 7.0's C-4 negative transfer was confounded: `Cam7Learner.load` restored AdamW moments and the full source replay,
and 52-58% of the pretrained arm's meta-training draws stayed source episodes after 120k target steps. Here each
part of the source lifetime is restored separately, each condition in its own fresh process.

## Design (sequential multi-domain learning)

Domain 1 (source) = `scm` (U-X1 training runs, seeds 0-2). Domain 2 (target) = `abm` (agent-based markets: a different
simulator implementation; training families abm_congestion, abm_utility, abm_wom, cw1_abm_congest_wom,
cw1_abm_util_congest, cw1_abm_util_wom, cw1_abm_wom_congest). Each target run trains on the same guarded `abm` task
stream (stream start 100,000 x (seed+1) + 50,000; same per-batch presentation draws) for a **matched budget of target
decisions** from the source checkpoint of the same seed under:

* **A** fresh model and optimiser (target-only training, spec baseline 13);
* **B** source weights only (`load_weights_only` + `reset_experience_access`);
* **C** source weights + optimiser state;
* **D** source weights + optimiser + historical replay / archive / memory index (copied into the target run);
* **F** the full source lifetime (also fast/latent state, RNG, counters).

Budget: as many target decisions as two locked 22-minute chunks per run allow at dev scale, the same number of
decisions for every condition of a seed (the smallest reached is used, from checkpoints at matched decision
counts); seeds 0 and 1 (seed 2 if compute permits).

## Evaluation

Weights-only, fresh process, identical support and queries (`predict`, condition A design), k = 0..64, at the target
runs' checkpoints (zero-shot = the loaded state before any target update; milestones; end), on the category sets:
familiar (new instances of source families; **retention**), compositional (new combinations), structural (unseen
causal structures: inverted-U), abm (target families, new instances; **acquisition**), abm_heldout (held-out target
families), tech (queueing systems: never trained), scm_rescaled (familiar tasks behind a new semantic interface:
rescaled gauge units, renamed levers, other wording). Development offset **120,000** (2 tasks per family; familiar /
compositional / structural reuse U-X1's development tasks), confirmation offset **220,000** (3 per family; U-X1's
confirmation tasks). Unfamiliar full OpenWorld tasks: not part of this numeric experiment.

Plasticity (E's `memory.plasticity`): dormant-unit fraction, activation effective rank, effective rank / srank of the
latent state on a probe pass, relative weight change from the source, per checkpoint.

## Primary criterion (confirmation set)

**X3-P1 (weights-only transfer after matched target experience)**: log-score area on the `abm` set at the end of
target training, **B - A**, two-level 95% interval; positive transfer if it excludes 0 above, negative transfer if
below, null otherwise.

Diagnosis contrasts (reported with intervals, not criteria): C - B (optimiser state), D - C (historical replay), F - D
(the rest of the lifetime), B@0 - A@0 (zero-shot), retention = familiar-set area at the end vs at zero-shot per
condition, coverage at small k on abm (over-confident priors), plasticity measures.

## Statistics

Two-level bootstrap over training seeds and worlds, paired per world, 95% percentile intervals.
