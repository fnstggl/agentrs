# U-X2 — Experiment 2: causal discovery, learned exploration vs knowledge gradient and Thompson sampling (pre-registration)

Spec Part 17 Experiment 2, Part 8.5, Part 19; manifest X02. Committed before any confirmation-offset evaluation.
Code: `experiments/exp2.py`, `experiments/nig_agent.py`, `experiments/predict.py` (`TrajectoryRecorder`). Raw rows:
`/workspace/agentrs/runs/cam_unified/wave2/X/exp2/<set>/`.

## Worlds

Real Crucible mechanism worlds through G's `MechWorldAdapter`, from families whose hidden mechanisms contain
**confounders** (cw1_scm_confounder, cw1_scm_confounder_interaction), **interacting effects** (cw1_scm_interaction,
cw1_scm_gate_interaction) and **delayed consequences** (cw1_scm_delay, cw1_scm_delay_interaction,
cw1_scm_delay_stock, scm_delay). All are training families (familiar structures, new instances). Every world starts
at its resting settings with the same presentation (facility brief, neutral lever / gauge names in random order,
one typed tool per lever): before an intervention the stream does not identify the mechanism; only interventions
and their consequences do.

Validated tasks at `VAL_SEED0 + offset`: development offset **110,000**, 2 tasks per family (16 worlds);
confirmation offset **210,000**, 3 tasks per family (24 worlds).

## Learners (agent-selected interventions; one lever setting = one decision = one tick)

* `cam` - the trained CAM Unified checkpoints of U-X1 (seeds 0-2), weights-only, fresh process; exploration is its
  own learned policy (learning-progress reward in training, no KG input: `cfg.math_explore_assist = False`).
* `cam_untrained` - fresh weights (seeds 0-2).
* `kg` - online knowledge gradient on an exact NIG regression of the reward (CAM 7.0 math, D's `math_assist`),
  meta-learned prior of U-X1's `nig_meta` (seeds 0-2), candidates = hold + 5 values per lever.
* `thompson` - one posterior draw per decision, same model and candidates.
* `greedy`, `random` - references.

## Measures

Intervention-effect accuracy at k = 0..64 own decisions (log score and squared error of each learner's own
prediction of 12 forked single-lever interventions; evaluator-side forks, never shown to a learner);
decision-relevant accuracy (normalised regret of acting on the model's best-ranked query); observations needed
(smallest k from which that regret stays <= 0.1); cumulative regret sum_t (reference - net_t) / (reference -
default) over the first 64 ticks (primary) and the whole episode; goal achievement (`Task.success`); compute per
decision (wall, CPU, GPU-section seconds).

## Primary criterion (confirmation set)

**X2-P1**: cumulative normalised regret over the first 64 ticks, `cam` - `kg` < 0 (CAM's learned exploration
incurs less regret than the knowledge-gradient baseline), two-level 95% interval excluding 0.

Reported descriptively: the same against `thompson`, `greedy`, `random`, `cam_untrained`; per-motif breakdown;
observations needed; effect accuracy curves; success; compute.

## Statistics and budget

Two-level bootstrap over training seeds and worlds (paired per world), 95% percentile intervals. CAM evaluation:
one locked job per checkpoint set (< 25 min); the NIG experimenters run on CPU.
