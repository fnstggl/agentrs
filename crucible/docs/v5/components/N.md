# Workstream N: training distribution, curriculum and the research instrument

Code: `src/crucible/research/v5/`. Gate tests: `tests/v5/test_gate_T.py`, `test_gate_U.py`,
`test_gate_SP_14_*.py`, `test_gate_SP_15_*.py`, `test_gate_SP_11_6b.py`, and the N tests in `test_gate_I.py` (I.n2),
`test_gate_K.py` (K.n2), `test_gate_S.py` (S.n1). Unit tests: `tests/test_v5_research_unit.py`. Status per
requirement: `docs/v5/workstreams/N.json`.

## Pieces

| module | what it does |
|---|---|
| `distribution` | frozen, versioned registry of the nine Part 14.1 kinds over all tiers; partitions train / validation / evaluation / sealed by family and seed range; structural guard (measured fingerprint distance 0 to an evaluation family is refused); training-log audit |
| `sampler` | seeded, resumable mixture sampler; JSON episode specs any learner stack can rebuild (`python -m crucible.research.v5.sampler make spec.json`); deficit scheduling on measured compute (default 50/30/20, refuses to start without every tier); `run_mixture` |
| `episodes` | drives a lever learner through `crucible.experience` (own cash signal only); verified per-tick outcome from `verify()`; normalised score against evaluator-side references; blind (no-feedback) ablation |
| `learners` | control learners with known capabilities: hold, adapter, memoriser, weights (small network), stateful (process stash), pretrained, meta (learns its search strategy), transfer (learns how to search across tasks); state split into parameters / memory / optimizer / within-episode |
| `evaluator` | the independent evaluator: `python -I` process with JSON socket channels; holds worlds, verifiers, registry, sealed salt, preregistrations, budgets; participant channel serves only participant operations |
| `fresh` | fresh-process weights-only evaluation: 15 checks computed in a child that runs under the sandbox Landlock jail (own uid, no sockets, read-only stdlib/numpy/source, private read-write directory); safe loading; declared configuration schema; canary confinement check; adversarial checkpoints |
| `adversarial` | SP-15.8 hooks: process stash, file stash outside the checkpoint, configuration smuggling, memory in parameters, pickle, architecture edit, generator calibration shortcut - each run against the real instrument (`run_controls`) |
| `hstats` | crossed learner-seed x world interval (calibrated with 3-5 seeds), pigeonhole bootstrap, null calibration |
| `curves` | budgets 0..64 (+128, 256) informative interactions, per-run N_q with censoring, Kaplan-Meier, RMST, acceleration that is never an invented finite number |
| `instrument` | independent establishment of control capabilities, battery, classification into the seven explanations of Part 15 with prespecified effects |
| `levels` | Levels 1-5 splits from the registry and their runs (Level 5: ReferenceLearner trained on Tier 1, paired untrained copy, held-out Tier 2 configurations, Tier 3 profiles and 4.0 tasks; D-70); `Level6Adapter` (docs/v5/LEVEL6_METHODOLOGY.md) |
| `curriculum`, `curriculum_lab` | ten weakness categories, four measures, targeted generation with a 0.3 random share and an entropy floor, partition guard; planted-deficiency experiments |
| `proposal` | world proposals validated for safety, feasibility, diversity, resources, difficulty and usefulness inside the evaluator |
| `comparison` | A/B/C/D arms with matched experience or compute (arms C and D with the architecture-neutral ReferenceLearner, which acts in every tier) and seven control kinds |
| `integrity` | clause-level claim scanners with planted controls; banned-proxy checks; failure classification |
| `compose`, `genadmit` | structural composition of Tier 1 worlds (calibration without the anti-default shortcut) and their full rapid admission |

## Measured in development runs (not official evidence)

* Statistics: on a real 12-seed pool of one learner, the crossed interval gives a false-positive rate of 0.015 at
  nominal 0.05; the 4.0 analysis (seeds averaged, worlds bootstrapped) gives 0.38.
* Fresh process: a learner that keeps what it learned in process memory gains 0.33 in process and exactly 0.0 from
  its checkpoint in a fresh process; a pickle-bearing, a memory-smuggling and an architecture-edited checkpoint are
  refused.
* Matched budgets: an identical meta learner trained on the v5 rapid families (arm B) scores 0.11 higher on held-out
  rapid families than the same learner trained on the original six 4.0 families (arm A), under matched experience
  and under matched compute (three seeds).
* Levels: a weights learner gains on Level 1 (+0.14) and Level 2 (+0.09) and loses on Level 3 (-0.35) and Level 4
  (-0.59): family-specific settings do not transfer to new mechanisms. Level 5 cannot run while Tier 2 is absent and
  `crucible.civ.generate` fails.


## Wave 3b changes (D-69, D-70)

* Controls: the transfer control was never trained in the battery (strategy credited at a task's 4th episode, the
  battery plays each world 3 times) - credited from the 2nd episode; the weights control trains from a replay buffer
  (memory, cleared for weights-only) with 64-bucket briefs; the meta controls credit strategies by their own episode
  gain (within-world rank agreement with the verified score 0.997, against 0.35 for the old measure) and the meta
  prior does not adapt; a frozen transfer control writes no task memory; `transfer_sham` is the known negative for
  transfer only (it keeps per-task states by construction).
* Design: 10 unfamiliar worlds per family, 24 repeat probes, 4 transfer tasks per family; families established in
  the battery's own mixed regime, with distinct public faces (`establish_mixed`, `FACE_DISTINCT`).
* Development battery (seed 0, establishment design, not evidence): adapter context 0.45 (world-interval low 0.32),
  memoriser 0.58 (0.34), weights 0.61 (0.50), stateful persistent state 0.61 (0.50), pretrained 0.41 (0.28), meta
  0.51 (0.40), transfer 0.31 (0.18), sham transfer -0.15.
* Generator: driver `drifting_optimum` (n-compose/3) with claim `ideal_drifts`. Admission's dominance check still
  rejects almost every generated world (T.4; see N.json limitations).
* Tier 2 through the merged domain framework (9 domains register): registry kinds 9/9; mixtures run on compute
  budgets; Level 5 includes two Tier 2 evaluation configurations.

## Known limits

See `limitations` in `docs/v5/workstreams/N.json`.
