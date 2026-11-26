# Audit: workstream N (research evaluation, training diversity, curriculum)

Auditor: independent audit agent, 2026-11-26. Worktree `.worktrees/c5-complete` (branch `feat/crucible-5-complete`);
audited at HEAD `dd745b9` (the orchestrator committed Clarifications 2 and 3 on top of `a881d18` while this audit was
running; nothing under `src/` changed in that commit). No existing file was modified. Probe scripts and logs:
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/N_research/` (`probes/p1..p9*.py`, `*.log`).

Scope: MASTER_SPEC Parts 14 and 15, Gates T and U, SP-14.1/14.2/14.4, SP-15.3/15.5/15.6, notes I.n2, K.n2 and S.n1;
Clarification 1 (no CAM dependency), Clarification 2 (observation-only by default) and Clarification 3 (recovery
preamble: earlier claims count for nothing without evidence).

## 1. Bottom line

Almost none of workstream N's Crucible 5.0 scope exists. Both N agents were interrupted:

* N1 (`0628399`, merged in `95f2efb`) left one file, `src/crucible/research/v5/compose.py` (607 lines). Nothing
  imports it and no test covers it.
* N2 (`808309b`, merged in `34e3f74`) changed only `src/crucible/experience/tier1.py` (RapidExperienceEnv), which
  belongs to the learner interface (workstream L).

None of these exist anywhere in `src/`:

* a training-distribution registry across tiers, or a sampler;
* the 50/30/20 tier mixture;
* an adaptive curriculum for v5;
* a world-proposal interface;
* a framework for the six generalization levels;
* an A/B/C/D comparison configuration;
* positive controls on the new families.

`rapid/curriculum`, which WORKSTREAMS.md names as N's location, does not exist. None of N's gate test files exist
(`tests/v5/test_gate_T.py`, `test_gate_U.py`, `test_gate_SP_14_*.py`, `test_gate_SP_15_*.py`). A read-only call to
`crucible.acceptance.v5.checker.verify_all()` returns `NOT_STARTED` for all 25 N requirements, with the reasons "no
implementation files listed", "test files missing" and "no evidence record".

What does exist is the Crucible 4.0 research instrument: `research/protocols.py`, `stats.py`, `sealed.py`, `tasks.py`,
`envs.py` and `learners/`.

* Its 45 tests pass (I re-ran them).
* It is carefully built: frozen evaluation with checkpoint hashing, paired tasks, an eight-step internalisation
  protocol, a timescale split, transfer with censoring-aware ratio bounds, and sealed sets with budgets.
* It is bound to the 25 Crucible 4.0 hidden-mechanism families through `Task`/`MechVecEnv`, and to padded
  lever/gauge vectors.
* It always runs in `legacy_shaped_reward`, and cannot be given another reward mode.
* It cannot run a single v5 Tier 1 family, and Tier 2 and Tier 3 do not exist (`tiers_available() = {1: True,
  2: False, 3: False}`, 0 domains).

My probes found defects that matter for the research objective, beyond the missing pieces:

1. **The "weights-only" protocol can be fooled.** A learner whose checkpoint never changes, but which keeps its
   learned settings in process memory, gets `verdict: internalised`, `valid: true`, gain 0.998, p = 0.0005, with
   `weights_changed_by_training: false` and no warning (probe p2). Evaluation is in-process, not fresh-process.
2. **Intervals ignore learner-seed variation.** Two identical learners were compared through the protocol's
   statistics: 3 seeds × 12 paired tasks, seed-level effect sd 0.15. The protocol called a significant difference in
   94% of 400 null trials, against a nominal 5% (probe p5). `compare_runs` averages over seeds and bootstraps only
   over tasks ("unit: task (mean over seeds)", `protocols.py:1889`).
3. **Positive controls collapse under the mandated default.** With the environment forced to `observation_only`
   (Clarification 2), the weight-updating control learns nothing: verdict goes from `internalised` (gain 0.998) to
   `no_gain_detected` (gain 0.0). The context control's within-episode gain falls from 0.86 to 0.02, and its
   late-minus-early goes from +0.32 to -0.28, yet it is still classified "fast" (`min_effect = 0.0`) (probe p7).
4. **The only adaptive curriculum moves away from a measured weakness.** `goals.curriculum.GoalSampler` cut the
   weak goal family's share from 0.22/0.17/0.16 to 0.07/0.07/0.13 over 400 samples in three seeds. It has no uniform
   random component. It accepted a held-out test world family (`engineering_lab_heldout`) under `split="train"` and
   produced a goal labelled `train` on it (probe p3).
5. **Generated worlds carry a simulator shortcut** (`compose.calibrate`). Each lever's optimum is drawn from
   {0.2, 0.25, 0.3, 0.7, 0.75, 0.8} of its range, and its public default is placed at the opposite end. The tool
   schema prints that default ("default 0.0"). A fixed public rule, "75% of range if the default is the minimum,
   else 25%", reaches a median 0.82 of the oracle's improvement, and more than 0.75 (the admission dominance
   threshold) in 7 of 13 calibrated worlds. Admission's canonical strategies reach a median of only 0.28, so
   admission does not catch it (probe p9). Training on such worlds teaches a rule about the generator that will not
   carry over to the open internet.
6. **Evaluation partitions are protected only nominally.**
   * Learner code in the same process can relabel a v5 `val` family as `train`.
   * `rapid.make` hands out evaluation families to anyone.
   * A `Learner` method can read the sealed salt with plain `open()`, although `sealed.open_set` refuses
     learner stack frames (probe p8).

Verdict counts:

* Subsystems (17, as listed in the JSON): 0 RETAIN, 5 REPAIR, 5 EXTEND, 1 REDESIGN, 1 REBUILD, 5 IMPLEMENT.
* Requirements (25): 10 IMPLEMENT, 1 REBUILD, 1 REDESIGN, 7 EXTEND, 6 REPAIR, 0 RETAIN. No N requirement can pass
  its criteria in the integrated system today.

## 2. What exists, file by file

| Path | Origin | Wired into the v5 system? |
|---|---|---|
| `src/crucible/research/v5/compose.py` | ws-N1, interrupted | No: no importer, no test (`grep -rn "research.v5" src tests` is empty) |
| `src/crucible/research/protocols.py` (1,889 lines) | 4.0 | Only `examples/ow_benchmark.py` and the 4.0 tests use it; it accepts only `research.tasks.Task` (4.0 mechanism families) |
| `src/crucible/research/stats.py` | 4.0 | Used by protocols |
| `src/crucible/research/sealed.py` | 4.0 | 4.0 test-split mechanism families only |
| `src/crucible/research/tasks.py`, `envs.py`, `bridge.py`, `learner.py`, `learners/*` | 4.0 (+ v5 reward_mode parameter on envs) | `MechVecEnv` defaults to legacy (`envs.py:47`) |
| `src/crucible/goals/curriculum.py` (`GoalSampler`) | 4.0 | No user outside `tests/test_goals.py:170` |
| `src/crucible/goals/taskgen.py` | 4.0 | Kernel-world task mining, not part of the v5 curriculum |
| `src/crucible/evaluation/*` | 4.0 | Goal-harness evaluation, open-endedness and longrun metrics; nothing v5-specific for N |
| `src/crucible/realworld/*` | 4.0 security | Gated bridge, off by default; there is no Level 6 evaluation adapter |
| `src/crucible/rapid/families.py` | ws-H | Tier 1 family registry with `split`/`PARTITIONS` (`families.py:44`); no sampler; no cross-tier registry |
| `tests/v5/test_gate_I.py::test_In2_*` | ws-C2 for N | Regex scan of realism claims |
| `tests/v5/test_gate_T.py`, `test_gate_U.py`, `test_gate_SP_14_{1,2,4}.py`, `test_gate_SP_15_{3,5,6}.py` | — | **Missing** |
| `src/crucible/civ/__init__.py`, `src/crucible/multiagent/__init__.py` | — | 5-line stubs; Tier 3 does not exist |

## 3. Tests and probes run

All commands ran from `crucible/` with `PYTHONPATH=src`, `OMP_NUM_THREADS=1` and `TMPDIR` pointing into the
scratchpad audit directory.

| Command | Result |
|---|---|
| `pytest tests/test_research_stats.py tests/test_goals.py` | 21 passed (69.9 s) |
| `pytest tests/test_research_protocols.py -k "timescales or internalisation or memoriser or overlapping or transfer_efficiency or sealed or frozen"` | 19 passed (17.7 s) |
| `pytest tests/test_research_protocols.py tests/test_research_envs.py` | 45 passed (61.9 s) |
| `pytest tests/v5/test_gate_I.py -k In2` | 1 passed. Unofficial evidence: `unsupported_realism_claims=0`, `statements_checked=3`, `files_scanned=34` |
| `pytest tests/test_realworld_gate.py` | 14 passed (17.7 s) |
| `checker.verify_all()`, called in-process and read-only (the CLI `report` would overwrite `docs/v5/evidence/checker_report.json`) | All 25 N requirements `NOT_STARTED` |
| p1 `compose` + admission (slot scheduler, 1 CPU) | 60 random genomes: 60 distinct ids, 28 calibrated. 24 admitted through `rapid.admit` on seeds 0-2: 7 admitted, 23 feasible. Failures: no_dominant_strategy 9, identifiability 8, difficulty 6, controllability 1, achievable 1. About 10 s of CPU per family. Registry restored afterwards. |
| p2 weights-only vs process-memory stash | `GlobalStash` (constant checkpoint, `state_hash` constant): `internalised`, valid, gain 0.998, `weights_changed_by_training: false`, no warning |
| p3 GoalSampler weakness targeting and partitions | Weak share in the first vs last quarter: 0.22→0.07, 0.17→0.07, 0.16→0.13. Frontier share about 0.09; uniform random share 0. Test world family accepted and labelled `train` |
| p4 integration | `tiers_available={1:T,2:F,3:F}`, 0 domains. `check_mode(None)`, `MechVecEnv`, `RapidExperienceEnv` all default to legacy. Protocols take no reward_mode. A rapid family passed to `P.evaluate` gives `AttributeError`; `make_task("net_routing")` gives `KeyError`. Default env: nonzero reward on 10 of 10 steps, and the reward slot in the observation is nonzero |
| p5 statistical calibration | Protocol false-positive rate 0.94 for identical learners (seed effect sd 0.15, 3 seeds, 12 tasks); a naive two-level bootstrap gives 0.23 (3 seeds are too few for any bootstrap) |
| p6 censoring | `_steps_pair` correctly returns bounds, not finite ratios, when one arm is censored. A seed-mean N_q of [4, None, None] reports 4.0 (censored seeds dropped). N_q is interpolated between measured points (3.25 where measurements are at 2 and 4). On the mean curve, half the tasks reaching gives N_q None. No survival estimator |
| p7 positive controls under observation_only | WeightLearner: internalised 0.998 → no_gain 0.0. ContextLearner: gain 0.86 → 0.02, late-early +0.32 → -0.28, yet still attributed `fast: True` |
| p8 tamper | `val` family relabelled `train` by learner code: True. `rapid.make` on a `val` family: unguarded. Sealed salt readable from a `Learner` method: True. `sealed.open_set` from a `Learner` method: refused |
| p9 calibration shortcut | Anti-default 25/75 rule: median 0.82 of the oracle's improvement, more than 0.75 in 7 of 13 calibrated worlds; best canonical median 0.28 |

## 4. Subsystem verdicts

### 4.1 Training-distribution registry and sampler (Part 14.1, SP-14.1): IMPLEMENT, effort L

* Required: one versioned registry of engines, mechanism families, world families, domain configurations,
  objectives, participant-policy distributions, action interfaces, difficulty levels and train/validation/test
  partitions; a sampler for learners; balanced and configurable mixtures; prevention of accidental evaluation reuse.
* What exists is per-tier fragments with no common schema:
  * `rapid.families.FAMILIES/REGISTRY`: Tier 1. Records engine, category, objective, axes, participants and action
    kinds; `PARTITIONS` maps both `val` and `test` to `v5-eval-1` (`families.py:44`).
  * `mechanisms.base.MECH_FAMILIES`: 4.0, 25 families.
  * `worldgen.generator.FAMILIES`: 4.0 world families, 7 train / 5 val / 5 test / 1 dev.
  * `domains.REGISTRY`: Tier 2, empty.
  * `civ`: Tier 3, a stub.
* Nothing samples across these; there is no version on the distribution; reuse prevention exists only inside the
  4.0 protocols (`_Watch`, `overlap()`).
* Rapid objectives are not a dimension: 69 of 74 families are `maximize_net`, with 3 objective types in total.
* Work needed:
  * a `crucible.research.distribution` (or similar) registry over all tiers, keyed by the six diversity dimensions
    of Part 14 (parameters, causal structure, composition, objective, participant policies, action space), with a
    version and digest;
  * a seeded, resumable sampler that refuses evaluation partitions by identity and by fingerprint, plus
    balanced/configurable mixture specs;
  * a gate test that samples a long stream and proves zero eval items, and proves the declared mixture is met.

### 4.2 Mixed curriculum 50/30/20 (Part 14.2, SP-14.2): IMPLEMENT, effort M, blocked by Tier 2 and Tier 3

* There is no mixture configuration.
* With 0 Tier 2 domains and no Tier 3, the default mixture cannot be instantiated at all.
* "Measure compute and environment experience separately" is done by `research.accounting` for 4.0 tasks only.
* Work needed:
  * a default mixture config `{tier1: 0.5, tier2: 0.3, tier3: 0.2}` over *compute*, with experience counted
    separately;
  * a scheduler that allocates measured compute, not episodes;
  * evidence from an executed run that the realised compute shares are within tolerance and that all three tiers
    were actually instantiated.

### 4.3 Adaptive curriculum (Part 14.3, T.5, T.6, T.8): REBUILD, effort L

* The only implementation is the 4.0 `goals.curriculum.GoalSampler`:
  * Its score is learning progress, plus uncertainty, plus novelty over fixed arms (goal family × level × world
    family).
  * It drops arms with p_success < 0.05 (`curriculum.py:76`). A persistent weakness is therefore *de-prioritised*,
    which p3 shows: the weak family's share fell from about 0.2 to 0.07.
  * It has no uniform random component; `p_frontier = 0.1` only re-samples the hardest level.
  * It does not generate environments.
  * It measures none of the ten weaknesses listed in 14.3.
  * It has no guard on partitions (`generate_goal` trusts the caller's `split`, `contracts.py:330`; p3 sampled a test
    world family under `split="train"`).
  * Nothing outside `tests/test_goals.py` uses it.
* Building blocks exist in `compose.py`: its drivers and population kinds map onto several weaknesses.
  * delayed_payoff → delayed credit
  * hidden_state → memory
  * pilot_experiments → experimentation
  * rivals → adaptive competitors
  * regime_shift → changing environments
  * long_horizon → long-horizon execution
* Nothing measures a learner's weaknesses or maps them to structures.
* Work needed:
  * learner weakness profiles computed from evaluator-side results only (never sealed);
  * a generator that targets the weak dimensions, with a fixed uniform-random share of at least 20% and a
    collapse guard (entropy floor over families and structures);
  * a planted-weakness positive-control learner, to show the curriculum shifts mass to the right structures
    (p-value) and that targeted training improves that weakness more than untargeted training (T.8).

### 4.4 Procedural structural composition (`research/v5/compose.py`; T.1-T.4): EXTEND, effort M

Sound pieces:

* The genome grammar has 8 topologies, 9 motifs, 4 combiners, 4 temporal classes, 7 channels, 1-3 levers, 7 drivers
  and 5 population kinds.
* Composition goes through H's `GraphBuilder`.
* `registered()` is a context manager that restores the registry exactly (p1 confirmed this).
* In p1 the 60 genomes gave 60 distinct ids; 28 of 60 calibrated; 7 of 24 passed the full ten-check admission.

Defects:

1. Nothing uses it and nothing tests it.
2. The objective is always `maximize_net` (`compose.py:404`, `:590`), so objectives do not vary independently (T.3).
3. "Distinct structures" counts genome digests. A count of 50 says nothing about structural distinctness; it should
   be counted by fingerprint distance above a minimum, after admission.
4. Generated families get `claims=()`, `reference=None`, `oracle=None`, and identification is always `lever_sign`.
   As a result, 8 of 24 failed identifiability, and there are no mechanism-level intervention tests.
5. The calibration shortcut found in p9: the target set {0.2, 0.25, 0.3, 0.7, 0.75, 0.8} (`compose.py:468`), with
   the default placed at the opposite end (`compose.py:495`) and published in the tool schema.
6. The "participant populations" are equation primitives (`adaptive_opponent`, `learning_behaviour`,
   `endogenous_demand`), not policies from the population or self-play packages, and exist only in `primitive_graph`
   worlds.
7. There is no admission loop. Admitting 50 needs about 170 draws at a 29% admission rate, about 30 minutes of CPU;
   nobody has run it.

Work needed:

* Randomise targets continuously and decouple the default from the optimum side; add "anti-default" and "quartile"
  rules to admission's trivial-strategy set.
* Draw objectives from an objective registry independently of structure.
* Generate per-structure identification and intervention claims.
* Write a batch admission driver with fingerprint-distance dedup and evidence.
* Use real participant policies.

### 4.5 World-proposal interface (Part 14.4, SP-14.4): IMPLEMENT, effort M

* No proposal API exists. `family_spec(..., proposer=...)` only records a name.
* Validating feasibility, diversity, safety, resources, difficulty and scientific usefulness needs an API that:
  * accepts a genome from a learner;
  * runs admission, fingerprint novelty against train **and** evaluation families, resource estimation (CPU s per
    episode, memory) and difficulty bands evaluator-side, in a separate process;
  * returns only accept/reject and public reasons.
* Tamper resistance needs the registry and evaluator outside the learner's process. p8 shows that today learner
  code can rewrite `RF.FAMILIES[...].split` and read the sealed salt.

### 4.6 Research protocol harness (`research/protocols.py`, `tasks.py`, `envs.py`; Levels 1-5, U.1): REDESIGN, effort XL

The protocol logic is sound:

* frozen evaluation with hashes before and after;
* id-plus-mechanism-fingerprint disjointness;
* identical task streams for every learner;
* resumable learning curves with separate costs;
* Holm correction;
* explicit failure diagnosis.

Its environment abstraction is the 4.0 assumption that every world is a `Task` of a hidden-mechanism family run in
`MechVecEnv`. Observations are padded lever/gauge vectors (`learner.py`, `EnvSpec`), with a scalar reward that
learners read from `observe(reward)` and from observation slot `scalars[0]`. This assumption blocks the v5 research
objective in three ways:

1. No v5 Tier 1 family, Tier 2 domain or Tier 3 civilisation can be evaluated (p4: `P.evaluate(..., ["net_routing"])`
   gives `AttributeError`; `make_task` gives `KeyError`).
2. Every protocol constructs `MechVecEnv(tasks, spec)` with the legacy default (`protocols.py:362`), and none of
   `evaluate`, `lifetime`, `internalisation`, `transfer`, `compare` accepts a reward mode (p4). This conflicts with
   Clarification 2.
3. The generalization levels are the 4.0 transfer categories (`within`, `compositional`, `structural`, `cross_impl`;
   `protocols.py` `CATEGORIES`):
   * Level 2 (new parameter distributions) is absent: `within` is new seeds from the same distribution.
   * Level 5 (new domains *and action interfaces*) is absent: `cross_impl` is the techsys engine with the same
     lever/gauge interface.

Work needed:

* Rebase the protocols on the architecture-neutral `crucible.experience` interface (`ExperienceEnv` records, tool
  actions), with an evaluator-side outcome verifier per world, observation-only by default and legacy only by
  explicit selection.
* Keep the statistics and protocol bodies.
* Add a `GeneralizationLevel` spec that builds level-1..5 splits from the distribution registry: level 2 by shifted
  parameter ranges; level 3 by held-out compositions of familiar primitives; level 4 by held-out primitives or
  engines; level 5 by held-out domains, action interfaces and engines.

### 4.7 Fresh-process weights-only evaluation (Part 15.4, U.3): REPAIR, effort M

* `internalisation` reloads the checkpoint into a new object *in the same process* (`protocols.py:1245`).
* `examples/ow_benchmark.py` runs each unit, training and evaluation together, in one worker process.
* p2 proves the consequence: process-resident state survives, and the protocol certifies it as internalised
  knowledge, with no warning even though the checkpoint hash did not change.
* The checkpoint is "whatever `save` writes", so a memory or replay store written into it is indistinguishable from
  weights (the test Memoriser stores its table in its checkpoint).
* Spec 15.4 requires more than that:
  * clear written memories, retrieval indexes, replay access and optimizer state where required;
  * load only authorised parameters and the immutable architecture config;
  * verify parameter hashes.
* Work needed:
  * a subprocess evaluator (spawn, `-I`, an empty temporary HOME and cache, no network, a read-only copy of the
    checkpoint) that receives only `{architecture config, parameter file, expected hash}`;
  * verify the hash in the child;
  * a manifest-declared split of checkpoint tensors into "parameters" and "memory/replay/optimizer", refusing
    non-parameter entries for weights-only runs;
  * a mandatory warning or invalidation when `weights_changed_by_training` is false but a gain is detected;
  * adversarial controls (process-stash, file-stash outside the checkpoint, memory-in-checkpoint) in the gate test.

### 4.8 Positive controls (Part 15.6, U.4, SP-15.6): EXTEND, effort L

* The 4.0 tests contain five single-mechanism controls: ContextLearner, WeightLearner, Memoriser (honest and
  cheating), MetaLearner, and HoldLearner. With a transfer test on top, they cover the five kinds of 15.6.
* The timescales, internalisation and transfer tests pass (45 of 45).
* Limitations:
  * the controls live in `tests/test_research_protocols.py`, not in `src`;
  * they run only on hand-made one-lever inverted-U tasks (`peak_mech`) under legacy reward;
  * the "genuine transfer" control is a constant setting that happens to be near-optimal for every seed of the
    source family, which is Level 1 only;
  * p7: under observation_only they collapse, or are misclassified (`min_effect = 0.0`, `protocols.py:1032`, `:1058`);
  * there are no controls on new nontrivial v5 families;
  * `meta_np` failed its own positive-control criteria (`docs/experiments/ow_meta_np_result.json`).
* Work needed:
  * move the controls to `src/crucible/research/controls/`;
  * make every control derive its learning signal from observed consequences (revenue minus cost in the
    observation), which is what Clarification 2 requires;
  * add controls for pretrained competence, persistent state, meta-learning and transferable efficiency (the seven
    kinds of the Part 15 preamble);
  * run them on at least three admitted v5 families of different engines, with independently established ground
    truth;
  * prespecify a minimum effect size;
  * the gate passes only when every control gets its known label and the null controls get none.

### 4.9 Learning curves and N_q (Part 15.3, U.6, SP-15.3): REPAIR, effort M

What works:

* `lifetime`, `derive_curve`, `transfer` and `_steps_pair` record a curve from the untrained point with compute,
  wall time and simulated resources.
* Censoring of the pretrained/scratch ratio gives bounds, not finite values (p6).

Defects:

* There is no schedule of 1, 2, 4, 8, 16, 32, 64 *informative interactions* per unfamiliar task; points are uniform
  `eval_every` env-step multiples in batches of `n_envs` episodes.
* N_q is taken on the mean curve over tasks, not per (learner seed, task) run, and is linearly interpolated, so it is
  not an observed count (`stats.py:179`).
* Across seeds, `compare_runs` averages `steps` over only the runs that reached the threshold (`protocols.py:1856`)
  and reports `reached_runs` separately (p6: [4, None, None] → 4.0).
* In the ratio bootstrap, `if a and b is not None` (`protocols.py:1444`) discards resamples where a = 0.
* There is no survival or restricted-mean estimator, and no acceleration with censoring-aware intervals per run.

Work needed:

* geometric per-task interaction schedules;
* per-run N_q, with right-censoring at the budget;
* Kaplan-Meier median N_q and restricted-mean N_q, with intervals;
* acceleration as a ratio of survival quantiles with bounds when censored;
* time-to-competence, action cost, data, retention and transfer columns per run.

### 4.10 Statistics (Part 15.7, U.8): REPAIR, effort M

* `stats.py` is careful: paired permutation tests, Wilson intervals, Holm correction, `low_n` flags, and ratios
  undefined when the denominator is not safely positive.
* But `compare_runs`, `combine_evaluations` and `evaluate_many` average seeds away. The analysis unit is "task (mean
  over seeds)", so intervals contain world variation only.
* p5: a 94% false-positive rate for identical learners when seed-level variation exists. Seed SD is reported but
  never used in an interval.
* Development and confirmation are separated only by the 4.0 val/sealed split.
* There is no preregistration record of metrics.
* Work needed:
  * a crossed (world × learner-seed) hierarchical bootstrap, or a mixed-effects permutation, for every comparison;
  * at least 5 seeds required before a "significant" verdict;
  * a gate test of null calibration (false-positive rate ≤ nominal + tolerance in simulation);
  * a preregistered metrics file hashed before runs;
  * confirmation runs only on sealed v5 partitions.

### 4.11 Matched-budget comparisons and A/B/C/D (Part 15.5, U.5, SP-15.5)

Matched budgets: EXTEND, effort M.

* `compare` enforces identical training streams and step grids (matched experience).
* It reads scores at a common compute budget by interpolation (`compute_budget_s`), which is matched compute after
  the fact, not enforced.
* All of this is for 4.0 tasks only.

A/B/C/D configuration: IMPLEMENT, effort L, blocked by Tier 2 and Tier 3.

* No configuration exists. Arm C needs Tier 2 and arm D needs Tier 3.
* Arm A ("the original six families") is not even defined in the repository: 4.0 has 25 mechanism families, of
  which 7 are `train/base`.
* Controls available: untrained (point 0 of `lifetime`), frozen pretrained (frozen evaluate), scripted
  (`probe_exploit`, `hill_climb`), statistical/model-based (`model_based`, `evo_probe`, `meta_np`),
  architecture-neutral interface (`Learner`).
* "Existing CAM checkpoints" are replaced by Clarification 1 with an external-checkpoint adapter, which does not
  exist for the research harness.
* Work needed: an executable config with four arms on one learner class, identical seeds, and matched experience and
  compute budgets (enforced stop at a compute budget); the seven control kinds; a multi-seed smoke run with evidence.

### 4.12 Sealed sets and partition protection (T.7): REPAIR, effort M

What works:

* `research/sealed.py` keeps a secret salt outside the repo, with an acknowledgement string, an access budget,
  generations and a contamination flag.
* It refuses learner frames (p8 confirms `open_set` is refused).

Defects:

* It accepts only 4.0 test-split families. There is no sealed v5 partition; `val` and `test` both map to the public
  `v5-eval-1`.
* The trust boundary is in-process and same-uid: the salt is readable from a `Learner` method (p8).
* The rapid registry is mutable from learner code (p8 relabelled a `val` family as `train`).
* `rapid.make` serves evaluation families to any caller.
* GoalSampler leaks test worlds (p3).

Work needed:

* a v5 sealed partition per tier derived from a salt;
* evaluation in a separate process or uid, so the learner process never holds the salt;
* a read-only (frozen) registry view for learners;
* guarded factories that refuse evaluation partitions outside evaluator processes;
* leak probes as gate tests, at least five vectors.

### 4.13 Level 6 adapter (Part 15.1, U.2): EXTEND, effort M

* `crucible.realworld` is a hardened, authorisation-gated bridge, off by default. `RealWorldBridge()` refuses with
  "no authorisation document" (my probe), and 14 tests pass.
* It is not a Level 6 *evaluation adapter*. Nothing connects a learner or protocol to an authorised real-world
  objective with an independent measurement function and a methodology document.
* U.2's criteria (disabled by default, requires authorisation) would already pass on the bridge alone, which shows
  they are too weak.
* Work needed: a `Level6Adapter` implementing the same protocol interface (episodes are authorised objectives,
  outcomes come from `RealWorldBridge.measure`); a methodology document; a test that it is refused without
  authorisation and that no generalisation claim is emitted without executed experiments.

### 4.14 Independent evaluation (Part 15.2, U.7): EXTEND, effort L

* For 4.0 tasks, scores come from operator-side private references (`Task.score`, the `MechVecEnv.finished`
  records), not from the learner.
* For v5 Tier 1, evaluator-side oracles exist (`rapid.oracles`, `admission`), but no learner protocol uses them.
* For Tier 2 and Tier 3 nothing exists.
* There is no check that banned proxies (tool calls, reasoning length, posts, websites, agent-reported success, LLM
  judgments) are never used as outcomes.
* Work needed: a per-world evaluator-side verifier contract used by all protocols; a static and runtime test that
  outcome metrics come only from verifier-owned state; a test that reports from the learner side cannot change
  outcomes.

### 4.15 Claim-integrity notes (I.n2 REPAIR S; K.n2 IMPLEMENT S; S.n1 IMPLEMENT S)

* **I.n2.** The test passes, and the content is currently honest: `FIDELITY_REGISTER.md:52` says "known
  unrealistic", and `social/cascade.py:14` says "No realism is claimed".
  * The scanner, however, treats any sentence containing "not", "no", "without" and similar words as disclaimed.
  * It scans only `components/C2.md`, `FIDELITY_REGISTER.md` and `social/**.py`, not README, PROGRESS,
    IMPLEMENTATION_STATUS, ARCHITECTURE, agent reports, evidence notes or figures.
  * It asserts only `checked > 0`.
  * Fix: a broader corpus and a claim/negation parse of the same clause, with planted positive and negative
    sentences as controls.
* **K.n2.** The reference result dict carries `"strategy": "reference"` and the docstring disclaims, but there is no
  machine-readable `scripted: true` / `learner: false` label, no test and no metric.
* **S.n1.** No test or metric exists.

## 5. Requirement verdicts (workstream N)

| ID | Verdict | Reason (short) | Criteria weak? |
|---|---|---|---|
| T.1 | EXTEND | compose.py composes 60/60 distinct genomes but is unused and untested; distinctness is by digest | yes: count admitted, fingerprint-distinct structures |
| T.2 | EXTEND | 5 population kinds in compose (equation primitives, Tier 1 only) | yes: should require measured behavioural/outcome differences |
| T.3 | IMPLEMENT | Objective fixed to maximize_net; rapid has 3 objective types | yes: should require objectives crossed with structures |
| T.4 | EXTEND | 7/24 admitted in probe; no batch admission; shortcut not caught | no (but must count full 10-check admission plus the shortcut check) |
| T.5 | REBUILD | GoalSampler de-prioritises measured weakness (p3) | yes: p-value of what is undefined; needs a planted-weakness control |
| T.6 | IMPLEMENT | No uniform random component; frontier 10% on hard arms only | yes: share alone does not detect collapse |
| T.7 | REPAIR | 4.0 sealed only; v5 partitions mutable and unguarded; salt readable; GoalSampler leak | yes: requires a minimum number of leak vectors probed |
| T.8 | IMPLEMENT | Nothing measures weaknesses or targeted improvement | yes: should require at least 3 distinct weaknesses |
| U.1 | REDESIGN | 4.0 categories on 4.0 tasks only; no Level 2; Level 5 lacks new domains and interfaces | yes |
| U.2 | EXTEND | Gated bridge exists; no evaluation adapter or methodology | yes |
| U.3 | REPAIR | In-process reload; fooled by process stash (p2) | yes: needs adversarial controls |
| U.4 | EXTEND | 5 kinds in tests on toy tasks, legacy reward; collapse in observation-only (p7) | yes |
| U.5 | EXTEND | Matched experience plus post-hoc compute interpolation; 4.0 only | yes |
| U.6 | REPAIR | Uniform env-step curves; no 1..64 schedule | yes |
| U.7 | EXTEND | Private-reference scores for 4.0; none for v5 learners | yes: a boolean self-report |
| U.8 | REPAIR | Seeds averaged away; 94% FPR under null (p5) | yes: needs null-calibration evidence |
| I.n2 | REPAIR | Currently honest; scanner heuristic and scope weak | yes |
| K.n2 | IMPLEMENT | No label, test or metric | no |
| S.n1 | IMPLEMENT | No test or metric | no |
| SP-14.1 | IMPLEMENT | No cross-tier registry or sampler | yes: must exercise sampler and partition refusal |
| SP-14.2 | IMPLEMENT | No mixture; Tier 2/3 absent | yes: config match is not executed compute shares |
| SP-14.4 | IMPLEMENT | No proposal interface | yes: 2 tamper attempts is too few; include in-process vectors |
| SP-15.3 | REPAIR | Censoring partly right; per-run N_q, survival and 1..64 schedule missing | yes |
| SP-15.5 | IMPLEMENT | No A/B/C/D config; arms C/D blocked | yes: must be executed multi-seed |
| SP-15.6 | IMPLEMENT | No control on new families | yes: needs a known-negative control on the same families |

## 6. Spec requirements with no manifest entry

* **14 preamble.** Diversity measured on all six dimensions (parameters, causal structures, compositions,
  objectives, participant policies, action spaces); seeds alone insufficient.
* **14.1.** Sampler compatible with external learner infrastructure; balanced and configurable mixtures; a version
  on the distribution.
* **14.2.** Compute and experience measured separately; all tiers available from the start; full OpenWorld training
  not postponed.
* **14.3.**
  * The ten weakness categories.
  * Learning progress, regret, uncertainty and success as curriculum measures.
  * Non-collapse.
  * **The curriculum never accesses sealed results.** No requirement states this.
* **15 preamble.** Distinguishing all seven: memorization, pretrained competence, context adaptation, persistent
  state, neural weight learning, meta-learning, transferable efficiency. U.4 covers 5.
* **15.1.** No claim of real-world generalization without executed authorised experiments.
* **15.2.** Banned proxy outcomes (tool calls, reasoning length, posts, websites, agent-reported success, LLM
  judgments, unverified milestones) never used as outcomes.
* **15.3.** Time to competence, total action cost, wall-clock, data usage, retention, transfer, sample efficiency.
* **15.4.** Clearing written memories, retrieval indexes and replay; optimizer reset; immutable architecture config;
  distinguishing within-episode adaptation runs.
* **15.7.** Paired world comparisons, prespecified metrics, dev/confirmation separation, independent holdouts,
  transparent failure reporting, no winner selection on the sealed set.
* **Clarification 2 × Part 15.** Research protocols and positive controls must run with observation_only as the
  default, with learner-supplied rewards. OO.1 and OO.6 cover environments, not the instrument.
* **Instrument integrity.** No requirement demands adversarial controls against the instrument itself (process
  stash, out-of-checkpoint files, calibration shortcuts).

## 7. Transfer to the open internet: shortcuts and integrity risks

* **The calibration shortcut (p9).** Optimum fractions come from a 6-value set, and the default always sits on the
  opposite side and is printed in the tool schema. A learner trained on generated worlds can learn this generator
  artifact; it would carry over to no real market. The trivial-strategy set of admission
  (`policies.canonical_actions`: min/mid/max/random/wait) does not include it.
* **Structure is not the same as genome.** Distinctness is counted by genome digest. Admission's `distinct` check is
  `distance > 0` (`rapid/admission.py:157`): any fingerprint difference counts. A 50-structure count could still be
  structurally narrow.
* **Equation "participants".** Rivals, customers and word of mouth in generated worlds are primitives inside one
  causal graph, not autonomous policies; strategic diversity is shallow.
* **The instrument certifies what it cannot see (p2).** An internalisation claim can be produced by process-resident
  state; claims of "learned in weights" are unreliable until fresh-process evaluation exists.
* **Statistics over-claim (p5)** when learner-seed variation matters, which it does for neural learners.
* **Legacy reward everywhere.** Every existing research result (`docs/experiments/ow_benchmark_*`) was produced with
  shaped reward on 4.0 families. None of it is evidence about observation-only learning or about v5 worlds.
* **Positive controls are toy-task, test-only and reward-reading** (p7). That the instrument detects learning has
  not been shown where it matters.

## 8. Recommended order of work (N, after L delivers observation-only defaults)

1. Rebase the protocols on `crucible.experience`, with observation_only by default and an explicit legacy mode.
   Port the 4.0 tests (keep all 45 passing in legacy mode; add observation-only equivalents with
   consequence-derived learner signals).
2. Fresh-process weights-only evaluator, plus adversarial controls (p2-style stash, file stash, memory in
   checkpoint).
3. Hierarchical world × seed statistics, plus a null-calibration test; per-run censored N_q with survival
   estimators; a 1..64 interaction schedule.
4. Cross-tier training-distribution registry and sampler, with frozen views and guarded factories; v5 sealed
   partitions in a separate evaluator process.
5. compose: fix the shortcut; add an objective registry, generated claims and identification, a batch admission
   driver, and fingerprint-distance dedup.
6. Weakness profiler, targeted generator, and curriculum with at least 20% uniform share and an entropy floor; a
   planted-weakness control (T.5, T.8).
7. World-proposal service, evaluator-side and out of process.
8. Positive controls in `src`, on at least 3 admitted v5 families, observation-only; the transfer experiment
   (SP-15.6).
9. A/B/C/D config with matched experience and compute (A/B runnable now on Tier 1; C/D once I and J land); Level 6
   adapter and methodology.
10. Gate tests `test_gate_T.py`, `test_gate_U.py`, `test_gate_SP_14_*`, `test_gate_SP_15_*`, and evidence for K.n2
    and S.n1.

## Skeptic review

Reviewer: skeptic agent, 2026-11-26, same worktree and HEAD (`dd745b9`). Nothing outside this section and the
`skeptic` key of `N_research.json` was modified (`git status` shows only the untracked `docs/v5/audit/`). Probe scripts
and logs: `/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/N_research_skeptic/`
(`s1..s9*.py`, `s*.log`). All runs were single-process with `OMP_NUM_THREADS=1`; none needed more than 20 s of CPU.

### What I tried to refute, and what happened

The audit gives no RETAIN verdicts, so I tested two things. First, the REPAIR and EXTEND verdicts that might deserve
something harsher. Second, the harsh verdicts and factual claims that rest on a single probe.

| Probe | Question | Result |
|---|---|---|
| s1 `s1_mechview_protocols.py` | Can the 4.0 protocols really not run a single v5 Tier 1 family (critical finding, p4)? | **Refuted.** `rapid.mechview.make_task(family, seed, validate=True)` (`rapid/mechview.py:138`) builds a `research.tasks.Task` for every lever-view family. 26 of 74 rapid families qualify (23 train, 3 val; primitive_graph 21, discrete_event 2, graph_dynamical 2, state_machine 1). `P.evaluate` (random/hold/hill_climb) and `P.internalisation` (model_based) ran end to end in 9 s. p4 called `research.tasks.make_task("net_routing")`, which is the wrong constructor. The other 48 families (discrete decisions, queries, experiments, programs), Tier 2 and Tier 3 still cannot run. |
| s3 `s3_obsonly_controls_fixed.py` | Is p7 ("positive controls collapse under observation_only") a defect of the instrument? | **Refuted as an instrument defect.** In `MechVecEnv` the legacy reward equals `scalars[1] - scalars[2]` (observed revenue minus observed cost, `envs.py:112`) to within 3e-8. Apart from the reward slot, the observations of the two modes are byte-identical. When the two test controls take their signal from observed net instead of the reward channel (a two-line change), observation_only gives exactly the legacy results: WeightLearner `internalised`, gain 0.9977; ContextLearner gain 0.8593, late minus early +0.3172. The collapse comes from the controls reading the reward slot (`tests/test_research_protocols.py` ContextLearner reads `scalars[0]`; WeightLearner reads `reward`). Two consequences: observation-only in the 4.0 harness is a one-parameter change, not a redesign driver; and the 4.0 "shaped" reward was simply observable net income, so 4.0 results are not informationally tainted. They do, however, rely on the environment naming net income as the objective. |
| s8 `s8_controls_on_v5.py` | How far is SP-15.6 / U.4 from runnable on new v5 families? | Closer than IMPLEMENT. On 4 v5 train families (mk_elastic, mk_drift, hc_conditional, cx_competing_causes), 24 training and 12 evaluation tasks, Level 1, all three controls were classified correctly in 14 s: WeightLearner `internalised` (gain 1.44, p 0.034), Hold `no_gain_detected` (0.0), ContextLearner `fast` only (gain 1.72, p 0.004). Still missing: the transfer experiment to a held-out family, a control with an independently established transferring capability, a known-negative transfer control, and the gate test. |
| s2 `s2_evaluator_tamper.py` | Can a learner alter the "independent" evaluator or fool the frozen checks? (U.3, U.7, SP-14.4) | **Yes, worse than p2.** (a) `checkpoint_digest` → `_parse_checkpoint` calls `torch.load(weights_only=False)` or `pickle.loads` on any learner-written file that starts with 0x80 (`protocols.py:267`, `:272`, `:275`). This executes learner bytes inside the evaluator. A learner that never learns (HoldLearner) saved a pickle whose payload rebinds `protocols._paired`. Result: `verdict: internalised`, `valid: true`, gain 0.9 [0.8, 1.0], p 0.0005, no warning, no violation. (b) Frozen verification uses the learner's own `state_hash()` when it defines one (`protocols.py:341-346`). A learner that keeps adapting across episodes during "frozen" evaluation and returns a constant hash got `frozen.verified: true, unchanged: true`, scoring 0.36 against Hold's 0.0. Run through `internalisation` with **0 updates**, it was certified `internalised` and `valid`; the only sign was a non-reproduction warning. |
| s4 `s4_family_partition_leak.py` | Are held-out *families* protected, or only held-out seeds? (T.7) | **Only seeds.** `tasks.train_stream` and `protocols.task_stream` accept the 4.0 `test` families that `research.sealed` exists to protect: the first item is `abm_network_fx:0` with `split: "test"`. `P.lifetime`, trained on the v5 val family `des_perishable` (seeds 0-7) and evaluated on the same family's seeds 900-903, ran to completion and reported `disjoint: {ok: true}`. The guard compares only task id and mechanism fingerprint (`protocols.py:186`, `:211`), and both differ for every seed. A "Level 4: entirely unfamiliar mechanism" result can therefore silently be a Level 1 result. The audit did not report this leak class. |
| s5 `s5_compose_admitted_static.py` | Does the p9 calibration shortcut survive admission, and are generated worlds just static optimisations? | **p9 is overstated for admitted worlds.** The probe ran on the 7 structures that passed full admission in p1, 3 seeds each (21 worlds). The anti-default 25/75 rule reached a **median 0.045** of the oracle's improvement, above 0.75 in only 2 of 21 worlds. p9's median 0.82 came from calibrated, rival-free worlds that had not been through admission. The best constant setting reached a median 0.76 of the oracle, and at least 0.95 in only 38% of worlds, so drivers and populations do create policy-relevant dynamics in most admitted worlds. 13 of the 21 instances were **uncalibrated** fallbacks (`compose.py:507-509`). Calibration succeeds or fails per seed, so one family mixes two different cost regimes. That contradicts the module's "the seed draws names and coefficients only". |
| s9 `s9_compose_populations.py` | Are compose's populations participant policies? (T.2) | No. Every population kind (none, rivals, learning_customers, word_of_mouth, mixed) yields a single agent `a0`; "rivals" are `adaptive_opponent`/`strategic_interaction` nodes in the causal graph. The kinds do change node sets and values. Several calibrated instances leave almost no room to improve: default 118,002 against oracle 119,011; 249,274 against 253,877. This confirms the audit's EXTEND and its criteria-weak flag. |
| s6 `s6_goalsampler_learnable_weakness.py` | Does `GoalSampler` really move away from weaknesses (p3 → REBUILD)? | **Only from hopeless ones, by design.** p3's weak family had success 0 and progress 0 forever, so the documented 0.05-0.95 success band (`curriculum.py:76`) drops it. With a *learnable* weakness (success 0.10, rising 0.02 per practice to 0.7), the weak family got more than the uniform share in **10/10 seeds** (mean 0.245 against 0.20; sign test p ≈ 0.001). It was front-loaded: 0.365 in the first quarter, 0.19 in the last, as it was mastered. With a weakness that is hopeless only at the hard levels, it got 0.186 (10/10 seeds below uniform). The learning-progress and uncertainty selection signal works. What is wrong is the arm space (4.0 goal contracts), the absence of generation and of a weakness taxonomy, dropping rather than decomposing hopeless arms, the trusted `split`, and the lack of any v5 user. |
| s7 `s7_In2_scanner.py` | Would the I.n2 scanner pass a broken corpus? | **Yes.** It missed 5 of 6 planted unsupported claims, for example "Crucible reproduces empirically realistic ten-million-view viral cascades with no hand tuning.", because "no", "not", "never" and "without" anywhere in a sentence count as a disclaimer. The test comment says "the evidence notes of this gate are scanned too", but no code scans them. Across the 681 files the scanner never reads, I found 1 hit, and it is a false positive (a MASTER_SPEC table). So the content is honest today, as the audit says, and REPAIR stands. |
| checker | Reproduce "all 25 N requirements NOT_STARTED" | Reproduced: `verify_all()` in-process gives `{'NOT_STARTED': 253}` overall and 25 of 25 for N, with no file written. |
| pytest | `tests/test_research_protocols.py -k "internalisation or sealed or timescales"`; `tests/v5/test_gate_I.py -k In2` | 13 passed (10.3 s); 1 passed (2.4 s). These only show that the 4.0 suite is green. s2, s4 and s7 show the defects these tests do not catch. |

I also re-read `compare_runs`/`combine_evaluations` (`protocols.py:1612-1700`, `:1771-1889`). They feed
`_per_task_mean` straight into `_paired`, exactly as p5 modelled, so p5's 94% false-positive rate applies to the
real code path. `to_threshold` uses `_f` (None → NaN) followed by a NaN-dropping mean, so censored seeds vanish, as p6
says. `_steps_pair` drops bootstrap resamples where either arm is censored (`protocols.py:1444`).

### Verdict changes

| Target | Auditor | Skeptic | Why (probe) |
|---|---|---|---|
| Fresh-process weights-only evaluation | REPAIR (M) | **REDESIGN (L)** | s2: the evaluator deserialises learner pickles, so learner code runs inside it and rewrote the statistics; frozen verification trusts the learner's own `state_hash`; a learner with 0 updates was certified `internalised`. The trust model (learner and evaluator share a process; integrity is self-reported) is the architectural assumption at fault. A subprocess alone does not fix it while the evaluator still unpickles learner bytes. |
| U.3 | REPAIR | **REDESIGN** | As above. 0 of the 9 checks in 15.4 run in a fresh process, and the ones that exist rely on learner self-report. |
| Sealed sets and partition protection | REPAIR (M) | **REDESIGN (L)** | s4: family-level partitions are never enforced; held-out families train freely and the run still reports `disjoint ok`. p8 and s2: every guard sits in a process the learner controls. v5 has no sealed partition at all (`PARTITIONS` maps val and test to the public `v5-eval-1`). The auditor's own work list (separate process or uid, frozen registry views) is itself a redesign. |
| T.7 | REPAIR | **REDESIGN** | `evaluation_partition_leaks == 0` fails by design. Leak vectors: seed-only guard (s4); mutable registry and unguarded `rapid.make` (p8); readable salt (p8); evaluator code execution (s2); `GoalSampler` trusting `split` (p3). |
| Independent evaluation for learners | EXTEND (L) | **REDESIGN (L)** | s2: the "independent" evaluator shares memory with the learner and runs its bytes. Scores come from private references, but a learner can rewrite how they are compared. 15.2 and SP-14.4 independence needs a separate-process verifier with a data-only interface. |
| U.7 | EXTEND | **REDESIGN** | Same evidence. The criterion `independent_evaluator_used == True` is a self-report and would pass today. |
| Adaptive curriculum (`GoalSampler`) | REBUILD (L) | **REDESIGN (L)** | s6: the selection signal targets learnable weaknesses (10/10 seeds above uniform). The scoring core is reusable; the arm space, generation, weakness profiling, random share and partition guard must be re-architected. "Fundamentally inadequate" is not supported. |
| T.5 | REBUILD | **REDESIGN** | As above. A planted learnable-weakness control already gives p ≈ 0.001 on 4.0 goal arms. What is missing is v5 worlds and the weakness-to-structure mapping. |
| SP-15.6 | IMPLEMENT | **EXTEND** | s1, s8: v5 Tier 1 families run through `transfer`/`internalisation`, and the controls classify correctly there. The transfer experiment, a genuinely transferring control with independently established ground truth, a negative control and the gate test are missing. That is extension work, not a missing subsystem. |

### Verdicts upheld, with corrected evidence

* **Research protocol harness: REDESIGN upheld, but two of its three stated reasons are wrong.**
  * Reason 1 says "no v5 Tier 1 family can be evaluated"; s1 shows 26 of 74 can.
  * Reason 2 says the legacy reward default blocks observation-only work; s3 shows this is a one-parameter switch
    with identical results.
  * REDESIGN still holds on reason 3 alone. The padded lever/gauge `Learner`/`EnvSpec` contract cannot represent the
    other 48 Tier 1 families, any Tier 2 domain, any Tier 3 civilisation, or Level 5's new action interfaces.
  * Effort is probably L rather than XL. Levels 1-4 (including Level 2 through rapid difficulty axes such as
    `observation_noise`) can be built on the existing harness through `mechview`; only Level 5 and Tiers 2/3 need
    the `crucible.experience` rebase.
* **Positive controls: EXTEND upheld.** The audit's own summary says REPAIR; its JSON and §4.8 say EXTEND. Critical
  finding 3 (controls collapse under observation_only) should read: the two test controls read the reward channel
  and need a two-line change (s3). The genuine limits remain: the controls live only in the test file, run on toy
  tasks, use `min_effect = 0.0`, and cover 5 of the 7 kinds.
* **compose: EXTEND upheld.** Critical finding 5 is overstated. On admitted worlds the anti-default rule reaches a
  median 0.045, not 0.82 (s5). Adding anti-default and quartile rules to admission's trivial set is still worthwhile,
  because calibrated rival-free worlds do carry the shortcut. New defect: per-seed calibration fallback mixes cost
  regimes inside one family (13 of 21 admitted instances uncalibrated). T.2 EXTEND is upheld by s9.
* **Statistics (U.8) REPAIR, learning curves (U.6, SP-15.3) REPAIR, matched budgets (U.5) EXTEND, I.n2 REPAIR**:
  upheld. The code paths were verified, and s7 showed the scanner failing.
* **IMPLEMENT verdicts** for the registry and sampler, the 50/30/20 mixture, world proposal, A/B/C/D, K.n2, S.n1,
  SP-14.x and the N gate tests: upheld. `experience.registry.make` is a cross-tier *constructor*. It has no version,
  sampler, mixture or reuse guard, and its default is `LEGACY` (`experience/registry.py:99`). s2 also means that any
  world-proposal service built in-process would be subvertible.
* **Not re-probed**: Level 6 adapter (EXTEND) and T.3, T.4, T.6, U.2. I found nothing that contradicts them.

### Additional findings

1. Evaluator code execution from learner checkpoints (`protocols.py:250-276`): `torch.load(weights_only=False)` and
   `pickle.loads` on learner-written bytes. This must be fixed with safe loaders (tensors, `npz` with
   `allow_pickle=False`, `torch.load(weights_only=True)`) in any redesign, fresh-process or not.
2. Self-reported integrity. `state_hash()` overrides the evaluator's own checkpoint hashing, and `freeze(True)` is
   checked only through the learner's own `frozen` flag. A gain together with a non-reproducing frozen evaluation is
   only a warning. It should invalidate a weights-only claim.
3. Held-out families are not enforced anywhere in the 4.0 harness. The partition is by seed range (`tasks.py:19-23`,
   `train_stream` at `:256`), so the sealed-set machinery protects seeds of families that training may already have
   visited.
4. `tests/test_research_protocols.py:26` hard-codes another session's scratch path as `CRUCIBLE_CACHE`. This is
   harmless when the path is absent, but it is a hidden cross-session dependency.

### Bottom line of the review

The audit's central conclusion holds: workstream N is essentially unimplemented, and no N requirement can pass in the
integrated system. Its integrity findings were too gentle. The instrument's trust boundary is the learner's own
process, so weights-only (U.3), partition protection (T.7) and evaluator independence (U.7) need redesign, not repair:
a non-learning learner can obtain an "internalised, valid" certificate in two different ways (s2). Three of its
negative factual claims were too strong:

* the harness does run 26 of the v5 Tier 1 families (s1);
* the observation-only "collapse" is a two-line control bug (s3);
* the compose shortcut largely disappears on admitted worlds (s5).

The `GoalSampler`'s selection signal is sound for learnable weaknesses (s6).

Verdict counts after this review:

* Subsystems (17): 0 RETAIN, 3 REPAIR, 4 EXTEND, 5 REDESIGN, 0 REBUILD, 5 IMPLEMENT.
* Requirements (25): 0 RETAIN, 4 REPAIR, 7 EXTEND, 5 REDESIGN, 0 REBUILD, 9 IMPLEMENT.

The machine-readable record is under the `skeptic` key of `N_research.json`: `overturned`, `upheld`,
`corrected_critical_findings`, `additional_findings`, `tests_run` and `verdict_counts_after_review`.
