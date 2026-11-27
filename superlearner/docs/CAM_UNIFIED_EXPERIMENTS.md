# CAM 7.0 Unified — experiment register

Every confirmation is declared here (criteria, sets, seeds, baselines, statistics) and committed **before** its untouched
set is evaluated. Development sets may be inspected freely and are labelled as such. Results are appended with links
to raw result files; null and negative results are reported. `python -m superlearner.verify_completion` checks this
table against `docs/experiments/U-*.md` (every declared row has its pre-registration file at the declaring commit,
every pre-registration has a row) and checks experiment result files against the protocol below.

## Rules

* Statistics (repaired 2026-11-26, `experiments/stats.py`): studentised random-effects intervals over training seeds
  and evaluation worlds (method "t", calibrated degrees of freedom; the earlier two-level percentile bootstrap
  covered only 0.77–0.79 at 3 seeds and is no longer used for decisions); paired per cell when the seeds are matched,
  independent-seed intervals otherwise; 95% intervals; one pre-declared primary criterion per claim; the verdict is
  the interval rule of `stats.verdict` (DEMONSTRATED / FAILED / INCONCLUSIVE; BLOCKED when it could not run).
* Seeds: development primaries use ≥ 5 training seeds; a DEMONSTRATED verdict needs ≥ 3; 1B runs are single-seed
  demonstrations with no inferential claim.
* Evaluation seed ranges: Crucible `VAL_SEED0` + offset. Spent by CAM 7.0: 0–7 (dev), 60,000 (dev), 70,000 (C-2/C-3),
  80,000 (C-4). CAM Unified uses **100,000+** for new development sets and **200,000+** for the U-X confirmations;
  the DOM rebuild's confirmations (U-DOM-1, U-MR-1) use **500,000+** with the rapid val split (19 held-out
  families); sealed test families are untouched. Each (learner, seed) is evaluated on a confirm set exactly once.
* Every CAM condition runs reward_mode observation_only and passes the run-directory integrity scan; non-compliant
  reference conditions (native shaped reward, sparse verified terminal reward as the actor reward) are labelled.
* Every run records a provenance row (`eval/provenance.jsonl`: ckpt sha256, cfg hash, arch hash, code commit,
  reward_mode, decisions, updates, GPU-busy seconds, semantic-model tokens, loop and factory used, integrity scan)
  and compute: GPU-busy s, CPU s, semantic-model tokens, env steps, consequential decisions, simulated days, peak
  memory, wall time. Budgets are matched on environment decisions; a compute ratio above 2× is flagged.
* Separate: more data, more training compute, more inference-time search, sample efficiency, final competence.
* Weights-only conditions use `load_weights_only` (full lifetime reset; see memory/checkpoint.py) in a fresh process.

## Register

| ID | Experiment | Status | Declared in commit | Result |
|---|---|---|---|---|
| U-X1 | Exp 1 one/few-shot adaptation (A identical support, B agent-selected, C semantic, D delayed/partial) | declared (dev only; reward-supervised dev-scale design, superseded for DOM learners by U-DOM-1) | 4fe037d | not run on confirm |
| U-X2 | Exp 2 causal discovery vs KG / Thompson | declared (dev only; superseded for DOM learners) | 4fe037d | not run on confirm |
| U-X3 | Exp 3 parametric transfer & retention; negative-transfer diagnosis (A fresh, B weights-only, C +optimizer, D +optimizer+replay) | declared (dev only; superseded for DOM learners) | 4fe037d | not run on confirm |
| U-X4 | Exp 4 consequential long-horizon transfer over repeated OpenWorld lifetimes | not declared (only the U-OW-1 demonstration is feasible this rental) | | |
| U-DOM-1 | DOM-11 strategic dominance, conditions A–I (dom_dev inferential, 1B demonstration rows) | declared | 5b7095e | |
| U-MR-1 | Meta-learned rewards: Experiments D, E, F, I, J | declared | 5b7095e | |
| U-OW-1 | OpenWorld lifetime demonstration at 1B (ow_tiny, 2 seeds, descriptive) | declared | 5b7095e | |
| U-SC-1 | Compute-normalised scaling comparison (dom_dev / dom_150m / dom, descriptive) | declared | 5b7095e | |
| U-OW1 … U-OW7 | OpenWorld protocols (Part 16) | not declared (no CAM protocol claims this rental) | | |
