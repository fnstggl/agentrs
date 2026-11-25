# CAM 7.0 Unified — experiment register

Every confirmation is declared here (criteria, sets, seeds, baselines, statistics) and committed **before** its untouched
set is evaluated. Development sets may be inspected freely and are labelled as such. Results are appended with links
to raw result files; null and negative results are reported.

## Rules

* Statistics: two-level bootstrap over training seeds and evaluation worlds/tasks, paired per world where the design
  allows; 95% intervals; one pre-declared primary criterion per claim.
* Evaluation seed ranges: Crucible `VAL_SEED0` + offset. Spent by CAM 7.0: 0–7 (dev), 60,000 (dev), 70,000 (C-2/C-3),
  80,000 (C-4). CAM Unified uses **100,000+** for new development sets and **200,000+** for confirmations; sealed test
  families are untouched.
* Every run records compute: GPU s, CPU s, semantic-model tokens (prefill / decode), env steps, consequential decisions,
  simulated days, peak memory, wall time.
* Separate: more data, more training compute, more inference-time search, sample efficiency, final competence.
* Weights-only conditions use `load_weights_only` in a fresh process with `reset_experience_access`.

## Register

| ID | Experiment | Status | Declared in commit | Result |
|---|---|---|---|---|
| U-X1 | Exp 1 one/few-shot adaptation (A identical support, B agent-selected, C semantic, D delayed/partial) | not declared | | |
| U-X2 | Exp 2 causal discovery vs KG / Thompson | not declared | | |
| U-X3 | Exp 3 parametric transfer & retention; negative-transfer diagnosis (A fresh, B weights-only, C +optimizer, D +optimizer+replay) | not declared | | |
| U-X4 | Exp 4 consequential long-horizon transfer over repeated OpenWorld lifetimes | not declared | | |
| U-OW1 … U-OW7 | OpenWorld protocols (Part 16) | not declared | | |
