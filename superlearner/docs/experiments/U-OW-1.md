# U-OW-1 — OpenWorld lifetime demonstration at 1B (pre-registration)

Spec: `CAM_UNIFIED_SPEC.md` Part 17 Experiment 4 (minimal), meta-reward spec Experiment K (partial); plan
`CAM_UNIFIED_REDESIGN_PLAN_2026-11-26.md` §5 E4. Manifest: A18 (partial), X04 (partial), DOM-13 (partial).
Committed **before** any of the lifetimes below is run. This is a **demonstration**: two seeds, no inferential
claim, no EMPIRICALLY_DEMONSTRATED status can rest on it (`verify_completion` requires ≥ 3 seeds for that).

## System and worlds

* CAM: the 1B `dom` preset (32 × 1536 × 24, Qwen3-4B), weights from the E1 checkpoint of U-DOM-1 (the checkpoint
  sha256 is recorded in the provenance row), reward_mode observation_only, continual learning on during the
  lifetime, memory off (declared).
* World: Crucible 5.0 `ow_tiny` open_economy, observation-only adapter, **2 training-range seeds** (100,000 ≤ seed <
  1,000,000, never a val/test seed), one lifetime of **8 simulated days** each (~1,100 decisions), through
  `runtime.run_lifetime` with the TraceRecorder on.
* References on the same two worlds at matched decisions: `ExampleInductor` (scripted, cheap), `FrozenSemanticAgent`
  with Qwen3-4B (token-matched to CAM's recorded semantic tokens), untrained CAM (initial weights of E1).

## What is reported (fixed in advance)

Verified success (operator-side evaluator) per lifetime; executed-ok rate of CAM's actions; distribution of
goal-directed action classes; how often the decision evaluator vetoed or regenerated candidates; verified net income
trajectory; compute (decisions, GPU-busy seconds, semantic tokens, wall time) for every agent.

## Reading rule

* **Demonstrated (descriptive):** CAM reaches verified success in at least one of the two lifetimes AND its
  executed-ok rate ≥ 0.5.
* **Failed (descriptive):** 0 / 2 verified successes, or executed-ok rate < 0.5. Recorded as a negative result.
* Comparisons against the references are reported side by side without a test.

## Amendments

None.
