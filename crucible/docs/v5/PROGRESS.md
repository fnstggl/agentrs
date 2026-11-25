# Crucible 5.0 progress

Project status: **IN_PROGRESS** (the checker decides; see `python -m crucible.acceptance.v5 --verify-all`).

## Milestone 0 - foundation (2026-11-25)

* Branch `feat/crucible-5-openworld` at worktree `.worktrees/crucible-5`, based on `c07098b`.
* Backup bundle `/workspace/backups/crucible-pre5-20261125.bundle` (crucible branches and both CAM 7.0 branches).
* MASTER_SPEC.md saved verbatim from the request (sha256 of the file recorded in the commit); Clarification 1 saved.
* Regression baseline: 1,076 tests, 1,074 passed, 2 skipped, 0 failed (Gate A evidence, raw).
* Crucible 4.0 performance baseline: `bench_openworld.py --quick` (log in the scratchpad, copied to evidence later).
* Subsystem map of Crucible 4.0 by eight readers: `docs/v5/map/*.json`.
* Compute scheduler `crucible.scale.slots`; acceptance manifest (241 requirements) and checker with tests.
* Architecture, contracts, workstreams, decisions, blockers, fidelity register, reproduction notes.

Requirements verified: 0 of 241.

## Milestone 1 - substrate, calibration and Tier 1 foundation merged (2026-11-25)

* ws-A substrate merged (7362000): artifacts, columns with batch events, ids, typed bus with same-transaction
  subscribers, 20 shared-primitive owners, attention, recorded inference, execution-mode registry. 4.0 kernel scenarios
  byte-identical to c07098b. Fixed a 4.0 order dependency in the deterministic cost model (sandbox/bridge.py).
  Full regression after the merge: 1,222 tests, 0 failed, 2 skipped (668 s).
* ws-E1 (a04393d) and ws-E2 (84231a1) merged: nine fitted behavioural models on KuaiRand-Pure, MIND-small, Criteo
  Uplift v2.1 and GH Archive, each beating its baseline on held-out data, plans committed before fits. The two
  registries are being unified (Gate F tests error until then: E2 card schema differs from E1's audit).
* ws-H0 merged (9879204): ActionSpace with 10 kinds, 28 primitives, generator, 5 v5 engines, fingerprints, admission,
  29 families (28 admitted on seeds 0-2: 21 train, 7 eval). Tier 1 and gate M/SP-7/SP-8 tests pass (200 passed).

Requirements verified: 0 of 241 (no official evidence run yet).

## Next actions

1. Merge wave 1 systems as they land: D1 encoders, D2 inference + benchmark, F1 commerce, F2 institutions,
   G innovation, L learner interface; and the calibration unification.
2. Wave 2 running: B population, C1 platforms, C2 recommendation/consumption/audience, H1-H3 Tier 1 families.
3. Then wave 3: I (15 domains), J (civilization), K (self-play), M (performance), N (curriculum, protocols), O (security).

## Compute in use

Slots budget 9 CPUs / 24 GB GPU / 8 jobs. GPU idle at the start of the project; no other GPU tenant observed.
Disk: 21 GB free after calibration data (1.8 GB under /workspace/data/crucible).
