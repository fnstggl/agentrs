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

## Next actions

1. Wave 1: A substrate, D encoders and inference service, E calibration (two parts), F commerce (two parts),
   G innovation, H Tier 1 foundation, L learner interface.
2. Merge each as it lands; then wave 2 (B population, C social, H families, M performance, O security).

## Compute in use

Slots budget 9 CPUs / 24 GB GPU / 8 jobs. GPU idle at the start of the project; no other GPU tenant observed.
