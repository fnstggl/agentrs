# Crucible 5.0 progress

The machine-readable ledger is `docs/v5/PROGRESS.json` (every Part 1.5 field per milestone, exact remaining
requirement ids, interrupted workstreams and their reassignment); validate it with
`python -m crucible.acceptance.v5 progress validate` and add a milestone with `... progress milestone`. This file is
the narrative.

Project status: **IN_PROGRESS** (the checker decides; see `python -m crucible.acceptance.v5 --verify-all`).

## Milestone 0 - foundation (2026-11-25)

* Branch `feat/crucible-5-openworld` at worktree `.worktrees/crucible-5`, based on `c07098b`.
* Backup bundle `/workspace/backups/crucible-pre5-20261125.bundle` (crucible branches and both CAM 7.0 branches).
* MASTER_SPEC.md saved verbatim from the request; Clarification 1 saved. [Corrected 2026-11-26 by workstream P: commit 4a7591e's message carries no hash. The SHA-256 7b35feb343ce... is now anchored in `acceptance/v5/requirements.py` (MASTER_SPEC_SHA256) and in the acceptance ledger; the file is unchanged since 4a7591e.]
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

## Milestone 2 - wave 1 systems merged (2026-11-25)

* Merged: D1 encoders and media tooling (7ba706b), D2 inference service, semantic participants and model benchmark
  (a423433), F1 subscriptions/products/exchanges/business lifecycle (cc74374), F2 institutions/commitments/labour
  (c30d1ac), G capability registry/needs graph/software ecosystem (8d242a2), L learner interface (d85c8df); the Tier 2
  domain framework (f08ddb8). 61 kernel modules load, 19 of them new.
* Integration regression (tests/, without the gate tests): 1,548 tests, 3 failures, 2 skipped. Two failures were a race
  with an edit in progress (rerun: pass); one is the calibration registry split being unified.
* Wave 3 started: I1-I4 (12 domains), N1 (registry, curriculum, proposals), N2 (protocols, positive controls), O (security).

## Next actions

1. Merge wave 1 systems as they land: D1 encoders, D2 inference + benchmark, F1 commerce, F2 institutions,
   G innovation, L learner interface; and the calibration unification.
2. Wave 2 running: B population, C1 platforms, C2 recommendation/consumption/audience, H1-H3 Tier 1 families.
3. Then wave 3: I (15 domains), J (civilization), K (self-play), M (performance), N (curriculum, protocols), O (security).

## Compute in use

Slots budget 9 CPUs / 24 GB GPU / 8 jobs. GPU idle at the start of the project; no other GPU tenant observed.
Disk: 21 GB free after calibration data (1.8 GB under /workspace/data/crucible).

## Milestone 3 - recovery after the usage limit (2026-11-26)

* 13 workstream agents (B, C1, C2, H1-H3, I1-I4, N1, N2, O) were stopped by the weekly usage limit on 2026-11-25.
  Their uncommitted work was committed on their own branches, preserved as `recovery/c5-*` branches on GitHub, and
  merged into this branch together with the calibration-registry unification (all nine models in one registry).
* Full regression of the consolidated tree (tests/, without the gate tests): 1,648 tests, 5 failed, 2 skipped
  (`docs/v5/evidence/integration/regression_consolidated_g*.xml`). Four failures are Tier 1 tests: three still assert the
  foundation's engine count and module boundaries (the expansion added engines; `program_exec` uses the sandbox) and
  one is a real bug in the new family `cxs_selected_layers` (batch path shape error); the fifth is the 4.0 discovery
  test, which does not list the new error code `media_error`. [Count corrected 2026-11-26 by workstream P from the JUnit.]
* Interrupted work is incomplete: no Tier 2 domain is registered yet, the new Tier 1 families have not been through
  admission, population/social gate tests have not been run officially. Requirements verified: 0 of 241.

## Milestone 4 - recovery audit and redesign (2026-11-26, branch feat/crucible-5-complete)

* New integration branch `feat/crucible-5-complete` from a881d18 (`feat/crucible-5-openworld` left untouched).
* Clarification 2 (observation-only default) and 3 (recovery preamble) recorded; manifest grew to 253, then 354
  requirements (D-22 additions from the audits, D-23 stricter criteria, D-24 hardened checker).
* Fifteen independent audits, each with an adversarial skeptic review (`docs/v5/audit/`). Common finding: subsystems
  worked in isolation; six disjoint populations; no fitted model used in-world; lexical content judgement; checker
  could be fooled (11/11 fabrication probes). Decisions D-14..D-21 (one population, one semantic space, domains as
  civilization configurations, usefulness through activities, observation-only everywhere, metering outside the
  measured process, kernel repairs); `ONE_WORLD_CONTRACTS.md`.

## Milestone 5 - wave 1 and wave 2a merged (2026-11-26, 87a6c54)

* Ten workstream branches merged after a second usage-limit interruption (all their work had been committed; the
  three worktrees with uncommitted changes were committed as recovery commits and pushed first).
* Integrated regression (tests/test_*.py, 1,643 tests): 18 failed, 24 errors; 21 of these are caused by the
  orchestrator's root-only TMPDIR (postgres initdb, Chromium profile directory), the rest are merge interactions
  (substrate streams/facts/modes/resume/history, population core, semantic cache, econ_work, innovation needs, rapid
  fingerprints, scale worker recycling, one 4.0 world test) assigned to the owning workstreams in wave 3.
* Known integration bug: two kernel modules named `space`; `civ.generate` fails -> integration fix running.
* Machine-readable ledger: `docs/v5/PROGRESS.json` milestone M5 (remaining requirement ids, failing tests, next actions).
* Requirements verified: 0 of 354 (no official evidence run yet; the checker reports IN_PROGRESS).
