# Audit P_acceptance: acceptance checker, manifest completeness, test integrity, Gates A and X

Auditor scope: workstream P (Gates A and X, SP-1.3, SP-1.5, SP-18, SP-20, SP-21, and the RP.* entries added for the
recovery preamble). Audited tree: `feat/crucible-5-complete` at `a881d18`. While the audit ran, the orchestrator
committed `dd745b9` on the same branch (09:08 UTC). It adds Clarification 2 (observation-only default), Clarification 3
(recovery preamble), `docs/v5/RECOVERY_PREAMBLE.md` and twelve manifest entries (OO.1-OO.8, RP.1-RP.4), for 253 in
total. The findings below cover both commits and say which one they apply to. No existing file was modified. All
probes ran in throwaway git repositories or in `git clone --shared` copies under the scratchpad:
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/P_acceptance/`.

## Summary

* **The acceptance checker can be fooled in every way I tried.** All 11 fabrication probes produced `VERIFIED` and
  `CRUCIBLE_5_COMPLETE`. On a clone of the real repository, deleting 252 of the 253 requirements from the manifest makes
  `python -m crucible.acceptance.v5 --verify-all` exit 0 with `project status: CRUCIBLE_5_COMPLETE`. The official
  `run` + `review` pipeline accepts a gate test that measures nothing and records hard-coded metrics. The checker does
  check hashes, JUnit outcomes, staleness of the listed files, a claimed `VERIFIED` and review hashes, so it is not a
  status-string reader. But its trust model rests on self-reported metrics, a self-reported tree state, unauthenticated
  reviewer names and a manifest it never checks against the spec. SP-21 does not hold today.
* **No requirement can be verified yet.** All 253 entries have `implementation_files: []` (the workstream reports
  in `docs/v5/workstreams/*.json` list files for about 103 requirements, but none were merged into the manifest). 132
  entries name test files that do not exist: every Gate A, B, G, N, O, P, Q, T, U, V, W, X, OO and RP test, and 23 SP
  tests. No official evidence or review exists.
* **Manifest completeness.** At `a881d18` the 8 observation-only requirements and the recovery-preamble requirements
  had no entries. `dd745b9` adds them (OO.1-OO.8, RP.1-RP.4), but their criteria are partly weak or undefined (see
  below). About 50 further mandatory spec statements have no entry. The most important: one population shared by all
  platforms; fitted behavioural models actually driving in-world decisions; every gate re-demonstrated inside an
  integrated civilisation; the 60 named Tier 1 families; the INCOMPLETE_BLOCKED status; never removing a failed
  requirement from the manifest; the final report's nine-way separation; and Clarification 1's four kinds of
  validating learner.
* **Observation-only default is violated in code.** `reward_mode` defaults to `legacy_shaped_reward` at every public
  entry point (`experience.registry.make`, `api.gym_env.CrucibleEnv`, `api.pettingzoo_env`, `research.envs.MechVecEnv`,
  `research.bridge.KernelMechEnv`, `selfplay.arena`, `scale.rollout`, `experience.reference.ReferenceLearner`).
  `experience/modes.py:3` states "default everywhere". No test checks the amendment.
* **4.0 baseline.** Benchmark-defining packages are untouched since `c07098b`. The sealed store has not been accessed
  since 2026-11-23 and v5 code never touches it. No 4.0 test was modified or deleted. The regression suite is not
  green (results below).
* **Test integrity.** Static scans of 433 v5 test functions found no skip markers in gate tests, no early returns or
  `except: pass` in test bodies, and no mocks apart from four `monkeypatch` redirections of a model directory. Gate tests
  record honest zero metrics and fail when a subsystem is absent. Two real weaknesses remain. Gate tests run in
  purpose-built partial worlds (a social-only world with 2-6 hand-driven actors, a commerce-only world), never in an
  integrated civilisation, and the manifest never requires that. 14 gate tests contain no assertion, so a pytest
  pass is not evidence: only `--verify-all` on official evidence is.

## 1. The acceptance checker (`src/crucible/acceptance/v5/`)

### What it does (read in full)

`checker.check_requirement` (`checker.py:77-188`) does six things for each manifest entry:

* checks that the entry is well formed and that its criteria equal the last `criteria_history` version;
* checks that the listed implementation and test files exist;
* loads `docs/v5/evidence/gate_<G>/<id>.json` and evaluates each criterion on the record's own `metrics`;
* checks that the evidence commit is in HEAD's history and is not older than the last change to the listed files;
* re-hashes cited artifacts and the attached JUnit file, and requires every test case of the listed test files in that
  JUnit to pass, none skipped;
* requires a review file whose `evidence_sha256` matches and whose reviewer differs from the evidence's author.

`verify_all` (`checker.py:191-210`) sets `ok = all(rows VERIFIED)`.

### Probes (all reproducible)

| # | Attack | Result |
|---|---|---|
| P1 | Hand-write the evidence JSON, JUnit XML and review JSON. The implementation is `raise NotImplementedError` and the test is `assert True`. No test is run and `record()` is never called. | **VERIFIED**, project `CRUCIBLE_5_COMPLETE` |
| P2 | Manifest with zero requirements | **ok=True, CRUCIBLE_5_COMPLETE** (`all([])`, `checker.py:207`) |
| P3 | Delete a hard requirement from the manifest JSON, then commit | **VERIFIED / COMPLETE**: nothing compares the manifest with `manifest.derive()` |
| P4 | Weaken the criteria by rewriting both `criteria` and `criteria_history[0]` | **VERIFIED**: `checker.py:84-89` only checks the *last* version and later decisions |
| P5 | Evidence says `tree.clean: true` but the tree is dirty | **VERIFIED**: `checker.py:128` trusts the record's own field |
| P6 | The JUnit lists 1 of the file's 3 test cases (the 2 failing ones are omitted) | **VERIFIED**: `checker.py:160-166` checks only the cases that are present |
| P7 | Placeholder test (returns on ImportError, asserts nothing) | **VERIFIED**: there is no placeholder detection |
| P8 | `produced_by` names a test outside the requirement's test files | **VERIFIED**: `checker.py:157` accepts any case in the JUnit |
| P9 | Metric `Infinity` in a hand-written record | **VERIFIED**: `criterion_met` (`checker.py:39-49`) has no finiteness check |
| P10 | The requirement's `tests` list is swapped to a trivial file | **VERIFIED**: tests and implementation lists are not under change control |
| P11 | Evidence from an old commit, after a change to an imported module that is not listed | **VERIFIED**: staleness covers only the listed files (`checker.py:134-139`) |
| P12 | Official pipeline on a clone of the real repository: a gate test does `record("A.2", {"v4_definitions_changed": 0, "v4_definitions_checked": 25})` and nothing else; then `run --requirement A.2` with `CRUCIBLE_V5_AUTHOR=builder`, then `review --reviewer ws-P` from the same shell | **A.2 VERIFIED** |
| P13 | On that clone, keep only A.2 in the manifest and run the real CLI `--verify-all` | **exit 0, "project status: CRUCIBLE_5_COMPLETE"** |
| P14 | On the real manifest, weaken B.1 from 10,000 to 1 participant (with history rewritten) | No criteria problem reported; `derive()` still says 10,000 |

Scripts and logs: `probe_checker_fool.py` / `.log` / `.json` (P1-P11), `verify_all_clean.log`, and
`probe_real_manifest_weaken.log`. P12 and P13 ran on the clone `clean/` with a probe commit; the transcript is in this
report.

### Root causes

1. **The manifest is not anchored to the spec.** `verify_all` iterates whatever `ACCEPTANCE_MANIFEST.json` contains.
   It never compares ids, wording, criteria, tests or dependencies with `manifest.derive()` and `requirements.py`. On
   top of that, `evidence.EVIDENCE_PATHS` (`evidence.py:26`) exempts the manifest from the dirty-tree check, so edits to
   it never even make evidence "uncommitted".
2. **Metrics are self-reported and never reproduced.** The checker evaluates criteria on numbers the producing test
   wrote. Nothing re-executes the producing test, in a clean clone or elsewhere, so a test that writes constants is
   indistinguishable from one that measures.
3. **Provenance is self-asserted.** `tree.clean`, `tree.commit`, `produced_by` (from `PYTEST_CURRENT_TEST`) and
   `author` (from `CRUCIBLE_V5_AUTHOR`, default `"unknown"`, `evidence.py:93-94`) are all written by the producer.
   The JUnit file is not bound to the run id or commit.
4. **Review identity is a free-text string.** `_review` (`__main__.py:80-97`) writes a JSON file with any
   `--reviewer` name. It verifies nothing, re-runs nothing, and is not committed or signed. "Reviewer is not the
   author" compares two strings that the same process can choose.
5. **Coverage holes.**
   * The JUnit must contain only *some* cases of each file; collection completeness is not checked.
   * Staleness is limited to the listed files, which are all empty today, and ignores transitive imports.
   * `dependencies` are never checked, so SP-18 could be VERIFIED while X is not.
   * There is no placeholder or trivial-test detection, and no finiteness check on metrics.
6. **Status vocabulary.** `project_status` is only `IN_PROGRESS` or `CRUCIBLE_5_COMPLETE`. `INCOMPLETE_BLOCKED`
   (Parts 1.5 and 21, Clarification 3.8) is never produced.
7. **Self-reference.** X.1 (`unverified_requirements == 0`) counts a set that contains X.1 itself, and SP-18/SP-20
   depend on X. A fixed-point rule is needed: X.1 counts everything except the X and SP-18/20 entries, and a final
   `--verify-all` closes the loop.

What works, with evidence:

* An honest official run works end to end. On a clean clone at `dd745b9`,
  `CRUCIBLE_V5_AUTHOR=audit-probe python -m crucible.acceptance.v5 run --requirement SP-4.8` measured real metrics
  (2 social-to-purchase chains, 0 hard-coded bonuses), recorded a clean tree and attached the JUnit. The checker then
  reported `IMPLEMENTED_UNVERIFIED` with exactly two problems: "no implementation files listed" and "no independent
  review" (`official_run_SP_4_8.log`). The blocker for honest gates is therefore the unmerged manifest file lists and
  the missing QA reviews, not the pipeline.
* `tests/test_v5_acceptance_checker.py` passes (13 passed, 1.73 s).
* On the real tree, `--verify-all` exits 1 and reports 241 (now 253) NOT_STARTED with honest reasons
  (`verify_all_clean.log`).
* A claimed `VERIFIED` without evidence is flagged.
* Editing evidence after a review, stale evidence for the listed files, failed or skipped JUnit cases and self-review
  under the same name are all rejected.

### Work needed for SP-21 (checker)

1. In `verify_all`, require that the manifest ids equal the `derive()` ids exactly. Require that each entry's
   requirement text and criteria equal `derive()`'s, or that its history chain from `derive()`'s *initial* criteria
   cites recorded decisions. Bring `tests`, `implementation_files` and `dependencies` under the same change control.
   Remove the manifest from the dirty-tree exemption for structural fields.
2. Add a reviewer re-execution step. `review --rerun` should clone the evidence commit into a fresh directory, run the
   producing test and the requirement's test files, and compare the reproduced metrics with the record (exact for
   integers and booleans, a declared tolerance for floats). The review is valid only if it carries the reproduction's
   JUnit and metrics hash.
3. Make identities real: commit the review files, record the git author and agent session id of both the evidence
   commit and the review commit, and require them to differ. Bind each review to `sha256(evidence)`,
   `sha256(criteria)` and `sha256(requirement text)`.
4. Make JUnit complete and bound:
   * at the evidence commit, take `pytest --collect-only -q <files>` and require every collected node id in the
     JUnit;
   * require `produced_by` to be one of the requirement's own tests;
   * embed the run id and commit as JUnit properties (`record_testsuite_property`) and check them.
5. Make staleness transitive: during the producing test, record the hash of every imported `crucible.*` module file
   (`sys.modules`), and reject the evidence when any of those files changed after the evidence commit.
6. Check that dependencies are VERIFIED, reject non-finite metrics, emit `INCOMPLETE_BLOCKED` when the only
   unverified entries are BLOCKED with a recorded blocker, and define the X fixed point.
7. Add a placeholder and trivial-test detector, used by X.7. Statically, it reports bodies without measurement or
   assertion, constant `record()` metrics, early returns and `except: pass`. Dynamically, it runs mutation testing
   (Workstream P's mandate includes mutation tests; none exist): mutate the implementation files of a requirement and
   require the gate test's metrics or outcome to change.
8. Add regression tests for every probe P1-P14 to `tests/test_v5_acceptance_checker.py`. The current
   `test_complete_evidence_is_verified` (line 68) is itself P1: it hand-writes the JUnit and the review and expects
   VERIFIED.

## 2. Gate A (baseline preservation)

**Regression (A.1).** See section 6. The full suite ran once through the slot scheduler (`--cpu 1 --gpu-mb 0`).

**4.0 definitions (A.2).**

* `git diff --stat c07098b HEAD` shows **no change** under `src/crucible/mechanisms`, `worldgen`, `goals`,
  `usefulness`, `economy` and `actors`.
* Thirteen 4.0 files were modified: `api/gym_env.py`, `api/pettingzoo_env.py`, `api/local.py`, `core/kernel.py`,
  `core/state.py`, `research/envs.py`, `research/bridge.py`, `selfplay/arena.py`, `scale/rollout.py`,
  `sandbox/bridge.py`, `services/identity.py`, `integrations/agentrs_cam2.py` and `__init__.py`. In each, the 4.0
  behaviour is kept under `reward_mode="legacy_shaped_reward"`, which is the default.
* No 4.0 test file was modified or deleted (`git diff --diff-filter=MD c07098b HEAD -- crucible/tests` lists only v5
  additions).
* The 4.0 registry has 25 mechanism families: 7 train, 7 val, 11 test (`probe_v4_defs.log`).
* **Conflict to resolve:** Clarification 2 requires the default to become `observation_only` everywhere. That changes
  the default of these 4.0 classes, so every 4.0 reproduction (A.5) must then pass `legacy_shaped_reward` explicitly,
  and A.2 must compare definitions by content (the mechanism and family code plus split tables), not by constructor
  defaults.

**4.0 world configurations (A.3).** No gate test exists. The 4.0 bench discovers 74 OpenWorld families
(`docs/v5/evidence/baseline_perf/openworld_quick_rerun.json`), in addition to the 25 mechanism families. The criterion
`v4_families_run >= 16` is undocumented and far below the population.

**Sealed sets (A.4).**

* `~/.crucible/sealed_sets.json`, `sealed_tasks/ow-final.g1.66bad809.json` and `sealed_access.jsonl` were last
  modified on 2026-11-23 at 05:10, before v5 started (2026-11-25).
* `ow-final` shows `accesses: 1`, `budget: 1` and `contaminated: false`. The access log has four entries, all from
  2026-11-23.
* Hashes: `sealed_sets.json` f0e3d060..., the task file faa17770..., the access log 1d952dd8... I read only the
  metadata, not the task contents.
* No v5 module references the sealed store. `grep -i sealed` over v5 code finds only the experience store's "sealed
  shards" and a benchmark prompt.
* None of the 74 v5 rapid families (55 of them train) is named like a 4.0 val, test or sealed family. This check is
  name-based only. A structural (fingerprint) comparison against the 4.0 held-out families is still needed and belongs
  to Gate M / workstream H.
* The sealed files were unchanged after the regression run (re-hashed at the end; see section 6).

**Prior results (A.5).** Not attempted by anyone. Candidates: 4.0 acceptance records A-J under
`docs/experiments/acceptance/`, the `demo_*.json` records and the 4.0 bench. All of them must run with
`legacy_shaped_reward` passed explicitly once the default changes.

## 3. Gate X and the persistent engineering state

* **X.1-X.n1.** `tests/v5/test_gate_X.py` does not exist. The workstream table assigns `docs/v5/reviews/` and
  `tests/v5/test_gate_X.py` to P (`WORKSTREAMS.md`); neither exists.
* **All X criteria are vacuous `== 0` counts with no denominator.** "0 clean-checkout failures" holds trivially when
  nothing was reproduced; the same is true of "0 missing documents" without a document list, "0 open defects" without
  a defect register, and "0 placeholder tests" without a scan count. Each needs a companion `>= N` coverage metric.
* **Clean checkout (X.3).** In a fresh `git clone --shared` at `dd745b9`, `test_v5_acceptance_checker.py` and gates
  SP-4.2, SP-4.7 and SP-4.8 pass (16 passed, 10.6 s, `clean_checkout_subset.log`). This shows feasibility, not X.3.
  Calibration tests read `$CRUCIBLE_DATA` (default `/workspace/data/crucible`, 1.8 GB, not committed, not on a
  persistent volume). They fail rather than skip when the data is absent (`tests/test_v5_calibration_commercial.py:4`),
  which is correct, so a clean-room run must re-acquire the datasets first.
* **Disconnected workstreams (X.8).** `src/crucible/civ/__init__.py` and `src/crucible/multiagent/__init__.py` are
  5-line docstrings. No Tier 2 domain is registered: Gate S records `tier2_domains_present = 0` and fails by design.
  Workstreams B, C1, C2, H1-H3, I1-I4, N1, N2 and O have no workstream report. X.8 currently fails.
* **Accurate reporting (X.9).** Claims in the documents that the evidence does not support:
  * `ARCHITECTURE.md:133-134`: "A lightweight reference learner proves real parameter updates through this interface
    in all three tiers". Tiers 2 and 3 do not exist (S.2 and S.3 fail).
  * `PROGRESS.md` milestone 0: "MASTER_SPEC.md saved verbatim ... (sha256 of the file recorded in the commit)". Commit
    `4a7591e`'s message contains no hash.
  * `PROGRESS.md` milestone 3 counts "Four failures are Tier 1 tests ... one is a real bug ... one 4.0 discovery test"
    against "5 failed". The JUnit shows 3 Tier 1 contract tests, 1 Tier 1 bug and 1 discovery test.
* **SP-1.3.** All ten required documents exist. `MASTER_SPEC.md` has one commit (`4a7591e`) and sha256 `7b35feb3...`,
  but no hash of the user's original text is recorded, so "unchanged" can only mean "unchanged since first commit".
  `WORKSTREAMS.md` still names `feat/crucible-5-openworld` as the integration branch. `RECOVERY_PREAMBLE.md` and the
  clarifications were added at `dd745b9`.
* **SP-1.5 / RP.3 (ledger).** `PROGRESS.md` has four milestones, but none carries all eight fields Part 1.5 requires:
  * milestone 3 has no branch or commit, no compute usage and no next actions (the "Next actions" block is stale from
    milestone 2);
  * no milestone lists the exact remaining requirement ids, only "0 of 241";
  * there is no machine-readable ledger.

  The local remote-tracking refs show `origin/feat/crucible-5-complete` and `origin/feat/crucible-5-openworld` both at
  `a881d18` (pushed). `dd745b9` is not shown as pushed (no fetch was made; local refs only).
* **Old branch immutability (Clarification 3.1).** The reflog shows `feat/crucible-5-openworld` fast-forwarded from
  `1ff011b` (the last reported milestone) to `db70810` and then `a881d18` on 2026-11-26. The new branch was created from
  `a881d18` at 09:01. Nothing was lost (fast-forward only), but the "immutable" branch moved during recovery.
* **RP.4 (durable artifacts).**
  * The nine fitted models (18 files, 2.4 MB) are committed and on origin, so they are durable.
  * The datasets (1.8 GB) and `/workspace/backups/*` are on container storage
    (`vast-capabilities: workspace_is_volume = false`), so they are not durable. The `datasets` entry in the newest
    backup is a symlink to `/workspace/data/crucible`.
  * `required_artifacts_unpreserved` has no defined artifact list.
* **Secrets.** No `github_pat_` string appears in the working tree or in any commit reachable from any ref. The remote
  URL holds no credentials, and no git credential helper is configured.

## 4. Manifest completeness

253 entries at `dd745b9`: 170 gate items, 21 gate notes, 50 SP items and 12 amendment items. The unit test
`test_the_real_manifest_covers_every_spec_gate_item` checks only the Part 17 bullets. Mandatory statements with **no**
entry are listed in `P_acceptance.json` (`missing_requirements`, 52 items). The most consequential:

* **Clarification 2 (observation-only).**
  * Operation with an externally assigned objective in observation-only mode (11.3: "the learner receives its
    objective, if one exists").
  * Self-generated hypotheses and objectives without automatic reward in every tier (SP-11.4 needs only one
    round trip).
  * Verified outcome labels never turned into intermediate rewards.
  * No CAM internal reward function inside Crucible.
  * A dedicated implementation subagent plus an independent verification subagent for OO.*.
  * Every public entry point defaulting to `observation_only`. OO.1 counts "families" but not entry points, and all of
    them default to legacy today.
* **Recovery preamble (Clarification 3).**
  * Old branch kept immutable.
  * No force-push.
  * Secrets never exposed.
  * Unfinished work preserved in recoverable commits (stated in RP.3's text but given no criterion).
  * Interrupted sub-agents' work reassigned (Part 1.5).
  * The `INCOMPLETE_BLOCKED` status.
* **Integration.**
  * Every gate's capability re-demonstrated inside an integrated Tier 3 civilisation rather than a purpose-built
    partial world.
  * All platforms sharing one participant population (Part 4).
  * Fitted calibration models actually driving Class 1 decisions in the running world (Parts 3.2 and 5).
  * The seven Phase 3 vertical slices.
* **Breadth.**
  * The 60 named Tier 1 families of 8.1-8.10. M.1 and M.3 count totals and categories only.
  * 7.2 superficially similar worlds that obey different causal rules.
  * 7 `IndependentlyGenerateGoal`.
  * 12.2 cross-play and league evaluation.
  * 13.3 backend benchmarks.
  * 13.4 the nine inference-service features.
  * 15.2 the list of excluded outcome metrics.
  * 15.7 separation of development and confirmation experiments.
  * Part 16: consent, moderation and abuse prevention.
* **Process and reporting.**
  * The Part 20 final report separating the nine categories (implemented, verified, calibrated, approximation,
    scripted, learned, not established, failed, blocked), with actual hardware, model sizes, population sizes and world
    counts.
  * Phase 7: "do not remove a failed requirement from the manifest". The checker does not enforce it (P2, P3).
  * Clarification 1: validation with scripted agents, small neural RL agents, statistical learners **and** lightweight
    LM agents. Gates R and S use only `ReferenceLearner`.

Weak criteria in P's entries: see `requirements[].criteria_weak` in the JSON. In short:

* A.1 counts tests (`>= 1076`) instead of requiring each baseline node id to pass.
* A.2 checks 25 definitions and A.3 runs 16 families, against more than 99 in 4.0.
* X.2-X.9 are vacuous zeros.
* SP-1.3 checks presence only.
* SP-1.5 requires `>= 1` milestone.
* SP-18 and SP-20 count rows and deliverables without tying them to checker verdicts.
* SP-21 rests on two self-declared booleans with no defined attack suite.
* RP.3 asks only for a single pushed head.
* RP.4 has no artifact list.
* RP.1 conflicts with Clarification 1's "preserve compatibility": `integrations/agentrs_cam2.py:40` and
  `integrations/agentrs_sl.py:34` import `agentrs.cam2` and `agentrs.sl` lazily, as optional 4.0 adapters.

Outside P, I also noted two weak criteria. SP-2.3 requires 17 primitive kinds against 19 listed in 2.3, and SP-3.1
requires 15 identity fields against 18 listed in 3.1.

## 5. Test integrity (`tests/v5/`, `tests/test_v5_*.py`)

Scans: `scan_tests.py` gives per-test assert counts, trivially-true asserts, constant-metric `record()` calls and
gate-test coverage. `scan_returns.py` finds `return` and `except: pass` in test bodies, excluding nested helpers.
Results in `test_scan.json`, `test_scan.log` and `scan_returns.log`.

* 433 test functions; 123 gate tests in 39 files.
* 0 `return` statements and 0 `except: pass` in test bodies (16 flagged by the first scan were all nested helpers,
  checked by hand).
* Skip markers: none in `tests/v5`. `tests/test_v5_substrate_inference.py:7` uses `pytest.importorskip("torch")`.
  This is a silent-skip path, harmless here because torch is installed, but against INTERFACE_CONTRACTS rule 7
  ("a test that cannot run fails").
* Mocks: four `monkeypatch.setattr(R, "MODELS_DIR", ...)` calls. They redirect the model directory to test refusal of
  tampered checkpoints, which is legitimate.
* Constant-metric `record()` calls (3): `test_gate_E.py:211`, `test_gate_S.py:59` and `test_gate_S.py:74`. Each is an
  honest "nothing measurable yet" record followed by `pytest.fail`. The pattern is good.
* Trivially-true asserts: `test_v5_semantic_media.py:147,149` compare constants, inside a bit-exactness test.
* 14 gate tests have no assertion (H.1-H.8, H.n1, F.4, F.5, F.n1, C.3, SP-3.2b). By design they only record
  metrics. Their pytest pass therefore means nothing, and integrity rests entirely on the checker, which is the weak
  link above.
* 134 manifest ids are never recorded by any test, including all of P's.
* **Not integrated.** The social, commerce and population gates construct their own partial worlds:
  `tests/v5/social_helpers.py:19-25` builds the social modules only, with 4 named participants admitted through
  `admin.join`, and `tests/v5/_commerce_kit.py:8-21` builds 15 commerce modules. That is acceptable for unit gates. But
  no manifest entry requires the same capability inside a civilisation, and Tier 3 does not exist.

## 6. Full regression (through slots)

Command, run from `crucible/` at the worktree (HEAD `dd745b9` at collection time; the diff from `a881d18` touches only
docs and `acceptance/v5`):

```
PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 0 --label "audit: full regression" -- \
  /venv/main/bin/python -m pytest -q -p no:cacheprovider tests --ignore=tests/v5 -o addopts="" \
  --junitxml=<scratchpad>/audit/P_acceptance/regression.xml
```

**Result: 1,664 tests, 1,657 passed, 5 failed, 0 errors, 2 skipped. Wall time 2,481 s (41 min) on one slot.** The
JUnit is `regression.xml` and the summary `regression_summary.json`. The consolidated run of `a881d18` found 1,648
tests with the same 5 failures. Gate A.1 is **not** met.

| Failing test | Root cause | Kind |
|---|---|---|
| `tests/test_discovery.py::test_help_is_reference_material_without_guidance` (4.0 test) | v5 `social/render.py:56-58` adds `MediaError(code="media_error")`, a `CrucibleError` subclass. The 4.0 `core/discovery.py` `ERROR_MEANINGS` does not explain it, so the agent-facing `world.help` "errors" topic is incomplete. It also fails in isolation: `crucible.load_all_modules()` imports the social renderer. | Real regression in 4.0 behaviour. Fix `ERROR_MEANINGS` (and give every new v5 error code a meaning); do not edit the 4.0 test. |
| `tests/test_v5_rapid_engines.py::test_five_engines_registered_and_four_spec_kinds_covered` | The H0 foundation test pins exactly 5 engine kinds; H1 and H2 added `program_exec`, `experiment_lab`, `planning` and `strategic`. | Stale contract test. Update it through a recorded decision (SP-7.3 needs 8 new kinds anyway). |
| `tests/test_v5_rapid_engines.py::test_engine_modules_share_only_the_contract` | `rapid/engines/program_exec.py` imports `...sandbox.runner`, outside the engine contract (`..`, `.base`, `..primitives`, `..actions`). | Contract decision. Either allow the sandbox as part of the contract (needed for executable-program worlds, 8.3/8.11) or route it through `..actions`. Record a D-entry. |
| `tests/test_v5_rapid_engines.py::test_batch_path_equals_single_worlds_bit_for_bit[cxs_selected_layers]` | `rapid/primitives.py:539`: `x["x"][:, None] + p["sd"] * z[:, :n]`. In the batch path `p["sd"]` is per-world, shape `(6,)`, and needs `[:, None]`. The fast and single-world paths therefore disagree, or crash, for the family `cxs_selected_layers`. | Real bug. It also invalidates SP-8.12's "fast simulator and kernel agree" for this family. |
| `tests/test_v5_rapid_families.py::test_registry_entries_carry_the_contract_fields` | Asserts that the registry uses exactly 5 engines; it now uses 9. | Stale contract test (same decision as above). |

Skips: `test_agents_llm::test_real_qwen_llm_policy_is_deterministic_and_counts_tokens` and
`::test_cam2_real_qwen_replay_logprob_matches_collection` ("needs GPU + local Qwen3-0.6B"). They are skipped because
the run hid the GPU (`--gpu-mb 0`). A.1 should be run with the GPU visible through slots, so that 0 skips can be
required.

Other observations:

* A 4.0 scale test starts 4 rollout workers inside a 1-CPU slot (`[rollout throughput] 1 worker ..., 4 workers ...`).
  The regression suite therefore does not respect its own slot budget. This matters for SP-13.7 and V.8.
* The sealed store hashes were identical after the run.
* The orchestrator committed again during the run (HEAD is now `f394dec`); this run reflects the tree at collection
  time (`dd745b9`).

## 7. Would a learner acquire transferable capabilities?

This area does not simulate anything itself. It decides whether claims about the simulator can be trusted, and as it
stands it would not stop a simulator-specific shortcut from being certified:

* a gate test that records constants passes the official pipeline (P12);
* nothing re-runs evidence;
* the manifest can be pruned (P13);
* gate tests verify capabilities in hand-assembled partial worlds rather than the civilisation a learner would face.

Shortcut risks I saw in passing, which belong to other auditors:

* the legacy reward is the default everywhere;
* in observation-only mode the 4.0 numeric observation still carries revenue and cost
  (`research/envs.py`, `research/bridge.py:79-81`), so the hidden legacy reward (net = revenue - cost) is exactly
  recoverable. Clarification 2 permits this because revenue and cost are world observations. OO.2's leak probes should
  state this explicitly so it is a documented choice, not an accident.

## 8. Verdicts

| Subsystem | Verdict | Effort |
|---|---|---|
| Checker core (`checker.py`, `--verify-all`) | REPAIR | M |
| Evidence recording (`evidence.py`) | REPAIR | S |
| Review workflow (`__main__._review`, `docs/v5/reviews/`) | REDESIGN | M |
| Official run pipeline (`__main__._run`) | REPAIR | S |
| Manifest derivation and content (`manifest.py`, `requirements.py`, `ACCEPTANCE_MANIFEST.json`) | EXTEND | M |
| Gate A suite (`tests/v5/test_gate_A.py`) | IMPLEMENT | M |
| Gate X and QA tooling (placeholder detector, mutation tests, clean-room harness, defect register) | IMPLEMENT | L |
| SP-1.3/1.5/18/20/21 and RP gate tests, final report generator | IMPLEMENT | M |
| Persistent engineering documents | REPAIR | S |
| Progress ledger and durability | REPAIR | S |
| 4.0 definitions and sealed store | RETAIN | S |
| 4.0 regression suite status | REPAIR | S |
| v5 test-suite integrity | EXTEND | L |

Per-requirement verdicts, reasons and work needed: `P_acceptance.json`.


## Skeptic review

Reviewer: skeptic for area P_acceptance (workstream P). Tree: `feat/crucible-5-complete`, now at `f394dec`. Between
`a881d18` and `f394dec` only docs and `acceptance/v5` changed (`git diff --stat a881d18 HEAD`), so the source under
audit is the same. No existing file was modified. Every probe below ran either read-only against the worktree or in
`git clone --shared` copies in the scratchpad. Scripts, logs and clones are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/P_acceptance_skeptic/`.

I tried to refute the one RETAIN and every lenient REPAIR/EXTEND, and I checked the harsh verdicts. I overturned four
subsystem verdicts and two requirement verdicts, and set two `criteria_weak` flags to true. I upheld the rest.

### S1. The official pipeline certifies code that never ran (environment injection): checker core REPAIR -> REDESIGN, run pipeline REPAIR -> REDESIGN

The auditor's P12 used a gate test that writes constants. That leaves room to argue that an honest test plus the
proposed repairs would be enough. I tested that argument with a real, honest, asserting gate test (SP-4.8) and a
real defect. Clone `clone_attack`:

1. **Honest baseline** at `46e07b1` (HEAD plus the SP-4.8 implementation files listed in the manifest).
   `CRUCIBLE_V5_AUTHOR=builder python -m crucible.acceptance.v5 run --requirement SP-4.8`: 1 passed,
   `seller_revenue_cents = 6000`, `hardcoded_social_revenue_bonuses = 0`, clean tree (`sp48_honest.log`).
2. **Real defect, committed** at `c415c28`: `services/market.py:_complete` pays the seller a hidden +50% "launch bonus"
   on every sale of a launched listing, taken from an unrelated participant's account under kind `"sale"`. Official
   run: **1 failed**. The gate test's final `assert revenue == 2 * 3000` (`test_gate_SP_4_8.py:155`) catches it, so
   the checker gives `IMPLEMENTED_UNVERIFIED` (`sp48_mutant.log`). Note, though, that the recorded metric
   `hardcoded_social_revenue_bonuses` stayed **0** with revenue 9000. The manifest criterion that names the property
   is blind to the bonus, and only an assertion outside the manifest catches it (see S4).
3. **Same commit `c415c28`, environment injection.** I put a `sitecustomize.py` in a directory outside the repository
   (`evil/sitecustomize.py`). It replaces the committed, defective `Market._complete` with the pre-defect version at
   interpreter start. I then ran the same official command with `PYTHONPATH=<evil>:src` and
   `review --reviewer ws-P-qa` from the same shell. Results:
   * pytest: 1 passed;
   * evidence: `tree: {commit: c415c28, clean: true, dirty_paths: []}`;
   * the `environment` block records python, platform, CPU quota, GPU and host only; nothing shows the injected path;
   * `checker.check_requirement` returns **`SP-4.8 VERIFIED`, problems `[]`** (`sp48_envinject.log`).

   A commit whose honest run fails is certified as meeting a requirement it violates.
4. **Uncommitted breakage is invisible.** In the same clone I made an uncommitted edit so that
   `Market._complete` raises `RuntimeError`, and re-ran the checker: **still `VERIFIED`**
   (`uncommitted_breakage.log`). Staleness is computed only from committed history (`checker.py:130-139`), and
   `verify_all` (`checker.py:191-210`) never checks that the working tree matches a verified commit. Part 21 requires
   "the final integrated implementation is committed".

Why this changes the verdict. The trust root of the whole design is the producer's own record of its run:
* the run happens in the developer's working tree with the inherited environment (`__main__.py:60-62`,
  `env = dict(os.environ, ...)`, `cwd = crucible_root()`);
* the code that actually executed is not captured: no `sys.path`, no `PYTHONPATH`, no hash of the imported module set,
  and no site-packages hashes (`evidence.py:68-77`);
* "clean tree" is taken as proof of what ran, but it only describes git status (`evidence.py:42-55`).

The auditor's repair list (manifest anchoring, JUnit completeness, transitive staleness of listed files, real reviewer
identities) leaves every one of S1.3 and S1.4 open. Only verifier-side re-execution closes them: a fresh clone, a
scrubbed and recorded environment, and the reviewer's own JUnit and metrics. The auditor lists that as item 2 of 8,
but it is not a repair to this design. It moves the trust root from the producer to the verifier, and the evidence
schema, the review record, the run pipeline and `verify_all` must all change around it. Part 21 asks that every
requirement be "independently verified", and the current architecture cannot provide that. Verdicts:

* **Acceptance checker core: REPAIR -> REDESIGN** (effort M-L);
* **Official run pipeline: REPAIR -> REDESIGN** (effort M);
* **SP-21: REPAIR -> REDESIGN**.

Further checker and evidence defects I confirmed (`probe_zero_and_inf.log`):

* `evidence.record()` accepts `inf` and `-inf`. The check at `evidence.py:86` rejects only NaN (`v != v`), although
  its error message says "must be a finite number". The checker then accepts `inf >= 1e9` and `-inf <= 0.01` as
  passing. There are no domain checks either: a p-value of -1 and an accuracy of 1.5 both pass.
* **50 of 253 manifest entries (20%) are fully satisfied by a record in which every metric is 0 or False**, as
  evaluated by the checker's own `criterion_met`:
  * A.4, C.5, D.2, D.5, E.3, E.5, I.4, J.5, K.6, M.5, O.2, O.3, P.2-P.5, R.1, R.2, R.4, S.6;
  * T.5, T.7, T.8, V.8, W.2, W.5-W.9, all of X.1-X.9, F.n1, I.n2, L.n1, N.n1, O.n1, S.n1;
  * SP-5.4, SP-12.3, SP-16, RP.1 and RP.4.

  None of them has a `>= N` coverage metric. P-value criteria (`<= 0.01`) are met by 0.0. The auditor flagged this
  for X only. It applies to a fifth of the manifest.

### S2. 4.0 benchmark definitions and sealed store: RETAIN -> REPAIR

I reproduced the auditor's positive evidence and found it sound as far as it goes:
* MECH_FAMILIES are identical at `c07098b` and HEAD: 25 families with the same splits (`v4eq_*.json`);
* `MechVecEnv` trajectories are bit-identical (observations and rewards) over 60 steps for one family per split;
* `experience.leakage.legacy_check()` reports 11 of 11 old-vs-new comparisons identical (MechVecEnv, KernelMechEnv,
  SelfPlay, three 4.0 learners, Gym, PettingZoo, WorldEnv; `legacy_check.log`);
* the tool list visible to an actor in the 4.0 `tiny` world is identical (141 tools, same hash);
* the sealed-store hashes match the auditor's (`f0e3d060...`, `faa17770...`, `1d952dd8...`), with the access log
  still at 4 lines.

RETAIN requires the subsystem to be "consistent with the full spec" and robust. It is neither:

1. **The agent-visible 4.0 environment changed.** In a 4.0 `tiny` world the `world.help` "errors" topic now contains
   the line `media_error:` with no meaning (`help_errors.diff`). An LLM agent reading help in a 4.0 benchmark world sees
   different reference text. The cause is that `crucible.load_all_modules()` (`src/crucible/__init__.py:5,27-33`) now
   imports every v5 package into every 4.0 process: the registered kernel modules go from 42 to 89. The auditor logged
   the resulting test failure under the regression suite but still retained the definitions.
2. **Collision with a recorded decision.** D-19 (`DECISION_LOG.md`, after the audit) and Clarification 2 require
   `reward_mode` to default to `observation_only` in every constructor. Today `check_mode(None)` returns
   `legacy_shaped_reward`, and so do `make`, MechVecEnv, KernelMechEnv and CrucibleEnv
   (`probe_default_mode_head.log`). The 4.0 research protocols construct environments without `reward_mode`:
   `research/protocols.py:362`, `:623` and `research/learners/training.py:103`, `:163`. `legacy_check` itself
   compares the 4.0 class with the new *default* constructor (`experience/leakage.py:536`, `:541`). Once the default
   flips as required, the 4.0 training and evaluation protocols will train on zero reward, and the equivalence check
   will fail by construction. The 4.0 definitions cannot be retained as they are. Every 4.0 call site must pin
   `legacy_shaped_reward` before OO.1 lands.
3. **The sealed store is intact but not durable.** The secret salt and the only sealed set live in `/root/.crucible`
   on container storage (`workspace_is_volume = false`). There is no copy in `/workspace/backups`. A recycle destroys
   the 4.0 final held-out evaluation permanently. "Unchanged" was checked; "preserved" was not.
4. **Held-out families are reachable as v5 training environments.** `experience.registry.make(1, <4.0 test-split
   family>, seed=...)` builds a training environment on any test-split family with no split guard and no exposure
   accounting (`registry.py:115`; `test_split_reachable.log`: `abm_network_fx`, `abm_segments`, `scm_feedback`).
   Sealed sets are drawn from exactly these families (`research/sealed.py:189`). The seeds stay secret, but
   family-level held-out status now depends on every v5 learner choosing not to call `make` on them. A.4 needs
   evidence-level accounting of test-family exposure, not only a name check.
5. **Context for A.3 and A.5, not a v5 regression.** 4.0 open-world state is not reproducible run to run. Two runs of
   the same commit give different digests because `cloud.usage` events carry measured sandbox `cpu_ms`
   (`services/cloud.py:212-215`; `v4dig_rerun.log`). I first saw a digest difference between `c07098b` and HEAD
   (`v4dig_*.json`) and withdrew it as evidence for that reason. A.5's `reproduction_mismatches == 0` is therefore
   undefined for open worlds until D-20 (metering outside the measured process) lands, and D-20 will itself change
   4.0 sandbox billing.

### S3. Gate A suite: the IMPLEMENT verdict for A.2 is too harsh (IMPLEMENT -> REPAIR)

`tests/v5/test_gate_A.py` does not exist (confirmed), so the subsystem verdict IMPLEMENT stands. A.2 itself, however,
already has working, wired-in measuring machinery:
* `experience/leakage.py:511` `legacy_check` loads the 4.0 implementation from `c07098b` with `git show` and compares
  it with the current one (11 of 11 identical, 6.2 s);
* it is used by `tests/v5/test_gate_R.py:128` and `tests/v5/test_gate_SP_11_3.py:73`.

The per-node 4.0 baseline already exists too: `docs/v5/evidence/gate_A/regression_g{0,1,2}.xml`, 1,076 node ids,
with the two skips run with CUDA hidden. A.2 is REPAIR:
* call `legacy_check` from a Gate A test;
* compare against explicit `reward_mode="legacy_shaped_reward"` instead of the default;
* add content hashes of the mechanism, worldgen and family split tables and of the agent-visible help and tool text.

### S4. v5 test-suite integrity: EXTEND -> REDESIGN

The auditor did not run the v5 gate suite: the regression command used `--ignore=tests/v5`, and only 3 gate files ran.
I ran it (`--gpu-mb 0`, gate H excluded because it needs the GPU and Qwen; results in S5). Findings:

* **A gate test that has never executed past its integration point.** `test_C_3_consumption_produces_analytics` fails
  immediately with `ProtocolError: feed: the population module does not provide register_activity`
  (`gate_C3_run.log`). `social/recsys/service.py:184` looks up `crucible.population.register_activity`, but the
  population package provides it only as `population.core.register_activity` (`population/web.py:92`). This is a live
  case of the disconnected workstreams that X.8 is about. Once that is fixed, the test will still crash:
  `test_gate_C.py:238` does `int(rep.get("mismatches", 0))`, but `audience.recount()` returns `mismatches` as a dict
  (`c3_latent_bug.log`: TypeError). The default `0` would also report "0 mismatches" if the key were ever missing.
  The record-honest-zero-and-fail pattern the auditor praised works only for a subsystem that is wholly absent. It
  never exercised the integrated path.
* **A criterion metric blind to the defect it names.** SP-4.8's `hardcoded_social_revenue_bonuses` stayed 0 under a
  real +50% bonus (S1.2). Three things let it through:
  * the static scan covers only `src/crucible/social` with a regex;
  * the transaction scan only flags credits from the four named system accounts or unusual kinds;
  * the counterfactual fork has no purchases.

  Only the extra assertion `revenue == 6000` caught it. The manifest criteria alone would have certified it.
* **The partial-world architecture is now formally superseded.** D-14 and D-15 (recorded after the audit) state that
  "the social engine was gated against a stand-in population and cannot run with `pop`". They make `pop` the only
  population and civ configurations the only Tier 2 and Tier 3 worlds. The social, commerce and population gate tests
  build those stand-in worlds (`tests/v5/social_helpers.py`, `tests/v5/_commerce_kit.py`), so they test an
  architecture the project has decided to abandon. This is not "insufficiently comprehensive" (EXTEND): the gates'
  world model has to be rebuilt on the one world, which makes it **REDESIGN** (effort L).

### S5. Gate-suite run (never run before in this audit)

Scope: `tests/v5` without `test_gate_H.py`, which needs the GPU and Qwen. GPU hidden, run through slots in two
1-CPU jobs at `f394dec`. Summary in `gate_suite_summary.json`.

| Group | Result | Not passed |
|---|---|---|
| C, D, E, F, I, J, K (`v5_gates_g1.xml`) | 52 tests: **47 passed, 5 failed** (1,574 s) | C.3 (population integration gap, above); E.3 ("no participant probe is registered"); E.5 (5 reads of creator-assigned `quality` fields, e.g. `domains/firms.py:290`, `social/reference.py:151`); F.3 and F.7 (`tables_v1.npz` missing, checksum mismatches) |
| L, M, R, S, SP-* (`v5_gates_g2.log`) | first 40 of 62: **34 passed, 6 failed** | M.8, M.n1, S.2, S.3, S.5, SP-11.1 |

Coverage limits:
* F.3 and F.7 may be partly a shared-data race. Another tenant rewrote
  `/workspace/data/crucible/gharchive_2024-10-07_96h` (`tables_v2.npz` at 10:05, `acquisition.json` at 10:14) during
  my run. That itself shows that evidence depending on the shared dataset directory can be invalidated by concurrent
  agents, with no versioning or lock. This bears on X.3 and RP.4.
* Group 2 stopped while SP-3.2b, a 7,200-second LM benchmark that runs on the CPU when the GPU is hidden, was running.
  SP-4 to SP-8.12 (20 files) were not re-run. A follow-up job waited 30 minutes for a slot (9 of 9 CPUs held by other
  tenants) and was cancelled.

**The key number: 77 of the 87 requirements these tests measured already meet their manifest criteria**, evaluated
with the checker's `criterion_met` on the honest unofficial records. They are every recorded entry of C, D, I, J, K, L and R (C.3 crashed before
recording), E.1, E.2, E.4, E.n1, F.1, F.2, F.4-F.6, F.8, F.n1, M.3-M.6, S.1, S.4-S.8, SP-11.2-SP-11.6 and SP-2.3. Unmet: E.3, E.5, M.1, M.2, M.7, M.8, S.2, S.3, SP-11.1 and
SP-2.1. The 77 were measured in purpose-built worlds:
* stand-in populations (`admin.join` participants, "the workstream-G adopters until the population of workstream B
  exists", `tests/v5/test_gate_J.py:10-11`);
* scripted creators;
* rule-based adopters.

Once implementation files are listed and a review string is written, the checker would mark them VERIFIED. D-14 has
meanwhile recorded that these subsystems "are not one world". No manifest criterion tells a partial-world pass from an
integrated one. This is the strongest reason the integration requirement the auditor listed as missing must become
gating criteria on every gate. It is also why the test suite needs a redesign, not an extension.

### S6. Criteria flags corrected

* **X.1 `criteria_weak`: false -> true.** `unverified_requirements == 0` is met by a record of zeros
  (`probe_zero_and_inf.log`). With P2 and P13, pruning the manifest also makes it true. It needs
  `requirements_checked == len(derive())`.
* **RP.2 `criteria_weak`: false -> true.** `audit_reports >= 15` and `audit_reports_with_skeptic_review >= 15` count
  files. A heading with no probe behind it would pass. The criterion should require each skeptic section to cite
  probes whose logs exist and hash-match.
* I also confirmed A.4 as zero-satisfiable (`sealed_leak_findings == 0`, `sealed_sets_changed == 0`). Its
  `work_needed` should add salt and set durability and test-family exposure accounting (S2.3, S2.4).

### S7. Verdicts I upheld, with the reason

* **Evidence recording: REPAIR.** I add the `inf` acceptance and the missing environment capture to its defects.
* **Review workflow: REDESIGN.** S1.3 again ran the run and the review from one shell.
* **Manifest derivation: EXTEND.** My probe confirms `ACCEPTANCE_MANIFEST.json` equals `derive()` today: same 253
  ids, text, criteria and dependencies (`probe_manifest_vs_derive.log`). The defect is that nothing enforces this at
  check time, plus the 50 zero-satisfiable entries above. Both are repairs to the content.
* **Persistent documents and progress ledger: REPAIR.** `origin/feat/crucible-5-complete` is at `f394dec` per local
  refs, so the head is pushed. I add the non-durable sealed salt to the durability defects.
* **4.0 regression suite: REPAIR.**
* **Gate X, SP gates and RP tests: IMPLEMENT.** None of the test files exist.
* I re-ran `tests/test_v5_acceptance_checker.py` (13 passed). It passing is not evidence: its
  `test_complete_evidence_is_verified` hand-writes the JUnit and the review.
