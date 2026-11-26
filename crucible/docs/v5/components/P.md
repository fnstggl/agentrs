# Component P: acceptance, independent QA and the persistent engineering state

Owner: workstream P (independent quality assurance). Code: `src/crucible/acceptance/v5/`. Gate tests:
`tests/v5/test_gate_{A,X,SP_1_3,SP_1_5,SP_18,SP_20,SP_21,RP}.py`. Unit tests: `tests/test_v5_acceptance_checker.py`
(with `tests/acceptance_kit.py`). Decisions: D-22 (requirements added from the audit), D-23 (stricter criteria),
D-24 (trust model), D-25 (gate files of P's added requirements).

## What the checker trusts, and what it does not

`python -m crucible.acceptance.v5 --verify-all` never reads a status string as a verdict. It verifies:

| Layer | Module | What is checked |
|---|---|---|
| manifest anchoring | `manifest.consistency`, `ledger` | manifest == `derive()` (spec text, SHA-256-anchored MASTER_SPEC, `requirements.py`); == the append-only hash-chained ledger `docs/v5/ACCEPTANCE_LEDGER.jsonl`; ledger append-only across every committed version and merge, prefix of the pushed copy; no requirement removed; criteria changes decided and provably at least as strict; test/implementation list removals, wording and tolerance changes decided |
| provenance | `official`, `evidence` | evidence only from `run`: a fresh clone of a committed tree (marker in the clone's git dir), collect-only then run; the run record (commit, clone clean before and after, collected node ids, exit codes, identity) and the JUnit (suite name = run id) are committed |
| completeness | `checker.junit_problems` | every collected node passed; every test function present at the commit (AST) was collected and passed; the producing test is one of the requirement's own tests |
| genuineness | `testscan`, `checker.integrity_problems`, `checker.module_problems` | the producing test asserts on measured values and records non-constant metrics; no skip, early return, swallowed exception, mock or unreverted patch; the imported-module blob ids equal the commit's; no crucible module came from outside the clone; a listed implementation module was imported; nothing imported changed since (transitive staleness) |
| criteria | `checker.criterion_met` | every criterion on the evidence's own finite metrics |
| independent review | `official.review`, `checker.review_problems` | a different agent name and session re-executed the tests in another fresh clone of the evidence commit; the checker compares the reproduced metrics with the evidence within the declared tolerance; the review is bound to requirement text, criteria version, evidence hash and commit, and is committed |
| dependencies | `checker.verify_all` | greatest fixed point; X depends on everything except X, SP-18, SP-20 |
| release | `--verify-all --reproduce` | re-executes every VERIFIED requirement in a fresh clone |

Project status: CRUCIBLE_5_COMPLETE (no project-level problem, all VERIFIED), INCOMPLETE_BLOCKED (no project-level
problem and every unverified requirement BLOCKED by an open `BLK-<n>` of BLOCKERS.md), otherwise IN_PROGRESS. Exit
codes 0, 3, 1.

Residual trust (stated, not hidden): identities are declared agent names plus hashed session ids, not signatures. A
forger with write access can fabricate every file consistently; the release-time `--reproduce` re-executes the tests
and exposes it. The static genuineness checks catch constants and missing assertions, not every way a test could
measure the wrong thing; mutation testing of implementation files is not implemented.

## Workflow

See REPRODUCE.md "Gate evidence" and "Clean-checkout reproduction". After every merge on the integration branch:
`build-manifest` (with `--decision D-<n>` for any change) and `merge-workstreams`, then commit the manifest and the
ledger. Only the integration branch appends to the ledger: two branches that both append cannot be merged without
rewriting it, which the checker rejects.

## QA tooling (`qa.py`)

X.3 `clean_checkout`, X.5 `documents`, X.6 `defects` (docs/v5/DEFECTS.md, one ```json block per defect), X.7
`scan_acceptance_tests`, X.8 `integration_audit`, X.9 `claims_audit`; RP.1 `cam_scan`, RP.2 `audit_reports`, RP.3/RP.5
`remote_state`, RP.4 `artifacts` (docs/v5/ARTIFACTS.json), RP.6 `secrets_scan`, `remote_forward_only`. Gate A:
`baseline.py` with the standalone `_v4probe.py`. Final report: `report.py` (Part 18 table and Part 20 separation from
checker verdicts only, every number tagged). Progress ledger: `progress.py` (docs/v5/PROGRESS.json).

## Limitations

* Gate tests of requirements P added for other workstreams (most of D-22) do not exist; their owners write them.
* OO.13 and OO.15 are P-owned but their canonical gate file `tests/v5/test_gate_OO.py` belongs to workstream L; the
  measurements are not written yet (an import/AST scan for learner reward functions, and the evidence/review identities
  of OO.1-OO.8).
* X.8's integration audit attributes events to packages by event-type prefix; it needs `crucible.civ.generate`
  (workstream CIV), which does not exist yet, so X.8 currently measures every world workstream as disconnected.
* A.5 compares measured values of the 4.0 acceptance records exactly except wall-clock fields; records that are
  nondeterministic by design will need declared tolerances through change control.
