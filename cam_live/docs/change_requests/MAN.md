# Change requests from MAN (manifest / completion checker), 2026-11-27

The checker's rules are in `docs/COMPLETION.md` (generated) and in the docstring of
`src/cam_live/verify_completion.py`. None of the items below was applied by MAN; each names its owner.

## MAN-1 — claims for the 40 new requirement ids (owner: master M, then each owner)

The manifest now has 198 requirements (158 before). The new ids carry `declared` facts (files, tests, integration
points that exist; see `tools/gen_manifest.py::DECLARED`) but no state: a state comes only from a status file. Until
an owner claims them they are `partial (unclaimed)` or `missing`. Proposed owners (manifest `workstream` field):

| Owner | New ids |
|---|---|
| B | PI-11 |
| D | CL-10 |
| EB | IH-13, CU-35, CU-36 |
| EA | IH-14, IH-15, IH-16, IH-17, CU-37, OP-09 |
| EC | CU-38 |
| ES | OP-04 |
| F | LX-17, LX-19, LX-20, CM-13 |
| S | SV-13, SV-14, SV-15, OP-10 (and CU-33, which has ES's measurement `docs/evidence/ES/skills_vs_primitives.json`) |
| M | OB-01..OB-04, OP-01..OP-03, OP-05..OP-07, OP-11, LX-18, DL-02, DL-04, DL-19, DL-20, IN-05, IN-06 |
| G | IN-08 (and IN-02, IN-03) |

Diff for a status file (example, `docs/status/EA.json`, inside `"requirements"`):

```json
"IH-16": {"state": "implemented", "files": ["src/cam_live/harness/auth/audit.py"],
          "integration_points": ["cam_live.harness.auth.audit.AuditLog"],
          "tests": ["tests/harness/auth/test_ea_audit.py::test_tampering_is_detected", "..."],
          "commands": ["..."], "evidence": ["docs/evidence/test_results/<run>.xml"], "limitations": "..."}
```

Workstreams A–ES are finished; the master decides who claims for them. A status file whose `workstream` is
`<owner>-<suffix>` (e.g. `M-int`, `M-MAN`) counts as the owner's claim.

## MAN-2 — verdict records carry their time and commit (owner: G)

Why: a verdict counts only if it is newer than the last change to the requirement's files. Without a recorded time
the checker infers it from the git history of `verdicts_<lens>.json` (the first commit since which the entry is
unchanged), which is correct but slow and breaks if a lens file is rewritten without content change. Add to every
entry of `docs/verification/verdicts_<lens>.json`:

```json
"RC-02": {"status": "verified", "by": "G-cl2", "notes": "...", "evidence": ["..."],
          "verified_at": "2026-11-28T09:15:00+00:00", "commit": "<sha the verifier inspected>",
          "supersedes": ["G-cl"]}
```

`commit` makes the verdict fresh exactly while the requirement's files are unchanged since that commit;
`supersedes` lets a re-verification lens replace an older lens's verdict for the same id (a newer verdict with the
same `by` replaces the older one automatically). Proposed addition to `docs/verification/README.md` (bullet list):

```
* Each verdict entry also records `verified_at` (ISO time) and `commit` (the commit inspected); a re-verification
  under a new lens lists the lenses it replaces in `supersedes`. The completion checker reads the lens files
  directly and ignores verdicts older than the last change to a requirement's files.
```

## MAN-3 — evaluation status file (owner: S)

`docs/status/S.json` names the directory `src/cam_live/eval/` as files (accepted: every `.py` under it is scanned)
and `tests/eval/` as the acceptance test. A directory is not an acceptance test: name `file::test` node ids. Evidence
JSON should state `"executed": true`, `"requirements": [ids]` and the code commit it ran on (e.g. `"commit"` or
`"git_head"`), as `docs/evidence/validation/*.json` already does. `src/cam_live/eval/openended.py`,
`oe_runner.py` and `proposal_probe.py` are not import-reachable from any entry point yet (finding).

## MAN-4 — staging validation evidence and integration claims (owner: M / INT2)

Every `docs/evidence/validation/staging_*.json` records `"ok": false`, so LX-11 ("staging validation passes") and
IN-07 have no passing executed evidence. The integration requirements AE-02, LX-04, LX-05, LX-06 name no
implementation at all (`missing`).

## MAN-5 — preflight evidence for status level 3 (owner: F or M)

Level 3 needs a recorded preflight: write the output of `cam_live.runtime.preflight.preflight(task)` for
`tasks/discord_100.yaml` to `docs/evidence/preflight_<date>.json` (its `live.ready` / `live.missing` format is read
as is). None is recorded.

## MAN-6 — record the final test run after the commit it tests (all workstreams)

A result counts only if the run started after the last commit (or uncommitted edit) of the requirement's files and
of the test file. Commit the code first, then run the tests with `--junitxml=docs/evidence/test_results/<id>.xml`,
then commit the XML. Most results recorded during the fix pass (05:39–07:22 UTC) predate later commits to the same
files and are stale.

## MAN-7 — modules no entry point reaches (owners: F, D)

`cam_live.runtime.multi` (AE-06), `cam_live.credit.smdp` (CU-20's ledger), `cam_live.credit.estimators` and
`cam_live.credit.longdelay` (RC-04's reference estimators) are imported only by tests: no runtime process, operator
CLI, evaluation or diagnostic program loads them. AE-06 names integration points in `runtime.multi` only, so the
checker reports it as not reachable; CU-20 (through `credit.replay` / `credit.providers`) and RC-04 (through
`diagnostics.tabular`, a diagnostic program) keep one reachable integration point each, so the disconnected modules
appear as findings and notes only.
