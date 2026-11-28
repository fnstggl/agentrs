# Change requests of HX (harness requirements no status file claimed; owner of gap fixes in `src/cam_live/harness/**`)

Context: `docs/HX.md`. HX's code changes are in HX's paths; nothing below blocks HX's claims. Each request says what,
why and the exact change.

## HX-1 (status file M-int, owner M): adopt or acknowledge the OP-07 claim

**What.** OP-07 (operator credential entry, `src/cam_live/harness/credentials.py`) has workstream `M` in the
manifest, so only `docs/status/M-int.json` (or another `M-*` file) can make an *owner* claim. HX claimed it in
`docs/status/EA.json` (the vault and credential isolation are EA's) with `state: implemented`; the checker counts it
as a claim of another workstream (`claim_source: other workstream`).

**Why.** HX ran the three named acceptance tests (`tests/integration/test_credentials_membership.py`, M's
directory, not changed) and added a privileged test in its own directory
(`tests/harness/auth/test_hx_credentials_privileged.py`): the CLI stores into a launched run directory, the value is
in exactly the vault file and the verifier copy, both unreadable for the learner and sandbox uids, never printed or
taken from argv, known to the running authorization service by name only, refused as a leak and redacted, revocable.

**Change (if M agrees).** Add to `docs/status/M-int.json` `requirements`:
```json
"OP-07": {"state": "implemented", "files": ["src/cam_live/harness/credentials.py"],
          "integration_points": ["cam_live.harness.credentials.main", "cam_live.harness.credentials.store"],
          "tests": ["tests/integration/test_credentials_membership.py::test_store_puts_the_value_only_in_vault_and_verifier_copy",
                    "tests/integration/test_credentials_membership.py::test_bad_values_and_loose_files_are_refused",
                    "tests/integration/test_credentials_membership.py::test_cli_never_prints_the_value_and_revoke_removes_the_copy",
                    "tests/harness/auth/test_hx_credentials_privileged.py::test_operator_entered_credential_is_isolated_from_learner_sandbox_training_data_and_logs"],
          "evidence": ["docs/evidence/test_results/HX_fix.xml"],
          "limitations": "see docs/status/EA.json OP-07"}
```

## HX-2 (docs, owner EA): document the hard platform constraints

**What.** `docs/EA/SECURITY.md` / `docs/EA/POLICY_SCHEMA.md` describe every refusal as policy data. Since commit
`a1cdc0e` two refusals are fixed in code and cannot be changed by the policy file:
`X-HUMAN-VERIFICATION` and `X-USER-ACCOUNT` (`src/cam_live/harness/auth/platform_constraints.py`, called from
`Evaluator.node` for `authorize` and `check_primitive`).

**Why.** Operators editing `configs/authorization.yaml` should know that no `owned` / `preauthorized` /
`interactive` entry, record or approval can permit CAPTCHA solving or Discord user-account automation.

**Change.** A section "Hard platform constraints (not policy data)" in `docs/EA/SECURITY.md` pointing at the module
docstring of `platform_constraints.py` (it lists what is refused and why).

## HX-3 (agent / credit, owners INT3A / INT3B / D): temporal-abstraction traces of programs (CU-19, for review)

**What.** `credit.smdp.attach_trace(rec, step_results)` assigns a program's primitive operations to its steps only
when it is given the steps' own results; the environment's program result (`ComputerEnvironment._dispatch`) merges
the step results into one `ActionResult` and nothing calls `attach_trace` (the checker lists `cam_live.credit.smdp`
as unreachable from the runtime entry points).

**Why.** Spec 12A.9 asks for start state, underlying operations, elapsed time, resources, termination condition and
result per extended action. Skills (`skill.run`) record this at the procedure level; environment programs and the
intervention level do not yet reach the credit module.

**Change (suggested, not made by HX: `env.py` is verified EB behaviour and the agent / credit are not HX's).** Either
the environment records per-step `{action_id, operation, level, started, ended, status, failure_class, n_ops}` for
programs in its result, or the agent calls `credit.smdp.attach_trace` on each executed decision with the step
results it already has. HX left CU-19 `partial` in `docs/status/ES.json`.

## HX-4 (tests, owners EB / EC): an asyncio loop left behind breaks two Playwright-sync tests when run after compute

**What.** In one pytest process, `tests/harness/compute` (browser-marked tests) followed by
`tests/harness/env/test_adapter_contract.py`, `test_browser.py` and then
`test_browser_sessions.py::test_cdp_attach_to_a_separately_launched_chromium` /
`test_browser_uid.py::test_chromium_runs_under_the_browser_uid_with_a_private_profile` fails with Playwright's
"It looks like you are using Playwright Sync API inside the asyncio loop". Each test passes alone and when `env` runs
before `compute`.

**Why it is not HX's.** Reproduced on the code before HX (40734d6, scratch copy: 3 failed, 90 passed with the same
selection) and on HX's code (1 failed, 92 passed); HX changed no browser, compute or env-core code.

**Change (suggested).** The two tests run `sync_playwright()` in a fresh thread (or the test that leaves the event
loop set closes it / resets `asyncio.set_event_loop(None)` in a fixture finalizer). HX's final run orders the
directories env, auth, skills, integration, compute.

## Status 2026-11-28 (round 2)

* HX-4 done by HX (`tests/harness/compute/conftest.py` fixture module-scoped, commit 863bdae).
* HX-2 still open for EA: the hard constraints (now three ids, `X-PLATFORM-ADAPTER` added) are documented in the
  module docstring of `platform_constraints.py` and in `docs/HX.md`; `docs/EA/SECURITY.md` is not HX's file.
* HX-1, HX-3 unchanged.
* New, HX-5 (runtime / operator, owners F / M): a staging run through `cam-live run` must register its staging
  sites (`opctl local-service add`); nothing in `runtime/` does (G2-harness §2). Not changed by HX.
