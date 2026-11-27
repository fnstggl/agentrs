# Workstream F — fixes of the G-rt verification (D1–D25)

Source: `docs/verification/G_rt.md` (lens `rt`). Every defect below has a regression test in `tests/runtime/` and the
verifier's probe `docs/verification/repro/rt/<probe>.py` was run before and after the fix
(`G_RT_SCRATCH=<scratch>/F/g_rt .venv/bin/python cam_live/docs/verification/repro/rt/<probe>.py`, from
`/workspace/camlive`). Commits: `bf902ac` (verifier), `a30c993` (dashboard), `eee6989` (tasks), `5250714` (runtime,
CLI, journal, preflight, control), `2c89aa0` (regression tests, redaction speed), `539a726` (compute budget of
learning calls) and the documentation commit.

## Probes before / after

| Probe | Before (reproduced, = G) | After | Checks still reported BROKEN, and why |
|---|---|---|---|
| p01_loop_steps | 6 of 36 | **0 of 36** | — |
| p02_crash | 19 of 477 | **0 of 477** | — |
| p02_crash `--no-dedup` | 27 of 477 (G) | 4 of 477 | `after_wal_begin:2`, `before_execute:2`: money / external-action counters are *above* the world's (7.2 vs 6.9, 7 vs 6). Without de-duplication the runtime cannot know whether an interrupted action ran; it now books it at its estimate (D18), so it can over-count by one action, never under-count. |
| p03_control | 1 of 15 | **0 of 15** | — |
| p04_limits | 3 of 11 | **0 of 11** | — |
| p05_tasks | 19 of 34 | 1 of 33 | `1a` hardcodes 17 REQUIRED items; there are 16 because `success.evidence_source` of `discord_100.yaml` is now the Discord membership source (requested). Check 4 no longer runs: no disguise is accepted any more. |
| p06_verifier | 10 of 38 | **0 of 38** | — |
| p07_redaction | 43 of 62 | **0 of 62** | — |
| p08_dashboard (Chromium) | 3 of 18 | 2 of 18 | **12 and 13 hold** (nothing injected runs; no approval). `6`: the heuristic `"src=" not in page` flags `<script src="/app.js">`, which a CSP without inline script requires (same origin; no external resource). `8`: approving with only the viewing token is refused by design (operator credential + CSRF now required). |
| p09_preflight | 2 of 28 | 1 of 28 | `1b` hardcodes 17 REQUIRED items (16, see p05); every unset item is still named. |
| p10_cli | 2 of 36 | probe aborts at 4a | `cam_live.validation` (another workstream) now exists and its staging run exceeds the probe's 240 s timeout (`TimeoutExpired`), not an F defect. A copy of the probe without step 4a (`<scratch>/F/p10copy/`) reports **0 of 35** (5c, 9b hold; 2a now runs the real `cam_live.agent:build_agent`). |
| p11_separated (root) | 1 of 39 | 1 of 39 | `14` expects a supervisor started by `cam-live resume` on an *emergency-stopped* run; resume now refuses a stopped run (13 holds, nothing lingers), so none is started. The D20 fix itself (pid written, records passed on) is pinned by `test_D20_*`. |
| p12_numbers | 0 of 5 | **0 of 5** (against the updated `docs/status/F.json`) | re-measured after the fixes (`docs/F/measure_fix.py` → `docs/F/loop_overhead_fix.json`). On this shared machine the same 300-step run measured 33–101 ms per decision without fsync and 76–212 ms with fsync at load averages 25–49 (G measured 37.8 / 79.5 before the fixes at load 20); log bytes per decision 41586 → 42689 (+2.7 %: `beliefs_updated` events). The extra cost per decision is sealing the journal and the resume state and redacting `state.json`. |
| p13_journal | 2 of 8 | **0 of 8** | — |
| p14_multi | 1 of 11 | **0 of 11** | — |
| p15_malformed_result | 2 of 3 | **0 of 3** | — |

Change request F-8 asks G to update the four checks that encode the old design (p05 1a, p09 1b, p08 6/8, p11 14).

## Per defect

**D1 (critical) dashboard** — `runtime/dashboard.py`. The client builds every element with `createElement` +
`textContent` (no `innerHTML` with data; numbers are formatted only when they are finite numbers, anything else is
text). JavaScript and CSS are served from `/app.js`, `/app.css`; every response carries
`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; ... frame-ancestors 'none'`
plus nosniff / no-referrer / DENY / no-store. Operator actions (approve, reject, pause, resume, stop, emergency stop,
reset of the emergency stop, export, assign objective) need the viewing token **and** an operator credential
(`dashboard.operator`, 0600, distinct from `dashboard.token`, header `X-CAM-Operator` only) **and** a double-submit CSRF
token (`cam_live_csrf` cookie = `X-CSRF-Token` header) with same-origin Origin / Sec-Fetch-Site; the uid owning the
client's loopback socket is read from `/proc/net/tcp{,6}` and a service uid (camlearn, …) is refused even with every
credential. Experiments show only runtime-measured outcomes. `cam-live dashboard` loads the operator credential
(`--view-only` disables actions). Tests: `test_rt_fix_dashboard.py` (10, incl. Chromium and a camlearn process).

**D2 wait lost by kill -9** — `loop.py`: a wait in progress is carried into the journal of the next commit (written
under the new generation before the commit point; the journal reset is one atomic rewrite holding it).
`test_D2_kill_during_a_wait_loses_no_decision`; p02 `in_wait_sleep:*`, `after_outcome_log:1`.

**D3 journalled result never logged** — on replay the runtime reads the logs since the committed position backwards
(`logs.iter_since`) and writes the `action_result` / observation records a crash prevented, never a second one.
`test_D3_a_journalled_result_reaches_events_after_restart`; p02 `after_wal_end:2`, `torn_event:2`.

**D4 approved batch without limit checks** — `_process_approvals` re-checks `_exhausted()`, the money estimate (the
larger of the learner's and the tool's declared estimate) and the external-action count before *each* approved
execution; refusals are reported as `over_limit`. `test_D4_actions_approved_together_are_rechecked_against_the_limits`;
p04 check 7.

**D5 NaN / inf / negative / bool limits** — rejected at load in every mode (`tasks.py`, fork); a TaskSpec built in code
with such a limit is refused in live mode and, outside live mode, ends the run at once as exhausted (fail closed,
`limits_invalid` in `run_started`; state.json stays strict JSON). `test_D5_*`, `test_rt_fix_tasks.py::test_degenerate_*`;
p05 check 2, p04 check 6.

**D6 task integrity** — `tasks.py` (fork): closed schema with dotted paths for unknown keys, bounded free text, YAML
anchors / aliases / merge keys and duplicate keys refused, strategy scan of every free text and free-form key,
objective text with strategy-like content refused unless confirmed by `objective_confirmation: sha256:<hash>`,
verifier texts scanned. `learner_view(task)` is what `agent.set_objective`, `run_started`, `objective_assigned`,
`state.json` and `task.json` carry: never the definition, exclusions, evidence source, measurement protocol,
credential names or test accounts. `test_rt_fix_tasks.py` (22 tests, parametrised), `test_D6_*`; p05 all 19 disguises.
Limitation: the runtime process itself still reads the full task file (the authorization service pins the task by its
content hash), and in separated mode it shares the learner's uid.

**D7 redaction** — `logs.py`: `NAME=value` / `"name": "value"` / `?name=` with any name containing a secret word,
credentials in URLs, secret keys by word (`client_secret`, `DISCORD_TOKEN`, `accessToken`; numbers stay), bytes never
written, member identifiers as keyed hashes in text (17–20 digit ids, `name#1234`, mentions), in lists and person
objects, metric names and entities; ids the operator configured stay readable. `state.json` is redacted (errors,
control results, approvals, pending-query summaries); the exact resume state, the raw decision of each pending
approval and the write-ahead journal are sealed with AES-256-GCM (`runtime/private.py`, key in `<run_dir>/learner/`
0600). One combined pre-check keeps the redactor's cost per record low. `test_D7_*`; p07 all 62.

**D8 learner-defined prediction queries** — for an action that happened (or a wait), each registered prediction whose
query no environment query covers is queued as a query of the learner's (`PendingItem.origin = "learner"`, asked from
the moment the action started, bounded per decision and in total) and measured like any other.
`test_D8_a_prediction_with_a_learner_defined_query_is_measured`; p01 F1.

**D9 verifier counting** — `verifier.py` (fork): members merged per normalised id, any bot / flag record wins, flags
case-insensitive, link keys followed transitively with every key of an excluded account tainted, unknown join time and
(with a minimum age) unknown account age are uncertain. `test_rt_fix_verifier.py`; p06 A13–A20.

**D10 learner claims recorded as outcomes** — registered predictions are cleared of resolved_* fields
(`prediction_fields_cleared` event); an `ExperimentRecord`'s `actual` is stored as `learner_reported_actual` with
`verified=False`; the journal states `supported` as the learner's assessment (`[HYPOTHESIS]`); the dashboard shows only
`experiment_outcome` records; a verifier observation counts as verified only when the verifier's signature on it checks
(`check_measurement`; `SignedVerifier` now signs measurements and `RemoteVerifier` / the service expose the check —
M-int-3). `test_D10_*`, `test_signed_verifier_measurements_check_locally_and_over_rpc`; p01 E2–E5.

**D11 journal injection** — `journal.py`: every learner- or environment-derived string is quoted with `untrusted()`
(one line, control characters removed, `[FACT]` / `[HYPOTHESIS]` neutralised). `test_D11_*`; p13 B1.

**D12 preflight** — `preflight.py`: the diagnostic policy is recognised by type (subclasses too) or by lacking the
service methods; every component inside the environment's capabilities that names a staging / diagnostic / loopback
service blocks; every permitted interface needs an available tool; a Discord task needs the Discord tools and a
read-only `discord.get_identity` / `discord.get_permissions` check executed through the environment under the
authorization service (bot in the guild, GUILD_MEMBERS — plus MESSAGE_CONTENT when content permissions are granted —,
the operator's scope held, executor answering from the live platform), the credential in the vault and the verifier's
own copy (`<run_dir>/verifier/secrets/<name>`); without the services each item is named as unconfirmed.
`cmd_supervise` passes the run directory. `test_D12_*`; p09 2o, 2p.

**D13 authorization_revoked** — when listed, each loop iteration reads the authorization service's task state and stops
the run when access for the task is revoked. `test_D13_*`; p04 check 8.

**D14 torn journal line** — the journal's torn tail is truncated before the first append. `test_D14_*`; p02
`torn_wal_begin:2+in_learn:1`, `torn_wal_end:2+before_wal_end:2`.

**D15 open / doubled records** — a decision is journalled before its decision record (a crash in between closes it as
`interrupted` at restart, writing a missing decision record); a matured outcome is journalled before its records, so a
restart neither measures it again nor loses it (re-delivered to the learner). `test_D15_*`; p02 `after_decision_log:2`,
`after_outcome_log:1`.

**D16 step 5 record** — `beliefs_updated` event with the learner's report each cycle. `test_D16_*`; p01 "B step 5".

**D17 exception after execution** — the result is normalised (non-numeric / negative / non-finite costs, wrong types,
malformed queries) before anything else, the journal entry comes first, and `after_cycle` hooks are guarded.
`test_D17_*`; p15, p14 check 8.

**D18 interrupted action without de-duplication** — booked at its estimated cost (and as an external action), stated in
the result. `test_D18_*`; p02 `after_execute:2 --no-dedup` (see the table for the resulting over-count elsewhere).

**D19 verify-completion options** — `cam-live verify-completion …` hands every argument to the checker (`--help`
shows the checker's help); `evaluate-transfer` passes unknown options through. Hooks `cam_live.agent:build_agent`,
`cam_live.validation:run_staging`, `cam_live.eval.transfer:main` say "not available yet" only when the module is not
installed (`ComponentMissing`); an existing module that fails to import or lacks the attribute is reported as such.
`test_D19_*`, `test_hooks_report_not_available_only_for_a_missing_module`; p10 9b.

**D20 separated resume** — `run.json` records the authorization record and verifier source; `cam-live resume` passes
them to `_supervise`, writes `supervisor.pid`, refuses while a supervisor is alive, and refuses a stopped run (no
lingering supervisor). The pending `resume` command is signed. `test_D20_*`; p11 13 (14: see table).

**D21 inspect-harness** — missing integrations named tool by tool (`preflight.missing_integrations`): interfaces without
tools, the Discord integration and the read tools of its check. `test_D21_*`; p10 5c.

**D22 verifier log** — whole chain verified at restart; member and invite identifiers only as keyed hashes in the log
(fork). `test_rt_fix_verifier.py`; p06 D6, E2.

**D23 journal provenance** — decision records are counted by their own provenance in the journal.
`test_D23_*`; p13 C1.

**D24 control commands** — `OperatorControl` signs every command with the run's operator Ed25519 key
(`<run_dir>/operator/control.key`, created by the operator's uid; `cam-live run` creates it before any service
starts); the runtime trusts `control.pub` only when root owns it (or its own uid in single-uid mode), applies
`resume / approve / reject / assign_objective / export_checkpoint` only when signed, a signed id at most once, and logs
unsigned commands as `unauthenticated` (brakes and wake-ups still apply). `test_D24_*`; p03 E1. Limitation: in
single-uid in-process mode the learner's code could read the same key.

**D25 task comments** — both comments of `discord_100.yaml` now say what the code does (fork).
`test_rt_fix_tasks.py::test_discord_task_comments_state_what_the_code_does`.

**Discord evidence default** — `tasks.evidence_source(task)`: explicit `success.evidence_source`, else
`cam_live.harness.discord_membership:membership_source` for a task with Discord resources; used by the verifier service
and in-process runs; `discord_100.yaml` names it. `test_rt_fix_tasks.py::test_discord_tasks_default_to_*`.

## Regression tests

`tests/runtime/test_rt_fix_crash.py` (7), `test_rt_fix_runtime.py` (40 incl. parametrised), `test_rt_fix_tasks.py`
(62 incl. parametrised), `test_rt_fix_verifier.py` (13), `test_rt_fix_dashboard.py` (10, 2 of them `browser` /
`privileged`). Run against the code G examined (commit `3700f9b`, tree exported to `<scratch>/F/old`, default marks):
crash 7 of 7 fail; runtime 39 of 40 fail (the one that passes, `test_D7_text_shapes_are_redacted[url]`, pins a shape the
old e-mail pattern caught by accident); tasks 53 of 62 fail (the 9 that pass are controls of behaviour that already
held, e.g. the shipped files load); dashboard 8 of 8 fail (2 more are `browser` / `privileged`); the verifier file does
not import on the old tree (it uses the new `norm_id`) — its author reports 13 of 13 failing with only `verifier.py`
reverted, which I did not re-run.
