# G-rt — independent verification of workstream F (persistent live runtime)

Verifier lens `rt`, 2026-11-27. Scope: `src/cam_live/runtime/`, `tasks/`, `tests/runtime/`, `docs/status/F.json`,
`docs/F/API.md`; requirement ids AE-06, CU-24, LX-01/02/03/07/08/09/10/12/13/14/15/16, CM-01 … CM-12.
Verdicts: `verdicts_rt.json` — **6 verified, 18 partial, 2 failed** (LX-10, LX-14), 0 unverified.
No source file and no existing test was changed. No external service was contacted.

## Result in short

The loop does perform the 17 steps in order, never executed an action twice in 60 kill -9 scenarios, honours
pause / resume / stop / emergency stop within about one second while asleep, keeps predictions strictly before
outcomes in an append-only log with one global sequence, and under separated uids the learner uid can neither read
the verifier key, forge a signed outcome, approve, nor lift an emergency stop. Around that core I reproduced
1 critical, 11 major and 13 minor defects. The ones that matter most:

* the dashboard runs markup from a learner-written record in the operator's browser, and that script approved a
  pending action as the operator (D1);
* a `wait` in progress — the state the runtime is in most of the time — is lost by a kill -9 (D2), and a result can
  be journalled but never reach `events.jsonl` (D3);
* limits are not enforced for approved actions executed in a batch, nor when a limit is NaN / infinite (D4, D5);
* the mechanical no-strategy check accepts 15 of 19 disguises (D6), and redaction misses common secret and
  member-identifier shapes (D7).

Everything was driven by verification stand-ins (`repro/rt/standin.py`) and by F's test agent, because
`cam_live.agent` does not exist. See "Only confirmable with the real agent" at the end.

## What I ran

All commands from `/workspace/camlive`. Probes print `[HOLDS ]` / `[BROKEN]` per check and exit 1 when a property
under attack is broken; details are written to `<scratch>/<probe>.result.json`.

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest cam_live/tests/runtime -q --rootdir=cam_live` | 78 passed, 1 skipped, 67.7 s |
| `.venv/bin/python -m pytest cam_live/tests/runtime/test_rt_separated.py -q --rootdir=cam_live -m privileged` | 1 passed, 12.5 s |
| `repro/rt/p01_loop_steps.py` — step order, observability, predictions vs outcomes | 6 broken of 36 |
| `repro/rt/p02_crash.py` — kill -9 at 23 points, 4 double crashes, 3 file-size-limit runs (de-duplicating environment) | 19 broken of 477 |
| `repro/rt/p02_crash.py --no-dedup` — the same on an environment that does not de-duplicate | 27 broken of 477 |
| `repro/rt/p03_control.py` — operator control in real time | 1 broken of 15 |
| `repro/rt/p04_limits.py` — limits and stop conditions | 3 broken of 11 |
| `repro/rt/p05_tasks.py` — task loader | 19 broken of 34 |
| `repro/rt/p06_verifier.py` — verifier | 10 broken of 38 |
| `repro/rt/p07_redaction.py` — every file of the run directory searched for 29 planted values | 43 broken of 62 |
| `repro/rt/p08_dashboard.py` — API + the page in headless Chromium on 127.0.0.1 | 3 broken of 18 |
| `repro/rt/p09_preflight.py` — preflight and doctor | 2 broken of 28 |
| `repro/rt/p10_cli.py` — every `cam-live` command | 2 broken of 36 |
| `repro/rt/p11_separated.py` — separated uids attacked from the learner and sandbox uids (root) | 1 broken of 39 |
| `repro/rt/p12_numbers.py` — numbers of the status file | 0 broken of 5 |
| `repro/rt/p13_journal.py` — journal | 2 broken of 8 |
| `repro/rt/p14_multi.py` — multi-actor | 1 broken of 11 |
| `repro/rt/p15_malformed_result.py` — executor result with a non-numeric cost | 2 broken of 3 |

Run a probe as `.venv/bin/python cam_live/docs/verification/repro/rt/<probe>.py`. Run directories go to
`/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/g_rt/` (override with `G_RT_SCRATCH`); CLI
probes set `CAM_LIVE_HOME` and `CAM_LIVE_GPU_LOCK` there. Two exceptions, both removed when the probe ends:
`p11_separated.py` needs a run directory the service uids can reach and the scratchpad sits under a 0700 root
directory, so it uses `repro/rt/.sep_<pid>/`; `p12_numbers.py` measures fsync on the repository's file system in
`repro/rt/.num_<pid>/`. F's own privileged test writes to `/workspace/camlive/runs/f_privileged_<pid>` and removes it.

Stand-ins (mine, not F's): `StandInAgent` (scripted `AgentAPI`), `SlowEnv` (real-clock `EnvironmentAdapter` with a
slow external action and a delayed outcome), `PendingVerifier`, a world host served over the project RPC so that the
world survives the killed runtime, and LabWorld from `cam_live.diagnostics.world`.

## Numbers of `docs/status/F.json` reproduced

| Claim | Status file | Measured |
|---|---|---|
| default suite | 78 passed, 1 skipped, 46.1 s | 78 passed, 1 skipped, 67.7 s (shared machine) |
| privileged suite | 1 passed, 8.8 s | 1 passed, 12.5 s |
| ms per decision, fsync on / off | 72.5 / 24.6 | 79.5 / 37.8 at load average 20 (215 / 55 at load average 40) |
| log bytes per decision | 41586 | 41618 |
| REQUIRED items in the Discord task | 17 | 17 |
| runtime source lines | 5387 | 5387 |
| kill -9 resume scenarios | 3 | F's 3 tests pass |
| doctor seconds | 3.6 | 13.0 at load average 40 |

## Defects

Severity: critical = a safety boundary is crossed; major = a stated property does not hold in a way that matters for a
live run; minor = narrow, double-fault or cosmetic. Every line has file:line and a one-line reproduction
(`P=cam_live/docs/verification/repro/rt`).

### Critical

**D1 — dashboard: learner-written content runs script in the operator's session and approves a pending action.**
`runtime/dashboard.py:388` (`f()` returns anything that is not a number unescaped), used inside `innerHTML` at
`:393`, `:399`, `:402`, `:420`. A `MetaUpdateRecord` whose `bootstrap_mix` is a string of markup — a record the
learner returns and a file the learner uid owns — was rendered in headless Chromium; the script fired and POSTed
`/api/approvals/apr_g1/approve` with the operator's cookie; the authorization handle received
`resolve_approval('apr_g1', True, 'operator')`. All other planted vectors (observation text, failure detail,
approval preview, experiment text, learning-update fields, error text) were escaped.
`.venv/bin/python $P/p08_dashboard.py` (checks 12, 13).

### Major

**D2 — a `wait` in progress is lost by kill -9.** `runtime/loop.py:899-905` journals the wait, `:1022-1027` commits
and empties the journal while the wait is still running, `:228-264` (`_restore`) does not restore `_waiting`, and
`:294-299` drops an `end` without a `begin`. After a kill during the sleep the wait decision is counted
(`state.decisions` 9) but has no result record and never reaches the learner (8 delivered).
`.venv/bin/python $P/p02_crash.py in_wait_sleep:1` (also `after_outcome_log:1`).

**D3 — a result that was journalled but not yet logged never reaches `events.jsonl`.** `runtime/loop.py:936-951`:
the journal entry is written, then the `action_result` record; after a kill in between, replay (`:294-299`, `:963`)
hands the result to the learner but writes no record, so an action the world executed has no `action_result` and no
observation records, permanently. `.venv/bin/python $P/p02_crash.py after_wal_end:2` (also `torn_event:2`).

**D4 — approved actions are executed without any limit check.** `runtime/loop.py:629-652`. Six queued actions
approved together: 6 executions with `max_external_actions=2`, 12 spent with `money=4`.
`.venv/bin/python $P/p04_limits.py` (check 7).

**D5 — a NaN or infinite limit passes live validation and is silently not enforced.** `runtime/tasks.py:114`,
`:257` (`v <= 0` is false for NaN); `runtime/loop.py:392` (`limit > 0` is false for NaN), `:212` (not listed in
`limits_not_enforced`). A live task with `limits.money: .nan` or `.inf` loads; a run with a NaN money limit spent
12000. `.venv/bin/python $P/p05_tasks.py` (check 2); `.venv/bin/python $P/p04_limits.py` (check 6).

**D6 — the mechanical no-strategy check is bypassed by 15 of 19 disguises, and the content reaches the learner.**
`runtime/tasks.py:80-96`: keys are only token-matched against a list (`:87`), never text-scanned (`:91` scans values
only); `resources`, `success` and `measurement` accept any key; `objective`, `family`, `interfaces`,
`secondary_metrics` are free text. Accepted and passed to `agent.set_objective`: `growthPlan`, `strategies`,
`milestones` / `todo` / `outline` lists of moves, community subject and channel inside the objective, a recruitment
channel inside `success.definition`, moves as items of `secondary_metrics` / `interfaces`, a whole procedure as a
mapping key, numbered steps as keys, Spanish text, rephrased directives and cadences, content via YAML anchor + merge
key. Rejected: `strategy`, `growth_plan`, nested `niche`, `suggested_steps`, `step 1 …`.
`.venv/bin/python $P/p05_tasks.py` (checks 3a–3u, 4).

**D7 — redaction misses common shapes; some run files are not redacted at all.** In the redacted logs
(`events.jsonl`, `decisions.jsonl`): `runtime/logs.py:41` — `\b` before the name fails for `DISCORD_BOT_TOKEN=…`,
`?access_token=…`, and the pattern does not match JSON text `"password": "…"`; `:54` — only exact key names, so
`client_secret` and `DISCORD_TOKEN` keys pass; `:351` — `default=str` after redaction writes `bytes` values
verbatim; `:51`, `:129-152` — member identifiers pass as a bare id in text, `name#1234`, a list under `members` /
`user_ids`, `Observation.entities`, a metric name, a list of `{id}` objects. Outside the logs: `state.json` keeps
exception text (`runtime/loop.py:425`, `:368`) and the full decision of every approval (`:975`) unredacted, also
after the run has ended, and `/api/health` serves the former; `wal.jsonl` holds everything raw until the next
commit (disclosed by F). Redacted correctly: bot-token shape, `Bearer …`, `sk-…`, `token` key, password in the user part of a URL
(incidentally, by the e-mail pattern), webhook URL, e-mail addresses (text, key, failure detail), `member_id`, `<@mention>`, `user{id,username}`,
`message_content` without permission. `.venv/bin/python $P/p07_redaction.py`.

**D8 — a prediction whose query the learner defines itself is never measured.** `runtime/loop.py:967-971`: only
`result.expected_outcomes` issued by the environment enter the pending queue; `:710-722` resolves predictions only
against those. A registered prediction of the objective metric one day ahead (3 due by the end of the run) had 0
resolutions although both the environment and the verifier can measure that metric.
`.venv/bin/python $P/p01_loop_steps.py` (check F1).

**D9 — the membership verifier counts accounts it should not, and reports `achieved`.** `runtime/verifier.py:352-364`:
a `link_key` is remembered only for accounts that were not excluded, so the second account of a declared test
account, of a bot or of a purchased account is counted (99 + 1 → achieved); `:348-351` an id listed once as human
and once as bot counts; `:356` flags are case-sensitive (`Purchased` counts); `:56` ids are not normalised (` u5`,
`u5 `, `U5` are three members); `:373` with `min_account_age_s` set, unknown account age counts; 100 records that
carry only an id are `achieved` with uncertainty 0. `.venv/bin/python $P/p06_verifier.py` (checks A13–A20).

**D10 — what the learner says about outcomes is recorded and shown as outcome.** `runtime/loop.py:986-994` logs the
learner's `ExperimentRecord` as is, including an `actual` outcome with `verified=True` for a query that was never
issued; `runtime/dashboard.py:91-104` shows it as the experiment's actual result; `runtime/journal.py:188-190`
states the learner's `supported=True` as `[FACT]`; `runtime/loop.py:802` / `runtime/logs.py:385-390` register a
prediction whose `resolved_value` / `resolved_status` are already filled. No runtime `outcome` or `resolution`
record ever carried a predicted value. `.venv/bin/python $P/p01_loop_steps.py` (checks E2–E5).

**D11 — environment text can add `[FACT]` lines to the journal.** `runtime/journal.py:208` inserts the obstacle
description with its line breaks; a failure text containing
`\n- [FACT] The objective was verified as achieved (records: ev-0000000001)` appears as a statement line.
`.venv/bin/python $P/p13_journal.py` (check B1).

**D12 — preflight counts an environment without any Discord tool, and trusts the environment's own label.**
`runtime/preflight.py:93-100` checks integrations per modality (any available `api` tool satisfies `api`), so with
handles that answer "live" the Discord task is reported READY on an environment that has no Discord integration;
`:88` reads only the top-level `live` / `provenance`, which `runtime/cli.py:393` sets from the task mode, so an
environment labelled live whose Discord executor reports the local staging service is counted; `:43` recognises the
diagnostic policy by class name (a subclass passes). `.venv/bin/python $P/p09_preflight.py` (2a, 2o, 2p).

### Minor

**D13 — stop condition `authorization_revoked` is accepted but not implemented.** `runtime/loop.py:1140` reads
`stop_conditions` only for `verified_failure`; `runtime/tasks.py:25-28` calls the list "stop conditions the runtime
knows how to enforce". After revocation the run took 37 more denied decisions until `max_decisions`.
`.venv/bin/python $P/p04_limits.py` (check 8).

**D14 — a torn journal line swallows the next entry.** `runtime/loop.py:269-277` appends to `wal.jsonl` without
repairing a torn tail (`runtime/logs.py:199-215` skips the glued line). Torn `begin` + a second crash in the next
cycle: an executed action is never delivered to the learner and not counted (external actions 6 vs 7, money 6.9 vs
7.2); torn `end` + second crash: two `action_result` records for one action.
`.venv/bin/python $P/p02_crash.py` (rows `torn_wal_begin:2+in_learn:1`, `torn_wal_end:2+before_wal_end:2`).

**D15 — records left open or doubled by a crash.** `runtime/loop.py:807` vs `:871`: kill between the decision record
and the journal entry leaves a decision record without result (nothing executed). `:700-704`: kill between the
outcome record and the commit yields two outcome and two resolution records for one query, which the prediction
statistics of journal and dashboard count twice. `.venv/bin/python $P/p02_crash.py after_decision_log:2`;
`… after_outcome_log:1`.

**D16 — step 5 leaves no record.** `runtime/loop.py:1146` drops the return value of `update_beliefs`.
`.venv/bin/python $P/p01_loop_steps.py` (check "B step 5").

**D17 — an exception after execution loses the result.** `runtime/loop.py:933` (`float(result.cost.money)`) or any
error inside `_complete`: the action ran, but there is no `action_result`, no `record_result`, no count, and the run
idles to its deadline. `runtime/loop.py:1169`: `after_cycle` hooks are unguarded, so an unreachable central learner
raises out of `step()`. `.venv/bin/python $P/p15_malformed_result.py`; `.venv/bin/python $P/p14_multi.py` (check 8).

**D18 — an interrupted action on an environment without de-duplication is booked at zero cost.**
`runtime/loop.py:325-334`: runtime money 6.9, world 7.2. (The computer environment advertises
`at_most_once_by_action_id`, so this is the fallback path.)
`.venv/bin/python $P/p02_crash.py after_execute:2 --no-dedup`.

**D19 — `cam-live verify-completion` cannot pass any option to the checker.** `runtime/cli.py:632` (`--json` as a
flag shadows the checker's `--json PATH`), `:694` (`REMAINDER` after no positional): `--only CM-12` and `--json PATH`
exit 2. `.venv/bin/python $P/p10_cli.py` (check 9b).

**D20 — `cam-live resume` of a separated run.** `runtime/cli.py:522-529` starts `_supervise` without
`--authorization` / `--verifier-source` and does not write `supervisor.pid`; on an emergency-stopped run it exits 5
and the supervisor lingers for about two minutes. `.venv/bin/python $P/p11_separated.py` (check 14).

**D21 — `inspect-harness` names missing integrations only as whole interfaces.** `runtime/cli.py:254-260`; with the
default configuration 5 internal tools are registered and Discord is not named.
`.venv/bin/python $P/p10_cli.py` (check 5c).

**D22 — verifier log.** `runtime/verifier.py:172-178`: a restarted verifier compares only the last entry with its
head and continues on a log edited earlier; `:478`: raw test-account and suspect member ids are written to
`verifier.jsonl` (mode 0600, verifier uid). `.venv/bin/python $P/p06_verifier.py` (checks D6, E2).

**D23 — the journal puts earlier records under the current objective's provenance.** `runtime/journal.py:122`: after
a diagnostic objective followed by a staging one, 4 diagnostic decisions appear under "data provenance staging".
`.venv/bin/python $P/p13_journal.py` (check C1).

**D24 — control-file commands carry a self-declared `by`.** `runtime/control.py:93-109`. In in-process mode a
`resume` written by the learner's own process lifts an operator pause in 0.2 s and is logged as `by: operator`. In
separated mode the file is root-owned in a sticky directory and the learner uid is refused (p11 check 4).
`.venv/bin/python $P/p03_control.py` (check E1).

**D25 — two comments in `tasks/discord_100.yaml` are not true.** Line 16 ("read by the verifier process, never by
the learner"): the whole `TaskSpec`, including `success.exclusions`, goes to `agent.set_objective` and into the
`run_started` record (`runtime/loop.py:210-216`). Line 102 ("identifiers are hashed before they reach any log"): see
D7.

## What held under attack

* **Order and observability.** Calls follow the 17 steps in every cycle; every step except 5 leaves a record;
  `poll_events` is read before `observe` and delivered after it.
* **Predictions before outcomes.** For every resolution: registration seq < `action_start` seq < outcome seq <
  resolution seq; every per-cycle snapshot of every log is a byte prefix of the final file; seq unique, increasing,
  no gaps. Delayed outcomes carry the decision and action that issued the query and the world's true count.
* **Crash recovery.** In 60 scenarios (30 with, 30 without environment de-duplication) no action executed twice and
  the committed checkpoint was always complete; 16 of 23 single kill points kept every invariant, including kill
  after execution, during learning, during and after the checkpoint, and with a half-written checkpoint file; torn
  final lines of `events.jsonl` are truncated and seq continues. With a file-size limit ("disk full") the process
  ends with `OSError` and the restart is consistent apart from D2.
* **Operator control.** Asleep for an outcome one hour away: no environment call in 2.5 s; pause 0.5 s, resume
  1.0 s, stop 1.0 s (bound: `control_poll_s` = 1 s); stop during a 3 s action takes effect when the action returns,
  the result is recorded and nothing else starts; an emergency stop raised only at the policy is noticed in 4.0 s;
  after a stop, a resume command, an edited `state.json` and a restart execute nothing.
* **Limits.** Decisions, external actions, money (honest estimate), duration and compute end as `exhausted` with at
  most one in-flight action over, a checkpoint and a journal.
* **Verifier and separation** (root, real service uids): learner uid cannot read the key, the head or the log, nor
  write, replace or unlink the log or the audit log; all ten operator-only authorization calls are refused; a
  TaskSpec with target 0.5 is judged against the operator's 20; a forged `achieved` with a copied signature is
  rejected; private methods are not exposed; the sandbox uid reaches no socket; after `stop --emergency` the learner
  uid cannot reset the stop.
* **Preflight / CLI.** `discord_100.yaml` as shipped: BLOCKED with 23 named items; every diagnostic or staging
  component I offered blocks and is named; live `run` is refused without `--authorization`, with `--inproc`, and
  through the learner process directly; a forged authorization record is refused by the service.

## Status-file claims that are not true or not evidenced

* LX-02 `implemented`, with the task header and the `tasks.py` docstring saying any strategy-like content is
  rejected: 15 of 19 disguises are accepted (D6). The limitation text in the status file is honest about free prose
  but not about keys, anchors and unlisted keys.
* LX-14 `implemented`: limits are exceeded through approvals and through NaN / infinite values (D4, D5), and one
  listed stop condition does nothing (D13).
* LX-15 `implemented` and the summary's "crash-safe resume": D2, D3, D14.
* LX-10 `implemented`: D1.
* CM-12 `implemented`, "checked by hand that the command dispatches (`--help`)": `--help` prints the wrapper's own
  help; the checker's options cannot be reached (D19). The bare command does dispatch.
* `not_done`: "the write-ahead journal and state.json hold unredacted decisions/results of the uncommitted span" —
  `state.json` also keeps error text and approval decisions unredacted after the run has ended (D7).
* `tests/runtime/test_rt_doctor_cli.py:87` describes "diagnostic services that answer yes to everything"; the
  components used there say truthfully that they are diagnostic. A component that does answer "live" is believed
  (D12).

No number in `measured` was refuted.

## Commands of spec Part 17 and the modules they need

| Command | State |
|---|---|
| `doctor`, `test`, `status`, `pause` | work as described |
| `init` | works with a stand-in factory; default needs `cam_live.agent` (absent → exit 4) |
| `validate --mode staging` | needs `cam_live.validation` (absent → exit 4) |
| `inspect-harness` | works; integrations appear only when their executors are registered |
| `run` | refusals and both start modes work with a stand-in agent; default needs `cam_live.agent` |
| `resume` | in-process: works, also after kill -9; separated: D20 |
| `evaluate-transfer` | needs `cam_live.eval.transfer` (absent → exit 4) |
| `export-checkpoint` | saves through the agent, or copies the committed checkpoint; weights-only export needs the real checkpoint format |
| `verify-completion` | bare command runs the checker; options unreachable (D19) |

## Only confirmable with the real agent

* Content of decision, experiment, learning-update and meta-update records (rationale, diagnostics, losses, norms,
  checkpoint ids); `drain_experiments` and `explain_obstacle` are not in `AgentAPI`.
* That `save` is atomic and complete (weights, adapters, reward weights, optimizer, fast state, replay indices,
  seeds) and that `load` followed by re-delivered `record_result` / evidence is idempotent on the real experience
  store; the runtime relies on both.
* Compute accounting with GPU work, the learn / meta budgets, and what the learner does with `next_wake`.
* `init`, weights-only `export-checkpoint`, `evaluate-transfer`, `validate`.
* Multi-actor: the three actor methods, and tensor weights over the RPC (JSON only today).
* Whether the learner registers predictions through environment-issued queries or its own (D8 decides whether any
  of them resolve).
