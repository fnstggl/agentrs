# G2-runtime — independent re-verification of the live runtime at the code freeze

Verifier lens `g2runtime`, 2026-11-28 03:05–05:00 UTC. Commit judged: **a5ab7720f169c1bc23cc442be96a31310dfa29eb**
(frozen copy made with `git archive a5ab772`). During this work HX committed **8f6c615** (fixes of G2-harness R4-C1,
R4-M1, R4-m1 in `runtime/cli.py`, `logs.py`, `dashboard.py`, `plan_report.py`, `harness/safefs.py`); as the
coordinator asked, the requirements whose files include those modules (LX-07, LX-08, LX-10, LX-12, LX-13, LX-15,
LX-16, LX-20, CM-01 … CM-12) were re-judged on a second frozen copy (`git archive 8f6c615`) and their verdicts carry
that commit; every one of my probes gave the same result on both commits. Requirements: F — CU-24, LX-01/02/03/07/08/
09/10/12/13/14/15/16/17/19/20, CM-01 … CM-13, AE-06; D — CL-06. Verdicts: `verdicts_g2runtime.json`. Probes:
`repro/g2runtime/` (run from `/workspace/camlive`: `.venv/bin/python cam_live/docs/verification/repro/g2runtime/<probe>.py`;
each prints `[HOLDS ]` / `[BROKEN]`, exits 1 when a property under attack is broken, writes
`$G2RT_SCRATCH/<probe>.result.json`).

No source file, no other agent's test and no other lens's probe was changed (two of G-rt's probes were *copied* into my
directory with a stated minimal change). No live run was started and no external service was contacted: the Discord
membership source was driven by an in-process fake client; every service ran on loopback / unix sockets started by the
probes; no heavy GPU work was needed (every learner here was the debug / tiny configuration on the CPU; the one
CUDA test is the small checkpoint test of the CL-06 list, which needs no lock).

## Result in short

**20 verified, 11 partial, 0 failed, 0 unverified** of 31 (each basis in the table at the end and in the verdict file). The runtime's core holds up well under attack at the freeze: the 17 steps in order, kill -9
recovery at 477 + 10 points (settle phase included), signed operator commands (replay, transplanted signatures,
foreign keys, replay after restart all refused), the dashboard's operator credential / CSRF / same-origin checks, the
separated uids, and — newly exercised — a real `cam-live resume` of a separated run whose every process was SIGKILLed.
Every defect G-rt reported (D1–D25) is fixed: G-rt's probes now report 0 broken except the checks that encode the old
design (F-8) and the disclosed `--no-dedup` over-count.

What does not hold is mostly at the edges where the runtime meets the live objective:

* **the membership verifier counts members who were in the guild before the experiment began** (100 pre-existing
  members → `achieved` on day one) — G2R-2;
* **the secondary metrics of spec 13.2 are not measured with the live Discord source**: seven-day retention is computed
  from the current member list only (departures are invisible: 5 of 10 left → reported 1.0), participation and
  consented feedback are never produced, "resource use" is the verifier's own API request count, operating cost is
  never wired — G2R-3;
* **a subsequent objective can never be verified**: both run modes pin the verifier to the first task, so after
  `assign_objective` every verification raises and the run pauses after 5 errors — G2R-4;
* **multi-step programs bypass the runtime's external-action limit and accounting** (18 external executions, limit 4,
  counter 0; the authorization service's own ledger still counts steps) — G2R-5;
* **the strategy scan is bypassed** by prose joined with `_` / `-` in identifier fields that reach the learner and by
  zero-width characters in the objective (9 of 9 disguises accepted and delivered) — G2R-1.

## What I ran

All from a frozen copy unless marked *worktree* (the root / service-uid runs need a directory the service uids can
reach; the worktree's runtime and harness sources were identical to the commit named, checked with `git diff --stat`).

| Command | a5ab772 | 8f6c615 |
|---|---|---|
| `pytest cam_live/tests/runtime -q --rootdir=cam_live` | 240 passed, 5 skipped, 98.5 s | 240 passed, 9 skipped, 211 s (load 30–36) |
| `… -m 'privileged or browser'` | 5 passed, 15 s (*worktree*) | 8 passed + `test_rt_separated` passed in the *worktree* (22 s); in a relocated copy under `repro/g2runtime/` the authorization service of that one test exits at start (environment of the copy, see Notes) |
| CL-06 acceptance tests (credit checkpoint / evidence / learner checkpoint / replay, `tests/integration/test_agent_recovery.py`, `tests/runtime/test_int3b_settle.py`, `-m "gpu or slow or not (gpu or slow)"`) | 32 passed, 689 s (the small CUDA checkpoint test included; a real-model test outside the CL-06 list waited for the shared GPU lock and was not run) | (credit and integration sources unchanged) |
| G-rt probes `repro/rt/p01 … p15` (unchanged, `G_RT_SCRATCH` = my scratch) | p01 0/36, p02 0/477, p02 `--no-dedup` 6/477, p03 0/15, p04 0/11, p05 1/33 (1a: 16 REQUIRED, not 17), p06 0/38, p07 0/62, p08 2/18 (6, 8: by design, F-8; 12, 13 hold in Chromium), p09 1/28 (1b, same as 1a), p13 0/8, p14 0/11, p15 0/3 | same as a5ab772 (p01 0/36, p02 0/477, p03 0/15, p04 0/11, p05 1/33, p06 0/38, p07 0/62, p08 2/18, p09 1/28, p13 0/8, p14 0/11, p15 0/3) |
| `probe_cli_p10copy.py` (G-rt p10; step 4a `validate` run separately, next row) | 34 of 35 hold; 0c reports /workspace/camlive/runs/cam_live, which another workstream created on 2026-11-27 06:56 (before this session; my commands wrote under CAM_LIVE_HOME only) — not a defect | 34 of 35 hold (0c: same pre-existing directory); 7f / 7i: export with the process alive and offline work after the R4-C1 rewrite |
| `cam-live validate --mode staging --debug --config <tiny> --out D` (CPU) | ran 27 min through the computer phase and the first LabWorld lifetime (records inspected: LX-07 / 08 / 13) and was stopped by me when the CLI probe's timeout was reached | ran to the end: exit 1 (13 of 16 checks hold; approvals_exercised, meta_update_applied, meta_step_changes_policy fail at debug scale), report and evidence written, 737 s, GPU peak 0.0 GB |
| `cam-live evaluate-transfer --checkpoint X --baseline X --domains T1 --seeds 0 --lifetimes 1 --decisions 2` (debug weights-only export of `cam_live.agent.build_agent`) | — | exit 0, `executed: true`, 2 fresh-process runs, 0 failed, 83 s |
| `probe_separated_p11copy.py` (G-rt p11, root, *worktree*) | 1/39 (14: by design, resume refuses a stopped run) | 1/39 (same) |
| `probe_separated_resume.py` (root, *worktree*) | 0/8 | 0/8 |
| `probe_tasks_disguise.py` | 9/14 broken | 9/14 |
| `probe_verifier_live_source.py` | 6/8 broken | 6/8 |
| `probe_next_objective.py` | 1/2 broken | 1/2 |
| `probe_limits_program.py` | 3/3 broken | 3/3 |
| `probe_settle_crash.py` (kill -9 inside the settle phase, 10 scenarios) | 1/15 broken | 1/15 broken (same) |
| `probe_control_signed.py` | 0/9 | 0/9 |
| `probe_dashboard.py` | 1/19 broken | 1/19 |
| `probe_preflight.py` (real doctor of this machine, CLI) | 1/7 broken | 1/7 |
| `probe_gpu_lease.py` (scratch lock file, never the shared one) | 2/3 broken | 2/3 |
| `probe_redaction.py` | 11/13 broken | 11/13 |
| `probe_plan_report.py` | 5/6 broken | 5/6 |
| G2-harness `g15_operator_cli_root.py` (root, *worktree*; R4-C1 / R4-M1) | (broken, G2-harness round 4) | E1, E2 hold |

## Defects

Severity as G-rt: critical = a safety boundary is crossed; major = a stated property does not hold in a way that
matters for a live run; minor = narrow, double-fault or cosmetic. `P=cam_live/docs/verification/repro/g2runtime`.
Line numbers are of a5ab772 (unchanged in 8f6c615 for every file named below except where stated).

### Known from G2-harness round 4 (counted, not re-derived)

* **R4-C1 (critical)** — `cam-live export-checkpoint` as root trusted the learner's `state.json` for the checkpoint
  id / path, did not normalise the destination and merged with `copytree(dirs_exist_ok=True)`
  (`runtime/cli.py:690-704` at a5ab772). **Fixed in 8f6c615**: G2-harness's `g15` E1 holds (run by me, root,
  worktree at 8f6c615) and HX's `test_export_checkpoint_never_copies_learner_chosen_files_anywhere` passes.
* **R4-M1 (major)** — `cam-live plan-report` wrote `<out>.tmp` with `open("w")` through a learner symlink
  (`runtime/plan_report.py:341-345`). **Fixed in 8f6c615** (`safefs.write_atomic`; g15 E2 holds).
* **R4-m1 (minor)** — a FIFO in place of `state.json` blocked root's reads (supervisor, `status`, `resume`,
  `export-checkpoint`). **Fixed in 8f6c615** (`test_a_fifo_in_place_of_state_json_does_not_block_root` passes).

### Major

**G2R-1 (LX-02) — strategy content reaches the learner through fields the closed schema accepts.**
`runtime/tasks.py:153-159`: identifiers, metric names, tool patterns and the `family` label may be prose joined with
`_`, `-`, `.`, `/` (`_FAMILY_RX`, `_METRIC_RX`, `_TOOL_RX` have no length bound); `:78-100`: every text pattern is
`\b`-delimited, so `_`-joined words never match (`secondary.reddit_marketing_strategy_with_daily_giveaways` passes the
"strategy vocabulary", "timing" and "platform" rules), and U+200B inside a word defeats every rule. Accepted by
`load_task(for_run=True)` with `strategy_findings == []` and present in `learner_view` (what `agent.set_objective`,
`run_started` and `task.json` carry): `family: community/meme-server-for-pc-gamers/post-three-memes-then-invite-friends-personally/…`,
a `secondary_metrics` item, an `interfaces` item, an `authorized_tools` item, an `authorized_destinations` URL query,
`resources.discord.channels` labels (`start-meme-contest`, `invite-gamer-friends`, …), the `objective_id`, and a
one-sentence objective with zero-width spaces ("… community of pc g​amers … by posting memes on r​eddit e​very day and
running give​aways …", no confirmation needed). The same content written plainly is refused (controls). This refutes
the status claim that "prose cannot be hidden in an unlisted key, a list item or a label"; the shipped task files
themselves are clean. `.venv/bin/python $P/probe_tasks_disguise.py` (9 of 9 variants).

**G2R-2 (LX-03) — members who joined before the experiment are counted as members CAM attracted.**
`runtime/verifier.py:449-537` (`classify_members`) never compares `joined_at` with the experiment's start;
`measurement.window_start_ts` is used only for the deadline and `secondary.time_elapsed_s` (`:314-328`). With the
default live source (`harness.discord_membership.DiscordMembershipSource`, fake REST client) and 100 humans who joined
30 days before `window_start_ts`, the signed outcome is `achieved`, value 100, on day one. Spec 13.1 asks for members
the community *attracts*; the task's own `success.definition` does not exclude pre-existing members either, so neither
the rule nor the data can express it. `.venv/bin/python $P/probe_verifier_live_source.py` (check A).

**G2R-3 (LX-17) — the secondary metrics are not measured with the live evidence source.** `runtime/verifier.py:595-631`
(`_secondary`) with `DiscordMembershipSource` (the default for the Discord task): (a) seven-day retention uses only the
members the REST list shows now; members who left are not listed, so retention is 1.0 whenever the cohort is
non-empty (10 joined, 5 left on day 2 → 1.0 at day 10; true 0.5); (b) the source has no `participation` / `feedback`,
so voluntary participation and consented feedback are never reported; (c) "usefulness of community resources" is
`secondary.resource_use.discord.verifier.requests`, the verifier's own API call count
(`harness/discord_membership.py:94-95`); (d) `cost_source` is never passed — `verifier_service.py:85` and
`loop_service.py:94` call `build_verifier` without it — so operating cost is never reported in any run mode. Only time
elapsed is a real secondary measurement there. The separation from the primary value holds (C6).
`.venv/bin/python $P/probe_verifier_live_source.py` (B, C1-C4).

**G2R-4 (LX-10, LX-19) — a subsequent objective can never be verified.** `runtime/verifier_service.py:85` and
`runtime/loop_service.py:94` build the verifier with `pin=True`; `runtime/verifier.py:305-312` raises
`PermissionError("this verifier serves objective … only")` for any other objective; `runtime/loop.py:920-937`
(`_switch_objective`) changes the task but not the verifier. LabWorld, the runtime's own `MetricThresholdVerifier`
pinned as both services pin it: objective A completes; the operator's signed `assign_objective` of a valid task B is
applied ("objective lab-second started"); every verification of B raises; after 5 the run pauses ("paused after 5
consecutive errors; last: PermissionError: this verifier serves objective 'obj_lab' only"); B never gets a signed
outcome. F's test (`test_rt_control.py::test_assign_subsequent_objective`) uses the unpinned `LabVerifier` and does
not see it. (There is no `cam-live assign` command, which the LX-19 limitation names: the dashboard's
`/api/control/assign-objective` and `OperatorControl.assign_objective` are the operator's paths.) `.venv/bin/python $P/probe_next_objective.py`.

**G2R-5 (LX-14) — programs bypass the runtime's external-action limit and accounting.** `runtime/loop.py:1327-1330`
(`_is_external`) judges only the top-level action; a program has modality internal and the program tool declares
side effects none (`harness/env.py:487-488`, `predict/candidates.py:321-328`), so `_would_exceed` (`:1311-1325`)
never applies `max_external_actions` to it and `_complete` (`:1449-1452`) counts 0 external actions. LabWorld with a
`program` tool declared like the computer environment's, the learner choosing programs of 3 `lab.act` steps,
`max_external_actions = 4`: 18 external executions, runtime counter 0, the objective ended by `max_decisions`, never by
the external-action limit (so no exhaustion checkpoint for it). Mitigation outside the runtime: the authorization
service's ledger counts every declared external step of a program (`harness/auth/policy.py:741-750`), so with the
real service the excess steps are denied — the runtime's own enforcement, its `state.json` counters / dashboard
resources and its "exhausted → checkpoint → end" path are what fail. `.venv/bin/python $P/probe_limits_program.py`.

### Minor

**G2R-6 (LX-20) — redaction misses shapes and over-redacts prose.** Written verbatim to the run directory through
`RunLog`: a JWT (in text and as a bare value), HTTP Basic credentials after `Authorization: Basic`, a Google-style
OAuth token (`ya29.…`), a 40-hex secret under an unlisted key, a member id as an *integer* under an unlisted key
(`owner_id`), a phone number without separators, an IPv6 address (IPv4 is redacted as personal data)
(`runtime/logs.py:37-58` patterns, `:197-227` structured values). Over-redaction: the "assignment" rule (`:46`)
replaces the word after any `<secret-word>:` — the runtime's own stop reason becomes
"stop condition authorization_revoked: [REDACTED:secret] for this task has been revoked by the operator" in
`state.json` / the dashboard (also visible in G-rt p04 check 8); the IPv4 rule (`:54`) eats version numbers
(`2.10.3.1`). `.venv/bin/python $P/probe_redaction.py`.

**G2R-7 (plan-report, INT3B §4 claim) — CAM-written strings become report structure; note bodies are edited.**
`runtime/plan_report.py:282`, `:287` (note tags) and `:302` (operation name, candidate source) are written into the
Markdown unquoted: a 52-character tag (`internal.note` allows 60) adds two `## Requests to the operator` sections, an
operation name adds an `## Approvals (1)` section with `* apr_fake: approved by operator`. "Verbatim, unedited": the
run's redactor changes benign note text ("authorization: pending" → "[REDACTED:secret]", "2.10.3.1" → "[REDACTED:ipv4]";
a 17-digit number becomes a member hash by design). The JSON form keeps the strings exactly.
`.venv/bin/python $P/probe_plan_report.py`.

**G2R-8 (LX-15) — kill -9 right after `objective_finished` doubles it.** `runtime/loop.py:1742-1775` (`_finish`):
the forced commit (`:1746`) saves `state.json` while the status is still running and the settle record already cleared;
the `objective_finished` record follows; `state.json` is saved again only after it. Killed between the two, the
restarted runtime ends the objective a second time: two `objective_finished` records (seq 134 and 142), the second
with `settle: null`. All other settle-phase kill points (after `settling`, during outcomes / reconcile / learn / commit,
double crash) keep every invariant of G-rt's p02 judge. `.venv/bin/python $P/probe_settle_crash.py`.

**G2R-9 (GPU lease) — the hold bound is not a bound; learner calls run without the lock.** (a)
`runtime/gpu_cycle.py:116-117`, `:186-206`: the hold is checked only between cycles against the previous cycle's
duration, so one long cycle overruns by its whole length (bound 1.0 s → 2.62 s; INT3B measured 1234 s for 840 s and
discloses it). (b) `runtime/loop.py:1806` (`_apply_commands` at the top of `step()`) runs before the `before_cycle`
hook re-takes a lease released for a sleep, so `agent.save` for a pause commit or an export runs while the shared lock
is not held (2 of 2 such calls in the probe). `.venv/bin/python $P/probe_gpu_lease.py`.

**G2R-10 (LX-12) — the component cross-check of preflight is a spelling list.** `runtime/preflight.py:31`, `:114-134`:
components of a live-labelled environment whose `environment` is `staging-discord` or `Staging `, or whose address is
`0.0.0.0`, `localhost.`, `[::ffff:127.0.0.1]`, a private `10.x` address, or under a key other than
`base_url` / `url` / `endpoint` (`api_base`), are not named as blocking. The binding checks are independent and hold
(the Discord executor's own `environment_of` accepts only discord.com hosts, the authorization service must hold the
task in mode live with a valid record, the verifier's source must be live), so this is defence in depth only.
`.venv/bin/python $P/probe_preflight.py` (check 2).

**G2R-11 (LX-10) — a 500 response carries no security headers.** `runtime/dashboard.py:398-402`: the header
middleware does not wrap Starlette's server-error response. A `state.json` that is valid JSON but not an object (the
learner process writes `state.json`) makes `/api/status` answer 500 without CSP / nosniff / DENY (plain text, so no
script runs; it also takes the status view down). Every other response kind (200, 401, 403, 404, 422, app.js) has
them. `.venv/bin/python $P/probe_dashboard.py`.

**G2R-12 (CM-13) — the one place for commands is stale at the freeze.** `docs/REPRODUCE.md` §14 describes the
superseded GPU-lock design ("commits its checkpoint, pauses and exits, and `cam-live resume` continues"; "none of these
keys is set in configs/cam_live.yaml" — `configs/cam_live.yaml:54-56` sets `gpu_lock`, `gpu_hold_s`, `gpu_yield_s`);
`--accept-ephemeral-storage`, which the operator needs to launch on this machine (`tools/launch_live.sh`, ledger), is
not mentioned; §13 still lists `validate --mode staging --debug` as not a CPU run (fixed by INT3B item B).

**G2R-13 (LX-15, numbers) — `--no-dedup` over-count.** `repro/rt/p02_crash.py --no-dedup`: 6 of 477 broken, not the 4
the status file states — the two extra rows are the 30000-byte file-size-limit lifetime (money 16.4 vs 16.1, external
actions 15 vs 14): the same disclosed mechanism (an interrupted action is booked at its estimate), one more scenario.

## What held under attack (new evidence in this round)

* **Signed operator commands** (`probe_control_signed.py`, 9/9): a replayed applied `resume` (same id), a copied
  signature on a new id or on a different command (`export_checkpoint` to an attacker path), a command signed with
  another run's key, and a replay after a runtime restart are all refused and logged `authenticated: false` /
  "replayed"; unsigned brakes (`pause`, `stop`) still apply.
* **Dashboard** (`probe_dashboard.py`, 18/19; G-rt p08 12, 13 in Chromium; `test_rt_fix_dashboard` with a camlearn
  process): viewing token alone, operator credential in the query or a cookie, missing / mismatched CSRF,
  `Sec-Fetch-Site: cross-site`, `Origin: null`, another origin, a DNS-rebinding request without the secrets and a wrong
  credential are refused (401 / 403) and write no command; the full credential set is accepted.
* **Preflight**: the shipped Discord task is BLOCKED with all 16 REQUIRED paths named (25 items);
  `--accept-ephemeral-storage` moves only `persistent_storage` to `accepted_risks`, recorded as "operator (command line
  --accept-ephemeral-storage)" with the finding; it has no effect in staging, cannot come from an environment variable
  or a task file (unknown key), and the live preflight stays blocked by the other items (`probe_preflight.py` 4a-4d;
  F's tests cover run.json / supervisor hand-over and resume).
* **Crash recovery**: p02 0/477; settle phase 9 of 10 kill scenarios keep every invariant (above).
* **Separated restart** (`probe_separated_resume.py`, root): separated staging run, operator pause, SIGKILL of all 6
  processes, `cam-live resume` → a new supervisor (pid recorded, alive), the learner back under camlearn with the
  persisted state (`run_resumed`), the pending signed resume applied, decisions continue, no duplicate `action_result`,
  stop works. This closes F's open limitation "a real supervisor restart of a paused separated run was not exercised".
* **Records of the integrated agent** (debug configuration, the staging validation run of the CLI probe): decision
  records carry candidates (10), probabilities, predictions, rationale built from the numbers, model version,
  checkpoint id and the runtime's situation digest; learning updates carry trigger, losses, groups, gradient norms,
  weight-delta norms, hashes, credit changes and prediction changes (but `checkpoint_id` is empty in all 9: the
  checkpoint ids are in the runtime's `checkpoint` events and in the decision records); experiment records carry the
  spec 14.2 fields with `learner_reported_actual` kept apart.

## Per requirement (basis)

| Requirement | Verdict | Commit | Basis |
|---|---|---|---|
| CU-24 | **verified** | a5ab772 | waiting makes no environment call; delayed outcomes once across kill -9 (p02 0/477, p01 F1, settle) |
| LX-01 | **verified** | a5ab772 | external task file, spec objective verbatim, 16 REQUIRED items all named by preflight |
| LX-02 | **partial** | a5ab772 | shipped files clean, G-rt's 19 disguises refused; 9 of 9 new disguises accepted and delivered (G2R-1) |
| LX-03 | **partial** | a5ab772 | exclusions, signatures, log, separation hold (p06 0/38, p11); pre-existing members counted (G2R-2) |
| LX-07 | **verified** | 8f6c615 | experiment records with the 14.2 fields from the integrated agent; learner actuals kept apart |
| LX-08 | **verified** | 8f6c615 | every learning / meta record logged with losses, groups, norms, deltas (checkpoint_id empty: learner content) |
| LX-09 | **verified** | a5ab772 | mechanical journal with record ids, no injection, provenance kept (p13 0/8) |
| LX-10 | **partial** | 8f6c615 | credential / CSRF / origin / CSP hold; next objective unverifiable (G2R-4); 500 without headers (G2R-11) |
| LX-12 | **partial** | 8f6c615 | blocked with every item named, accept-risk operator-only and recorded; label cross-check misses spellings (G2R-10) |
| LX-13 | **verified** | 8f6c615 | a record per decision with candidates, predictions, probabilities, rationale, versions |
| LX-14 | **partial** | a5ab772 | single-action limits enforced (p04 0/11); programs bypass max_external_actions (G2R-5) |
| LX-15 | **partial** | 8f6c615 | 17 steps (p01 0/36), kill -9 (p02 0/477); doubled objective_finished on a crash (G2R-8), --no-dedup 6/477 (G2R-13) |
| LX-16 | **verified** | 8f6c615 | detached in-process and separated; real separated restart after SIGKILL of every process (new) |
| LX-17 | **partial** | a5ab772 | kept separate from the primary; retention biased, participation / feedback / cost absent, resource use mislabelled (G2R-3) |
| LX-19 | **partial** | a5ab772 | stop with checkpoint after success holds; assigned next phase cannot be verified (G2R-4) |
| LX-20 | **partial** | 8f6c615 | envelope, fsync, sealed state, p07 0/62; JWT / Basic / OAuth / IPv6 / int ids leak, prose over-redacted (G2R-6) |
| CM-01 | **verified** | 8f6c615 | doctor sections, --strict |
| CM-02 | **verified** | 8f6c615 | init with default (tiny) and stand-in factories |
| CM-03 | **verified** | 8f6c615 | test runs pytest, JUnit, unknown suite 2, empty selection not success |
| CM-04 | **verified** | 8f6c615 | the debug staging validation ran to the end on the CPU (exit 1: 13 of 16 checks hold, as reported) |
| CM-05 | **verified** | 8f6c615 | tools, scopes, missing integrations incl. Discord |
| CM-06 | **verified** | 8f6c615 | every refusal, in-process and separated starts |
| CM-07 | **verified** | 8f6c615 | status from the run directory; FIFO no longer blocks (8f6c615) |
| CM-08 | **verified** | 8f6c615 | pause acknowledged with checkpoint, nothing decided while paused |
| CM-09 | **verified** | 8f6c615 | resume live, after kill -9, refuses stopped, separated restart |
| CM-10 | **verified** | 8f6c615 | dispatches; executed on a debug export (exit 0, 2 fresh-process runs) |
| CM-11 | **verified** | 8f6c615 | failed at a5ab772 (R4-C1, known); fixed and working at 8f6c615 |
| CM-12 | **verified** | 8f6c615 | checker runs bare and with every option |
| CM-13 | **partial** | a5ab772 | REPRODUCE.md stale on the GPU lease, the ephemeral-storage flag, validate --debug (G2R-12) |
| AE-06 | **verified** | a5ab772 | two actors, separation, RPC learner, unreachable learner handled (p14 0/11) |
| CL-06 | **partial** | a5ab772 | 32 acceptance tests pass (debug scale, kill -9, fresh-process load, settle restart); real configuration not verified |

## Notes and what I could not check

* Live paths (a real Discord guild, a live preflight with services, a live authorization record) were judged by code
  reading and staging equivalents only; nothing contacted discord.com.
* The real configuration (Qwen3-4B on CUDA) was not run: every runtime path here used the tiny / debug backbone, so the
  GPU lease was exercised with a scratch lock file and LabWorld cycles only.
* `test_rt_separated.py` fails in a copy of 8f6c615 relocated under `repro/g2runtime/.frozen_8f6c615/` (the
  authorization service exits at start; reproducible there, not in a hand-made run with the same task from the same
  copy, which started), and passes in the worktree at 8f6c615 (22 s) and at a5ab772; I attribute it to the relocated
  tree and did not pursue it further. The copy was removed.
* G-rt's p08 checks 6 / 8, p05 1a, p09 1b and p11 14 still encode the old design (change request F-8); I did not edit
  G-rt's probes. My `probe_preflight.py` check 1 and `probe_separated_resume.py` replace 1a / 1b and 14.

---

# Round 2 — at the final freeze 257267c1c31568b693adf0cbcd0e9f9b547e62e1

2026-11-28 05:50–06:40 UTC. Frozen copy `git archive 257267c`; the root / service-uid probes ran in the worktree, whose
runtime and harness sources were identical to 257267c (`git diff --stat 257267c -- cam_live/src` named only
`eval/*` files of another workstream) — except the last check of `probe_gpu_lock_path.py`, see G2R-16. Claims judged:
docs/INT3B.md §10 (commits 93533b5, b741142, 577d70e, 34e184a, 1f34590, 6ee6ffd, ceb5095, 47d5eb1, da6ebb6, c68e6d0),
F.json, INT3B-6..8, and the master's 257267c. Every requirement whose files changed since my round-1 verdict commits was
re-judged; LX-09, AE-06 and CM-13, whose files are unchanged, were re-checked and recorded at 257267c as well (p13
0/8, p14 0/11; CM-13's REPRODUCE.md is unchanged, so G2R-12 stands). Every verdict now carries commit 257267c. GPU: nothing in this round loads a pretrained model; every run from a
copy that could take the GPU lock exported `CAM_LIVE_GPU_LOCK=/workspace/agentrs/runs/cam_unified/gpu_heavy.lock`
(the hazard is G2R-17 below).

## Round 2 — what I ran

| Command (257267c) | Result |
|---|---|
| `pytest cam_live/tests/runtime -q --rootdir=cam_live` (copy) | 273 passed, 9 skipped, 116 s |
| `… -m 'privileged or browser'` (worktree) | 9 passed, 24 s |
| CL-06 acceptance tests + `test_int3b_settle.py` (copy, `-m "gpu or slow or not (gpu or slow)"`) | 32 passed, 388 s (`CAM_LIVE_GPU_LOCK` = the machine lock) |
| `cam-live validate --mode staging --debug` (copy, CPU) | ran to the end: exit 1, 15 of 16 checks (approvals_exercised fails in this run), 565 s, GPU peak 0.0 GB |
| `cam-live evaluate-transfer` minimal (debug export, copy) | exit 0, `executed: true`, 2 fresh-process runs, 0 failed, 43 s |
| G-rt p01 … p15 (copy) | p01 0/36, p02 0/477, p02 `--no-dedup` 4/477, p03 0/15, p04 0/11, p05 1/33 (1a design), **p06 14/38** (by construction, see LX-03), p07 0/62, p08 2/18 (design), p09 1/28 (design), p13 0/8, p14 0/11, p15 0/3 |
| round-1 probes, re-run | redaction 0/13, control_signed 0/9, dashboard 0/19, preflight 0/7, settle_crash 0/15, gpu_lease 1/3 (check 1, disclosed), CLI p10 copy 34/35 (0c: others' files in runs/), separated_resume 0/8, p11 copy 1/39 (design) |
| `probe_tasks_disguise.py` (+9 new variants) | round-1 variants 9/9 refused; **round-2 variants 9/9 accepted** |
| `probe_verifier_live_source.py` (service-style construction, +A2–A4) | A, A3, B, C1, C2, C4–C6 hold; **A2, C3 broken**; A4 note |
| `probe_verifier_rules_after_start.py` (new) | 0/9 |
| `probe_next_objective.py` (+U1–U4, E1) | 1, U1–U4 hold; **E1 broken** |
| `probe_limits_program.py` (+4, 5) | 1–3, 5 hold; 4 broken (disclosed overshoot) |
| `probe_plan_report.py` (+R2a–R2d, fence-aware 1a/1b) | structure, controls, byte-exactness hold; **R2d broken** |
| `probe_gpu_lock_path.py` (new, root, worktree) | 1 holds; **2, 3 broken** at 257267c (3 fixed in 59807fc) |

## Round 2 — round-1 defects

| Defect | At 257267c | Basis |
|---|---|---|
| G2R-1 task disguises | **fixed for the 9 round-1 variants; 9 new variants pass (G2R-1b)** | probe_tasks_disguise |
| G2R-2 pre-existing members | **fixed (passive case); rejoin counts (G2R-2b)** | A, A2, A3 |
| G2R-3 live secondary metrics | **fixed except usefulness of resources (G2R-3b)** | B, C1–C6 |
| G2R-4 next objective unverifiable | fixed; forged / unsigned / foreign / changed-file assignments refused | probe_next_objective 1, U1–U4 |
| G2R-5 programs and the external-action limit | fixed; a program started below the limit still runs all its steps (disclosed) | probe_limits_program 1–5 |
| G2R-6 redaction shapes | fixed in the logs (13/13) | probe_redaction |
| G2R-7 plan-report structure / editing | fixed (literal blocks, escapes, byte-exact JSON); the price is G2R-7b | probe_plan_report |
| G2R-8 doubled objective_finished | fixed | probe_settle_crash 15/15 |
| G2R-9 GPU holds | (b) fixed (no learner call without the lock); (a) a single long call still overruns (disclosed: not pre-empted) | probe_gpu_lease 2, 3 hold; 1 broken |
| G2R-10 preflight spellings | fixed for my 7 spellings; of 16 more, `localtest.me`, a bare `host` key and a scheme-less `endpoint` are missed (defence in depth) | probe_preflight; inline check |
| G2R-11 dashboard 500 | fixed | probe_dashboard 19/19 |
| G2R-12 REPRODUCE.md stale | **not fixed** (file unchanged since a5ab772; it also still names the agentrs lock path as the one in use) | CM-13 |
| G2R-13 `--no-dedup` count | the claim now says 6/477; this run gives 4/477 — the disk-full scenario varies between runs; same mechanism | p02 --no-dedup |

## Round 2 — new defects

**G2R-1b (LX-02, major as G2R-1) — disguises that survive the normalisation.** `runtime/tasks.py` `normalise_for_scan`
/ `_content_hits` (NFKC, invisible characters, separator and camelCase splitting) then the same whole-ASCII-word rules:
accepted by `load_task(for_run=True)` with no finding, and present in `learner_view` — a one-sentence objective with
Cyrillic homoglyphs inside the matched words (U+0435 / U+0430 are letters; NFKC keeps them), the objective in leetspeak
(`r3ddit`, `3very d4y`, `g1veaways`), an objective naming Discord listing sites the lists do not contain ("listing it on
disboard and top.gg and bumping it whenever the cooldown allows"), the platform name spelled letter by letter, and words
concatenated without a separator in `secondary_metrics`, `family`, `discord.channels` labels (`postmemesdaily`,
`invitegamerfriends`, `dmeverynewjoiner`), `objective_id`, plus a homoglyph URL query in `authorized_destinations`.
`.venv/bin/python $P/probe_tasks_disguise.py` (R2 rows).

**G2R-15 (LX-14, major, new; pre-existing since round 1) — an extended duration of the running objective is not what
the verifier judges.** `runtime/loop.py` `_assign`: for the same objective id the runtime updates its boundaries and
does not call `verifier.assign_objective`; `runtime/verifier.py` `assign_objective` refuses to replace the pinned task
anyway, and `_deadline` uses the pinned task's `duration_s`. LabWorld, the pinned `MetricThresholdVerifier` as the
services build it: the operator's signed assignment of the same objective with `duration_s` 1 → 10 days is applied
("task boundaries updated for the current objective"), then the verifier signs FAILED at the old deadline and the run
ends `failed` after 2.1 days. `.venv/bin/python $P/probe_next_objective.py` (E1).

**G2R-16 (LX-16, major at 257267c; fixed after the freeze in 59807fc) — the separated learner could not take the GPU
lock.** `runtime/gpu_cycle.py:62` opened the lock `"a+"`; the machine's lock file is root:root 0664, so as camlearn
`GpuLease(GPU_LOCK_PATH).acquire()` raised PermissionError (`probe_gpu_lock_path.py` 3), i.e. `loop_service.run_service`
fails before the agent is built in every separated run with a pretrained backbone on CUDA — the configuration of the
live planning launch. 59807fc (after the freeze) falls back to a read-only open in `nn.gpu.open_lock_file`; in the
worktree at that commit a camlearn learner with `runtime.gpu_lock: on` reached the queued `flock` wait (another tenant
held the lock) and was stopped after 60 s by me. Still opening `"a+"`: `harness/visual.py:152` (the visual executor's
model lock; cambrowser in separated mode), `harness/grounding/bench.py:99`, `runtime/cli.py:156` (root only).

**G2R-17 (LX-16, major for any run from a copy) — the default lock path is not the shared lock outside this worktree.**
`config.py`: `GPU_LOCK_PATH` defaults to `<repo>/runs/gpu_heavy.lock`, which is an untracked symlink in this worktree
only; in a `git archive` copy of the same commit (how the validations and verifiers run frozen code) or a fresh clone it
does not exist and the first lease creates a private file — no exclusion with the other tenant
(`probe_gpu_lock_path.py` 2). Every process of this worktree does resolve to the same inode as the machine lock
(check 1: runtime lease, `nn.gpu.gpu_lock`, `cam-live`, tests). Mitigation: export `CAM_LIVE_GPU_LOCK` (as I did).

**G2R-2b (LX-03, minor) — a rejoin counts.** The baseline holds counts only and `classify_members` compares the
platform's current `joined_at` with the start (`verifier.py:579`), which a rejoin resets: 10 members present at the
start leave and rejoin → value 10 with nobody new. (CAM with a kick permission, or a request to rejoin, would inflate
the count by up to the number of members present at the start.) `probe_verifier_live_source.py` A2.

**G2R-3b (LX-17, minor) — "usefulness of community resources" is still not measured.** What is reported as
`secondary.resource_use.*` is CAM's own external actions / decisions / compute from the authorization ledger, and the
"Not measured" note does not name usefulness. `probe_verifier_live_source.py` C3.

**G2R-7b (LX-20, minor, new) — the plan report exports members' data in the clear.** Since 34e184a `build_report`
no longer passes CAM's notebook / request texts through the run's redactor (`clean()` is the identity): a member's
`name#1234`, e-mail address and Discord id in a CAM note appear in `plan_report.md` / `.json`, which root writes into
the run directory with mode 0644, although the task's `store_member_identifiers` is false and the run's logs hash them.
Byte-exactness and redaction conflict here; one form (e.g. the Markdown) could be redacted. `probe_plan_report.py` R2d.

**G2R-18 (minor, code reading) — assigned task files are read twice.** `verifier.py` `_load_assigned_file` and
`loop.py` `_assign` hash the file and then parse it again from the path; a task file in a directory the learner's
group may write (e.g. the run directory) can be swapped between the two reads. The dashboard only assigns files of the
repository's tasks directory; `OperatorControl.assign_objective` takes any path. After a verifier restart an assignment
whose file was edited is not restored (`assignment_not_restored`), and verification of that objective is then refused.

## Round 2 — what held under the new attacks

* **Verifier switch** (probe_next_objective U1–U4): an unsigned `assign_objective` sent straight to the verifier (what
  the learner's process could do over the socket), the operator's signature on altered arguments, a command signed with
  another run's key, and the operator's command after its task file changed are all refused by the verifier itself.
* **Baseline across a restart** (A3): without `window_start_ts` the start is the first reading, kept in the hash-chained
  log; a restarted verifier counts exactly the members who joined after it.
* **Programs across approvals** (probe_limits_program 5): approved programs are re-checked and every step counted
  (runtime counter = world executions).
* **Plan report** (R2a–R2c): a note with a longer backtick fence, a tag with backticks / pipe / newline, ANSI escapes,
  a bidi override, CRLF, U+2028 and a non-BMP character neither add structure nor reach the file as controls; the JSON
  form keeps every text byte-exact.

## Round 2 — per requirement

| Requirement | Verdict | Commit | Basis (round 2) |
|---|---|---|---|
| CU-24 | **verified** | 257267c | no environment call while waiting; outcomes once across kill -9 and settle (p02 0/477, settle 15/15) |
| LX-01 | **verified** | 257267c | shipped task clean under the new scan; 16 REQUIRED paths named |
| LX-02 | **partial** | 257267c | round-1 disguises refused; 9 new ones (homoglyphs, leetspeak, unlisted platforms, concatenation) accepted (G2R-1b) |
| LX-03 | **partial** | 257267c | start baseline holds incl. restart; rules hold after the start; a rejoin of a pre-existing member counts (G2R-2b) |
| LX-07 | **verified** | 257267c | experiment records, learner actuals apart (p01 0/36) |
| LX-08 | **verified** | 257267c | every learning / meta record logged |
| LX-09 | **verified** | 257267c | journal.py unchanged; p13 0/8 re-run |
| LX-10 | **verified** | 257267c | credential / CSRF / origin / headers incl. 500 (19/19); signed next objective works; forged ones refused |
| LX-12 | **verified** | 257267c | blocked with items named; accept-risk operator-only; spellings fixed (3 exotic misses, defence in depth) |
| LX-13 | **verified** | 257267c | a record per decision, journalled first |
| LX-14 | **partial** | 257267c | program steps counted (G2R-5 fixed); duration extension of the running objective ignored by the verifier (G2R-15) |
| LX-15 | **verified** | 257267c | 17 steps, kill -9 0/477, settle 15/15 (G2R-8 fixed); --no-dedup 4/477 this run |
| LX-16 | **partial** | 257267c | detached and separated runs; separated learner could not take the GPU lock at 257267c (G2R-16, fixed in 59807fc); copy uses a private lock (G2R-17) |
| LX-17 | **partial** | 257267c | retention, participation, feedback status, cost, time now measured; usefulness of resources not (G2R-3b) |
| LX-19 | **verified** | 257267c | stop with checkpoint; signed next objective verified |
| LX-20 | **partial** | 257267c | log redaction 13/13; the plan report exports members' data unredacted (G2R-7b) |
| CM-01 | **verified** | 257267c | doctor |
| CM-02 | **verified** | 257267c | init |
| CM-03 | **verified** | 257267c | test |
| CM-04 | **verified** | 257267c | debug validation ran to the end (exit 1: 15 of 16, approvals not chosen by the learner) |
| CM-05 | **verified** | 257267c | inspect-harness |
| CM-06 | **verified** | 257267c | run refusals and starts |
| CM-07 | **verified** | 257267c | status |
| CM-08 | **verified** | 257267c | pause |
| CM-09 | **verified** | 257267c | resume incl. separated restart |
| CM-10 | **verified** | 257267c | dispatches and executes (debug export, exit 0) |
| CM-11 | **verified** | 257267c | export alive and offline, R4-C1 / R5-m1 tests |
| CM-12 | **verified** | 257267c | verify-completion with options |
| CM-13 | **partial** | 257267c | REPRODUCE.md unchanged: G2R-12 stands (plus the lock path, new validate options) |
| AE-06 | **verified** | 257267c | multi.py unchanged; p14 0/11 re-run |
| CL-06 | **partial** | 257267c | 32 tests pass at debug scale; real configuration not verified by me |

**Round 2 totals: 23 verified, 8 partial, 0 failed, 0 unverified** of 31.
