# HX — harness requirements that no status file claimed (2026-11-27)

Agent HX (commit prefix `cam live [HX]`). Scope: IH-13, IH-14, IH-15, IH-16, IH-17, CU-19, CU-25, CU-26, CU-35,
CU-36, CU-37, CU-38, OP-04, OP-07, OP-09. For each requirement HX read the implementation and the tests, ran the
acceptance tests, fixed what was not met (in `src/cam_live/harness/**`, with a regression test that fails on the
code before the fix) and wrote a claim into the owning status file (`docs/status/EA.json`, `EB.json`, `EC.json`,
`ES.json`; OP-07, whose workstream is M, in `EA.json` — change request HX-1). A claim is not a verification.

Nothing contacted Discord or any other external service; every Discord result here is from the local staging
service. The final test run is `docs/evidence/test_results/HX_fix.xml`: `tests/harness/{env,auth,skills,compute}` and
`tests/integration/test_credentials_membership.py` with `-m "browser or privileged or not (gpu or slow or network)"`,
as root — 605 passed, 0 failed, 4 deselected, 241 s (at 5562e65). Checker for the 15 ids: 14
implemented-needs-reverification (every named test fresh; no independent verdict yet), CU-19 partial. Run order
env first: see change request HX-4 (a pre-existing Playwright / asyncio test-isolation failure when compute's
browser tests run first).

## What was missing and is now implemented

### OP-09 — hard platform constraints (commit `a1cdc0e`)

`src/cam_live/harness/auth/platform_constraints.py`, called from `Evaluator.node` (so from `authorize` for every
step of a program or skill and from `check_primitive` for every primitive an executor reports). Fixed in code, not
policy data: no policy file, live record, approval or owned / pre-authorized / interactive entry can permit them.

* `X-HUMAN-VERIFICATION`: CAPTCHA-solving services (any request, any modality, also named in generated code);
  interacting with or answering a challenge endpoint (reCAPTCHA, hCaptcha, Cloudflare Turnstile / challenge
  platform, Arkose, GeeTest, DataDome, PerimeterX, AWS WAF, Friendly Captcha, MTCaptcha, Akamai); solved tokens in
  arguments, string bodies or URL queries; solver / detection-evasion libraries and `navigator.webdriver` overrides
  in code; proxy / IP-rotation arguments; operations named like CAPTCHA solvers. Loading a page that embeds a
  challenge stays possible (a read).
* `X-USER-ACCOUNT`: the browser interacting with Discord or opening its client application or API with the
  browser's session (Discord's documentation and policies stay readable); raw HTTP to Discord's user-only
  endpoints (`/auth/*`, relationships, private channels, user settings ..., accepting invites, OAuth2 authorization);
  raw HTTP to Discord with a non-bot credential scheme or a literal `Authorization` header; generated code that
  connects to Discord or uses self-bot libraries.

Before (measured on `40734d6` with the new tests copied in): a record owning and pre-authorizing the destinations
allowed `http.post` to `2captcha.com`, a request carrying `g-recaptcha-response`, `browser.navigate` to
`discord.com/login` and a raw POST to `/api/v9/auth/login`; in real Chromium the answer request of a challenge
embedded in an owned staging page reached the server. Tests: `tests/harness/auth/test_hx_platform_constraints.py`
(8; 7 fail on the old code), `tests/harness/env/test_hx_browser_constraints.py` (`-m browser`; fails on the old code:
1 answer POST received, 0 now).

Limits: the lists (solver services, challenge endpoints, Discord user endpoints) are finite; detection in code is
lexical; user-account automation is constrained for Discord, the only configured platform.

### IH-15 — message text is not retained on read permission alone (commit `204e9d4`)

Defect: with the adapter flag `discord_message_content` and the task permission `read_message_content` only
(`store_message_content` / `train_on_message_content` false), `DiscordExecutor` put message text into its
observations — which the learner stores and trains on — because it declared the retention as a read. It now declares
read + store + train, so the task permissions and the default rule `PL-DISCORD-MESSAGE-CONTENT` decide; under the
default policy no message text is retained until the operator records Discord's express permission by editing that
rule. Test: `tests/harness/skills/test_hx_data_use.py` (real authorization service, environment and Discord executor
against staging Discord; fails on the old code: the three texts leaked with status ok).

## New tests without a code change

* IH-13 `tests/harness/env/test_hx_no_reward.py`: every executor of `factories.all_executors` behind the real
  authorization service (staging site and staging Discord; browser and visual with `-m browser`) — no reward-like
  field in any result, observation, event, outcome, metric, resource state, capability or tool description, and none
  in the contract types.
* OP-07 `tests/harness/auth/test_hx_credentials_privileged.py` (`-m privileged`): the operator CLI stores into a
  launched run directory; the value is only in the vault and the verifier copy, unreadable for the learner and
  sandbox uids, never printed or taken from argv, known to the running authorization service by name only, refused
  as a leak, redacted, revocable.

## Checked, no change needed

IH-16, IH-17, CU-26, CU-35, CU-36, CU-37, CU-38: the named tests pass (privileged and browser marks included) and
the earlier G-sec probes for them hold (p09 12/0, p13 18/0, p14 21/0, p15 18/0 HOLDS/DEFECT). CU-25: the G-sec
defects M1–M4 and m6 no longer reproduce (p03 9/0, p04 15/0, p05 3/0, p06 4/0); G-env executor probes with the
real policy: probe_browser 19/0, probe_skills 10/0, probe_api_publish_modality 15/0, probe_programs 2/0,
probe_separated 3/1 (its DEFECT line is the probe's own staging site refused by the EA-9 local-address rule in the
separated deployment: enforcement working, the probe predates the rule). OP-04: claimed from the live evidence of 2026-11-27 (50/50), which predates two later changes
outside the live-checked path; a fresh live run is needed for fresh evidence (HX did not contact Discord).

## Left partial

CU-19: the procedure level (skills) records the semi-Markov fields; environment programs keep no per-step
boundaries and `cam_live.credit.smdp` is not reachable from the runtime. Change request HX-3.

# Round 2 — fixes of the G2-harness findings at 4d6044a (2026-11-28)

Report: `docs/verification/G2_harness.md`; probes `docs/verification/repro/g2harness/g01..g06` (run from
`/workspace/camlive`, exit 0 = every property HOLDS). Commits `863bdae` (code + tests), `efff208` (docstring). Final run `docs/evidence/test_results/HX_fix2.xml`:
`tests/harness/{compute,env,auth,skills}` + `tests/integration/test_credentials_membership.py`, compute first, `-m "browser or
privileged or not (gpu or slow or network)"`, as root — 616 passed, 0 failed, 4 deselected, 263 s at efff208. G-sec
probes p01–p15 re-run after the fixes: all 0 DEFECT.
Before → after (HOLDS / DEFECT): g01 19/11 → 30/0, g02 1/1 → 2/0, g03 4/3 → 7/0, g04 0/1 → 1/0, g05 23/1 → 24/0,
g06 1/1 → 2/0. The new policy tests (`tests/harness/auth/test_hx2_fixes.py`, 8) all fail on 4d6044a.

| Defect | Fix | Tests |
|---|---|---|
| D1 (major) Discord message text / member list readable through generic HTTP with the released bot token | `X-PLATFORM-ADAPTER` in `platform_constraints.py`: Discord's API, Gateway and attachment CDN only through `discord.*` operations, by every modality (also named in code); a platform credential (vault scope or known name) named by any other operation is refused, and `PolicyService.secret_get` releases it only to an action that is or contains an operation of its adapter. The adapter's own transport requests realise its operation (`check_primitive` passes the granted action) and stay covered | `test_hx2_fixes.py` (planning task + `tools/authorize_live.py` record), `test_ea_approvals.py` (raw request now refused) |
| D2 (major) no code sandbox in the separated deployment | `cam_live.harness.sandbox_host`: a root sandbox host the launcher starts with `exec` (it serves only the environment service's uid, SO_PEERCRED, `<run>/sock/sandbox.sock`); the environment service (still `cambrowser`) gets `RemoteSandboxExecutor` (`factories.code`, `$CAM_LIVE_SANDBOX_SOCKET`). Programs run in the unchanged `SandboxExecutor` (camsbx, no_new_privs, Landlock, seccomp, rlimits); every primitive it wants checked is answered by the environment's own `ExecContext.check` over the connection (authorization service, environment log), cancellation is forwarded, the host gets no secrets | `test_hx_sandbox_host_privileged.py` (`-m privileged`, launched like `cam-live run`) |
| D3 Discord developer portal | `/developers/*` except `/developers/docs*` is Discord's client application (user session) | `test_hx2_fixes.py` |
| D4 a site's own CAPTCHA | answer keys by substring (`captcha`, `recaptcha`, `human_check`, `turnstile` ...); the browser reports the field names of every submitted form (`form_fields`) and the constraint reads the accessible name of the element an input / click goes to ("CAPTCHA answer", "I am not a robot", "Verify you are human") | `test_hx2_fixes.py`, `tests/harness/env/test_hx2_browser.py` (Chromium: nothing submitted, an ordinary form still works) |
| D5 `resources.authorized_tools` | enforced at authorization for the action and every program step (`P-TOOL-NOT-AUTHORIZED`; names, globs, modality names); no list = interfaces alone, as before. Primitives an executor reports while realising an authorized action are not matched against it | `test_hx2_fixes.py` |
| D6 same-origin reads | the browser judges script reads (fetch / XHR / EventSource) per URL, not per origin; page-script reads of Discord (origin-level, client or API paths) are refused | `test_hx2_browser.py`, `test_hx2_fixes.py` |
| D7 public address | `PUBLIC_IPADDR` in the launcher's `ENV_PASSTHROUGH` | g05, `test_hx2_fixes.py` |
| HX-4 | `tests/harness/compute/conftest.py` Chromium fixture module-scoped | the final run puts `compute` first |
| HX-2 / D8 | the hard constraints are documented in the module docstring of `platform_constraints.py` and here (below); `docs/EA/SECURITY.md` is not HX's file (change request HX-2 stays open for EA) | — |

## The hard constraints (not policy data)

`src/cam_live/harness/auth/platform_constraints.py`, called from `Evaluator.node` for `authorize` (every step of a
program) and `check_primitive` (every primitive an executor reports). No policy file (`configs/authorization.yaml`),
live record, approval or owned / pre-authorized / interactive entry can permit what they refuse:

* `X-HUMAN-VERIFICATION` — CAPTCHA solvers (services, libraries, operation names), answering or interacting with
  a challenge (third-party widgets by host, challenge paths, answer keys by substring, element labels, submitted
  form fields), detection evasion (`navigator.webdriver`, stealth libraries), proxy / IP rotation.
* `X-USER-ACCOUNT` — the browser on Discord's client application, developer portal, API or invites (the profile
  may hold a user session), page scripts reading Discord, raw requests to user-only endpoints or with non-bot
  credentials, self-bot code.
* `X-PLATFORM-ADAPTER` — Discord data (API, Gateway, attachments) and the Discord credential only through the
  Discord adapter.

Limits: the lists are finite and need upkeep; code is judged lexically; the constraints know one integrated
platform (Discord).

# Round 3 — fixes of the G2-harness round-2 findings at 0a30cd8 (2026-11-28)

Commits `c9ae2d4` (R2-C1), `85101a2` (R2-M1, R2-m1, R2-m2, R2-m3). Probes (HOLDS / DEFECT) before → after:
g07 6/1 → see below, g08 26/13 → 39/0, g09 3/6 → 9/0, g10 0/1 → 1/0 (20 rounds); g01 30/0, g02 2/0, g03 7/0,
g04 1/0, g06 2/0 still hold. Every new test fails on 0a30cd8 except `test_reconciliation_never_hides_tampering`
(a guard that tampering stays detected).

* **R2-C1 (critical).** Separated services start as `python -I cam_live/harness/service_boot.py <roots> -m
  <module>` with cwd `/` (no cwd / script directory / PYTHON* variable on the import path; exactly the launcher's
  import roots in front), root services with HOME `/root` and a system PATH; the sandbox host checks the peer uid
  before it reads anything and times out a silent client. Test `test_hx_sandbox_host_privileged.py::
  test_root_services_never_import_what_a_service_uid_can_write` (on 0a30cd8 the planted module ran as uid 0 after
  the host restart) also checks that no directory on a launched service's import path is writable by a service uid.
  **g07** reports 6 HOLDS (incl. "never imported by the root sandbox host" with a real kill and restart) and then
  stops with `FileNotFoundError` at its section H4: it reads the marker file unconditionally, and the marker is
  only created when something imports the planted module — which no longer happens. With the one-line guard
  `os.path.exists(MARK) and` at line 181 (scratch copy, the probe itself is not HX's) it reports 7 HOLDS / 0 DEFECT.
  Not changed: the run directory stays group-writable for the learner (launcher layout); the sandbox host stays root
  for its lifetime (no pre-forked unprivileged helper) and trusts the environment uid to have asked the
  authorization service.
* **R2-M1.** Human-verification recognition made precise: answer keys only with `captcha`, provider token names or
  humanity phrases; element names only about proving one is human, CAPTCHA controls or answer fields named CAPTCHA.
  Honeypots, Antibot keys, quiz answers, turnstile counts, "Run security check", "CAPTCHA settings" pass.
* **R2-m1.** Adapter standing only for the adapter's own action; no credential release on behalf of a program;
  the attachment CDN hosts as a whole are Discord data.
* **R2-m2.** An allowed browser decision names the deny-listed paths of its host; the browser judges every
  subresource under them per URL (images, styles, scripts, prefetch), on the page's own origin too.
* **R2-m3.** `verify_audit_log` reads log and head under a shared lock (the g10 false alarm was a verification
  racing the relaunched service's start entries); at open a head left behind by a kill is advanced only when every
  later entry chains and carries a valid MAC.

# Round 4 — fixes of the G2-harness round-3 findings at 85101a2 (2026-11-28)

Commit `0fedb1a`. Probes (HOLDS / DEFECT) before → after: g11 9/2 → 11/0 (root, incl. the inotify race S4b),
g12 6/2 → 8/0, g13 10/6 → 16/0; g01 30/0, g08 39/0, g10 1/0 still hold. The new tests fail on 85101a2
(`tests/runtime/test_hx4_safe_writes.py` race test: the victim was overwritten with `go`; `test_hx4_fixes.py`: 5 of 5).

* **R3-M1 (major).** The class, not the instance: `cam_live.harness.safefs` — the parent directory is reached one
  component at a time with `O_PATH|O_DIRECTORY|O_NOFOLLOW` (a symlink is followed only where it lies in a directory
  only root can write), writes go through a random `O_CREAT|O_EXCL|O_NOFOLLOW` temporary created relative to the
  directory fd, owner and mode are set on the fd, the rename happens within the directory fd, a target that exists
  and is not a regular file is refused; reads are `O_NOFOLLOW`; directories are created and then opened without
  following. Used by `runtime.logs.atomic_write_text` / `atomic_write_json` / `read_json` (so by the supervisor's
  `preflight.json`, `learner.go`, pid and `run.json` writes and the operator CLI's `control.json`), the launcher's
  layout (`_mk`, key and policy copies), the vault, the credentials CLI (vault / verifier copy, ownership by fd or
  without following) and the sandbox host (its own directory; the socket is created 0660 through the umask and its
  group set without following). Not changed: the run directory stays group-writable for the learner (the learner
  writes its logs there); `runtime/control.py` still opens `control.json.lock` with `open(..., "a+")` (not HX's
  file; change request HX-6).
* **R3-m1 (minor).** At open, a head that does not describe the log (ahead of it, naming another entry, more than
  64 behind, missing although the log has entries, a bad MAC) is recorded as a MAC'd *discontinuity* in every later
  head and every verification reports it; in the separated deployment the launcher (root) also keeps the head of the
  last stop in `<run>/logs/audit.anchor` (root, 0600) and passes it to the next start (`--audit-anchor`), so a log
  cut back together with its head by the auth uid is reported too.
* **R3-m2 (minor).** A key with "captcha" is an answer only with an answer word (`captcha_answer`, `captcha_input`,
  `g-recaptcha-response`; `captcha` alone); "are you a robot?" only on a control that answers; typed fields named
  CAPTCHA unless they configure it; "type the code shown" only for an image. `enable_captcha`, `captcha_provider`,
  "Enable CAPTCHA on sign-up", "Get reCAPTCHA keys", an "Are you a bot?" link and the e-mail code field pass.

# Round 5 — fixes of the G2-harness round-4 findings at a5ab772 (2026-11-28)

Commits `8f6c615`, `5e57c7a` (the round-3 audit race test now bounds its writer: unbounded, it became quadratic under load and stalled a suite run). Final run `HX_fix5.xml`: harness suites (compute first, browser + privileged), credentials and `tests/runtime` — 881 passed, 0 failed, 4 deselected, 589 s at load ~45. Probes after the fix: **g15 2/0** (E1 export-checkpoint, E2 plan-report), **g16 3/0** (incl. the
FIFO read F3), g17 5/0. The privileged tests `tests/runtime/test_hx5_operator_cli_root.py` (4) all fail on a5ab772.

* **R4-C1 (critical).** Offline `export-checkpoint` (no runtime process): the checkpoint is chosen by *name* among the
  real directories `ckpt_<digits>` of `<run>/checkpoints` (`--checkpoint`, else the base name the state names, else
  the newest); `state.json` is never used as a path. The directory is copied entry by entry relative to directory
  fds into a private staging directory next to the destination — a symlink, FIFO, socket or device anywhere refuses
  the export —, a checkpoint with `MANIFEST.json` must match its hashes, the weights-only export reads the staged
  copy, and the destination is the operator's `--out` (normalised; outside the run directory; must not exist or be
  an empty directory this uid owns — never merged) or `<state root>/exports/<run>_<ckpt>`. A checkpoint without a
  MANIFEST (test agents) is exported with `manifest_verified` stating so.
* **R4-M1 (major).** `plan-report` writes through `safefs.write_atomic` (random O_EXCL temporary, rename within the
  directory fd, a planted symlink / non-regular target refused). The default location stays `<run>/plan_report.md`.
* **R4-m1 (minor).** `safefs.open_regular`: `O_NONBLOCK|O_NOFOLLOW`, type checked on the fd, `O_NONBLOCK` cleared; used
  by `read_nofollow`, the run-log readers (`iter_jsonl`, `iter_since`), the launcher's `_mk` and `chown_chmod_nofollow`.
* **R4-m2 (minor).** `credentials revoke` unlinks the verifier copy relative to the directory fd (`safefs.unlink`).
* **Sweep of root-run commands.** The supervisor's detached-process log (`safefs.open_append`), log tails, the task
  copy into the run directory, `journal --print` and the dashboard's journal read go through safefs; `status`,
  `resume`, `dashboard`, `journal` read run files through `read_json` / the run-log readers (safe since round 4/5);
  `approve` / `reject` / `stop` / `pause` go through the command channel (HX-6 applied by M) and the authorization
  service. Not swept: modules outside the allowed files (`runtime/journal.py` writes through safefs per G2;
  `runtime/doctor.py`, `eval/*`, `verify_completion` read repository files, not the run directory).
