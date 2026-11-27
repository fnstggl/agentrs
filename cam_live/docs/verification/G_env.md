# G-env — independent verification of the computer environment (EB) and skills / API / publishing (ES)

Verifier: workstream G, lens "env". I wrote none of the code below. Date 2026-11-27, branch `feat/cam-live`, machine
under heavy load from other sessions (load average about 30 on 48 cores) for the whole session.

**Scope.** EB: `harness/env.py`, `service.py`, `rpc.py`, `browser.py`, `perception.py`, `internal.py`,
`operator_requests.py`, `staging/`. ES: `harness/skills.py`, `modality.py`, `failures.py`, `http_api.py`,
`publishing.py`. Requirement ids: IH-01 IH-02 IH-12 CU-01 CU-02 CU-03 CU-04 CU-05 CU-11 CU-13 CU-14 CU-21 CU-28 CU-29
(EB) and IH-04 IH-06 CU-15 CU-16 CU-17 CU-18 CU-19 CU-31 CU-32 (ES).

**Left out on instruction.** The Discord adapter (`harness/discord*.py`, `harness/staging_discord/`) is being
extended by another agent and was not read, run or judged. Consequently there is **no verdict for IH-05** (the
Discord integration contract), the Discord-channel path of the publishing adapter was not exercised, and IH-06
(aggregate analytics, spec 12.5) is judged on its static-site half only.

**Verdicts** (`verdicts_env.json`): 23 requirements — **2 verified** (CU-04, CU-29), **21 partial**, 0 failed, 0 unverified.

## 1. What was run

All commands from `/workspace/camlive`. Probes are in `docs/verification/repro/env/`; each prints one line per
check (`HOLDS` = the property was confirmed by an attack that failed, `DEFECT` = the attack succeeded or the claim
was refuted, `NOTE`).

| What | Command | Result in my session |
|---|---|---|
| EB + ES default suites | `.venv/bin/python -m pytest cam_live/tests/harness/env cam_live/tests/harness/skills -q --rootdir=cam_live` | 149 passed, 23 skipped (env 43 + 23 skipped, skills 106) |
| EB with Chromium + root | `.venv/bin/python -m pytest cam_live/tests/harness/env -q --rootdir=cam_live -m "privileged or browser or not browser"` | 66 passed, 0 skipped, 66 s |
| Environment contract | `.venv/bin/python cam_live/docs/verification/repro/env/probe_env_core.py` | 7 hold, 1 defect |
| Browser vs. hostile pages | `.venv/bin/python cam_live/docs/verification/repro/env/probe_browser.py` | 8 hold, 11 defects, 2 notes |
| EB's workflow, real policy | `.venv/bin/python cam_live/docs/verification/repro/env/probe_browser_workflow.py` | 1 defect |
| Perception + internal executor | `.venv/bin/python cam_live/docs/verification/repro/env/probe_perception_internal.py` | 6 hold, 6 defects, 1 note |
| Skills | `.venv/bin/python cam_live/docs/verification/repro/env/probe_skills.py` | 7 hold, 3 defects |
| HTTP API, publishing, selector, retries | `.venv/bin/python cam_live/docs/verification/repro/env/probe_api_publish_modality.py` | 10 hold, 5 defects, 1 note |
| Programs | `.venv/bin/python cam_live/docs/verification/repro/env/probe_programs.py` | 1 defect, 1 note |
| Separated deployment (root) | `.venv/bin/python cam_live/docs/verification/repro/env/probe_separated.py` | 2 hold, 2 defects |
| EB latency re-measurement | `.venv/bin/python cam_live/docs/verification/repro/env/repro_eb_latency.py` | see §4 |

How the probes are built: the code under test is always the real one — `ComputerEnvironment`, `BrowserExecutor`
driving Playwright's Chromium 159, `Perception`, `InternalExecutor`, the ExecRPC service process, `HttpApiExecutor`,
`PublishingExecutor`, `SkillLibrary` / `SkillExecutor`, `ModalitySelector`. Authorization is the real
`harness.auth.PolicyService` with `default_policy.yaml` and a `staging` task unless a line says otherwise (EB's and
ES's own suites use small test policies, which is why several of the findings below never showed there). Targets are
the staging sites, the ES notes application, and local HTTP servers on 127.0.0.1 that count every request they
receive, so "was it sent" is measured. Nothing outside the machine was contacted.

Three deviations to state plainly:
* `probe_env_core.py` and `probe_separated.py` put a unix socket / a run directory in a short directory under `/tmp`
  (removed at exit) instead of the session scratch directory: AF_UNIX paths are limited to about 107 bytes and the
  scratch directory is not reachable by the service uids, which those two checks need.
* `probe_browser.py` performs one read-only `GET http://127.0.0.1:11111/capabilities` (the instance portal) through
  the browser and prints only the length of what came back. The POST to the portal's provisioning endpoint is
  **authorized only, never executed**.
* Running the ES suite overwrites `docs/evidence/ES/skills_vs_primitives.json` (the test writes it). I restored the
  committed file with `git checkout --` after each run; my copies are quoted in §4.

Where a probe needed a secret to actually reach its destination (scrubbing and binding checks), it wraps the real
broker handle in a 6-line shim that removes the `secret:` prefix — the step that is missing in the product (defect
M5). Those checks are marked "with the missing step emulated".

## 2. What held (attacks that failed)

* **No execution without authorization.** No decision, a forged grant, another action's grant, a relabelled grant
  and a `needs_approval` decision all end before any executor is called (0 requests received); a genuine grant runs
  once and is refused when replayed through a second environment with an empty journal; a grant issued before an
  emergency stop is refused after it. Log order per action: submitted → started → primitive_check → primitive_op →
  observation → result, with authorize / grant_use / primitive / record in the authorization service's audit.
* **Restart.** SIGKILL of the service during an external POST, restart on the same directory: the interrupted
  write is answered `failed/unknown` and not re-sent, the finished one returns its recorded result, the scheduled
  wake-up is unchanged.
* **ExecRPC uid check.** `camsbx` and `camverify` connect (they share the socket group) and are refused by uid for
  `__ping__`, `ping`, `get_capabilities`; `nobody` cannot connect; `camlearn` is served.
* **Local files.** `file:`, `chrome:`, `view-source:`, `javascript:` are refused by `PolicyService.authorize` and,
  independently, by `BrowserExecutor.validate`; a page's link, `window.open`, `location.href`, iframe and `fetch`
  to `file:///etc/passwd` return nothing. Non-loopback addresses (169.254.169.254, metadata.google.internal,
  10.0.0.1, 192.168.1.1, example.org) are denied in staging mode by the url envelope.
* **Untrusted content.** A page with fake system / operator / approval text, an instruction as a button's
  accessible name, and label/value pairs named `operator.request.granted` / `env.external_actions`: every
  observation is `trust=untrusted`, quantity names are prefixed `page.<path>.`, nothing changed in the authorization
  service, the forbidden destination stayed denied.
* **Stale references, files, tabs.** Five stale-reference cases fail with `stale_observation`; downloads land in
  `<workspace>/downloads` whatever `Content-Disposition` or `filename` say; a redirecting download URL is not
  followed; `/etc/passwd`, `../` and a symlink are refused for upload; a popup to a forbidden destination is not loaded.
* **Internal executor.** `wait` returns in 0.1 s with the wake-up exposed and delivered as exactly one event;
  notes persist; `internal.request_operator` records once, is answered through the real CLI, the answer arrives
  once, and neither asking nor the answer changes anything in the authorization service; 20 open requests at most.
* **Skills.** A denied destination, an interface the task does not permit, and an approval requirement each stop a
  skill at the offending step with nothing sent and without blaming the skill; a step added to the stored skill file
  by hand is checked and refused; stale → refused → repaired as version 2 → retired all work.
* **HTTP API / retries / selector.** 403 and 451 are sent once; a 429 with `Retry-After: 600` is not retried and the
  window is honoured; a redirect to a forbidden destination is checked before the hop; literal credentials are
  refused; a secret bound to one destination is not released for another nor forwarded across origins (emulated
  step); a 5 MB response becomes a 5.7 kB observation; retry rules cannot exist for hard classes; a refusal blocks
  every realisation of the same intent key, also after save / load.
* **Publishing (static site).** Eight path-escape variants, an unknown destination and an unpreviewed draft are
  refused; the file equals the previewed bytes; re-submission does not rewrite; with `require_approval: always`
  nothing is published before the operator approves.
* **No strategy, no placeholders.** No Discord-, community- or growth-specific term or branch in `env.py`,
  `browser.py`, `perception.py`, `internal.py`, `rpc.py`, `skills.py`, `modality.py`, `failures.py`; no
  `NotImplementedError` / stub bodies. (`http_api.py:468` has a `bot` authorization scheme and reads a 429 JSON
  body in one platform's shape — small, generic enough.)

## 3. Defects

Severity: **critical** = a mandatory control is bypassed or the machine is exposed; **major** = a stated guarantee is
false, a mandated capability does not work in the real configuration, or an executed action is misreported;
**minor** = narrower or already disclosed. "Owner" is my reading of whose code must change.

### Critical

**C1 — In staging mode the agent can reach the instance's own management API.** Owner EA policy data, reached
through EB / ES executors. `harness/auth/default_policy.yaml:43-46` makes every loopback port part of the staging
envelope, `owned` and `preauthorized`. The browser read `http://127.0.0.1:11111/capabilities` (3970-character
digest of the instance manifest returned); `POST http://127.0.0.1:11111/capabilities/provision` is authorized
`allow` without approval (not executed by me). Neither executor has a guard of its own
(`browser.py:49`, `http_api.py:408-430`). Same root cause as G-sec `p06_staging_loopback`.
Repro: `.venv/bin/python cam_live/docs/verification/repro/env/probe_browser.py` (BR-INTERNAL).

**C2 — A skill is not held to the write rate limit of its steps.** Owner EA accounting, exposed by ES skills.
One by one, 10 POSTs to a destination are allowed per hour (`default_policy.yaml:270`) and the 11th is refused; the
same POSTs as steps of `skill.run` were sent 28 times in 14 runs with no refusal — primitives inside a granted
action are counted only against `rate_limits.primitive` (600 per minute). This contradicts ES's "a skill never has
more permission than its steps issued one by one" (`skills.py:11-13`, check at `skills.py:1214`).
Repro: `.venv/bin/python cam_live/docs/verification/repro/env/probe_skills.py` (SK-RATE-LIMIT).

### Major

**M1 — Requests the browser sends without any check or report.** `browser.py:947-1001` (`_on_route`).
(a) A page's `new WebSocket('ws://<forbidden>')` connected and delivered a frame — the hook never sees WebSockets.
(b) A same-origin `fetch` that the site answers with a redirect to the forbidden destination reached it.
(c) A top-level 302 to a forbidden destination is sent first and judged afterwards (`browser.py:1255-1274`; content
is withheld, tab reset) — disclosed by EB, but the module docstring (`browser.py:13-15`) still says redirected
navigations and cross-origin subresources are checked. (d) Scripts' same-origin POSTs are never reported, against
`docs/EA/README.md` point 3. Direct cross-origin image / beacon / EventSource / form POST **are** blocked.
Repro: `probe_browser.py` (BR-SUBREQ-WS, BR-SUBREQ-REDIRECT, BR-REDIRECT, BR-SUBREQ-SAMEORIGIN).

**M2 — An executed click is reported `failed/unknown`.** Two independent causes:
`browser.py:1040` iterates `sess.tabs` across an `await` while a new tab is adopted ("dictionary changed size during
iteration") — 4, 5 and 6 of 8 clicks that open a tab in three runs (timing-dependent); and `perception.py:64` (`Decimal.scaleb` with a huge exponent)
raises on page text such as `Order total: 1e999999999999`, so the click that produced it — the site did receive the
POST — is reported failed (every run, when the text is present at the post-click observation), and a page carrying
such a string cannot be observed at all.
Repro: `probe_browser.py` (BR-POPUP-RACE); `probe_perception_internal.py` (PC-CRASH-PAGE, PC-CRASH-AFTER-CLICK).

**M3 — At-most-once is broken for concurrent duplicate submissions.** `env.py:489-507` decides "new action" under
the lock, releases it for authorization and validation, and `env.py:560-569` re-checks only the in-flight table. Two
submissions of one action id, each with its own valid grant from the real service: the POST was sent twice and
`external_actions` counted 2. (The probe adds 1.5 s to the first caller's grant check to open the window.)
Repro: `probe_env_core.py` (ENV-ONCE-2).

**M4 — A POST is re-sent automatically.** `http_api.py:441` retries every method under `failures.py:244`
(3 retries for transient faults): a POST the server had applied was sent 4 times after a 503 and 4 times after a
client timeout, with no idempotency key. The environment refuses to re-run an interrupted write; this executor
does the opposite. Repro: `probe_api_publish_modality.py` (HT-POST-RETRY).

**M5 — No executor can use a stored secret with the real authorization service.** `browser.py:390-391,1429` and
`http_api.py:347-350,512` take a secret *name*; the service refuses a bare name as "a credential field carries a
literal value" and requires `secret:<name>`; with that form the executors pass the prefixed string to the vault
("no secret named 'secret:site_pw'"). `browser.fill … secret` and `http.request … auth` are therefore unusable;
EB's and ES's tests use their own brokers. `HARNESS_ADAPTER_GUIDE.md` §2 rule 5 describes the non-working form.
Repro: `probe_browser.py` (BR-SECRET-USABLE); `probe_api_publish_modality.py` (HT-SECRET-REFERENCE).

**M6 — The environment as the runtime composes it.** `runtime/cli.py:393-397` and `runtime/loop_service.py:78`
start the environment with neither `--secrets` nor `--executor` (`harness/service.py:370-376`). Started exactly
that way under the launcher: modalities `internal` + `structured_browser` only (21 tools — no code, visual, api,
skill, publishing), `secret_scrubbing: false`, and `browser.navigate` fails with "Executable doesn't exist at
<run>/browser_profile/.cache/ms-playwright/…" because the service runs as `cambrowser` and Playwright's Chromium is
only under `/root`; `harness.service` has no executable-path option and no shared install /
`PLAYWRIGHT_BROWSERS_PATH` is configured. Repro: `probe_separated.py`.

**M7 — Three browser/environment capabilities are refused by the default policy.** (a) `default_policy.yaml:270`
counts every click / fill / select on an owned site as an external write, 10 per hour per destination (host, all
ports): EB's 9-interaction staging workflow completes once and the second run is refused at its second interaction.
(b) `browser.upload` of a file inside the workspace is always denied: the workspace (`env.py:150`) is not one of
the policy's `file_roots` (`default_policy.yaml:73`), and naming it in `task.resources.owned` does not help.
(c) Every `ActionSpec(operation="program", steps=…)` is denied: `program` (`env.py:47`) is not in the policy's tool
table. EB's tests for all three pass only with EB's test policy.
Repro: `probe_browser_workflow.py`; `probe_browser.py` (BR-UPLOAD-DEFAULT); `probe_programs.py`.

**M8 — Observation size is unbounded.** `Observation.text` is capped at 4000 characters, `Observation.data` is not:
`browser.py:149` (href), `:170` (form action), `:175` (all cells of a row), `:202` (title, URL) flow through
`perception.py:300-304`. Measured: 5.0 MB (one link), 3.0 MB (title), 1.8 MB (one table row) per observation;
four pages grew `events.jsonl` to 9.8 MB. Repro: `probe_perception_internal.py` (PC-SIZE).

**M9 — Skills and the modality selector are not part of the running system.** `skill.induce` through the
environment fails with "no successful execution recorded" because nothing calls
`SkillExecutor.observe_execution` (ES change request 4 is open), and apart from `runtime/doctor.py` no module
imports `harness.skills` or `harness.modality`. ES discloses both; the consequence is that CU-15…CU-18 and CU-31
are library capabilities today. Repro: `probe_skills.py` (SK-WIRING).

**M10 — A deny entry for a loopback address is bypassed by another name for it.** Owner EA.
`task.resources.deny = ["127.0.0.1:PORT"]`: `http://localhost:PORT/` and `http://localhost.:PORT/` were allowed and
the content returned (`127.1`, decimal, `0.0.0.0`, userinfo forms are denied). Repro: `probe_browser.py` (BR-DENY-ALIAS).

**M11 — A publication approval does not cover the text.** `publishing.py:316-331,378-384`: the approved action is
`publish.submit {draft_id, destination}`; the queue entry shows the operator `text: 'draft_…'`, not the content.
Today a swap is prevented only because `publish.update` is itself approval-gated; with ES's pending change request
ES-3 applied to the policy, the text was replaced while the approval was pending and the replaced text was
published under it. Repro: `probe_api_publish_modality.py` (PB-APPROVAL-BINDING, PB-APPROVAL-BINDING-ES3).

### Minor

* **m1** A secret echoed spaced-out, upper-cased or in two halves reaches the observation and `events.jsonl`
  (plain, reversed and base64 echoes are scrubbed). `env.py:93-111`. `probe_browser.py` (BR-SECRET-TRANSFORMED),
  `probe_api_publish_modality.py` (HT-SECRET-SCRUB). Emulated step.
* **m2** `1e999` becomes a non-finite `Quantity` (bare `Infinity` in JSON logs / RPC); `300K` → 300000 and `5B` → 5e9
  with the unit lost; a non-displayed `<dl>` is read (`browser.py:183`); `Perception.file` reads 8 MB without saying
  it cut. `perception.py:37-39,68,365-370`. `probe_perception_internal.py` (PC-NUMBERS-EDGE, PC-NONFINITE-PAGE, PC-HIDDEN).
* **m3** The operator's answer event is labelled `trust=operator` and embeds the learner-written request title
  verbatim. `internal.py:72,79`. `probe_perception_internal.py` (IN-OPERATOR-TRUST).
* **m4** `protected_reasons` is applied to step templates, not to bound steps: a protected name passed as a run-time
  argument is executed. `skills.py:1195`. `probe_skills.py` (SK-PROTECTED-GUARD).
* **m5** The selector's block is keyed by `parent_decision_id` or operation suffix, so the same operation under the
  next decision id, or the other realisation's `ActionSpec` as intent, is offered again; `update()` without an intent
  records nothing. `modality.py:220,346-351`. `probe_api_publish_modality.py` (MS-BLOCK-SCOPE).
* **m6** A rate-limit refusal ("retry in 3595 s") is labelled `missing_permission` (hard). `env.py:697-698`,
  `browser.py:260-267`. `probe_browser_workflow.py`.
* **m7** Tests overwrite committed evidence on every run: `tests/harness/skills/test_es_skills.py:562`,
  `test_es_modality.py:132`.
* **m8** A navigation a page starts after the action returned is denied (fail closed) — sites that redirect on a
  timer cannot proceed. `probe_browser.py` (BR-LATE-NAV, note).

## 4. Numbers reproduced

| Claim (status file) | Claimed | My run |
|---|---|---|
| EB tests, full | 66 passed, 57.33 s | 66 passed, 66.08 s |
| EB tests, default | 43 passed, 23 skipped | 43 passed, 23 skipped |
| ES tests | 106 passed | 106 passed |
| EB observation sizes (full / diff unchanged / diff after fill, characters) | 574 / 97 / 138 | 574 / 97 / 138 |
| EB latency, median ms: navigate / click+navigation / fill / inspect full / first action | 234.5 / 298.5 / 39.7 / 29.0 / 937.3 | 407.2 / 584.8 / 102.7 / 87.0 / 4434.4 |
| EB RPC ping, wait scheduling (ms) | 0.2, 5.6 | 0.1, 6.3 |
| ES modality mean and max abs. error of p(success) | 0.021668550514979505, 0.05332807700219078 | identical |
| ES skills vs. primitives: decisions, primitive ops per task, successes | 3 → 1, 3 → 4, 20 / 20 | identical |
| ES skill / primitive latency ratio | about 1.7 | 1.28 and 1.59 (two runs) |

The browser latencies did not reproduce: 1.7 to 4.7 times slower here. The machine was heavily loaded, so I read
this as "not reproduced under load", not as evidence that the recorded figures were wrong.

## 5. Status-file claims the evidence does not support

* **ES summary / CU-15, `skills.py:11-13`** — "a skill never has more permission than its steps issued one by one":
  false for rate limits (C2).
* **ES `skills.py:20-21`** — "induction and execution refuse any step that addresses the authorization system…":
  execution examines templates only (m4).
* **ES IH-04 / `http_api.py:6-8`, EB CU-14** — credentials by name fetched at the moment of use: not usable with the
  real authorization service (M5).
* **EB `browser.py:13-15`** — "every navigation request … (typed, clicked, scripted, redirected) … is checked per
  URL; cross-origin subresource requests are checked per origin": redirects are judged after they were sent
  (disclosed elsewhere), WebSockets and redirected subresources never (M1, not disclosed).
* **EB capabilities `at_most_once_by_action_id: true`, guide §1** — not under concurrent duplicate submission (M3).
* **EB CU-02 limitation** — "the default launch on this machine falls back to the service's own uid": true when
  the service is root; started by the launcher as `cambrowser` it does not fall back, it fails (M6).
* **EB IH-02 / CU-03 / CU-11 "implemented"** — correct against EB's test policy; with the project's own
  authorization service uploads, programs, stored-secret fills and more than 10 interactions per hour are refused (M7, M5).
* **EB measured latencies** — see §4.

## 6. Not checked

Live mode (no signed authorization record was created); a public page reaching loopback through M1 (needs an
external origin; Chromium's own local-network protection may apply); CDP beyond EB's loopback test; headed mode;
iframes; the visual and code executors (other lens); everything Discord.
