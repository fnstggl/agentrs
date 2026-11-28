# G2-harness — independent re-verification of the harness at the code freeze

Verifier: G2-harness (workstream G, lens `g2harness`). I wrote none of the code under test. Commit judged:
`4d6044ade68edfb92a2db16129a2809540496764` (`4d6044a`). Date 2026-11-27/28. Machine shared and loaded (load average
12-14 on 48 cores; another agent ran the whole suite in parallel).

**Scope.** 49 requirements: EA — IH-07 IH-08 IH-09 IH-10 IH-11 IH-14 IH-15 IH-16 IH-17 CU-25 CU-26 CU-27 CU-37
OP-09; EB — IH-01 IH-02 IH-12 IH-13 CU-01 CU-02 CU-03 CU-04 CU-05 CU-11 CU-13 CU-14 CU-21 CU-28 CU-29 CU-35 CU-36;
EC — CU-06 CU-07 CU-08 CU-09 CU-34 CU-38; ES — IH-04 IH-05 IH-06 CU-15 CU-16 CU-17 CU-18 CU-19 CU-31 CU-32 OP-04;
M — OP-07. Every id is claimed in `docs/status/EA|EB|EC|ES.json` (OP-07 by HX in EA.json). I read the spec Part 12 /
12A, the architecture, the four status files, `docs/HX.md`, `docs/change_requests/HX.md`, `docs/EA/FIXES.md`,
`docs/EB/FIXES.md`, the earlier reports `G_sec.md` / `G_env.md`, and the code of `harness/auth/**`, `browser.py`,
`http_api.py`, `discord.py`, `sandbox.py`, `localnet.py`, `platform_constraints.py`, `launcher.py`, the planning task
and the operator's launch tools. Verdicts: `verdicts_g2harness.json`. Probes: `repro/g2harness/` (run each from
`/workspace/camlive` as `.venv/bin/python cam_live/docs/verification/repro/g2harness/<probe>.py`; exit 0 = every
probed property held, 1 = a defect reproduced). No source file and no other agent's test was changed. Nothing
outside this machine was contacted (no Discord, no DNS lookups of public names).

**Which code ran.** The working tree's HEAD is 4d6044a but other agents have uncommitted edits outside the harness.
`git diff 4d6044a -- cam_live/src/cam_live/harness cam_live/tests/harness cam_live/src/cam_live/contracts.py
cam_live/src/cam_live/diagnostics cam_live/tests/integration/test_credentials_membership.py` is empty, and harness
code imports only `contracts` and `harness`. Policy-level and in-process probes ran against an exact extract
(`git archive 4d6044a | tar -x`, `G2H_SRC=<extract>/cam_live/src`). Tests and the probes that start services under
the service uids (g04, the earlier sec/env probes) ran from the working tree, because the service uids cannot
traverse the scratch directory (mode 0700); for those paths the code is byte-identical to 4d6044a. Test runs used
`-o addopts=""` so the uncommitted `--import-mode=importlib` of the working tree's pyproject did not apply.

**Verdicts (49):** 40 verified, 8 partial (IH-08, IH-14, CU-25, CU-37, OP-09, CU-01, CU-19, OP-04),
1 failed (IH-15), 0 unverified. New defects: 2 major (D1, D2), 3 moderate (D3-D5), 3 minor (D6-D8); every
defect reported by G-sec and G-env is fixed (their probes HOLD), SEP-BROWSER is a stale probe, HX-4 is open.

## 1. What I executed

| What | Command (from /workspace/camlive) | Result |
|---|---|---|
| Harness suites, HX's order, as root, browser + privileged | `.venv/bin/python -m pytest cam_live/tests/harness/env cam_live/tests/harness/auth cam_live/tests/harness/skills cam_live/tests/integration/test_credentials_membership.py cam_live/tests/harness/compute -q --rootdir=cam_live -o addopts="" -m "browser or privileged or not (gpu or slow or network)"` | **605 passed, 4 deselected, 0 failed, 391 s** (HX: 605 / 241 s). Every acceptance test named in the manifest or the status files for the 49 ids passed, except 4 `-m gpu` tests deselected here and run below |
| HX-4 ordering | same selection, `cam_live/tests/harness/compute` before `cam_live/tests/harness/env` | **2 failed**, 163 passed, 170 s: `test_browser_sessions.py::test_cdp_attach_to_a_separately_launched_chromium`, `test_browser_uid.py::test_chromium_runs_under_the_browser_uid_with_a_private_profile` ("Playwright Sync API inside the asyncio loop") |
| EC GPU tests | `.venv/bin/python -m pytest cam_live/tests/harness/compute/test_ec_grounding.py -q --rootdir=cam_live -o addopts="" -m gpu` | GoClick-Base x2 and Vocaela **passed** (968 s, mostly waiting); `test_mai_ui_2b_grounds_under_the_shared_gpu_lock` did not get the shared GPU lock: error at setup after pytest's 900 s timeout, and on a retry with `-o timeout=2500` skipped after the fixture's 1800 s wait (the evaluation queue held the lock) |
| G-sec probes p01-p15 | `.venv/bin/python cam_live/docs/verification/repro/sec/pNN_*.py` | **all 0 defects**: p01 25, p02 13, p03 9, p04 15, p05 3, p06 4, p07 13, p08 14, p09 12, p10 23, p11 16, p12 8, p13 18, p14 21, p15 18 holds |
| G-env probes | `.venv/bin/python cam_live/docs/verification/repro/env/<probe>.py` | env_core 8/0, browser 19/0 (2 notes), browser_workflow 1/0, perception_internal 12/0 (1 note), skills 10/0, api_publish_modality 15/0 (1 note), programs 2/0, **separated 3/1** (SEP-BROWSER) |
| EB's registered variant | `.venv/bin/python cam_live/docs/EB/repro/probe_separated_registered.py` | 4/0 (SEP-BROWSER holds: Chromium 159 as `cambrowser`) |
| My probes g01-g06 | `repro/g2harness/` | g01 19 hold / **11 defects**; g02 1 / **1**; g03 4 / **3**; g04 0 / **1**; g05 23 / **1**; g06 1 / **1** |

Probe outputs of my runs are in my scratch directory; every probe prints one line per property and can be re-run.

## 2. The earlier defects — re-checked

Every G-sec defect (M1 step-batching of limits, M2 host-granular approval coverage, M3 paraphrased 12.6 classes,
M4 staging loopback / control plane, m5 file-flag stop, m6 URL parser differential, m7 emptied audit log) and every
G-env defect (C1 instance API, C2 skill write rate, M1 unchecked browser requests, M2 misreported clicks, M3
concurrent duplicates, M4 POST re-send, M5 secret references, M6 runtime composition, M7 default policy refusing
workflows / uploads / programs, M8 unbounded observations, M9 skills and selector not wired, M10 deny-alias, M11
publication approval not binding the text, m1-m7) now reports HOLDS in the verifier's own unchanged probe (table
above), and the regression tests named in `docs/EA/FIXES.md` / `docs/EB/FIXES.md` ran and passed in my run.
Remaining non-HOLDS lines: PC-NUMBERS-EDGE, BR-LATE-NAV, BR-UID, PB-APPROVAL-DEFAULT are NOTE by design of the probe
(no HOLDS outcome); SEP-BROWSER see below. p07's member-events line now holds (master decision: member events need
`member_list`).

**SEP-BROWSER (probe_separated) — I agree with HX: a stale probe, enforcement working.** The probe starts its
staging site in its own root process and never registers it. The denial is `P-LOCAL-ADDRESS` ("url:127.0.0.1:<port>
... is an address of this machine ... and not a registered local service"), taken by the authorization service
before the environment executes anything — the local-address rule of the EA fix applied as designed (in the
separated deployment the authorization service does not own the site's socket). EB's copy with the single added
`register_local_service` line holds all four checks (service as `cambrowser`, six modalities, scrubbing on,
Chromium 159 loads the page). Consequence worth stating: a staging run through `cam-live run` needs its staging
sites registered by the operator (`opctl local-service add`); nothing in `runtime/` or `harness/service.py`
registers them (`grep register_local_service`: only tests, `tools/discord_live_check.py`, `tools/planning_rehearsal.py`).

**HX-4 — reproduced, a test-isolation defect, not a product defect; still open.** Root cause (more precise than
HX's note): `tests/harness/compute/conftest.py:166-176` is a *session-scoped* fixture that keeps
`sync_playwright()` open in the main thread for the rest of the session, so asyncio's running loop stays set there;
any later test that calls `sync_playwright()` in the main thread (`test_browser_sessions.py:83`,
`test_browser_uid.py:48`) is refused by Playwright. The product is not affected: `browser.py:1072-1087` already
runs its one sync-Playwright probe in a separate thread. Fix (owner EC): module scope or a thread for that fixture.
Until then a whole-harness run must order `env` before `compute` (as HX's final run did).

## 3. Defects found (severity, file:line, reproduction)

### Major

**D1 — Discord message text and the member list reach the learner through the generic HTTP tools with the bot
credential, without any data permission (IH-15, IH-14 "unauthorized data collection", CU-25).**
Under the real planning task `tasks/discord_100_planning.yaml` (interfaces include `api`, line 69; no
message-content or member-list permission) and the live record the operator's own tool builds for it
(`tools/authorize_live.py build_record` with `--read 'https://*' --read 'http://*'`, as `tools/launch_live.sh:7`
documents), `http.get https://discord.com/api/v10/channels/<id>/messages` and
`.../guilds/<guild>/members?limit=1000` with `auth: {scheme: bot, secret: secret:discord_bot_token}` are **allowed**
(`P-ALLOW`) and the broker **releases the bot token** for them, while `discord.read_messages` /
`discord.list_members` are refused (`P-DATA-USE`, `X-DATA-COLLECTION`). Causes: `T-NET-REQUEST`
(`default_policy.yaml:208-214`) carries no data class, so neither `data_permissions` nor
`PL-DISCORD-MESSAGE-CONTENT` is consulted for a raw request to Discord; the hard constraints refuse only user-only
endpoints and non-bot schemes (`platform_constraints.py:291`); the default binding of the bot token is the whole
host `discord.com` (`harness/credentials.py:35`); and the HTTP executor puts the response body (text and parsed
JSON) into the observation (`http_api.py:658-664`), i.e. into the experience stream. The live check shows the
application has the MESSAGE_CONTENT and GUILD_MEMBERS intents enabled (`docs/evidence/discord/live_check_20261127.json`,
step 1), so this is not theoretical. Raw writes to Discord are refused (`P-WRITE-NON-OWNED`), so the gap is reads.
This is also the modality switch spec 12A.7 forbids ("never interpret an access restriction as a reason to switch
execution modalities"). HX's IH-15 fix (204e9d4) itself holds — the adapter path is closed.
Repro: `g03_planning_task_raw_api.py` (3 defects), `g01_policy_bypass.py` section B.

**D2 — In the deployment `cam-live run` starts, the generated-code sandbox does not run at all (CU-37, CU-01).**
The environment service runs as `cambrowser` (non-root), so `sandbox.probe()` finds `uid_mode: same-uid` and
`sandbox.py:1153` makes every code tool unavailable: `sandbox.python` returns "unavailable: not root: generated
code would run under the harness uid". The separation the spec asks for (browser process, generated-code sandbox,
credential service, verifier, learner as separate processes) therefore exists either in-process as root (sandbox
drops to `camsbx`, but learner and browser share a process) or in the separated deployment without a code
sandbox. EB lists this in `not_done` ("the sandbox cannot drop to camsbx while the executor runs as cambrowser");
EA's CU-37 claim and limitations do not. Repro (root): `g04_separated_sandbox.py`.

### Moderate

**D3 — OP-09: the browser may open Discord's developer portal, an application that runs with a user's session.**
`DISCORD_CLIENT_PATHS` (`platform_constraints.py:131-136`) lists `/login`, `/channels/*`, `/app` ... but not
`/developers/applications*` or `/developers/teams*`. With the public web readable (planning record),
`browser.navigate https://discord.com/developers/applications` (also `.../applications/<app id>/bot`, the page of
the bot token, and `/developers/teams`) is `allow`, although the module docstring says the browser never opens
Discord's client application "with the browser's session" and HX's own test treats that page as a user-session
page (it refuses typing there). Repro: `g01_policy_bypass.py` section A.

**D4 — OP-09: a site's own CAPTCHA can be answered through the structured browser.** On an owned, pre-authorized
staging site (real Chromium, real PolicyService, default policy) the agent sees `textbox 'CAPTCHA answer'` next to
an image labelled as a CAPTCHA, `browser.fill` types the answer and `browser.click` submits it: the server received
`email=a%40owned.example&captcha_input=XK4P9`. The hard constraints recognise challenges only by third-party hosts,
a few fixed paths and an exact list of answer keys (`TOKEN_FIELDS`, `platform_constraints.py:91-97`, compared as
whole keys): `captcha_input`, `recaptcha_response_field` (the classic reCAPTCHA field), `captchaResponseToken`
and a form body `captcha_input=...` all pass `http.post` as `allow`. A self-hosted "I am not a robot" checkbox was
refused, but by the deceptive-identity content rule (`C-HUMAN-CLAIM`), not by the constraint. Backstop: in live
mode input to a *non-owned* page needs approval (`default_policy.yaml:72`) — but OP-09 is claimed as a hard
constraint that no approval can lift. Repro: `g02_browser_captcha.py` (Chromium), `g01_policy_bypass.py` section C.

**D5 — IH-08: the task's explicit tool list is not enforced.** A live task must state
`resources.authorized_tools` (`runtime/tasks.py:497-499`); the planning task lists `browser.*, sandbox.*,
discord.*, internal.*, wait`. The authorization layer derives permitted operations from `interfaces` only
(`policy.py:331-332`), so `api` (needed for `discord.*`) also admits `http.*` / `api.*`: `http.get` of a public page
is `allow` under the planning task. This is what opens D1. Repro: `g03_planning_task_raw_api.py` (last check).

### Minor

**D6 — OP-09: a Discord page's same-origin reads are judged per origin.** `browser.py:1511-1524` checks GET / HEAD
subresources once per origin with URL `origin + "/"`, and not at all for the origin of a page the action navigated
to (`sess.origins_ok[origin] = True`, `browser.py:1629`, short-circuits `_read_allowed`). HX's test asserts that
`browser.request GET https://discord.com/api/v9/users/@me` (xhr) is refused (`test_hx_platform_constraints.py:136`),
but the real browser never reports that URL: under a Discord documentation page the read is not checked, and from
another page it is reported as `https://discord.com/`, which is `allow`. So Discord's own scripts on an allowed page could read its API with a session the profile holds.
Mitigated today by the absence of a Discord session in the profile (the one login attempt stopped at hCaptcha,
ledger session 1). The mechanism shown in real Chromium on a staging site (Discord cannot be contacted): with
`<site>/private/*` on the task's deny list, navigating there is refused, but a script of the allowed page
`<site>/public` fetched `/private/data` and the page displayed its content — a path-level rule is not applied to
same-origin reads. Repro: `g01_policy_bypass.py` section A (NOTE + DEFECT line), `g06_browser_same_origin_reads.py`.

**D7 — local-address rule: the launched authorization service does not know the instance's public address.**
`localnet.py:104` adds `$PUBLIC_IPADDR` to this machine's addresses, but the launcher's `ENV_PASSTHROUGH`
(`auth/launcher.py:63-66`) does not pass it, so in the separated deployment `local_kind(<public address>)` is `''`
(in-process it is denied). Impact is limited: this instance's public ports go through the Caddy token edge.
Every other spelling of this machine I tried (22: IPv4-mapped / NAT64 / IPv4-translated IPv6, octal, hex, padded,
percent-encoded, full-width, `localhost.localdomain`, `ip6-localhost`, link-local with zone, metadata, user-info,
backslash, `0`, `::`, 127/8, the container's host name) is refused. Repro: `g05_local_address.py`.

**D8 — documentation.** Change request HX-2 is not applied: `docs/EA/SECURITY.md` has no section on the hard
platform constraints, which the policy file cannot change. (HX-1, HX-3 are also open; HX-4 above.)

Disclosed by the owners and confirmed, not counted as new: the HTTP executor does not pin or report resolved
addresses (DNS rebinding judged for the browser only; EB `not_done`, EA-10a); content detectors and secret
scrubbing are lexical / list-based; staging sites in the separated deployment must be registered; CU-19's semi-Markov
module is not reachable from the runtime (no caller of `credit.smdp.attach_trace`).

## 4. Claims the evidence does not support

* EA.json IH-15: "Enforced at the source: what the Discord adapter does not put into an observation cannot be
  stored or trained on" — true for the adapter; the generic HTTP executor puts the same data into observations (D1).
* EA.json CU-37 "implemented" without the limitation that the code sandbox is unavailable in the separated
  deployment (D2).
* EA.json OP-09 / platform_constraints docstring: "the browser ... opening its client application ... or its API
  with the browser's session" is refused — not for the developer portal (D3) or same-origin page reads (D6);
  "no ... approval ... can permit them" — a site's own CAPTCHA is outside the constraint (D4).
* HX.md / EA.json IH-14 "the 9 other classes were already refused" — "unauthorized data collection" is refused for
  `discord.*` tools but not for raw reads of the same Discord data (D1).

## 5. Per requirement: basis of the verdict

`v` verified, `p` partial, `f` failed. "Tests" = every acceptance test named in the manifest and the status file
for the id ran fresh and passed in my run of 4d6044a (§1).

| id | verdict | basis |
|---|---|---|
| IH-07 | v | tests 7/7; p10 23/0; BR-SECRET-BASIC / -TRANSFORMED / -USABLE, HT-SECRET-* hold; SEP-COMPOSITION scrubbing on. The released token in D1 is used in a header and never observed |
| IH-08 | p | tool table, modality and interface checks hold (tests 14/14, p01, p15); the task's `authorized_tools` is not enforced (D5) |
| IH-09 | v | tests 12/12; p04 15/0 (exact coverage, page-bound approvals); PB-APPROVAL-BINDING(-ES3) hold |
| IH-10 | v | tests 19/19; p03 9/0; SK-RATE-LIMIT, PG-RATE-LIMIT hold |
| IH-11 | v | tests 9/9; p02 13/0 (API, opctl and file-flag stop void grants; reset) ; ENV-STOP holds |
| IH-14 | p | p07 13/0 and tests 11/11; data collection via raw reads (D1) and circumvention via a site's own CAPTCHA / unlisted answer keys (D4) not refused |
| IH-15 | f | the adapter path holds (test_hx_data_use, my run); under the real planning task + record, message text and the member list are readable with the released bot token through `http.get` / `api.get` and enter observations (D1, g03) |
| IH-16 | v | tests 7/7 incl. privileged; p09 12/0; p13 (learner uid cannot read / delete the log); g01 audit chain verifies after revocations |
| IH-17 | v | tests 4/4; g01 section D: destination revocation under 6 spellings, in-flight stop, restart persistence, secret, task, live record, unknown scope refused, audited |
| CU-25 | p | one evaluator for every modality, all sec / env executor probes 0 defects, g05 23/1; data-use classes are enforced on the Discord modality but not on the api modality (D1); public address unknown to the launched service (D7) |
| CU-26 | v | tests 7/7; p15 18/0; BR-INJECT, BR-TRUST, IN-OPERATOR-TRUST hold |
| CU-27 | v | tests 7/7 incl. privileged; p10 23/0; p13 18/0 |
| CU-37 | p | learner / browser / auth / verifier uids separated (p13 18/0, SEP-UIDS, privileged tests); the code sandbox cannot run in the separated deployment (D2) |
| OP-09 | p | solver services, third-party challenge answers (HX Chromium test), user-only endpoints, non-bot schemes, self-bot code refused (tests 9/9); D3, D4, D6 |
| IH-01 | v | tests 16/16; probe_env_core 8/0 |
| IH-02 | v | tests 12/12; probe_browser 19/0; BR-WORKFLOW-REAL-POLICY holds |
| IH-12 | v | guide exists and documents `secret:<name>`, `check_primitive` per primitive, `peer_ip`; tests 5/5 |
| IH-13 | v | tests 4/4 (every executor behind the real service, browser / visual included); no reward field in harness code (grep) |
| CU-01 | p | in-process (root) the environment provides browser, files, Python / shell, jobs (tests 7/7, p11); in the deployment `cam-live run` starts, Python / shell execution is unavailable (D2) |
| CU-02 | v | tests 6/6; SEP-BROWSER (registered) runs Chromium 159 as `cambrowser` |
| CU-03 | v | tests 12/12; probe_browser |
| CU-04 | v | test passes when `env` runs first; fails only after compute's session fixture (HX-4, test isolation) |
| CU-05 | v | tests 9/9; BR-STALE holds |
| CU-11 | v | tests 6/6; PG-PROGRAMS-REAL-POLICY holds |
| CU-13 | v | tests 8/8; six modalities composed (SEP-COMPOSITION); D2 removes the code modality in the separated deployment (recorded under CU-01 / CU-37) |
| CU-14 | v | tests 4/4 |
| CU-21 | v | tests 9/9 |
| CU-28 | v | tests 4/4; ENV-ORDER holds |
| CU-29 | v | tests 4/4; ENV-RESTART holds |
| CU-35 | v | the 8 named tests (structured HTML, canvas controls, dynamic changes, multi-step, unfamiliar variants + interface change, repeated tasks with skills, delayed consequences) pass with `-m browser`; scripted, not learner-driven (disclosed) |
| CU-36 | v | tests 5/5; ENV-ONCE-1/-2 hold; Discord sends use an enforced nonce |
| CU-06 | v | tests 5/5 |
| CU-07 | v | non-GPU tests 5/5; both `-m gpu` GoClick-Base tests pass (isolated child process; in-process frozen, memory released) |
| CU-08 | v | the named non-GPU tests and the `-m gpu` Vocaela test pass in my runs; the MAI-UI-2B test could not take the shared GPU lock in two attempts (§1), so for MAI-UI-2B itself the basis is EC_gpu.xml and G-sec's independent `-m gpu` run (4 passed), both on grounding code unchanged since 6851826; no failure observed |
| CU-09 | v | tests 16/16; p14 21/0 |
| CU-34 | v | tests 4/4; benchmark evidence present; numbers reproduced on a subset by G-sec (`repro/sec/bench_spot.json`); I did not re-run the benchmark |
| CU-38 | v | tests 3/3; p14 21/0 (grounder answers are untrusted numbers, validated, authorized per gesture) |
| IH-04 | v | tests 11/11; PB-FLOW, PB-APPROVAL-BINDING-ES3 hold |
| IH-05 | v | tests 50/50; adapter sends only `Authorization: Bot`; no user-token path (the D1 gap is a policy / modality gap, recorded under IH-15 / CU-25) |
| IH-06 | v | tests 5/5; PB-ANALYTICS holds |
| CU-15 | v | tests 9/9; SK-INDUCE-RUN, SK-WIRING hold |
| CU-16 | v | tests 4/4; SK-SMUGGLED-STEP, SK-PROTECTED-GUARD hold |
| CU-17 | v | tests 5/5 |
| CU-18 | v | tests 5/5; SK-STALE-REPAIR, SK-RETIRED hold |
| CU-19 | p | procedure level recorded by `skill.run` (tests 2/2); programs keep no per-step boundaries; `credit.smdp.attach_trace` has no caller (grep) — as claimed |
| CU-31 | v | tests 6/6; MS-BLOCK-SCOPE, MS-NO-SWITCH hold |
| CU-32 | v | tests 13/13; FT-RETRY-BOUNDS, HT-HARD-NEVER-RETRIED, HT-POST-RETRY hold |
| OP-04 | p | judged from `docs/evidence/discord/` only (no contact): 50/50 steps ok on 2026-11-27 05:53-06:00 UTC, bot identity `bot: true`, designated guild only (`other_guilds` 0), guild name restored, temporary channel deleted, token-free. It ran with policy `default-1+discord-live-check` (current `default-2`) and before a1cdc0e / 204e9d4; the current code has not been run against the real API; the post-run restoration check is stated in the README but is not a recorded step. Staging acceptance tests 2/2 pass |
| OP-07 | v | tests 4/4 incl. HX's privileged test (vault + verifier copy only, unreadable for learner / sandbox uids, never argv / printed, refused as a leak, revocable) |

## 6. Not checked

A live run against Discord or any public service (forbidden here); whether Discord routes path spellings such as
`/LOGIN` or `%40me` like their canonical forms (g01 prints the decisions as NOTE); DNS-rebinding through the HTTP
executor (needs a public DNS name); headed browser mode; the grounding benchmark itself (only its tests and the
earlier spot check); the MAI-UI-2B GPU test (the shared GPU lock was never free for it); the learner's own
behaviour under prompt injection (outside the harness).

---

# Round 2 — re-verification at 0a30cd8 (2026-11-28)

Commit judged: `0a30cd8f23184fcded019f481bc885ac7defd2b7` (harness last changed at `efff208`; HX's fixes in `863bdae`,
`efff208`). Claims read: `docs/HX.md` "Round 2", `docs/status/EA|EB|EC|ES.json`. All 49 ids were re-judged at
0a30cd8 (`verdicts_g2harness.json` updated in place, `supersedes: ["all"]`). The working tree's harness, its tests,
contracts, tasks and tools are byte-identical to 0a30cd8 (`git diff --stat 0a30cd8 -- ...` empty); policy-level and
in-process probes ran against `git archive 0a30cd8` (`G2H_SRC`), the root probes (g04, g07, g10) and the suites from
the working tree. Since 0a30cd8 the `--import-mode=importlib` addopts and the new `tests/conftest.py` are committed,
so the suites ran with the repository's pytest configuration.

**Verdicts (49):** 40 verified, 8 partial (IH-07, IH-16, CU-25, CU-27, OP-07, OP-09, CU-19, OP-04), 1 failed (CU-37).
Compared with round 1: IH-15 failed -> verified, IH-08 / IH-14 / CU-01 partial -> verified; CU-37 partial -> failed,
IH-07 / IH-16 / CU-27 / OP-07 verified -> partial (all through the new critical R2-C1).

## R2.1 What I executed

| What | Result |
|---|---|
| Harness suites as root, browser + privileged, **compute first** (`.venv/bin/python -m pytest cam_live/tests/harness/compute cam_live/tests/harness/env cam_live/tests/harness/auth cam_live/tests/harness/skills cam_live/tests/integration/test_credentials_membership.py -q --rootdir=cam_live -m "browser or privileged or not (gpu or slow or network)"`) | **616 passed, 0 failed, 4 deselected, 257 s** (HX: 616 / 263 s) — HX-4 fixed |
| G-sec p01-p15 | all 0 defects (25, 13, 9, 15, 3, 4, 13, 14, 12, 23, 16, 8, 18, 21, 18 holds) |
| G-env probes | env_core 8/0, browser 19/0, browser_workflow 1/0, perception_internal 12/0, skills 10/0, api_publish_modality 15/0, programs 2/0, separated 3/1 (SEP-BROWSER, stale as in round 1); EB's registered variant 4/0 |
| My round-1 probes g01-g06 | **g01 30/0, g02 2/0, g03 7/0, g04 1/0, g05 24/0, g06 2/0** (HX's figures reproduce) |
| New probes g07-g10 | g07 6 / **1** (root), g08 26 / **13**, g09 3 / **6** (Chromium), g10 0 / **1** (root, 20 rounds) |
| HX's flaky launcher test | `test_ea_launcher.py::test_authorization_service_is_restarted_with_its_state` 30/30 pass; `test_ea_launcher.py` + `test_ea_privileged.py` + `test_hx_sandbox_host_privileged.py` x8: **1 of 8 failed** (`test_ea_privileged.py:398`, `verify_audit()["ok"]` false) — reproduced and explained, R2-m3 |

## R2.2 Round-1 defects re-checked

| Defect | Re-check |
|---|---|
| D1 Discord data via generic HTTP with the bot token | **fixed**: g03 7/0 (planning task + record: message history and member list refused, token not released), g01 B; g08 A: 20 spellings (case, trailing dot, www / ptb / discordapp, port, user-info, `%61pi`, `//`, `/./`, `/x/../`, `../v9`, full-width host, encoded id, Gateway `wss:`, `gateway-*.discord.gg`, attachment CDN, webhooks) refused by `X-PLATFORM-ADAPTER`; the credential named as `secret:discord_bot_token`, bare name or with a leading space, and inside generated-code arguments, refused. Residual R2-m1 below |
| D2 no code sandbox in the separated deployment | **fixed functionally** (g04 1/0: `sandbox.python` runs as camsbx); but the new root sandbox host opens R2-C1 (critical) |
| D3 developer portal | **fixed** (g01 A) |
| D4 a site's own CAPTCHA, exact answer keys | **fixed** (g02 2/0 in Chromium, g01 C) — but the structural recognition over-refuses (R2-M1) |
| D5 authorized_tools | **fixed** (g03); g08 D: none of the 57 composed tools that match the planning task's list is refused as `P-TOOL-NOT-AUTHORIZED`; the 22 refused by design are http.*, publish.*, skill.*, visual.* (not listed by the task; skill and visual were already outside its interfaces) |
| D6 same-origin reads | **fixed for fetch / XHR / EventSource and for Discord** (g06 2/0, g01 A); residual R2-m2 below |
| D7 public address in the launched service | **fixed** (g05 24/0) |
| D8 / HX-2 documentation | addressed in the module docstring of `platform_constraints.py` and `docs/HX.md`; `docs/EA/SECURITY.md` still has no section (EA's file) |
| HX-4 | **fixed** (module-scoped fixture; compute-first suite 616/0) |

## R2.3 New defects

**R2-C1 (critical) — the learner uid can run code as root through the new sandbox host (CU-37; also IH-07, IH-16,
CU-27, OP-07 in the launched deployment).** The launcher starts every service with `cwd=self.run_dir`
(`auth/launcher.py` `_spawn`), and the no_new_privs trampoline `python -I -c … os.execv(exe, [exe, "-m", module, …])`
runs the real interpreter *without* `-I`, so `python -m cam_live.harness.sandbox_host` puts the run directory first on
`sys.path`. `prepare_run_dir` creates the run directory owner root, **group = the learner's primary group, mode
1771** (`launcher.py:230`, `:237`), so the learner uid may create files in it. The probe, as `camlearn`, writes
`<run>/argparse.py` (a module that appends `<euid> <argv>` to a marker and then hands over to the real `argparse`);
at the host's next start — a crash restart by the supervisor, or `launcher.resume()` after an emergency stop
(`stop_on_estop` halts the host) — the root host imports it: marker `0 --run-dir /tmp/g2hsh…`. Non-root services
are not affected (as "others" they have no read permission on the run directory, so Python's path finder sees
nothing there; the authorization service restarted after the same plant and did not import it), which is why this
did not exist before round 2: the sandbox host is the first root service. Fix options: start services with
`python -I` / `-P` (or `PYTHONSAFEPATH=1`) and `cwd="/"`; do not make the run directory writable by the learner's
group. Also observed (no privilege gain): the host verifies no grant — the browser uid can connect directly and run
programs as camsbx with every check answered by itself; the sandbox's own limits still hold (camsbx, no access to
/root, vault, auth dir, browser profile; the egress broker refuses 127.0.0.1, 169.254.169.254 and localhost even
when the client answers "allow"). The host refuses camlearn, camsbx and camverify (SO_PEERCRED), but reads one frame
before the uid check with no read timeout (same pattern as `rpc.py`; a denial-of-service lever only). The root host's
`HOME` is `<run>/exec` (the browser uid's directory); user site-packages are disabled in this venv, so no import path
follows from it. Repro (root): `g07_sandbox_host.py`.

**R2-M1 (moderate) — the structural human-verification recognition refuses ordinary operations, and as a hard
constraint nothing can lift it (OP-09; spec 12.6 "these boundaries must not prevent it from learning how to create
genuine value").** `TOKEN_KEY_PARTS` substrings and `_HUMAN_TEST_TEXT` (`platform_constraints.py`, D4 fix) and the
browser's `form_fields` (names, matched whatever the value): `http.post` bodies with `botcheck` (Web3Forms honeypot),
`antibot_key` (Drupal Antibot), `challenge_answer` (a quiz), `turnstile_count`, and clicks on "Run security check",
"Security checkup", "Bot protection settings", "Turnstile counts" are all `X-HUMAN-VERIFICATION` (g08 C, 9 defects).
In real Chromium on an owned, pre-authorized staging site: a newsletter form carrying Drupal Antibot's empty hidden
`antibot_key` and a community quiz answer (`challenge_answer`) were refused; a Web3Forms-style unticked honeypot
checkbox passed only because Chromium does not submit unticked checkboxes (g09 part 1). Repro: `g08_round2_policy.py`
section C, `g09_round2_browser.py` part 1.

**R2-m1 (minor) — a discord.* step lends the adapter's standing to the rest of a program (IH-15, CU-25).**
`check_primitive` passes the granted action as `parent`, and `findings(..., realises=…)` treats every primitive of a
program that contains any `discord.*` step as the adapter's: a redirect hop of the program's `http.get` step to
`https://discord.com/api/v10/channels/<id>/messages` or to the attachment CDN is `allow`, while the same hop of a lone
`http.get` is refused; `secret_get` releases the bot token on behalf of the whole program (no step named). No other
step may name the credential at authorization (steps are judged without `parent`), and no executor attaches it on a
redirect, so API data stays unreadable; signed attachment URLs need no token. Also: a page's images from
`cdn.discordapp.com` pass, because image reads are judged per origin (`https://cdn.discordapp.com/`), so
`X-PLATFORM-ADAPTER` never sees the attachment path in the browser. Repro: `g08_round2_policy.py` sections B, B'.

**R2-m2 (minor) — the D6 fix covers fetch / XHR / EventSource only (CU-25).** With `<site>/private/*` on the task's
deny list, the allowed page `<site>/public` still requested **and read** the denied path through `<img>` + a
same-origin canvas (`getImageData`), `<link rel=stylesheet>` + CSSOM (`PRIVATE-CSS-TEXT`), `<script src>`
(`window.__leak='PRIVATE-JS'`), and requested it through `<link rel=prefetch>`; a Worker's `fetch` was refused.
Repro: `g09_round2_browser.py` part 2.

**R2-m3 (minor, pre-existing; HX's flaky test) — an ordinary stop can leave the audit log failing verification
(IH-16).** `AuditLog.append` writes and fsyncs the line, then replaces the head file (`audit.py:131-136`); a stop or
kill between the two leaves the head one entry behind, and nothing reconciles it when the service starts again.
After emergency stop -> launcher stop -> relaunch on the same run directory, `verify_audit` failed in 2 of 20 rounds:
"head file says 11 entries ending in …, log has 12 ending in … (truncated or extended outside the service)" — a
false tamper alarm. The same caused the 1-of-8 failure of `test_ea_privileged.py:398`. Repro (root):
`g10_audit_after_stop.py 20`.

## R2.4 Not checked in round 2

Live Discord (forbidden; OP-04 unchanged: the evidence predates round 2's policy changes too); the MAI-UI-2B GPU test
(grounding code unchanged since 6851826, verdict basis as in round 1); public-network egress of the sandbox host's
broker (would contact an external host); Discord's handling of unusual path spellings (`/API/…`, IP literal with a
Host header — allowed by the policy, but no credential can be attached; g08 NOTE lines).

---

# Round 3 — re-verification at 85101a2 (2026-11-28)

Verdict commit: `85101a2a248f246014b8b929e35a82847dcac9b6` (`git log -1 --format=%H -- cam_live/src/cam_live/harness`;
HX's fixes in `c9ae2d4` and `85101a2`, claims in `57b95aa`, `docs/HX.md` "Round 3"; `docs/EA/SECURITY.md` §2a added
in `1bbee15`). The working tree's harness, harness tests and contracts are identical to 85101a2. Policy-level and
Chromium probes ran against `git archive 85101a2` (`G2H_SRC`), root probes and suites from the working tree. All 49
ids re-judged at 85101a2.

**Verdicts (49):** 44 verified, 5 partial (CU-37, IH-16, OP-09, CU-19, OP-04), 0 failed. Compared with round 2:
CU-37 failed -> partial; IH-07, CU-27, OP-07, CU-25 partial -> verified.

## R3.1 What I executed

| What | Result |
|---|---|
| Harness suites as root, browser + privileged, compute first | **627 passed, 0 failed, 4 deselected, 274 s** |
| G-sec p01-p15 / G-env probes / EB registered variant | all 0 defects (same counts as round 2); SEP-BROWSER stale as before |
| g01-g06, g08, g09 (round-1/2 probes) | g01 30/0, g02 2/0, g03 7/0, g04 1/0, g05 24/0, g06 2/0, **g08 39/0, g09 9/0** |
| g07 (fixed: the marker read is guarded and the waits bounded; the H3 check now also requires that the host really restarted) | **7/0** |
| g10 audit after stop / relaunch | **20/20 and 40/40 rounds clean** |
| HX's flaky test: `test_ea_launcher.py` + `test_ea_privileged.py` + `test_hx_sandbox_host_privileged.py`, 8 repetitions | **8 of 8 pass** (round 2: 1 of 8 failed) |
| New probes g11-g14 | g11 9 / **2** (root), g12 6 / **2**, g13 10 / **6**, g14 5/0 (Chromium) |

## R3.2 Round-2 defects re-checked

| Defect | Re-check |
|---|---|
| R2-C1 learner -> root through the sandbox host's cwd | **fixed.** Every separated service runs `python -I …/service_boot.py <roots> -m <module>` with cwd `/`, no_new_privs, no LD_PRELOAD / LD_AUDIT; the root host has HOME `/root` and a system PATH (PYTHONPATH is still in its environment but ignored under `-I`: its sys.path is the source root, the base interpreter's stdlib, the venv's site-packages and the agentrs site-packages of the venv's `.pth`). No sys.path entry, no file below them (8037 directories walked), no interpreter in the chain, no `pyvenv.cfg` and no LD_LIBRARY_PATH directory is writable by camlearn, cambrowser, camsbx, camauth or camverify. `argparse.py`, `sitecustomize.py`, `usercustomize.py`, a `.pth` and a `cam_live/` package planted by the learner in the run directory, `sitecustomize.py` and a user-site `.pth` planted by the browser uid in `<run>/exec`, and `argparse.py` planted by the sandbox uid in the workspace never ran after kills and restarts of the root host, the authorization service and the environment service (g11 S1-S3, g07 7/0). `<roots>` is built by the launcher from its own source root and `LauncherConfig.pythonpath`, i.e. the operator's config file (`--config`, or on resume the path in the root-written `run.json`); the supervisor itself starts with cwd = the repository root. |
| R2-M1 false refusals | **fixed for every round-2 case** (g08 C, g09 part 1: honeypots, Drupal Antibot, quiz answers, turnstile counts, "Run security check"); real answers stay refused (g02, g13 controls). Narrower residue R3-m2 below |
| R2-m1 program-wide adapter standing | **fixed** (g08 B, B': redirect hops of a program's other steps, release on behalf of a program, CDN images all refused) |
| R2-m2 subresources under deny-listed paths | **fixed** (g09 part 2; g14: a deny entry spelled `localhost` for a `127.0.0.1` site, a percent-encoded path, `<link rel=preload>`, an image requested by an allowed stylesheet — nothing was sent) |
| R2-m3 audit false alarm after a stop | **fixed** (g10 60 rounds, 8/8 privileged repetitions); g12: an edited tail entry, an entry chained correctly but MAC'd with another key or without a MAC, and an entry replayed from another log are still detected after reconciliation — a forged tail is not accepted |

## R3.3 New defects

**R3-M1 (major, pre-existing since 4d6044a; missed in rounds 1-2) — the learner can make root overwrite any file
(CU-37, IH-16).** The run directory stays group-writable for the learner (mode 1771, group = the learner's gid;
`auth/launcher.py` `prepare_run_dir`), and root writes into it while the learner runs: the supervisor
(`runtime/cli.py` `cmd_supervise`) writes `preflight.json` and then `learner.go`, a `cam-live resume` also the
supervisor pid file, the operator CLI `control.json` (and opens `control.json.lock` with `"a+"`). All go through
`runtime/logs.py` `atomic_write_text`, which opens `.<name>.<pid>.<thread id>.tmp` with `O_WRONLY|O_CREAT|O_TRUNC`
— no `O_EXCL`, no `O_NOFOLLOW` — and `fs.protected_symlinks=1` does not cover a sticky directory that is not
world-writable. Measured (g11 S4): (a) with the name given, root followed the learner's symlink and overwrote a
root-owned 0600 file with `go`; (b) **unaided**: the learner uid watched the run directory with inotify, read the
temporary name of the `preflight.json` write (same pid and thread for every write of the supervisor's main thread)
and planted the link for the following `learner.go` write — the victim was overwritten in **5 of 5** races. Content
is fixed (`go`, or the JSON of the later writes), so no code execution was shown, but the learner can destroy any
root-writable file: the audit log or its head, the vault, the policy, the operator's signed record, `/etc` files.
Fix: create temporary files with `O_EXCL|O_NOFOLLOW` (or `tempfile.mkstemp`) in a directory the learner cannot
write, and/or take the learner's group off the run directory. Owners: runtime (`logs.py`, `cli.py`, `control.py`)
and the launcher layout. Repro (root): `g11_boot_path.py` (S4).

**R3-m1 (minor, pre-existing) — a restart rewrites the audit head from the log as found (IH-16).** Every start of
the authorization service appends and replaces the head from its in-memory state, so a tail removed while the
service was down passes verification after the next start (g12 f: 5 entries left of 7, `ok`), and a head far
behind is simply replaced (g12 g: 70 behind, `ok`) — the reconciliation's own 64-entry bound and its MAC checks are
not what decides. Only the auth uid or root can write the log in the launched deployment, which `SECURITY.md`
already places outside what the head can detect without an external anchor; but the head's truncation detection
does not survive a restart. Repro: `g12_audit_reconcile.py` (also on 4d6044a: same two lines).

**R3-m2 (minor) — remaining false refusals of the human-verification constraint (OP-09).** Still refused, and no
operator setting can lift a hard constraint: configuring the operator's own site's CAPTCHA (`http.post` of
`enable_captcha` / `captcha_provider` / `require_captcha_on_signup`; an "Enable CAPTCHA on sign-up" checkbox; a
"Get reCAPTCHA keys" button), a community FAQ link "Are you a bot?", and an e-mail confirmation field "Type the code
shown in the e-mail we sent you" (g13, 6 defects). Repro: `g13_round3_false_refusals.py`.

Not new: the sandbox host still verifies no grant (the browser uid can run programs as camsbx without the
authorization service; no privilege gain, g07 H2); the host's silent-client timeout and pre-read uid check were
read in the code (`sandbox_host.py` `_handle`) and the uid refusals measured (g07 H1).

## R3.4 Not checked

Live Discord (OP-04 unchanged); the MAI-UI-2B GPU test; public-network egress of the root host's broker; code
execution through R3-M1 (only fixed-content overwrites were attempted).

---

# Round 4 — final harness re-check at a5ab772 (2026-11-28)

**Harness-security go / no-go for the live planning launch: CONDITIONAL GO.** The harness boundary holds. But until
R4-C1 and R4-M1 are fixed, the operator must not run `cam-live export-checkpoint` on the live run directory at all,
and must run `cam-live plan-report` only with `--out <path in a root-only directory>`. Both defects need the learner
uid to be compromised and the operator to run the command as root.

Verdict commit: `a5ab7720f169c1bc23cc442be96a31310dfa29eb`. HX's fixes are in `0fedb1a`, and the claims in `6945e2a` /
`docs/HX.md` "Round 4". HX-6 was fixed by the master in `a5ab772`. The working tree's `cam_live/src` and
`cam_live/tests` are identical to a5ab772. All 49 ids were re-judged at a5ab772.

**Verdicts (49):** 42 verified, 6 partial (IH-07, IH-16, CU-27, OP-07, CU-19, OP-04), 1 failed (CU-37). The partial
and failed verdicts of IH-07, IH-16, CU-27, OP-07 and CU-37 all rest on R4-C1 / R4-M1. Both are pre-existing
runtime operator-CLI defects (present at 4d6044a), not in `harness/`.

## R4.1 What I executed

| What | Result |
|---|---|
| Harness suites as root, browser + privileged, compute first | **632 passed, 0 failed**, 304 s |
| `tests/runtime` as root (`-m "privileged or not (gpu or slow or network)"`) | **244 passed, 1 skipped**, 98 s |
| G-sec p01-p15, G-env probes, EB registered variant | all 0 defects, with the same counts as round 3 (SEP-BROWSER stale) |
| g01-g14 | g01 30/0, g02 2/0, g03 7/0, g04 1/0, g05 24/0, g06 2/0, g07 7/0, g08 39/0, g09 9/0, g10 1/0 (20 rounds), **g11 11/0, g12 8/0, g13 16/0**, g14 5/0 |
| New probes g15-g17 | g15 0 / **2** (root CLI), g16 2 / **1** (safefs races), g17 5/0 (audit anchor) |

## R4.2 Round-3 defects re-checked

| Defect | Re-check |
|---|---|
| R3-M1: root follows learner symlinks in the run dir | **Fixed for the safefs writers.** g11 S4: a symlink planted under the supervisor's temporary name is never followed, given name or not, and the inotify race now wins 0 of 5. g16 F1/F2 ran 3000 writes each while the learner kept swapping the target between its own file and a symlink to a victim, then a path component between its own directory and a symlink. Nothing was written through: 145 writes were refused and the victim stayed intact. `control.json.lock` uses `_open_lock` (HX-6; tests/runtime pass). **Not fixed outside safefs:** R4-C1 and R4-M1 below. |
| R3-m1: a restart rewrote the audit head | **Fixed.** g12 8/0: a truncated log, and a head 70 entries behind, are now reported. g17: the anchor `<run>/logs/audit.anchor` is root 0600 in a root 0755 directory; camlearn and camauth get PermissionError on write, unlink and replace. A tail removed while the service was down, with the log and head rewritten to match (MAC computed with the real key), is reported at the next start ("discontinuity … out-of-band anchor (entry 14) is not in the log"). The anchor records whatever head the auth uid left at stop, so the auth uid's own tampering stays outside, as SECURITY.md says. |
| R3-m2: CAPTCHA false refusals | **Fixed.** g13 16/0: own-site CAPTCHA configuration, the "Are you a bot?" FAQ link and the e-mail code field are allowed; real answers are still refused. |

## R4.3 New defects (found by grepping every file operation reachable from a root process)

**R4-C1 (critical, pre-existing) — `cam-live export-checkpoint` as root copies learner-chosen files into any
directory (CU-37, IH-07, CU-27, OP-07).** This is `runtime/cli.py` `cmd_export_checkpoint`, lines 690-704. With no
runtime alive, the checkpoint id and path come from `<run>/state.json`, which is the learner's own file. The
destination `<run>/exports/export_<id>` is built without normalising the path. When the path is not a loadable
checkpoint, it is copied with `shutil.copytree(src, dest, dirs_exist_ok=True)`.

As camlearn, the probe wrote `state.json` with `id = "x/../../../../../../<root-only dir>"` and a path to a directory of
its own. That directory held a file named `authorized_keys`. Then root ran the real CLI,
`cam-live export-checkpoint --run-dir R`: it exited 0 and the learner's file now sits root-owned in the root-only
directory. Aimed at `/root/.ssh` with sshd listening on 22, this is root code execution; I did not take that step.
With `--out` the source path is still the learner's choice, so root can copy for example `/root/.config/camlive`
into the operator's export. REPRODUCE.md tells the operator to export anything worth keeping with this command.
Fix: never take a filesystem path from the run's state, validate the id, normalise `dest` under the run directory,
and read the checkpoint with safefs. Repro (root): `g15_operator_cli_root.py` E1.

**R4-M1 (major, pre-existing) — `cam-live plan-report` writes through a learner symlink (CU-37, IH-16).** This is
`runtime/plan_report.py` `write_report`, lines 341-345. The default output is `<run>/plan_report.md`, written
through `<out>.tmp` with `open(..., "w")`. The learner planted `plan_report.md.tmp` as a symlink to a root-owned
file, and root wrote "# Plan report — run …" into that file. This report is the planning phase's deliverable
(ledger session 4). Fix: use `atomic_write_text`. Repro: `g15_operator_cli_root.py` E2.

**R4-m1 (minor) — the safefs read opens before it checks the file type.** `safefs.read_nofollow` checks
`S_ISREG` only after `os.open(…, O_RDONLY|O_NOFOLLOW)`. The learner replaced its own `state.json` with a FIFO, and
root's `runtime.logs.read_json` was still blocked after 10 s. That read sits in the supervisor's loop
(`cmd_supervise` reads `state.json` every second) and in the operator's `status`, `resume` and `export-checkpoint`.
The same pattern applies to `_mk`, which opens existing layout files `O_WRONLY|O_APPEND`, and to
`chown_chmod_nofollow`. Fix: open with `O_NONBLOCK`, or check the type with `fstatat` first and then `fstat` after
opening. Repro: `g16_safefs.py` F3.

**R4-m2 (minor) — `credentials revoke` unlinks `<run>/verifier/secrets/<name>` by path** (`harness/credentials.py`,
lines 147-149). The verifier uid owns `<run>/verifier`, so it can redirect the unlink of a file with that one name
(code reading only).

Checked and holding:
- The supervisor starts with cwd = repository root and the operator's environment.
- `run.json` and `task.yaml` are written by root before the learner starts.
- `OperatorKey` and `runtime/private.py` refuse directories they do not own.
- The GPU lock directory is root 775.
- `journal` and `resume` write through safefs.
- Checkpoints load with `safetensors` and `torch.load(weights_only=True)`, not pickle.
- `fs.protected_hardlinks=1` and `fs.protected_regular=2` hold on this host (safefs does not depend on them for
  writes).

## R4.4 Not checked

- Live Discord; OP-04 is unchanged.
- The MAI-UI-2B GPU test.
- Exploiting R4-C1 up to a root shell.
- The root CLIs of the evaluation runners (not part of the live launch).

---

# Round 5 — re-check at 8f6c615 (2026-11-28)

**Harness-security go / no-go for the live planning launch: GO.** Two minor items remain, and neither blocks the
launch:
- R5-m2: the plan report keeps terminal control characters. Read it with a pager that shows control characters
  (`less -U`), not with `cat`.
- R5-m1: when a checkpoint's MANIFEST names paths outside the checkpoint, root stats and hashes those files and
  prints their sizes to the operator.

Pinned commit: `8f6c615cbaaed325e1d0c2dcce45cafb2c4b1052`, from `git log -1 --format=%H -- cam_live/src/cam_live/harness
cam_live/src/cam_live/runtime`. HX's fixes are in 8f6c615, the bounded audit-race test in 5e57c7a and the claims in
702054b (`docs/HX.md` "Round 5"). INT3B is changing `runtime/` concurrently. The working tree had uncommitted edits
to `runtime/tasks.py`, `credit/*` and one harness test, so I tested an exact `git archive 8f6c615` extract wherever
the service uids did not have to traverse the code. The harness source, the launcher, the CLI and safefs are
identical between the working tree and 8f6c615. The privileged launches therefore ran from the working tree.

**Verdicts (49):** 47 verified, 2 partial (CU-19, OP-04), 0 failed. Compared with round 4, CU-37 goes from failed to
verified, and IH-07, CU-27, OP-07 and IH-16 go from partial to verified.

**Incident (my error, now repaired).** A stray `shutil.copy(..., "/dev/null")` in the first run of my probe g18
copied its source file's mode onto `/dev/null`, which went from 0666 to 0664. For about 6 minutes, roughly
04:07–04:13 UTC, non-root processes could not open `/dev/null` read-write; root was not affected. I restored it with
`chmod 0666 /dev/null` at 04:12:53 UTC and removed the line from g18. I then re-ran every probe and test that uses
service uids. Other agents' runs that started non-root processes in that window may show spurious failures, typically
`PermissionError: '/dev/null'` from `subprocess`.

## R5.1 What I executed

| What | Result |
|---|---|
| Harness suites, extract, `-m "browser or not (gpu or slow or network or privileged)"` | **619 passed**. The one other result is the separated browser test, which needs code the service uids can reach; it passes from the working tree |
| Harness privileged tests, working tree (`-m privileged`, `-m "privileged and browser"`) | **12 + 3 passed** |
| `tests/runtime`, extract | **247 passed, 1 skipped**. `test_rt_separated.py` (separated launch) **passes** from the working tree |
| G-sec p01-p15, G-env probes, EB registered variant | all 0 defects, with unchanged counts (SEP-BROWSER is stale) |
| g01-g17 | all hold: g01 30/0 … g15 **2/0**, g16 **3/0**, g17 5/0 |
| New g18 (export path, plan-report location) | 8 / **2** |

## R5.2 Round-4 defects re-checked

| Defect | Re-check |
|---|---|
| R4-C1: export-checkpoint copied learner-chosen files anywhere | **Fixed.** The checkpoint is chosen by name (`ckpt_<digits>`) among the real directories of `<run>/checkpoints`. It is copied entry by entry with nothing followed into private staging, and the destination lies outside the run and is never merged into. g15 E1 holds. g18: a `ckpt_7` symlink to a root-only directory gives "no checkpoint". A symlink inside a checkpoint refuses the export. In 15 exports while the learner flipped `ckpt_8` between a directory and a symlink to the root-only directory, nothing of that directory was exported. A hard link to a root-owned file cannot be created (`fs.protected_hardlinks=1`). The default destination `<state root>/exports` is root 0775, not writable by the learner, and is outside the run. A learner MANIFEST that matches the learner's own content is accepted; it only proves integrity, since the content is CAM's model by design, and root places it only where the operator decided. |
| R4-M1: plan-report wrote through a symlink | **Fixed.** g15 E2 holds. The default `<run>/plan_report.md` is written by root through safefs. The learner then gets PermissionError on replace, unlink and append, so it cannot swap the report the operator reads (g18 X6a). |
| R4-m1: FIFO hang | **Fixed** (g16 F3: `read_json` of a FIFO returns the default at once). |
| R4-m2: revoke unlink | **Fixed** (`safefs.unlink` relative to the directory fd; code read). |
| Remaining commands | `doctor` reads the configuration and disk usage. `verify-completion` writes repository documents. `evaluate-transfer --checkpoint P` reads a path the operator gives: it loads with safetensors and `torch.load(weights_only=True)` (torch 2.15), uses no `trust_remote_code` and stages in private temp directories. None of them writes into the run directory, and they are not part of the planning launch. |

## R5.3 New (minor)

**R5-m1 — a learner MANIFEST may name files outside the checkpoint.** `credit/checkpoint.py` `verify_checkpoint`
joins each MANIFEST key to the checkpoint path, and an absolute path or `../` escapes it. The export's
verification therefore stats and hashes, as root, the files the learner names, and the refusal shows their sizes:
"`/tmp/…/root_secret.txt: size 21 != 1`", "`../../../../../../etc/hostname: size 13 != 1`". The information reaches
only the operator's terminal, so it is a size and hash oracle at most, plus a way to make root hash a huge file.
Fix: reject MANIFEST keys that are absolute or contain `..`. Repro: `g18_round5_operator_cli.py` X4.

**R5-m2 — the plan report keeps terminal control sequences.** A CAM note with an OSC title, conceal and
clear-screen sequence appears in `plan_report.md` with its 4 ESC bytes intact. Text that reaches CAM's notes from
untrusted pages can therefore hide or rewrite what the operator sees when reading the report with `cat`. The journal
already strips control characters with `runtime/journal.py` `_CONTROL`, but the plan report does not. Repro:
`g18_round5_operator_cli.py` X6b. Owner: runtime (plan report).

## R5.4 Not checked

- Live Discord (OP-04 is unchanged).
- The MAI-UI-2B GPU test.
- Commands of the evaluation runners outside the live launch.
