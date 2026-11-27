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
