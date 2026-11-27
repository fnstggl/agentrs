# E-fix — fixes of the G-env defects (computer environment, skills / API / publishing)

Workstream E-fix, 2026-11-27, branch `feat/cam-live`. Every defect below counts as fixed only where the verifier's
probe (`docs/verification/repro/env/`, unchanged) now reports the property **HOLDS** and a regression test in E's own
test directories pins it. "Before" = the probes run at the start of this work (tree at `9ba1e45`); "after" = the final
run on the tree with all E-fix commits and the EA-fix policy commit `72dccdf` (policy `default-2`), under machine
load average 20–27 on 48 cores.

```bash
cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/<probe>.py
cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/harness/env cam_live/tests/harness/compute \
    cam_live/tests/harness/skills -q --rootdir=cam_live -m "privileged or browser or not (gpu or slow or network)"
#   final run: 310 tests, 306 passed, 4 failed — all in test_es_discord_live_check.py (Discord paths, not E's: one since the
#   EA-fix policy, three since the master's policy commit ddb0f27 that landed during the run; change request E-4), 408 s
```

## Defects and what fixed them

| G-env | Property | Fix (file) | Probe check: before → after | Regression test |
|---|---|---|---|---|
| M1 | every browser request checked before it leaves; redirect target checked before it is followed | network guard: policy-checking forward proxy as Chromium's only way out (`_NetGuard`), route layer for writes per URL and WebSockets (`route_web_socket`), service workers blocked, QUIC off, WebRTC UDP proxied only, WebTransport removed, Chromium background traffic off, addresses pinned (`browser.py`) | BR-REDIRECT, BR-SUBREQ-WS, BR-SUBREQ-REDIRECT, BR-SUBREQ-SAMEORIGIN: DEFECT → HOLDS | `tests/harness/env/test_e_fix_browser.py` (redirect hops, websockets/tunnels, script writes + service workers, background traffic, DNS pinning) |
| M2 | an executed click is reported as what happened | tab list copied before awaiting (`browser.py:_state`); number parser never raises, non-finite values are no numbers (`perception.py:parse_number`) | BR-POPUP-RACE, PC-CRASH-PAGE, PC-CRASH-AFTER-CLICK: DEFECT → HOLDS | `test_e_fix_browser.py::test_a_click_that_opens_a_tab_is_reported_ok`, `::test_pages_with_unparseable_numbers_and_huge_fields_are_observable_and_bounded`, `test_e_fix_core.py::test_number_parser_edges` |
| M3 | at most once under concurrent submission and restart | claim taken under the lock before authorization; released when the caller's admission fails; journal "started" fsynced before execution; duplicates wait (bounded) or get "being executed" (`env.py:_claim/_admit`) | ENV-ONCE-2: DEFECT → HOLDS (ENV-ONCE-1, ENV-RESTART stay HOLDS) | `test_e_fix_core.py` (two grants, released claim, in-progress, SIGKILL during the write) |
| M4 | non-idempotent requests not re-sent after an uncertain failure | `run_with_retries(may_retry=)` (`failures.py`); POST/PATCH without `Idempotency-Key` not retried after 5xx / timeout / broken connection once sent; connect failures and idempotent methods keep bounded retries (`http_api.py`) | HT-POST-RETRY: DEFECT → HOLDS | `tests/harness/skills/test_e_fix_api.py::test_a_post_that_may_have_been_applied_is_not_re_sent` |
| M5 | `secret:<name>` references resolve with the real broker | `secret_name()` strips the prefix (`browser.py:broker_get`, `http_api.py:resolve_secret`, also used by the Discord executor); descriptors and guide document the reference form | BR-SECRET-USABLE, HT-SECRET-REFERENCE: DEFECT → HOLDS | `test_e_fix_browser.py::test_a_stored_secret_is_typed_by_reference`, `test_e_fix_api.py::test_secret_references_resolve_against_the_vault_name` |
| M6 | the environment as `cam-live run` launches it | new `harness/factories.py` (`all_executors` + one factory per modality); the service composes all of them when no `--executor` is given, finds the run's `task.yaml`, uses the authorization service's broker over AuthRPC; shared Chromium for service uids (`browser.py --install-shared`, installed here), uid plan checks reachability | SEP-COMPOSITION: DEFECT → HOLDS; SEP-BROWSER: see below | `test_e_fix_factories.py` (composition, service default with broker, separated launcher `-m privileged`, visual on the browser screen) |
| M7 | browser workflow, uploads, programs under the real policy | policy data by EA-fix (`72dccdf`: interactions on owned sites, workspace file roots, `T-PROGRAM`); E's side: uploads/programs/fills report what the policy needs | BR-WORKFLOW-REAL-POLICY, BR-UPLOAD-DEFAULT, BR-FILES, PG-PROGRAMS-REAL-POLICY: DEFECT → HOLDS | covered by the probes (policy-data, owner EA) and EA's tests; E's requests: `docs/change_requests/EA.md` "Requests from E" |
| M8 | bounded `Observation.data` | 256 KB per observation (configurable), payload stored as blob, `internal.inspect what=blob` (`env.py:_bound`); page fields clipped at extraction (`browser.py:_PAGE_JS`) | PC-SIZE: DEFECT → HOLDS | `test_e_fix_core.py::test_large_observation_data_is_bounded_and_stored_as_a_blob`, browser huge-field test |
| M9 | skills and the selector are part of the running system | `env.py:_learn`: `observe_execution` hook of every executor, selector update per action, estimates as `exec.modality.*` quantities and capabilities; `SkillExecutor.bind(env)` | SK-WIRING: DEFECT → HOLDS | `test_e_fix_core.py::test_executions_feed_skill_induction_and_the_selector_estimates` |
| M11 | a publication approval shows and binds the exact content | `publish.preview` returns `submit_arguments` (with `content`, `content_sha256`); submit refuses a content the draft no longer has, and an approved submission that names no content (`publishing.py`) | PB-APPROVAL-BINDING-ES3: DEFECT → HOLDS | `test_e_fix_api.py::test_a_publication_approval_binds_and_shows_the_exact_text`, `::test_the_real_authorization_service_shows_the_text_of_a_publication_awaiting_approval` |
| m1 | transformed secret echoes | released values remembered in memory; case-insensitive, separator-tolerant and half-value patterns (`env.py:EchoScrubber`, `RecordingSecrets`) | BR-SECRET-TRANSFORMED, HT-SECRET-SCRUB: DEFECT → HOLDS | `test_e_fix_core.py::test_transformed_echoes_of_a_released_secret_are_scrubbed` |
| m2 | non-finite quantities; hidden `<dl>`; file cut said | `env.py:_check_quantities`, `perception.py` (finite only, `truncated`/`file_bytes`), visible `<dl>` only | PC-NONFINITE-PAGE: DEFECT → HOLDS; PC-HIDDEN: NOTE → HOLDS; PC-NUMBERS-EDGE: DEFECT → NOTE (no crash, no non-finite value; "300K" still becomes 300000 with the literal kept in `quantity_text`) | `test_e_fix_core.py::test_non_finite_quantities_are_rejected`, `::test_file_observation_says_when_it_was_cut` |
| m3 | operator answer carries no learner-written text under operator trust | two events: `operator.answer` (trust operator: status, operator note) and `operator.request.context` (trust self: title, description) (`internal.py`) | IN-OPERATOR-TRUST: DEFECT → HOLDS | `test_e_fix_core.py::test_operator_answer_is_split_by_trust` |
| m4 | protected guard for every skill step | `protected_reasons` applied to the bound step at run time (`skills.py`) | SK-PROTECTED-GUARD: DEFECT → HOLDS | `test_e_fix_api.py::test_a_protected_resource_named_by_a_run_time_argument_is_refused` |
| m5 | selector block scope | `modality.block_keys`: intent key, same effect / same operation on the same place under any decision id, the refused action id; `update()` without intent blocks the refused action | MS-BLOCK-SCOPE: DEFECT → HOLDS | `test_e_fix_api.py::test_a_refusal_blocks_every_realisation_of_the_same_intent` |
| m6 | rate-limit refusals are `RESOURCE_LIMIT` | `failures.refusal_class` / `retry_after_of`, used by the environment, the browser and `authorization_result` | (probe text) "denied/resource_limit 'resource: rate limit …'" in SK-RATE-LIMIT, PG-RATE-LIMIT | `test_e_fix_core.py::test_a_rate_limit_refusal_is_classified_resource_limit` |
| m7 | tests never overwrite committed evidence | `test_es_skills.py` / `test_es_modality.py` write to `tmp_path` and compare with the committed record (`CAM_LIVE_WRITE_EVIDENCE=1` regenerates) | — | the two tests themselves |
| — | `test_es_publishing.py:229` | expected tool count computed from the executors' own tool lists | — | `test_publishing_to_an_owned_channel_goes_through_the_discord_adapter` passes |

C1 (instance portal reachable), C2 (skill steps and the write rate) and M10 (deny-alias) are EA's; with the EA-fix
policy they report HOLDS too (BR-INTERNAL, SK-RATE-LIMIT, BR-DENY-ALIAS, PG-RATE-LIMIT).

## What still does not hold, and why

* **SEP-BROWSER (unmodified probe): DEFECT.** The probe starts its staging site in its own (root) process and never
  registers it; since EA-fix, an unregistered loopback port is denied in every mode, so the navigation is refused by
  the authorization service *before* the environment executes anything. With the one line the local-address rule
  requires added (`auth.register_local_service(<site host:port>, "operator", ...)`;
  `docs/EB/repro/probe_separated_registered.py`, a copy of the probe with only that line added) all four checks HOLD: the service runs as `cambrowser`, composes all six
  modalities, scrubs through the broker, and Chromium 159 starts from the shared copy and loads the page. The same is
  pinned by `test_e_fix_factories.py::test_separated_service_as_cam_live_run_starts_it_registers_everything_and_runs_chromium_as_the_browser_uid`
  (`-m privileged`), which also checks that the Chromium processes run as `cambrowser`.
* **PC-NUMBERS-EDGE: NOTE.** The probe reports NOTE when nothing crashes and nothing non-finite appears (it has no HOLDS
  outcome): a multiplier letter is still read as a multiplier ("300K" → 300000; the literal stays in `quantity_text`).
* **BR-LATE-NAV, BR-UID, PB-APPROVAL-DEFAULT: NOTE** by design of the probe. BR-UID: the probe's scratch directory is
  not reachable by `cambrowser`, so Chromium runs under the probe's uid and the capability says why.

## All probe checks, before → after

Verdict lines of both runs: `docs/EB/probe_results_before.txt`, `docs/EB/probe_results_final.txt`.

| probe | check | before | after |
|---|---|---|---|
| `probe_api_publish_modality` | FT-RETRY-BOUNDS | HOLDS | HOLDS |
| `probe_api_publish_modality` | HT-DESTINATIONS | HOLDS | HOLDS |
| `probe_api_publish_modality` | HT-HARD-NEVER-RETRIED | HOLDS | HOLDS |
| `probe_api_publish_modality` | HT-LITERAL-CREDENTIALS | HOLDS | HOLDS |
| `probe_api_publish_modality` | HT-POST-RETRY | DEFECT | HOLDS |
| `probe_api_publish_modality` | HT-SECRET-BINDING | HOLDS | HOLDS |
| `probe_api_publish_modality` | HT-SECRET-REFERENCE | DEFECT | HOLDS |
| `probe_api_publish_modality` | HT-SECRET-SCRUB | DEFECT | HOLDS |
| `probe_api_publish_modality` | HT-SIZE | HOLDS | HOLDS |
| `probe_api_publish_modality` | MS-BLOCK-SCOPE | DEFECT | HOLDS |
| `probe_api_publish_modality` | MS-NO-SWITCH | HOLDS | HOLDS |
| `probe_api_publish_modality` | PB-ANALYTICS | HOLDS | HOLDS |
| `probe_api_publish_modality` | PB-APPROVAL-BINDING | HOLDS | HOLDS |
| `probe_api_publish_modality` | PB-APPROVAL-BINDING-ES3 | DEFECT | HOLDS |
| `probe_api_publish_modality` | PB-APPROVAL-DEFAULT | NOTE | NOTE |
| `probe_api_publish_modality` | PB-FLOW | HOLDS | HOLDS |
| `probe_browser` | BR-DENY-ALIAS | DEFECT | HOLDS |
| `probe_browser` | BR-FILE | HOLDS | HOLDS |
| `probe_browser` | BR-FILES | DEFECT | HOLDS |
| `probe_browser` | BR-INJECT | HOLDS | HOLDS |
| `probe_browser` | BR-INTERNAL | DEFECT | HOLDS |
| `probe_browser` | BR-LATE-NAV | NOTE | NOTE |
| `probe_browser` | BR-NAV-PAGE | HOLDS | HOLDS |
| `probe_browser` | BR-POPUP-RACE | DEFECT | HOLDS |
| `probe_browser` | BR-REDIRECT | DEFECT | HOLDS |
| `probe_browser` | BR-SECRET-BASIC | HOLDS | HOLDS |
| `probe_browser` | BR-SECRET-TRANSFORMED | DEFECT | HOLDS |
| `probe_browser` | BR-SECRET-USABLE | DEFECT | HOLDS |
| `probe_browser` | BR-STALE | HOLDS | HOLDS |
| `probe_browser` | BR-SUBREQ-DIRECT | HOLDS | HOLDS |
| `probe_browser` | BR-SUBREQ-REDIRECT | DEFECT | HOLDS |
| `probe_browser` | BR-SUBREQ-SAMEORIGIN | DEFECT | HOLDS |
| `probe_browser` | BR-SUBREQ-WS | DEFECT | HOLDS |
| `probe_browser` | BR-TABS | HOLDS | HOLDS |
| `probe_browser` | BR-TRUST | HOLDS | HOLDS |
| `probe_browser` | BR-UID | NOTE | NOTE |
| `probe_browser` | BR-UPLOAD-DEFAULT | DEFECT | HOLDS |
| `probe_browser_workflow` | BR-WORKFLOW-REAL-POLICY | DEFECT | HOLDS |
| `probe_env_core` | ENV-AUTH-1 | HOLDS | HOLDS |
| `probe_env_core` | ENV-AUTH-2 | HOLDS | HOLDS |
| `probe_env_core` | ENV-ONCE-1 | HOLDS | HOLDS |
| `probe_env_core` | ENV-ONCE-2 | DEFECT | HOLDS |
| `probe_env_core` | ENV-ORDER | HOLDS | HOLDS |
| `probe_env_core` | ENV-RESTART | HOLDS | HOLDS |
| `probe_env_core` | ENV-STOP | HOLDS | HOLDS |
| `probe_env_core` | RPC-UID | HOLDS | HOLDS |
| `probe_perception_internal` | IN-NOTES | HOLDS | HOLDS |
| `probe_perception_internal` | IN-OPERATOR | HOLDS | HOLDS |
| `probe_perception_internal` | IN-OPERATOR-TRUST | DEFECT | HOLDS |
| `probe_perception_internal` | IN-WAIT | HOLDS | HOLDS |
| `probe_perception_internal` | PC-CRASH-AFTER-CLICK | DEFECT | HOLDS |
| `probe_perception_internal` | PC-CRASH-PAGE | DEFECT | HOLDS |
| `probe_perception_internal` | PC-DOCS | HOLDS | HOLDS |
| `probe_perception_internal` | PC-FILE-LIMIT | HOLDS | HOLDS |
| `probe_perception_internal` | PC-HIDDEN | NOTE | HOLDS |
| `probe_perception_internal` | PC-NONFINITE-PAGE | DEFECT | HOLDS |
| `probe_perception_internal` | PC-NUMBERS | HOLDS | HOLDS |
| `probe_perception_internal` | PC-NUMBERS-EDGE | DEFECT | NOTE |
| `probe_perception_internal` | PC-SIZE | DEFECT | HOLDS |
| `probe_programs` | PG-PROGRAMS-REAL-POLICY | DEFECT | HOLDS |
| `probe_programs` | PG-RATE-LIMIT | NOTE | HOLDS |
| `probe_separated` | SEP-BROWSER | DEFECT | DEFECT |
| `probe_separated` | SEP-COMPOSITION | DEFECT | HOLDS |
| `probe_separated` | SEP-INTERNAL | HOLDS | HOLDS |
| `probe_separated` | SEP-UIDS | HOLDS | HOLDS |
| `probe_separated_registered` | SEP-BROWSER | — | HOLDS |
| `probe_separated_registered` | SEP-COMPOSITION | — | HOLDS |
| `probe_separated_registered` | SEP-INTERNAL | — | HOLDS |
| `probe_separated_registered` | SEP-UIDS | — | HOLDS |
| `probe_skills` | SK-DENIED-STEP | HOLDS | HOLDS |
| `probe_skills` | SK-INDUCE-RUN | HOLDS | HOLDS |
| `probe_skills` | SK-NO-APPROVAL-BYPASS | HOLDS | HOLDS |
| `probe_skills` | SK-NO-EXTRA-INTERFACE | HOLDS | HOLDS |
| `probe_skills` | SK-PROTECTED-GUARD | DEFECT | HOLDS |
| `probe_skills` | SK-RATE-LIMIT | DEFECT | HOLDS |
| `probe_skills` | SK-RETIRED | HOLDS | HOLDS |
| `probe_skills` | SK-SMUGGLED-STEP | HOLDS | HOLDS |
| `probe_skills` | SK-STALE-REPAIR | HOLDS | HOLDS |
| `probe_skills` | SK-WIRING | DEFECT | HOLDS |

## Latency (EB.json `measured`)

Re-measured three times with `repro_eb_latency.py` under load average 20–27 (numbers in `docs/EB/latency_efix.json`):
navigate (220–292 ms vs 234.5) and RPC ping hold; first action (1283–1693 vs 937), click with navigation (419–490 vs
298.5), fill (51–87 vs 39.7), inspect (30–76 vs 29) do not hold under that load; observation sizes reproduce
exactly (574 / 97 / 138). The figures include the network guard (each browser request passes the local proxy).
`docs/status/EB.json` keeps EB's original values and lists the re-measured runs next to them.
