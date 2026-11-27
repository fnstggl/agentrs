# Change requests — E-fix (computer environment: EB / ES / EC paths)

Each is worked around locally; none blocks the fixes in `docs/EB/FIXES.md`.

## E-1. `runtime/cli.py` (`cmd_supervise`) — pass the task file to the environment service

**What.** `extra["exec"] += ["--task", <run>/task.yaml]` next to `--auth-socket` / `--provenance` / `--run-dir`.
**Why.** The service configures Discord (`task.resources.discord`) and publishing (`task.resources.owned_sites`,
`discord.channels`) from the task. **Workaround.** `harness/service.py:find_task_file` reads `<run>/task.yaml`
when the run directory is `<run>/exec` (the layout `cam-live run` creates). `--executor` is no longer needed: without
it the service composes every executor (`cam_live.harness.factories.all_executors`).
```diff
-        extra["exec"] += ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", task.mode]
+        extra["exec"] += ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", task.mode,
+                          "--task", os.path.join(run_dir, "task.yaml")]
```

## E-2. `runtime/loop_service.py:build_inproc` — the in-process environment should be the full composition too

**What.**
```diff
-    env = build_environment(os.path.join(run_dir, "exec"), policy, provenance=task.mode, clock=clock,
-                            executors=tuple(sec.get("executors", ())), browser=bool(sec.get("browser", True)))
+    env = build_environment(os.path.join(run_dir, "exec"), policy, provenance=task.mode, clock=clock,
+                            secrets=policy.secrets_handle(), compose="all" if not sec.get("executors") else "minimal",
+                            task=task, executors=tuple(sec.get("executors", ())),
+                            browser=bool(sec.get("browser", True)))
```
**Why.** In-process runs (staging, `--inproc`) otherwise get internal + browser only and no secret broker (no
scrubbing). `build_environment(compose="minimal")` stays the default for existing callers.

## E-3. Operator setup — a shared Chromium for the service uids (not a diff)

The environment service runs as `cambrowser`; Playwright's download under `/root` is not reachable for it. Once per
machine (operator, root): `python -m cam_live.harness.browser --install-shared` (copies the installed revision to
`<venv>/share/cam_live/ms-playwright`, nothing is downloaded). `runtime/doctor.py` could check it with
`cam_live.harness.browser.find_chromium(uid_of("browser"))` and print that command when it returns "". Done on this
machine on 2026-11-27 (chromium-1255).

## E-4. `tests/harness/skills/test_es_discord_live_check.py::test_the_policys_live_mode_with_a_signed_record_and_a_narrower_scope`

Fails in the current tree with `discord.get_identity -> denied: blocked: no Discord bot token provided` (live mode
with a signed record). It also fails with the environment's recording secret broker switched off (checked); the
token is not released by the authorization service in this live-mode sequence. Not investigated further: the
Discord paths and live-mode secret release are outside E's paths. Owner: ES / EA.
In the final E-fix run three more tests of that file failed (`test_full_sequence_in_staging_leaves_the_server_as_it_was`,
`test_stops_before_any_write_when_the_policy_write_rate_is_too_low`,
`test_the_documented_live_preparation_yields_grants_and_releases_the_token_only_to_discord`, e.g. `listen_events: data
permission 'member_list' is not granted`); they passed in the run before, and the master's policy commit `ddb0f27`
('per-member Discord events need the member_list data permission') landed in between.
