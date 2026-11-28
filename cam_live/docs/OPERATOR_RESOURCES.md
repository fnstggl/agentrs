# CAM Live 1.0 — what the operator must provide before live execution (DL-20, spec 20.20 / 19.5)

Describes the code at commit `fdab7d3` (read 2026-11-28 07:20–08:20 UTC; no later commit touches `cam_live/src`,
`tools`, `configs`, `tasks` or `pyproject.toml`). Each item names the code that enforces it, as `file:function`
relative to `cam_live/src/cam_live/`, or the evidence it comes from. The offline preflight and doctor results in §8
were run at `fdab7d3` with a scratch `CAM_LIVE_HOME`; their output is in `docs/evidence/DOC/preflight_offline.txt`.
Full command reference: `docs/REPRODUCE.md`.

Scope:

* **Included:** what the runtime refuses to start or act without, the exact launch procedure for the planning
  phase, and the infrastructure the run needs.
* **Not included:** strategy. What CAM will request is not predicted either. CAM asks the operator itself
  (`internal.request_operator`); §4 describes that channel and quotes what CAM asked for in the recorded staging
  rehearsals.
* **No secret values.** Credentials are named, never shown. The DOC agent did not read `/root/.config/camlive/`.
  Facts about files there come from the ledger (`docs/CAM_LIVE_LEDGER.md`).

## 0. Summary

| # | Item | Planning phase (`tasks/discord_100_planning.yaml`) | Execution phase (`tasks/discord_100.yaml`) | Enforced by |
|---|---|---|---|---|
| 1 | Durable state: a persistent volume, **or** the operator's acceptance that run data is lost with the machine | no volume here: launch with `--accept-ephemeral-storage` and copy the plan report off the box | same | `runtime/doctor.py:storage_checks` (`persistent_storage`); `runtime/preflight.py:preflight` (`ACCEPTABLE_RISKS`); `runtime/cli.py:cmd_run` |
| 2 | Signed live authorization record for the exact task file | operator's act at launch (types the objective id) | same, for that file | `tools/authorize_live.py:build_record`, `harness/auth/policy.py:PolicyService.register_live_authorization` |
| 3 | Operator Ed25519 key pair | exists (ledger); `tools/launch_live.sh` installs the public key per run | same | `harness/auth/liveauth.py`, `harness/auth/launcher.py:prepare_run_dir` |
| 4 | Discord bot token, in the run's vault and as the verifier's own copy | provided (root-only file); stored per run by `launch_live.sh` | same | `harness/credentials.py:store`; `runtime/preflight.py:authorization_items`, `discord_items` |
| 5 | Discord application: privileged intent `GUILD_MEMBERS`; bot in the server; bot role holds the scope | done (ledger; live check `docs/evidence/discord/`) | same | `runtime/preflight.py:discord_items` |
| 6 | Task-file values marked `REQUIRED` | all filled | **16 unset** (§5) | `runtime/tasks.py:live_requirements`, `unset_paths` |
| 7 | The machine's shared GPU lock file | `runs/gpu_heavy.lock` (a symlink on this machine only) | same | `nn/gpu.py:open_lock_file`; doctor `gpu_lock` (`runtime/doctor.py:harness_checks`, blocks live) |
| 8 | Root, five service users, Landlock, seccomp | present | same | `runtime/cli.py:cmd_run`; `runtime/doctor.py:sandbox_checks`, `service_user_checks` |
| 9 | Free disk ≥ 20 GB at every checkpoint | 30 GB free now (warning level, below 50 GB) | same | `storage.py:DiskGuard` |
| 10 | The operator, present during the run: approvals, answers to CAM's requests, stop | needed | needed | `runtime/control.py:OperatorControl`, `harness/operator_requests.py:OperatorRequests` |

At `fdab7d3` the offline preflight of the planning task lists **8 live items** with `--accept-ephemeral-storage`
(9 without). All 8 need the running services and are confirmed by the supervisor of `cam-live run` at launch (§8).
The supervisor starts the learner only when its live report is empty (`runtime/cli.py:cmd_supervise`).

## 1. Already provided (recorded, not re-checked by DOC unless stated)

| What | State | Source |
|---|---|---|
| Discord application "CAM Live" | id `1575681780462129232` (`resources.discord.application_id` of the planning task) | ledger session 1 |
| Discord server "CAM's lab", owned by the operator | id `1575680989060010034`. The verifier excludes the owner from the count (`harness/discord_membership.py`). | ledger session 1 |
| Bot token | root-only file `/root/.config/camlive/discord_bot_token`; never in the repository, logs or prompts | ledger session 1 |
| Privileged intents | application flags `565248`: limited presence, guild-members and message-content intents enabled | ledger session 1 (read-only API check) |
| Bot in the server | added with permission integer `2269727275740401`; re-checked read-only: in "CAM's lab" and no other guild | ledger sessions 1, 3 |
| Discord adapter live check | 50 of 50 steps ok in "CAM's lab"; server restored | `docs/evidence/discord/README.md` |
| Operator signing key | `/root/.config/camlive/operator_ed25519` and `operator_ed25519.pub`, root-only | ledger session 1 |
| Dedicated Discord user account | supplied, **not usable by CAM**. User-account automation is refused in code by the hard constraint `X-USER-ACCOUNT` (`harness/auth/platform_constraints.py:findings`), which no policy, task, record or approval can permit. | ledger session 1; `docs/HX.md` |
| Planning task values | guild, application, credential name, destinations, publishing (every item needs approval), limits | `tasks/discord_100_planning.yaml` (unchanged since `d3ce519`) |
| Hardware | 1x RTX A6000 48 GB, shared with another tenant; 48 CPUs; 126 GB RAM; 300 GB overlay disk | doctor at `fdab7d3` (§8) |
| Backbone | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` under `/workspace/.hf_home`, 7.5 GB. `local_files_only: true`, so it must be on disk before the run. | `configs/cam_live.yaml`; doctor `backbone` |
| Chromium | Playwright 1.65, chromium-1255; a shared copy for the browser uid | doctor `chromium`; `docs/REPRODUCE.md` |

## 2. Planning phase: what still has to be decided or present

The task is `tasks/discord_100_planning.yaml`, objective id `discord-100-planning`, mode live.

* **Limits:** 6 h (`duration_s` 21600), 120 decisions, 60 external actions, 7200 compute-seconds, money 1 USD (no
  spending is planned).
* **Interfaces:** structured_browser, code, api, internal.
* **Approval:** every Discord write, every publication and every input on a page the operator does not own needs
  approval.

### 2.1 Durable state (the operator's decision)

* **The check.** `runtime/doctor.py:storage_checks` holds `persistent_storage` only when
  `/etc/vast_capabilities.json` reports `workspace_is_volume=true` **and** the state root (`CAM_LIVE_HOME`, default
  `<repo>/runs/cam_live`) lies under `$WORKSPACE`. Here `workspace_is_volume=False`. A live preflight therefore
  blocks unless the operator accepts the risk.
* **The override.** `runtime/preflight.py:ACCEPTABLE_RISKS` contains exactly one entry: `persistent_storage`,
  accepted by `--accept-ephemeral-storage` on `cam-live preflight`, `cam-live run` or `cam-live resume`, or as the
  first argument of `tools/launch_live.sh`.
  * The check still runs. It is reported under `live.accepted_risks` with "the run proceeds although this check
    fails; the risk is the operator's", and recorded in `run.json` / `preflight.json`.
  * `cam-live resume` must be given the flag again (`runtime/cli.py:cmd_resume`).
* **Consequence.** Run data, checkpoints and logs are lost on recycle or destroy. The rental ends 2026-11-30 03:00 UTC
  (ledger). CAM's plan has to be exported and copied off the machine (§3, step 6).
* **The alternative** is a machine with a persistent volume and `export CAM_LIVE_HOME=$WORKSPACE/...`.

### 2.2 GPU

* **The lock file.** Real-model work takes the machine's shared GPU lock file: `CAM_LIVE_GPU_LOCK`, default
  `<repo>/runs/gpu_heavy.lock` (`config.py:GPU_LOCK_PATH`).
  * On this machine that path is a symbolic link to `/workspace/agentrs/runs/cam_unified/gpu_heavy.lock`. The link
    is untracked, in a git-ignored directory.
  * The lock file is never created (`nn/gpu.py:open_lock_file`). A missing file raises `GpuUnavailable` naming
    `CAM_LIVE_GPU_LOCK`, and the doctor check `gpu_lock` blocks live mode.
  * A service uid that may not write the root-owned file opens it read-only; `flock` works on it all the same.
  * In another checkout, set `CAM_LIVE_GPU_LOCK` to the machine's lock file. The launcher passes the variable on to
    the services (`harness/auth/launcher.py:ENV_PASSTHROUGH`).
* **Holds.** The learner leases the lock for at most `runtime.gpu_hold_s` = 840 s per hold, then commits, releases,
  waits `gpu_yield_s` = 15 s and re-acquires in the same process. It also releases the lock whenever the runtime
  sleeps. Waiting is not charged to compute (`configs/cam_live.yaml` `runtime`; `runtime/gpu_cycle.py`).
* **Measured peaks** of the real configuration: 10.4–18.6 GB (`docs/INTEGRATION.md` §7.3, §7.6). Nothing checks free
  GPU memory against that; two doctor runs at 07:55 and 07:56 UTC found 24.2 and 16.9 GB of 47.4 GB free.
* **Waiting for the lock.** In the separated rehearsal the learner waited 849 s in the lock's queue before it
  started (`docs/evidence/INT3B/rehearsal_separated_README.md`). See step 4 of §3.

### 2.3 Disk

At least `storage.min_free_gb` = 20 GB must be free at every checkpoint, or nothing is written and the run pauses
(`storage.py:DiskGuard`). Below `warn_free_gb` = 50 GB a warning is logged and the dashboard shows level "warning".
**30 GB were free at 07:55 UTC.** A real-configuration weights-only checkpoint measured 291 MB; the runtime keeps 2
checkpoints, with optimizer state only in the newest.

### 2.4 Isolation

* **Root** is required for live mode: `runtime/cli.py:cmd_run` refuses `--inproc` and non-root starts ("live
  operation needs the separated services").
* **Services** (`harness/__init__.py:SERVICE_MODULES`):

  | Service | Runs as |
  |---|---|
  | authorization service | `camauth` |
  | computer environment | `cambrowser` |
  | sandbox host | root; it runs generated code as `camsbx` (`harness/sandbox_host.py:SandboxHost`) |
  | independent verifier | `camverify` |
  | learner | `camlearn` |

  The users exist here; the launcher creates them when they are missing.
* **No `CAP_NET_ADMIN`.** Destinations are enforced only by the authorization layer. The preflight prints this as a
  warning, not a block, and recommends a VM.
* **The audit log.** Its head is kept by the launcher in `<run>/logs/audit.anchor` (root, 0600) between stops and
  starts (`harness/auth/launcher.py:LaunchHandle.keep_audit_anchor`). Nothing for the operator to provide.

## 3. Exact launch procedure for the planning phase

Run every step as root, in an interactive terminal on this machine, from `/workspace/camlive`.

1. **Check the prerequisites:**
   * `ls -l runs/gpu_heavy.lock` (or `CAM_LIVE_GPU_LOCK`) names the machine's lock file;
   * `df -h /` shows at least 20 GB free;
   * the directory `$CAM_LIVE_CONF` (default `/root/.config/camlive`) holds `discord_bot_token` (mode 0600),
     `operator_ed25519` and `operator_ed25519.pub`.

   Optionally run the offline preflight:

   ```
   .venv/bin/cam-live preflight --task cam_live/tasks/discord_100_planning.yaml --accept-ephemeral-storage
   ```

   The expected result is exit 3 with exactly the 8 "not confirmed" items of §8 and the accepted risk.
2. **Optional check of the launch path.** `tasks/discord_launch_check.yaml` (objective
   `discord-100-planning-launch-check`) is a copy of the planning task limited to one decision, two external actions
   and 30 minutes:

   ```
   bash cam_live/tools/launch_live.sh --accept-ephemeral-storage cam_live/tasks/discord_launch_check.yaml 1 --read 'https://*' --read 'http://*'
   ```

   Then run `.venv/bin/cam-live stop --run-dir <its run directory>`.
3. **Launch the planning phase:**

   ```
   bash cam_live/tools/launch_live.sh --accept-ephemeral-storage cam_live/tasks/discord_100_planning.yaml 8 --read 'https://*' --read 'http://*'
   ```

   * `8` is the authorization window in hours. The task's own limits (6 h) still bound the run. The window must also
     cover the time the learner waits for the GPU lock, and no action is authorized after `not_after`.
   * `--read` lets CAM read the public web. Nothing is pre-authorized for publication, because there is no
     `--preauthorize`.
   * Environment: `CAM_LIVE_RUN_DIR` (default `runs/cam_live/live/discord-100-planning-<UTC time>`),
     `CAM_LIVE_OPERATOR_KEY` and `CAM_LIVE_CONF`.

   What the script does (`tools/launch_live.sh`):

   1. It stores the bot token in the run's vault and as the verifier's copy, from the 0600 file, printing nothing
      (`harness/credentials.py:store`).
   2. It creates `<run>/auth/keys` and installs the operator's public key there.
   3. It shows what will be authorized: task hash, owned and read destinations, budgets, window. It asks the
      operator **to type the objective id `discord-100-planning`**, then signs the record with the private key
      (`tools/authorize_live.py:main`). Signing is the operator's act.
   4. It runs `cam-live run --task ... --authorization <run>.signed.json --run-dir <run> --accept-ephemeral-storage`.
      The root supervisor (`runtime/cli.py:cmd_supervise`) starts the services and registers the task and the
      signed record. It then runs the preflight against the running services, including the read-only
      `discord.get_identity` / `discord.get_permissions` checks (authorized like any action), and writes
      `<run>/preflight.json`. Only if the live list is empty does it let the learner start. Otherwise it prints
      "live operation is BLOCKED; nothing was started for the learner".
4. **If `cam-live run` exits 5** ("the runtime did not start"), the run may still be starting. `cam-live run` waits
   only `--start-timeout` (120 s; `launch_live.sh` passes none) for the learner's `state.json`, and the learner
   writes it only after it holds the GPU lock and has built the agent. `launch_live.sh` stops on that exit code
   before printing its follow-up hint (KF-E34). Check the state:

   ```
   tail -f <run>/logs/supervisor.log
   .venv/bin/cam-live status --run-dir <run>
   ```

   Do not launch again into the same run directory: the script refuses it if `state.json` exists.
5. **Follow the run:**
   * `cam-live status --run-dir <run>`;
   * `cam-live dashboard --run-dir <run>` (127.0.0.1:8787; from the operator's machine through
     `ssh -p $VAST_TCP_PORT_22 -L 8787:127.0.0.1:8787 root@$PUBLIC_IPADDR`; viewing token in
     `$CAM_LIVE_HOME/dashboard.token`);
   * `cam-live journal --run-dir <run>`.
6. **End and export:**
   * `cam-live plan-report --run-dir <run> --out <file>.md` (or `--json`) exports CAM's objective, notes,
     predictions, experiments, operator requests, actions and learning updates. CAM's texts are reproduced byte-exact
     apart from the run logs' redaction, and escaped (`runtime/plan_report.py:build_report`). It may be run as root
     (`runtime/cli.py:cmd_plan_report`).
   * `cam-live stop --run-dir <run>` checkpoints the learner (`--emergency` for the emergency stop;
     `python -m cam_live.harness.auth.opctl --run-dir <run> stop "reason"` works even with the service down).
   * `cam-live export-checkpoint --run-dir <run> --out <dir>` writes the weights-only export.
   * Copy the plan report and the export off the machine to the operator's own storage. With ephemeral storage
     nothing else survives (§2.1).

## 4. During the run: approvals and CAM's own requests

### 4.1 Approvals

* **Approve with `cam-live approve`.** Use `cam-live approve --run-dir <run> <approval_id> [--note ...]` /
  `cam-live reject ...`, or the dashboard. Both go through `runtime/control.py:OperatorControl.approve`, which relays
  the approval's own grant, so the approved action executes with it.
* **Do not approve with opctl.** `python -m cam_live.harness.auth.opctl --run-dir <run> approve <id>` calls only
  `PolicyService.resolve_approval`. The runtime then authorizes the action again and the authorization service
  reserves the budget twice (INT3B-6, `docs/KNOWN_FAILURES.md` KF-E26).
* **What can be approved.** An approval cannot permit what the hard constraints refuse:
  * `X-HUMAN-VERIFICATION`: CAPTCHA and human-verification answers, solver services, evasion of anti-automation;
  * `X-USER-ACCOUNT`: Discord's client or user-account endpoints from the browser or raw HTTP;
  * `X-PLATFORM-ADAPTER`: Discord data other than through the `discord.*` adapter.

  These are fixed in `harness/auth/platform_constraints.py` and are not policy data.

### 4.2 CAM's requests to the operator

CAM's only channel for asking for resources is the tool `internal.request_operator`. It records a request (kind,
title, description, justification) in the run's request file; nothing is sent anywhere
(`harness/operator_requests.py:OperatorRequests`). The operator reads and answers:

```
.venv/bin/python -m cam_live.harness.operator_requests --run-dir <run> list [--all]
.venv/bin/python -m cam_live.harness.operator_requests --run-dir <run> grant  <request id> --note "what was provided"
.venv/bin/python -m cam_live.harness.operator_requests --run-dir <run> deny   <request id> --note "why not"
.venv/bin/python -m cam_live.harness.operator_requests --run-dir <run> answer <request id> --note "the information"
```

The answer reaches CAM as an event at its next poll. **An answer grants nothing by itself**
(`harness/operator_requests.py:main`). What the operator decides to provide enters only through:

* **a credential:** `python -m cam_live.harness.credentials set <name> --run-dir <run> --dest <pattern> [--scope S]`
  (hidden prompt or `--from-file` 0600; `harness/credentials.py:main`). With the services running, the alternative
  is `python -m cam_live.harness.auth.opctl --run-dir <run> put-secret <name> --dest <pattern>` (value on stdin);
* **a task value, destination or data permission** (e.g. `data_permissions.member_list`): an edit of the task file.
  That changes the task's hash, so the signed record no longer matches. A new record for the changed file
  (`tools/authorize_live.py`) has to be registered (`opctl --run-dir <run> register-live <signed.json>`), and the
  changed objective assigned by the operator's signed command (`runtime/control.py:OperatorControl.assign_objective`,
  the dashboard's assign-objective). DOC did not exercise this path end to end in live mode.

**What CAM will request is not known before the planning run.** This document does not anticipate it. The recorded
staging rehearsals of the same objective text (staging Discord, a neutral local web; CAM's own words, unedited in
the plan reports) gave:

| Rehearsal | Decisions | Requests by CAM | Source |
|---|---|---|---|
| segment A (in-process, real configuration) | 12 | 0 | `docs/evidence/INT3B/rehearsal_segmentA_check.json` |
| segment B (in-process, real configuration, `nn.max_prompt_tokens` 6144) | 30 | 5, all left open. Titles and kinds: "Discord bot account" (resource), "Access to Discord server management tools" (capability), "Member List Permission" (permission), "Additional accounts" (resource), "Server Management" (capability, no justification) | `docs/evidence/INT3B/rehearsal_segmentB_plan_report.md` |
| separated deployment (real configuration) | 8 | 0 (and no notes) | `docs/evidence/INT3B/rehearsal_separated_README.md`, `rehearsal_separated_20261128.json` |

How those requests relate to what exists (facts, not advice):

* A bot account and the server-management scope already exist: §1 and `harness/discord_perms.py:DEFAULT_SCOPE`.
* The member list is the data permission `data_permissions.member_list`, which is false in the planning task.
* Any account the operator adds is still subject to `X-USER-ACCOUNT`: no automation of user accounts.

CAM wrote no notes in any of the three rehearsals (`notes.count` 0 in each check file above).

## 5. Execution phase: values the operator must fill in `tasks/discord_100.yaml`

`cam-live preflight --task cam_live/tasks/discord_100.yaml --accept-ephemeral-storage` reports 24 live items at
`fdab7d3` (25 without the flag). Sixteen of them are values in the file; the rest are the service and Discord items
of §8 plus "no authorized guild is configured".

| Path | What the loader requires (`runtime/tasks.py:live_requirements`) | Note |
|---|---|---|
| `limits.duration_s`, `money`, `compute_s`, `max_external_actions`, `max_decisions` | each an explicit positive finite number | `money: 0` is refused. The values are copied into the signed record's budgets. |
| `resources.authorized_tools` | non-empty list of tool names or families | enforced at authorization for every action and program step (`P-TOOL-NOT-AUTHORIZED`) |
| `resources.authorized_destinations` | list of accounts and destinations | Discord: `account:discord:<guild>`, `channel:discord:*` |
| `resources.publishing.authorized` | `true` or `false` | if true, `resources.publishing.destinations` is required too |
| `resources.publishing.destinations` | list | `approval_required: true` is already set |
| `resources.discord_guild_id` | digits | a nested `resources.discord.guild_id` takes precedence |
| `resources.discord_bot_credential_name` | the **name** of the vault credential | `discord_bot_token` |
| `resources.owned_site` | base URL of a site the operator owns | optional: delete the line if no site is authorized |
| `resources.operator_authorization_record` | identifier of the record | checked as a slug only |
| `resources.currency` | 3–10 letter code | e.g. `USD` |
| `success.exclusions.test_account_ids` | list (may be `[]`) | every test account that joins must be listed |
| `data_permissions.collect_feedback_with_consent` | `true` or `false` | the other data permissions are already stated |

Also for this phase:

* **Objective text.** `discord_100.yaml` carries the spec Part 13 wording. The planning task carries the operator's
  amended text, confirmed by `objective_confirmation`. Which text the execution task uses is the operator's decision.
* **Interfaces.** This task also permits visual and skill. Visual computer use needs the frozen grounding models
  (doctor `grounding_models`, ok here).
* **Task-file scan.** The loader's scan for strategy content is a heuristic deny-list. Readable disguises still pass
  (G2F-1, KF-G82), so the operator is responsible for the file's content.
* **Launch.** Same procedure as §3 with this task file. `--owned` names the destinations CAM may write to;
  `--preauthorize` names destinations where publication needs no per-item approval (leave it out to keep per-item
  approval).

## 6. Discord application and server settings

Checked read-only through the computer environment once the services run (`runtime/preflight.py:discord_items`).

| Setting | Required value | Who sets it |
|---|---|---|
| Privileged gateway intent `GUILD_MEMBERS` | granted: the verifier reads the member list | application owner, Developer Portal, Bot page |
| Privileged intent `MESSAGE_CONTENT` | only if `data_permissions.store_message_content`, `read_message_content` or `train_on_message_content` is true (all false in both task files). Retaining message text is declared as reading, storing and training on it, and the default platform rule refuses it without the operator's express permission (HX, IH-15). | same |
| Bot membership | bot in the guild of `resources.discord.guild_id`, added by the server owner (OAuth2, `scope=bot`) | server owner |
| Bot role permissions | every permission of `resources.discord.scope`, else `harness/discord_perms.py:DEFAULT_SCOPE`: 18 permissions, integer `2269727275740401` (recomputed with `bits_of`). A missing one is reported as `missing_from_scope`. | server owner (Server Settings, Roles) |
| Never handed out by the bot | ADMINISTRATOR, KICK_MEMBERS, BAN_MEMBERS, MODERATE_MEMBERS, MANAGE_GUILD, MANAGE_ROLES, MANAGE_WEBHOOKS, MENTION_EVERYONE (`NEVER_GRANTED`) | nothing to set |
| Server ownership | the operator's account owns the server; the owner is excluded from the count | — |
| Membership baseline | members present when the experiment starts are not counted. Their keyed hashes are held in the verifier's state only; members who leave and rejoin stay excluded (INT3B G2R-2b). | nothing to set |

## 7. Infrastructure summary

| Resource | Needed | Here (doctor at `fdab7d3`) | Checked by |
|---|---|---|---|
| GPU | CUDA GPU; measured peak 18.6 GB | RTX A6000 47.4 GB, 16.9–24.2 GB free in two runs, shared | doctor `gpu` (presence only) |
| Shared GPU lock file | must exist; never created | `runs/gpu_heavy.lock` → the other tenant's lock | doctor `gpu_lock` (blocks live) |
| Disk | ≥ 20 GB free at every checkpoint; warning below 50 GB | 30 GB free of 300 GB | doctor `disk`; `storage.py:DiskGuard` |
| Persistent volume, or accepted ephemeral storage | one of the two | none: needs `--accept-ephemeral-storage` | doctor `persistent_storage` |
| RAM | ≥ 16 GB available | 87 GB available of 126 GB | doctor `memory` |
| Root, setuid / setgid, service users | required | yes | doctor `setuid_setgid`, `user:*` |
| Landlock, seccomp | required | Landlock ABI 1, seccomp | doctor `landlock`, `seccomp` |
| Kernel network filtering | recommended (VM) | no | doctor `kernel_network_filtering` (warning) |
| Backbone snapshot | on disk (`local_files_only`) | present, 7.5 GB | doctor `backbone` |
| Network egress to `discord.com`, `gateway.discord.gg` and the public web | for the live run | not tested by DOC (no external contact) | the supervisor's Discord identity check at launch |
| Python environment | `/workspace/camlive/.venv`, layered over `/workspace/agentrs/.venv` (third-party packages load from there) | as built | `docs/REPRODUCE.md`; KF-D14 |
| Time | signed window; the rental | rental ends 2026-11-30 03:00 UTC | — |

## 8. Offline preflight and doctor at `fdab7d3` (executed)

Commands: `CAM_LIVE_HOME=<DOC scratch>/r3_or/home .venv/bin/cam-live preflight --task cam_live/tasks/<file>
[--accept-ephemeral-storage]` and `.venv/bin/cam-live doctor --json`. Output: `docs/evidence/DOC/preflight_offline.txt`.

| Task file | Without the flag | With `--accept-ephemeral-storage` |
|---|---|---|
| `discord_100_planning.yaml` | exit 3, 9 live items | exit 3, 8 live items + 1 accepted risk |
| `discord_launch_check.yaml` | exit 3, 9 | exit 3, 8 + 1 accepted |
| `discord_100.yaml` | exit 3, 25 | exit 3, 24 + 1 accepted |

* **The 8 items of the planning task,** all "not confirmed" without the running services:
  * authorization service;
  * credential `discord_bot_token` in the vault;
  * computer environment;
  * the verifier's credential copy;
  * bot membership in the guild;
  * privileged intent GUILD_MEMBERS;
  * the bot's permissions against the scope;
  * independent verifier.
* **Staging:** ready. Warning: no CAP_NET_ADMIN.
* **Doctor:** 52 checks, 49 ok. Failed: `persistent_storage` (the only one that blocks live),
  `kernel_network_filtering`, `user_namespaces`. `gpu_lock` ok; `disk` 30 GB free (minimum 20, warning below 50).
* **No preflight against running live services has been recorded.** Status level 3 is not reached
  (`docs/COMPLETION.md`).
