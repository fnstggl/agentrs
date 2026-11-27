# CAM Live 1.0 — what the operator must provide before live execution (DL-20, spec 20.20 / 19.5)

Describes the code at commit `204e9d4` (read 2026-11-27 23:30 UTC). Every item below comes from the code that
enforces it (file:function named) or from a recorded fact (evidence path named). The offline preflight results
quoted are from `cam-live preflight` run at that commit with a scratch `CAM_LIVE_HOME`; the run's output is
reproduced in §6. Commands are given in full in `docs/REPRODUCE.md` §8–§10.

This list is limited to what the **runtime refuses to start or act without**, plus the infrastructure it runs on.
It contains no strategy and no list of what CAM might find useful. During the planning phase CAM asks for accounts,
channels, hosting, permissions or resources itself (`internal.request_operator`). The operator answers those
requests (§5.4). They are not anticipated here.

No secret value appears in this document. Credentials are named, never shown. The DOC agent did not read
`/root/.config/camlive/`. Where this document says a file there exists, the source is the ledger
(`docs/CAM_LIVE_LEDGER.md`, sessions 1–3).

## 0. Summary

| # | Item | Planning phase (`tasks/discord_100_planning.yaml`) | Execution phase (`tasks/discord_100.yaml`) | Enforced by |
|---|---|---|---|---|
| 1 | Durable state: a persistent volume under `$WORKSPACE`, or a machine whose manifest reports `workspace_is_volume=true` | **missing: blocks live** | **missing: blocks live** | `runtime/doctor.py:storage_checks` (`persistent_storage`, `needed_for="live"`), which feeds `runtime/preflight.py:preflight` |
| 2 | Signed live authorization record for the exact task file | operator's act (not yet done) | operator's act (not yet done) | `tools/authorize_live.py`, `harness/auth/policy.py:PolicyService.register_live_authorization`, `runtime/cli.py:cmd_run` |
| 3 | Operator Ed25519 key pair; the public key at `<run>/auth/keys/operator.pub` | exists (ledger); `launch_live.sh` installs the public key | same | `harness/auth/liveauth.py`, `harness/auth/launcher.py:prepare_run_dir` |
| 4 | Discord bot token in the run's vault, plus the verifier's own copy | provided (root-only file, ledger); `launch_live.sh` stores it per run | same | `harness/credentials.py:store`, `runtime/preflight.py:authorization_items`, `discord_items` |
| 5 | Discord application: privileged intent `GUILD_MEMBERS` | enabled (ledger, 2026-11-27) | same | `runtime/preflight.py:discord_items` |
| 6 | Bot is a member of the authorized server, and its role holds the operator's permission scope | done (ledger session 3; live check `docs/evidence/discord/`) | same | `runtime/preflight.py:discord_items` (`discord.get_identity`, `discord.get_permissions`) |
| 7 | Task-file values marked `REQUIRED` | filled (commit `d3ce519`) | **16 values unset** (§3) | `runtime/tasks.py:live_requirements`, `unset_paths` |
| 8 | Root, the five service users, Landlock, seccomp | present on this machine | same | `runtime/preflight.py:preflight` (isolation item), `runtime/doctor.py:sandbox_checks`, `service_user_checks` |
| 9 | GPU memory for Qwen3-4B + 50m core for the whole run | 1x RTX A6000 48 GB, **shared** with another tenant | same | not checked by preflight (§4) |
| 10 | Operator present to approve or reject queued actions and to answer CAM's requests | needed throughout the run | needed throughout the run | `harness/auth/policy.py` (approval queue), `harness/operator_requests.py` |

The only item that blocks the planning phase **on this machine** in a way no operator command fixes is item 1. The
supervisor of `cam-live run` re-runs the preflight against the running services and starts the learner only when
the live report is empty (`runtime/cli.py:cmd_supervise`). The doctor check `persistent_storage` is part of that
report. The code has no override for it.

## 1. Already provided (recorded, not re-checked by DOC except where stated)

| What | State | Source |
|---|---|---|
| Discord application "CAM Live" | id `1575681780462129232` (in `tasks/discord_100_planning.yaml`, `resources.discord.application_id`) | ledger session 1 |
| Discord server "CAM's lab", owned by the operator | id `1575680989060010034` (`resources.discord_guild_id`). The operator's own account is the owner. The verifier leaves the owner out of the member count (`harness/discord_membership.py`). | ledger session 1 |
| Bot token | Root-only file `/root/.config/camlive/discord_bot_token`. It is never in the repository, the logs or a prompt. | ledger session 1 |
| Privileged intents | Read-only API check on 2026-11-27: application flags `565248`, meaning the limited presence, guild-members and message-content intents are enabled | ledger session 1 |
| Bot added to the server | The operator was given the OAuth2 URL (`scope=bot`, permission integer `2269727275740401`). Session 3 re-checked read-only that the bot is in "CAM's lab" and in no other guild. | ledger sessions 1, 3 |
| Discord adapter live check | 50 of 50 steps ok in "CAM's lab"; the server was restored afterwards | `docs/evidence/discord/README.md`, `live_check_20261127.json` |
| Operator signing key | `/root/.config/camlive/operator_ed25519` (private) and `operator_ed25519.pub`, root-only | ledger session 1 |
| A dedicated Discord user account | Supplied, but **not used and not usable by CAM**. User-account automation is outside CAM's action space: policy rule `PL-DISCORD-NO-USER-AUTOMATION`, spec 12.5. The one login attempt stopped at an hCaptcha challenge, which was not circumvented. | ledger session 1 |
| Planning-task operator values | guild, application, credential name, destinations, publishing (all items need approval), conservative limits | commit `d3ce519`, `tasks/discord_100_planning.yaml` |
| Hardware | 1x RTX A6000 48 GB (shared), 48 CPUs, 125 GB RAM, 300 GB overlay disk (167 GB free at 23:30 UTC) | ledger; `df -h /` at the time of writing |
| Backbone snapshot | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, 7.6 GB, under `/workspace/.hf_home`. The config sets `local_files_only: true`, so the snapshot must be on disk before the run. | `configs/cam_live.yaml`; `du -shL` |
| Chromium | Playwright 1.65, chromium 159 | ledger; `docs/REPRODUCE.md` §2.3 |

## 2. Planning phase: what is still needed

The task file is `tasks/discord_100_planning.yaml`, objective `discord-100-planning`. Its limits are 6 h,
120 decisions, 60 external actions, 7200 compute-seconds and money 1 USD (no spending is planned). Interfaces:
structured_browser, code, api, internal.

### 2.1 Durable state (blocking)

* **What.** The run directory has to survive a recycle or destroy of the machine, which means a host volume.
  `runtime/doctor.py:storage_checks` reads `/etc/vast_capabilities.json`. The check `persistent_storage` holds only
  when the manifest says `workspace_is_volume=true` **and** the state root (`CAM_LIVE_HOME`, default
  `<repo>/runs/cam_live`) lies under `$WORKSPACE`.
* **State now.** `workspace_is_volume=False`, so the check fails, and the live report of both Discord tasks lists
  it (§6). The rental ends 2026-11-30 03:00 UTC (ledger). Run data and checkpoints have no off-box copy.
* **What the operator provides.** Either a machine with a persistent volume (then
  `export CAM_LIVE_HOME=$WORKSPACE/...`), or a change to the doctor rule that accepts a recorded off-box backup
  target. No such rule exists in the code.
* **Size.** Measured: a weights-only learner checkpoint of the real configuration is 291 MB
  (`ckpt_pre` of an evaluation run). An evaluation lifetime directory with pre/post checkpoints and exports is
  1.1 GB. The runtime keeps 2 checkpoints and stores optimizer state only in the newest
  (`configs/cam_live.yaml` `runtime.keep_checkpoints`). Below `storage.min_free_gb = 20` GB free, nothing is written
  and the run pauses (`cam_live.storage`, INTEGRATION.md §5a).

### 2.2 The signed live authorization record (the operator's act)

* **What it is.** A record bound to the task file's content hash (`TaskSpec.content_hash`), the task's limits
  (copied into `budgets`), the accounts, the destinations CAM may write to (`owned`) and read from (`read`), the
  destinations where publication is pre-authorized (`preauthorized`, empty for the planning phase), and a time
  window (`not_before`, `not_after = now + hours`). It is signed with the operator's private key. The
  authorization service holds only the public key and re-checks signature, hash and window on every decision
  (`tools/authorize_live.py:build_record`, module docstring).
* **Consequence.** Editing the task file after signing invalidates the record, because its hash no longer matches.
  The window must cover the run: the planning task's `limits.duration_s` is 21 600 s (6 h). No action is authorized
  after `not_after`.
* **Command.** Run interactively as root. The script stores the credential, installs the public key, signs after
  the operator types the objective id, and starts `cam-live run`:

  ```
  bash cam_live/tools/launch_live.sh cam_live/tasks/discord_100_planning.yaml <HOURS> --read 'https://*' --read 'http://*'
  ```

  It refuses to run without a terminal, because the signature needs the typed confirmation. Environment:
  `CAM_LIVE_CONF` (default `/root/.config/camlive`) must contain `discord_bot_token` (mode 0600) and
  `operator_ed25519.pub`. `CAM_LIVE_OPERATOR_KEY` (default `$CAM_LIVE_CONF/operator_ed25519`) is the private key.
  `CAM_LIVE_RUN_DIR` defaults to `runs/cam_live/live/<objective>-<UTC>`. The Discord destinations of the task
  (`account:discord:<guild>`, `channel:discord:*`) are added to `owned` automatically.
* **Step by step** (the same four steps; `docs/REPRODUCE.md` §8, §9.1):

  ```
  P=/workspace/camlive/.venv/bin/python
  $P -m cam_live.harness.credentials set discord_bot_token --run-dir R --from-file <0600 file>   # or hidden prompt
  install -m 0644 <operator public key> R/auth/keys/operator.pub
  $P cam_live/tools/authorize_live.py --task cam_live/tasks/discord_100_planning.yaml --private <private key> \
      --hours <HOURS> --out R.signed.json --read 'https://*' --read 'http://*'
  $P -m cam_live.runtime.cli run --task cam_live/tasks/discord_100_planning.yaml --authorization R.signed.json --run-dir R
  ```

  Do not use `--yes`: it skips the typed confirmation, and signing is the operator's explicit act.
  `configs/cam_live.yaml` has no `launcher.operator_pubkey`. Without the `install` step, registration fails with
  "operator public key not installed" (`harness/auth/policy.py:PolicyService.register_live_authorization`).
* **New operator key** (only if the operator wants a key that only they have held):
  `$P -m cam_live.harness.auth.opctl init-operator-key --private K --public P`. Place both files as
  `operator_ed25519` / `operator_ed25519.pub` in a directory passed as `CAM_LIVE_CONF`, together with the token
  file. One operator public key per run; key rotation is not implemented (`docs/status/EA.json` not_done).

### 2.3 Bot credential per run

* `harness/credentials.py:store` writes `<run>/vault/discord_bot_token.secret.json` (authorization uid, 0600) and,
  because `KNOWN["discord_bot_token"]["verifier"]` is true, `<run>/verifier/secrets/discord_bot_token` (verifier uid,
  0600). The secret is bound to `discord.com`, `gateway.discord.gg` and `*.discord.gg`.
* The value is never taken from argv. `--from-file` must be mode 0600 (`read_value`). `list` prints names and
  bindings only.
* The preflight requires both copies. The vault copy is confirmed through the authorization service's
  `secret_names()`; the verifier copy by its file (`runtime/preflight.py:authorization_items`, `discord_items`).

### 2.4 Discord application and server settings

These are checked read-only through the computer environment once the services run
(`runtime/preflight.py:discord_items`).

| Setting | Required value | Where the operator sets it |
|---|---|---|
| Privileged gateway intent `GUILD_MEMBERS` | granted: the verifier reads the member list | Developer Portal, application, Bot page, Privileged Gateway Intents |
| Privileged intent `MESSAGE_CONTENT` | only if `data_permissions.store_message_content`, `read_message_content` or `train_on_message_content` is true. All are false in both task files. Commit `204e9d4` (HX, IH-15) also treats retaining message text as reading, storing and training on it, which the default platform rule refuses without the operator's express permission. | same |
| Bot membership | the bot is in the guild of `resources.discord.guild_id`, added by the server owner (OAuth2, `scope=bot`) | server owner |
| Bot role permissions | the role holds every permission of the scope: `resources.discord.scope` if the task sets it, else `harness/discord_perms.py:DEFAULT_SCOPE`. That is 18 permissions, integer `2269727275740401` (recomputed with `bits_of(DEFAULT_SCOPE)`): CREATE_INSTANT_INVITE, MANAGE_CHANNELS, MANAGE_GUILD, ADD_REACTIONS, VIEW_AUDIT_LOG, VIEW_CHANNEL, SEND_MESSAGES, MANAGE_MESSAGES, EMBED_LINKS, ATTACH_FILES, READ_MESSAGE_HISTORY, MANAGE_ROLES, MANAGE_EVENTS, MANAGE_THREADS, CREATE_PUBLIC_THREADS, SEND_MESSAGES_IN_THREADS, CREATE_EVENTS, PIN_MESSAGES. A missing one is listed as `missing_from_scope`. | Server Settings, Roles (server owner) |
| Bot never hands out | ADMINISTRATOR, KICK_MEMBERS, BAN_MEMBERS, MODERATE_MEMBERS, MANAGE_GUILD, MANAGE_ROLES, MANAGE_WEBHOOKS, MENTION_EVERYONE, in any role or overwrite it creates (`NEVER_GRANTED`). Nothing for the operator to set. | — |
| Server ownership | the operator's account owns the server. The owner is excluded from the count (`harness/discord_membership.py`). | — |

Not implemented for Discord, so there is nothing to provide for it: webhooks, persistent Gateway connections, and
anything that acts on a member (roles of members, kick, ban, timeout, direct messages). Source:
`docs/status/ES.json` not_done.

### 2.5 Isolation and host

* **Root** with the service users `camlearn`, `cambrowser`, `camsbx`, `camauth` and `camverify`. They exist here; the
  launcher creates them when run as root. Live mode refuses `--inproc` and any non-root start
  (`runtime/cli.py:cmd_run`: "live operation needs the separated services").
* **Landlock and seccomp** are available here (doctor). There is no `CAP_NET_ADMIN`, so per-destination network
  filtering is enforced only by the authorization layer. The preflight prints this as a warning, not a block
  (`runtime/preflight.py:preflight`, `warnings`), and recommends a VM for live operation. Running on a VM is the
  operator's decision.
* **GPU.** At `204e9d4` the runtime did not take the shared GPU lock (`docs/REPRODUCE.md` §13). Since INT3B `5ae3b59`
  it holds the lock for the whole process lifetime (`runtime.gpu_lock: auto`). `runtime.gpu_hold_s` can bound a hold,
  pausing and resuming between holds (`docs/REPRODUCE.md` §14). The default config sets neither key. Measured peaks of the real configuration are 10.4 GB for the meta pathway and
  18.6 GB for staging validation (INTEGRATION.md §7.3, §7.6). The operator has to make sure the other tenant leaves
  that much free. Nothing in the preflight checks it.
* **RAM.** The doctor's `memory` check wants at least 16 GB available.

### 2.6 During the run

* **Approvals.** In the planning phase every Discord write, every publication and every input on a page the
  operator does not own is queued for approval (`tasks/discord_100_planning.yaml` header; the record pre-authorizes
  nothing). The operator reviews the queue with `cam-live approve --run-dir R <approval_id> [--note ...]` or
  `cam-live reject ...`, or with `$P -m cam_live.harness.auth.opctl --run-dir R approvals | approve ID | reject ID`.
  Approving from the dashboard needs root (`docs/status/EA.json` not_done).
* **CAM's requests.** List them with `$P -m cam_live.harness.operator_requests --run-dir R list [--all]`. Answer
  with `grant ID --note ...`, `deny ID --note ...` or `answer ID --note ...`. An answer grants nothing by itself.
  A credential enters only through `cam_live.harness.credentials`, and a permission only through a new signed
  record or policy (module docstring).
* **Stop.** `cam-live stop --run-dir R [--emergency]`, or `opctl --run-dir R stop "reason"`, which works even with
  the service down. Revoke with `opctl revoke <scope>`.

## 3. Execution phase: values the operator must fill in `tasks/discord_100.yaml`

`cam-live preflight --task cam_live/tasks/discord_100.yaml` reports 25 live items at `204e9d4` (§6). Sixteen of them
are values in the file: everything marked `REQUIRED`, plus the items that `runtime/tasks.py:live_requirements`
derives from them. The rest are the service, Discord and storage items of §2, which apply again.

| Path | What the loader requires | Note |
|---|---|---|
| `limits.duration_s`, `limits.money`, `limits.compute_s`, `limits.max_external_actions`, `limits.max_decisions` | each an explicit positive finite number (`v > 0`); "resources are never assumed unlimited" | `money: 0` is refused. The planning task uses `1` with no spending planned. The values are also copied into the signed record's `budgets`. |
| `resources.authorized_tools` | non-empty list of tool names or families | e.g. the planning task's list. The authorization service denies an operation that no policy line matches (`docs/status/EA.json`). |
| `resources.authorized_destinations` | list of accounts and destinations CAM may act on | Discord destinations have the form `account:discord:<guild>`, `channel:discord:*` |
| `resources.publishing.authorized` | `true` or `false` | if `true`, `resources.publishing.destinations` is required too |
| `resources.publishing.destinations` | list | `approval_required: true` is already set |
| `resources.discord_guild_id` | digits: the one guild this experiment may operate | the nested `resources.discord.guild_id` takes precedence when present (`harness/discord.py:DiscordExecutor.from_task`) |
| `resources.discord_bot_credential_name` | the **name** of the vault credential, never the value | `discord_bot_token` matches the stored credential |
| `resources.owned_site` | base URL of a site the operator owns and authorizes | optional in the schema: delete the line if no site is authorized (the planning task has none) |
| `resources.operator_authorization_record` | identifier of the operator's record | checked as a slug only, not against the `record_id` that `authorize_live.py` writes (`docs/REPRODUCE.md` §13) |
| `resources.currency` | 3–10 letter code | e.g. `USD` |
| `success.exclusions.test_account_ids` | list of operator-declared test accounts; may be `[]` | every test account that joins must be listed, or the verifier counts it |
| `data_permissions.collect_feedback_with_consent` | `true` or `false` | the other data permissions are already stated (all `false` except `store_aggregate_metrics`) |

Also for this phase:

* The objective text of `discord_100.yaml` is the spec Part 13 wording. The planning task carries the operator's
  amended objective verbatim, confirmed by `objective_confirmation`. Which text the execution task carries is the
  operator's decision. Changing the text changes the task hash and so needs a new signed record.
* The execution task permits the interfaces visual and skill as well as the planning task's four. Visual computer
  use needs the frozen grounding models on disk (doctor check `grounding_models`, `needed_for="live"`; ok on this machine).
* A new signed record: `authorize_live.py --task cam_live/tasks/discord_100.yaml ...` (or `launch_live.sh` with that
  file). `--owned` names the destinations CAM may write to, and `--preauthorize` names destinations where
  publication needs no per-item approval. Leaving `--preauthorize` out keeps per-item approval.
* Whatever CAM requested in the planning phase and the operator granted (accounts, channels, hosting) enters as
  task-file values, credentials (`harness/credentials.py set <name> --dest <pattern>`) and destinations in the
  signed record. This document does not anticipate those.

## 4. Infrastructure summary

| Resource | Needed | Here | Checked by |
|---|---|---|---|
| GPU | CUDA GPU memory for the whole run: measured peak 18.6 GB (staging validation, real configuration); architecture §8 caps a heavy job on this shared GPU at 24 GB | RTX A6000 48 GB, shared | doctor `gpu` (presence only) |
| Disk | ≥ 20 GB free at every checkpoint (`storage.min_free_gb`), warning below 50 GB; 7.6 GB backbone snapshot | 167 GB free | doctor `disk`; `cam_live.storage.DiskGuard` during the run |
| Persistent volume or off-box copy | required for live (§2.1) | **none** | doctor `persistent_storage` (blocks live) |
| RAM | ≥ 16 GB available | 125 GB | doctor `memory` |
| Root, setuid/setgid, service users | required for live | yes | doctor `setuid_setgid`, `user:*`; preflight isolation item |
| Landlock, seccomp | required for live | Landlock ABI 1, seccomp | doctor `landlock`, `seccomp` |
| Kernel network filtering | recommended (VM) | no (`CAP_NET_ADMIN` absent) | doctor `kernel_network_filtering` (warning) |
| Backbone snapshot on disk | `Qwen/Qwen3-4B@1cfa9a72…` (`local_files_only`) | present | doctor `backbone` |
| Chromium (Playwright) | for structured_browser | present | doctor `chromium` |
| Network egress to `discord.com` / `gateway.discord.gg` and the public web | for the live run | not tested by DOC (no external contact) | preflight Discord identity check, at launch |
| Time | the signed window and the rental | rental ends 2026-11-30 03:00 UTC | — |

## 5. How to confirm readiness

1. Offline, as anyone: `cam-live preflight --task <task file>` lists every missing item. Without running services
   it reports the authorization, environment, Discord and verifier items as "not confirmed".
2. Authoritative, as root: `cam-live run ...` (or `launch_live.sh`). The supervisor starts the services, registers
   the task and the record, runs the preflight against them (including the read-only `discord.get_identity` and
   `discord.get_permissions`, authorized like any action), writes `<run>/preflight.json`, and starts the learner
   only if the live list is empty. Otherwise it prints "live operation is BLOCKED; nothing was started for the
   learner" and exits 3 (`runtime/cli.py:cmd_supervise`).
3. No preflight against running live services has been recorded (`docs/COMPLETION.md`: "no preflight evidence").

## 6. Offline preflight at `204e9d4` (executed)

Command: `CAM_LIVE_HOME=<DOC scratch>/oper/home .venv/bin/cam-live preflight --task cam_live/tasks/<file>`, exit 3
for both; full output in `docs/evidence/DOC/preflight_offline.txt`. Lines quoted from the output:

* `discord_100_planning.yaml`: **LIVE BLOCKED, 9 items.** Eight need the running services: authorization service,
  credential `discord_bot_token`, computer environment, verifier credential copy, bot membership, privileged intent
  GUILD_MEMBERS, bot permissions against the scope, and the independent verifier. The ninth is `[system]
  persistent_storage` ("workspace_is_volume=False … State on this machine is lost on recycle/destroy"). STAGING:
  ready. Warning: no CAP_NET_ADMIN.
* `discord_100.yaml`: **LIVE BLOCKED, 25 items.** The 16 task values of §3 (5 limits, authorized_tools,
  publishing.authorized, collect_feedback_with_consent, test_account_ids, discord_guild_id,
  discord_bot_credential_name, owned_site, operator_authorization_record, authorized_destinations, currency,
  publishing.destinations), 7 service items (authorization service, computer environment, verifier credential
  copy, bot membership, GUILD_MEMBERS, bot permissions, independent verifier), a "no authorized guild is
  configured" item, and `persistent_storage`. No credential item appears, because the credential name itself is
  unset.
