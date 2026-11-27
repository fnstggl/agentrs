# Workstream ES — wiring notes

```python
from cam_live.harness.http_api import HttpApiExecutor, ApiModality
from cam_live.harness.discord import DiscordExecutor
from cam_live.harness.publishing import PublishingExecutor
from cam_live.harness.skills import SkillExecutor, SkillLibrary
from cam_live.harness.modality import ModalitySelector, ExecContextFeatures, Realisation, default_kind

discord = DiscordExecutor.from_task(task, secrets=broker_handle, credential_probe=lambda name: <secret exists>,
                                    state_dir=<private directory for the member snapshot>)
publishing = PublishingExecutor.from_resources(task.resources, state_dir, discord=discord)
env.register(ApiModality([HttpApiExecutor(), publishing, discord]))     # one executor for modality "api"
skills = SkillExecutor(SkillLibrary(state_dir, clock))                  # steps go through ctx.dispatch
env.register(skills); skills.inducer.set_tools(env.discover_tools())
# after every env.execute(action, authorization) -> result:
skills.observe_execution(action, result)                                # material for skill.induce
```

* `TaskSpec.resources`: `discord: {guild_id, test_account_ids, token_secret?, base_url?, scope?, state_dir?,
  channels: {name: id}}`, `owned_sites: [{name, root, base_url?, access_log?}]`.
  `TaskSpec.data_permissions["discord_message_content"]`. The flat task-file keys `resources.discord_guild_id` and
  `resources.discord_bot_credential_name` are read when `discord.guild_id` / `discord.token_secret` are absent.
  * `scope`: the Discord permission names the operator lets CAM use (default: `discord_perms.DEFAULT_SCOPE`, the
    18 permissions of integer 2269727275740401). A tool needing a permission outside it is `available=False`
    ("outside the operator's permission scope: needs ...") and refused at execution (`DENIED`,
    `MISSING_PERMISSION`), whatever the bot's role allows. An unknown name is a `ValueError` at construction.
  * `state_dir` (or the constructor argument; else `<ExecContext.run_dir>/discord_state`; else memory only):
    where `discord.get_member_events` keeps its pseudonymous member snapshot (`member_snapshot_<hash>.json`,
    0600, atomic). It must not be readable by the learner uid if pseudonyms are to stay out of its reach.
* Operations put to `ExecContext.check` (all deterministic in their arguments, so an approval can be bound to
  them): `http.request {method, url, host, hop, body_sha256, body_bytes, auth_secret}`; `publish.write
  {destination (URL | file:/abs | channel:discord:<id>), destination_name, kind, location, url, content,
  content_sha256, bytes, draft_id, revision}`; `publish.get_analytics`; `discord.<tool> {guild_id, channel_id?,
  content?, content_sha256?, ...}`; `discord.read_messages` (only when message text is retained);
  the administration tools put their own name and arguments: `discord.get_identity`, `discord.get_permissions
  {channel_id?}`, `discord.get_guild_settings`, `discord.modify_guild {name?, description?, verification_level?,
  default_message_notifications?, explicit_content_filter?, system_channel_id?, rules_channel_id?,
  public_updates_channel_id?, preferred_locale?}`, `discord.modify_channel {channel_id, name?, topic?, position?,
  parent_id?, nsfw?, rate_limit_per_user?}`, `discord.delete_channel {channel_id}`,
  `discord.set_channel_permissions {channel_id, role_id ("everyone" | role id), allow?, deny?}`,
  `discord.list_roles`, `discord.create_role {name, permissions?, color?, hoist?, mentionable?}`,
  `discord.modify_role {role_id, ...}`, `discord.delete_role {role_id}`, `discord.delete_message {channel_id,
  message_id}` (and a second check of the same name with `own: false, author_is_bot` when the message is another
  author's), `discord.pin_message` / `discord.unpin_message {channel_id, message_id}`, `discord.delete_invite
  {code}`, `discord.get_member_events {reads, aggregate_only: true}`, `discord.get_audit_summary {limit?, reads,
  aggregate_only: true}`, `discord.listen_events {seconds, max_events?, reads, intents: [...], aggregate_only}`
  (followed by `discord.read_messages {content_retained: true}` under the content permission). Every check also
  carries `guild_id` and `environment`. The permission pre-check (roles, the bot's own member object) and the
  channel-membership lookup add no operation names beyond `discord.list_channels`;
  `<step operation> <bound arguments>` with `note="skill:<id>:v<n>:<step>"` for every skill step.
* Approval: a refused publication keeps its draft (`pending_approval` / `rejected`); executing the same
  `publish.submit` action again after the operator approved publishes it.
* Modality selection is advisory bookkeeping for the decision-maker: `selector.choose(intent, [Realisation(kind,
  action)...], ExecContextFeatures.of(action, observation))`, then `selector.update_choice(choice, ctx, result)`.
  A blocked choice (`choice.blocked`) means the intent was refused: do not execute any realisation.
  `selector.authorization_changed(intent)` is for the runtime to call when the authorization service reports a new
  grant; nothing else lifts a block. `selector.quantities(ctx)` / `selector.annotate(evaluations)` expose the
  estimates to the learner.
* **Secret binding.** The token is requested from the broker with the REST base URL as destination for REST
  calls and with the URL returned by `GET /gateway/bot` for the Gateway window: the secret must be bound to both
  (`discord.com` and `gateway.discord.gg`; `harness.credentials` binds `discord_bot_token` to both by default),
  and in live mode the authorization record must list both under `read`. A token not released for the gateway
  destination is a `DENIED` result saying so; nothing connects.
* `discord.missing_capabilities()` -> `[{capability, reason, how_to_grant}]` for the operator-request file: tools
  outside the scope, permissions the bot lacks, missing gateway intents, bot not in the guild, no token, secret not
  bound to the gateway. It contacts nothing: scope / token / guild come from the configuration, the rest from
  what executed actions learned (`get_identity`, `get_permissions`, any pre-check, `listen_events`).
* **Authorization policy (EA) as it stands.** `discord.get_*` / `discord.list_*` are classified as reads. Every
  other `discord.*` name — the administration writes and also `discord.listen_events` — falls under
  `T-DISCORD-WRITE` (external write, kind `publish`): it needs the server as an owned destination and an approval
  unless the destination is pre-authorized, and counts against the write rate (10 per hour and destination by
  default, which is fewer than `tools/discord_live_check.py` makes).
* Staging: `StagingDiscordServer().base_url` is a drop-in `base_url`; `server.state` has the hooks
  (`add_human`, `add_bot`, `add_test_account`, `remove_member`, `update_member`, `use_invite`,
  `post_member_message`, `add_role`, `set_bot_permissions`, `grant_bot`, `revoke_bot`, `make_bot_administrator`,
  `set_everyone_permissions`, `set_overwrite`, `remove_bot_from_guild`, `add_other_guild`, `add_foreign_invite`,
  `set_permission` (the service refuses although the roles allow), ...). `server.gateway` is the staging Gateway
  (`GET /gateway/bot` returns its `ws://127.0.0.1:<port>`): state changes are dispatched to connected sessions, and
  its hooks are `emit`, `emit_other_guild`, `send_op` (7, 9, 1), `close_sessions(code)`, `identify_response`,
  `acknowledge_heartbeats`, `on_ready`, `wait_ready`.
* A manual end-to-end check with the real environment (EB) and authorization service (EA) is in
  `docs/evidence/ES/integration_smoke.py` (output in `integration_smoke.txt`).

## Live check (`tools/discord_live_check.py`)

Runs the tools through the real `ComputerEnvironment` and `PolicyService` (in-process): every step is
`env.validate_action -> policy.authorize(action, task, {}) -> env.execute(action, decision)`. It creates a
temporary channel `cam-integration-test`, a temporary role and renames the server, and undoes all three in
`finally` blocks. **It has been run in `--staging` only** (`docs/evidence/ES/discord_live_check_staging.json`;
`..._live_policy.json` is the same sequence under the policy's live mode with a record signed for the run).

What the operator prepares for a live run (`R` = a run directory; the token is never typed on a command line):

```
cd /workspace/camlive; R=runs/cam_live/discord_check; mkdir -p $R
# 1. the bot token into R's vault (bound to discord.com, gateway.discord.gg and *.discord.gg by default)
.venv/bin/python -m cam_live.harness.credentials set discord_bot_token --run-dir $R --no-verifier --from-file <0600 file>
# 2. the task the record is signed for
cat > $R/task.yaml <<'YAML'
version: 1
objective_id: discord-live-check
mode: live
objective: "Integration check of the Discord adapter in the authorized server."
resources:
  discord:
    guild_id: "<server id>"
interfaces: [api]
limits: {duration_s: 3600, money: 0, compute_s: 0, max_external_actions: 200, max_decisions: 200}
YAML
# 3. operator key (once; the private key stays with the operator) and its public half where the service looks
.venv/bin/python -m cam_live.harness.auth.opctl init-operator-key --private <operator.key> --public $R/auth/keys/operator.pub
# 4. the authorization record: print, review, sign, register
.venv/bin/python cam_live/tools/discord_live_check.py --task $R/task.yaml --print-record-template > $R/record.json
.venv/bin/python -m cam_live.harness.auth.opctl sign-live $R/record.json --private <operator.key> > $R/record.signed.json
.venv/bin/python -c "import json,sys; from cam_live.harness.auth import PolicyService; p=PolicyService(sys.argv[1]); \
print(p.register_live_authorization(json.load(open(sys.argv[2])), 'operator')); p.close()" $R $R/record.signed.json
# 5. a policy whose write rate allows the check (default policy + two rate_limits.per_destination entries)
.venv/bin/python cam_live/tools/discord_live_check.py --task $R/task.yaml --write-policy $R/policy.yaml
# 6. the check (add --skip-writes for the read-only part; --await-reply-s N to wait for a person's reply)
.venv/bin/python cam_live/tools/discord_live_check.py --run-dir $R --task $R/task.yaml --policy $R/policy.yaml \
    --await-reply-s 120 --out $R/discord_live_check.json
```

The record printed in step 4 names the server as account (`account:discord:<id>`), owns `channel:discord:*`,
pre-authorizes both (so that no step waits for an approval), lets `discord.com` and `gateway.discord.gg` be read,
sets `max_external_actions` / `max_decisions` to 200 and `duration_s` to 3600, and is valid for one hour from the
moment it was printed. If the registration is done through a running authorization service instead, use `opctl
--run-dir $R register-live`. Exit codes: 0 every step the scope allows succeeded, 1 a step failed, 2 setup error,
3 the bot is not a member of the server (stops after `get_identity`), 4 the policy's write rate is too low
(stops before any write), 5 a step waits for approval (`--approve-wait-s N` re-issues it for N seconds), 6 a vault
secret was found in the evidence (written redacted). The preparation above is exercised on a made-up token and
server id, without executing any action, in `test_es_discord_live_check.py`.
