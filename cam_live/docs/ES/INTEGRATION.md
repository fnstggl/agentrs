# Workstream ES — wiring notes

```python
from cam_live.harness.http_api import HttpApiExecutor, ApiModality
from cam_live.harness.discord import DiscordExecutor
from cam_live.harness.publishing import PublishingExecutor
from cam_live.harness.skills import SkillExecutor, SkillLibrary
from cam_live.harness.modality import ModalitySelector, ExecContextFeatures, Realisation, default_kind

discord = DiscordExecutor.from_task(task, secrets=broker_handle, credential_probe=lambda name: <secret exists>)
publishing = PublishingExecutor.from_resources(task.resources, state_dir, discord=discord)
env.register(ApiModality([HttpApiExecutor(), publishing, discord]))     # one executor for modality "api"
skills = SkillExecutor(SkillLibrary(state_dir, clock))                  # steps go through ctx.dispatch
env.register(skills); skills.inducer.set_tools(env.discover_tools())
# after every env.execute(action, authorization) -> result:
skills.observe_execution(action, result)                                # material for skill.induce
```

* `TaskSpec.resources`: `discord: {guild_id, test_account_ids, token_secret?, base_url?, channels: {name: id}}`,
  `owned_sites: [{name, root, base_url?, access_log?}]`. `TaskSpec.data_permissions["discord_message_content"]`.
* Operations put to `ExecContext.check` (all deterministic in their arguments, so an approval can be bound to
  them): `http.request {method, url, host, hop, body_sha256, body_bytes, auth_secret}`; `publish.write
  {destination (URL | file:/abs | channel:discord:<id>), destination_name, kind, location, url, content,
  content_sha256, bytes, draft_id, revision}`; `publish.get_analytics`; `discord.<tool> {guild_id, channel_id?,
  content?, content_sha256?, ...}`; `discord.read_messages` (only when message text is retained);
  `<step operation> <bound arguments>` with `note="skill:<id>:v<n>:<step>"` for every skill step.
* Approval: a refused publication keeps its draft (`pending_approval` / `rejected`); executing the same
  `publish.submit` action again after the operator approved publishes it.
* Modality selection is advisory bookkeeping for the decision-maker: `selector.choose(intent, [Realisation(kind,
  action)...], ExecContextFeatures.of(action, observation))`, then `selector.update_choice(choice, ctx, result)`.
  A blocked choice (`choice.blocked`) means the intent was refused: do not execute any realisation.
  `selector.authorization_changed(intent)` is for the runtime to call when the authorization service reports a new
  grant; nothing else lifts a block. `selector.quantities(ctx)` / `selector.annotate(evaluations)` expose the
  estimates to the learner.
* Staging: `StagingDiscordServer().base_url` is a drop-in `base_url`; `server.state` has the hooks
  (`add_human`, `add_bot`, `add_test_account`, `remove_member`, `use_invite`, `post_member_message`, ...).
* A manual end-to-end check with the real environment (EB) and authorization service (EA) is in
  `docs/evidence/ES/integration_smoke.py` (output in `integration_smoke.txt`).
