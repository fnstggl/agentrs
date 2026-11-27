# Discord: live integration check on the operator's server (2026-11-27)

`live_check_20261127.json` — `tools/discord_live_check.py` run at 2026-11-27 ~05:53–06:00 UTC against the real
Discord API (`https://discord.com/api/v10` and the Gateway) in the operator-designated server "CAM's lab"
(id 1575680989060010034), through the real `ComputerEnvironment` and `PolicyService` in **live** mode with an
operator-signed authorization record valid for one hour (task `discord-live-check`, data permissions
`aggregate_metrics` + `member_list`). Executed by the master agent at the operator's explicit request
("perform the authorized integration tests now"). This is an integration test of the adapter, not the experiment:
CAM's learner was not involved and nothing public-facing was published beyond the temporary test channel.

Result: 50 of 50 steps `ok`, exit code 0. Identity, permissions (18 role permissions; 37 effective because of
@everyone, scope enforced by the adapter), settings, channels, counts, invites; create temporary channel, post,
read back (counts only), poll 120 s for a human reply (none arrived), edit, pin/unpin, create/list/delete invite,
modify channel, deny SEND_MESSAGES for @everyone, create/modify/delete a permissionless role, rename the server and
restore its name, membership baseline + change reading, a 10 s Gateway window (READY, GUILD_CREATE), audit-log
summary, delete the temporary channel. A read-only check afterwards found the server as before: name "CAM’s lab",
channels Text Channels / Voice Channels / general / General, roles @everyone / CAM Live, invites 8fGfBMu24x and
cHU5tpF4e. The evidence file was checked to contain no vault secret.

`../ES/discord_live_check_staging*.json` — the same sequence against the local staging services in staging and
live policy mode (27/27 each) with the post-fix authorization layer.
