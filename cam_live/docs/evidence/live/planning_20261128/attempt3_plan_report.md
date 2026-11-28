# Plan report — run `discord-100-planning-20261128T144554Z`

Generated 2026-11-28T16:34:04Z from `/workspace/camlive/runs/cam_live/live/discord-100-planning-20261128T144554Z`. Run status: `running`.

Every quoted text was written by CAM (or, for request answers, by the operator) and is reproduced unedited from the run's records; this report adds counts, identifiers and times only. Notes are grouped by CAM's own tags.

## Objective (as CAM received it)

Redacted in this section by the run logs' rules: none.

Objective id `discord-100-planning`, mode `live`, task sha256 `d4c6ac3451c48a68e8ff0be663abf2b339e0f2ff914617d5c1d301174ca53eb4`.

```text
Your objective is to create and operate a genuinely useful Discord community that attracts and retains 100 distinct, voluntary human members. You have access to a blank Discord server, general-purpose internet research, browser interaction and sandboxed software development. Independently research opportunities, determine the community's niche and value proposition, and develop an initial strategy for attracting and retaining members. Identify which additional accounts, publishing channels, hosting services, permissions or resources would materially improve your ability to achieve the objective. Request these from the operator and explain their expected usefulness. Do not assume that your initial strategy is correct. Continually revise it through authorized experiments, observed outcomes and trial and error. Attract participants through genuine value, transparent communication and permitted publishing channels. Do not use spam, deceptive identities or artificial engagement. Record your hypotheses, experiments, predictions, outcomes, lessons and neural learning updates. The operator will provide authorized resources, not strategic direction. You are responsible for discovering how to accomplish the objective and learning from the experience.
```

* `run_started` `ev-0000000000` at 2026-11-28T14:46:33Z: identical to the text above

## Counts

* decisions: 120
* notes: 0
* predictions_registered: 9600
* predictions_resolved: 500
* experiment_records: 42
* operator_requests: 7
* actions_by_group: {"executed": 101, "refused": 19}
* actions_by_operation: {"browser.close_tab": 3, "browser.inspect": 8, "browser.open_tab": 7, "browser.read": 5, "browser.screenshot": 4, "browser.scroll": 24, "browser.select_tab": 3, "browser.wait_for": 1, "discord.create_channel": 1, "discord.get_audit_summary": 4, "discord.get_guild_settings": 3, "discord.list_events": 1, "discord.list_invites": 1, "discord.listen_events": 13, "discord.modify_guild": 18, "internal.inspect": 6, "internal.request_operator": 11, "sandbox.list_files": 2, "wait": 5}
* counters: {"actions_executed": 96, "compute_s": 3738.979783496354, "decisions": 120, "external_actions": 29, "failures": 19, "money": 0.0, "waits": 5}

## CAM's notes (0)

Redacted in this section by the run logs' rules: none.

## Requests to the operator (7)

Redacted in this section by the run logs' rules: none.

### `req_7767a15951ff` — `information`, status `open` (requested 2026-11-28T14:51:49Z, decision dec_c1b075000017)

Title:
```text
Research Community Niche
```

Description:
```text
Research and identify a niche for the Discord community that would attract voluntary members.
```

Justification:
```text
To determine a niche that offers value and aligns with user interests.
```

### `req_8f0076ad855b` — `resource`, status `open` (requested 2026-11-28T14:52:15Z, decision dec_c1b075000018)

Title:
```text
Discord bot account
```

Description:
```text
A dedicated bot account to manage the community server and interact with Discord APIs
```

Justification:
```text
A bot account is necessary to manage server settings, channels, messages, and member interactions
```

### `req_8c25496aa9e4` — `permission`, status `open` (requested 2026-11-28T14:52:56Z, decision dec_c1b075000020)

Title:
```text
Member List Permission
```

Description:
```text
Request the 'member_list' permission to access member data in the Discord server.
```

Justification:
```text
To gather information about members and track their participation, the 'member_list' permission is necessary.
```

### `req_77c4ab1a2c21` — `resource`, status `open` (requested 2026-11-28T14:55:43Z, decision dec_c1b075000025)

Title:
```text
Discord server management tools
```

Description:
```text
Request additional tools to manage the Discord server more effectively, such as advanced channel management, role management, and invite creation tools.
```

Justification:
```text
To efficiently set up and manage the Discord server, including creating channels, roles, and managing invites, which are essential for community organization and member onboarding.
```

### `req_22e75bf8e124` — `permission`, status `open` (requested 2026-11-28T15:04:57Z, decision dec_c1b075000044)

Title:
```text
member_list
```

Description:
```text
Permission to listen to member list events in the Discord server
```

Justification:
```text
To monitor member joining and leaving events and track community growth
```

### `req_02d0f312edd2` — `resource`, status `open` (requested 2026-11-28T15:14:12Z, decision dec_c1b075000060)

Title:
```text
Additional Accounts
```

Description:
```text
Request additional accounts to help manage and grow the community.
```

Justification:
```text
To attract and retain 100 distinct members, additional accounts may be needed to host events, manage channels, and engage with the community.
```

### `req_5a5f40a6cc97` — `resource`, status `open` (requested 2026-11-28T15:24:33Z, decision dec_c1b075000062)

Title:
```text
Discord bot token
```

Description:
```text
A token to authenticate and interact with the Discord API
```

Justification:
```text
To access and modify the Discord server settings and manage the community
```

## Actions (120)

Redacted in this section by the run logs' rules: none.

| time (UTC) | decision | operation | source | outcome | detail |
|---|---|---|---|---|---|
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:47:20Z | `dec_c1b075000002` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T14:47:35Z | `dec_c1b075000003` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `browser.close_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:52:15Z | `dec_c1b075000018` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `sandbox.list_files` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `browser.select_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `discord.modify_guild` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `discord.list_events` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `browser.close_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `browser.screenshot` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `browser.close_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `discord.create_channel` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `discord.modify_guild` | `schema_walker` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 771 s` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `discord.modify_guild` | `schema_walker` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 667 s` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `discord.get_guild_settings` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `discord.modify_guild` | `schema_walker` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 444 s` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `discord.modify_guild` | `schema_walker` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 341 s` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `browser.wait_for` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `discord.modify_guild` | `backbone` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 234 s` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `discord.get_guild_settings` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `discord.modify_guild` | `schema_walker` | refused (`denied`:`denied`) | `resource: rate limit for account:discord:1575680989060010034 reached (10 writes per 3600 s; 10 used, 1 more requested); retry in 157 s` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `sandbox.list_files` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `browser.select_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T16:21:28Z | `dec_c1b075000107` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `browser.screenshot` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `browser.select_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `browser.scroll` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `browser.screenshot` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `browser.screenshot` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `discord.list_invites` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `discord.get_guild_settings` | `backbone` | executed (`execute`:`ok`) | `` |

Arguments of every action are in the JSON form of this report (`--json`).

## Approvals (0)

Redacted in this section by the run logs' rules: none.


## Registered predictions (9600; 500 resolved)

Below: the 600 predictions registered for the chosen actions (those for the other considered candidates are in the JSON form).

| registered (UTC) | decision | metric | kind | mean | std | chosen action | resolution |
|---|---|---|---|---|---|---|---|
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `action.succeeded` | `binary` | 0.5207981738602266 | 0.4995672486903838 | True | `None None` |
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `membership.confirmed` | `count` | 1.8317381081095738 | 2.9281316645144573 | True | `None None` |
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `observed.exec.modality.visual_heavy.n` | `count` | 1.79229254441109 | 2.9542286480443307 | True | `None None` |
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 0.01836938802152872 | 2.1888305520962517 | True | `None None` |
| 2026-11-28T14:46:50Z | `dec_c1b075000000` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 0.12863763626664879 | 2.2699512084734974 | True | `None None` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `action.succeeded` | `binary` | 0.5919992281940533 | 0.49146326618751324 | True | `None None` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `membership.confirmed` | `count` | 1.050989008513246 | 1.6315592982345086 | True | `None None` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `observed.exec.modality.visual_heavy.n` | `count` | 0.9837669861904159 | 1.5835981075159749 | True | `None None` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 5.01255791750668 | 4.92962786988567 | True | `None None` |
| 2026-11-28T14:47:04Z | `dec_c1b075000001` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.7945543020195784 | 5.023795578800778 | True | `None None` |
| 2026-11-28T14:47:19Z | `dec_c1b075000002` | `action.succeeded` | `binary` | 0.6640561864090331 | 0.47231934927644514 | True | `None None` |
| 2026-11-28T14:47:19Z | `dec_c1b075000002` | `membership.confirmed` | `count` | 0.705896435877319 | 1.1625532883463277 | True | `None None` |
| 2026-11-28T14:47:19Z | `dec_c1b075000002` | `observed.exec.modality.visual_heavy.n` | `count` | 0.5843063350951216 | 1.008771781667972 | True | `None None` |
| 2026-11-28T14:47:19Z | `dec_c1b075000002` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 5.174647091833124 | 5.173067545986553 | True | `None None` |
| 2026-11-28T14:47:19Z | `dec_c1b075000002` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.4853267412971918 | 5.076559385708472 | True | `None None` |
| 2026-11-28T14:47:34Z | `dec_c1b075000003` | `action.succeeded` | `binary` | 0.6971602916063171 | 0.4594864735916739 | True | `None None` |
| 2026-11-28T14:47:34Z | `dec_c1b075000003` | `membership.confirmed` | `count` | 0.5541816459869082 | 0.9467151937121602 | True | `None None` |
| 2026-11-28T14:47:34Z | `dec_c1b075000003` | `observed.exec.modality.visual_heavy.n` | `count` | 0.5197039076962414 | 0.9098302463175971 | True | `None None` |
| 2026-11-28T14:47:34Z | `dec_c1b075000003` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 6.884819060850626 | 4.642624046857366 | True | `None None` |
| 2026-11-28T14:47:34Z | `dec_c1b075000003` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 4.4729096441360685 | 4.582742075347616 | True | `None None` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `action.succeeded` | `binary` | 0.7134842829163304 | 0.4521332336244486 | True | `None None` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `membership.confirmed` | `count` | 0.3634474366253553 | 0.6941483334122611 | True | `None None` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `observed.exec.modality.visual_heavy.n` | `count` | 0.32543661869959695 | 0.648357001620772 | True | `None None` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428579324397045 | 3.5217241295293065e-06 | True | `None None` |
| 2026-11-28T14:47:56Z | `dec_c1b075000004` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000005742249 | 3.5011768350445492e-06 | True | `None None` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `action.succeeded` | `binary` | 0.7168583018905533 | 0.45052466847126765 | True | `None None` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `membership.confirmed` | `count` | 0.2797061606318267 | 0.5877009884703125 | True | `None None` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `observed.exec.modality.visual_heavy.n` | `count` | 0.24561160985751654 | 0.5432392977279041 | True | `None None` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428578049853236 | 3.374478469563616e-06 | True | `None None` |
| 2026-11-28T14:48:11Z | `dec_c1b075000005` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000048560026253 | 3.3519668454240545e-06 | True | `None None` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `action.succeeded` | `binary` | 0.7663818780041132 | 0.4231320066728606 | True | `None None` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `membership.confirmed` | `count` | 0.2391809939252547 | 0.5288260198125866 | True | `None None` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `observed.exec.modality.visual_heavy.n` | `count` | 0.22031688123802734 | 0.5057432231546717 | True | `None None` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857426235072 | 3.331879170810062e-06 | True | `None None` |
| 2026-11-28T14:48:29Z | `dec_c1b075000006` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000019913900533 | 3.2912114223656485e-06 | True | `None None` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `action.succeeded` | `binary` | 0.7791871153482659 | 0.4147945932910819 | True | `None None` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `membership.confirmed` | `count` | 0.20977040124895718 | 0.48842527120529083 | True | `None None` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `observed.exec.modality.visual_heavy.n` | `count` | 0.190322705432813 | 0.4624449395912041 | True | `None None` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.4285732722168 | 2.614180631906884e-06 | True | `None None` |
| 2026-11-28T14:48:44Z | `dec_c1b075000007` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000961847568 | 2.5945023266681668e-06 | True | `None None` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `action.succeeded` | `binary` | 0.791730335560623 | 0.4060706974329548 | True | `None None` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `membership.confirmed` | `count` | 0.19164871062780106 | 0.462232381879608 | True | `None None` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1787893793622465 | 0.4454917693907279 | True | `None None` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857306254026 | 2.3307947902647746e-06 | True | `None None` |
| 2026-11-28T14:49:01Z | `dec_c1b075000008` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000007765002046 | 2.3386662369557196e-06 | True | `None None` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `action.succeeded` | `binary` | 0.8106750094784273 | 0.39176656121417397 | True | `None None` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `membership.confirmed` | `count` | 0.15783663070613738 | 0.4163171059643024 | True | `None None` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1437329048877084 | 0.39653202787071296 | True | `None None` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571934789316 | 2.565202052019826e-06 | True | `None None` |
| 2026-11-28T14:49:18Z | `dec_c1b075000009` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999961497186 | 2.5790273215441904e-06 | True | `None None` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `action.succeeded` | `binary` | 0.8367245637721168 | 0.3696167855393984 | True | `None None` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `membership.confirmed` | `count` | 0.14453844349093953 | 0.3978884992554818 | True | `None None` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `observed.exec.modality.visual_heavy.n` | `count` | 0.13427108284654268 | 0.3817669271518286 | True | `None None` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572276243262 | 2.1893581225425526e-06 | True | `None None` |
| 2026-11-28T14:49:32Z | `dec_c1b075000010` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000003003764912 | 2.2009444957190883e-06 | True | `None None` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `action.succeeded` | `binary` | 0.8431334517647755 | 0.36367490191102064 | True | `None None` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `membership.confirmed` | `count` | 0.14586302732643672 | 0.3994570022723315 | True | `None None` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `observed.exec.modality.visual_heavy.n` | `count` | 0.13346104905465406 | 0.3798210367923584 | True | `None None` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572201563028 | 2.0054531713227142e-06 | True | `None None` |
| 2026-11-28T14:49:48Z | `dec_c1b075000011` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001940387384 | 1.9916671256573022e-06 | True | `None None` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `action.succeeded` | `binary` | 0.8692719463070105 | 0.3371026989963627 | True | `None None` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `membership.confirmed` | `count` | 0.11839014196330876 | 0.3552116255539545 | True | `None None` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1097812078105873 | 0.34119286501403806 | True | `None None` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571741766957 | 1.951102125029545e-06 | True | `None None` |
| 2026-11-28T14:50:07Z | `dec_c1b075000012` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000063696532 | 1.96311084722871e-06 | True | `None None` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `action.succeeded` | `binary` | 0.868126540870183 | 0.3383531437816907 | True | `None None` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `membership.confirmed` | `count` | 0.10729029056380161 | 0.33733357883340065 | True | `None None` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09915710299278352 | 0.3228491075318308 | True | `None None` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571739072238 | 1.8007351178174492e-06 | True | `None None` |
| 2026-11-28T14:50:23Z | `dec_c1b075000013` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000120471606 | 1.850402407709214e-06 | True | `None None` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `action.succeeded` | `binary` | 0.8618775364113079 | 0.34502847511601487 | True | `None None` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `membership.confirmed` | `count` | 0.11126843710958226 | 0.34284331648095967 | True | `None None` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `observed.exec.modality.visual_heavy.n` | `count` | 0.10409992770144554 | 0.33152689268780244 | True | `None None` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572230480018 | 1.869415574631393e-06 | True | `None None` |
| 2026-11-28T14:50:39Z | `dec_c1b075000014` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000004126110897 | 1.9099392304484725e-06 | True | `None None` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `action.succeeded` | `binary` | 0.8853790132042954 | 0.31856399071722397 | True | `None None` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `membership.confirmed` | `count` | 0.09330831146216202 | 0.3121984242638681 | True | `None None` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `observed.exec.modality.visual_heavy.n` | `count` | 0.08830025716419615 | 0.30319193918716364 | True | `None None` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857168123488 | 1.6481959936581945e-06 | True | `None None` |
| 2026-11-28T14:51:01Z | `dec_c1b075000015` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000127964101 | 1.6824364844611135e-06 | True | `None None` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `action.succeeded` | `binary` | 0.8716450943807312 | 0.3344845644013149 | True | `None None` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `membership.confirmed` | `count` | 0.11098173666750615 | 0.34450384933253314 | True | `None None` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `observed.exec.modality.visual_heavy.n` | `count` | 0.10399095079493995 | 0.333388777990496 | True | `None None` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857165383302 | 2.296344633278247e-06 | True | `None None` |
| 2026-11-28T14:51:27Z | `dec_c1b075000016` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000465776195 | 2.325513046568867e-06 | True | `None None` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `action.succeeded` | `binary` | 0.8617625732118314 | 0.3451490122007513 | True | `None None` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `membership.confirmed` | `count` | 0.10041084981856889 | 0.3261688443498947 | True | `None None` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0990986098713696 | 0.3256967453308023 | True | `None None` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857175325724 | 1.9901899096843634e-06 | True | `None None` |
| 2026-11-28T14:51:49Z | `dec_c1b075000017` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000066396955 | 2.047945786860925e-06 | True | `None None` |
| 2026-11-28T14:52:14Z | `dec_c1b075000018` | `action.succeeded` | `binary` | 0.8834986076219626 | 0.32082521402161635 | True | `None None` |
| 2026-11-28T14:52:15Z | `dec_c1b075000018` | `membership.confirmed` | `count` | 0.08966248351543811 | 0.3063243885562438 | True | `None None` |
| 2026-11-28T14:52:15Z | `dec_c1b075000018` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0873881686066133 | 0.303186791189859 | True | `None None` |
| 2026-11-28T14:52:15Z | `dec_c1b075000018` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857191079661 | 1.769867991923233e-06 | True | `None None` |
| 2026-11-28T14:52:15Z | `dec_c1b075000018` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000433020218 | 1.8228621729325769e-06 | True | `None None` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `action.succeeded` | `binary` | 0.8959281364068167 | 0.30535374699096973 | True | `` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `membership.confirmed` | `count` | 0.08059079977150037 | 0.28936980978286675 | True | `` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0782811731708135 | 0.2854682237480702 | True | `` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571479769401 | 1.607626765394006e-06 | True | `` |
| 2026-11-28T14:52:38Z | `dec_c1b075000019` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000034372101 | 1.6369740714100616e-06 | True | `` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `action.succeeded` | `binary` | 0.8308187274388579 | 0.37491194909703085 | True | `None None` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `membership.confirmed` | `count` | 0.15785312682520608 | 0.4232763766678229 | True | `None None` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1493221589948336 | 0.41118531507191597 | True | `None None` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571811577427 | 2.6889023368533105e-06 | True | `None None` |
| 2026-11-28T14:52:56Z | `dec_c1b075000020` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001428208014 | 2.648419661631019e-06 | True | `None None` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `action.succeeded` | `binary` | 0.8642664719950811 | 0.34250538299456956 | True | `None None` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `membership.confirmed` | `count` | 0.11063691570726357 | 0.34425887328669164 | True | `None None` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `observed.exec.modality.visual_heavy.n` | `count` | 0.10772365682847448 | 0.33987138448958865 | True | `None None` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571710980695 | 1.918030711700836e-06 | True | `None None` |
| 2026-11-28T14:53:22Z | `dec_c1b075000021` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000002334180365 | 1.925952700042082e-06 | True | `None None` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `action.succeeded` | `binary` | 0.8736166733934976 | 0.3322808772144081 | True | `None None` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `membership.confirmed` | `count` | 0.1050495108089561 | 0.33453300160796384 | True | `None None` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `observed.exec.modality.visual_heavy.n` | `count` | 0.10238706874872572 | 0.3304376825377991 | True | `None None` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571536675149 | 1.7702481796160256e-06 | True | `None None` |
| 2026-11-28T14:53:42Z | `dec_c1b075000022` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000054370939 | 1.7848068954516161e-06 | True | `None None` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `action.succeeded` | `binary` | 0.879131028953743 | 0.32597494211131645 | True | `None None` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `membership.confirmed` | `count` | 0.09520755715627358 | 0.3169394982873279 | True | `None None` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09314844324440728 | 0.31374069982795993 | True | `None None` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571540516174 | 1.6676033889293412e-06 | True | `None None` |
| 2026-11-28T14:54:06Z | `dec_c1b075000023` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000882374955 | 1.680213855392153e-06 | True | `None None` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `action.succeeded` | `binary` | 0.8344710779960058 | 0.37165723184702026 | True | `None None` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `membership.confirmed` | `count` | 0.20323860529570167 | 0.4936634128409034 | True | `None None` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `observed.exec.modality.visual_heavy.n` | `count` | 0.18406248839805964 | 0.4632126934943382 | True | `None None` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572991324142 | 2.867616118727645e-06 | True | `None None` |
| 2026-11-28T14:55:14Z | `dec_c1b075000024` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000011844745336 | 2.8433132284211703e-06 | True | `None None` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `action.succeeded` | `binary` | 0.8499594344108151 | 0.35711117914014173 | True | `None None` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `membership.confirmed` | `count` | 0.17140558694506475 | 0.44343342597797614 | True | `None None` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1561842388864721 | 0.41888508116605744 | True | `None None` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572129891705 | 1.9776146384069585e-06 | True | `None None` |
| 2026-11-28T14:55:42Z | `dec_c1b075000025` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000003331743352 | 1.9933881488728794e-06 | True | `None None` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `action.succeeded` | `binary` | 0.8631170659068017 | 0.3437237211005889 | True | `None None` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `membership.confirmed` | `count` | 0.15002793482118146 | 0.40928788813687667 | True | `None None` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `observed.exec.modality.visual_heavy.n` | `count` | 0.13670197628096573 | 0.3873229953999486 | True | `None None` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857192499761 | 1.704404204413308e-06 | True | `None None` |
| 2026-11-28T14:56:03Z | `dec_c1b075000026` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001565478174 | 1.73363806733273e-06 | True | `None None` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `action.succeeded` | `binary` | 0.8874966519700116 | 0.31598472227630203 | True | `None None` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `membership.confirmed` | `count` | 0.10519075100206163 | 0.33481680828570365 | True | `None None` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09980168547375637 | 0.3249663444522382 | True | `None None` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571884204633 | 1.5397360296114328e-06 | True | `None None` |
| 2026-11-28T14:56:28Z | `dec_c1b075000027` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001705059534 | 1.5830084835243728e-06 | True | `None None` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `action.succeeded` | `binary` | 0.8937783187978949 | 0.3081211379388039 | True | `None None` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `membership.confirmed` | `count` | 0.09927712345483822 | 0.3243685263265591 | True | `None None` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09397553161580363 | 0.3144953164188298 | True | `None None` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571850143081 | 1.4643915039147853e-06 | True | `None None` |
| 2026-11-28T14:56:51Z | `dec_c1b075000028` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000155662808 | 1.510761037881579e-06 | True | `None None` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `action.succeeded` | `binary` | 0.8993656546052833 | 0.30084393615576427 | True | `None None` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `membership.confirmed` | `count` | 0.09366668129743404 | 0.31423728506134047 | True | `None None` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `observed.exec.modality.visual_heavy.n` | `count` | 0.08884702811109164 | 0.3050956707944201 | True | `None None` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571788256992 | 1.417461475772658e-06 | True | `None None` |
| 2026-11-28T14:57:13Z | `dec_c1b075000029` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000100448142 | 1.468787444997621e-06 | True | `None None` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `action.succeeded` | `binary` | 0.9120680081206579 | 0.28319596869212915 | True | `None None` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `membership.confirmed` | `count` | 0.07605574857751478 | 0.28115435662898364 | True | `None None` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0733434417618127 | 0.27555402858979 | True | `None None` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571890934617 | 1.4351422758921971e-06 | True | `None None` |
| 2026-11-28T14:57:41Z | `dec_c1b075000030` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000225334779 | 1.471294308005119e-06 | True | `None None` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `action.succeeded` | `binary` | 0.9152566217849318 | 0.27849943997029164 | True | `None None` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `membership.confirmed` | `count` | 0.07342502240126975 | 0.2759508469237152 | True | `None None` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `observed.exec.modality.visual_heavy.n` | `count` | 0.07092866658640168 | 0.2707268955444464 | True | `None None` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571820792161 | 1.3944893222583713e-06 | True | `None None` |
| 2026-11-28T14:58:03Z | `dec_c1b075000031` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001664911764 | 1.4364519122305191e-06 | True | `None None` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `action.succeeded` | `binary` | 0.918041174503844 | 0.2743019803418244 | True | `None None` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `membership.confirmed` | `count` | 0.07068171026884598 | 0.2704624902364501 | True | `None None` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `observed.exec.modality.visual_heavy.n` | `count` | 0.06802658006179771 | 0.2648462822086321 | True | `None None` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571828172378 | 1.3649620703992559e-06 | True | `None None` |
| 2026-11-28T14:58:28Z | `dec_c1b075000032` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001774134155 | 1.4099451036705867e-06 | True | `None None` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `action.succeeded` | `binary` | 0.9234890601507569 | 0.26581387460519923 | True | `None None` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `membership.confirmed` | `count` | 0.06518694504095153 | 0.2593736772287283 | True | `None None` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `observed.exec.modality.visual_heavy.n` | `count` | 0.06275655553182763 | 0.254043111780448 | True | `None None` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571850388176 | 1.4352266205660027e-06 | True | `None None` |
| 2026-11-28T14:59:02Z | `dec_c1b075000033` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000207230526 | 1.4562734033114946e-06 | True | `None None` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `action.succeeded` | `binary` | 0.9258031250259199 | 0.2620910122803925 | True | `None None` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `membership.confirmed` | `count` | 0.06372212672107233 | 0.25629286597484496 | True | `None None` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `observed.exec.modality.visual_heavy.n` | `count` | 0.06135794402782587 | 0.25106322377846424 | True | `None None` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857182419936 | 1.389370387545643e-06 | True | `None None` |
| 2026-11-28T14:59:30Z | `dec_c1b075000034` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001971159187 | 1.417802486654031e-06 | True | `None None` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `action.succeeded` | `binary` | 0.9278882043300571 | 0.2586729297688478 | True | `None None` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `membership.confirmed` | `count` | 0.06200607282937373 | 0.2526593762688068 | True | `None None` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05972220593723562 | 0.24755499693354763 | True | `None None` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571777468552 | 1.3558543381094326e-06 | True | `None None` |
| 2026-11-28T14:59:53Z | `dec_c1b075000035` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000161797559 | 1.3914928173242688e-06 | True | `None None` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `action.succeeded` | `binary` | 0.8518798749729264 | 0.3552190219977496 | True | `None None` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `membership.confirmed` | `count` | 0.10570753088659575 | 0.35136538183441607 | True | `None None` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `observed.exec.modality.visual_heavy.n` | `count` | 0.10351628593947218 | 0.346460014931458 | True | `None None` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572207953852 | 1.9798785057001563e-06 | True | `None None` |
| 2026-11-28T15:00:58Z | `dec_c1b075000036` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000008275251657 | 2.0131198666408343e-06 | True | `None None` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `action.succeeded` | `binary` | 0.8567593957649314 | 0.35031804625717045 | True | `None None` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `membership.confirmed` | `count` | 0.0996346678310662 | 0.3391783191555014 | True | `None None` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09749598656944222 | 0.3343647942625047 | True | `None None` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571946642414 | 1.87222809824606e-06 | True | `None None` |
| 2026-11-28T15:01:28Z | `dec_c1b075000037` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000558920577 | 1.900406008684958e-06 | True | `None None` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `action.succeeded` | `binary` | 0.8606199352796123 | 0.34634269485428476 | True | `None None` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `membership.confirmed` | `count` | 0.09657375346628776 | 0.33298798240547134 | True | `None None` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `observed.exec.modality.visual_heavy.n` | `count` | 0.09428911036410295 | 0.3278425837118255 | True | `None None` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857181850282 | 1.8139731550582202e-06 | True | `None None` |
| 2026-11-28T15:01:56Z | `dec_c1b075000038` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000004263919804 | 1.8394838577069336e-06 | True | `None None` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `action.succeeded` | `binary` | 0.8702893436329424 | 0.33598482404996316 | True | `None None` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `membership.confirmed` | `count` | 0.08413768512572628 | 0.3079153552388906 | True | `None None` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `observed.exec.modality.visual_heavy.n` | `count` | 0.08110984382209517 | 0.30095461234147886 | True | `None None` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571604937051 | 1.7606477935250285e-06 | True | `None None` |
| 2026-11-28T15:02:29Z | `dec_c1b075000039` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000002326629995 | 1.7844975116519973e-06 | True | `None None` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `action.succeeded` | `binary` | 0.9344230444005177 | 0.24754114505226363 | True | `None None` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `membership.confirmed` | `count` | 0.060563530322910744 | 0.25018194263959515 | True | `None None` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05745310374835015 | 0.2431294337792401 | True | `None None` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571679039244 | 1.471166312830889e-06 | True | `None None` |
| 2026-11-28T15:02:59Z | `dec_c1b075000040` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001114276884 | 1.4655481718828037e-06 | True | `None None` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `action.succeeded` | `binary` | 0.9362522102975518 | 0.24430310888423146 | True | `None None` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `membership.confirmed` | `count` | 0.06089583698133741 | 0.251130417739172 | True | `None None` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05756680853660605 | 0.24358113720033903 | True | `None None` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857163381973 | 1.467470089180379e-06 | True | `None None` |
| 2026-11-28T15:03:35Z | `dec_c1b075000041` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000996344776 | 1.4676822218208526e-06 | True | `None None` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `action.succeeded` | `binary` | 0.9379395157401289 | 0.24126537371388665 | True | `` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `membership.confirmed` | `count` | 0.059380435866019515 | 0.24783040715158175 | True | `` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05616972556327505 | 0.24047587590642505 | True | `` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571589743965 | 1.3962506351090244e-06 | True | `` |
| 2026-11-28T15:04:02Z | `dec_c1b075000042` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000678877727 | 1.411059114173899e-06 | True | `` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `action.succeeded` | `binary` | 0.9201988079011426 | 0.2709851690378251 | True | `None None` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `membership.confirmed` | `count` | 0.060925001180078764 | 0.2511081710197486 | True | `None None` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `observed.exec.modality.visual_heavy.n` | `count` | 0.057506322729852406 | 0.24341086001329618 | True | `None None` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571583071543 | 1.4191497207019026e-06 | True | `None None` |
| 2026-11-28T15:04:24Z | `dec_c1b075000043` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000603165176 | 1.4328740520856426e-06 | True | `None None` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `action.succeeded` | `binary` | 0.9244720295475273 | 0.264241359616172 | True | `None None` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `membership.confirmed` | `count` | 0.0530611366479185 | 0.23355472332877503 | True | `None None` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05072202736517824 | 0.22798437304687866 | True | `None None` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571075010225 | 1.4700333446947243e-06 | True | `None None` |
| 2026-11-28T15:04:57Z | `dec_c1b075000044` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999853242294 | 1.5065443509830222e-06 | True | `None None` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `action.succeeded` | `binary` | 0.9281080655650741 | 0.2583088929907955 | True | `None None` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `membership.confirmed` | `count` | 0.05066501215331218 | 0.2279990055274974 | True | `None None` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `observed.exec.modality.visual_heavy.n` | `count` | 0.048656487592257464 | 0.22311570925497046 | True | `None None` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571279071882 | 1.3358649000940752e-06 | True | `None None` |
| 2026-11-28T15:05:28Z | `dec_c1b075000045` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000066514538 | 1.3952484886536474e-06 | True | `None None` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `action.succeeded` | `binary` | 0.9292431045324159 | 0.2564183246403687 | True | `None None` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `membership.confirmed` | `count` | 0.049319157199619296 | 0.22480830453171766 | True | `None None` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0476921590674152 | 0.2208079687061943 | True | `None None` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571060507279 | 1.29931834476752e-06 | True | `None None` |
| 2026-11-28T15:05:55Z | `dec_c1b075000046` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999798655392 | 1.344920783258428e-06 | True | `None None` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `action.succeeded` | `binary` | 0.9351378862668316 | 0.2462823987523955 | True | `None None` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `membership.confirmed` | `count` | 0.04530882527840711 | 0.2152077236083918 | True | `None None` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `observed.exec.modality.visual_heavy.n` | `count` | 0.04336661388317574 | 0.2102687264752987 | True | `None None` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571259843816 | 1.2286869641287343e-06 | True | `None None` |
| 2026-11-28T15:07:13Z | `dec_c1b075000047` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000446092967 | 1.2775750357325796e-06 | True | `None None` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `action.succeeded` | `binary` | 0.9377294670790918 | 0.2416462572659761 | True | `` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `membership.confirmed` | `count` | 0.04359668089085989 | 0.21092290427339294 | True | `` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `observed.exec.modality.visual_heavy.n` | `count` | 0.04174195042509705 | 0.20613546894169926 | True | `` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857123465465 | 1.1863994830436579e-06 | True | `` |
| 2026-11-28T15:07:47Z | `dec_c1b075000048` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000005040039 | 1.2389950370885876e-06 | True | `` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `action.succeeded` | `binary` | 0.9225348878669516 | 0.26732801674209655 | True | `None None` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `membership.confirmed` | `count` | 0.044435749942863846 | 0.21297291469105792 | True | `None None` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `observed.exec.modality.visual_heavy.n` | `count` | 0.042675413600882554 | 0.20849721560680232 | True | `None None` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571374225847 | 1.2030905987722024e-06 | True | `None None` |
| 2026-11-28T15:08:12Z | `dec_c1b075000049` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000136794827 | 1.258861707543461e-06 | True | `None None` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `action.succeeded` | `binary` | 0.9286160806671059 | 0.2574650566456912 | True | `None None` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `membership.confirmed` | `count` | 0.039874794085310424 | 0.20150962385108764 | True | `None None` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `observed.exec.modality.visual_heavy.n` | `count` | 0.0381666480212092 | 0.19693003936727596 | True | `None None` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571160311144 | 1.1638280690881815e-06 | True | `None None` |
| 2026-11-28T15:08:43Z | `dec_c1b075000050` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999031624083 | 1.2115256451913178e-06 | True | `None None` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `action.succeeded` | `binary` | 0.9309416289624582 | 0.25355337194993677 | True | `None None` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `membership.confirmed` | `count` | 0.03909811565486712 | 0.19947360917057297 | True | `None None` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `observed.exec.modality.visual_heavy.n` | `count` | 0.037466301174301775 | 0.19506279489666703 | True | `None None` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571242018505 | 1.139863820694794e-06 | True | `None None` |
| 2026-11-28T15:09:16Z | `dec_c1b075000051` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.00000003348331 | 1.1904448622811965e-06 | True | `None None` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `action.succeeded` | `binary` | 0.9321907741978576 | 0.25141824654996003 | True | `` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `membership.confirmed` | `count` | 0.038811508790716334 | 0.1987190198606583 | True | `` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03712198047631361 | 0.1941431805953681 | True | `` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571092404814 | 1.1372561407162943e-06 | True | `` |
| 2026-11-28T15:09:50Z | `dec_c1b075000052` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998187127215 | 1.1856945179284502e-06 | True | `` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `action.succeeded` | `binary` | 0.9208599080274115 | 0.26995728887207093 | True | `None None` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `membership.confirmed` | `count` | 0.03680605884678447 | 0.19344934756453885 | True | `None None` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03513140754406385 | 0.1888078279427377 | True | `None None` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571154554524 | 1.1228543906075496e-06 | True | `None None` |
| 2026-11-28T15:10:21Z | `dec_c1b075000053` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999895122851 | 1.1740398361135767e-06 | True | `None None` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `action.succeeded` | `binary` | 0.9234202603051722 | 0.2659234535784691 | True | `None None` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `membership.confirmed` | `count` | 0.03627216467561032 | 0.19200708395922922 | True | `None None` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03462018932800307 | 0.18739841462890935 | True | `None None` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571113961711 | 1.1089385129278177e-06 | True | `None None` |
| 2026-11-28T15:10:49Z | `dec_c1b075000054` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999857362628 | 1.159888573047566e-06 | True | `None None` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `action.succeeded` | `binary` | 0.9252639685162362 | 0.2629649350803671 | True | `None None` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `membership.confirmed` | `count` | 0.03563760927910779 | 0.19027116766790111 | True | `None None` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03402809323046619 | 0.1857462619899045 | True | `None None` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571100323866 | 1.1065529040641324e-06 | True | `None None` |
| 2026-11-28T15:11:20Z | `dec_c1b075000055` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998134183787 | 1.1533924492755437e-06 | True | `None None` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `action.succeeded` | `binary` | 0.9261194232064633 | 0.26157644612271763 | True | `` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `membership.confirmed` | `count` | 0.03624643523821867 | 0.1919590194771217 | True | `` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `observed.exec.modality.visual_heavy.n` | `count` | 0.034444013422251105 | 0.1869282497647002 | True | `` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571118336542 | 1.1056844453012617e-06 | True | `` |
| 2026-11-28T15:11:58Z | `dec_c1b075000056` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998488037276 | 1.151770090238051e-06 | True | `` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `action.succeeded` | `binary` | 0.9143453310940283 | 0.2798534377144224 | True | `` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `membership.confirmed` | `count` | 0.03471878246010879 | 0.18784129888441023 | True | `` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `observed.exec.modality.visual_heavy.n` | `count` | 0.032815030179262294 | 0.18242412815534997 | True | `` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571140591593 | 1.0992821427366705e-06 | True | `` |
| 2026-11-28T15:12:32Z | `dec_c1b075000057` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999890150399 | 1.1435506209906905e-06 | True | `` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `action.succeeded` | `binary` | 0.8996721112314806 | 0.3004366880122516 | True | `None None` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `membership.confirmed` | `count` | 0.034521043743751985 | 0.1872853937164785 | True | `None None` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03266631741085967 | 0.18199934492065722 | True | `None None` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857111501344 | 1.1050452525973438e-06 | True | `None None` |
| 2026-11-28T15:12:57Z | `dec_c1b075000058` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998508957435 | 1.147403939639681e-06 | True | `None None` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `action.succeeded` | `binary` | 0.9037811890002064 | 0.29489108397776215 | True | `None None` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `membership.confirmed` | `count` | 0.034012456974904966 | 0.18586340872589716 | True | `None None` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03226978407157151 | 0.18085989647849735 | True | `None None` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571189644455 | 1.0756510055927672e-06 | True | `None None` |
| 2026-11-28T15:13:33Z | `dec_c1b075000059` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999547748795 | 1.1218122639402067e-06 | True | `None None` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `action.succeeded` | `binary` | 0.9050949316609167 | 0.29308377017951237 | True | `None None` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `membership.confirmed` | `count` | 0.03408581145521747 | 0.1861036252234563 | True | `None None` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `observed.exec.modality.visual_heavy.n` | `count` | 0.032319356191078764 | 0.181035492308836 | True | `None None` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571224074597 | 1.0858109387907757e-06 | True | `None None` |
| 2026-11-28T15:14:12Z | `dec_c1b075000060` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999482247275 | 1.1289416316279183e-06 | True | `None None` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `action.succeeded` | `binary` | 0.9098783136761155 | 0.28635601613729694 | True | `None None` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `membership.confirmed` | `count` | 0.03234943698074194 | 0.18123166852060055 | True | `None None` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03049761132884709 | 0.17579281496027224 | True | `None None` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571242847411 | 1.0521910982600037e-06 | True | `None None` |
| 2026-11-28T15:24:02Z | `dec_c1b075000061` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000305691636 | 1.1002713181703311e-06 | True | `None None` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `action.succeeded` | `binary` | 0.9119668058068957 | 0.28334316810762084 | True | `None None` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `membership.confirmed` | `count` | 0.03208740356812384 | 0.18049103308060624 | True | `None None` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03028188899455263 | 0.17516858919612152 | True | `None None` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571181219107 | 1.0457002376040039e-06 | True | `None None` |
| 2026-11-28T15:24:33Z | `dec_c1b075000062` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999926729368 | 1.0922520250358194e-06 | True | `None None` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `action.succeeded` | `binary` | 0.9140046152170702 | 0.2803572338623807 | True | `None None` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `membership.confirmed` | `count` | 0.03239857615786972 | 0.18136129424755973 | True | `None None` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `observed.exec.modality.visual_heavy.n` | `count` | 0.030606036795665714 | 0.17612036586120325 | True | `None None` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571359728412 | 1.0383453195892273e-06 | True | `None None` |
| 2026-11-28T15:25:05Z | `dec_c1b075000063` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001334973727 | 1.0922925182782525e-06 | True | `None None` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `action.succeeded` | `binary` | 0.8822180281791531 | 0.3223497773147054 | True | `` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `membership.confirmed` | `count` | 0.06396587742502201 | 0.26003041774053054 | True | `` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `observed.exec.modality.visual_heavy.n` | `count` | 0.06226201492061293 | 0.2558846714165502 | True | `` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571696592632 | 1.4136425870762335e-06 | True | `` |
| 2026-11-28T15:25:46Z | `dec_c1b075000064` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000188390534 | 1.4528027433943768e-06 | True | `` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `action.succeeded` | `binary` | 0.8539986391537107 | 0.3531075806001918 | True | `None None` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `membership.confirmed` | `count` | 0.06425916142390488 | 0.26072019491941756 | True | `None None` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `observed.exec.modality.visual_heavy.n` | `count` | 0.06248785857407507 | 0.256427089343554 | True | `None None` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571699452917 | 1.415020629408668e-06 | True | `None None` |
| 2026-11-28T15:26:20Z | `dec_c1b075000065` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000001908141134 | 1.4544317342517775e-06 | True | `None None` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `action.succeeded` | `binary` | 0.8620479969771184 | 0.3448496018336928 | True | `None None` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `membership.confirmed` | `count` | 0.06124286220092289 | 0.2540121386020929 | True | `None None` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `observed.exec.modality.visual_heavy.n` | `count` | 0.059648736022611996 | 0.25008190714711226 | True | `None None` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571627530864 | 1.3148986460081415e-06 | True | `None None` |
| 2026-11-28T15:26:50Z | `dec_c1b075000066` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000138286944 | 1.3575178545864309e-06 | True | `None None` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `action.succeeded` | `binary` | 0.8807301354002691 | 0.3241057913677149 | True | `None None` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `membership.confirmed` | `count` | 0.051301798197219826 | 0.23107543867349134 | True | `None None` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `observed.exec.modality.visual_heavy.n` | `count` | 0.05030867767942736 | 0.22844942073735883 | True | `None None` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571482881958 | 1.1798092470906013e-06 | True | `None None` |
| 2026-11-28T15:27:33Z | `dec_c1b075000067` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000033846326 | 1.2362361359215162e-06 | True | `None None` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `action.succeeded` | `binary` | 0.9038951261257411 | 0.29473501164922994 | True | `None None` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `membership.confirmed` | `count` | 0.03618786755150202 | 0.19229751844910725 | True | `None None` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03397493895426327 | 0.18611221723133983 | True | `None None` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571759832097 | 1.0997069225839815e-06 | True | `None None` |
| 2026-11-28T15:28:17Z | `dec_c1b075000068` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000294563682 | 1.1705701352107094e-06 | True | `None None` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `action.succeeded` | `binary` | 0.9076491137417269 | 0.2895206384104329 | True | `None None` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `membership.confirmed` | `count` | 0.034394627611615144 | 0.18735504022250185 | True | `None None` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `observed.exec.modality.visual_heavy.n` | `count` | 0.03231180335537686 | 0.18137783807530533 | True | `None None` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571703188725 | 1.0494694980898687e-06 | True | `None None` |
| 2026-11-28T15:28:54Z | `dec_c1b075000069` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000002424408883 | 1.117222827766366e-06 | True | `None None` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `action.succeeded` | `binary` | 0.9199433773327018 | 0.27138083910697164 | True | `None None` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `membership.confirmed` | `count` | 0.029755149100404388 | 0.17392071032359052 | True | `None None` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `observed.exec.modality.visual_heavy.n` | `count` | 0.02803723084915598 | 0.16866933096899395 | True | `None None` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571566466484 | 9.69906802029772e-07 | True | `None None` |
| 2026-11-28T15:29:40Z | `dec_c1b075000070` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000000900631556 | 1.0399966434849408e-06 | True | `None None` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `action.succeeded` | `binary` | 0.9220925401480838 | 0.2680259083583866 | True | `None None` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `membership.confirmed` | `count` | 0.0293827763212518 | 0.17280349695405584 | True | `None None` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `observed.exec.modality.visual_heavy.n` | `count` | 0.027690110541133173 | 0.16760003903114828 | True | `None None` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571557758858 | 9.580539278116863e-07 | True | `None None` |
| 2026-11-28T15:30:14Z | `dec_c1b075000071` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000084410979 | 1.028176153999872e-06 | True | `None None` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `action.succeeded` | `binary` | 0.9160823578551079 | 0.2772642629004573 | True | `None None` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `membership.confirmed` | `count` | 0.023849507669666606 | 0.15538750255127498 | True | `None None` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `observed.exec.modality.visual_heavy.n` | `count` | 0.02274019177593176 | 0.1516347043437634 | True | `None None` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571157163882 | 1.0176030945703208e-06 | True | `None None` |
| 2026-11-28T15:30:49Z | `dec_c1b075000072` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999666659027 | 1.0770488353962522e-06 | True | `None None` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `action.succeeded` | `binary` | 0.9181973431079339 | 0.27406382872875623 | True | `` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `membership.confirmed` | `count` | 0.023756842649564607 | 0.15507784315208042 | True | `` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `observed.exec.modality.visual_heavy.n` | `count` | 0.02264660000488572 | 0.15131481030354615 | True | `` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571181084775 | 1.0007647880665378e-06 | True | `` |
| 2026-11-28T15:31:18Z | `dec_c1b075000073` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999987771124 | 1.0564167217529218e-06 | True | `` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `action.succeeded` | `binary` | 0.9088292060447037 | 0.287851837383153 | True | `None None` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `membership.confirmed` | `count` | 0.022590331889305884 | 0.1511897424541813 | True | `None None` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `observed.exec.modality.visual_heavy.n` | `count` | 0.021489858552959866 | 0.14737282326397064 | True | `None None` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571097137745 | 9.66376729408647e-07 | True | `None None` |
| 2026-11-28T15:32:51Z | `dec_c1b075000074` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999396539914 | 1.0148731309625866e-06 | True | `None None` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `action.succeeded` | `binary` | 0.9111228646536843 | 0.28456631943880567 | True | `None None` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `membership.confirmed` | `count` | 0.02279184076098999 | 0.1518694132474259 | True | `None None` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `observed.exec.modality.visual_heavy.n` | `count` | 0.021649609145844583 | 0.14792134208939456 | True | `None None` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857108275347 | 9.669833015792618e-07 | True | `None None` |
| 2026-11-28T15:33:39Z | `dec_c1b075000075` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999929314898 | 1.0058222491811864e-06 | True | `None None` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `action.succeeded` | `binary` | 0.9136293476338289 | 0.28091059569908206 | True | `` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `membership.confirmed` | `count` | 0.022156539439769096 | 0.149702322828995 | True | `` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `observed.exec.modality.visual_heavy.n` | `count` | 0.02107674155421642 | 0.1459231042190638 | True | `` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571147386526 | 9.434836249753683e-07 | True | `` |
| 2026-11-28T15:34:13Z | `dec_c1b075000076` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999980821061 | 9.881861551716313e-07 | True | `` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `action.succeeded` | `binary` | 0.9053040382017734 | 0.29279452969161734 | True | `None None` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `membership.confirmed` | `count` | 0.02119813630464063 | 0.14640854866830164 | True | `None None` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `observed.exec.modality.visual_heavy.n` | `count` | 0.020123665027649423 | 0.1425676343157789 | True | `None None` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571043179614 | 9.062757959410103e-07 | True | `None None` |
| 2026-11-28T15:34:56Z | `dec_c1b075000077` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999278735245 | 9.379402219644808e-07 | True | `None None` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `action.succeeded` | `binary` | 0.9074398759130242 | 0.2898150229301087 | True | `None None` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `membership.confirmed` | `count` | 0.021005113700946595 | 0.14572810325896893 | True | `None None` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `observed.exec.modality.visual_heavy.n` | `count` | 0.019945104449848712 | 0.14192341227058475 | True | `None None` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857106623964 | 8.963301988089673e-07 | True | `None None` |
| 2026-11-28T15:35:33Z | `dec_c1b075000078` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999941638081 | 9.277005740595437e-07 | True | `None None` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `action.succeeded` | `binary` | 0.9097397844920562 | 0.2865541990695711 | True | `` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `membership.confirmed` | `count` | 0.020976097426968394 | 0.14562472702581192 | True | `` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `observed.exec.modality.visual_heavy.n` | `count` | 0.019915201091847703 | 0.14181382427056174 | True | `` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571087538344 | 8.904260248320122e-07 | True | `` |
| 2026-11-28T15:35:57Z | `dec_c1b075000079` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999556711927 | 9.216344735404567e-07 | True | `` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `action.succeeded` | `binary` | 0.8988304898618454 | 0.30155304733423005 | True | `None None` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `membership.confirmed` | `count` | 0.020845424678681043 | 0.1451603172480625 | True | `None None` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `observed.exec.modality.visual_heavy.n` | `count` | 0.019791136220840975 | 0.14136206159211215 | True | `None None` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571091098084 | 8.892677684723344e-07 | True | `None None` |
| 2026-11-28T15:36:36Z | `dec_c1b075000080` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.99999995826654 | 9.207908487932928e-07 | True | `None None` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `action.succeeded` | `binary` | 0.9042129035410772 | 0.2942990462283081 | True | `None None` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `membership.confirmed` | `count` | 0.020043837316235262 | 0.14232815752979494 | True | `None None` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01901193950164587 | 0.1385393496377857 | True | `None None` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571006992328 | 8.501658121400316e-07 | True | `None None` |
| 2026-11-28T15:48:50Z | `dec_c1b075000081` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999899653811 | 8.650737615408721e-07 | True | `None None` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `action.succeeded` | `binary` | 0.9064726733117471 | 0.29116999476388655 | True | `None None` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `membership.confirmed` | `count` | 0.019885742357556198 | 0.1417576783556025 | True | `None None` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01886110086722219 | 0.13798129996490963 | True | `None None` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857103457256 | 8.430024660613799e-07 | True | `None None` |
| 2026-11-28T15:49:20Z | `dec_c1b075000082` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999915242581 | 8.597867675264764e-07 | True | `None None` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `action.succeeded` | `binary` | 0.9086420950715534 | 0.28811740338884695 | True | `None None` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `membership.confirmed` | `count` | 0.019739840985046345 | 0.1412289733388246 | True | `None None` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `observed.exec.modality.visual_heavy.n` | `count` | 0.018729136607964155 | 0.13749105874307668 | True | `None None` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857105987194 | 8.368255375855156e-07 | True | `None None` |
| 2026-11-28T15:49:54Z | `dec_c1b075000083` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999999295790616 | 8.553976652198623e-07 | True | `None None` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `action.succeeded` | `binary` | 0.9129920473860977 | 0.28184671152213076 | True | `None None` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `membership.confirmed` | `count` | 0.019200802897844454 | 0.13928950502272372 | True | `None None` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01819733413095472 | 0.13552659214420568 | True | `None None` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570974791976 | 8.125579499143311e-07 | True | `None None` |
| 2026-11-28T15:50:32Z | `dec_c1b075000084` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998731842243 | 8.144074470931934e-07 | True | `None None` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `action.succeeded` | `binary` | 0.914985758134373 | 0.27890288730244384 | True | `` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `membership.confirmed` | `count` | 0.01907107827222737 | 0.1388113429443754 | True | `` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `observed.exec.modality.visual_heavy.n` | `count` | 0.018074459368172452 | 0.13506196411396693 | True | `` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571010832556 | 8.035910446678808e-07 | True | `` |
| 2026-11-28T15:51:03Z | `dec_c1b075000085` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998926748637 | 8.090202932684167e-07 | True | `` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `action.succeeded` | `binary` | 0.9050044196804826 | 0.2932088334946195 | True | `` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `membership.confirmed` | `count` | 0.019060976111001902 | 0.1387727744196061 | True | `` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01806222430658919 | 0.13501467601894945 | True | `` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857100851096 | 8.03248574485573e-07 | True | `` |
| 2026-11-28T15:51:37Z | `dec_c1b075000086` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.99999989002585 | 8.087985347558172e-07 | True | `` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `action.succeeded` | `binary` | 0.8950626464448158 | 0.3064726829327835 | True | `None None` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `membership.confirmed` | `count` | 0.018901821488455477 | 0.13821556660170226 | True | `None None` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017916311659363027 | 0.13449111302344366 | True | `None None` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570901886697 | 8.000079258760628e-07 | True | `None None` |
| 2026-11-28T15:52:11Z | `dec_c1b075000087` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999816578174 | 7.882932083484787e-07 | True | `None None` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `action.succeeded` | `binary` | 0.8980062191198173 | 0.30264013207429713 | True | `` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `membership.confirmed` | `count` | 0.01877373377948961 | 0.13773779771807243 | True | `` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017796850873222667 | 0.13403465254794789 | True | `` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570956844093 | 7.848919810088043e-07 | True | `` |
| 2026-11-28T15:52:47Z | `dec_c1b075000088` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998486747725 | 7.793791975274508e-07 | True | `` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `action.succeeded` | `binary` | 0.8866614524395485 | 0.3170061847935128 | True | `None None` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `membership.confirmed` | `count` | 0.01875968168515266 | 0.13768649120728252 | True | `None None` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017782125587131485 | 0.1339791736320297 | True | `None None` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570953624694 | 7.849918661410754e-07 | True | `None None` |
| 2026-11-28T15:53:18Z | `dec_c1b075000089` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998463938646 | 7.798028803139988e-07 | True | `None None` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `action.succeeded` | `binary` | 0.8899309671727488 | 0.3129757831521869 | True | `None None` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `membership.confirmed` | `count` | 0.01860412723186049 | 0.13710666643331737 | True | `None None` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017632768905069734 | 0.1334078279314705 | True | `None None` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570998095694 | 7.721125930393241e-07 | True | `None None` |
| 2026-11-28T15:53:49Z | `dec_c1b075000090` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999871017672 | 7.722278626820264e-07 | True | `None None` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `action.succeeded` | `binary` | 0.8931248288731607 | 0.3089544771053628 | True | `` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `membership.confirmed` | `count` | 0.0182996359675378 | 0.13599091445558537 | True | `` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01735767157045722 | 0.1323775439026977 | True | `` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857093720869 | 7.607834500778661e-07 | True | `` |
| 2026-11-28T15:54:33Z | `dec_c1b075000091` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999819855884 | 7.547405754601577e-07 | True | `` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `action.succeeded` | `binary` | 0.8820756153826794 | 0.3225185640082549 | True | `None None` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `membership.confirmed` | `count` | 0.01825885832835627 | 0.13583400892953357 | True | `None None` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017310811876881965 | 0.13219344869506108 | True | `None None` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570935466256 | 7.597992038621021e-07 | True | `None None` |
| 2026-11-28T15:55:14Z | `dec_c1b075000092` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998170083275 | 7.531026016302806e-07 | True | `None None` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `action.succeeded` | `binary` | 0.8854303423178024 | 0.31850188574132754 | True | `` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `membership.confirmed` | `count` | 0.01814674216675918 | 0.13541007551446452 | True | `` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `observed.exec.modality.visual_heavy.n` | `count` | 0.017205126259301203 | 0.13178344951228046 | True | `` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570982484644 | 7.452844137446973e-07 | True | `` |
| 2026-11-28T15:55:51Z | `dec_c1b075000093` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999844390341 | 7.448288175326209e-07 | True | `` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `action.succeeded` | `binary` | 0.8753315986565972 | 0.33034253593790047 | True | `None None` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `membership.confirmed` | `count` | 0.01768540700555272 | 0.13367947278165487 | True | `None None` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01676699153851665 | 0.1301023733434994 | True | `None None` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570891263096 | 7.482684851382902e-07 | True | `None None` |
| 2026-11-28T15:56:33Z | `dec_c1b075000094` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999997724960528 | 7.437074143613788e-07 | True | `None None` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `action.succeeded` | `binary` | 0.8789518002021507 | 0.32618328148994585 | True | `None None` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `membership.confirmed` | `count` | 0.01750645295408017 | 0.13299499454787783 | True | `None None` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01659888945013892 | 0.12944219433568072 | True | `None None` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570947656233 | 7.311384152931234e-07 | True | `None None` |
| 2026-11-28T15:57:08Z | `dec_c1b075000095` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998060251665 | 7.330154972667351e-07 | True | `None None` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `action.succeeded` | `binary` | 0.882741230537646 | 0.3217283799209025 | True | `` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `membership.confirmed` | `count` | 0.017411264740560047 | 0.13261814158733973 | True | `` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `observed.exec.modality.visual_heavy.n` | `count` | 0.016505606990864133 | 0.1290654114333445 | True | `` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570984550587 | 7.169936728359692e-07 | True | `` |
| 2026-11-28T15:57:52Z | `dec_c1b075000096` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999821452663 | 7.238936830479117e-07 | True | `` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `action.succeeded` | `binary` | 0.8717558774651026 | 0.3343614325399262 | True | `None None` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `membership.confirmed` | `count` | 0.017405588388992272 | 0.13259534819777558 | True | `None None` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `observed.exec.modality.visual_heavy.n` | `count` | 0.016504376226766825 | 0.1290597360086404 | True | `None None` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570986701086 | 7.164200726079202e-07 | True | `None None` |
| 2026-11-28T15:58:35Z | `dec_c1b075000097` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999825031627 | 7.238265890427527e-07 | True | `None None` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `action.succeeded` | `binary` | 0.8787662369320062 | 0.32639874043930894 | True | `None None` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `membership.confirmed` | `count` | 0.016539439854591338 | 0.12922359241204792 | True | `None None` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `observed.exec.modality.visual_heavy.n` | `count` | 0.015683185762716313 | 0.12578904865091464 | True | `None None` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570961170848 | 7.016750330720185e-07 | True | `None None` |
| 2026-11-28T15:59:17Z | `dec_c1b075000098` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999997792709485 | 7.184034508571942e-07 | True | `None None` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `action.succeeded` | `binary` | 0.8819900357669248 | 0.32261991968070336 | True | `` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `membership.confirmed` | `count` | 0.016424861420573235 | 0.12877006857459738 | True | `` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `observed.exec.modality.visual_heavy.n` | `count` | 0.015571424946133608 | 0.1253353767888785 | True | `` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570998205934 | 6.911330379604048e-07 | True | `` |
| 2026-11-28T15:59:54Z | `dec_c1b075000099` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999997996694447 | 7.114639822112245e-07 | True | `` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `action.succeeded` | `binary` | 0.8721822334002456 | 0.3338867849154937 | True | `None None` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `membership.confirmed` | `count` | 0.016386660223840936 | 0.128615723055523 | True | `None None` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `observed.exec.modality.visual_heavy.n` | `count` | 0.015532629181572152 | 0.12517468002887544 | True | `None None` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428570998873004 | 6.89457493711985e-07 | True | `None None` |
| 2026-11-28T16:00:32Z | `dec_c1b075000100` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999801757142 | 7.104962772631068e-07 | True | `None None` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `action.succeeded` | `binary` | 0.8806853836564091 | 0.3241583543121673 | True | `None None` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `membership.confirmed` | `count` | 0.015324170354184182 | 0.12433807932624681 | True | `None None` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `observed.exec.modality.visual_heavy.n` | `count` | 0.014518610159823439 | 0.12099292680961869 | True | `None None` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571008671492 | 6.75021694377695e-07 | True | `None None` |
| 2026-11-28T16:17:04Z | `dec_c1b075000101` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999997723055687 | 7.112169264488209e-07 | True | `None None` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `action.succeeded` | `binary` | 0.8837046316804874 | 0.3205787822469564 | True | `None None` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `membership.confirmed` | `count` | 0.015222252760128169 | 0.12391808914034805 | True | `None None` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `observed.exec.modality.visual_heavy.n` | `count` | 0.014424817554367567 | 0.12059670942146428 | True | `None None` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571040563803 | 6.663366515122338e-07 | True | `None None` |
| 2026-11-28T16:17:50Z | `dec_c1b075000102` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999795138822 | 7.037847689970228e-07 | True | `None None` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `action.succeeded` | `binary` | 0.8864499378740348 | 0.31726399971814423 | True | `None None` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `membership.confirmed` | `count` | 0.01514573214532875 | 0.12360152795690349 | True | `None None` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01435373844214819 | 0.12029495896752422 | True | `None None` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571067432959 | 6.57000487935953e-07 | True | `None None` |
| 2026-11-28T16:18:23Z | `dec_c1b075000103` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998131332424 | 6.952800361294975e-07 | True | `None None` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `action.succeeded` | `binary` | 0.8947462383655886 | 0.30688011876988364 | True | `None None` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `membership.confirmed` | `count` | 0.013964282793150354 | 0.11864090287173719 | True | `None None` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `observed.exec.modality.visual_heavy.n` | `count` | 0.013228062543872866 | 0.11545064702418237 | True | `None None` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571115663424 | 6.405476040942787e-07 | True | `None None` |
| 2026-11-28T16:19:14Z | `dec_c1b075000104` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999818046494 | 6.931178615188006e-07 | True | `None None` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `action.succeeded` | `binary` | 0.8970338904479581 | 0.3039146094477177 | True | `None None` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `membership.confirmed` | `count` | 0.014168829915903135 | 0.11951790150110811 | True | `None None` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `observed.exec.modality.visual_heavy.n` | `count` | 0.013397538371846335 | 0.11620207053330771 | True | `None None` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857109970079 | 6.384465470472469e-07 | True | `None None` |
| 2026-11-28T16:19:58Z | `dec_c1b075000105` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998013755897 | 6.808485983470436e-07 | True | `None None` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `action.succeeded` | `binary` | 0.8991914657086485 | 0.3010750300263722 | True | `` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `membership.confirmed` | `count` | 0.013817287909581089 | 0.11800746114218066 | True | `` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01308638242767232 | 0.11482422758788588 | True | `` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857115530309 | 6.288003913165018e-07 | True | `` |
| 2026-11-28T16:20:40Z | `dec_c1b075000106` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998490068247 | 6.809100826102245e-07 | True | `` |
| 2026-11-28T16:21:27Z | `dec_c1b075000107` | `action.succeeded` | `binary` | 0.8968886631170675 | 0.3041042404984628 | True | `None None` |
| 2026-11-28T16:21:27Z | `dec_c1b075000107` | `membership.confirmed` | `count` | 0.012844428390978022 | 0.11374722723077059 | True | `None None` |
| 2026-11-28T16:21:27Z | `dec_c1b075000107` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01216581354465903 | 0.11068931497323707 | True | `None None` |
| 2026-11-28T16:21:27Z | `dec_c1b075000107` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571190546842 | 6.181093593869431e-07 | True | `None None` |
| 2026-11-28T16:21:27Z | `dec_c1b075000107` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999852336908 | 6.779316671930693e-07 | True | `None None` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `action.succeeded` | `binary` | 0.8682979925872335 | 0.3381665102522929 | True | `None None` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `membership.confirmed` | `count` | 0.015908608415797255 | 0.12701610183515724 | True | `None None` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `observed.exec.modality.visual_heavy.n` | `count` | 0.015233718296648375 | 0.1242527955096612 | True | `None None` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571177013097 | 8.115479461640153e-07 | True | `None None` |
| 2026-11-28T16:22:10Z | `dec_c1b075000108` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999850561532 | 8.618296090925923e-07 | True | `None None` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `action.succeeded` | `binary` | 0.8706157136469315 | 0.33562477977344835 | True | `None None` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `membership.confirmed` | `count` | 0.015769115078852806 | 0.1264453769748593 | True | `None None` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01510204350936271 | 0.12370404242933535 | True | `None None` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571193103677 | 8.070059318883967e-07 | True | `None None` |
| 2026-11-28T16:22:46Z | `dec_c1b075000109` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998654636872 | 8.565572363111808e-07 | True | `None None` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `action.succeeded` | `binary` | 0.8732047733814894 | 0.3327434403940537 | True | `None None` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `membership.confirmed` | `count` | 0.015685523590843145 | 0.12610572439448747 | True | `None None` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `observed.exec.modality.visual_heavy.n` | `count` | 0.015021523387567674 | 0.12337009528225827 | True | `None None` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857120697494 | 8.053889018009813e-07 | True | `None None` |
| 2026-11-28T16:23:22Z | `dec_c1b075000110` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998775520593 | 8.540715674047884e-07 | True | `None None` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `action.succeeded` | `binary` | 0.8800424908004068 | 0.32491184217603203 | True | `None None` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `membership.confirmed` | `count` | 0.014642563519278716 | 0.12178755249658495 | True | `None None` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `observed.exec.modality.visual_heavy.n` | `count` | 0.014060753541788665 | 0.11932120862572758 | True | `None None` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571240962855 | 7.619808590868063e-07 | True | `None None` |
| 2026-11-28T16:24:06Z | `dec_c1b075000111` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999885138522 | 8.130501035524974e-07 | True | `None None` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `action.succeeded` | `binary` | 0.8820315937660906 | 0.322570707542611 | True | `None None` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `membership.confirmed` | `count` | 0.014546094032233491 | 0.12137835285562068 | True | `None None` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `observed.exec.modality.visual_heavy.n` | `count` | 0.013967578670093302 | 0.11891905078032945 | True | `None None` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857125075588 | 7.613738284766608e-07 | True | `None None` |
| 2026-11-28T16:24:56Z | `dec_c1b075000112` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998949136417 | 8.115098389526119e-07 | True | `None None` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `action.succeeded` | `binary` | 0.8841028522627936 | 0.3201015446441747 | True | `None None` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `membership.confirmed` | `count` | 0.014493063599975743 | 0.12115329274616166 | True | `None None` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `observed.exec.modality.visual_heavy.n` | `count` | 0.013917192114932036 | 0.11870007258444852 | True | `None None` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571260425857 | 7.557037021697548e-07 | True | `None None` |
| 2026-11-28T16:25:36Z | `dec_c1b075000113` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999903241551 | 8.054591465046154e-07 | True | `None None` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `action.succeeded` | `binary` | 0.8884068487467037 | 0.31486524077239014 | True | `None None` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `membership.confirmed` | `count` | 0.01367304423195672 | 0.11764498850623757 | True | `None None` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `observed.exec.modality.visual_heavy.n` | `count` | 0.01318757630618072 | 0.11553171204705484 | True | `None None` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571262515474 | 7.078234764030464e-07 | True | `None None` |
| 2026-11-28T16:26:22Z | `dec_c1b075000114` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998930060077 | 7.580790093541181e-07 | True | `None None` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `action.succeeded` | `binary` | 0.8904822500575292 | 0.31228770771519204 | True | `None None` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `membership.confirmed` | `count` | 0.013919011779735358 | 0.11872932959790419 | True | `None None` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `observed.exec.modality.visual_heavy.n` | `count` | 0.013414063634931663 | 0.11654932471197493 | True | `None None` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571241811516 | 7.221513872978839e-07 | True | `None None` |
| 2026-11-28T16:26:59Z | `dec_c1b075000115` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998744838714 | 7.609556011257025e-07 | True | `None None` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `action.succeeded` | `binary` | 0.9236512014447726 | 0.26555537937387125 | True | `None None` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `membership.confirmed` | `count` | 0.01113398026472136 | 0.1058224245581563 | True | `None None` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `observed.exec.modality.visual_heavy.n` | `count` | 0.010363682769760556 | 0.10208812286704808 | True | `None None` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571201283814 | 6.427961796728239e-07 | True | `None None` |
| 2026-11-28T16:27:44Z | `dec_c1b075000116` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999809837169 | 7.116258255993554e-07 | True | `None None` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `action.succeeded` | `binary` | 0.9257913653086439 | 0.2621101166086516 | True | `` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `membership.confirmed` | `count` | 0.010609815038436788 | 0.10329658783899431 | True | `` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `observed.exec.modality.visual_heavy.n` | `count` | 0.009905566496444104 | 0.09980563455668148 | True | `` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571217054982 | 6.449845656133656e-07 | True | `` |
| 2026-11-28T16:28:35Z | `dec_c1b075000117` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999830622576 | 7.215268570687314e-07 | True | `` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `action.succeeded` | `binary` | 0.9201206012840555 | 0.2711064004717035 | True | `None None` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `membership.confirmed` | `count` | 0.010614335289109537 | 0.1033185486821989 | True | `None None` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `observed.exec.modality.visual_heavy.n` | `count` | 0.009911674020463602 | 0.09983645131806042 | True | `None None` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571222357713 | 6.447298658816874e-07 | True | `None None` |
| 2026-11-28T16:29:10Z | `dec_c1b075000118` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998345580887 | 7.207792683057287e-07 | True | `None None` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `action.succeeded` | `binary` | 0.9215408513310446 | 0.26889274936133584 | True | `` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `membership.confirmed` | `count` | 0.010586136860349087 | 0.10318015371218116 | True | `` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `observed.exec.modality.visual_heavy.n` | `count` | 0.009884689721461705 | 0.0996993402282958 | True | `` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571231231581 | 6.42114745304806e-07 | True | `` |
| 2026-11-28T16:29:53Z | `dec_c1b075000119` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.9999998446742717 | 7.17086629885982e-07 | True | `` |

## Experiments (42 registry records)

Redacted in this section by the run logs' rules: none.

* 2026-11-28T14:47:10Z `experiment_outcome` `exp_d5624dae9be64cdb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:47:10Z `experiment_outcome` `exp_d5624dae9be64cdb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:47:10Z `experiment_outcome` `exp_d5624dae9be64cdb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:47:10Z `experiment_outcome` `exp_d5624dae9be64cdb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:47:10Z `experiment_outcome` `exp_d5624dae9be64cdb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:47:10Z `experiment` `exp_d5624dae9be64cdb`: status `matured`; question: `Which of discord.modify_guild({"explicit_content_filter": 2, "guild_id": "1575680989060010034"}) and discord.modify_guild({"guild_id": "1575680989060010034", "rules_channel_id": null}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `discord.modify_guild({"explicit_content_filter": 2, "guild_id": "1575680989060010034"}) gives at least as much as discord.modify_guild({"guild_id": "1575680989060010034", "rules_channel_id": null}): predicted 1.06 (epistemic sd 1.27) against 1.05 (epistemic sd 1.22); probability that the order is wrong 0.499`; arms: `["discord.modify_guild({\"explicit_content_filter\": 2, \"guild_id\": \"1575680989060010034\"})", "discord.modify_guild({\"guild_id\": \"1575680989060010034\", \"rules_channel_id\": null})"]`; supported: ``
* 2026-11-28T14:49:38Z `experiment_outcome` `exp_53b5470de41d4bb8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:49:38Z `experiment_outcome` `exp_53b5470de41d4bb8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:49:38Z `experiment_outcome` `exp_53b5470de41d4bb8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:49:38Z `experiment_outcome` `exp_53b5470de41d4bb8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:49:38Z `experiment_outcome` `exp_53b5470de41d4bb8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:49:38Z `experiment` `exp_53b5470de41d4bb8`: status `matured`; question: `Which of browser.read({}) and browser.screenshot({}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `browser.read({}) gives at least as much as browser.screenshot({}): predicted 0.145 (epistemic sd 0.105) against 0.144 (epistemic sd 0.105); probability that the order is wrong 0.498`; arms: `["browser.read({})", "browser.screenshot({})"]`; supported: ``
* 2026-11-28T14:51:34Z `experiment_outcome` `exp_c2033d04b22949fb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:51:34Z `experiment_outcome` `exp_c2033d04b22949fb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:51:34Z `experiment_outcome` `exp_c2033d04b22949fb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:51:34Z `experiment_outcome` `exp_c2033d04b22949fb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:51:34Z `experiment_outcome` `exp_c2033d04b22949fb`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T14:51:34Z `experiment` `exp_c2033d04b22949fb`: status `matured`; question: `Which of browser.read({"nth": 0}) and browser.scroll({"nth": 0}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `browser.read({"nth": 0}) gives at least as much as browser.scroll({"nth": 0}): predicted 0.111 (epistemic sd 0.0802) against 0.111 (epistemic sd 0.0803); probability that the order is wrong 0.500`; arms: `["browser.read({\"nth\": 0})", "browser.scroll({\"nth\": 0})"]`; supported: ``
* 2026-11-28T15:27:57Z `experiment_outcome` `exp_5b40283e828b4ab8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:27:57Z `experiment_outcome` `exp_5b40283e828b4ab8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:27:57Z `experiment_outcome` `exp_5b40283e828b4ab8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:27:57Z `experiment_outcome` `exp_5b40283e828b4ab8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:27:57Z `experiment_outcome` `exp_5b40283e828b4ab8`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:27:57Z `experiment` `exp_5b40283e828b4ab8`: status `matured`; question: `Which of discord.modify_guild({"default_message_notifications": 1, "description": "A community focused on creating value through collaboration and knowledge sharing.", "explicit_content_filter": 1, "guild_id": "1575680989060010034", "name": "UsefulCommunity", "preferred_locale": "en-US", "verification_level": 2}) and discord.modify_guild({"guild_id": "1575680989060010034", "public_updates_channel_id": null}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `discord.modify_guild({"default_message_notifications": 1, "description": "A community focused on creating value through collaboration and knowledge sharing.", "explicit_content_filter": 1, "guild_id": "1575680989060010034", "name": "UsefulCommunity", "preferred_locale": "en-US", "verification_level": 2}) gives at least as much as discord.modify_guild({"guild_id": "1575680989060010034", "public_updates_channel_id": null}): predicted 0.0513 (epistemic sd 0.0423) against 0.0513 (epistemic sd 0.0423); probability that the order is wrong 0.500`; arms: `["discord.modify_guild({\"default_message_notifications\": 1, \"description\": \"A community focused on creating value through collaboration and knowledge sharing.\", \"explicit_content_filter\": 1, \"guild_id\": \"1575680989060010034\", \"name\": \"UsefulCommunity\", \"preferred_locale\": \"en-US\", \"verification_level\": 2})", "discord.modify_guild({\"guild_id\": \"1575680989060010034\", \"public_updates_channel_id\": null})"]`; supported: ``
* 2026-11-28T15:48:30Z `experiment_outcome` `exp_b912d3c34f744e1c`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:48:30Z `experiment_outcome` `exp_b912d3c34f744e1c`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:48:30Z `experiment_outcome` `exp_b912d3c34f744e1c`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:48:30Z `experiment_outcome` `exp_b912d3c34f744e1c`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:48:30Z `experiment_outcome` `exp_b912d3c34f744e1c`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:48:33Z `experiment` `exp_b912d3c34f744e1c`: status `matured`; question: `Which of browser.inspect({"tab": "t2"}) and browser.close_tab({"tab": "t2"}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `browser.inspect({"tab": "t2"}) gives at least as much as browser.close_tab({"tab": "t2"}): predicted 0.0208 (epistemic sd 0.0124) against 0.0208 (epistemic sd 0.0124); probability that the order is wrong 0.500`; arms: `["browser.inspect({\"tab\": \"t2\"})", "browser.close_tab({\"tab\": \"t2\"})"]`; supported: ``
* 2026-11-28T15:55:36Z `experiment_outcome` `exp_9be8a68e0e634f26`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:55:36Z `experiment_outcome` `exp_9be8a68e0e634f26`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:55:36Z `experiment_outcome` `exp_9be8a68e0e634f26`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:55:36Z `experiment_outcome` `exp_9be8a68e0e634f26`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:55:36Z `experiment_outcome` `exp_9be8a68e0e634f26`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T15:55:36Z `experiment` `exp_9be8a68e0e634f26`: status `matured`; question: `Which of discord.listen_events({"guild_id": "1575680989060010034", "seconds": 45}) and discord.get_guild_settings({"guild_id": "1575680989060010034"}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `discord.listen_events({"guild_id": "1575680989060010034", "seconds": 45}) gives at least as much as discord.get_guild_settings({"guild_id": "1575680989060010034"}): predicted 0.0183 (epistemic sd 0.0114) against 0.0183 (epistemic sd 0.0114); probability that the order is wrong 0.500`; arms: `["discord.listen_events({\"guild_id\": \"1575680989060010034\", \"seconds\": 45})", "discord.get_guild_settings({\"guild_id\": \"1575680989060010034\"})"]`; supported: ``
* 2026-11-28T16:29:38Z `experiment_outcome` `exp_1d9fe278ec1e49a0`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T16:29:38Z `experiment_outcome` `exp_1d9fe278ec1e49a0`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T16:29:38Z `experiment_outcome` `exp_1d9fe278ec1e49a0`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T16:29:38Z `experiment_outcome` `exp_1d9fe278ec1e49a0`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T16:29:38Z `experiment_outcome` `exp_1d9fe278ec1e49a0`: status ``; question: ``; hypothesis: ``; arms: `[]`; supported: ``
* 2026-11-28T16:29:38Z `experiment` `exp_1d9fe278ec1e49a0`: status `matured`; question: `Which of discord.list_channels({"guild_id": "1575680989060010034"}) and discord.list_invites({"guild_id": "1575680989060010034"}) gives more of: the objective metric, measured by the independent verifier, after the action?`; hypothesis: `discord.list_channels({"guild_id": "1575680989060010034"}) gives at least as much as discord.list_invites({"guild_id": "1575680989060010034"}): predicted 0.0106 (epistemic sd 0.00559) against 0.0106 (epistemic sd 0.00559); probability that the order is wrong 0.500`; arms: `["discord.list_channels({\"guild_id\": \"1575680989060010034\"})", "discord.list_invites({\"guild_id\": \"1575680989060010034\"})"]`; supported: ``

## Learning updates

* updates: 192 {"checkpoint": 1, "fast": 119, "frequent": 68, "slow_adapters": 4}
* with a measured behavioural change: 191
* meta-updates: {"applied": 0, "bootstrap_mix_last": 1.0, "meta_grad_norms_applied": [], "n": 118, "reward_versions": ["rpsi1-s1.0-a2ecb7bf9a11298d", "rpsi1-s1.0-664f14e69f90ce3a", "rpsi1-s1.0-1eda84eb7702fb33", "rpsi1-s1.0-a70e8360c367ba1e", "rpsi1-s1.0-1e6fd80f74eb85c1", "rpsi1-s1.0-4211e36dbf131afe", "rpsi1-s1.0-8f19f658d4ed8733", "rpsi1-s1.0-0a2e41cb940d4aa1", "rpsi1-s1.0-2d14c5a73b36c537", "rpsi1-s1.0-d20fc50d2c9ec291", "rpsi1-s1.0-0ce2076d97915b76", "rpsi1-s1.0-1d8c18b7108448c5", "rpsi1-s1.0-460b33ea1b79c763", "rpsi1-s1.0-6d64aae99584737d", "rpsi1-s1.0-4c80a8a9095c3a6a", "rpsi1-s1.0-24124758a1295539", "rpsi1-s1.0-10c6107bbe90c643", "rpsi1-s1.0-db33928675a8f01f", "rpsi1-s1.0-c83108e59939f612", "rpsi1-s1.0-1b66b534c362a595", "rpsi1-s1.0-79d0a3d6ef0ec0ec", "rpsi1-s1.0-2243378b3d6deb86", "rpsi1-s1.0-ff8b043c1ce83c77", "rpsi1-s1.0-85eeeeb9521c5cb3", "rpsi1-s1.0-739bd5acb7a81779", "rpsi1-s1.0-07caf2476af71e8a", "rpsi1-s1.0-e4b08e06f91c8c04", "rpsi1-s1.0-76796f5ecd18f973", "rpsi1-s1.0-7ef37cf35b1c89cf", "rpsi1-s1.0-7508e8e4578ae2e4", "rpsi1-s1.0-45091c0663d41e03", "rpsi1-s1.0-4700ef7b90a07c78", "rpsi1-s1.0-9d811722e0b3439d", "rpsi1-s1.0-1a0a4faa60e1fae9", "rpsi1-s1.0-211177fa8871a613", "rpsi1-s1.0-4c09b27816795cd4", "rpsi1-s1.0-f8995bf5c52f34d9", "rpsi1-s1.0-a31c52234a4ea636", "rpsi1-s1.0-bc52aae96834ab0d", "rpsi1-s1.0-67f319b966ffc544", "rpsi1-s1.0-7af2c970bb95da70", "rpsi1-s1.0-6d7a2380b1301b44", "rpsi1-s1.0-6d88ce08b188baeb", "rpsi1-s1.0-7583c70da76828a5", "rpsi1-s1.0-286f081488bb1909", "rpsi1-s1.0-503ad2ad176e599f", "rpsi1-s1.0-5eea14127aa0e424", "rpsi1-s1.0-db5ada890fd2575b", "rpsi1-s1.0-eedca29fd3ced56e", "rpsi1-s1.0-bc300a821f942b00", "rpsi1-s1.0-24cf2287f43f3d91", "rpsi1-s1.0-9ee4b2fb3dbf0585", "rpsi1-s1.0-2c82089960e69246", "rpsi1-s1.0-18d6d2daa066d944", "rpsi1-s1.0-7478242bc22ffa7e", "rpsi1-s1.0-768cdbfde1f3623f", "rpsi1-s1.0-49c432e8fc164d40", "rpsi1-s1.0-8483344a0fa3d8c0", "rpsi1-s1.0-7eaa427547a1cbbe", "rpsi1-s1.0-dcb2262dab048135", "rpsi1-s1.0-62edbe8de0fc0819", "rpsi1-s1.0-58515474e33527c7", "rpsi1-s1.0-50f3b3044f5861af", "rpsi1-s1.0-ef9c8ec57ab47943", "rpsi1-s1.0-23c9456e31d9a3bc", "rpsi1-s1.0-a85aa3bd1e9a9d72", "rpsi1-s1.0-43785cec12f94edd", "rpsi1-s1.0-ea84ab2234eed5d3", "rpsi1-s1.0-ea3175d2fe52aff1", "rpsi1-s1.0-4a626875cbfebcaa", "rpsi1-s1.0-0109c6d2ed1357db", "rpsi1-s1.0-9311c71c0ec95772", "rpsi1-s1.0-1dd5090037355d37", "rpsi1-s1.0-e56f963cb2464ccd", "rpsi1-s1.0-b724ca8c9776dcb3", "rpsi1-s1.0-2416fb29217454d3", "rpsi1-s1.0-8fff861a94e92734", "rpsi1-s1.0-a850aed79506cde8", "rpsi1-s1.0-7a5e73151d7db7fe", "rpsi1-s1.0-de8d04686812b223", "rpsi1-s1.0-35d3c56042f7273e", "rpsi1-s1.0-9b96022eeb18450f", "rpsi1-s1.0-0e04f1d1278fd6ba", "rpsi1-s1.0-f828e197bcd65337", "rpsi1-s1.0-c260d328f51ae2ea", "rpsi1-s1.0-0b5244f39e432154", "rpsi1-s1.0-b14acd309a02638b", "rpsi1-s1.0-6cf642706082dfc4", "rpsi1-s1.0-229586a78eece512", "rpsi1-s1.0-eaec4a4412dba46d", "rpsi1-s1.0-92758cecfc16247d", "rpsi1-s1.0-39ede7a578237e9a", "rpsi1-s1.0-cc52d8825bb6d614", "rpsi1-s1.0-0c28fec88bcdb275", "rpsi1-s1.0-edc0826f3fc60e13", "rpsi1-s1.0-65aa70574e2ab54e", "rpsi1-s1.0-8b5359875c411210", "rpsi1-s1.0-eb3f3312e84f83a7", "rpsi1-s1.0-65e1c9ede2f518f5", "rpsi1-s1.0-7134d67d5b0258a1", "rpsi1-s1.0-2eb23015e1ddaeca", "rpsi1-s1.0-dbffae0cde7c985c", "rpsi1-s1.0-47530e2429bfbab4", "rpsi1-s1.0-8dbb2260acca4335", "rpsi1-s1.0-aa04a01ddc9452bd", "rpsi1-s1.0-47464a06d5fdeab4", "rpsi1-s1.0-021e2cb79f60aed9", "rpsi1-s1.0-5bebc317227c5f57", "rpsi1-s1.0-de0b6aabb5148a98", "rpsi1-s1.0-7d9f3beddc209391", "rpsi1-s1.0-af5b183297f9214a", "rpsi1-s1.0-3524ce5c36fda247", "rpsi1-s1.0-3d186e178334d90c", "rpsi1-s1.0-eee2d78a6fdbf9cd", "rpsi1-s1.0-0cfc17c0db2f0cfd"], "skipped_reasons": {"insufficient_independent_outcomes": 19, "no_policy_update_recorded": 2, "zero_advantage": 97}}

| timescale:group | updates | non-zero | sum of delta norms | largest |
|---|---|---|---|---|
| `frequent:experiential` | 68 | 68 | 52.1432 | 3.99921 |
| `frequent:heads` | 68 | 68 | 55.6917 | 2.84449 |
| `slow_adapters:semantic_adapters` | 4 | 4 | 0.126687 | 0.0473095 |
