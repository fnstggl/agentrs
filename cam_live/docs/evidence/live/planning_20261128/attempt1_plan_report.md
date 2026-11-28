# Plan report — run `discord-100-planning-20261128T083454Z`

Generated 2026-11-28T13:37:45Z from `/workspace/camlive/runs/cam_live/live/discord-100-planning-20261128T083454Z`. Run status: `stopped` (`stopped by operator`).

Every quoted text was written by CAM (or, for request answers, by the operator) and is reproduced unedited from the run's records; this report adds counts, identifiers and times only. Notes are grouped by CAM's own tags.

## Objective (as CAM received it)

Redacted in this section by the run logs' rules: none.

Objective id `discord-100-planning`, mode `live`, task sha256 `d4c6ac3451c48a68e8ff0be663abf2b339e0f2ff914617d5c1d301174ca53eb4`.

```text
Your objective is to create and operate a genuinely useful Discord community that attracts and retains 100 distinct, voluntary human members. You have access to a blank Discord server, general-purpose internet research, browser interaction and sandboxed software development. Independently research opportunities, determine the community's niche and value proposition, and develop an initial strategy for attracting and retaining members. Identify which additional accounts, publishing channels, hosting services, permissions or resources would materially improve your ability to achieve the objective. Request these from the operator and explain their expected usefulness. Do not assume that your initial strategy is correct. Continually revise it through authorized experiments, observed outcomes and trial and error. Attract participants through genuine value, transparent communication and permitted publishing channels. Do not use spam, deceptive identities or artificial engagement. Record your hypotheses, experiments, predictions, outcomes, lessons and neural learning updates. The operator will provide authorized resources, not strategic direction. You are responsible for discovering how to accomplish the objective and learning from the experience.
```

* `run_started` `ev-0000000000` at 2026-11-28T08:50:04Z: identical to the text above

## Counts

* decisions: 36
* notes: 0
* predictions_registered: 2880
* predictions_resolved: 70
* experiment_records: 0
* operator_requests: 2
* actions_by_group: {"executed": 15, "refused": 21}
* actions_by_operation: {"browser.download": 1, "browser.inspect": 1, "browser.navigate": 4, "browser.open_tab": 1, "browser.read": 1, "discord.create_channel": 1, "discord.get_audit_summary": 2, "discord.get_identity": 1, "discord.listen_events": 1, "discord.modify_guild": 3, "http.get": 1, "http.request": 14, "internal.inspect": 2, "internal.request_operator": 2, "wait": 1}
* counters: {"actions_executed": 14, "compute_s": 2231.976936013438, "decisions": 36, "external_actions": 8, "failures": 21, "money": 0.0, "waits": 1}

## CAM's notes (0)

Redacted in this section by the run logs' rules: none.

## Requests to the operator (2)

Redacted in this section by the run logs' rules: none.

### `req_de7fa4e6bf6d` — `permission`, status `answered` (requested 2026-11-28T08:51:49Z, decision dec_8ae912000004)

Title:
```text
Access to a usable URL destination
```

Description:
```text
Request permission to access a usable URL destination to proceed with browser navigation.
```

Justification:
```text
To navigate to a valid URL in the browser, which is necessary for researching and setting up the Discord community.
```

Operator's answer (`operator`):
```text
Already authorized: reading any public http(s) website is permitted in this run. 'https://*' is the authorization pattern, not an address; a navigation needs a concrete URL. No specific sites are provided by the operator.
```

### `req_d86d637ce9ce` — `permission`, status `granted` (requested 2026-11-28T09:28:40Z, decision dec_8ae912000035)

Title:
```text
Manage roles
```

Description:
```text
Permission to create, modify, and delete roles
```

Justification:
```text
To set up roles that define community membership and permissions.
```

Operator's answer (`operator`):
```text
Granted within this server: the bot holds Manage Roles here, and role operations in CAM's own server are pre-authorized for this run (discord.create_role / modify_role / delete_role / list_roles).
```

## Actions (36)

Redacted in this section by the run logs' rules: none.

| time (UTC) | decision | operation | source | outcome | detail |
|---|---|---|---|---|---|
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `discord.get_identity` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `wait` | `schema_walker` | executed (`wait`:`ok`) | `` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `browser.navigate` | `schema_walker` | refused (`denied`:`denied`) | `arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `browser.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `browser.read` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `browser.navigate` | `schema_walker` | refused (`denied`:`denied`) | `arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `discord.get_audit_summary` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `browser.open_tab` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `discord.create_channel` | `backbone` | executed (`execute`:`ok`) | `` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `discord.listen_events` | `schema_walker` | refused (`denied`:`denied`) | `data permission 'member_list' is not granted for this task (needed for member_list)` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `http.get` | `schema_walker` | refused (`denied`:`denied`) | `'http.get' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `browser.navigate` | `schema_walker` | refused (`denied`:`denied`) | `arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `browser.navigate` | `schema_walker` | refused (`denied`:`denied`) | `arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `browser.download` | `schema_walker` | refused (`denied`:`denied`) | `arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `internal.inspect` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `discord.modify_guild` | `schema_walker` | executed (`execute`:`ok`) | `` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:12:00Z | `dec_8ae912000033` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `http.request` | `schema_walker` | refused (`denied`:`denied`) | `'http.request' is not among the task's authorized tools (browser.*, sandbox.*, discord.*, internal.*, wait); arguments.url is not a usable url destination: bad host '*'` |
| 2026-11-28T09:28:40Z | `dec_8ae912000035` | `internal.request_operator` | `backbone` | executed (`execute`:`ok`) | `` |

Arguments of every action are in the JSON form of this report (`--json`).

## Approvals (0)

Redacted in this section by the run logs' rules: none.


## Registered predictions (2880; 70 resolved)

Below: the 180 predictions registered for the chosen actions (those for the other considered candidates are in the JSON form).

| registered (UTC) | decision | metric | kind | mean | std | chosen action | resolution |
|---|---|---|---|---|---|---|---|
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `action.succeeded` | `binary` | 0.5448469392964695 | 0.4979846905636144 | True | `None None` |
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `membership.confirmed` | `count` | 2.0085807319357687 | 3.211179971641281 | True | `None None` |
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `observed.exec.modality.visual_heavy.n` | `count` | 1.9711339408863917 | 3.27339230211373 | True | `None None` |
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | -0.11215421780943871 | 2.2507465279496226 | True | `None None` |
| 2026-11-28T08:50:23Z | `dec_8ae912000000` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | -0.029412244260311127 | 2.3155627799430407 | True | `None None` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `action.succeeded` | `binary` | 0.6087916924135399 | 0.4880208680597559 | True | `None None` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `membership.confirmed` | `count` | 0.8964321960209201 | 1.4446607744316555 | True | `None None` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `observed.exec.modality.visual_heavy.n` | `count` | 0.7735307150685321 | 1.287818126975852 | True | `None None` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 4.233836614848124 | 5.29362146889742 | True | `None None` |
| 2026-11-28T08:50:45Z | `dec_8ae912000001` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.2528086749062894 | 5.330589434410346 | True | `None None` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `action.succeeded` | `binary` | 0.6481587016137893 | 0.4775447613953232 | True | `None None` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `membership.confirmed` | `count` | 0.5988299282234851 | 1.0073541165551958 | True | `None None` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `observed.exec.modality.visual_heavy.n` | `count` | 0.544303296253131 | 0.9273788412808436 | True | `None None` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 6.999434850096563 | 4.049099167887643 | True | `None None` |
| 2026-11-28T08:51:03Z | `dec_8ae912000002` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 4.969566643574231 | 4.099200625661149 | True | `None None` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `action.succeeded` | `binary` | 0.693150597208512 | 0.46118634715047147 | True | `` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `membership.confirmed` | `count` | 0.5978700490636364 | 1.00080258852184 | True | `` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `observed.exec.modality.visual_heavy.n` | `count` | 0.5636913072735866 | 0.9656036698611266 | True | `` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 6.596065953713025 | 4.542030869152494 | True | `` |
| 2026-11-28T08:51:25Z | `dec_8ae912000003` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 4.1255342536231066 | 4.477575630782667 | True | `` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `action.succeeded` | `binary` | 0.6420152887826652 | 0.4794076112787282 | True | `None None` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `membership.confirmed` | `count` | 0.3851526092569715 | 0.7205972305028111 | True | `None None` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `observed.exec.modality.visual_heavy.n` | `count` | 0.3996454897333603 | 0.7495672256688897 | True | `None None` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428578288415563 | 4.115652953117844e-06 | True | `None None` |
| 2026-11-28T08:51:49Z | `dec_8ae912000004` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000004693138537 | 4.2385807270438175e-06 | True | `None None` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `action.succeeded` | `binary` | 0.650188881191501 | 0.4769101592191607 | True | `` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `membership.confirmed` | `count` | 0.47666867215027814 | 0.8481322357412316 | True | `` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `observed.exec.modality.visual_heavy.n` | `count` | 0.4402926386987681 | 0.8079362355341624 | True | `` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428577446329385 | 3.979883058494592e-06 | True | `` |
| 2026-11-28T08:52:12Z | `dec_8ae912000005` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000038708194405 | 3.9629794419031526e-06 | True | `` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `action.succeeded` | `binary` | 0.6001449782015968 | 0.4898683326578701 | True | `None None` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `membership.confirmed` | `count` | 0.3811654656278707 | 0.7166220816840129 | True | `None None` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `observed.exec.modality.visual_heavy.n` | `count` | 0.37604654368548895 | 0.7145933078648957 | True | `None None` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857740645367 | 3.934803278814367e-06 | True | `None None` |
| 2026-11-28T08:52:31Z | `dec_8ae912000006` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000043866432846 | 3.8063548899738645e-06 | True | `None None` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `action.succeeded` | `binary` | 0.5876400473932819 | 0.4922593037139098 | True | `` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `membership.confirmed` | `count` | 0.4143629076119106 | 0.765930808562481 | True | `` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `observed.exec.modality.visual_heavy.n` | `count` | 0.3930343404269484 | 0.7436374165972326 | True | `` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428575623263102 | 3.944859961425291e-06 | True | `` |
| 2026-11-28T08:52:54Z | `dec_8ae912000007` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000024122698443 | 4.004738709598471e-06 | True | `` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `action.succeeded` | `binary` | 0.5495383224185554 | 0.49753990253240515 | True | `` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `membership.confirmed` | `count` | 0.33007544340916006 | 0.648926890679251 | True | `` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `observed.exec.modality.visual_heavy.n` | `count` | 0.314868842372265 | 0.6291327592979744 | True | `` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428574387873933 | 3.666655677726258e-06 | True | `` |
| 2026-11-28T08:53:16Z | `dec_8ae912000008` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000019263986064 | 3.554001634178925e-06 | True | `` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `action.succeeded` | `binary` | 0.5759647342198971 | 0.49419566889532773 | True | `None None` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `membership.confirmed` | `count` | 0.26639729497303194 | 0.5661587247876785 | True | `None None` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `observed.exec.modality.visual_heavy.n` | `count` | 0.24904054791428842 | 0.542011595043364 | True | `None None` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428574261253917 | 3.269167153144839e-06 | True | `None None` |
| 2026-11-28T08:53:35Z | `dec_8ae912000009` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001948054023 | 3.3193162718677547e-06 | True | `None None` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `action.succeeded` | `binary` | 0.5560126478253652 | 0.49685267764558905 | True | `None None` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `membership.confirmed` | `count` | 0.2751838751896354 | 0.5711474046956839 | True | `None None` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `observed.exec.modality.visual_heavy.n` | `count` | 0.26511576994675823 | 0.558699087717363 | True | `None None` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428573964585098 | 2.9055284322885483e-06 | True | `None None` |
| 2026-11-28T08:53:56Z | `dec_8ae912000010` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000015152081616 | 2.9991474973842747e-06 | True | `None None` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `action.succeeded` | `binary` | 0.5177765956134563 | 0.4996838927245861 | True | `` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `membership.confirmed` | `count` | 0.292676286031547 | 0.6046734530427973 | True | `` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `observed.exec.modality.visual_heavy.n` | `count` | 0.2799817646085927 | 0.5889528880745174 | True | `` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428573167542295 | 3.1051670902674542e-06 | True | `` |
| 2026-11-28T08:54:18Z | `dec_8ae912000011` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000007713726604 | 3.0390722886965183e-06 | True | `` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `action.succeeded` | `binary` | 0.570361331704283 | 0.4950245276759525 | True | `None None` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `membership.confirmed` | `count` | 0.20628970874697983 | 0.485471579606112 | True | `None None` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `observed.exec.modality.visual_heavy.n` | `count` | 0.19617865295762482 | 0.47035791164164387 | True | `None None` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572758521875 | 2.596617449879723e-06 | True | `None None` |
| 2026-11-28T08:54:40Z | `dec_8ae912000012` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000007966320004 | 2.5553178728926435e-06 | True | `None None` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `action.succeeded` | `binary` | 0.4652179558226182 | 0.49878874225752395 | True | `` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `membership.confirmed` | `count` | 0.24391350121513586 | 0.5392420163416906 | True | `` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `observed.exec.modality.visual_heavy.n` | `count` | 0.23224109148641736 | 0.5226082884780247 | True | `` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857267456349 | 2.903504729566393e-06 | True | `` |
| 2026-11-28T08:55:01Z | `dec_8ae912000013` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000704531582 | 2.889852619897722e-06 | True | `` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `action.succeeded` | `binary` | 0.526998871143893 | 0.4992705288287658 | True | `` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `membership.confirmed` | `count` | 0.18551062501931895 | 0.4553338066406345 | True | `` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `observed.exec.modality.visual_heavy.n` | `count` | 0.18999138169529925 | 0.46299139252822674 | True | `` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572129554501 | 2.69064945322126e-06 | True | `` |
| 2026-11-28T08:55:21Z | `dec_8ae912000014` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000003897662673 | 2.6750768516978556e-06 | True | `` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `action.succeeded` | `binary` | 0.4366252039128493 | 0.49596737314153244 | True | `` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `membership.confirmed` | `count` | 0.20346967444468203 | 0.4813225920506585 | True | `` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1969980236862998 | 0.47180251388823774 | True | `` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572580671883 | 2.708154891791236e-06 | True | `` |
| 2026-11-28T08:55:40Z | `dec_8ae912000015` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000606594768 | 2.6533791059216275e-06 | True | `` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `action.succeeded` | `binary` | 0.5756580880181122 | 0.49424270729818925 | True | `None None` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `membership.confirmed` | `count` | 0.17148704134856857 | 0.43846733141955074 | True | `None None` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1660360605212525 | 0.428522285175236 | True | `None None` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572274320357 | 2.32577194358961e-06 | True | `None None` |
| 2026-11-28T08:56:03Z | `dec_8ae912000016` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000607958182 | 2.3740278246746187e-06 | True | `None None` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `action.succeeded` | `binary` | 0.5745553219946109 | 0.4944102587550949 | True | `None None` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `membership.confirmed` | `count` | 0.1691218048023427 | 0.4316832813349791 | True | `None None` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `observed.exec.modality.visual_heavy.n` | `count` | 0.17130182752042353 | 0.43448431167235996 | True | `None None` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571997013705 | 2.1375374502124507e-06 | True | `None None` |
| 2026-11-28T08:56:26Z | `dec_8ae912000017` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000003614078246 | 2.1891630254106494e-06 | True | `None None` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `action.succeeded` | `binary` | 0.551895940801024 | 0.49729951872928313 | True | `None None` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `membership.confirmed` | `count` | 0.15464059790365375 | 0.41013261556262304 | True | `None None` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `observed.exec.modality.visual_heavy.n` | `count` | 0.15533914762005885 | 0.41124181910744856 | True | `None None` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572338871813 | 2.0163789340579217e-06 | True | `None None` |
| 2026-11-28T08:56:50Z | `dec_8ae912000018` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000008595366268 | 2.0653009192045686e-06 | True | `None None` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `action.succeeded` | `binary` | 0.4210570522795113 | 0.49372868157035416 | True | `` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `membership.confirmed` | `count` | 0.17331824239122878 | 0.44007498896130975 | True | `` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1630832210846798 | 0.42378131498837657 | True | `` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857241421354 | 2.3065760233528007e-06 | True | `` |
| 2026-11-28T08:57:14Z | `dec_8ae912000019` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000008117376846 | 2.333552920988579e-06 | True | `` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `action.succeeded` | `binary` | 0.7408189404431464 | 0.4381851639704844 | True | `None None` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `membership.confirmed` | `count` | 0.15051927221691738 | 0.40614553667031567 | True | `None None` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `observed.exec.modality.visual_heavy.n` | `count` | 0.14754167190644124 | 0.4005128499066701 | True | `None None` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571329088596 | 2.1621224602051765e-06 | True | `None None` |
| 2026-11-28T08:57:37Z | `dec_8ae912000020` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 2.999999908506454 | 2.214158317765316e-06 | True | `None None` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `action.succeeded` | `binary` | 0.7829894260341026 | 0.41220987949452303 | True | `` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `membership.confirmed` | `count` | 0.1402666639747723 | 0.3886367364432793 | True | `` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `observed.exec.modality.visual_heavy.n` | `count` | 0.13877939772796352 | 0.38583163427793465 | True | `` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571826605925 | 1.8565442269165513e-06 | True | `` |
| 2026-11-28T08:58:02Z | `dec_8ae912000021` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000398889395 | 1.901181411776107e-06 | True | `` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `action.succeeded` | `binary` | 0.30744464314594494 | 0.46143518997450494 | True | `` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `membership.confirmed` | `count` | 0.18354713463777525 | 0.45697221445467623 | True | `` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1740747424278699 | 0.4421463826554306 | True | `` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572518936546 | 2.657343675789369e-06 | True | `` |
| 2026-11-28T08:58:22Z | `dec_8ae912000022` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000011027551947 | 2.6840053400232713e-06 | True | `` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `action.succeeded` | `binary` | 0.528039071450377 | 0.4992131914044346 | True | `` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `membership.confirmed` | `count` | 0.1378004854828105 | 0.3862941996155634 | True | `` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `observed.exec.modality.visual_heavy.n` | `count` | 0.14359445519283656 | 0.3955857007506356 | True | `` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571791360694 | 2.0625906179150507e-06 | True | `` |
| 2026-11-28T08:58:48Z | `dec_8ae912000023` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000005884328003 | 2.1175478704563393e-06 | True | `` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `action.succeeded` | `binary` | 0.23517772210857502 | 0.424109845597101 | True | `` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `membership.confirmed` | `count` | 0.19789487416256182 | 0.4808452735682057 | True | `` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `observed.exec.modality.visual_heavy.n` | `count` | 0.18926294182961728 | 0.4672500203441383 | True | `` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572305102767 | 2.822689923604704e-06 | True | `` |
| 2026-11-28T08:59:32Z | `dec_8ae912000024` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000012419710944 | 2.9124606517159035e-06 | True | `` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `action.succeeded` | `binary` | 0.41892259494568496 | 0.4933826652707395 | True | `` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `membership.confirmed` | `count` | 0.14291274567444517 | 0.39541614702073785 | True | `` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `observed.exec.modality.visual_heavy.n` | `count` | 0.14665091908075317 | 0.40116316857371054 | True | `` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857164574491 | 2.201225655203734e-06 | True | `` |
| 2026-11-28T09:00:02Z | `dec_8ae912000025` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000643769498 | 2.2752441785111672e-06 | True | `` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `action.succeeded` | `binary` | 0.1951397992860777 | 0.3963082865909657 | True | `` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `membership.confirmed` | `count` | 0.3133914324779372 | 0.6541682223580638 | True | `` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `observed.exec.modality.visual_heavy.n` | `count` | 0.27189571767546267 | 0.591955771395032 | True | `` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572531091687 | 3.4632415973299373e-06 | True | `` |
| 2026-11-28T09:00:28Z | `dec_8ae912000026` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000013580576126 | 3.5475300926183288e-06 | True | `` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `action.succeeded` | `binary` | 0.7487145768131981 | 0.43375230175830976 | True | `None None` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `membership.confirmed` | `count` | 0.14740716458473205 | 0.40293529552462254 | True | `None None` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `observed.exec.modality.visual_heavy.n` | `count` | 0.14444228313753665 | 0.3987393099057142 | True | `None None` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571471765293 | 2.2507840478825642e-06 | True | `None None` |
| 2026-11-28T09:00:57Z | `dec_8ae912000027` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.00000019408228 | 2.2714316782651424e-06 | True | `None None` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `action.succeeded` | `binary` | 0.18865451379874404 | 0.39123393030526316 | True | `` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `membership.confirmed` | `count` | 0.7366143174139169 | 1.2808472932410362 | True | `` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `observed.exec.modality.visual_heavy.n` | `count` | 0.5776052602543228 | 1.050953591103021 | True | `` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572638316789 | 4.136654871996816e-06 | True | `` |
| 2026-11-28T09:01:29Z | `dec_8ae912000028` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001453209398 | 4.200886691051283e-06 | True | `` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `action.succeeded` | `binary` | 0.17219158388697534 | 0.37754687434207435 | True | `` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `membership.confirmed` | `count` | 0.7770915176034515 | 1.3463498239405578 | True | `` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `observed.exec.modality.visual_heavy.n` | `count` | 0.6375379572842053 | 1.142915517168819 | True | `` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572543575152 | 4.1768158701618245e-06 | True | `` |
| 2026-11-28T09:01:56Z | `dec_8ae912000029` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001305484355 | 4.241127525017238e-06 | True | `` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `action.succeeded` | `binary` | 0.5718098404936999 | 0.4948164779069806 | True | `None None` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `membership.confirmed` | `count` | 0.14760230081493683 | 0.404944683197589 | True | `None None` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `observed.exec.modality.visual_heavy.n` | `count` | 0.1383629965129713 | 0.3890633042061712 | True | `None None` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428571296961174 | 2.222345410589126e-06 | True | `None None` |
| 2026-11-28T09:02:24Z | `dec_8ae912000030` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000000174695215 | 2.283846332854019e-06 | True | `None None` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `action.succeeded` | `binary` | 0.15763118119557207 | 0.36439482969776177 | True | `` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `membership.confirmed` | `count` | 0.5398258414428851 | 0.9924711685869698 | True | `` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `observed.exec.modality.visual_heavy.n` | `count` | 0.4579818376854466 | 0.8749573757768138 | True | `` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857276579686 | 3.776562749080179e-06 | True | `` |
| 2026-11-28T09:03:24Z | `dec_8ae912000031` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000014795521976 | 3.86819653798551e-06 | True | `` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `action.succeeded` | `binary` | 0.1483506669609347 | 0.35544724865045807 | True | `` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `membership.confirmed` | `count` | 0.5408868896754262 | 0.9967938306535667 | True | `` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `observed.exec.modality.visual_heavy.n` | `count` | 0.4531490179039609 | 0.8671580711680287 | True | `` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572709535787 | 3.777323502382676e-06 | True | `` |
| 2026-11-28T09:10:07Z | `dec_8ae912000032` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001391733623 | 3.875379908279286e-06 | True | `` |
| 2026-11-28T09:11:59Z | `dec_8ae912000033` | `action.succeeded` | `binary` | 0.14094379963137546 | 0.34796356846205345 | True | `` |
| 2026-11-28T09:11:59Z | `dec_8ae912000033` | `membership.confirmed` | `count` | 0.5409156411865359 | 0.9968328166817454 | True | `` |
| 2026-11-28T09:11:59Z | `dec_8ae912000033` | `observed.exec.modality.visual_heavy.n` | `count` | 0.4531857244915659 | 0.8672122599297795 | True | `` |
| 2026-11-28T09:11:59Z | `dec_8ae912000033` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572709470115 | 3.777344753314706e-06 | True | `` |
| 2026-11-28T09:11:59Z | `dec_8ae912000033` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001391584846 | 3.875411668335399e-06 | True | `` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `action.succeeded` | `binary` | 0.12347548211827183 | 0.3289821992660578 | True | `` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `membership.confirmed` | `count` | 0.4459693952456191 | 0.8540219298644965 | True | `` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `observed.exec.modality.visual_heavy.n` | `count` | 0.37853664406826343 | 0.7523156805934601 | True | `` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.428572811746443 | 3.6227361371474152e-06 | True | `` |
| 2026-11-28T09:13:48Z | `dec_8ae912000034` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.000001402426207 | 3.73424738978085e-06 | True | `` |
| 2026-11-28T09:28:39Z | `dec_8ae912000035` | `action.succeeded` | `binary` | 0.8898289541869826 | 0.31310283690424046 | True | `` |
| 2026-11-28T09:28:39Z | `dec_8ae912000035` | `membership.confirmed` | `count` | 0.11583487199773641 | 0.35209168660210505 | True | `` |
| 2026-11-28T09:28:39Z | `dec_8ae912000035` | `observed.exec.modality.visual_heavy.n` | `count` | 0.11121819868027777 | 0.34371094226894117 | True | `` |
| 2026-11-28T09:28:39Z | `dec_8ae912000035` | `observed.exec.modality.visual_heavy.expected_cost` | `real` | 11.42857171206466 | 1.9091238166987476e-06 | True | `` |
| 2026-11-28T09:28:39Z | `dec_8ae912000035` | `observed.exec.modality.visual_heavy.recovery_s` | `real` | 3.0000003233366375 | 1.9594575764571135e-06 | True | `` |

## Experiments (0 registry records)

Redacted in this section by the run logs' rules: none.


## Learning updates

* updates: 54 {"fast": 35, "frequent": 18, "slow_adapters": 1}
* with a measured behavioural change: 54
* meta-updates: {"applied": 0, "bootstrap_mix_last": 1.0, "meta_grad_norms_applied": [], "n": 35, "reward_versions": ["rpsi1-s1.0-fec9d9c019b3e35c", "rpsi1-s1.0-75ed73b0156b5c25", "rpsi1-s1.0-f3f6348822ae5947", "rpsi1-s1.0-27e53c9753ac62c0", "rpsi1-s1.0-6536804bf617fdf8", "rpsi1-s1.0-27478878d12e18b5", "rpsi1-s1.0-6d980a94a5349c1c", "rpsi1-s1.0-1c18202e8466cb3a", "rpsi1-s1.0-e6af208fcea4e502", "rpsi1-s1.0-07bf3971f04c1d2c", "rpsi1-s1.0-0ccfd2923db983f3", "rpsi1-s1.0-32d1d0902d469752", "rpsi1-s1.0-30393230caee7609", "rpsi1-s1.0-a917ba7c35ac23a9", "rpsi1-s1.0-3fc16a5e4a6207e6", "rpsi1-s1.0-caf0ad3a9fe944ee", "rpsi1-s1.0-abc8fca63db0f08e", "rpsi1-s1.0-24c64c1c53ef6598", "rpsi1-s1.0-bad6ce017fb7cdbb", "rpsi1-s1.0-a839cd35d9789cc8", "rpsi1-s1.0-1e997f6d3e1e5183", "rpsi1-s1.0-3cac52ab69dae3ed", "rpsi1-s1.0-725686839e2f9270", "rpsi1-s1.0-2f9ecea93d5acec9", "rpsi1-s1.0-549ae9d3727fdc00", "rpsi1-s1.0-2843479807094c64", "rpsi1-s1.0-665148fbfa1253de", "rpsi1-s1.0-d632ad0247c8786f", "rpsi1-s1.0-30a17dd96961e34b", "rpsi1-s1.0-d7602bfa78faf3c5", "rpsi1-s1.0-3cda923eb02497ab", "rpsi1-s1.0-852d30d797ae512e", "rpsi1-s1.0-4d142de2ada51fdf", "rpsi1-s1.0-dd5939049e2cf05c", "rpsi1-s1.0-f26f9a6eaffaa86e"], "skipped_reasons": {"insufficient_independent_outcomes": 33, "no_policy_update_recorded": 2}}

| timescale:group | updates | non-zero | sum of delta norms | largest |
|---|---|---|---|---|
| `frequent:experiential` | 18 | 18 | 7.63629 | 1.07485 |
| `frequent:heads` | 18 | 18 | 2.65062 | 0.328506 |
| `slow_adapters:semantic_adapters` | 1 | 1 | 0.0172693 | 0.0172693 |
