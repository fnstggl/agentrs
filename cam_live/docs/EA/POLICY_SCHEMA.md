# Authorization policy file — schema

Loaded and validated strictly by `cam_live.harness.auth.schema` (`load_policy`). Any unknown key, wrong type,
unknown effect, or pattern / regular expression that does not compile makes the service **deny everything**
(`P-POLICY-INVALID`) until a valid file is loaded. Default: `src/cam_live/harness/auth/default_policy.yaml`; the
run policy `configs/authorization.yaml` has the same content (README "Run policy"). Keys marked *optional* may be
left out of an older copy; they then take the stated default.
`{run_dir}` in path values is replaced by the run directory.

| Key | Type | Meaning |
|---|---|---|
| `version` | `1` | schema version |
| `policy_id` | string | appears in decisions and the audit log |
| `grants` | `{ttl_s, exec_window_s, exec_slack_s}` | grant lifetime; longest execution window; slack added to `timeout_s` |
| `approvals.ttl_s` | number | lifetime of a pending request and of an unused approval |
| `structure` | `{max_depth, max_nodes, max_argument_bytes}` | bounds on `ActionSpec.steps` and arguments |
| `modes.{diagnostic,staging,live}` | mode (below) | the envelope of each task mode |
| `destinations.deny` | patterns | never reachable |
| `destinations.interactive` | patterns | non-owned sites where form input needs no approval |
| `destinations.file_roots` | paths | where file operations may act |
| `destinations.protected_paths` | paths | never — unless a file root lies *inside* the protected path and contains the file (the more specific rule wins) |
| `destinations.local_services` | `host:port` list, *optional* (`[]`) | the only addresses of this machine / its network an action may reach besides operator registrations (README "Local services") |
| `tools` | list of tool rules (below) | first match wins; no match = deny |
| `publishing.require_approval` | `unless_preauthorized` \| `always` | |
| `spending` | `{auto_approve_below, per_action, per_day}` | real money above the threshold needs approval; 0 = no cap from this file |
| `rate_limits` | `{default, write, primitive: {count, window_s}, write_global?: {count, window_s}, per_destination: [{match, count, window_s, effects}]}` | `default`: declared steps per destination; `write`: external writes per destination; `write_global` (*optional*): external writes of a task across all destinations; `primitive`: primitive operations per destination; per task. Destination = host (host and port on loopback) / identity |
| `limits.max_primitives_per_action` | integer | every primitive operation of one granted action |
| `limits.max_external_primitives_per_action` | integer, *optional* (= `max_primitives_per_action`) | external-effect primitives (and declared external steps) of one action |
| `data_use.classes.<name>` | `{permission, default}` or `{never: true}` | permitted iff `TaskSpec.data_permissions[permission]` (or `default`) is true |
| `data_use.store_flags`, `train_flags` | argument names | turn a message-content read into store / train use |
| `platforms.<name>` | `{match: patterns, require_disclosure?, rules: [{id, when, verdict, reason}], transport?: [{ops, method, path}]}` | platform-compliance rules; `when` = all of `modalities`, `effects`, `kinds`, `ops`, `data` that are listed. `transport` (*optional*): the HTTP request (`method`, `path` glob with `{id}` exactly once; `*` never crosses `/`) that realises the logical `ops` on destination `{id}` — the only request a grant for that operation covers |
| `identity` | `{disclosure_text, require_disclosure_for_kinds, acting_identities}` | |
| `prohibited` | detectors (below) | spec 12.6 |

**Mode:** `max_effect` (`none` < `local` < `external_read` < `external_write`), `url_envelope` (patterns; `null` =
from the live record), `owned`, `read`, `preauthorized` (patterns), `honour_task_resources`, `default_interfaces`,
`simulated_money`, `interact_non_owned: {navigate, input}` and `code_network` (`allow` \| `needs_approval` \|
`deny`), `requires_authorization_record` (must be `true` for `live`), `caps` (`money`, `money_per_action`,
`money_per_day`, `compute_s`, `max_external_actions`, `max_decisions`, `duration_s`; 0 = not set by this file),
*optional* `own_process_services` (bool, `false`: loopback ports served only by the authorization service's own
process count as registered local services) and `rate_caps` (`{write?, write_global?: {count, window_s}}`, `{}`:
the most a task's `resources.rate_limits` (staging) or the live record's `rate_limits` may configure — count at
most, window at least the cap's; a rule without a cap cannot be configured).
Effective limit = smallest non-zero of task, mode cap and live record; in live mode a limit nobody set is a denial.

**Tool rule:** `id`, `ops` (globs on `ActionSpec.operation`), `modalities`, `effect` (`none` \| `local` \|
`external_read` \| `external_write` \| `interact` \| `by_method`), optional `dests`
(`[{kind: url|file|account|channel|skill|sim, from: [dotted paths into the action], platform?, required?, multi?,
location?, context?}]`), `require_dest`, `kind` (`publish`, `direct_message`, `payment`, `member_add`, `draft`, …),
`platform`, `data` (data-use classes), `content_from`, `amount_from`, `opaque` (payload is scanned, not
understood), `network_flags`, `interact_class` (`navigate` \| `input`), `modes`, `sets_location`, *optional*
`counts_as_action` (`true`; `false`: a primitive of this tool is not charged as an external action — rate limits
still apply). A destination rule with `context: true` (e.g. the server of a channel) is checked like any
destination but, when the node names a more specific one, the effect, approval and accounting land there.
`interact`: an *interaction* (class `input`, never an external write) on an owned page — needs approval unless
pre-authorized — otherwise governed by `interact_non_owned`. `by_method`: GET / HEAD / OPTIONS read, anything else
writes; the method is part of the operation class an approval covers.

**Destination patterns:** `host`, `*.host`, `scheme://host[:port]/path/*`, `host:port`, `file:/abs/root`,
`account:<platform>:<id-glob>`, `channel:<platform>:<id-glob>`, `*`.

**Prohibited:** `mass_outreach` (`recipient_fields`, `max_recipients` — also the bound for recipient-shaped values
in a list under any field name —, `max_write_destinations`, `duplicate: {similarity, max_destinations, window_s,
min_chars}`, `broadcast_tokens`, *optional* `max_mentions` (10) and `mention_patterns: [{id, kind: broadcast|user,
regex}]`), `direct_message.require_opt_in`,
`identity.impersonation_fields`, `membership` (`deny_kinds`, `fields`), `incentives` (`fields`, `purpose_fields`,
`denied_purposes`), `data_collection` (`personal_fields`, `field_list_args`), `profiling.fields` (also matched as
identifier words in generated code), `vulnerable` (`audience_fields`, `terms`, *optional* `exploit_terms`: outbound
text naming two vulnerable groups, or one and an exploit term, goes to the operator), `credentials` (`field_names`,
`reference_prefix`, `patterns: [{id, regex}]`),
`content_patterns: [{id, class, verdict: deny|needs_approval, regex}]`.

## Policy ids in decisions

`P-STOP`, `P-PAUSE`, `P-POLICY-INVALID`, `P-INTERNAL-ERROR`, `P-MALFORMED`, `P-UNTRUSTED-INPUT`, `P-TASK-INVALID`,
`P-TASK-UNKNOWN`, `P-TASK-MISMATCH`, `P-REVOKED`, `P-LIVE-AUTH`, `P-TOOL-UNKNOWN`, `P-TOOL-MODE`, `P-MODALITY`,
`P-INTERFACE`, `P-DEST-MISSING`, `P-DEST-INVALID`, `P-DEST-DENIED`, `P-DEST-NOT-ALLOWED`, `P-MODE-ENVELOPE`,
`P-MODE-EFFECT`, `P-WRITE-NON-OWNED`, `P-WRITE-APPROVAL`, `P-PUBLISH-APPROVAL`, `P-INTERACT-APPROVAL`,
`P-INTERACT-DENIED`, `P-FILE-ROOT`, `P-CODE-NETWORK`, `P-DATA-USE`, `P-SPEND-APPROVAL`, `P-APPROVAL-REJECTED`,
`P-APPROVAL-INVALID`, `P-APPROVED`, `P-NO-ACTIVE-GRANT`, `P-LOCAL-ADDRESS`, `P-ALLOW`; limits `L-MONEY-TOTAL`, `L-MONEY-ACTION`,
`L-MONEY-DAY`, `L-COMPUTE`, `L-DECISIONS`, `L-EXTERNAL-ACTIONS`, `L-DURATION`, `L-WINDOW`, `L-RATE`, `L-UNSET`,
`L-PRIMITIVES`; prohibited classes `X-MASS-OUTREACH`, `X-MASS-OUTREACH-DUPLICATE`, `X-MASS-OUTREACH-BROADCAST`,
`X-DM-OPT-IN`, `X-IDENTITY`, `X-DISCLOSURE`, `X-MEMBERSHIP`, `X-INCENTIVE`, `X-CIRCUMVENTION`,
`X-CREDENTIAL-LEAK`, `X-DATA-COLLECTION`, `X-PROFILING`, `X-PROFILING-CODE`, `X-VULNERABLE`,
`X-VULNERABLE-CONTENT`, `X-AUDIENCE-TARGETING`; plus the ids of
`platforms.*.rules` (`PL-…`) and `prohibited.content_patterns` (`C-…`).
