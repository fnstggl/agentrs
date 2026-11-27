# harness/auth — integration notes (workstream EA)

The independent authorization layer. Other workstreams touch it only through `contracts.PolicyEngine` and
`ExecContext.secrets`. Security statement: `SECURITY.md`. Policy file: `POLICY_SCHEMA.md`. What the verification
round G-sec / G-env changed: `FIXES.md`.

## Getting a PolicyEngine

```python
from cam_live.harness.auth import PolicyService            # in-process (tests, diagnostics, staging)
policy = PolicyService(run_dir, policy_path=None, clock=None)      # policy_path=None: the run policy (below)

from cam_live.harness.auth.client import connect           # separated mode: same interface over R/sock/auth.sock
policy = connect(run_dir)
```

## The call sequence every modality must follow

1. **Runtime (learner uid):** `d = policy.authorize(action, task, context)`.
   `d.verdict` is `allow` (with `d.grant`), `deny` (never retry; `is_resource_denial(d)` tells budget/rate apart) or
   `needs_approval` (`d.approval_id`; re-issue the *identical* action after the operator approved).
   `context` is an allowlist — only `page_url`, `tab_id`, `session_id`, `decision_id`, `experiment_id` are read;
   anything else is dropped. Never put page text in it.
2. **Environment (executor uid), once, immediately before executing:** `policy.check_grant(action, d)` → `True`.
   This **consumes** the grant (single use) and opens the action's execution window
   (`timeout_s` + 30 s, at most 900 s). A second call returns `False`. Use `consume=False` only to look.
3. **Executor, for every primitive that leaves the process:** `ctx.check(op, action)` = `policy.check_primitive`.
   Put the facts the executor measured into `op.arguments`: `url` (navigation / request), `method` (requests),
   `page_url` (interaction on a page), `tab_id`. Report outgoing requests as `PrimitiveOp("net.request", modality,
   {"method": ..., "url": ...})` — this is the only place redirects and form submissions are seen.
   `needs_approval` / `deny` here means: stop the action and return that status.
4. **Environment, after execution:** `policy.record(action, result)` with the real `result.cost`. Without it the
   action is charged its estimate when the window closes.

`action.content_hash()` is the binding everywhere (grants, approvals): any edit, including `action_id` or
`rationale`, needs a new authorization. An action whose destination is *the page the browser is on* (a click or
typing without `page_url`) is in addition bound to that page: an approval given on one page is not valid on
another.

**What a granted action may do.** An approval or pre-authorization covers, at primitive level, only a primitive of
the same tool class, the same HTTP method and the *exact* destination (scheme, host, port and full path) that the
action declared. A grant on a logical destination (`channel:<platform>:<id>`, `account:…`) covers the same
operation on that id, or the HTTP request the platform's `transport` table maps that operation and id to (e.g.
`discord.send_message` on channel 123 covers `POST /api/v*/channels/123/messages` and nothing else). Anything
else the executor reports needs its own approval or is denied.

**How limits are charged.** Every declared step (each node of `ActionSpec.steps`) is one external action and one
hit of every rate rule it falls under, all at authorization time — a program that declares more writes than the
window has left is refused before anything runs. Every external-effect primitive is charged as well: one that
realises a declared step uses that step's share, any other is checked against the remaining budget (external
actions, per-destination write rate, the task-wide `write_global` rate, money) and charged anew. A skill step that
the skill executor announces before dispatch (`PrimitiveOp.note` starting with `skill:`) is a step declared at run
time: it is charged like a declared step and prepays the one realisation its executor reports next. Decisions are
counted per `authorize`. Clicks and typing are *interactions*, not external writes (rate: the per-step rule).

## Secrets

`ExecContext.secrets = policy.secrets_handle().bind(action, decision)` (in-process) or
`connect(run_dir).secrets.bind(action, decision)` (executor uid). Then inside an executor:

```python
token = ctx.secrets.get("site_token", "https://api.example.org/v1/items")   # only for this granted, running action
obs = ctx.secrets.scrub_record(obs)                                          # before anything is returned
```

Actions never carry credentials: a credential field holds a reference `"secret:<name>"`, which the executor
resolves. A literal value in `password` / `token` / `authorization` … is denied as credential leakage.

## What the task must say (`TaskSpec`)

* `mode`: `diagnostic` (nothing leaves the machine), `staging` (loopback staging services), `live` (only with a
  signed operator record — see below). A task is pinned by `objective_id` on first use; a different `TaskSpec`
  under the same id is denied (`P-TASK-MISMATCH`) until the operator calls `register_task`.
* `interfaces`: modality names and/or tool patterns (`"browser.*"`). Empty = mode default; in live mode empty = none.
* `resources` keys read by the policy (staging; in live mode the record's lists are used instead):
  `owned` / `owned_destinations` / `accounts`, `read` / `allowed` / `allowed_destinations`, `deny`,
  `preauthorized`, `opted_in_recipients`. Values are destination patterns: `host`, `*.host`, `https://host/path/*`,
  `host:port`, `channel:<platform>:<id>`, `account:<platform>:<id>`.
* `resources.rate_limits` (staging only; in live mode the signed record's `rate_limits`):
  `{"write": {"count", "window_s"}, "write_global": {...}}` — clamped to the mode's `rate_caps` (count at most, window
  at least the cap's).
* `limits`, `data_permissions` (`aggregate_metrics`, `read_message_content`, `store_message_content`,
  `train_on_message_content`, `member_list`, `collect_personal_data`).

## Local services: what of this machine an action may reach

No mode reaches this instance's own services (the instance portal on 127.0.0.1:11111, Jupyter, SSH, tunnel
managers, IDE servers, the metadata service). Every spelling of loopback (`localhost`, `localhost.`, `*.localhost`,
any 127.x address, `::1`, `0.0.0.0`, `::`, decimal / octal / hex / short IPv4, IPv4-mapped IPv6), link-local and
metadata addresses, private networks, this machine's own interface addresses and names `/etc/hosts` maps to them are
denied (`P-LOCAL-ADDRESS`) unless the exact `host:port` is a **registered local service**:

1. listed in the policy file (`destinations.local_services`; the default lists `127.0.0.1:8765`, the reference
   staging port of the EA tests and probes);
2. registered by the operator at run start: `policy.register_local_service("127.0.0.1:18600", by, note)`, or
   `opctl --run-dir R local-service add 127.0.0.1:18600` (also `remove`, `list`). Whoever starts staging services
   for a run (the env service's `--staging-sites`, `tools/discord_live_check.py`, a test) registers their addresses
   this way; every registration is in the audit log;
3. in modes with `own_process_services: true` (staging in the default policy): a loopback port whose every
   listening socket belongs to the process the authorization service runs in — staging sites started in-process.
   A port another process listens on never qualifies.

Executors that know the address they actually connected to report it as `peer_ip` in the primitive; a public name
that resolved to this machine is then refused too.

## Run policy: which policy file a run uses

`cam_live/configs/authorization.yaml` is the run policy. Its content is the fixed default
(`src/cam_live/harness/auth/default_policy.yaml`; a test pins that both are byte-identical). A run selects it as
follows:

* **In-process** (`PolicyService(run_dir)` with `policy_path=None`, which is what `runtime/loop_service.build_inproc`
  and the probes use): `schema.run_policy_path()` — `configs/authorization.yaml` when it exists (a source checkout),
  otherwise the packaged `default_policy.yaml`. The service logs the path it loaded (`service_start.policy_path`).
* **Separated** (`launcher`): `LauncherConfig.policy_path` / `--policy P`, else the same default; the launcher copies
  the file to `R/auth/policy.yaml` (owned by the authorization uid) and starts the service with `--policy` on that
  copy, so the learner can neither read nor change it.
* **Changing it during a run:** edit a copy, then `opctl --run-dir R reload-policy <path>`. A file that does not
  validate leaves the service denying everything (`P-POLICY-INVALID`).

To run with different rules, pass another file explicitly; do not edit `configs/authorization.yaml` in place unless
the change is meant to become the default (then change `default_policy.yaml` with it).

## Operator

```
python -m cam_live.harness.auth.launcher --run-dir R [--services auth,exec,...] [--policy P]
python -m cam_live.harness.auth.opctl --run-dir R approvals | approve ID | reject ID | stop "why" | reset-stop |
        pause "why" | resume | revoke SCOPE | reload-policy [P] | limits | secrets | audit-tail [N] | verify-audit |
        put-secret NAME --dest PATTERN   (value on stdin) | local-service add HOST:PORT | remove HOST:PORT | list
python -m cam_live.harness.auth.opctl init-operator-key --private K --public P        # once; K stays with the operator
python -m cam_live.harness.auth.opctl sign-live record.json --private K > signed.json
python -m cam_live.harness.auth.opctl --run-dir R register-live signed.json
```

Live authorization record: see `harness/auth/liveauth.py` (task id + `TaskSpec.content_hash()`, accounts, owned /
read / preauthorized destinations, budgets, time window, Ed25519 signature; the service holds only the public key,
installed with `launch(..., config={"operator_pubkey": P})`).

`resolve_approval`, `reset_stop`, `resume`, `revoke`, `reload_policy`, `register_task`,
`register_live_authorization`, `put_secret`, `register_local_service`, `unregister_local_service` are accepted only
from root. The live record may also carry `rate_limits` (clamped to the policy's live `rate_caps`). A dashboard running as the learner uid can
*list* approvals and *engage* the stop; approving and resetting go through a root-owned process (`opctl`).

## Emergency stop

`policy.emergency_stop(reason, by)`, `opctl stop` (also with the service down) or `touch R/auth/STOP` — all three
are equivalent. The flag is checked first by every call; the first time the service sees a stop flag (however it
was set, also at start-up after an offline stop) it voids every outstanding grant, and `reset_stop` voids every grant
still outstanding, so a grant issued before a stop is never usable after the reset. Survives restarts; the launcher
terminates every service except the authorization service and restarts nothing until `reset-stop` +
`handle.resume()`. Use `opctl reset-stop`, not `rm R/auth/STOP`: a flag created and removed by hand while the
service never looked is invisible to it.

## Tool names

The policy's tool table (`default_policy.yaml`, `tools:`) maps operation names to side-effect classes; an
operation that matches no entry is **denied**. It currently knows: `wait`, `inspect.*`, `internal.*`, `program`
(each step judged on its own), `lab.*` (diagnostic), `browser.*` (`browser.request` = a cross-origin subresource,
read or write by method), `visual.*`, `sandbox.*`, `code.*`, `job.*`, `file.*`, `net.request`, `http.*`, `api.*`,
`publish.*`, `discord.*` (`discord.listen_events` and `discord.get_member_events` are aggregate reads), `*.dm`,
`payment.*`, `skill.*`. New tools need a line there (data, not code).
