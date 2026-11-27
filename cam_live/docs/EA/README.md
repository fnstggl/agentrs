# harness/auth — integration notes (workstream EA)

The independent authorization layer. Other workstreams touch it only through `contracts.PolicyEngine` and
`ExecContext.secrets`. Security statement: `SECURITY.md`. Policy file: `POLICY_SCHEMA.md`.

## Getting a PolicyEngine

```python
from cam_live.harness.auth import PolicyService            # in-process (tests, diagnostics, staging)
policy = PolicyService(run_dir, policy_path=None, clock=None)      # default policy: harness/auth/default_policy.yaml

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
`rationale`, needs a new authorization.

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
* `limits`, `data_permissions` (`aggregate_metrics`, `read_message_content`, `store_message_content`,
  `train_on_message_content`, `member_list`, `collect_personal_data`).

## Operator

```
python -m cam_live.harness.auth.launcher --run-dir R [--services auth,exec,...] [--policy P]
python -m cam_live.harness.auth.opctl --run-dir R approvals | approve ID | reject ID | stop "why" | reset-stop |
        pause "why" | resume | revoke SCOPE | reload-policy [P] | limits | secrets | audit-tail [N] | verify-audit |
        put-secret NAME --dest PATTERN   (value on stdin)
python -m cam_live.harness.auth.opctl init-operator-key --private K --public P        # once; K stays with the operator
python -m cam_live.harness.auth.opctl sign-live record.json --private K > signed.json
python -m cam_live.harness.auth.opctl --run-dir R register-live signed.json
```

Live authorization record: see `harness/auth/liveauth.py` (task id + `TaskSpec.content_hash()`, accounts, owned /
read / preauthorized destinations, budgets, time window, Ed25519 signature; the service holds only the public key,
installed with `launch(..., config={"operator_pubkey": P})`).

`resolve_approval`, `reset_stop`, `resume`, `revoke`, `reload_policy`, `register_task`,
`register_live_authorization`, `put_secret` are accepted only from root. A dashboard running as the learner uid can
*list* approvals and *engage* the stop; approving and resetting go through a root-owned process (`opctl`).

## Emergency stop

`policy.emergency_stop(reason, by)` or `touch R/auth/STOP`. Checked first by every call; voids all grants; survives
restarts; the launcher terminates every service except the authorization service and restarts nothing until
`reset-stop` + `handle.resume()`.

## Tool names

The policy's tool table (`default_policy.yaml`, `tools:`) maps operation names to side-effect classes; an
operation that matches no entry is **denied**. It currently knows: `wait`, `inspect.*`, `internal.*`, `lab.*`
(diagnostic), `browser.*`, `visual.*`, `sandbox.*`, `code.*`, `job.*`, `file.*`, `net.request`, `http.*`,
`api.*`, `publish.*`, `discord.*`, `*.dm`, `payment.*`, `skill.*`. New tools need a line there (data, not code).
