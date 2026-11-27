# Change requests — workstream EA

Each is worked around inside `harness/auth/`; none blocks EA.

## 1. `contracts.ExecContext` does not carry the action's authorization

**Why.** A modality executor gets `(action, ctx)`. To obtain a secret it must present the grant of the action it is
executing, but only `EnvironmentAdapter.execute(action, authorization)` has it.
**Proposed diff (`src/cam_live/contracts.py`, `ExecContext`):**
```diff
     cancel: Any = None               # threading.Event-like; executors poll it
     secrets: Any = None              # SecretBroker handle (executor side only; never the learner)
+    authorization: Optional["AuthorizationDecision"] = None   # the consumed grant of the action being executed
```
**Workaround.** `ExecutorSecrets.bind(action, decision)` returns a handle with `get(name, destination)`; the
environment sets `ctx.secrets = policy.secrets_handle().bind(action, authorization)`.

## 2. `contracts.PolicyEngine.check_grant` — say that it consumes

**Why.** Single use needs one defined point of consumption. `PolicyService.check_grant(action, decision,
consume=True)` consumes; a second call returns `False`. A runtime that also calls it before `execute` would burn
the grant.
**Proposed diff (docstring only):**
```diff
-    def check_grant(self, action: ActionSpec, decision: AuthorizationDecision) -> bool: ...
+    def check_grant(self, action: ActionSpec, decision: AuthorizationDecision) -> bool: ...
+    # called exactly once, by the executor, immediately before execution: verifies binding / expiry and USES UP
+    # the grant (implementations may offer consume=False to only look)
```

## 3. `harness/__init__.py` — the executor has no declared uid

**Why.** `SERVICE_MODULES` has `"exec"` but `SERVICE_USERS` has no role for it; the authorization service must know
which uid may call `check_primitive` / `record` / `secret_get`.
**Proposed diff:**
```diff
+#: service -> role (key of SERVICE_USERS) whose uid runs it
+SERVICE_ROLES: Dict[str, str] = {"auth": "auth", "exec": "browser", "verifier": "verifier", "learner": "learner"}
```
**Workaround.** `launcher.DEFAULT_ROLES` (same mapping, overridable via `LauncherConfig.roles`) and the service's
`--executor-role` / `--executor-uid` options. If the executor must start as root to drop to `camsbx` for sandbox
jobs, set `roles={"exec": "root"}`: root is the operator for the service, so that weakens nothing in the matrix but
removes the uid separation of the executor itself.

## 4. `configs/authorization.yaml` does not exist

**Why.** The architecture names it as the policy file; it is master-owned.
**Proposed.** Copy `src/cam_live/harness/auth/default_policy.yaml` to `configs/authorization.yaml` and pass it as
`LauncherConfig.policy_path` / `--policy`. Until then the packaged default is used.

## 5. `pyproject.toml` — declare `cryptography`

**Why.** Live authorization records are Ed25519-signed (`harness/auth/liveauth.py`). `cryptography` 50.0.2 is
already installed in the venv (nothing was installed by EA) but is not a declared dependency.
```diff
-                "fastapi>=0.115", "uvicorn>=0.30", "playwright>=1.48", "pillow>=10"]
+                "fastapi>=0.115", "uvicorn>=0.30", "playwright>=1.48", "pillow>=10", "cryptography>=42"]
```

## 6. `tests/conftest.py` — `privileged` / `browser` / `network` are not skipped by default

**Why.** Only `gpu` and `slow` are auto-skipped; the ground rules say all five run only with `-m`.
```diff
-        for mark in ("gpu", "slow"):
+        for mark in ("gpu", "slow", "privileged", "browser", "network"):
```
**Workaround.** `tests/harness/auth/conftest.py` skips `privileged` unless `-m` names it (and unless root).

## 7. `harness/rpc.py` — client cannot verify who is listening

**Why.** A client should refuse a socket served by an unexpected uid (e.g. one planted before the service started).
**Proposed.** `RpcClient(path, timeout, server_uids: Optional[Set[int]] = None)` checking `SO_PEERCRED` after
`connect`. **Workaround.** `harness/auth/client._VerifiedRpcClient` subclasses `RpcClient._connect`.

## 8. Integration requirement on executors (EB / EC / ES) — not a diff

Per-destination network scoping exists only on the policy path in this container (docs/EA/SECURITY.md §2). It holds
only if executors report each outgoing request as `PrimitiveOp("net.request", modality, {"method", "url"})` through
`ctx.check`, put the measured `page_url` into interaction primitives, and call `record`. Tool names must appear in
the policy's tool table (unknown operations are denied).
