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

## 9. ES — `tools/discord_live_check.py --staging --live-policy`: register the staging services (since the local-address rule)

**Why.** The live mode of the policy now refuses every address of this machine that is not a registered local
service (G-sec M4 / G-env C1). The check runs the live policy against the in-process staging Discord server on
ephemeral loopback ports and never registers them, so the bot secret is not released for them and
`tests/harness/skills/test_es_discord_live_check.py::test_the_policys_live_mode_with_a_signed_record_and_a_narrower_scope`
fails (`blocked: no Discord bot token provided`). The staging-mode run is unaffected (staging counts in-process
listeners as registered).
**Proposed diff (`tools/discord_live_check.py`, in the staging set-up):**
```diff
     policy, env, discord = build(
         run_dir, task, policy_path=policy_path, base_url=server.base_url, provenance="staging", fsync=False,
-        prepare=lambda p: p.put_secret("discord_bot_token", token, [rest, gateway], ["discord"], by="staging check"))
+        prepare=lambda p: (p.register_local_service(rest, "staging check"),
+                           p.register_local_service(gateway.split("/", 1)[0], "staging check"),
+                           p.put_secret("discord_bot_token", token, [rest, gateway], ["discord"], by="staging check")))
```
**Evidence.** With exactly these two registrations patched in at run time, the test's scenario ends with exit code
0 and `{"ok": 23, "scope_refusal": 2, "skipped": 1}`, which is what the test asserts.

## 10. EB — report the connected address; register staging sites

**Why.** (a) The policy never resolves DNS. A public host name whose DNS record points at this machine (DNS
rebinding) is refused only if the executor reports the address it actually connected to. `check_primitive` now
applies the local-address rule to `op.arguments["peer_ip"]` (also `resolved_ip` / `remote_ip`).
(b) `harness/service.py --staging-sites` starts sites on ephemeral loopback ports in the environment service's
process; in the separated deployment the authorization service does not own those sockets, so the sites are
refused until the operator registers them.
**Proposed.** (a) `browser.py` (`_on_route`: Playwright's `response.server_addr()` / `request` timing gives the
peer) and `http_api.py` (httpx `response.extensions["network_stream"].get_extra_info("server_addr")`) put
`peer_ip` into a follow-up `browser.navigate` / `http.request` primitive after connecting, and abort if it is
refused. (b) Keep printing the site addresses in the ready line; the operator (or the launcher, given the ready
line) runs `opctl --run-dir R local-service add <host:port>` for each. No EA-side workaround beyond (a)'s hook.

## 11. ES — `harness/skills.py`: the `note="skill:..."` of a step pre-check is now part of the contract

**What.** The policy treats a primitive whose `note` starts with `skill:` as a step declared at run time: it is
charged like a declared step (one external action, one hit of each rate rule) and prepays the one realisation that
the step's own executor reports next; an approval given for it covers that realisation. Without the prefix every
skill step is charged twice (pre-check + realisation). Please keep the prefix (`skills.py:1212-1213`). Not a diff.

## 12. ES-3 (`publish.update` as a draft operation) — not applied

**Why.** G-env M11 showed that with ES-3 applied, the text of a draft can be replaced while its publication
approval is pending and the replaced text is published under it (`publish.submit {draft_id, destination}` does
not carry the content). The policy keeps `publish.update` an external write until the publication approval binds
the draft content (e.g. `publish.submit` carrying `content_sha256`, which the policy would then put in the
approval's preview and bind). Owner of that change: ES (`publishing.py`).

## 13. Master — `docs/HARNESS_ADAPTER_GUIDE.md` §2: accounting and coverage rules for executors

**What.** Add to the executor rules: (1) every request that leaves the process is reported through `ctx.check`
(it is charged there: external actions, write rates, money); (2) an approval covers only a primitive of the same
tool class, HTTP method and exact destination as the declared action (or the platform `transport` request of a
logical operation), so report the real method and full URL; (3) a sub-step announced before dispatch carries
`note="skill:<id>:..."`; (4) report `peer_ip` when known; (5) staging services must be registered
(`register_local_service`) unless they run in the authorization service's own process. Text in docs/EA/README.md.
