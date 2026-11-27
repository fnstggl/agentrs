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

## Requests from E

Filed by the E-fix workstream (computer environment); each is worked around on the executor side.

**E→EA-1. Classify the browser network layer's `CONNECT` as a read-level tunnel.** The browser's network guard
(`harness/browser.py:_NetGuard`) reports a tunnel it was not told about by the page layer (an HTTPS or WebSocket
redirect hop to a new host, a worker's connection) as `PrimitiveOp("browser.request", {"url": "https://host:port/",
"method": "CONNECT", "resource_type": "tunnel", "via": "network"})`. `T-BROWSER-REQUEST` is `effect: by_method`, and
`evaluate.py:361-364` turns an unknown method into `WRITE` / `external_write`, so each such tunnel is charged as an
external write. Proposed (`default_policy.yaml` / `configs/authorization.yaml`, before `T-BROWSER-REQUEST`):
```diff
+  - id: T-BROWSER-TUNNEL            # an HTTPS / WebSocket tunnel the browser opens; what it carries is judged by the
+    ops: [browser.request]          # page layer per request (path and method), the tunnel only by host and port
+    modalities: [structured_browser, visual]
+    when: {arguments.method: CONNECT}
+    effect: external_read
+    counts_as_action: false
+    dests: [{kind: url, from: [arguments.url], required: true}]
```
(or `_READ_METHODS += ("CONNECT",)` for `by_method` tools). Workaround: page-initiated HTTPS requests hand a
token to the network layer, so only tunnels the page layer did not see are checked; in staging (plain HTTP on
loopback) none occur.

**E→EA-2. `T-PUBLISH` should accept the `file:` address of an owned static site.** `PublishingExecutor` addresses a
site without `base_url` as `file:<absolute path>`; `T-PUBLISH` reads `arguments.destination` as `kind: url` only,
so `publish.submit` is denied "names no destination" (`P-DEST-MISSING`). Proposed: add
`{kind: file, from: [arguments.destination], when_prefix: "file:"}` (or let the url reader fall through to a file
destination for `file:`). Workaround: configure owned sites with a `base_url` (documented in docs/ES/INTEGRATION.md).

**E→EA-3. Approval previews: show the whole publication text, or say it was cut.** `policy.py:_preview` stores
`ne.content[:4000]`. `publish.submit` now carries the exact text and its `content_sha256` (G-env M11), and the
grant binds them, but an operator approving a longer text sees only its first 4000 characters without being told.
Proposed: keep up to the policy's `structure.max_argument_bytes` for `kind: publish`, or add
`"content_truncated": true, "content_chars": len(ne.content)` to the preview entry.

**E→EA-4. A names-only secret probe for the executor uid.** `DiscordExecutor` reports availability with a
`credential_probe(name) -> bool`; over AuthRPC the executor uid may not call `secret_names` (operator only), so
`harness/factories.py:credential_probe` falls back to "the operator named it in the task". Proposed: an executor
method `secret_has(name) -> bool` (no value, no binding information) — the `has(name)` that `contracts.SecretHandle`
lists as optional (M-int-5).

**E→EA-5. Scrub transformed echoes in the vault too.** The environment now scrubs case-changed, character-separated
and half echoes of every value it released during its process (`harness/env.py:EchoScrubber`). After a restart, or
for a value the environment never released, only `vault.variants` apply. Proposed: `vault.scrub` additionally
matches case-insensitively, with separators between characters, and any run of at least half of a value of 12 or
more characters (`env.echo_patterns` is a reference implementation).

**E→EA-6 (re EA #10).** (a) Done for the browser: the network guard resolves a name once, connects only to those
addresses and puts every non-global answer to `check_primitive` as `peer_ip` (test
`tests/harness/env/test_e_fix_browser.py::test_a_host_name_is_judged_by_the_address_it_resolves_to`). Not done for
`http_api.py` (httpx resolves again at connect time; pinning needs a custom transport with SNI). (b) The service's
ready line already lists the staging sites it started (`"staging": {...}`); the exec uid cannot call
`register_local_service`, so the operator / launcher registers them.

### EA's answers to the requests from E (2026-11-27, EA-fix)

* **E→EA-1 — done.** `CONNECT` is a read-level method for every `by_method` tool (`evaluate._TUNNEL_METHODS`): a
  tunnel is judged by host and port (the local-address rule and deny list included), never charged as an external
  write; `T-BROWSER-REQUEST` is in any case `counts_as_action: false`. No `when:` key was added to the schema.
  Test: `test_ea_fix_policy.py::test_a_browser_tunnel_is_a_read_level_check_not_an_external_write`.
* **E→EA-2 — not done.** A `file:` destination of `T-PUBLISH` is a file destination and must lie inside the policy's
  `file_roots`; publishing into an arbitrary owned directory needs a decision on how a task / record names owned
  file sites (ownership of files is not modelled today). Until then: owned sites with a `base_url`, as E documented.
* **E→EA-3 — done.** Preview content entries carry `content_chars` and `content_truncated`; for tools of kind
  `publish` up to 64 KiB of text is kept (4000 characters otherwise).
  Test: `test_ea_fix_policy.py::test_approval_previews_say_when_content_was_cut`.
* **E→EA-4 — done.** `PolicyService.secret_has(name) -> bool` (also over AuthRPC, allowed for the executor uid and
  the operator; no value, no bindings; audited by name). Test: `test_ea_fix_policy.py::test_secret_has_tells_presence_only`,
  RPC matrix in `test_ea_service.py` / `test_ea_privileged.py`.
* **E→EA-5 — not done in `vault.scrub`.** Actions are already refused when they carry a stored secret spaced out,
  re-cased, punctuated or in parts (`vault.find_loose`, G-sec p07); changing what `scrub` *replaces* in observations
  was not part of this round and the environment's `EchoScrubber` covers released values. Kept open.
* **E→EA-6 — noted.** `check_primitive` applies the local-address rule to `peer_ip` / `resolved_ip` / `remote_ip`
  (test `test_a_public_name_that_resolved_to_this_machine_is_refused`).
