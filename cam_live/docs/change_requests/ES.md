# Change requests from workstream ES

None of these blocks ES: each is worked around inside ES code as described.

## ES-1 — `contracts.ExecContext`: declare the fields executors already exchange (master)

**What.** `ExecContext` is extended at run time by attribute assignment: the environment sets `authorization`,
`dispatch` and `workspace` (`harness/env.py::_context`); ES sets `granted_action` while a skill or a publication
runs sub-steps. These are part of the executor contract and should be declared.

**Why.** The authorization service releases a secret only for `(granted top-level action, its decision,
destination)` (`auth.vault.ExecutorSecrets.get`). An executor that runs a *step* of a skill, of a program or of a
publication therefore needs the top-level action and its decision, not the step.

**Proposed diff** (`src/cam_live/contracts.py`, class `ExecContext`):

```diff
     cancel: Any = None               # threading.Event-like; executors poll it
     secrets: Any = None              # SecretBroker handle (executor side only; never the learner)
+    authorization: Any = None        # AuthorizationDecision of the top-level action being executed
+    granted_action: Any = None       # the top-level ActionSpec whose grant covers the operations being executed
+    dispatch: Any = None             # Callable[[ActionSpec], ActionResult]: run a sub-step through the environment
+    workspace: str = ""              # persistent workspace directory of the environment
```

and in `harness/env.py::_context` (EB), so that program steps can obtain secrets too:

```diff
         ctx.authorization = authorization                                 # type: ignore[attr-defined]
+        ctx.granted_action = top                                          # type: ignore[attr-defined]
```

**Workaround in ES.** `http_api.granted_action(ctx, action)` and `http_api.acting_under(ctx, action)` read / set the
attribute dynamically; `http_api.resolve_secret` accepts both the grant-bound and a plain `get(name)` broker.

## ES-2 — a way to ask whether a secret exists without releasing it (master / EA)

**What.** Add `has(name) -> bool` to the handle placed in `ExecContext.secrets`.

**Why.** `ToolDescriptor.available` / `unavailable_reason` for the Discord tools depends on whether the bot token
exists. The grant-bound broker cannot answer outside an executing granted action, so availability cannot be
computed from the handle.

**Proposed diff** (`src/cam_live/harness/auth/vault.py`, class `ExecutorSecrets`):

```diff
-    def __init__(self, getter: Any, scrubber: Any, obj_scrubber: Any = None):
-        self._get, self._scrub, self._scrub_obj = getter, scrubber, obj_scrubber
+    def __init__(self, getter: Any, scrubber: Any, obj_scrubber: Any = None, has: Any = None):
+        self._get, self._scrub, self._scrub_obj, self._has = getter, scrubber, obj_scrubber, has
+
+    def has(self, name: str) -> bool:
+        """Whether a non-revoked secret of this name exists. Releases nothing."""
+        return bool(self._has(name)) if self._has is not None else False
```

**Workaround in ES.** `DiscordExecutor(credential_probe=lambda name: ...)`; the integrator supplies the probe.

## ES-3 — `publish.update` is a local draft operation (EA policy data)

**What / why.** `publish.update` only writes a new revision of a local draft; the external write happens in
`publish.submit`. The default policy classifies it with `publish.*` as an external write that must name a
destination, so a top-level `publish.update {draft_id, content}` is denied ("names no url destination").

**Proposed diff** (`src/cam_live/harness/auth/default_policy.yaml`, tool `T-PUBLISH-DRAFT`):

```diff
-    ops: [publish.draft, publish.preview, "publish.list*", "publish.get*", publish.status,
+    ops: [publish.draft, publish.preview, publish.update, "publish.list*", "publish.get*", publish.status,
           publishing.draft, publishing.preview]
```

**Workaround in ES.** `publish.update` (and `publish.submit`) accept an optional `destination` argument equal to the
address shown by `publish.preview` (`where.address`); with it the policy can judge the action, at the price of
treating a draft revision as a publication.

## ES-4 — feed executed actions to the skill executor (EB / master)

**What / why.** `skill.induce` refers to earlier successful executions by action id. The skill executor has to see
them.

**Proposed diff** (`src/cam_live/harness/env.py`, end of `execute`):

```diff
         self._finish(run, action, res, ahash, side)
+        sk = self._executors.get(Modality.SKILL)
+        if sk is not None and hasattr(sk, "observe_execution"):
+            sk.observe_execution(action, res)
         return res
```

**Workaround in ES.** `SkillExecutor.observe_execution(action, result)` is public; whoever calls `env.execute` can
call it (as `docs/evidence/ES/integration_smoke.py` does).
