# Change requests from workstream EB

Each item: what, why, proposed diff, and the local workaround in place meanwhile.

## 1. `contracts.ExecContext`: three more fields

*Why.* (a) A skill executor (ES) must run its primitive steps through the environment so they get the same
authorization checks, staleness checks and logging; it has no way to reach other executors through the current
context. (b) Executors need the persistent workspace path. (c) The authorization workstream's secret handle is
`ExecutorSecrets.get(name, action, decision, destination)` — it needs the action's `AuthorizationDecision`, which
executors do not receive.

```diff
 @dataclass
 class ExecContext:
     ...
     cancel: Any = None
     secrets: Any = None              # SecretBroker handle (executor side only; never the learner)
+    dispatch: Any = None             # Callable[[ActionSpec], ActionResult]: run a sub-action through the environment
+    workspace: str = ""              # persistent workspace directory (<run_dir>/workspace)
+    authorization: Any = None        # the AuthorizationDecision of the action being executed
```

*Workaround.* `ComputerEnvironment._context` sets these three as plain attributes on the `ExecContext` instance.

## 2. `contracts`: name the secret-broker protocol

*Why.* `ExecContext.secrets` is typed `Any`. EB (scrubbing, `browser.fill` with `secret`) and ES (`http_api`) each
guessed the interface.

```diff
+@runtime_checkable
+class SecretHandle(Protocol):
+    """Executor-side handle: releases a secret only for a granted action being executed; can scrub text."""
+    def get(self, name: str, action: ActionSpec, decision: AuthorizationDecision, destination: str) -> str: ...
+    def scrub(self, text: str) -> str: ...
```

*Workaround.* `harness.browser.broker_get` inspects the handle's `get` signature (4-, 3-, 2- or 1-argument forms);
`harness.env.scrub_with` uses `scrub` / `redact` / an iterable of known values.

## 3. State who calls `PolicyEngine.record`

*Why.* The contract does not say whether the runtime or the environment reports results for audit and limit
accounting; if both do, limits are counted twice. The executor process is the tamper-proof place.

*Proposed text* (docstring of `PolicyEngine.record`): "Called exactly once per executed action by the computer
environment (`ComputerEnvironment(record_results=True)`, the default). The runtime must not call it."

*Workaround.* `ComputerEnvironment(..., record_results=False)` turns the call off if the integration decides
otherwise. Pre-execution refusals (no grant, invalid, stale, limit) are not recorded: nothing was executed.

## 4. `tests/conftest.py`: gate `browser`, `privileged`, `network` like `gpu` and `slow`

```diff
-        for mark in ("gpu", "slow"):
+        for mark in ("gpu", "slow", "browser", "privileged", "network"):
```

*Workaround.* `tests/harness/env/conftest.py` applies the same rule to its own directory (strictest mark wins, so
`-m privileged` runs tests marked `privileged` + `browser`).

## 5. `pyproject.toml`: declare the packages perception imports

```diff
-                "fastapi>=0.115", "uvicorn>=0.30", "playwright>=1.48", "pillow>=10"]
+                "fastapi>=0.115", "uvicorn>=0.30", "playwright>=1.48", "pillow>=10", "beautifulsoup4>=4.12",
+                "pypdf>=4"]
```

`beautifulsoup4` was already in the venv; `pypdf` was installed by EB
(`uv pip install --python /workspace/camlive/.venv/bin/python pypdf`). Without `pypdf`, PDF documents are still
observed (size, hash) with `data["pdf_text"] = "unavailable: ..."`.

## 6. `docs/CAM_LIVE_ARCHITECTURE.md` ownership table

Row EB lists `harness/rpc.py` (it is master-owned) and omits `harness/service.py`, `harness/internal.py` and
`docs/HARNESS_ADAPTER_GUIDE.md`. Documentation fix only.

## 7. For the authorization workstream (EA): primitive operations emitted by the browser executor

Not a master-owned file; listed here so integration can route it. Besides the tool names in
`harness.browser.TOOLS`, `check_primitive` receives:

| `op.op` | `op.arguments` | when |
|---|---|---|
| `browser.navigate` | `url`, `via` = `page` \| `redirect` \| `session_restore`, `method`, `session` | every navigation request not typed by the action itself, the final URL after a redirect, tabs reopened after a restart |
| `browser.request` | `url` (origin + `/`), `resource_type`, `method`, `session` | first cross-origin subresource request per origin (cached 30 s) |
| `browser.download` | `url`, `session` | a click-triggered download whose URL differs from the link target |

Interactions carry `page_url` and `element` (`role`, `name`, `href`, ...); uploads carry `path`, `bytes`, `sha256`.
`default_policy.yaml` currently catches `browser.request` with the generic `browser.*` rule (effect `interact`);
a dedicated read rule would be more accurate:

```yaml
  - id: T-BROWSER-REQUEST
    ops: [browser.request]
    modalities: [structured_browser, visual]
    effect: external_read
    dests: [{kind: url, from: [arguments.url], required: true}]
```
