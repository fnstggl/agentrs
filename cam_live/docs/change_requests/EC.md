# Change requests from workstream EC

None of these blocks EC; each is worked around inside EC's own files as described.

## EC-CR-1 `tests/conftest.py`: opt-in markers other than gpu / slow

**What.** `pytest_collection_modifyitems` skips only `gpu` and `slow` unless named with `-m`. `browser`,
`privileged` and `network` are declared in `pyproject.toml` but run by default, so each workstream decides for
itself (EC skips them in `tests/harness/compute/conftest.py`; a `privileged` test creates system users).

**Proposed diff.**
```diff
-        for mark in ("gpu", "slow"):
+        for mark in ("gpu", "slow", "browser", "privileged", "network"):
             if mark in item.keywords and mark not in expr:
                 item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
```

## EC-CR-2 `contracts.py`: say what an executor may call on `ExecContext.secrets`

**What / why.** Architecture §1 says every observation is scrubbed of known secret values, but the contract gives
`secrets: Any`. EC calls `ctx.secrets.scrub(text) -> str` when that attribute exists (sandbox output, file
contents, job output) and never asks the broker for a secret.

**Proposed diff.**
```diff
+@runtime_checkable
+class SecretBroker(Protocol):
+    """Executor-side handle. ``get`` releases one secret for one authorized primitive; ``scrub`` removes every
+    known secret value from text that is about to become an observation."""
+    def get(self, name: str, op: "PrimitiveOp", action: "ActionSpec") -> str: ...
+    def scrub(self, text: str) -> str: ...
 ...
-    secrets: Any = None              # SecretBroker handle (executor side only; never the learner)
+    secrets: Optional[SecretBroker] = None   # executor side only; never the learner
```

## EC-CR-3 `contracts.py`: `VerificationCondition.spec` keys and `VisualGrounder` optional members

**What / why.** `VerificationCondition.kind` is a shared vocabulary but `spec` is free-form; candidate generators
(B) and executors must agree. EC evaluates these and returns `verified=None` (never `True`) for anything else:

| modality | kind | spec |
|---|---|---|
| visual | `element_state` | `{"color_at": [x, y], "rgb": [r, g, b], "tolerance": 60}` or `{"changed": bool, "region": [x, y, w, h], "min_fraction": f}` |
| visual | `url_matches` | `{"pattern": regex}` (only when the screen provider has `url()`) |
| code | `text_present` | `{"text": str, "stream": "stdout" \| "stderr"}` |
| code | `program` | `{"command": shell text, "expect_exit": 0}` (run in the sandbox after the action) |

`VisualGrounder`: EC reads the optional attribute `heavy: bool` (needs the shared GPU lock) and calls the optional
method `close()`. Proposed diff: add both to the protocol docstring (attributes stay optional).

## EC-CR-4 environment: `torchvision` is not installed in `/workspace/camlive/.venv`

**What / why.** `transformers`' packaged processors for Qwen3-VL and SmolVLM import a video processor that needs
`torchvision` (and `num2words` for SmolVLM). Installing `torchvision` would pull a second `torch` into the venv
that shadows the shared one, so EC did not install it and builds model inputs with the PIL image processors
instead (`grounding/maiui.py`, `grounding/vocaela.py`). If the master adds a matching `torchvision` build, nothing
in EC needs to change; the hand-built path keeps working.

## EC-CR-5 primitive names EC sends to `ExecContext.check` (for EA's policy tables)

Not a contract change; listed so the authorization service can recognise them:
`sandbox.python`, `sandbox.shell`, `sandbox.write_file`, `sandbox.read_file`, `sandbox.list_files`,
`sandbox.job_start`, `sandbox.job_status`, `sandbox.job_result`, `sandbox.job_cancel`,
`net.connect` (arguments `host`, `port`, `ip` — empty unless the job gave an address literal —, `private`, `url`
= `tcp://host:port`, `from` = the sandbox operation; issued once per outbound TCP connection of a scoped-network
job, **before** the name is resolved), `visual.screenshot`, `visual.click`, `visual.double_click`, `visual.drag`,
`visual.scroll`, `visual.type` (arguments carry `chars` and `sha256`, never the text), `visual.key`.
A grounded click arrives as `visual.click` with extra arguments `target`, `grounder`, `confidence`.
