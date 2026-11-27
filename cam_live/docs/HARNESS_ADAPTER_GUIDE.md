# Harness adapter guide

How to give CAM Live a different browser, tool set or whole computer environment without touching the learning
system (spec 12.7). Everything the learner knows about the outside world goes through two interfaces in
`src/cam_live/contracts.py`:

| You want to replace | Implement | Register with |
|---|---|---|
| one way of acting (another browser driver, a desktop, an API family, a sandbox) | `ModalityExecutor` | `ComputerEnvironment.register(executor)` |
| the whole environment (a remote VM, a vendor "computer use" service) | `EnvironmentAdapter` | hand it to the runtime instead of `ComputerEnvironment` |
| a measurement feed (analytics, counters) | an object with `pending()`, `measure(query, now)`, `metrics()` | `ComputerEnvironment.add_source(source)` |

Nothing else needs to change: the learner discovers tools from `discover_tools()`, proposes `ActionSpec`s, and
learns from `ActionResult`s, `Observation`s and later `OutcomeObservation`s.

## 1. What the learner assumes

* **Typed records only.** Every request and response is a `contracts` record. The interface carries observations
  and consequences; it never carries a reward, score or bonus, and an adapter must not add one.
* **Environment content is untrusted data.** Every `Observation` has `provenance.trust == "untrusted"`. Text read
  from a page, file or tool is returned as text and changes nothing: not the objective, not permissions, not what
  the adapter does next.
* **Exact numbers travel as `Quantity`.** Put measured numbers in `Observation.quantities` (name, value, unit);
  `Observation.text` is a bounded digest for semantic encoding. `harness.perception.Perception` does this for
  pages, documents (HTML, text, CSV, JSON, PDF text), images, tool results, traces and events.
* **Observation ids.** An action that points at something seen earlier names the observation (`arguments.obs_id`,
  or `ref@obs_id`). If that observation is outdated the action fails with `FailureClass.STALE_OBSERVATION`.
* **Failures are classified** with `FailureClass`. `DENIED` results carry a hard class
  (`missing_permission` / `platform_restriction`); they are never retried and never routed to another modality.
  The one exception: a refusal whose every reason is a limit (the authorization service's `resource: ...` reasons,
  e.g. a rate window with "retry in N s") is `DENIED` with `resource_limit` — waiting changes it
  (`harness.failures.refusal_class`).
* **Verification is observed.** `ActionSpec.verify` is checked against the resulting state; `ActionResult.verified`
  is `True`, `False` or `None` (could not be checked) — never assumed.
* **At most once.** An `action_id` is executed at most once, also when two callers submit it concurrently, each
  with its own valid grant, and across restarts: the first caller holds a claim; the claim is written to the
  journal (`<run_dir>/env/actions.jsonl`, fsynced) before the action is executed. A second caller waits for the
  recorded result (at most the action's own timeout + 30 s, `ComputerEnvironment(duplicate_wait_s=...)`) and
  otherwise gets `timeout` "being executed" — never a second execution. A caller whose own admission fails (no or
  forged grant, invalid, stale) releases the claim without recording anything. An action interrupted by a crash is
  answered `failed/unknown` and not executed again (read-only ones are re-run).
* **Bounded observations.** One observation is at most 256 KB in its JSON form
  (`ComputerEnvironment(max_observation_bytes=...)`). Larger content is stored as a blob under
  `<run_dir>/env/blobs/`; the observation keeps a stub per omitted field and `refs["data_blob"]`, which
  `internal.inspect what=blob blob=<ref> offset max_chars` reads in slices. Quantities must be finite numbers;
  anything else is dropped and counted in `data["quantities_rejected"]`.
* **Delayed outcomes.** `ActionResult.expected_outcomes` and `get_pending_outcomes()` list measurements that
  mature later; `measure()` returns `pending` until `due_at`, then a real value, or `unobservable`. It never
  fabricates a value. Environment measurements are unverified; verified ones come from the independent verifier.

## 2. Writing a `ModalityExecutor`

```python
class MyExecutor:
    modality = Modality.API                      # one executor per modality

    def tools(self) -> list[ToolDescriptor]: ...  # name, description of what it DOES, JSON-Schema arg_schema,
                                                  # side_effects: none | local | external_read | external_write
    def validate(self, action) -> list[str]: ...  # [] when structurally valid
    def execute(self, action, ctx) -> ActionResult: ...
    def close(self) -> None: ...
```

Rules for `execute(action, ctx)`:

1. **Call `ctx.check(op, action)` before every primitive operation** that touches anything outside your process
   (request, navigation, file write, publish, spend) and stop on anything but `allow`. Build a
   `PrimitiveOp(op=<tool name>, modality=..., arguments={...})`; put the destination where the authorization
   service looks for it: `arguments.url` for a request or navigation, `arguments.page_url` for an interaction with a
   loaded page, `arguments.path` for a file. On `deny` return `status=DENIED` with a hard failure class; on
   `needs_approval` return `status=NEEDS_APPROVAL`.
2. Append one `PrimitiveOp` per executed primitive to `ActionResult.ops` (status, failure class, `obs_before`,
   `obs_after`, a short `note`).
3. Return what you saw as `Observation`s (use `Perception`). To take part in staleness detection set
   `obs.data["surface"]` (a stable name of the thing observed, e.g. `browser:default:t1`) and
   `obs.data["generation"]` (changes when the surface is replaced, e.g. a new document). The environment marks
   older observations of that surface outdated when a newer generation arrives.
4. Honour `ctx.cancel` (an event) and `ctx.deadline`. Never raise for an expected failure — return a classified
   `ActionResult`. (An exception is caught by the environment and reported as `failed/unknown`.)
5. Secrets: never accept a secret value in arguments. Take a secret *reference* `secret:<name>` (the only form the
   authorization service accepts in a credential field), strip the prefix (`harness.http_api.secret_name`) and
   fetch the value at the moment of use from `ctx.secrets` (`get(name, action, decision, destination)` with
   `decision = ctx.authorization`). Do not put the value in ops, observations or errors; the environment scrubs
   everything it returns or logs with the broker's `scrub` and, for values released during this process, also their
   transformed echoes (any letter case, characters separated by spaces or punctuation, either half).
6. A request that changes something remotely and is not idempotent (POST, PATCH) is never re-sent automatically
   after an uncertain failure (a 5xx answer, a timeout or a broken connection once it was sent), unless it carries
   an `Idempotency-Key` header; a failure to connect may be retried (`run_with_retries(..., may_retry=...)`).
7. Report measurements that will mature later in `ActionResult.expected_outcomes`.

`ExecContext` fields set by `ComputerEnvironment` (contract 1.1): `ctx.dispatch(step)` runs a sub-action through the
environment (same checks, same grant) — this is how a skill executor runs its primitives; `ctx.workspace` is the
persistent workspace directory; `ctx.authorization` is the action's grant.

Optional hooks the environment calls when present: `bind(env)`, `observe()`, `capabilities()`, `metrics()`,
`pending()`, `measure(query, now)`, `poll()` (events to append to the stream), `next_wake()`,
`verify(cond, action, ctx)` (evaluate a verification condition natively), `snapshot_state()` /
`restore_state(d)` (restart recovery; the snapshot is stored in `<run_dir>/env/state.json`),
`handles_steps = True` (the executor expands `ActionSpec.steps` itself), `observe_execution(action, result)`
(called after every executed action — the skill executor collects induction traces this way).

Every executed action also updates the environment's `ModalitySelector` (`harness.modality`); its estimates for
the realisations this environment offers (success probability, latency, compute, primitive operations, recovery,
expected cost, attempts) are readable as `exec.modality.<kind>.<field>` quantities of the metrics observation of
`observe()` and in `get_capabilities()["modality_estimates"]`; a refusal blocks every realisation of the intent.

What you get for free from `ComputerEnvironment`: grant verification, primitive checks bound to the granted
action, at-most-once execution and crash-retry rules, serialisation of `external_write` actions, staleness
pre-checks, programs (`operation="program"` with `steps`), `verify`, the event stream, pending outcomes, resource
accounting, the execution log (`<run_dir>/events.jsonl`) and restart recovery.

### Composition: `cam_live.harness.factories`

`all_executors(run_dir, clock, env, policy, secrets, task=None)` returns every executor of the harness, configured
from the task: `internal`, `browser` (structured browser), `code` (sandbox + background jobs, workspace
`<run>/sandbox`), `visual` (on the browser's screens), `api` (`ApiModality([HttpApiExecutor, PublishingExecutor,
DiscordExecutor])` — publishing destinations from `task.resources` via `PublishingExecutor.from_resources`, Discord via
`DiscordExecutor.from_task`) and `skills` (library in `<run_dir>/skills`). Each of these names is also a factory of
its own (`cam_live.harness.factories:code`, ...). An executor that cannot work in a deployment is still registered
and its tools say `available: false` with the reason. `build_environment(..., compose="all", task=task)` builds a
`ComputerEnvironment` with all of them.

The environment service (`python -m cam_live.harness.service`) composes all executors when no `--executor` is
given — this is how `cam-live run` starts it — configured from `--task` or, by default, the run's `task.yaml`
next to an `exec` run directory; it uses the authorization service's secret broker over `--auth-socket`
(observations and the log are scrubbed) unless `--secrets` / `--no-secrets` say otherwise. With `--executor
my_pkg.mod:factory` (repeatable) or `--compose minimal` it registers internal + browser + the listed factories
(called with those of `run_dir, clock, env, policy, secrets, task` they accept).

## 3. Replacing the browser

`harness/browser.py` is one `ModalityExecutor` (`structured_browser`) on Playwright Chromium. Three ways to change it:

* **Another Chromium**: `BrowserConfig(executable_path=...)`, or attach a running one over CDP:
  `BrowserConfig(cdp_endpoints={"ext": "http://host:9222"})` / `executor.attach_cdp("ext", url)` /
  `--cdp ext=http://host:9222`, then use `session="cdp:ext"` in actions.
* **Another driver or browser harness**: write an executor with the same tool names (`browser.navigate`,
  `browser.click`, ... — see `browser.TOOLS`) so stored procedures keep working, and register it instead of
  `BrowserExecutor` (`build_environment(..., browser=False, executors=["my_pkg:make_browser"])`). Keep the
  observation shape produced by `Perception.page()` (`data.elements[*].ref/role/name`, `data.diff`, `surface`,
  `generation`).
* **Screen access for visual control**: `executor.screen(session, tab, ctx, action)` returns an object with
  `screenshot() -> PNG bytes`, `size()`, `click(x, y)`, `double_click`, `drag`, `scroll(dx, dy)`, `type(text)`,
  `key(combo)`, `element_at(x, y)`. A replacement browser that should serve the visual executor needs the same.

**Network guard.** Every request the browser makes is checked against the policy before it leaves: the route
layer (Playwright `context.route` / `route_web_socket`) judges navigations, popups, downloads and writes per URL,
reads per origin and action, WebSockets per URL; and Chromium's only way out is a password-protected forward proxy
in the executor process that checks what the route layer cannot see — each redirect hop before it is sent,
worker requests, WebSocket/HTTPS tunnels (per host:port) — and forwards only what the route layer has just
authorized or the policy allows now. The proxy resolves a host name once and connects only to those addresses; an
answer that is not a global address is put to the policy as `peer_ip` (DNS rebinding is judged by the address
reached). Service workers are blocked, QUIC is off, WebRTC may not use unproxied UDP,
`WebTransport` is removed, and Chromium's own background requests (autofill server queries, network time, search
preconnect) are switched off. Externally supplied browsers (`cdp:` sessions) cannot be given the proxy: for them
only the route layer applies and a redirect hop is judged after it happened.

Separated deployment: when the `cambrowser` user exists and the service runs as root, Chromium is started under
that uid through a `setpriv` wrapper with a profile directory only it can read; when the service itself runs as
`cambrowser` (as `cam-live run` starts it) Chromium inherits that uid. Either way Chromium must be executable by that
uid; Playwright's download under `/root` is not. The operator installs a shared copy once:

```bash
cd /workspace/camlive && .venv/bin/python -m cam_live.harness.browser --install-shared   # -> <venv>/share/cam_live/ms-playwright
.venv/bin/python -m cam_live.harness.browser --find cambrowser                            # what that uid will run
```

The executor looks for Chromium in `$CAM_LIVE_CHROMIUM`, `$PLAYWRIGHT_BROWSERS_PATH`, the shared copy
(`$CAM_LIVE_BROWSERS` or `<python prefix>/share/cam_live/ms-playwright`) and Playwright's own download, in that
order, and uses the first one the browser uid can execute; `BrowserConfig.executable_path` overrides. The run
directory must also be reachable by that uid. The capability `executors.structured_browser.uid_separation` says
whether separation is active and, if not, why; `chromium_executable` names the binary.

## 4. Replacing the whole environment

Implement the ten methods of `contracts.EnvironmentAdapter` (+ `interface_version = "1.x"`) and enforce inside
your adapter what `ComputerEnvironment` enforces: verify the `AuthorizationDecision` with
`PolicyEngine.check_grant` before anything runs, call `PolicyEngine.check_primitive` for every primitive, return
`DENIED` with a hard failure class otherwise, execute an `action_id` at most once, and mark everything you return
as untrusted. To run it out of process, serve it with `harness.service.serve(adapter, socket_path)` and connect with
`RemoteEnvironment(socket_path)` (ExecRPC v1: the adapter methods plus `hello` for version negotiation, `ping`,
`cancel_action`, `close`).

## 5. Running the contract tests against your adapter

```bash
cd /workspace/camlive
# generic EnvironmentAdapter conformance kit against any adapter: factory(run_dir) -> (adapter, policy, task)
CAM_LIVE_ADAPTER_FACTORY=my_pkg.my_env:make \
  .venv/bin/python -m pytest cam_live/tests/harness/env/test_adapter_contract.py -q --rootdir=cam_live
# the built-in environment: core (no browser), then everything that launches Chromium, then uid separation
.venv/bin/python -m pytest cam_live/tests/harness/env -q --rootdir=cam_live
.venv/bin/python -m pytest cam_live/tests/harness/env -q --rootdir=cam_live -m "browser or not browser"
.venv/bin/python -m pytest cam_live/tests/harness/env -q --rootdir=cam_live -m privileged
```

The staging sites used by these tests (`python -m cam_live.harness.staging`) are local, labelled `staging` in
`get_capabilities()`, and abstract: a structured HTML application, the same task on a canvas, a dynamic page, two
unfamiliar layouts with a mid-run switch, and LabWorld behind a web UI for delayed consequences. A harness that
passes against them has been shown to work on controlled services only; live operation additionally needs the
operator's authorization and credentials.

## 6. Known limits of the built-in environment

* Elements inside iframes and closed shadow roots are not listed in the structured observation (the frame is
  reported in `data.outline`; screen-based control still reaches it).
* HTTPS and WebSocket traffic is judged by the network layer per host and port (it cannot see inside TLS); paths
  and methods of the requests a page starts are judged by the route layer, redirect hops inside a TLS connection
  to an already allowed host only by host.
* For `cdp:` sessions (an externally supplied Chromium) a redirect hop is checked after it happened (the tab is sent
  to `about:blank` when the final URL is refused).
* A navigation a page starts on its own after the action returned is checked under the finished action and is
  refused when that grant no longer covers it (fail closed).
* Per-destination enforcement lives in the browser's network guard and the authorization service, not in the
  kernel (no network namespaces in this container); a VM is the recommended boundary for live operation.
* With several writers on `events.jsonl` each append is `flock`ed; the file must be writable by every writer's uid.
