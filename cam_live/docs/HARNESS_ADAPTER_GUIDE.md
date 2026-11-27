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
* **Verification is observed.** `ActionSpec.verify` is checked against the resulting state; `ActionResult.verified`
  is `True`, `False` or `None` (could not be checked) — never assumed.
* **At most once.** Submitting the same `action_id` again returns the recorded result.
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
5. Secrets: never accept a secret value in arguments. Take a secret *name* and fetch the value at the moment of
   use from `ctx.secrets` (`get(name, action, decision, destination)` with `decision = ctx.authorization`). Do not
   put the value in ops, observations or errors; the environment additionally scrubs everything it returns or logs
   with `ctx.secrets.scrub`.
6. Report measurements that will mature later in `ActionResult.expected_outcomes`.

`ExecContext` extras set by `ComputerEnvironment` (see `docs/change_requests/EB.md`): `ctx.dispatch(step)` runs a
sub-action through the environment (same checks, same grant) — this is how a skill executor runs its primitives;
`ctx.workspace` is the persistent workspace directory; `ctx.authorization` is the action's grant.

Optional hooks the environment calls when present: `bind(env)`, `observe()`, `capabilities()`, `metrics()`,
`pending()`, `measure(query, now)`, `poll()` (events to append to the stream), `next_wake()`,
`verify(cond, action, ctx)` (evaluate a verification condition natively), `snapshot_state()` /
`restore_state(d)` (restart recovery; the snapshot is stored in `<run_dir>/env/state.json`),
`handles_steps = True` (the executor expands `ActionSpec.steps` itself).

What you get for free from `ComputerEnvironment`: grant verification, primitive checks bound to the granted
action, at-most-once execution and crash-retry rules, serialisation of `external_write` actions, staleness
pre-checks, programs (`operation="program"` with `steps`), `verify`, the event stream, pending outcomes, resource
accounting, the execution log (`<run_dir>/events.jsonl`) and restart recovery.

To load an executor in the service process: `python -m cam_live.harness.service ... --executor my_pkg.mod:factory`
(the factory is called with those of `run_dir, clock, env, policy, secrets` it accepts).

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

Separated deployment: when the `cambrowser` user exists and the service runs as root, Chromium is started under
that uid through a `setpriv` wrapper with a profile directory only it can read. Chromium must be executable by
that uid: Playwright's download under `/root` is not, so install or copy it to a shared path and set
`executable_path` (the capability `executors.structured_browser.uid_separation` says whether separation is active
and, if not, why).

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
* A redirect hop is checked after it happened (the tab is sent to `about:blank` when the final URL is refused);
  typed, clicked and scripted navigations are checked before the request leaves.
* Per-destination enforcement lives in the browser's request hook and the authorization service, not in the
  kernel (no network namespaces in this container); a VM is the recommended boundary for live operation.
* With several writers on `events.jsonl` each append is `flock`ed; the file must be writable by every writer's uid.
