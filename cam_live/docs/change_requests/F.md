# Change requests from workstream F (runtime)

Each has a local workaround in `runtime/`; none blocks F's tests.

## F-1 AgentAPI: experiment and obstacle hooks  (contracts.py)
**What/why.** Spec 14.2 needs experiment records registered before outcomes; the obstacle section needs the
learner's explanations. `AgentAPI` has no way to hand either to the runtime, so `experiments.jsonl` and
`ObstacleRecord.explanations` depend on duck-typed optional methods.
**Proposed diff** (in `class AgentAPI`):
```python
    def drain_experiments(self) -> List[ExperimentRecord]: ...          # new / updated records since the last call
    def explain_obstacle(self, obstacle: ObstacleRecord) -> List[Dict[str, Any]]: ...   # [{"hypothesis", "p"}]
```
**Workaround.** `LiveRuntime` calls them when present (`getattr`).

## F-2 Run-directory collision: `events.jsonl` and the executor's state directory  (harness/env.py, auth/launcher.py)
**What/why.** `ComputerEnvironment` writes its execution log to `<run_dir>/events.jsonl` by default, which is
`contracts.RunFiles.EVENTS`, the runtime's event log (spec 14.4). In separated mode the launcher passes
`--run-dir R` to the exec service, whose uid cannot write in `R/` (it failed with
`PermissionError: R/workspace` when started through `cam-live run`).
**Proposed diff.** `env.py`: `self._events_path = events_log or os.path.join(self._dir, "execution.jsonl")`;
`launcher._argv`: for the exec service use `--run-dir <R>/exec` (the directory its layout already creates).
**Workaround.** `cli._supervise` appends `--run-dir <R>/exec` to the exec service arguments; `loop_service`
builds in-process environments in `<R>/exec`; `RunLog` moves its JSONL files to `<R>/learner/` (pointer file
`runtime_logs.json`) if a log name in `R/` already belongs to another writer; readers skip foreign lines.

## F-3 PolicyEngine.record: who calls it  (contracts.py comment)
**What/why.** The authorization service accepts `record` from the executor uid only, and the computer environment
calls it itself. A runtime-side call is refused in separated mode and would double-count in-process.
**Proposed diff.** Comment on `PolicyEngine.record`: "called by the environment/executor after execution; the
runtime does not call it".
**Workaround.** `RuntimeConfig.record_at_policy=False` by default.

## F-4 At-most-once capability  (contracts.py, diagnostics/world.py)
**What/why.** Crash recovery of an in-flight action needs the environment's de-duplication by action id. The
computer environment declares `get_capabilities()["features"]["at_most_once_by_action_id"]`; the contract does not
name the key and LabWorld has no de-duplication.
**Proposed diff.** Document the key next to `EnvironmentAdapter.get_capabilities`; optionally give `LabWorld.execute`
a result cache by `action_id` and `"features": {"at_most_once_by_action_id": True}`.
**Workaround.** The runtime accepts `features.at_most_once_by_action_id` / `idempotent_actions`; without it an
in-flight action is reported `cancelled/unknown` and never re-issued. The kill tests wrap LabWorld in a
de-duplicating host (`tests/runtime/rt_helpers.LabHost`).

## F-5 Verified measurements carry no signature  (contracts.py)
**What/why.** `VerifiedOutcome` is signed; `OutcomeObservation(verified=True)` from `OutcomeVerifier.measure` is not.
The runtime trusts the channel: only observations returned by `verifier.measure` keep `verified=True`, anything
from `env.measure` is forced to `False`.
**Proposed diff.** `OutcomeObservation.signature: str = ""` and `OutcomeVerifier.check_measurement(obs) -> bool`.
**Workaround.** Channel-based trust as described.

## F-6 Restore semantics of AgentAPI.load  (contracts.py docstring)
**What/why.** After a crash the runtime loads the last committed checkpoint and then (a) calls
`record_result(decision, result)` for decisions journalled after it, (b) re-delivers events after the committed
cursor and re-measures outcomes that were pending at the commit. The integrated agent must accept this.
**Proposed diff.** Docstring on `AgentAPI.load` / `record_result` stating it.
**Workaround.** None needed in F; documented in `docs/F/API.md`.

## F-7 inspect-harness / in-process runs need the executor factory list  (configs/cam_live.yaml)
**What/why.** `harness.service.build_environment` registers only the internal and browser executors unless given
`module:factory` executors. `cam-live inspect-harness` and in-process runs therefore list only those tools.
**Proposed diff.** `configs/cam_live.yaml`: `runtime: {executors: ["<module>:<factory>", ...]}` for the visual, code,
api, skill and Discord executors (and the same list in `launcher.extra_args.exec` as `--executor` flags).
**Workaround.** F reads `runtime.executors` if present.

---
## Requests filed with the G-rt fixes (2026-11-27; details in `docs/F/FIXES.md`)

## F-8 Probe checks that encode the old design  (docs/verification/repro/rt, owner G)
**What/why.** After the fixes four checks report BROKEN although the property they protect holds or was changed on
request: `p05_tasks.py` 1a and `p09_preflight.py` 1b hardcode 17 REQUIRED items (16 now: `success.evidence_source` of
`tasks/discord_100.yaml` is the Discord membership source by default, as requested); `p08_dashboard.py` check 6 flags any
`src=` (the no-inline-script CSP requires the same-origin `<script src="/app.js" defer>`), check 8 approves with the
viewing token only (refused by design: operator credential + CSRF are required); `p11_separated.py` check 14 expects
`cam-live resume` to start a supervisor for an emergency-stopped run (it now refuses a stopped run; check 13 holds).
`p10_cli.py` 4a times out (240 s) now that `cam_live.validation` exists.
**Proposed diff.**
```python
# p05 1a / p09 1b
R.check("1a ... equals the status file's count", len(unset) == json.load(open(F_JSON))["measured"]["discord_task_required_items_unset"], len(unset))
# p08 6: no external resource
R.check("6 ...", "http://" not in page.text and "https://" not in page.text and all(s.startswith("/") for s in re.findall(r'src="([^"]*)"', page.text)))
# p08 8: create_app(run_dir, TOKEN, policy=pol, operator_token=OP); csrf = page.cookies["cam_live_csrf"]
r = c.post("/api/approvals/apr_g1/approve", headers={**H, "X-CAM-Operator": OP, "X-CSRF-Token": csrf, "Origin": "http://testserver"}, json={"note": "ok"})
# p11 14: pause, SIGKILL supervisor + services, then `cam-live resume` -> supervisor.pid is the new, live supervisor
# p10 4a: timeout=1800 for `validate --mode staging`
```
**Workaround.** None needed in F; the results are explained in `docs/F/FIXES.md`.

## F-9 Machine-readable reasons in `limits_state()["tasks"][id]["unavailable"]`  (harness/auth/policy.py, owner EA)
**What/why.** The stop condition `authorization_revoked` (D13) has to recognise a revocation from the reason text
("... revoked ..."). The policy has codes (`P-REVOKED`, `P-LIVE-AUTH`) but `limits_state` drops them.
**Proposed diff.** `tasks[row["task_id"]] = {"unavailable": [r for _p, r in err], "unavailable_codes": [p for p, _r in err]}`.
**Workaround.** `LiveRuntime._authorization_revoked` matches `revok` in the reasons (and will prefer
`unavailable_codes` containing `P-REVOKED` once present).

## F-10 Discord executor capabilities  (harness/discord.py, owner ES)
**What/why.** The preflight cross-checks every component the environment reports (D12); `DiscordExecutor` has no
`capabilities()`, so its target (live discord.com or the local staging service) is only seen through the read-only
identity check's observation data.
**Proposed diff.** `def capabilities(self): return {"environment": self.environment, "base_url": self.base_url,
"guild_id": self.guild_id, "facts": dict(self.facts)}` (and include it in the api executor's capabilities).
**Workaround.** `preflight.discord_items` executes `discord.get_identity` / `discord.get_permissions` through the
environment under the authorization service and reads `data.environment`.

## F-11 What the integrated agent receives and relies on  (agent.py / contracts.py, owner M)
**What/why.** (a) `agent.set_objective` now receives `runtime.tasks.learner_view(task)`: `success` reduced to
`{metric, target}`, `measurement` empty, no credential names or test accounts (D6). `CamLiveAgent` must not need the
rest. (b) The runtime counts a verifier measurement as verified only when `verifier.check_measurement(obs)` is true;
`runtime.verifier.SignedVerifier` now signs measurements and the verifier service / `RemoteVerifier` expose
`check_measurement` (M-int-3 done). This relies on `contracts.OutcomeObservation.signature` (currently an uncommitted
change in the working tree) being committed. (c) After a restart `reconcile` may receive outcomes that were
journalled before the crash (re-delivery; F-6 applies).
**Proposed diff.** None for (a)/(c) if the agent already tolerates it; commit the `signature` field for (b).
**Workaround.** `getattr` for `check_measurement`: a verifier without it yields unverified measurements.
