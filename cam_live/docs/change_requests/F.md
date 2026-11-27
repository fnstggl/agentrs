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
