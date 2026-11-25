"""CAM Unified environments (workstream G): the real Crucible OpenWorld and hidden-mechanism worlds behind the
contract's `EnvAdapter` / `GoalEvaluator`, the SMDP lifetime harness, procedural world generation with structural
holdouts, operator-side goal metrics, OW-1..OW-7 task protocols, demonstrations and frozen-semantic baselines.

Two sides, never mixed:

* **agent side** (`OpenWorldAdapter`, `MechWorldAdapter`): partially observable records only (`Observation`,
  `ToolSchema`, `ExecutionResult`, `GoalSpec`). Nothing reachable from these objects exposes a world kernel,
  hidden state, world family or seed (enforced by tests).
* **operator side** (`OpenWorldOperator`, `OpenWorldEvaluator`, `MechWorldEvaluator`, protocols' calibration):
  owns the clock, the goal verifiers and hidden state; its outputs reach the learner only as scalar verified
  return increments through `harness.run_lifetime`.

Entry points: **learner runs** use `openworld.open_world_isolated(tasks, evaluator={})` (worlds, operator and
evaluator in a spawned process: the hidden-state firewall is an OS boundary; checkpoints through the returned
`RemoteOperator`) or `protocols.load("ow7").open(split, i, isolated=True)`; `openworld.open_world(tasks)` ->
(adapter, operator) with `openworld_eval.OpenWorldEvaluator(operator)` keeps everything in one process - API
discipline, **not** a security boundary (closures and `gc` reach the operator) - for scripted controls, calibration
and tests. `mechworld.mech_world(tasks)` + `MechWorldEvaluator`; `harness.run_lifetime(agent, adapter, evaluator)`
with any `agent_protocol.AgentP` (`PolicyAgent` for Crucible policies, `MixedAgent` for heterogeneous populations);
`worldgen.register_families()` / `SplitGuard` / `validity_sample`; `protocols.load("ow1")` etc. with
`calibrate()` / `open()`; `demos.record_demos`; `baselines.FrozenSemanticAgent`.

Importing this package does not import torch (the harness and adapters are record-only); `baselines` loads a
language model only when asked.
"""
from __future__ import annotations

__all__ = ["agent_protocol", "ow_convert", "openworld", "openworld_eval", "harness", "mechworld", "worldgen", "goals",
           "protocols", "demos", "baselines"]
