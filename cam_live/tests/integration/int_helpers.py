"""Builders shared by the integration tests of the integrated agent (``cam_live.agent.CamLiveAgent``).

Everything here is real code of the project: LabWorld (``diagnostics.world``) with its signing verifier, the
authorization service (``harness.auth.PolicyService``), the persistent runtime (``runtime.loop.LiveRuntime``) and
the agent on the tiny CPU configuration (tiny random backbone + debug core). LabWorld results are diagnostic.
"""
from __future__ import annotations

import collections
import dataclasses
import os
from typing import Any, Dict, List, Optional, Sequence

from cam_live.agent import CamLiveAgent
from cam_live.config import CamLiveConfig
from cam_live.contracts import SimClock, TaskSpec
from cam_live.diagnostics.world import DAY, LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.harness.auth import PolicyService
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig

AGENT_METHODS = ("set_objective", "ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select",
                 "record_result", "internal_rewards", "learn", "meta_learn", "save", "load", "next_wake", "status",
                 "drain_experiments", "explain_obstacle")


def agent_config(seed: int = 0, **agent: Any) -> CamLiveConfig:
    """Tiny backbone + debug core on CPU, with schedules short enough for a test-sized lifetime."""
    cfg = CamLiveConfig.debug(seed=seed)
    cfg.backbone.tiny_vocab = 4096                 # distinct digits in rendered actions (B-CR-9 is per config)
    cfg.optim.meta_min_independent_outcomes = 3     # independent verified outcome groups per meta-update
    cfg.optim.adapter_every = 3
    cfg.optim.batch_size = 8
    a = dict(proposer_max_new_tokens=16, durable=False, policy_update_every=3, policy_update_batch=12,
             experiment_rate=0.15, planner_measure_every=4)
    a.update(agent)
    cfg.sections["agent"] = a
    cfg.sections["credit"] = {"learner": {"min_independent_outcomes": 2, "warmup_steps": 0, "slow_batch_size": 3,
                                          "grad_last": 2}}
    return cfg


def lab_config(seed: int, **kw: Any) -> LabConfig:
    base = dict(seed=seed, n_channels=4, target=12, budget=40.0, horizon_s=2 * DAY, outcome_horizon_s=6 * 3600.0,
                mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True, family="lab/integration")
    base.update(kw)
    return LabConfig(**base)


class Recorder:
    """Forwards every call to the agent and records its name, in order (the runtime only ever sees this proxy)."""

    def __init__(self, agent: CamLiveAgent):
        self._agent = agent
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._agent, name)
        if name in AGENT_METHODS and callable(target):
            def call(*a: Any, **k: Any) -> Any:
                self.calls.append(name)
                return target(*a, **k)
            return call
        return target


@dataclasses.dataclass
class Lifetime:
    task: TaskSpec
    world: LabWorld
    verifier: LabVerifier
    policy: PolicyService
    runtime: LiveRuntime
    run_dir: str
    status: str = ""
    cycles: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    report: Dict[str, Any] = dataclasses.field(default_factory=dict)


def run_lifetime(rec: Recorder, clock: SimClock, root: str, seed: int, *, max_steps: int = 80,
                 max_decisions: int = 0, lab_kw: Optional[Dict[str, Any]] = None,
                 rt_cfg: Optional[RuntimeConfig] = None) -> Lifetime:
    """One objective in a fresh LabWorld of the family, driven by the real runtime until it ends."""
    run_dir = os.path.join(root, f"lifetime_{seed}")
    world = LabWorld(lab_config(seed, **(lab_kw or {})), clock)
    task = lab_task(world, f"obj_int_{seed}")
    if max_decisions:
        task.limits.max_decisions = int(max_decisions)
    verifier = LabVerifier(world)
    rec._agent.attach_verifier(verifier)
    policy = PolicyService(os.path.join(run_dir, "auth_state"), clock=clock)
    policy.register_task(task, "operator")
    rt = LiveRuntime(rec, world, policy, verifier, task, run_dir, clock,
                     rt_cfg or RuntimeConfig(fsync=False, candidates_k=8, keep_checkpoints=2))
    lt = Lifetime(task, world, verifier, policy, rt, run_dir)
    for _ in range(max_steps):
        lt.cycles.append(rt.step())
        if rt.status != "running":
            break
    lt.status = rt.status
    lt.report = rt.status_report()                      # the runtime's status report (asks the agent's status)
    rt.close()
    return lt


def count(xs: Sequence[str]) -> Dict[str, int]:
    return dict(collections.Counter(xs))


# --------------------------------------------------------------------------------------------------------------
# a plain driver of the agent API (for the save / fresh-process / load comparison: no runtime state involved)
# --------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class DriverState:
    cursor: str = ""
    pending: List[Any] = dataclasses.field(default_factory=list)       # OutcomeQuery issued by the world


def drive(agent: CamLiveAgent, world: LabWorld, clock: SimClock, verifier: LabVerifier, state: DriverState,
          n: int) -> List[Dict[str, Any]]:
    """``n`` decisions through the AgentAPI in the order of the runtime loop. Returns what was decided."""
    from cam_live.diagnostics.world import AllowAllPolicy
    policy, out = AllowAllPolicy(), []
    for _ in range(n):
        events, state.cursor = world.poll_events(state.cursor)
        agent.ingest(list(world.observe()) + list(events))
        now = clock.now()
        due = [q for q in state.pending if q.due_at <= now]
        outs = [o for o in verifier.measure(due) if o.status == "observed"]
        done = {o.query_id for o in outs}
        state.pending = [q for q in state.pending if q.query_id not in done]
        if outs:
            agent.reconcile(outs, None)
        agent.update_beliefs()
        cands = agent.propose(world.discover_tools(), 8)
        evals = agent.evaluate([c for c in cands if not world.validate_action(c)])
        d = agent.select(evals)
        if d.chosen.operation == "wait":
            from cam_live.contracts import ActionResult, Cost
            t0 = clock.now()
            clock.advance(float(d.chosen.arguments["seconds"]))
            res = ActionResult(action_id=d.chosen.action_id, status="ok", started=t0, ended=clock.now(),
                               cost=Cost(seconds=clock.now() - t0), verified=True)
        else:
            res = world.execute(d.chosen, policy.authorize(d.chosen, agent.task, {}))
        state.pending += list(res.expected_outcomes)
        agent.record_result(d, res)
        agent.internal_rewards()
        ups = agent.learn(1e9)              # no wall-clock budget: what is learned must not depend on machine load
        metas = agent.meta_learn(1e9)
        out.append({"operation": d.chosen.operation, "arguments": d.chosen.arguments, "rule": d.exploration,
                    "probabilities": list(d.probabilities), "n_updates": len(ups),
                    "meta_applied": [bool(m.applied) for m in metas], "reward_version": agent.meta.reward.version,
                    "model_version": agent.learner.model_version})
    return out


# --------------------------------------------------------------------------------------------------------------
# a LabWorld that outlives the runtime process (served over the project's RPC) — the kill -9 test
# --------------------------------------------------------------------------------------------------------------
class LabHost:
    """World, verifier and clock in the test process. LabWorld runs each action id at most once
    (``LabConfig.at_most_once``); the host counts how often the runtime asked and how often the world acted."""
    METHODS = ("observe", "discover_tools", "validate_action", "execute", "poll_events", "get_resource_state",
               "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure", "v_verify",
               "v_check_signature", "v_check_measurement", "v_measure", "clock_now", "clock_advance")

    def __init__(self, cfg: LabConfig):
        self.clock = SimClock()
        self.world = LabWorld(cfg, self.clock)
        self.verifier = LabVerifier(self.world)
        self.execute_calls: Dict[str, int] = {}
        self.real_executions: Dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        if name in ("observe", "discover_tools", "validate_action", "poll_events", "get_resource_state",
                    "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure"):
            return getattr(self.world, name)
        raise AttributeError(name)

    def execute(self, action: Any, authorization: Any) -> Any:
        aid = action.action_id
        self.execute_calls[aid] = self.execute_calls.get(aid, 0) + 1
        before = self.world._n_exec
        res = self.world.execute(action, authorization)
        if self.world._n_exec > before:
            self.real_executions[aid] = self.real_executions.get(aid, 0) + 1
        return res

    def v_verify(self, task): return self.verifier.verify(task)
    def v_check_signature(self, outcome): return self.verifier.check_signature(outcome)
    def v_check_measurement(self, obs): return self.verifier.check_measurement(obs)
    def v_measure(self, queries): return self.verifier.measure(queries)
    def clock_now(self): return self.clock.now()
    def clock_advance(self, dt): return self.clock.advance(dt)
