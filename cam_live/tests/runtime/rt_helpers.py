"""Shared builders for the runtime tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from cam_live.contracts import AuthorizationDecision, AuthVerdict, SimClock
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
from rt_lab_agent import LabAgent


def make_lab(run_dir: str, *, lab_cfg: Optional[LabConfig] = None, agent_kw: Optional[Dict[str, Any]] = None,
             rt_cfg: Optional[RuntimeConfig] = None, policy: Any = None, verifier: Any = None, task_edit=None,
             objective_id: str = "obj_lab") -> SimpleNamespace:
    clock = SimClock()
    world = LabWorld(lab_cfg or LabConfig(seed=3, target=5), clock)
    task = lab_task(world, objective_id)
    if task_edit is not None:
        task_edit(task)
    agent = LabAgent(clock, **(agent_kw or {}))
    policy = policy or AllowAllPolicy()
    verifier = verifier or LabVerifier(world)
    rt = LiveRuntime(agent, world, policy, verifier, task, run_dir, clock, rt_cfg or RuntimeConfig(fsync=False))
    return SimpleNamespace(rt=rt, agent=agent, world=world, policy=policy, verifier=verifier, clock=clock,
                           run_dir=run_dir, task=task)


class ApprovalPolicy(AllowAllPolicy):
    """A diagnostic PolicyEngine with an approval queue: operations listed in ``needs`` wait for the operator."""

    def __init__(self, needs=("lab.unlock",)):
        super().__init__()
        self.needs = set(needs)
        self.queue: Dict[str, Dict[str, Any]] = {}
        self.authorize_calls: List[str] = []

    def authorize(self, action, task, context=None):
        self.authorize_calls.append(action.action_id)
        if self._stopped or action.operation not in self.needs:
            return super().authorize(action, task, context)
        h = action.content_hash()
        for e in self.queue.values():
            if e["action_hash"] == h:
                if e["status"] == "approved":
                    return AuthorizationDecision(AuthVerdict.ALLOW, h, ["approved by operator"], grant="diag",
                                                 approval_id=e["approval_id"])
                if e["status"] == "rejected":
                    return AuthorizationDecision(AuthVerdict.DENY, h, ["rejected by operator"])
                return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, h, ["operator approval required"],
                                             approval_id=e["approval_id"])
        aid = f"apr_{len(self.queue) + 1}"
        self.queue[aid] = {"approval_id": aid, "action_hash": h, "status": "pending", "operation": action.operation}
        return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, h, ["operator approval required"], approval_id=aid)

    def approvals(self):
        return [dict(e) for e in self.queue.values() if e["status"] == "pending"]

    def resolve_approval(self, approval_id, approve, by, note=""):
        e = self.queue.get(approval_id)
        if e is None or e["status"] != "pending":
            return AuthorizationDecision(AuthVerdict.DENY, reasons=["unknown approval"])
        e["status"] = "approved" if approve else "rejected"
        if approve:
            return AuthorizationDecision(AuthVerdict.ALLOW, e["action_hash"], ["approved by " + by], grant="diag",
                                         approval_id=approval_id)
        return AuthorizationDecision(AuthVerdict.DENY, e["action_hash"], ["rejected by " + by + (": " + note if note else "")])


# --------------------------------------------------------------------------------------------------------------
# a LabWorld that outlives the runtime process (served over the project's RPC) — used by the kill -9 tests
# --------------------------------------------------------------------------------------------------------------
class LabHost:
    """Holds the world, verifier, policy and clock in the test process. ``execute`` de-duplicates by action id,
    which is the environment-side guarantee the runtime relies on when it resolves an in-flight action."""
    ENV = ("observe", "discover_tools", "validate_action", "execute", "poll_events", "get_resource_state",
           "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure")
    METHODS = ENV + ("v_verify", "v_check_signature", "v_check_measurement", "v_measure", "p_authorize", "p_check_grant", "p_record",
                     "p_approvals", "p_resolve_approval", "p_stopped", "p_limits_state", "clock_now",
                     "clock_advance")

    def __init__(self, lab_cfg: LabConfig):
        self.clock = SimClock()
        self.world = LabWorld(lab_cfg, self.clock)
        self.verifier = LabVerifier(self.world)
        self.policy = AllowAllPolicy()
        self.results: Dict[str, Any] = {}             # action id -> stored result
        self.execute_calls: Dict[str, int] = {}       # action id -> times the runtime asked
        self.real_executions: Dict[str, int] = {}     # action id -> times the world actually ran it
        self.issued_queries: List[str] = []

    def __getattr__(self, name: str):
        if name in ("observe", "discover_tools", "validate_action", "poll_events", "get_resource_state",
                    "get_pending_outcomes", "get_observable_metrics", "measure"):
            return getattr(self.world, name)
        raise AttributeError(name)

    def get_capabilities(self):
        return {**self.world.get_capabilities(), "idempotent_actions": True}

    def execute(self, action, authorization):
        aid = action.action_id
        self.execute_calls[aid] = self.execute_calls.get(aid, 0) + 1
        if aid in self.results:
            return self.results[aid]
        res = self.world.execute(action, authorization)
        self.real_executions[aid] = self.real_executions.get(aid, 0) + 1
        self.results[aid] = res
        self.issued_queries += [q.query_id for q in res.expected_outcomes]
        return res

    def v_verify(self, task): return self.verifier.verify(task)
    def v_check_signature(self, outcome): return self.verifier.check_signature(outcome)
    def v_measure(self, queries): return self.verifier.measure(queries)
    # signed measurements are checked like the verifier's outcomes (without it every verified measurement the runtime
    # took over RPC was an error: harmless between decisions, a pause once an ended objective's outcomes are settled)
    def v_check_measurement(self, o): return self.verifier.check_measurement(o)
    def p_authorize(self, action, task, context=None): return self.policy.authorize(action, task, context)
    def p_check_grant(self, action, decision): return self.policy.check_grant(action, decision)
    def p_record(self, action, result): return self.policy.record(action, result)
    def p_approvals(self): return self.policy.approvals()
    def p_resolve_approval(self, approval_id, approve, by, note=""):
        return self.policy.resolve_approval(approval_id, approve, by, note)
    def p_stopped(self): return self.policy.stopped()
    def p_limits_state(self): return self.policy.limits_state()
    def clock_now(self): return self.clock.now()
    def clock_advance(self, dt): return self.clock.advance(dt)
