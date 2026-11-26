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
