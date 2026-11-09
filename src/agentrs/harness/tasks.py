"""Task registry, and practice-task generation kept separate from solving and verification.

* ``TaskRegistry`` stores tasks (goal, environment, required capabilities, allowed actions, budget,
  initial observable state, verifier, failure conditions, time limit, approval need, authorised tier)
  from any source: human assignment, authorised external workflow, benchmark, practice, self-hosted
  multi-agent environment.
* ``PracticeTaskGenerator`` may only emit tasks whose verifier kind is registered with the
  independent ``VerificationService``; the solver (agent runtime) never decides success, and private
  verifier parameters are never shown to the agent (``public_view``).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy import select, update

from agentrs.harness.db import Database, Task


class TaskError(ValueError):
    pass


class TaskRegistry:
    def __init__(self, db: Database, known_verifiers: Optional[set[str]] = None):
        self.db = db
        self.known_verifiers = known_verifiers

    def create(self, task_id: str, source: str, goal: str, environment: str, required_capabilities: list[str],
               allowed_actions: list[str], verifier: dict[str, Any], budget: Optional[dict] = None,
               initial_state: Optional[dict] = None, failure_conditions: Optional[list] = None,
               time_limit_s: float = 3600.0, needs_approval: bool = False, authorized_tier: int = 1,
               split: str = "train") -> str:
        if not verifier or "kind" not in verifier:
            raise TaskError("every task needs an independent verifier")
        if self.known_verifiers is not None and verifier["kind"] not in self.known_verifiers:
            raise TaskError(f"unknown verifier kind {verifier['kind']!r}")
        with self.db.session() as s:
            if s.get(Task, task_id) is not None:
                return task_id
            s.add(Task(id=task_id, source=source, goal=goal, environment=environment,
                       required_capabilities=required_capabilities, allowed_actions=allowed_actions,
                       budget=budget or {}, initial_state=initial_state or {}, verifier=verifier,
                       failure_conditions=failure_conditions or [], time_limit_s=time_limit_s,
                       needs_approval=needs_approval, authorized_tier=authorized_tier, split=split))
        return task_id

    def get(self, task_id: str) -> Optional[Task]:
        with self.db.session() as s:
            return s.get(Task, task_id)

    def claim(self, agent_id: str, environments: Optional[list[str]] = None, split: Optional[str] = None) -> Optional[Task]:
        with self.db.session() as s:
            q = select(Task).where(Task.state == "queued")
            if environments:
                q = q.where(Task.environment.in_(environments))
            if split:
                q = q.where(Task.split == split)
            q = q.order_by(Task.created_at).limit(1)
            if self.db.is_postgres:
                q = q.with_for_update(skip_locked=True)
            t = s.scalars(q).first()
            if t is None:
                return None
            t.state, t.assigned_agent, t.updated_at = "running", agent_id, time.time()
            return t

    def set_state(self, task_id: str, state: str, result: Optional[dict] = None) -> None:
        with self.db.session() as s:
            vals: dict[str, Any] = {"state": state, "updated_at": time.time()}
            if result is not None:
                vals["result"] = result
            s.execute(update(Task).where(Task.id == task_id).values(**vals))

    def list(self, state: Optional[str] = None) -> list[Task]:
        with self.db.session() as s:
            q = select(Task)
            if state:
                q = q.where(Task.state == state)
            return list(s.scalars(q.order_by(Task.created_at)))

    @staticmethod
    def public_view(t: Task) -> dict:
        """What the solving agent may see (no verifier parameters)."""
        return {"id": t.id, "goal": t.goal, "environment": t.environment, "allowed_actions": t.allowed_actions,
                "required_capabilities": t.required_capabilities, "initial_state": t.initial_state,
                "time_limit_s": t.time_limit_s}


class PracticeTaskGenerator:
    """Proposes SocialWebLab practice tasks ('get a helpful answer within a day'). Their success is
    decided by the environment's own verifier, not by the generator or the solver."""

    def __init__(self, registry: TaskRegistry):
        self.registry = registry

    def generate(self, split: str, index: int, group: Optional[str] = None, attempt: int = 0, within_s: float = 86400.0,
                 accounts_hint: Optional[str] = None) -> str:
        from agentrs.socialweblab.agent_tasks import make_help_task

        spec = make_help_task(split, index, account=accounts_hint or "agent")
        tid = f"{spec.task_id}#{attempt}"
        return self.registry.create(
            task_id=tid, source="practice", goal=spec.goal, environment="socialweblab",
            required_capabilities=["socialweblab:create_post", "socialweblab:read_content"],
            allowed_actions=["socialweblab:create_post", "socialweblab:read_user", "socialweblab:read_feed"],
            verifier={"kind": "socialweblab_help_within", "within_s": within_s, "correct_space": spec.private["space"]},
            budget={"max_actions": 3, "group": group or spec.task_id}, time_limit_s=30 * 86400,
            authorized_tier=1, split=split)
