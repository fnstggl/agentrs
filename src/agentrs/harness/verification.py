"""Outcome verification service: independent, environment-side checks of task success.

Verifiers read privileged ground truth or authoritative provider state (e.g. the lab's answer
helpfulness truth through its control channel, a pull request's merge state) and never the agent's
claim. Verifiers that depend on delayed outcomes return ``pending`` until they can decide; pending
experiences are not rewards of 0.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sqlalchemy import select

from agentrs.harness.db import Database, Experience, Task
from agentrs.harness.tasks import TaskRegistry

Verdict = dict[str, Any]  # {"status": pending|matured|censored, "success": bool, "evidence": {...}, "at": t}


class VerificationService:
    def __init__(self, db: Database, tasks: TaskRegistry, ledger=None):
        self.db, self.tasks, self.ledger = db, tasks, ledger
        self.verifiers: dict[str, Callable[[Task, Experience], Verdict]] = {}

    def register(self, kind: str, fn: Callable[[Task, Experience], Verdict]) -> None:
        self.verifiers[kind] = fn

    def kinds(self) -> set[str]:
        return set(self.verifiers)

    def record_attempt(self, task: Task, agent_id: str, action_id: Optional[str], ledger_action_id: Optional[str],
                       checkpoint: Optional[str], evidence: dict) -> str:
        with self.db.session() as s:
            e = Experience(task_id=task.id, agent_id=agent_id, action_id=action_id, ledger_action_id=ledger_action_id,
                           model_checkpoint=checkpoint, evidence=evidence, split=task.split,
                           verified_by=task.verifier.get("kind", ""))
            s.add(e)
            s.flush()
            return e.id

    def run(self) -> dict[str, int]:
        stats = {"matured": 0, "pending": 0, "censored": 0, "success": 0}
        with self.db.session() as s:
            pend = list(s.scalars(select(Experience).where(Experience.reward_status == "pending")))
        for e in pend:
            t = self.tasks.get(e.task_id)
            fn = self.verifiers.get((t.verifier or {}).get("kind", ""))
            if fn is None:
                continue
            v = fn(t, e)
            stats[v["status"]] += 1
            if v["status"] == "pending":
                continue
            with self.db.session() as s:
                row = s.get(Experience, e.id)
                row.reward_status = v["status"]
                row.reward = (1.0 if v.get("success") else 0.0) if v["status"] == "matured" else None
                row.matured_at = v.get("at", time.time())
                row.evidence = {**(row.evidence or {}), "verdict": v.get("evidence", {})}
            if v["status"] == "matured":
                stats["success"] += int(bool(v.get("success")))
                self.tasks.set_state(t.id, "succeeded" if v.get("success") else "failed",
                                     {"verified_by": t.verifier["kind"], **v.get("evidence", {})})
                if self.ledger is not None and e.ledger_action_id:
                    a = self.ledger.action(e.ledger_action_id)
                    self.ledger.record_reward(e.ledger_action_id, "delayed", 1.0 if v.get("success") else 0.0,
                                              env_time=v.get("at"), maturity_time=v.get("at"),
                                              checkpoint=a["policy_checkpoint"] if a else e.model_checkpoint,
                                              objective=f"task:{t.verifier['kind']}",
                                              idem=f"{e.ledger_action_id}:task:{t.verifier['kind']}")
            else:
                self.tasks.set_state(t.id, "failed", {"censored": True})
        return stats


def socialweblab_help_verifier(control: Callable[..., dict], now: Callable[[], float]):
    """Helpful answer within the window, from the lab's privileged truth (control channel)."""

    def v(task: Task, e: Experience) -> Verdict:
        item = (e.evidence or {}).get("item_id")
        if item is None:
            return {"status": "censored", "evidence": {"reason": "nothing was posted"}}
        within = float(task.verifier["within_s"])
        created = float(e.evidence["posted_at"])
        if now() - created < within:
            return {"status": "pending"}
        r = control("verify_help", item_id=int(item), within_s=within)
        return {"status": "matured", "success": bool(r["success"]), "at": created + within,
                "evidence": {"item_id": item, "n_answers": r.get("n_answers"), "first_helpful_at": r.get("first_helpful_at")}}

    return v
