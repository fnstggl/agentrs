"""Durable job queue on the control-plane database (the workflow engine used here instead of a
Temporal.io server). Jobs survive restarts; workers take leases (``SELECT … FOR UPDATE SKIP LOCKED``
on PostgreSQL), expired leases are recovered, failures retry with exponential backoff, and every job
has an idempotency key."""

from __future__ import annotations

import random
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from agentrs.harness.db import Database, Job

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


class JobQueue:
    def __init__(self, db: Database, lease_s: float = 120.0, backoff_s: float = 15.0):
        self.db, self.lease_s, self.backoff_s = db, lease_s, backoff_s

    def enqueue(self, kind: str, payload: dict, run_at: Optional[float] = None, idem: Optional[str] = None,
                max_attempts: int = 5) -> Optional[str]:
        jid = uuid.uuid4().hex
        try:
            with self.db.session() as s:
                s.add(Job(id=jid, kind=kind, payload=payload, run_at=run_at or time.time(), idem=idem or jid,
                          max_attempts=max_attempts))
        except IntegrityError:
            return None  # same idempotency key already queued
        return jid

    def claim(self, owner: str, kinds: Optional[list[str]] = None, limit: int = 16, now: Optional[float] = None) -> list[Job]:
        now = now or time.time()
        with self.db.session() as s:
            s.execute(update(Job).where(Job.status == "leased", Job.lease_until < now)
                      .values(status="pending", lease_owner=None))
            q = select(Job).where(Job.status == "pending", Job.run_at <= now)
            if kinds:
                q = q.where(Job.kind.in_(kinds))
            q = q.order_by(Job.run_at).limit(limit)
            if self.db.is_postgres:
                q = q.with_for_update(skip_locked=True)
            rows = list(s.scalars(q))
            for j in rows:
                j.status, j.lease_owner, j.lease_until, j.attempts = "leased", owner, now + self.lease_s, j.attempts + 1
            return rows

    def complete(self, job_id: str, result: dict) -> None:
        with self.db.session() as s:
            s.execute(update(Job).where(Job.id == job_id).values(status="done", result=result, done_at=time.time(),
                                                                lease_owner=None))

    def fail(self, job_id: str, error: str) -> str:
        with self.db.session() as s:
            j = s.get(Job, job_id)
            if j.attempts >= j.max_attempts:
                j.status, j.error, j.done_at = "failed", error[-2000:], time.time()
                return "failed"
            j.status, j.error, j.lease_owner = "pending", error[-2000:], None
            j.run_at = time.time() + self.backoff_s * 2 ** (j.attempts - 1) * (1 + 0.2 * random.random())
            return "retry"

    def cancel(self, idem_prefix: str) -> int:
        with self.db.session() as s:
            return s.execute(update(Job).where(Job.idem.like(idem_prefix + "%"), Job.status.in_(["pending", "leased"]))
                             .values(status="cancelled")).rowcount

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self.db.session() as s:
            for j in s.scalars(select(Job)):
                out[f"{j.kind}:{j.status}"] = out.get(f"{j.kind}:{j.status}", 0) + 1
        return out

    async def run_due(self, handlers: dict[str, Handler], owner: Optional[str] = None, limit: int = 16) -> int:
        owner = owner or f"worker-{uuid.uuid4().hex[:6]}"
        jobs = self.claim(owner, list(handlers), limit)
        for j in jobs:
            try:
                res = handlers[j.kind](j.payload)
                if hasattr(res, "__await__"):
                    res = await res
                self.complete(j.id, res or {})
            except Exception:  # noqa: BLE001 - record and retry
                self.fail(j.id, traceback.format_exc())
        return len(jobs)
