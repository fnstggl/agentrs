"""Persistent approval queue.

An entry is created when a decision is ``needs_approval``. It stores a preview of exactly what would be executed
and is bound to a hash of that content (``bind_hash``; for an action it is ``ActionSpec.content_hash()``): an
edited action has a different hash and therefore needs its own approval. An approval is used once.

States: ``pending`` -> ``approved`` | ``rejected`` | ``expired``; ``approved`` -> ``consumed`` (via consumed_at).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ... import contracts as C
from .store import Store, j, uj

PENDING, APPROVED, REJECTED, EXPIRED = "pending", "approved", "rejected", "expired"


def _row(r: Any) -> Dict[str, Any]:
    return {"approval_id": r["approval_id"], "status": r["status"], "kind": r["kind"], "task_id": r["task_id"],
            "action_hash": r["action_hash"], "bind_hash": r["bind_hash"], "created_at": r["created_at"],
            "expires_at": r["expires_at"], "resolved_at": r["resolved_at"], "resolved_by": r["resolved_by"],
            "note": r["note"], "preview": uj(r["preview_json"], {}), "reasons": uj(r["reasons_json"], []),
            "policy_ids": uj(r["policy_ids_json"], []), "consumed": r["consumed_at"] is not None}


class ApprovalQueue:
    def __init__(self, store: Store, ttl_s: float):
        self.store, self.ttl_s = store, float(ttl_s)

    def expire(self, now: float) -> int:
        cur = self.store.x("UPDATE approvals SET status=? WHERE status IN (?, ?) AND consumed_at IS NULL AND expires_at < ?",
                           (EXPIRED, PENDING, APPROVED, now))
        return cur.rowcount

    def find(self, bind_hash: str, now: float) -> Optional[Dict[str, Any]]:
        """The newest entry for this content, after expiry; consumed approvals are not returned."""
        self.expire(now)
        r = self.store.one("SELECT * FROM approvals WHERE bind_hash=? AND consumed_at IS NULL "
                           "ORDER BY created_at DESC, rowid DESC LIMIT 1", (bind_hash,))
        return None if r is None else _row(r)

    def request(self, *, bind_hash: str, action_hash: str, task_id: str, kind: str, preview: Dict[str, Any],
                action: Dict[str, Any], reasons: List[str], policy_ids: List[str], now: float) -> Dict[str, Any]:
        aid = C.new_id("apr")
        self.store.x(
            "INSERT INTO approvals(approval_id, bind_hash, action_hash, task_id, kind, status, created_at, expires_at,"
            " preview_json, action_json, reasons_json, policy_ids_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, bind_hash, action_hash, task_id, kind, PENDING, now, now + self.ttl_s, j(preview), j(action),
             j(reasons), j(policy_ids)))
        return self.get(aid)

    def get(self, approval_id: str) -> Optional[Dict[str, Any]]:
        r = self.store.one("SELECT * FROM approvals WHERE approval_id=?", (approval_id,))
        return None if r is None else _row(r)

    def stored_action(self, approval_id: str) -> Optional[Dict[str, Any]]:
        r = self.store.one("SELECT action_json FROM approvals WHERE approval_id=?", (approval_id,))
        return None if r is None else uj(r["action_json"], {})

    def resolve(self, approval_id: str, approve: bool, by: str, note: str, now: float) -> Dict[str, Any]:
        """Decide a pending request. Raises ``KeyError`` for an unknown id and ``ValueError`` if it is not pending
        (already decided or expired): a decision is never silently changed."""
        self.expire(now)
        with self.store.tx():
            r = self.store.one("SELECT status FROM approvals WHERE approval_id=?", (approval_id,))
            if r is None:
                raise KeyError(f"unknown approval {approval_id!r}")
            if r["status"] != PENDING:
                raise ValueError(f"approval {approval_id} is {r['status']}, not pending")
            self.store.x("UPDATE approvals SET status=?, resolved_at=?, resolved_by=?, note=?, expires_at=? "
                         "WHERE approval_id=?", (APPROVED if approve else REJECTED, now, str(by)[:200],
                                                 str(note)[:2000], now + self.ttl_s, approval_id))
        return self.get(approval_id)

    def consume(self, approval_id: str, now: float) -> bool:
        """Mark an approval as used. False when it is not (or no longer) an unused, unexpired approval."""
        cur = self.store.x("UPDATE approvals SET consumed_at=? WHERE approval_id=? AND status=? AND consumed_at IS NULL "
                           "AND expires_at >= ?", (now, approval_id, APPROVED, now))
        return cur.rowcount == 1

    def list(self, now: float, status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        self.expire(now)
        if status:
            rows = self.store.q("SELECT * FROM approvals WHERE status=? ORDER BY created_at DESC LIMIT ?",
                                (status, int(limit)))
        else:
            rows = self.store.q("SELECT * FROM approvals ORDER BY (status='pending') DESC, created_at DESC LIMIT ?",
                                (int(limit),))
        return [_row(r) for r in rows]

    def count(self) -> Dict[str, int]:
        return {r["status"]: r["n"] for r in self.store.q("SELECT status, COUNT(*) AS n FROM approvals GROUP BY status")}
