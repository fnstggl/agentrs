"""cam_live.harness.operator_requests — CAM asks the operator for something it cannot obtain itself. MASTER-OWNED.

During its lifetime the learner may conclude that it needs a capability, permission, credential, resource, piece of
information or decision that only the operator can provide. The tool ``internal.request_operator`` (see
``harness/internal.py``) records such a request here; nothing is sent anywhere. The operator reads and answers:

    python -m cam_live.harness.operator_requests --run-dir R list [--all]
    python -m cam_live.harness.operator_requests --run-dir R grant  <request id> --note "what was provided"
    python -m cam_live.harness.operator_requests --run-dir R deny   <request id> --note "why not"
    python -m cam_live.harness.operator_requests --run-dir R answer <request id> --note "the information"

The answer comes back to the learner as an ordinary event observation at its next poll. An answer **grants nothing
by itself**: authorization is decided only by the authorization service, credentials only enter through
``cam_live.harness.credentials``. The file is append-only; every change appends a new revision of the request.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional

#: ``account``: an account the operator would create or that CAM may register only with the operator's approval;
#: ``verification``: a step only a person may take (a human-verification challenge, an e-mail or phone confirmation,
#: an identity check) -- CAM never answers those itself (hard constraint X-HUMAN-VERIFICATION)
KINDS = ("capability", "permission", "credential", "resource", "information", "decision", "account", "verification")
STATUSES = ("open", "granted", "denied", "answered", "withdrawn")
FILE = "operator_requests.jsonl"


def _key(kind: str, title: str) -> str:
    return kind + ":" + re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


class OperatorRequests:
    """Append-only request file in ``directory``. Safe for one writer process per side (executor / operator)."""

    def __init__(self, directory: str, now: Any = time.time, max_open: int = 20):
        self.dir, self.now, self.max_open = os.path.abspath(directory), now, max_open
        self.path = os.path.join(self.dir, FILE)
        self._lock = threading.Lock()

    # ---- storage -------------------------------------------------------------------------------------------
    def _rows(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path) as f:
                return [json.loads(x) for x in f if x.strip()]
        except OSError:
            return []

    def _append(self, rec: Dict[str, Any]) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all(self) -> List[Dict[str, Any]]:
        """Latest revision of every request, oldest first."""
        latest: Dict[str, Dict[str, Any]] = {}
        for r in self._rows():
            latest[r["request_id"]] = dict(latest.get(r["request_id"], {}), **r)
        return sorted(latest.values(), key=lambda r: r.get("ts", 0.0))

    def open_requests(self) -> List[Dict[str, Any]]:
        return [r for r in self.all() if r.get("status") == "open"]

    # ---- learner side --------------------------------------------------------------------------------------
    def request(self, kind: str, title: str, description: str, justification: str = "", *, blocking: bool = False,
                action_id: str = "", source: str = "learner") -> Dict[str, Any]:
        """Record a request. An identical open request is returned instead of being duplicated."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        with self._lock:
            existing = self.all()
            key = _key(kind, title)
            for r in existing:
                if r.get("status") == "open" and r.get("key") == key:
                    return dict(r, duplicate=True)
            if sum(1 for r in existing if r.get("status") == "open") >= self.max_open:
                raise PermissionError(f"{self.max_open} requests are already waiting for the operator")
            rec = {"request_id": "req_" + os.urandom(6).hex(), "ts": float(self.now()), "status": "open", "kind": kind,
                   "key": key, "title": title[:200], "description": description[:4000],
                   "justification": justification[:4000], "blocking": bool(blocking), "action_id": action_id,
                   "source": source, "delivered": False}
            self._append(rec)
            return rec

    def undelivered_answers(self) -> List[Dict[str, Any]]:
        """Resolved requests whose answer the learner has not seen yet; marks them delivered."""
        with self._lock:
            out = [r for r in self.all() if r.get("status") in ("granted", "denied", "answered") and not r.get("delivered")]
            for r in out:
                self._append({"request_id": r["request_id"], "delivered": True, "delivered_at": float(self.now())})
            return out

    # ---- operator side -------------------------------------------------------------------------------------
    def resolve(self, request_id: str, status: str, note: str = "", by: str = "operator") -> Dict[str, Any]:
        if status not in ("granted", "denied", "answered", "withdrawn"):
            raise ValueError("status must be granted, denied, answered or withdrawn")
        with self._lock:
            cur = {r["request_id"]: r for r in self.all()}.get(request_id)
            if cur is None:
                raise KeyError(f"no request {request_id}")
            if cur.get("status") != "open":
                raise ValueError(f"request {request_id} is already {cur.get('status')}")
            upd = {"request_id": request_id, "status": status, "resolution_note": note[:4000], "resolved_by": by,
                   "resolved_at": float(self.now()), "delivered": False}
            self._append(upd)
            return dict(cur, **upd)


def find_file(run_dir: str) -> str:
    """The request file of a run: the executor's state directory in separated mode, else the run directory."""
    R = os.path.abspath(run_dir)
    for d in (os.path.join(R, "exec"), R, os.path.join(R, "workspace")):
        if os.path.exists(os.path.join(d, FILE)):
            return d
    return os.path.join(R, "exec") if os.path.isdir(os.path.join(R, "exec")) else R


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.operator_requests")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("command", choices=["list", "grant", "deny", "answer"])
    ap.add_argument("request_id", nargs="?", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    reqs = OperatorRequests(find_file(a.run_dir))
    if a.command == "list":
        rows = reqs.all() if a.all else reqs.open_requests()
        print(json.dumps(rows, indent=1, sort_keys=True))
        return 0
    if not a.request_id:
        ap.error("a request id is required")
    try:
        out = reqs.resolve(a.request_id, {"grant": "granted", "deny": "denied", "answer": "answered"}[a.command], a.note)
    except (KeyError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
