"""Append-only, hash-chained log of everything the bridge does (for reproducibility and audit).

One JSON object per line: {"i": n, "prev": <hash of line n-1>, "body": {...}, "hash": blake2b(prev + canonical(body))}.
The file is opened with O_APPEND; an existing log is verified before anything is appended and a corrupt log is
never extended. `verify_log(path)` recomputes the chain and reports the first line that does not fit.
"""
from __future__ import annotations

import hashlib
import json
import os

GENESIS = "0" * 32


class LogCorrupt(Exception):
    pass


def _canon(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(prev: str, body: dict) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(prev.encode())
    h.update(_canon(body))
    return h.hexdigest()


def verify_log(path: str) -> dict:
    """{"ok", "records", "head", "error"}: recompute the chain of a log file."""
    prev, n = GENESIS, 0
    try:
        with open(path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    return {"ok": False, "records": n, "head": prev, "error": f"line {n + 1}: torn write (no newline)"}
                try:
                    rec = json.loads(raw)
                    body = rec["body"]
                    if rec["i"] != n or rec["prev"] != prev or rec["hash"] != _hash(prev, body):
                        raise ValueError("chain mismatch")
                except (ValueError, KeyError, TypeError) as e:
                    return {"ok": False, "records": n, "head": prev, "error": f"line {n + 1}: {e}"}
                prev, n = rec["hash"], n + 1
    except FileNotFoundError:
        return {"ok": True, "records": 0, "head": GENESIS, "error": None}
    return {"ok": True, "records": n, "head": prev, "error": None}


def read_log(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return [json.loads(line)["body"] for line in f]


class ChainedLog:
    def __init__(self, path: str):
        if not isinstance(path, str) or not path:
            raise ValueError("a log path is required: the bridge never runs unrecorded")
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        v = verify_log(self.path)
        if not v["ok"]:
            raise LogCorrupt(f"{self.path}: {v['error']}; a corrupt log is never extended")
        self.head, self.n = v["head"], v["records"]
        self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)

    def append(self, body: dict) -> str:
        if self._fd is None:
            raise LogCorrupt("log is closed")
        h = _hash(self.head, body)
        line = json.dumps({"i": self.n, "prev": self.head, "body": body, "hash": h}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        os.write(self._fd, line)
        os.fsync(self._fd)
        self.head, self.n = h, self.n + 1
        return h

    def count(self, kind: str, **match) -> int:
        """Records of one kind already in the file (used for budgets that must survive restarts)."""
        n = 0
        for body in read_log(self.path):
            if body.get("kind") == kind and all(body.get(k) == v for k, v in match.items()):
                n += 1
        return n

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
