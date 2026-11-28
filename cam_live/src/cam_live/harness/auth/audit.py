"""Append-only, hash-chained audit log of the authorization service.

One JSON object per line::

    {"seq": n, "ts": t, "event": "...", "data": {...}, "prev": "<hash of entry n-1>", "hash": "...", "mac": "..."}

``hash = sha256(prev || canonical(seq, ts, event, data))`` chains every entry to all earlier ones, so an edit, a
deletion or a reordering anywhere breaks the chain at that point. ``mac = HMAC(key, hash)`` (key readable only by
the authorization uid) stops anyone without the key from rebuilding a consistent chain after a change, and a
separate *head* file (sequence number + hash + MAC, replaced atomically after every append) makes removal of the
newest entries detectable as well.

The file is created 0600 and, in separated mode, owned by the authorization uid inside a root-owned sticky run
directory: the learner uid can neither write, replace nor delete it. Linux' append-only inode flag would need
CAP_LINUX_IMMUTABLE, which this container does not have; see docs/EA/SECURITY.md.

Entries never contain secret values: the caller passes a scrubber that is applied to every entry.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

GENESIS = "0" * 64


def _canon(seq: int, ts: float, event: str, data: Any) -> bytes:
    return json.dumps({"seq": seq, "ts": ts, "event": event, "data": data}, sort_keys=True, default=str,
                      separators=(",", ":")).encode()


def entry_hash(prev: str, seq: int, ts: float, event: str, data: Any) -> str:
    return hashlib.sha256(prev.encode() + b"\n" + _canon(seq, ts, event, data)).hexdigest()


def _mac(key: Optional[bytes], h: str) -> str:
    return hmac.new(key, h.encode(), hashlib.sha256).hexdigest() if key else ""


@dataclass
class AuditVerification:
    ok: bool
    entries: int = 0
    head_hash: str = GENESIS
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


class AuditLog:
    def __init__(self, path: str, key: Optional[bytes] = None, head_path: Optional[str] = None,
                 scrub: Optional[Callable[[Any], Any]] = None, fsync: bool = True):
        self.path, self.key, self.fsync = path, key, fsync
        self.head_path = head_path or (path + ".head")
        self.scrub = scrub
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self._seq, self._prev, self._size = 0, GENESIS, -1
        with self._lock:
            self._locked(self._open)

    # -- internals -------------------------------------------------------------------------------------------
    def _locked(self, fn: Callable[[], Any]) -> Any:
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _resync(self) -> None:
        """Pick up the chain position from the file (another instance may have appended since)."""
        size = os.fstat(self._fd).st_size
        if size == self._size:
            return
        seq, prev = 0, GENESIS
        if size:
            with open(self.path, "rb") as f:
                f.seek(max(0, size - (1 << 20)))
                tail = f.read().splitlines()
            last = None
            for line in reversed(tail):
                if line.strip():
                    last = line
                    break
            try:
                e = json.loads(last.decode()) if last is not None else None
                seq, prev = int(e["seq"]), str(e["hash"])
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError):
                # a damaged tail must stay visible to the verifier: continue the numbering after it, chained to a
                # marker that can never be a real hash
                seq, prev = self._count_lines(), "damaged-tail"
        self._seq, self._prev, self._size = seq, prev, size

    def _open(self) -> None:
        self._resync()
        self._reconcile_head()

    def _reconcile_head(self) -> None:
        """An entry is written and synced before the head file is replaced, so a stop or kill between the two
        leaves the head behind the log (G2-harness R2-m3: a false tamper alarm). At open, if the head names an
        entry of this log and every entry after it chains from it and carries a valid MAC under this key, the head
        is advanced to the log's last entry. Anything else -- a head ahead of the log, a head naming no entry, a
        break in the chain, a bad MAC, no head at all -- is left exactly as it is, for the verifier to report."""
        if not self.key or not os.path.exists(self.head_path) or self._seq <= 0 or self._prev == "damaged-tail":
            return
        try:
            with open(self.head_path, "rb") as f:
                h = json.loads(f.read().decode())
            hseq, hhash = int(h["seq"]), str(h["hash"])
            if not hmac.compare_digest(_mac(self.key, f"{hseq}:{hhash}"), str(h.get("mac", ""))):
                return
        except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError):
            return
        if hseq >= self._seq or self._seq - hseq > 64:
            return
        prev, found = (GENESIS if hseq == 0 else None), hseq == 0
        with open(self.path, "rb") as f:
            lines = [ln for ln in f.read().split(b"\n") if ln.strip()]
        if len(lines) != self._seq:
            return
        for raw in lines[max(0, hseq - 1):]:
            try:
                e = json.loads(raw.decode())
                seq, ts, event, data = int(e["seq"]), float(e["ts"]), str(e["event"]), e["data"]
                eprev, ehash, emac = str(e["prev"]), str(e["hash"]), str(e.get("mac", ""))
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                return
            if not found:
                if seq != hseq or ehash != hhash:
                    return
                found, prev = True, ehash
                continue
            if eprev != prev or entry_hash(eprev, seq, ts, event, data) != ehash or \
                    not hmac.compare_digest(_mac(self.key, ehash), emac):
                return
            prev = ehash
        if found and prev == self._prev:
            self._write_head()

    def _count_lines(self) -> int:
        with open(self.path, "rb") as f:
            return sum(1 for line in f if line.strip())

    def _write_head(self) -> None:
        tmp = f"{self.head_path}.{os.getpid()}.tmp"
        body = {"seq": self._seq, "hash": self._prev, "mac": _mac(self.key, f"{self._seq}:{self._prev}")}
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, json.dumps(body, sort_keys=True).encode())
            if self.fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, self.head_path)

    # -- API -------------------------------------------------------------------------------------------------
    def append(self, event: str, ts: float, **data: Any) -> Dict[str, Any]:
        if self.scrub is not None:
            data = self.scrub(data)
        data = json.loads(json.dumps(data, sort_keys=True, default=str))     # exactly what will be re-read

        def do() -> Dict[str, Any]:
            self._resync()
            seq = self._seq + 1
            h = entry_hash(self._prev, seq, float(ts), event, data)
            entry = {"seq": seq, "ts": float(ts), "event": event, "data": data, "prev": self._prev, "hash": h,
                     "mac": _mac(self.key, h)}
            line = (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
            os.write(self._fd, line)
            if self.fsync:
                os.fsync(self._fd)
            self._seq, self._prev = seq, h
            self._size = os.fstat(self._fd).st_size
            self._write_head()
            return entry

        with self._lock:
            return self._locked(do)

    def head(self) -> Dict[str, Any]:
        with self._lock:
            return {"seq": self._seq, "hash": self._prev}

    def tail(self, n: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(self.path, "rb") as f:
            lines = f.read().splitlines()
        for line in lines[-max(0, int(n)):]:
            try:
                out.append(json.loads(line.decode()))
            except (ValueError, UnicodeDecodeError):
                out.append({"damaged": True})
        return out

    def close(self) -> None:
        with self._lock:
            try:
                os.close(self._fd)
            except OSError:
                pass


def verify_audit_log(path: str, key: Optional[bytes] = None, head_path: Optional[str] = None,
                     require_head: Optional[bool] = None, expect_head: Optional[Tuple[int, str]] = None) -> AuditVerification:
    """Check the whole chain. Detects edited, deleted, inserted, duplicated and reordered entries; with the head
    file (or an externally kept ``expect_head = (seq, hash)`` anchor) also a truncated tail. With ``key`` every MAC
    is checked too, so a chain rebuilt without the key fails.

    The head file is REQUIRED by default: a log that was emptied (or truncated) together with the removal of its
    head file fails. ``require_head=False`` checks the bare chain only (consistency, not completeness); an
    ``expect_head`` anchor replaces the head file. Without ``head_path`` the head is looked for next to the log
    (``<log>.head``) and where the authorization service keeps it (``<dir of log>/auth/audit.head``)."""
    if require_head is None:
        require_head = expect_head is None or expect_head[0] <= 0
    res = AuditVerification(ok=True)
    hp = head_path or next((c for c in (path + ".head", os.path.join(os.path.dirname(os.path.abspath(path)), "auth",
                                                                     "audit.head")) if os.path.exists(c)), path + ".head")
    head_raw: Optional[bytes] = None
    try:
        with open(path, "rb") as f:
            # an append writes its line and then the head under an exclusive lock on the log: reading both under a
            # shared lock never sees the moment in between (G2-harness R2-m3: a false tamper alarm otherwise)
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            except OSError:
                pass
            try:
                lines = f.read().split(b"\n")
                if os.path.exists(hp):
                    try:
                        with open(hp, "rb") as hf:
                            head_raw = hf.read()
                    except OSError:
                        head_raw = b""
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError as e:
        return AuditVerification(False, errors=[f"cannot read {path}: {e.strerror}"])
    prev, n, anchored = GENESIS, 0, None
    for i, raw in enumerate(lines, 1):
        if not raw.strip():
            if i != len(lines):
                res.errors.append(f"line {i}: blank line inside the log")
            continue
        try:
            e = json.loads(raw.decode())
            seq, ts, event, data = int(e["seq"]), float(e["ts"]), str(e["event"]), e["data"]
            eprev, ehash, emac = str(e["prev"]), str(e["hash"]), str(e.get("mac", ""))
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            res.errors.append(f"line {i}: not a valid audit entry")
            prev = "invalid"
            n += 1
            continue
        n += 1
        if seq != n:
            res.errors.append(f"line {i}: sequence {seq} where {n} was expected (deleted, inserted or reordered)")
        if eprev != prev:
            res.errors.append(f"line {i}: does not chain to the previous entry")
        if entry_hash(eprev, seq, ts, event, data) != ehash:
            res.errors.append(f"line {i}: content does not match its hash (edited)")
        if key and not hmac.compare_digest(_mac(key, ehash), emac):
            res.errors.append(f"line {i}: bad MAC")
        prev = ehash
        if expect_head is not None and n == expect_head[0]:
            anchored = ehash
    res.entries, res.head_hash = n, prev
    if head_raw is not None:
        try:
            h = json.loads(head_raw.decode())
            if int(h["seq"]) != n or str(h["hash"]) != prev:
                res.errors.append(f"head file says {h['seq']} entries ending in {str(h['hash'])[:12]}, "
                                  f"log has {n} ending in {prev[:12]} (truncated or extended outside the service)")
            if key and not hmac.compare_digest(_mac(key, f"{h['seq']}:{h['hash']}"), str(h.get("mac", ""))):
                res.errors.append("head file: bad MAC")
        except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError):
            res.errors.append("head file is unreadable")
    elif require_head or (n and head_path):
        res.errors.append("head file is missing")
    if expect_head is not None and expect_head[0] > 0 and anchored != expect_head[1]:
        res.errors.append("log does not contain the anchored head (entries were removed or replaced)")
    res.ok = not res.errors
    return res
