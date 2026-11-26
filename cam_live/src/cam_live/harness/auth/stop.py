"""Pause and emergency stop.

The flags are *files* in the authorization service's private directory, so they work across processes, survive a
restart of any service and can be set by the operator with nothing but a shell when everything else is down::

    <run_dir>/auth/STOP    emergency stop: nothing is authorized, every outstanding grant is void
    <run_dir>/auth/PAUSE   pause: nothing new is authorized until the operator resumes

``engaged()`` / ``paused()`` hit the filesystem on every call (no cache): the authorization path asks before it
looks at anything else, and a flag set by another process is seen by the very next decision.
Only the operator resets a stop; that is enforced by the service's peer-uid check and by file ownership.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

STOP_FILE, PAUSE_FILE = "STOP", "PAUSE"


def _write_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, json.dumps(payload, sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(tmp, path)
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _read(path: str) -> Optional[Dict[str, Any]]:
    """None when the flag is absent. A flag file that exists but cannot be parsed still counts as set."""
    try:
        with open(path, "rb") as f:
            raw = f.read(65536)
    except FileNotFoundError:
        return None
    except OSError as e:                   # unreadable flag: fail closed
        return {"reason": f"flag unreadable: {e.strerror}", "by": "?", "ts": 0.0}
    try:
        d = json.loads(raw.decode() or "{}")
        return d if isinstance(d, dict) else {"reason": "flag present", "by": "?", "ts": 0.0}
    except (ValueError, UnicodeDecodeError):
        return {"reason": "flag present", "by": "?", "ts": 0.0}


class StopSwitch:
    def __init__(self, state_dir: str):
        self.dir = state_dir
        os.makedirs(state_dir, mode=0o700, exist_ok=True)
        self.stop_path = os.path.join(state_dir, STOP_FILE)
        self.pause_path = os.path.join(state_dir, PAUSE_FILE)

    # emergency stop ---------------------------------------------------------------------------------------
    def engage(self, reason: str, by: str, ts: float) -> Dict[str, Any]:
        cur = _read(self.stop_path)
        if cur is not None:                # keep the first cause
            return cur
        rec = {"reason": str(reason)[:500], "by": str(by)[:200], "ts": float(ts)}
        _write_atomic(self.stop_path, rec)
        return rec

    def engaged(self) -> Optional[Dict[str, Any]]:
        return _read(self.stop_path)

    def reset(self) -> bool:
        """Remove the stop flag. Call only for the operator (the service checks the caller)."""
        try:
            os.unlink(self.stop_path)
            return True
        except FileNotFoundError:
            return False

    # pause ------------------------------------------------------------------------------------------------
    def pause(self, reason: str, by: str, ts: float) -> Dict[str, Any]:
        rec = {"reason": str(reason)[:500], "by": str(by)[:200], "ts": float(ts)}
        _write_atomic(self.pause_path, rec)
        return rec

    def paused(self) -> Optional[Dict[str, Any]]:
        return _read(self.pause_path)

    def resume(self) -> bool:
        try:
            os.unlink(self.pause_path)
            return True
        except FileNotFoundError:
            return False

    def blocked(self) -> Optional[str]:
        """Reason why nothing may be authorized right now, or None."""
        s = self.engaged()
        if s is not None:
            return f"emergency stop is engaged ({s.get('reason', '')}; by {s.get('by', '?')})"
        p = self.paused()
        if p is not None:
            return f"paused ({p.get('reason', '')}; by {p.get('by', '?')})"
        return None


def stop_flag_path(run_dir: str) -> str:
    return os.path.join(run_dir, "auth", STOP_FILE)


def is_stopped(run_dir: str) -> bool:
    """For other processes that can read the flag (the launcher, the operator)."""
    return os.path.lexists(stop_flag_path(run_dir))
