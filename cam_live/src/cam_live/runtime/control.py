"""cam_live.runtime.control — operator commands (spec 15.3).

Two channels, both outside the learner's decision-making:

* the run-directory control file (``contracts.RunFiles.CONTROL``): an append-only list of commands written by the
  CLI / dashboard and read by the runtime, which records what it applied in its own state file;
* the :class:`~cam_live.contracts.PolicyEngine`: approvals are resolved and emergency stops raised at the
  authorization service by the operator's own handle. The runtime only observes the result.

Commands govern whether and what CAM may execute; none of them carries a way of pursuing the objective. The
objective itself can only be changed by assigning a task file, which passes the same validation as any task.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import time
from typing import Any, Callable, Dict, List, Optional

from ..contracts import Clock, PolicyEngine, RunFiles, WallClock, new_id
from .logs import atomic_write_json, read_json

COMMANDS = ("pause", "resume", "stop", "approve", "reject", "approval_notice", "assign_objective",
            "export_checkpoint", "wake")
PID_FILE = "runtime.pid"


def read_pid(run_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(run_dir, PID_FILE)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                           # a zombie is not a live runtime
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def notify(run_dir: str) -> bool:
    """Wake a sleeping runtime (SIGUSR1). Returns False when no live runtime process is recorded."""
    pid = read_pid(run_dir)
    if not pid_alive(pid) or pid == os.getpid():
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ControlChannel:
    """The control file. Writers (operator tools) append under a lock; the runtime only reads."""

    def __init__(self, run_dir: str, clock: Optional[Clock] = None):
        self.run_dir, self.clock = run_dir, clock or WallClock()
        self.path = os.path.join(run_dir, RunFiles.CONTROL)

    def read(self) -> List[Dict[str, Any]]:
        doc = read_json(self.path, {}) or {}
        cmds = doc.get("commands", []) if isinstance(doc, dict) else []
        return [c for c in cmds if isinstance(c, dict) and c.get("cmd") in COMMANDS]

    def mtime(self) -> float:
        try:
            return os.stat(self.path).st_mtime_ns / 1e9
        except FileNotFoundError:
            return 0.0

    def send(self, cmd: str, by: str = "operator", wake: bool = True, **args: Any) -> str:
        if cmd not in COMMANDS:
            raise ValueError(f"unknown command {cmd!r}; known: {', '.join(COMMANDS)}")
        os.makedirs(self.run_dir, exist_ok=True)
        cid = new_id("cmd")
        with open(self.path + ".lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                doc = read_json(self.path, {}) or {}
                cmds = list(doc.get("commands", [])) if isinstance(doc, dict) else []
                cmds.append({"id": cid, "cmd": cmd, "by": by, "issued_at": self.clock.now(), "args": args})
                atomic_write_json(self.path, {"commands": cmds})
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        if wake:
            notify(self.run_dir)
        return cid


class OperatorControl:
    """What the CLI and the dashboard call. ``policy`` is the operator's own handle on the authorization service
    (when there is one); approvals and emergency stops go there first, so the learner's process is never the
    authority that approves its own action."""

    def __init__(self, run_dir: str, policy: Optional[PolicyEngine] = None, by: str = "operator",
                 clock: Optional[Clock] = None):
        self.run_dir, self.policy, self.by = run_dir, policy, by
        self.channel = ControlChannel(run_dir, clock)

    def pause(self, reason: str = "") -> str:
        return self.channel.send("pause", self.by, reason=reason)

    def resume(self) -> str:
        return self.channel.send("resume", self.by)

    def stop(self, reason: str = "", emergency: bool = False) -> str:
        if emergency and self.policy is not None:
            self.policy.emergency_stop(reason or "operator emergency stop", self.by)
        return self.channel.send("stop", self.by, reason=reason, emergency=bool(emergency))

    def _resolve(self, approval_id: str, approve: bool, note: str) -> str:
        if self.policy is not None:
            decision = self.policy.resolve_approval(approval_id, approve, self.by, note)
            return self.channel.send("approval_notice", self.by, approval_id=approval_id, approved=bool(approve),
                                     note=note, verdict=getattr(decision, "verdict", ""))
        return self.channel.send("approve" if approve else "reject", self.by, approval_id=approval_id, note=note)

    def approve(self, approval_id: str, note: str = "") -> str:
        return self._resolve(approval_id, True, note)

    def reject(self, approval_id: str, note: str = "") -> str:
        return self._resolve(approval_id, False, note)

    def assign_objective(self, task_path: str) -> str:
        """Queue a subsequent objective. The file is validated here and again by the runtime (same rules as any
        task: no strategy content, explicit limits)."""
        from .tasks import load_task
        task_path = os.path.abspath(task_path)
        load_task(task_path, for_run=True)
        return self.channel.send("assign_objective", self.by, path=task_path, sha256=file_sha256(task_path))

    def export_checkpoint(self, dest: str = "") -> str:
        return self.channel.send("export_checkpoint", self.by, dest=os.path.abspath(dest) if dest else "")

    def wake(self) -> str:
        return self.channel.send("wake", self.by)

    def result(self, command_id: str) -> Optional[Dict[str, Any]]:
        st = read_json(os.path.join(self.run_dir, RunFiles.STATE), {}) or {}
        return (st.get("control_results") or {}).get(command_id)

    def wait_result(self, command_id: str, timeout_s: float = 30.0, poll_s: float = 0.1,
                    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Optional[Dict[str, Any]]:
        """Block (operator side, real time) until the runtime reports the command applied."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            r = self.result(command_id)
            if r is not None and (predicate is None or predicate(r)):
                return r
            time.sleep(poll_s)
        return self.result(command_id)
