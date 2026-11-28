"""cam_live.runtime.control — operator commands (spec 15.3).

Two channels, both outside the learner's decision-making:

* the run-directory control file (``contracts.RunFiles.CONTROL``): an append-only list of commands written by the
  CLI / dashboard and read by the runtime, which records what it applied in its own state file;
* the :class:`~cam_live.contracts.PolicyEngine`: approvals are resolved and emergency stops raised at the
  authorization service by the operator's own handle. The runtime only observes the result.

Commands are authenticated. :class:`OperatorControl` signs every command with the operator's Ed25519 key of the run
(``<run_dir>/operator/control.key``: directory 0700 and key 0600, owned by the operator's uid); the runtime verifies
the signature with ``<run_dir>/control.pub`` and trusts that file only when root owns it — or, in single-uid mode
(the runtime does not run under a service uid), when its own uid does. Commands that lift a brake or let something
happen (``resume``, ``approve``, ``reject``, ``assign_objective``, ``export_checkpoint``) are applied only when
signed; the brake (``pause``, ``stop``) and ``wake`` / ``approval_notice`` are applied from anyone who can write the
control file, and are recorded as unauthenticated. A signed command is applied at most once (its id).

Commands govern whether and what CAM may execute; none of them carries a way of pursuing the objective. The
objective itself can only be changed by assigning a task file, which passes the same validation as any task.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import signal
import stat
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ..contracts import Clock, PolicyEngine, RunFiles, WallClock, new_id
from .logs import atomic_write_json, atomic_write_text, read_json

COMMANDS = ("pause", "resume", "stop", "approve", "reject", "approval_notice", "assign_objective",
            "export_checkpoint", "wake")
PID_FILE = "runtime.pid"
#: commands applied only with a valid operator signature
PRIVILEGED = ("resume", "approve", "reject", "assign_objective", "export_checkpoint")
OPERATOR_DIR, OPERATOR_KEY, CONTROL_PUB = "operator", "control.key", "control.pub"


def _canon(cmd: Dict[str, Any]) -> bytes:
    body = {k: cmd.get(k) for k in ("id", "cmd", "by", "issued_at", "args")}
    return json.dumps(body, sort_keys=True, default=str, separators=(",", ":")).encode()


def _service_uids() -> set:
    try:
        from ..harness import SERVICE_USERS, uid_of, users_exist
        return {uid_of(r) for r in SERVICE_USERS} if users_exist() else set()
    except Exception:  # noqa: BLE001
        return set()


class OperatorKey:
    """The operator's command-signing key pair of one run directory (created on first use by the operator's uid)."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.dir = os.path.join(self.run_dir, OPERATOR_DIR)
        self.key_path = os.path.join(self.dir, OPERATOR_KEY)
        self.pub_path = os.path.join(self.run_dir, CONTROL_PUB)
        self._key: Optional[Ed25519PrivateKey] = None

    def _own_dir(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        try:
            os.mkdir(self.dir, 0o700)
        except FileExistsError:
            pass
        st = os.lstat(self.dir)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            raise PermissionError(f"{self.dir} is not a directory owned by this uid ({os.geteuid()}); it was not "
                                  "created by the operator — remove it as the operator")
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.chmod(self.dir, 0o700)

    def private_key(self) -> Ed25519PrivateKey:
        if self._key is not None:
            return self._key
        self._own_dir()
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            st = os.lstat(self.key_path)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) & 0o077:
                raise PermissionError(f"operator key {self.key_path} must be a 0600 file owned by uid {os.geteuid()}")
            with open(self.key_path, "rb") as f:
                key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(f.read().strip()))
        else:
            key = Ed25519PrivateKey.generate()
            raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                    serialization.NoEncryption())
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64encode(raw))
                f.flush()
                os.fsync(f.fileno())
        self._key = key
        self._publish(key.public_key())
        return key

    def _publish(self, pub: Ed25519PublicKey) -> None:
        raw = base64.b64encode(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        try:
            st = os.lstat(self.pub_path)
            with open(self.pub_path) as f:
                current = f.read().strip()
            if stat.S_ISREG(st.st_mode) and st.st_uid == os.geteuid() and current == raw \
                    and not stat.S_IMODE(st.st_mode) & 0o022:
                return
        except OSError:
            pass
        atomic_write_text(self.pub_path, raw + "\n", 0o644)        # replaces anything another uid put there

    def ensure(self) -> str:
        """Create the key pair now (``cam-live run`` does this before any service starts). Returns the pub path."""
        self.private_key()
        return self.pub_path

    def sign(self, cmd: Dict[str, Any]) -> str:
        return base64.b64encode(self.private_key().sign(_canon(cmd))).decode()


def trusted_public_key(run_dir: str) -> Tuple[Optional[Ed25519PublicKey], str]:
    """The operator's command key as the runtime may trust it, or (None, why not)."""
    path = os.path.join(run_dir, CONTROL_PUB)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, "no operator command key (control.pub) in the run directory"
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) & 0o022:
        return None, "control.pub is not a regular file writable only by its owner"
    me = os.geteuid()
    owners = {0} if me in _service_uids() else {0, me}
    if st.st_uid not in owners:
        return None, f"control.pub is owned by uid {st.st_uid}, which is not the operator"
    try:
        with open(path) as f:
            return Ed25519PublicKey.from_public_bytes(base64.b64decode(f.read().strip())), ""
    except (OSError, ValueError) as e:
        return None, f"control.pub is unreadable or malformed: {type(e).__name__}"


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


def _open_lock(path: str):
    """The command channel's lock file, opened without following a symlink (change request HX-6, G2-harness R3-M1):
    the run directory is writable by the learner's group, so ``open(path, "a+")`` by a root process could be
    redirected by a planted symlink to create or open any file. The parent is reached with ``safefs.open_dir``
    (no symlink in a directory another uid can write), the entry is opened relative to it with ``O_NOFOLLOW``, and
    anything but a regular file is refused. A process that may not write the lock opens it read-only (``flock``
    works on any open file)."""
    from ..harness import safefs
    d, name = os.path.split(os.path.abspath(path))
    dfd = safefs.open_dir(d)
    try:
        base = os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(name, base | os.O_RDWR | os.O_CREAT, 0o644, dir_fd=dfd)
        except PermissionError:
            fd = os.open(name, base | os.O_RDONLY, dir_fd=dfd)
    finally:
        os.close(dfd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise safefs.UnsafePath(f"{path}: not a regular file")
    return os.fdopen(fd, "rb")


class ControlChannel:
    """The control file. Writers (operator tools) append under a lock; the runtime only reads and verifies."""

    def __init__(self, run_dir: str, clock: Optional[Clock] = None):
        self.run_dir, self.clock = run_dir, clock or WallClock()
        self.path = os.path.join(run_dir, RunFiles.CONTROL)
        self._pub: Tuple[Any, Optional[Ed25519PublicKey], str] = (None, None, "")

    def authenticate(self, cmd: Dict[str, Any]) -> Tuple[bool, str]:
        """(signed by the operator, why not). The trusted key is re-read when control.pub changes."""
        path = os.path.join(self.run_dir, CONTROL_PUB)
        try:
            st = os.lstat(path)
            ident: Any = (st.st_ino, st.st_mtime_ns, st.st_size, st.st_uid, st.st_mode)
        except OSError:
            ident = None
        if ident != self._pub[0] or ident is None:
            key, why = trusted_public_key(self.run_dir)
            self._pub = (ident, key, why)
        _ident, key, why = self._pub
        sig = cmd.get("sig")
        if not sig:
            return False, "the command is not signed"
        if key is None:
            return False, why
        try:
            key.verify(base64.b64decode(str(sig)), _canon(cmd))
            return True, ""
        except (InvalidSignature, ValueError):
            return False, "the signature does not verify with the operator's key"

    def read(self) -> List[Dict[str, Any]]:
        doc = read_json(self.path, {}) or {}
        cmds = doc.get("commands", []) if isinstance(doc, dict) else []
        return [c for c in cmds if isinstance(c, dict) and c.get("cmd") in COMMANDS]

    def mtime(self) -> float:
        try:
            return os.stat(self.path).st_mtime_ns / 1e9
        except FileNotFoundError:
            return 0.0

    def send(self, cmd: str, by: str = "operator", wake: bool = True,
             signer: Optional[Callable[[Dict[str, Any]], str]] = None, **args: Any) -> str:
        """Append a command. Without ``signer`` it is unsigned (applied only if it is a brake or a wake-up)."""
        if cmd not in COMMANDS:
            raise ValueError(f"unknown command {cmd!r}; known: {', '.join(COMMANDS)}")
        os.makedirs(self.run_dir, exist_ok=True)
        cid = new_id("cmd")
        entry: Dict[str, Any] = {"id": cid, "cmd": cmd, "by": by, "issued_at": self.clock.now(), "args": args}
        if signer is not None:
            entry["sig"] = signer(entry)
        with _open_lock(self.path + ".lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                doc = read_json(self.path, {}) or {}
                cmds = list(doc.get("commands", [])) if isinstance(doc, dict) else []
                cmds.append(entry)
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
        self.key = OperatorKey(run_dir)

    def _send(self, cmd: str, wake: bool = True, **args: Any) -> str:
        return self.channel.send(cmd, self.by, wake=wake, signer=self.key.sign, **args)

    def pause(self, reason: str = "") -> str:
        return self._send("pause", reason=reason)

    def resume(self, wake: bool = True) -> str:
        return self._send("resume", wake=wake)

    def stop(self, reason: str = "", emergency: bool = False) -> str:
        if emergency and self.policy is not None:
            self.policy.emergency_stop(reason or "operator emergency stop", self.by)
        return self._send("stop", reason=reason, emergency=bool(emergency))

    def reset_stop(self) -> Any:
        """Release the emergency stop at the authorization service (the operator's handle only)."""
        if self.policy is None or not callable(getattr(self.policy, "reset_stop", None)):
            raise RuntimeError("no operator handle on an authorization service that can reset the emergency stop")
        return self.policy.reset_stop(self.by)

    def _resolve(self, approval_id: str, approve: bool, note: str) -> str:
        if self.policy is not None:
            decision = self.policy.resolve_approval(approval_id, approve, self.by, note)
            extra: Dict[str, Any] = {}
            to_dict = getattr(decision, "to_dict", None)
            if approve and getattr(decision, "verdict", "") == "allow" and getattr(decision, "grant", "") and callable(to_dict):
                # G2I-15: the grant the service issued (and reserved the action's budget for) when the operator
                # approved goes to the runtime with the signed notice; the runtime executes the action with it after
                # checking it, instead of asking the service again (a second grant would reserve the budget twice)
                extra["grant"] = to_dict()
            return self._send("approval_notice", approval_id=approval_id, approved=bool(approve), note=note,
                              verdict=getattr(decision, "verdict", ""), **extra)
        return self._send("approve" if approve else "reject", approval_id=approval_id, note=note)

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
        return self._send("assign_objective", path=task_path, sha256=file_sha256(task_path))

    def export_checkpoint(self, dest: str = "") -> str:
        return self._send("export_checkpoint", dest=os.path.abspath(dest) if dest else "")

    def wake(self) -> str:
        return self._send("wake")

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
