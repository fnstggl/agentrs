"""cam_live.harness.sandbox_host — the generated-code sandbox in the separated deployment (CU-37 / CU-01;
G2-harness D2).

In the deployment ``cam-live run`` starts, the environment service runs as the browser uid (``cambrowser``). A
process that is not root cannot drop generated code to the sandbox uid (``camsbx``), so ``SandboxExecutor`` in that
process refuses to run code (``sandbox.py``: "not root"). This module splits the code modality across two
processes without giving the environment service root:

* **sandbox host** (``python -m cam_live.harness.sandbox_host``), started by the launcher as root next to the
  environment service. It holds the real :class:`~cam_live.harness.sandbox.SandboxExecutor` (workspace
  ``<run>/sandbox``), so every program still runs under ``camsbx`` with no_new_privs, Landlock, seccomp (no socket
  creation), rlimits and the per-job monitor. It does nothing else: no policy, no secrets, no network of its own.
  It accepts connections only from the environment service's uid (SO_PEERCRED) on ``<run>/sock/sandbox.sock``.
* :class:`RemoteSandboxExecutor` — the ``code`` modality inside the environment service. ``execute`` sends the
  action to the host and serves the host's callbacks over the same connection: every primitive the sandbox wants
  checked (each program run, each file operation, each outbound connection of the egress broker) is answered by the
  environment's own ``ExecContext.check`` (so it is judged by the authorization service against the running grant
  and logged by the environment exactly as in-process), observations go to the environment's sink, and the
  environment's cancellation is forwarded. The host never sees a secret.

Wire format: rpc frames (4-byte length + JSON), contract records via ``rpc.to_wire`` / ``rpc.from_wire``.
"""
from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .. import contracts as c
from . import rpc

SOCK_SANDBOX = "sandbox.sock"
#: environment variable through which the launcher tells the environment service where the host listens
ENV_SOCKET = "CAM_LIVE_SANDBOX_SOCKET"
PROTOCOL = "camlive-sandbox-host/1"


class SandboxHost:
    """Serves one :class:`SandboxExecutor` to the environment service's uid."""

    def __init__(self, run_dir: str, socket_path: str, client_uids: Optional[List[int]] = None,
                 workspace: str = "", executor: Any = None):
        from .sandbox import SandboxExecutor
        self.run_dir = os.path.abspath(run_dir)
        self.path = socket_path
        self.client_uids = set(client_uids or []) | {0}
        self.workspace = workspace or os.path.join(self.run_dir, "sandbox")
        self.executor = executor or SandboxExecutor(workspace=self.workspace)
        from . import safefs
        self.jobs_dir = os.path.join(self.run_dir, "sandbox_host")
        # the run directory is writable by the learner's group: the host's own directory is created without
        # following a planted entry and owned by root alone (G2-harness R3-M1)
        safefs.ensure_dir(self.jobs_dir, 0o700, 0 if os.geteuid() == 0 else None, 0 if os.geteuid() == 0 else None)
        if not os.path.isdir(os.path.dirname(socket_path)):
            os.makedirs(os.path.dirname(socket_path), exist_ok=True)
        if os.path.lexists(socket_path):
            os.unlink(socket_path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_mask = os.umask(0o117)               # the socket is created 0660: no chmod by path afterwards
        try:
            self._sock.bind(socket_path)
        finally:
            os.umask(old_mask)
        try:
            import grp
            from . import SERVICE_GROUP
            safefs.chown_chmod_nofollow(socket_path, None, grp.getgrnam(SERVICE_GROUP).gr_gid, None)
        except (KeyError, PermissionError, ImportError):
            pass
        self._sock.listen(16)
        self._stop = threading.Event()

    # -- serving -----------------------------------------------------------------------------------------------
    def serve_forever(self) -> None:
        self._sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        finally:
            if os.path.lexists(self.path):
                os.unlink(self.path)
            try:
                self.executor.close()
            except Exception:  # noqa: BLE001
                pass

    def _handle(self, conn: socket.socket) -> None:
        try:
            _pid, uid, _gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                                 struct.calcsize("3i")))
            if uid not in self.client_uids:          # refused before anything is read from the peer
                rpc._send(conn, {"error": f"uid {uid} may not use the sandbox host"})
                return
            conn.settimeout(30.0)                    # a client that connects and sends nothing does not hold a thread
            req = rpc._recv(conn)
            conn.settimeout(None)
            if req is None:
                return
            op = req.get("op")
            if op == "ping":
                rpc._send(conn, {"ok": True, "protocol": PROTOCOL, "pid": os.getpid(), "uid": os.geteuid()})
            elif op == "tools":
                rpc._send(conn, {"ok": True, "tools": rpc.to_wire(self.executor.tools())})
            elif op == "capabilities":
                caps = dict(self.executor.capabilities())
                caps["host"] = {"protocol": PROTOCOL, "pid": os.getpid(), "uid": os.geteuid()}
                rpc._send(conn, {"ok": True, "capabilities": rpc.to_wire(caps)})
            elif op == "validate":
                rpc._send(conn, {"ok": True, "errors": self.executor.validate(rpc.from_wire(req["action"]))})
            elif op == "execute":
                self._execute(conn, req)
            else:
                rpc._send(conn, {"error": f"unknown operation {op!r}"})
        except (ConnectionError, OSError, ValueError, socket.timeout):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _execute(self, conn: socket.socket, req: Dict[str, Any]) -> None:
        action: c.ActionSpec = rpc.from_wire(req["action"])
        lock = threading.Lock()
        waiting: Dict[int, Any] = {}
        answers: Dict[int, Any] = {}
        cond = threading.Condition()
        cancel = threading.Event()
        closed = threading.Event()
        counter = [0]

        def send(obj: Dict[str, Any]) -> bool:
            if closed.is_set():
                return False
            with lock:
                try:
                    rpc._send(conn, obj)
                    return True
                except (OSError, ValueError):
                    closed.set()
                    return False

        def check(op: c.PrimitiveOp, _action: Optional[c.ActionSpec] = None) -> c.AuthorizationDecision:
            with cond:
                counter[0] += 1
                n = counter[0]
                waiting[n] = True
            if not send({"check": rpc.to_wire(op), "id": n}):
                return c.AuthorizationDecision(c.AuthVerdict.DENY, reasons=["the environment service is gone"])
            with cond:
                end = time.monotonic() + 120.0
                while n not in answers and not closed.is_set() and time.monotonic() < end:
                    cond.wait(0.2)
                dec = answers.pop(n, None)
                waiting.pop(n, None)
            if not isinstance(dec, c.AuthorizationDecision):
                return c.AuthorizationDecision(c.AuthVerdict.DENY, reasons=["no authorization answer (fail closed)"])
            return dec

        def sink(obs: c.Observation) -> None:
            send({"obs": rpc.to_wire(obs)})

        def reader() -> None:
            while not closed.is_set():
                try:
                    msg = rpc._recv(conn)
                except (OSError, ValueError):
                    msg = None
                if msg is None:
                    closed.set()
                    cancel.set()
                    break
                if msg.get("cancel"):
                    cancel.set()
                if "id" in msg and "decision" in msg:
                    with cond:
                        answers[int(msg["id"])] = rpc.from_wire(msg["decision"])
                        cond.notify_all()
            with cond:
                cond.notify_all()

        deadline = float(req.get("deadline") or 0.0)
        ctx = c.ExecContext(clock=c.WallClock(), check=check, obs_sink=sink, run_dir=self.jobs_dir,
                            deadline=deadline, cancel=cancel, secrets=None)
        ctx.authorization = rpc.from_wire(req.get("authorization"))     # type: ignore[attr-defined]
        ctx.workspace = self.workspace                                     # type: ignore[attr-defined]
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            res = self.executor.execute(action, ctx)
        except Exception as e:  # noqa: BLE001 — an executor bug is a failed action
            res = c.ActionResult(action_id=action.action_id, status=c.ResultStatus.FAILED,
                                 failure_class=c.FailureClass.UNKNOWN,
                                 failure_detail=f"sandbox host error: {type(e).__name__}: {e}")
        send({"result": rpc.to_wire(res)})
        closed.set()


class RemoteSandboxExecutor:
    """The ``code`` modality of an environment service that is not root: programs run in the sandbox host."""

    modality = c.Modality.CODE

    def __init__(self, socket_path: str, connect_timeout_s: float = 10.0):
        self.path = socket_path
        self.connect_timeout_s = connect_timeout_s
        self._tools: Optional[List[c.ToolDescriptor]] = None
        self._tools_at = 0.0
        self._why = ""

    # -- transport ---------------------------------------------------------------------------------------------
    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout_s)
        s.connect(self.path)
        return s

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        s = self._connect()
        try:
            rpc._send(s, req)
            out = rpc._recv(s)
        finally:
            s.close()
        if not isinstance(out, dict) or out.get("error"):
            raise RuntimeError((out or {}).get("error") or "no answer from the sandbox host")
        return out

    def ping(self) -> Dict[str, Any]:
        return self._call({"op": "ping"})

    # -- ModalityExecutor --------------------------------------------------------------------------------------
    def tools(self) -> List[c.ToolDescriptor]:
        now = time.monotonic()
        if self._tools is None or now - self._tools_at > 5.0:
            try:
                self._tools = list(rpc.from_wire(self._call({"op": "tools"})["tools"]))
                self._why = ""
            except Exception as e:  # noqa: BLE001
                self._why = f"the sandbox host at {self.path} is not reachable: {type(e).__name__}: {e}"
                self._tools = [] if self._tools is None else self._tools
                for t in self._tools:
                    t.available, t.unavailable_reason = False, self._why
            self._tools_at = now
        return list(self._tools or [])

    def capabilities(self) -> Dict[str, Any]:
        try:
            caps = dict(rpc.from_wire(self._call({"op": "capabilities"})["capabilities"]))
        except Exception as e:  # noqa: BLE001
            return {"modality": self.modality, "available": False,
                    "unavailable_reason": f"the sandbox host is not reachable: {type(e).__name__}: {e}",
                    "remote": {"socket": self.path}}
        caps["remote"] = {"socket": self.path, "environment_uid": os.geteuid()}
        return caps

    def validate(self, action: c.ActionSpec) -> List[str]:
        try:
            return list(self._call({"op": "validate", "action": rpc.to_wire(action)})["errors"])
        except Exception as e:  # noqa: BLE001
            return [f"the sandbox host is not reachable: {type(e).__name__}: {e}"]

    def close(self) -> None:
        return None

    def execute(self, action: c.ActionSpec, ctx: c.ExecContext) -> c.ActionResult:
        t0 = ctx.clock.now()
        try:
            s = self._connect()
        except OSError as e:
            return c.ActionResult(action_id=action.action_id, status=c.ResultStatus.FAILED, started=t0,
                                  ended=ctx.clock.now(), failure_class=c.FailureClass.RESOURCE_LIMIT,
                                  failure_detail=f"sandbox host not reachable: {e}")
        s.settimeout(None)
        cancel_sent = False
        try:
            rpc._send(s, {"op": "execute", "action": rpc.to_wire(action),
                          "deadline": float(getattr(ctx, "deadline", 0.0) or 0.0),
                          "authorization": rpc.to_wire(getattr(ctx, "authorization", None))})
            while True:
                cancel = getattr(ctx, "cancel", None)
                if cancel is not None and cancel.is_set() and not cancel_sent:
                    rpc._send(s, {"cancel": True})
                    cancel_sent = True
                r, _w, _x = select.select([s], [], [], 0.25)
                if not r:
                    continue
                msg = rpc._recv(s)
                if msg is None:
                    return c.ActionResult(action_id=action.action_id, status=c.ResultStatus.FAILED, started=t0,
                                          ended=ctx.clock.now(), failure_class=c.FailureClass.UNKNOWN,
                                          failure_detail="the sandbox host closed the connection")
                if "check" in msg:
                    dec = ctx.check(rpc.from_wire(msg["check"]), action)
                    rpc._send(s, {"id": msg["id"], "decision": rpc.to_wire(dec)})
                elif "obs" in msg:
                    sink = getattr(ctx, "obs_sink", None)
                    if callable(sink):
                        sink(rpc.from_wire(msg["obs"]))
                elif "result" in msg:
                    res = rpc.from_wire(msg["result"])
                    if isinstance(res, c.ActionResult):
                        return res
                    break
                elif msg.get("error"):
                    return c.ActionResult(action_id=action.action_id, status=c.ResultStatus.DENIED, started=t0,
                                          ended=ctx.clock.now(), failure_class=c.FailureClass.MISSING_PERMISSION,
                                          failure_detail=str(msg["error"]))
        except (OSError, ValueError) as e:
            return c.ActionResult(action_id=action.action_id, status=c.ResultStatus.FAILED, started=t0,
                                  ended=ctx.clock.now(), failure_class=c.FailureClass.UNKNOWN,
                                  failure_detail=f"sandbox host connection failed: {e}")
        finally:
            s.close()
        return c.ActionResult(action_id=action.action_id, status=c.ResultStatus.FAILED, started=t0,
                              ended=ctx.clock.now(), failure_class=c.FailureClass.UNKNOWN,
                              failure_detail="the sandbox host returned no result")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.sandbox_host", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--client-uid", type=int, action="append", default=[],
                    help="uid allowed to connect (the environment service); root is always allowed")
    ap.add_argument("--workspace", default="")
    a = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("the sandbox host must run as root (it drops every program to the sandbox uid)", file=sys.stderr)
        return 2
    host = SandboxHost(a.run_dir, a.socket, a.client_uid, workspace=a.workspace)
    if not host.executor.sandbox.available:
        print(f"sandbox unavailable: {host.executor.sandbox.unavailable_reason}", file=sys.stderr)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    th = threading.Thread(target=host.serve_forever, daemon=True)
    th.start()
    while not stop.wait(0.5):
        pass
    host.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
