"""cam_live.harness.rpc — minimal versioned JSON-RPC over unix-domain sockets. MASTER-OWNED.

Used for every cross-process boundary (learner <-> computer environment "ExecRPC", <-> authorization service
"AuthRPC", <-> verifier "VerifyRPC"). Frames are 4-byte big-endian length + UTF-8 JSON. Contract records
(``contracts.Record`` subclasses) round-trip automatically. The server authenticates the *peer uid* with
SO_PEERCRED and exposes only an explicit method allowlist.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import socket
import struct
import threading
import traceback
from typing import Any, Callable, Dict, Iterable, Optional, Set

from .. import contracts as _c

RPC_VERSION = "1.0"
_MAX_FRAME = 64 * 1024 * 1024
_local = threading.local()


def _record_types() -> Dict[str, type]:
    out, stack = {}, [_c.Record]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            out[sub.__name__] = sub
            stack.append(sub)
    return out


def to_wire(o: Any) -> Any:
    if isinstance(o, _c.Record):
        return {"__rec__": type(o).__name__, "d": to_wire({f.name: getattr(o, f.name) for f in dataclasses.fields(o)})}
    if isinstance(o, (bytes, bytearray)):
        return {"__b64__": base64.b64encode(bytes(o)).decode()}
    if isinstance(o, dict):
        return {str(k): to_wire(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_wire(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def from_wire(o: Any, _types: Optional[Dict[str, type]] = None) -> Any:
    types = _types or _record_types()
    if isinstance(o, dict):
        if "__rec__" in o and o["__rec__"] in types:
            cls = types[o["__rec__"]]
            d = from_wire(o["d"], types)
            return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})
        if "__b64__" in o and len(o) == 1:
            return base64.b64decode(o["__b64__"])
        return {k: from_wire(v, types) for k, v in o.items()}
    if isinstance(o, list):
        return [from_wire(v, types) for v in o]
    return o


def _send(sock: socket.socket, obj: Any) -> None:
    data = json.dumps(obj).encode()
    if len(data) > _MAX_FRAME:
        raise ValueError("rpc frame too large")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv(sock: socket.socket) -> Optional[Any]:
    head = b""
    while len(head) < 4:
        chunk = sock.recv(4 - len(head))
        if not chunk:
            return None
        head += chunk
    (n,) = struct.unpack(">I", head)
    if n > _MAX_FRAME:
        raise ValueError("rpc frame too large")
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return json.loads(bytes(buf).decode())


class RemoteError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message = kind, message


def caller_uid() -> Optional[int]:
    """uid of the peer whose request the current server thread is handling (None outside a request)."""
    return getattr(_local, "uid", None)


class RpcServer:
    def __init__(self, path: str, target: Any, methods: Iterable[str], *, allowed_uids: Optional[Set[int]] = None,
                 mode: int = 0o660, group: Optional[str] = None, name: str = "rpc"):
        self.path, self.target, self.methods, self.name = path, target, set(methods), name
        self.allowed_uids = None if allowed_uids is None else set(allowed_uids) | {0}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        os.chmod(path, mode)
        if group is not None:
            import grp
            try:
                os.chown(path, -1, grp.getgrnam(group).gr_gid)
            except (KeyError, PermissionError):
                pass
        self._sock.listen(64)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _handle(self, conn: socket.socket) -> None:
        try:
            cred = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", cred)
            _local.uid = uid
            while not self._stop.is_set():
                req = _recv(conn)
                if req is None:
                    return
                rid = req.get("id")
                try:
                    if self.allowed_uids is not None and uid not in self.allowed_uids:
                        raise PermissionError(f"uid {uid} may not call {self.name}")
                    m = req.get("method", "")
                    if m == "__ping__":
                        result = {"rpc": RPC_VERSION, "name": self.name, "methods": sorted(self.methods)}
                    elif m not in self.methods:
                        raise PermissionError(f"method {m!r} is not exposed by {self.name}")
                    else:
                        fn: Callable = getattr(self.target, m)
                        result = fn(*from_wire(req.get("args", [])), **from_wire(req.get("kwargs", {})))
                    _send(conn, {"id": rid, "ok": True, "result": to_wire(result)})
                except Exception as e:  # noqa: BLE001 — every failure is returned to the caller, never swallowed
                    _send(conn, {"id": rid, "ok": False, "error": {"kind": type(e).__name__, "message": str(e),
                                                                   "trace": traceback.format_exc(limit=6)}})
        except (ConnectionError, OSError):
            return
        finally:
            _local.uid = None
            conn.close()

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

    def start(self) -> "RpcServer":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True, name=f"rpc-{self.name}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        finally:
            if os.path.exists(self.path):
                os.unlink(self.path)


class RpcClient:
    """Thread-safe client. ``client.call("method", ...)`` or ``client.method(...)``."""

    def __init__(self, path: str, timeout: float = 600.0):
        self.path, self.timeout = path, timeout
        self._lock, self._sock, self._n = threading.Lock(), None, 0

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.path)
        return s

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    self._n += 1
                    _send(self._sock, {"id": self._n, "method": method, "args": to_wire(list(args)),
                                       "kwargs": to_wire(kwargs)})
                    resp = _recv(self._sock)
                    if resp is None:
                        raise ConnectionError("connection closed")
                    break
                except (ConnectionError, BrokenPipeError, FileNotFoundError):
                    self.close_socket()
                    if attempt:
                        raise
        if not resp.get("ok"):
            err = resp.get("error", {})
            raise RemoteError(err.get("kind", "Error"), err.get("message", ""))
        return from_wire(resp.get("result"))

    def close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def close(self) -> None:
        with self._lock:
            self.close_socket()

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.call(name, *a, **k)
