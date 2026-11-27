"""Client side of the authorization service: a ``contracts.PolicyEngine`` that talks to the service socket.

    from cam_live.harness.auth.client import connect
    policy = connect(run_dir)                 # learner / runtime: authorize, approvals, stopped, limits_state
    ctx.secrets = policy.secrets              # executor: named secret for a granted action, scrub

In separated mode the client checks (SO_PEERCRED) that the process listening on the socket really runs as the
authorization uid before sending anything.
"""
from __future__ import annotations

import os
import socket
import struct
from typing import Any, Dict, List, Optional, Set

from ... import contracts as C
from .. import SOCK_AUTH, uid_of, users_exist
from .. import rpc
from .vault import ExecutorSecrets, SecretError


class _VerifiedRpcClient(rpc.RpcClient):
    def __init__(self, path: str, timeout: float, server_uids: Optional[Set[int]]):
        super().__init__(path, timeout)
        self._server_uids = server_uids

    def _connect(self) -> socket.socket:
        s = super()._connect()
        if self._server_uids is not None:
            cred = s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", cred)
            if uid not in self._server_uids:
                s.close()
                raise PermissionError(f"{self.path} is served by uid {uid}, not by the authorization service")
        return s


class AuthClient:
    """``contracts.PolicyEngine`` over RPC, plus the operator and executor extras of ``PolicyService``."""

    def __init__(self, socket_path: str, timeout: float = 60.0, server_uids: Optional[Set[int]] = None):
        self._rpc = _VerifiedRpcClient(socket_path, timeout, server_uids)

    # -- PolicyEngine ----------------------------------------------------------------------------------------
    def authorize(self, action: C.ActionSpec, task: C.TaskSpec, context: Optional[Dict[str, Any]] = None) -> C.AuthorizationDecision:
        return self._rpc.call("authorize", action, task, context or {})

    def check_grant(self, action: C.ActionSpec, decision: C.AuthorizationDecision, consume: bool = True) -> bool:
        return bool(self._rpc.call("check_grant", action, decision, consume))

    def check_primitive(self, op: C.PrimitiveOp, action: C.ActionSpec) -> C.AuthorizationDecision:
        return self._rpc.call("check_primitive", op, action)

    def record(self, action: C.ActionSpec, result: C.ActionResult) -> None:
        self._rpc.call("record", action, result)

    def approvals(self) -> List[Dict[str, Any]]:
        return self._rpc.call("approvals")

    def resolve_approval(self, approval_id: str, approve: bool, by: str, note: str = "") -> C.AuthorizationDecision:
        return self._rpc.call("resolve_approval", approval_id, approve, by, note)

    def emergency_stop(self, reason: str, by: str) -> None:
        self._rpc.call("emergency_stop", reason, by)

    def stopped(self) -> bool:
        return bool(self._rpc.call("stopped"))

    def revoke(self, scope: str, by: str) -> None:
        self._rpc.call("revoke", scope, by)

    def limits_state(self) -> Dict[str, Any]:
        return self._rpc.call("limits_state")

    # -- extras ----------------------------------------------------------------------------------------------
    def paused(self) -> bool:
        return bool(self._rpc.call("paused"))

    def pause(self, reason: str, by: str) -> None:
        self._rpc.call("pause", reason, by)

    def resume(self, by: str) -> bool:
        return bool(self._rpc.call("resume", by))

    def reset_stop(self, by: str) -> bool:
        return bool(self._rpc.call("reset_stop", by))

    def reload_policy(self, by: str, path: Optional[str] = None) -> Dict[str, Any]:
        return self._rpc.call("reload_policy", by, path)

    def register_task(self, task: C.TaskSpec, by: str) -> Dict[str, Any]:
        return self._rpc.call("register_task", task, by)

    def register_live_authorization(self, record: Dict[str, Any], by: str) -> Dict[str, Any]:
        return self._rpc.call("register_live_authorization", record, by)

    def register_local_service(self, address: str, by: str, note: str = "") -> Dict[str, Any]:
        return self._rpc.call("register_local_service", address, by, note)

    def unregister_local_service(self, address: str, by: str) -> bool:
        return bool(self._rpc.call("unregister_local_service", address, by))

    def local_services(self) -> List[Dict[str, Any]]:
        return self._rpc.call("local_services")

    def put_secret(self, name: str, value: str, destinations: List[str], scopes: List[str] = (), by: str = "operator",
                   note: str = "") -> Dict[str, Any]:
        return self._rpc.call("put_secret", name, value, list(destinations), list(scopes), by, note)

    def secret_names(self) -> List[Dict[str, Any]]:
        return self._rpc.call("secret_names")

    def audit_head(self) -> Dict[str, Any]:
        return self._rpc.call("audit_head")

    def audit_tail(self, n: int = 50) -> List[Dict[str, Any]]:
        return self._rpc.call("audit_tail", n)

    def verify_audit(self) -> Dict[str, Any]:
        return self._rpc.call("verify_audit")

    def secret_get(self, name: str, action: C.ActionSpec, decision: C.AuthorizationDecision, destination: str) -> str:
        try:
            return self._rpc.call("secret_get", name, action, decision, destination)
        except rpc.RemoteError as e:
            if e.kind in ("SecretError", "PermissionError"):
                raise SecretError(e.message) from None
            raise

    def scrub(self, text: str) -> str:
        return self._rpc.call("scrub", text)

    def scrub_obj(self, obj: Any) -> Any:
        return self._rpc.call("scrub_obj", obj)

    @property
    def secrets(self) -> ExecutorSecrets:
        """Handle for ``ExecContext.secrets`` (works only when this process runs as the executor uid)."""
        return ExecutorSecrets(self.secret_get, self.scrub, self.scrub_obj)

    def ping(self) -> Dict[str, Any]:
        return self._rpc.call("__ping__")

    def close(self) -> None:
        self._rpc.close()


def connect(run_dir: str, timeout: float = 60.0, verify_server: Optional[bool] = None) -> AuthClient:
    """Client for ``<run_dir>/sock/auth.sock``. With the service users present the server's uid is verified."""
    path = os.path.join(run_dir, "sock", SOCK_AUTH)
    check = users_exist() if verify_server is None else verify_server
    # acceptable servers: the authorization uid, root (the operator), or this very uid (single-uid mode)
    return AuthClient(path, timeout, {uid_of("auth"), 0, os.getuid()} if check else None)
