"""Authorization service process: ``PolicyEngine`` over ``cam_live.harness.rpc`` with peer-uid authorization.

    python -m cam_live.harness.auth.service --run-dir R --socket R/sock/auth.sock --policy <yaml>

Who may call what (the peer uid comes from SO_PEERCRED, i.e. from the kernel, not from the request):

=================  ==========================================================================================
learner uid        authorize, check_grant (look only), approvals, stopped, paused, limits_state, audit_head,
                   emergency_stop / pause (anyone may pull the brake; nobody but the operator releases it)
executor uid       check_grant (consumes), check_primitive, record, secret_get, scrub, scrub_obj, + the read
                   calls and the brake
verifier uid       the read calls, scrub, and the brake
operator (root)    everything, and the only caller of resolve_approval, reset_stop, resume, revoke,
                   reload_policy, register_task, register_live_authorization, put_secret, secret_names,
                   audit_tail, verify_audit
anyone else        nothing (the sandbox uid, for instance)
=================  ==========================================================================================

Every response to a caller that is not the executor or the operator passes through the secret scrubber.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from ... import contracts as C
from .. import SERVICE_GROUP, uid_of, users_exist
from .. import rpc
from . import landlock
from .policy import PolicyService

LEARNER, EXECUTOR, VERIFIER, OPERATOR = "learner", "executor", "verifier", "operator"
_READ = (LEARNER, EXECUTOR, VERIFIER)

#: method -> roles allowed besides the operator
METHODS: Dict[str, tuple] = {
    "authorize": (LEARNER,),
    "check_grant": (LEARNER, EXECUTOR),
    "approvals": _READ,
    "stopped": _READ,
    "paused": _READ,
    "limits_state": _READ,
    "audit_head": _READ,
    "emergency_stop": _READ,
    "pause": _READ,
    "check_primitive": (EXECUTOR,),
    "record": (EXECUTOR,),
    "secret_get": (EXECUTOR,),
    "scrub": (EXECUTOR, VERIFIER),
    "scrub_obj": (EXECUTOR, VERIFIER),
    "resolve_approval": (),
    "reset_stop": (),
    "resume": (),
    "revoke": (),
    "reload_policy": (),
    "register_task": (),
    "register_live_authorization": (),
    "put_secret": (),
    "secret_names": (),
    "audit_tail": (),
    "verify_audit": (),
}
#: methods whose ``by`` argument is replaced by the kernel-attested caller
_BY_ARG = {"emergency_stop": 1, "pause": 1, "resolve_approval": 2, "revoke": 1, "reset_stop": 0, "resume": 0,
           "reload_policy": 0, "register_task": 1, "register_live_authorization": 1, "put_secret": 4}


class AuthService:
    """RPC target. Wraps a :class:`PolicyService` and checks the caller's uid for every method."""

    def __init__(self, policy: PolicyService, roles: Dict[str, Iterable[int]],
                 uid_getter: Callable[[], Optional[int]] = rpc.caller_uid):
        self.policy = policy
        self.roles: Dict[str, Set[int]] = {k: set(v) for k, v in roles.items()}
        self._uid = uid_getter
        policy.caller = lambda: f"uid={self._uid()}"

    def roles_of(self, uid: Optional[int]) -> List[str]:
        return sorted(r for r, uids in self.roles.items() if uid is not None and uid in uids)

    def _guard(self, method: str) -> List[str]:
        uid = self._uid()
        mine = self.roles_of(uid)
        if OPERATOR in mine or any(r in METHODS[method] for r in mine):
            return mine
        self.policy._log("rpc_denied", method=method, roles=mine)
        raise PermissionError(f"uid {uid} ({', '.join(mine) or 'no role'}) may not call {method}")

    def _out(self, value: Any, mine: List[str]) -> Any:
        if OPERATOR in mine or EXECUTOR in mine:
            return value
        if isinstance(value, C.Record):
            return type(value).from_dict(self.policy.scrub_obj(value.to_dict()))
        return self.policy.scrub_obj(value)

    def _by(self, method: str, args: list, kwargs: dict, mine: List[str]) -> None:
        tag = f"[uid={self._uid()} {'/'.join(mine)}]"
        if "by" in kwargs:
            kwargs["by"] = f"{kwargs['by']} {tag}"
        elif method in _BY_ARG and len(args) > _BY_ARG[method]:
            args[_BY_ARG[method]] = f"{args[_BY_ARG[method]]} {tag}"
        elif method in _BY_ARG:
            kwargs["by"] = tag

    def _call(self, method: str, args: tuple, kwargs: dict) -> Any:
        mine = self._guard(method)
        a, k = list(args), dict(kwargs)
        self._by(method, a, k, mine)
        if method == "check_grant" and OPERATOR not in mine and EXECUTOR not in mine:
            a, k = a[:2], {"consume": False}      # the learner may look at a grant, never use it up
        out = getattr(self.policy, method)(*a, **k)
        if method == "verify_audit":
            out = {"ok": out.ok, "entries": out.entries, "head_hash": out.head_hash, "errors": out.errors}
        return self._out(out, mine)


def _make(method: str) -> Callable[..., Any]:
    def call(self: AuthService, *args: Any, **kwargs: Any) -> Any:
        return self._call(method, args, kwargs)
    call.__name__ = method
    return call


for _m in METHODS:
    setattr(AuthService, _m, _make(_m))


def default_roles(executor_roles: Iterable[str] = ("browser",), extra_executor_uids: Iterable[int] = (),
                  single_uid: Optional[bool] = None) -> Dict[str, Set[int]]:
    """Role -> uids from the service users. In single-uid development mode (no service users, or asked for) every
    role is the current uid — one trust domain, no isolation claimed."""
    if single_uid is None:
        single_uid = not users_exist()
    if single_uid:
        return {r: {os.getuid(), 0} for r in (LEARNER, EXECUTOR, VERIFIER, OPERATOR)}
    return {LEARNER: {uid_of("learner")}, EXECUTOR: {uid_of(r) for r in executor_roles} | set(extra_executor_uids),
            VERIFIER: {uid_of("verifier")}, OPERATOR: {0}}


def serve(run_dir: str, socket_path: str, policy_path: Optional[str] = None, *, roles: Optional[Dict[str, Iterable[int]]] = None,
          fsync: bool = True, group: Optional[str] = None) -> "rpc.RpcServer":
    """Build the service and start serving (background thread). Returns the server; ``server.target`` is the
    :class:`AuthService`."""
    policy = PolicyService(run_dir, policy_path, fsync=fsync)
    target = AuthService(policy, roles if roles is not None else default_roles())
    return rpc.RpcServer(socket_path, target, METHODS.keys(), allowed_uids=None, mode=0o660, group=group, name="auth").start()


def _harden() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(38, 1, 0, 0, 0)       # PR_SET_NO_NEW_PRIVS
        libc.prctl(4, 0, 0, 0, 0)        # PR_SET_DUMPABLE = 0: no core dumps or ptrace attach holding secrets
    except OSError:
        pass
    os.umask(0o077)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.auth.service")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--executor-role", action="append", default=None,
                    help="service role(s) whose uid is the executor (default: browser)")
    ap.add_argument("--executor-uid", action="append", type=int, default=[])
    ap.add_argument("--no-fsync", action="store_true")
    ap.add_argument("--single-uid", action="store_true", help="development mode: every role is this process's uid")
    ap.add_argument("--no-landlock", action="store_true", help="do not confine this process's file writes")
    a = ap.parse_args(argv)
    _harden()
    single = a.single_uid or not users_exist()
    roles = default_roles(a.executor_role or ("browser",), a.executor_uid, single_uid=single)
    server = serve(a.run_dir, a.socket, a.policy, roles=roles, fsync=not a.no_fsync,
                   group=None if single else SERVICE_GROUP)
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    pol: PolicyService = server.target.policy
    confined = "off"
    if not a.no_landlock:
        # from here on the kernel lets this process write only to its own state, the vault, the audit log and the
        # socket directory, whatever a request manages to make it do
        try:
            ok = landlock.restrict_writes([pol.auth_dir, pol.vault_dir, os.path.dirname(os.path.abspath(a.socket))],
                                          [pol.audit.path])
            confined = "enforced" if ok else "unavailable"
        except OSError as e:
            confined = f"failed ({e})"
    pol._log("service_confinement", landlock=confined, no_new_privs=True)
    print(f"auth service ready on {a.socket} (landlock {confined}; policy {'ok' if pol.policy else 'INVALID: ' + pol.policy_error}, "
          f"roles {{{', '.join(f'{k}: {sorted(v)}' for k, v in sorted(server.target.roles.items()))}}})", flush=True)
    done.wait()
    pol.caller = lambda: "service"
    pol._log("service_stop")
    server.stop()
    pol.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
