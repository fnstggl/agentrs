"""cam_live.harness.service — the computer environment as a separate process, and its clients. Workstream EB.

Server::

    python -m cam_live.harness.service --run-dir R --socket R/sock/exec.sock --auth-socket R/sock/auth.sock

exposes a :class:`~cam_live.harness.env.ComputerEnvironment` over ``cam_live.harness.rpc`` as **ExecRPC v1**: the
``contracts.EnvironmentAdapter`` methods plus ``hello`` (version negotiation), ``ping``, ``cancel_action`` and
``close``. The process authorizes everything through the authorization service on ``--auth-socket``
(:class:`RemotePolicy`); it has no policy of its own and fails closed when that service cannot be reached.

Clients: :class:`RemoteEnvironment` implements ``contracts.EnvironmentAdapter`` over the socket (what the learner's
runtime holds); :class:`RemotePolicy` implements ``contracts.PolicyEngine`` over the authorization socket.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import (CONTRACT_VERSION, EXEC_INTERFACE_VERSION, ActionResult, ActionSpec, AuthorizationDecision,
                         AuthVerdict, Observation, OutcomeObservation, OutcomeQuery, PrimitiveOp, Quantity, TaskSpec,
                         ToolDescriptor)
from . import SERVICE_GROUP, uid_of
from .rpc import RPC_VERSION, RemoteError, RpcClient, RpcServer

#: ExecRPC protocol versions this build can speak, oldest first
EXEC_RPC_VERSIONS: Tuple[str, ...] = ("1.0",)
ADAPTER_METHODS = ("observe", "discover_tools", "validate_action", "execute", "poll_events", "get_resource_state",
                   "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure")
EXEC_METHODS = ADAPTER_METHODS + ("hello", "ping", "close", "cancel_action")
POLICY_METHODS = ("authorize", "check_grant", "check_primitive", "record", "approvals", "resolve_approval",
                  "emergency_stop", "stopped", "revoke", "limits_state")


class VersionMismatch(RuntimeError):
    pass


def negotiate(offered: Sequence[str], supported: Sequence[str] = EXEC_RPC_VERSIONS) -> str:
    """Highest protocol version both sides speak (compared numerically)."""
    common = [v for v in offered if v in supported]
    if not common:
        raise VersionMismatch(f"no common ExecRPC version: client offers {list(offered)}, server speaks {list(supported)}")
    return max(common, key=lambda v: tuple(int(x) for x in v.split(".")))


class ExecService:
    """Server-side object whose allow-listed methods are ExecRPC v1."""

    def __init__(self, env: Any, on_close: Optional[Callable[[], None]] = None):
        self.env, self._on_close, self._t0 = env, on_close, time.time()

    def hello(self, versions: Sequence[str], client: str = "") -> Dict[str, Any]:
        return {"version": negotiate(list(versions)), "interface_version": self.env.interface_version,
                "contract_version": CONTRACT_VERSION, "rpc": RPC_VERSION, "server": "cam_live.harness.service",
                "pid": os.getpid(), "methods": sorted(EXEC_METHODS)}

    def ping(self) -> Dict[str, Any]:
        return {"ok": True, "interface_version": self.env.interface_version, "uptime_s": time.time() - self._t0,
                "pid": os.getpid()}

    def close(self) -> Dict[str, Any]:
        self.env.close()
        if self._on_close is not None:
            threading.Timer(0.2, self._on_close).start()      # answer first, then stop serving
        return {"closed": True}

    def cancel_action(self, action_id: str) -> bool:
        return bool(self.env.cancel_action(action_id))

    def observe(self) -> List[Observation]:
        return self.env.observe()

    def discover_tools(self) -> List[ToolDescriptor]:
        return self.env.discover_tools()

    def validate_action(self, action: ActionSpec) -> List[str]:
        return self.env.validate_action(action)

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        return self.env.execute(action, authorization)

    def poll_events(self, cursor: str) -> List[Any]:
        events, cur = self.env.poll_events(cursor)
        return [events, cur]

    def get_resource_state(self) -> Dict[str, Any]:
        return self.env.get_resource_state()

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        return self.env.get_pending_outcomes()

    def get_observable_metrics(self) -> List[Quantity]:
        return self.env.get_observable_metrics()

    def get_capabilities(self) -> Dict[str, Any]:
        caps = self.env.get_capabilities()
        caps["exec_rpc"] = {"versions": list(EXEC_RPC_VERSIONS), "pid": os.getpid()}
        return caps

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return self.env.measure(list(queries))


def serve(env: Any, socket_path: str, *, allowed_uids: Optional[set] = None, group: Optional[str] = SERVICE_GROUP,
          on_close: Optional[Callable[[], None]] = None) -> RpcServer:
    """Start serving ``env`` on ``socket_path`` (background thread). Returns the server; call ``.stop()``."""
    return RpcServer(socket_path, ExecService(env, on_close), EXEC_METHODS, allowed_uids=allowed_uids, group=group,
                     name="exec").start()


class _Pool:
    """One ``RpcClient`` connection per calling thread, so a long call does not block other threads."""

    def __init__(self, path: str, timeout: float):
        self.path, self.timeout = path, timeout
        self._local, self._all, self._lock = threading.local(), [], threading.Lock()

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        c = getattr(self._local, "client", None)
        if c is None:
            c = self._local.client = RpcClient(self.path, self.timeout)
            with self._lock:
                self._all.append(c)
        return c.call(method, *args, **kwargs)

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                c.close()
            self._all.clear()
        self._local = threading.local()


def wait_for_socket(path: str, timeout: float) -> None:
    t0 = time.time()
    while True:
        try:
            c = RpcClient(path, timeout=5.0)
            c.call("__ping__")
            c.close()
            return
        except (OSError, RemoteError) as e:
            if time.time() - t0 > timeout:
                raise ConnectionError(f"no service on {path}: {e}")
            time.sleep(0.05)


class RemoteEnvironment:
    """``contracts.EnvironmentAdapter`` over ExecRPC v1."""

    def __init__(self, socket_path: str, *, timeout: float = 900.0, connect_timeout: float = 30.0,
                 versions: Sequence[str] = EXEC_RPC_VERSIONS, client_name: str = "learner"):
        self.socket_path = socket_path
        wait_for_socket(socket_path, connect_timeout)
        self._pool = _Pool(socket_path, timeout)
        try:
            hello = self._pool.call("hello", list(versions), client_name)
        except RemoteError as e:
            self._pool.close()
            if e.kind == "VersionMismatch":
                raise VersionMismatch(e.message)
            raise
        self.rpc_version: str = hello["version"]
        self.interface_version: str = hello["interface_version"]
        self.server_info: Dict[str, Any] = hello
        if self.interface_version.split(".")[0] != EXEC_INTERFACE_VERSION.split(".")[0]:
            self._pool.close()
            raise VersionMismatch(f"environment speaks execution interface {self.interface_version}, "
                                  f"this client {EXEC_INTERFACE_VERSION}")

    def observe(self) -> List[Observation]:
        return self._pool.call("observe")

    def discover_tools(self) -> List[ToolDescriptor]:
        return self._pool.call("discover_tools")

    def validate_action(self, action: ActionSpec) -> List[str]:
        return self._pool.call("validate_action", action)

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        return self._pool.call("execute", action, authorization)

    def poll_events(self, cursor: str) -> Tuple[List[Observation], str]:
        events, cur = self._pool.call("poll_events", cursor)
        return list(events), str(cur)

    def get_resource_state(self) -> Dict[str, Any]:
        return self._pool.call("get_resource_state")

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        return self._pool.call("get_pending_outcomes")

    def get_observable_metrics(self) -> List[Quantity]:
        return self._pool.call("get_observable_metrics")

    def get_capabilities(self) -> Dict[str, Any]:
        return self._pool.call("get_capabilities")

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return self._pool.call("measure", list(queries))

    def ping(self) -> Dict[str, Any]:
        return self._pool.call("ping")

    def cancel_action(self, action_id: str) -> bool:
        return bool(self._pool.call("cancel_action", action_id))

    def shutdown(self) -> None:
        """Ask the service to close its environment and exit, then disconnect."""
        try:
            self._pool.call("close")
        finally:
            self._pool.close()

    def close(self) -> None:
        """Disconnect this client. The service keeps running (use :meth:`shutdown` to stop it)."""
        self._pool.close()


class RemotePolicy:
    """``contracts.PolicyEngine`` over the authorization service's socket. Decisions fail closed: when the service
    cannot be reached, ``authorize`` / ``check_primitive`` deny, ``check_grant`` is ``False`` and ``stopped`` is
    ``True``."""

    def __init__(self, socket_path: str, *, timeout: float = 60.0, connect_timeout: float = 30.0):
        self.socket_path = socket_path
        wait_for_socket(socket_path, connect_timeout)
        probe = RpcClient(socket_path, timeout=10.0)
        info = probe.call("__ping__")
        probe.close()
        if str(info.get("rpc", "")).split(".")[0] != RPC_VERSION.split(".")[0]:
            raise VersionMismatch(f"authorization service speaks rpc {info.get('rpc')}, this client {RPC_VERSION}")
        missing = [m for m in ("check_grant", "check_primitive", "stopped") if m not in info.get("methods", [])]
        if missing:
            raise VersionMismatch(f"authorization service does not expose {missing}")
        self.server_info = info
        self._pool = _Pool(socket_path, timeout)

    @staticmethod
    def _deny(e: Exception) -> AuthorizationDecision:
        return AuthorizationDecision(AuthVerdict.DENY, reasons=[f"authorization service unavailable: {type(e).__name__}"])

    def authorize(self, action: ActionSpec, task: TaskSpec, context: Optional[Dict[str, Any]] = None) -> AuthorizationDecision:
        try:
            return self._pool.call("authorize", action, task, context or {})
        except (OSError, RemoteError) as e:
            return self._deny(e)

    def check_grant(self, action: ActionSpec, decision: AuthorizationDecision) -> bool:
        try:
            return self._pool.call("check_grant", action, decision) is True
        except (OSError, RemoteError):
            return False

    def check_primitive(self, op: PrimitiveOp, action: ActionSpec) -> AuthorizationDecision:
        try:
            dec = self._pool.call("check_primitive", op, action)
        except (OSError, RemoteError) as e:
            return self._deny(e)
        return dec if isinstance(dec, AuthorizationDecision) else self._deny(TypeError("no decision"))

    def record(self, action: ActionSpec, result: ActionResult) -> None:
        self._pool.call("record", action, result)

    def approvals(self) -> List[Dict[str, Any]]:
        return self._pool.call("approvals")

    def resolve_approval(self, approval_id: str, approve: bool, by: str, note: str = "") -> AuthorizationDecision:
        return self._pool.call("resolve_approval", approval_id, approve, by, note)

    def emergency_stop(self, reason: str, by: str) -> None:
        self._pool.call("emergency_stop", reason, by)

    def stopped(self) -> bool:
        try:
            return self._pool.call("stopped") is not False
        except (OSError, RemoteError):
            return True

    def revoke(self, scope: str, by: str) -> None:
        self._pool.call("revoke", scope, by)

    def limits_state(self) -> Dict[str, Any]:
        return self._pool.call("limits_state")

    def close(self) -> None:
        self._pool.close()


# ----------------------------------------------------------------------------------------------------------------
# Process entry point
# ----------------------------------------------------------------------------------------------------------------
def load_factory(spec: str) -> Callable[..., Any]:
    """``"package.module:attribute"`` -> the attribute."""
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"expected module:attribute, got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


def call_factory(fn: Callable[..., Any], **available: Any) -> Any:
    """Call a factory with whichever of the offered keyword arguments it accepts."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**available)
    return fn(**{k: v for k, v in available.items() if k in params})


def build_environment(run_dir: str, policy: Any, *, provenance: str = "staging", clock: Any = None, secrets: Any = None,
                      executors: Sequence[str] = (), browser: bool = True, browser_config: Any = None,
                      limits: Any = None) -> Any:
    """A ``ComputerEnvironment`` with the internal and browser executors plus any ``module:factory`` executors.
    A factory is called with those of ``run_dir, clock, env, policy, secrets`` it accepts and must return a
    ``contracts.ModalityExecutor`` (or a list of them)."""
    from .env import ComputerEnvironment
    from .internal import InternalExecutor
    env = ComputerEnvironment(run_dir, policy, clock=clock, secrets=secrets, provenance=provenance, limits=limits)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    if browser:
        from .browser import BrowserExecutor
        env.register(BrowserExecutor(run_dir, clock=env.clock, config=browser_config))
    for spec in executors:
        made = call_factory(load_factory(spec), run_dir=run_dir, clock=env.clock, env=env, policy=policy,
                            secrets=secrets)
        for ex in (made if isinstance(made, (list, tuple)) else [made]):
            env.register(ex)
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cam_live.harness.service",
                                 description="CAM Live computer environment service (ExecRPC v1)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", required=True, help="unix socket to serve ExecRPC on")
    ap.add_argument("--auth-socket", default="", help="unix socket of the authorization service")
    ap.add_argument("--provenance", default="staging", choices=["staging", "live", "diagnostic"])
    ap.add_argument("--executor", action="append", default=[], metavar="MODULE:FACTORY",
                    help="additional modality executor factory (repeatable)")
    ap.add_argument("--secrets", default="", metavar="MODULE:FACTORY",
                    help="factory for the executor-side secret broker handle (called with run_dir, auth_socket)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--headed", action="store_true", help="show the browser window (needs a display)")
    ap.add_argument("--cdp", action="append", default=[], metavar="NAME=ENDPOINT",
                    help="attach an externally supplied Chromium as session cdp:NAME (repeatable)")
    ap.add_argument("--staging-sites", action="store_true", help="also start the local staging sites")
    ap.add_argument("--diagnostic-allow-all", action="store_true",
                    help="no authorization service: use the diagnostic allow-all policy (never with --provenance live)")
    ap.add_argument("--allow-uid", action="append", type=int, default=[], help="extra uid allowed to call")
    ap.add_argument("--connect-timeout", type=float, default=60.0)
    a = ap.parse_args(argv)

    if a.auth_socket:
        policy: Any = RemotePolicy(a.auth_socket, connect_timeout=a.connect_timeout)
    elif a.diagnostic_allow_all and a.provenance != "live":
        from ..diagnostics.world import AllowAllPolicy
        policy = AllowAllPolicy()
    else:
        print("refusing to start: --auth-socket is required (or --diagnostic-allow-all for non-live staging)",
              file=sys.stderr)
        return 2
    secrets = call_factory(load_factory(a.secrets), run_dir=a.run_dir, auth_socket=a.auth_socket) if a.secrets else None
    bcfg = None
    if not a.no_browser:
        from .browser import BrowserConfig
        bcfg = BrowserConfig(headless=not a.headed,
                             cdp_endpoints=dict(x.split("=", 1) for x in a.cdp))
    env = build_environment(a.run_dir, policy, provenance=a.provenance, secrets=secrets, executors=a.executor,
                            browser=not a.no_browser, browser_config=bcfg)
    sites = None
    if a.staging_sites:
        if a.provenance == "live":
            print("refusing to start: staging sites cannot be part of a live environment", file=sys.stderr)
            env.close()
            return 2
        from .staging import StagingSites
        sites = StagingSites().start()
        env.add_source(sites.outcome_source())
        env.annotate("staging", sites.capabilities())
    stop = threading.Event()
    server = serve(env, a.socket, allowed_uids={uid_of("learner"), os.getuid(), *a.allow_uid}, on_close=stop.set)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(json.dumps({"ready": True, "socket": a.socket, "pid": os.getpid(), "versions": list(EXEC_RPC_VERSIONS),
                      "staging": sites.urls if sites else None}), flush=True)
    stop.wait()
    server.stop()
    env.close()
    if sites is not None:
        sites.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
