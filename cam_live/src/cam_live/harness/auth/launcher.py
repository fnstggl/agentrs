"""Launcher: starts the separated CAM Live services under their own uids, supervises them and stops them.

    handle = launch(run_dir)                 # all services of cam_live.harness.SERVICE_MODULES
    handle.status()                          # {"services": {name: {"state", "pid", "uid", "restarts", ...}}, ...}
    handle.stop()

    python -m cam_live.harness.auth.launcher --run-dir R [--services auth,exec] [--policy P]

Separated mode (root): the service users are created, the run directory is prepared with the ownership below, each
service is started as its own uid with a minimal environment (no inherited tokens), ``no_new_privs`` set and all
capabilities gone (they are dropped by the kernel on the uid change).

====================  =========================  ======  =============================================================
path                  owner:group                mode    purpose
====================  =========================  ======  =============================================================
R/                    root:<learner group>       1771    learner writes its logs here; sticky: nobody can remove or
                                                         replace another uid's entries (audit / verifier logs)
R/sock/               root:camlive               1770    one socket per service; sticky
R/auth/, R/vault/     camauth                    0700    authorization state, keys, stop flags; secrets
R/audit.jsonl         camauth                    0600    hash-chained audit log
R/verifier/, .jsonl   camverify                  0700/0600  verifier key and log
R/browser_profile/    cambrowser                 0700    persistent browser sessions (cookies)
R/exec/               executor uid               0700    executor state and its copy of the grant key
R/sandbox/            camsbx:<executor group>    2770    generated-code workspace
R/artifacts/          executor uid:camlive       2750    artifacts readable by the other services
R/learner/            camlearn                   0700    learner state
R/logs/               root                       0755    service stdout/stderr (files 0640 root)
====================  =========================  ======  =============================================================

Supervision: a crashed service is restarted with exponential backoff (bounded number of restarts per window).
When the emergency-stop flag appears, every service except the authorization service is terminated together with
everything it started, and nothing is restarted until the operator resets the stop and calls ``resume()``.
"""
from __future__ import annotations

import argparse
import dataclasses
import grp
import importlib.util
import os
import pwd
import secrets as _secrets
import shutil
import signal
import socket as _socket
import stat
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .. import SERVICE_GROUP, SERVICE_MODULES, SERVICE_USERS, SOCK_AUTH, SOCK_EXEC, SOCK_VERIFY, ensure_service_users
from . import stop as _stop
from .grants import KEY_BYTES
from .schema import DEFAULT_POLICY_PATH

START_ORDER = ("auth", "verifier", "sandbox", "exec", "learner")
#: "sandbox" is the sandbox host (cam_live.harness.sandbox_host): root, so that it can drop every program to the
#: sandbox uid; it serves only the environment service's uid. Started with "exec" in the separated deployment.
DEFAULT_ROLES = {"auth": "auth", "exec": "browser", "verifier": "verifier", "learner": "learner", "sandbox": "root"}
DEFAULT_SOCKETS = {"auth": SOCK_AUTH, "exec": SOCK_EXEC, "verifier": SOCK_VERIFY, "learner": "",
                   "sandbox": "sandbox.sock"}
ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ", "HF_HOME", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM",
                   "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH",
                   "PLAYWRIGHT_BROWSERS_PATH", "CAM_LIVE_GPU_LOCK", "CAM_LIVE_HOME", "OMP_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   # this instance's public address: the authorization service counts it as an address of this
                   # machine (localnet), so it must see it in the separated deployment too (G2-harness D7)
                   "PUBLIC_IPADDR")
RUN_TAG_ENV, SERVICE_ENV = "CAM_LIVE_RUN_TAG", "CAM_LIVE_SERVICE"
# sets no_new_privs, then becomes the real command
_TRAMPOLINE = ("import ctypes,os,sys\n"
               "ctypes.CDLL(None,use_errno=True).prctl(38,1,0,0,0)\n"
               "os.execv(sys.argv[1],sys.argv[1:])\n")
#: How a separated service's Python starts (G2-harness R2-C1): ``python -I service_boot.py <roots> -m <module>``
#: with cwd "/": isolated mode (no PYTHON* variable, no user site, neither the cwd nor a script's directory on
#: sys.path), then exactly the launcher's import roots in front of sys.path (cam_live.harness.service_boot)
_BOOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service_boot.py")
#: PATH of services that run as root: system directories only (none of them writable by a service uid)
ROOT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


@dataclass
class ServiceSpec:
    name: str
    module: str = ""                     # started as: python -m <module> --run-dir R [--socket R/sock/<socket>]
    role: str = ""                       # key of SERVICE_USERS, or "root"
    socket: str = ""                     # socket file name under R/sock/ ("" = the service has none)
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    restart: bool = True
    stop_on_estop: bool = True
    ready_timeout_s: float = 30.0
    argv: Optional[List[str]] = None     # command prefix instead of ``python -m module``; gets the same arguments


@dataclass
class LauncherConfig:
    separated: Optional[bool] = None     # None: separate when running as root
    python: str = sys.executable
    policy_path: Optional[str] = None    # copied to R/auth/policy.yaml; default: the packaged default policy
    operator_pubkey: Optional[str] = None
    roles: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLES))
    extra_args: Dict[str, List[str]] = field(default_factory=dict)
    extra_env: Dict[str, str] = field(default_factory=dict)
    env_passthrough: Tuple[str, ...] = ENV_PASSTHROUGH
    pythonpath: List[str] = field(default_factory=list)
    poll_s: float = 0.2
    restart_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    max_restarts: int = 5
    restart_window_s: float = 300.0
    stop_grace_s: float = 5.0
    wait_ready: bool = True
    subreaper: bool = True
    require_auth: bool = True            # refuse to run anything if the authorization service does not come up

    @classmethod
    def coerce(cls, config: Any) -> "LauncherConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if hasattr(config, "section"):                      # CamLiveConfig: section "launcher"
            config = config.section("launcher")
        if isinstance(config, dict):
            known = {f.name for f in dataclasses.fields(cls)}
            unknown = set(config) - known
            if unknown:
                raise ValueError(f"unknown launcher setting(s): {sorted(unknown)}")
            return cls(**config)
        raise TypeError("config must be a LauncherConfig, a mapping or a CamLiveConfig")


# ---------------------------------------------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------------------------------------------
def _is_python(exe: str) -> bool:
    return os.path.basename(exe).startswith("python")


def _src_root() -> str:
    import cam_live
    return os.path.dirname(os.path.dirname(os.path.abspath(cam_live.__file__)))


def _ensure_users(retries: int = 8) -> Dict[str, int]:
    last: Optional[Exception] = None
    for i in range(retries):                       # useradd fails while another process holds the passwd lock
        try:
            return ensure_service_users()
        except subprocess.CalledProcessError as e:
            last = e
            time.sleep(0.3 * (i + 1))
    raise RuntimeError(f"could not create the service users: {last}")


def _ids(role: str) -> Tuple[int, int]:
    if role == "root":
        return 0, 0
    pw = pwd.getpwnam(SERVICE_USERS[role])
    return pw.pw_uid, pw.pw_gid


def _mk(path: str, is_dir: bool, uid: int, gid: int, mode: int, chown: bool) -> None:
    """Create (or adopt) one entry without ever following a symlink planted in a directory others can write."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        st = None
    if st is not None and (stat.S_ISLNK(st.st_mode) or stat.S_ISDIR(st.st_mode) != is_dir):
        raise RuntimeError(f"refusing to use {path}: it is a symlink or of the wrong type")
    if is_dir:
        if st is None:
            os.mkdir(path, 0o700)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    elif st is None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    else:
        # no O_CREAT on an existing file: with fs.protected_regular the kernel refuses even root an O_CREAT open of
        # another uid's file inside a sticky directory
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        if chown:
            os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _write_private(path: str, data: bytes, uid: int, gid: int, chown: bool, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW
    fd = os.open(path, flags if os.path.lexists(path) else flags | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, data)
        os.fsync(fd)
        if chown:
            os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def can_traverse(path: str, uid: int, gids: Sequence[int]) -> Optional[str]:
    """None if ``uid`` can reach ``path`` (search permission on every directory), else the blocking component."""
    if uid == 0:
        return None
    cur = "/"
    parts = [p for p in os.path.abspath(path).split("/") if p]
    for i, part in enumerate([""] + parts):
        cur = os.path.join(cur, part) if part else cur
        try:
            st = os.stat(cur)
        except OSError:
            return cur
        if not stat.S_ISDIR(st.st_mode):
            continue
        bit = stat.S_IXUSR if st.st_uid == uid else (stat.S_IXGRP if st.st_gid in gids else stat.S_IXOTH)
        if not st.st_mode & bit:
            return cur
    return None


def prepare_run_dir(run_dir: str, separated: bool, roles: Optional[Dict[str, str]] = None,
                    policy_path: Optional[str] = None, operator_pubkey: Optional[str] = None) -> Dict[str, Any]:
    """Create the run directory layout (idempotent). Returns the layout: paths, owners, the policy copy."""
    R = os.path.abspath(run_dir)
    roles = dict(DEFAULT_ROLES, **(roles or {}))
    os.makedirs(R, exist_ok=True)
    me = (os.getuid(), os.getgid())
    if separated:
        if os.geteuid() != 0:
            raise PermissionError("separated mode needs root")
        ids = {r: _ids(r) for r in SERVICE_USERS}
        shared = grp.getgrnam(SERVICE_GROUP).gr_gid
        lgid = ids["learner"][1]
        root_gid = lgid if all(g != lgid for r, (_u, g) in ids.items() if r != "learner") else shared
        ex = _ids(roles["exec"])
    else:
        ids = {r: me for r in SERVICE_USERS}
        shared, root_gid, ex = me[1], me[1], me
    A, V, B, S, L = ids["auth"], ids["verifier"], ids["browser"], ids["sandbox"], ids["learner"]
    plan = [
        ("", True, (0, root_gid) if separated else me, 0o1771 if separated else 0o700),
        ("sock", True, (0, shared) if separated else me, 0o1770 if separated else 0o700),
        ("logs", True, (0, 0) if separated else me, 0o755 if separated else 0o700),
        ("auth", True, A, 0o700), ("auth/keys", True, A, 0o700), ("vault", True, A, 0o700),
        ("audit.jsonl", False, A, 0o600),
        ("verifier", True, V, 0o700), ("verifier.jsonl", False, V, 0o600),
        ("browser_profile", True, B, 0o700),
        ("exec", True, ex, 0o700),
        ("sandbox", True, (S[0], ex[1]), 0o2770 if separated else 0o700),
        ("artifacts", True, (ex[0], shared), 0o2750 if separated else 0o700),
        ("learner", True, L, 0o700),
    ]
    for rel, is_dir, (uid, gid), mode in plan:
        if not rel and not separated:
            continue                     # single-uid mode: the caller's directory keeps its own mode
        _mk(os.path.join(R, rel) if rel else R, is_dir, uid, gid, mode, chown=separated)
    # keys: the grant key exists once per reader, each copy readable by exactly one uid
    key_auth, key_exec = os.path.join(R, "auth", "keys", "grant.key"), os.path.join(R, "exec", "grant.key")
    if os.path.lexists(key_auth):
        with open(key_auth, "rb") as f:
            key = f.read()
    else:
        key = _secrets.token_bytes(KEY_BYTES)
        _write_private(key_auth, key, A[0], A[1], separated)
    _write_private(key_exec, key, ex[0], ex[1], separated)
    src = policy_path or DEFAULT_POLICY_PATH
    pol = os.path.join(R, "auth", "policy.yaml")
    with open(src, "rb") as f:
        _write_private(pol, f.read(), A[0], A[1], separated)
    if operator_pubkey:
        with open(operator_pubkey, "rb") as f:
            _write_private(os.path.join(R, "auth", "keys", "operator.pub"), f.read(), A[0], A[1], separated)
    return {"run_dir": R, "separated": separated, "policy": pol, "policy_source": src,
            "uids": {r: u for r, (u, _g) in ids.items()}, "executor_uid": ex[0], "shared_gid": shared,
            "paths": {rel or ".": {"uid": uid, "gid": gid, "mode": oct(mode)} for rel, _d, (uid, gid), mode in plan}}


# ---------------------------------------------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------------------------------------------
def _proc_table() -> Dict[int, Tuple[int, int, int]]:
    """pid -> (ppid, session id, real uid) for every process visible in /proc."""
    out: Dict[int, Tuple[int, int, int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                rest = f.read().rsplit(b")", 1)[1].split()
            out[int(name)] = (int(rest[1]), int(rest[3]), os.stat(f"/proc/{name}").st_uid)
        except (OSError, IndexError, ValueError):
            continue
    return out


def _environ(pid: int) -> List[bytes]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return f.read().split(b"\0")
    except OSError:
        return []


def _tagged(pid: int, tag: bytes, service: bytes) -> bool:
    env = _environ(pid)
    return tag in env and (not service or service in env)


def _other_run(pid: int, tag: bytes) -> bool:
    """True when the process carries the run tag of a *different* launcher."""
    prefix = RUN_TAG_ENV.encode() + b"="
    return any(e.startswith(prefix) and e != tag for e in _environ(pid))


@dataclass
class _Svc:
    spec: ServiceSpec
    uid: int = 0
    gid: int = 0
    state: str = "new"                   # new|starting|running|backoff|exited|failed|halted|stopped|unavailable
    proc: Optional[subprocess.Popen] = None
    pid: int = 0
    restarts: int = 0
    restart_times: List[float] = field(default_factory=list)
    next_start: float = 0.0
    last_exit: Optional[int] = None
    started_at: float = 0.0
    reason: str = ""
    ready: bool = False
    log_path: str = ""


class LaunchHandle:
    def __init__(self, run_dir: str, cfg: LauncherConfig, separated: bool, layout: Dict[str, Any], specs: List[ServiceSpec]):
        self.run_dir, self.cfg, self.separated, self.layout = os.path.abspath(run_dir), cfg, separated, layout
        self.tag = _secrets.token_hex(6)
        self._svcs: Dict[str, _Svc] = {s.name: _Svc(spec=s) for s in specs}
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._halted = False
        self._halt_reason = ""
        self._thread: Optional[threading.Thread] = None
        self._events: List[Dict[str, Any]] = []
        self._known: set = set()         # pids seen in our services' sessions (to reap the ones we adopt)
        self._last_reap = 0.0

    # -- helpers ---------------------------------------------------------------------------------------------
    def _event(self, kind: str, **kw: Any) -> None:
        self._events.append(dict(kw, kind=kind, ts=time.time()))
        del self._events[:-200]

    def _sock(self, spec: ServiceSpec) -> str:
        return os.path.join(self.run_dir, "sock", spec.socket) if spec.socket else ""

    def _env(self, spec: ServiceSpec) -> Dict[str, str]:
        env = {k: os.environ[k] for k in self.cfg.env_passthrough if k in os.environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["PYTHONPATH"] = os.pathsep.join([_src_root()] + list(self.cfg.pythonpath))
        env["PYTHONUNBUFFERED"] = "1"
        env["HOME"] = os.path.join(self.run_dir, {"auth": "auth", "verifier": "verifier", "learner": "learner",
                                                  "browser": "browser_profile", "sandbox": "sandbox"}.get(spec.role, "exec"))
        if spec.role == "root":
            # a root service gets nothing from a directory a service uid owns (R2-C1)
            env["HOME"] = "/root" if os.path.isdir("/root") else "/"
            env["PATH"] = ROOT_PATH
        env[RUN_TAG_ENV], env[SERVICE_ENV] = self.tag, spec.name
        env.update(self.cfg.extra_env)
        env.update(spec.env)
        return env

    def _argv(self, spec: ServiceSpec) -> List[str]:
        cmd = list(spec.argv) if spec.argv is not None else [self.cfg.python, "-m", spec.module]
        cmd += ["--run-dir", self.run_dir]
        if spec.socket:
            cmd += ["--socket", self._sock(spec)]
        if spec.argv is None:
            if spec.name == "auth":
                cmd += ["--policy", self.layout["policy"], "--executor-role", self.cfg.roles.get("exec", "browser")]
                if not self.separated:
                    cmd += ["--single-uid"]
        cmd += list(spec.args) + list(self.cfg.extra_args.get(spec.name, []))
        if self.separated:
            exe = os.path.abspath(shutil.which(cmd[0]) or cmd[0])     # absolute, not resolved: a venv is its symlink
            if _is_python(exe):
                roots = os.pathsep.join([_src_root()] + list(self.cfg.pythonpath))
                cmd = [self.cfg.python, "-I", "-c", _TRAMPOLINE, exe, "-I", _BOOT_PATH, roots] + cmd[1:]
            else:
                cmd = [self.cfg.python, "-I", "-c", _TRAMPOLINE, exe] + cmd[1:]
        return cmd

    def _spawn(self, svc: _Svc) -> None:
        spec = svc.spec
        svc.uid, svc.gid = _ids(spec.role) if self.separated else (os.getuid(), os.getgid())
        sock = self._sock(spec)
        if sock and os.path.lexists(sock):
            try:
                os.unlink(sock)                 # stale socket of a crashed or earlier instance
            except OSError:
                pass
        if self.separated:
            gids = [svc.gid, self.layout["shared_gid"]]
            for p in (self.run_dir, self.cfg.python, _src_root()):
                blocked = can_traverse(p, svc.uid, gids)
                if blocked:
                    raise PermissionError(f"uid {svc.uid} ({spec.role}) cannot reach {p}: no search permission on {blocked}")
        svc.log_path = os.path.join(self.run_dir, "logs", f"{spec.name}.log")
        logfd = os.open(svc.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o640)
        kw: Dict[str, Any] = dict(stdin=subprocess.DEVNULL, stdout=logfd, stderr=subprocess.STDOUT, close_fds=True,
                                  start_new_session=True, env=self._env(spec),
                                  cwd="/" if self.separated else self.run_dir)    # R2-C1: never a writable cwd
        if self.separated and spec.role != "root":
            kw.update(user=svc.uid, group=svc.gid, extra_groups=[self.layout["shared_gid"]], umask=0o077)
        try:
            svc.proc = subprocess.Popen(self._argv(spec), **kw)
        finally:
            os.close(logfd)
        svc.pid, svc.started_at, svc.state, svc.ready, svc.reason = svc.proc.pid, time.time(), "starting", False, ""
        self._event("start", service=spec.name, pid=svc.pid, uid=svc.uid)

    def _is_ready(self, svc: _Svc) -> bool:
        sock = self._sock(svc.spec)
        if not sock:
            return svc.proc is not None and svc.proc.poll() is None
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(sock)                     # something is really accepting, not just a file left behind
            _pid, uid, _gid = struct.unpack("3i", s.getsockopt(_socket.SOL_SOCKET, _socket.SO_PEERCRED, struct.calcsize("3i")))
            return not self.separated or uid == svc.uid
        except OSError:
            return False
        finally:
            s.close()

    def wait_ready(self, name: str, timeout: Optional[float] = None) -> bool:
        svc = self._svcs[name]
        end = time.time() + (svc.spec.ready_timeout_s if timeout is None else timeout)
        while time.time() < end:
            if svc.state in ("failed", "exited", "unavailable", "halted", "stopped"):
                return False
            if svc.proc is None or svc.proc.poll() is not None:
                return False                    # it exited before becoming ready: do not wait out the timeout
            if self._is_ready(svc):
                svc.ready = True
                if svc.state == "starting":
                    svc.state = "running"
                return True
            time.sleep(0.02)
        return False

    def _tree(self, svc: _Svc) -> List[int]:
        """The service process and everything it started that is still alive: same session, same run tag, or (in
        separated mode) a descendant of this launcher running under the service's uid."""
        table = _proc_table()
        me = os.getpid()
        pids = {p for p, (_pp, sid, _u) in table.items() if svc.pid and sid == svc.pid}
        tag, name = f"{RUN_TAG_ENV}={self.tag}".encode(), f"{SERVICE_ENV}={svc.spec.name}".encode()
        for p in table:
            if p not in pids and p != me and _tagged(p, tag, name):
                pids.add(p)
        if self.separated and svc.uid != 0:
            kids: Dict[int, List[int]] = {}
            for p, (pp, _sid, _u) in table.items():
                kids.setdefault(pp, []).append(p)
            stack, mine = [me], set()
            while stack:
                for c in kids.get(stack.pop(), []):
                    if c not in mine:
                        mine.add(c)
                        stack.append(c)
            others = {s.uid for s in self._svcs.values() if s is not svc}
            if svc.uid not in others:
                pids |= {p for p in mine if table[p][2] == svc.uid and not _other_run(p, tag)}
        pids.discard(me)
        return sorted(pids)

    def _terminate(self, svc: _Svc, state: str, reason: str = "") -> None:
        pids = self._tree(svc) if svc.pid else []
        for sig, wait in ((signal.SIGTERM, self.cfg.stop_grace_s), (signal.SIGKILL, 2.0)):
            for p in pids:
                try:
                    os.kill(p, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            end = time.time() + wait
            while time.time() < end:
                if svc.proc is not None:
                    svc.proc.poll()
                pids = [p for p in pids if os.path.exists(f"/proc/{p}") and _alive(p)]
                if not pids:
                    break
                time.sleep(0.02)
            if not pids:
                break
            pids = sorted(set(pids) | set(self._tree(svc)))
        if svc.proc is not None:
            try:
                svc.last_exit = svc.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        svc.state, svc.reason, svc.ready = state, reason, False
        self._event("terminate", service=svc.spec.name, state=state, reason=reason)

    # -- lifecycle -------------------------------------------------------------------------------------------
    def _start_all(self) -> None:
        if self.cfg.subreaper:
            try:
                import ctypes
                ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)      # PR_SET_CHILD_SUBREAPER
            except OSError:
                pass
        if _stop.is_stopped(self.run_dir):
            self._halted, self._halt_reason = True, "emergency stop is engaged"
        for name in sorted(self._svcs, key=lambda n: START_ORDER.index(n) if n in START_ORDER else len(START_ORDER)):
            svc = self._svcs[name]
            if svc.state == "unavailable":
                continue
            if self._halted and svc.spec.stop_on_estop:
                svc.state, svc.reason = "halted", self._halt_reason
                continue
            try:
                self._spawn(svc)
            except (OSError, PermissionError, KeyError, subprocess.SubprocessError) as e:
                svc.state, svc.reason = "failed", f"{type(e).__name__}: {e}"
                self._event("spawn_failed", service=name, error=svc.reason)
            if self.cfg.wait_ready and svc.state == "starting":
                if not self.wait_ready(name):
                    rc = svc.proc.poll() if svc.proc is not None else None
                    if rc is not None:
                        svc.last_exit = rc
                    self._terminate(svc, "failed", f"exited with status {rc} before becoming ready" if rc is not None
                                    else f"not ready after {svc.spec.ready_timeout_s:g} s")
            if name == "auth" and self.cfg.require_auth and svc.state != "running" and self.cfg.wait_ready:
                reason = svc.reason
                self.stop()
                raise RuntimeError(f"the authorization service did not start ({reason}); nothing else is started. "
                                   f"See {svc.log_path}")
        self._thread = threading.Thread(target=self._supervise, daemon=True, name="cam-live-launcher")
        self._thread.start()

    def _reap(self) -> None:
        """As a child subreaper this process inherits orphaned descendants of the services; collect the dead ones
        (only pids previously seen inside one of our services' sessions — never somebody else's child)."""
        table = _proc_table()
        own = {s.pid for s in self._svcs.values() if s.pid}
        for p, (_pp, sid, _u) in table.items():
            if sid in own and p not in own:
                self._known.add(p)
        me = os.getpid()
        for p in list(self._known):
            if p not in table:
                self._known.discard(p)
            elif table[p][0] == me and not _alive(p):
                try:
                    os.waitpid(p, os.WNOHANG)
                except ChildProcessError:
                    pass
                self._known.discard(p)

    def _supervise(self) -> None:
        while not self._stopping.wait(self.cfg.poll_s):
            with self._lock:
                if self.cfg.subreaper and time.time() - self._last_reap > 2.0:
                    self._last_reap = time.time()
                    self._reap()
                if not self._halted and _stop.is_stopped(self.run_dir):
                    self._halted, self._halt_reason = True, "emergency stop is engaged"
                    self._event("halt", reason=self._halt_reason)
                    for svc in reversed(list(self._svcs.values())):
                        if svc.spec.stop_on_estop and svc.state in ("starting", "running", "backoff"):
                            self._terminate(svc, "halted", self._halt_reason)
                now = time.time()
                for svc in self._svcs.values():
                    if svc.state in ("starting", "running") and svc.proc is not None:
                        rc = svc.proc.poll()
                        if rc is None:
                            if svc.state == "starting" and self._is_ready(svc):
                                svc.state, svc.ready = "running", True
                            continue
                        svc.last_exit, svc.ready = rc, False
                        self._event("exit", service=svc.spec.name, status=rc)
                        for p in self._tree(svc):                # whatever the dead service left behind
                            try:
                                os.kill(p, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                pass
                        if not svc.spec.restart or (self._halted and svc.spec.stop_on_estop):
                            svc.state, svc.reason = "exited", f"exit status {rc}"
                            continue
                        svc.restart_times = [t for t in svc.restart_times if now - t < self.cfg.restart_window_s]
                        if len(svc.restart_times) >= self.cfg.max_restarts:
                            svc.state = "failed"
                            svc.reason = (f"crashed {len(svc.restart_times) + 1} times within "
                                          f"{self.cfg.restart_window_s:g} s (last exit status {rc})")
                            self._event("gave_up", service=svc.spec.name)
                            continue
                        delay = min(self.cfg.max_backoff_s, self.cfg.restart_backoff_s * (2 ** len(svc.restart_times)))
                        svc.state, svc.next_start, svc.reason = "backoff", now + delay, f"exit status {rc}; restarting"
                    elif svc.state == "backoff" and now >= svc.next_start:
                        svc.restart_times.append(now)
                        svc.restarts += 1
                        try:
                            self._spawn(svc)
                        except (OSError, PermissionError, subprocess.SubprocessError) as e:
                            svc.state, svc.reason = "failed", f"{type(e).__name__}: {e}"

    def resume(self) -> Dict[str, Any]:
        """After the operator reset the emergency stop: start the halted services again."""
        with self._lock:
            if _stop.is_stopped(self.run_dir):
                raise RuntimeError("the emergency stop is still engaged; only the operator can reset it")
            self._halted, self._halt_reason = False, ""
            for name in sorted(self._svcs, key=lambda n: START_ORDER.index(n) if n in START_ORDER else len(START_ORDER)):
                svc = self._svcs[name]
                if svc.state in ("halted", "exited"):
                    svc.restart_times = []
                    self._spawn(svc)
        for name, svc in self._svcs.items():
            if svc.state == "starting" and self.cfg.wait_ready:
                self.wait_ready(name)
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            services = {}
            for name, svc in self._svcs.items():
                alive = svc.proc is not None and svc.proc.poll() is None
                if svc.state == "starting" and alive and self._is_ready(svc):
                    svc.state, svc.ready = "running", True
                services[name] = {"state": svc.state, "pid": svc.pid if alive else 0, "uid": svc.uid,
                                  "role": svc.spec.role, "restarts": svc.restarts, "last_exit": svc.last_exit,
                                  "ready": svc.ready and alive, "socket": self._sock(svc.spec), "reason": svc.reason,
                                  "started_at": svc.started_at, "log": svc.log_path}
            return {"run_dir": self.run_dir, "separated": self.separated, "tag": self.tag, "halted": self._halted,
                    "halt_reason": self._halt_reason, "stopped": self._stopping.is_set(), "services": services,
                    "events": list(self._events[-20:])}

    def stop(self) -> Dict[str, Any]:
        """Stop every service (reverse start order) and everything they started. Idempotent."""
        self._stopping.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)
        with self._lock:
            order = sorted(self._svcs, key=lambda n: START_ORDER.index(n) if n in START_ORDER else len(START_ORDER))
            for name in reversed(order):
                svc = self._svcs[name]
                if svc.pid and svc.state not in ("stopped", "unavailable"):
                    keep = svc.state if svc.state in ("failed",) else "stopped"
                    self._terminate(svc, keep, svc.reason if keep == "failed" else "")
                sock = self._sock(svc.spec)
                if sock and os.path.lexists(sock):
                    try:
                        os.unlink(sock)
                    except OSError:
                        pass
        return self.status()

    def __enter__(self) -> "LaunchHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except (OSError, IndexError):
        return False


def _resolve(services: Optional[Sequence[Union[str, ServiceSpec]]], cfg: LauncherConfig) -> Tuple[List[ServiceSpec], Dict[str, str]]:
    specs: List[ServiceSpec] = []
    unavailable: Dict[str, str] = {}
    for s in (list(SERVICE_MODULES) if services is None else services):
        if isinstance(s, ServiceSpec):
            spec = dataclasses.replace(s)
            spec.role = spec.role or cfg.roles.get(spec.name, "")
        elif s in SERVICE_MODULES:
            spec = ServiceSpec(name=s, module=SERVICE_MODULES[s], role=cfg.roles.get(s, DEFAULT_ROLES.get(s, "")),
                               socket=DEFAULT_SOCKETS.get(s, ""),
                               stop_on_estop=(s != "auth"))
        else:
            raise KeyError(f"unknown service {s!r}; known: {sorted(SERVICE_MODULES)}")
        if spec.role != "root" and spec.role not in SERVICE_USERS:
            raise KeyError(f"service {spec.name!r}: unknown role {spec.role!r}")
        if spec.argv is None:
            try:
                found = importlib.util.find_spec(spec.module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                unavailable[spec.name] = f"module {spec.module} is not importable"
        specs.append(spec)
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("duplicate service name")
    return specs, unavailable


def launch(run_dir: str, services: Optional[Sequence[Union[str, ServiceSpec]]] = None, config: Any = None) -> LaunchHandle:
    """Prepare ``run_dir`` and start the services (names from ``SERVICE_MODULES`` and/or :class:`ServiceSpec`
    objects; default: all). Services whose module does not exist are reported as ``unavailable`` and not started.
    Raises ``RuntimeError`` if the authorization service was requested and does not come up."""
    cfg = LauncherConfig.coerce(config)
    separated = (os.geteuid() == 0) if cfg.separated is None else bool(cfg.separated)
    if separated:
        _ensure_users()
    if separated and services is not None:
        names = [x.name if isinstance(x, ServiceSpec) else x for x in services]
        if "exec" in names and "sandbox" not in names and cfg.roles.get("exec", "browser") != "root":
            # the environment service is not root: its generated-code sandbox runs in the sandbox host (G2-harness D2)
            services = list(services) + ["sandbox"]
    specs, unavailable = _resolve(services, cfg)
    by_name = {sp.name: sp for sp in specs}
    if "sandbox" in by_name and by_name["sandbox"].argv is None:
        sb = by_name["sandbox"]
        exec_role = cfg.roles.get("exec", "browser")
        if separated:
            sb.args = list(sb.args) + ["--client-uid", str(_ids(exec_role)[0])]
        if "exec" in by_name:
            by_name["exec"].env = dict(by_name["exec"].env, CAM_LIVE_SANDBOX_SOCKET=os.path.join(
                os.path.abspath(run_dir), "sock", by_name["sandbox"].socket or "sandbox.sock"))
    layout = prepare_run_dir(run_dir, separated, cfg.roles, cfg.policy_path, cfg.operator_pubkey)
    handle = LaunchHandle(run_dir, cfg, separated, layout, specs)
    for name, why in unavailable.items():
        handle._svcs[name].state, handle._svcs[name].reason = "unavailable", why
    handle._start_all()
    return handle


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cam_live.harness.auth.launcher")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--services", default="", help="comma-separated subset of " + ",".join(SERVICE_MODULES))
    ap.add_argument("--policy", default=None)
    ap.add_argument("--operator-pubkey", default=None)
    ap.add_argument("--no-separation", action="store_true", help="single-uid development mode (no isolation)")
    a = ap.parse_args(argv)
    cfg = LauncherConfig(policy_path=a.policy, operator_pubkey=a.operator_pubkey,
                         separated=False if a.no_separation else None)
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    handle = launch(a.run_dir, [s for s in a.services.split(",") if s] or None, cfg)
    import json
    print(json.dumps(handle.status(), indent=1, default=str), flush=True)
    done.wait()
    handle.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
