"""cam_live.harness.sandbox — sandboxed code execution: the ``code`` execution modality (workstream EC).

Generated programs (Python, shell) run in a persistent workspace directory and nowhere else. Layers, all applied
to the job's first process before any generated code runs and inherited by everything it starts:

1. a **monitor** process per job (not sandboxed, harness uid) that is the child-subreaper of the job: it enforces
   the wall-clock deadline, a total-memory and task-count watchdog, cancellation, kills the whole process group
   and reaps every descendant, so nothing is left behind;
2. **rlimits**: address space, CPU seconds, tasks (per uid), open files, file size, no core dumps;
3. **uid drop** to the sandbox user (``camsbx`` when it exists and we are root; ``nobody`` otherwise as a clearly
   reported weaker mode; no drop at all when not root — weakest, must be allowed explicitly);
4. ``no_new_privs``;
5. **Landlock**: read/execute on the interpreter, standard library, approved packages and system binaries; write
   only inside the workspace; nothing else on the filesystem can be opened;
6. **seccomp**: no socket creation of any family (a local ``socketpair`` is allowed), no connect/bind/listen, no
   ptrace / process_vm_* / mount / namespaces / bpf / keyrings / io_uring / kernel modules, and no ``setsid`` /
   ``setpgid`` so a job cannot leave the process group that the monitor kills;
7. a clean environment (no variable of the harness is inherited), output truncation.

**Scoped network.** A job that is authorized to reach the network still cannot create a socket. It inherits one
file descriptor to an *egress broker* running in the harness. For every outbound TCP connection the in-sandbox
shim hands the broker one end of a fresh ``socketpair`` together with ``host:port``; the broker asks
``ExecContext.check`` (one ``net.connect`` primitive per connection), resolves the name itself, refuses private
and loopback destinations unless configured otherwise, connects, and relays bytes. ``capabilities()`` states
which of this the kernel enforces and which it does not.

Provenance: the rlimit / uid-drop / Landlock / seccomp sequence is adapted from
``crucible/src/crucible/sandbox/{runner.py,boot.py}`` (agentrs commit 3968583, verified there by
``tests/test_sandbox.py``). Changes: exec and fork are allowed (general programs and shells), process-targeted
signal rules are replaced by a per-job monitor with a pinned process group, the JSON job protocol is replaced by
exec of the target program, and the socket-fd broker was added.
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.util
import errno
import hashlib
import importlib.util
import ipaddress
import json
import os
import pwd
import re
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import contracts as c
from . import SERVICE_USERS

__all__ = ["SandboxConfig", "SandboxLimits", "SandboxUnavailable", "RunResult", "SandboxProcess", "Sandbox",
           "EgressBroker", "SandboxExecutor", "probe"]


# ==============================================================================================================
# Configuration
# ==============================================================================================================
@dataclass
class SandboxLimits:
    """Resource limits of one execution. ``k`` = kernel-enforced, ``m`` = enforced by the monitor (polling)."""
    wall_s: float = 30.0          # m: wall-clock deadline, whole process group killed
    cpu_s: int = 60               # k: RLIMIT_CPU, per process
    mem_mb: int = 1024            # k: RLIMIT_AS, per process
    mem_total_mb: int = 2048      # m: resident memory summed over the job's processes
    nproc: int = 64               # m: tasks (processes + threads) of this job
    nofile: int = 256             # k: RLIMIT_NOFILE
    fsize_mb: int = 64            # k: RLIMIT_FSIZE, per file
    out_bytes: int = 65536        # harness: captured bytes kept per stream


@dataclass
class SandboxConfig:
    base_dir: str = field(default_factory=lambda: os.environ.get("CAM_LIVE_SANDBOX_DIR", "/var/tmp/cam-live-sbx"))
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    max_wall_s: float = 600.0             # ceiling for a foreground execution, whatever the action asks for
    uid_nproc: int = 256                  # k: RLIMIT_NPROC ceiling for the sandbox uid (all jobs together)
    workspace_max_mb: int = 2048          # harness: refuse new work when the workspace is larger than this
    packages: Tuple[str, ...] = ("numpy", "PIL", "yaml")      # importable packages exposed read-only
    extra_read_paths: Tuple[str, ...] = ()                    # more read-only paths (toolchains, datasets)
    allow_same_uid: bool = False          # run without a uid drop when not root (weakest mode)
    landlock: bool = True                 # switching a layer off is reported by capabilities() as weak
    seccomp: bool = True
    allow_private_destinations: bool = False   # scoped network: loopback / private / link-local destinations
    egress_max_connections: int = 16
    egress_max_bytes: int = 256 * 2**20   # per execution, both directions
    egress_connect_timeout_s: float = 10.0
    max_file_bytes: int = 16 * 2**20      # write_file / read_file payload ceiling
    keep_exec_files: int = 200


class SandboxUnavailable(RuntimeError):
    pass


# ==============================================================================================================
# Host probe
# ==============================================================================================================
def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _seccomp_path() -> Optional[str]:
    p = ctypes.util.find_library("seccomp")
    if p:
        return p
    for cand in ("/usr/lib/x86_64-linux-gnu/libseccomp.so.2", "/usr/lib64/libseccomp.so.2",
                 "/usr/lib/aarch64-linux-gnu/libseccomp.so.2", "/lib/x86_64-linux-gnu/libseccomp.so.2"):
        if os.path.exists(cand):
            return cand
    return None


def _user(name: str) -> Optional[pwd.struct_passwd]:
    try:
        return pwd.getpwnam(name)
    except KeyError:
        return None


def probe() -> Dict[str, Any]:
    """What this host can actually provide (no side effects)."""
    abi = _libc().syscall(444, None, 0, 1)                     # landlock_create_ruleset(NULL, 0, VERSION)
    root = os.geteuid() == 0
    dedicated = _user(SERVICE_USERS["sandbox"])
    nobody = _user("nobody")
    if root and dedicated is not None:
        mode, pw = "dedicated-uid", dedicated
    elif root and nobody is not None:
        mode, pw = "shared-unprivileged-uid", nobody
    elif root:
        mode, pw = "unavailable", None
    else:
        mode, pw = "same-uid", None
    return {"landlock_abi": max(int(abi), 0), "landlock_net": abi >= 4, "seccomp_lib": _seccomp_path(),
            "root": root, "uid_mode": mode, "uid": pw.pw_uid if pw else os.geteuid(),
            "gid": pw.pw_gid if pw else os.getegid(), "user": pw.pw_name if pw else ""}


# ==============================================================================================================
# The monitor + boot program (standard library only; run as ``python -I -S -B -c``; never imports cam_live)
# ==============================================================================================================
_BOOT = r'''
import ctypes, json, os, resource, signal, sys, time

cfg_fd, status_fd = int(sys.argv[1]), int(sys.argv[2])
with os.fdopen(cfg_fd, "r") as _f:
    CFG = json.load(_f)
libc = ctypes.CDLL(None, use_errno=True)
PR_SET_PDEATHSIG, PR_SET_CHILD_SUBREAPER, PR_SET_NO_NEW_PRIVS = 1, 36, 38
SC = None
if CFG["seccomp"]:
    try:
        SC = ctypes.CDLL(CFG["seccomp_lib"])
    except (OSError, TypeError):
        SC = None


def status(obj):
    try:
        os.write(status_fd, (json.dumps(obj) + "\n").encode())
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------- landlock
LL_CREATE, LL_ADD, LL_RESTRICT = 444, 445, 446
FS_EXEC, FS_WRITE, FS_READ, FS_READDIR = 1, 2, 4, 8
FS_ALL_ABI1 = (1 << 13) - 1
FS_REFER, FS_TRUNCATE = 1 << 13, 1 << 14
NET_BIND, NET_CONNECT = 1, 2


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def landlock():
    abi = libc.syscall(LL_CREATE, None, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock unavailable")
    fs = FS_ALL_ABI1 | (FS_REFER if abi >= 2 else 0) | (FS_TRUNCATE if abi >= 3 else 0)
    net = (NET_BIND | NET_CONNECT) if abi >= 4 else 0
    attr = RulesetAttr(fs, net)
    rfd = libc.syscall(LL_CREATE, ctypes.byref(attr), ctypes.sizeof(RulesetAttr) if abi >= 4 else 8, 0)
    if rfd < 0:
        raise RuntimeError("landlock_create_ruleset errno=%d" % ctypes.get_errno())
    file_bits = FS_EXEC | FS_WRITE | FS_READ | (FS_TRUNCATE if abi >= 3 else 0)

    def allow(path, access):
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not os.path.isdir(path):
                access &= file_bits
            pb = PathBeneath(access & fs, fd)
            if libc.syscall(LL_ADD, rfd, 1, ctypes.byref(pb), 0) != 0:
                raise RuntimeError("landlock_add_rule(%s) errno=%d" % (path, ctypes.get_errno()))
        finally:
            os.close(fd)

    for p in CFG["list_paths"]:                      # directory listing only: no file below can be read
        allow(p, FS_READDIR)
    for p in CFG["read_paths"]:
        allow(p, FS_EXEC | FS_READ | FS_READDIR)
    rw = fs & ~(1 << 6) & ~(1 << 11)                 # everything except creating character / block devices
    for p in CFG["write_paths"]:
        allow(p, rw)
    allow("/dev/null", FS_READ | FS_WRITE)
    for p in ("/dev/urandom", "/dev/random", "/dev/zero"):
        allow(p, FS_READ)
    if libc.syscall(LL_RESTRICT, rfd, 0) != 0:
        raise RuntimeError("landlock_restrict_self errno=%d" % ctypes.get_errno())
    os.close(rfd)
    return abi


# ---------------------------------------------------------------------------------------------- seccomp
DENY = ["socket", "connect", "bind", "listen", "accept", "accept4",
        "ptrace", "process_vm_readv", "process_vm_writev", "kcmp", "pidfd_open", "pidfd_send_signal",
        "pidfd_getfd", "process_madvise", "process_mrelease",
        "mount", "umount2", "pivot_root", "chroot", "open_tree", "move_mount", "fsopen", "fsconfig", "fsmount",
        "fspick", "mount_setattr", "unshare", "setns",
        "bpf", "perf_event_open", "userfaultfd", "io_uring_setup", "io_uring_enter", "io_uring_register",
        "keyctl", "add_key", "request_key",
        "kexec_load", "kexec_file_load", "init_module", "finit_module", "delete_module", "reboot", "swapon",
        "swapoff", "sethostname", "setdomainname", "acct", "quotactl", "syslog", "iopl", "ioperm",
        "clock_settime", "settimeofday", "adjtimex", "clock_adjtime",
        "name_to_handle_at", "open_by_handle_at", "personality",
        "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "setgroups", "setfsuid", "setfsgid",
        "setsid", "setpgid"]
CLONE_NS = [0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000]
AF_UNIX = 1


def seccomp():
    sc = SC
    if sc is None:
        raise RuntimeError("libseccomp unavailable")
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sc.seccomp_load.argtypes = [ctypes.c_void_p]
    ctx = sc.seccomp_init(0x7FFF0000)                # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    eperm, enosys = 0x00050000 | 1, 0x00050000 | 38
    for name in DENY:
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:
            continue
        if sc.seccomp_rule_add(ctx, eperm, nr, 0) != 0:
            raise RuntimeError("seccomp_rule_add(%s) failed" % name)
    nr = sc.seccomp_syscall_resolve_name(b"clone3")   # arguments are behind a pointer: make libc fall back to clone
    if nr >= 0 and sc.seccomp_rule_add(ctx, enosys, nr, 0) != 0:
        raise RuntimeError("seccomp_rule_add(clone3) failed")

    class ArgCmp(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64),
                    ("datum_b", ctypes.c_uint64)]
    sc.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
                                          ctypes.POINTER(ArgCmp)]
    nr = sc.seccomp_syscall_resolve_name(b"clone")
    for bit in CLONE_NS:                              # SCMP_CMP_MASKED_EQ: (flags & bit) == bit -> new namespace
        cmp = (ArgCmp * 1)(ArgCmp(0, 7, bit, bit))
        if sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
            raise RuntimeError("seccomp clone rule failed")
    nr = sc.seccomp_syscall_resolve_name(b"socketpair")
    cmp = (ArgCmp * 1)(ArgCmp(0, 1, AF_UNIX, 0))      # SCMP_CMP_NE: only a local AF_UNIX pair may be created
    if sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
        raise RuntimeError("seccomp socketpair rule failed")
    nr = sc.seccomp_syscall_resolve_name(b"ioctl")
    for cmd in (0x5412, 0x541C):                      # TIOCSTI, TIOCLINUX
        cmp = (ArgCmp * 1)(ArgCmp(1, 4, cmd, 0))      # SCMP_CMP_EQ
        if sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
            raise RuntimeError("seccomp ioctl rule failed")
    if sc.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp_load failed")
    return True


# ---------------------------------------------------------------------------------------------- child
def child(ready_w):
    info = {}
    try:
        os.setsid()
        signal.pthread_sigmask(signal.SIG_SETMASK, [])
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD, signal.SIGPIPE, signal.SIGXFSZ):
            signal.signal(s, signal.SIG_DFL)
        r = CFG["rlimits"]
        resource.setrlimit(resource.RLIMIT_AS, (r["as"], r["as"]))
        resource.setrlimit(resource.RLIMIT_CPU, (r["cpu"], r["cpu"] + 2))
        resource.setrlimit(resource.RLIMIT_NPROC, (r["nproc"], r["nproc"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (r["fsize"], r["fsize"]))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        keep = set(CFG["keep_fds"]) | {0, 1, 2, ready_w}
        for name in os.listdir("/proc/self/fd"):      # before the uid drop: /proc/self is root-owned afterwards
            fd = int(name)
            if fd not in keep:
                try:
                    os.close(fd)
                except OSError:
                    pass
        resource.setrlimit(resource.RLIMIT_NOFILE, (r["nofile"], r["nofile"]))
        if CFG["uid"] is not None and os.getuid() == 0:
            os.setgroups([])
            os.setgid(CFG["gid"])
            os.setuid(CFG["uid"])
        if os.getuid() == 0 or os.geteuid() == 0:
            raise RuntimeError("refusing to run generated code as root")
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise RuntimeError("prctl(NO_NEW_PRIVS) failed")
        libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        os.chdir(CFG["cwd"])
        os.umask(0o077)
        info["uid"] = os.getuid()
        info["landlock_abi"] = landlock() if CFG["landlock"] else 0
        info["seccomp"] = seccomp() if CFG["seccomp"] else False
        info["no_new_privs"] = True
        os.write(ready_w, (json.dumps({"ok": True, "info": info}) + "\n").encode())
        os.execve(CFG["argv"][0], CFG["argv"], CFG["env"])
    except BaseException as e:  # noqa: BLE001 - report every setup failure, then die
        try:
            os.write(ready_w, (json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}) + "\n").encode())
        except OSError:
            pass
        os._exit(127)


# ---------------------------------------------------------------------------------------------- monitor
def scan(sid):
    """(tasks, resident bytes, pids) of the live processes in session ``sid``."""
    tasks, rss, pids = 0, 0, []
    page = os.sysconf("SC_PAGE_SIZE")
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % name, "rb") as f:
                rest = f.read().rsplit(b")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(rest[3]) != sid or rest[0] == b"Z":
            continue
        tasks += int(rest[17])
        rss += int(rest[21]) * page
        pids.append(int(name))
    return tasks, rss, pids


def monitor():
    libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    watch = {signal.SIGCHLD, signal.SIGTERM, signal.SIGINT}
    signal.pthread_sigmask(signal.SIG_BLOCK, watch)
    harness = CFG["harness_pid"]
    if os.getppid() != harness:
        status({"event": "exit", "error": "harness exited before the job started"})
        return
    ready_r, ready_w = os.pipe2(os.O_CLOEXEC)
    t0 = time.monotonic()
    pid = os.fork()
    if pid == 0:
        os.close(ready_r)
        child(ready_w)
        os._exit(127)
    os.close(ready_w)
    buf = b""
    while True:
        chunk = os.read(ready_r, 65536)
        if not chunk:
            break
        buf += chunk
    os.close(ready_r)
    try:
        ready = json.loads(buf.decode() or "{}")
    except ValueError:
        ready = {}
    status({"event": "started", "pgid": pid, "ready": ready})
    lim = CFG["monitor"]
    deadline = t0 + lim["wall_s"]
    st = {"main": None, "killed": False, "reason": None, "kill_at": 0.0, "tasks": 0, "rss": 0,
          "cancelled": False, "timed_out": False}

    def killgroup(why):
        if not st["killed"]:
            st["killed"], st["reason"], st["kill_at"] = True, why, time.monotonic()
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    next_scan = 0.0
    while True:
        try:
            while True:
                p, wst, _ru = os.wait4(-1, os.WNOHANG)
                if p == 0:
                    break
                if p == pid:
                    st["main"] = wst
        except ChildProcessError:
            break                                     # every descendant has been reaped
        now = time.monotonic()
        first = not st["killed"]
        if st["main"] is not None:
            killgroup(None)                           # the program ended: nothing it started may outlive it
        elif st["cancelled"] or os.getppid() != harness:
            st["cancelled"] = True
            killgroup("cancelled")
        elif now >= deadline:
            st["timed_out"] = True
            killgroup("timeout")
        elif now >= next_scan:
            next_scan = now + 0.1
            tasks, rss, _ = scan(pid)
            st["tasks"], st["rss"] = max(st["tasks"], tasks), max(st["rss"], rss)
            if rss > lim["mem_total"]:
                killgroup("memory_total")
            elif tasks > lim["nproc"]:
                killgroup("nproc")
        if st["killed"]:
            if first:
                continue                              # reap at once
            if now - st["kill_at"] > 10.0:
                break                                 # something unkillable: reported below as leftover
        wait = 0.02 if st["killed"] else max(0.0, min(0.05, deadline - now))
        got = signal.sigtimedwait(watch, wait)
        if got is not None and got.si_signo in (signal.SIGTERM, signal.SIGINT):
            st["cancelled"] = True
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    _, _, left = scan(pid)
    out = {"event": "exit", "wall_s": time.monotonic() - t0, "cpu_s": ru.ru_utime + ru.ru_stime,
           "max_rss_kb": ru.ru_maxrss, "peak_tree_rss_kb": st["rss"] // 1024, "peak_tasks": st["tasks"],
           "timed_out": st["timed_out"], "cancelled": st["reason"] == "cancelled",
           "killed_for": st["reason"], "leftover": len(left), "exit_code": None, "signal": None}
    if st["main"] is not None:
        if os.WIFEXITED(st["main"]):
            out["exit_code"] = os.WEXITSTATUS(st["main"])
        elif os.WIFSIGNALED(st["main"]):
            out["signal"] = os.WTERMSIG(st["main"])
    status(out)

monitor()
'''

# ==============================================================================================================
# In-sandbox network shim (``sitecustomize`` on PYTHONPATH of scoped-network jobs; standard library only)
# ==============================================================================================================
_SHIM = r'''
"""Brokered sockets for sandboxed Python. The kernel denies socket(); every outbound TCP stream is requested
from the egress broker over the inherited descriptor CAM_EGRESS_FD and authorized there, one by one."""
import os as _os

_FD = _os.environ.get("CAM_EGRESS_FD", "")
if _FD.isdigit():
    import json as _json
    import socket as _socket_mod
    import threading as _threading

    _Real = _socket_mod.socket
    _ctl_lock = _threading.Lock()
    _STREAM = _socket_mod.SOCK_STREAM

    def _request(host, port, timeout):
        a, b = _socket_mod.socketpair(_socket_mod.AF_UNIX, _STREAM)
        try:
            msg = _json.dumps({"op": "connect", "host": str(host), "port": int(port)}).encode()
            ctl = _Real(fileno=_os.dup(int(_FD)))
            try:
                with _ctl_lock:
                    _socket_mod.send_fds(ctl, [msg], [b.fileno()])
            finally:
                ctl.close()
        finally:
            b.close()
        a.settimeout(timeout if timeout else 60.0)
        line = b""
        try:
            while not line.endswith(b"\n") and len(line) < 512:
                ch = a.recv(1)
                if not ch:
                    break
                line += ch
        except OSError:
            a.close()
            raise
        if not line.startswith(b"OK"):
            a.close()
            text = line.decode("utf-8", "replace").strip() or "egress broker closed the request"
            if text.startswith("ERR denied"):
                raise PermissionError(13, "outbound connection not authorized: " + text[4:])
            raise ConnectionRefusedError(111, text)
        return a.detach()

    class socket(_Real):
        """AF_INET/AF_INET6 stream sockets are opened lazily through the broker at connect()."""

        def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
            fam = _socket_mod.AF_INET if family == -1 else family
            typ = _STREAM if type == -1 else type
            base = typ & 0xF
            if fileno is None and fam in (_socket_mod.AF_INET, _socket_mod.AF_INET6) and base == _STREAM:
                self._cam = {"family": fam, "type": typ, "proto": 0 if proto == -1 else proto,
                             "timeout": _socket_mod.getdefaulttimeout(), "peer": None, "open": False}
                if typ & getattr(_socket_mod, "SOCK_NONBLOCK", 0):
                    self._cam["timeout"] = 0.0
                return
            self._cam = None
            _Real.__init__(self, family, type, proto, fileno)

        def _pending(self):
            st = self.__dict__.get("_cam")
            return st is not None and not st["open"]

        @property
        def family(self):
            st = self.__dict__.get("_cam")
            return _socket_mod.AddressFamily(st["family"]) if st else _Real.family.fget(self)

        @property
        def type(self):
            st = self.__dict__.get("_cam")
            return _socket_mod.SocketKind(st["type"] & 0xF) if st else _Real.type.fget(self)

        @property
        def proto(self):
            st = self.__dict__.get("_cam")
            return st["proto"] if st else _Real.proto.fget(self)

        def settimeout(self, value):
            if self._pending():
                self._cam["timeout"] = value
            else:
                _Real.settimeout(self, value)

        def gettimeout(self):
            return self._cam["timeout"] if self._pending() else _Real.gettimeout(self)

        def setblocking(self, flag):
            self.settimeout(None if flag else 0.0)

        def getblocking(self):
            return self.gettimeout() != 0.0

        def setsockopt(self, *a):
            st = self.__dict__.get("_cam")
            if st is None:
                return _Real.setsockopt(self, *a)
            return None                               # TCP options have no meaning on the brokered stream

        def getsockopt(self, *a):
            st = self.__dict__.get("_cam")
            if st is None or (st["open"] and a[:2] == (_socket_mod.SOL_SOCKET, _socket_mod.SO_ERROR)):
                return _Real.getsockopt(self, *a)
            return 0

        def bind(self, address):
            if self.__dict__.get("_cam") is None:
                return _Real.bind(self, address)
            raise PermissionError(13, "listening sockets are not available in the sandbox")

        def fileno(self):
            return -1 if self._pending() else _Real.fileno(self)

        def close(self):
            if self._pending():
                self._cam["open"] = None
                return
            _Real.close(self)

        def getpeername(self):
            st = self.__dict__.get("_cam")
            if st is None:
                return _Real.getpeername(self)
            if not st["open"]:
                raise OSError(107, "Transport endpoint is not connected")
            return st["peer"]

        def getsockname(self):
            st = self.__dict__.get("_cam")
            if st is None:
                return _Real.getsockname(self)
            return ("0.0.0.0", 0)

        def connect(self, address):
            st = self.__dict__.get("_cam")
            if st is None:
                return _Real.connect(self, address)
            if st["open"]:
                raise OSError(106, "Transport endpoint is already connected")
            host, port = address[0], address[1]
            timeout = st["timeout"]
            fd = _request(host, port, timeout if timeout else None)
            _Real.__init__(self, st["family"], st["type"] & 0xF, st["proto"], fd)
            st["open"], st["peer"] = True, (host, port)
            _Real.settimeout(self, timeout)

        def connect_ex(self, address):
            try:
                self.connect(address)
                return 0
            except OSError as e:
                return e.errno or 111

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        """No resolver inside the sandbox: names travel to the broker, which resolves after authorization."""
        if host is None:
            host = "localhost"
        if isinstance(host, bytes):
            host = host.decode()
        if isinstance(port, str):
            port = int(port) if port.isdigit() else {"http": 80, "https": 443}.get(port, 0)
        fam = _socket_mod.AF_INET6 if (":" in host and family != _socket_mod.AF_INET) else _socket_mod.AF_INET
        if family not in (0, fam):
            fam = family
        sa = (host, port or 0) if fam == _socket_mod.AF_INET else (host, port or 0, 0, 0)
        return [(fam, _STREAM, 6, "", sa)]

    def create_connection(address, timeout=_socket_mod._GLOBAL_DEFAULT_TIMEOUT, source_address=None, **_kw):
        s = socket(_socket_mod.AF_INET, _STREAM)
        if timeout is not _socket_mod._GLOBAL_DEFAULT_TIMEOUT:
            s.settimeout(timeout)
        s.connect(address)
        return s

    _socket_mod.socket = socket
    _socket_mod.SocketType = socket
    _socket_mod.getaddrinfo = getaddrinfo
    _socket_mod.create_connection = create_connection
'''


# ==============================================================================================================
# Safe file access below a directory (the harness may be root: never follow a path out of the workspace)
# ==============================================================================================================
class _OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


_RESOLVE_NO_MAGICLINKS, _RESOLVE_BENEATH = 0x02, 0x08
_SYS_OPENAT2 = 437


def _safe_rel(path: str) -> str:
    if not isinstance(path, str) or "\x00" in path or path == "":
        raise ValueError(f"invalid path {path!r}")
    if os.path.isabs(path):
        raise ValueError("path must be relative to the workspace")
    p = os.path.normpath(path)
    if p == ".." or p.startswith("../"):
        raise ValueError("path escapes the workspace")
    return p


def _open_beneath(dir_fd: int, rel: str, flags: int, mode: int = 0) -> int:
    """openat2(RESOLVE_BENEATH): symlinks may be followed only while they stay below ``dir_fd``."""
    how = _OpenHow(flags | os.O_CLOEXEC, mode if flags & os.O_CREAT else 0,
                   _RESOLVE_BENEATH | _RESOLVE_NO_MAGICLINKS)
    libc = _libc()
    fd = libc.syscall(_SYS_OPENAT2, dir_fd, os.fsencode(rel), ctypes.byref(how), ctypes.sizeof(how))
    if fd >= 0:
        return fd
    err = ctypes.get_errno()
    if err != errno.ENOSYS:
        raise OSError(err, os.strerror(err), rel)
    cur = os.dup(dir_fd)                               # kernel < 5.6: walk, refusing every symlink
    try:
        parts = [x for x in rel.split("/") if x not in ("", ".")]
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            nxt = os.open(part, (flags if last else os.O_RDONLY | os.O_DIRECTORY) | os.O_NOFOLLOW | os.O_CLOEXEC,
                          mode, dir_fd=cur)
            os.close(cur)
            cur = nxt
        return cur
    except BaseException:
        os.close(cur)
        raise


# ==============================================================================================================
# Egress broker (scoped network)
# ==============================================================================================================
class EgressBroker:
    """Hands authorized TCP streams to a sandboxed job. ``authorize(host, port, ip, private) -> (ok, reason)``
    is called once for every connection request, before the name is resolved (``ip`` is set only when the job
    gave an address literal). Private destinations are refused before and after resolution."""

    def __init__(self, authorize: Callable[[str, int, str, bool], Tuple[bool, str]], *,
                 allow_private: bool = False, max_connections: int = 16, max_bytes: int = 256 * 2**20,
                 connect_timeout_s: float = 10.0):
        self._authorize = authorize
        self.allow_private = allow_private
        self.max_connections = max_connections
        self.max_bytes = max_bytes
        self.connect_timeout_s = connect_timeout_s
        self._parent, self._child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._child.set_inheritable(True)
        self._lock = threading.Lock()
        self._active = 0
        self._bytes = 0
        self._closed = threading.Event()
        self._threads: List[threading.Thread] = []
        self.log: List[Dict[str, Any]] = []            # one entry per connection request
        self._thread = threading.Thread(target=self._serve, name="cam-egress", daemon=True)
        self._thread.start()

    @property
    def child_fd(self) -> int:
        return self._child.fileno()

    def after_spawn(self) -> None:
        """Close our copy of the job's end (the job holds it now)."""
        try:
            self._child.close()
        except OSError:
            pass

    def close(self) -> None:
        self._closed.set()
        for s in (self._parent, self._child):
            try:
                s.close()
            except OSError:
                pass
        for t in list(self._threads):
            t.join(timeout=2.0)

    # ---------------------------------------------------------------------------------------- internals
    def _serve(self) -> None:
        self._parent.settimeout(0.2)
        while not self._closed.is_set():
            try:
                msg, fds, _flags, _addr = socket.recv_fds(self._parent, 4096, 4)
            except socket.timeout:
                continue
            except OSError:
                return
            if not msg and not fds:
                if self._closed.wait(0.05):
                    return
                continue
            if len(fds) != 1:
                for fd in fds:
                    os.close(fd)
                continue
            peer = socket.socket(fileno=fds[0])
            t = threading.Thread(target=self._handle, args=(msg, peer), name="cam-egress-conn", daemon=True)
            self._threads.append(t)
            t.start()

    @staticmethod
    def _is_private(ip: str) -> bool:
        a = ipaddress.ip_address(ip)
        return bool(a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved
                    or a.is_unspecified)

    def _refuse(self, peer: socket.socket, entry: Dict[str, Any], kind: str, reason: str) -> None:
        entry.update(status=kind, reason=reason, ended=time.time())
        try:
            peer.sendall(f"ERR {kind} {reason}"[:400].replace("\n", " ").encode() + b"\n")
        except OSError:
            pass
        peer.close()

    def _handle(self, msg: bytes, peer: socket.socket) -> None:
        entry: Dict[str, Any] = {"host": "", "port": 0, "ip": "", "status": "error", "reason": "",
                                 "bytes_out": 0, "bytes_in": 0, "started": time.time(), "ended": 0.0}
        with self._lock:
            self.log.append(entry)
        try:
            req = json.loads(msg.decode())
            host, port = str(req["host"]), int(req["port"])
            if req.get("op") != "connect" or not (0 < port < 65536) or not re.fullmatch(r"[A-Za-z0-9._:\-]{1,253}", host):
                raise ValueError("malformed request")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return self._refuse(peer, entry, "invalid", "malformed connection request")
        entry.update(host=host, port=port)
        with self._lock:
            if self._active >= self.max_connections:
                return self._refuse(peer, entry, "limit", "too many open connections")
            self._active += 1
        try:
            literal: Optional[bool] = None
            try:
                literal = self._is_private(host)       # an address literal: nothing to resolve
                entry.update(ip=host, private=literal)
            except ValueError:
                pass
            if literal and not self.allow_private:
                return self._refuse(peer, entry, "denied", "private, loopback and link-local destinations are closed")
            # authorization comes before name resolution: a refused name must not even reach a resolver
            ok, reason = self._authorize(host, port, host if literal is not None else "", bool(literal))
            if not ok:
                return self._refuse(peer, entry, "denied", reason or "not authorized")
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError as e:
                return self._refuse(peer, entry, "unresolved", f"cannot resolve {host}: {e}")
            fam, _, _, _, sa = infos[0]
            private = self._is_private(sa[0])
            entry.update(ip=sa[0], private=private)
            if private and not self.allow_private:
                return self._refuse(peer, entry, "denied", "the name resolves to a private or loopback address")
            try:
                up = socket.socket(fam, socket.SOCK_STREAM)
                up.settimeout(self.connect_timeout_s)
                up.connect(sa)                         # the address that was authorized, not a second lookup
            except OSError as e:
                return self._refuse(peer, entry, "unreachable", f"{type(e).__name__}: {e}")
            entry["status"] = "open"
            try:
                peer.sendall(b"OK\n")
                self._relay(peer, up, entry)
            finally:
                up.close()
                peer.close()
                entry["status"] = "closed" if entry["status"] == "open" else entry["status"]
                entry["ended"] = time.time()
        finally:
            with self._lock:
                self._active -= 1

    def _relay(self, a: socket.socket, b: socket.socket, entry: Dict[str, Any]) -> None:
        sel = selectors.DefaultSelector()
        a.setblocking(True)
        b.setblocking(True)
        sel.register(a, selectors.EVENT_READ, (b, "bytes_out"))
        sel.register(b, selectors.EVENT_READ, (a, "bytes_in"))
        open_dirs = 2
        try:
            while open_dirs and not self._closed.is_set():
                for key, _ in sel.select(0.2):
                    dst, counter = key.data
                    try:
                        data = key.fileobj.recv(65536)
                    except OSError:
                        data = b""
                    if not data:
                        sel.unregister(key.fileobj)
                        open_dirs -= 1
                        try:
                            dst.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    with self._lock:
                        self._bytes += len(data)
                        over = self._bytes > self.max_bytes
                    if over:
                        entry.update(status="limit", reason="egress byte limit reached")
                        return
                    try:
                        dst.sendall(data)
                    except OSError:
                        return
                    entry[counter] += len(data)
        finally:
            sel.close()


# ==============================================================================================================
# Results
# ==============================================================================================================
@dataclass
class RunResult:
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    killed_for: Optional[str] = None      # "timeout" | "cancelled" | "memory_total" | "nproc" | None
    wall_s: float = 0.0
    cpu_s: float = 0.0
    max_rss_kb: int = 0
    peak_tree_rss_kb: int = 0
    peak_tasks: int = 0
    leftover: int = 0                     # processes that survived the group kill (must be 0)
    setup_error: str = ""                 # the sandbox could not be established; nothing ran
    isolation: Dict[str, Any] = field(default_factory=dict)   # what the job's first process reported

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.setup_error

    @property
    def limit_hit(self) -> str:
        """Name of the resource limit that ended the run, or ''. ``timeout`` / ``memory_total`` / ``nproc`` are
        the monitor's own decisions and ``cpu`` / ``fsize`` come from the terminating signal; ``memory`` (and
        ``fsize`` for programs that ignore SIGXFSZ, as Python does) is inferred from the program's error output."""
        if self.killed_for in ("timeout", "memory_total", "nproc"):
            return self.killed_for
        if self.signal == signal.SIGXCPU or self.exit_code == 128 + signal.SIGXCPU or (
                self.signal == signal.SIGKILL and not self.cancelled and self.killed_for is None):
            return "cpu"
        if self.signal == signal.SIGXFSZ or self.exit_code == 128 + signal.SIGXFSZ:
            return "fsize"
        if self.exit_code not in (0, None):
            tail = self.stderr[-4000:]
            if "MemoryError" in tail or "Cannot allocate memory" in tail or "bad_alloc" in tail:
                return "memory"
            if "File too large" in tail or "File size limit exceeded" in tail:
                return "fsize"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _Capture(threading.Thread):
    """Drains a pipe, keeping the first 3/4 and the last 1/4 of ``cap`` bytes; optionally tees to a file."""

    def __init__(self, fd: int, cap: int, tee_path: str = "", tee_cap: int = 0):
        super().__init__(name="cam-sbx-capture", daemon=True)
        self.fd, self.cap = fd, max(int(cap), 256)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self._head_cap = self.cap * 3 // 4
        self._tail_cap = self.cap - self._head_cap
        self._tee = open(tee_path, "ab") if tee_path else None
        self._tee_left = tee_cap
        self.lock = threading.Lock()
        self.start()

    def run(self) -> None:
        try:
            while True:
                chunk = os.read(self.fd, 65536)
                if not chunk:
                    break
                with self.lock:
                    self.total += len(chunk)
                    room = self._head_cap - len(self.head)
                    if room > 0:
                        self.head += chunk[:room]
                        chunk_tail = chunk[room:]
                    else:
                        chunk_tail = chunk
                    if chunk_tail:
                        self.tail += chunk_tail
                        if len(self.tail) > self._tail_cap:
                            del self.tail[:len(self.tail) - self._tail_cap]
                if self._tee is not None and self._tee_left > 0:
                    part = chunk[:self._tee_left]
                    self._tee.write(part)
                    self._tee.flush()
                    self._tee_left -= len(part)
        except OSError:
            pass
        finally:
            os.close(self.fd)
            if self._tee is not None:
                self._tee.close()

    def text(self) -> Tuple[str, bool]:
        with self.lock:
            kept = len(self.head) + len(self.tail)
            truncated = self.total > kept
            if truncated:
                data = bytes(self.head) + f"\n...[{self.total - kept} bytes truncated]...\n".encode() + bytes(self.tail)
            else:
                data = bytes(self.head) + bytes(self.tail)
        return data.decode("utf-8", "replace"), truncated


class SandboxProcess:
    """A running sandboxed execution. ``wait()`` returns the :class:`RunResult`; ``cancel()`` kills the job."""

    def __init__(self, proc: subprocess.Popen, status_r: int, out: _Capture, err: _Capture, wall_s: float,
                 broker: Optional[EgressBroker] = None):
        self._proc, self._status_r = proc, status_r
        self._out, self._err = out, err
        self._wall_s = wall_s
        self._broker = broker
        self._buf = b""
        self._events: List[Dict[str, Any]] = []
        self._result: Optional[RunResult] = None
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.pgid: Optional[int] = None
        self.monitor_pid = proc.pid

    # ------------------------------------------------------------------------------------------ status
    def _pump(self, timeout: float) -> None:
        sel = selectors.DefaultSelector()
        sel.register(self._status_r, selectors.EVENT_READ)
        try:
            if sel.select(timeout):
                chunk = os.read(self._status_r, 65536)
                if chunk:
                    self._buf += chunk
                    while b"\n" in self._buf:
                        line, _, self._buf = self._buf.partition(b"\n")
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        self._events.append(ev)
                        if ev.get("event") == "started":
                            self.pgid = ev.get("pgid")
                else:
                    self._events.append({"event": "eof"})
        finally:
            sel.close()

    def _final(self) -> Optional[Dict[str, Any]]:
        for ev in self._events:
            if ev.get("event") == "exit":
                return ev
        return None

    def running(self) -> bool:
        with self._lock:
            if self._result is not None:
                return False
            self._pump(0.0)
            return self._final() is None and not any(e.get("event") == "eof" for e in self._events)

    def cancel(self) -> None:
        """Ask the monitor to kill the job (it reaps everything and reports ``cancelled``)."""
        try:
            self._proc.send_signal(signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    def _hard_kill(self) -> None:
        if self.pgid:
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            self._proc.kill()
        except (ProcessLookupError, OSError):
            pass

    def wait(self, cancel: Any = None, deadline_monotonic: Optional[float] = None) -> RunResult:
        """Block until the job has ended and every process of it is gone. ``cancel`` is an Event-like object
        polled while waiting; ``deadline_monotonic`` cancels the job when it passes."""
        with self._lock:
            if self._result is not None:
                return self._result
            asked = False
            hard_at = self._t0 + self._wall_s + 20.0
            while self._final() is None and not any(e.get("event") == "eof" for e in self._events):
                self._pump(0.05)
                now = time.monotonic()
                if not asked and ((cancel is not None and cancel.is_set())
                                  or (deadline_monotonic is not None and now >= deadline_monotonic)):
                    asked = True
                    self.cancel()
                    hard_at = min(hard_at, now + 15.0)
                if now >= hard_at:
                    self._hard_kill()
                    break
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._hard_kill()
                self._proc.wait(timeout=5)
            self._pump(0.0)
            os.close(self._status_r)
            self._out.join(timeout=5)
            self._err.join(timeout=5)
            if self._broker is not None:
                self._broker.close()
            fin = self._final() or {}
            started = next((e for e in self._events if e.get("event") == "started"), {})
            ready = started.get("ready") or {}
            res = RunResult(isolation=ready.get("info") or {})
            res.stdout, res.stdout_truncated = self._out.text()
            res.stderr, res.stderr_truncated = self._err.text()
            res.stdout_bytes, res.stderr_bytes = self._out.total, self._err.total
            if not fin:
                res.setup_error = "" if asked else "sandbox monitor ended without a result: " + res.stderr[-600:]
            elif fin.get("error"):
                res.setup_error = str(fin["error"])
            else:
                for k in ("exit_code", "signal", "timed_out", "cancelled", "killed_for", "wall_s", "cpu_s",
                          "max_rss_kb", "peak_tree_rss_kb", "peak_tasks", "leftover"):
                    setattr(res, k, fin.get(k, getattr(res, k)))
                if ready and not ready.get("ok", False):
                    res.setup_error = str(ready.get("error", "sandbox setup failed"))
            if asked and not res.cancelled and res.exit_code is None and not res.timed_out:
                res.cancelled = True
            self._result = res
            return res


# ==============================================================================================================
# The sandbox
# ==============================================================================================================
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_RESERVED_ENV = {"PATH", "HOME", "TMPDIR", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "CAM_EGRESS_FD",
                 "PYTHONSTARTUP", "PYTHONHOME", "BASH_ENV", "ENV"}
_ETC_READ = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime", "/etc/passwd",
             "/etc/group", "/etc/nsswitch.conf", "/etc/ssl/certs", "/etc/ca-certificates", "/etc/alternatives",
             "/etc/mime.types", "/etc/shells", "/etc/os-release")
_SYS_READ = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/proc/cpuinfo", "/proc/meminfo",
             "/proc/loadavg", "/proc/uptime", "/proc/version", "/proc/filesystems", "/sys/devices/system/cpu")


class Sandbox:
    """A persistent workspace directory plus the machinery to run programs confined to it."""

    def __init__(self, workspace: Optional[str] = None, cfg: Optional[SandboxConfig] = None):
        self.cfg = cfg or SandboxConfig()
        self._probe = probe()
        self._lock = threading.Lock()
        self._seq = 0
        self._procs: List[SandboxProcess] = []
        reason = self._unavailable_reason()
        self.available = reason == ""
        self.unavailable_reason = reason
        self.uid, self.gid = self._probe["uid"], self._probe["gid"]
        self.drops_uid = self._probe["uid_mode"] in ("dedicated-uid", "shared-unprivileged-uid")
        self.workspace = os.path.abspath(workspace or os.path.join(self.cfg.base_dir, "ws", "default"))
        self.support = os.path.join(self.cfg.base_dir, "_support")
        self._root_fd: Optional[int] = None
        if self.available:
            self._prepare()

    # ------------------------------------------------------------------------------------------ setup
    def _unavailable_reason(self) -> str:
        p, cfg = self._probe, self.cfg
        if p["uid_mode"] == "unavailable":
            return "running as root and neither the sandbox user nor 'nobody' exists: refusing to run code as root"
        if p["uid_mode"] == "same-uid" and not cfg.allow_same_uid:
            return ("not root: generated code would run under the harness uid; start the executor as root (uid drop "
                    "to the sandbox user) or set SandboxConfig.allow_same_uid=True to accept the weaker mode")
        if cfg.landlock and p["landlock_abi"] < 1:
            return "Landlock is not available in this kernel"
        if cfg.seccomp and not p["seccomp_lib"]:
            return "libseccomp is not installed"
        return ""

    def _traversable(self, path: str) -> bool:
        cur = os.path.dirname(path)
        while True:
            st = os.stat(cur)
            ok = bool(st.st_mode & 0o001) or (st.st_uid == self.uid and bool(st.st_mode & 0o100)) \
                or (st.st_gid == self.gid and bool(st.st_mode & 0o010))
            if not ok:
                return False
            if cur == "/":
                return True
            cur = os.path.dirname(cur)

    def _mkdir_owned(self, path: str, mode: int, owned: bool) -> None:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, mode)
        if owned and self.drops_uid:
            os.chown(path, self.uid, self.gid)

    def _prepare(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.base_dir, exist_ok=True)
        if os.stat(cfg.base_dir).st_uid == os.geteuid():
            os.chmod(cfg.base_dir, 0o711)
        os.makedirs(os.path.dirname(self.workspace), exist_ok=True)
        if os.path.dirname(self.workspace).startswith(cfg.base_dir):
            os.chmod(os.path.dirname(self.workspace), 0o711)
        fresh = not os.path.isdir(self.workspace)
        if not fresh and self.drops_uid and os.lstat(self.workspace).st_uid != self.uid:
            for root, dirs, files in os.walk(self.workspace):      # the sandbox user changed: hand the tree over
                for x in dirs + files:
                    os.lchown(os.path.join(root, x), self.uid, self.gid)
        self._mkdir_owned(self.workspace, 0o700, True)
        if not self._traversable(self.workspace):
            self.available = False
            self.unavailable_reason = (f"workspace {self.workspace} is below a directory the sandbox uid "
                                       f"{self.uid} cannot traverse")
            return
        self._root_fd = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        for sub in (".tmp", ".cam", ".cam/exec"):
            self._mkdirs(sub)
        # support directory: harness-owned, readable by the sandbox, never writable by it
        for sub in ("", "bin", "shim", "pkgs"):
            d = os.path.join(self.support, sub)
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o755)
        self.python = os.path.realpath(sys.executable)
        for name in ("python", "python3"):
            link = os.path.join(self.support, "bin", name)
            if not os.path.lexists(link):
                try:
                    os.symlink(self.python, link)
                except FileExistsError:
                    pass
        shim = os.path.join(self.support, "shim", "sitecustomize.py")
        digest = hashlib.sha256(_SHIM.encode()).hexdigest()
        marker = os.path.join(self.support, "shim", ".sha256")
        if not os.path.exists(marker) or open(marker).read() != digest:
            tmp = f"{shim}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(_SHIM)
            os.chmod(tmp, 0o644)
            os.replace(tmp, shim)
            with open(marker, "w") as f:
                f.write(digest)
        self._pkg_reads: List[str] = []
        for pkg in cfg.packages:
            spec = importlib.util.find_spec(pkg)
            if spec is None or not spec.submodule_search_locations:
                continue
            src = os.path.realpath(list(spec.submodule_search_locations)[0])
            dst = os.path.join(self.support, "pkgs", pkg)
            if not os.path.lexists(dst):
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
            self._pkg_reads.append(src)
            site = os.path.dirname(src)
            for extra in os.listdir(site):
                low = extra.lower()
                if low.startswith(pkg.lower()) and low.endswith(".libs"):
                    dst2 = os.path.join(self.support, "pkgs", extra)
                    if not os.path.lexists(dst2):
                        try:
                            os.symlink(os.path.join(site, extra), dst2)
                        except FileExistsError:
                            pass
                    self._pkg_reads.append(os.path.join(site, extra))
        self._read_paths = self._compute_read_paths()

    def _compute_read_paths(self) -> List[str]:
        reads: List[str] = [p for p in _SYS_READ + _ETC_READ if os.path.exists(p)]
        base = sys.base_prefix
        stdlib = os.path.dirname(os.path.realpath(os.__file__))
        self._list_paths = [stdlib]                      # the import system lists it; entries are granted below
        for name in sorted(os.listdir(stdlib)):          # the standard library, never site-packages
            if name in ("site-packages", "dist-packages"):
                continue
            reads.append(os.path.join(stdlib, name))
        reads.append(self.python)
        lib = os.path.join(base, "lib")
        if os.path.isdir(lib) and not lib.startswith("/usr"):
            for name in sorted(os.listdir(lib)):
                if ".so" in name:
                    reads.append(os.path.join(lib, name))
        reads += self._pkg_reads + [self.support] + [p for p in self.cfg.extra_read_paths if os.path.exists(p)]
        return reads

    # ------------------------------------------------------------------------------------------ capabilities
    def capabilities(self) -> Dict[str, Any]:
        p, cfg = self._probe, self.cfg
        mode = p["uid_mode"]
        kernel = ["rlimits: address space and CPU seconds per process, file size per file, open files, task "
                  "ceiling for the sandbox uid, no core dumps", "no_new_privs"]
        weak: List[str] = []
        if cfg.landlock and p["landlock_abi"] >= 1:
            kernel.append(f"Landlock ABI {p['landlock_abi']}: files can be opened read-only in the interpreter, "
                          "standard library, approved packages and system binaries, read-write only in the workspace")
        else:
            weak.append("Landlock is off: the filesystem is protected only by file ownership and modes")
        if cfg.seccomp and p["seccomp_lib"]:
            kernel.append("seccomp: no socket() of any family, no connect/bind/listen/accept, no ptrace, "
                          "process_vm_*, mount, namespaces, bpf, keyrings, io_uring, kernel modules; no setsid/setpgid "
                          "(the job cannot leave the process group that is killed)")
        else:
            weak.append("seccomp is off: sockets and kernel interfaces are not filtered")
        if mode == "dedicated-uid":
            kernel.append(f"dedicated uid {p['uid']} ({p['user']}): file ownership separates the job from every "
                          "service uid")
        elif mode == "shared-unprivileged-uid":
            weak.append(f"uid {p['uid']} ({p['user']}) is shared with any other process of that user on this host "
                        f"(the dedicated user {SERVICE_USERS['sandbox']!r} does not exist)")
        elif mode == "same-uid":
            weak.append("no uid drop (not root): the job has the harness uid; it can signal harness processes, and "
                        "file metadata or truncation outside the workspace is limited only by Landlock's coverage")
        not_enforced = [
            "memory is limited per process (RLIMIT_AS); the total of a job is watched by the monitor every 100 ms "
            "(no cgroup is available in this container)",
            "the per-job task limit is watched by the monitor; the kernel only enforces the ceiling of the uid",
            "jobs under the same sandbox uid can signal each other",
            "disk use is bounded per file (RLIMIT_FSIZE) and checked against workspace_max_mb before each "
            "operation, not by a quota",
        ]
        if p["landlock_abi"] < 3:
            not_enforced.append("Landlock ABI < 3 cannot deny truncate(); with a uid drop, ownership does")
        scoped = {
            "kernel_enforced": ["the job cannot create a socket of any family; its only route out is the inherited "
                                "broker descriptor", "the job cannot connect, bind or listen"],
            "harness_enforced": ["every outbound TCP connection is one net.connect primitive authorized by "
                                 "ExecContext.check before the broker resolves and connects",
                                 "private, loopback and link-local destinations are refused unless "
                                 "allow_private_destinations is set", "connection count and byte ceilings"],
            "not_enforced": ["authorization is per connection (host, port): requests reusing a connection and "
                             "anything inside TLS are not inspected",
                             "only Python processes get the transparent shim; other programs have no network",
                             "kernel-level per-destination filtering (network namespace, nftables) is not available "
                             "in this unprivileged container"],
        }
        return {"modality": c.Modality.CODE, "available": self.available,
                "unavailable_reason": self.unavailable_reason,
                "isolation": {"uid_mode": mode, "uid": p["uid"], "user": p["user"],
                              "landlock_abi": p["landlock_abi"] if cfg.landlock else 0,
                              "landlock_net": bool(p["landlock_net"] and cfg.landlock),
                              "seccomp": bool(cfg.seccomp and p["seccomp_lib"]), "no_new_privs": True},
                "weak": bool(weak), "weaknesses": weak, "kernel_enforced": kernel, "not_enforced": not_enforced,
                "network": {"default": "none", "modes": ["none", "scoped"], "scoped": scoped},
                "limits": asdict(cfg.limits), "max_wall_s": cfg.max_wall_s, "workspace": self.workspace,
                "packages": [os.path.basename(x) for x in getattr(self, "_pkg_reads", []) if not x.endswith(".libs")],
                "boundary": "process isolation on a shared kernel; a virtual machine is recommended for live use"}

    # ------------------------------------------------------------------------------------------ files
    def _require(self) -> None:
        if not self.available or self._root_fd is None:
            raise SandboxUnavailable(self.unavailable_reason or "sandbox closed")

    def _own(self, fd: int) -> None:
        if self.drops_uid:
            os.fchown(fd, self.uid, self.gid)

    def _mkdirs(self, rel: str) -> int:
        """Create ``rel`` below the workspace without following symlinks; returns an fd of the last directory."""
        cur = os.dup(self._root_fd)
        try:
            for part in [x for x in rel.split("/") if x not in ("", ".")]:
                try:
                    os.mkdir(part, 0o700, dir_fd=cur)
                    created = True
                except FileExistsError:
                    created = False
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=cur)
                if created:
                    self._own(nxt)
                os.close(cur)
                cur = nxt
            return cur
        except BaseException:
            os.close(cur)
            raise

    def write_file(self, path: str, data: bytes, append: bool = False) -> Dict[str, Any]:
        self._require()
        rel = _safe_rel(path)
        if len(data) > self.cfg.max_file_bytes:
            raise ValueError(f"content larger than {self.cfg.max_file_bytes} bytes")
        self._check_quota(len(data))
        parent, name = os.path.split(rel)
        dfd = self._mkdirs(parent)
        try:
            if append:
                fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
                             | os.O_CLOEXEC, 0o600, dir_fd=dfd)
                try:
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        raise ValueError("not a plain file")
                    self._own(fd)
                    os.write(fd, data)
                finally:
                    os.close(fd)
            else:                                      # new inode, then atomic replace: never write through a link
                tmp = f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                             dir_fd=dfd)
                try:
                    self._own(fd)
                    view = memoryview(data)
                    while view:
                        n = os.write(fd, view)
                        view = view[n:]
                finally:
                    os.close(fd)
                try:
                    st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                    if stat.S_ISDIR(st.st_mode):
                        os.unlink(tmp, dir_fd=dfd)
                        raise IsADirectoryError(errno.EISDIR, "is a directory", rel)
                except FileNotFoundError:
                    pass
                os.rename(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        finally:
            os.close(dfd)
        return {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "append": append}

    def read_file(self, path: str, offset: int = 0, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        self._require()
        rel = _safe_rel(path)
        cap = min(int(max_bytes or self.cfg.max_file_bytes), self.cfg.max_file_bytes)
        fd = _open_beneath(self._root_fd, rel, os.O_RDONLY | os.O_NONBLOCK)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("not a plain file")
            os.lseek(fd, max(int(offset), 0), os.SEEK_SET)
            chunks, left = [], cap
            while left > 0:
                b = os.read(fd, min(left, 1 << 20))
                if not b:
                    break
                chunks.append(b)
                left -= len(b)
            data = b"".join(chunks)
        finally:
            os.close(fd)
        return {"path": rel, "size": st.st_size, "offset": max(int(offset), 0), "data": data,
                "truncated": max(int(offset), 0) + len(data) < st.st_size, "mtime": st.st_mtime}

    def list_files(self, path: str = ".", recursive: bool = False, max_entries: int = 1000) -> Dict[str, Any]:
        self._require()
        rel = _safe_rel(path)
        entries: List[Dict[str, Any]] = []
        truncated = False

        def walk(dfd: int, prefix: str) -> None:
            nonlocal truncated
            with os.scandir(dfd) as it:
                names = sorted((e.name, e) for e in it)
            for name, e in names:
                if len(entries) >= max_entries:
                    truncated = True
                    return
                st = e.stat(follow_symlinks=False)
                kind = "dir" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else \
                    "symlink" if stat.S_ISLNK(st.st_mode) else "other"
                full = name if prefix in ("", ".") else f"{prefix}/{name}"
                entries.append({"path": full, "type": kind, "size": st.st_size if kind == "file" else 0,
                                "mtime": st.st_mtime})
                if recursive and kind == "dir":
                    try:
                        sub = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dfd)
                    except OSError:
                        continue
                    try:
                        walk(sub, full)
                    finally:
                        os.close(sub)

        fd = _open_beneath(self._root_fd, rel, os.O_RDONLY | os.O_DIRECTORY)
        try:
            walk(fd, "" if rel == "." else rel)
        finally:
            os.close(fd)
        return {"path": rel, "entries": entries, "truncated": truncated}

    def usage(self, max_files: int = 200_000) -> Dict[str, int]:
        """Bytes and entries in the workspace (walk without following symlinks)."""
        self._require()
        total = files = 0
        stack = [os.dup(self._root_fd)]
        while stack:
            dfd = stack.pop()
            try:
                with os.scandir(dfd) as it:
                    for e in it:
                        try:
                            st = e.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        files += 1
                        total += st.st_blocks * 512
                        if stat.S_ISDIR(st.st_mode) and files < max_files:
                            try:
                                stack.append(os.open(e.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                                                     | os.O_CLOEXEC, dir_fd=dfd))
                            except OSError:
                                pass
            finally:
                os.close(dfd)
        return {"bytes": total, "entries": files}

    def _check_quota(self, incoming: int = 0) -> None:
        used = self.usage()["bytes"]
        if used + incoming > self.cfg.workspace_max_mb * 2**20:
            raise OSError(errno.EDQUOT, f"workspace holds {used} bytes; limit is {self.cfg.workspace_max_mb} MB")

    # ------------------------------------------------------------------------------------------ execution
    def _env(self, cwd: str, network: str, egress_fd: Optional[int], extra: Optional[Dict[str, str]]) -> Dict[str, str]:
        ws = self.workspace
        pypath = [cwd, os.path.join(self.support, "pkgs")]
        if network == "scoped":
            pypath.insert(0, os.path.join(self.support, "shim"))
        env = {"PATH": f"{os.path.join(self.support, 'bin')}:/usr/local/bin:/usr/bin:/bin",
               "HOME": ws, "TMPDIR": os.path.join(ws, ".tmp"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "PYTHONPATH": ":".join(pypath), "PYTHONPYCACHEPREFIX": os.path.join(ws, ".cam", "pycache"),
               "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
               "CAM_SANDBOX": "1", "CAM_SANDBOX_NETWORK": network, "USER": "sandbox", "LOGNAME": "sandbox",
               "SHELL": "/bin/sh", "TERM": "dumb"}
        for k, v in (extra or {}).items():
            if _ENV_KEY.match(str(k)) and k not in _RESERVED_ENV and not k.startswith("CAM_") and len(str(v)) < 4096:
                env[str(k)] = str(v)
        if egress_fd is not None:
            env["CAM_EGRESS_FD"] = str(egress_fd)
        return env

    def _uid_tasks(self) -> int:
        n = 0
        for name in os.listdir("/proc"):
            if name.isdigit():
                try:
                    if os.stat(f"/proc/{name}").st_uid == self.uid:
                        with open(f"/proc/{name}/stat", "rb") as f:
                            n += int(f.read().rsplit(b")", 1)[1].split()[17])
                except (OSError, IndexError, ValueError):
                    continue
        return n

    def spawn(self, argv: Sequence[str], *, limits: Optional[SandboxLimits] = None, network: str = "none",
              broker: Optional[EgressBroker] = None, stdin: Optional[bytes] = None,
              env: Optional[Dict[str, str]] = None, cwd: str = ".", stdout_path: str = "",
              stderr_path: str = "", log_cap: int = 8 * 2**20) -> SandboxProcess:
        """Start ``argv`` confined to the workspace and return a handle. ``network='scoped'`` needs ``broker``."""
        self._require()
        if network not in ("none", "scoped"):
            raise ValueError("network must be 'none' or 'scoped'")
        if network == "scoped" and broker is None:
            raise ValueError("scoped network needs an EgressBroker")
        lim = limits or self.cfg.limits
        rel = _safe_rel(cwd)
        cwd_fd = self._mkdirs(rel)
        os.close(cwd_fd)
        cwd_abs = self.workspace if rel == "." else os.path.join(self.workspace, rel)
        if os.path.realpath(cwd_abs) != cwd_abs and not os.path.realpath(cwd_abs).startswith(self.workspace + "/"):
            raise ValueError("working directory escapes the workspace")
        argv = [str(a) for a in argv]
        if not os.path.isabs(argv[0]):
            raise ValueError("argv[0] must be an absolute path")
        nproc_kernel = self.cfg.uid_nproc
        if self._probe["uid_mode"] != "dedicated-uid":
            nproc_kernel += self._uid_tasks()          # the uid is shared with processes that are not ours
        egress_fd = broker.child_fd if (network == "scoped" and broker is not None) else None
        cfg = {"argv": argv, "env": self._env(cwd_abs, network, egress_fd, env), "cwd": cwd_abs,
               "uid": self.uid if self.drops_uid else None, "gid": self.gid if self.drops_uid else None,
               "harness_pid": os.getpid(),
               "rlimits": {"as": int(lim.mem_mb) * 2**20, "cpu": int(lim.cpu_s), "nproc": int(nproc_kernel),
                           "nofile": int(lim.nofile), "fsize": int(lim.fsize_mb) * 2**20},
               "monitor": {"wall_s": float(lim.wall_s), "mem_total": int(lim.mem_total_mb) * 2**20,
                           "nproc": int(lim.nproc)},
               "read_paths": self._read_paths, "list_paths": self._list_paths,
               "write_paths": [self.workspace],
               "landlock": bool(self.cfg.landlock), "seccomp": bool(self.cfg.seccomp),
               "seccomp_lib": self._probe["seccomp_lib"],
               "keep_fds": [egress_fd] if egress_fd is not None else []}
        cfg_r, cfg_w = os.pipe()
        status_r, status_w = os.pipe()
        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        pass_fds = [cfg_r, status_w] + ([egress_fd] if egress_fd is not None else [])
        try:
            proc = subprocess.Popen(
                [os.path.realpath(sys.executable), "-I", "-S", "-B", "-c", _BOOT, str(cfg_r), str(status_w)],
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL, stdout=out_w, stderr=err_w,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, cwd="/", pass_fds=pass_fds, close_fds=True,
                start_new_session=True)
        except BaseException:
            for fd in (cfg_r, cfg_w, status_r, status_w, out_r, out_w, err_r, err_w):
                os.close(fd)
            raise
        for fd in (cfg_r, status_w, out_w, err_w):
            os.close(fd)
        with os.fdopen(cfg_w, "w") as f:
            json.dump(cfg, f)
        if broker is not None:
            broker.after_spawn()
        if stdin is not None:
            def feed(p=proc, data=stdin):
                try:
                    p.stdin.write(data)
                    p.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            threading.Thread(target=feed, name="cam-sbx-stdin", daemon=True).start()
        handle = SandboxProcess(proc, status_r, _Capture(out_r, lim.out_bytes, stdout_path, log_cap),
                                _Capture(err_r, lim.out_bytes, stderr_path, log_cap), float(lim.wall_s),
                                broker if network == "scoped" else None)
        with self._lock:
            self._procs = [p for p in self._procs if p._result is None] + [handle]
        return handle

    def run(self, argv: Sequence[str], *, cancel: Any = None, deadline_monotonic: Optional[float] = None,
            **kw: Any) -> RunResult:
        return self.spawn(argv, **kw).wait(cancel=cancel, deadline_monotonic=deadline_monotonic)

    def _exec_file(self, code: str, suffix: str, label: str) -> str:
        with self._lock:
            self._seq += 1
            n = self._seq
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)[:48] or "run"
        rel = f".cam/exec/{int(time.time() * 1000):013d}_{n:04d}_{safe}{suffix}"
        self.write_file(rel, code.encode())
        try:
            dfd = _open_beneath(self._root_fd, ".cam/exec", os.O_RDONLY | os.O_DIRECTORY)
            try:
                for old in sorted(os.listdir(dfd))[:-self.cfg.keep_exec_files]:
                    os.unlink(old, dir_fd=dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
        return rel

    def python_argv(self, code: str, args: Sequence[str] = (), label: str = "py") -> Tuple[List[str], str]:
        rel = self._exec_file(code, ".py", label)
        return [os.path.join(self.support, "bin", "python"), os.path.join(self.workspace, rel), *map(str, args)], rel

    def shell_argv(self, command: str, label: str = "sh") -> Tuple[List[str], str]:
        rel = self._exec_file(command, ".sh", label)
        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        return [shell, os.path.join(self.workspace, rel)], rel

    def run_python(self, code: str, args: Sequence[str] = (), **kw: Any) -> RunResult:
        argv, _ = self.python_argv(code, args)
        return self.run(argv, **kw)

    def run_shell(self, command: str, **kw: Any) -> RunResult:
        argv, _ = self.shell_argv(command)
        return self.run(argv, **kw)

    def close(self) -> None:
        with self._lock:
            procs, self._procs = self._procs, []
        for p in procs:
            if p._result is None:
                p.cancel()
                p.wait()
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None


# ==============================================================================================================
# The modality executor
# ==============================================================================================================
def _authorize(ctx: c.ExecContext, op: c.PrimitiveOp, action: c.ActionSpec) -> Tuple[str, str]:
    """Ask the independent authorization layer about one primitive. Fail-closed: no callback, no execution."""
    chk = getattr(ctx, "check", None)
    if not callable(chk):
        return c.AuthVerdict.DENY, "no authorization callback in the execution context"
    try:
        d = chk(op, action)
    except Exception as e:  # noqa: BLE001 - an unreachable or failing policy service means no permission
        return c.AuthVerdict.DENY, f"authorization check failed: {type(e).__name__}: {e}"
    verdict = getattr(d, "verdict", None)
    reasons = "; ".join(getattr(d, "reasons", None) or [])
    if verdict == c.AuthVerdict.ALLOW:
        return c.AuthVerdict.ALLOW, ""
    if verdict == c.AuthVerdict.NEEDS_APPROVAL:
        return c.AuthVerdict.NEEDS_APPROVAL, reasons or "operator approval required"
    return c.AuthVerdict.DENY, reasons or "denied by policy"


def _scrub(ctx: c.ExecContext, text: str) -> str:
    fn = getattr(getattr(ctx, "secrets", None), "scrub", None)
    if callable(fn):
        try:
            return str(fn(text))
        except Exception:  # noqa: BLE001 - scrubbing is best effort here; the sandbox never receives a secret
            return text
    return text


_LIMIT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "cpu_s": {"type": "integer", "minimum": 1}, "mem_mb": {"type": "integer", "minimum": 16},
    "mem_total_mb": {"type": "integer", "minimum": 16}, "nproc": {"type": "integer", "minimum": 1},
    "nofile": {"type": "integer", "minimum": 16}, "fsize_mb": {"type": "integer", "minimum": 1},
    "out_bytes": {"type": "integer", "minimum": 256}}}
_RUN_PROPS = {"timeout_s": {"type": "number", "exclusiveMinimum": 0},
              "network": {"type": "string", "enum": ["none", "scoped"]},
              "cwd": {"type": "string"}, "stdin": {"type": "string"},
              "env": {"type": "object", "additionalProperties": {"type": "string"}}, "limits": _LIMIT_SCHEMA}


class SandboxExecutor:
    """``contracts.ModalityExecutor`` for modality ``code``."""

    modality = c.Modality.CODE

    def __init__(self, workspace: Optional[str] = None, cfg: Optional[SandboxConfig] = None,
                 sandbox: Optional[Sandbox] = None, jobs: Any = None):
        self.sandbox = sandbox or Sandbox(workspace, cfg)
        self.cfg = self.sandbox.cfg
        self._jobs = jobs
        self._jobs_lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ contract
    def capabilities(self) -> Dict[str, Any]:
        caps = self.sandbox.capabilities()
        caps["tools"] = [t.name for t in self.tools()]
        return caps

    def tools(self) -> List[c.ToolDescriptor]:
        ok, why = self.sandbox.available, self.sandbox.unavailable_reason
        lim = self.cfg.limits

        def tool(name: str, desc: str, props: Dict[str, Any], required: List[str], side: str = "local",
                 seconds: float = 0.1) -> c.ToolDescriptor:
            return c.ToolDescriptor(
                name=name, modality=self.modality, description=desc, level=c.Level.PRIMITIVE,
                arg_schema={"type": "object", "properties": props, "required": required,
                            "additionalProperties": False},
                auth_scope="sandbox", side_effects=side, est_cost=c.Cost(seconds=seconds, primitive_ops=1),
                available=ok, unavailable_reason="" if ok else why)

        run_note = (f"Runs confined to the sandbox workspace with no network unless network='scoped' is "
                    f"authorized; default limits {lim.wall_s:g}s wall, {lim.cpu_s}s CPU, {lim.mem_mb} MB. Returns "
                    f"exit code, stdout and stderr (truncated at {lim.out_bytes} bytes each).")
        out = [
            tool("sandbox.python", "Run a Python program given as source text. " + run_note,
                 {"code": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}, **_RUN_PROPS},
                 ["code"], seconds=1.0),
            tool("sandbox.shell", "Run a shell script given as text. " + run_note,
                 {"command": {"type": "string"}, **_RUN_PROPS}, ["command"], seconds=1.0),
            tool("sandbox.write_file", "Create or replace a file in the sandbox workspace (text, or base64 for "
                 "binary content); append=true adds to the end.",
                 {"path": {"type": "string"}, "content": {"type": "string"},
                  "encoding": {"type": "string", "enum": ["utf-8", "base64"]}, "append": {"type": "boolean"}},
                 ["path", "content"]),
            tool("sandbox.read_file", "Read a file from the sandbox workspace; returns text, or base64 when the "
                 "content is not UTF-8.",
                 {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
                  "max_bytes": {"type": "integer", "minimum": 1}}, ["path"], side="none"),
            tool("sandbox.list_files", "List files and directories in the sandbox workspace.",
                 {"path": {"type": "string"}, "recursive": {"type": "boolean"},
                  "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000}}, [], side="none"),
            tool("sandbox.job_start", "Start a background job (Python source or shell script) in the sandbox. "
                 "Needs a purpose text and a deadline; returns a job id at once. The result arrives later as an "
                 "event and is kept as a record.",
                 {"purpose": {"type": "string", "minLength": 1}, "kind": {"type": "string", "enum": ["python", "shell"]},
                  "code": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
                  "deadline_s": {"type": "number", "exclusiveMinimum": 0},
                  "network": {"type": "string", "enum": ["none", "scoped"]}, "cwd": {"type": "string"},
                  "env": {"type": "object", "additionalProperties": {"type": "string"}}, "limits": _LIMIT_SCHEMA},
                 ["purpose", "kind", "code", "deadline_s"], seconds=0.1),
            tool("sandbox.job_status", "Report the state of one background job, or of all jobs when no id is given.",
                 {"job_id": {"type": "string"}}, [], side="none"),
            tool("sandbox.job_result", "Return the persisted result record of a background job with the tail of "
                 "its output.", {"job_id": {"type": "string"}, "tail_bytes": {"type": "integer", "minimum": 1}},
                 ["job_id"], side="none"),
            tool("sandbox.job_cancel", "Cancel a queued or running background job.",
                 {"job_id": {"type": "string"}}, ["job_id"]),
        ]
        return out

    _OPS = ("sandbox.python", "sandbox.shell", "sandbox.write_file", "sandbox.read_file", "sandbox.list_files",
            "sandbox.job_start", "sandbox.job_status", "sandbox.job_result", "sandbox.job_cancel")

    def validate(self, action: c.ActionSpec) -> List[str]:
        errs: List[str] = []
        a = action.arguments if isinstance(action.arguments, dict) else None
        if action.operation not in self._OPS:
            return [f"unknown operation {action.operation!r} for modality {self.modality}"]
        if a is None:
            return ["arguments must be an object"]
        schema = next(t.arg_schema for t in self.tools() if t.name == action.operation)
        props = schema["properties"]
        for k in schema["required"]:
            if k not in a:
                errs.append(f"missing argument {k!r}")
        for k, v in a.items():
            if k not in props:
                errs.append(f"unknown argument {k!r}")
                continue
            errs += _check_type(k, v, props[k])
        if errs:
            return errs
        for k in ("path", "cwd"):
            if k in a:
                try:
                    _safe_rel(a[k])
                except ValueError as e:
                    errs.append(f"{k}: {e}")
        for k in ("code", "command", "content"):
            if k in a and len(a[k].encode("utf-8", "replace")) > self.cfg.max_file_bytes:
                errs.append(f"{k} is larger than {self.cfg.max_file_bytes} bytes")
        if a.get("encoding") == "base64":
            try:
                base64.b64decode(a.get("content", ""), validate=True)
            except ValueError:
                errs.append("content is not valid base64")
        if action.operation == "sandbox.job_start" and not str(a.get("purpose", "")).strip():
            errs.append("purpose must not be empty")
        return errs

    def close(self) -> None:
        with self._jobs_lock:
            jobs, self._jobs = self._jobs, None
        if jobs is not None and hasattr(jobs, "close"):
            jobs.close()
        self.sandbox.close()

    # ------------------------------------------------------------------------------------------ helpers
    def _limits(self, a: Dict[str, Any], timeout_s: float, ceiling_s: float) -> SandboxLimits:
        """Requested limits can only tighten the configured ones."""
        base = self.cfg.limits
        req = a.get("limits") or {}
        vals = {k: min(int(req[k]), getattr(base, k)) if k in req else getattr(base, k)
                for k in ("cpu_s", "mem_mb", "mem_total_mb", "nproc", "nofile", "fsize_mb", "out_bytes")}
        return replace(base, wall_s=max(0.05, min(float(timeout_s), float(ceiling_s))), **vals)

    def _result(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float, status: str,
                failure: str = c.FailureClass.NONE, detail: str = "", obs: Optional[c.Observation] = None,
                ops: Optional[List[c.PrimitiveOp]] = None, compute_s: float = 0.0,
                verified: Optional[bool] = None, artifacts: Optional[Dict[str, str]] = None) -> c.ActionResult:
        ops = ops or []
        observations = [obs] if obs is not None else []
        for o in observations:
            sink = getattr(ctx, "obs_sink", None)
            if callable(sink):
                sink(o)
        return c.ActionResult(action_id=action.action_id, status=status, failure_class=failure,
                              failure_detail=detail, started=t0, ended=ctx.clock.now(), observations=observations,
                              ops=ops, cost=c.Cost(seconds=time.monotonic() - m0, compute_s=compute_s,
                                                   primitive_ops=len(ops)),
                              verified=verified, artifacts=artifacts or {})

    def _obs(self, action: c.ActionSpec, ctx: c.ExecContext, kind: str, text: str, data: Dict[str, Any],
             quantities: Optional[List[c.Quantity]] = None, uri: str = "") -> c.Observation:
        text = _scrub(ctx, text)
        return c.Observation(
            obs_id=c.new_id("obs"), ts=ctx.clock.now(), kind=kind, text=text[:8000], quantities=quantities or [],
            provenance=c.Provenance(source="sandbox", uri=uri or f"sandbox:{action.operation}",
                                    retrieved_at=ctx.clock.now(), trust=c.Trust.UNTRUSTED,
                                    content_sha256=hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()),
            parent_action_id=action.action_id, data=data)

    def _denied(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float, verdict: str, reason: str,
                op: c.PrimitiveOp) -> c.ActionResult:
        op.status = c.ResultStatus.NEEDS_APPROVAL if verdict == c.AuthVerdict.NEEDS_APPROVAL else c.ResultStatus.DENIED
        op.failure_class = c.FailureClass.MISSING_PERMISSION
        op.note = reason
        op.ended = ctx.clock.now()
        return self._result(action, ctx, t0, m0, op.status, c.FailureClass.MISSING_PERMISSION, reason, ops=[op])

    def jobs(self, ctx: c.ExecContext):
        """The background-job manager bound to this executor's sandbox (created on first use)."""
        with self._jobs_lock:
            if self._jobs is None:
                from .jobs import JobManager
                run_dir = getattr(ctx, "run_dir", "") or os.path.join(self.cfg.base_dir, "runs", "default")
                self._jobs = JobManager(run_dir, sandbox=self.sandbox, clock=ctx.clock,
                                        event_sink=getattr(ctx, "obs_sink", None))
            return self._jobs

    # ------------------------------------------------------------------------------------------ execute
    def execute(self, action: c.ActionSpec, ctx: c.ExecContext) -> c.ActionResult:
        t0, m0 = ctx.clock.now(), time.monotonic()
        errs = self.validate(action)
        if errs:
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.INVALID_ARGUMENTS,
                                "; ".join(errs))
        if not self.sandbox.available:
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.RESOURCE_LIMIT,
                                f"sandbox unavailable: {self.sandbox.unavailable_reason}")
        cancel = getattr(ctx, "cancel", None)
        if cancel is not None and cancel.is_set():
            return self._result(action, ctx, t0, m0, c.ResultStatus.CANCELLED, detail="cancelled before start")
        a = action.arguments
        name = action.operation
        try:
            if name in ("sandbox.python", "sandbox.shell"):
                return self._run(action, ctx, t0, m0)
            if name == "sandbox.write_file":
                return self._write(action, ctx, t0, m0)
            if name == "sandbox.read_file":
                return self._read(action, ctx, t0, m0)
            if name == "sandbox.list_files":
                return self._list(action, ctx, t0, m0)
            from .jobs import execute_job_tool
            return execute_job_tool(self, action, ctx, t0, m0)
        except SandboxUnavailable as e:
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.RESOURCE_LIMIT, str(e))
        except (ValueError, IsADirectoryError, NotADirectoryError, FileNotFoundError) as e:
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.INVALID_ARGUMENTS,
                                f"{type(e).__name__}: {e}")
        except OSError as e:
            cls = c.FailureClass.RESOURCE_LIMIT if e.errno in (errno.EDQUOT, errno.ENOSPC, errno.EFBIG) \
                else c.FailureClass.INVALID_ARGUMENTS if e.errno in (errno.EXDEV, errno.ELOOP, errno.EACCES,
                                                                      errno.ENOENT, errno.ENOTDIR) \
                else c.FailureClass.UNKNOWN
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, cls, f"{type(e).__name__}: {e}")

    def _net_authorizer(self, action: c.ActionSpec, ctx: c.ExecContext, ops: List[c.PrimitiveOp],
                        lock: threading.Lock) -> Callable[[str, int, str, bool], Tuple[bool, str]]:
        def authorize(host: str, port: int, ip: str, private: bool) -> Tuple[bool, str]:
            op = c.PrimitiveOp(op="net.connect", modality=self.modality,
                               arguments={"host": host, "port": port, "ip": ip, "private": private,
                                          "url": f"tcp://{host}:{port}", "from": action.operation},
                               started=ctx.clock.now())
            verdict, reason = _authorize(ctx, op, action)
            op.ended = ctx.clock.now()
            if verdict != c.AuthVerdict.ALLOW:
                op.status = c.ResultStatus.DENIED
                op.failure_class = c.FailureClass.MISSING_PERMISSION
                op.note = reason
            with lock:
                ops.append(op)
            return verdict == c.AuthVerdict.ALLOW, reason
        return authorize

    def make_broker(self, action: c.ActionSpec, ctx: c.ExecContext, ops: List[c.PrimitiveOp],
                    lock: threading.Lock) -> EgressBroker:
        return EgressBroker(self._net_authorizer(action, ctx, ops, lock),
                            allow_private=self.cfg.allow_private_destinations,
                            max_connections=self.cfg.egress_max_connections, max_bytes=self.cfg.egress_max_bytes,
                            connect_timeout_s=self.cfg.egress_connect_timeout_s)

    def _run(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float) -> c.ActionResult:
        a, sb = action.arguments, self.sandbox
        is_py = action.operation == "sandbox.python"
        source = a["code"] if is_py else a["command"]
        network = a.get("network", "none")
        timeout = float(a.get("timeout_s", min(action.timeout_s or self.cfg.limits.wall_s, self.cfg.limits.wall_s)))
        ceiling = self.cfg.max_wall_s
        deadline = getattr(ctx, "deadline", 0.0) or 0.0
        deadline_mono = None
        if deadline:
            left = deadline - ctx.clock.now()
            if left <= 0:
                return self._result(action, ctx, t0, m0, c.ResultStatus.TIMEOUT, c.FailureClass.RESOURCE_LIMIT,
                                    "the execution deadline had already passed")
            ceiling = min(ceiling, left)
            deadline_mono = time.monotonic() + left
        if action.timeout_s:
            ceiling = min(ceiling, float(action.timeout_s))
        limits = self._limits(a, timeout, ceiling)
        op = c.PrimitiveOp(op=action.operation, modality=self.modality, started=t0,
                           arguments={"sha256": hashlib.sha256(source.encode("utf-8", "replace")).hexdigest(),
                                      "bytes": len(source.encode("utf-8", "replace")), "network": network,
                                      "cwd": a.get("cwd", "."), "wall_s": limits.wall_s})
        verdict, reason = _authorize(ctx, op, action)
        if verdict != c.AuthVerdict.ALLOW:
            return self._denied(action, ctx, t0, m0, verdict, reason, op)
        sb._check_quota()
        ops: List[c.PrimitiveOp] = [op]
        lock = threading.Lock()
        broker = self.make_broker(action, ctx, ops, lock) if network == "scoped" else None
        try:
            if is_py:
                argv, rel = sb.python_argv(source, a.get("args", ()), label=action.action_id)
            else:
                argv, rel = sb.shell_argv(source, label=action.action_id)
            proc = sb.spawn(argv, limits=limits, network=network, broker=broker, env=a.get("env"),
                            stdin=a["stdin"].encode() if "stdin" in a else None, cwd=a.get("cwd", "."))
        except BaseException:
            if broker is not None:
                broker.close()
            raise
        res = proc.wait(cancel=getattr(ctx, "cancel", None), deadline_monotonic=deadline_mono)
        op.ended = ctx.clock.now()
        return self._finish_run(action, ctx, t0, m0, res, ops, rel, broker)

    def _finish_run(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float, res: RunResult,
                    ops: List[c.PrimitiveOp], rel: str, broker: Optional[EgressBroker]) -> c.ActionResult:
        op = ops[0]
        status, failure, detail = c.ResultStatus.OK, c.FailureClass.NONE, ""
        limit = res.limit_hit
        if res.setup_error:
            status, failure, detail = c.ResultStatus.FAILED, c.FailureClass.UNKNOWN, f"sandbox setup: {res.setup_error}"
        elif res.cancelled:
            status, detail = c.ResultStatus.CANCELLED, "cancelled"
        elif res.timed_out:
            status, failure, detail = c.ResultStatus.TIMEOUT, c.FailureClass.RESOURCE_LIMIT, "wall-clock limit reached"
        elif limit:
            status, failure, detail = c.ResultStatus.FAILED, c.FailureClass.RESOURCE_LIMIT, f"{limit} limit reached"
        elif res.exit_code != 0:
            status, failure = c.ResultStatus.FAILED, c.FailureClass.UNKNOWN
            detail = f"exit code {res.exit_code}" if res.exit_code is not None else f"killed by signal {res.signal}"
        op.status, op.failure_class, op.note = status, failure, detail
        stdout, stderr = _scrub(ctx, res.stdout), _scrub(ctx, res.stderr)
        head = f"exit_code={res.exit_code} signal={res.signal} wall_s={res.wall_s:.3f}"
        if detail:
            head += f" ({detail})"
        text = head + (f"\n--- stdout ---\n{stdout}" if stdout else "") + (f"\n--- stderr ---\n{stderr}" if stderr else "")
        q = [c.Quantity("sandbox.wall_s", float(res.wall_s), "s"), c.Quantity("sandbox.cpu_s", float(res.cpu_s), "s"),
             c.Quantity("sandbox.max_rss_mb", res.max_rss_kb / 1024.0, "MB"),
             c.Quantity("sandbox.stdout_bytes", float(res.stdout_bytes), "bytes", c.OutcomeKind.COUNT)]
        if res.exit_code is not None:
            q.insert(0, c.Quantity("sandbox.exit_code", float(res.exit_code), "", c.OutcomeKind.COUNT))
        data = {"exit_code": res.exit_code, "signal": res.signal, "stdout": stdout, "stderr": stderr,
                "stdout_truncated": res.stdout_truncated, "stderr_truncated": res.stderr_truncated,
                "timed_out": res.timed_out, "cancelled": res.cancelled, "killed_for": res.killed_for,
                "limit_hit": limit, "leftover_processes": res.leftover, "peak_tasks": res.peak_tasks,
                "peak_tree_rss_kb": res.peak_tree_rss_kb, "isolation": res.isolation, "source_file": rel,
                "network": action.arguments.get("network", "none"),
                "connections": [dict(e) for e in broker.log] if broker is not None else []}
        obs = self._obs(action, ctx, "tool_result", text, data, q, uri=f"sandbox:{rel}")
        verified = self._verify(action, ctx, res) if status == c.ResultStatus.OK else (
            False if action.verify.kind in ("text_present", "program") else None)
        return self._result(action, ctx, t0, m0, status, failure, detail, obs, ops, compute_s=float(res.cpu_s),
                            verified=verified, artifacts={"source": rel})

    def _verify(self, action: c.ActionSpec, ctx: c.ExecContext, res: RunResult) -> Optional[bool]:
        """Evaluate ``ActionSpec.verify`` against what the execution actually produced (None = not checkable)."""
        v = action.verify
        if v.kind == "text_present":
            needle = str(v.spec.get("text", ""))
            stream = res.stderr if v.spec.get("stream") == "stderr" else res.stdout
            return bool(needle) and needle in stream
        if v.kind == "program":
            cmd = v.spec.get("command")
            if not isinstance(cmd, str) or not cmd:
                return None
            op = c.PrimitiveOp(op="sandbox.shell", modality=self.modality, started=ctx.clock.now(),
                               arguments={"sha256": hashlib.sha256(cmd.encode()).hexdigest(), "verify": True})
            if _authorize(ctx, op, action)[0] != c.AuthVerdict.ALLOW:
                return None
            argv, _ = self.sandbox.shell_argv(cmd, label="verify")
            chk = self.sandbox.run(argv, limits=replace(self.cfg.limits, wall_s=min(30.0, self.cfg.limits.wall_s)))
            return chk.exit_code == int(v.spec.get("expect_exit", 0))
        return None

    def _write(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float) -> c.ActionResult:
        a = action.arguments
        data = base64.b64decode(a["content"]) if a.get("encoding") == "base64" else a["content"].encode()
        op = c.PrimitiveOp(op=action.operation, modality=self.modality, started=t0,
                           arguments={"path": a["path"], "bytes": len(data), "append": bool(a.get("append", False)),
                                      "sha256": hashlib.sha256(data).hexdigest()})
        verdict, reason = _authorize(ctx, op, action)
        if verdict != c.AuthVerdict.ALLOW:
            return self._denied(action, ctx, t0, m0, verdict, reason, op)
        info = self.sandbox.write_file(a["path"], data, append=bool(a.get("append", False)))
        back = self.sandbox.read_file(info["path"], max_bytes=1)       # inspect the result, do not assume it
        op.ended = ctx.clock.now()
        obs = self._obs(action, ctx, "tool_result", f"wrote {info['bytes']} bytes to {info['path']}",
                        {**info, "size": back["size"]},
                        [c.Quantity("sandbox.file_bytes", float(back["size"]), "bytes", c.OutcomeKind.COUNT)],
                        uri=f"sandbox:{info['path']}")
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=obs, ops=[op],
                            artifacts={"file": info["path"]})

    def _read(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float) -> c.ActionResult:
        a = action.arguments
        op = c.PrimitiveOp(op=action.operation, modality=self.modality, started=t0, arguments={"path": a["path"]})
        verdict, reason = _authorize(ctx, op, action)
        if verdict != c.AuthVerdict.ALLOW:
            return self._denied(action, ctx, t0, m0, verdict, reason, op)
        info = self.sandbox.read_file(a["path"], offset=int(a.get("offset", 0)),
                                      max_bytes=min(int(a.get("max_bytes", 65536)), self.cfg.max_file_bytes))
        op.ended = ctx.clock.now()
        raw = info.pop("data")
        try:
            content, encoding = raw.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, encoding = base64.b64encode(raw).decode(), "base64"
        content = _scrub(ctx, content) if encoding == "utf-8" else content
        obs = self._obs(action, ctx, "document", content if encoding == "utf-8" else f"[{len(raw)} binary bytes]",
                        {**info, "content": content, "encoding": encoding,
                         "sha256": hashlib.sha256(raw).hexdigest()},
                        [c.Quantity("sandbox.file_bytes", float(info["size"]), "bytes", c.OutcomeKind.COUNT)],
                        uri=f"sandbox:{info['path']}")
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=obs, ops=[op])

    def _list(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float) -> c.ActionResult:
        a = action.arguments
        op = c.PrimitiveOp(op=action.operation, modality=self.modality, started=t0,
                           arguments={"path": a.get("path", "."), "recursive": bool(a.get("recursive", False))})
        verdict, reason = _authorize(ctx, op, action)
        if verdict != c.AuthVerdict.ALLOW:
            return self._denied(action, ctx, t0, m0, verdict, reason, op)
        info = self.sandbox.list_files(a.get("path", "."), recursive=bool(a.get("recursive", False)),
                                       max_entries=int(a.get("max_entries", 1000)))
        op.ended = ctx.clock.now()
        lines = [f"{e['type'][0]} {e['size']:>10} {e['path']}" for e in info["entries"]]
        obs = self._obs(action, ctx, "tool_result", "\n".join(lines) or "(empty)", info,
                        [c.Quantity("sandbox.entries", float(len(info["entries"])), "", c.OutcomeKind.COUNT)],
                        uri=f"sandbox:{info['path']}")
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=obs, ops=[op])


def _check_type(name: str, value: Any, schema: Dict[str, Any]) -> List[str]:
    """Minimal JSON-Schema check for the argument shapes used by this module's tools."""
    t = schema.get("type")
    ok = {"string": isinstance(value, str), "boolean": isinstance(value, bool),
          "integer": isinstance(value, int) and not isinstance(value, bool),
          "number": isinstance(value, (int, float)) and not isinstance(value, bool),
          "array": isinstance(value, (list, tuple)), "object": isinstance(value, dict)}.get(t, True)
    if not ok:
        return [f"{name}: expected {t}"]
    errs: List[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{name}: must be one of {schema['enum']}")
    if t in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{name}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{name}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errs.append(f"{name}: must be greater than {schema['exclusiveMinimum']}")
    if t == "string" and len(value) < schema.get("minLength", 0):
        errs.append(f"{name}: must not be empty")
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errs += _check_type(f"{name}[{i}]", item, schema["items"])
    if t == "object":
        props = schema.get("properties")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if props is not None and k in props:
                errs += _check_type(f"{name}.{k}", v, props[k])
            elif isinstance(extra, dict):
                errs += _check_type(f"{name}.{k}", v, extra)
            elif extra is False:
                errs.append(f"{name}: unknown key {k!r}")
    return errs
