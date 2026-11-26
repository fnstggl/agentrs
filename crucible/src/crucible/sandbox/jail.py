"""Jail for native tools that parse agent-supplied bytes (FFmpeg, ffprobe, SoX) and for the Chromium that renders
agent-authored HTML/JavaScript (MASTER_SPEC Part 16: no unrestricted code execution outside the sandbox).

Executed as ``python -I -S -B jail.py <config.json> <program> [args...]`` (as root, with rlimits set by the
caller). Before it replaces itself with the program it

1. chowns the listed writable directories to the jail uid, then drops to that uid (no supplementary groups) and sets
   ``no_new_privs``;
2. applies Landlock: read-only access to the listed system paths, execute access only to the listed binaries (and the
   dynamic loader), read-write access only to the listed private directories, and (ABI >= 4) no TCP bind/connect;
3. applies seccomp: no socket other than AF_UNIX for the browser and no socket at all for tools; no ptrace, mount,
   namespaces, BPF, perf, io_uring, keyrings, kernel modules or uid changes; no System V IPC or POSIX message
   queues; no descriptor ownership (F_SETOWN/F_SETSIG, FIOSETOWN: kernel-sent signals); signals and process limits
   only on processes of the jail itself (by pid for tools; the browser signals its own children, which share the
   jail uid);
4. execs the program. Landlock and seccomp survive the exec and are inherited by every child.

The configuration is written by the trusted caller (`crucible.sandbox.isolation`); nothing in it comes from an
agent. Standard library only, like boot.py.
"""
import ctypes
import ctypes.util
import json
import os
import sys

CFG = json.load(open(sys.argv[1]))
ARGV = sys.argv[2:]
libc = ctypes.CDLL(None, use_errno=True)
def _traversable(path: str, uid: int, gid: int) -> bool:
    """Whether `uid` (no supplementary groups) may search every directory above `path` (DAC; Landlock comes on top)."""
    d = os.path.dirname(path)
    while True:
        try:
            st = os.stat(d)
        except OSError:
            return False
        if not (st.st_mode & 0o001 or (st.st_uid == uid and st.st_mode & 0o100)
                or (st.st_gid == gid and st.st_mode & 0o010)):
            return False
        if d == "/":
            return True
        d = os.path.dirname(d)


def _relocate(d: str) -> str:
    """A profile directory the jail uid cannot reach (its caller's TMPDIR below a root-only directory, e.g. an agent
    scratchpad of mode 700) is replaced by a fresh one in the jail's own home; stale ones (a day old) are pruned.
    DEFECTS DEF-P-16: the jailed Chromium could not create its profile and every browser test failed there."""
    import shutil
    import time as _t
    home = CFG.get("env", {}).get("HOME") or os.path.dirname(sys.argv[1])
    base = os.path.join(home, "profiles")
    os.makedirs(base, mode=0o700, exist_ok=True)
    for x in os.listdir(base):
        p = os.path.join(base, x)
        try:
            if _t.time() - os.stat(p).st_mtime > 86400:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
    new = os.path.join(base, f"{os.path.basename(d)}-{os.getpid()}")
    os.makedirs(new, mode=0o700, exist_ok=True)
    return new


def _private_home() -> None:
    """A fresh HOME / TMPDIR / XDG directory for this launch instead of the kind's shared home: two concurrent browsers
    of one kind (one uid) - e.g. two worlds' agent browsers - must not share writable state (profiles, caches), which a
    compromised renderer of one could otherwise read or poison for the other (wave 3b review, cross-tenant state).
    Directories of launches whose process is gone are pruned."""
    import secrets
    import shutil
    shared = CFG["env"]["HOME"]
    runs = os.path.join(shared, "runs")
    os.makedirs(runs, mode=0o755, exist_ok=True)
    os.chmod(runs, 0o755)
    for x in os.listdir(runs):
        p = os.path.join(runs, x)
        try:
            pid = int(x.split("-")[1])
            os.kill(pid, 0)
            continue                               # still running
        except (IndexError, ValueError, ProcessLookupError, PermissionError):
            pass
        except OSError:
            continue
        shutil.rmtree(p, ignore_errors=True)
    new = os.path.join(runs, f"run-{os.getpid()}-{secrets.token_hex(4)}")
    os.makedirs(new, mode=0o700)
    for k in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        if k in CFG["env"]:
            CFG["env"][k] = new
    CFG["rw"] = [p for p in CFG.get("rw", []) if p != shared] + [new]
    CFG["chown"] = [p for p in CFG.get("chown", []) if p != shared] + [new]


if CFG.get("private_home") and CFG.get("env", {}).get("HOME"):
    _private_home()

for _pre in CFG.get("rw_from_argv", []):          # e.g. the profile directory Playwright created for this launch
    for _i, _a in enumerate(ARGV[1:], 1):
        if _a.startswith(_pre):
            _d = os.path.realpath(_a[len(_pre):])
            if os.path.isdir(_d) and _d.startswith(("/tmp/", "/var/tmp/")) and _d.count("/") >= 2:
                if not _traversable(_d, CFG["uid"], CFG["gid"]):
                    _d = _relocate(_d)
                    ARGV[_i] = _pre + _d
                CFG.setdefault("rw", []).append(_d)
                CFG.setdefault("chown", []).append(_d)
ARGV += CFG.get("append_args", [])

# ------------------------------------------------------------------ 0. writable directories belong to the jail
for d in CFG.get("chown", []):
    for root, dirs, files in os.walk(d):
        for x in dirs + files:
            try:
                os.lchown(os.path.join(root, x), CFG["uid"], CFG["gid"])
            except OSError:
                pass
    os.chown(d, CFG["uid"], CFG["gid"])

# ------------------------------------------------------------------ 2. landlock
LL_CREATE, LL_ADD, LL_RESTRICT = 444, 445, 446
FS_EXEC, FS_WRITE, FS_READ, FS_READDIR = 1, 2, 4, 8
FS_ALL_ABI1 = (1 << 13) - 1
FS_REFER, FS_TRUNCATE = 1 << 13, 1 << 14


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class NetPort(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


def landlock() -> int:
    abi = libc.syscall(LL_CREATE, None, 0, 1)
    if abi < 1:
        raise SystemExit("jail: Landlock unavailable")
    fs = FS_ALL_ABI1 | (FS_REFER if abi >= 2 else 0) | (FS_TRUNCATE if abi >= 3 else 0)
    net = 3 if abi >= 4 else 0                     # TCP bind + connect handled, nothing allowed
    attr = RulesetAttr(fs, net)
    size = ctypes.sizeof(RulesetAttr) if abi >= 4 else 8
    rfd = libc.syscall(LL_CREATE, ctypes.byref(attr), size, 0)
    if rfd < 0:
        raise SystemExit(f"jail: landlock_create_ruleset errno={ctypes.get_errno()}")
    file_bits = FS_EXEC | FS_WRITE | FS_READ | (FS_TRUNCATE if abi >= 3 else 0)

    def allow(path: str, access: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not os.path.isdir(path):
                access &= file_bits
            pb = PathBeneath(access & fs, fd)
            if libc.syscall(LL_ADD, rfd, 1, ctypes.byref(pb), 0) != 0:
                raise SystemExit(f"jail: landlock_add_rule({path}) errno={ctypes.get_errno()}")
        finally:
            os.close(fd)

    for p in CFG.get("ro", []):
        allow(p, FS_READ | FS_READDIR)
    for p in CFG.get("exec", []):
        allow(p, FS_EXEC | FS_READ | FS_READDIR)
    rw = fs & ~(1 << 6) & ~(1 << 11)               # everything except creating char/block devices
    for p in CFG.get("rw", []):
        allow(p, rw)
    for p in ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random"):
        allow(p, FS_READ | FS_WRITE)
    for port in CFG.get("tcp_connect_ports", []):  # e.g. the world's own web app on loopback; nothing else
        if abi < 4:
            raise SystemExit("jail: per-port TCP rules need Landlock ABI 4")
        np_ = NetPort(2, int(port))                # LANDLOCK_ACCESS_NET_CONNECT_TCP
        if libc.syscall(LL_ADD, rfd, 2, ctypes.byref(np_), 0) != 0:   # LANDLOCK_RULE_NET_PORT
            raise SystemExit(f"jail: landlock net rule errno={ctypes.get_errno()}")
    return rfd


# the rules are built as root (a private directory may sit below directories the jail uid cannot traverse; the
# rule holds a descriptor, so the jail can use the directory through its working directory afterwards)
_RFD = landlock()
_SECCOMP = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2")   # resolved before any restriction

# ------------------------------------------------------------------ 1b. drop privileges
if os.getuid() == 0:
    os.setgroups([])
    os.setgid(CFG["gid"])
    os.setuid(CFG["uid"])
if os.getuid() == 0:
    raise SystemExit("jail: refusing to run as root")
if libc.prctl(38, 1, 0, 0, 0) != 0:              # PR_SET_NO_NEW_PRIVS
    raise SystemExit("jail: prctl(NO_NEW_PRIVS) failed")
for k, v in CFG.get("env", {}).items():
    os.environ[k] = v

if libc.syscall(LL_RESTRICT, _RFD, 0) != 0:
    raise SystemExit(f"jail: landlock_restrict_self errno={ctypes.get_errno()}")
os.close(_RFD)

# ------------------------------------------------------------------ 3. seccomp
DENY = ["ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2", "pivot_root", "chroot", "unshare",
        "setns", "keyctl", "add_key", "request_key", "bpf", "perf_event_open", "userfaultfd", "io_uring_setup",
        "io_uring_enter", "io_uring_register", "kexec_load", "kexec_file_load", "init_module", "finit_module",
        "delete_module", "reboot", "swapon", "swapoff", "sethostname", "setdomainname", "acct", "quotactl",
        "name_to_handle_at", "open_by_handle_at", "setuid", "setgid", "setreuid", "setregid", "setresuid",
        "setresgid", "setgroups", "personality", "pidfd_getfd", "kcmp", "process_madvise", "process_mrelease",
        "mount_setattr", "fsopen", "fsmount", "fspick", "move_mount", "open_tree",
        # System V IPC and POSIX message queues: kernel objects named by a key (not mediated by Landlock on this
        # kernel) that outlive the process - a store carried from one jailed learner process to the next and a
        # channel between processes of one uid (wave-3b review, v5.sandbox.hosted_learners_cannot_reach_each_other)
        "shmget", "shmat", "shmdt", "shmctl", "msgget", "msgsnd", "msgrcv", "msgctl", "semget", "semop",
        "semtimedop", "semctl", "mq_open", "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify",
        "mq_getsetattr"]
# descriptor ownership: the kernel signals the owner of a descriptor on I/O (SIGIO, or any signal chosen with
# F_SETSIG - SIGKILL included) after only a same-uid check, and F_SETOWN's ESRCH answers whether a pid exists. A
# jailed learner killed another learner process of its uid this way (same review). Refused in every mode.
FCNTL_DENY = [8, 10, 15]             # F_SETOWN, F_SETSIG, F_SETOWN_EX
IOCTL_DENY = [0x8901, 0x8902]        # FIOSETOWN, SIOCSPGRP (sockets)
# Advisory locks in tool mode (learner processes, FFmpeg, SoX): the only files a jailed tool shares with another
# process are read-only system files and /dev/null, where a lock is a channel between processes (flock() carried 16
# bits between two worlds' programs in the sandbox; same review). Taking a lock succeeds without effect; queries
# stay real (no jailed tool holds a lock, so they report none; sqlite3 needs them). (The browser keeps real locks:
# its profile databases lock their own private files.)
LOCK_NOOP_FCNTL = [6, 7, 37, 38]     # F_SETLK, F_SETLKW, F_OFD_SETLK, F_OFD_SETLKW
LOCK_DENY_FCNTL: list = []
TOOL_DENY = ["socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "fork", "vfork",
             "pidfd_open", "pidfd_send_signal"]
SELF_ONLY = [("kill", 0), ("tkill", 0), ("tgkill", 0), ("rt_sigqueueinfo", 0), ("rt_tgsigqueueinfo", 0)]


def seccomp() -> None:
    sc = _SECCOMP
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sc.seccomp_load.argtypes = [ctypes.c_void_p]

    class ArgCmp(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64),
                    ("datum_b", ctypes.c_uint64)]
    sc.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
                                          ctypes.POINTER(ArgCmp)]
    ctx = sc.seccomp_init(0x7FFF0000)              # allow by default
    eperm = 0x00050000 | 1
    tool = CFG.get("mode", "tool") == "tool"
    for name in DENY + (TOOL_DENY if tool else []):
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        if nr >= 0 and sc.seccomp_rule_add(ctx, eperm, nr, 0) != 0:
            raise SystemExit(f"jail: seccomp_rule_add({name}) failed")
    noop = 0x00050000                                  # SCMP_ACT_ERRNO(0): returns 0, does nothing
    rules = [("fcntl", c, eperm) for c in FCNTL_DENY] + [("ioctl", c, eperm) for c in IOCTL_DENY]
    if tool:
        rules += [("fcntl", c, eperm) for c in LOCK_DENY_FCNTL] + [("fcntl", c, noop) for c in LOCK_NOOP_FCNTL]
    for name, c, action in rules:
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        cmp = (ArgCmp * 1)(ArgCmp(1, 4, c, 0))           # SCMP_CMP_EQ on the command
        if nr >= 0 and sc.seccomp_rule_add_array(ctx, action, nr, 1, cmp) != 0:
            raise SystemExit(f"jail: seccomp {name} rule {c:#x} failed")
    if tool:
        nr = sc.seccomp_syscall_resolve_name(b"flock")
        if nr >= 0 and sc.seccomp_rule_add(ctx, noop, nr, 0) != 0:
            raise SystemExit("jail: seccomp flock rule failed")
    if tool:
        cmp = (ArgCmp * 1)(ArgCmp(0, 7, 0x00010000, 0))   # clone without CLONE_THREAD: no new processes
        nr = sc.seccomp_syscall_resolve_name(b"clone")
        if sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
            raise SystemExit("jail: seccomp clone rule failed")
        nr = sc.seccomp_syscall_resolve_name(b"clone3")
        if nr >= 0 and sc.seccomp_rule_add(ctx, 0x00050000 | 38, nr, 0) != 0:
            raise SystemExit("jail: seccomp clone3 rule failed")
        me = os.getpid()                           # exec keeps the pid: signals only to the tool itself
        for name, arg in SELF_ONLY:
            nr = sc.seccomp_syscall_resolve_name(name.encode())
            cmp = (ArgCmp * 1)(ArgCmp(arg, 1, me, 0))  # SCMP_CMP_NE
            if nr >= 0 and sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
                raise SystemExit(f"jail: seccomp self-only rule for {name} failed")
    elif not CFG.get("tcp_connect_ports"):
        for name in ("socket", "socketpair"):     # browser: only AF_UNIX (its own IPC)
            nr = sc.seccomp_syscall_resolve_name(name.encode())
            cmp = (ArgCmp * 1)(ArgCmp(0, 1, 1, 0))     # domain != AF_UNIX -> EPERM
            if nr >= 0 and sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
                raise SystemExit(f"jail: seccomp socket rule for {name} failed")
    else:
        # browser that must reach the world's web app: AF_UNIX, and IPv4 *stream* sockets (Landlock then allows
        # connect() only to the listed ports); no IPv6, raw, packet or netlink sockets, no UDP (no WebRTC/QUIC)
        for name in ("socket", "socketpair"):
            nr = sc.seccomp_syscall_resolve_name(name.encode())
            for fam in range(0, 46):
                if fam in (1, 2):
                    continue
                cmp = (ArgCmp * 1)(ArgCmp(0, 4, fam, 0))   # SCMP_CMP_EQ
                if nr >= 0 and sc.seccomp_rule_add_array(ctx, eperm, nr, 1, cmp) != 0:
                    raise SystemExit(f"jail: seccomp socket family rule {fam} failed")
            for typ in range(2, 11):
                cmp = (ArgCmp * 2)(ArgCmp(0, 4, 2, 0), ArgCmp(1, 7, 0xF, typ))   # AF_INET and (type & 0xf) == typ
                if nr >= 0 and sc.seccomp_rule_add_array(ctx, eperm, nr, 2, cmp) != 0:
                    raise SystemExit(f"jail: seccomp socket type rule {typ} failed")
    if sc.seccomp_load(ctx) != 0:
        raise SystemExit("jail: seccomp_load failed")


seccomp()
os.execv(ARGV[0], ARGV)
