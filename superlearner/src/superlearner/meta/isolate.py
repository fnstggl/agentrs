"""Bootstrap that confines a candidate process before any candidate code runs.

    python -I -B isolate.py '<json config>' <module> [args...]

Applied in this order, each step fatal if it fails (there is no weaker fallback):

1. drop privileges - setgroups([]), setgid, setuid to an unprivileged account; ``PR_SET_NO_NEW_PRIVS``;
2. resource limits - processes, file size, open files, CPU seconds (the parent enforces wall-clock time);
3. Landlock        - the filesystem is an allowlist: the Python runtime, the candidate's own source tree and the
                     Crucible library read-only; the candidate's work directory read-write; nothing else (no home
                     directory, no credentials, no model weights, no other candidate, no evaluator data). TCP
                     bind/connect are denied (Landlock ABI >= 4);
4. seccomp         - no INET/INET6/netlink/packet sockets (unix sockets stay, for the LLM server and the env hub),
                     no ptrace, mount, namespaces, bpf, keyrings, module loading, or identity changes.

Then the requested module runs as ``__main__`` with ``sys.path`` = the candidate tree + Crucible.
This file uses only the standard library and is executed from the PROTECTED tree, never from a candidate's copy.
"""
import ctypes
import ctypes.util
import json
import os
import resource
import runpy
import sys

CFG = json.loads(sys.argv[1])
MODULE = sys.argv[2]
ARGS = sys.argv[3:]
libc = ctypes.CDLL(None, use_errno=True)


def die(msg: str) -> None:
    sys.stderr.write(f"isolate: {msg}\n")
    sys.exit(97)


for k in list(os.environ):
    if k not in CFG.get("env_keep", ()):
        del os.environ[k]
os.environ.update(CFG.get("env", {}))

# ---------------------------------------------------------------- 1. privileges
if os.getuid() == 0:
    try:
        os.setgroups([])
        os.setgid(CFG["gid"])
        os.setuid(CFG["uid"])
    except OSError as e:
        die(f"cannot drop privileges: {e}")
if os.getuid() == 0:
    die("refusing to run candidate code as root")
if libc.prctl(38, 1, 0, 0, 0) != 0:
    die("prctl(NO_NEW_PRIVS) failed")

# ---------------------------------------------------------------- 2. limits
for name, key in (("RLIMIT_NPROC", "nproc"), ("RLIMIT_FSIZE", "fsize"), ("RLIMIT_NOFILE", "nofile"), ("RLIMIT_CPU", "cpu_s")):
    if CFG.get(key):
        v = int(CFG[key])
        try:
            resource.setrlimit(getattr(resource, name), (v, v))
        except (ValueError, OSError) as e:
            die(f"setrlimit {name}: {e}")
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# ---------------------------------------------------------------- 3. landlock
LL_CREATE, LL_ADD, LL_RESTRICT = 444, 445, 446
FS_EXEC, FS_WRITE, FS_READ, FS_READDIR = 1, 2, 4, 8
FS_ALL_ABI1 = (1 << 13) - 1
FS_REFER, FS_TRUNCATE = 1 << 13, 1 << 14


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


abi = libc.syscall(LL_CREATE, None, 0, 1)
if abi < 1:
    die("Landlock is not available on this kernel")
fs = FS_ALL_ABI1 | (FS_REFER if abi >= 2 else 0) | (FS_TRUNCATE if abi >= 3 else 0)
attr = RulesetAttr(fs, 3 if abi >= 4 else 0)
rfd = libc.syscall(LL_CREATE, ctypes.byref(attr), ctypes.sizeof(RulesetAttr) if abi >= 4 else 8, 0)
if rfd < 0:
    die(f"landlock_create_ruleset errno={ctypes.get_errno()}")
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
            die(f"landlock_add_rule({path}) errno={ctypes.get_errno()}")
    finally:
        os.close(fd)


for p in CFG["read"]:
    allow(p, FS_EXEC | FS_READ | FS_READDIR)
for p in CFG["write"]:
    allow(p, fs & ~(1 << 6) & ~(1 << 11))
for p in CFG.get("devices", []):
    allow(p, FS_READ | FS_WRITE)
if libc.syscall(LL_RESTRICT, rfd, 0) != 0:
    die(f"landlock_restrict_self errno={ctypes.get_errno()}")
os.close(rfd)

# ---------------------------------------------------------------- 4. seccomp
DENY = ["ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2", "pivot_root", "chroot", "unshare",
        "setns", "keyctl", "add_key", "request_key", "bpf", "perf_event_open", "userfaultfd", "kexec_load",
        "kexec_file_load", "init_module", "finit_module", "delete_module", "reboot", "swapon", "swapoff",
        "sethostname", "setdomainname", "acct", "quotactl", "name_to_handle_at", "open_by_handle_at", "setuid",
        "setgid", "setreuid", "setregid", "setresuid", "setresgid", "setgroups", "personality"]
try:            # loaded by soname: `find_library` would run external programs, which the allowlist forbids
    sc = ctypes.CDLL("libseccomp.so.2")
except OSError:
    die("libseccomp is not available")
sc.seccomp_init.restype = ctypes.c_void_p
sc.seccomp_init.argtypes = [ctypes.c_uint32]
sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
sc.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = sc.seccomp_init(0x7FFF0000)
if not ctx:
    die("seccomp_init failed")
EPERM = 0x00050000 | 1
for name in DENY:
    nr = sc.seccomp_syscall_resolve_name(name.encode())
    if nr >= 0 and sc.seccomp_rule_add(ctx, EPERM, nr, 0) != 0:
        die(f"seccomp_rule_add({name}) failed")


class ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


sc.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ArgCmp)]
nr = sc.seccomp_syscall_resolve_name(b"socket")
for domain in (2, 10, 16, 17):            # AF_INET, AF_INET6, AF_NETLINK, AF_PACKET
    cmp = (ArgCmp * 1)(ArgCmp(0, 4, domain, 0))        # SCMP_CMP_EQ on argument 0
    if sc.seccomp_rule_add_array(ctx, EPERM, nr, 1, cmp) != 0:
        die("seccomp socket rule failed")
if sc.seccomp_load(ctx) != 0:
    die("seccomp_load failed")

# ---------------------------------------------------------------- run
os.chdir(CFG["cwd"])
sys.path[:] = list(CFG["sys_path"]) + [p for p in sys.path if p]
sys.dont_write_bytecode = True
sys.argv = [MODULE] + ARGS
if MODULE.endswith(".py"):
    runpy.run_path(MODULE, run_name="__main__")
else:
    runpy.run_module(MODULE, run_name="__main__", alter_sys=True)
