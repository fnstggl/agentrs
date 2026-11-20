"""Sandbox bootstrap. Executed as `python -I -S -B boot.py <workdir> <config-json>` by runner.py (as root, with
rlimits already set by the parent). It drops to an unprivileged uid, applies Landlock and seccomp, and then serves
JSON-lines requests on stdin/stdout. User code never sees the real stdout (its prints are captured).

This file must only use the standard library and must not import crucible (it runs before any user code and
must be small and auditable).
"""
import ctypes
import ctypes.util
import io
import json
import os
import resource
import sys
import time
import traceback

WORKDIR = sys.argv[1]
CFG = json.loads(sys.argv[2])
PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
PROTO_IN = os.fdopen(os.dup(0), "r", buffering=1)
os.close(0)
os.open("/dev/null", os.O_RDONLY)  # fd 0

libc = ctypes.CDLL(None, use_errno=True)

# ------------------------------------------------------------------ 1. drop privileges
if os.getuid() == 0:
    os.setgroups([])
    os.setgid(CFG["gid"])
    os.setuid(CFG["uid"])
if os.getuid() == 0 and not CFG.get("allow_root"):
    raise SystemExit("sandbox: refusing to run user code as root")

PR_SET_NO_NEW_PRIVS = 38
if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
    raise SystemExit("sandbox: prctl(NO_NEW_PRIVS) failed")

# ------------------------------------------------------------------ 2. landlock (filesystem + TCP)
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


def landlock() -> int:
    abi = libc.syscall(LL_CREATE, None, 0, 1)
    if abi < 1:
        if CFG.get("require_landlock", True):
            raise SystemExit("sandbox: Landlock unavailable")
        return 0
    fs = FS_ALL_ABI1 | (FS_REFER if abi >= 2 else 0) | (FS_TRUNCATE if abi >= 3 else 0)
    net = (NET_BIND | NET_CONNECT) if abi >= 4 else 0
    attr = RulesetAttr(fs, net)
    size = ctypes.sizeof(RulesetAttr) if abi >= 4 else 8
    rfd = libc.syscall(LL_CREATE, ctypes.byref(attr), size, 0)
    if rfd < 0:
        raise SystemExit(f"sandbox: landlock_create_ruleset failed errno={ctypes.get_errno()}")
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
                raise SystemExit(f"sandbox: landlock_add_rule({path}) errno={ctypes.get_errno()}")
        finally:
            os.close(fd)

    ro = FS_EXEC | FS_READ | FS_READDIR
    for p in CFG["read_paths"]:
        allow(p, ro)
    rw = fs & ~(1 << 6) & ~(1 << 11)  # everything except creating char/block devices
    allow(WORKDIR, rw)
    allow("/dev/null", FS_READ | FS_WRITE)
    allow("/dev/urandom", FS_READ)
    if libc.syscall(LL_RESTRICT, rfd, 0) != 0:
        raise SystemExit(f"sandbox: landlock_restrict_self errno={ctypes.get_errno()}")
    os.close(rfd)
    return abi


LANDLOCK_ABI = landlock()

# ------------------------------------------------------------------ 3. seccomp (no sockets, no exec, no kernel attack surface)
DENY = ["socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "execve", "execveat", "ptrace",
        "process_vm_readv", "process_vm_writev", "mount", "umount2", "pivot_root", "chroot", "unshare", "setns",
        "keyctl", "add_key", "request_key", "bpf", "perf_event_open", "userfaultfd", "io_uring_setup",
        "io_uring_enter", "io_uring_register", "kexec_load", "kexec_file_load", "init_module", "finit_module",
        "delete_module", "reboot", "swapon", "swapoff", "sethostname", "setdomainname", "acct", "quotactl",
        "name_to_handle_at", "open_by_handle_at", "setuid", "setgid", "setreuid", "setregid", "setresuid",
        "setresgid", "setgroups", "personality"]


def seccomp() -> bool:
    path = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    try:
        sc = ctypes.CDLL(path)
    except OSError:
        if CFG.get("require_seccomp", True):
            raise SystemExit("sandbox: libseccomp unavailable")
        return False
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sc.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sc.seccomp_load.argtypes = [ctypes.c_void_p]
    ctx = sc.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise SystemExit("sandbox: seccomp_init failed")
    errno_eperm = 0x00050000 | 1  # SCMP_ACT_ERRNO(EPERM)
    errno_enosys = 0x00050000 | 38  # SCMP_ACT_ERRNO(ENOSYS): glibc then falls back from clone3 to clone
    for name in DENY + ["fork", "vfork"]:
        nr = sc.seccomp_syscall_resolve_name(name.encode())
        if nr < 0:
            continue
        if sc.seccomp_rule_add(ctx, errno_eperm, nr, 0) != 0:
            raise SystemExit(f"sandbox: seccomp_rule_add({name}) failed")
    nr = sc.seccomp_syscall_resolve_name(b"clone3")
    if nr >= 0 and sc.seccomp_rule_add(ctx, errno_enosys, nr, 0) != 0:
        raise SystemExit("sandbox: seccomp_rule_add(clone3) failed")

    class ArgCmp(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("datum_a", ctypes.c_uint64),
                    ("datum_b", ctypes.c_uint64)]
    sc.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
                                          ctypes.POINTER(ArgCmp)]
    clone_thread = 0x00010000
    cmp = (ArgCmp * 1)(ArgCmp(0, 7, clone_thread, 0))  # SCMP_CMP_MASKED_EQ: (flags & CLONE_THREAD) == 0
    nr = sc.seccomp_syscall_resolve_name(b"clone")
    if sc.seccomp_rule_add_array(ctx, errno_eperm, nr, 1, cmp) != 0:
        raise SystemExit("sandbox: seccomp clone rule failed")
    if sc.seccomp_load(ctx) != 0:
        raise SystemExit("sandbox: seccomp_load failed")
    return True


SECCOMP = seccomp()

# ------------------------------------------------------------------ 4. environment for user code
os.chdir(WORKDIR)
sys.path[:] = [WORKDIR] + list(CFG.get("sys_path", []))
sys.dont_write_bytecode = True
_rpc_id = 0


def _send(msg: dict) -> None:
    PROTO_OUT.write(json.dumps(msg, separators=(",", ":")) + "\n")
    PROTO_OUT.flush()


def _recv() -> dict:
    line = PROTO_IN.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


class WorldError(Exception):
    def __init__(self, err: dict):
        super().__init__(err.get("message", "error"))
        self.code = err.get("code")
        self.error = err


def call(tool: str, **args):
    """Call a Crucible tool as this program's principal (brokered and authorised by the kernel)."""
    global _rpc_id
    _rpc_id += 1
    _send({"op": "rpc", "id": _rpc_id, "tool": tool, "args": args})
    while True:
        m = _recv()
        if m.get("op") == "rpc_result" and m.get("id") == _rpc_id:
            if m["ok"]:
                return m["result"]
            raise WorldError(m["error"])


sdk = type(sys)("crucible_sdk")
sdk.call = call
sdk.WorldError = WorldError
sdk.__doc__ = "In-sandbox SDK: call(tool, **args) reaches the synthetic internet through the kernel broker."
sys.modules["crucible_sdk"] = sdk


def _cpu_ms() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime + r.ru_stime) * 1000.0


def _resolve(spec: str):
    mod_name, _, fn_name = spec.partition(":")
    if mod_name.endswith(".py"):
        mod_name = mod_name[:-3]
    mod_name = mod_name.replace("/", ".")
    mod = __import__(mod_name, fromlist=["_"])
    return getattr(mod, fn_name) if fn_name else mod


def _run(job: dict) -> dict:
    out_cap = int(CFG.get("out_bytes", 65536))
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    c0, w0 = _cpu_ms(), time.perf_counter()
    try:
        kind = job.get("kind", "call")
        if kind == "call":
            fn = _resolve(job["target"])
            result = fn(job.get("args"), job.get("caller")) if job.get("two_args", True) else fn(job.get("args"))
        elif kind == "import":
            __import__(job["target"], fromlist=["_"])
            result = {"imported": job["target"]}
        elif kind == "tests":
            result = _run_tests(job.get("pattern", "tests"))
        else:
            raise ValueError(f"unknown job kind {kind}")
        json.dumps(result)
        ok, err = True, None
    except SystemExit as e:
        ok, result, err = False, None, {"type": "SystemExit", "message": str(e)}
    except BaseException as e:  # noqa: BLE001 - user code may raise anything
        ok, result = False, None
        tb = traceback.format_exc(limit=6)
        err = {"type": type(e).__name__, "message": str(e)[:2000], "traceback": tb[-4000:]}
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return {"ok": ok, "result": result, "error": err, "stdout": buf.getvalue()[:out_cap],
            "cpu_ms": round(_cpu_ms() - c0, 3), "wall_ms": round((time.perf_counter() - w0) * 1000.0, 3)}


def _run_tests(folder: str) -> dict:
    import importlib
    results = []
    root = os.path.join(WORKDIR, folder)
    files = sorted(f for f in os.listdir(root) if f.startswith("test_") and f.endswith(".py")) if os.path.isdir(root) else []
    for f in files:
        modname = f"{folder}.{f[:-3]}".replace("/", ".")
        try:
            mod = importlib.import_module(modname)
        except BaseException as e:  # noqa: BLE001
            results.append({"test": modname, "ok": False, "error": f"import: {type(e).__name__}: {e}"[:500]})
            continue
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                try:
                    getattr(mod, name)()
                    results.append({"test": f"{modname}.{name}", "ok": True})
                except BaseException as e:  # noqa: BLE001
                    results.append({"test": f"{modname}.{name}", "ok": False,
                                    "error": f"{type(e).__name__}: {e}"[:500]})
    return {"passed": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results), "results": results}


_send({"op": "ready", "landlock_abi": LANDLOCK_ABI, "seccomp": SECCOMP, "uid": os.getuid()})
while True:
    msg = _recv()
    if msg.get("op") == "job":
        res = _run(msg)
        res["op"], res["id"] = "done", msg["id"]
        _send(res)
    elif msg.get("op") == "exit":
        break
