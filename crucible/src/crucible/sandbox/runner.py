"""Isolated execution of untrusted (agent-written) programs.

Layers (all verified by tests/test_sandbox.py):

1. separate process in its own session, started with an empty environment;
2. rlimits: address space, CPU seconds, processes (per uid), open files, file size, no core dumps;
3. privileges dropped to an unprivileged uid (default `nobody`) before any user code runs;
4. Landlock: read-only access to the Python standard library, approved packages and system libraries; read-write
   access only to the job's own working directory; all TCP bind/connect denied (ABI >= 4);
5. seccomp: no socket creation, no exec, no ptrace/mount/namespace/BPF/io_uring/keyring syscalls; signals,
   resource limits and scheduling changes only on the calling process itself (all workers share one unprivileged
   uid, so the kernel alone would let one tenant's code signal another tenant's worker);
6. wall-clock timeouts per job, enforced by the parent (the whole process group is killed);
7. the only channel to the outside is a JSON-lines pipe to the parent; calls into the synthetic internet go through
   `crucible_sdk.call(tool, **args)`, which the kernel authorises and records like any other action;
8. accounting from outside (D-20, `crucible.sandbox.metering`): CPU, wall time, memory, threads and disk are measured
   by the parent; the process only supplies the program's output. Threads that outlive a job, work done while idle
   and a working directory over its quota kill the worker; per-tenant uids (`SandboxConfig.tenant`) keep one world's
   processes out of another world's process budget.

If Landlock or seccomp is unavailable the runner refuses to start unless `allow_weak=True` is passed explicitly.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import pwd
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import metering as M

BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")
APPROVED_PACKAGES = ("numpy",)
_lock = threading.Lock()
_counter = 0
TICK_S = 0.025                 # how often the parent checks a running job's disk use
TENANT_UID_BASE = 0x5C000000   # per-tenant sandbox uids: TENANT_UID_BASE + 16-bit hash of the tenant
# Trusted modules the boot module imports before the first job when the program's files name them (their import cost
# is the platform's, not the program's work; D-20 bills the program's own load-time work). Pure-computation stdlib
# modules without import-time side effects; approved packages are added from SandboxConfig.packages.
PRELOAD_STDLIB = frozenset((
    "abc", "array", "ast", "base64", "binascii", "bisect", "bz2", "calendar", "cmath", "codecs", "collections",
    "colorsys", "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "difflib", "enum", "fnmatch",
    "fractions", "functools", "graphlib", "gzip", "hashlib", "heapq", "hmac", "html", "inspect", "io", "itertools",
    "json", "keyword", "lzma", "math", "numbers", "operator", "pathlib", "pprint", "queue", "random", "re",
    "secrets", "shlex", "statistics", "string", "struct", "textwrap", "threading", "time", "tokenize", "types",
    "typing", "unicodedata", "uuid", "weakref", "zipfile", "zlib"))
MAX_PRELOAD = 32
_IMPORT_RE = __import__("re").compile(rb"^[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import\b|import[ \t]+([^\n#;]+))",
                                      __import__("re").M)


@dataclass
class SandboxConfig:
    user: str = "nobody"
    cpu_s: int = 600            # lifetime CPU cap of a worker process
    mem_mb: int = 1536          # address-space cap
    job_wall_s: float = 10.0    # per-job wall-clock timeout
    nproc: int = 512            # per-uid process/thread cap (protects the container pid budget)
    nofile: int = 64
    fsize_mb: int = 16
    out_bytes: int = 65536
    max_files: int = 200
    max_file_bytes: int = 256 * 1024
    packages: tuple = APPROVED_PACKAGES
    base_dir: str = field(default_factory=lambda: os.environ.get("CRUCIBLE_SANDBOX_DIR", "/var/tmp/crucible-sbx"))
    allow_weak: bool = False
    allow_root: bool = False
    count_lines: bool = False   # deterministic cost model: bill the first parent-side measurement of a computation
    meter_scope: str | None = None  # the world lineage the deterministic memo is keyed by
    memo_path: str | None = None    # machine-level memo file shared across processes (metering.DiskMemo); None: per process
    tenant: str | None = None   # own uid per tenant (e.g. per world); None = the shared `user`
    disk_quota_mb: int = 64     # bytes a worker's working directory may hold (checked by the parent)
    max_entries: int = 2000     # files and directories a worker's working directory may hold


class SandboxUnavailable(RuntimeError):
    pass


def tenant_ids(cfg: SandboxConfig) -> tuple[int, int]:
    """(uid, gid) the sandboxed process runs as: the configured user, or one uid per tenant."""
    if cfg.tenant:
        h = int(hashlib.blake2b(str(cfg.tenant).encode(), digest_size=2).hexdigest(), 16)
        return TENANT_UID_BASE + h, TENANT_UID_BASE + h
    pw = pwd.getpwnam(cfg.user)
    return pw.pw_uid, pw.pw_gid


def capabilities() -> dict:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, None, 0, 1)
    seccomp = ctypes.util.find_library("seccomp") is not None or os.path.exists("/usr/lib/x86_64-linux-gnu/libseccomp.so.2")
    try:
        pwd.getpwnam("nobody")
        nobody = True
    except KeyError:
        nobody = False
    return {"landlock_abi": max(abi, 0), "landlock_net": abi >= 4, "seccomp": seccomp, "nobody": nobody,
            "root": os.getuid() == 0}


def _read_paths(packages: tuple) -> tuple[list[str], list[str]]:
    """(paths readable inside the sandbox, sys.path for the sandbox)."""
    stdlib = sysconfig.get_paths()["stdlib"]
    site = sysconfig.get_paths()["purelib"]
    reads: list[str] = []
    for name in sorted(os.listdir(stdlib)):
        p = os.path.join(stdlib, name)
        if os.path.realpath(p) == os.path.realpath(site) or name in ("site-packages", "dist-packages"):
            continue
        reads.append(p)
    prefix_lib = os.path.join(sys.prefix, "lib")
    for name in sorted(os.listdir(prefix_lib)):
        if ".so" in name:
            reads.append(os.path.join(prefix_lib, name))
    for p in ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/etc/ld.so.cache", "/etc/localtime", "/usr/share/zoneinfo"):
        if os.path.exists(p):
            reads.append(p)
    pkg_paths: list[str] = []
    for pkg in packages:
        d = os.path.join(site, pkg)
        if os.path.isdir(d):
            reads.append(d)
            for extra in sorted(os.listdir(site)):
                if extra.startswith(pkg) and (extra.endswith(".libs") or extra.endswith(".dist-info")):
                    reads.append(os.path.join(site, extra))
    syspath = [stdlib, os.path.join(stdlib, "lib-dynload")]
    return reads, syspath


def _pkg_dir(cfg: SandboxConfig) -> str:
    """A directory containing symlinks to the approved packages only (put on the sandbox sys.path)."""
    site = sysconfig.get_paths()["purelib"]
    d = os.path.join(cfg.base_dir, "_pkgs")
    os.makedirs(d, exist_ok=True)
    os.chmod(d, 0o755)
    for pkg in cfg.packages:
        src = os.path.join(site, pkg)
        dst = os.path.join(d, pkg)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
        for extra in os.listdir(site):
            if extra.startswith(pkg) and extra.endswith(".libs"):
                dst2 = os.path.join(d, extra)
                if not os.path.lexists(dst2):
                    os.symlink(os.path.join(site, extra), dst2)
    return d


def preload_list(files: dict, packages: tuple) -> list[str]:
    """Trusted modules the program's files import (a parent-side text scan; nothing of the program runs): stdlib
    modules of PRELOAD_STDLIB and approved packages, with their dotted submodules, at most MAX_PRELOAD. A module the
    scan misses (a computed `__import__`) is simply imported by the program itself and billed as its load work."""
    trusted = PRELOAD_STDLIB | set(packages)
    out: list[str] = []
    for rel in sorted(files):
        if not rel.endswith(".py"):
            continue
        data = files[rel]
        data = data.encode() if isinstance(data, str) else bytes(data)
        for m in _IMPORT_RE.finditer(data):
            names = [m.group(1)] if m.group(1) else [x.strip().split()[0] for x in m.group(2).split(b",") if x.strip()]
            for raw in names:
                try:
                    name = raw.decode("ascii")
                except UnicodeDecodeError:
                    continue
                parts = name.split(".")
                if parts[0] in trusted and all(p.isidentifier() for p in parts) and name not in out:
                    out.append(name)
                    if len(out) >= MAX_PRELOAD:
                        return out
    return out


def _no_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def _size(x: Any) -> int:
    try:
        return len(json.dumps(x, separators=(",", ":")))
    except (TypeError, ValueError):
        return 0


def _safe_rel(path: str) -> str:
    p = os.path.normpath(path).lstrip("/")
    if p.startswith("..") or os.path.isabs(p) or "\x00" in p or p in ("", "."):
        raise ValueError(f"unsafe path {path!r}")
    return p


class Worker:
    """A sandboxed Python process serving jobs over a pipe. Used once (programs) or kept alive (services).

    The parent decides everything about a job from outside the process (D-20, `crucible.sandbox.metering`): CPU,
    wall time, memory, threads and disk are measured here; the process only supplies the program's own output
    (result, error, captured stdout), which is validated and bounded. A worker that leaves threads running after a
    job, works while idle, or exceeds its disk quota is killed (and the work is billed)."""

    def __init__(self, files: dict[str, str], cfg: SandboxConfig | None = None, label: str = "job"):
        self.cfg = cfg or SandboxConfig()
        self.files = files
        self.label = label
        self.proc: subprocess.Popen | None = None
        self.workdir: str | None = None
        self.info: dict = {}
        self._buf = b""
        self.jobs = 0
        self.dead_reason: str | None = None
        self.files_digest = M.digest(sorted((k, hashlib.blake2b(v.encode() if isinstance(v, str) else bytes(v),
                                                                  digest_size=16).hexdigest())
                                            for k, v in files.items()))
        self._idle_cpu: int | None = None     # process CPU when the last job's result arrived
        self._loaded: set[str] = set()
        self.carry_ns = 0                     # background work of a predecessor worker, billed with the next job
        self.uid: int | None = None
        self._last_bytes = 0
        # Crucible 4.0 historical cost model (D-82): only with the deterministic cost model, in a process explicitly
        # started as the 4.0 historical baseline (CRUCIBLE_HISTORICAL_BASELINE=4.0); never in a 5.0 world
        from ..experience.modes import historical_baseline
        self._v4 = bool(self.cfg.count_lines) and historical_baseline() == "4.0"

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Worker":
        global _counter
        cfg = self.cfg
        caps = capabilities()
        if not cfg.allow_weak and (caps["landlock_abi"] < 1 or not caps["seccomp"]):
            raise SandboxUnavailable(f"Landlock/seccomp required for untrusted code (found {caps})")
        if len(self.files) > cfg.max_files:
            raise ValueError("too many files")
        uid, gid = tenant_ids(cfg)
        self.uid = uid
        os.makedirs(cfg.base_dir, exist_ok=True)
        os.chmod(cfg.base_dir, 0o711)
        with _lock:
            _counter += 1
            n = _counter
        self.workdir = os.path.join(cfg.base_dir, f"{os.getpid()}-{n}-{int(time.time() * 1000) % 10**9}")
        os.makedirs(self.workdir, mode=0o700)
        for rel, content in self.files.items():
            rel = _safe_rel(rel)
            data = content.encode() if isinstance(content, str) else bytes(content)
            if len(data) > cfg.max_file_bytes:
                raise ValueError(f"file too large: {rel}")
            full = os.path.join(self.workdir, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)
        for root, dirs, fnames in os.walk(self.workdir):
            for x in dirs + fnames:
                os.chown(os.path.join(root, x), uid, gid)
            if root != self.workdir:
                os.chown(root, uid, gid)
        os.chown(self.workdir, uid, gid)
        reads, syspath = _read_paths(cfg.packages)
        pkgs = _pkg_dir(cfg)
        reads.append(pkgs)
        # boot.py itself is not readable from inside: it is fully loaded before Landlock is applied, and the
        # simulator's sandbox package (isolation settings, broker protocol, host paths) is none of the tenant's
        # business
        boot_cfg = {"uid": uid, "gid": gid, "read_paths": reads,
                    "sys_path": syspath + [pkgs], "out_bytes": cfg.out_bytes, "allow_root": cfg.allow_root,
                    "require_landlock": not cfg.allow_weak, "require_seccomp": not cfg.allow_weak,
                    "preload": preload_list(self.files, tuple(cfg.packages)),
                    "v4_count_lines": self._v4}               # 4.0 historical cost model only (D-82, boot.py)
        env = {"PATH": "/nonexistent", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": self.workdir}
        limits = (cfg.mem_mb, cfg.cpu_s, cfg.nproc, cfg.nofile, cfg.fsize_mb)

        def pre():
            mem, cpu, nproc, nofile, fsize = limits
            resource.setrlimit(resource.RLIMIT_AS, (mem * 2**20, mem * 2**20))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize * 2**20, fsize * 2**20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        backend = getattr(cfg, "backend", None) or os.environ.get("CRUCIBLE_SANDBOX_BACKEND") or "landlock-seccomp"
        from .backends import enforce_min_isolation, require, wrap
        enforce_min_isolation(backend, getattr(cfg, "min_isolation", None) or os.environ.get("CRUCIBLE_MIN_ISOLATION"))
        if backend != "landlock-seccomp":
            require(backend)                      # never a silent fallback to weaker isolation
        argv = wrap(backend, [sys.executable, "-I", "-S", "-B", BOOT, self.workdir, json.dumps(boot_cfg)], self.workdir)
        self.proc = subprocess.Popen(argv,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, cwd="/", preexec_fn=pre, start_new_session=True, close_fds=True)
        msg = self._read_msg(time.monotonic() + 30)
        if msg is None or msg.get("op") != "ready":
            err = self._stderr_tail()
            self.close()
            raise SandboxUnavailable(f"sandbox failed to start: {msg} {err}")
        self.info = msg
        self._idle_cpu = M.cpu_ns(self.proc.pid)
        return self

    def exited(self) -> bool:
        """Whether the process has terminated, WITHOUT reaping it: a terminated, unreaped process still reports its
        CPU clock, so work done right before it died (e.g. a program that kills itself after its world calls) is
        measured before the parent reaps it (in close/_kill)."""
        if self.proc is None or self.proc.returncode is not None:
            return True
        try:
            return os.waitid(os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True

    def alive(self) -> bool:
        return self.proc is not None and not self.exited()

    def idle_cpu_ns(self) -> int:
        """CPU the process used since its last job's result (0 when busy or unknown); readable after it died."""
        if self.proc is None or self._idle_cpu is None:
            return 0
        c = M.cpu_ns(self.proc.pid)
        return max(0, c - self._idle_cpu) if c is not None else 0

    def close(self) -> None:
        """Stop the process at once (SIGKILL to its process group: nothing in it outlives the worker, and nothing
        runs unbilled during a graceful shutdown) and delete its working directory."""
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                pass
            for s in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            self.proc = None
        if self.workdir and os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ protocol
    def _read_msg(self, deadline: float) -> dict | None:
        fd = self.proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        try:
            while b"\n" not in self._buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                if not sel.select(left):
                    return None
                chunk = os.read(fd, 65536)
                if not chunk:
                    return {"op": "eof"}
                self._buf += chunk
                if len(self._buf) > 8 * 2**20:
                    return {"op": "overflow"}
            line, _, self._buf = self._buf.partition(b"\n")
            self._last_bytes = len(line)
            try:                              # hostile bytes: deep nesting, huge integers, NaN/Infinity
                m = json.loads(line, parse_constant=_no_constant)
            except (ValueError, RecursionError, MemoryError):
                return {"op": "garbage"}
            return m if isinstance(m, dict) else {"op": "garbage"}
        finally:
            sel.close()

    def _write(self, msg: dict) -> int:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        return len(data)

    def _stderr_tail(self) -> str:
        try:
            if self.proc and self.exited():
                return self._clean(self.proc.stderr.read()[-2000:].decode(errors="replace"))
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _clean(self, text: Any) -> Any:
        """Host paths never reach a participant (and recorded outcomes do not depend on the worker's directory)."""
        if isinstance(text, str):
            if self.workdir:
                text = text.replace(self.workdir, "/sandbox")
            text = text.replace(BOOT, "<sandbox>").replace(os.path.dirname(BOOT), "<sandbox>")
            return text.replace(self.cfg.base_dir, "/sandbox")
        if isinstance(text, dict):
            return {k: self._clean(v) for k, v in text.items()}
        return text

    def _kill_measured(self) -> int | None:
        """CPU of the process just before it is killed."""
        c = M.cpu_ns(self.proc.pid) if self.proc is not None else None
        self._kill()
        return c

    def _over_quota(self) -> str | None:
        b, n = M.dir_usage(self.workdir, self.cfg.max_entries)
        if n > self.cfg.max_entries:
            return f"more than {self.cfg.max_entries} files in the working directory"
        if b > self.cfg.disk_quota_mb * 2**20:
            return f"working directory exceeds its {self.cfg.disk_quota_mb} MiB quota"
        return None

    def _finish(self, ok: bool, result: Any, error: dict | None, stdout: str, *, cpu: int, job: dict,
                transcript: list, wall_s: float, meter: dict, proto_bytes: int = 0, v4_lines: Any = None) -> dict:
        """The job's outcome with the parent's measurements; the deterministic cost model bills through the memo
        (or, in a Crucible 4.0 historical baseline process, by 4.0's executed-line count: D-82)."""
        error = self._clean(error) if error else None
        stdout = self._clean(stdout or "")[: self.cfg.out_bytes]
        cpu = max(0, int(cpu))
        billed, memo = cpu, None
        if self._v4 and isinstance(v4_lines, int) and not isinstance(v4_lines, bool):
            meter = {**meter, "cpu_ns": cpu, "billed_ns": cpu, "wall_ms": round(wall_s * 1000.0, 3),
                     "cost_model": "crucible-4.0 executed lines (historical baseline, D-82)"}
            return {"ok": bool(ok), "result": result if ok else None, "error": error, "stdout": stdout,
                    "cpu_ms": round(cpu / 1e6, 4), "wall_ms": meter["wall_ms"], "rpc_calls": len(transcript),
                    "meter": meter, "work_lines": max(0, v4_lines)}
        if self.cfg.count_lines:
            outcome = {"ok": ok, "result": result if ok else None,
                       "error": [(error or {}).get("type"), (error or {}).get("message")] if error else None}
            billed, memo = M.MEMO.bill(M.computation_key(self.cfg.meter_scope, self.files_digest, job, transcript,
                                                         outcome), cpu, disk=self.cfg.memo_path)
        meter = {**meter, "cpu_ns": cpu, "billed_ns": int(billed), "wall_ms": round(wall_s * 1000.0, 3),
                 **({"memo": memo} if memo else {})}
        out = {"ok": bool(ok), "result": result if ok else None, "error": error, "stdout": stdout,
               "cpu_ms": round(billed / 1e6, 4), "wall_ms": meter["wall_ms"], "rpc_calls": len(transcript),
               "meter": meter}
        if self.cfg.count_lines:
            out["work_lines"] = M.work_units(billed, proto_bytes)   # parent-measured work units (never self-counted)
        return out

    def run(self, job: dict, rpc: Callable[[str, dict], tuple[bool, Any]] | None = None,
            timeout: float | None = None, max_rpc: int = 50) -> dict:
        """Run one job. `rpc(tool, args) -> (ok, result_or_error)` serves nested world calls."""
        t0 = time.monotonic()
        empty = {"threads": 0, "maxrss_kb": 0, "disk_bytes": 0, "background_ns": 0, "load_ns": 0, "killed": None}
        if not self.alive():
            # work it did after its previous result before it died (a forged completion, then exit) is billed here
            bg = int(self.carry_ns) + self.idle_cpu_ns()
            self.carry_ns = 0
            self._idle_cpu = None
            self._kill()
            return self._finish(False, None, {"type": "WorkerDead", "message": self.dead_reason or "not running"}, "",
                                cpu=bg, job=job, transcript=[], wall_s=0.0, meter={**empty, "background_ns": bg})
        pid = self.proc.pid
        timeout = timeout or self.cfg.job_wall_s
        transcript: list = []
        # 1. work done since the previous job's result (threads or code left running): billed here, worker killed.
        # A predecessor's background work (carry_ns, the pool's debt of a worker it already killed) is billed with
        # this job too, but does not stop this fresh worker.
        carry = int(self.carry_ns)
        self.carry_ns = 0
        own = 0
        now = M.cpu_ns(pid)
        if self._idle_cpu is not None and now is not None:
            own = max(0, now - self._idle_cpu)
        bg = carry + own
        self._idle_cpu = None                 # busy: a pool sweep (e.g. from a nested world call) leaves it alone
        if own > M.BACKGROUND_SLACK_NS:
            reason = "the process kept working after its previous job returned"
            self.dead_reason = reason
            c = self._kill_measured()
            if c is not None and now is not None:
                bg += max(0, c - now)
            return self._finish(False, None, {"type": "WorkerDied", "message": reason}, "", cpu=bg, job=job,
                                transcript=transcript, wall_s=time.monotonic() - t0,
                                meter={**empty, "background_ns": bg, "killed": reason})

        def died(reason: str, etype: str = "WorkerDied", c_start: int | None = None, extra: int = 0) -> dict:
            self.dead_reason = reason
            c = M.cpu_ns(pid)                 # first: a terminated process is readable until it is reaped
            tail = self._stderr_tail()
            c2 = self._kill_measured()
            if c is None or (c2 is not None and c2 > c):
                c = c2
            used = max(0, (c or 0) - (c_start or 0)) if c is not None and c_start is not None else 0
            err = {"type": etype, "message": reason, **({"stderr": tail[-800:]} if tail and etype != "Timeout" else {})}
            return self._finish(False, None, err, "", cpu=used + bg + extra, job=job, transcript=transcript,
                                wall_s=time.monotonic() - t0, meter={**empty, "background_ns": bg, "killed": reason})

        # 2. load the program (its import-time work is measured apart; only what exceeds LOAD_FREE_NS is billed)
        load_ns = 0
        target = job.get("target") if job.get("kind", "call") == "call" else None
        if isinstance(target, str) and target not in self._loaded:
            self.jobs += 1
            lid = self.jobs
            c0 = M.cpu_ns(pid)
            try:
                self._write({"op": "load", "id": lid, "target": target})
            except OSError:
                return died("process exited before the program could be loaded", c_start=c0)
            msg = self._read_msg(time.monotonic() + timeout)
            if msg is None:
                return died(f"exceeded {timeout}s while loading the program", "Timeout", c_start=c0)
            if msg.get("op") != "loaded" or msg.get("id") != lid:
                reason = {"eof": "process exited (crash, memory or CPU limit)", "overflow": "output too large",
                          "garbage": "protocol violation"}.get(msg.get("op"), "protocol violation")
                return died(reason, c_start=c0)
            M.settle(pid)
            c1 = M.cpu_ns(pid)
            load_ns = max(0, (c1 or 0) - (c0 or 0))
            if msg.get("ok") is True:
                self._loaded.add(target)
        load_bill = max(0, load_ns - M.LOAD_FREE_NS)
        # 3. the job
        self.jobs += 1
        jid = self.jobs
        deadline = time.monotonic() + timeout
        broker_s = 0.0
        c_start = M.cpu_ns(pid)
        proto = _size(job)                    # protocol work (sizes of what is decoded/encoded), never billed
        try:
            self._write({"op": "job", "id": jid, **job})
        except OSError:                       # the process is gone (killed, crashed): an ordinary failed job
            return died("process exited before the job could be sent", c_start=c_start, extra=load_bill)
        while True:
            msg = self._read_msg(min(deadline, time.monotonic() + TICK_S))
            if msg is None:
                if time.monotonic() >= deadline:
                    return died(f"exceeded {timeout}s", "Timeout", c_start=c_start, extra=load_bill)
                q = self._over_quota()
                if q:
                    return died(q, "DiskQuota", c_start=c_start, extra=load_bill)
                continue
            op = msg.get("op")
            if op == "rpc":
                rid = msg.get("id")
                if rpc is None or len(transcript) >= max_rpc:
                    ok, val = False, {"code": "forbidden", "message": "no network calls available"}
                else:
                    pause = time.monotonic()
                    ok, val = rpc(str(msg.get("tool", "")), msg.get("args") if isinstance(msg.get("args"), dict) else {})
                    spent = time.monotonic() - pause
                    deadline += spent                 # time spent in the broker is not the program's
                    broker_s += spent
                transcript.append([str(msg.get("tool", "")), M.digest(msg.get("args")), bool(ok), M.digest(val)])
                reply = {"op": "rpc_result", "id": rid, "ok": ok, **({"result": val} if ok else {"error": val})}
                proto += _size(msg.get("args")) + _size(val)
                try:
                    self._write(reply)
                except OSError:               # the process died while its call was being served
                    return died("process exited during a world call", c_start=c_start, extra=load_bill)
                continue
            if op == "done" and msg.get("id") == jid and isinstance(msg.get("ok"), bool) \
                    and (msg.get("error") is None or isinstance(msg.get("error"), dict)) \
                    and isinstance(msg.get("stdout", ""), str):
                M.settle(pid)
                c_end = M.cpu_ns(pid)
                n_thr = M.threads(pid)
                hwm = M.hwm_kb(pid)
                disk, _n = M.dir_usage(self.workdir, self.cfg.max_entries)
                killed = None
                if n_thr > 1:
                    killed = f"{n_thr - 1} thread(s) outlived the job"
                q = self._over_quota()
                if q:
                    killed = q
                if killed:
                    self.dead_reason = killed
                    c_kill = self._kill_measured()
                    if c_kill is not None and c_end is not None:
                        c_end = max(c_end, c_kill)
                else:
                    self._idle_cpu = c_end
                used = max(0, (c_end or 0) - (c_start or 0)) if c_end is not None and c_start is not None else 0
                meter = {"threads": n_thr, "maxrss_kb": hwm, "disk_bytes": disk, "background_ns": bg,
                         "load_ns": load_ns, "killed": killed, "broker_ms": round(broker_s * 1000.0, 3)}
                ok, err = msg["ok"], msg.get("error")
                if q:
                    ok, err = False, {"type": "DiskQuota", "message": q}
                return self._finish(ok, msg.get("result"), err, msg.get("stdout") or "", cpu=used + bg + load_bill,
                                    job=job, transcript=transcript, wall_s=time.monotonic() - t0 - broker_s,
                                    meter=meter, proto_bytes=proto + _size(msg.get("result")) + len(msg.get("stdout") or ""),
                                    v4_lines=msg.get("v4_work_lines") if self._v4 else None)
            reason = {"eof": "process exited (crash, memory or CPU limit)", "overflow": "output too large",
                      "garbage": "protocol violation"}.get(op, f"protocol violation (unexpected {op!r} message)")
            return died(reason, c_start=c_start, extra=load_bill)

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                pass


def run_program(files: dict[str, str], target: str, args: Any = None, caller: Any = None,
                cfg: SandboxConfig | None = None, rpc=None, timeout: float | None = None) -> dict:
    """One-shot execution: start a worker, run `target` ("module:function" or "path.py:function"), close."""
    w = Worker(files, cfg, "program")
    try:
        w.start()
        return w.run({"kind": "call", "target": target, "args": args, "caller": caller}, rpc=rpc, timeout=timeout)
    finally:
        w.close()


def run_tests(files: dict[str, str], folder: str = "tests", cfg: SandboxConfig | None = None,
              timeout: float | None = None) -> dict:
    w = Worker(files, cfg, "tests")
    try:
        w.start()
        return w.run({"kind": "tests", "pattern": folder}, timeout=timeout or 30)
    finally:
        w.close()


class WorkerPool:
    """Long-lived service workers keyed by (world, deployment). Bounded: least-recently-used workers are closed
    when the pool is full (the next request restarts them). Every access also sweeps the idle workers: one that
    uses CPU while it serves no job is killed, and that work is billed with the next job of its key."""

    def __init__(self, max_workers: int = 32, cfg: SandboxConfig | None = None):
        self.max_workers = max_workers
        self.cfg = cfg or SandboxConfig()
        self.workers: dict[str, Worker] = {}
        self.last_used: dict[str, float] = {}
        self.starts = 0
        self.debt: dict[str, int] = {}
        self.swept: list[dict] = []

    def _bill_idle(self, k: str, w: Worker, why: str) -> None:
        used = w.idle_cpu_ns()
        if used > M.BACKGROUND_SLACK_NS:
            ck = w._kill_measured()
            if ck is not None and w._idle_cpu is not None:
                used = max(used, ck - w._idle_cpu)
            w._idle_cpu = None
            self.debt[k] = self.debt.get(k, 0) + used
            self.swept.append({"key": k, "background_ns": used})
            del self.swept[:-50]
            w.dead_reason = why

    def sweep(self, skip: str | None = None) -> None:
        for k, w in list(self.workers.items()):
            if k == skip or w.proc is None or w._idle_cpu is None:
                continue
            self._bill_idle(k, w, "the process kept working while idle" if w.alive()
                            else "the process worked after its result and exited")

    def get(self, key: str, files_fn: Callable[[], dict[str, str]]) -> Worker:
        self.sweep(skip=key)
        w = self.workers.get(key)
        if w is None or not w.alive():
            if w is not None:
                self._bill_idle(key, w, "the process worked after its result and exited")
                w.close()
            while len(self.workers) >= self.max_workers:
                old = min(self.last_used, key=self.last_used.get)
                self.drop(old)
            w = Worker(files_fn(), self.cfg, key).start()
            w.carry_ns = self.debt.pop(key, 0)
            self.workers[key] = w
            self.starts += 1
        self.last_used[key] = time.monotonic()
        return w

    def drop(self, key: str) -> None:
        w = self.workers.pop(key, None)
        self.last_used.pop(key, None)
        if w is not None:
            self._bill_idle(key, w, "the process kept working while idle")   # billed with the key's next job
            w.close()

    def close(self) -> None:
        for k in list(self.workers):
            self.drop(k)
