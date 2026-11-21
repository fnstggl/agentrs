"""Isolated execution of untrusted (agent-written) programs.

Layers (all verified by tests/test_sandbox.py):

1. separate process in its own session, started with an empty environment;
2. rlimits: address space, CPU seconds, processes (per uid), open files, file size, no core dumps;
3. privileges dropped to an unprivileged uid (default `nobody`) before any user code runs;
4. Landlock: read-only access to the Python standard library, approved packages and system libraries; read-write
   access only to the job's own working directory; all TCP bind/connect denied (ABI >= 4);
5. seccomp: no socket creation, no exec, no ptrace/mount/namespace/BPF/io_uring/keyring syscalls;
6. wall-clock timeouts per job, enforced by the parent (the whole process group is killed);
7. the only channel to the outside is a JSON-lines pipe to the parent; calls into the synthetic internet go through
   `crucible_sdk.call(tool, **args)`, which the kernel authorises and records like any other action.

If Landlock or seccomp is unavailable the runner refuses to start unless `allow_weak=True` is passed explicitly.
"""
from __future__ import annotations

import ctypes
import ctypes.util
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

BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boot.py")
APPROVED_PACKAGES = ("numpy",)
_lock = threading.Lock()
_counter = 0


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
    count_lines: bool = False   # deterministic work accounting (executed Python lines) instead of CPU timing


class SandboxUnavailable(RuntimeError):
    pass


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


def _safe_rel(path: str) -> str:
    p = os.path.normpath(path).lstrip("/")
    if p.startswith("..") or os.path.isabs(p) or "\x00" in p or p in ("", "."):
        raise ValueError(f"unsafe path {path!r}")
    return p


class Worker:
    """A sandboxed Python process serving jobs over a pipe. Used once (programs) or kept alive (services)."""

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

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Worker":
        global _counter
        cfg = self.cfg
        caps = capabilities()
        if not cfg.allow_weak and (caps["landlock_abi"] < 1 or not caps["seccomp"]):
            raise SandboxUnavailable(f"Landlock/seccomp required for untrusted code (found {caps})")
        if len(self.files) > cfg.max_files:
            raise ValueError("too many files")
        pw = pwd.getpwnam(cfg.user)
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
                os.chown(os.path.join(root, x), pw.pw_uid, pw.pw_gid)
            if root != self.workdir:
                os.chown(root, pw.pw_uid, pw.pw_gid)
        os.chown(self.workdir, pw.pw_uid, pw.pw_gid)
        reads, syspath = _read_paths(cfg.packages)
        pkgs = _pkg_dir(cfg)
        reads.append(pkgs)
        boot_cfg = {"uid": pw.pw_uid, "gid": pw.pw_gid, "read_paths": reads + [os.path.dirname(BOOT)],
                    "sys_path": syspath + [pkgs], "out_bytes": cfg.out_bytes, "allow_root": cfg.allow_root,
                    "require_landlock": not cfg.allow_weak, "require_seccomp": not cfg.allow_weak,
                    "count_lines": cfg.count_lines}
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
        from .backends import require, wrap
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
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    try:
                        self.proc.stdin.write(b'{"op":"exit"}\n')
                        self.proc.stdin.flush()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        self.proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                if self.proc.poll() is None:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
            except (ProcessLookupError, OSError):
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
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {"op": "garbage"}
        finally:
            sel.close()

    def _write(self, msg: dict) -> None:
        self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()

    def _stderr_tail(self) -> str:
        try:
            if self.proc and self.proc.poll() is not None:
                return self.proc.stderr.read()[-2000:].decode(errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return ""

    def run(self, job: dict, rpc: Callable[[str, dict], tuple[bool, Any]] | None = None,
            timeout: float | None = None, max_rpc: int = 50) -> dict:
        """Run one job. `rpc(tool, args) -> (ok, result_or_error)` serves nested world calls."""
        if not self.alive():
            return {"ok": False, "error": {"type": "WorkerDead", "message": self.dead_reason or "not running"},
                    "result": None, "stdout": "", "cpu_ms": 0.0, "wall_ms": 0.0, "rpc_calls": 0}
        self.jobs += 1
        jid = self.jobs
        timeout = timeout or self.cfg.job_wall_s
        t0 = time.monotonic()
        deadline = t0 + timeout
        n_rpc = 0
        self._write({"op": "job", "id": jid, **job})
        while True:
            msg = self._read_msg(deadline)
            if msg is None:
                self.dead_reason = f"timeout after {timeout}s"
                self._kill()
                return {"ok": False, "error": {"type": "Timeout", "message": f"exceeded {timeout}s"}, "result": None,
                        "stdout": "", "cpu_ms": timeout * 1000.0, "wall_ms": timeout * 1000.0, "rpc_calls": n_rpc}
            op = msg.get("op")
            if op == "rpc":
                n_rpc += 1
                if rpc is None or n_rpc > max_rpc:
                    self._write({"op": "rpc_result", "id": msg["id"], "ok": False,
                                 "error": {"code": "forbidden", "message": "no network calls available"}})
                    continue
                pause = time.monotonic()
                ok, val = rpc(msg.get("tool", ""), msg.get("args") or {})
                deadline += time.monotonic() - pause  # time spent in the broker is not the program's
                self._write({"op": "rpc_result", "id": msg["id"], "ok": ok, **({"result": val} if ok else {"error": val})})
            elif op == "done" and msg.get("id") == jid:
                msg["rpc_calls"] = n_rpc
                return msg
            else:
                reason = {"eof": "process exited (crash, memory or CPU limit)", "overflow": "output too large",
                          "garbage": "protocol violation"}.get(op, f"unexpected message {op}")
                self.dead_reason = reason
                tail = self._stderr_tail()
                self._kill()
                return {"ok": False, "error": {"type": "WorkerDied", "message": reason, "stderr": tail[-800:]},
                        "result": None, "stdout": "", "cpu_ms": (time.monotonic() - t0) * 1000.0,
                        "wall_ms": (time.monotonic() - t0) * 1000.0, "rpc_calls": n_rpc}

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
    when the pool is full (the next request restarts them)."""

    def __init__(self, max_workers: int = 32, cfg: SandboxConfig | None = None):
        self.max_workers = max_workers
        self.cfg = cfg or SandboxConfig()
        self.workers: dict[str, Worker] = {}
        self.last_used: dict[str, float] = {}
        self.starts = 0

    def get(self, key: str, files_fn: Callable[[], dict[str, str]]) -> Worker:
        w = self.workers.get(key)
        if w is None or not w.alive():
            if w is not None:
                w.close()
            while len(self.workers) >= self.max_workers:
                old = min(self.last_used, key=self.last_used.get)
                self.drop(old)
            w = Worker(files_fn(), self.cfg, key).start()
            self.workers[key] = w
            self.starts += 1
        self.last_used[key] = time.monotonic()
        return w

    def drop(self, key: str) -> None:
        w = self.workers.pop(key, None)
        self.last_used.pop(key, None)
        if w is not None:
            w.close()

    def close(self) -> None:
        for k in list(self.workers):
            self.drop(k)
