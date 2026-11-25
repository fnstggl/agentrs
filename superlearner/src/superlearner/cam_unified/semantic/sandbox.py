"""Local sandboxed code execution for code tasks outside OpenWorld (spec 9.3 / 15.4; architecture §2).

Backend "crucible" (default, required unless allow_weak=True) reuses Crucible's own sandbox runner
(`crucible.sandbox.runner.Worker`): a fresh process per execution with an empty environment, rlimits (address space,
CPU, processes, files, file size, no core), privileges dropped to uid `nobody`, Landlock (read-only stdlib/numpy,
read-write only in the job's own directory), seccomp (no sockets, no exec/fork, no ptrace/mount/namespaces) and a
wall-clock timeout that kills the whole process group. OpenWorld code keeps running inside OpenWorld (code.run /
cloud.deploy); this module is for local tasks (e.g. "write a program that ...", code<->execution objectives).

Backend "subprocess" is a WEAKER fallback (temp cwd, empty environment, rlimits, timeout, `python -I`, process group
kill) with no filesystem or network isolation; it is only used when explicitly requested with allow_weak=True - there
is never a silent fallback.

Both backends run a program through the same generated runner module (`__cam_runner__.py`), so they agree on every
outcome: a script (`main.py`) runs as `__main__` via `runpy.run_path` (the usual `if __name__ == "__main__":` guard
runs, sibling modules import); `sys.exit(0)` / `sys.exit(None)` / falling off the end is success, any other exit
status is a failure with error type `NonZeroExit` (a string exit message is printed to stderr, status 1, as CPython
does); an uncaught exception is a failure with its type, message and traceback. A function entry
(`file.py:function`) is called with the JSON input and returns its JSON result.

Results come back as `ExecutionResult` with EXEC_RESULT (stdout + JSON result for function entries) or ERROR
(exception type, message, traceback tail, stdout) observations. Executing never moves any simulated clock
(`duration` = 0); wall/CPU time are in `meta`.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..interfaces import Action, ExecutionResult, EvidenceKind, Observation, ObsKind, ToolSchema


RUNNER_FILE = "__cam_runner__.py"
EXIT_KEY = "__cam_exit__"
MARKER = "__CAM_RESULT__"
RUNNER_SOURCE = r'''"""CAM runner: executes main.py as __main__ or calls file.py:function (generated; not user code)."""
import json
import runpy
import sys
import traceback


def _exit_status(e):
    c = e.code
    if c is None:
        return 0
    if isinstance(c, int):
        return int(c)
    print(c, file=sys.stderr)                 # CPython prints a non-integer exit message and exits with status 1
    return 1


def run_script(_args=None):
    sys.argv = ["main.py"]
    try:
        runpy.run_path("main.py", run_name="__main__")
    except SystemExit as e:
        return {"__cam_exit__": _exit_status(e)}
    return {"__cam_exit__": 0}


def call_entry(target, args):
    mod, _, fn = target.partition(":")
    mod = mod[:-3] if mod.endswith(".py") else mod
    m = __import__(mod.replace("/", "."), fromlist=["_"])
    return getattr(m, fn)(args)


if __name__ == "__main__":                    # subprocess backend driver
    sys.path.insert(0, ".")
    try:
        if sys.argv[1] == "script":
            r = run_script()
        else:
            r = call_entry(sys.argv[2], json.loads(sys.stdin.read() or "null"))
        json.dumps(r)
        out = {"ok": True, "result": r}
    except BaseException as e:  # noqa: BLE001 - user code may raise anything
        out = {"ok": False, "error": {"type": type(e).__name__, "message": str(e)[:2000],
                                      "traceback": traceback.format_exc(limit=6)[-4000:]}}
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.write("\n__CAM_RESULT__" + json.dumps(out, default=str) + "\n")
    sys.stdout.flush()
'''


@dataclass
class SandboxLimits:
    timeout_s: float = 10.0
    mem_mb: int = 1536
    cpu_s: int = 60
    out_bytes: int = 65536
    max_files: int = 200
    max_file_bytes: int = 256 * 1024


class SandboxUnavailable(RuntimeError):
    pass


def crucible_sandbox_capabilities() -> dict:
    try:
        from crucible.sandbox.runner import capabilities
        caps = capabilities()
        caps["available"] = caps["landlock_abi"] >= 1 and bool(caps["seccomp"]) and caps["nobody"]
        return caps
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


class CodeSandbox:
    """Run untrusted Python: `run(code=...)` executes a script; `run(files=..., entry='mod.py:fn', args=...)` calls
    fn(args) and returns its (JSON) result."""

    def __init__(self, backend: str = "crucible", limits: Optional[SandboxLimits] = None, allow_weak: bool = False):
        self.limits = limits or SandboxLimits()
        if backend not in ("crucible", "subprocess"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "crucible":
            caps = crucible_sandbox_capabilities()
            if not caps.get("available"):
                raise SandboxUnavailable(f"Crucible sandbox unavailable here: {caps}")
        elif not allow_weak:
            raise SandboxUnavailable("the subprocess backend has no filesystem/network isolation; pass "
                                     "allow_weak=True to use it explicitly")
        self.backend = backend
        self.runs = 0

    # ------------------------------------------------------------------ backends
    def _crucible(self, files: dict, entry: Optional[str], args: Any) -> dict:
        from crucible.sandbox.runner import SandboxConfig, Worker
        L = self.limits
        cfg = SandboxConfig(job_wall_s=L.timeout_s, mem_mb=L.mem_mb, cpu_s=L.cpu_s, out_bytes=L.out_bytes,
                            max_files=L.max_files, max_file_bytes=L.max_file_bytes)
        w = Worker({**files, RUNNER_FILE: RUNNER_SOURCE}, cfg, "cam-local")
        try:
            w.start()
            if entry:
                job = {"kind": "call", "target": entry, "args": args, "two_args": False}
            else:
                job = {"kind": "call", "target": f"{RUNNER_FILE}:run_script", "args": None, "two_args": False}
            return w.run(job)
        finally:
            w.close()

    def _subprocess(self, files: dict, entry: Optional[str], args: Any) -> dict:
        L = self.limits
        work = tempfile.mkdtemp(prefix="cam-sbx-")
        try:
            for rel, content in files.items():
                p = os.path.normpath(rel)
                if p.startswith("..") or os.path.isabs(p):
                    raise ValueError(f"unsafe path {rel!r}")
                full = os.path.join(work, p)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w") as f:
                    f.write(content)
            with open(os.path.join(work, RUNNER_FILE), "w") as f:
                f.write(RUNNER_SOURCE)
            argv = [sys.executable, "-I", RUNNER_FILE] + (["entry", entry] if entry else ["script"])
            limits = (L.mem_mb, L.cpu_s)

            def pre():
                mem, cpu = limits
                resource.setrlimit(resource.RLIMIT_AS, (mem * 2**20, mem * 2**20))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

            env = {"PATH": "/usr/bin:/bin", "HOME": work, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
                   "OMP_NUM_THREADS": "1", "no_proxy": "*", "NO_PROXY": "*"}
            t0 = time.monotonic()
            proc = subprocess.Popen(argv, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, preexec_fn=pre, start_new_session=True)
            try:
                out, _ = proc.communicate(json.dumps(args).encode() if entry else b"", timeout=L.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                return {"ok": False, "error": {"type": "Timeout", "message": f"exceeded {L.timeout_s}s"},
                        "result": None, "stdout": "", "wall_ms": L.timeout_s * 1000.0, "cpu_ms": None}
            wall = (time.monotonic() - t0) * 1000.0
            text = out.decode(errors="replace")
            at = text.rfind("\n" + MARKER)
            body = (text if at < 0 else text[:at])[: L.out_bytes]
            if at >= 0:
                r = json.loads(text[at + 1 + len(MARKER):].strip())
                return {"ok": r["ok"], "result": r.get("result"), "error": r.get("error"), "stdout": body,
                        "wall_ms": wall, "cpu_ms": None}
            return {"ok": False, "result": None, "error": {"type": "Crashed", "message": f"exit {proc.returncode}"},
                    "stdout": body, "wall_ms": wall, "cpu_ms": None}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------------ public
    def run(self, code: Optional[str] = None, files: Optional[dict] = None, entry: Optional[str] = None,
            args: Any = None, t: float = 0.0, source: str = "sandbox", caused_by: Optional[str] = None,
            entity: Optional[str] = None) -> ExecutionResult:
        files = {str(k): v for k, v in (files or {}).items() if v is not None}
        if code is not None:
            files["main.py"] = code
        self.runs += 1
        try:
            if not files:
                raise ValueError("nothing to run: no files")
            if not entry and "main.py" not in files:
                raise ValueError("a script run needs main.py (or an entry 'file.py:function')")
            if entry and entry.partition(":")[0] not in files and entry.partition(":")[0] + ".py" not in files:
                raise ValueError(f"entry {entry!r} names a file that was not written")
            if RUNNER_FILE in files:
                raise ValueError(f"{RUNNER_FILE} is reserved")
            res = self._crucible(files, entry, args) if self.backend == "crucible" else \
                self._subprocess(files, entry, args)
            if not entry and res.get("ok"):                       # script: map the exit status
                status = (res.get("result") or {}).get(EXIT_KEY) if isinstance(res.get("result"), dict) else None
                if status is None:
                    res = {**res, "ok": False, "error": {"type": "Crashed", "message": "runner returned no status"}}
                elif status != 0:
                    res = {**res, "ok": False, "result": None,
                           "error": {"type": "NonZeroExit", "message": f"exit status {status}"}}
                else:
                    res = {**res, "result": None}
        except ValueError as e:     # nothing to run, unsafe path, file too large, too many files: a real outcome
            res = {"ok": False, "error": {"type": "InvalidProgram", "message": str(e)}, "result": None, "stdout": ""}
        stdout = res.get("stdout") or ""
        meta = {"backend": self.backend, "wall_ms": res.get("wall_ms"), "cpu_ms": res.get("cpu_ms"),
                "entry": entry, "files": sorted(files)}
        if res.get("ok"):
            body = f"stdout:\n{stdout}" if stdout else "stdout: (empty)"
            if entry or res.get("result") is not None:
                body += "\nresult: " + json.dumps(res.get("result"), default=str)
            obs = [Observation(ObsKind.EXEC_RESULT, body, t=t, source=source, entity=entity,
                               evidence=EvidenceKind.CONTROLLED, caused_by=caused_by,
                               meta={"ok": True, "result": res.get("result")})]
            return ExecutionResult(True, obs, None, 0.0, 0.0, t, t, meta)
        err = res.get("error") or {}
        msg = f"{err.get('type', 'Error')}: {err.get('message', '')}"
        body = msg + ("\n" + err["traceback"] if err.get("traceback") else "") + (f"\nstdout:\n{stdout}" if stdout else "")
        obs = [Observation(ObsKind.ERROR, body, t=t, source=source, entity=entity, evidence=EvidenceKind.CONTROLLED,
                           caused_by=caused_by, meta={"ok": False, "error_type": err.get("type")})]
        return ExecutionResult(False, obs, msg, 0.0, 0.0, t, t, meta)


class LocalCodeTools:
    """Tool schemas + executor for local code tasks: the action generator can call these like any other tool."""

    def __init__(self, sandbox: Optional[CodeSandbox] = None, max_code_chars: int = 20000):
        self.sandbox = sandbox or CodeSandbox()
        self.max_code_chars = max_code_chars

    def tools(self) -> list:
        return [
            ToolSchema("run_python", "Run a Python program in an isolated sandbox (standard library and numpy only, "
                                     "no network, no files outside its directory) and return stdout or the error.",
                       {"type": "object", "properties": {"code": {"type": "string", "minLength": 1,
                                                                  "maxLength": self.max_code_chars,
                                                                  "description": "complete Python source"}},
                        "required": ["code"], "additionalProperties": False}, returns="stdout or traceback"),
            ToolSchema("run_function", "Write Python files and call one function with a JSON input in the sandbox; "
                                       "returns the function's JSON result.",
                       {"type": "object", "properties": {
                           "files": {"type": "object", "additionalProperties": {"type": "string",
                                                                                "maxLength": self.max_code_chars},
                                     "description": "path -> source"},
                           "target": {"type": "string", "pattern": r"[A-Za-z_][A-Za-z0-9_/]*\.py:[A-Za-z_][A-Za-z0-9_]*",
                                      "description": "file.py:function"},
                           "input": {"description": "JSON input passed to the function"}},
                        "required": ["files", "target"], "additionalProperties": False}, returns="JSON result"),
        ]

    def execute(self, action: Action, t: float = 0.0) -> ExecutionResult:
        a = action.args or {}
        if action.tool == "run_python":
            return self.sandbox.run(code=a.get("code", action.code or ""), t=t, source="run_python",
                                    caused_by=action.action_id)
        if action.tool == "run_function":
            return self.sandbox.run(files=a.get("files") or {}, entry=a.get("target"), args=a.get("input"), t=t,
                                    source="run_function", caused_by=action.action_id)
        return ExecutionResult(False, [Observation(ObsKind.ERROR, f"unknown_tool: {action.tool}", t=t,
                                                   source="local_code_tools", evidence=EvidenceKind.CONTROLLED,
                                                   caused_by=action.action_id)], f"unknown tool {action.tool}")
