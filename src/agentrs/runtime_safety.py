"""Crash prevention and recovery helpers shared by AGENTRS trainers and experiment queues.

* ``main_guard_problems`` - static check that every module run with ``python -m`` defines everything before its
  ``if __name__ == "__main__":`` guard (the 2026-11-18 failure: ``main()`` ran before ``Trainer2.hybrid_final`` was
  attached at the bottom of the module, so every final phase crashed after training had finished).
* ``smoke_run`` - run the *exact* production command with budget-only overrides, start to finish, in a scratch
  run directory, and check that it reached its final phase (used by queues before launching expensive jobs).
* ``crash_signature`` / ``CrashLedger`` - recognise the same failure repeating and stop restarting it, with a
  written crash report instead of a crash loop.
* ``disk_free_gb`` - the queue does not start jobs when the disk is nearly full (a full disk killed every run).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


def main_guard_problems(root: Path) -> list[str]:
    """Modules whose ``__main__`` guard is followed by further top-level statements (or that have several)."""
    out = []
    for f in sorted(Path(root).rglob("*.py")):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError as e:
            out.append(f"{f}: syntax error {e}")
            continue
        guards = [i for i, n in enumerate(tree.body) if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
                  and "__main__" in ast.unparse(n.test)]
        if not guards:
            continue
        if len(guards) > 1:
            out.append(f"{f}: {len(guards)} __main__ guards")
        if guards[-1] != len(tree.body) - 1:
            after = [type(n).__name__ for n in tree.body[guards[-1] + 1:]]
            out.append(f"{f}: statements after the __main__ guard: {after}")
    return out


def disk_free_gb(path: str = "/workspace") -> float:
    return shutil.disk_usage(path).free / 2**30


_TB_LAST = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt|Exit)\b.*)$")
_TB_FILE = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')


def crash_signature(log_text: str) -> Optional[dict]:
    """The last Python traceback in a log: exception line + innermost frame, and a stable hash of both."""
    idx = log_text.rfind("Traceback (most recent call last):")
    if idx < 0:
        m = re.search(r"(CUDA out of memory|OutOfMemoryError|No space left on device|Killed)", log_text[-20000:])
        if not m:
            return None
        sig = m.group(1)
        return {"exception": sig, "frame": None, "hash": hashlib.sha1(sig.encode()).hexdigest()[:12], "traceback": log_text[-3000:]}
    tb = log_text[idx:]
    lines = tb.splitlines()
    frames = _TB_FILE.findall(tb)
    exc = next((ln for ln in lines[1:] if _TB_LAST.match(ln.strip())), lines[-1] if lines else "")
    frame = frames[-1] if frames else None
    key = exc.strip() + "|" + (f"{Path(frame[0]).name}:{frame[1]}:{frame[2]}" if frame else "")
    key = re.sub(r"0x[0-9a-f]+|\d{4,}", "#", key)  # addresses, pids, sizes do not make a new failure
    return {"exception": exc.strip(), "frame": frame, "hash": hashlib.sha1(key.encode()).hexdigest()[:12],
            "traceback": "\n".join(lines[-60:])}


START_MARKER = "=== queue start "


class CrashLedger:
    """Per-job crash history inside a queue state dict. ``record`` returns True when the job should be stopped:
    the same crash signature occurred ``repeat_limit`` times, or ``max_starts`` was reached. A report is written
    to ``<run_dir>/CRASH_REPORT.md``."""

    def __init__(self, repeat_limit: int = 2, max_starts: int = 4):
        self.repeat_limit, self.max_starts = repeat_limit, max_starts

    def record(self, job_state: dict, run_dir: Path, log_path: Path) -> bool:
        try:
            text = log_path.read_text(errors="ignore")[-400_000:]
        except OSError:
            text = ""
        cut = text.rfind(START_MARKER)  # only the output of the latest start counts
        if cut >= 0:
            text = text[cut:]
        sig = crash_signature(text) or {"exception": "exited without a traceback", "frame": None,
                                          "hash": "no-traceback", "traceback": text[-2000:]}
        hist = job_state.setdefault("crashes", [])
        hist.append({"time": time.time(), "hash": sig["hash"], "exception": sig["exception"][:300]})
        same = sum(1 for h in hist if h["hash"] == sig["hash"])
        stop = same >= self.repeat_limit or job_state.get("starts", 0) >= self.max_starts
        if stop:
            reason = (f"the same failure occurred {same} times" if same >= self.repeat_limit
                      else f"{job_state.get('starts')} starts")
            report = [f"# Crash report: {run_dir.name}", "", f"Stopped restarting: {reason}.", "",
                      f"Last exception: `{sig['exception']}`", "",
                      f"Innermost frame: {sig['frame']}", "", "## Crash history", ""]
            report += [f"- {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(h['time']))} UTC `{h['hash']}` {h['exception']}" for h in hist]
            report += ["", "## Last traceback", "", "```", sig["traceback"], "```", ""]
            try:
                (run_dir / "CRASH_REPORT.md").write_text("\n".join(report))
            except OSError:
                pass
        return stop


def config_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode() if not isinstance(p, (bytes, str)) else
                 (p.encode() if isinstance(p, str) else p))
    return h.hexdigest()[:16]


def smoke_run(cmd: list[str], cwd: str, env: dict, run_dir: Path, done_check, timeout_s: int = 1800,
              log_path: Optional[Path] = None) -> dict:
    """Run ``cmd`` (the production command with budget-only overrides pointing at ``run_dir``) to completion.
    ``done_check(run_dir) -> (ok, detail)`` decides success after a zero exit code."""
    t0 = time.time()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log_path = log_path or (run_dir / "smoke.log")
    with open(log_path, "w") as lf:
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    ok, detail = (False, f"exit {code}") if code != 0 else done_check(run_dir)
    res = {"ok": bool(ok), "detail": detail, "exit": code, "seconds": round(time.time() - t0, 1), "cmd": cmd,
           "log": str(log_path)}
    if not ok:
        sig = crash_signature(Path(log_path).read_text(errors="ignore"))
        res["exception"] = sig and sig["exception"]
    return res


def state_phase_done(run_dir: Path):
    try:
        st = json.loads((run_dir / "state.json").read_text())
    except (OSError, ValueError) as e:
        return False, f"no readable state.json: {e}"
    if st.get("phase") != "done":
        return False, f"phase {st.get('phase')!r}"
    errs = st.get("diagnostic_errors") or {}
    return (not errs), ("done" if not errs else f"done with diagnostic errors: {sorted(errs)}")


def gpu_env(visible: bool) -> dict:
    env = dict(os.environ)
    if not visible:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env
