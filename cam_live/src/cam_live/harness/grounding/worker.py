"""Run a neural grounder in its own process (workstream EC).

``IsolatedGrounder`` implements ``contracts.VisualGrounder`` by talking to ``python -m
cam_live.harness.grounding.worker <name>`` over pipes. The child is started with an *empty* environment (only the
variables listed in ``_child_env``): no token, password, cookie path or socket address of the parent reaches the
process that holds the model, it is offline (``HF_HUB_OFFLINE=1``), and it never receives an ``ExecContext``. Its
answer is data: a point and a confidence, validated again by the parent (:func:`base.point_result`).

Frames: parent -> child ``uint32 json_len, uint32 png_len, json, png``; child -> parent one JSON line.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

from .base import GrounderUnavailable, GroundingError, hf_home, point_result


def _child_env() -> Dict[str, str]:
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8", "HF_HOME": hf_home(),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": src, "CAM_LIVE_GROUNDER_WORKER": "1",
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class IsolatedGrounder:
    """A grounder living in a child process with a clean environment. Started on first use."""

    def __init__(self, name: str, device: str = "auto", start_timeout_s: float = 300.0, call_timeout_s: float = 120.0,
                 **options: Any):
        from . import REGISTRY
        if name not in REGISTRY or REGISTRY[name].get("baseline"):
            raise GrounderUnavailable(f"{name!r} is not a model-backed grounder")
        self.name = name
        self.revision = REGISTRY[name]["revision"]
        self.heavy = bool(REGISTRY[name]["needs_gpu_lock"])
        self.noncommercial = bool(REGISTRY[name]["noncommercial"])
        self.baseline = False
        self._options = dict(options, device=device)
        self._start_timeout, self._call_timeout = start_timeout_s, call_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._info: Dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------ process
    def _readline(self, timeout: float) -> Dict[str, Any]:
        box: Dict[str, Any] = {}

        def read() -> None:
            try:
                box["line"] = self._proc.stdout.readline()
            except (OSError, ValueError) as e:
                box["err"] = e
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or not box.get("line"):
            self._kill()
            raise GrounderUnavailable(f"{self.name}: grounder process gave no answer within {timeout:g}s"
                                      if t.is_alive() else f"{self.name}: grounder process ended unexpectedly")
        try:
            return json.loads(box["line"])
        except ValueError as e:
            self._kill()
            raise GrounderUnavailable(f"{self.name}: unreadable answer from the grounder process") from e

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [os.path.realpath(sys.executable), "-m", "cam_live.harness.grounding.worker", self.name,
             json.dumps(self._options)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_child_env(), cwd="/",
            close_fds=True, start_new_session=True)
        ready = self._readline(self._start_timeout)
        if not ready.get("ok"):
            self._kill()
            raise GrounderUnavailable(f"{self.name}: {ready.get('error', 'grounder process failed to start')}")
        self._info = ready.get("info", {})

    def _kill(self) -> None:
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.kill()
                p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for s in (p.stdin, p.stdout):
                try:
                    s.close()
                except (OSError, ValueError):
                    pass

    def _call(self, header: Dict[str, Any], png: bytes = b"") -> Dict[str, Any]:
        with self._lock:
            self._ensure()
            body = json.dumps(header).encode()
            try:
                self._proc.stdin.write(struct.pack(">II", len(body), len(png)) + body + png)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise GrounderUnavailable(f"{self.name}: grounder process is gone") from e
            return self._readline(self._call_timeout)

    # ------------------------------------------------------------------------------------------ contract
    def ground(self, image_png: bytes, instruction: str) -> Dict[str, Any]:
        t0 = time.monotonic()
        ans = self._call({"op": "ground", "instruction": instruction}, bytes(image_png))
        if not ans.get("ok"):
            if ans.get("kind") == "unavailable":
                raise GrounderUnavailable(ans.get("error", "grounder unavailable"))
            raise GroundingError(ans.get("error", "grounding failed"))
        r = ans["result"]
        out = point_result(float(r["x"]), float(r["y"]), int(ans["width"]), int(ans["height"]),
                           float(r.get("confidence", 0.0)), self.name, self.revision, raw=str(r.get("raw", "")),
                           confidence_kind=str(r.get("confidence_kind", "")))
        out["latency_s"] = time.monotonic() - t0
        out["model_latency_s"] = float(r.get("latency_s", 0.0))
        return out

    def info(self) -> Dict[str, Any]:
        from . import REGISTRY
        if self._proc is not None and self._proc.poll() is None:
            ans = self._call({"op": "info"})
            if ans.get("ok"):
                self._info = ans["info"]
        base = dict(REGISTRY[self.name])
        base.update(self._info)
        base.update(isolated=True, pid=self._proc.pid if self._proc is not None else None)
        return base

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    body = json.dumps({"op": "exit"}).encode()
                    self._proc.stdin.write(struct.pack(">II", len(body), 0) + body)
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
            self._kill()


def _read_exact(stream: Any, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def main(argv: Any = None) -> int:
    """Child side: load one grounder, then answer requests until the pipe closes."""
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[0]
    options = json.loads(argv[1]) if len(argv) > 1 else {}
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                                      # libraries that print must not corrupt the protocol
    inp = sys.stdin.buffer

    def send(obj: Dict[str, Any]) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    from . import load_grounder
    try:
        grounder = load_grounder(name, isolated=False, allow_inprocess=True, **options)
        grounder.ensure_loaded()
    except Exception as e:  # noqa: BLE001 - reported to the parent, which decides
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    send({"ok": True, "info": grounder.info()})
    while True:
        try:
            jl, pl = struct.unpack(">II", _read_exact(inp, 8))
            if jl > 1 << 20 or pl > 64 << 20:
                return 2
            header = json.loads(_read_exact(inp, jl))
            png = _read_exact(inp, pl)
        except (EOFError, ValueError, struct.error):
            return 0
        op = header.get("op")
        if op == "exit":
            grounder.close()
            return 0
        if op == "info":
            send({"ok": True, "info": grounder.info()})
            continue
        try:
            from .base import decode_png
            img = decode_png(png)
            res = grounder.ground(png, str(header.get("instruction", "")))
            send({"ok": True, "result": res, "width": img.width, "height": img.height})
        except GrounderUnavailable as e:
            send({"ok": False, "kind": "unavailable", "error": str(e)})
        except Exception as e:  # noqa: BLE001 - one failed request must not end the process
            send({"ok": False, "kind": "grounding", "error": f"{type(e).__name__}: {e}"[:500]})


if __name__ == "__main__":
    sys.exit(main())
