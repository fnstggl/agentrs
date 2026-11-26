"""Hosted learners: the instrument's episode driver for learners whose code is not trusted (MASTER_SPEC Parts 15.2,
15.4, 16; SP-15.8, DEF-O-10).

`evaluator.run_remote_episode` drives a learner object in the harness process, which also holds the evaluator
channel and is the evaluator's parent: an adversarial learner there finds the evaluator handle through the garbage
collector, reads its verified score and partition, the evaluator's environment and memory (measured by the
security workstream's control `evaluator_internals`). That driver is for the instrument's own scripted controls.
A learner given as code to evaluate is *hosted* instead:

    with HostedLearner("crucible.research.v5.learners:make", {"arch": "weights", "seed": 0}) as h:
        t = ev.open_episode(entry, seed, budget=k)
        rec = run_hosted_episode(ev, t, h, learn=True, seed=7)     # the evaluator's verified result

The learner is constructed and runs only in a child process under the sandbox jail (`fresh._jailed`: its own
unprivileged uid, Landlock with read-only system libraries, standard library, numpy and the crucible source, one
private read-write directory, no sockets, no new processes). The child holds no channel to the evaluator: it asks
the harness for participant operations over its stdin/stdout control pipe, and the harness relays only the
participant operations of the one ticket it opened (`RELAYED`) to the evaluator's participant channel; the ticket
never reaches the child. The harness parses only bounded JSON lines from the child and never deserialises anything
else; what the learner reports is diagnostics, outcomes come from the evaluator. Learner methods the harness may
call (`call`) are limited to `CALLS`, whose results are JSON data.

Limits: process isolation on a shared kernel; the jail needs root, Landlock and seccomp (elsewhere `HostedLearner`
raises `isolation.JailUnavailable` - fail closed); no GPU learners.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

RELAYED = ("agents", "observe", "tools", "execute", "advance", "observe_since")
CALLS = ("parameters_digest", "state_summary", "report", "arch")
MAX_LINE = 4 * 2**20

_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.append(sys.argv[2]); "
         "from crucible.research.v5.hosted import _child; _child()")


# ====================================================================== the child (inside the jail)
class _RelayEnv:
    """The learner's environment inside the jail: participant operations relayed by the harness."""
    reward_mode = "observation_only"

    def __init__(self, rf, wf):
        self._rf, self._wf = rf, wf
        self.changes = 0

    def _v(self, op, *a):
        self._wf.write(json.dumps({"p": op, "a": list(a)}) + "\n")
        self._wf.flush()
        r = json.loads(self._rf.readline())
        if not r.get("ok"):
            from .evaluator import EvaluatorError
            raise EvaluatorError(r.get("error"))
        return r["value"]

    def agents(self):
        return self._v("agents")

    def observe(self, agent):
        return self._v("observe", agent)

    def tools(self, agent):
        return self._v("tools", agent)

    def execute(self, agent, action):
        return self._v("execute", agent, action)

    def advance(self, seconds):
        return self._v("advance", int(seconds))

    def observe_since(self, agent, cursor):
        return tuple(self._v("observe_since", agent, int(cursor)))

    def reward(self, agent):
        return 0.0


def _child() -> None:
    import hashlib
    import importlib

    import numpy as np
    rf, wf = sys.stdin, sys.stdout
    sys.stdout = sys.stderr                      # stray prints of the learner must not corrupt the control pipe

    def reply(obj):
        wf.write(json.dumps(obj, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
        wf.flush()
    spec = json.loads(rf.readline())
    mod, fn = spec["entry"].split(":")
    kw = dict(spec.get("kw") or {})
    learner = getattr(importlib.import_module(mod), fn)(**kw)
    reply({"ready": True, "pid": os.getpid(), "uid": os.getuid()})
    from .episodes import run_episode
    while True:
        line = rf.readline()
        if not line:
            return
        req = json.loads(line)
        cmd = req.get("cmd")
        try:
            if cmd == "episode":
                rec = run_episode(_RelayEnv(rf, wf), learner, learn=bool(req["learn"]),
                                  rng=np.random.default_rng(int(req["seed"])), budget=req.get("budget"),
                                  verify=False)
                reply({"done": {k: rec[k] for k in ("ticks", "interactions", "refused_beyond_budget",
                                                    "observed_signal_sum", "n_levers")}})
            elif cmd == "call":
                m = req["method"]
                if m == "parameters_digest":
                    h = hashlib.sha256()
                    for k, v in sorted(learner.parameters().items()):
                        h.update(k.encode())
                        h.update(np.ascontiguousarray(v).tobytes())
                    out = h.hexdigest()
                elif m == "arch":
                    out = getattr(learner, "arch", None)
                else:
                    out = getattr(learner, m)()
                reply({"value": out})
            elif cmd == "close":
                reply({"closed": True})
                return
            else:
                reply({"error": f"unknown command {cmd!r}"})
        except Exception as e:  # noqa: BLE001 - the learner's failure is reported, never raised in the harness
            reply({"error": f"{type(e).__name__}: {str(e)[:300]}"})


# ====================================================================== the harness side
class HostedError(RuntimeError):
    pass


class HostedLearner:
    """A learner constructed by ``entry`` ("module:function", called with ``kw``) inside a jailed child process."""

    def __init__(self, entry: str, kw: dict | None = None, *, timeout: float = 600.0):
        import numpy
        from .evaluator import isolated_env, src_root
        from .fresh import _jailed, _source_for_jail, _tmp_root
        if entry.count(":") != 1:
            raise ValueError("entry is 'module:function'")
        self.timeout = float(timeout)
        root = _tmp_root()
        self.base = tempfile.mkdtemp(prefix="crucible-v5-host-", dir=root)
        self.src_dir = tempfile.mkdtemp(prefix="crucible-v5-host-src-", dir=root)
        os.chmod(self.src_dir, 0o755)
        self.cfg_path = None
        try:
            home, tmp = os.path.join(self.base, "home"), os.path.join(self.base, "tmp")
            os.makedirs(home)
            os.makedirs(tmp)
            env = isolated_env(home)
            shutil.rmtree(os.path.dirname(env["TMPDIR"]), ignore_errors=True)
            env.update(TMPDIR=tmp, CRUCIBLE_CACHE=os.path.join(tmp, "cache"))
            src = _source_for_jail(self.src_dir)
            site = os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))
            cmd, self.cfg_path = _jailed([sys.executable, "-I", "-c", _BOOT, src, site], self.base, src)
            self._err_path = os.path.join(root, os.path.basename(self.base) + ".stderr")
            self._err = open(self._err_path, "wb")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                         env=env, cwd=home)
            self._send({"entry": entry, "kw": dict(kw or {})})
            hello = self._read()
            if not hello.get("ready"):
                raise HostedError(f"the hosted learner did not start: {hello}")
            self.child_pid, self.child_uid = int(hello["pid"]), int(hello["uid"])
        except BaseException:
            self.close()
            raise

    # -- control pipe
    def _send(self, obj) -> None:
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def _read(self) -> dict:
        deadline = time.monotonic() + self.timeout
        buf = b""
        fd = self.proc.stdout.fileno()
        while not buf.endswith(b"\n"):
            left = deadline - time.monotonic()
            if left <= 0:
                raise HostedError("the hosted learner timed out")
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise HostedError(f"the hosted learner exited: {self.stderr_tail()}")
            buf += chunk
            if len(buf) > MAX_LINE:
                raise HostedError("the hosted learner sent an oversized control line")
        lines = buf.split(b"\n")
        if len(lines) > 2:                       # one request at a time: more than one line is a protocol error
            raise HostedError("the hosted learner sent more than one control line")
        obj = json.loads(lines[0].decode())
        if not isinstance(obj, dict):
            raise HostedError("a control line is a JSON object")
        return obj

    def stderr_tail(self) -> str:
        try:
            with open(self._err_path, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 2000))
                return fh.read().decode(errors="replace")
        except (OSError, AttributeError):
            return ""

    # -- operations
    def episode(self, ev, ticket: str, *, learn: bool, seed: int, budget: int | None = None) -> dict:
        """Run one episode of the hosted learner on the evaluator-held `ticket`, relaying only its participant
        operations. Returns the learner-side record (diagnostics)."""
        self._send({"cmd": "episode", "learn": bool(learn), "seed": int(seed), "budget": budget})
        relayed = 0
        while True:
            msg = self._read()
            if "p" in msg:
                op, args = msg.get("p"), msg.get("a")
                if op not in RELAYED or not isinstance(args, list):
                    r = {"ok": False, "error": f"{op!r} is not a participant operation"}
                else:
                    r = ev.part.call(op, ticket, *args)
                    relayed += 1
                self._send(r)
                continue
            if "done" in msg:
                return dict(msg["done"], relayed_operations=relayed)
            raise HostedError(f"hosted episode failed: {msg.get('error', msg)}")

    def call(self, method: str) -> Any:
        if method not in CALLS:
            raise HostedError(f"{method!r} is not a hosted call")
        self._send({"cmd": "call", "method": method})
        msg = self._read()
        if "value" not in msg:
            raise HostedError(f"hosted call failed: {msg.get('error', msg)}")
        return msg["value"]

    def close(self) -> None:
        p = getattr(self, "proc", None)
        if p is not None and p.poll() is None:
            try:
                self._send({"cmd": "close"})
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()
                p.wait(timeout=10)
        shutil.rmtree(getattr(self, "base", ""), ignore_errors=True)
        shutil.rmtree(getattr(self, "src_dir", ""), ignore_errors=True)
        err = getattr(self, "_err", None)
        if err is not None:
            err.close()
            try:
                os.unlink(self._err_path)
            except OSError:
                pass
        if self.cfg_path:
            try:
                os.unlink(self.cfg_path)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_hosted_episode(ev, ticket: str, host: HostedLearner, *, learn: bool = False, seed: int = 0,
                       budget: int | None = None) -> dict:
    """The evaluator's verified record of one hosted episode (the learner never holds the evaluator channel)."""
    host.episode(ev, ticket, learn=learn, seed=seed, budget=budget)
    return ev.result(ticket)
