"""An `ExperienceEnv` hosted in its own process: the learner gets a participant handle, the evaluator a separate one
(Clarification 2, OO.4: independent outcome verification through a separate evaluator-side interface unreachable by
the learner).

In one process a learner can always reach the environment object it drives (``env.verify``, ``env.world.state``):
Python has no access control. A process boundary is the only isolation that holds against an adversarial learner,
so evaluation runs that need it host the world here:

    part, ev = host({"tier": 1, "name": "rapid:net_routing", "seed": 3})
    obs = part.observe(part.agents()[0])          # the participant protocol only
    part.execute(agent, action); part.advance(3600); part.reward(agent)
    ev.verify()                                   # evaluator side; the participant pipe refuses it
    ev.close()

The world lives in a spawned child process that holds two pipes. Requests on the participant pipe are limited to
`PARTICIPANT_OPS`; anything else (``verify``, ``digest``, attribute access, ...) is refused with an error and never
evaluated. The evaluator pipe serves `EVALUATOR_OPS`. The learner's process never holds the environment object, its
world, its verifier or the evaluator pipe (give it only `part`).

The participant pipe carries bounded JSON frames, never pickles (workstream O, wave 3): `Connection.recv` unpickles,
and a pickle runs code while it is loaded - a learner holding only `part` could send an object whose `__reduce__`
executes in the world process and reads the grader's hidden parameters and holdout (demonstrated: probe
v5.isolation.remote_participant_cannot_run_code_in_the_world). The evaluator pipe is trusted and keeps pickles.
Keep the evaluator handle out of the learner's process too: a learner sharing a process with `ev` can find it.

The evaluation entry point is `evaluate` (W.2, SP-16c, OO.4): three processes. The world - with its verifier, its
hidden configuration and its grader - lives in a spawned world process (`host`); the evaluated learner runs in a
second spawned process that holds nothing but the participant pipe (no environment object, no world, no evaluator
handle, no seed); the calling process is the evaluator and verifies after the learner returns:

    out = evaluate({"tier": 1, "name": "pe_rule_induction"}, "crucible.experience.remote:reference_learner",
                   {"steps": 200})
    out["verification"]      # the evaluator's verdict; out["learner_result"] is what the learner returned (JSON)

`make()` stays the in-process development path: a learner holding the object `make()` returns can read everything in
its process (Python has no access control), so `make()` is never an evaluation path. The world seed is drawn here
(`sealed_seed`, 62 random bits) unless the caller names one: a learner that can guess the seed of a deterministic
world can rebuild it in its own process and read the hidden configuration (`crucible.experience.isolation` measures
that attack; its positive control shows that it succeeds with a small seed). Processes of one uid are kept apart by
the kernel's ptrace rules (Yama scope >= 1: a sibling cannot read the world process's memory); stronger isolation
(another uid, seccomp) is the sandbox's job (`crucible.sandbox.isolation`).
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import time
import traceback
from typing import Any

PARTICIPANT_OPS = ("agents", "observe", "execute", "observe_since", "tools", "advance", "reward", "reward_mode",
                   "now", "done", "tick_s")
EVALUATOR_OPS = ("verify", "digest", "describe", "close", "close_participant")


MAX_FRAME = 32 * 2**20        # bytes of one participant request or reply


class RemoteError(RuntimeError):
    pass


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f"{type(x).__name__} is not JSON")


def _send_json(conn, obj) -> None:
    data = json.dumps(obj, default=_jsonable, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_FRAME:
        data = json.dumps(["error", f"reply larger than {MAX_FRAME} bytes"]).encode()
    conn.send_bytes(data)


def _recv_json(conn):
    """A participant frame: bytes parsed as JSON (no pickle, nothing executes while it is read)."""
    return json.loads(conn.recv_bytes(MAX_FRAME))


def _request(msg):
    """(op, args, kwargs) of a participant frame, or None when it is not of that shape."""
    if not (isinstance(msg, list) and len(msg) == 3 and isinstance(msg[0], str) and isinstance(msg[1], list)
            and isinstance(msg[2], dict) and all(isinstance(k, str) for k in msg[2])):
        return None
    return msg[0], tuple(msg[1]), msg[2]


def _serve(ref: dict, part, ev) -> None:
    from multiprocessing.connection import wait
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .registry import make
    try:
        env = make(int(ref["tier"]), str(ref["name"]), int(ref.get("seed", 0)), ref.get("reward_mode"),
                   **dict(ref.get("kw") or {}))
        _send_json(part, ["ready", None])
    except Exception as e:  # noqa: BLE001 - reported to the parent, then the child exits
        _send_json(part, ["error", f"{type(e).__name__}: {e}"])
        return
    conns = [part, ev]

    def reply(c, status, out):
        if c is part:
            try:
                _send_json(c, [status, out])
            except (TypeError, ValueError) as e:
                _send_json(c, ["error", f"the reply is not JSON: {e}"])
        else:
            c.send((status, out))
    while conns:
        for c in wait(conns):
            try:
                if c is part:
                    req = _request(_recv_json(c))
                    if req is None:
                        reply(c, "error", "a request is a JSON list [op, args, kwargs]")
                        continue
                    op, args, kw = req
                else:
                    op, args, kw = c.recv()
            except EOFError:
                conns.remove(c)
                if c is ev:                 # the evaluator went away: the world ends
                    conns = []
                continue
            except (ValueError, OSError) as e:      # not JSON, too large: refused, never evaluated
                reply(c, "error", f"malformed request: {type(e).__name__}")
                continue
            allowed = PARTICIPANT_OPS if c is part else EVALUATOR_OPS
            if op not in allowed:
                reply(c, "error", f"{op!r} is not available on this side (allowed: {list(allowed)})")
                continue
            try:
                if op == "close":
                    env.close()
                    reply(c, "ok", None)
                    conns = []
                    break
                if op == "close_participant":
                    # the evaluation is over: no participant request is served any more (a process the learner
                    # left behind with a copy of its pipe cannot act while the evaluator verifies)
                    if part in conns:
                        conns.remove(part)
                    part.close()
                    reply(c, "ok", None)
                    break
                attr = getattr(env, op)
                out = attr(*args, **kw) if callable(attr) else attr
                reply(c, "ok", out)
            except Exception as e:  # noqa: BLE001 - an environment error goes back to the caller
                reply(c, "error", f"{type(e).__name__}: {e}" + ("" if c is part else
                                                                  f"\n{traceback.format_exc(limit=3)}"))


class _Side:
    _json = False

    def __init__(self, conn, ops: tuple):
        self._conn = conn
        self._ops = ops

    def _exchange(self, op: str, args, kw):
        if self._json:
            _send_json(self._conn, [op, list(args), dict(kw)])
            return tuple(_recv_json(self._conn))
        self._conn.send((op, args, kw))
        return self._conn.recv()

    def _call(self, op: str, *args, **kw) -> Any:
        if op not in self._ops:
            raise RemoteError(f"{op!r} is not available on this side")
        status, out = self._exchange(op, args, kw)
        if status != "ok":
            raise RemoteError(out)
        return out


class RemoteParticipant(_Side):
    """The learner's handle: the participant protocol of `ExperienceEnv` and nothing else (JSON frames)."""
    _json = True

    def agents(self) -> list[str]:
        return self._call("agents")

    def observe(self, agent: str) -> dict:
        return self._call("observe", agent)

    def execute(self, agent: str, action: dict) -> dict:
        return self._call("execute", agent, action)

    def observe_since(self, agent: str, cursor: int):
        evs, cur = self._call("observe_since", agent, cursor)
        return evs, cur

    def tools(self, agent: str) -> list[dict]:
        return self._call("tools", agent)

    def advance(self, seconds: int) -> None:
        self._call("advance", int(seconds))

    def reward(self, agent: str) -> float:
        return float(self._call("reward", agent))

    def _attr(self, op: str):
        """An attribute of the hosted environment; AttributeError when the environment has none (so `getattr(part,
        name, default)` behaves as on a local environment)."""
        status, out = self._exchange(op, (), {})
        if status != "ok":
            if isinstance(out, str) and out.startswith("AttributeError"):
                raise AttributeError(out)
            raise RemoteError(out)
        return out

    @property
    def reward_mode(self) -> str:
        return self._attr("reward_mode")

    @property
    def done(self) -> bool:
        return bool(self._attr("done"))

    @property
    def tick_s(self) -> int:
        return int(self._attr("tick_s"))

    @property
    def now(self) -> int:
        return int(self._attr("now"))

    def _raw(self, op: str, *args, **kw):
        """Send any request on the participant pipe (for tests that try what a participant may not do)."""
        return self._exchange(op, args, kw)


class RemoteEvaluator(_Side):
    """The evaluator's handle: verification and the operator-side identity of the run."""

    def __init__(self, conn, ops, proc):
        super().__init__(conn, ops)
        self._proc = proc

    def verify(self) -> dict:
        return self._call("verify")

    def digest(self) -> str:
        return self._call("digest")

    def describe(self) -> dict:
        return self._call("describe")

    def close_participant(self) -> None:
        """Stop serving the participant pipe (the end of a learner's evaluation, before verification)."""
        self._call("close_participant")

    def close(self) -> None:
        try:
            self._call("close")
        except (RemoteError, EOFError, OSError):
            pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.terminate()


def host(ref: dict, timeout: float = 300.0) -> tuple[RemoteParticipant, RemoteEvaluator]:
    """Start a child process that builds ``make(ref["tier"], ref["name"], ref.get("seed"), ref.get("reward_mode"),
    **ref.get("kw", {}))``; return (participant handle, evaluator handle)."""
    ctx = mp.get_context("spawn")
    p_parent, p_child = ctx.Pipe()
    e_parent, e_child = ctx.Pipe()
    proc = ctx.Process(target=_serve, args=(dict(ref), p_child, e_child), daemon=True)
    proc.start()
    p_child.close()
    e_child.close()
    if not p_parent.poll(timeout):
        proc.terminate()
        raise RemoteError("the environment process did not start in time")
    status, msg = _recv_json(p_parent)
    if status != "ready":
        proc.join(timeout=10)
        raise RemoteError(f"the environment process failed: {msg}")
    return RemoteParticipant(p_parent, PARTICIPANT_OPS), RemoteEvaluator(e_parent, EVALUATOR_OPS, proc)


# ====================================================================== the evaluation entry point (three processes)
LEARNER_RESULT_MAX = 8 * 2**20        # bytes of the learner's returned summary (JSON)


def sealed_seed() -> int:
    """A world seed no learner can guess (62 random bits from the operating system)."""
    import secrets
    return secrets.randbits(62)


def _target(spec: str):
    """The callable named by ``"package.module:function"`` or ``"path/to/file.py:function"``."""
    mod, _, fn = str(spec).rpartition(":")
    if not mod or not fn:
        raise ValueError(f"a learner is named 'module:function', got {spec!r}")
    if mod.endswith(".py"):
        from importlib import util
        name = "crucible_evaluated_" + os.path.basename(mod)[:-3]
        s = util.spec_from_file_location(name, mod)
        if s is None or s.loader is None:
            raise ImportError(f"cannot load {mod}")
        import sys
        m = util.module_from_spec(s)
        sys.modules[name] = m                     # dataclasses look their module up while the class is built
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


def _learn(target: str, kw: dict, conn, out) -> None:
    """The learner process: its only handle on the world is the participant pipe `conn`; it reports to the evaluator
    through `out` (JSON frames)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        part = RemoteParticipant(conn, PARTICIPANT_OPS)
        res = _target(target)(part, **dict(kw or {}))
        data = json.dumps(["ok", res], default=_jsonable, separators=(",", ":"), allow_nan=False).encode()
    except BaseException as e:  # noqa: BLE001 - every failure of the learner is reported, never raised in the world
        data = json.dumps(["error", f"{type(e).__name__}: {e}",
                           traceback.format_exc(limit=6)[-4000:]]).encode()
    if len(data) > LEARNER_RESULT_MAX:
        data = json.dumps(["error", f"the learner's result is larger than {LEARNER_RESULT_MAX} bytes"]).encode()
    try:
        out.send_bytes(data)
    finally:
        out.close()
        conn.close()


def evaluate(ref: dict, learner: str, learner_kw: dict | None = None, *, timeout: float = 3600.0,
             start_timeout: float = 900.0, seed_draws: int = 3) -> dict:
    """Evaluate a learner out of process. `ref` names the world as `host` takes it ({"tier", "name", "seed",
    "reward_mode", "kw"}); without ``"seed"`` the world gets a `sealed_seed` that only this process knows. `learner`
    ("module:function" or "file.py:function") is called in its own spawned process as ``fn(participant,
    **learner_kw)`` with a `RemoteParticipant` and must return JSON data. This process keeps the evaluator handle:
    after the learner returns (or fails, or exceeds `timeout` seconds) it verifies, records the world's identity and
    closes the world. Returns the evaluator's record (the seed and the verdict included: evaluator side). A world that
    cannot be built with a drawn sealed seed (a generator failure of that seed) is redrawn up to `seed_draws` times;
    the failed draws are recorded."""
    ref = dict(ref)
    sealed = "seed" not in ref or ref.get("seed") is None
    t0 = time.perf_counter()
    failed: list[dict] = []
    while True:
        if sealed:
            ref["seed"] = sealed_seed()
        try:
            part, ev = host(ref, timeout=start_timeout)
            break
        except RemoteError as e:
            if not sealed or len(failed) + 1 >= int(seed_draws) or "environment process failed" not in str(e):
                raise
            failed.append({"seed": ref["seed"], "error": str(e)[:300]})
    ctx = mp.get_context("spawn")
    r_parent, r_child = ctx.Pipe(duplex=False)
    lp = ctx.Process(target=_learn, args=(str(learner), dict(learner_kw or {}), part._conn, r_child), daemon=True)
    try:
        lp.start()
    except BaseException:
        ev.close()
        raise
    # the evaluator keeps no participant handle and no write end of the learner's report pipe
    part._conn.close()
    r_child.close()
    status, result, error = "timeout", None, None
    if r_parent.poll(timeout):
        try:
            msg = json.loads(r_parent.recv_bytes(LEARNER_RESULT_MAX + 1024))
            status = msg[0] if isinstance(msg, list) and msg and msg[0] in ("ok", "error") else "malformed"
            result = msg[1] if status == "ok" else None
            error = msg[1:] if status == "error" else None
        except (EOFError, OSError, ValueError) as e:
            status, error = "crashed", f"{type(e).__name__}: {e}"
    lp.join(timeout=30)
    if lp.is_alive():
        lp.terminate()
        lp.join(timeout=10)
    r_parent.close()
    try:
        ev.close_participant()
        verification = ev.verify()
        identity = ev.describe()
        digest = ev.digest()
    finally:
        ev.close()
    return {"interface": "crucible-experience-evaluation/1", "ref": ref, "seed_sealed": sealed,
            "seed_draws_failed": failed,
            "learner": str(learner), "learner_status": status, "learner_result": result, "learner_error": error,
            "learner_exitcode": lp.exitcode, "verification": verification, "describe": identity, "digest": digest,
            "processes": {"evaluator": os.getpid(), "learner": lp.pid, "world": ev._proc.pid},
            "wall_s": round(time.perf_counter() - t0, 3)}


def reference_learner(part, *, steps: int = 200, seed: int = 0, update_every: int = 32,
                      think_s: int | None = None) -> dict:
    """An evaluated learner for `evaluate`: the architecture-independent `ReferenceLearner` driven by `run_learner`
    over the participant handle (world changes are the evaluator's to count; the learner reports its own actions,
    updates and parameter hashes)."""
    from .loop import run_learner
    from .reference import ReferenceLearner
    L = ReferenceLearner(seed=int(seed), reward_mode=part.reward_mode)
    before = L.params_hash()
    s = run_learner(part, L, int(steps), update_every=int(update_every), think_s=think_s, count_changes=False)
    return {"steps": s["steps"], "actions": s["actions"], "ok": s["ok"], "rejected": s["rejected"],
            "updates": s["updates"], "params_changed_updates": s["params_changed_updates"],
            "nonzero_rewards": s["nonzero_rewards"], "schema_failures": s["schema_failures"], "kinds": s["kinds"],
            "params_hash_before": before, "params_hash_after": L.params_hash()}
