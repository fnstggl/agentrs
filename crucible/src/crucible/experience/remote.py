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
"""
from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from typing import Any

PARTICIPANT_OPS = ("agents", "observe", "execute", "observe_since", "tools", "advance", "reward", "reward_mode",
                   "now", "done", "tick_s")
EVALUATOR_OPS = ("verify", "digest", "describe", "close")


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

    @property
    def reward_mode(self) -> str:
        return self._call("reward_mode")

    @property
    def done(self) -> bool:
        return bool(self._call("done"))

    @property
    def tick_s(self) -> int:
        return int(self._call("tick_s"))

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
