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
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any

PARTICIPANT_OPS = ("agents", "observe", "execute", "observe_since", "tools", "advance", "reward", "reward_mode",
                   "now", "done", "tick_s")
EVALUATOR_OPS = ("verify", "digest", "describe", "close")


class RemoteError(RuntimeError):
    pass


def _serve(ref: dict, part, ev) -> None:
    from multiprocessing.connection import wait
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .registry import make
    try:
        env = make(int(ref["tier"]), str(ref["name"]), int(ref.get("seed", 0)), ref.get("reward_mode"),
                   **dict(ref.get("kw") or {}))
        part.send(("ready", None))
    except Exception as e:  # noqa: BLE001 - reported to the parent, then the child exits
        part.send(("error", f"{type(e).__name__}: {e}"))
        return
    conns = [part, ev]
    while conns:
        for c in wait(conns):
            try:
                msg = c.recv()
            except EOFError:
                conns.remove(c)
                if c is ev:                 # the evaluator went away: the world ends
                    conns = []
                continue
            op, args, kw = msg
            allowed = PARTICIPANT_OPS if c is part else EVALUATOR_OPS
            if op not in allowed:
                c.send(("error", f"{op!r} is not available on this side (allowed: {list(allowed)})"))
                continue
            try:
                if op == "close":
                    env.close()
                    c.send(("ok", None))
                    conns = []
                    break
                attr = getattr(env, op)
                out = attr(*args, **kw) if callable(attr) else attr
                c.send(("ok", out))
            except Exception as e:  # noqa: BLE001 - an environment error goes back to the caller
                c.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


class _Side:
    def __init__(self, conn, ops: tuple):
        self._conn = conn
        self._ops = ops

    def _call(self, op: str, *args, **kw) -> Any:
        if op not in self._ops:
            raise RemoteError(f"{op!r} is not available on this side")
        self._conn.send((op, args, kw))
        status, out = self._conn.recv()
        if status != "ok":
            raise RemoteError(out)
        return out


class RemoteParticipant(_Side):
    """The learner's handle: the participant protocol of `ExperienceEnv` and nothing else."""

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
        self._conn.send((op, args, kw))
        return self._conn.recv()


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
    status, msg = p_parent.recv()
    if status != "ready":
        proc.join(timeout=10)
        raise RemoteError(f"the environment process failed: {msg}")
    return RemoteParticipant(p_parent, PARTICIPANT_OPS), RemoteEvaluator(e_parent, EVALUATOR_OPS, proc)
