"""cam_live.runtime.multi — several acting CAM instances sharing one central learner (spec 11.3).

Each :class:`Actor` is a complete :class:`~cam_live.runtime.loop.LiveRuntime` with its own objective, local
beliefs, experience history, authorization, environment and resource limits. What is shared is the learner:

    actor.agent.export_experience()  ──►  SharedLearner.submit(actor_id, records)        experience flows in
    SharedLearner.step()                  the central learner updates its weights (version += 1)
    SharedLearner.pull(have_version) ──►  actor.agent.set_weights(weights, version)      weights flow back

The same three calls work across processes through :func:`serve_learner` / :class:`RemoteSharedLearner`.
The first live experiment uses one actor. Several actors never mean several public identities: at most one actor
may hold a live task, and nothing here creates accounts or identities.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from ..contracts import Clock, LearningUpdateRecord
from ..harness.rpc import RpcClient, RpcServer
from .loop import RUNNING, LiveRuntime

LEARNER_METHODS = ("submit", "pull", "stats")


@runtime_checkable
class ActorAgent(Protocol):
    """What an actor's agent adds to ``contracts.AgentAPI`` to take part in shared learning."""

    def export_experience(self) -> List[Any]: ...                  # experience gathered since the last export
    def weights_version(self) -> int: ...
    def set_weights(self, weights: Any, version: int) -> None: ...


@runtime_checkable
class CentralLearner(Protocol):
    """The learner object the actors share."""

    def add_experience(self, actor_id: str, records: Sequence[Any]) -> int: ...
    def learn(self, budget_s: float) -> List[LearningUpdateRecord]: ...
    def weights(self) -> Tuple[Any, int]: ...                      # (weights, version)


class SharedLearner:
    """Thread-safe front of one central learner: accepts experience tagged by actor, runs learning, and hands out
    versioned weights."""

    def __init__(self, learner: CentralLearner, clock: Optional[Clock] = None, learn_budget_s: float = 10.0):
        self.learner, self.clock, self.learn_budget_s = learner, clock, learn_budget_s
        self._lock = threading.RLock()
        self.received: Dict[str, int] = {}
        self.pulled: Dict[int, int] = {}
        self.updates: List[LearningUpdateRecord] = []

    def submit(self, actor_id: str, records: Sequence[Any]) -> int:
        with self._lock:
            n = self.learner.add_experience(actor_id, list(records)) if records else 0
            self.received[actor_id] = self.received.get(actor_id, 0) + len(records)
            return n

    def step(self, budget_s: Optional[float] = None) -> List[LearningUpdateRecord]:
        with self._lock:
            ups = list(self.learner.learn(self.learn_budget_s if budget_s is None else budget_s) or [])
            self.updates.extend(ups)
            return ups

    def pull(self, have_version: int = -1) -> Optional[Tuple[Any, int]]:
        """Weights newer than ``have_version``, or None when the caller is up to date."""
        with self._lock:
            w, v = self.learner.weights()
            if v <= have_version:
                return None
            self.pulled[v] = self.pulled.get(v, 0) + 1
            return w, v

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"version": self.learner.weights()[1], "received": dict(self.received),
                    "updates": len(self.updates), "pulled": {str(k): v for k, v in self.pulled.items()}}


def serve_learner(shared: SharedLearner, socket_path: str, allowed_uids: Optional[set] = None) -> RpcServer:
    """Expose a SharedLearner to actors in other processes."""
    return RpcServer(socket_path, shared, LEARNER_METHODS, allowed_uids=allowed_uids, name="learner").start()


class RemoteSharedLearner:
    """Actor-side handle on a SharedLearner served elsewhere (same three calls; learning runs on the server)."""

    def __init__(self, socket_path: str, timeout: float = 600.0):
        self._rpc = RpcClient(socket_path, timeout)

    def submit(self, actor_id: str, records: Sequence[Any]) -> int:
        return int(self._rpc.call("submit", actor_id, list(records)))

    def pull(self, have_version: int = -1) -> Optional[Tuple[Any, int]]:
        got = self._rpc.call("pull", have_version)
        return None if got is None else (got[0], int(got[1]))

    def stats(self) -> Dict[str, Any]:
        return self._rpc.call("stats")

    def step(self, budget_s: Optional[float] = None) -> List[LearningUpdateRecord]:
        return []                                         # the central process decides when its learner runs

    def close(self) -> None:
        self._rpc.close()


@dataclass
class Actor:
    actor_id: str
    runtime: LiveRuntime


class MultiActorRuntime:
    def __init__(self, actors: Sequence[Actor], learner: Any):
        ids = [a.actor_id for a in actors]
        if len(set(ids)) != len(ids) or not all(ids):
            raise ValueError("actor ids must be unique and non-empty")
        for attr, what in (("agent", "agent (local beliefs and experience history)"), ("env", "environment"),
                           ("policy", "authorization"), ("run_dir", "run directory")):
            objs = [getattr(a.runtime, attr) for a in actors]
            if len({id(o) if attr != "run_dir" else o for o in objs}) != len(objs):
                raise ValueError(f"each actor needs its own {what}")
        live = [a.actor_id for a in actors if a.runtime.task.mode == "live"]
        if len(live) > 1:
            raise ValueError(f"at most one actor may act live (got {live}): several actors must never appear as "
                             "several public identities")
        for a in actors:
            if not isinstance(a.runtime.agent, ActorAgent):
                raise TypeError(f"the agent of actor {a.actor_id} lacks export_experience / weights_version / set_weights")
            a.runtime.actor_id = a.actor_id
            a.runtime.hooks["after_cycle"].append(self._sync)
        self.actors, self.learner = list(actors), learner
        self.synced: Dict[str, int] = {a.actor_id: 0 for a in actors}

    def _sync(self, rt: LiveRuntime) -> None:
        """After an actor's cycle: its new experience goes to the learner, newer weights come back."""
        agent = rt.agent
        records = agent.export_experience()
        if records:
            self.learner.submit(rt.actor_id, records)
            for u in self.learner.step():
                rt.log.learning_update(u)
        got = self.learner.pull(agent.weights_version())
        if got is not None:
            weights, version = got
            agent.set_weights(weights, version)
            self.synced[rt.actor_id] = version
            rt.log.event("weights_synced", {"actor_id": rt.actor_id, "version": version,
                                            "experience_sent": len(records)})

    def step(self) -> Dict[str, Dict[str, Any]]:
        """One loop iteration of every actor that is still running (round robin)."""
        out = {}
        for a in self.actors:
            a.runtime.start()
            if a.runtime.status == RUNNING:
                out[a.actor_id] = a.runtime.step()
        return out

    def run(self, max_rounds: int = 1000) -> Dict[str, str]:
        for _ in range(max_rounds):
            reports = self.step()
            if not reports or all(r.get("idle") for r in reports.values()):
                break
        return {a.actor_id: a.runtime.status for a in self.actors}

    def status(self) -> Dict[str, Any]:
        return {"learner": self.learner.stats(), "synced": dict(self.synced),
                "actors": {a.actor_id: {"status": a.runtime.status, "objective_id": a.runtime.task.objective_id,
                                        "decisions": a.runtime.counters["decisions"],
                                        "weights_version": a.runtime.agent.weights_version()} for a in self.actors}}
