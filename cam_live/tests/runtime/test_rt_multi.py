"""Two LabWorld actors sharing one central learner (the future multi-actor design, exercised now)."""
import hashlib
import math
import os

import numpy as np
import pytest

from cam_live.contracts import LearningUpdateRecord, RunFiles, SimClock, new_id
from cam_live.diagnostics.world import DAY, AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.runtime.logs import RunReader
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig
from cam_live.runtime.multi import Actor, MultiActorRuntime, RemoteSharedLearner, SharedLearner, serve_learner
from rt_lab_agent import LabAgent


class RidgeLearner:
    """The shared learner object: one ridge regression over channel features, fed by every actor."""

    def __init__(self):
        self.rows, self.by_actor, self.w, self.version = [], {}, [0.0, 0.0, 0.0], 0
        self._dirty = False
        self.history = {}

    def add_experience(self, actor_id, records):
        for r in records:
            self.rows.append((r["features"], r["effort"], r["count"]))
        self.by_actor[actor_id] = self.by_actor.get(actor_id, 0) + len(records)
        self._dirty = self._dirty or bool(records)
        return len(self.rows)

    def learn(self, budget_s):
        if not self._dirty:
            return []
        self._dirty = False
        X = np.array([f for f, _, _ in self.rows], dtype=float)
        y = np.array([math.log1p(c / max(e, 1e-9)) for _, e, c in self.rows], dtype=float)
        old = np.array(self.w)
        new = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y)
        self.w, self.version = [float(v) for v in new], self.version + 1
        self.history[self.version] = (list(self.w), set(self.by_actor))
        h = lambda w: hashlib.sha256(np.asarray(w).round(12).tobytes()).hexdigest()[:16]  # noqa: E731
        return [LearningUpdateRecord(new_id("upd"), 0.0, f"experience from {len(self.by_actor)} actor(s)", "frequent",
                                     param_groups=["heads"], weight_delta={"heads": float(np.linalg.norm(new - old))},
                                     hash_before={"heads": h(old)}, hash_after={"heads": h(new)},
                                     n_experiences=len(self.rows), model_version=f"shared-v{self.version}",
                                     prediction_change={"weights_l2": float(np.linalg.norm(new - old))},
                                     notes="central learner shared by all actors")]

    def weights(self):
        return list(self.w), self.version


def _actor(tmp_path, name, seed, target, policy=None, mode=None):
    clock = SimClock()
    world = LabWorld(LabConfig(seed=seed, target=target, horizon_s=5 * DAY, budget=2000.0), clock)
    task = lab_task(world, f"obj_{name}")
    if mode:
        task.mode = mode
    agent = LabAgent(clock, seed=seed, only_ops=["lab.act", "wait"], wait_seconds=DAY)
    rt = LiveRuntime(agent, world, policy or AllowAllPolicy(), LabVerifier(world), task, str(tmp_path / name), clock,
                     RuntimeConfig(fsync=mode == "live"))
    return Actor(name, rt)


def test_two_actors_share_one_learner(tmp_path):
    learner = RidgeLearner()
    shared = SharedLearner(learner)
    a, b = _actor(tmp_path, "a", 11, 10000), _actor(tmp_path, "b", 12, 20000)
    multi = MultiActorRuntime([a, b], shared)
    multi.run(max_rounds=1000)
    st = multi.status()
    # experience from both actors reached the one learner object
    assert learner.by_actor["a"] > 0 and learner.by_actor["b"] > 0
    assert st["learner"]["received"] == learner.by_actor and len(learner.rows) == sum(learner.by_actor.values())
    assert learner.by_actor["a"] == len(a.runtime.agent.samples) and learner.by_actor["b"] == len(b.runtime.agent.samples)
    # weights flowed back with versions: both actors hold the learner's current weights
    assert learner.version >= 2
    assert max(a.runtime.agent.weights_version(), b.runtime.agent.weights_version()) == learner.version
    for actor in (a, b):
        ag = actor.runtime.agent
        w_then, contributors = learner.history[ag.weights_version()]
        # the weights an actor holds are exactly a published version, built from BOTH actors' experience
        assert ag.shared_w == w_then and contributors == {"a", "b"}
        ev = RunReader(actor.runtime.run_dir).records(RunFiles.EVENTS, types=["weights_synced"])
        versions = [e["data"]["version"] for e in ev]
        assert versions == sorted(versions) and versions[-1] == ag.weights_version() == st["synced"][actor.actor_id]
        assert e_ok(ev)
    # the shared weights contain knowledge neither actor produced alone
    assert st["learner"]["updates"] == learner.version
    # everything else stays per actor: objective, beliefs, experience history, environment, authorization, limits
    assert a.runtime.task.objective_id != b.runtime.task.objective_id
    assert a.runtime.task.success["target"] != b.runtime.task.success["target"]
    assert a.runtime.agent.belief != b.runtime.agent.belief
    assert {e["action_id"] for e in a.runtime.agent.experience}.isdisjoint(
        {e["action_id"] for e in b.runtime.agent.experience})
    assert a.runtime.env is not b.runtime.env and a.runtime.policy is not b.runtime.policy
    assert a.runtime.counters != b.runtime.counters and a.runtime.run_dir != b.runtime.run_dir
    assert a.runtime.env._n_exec == a.runtime.counters["actions_executed"] > 0
    # the central learner's updates are in the actors' learning logs, marked as such
    notes = [u["data"]["notes"] for u in RunReader(a.runtime.run_dir).records(RunFiles.LEARNING)]
    assert "central learner shared by all actors" in notes
    # shared weights change an actor's prior for a channel it has not tried
    ag = a.runtime.agent
    untried = [ch for ch in ag.features if ch not in ag.belief]
    if untried:
        assert ag._ab(untried[0])[0] == pytest.approx(max(0.05, math.expm1(float(np.dot(ag.shared_w, ag.features[untried[0]])))))


def e_ok(events):
    return all(e["data"]["actor_id"] for e in events)


def test_actor_separation_is_enforced(tmp_path):
    shared = SharedLearner(RidgeLearner())
    a, b = _actor(tmp_path, "a", 1, 100), _actor(tmp_path, "b", 2, 100)
    with pytest.raises(ValueError, match="unique"):
        MultiActorRuntime([a, Actor("a", b.runtime)], shared)
    same_policy = AllowAllPolicy()
    c, d = _actor(tmp_path, "c", 3, 100, policy=same_policy), _actor(tmp_path, "d", 4, 100, policy=same_policy)
    with pytest.raises(ValueError, match="own authorization"):
        MultiActorRuntime([c, d], shared)
    e, f = _actor(tmp_path, "e", 5, 100), _actor(tmp_path, "f", 6, 100)
    f.runtime.agent = e.runtime.agent
    with pytest.raises(ValueError, match="own agent"):
        MultiActorRuntime([e, f], shared)
    # several actors never mean several public identities: at most one may hold a live task
    g, h = _actor(tmp_path, "g", 7, 100, mode="live"), _actor(tmp_path, "h", 8, 100, mode="live")
    with pytest.raises(ValueError, match="at most one actor may act live"):
        MultiActorRuntime([g, h], shared)


def test_shared_learner_across_processes_interface(tmp_path):
    learner = RidgeLearner()
    shared = SharedLearner(learner)
    sock = str(tmp_path / "learner.sock")
    server = serve_learner(shared, sock, allowed_uids={os.getuid()})
    try:
        remote = RemoteSharedLearner(sock)
        assert remote.pull(-1) == ([0.0, 0.0, 0.0], 0) and remote.pull(0) is None
        n = remote.submit("far", [{"features": [1.0, 0.0, 0.5], "effort": 2, "count": 3},
                                  {"features": [0.0, 1.0, 0.2], "effort": 1, "count": 0}])
        assert n == 2 and learner.by_actor == {"far": 2}
        assert shared.step() and remote.stats()["version"] == 1
        w, v = remote.pull(0)
        assert v == 1 and w == learner.w and remote.pull(1) is None
        with pytest.raises(Exception, match="not exposed"):
            remote._rpc.call("step")                       # actors cannot make the central learner do anything else
        remote.close()
    finally:
        server.stop()
