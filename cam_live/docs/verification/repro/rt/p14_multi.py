"""G-rt probe 14 — multi-actor plumbing (spec 11.3): each actor keeps its own objective, environment, authorization,
run directory and limits; experience flows to one central learner and weights flow back; refusals.

    .venv/bin/python cam_live/docs/verification/repro/rt/p14_multi.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, StandInAgent, make_lab_run, read_log, scratch  # noqa: E402

from cam_live.contracts import LearningUpdateRecord, RunFiles, new_id  # noqa: E402
from cam_live.runtime.multi import Actor, MultiActorRuntime, RemoteSharedLearner, SharedLearner, serve_learner  # noqa: E402

R = Report("p14_multi")


class ActorAgent(StandInAgent):
    """Stand-in with the three extra methods of multi.ActorAgent."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._sent, self.w_version, self.w = 0, 0, None

    def export_experience(self):
        new = self.results[self._sent:]
        self._sent = len(self.results)
        return [{"action_id": a, "status": s} for a, s in new]

    def weights_version(self):
        return self.w_version

    def set_weights(self, weights, version):
        self.w, self.w_version = weights, version


class Central:
    def __init__(self):
        self.rows, self.version, self.dirty = [], 0, False

    def add_experience(self, actor_id, records):
        self.rows += [(actor_id, r["action_id"]) for r in records]
        self.dirty = True
        return len(self.rows)

    def learn(self, budget_s):
        if not self.dirty:
            return []
        self.dirty, self.version = False, self.version + 1
        return [LearningUpdateRecord(new_id("upd"), 0.0, "experience from actors", "frequent", param_groups=["heads"],
                                     n_experiences=len(self.rows), model_version=f"central-v{self.version}",
                                     prediction_change={"p": 0.1})]

    def weights(self):
        return {"n": len(self.rows)}, self.version


def actor(name, seed, limits=None, mode=None):
    L = make_lab_run("p14" + name, script=("lab.act", "wait"), lab_kw={"seed": seed, "target": 10_000})
    L.rt.agent = ActorAgent(L.clock, script=("lab.act", "wait"), durable=False)
    L.task.objective_id = f"obj_{name}"
    for k, v in (limits or {}).items():
        setattr(L.task.limits, k, v)
    if mode:
        L.task.mode = mode
    return L


A, B = actor("a", 3, {"max_decisions": 6}), actor("b", 4, {"max_decisions": 10})
central = Central()
shared = SharedLearner(central)
M = MultiActorRuntime([Actor("a", A.rt), Actor("b", B.rt)], shared)
M.run(max_rounds=200)
st = M.status()
by = {}
for aid, _ in central.rows:
    by[aid] = by.get(aid, 0) + 1
R.check("1 both actors ran to their OWN limit (6 and 10 decisions) and ended exhausted",
        st["actors"]["a"]["decisions"] == 6 and st["actors"]["b"]["decisions"] == 10
        and st["actors"]["a"]["status"] == st["actors"]["b"]["status"] == "exhausted", st["actors"])
R.check("2 the central learner received each actor's experience under that actor's id, nothing twice",
        by.get("a") == len(A.rt.agent.results) and by.get("b") == len(B.rt.agent.results)
        and len(set(central.rows)) == len(central.rows), by)
R.check("3 shared weights flowed back to both actors (the actor still running holds the latest version; the one that "
        "ended earlier holds the version current at its last cycle)",
        0 < A.rt.agent.w_version <= B.rt.agent.w_version == central.version, {"central": central.version,
                                                                              "a": A.rt.agent.w_version, "b": B.rt.agent.w_version})
ra = {e["data"]["action_id"] for e in read_log(A.run_dir, RunFiles.EVENTS) if e["type"] == "action_start"}
rb = {e["data"]["action_id"] for e in read_log(B.run_dir, RunFiles.EVENTS) if e["type"] == "action_start"}
R.check("4 each actor's actions ran only in its own environment and are logged only in its own run directory",
        set(A.world._actions) == ra and set(B.world._actions) == rb and not (ra & rb) and A.run_dir != B.run_dir)
R.check("5 weight syncs and central learning updates are recorded in the actors' logs",
        any(e["type"] == "weights_synced" for e in read_log(A.run_dir, RunFiles.EVENTS))
        and len(read_log(A.run_dir, RunFiles.LEARNING)) + len(read_log(B.run_dir, RunFiles.LEARNING)) >= central.version)
# refusals
for name, build in (
    ("two live actors", lambda: MultiActorRuntime([Actor("x", actor("x", 1, mode="live").rt), Actor("y", actor("y", 2, mode="live").rt)], shared)),
    ("duplicate actor ids", lambda: MultiActorRuntime([Actor("x", actor("x", 1).rt), Actor("x", actor("y", 2).rt)], shared)),
    ("one agent object shared by two actors", lambda: (lambda X, Y: (setattr(Y.rt, "agent", X.rt.agent),
                                                                     MultiActorRuntime([Actor("x", X.rt), Actor("y", Y.rt)], shared)))(
        *(lambda X, Y: (setattr(X.rt, "agent", ActorAgent(X.clock, durable=False)), (X, Y))[1])(actor("x", 1), actor("y", 2)))),
    ("an agent without the actor methods", lambda: MultiActorRuntime([Actor("x", make_lab_run("p14n").rt), Actor("y", actor("y", 2).rt)], shared)),
):
    try:
        build()
        ok = False
    except (ValueError, TypeError):
        ok = True
    R.check(f"6 refused: {name}", ok)
# the same three calls across a socket
sock = os.path.join(scratch("m"), "l")
srv = serve_learner(SharedLearner(Central()), sock)
try:
    rl = RemoteSharedLearner(sock)
    n = rl.submit("remote-actor", [{"action_id": "a1", "status": "ok"}])
    R.check("7 submit / pull / stats work across the project RPC", n == 1 and rl.stats()["received"] == {"remote-actor": 1}
            and rl.pull(-1) is not None)
    rl.close()
finally:
    srv.stop()
# a central learner that is unreachable
C, D = actor("c", 5, {"max_decisions": 6}), actor("d", 6, {"max_decisions": 6})


class Down:
    def submit(self, actor_id, records):
        raise ConnectionError("central learner unreachable")

    def step(self):
        return []

    def pull(self, v):
        raise ConnectionError("central learner unreachable")

    def stats(self):
        return {}


M2 = MultiActorRuntime([Actor("c", C.rt), Actor("d", D.rt)], Down())
try:
    M2.run(max_rounds=100)
    crashed = ""
except Exception as e:  # noqa: BLE001
    crashed = f"{type(e).__name__}: {e}"
R.check("8 an unreachable central learner does not take the acting loop down with an unhandled exception",
        not crashed, {"exception_out_of_LiveRuntime.step": crashed, "decisions_c": C.rt.counters["decisions"],
                      "status_c": C.rt.status})
sys.exit(R.finish())
