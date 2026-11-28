"""G2-runtime probe: limits.max_external_actions and multi-step programs (spec 13.4, LX-14).

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_limits_program.py

The computer environment declares the composite tool ``program`` as modality internal, side effects none
(harness/env.py), and the candidate parser gives a program that modality (predict/candidates.py, INT2-9); each step
keeps its own modality. LiveRuntime decides whether an action is external from the top-level action only
(``_is_external``: modality / declared side effects of ``action.operation``). Here LabWorld is wrapped with such a
``program`` tool (the steps are LabWorld's external, paid ``lab.act``); the learner chooses programs of 3 ``lab.act``
steps; limits.max_external_actions = 4. The diagnostic allow-all policy is used, so the runtime's own enforcement
and accounting are what is measured (the harness authorization service additionally counts external steps itself:
harness/auth/policy.py, n_ext of the program's nodes).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rt"))
from g2common import Report, scratch  # noqa: E402
from standin import StandInAgent  # noqa: E402

from cam_live.contracts import (ActionResult, ActionSpec, Cost, Level, Modality, ResultStatus, SimClock,  # noqa: E402
                                ToolDescriptor, new_id)
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task  # noqa: E402
from cam_live.runtime.loop import LiveRuntime, RuntimeConfig  # noqa: E402

R = Report("probe_limits_program")
clock = SimClock()
world = LabWorld(LabConfig(seed=3, target=100000, budget=1000.0), clock)


class ProgramEnv:
    """LabWorld plus a composite 'program' tool declared like the computer environment declares it."""
    interface_version = "1.0"

    def __getattr__(self, name):
        return getattr(world, name)

    def discover_tools(self):
        return list(world.discover_tools()) + [ToolDescriptor(name="program", modality=Modality.INTERNAL,
                                                              level=Level.PROCEDURE, side_effects="none",
                                                              description="run steps in order")]

    def validate_action(self, action):
        if action.operation == "program":
            return [e for s in action.steps for e in world.validate_action(s)]
        return world.validate_action(action)

    def execute(self, action, auth):
        if action.operation != "program":
            return world.execute(action, auth)
        cost, obs = Cost(), []
        for s in action.steps:
            r = world.execute(s, auth)
            cost = Cost(seconds=cost.seconds + r.cost.seconds, money=cost.money + r.cost.money)
            obs += r.observations
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=clock.now(),
                            ended=clock.now(), observations=obs, cost=cost)


class ProgramAgent(StandInAgent):
    def propose(self, tools, k):
        by = {t.name: t for t in tools}
        act = by["lab.act"]
        self.n_select += 0
        steps = [ActionSpec(action_id=new_id("act"), operation="lab.act", modality=act.modality, level=act.level,
                            arguments={"channel": i % 3, "effort": 1}, est_cost=act.est_cost) for i in range(3)]
        prog = ActionSpec(action_id=new_id("act"), operation="program", modality=Modality.INTERNAL,
                          level=Level.PROCEDURE, steps=steps,
                          est_cost=Cost(money=sum(s.est_cost.money for s in steps)), provenance="g2-standin")
        return [prog]


task = lab_task(world, "obj_prog")
task.limits.max_external_actions = 4
task.limits.max_decisions = 6
run = os.path.join(scratch("prog"), "run")
agent = ProgramAgent(clock, script=["program"], durable=False)
rt = LiveRuntime(agent, ProgramEnv(), AllowAllPolicy(), LabVerifier(world), task, run, clock, RuntimeConfig(fsync=False))
status = rt.run(max_iterations=200)
R.check("1 the runtime's external-action counter equals the external actions the world executed",
        rt.counters["external_actions"] == world._n_exec,
        {"runtime_counter": rt.counters["external_actions"], "world_executed": world._n_exec,
         "decisions": rt.counters["decisions"], "status": status, "reason": rt.status_reason})
R.check("2 no more external actions are executed than limits.max_external_actions (4) allows (+1 in flight at most "
        "one program's steps)", world._n_exec <= 4 + 3, {"world_executed": world._n_exec, "limit": 4})
R.check("3 the objective ends as exhausted by max_external_actions", "max_external_actions" in rt.status_reason,
        rt.status_reason)
sys.exit(R.done())
