"""The original CAM simulators (negotiation, product, business development) as ``World`` families.

A task fixes the simulator seed, so every episode of a trial faces the same hidden counterparty preferences /
market / partners: what the agent discovers in one episode can be used in the next (fast adaptation), while
different tasks have different hidden parameters (the meta-learning distribution). Observations are the
simulators' agent-visible tokens only; verified progress / success / violations come from the simulator's own
verifier (``task_status``).

Decision budget (as in the CAM-v2 collector): an episode ends as a failure after 2 x the simulator's ``max_steps``
decisions. Invalid actions do not advance the simulator clock, so without this cap and a per-decision cost a policy
can avoid every terminal outcome by stalling (observed in the first Phase 1 runs, which were discarded). Native
reward: -0.02 per decision, a further -0.02 for an invalid action."""
from __future__ import annotations

from agentrs.cam.envs import ENVS
from agentrs.sl.world import ActionSpace, Consequence, GoalStatus, Obs, Task, World

DECISION_COST, INVALID_COST = 0.02, 0.02
SPLIT_BASE = {"train": 10_000_000, "val": 1_000_000, "test": 0}  # same seed ranges as the CAM experiments


class CamWorld(World):
    episodic = True

    def __init__(self, name: str):
        self.name = name
        self.family = f"cam/{name}"
        self.env = None
        self.task = None
        self._last = None

    def initialize_world(self, task: Task) -> None:
        self.task = task
        self.env = ENVS[self.name]()
        self.reset_world()

    def reset_world(self) -> None:
        self.env.reset(SPLIT_BASE[self.task.split] + self.task.seed, self.task.split)
        self._last = None
        self.n_dec = 0
        self.timed_out = False

    def observe(self, agent="0") -> Obs:
        st = self.env.task_status()
        nums = {"time_left": st.time_left, "progress_visible": 0.0}
        nums.update({f"res_{k}": float(v) for k, v in st.resources.items()})
        return Obs(tokens=list(self.env.obs_tokens()), numbers=nums)

    def goal(self, agent="0") -> Obs:
        return Obs(tokens=list(self.env.goal_tokens()))

    def available_actions(self, agent="0") -> ActionSpace:
        g = self.env.grammar()
        return ActionSpace(kind="sequential", allowed=g.allowed, max_len=10)

    def execute(self, agent, action) -> None:
        toks = [t for t in action if t != "END"]
        self._last = self.env.step(toks)
        self.n_dec += 1
        if not self.env.done and self.n_dec >= 2 * self.env.max_steps:
            self.timed_out = True

    def observe_consequences(self, agent="0") -> Consequence:
        r = self._last
        ev = [f"{k}" for k, v in (r.events or {}).items() if v] if r else []
        cost = DECISION_COST + (INVALID_COST if r is not None and not r.valid else 0.0)
        if r is not None and not r.valid:
            ev.append("invalid")
        return Consequence(reward=-cost, dt=float(r.elapsed_h) if r else 0.0, done=bool(self.env.done or self.timed_out),
                           events=ev)

    def evaluate_goal(self, agent="0") -> GoalStatus:
        st = self.env.task_status()
        return GoalStatus(success=bool(self.env.success) if self.env.done else False, progress=float(st.progress),
                          violated=st.violated, score=float(st.progress), time_left=float(st.time_left))

    def save_world_state(self):
        return self.env.snapshot()

    def restore_world_state(self, state) -> None:
        self.env = state.snapshot()


def make(name: str):
    return lambda: CamWorld(name)
