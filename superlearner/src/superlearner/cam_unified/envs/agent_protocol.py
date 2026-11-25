"""The agent protocol the lifetime harness drives (`AgentP`), the per-decision feedback it delivers (`Transition`),
and `PolicyAgent`, which runs any Crucible `agents.base.Policy` (scripted controls, frozen language models) as an
`AgentP` through the *same* adapter records a learned agent receives.

An agent only ever sees records: `ObservationBatch`, `ToolSchema` lists, `GoalSpec`s, its own `Action`s with
their `ExecutionResult`s, and scalar verified returns. It never gets the adapter, the operator or the evaluator.

`PolicyAgent` reconstructs the exact observation dict and result dict a Crucible policy expects from the records
(`ow_convert.event_dict`, `summary_dict`, `result_dict`), so a policy behaves as it does under Crucible's own
`Runner` - the harness test of the ExampleInductor reaching verified success is therefore also a test that the
records are lossless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..interfaces import Action, ActionSource, ObservationBatch, ObsKind
from . import ow_convert as C
from .openworld import WAIT_TOOL

DAY = 86400


@dataclass
class Transition:
    """Feedback after one decision of the batch: what was done, what it caused, what it earned.

    records  [B] EventRecord: the agent's view before acting (observations), the action, its ExecutionResult, the
             verified return increment (`reward`), cost, done flag, behaviour log-prob, provenance REAL.
    next_obs ObservationBatch of everything that arrived while time advanced (also the input of the next `act`).
    rewards  [B] float verified return increments (= records[b].reward).
    dones    [B] bool.
    dt       [B] float simulated days that elapsed for the decision (semi-Markov duration)."""
    records: list
    next_obs: ObservationBatch
    rewards: list
    dones: list
    dt: list


@runtime_checkable
class AgentP(Protocol):
    """Anything the harness can run. Optional extras the harness uses when present: ``begin(env_ids, goals)`` at
    episode start, ``usage() -> dict`` (tokens, GPU seconds), ``model_version: int``, ``save(path)`` /
    ``load(path)`` for checkpoints."""

    def act(self, obs: ObservationBatch, tools: list) -> list: ...        # -> [B] Action | None
    def observe(self, transition: Transition) -> None: ...
    def update(self) -> dict: ...


PRIVILEGED_POLICIES = frozenset({"builder", "employer", "composer", "bounty_hunter", "schedule"})


def is_privileged(policy: Any) -> bool:
    """Scripted controls that use hidden reference solvers or private task fields (feasibility ceilings only)."""
    name = str(getattr(policy, "name", "")).lower()
    return name in PRIVILEGED_POLICIES or type(policy).__name__ in ("Builder", "Employer", "Composer",
                                                                     "BountyHunter", "SchedulePolicy")


class PolicyAgent:
    """Run one Crucible `Policy` per environment as an `AgentP`.

    `policies`: a list (one per env) or a factory ``(env_index) -> Policy`` (called at every episode). A policy's
    ``None`` (wait a turn) becomes the harness ``wait`` tool for `idle_wait_s` seconds, so with the default think
    time of 600 s an idle turn lasts one simulated hour as under Crucible's Runner. `source` labels the actions
    (SCRIPTED for controls, BASELINE for frozen language models)."""

    def __init__(self, policies: Sequence[Any] | Callable[[int], Any], *, source: ActionSource = ActionSource.SCRIPTED,
                 idle_wait_s: int = 3000, max_events: int = 30, seed: int = 0, give_goal: bool = True):
        self._factory = policies if callable(policies) else None
        self._fixed = None if callable(policies) else list(policies)
        self.source = source
        self.idle_wait_s = int(idle_wait_s)
        self.max_events = int(max_events)
        self.seed = int(seed)
        self.give_goal = give_goal
        self.policies: list = []
        self.principal: list = []
        self._summary: list = []
        self._events: list = []
        self._dropped: list = []
        self._raw: list = []
        self._goals: list = []
        self.model_version = 0
        self.stats = {"acts": 0, "waits": 0}

    @property
    def name(self) -> str:
        return ",".join(sorted({getattr(p, "name", "policy") for p in self.policies})) or "policy_agent"

    def begin(self, env_ids: list, goals: list) -> None:
        n = len(env_ids)
        self.policies = [self._factory(b) if self._factory else self._fixed[b] for b in range(n)]
        self.principal = [None] * n
        self._summary = [{} for _ in range(n)]
        self._events = [[] for _ in range(n)]
        self._dropped = [0] * n
        self._raw = [None] * n
        self._goals = list(goals)

    def _absorb(self, b: int, observations: list) -> None:
        summ = [o for o in observations if o.kind is ObsKind.TEXT and o.meta.get("summary_module")]
        if summ:
            self._summary[b] = C.summary_dict(summ)
        for o in observations:
            if o.kind is ObsKind.TEXT and o.meta.get("self") and self.principal[b] is None:
                self.principal[b] = o.entity
                pol = self.policies[b]
                pol.reset(o.entity, np.random.default_rng([self.seed, b]))
                if self.give_goal and hasattr(pol, "set_goal") and b < len(self._goals) and self._goals[b]:
                    pol.set_goal(self._goals[b].text)
            elif o.kind in (ObsKind.EVENT, ObsKind.MESSAGE, ObsKind.TRANSACTION) and o.seq is not None:
                self._events[b].append(C.event_dict(o))

    def act(self, obs: ObservationBatch, tools: list) -> list:
        if not self.policies:
            raise RuntimeError("PolicyAgent.begin(env_ids, goals) must be called first (the harness does)")
        out: list[Optional[Action]] = []
        for b, olist in enumerate(obs.observations):
            self._absorb(b, olist)
            if obs.done[b] or self.principal[b] is None:
                self._raw[b] = None
                out.append(None)
                continue
            evs = self._events[b]
            dropped = max(0, len(evs) - self.max_events)
            d = {**self._summary[b], "time": int(round(obs.t[b] * DAY)), "principal": self.principal[b],
                 "events": evs[-self.max_events:], "events_dropped": dropped}
            self._events[b] = []
            specs = [C.tool_spec(t) for t in tools[b] if t.name != WAIT_TOOL]
            raw = self.policies[b].act(d, specs)
            self._raw[b] = raw
            self.stats["acts"] += 1
            if raw is None:
                self.stats["waits"] += 1
                out.append(Action(WAIT_TOOL, {"seconds": self.idle_wait_s}, source=self.source,
                                  purpose="policy idle turn"))
            else:
                out.append(Action(str(raw.get("tool")), dict(raw.get("args") or {}), source=self.source))
        return out

    def observe(self, transition: Transition) -> None:
        for b, rec in enumerate(transition.records):
            raw = self._raw[b] if b < len(self._raw) else None
            if raw is None or rec is None or rec.result is None or rec.action is None:
                continue
            self.policies[b].observe_result(raw, C.result_dict(rec.result, rec.action.tool))

    def update(self) -> dict:
        return {"updates": 0}

    def usage(self) -> dict:
        u: dict = {"acts": self.stats["acts"], "idle_turns": self.stats["waits"]}
        for p in self.policies:
            pu = p.usage() if hasattr(p, "usage") else {}
            for k, v in (pu or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    u[k] = u.get(k, 0) + v
        return u


class MixedAgent:
    """A heterogeneous population as one `AgentP`: each member agent controls a subset of the batch rows (e.g. the
    current CAM learner, historical CAM checkpoints, frozen language agents and scripted Crucible actors joined to
    one shared world with ``open_world(shared_world=True)``). Members only see their own rows; `update` is
    delegated (frozen members return zero updates)."""

    def __init__(self, members: Sequence[tuple]):
        self.members = [(agent, [int(i) for i in rows]) for agent, rows in members]
        seen = [i for _, rows in self.members for i in rows]
        if len(seen) != len(set(seen)):
            raise ValueError("a batch row belongs to exactly one member")
        self.n = len(seen)
        if sorted(seen) != list(range(self.n)):
            raise ValueError("members must cover rows 0..B-1")
        self.model_version = 0

    @staticmethod
    def _sub(batch: ObservationBatch, rows: list) -> ObservationBatch:
        return ObservationBatch([batch.env_ids[i] for i in rows], [batch.observations[i] for i in rows],
                                [batch.t[i] for i in rows], [batch.done[i] for i in rows])

    def begin(self, env_ids: list, goals: list) -> None:
        for agent, rows in self.members:
            if hasattr(agent, "begin"):
                agent.begin([env_ids[i] for i in rows], [goals[i] for i in rows])

    def act(self, obs: ObservationBatch, tools: list) -> list:
        out: list = [None] * len(obs.env_ids)
        for agent, rows in self.members:
            acts = agent.act(self._sub(obs, rows), [tools[i] for i in rows])
            for i, a in zip(rows, acts):
                out[i] = a
        return out

    def observe(self, transition: Transition) -> None:
        for agent, rows in self.members:
            agent.observe(Transition([transition.records[i] for i in rows], self._sub(transition.next_obs, rows),
                                     [transition.rewards[i] for i in rows], [transition.dones[i] for i in rows],
                                     [transition.dt[i] for i in rows]))

    def update(self) -> dict:
        out = {}
        for k, (agent, _) in enumerate(self.members):
            out[f"member{k}:{getattr(agent, 'name', type(agent).__name__)}"] = agent.update()
        return out

    def usage(self) -> dict:
        return {f"member{k}": (agent.usage() if hasattr(agent, "usage") else {})
                for k, (agent, _) in enumerate(self.members)}
