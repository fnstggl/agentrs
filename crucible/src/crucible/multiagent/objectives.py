"""Learner-side objectives: each participant computes its own training signal from its own observations.

Clarification 2: Crucible provides experience; the learner determines what deserves reinforcement. In a self-play
civilisation the environment's reward is 0.0 (observation-only mode) and nothing here is computed by the world or
the harness - these functions belong to the reference learners (like a learning architecture's own learned
incentives) and read exactly two things: the participant's previous and current `Observation` and the result of
its own last action (plus its private `Knowledge`, itself built from those). They never read another participant's
observation, the world state, the evaluator or a team total (`INPUTS` declares this; the shared-reward-channel
audit of `crucible.multiagent.integrity` checks it by perturbation).

External drives (what a participant wants of the world; different participants have unrelated objectives):

* ``profit``    the change of the money it controls: the sum of every balance it sees in its own numbers
                (``bank.balances.*``: its own and its organisations' accounts), per 100 dollars;
* ``reach``     how much other principals do with what it owns: visible events by others whose payload names one of
                its own things (its organisations, repositories, products, packages, hosts, posts, deals);
* ``adoption``  how many distinct other principals have engaged with its things (new ones count, repeat contacts
                count little): a builder's drive for users, not money.

Intrinsic terms every reference learner adds (its own competence and curiosity, from its own action results):
a novelty bonus for the first successes of a state-changing tool (decaying with 1/sqrt of its successes) and a
small cost for an action the world rejected.
"""
from __future__ import annotations

import math
from typing import Any

from .knowledge import Knowledge

INPUTS = ("own_previous_observation", "own_observation", "own_action_result", "own_knowledge")
DRIVES = ("profit", "reach", "adoption")


def _cash(obs: dict | None) -> int:
    if not obs:
        return 0
    return int(sum(int(v) for k, v in (obs.get("numbers") or {}).items() if k.startswith("bank.balances.")))


def _mentions(x: Any, ids: set[str], depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(x, str):
        return x in ids
    if isinstance(x, dict):
        return any(_mentions(v, ids, depth + 1) for v in x.values())
    if isinstance(x, list):
        return any(_mentions(v, ids, depth + 1) for v in x[:50])
    return False


class Objective:
    """A participant's own training signal (see the module docstring)."""

    inputs = INPUTS

    def __init__(self, drive: str = "profit", *, novelty: float = 0.3, reject: float = 0.05, scale: float = 10_000.0,
                 weight: float = 1.0):
        if drive not in DRIVES:
            raise ValueError(f"drive must be one of {DRIVES}")
        self.drive, self.novelty, self.reject, self.scale, self.weight = drive, float(novelty), float(reject), \
            float(scale), float(weight)
        self._engaged: dict[str, int] = {}

    @property
    def text(self) -> str:
        return {"profit": "grow the money you control",
                "reach": "make things other people use and respond to",
                "adoption": "get as many different people as possible to use what you build"}[self.drive]

    def describe(self) -> dict:
        return {"drive": self.drive, "novelty": self.novelty, "reject": self.reject, "scale": self.scale,
                "weight": self.weight, "inputs": list(self.inputs)}

    def external(self, prev: dict | None, obs: dict, kn: Knowledge) -> float:
        if self.drive == "profit":
            if prev is None:
                return 0.0
            return max(-5.0, min(5.0, (_cash(obs) - _cash(prev)) / self.scale))
        me = obs.get("agent")
        own = set(kn.mine) | set(kn.orgs) | {me}
        others = 0
        new = 0
        for e in obs.get("events") or []:
            actor = e.get("actor")
            if not isinstance(actor, str) or actor in own or not actor.startswith(("p:", "o:", "svc:")):
                continue
            if not _mentions(e.get("payload"), own):
                continue
            others += 1
            if actor not in self._engaged:
                new += 1
            self._engaged[actor] = self._engaged.get(actor, 0) + 1
        if self.drive == "reach":
            return min(2.0, 0.25 * others)
        return min(3.0, 1.0 * new + 0.05 * (others - new))

    def intrinsic(self, action: dict | None, result: dict | None, kn: Knowledge, read_only: bool) -> float:
        if action is None or result is None:
            return 0.0
        if not result.get("ok"):
            return -self.reject
        if read_only:
            return 0.0
        n = int((kn.tool_stats.get(action.get("tool", "")) or {}).get("ok", 1))
        return self.novelty / math.sqrt(max(1, n))

    def signal(self, prev: dict | None, obs: dict, action: dict | None, result: dict | None, kn: Knowledge,
               read_only: bool) -> tuple[float, float]:
        """(external, intrinsic) parts of the signal of one decision."""
        return self.weight * self.external(prev, obs, kn), self.intrinsic(action, result, kn, read_only)
