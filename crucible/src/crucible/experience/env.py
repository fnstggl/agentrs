"""The `ExperienceEnv` protocol (INTERFACE_CONTRACTS section 11) and the base class the adapters share.

    env = crucible.experience.make(tier, name, seed, reward_mode="observation_only")
    for agent in env.agents():
        obs = env.observe(agent)                     # Observation (crucible.experience.observation)
        res = env.execute(agent, {"tool": "rig.set", "args": {...}})   # {ok, result | error, cost_s, t, tool}
    env.advance(3600)                                # time passes (execute never moves the clock)
    events, cursor = env.observe_since(agent, cursor)   # lossless: every visible event after `cursor`
    r = env.reward(agent)                            # 0.0 in observation_only, always
    outcome = env.verify()                           # evaluator side; never part of an observation

Executing never moves the clock: an action reports its duration (`cost_s`) and the driver decides how much time
passes (`advance`), so the interface is a semi-Markov decision process in every tier.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .modes import check_mode
from .observation import Observation

INTERFACE_VERSION = "crucible-experience/1"
RewardMode = Literal["legacy_shaped_reward", "verified_terminal_outcome", "observation_only"]


@runtime_checkable
class ExperienceEnv(Protocol):
    reward_mode: RewardMode

    def agents(self) -> list[str]: ...

    def observe(self, agent: str) -> Observation: ...

    def execute(self, agent: str, action: dict) -> dict: ...

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]: ...

    def tools(self, agent: str) -> list[dict]: ...

    def advance(self, seconds: int) -> None: ...

    def reward(self, agent: str) -> float: ...

    def verify(self) -> dict: ...


class BaseExperienceEnv:
    """Shared bookkeeping. Subclasses implement the protocol methods; `describe()` and `digest()` are the
    operator-side identity of the run (used for experience records and reproducibility checks)."""
    tier: int = 0
    kind: str = "abstract"

    def __init__(self, reward_mode: str, name: str, seed: int):
        self.reward_mode = check_mode(reward_mode)
        self.name = str(name)
        self.seed = int(seed)
        self.verifications = 0             # how many independent verifier outcomes `verify()` has produced
        self.steps = 0

    def describe(self) -> dict:
        return {"interface": INTERFACE_VERSION, "tier": self.tier, "kind": self.kind, "name": self.name,
                "seed": self.seed, "reward_mode": self.reward_mode}

    def digest(self) -> str:                # pragma: no cover - every adapter overrides it
        raise NotImplementedError

    @property
    def now(self) -> int:                   # pragma: no cover - every adapter overrides it
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def error_result(t: int, tool: str, code: str, message: str) -> dict:
    return {"ok": False, "t": int(t), "tool": tool, "cost_s": 0, "error": {"code": code, "message": message}}
