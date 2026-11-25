"""The contract every Tier 1 engine implements.

An engine is the transition code of a family of worlds. It is built from a JSON config (the whole hidden mechanism,
drawn by a family's builder) and holds no mutable state: the world state is a value passed in and returned. One tick
has two phases:

1. actions: `apply(state, agent, action, rng, t)` for each atomic action (levers, discrete, graph_edit, allocate,
   coordinate plans, engine queries, experiments). It may change the controls held in the state and returns the
   action's result; information it reveals is part of the result. Errors raise `ActionError`.
2. dynamics: `advance(state, rng, t)` runs the tick and returns what each agent sees and what money moved.

The dynamics phase draws only from the tick generator of the world (`tick_rng(world_seed, env_id, t)`); the action
phase from a separate stream, so two runs that differ only in queries or experiments see the same world noise
(common random numbers), and a kernel host that only ever sets levers reproduces the dynamics exactly.

Everything an engine reveals in `advance` and `apply` results is public to the acting agent. `truth`, `outcome`,
`conservation` and `motifs` are evaluator-side and never reach an observation.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..actions import ActionSpace


class ActionError(Exception):
    """An action that is valid by schema but cannot be carried out now (no budget, unknown edge, ...)."""


class RapidEngine:
    kind: str = ""
    version: str = "1"
    batched: bool = False               # True: `batch_step` runs many instances of one structure vectorised

    def __init__(self, config: dict):
        self.config = config

    # ---- public surface ------------------------------------------------------------------------------
    @property
    def horizon(self) -> int:
        return int(self.config.get("horizon", 60))

    @property
    def agents(self) -> list[str]:
        return list(self.config.get("agents", ["a0"]))

    def action_space(self, agent: str) -> ActionSpace:
        raise NotImplementedError

    def brief(self, agent: str) -> str:
        return str(self.config.get("brief", ""))

    def objective_text(self, agent: str) -> str | None:
        return self.config.get("objective", {}).get("text")

    def query_targets(self, agent: str) -> dict[str, str]:
        """Engine query targets (name -> what it returns); the environment adds `tools`, `history` and `spec`."""
        return {}

    # ---- dynamics ------------------------------------------------------------------------------------
    def init_state(self, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def copy_state(self, state: dict) -> dict:
        out = {}
        for k, v in state.items():
            out[k] = v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)
        return out

    def apply(self, state: dict, agent: str, action: dict, rng: np.random.Generator, t: int) -> dict:
        raise ActionError(f"action kind {action.get('kind')!r} is not handled by this engine")

    def advance(self, state: dict, rng: np.random.Generator, t: int) -> dict:
        """Run tick t on `state` (a private copy the engine may change in place). Returns
        {"obs": {agent: {numbers, arrays, entities, text, events, resources, analytics}},
         "revenue": {agent: float}, "cost": {agent: float}, "info": {...private...}}."""
        raise NotImplementedError

    def coordinate(self, state: dict, plan: dict, parties: list[str], rng: np.random.Generator, t: int) -> dict:
        raise ActionError("this world has no joint plans")

    def history(self, rng_for) -> dict:
        """Observations of the world's past before the participant arrived (`arrays` and `text`), or {}."""
        return {}

    # ---- evaluator side ------------------------------------------------------------------------------
    def truth(self) -> dict:
        raise NotImplementedError

    def motifs(self) -> list[str]:
        """Mechanism motifs present in this instance, measured from its config (primitive vocabulary)."""
        raise NotImplementedError

    def outcome(self, state: dict, ledger: dict) -> dict:
        """The objective quantity so far, from the state and the environment's ledger (money per agent, journal)."""
        raise NotImplementedError

    def conservation(self, before: dict, after: dict, out: dict) -> list[str]:
        return []

    def check_state(self, state: dict) -> list[str]:
        """Consistency of a state: finite numbers within bounds."""
        errs = []
        for k, v in state.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.all(np.isfinite(v[~np.isnan(v)])):
                errs.append(f"state {k} has non-finite values")
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.size and np.nanmax(np.abs(v)) > 1e12:
                errs.append(f"state {k} is unbounded")
        return errs


ENGINE_KINDS: dict[str, type[RapidEngine]] = {}

# The engine kinds Part 7.3 asks for, and which registered engine implements each.
SPEC_ENGINE_KINDS = {
    "discrete-event": "discrete_event",
    "graph dynamical": "graph_dynamical",
    "search/information": "search",
    "executable program": None,
    "multi-agent strategic": None,
    "scientific experiment": None,
    "planning/resource": None,
    "partially observed state machine": "state_machine",
}


def register(cls: type[RapidEngine]) -> type[RapidEngine]:
    if not cls.kind:
        raise ValueError("engine without a kind")
    if cls.kind in ENGINE_KINDS and ENGINE_KINDS[cls.kind] is not cls:
        raise ValueError(f"engine kind {cls.kind} registered twice")
    ENGINE_KINDS[cls.kind] = cls
    return cls


def build(config: dict) -> RapidEngine:
    return ENGINE_KINDS[config["engine"]](config)


def jsonable(x: Any) -> Any:
    from ...core.canonical import to_jsonable
    return to_jsonable(x)
