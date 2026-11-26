"""Class 4 of the population: compact semantic generalists (MASTER_SPEC Part 3.2 class 4, Gate H, SP-3.2d,
SP-3.3c).

The semantic participants of a world are the class-4 rows of its one population (`crucible.population`): kernel
module `pop_semantic` (`module.py`) binds that class to this runtime, so each of them is an individual with the
population's persistent identity and state, plus its own mind state in table `semp_state` (goals, memory, history,
known identifiers), persisted through snapshots, forks and replay. In every activation a participant perceives its
own situation through read-only tools, forms its own goal when it has none, and acts through the ordinary tools of
the world (`World.available_tools(principal)`, the universal tool interface) - every world-changing call chosen and
filled by the shared small language models (`brain.py`, routed per capability by `crucible.semantic.routing`), all
inference recorded so replay never runs a model (`mind.py`).

    civ_world(profile, seed, population="standard", **kw) -> World     a civilisation with the runtime bound
    members(world) -> list[dict]                                          evaluator view of the participants' state
    verify.measure(world) -> dict                                         evaluator-side outcome measurements

`brain="ablated"` in the module config replaces every model decision by a uniform random one (the SP-3.2d ablation).
"""
from __future__ import annotations

from .module import NAME, TABLE, SemanticGeneralists, rows as members  # noqa: F401
from .world import civ_world, ensure_module  # noqa: F401

__all__ = ["NAME", "TABLE", "SemanticGeneralists", "members", "civ_world", "ensure_module"]
