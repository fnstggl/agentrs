"""Crucible 5.0 - The autonomous simulated population: the ONE population of every v5 Tier 2 and Tier 3 world
(DECISION_LOG D-15, ONE_WORLD_CONTRACTS section 1).

Stable API (other workstreams import only these names):

    register_activity(world, name, handler, *, owner, prior=0.0, goals=())   handler(ctx, idx, now) -> utility array
    n_individuals(world) -> int
    features(world, idx) -> dict[str, np.ndarray]        public decision features (no evaluator data)
    interests(world, idx) -> np.ndarray                  (n, semantics.DIM) float32 unit vectors in the shared space
    spendable(world, idx) -> np.ndarray                  cents an individual can spend now
    pay(ctx, i, to_account, cents, *, memo) -> str       materialise principal + bank.move from its own account
    wake(ctx, idx, at, activity)
    class_members(world, cls) -> np.ndarray
    record_decisions(ctx, kind, idx, values, *, model_id, model_version, fidelity_id, mode="set", table=None) -> str
    experience(ctx, idx, activity, utility, *, item_vecs=None, satisfaction=None)

plus `interest_coords` / `to_coords` (the compact coordinates the population stores) and the evaluator-side
`check_provenance`. Kernel modules live in `population/modules.py` (`crucible.load_all_modules()` imports it).
"""
from .core import (activities, activity_slot, class_members, features, n_individuals, register_activity,  # noqa: F401
                   spendable, wake)
from .api import (check_provenance, experience, interest_coords, interests, pay, participant_decision,  # noqa: F401
                  record_decisions, to_coords)
from . import semantics  # noqa: F401

__all__ = ["register_activity", "n_individuals", "features", "interests", "spendable", "pay", "wake",
           "class_members", "record_decisions", "experience", "interest_coords", "to_coords", "check_provenance",
           "activities", "activity_slot", "participant_decision", "semantics"]
