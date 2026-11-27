"""cam_live.credit — retrospective credit assignment and continual learning (Workstream D).

Stage 1 (this package): durable storage (``replay``), outcome linking (``links``), reference credit estimators
(``estimators``, ``longdelay``), learned estimators (``rudder``, ``cocoa``), the simple live attribution and the
evidence-based selector (``simple``, ``select``), temporal abstraction (``smdp``) and checkpoint IO
(``checkpoint``). The public API is documented in ``docs/D/API.md``.

Submodules are imported lazily by name so that importing the store does not import torch.
"""
from .replay import DEFAULT_STRATA, PROVENANCES, SqliteExperienceStore, StoreLockedError, outcome_key  # noqa: F401

__all__ = ["SqliteExperienceStore", "StoreLockedError", "DEFAULT_STRATA", "PROVENANCES", "outcome_key"]
