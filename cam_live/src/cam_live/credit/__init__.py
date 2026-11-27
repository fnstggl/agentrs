"""cam_live.credit — retrospective credit assignment and continual learning (Workstream D).

Model-independent parts (stage 1): durable storage (``replay``), outcome linking (``links``), reference credit
estimators (``estimators``, ``longdelay``), learned estimators (``rudder``, ``cocoa``), the simple live attribution
and the evidence-based selector (``simple``, ``select``), temporal abstraction (``smdp``), checkpoint IO
(``checkpoint``).

On the real model (stage 2): the continual learner with its three timescales (``learner``), the RUDDER / COCOA /
bridge loss providers and the credit assigner (``providers``), retention and measured forgetting (``retention``),
complete learner checkpoints, promotion and weights-only transfer (``checkpoint``), and the LabWorld evidence
(``labkit``, ``python -m cam_live.credit.diagnose``). The public API is documented in ``docs/D/API.md``.

Submodules are imported lazily by name so that importing the store does not import torch.
"""
from .replay import DEFAULT_STRATA, PROVENANCES, SqliteExperienceStore, StoreLockedError, outcome_key  # noqa: F401

__all__ = ["SqliteExperienceStore", "StoreLockedError", "DEFAULT_STRATA", "PROVENANCES", "outcome_key"]
