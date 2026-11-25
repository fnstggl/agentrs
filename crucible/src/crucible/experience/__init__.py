"""Crucible 5.0 - The architecture-independent learner interface: observe/execute/observe_since, reward modes,
experience storage, adapters, reference learner.

Owned by workstream L (docs/v5/WORKSTREAMS.md). Kernel modules of this package live in `experience/modules.py`
(`crucible.load_all_modules()` imports it). How a learner connects: docs/v5/LEARNER_GUIDE.md.

Public interface (everything a learner needs; nothing else is required to connect):

    make(tier, name, seed, reward_mode, **kw) -> ExperienceEnv
    ExperienceEnv: agents, observe, execute, observe_since, tools, advance, reward, verify
    Observation, validate_observation, validate_action, validate_result, kinds_present, OBSERVATION_KINDS
    REWARD_MODES, LEGACY, VERIFIED, OBSERVATION_ONLY
    ExperienceStore, make_record
    ReferenceLearner (crucible.experience.reference)
    run_learner (crucible.experience.loop), VecExperience / farm hooks (crucible.experience.parallel)

Attributes are imported lazily so that the numeric environments (`research.envs`, which import
`crucible.experience.modes`) never pull in the kernel adapters.
"""
from __future__ import annotations

import importlib

from .modes import LEGACY, OBSERVATION_ONLY, REWARD_MODES, VERIFIED, check_mode  # noqa: F401
from .observation import (OBSERVATION_KINDS, Observation, kinds_present, validate_action,  # noqa: F401
                          validate_observation, validate_result)

_LAZY = {
    "ExperienceEnv": ".env", "BaseExperienceEnv": ".env", "INTERFACE_VERSION": ".env",
    "make": ".registry", "TierUnavailable": ".registry", "tiers_available": ".registry", "domains": ".registry",
    "MechExperienceEnv": ".tier1", "RapidExperienceEnv": ".tier1",
    "KernelExperienceEnv": ".kernel",
    "ExperienceStore": ".store", "make_record": ".store",
    "ReferenceLearner": ".reference",
    "run_learner": ".loop",
    "VecExperience": ".parallel", "FarmRun": ".parallel",
    "TOOL_CATEGORIES": ".features", "tool_category": ".features",
}

__all__ = sorted(["LEGACY", "OBSERVATION_ONLY", "REWARD_MODES", "VERIFIED", "check_mode", "OBSERVATION_KINDS",
                  "Observation", "kinds_present", "validate_action", "validate_observation", "validate_result",
                  *_LAZY])


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'crucible.experience' has no attribute {name!r}")
    val = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = val
    return val
