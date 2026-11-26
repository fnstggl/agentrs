"""Experience entry points of Tier 3 (ONE_WORLD_CONTRACTS section 4): `tier3:<profile>`.

`make(name, seed, reward_mode=None, **kw)` builds the civilisation and wraps it in `crucible.experience`'s
`KernelExperienceEnv`. The reward mode defaults to `observation_only` (Clarification 2, D-19): learners receive the
world's observable consequences and a scalar reward of 0.0; the legacy shaped reward is available only when asked
for by name. `crucible.experience.registry.make(3, "<profile>", ...)` reaches the same generator (workstream L owns
that registry and its defaults; this function is the Tier 3 side of the contract).
"""
from __future__ import annotations

from . import profiles

PREFIX = "tier3:"
OBSERVATION_ONLY = "observation_only"


def names() -> list[str]:
    return [PREFIX + p for p in profiles.PROFILES]


def make(name: str, seed: int = 0, reward_mode: str | None = None, *, population: str = "development",
         overrides: dict | None = None, **kw):
    from ..experience.kernel import KernelExperienceEnv
    from ..experience.modes import check_mode
    from .generator import generate
    prof = name[len(PREFIX):] if name.startswith(PREFIX) else name
    mode = check_mode(reward_mode if reward_mode is not None else OBSERVATION_ONLY)
    world = generate(prof, int(seed), population, overrides=overrides)
    return KernelExperienceEnv(world, int(seed), reward_mode=mode, tier=3, name=f"civ:{prof}", **kw)
