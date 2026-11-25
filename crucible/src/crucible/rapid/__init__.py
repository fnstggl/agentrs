"""Crucible 5.0 - Tier 1 rapid scientific environments: causal primitives, new engines, structural generator,
fingerprints, families, general action interface (INTERFACE_CONTRACTS section 8).

Owned by workstream H (docs/v5/WORKSTREAMS.md). Kernel modules of this package live in `rapid/modules.py` or are
imported from it; `crucible.load_all_modules()` imports that file when it exists.

    from crucible import rapid
    env = rapid.make("hc_confounded", seed=0, horizon=60)       # a RapidEnv
    obs = env.reset(); r = env.step({"kind": "levers", "values": {...}}); env.outcome()
    rapid.primitives.REGISTRY      # the 28 versioned causal primitives
    rapid.families.REGISTRY        # families: engine, category, split, fingerprint, doc, ...
    rapid.fingerprint("net_routing"); rapid.admit("net_routing", seeds=(0, 1, 2))   # evaluator side

World-side modules (env, engines, families, actions, observation, generator, primitives, mechview, batch,
program) never import the evaluator-side ones (oracles, admission, claims, identify, references, fingerprints, report,
primitive_checks); `fingerprint` and `admit` here import them only when called.
"""
from . import primitives  # noqa: F401
from .actions import KINDS, SPEC_ACTION_TYPES, ActionSpace  # noqa: F401
from .env import Env, RapidEnv  # noqa: F401
from . import families  # noqa: F401  (the module: crucible.rapid.families.REGISTRY)
from .families import CATEGORIES, DIFFICULTY_AXES, FAMILIES, REGISTRY, make  # noqa: F401


def fingerprint(family: str, **difficulty) -> dict:
    """The measured structural fingerprint of a family (evaluator side)."""
    from .fingerprints import family_fingerprint
    return family_fingerprint(family, **difficulty)


def admit(family: str, seeds=(0, 1, 2), **difficulty) -> dict:
    """The Part 7.4 admission checks of a family on `seeds` (evaluator side)."""
    from .admission import admit as _admit
    return _admit(family, seeds, **difficulty)
