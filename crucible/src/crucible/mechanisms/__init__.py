"""Hidden causal mechanisms (Crucible 4.0).

`base` is the contract; `scm`, `abm` and `techsys` are three independently authored engines; `module` hosts a
mechanism inside a world as a facility (`rig`) operated through ordinary tools; `sim` runs the same mechanism
without a kernel (bit-identical, for batched learning); `evaluator` holds the private measures of discovery;
`validity` checks generated mechanisms before they are used as tasks.
"""
from .base import (ENGINES, MECH_FAMILIES, Engine, Gauge, Lever, MechFamily, StepResult, build_engine,  # noqa: F401
                   make_mechanism, mech_families, mech_family, neutral_names, register_engine, tick_rng)

_LOADED = False


def load_engines() -> None:
    """Import every engine module (each registers its engine and families on import)."""
    global _LOADED
    if _LOADED:
        return
    from . import abm, scm, techsys  # noqa: F401
    _LOADED = True
