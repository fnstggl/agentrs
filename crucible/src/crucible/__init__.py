"""Crucible: a persistent, event-sourced synthetic digital civilization for continual and multi-agent learning."""
__version__ = "0.4.0"


def load_all_modules() -> None:
    """Register every built-in module (services, economy, actors, worlds)."""
    from . import services  # noqa: F401
    from .economy import consumers, operations, industry, shoppers  # noqa: F401
    from .actors import npcs, minds  # noqa: F401
    from .worlds import digital_civilization, open_industry  # noqa: F401
    from .worldgen import sitegen, adversarial  # noqa: F401
    from .goals import selfgoals  # noqa: F401
    try:
        from .worlds import engineering_lab  # noqa: F401
    except ImportError:
        pass
    # Crucible 4.0 (OpenWorld): the usefulness substrate, hidden-mechanism facilities, market entry, and the
    # world families built on them (`ow_*`, `mech_*`)
    from .usefulness import module as _use  # noqa: F401
    from .mechanisms import module as _rig  # noqa: F401
    from .actors import entry  # noqa: F401
    from .worldgen import mech_families, ow_families  # noqa: F401
