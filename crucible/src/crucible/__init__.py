"""Crucible: a persistent, event-sourced synthetic digital civilization for continual and multi-agent learning."""
__version__ = "0.1.0"


def load_all_modules() -> None:
    """Register every built-in module (services, economy, actors, worlds)."""
    from . import services  # noqa: F401
    from .economy import consumers  # noqa: F401
    from .actors import npcs  # noqa: F401
    from .worlds import digital_civilization  # noqa: F401
    try:
        from .worlds import engineering_lab  # noqa: F401
    except ImportError:
        pass
