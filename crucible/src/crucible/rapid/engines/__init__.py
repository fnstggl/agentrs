"""Tier 1 engines. Each module holds one independently written transition function and registers it with
`base.register`; they share the contract in `base` (and the primitive registry where a mechanism is reused) and
nothing else."""
from .base import ENGINE_KINDS, SPEC_ENGINE_KINDS, ActionError, RapidEngine, build, register  # noqa: F401
from . import primitive_graph  # noqa: F401,E402
from . import discrete_event  # noqa: F401,E402
from . import graph_dynamical  # noqa: F401,E402
from . import state_machine  # noqa: F401,E402
from . import search  # noqa: F401,E402

V5_ENGINES = tuple(sorted(ENGINE_KINDS))
