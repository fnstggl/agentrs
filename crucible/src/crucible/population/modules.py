"""Kernel modules of the population package (imported by `crucible.load_all_modules()`).

`pop` (core), `popweb` (resource-visit engine), `popsites` (genesis web sites), `pop_adaptive` (class 2) and
`pop_strategic` (class 3). The torch networks are imported only when a world uses a learning class.

The kernel module `space` that population worlds list is the ONE shared semantic space's record
(`crucible.semantic.space.store`, CORE-SEM); `population.semantics` delegates to it and registers no module.
"""
from . import core, sites, web, adaptive, strategic  # noqa: F401
from ..semantic.space import store as _space_store  # noqa: F401  (module `space`, registered by CORE-SEM)
