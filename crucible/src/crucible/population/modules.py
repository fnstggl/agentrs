"""Kernel modules of the population package (imported by `crucible.load_all_modules()`).

`pop` (core), `popweb` (resource-visit engine), `popsites` (genesis web sites), `pop_adaptive` (class 2) and
`pop_strategic` (class 3). The torch networks are imported only when a world uses a learning class.
"""
from . import core, semantics, sites, web, adaptive, strategic  # noqa: F401
