"""Kernel modules of the civilisation package (imported by `crucible.load_all_modules()`): `civgoods` (the
population's goods activity), `civgen` (cross-module genesis) and `civlinks` (couplings). It also registers the
Tier 2 domain kernel modules a civilisation composes (`fleet`, `learn`, `firms`), which the domains package
registers only on demand."""
from . import couplings, genesis, goods  # noqa: F401
from ..domains import infra_world, learning, firms  # noqa: F401
