"""Kernel modules of the shared substrate; `crucible.load_all_modules()` imports this file.

* `artifacts` (host `artifact`): content-addressed immutable artifacts (crucible.substrate.artifacts)
* `cols` (no tools): the `cols.batch` / `cols.grow` appliers of columnar state (crucible.substrate.columns)
* `assets` (host `asset`): generic ownable, transferable assets (crucible.substrate.assets)
"""
from . import artifacts, assets, columns  # noqa: F401  (registration on import)
from . import primitives

# the typed schemas of the 4.0 owners' shared-fact events, registered explicitly when the modules are loaded
primitives.register_owner_events()
