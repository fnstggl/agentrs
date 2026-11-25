"""Kernel modules of the social internet, imported by `crucible.load_all_modules()`.

Workstream C2: `crowd` (columnar population of the scale path), `aud` (content index and exact audience
accounting), `feed` (recommendation and the consumption engine)."""
from . import scale  # noqa: F401  (module crowd)
from . import audience  # noqa: F401  (module aud)
from .recsys import service  # noqa: F401  (module feed)
