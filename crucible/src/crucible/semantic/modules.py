"""Kernel modules of `crucible.semantic` (imported by `crucible.load_all_modules()`)."""
from .encoders import module as _embed  # noqa: F401  (module `embed`: recorded content embeddings)
from . import recording  # noqa: F401  (module `semantic_inference`, workstream D2)
