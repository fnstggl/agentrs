"""Kernel modules of `crucible.semantic` (imported by `crucible.load_all_modules()`)."""
from .encoders import module as _embed  # noqa: F401  (module `embed`: recorded content embeddings)
from . import recording  # noqa: F401  (module `semantic_inference`, workstream D2)
from .space import store as _space_store  # noqa: F401  (module `space`: shared-space records, workstream D)
