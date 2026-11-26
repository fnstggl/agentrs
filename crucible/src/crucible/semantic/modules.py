"""Kernel modules of `crucible.semantic` (imported by `crucible.load_all_modules()`)."""
from .encoders import module as _embed  # noqa: F401  (module `embed`: recorded content embeddings)
from . import recording  # noqa: F401  (module `semantic_inference`, workstream D2)
from .space import store as _space_store  # noqa: F401  (module `space`: shared-space records, workstream D)
from .space import probe as _space_probe  # noqa: F401  (module `space_read`: Gate E.3 probe world)
from .generalists import module as _generalists  # noqa: F401  (module `pop_semantic`: class-4 runtime)
from . import feedprobe as _feedprobe  # noqa: F401  (Gate E.3 probe `space_feed`: the social internet's feeds)
