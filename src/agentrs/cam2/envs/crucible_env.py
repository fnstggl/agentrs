"""Crucible (event-sourced persistent synthetic-internet worlds, ``crucible/`` in this repository) as a CAM 2.0
environment. The adapter itself lives with Crucible (``crucible.integrations.agentrs_cam2``) so that it is versioned
with the world's tool and goal APIs; this module only makes the Crucible package importable and re-exports it.

Task families: ``"<goal family>:<world family>:<level>[:<start-state prefix>]"``, e.g. ``"service:tiny:0"``; the
TaskRef seed is the goal/world seed. See ``crucible/docs/RL_INTEGRATION.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import crucible  # noqa: F401
except ImportError:  # running from the AGENTRS tree: use the in-repo Crucible sources
    _src = Path(__file__).resolve().parents[4] / "crucible" / "src"
    if not (_src / "crucible").is_dir():
        from agentrs.cam2.envs.base import DependencyUnavailable

        raise DependencyUnavailable(f"Crucible sources not found at {_src}; add crucible/src to PYTHONPATH")
    sys.path.insert(0, str(_src))

from crucible.integrations.agentrs_cam2 import (CrucibleAdapter, crucible_tool_spec, parse_family,  # noqa: E402,F401
                                                SCHEMA_CODES)

DEFAULT_TRAIN_FAMILIES = ["service:tiny:0"]
DEFAULT_VAL_FAMILIES = ["service:tiny:0", "revenue:tiny:0"]
SEED_RANGES = {"train": (0, 1_000_000), "val": (2_000_000, 2_001_000)}
