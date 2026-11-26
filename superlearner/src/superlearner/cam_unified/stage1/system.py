"""Assembly of the complete CAM Unified system for Stage-1 runs: a wrapper over the canonical factory
`build.build_agent` (plan 2026-11-26 §2; every runner builds through it).

`build_system(run_dir, SystemConfig)` = `build.build_agent(sc.scale, system=sc, run_dir=run_dir, ...)`; the System
(QwenSemantic, MemoryIndex, CamUnifiedModel from the real default components, CombinedTrainer, ArtifactStore +
EventArchive, ReplaySampler through `BranchView`, Consolidator, TraceRecorder, CamUnifiedAgent, CheckpointOps,
LabelStore, RewardModel, the LearnerHook) is described in build.py. `SystemConfig.scale` may be a DOM preset
("dom", "dom_dev", "dom_150m") or a base scale. Nothing here imports `testing` or `integration_doubles`
(build.assert_no_doubles refuses a model that contains one).
"""
from __future__ import annotations

from typing import Optional

from ..build import (System, SystemConfig, describe, make_index, make_semantic, make_trainer,  # noqa: F401
                     seed_everything, set_deterministic)
from ..build import build_agent


def build_system(run_dir: str, sc: SystemConfig, semantic=None, exclude_events: Optional[set] = None) -> System:
    """The full real stack over `run_dir` (created). `semantic` reuses an already loaded QwenSemantic (tests)."""
    return build_agent(sc.scale, system=sc, run_dir=run_dir, semantic=semantic, exclude_events=exclude_events)
