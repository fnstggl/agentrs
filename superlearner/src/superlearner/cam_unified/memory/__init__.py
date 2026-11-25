"""CAM Unified lifetime memory and continual learning (workstream E; spec Parts 10, 11, 13.7).

    archive.EventArchive        durable append-only event archive (ArchiveP)
    artifacts.ArtifactStore     content-addressed originals (ArtifactStoreP)
    index.MemoryIndex           version-partitioned keys/values, exact GPU top-k with metadata prefilters
    retrieval.Retriever         learned query / segment encoder / usefulness InfoNCE (RetrieverP)
    consolidate.Consolidator    events -> segments -> longer representations (+ artifacts, weight links)
    replay.ReplaySampler        stratified persistent replay over the archive
    checkpoint.CheckpointOps    weights-only vs full-lifetime checkpoints and resets (CheckpointOpsP)
    plasticity                  plasticity instruments and switchable mechanisms
"""
from .archive import EventArchive
from .artifacts import ArtifactStore
from .checkpoint import CheckpointOps, inspect_weights, read_weights
from .consolidate import Consolidator
from .errors import (ArchiveCorrupted, ArchiveLocked, CheckpointMismatch, IncompatibleMemoryVersion,
                     StorageLimitExceeded)
from .featurize import HashedEventEncoder
from .index import MemoryIndex
from .plasticity import (PlasticityMechanisms, PlasticityMonitor, PlasticitySwitches, RetentionTracker,
                         effective_rank, gradient_interference, learning_speed)
from .replay import ActiveCache, ReplaySampler, ReplaySlice
from .retrieval import Retriever, SegmentEncoder

__all__ = ["EventArchive", "ArtifactStore", "CheckpointOps", "inspect_weights", "read_weights", "Consolidator",
           "ArchiveCorrupted", "ArchiveLocked", "CheckpointMismatch", "IncompatibleMemoryVersion",
           "StorageLimitExceeded", "HashedEventEncoder", "MemoryIndex", "PlasticityMechanisms", "PlasticityMonitor",
           "PlasticitySwitches", "RetentionTracker", "effective_rank", "gradient_interference", "learning_speed",
           "ActiveCache", "ReplaySampler", "ReplaySlice", "Retriever", "SegmentEncoder"]
