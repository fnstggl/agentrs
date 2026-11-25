"""Explicit failure types of the lifetime-memory package (never silent dropping)."""
from __future__ import annotations


class StorageLimitExceeded(RuntimeError):
    """A write would exceed an explicit storage budget even after the cold-archive policy ran.

    Nothing was dropped: the records that triggered it were NOT accepted (``append``) or remain durable in the write-ahead
    log (``seal``). The caller decides: raise the limit, move data to a cold volume, or stop the lifetime."""

    def __init__(self, message: str, used: int = 0, limit: int = 0, needed: int = 0):
        super().__init__(message)
        self.used = int(used)
        self.limit = int(limit)
        self.needed = int(needed)


class ArchiveCorrupted(RuntimeError):
    """A committed chunk is missing or fails validation (data loss must never be papered over)."""


class ArchiveLocked(RuntimeError):
    """Another process holds the archive's single-writer lock."""


class IncompatibleMemoryVersion(RuntimeError):
    """A query was issued against memory keys produced by an encoder version it was not trained against."""


class CheckpointMismatch(RuntimeError):
    """A checkpoint's architecture configuration does not match the object it is loaded into."""
