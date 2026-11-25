"""Content-addressed artifact store (implements ``ArtifactStoreP``).

Exact originals (documents, code, long tool results, screenshots) are kept verbatim so that retrieval can return the
original when an embedding is not enough (spec 10.3: "Support retrieval of original documents, code and observations
when exact details are required").

* Reference = ``"sha256:<hex>"`` of the *uncompressed* bytes, so identical content always has the same reference
  (deduplication) independent of the codec.
* Layout: ``root/objects/<hex[0:2]>/<hex[2:4]>/<hex>.<ext>`` with ``ext`` in ``gz`` / ``xz`` / ``raw``.
* Writes are atomic (temp file -> fsync -> rename); a crash never leaves a half-written object under its final name.
* ``get`` verifies the digest of what it returns (corruption is an error, not silently wrong data).
* An explicit byte budget raises ``StorageLimitExceeded`` instead of dropping anything.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import os
from typing import Iterator, Optional

from ._fs import atomic_write_bytes, dir_bytes
from .errors import StorageLimitExceeded

_EXT = {"gzip": "gz", "lzma": "xz", "none": "raw"}
_CODEC_OF_EXT = {v: k for k, v in _EXT.items()}


def _to_bytes(data) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (dict, list, tuple, int, float, bool)) or data is None:
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raise TypeError(f"cannot store {type(data).__name__} as an artifact (bytes, str or JSON values)")


def parse_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.startswith("sha256:") or len(ref) != 7 + 64:
        raise ValueError(f"not an artifact reference: {ref!r}")
    h = ref[7:]
    int(h, 16)
    return h


class ArtifactStore:
    """Durable content-addressed blob store.

    Args:
        root: directory of the store (created if needed).
        codec: ``"gzip"`` (default), ``"lzma"`` (smaller, slower) or ``"none"``.
        level: compression level (gzip 1-9, lzma preset 0-9); ``None`` = codec default.
        min_compress: blobs shorter than this many bytes are stored raw (compression overhead exceeds the gain).
        max_bytes: explicit budget for the stored bytes; ``None`` = bounded only by the volume (an ``OSError`` from a
            full disk propagates).
        fsync: fsync every object (durability); tests of throughput may disable it.
    """

    def __init__(self, root: str, codec: str = "gzip", level: Optional[int] = None, min_compress: int = 128,
                 max_bytes: Optional[int] = None, fsync: bool = True):
        if codec not in _EXT:
            raise ValueError(f"unknown codec {codec!r}; one of {sorted(_EXT)}")
        self.root = os.path.abspath(root)
        self.codec = codec
        self.level = level
        self.min_compress = int(min_compress)
        self.max_bytes = max_bytes
        self.fsync = fsync
        os.makedirs(os.path.join(self.root, "objects"), exist_ok=True)
        self._bytes = dir_bytes(os.path.join(self.root, "objects"))
        self.stats = {"puts": 0, "dedup_hits": 0, "raw_bytes_in": 0, "stored_bytes_in": 0, "gets": 0}

    # ------------------------------------------------------------------ paths
    def _dir(self, h: str) -> str:
        return os.path.join(self.root, "objects", h[:2], h[2:4])

    def _find(self, h: str) -> Optional[str]:
        d = self._dir(h)
        for ext in ("gz", "xz", "raw"):
            p = os.path.join(d, f"{h}.{ext}")
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------ API
    def put(self, data) -> str:
        """Store ``data`` (bytes / str / JSON value) and return ``"sha256:<hex>"``. Idempotent (deduplicated)."""
        raw = _to_bytes(data)
        h = hashlib.sha256(raw).hexdigest()
        ref = "sha256:" + h
        self.stats["puts"] += 1
        self.stats["raw_bytes_in"] += len(raw)
        if self._find(h) is not None:
            self.stats["dedup_hits"] += 1
            return ref
        codec = self.codec if len(raw) >= self.min_compress else "none"
        blob = self._compress(raw, codec)
        if codec != "none" and len(blob) >= len(raw):        # incompressible content: keep it raw
            codec, blob = "none", raw
        if self.max_bytes is not None and self._bytes + len(blob) > self.max_bytes:
            raise StorageLimitExceeded(
                f"artifact store {self.root}: {self._bytes} + {len(blob)} bytes exceeds the limit {self.max_bytes}",
                used=self._bytes, limit=self.max_bytes, needed=len(blob))
        atomic_write_bytes(os.path.join(self._dir(h), f"{h}.{_EXT[codec]}"), blob, fsync=self.fsync)
        self._bytes += len(blob)
        self.stats["stored_bytes_in"] += len(blob)
        return ref

    def get(self, ref: str) -> bytes:
        """Exact original bytes of ``ref``; raises ``KeyError`` if absent, ``IOError`` if the stored blob is corrupt."""
        h = parse_ref(ref)
        p = self._find(h)
        if p is None:
            raise KeyError(ref)
        with open(p, "rb") as fh:
            blob = fh.read()
        raw = self._decompress(blob, _CODEC_OF_EXT[p.rsplit(".", 1)[1]])
        if hashlib.sha256(raw).hexdigest() != h:
            raise IOError(f"artifact {ref} is corrupt on disk ({p})")
        self.stats["gets"] += 1
        return raw

    def get_text(self, ref: str) -> str:
        return self.get(ref).decode("utf-8")

    def get_json(self, ref: str):
        return json.loads(self.get(ref))

    def exists(self, ref: str) -> bool:
        return self._find(parse_ref(ref)) is not None

    def verify(self, ref: str) -> bool:
        try:
            self.get(ref)
            return True
        except (IOError, KeyError, lzma.LZMAError, gzip.BadGzipFile, EOFError):
            return False

    def refs(self) -> Iterator[str]:
        base = os.path.join(self.root, "objects")
        for a in sorted(os.listdir(base)):
            for b in sorted(os.listdir(os.path.join(base, a))):
                for f in sorted(os.listdir(os.path.join(base, a, b))):
                    if not f.startswith(".tmp"):
                        yield "sha256:" + f.split(".", 1)[0]

    def size_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return sum(1 for _ in self.refs())

    # ------------------------------------------------------------------ codecs
    def _compress(self, raw: bytes, codec: str) -> bytes:
        if codec == "gzip":
            return gzip.compress(raw, compresslevel=6 if self.level is None else self.level, mtime=0)
        if codec == "lzma":
            return lzma.compress(raw, preset=6 if self.level is None else self.level)
        return raw

    @staticmethod
    def _decompress(blob: bytes, codec: str) -> bytes:
        if codec == "gzip":
            return gzip.decompress(blob)
        if codec == "lzma":
            return lzma.decompress(blob)
        return blob
