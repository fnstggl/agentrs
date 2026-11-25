"""Content encoders (INTERFACE_CONTRACTS section 4; MASTER_SPEC 4.3, 13.5): real pretrained models turn artifacts into
vectors, cached by (artifact SHA-256, encoder id, encoder version, preprocessing hash) and recorded for replay.

    encode(world, ref, modality, *, ctx=None, data=None) -> np.ndarray        float32 [dim]
    encode_many(world, refs, modality, *, ctx=None, data=None) -> np.ndarray  float32 [n, dim]
    encoder_info(modality) -> {id, version, dim, params, preprocessing_hash, ...}
    cache_key(ref, modality) -> "{ref}|{encoder_id}@{version}|{preprocessing_hash}"
    encode_bytes(data, modality) -> np.ndarray                                 (no world: tools, calibration, tests)

| modality | encoder | dim |
|---|---|---|
| text  | MiniLM-L6 mean pooling over 256-token windows (HTML -> visible text) | 384 |
| image | CLIP ViT-B/32 image features | 512 |
| audio | AST (AudioSet) sound embedding + Whisper-tiny transcript through the text encoder | 1152 |
| video | 8 sampled frames through CLIP in 4 ordered segments + audio track + 8 temporal features | 3208 |
| code  | declaration summary through the text encoder + 32 structural features | 416 |

Where a vector comes from, in order:

1. the world's own record (module `embed`, table `embeddings`): a vector already recorded in this world is returned
   exactly as recorded, so replay, regeneration, snapshots and forks see identical vectors;
2. the process memory cache, then the disk cache (`$CRUCIBLE_CACHE/embeddings`), shared by every participant and
   every world on the machine - a video is encoded once, never per viewer;
3. the models, on the artifact's bytes (from `crucible.substrate.artifacts.get(world, ref)`, or `data=` when the
   caller already holds them; the bytes must hash to `ref`).

With `ctx` (inside a command or process), a vector not yet recorded in the world is recorded through
`crucible.substrate.inference.recorded` (falling back to `ctx.external` with the same record shape where the substrate
is not installed) and, when the `embed` module is installed, stored in its table by one `embed.recorded` event.
Without `ctx` nothing is recorded (evaluators, tools, offline analysis).

Models run on the GPU when the slot provides one (`CUDA_VISIBLE_DEVICES`), else on CPU; `CRUCIBLE_ENCODER_DEVICE`
overrides. Weights come only from the local Hugging Face cache (`fetch()` downloads the pinned revisions once).
"""
from __future__ import annotations

import base64
import hashlib
import threading
from typing import Any, Sequence

import numpy as np

from ..cache import EmbeddingCache, default_cache, vector_sha256
from . import models
from .base import REGISTRY, Encoder, recorded_json
from . import text as _text, image as _image, audio as _audio, video as _video, code as _code  # noqa: F401  (register)

MODALITIES = ("text", "image", "audio", "video", "code")
TABLE = "embeddings"
_LOCK = threading.RLock()
_COUNTS = {"vectors_computed": {m: 0 for m in MODALITIES}, "recorded": 0, "world_hits": 0}


class EncodingError(ValueError):
    """The artifact could not be encoded in this modality (unreadable bytes, wrong kind of content)."""


def ref_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hex(ref: str) -> str:
    if not isinstance(ref, str) or not ref.startswith("sha256:") or len(ref) != 71:
        raise EncodingError(f"not an artifact reference: {str(ref)[:80]!r}")
    h = ref[7:]
    if any(c not in "0123456789abcdef" for c in h):
        raise EncodingError(f"not an artifact reference: {ref[:80]!r}")
    return h


def encoder(modality: str) -> Encoder:
    try:
        return REGISTRY[modality]
    except KeyError:
        raise EncodingError(f"unknown modality {modality!r} (one of {', '.join(MODALITIES)})") from None


def cache_key(ref: str, modality: str) -> str:
    _hex(ref)
    e = encoder(modality)
    return f"{ref}|{e.id}@{e.version()}|{e.preprocessing_hash()}"


def encoder_info(modality: str) -> dict:
    """{id, version, dim, params, preprocessing_hash} plus models, model_hash, preprocessing and device. Loads the
    models (to count their parameters)."""
    return encoder(modality).info()


def stats() -> dict:
    """Cache statistics, vectors computed per modality, recorded batches, and model forward passes."""
    with _LOCK:
        return {"cache": dict(default_cache().stats), "vectors_computed": dict(_COUNTS["vectors_computed"]),
                "recorded": _COUNTS["recorded"], "world_hits": _COUNTS["world_hits"],
                "forward_passes": models.forward_passes()}


# ============================================================================================== computing
def _artifact_bytes(world, ref: str) -> bytes:
    try:
        from ...substrate import artifacts as _art
        get = getattr(_art, "get", None)
    except ImportError:
        get = None
    if get is None or world is None:
        raise EncodingError(f"no bytes for {ref}: pass data= or install the substrate artifact store")
    return get(world, ref)


def _embed_bytes(modality: str, datas: Sequence[bytes]) -> np.ndarray:
    from .. import media
    enc = encoder(modality)
    try:
        mat = np.asarray(enc.embed(list(datas)), dtype=np.float32)
    except media.MediaError as e:
        raise EncodingError(f"{modality}: {e}") from e
    if mat.shape != (len(datas), enc.dim) or not np.all(np.isfinite(mat)):
        raise EncodingError(f"{modality} encoder returned {mat.shape} / non-finite values")
    with _LOCK:
        _COUNTS["vectors_computed"][modality] += len(datas)
    return mat


def _vectors(world, refs: Sequence[str], keys: Sequence[str], modality: str, datas: Sequence[bytes | None] | None,
             cache: EmbeddingCache) -> np.ndarray:
    """Cache, then models (batched) for the misses; every computed vector goes into the cache."""
    enc = encoder(modality)
    out = np.zeros((len(refs), enc.dim), dtype=np.float32)
    todo, todo_bytes = [], []
    for i, (ref, key) in enumerate(zip(refs, keys)):
        v = cache.get(key)
        if v is not None:
            out[i] = v
            continue
        data = datas[i] if datas is not None and datas[i] is not None else _artifact_bytes(world, ref)
        if hashlib.sha256(data).hexdigest() != _hex(ref):
            raise EncodingError(f"bytes do not match {ref}")
        todo.append(i)
        todo_bytes.append(data)
    if todo:
        mat = _embed_bytes(modality, todo_bytes)
        for j, i in enumerate(todo):
            out[i] = cache.put(keys[i], mat[j], {"ref": refs[i], "modality": modality})
    return out


def encode_bytes(data: bytes, modality: str, *, cache: EmbeddingCache | None = None) -> np.ndarray:
    """Encode raw bytes (no world, nothing recorded); cached by the same key as `encode`."""
    return encode_bytes_many([data], modality, cache=cache)[0]


def encode_bytes_many(datas: Sequence[bytes], modality: str, *, cache: EmbeddingCache | None = None) -> np.ndarray:
    refs = [ref_of(d) for d in datas]
    keys = [cache_key(r, modality) for r in refs]
    out = _vectors(None, refs, keys, modality, list(datas), cache or default_cache())
    out.setflags(write=False)
    return out


# ============================================================================================== recording
def _b64(mat: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(mat, dtype="<f4").tobytes()).decode("ascii")


def _unb64(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float32).reshape(shape)


def _record(ctx, modality: str, keys: list[str], compute) -> np.ndarray:
    """Run `compute()` -> [n, dim] once and record it (substrate.inference.recorded when available, else
    ctx.external); regeneration returns the recorded matrix without running the models."""
    enc = encoder(modality)
    inputs = {"keys": list(keys)}
    with _LOCK:
        _COUNTS["recorded"] += 1
    if "artifacts" in ctx.world.modules:
        try:
            from ...substrate import inference as _inf
            recorded = getattr(_inf, "recorded", None)
        except ImportError:
            recorded = None
        if recorded is not None:       # array output -> content-addressed array blob, its REF in the record
            mat = recorded(ctx, f"embed.{modality}", enc.id, enc.model_hash(), inputs,
                           lambda: np.ascontiguousarray(compute(), dtype=np.float32))
            return np.array(mat, dtype=np.float32)
    out = recorded_json(ctx, f"embed.{modality}", enc.id, enc.model_hash(), inputs,
                        lambda: (lambda m: {"shape": list(m.shape), "f32le_b64": _b64(m)})(compute()))
    return _unb64(out["f32le_b64"], out["shape"])


def _world_vector(world, key: str) -> np.ndarray | None:
    if world is None or "embed" not in getattr(world, "modules", {}):
        return None
    rec = world.state.get(TABLE, key)
    if rec is None:
        return None
    dec = world.runtime.setdefault("embed.decoded", {})        # runtime cache of decoded records (never persisted)
    hit = dec.get(key)
    if hit is not None and hit[0] == rec["sha256"]:
        return hit[1]
    v = _unb64(rec["vec"], (rec["dim"],))
    v.setflags(write=False)
    dec[key] = (rec["sha256"], v)
    return v


def encode_many(world, refs: Sequence[str], modality: str, *, ctx=None,
                data: Sequence[bytes | None] | None = None) -> np.ndarray:
    """[n, dim] float32 vectors of artifacts `refs` (see the module doc for where each vector comes from)."""
    enc = encoder(modality)
    refs = list(refs)
    keys = [cache_key(r, modality) for r in refs]
    if data is not None and len(data) != len(refs):
        raise EncodingError("data must align with refs")
    out = np.zeros((len(refs), enc.dim), dtype=np.float32)
    missing: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        v = _world_vector(world, k)
        if v is not None:
            out[i] = v
            with _LOCK:
                _COUNTS["world_hits"] += 1
        else:
            missing.setdefault(k, []).append(i)
    if missing:
        uniq = list(missing)                                   # first-occurrence order: deterministic
        first = [missing[k][0] for k in uniq]
        u_refs = [refs[i] for i in first]
        u_data = [data[i] for i in first] if data is not None else None
        cache = default_cache()

        def compute() -> np.ndarray:
            return _vectors(world, u_refs, uniq, modality, u_data, cache)
        mat = _record(ctx, modality, uniq, compute) if ctx is not None else compute()
        for j, k in enumerate(uniq):
            for i in missing[k]:
                out[i] = mat[j]
        if ctx is not None and "embed" in ctx.world.modules:
            items = [{"key": k, "ref": u_refs[j], "modality": modality, "encoder": enc.id, "version": enc.version(),
                      "prep": enc.preprocessing_hash(), "dim": int(enc.dim), "sha256": vector_sha256(mat[j]),
                      "vec": _b64(mat[j])} for j, k in enumerate(uniq)]
            ctx.emit("embed.recorded", {"items": items}, vis="internal")
    out.setflags(write=False)
    return out


def encode(world, ref: str, modality: str, *, ctx=None, data: bytes | None = None) -> np.ndarray:
    """float32 [dim] vector of artifact `ref` in `modality` (text|image|audio|video|code)."""
    return encode_many(world, [ref], modality, ctx=ctx, data=[data] if data is not None else None)[0]


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def fetch() -> dict[str, str]:
    """Download the pinned model files into the local Hugging Face cache."""
    return models.fetch()


__all__ = ["MODALITIES", "encode", "encode_many", "encode_bytes", "encode_bytes_many", "encoder_info", "cache_key",
           "ref_of", "stats", "similarity", "fetch", "EncodingError", "encoder"]
