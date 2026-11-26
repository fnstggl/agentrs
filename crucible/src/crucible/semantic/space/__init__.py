"""The shared semantic space (ONE_WORLD_CONTRACTS section 2, decision D-16): one vector space for content of every
modality, queries, needs, descriptions and individuals' interests.

    DIM, SPACE_ID, SPACE_VERSION
    item_vector(world_or_ctx, ref) -> (DIM,) float32 unit        item_vectors(world_or_ctx, refs) -> (n, DIM)
    item_chunks(world_or_ctx, ref) -> (k, DIM)                    chunk-level vectors (local information)
    text_vector(world_or_ctx, text) -> (DIM,)                    text_vectors(world_or_ctx, texts) -> (n, DIM)
    item_features(world_or_ctx, ref) -> dict                      intrinsic features computed from the bytes
    interest_prior(world_or_ctx, rng, n) -> (n, DIM)              initial interests from real reading histories (MIND)
    vectors_of_bytes(datas) / features_of_bytes(datas)            the same without a world (tools, tests, analysis)

Pass the handler's `ctx` as the first argument (or `ctx=`) inside a command, process or population activity: what is
computed is then recorded in the world (`crucible.substrate.inference.recorded` + module `space`), so replay,
regeneration, snapshots and forks return exactly the recorded vectors and never run an encoder. With a bare world
the world's records are read and anything missing is computed (deterministically on the same machine) but not recorded.

Layout (`layout()`): ``[topic 384 | affect 4 | quantity 34]`` (block weights 0.78 / 0.10 / 0.12 of the squared norm).
The topic block is all-MiniLM-L6-v2 over 128-piece chunks (other modalities mapped in by fitted, measured maps,
`align`); affect = [valence (Qwen3-0.6B), negation, hedging, questions]; quantity = typed log-magnitude features of
every number, date and time (`textual`). See `compute` for each modality.

Versioning (`SPACE_VERSION`) is derived, not declared: a hash of the source of every module that computes vectors
(space and encoders and media decoding), the pinned model revisions and the alignment-map fingerprint. The cache key
adds a hash of the live parameter values actually used (`preprocessing()`: module constants, encoder attributes,
library versions, device class), so changing any of them - in code or at run time - invalidates the cache.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
import threading
from typing import Any, Sequence

import numpy as np

from . import store as _store
from .compute import AFFECT, AFFECT_DIM, DIM, MODALITIES, QUANT, QUANT_DIM, TOPIC, TOPIC_DIM, W_AFFECT, W_QUANT, \
    W_TOPIC

SPACE_ID = "crucible.space.minilm384+affect4+quant34"
_PKG = os.path.dirname(os.path.dirname(__file__))           # crucible/semantic
VECTOR_SOURCES = ("space/textual.py", "space/lm.py", "space/align.py", "space/compute.py", "encoders/text.py",
                  "encoders/image.py", "encoders/audio.py", "encoders/video.py", "encoders/code.py",
                  "encoders/models.py", "encoders/base.py", "media.py")
PARAM_MODULES = ("crucible.semantic.space.textual", "crucible.semantic.space.lm", "crucible.semantic.space.align",
                 "crucible.semantic.space.compute")
_LOCK = threading.RLock()
COUNTS = {"computed": {m: 0 for m in MODALITIES}, "world_hits": 0, "recorded_batches": 0, "cache_hits": 0}


# ============================================================================================== identity
def source_hash(paths: Sequence[str] = VECTOR_SOURCES) -> str:
    h = hashlib.sha256()
    for rel in sorted(paths):
        with open(os.path.join(_PKG, rel), "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def _models() -> dict:
    from ..encoders.models import SPECS
    out = {r: s.revision for r, s in SPECS.items()}
    try:
        # model id and revision only: `lm.identity()` also resolves the device (torch.cuda.is_available() starts a
        # CUDA driver thread), which must not happen at import time in every process (DEF-P-25)
        from ..service import local_revision
        from .lm import MODEL_ID
        out["lm"] = MODEL_ID + "@" + local_revision(MODEL_ID)
    except (OSError, FileNotFoundError, ImportError):
        out["lm"] = "unavailable"
    return out


def _version() -> str:
    from .align import maps_fingerprint
    ident = {"code": source_hash(), "models": _models(), "maps": maps_fingerprint(), "dim": DIM}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


SPACE_VERSION = _version()


def layout() -> dict:
    return {"dim": DIM, "blocks": {"topic": [TOPIC.start, TOPIC.stop], "affect": [AFFECT.start, AFFECT.stop],
                                   "quantity": [QUANT.start, QUANT.stop]},
            "weights": {"topic": W_TOPIC, "affect": W_AFFECT, "quantity": W_QUANT},
            "affect": ["valence", "negation", "hedging", "questions"]}


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


# module-level names that are run-time state, not parameters: `lm.STATS` counts forward passes, so hashing it made
# the parameter hash (and with it every world key and cache key) change after each LM call - a world could not find
# its own recorded vectors again and regeneration diverged (found by the integration of the population, wave 3)
RUNTIME_STATE = frozenset({"STATS", "COUNTS"})


def _constants(modname: str) -> dict:
    mod = importlib.import_module(modname)
    return {k: v for k, v in sorted(vars(mod).items())
            if k.isupper() and not k.startswith("_") and k not in RUNTIME_STATE and not callable(v) and _jsonable(v)}


def _encoder_params() -> dict:
    from ..encoders import REGISTRY
    out = {}
    for m, enc in sorted(REGISTRY.items()):
        attrs = {}
        for k in sorted(set(dir(type(enc))) | set(vars(enc))):
            if k.startswith("_") or k in ("batch_size",):
                continue
            v = getattr(enc, k, None)
            if not callable(v) and _jsonable(v):
                attrs[k] = v
        out[m] = attrs
    return out


def preprocessing() -> dict:
    """Every parameter value that determines a representation, read live: ``params`` (module constants, encoder
    attributes, the LM identity) and ``machine`` (device class, LM dtype, library versions)."""
    import PIL
    import torch
    import transformers
    from ..encoders.models import device as enc_device
    from .lm import identity
    lm = identity()
    return {"params": {"modules": {m.rsplit(".", 1)[1]: _constants(m) for m in PARAM_MODULES},
                       "encoders": _encoder_params(), "lm": {k: lm[k] for k in ("model_id", "revision")}},
            "machine": {"encoder_device_class": enc_device().split(":")[0], "lm_device_class": lm["device_class"],
                        "lm_dtype": lm["dtype"],
                        "libs": {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__,
                                 "numpy": np.__version__, "pillow": PIL.__version__}}}


def _h(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def preprocessing_hashes() -> tuple[str, str]:
    """(params hash, machine hash) of `preprocessing()`."""
    p = preprocessing()
    return _h(p["params"]), _h(p["machine"])


def world_key(ref: str, modality: str | None = None, params_hash: str | None = None) -> str:
    """Key of a representation in a world's record: artifact, space id and version, parameter hash, modality. It
    has no machine part, so a world regenerated on another device finds its recorded vectors."""
    if not isinstance(ref, str) or not ref.startswith("sha256:") or len(ref) != 71:
        raise ValueError(f"not an artifact reference: {str(ref)[:80]!r}")
    return f"{ref}|{SPACE_ID}@{SPACE_VERSION}|{params_hash or preprocessing_hashes()[0]}|{modality or 'auto'}"


def cache_key(ref: str, modality: str | None = None, hashes: tuple[str, str] | None = None) -> str:
    """Key of a representation in the machine-wide cache: the world key plus the machine hash (device class, dtype,
    library versions)."""
    ph, mh = hashes or preprocessing_hashes()
    return world_key(ref, modality, ph) + "|" + mh


def ref_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ============================================================================================== core
def _split(x) -> tuple[Any, Any]:
    if hasattr(x, "emit") and hasattr(x, "world") and hasattr(x, "reserve"):
        return x.world, x
    return x, None


def _bytes(world, ref: str) -> bytes:
    if world is None:
        raise ValueError(f"no bytes for {ref}: pass data= or a world with the artifact store")
    from ...substrate import artifacts as art
    return art.get(world, ref)


_MODALITY: dict[str, str] = {}


def _modality(world, ref: str, data: bytes | None) -> str:
    """The modality of an artifact, detected from its bytes once per content address."""
    m = _MODALITY.get(ref)
    if m is None:
        from .compute import detect
        b = data if data is not None else _bytes(world, ref)
        m = detect(b)
        with _LOCK:
            if len(_MODALITY) > 500_000:
                _MODALITY.clear()
            _MODALITY[ref] = m
    return m


def _compute(datas: list[bytes], mods: list[str | None]) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    from . import compute
    ms = [m or compute.detect(d) for d, m in zip(datas, mods)]
    reps = compute.reps(datas, ms)
    with _LOCK:
        for m in ms:
            COUNTS["computed"][m] += 1
    return [(r.vec, r.chunks, r.features) for r in reps]


def _record_entries(ctx, keys: list[str], refs: list[str], compute_fn) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Run `compute_fn() -> entries` once in the world's record and return the entries as recorded (vec float32,
    chunks float16-rounded, features JSON)."""
    from ..encoders.base import recorded_json

    def run() -> dict:
        items = []
        for vec, chunks, feats in compute_fn():
            items.append({"vec": _store.b64(vec), "dim": int(vec.size), "chunks": _store.b64(chunks, "<f2"),
                          "k": int(chunks.shape[0]), "features": _store.clean_features(feats)})
        return {"items": items}
    out = recorded_json(ctx, "space.items", SPACE_ID, SPACE_VERSION, {"keys": keys}, run)
    with _LOCK:
        COUNTS["recorded_batches"] += 1
    ents = []
    for it in out["items"]:
        d = int(it["dim"])
        ents.append((_store.unb64(it["vec"], (d,)), _store.unb64(it["chunks"], (int(it["k"]), d), "<f2"),
                     it["features"]))
    if "space" in ctx.world.modules:
        ctx.emit("space.recorded", {"items": [
            {"key": k, "ref": r, "modality": it["features"].get("modality"), "space": SPACE_ID + "@" + SPACE_VERSION,
             "dim": int(it["dim"]), "vec": it["vec"], "chunks": it["chunks"],
             "sha256": hashlib.sha256(base64.b64decode(it["vec"])).hexdigest(), "features": it["features"]}
            for k, r, it in zip(keys, refs, out["items"])]}, vis="internal")
    return ents


def reps(world_or_ctx, refs: Sequence[str], *, modalities: Sequence[str | None] | None = None, ctx=None,
         data: Sequence[bytes | None] | None = None) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """(vector, chunk vectors, features) per artifact: world record, then cache, then the models (batched)."""
    world, c0 = _split(world_or_ctx)
    ctx = ctx or c0
    if ctx is not None and world is None:
        world = ctx.world
    refs = list(refs)
    mods = list(modalities) if modalities is not None else [None] * len(refs)
    if data is not None and len(data) != len(refs):
        raise ValueError("data must align with refs")
    mods = [m or _modality(world, r, data[i] if data is not None else None) for i, (r, m) in enumerate(zip(refs, mods))]
    hashes = preprocessing_hashes()
    wkeys = [world_key(r, m, hashes[0]) for r, m in zip(refs, mods)]
    keys = [k + "|" + hashes[1] for k in wkeys]
    out: list = [None] * len(refs)
    cache = _store.default_cache()
    missing: dict[str, list[int]] = {}
    for i, (wk, k) in enumerate(zip(wkeys, keys)):
        e = _store.world_entry(world, wk)
        if e is not None:
            out[i] = e
            with _LOCK:
                COUNTS["world_hits"] += 1
            continue
        if ctx is None:
            e = cache.get(k)
            if e is not None:
                out[i] = e
                with _LOCK:
                    COUNTS["cache_hits"] += 1
                continue
        missing.setdefault(k, []).append(i)
    if missing:
        uniq = list(missing)
        first = [missing[k][0] for k in uniq]

        def compute_fn():
            got: list = [None] * len(uniq)
            todo, tb, tm = [], [], []
            for j, i in enumerate(first):
                e = cache.get(keys[i]) if ctx is not None else None     # without ctx the cache was already read
                if e is not None:
                    got[j] = e
                    continue
                b = data[i] if data is not None and data[i] is not None else _bytes(world, refs[i])
                if ref_of(b) != refs[i]:
                    raise ValueError(f"bytes do not match {refs[i]}")
                todo.append(j)
                tb.append(b)
                tm.append(mods[i])
            if todo:
                for j, e in zip(todo, _compute(tb, tm)):
                    got[j] = cache.put(keys[first[j]], *e)
            return got
        ents = _record_entries(ctx, [wkeys[i] for i in first], [refs[i] for i in first], compute_fn) \
            if ctx is not None else compute_fn()
        for k, e in zip(uniq, ents):
            for i in missing[k]:
                out[i] = e
    return out


def item_vectors(world_or_ctx, refs: Sequence[str], *, modalities=None, ctx=None, data=None) -> np.ndarray:
    rs = reps(world_or_ctx, refs, modalities=modalities, ctx=ctx, data=data)
    m = np.stack([r[0] for r in rs]).astype(np.float32) if rs else np.zeros((0, DIM), np.float32)
    m.setflags(write=False)
    return m


def item_vector(world_or_ctx, ref: str, *, modality: str | None = None, ctx=None, data: bytes | None = None) -> np.ndarray:
    return item_vectors(world_or_ctx, [ref], modalities=[modality], ctx=ctx,
                        data=[data] if data is not None else None)[0]


def item_chunks(world_or_ctx, ref: str, *, modality: str | None = None, ctx=None, data: bytes | None = None) -> np.ndarray:
    return reps(world_or_ctx, [ref], modalities=[modality], ctx=ctx, data=[data] if data is not None else None)[0][1]


def item_features(world_or_ctx, ref: str, *, modality: str | None = None, ctx=None,
                  data: bytes | None = None) -> dict:
    return dict(reps(world_or_ctx, [ref], modalities=[modality], ctx=ctx,
                     data=[data] if data is not None else None)[0][2])


def text_vectors(world_or_ctx, texts: Sequence[str], *, ctx=None) -> np.ndarray:
    datas = [str(t).encode("utf-8") for t in texts]
    return item_vectors(world_or_ctx, [ref_of(d) for d in datas], modalities=["text"] * len(datas), ctx=ctx,
                        data=datas)


def text_vector(world_or_ctx, text: str, *, ctx=None) -> np.ndarray:
    return text_vectors(world_or_ctx, [text], ctx=ctx)[0]


def vectors_of_bytes(datas: Sequence[bytes], modalities: Sequence[str | None] | None = None) -> np.ndarray:
    return item_vectors(None, [ref_of(d) for d in datas], modalities=modalities, data=list(datas))


def features_of_bytes(datas: Sequence[bytes], modalities: Sequence[str | None] | None = None) -> list[dict]:
    return [dict(r[2]) for r in reps(None, [ref_of(d) for d in datas], modalities=modalities, data=list(datas))]


def reps_of_bytes(datas: Sequence[bytes], modalities: Sequence[str | None] | None = None):
    return reps(None, [ref_of(d) for d in datas], modalities=modalities, data=list(datas))


def interest_prior(world_or_ctx, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, DIM) float32 unit interest vectors drawn from real reading histories (MIND train users, `prior`). With a
    ctx, the prior file's SHA-256 is recorded so regeneration fails loudly if the prior changed."""
    from . import prior
    world, ctx = _split(world_or_ctx)
    vecs, info = prior.draw(SPACE_VERSION, rng, int(n))
    if ctx is not None:
        rec = ctx.external("space.prior", lambda: {"profiles_sha256": info["profiles_sha256"], "n": int(n),
                                                   "space": SPACE_ID + "@" + SPACE_VERSION})
        if rec["profiles_sha256"] != info["profiles_sha256"]:
            from ...core.errors import ProtocolError
            raise ProtocolError("the interest prior differs from the one this world was generated with")
    vecs.setflags(write=False)
    return vecs


def click_calibration() -> dict:
    """Held-out logistic calibration of P(click | cos(interest, item)) on MIND dev impressions (see `prior`)."""
    from . import prior
    return dict(prior.load(SPACE_VERSION)["meta"]["calibration"])


def comprehension(features: dict) -> float:
    """Probability-like index in [0, 1] that a reader can extract meaning from a text artifact (or the speech or
    subtitles of an audio/video artifact), from its intrinsic features relative to real professional text: the LM
    log-likelihood of its novel tokens as a z-score against MIND news (`prior` text norms), shifted by two standard
    deviations and passed through the normal CDF; times the share of valid words (relative to 0.8); times the share
    of tokens that are new information (not completing a 4-gram already seen, relative to 0.8). Gibberish, word
    salad and keyword stuffing score low; ordinary prose about 1. Artifacts without language (pictures, music)
    return 1.0: their value is not limited by comprehension of text. A modelling assumption (functional form),
    with data-derived scale; registered in the fidelity register by its consumers."""
    from . import prior
    m = features.get("modality")
    novel = features.get("lm_novel_token_share")
    if m in ("text", None):
        lp, val = features.get("lm_novel_logprob_per_token"), features.get("language_validity")
    elif m in ("audio", "video") and features.get("transcript_lm_logprob_per_token") is not None:
        lp, val, novel = features.get("transcript_lm_logprob_per_token"), 1.0, None
    else:
        return 1.0
    if lp is None or not np.isfinite(lp):
        return 0.0
    norms = prior.load(SPACE_VERSION)["meta"]["text_norms"]
    z = (float(lp) - norms["lm_novel_logprob_mean"]) / max(1e-6, norms["lm_novel_logprob_sd"])
    q = 0.5 * (1.0 + math.erf((z + 2.0) / math.sqrt(2.0)))
    q *= min(1.0, max(0.0, float(val if val is not None else 1.0)) / 0.8)
    if novel is not None and np.isfinite(novel):
        q *= min(1.0, max(0.0, float(novel)) / 0.8)
    return float(q)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))


def stats() -> dict:
    from . import lm
    with _LOCK:
        return {"computed": dict(COUNTS["computed"]), "world_hits": COUNTS["world_hits"],
                "cache_hits": COUNTS["cache_hits"], "recorded_batches": COUNTS["recorded_batches"],
                "cache": dict(_store.default_cache().stats), "lm": dict(lm._STATS)}


__all__ = ["DIM", "SPACE_ID", "SPACE_VERSION", "TOPIC", "AFFECT", "QUANT", "MODALITIES", "item_vector",
           "item_vectors", "item_chunks", "item_features", "text_vector", "text_vectors", "interest_prior",
           "vectors_of_bytes", "features_of_bytes", "reps", "reps_of_bytes", "cache_key", "ref_of", "layout",
           "preprocessing", "world_key", "click_calibration", "comprehension", "similarity", "stats"]
