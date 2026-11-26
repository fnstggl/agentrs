"""The shared semantic space as the population uses it (ONE_WORLD_CONTRACTS section 2, D-16).

There is ONE space: `crucible.semantic.space` (workstream CORE-SEM; DIM = 384 topic + 4 affect + 34 quantity). This
module computes no representation of its own. `DIM`, `SPACE_ID`, `SPACE_VERSION`, `item_vector(s)`, `text_vector(s)`,
`item_features` and `interest_prior` are the space's, called through its recorded path: pass the handler's `ctx` (as
`ctx=` or as the first argument) inside a command, process or population activity and the representation is recorded
in the world (`substrate.inference.recorded` + kernel module `space`, table `space_items`), so replay and
regeneration never run an encoder; with a bare world the world's records are read and anything missing is computed
(cached on disk) but not recorded. The kernel module `space` is the space's (`crucible.semantic.space.store`); this
module registers none.

On top of the space the population keeps thin, model-free helpers:

* **Topic subspace.** Individuals' interests are stored as K coordinates (K = the population profile's interest
  dimension, at most 64) in the principal subspace of a real topic corpus inside the space's TOPIC block (the
  all-MiniLM-L6-v2 block; `asset()`): 93 MIND-small news subcategories, each item represented by the space itself
  (`semantic.space.vectors_of_bytes`, `space_asset.py`). `basis(k)` is [k, DIM] with orthonormal rows that are zero
  outside the topic block, `lift(c) = unit(c @ basis(k))` is a unit vector of the space, and
  `project(e, k) = basis(k) @ topic(e)` with `topic(e)` the unit-normalised topic block of a space vector. Hence for
  a unit coordinate row c and any space vector e, `c . project(e) = lift(c) . topic(e)`: an affinity on projected
  coordinates is the cosine of the interest with the item's topic block, computed in the full space. The topic block
  is the part of the space in which reading interest is measured best (the space's own MIND validation: click AUC
  0.646 for the topic block vs 0.638 for the whole vector), it is the MiniLM cosine workstream E's fitted click model
  `news_click_semantic@1` takes as input (`affinity_space: minilm-l6-cosine`), and it keeps the affect and quantity
  blocks from scaling affinities (an interest has no sentiment, so a whole-vector cosine would penalise items with
  strong valence or with numbers).
* **Topics.** `topic_names`, `topic_words`, `popularity`, `topic_coords`, `nearest_topic` and the real abstracts
  population creators retell (`corpus`).
* **Initial interests.** `interest_prior_coords`: a popularity-weighted mixture of 1-3 topic centroids plus
  individual variation (CORE-POP's model, on the space's topic centroids). `interest_prior` is the space's own prior
  (real MIND reading histories) for callers that want full-space initial interests.
* **Query words** of individuals' searches (`query_words`) from the space vectors of a fixed query vocabulary.
* **Lexical reader model** (`text_features`, `legibility`): pure functions of the text the space decodes
  (`encoders.text.to_text`) and of the space's word list (the MiniLM WordPiece vocabulary, `compute.TEXT_VOCAB`).
  `item_rep` / `item_features` add `legibility` to the space's intrinsic features of a text artifact.

Undecodable media (the space's `MediaError`, e.g. a corrupt image) raise `EncodingError`; callers then represent the
item by the text a viewer sees with it (title, description) and label that (`rep: "caption"`).
"""
from __future__ import annotations

import functools
import gzip
import json
import math
import os
import re
import zlib

import numpy as np

from ..core.errors import ProtocolError
from ..semantic import space as _space
from ..semantic.encoders import EncodingError as _EncoderError
from ..semantic.space.compute import TOPIC, TOPIC_DIM

DIM = _space.DIM
SPACE_ID = _space.SPACE_ID
SPACE_VERSION = _space.SPACE_VERSION
ASSET = "interest_space_v2"
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WORDS_PER_S = 4.0


class EncodingError(_EncoderError):
    """The space cannot represent the artifact from its bytes (undecodable or over-limit media)."""


# ============================================================================================ the asset
@functools.lru_cache(maxsize=1)
def asset() -> dict:
    """The topic asset (`space_asset.py`). Arrays of the topic block are widened to the full space on load:
    `centroids` [T, DIM] and `basis` [K_MAX, DIM] are zero outside the TOPIC block; `centroids_topic` and
    `basis_topic` hold their [*, TOPIC_DIM] blocks."""
    z = np.load(os.path.join(_HERE, ASSET + ".npz"))
    with open(os.path.join(_HERE, ASSET + ".json")) as f:
        meta = json.load(f)
    if meta.get("space_id") != SPACE_ID or meta.get("block") != "topic" or \
            meta.get("block_slice") != [TOPIC.start, TOPIC.stop]:
        raise ProtocolError(f"interest asset {ASSET} was not built on the topic block of {SPACE_ID}")
    out = {k: np.asarray(z[k]) for k in z.files}
    out.update(meta)
    if out["centroids_topic"].shape[1] != TOPIC_DIM or out["basis_topic"].shape[1] != TOPIC_DIM:
        raise ProtocolError("interest space asset has the wrong dimension")
    for name in ("centroids", "basis"):
        t = out[name + "_topic"].astype(np.float32)
        full = np.zeros((t.shape[0], DIM), np.float32)
        full[:, TOPIC] = t
        full.setflags(write=False)
        out[name] = full
    return out


@functools.lru_cache(maxsize=1)
def corpus() -> list[list[str]]:
    """Real abstracts per topic (same order as `topic_names()`)."""
    with gzip.open(os.path.join(_HERE, ASSET + ".corpus.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    if d["topics"] != asset()["topics"]:
        raise ProtocolError("corpus and asset topics differ")
    return d["abstracts"]


def topic_names() -> list[str]:
    return list(asset()["topics"])


def topic_words(t: int) -> list[str]:
    return list(asset()["words"][asset()["topics"][int(t)]])


def popularity() -> np.ndarray:
    return asset()["popularity"]


# ============================================================================================ topic subspace
@functools.lru_cache(maxsize=8)
def topic_basis(k: int) -> np.ndarray:
    """[k, TOPIC_DIM] orthonormal rows spanning the topic corpus' principal subspace of the topic block (float32)."""
    k = int(k)
    B = asset()["basis_topic"]
    if not 1 <= k <= B.shape[0]:
        raise ProtocolError(f"interest subspace dimension must be in [1, {B.shape[0]}]")
    q, _ = np.linalg.qr(B[:k].astype(np.float64).T)      # re-orthonormalise in float64
    b = q.T.astype(np.float32)
    s = np.sign(np.sum(b * B[:k], axis=1, keepdims=True))
    b = (b * np.where(s == 0, 1, s)).astype(np.float32)
    b.setflags(write=False)
    return b


@functools.lru_cache(maxsize=8)
def basis(k: int) -> np.ndarray:
    """[k, DIM] orthonormal rows of the full space, zero outside the TOPIC block (read-only)."""
    b = np.zeros((int(k), DIM), np.float32)
    b[:, TOPIC] = topic_basis(k)
    b.setflags(write=False)
    return b


def _check(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.shape[-1] != DIM:
        raise ProtocolError(f"space vectors have {DIM} columns, not {v.shape[-1]}")
    return v


def topic(vecs: np.ndarray) -> np.ndarray:
    """Space vectors [n, DIM] -> their unit-normalised topic blocks widened to [n, DIM] (zero elsewhere)."""
    v = _check(vecs)
    out = np.zeros(v.shape, np.float32)
    t = v[..., TOPIC]
    out[..., TOPIC] = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
    return out


def project(vecs: np.ndarray, k: int) -> np.ndarray:
    """Coordinates [n, k] of space vectors' topic directions in the k-dim subspace (not renormalised: the norm is the
    share of the topic direction the subspace captures)."""
    v = _check(vecs)
    t = v[..., TOPIC]
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
    return (t @ topic_basis(k).T).astype(np.float32)


def lift(coords: np.ndarray) -> np.ndarray:
    """Unit space vectors [n, DIM] of subspace coordinates [n, k] (zero outside the topic block)."""
    c = np.asarray(coords, dtype=np.float32)
    u = c @ basis(c.shape[-1])
    return (u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)).astype(np.float32)


@functools.lru_cache(maxsize=8)
def topic_coords(k: int) -> np.ndarray:
    """[T, k] subspace coordinates of the topic centroids."""
    c = project(asset()["centroids"], k)
    c.setflags(write=False)
    return c


def nearest_topic(coords: np.ndarray) -> np.ndarray:
    """Index of the topic centroid most similar to each coordinate row."""
    c = np.asarray(coords, dtype=np.float32)
    return np.argmax(c @ topic_coords(c.shape[-1]).T, axis=-1)


# ============================================================================================ the space (delegated)
def text_vectors(world_or_ctx, texts, *, ctx=None) -> np.ndarray:
    """[n, DIM] unit space vectors of texts (queries, needs, interests, descriptions, page text)."""
    return _space.text_vectors(world_or_ctx, [str(t) for t in texts], ctx=ctx)


def text_vector(world_or_ctx, text: str, *, ctx=None) -> np.ndarray:
    return text_vectors(world_or_ctx, [text], ctx=ctx)[0]


def _reps(world_or_ctx, refs, ctx):
    from ..semantic.media import MediaError
    try:
        return _space.reps(world_or_ctx, list(refs), ctx=ctx)
    except MediaError as e:
        raise EncodingError(str(e)) from e


def item_vectors(world_or_ctx, refs, *, ctx=None) -> np.ndarray:
    """[n, DIM] unit space vectors of artifacts of any modality (EncodingError for undecodable media)."""
    rs = _reps(world_or_ctx, refs, ctx)
    return np.stack([r[0] for r in rs]).astype(np.float32) if rs else np.zeros((0, DIM), np.float32)


def item_vector(world_or_ctx, ref: str, *, ctx=None) -> np.ndarray:
    return item_vectors(world_or_ctx, [ref], ctx=ctx)[0]


def _world_of(world_or_ctx, ctx):
    if ctx is not None:
        return ctx.world
    x = world_or_ctx
    return x.world if hasattr(x, "emit") and hasattr(x, "world") else x


def item_rep(world_or_ctx, ref: str, *, ctx=None) -> tuple[np.ndarray, dict]:
    """(space vector, intrinsic features) of an artifact from ONE (recorded) representation of the space; the
    features of a text artifact gain the population's reader model: `legibility` and its lexical inputs
    (`reader_*`, `text_features` of the text the space decodes)."""
    vec, _, feats = _reps(world_or_ctx, [ref], ctx)[0]
    out = dict(feats)
    if out.get("modality") == "text":
        from ..semantic.encoders.text import to_text
        from ..substrate import artifacts
        f = text_features(to_text(artifacts.get(_world_of(world_or_ctx, ctx), ref)))
        out.update({f"reader_{k}": v for k, v in f.items()})
        out["legibility"] = legibility(f)
    return np.asarray(vec, np.float32), out


def item_features(world_or_ctx, ref: str, *, ctx=None) -> dict:
    """Artifact-intrinsic features computed from the bytes (contract section 2): the space's features, plus the
    population's reader model for text (`item_rep`)."""
    return item_rep(world_or_ctx, ref, ctx=ctx)[1]


def interest_prior(world_or_ctx, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, DIM) float32 unit initial interests of the space (real MIND reading histories; contract section 2)."""
    return _space.interest_prior(world_or_ctx, rng, n)


# ============================================================================================ interests
def interest_prior_coords(rng: np.random.Generator, n: int, k: int, *, noise: float = 0.35,
                          concentration: float = 0.7) -> np.ndarray:
    """(n, k) unit subspace coordinates: popularity-weighted mixture of 1-3 topic centroids of the space plus
    individual variation (isotropic in the subspace). Vectorised (works for 10M rows in chunks)."""
    n = int(n)
    C = topic_coords(k).astype(np.float64)
    T = C.shape[0]
    pw = popularity() ** concentration
    pw = pw / pw.sum()
    m = rng.integers(1, 4, n)                                   # 1..3 topics
    tops = rng.choice(T, size=(n, 3), p=pw)
    w = rng.dirichlet(np.ones(3) * 1.2, size=n)
    w = np.where(np.arange(3)[None, :] < m[:, None], w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    v = np.einsum("nj,njk->nk", w, C[tops])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    v += noise * rng.standard_normal((n, k)) / math.sqrt(k)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return v.astype(np.float32)


# ============================================================================================ lexical reader model
FUNCTION_WORDS = frozenset("""a an the and or but if then so because as of in on at by for with about against between
into through during before after above below to from up down out over under again further once here there when where
why how all any both each few more most other some such no nor not only own same than too very can will just should
now is are was were be been being have has had having do does did doing i me my we our you your he him his she her
it its they them their what which who whom this that these those am would could may might must shall""".split())


@functools.lru_cache(maxsize=1)
def _vocab() -> frozenset:
    """Whole-word entries of the space's text vocabulary (the MiniLM WordPiece vocabulary, `compute.TEXT_VOCAB`)."""
    from ..semantic.space.compute import TEXT_VOCAB
    return frozenset(TEXT_VOCAB())


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+")
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def text_features(text: str) -> dict[str, float]:
    """Lexical features of a text: word counts, distinct-word share, English function-word share, share of words the
    space's vocabulary knows as whole words, compressibility (zlib), sentence statistics, reading duration."""
    toks = _WORD.findall(text)
    low = [t.lower() for t in toks]
    n = len(low)
    alpha = [t for t in low if t.isalpha()]
    voc = _vocab()
    b = text.encode("utf-8")
    sents = max(1, len([s for s in _SENT.split(text) if s.strip()]))
    return {
        "words": float(n), "chars": float(len(text)),
        "distinct_ratio": float(len(set(low)) / n) if n else 0.0,
        "function_ratio": float(sum(t in FUNCTION_WORDS for t in low) / n) if n else 0.0,
        "valid_ratio": float(sum(t in voc for t in alpha) / len(alpha)) if alpha else 0.0,
        "compress_ratio": float(len(zlib.compress(b, 6)) / max(1, len(b))) if b else 1.0,
        "sentences": float(sents), "words_per_sentence": float(n / sents),
        "duration_s": float(max(5.0, n / WORDS_PER_S)),
    }


def legibility(f: dict[str, float]) -> float:
    """How readable a text is for an ordinary reader, in [0, 1], from intrinsic features only: real words of the
    language, natural function-word use, little verbatim repetition. (Fidelity: plausible, unvalidated - a reader
    model, not a quality label: it says nothing about topic or usefulness.)"""
    valid = f.get("valid_ratio", 0.0)
    func = min(1.0, f.get("function_ratio", 0.0) / 0.30)
    distinct = min(1.0, f.get("distinct_ratio", 0.0) / 0.45) if f.get("words", 0) >= 40 else \
        min(1.0, f.get("distinct_ratio", 0.0) / 0.7)
    comp = min(1.0, f.get("compress_ratio", 0.0) / 0.35) if f.get("chars", 0) >= 400 else 1.0
    return float(np.clip(valid ** 2 * (0.25 + 0.75 * func) * distinct * (0.5 + 0.5 * comp), 0.0, 1.0))


# ============================================================================================ query vocabulary
def vocab() -> list[str]:
    return list(asset()["vocab"])


def vocab_coords(k: int) -> np.ndarray:
    return asset()["vocab_coords"][:, :int(k)]


def query_words(coords: np.ndarray, rng: np.random.Generator, tau, k: int = 2) -> list[str]:
    """For each individual, `k` words of the query vocabulary whose space vectors best match its interests (Gumbel
    noise of scale `tau`: less skilled searchers pick less fitting words). One query string per individual."""
    c = np.asarray(coords, np.float32)
    V = vocab_coords(c.shape[1]).astype(np.float32)
    s = c @ V.T
    g = rng.gumbel(size=s.shape).astype(np.float32) * np.asarray(tau, np.float32).reshape(-1, 1)
    top = np.argsort(-(s + g), axis=1, kind="stable")[:, :k]
    words = vocab()
    return [" ".join(sorted(words[j] for j in row)) for row in top]
