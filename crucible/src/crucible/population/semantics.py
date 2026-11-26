"""The shared semantic space as the population sees it (ONE_WORLD_CONTRACTS section 2, D-16).

This module implements exactly the section-2 signatures (`DIM`, `SPACE_ID`, `SPACE_VERSION`, `item_vector`,
`item_vectors`, `text_vector`, `item_features`, `interest_prior`) on the EXISTING pretrained text encoder of
`crucible.semantic` (MiniLM-L6-v2, real model execution, disk-cached by artifact hash and encoder version). It is the
bridge workstream CORE-SEM replaces with `crucible.semantic.space` at merge (orchestrator: repoint the imports of
`population.semantics` to `semantic.space`; the extra `ctx=` keyword stays optional there or is dropped).

Recording. A vector computed inside a transaction goes through `crucible.substrate.inference.recorded` (kind
`space.text`) and is stored in this world by kernel module `space` (table `space_keys`, array `space.vec`), so
replay never reruns the encoder and regeneration returns the recorded matrix; a vector already recorded in the world
is returned from the world, so the event stream never depends on a process cache. Without `ctx` (evaluators,
offline analysis) nothing is recorded.

Interests. `interest_prior` draws initial interests from embeddings of a real topic corpus (the 93 MIND-small news
subcategories, `space_asset.py`): a popularity-weighted mixture of 1-3 topic centroids plus individual variation.
The population stores interests as K coordinates in the corpus' principal subspace of the space (`basis(K)`,
orthonormal rows), so `lift(c) = c @ B` is a unit vector of the full space and, for any item vector e of the space,
`c . (B e) = lift(c) . e` exactly: affinities computed on projected item coordinates are the full-space cosines.

Modalities. Text, HTML, markdown and other text-bearing artifacts are encoded from their bytes. The bridge has no
aligned image/audio/video map (CORE-SEM fits those); `item_vector` raises `EncodingError` for such artifacts and the
caller distributes them by the text the viewer sees with them (title and description), which is labelled in their
item record (`rep: "caption"`).

Artifact-intrinsic features (`item_features`) are pure functions of the bytes (no model): word counts, distinct-word
share, English function-word share, share of words the encoder's WordPiece vocabulary knows as whole words,
compressibility (zlib), sentence statistics and reading duration. Nothing in them is declared by a creator.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import os
import re
import zlib

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, register_module
from ..substrate import artifacts, bus

DIM = 384
SPACE_ID = "crucible.pop-space/minilm-l6-v2+mind93"
SPACE_VERSION = "1"
ASSET = "interest_space_v1"
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WORDS_PER_S = 4.0
TABLE = "space_keys"
ARRAY = "space.vec"


class EncodingError(ValueError):
    """The artifact carries no text the bridge can encode (images, audio, video without CORE-SEM's aligned maps)."""


# ============================================================================================ the asset
@functools.lru_cache(maxsize=1)
def asset() -> dict:
    z = np.load(os.path.join(_HERE, ASSET + ".npz"))
    with open(os.path.join(_HERE, ASSET + ".json")) as f:
        meta = json.load(f)
    out = {k: np.asarray(z[k]) for k in z.files}
    out.update(meta)
    if out["centroids"].shape[1] != DIM or out["basis"].shape[1] != DIM:
        raise ProtocolError("interest space asset has the wrong dimension")
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


@functools.lru_cache(maxsize=8)
def basis(k: int) -> np.ndarray:
    """[k, DIM] orthonormal rows spanning the topic corpus' principal subspace (float32, read-only)."""
    k = int(k)
    if not 1 <= k <= asset()["basis"].shape[0]:
        raise ProtocolError(f"interest subspace dimension must be in [1, {asset()['basis'].shape[0]}]")
    q, _ = np.linalg.qr(asset()["basis"][:k].astype(np.float64).T)      # re-orthonormalise in float64
    b = q.T.astype(np.float32)
    s = np.sign(np.sum(b * asset()["basis"][:k], axis=1, keepdims=True))
    b = (b * np.where(s == 0, 1, s)).astype(np.float32)
    b.setflags(write=False)
    return b


def project(vecs: np.ndarray, k: int) -> np.ndarray:
    """Coordinates of full-space vectors in the k-dim subspace: [n, k] float32 (not renormalised)."""
    v = np.asarray(vecs, dtype=np.float32)
    return (v @ basis(k).T).astype(np.float32)


def lift(coords: np.ndarray) -> np.ndarray:
    """Unit full-space vectors [n, DIM] of subspace coordinates [n, k]."""
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


# ============================================================================================ interests
def interest_prior(world, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, DIM) float32 unit initial interests from the embeddings of a real topic corpus (contract section 2)."""
    return lift(interest_prior_coords(rng, n, asset()["basis"].shape[0]))


def interest_prior_coords(rng: np.random.Generator, n: int, k: int, *, noise: float = 0.35,
                          concentration: float = 0.7) -> np.ndarray:
    """(n, k) unit subspace coordinates: popularity-weighted mixture of 1-3 topic centroids + individual variation
    (isotropic in the subspace). Vectorised (works for 10M rows in chunks)."""
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


# ============================================================================================ encoding
def _enc():
    from ..semantic.encoders import encoder
    return encoder("text")


@functools.lru_cache(maxsize=1)
def _enc_tag() -> str:
    e = _enc()
    return f"{e.id}@{e.version()}|{e.preprocessing_hash()}|{SPACE_ID}@{SPACE_VERSION}"


def text_key(text: str | bytes) -> str:
    b = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return "t:" + hashlib.sha256(b).hexdigest() + "|" + _enc_tag()


def ref_key(ref: str) -> str:
    artifacts.parse_ref(ref)
    return "a:" + ref + "|" + _enc_tag()


def _world_rows(world, keys: list[str]) -> list[int | None]:
    tbl = world.state.tables.get(TABLE)
    if tbl is None:
        return [None] * len(keys)
    return [None if (r := tbl.get(k)) is None else int(r["row"]) for k in keys]


def _compute(datas: list[bytes]) -> np.ndarray:
    from ..semantic.encoders import encode_bytes_many
    m = np.asarray(encode_bytes_many(list(datas), "text"), dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _vectors(world, keys: list[str], datas_of, ctx: Ctx | None) -> np.ndarray:
    out = np.zeros((len(keys), DIM), np.float32)
    rows = _world_rows(world, keys)
    miss: dict[str, list[int]] = {}
    arr = world.state.arrays.get(ARRAY)
    for j, (k, r) in enumerate(zip(keys, rows)):
        if r is not None and arr is not None:
            out[j] = arr[r].astype(np.float32)
        else:
            miss.setdefault(k, []).append(j)
    if not miss:
        return out
    uk = list(miss)
    datas = [datas_of(miss[k][0]) for k in uk]
    if ctx is None:
        mat = _compute(datas)
    else:
        if "space" not in ctx.world.modules:
            raise ProtocolError("recording vectors needs the kernel module 'space' (crucible.population.semantics)")
        from ..substrate import inference
        e = _enc()
        mat = inference.recorded(ctx, "space.text", e.id, e.model_hash(),
                                 {"keys": uk, "space": SPACE_ID, "version": SPACE_VERSION},
                                 lambda: np.ascontiguousarray(_compute(datas), dtype=np.float32))
        mat = np.asarray(mat, dtype=np.float32)
        ref = artifacts.put_array_blob(ctx, {"vec": mat.astype(np.float16)})
        ctx.emit("space.recorded", {"keys": uk, "ref": ref}, vis="internal")
    for i, k in enumerate(uk):
        for j in miss[k]:
            out[j] = mat[i]
    return out


def text_vectors(world, texts: list[str], *, ctx: Ctx | None = None) -> np.ndarray:
    """[n, DIM] unit vectors of texts (queries, needs, interests, descriptions, page text)."""
    texts = [str(t) for t in texts]
    keys = [text_key(t) for t in texts]
    return _vectors(world, keys, lambda j: texts[j].encode("utf-8"), ctx)


def text_vector(world, text: str, *, ctx: Ctx | None = None) -> np.ndarray:
    return text_vectors(world, [text], ctx=ctx)[0]


_TEXT_MIMES = ("text/", "application/json", "application/xml", "application/javascript", "image/svg+xml")


def _text_bytes(world, ref: str) -> bytes:
    a = artifacts.info(world, ref)
    mime = str(a.get("mime", ""))
    if not mime.startswith(_TEXT_MIMES):
        raise EncodingError(f"{mime} artifacts need CORE-SEM's aligned encoders (this bridge encodes text)")
    return artifacts.get(world, ref)


def item_vectors(world, refs: list[str], *, ctx: Ctx | None = None) -> np.ndarray:
    """[n, DIM] unit vectors of text-bearing artifacts (EncodingError for other modalities)."""
    refs = list(refs)
    keys = [ref_key(r) for r in refs]
    return _vectors(world, keys, lambda j: _text_bytes(world, refs[j]), ctx)


def item_vector(world, ref: str, *, ctx: Ctx | None = None) -> np.ndarray:
    return item_vectors(world, [ref], ctx=ctx)[0]


# ============================================================================================ intrinsic features
FUNCTION_WORDS = frozenset("""a an the and or but if then so because as of in on at by for with about against between
into through during before after above below to from up down out over under again further once here there when where
why how all any both each few more most other some such no nor not only own same than too very can will just should
now is are was were be been being have has had having do does did doing i me my we our you your he him his she her
it its they them their what which who whom this that these those am would could may might must shall""".split())


@functools.lru_cache(maxsize=1)
def _vocab() -> frozenset:
    """Whole-word entries of the text encoder's WordPiece vocabulary (the tokenizer's own vocab file)."""
    from ..semantic.encoders import models
    path = os.path.join(models.snapshot_dir("text"), "vocab.txt")
    if not os.path.exists(path):
        raise ProtocolError("the text encoder's vocabulary file is not in the local model cache")
    with open(path, encoding="utf-8") as f:
        words = [w.strip() for w in f]
    return frozenset(w for w in words if w.isalpha() and not w.startswith("##") and len(w) > 1)


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+")
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def text_features(text: str) -> dict[str, float]:
    """Intrinsic features of a text (see the module docstring)."""
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


def item_features(world, ref: str) -> dict[str, float]:
    """Artifact-intrinsic features computed from the bytes (contract section 2)."""
    a = artifacts.info(world, ref)
    mime = str(a.get("mime", ""))
    out = {"modality": mime.split("/", 1)[0], "bytes": float(a.get("size", 0))}
    if mime.startswith(_TEXT_MIMES):
        raw = artifacts.get(world, ref).decode("utf-8", "replace")
        if "html" in mime:
            raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
            raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        out.update(text_features(raw))
        out["legibility"] = legibility(out)
    return out


# ============================================================================================ kernel module
bus.event_type("space.recorded", S.obj({"keys": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                        "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern}},
                                       ["keys", "ref"]), owner="space",
               doc="vectors of the shared space computed by the recorded text encoder (array blob `ref`)")


@register_module
class Space(Module):
    """Owner of the vectors of the shared space recorded in this world (table `space_keys`, array `space.vec`).
    No tools: a participant reads content, never the platform's representation of it."""
    name = "space"
    host = "space"
    version = "1"
    depends = ("artifacts",)

    def install(self, world):
        world.state.declare_table(TABLE)
        if ARRAY not in world.state.arrays:
            world.state.set_array(ARRAY, np.zeros((0, DIM), np.float16))
        bus.enforce(world, owner="space")

    @applier("space.recorded")
    def ap_recorded(self, ev):
        p = ev.payload
        st = self.world.state
        try:
            m = artifacts.get_array_blob(self.world, p["ref"])["vec"]
        except NotFound as e:
            raise InvalidAction(f"space.recorded seq {ev.seq}: {e}") from None
        if m.shape != (len(p["keys"]), DIM):
            raise InvalidAction("space.recorded: blob shape does not match the keys")
        a = st.arrays[ARRAY]
        n0 = int(st.meta.get("space.n", 0))
        need = n0 + m.shape[0]
        if need > a.shape[0]:
            grown = np.zeros((max(need, 2 * a.shape[0], 64), DIM), np.float16)
            grown[:n0] = a[:n0]
            st.set_array(ARRAY, grown)
        st.array_rows_mut(ARRAY, np.arange(n0, need))[n0:need] = m.astype(np.float16)
        st.meta_set("space.n", need)
        for j, k in enumerate(p["keys"]):
            if st.has(TABLE, k):
                raise InvalidAction(f"space.recorded: {k[:40]} recorded twice")
            st.insert(TABLE, k, {"row": n0 + j})

    def check_invariants(self, world):
        n = int(world.state.meta.get("space.n", 0))
        if len(world.state.tables[TABLE]) != n:
            return [f"space: {len(world.state.tables[TABLE])} keys for {n} vectors"]
        return []


# ============================================================================================ query vocabulary
def vocab() -> list[str]:
    return list(asset()["vocab"])


def vocab_coords(k: int) -> np.ndarray:
    return asset()["vocab_coords"][:, :int(k)]


def query_words(coords: np.ndarray, rng: np.random.Generator, tau, k: int = 2) -> list[str]:
    """For each individual, `k` words of the query vocabulary whose embeddings best match its interests (Gumbel
    noise of scale `tau`: less skilled searchers pick less fitting words). One query string per individual."""
    c = np.asarray(coords, np.float32)
    V = vocab_coords(c.shape[1]).astype(np.float32)
    s = c @ V.T
    g = rng.gumbel(size=s.shape).astype(np.float32) * np.asarray(tau, np.float32).reshape(-1, 1)
    top = np.argsort(-(s + g), axis=1, kind="stable")[:, :k]
    words = vocab()
    return [" ".join(sorted(words[j] for j in row)) for row in top]
