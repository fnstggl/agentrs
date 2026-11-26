"""A real, deterministic text representation of content artifacts (workstream C2).

Until workstream D's pretrained encoders are merged, the consumption engine represents a text artifact by a
*random-indexing* embedding computed from its bytes: every word type gets a fixed pseudo-random Gaussian vector
(seeded by the word's BLAKE2b hash), and a document is the L2-normalised sum of the vectors of its non-stopword
tokens weighted by `1 + log(tf)`. This is a Johnson-Lindenstrauss random projection of the document's weighted
bag of words, so cosine similarity between two representations approximates the cosine similarity of their word
distributions: documents that share words are close, documents that do not are nearly orthogonal.

It is lexical, not semantic (synonyms are not close), and it is labelled so (`ENCODER`, FIDELITY_REGISTER). The
same space holds individuals' interests (mixtures of word vectors), so affinity is a cosine in one shared space.
Representations are computed once per artifact (at indexing) and stored with the item; they are never recomputed per
viewer. Nothing here reads a creator-supplied score: the only input is the artifact's bytes.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from .vocab import STOPWORDS, TOPICS

DIM = 32
TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")
ENCODER = {"id": "c2-random-indexing", "version": "1", "dim": DIM, "modality": "text",
           "kind": "lexical random projection (not a pretrained semantic encoder)"}
ENCODER["preprocessing_hash"] = hashlib.sha256(
    f"{TOKEN.pattern}|{sorted(STOPWORDS)}|{DIM}|1+log(tf)|l2".encode()).hexdigest()[:16]
MAX_TERMS = 64          # terms kept per item for search


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@lru_cache(maxsize=1 << 16)
def word_vector(word: str, dim: int = DIM) -> np.ndarray:
    """The fixed random vector of one word type (unit expected norm)."""
    seed = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")
    v = np.random.Generator(np.random.PCG64(seed)).standard_normal(dim) / math.sqrt(dim)
    v.flags.writeable = False
    return v


def term_counts(text: str) -> Counter:
    return Counter(t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1)


def embed_counts(counts: Counter, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    for w, c in counts.items():
        v += (1.0 + math.log(c)) * word_vector(w, dim)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def embed_text(text: str, dim: int = DIM) -> np.ndarray:
    """Unit-norm float64 representation of a text (zero vector for a text without content words)."""
    return embed_counts(term_counts(text), dim)


def encode_bytes(data: bytes, dim: int = DIM) -> tuple[np.ndarray, dict[str, int], int]:
    """(embedding, top terms {word: count}, number of words) of UTF-8 text bytes."""
    text = data.decode("utf-8", errors="replace")
    counts = term_counts(text)
    top = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_TERMS])
    return embed_counts(counts, dim), top, len(tokenize(text))


@lru_cache(maxsize=4)
def topic_centroids(dim: int = DIM) -> np.ndarray:
    """[n_topics, dim] unit vectors: the representation of each topic's whole vocabulary (used only to seed the
    population's interests in the same space as content; never read for any piece of content)."""
    out = np.stack([embed_counts(Counter(words), dim) for words in TOPICS.values()])
    out.flags.writeable = False
    return out


def normalise_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)
