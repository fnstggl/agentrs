"""Content representation for the population: what an individual "gets" from a page it reads.

The representation is computed from the actual text of the page (never from a score its author supplied):

* `token_vec(token, dim)` is a deterministic sparse ternary random-index vector (8 non-zero +/-1 entries chosen
  by BLAKE2b of the token) - random indexing (Kanerva et al. 2000; Sahlgren 2005). Texts sharing words get similar
  vectors; texts sharing no words are nearly orthogonal.
* `text_vector(text, dim)` sums the token vectors of the non-stopword tokens with sublinear term frequency
  (1 + log tf) and normalises; it also returns page statistics an individual can perceive: visible characters,
  tokens, the share of distinct tokens (keyword-stuffed pages score low) and the share of stopwords (prose vs
  word lists).
* `TOPICS` is a lexicon of everyday interest clusters. An individual's interests are a mixture of topic vectors
  (`interest_vector`), so they live in the same space as the pages it reads; `query_words` turns interests into
  search terms.

Fidelity: deliberate simplification (bag of words, no synonymy or word order). When workstream D's encoders are
merged, `ENCODER` can be switched to "semantic" (config `encoder`): page text is stored as an artifact and encoded
with `crucible.semantic.encode(world, ref, "text")`, then projected to `dim` with a fixed seeded projection.
"""
from __future__ import annotations

import hashlib
import html as _html
import math
import re
from functools import lru_cache

import numpy as np

TOKEN = re.compile(r"[a-z]+")
TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.I)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
PRICE_RE = re.compile(r"Buy for \$([0-9]+(?:\.[0-9]{1,2})?)")
NNZ = 8

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they this those through
to too under until up very was we were what when where which while who whom why will with you your yours log home
about none yet all back new page""".split())

# everyday interest clusters (topic -> words people use about it); overlaps the themes of the generated sites
TOPICS: dict[str, list[str]] = {
    "cooking": ["recipe", "cooking", "bread", "soup", "kitchen", "baking", "dinner", "salad", "dessert", "spices"],
    "cycling": ["bike", "cycling", "road", "cargo", "wheel", "ride", "repair", "helmet", "tyre", "commute"],
    "gardening": ["garden", "plot", "plants", "soil", "seeds", "greenhouse", "compost", "flowers", "harvest", "sunny"],
    "coding": ["code", "coding", "python", "software", "program", "bug", "library", "api", "developer", "data"],
    "music": ["music", "song", "guitar", "album", "concert", "band", "piano", "melody", "playlist", "singer"],
    "fitness": ["fitness", "running", "workout", "gym", "training", "health", "yoga", "strength", "diet", "sleep"],
    "travel": ["travel", "trip", "hotel", "flight", "city", "beach", "map", "visa", "hostel", "museum"],
    "finance": ["money", "savings", "budget", "loan", "invest", "tax", "bank", "price", "income", "rent"],
    "games": ["game", "games", "player", "level", "puzzle", "board", "chess", "strategy", "score", "team"],
    "learning": ["study", "learning", "course", "math", "history", "languages", "lesson", "exam", "teacher", "notes"],
    "home": ["room", "rental", "furniture", "lamp", "repair", "tools", "drill", "ladder", "cleaning", "paint"],
    "local": ["local", "events", "market", "meetup", "workshop", "talk", "neighbours", "produce", "farm", "honey"],
    "pets": ["pet", "dog", "cat", "vet", "food", "walk", "puppy", "kitten", "toys", "training"],
    "work": ["job", "gig", "freelance", "writing", "design", "support", "career", "office", "contract", "skills"],
    "science": ["science", "research", "experiment", "physics", "biology", "space", "chemistry", "lab", "theory",
                "climate"],
    "fashion": ["fashion", "clothes", "style", "shoes", "jacket", "vintage", "dress", "colour", "fabric", "shop"],
}
TOPIC_NAMES = tuple(TOPICS)
LEXICON = tuple(sorted({w for ws in TOPICS.values() for w in ws}))


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(str(text).lower()) if 1 < len(t) <= 30]


@lru_cache(maxsize=200_000)
def _tv(token: str, dim: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    h = hashlib.blake2b(f"{dim}|{token}".encode(), digest_size=32).digest()
    idx, sgn, seen = [], [], set()
    for k in range(0, 32, 2):
        j = int.from_bytes(h[k:k + 2], "little") % dim
        if j in seen:
            continue
        seen.add(j)
        idx.append(j)
        sgn.append(1.0 if h[k] & 1 else -1.0)
        if len(idx) == min(NNZ, dim):
            break
    return tuple(idx), tuple(sgn)


def token_vec(token: str, dim: int) -> np.ndarray:
    idx, sgn = _tv(token, int(dim))
    v = np.zeros(dim, np.float32)
    v[list(idx)] = sgn
    return v


def text_vector(text: str, dim: int) -> tuple[np.ndarray, dict]:
    """(unit vector float32[dim], {chars, tokens, uniq, prose}) of a text."""
    toks = tokens(text)
    content = [t for t in toks if t not in STOPWORDS]
    v = np.zeros(dim, np.float64)
    tf: dict[str, int] = {}
    for t in content:
        tf[t] = tf.get(t, 0) + 1
    for t in sorted(tf):
        idx, sgn = _tv(t, int(dim))
        w = 1.0 + math.log(tf[t])
        for j, s in zip(idx, sgn):
            v[j] += w * s
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    stats = {"chars": len(text), "tokens": len(toks), "uniq": (len(tf) / len(content)) if content else 0.0,
             "prose": (1.0 - len(content) / len(toks)) if toks else 0.0}
    return v.astype(np.float32), stats


def page_text(body: str) -> str:
    body = SCRIPT_RE.sub(" ", body)
    return re.sub(r"\s+", " ", _html.unescape(TAG_RE.sub(" ", body))).strip()


def parse_page(body: str, content_type: str = "text/html") -> dict:
    """Visible text, title, internal paths and external .crucible hosts linked, and item prices on a page."""
    if not isinstance(body, str):
        body = ""
    if content_type != "text/html":
        return {"text": body[:20000], "title": "", "paths": [], "hosts": [], "prices": []}
    m = TITLE_RE.search(body)
    title = m.group(1).strip()[:200] if m else ""
    paths, hosts = [], []
    for href in HREF_RE.findall(body)[:80]:
        h = href.strip()
        if h.startswith("http://") or h.startswith("https://") or ".crucible" in h:
            hm = re.match(r"(?:https?://)?([a-z0-9][a-z0-9-]{1,40})\.crucible(/[^?\s]*)?", h.lower())
            if hm:
                hosts.append(hm.group(1))
        elif h.startswith("/") and not h.startswith("/_"):
            p = h.split("?")[0]
            if p not in paths and len(p) <= 120:
                paths.append(p)
    prices = [int(round(float(x) * 100)) for x in PRICE_RE.findall(body)[:20]]
    return {"text": page_text(body)[:20000], "title": title, "paths": paths[:40], "hosts": sorted(set(hosts))[:20],
            "prices": prices}


@lru_cache(maxsize=64)
def topic_matrix(dim: int) -> np.ndarray:
    """float32[len(TOPICS), dim]: unit vector of each topic's words."""
    out = np.zeros((len(TOPICS), dim), np.float32)
    for i, name in enumerate(TOPIC_NAMES):
        out[i] = text_vector(" ".join(TOPICS[name]), dim)[0]
    return out


@lru_cache(maxsize=64)
def lexicon_matrix(dim: int) -> np.ndarray:
    """float32[len(LEXICON), dim]: token vector of every lexicon word (unit length)."""
    m = np.stack([token_vec(w, dim) for w in LEXICON])
    return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def interest_vector(weights: np.ndarray, dim: int, noise: np.ndarray | None = None) -> np.ndarray:
    """Unit interest vectors float32[n, dim] from topic weights float[n, len(TOPICS)] (+ optional noise)."""
    v = weights.astype(np.float32) @ topic_matrix(dim)
    if noise is not None:
        v = v + noise.astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def query_words(interests: np.ndarray, rng: np.random.Generator, tau: np.ndarray, k: int = 2) -> list[str]:
    """For each individual, `k` lexicon words that match its interests, with Gumbel noise of scale `tau`
    (less skilled searchers pick less fitting words). Returns one query string per individual."""
    dim = interests.shape[1]
    lx = lexicon_matrix(dim)
    s = interests.astype(np.float32) @ lx.T
    g = rng.gumbel(size=s.shape).astype(np.float32) * np.asarray(tau, np.float32)[:, None]
    top = np.argsort(-(s + g), axis=1, kind="stable")[:, :k]
    return [" ".join(sorted(LEXICON[j] for j in row)) for row in top]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two float arrays [..., d]."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-6)


def body_digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:24]
