"""The interest-space asset: a real topic corpus (MIND-small news, Microsoft, US, Oct-Nov 2019) embedded by the
pretrained text encoder of `crucible.semantic` (MiniLM-L6-v2), reduced to what a world needs at creation.

`python -m crucible.population.space_asset build` writes `population/data/interest_space_v1.npz` and
`interest_space_v1.json` (provenance, topic names, query words, a corpus of real abstracts per topic). It is derived
data like a calibration checkpoint: computed once, offline, from a checksummed dataset with a pinned encoder, and
loaded by `crucible.population.semantics` without running any model at world creation (so creation, replay and
regeneration of a world never run the encoder for the interest prior).

What it holds:

* `topics`: the 93 MIND subcategories with at least 40 news items in the train split (category/subcategory);
* `centroids` [T, 384]: unit mean embedding of up to `PER_TOPIC` items (title + abstract) of each topic;
* `basis` [K_MAX, 384]: orthonormal top right-singular vectors of the (uncentred) matrix of all sampled item
  embeddings - the principal subspace of the topic corpus inside the encoder's space. Individuals' interests live in
  this subspace (stored as K coordinates; `semantics.interests` returns the full 384-d unit vector);
* `popularity` [T]: each topic's share of all train items (the real popularity prior of interests);
* `energy`: share of each centroid's squared norm captured by the first k basis vectors (k = 8, 16, 32, 48, 64);
* `vocab_coords` [V, K_MAX]: basis coordinates of the query vocabulary (topic query words + the population's web
  lexicon), each word embedded by the encoder - what individuals type into a search box;
* JSON: provenance (dataset file SHA-256, encoder id / version / model hash, sampling seed), query words per topic
  (top tf-idf title words), and up to `CORPUS_PER_TOPIC` real abstracts per topic (what population creators draw
  their writing from).
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NAME = "interest_space_v1"
MIND_NEWS = "/workspace/data/crucible/mind_small/download/train/news.tsv"
MIN_ITEMS = 40
PER_TOPIC = 60
CORPUS_PER_TOPIC = 40
K_MAX = 64
SEED = 20261126
STOP = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our out over
own same she should so some such than that the their them then there these they this those through to too under until
up very was we were what when where which while who whom why will with you your yours new says say said get gets got
one two three first last year years day days week time way make makes made like know see top best worst ways things
thing people man woman may might must also back still even many much us vs""".split())


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z'-]{2,}", s.lower()) if t not in STOP]


def build(news_path: str = MIND_NEWS, out_dir: str = DATA) -> dict:
    from ..semantic.encoders import encode_bytes_many, encoder_info
    with open(news_path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    rows = []
    for line in raw.decode("utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 5:
            continue
        rows.append((p[0], p[1], p[2], p[3].strip(), p[4].strip()))
    by = collections.defaultdict(list)
    for r in rows:
        by[r[2]].append(r)
    topics = sorted(t for t, v in by.items() if len(v) >= MIN_ITEMS)
    rng = np.random.default_rng(SEED)
    texts, owner, corpus, cat = [], [], {}, {}
    for j, t in enumerate(topics):
        items = sorted(by[t], key=lambda r: r[0])
        pick = rng.permutation(len(items))
        cat[t] = items[0][1]
        chosen = [items[i] for i in pick[:PER_TOPIC]]
        for r in chosen:
            texts.append((r[3] + ". " + r[4]).strip())
            owner.append(j)
        absq = [items[i][4] for i in pick if len(items[i][4].split()) >= 12][:CORPUS_PER_TOPIC]
        corpus[t] = absq
    E = np.asarray(encode_bytes_many([s.encode("utf-8") for s in texts], "text"), dtype=np.float64)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    owner = np.asarray(owner)
    T = len(topics)
    C = np.stack([E[owner == j].mean(axis=0) for j in range(T)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(E, full_matrices=False)
    basis = vt[:K_MAX]
    energy = {}
    for k in (8, 16, 32, 48, 64):
        proj = C @ basis[:k].T
        energy[str(k)] = float(np.mean(np.sum(proj ** 2, axis=1)))
    # query words: tf-idf of title words per topic
    df = collections.Counter()
    tf = {t: collections.Counter() for t in topics}
    for t in topics:
        for r in by[t]:
            toks = set(_tokens(r[3]))
            tf[t].update(toks)
    for t in topics:
        df.update(set(tf[t]))
    words = {}
    for t in topics:
        sc = {w: c * math.log(T / df[w]) for w, c in tf[t].items() if c >= 3}
        words[t] = [w for w, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:16]]
    # query vocabulary: every topic's query words plus the population's web lexicon (population.text), each word
    # embedded by the same encoder; stored as coordinates in the basis (the first k columns are the k-dim ones)
    from .text import LEXICON
    vocab = sorted({w for ws in words.values() for w in ws} | {str(w) for w in LEXICON})
    VW = np.asarray(encode_bytes_many([w.encode("utf-8") for w in vocab], "text"), dtype=np.float64)
    VW /= np.maximum(np.linalg.norm(VW, axis=1, keepdims=True), 1e-12)
    popularity = np.array([len(by[t]) for t in topics], dtype=np.float64)
    popularity /= popularity.sum()
    info = encoder_info("text")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, NAME + ".npz"), centroids=C.astype(np.float32),
                        basis=basis.astype(np.float32), popularity=popularity.astype(np.float64),
                        singular_values=s[:K_MAX].astype(np.float64),
                        vocab_coords=(VW @ basis.T).astype(np.float32))
    meta = {"name": NAME, "space_dim": int(E.shape[1]), "k_max": K_MAX, "topics": topics,
            "categories": [cat[t] for t in topics], "words": words, "energy": energy, "vocab": vocab,
            "provenance": {"dataset": "MIND-small (Microsoft News Dataset), train split, news.tsv",
                           "dataset_sha256": sha, "dataset_rows": len(rows), "min_items_per_topic": MIN_ITEMS,
                           "items_per_topic": PER_TOPIC, "sampling_seed": SEED, "encoded_items": len(texts),
                           "encoder": {k: info[k] for k in ("id", "version", "dim", "model_hash",
                                                            "preprocessing_hash")},
                           "basis": "top right-singular vectors of the uncentred matrix of sampled item embeddings",
                           "licence": "MIND: Microsoft Research License Terms (research use); see "
                                      "calibration/datasets.py for the acquisition route"},
            "corpus_per_topic": CORPUS_PER_TOPIC}
    with open(os.path.join(out_dir, NAME + ".json"), "w") as f:
        json.dump({k: v for k, v in meta.items()}, f, indent=1, sort_keys=True)
    with gzip.open(os.path.join(out_dir, NAME + ".corpus.json.gz"), "wt", encoding="utf-8") as f:
        json.dump({"topics": topics, "abstracts": [corpus[t] for t in topics]}, f, sort_keys=True)
    return meta


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        m = build()
        print(json.dumps({"topics": len(m["topics"]), "energy": m["energy"], "provenance": m["provenance"]},
                         indent=1))
    else:
        print("usage: python -m crucible.population.space_asset build")
