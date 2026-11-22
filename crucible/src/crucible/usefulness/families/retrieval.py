"""Information retrieval: rank a generated corpus for a query under hidden relevance (exemplified; NDCG@10).

What makes a document relevant to this world's readers is the hidden rule: how much a match in the title counts
against matches in the body, whether readers want recent documents or much-cited ones, and which kind of document
they prefer or distrust. Each (query, document) pair also has an idiosyncratic component no feature explains, so
even the true rule does not rank perfectly. Published examples show queries with graded relevance labels.
"""
from __future__ import annotations

import math

import numpy as np

from ..category import Population, category_family
from ._common import WORDS, Stream, pick, r3

KINDS = ("guide", "paper", "forum", "news", "manual", "blog", "memo", "wiki")
TOPICS = [WORDS[i:i + 10] for i in range(0, 60, 10)]
_LOG501 = math.log(501.0)

_BOOLEAN = r'''
def solve(args, caller):
    terms = args['query'].split()
    hits = []
    for d in args['docs']:
        words = d['title'].split() + d['body'].split()
        if any(x in words for x in terms):
            hits.append(d['id'])
    return {'ranking': hits[:10]}
'''
_LINEAR = r'''
import math


def solve(args, caller):
    terms = args['query'].split()
    nq = len(terms) or 1
    scored = []
    for d in args['docs']:
        title = d['title'].split()
        body = d['body'].split()
        t = sum(1 for x in terms if x in title) / nq
        b = sum(min(body.count(x), 3) for x in terms) / (3.0 * nq)
        if t + b <= 0:
            continue
        s = W[0] * t + W[1] * b + W[2] * (d['year'] - 2000) / 24.0 + W[3] * math.log1p(d['cites']) / 6.2166 \
            + BOOST.get(d['kind'], 0.0)
        scored.append((-s, d['id']))
    scored.sort()
    return {'ranking': [i for _, i in scored[:10]]}
'''


def _linear(w, boost) -> str:
    return f"W = {list(w)!r}\nBOOST = {boost!r}\n" + _LINEAR


def _dcg(grades) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:10]))


def _score(inp, grades, out) -> float:
    rk = out.get("ranking")
    if not isinstance(rk, list) or len(rk) > len(inp["docs"]):
        return 0.0
    seen = set()
    for x in rk:
        if not isinstance(x, str) or x not in grades or x in seen:
            return 0.0
        seen.add(x)
    ideal = _dcg(sorted(grades.values(), reverse=True))
    return _dcg([grades[x] for x in rk[:10]]) / ideal if ideal > 0 else 0.0


@category_family("retrieval", kind="information retrieval", relation="exemplified", split="val",
                 doc="Rank a generated corpus for queries under a hidden relevance function; NDCG@10.",
                 population=Population(1.1, (8.0, 40.0), (0.5, 2.5), "readers searching an organisation's documents"),
                 demand_links=("discover", "triage"), resource_intensity=0.8)
def build(rng, params):
    text = float(rng.uniform(0.4, 0.6))
    title_share = float(pick(rng, (0.2, 0.8)))
    lead = pick(rng, ("recency", "authority"))
    rest = 1.0 - text
    w = [r3(text * title_share), r3(text * (1 - title_share)),
         r3(rest * (0.85 if lead == "recency" else 0.15)), r3(rest * (0.85 if lead == "authority" else 0.15))]
    kinds = pick(rng, KINDS, 3)
    boost = {kinds[0]: 0.2, kinds[1]: -0.2, kinds[2]: 0.0}
    noise = 0.09

    def generate(rng, size):
        n = int(14 + 16 * size)
        st = Stream(rng, 128)
        topics = st.sample(range(len(TOPICS)), 3)
        # one block of draws for the whole corpus: per word, topical or not and which word
        u = rng.random((n, 24))
        wt = rng.integers(0, 10, (n, 24))
        wa = rng.integers(0, len(WORDS), (n, 24))
        meta = rng.random((n, 6))
        doc_topic = np.array(topics)[(meta[:, 0] * 3).astype(int)]
        topical = u < np.r_[np.full(4, 0.8), np.full(20, 0.7)]             # 4 title slots, 20 body slots
        words = np.where(topical, wt + 10 * doc_topic[:, None], wa).tolist()
        meta = meta.tolist()
        docs = []
        for i in range(n):
            row, m = words[i], meta[i]
            n_title, n_body = 3 + int(m[1] * 2), 12 + int(m[2] * 7)
            docs.append({"id": f"d{i:02d}", "title": " ".join(WORDS[k] for k in row[:n_title]),
                         "body": " ".join(WORDS[k] for k in row[4:4 + n_body]), "year": 2000 + int(m[3] * 25),
                         "cites": int(min(500.0, 12.0 * ((1.0 - m[4]) ** (-1.0 / 1.2) - 1.0))),     # heavy-tailed
                         "kind": kinds[int(m[5] * 3)]})
        topics = [TOPICS[k] for k in topics]
        terms, feats = [], {}
        for _ in range(6):
            terms = st.sample(st.pick(topics), st.between(2, 3))
            feats = {}
            for d in docs:
                title, body = d["title"].split(), d["body"].split()
                t = sum(1 for x in terms if x in title) / len(terms)
                b = sum(min(body.count(x), 3) for x in terms) / (3.0 * len(terms))
                if t + b > 0:
                    feats[d["id"]] = (w[0] * t + w[1] * b + w[2] * (d["year"] - 2000) / 24.0
                                      + w[3] * math.log1p(d["cites"]) / _LOG501 + boost[d["kind"]]
                                      + st.normal(0.0, noise))
            if len(feats) >= 6:
                break
        order = sorted(feats, key=lambda i: (-feats[i], i))
        m = len(order)
        n3 = max(1, round(0.12 * m))
        n2 = n3 + round(0.2 * m)
        n1 = n2 + round(0.3 * m)
        grades = {d["id"]: 0 for d in docs}
        for k, i in enumerate(order):
            grades[i] = 3 if k < n3 else 2 if k < n2 else 1 if k < n1 else 0
        return {"query": " ".join(terms), "docs": docs}, grades

    def truth(inp, grades):
        ideal = sorted((i for i, g in grades.items() if g > 0), key=lambda i: (-grades[i], i))
        return {"ranking": ideal[:10], "relevance": {i: g for i, g in sorted(grades.items()) if g > 0}}

    doc = ("Rank documents for a reader's query.\n"
           "Input: {\"query\": \"space separated terms\", \"docs\": [{\"id\": str, \"title\": str, \"body\": str, "
           "\"year\": int, \"cites\": int, \"kind\": str}, ...]}.\n"
           "Output: {\"ranking\": [up to 10 document ids, best first]}.\n"
           "Score: NDCG@10 against graded relevance (0-3) judged by this world's readers. Documents that match no "
           "query term are never relevant; beyond that, what readers value (title or body matches, recent or "
           "much-cited documents, which kinds of document) is not documented.\n"
           "Published worked examples show queries with the relevance grade of every relevant document.")
    return {"title": "Rank documents for a query", "doc": doc,
            "keywords": ("search", "ranking", "retrieval", "documents", "relevance", "query"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 3.0,
            "tiers": [("boolean-match", _BOOLEAN), ("text-match", _linear((0.5, 0.5, 0.0, 0.0), {})),
                      ("fitted-relevance", _linear(w, boost))],
            "truth": truth, "rule": {"weights": w, "boost": boost, "lead": lead, "noise": noise},
            "table_key": "docs", "output_key": "ranking"}
