"""Network reliability: choose routes under hidden per-link failure rates (experiential).

A request is a network whose links have a class and a length. The documentation states only a nominal rating (the
same failure rate per unit length for every class). In each world the true rates differ by class - some classes
fail far more often than rated, some carry a fixed per-hop risk - and nothing states them: they can only be
learned from how well routes turn out. The score of a route is its probability of surviving divided by that of
the most reliable route.
"""
from __future__ import annotations

import heapq
import math

from ..category import Population, category_family
from ._common import Stream, is_int, pick, r3

CLASSES = ("fiber", "copper", "radio", "sat", "laser", "coax", "mesh", "relay")
NOMINAL = 0.03          # the documented rating: failure hazard per unit length, every class

_FIRST_FOUND = r'''
def solve(args, caller):
    s, t = args['source'], args['target']
    adj = {}
    for l in args['links']:
        adj.setdefault(l['a'], []).append(l['b'])
        adj.setdefault(l['b'], []).append(l['a'])
    path, seen = [s], {s}
    while path[-1] != t:
        nxt = next((v for v in adj.get(path[-1], ()) if v not in seen), None)
        if nxt is None:
            path.pop()
            continue
        seen.add(nxt)
        path.append(nxt)
    return {'path': path}
'''
_DIJKSTRA = r'''
import heapq


def solve(args, caller):
    s, t = args['source'], args['target']
    adj = {}
    for l in args['links']:
        a, b = HOP.get(l['class'], 0.0), LEN.get(l['class'], DEFAULT)
        c = a + b * l['len']
        adj.setdefault(l['a'], []).append((l['b'], c))
        adj.setdefault(l['b'], []).append((l['a'], c))
    dist = {s: 0.0}
    prev = {s: None}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist[u]:
            continue
        for v, c in adj.get(u, ()):
            if d + c < dist.get(v, 1e18):
                dist[v] = d + c
                prev[v] = u
                heapq.heappush(pq, (d + c, v))
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return {'path': path[::-1]}
'''


def _router(hop: dict, per_len: dict) -> str:
    return f"HOP = {hop!r}\nLEN = {per_len!r}\nDEFAULT = {NOMINAL!r}\n" + _DIJKSTRA


def _best(n, cost, s, t) -> float:
    adj: list[list] = [[] for _ in range(n)]
    for (a, b), c in cost.items():
        adj[a].append((b, c))
        adj[b].append((a, c))
    dist = [math.inf] * n
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            return d
        if d > dist[u]:
            continue
        for v, c in adj[u]:
            if d + c < dist[v]:
                dist[v] = d + c
                heapq.heappush(pq, (d + c, v))
    return dist[t]


def _score(inp, hidden, out) -> float:
    path = out.get("path")
    cost, best = hidden
    if not isinstance(path, list) or not 2 <= len(path) <= 2 * inp["n"]:
        return 0.0
    if not all(is_int(x) for x in path) or path[0] != inp["source"] or path[-1] != inp["target"]:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        c = cost.get((a, b) if a < b else (b, a))
        if c is None:
            return 0.0
        total += c
    return math.exp(-(total - best))


@category_family("netrel", kind="network reliability", relation="experiential", split="test",
                 doc="Choose routes under hidden per-link failure rates; survival relative to the best route.",
                 population=Population(0.9, (8.0, 40.0), (0.6, 3.0), "network operators routing critical traffic"),
                 demand_links=("schedule", "coord"), resource_intensity=1.0)
def build(rng, params):
    classes = pick(rng, CLASSES, int(rng.integers(3, 5)))
    # true hazards: one class much worse than rated, one much better, and one with a fixed per-hop risk
    mult = [4.0, 0.3, 1.0, 2.0][:len(classes)]
    order = [int(x) for x in rng.permutation(len(classes))]
    per_len = {c: r3(NOMINAL * mult[order[k]] * rng.uniform(0.8, 1.25)) for k, c in enumerate(classes)}
    hop = {c: 0.0 for c in classes}
    hop[classes[int(rng.integers(0, len(classes)))]] = r3(rng.uniform(0.12, 0.3))
    freq = [float(x) for x in rng.dirichlet([3.0] * len(classes))]
    # what an experienced operator believes: per-length hazards known up to estimation error, and the fixed
    # per-hop risk of one class underestimated by half (what a newcomer with better evidence can still discover)
    est_len = {c: r3(v * math.exp(rng.normal(0.0, 0.25))) for c, v in per_len.items()}
    est_hop = {c: r3(0.5 * v) for c, v in hop.items()}
    rule = {"classes": classes, "per_length": per_len, "per_hop": hop, "frequency": freq}

    cum = [sum(freq[:k + 1]) for k in range(len(freq))]

    def generate(rng, size):
        st = Stream(rng)
        n = int(10 + 8 * size)
        pairs = set()
        for v in range(1, n):
            pairs.add((st.below(v), v))
        for _ in range(int(1.4 * n)):
            a, b = st.below(n), st.below(n)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        links, cost = [], {}
        for k, (a, b) in enumerate(sorted(pairs)):
            u = st.u()
            c = classes[next((j for j, x in enumerate(cum) if u < x), len(classes) - 1)]
            ln = st.between(1, 9)
            links.append({"id": f"l{k:02d}", "a": a, "b": b, "class": c, "len": ln})
            cost[(a, b)] = hop[c] + per_len[c] * ln
        s = st.below(n)
        t = st.below(n - 1)
        t = t + 1 if t >= s else t
        return {"n": n, "links": links, "source": s, "target": t}, (cost, _best(n, cost, s, t))

    doc = ("Route traffic over the path most likely to survive.\n"
           "Input: {\"n\": int, \"links\": [{\"id\": str, \"a\": int, \"b\": int, \"class\": str, \"len\": int}, ...] "
           "(undirected), \"source\": int, \"target\": int}.\n"
           "Output: {\"path\": [source, ..., target]} following existing links.\n"
           f"Rated reliability: every class of link ({', '.join(sorted(classes))}) is rated to survive a period "
           f"with probability exp(-{NOMINAL} x len). Ratings are nominal: how links of each class really fail in "
           "this world is not documented and shows only in how routes turn out.\n"
           "Score: survival probability of your path / survival probability of the most reliable path; 0 for an "
           "invalid path.")
    return {"title": "Route traffic over unreliable links", "doc": doc,
            "keywords": ("routing", "reliability", "network", "links", "failure", "paths", "resilience"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 4.0,
            "tiers": [("first-found", _FIRST_FOUND), ("rated-reliability", _router({}, {})),
                      ("field-experience", _router(est_hop, est_len))],
            "rule": rule, "table_key": "links", "output_key": "path"}
