"""Needs: the computational work that the simulated economy pays for.

A need is a public problem format (documented for everyone), a seeded generator of real request payloads, an
exact verifier that scores an output in [0, 1] against ground truth only the verifier holds, and reference
implementations at several quality tiers (used by incumbent firms and by simulated developer labour; agents never
see them unless they read an incumbent's open-source code). Software quality is therefore *measured*: correctness,
solution quality relative to a strong reference, and CPU cost of real execution.

`work_scale` converts measured CPU milliseconds of one request into compute units (CU) consumed on the hosting
instance, so slower algorithms cost real money in operating expenses.
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Need:
    name: str
    title: str
    doc: str
    keywords: tuple[str, ...]
    work_scale: float                      # CU per CPU-millisecond
    generate: Callable[[np.random.Generator, float], tuple[dict, Any]]   # (public input, hidden truth)
    score: Callable[[dict, Any, Any], float]                          # (input, hidden, output) -> [0,1]
    tiers: list[tuple[str, str]] = field(default_factory=list)         # (label, python source defining solve)
    base_price_cents: float = 3.0

    def spec_text(self) -> str:
        return f"# Need: {self.name} — {self.title}\n\n{self.doc}\n"


NEEDS: dict[str, Need] = {}


def register(n: Need) -> Need:
    NEEDS[n.name] = n
    return n


def compile_solver(source: str) -> Callable:
    ns: dict = {"__name__": "crucible_ref"}
    exec(compile(source, "<reference>", "exec"), ns)  # trusted reference code only
    return ns["solve"]


# =========================================================================== sort.records
def _gen_sort(rng, size):
    n = int(120 + 280 * size)
    keys = rng.integers(0, 10_000, n)
    recs = [{"id": f"r{i}", "key": int(k)} for i, k in enumerate(keys)]
    return {"records": recs, "descending": bool(rng.random() < 0.3)}, None


def _score_sort(inp, hidden, out):
    try:
        ids = out["ids"]
    except (TypeError, KeyError):
        return 0.0
    recs = inp["records"]
    keymap = {r["id"]: r["key"] for r in recs}
    if not isinstance(ids, list) or len(ids) != len(recs) or set(ids) != set(keymap):
        return 0.0
    ks = [keymap[i] for i in ids]
    if inp["descending"]:
        ks = ks[::-1]
    good = sum(1 for a, b in zip(ks, ks[1:]) if a <= b)
    return 1.0 if good == len(ks) - 1 else 0.5 * good / max(1, len(ks) - 1)


register(Need(
    "sort.records", "Sort records by key",
    "Input: {\"records\": [{\"id\": str, \"key\": int}, ...], \"descending\": bool}.\n"
    "Output: {\"ids\": [record ids in sorted order]} (ascending by key unless descending is true; ties in any order).\n"
    "Score: 1 for a correctly sorted permutation of all ids, partial credit for partially ordered permutations, "
    "0 otherwise.", ("sort", "sorting", "records", "order", "data"), 40.0,
    _gen_sort, _score_sort, [
        ("insertion", "def solve(args, caller):\n    recs = list(args['records'])\n    out = []\n"
                      "    for r in recs:\n        i = len(out)\n        while i > 0 and out[i-1]['key'] > r['key']:\n"
                      "            i -= 1\n        out.insert(i, r)\n    if args.get('descending'):\n        out.reverse()\n"
                      "    return {'ids': [r['id'] for r in out]}\n"),
        ("builtin", "def solve(args, caller):\n    recs = sorted(args['records'], key=lambda r: r['key'], "
                    "reverse=bool(args.get('descending')))\n    return {'ids': [r['id'] for r in recs]}\n"),
    ], base_price_cents=1.0))


# =========================================================================== route.shortest
def _gen_sp(rng, size):
    n = int(50 + 100 * size)
    vs = np.arange(1, n)
    us = (rng.random(n - 1) * vs).astype(np.int64)          # spanning tree keeps it connected
    m = int(2.5 * n)
    a, b = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = a != b
    lo = np.concatenate([us, np.minimum(a, b)[keep]])
    hi = np.concatenate([vs, np.maximum(a, b)[keep]])
    pairs = np.unique(lo * n + hi)
    ws = rng.integers(1, 100, len(pairs))
    E = [[int(p // n), int(p % n), int(w)] for p, w in zip(pairs, ws)]
    s, t = int(rng.integers(0, n)), int(rng.integers(0, n))
    while t == s:
        t = int(rng.integers(0, n))
    return {"n": n, "edges": E, "source": s, "target": t}, None


def _dijkstra(n, edges, s, t):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = [math.inf] * n
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            return d
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    return dist[t]


def _score_sp(inp, hidden, out):
    try:
        path = [int(x) for x in out["path"]]
    except (TypeError, KeyError, ValueError):
        return 0.0
    if not path or path[0] != inp["source"] or path[-1] != inp["target"]:
        return 0.0
    w = {}
    for u, v, c in inp["edges"]:
        w[(u, v)] = min(c, w.get((u, v), math.inf))
        w[(v, u)] = min(c, w.get((v, u), math.inf))
    total = 0
    for a, b in zip(path, path[1:]):
        if (a, b) not in w:
            return 0.0
        total += w[(a, b)]
    opt = _dijkstra(inp["n"], inp["edges"], inp["source"], inp["target"])
    return float(min(1.0, opt / total)) if total > 0 else 0.0


register(Need(
    "route.shortest", "Shortest path in a weighted road network",
    "Input: {\"n\": int, \"edges\": [[u, v, weight], ...] (undirected), \"source\": int, \"target\": int}.\n"
    "Output: {\"path\": [source, ..., target]} following existing edges.\n"
    "Score: optimal length / your length for a valid path, 0 for an invalid one.",
    ("route", "routing", "path", "shortest", "graph", "navigation", "logistics"), 60.0,
    _gen_sp, _score_sp, [
        ("bfs-hops", "from collections import deque\n"
                     "def solve(args, caller):\n    n, s, t = args['n'], args['source'], args['target']\n"
                     "    adj = [[] for _ in range(n)]\n    for u, v, w in args['edges']:\n"
                     "        adj[u].append(v); adj[v].append(u)\n    prev = {s: None}; q = deque([s])\n"
                     "    while q:\n        u = q.popleft()\n        if u == t: break\n        for v in adj[u]:\n"
                     "            if v not in prev:\n                prev[v] = u; q.append(v)\n"
                     "    path = [t]\n    while path[-1] != s:\n        path.append(prev[path[-1]])\n"
                     "    return {'path': path[::-1]}\n"),
        ("dijkstra", "import heapq\n"
                     "def solve(args, caller):\n    n, s, t = args['n'], args['source'], args['target']\n"
                     "    adj = [[] for _ in range(n)]\n    for u, v, w in args['edges']:\n"
                     "        adj[u].append((v, w)); adj[v].append((u, w))\n"
                     "    dist = {s: 0}; prev = {s: None}; pq = [(0, s)]\n    while pq:\n"
                     "        d, u = heapq.heappop(pq)\n        if u == t: break\n        if d > dist[u]: continue\n"
                     "        for v, w in adj[u]:\n            if d + w < dist.get(v, 1e18):\n"
                     "                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))\n"
                     "    path = [t]\n    while path[-1] != s:\n        path.append(prev[path[-1]])\n"
                     "    return {'path': path[::-1]}\n"),
    ], base_price_cents=3.0))


# =========================================================================== route.tsp
def _tour_len(pts, tour):
    return sum(math.dist(pts[tour[i]], pts[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def _nn_2opt(pts, iters=2):
    n = len(pts)
    tour, left = [0], set(range(1, n))
    while left:
        last = pts[tour[-1]]
        nxt = min(left, key=lambda j: (math.dist(last, pts[j]), j))
        tour.append(nxt)
        left.remove(nxt)
    for _ in range(iters):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, d = pts[tour[i - 1]], pts[tour[i]], pts[tour[j]], pts[tour[(j + 1) % n]]
                if math.dist(a, c) + math.dist(b, d) < math.dist(a, b) + math.dist(c, d) - 1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _gen_tsp(rng, size):
    n = int(25 + 35 * size)
    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in rng.random((n, 2)) * 100]
    return {"points": pts}, None


def _score_tsp(inp, hidden, out):
    pts = inp["points"]
    try:
        tour = [int(x) for x in out["tour"]]
    except (TypeError, KeyError, ValueError):
        return 0.0
    if sorted(tour) != list(range(len(pts))):
        return 0.0
    ref = _tour_len(pts, _nn_2opt(pts, iters=4))
    return float(min(1.0, ref / max(1e-9, _tour_len(pts, tour))))


register(Need(
    "route.tsp", "Delivery tour through all stops",
    "Input: {\"points\": [[x, y], ...]}.\nOutput: {\"tour\": [a permutation of all point indices]} (a closed tour).\n"
    "Score: reference tour length / your tour length (capped at 1); 0 if not a permutation.",
    ("tour", "delivery", "route", "travelling", "salesman", "logistics", "tsp", "optimization"), 25.0,
    _gen_tsp, _score_tsp, [
        ("identity", "def solve(args, caller):\n    return {'tour': list(range(len(args['points'])))}\n"),
        ("nearest", "import math\n"
                    "def solve(args, caller):\n    pts = args['points']; tour = [0]; left = set(range(1, len(pts)))\n"
                    "    while left:\n        last = pts[tour[-1]]\n"
                    "        nxt = min(left, key=lambda j: (math.dist(last, pts[j]), j))\n"
                    "        tour.append(nxt); left.remove(nxt)\n    return {'tour': tour}\n"),
        ("nn-2opt", "import math\n"
                    "def solve(args, caller):\n    pts = args['points']; n = len(pts); tour = [0]; left = set(range(1, n))\n"
                    "    while left:\n        last = pts[tour[-1]]\n"
                    "        nxt = min(left, key=lambda j: (math.dist(last, pts[j]), j))\n"
                    "        tour.append(nxt); left.remove(nxt)\n    d = math.dist\n    for _ in range(6):\n"
                    "        improved = False\n        for i in range(1, n - 1):\n            for j in range(i + 1, n):\n"
                    "                a, b, c, e = pts[tour[i-1]], pts[tour[i]], pts[tour[j]], pts[tour[(j+1) % n]]\n"
                    "                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:\n"
                    "                    tour[i:j+1] = tour[i:j+1][::-1]; improved = True\n"
                    "        if not improved: break\n    return {'tour': tour}\n"),
    ], base_price_cents=5.0))


# =========================================================================== schedule.jobs
def _gen_sched(rng, size):
    n = int(30 + 70 * size)
    m = int(rng.integers(3, 9))
    durs = [int(x) for x in rng.integers(1, 100, n)]
    return {"durations": durs, "machines": m}, None


def _score_sched(inp, hidden, out):
    d, m = inp["durations"], inp["machines"]
    try:
        a = [int(x) for x in out["assignment"]]
    except (TypeError, KeyError, ValueError):
        return 0.0
    if len(a) != len(d) or any(x < 0 or x >= m for x in a):
        return 0.0
    loads = [0] * m
    for job, mach in zip(d, a):
        loads[mach] += job
    lb = max(max(d), math.ceil(sum(d) / m))
    return float(lb / max(loads))


register(Need(
    "schedule.jobs", "Assign batch jobs to machines",
    "Input: {\"durations\": [int], \"machines\": int}.\nOutput: {\"assignment\": [machine index for each job]}.\n"
    "Score: lower bound on the makespan / your makespan.",
    ("schedule", "scheduling", "jobs", "machines", "makespan", "batch", "cluster", "compute"), 30.0,
    _gen_sched, _score_sched, [
        ("round-robin", "def solve(args, caller):\n    m = args['machines']\n"
                        "    return {'assignment': [i % m for i in range(len(args['durations']))]}\n"),
        ("lpt", "import heapq\n"
                "def solve(args, caller):\n    d, m = args['durations'], args['machines']\n"
                "    heap = [(0, k) for k in range(m)]; a = [0] * len(d)\n"
                "    for i in sorted(range(len(d)), key=lambda i: -d[i]):\n        load, k = heapq.heappop(heap)\n"
                "        a[i] = k; heapq.heappush(heap, (load + d[i], k))\n    return {'assignment': a}\n"),
    ], base_price_cents=2.0))


# =========================================================================== pack.knapsack
def _gen_knap(rng, size):
    n = int(20 + 40 * size)
    ws = rng.integers(10, 60, n)
    items = [[int(w), int(w + rng.integers(0, 12))] for w in ws]  # strongly correlated: greedy is not optimal
    cap = int(sum(w for w, _ in items) * rng.uniform(0.25, 0.45))
    return {"items": items, "capacity": cap}, None


def _knap_opt(items, cap):
    best = np.zeros(cap + 1, np.int64)
    for w, v in items:
        if w <= cap:
            best[w:] = np.maximum(best[w:], best[:cap + 1 - w] + v)
    return int(best[cap])


def _score_knap(inp, hidden, out):
    items, cap = inp["items"], inp["capacity"]
    try:
        chosen = sorted({int(x) for x in out["chosen"]})
    except (TypeError, KeyError, ValueError):
        return 0.0
    if any(i < 0 or i >= len(items) for i in chosen):
        return 0.0
    w = sum(items[i][0] for i in chosen)
    if w > cap:
        return 0.0
    v = sum(items[i][1] for i in chosen)
    return float(v / max(1, _knap_opt(items, cap)))


register(Need(
    "pack.knapsack", "Choose the most valuable load under a weight limit",
    "Input: {\"items\": [[weight, value], ...], \"capacity\": int}.\nOutput: {\"chosen\": [item indices]}.\n"
    "Score: your total value / the optimal value; 0 if over capacity.",
    ("knapsack", "packing", "load", "cargo", "capacity", "optimization", "budget"), 35.0,
    _gen_knap, _score_knap, [
        ("first-fit", "def solve(args, caller):\n    cap = args['capacity']; out = []\n"
                      "    for i, (w, v) in enumerate(args['items']):\n        if w <= cap:\n"
                      "            out.append(i); cap -= w\n    return {'chosen': out}\n"),
        ("greedy-ratio", "def solve(args, caller):\n    items = args['items']; cap = args['capacity']; out = []\n"
                         "    for i in sorted(range(len(items)), key=lambda i: -items[i][1] / items[i][0]):\n"
                         "        if items[i][0] <= cap:\n            out.append(i); cap -= items[i][0]\n"
                         "    return {'chosen': out}\n"),
        ("dp", "def solve(args, caller):\n    items = args['items']; cap = args['capacity']\n"
               "    best = [0] * (cap + 1); take = [[False] * (cap + 1) for _ in items]\n"
               "    for i, (w, v) in enumerate(items):\n        for c in range(cap, w - 1, -1):\n"
               "            if best[c - w] + v > best[c]:\n                best[c] = best[c - w] + v; take[i][c] = True\n"
               "    out = []; c = cap\n    for i in range(len(items) - 1, -1, -1):\n"
               "        if take[i][c]:\n            out.append(i); c -= items[i][0]\n    return {'chosen': out}\n"),
    ], base_price_cents=4.0))


# =========================================================================== match.stable
def _gen_match(rng, size):
    n = int(8 + 16 * size)
    men = [[int(x) for x in rng.permutation(n)] for _ in range(n)]
    women = [[int(x) for x in rng.permutation(n)] for _ in range(n)]
    return {"proposers": men, "acceptors": women}, None


def _score_match(inp, hidden, out):
    P, A = inp["proposers"], inp["acceptors"]
    n = len(P)
    try:
        pairs = [(int(a), int(b)) for a, b in out["pairs"]]
    except (TypeError, KeyError, ValueError):
        return 0.0
    if sorted(p for p, _ in pairs) != list(range(n)) or sorted(a for _, a in pairs) != list(range(n)):
        return 0.0
    mp = dict(pairs)
    ma = {a: p for p, a in pairs}
    rp = [{a: r for r, a in enumerate(pr)} for pr in P]
    ra = [{p: r for r, p in enumerate(pr)} for pr in A]
    blocking = sum(1 for p in range(n) for a in range(n)
                   if rp[p][a] < rp[p][mp[p]] and ra[a][p] < ra[a][ma[a]])
    return float(max(0.0, 1.0 - 4.0 * blocking / (n * n)))


register(Need(
    "match.stable", "Stable matching of two sides of a market",
    "Input: {\"proposers\": [[acceptor indices by preference]], \"acceptors\": [[proposer indices by preference]]}.\n"
    "Output: {\"pairs\": [[proposer, acceptor], ...]} — a perfect matching.\n"
    "Score: 1 for a stable matching, reduced by the number of blocking pairs; 0 if not a perfect matching.",
    ("matching", "stable", "market", "pairs", "assignment", "two-sided", "dating", "hiring"), 30.0,
    _gen_match, _score_match, [
        ("identity", "def solve(args, caller):\n    return {'pairs': [[i, i] for i in range(len(args['proposers']))]}\n"),
        ("gale-shapley", "def solve(args, caller):\n    P, A = args['proposers'], args['acceptors']; n = len(P)\n"
                         "    rank = [{p: r for r, p in enumerate(pr)} for pr in A]\n"
                         "    nxt = [0] * n; held = {}; free = list(range(n))\n    while free:\n        p = free.pop()\n"
                         "        a = P[p][nxt[p]]; nxt[p] += 1\n        if a not in held:\n            held[a] = p\n"
                         "        elif rank[a][p] < rank[a][held[a]]:\n            free.append(held[a]); held[a] = p\n"
                         "        else:\n            free.append(p)\n"
                         "    return {'pairs': [[p, a] for a, p in sorted(held.items())]}\n"),
    ], base_price_cents=3.0))


# =========================================================================== forecast.series
def _gen_fc(rng, size):
    period = int(rng.choice([7, 12, 24]))
    n = int(60 + 120 * size)
    h = 8
    t = np.arange(n + h)
    trend = rng.uniform(-0.05, 0.15) * t
    season = rng.uniform(3, 12) * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28))
    ar = np.zeros(n + h)
    for i in range(1, n + h):
        ar[i] = 0.6 * ar[i - 1] + rng.normal(0, 1.5)
    y = 50 + trend + season + ar
    hist = [round(float(v), 3) for v in y[:n]]
    truth = [float(v) for v in y[n:]]
    return {"history": hist, "horizon": h}, truth


def _score_fc(inp, truth, out):
    try:
        f = [float(x) for x in out["forecast"]]
    except (TypeError, KeyError, ValueError):
        return 0.0
    if len(f) != inp["horizon"] or any(not math.isfinite(x) for x in f):
        return 0.0
    naive = inp["history"][-1]
    mae = sum(abs(a - b) for a, b in zip(f, truth)) / len(truth)
    mae_naive = sum(abs(naive - b) for b in truth) / len(truth)
    return float(mae_naive / (mae_naive + mae + 1e-9))


register(Need(
    "forecast.series", "Demand forecasting",
    "Input: {\"history\": [float], \"horizon\": int}.\nOutput: {\"forecast\": [horizon floats]}.\n"
    "Score: mae_naive / (mae_naive + mae) on the realised future (the last-value forecast scores 0.5).",
    ("forecast", "prediction", "demand", "time-series", "analytics", "planning"), 20.0,
    _gen_fc, _score_fc, [
        ("last-value", "def solve(args, caller):\n    return {'forecast': [args['history'][-1]] * args['horizon']}\n"),
        ("moving-avg", "def solve(args, caller):\n    h = args['history'][-12:]\n"
                       "    return {'forecast': [sum(h) / len(h)] * args['horizon']}\n"),
        ("seasonal", "def solve(args, caller):\n    y = args['history']; n = len(y); H = args['horizon']\n"
                     "    best, bp = None, 1\n    for p in range(2, min(40, n // 3)):\n"
                     "        e = sum(abs(y[i] - y[i - p]) for i in range(p, n)) / (n - p)\n"
                     "        if best is None or e < best:\n            best, bp = e, p\n"
                     "    k = max(1, (n // bp) - 1)\n    drift = (sum(y[-bp:]) - sum(y[-2*bp:-bp])) / (bp * bp) if n >= 2*bp else 0.0\n"
                     "    return {'forecast': [y[n - bp + (i % bp)] + drift * bp * (1 + i // bp) for i in range(H)]}\n"),
    ], base_price_cents=4.0))


# =========================================================================== text.extract
_EMAIL = re.compile(r"\b[a-z][a-z0-9._]{1,20}@[a-z][a-z0-9-]{1,20}\.(?:crucible|net|org)\b")
_AMOUNT = re.compile(r"\$(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?")


def _gen_text(rng, size):
    words = ["invoice", "please", "send", "payment", "to", "contact", "regarding", "order", "total", "due", "the",
             "supplier", "account", "for", "services", "rendered", "and", "thanks", "balance", "refund"]
    names, hosts = ["ana", "bo", "cy", "dee", "eli"], ["mail", "shop", "acme"]
    ewraps = ["{}", "<{}>", "({}),", "{}.", "mailto:{};"]
    awraps = ["{}", "({})", "{},", "total:{}", "{}."]
    docs, truth = [], []
    for _ in range(int(4 + 8 * size)):
        k = int(rng.integers(15, 40))
        r = rng.random(k)
        wi = rng.integers(0, len(words), k)
        ints = rng.integers(0, 10_000_000, (k, 4))
        parts, emails, amounts = [], [], []
        for j in range(k):
            if r[j] < 0.06:
                e = f"{names[ints[j, 0] % 5]}{1 + ints[j, 1] % 98}@{hosts[ints[j, 2] % 3]}.crucible"
                parts.append(ewraps[ints[j, 3] % 5].format(e))
                emails.append(e)
            elif r[j] < 0.12:
                c = 100 + int(ints[j, 0]) % 9_999_900
                if ints[j, 1] % 10 < 7:
                    txt = f"${c // 100:,}.{c % 100:02d}"
                else:
                    c = (c // 100) * 100
                    txt = f"${c // 100:,}"
                parts.append(awraps[ints[j, 2] % 5].format(txt))
                amounts.append(c)
            elif r[j] < 0.14:
                parts.append(f"user{1 + ints[j, 0] % 98}@example")  # distractor: not a valid address
            else:
                parts.append(words[wi[j]])
        docs.append(" ".join(parts))
        truth.append({"emails": emails, "amounts_cents": amounts})
    return {"documents": docs}, truth


def _f1(pred, gold):
    from collections import Counter
    p, g = Counter(pred), Counter(gold)
    tp = sum((p & g).values())
    if not pred and not gold:
        return 1.0
    if tp == 0:
        return 0.0
    prec, rec = tp / sum(p.values()), tp / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _score_text(inp, truth, out):
    try:
        recs = out["records"]
        if len(recs) != len(truth):
            return 0.0
        s = 0.0
        for r, t in zip(recs, truth):
            s += 0.5 * _f1([str(e) for e in r["emails"]], t["emails"]) + \
                 0.5 * _f1([int(a) for a in r["amounts_cents"]], t["amounts_cents"])
        return float(s / len(truth))
    except (TypeError, KeyError, ValueError):
        return 0.0


register(Need(
    "text.extract", "Extract contacts and amounts from documents",
    "Input: {\"documents\": [str]}.\nOutput: {\"records\": [{\"emails\": [str], \"amounts_cents\": [int]} per document]}.\n"
    "Emails look like name@host.crucible; amounts are dollar amounts like $1,234.56 (in cents: 123456).\n"
    "Score: mean F1 over emails and amounts.",
    ("extract", "parsing", "documents", "invoices", "emails", "data-entry", "text"), 30.0,
    _gen_text, _score_text, [
        ("split", "def solve(args, caller):\n    out = []\n    for d in args['documents']:\n"
                  "        toks = [t.strip('.,;()<>') for t in d.split()]\n        am = []\n        for t in toks:\n"
                  "            if t.startswith('$'):\n                try:\n"
                  "                    am.append(int(round(float(t[1:].replace(',', '')) * 100)))\n"
                  "                except ValueError:\n                    pass\n"
                  "        out.append({'emails': [t for t in toks if '@' in t], 'amounts_cents': am})\n"
                  "    return {'records': out}\n"),
        ("regex", "import re\nE = re.compile(r'\\b[a-z][a-z0-9._]{1,20}@[a-z][a-z0-9-]{1,20}\\.(?:crucible|net|org)\\b')\n"
                  "A = re.compile(r'\\$(\\d{1,3}(?:,\\d{3})*|\\d+)(?:\\.(\\d{2}))?')\n"
                  "def solve(args, caller):\n    out = []\n    for d in args['documents']:\n"
                  "        am = [int(m.group(1).replace(',', '')) * 100 + int(m.group(2) or 0) for m in A.finditer(d)]\n"
                  "        out.append({'emails': E.findall(d), 'amounts_cents': am})\n    return {'records': out}\n"),
    ], base_price_cents=3.0))


def tier_source(need: str, tier: int) -> str:
    tiers = NEEDS[need].tiers
    return tiers[max(0, min(tier, len(tiers) - 1))][1]


def best_tier(need: str) -> int:
    return len(NEEDS[need].tiers) - 1
