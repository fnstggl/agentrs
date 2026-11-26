"""Solvers the scripted reference founder writes itself from the public format documents.

These are the reference strategy's own programs (written for this module from each format's public input/output
documentation), not the kernel's reference tiers in `economy.needs` (which world-generated incumbents run and no
agent sees). They run in the sandbox like any agent's code. `SOLVERS[format]` is Python source defining
`solve(args, caller)`; `GARBAGE[format]` is an input-ignoring program with the same interface (used by tests to show
that a useless service earns about nothing).
"""
from __future__ import annotations

SOLVERS: dict[str, str] = {
    "schedule.jobs": '''
def solve(args, caller):
    d = [int(x) for x in args["durations"]]
    m = int(args["machines"])
    loads = [0] * m
    out = [0] * len(d)
    for j in sorted(range(len(d)), key=lambda j: (-d[j], j)):
        k = min(range(m), key=lambda k: (loads[k], k))
        out[j] = k
        loads[k] += d[j]
    # local improvement: move a job from the busiest machine when it shortens the makespan
    for _ in range(200):
        hi = max(range(m), key=lambda k: loads[k])
        lo = min(range(m), key=lambda k: loads[k])
        best = None
        for j in range(len(d)):
            if out[j] == hi and loads[lo] + d[j] < loads[hi]:
                if best is None or d[j] > d[best]:
                    best = j
        if best is None:
            break
        out[best] = lo
        loads[hi] -= d[best]
        loads[lo] += d[best]
    return {"assignment": out}
''',
    "pack.knapsack": '''
def solve(args, caller):
    items = [(int(w), int(v)) for w, v in args["items"]]
    cap = int(args["capacity"])
    best = [0] * (cap + 1)
    keep = [[False] * (cap + 1) for _ in items]
    for i, (w, v) in enumerate(items):
        for c in range(cap, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
                keep[i][c] = True
    out, c = [], cap
    for i in range(len(items) - 1, -1, -1):
        if keep[i][c]:
            out.append(i)
            c -= items[i][0]
    return {"chosen": sorted(out)}
''',
    "route.tsp": '''
import math
def solve(args, caller):
    pts = [tuple(p) for p in args["points"]]
    n = len(pts)
    if n < 3:
        return {"tour": list(range(n))}
    left = set(range(1, n))
    tour = [0]
    while left:
        a = pts[tour[-1]]
        nxt = min(left, key=lambda j: (math.dist(a, pts[j]), j))
        tour.append(nxt)
        left.discard(nxt)
    d = math.dist
    improved = True
    rounds = 0
    while improved and rounds < 12:
        improved = False
        rounds += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, e = pts[tour[i - 1]], pts[tour[i]], pts[tour[j]], pts[tour[(j + 1) % n]]
                if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return {"tour": tour}
''',
    "route.shortest": '''
import heapq
def solve(args, caller):
    n, s, t = int(args["n"]), int(args["source"]), int(args["target"])
    adj = [[] for _ in range(n)]
    for u, v, w in args["edges"]:
        adj[int(u)].append((int(v), int(w)))
        adj[int(v)].append((int(u), int(w)))
    dist = [None] * n
    prev = [None] * n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        if u == t:
            break
        for v, w in adj[u]:
            nd = du + w
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    path, x = [], t
    while x is not None:
        path.append(x)
        x = prev[x]
    return {"path": path[::-1]}
''',
    "match.stable": '''
def solve(args, caller):
    P = [list(map(int, p)) for p in args["proposers"]]
    A = [list(map(int, a)) for a in args["acceptors"]]
    rank = [{p: r for r, p in enumerate(a)} for a in A]
    nxt = [0] * len(P)
    held = [None] * len(A)
    free = list(range(len(P)))
    while free:
        p = free.pop()
        if nxt[p] >= len(P[p]):
            continue
        a = P[p][nxt[p]]
        nxt[p] += 1
        cur = held[a]
        if cur is None:
            held[a] = p
        elif rank[a].get(p, 10**9) < rank[a].get(cur, 10**9):
            held[a] = p
            free.append(cur)
        else:
            free.append(p)
    return {"pairs": sorted([p, a] for a, p in enumerate(held) if p is not None)}
''',
    "text.extract": '''
import re
EM = re.compile(r"\\b[a-z][a-z0-9._]{1,20}@[a-z][a-z0-9-]{1,20}\\.(?:crucible|net|org)\\b")
AM = re.compile(r"\\$(\\d{1,3}(?:,\\d{3})*|\\d+)(?:\\.(\\d{2}))?")
def solve(args, caller):
    out = []
    for doc in args["documents"]:
        amounts = [int(m.group(1).replace(",", "")) * 100 + int(m.group(2) or 0) for m in AM.finditer(doc)]
        out.append({"emails": EM.findall(doc), "amounts_cents": amounts})
    return {"records": out}
''',
}

GARBAGE: dict[str, str] = {
    "schedule.jobs": "def solve(args, caller):\n    return {'assignment': [0 for _ in args['durations']]}\n",
    "pack.knapsack": "def solve(args, caller):\n    return {'chosen': []}\n",
    "route.tsp": "def solve(args, caller):\n    return {'tour': list(range(len(args['points'])))}\n",
    "route.shortest": "def solve(args, caller):\n    return {'path': [args['source'], args['target']]}\n",
    "match.stable": "def solve(args, caller):\n    return {'pairs': [[i, i] for i in range(len(args['proposers']))]}\n",
    "text.extract": "def solve(args, caller):\n    return {'records': [{'emails': [], 'amounts_cents': []} for _ in args['documents']]}\n",
}
