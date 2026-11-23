"""Scheduling under a generated mix of constraints: deadlines, precedence, resource limits (specified).

Everything is documented; the difficulty is algorithmic. Each world draws which constraints bind: a machine count
or a cumulative resource with per-task demand, sparse or dense precedence, release times or none, tight or loose
deadlines and how heavily lateness is penalised. Every instance has a planted schedule with no late task; the
score compares the submitted schedule's cost with the planted one.
"""
from __future__ import annotations

import numpy as np

from ..category import Population, category_family
from ._common import Stream, is_int, pick

_SOLVER = r'''
import heapq


def _order(n, preds, key):
    npred = [len(p) for p in preds]
    succ = [[] for _ in range(n)]
    for b in range(n):
        for a in preds[b]:
            succ[a].append(b)
    heap = [(key[i], i) for i in range(n) if npred[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, i = heapq.heappop(heap)
        out.append(i)
        for b in succ[i]:
            npred[b] -= 1
            if npred[b] == 0:
                heapq.heappush(heap, (key[b], b))
    return out, succ


def _sgs(order, dur, need, rel, preds, cap):
    use = [0] * (sum(dur) + max(rel) + 2)
    start = [0] * len(dur)
    for i in order:
        t = rel[i]
        for p in preds[i]:
            if start[p] + dur[p] > t:
                t = start[p] + dur[p]
        d = dur[i]
        lim = cap - need[i]
        while max(use[t:t + d]) > lim:
            t += 1
        for u in range(t, t + d):
            use[u] += need[i]
        start[i] = t
    return start


def _cost(start, dur, due, wt, lam):
    late = sum(w * max(0, s + d - e) for s, d, e, w in zip(start, dur, due, wt))
    return max(s + d for s, d in zip(start, dur)) + lam * late / sum(wt)


def solve(args, caller):
    tasks = args['tasks']
    n = len(tasks)
    dur = [t['dur'] for t in tasks]
    need = [t['need'] for t in tasks]
    rel = [t['release'] for t in tasks]
    due = [t['due'] for t in tasks]
    wt = [t['weight'] for t in tasks]
    preds = [[] for _ in range(n)]
    for a, b in args['before']:
        preds[b].append(a)
    order, succ = _order(n, preds, list(range(n)))
    if MODE == 0:
        start, t = [0] * n, 0
        for i in order:
            t = max(t, rel[i])
            start[i] = t
            t += dur[i]
        return {'start': start}
    best = _sgs(order, dur, need, rel, preds, args['capacity'])
    if MODE == 1:
        return {'start': best}
    adj = list(due)
    for i in reversed(order):
        for b in succ[i]:
            if adj[b] - dur[b] < adj[i]:
                adj[i] = adj[b] - dur[b]
    order2, _ = _order(n, preds, adj)
    alt = _sgs(order2, dur, need, rel, preds, args['capacity'])
    lam = args['late_penalty']
    return {'start': alt if _cost(alt, dur, due, wt, lam) < _cost(best, dur, due, wt, lam) else best}
'''


def _sgs(order, dur, need, rel, preds, cap):
    use = [0] * (sum(dur) + max(rel) + 2)
    start = [0] * len(dur)
    for i in order:
        t = rel[i]
        for p in preds[i]:
            t = max(t, start[p] + dur[p])
        d, lim = dur[i], cap - need[i]
        while max(use[t:t + d]) > lim:
            t += 1
        for u in range(t, t + d):
            use[u] += need[i]
        start[i] = t
    return start


def _score(inp, ref, out) -> float:
    st, tasks = out.get("start"), inp["tasks"]
    n = len(tasks)
    if not isinstance(st, list) or len(st) != n:
        return 0.0
    total = sum(t["dur"] for t in tasks)
    horizon = 2 * total + max(t["release"] for t in tasks) + 1
    for s, t in zip(st, tasks):
        if not is_int(s) or s < t["release"] or s + t["dur"] > horizon:
            return 0.0
    for a, b in inp["before"]:
        if st[b] < st[a] + tasks[a]["dur"]:
            return 0.0
    use = np.zeros(horizon + 1, np.int64)
    for s, t in zip(st, tasks):
        use[s:s + t["dur"]] += t["need"]
    if use.max() > inp["capacity"]:
        return 0.0
    late = sum(t["weight"] * max(0, s + t["dur"] - t["due"]) for s, t in zip(st, tasks))
    cost = max(s + t["dur"] for s, t in zip(st, tasks)) + inp["late_penalty"] * late / sum(t["weight"] for t in tasks)
    return min(1.0, ref / cost) if cost > 0 else 0.0


@category_family("schedule", kind="scheduling", relation="specified", split="train",
                 doc="Scheduling with a generated mix of constraints (deadlines, precedence, resource limits).",
                 population=Population(1.0, (4.0, 24.0), (0.7, 3.0), "operations planners sequencing work orders"),
                 demand_links=("coord", "allocate"), resource_intensity=1.3)
def build(rng, params):
    cumulative = bool(rng.random() < 0.5)
    cap = int(rng.integers(4, 7)) if cumulative else int(rng.integers(2, 5))
    p_edge = float(pick(rng, (0.06, 0.12, 0.2)))
    release = bool(rng.random() < 0.5)
    lam = int(pick(rng, (2, 4, 8)))
    tight = bool(rng.random() < 0.5)
    rule = {"cumulative": cumulative, "capacity": cap, "p_edge": p_edge, "release": release, "late_penalty": lam,
            "tight": tight}

    def generate(rng, size):
        g = Stream(rng)
        n = int(8 + 14 * size)
        dur = [g.between(1, 9) for _ in range(n)]
        need = [g.between(1, 3) for _ in range(n)] if cumulative else [1] * n
        rank = g.shuffled(range(n))                              # precedence only from lower to higher rank
        preds = [[] for _ in range(n)]
        for b in range(n):
            for a in range(n):
                if rank[a] < rank[b] and len(preds[b]) < 2 and g.chance(p_edge):
                    preds[b].append(a)
        late = max(1, sum(dur) // (2 * cap))
        rel = [g.below(late) if release and g.chance(0.3) else 0 for _ in range(n)]
        best = None
        for _ in range(4):                                       # planted schedule: best of four random priorities
            pri = g.shuffled(range(n))
            st = _sgs(_topo(sorted(range(n), key=pri.__getitem__), preds), dur, need, rel, preds, cap)
            mk = max(s + d for s, d in zip(st, dur))
            if best is None or mk < best[0]:
                best = (mk, st)
        mk, st = best
        slack = 2 if tight else max(2, mk // 4)
        tasks = [{"id": i, "dur": dur[i], "need": need[i], "release": rel[i],
                  "due": st[i] + dur[i] + g.between(0, slack), "weight": g.between(1, 3)} for i in range(n)]
        before = [[a, b] for b in range(n) for a in preds[b]]
        return {"tasks": tasks, "before": before, "capacity": cap, "late_penalty": lam}, float(mk)

    res = (f"a cumulative resource of capacity {cap} (each task draws its \"need\" while it runs)" if cumulative
           else f"{cap} identical machines (every task has need 1)")
    doc = ("Schedule tasks to finish early and on time.\n"
           "Input: {\"tasks\": [{\"id\": int (its index), \"dur\": int, \"need\": int, \"release\": int, \"due\": int, "
           "\"weight\": int}, ...], \"before\": [[a, b], ...], \"capacity\": int, \"late_penalty\": int}.\n"
           "Output: {\"start\": [integer start time of every task, by index]}.\n"
           "Constraints: a task starts at or after its release; for every [a, b] in \"before\", b starts after a "
           "ends; at every time the running tasks' total need is at most the capacity.\n"
           "Cost: makespan + late_penalty x (weighted mean lateness), where a task's lateness is the time by "
           "which it ends after its due time and the mean is weighted by \"weight\" over all tasks.\n"
           f"In this world: {res}; precedence is {'dense' if p_edge > 0.15 else 'sparse'}; "
           f"{'some tasks have release times' if release else 'all tasks are released at 0'}; due times are "
           f"{'tight' if tight else 'loose'} and late_penalty is {lam}.\n"
           "Score: cost of a planted schedule with no late task / your cost (capped at 1); 0 if infeasible.")
    return {"title": "Schedule work orders under deadlines and limits", "doc": doc,
            "keywords": ("schedule", "scheduling", "deadlines", "precedence", "resources", "planning", "tasks"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 4.0,
            "tiers": [("sequential", "MODE = 0\n" + _SOLVER), ("list-greedy", "MODE = 1\n" + _SOLVER),
                      ("due-date-greedy", "MODE = 2\n" + _SOLVER)],
            "rule": rule, "table_key": "tasks", "output_key": "start"}


def _topo(order, preds):
    """Stable precedence-feasible reordering of a priority order."""
    pos = {i: k for k, i in enumerate(order)}
    done, out = set(), []

    def visit(i):
        if i in done:
            return
        done.add(i)
        for p in sorted(preds[i], key=pos.get):
            visit(p)
        out.append(i)
    for i in order:
        visit(i)
    return out
