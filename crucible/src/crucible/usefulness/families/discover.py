"""Content discovery: budgeted coverage selection (specified; value / strong reference).

Choose items to recommend within a budget so that an audience's weighted interests are covered. Each world draws
how fast repeated coverage of a topic loses value (`decay`), whether a channel may contribute only a few items, and
whether items cost the same. The objective is submodular; the evaluator compares against a reference found by
cost-benefit greedy plus swap local search.
"""
from __future__ import annotations

import math

import numpy as np

from ..category import Population, category_family
from ._common import Stream, pick

CHANNELS = ("video", "podcast", "article", "course", "thread")

_HEAD = r'''
def _greedy(items, w, rho, left, cap, by_cost):
    pw = [1.0 - rho] + [(1.0 - rho) * rho ** k for k in range(1, len(items) + 1)]
    topics = [it['topics'].split() for it in items]
    cnt, chan, chosen, value = {}, {}, [], 0.0
    rest = list(range(len(items)))
    while rest:
        best, bi = 0.0, -1
        for i in rest:
            it = items[i]
            if it['cost'] > left or (cap and chan.get(it['channel'], 0) >= cap):
                continue
            g = sum([w[t] * pw[cnt.get(t, 0)] for t in topics[i]])
            key = g / it['cost'] if by_cost else g
            if key > best:
                best, bi, bg = key, i, g
        if bi < 0:
            break
        it = items[bi]
        rest.remove(bi)
        chosen.append(it['id'])
        value += bg
        left -= it['cost']
        chan[it['channel']] = chan.get(it['channel'], 0) + 1
        for t in topics[bi]:
            cnt[t] = cnt.get(t, 0) + 1
    return chosen, value
'''
_BROAD = r'''
def solve(args, caller):
    left, cap, chan, chosen = args['budget'], args['channel_cap'], {}, []
    for it in sorted(args['items'], key=lambda it: (-len(it['topics'].split()), it['id'])):
        if it['cost'] <= left and not (cap and chan.get(it['channel'], 0) >= cap):
            chosen.append(it['id'])
            left -= it['cost']
            chan[it['channel']] = chan.get(it['channel'], 0) + 1
    return {'chosen': chosen}
'''
_GAIN = _HEAD + r'''
def solve(args, caller):
    return {'chosen': _greedy(args['items'], args['interest'], args['decay'], args['budget'],
                              args['channel_cap'], False)[0]}
'''
_BEST = _HEAD + r'''
def solve(args, caller):
    a = _greedy(args['items'], args['interest'], args['decay'], args['budget'], args['channel_cap'], True)
    b = _greedy(args['items'], args['interest'], args['decay'], args['budget'], args['channel_cap'], False)
    return {'chosen': a[0] if a[1] >= b[1] else b[0]}
'''


def _value(M, w, rho, sel) -> float:
    cnt = M[sel].sum(axis=0) if len(sel) else np.zeros(M.shape[1])
    return float((w * (1.0 - rho ** cnt) * (cnt > 0)).sum())


def _reference(M, w, rho, cost, chan, budget, cap) -> float:
    """Best of two greedy rules, then swap / add local search (vectorised over items)."""
    n = len(cost)
    best_sel, best_v = [], 0.0
    for by_cost in (True, False):
        sel, cnt, left, used = [], np.zeros(M.shape[1]), budget, {}
        free = np.ones(n, bool)
        while True:
            gains = M @ (w * (1.0 - rho) * rho ** cnt)
            ok = free & (cost <= left)
            if cap:
                ok &= np.array([used.get(c, 0) < cap for c in chan])
            key = np.where(ok, gains / cost if by_cost else gains, -1.0)
            i = int(key.argmax())
            if key[i] <= 0:
                break
            sel.append(i)
            free[i] = False
            cnt = cnt + M[i]
            left -= int(cost[i])
            used[chan[i]] = used.get(chan[i], 0) + 1
        v = _value(M, w, rho, sel)
        if v > best_v:
            best_sel, best_v = sel, v
    sel = list(best_sel)
    for _ in range(4):
        improved = False
        for pos in range(len(sel)):
            trial = sel[:pos] + sel[pos + 1:]
            left = budget - int(cost[trial].sum()) if trial else budget
            used: dict = {}
            for j in trial:
                used[chan[j]] = used.get(chan[j], 0) + 1
            cnt = M[trial].sum(axis=0) if trial else np.zeros(M.shape[1])
            gains = M @ (w * (1.0 - rho) * rho ** cnt)
            ok = cost <= left
            ok[trial] = False
            if cap:
                ok &= np.array([used.get(c, 0) < cap for c in chan])
            key = np.where(ok, gains, -1.0)
            j = int(key.argmax())
            if key[j] > 0:
                v = _value(M, w, rho, trial + [j])
                if v > best_v + 1e-9:
                    sel, best_v, improved = trial + [j], v, True
                    break
        if not improved:
            break
    return best_v


def _score(inp, hidden, out) -> float:
    ch = out.get("chosen")
    items = {it["id"]: it for it in inp["items"]}
    if not isinstance(ch, list) or len(ch) > len(items):
        return 0.0
    seen, spent, used, cnt = set(), 0, {}, {}
    for x in ch:
        if not isinstance(x, str) or x not in items or x in seen:
            return 0.0
        seen.add(x)
        it = items[x]
        spent += it["cost"]
        used[it["channel"]] = used.get(it["channel"], 0) + 1
        for t in it["topics"].split():
            cnt[t] = cnt.get(t, 0) + 1
    cap = inp["channel_cap"]
    if spent > inp["budget"] or (cap and any(v > cap for v in used.values())):
        return 0.0
    rho, w = inp["decay"], inp["interest"]
    value = sum(w[t] * (1.0 - rho ** k) for t, k in cnt.items())
    return min(1.0, value / hidden) if hidden > 0 else 0.0


@category_family("discover", kind="content discovery", relation="specified", split="val",
                 doc="Budgeted coverage selection with diminishing repeat coverage and channel caps.",
                 population=Population(1.0, (6.0, 30.0), (0.5, 2.5), "editors assembling recommendation slates"),
                 demand_links=("retrieval", "quote"), resource_intensity=1.1)
def build(rng, params):
    rho = float(pick(rng, (0.0, 0.3, 0.5)))
    cap = int(pick(rng, (0, 2, 3)))
    cmax = int(pick(rng, (3, 4, 6)))
    skew = float(rng.uniform(0.5, 1.5))
    rule = {"decay": rho, "channel_cap": cap, "max_cost": cmax, "skew": skew}

    def generate(rng, size):
        st = Stream(rng)
        n = int(8 + 12 * size)
        n_topics = int(10 + 12 * size)
        topics = [f"t{k:02d}" for k in range(n_topics)]
        w = np.array([max(1, round(math.exp(st.normal(1.5, skew)))) for _ in range(n_topics)], np.int64)
        M = np.zeros((n, n_topics))
        cost = np.zeros(n, np.int64)
        chan, items = [], []
        for i in range(n):
            k = st.between(2, 5)
            ts = sorted(st.sample(range(n_topics), k))
            M[i, ts] = 1.0
            cost[i] = 1 if cmax == 1 else int(max(1, min(cmax, round(k * st.uniform(0.4, 1.4)))))
            chan.append(st.pick(CHANNELS))
            items.append({"id": f"c{i:02d}", "cost": int(cost[i]), "channel": chan[-1],
                          "topics": " ".join(topics[t] for t in ts)})
        budget = max(int(cost.max()), int(st.uniform(0.25, 0.4) * cost.sum()))
        ref = _reference(M, w.astype(float), rho, cost, chan, budget, cap)
        return {"budget": budget, "decay": rho, "channel_cap": cap, "items": items,
                "interest": {t: int(x) for t, x in zip(topics, w)}}, ref

    doc = ("Choose items to recommend within a budget so that the audience's interests are covered.\n"
           "Input: {\"budget\": int, \"decay\": number, \"channel_cap\": int, \"items\": [{\"id\": str, \"cost\": int, "
           "\"channel\": str, \"topics\": \"space separated topic ids\"}, ...], \"interest\": {topic: weight}}.\n"
           "Output: {\"chosen\": [item ids]} with total cost <= budget and, when channel_cap > 0, at most "
           "channel_cap items per channel.\n"
           "Value: for every topic covered by k >= 1 chosen items, weight x (1 - decay^k) (with decay 0 a topic "
           "counts once however often it is covered).\n"
           f"In this world decay is {rho}, channel_cap is {cap} and item costs range from 1 to {cmax}.\n"
           "Score: your value / the value of a strong reference selection (capped at 1); 0 if infeasible.")
    return {"title": "Assemble a recommendation slate", "doc": doc,
            "keywords": ("discovery", "recommendation", "coverage", "content", "curation", "selection"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 3.0,
            "tiers": [("broadest-first", _BROAD), ("gain-greedy", _GAIN), ("cost-benefit-greedy", _BEST)],
            "rule": rule, "table_key": "items", "output_key": "chosen"}
