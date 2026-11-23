"""Resource allocation: generated concave returns under a budget (specified; value / exact optimum).

A request lists options with diminishing returns of documented shapes, a unit cost and a cap, and a budget. Each
world draws which return shapes occur, how unequal the options are, how lumpy unit costs are and how tight the
budget is. The evaluator knows the exact optimum (a dynamic programme over the budget).
"""
from __future__ import annotations

import math

import numpy as np

from ..category import Population, category_family
from ._common import Stream, is_int, pick, r3

_HEAD = r'''
import heapq
import math


def _f(k, a, b, x):
    if k == 'log':
        return a * math.log(1.0 + b * x)
    if k == 'sat':
        return a * (1.0 - math.exp(-b * x))
    return a * math.sqrt(b * x)
'''
_IN_ORDER = r'''
def solve(args, caller):
    left = args['budget']
    x = []
    for o in args['options']:
        k = min(o['cap'], left // o['cost'])
        x.append(k)
        left -= k * o['cost']
    return {'units': x}
'''
_EQUAL = r'''
def solve(args, caller):
    opts = args['options']
    left = args['budget']
    x = [0] * len(opts)
    moved = True
    while moved:                      # one unit to every option in turn, for as long as the budget lasts
        moved = False
        for i, o in enumerate(opts):
            if x[i] < o['cap'] and o['cost'] <= left:
                x[i] += 1
                left -= o['cost']
                moved = True
    return {'units': x}
'''
_GREEDY = _HEAD + r'''
def solve(args, caller):
    opts = args['options']
    left = args['budget']
    x = [0] * len(opts)
    heap = [(-_f(o['kind'], o['a'], o['b'], 1) / o['cost'], i) for i, o in enumerate(opts)
            if o['cap'] > 0 and o['cost'] <= left]
    heapq.heapify(heap)
    while heap:
        _, i = heapq.heappop(heap)
        o = opts[i]
        c = o['cost']
        if c > left:
            continue
        x[i] += 1
        left -= c
        if x[i] < o['cap'] and c <= left:
            k, a, b = o['kind'], o['a'], o['b']
            heapq.heappush(heap, (-(_f(k, a, b, x[i] + 1) - _f(k, a, b, x[i])) / c, i))
    return {'units': x}
'''


def _f(k: str, a: float, b: float, x: int) -> float:
    if k == "log":
        return a * math.log(1.0 + b * x)
    if k == "sat":
        return a * (1.0 - math.exp(-b * x))
    return a * math.sqrt(b * x)


def _optimum(options: list[dict], budget: int) -> float:
    best = np.zeros(budget + 1)
    for o in options:
        new = best.copy()
        for x in range(1, min(o["cap"], budget // o["cost"]) + 1):
            c = o["cost"] * x
            np.maximum(new[c:], best[:budget + 1 - c] + _f(o["kind"], o["a"], o["b"], x), out=new[c:])
        best = new
    return float(best[budget])


def _score(inp, opt, out) -> float:
    x, opts = out.get("units"), inp["options"]
    if not isinstance(x, list) or len(x) != len(opts):
        return 0.0
    spent, value = 0, 0.0
    for k, o in zip(x, opts):
        if not is_int(k) or k < 0 or k > o["cap"]:
            return 0.0
        spent += k * o["cost"]
        value += _f(o["kind"], o["a"], o["b"], k)
    if spent > inp["budget"]:
        return 0.0
    return value / opt if opt > 0 else 0.0


@category_family("allocate", kind="resource allocation", relation="specified", split="train",
                 doc="Generated concave returns under a budget; value relative to the exact optimum.",
                 population=Population(0.9, (4.0, 20.0), (0.8, 3.5), "budget owners splitting spend across options"),
                 demand_links=("schedule", "quote"), resource_intensity=0.8)
def build(rng, params):
    kinds = pick(rng, ("log", "sat", "root"), int(rng.integers(2, 4)))
    sigma = float(rng.uniform(0.6, 1.3))
    cmax = int(pick(rng, (1, 3, 5)))
    caps = (3, 9) if rng.random() < 0.5 else (6, 18)
    frac = (float(rng.uniform(0.12, 0.2)), float(rng.uniform(0.25, 0.4)))
    rule = {"kinds": kinds, "sigma": sigma, "max_cost": cmax, "caps": list(caps), "budget_fraction": list(frac)}

    def generate(rng, size):
        st = Stream(rng, 128)
        options = [{"id": f"o{i:02d}", "kind": st.pick(kinds), "a": r3(max(0.5, math.exp(st.normal(2.0, sigma)))),
                    "b": r3(st.uniform(0.05, 1.2)), "cost": st.between(1, cmax), "cap": st.between(caps[0], caps[1])}
                   for i in range(int(6 + 10 * size))]
        budget = max(cmax, int(st.uniform(*frac) * sum(o["cost"] * o["cap"] for o in options)))
        return {"budget": budget, "options": options}, _optimum(options, budget)

    shapes = {"log": "log: a x ln(1 + b x units)", "sat": "sat: a x (1 - exp(-b x units))",
              "root": "root: a x sqrt(b x units)"}
    doc = ("Split a budget across options with diminishing returns.\n"
           "Input: {\"budget\": int, \"options\": [{\"id\": str, \"kind\": str, \"a\": number, \"b\": number, "
           "\"cost\": int, \"cap\": int}, ...]}.\n"
           "Output: {\"units\": [integer units bought of every option, in order]} with 0 <= units <= cap and "
           "sum(units x cost) <= budget.\n"
           "Return of an option by kind: " + "; ".join(shapes[k] for k in sorted(kinds)) + ".\n"
           f"In this world unit costs range from 1 to {cmax} and caps from {caps[0]} to {caps[1]}.\n"
           "Score: your total return / the optimal total return; 0 if over budget or over a cap.")
    return {"title": "Allocate a budget across options", "doc": doc,
            "keywords": ("allocation", "budget", "returns", "portfolio", "spend", "optimisation"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 3.0,
            "tiers": [("in-order", _IN_ORDER), ("even-spread", _EQUAL), ("marginal-greedy", _GREEDY)],
            "rule": rule, "table_key": "options", "output_key": "units"}
