"""Price quoting under hidden demand (experiential; expected profit / best expected profit).

A request is a batch of offers to price. Each offer belongs to a customer segment, has a unit cost and a list
price. The documentation gives a nominal market model (the same willingness to pay for every segment). In each
world the segments really differ - some pay far above list, some walk away well below it, some are much more
price sensitive - and nothing states it: it can only be learned from how quotes turn out.
"""
from __future__ import annotations

import math

from ..category import Population, category_family
from ._common import Stream, is_num, pick, r3

SEGMENTS = ("smb", "enterprise", "education", "public", "retail", "startup", "agency", "nonprofit")
NOMINAL = (1.0, 6.0)        # documented: purchase probability is 1/2 at the list price, steepness 6

_COST_PLUS = r'''
def solve(args, caller):
    return {'prices': [round(o['cost'] * 1.3, 2) for o in args['offers']]}
'''
_MODEL = r'''
import math


def solve(args, caller):
    out = []
    for o in args['offers']:
        m, k = SEG.get(o['segment'], DEFAULT)
        c, L = o['cost'], o['list']
        best, bp = -1.0, c
        for j in range(1, 33):
            p = c + (2.6 * L - c) * j / 32.0
            v = (p - c) / (1.0 + math.exp(min(60.0, k * (p / L - m))))
            if v > best:
                best, bp = v, p
        out.append(round(bp, 2))
    return {'prices': out}
'''


def _model(seg: dict) -> str:
    return f"SEG = {seg!r}\nDEFAULT = {list(NOMINAL)!r}\n" + _MODEL


def _profit(p: float, c: float, L: float, m: float, k: float) -> float:
    return (p - c) / (1.0 + math.exp(min(60.0, k * (p / L - m))))


def _best(c: float, L: float, m: float, k: float) -> float:
    lo, hi = c, 3.0 * L
    for _ in range(40):                       # the profit curve is unimodal above cost: ternary search
        a, b = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if _profit(a, c, L, m, k) < _profit(b, c, L, m, k):
            lo = a
        else:
            hi = b
    return _profit((lo + hi) / 2.0, c, L, m, k)


def _score(inp, hidden, out) -> float:
    prices, offers = out.get("prices"), inp["offers"]
    if not isinstance(prices, list) or len(prices) != len(offers):
        return 0.0
    total = 0.0
    for p, o, (m, k, best) in zip(prices, offers, hidden[0]):
        if not is_num(p) or p < 0 or p > 100 * o["list"]:
            return 0.0
        total += _profit(float(p), o["cost"], o["list"], m, k)
    return total / hidden[1] if hidden[1] > 0 else 0.0


@category_family("quote", kind="pricing under uncertain demand", relation="experiential", split="train",
                 doc="Quote prices under hidden per-segment demand; expected profit relative to the best.",
                 population=Population(0.9, (6.0, 30.0), (0.7, 3.0), "sales desks quoting custom orders"),
                 demand_links=("allocate", "discover"), resource_intensity=0.9)
def build(rng, params):
    segments = pick(rng, SEGMENTS, int(rng.integers(3, 5)))
    mids = [1.6, 0.6, 1.15, 0.85][:len(segments)]
    order = [int(x) for x in rng.permutation(len(segments))]
    true = {s: [r3(mids[order[i]] * rng.uniform(0.9, 1.1)), r3(rng.uniform(3.0, 12.0))]
            for i, s in enumerate(segments)}
    est = {s: [r3(m * math.exp(rng.normal(0.0, 0.06))), r3(k * math.exp(rng.normal(0.0, 0.2)))]
           for s, (m, k) in true.items()}
    rule = {"segments": true}

    def generate(rng, size):
        st = Stream(rng, 128)
        offers, hid, total = [], [], 0.0
        for i in range(int(6 + 14 * size)):
            seg = st.pick(segments)
            L = st.between(40, 899)
            c = int(max(1, L * st.uniform(0.3, 0.7)))
            m, k = true[seg]
            best = _best(c, L, m, k)
            offers.append({"id": f"q{i:02d}", "segment": seg, "cost": c, "list": L})
            hid.append((m, k, best))
            total += best
        return {"offers": offers}, (hid, total)

    doc = ("Quote a price for every offer.\n"
           "Input: {\"offers\": [{\"id\": str, \"segment\": str, \"cost\": int, \"list\": int}, ...]}.\n"
           "Output: {\"prices\": [one price per offer, in order]}.\n"
           "A customer buys with a probability that falls as the price rises; you earn (price - cost) when they "
           "buy. Nominal market model: probability 1 / (1 + exp(6 x (price / list - 1))) for every segment "
           f"({', '.join(sorted(segments))}). The model is nominal: how each segment really responds to price in "
           "this world is not documented and shows only in how quotes turn out.\n"
           "Score: your expected profit / the best attainable expected profit, over the batch.")
    return {"title": "Quote prices for custom orders", "doc": doc,
            "keywords": ("pricing", "quotes", "revenue", "demand", "segments", "margin"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 3.0,
            "tiers": [("cost-plus", _COST_PLUS), ("nominal-model", _model({})), ("field-experience", _model(est))],
            "rule": rule, "table_key": "offers", "output_key": "prices"}
