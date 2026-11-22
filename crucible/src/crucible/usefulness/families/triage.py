"""Ticket triage: route support tickets to queues under a hidden decision policy (exemplified; accuracy).

Each world draws its own queues and an ordered routing policy over ticket attributes (which channel, product and
plan a ticket came through, how old it is, how many errors it reports, whether it has an attachment). A small share
of tickets is escalated for reasons no attribute explains. Published examples show tickets with their queues.
"""
from __future__ import annotations

import numpy as np

from ..category import Population, category_family
from ._common import Stream, pick

QUEUES = ("billing", "outage", "onboarding", "abuse", "hardware", "legal", "retention", "bugs")
CHANNELS = ("email", "chat", "phone", "portal")
PRODUCTS = ("ledger", "relay", "vault", "canvas", "pulse")
PLANS = ("free", "pro", "ent")

_RULES = r'''
def solve(args, caller):
    out = []
    for t in args['tickets']:
        q = DEFAULT
        for conds, queue in RULES:
            ok = True
            for f, op, v in conds:
                x = t.get(f)
                if (op == '==' and x != v) or (op == '>' and not (x is not None and x > v)) or \
                        (op == '<=' and not (x is not None and x <= v)):
                    ok = False
                    break
            if ok:
                q = queue
                break
        out.append(q)
    return {'queue': out}
'''


def _solver(rules, default) -> str:
    return f"RULES = {rules!r}\nDEFAULT = {default!r}\n" + _RULES


def _route(t: dict, rules, default: str) -> str:
    for conds, queue in rules:
        if all((op == "==" and t[f] == v) or (op == ">" and t[f] > v) or (op == "<=" and t[f] <= v)
               for f, op, v in conds):
            return queue
    return default


def _score(inp, labels, out) -> float:
    q = out.get("queue")
    if not isinstance(q, list) or len(q) != len(labels):
        return 0.0
    return sum(1 for a, b in zip(q, labels) if isinstance(a, str) and a == b) / len(labels)


def _ticket(st, i: int) -> dict:
    return {"id": f"t{i:02d}", "channel": st.pick(CHANNELS), "product": st.pick(PRODUCTS), "plan": st.pick(PLANS),
            "age_h": int(st.exponential(30)), "errors": int(st.exponential(2.5)),
            "words": int(20 + st.exponential(120)), "attach": st.chance(0.35)}


@category_family("triage", kind="classification / routing", relation="exemplified", split="test",
                 doc="Route tickets under a hidden decision policy; accuracy.",
                 population=Population(0.9, (10.0, 40.0), (0.5, 2.5), "support desks routing incoming tickets"),
                 demand_links=("coord", "netrel"), resource_intensity=0.6)
def build(rng, params):
    queues = pick(rng, QUEUES, 4)
    default = queues[3]
    # conditions that hold for a sizeable share of tickets, so every queue gets real traffic
    broad = [["channel", "==", c] for c in CHANNELS] + [["product", "==", c] for c in PRODUCTS] + \
            [["plan", "==", c] for c in PLANS] + [["age_h", ">", 24], ["age_h", "<=", 12], ["errors", ">", 3],
                                                  ["errors", "<=", 0], ["words", ">", 120], ["words", "<=", 60],
                                                  ["attach", "==", True]]
    idx = [int(x) for x in rng.permutation(len(broad))]
    rules = []
    for k in range(3):                                       # three broad rules, one per queue
        rules.append([[broad[idx[k]]], queues[k]])
    for k in range(int(rng.integers(3, 6))):                 # refinements: conjunctions, any queue
        a, b = broad[idx[3 + 2 * k]], broad[idx[4 + 2 * k]]
        conds = [a] if a[0] == b[0] else [a, b]
        rules.insert(int(rng.integers(0, len(rules))), [conds, pick(rng, queues)])
    noise = 0.04

    def generate(rng, size):
        st = Stream(rng)
        tickets, labels = [], []
        for i in range(int(10 + 22 * size)):
            t = _ticket(st, i)
            q = _route(t, rules, default)
            if st.chance(noise):
                q = st.pick(queues)
            tickets.append(t)
            labels.append(q)
        return {"tickets": tickets, "queues": sorted(queues)}, labels

    # reference tiers: one queue for everything; the first two rules; the whole policy
    probe = Stream(np.random.default_rng([int(rng.integers(0, 2**31)), 23]))
    sample = [_ticket(probe, i) for i in range(500)]
    routed = [_route(t, rules, default) for t in sample]

    def majority(labels):
        return max(sorted(set(labels) | {default}), key=labels.count)
    rest = [q for t, q in zip(sample, routed) if _route(t, rules[:2], "") == ""]
    doc = ("Route support tickets to the right queue.\n"
           "Input: {\"tickets\": [{\"id\": str, \"channel\": str, \"product\": str, \"plan\": str, \"age_h\": int, "
           "\"errors\": int, \"words\": int, \"attach\": bool}, ...], \"queues\": [queue names]}.\n"
           "Output: {\"queue\": [one queue name per ticket, in order]}.\n"
           "Score: the fraction of tickets sent to the queue this world's desk sends them to. The desk's routing "
           "policy is not documented.\nPublished worked examples show tickets with their queues.")
    return {"title": "Route support tickets", "doc": doc,
            "keywords": ("triage", "tickets", "routing", "support", "classification", "queue"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 2.0,
            "tiers": [("one-queue", _solver([], majority(routed))), ("two-rules", _solver(rules[:2], majority(rest))),
                      ("full-policy", _solver(rules, default))],
            "truth": lambda inp, labels: {"queue": list(labels)},
            "rule": {"rules": rules, "default": default, "queues": queues, "noise": noise},
            "table_key": "tickets", "output_key": "queue"}
