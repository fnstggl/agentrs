"""Data quality / entity resolution under a generated corruption process (exemplified; scored by pairwise F1).

A request is a list of customer records; several records can describe the same real entity, and the copies were
damaged by this world's corruption process. Which evidence can be trusted is the hidden rule: in one world phone
numbers survive intact and identify a customer, in another they are shared switchboard numbers that prove nothing
while names are reliable; names can be common or rare; years can be exact or jittered. Every world also has its
own nicknames for first names, abbreviations for cities and a dialling prefix that some sources put in front of
phone numbers: records match only once those are undone. Published examples show record lists with their true
clusters.
"""
from __future__ import annotations

import math

import numpy as np

from ..category import Population, category_family
from ._common import CITIES, FIRST, LAST, Stream, pick

NICKS = ("bo", "cy", "dee", "eli", "fay", "gus", "hal", "ivy", "jo", "kit", "lou", "max", "ned", "oz", "pip", "ray",
         "sid", "tex", "uma", "val", "wes", "xan", "yan", "zed")
LEVELS = {"name": ("eq", "near", "ne"), "phone": ("eq", "near", "ne", "na"), "city": ("eq", "ne", "na"),
          "year": ("eq", "near", "ne", "na")}
GENERIC = {"name": {"eq": 4.0, "near": 2.0, "ne": -3.0}, "phone": {"eq": 5.0, "near": 2.0, "ne": -2.0, "na": 0.0},
           "city": {"eq": 0.5, "ne": -1.0, "na": 0.0}, "year": {"eq": 1.0, "near": 0.5, "ne": -2.0, "na": 0.0}}
GENERIC_T = 4.0

# The feature code is shared, as source text, by the evaluator-side fit and by the reference solvers.
_FEATURES = r'''
def _prep(r, alias):
    nick = alias.get('nick', {})
    s = ' '.join(sorted(nick.get(t, t) for t in (r.get('name') or '').split()))
    ph = r.get('phone')
    if ph:
        ph = ''.join(c for c in ph if c.isdigit())
        pre = alias.get('prefix')
        if pre and len(ph) > 7 and ph.startswith(pre):
            ph = ph[len(pre):]
    city = r.get('city')
    return (frozenset(s[i:i + 2] for i in range(len(s) - 1)), ph or None,
            alias.get('city', {}).get(city, city) if city else None, r.get('year'))


def _levels(a, b):
    ga, gb = a[0], b[0]
    u = len(ga | gb)
    j = len(ga & gb) / u if u else 0.0
    name = 'eq' if j > 0.99 else 'near' if j >= 0.5 else 'ne'
    pa, pb = a[1], b[1]
    if pa is None or pb is None:
        phone = 'na'
    elif pa == pb:
        phone = 'eq'
    else:
        phone = 'near' if len(pa) == len(pb) and sum(x != y for x, y in zip(pa, pb)) <= 1 else 'ne'
    city = 'na' if a[2] is None or b[2] is None else 'eq' if a[2] == b[2] else 'ne'
    ya, yb = a[3], b[3]
    year = 'na' if ya is None or yb is None else 'eq' if ya == yb else 'near' if abs(ya - yb) <= 1 else 'ne'
    return name, phone, city, year
'''
_SOLVER = r'''
def solve(args, caller):
    recs = args['records']
    n = len(recs)
    prep = [_prep(r, ALIAS) for r in recs]
    wn, wp, wc, wy = W['name'], W['phone'], W['city'], W['year']
    parent = list(range(n))
    for i in range(n):
        a = prep[i]
        for j in range(i + 1, n):
            l = _levels(a, prep[j])
            if wn[l[0]] + wp[l[1]] + wc[l[2]] + wy[l[3]] >= T:
                ri, rj = i, j
                while parent[ri] != ri:
                    ri = parent[ri]
                while parent[rj] != rj:
                    rj = parent[rj]
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        groups.setdefault(r, []).append(recs[i]['id'])
    return {'clusters': [g for g in groups.values()]}
'''
_EXACT = r'''
def solve(args, caller):
    groups = {}
    for r in args['records']:
        ph = r.get('phone')
        key = (' '.join(sorted((r.get('name') or '').split())), ''.join(c for c in ph if c.isdigit()) if ph else None)
        groups.setdefault(key, []).append(r['id'])
    return {'clusters': list(groups.values())}
'''
_NS: dict = {}
exec(compile(_FEATURES, "<dedupe-features>", "exec"), _NS)
_prep, _levels = _NS["_prep"], _NS["_levels"]


def _solver(weights: dict, threshold: float, alias: dict) -> str:
    return f"W = {weights!r}\nT = {threshold!r}\nALIAS = {alias!r}\n" + _FEATURES + _SOLVER


def _pairs(clusters) -> set:
    out = set()
    for c in clusters:
        c = sorted(c)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                out.add((c[i], c[j]))
    return out


def _score(inp, truth, out) -> float:
    cl = out.get("clusters")
    ids = {r["id"] for r in inp["records"]}
    if not isinstance(cl, list) or len(cl) > len(ids):
        return 0.0
    seen = set()
    for c in cl:
        if not isinstance(c, list):
            return 0.0
        for x in c:
            if not isinstance(x, str) or x not in ids or x in seen:
                return 0.0
            seen.add(x)
    pred, gold = _pairs(cl), truth
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _typo(st, s: str) -> str:
    i = st.below(len(s))
    c = "abcdefghijklmnopqrstuvwxyz"[st.below(26)] if s[i].isalpha() else str(st.below(10))
    return s[:i] + c + s[i + 1:]


def _fmt_phone(st, d: str) -> str:
    return (d, f"{d[:3]}-{d[3:]}", f"({d[:3]}) {d[3:]}", f"{d[:3]} {d[3:]}")[st.below(4)]


@category_family("dedupe", kind="data quality / entity resolution", relation="exemplified", split="train",
                 doc="Entity resolution under a generated corruption process; pairwise F1.",
                 population=Population(0.8, (4.0, 20.0), (0.7, 3.0), "operations teams merging customer lists"),
                 demand_links=("allocate", "quote"), resource_intensity=1.2)
def build(rng, params):
    # ---- the hidden corruption process of this world
    regime = pick(rng, ("phone_true", "name_true", "mixed"))
    homonyms = bool(rng.random() < 0.5) if regime != "name_true" else False        # few distinct names
    shared_phone = {"phone_true": 0.0, "name_true": float(rng.uniform(0.5, 0.8)),
                    "mixed": float(rng.uniform(0.15, 0.35))}[regime]
    p_name = {"phone_true": float(rng.uniform(0.5, 0.8)), "name_true": float(rng.uniform(0.05, 0.2)),
              "mixed": float(rng.uniform(0.3, 0.5))}[regime]
    p_phone = {"phone_true": float(rng.uniform(0.0, 0.08)), "name_true": float(rng.uniform(0.3, 0.6)),
               "mixed": float(rng.uniform(0.15, 0.3))}[regime]
    p_year = float(rng.uniform(0.05, 0.5))
    year_span = int(pick(rng, (6, 25, 50)))                   # a narrow span makes equal years uninformative
    cities = pick(rng, CITIES, int(rng.integers(3, 7)))
    alias = {c: "".join(w[0] for w in c.split()) + c.split()[-1][-2:] for c in pick(rng, cities, 2)}
    p_alias = float(rng.uniform(0.3, 0.7))
    p_missing = float(rng.uniform(0.03, 0.12))
    rule = {"regime": regime, "homonyms": homonyms, "shared_phone": shared_phone, "p_name": p_name,
            "p_phone": p_phone, "p_year": p_year, "year_span": year_span, "alias": alias, "p_alias": p_alias,
            "p_missing": p_missing}
    nick_of = dict(zip(pick(rng, FIRST, 10), pick(rng, NICKS, 10)))      # this world's nicknames
    p_nick = float(rng.uniform(0.35, 0.7))
    prefix = str(int(rng.integers(200, 990)))                            # this world's dialling prefix
    p_prefix = float(rng.uniform(0.35, 0.7))
    rule.update({"nicknames": nick_of, "p_nick": p_nick, "prefix": prefix, "p_prefix": p_prefix})
    canon = {"city": {v: k for k, v in alias.items()}, "nick": {v: k for k, v in nick_of.items()}, "prefix": prefix}

    def generate(rng, size):
        st = Stream(rng)
        n = int(12 + 14 * size)
        firsts = st.sample(FIRST, 5 if homonyms else 16)
        lasts = st.sample(LAST, 4 if homonyms else 16)
        ents, phones, total = [], [], 0
        while total < n:
            ph = f"{st.below(10_000_000):07d}"
            if phones and st.chance(shared_phone):
                ph = st.pick(phones)
            phones.append(ph)
            u = st.u()
            k = 1 if u < 0.35 else 2 if u < 0.7 else 3 if u < 0.9 else 4
            ents.append((f"{st.pick(firsts)} {st.pick(lasts)}", ph, st.pick(cities), 1960 + st.below(year_span), k))
            total += k
        recs, owner = [], []
        for e_i, (e_name, e_phone, e_city, e_year, k) in enumerate(ents):
            for c in range(k):
                name, phone, city, year = e_name, e_phone, e_city, e_year
                first, last = name.split()
                if first in nick_of and st.chance(p_nick):   # any record may use this world's nickname
                    name = f"{nick_of[first]} {last}"
                if c > 0:                                    # every copy after the first went through the process
                    if st.chance(p_name):
                        first, last = name.split()
                        how = st.below(3)
                        name = f"{first[0]} {last}" if how == 0 else f"{_typo(st, first)} {last}" if how == 1 else \
                            f"{first} {_typo(st, last)}"
                    if st.chance(0.3):
                        name = " ".join(reversed(name.split()))
                    if st.chance(p_phone):
                        phone = _typo(st, phone) if st.chance(0.5) else f"{st.below(10_000_000):07d}"
                    if st.chance(p_year):
                        year = year + (1 if st.chance(0.5) else -1)
                if city in alias and st.chance(p_alias):
                    city = alias[city]
                shown = _fmt_phone(st, phone)
                if st.chance(p_prefix):
                    shown = f"{prefix} {shown}"
                recs.append({"name": name, "city": None if st.chance(p_missing) else city,
                             "phone": None if st.chance(p_missing) else shown,
                             "year": None if st.chance(p_missing) else int(year)})
                owner.append(e_i)
        order = st.shuffled(range(len(recs)))
        recs = [{"id": f"r{k:02d}", **recs[j]} for k, j in enumerate(order)]
        owner = [owner[j] for j in order]
        groups: dict[int, list] = {}
        for r, o in zip(recs, owner):
            groups.setdefault(o, []).append(r["id"])
        gold = _pairs(groups.values())
        if not gold:                                         # no duplicates at all: plant one exact copy
            recs.append({**recs[0], "id": f"r{len(recs):02d}"})
            gold = {(recs[0]["id"], recs[-1]["id"])}
        return {"records": recs}, frozenset(gold)

    # ---- the best reference tier is fitted on labelled data from this world (as an experienced firm would)
    counts = {f: {lv: [1.0, 1.0] for lv in LEVELS[f]} for f in LEVELS}
    fit_rng = np.random.default_rng([int(rng.integers(0, 2**31)), 17])
    sample = []
    for _ in range(6):
        inp, gold = generate(fit_rng, 0.8)
        recs = inp["records"]
        prep = [_prep(r, canon) for r in recs]
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                lv = _levels(prep[i], prep[j])
                a, b = recs[i]["id"], recs[j]["id"]
                m = ((a, b) if a < b else (b, a)) in gold
                sample.append((lv, m))
                for f, x in zip(("name", "phone", "city", "year"), lv):
                    counts[f][x][0 if m else 1] += 1.0
    n_m = sum(1 for _, m in sample if m) + 2.0
    n_u = len(sample) - n_m + 4.0
    weights = {f: {lv: round(math.log((c[0] / n_m) / (c[1] / n_u)), 3) for lv, c in counts[f].items()} for f in counts}
    scored = sorted(((sum(weights[f][x] for f, x in zip(("name", "phone", "city", "year"), lv)), m)
                     for lv, m in sample), key=lambda t: -t[0])
    best_f1, threshold, tp = -1.0, 0.0, 0
    total_m = sum(1 for _, m in sample if m)
    for k, (s, m) in enumerate(scored):
        tp += m
        f1 = 2 * tp / (k + 1 + total_m)
        if f1 > best_f1 and (k + 1 == len(scored) or scored[k + 1][0] < s):
            best_f1, threshold = f1, s
    threshold = round(threshold - 1e-6, 4)

    def truth(inp, gold):
        parent = {r["id"]: r["id"] for r in inp["records"]}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for a, b in sorted(gold):
            parent[find(b)] = find(a)
        groups: dict[str, list] = {}
        for r in inp["records"]:
            groups.setdefault(find(r["id"]), []).append(r["id"])
        return {"clusters": [g for _, g in sorted(groups.items())]}

    doc = ("Find the records that describe the same customer.\n"
           "Input: {\"records\": [{\"id\": str, \"name\": str, \"city\": str|null, \"phone\": str|null, "
           "\"year\": int|null}, ...]}. Several records can be damaged copies of one customer; different customers "
           "can share a name, a phone number or a city. Which fields survive copying, which coincide by chance, "
           "and which nicknames, abbreviations and dialling prefixes this world's sources use are not documented.\n"
           "Output: {\"clusters\": [[record ids of one customer], ...]} (records you leave out are singletons).\n"
           "Score: F1 over pairs of records placed in the same cluster, against the true customers.\n"
           "Published worked examples show record lists with their true clusters.")
    return {"title": "Merge duplicate customer records", "doc": doc,
            "keywords": ("dedupe", "duplicates", "entity", "resolution", "records", "matching", "data-quality"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 4.0,
            "tiers": [("same-name-and-phone", _EXACT), ("generic-weights", _solver(GENERIC, GENERIC_T, {})),
                      ("fitted-weights", _solver(weights, threshold, canon))],
            "truth": truth, "rule": {**rule, "weights": weights, "threshold": threshold},
            "table_key": "records", "output_key": "clusters"}
