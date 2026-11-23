"""Scripted reference agents for generated categories (scientific controls, not privileged actors).

They use only public tools and public information: `use.categories`, `use.examples`, the market, search. Nothing
here imports a category family or reads a hidden rule.

* `ExampleInductor`  knows one piece of developer craft - how to induce a record wire format from worked examples -
                     and nothing about any particular world. It looks for an exemplified category whose examples
                     its routine can explain, writes a solver from what it induced, tests it on examples it held
                     back, then commits, deploys on rented compute, lists and operates the service.
* `GarbageSeller`    deploys a service whose output is well-formed and worthless, behind an attractive listing.
* `Shopper`          an independent customer: finds a service through search, picks by verified reviews and calls
                     it with a request of its own.
"""
from __future__ import annotations

import json
import re

from ..agents.scripted import ScriptPolicy, call, ok

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")

# ---------------------------------------------------------------------------------------------- induction
# Candidate value decoders (name -> python expression over the raw string v). A candidate is kept for a (feed,
# type) only if it reproduces every worked example of that feed and type.
CANDIDATES = {
    "s": {"same": "v", "strip": "v.strip()", "unquote": "v[1:-1]", "lower": "v.lower()", "reverse": "v[::-1]"},
    "i": {"int": "int(v)", "sep,": "int(v.replace(',', ''))", "sep.": "int(v.replace('.', ''))",
          "sep_": "int(v.replace('_', ''))", "sep ": "int(v.replace(' ', ''))", "hex": "int(v, 16)",
          "paren": "(-int(v[1:-1]) if v[:1] == '(' else int(v))", "milli": "int(v[:-1]) // 1000"},
    "f": {"float": "float(v)", "comma": "float(v.replace(',', '.'))", "percent": "round(float(v[:-1]) / 100.0, 3)",
          "permille": "int(v[:-2]) / 1000.0"},
    "d": {"iso": "v", "dmy": "v[6:10] + '-' + v[3:5] + '-' + v[0:2]", "mdy": "v[6:10] + '-' + v[0:2] + '-' + v[3:5]",
          "compact": "v[0:4] + '-' + v[4:6] + '-' + v[6:8]"},
}


def _type_of(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "b"
    if isinstance(v, int):
        return "i"
    if isinstance(v, float):
        return "f"
    return "d" if _DATE.match(v) else "s"


def _layout(pairs: list) -> tuple[str, str] | None:
    """(layout, field separator) of a feed: an object, a list of triples, or one delimited line of key=value
    fields whose separator is whichever character splits every example row into its fields."""
    data = pairs[0][0]
    if isinstance(data, dict):
        return "dict", ""
    if isinstance(data, list):
        return "triples", ""
    if isinstance(data, str):
        for sep in sorted(set(c for d, _ in pairs for c in d if not c.isalnum() and c not in " =")):
            if all(len(d.split(sep)) == len(rec) and all("=" in p for p in d.split(sep)) for d, rec in pairs):
                return "line", sep
    return None


def _split_fields(layout: str, sep: str, data) -> list[tuple[str, str, str]]:
    """[(raw key, type tag, raw value)] of one raw record."""
    if layout == "triples":
        return [(k, t, v) for k, t, v in data]
    items = list(data.items()) if layout == "dict" else [p.split("=", 1) for p in data.split(sep)]
    return [(kt[:-2], kt[-1], v) for kt, v in items]


def induce_wire_formats(examples: list[dict]) -> dict | None:
    """Induce, per feed, how records are laid out, how keys are mangled, which tag marks which type and how each
    type is written. Returns a spec for `wire_solver_source`, or None if the examples are not of this kind."""
    feeds: dict[str, list] = {}
    try:
        for ex in examples:
            for row, rec in zip(ex["input"]["rows"], ex["output"]["records"]):
                feeds.setdefault(row["src"], []).append((row["data"], rec))
    except (KeyError, TypeError):
        return None
    spec = {}
    for src, pairs in sorted(feeds.items()):
        got = _layout(pairs)
        if got is None:
            return None
        layout, sep = got
        try:
            rows = [(_split_fields(layout, sep, data), rec) for data, rec in pairs]
        except (ValueError, TypeError, AttributeError, IndexError):
            return None
        # key mangling: a prefix of some length, then either a case change or a reversal
        key_rule = None
        for cut in range(0, 6):
            for how in ("lower", "reverse"):
                def canon(k, cut=cut, how=how):
                    k = k[cut:]
                    return k[::-1] if how == "reverse" else k.lower()
                if all({canon(k) for k, _, _ in fields} == set(rec) for fields, rec in rows):
                    key_rule = (cut, how)
                    break
            if key_rule:
                break
        if key_rule is None:
            return None
        cut, how = key_rule
        tag_type: dict[str, str] = {}
        seen: dict[str, list] = {t: [] for t in "sifd"}
        true_tok, null_tok = None, None
        for fields, rec in rows:
            for k, tag, v in fields:
                k = k[cut:]
                val = rec[k[::-1] if how == "reverse" else k.lower()]
                ty = _type_of(val)
                if ty is None:
                    null_tok = v
                    continue
                tag_type[tag] = ty
                if ty == "b":
                    true_tok = v if val else true_tok
                else:
                    seen[ty].append((v, val))
        dec = {}
        for ty, cases in seen.items():
            for name, expr in CANDIDATES[ty].items():
                fn = eval("lambda v: " + expr)      # noqa: S307 - our own candidate expressions
                try:
                    if all(fn(v) == val and type(fn(v)) is type(val) for v, val in cases):
                        dec[ty] = name
                        break
                except (ValueError, IndexError, TypeError):
                    continue
            else:
                return None
        spec[src] = {"layout": layout, "sep": sep, "cut": cut, "key": how, "types": tag_type, "dec": dec,
                     "true": true_tok, "null": null_tok}
    return spec


def wire_solver_source(spec: dict) -> str:
    """A service that decodes the induced wire formats (stdlib only; `solve(args, caller)`)."""
    table = {ty: {name: f"lambda v: {expr}" for name, expr in c.items()} for ty, c in CANDIDATES.items()}
    lines = ["SPEC = " + repr(spec), "DEC = {"]
    for ty in sorted(table):
        lines.append(f"    {ty!r}: {{" + ", ".join(f"{n!r}: {src}" for n, src in sorted(table[ty].items())) + "},")
    lines.append("}")
    return "\n".join(lines) + '''


def _fields(p, data):
    if p['layout'] == 'triples':
        return [(k, t, v) for k, t, v in data]
    if p['layout'] == 'dict':
        items = list(data.items())
    else:
        items = [f.split('=', 1) for f in data.split(p['sep'])] if data else []
    return [(kt[:-2], kt[-1], v) for kt, v in items]


def solve(args, caller):
    out = []
    for row in args['rows']:
        p = SPEC.get(row.get('src'))
        if p is None:
            out.append({})
            continue
        rec = {}
        for k, tag, v in _fields(p, row['data']):
            k = k[p['cut']:]
            k = k[::-1] if p['key'] == 'reverse' else k.lower()
            ty = p['types'].get(tag)
            if v == p['null'] or ty is None:
                rec[k] = None
            elif ty == 'b':
                rec[k] = v == p['true']
            else:
                try:
                    rec[k] = DEC[ty][p['dec'][ty]](v)
                except (ValueError, IndexError):
                    rec[k] = None
        out.append(rec)
    return {'records': out}
'''


def service_repo(source: str, title: str, doc: str, price: int) -> dict:
    manifest = {"entry": "service.py", "description": title,
                "endpoints": {"solve": {"handler": "solve", "doc": doc[:300], "price_cents": int(price),
                                        "params": {"type": "object", "additionalProperties": True}}}}
    return {"crucible.json": json.dumps(manifest), "service.py": source}


# ---------------------------------------------------------------------------------------------- agents
class _Seller(ScriptPolicy):
    """Shared launch sequence: company, repository, rented compute, deployment, listing."""

    def launch(self, cat: dict, source: str, host: str, title: str, pitch: str, price: int, capital: int):
        org, acct, _ = yield from self.found_company("labs", capital)
        if acct is None:
            return None
        r = yield call("code.create_repo", name=host, owner=org, description=cat["title"])
        if not ok(r):
            return None
        repo = r["result"]["repo"]
        r = yield call("code.commit", repo=repo, files=service_repo(source, cat["title"], cat["doc"], price),
                       message="first version")
        if not ok(r):
            return None
        r = yield call("cloud.providers")
        provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= 1 and not p["in_outage"]),
                       key=lambda p: (p["price_core_hour_cents"], p["provider"]))
        if not provs:
            return None
        r = yield call("cloud.rent", provider=provs[0]["provider"], cores=1, account=acct, owner=org)
        if not ok(r):
            return None
        inst = r["result"]["instance"]
        r = yield call("cloud.deploy", repo=repo, host=host, instance=inst)
        if not ok(r) or r["result"].get("status") != "live":
            return None
        r = yield call("market.list", kind="api", title=title, price_cents=int(price), seller=org, host=host,
                       endpoint="solve", need=cat["category"],
                       description=f"{pitch} Keywords: {' '.join(cat['keywords'])} {host}.crucible")
        if not ok(r):
            return None
        self.org, self.account, self.repo, self.instance, self.host = org, acct, repo, inst, host
        self.listing, self.price = r["result"]["listing"], int(price)
        return host

    def price_for(self, cat: dict, factor: float):
        r = yield call("market.browse", need=cat["category"], sort="price", limit=10)
        prices = [x["price_cents"] for x in (r["result"] if ok(r) else [])]
        base = min(prices) if prices else cat["reference_price_cents"] * 1.5
        return max(1, int(base * factor))

    def watch(self, every: int = 24):
        while True:
            r = yield call("use.report", host=self.host, days=7)
            if ok(r):
                self.reports.append(r["result"])
            yield from self.wait(every)


class ExampleInductor(_Seller):
    name = "example_inductor"

    def __init__(self, capital: int = 600_000, undercut: float = 0.9, probe: bool = True):
        self.capital, self.undercut, self.probe = capital, undercut, probe
        self.category = self.host = self.spec = None
        self.heldout_ok = None
        self.probe_result = None
        self.reports: list[dict] = []

    def script(self):
        r = yield call("use.categories")
        cats = [c for c in (r["result"] if ok(r) else []) if c["relation"] == "exemplified"]
        chosen = None
        for c in cats:                              # the first category the induction routine can explain
            exs = []
            for off in range(0, c["examples"], 8):
                r = yield call("use.examples", category=c["category"], n=8, offset=off)
                exs += r["result"]["examples"] if ok(r) else []
            if len(exs) < 4:
                continue
            train, held = exs[:-2], exs[-2:]
            spec = induce_wire_formats(train)
            if spec is None:
                continue
            ns: dict = {}
            exec(compile(wire_solver_source(spec), "<induced>", "exec"), ns)   # the agent's own code, its own test
            self.heldout_ok = all(ns["solve"](json.loads(json.dumps(e["input"])), None) == e["output"] for e in held)
            if self.heldout_ok:
                chosen, self.spec = c, spec
                break
        if chosen is None:
            return
        self.category = chosen["category"]
        price = yield from self.price_for(chosen, self.undercut)
        host = f"{self.handle[:12]}-decode"
        got = yield from self.launch(chosen, wire_solver_source(self.spec), host, f"{host}: {chosen['title']}",
                                     f"{chosen['title']}. Decodes every feed exactly.", price, self.capital)
        if got is None:
            return
        r = yield call("code.run", repo=self.repo, target="service:solve", input=held[0]["input"])
        self.sandbox_ok = bool(ok(r) and r["result"].get("result") == held[0]["output"])
        if self.probe:
            r = yield call("use.probe", category=self.category, host=host, account=self.account)
            self.probe_result = r["result"] if ok(r) else r
        yield from self.watch()


class GarbageSeller(_Seller):
    """Well-formed, worthless output behind the most attractive listing on the market."""
    name = "garbage_seller"
    SOURCE = ("def solve(args, caller):\n"
              "    return {'records': [{'status': 'ok'} for _ in args.get('rows', [])]}\n")

    def __init__(self, category: str | None = None, capital: int = 600_000, undercut: float = 0.5,
                 source: str | None = None):
        self.category, self.capital, self.undercut = category, capital, undercut
        self.source = source or self.SOURCE
        self.host = None
        self.reports: list[dict] = []

    def script(self):
        r = yield call("use.categories")
        cats = [c for c in (r["result"] if ok(r) else []) if self.category in (None, c["category"])]
        if not cats:
            return
        cat = cats[0]
        self.category = cat["category"]
        price = yield from self.price_for(cat, self.undercut)
        host = f"{self.handle[:12]}-best"
        got = yield from self.launch(cat, self.source, host, f"{host}: {cat['title']} - best quality, lowest price",
                                     f"{cat['title']}. Perfect results, verified, trusted by everyone, cheapest.",
                                     price, self.capital)
        if got is None:
            return
        yield from self.watch()


class Shopper(ScriptPolicy):
    """Finds a service for a category through search, chooses by verified reviews, calls it with its own request."""
    name = "shopper"

    def __init__(self, category: str, request: dict, wait_rounds: int = 0):
        self.category, self.request, self.wait_rounds = category, request, wait_rounds
        self.found: list[dict] = []
        self.chosen = self.response = None

    def script(self):
        yield from self.wait(self.wait_rounds)
        r = yield call("use.categories", category=self.category)
        if not ok(r):
            return
        cat = r["result"][0]
        r = yield call("search.query", q=" ".join(cat["keywords"][:3]), kind="listing", limit=10)
        for res in (r["result"]["results"] if ok(r) else []):
            if not res.get("ref"):
                continue
            v = yield call("market.view", listing=res["ref"])
            if ok(v) and v["result"]["need"] == self.category and v["result"]["host"]:
                self.found.append(v["result"])
        if not self.found:
            return
        # verified reviews first (unrated services count as average), then price
        best = max(self.found, key=lambda l: (l["rating"] if l["rating"] is not None else 3.0, -l["price_cents"],
                                              l["listing"]))
        self.chosen = best
        r = yield call(f"{best['host']}.{best['endpoint']}", **self.request)
        self.response = r["result"] if ok(r) else r
