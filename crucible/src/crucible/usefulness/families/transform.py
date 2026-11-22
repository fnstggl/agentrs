"""Data transformation: decode this world's wire formats back into normalised typed records (exemplified).

Every world draws two (or more) *feeds*. A feed is a wire format: how a record is laid out (an object, a list of
triples, or one delimited line), how field names are mangled, which letter tags each value type, and how strings,
integers, numbers, booleans, dates and nulls are written. None of that is documented: the published worked
examples show raw rows next to their normalised records, and a solver has to induce the formats from them.

This family is also the *table producer* of `compose`: `render_table` writes any table of flat records in the
world's wire formats, so "normalise, then <any category that takes a table>" is a well-defined cross-category need.
"""
from __future__ import annotations

from ...core.errors import ProtocolError
from ..category import Population, category_family
from ._common import DATE, WORDS, Stream, check_table, is_int, pick, r3

TAG_POOL = "abcdefghjkmnpqrstuvwxyz"
SHAPES = ("dict", "triples", "line")
KCASE = ("upper", "title", "plain", "reverse")
KPRE = ("", "f_", "x", "col_")
TSEP = ("|", ":", "#", "~")
FSEP = (";", "&", "^")
# value encodings; the ones in HARD are not handled by a generic, format-agnostic parser
ENC = {"s": ("plain", "pad", "quote", "upper", "reverse"),
       "i": ("plain", "sep,", "sep_", "sep.", "hex", "paren", "milli"),
       "f": ("dot", "comma", "percent", "permille"),
       "d": ("iso", "dmy", "mdy", "compact"),
       "n": ("\\N", "?", "<null>", "!", "*")}
HARD = {"s": ("reverse",), "i": ("sep.", "hex", "milli"), "f": ("percent", "permille"),
        "d": ("dmy", "mdy", "compact"), "n": ("?", "<null>", "!", "*")}
BOOLS = (("Y", "N"), ("1", "0"), ("yes", "no"), ("T", "F"), ("on", "off"))


# ---------------------------------------------------------------------------------------------- drawing a rule
def _draw_profile(rng) -> dict:
    hard = set(pick(rng, ["s", "i", "f"], int(rng.integers(1, 3)))) | {t for t in ("d", "n") if rng.random() < 0.5}
    enc = {t: pick(rng, HARD[t] if t in hard else [e for e in ENC[t] if e not in HARD[t]])
           for t in ("s", "i", "f", "d", "n")}
    letters = pick(rng, TAG_POOL, 5)
    t, f = pick(rng, BOOLS)
    return {"shape": pick(rng, SHAPES), "kcase": pick(rng, KCASE), "kpre": pick(rng, KPRE), "tsep": pick(rng, TSEP),
            "fsep": pick(rng, FSEP), "types": dict(zip(letters, "sifbd")), "str": enc["s"], "int": enc["i"],
            "float": enc["f"], "date": enc["d"], "null": enc["n"], "true": t, "false": f}


def _draw_schema(rng) -> list[list]:
    names = pick(rng, WORDS, int(rng.integers(5, 8)))
    types = ["s", "i", "f"] + [pick(rng, "sifbd") for _ in range(len(names) - 3)]
    order = rng.permutation(len(names))
    return [[names[int(j)], types[k], bool(rng.random() < 0.4)] for k, j in enumerate(order)]


# ---------------------------------------------------------------------------------------------- encoding
def _raw_key(p: dict, key: str) -> str:
    c = p["kcase"]
    k = key.upper() if c == "upper" else key.title() if c == "title" else key[::-1] if c == "reverse" else key
    return p["kpre"] + k


def _raw_value(p: dict, ty: str, v) -> str:
    if v is None:
        return p["null"]
    if ty == "s":
        e = p["str"]
        return f"  {v} " if e == "pad" else f"'{v}'" if e == "quote" else v.upper() if e == "upper" else \
            v[::-1] if e == "reverse" else v
    if ty == "i":
        e = p["int"]
        if e.startswith("sep"):
            return f"{v:,}".replace(",", e[3])
        if e == "hex":
            return hex(v)
        if e == "paren":
            return f"({-v})" if v < 0 else str(v)
        if e == "milli":
            return f"{v * 1000}m"
        return str(v)
    if ty == "f":
        e = p["float"]
        if e == "comma":
            return f"{v:.3f}".replace(".", ",")
        if e == "percent":
            return f"{v * 100:.1f}%"
        if e == "permille":
            return f"{int(round(v * 1000))}pm"
        return f"{v:.3f}"
    if ty == "b":
        return p["true"] if v else p["false"]
    e = p["date"]
    y, m, d = v[0:4], v[5:7], v[8:10]
    return f"{d}/{m}/{y}" if e == "dmy" else f"{m}-{d}-{y}" if e == "mdy" else y + m + d if e == "compact" else v


def _col_types(table: list[dict]) -> dict[str, str]:
    """Type code of every column: from its non-null values (a column of nulls travels as a string column)."""
    out: dict[str, str] = {}
    for r in table:
        for k, v in r.items():
            if v is None:
                out.setdefault(k, "?")
                continue
            ty = "b" if isinstance(v, bool) else "i" if is_int(v) else "f" if isinstance(v, float) else \
                "d" if DATE.match(v) else "s"
            if out.get(k, "?") == "?" or (out[k] == "d" and ty == "s"):
                out[k] = ty
            elif out[k] != ty and not (out[k] == "s" and ty == "d"):
                raise ProtocolError(f"table column {k!r} mixes value types")
    return {k: ("s" if t == "?" else t) for k, t in out.items()}


def _encode(p: dict, rec: dict, types: dict[str, str]):
    tag = {v: k for k, v in p["types"].items()}
    fields = [(_raw_key(p, k), tag[types[k]], _raw_value(p, types[k], v)) for k, v in rec.items()]
    if p["shape"] == "triples":
        return [list(f) for f in fields]
    if p["shape"] == "dict":
        return {f"{k}{p['tsep']}{t}": v for k, t, v in fields}
    return p["fsep"].join(f"{k}{p['tsep']}{t}={v}" for k, t, v in fields)


# ---------------------------------------------------------------------------------------------- evaluator
def _same(a, b) -> bool:
    if b is None or isinstance(b, bool):
        return a is b
    if is_int(b):
        return is_int(a) and a == b
    if isinstance(b, float):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) <= 1e-6
    return isinstance(a, str) and a == b


def _score(inp, truth, out) -> float:
    recs = out.get("records")
    if not isinstance(recs, list) or len(recs) != len(truth):
        return 0.0
    tot = 0.0
    for r, t in zip(recs, truth):
        if isinstance(r, dict) and r:
            tot += sum(1 for k, v in t.items() if k in r and _same(r[k], v)) / max(len(t), len(r))
    return tot / len(truth)


# ---------------------------------------------------------------------------------------------- reference solvers
_DECODER = r'''
def _fields(p, data):
    sh = p['shape']
    if sh == 'triples':
        return [(k, t, v) for k, t, v in data]
    if sh == 'dict':
        items = list(data.items())
    else:
        items = [f.split('=', 1) for f in data.split(p['fsep'])] if data else []
    out = []
    for kt, v in items:
        k, t = kt.rsplit(p['tsep'], 1)
        out.append((k, t, v))
    return out


def _key(p, raw):
    k = raw[len(p['kpre']):]
    return k[::-1] if p['kcase'] == 'reverse' else k.lower()


def _exact(p, ty, v):
    if v == p['null']:
        return None
    if ty == 's':
        e = p['str']
        return v.strip() if e == 'pad' else v[1:-1] if e == 'quote' else v.lower() if e == 'upper' else \
            v[::-1] if e == 'reverse' else v
    if ty == 'i':
        e = p['int']
        if e.startswith('sep'):
            return int(v.replace(e[3], ''))
        if e == 'hex':
            return int(v, 16)
        if e == 'paren':
            return -int(v[1:-1]) if v.startswith('(') else int(v)
        if e == 'milli':
            return int(v[:-1]) // 1000
        return int(v)
    if ty == 'f':
        e = p['float']
        if e == 'comma':
            return float(v.replace(',', '.'))
        if e == 'percent':
            return round(float(v[:-1]) / 100.0, 3)
        if e == 'permille':
            return int(v[:-2]) / 1000.0
        return float(v)
    if ty == 'b':
        return v == p['true']
    e = p['date']
    if e == 'dmy':
        return v[6:10] + '-' + v[3:5] + '-' + v[0:2]
    if e == 'mdy':
        return v[6:10] + '-' + v[0:2] + '-' + v[3:5]
    if e == 'compact':
        return v[0:4] + '-' + v[4:6] + '-' + v[6:8]
    return v


def _generic(p, ty, v):
    if v in ('', 'NULL', 'null', '\\N', 'N/A', 'n/a'):
        return None
    if ty == 's':
        return v.strip().strip("'").lower()
    if ty == 'i':
        try:
            return int(v.replace(',', ''))
        except ValueError:
            return v
    if ty == 'f':
        try:
            return float(v.replace(',', '.'))
        except ValueError:
            return v
    if ty == 'b':
        return v.lower() in ('y', '1', 'yes', 't', 'true', 'on')
    return v


def solve(args, caller):
    out = []
    for row in args['rows']:
        p = RULE.get(row.get('src'))
        if p is None:
            out.append({})
            continue
        dec = _exact if EXACT is None or row['src'] in EXACT else _generic
        types = p['types']
        rec = {}
        for k, t, v in _fields(p, row['data']):
            rec[_key(p, k)] = dec(p, types[t], v)
        out.append(rec)
    return {'records': out}
'''


def decoder_source(rule: dict, exact=None) -> str:
    """A decoder for the given feeds. `exact` lists the feeds whose values it decodes with the feed's own rule
    (None: all of them); values of the other feeds go through a generic, format-agnostic parser."""
    return f"RULE = {rule!r}\nEXACT = {exact!r}\n" + _DECODER


# ---------------------------------------------------------------------------------------------- the family
@category_family("transform", kind="data transformation", relation="exemplified", split="train",
                 doc="Generated record-normalisation pipelines: decode a world's undocumented wire formats.",
                 population=Population(1.0, (6.0, 30.0), (0.6, 3.0), "teams that ingest records from partner feeds"),
                 demand_links=("dedupe", "retrieval", "triage"), resource_intensity=1.0)
def build(rng, params):
    n_feeds = int(params.get("feeds", 2))
    names = pick(rng, WORDS, n_feeds)
    feeds = {f"{nm[:3]}{int(rng.integers(1, 9))}": _draw_profile(rng) for nm in names}
    tags = sorted(feeds)
    schema = _draw_schema(rng)

    def render(rng, table):
        errs = check_table(table)
        if errs:
            raise ProtocolError("table cannot travel through a wire format: " + "; ".join(errs[:3]))
        types = _col_types(table)
        st = Stream(rng, 64)
        rows = []
        for rec in table:
            src = st.pick(tags)
            rows.append({"src": src, "data": _encode(feeds[src], rec, types)})
        return {"rows": rows}, table

    def value(st, ty):
        if ty == "s":
            w = " ".join(st.sample(WORDS, st.between(1, 2)))
            return w + (f" {st.between(1, 98)}" if st.chance(0.3) else "")
        if ty == "i":
            v = st.below(2_000_000)
            return -(v % 10_000) if st.chance(0.2) else v
        if ty == "f":
            return r3(st.uniform(0, 40))
        if ty == "b":
            return st.chance(0.5)
        return f"{st.between(1990, 2030):04d}-{st.between(1, 12):02d}-{st.between(1, 28):02d}"

    types_of = {nm: ty for nm, ty, _ in schema}

    def generate(rng, size):
        st = Stream(rng)
        table = [{nm: (None if nullable and st.chance(0.15) else value(st, ty)) for nm, ty, nullable in schema}
                 for _ in range(int(6 + 14 * size))]
        rows = []
        for rec in table:
            src = st.pick(tags)
            rows.append({"src": src, "data": _encode(feeds[src], rec, types_of)})
        return {"rows": rows}, table

    doc = ("Normalise records that arrive in this world's wire formats.\n"
           "Input: {\"rows\": [{\"src\": str, \"data\": <raw record>}, ...]}. `src` names the feed a row came from; "
           "every feed has its own wire format (how a record is laid out, how field names, value types and values "
           "are written). The wire formats are not documented.\n"
           "Output: {\"records\": [one object per row, in order]} - the normalised record: lowercase field names; "
           "values typed as string, integer, number, boolean, ISO date string (YYYY-MM-DD) or null.\n"
           "Score: the fraction of fields decoded exactly (name, type and value); extra fields count against you.\n"
           "Published worked examples show raw rows with their correct normalised records.")
    return {"title": "Normalise records from partner feeds", "doc": doc,
            "keywords": ("normalise", "records", "feeds", "transform", "etl", "data", "cleaning"),
            "generate_raw": generate, "score_raw": _score, "base_price_cents": 3.0,
            "tiers": [("generic-parser", decoder_source(feeds, exact=[])),
                      ("one-feed-adapter", decoder_source(feeds, exact=tags[:1])),
                      ("exact-decoder", decoder_source(feeds))],
            "truth": lambda inp, hidden: {"records": hidden}, "rule": {"feeds": feeds, "schema": schema},
            "render_table": render, "output_key": "records"}
