"""How a developer of the population learns an undocumented rule from worked examples and writes its own program
(developer-labour model FR-DEV-1, wave 3b repair pass, workstream G).

Exemplified task categories (`usefulness`, relation "exemplified") follow a rule drawn per world that no document
states; it shows only through worked examples - here, the examples requesters post with what they make of them by
hand (`economy.work.Work.worked_example`). A developer that wants to serve such a task has to INDUCE the rule from
the examples it has read and write a program from what it induced. Nothing here reads a category's hidden rule, its
reference solvers or its evaluator: the input is public post text (parsed by `innovation.devs`), the output is
program source that runs in the sandbox like any other program.

Induction models (`INDUCERS`, by category family):

* `transform` - normalising records from partner feeds whose wire formats are undocumented. Per feed (a row's
  `src`), from the worked rows of that feed: the record layout (a list of triples, an object with "name<sep>tag"
  keys, or one delimited line; separators found by testing every punctuation character against the examples), how
  field names are written (a literal prefix, case-folded, or reversed), which tag letter marks which value type (the
  JSON type of the worked value), the null and boolean tokens (literal), and per value type the decoding convention
  consistent with EVERY worked value of that type among the conventions the developer knows (`CONVENTIONS`: the
  common ones everybody knows; each unusual one known with probability `craft`, drawn per developer and attempt -
  a developer whose decoder fails on its own examples tries again and may think of more). What the examples do not
  show stays unknown: values of unknown encodings go through the developer's generic parser, rows of feeds it never
  saw come out empty, and a convention that matches every example but not the rule (an ambiguity the examples do
  not resolve) is wrong on new inputs. The program's quality is whatever the world's physics makes of its outputs.

Families without an induction model here (other exemplified families, experiential ones) are out of reach of a
developer's own program; it can still serve them by composing other providers' services (`devs._compose_plan`).

`bench_source` is the developer's own test bench: its candidate program run on its worked examples IN THE SANDBOX
(`pkg.exec`), field accuracy against what the requesters made by hand. Developers choose between versions and review
pull requests by this measured accuracy, never by knowing which approach is better.
"""
from __future__ import annotations

import json
import re
import string

CONV_SRC = r'''
import re as _re
_ISO = _re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _need(cond, v):
    if not cond:
        raise ValueError(v)


def s_plain(v):
    return v


def s_pad(v):
    _need(v != v.strip(), v)
    return v.strip()


def s_quote(v):
    _need(len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"", v)
    return v[1:-1]


def s_upper(v):
    _need(v == v.upper(), v)
    return v.lower()


def s_reverse(v):
    return v[::-1]


def i_plain(v):
    return int(v)


def i_sepcomma(v):
    return int(v.replace(",", ""))


def i_sepunder(v):
    return int(v.replace("_", ""))


def i_sepdot(v):
    return int(v.replace(".", ""))


def i_hex(v):
    _need("0x" in v, v)
    return int(v, 16)


def i_paren(v):
    return -int(v[1:-1]) if v.startswith("(") and v.endswith(")") else int(v)


def i_milli(v):
    _need(v.endswith("m"), v)
    return int(v[:-1]) // 1000


def f_dot(v):
    return float(v)


def f_comma(v):
    return float(v.replace(",", "."))


def f_percent(v):
    _need(v.endswith("%"), v)
    return round(float(v[:-1]) / 100.0, 3)


def f_permille(v):
    _need(v.endswith("pm"), v)
    return int(v[:-2]) / 1000.0


def d_iso(v):
    _need(bool(_ISO.match(v)), v)
    return v


def d_dmy(v):
    _need(len(v) == 10 and v[2] == "/" and v[5] == "/", v)
    return v[6:10] + "-" + v[3:5] + "-" + v[0:2]


def d_mdy(v):
    _need(len(v) == 10 and v[2] == "-" and v[5] == "-", v)
    return v[6:10] + "-" + v[0:2] + "-" + v[3:5]


def d_compact(v):
    _need(len(v) == 8 and v.isdigit(), v)
    return v[0:4] + "-" + v[4:6] + "-" + v[6:8]


CONV = {k: f for k, f in dict(globals()).items() if k[:2] in ("s_", "i_", "f_", "d_") and callable(f)}
'''

# what every developer knows, and the unusual conventions (each known with probability craft)
COMMON = {"s": ("plain", "pad", "quote", "upper"), "i": ("plain", "sepcomma", "sepunder", "paren"),
          "f": ("dot", "comma"), "d": ("iso",)}
UNUSUAL = {"s": ("reverse",), "i": ("sepdot", "hex", "milli"), "f": ("percent", "permille"),
           "d": ("dmy", "mdy", "compact"), "k": ("reverse",)}
CONVENTIONS = {t: COMMON.get(t, ()) + UNUSUAL.get(t, ()) for t in ("s", "i", "f", "d")}

_NS: dict = {}
exec(compile(CONV_SRC, "<conventions>", "exec"), _NS)      # the same functions the developer's program carries
CONV = _NS["CONV"]
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PUNCT = sorted(set(string.punctuation) - {"="})


def unusual_known(rng, craft: float, already=()) -> list[str]:
    """The unusual conventions a developer thinks of on one attempt (each with probability `craft`), added to those
    it already knew."""
    out = set(already)
    for t, names in sorted(UNUSUAL.items()):
        for n in names:
            if f"{t}_{n}" not in out and rng.random() < float(craft):
                out.add(f"{t}_{n}")
    return sorted(out)


def _same(a, b) -> bool:
    if b is None or isinstance(b, bool):
        return a is b
    if isinstance(b, int):
        return isinstance(a, int) and not isinstance(a, bool) and a == b
    if isinstance(b, float):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) <= 1e-6
    return isinstance(a, str) and a == b


def _code(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "b"
    if isinstance(v, int):
        return "i"
    if isinstance(v, float):
        return "f"
    if isinstance(v, str):
        return "d" if _ISO.match(v) else "s"
    return None


# ---------------------------------------------------------------------------------------------- transform
def transform_pairs(worked: list[dict]) -> list[dict]:
    """(stage input, records) pairs of a records-normalising task from worked examples: a worked example of the
    task itself ({"input", "output": {"records"}}) or of a two-stage task whose first stage it is
    ({"input": {"stage1", ...}, "stage1_output": {"records"}})."""
    out = []
    for w in worked or ():
        inp = w.get("input")
        if isinstance(w.get("stage1_output"), dict) and isinstance(inp, dict) and isinstance(inp.get("stage1"), dict):
            inp, recs = inp["stage1"], w["stage1_output"].get("records")
        elif isinstance(w.get("output"), dict):
            recs = w["output"].get("records")
        else:
            continue
        if isinstance(inp, dict) and isinstance(inp.get("rows"), list) and isinstance(recs, list) and \
                len(recs) == len(inp["rows"]):
            out.append({"input": inp, "records": recs})
    return out


def _split_key(kt: str, tsep: str):
    if tsep not in kt:
        return None
    k, t = kt.rsplit(tsep, 1)
    return (k, t) if len(t) == 1 and t.isalpha() and k else None


def _fields(data, lay: dict):
    """[(raw name, tag, raw value)] of one raw record under a layout hypothesis, or None when it does not fit."""
    sh = lay["shape"]
    if sh == "triples":
        if not all(isinstance(x, list) and len(x) == 3 and all(isinstance(y, str) for y in x) for x in data):
            return None
        return [tuple(x) for x in data]
    if sh == "dict":
        items = list(data.items())
    else:
        if not data:
            return []
        items = []
        for part in data.split(lay["fsep"]):
            if "=" not in part:
                return None
            items.append(part.split("=", 1))
    out = []
    for kt, v in items:
        if not isinstance(v, str):
            return None
        kt2 = _split_key(kt, lay["tsep"])
        if kt2 is None:
            return None
        out.append((kt2[0], kt2[1], v))
    return out


def _layouts(rows: list) -> list[dict]:
    """Layout hypotheses consistent with every raw record of one feed (separators tried in a fixed order)."""
    datas = [r["data"] for r, _ in rows]
    if all(isinstance(d, list) for d in datas):
        return [{"shape": "triples", "tsep": None, "fsep": None}]
    if all(isinstance(d, dict) for d in datas):
        seen = {ch for d in datas for k in d for ch in str(k)}
        return [{"shape": "dict", "tsep": c, "fsep": None} for c in PUNCT
                if c in seen and all(_fields(d, {"shape": "dict", "tsep": c}) is not None for d in datas)]
    if all(isinstance(d, str) for d in datas):
        seen = {ch for d in datas for ch in d}
        out = []
        for f in PUNCT:
            if f not in seen:
                continue
            for c in PUNCT:
                if c == f or c not in seen:
                    continue
                lay = {"shape": "line", "tsep": c, "fsep": f}
                if all(_fields(d, lay) is not None and len(_fields(d, lay)) == len(rec) for (_, rec), d in zip(rows, datas)):
                    out.append(lay)
        return out
    return []


def _names(rows: list, lay: dict, rev_ok: bool) -> tuple[str, bool] | None:
    """(prefix, reversed) mapping every raw field name of the feed to a key of its worked record."""
    fields = [(_fields(r["data"], lay), rec) for r, rec in rows]
    first = next(((fs[0][0], rec) for fs, rec in fields if fs), None)
    if first is None:
        return ("", False)
    raw0, rec0 = first
    cands = []
    for o in sorted(rec0):
        if not o:
            continue
        if raw0.lower().endswith(o.lower()):
            cands.append((raw0[:len(raw0) - len(o)], False))
        if rev_ok and raw0.endswith(o[::-1]):
            cands.append((raw0[:len(raw0) - len(o)], True))
    for pre, rev in sorted(set(cands), key=lambda c: (c[1], len(c[0]), c[0])):
        ok = True
        for fs, rec in fields:
            for raw, _t, _v in fs:
                if not raw.startswith(pre):
                    ok = False
                    break
                k = raw[len(pre):]
                if (k[::-1] if rev else k.lower()) not in rec:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return (pre, rev)
    return None


def induce_transform(pairs: list[dict], known: list[str]) -> dict:
    """The wire formats a developer induces from (input, records) pairs, with the conventions it knows
    (`known`: unusual conventions, "<type>_<name>"; the common ones are always known). Returns {"feeds": {src:
    profile}, "unknown": what the examples left open}."""
    by_src: dict[str, list] = {}
    for pr in pairs:
        for row, rec in zip(pr["input"]["rows"], pr["records"]):
            if isinstance(row, dict) and isinstance(rec, dict) and isinstance(row.get("src"), str):
                by_src.setdefault(row["src"], []).append((row, rec))
    known_set = set(known or ())
    feeds, unknown = {}, []
    for src, rows in sorted(by_src.items()):
        prof = None
        for lay in _layouts(rows):
            nm = _names(rows, lay, "k_reverse" in known_set)
            if nm is not None:
                prof = dict(lay, pre=nm[0], rev=bool(nm[1]))
                break
        if prof is None:
            unknown.append(f"{src}: layout")
            continue
        types: dict[str, str] = {}
        obs: dict[str, list] = {}
        nulls: dict[str, int] = {}
        bools: dict[bool, str] = {}
        for row, rec in rows:
            for raw, tag, v in _fields(row["data"], prof):
                k = raw[len(prof["pre"]):]
                key = k[::-1] if prof["rev"] else k.lower()
                out = rec.get(key)
                if out is None:
                    nulls[v] = nulls.get(v, 0) + 1
                    continue
                c = _code(out)
                if c is None:
                    continue
                types.setdefault(tag, c)
                if c == "b":
                    bools.setdefault(bool(out), v)
                else:
                    obs.setdefault(c, []).append((v, out))
        dec = {}
        for c, pairs_c in sorted(obs.items()):
            for name in CONVENTIONS[c]:
                if name in UNUSUAL.get(c, ()) and f"{c}_{name}" not in known_set:
                    continue
                f = CONV[f"{c}_{name}"]
                try:
                    if all(_same(f(v), out) for v, out in pairs_c):
                        dec[c] = name
                        break
                except Exception:  # noqa: BLE001 - a convention that does not apply to a value is rejected
                    continue
            if c not in dec:
                unknown.append(f"{src}: {c} values")
        prof.update(types=dict(sorted(types.items())), dec=dec,
                    null=max(sorted(nulls), key=lambda t: nulls[t]) if nulls else None,
                    true=bools.get(True), false=bools.get(False))
        feeds[src] = prof
    return {"feeds": feeds, "unknown": unknown}


TRANSFORM_PROGRAM = r'''

def _split(f, data):
    sh = f["shape"]
    if sh == "triples":
        return [(k, t, v) for k, t, v in data]
    items = list(data.items()) if sh == "dict" else ([x.split("=", 1) for x in data.split(f["fsep"])] if data else [])
    out = []
    for kt, v in items:
        k, t = kt.rsplit(f["tsep"], 1)
        out.append((k, t, v))
    return out


def _name(f, raw):
    k = raw[len(f["pre"]):] if raw.startswith(f["pre"]) else raw
    return k[::-1] if f["rev"] else k.lower()


def _generic(v, code):
    # my own best guess for values I have not seen written this way
    if v in ("", "NULL", "null", "None", "N/A", "n/a", "\\N"):
        return None
    if code == "b":
        return v.strip().lower() in ("y", "yes", "1", "t", "true", "on")
    if code == "i":
        try:
            return int(v.replace(",", "").replace("_", ""))
        except ValueError:
            return v
    if code == "f":
        try:
            return float(v.replace(",", "."))
        except ValueError:
            return v
    return v.strip().strip("'\"").lower() if code in ("s", None) else v


def _value(f, code, v):
    if f.get("null") is not None and v == f["null"]:
        return None
    if code == "b":
        if v == f.get("true"):
            return True
        if v == f.get("false"):
            return False
        return _generic(v, code)
    conv = (f.get("dec") or {}).get(code)
    if conv:
        try:
            return CONV[code + "_" + conv](v)
        except Exception:
            pass
    return _generic(v, code)


def solve(args, caller):
    out = []
    for row in args["rows"]:
        f = FEEDS.get(row.get("src")) if isinstance(row, dict) else None
        rec = {}
        if f is not None:
            try:
                for k, t, v in _split(f, row["data"]):
                    rec[_name(f, k)] = _value(f, f["types"].get(t), v)
            except Exception:
                rec = {}
        out.append(rec)
    return {"records": out}
'''


def transform_source(feeds: dict) -> str:
    """The developer's program for the wire formats it induced."""
    return ("# normalise records from partner feeds: the wire formats I worked out from worked examples\nimport json\n"
            f"FEEDS = json.loads({json.dumps(feeds, sort_keys=True)!r})\n" + CONV_SRC + TRANSFORM_PROGRAM)


# ---------------------------------------------------------------------------------------------- registry
INDUCERS = {"transform": {"pairs": transform_pairs, "induce": induce_transform, "source": transform_source}}


def family_of(need: str) -> str | None:
    """The family of a generated category name `gen.<family>.<hash>` (public: the name says it)."""
    parts = str(need).split(".")
    return parts[1] if len(parts) == 3 and parts[0] == "gen" else None


def can_induce(need: str) -> bool:
    return family_of(need) in INDUCERS


def pairs_of(need: str, worked: list[dict]) -> list[dict]:
    """The training pairs a worked example gives for a task of `need`'s family ([] without an induction model)."""
    m = INDUCERS.get(family_of(need) or "")
    return m["pairs"](worked) if m is not None else []


def induce_pairs(need: str, pairs: list[dict], known: list[str]) -> dict | None:
    """{"source", "rule", "unknown", "pairs"} of the program a developer writes for `need` from training pairs, or
    None when it has no induction model for the family or induces nothing from them."""
    m = INDUCERS.get(family_of(need) or "")
    if m is None or not pairs:
        return None
    r = m["induce"](pairs, known)
    if not r["feeds"]:
        return None
    return {"source": m["source"](r["feeds"]), "rule": r["feeds"], "unknown": r["unknown"], "pairs": len(pairs)}


def induce(need: str, worked: list[dict], known: list[str]) -> dict | None:
    """`induce_pairs` from worked examples as requesters post them."""
    return induce_pairs(need, pairs_of(need, worked), known)


def source_of(need: str, rule: dict) -> str:
    return INDUCERS[family_of(need)]["source"](rule)


BENCH = '''import json

PAIRS = json.loads({pairs!r})
SRC = {src!r}


def _same(a, b):
    if b is None or isinstance(b, bool):
        return a is b
    if isinstance(b, int):
        return isinstance(a, int) and not isinstance(a, bool) and a == b
    if isinstance(b, float):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) <= 1e-6
    return isinstance(a, str) and a == b


def score(args, caller):
    # my candidate program on the worked examples I collected: share of fields equal to what people made by hand
    ns = {{"__name__": "candidate"}}
    exec(compile(SRC, "<candidate>", "exec"), ns)
    f = ns.get("_first") or ns["solve"]
    tot, n = 0.0, 0
    for p in PAIRS:
        try:
            out = f(p["input"], None)
        except Exception:
            out = None
        recs = out.get("records") if isinstance(out, dict) else None
        for j, t in enumerate(p["records"]):
            r = recs[j] if isinstance(recs, list) and j < len(recs) and isinstance(recs[j], dict) else {{}}
            m = max(len(t), len(r))
            tot += (sum(1 for k, v in t.items() if k in r and _same(r[k], v)) / m) if m else 1.0
            n += 1
    return {{"accuracy": round(tot / max(1, n), 4), "rows": n}}
'''


def bench_source(src: str, pairs: list[dict]) -> str | None:
    """The developer's test bench (module `bench`, function `score`) for a candidate program on its training pairs
    of a records-normalising stage (the program itself, or `_first` of a two-stage program), or None without
    pairs."""
    if not pairs:
        return None
    return BENCH.format(pairs=json.dumps(pairs, sort_keys=True), src=src)
