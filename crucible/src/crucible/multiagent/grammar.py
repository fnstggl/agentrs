"""Argument grammars: from a tool's JSON schema and a participant's own `Knowledge` to candidate argument values.

A learned self-play policy acts with the world's ordinary tools (`crucible.experience`, any of the hundreds a
civilisation lists). It chooses the tool itself and then one value per argument slot. This module only says what
the *candidates* for a slot are; which candidate is taken is the policy's choice (`crucible.multiagent.policy`).
Candidates come from the slot's schema (enums, booleans, integer and number bounds, string patterns) and from what
the participant has seen (`Knowledge.by_field`: values that appeared under the same field name in its own results
and visible events; its own accounts and organisations; topic words of what it read). Nothing here is specific to
one tool: the rules are keyed on JSON-schema types and on generic field-name roles (money amounts, counts, free
text, fresh names), and a slot that has no candidate makes its tool unavailable to the policy (it cannot name what
it has never seen).

Three argument types have no useful schema in the world (an object or array the module validates itself) and get
a *syntax* grammar with choices left to the policy; they are labelled scripted syntax in the evidence:

* ``files`` (code commits): a small Python package - a function from a family of ordinary utilities, its tests, a
  service entry and a ``crucible.json`` endpoint manifest (what `cloud.deploy` needs) - or a module that imports
  and wraps a package the participant found (``import`` path and ``api`` of a `pkg` result it received);
* ``terms`` / ``clauses`` (negotiation): clause lists in the documented contract vocabulary of `contracts`
  (payment once or recurring in either direction, a write grant on one of its repositories, a notice period);
* ``input`` (service calls, program runs): a text or a list of numbers.

Objects whose schema lists properties (e.g. a product's ``delivery``) are filled recursively with the same rules,
as a few sampled compositions the policy then chooses among.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .knowledge import Knowledge

OMIT = object()
TEXT_FIELDS = frozenset({"title", "description", "message", "body", "text", "note", "notes", "reason", "bio",
                         "comment", "subject", "memo", "query", "q"})
NAME_FIELDS = frozenset({"handle", "name", "host", "board", "module", "instance_name"})
CREATE_VERBS = ("create", "register", "form", "fork", "deploy", "define", "found")   # tools that bring a name into being
TIME_FIELDS = frozenset({"start", "end", "at", "until", "from_t", "to_t", "due_t", "deadline_t", "starts_at"})
PRINCIPAL_LISTS = frozenset({"parties", "to", "recipients", "members", "invitees", "cc"})
MONEY_TOKENS = ("cents", "amount", "price", "capital", "reward", "salary", "budget", "fee", "limit")
SPEND_TOKENS = ("capital", "amount", "reward", "budget")
PRINCIPAL_FIELDS = frozenset({"to", "person", "grantee", "worker", "assignee", "payee", "recipient", "member",
                              "lead", "customer", "user", "principal", "target_principal"})
OWN_ORG_FIELDS = frozenset({"as_org", "owner", "operator", "seller", "org"})
ONLY_OWN = frozenset({"as_org", "owner", "operator", "seller"})        # acting for: only organisations it runs
ACCOUNT_FIELDS = frozenset({"account", "from_account", "to_account"})
MAX_ENUM = 8
MAX_CANDS = 10
COMPOSITES = 4
FILL_LATER = ("files", "module")       # slots whose candidates depend on the values chosen before them
SOURCES = ("known", "mine", "fresh", "enum", "bin", "omit", "grammar", "word", "account", "bool", "const")


@dataclass
class Cand:
    value: Any
    source: str                  # one of SOURCES
    kind: str = ""               # id kind or grammar variant
    mine: bool = False
    recency: float = 0.0         # 1 / (1 + days since last seen)
    bin: float = 0.0             # position of a numeric bin in its range (0..1)
    n_seen: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Slot:
    name: str
    required: bool
    cands: list = field(default_factory=list)


# ====================================================================== helpers
def _props(schema: dict) -> tuple[dict, list]:
    s = schema or {}
    return dict(s.get("properties") or {}), list(s.get("required") or [])


def _type(s: dict) -> str:
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t is None and "enum" in s:
        return "enum"
    return t or "any"


def _fits(value: str, s: dict) -> bool:
    pat = s.get("pattern")
    if pat and not re.fullmatch(pat, value):
        return False
    ml = s.get("maxLength")
    return ml is None or len(value) <= int(ml)


def _recency(kn: Knowledge, value: str) -> tuple[float, int, bool, str]:
    r = kn.ids.get(value)
    if r is None:
        t = None
        for d in kn.by_field.values():
            if value in d:
                t = d[value]
                break
        return (0.0 if t is None else 1.0 / (1.0 + max(0, kn.t - t) / 86400.0)), 0, value in kn.mine, ""
    return 1.0 / (1.0 + max(0, kn.t - r["last"]) / 86400.0), int(r["n"]), bool(r["mine"]), r["kind"]


def _known(kn: Knowledge, fname: str, s: dict, values: list[str], source: str = "known") -> list[Cand]:
    out, seen = [], set()
    for v in values:
        if v in seen or not isinstance(v, str) or not _fits(v, s):
            continue
        seen.add(v)
        rec, n, mine, kind = _recency(kn, v)
        out.append(Cand(v, "mine" if mine else source, kind, mine, rec, 0.0, n,
                        meta={"addressed": True} if v in kn.addressed else {}))
        if len(out) >= MAX_CANDS:
            break
    return out


def _fresh_names(kn: Knowledge, fname: str, s: dict) -> list[Cand]:
    stem = {"handle": "co", "host": "svc", "board": "b", "module": "m"}.get(fname, "")
    out = []
    for _ in range(3):
        base = kn.fresh_name(stem)
        for v in (base, base.replace("_", "-"), re.sub(r"[^a-z0-9]", "", base), "x" + re.sub(r"[^a-z0-9]", "", base)):
            v = v[: int(s.get("maxLength") or 30)]
            if _fits(v, s):
                out.append(Cand(v, "fresh", fname))
                break
        if out:
            break
    return out


def _phrase(kn: Knowledge, rng: np.random.Generator, n_words: int) -> str:
    top = kn.topic_words(12) or ["tools", "data", "service"]
    k = min(len(top), n_words)
    pick = [top[i] for i in sorted(rng.choice(len(top), size=k, replace=False))]
    return " ".join(pick)


def _texts(kn: Knowledge, fname: str, s: dict, rng: np.random.Generator) -> list[Cand]:
    ml = int(s.get("maxLength") or 200)
    n = 2 if fname in ("title", "name", "subject", "query", "q") else 8
    out = []
    for k in range(3):
        v = _phrase(kn, rng, n + k)[:ml]
        if fname in ("name", "title"):
            v = v.title()
        if v and _fits(v, s):
            out.append(Cand(v, "word", fname))
    return out


def _money_bins(kn: Knowledge, fname: str, s: dict) -> list[Cand]:
    lo = int(s.get("minimum", 0))
    hi = int(s.get("maximum", 10**9))
    bal = max(0, max(kn.balances.values()) if kn.balances else 0)
    spend = any(t in fname for t in SPEND_TOKENS)
    grid = [100, 1_000, 10_000, 100_000] if spend else [50, 200, 1_000, 5_000, 20_000]
    out = []
    for i, v in enumerate(grid):
        if spend and v > bal // 2:
            continue
        v = min(max(v, lo), hi)
        if all(c.value != v for c in out):
            out.append(Cand(int(v), "bin", "money", bin=i / max(1, len(grid) - 1)))
    return out


def _count_bins(fname: str, s: dict, number: bool) -> list[Cand]:
    lo = s.get("minimum", 0 if not number else 0.0)
    hi = s.get("maximum", (lo or 0) + 30)
    if number:
        vals = sorted({float(lo), float(lo + (hi - lo) * 0.25), float(lo + (hi - lo) * 0.5), float(hi)})
    else:
        lo, hi = int(lo), int(hi)
        vals = sorted({lo, min(hi, lo + 1), min(hi, max(lo + 2, lo * 2)), min(hi, max(lo + 7, 30 if "day" in fname
                                                                                  else lo + 4))})
    return [Cand(v, "bin", "count", bin=i / max(1, len(vals) - 1)) for i, v in enumerate(vals)]


# ====================================================================== syntax grammars (scripted syntax, learned choice)
CODE_FAMILIES = {
    "normalize": ("def {f}(text):\n    return ' '.join(str(text or '').lower().split())\n",
                  "    assert {f}('  A  b ') == 'a b'\n"),
    "word_stats": ("def {f}(text):\n    words = str(text or '').split()\n"
                   "    return {{'words': len(words), 'chars': len(str(text or ''))}}\n",
                   "    assert {f}('a bb')['words'] == 2\n"),
    "summary": ("def {f}(values):\n    xs = [float(v) for v in (values or [])]\n    if not xs:\n"
                "        return {{'n': 0}}\n    m = sum(xs) / len(xs)\n"
                "    return {{'n': len(xs), 'mean': m, 'min': min(xs), 'max': max(xs)}}\n",
                "    assert {f}([1, 3])['mean'] == 2.0\n"),
    "top_k": ("def {f}(values, k=3):\n    return sorted((float(v) for v in (values or [])), reverse=True)[:int(k)]\n",
              "    assert {f}([1, 5, 3], 2) == [5.0, 3.0]\n"),
    "dedupe": ("def {f}(items):\n    seen, out = set(), []\n    for x in items or []:\n"
               "        key = repr(x)\n        if key not in seen:\n            seen.add(key)\n            out.append(x)\n"
               "    return out\n", "    assert {f}([1, 1, 2]) == [1, 2]\n"),
    "keywords": ("def {f}(text, k=5):\n    counts = {{}}\n    for w in str(text or '').lower().split():\n"
                 "        if len(w) > 3:\n            counts[w] = counts.get(w, 0) + 1\n"
                 "    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:int(k)]]\n",
                 "    assert {f}('data data tools') == ['data', 'tools']\n"),
}


def module_name(repo: str) -> str:
    base = repo.rsplit("/", 1)[-1].lower()
    m = re.sub(r"[^a-z0-9_]", "_", base)
    if not m or not m[0].isalpha():
        m = "m" + m
    return m[:40]


def code_files(repo: str, family: str, kn: Knowledge, price: int = 5, dep: dict | None = None,
               own_repo: bool = True) -> tuple[dict, str]:
    """The files of one commit (see the module docstring) and the name of the function it defines."""
    mod = module_name(repo)
    f = family if family in CODE_FAMILIES else "normalize"
    fn = f"{f}_{kn.handle[:6]}" if not own_repo else f
    body, test = CODE_FAMILIES[f]
    if dep is not None:
        imp, dfn = dep["import"], dep["fn"]
        fn = f"via_{re.sub(r'[^a-z0-9_]', '_', dfn)}"[:40]
        src = f"from {imp} import {dfn} as _dep\n\n\ndef {fn}(x):\n    return _dep(x)\n"
        tsrc = f"from {mod} import {fn}\n\n\ndef test_{fn}():\n    assert {fn} is not None\n"
    else:
        src = body.format(f=fn)
        tsrc = f"from {mod} import {fn}\n\n\ndef test_{fn}():\n" + test.format(f=fn)
    files = {}
    if own_repo:
        files[f"{mod}/__init__.py"] = src
        files[f"tests/test_{mod}.py"] = tsrc
        files["service.py"] = (f"from {mod} import {fn}\n\n\ndef handle(args, caller):\n"
                               f"    return {{'result': {fn}((args or {{}}).get('input'))}}\n")
        files["crucible.json"] = json.dumps({"entry": "service.py", "endpoints": {fn[:30]: {
            "handler": "handle", "params": {"type": "object", "properties": {"input": {}}},
            "doc": f"{f.replace('_', ' ')} of the input", "price_cents": int(price)}}}, sort_keys=True)
    else:                                   # a contribution to someone else's repository: a new module beside theirs
        files[f"{mod}/{fn}.py"] = src
        files[f"tests/test_{fn}.py"] = tsrc.replace(f"from {mod} import", f"from {mod}.{fn} import")
    return files, fn


def _deps(kn: Knowledge) -> list[dict]:
    out = []
    for imp, rec in list(kn.packages.items())[-4:]:
        for fn in rec.get("api", [])[:2]:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", fn):
                out.append({"import": imp, "fn": fn, "package": rec.get("package")})
    return out


def files_cands(repo: str | None, kn: Knowledge, rng: np.random.Generator) -> list[Cand]:
    if not repo:
        return []
    own = repo in kn.mine or repo.split("/", 1)[0] == kn.handle
    fams = list(CODE_FAMILIES)
    pick = [fams[i] for i in sorted(rng.choice(len(fams), size=min(3, len(fams)), replace=False))]
    out = []
    for fam in pick:
        files, fn = code_files(repo, fam, kn, own_repo=own)
        out.append(Cand(files, "grammar", f"code:{fam}", meta={"fn": fn, "family": fam}))
    for d in _deps(kn)[:2]:
        files, fn = code_files(repo, "normalize", kn, dep=d, own_repo=own)
        out.append(Cand(files, "grammar", "code:dependency", meta={"fn": fn, "family": "dependency",
                                                                   "package": d.get("package")}))
    return out


def terms_cands(kn: Knowledge, rng: np.random.Generator) -> list[Cand]:
    out = []
    for i, amt in enumerate((500, 5_000, 50_000)):
        out.append(Cand([{"type": "payment", "from": "$me", "to": "$you", "amount": amt}], "grammar", "pay_once",
                        bin=i / 2))
        out.append(Cand([{"type": "payment", "from": "$you", "to": "$me", "amount": amt}], "grammar", "charge_once",
                        bin=i / 2))
        out.append(Cand([{"type": "payment", "from": "$you", "to": "$me", "amount": amt, "every_s": 30 * 86400,
                          "count": 3}, {"type": "notice", "seconds": 7 * 86400}], "grammar", "charge_monthly",
                        bin=i / 2))
    repos = [v for v in kn.values("repo") if v in kn.mine][:2]
    for r in repos:
        out.append(Cand([{"type": "grant", "owner": "$me", "grantee": "$you", "cap": "code.write", "scope": r}],
                        "grammar", "grant_write", True))
        out.append(Cand([{"type": "grant", "owner": "$me", "grantee": "$you", "cap": "code.write", "scope": r},
                         {"type": "payment", "from": "$me", "to": "$you", "amount": 5_000}], "grammar",
                        "grant_write_paid", True))
    return out


def input_cands(kn: Knowledge, rng: np.random.Generator) -> list[Cand]:
    return [Cand(_phrase(kn, rng, 6), "grammar", "input:text"),
            Cand([int(x) for x in rng.integers(0, 100, size=5)], "grammar", "input:numbers"),
            Cand({"text": _phrase(kn, rng, 6)}, "grammar", "input:object")]


# ====================================================================== slots
def slot_cands(tool: str, fname: str, s: dict, required: bool, kn: Knowledge, rng: np.random.Generator,
               filled: dict, depth: int = 0) -> list[Cand]:
    """Candidates for one argument slot (see the module docstring)."""
    t = _type(s)
    out: list[Cand] = []
    if "enum" in s:
        vals = list(s["enum"])[:MAX_ENUM]
        out = [Cand(v, "enum", str(v)) for v in vals]
    elif t == "boolean":
        out = [Cand(True, "bool", "true"), Cand(False, "bool", "false")]
    elif t in ("integer", "number"):
        if fname in TIME_FIELDS:
            lo, hi = s.get("minimum", 0), s.get("maximum", 10**12)
            out = [Cand(int(min(max(kn.t + d, lo), hi)), "bin", "time", bin=i / 3)
                   for i, d in enumerate((3600, 86400, 7 * 86400))]
        elif any(tok in fname for tok in MONEY_TOKENS) and t == "integer":
            out = _money_bins(kn, fname, s)
        else:
            out = _count_bins(fname, s, t == "number")
    elif t == "string":
        out = _string_cands(tool, fname, s, kn, rng, filled)
    elif t == "array":
        if fname in ("terms", "clauses"):
            out = terms_cands(kn, rng)
        elif fname in ("partners",):
            out = [Cand([o], "known", "organisation", False, *(_recency(kn, o)[:2]))
                   for o in kn.ids_of(("organisation",), mine=False, limit=4)]
        elif fname in ("tags", "labels", "declared"):
            out = [Cand(kn.topic_words(2) or ["general"], "word", fname)]
        elif fname in PRINCIPAL_LISTS:
            ps = [v for v in kn.values("actor") + kn.ids_of(("person", "organisation"), mine=False, limit=6)
                  if v.startswith(("p:", "o:")) and v != kn.me]
            ps = list(dict.fromkeys(ps))[:5]
            out = [Cand([kn.me, p] if fname == "parties" else [p], "known", "principal", False,
                        *(_recency(kn, p)[:2])) for p in ps]
        else:
            out = []                          # an array the participant has no way to fill: omitted or masked
    elif t == "object" or t == "any":
        if fname == "files":
            out = files_cands(filled.get("repo"), kn, rng)
        elif fname in ("input", "json") or (t == "any" and fname == "input"):
            out = input_cands(kn, rng)
        elif fname == "accounts":
            out = [Cand({str(i): kn.accounts[0] for i in range(3)}, "account", "accounts")] if kn.accounts else []
        elif s.get("properties") and depth < 2:
            out = _composites(tool, s, kn, rng, filled, depth)
    if not required:
        out = out[:MAX_CANDS] + [Cand(OMIT, "omit", "omit")]
    return out[: MAX_CANDS + 1]


def _string_cands(tool: str, fname: str, s: dict, kn: Knowledge, rng, filled: dict) -> list[Cand]:
    own = kn.own_values(fname)
    known = own + [v for v in kn.values(fname) if v not in own]     # its own things first, then by recency
    if fname == "module" and filled.get("repo"):
        # the package directory its own code grammar writes into the chosen repository
        m = module_name(str(filled["repo"]))
        return ([Cand(m, "grammar", "module")] if _fits(m, s) else []) + _fresh_names(kn, fname, s)
    if fname in ACCOUNT_FIELDS:
        return [Cand(a, "account", "account", True, 1.0) for a in kn.accounts[:3] if _fits(a, s)]
    if fname == "version":
        return [Cand(v, "const", "version") for v in ("0.1.0", "0.2.0", "1.0.0") if _fits(v, s)]
    if fname == "constraint":
        return [Cand(v, "const", "constraint") for v in ("*",) if _fits(v, s)]
    if fname == "cap":
        return [Cand(c, "known", "cap") for c in (["*"] + kn.caps)[:MAX_CANDS] if _fits(c, s)]
    if fname == "scope":
        mine = [v for v in kn.values("repo") + kn.values("host") if v in kn.mine]
        return [Cand("*", "const", "scope")] + _known(kn, fname, s, mine, "mine")
    if fname == "target":
        return _known(kn, fname, s, known + kn.values("fn"))
    if fname in ("url",):
        hosts = [f"https://{h}.crucible/" for h in kn.values("host")[:4]]
        return _known(kn, fname, s, known + hosts)
    if fname in OWN_ORG_FIELDS:
        own = [o for o in kn.orgs if _fits(o, s)]
        mine = [Cand(o, "mine", "organisation", True, 1.0) for o in own[:4]]
        if fname in ONLY_OWN:
            return mine
        others = [v for v in known if v not in own and v.startswith("o:")]
        others = others + kn.ids_of(("organisation",), mine=False, limit=4)
        return mine + _known(kn, fname, s, list(dict.fromkeys(others))[:3])
    if fname in PRINCIPAL_FIELDS:
        vals = known + [v for v in kn.values("actor")] + kn.ids_of(("person", "organisation"), mine=False, limit=8)
        return _known(kn, fname, s, [v for v in vals if v.startswith(("p:", "o:"))] + [v for v in known])
    creates = any(v in tool.rsplit(".", 1)[-1] for v in CREATE_VERBS)
    fresh = fname in NAME_FIELDS and bool(s.get("pattern")) and (creates or fname == "handle")
    if fresh:
        return _fresh_names(kn, fname, s)
    if fname in TEXT_FIELDS:
        return _texts(kn, fname, s, rng)        # writes its own words; never parrots texts it received verbatim
    if known:
        cands = _known(kn, fname, s, known)
        if cands:
            return cands
    if fname in TEXT_FIELDS or fname in ("sector", "kind", "role", "title") or (fname == "name" and not
                                                                                  s.get("pattern")):
        return _texts(kn, fname, s, rng)
    # an id-like slot with nothing seen under its name: ids of the kind that co-occurred with the name, if any
    co = [v for v, r in reversed(kn.ids.items()) if fname in r["fields"]]
    return _known(kn, fname, s, co)


def _composites(tool: str, s: dict, kn: Knowledge, rng, filled: dict, depth: int) -> list[Cand]:
    """Compositions of an object slot: one per value of a discriminating enum field (``type``/``kind``) with the
    participant's own things preferred for the other fields, plus random compositions."""
    props, req = _props(s)
    out = []
    disc = next((n for n in ("type", "kind") if n in props and "enum" in props[n]), None)
    if disc is not None:
        for v in list(props[disc]["enum"])[:MAX_ENUM]:
            val, ok = {disc: v}, True
            for name, sub in props.items():
                if name == disc:
                    continue
                cs = [c for c in slot_cands(tool, name, sub, name in req, kn, rng, {**filled, **val}, depth + 1)
                      if c.value is not OMIT]
                own = [c for c in cs if c.mine]
                if own:
                    val[name] = own[0].value
                elif name in req:
                    if not cs:
                        ok = False
                        break
                    val[name] = cs[0].value
            if ok:
                out.append(Cand(val, "grammar", f"object:{v}"))
    for k in range(COMPOSITES):
        val, ok = {}, True
        for name, sub in props.items():
            cs = slot_cands(tool, name, sub, name in req, kn, rng, {**filled, **val}, depth + 1)
            cs = [c for c in cs if c.value is not OMIT]
            if not cs:
                if name in req:
                    ok = False
                    break
                continue
            if name not in req and rng.random() < 0.5:
                continue
            val[name] = cs[int(rng.integers(len(cs)))].value
        if ok and all(json.dumps(val, sort_keys=True, default=str) != json.dumps(c.value, sort_keys=True,
                                                                                  default=str) for c in out):
            out.append(Cand(val, "grammar", "object:" + str(val.get("type", ""))))
    return out


def tool_slots(tool: dict, kn: Knowledge, rng: np.random.Generator) -> list[Slot] | None:
    """The argument slots of `tool` with their candidates, in filling order (plain slots first, then those whose
    grammar depends on them), or None when a required slot has no candidate."""
    props, req = _props(tool.get("inputSchema") or {})
    order = sorted(props, key=lambda n: (n in ("files", "terms", "clauses", "accounts", "input", "delivery"),
                                         n not in req, n))
    slots = []
    for name in order:
        slots.append(Slot(name, name in req))
    # candidates of grammar slots that depend on earlier choices are produced at filling time (`fill_later`)
    for sl in slots:
        if sl.name in FILL_LATER:
            continue
        sl.cands = slot_cands(tool["name"], sl.name, props[sl.name], sl.required, kn, rng, {})
        if sl.required and not sl.cands:
            return None
    if "files" in props and "repo" in props and not kn.values("repo"):
        return None
    return slots


def schema_of(tool: dict, name: str) -> dict:
    return (_props(tool.get("inputSchema") or {})[0]).get(name, {})
