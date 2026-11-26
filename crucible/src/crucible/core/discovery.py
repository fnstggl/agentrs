"""Tool discovery without strategy (Crucible 4.0): search, a map of hosts, dry-run validation, reference help,
interface versions, structured error detail, and a catalogue renderer for learners with small contexts.

Why this exists. The 3.0 evaluation of frozen language models was confounded by the interface: a truncated
tool list hid most services, descriptions were cut to their first line, agents invented services and repeated
one invalid call. Everything here makes the catalogue *findable* and errors *specific*; nothing here says what
is worth doing. The line is: reference material describes what exists and how to call it; it never ranks,
recommends or sequences actions toward an outcome.

Core endpoints registered on host ``world`` (all read-only, cost 0 simulated seconds; `install`):

* ``world.find_tools {query, limit, detail, host}`` - ranked search (BM25 with name-field bonuses, `ToolIndex`)
  over tool names, hosts, complete descriptions and parameter names / descriptions, across the built-in tools
  and the agent-deployed services the caller has already discovered. It reads the caller's own tool list
  (`World.available_tools`), so it cannot reveal a service the caller does not know: finding those stays with
  the world's search engine (`search.query`).
* ``world.hosts {summary_chars}`` - every built-in host, its endpoint count and one line taken from the
  module's docstring (or an explicit ``summary`` class attribute).
* ``world.validate {tool, args}`` - dry run: resolve, apply defaults, validate against the schema; returns the
  normalised arguments and the cost, or structured errors (path, expected, given). Never calls the handler.
* ``world.help {topic}`` - reference: identifier formats, discovery, the action format, time, error kinds.
* ``world.api {}`` - kernel / API / discovery versions and a stable hash of the tool catalogue.

Research-side helpers (not reachable from inside a world): `render_catalogue`, `render_host_map`,
`render_result` / `render_error`, `interface_stamp`, `catalogue_hash`, `validate_structured`, `near_tools`,
`suggest`, `with_hints`.

World parameters (``spec.params``): ``discovery: false`` removes the five endpoints and the error detail (the
exact 3.0 interface); ``error_hints: false`` keeps the endpoints and removes only the error detail.

Determinism. Every function is a pure function of its arguments and of world state: sorted iteration, integer
term frequencies, scores accumulated in sorted term order, ties broken by name. Caches live in
``world.runtime`` (never in state) and hold only values that are functions of the static built-in catalogue.

Recorded behaviour. `with_hints` adds detail to the error *returned* to the caller of a rejected action; the
recorded ``cmd.rejected`` event keeps exactly the 3.0 error, so event logs, hash chains, replay and
regeneration are unchanged. The detail is a function of (state, tool, args) and can be recomputed from a log.
"""
from __future__ import annotations

import copy
import math
import re
import sys
from typing import Any, Iterable, Mapping

from . import schema as S
from .canonical import digest, to_jsonable
from .errors import CrucibleError, UnknownTool

DISCOVERY_VERSION = "crucible-discovery/1"
API_VERSION = "crucible-api/1"          # equals crucible.api.local.API_VERSION (asserted by tests/test_discovery.py)
ENDPOINT_VERSION = "1"                  # default version of an endpoint spec
WORLD_SUMMARY = ("Kernel services: caller identity, clock, service descriptions, tool search, dry-run validation, "
                 "reference help, interface versions.")
K1, B = 1.2, 0.75
W_NAME, W_HOST, W_PARAM, W_DESC, W_PDESC = 2, 1, 2, 1, 1     # term-frequency weights of the fields in the text
B_NAME, B_HOST, B_PARAM = 2.0, 1.0, 0.5                      # un-normalised bonuses for name-field matches
STEM_LEN = 5
MAX_SUGGESTIONS = 5
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by can do doe for from how i in is it my of on or that the this to "
                  "what which with".split())
_MALFORMED = "action must be {'tool': str, 'args': object}"   # kernel._MALFORMED (kept in sync by a test)


# ====================================================================== text
def _fold(t: str) -> str:
    """Fold common inflections so that ``accounts``/``account``, ``listing``/``list``/``listed`` and
    ``pricing``/``price`` are one term each. A deliberately small, fixed rule set (not a full stemmer)."""
    if len(t) > 3 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) >= 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    cut = False
    if len(t) >= 6 and t.endswith("ing"):
        t, cut = t[:-3], True
    elif len(t) >= 5 and t.endswith("ed"):
        t, cut = t[:-2], True
    if cut and len(t) >= 3 and t[-1] == t[-2] and t[-1] not in "lsz":     # running -> run, billing -> bill
        t = t[:-1]
    if len(t) >= 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: Any) -> list[str]:
    """Lowercase alphanumeric word tokens with plural folding (documents and queries use the same function)."""
    return [_fold(t) for t in _TOKEN.findall(str(text).lower()) if len(t) <= 40]


def _stem(t: str) -> str | None:
    """Truncation stem of a long token (``deployment`` and ``deploy`` share ``~deplo``)."""
    return "~" + t[:STEM_LEN] if len(t) > STEM_LEN else None


def _cut(s: str, n: int) -> str:
    """At most `n` characters, cut at a word boundary and marked with ``...`` when shortened."""
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    if n <= 3:
        return s[:n]
    head = s[: n - 3]
    sp = head.rfind(" ")
    if sp >= max(8, (n - 3) // 2):
        head = head[:sp]
    return head.rstrip(" ,;:(") + "..."


def flat(text: Any) -> str:
    """The complete text on one line (whitespace runs collapsed). Nothing is dropped."""
    return " ".join(str(text or "").split())


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[{(\"'`<])")


def first_sentence(text: Any, chars: int = 160) -> str:
    s = flat(text)
    m = _SENT_END.search(s)
    if m:
        s = s[: m.start()]
    return _cut(s, chars)


# ====================================================================== specs
def _type_str(s: Mapping) -> str:
    if "enum" in s:
        return "|".join(str(v) for v in s["enum"])
    t = s.get("type", "any")
    if isinstance(t, list):
        t = "|".join(t)
    if t == "array" and isinstance(s.get("items"), Mapping):
        return f"{_type_str(s['items'])}[]"
    return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj"}.get(t, t)


def _params(spec: Mapping) -> list[tuple[str, bool, Mapping]]:
    schema = spec.get("inputSchema") or {}
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    return [(n, n in req, props[n]) for n in sorted(props, key=lambda n: (n not in req, n))]


def signature(spec: Mapping, types: bool = True) -> str:
    """``name(required: type, optional?: type)`` - required arguments first, then optional ones, by name."""
    parts = []
    for n, req, p in _params(spec):
        s = n if req else n + "?"
        parts.append(f"{s}: {_type_str(p)}" if types else s)
    return f"{spec.get('name')}({', '.join(parts)})"


def _host_of(spec: Mapping) -> str:
    return (spec.get("annotations") or {}).get("host") or str(spec.get("name", "")).partition(".")[0]


def _cost(spec: Mapping) -> int:
    return int((spec.get("annotations") or {}).get("costSeconds") or 0)


def catalogue_hash(tools: Iterable[Mapping]) -> str:
    """Stable hash of a tool catalogue: names, complete descriptions, schemas, costs, read-only flags, versions.
    Independent of the order of `tools` and of dict insertion order."""
    rows = []
    for t in tools:
        a = t.get("annotations") or {}
        rows.append([t.get("name"), t.get("description") or "", t.get("inputSchema") or {},
                     int(a.get("costSeconds") or 0), bool(a.get("readOnlyHint")), str(a.get("version", ENDPOINT_VERSION))])
    rows.sort(key=lambda r: str(r[0]))
    return digest(rows)


def builtin_tools(world, principal: str | None = None) -> list[dict]:
    """Specs of the built-in tools an in-world actor can call (operator tools only for the system principal)."""
    from . import evaluator_only as EO
    from .kernel import SYSTEM
    return [e.spec() for n, e in sorted(world.endpoints.items()) if (e.host != "admin" or principal == SYSTEM)
            and EO.participant_tool_allowed(world, principal or "", n, e)]


def known_extra_hosts(world, principal: str) -> tuple[str, ...]:
    """Agent-deployed hosts `principal` has learned of (its knowledge minus the built-in hosts), sorted."""
    st = world.state
    hit = world.runtime.get("discovery.builtin_hosts")
    if hit is None or hit[0] != len(world.endpoints):
        hit = world.runtime["discovery.builtin_hosts"] = (len(world.endpoints),
                                                          frozenset(e.host for e in world.endpoints.values()))
    builtin = hit[1]
    know = st.tables["knowledge"]
    return tuple(sorted({know[k]["value"] for k in st.find_ids("knowledge", "principal", principal)
                         if know[k]["kind"] == "host"} - builtin))


def visible_tools(world, principal: str) -> list[dict]:
    """`World.available_tools(principal)`: the built-in tools plus those of the deployed hosts the principal
    knows. For a principal that knows no deployed host the (static) built-in list is built once per world and
    shared - callers must treat the specs as read-only."""
    from .kernel import SYSTEM
    if known_extra_hosts(world, principal):
        return world.available_tools(principal)
    from . import evaluator_only as EO
    cache = world.runtime.setdefault("discovery.builtin_tools", {})
    key = (principal == SYSTEM, len(world.endpoints), EO.policy(world))
    if key not in cache:
        cache[key] = world.available_tools(principal)
    return cache[key]


def interface_stamp(world, principal: str | None = None) -> dict:
    """What a learner or a log needs to state which interface it ran against. ``catalogue`` covers the built-in
    tools every actor sees; with `principal`, ``visible`` also covers the deployed services it has discovered."""
    from .kernel import KERNEL_VERSION
    base = world.runtime.get("discovery.stamp")          # the built-in catalogue is fixed for the life of a world
    if base is None or base["tools"] != sum(1 for e in world.endpoints.values() if e.host != "admin"):
        tools = builtin_tools(world)
        base = world.runtime["discovery.stamp"] = {
            "kernel": KERNEL_VERSION, "api": API_VERSION, "discovery": DISCOVERY_VERSION,
            "catalogue": catalogue_hash(tools), "tools": len(tools), "hosts": len({_host_of(t) for t in tools}),
            "modules": {n: str(m.version) for n, m in sorted(world.modules.items())}}
    out = copy.deepcopy(base)
    if principal is not None:
        vis = world.available_tools(principal)
        same = len(vis) == base["tools"] and all((t.get("annotations") or {}).get("builtin", True) for t in vis)
        out["visible"] = {"hash": base["catalogue"] if same else catalogue_hash(vis), "tools": len(vis)}
    return out


# ====================================================================== ranked search (BM25)
def _terms(text: Any) -> list[str]:
    """Tokens of `text` followed by the truncation stems of the long ones."""
    toks = tokenize(text)
    return toks + [s for s in map(_stem, toks) if s is not None]


def _add(tf: dict, text: Any, w: int) -> None:
    for t in _terms(text):
        tf[t] = tf.get(t, 0) + w


def _walk_params(schema: Any, tf: dict, names: set, depth: int = 0) -> None:
    if not isinstance(schema, Mapping) or depth > 4:
        return
    for name, p in sorted((schema.get("properties") or {}).items()):
        _add(tf, name, W_PARAM)
        if depth == 0:
            names.update(_terms(name))
        if isinstance(p, Mapping):
            _add(tf, p.get("description") or "", W_PDESC)
            for v in p.get("enum") or ():
                _add(tf, v, W_PDESC)
            _walk_params(p, tf, names, depth + 1)
            if isinstance(p.get("items"), Mapping):
                _add(tf, p["items"].get("description") or "", W_PDESC)
                _walk_params(p["items"], tf, names, depth + 1)


def doc_terms(spec: Mapping) -> tuple[dict[str, int], frozenset, frozenset, frozenset]:
    """One tool as a search document: (term frequencies of its whole text - endpoint name, host, complete
    description, parameter names and descriptions including nested ones, enum values - , terms of the endpoint
    name, terms of the host, terms of the top-level parameter names)."""
    tf: dict[str, int] = {}
    name = str(spec.get("name", ""))
    host, _, ep = name.partition(".")
    host = _host_of(spec) or host
    _add(tf, ep or name, W_NAME)
    _add(tf, host, W_HOST)
    _add(tf, spec.get("description") or "", W_DESC)
    pnames: set[str] = set()
    _walk_params(spec.get("inputSchema") or {}, tf, pnames)
    return tf, frozenset(_terms(ep or name)), frozenset(_terms(host)), frozenset(pnames)


class ToolIndex:
    """Ranked search over a list of tool specs.

    score(tool) = coverage * sum over query terms t of idf(t) * (bm25(tf_t) + B_NAME*[t in endpoint name]
                                                              + B_HOST*[t in host] + B_PARAM*[t in a parameter name])

    * bm25 is the usual saturating, length-normalised term weight (k1 1.2, b 0.75) over the tool's whole text;
    * the three bonuses are not length-normalised, so a word of the tool's own name counts fully whatever the
      length of its description (BM25F-style field weighting);
    * a query term is each distinct word, its truncation stem, and - for words of 4+ letters - every name term
      of the catalogue that is a prefix of it (``organisation`` reaches ``org``, ``repository`` reaches ``repo``);
    * coverage = (1 + fraction of the query's distinct words the tool matches) / 2.

    `cache` (name -> document) may be shared between indexes for specs that never change (built-in tools)."""

    def __init__(self, specs: Iterable[Mapping], cache: dict | None = None):
        self.specs: dict[str, Mapping] = {}
        self.docs: dict[str, tuple] = {}
        self.dl: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.post: dict[str, list[tuple[str, int]]] = {}      # term -> [(tool, term frequency)] in catalogue order
        self.name_terms: set[str] = set()
        for sp in specs:
            n = str(sp.get("name"))
            if n in self.specs:
                continue
            builtin = (sp.get("annotations") or {}).get("builtin", True)
            doc = cache.get(n) if (cache is not None and builtin) else None
            if doc is None:
                doc = doc_terms(sp)
                if cache is not None and builtin:
                    cache[n] = doc
            self.specs[n] = sp
            self.docs[n] = doc
            self.dl[n] = sum(doc[0].values())
            for t, f in doc[0].items():
                self.df[t] = self.df.get(t, 0) + 1
                self.post.setdefault(t, []).append((n, f))
            self.name_terms |= doc[1] | doc[2] | doc[3]
        self.n = len(self.specs)
        self.avg = (sum(self.dl.values()) / self.n) if self.n else 1.0
        self._abbr = frozenset(t for t in self.name_terms if len(t) >= 3 and not t.startswith("~"))

    def query_words(self, query: Any) -> list[tuple[str, tuple[str, ...]]]:
        """Distinct query words (stop words removed, sorted), each with the index terms that count as a match:
        the word, its stem, and name terms that abbreviate it."""
        out = []
        for w in sorted({t for t in tokenize(query) if t not in _STOP}):
            alts = [w]
            st = _stem(w)
            if st is not None:
                alts.append(st)
            if len(w) >= 4:                       # name terms that are proper prefixes of the word, shortest first
                alts.extend(w[:k] for k in range(3, len(w)) if w[:k] in self._abbr)
            out.append((w, tuple(alts)))
        return out

    def query_terms(self, query: Any) -> list[str]:
        return sorted({a for _, alts in self.query_words(query) for a in alts})

    def search(self, query: Any, limit: int = 8, host: str | None = None) -> tuple[list[tuple[float, str]], int]:
        """([(score, name)] best first, number of tools with a positive score)."""
        words = self.query_words(query)
        scores: dict[str, float] = {}
        for t in sorted({a for _, alts in words for a in alts}):   # sorted: float accumulation is reproducible
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for n, f in self.post[t]:
                _, nm, hs, pm = self.docs[n]
                w = f * (K1 + 1) / (f + K1 * (1 - B + B * self.dl[n] / self.avg))
                w += (B_NAME if t in nm else 0.0) + (B_HOST if t in hs else 0.0) + (B_PARAM if t in pm else 0.0)
                scores[n] = scores.get(n, 0.0) + idf * w
        if len(words) > 1:
            for n in scores:
                tf = self.docs[n][0]
                hit = sum(1 for _, alts in words if any(a in tf for a in alts))
                scores[n] *= 0.5 + 0.5 * hit / len(words)
        q = str(query).strip().lower()
        if q in self.specs:                      # an exact tool name always ranks first
            scores[q] = scores.get(q, 0.0) + 1000.0
        keyed = sorted((-s, n) for n, s in scores.items()
                       if s > 0 and (host is None or _host_of(self.specs[n]) == host))
        return [(-s, n) for s, n in keyed[: max(0, int(limit))]], len(keyed)


def _index(world, tools: list[dict]) -> ToolIndex:
    """Index over `tools`, cached in ``world.runtime`` by the exact content of the list (the built-in part is
    static for a world; discovered services are keyed by a hash of their specifications)."""
    extra = [t for t in tools if not (t.get("annotations") or {}).get("builtin", True)]
    key = (len(tools) - len(extra), catalogue_hash(extra) if extra else "")
    cache = world.runtime.setdefault("discovery.index", {})
    idx = cache.get(key)
    if idx is None:
        if len(cache) >= 64:
            cache.clear()
        idx = cache[key] = ToolIndex(tools, world.runtime.setdefault("discovery.docs", {}))
    return idx


def _entry(spec: Mapping, score: float | None, detail: bool) -> dict:
    out = {"name": spec["name"], "host": _host_of(spec), "signature": signature(spec),
           "summary": first_sentence(spec.get("description"), 160)}
    if score is not None:
        out["score"] = round(float(score), 4)
    if detail:
        out["spec"] = to_jsonable(spec)
    return out


def find_tools(world, principal: str, query: str, limit: int = 8, detail: bool = False,
               host: str | None = None) -> dict:
    """Ranked search over the tools `principal` can already see. Pure: reads state, changes nothing."""
    tools = visible_tools(world, principal)
    hosts = sorted({_host_of(t) for t in tools})
    out: dict[str, Any] = {"query": query, "searched": len(tools)}
    if host is not None and host not in hosts:
        out.update(results=[], matched=0, unknown_host=host, near_hosts=near_hosts(world, host, hosts))
        return out
    idx = _index(world, tools)
    if not idx.query_words(query) and str(query).strip().lower() not in idx.specs:
        if host is None:
            out.update(results=[], matched=0, note="the query has no searchable words; world.hosts lists every host")
            return out
        rows = [t for t in tools if _host_of(t) == host]            # browse one host, by name
        out.update(results=[_entry(t, None, detail) for t in rows[:limit]], matched=len(rows))
        return out
    hits, matched = idx.search(query, limit, host)
    out.update(results=[_entry(idx.specs[n], s, detail) for s, n in hits], matched=matched)
    return out


# ====================================================================== hosts
_PAREN = re.compile(r"\s*\((?:modules?|host)\b[^)]*\)")


def module_summary(mod) -> str:
    """One line for a module: its ``summary`` class attribute, else the first sentence of the class docstring
    (if the class has its own), else the first sentence of the file's docstring."""
    cls = type(mod)
    s = cls.__dict__.get("summary")
    if not isinstance(s, str) or not s.strip():
        s = cls.__dict__.get("__doc__") or ""
        if not s.strip():
            s = getattr(sys.modules.get(cls.__module__), "__doc__", None) or ""
    para = str(s).strip().split("\n\n")[0]
    return first_sentence(_PAREN.sub("", flat(para)), 400)


def host_summaries(world) -> dict[str, str]:
    """host -> one-line statement of what the service is, for every built-in host that has endpoints."""
    out = {"world": WORLD_SUMMARY}
    live = {e.host for e in world.endpoints.values()}
    for name in sorted(world.modules):
        mod = world.modules[name]
        h = mod.host or mod.name
        if h in live and h not in out:
            out[h] = module_summary(mod)
    for h in sorted(live):
        out.setdefault(h, "")
    return out


def host_aliases(world) -> dict[str, str]:
    """Other names under which a built-in host is plausibly sought: its module name and the name of the file that
    defines it (``identity`` -> ``id``, ``workflows`` -> ``wf``). Derived from the code, not hand-written."""
    out: dict[str, str] = {}
    live = {e.host for e in world.endpoints.values()}
    for name in sorted(world.modules):
        mod = world.modules[name]
        h = mod.host or mod.name
        if h not in live:
            continue
        for alias in (mod.name, type(mod).__module__.rsplit(".", 1)[-1]):
            alias = alias.lower()
            if alias != h and alias not in live:
                out.setdefault(alias, h)
    return out


def hosts(world, principal: str, summary_chars: int = 72) -> dict:
    """Map of the world's services for `principal`: ``{"hosts": {host: [number of tools, one-line description]},
    "tools": total, "discovered": {host: number of tools}}`` - every built-in host, and the agent-deployed hosts
    the principal already knows (count only). The keyed, positional form keeps the whole map small."""
    from . import evaluator_only as EO
    from .kernel import SYSTEM
    counts: dict[str, int] = {}
    for n, e in world.endpoints.items():
        if (e.host != "admin" or principal == SYSTEM) and EO.participant_tool_allowed(world, principal, n, e):
            counts[e.host] = counts.get(e.host, 0) + 1
    summ = host_summaries(world)
    known: dict[str, int] = {}
    for t in visible_tools(world, principal):
        if not (t.get("annotations") or {}).get("builtin", True):
            known[_host_of(t)] = known.get(_host_of(t), 0) + 1
    return {"hosts": {h: [counts[h], _cut(summ.get(h, ""), int(summary_chars))] for h in sorted(counts)},
            "tools": sum(counts.values()), "discovered": dict(sorted(known.items()))}


def render_host_map(result: Mapping) -> str:
    """Compact text of a ``world.hosts`` result: one line per host."""
    lines = [f"{h} ({n}): {s}".rstrip(": ") for h, (n, s) in result.get("hosts", {}).items()]
    for h, n in result.get("discovered", {}).items():
        lines.append(f"{h} ({n}): agent-deployed service")
    return "\n".join(lines)


# ====================================================================== near names
def edit_distance(a: str, b: str, cap: int = 99) -> int:
    """Optimal-string-alignment distance (insert, delete, substitute, adjacent transposition), or ``cap + 1`` as
    soon as it is known to exceed `cap`. Only the band of width ``2 * cap + 1`` around the diagonal is computed."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    big = cap + 1
    prev2: list[int] | None = None
    prev = [j if j <= cap else big for j in range(lb + 1)]
    for i in range(1, la + 1):
        lo, hi = max(1, i - cap), min(lb, i + cap)
        cur = [big] * (lb + 1)
        cur[0] = i if i <= cap else big
        best = cur[0]
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            v = prev[j - 1] + (0 if ai == b[j - 1] else 1)
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            if prev2 is not None and j > 1 and ai == b[j - 2] and a[i - 2] == b[j - 1] and prev2[j - 2] + 1 < v:
                v = prev2[j - 2] + 1
            if v > big:
                v = big
            cur[j] = v
            if v < best:
                best = v
        if best > cap:
            return big
        prev2, prev = prev, cur
    return prev[lb] if prev[lb] <= cap else big


def near_names(name: str, candidates: Iterable[str], limit: int = 3, cap: int | None = None) -> list[str]:
    """Candidates within edit distance `cap` (default: a quarter of the name's length, between 1 and 3)."""
    name = str(name)
    if cap is None:
        cap = max(1, min(3, len(name) // 4))
    low = name.lower()
    chars, ln = set(low), len(low)
    scored = []
    for c in sorted(set(candidates)):
        cl = c.lower()
        # exact lower bounds on the distance: the difference in length, and half the symmetric difference of
        # the two character sets (one edit changes that set difference by at most two)
        if abs(len(cl) - ln) > cap or len(chars ^ set(cl)) > 2 * cap:
            continue
        d = edit_distance(low, cl, cap)
        if d <= cap:
            scored.append((d, c))
    return [c for _, c in sorted(scored)[:limit]]


def near_host_names(host: str, known_hosts: Iterable[str], aliases: Mapping[str, str] | None = None,
                    limit: int = 3) -> list[str]:
    """Existing hosts a nonexistent host name may have meant: an alias (module or file name, see `host_aliases`),
    a host that is a prefix of the name or has the name as a prefix, or a host within edit distance 2."""
    host = str(host).lower()
    if host.endswith(".crucible"):
        host = host[: -len(".crucible")]
    known = sorted(set(known_hosts))
    out: list[str] = []
    alias = (aliases or {}).get(host)
    if alias in known:
        out.append(alias)
    for h in known:
        if h not in out and len(h) >= 2 and len(host) >= 2 and (host.startswith(h) or h.startswith(host)):
            out.append(h)
    for h in near_names(host, known, limit, cap=2 if len(host) > 3 else 1):
        if h not in out:
            out.append(h)
    return out[:limit]


def near_hosts(world, host: str, known_hosts: Iterable[str], limit: int = 3) -> list[str]:
    return near_host_names(host, known_hosts, host_aliases(world), limit)


def near_tools(tool: str, names: Iterable[str], aliases: Mapping[str, str] | None = None) -> list[str]:
    """Existing tool names of which `tool` is a *near miss* - the test for "the intended tool existed under a
    near name". Exactly these relations count, in this order:

    1. the whole name is within a small edit distance (a quarter of its length, between 1 and 3) of an existing
       name (a typo);
    2. the host does not exist but is an alias, a prefix or an extension of an existing host, or within edit
       distance 2 of one, and on that host the endpoint exists, or one within edit distance 2 does, or one made
       of the same words (``identity.create_org`` -> ``id.create_org``);
    3. the host exists and one of its endpoints is within a small edit distance of the one asked for (a third
       of the endpoint's length, between 1 and 3), or consists of a subset or superset of its words, a word and
       its abbreviation counting as the same (``cloud.deploy_service`` -> ``cloud.deploy``, ``id.org_create``
       and ``id.create_organization`` -> ``id.create_org``).

    A tool that merely does something similar under an unrelated name is not a near miss."""
    tool = str(tool)
    names = sorted(set(names))
    host, _, ep = tool.partition(".")
    hosts_ = sorted({n.partition(".")[0] for n in names})
    out: list[str] = []

    def add(ns: Iterable[str]) -> None:
        for n in ns:
            if n not in out and n != tool:
                out.append(n)

    def covers(a: set, b: set) -> bool:
        """Every word of `a` is in `b`, or abbreviates / is abbreviated by a word of `b` (``org``/``organisation``)."""
        return all(any(x == y or (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))) for y in b)
                   for x in a)

    def same_words(h: str) -> list[str]:
        want = set(tokenize(ep))
        hits = []
        for n in names:
            if n.startswith(h + "."):
                have = set(tokenize(n.partition(".")[2]))
                if want and have and (covers(want, have) or covers(have, want)):
                    hits.append((abs(len(want) - len(have)), n))
        return [n for _, n in sorted(hits)[:2]]
    add(near_names(tool, names, 3))
    if not ep:
        return out
    if host not in hosts_:
        for h in near_host_names(host, hosts_, aliases):
            on_h = [n for n in names if n.startswith(h + ".")]
            add([f"{h}.{ep}"] if f"{h}.{ep}" in on_h else [])
            add(near_names(f"{h}.{ep}", on_h, 2, cap=2))
            add(same_words(h))
    else:
        add(near_names(tool, [n for n in names if n.startswith(host + ".")], 3, cap=max(1, min(3, len(ep) // 3))))
        add(same_words(host))
    return out


def suggest(world, principal: str, tool: str) -> dict:
    """Near misses for a tool name that does not resolve, drawn only from tools `principal` can already see:
    first the near names (`near_tools`), then the best matches of a search for the words of the name."""
    tool = str(tool)[:200]
    # the caller's catalogue is the built-in tools plus the deployed hosts it knows: that set of hosts keys the cache
    known_extra = known_extra_hosts(world, principal)
    cache = world.runtime.setdefault("discovery.suggest", {})
    key = (principal == "sys", tool, known_extra, len(world.endpoints))
    hit = cache.get(key)
    if hit is not None and not known_extra:      # deployed services can change their endpoints: no caching then
        return copy.deepcopy(hit)
    tools = visible_tools(world, principal)
    names = [t["name"] for t in tools]
    host, _, ep = tool.partition(".")
    all_hosts = sorted({_host_of(t) for t in tools})
    aliases = host_aliases(world)
    out: list[str] = near_tools(tool, names, aliases)
    res: dict[str, Any] = {"host": host, "host_exists": host in all_hosts}
    if host not in all_hosts:
        res["near_hosts"] = near_host_names(host, all_hosts, aliases)
    res["near"] = len(out)                       # how many of the suggestions are near names (the first ones)
    if len(out) < MAX_SUGGESTIONS:               # then: ranked search for the words of the name; a suggestion must
        want = set(_terms(ep or host))           # share a word with the endpoint part of what was asked for - in
        idx = _index(world, tools)               # its own endpoint name, or anywhere in its text if it is a tool
        hits, _ = idx.search(tool.replace(".", " ").replace("_", " "), 4 * MAX_SUGGESTIONS)   # of the host named
        for _, n in hits:
            have = set(_terms(n.partition(".")[2] if ep else n))
            same_host = host in all_hosts and n.startswith(host + ".")
            if n not in out and n != tool and (want & have or (same_host and want & set(idx.docs[n][0]))):
                out.append(n)
    res["suggestions"] = out[:MAX_SUGGESTIONS]
    if host in all_hosts:
        eps = sorted(n.partition(".")[2] for n in names if n.startswith(host + "."))
        res["host_tools"] = eps[:40]
    if not known_extra:
        if len(cache) >= 2048:
            cache.clear()
        cache[key] = copy.deepcopy(res)
    return res


# ====================================================================== structured schema validation
def _brief(v: Any) -> Any:
    """A short, JSON-safe rendering of an offending value."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else repr(v)
    if isinstance(v, str):
        return v if len(v) <= 80 else v[:77] + "..."
    if isinstance(v, list):
        return f"array of {len(v)} items"
    if isinstance(v, dict):
        keys = sorted(str(k) for k in v)
        return "object with keys [" + ", ".join(keys[:8]) + (", ..." if len(keys) > 8 else "") + "]"
    return type(v).__name__


def _expect(s: Mapping) -> str:
    """Human-readable statement of what a schema node accepts."""
    if "enum" in s:
        return "one of " + ", ".join(repr(v) for v in s["enum"])
    t = s.get("type")
    parts = [" or ".join(t) if isinstance(t, list) else (t or "any value")]
    for k, label in (("pattern", "matching"), ("minLength", "min length"), ("maxLength", "max length"),
                     ("minimum", ">="), ("maximum", "<="), ("minItems", "min items"), ("maxItems", "max items")):
        if k in s and not (k == "maxLength" and s[k] >= 4000) and not (k == "maxItems" and s[k] >= 1000):
            parts.append(f"{label} {s[k]}")
    return ", ".join(parts)


def validate_structured(schema: Mapping, value: Any, path: str = "args") -> list[dict]:
    """The same checks, in the same order, as `schema.validate`, as structured errors
    ``{"path", "kind", "expected", "given", "message"}``; ``message`` is exactly the text `schema.validate`
    produces for that error (a test compares the two on random schemas and values)."""
    errs: list[dict] = []
    _vs(schema, value, path, errs)
    return errs


def _vs(s: Mapping, v: Any, path: str, errs: list[dict]) -> None:
    def err(kind: str, expected: Any, message: str, given: Any = None, **kw) -> None:
        errs.append({"path": path, "kind": kind, "expected": expected,
                     "given": _brief(v) if given is None else given, "message": message, **kw})
    t = s.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(S._TYPES[x](v) for x in ts):
            err("type", " or ".join(ts), f"{path}: expected {t}, got {type(v).__name__}",
                given_type=type(v).__name__)
            return
    if "enum" in s and v not in s["enum"]:
        near = near_names(v, [x for x in s["enum"] if isinstance(x, str)], 1) if isinstance(v, str) else []
        err("enum", "one of " + ", ".join(repr(x) for x in s["enum"]), f"{path}: {v!r} not one of {s['enum']}",
            **({"nearest": near[0]} if near else {}))
    if isinstance(v, str):
        if len(v) > s.get("maxLength", S.MAX_STRING):
            err("max_length", f"at most {s.get('maxLength', S.MAX_STRING)} characters",
                f"{path}: longer than {s.get('maxLength', S.MAX_STRING)}", given=f"{len(v)} characters")
        if len(v) < s.get("minLength", 0):
            err("min_length", f"at least {s['minLength']} characters", f"{path}: shorter than {s['minLength']}",
                given=f"{len(v)} characters")
        if "pattern" in s and not re.fullmatch(s["pattern"], v):
            err("pattern", f"a string matching {s['pattern']}", f"{path}: does not match {s['pattern']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            err("minimum", f">= {s['minimum']}", f"{path}: below minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            err("maximum", f"<= {s['maximum']}", f"{path}: above maximum {s['maximum']}")
        if isinstance(v, float) and v != v:
            err("nan", "a number", f"{path}: NaN")
    if isinstance(v, list):
        if len(v) > s.get("maxItems", 10_000):
            err("max_items", f"at most {s.get('maxItems', 10_000)} items",
                f"{path}: more than {s.get('maxItems', 10_000)} items", given=f"{len(v)} items")
        if len(v) < s.get("minItems", 0):
            err("min_items", f"at least {s['minItems']} items", f"{path}: fewer than {s['minItems']} items",
                given=f"{len(v)} items")
        if "items" in s:
            for i, x in enumerate(v):
                _vs(s["items"], x, f"{path}[{i}]", errs)
    if isinstance(v, dict) and (s.get("type") == "object" or "properties" in s):
        props = s.get("properties", {})
        for k in sorted(s.get("required", [])):
            if k not in v:
                p = props.get(k) or {}
                errs.append({"path": f"{path}.{k}", "kind": "required", "expected": _expect(p), "given": "missing",
                             "message": f"{path}.{k}: required"})
        addl = s.get("additionalProperties", False)
        for k, x in sorted(v.items(), key=lambda kv: str(kv[0])):
            if k in props:
                _vs(props[k], x, f"{path}.{k}", errs)
            elif addl is False:
                e = {"path": f"{path}.{k}", "kind": "unexpected",
                     "expected": "one of the declared arguments: " + (", ".join(sorted(props)) or "(none)"),
                     "given": str(k)[:80], "message": f"{path}.{k}: unexpected property"}
                near = near_names(str(k), list(props), 1)
                if near:
                    e["nearest"] = near[0]
                errs.append(e)
            elif isinstance(addl, dict):
                _vs(addl, x, f"{path}.{k}", errs)


def _sorted_keys(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _sorted_keys(x[k]) for k in sorted(x, key=str)}
    if isinstance(x, list):
        return [_sorted_keys(v) for v in x]
    return x


NOT_CHECKED = ["permissions", "existence of the things the arguments refer to", "funds and capacity",
               "the tool's own rules"]


def dry_run(world, principal: str, tool: str, args: Mapping | None = None) -> dict:
    """Check an action without running it: resolve the tool, apply defaults, validate the arguments against the
    schema. The handler is never called; nothing is emitted. ``valid`` therefore means only that the tool exists
    and the arguments are well-formed (``not_checked`` lists what only execution decides)."""
    from .kernel import SYSTEM
    tool = str(tool)
    try:
        ep = world.resolve(tool, principal)
    except UnknownTool as e:
        return {"valid": False, "tool": tool,
                "errors": [{"path": "tool", "kind": "unknown_tool", "expected": "the name of an existing tool, "
                            "<host>.<endpoint>", "given": tool[:120], "message": e.message}],
                **suggest(world, principal, tool)}
    if ep.host == "admin" and principal != SYSTEM:
        return {"valid": False, "tool": tool,
                "errors": [{"path": "tool", "kind": "forbidden", "expected": "a tool outside host admin",
                            "given": tool, "message": "admin tools are reserved for the simulator operator"}]}
    norm = _sorted_keys(S.with_defaults(ep.params, dict(args or {})))
    errs = validate_structured(ep.params, norm)
    spec = ep.spec()
    if errs:
        return {"valid": False, "tool": tool, "errors": errs[:20], "error_count": len(errs),
                "signature": signature(spec)}
    return {"valid": True, "tool": tool, "args": to_jsonable(norm), "cost_s": ep.cost_s, "read_only": ep.read_only,
            "version": spec["annotations"].get("version", ENDPOINT_VERSION), "checked": ["tool name", "argument schema"],
            "not_checked": list(NOT_CHECKED)}


# ====================================================================== error detail for rejected actions
def hints(world, actor: str, tool: Any, args: Any, error: Mapping, ep=None) -> dict:
    """Structured detail for the error of a rejected action (empty when there is nothing to add)."""
    code, msg = error.get("code"), str(error.get("message", ""))
    if code == "unknown_tool" and isinstance(tool, str):
        if tool == "world.describe" and isinstance(args, Mapping) and isinstance(args.get("host"), str):
            known = sorted({_host_of(t) for t in visible_tools(world, actor)})
            return {"near_hosts": near_hosts(world, args["host"], known), "see": "world.hosts"}
        if msg == f"unknown tool {tool!r}":         # not an unknown tool met by a nested call of an existing one
            return {**suggest(world, actor, tool), "see": "world.find_tools"}
        return {}
    if code == "invalid_action" and msg == _MALFORMED:
        errs = []
        if not isinstance(tool, str):
            errs.append({"path": "tool", "kind": "type", "expected": "string <host>.<endpoint>",
                         "given": type(tool).__name__, "message": "tool: expected string"})
        if not isinstance(args, Mapping):
            errs.append({"path": "args", "kind": "type", "expected": "object of JSON values (no NaN or Infinity)",
                         "given": type(args).__name__, "message": "args: expected object"})
        return {"errors": errs}
    if code == "invalid_action" and ep is not None and isinstance(args, Mapping):
        errs = validate_structured(ep.params, args)
        if errs:
            return {"errors": errs[:6], "error_count": len(errs), "signature": signature(ep.spec())}
    return {}


def with_hints(world, actor: str, tool: Any, args: Any, res, ep=None):
    """Return the `ActionResult` of a rejected action with `hints` merged into ``error["details"]``. Only the
    returned result changes: the recorded event keeps the original error. Disabled by world parameter
    ``error_hints: false``."""
    err = getattr(res, "error", None)
    if not err or getattr(res, "replayed", False) or world.spec.params.get("error_hints", True) is False:
        return res
    try:
        extra = hints(world, actor, tool, args, err, ep)
    except CrucibleError:
        return res
    except Exception:  # noqa: BLE001 - detail is best-effort: it must never turn a rejection into a crash
        if world.strict:
            raise
        return res
    if extra:
        res.error = {"code": err.get("code"), "message": err.get("message"),
                     "details": {**copy.deepcopy(err.get("details") or {}), **to_jsonable(extra)}}
    return res


def render_error(error: Mapping | None, chars: int = 400) -> str:
    """One line for prompts: ``code: message`` followed by the structured detail an agent can act on."""
    if not error:
        return ""
    out = f"{error.get('code', 'error')}: {error.get('message', '')}"
    d = error.get("details") or {}
    extra = []
    if d.get("suggestions"):
        extra.append("existing tools with near names: " + ", ".join(d["suggestions"]))
    if d.get("near_hosts"):
        extra.append("existing hosts with near names: " + ", ".join(d["near_hosts"]))
    for e in (d.get("errors") or [])[:4]:
        s = f"{e.get('path')}: expected {e.get('expected')}, given {e.get('given')!r}"
        if e.get("nearest"):
            s += f" (nearest: {e['nearest']})"
        extra.append(s)
    if d.get("signature"):
        extra.append("signature: " + d["signature"])
    if extra:
        out += " | " + " | ".join(extra)
    return _cut(out, chars)


def render_result(res: Mapping | None, chars: int = 600) -> str:
    """Like `api.render.render_result`, but a failure shows its structured detail (`render_error`)."""
    if not res:
        return "none"
    tool = res.get("tool") or "?"
    if res.get("ok"):
        import json
        s = json.dumps(res.get("result"), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
        out = f"{tool} -> ok {s if len(s) <= chars else s[: max(0, chars - 3)] + '...'}"
    else:
        out = f"{tool} -> FAILED {render_error(res.get('error') or {}, chars)}"
    if res.get("cost_s"):
        out += f" (took {res['cost_s']}s)"
    if res.get("replayed"):
        out += " (idempotent replay)"
    return out


# ====================================================================== reference help
TOPICS = ("index", "identifiers", "discovery", "actions", "time", "errors", "versions")

ERROR_MEANINGS = {
    "invalid_action": "The action was not of the form {tool, args}, or an argument did not satisfy the tool's "
                      "schema (details.errors gives path, expected and given), or the tool rejected the arguments "
                      "under one of its documented rules.",
    "unknown_tool": "No tool, or no service host, has that name (details.suggestions and details.near_hosts list "
                    "existing names that are close).",
    "unauthenticated": "The acting principal is not an active actor of this world.",
    "forbidden": "The principal lacks the role, grant or ownership the tool requires for this target.",
    "not_found": "An argument refers to something that does not exist, or that the caller cannot see.",
    "conflict": "The request collides with existing state: a name is taken, the object exists already, or it "
                "is in a status that does not admit the operation.",
    "insufficient_funds": "The paying account's balance plus its overdraft limit is below the amount.",
    "capacity_exceeded": "A finite resource has no free capacity for the request.",
    "rate_limited": "The limit on actions or requests in the current period is reached.",
    "service_unavailable": "The service is down, suspended, overloaded or disabled.",
    "timeout": "An execution exceeded its time limit.",
    "sandbox_error": "Running code in the sandbox failed before or outside the program itself.",
    "budget_exhausted": "A budget assigned to the caller is used up.",
    "error": "An internal error of the simulator; it is not caused by the content of the action.",
}

ID_FORMATS = {   # module name -> [(what, format)]; each prefix is the one the module passes to StateStore.new_id
    "identity": [("invitation", "inv_<6 digits>"), ("grant (delegation)", "grant_<6 digits>")],
    "bank": [("account", "acct_<6 digits>, in order of opening (acct_000001 is the first); system accounts have "
                         "fixed names such as acct_fees"),
             ("transaction", "tx_<6 digits>"), ("invoice", "inv_<6 digits>"), ("mandate", "mandate_<6 digits>")],
    "market": [("listing", "lst_<6 digits>"), ("order", "ord_<6 digits>")],
    "cloud": [("instance", "inst_<6 digits>"), ("deployment", "dep_<6 digits>"), ("hardware order", "hw_<6 digits>")],
}


def _mod_const(world, module: str, const: str, default: str) -> str:
    mod = world.modules.get(module)
    if mod is None:
        return default
    return str(getattr(sys.modules.get(type(mod).__module__), const, default))


def help_topics(world) -> dict[str, str]:
    """Reference text per topic, generated from this world's modules and parameters. It states formats and
    mechanics only; it contains no guidance about what to do."""
    from .kernel import KERNEL_VERSION
    handle = _mod_const(world, "identity", "HANDLE", "[a-z][a-z0-9-]{1,30}")
    host = _mod_const(world, "cloud", "HOST", "[a-z][a-z0-9-]{2,30}")
    ids = ["Principals: a person is p:<handle>, an organisation is o:<handle>, a deployed service is svc:<host>, "
           f"the kernel is sys. A handle matches {handle}; persons and organisations share one handle namespace. "
           "Arguments that name a principal accept the full id or the bare handle.",
           f"Hosts and tools: a tool is <host>.<endpoint>. Built-in hosts are listed by world.hosts. A deployed "
           f"service host matches {host}; its tools are <host>.<endpoint> and its network name is <host>.crucible.",
           "Allocated identifiers have the form <prefix>_<6 digits>, numbered from 000001 per prefix in order of "
           "creation, and are returned by the tool that creates the object."]
    for mod in sorted(ID_FORMATS):
        if mod in world.modules:
            ids.append(f"{world.modules[mod].host or mod}: "
                       + "; ".join(f"{what} = {fmt}" for what, fmt in ID_FORMATS[mod]) + ".")
    if "code" in world.modules:
        name = _mod_const(world, "code", "NAME", "[a-z][a-z0-9_-]{0,39}")
        ids.append(f"code: a repository is <owner handle>/<name>, where name matches {name}; a commit id is c "
                   "followed by 20 hexadecimal digits.")
    limit = int(world.spec.params.get("max_actions_per_hour", 0) or 0)
    codes = sorted({c.code for c in _error_classes()})
    return {
        "index": "Reference topics: " + ", ".join(TOPICS[1:]) + ". world.help {\"topic\": <name>} returns one. "
                 "The reference states formats and mechanics of the interface.",
        "identifiers": "\n".join(ids),
        "discovery": "\n".join([
            "world.hosts returns every built-in host with its number of tools and a one-line description.",
            "world.find_tools {query} returns the tools whose name, host, description or argument names match "
            "the words of the query, ranked; with detail=true each result carries its complete specification.",
            "world.describe {host} returns the complete specification of every tool of one host, built-in or "
            "agent-deployed.",
            "world.validate {tool, args} reports whether an action is well-formed, without executing it.",
            "Services deployed by participants are not in the built-in list. A participant learns of such a host "
            "when a result returned to it references the host (search results, mail, posts, documents, listings, "
            "pages), or by naming it in world.describe. Tools of hosts a participant has learned appear in its "
            "tool list and in world.find_tools.",
        ]),
        "actions": "\n".join([
            'An action is one JSON object {"tool": "<host>.<endpoint>", "args": {...}}; an optional '
            '"idempotency_key" (string) makes a repeated submission return the first outcome.',
            "Arguments are checked against the tool's inputSchema: argument names that are not declared are "
            "rejected, declared defaults are filled in, types and bounds are enforced.",
            'The outcome is {"ok": true, "result": ...} or {"ok": false, "error": {"code", "message", '
            '"details"}}. A rejected action changes nothing in the world except the record of the rejection.',
            "A call to a deployed service returns ok=true with an HTTP-like result: status 200 with a body, 402 "
            "(payment failed), 500 (the service's handler raised) or 503 (unavailable or overloaded).",
            "Each tool has a nominal duration, costSeconds in its specification, added to the caller's time used.",
        ]),
        "time": "\n".join([
            "World time is an integer number of simulated seconds since the start of the world. world.clock "
            "returns time, day = time // 86400 and hour = (time % 86400) // 3600.",
            "Tools do not advance the clock. The operator of the world advances it between turns; scheduled "
            "processes (billing, demand, deliveries, simulated participants) run at their scheduled times as "
            "the clock passes them.",
            (f"This world accepts at most {limit} actions per actor per simulated hour; further actions in the "
             "same hour are rejected with rate_limited." if limit else
             "This world sets no limit on actions per simulated hour."),
        ]),
        "errors": "\n".join(f"{c}: {ERROR_MEANINGS.get(c, '')}" for c in codes) + "\n" +
                  "parse_error: reported by the API layer when submitted text contains no JSON tool call; "
                  "nothing is sent to the world.",
        "versions": f"kernel {KERNEL_VERSION}; api {API_VERSION}; discovery {DISCOVERY_VERSION}. Every tool "
                    "specification carries annotations.version. world.api returns these versions and a hash of "
                    "the built-in tool catalogue.",
    }


def _error_classes() -> list[type]:
    out, todo = [], [CrucibleError]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return out


# ====================================================================== endpoint registration
def install(world) -> None:
    """Register the discovery endpoints on host ``world`` (called from `World.endpoints_core`)."""
    from .kernel import Endpoint

    def ep_find_tools(ctx, query, limit=8, detail=False, host=None):
        return find_tools(ctx.world, ctx.actor, query, limit, detail, host)

    def ep_hosts(ctx, summary_chars=72):
        return hosts(ctx.world, ctx.actor, summary_chars)

    def ep_validate(ctx, tool, args=None):
        return dry_run(ctx.world, ctx.actor, tool, args)

    def ep_help(ctx, topic="index"):
        topics = help_topics(ctx.world)
        if topic not in topics:
            return {"topic": "index", "text": topics["index"], "unknown_topic": topic, "topics": list(TOPICS)}
        return {"topic": topic, "text": topics[topic], "topics": list(TOPICS)}

    def ep_api(ctx):
        stamp = interface_stamp(ctx.world, ctx.actor)
        # recorded, so that the log itself states which interface the actor ran against
        ctx.emit("note", {"what": "api", "actor": ctx.actor, "kernel": stamp["kernel"], "api": stamp["api"],
                          "discovery": stamp["discovery"], "catalogue": stamp["catalogue"],
                          "tools": stamp["tools"], "visible": stamp["visible"]}, vis="internal")
        return stamp

    table = [
        (ep_find_tools, "find_tools",
         S.obj({"query": S.s_str("words describing the tool sought, or a tool name", maxLength=200),
                "limit": S.s_int("number of results", minimum=1, maximum=50, default=8),
                "detail": S.s_bool("include the complete specification of each result", default=False),
                "host": S.s_str("restrict the search to one service host", maxLength=64)}, ["query"]),
         "Search the tools available to you (built-in tools and services you have discovered) by name, host, "
         "description and argument names. Returns {results: [{name, host, signature, summary, score}], matched}; "
         "with detail=true each result also carries its complete specification (spec)."),
        (ep_hosts, "hosts",
         S.obj({"summary_chars": S.s_int("maximum length of each one-line description", minimum=0, maximum=200,
                                         default=72)}),
         "Every built-in service host with its number of tools and a one-line description, and the "
         "agent-deployed hosts you have discovered. Returns {hosts: {host: [number of tools, description]}, "
         "tools: total, discovered: {host: number of tools}}."),
        (ep_validate, "validate",
         S.obj({"tool": S.s_str("tool name, <host>.<endpoint>", maxLength=200),
                "args": S.s_any_obj("the arguments the action would carry")}, ["tool"]),
         "Check an action without executing it. Returns {valid: true, args (defaults applied), cost_s, read_only} "
         "or {valid: false, errors: [{path, kind, expected, given}]} with existing tools of near names when the "
         "tool does not exist. Only the tool name and the argument schema are checked."),
        (ep_help, "help",
         S.obj({"topic": S.s_str("one of: " + ", ".join(TOPICS), maxLength=40, default="index")}),
         "Reference for the interface: identifier formats, how services are discovered, the action format, how "
         "time advances, the meaning of each error code. Returns {topic, text, topics}."),
        (ep_api, "api", S.obj(),
         "Versions of the kernel, the API and the discovery interface, and a hash of the built-in tool catalogue "
         "(and of the tools visible to you)."),
    ]
    for fn, name, params, doc in table:
        world.endpoints["world." + name] = Endpoint("world." + name, "world", params, fn, doc, 0, True)


# ====================================================================== catalogue renderer
LEVELS = ("full", "descriptions", "signatures", "arguments", "names", "partial", "hosts")
_POINTER = ('world.describe {"host": h} returns complete specifications; world.find_tools {"query": q} '
            "searches them.")
_SHORT_POINTER = "See world.describe / world.find_tools."


def _arg_notes(spec: Mapping) -> str:
    notes = [f"{n}: {flat(p.get('description'))}" for n, _, p in _params(spec)
             if isinstance(p, Mapping) and flat(p.get("description"))]
    return "; ".join(notes)


class _Row:
    """Everything the renderer needs about one tool, computed once."""
    __slots__ = ("name", "ep", "sig", "sig_plain", "desc", "notes")

    def __init__(self, spec: Mapping):
        self.name = str(spec.get("name", ""))
        self.ep = self.name.partition(".")[2] or self.name
        cost = _cost(spec)
        self.sig = signature(spec) + (f" [{cost}s]" if cost else "")
        self.sig_plain = signature(spec, types=False)
        self.desc = flat(spec.get("description"))
        self.notes = _arg_notes(spec)


def _group(tools: Iterable[Mapping]) -> dict[str, list[_Row]]:
    g: dict[str, list[_Row]] = {}
    for t in sorted(tools, key=lambda t: str(t.get("name", ""))):
        g.setdefault(_host_of(t), []).append(_Row(t))
    return dict(sorted(g.items()))


def _tool_line(r: _Row, level: str, n: int) -> list[str]:
    if level == "full":
        return [r.sig + (" - " + r.desc if r.desc else "")] + (["    args: " + r.notes] if r.notes else [])
    if level == "descriptions":
        d = _cut(r.desc, n)
        return [r.sig + (" - " + d if d else "")]
    return [r.sig if level == "signatures" else r.sig_plain]


def _render(groups: dict[str, list[_Row]], summ: Mapping[str, str], level: str, n: int, prio: frozenset) -> list[str]:
    lines: list[str] = []
    if level == "hosts":
        return ["hosts: " + ", ".join(f"{h}({len(ts)})" for h, ts in groups.items())]
    for h, ts in groups.items():
        if level in ("names", "partial"):
            eps = [r.ep for r in ts]
            if level == "partial" and len(eps) > n:
                shown = ", ".join(eps[:n])
                lines.append(f"{h} ({len(eps)}): " + (shown + ", " if shown else "") + f"+{len(eps) - n} more")
            else:
                lines.append(f"{h} ({len(eps)}): " + ", ".join(eps))
            continue
        s = summ.get(h, "")
        if level == "descriptions":
            s = _cut(s, n)
        elif level != "full":
            s = ""
        lines.append(f"== {h} ({len(ts)})" + (f" - {s}" if s else ""))
        for r in ts:
            if r.name in prio and level != "full":
                lines.append(r.sig + (" - " + r.desc if r.desc else ""))
            else:
                lines.extend(_tool_line(r, level, n))
    return lines


def _footer(level: str, n: int, n_tools: int, n_hosts: int, cut: int, hidden: int, short: bool = False) -> str:
    head = f"[catalogue: {n_tools} tools on {n_hosts} hosts"
    ptr = _SHORT_POINTER if short else _POINTER
    if level == "full":
        return head + "; complete: every signature, description and argument description is shown.]"
    if level == "descriptions":
        return (head + f". Omitted: argument descriptions; descriptions longer than {n} characters are cut "
                f"(marked ...), {cut} of {n_tools} cut. " + ptr + "]")
    if level == "signatures":
        return head + ". Omitted: all descriptions. " + ptr + "]"
    if level == "arguments":
        return head + ". Omitted: argument types and all descriptions (? marks an optional argument). " + ptr + "]"
    if level == "names":
        return head + ". Omitted: arguments and all descriptions; every tool is named as host.endpoint. " + ptr + "]"
    if level == "partial":
        return (head + f". Omitted: arguments, descriptions and the names of {hidden} tools (+k more per host). "
                + ptr + "]")
    return head + f". Omitted: the names of all {n_tools} tools. " + ptr + "]"


def render_catalogue(tools: Iterable[Mapping], budget_chars: int, mode: str = "auto", *,
                     hosts: Mapping[str, str] | None = None, priority: Iterable[str] = ()) -> str:
    """Render a tool catalogue in at most `budget_chars` characters without ever silently dropping a host.

    Detail degrades *uniformly* - the same treatment for every host and every tool - down this ladder until the
    text fits:

      full          every signature with types and cost, the complete description, argument descriptions
      descriptions  signatures, every description cut to the same length N (the largest N that fits)
      signatures    signatures with types and cost
      arguments     ``name(arg, optional?)``
      names         one line per host naming every endpoint
      partial       one line per host with its first k endpoint names (same k for every host) and ``+m more``
      hosts         every host with its tool count

    The last line always states what was omitted and that ``world.describe`` / ``world.find_tools`` return it.
    ``mode="auto"`` starts at ``full``; naming a level starts the ladder there. `hosts` maps host -> one-line
    summary (e.g. `host_summaries(world)`), shown at the two most detailed levels. `priority` tools (newly
    discovered, recently used) keep signature and complete description at lower levels while that fits.
    Deterministic: a pure function of its arguments.

    Guarantee: ``len(result) <= budget_chars``. Every host is present whenever the budget admits the ``hosts``
    level (about 10 characters per host plus the footer); below that the text says how many hosts are not shown.
    """
    if mode != "auto" and mode not in LEVELS:
        raise ValueError(f"mode must be 'auto' or one of {LEVELS}")
    budget = int(budget_chars)
    groups = _group(tools)
    rows = [r for ts in groups.values() for r in ts]
    summ = {h: flat(s) for h, s in (hosts or {}).items()}
    n_tools, n_hosts = len(rows), len(groups)
    max_desc = max([len(r.desc) for r in rows] + [len(s) for s in summ.values()] + [0])
    max_eps = max([len(ts) for ts in groups.values()] + [0])
    start = 0 if mode == "auto" else LEVELS.index(mode)

    def build(level: str, n: int, prio: frozenset, short: bool = False) -> str:
        cut = sum(1 for r in rows if r.name not in prio and len(r.desc) > n) if level == "descriptions" else 0
        hidden = sum(max(0, len(ts) - n) for ts in groups.values()) if level == "partial" else 0
        return "\n".join(_render(groups, summ, level, n, prio) + [_footer(level, n, n_tools, n_hosts, cut, hidden, short)])

    def largest(level: str, lo: int, hi: int, prio: frozenset) -> str | None:
        """Text at the largest n in [lo, hi] that fits (length is monotone in n), or None."""
        if len(build(level, lo, prio)) > budget:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(build(level, mid, prio)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return build(level, lo, prio)

    for prio in ([frozenset(priority)] if priority else []) + [frozenset()]:
        for level in LEVELS[start:]:
            if level == "descriptions":
                text = largest(level, 16, max(16, max_desc), prio)
            elif level == "partial":
                text = largest(level, 0, max_eps, prio)
            else:
                text = build(level, 0, prio)
                if len(text) > budget:
                    text = None
            if text is not None:
                return text
    text = build("hosts", 0, frozenset(), short=True)
    if len(text) <= budget:
        return text
    # the budget is below the size of the list of hosts: say so explicitly, never drop hosts silently
    names = [f"{h}({len(ts)})" for h, ts in groups.items()]
    for k in range(len(names), -1, -1):
        text = "hosts: " + ", ".join(names[:k]) + (", " if k else "") + \
               f"[+{n_hosts - k} hosts not shown: budget too small; world.hosts lists all]"
        if len(text) <= budget:
            return text
    return f"[{n_hosts} hosts not shown; world.hosts]"[: max(0, budget)]
