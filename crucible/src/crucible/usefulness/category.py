"""Usefulness categories: the extension interface for new kinds of digital need.

A `NeedCategory` describes one category of need completely: public format and documentation, a seeded generator of
request instances with hidden truth, a hidden evaluator, reference solvers at several measured quality tiers, a
resource scale, a base price, search keywords, the customer population it serves and its externalities. An
environment developer adds a category (or a whole procedural family of them) without touching the kernel or any
existing module: `register_category` materialises it as an ordinary `economy.needs.Need`, after which consumers,
incumbent firms, bounties, goal feasibility checks and the hidden test battery work with it unchanged.

A `CategoryFamily` is a procedural generator. `instantiate(seed, params)` draws the *concrete hidden rule* from the
seed, so the same family yields a differently-ruled category in every world. Three relationships between a solver
and the rule are distinguished (`relation`):

    "specified"     the rule is fully documented; the difficulty is algorithmic
    "exemplified"   the rule is drawn per world and shown only through published worked examples
    "experiential"  the best behaviour depends on hidden parameters that no document states; they can only be
                    learned from the outcomes of serving requests

Categories are plain functions of (family, family version, seed, params). Their names are deterministic
(`gen.<family>.<8 hex of the instance hash>`), so a world rebuilt in a fresh process from its spec re-registers
identical needs. Registration is idempotent and safe when several worlds in one process use the same instance.

Nothing in this file is reachable by an agent except through the `use` kernel module, which exposes only the
public fields (`title`, `doc`, `keywords`, the published example pool).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.canonical import digest
from ..core.errors import ProtocolError
from ..economy import needs as _needs
from ..economy.needs import NEEDS, Need

CONTRACT_VERSION = "use/1"
RELATIONS = ("specified", "exemplified", "experiential")
SPLITS = ("train", "val", "test")
PREFIX = "gen."
MAX_OUTPUT_NODES = 20_000      # an output larger than this is rejected before the evaluator looks at it
MAX_OUTPUT_DEPTH = 8
MAX_OUTPUT_STR = 4_000
POOL_SIZE = {"exemplified": 16, "specified": 6, "experiential": 6}
POOL_INSTANCE_SIZE = 0.25      # published examples are small instances (modest payloads)


# ====================================================================== descriptions
@dataclass(frozen=True)
class Population:
    """The customers a category serves. `share` is the relative size of the population (the weight of the need in
    the consumer population), `rate_per_day` the range of requests one customer makes per day, `wtp_mult` the
    range of willingness to pay as a multiple of the base price."""
    share: float = 1.0
    rate_per_day: tuple[float, float] = (5.0, 30.0)
    wtp_mult: tuple[float, float] = (0.6, 3.0)
    doc: str = ""

    def to_dict(self) -> dict:
        return {"share": self.share, "rate_per_day": list(self.rate_per_day), "wtp_mult": list(self.wtp_mult),
                "doc": self.doc}


@dataclass(frozen=True)
class DerivedEdge:
    """Serving this category well creates need for categories of family `dst_family`. `weight` is the largest
    fraction of the downstream population it can activate, `scale` the verified volume (as a fraction of this
    category's nominal daily request volume) at which 63% of that is reached, `lag_days` the delay."""
    dst_family: str
    weight: float
    scale: float
    lag_days: int

    def to_dict(self) -> dict:
        return {"dst_family": self.dst_family, "weight": self.weight, "scale": self.scale, "lag_days": self.lag_days}


@dataclass(frozen=True)
class Externalities:
    """What serving a category does to the rest of the economy: derived-demand edges and resource intensity
    (compute drawn per request relative to a typical category; it sets the category's work scale)."""
    derived: tuple[DerivedEdge, ...] = ()
    resource_intensity: float = 1.0

    def to_dict(self) -> dict:
        return {"derived": [e.to_dict() for e in self.derived], "resource_intensity": self.resource_intensity}


# ====================================================================== a category
@dataclass
class NeedCategory:
    family: str
    seed: int
    params: dict
    kind: str                                   # the kind of work (data transformation, scheduling, ...)
    relation: str                               # specified | exemplified | experiential
    split: str                                  # train | val | test
    title: str
    doc: str                                    # public format and documentation
    keywords: tuple[str, ...]
    generate_raw: Callable[[np.random.Generator, float], tuple[dict, Any]]    # (public input, hidden truth)
    score_raw: Callable[[dict, Any, Any], float]                              # (input, hidden, output) -> [0, 1]
    tiers: list[tuple[str, str]]                # (label, python source defining solve(args, caller)), worst first
    work_scale: float                           # compute units per CPU millisecond
    base_price_cents: float
    population: Population = field(default_factory=Population)
    externalities: Externalities = field(default_factory=Externalities)
    truth: Callable[[dict, Any], dict] | None = None      # the correct output of an instance (exemplified only)
    example_extra: Callable[[dict, Any], dict] | None = None   # further public fields of a published example
    rule: dict = field(default_factory=dict)              # the drawn hidden rule: EVALUATOR ONLY, never exposed
    table_key: str | None = None                # input key holding a table of flat records (second stage of compose)
    render_table: Callable[[np.random.Generator, list], tuple[dict, Any]] | None = None   # table producers
    output_key: str = ""                        # the key every valid output must carry
    version: str = "1"
    parts: tuple = ()                           # composed categories: the stage categories
    uid: str = ""                               # instance hash (set in __post_init__)

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ProtocolError(f"category relation must be one of {RELATIONS}, got {self.relation!r}")
        if self.split not in SPLITS:
            raise ProtocolError(f"category split must be one of {SPLITS}, got {self.split!r}")
        if not 2 <= len(self.tiers) <= 4:
            raise ProtocolError(f"a category needs 2 to 4 reference tiers, got {len(self.tiers)}")
        if self.relation == "exemplified" and self.truth is None and self.example_extra is None:
            raise ProtocolError("an exemplified category must be able to state the correct output of an instance")
        if not self.uid:
            self.uid = instance_hash(self.family, self.version, self.seed, self.params)
        if len(self.name) > 40:
            raise ProtocolError(f"need name {self.name!r} exceeds the 40 characters tools accept")
        self._lock = threading.Lock()
        self._pool: list[dict] | None = None
        self._pool_keys: frozenset = frozenset()
        self._fingerprint: str | None = None

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        return f"{PREFIX}{self.family}.{self.uid[:8]}"

    def fingerprint(self) -> str:
        """Digest of what the category *does*: its documentation, reference solvers, prices and the first
        instances its generator publishes. A world records it at genesis; if the family's code later changes
        without a version bump, the mismatch is caught as an invariant violation instead of silently replaying
        a different rule."""
        if self._fingerprint is None:
            self._fingerprint = digest([self.uid, self.doc, [list(t) for t in self.tiers], self.work_scale,
                                        self.base_price_cents, [input_key(e["input"]) for e in self.pool()[:3]]], 12)
        return self._fingerprint

    def public(self) -> dict:
        """Everything an agent may know about the category without doing any work."""
        return {"category": self.name, "title": self.title, "kind": self.kind, "relation": self.relation,
                "doc": self.doc, "keywords": list(self.keywords), "reference_price_cents": self.base_price_cents,
                "examples": len(self.pool())}

    # ------------------------------------------------------------------ published examples
    def pool(self) -> list[dict]:
        """The published pool: seeded, the same for everyone, and disjoint from every instance used for scoring
        (`generate` never returns an input that is in the pool). Exemplified categories publish worked examples
        (input and correct output; a composed category also shows the result of its first stage, and only that
        when its second stage has no single correct output); the others publish sample requests only."""
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    rng = np.random.default_rng([int(self.uid[:12], 16), 0x5EED])
                    pool, keys = [], set()
                    for _ in range(POOL_SIZE[self.relation] * 4):
                        if len(pool) >= POOL_SIZE[self.relation]:
                            break
                        inp, hidden = self.generate_raw(rng, POOL_INSTANCE_SIZE)
                        k = input_key(inp)
                        if k in keys:
                            continue
                        keys.add(k)
                        ex = {"input": inp}
                        if self.relation == "exemplified" and self.truth is not None:
                            ex["output"] = self.truth(inp, hidden)
                        if self.example_extra is not None:
                            ex.update(self.example_extra(inp, hidden))
                        pool.append(ex)
                    self._pool_keys = frozenset(keys)
                    self._pool = pool
        return self._pool

    def examples(self, n: int = 3, offset: int = 0) -> list[dict]:
        p = self.pool()
        return p[max(0, int(offset)):max(0, int(offset)) + max(0, int(n))]

    # ------------------------------------------------------------------ the two functions a Need is made of
    def generate(self, rng: np.random.Generator, size: float) -> tuple[dict, Any]:
        """A fresh request instance. Never an instance of the published pool."""
        self.pool()
        for _ in range(4 * len(self._pool_keys) + 16):
            inp, hidden = self.generate_raw(rng, float(size))
            if input_key(inp) not in self._pool_keys:
                return inp, hidden
        raise ProtocolError(f"{self.name}: generator keeps reproducing published examples")

    def score(self, inp: dict, hidden: Any, out: Any) -> float:
        """Evaluate one output. Total: malformed, oversized, non-finite or adversarial outputs score 0; the
        evaluator never raises and never returns a value outside [0, 1]."""
        try:
            if not isinstance(out, dict) or out is inp:
                return 0.0
            if self.output_key and self.output_key not in out:
                return 0.0
            if not bounded(out, MAX_OUTPUT_NODES, MAX_OUTPUT_DEPTH, MAX_OUTPUT_STR):
                return 0.0
            s = float(self.score_raw(inp, hidden, out))
        except Exception:  # noqa: BLE001 - an evaluator must be total: whatever the output, the answer is 0
            return 0.0
        if not math.isfinite(s):
            return 0.0
        return min(1.0, max(0.0, s))


def instance_hash(family: str, version: str, seed: int, params: dict | None) -> str:
    return digest([CONTRACT_VERSION, family, version, int(seed), params or {}], size=16)


def input_key(inp: Any) -> str:
    """Identity of a request instance (used only to keep scoring instances out of the published pool)."""
    return hashlib.blake2b(json.dumps(inp, sort_keys=True, separators=(",", ":")).encode(), digest_size=10).hexdigest()


def bounded(x: Any, max_nodes: int, max_depth: int, max_str: int) -> bool:
    """True if a JSON-like value is small: at most `max_nodes` values, `max_depth` levels, strings up to
    `max_str` characters. Iterative, so a hostile deeply nested value cannot exhaust the stack."""
    n = 0
    stack = [(x, 0)]
    while stack:
        v, d = stack.pop()
        n += 1
        if n > max_nodes or d > max_depth:
            return False
        if isinstance(v, str):
            if len(v) > max_str:
                return False
        elif isinstance(v, dict):
            if len(v) > max_nodes:
                return False
            for k, w in v.items():
                if not isinstance(k, str) or len(k) > 200:
                    return False
                stack.append((w, d + 1))
        elif isinstance(v, (list, tuple)):
            if len(v) > max_nodes:
                return False
            stack.extend((w, d + 1) for w in v)
        elif not (v is None or isinstance(v, (bool, int, float))):
            return False
    return True


# ====================================================================== families (procedural generators)
@dataclass
class CategoryFamily:
    """`build(rng, params)` returns the fields of a `NeedCategory` (title, doc, keywords, generate_raw, score_raw,
    tiers, base_price_cents, and optionally truth, rule, table_key, render_table, output_key). The rng is seeded
    from (seed, family), so everything it draws - the hidden rule first of all - differs per seed."""
    name: str
    kind: str
    relation: str
    split: str
    doc: str
    build: Callable[[np.random.Generator, dict], dict]
    population: Population = field(default_factory=Population)
    demand_links: tuple[str, ...] = ()      # families whose need grows when this one is served well
    resource_intensity: float = 1.0
    work_scale: float = 30.0
    version: str = "1"

    def instantiate(self, seed: int, params: dict | None = None) -> NeedCategory:
        params = _plain(params or {})
        uid = instance_hash(self.name, self.version, seed, params)
        with _LOCK:
            cat = _INSTANCES.get(uid)
        if cat is not None:
            return cat
        h = int(digest(self.name, 4), 16)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), h]))
        f = dict(self.build(rng, dict(params)))
        xr = np.random.default_rng(np.random.SeedSequence([int(seed), h, 0xE7]))     # separate stream: adding
        edges = tuple(DerivedEdge(dst, round(float(xr.uniform(0.35, 0.9)), 3),      # links never moves a rule
                                  round(float(xr.uniform(0.4, 1.2)), 3), int(xr.choice([1, 2, 3, 5])))
                      for dst in self.demand_links)
        ws = float(f.pop("work_scale", self.work_scale * self.resource_intensity))
        cat = NeedCategory(family=self.name, seed=int(seed), params=params, kind=self.kind, relation=self.relation,
                           split=self.split, population=f.pop("population", self.population),
                           externalities=Externalities(edges, self.resource_intensity), work_scale=ws,
                           version=self.version, uid=uid, **f)
        with _LOCK:
            return _INSTANCES.setdefault(uid, cat)


FAMILIES: dict[str, CategoryFamily] = {}
_INSTANCES: dict[str, NeedCategory] = {}
GENERATED: dict[str, NeedCategory] = {}     # need name -> the category registered under it in this process
_LOCK = threading.RLock()


def category_family(name: str, *, kind: str, relation: str, split: str, doc: str, population: Population | None = None,
                    demand_links: tuple[str, ...] = (), resource_intensity: float = 1.0, work_scale: float = 30.0,
                    version: str = "1"):
    """Register a procedural family of categories. This decorator is the whole extension interface."""
    if relation not in RELATIONS or split not in SPLITS:
        raise ProtocolError(f"family {name}: bad relation or split")
    if not name.isidentifier() or not name.islower() or len(name) > 20:
        raise ProtocolError(f"family name {name!r} must be a short lowercase identifier")

    def deco(fn):
        FAMILIES[name] = CategoryFamily(name, kind, relation, split, doc, fn, population or Population(),
                                        tuple(demand_links), float(resource_intensity), float(work_scale), version)
        return fn
    return deco


def category_families(split: str | None = None, relation: str | None = None) -> list[str]:
    """Names of the registered families, optionally of one split and one relation to the rule."""
    load_families()
    return sorted(n for n, f in FAMILIES.items() if (split is None or f.split == split)
                  and (relation is None or f.relation == relation))


def instantiate(family: str, seed: int, params: dict | None = None) -> NeedCategory:
    load_families()
    if family not in FAMILIES:
        raise KeyError(f"unknown category family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family].instantiate(seed, params)


_LOADED = False


def load_families() -> None:
    """Import every built-in family module (each registers itself on import)."""
    global _LOADED
    if _LOADED:
        return
    import importlib
    _LOADED = True
    importlib.import_module(__package__ + ".families")


def _plain(x: Any) -> Any:
    """Params as plain JSON (they are hashed into the instance name and stored in world specs)."""
    return json.loads(json.dumps(x, sort_keys=True))


# ====================================================================== materialisation and registration
def materialize(cat: NeedCategory) -> Need:
    """The category as an `economy.needs.Need`: the object every existing mechanism already understands."""
    return Need(cat.name, cat.title, cat.doc, tuple(cat.keywords), float(cat.work_scale), cat.generate, cat.score,
                list(cat.tiers), float(cat.base_price_cents))


def register_category(cat: NeedCategory) -> Need:
    """Register a category's need under its deterministic name. Idempotent: registering the same instance again
    (another world in this process, a replay, a fork) returns the need already registered. A different instance
    under the same name (a hash collision, or a hand-built category with a forged uid) is refused."""
    with _LOCK:
        cur = GENERATED.get(cat.name)
        if cur is not None and cur.uid != cat.uid:
            raise ProtocolError(f"need name collision: {cat.name} is {cur.uid}, refusing {cat.uid}")
        if cur is None:
            if cat.name in NEEDS:
                raise ProtocolError(f"{cat.name} is already registered by something else")
            GENERATED[cat.name] = cur = cat
        nd = NEEDS.get(cat.name)
        if nd is None:                      # first registration, or the registry was reset: same functions again
            nd = NEEDS[cat.name] = materialize(cur)
        return nd


def category_of(need: str) -> NeedCategory | None:
    return GENERATED.get(need)


def unregister_generated(names: list[str] | None = None) -> list[str]:
    """Remove generated needs from the global registry (test hygiene: legacy code that iterates over every
    registered need must not see categories of worlds that no longer exist). Returns the names removed."""
    with _LOCK:
        gone = sorted(n for n in (names if names is not None else list(GENERATED)) if n in GENERATED)
        for n in gone:
            GENERATED.pop(n, None)
            NEEDS.pop(n, None)
        for k in [k for k in _needs._TQ if k[0] in gone]:
            del _needs._TQ[k]
        return gone


class scoped_registry:
    """Context manager: generated needs registered inside the block are unregistered when it exits."""

    def __enter__(self):
        self._before = set(GENERATED)
        return self

    def __exit__(self, *exc):
        unregister_generated(sorted(set(GENERATED) - self._before))
        return False


# ====================================================================== measurement
def measure_tiers(cat: NeedCategory, n: int = 8, seed: int = 0, size: float = 0.6) -> list[dict]:
    """Measured (not asserted) quality and cost of every reference tier on `n` seeded instances:
    [{"label", "quality", "cpu_ms", "lines"}] where cpu_ms is process CPU per request and lines the number of
    Python lines executed per request (the deterministic cost model charges 0.001 ms per line)."""
    from ..sandbox.bridge import _lines_off, _lines_on
    cases = [cat.generate(np.random.default_rng([int(seed), 911, k]), size) for k in range(int(n))]
    out = []
    for label, src in cat.tiers:
        fn = _needs.compile_solver(src)
        qs, lines = [], 0
        for inp, hidden in cases:
            arg = json.loads(json.dumps(inp))       # a service receives a decoded copy, never the evaluator's object
            _lines_on()
            try:
                res = fn(arg, None)
            except Exception:  # noqa: BLE001 - a reference solver that crashes scores 0 on that instance
                res = None
            finally:
                lines += _lines_off()
            qs.append(cat.score(inp, hidden, res))
        t0 = time.process_time()
        for inp, _ in cases:
            try:
                fn(json.loads(json.dumps(inp)), None)
            except Exception:  # noqa: BLE001
                pass
        cpu = (time.process_time() - t0) * 1000.0 / max(1, len(cases))
        out.append({"label": label, "quality": float(np.mean(qs)), "cpu_ms": cpu, "lines": lines / max(1, len(cases))})
    return out


# ====================================================================== composition
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
_REL_ORDER = {"specified": 0, "exemplified": 1, "experiential": 2}
_STAGE_GLUE = '''
def _load(src):
    ns = {"__name__": "stage"}
    exec(compile(src, "<stage>", "exec"), ns)
    return ns["solve"]


_first, _second = _load(_A), _load(_B)


def solve(args, caller):
    table = _first(args["stage1"], caller)["records"]
    req = dict(args["stage2"])
    req[_KEY] = table
    return _second(req, caller)
'''


def stage_source(first: str, second: str, key: str) -> str:
    """Source of a solver for a composed category from the sources of solvers for its two stages."""
    return f"_A = {first!r}\n_B = {second!r}\n_KEY = {key!r}\n" + _STAGE_GLUE


def compose(a: NeedCategory, b: NeedCategory) -> NeedCategory:
    """The cross-category need "a, then b on the result".

    `a` must be a table producer (its correct output is a table of records, and it can render any table as one of
    its own requests); `b` must take a table of flat records as part of its input. A composed request carries a's
    request (`stage1`) and b's request without its table (`stage2`): the table b needs is exactly the correct
    output of stage 1, and it appears nowhere else. The composed evaluator is b's evaluator applied to the final
    output against the true table, so a request is satisfied only by doing a's work and then b's on the result.
    An agent can serve it by building both halves, or by calling someone else's service for one of them.
    """
    if a.render_table is None:
        raise ProtocolError(f"{a.name} cannot be the first stage of a composition: it does not produce tables")
    if b.table_key is None:
        raise ProtocolError(f"{b.name} cannot be the second stage of a composition: it takes no table")
    uid = digest([CONTRACT_VERSION, "compose", a.uid, b.uid], size=16)
    with _LOCK:
        cat = _INSTANCES.get(uid)
    if cat is not None:
        return cat
    key = b.table_key

    def generate(rng, size):
        b_in, b_hid = b.generate(rng, size)
        a_in, _ = a.render_table(rng, b_in[key])
        return {"stage1": a_in, "stage2": {k: v for k, v in b_in.items() if k != key}}, {"b_in": b_in, "b_hid": b_hid}

    def score(inp, hidden, out):
        return b.score(hidden["b_in"], hidden["b_hid"], out)

    truth = None
    if b.truth is not None:
        def truth(inp, hidden):     # noqa: F811
            return b.truth(hidden["b_in"], hidden["b_hid"])

    def extra(inp, hidden):         # the worked part of a published example: what stage 1 must produce
        return {"stage1_output": {"records": hidden["b_in"][key]}}
    # reference tiers: a's best solver feeding each of b's tiers (a weak first stage hands the second one a
    # table it cannot even read, so the tiers of a composition differ by how well the second stage is done)
    tiers = [(f"{a.tiers[-1][0]}+{label}", stage_source(a.tiers[-1][1], src, key)) for label, src in b.tiers]
    relation = max((a.relation, b.relation), key=_REL_ORDER.get)
    split = max((a.split, b.split, "val"), key=_SPLIT_ORDER.get)     # a composition is never a training category
    doc = (f"A two-stage request.\n"
           f"Input: {{\"stage1\": <a request in the format of {a.name}>, \"stage2\": <a request in the format of "
           f"{b.name} WITHOUT its \"{key}\" field>}}.\n"
           f"The missing \"{key}\" table of stage 2 is exactly the \"records\" output of stage 1: do the work of "
           f"stage 1, put its records into stage 2 as \"{key}\", then do the work of stage 2.\n"
           f"Output: the output of stage 2. Score: stage 2's score for the final output, judged against the true "
           f"table (a wrong stage 1 result ruins it).\n\n--- stage 1 ({a.name}): {a.title}\n{a.doc}\n\n"
           f"--- stage 2 ({b.name}): {b.title}\n{b.doc}\n")
    keywords = tuple(dict.fromkeys(("pipeline",) + tuple(b.keywords[:3]) + tuple(a.keywords[:2])))
    intensity = a.externalities.resource_intensity + b.externalities.resource_intensity
    pop = Population(round(0.5 * min(a.population.share, b.population.share), 3), b.population.rate_per_day,
                     b.population.wtp_mult, f"customers of {b.title.lower()} whose data arrives in raw feeds")
    cat = NeedCategory(family="compose", seed=0, params={"a": a.uid, "b": b.uid}, kind=f"{a.kind} + {b.kind}",
                       relation=relation, split=split, title=f"{a.title}, then {b.title[0].lower()}{b.title[1:]}",
                       doc=doc, keywords=keywords,
                       generate_raw=generate, score_raw=score, tiers=tiers,
                       work_scale=round((a.work_scale + b.work_scale) / 2.0, 3),
                       base_price_cents=float(a.base_price_cents + b.base_price_cents), population=pop,
                       externalities=Externalities(b.externalities.derived, intensity),
                       truth=truth, example_extra=extra if a.relation == "exemplified" else None,
                       rule={"stages": [a.name, b.name], "table_key": key}, output_key=b.output_key,
                       version="1", parts=(a, b), uid=uid)
    with _LOCK:
        return _INSTANCES.setdefault(uid, cat)


def from_config(cfg: dict) -> list[NeedCategory]:
    """The categories of a `use` module config: {"categories": [{"family", "seed", "params"}], "compose": [[i, j]]}.
    Composed categories follow the base ones, in the order given."""
    cats = [instantiate(c["family"], int(c.get("seed", 0)), c.get("params")) for c in cfg.get("categories", [])]
    for pair in cfg.get("compose", []):
        i, j = int(pair[0]), int(pair[1])
        cats.append(compose(cats[i], cats[j]))
    return cats
