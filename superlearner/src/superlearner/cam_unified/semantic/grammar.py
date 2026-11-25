"""Schema-walking constrained argument generation (spec 9.2; architecture §11 decision 1).

An action's arguments are produced in two phases, both exactly re-scorable:

1. **Skeleton (CAM's typed heads).** `walk()` walks the tool's JSON schema (Crucible's strict subset, see
   `crucible.core.schema`) and emits the JSON structure deterministically. Every non-structural choice is a
   `Decision` answered by `ArgHeads`, conditioned on CAM's decision context [R; G; B], the tool token, the property's
   *schema embedding* (semantic embedding of its path, description and constraints) and a GRU walk state over the
   previous choices of the same action:
     presence of optional properties, the member of a type list (`"type": ["string", "null"]` or untyped values),
     array lengths, free-map key counts, booleans, enums (pointer over the semantically encoded enum values) and
     numbers/integers (NUM_LEVELS base-NUM_BASE digits of a grid index over [minimum, maximum]: an exact integer grid
     when the range has <= NUM_BASE**NUM_LEVELS values, otherwise an even grid in symlog space; digit masks keep the
     index inside the grid, so every value lies inside the schema's bounds).
   Decisions with a single admissible option are resolved without a head call and contribute log 1 = 0.
2. **Strings (Qwen + LoRA).** Free strings, patterned strings and free-map keys are left as `Hole`s. The template is
   serialised to JSON text: forced segments (structure, keys, head-chosen values) and holes. `HoleFiller` feeds the
   forced tokens and samples the hole tokens with the Qwen decoder (LoRA on) conditioned on the bridge prefix
   (inputs_embeds), the tool name/description/schema and the CAM-selected evidence, under the exact token masks of
   `constraints.py` (escape-aware JSON string body ending with the closing quote; regex DFA for patterns; length bounds;
   per-string budget `max_arg_tokens` and a per-action budget).

The result always satisfies `crucible.core.schema.validate` (checked after every generation; a violation would be a bug
and raises). The `record` (tool, evidence, head choices, hole token ids, prompt hash, temperature) is enough to
recompute the behaviour log-probability exactly (`log_prob` re-walks with teacher forcing and re-applies identical
masks).

Compute per action: the skeleton costs one ContextReader call per (non-trivial) decision, batched over the B rows in
lockstep rounds; Qwen costs one prefill of the prompt (+ prefix + stub + first forced segment) and one decoder step per
subsequent forced or sampled token, batched over the rows that have holes. Tools without string slots never touch
Qwen.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .constraints import (JsonStringTables, PatternMasks, UnsupportedSchema, compile_pattern)
from .layers import ContextReader, RMSNorm, masked_entropy, masked_log_softmax

MAX_STRING = 200_000                       # crucible.core.schema.MAX_STRING
JSON_TYPES = ["null", "boolean", "integer", "number", "string", "object", "array"]


class GrammarError(RuntimeError):
    """An internal inconsistency (a record that does not replay, an invalid result). Never swallowed."""


@dataclass
class GrammarOptions:
    num_base: int = 16
    num_levels: int = 3
    max_array_items: int = 8
    max_map_keys: int = 4
    untyped_depth: int = 2
    default_num_bound: float = 1e9
    float_digits: int = 6
    max_string_tokens: int = 1536          # cfg.max_arg_tokens
    max_key_tokens: int = 48
    max_action_tokens: int = 3072
    slot_chars: int = 320

    @property
    def kinds(self) -> list:
        return ["presence", "bool", "union", "count", "mapcount", "enum"] + [f"digit{l}" for l in range(self.num_levels)]

    def n_options(self, kind: str) -> int:
        return {"presence": 2, "bool": 2, "union": len(JSON_TYPES), "count": self.max_array_items + 1,
                "mapcount": self.max_map_keys + 1, "enum": 0}.get(kind, self.num_base)


def _validate(schema: dict, value) -> list:
    from crucible.core.schema import validate
    return validate(schema, value)


# ============================================================================ schema compilation
class NumberGrid:
    """Grid of admissible values of an integer/number slot, indexed by g in [0, n)."""

    def __init__(self, lo, hi, integer: bool, opts: GrammarOptions):
        self.lo, self.hi, self.integer = lo, hi, integer
        self.K, self.L = opts.num_base, opts.num_levels
        cap = self.K ** self.L
        self.digits = opts.float_digits
        if integer and hi - lo + 1 <= cap:
            self.exact, self.n = True, int(hi - lo + 1)
        else:
            self.exact, self.n = False, (1 if lo == hi else cap)
        self.ulo = math.copysign(math.log1p(abs(lo)), lo)
        self.uhi = math.copysign(math.log1p(abs(hi)), hi)

    def value(self, g: int):
        if self.exact:
            return int(self.lo + g)
        u = self.ulo if self.n == 1 else self.ulo + (self.uhi - self.ulo) * g / (self.n - 1)
        v = math.copysign(math.expm1(abs(u)), u)
        if self.integer:
            return int(min(max(round(v), self.lo), self.hi))
        v = float(f"{v:.{self.digits}g}")
        return float(min(max(v, self.lo), self.hi))

    def index(self, v) -> int:
        if self.exact:
            return int(min(max(round(v) - self.lo, 0), self.n - 1))
        if self.n == 1:
            return 0
        v = min(max(float(v), self.lo), self.hi)
        u = math.copysign(math.log1p(abs(v)), v)
        g = int(round((u - self.ulo) / (self.uhi - self.ulo) * (self.n - 1)))
        return int(min(max(g, 0), self.n - 1))

    def digit_mask(self, level: int, prefix: int) -> list:
        step = self.K ** (self.L - 1 - level)
        return [(prefix * self.K + d) * step < self.n for d in range(self.K)]

    def to_digits(self, g: int) -> list:
        out = []
        for l in range(self.L):
            step = self.K ** (self.L - 1 - l)
            out.append(g // step)
            g %= step
        return out


@dataclass(eq=False)
class Node:
    kind: str                       # object|array|string|enum|const|integer|number|boolean|null|union
    path: str
    desc: str = ""
    schema_text: str = ""
    props: list = field(default_factory=list)      # object: [(name, Node, required)]
    extra: Optional["Node"] = None                  # object: map-value node of additionalProperties
    item: Optional["Node"] = None                   # array
    lo: int = 0                                     # array: min items
    hi: int = 0                                     # array: max items (capped)
    values: list = field(default_factory=list)      # enum/const
    options: dict = field(default_factory=dict)     # union: JSON type name -> Node
    dfa: Any = None                                 # string with pattern
    min_len: int = 0
    max_len: int = MAX_STRING
    grid: Optional[NumberGrid] = None
    dropped: list = field(default_factory=list)     # optional properties that can never be generated (logged)

    @property
    def slot_text(self) -> str:
        return f"{self.path or '(arguments)'} [{self.kind}] {self.desc} {self.schema_text}"

    def may_hole(self) -> bool:
        if self.kind == "string":
            return True
        if self.kind == "object":
            return self.extra is not None or any(c.may_hole() for _, c, _ in self.props)
        if self.kind == "array":
            return self.hi > 0 and self.item.may_hole()
        if self.kind == "union":
            return any(o.may_hole() for o in self.options.values())
        return False


def _compact(s: dict) -> str:
    keep = {k: v for k, v in s.items() if k not in ("description", "properties", "items", "additionalProperties",
                                                      "required")}
    return json.dumps(keep, sort_keys=True, separators=(",", ":"), default=str)[:200]


def _untyped(path: str, depth: int, opts: GrammarOptions) -> Node:
    opt = {"null": Node("null", path), "boolean": Node("boolean", path)}
    b = opts.default_num_bound
    opt["integer"] = Node("integer", path, grid=NumberGrid(-int(b), int(b), True, opts))
    opt["number"] = Node("number", path, grid=NumberGrid(-float(b), float(b), False, opts))
    opt["string"] = Node("string", path)
    if depth > 0:
        opt["object"] = Node("object", path, extra=_untyped(path + ".*", depth - 1, opts))
        opt["array"] = Node("array", path, item=_untyped(path + "[]", depth - 1, opts), lo=0, hi=opts.max_array_items)
    return Node("union", path, desc="any JSON value", options=opt)


def compile_schema(s: Any, path: str, opts: GrammarOptions, depth: int) -> Node:
    """Compile a JSON schema (Crucible subset) into a Node tree. Raises UnsupportedSchema when no schema-valid value
    can be generated under the options (budget, caps, regex features)."""
    if s is True or s == {}:
        return _untyped(path, depth, opts)
    if not isinstance(s, dict):
        raise UnsupportedSchema(f"{path}: schema is not an object")
    desc = str(s.get("description", ""))
    st = _compact(s)
    t = s.get("type")
    if "enum" in s:
        rest = {k: v for k, v in s.items() if k != "enum"}
        vals = [v for v in s["enum"] if not _validate(rest, v)]
        if not vals:
            raise UnsupportedSchema(f"{path}: no enum value satisfies the schema")
        return Node("const" if len(vals) == 1 else "enum", path, desc, st, values=vals)
    if isinstance(t, list):
        opt = {}
        for ti in t:
            if ti in opt:
                continue
            try:
                opt[ti] = compile_schema({**s, "type": ti}, path, opts, depth)
            except UnsupportedSchema:
                continue
        if not opt:
            raise UnsupportedSchema(f"{path}: no generable member of type list {t}")
        if len(opt) == 1:
            return next(iter(opt.values()))
        return Node("union", path, desc, st, options=opt)
    if t is None:
        if any(k in s for k in ("properties", "additionalProperties", "required")):
            t = "object"
        elif "items" in s:
            t = "array"
        else:
            u = _untyped(path, depth, opts)
            u.desc, u.schema_text = desc, st
            return u
    if t == "object":
        props = s.get("properties") or {}
        req = list(s.get("required") or [])
        addl = s.get("additionalProperties", False)
        extra = None
        if addl is True:
            extra = _untyped(path + ".*", max(depth - 1, 0), opts)
        elif isinstance(addl, dict):
            extra = compile_schema(addl, path + ".*", opts, depth)
        node = Node("object", path, desc, st, extra=extra)
        names = list(props) + [r for r in req if r not in props]
        for name in names:
            sub = props.get(name)
            if sub is None:
                if extra is None:
                    raise UnsupportedSchema(f"{path}.{name}: required but not allowed")
                sub = addl if isinstance(addl, dict) else True
            required = name in req
            try:
                child = compile_schema(sub, f"{path}.{name}", opts, depth)
            except UnsupportedSchema:
                if required:
                    raise
                node.dropped.append(name)
                continue
            node.props.append((name, child, required))
        return node
    if t == "array":
        lo = int(s.get("minItems", 0))
        hi = min(int(s.get("maxItems", 10_000)), opts.max_array_items)
        if lo > hi:
            raise UnsupportedSchema(f"{path}: minItems {lo} exceeds the generator cap {hi}")
        items = s.get("items")
        try:
            item = compile_schema(items, path + "[]", opts, depth) if items is not None else \
                _untyped(path + "[]", max(depth - 1, 0), opts)
        except UnsupportedSchema:
            if lo > 0:
                raise
            item, hi = Node("null", path + "[]"), 0
        return Node("array", path, desc, st, item=item, lo=lo, hi=hi)
    if t == "string":
        mn = int(s.get("minLength", 0))
        mx = int(s.get("maxLength", MAX_STRING))
        node = Node("string", path, desc, st, min_len=mn, max_len=mx)
        if "pattern" in s:
            cap = min(mx, opts.max_string_tokens)
            node.dfa = compile_pattern(s["pattern"], cap)
            if not node.dfa.feasible_from(0, mn, min(mx, cap - 1)):
                raise UnsupportedSchema(f"{path}: pattern cannot be met within the length/token limits")
        elif not JsonStringTables.feasible(mn, mx, opts.max_string_tokens):
            raise UnsupportedSchema(f"{path}: length bounds infeasible")
        return node
    if t in ("integer", "number"):
        b = opts.default_num_bound
        integer = t == "integer"
        lo = s.get("minimum", -b)
        hi = s.get("maximum", b)
        if integer:
            lo, hi = int(math.ceil(lo)), int(math.floor(hi))
        else:
            lo, hi = float(lo), float(hi)
        if lo > hi:
            raise UnsupportedSchema(f"{path}: empty numeric range")
        return Node(t, path, desc, st, grid=NumberGrid(lo, hi, integer, opts))
    if t == "boolean":
        return Node("boolean", path, desc, st)
    if t == "null":
        return Node("null", path, desc, st)
    raise UnsupportedSchema(f"{path}: unknown type {t!r}")


class ToolGrammar:
    """Compiled argument grammar of one tool (cached by ToolSchema.key)."""

    def __init__(self, tool, opts: GrammarOptions):
        self.tool = tool
        self.key = tool.key
        self.error: Optional[str] = None
        self.root: Optional[Node] = None
        params = tool.parameters or {"type": "object", "properties": {}}
        try:
            root = compile_schema(params, "", opts, opts.untyped_depth)
            if root.kind != "object":
                raise UnsupportedSchema("top-level arguments schema must be an object")
            self.root = root
        except UnsupportedSchema as e:
            self.error = str(e)

    @property
    def generable(self) -> bool:
        return self.root is not None

    @property
    def may_hole(self) -> bool:
        return self.root is not None and self.root.may_hole()


# ============================================================================ walking
@dataclass(eq=False)
class Decision:
    kind: str
    node: Node
    mask: list
    values: Optional[list] = None


@dataclass(eq=False)
class Hole:
    kind: str                       # 'str' | 'pattern' | 'key'
    node: Node
    min_len: int = 0
    max_len: int = MAX_STRING
    dfa: Any = None
    tokens: Optional[list] = None
    value: Optional[str] = None


@dataclass(eq=False)
class TObject:
    node: Node
    entries: list                   # [(str | Hole, value)]


def _ask(dec: Decision):
    allowed = [i for i, m in enumerate(dec.mask) if m]
    if not allowed:
        raise GrammarError(f"{dec.node.path}: decision {dec.kind} has no admissible option")
    if len(allowed) == 1:
        return allowed[0]
    c = yield dec
    if not (0 <= c < len(dec.mask)) or not dec.mask[c]:
        raise GrammarError(f"{dec.node.path}: inadmissible choice {c} for {dec.kind}")
    return c


def walk(node: Node, opts: GrammarOptions):
    """Generator: yields Decisions, receives chosen indices, returns the template value (with Holes)."""
    k = node.kind
    if k == "object":
        entries = []
        for name, child, req in node.props:
            if not req:
                if (yield from _ask(Decision("presence", child, [True, True]))) == 0:
                    continue
            entries.append((name, (yield from walk(child, opts))))
        if node.extra is not None:
            n = yield from _ask(Decision("mapcount", node, [True] * (opts.max_map_keys + 1)))
            for _ in range(n):
                key = Hole("key", node, 0, MAX_STRING)
                entries.append((key, (yield from walk(node.extra, opts))))
        return TObject(node, entries)
    if k == "array":
        n = yield from _ask(Decision("count", node, [node.lo <= i <= node.hi for i in range(opts.max_array_items + 1)]))
        out = []
        for _ in range(n):
            out.append((yield from walk(node.item, opts)))
        return out
    if k == "string":
        return Hole("pattern" if node.dfa is not None else "str", node, node.min_len, node.max_len, node.dfa)
    if k == "enum":
        return node.values[(yield from _ask(Decision("enum", node, [True] * len(node.values), node.values)))]
    if k == "const":
        return node.values[0]
    if k in ("integer", "number"):
        g = 0
        for l in range(opts.num_levels):
            d = yield from _ask(Decision(f"digit{l}", node, node.grid.digit_mask(l, g)))
            g = g * opts.num_base + d
        return node.grid.value(g)
    if k == "boolean":
        return bool((yield from _ask(Decision("bool", node, [True, True]))))
    if k == "null":
        return None
    if k == "union":
        c = yield from _ask(Decision("union", node, [t in node.options for t in JSON_TYPES]))
        return (yield from walk(node.options[JSON_TYPES[c]], opts))
    raise GrammarError(f"unknown node kind {k}")


def encode_value(node: Node, value, opts: GrammarOptions, choices: list, holes: list, flags: dict) -> None:
    """Inverse of `walk` for behaviour cloning: append the choices (same auto-resolution rule) and hole strings that
    make `walk` reproduce `value` (numbers: nearest grid point, flagged in flags['approx'])."""
    def put(dec: Decision, c: int):
        allowed = [i for i, m in enumerate(dec.mask) if m]
        if c not in allowed:
            raise GrammarError(f"{node.path}: value not representable ({dec.kind}={c})")
        if len(allowed) > 1:
            choices.append(c)

    k = node.kind
    if k == "object":
        if not isinstance(value, dict):
            raise GrammarError(f"{node.path}: expected object")
        declared = {n for n, _, _ in node.props}
        for name, child, req in node.props:
            present = name in value
            if not req:
                put(Decision("presence", child, [True, True]), int(present))
            elif not present:
                raise GrammarError(f"{node.path}.{name}: required property missing")
            if present:
                encode_value(child, value[name], opts, choices, holes, flags)
        extra_keys = [kk for kk in value if kk not in declared]
        if node.extra is not None:
            if len(extra_keys) > opts.max_map_keys:
                raise GrammarError(f"{node.path}: {len(extra_keys)} map keys exceed the cap {opts.max_map_keys}")
            put(Decision("mapcount", node, [True] * (opts.max_map_keys + 1)), len(extra_keys))
            for kk in extra_keys:
                holes.append(("key", str(kk)))
                encode_value(node.extra, value[kk], opts, choices, holes, flags)
        elif extra_keys:
            raise GrammarError(f"{node.path}: unexpected properties {extra_keys}")
        return
    if k == "array":
        if not isinstance(value, list) or not (node.lo <= len(value) <= node.hi):
            raise GrammarError(f"{node.path}: array length outside [{node.lo}, {node.hi}]")
        put(Decision("count", node, [node.lo <= i <= node.hi for i in range(opts.max_array_items + 1)]), len(value))
        for v in value:
            encode_value(node.item, v, opts, choices, holes, flags)
        return
    if k == "string":
        if not isinstance(value, str):
            raise GrammarError(f"{node.path}: expected string")
        holes.append(("pattern" if node.dfa is not None else "str", value))
        return
    if k == "enum":
        if value not in node.values:
            raise GrammarError(f"{node.path}: {value!r} not an enum value")
        put(Decision("enum", node, [True] * len(node.values)), node.values.index(value))
        return
    if k == "const":
        if value != node.values[0]:
            raise GrammarError(f"{node.path}: expected constant {node.values[0]!r}")
        return
    if k in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GrammarError(f"{node.path}: expected a number")
        g = node.grid.index(value)
        if node.grid.value(g) != value:
            flags["approx"] = flags.get("approx", 0) + 1
        pre = 0
        for l, d in enumerate(node.grid.to_digits(g)):
            put(Decision(f"digit{l}", node, node.grid.digit_mask(l, pre)), d)
            pre = pre * opts.num_base + d
        return
    if k == "boolean":
        if not isinstance(value, bool):
            raise GrammarError(f"{node.path}: expected boolean")
        put(Decision("bool", node, [True, True]), int(value))
        return
    if k == "null":
        if value is not None:
            raise GrammarError(f"{node.path}: expected null")
        return
    if k == "union":
        for ti, t in enumerate(JSON_TYPES):
            sub = node.options.get(t)
            if sub is None:
                continue
            if t == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                continue
            if not _validate({"type": t}, value):
                put(Decision("union", node, [x in node.options for x in JSON_TYPES]), ti)
                encode_value(sub, value, opts, choices, holes, flags)
                return
        raise GrammarError(f"{node.path}: value matches no member of the type list")
    raise GrammarError(f"unknown node kind {k}")


def holes_of(template) -> list:
    """Holes in serialisation order."""
    out: list = []

    def rec(v):
        if isinstance(v, Hole):
            out.append(v)
        elif isinstance(v, TObject):
            for k, x in v.entries:
                if isinstance(k, Hole):
                    out.append(k)
                rec(x)
        elif isinstance(v, list):
            for x in v:
                rec(x)
    rec(template)
    return out


def serialize(template) -> list:
    """Template -> segments [str | Hole]; each Hole is preceded by a forced segment ending with its opening quote and
    includes its own closing quote."""
    segs: list = []
    buf: list = []

    def flush():
        if buf:
            segs.append("".join(buf))
            buf.clear()

    def rec(v):
        if isinstance(v, Hole):
            buf.append('"')
            flush()
            segs.append(v)
        elif isinstance(v, TObject):
            buf.append("{")
            for i, (k, x) in enumerate(v.entries):
                if i:
                    buf.append(", ")
                if isinstance(k, Hole):
                    rec(k)
                else:
                    buf.append(json.dumps(k, ensure_ascii=False))
                buf.append(": ")
                rec(x)
            buf.append("}")
        elif isinstance(v, list):
            buf.append("[")
            for i, x in enumerate(v):
                if i:
                    buf.append(", ")
                rec(x)
            buf.append("]")
        else:
            buf.append(json.dumps(v, ensure_ascii=False))
    rec(template)
    flush()
    return segs


def materialize(template, stats: Optional[dict] = None):
    """Template with filled holes -> plain JSON value. Free-map keys that repeat a declared property name or an
    earlier key are dropped (first occurrence wins; counted in stats['dropped_keys'])."""
    if isinstance(template, Hole):
        if template.value is None:
            raise GrammarError("unfilled hole")
        return template.value
    if isinstance(template, TObject):
        declared = {n for n, _, _ in template.node.props}
        out: dict = {}
        for k, x in template.entries:
            if isinstance(k, Hole):
                key = materialize(k)
                if key in declared or key in out:
                    if stats is not None:
                        stats["dropped_keys"] = stats.get("dropped_keys", 0) + 1
                    continue
            else:
                key = k
            out[key] = materialize(x, stats)
        return out
    if isinstance(template, list):
        return [materialize(x, stats) for x in template]
    return template


def drive_walks(roots: list, opts: GrammarOptions, decide: Callable[[list, list], list]) -> list:
    """Run one walk per row in lockstep; `decide(rows, decisions) -> choices` is called once per round with the
    pending decision of every unfinished row. Returns the templates (None for rows whose root is None)."""
    gens, pending, result = {}, {}, [None] * len(roots)
    for r, root in enumerate(roots):
        if root is None:
            continue
        g = walk(root, opts)
        gens[r] = g
        try:
            pending[r] = next(g)
        except StopIteration as e:
            result[r] = e.value
    while pending:
        rows = sorted(pending)
        choices = decide(rows, [pending[r] for r in rows])
        for r, c in zip(rows, choices):
            try:
                pending[r] = gens[r].send(int(c))
            except StopIteration as e:
                result[r] = e.value
                del pending[r]
    return result


# ============================================================================ typed heads
class ArgHeads(nn.Module):
    """Typed decision heads. A decision's query is q = h_walk + E_slot + E_kind + tool_token; it reads the decision
    context through a ContextReader; categorical kinds use per-kind linear outputs, enums a pointer onto the encoded
    enum values. After each choice the GRU walk state is updated with (E_slot + E_choice)."""

    def __init__(self, d: int, n_heads: int, sem_dim: int, opts: GrammarOptions):
        super().__init__()
        self.opts = opts
        self.kinds = opts.kinds
        self.kind_index = {k: i for i, k in enumerate(self.kinds)}
        self.slot_proj = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.kind_emb = nn.Embedding(len(self.kinds), d)
        self.reader = ContextReader(d, n_heads, 1)
        self.out = nn.ModuleDict({k: nn.Linear(d, opts.n_options(k)) for k in self.kinds if k != "enum"})
        self.choice_emb = nn.ModuleDict({k: nn.Embedding(opts.n_options(k), d) for k in self.kinds if k != "enum"})
        self.enum_q = nn.Linear(d, d)
        self.enum_k = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.gru = nn.GRUCell(d, d)
        self.h0 = nn.Linear(2 * d, d)
        self.d = d

    def init_state(self, tool_tok: torch.Tensor, ctx_summary: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.h0(torch.cat([tool_tok, ctx_summary], -1)))

    def decide(self, enc, decisions: list, h: torch.Tensor, tool_tok: torch.Tensor, ctx_x: torch.Tensor,
               ctx_mask: torch.Tensor, temperature: float, greedy: bool, forced: Optional[list],
               generator: Optional[torch.Generator]):
        """Evaluate n decisions (one per row given). h, tool_tok [n, D]; ctx_x [n, N, D]; ctx_mask [n, N].
        Returns (choices list[int], logp [n], entropy [n], h_new [n, D])."""
        n = len(decisions)
        dev = h.device
        slot = self.slot_proj(enc.embed([d.node.slot_text[: self.opts.slot_chars] for d in decisions]).to(dev))
        kind = torch.tensor([self.kind_index[d.kind] for d in decisions], device=dev)
        q = h + slot + self.kind_emb(kind) + tool_tok
        f = self.reader(q.unsqueeze(1), ctx_x, ctx_mask).squeeze(1)                   # [n, D]
        choices, logps, ents, cembs = [], [], [], []
        for i, d in enumerate(decisions):
            if d.kind == "enum":
                vk = self.enum_k(enc.embed([json.dumps(v, ensure_ascii=False) for v in d.values]).to(dev))  # [m, D]
                logits = (self.enum_q(f[i]) @ vk.t()) / math.sqrt(self.d)
            else:
                logits = self.out[d.kind](f[i])
            mask = torch.tensor(d.mask, dtype=torch.bool, device=dev)
            tau = temperature[i] if isinstance(temperature, torch.Tensor) else temperature
            lp = masked_log_softmax(logits, mask, tau)
            if forced is not None:
                c = int(forced[i])
            elif greedy:
                c = int(lp.argmax())
            else:
                c = int(torch.multinomial(lp.exp(), 1, generator=generator))
            if not d.mask[c]:
                raise GrammarError(f"{d.node.path}: choice {c} not admissible for {d.kind}")
            choices.append(c)
            logps.append(lp[c])
            ents.append(masked_entropy(lp, mask))
            cembs.append(vk[c] if d.kind == "enum" else self.choice_emb[d.kind].weight[c])
        cemb = torch.stack(cembs).to(slot.dtype)
        h_new = self.gru((slot + cemb).float(), h.float()).to(h.dtype)
        return choices, torch.stack(logps), torch.stack(ents), h_new


# ============================================================================ Qwen hole filling
@dataclass(eq=False)
class Program:
    """One row's decoder program: forced token segments interleaved with holes (forced[i] precedes holes[i])."""
    forced: list                   # list[list[int]] (len = len(holes) + 1; the last one is never fed)
    holes: list                    # list[Hole]
    reserve: list                  # per-hole minimal token need
    cap: list                      # per-hole base cap


def hole_reserve(h: Hole) -> int:
    if h.kind == "pattern":
        lens = h.dfa.reach[0]
        for l in range(max(h.min_len, 0), h.dfa.cap + 1):
            if (lens >> l) & 1:
                return l + 1
        raise GrammarError("pattern has no feasible completion")
    return h.min_len + 1


class HoleFiller:
    """Samples / rescores the string holes of a batch of programs with the LoRA-enabled Qwen decoder."""

    def __init__(self, enc, tables: JsonStringTables, opts: GrammarOptions, max_prompt_tokens: int = 3072):
        self.enc = enc
        self.tables = tables
        self.opts = opts
        self.max_prompt_tokens = max_prompt_tokens
        self._pmask: dict = {}
        tok = enc.tok
        self.stub_ids = tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]

    # ---------------------------------------------------------------- prompts and programs
    def context_ids(self, prompt: str) -> list:
        ids = self.enc.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_prompt_tokens:
            h = self.max_prompt_tokens // 2
            ids = ids[:h] + ids[-(self.max_prompt_tokens - h):]
        head = self.enc.tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        tail = self.enc.tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        return head + ids + tail

    @staticmethod
    def prompt_hash(ctx_ids: list) -> str:
        return hashlib.sha256(json.dumps(ctx_ids).encode()).hexdigest()[:24]

    def program(self, template) -> Program:
        segs = serialize(template)
        forced, holes, buf = [], [], ""
        for s in segs:
            if isinstance(s, Hole):
                forced.append(self.enc.tok(buf, add_special_tokens=False)["input_ids"] if buf else [])
                holes.append(s)
                buf = ""
            else:
                buf += s
        forced.append(self.enc.tok(buf, add_special_tokens=False)["input_ids"] if buf else [])
        reserve = [hole_reserve(h) for h in holes]
        if sum(reserve) > self.opts.max_action_tokens:
            raise GrammarError(f"holes need {sum(reserve)} tokens > action budget {self.opts.max_action_tokens}")
        cap = []
        for h in holes:
            # a token adds >= 1 decoded character except a lone backslash (which the next token completes), so a
            # string of at most max_len characters never needs more than 2*max_len + 1 tokens (incl. the quote)
            if h.kind == "pattern":
                cap.append(min(self.opts.max_string_tokens, h.max_len + 1))
            elif h.kind == "key":
                cap.append(min(self.opts.max_key_tokens, 2 * h.max_len + 1))
            else:
                cap.append(min(self.opts.max_string_tokens, 2 * h.max_len + 1))
        return Program(forced, holes, reserve, cap)

    # ---------------------------------------------------------------- masks
    def _pattern_masks(self, h: Hole) -> PatternMasks:
        pm = self._pmask.get(id(h.dfa))
        if pm is None:
            pm = PatternMasks(h.dfa, self.tables)
            self._pmask[id(h.dfa)] = pm
        return pm

    def masks(self, specs: list) -> torch.Tensor:
        """specs: list of (hole, state, count, remaining) -> allowed-token masks [P, V]."""
        T = self.tables
        P = len(specs)
        out = torch.zeros(P, T.V, dtype=torch.bool, device=T.valid.device)
        srows = [i for i, sp in enumerate(specs) if sp[0].kind != "pattern"]
        for i, sp in enumerate(specs):
            if sp[0].kind == "pattern":
                out[i] = self._pattern_masks(sp[0]).mask(sp[1], sp[2], sp[3], sp[0].min_len, sp[0].max_len)
        for c0 in range(0, len(srows), 64):
            rows = srows[c0: c0 + 64]
            dev = T.valid.device
            S = torch.tensor([specs[i][1] for i in rows], device=dev)
            cnt = torch.tensor([specs[i][2] for i in rows], device=dev).unsqueeze(1)
            rem = torch.tensor([specs[i][3] for i in rows], device=dev).unsqueeze(1)
            mn = torch.tensor([specs[i][0].min_len for i in rows], device=dev).unsqueeze(1)
            mx = torch.tensor([specs[i][0].max_len for i in rows], device=dev).unsqueeze(1)
            v, cl, e, n = T.valid[S], T.closes[S], T.end[S], T.nch[S]
            c1 = cnt + n
            close_ok = v & cl & (c1 >= mn) & (c1 <= mx)
            need0 = (mn - c1).clamp(min=0)
            need1 = (mn - c1 - 1).clamp(min=0)
            r1 = rem - 1
            cont0 = (e == 0) & (c1 <= mx) & (need0 + 1 <= r1)
            cont1 = (e == 1) & (c1 + 1 <= mx) & (need1 + 2 <= r1)
            out[torch.tensor(rows, device=dev)] = close_ok | (v & ~cl & (cont0 | cont1))
        return out

    def _advance(self, h: Hole, state: int, count: int, tok: int):
        if h.kind == "pattern":
            return self._pattern_masks(h).advance(state, count, tok)
        return self.tables.advance(state, count, tok)

    def _hole_value(self, h: Hole) -> str:
        text = "".join(self.tables.texts[t] for t in h.tokens)
        if h.kind == "pattern":
            if not text.endswith('"'):
                raise GrammarError("pattern hole not closed")
            return text[:-1]
        return json.loads('"' + text)

    # ---------------------------------------------------------------- assembly
    def _assemble(self, prefix: torch.Tensor, ctx: list, tails: list):
        """Left-padded [ctx][prefix][tail] rows -> (inputs_embeds [R, L, H], attention [R, L], positions [R, L],
        start index of each tail)."""
        enc = self.enc
        R = len(ctx)
        P = prefix.shape[1]
        lens = [len(c) + P + len(t) for c, t in zip(ctx, tails)]
        L = max(lens)
        dev = enc.device
        ids = torch.full((R, L), enc.pad_id, dtype=torch.long, device=dev)
        is_pre = torch.zeros(R, L, dtype=torch.bool, device=dev)
        att = torch.zeros(R, L, dtype=torch.long, device=dev)
        starts = []
        for i in range(R):
            off = L - lens[i]
            row = ctx[i] + [enc.pad_id] * P + tails[i]
            ids[i, off:] = torch.tensor(row, device=dev)
            att[i, off:] = 1
            is_pre[i, off + len(ctx[i]): off + len(ctx[i]) + P] = True
            starts.append(off + len(ctx[i]) + P)
        emb = enc.embed_ids(ids)
        pre = prefix.to(emb.dtype)
        emb = emb.masked_scatter(is_pre.unsqueeze(-1).expand_as(emb), pre.reshape(-1, pre.shape[-1]))
        pos = (att.cumsum(1) - 1).clamp(min=0)
        return emb, att, pos, starts

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, programs: list, prefix: torch.Tensor, ctx: list, temperature: float, greedy: bool,
               generator: Optional[torch.Generator] = None):
        """Fill the holes of every program in place. prefix [R, P, H]. Returns (logp [R], entropy [R], n_sampled [R],
        n_steps)."""
        enc = self.enc
        R = len(programs)
        dev = enc.device
        tails = [self.stub_ids + p.forced[0] for p in programs]
        emb, att, pos, _ = self._assemble(prefix, ctx, tails)
        cache = enc.new_cache()
        logits, cache = enc.step(emb, att, pos, cache)
        cur = pos[:, -1].clone()
        logp = torch.zeros(R, device=dev)
        ent = torch.zeros(R, device=dev)
        n_s = [0] * R
        # per-row cursor: [hole index, mode ('hole'|'forced'), forced position, state, count, used, budget_used]
        st = []
        for p in programs:
            for h in p.holes:
                h.tokens = []
            st.append({"hi": 0, "mode": "hole", "fp": 0, "s": 0, "c": 0, "used": 0, "spent": 0, "done": not p.holes})
        steps = 0
        while not all(s["done"] for s in st):
            specs, rows = [], []
            for r, (p, s) in enumerate(zip(programs, st)):
                if not s["done"] and s["mode"] == "hole":
                    h = p.holes[s["hi"]]
                    later = sum(p.reserve[s["hi"] + 1:])
                    cap = min(p.cap[s["hi"]], self.opts.max_action_tokens - s["spent"] - later)
                    specs.append((h, s["s"], s["c"], cap - s["used"]))
                    rows.append(r)
            feed = [enc.pad_id] * R
            live = [0] * R
            if rows:
                m = self.masks(specs)
                if not bool(m.any(1).all()):
                    raise GrammarError("empty token mask during hole sampling")
                lp = masked_log_softmax(logits[rows], m, temperature)
                if greedy:
                    tok = lp.argmax(-1)
                else:
                    tok = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
                chosen = lp.gather(1, tok.unsqueeze(1)).squeeze(1)
                e = masked_entropy(lp, m)
                rr = torch.tensor(rows, device=dev)
                logp.index_add_(0, rr, chosen)
                ent.index_add_(0, rr, e)
                tok_l = tok.tolist()
                for j, r in enumerate(rows):
                    t = tok_l[j]
                    feed[r] = t
                    live[r] = 1
                    p, s = programs[r], st[r]
                    h = p.holes[s["hi"]]
                    h.tokens.append(t)
                    n_s[r] += 1
                    s["s"], s["c"], closed = self._advance(h, s["s"], s["c"], t)
                    s["used"] += 1
                    if closed:
                        h.value = self._hole_value(h)
                        s["spent"] += s["used"]
                        s["hi"] += 1
                        s.update(s=0, c=0, used=0, mode="forced", fp=0)
                        if s["hi"] >= len(p.holes):
                            s["done"] = True
                        elif not p.forced[s["hi"]]:
                            s["mode"] = "hole"
            for r, (p, s) in enumerate(zip(programs, st)):
                if s["done"] or s["mode"] != "forced" or live[r]:
                    continue
                seg = p.forced[s["hi"]]
                feed[r] = seg[s["fp"]]
                live[r] = 1
                s["fp"] += 1
                if s["fp"] >= len(seg):
                    s["mode"] = "hole"
            if all(s["done"] for s in st):
                break
            att = torch.cat([att, torch.tensor(live, dtype=att.dtype, device=dev).unsqueeze(1)], 1)
            cur = cur + 1
            feed_t = torch.tensor(feed, dtype=torch.long, device=dev)
            logits, cache = enc.step(enc.embed_ids(feed_t.unsqueeze(1)), att, cur.unsqueeze(1), cache)
            steps += 1
        return logp, ent, n_s, steps

    # ---------------------------------------------------------------- exact rescoring
    def score(self, programs: list, prefix: torch.Tensor, ctx: list, temperature, chunk: int = 256,
              exact_rows: bool = False):
        """Differentiable log-probability of the recorded hole tokens (LoRA + prefix get gradients). `temperature` is
        a float or a per-row tensor [R] / [R, 1]. Returns (logp [R], entropy [R], n_tokens [R]).

        exact_rows=True evaluates every row alone without padding: in bf16 a padded batched forward depends on the
        batch's padding length (~0.01-0.06 nats), so row-wise scoring is batch-invariant (PPO ratios exactly 1 at the
        behaviour parameters whatever the minibatch composition)."""
        if exact_rows and len(programs) > 1:
            outs = []
            for r in range(len(programs)):
                tau = temperature.reshape(-1)[r: r + 1] if isinstance(temperature, torch.Tensor) else temperature
                outs.append(self.score(programs[r: r + 1], prefix[r: r + 1], ctx[r: r + 1], tau, chunk))
            return tuple(torch.cat([o[k] for o in outs]) for k in range(3))
        enc = self.enc
        R = len(programs)
        dev = enc.device
        tails, positions, targets, specs, owner = [], [], [], [], []
        for r, p in enumerate(programs):
            seq = list(self.stub_ids)
            spent = 0
            for k, h in enumerate(p.holes):
                seq += p.forced[k]
                later = sum(p.reserve[k + 1:])
                cap = min(p.cap[k], self.opts.max_action_tokens - spent - later)
                s, c = 0, 0
                if not h.tokens:
                    raise GrammarError("hole without recorded tokens")
                for u, t in enumerate(h.tokens):
                    specs.append((h, s, c, cap - u))
                    positions.append(len(seq))      # index inside the tail of the token being predicted
                    targets.append(t)
                    owner.append(r)
                    seq.append(t)
                    s, c, closed = self._advance(h, s, c, t)
                    if closed and u != len(h.tokens) - 1:
                        raise GrammarError("recorded hole closes before its last token")
                if not closed:
                    raise GrammarError("recorded hole does not close")
                spent += len(h.tokens)
                h.value = self._hole_value(h)
            tails.append(seq)
        emb, att, pos, starts = self._assemble(prefix, ctx, tails)
        hid = enc.decoder_hidden(emb, att, pos)                                     # [R, L, H]
        own = torch.tensor(owner, device=dev)
        abs_pos = torch.tensor([starts[o] + q - 1 for o, q in zip(owner, positions)], device=dev)
        tgt = torch.tensor(targets, device=dev)
        if isinstance(temperature, torch.Tensor):
            tau_all = temperature.reshape(-1).to(dev)[own].unsqueeze(1) if targets else None
        else:
            tau_all = None
        lps, ents = [], []
        for c0 in range(0, len(targets), chunk):
            sl = slice(c0, c0 + chunk)
            hsel = hid[own[sl], abs_pos[sl]]
            lg = enc.lm_logits(hsel)
            m = self.masks(specs[sl])
            bad = ~m.gather(1, tgt[sl].unsqueeze(1)).squeeze(1)
            if bool(bad.any()):
                raise GrammarError("recorded token outside the replayed constraint mask")
            lp = masked_log_softmax(lg, m, tau_all[sl] if tau_all is not None else temperature)
            lps.append(lp.gather(1, tgt[sl].unsqueeze(1)).squeeze(1))
            ents.append(masked_entropy(lp, m))
        logp = torch.zeros(R, device=dev, dtype=torch.float32)
        ent = torch.zeros(R, device=dev, dtype=torch.float32)
        if targets:
            logp = logp.index_add(0, own, torch.cat(lps))
            ent = ent.index_add(0, own, torch.cat(ents))
        n_tok = torch.bincount(own, minlength=R) if targets else torch.zeros(R, dtype=torch.long, device=dev)
        return logp, ent, n_tok

    # ---------------------------------------------------------------- behaviour cloning support
    def tokenize_hole(self, kind: str, value: str, h: Hole, cap: int) -> list:
        """Token ids that reproduce `value` in hole `h` under its constraint (encoder for demonstrations)."""
        T = self.tables
        if kind == "pattern":
            if not h.dfa.matches(value):
                raise GrammarError(f"value {value!r} does not match pattern {h.dfa.pattern!r}")
            ids = [T.char_id[c] for c in value] + [T.quote_id]
        else:
            lit = json.dumps(value, ensure_ascii=False)[1:]
            ids = self.enc.tok(lit, add_special_tokens=False)["input_ids"]
            s, ok = 0, True
            for t in ids:
                if t >= T.V or not bool(T.valid_h[s, t]):
                    ok = False
                    break
                s = int(T.end_h[s, t])
            if not ok:                                    # fall back to one token per character
                ids = []
                for ch in lit:
                    if ch not in T.char_id:
                        raise GrammarError(f"character {ch!r} has no single-token form")
                    ids.append(T.char_id[ch])
        if len(ids) > cap:
            raise GrammarError(f"string needs {len(ids)} tokens > cap {cap}")
        return ids
