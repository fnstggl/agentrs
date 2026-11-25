"""Schema-walking constrained argument generation (spec 9.2; architecture §11 decision 1).

An action's arguments are produced in two phases, both exactly re-scorable:

1. **Skeleton (CAM's typed heads).** `walk()` walks the tool's JSON schema (Crucible's strict subset, see
   `crucible.core.schema`) and emits the JSON structure deterministically. Every non-structural choice is a
   `Decision` answered by `ArgHeads`, conditioned on CAM's decision context [R; G; B], the tool token, the property's
   *schema embedding* (semantic embedding of its path, description and constraints) and a GRU walk state over the
   previous choices of the same action:
     presence of optional properties, the member of a type list (`"type": ["string", "null"]` or untyped values),
     array lengths, free-map key counts, booleans, enums (pointer over the semantically encoded enum values) and
     numbers. A number is produced in one of three ways, all exact and inside the schema's bounds:
       * **copy** (`numsrc` pointer): one of the numbers shown in the CAM-selected evidence (an observation's `value`
         or a numeric literal of its rendered text) that satisfies the slot - e.g. the exact amount of an invoice;
       * small integer ranges (<= num_base**num_levels values): num_levels base-num_base digits of the offset;
       * otherwise **exact decimal digits** (`DecimalSpec`): fractional-digit count k (numbers only, <= float_digits),
         sign, digit count L and L decimal digits MSB-first; every digit mask keeps a completion inside
         [minimum, maximum] reachable, so any integer (e.g. cents up to the bound) and any decimal with <= k
         fractional digits can be written exactly. Each value has exactly one choice sequence.
   Decisions with a single admissible option are resolved without a head call and contribute log 1 = 0.
2. **Strings (Qwen + LoRA).** Free strings, patterned strings and free-map keys are left as `Hole`s. The template is
   serialised to JSON text: forced segments (structure, keys, head-chosen values) and holes. `HoleFiller` feeds the
   forced tokens and samples the hole tokens with the Qwen decoder (LoRA on) conditioned on the bridge prefix
   (inputs_embeds), the tool name/description/schema and the CAM-selected evidence, under the exact token masks of
   `constraints.py` (escape-aware JSON string body ending with the closing quote; regex DFA for patterns; length bounds;
   per-string budget `max_arg_tokens` and a per-action budget).

Every capability reduction the compiled grammar makes relative to the schema (arrays capped at max_array_items,
free maps at max_map_keys keys, optional properties or type-list members that cannot be generated, missing numeric
bounds replaced by +-default_num_bound, fractional digits limited to float_digits, string token caps, untyped-value
depth) is recorded in `ToolGrammar.reductions`; the action generator surfaces them (`ActionGenerator.reduced`, stats).

The result always satisfies `crucible.core.schema.validate` (checked after every generation; a violation would be a bug
and raises). The `record` (tool, evidence, head choices, hole token ids, prompt hash, temperature) is enough to
recompute the behaviour log-probability exactly (`log_prob` re-walks with teacher forcing and re-applies identical
masks).

Compute per action: the skeleton costs one ContextReader call per (non-trivial) decision, batched over the B rows in
lockstep rounds (an exact decimal number costs 2-3 + L decisions); Qwen costs one prefill of the prompt (+ prefix +
stub + first forced segment) and one decoder step per subsequent forced or sampled token, batched over the rows that
have holes. Tools without string slots never touch Qwen.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraints import (JsonStringTables, PatternMasks, UnsupportedSchema, compile_pattern)
from .layers import ContextReader, RMSNorm, masked_entropy, masked_log_softmax

MAX_STRING = 200_000                       # crucible.core.schema.MAX_STRING
JSON_TYPES = ["null", "boolean", "integer", "number", "string", "object", "array"]


class GrammarError(RuntimeError):
    """An internal inconsistency (a record that does not replay, an invalid result). Never swallowed."""


@dataclass
class GrammarOptions:
    num_base: int = 16                     # small exact integer grids: num_levels digits of base num_base
    num_levels: int = 3
    max_digits: int = 22                   # exact decimal mode: at most this many digits of N (value = +-N / 10^k)
    float_digits: int = 6                  # at most this many fractional digits for `number` slots
    max_array_items: int = 8
    max_map_keys: int = 4
    untyped_depth: int = 2
    default_num_bound: float = 1e15        # replaces a missing minimum / maximum (recorded as a reduction)
    max_copy_candidates: int = 24          # numbers of the selected evidence offered to the copy pointer
    max_string_tokens: int = 1536          # cfg.max_arg_tokens: the cap of code / file fields (string class "code")
    max_key_tokens: int = 48
    max_action_tokens: int = 3072
    slot_chars: int = 320
    # schema-derived token caps of the other free-string classes (see `string_class`); never above max_string_tokens
    short_string_tokens: int = 96          # identifiers, handles, names, titles, short references
    text_string_tokens: int = 768          # prose: descriptions, bodies, notes, messages, comments
    length_prior: bool = True              # learned stop hazard mixed into the decoder's closing probability

    @property
    def kinds(self) -> list:
        return (["presence", "bool", "union", "count", "mapcount", "enum", "numsrc", "sign", "fdig", "ndig", "dec"]
                + [f"digit{l}" for l in range(self.num_levels)])

    def n_options(self, kind: str) -> int:
        """Number of outputs of a categorical head (0 = pointer kinds)."""
        return {"presence": 2, "bool": 2, "union": len(JSON_TYPES), "count": self.max_array_items + 1,
                "mapcount": self.max_map_keys + 1, "enum": 0, "numsrc": 0, "sign": 2, "fdig": self.float_digits + 1,
                "ndig": self.max_digits, "dec": 10}.get(kind, self.num_base)


def _validate(schema: dict, value) -> list:
    from crucible.core.schema import validate
    return validate(schema, value)


# ============================================================================ numbers
class NumberGrid:
    """Exact grid of a small integer range [lo, hi] (hi - lo + 1 <= num_base**num_levels): index g -> lo + g, chosen as
    num_levels base-num_base digits (MSB first); digit masks keep the index inside the range."""

    def __init__(self, lo: int, hi: int, opts: GrammarOptions):
        self.lo, self.hi = int(lo), int(hi)
        self.K, self.L = opts.num_base, opts.num_levels
        self.n = self.hi - self.lo + 1
        if not self.fits(self.lo, self.hi, opts):
            raise ValueError(f"range [{lo}, {hi}] does not fit the exact grid")

    @staticmethod
    def fits(lo: int, hi: int, opts: GrammarOptions) -> bool:
        return 1 <= hi - lo + 1 <= opts.num_base ** opts.num_levels

    def value(self, g: int) -> int:
        return self.lo + int(g)

    def index(self, v) -> int:
        return int(v) - self.lo

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


def _feasible(a: int, b: int, nz_last: bool) -> bool:
    """The integer interval [a, b] has a member (with a non-zero last decimal digit when nz_last)."""
    return a <= b and not (nz_last and a == b and a % 10 == 0)


def _dec(x) -> Decimal:
    """Exact decimal of an int, or the shortest round-tripping decimal of a float."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        return Decimal(x)
    return Decimal(repr(float(x)))


class DecimalSpec:
    """Exact decimal digits for an integer / number slot whose range does not fit a small grid.

    value = (-1)^s * N / 10^k with k in [0, F] fractional digits (F = float_digits for numbers, 0 for integers), sign s,
    and N >= 0 written with L decimal digits MSB-first (no leading zero; N = 0 is "0" with sign +). For k > 0 the last
    digit of N is non-zero, so every value has exactly one choice sequence (encode() is its inverse). Scaled bounds:
    N*(-1)^s in [ceil(lo * 10^k), floor(hi * 10^k)] computed in exact decimal arithmetic from the shortest decimal of
    the float bounds, so the emitted value always satisfies minimum <= v <= maximum. Magnitude caps keep every value
    exact: integers N <= 10^max_digits - 1 (Python ints); numbers N <= 2^53 for k = 0 (exact doubles) and
    N <= 10^15 - 1 for k > 0 (<= 15 significant digits round-trip through a double), so the emitted JSON number is
    exactly the decimal the digits spell. Bounds beyond the caps are clamped (`clamped`, recorded as a reduction)."""

    def __init__(self, lo, hi, integer: bool, opts: GrammarOptions):
        self.lo, self.hi, self.integer = lo, hi, integer
        self.F = 0 if integer else int(opts.float_digits)
        self.maxd = int(opts.max_digits)
        self.clamped = False
        self.bounds = []
        for k in range(self.F + 1):
            if integer:
                a, b = int(lo), int(hi)
                cap = 10 ** self.maxd - 1
            else:
                a = math.ceil(_dec(lo).scaleb(k))
                b = math.floor(_dec(hi).scaleb(k))
                cap = min(2 ** 53 if k == 0 else 10 ** 15 - 1, 10 ** self.maxd - 1)
            if a < -cap or b > cap:
                self.clamped = True
            self.bounds.append((max(a, -cap), min(b, cap)))
        if not any(self.k_mask()):
            raise UnsupportedSchema(f"no decimal value in [{lo}, {hi}] with <= {self.F} fractional digits")

    def k_mask(self) -> list:
        return [_feasible(a, b, k > 0) for k, (a, b) in enumerate(self.bounds)]

    def mag(self, k: int, s: int) -> tuple:
        a, b = self.bounds[k]
        return (max(a, 0), b) if s == 0 else (max(-b, 1), -a)

    def sign_mask(self, k: int) -> list:
        return [_feasible(*self.mag(k, s), k > 0) for s in (0, 1)]

    @staticmethod
    def len_interval(L: int) -> tuple:
        return (0 if L == 1 else 10 ** (L - 1), 10 ** L - 1)

    def len_mask(self, k: int, s: int) -> list:
        ma, mb = self.mag(k, s)
        out = []
        for L in range(1, self.maxd + 1):
            la, lb = self.len_interval(L)
            out.append(_feasible(max(ma, la), min(mb, lb), k > 0))
        return out

    def digit_mask(self, k: int, s: int, L: int, pos: int, prefix: int) -> list:
        ma, mb = self.mag(k, s)
        la, lb = self.len_interval(L)
        ma, mb = max(ma, la), min(mb, lb)
        r = 10 ** (L - 1 - pos)
        out = []
        for d in range(10):
            x = (prefix * 10 + d) * r
            out.append(_feasible(max(x, ma), min(x + r - 1, mb), k > 0))
        return out

    def value(self, k: int, s: int, n: int):
        signed = -n if s else n
        if self.integer:
            return int(signed)
        return float(Decimal(signed).scaleb(-k)) if k else float(signed)

    def encode(self, v) -> tuple:
        """Value -> (k, s, N, approx). approx = the value needed rounding (more than F fractional digits, or a
        non-integral value for an integer slot)."""
        approx = False
        if self.integer:
            if isinstance(v, float) and not v.is_integer():
                approx = True
            n, k = int(round(v)), 0
        else:
            d = _dec(v)
            for k in range(self.F + 1):
                x = d.scaleb(k)
                if x == x.to_integral_value():
                    break
            else:
                k = self.F
                x = d.scaleb(k).to_integral_value(rounding=ROUND_HALF_EVEN)
                approx = True
            n = int(x)
            while k > 0 and n % 10 == 0:
                n //= 10
                k -= 1
        return k, (1 if n < 0 else 0), abs(n), approx


@dataclass(eq=False)
class NumCand:
    """A number shown in the selected evidence: its text, exact value and the context the copy pointer reads."""
    text: str
    dec: Decimal
    key: str
    source: int = -1                        # evidence position it came from

    def value_for(self, node: "Node"):
        return int(self.dec) if node.kind == "integer" else float(self.dec)

    def admissible(self, node: "Node") -> bool:
        if not self.dec.is_finite():
            return False
        if node.kind == "integer":
            if self.dec != self.dec.to_integral_value():
                return False
            v = int(self.dec)
        else:
            v = float(self.dec)
            if not math.isfinite(v):
                return False
        return node.num.lo <= v <= node.num.hi


_NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")


def number_candidates(sources: list, max_n: int, ctx_chars: int = 48) -> list:
    """sources: per evidence observation (exact values [(value, key text)], rendered text) -> list[NumCand] in
    evidence order (exact `value` fields first, then numeric literals of the text; thousands separators allowed;
    digits inside identifiers like acct_9 are not numbers), deduplicated by exact value, at most max_n."""
    out, seen = [], set()

    def add(text, d, key, src):
        if len(out) >= max_n or d in seen or not d.is_finite():
            return
        seen.add(d)
        out.append(NumCand(text, d, key, src))

    for src, (values, text) in enumerate(sources):
        for v, key in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            add(repr(v), _dec(v), key, src)
        for m in _NUM_RE.finditer(text or ""):
            raw = m.group(0)
            ctx = text[max(0, m.start() - ctx_chars): m.end() + ctx_chars // 2].replace("\n", " ")
            add(raw, Decimal(raw.replace(",", "")), f"{raw} in: {ctx}", src)
    return out


# ============================================================================ schema compilation
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
    num: Any = None                                 # integer/number: NumberGrid | DecimalSpec
    dropped: list = field(default_factory=list)     # optional properties that can never be generated (logged)
    str_class: str = "short"                        # string: "short" | "text" | "code" (string_class)
    tok_cap: int = 0                                # string: schema-derived token cap of the hole (incl. the quote)

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

    def has_number(self) -> bool:
        if self.kind in ("integer", "number"):
            return True
        if self.kind == "object":
            return (self.extra is not None and self.extra.has_number()) or any(c.has_number() for _, c, _ in self.props)
        if self.kind == "array":
            return self.hi > 0 and self.item.has_number()
        if self.kind == "union":
            return any(o.has_number() for o in self.options.values())
        return False


STRING_CLASSES = ("short", "text", "code")
_CODE_NAME = re.compile(r"(?:^|[_\-])(code|source|src|script|program|patch|diff|files?|contents?|html|css|sql|"
                        r"notebook|module)$", re.IGNORECASE)
_CODE_DESC = re.compile(r"source code|python|program text|file contents?|script body", re.IGNORECASE)
_TEXT_NAME = re.compile(r"(?:^|[_\-])(body|text|description|desc|note|notes|message|comment|bio|memo|reason|summary|"
                        r"readme|prompt|answer|reply|details?|content_text|abstract|instructions?)$", re.IGNORECASE)
# token caps implied by JSON-schema `format` hints (generous: Qwen tokens per character are >= 1/4 on such text)
FORMAT_TOKENS = {"date-time": 48, "date": 24, "time": 24, "duration": 32, "email": 96, "idn-email": 96,
                 "hostname": 96, "idn-hostname": 96, "ipv4": 24, "ipv6": 64, "uuid": 48, "uri": 256, "url": 256,
                 "iri": 256, "uri-reference": 256, "iri-reference": 256, "json-pointer": 128, "regex": 128}


def _field_name(path: str) -> str:
    """Last named segment of a slot path ('.files.*' -> 'files', '.tags[]' -> 'tags')."""
    for seg in reversed(re.split(r"[.\[\]]+", path or "")):
        if seg and seg != "*":
            return seg
    return ""


def string_class(path: str, desc: str, s: dict) -> str:
    """Schema-derived class of a free string slot, which sets its token cap and its length prior:
    "code" (code / file-content fields by name or description: keeps opts.max_string_tokens = cfg.max_arg_tokens),
    "text" (prose fields by name, or any string whose maxLength is >= 1000 or absent) and "short" (everything else:
    identifiers, handles, names, titles, references). Names are the environment's own field names; a randomised field
    name with a long maxLength still lands in "text"."""
    name = _field_name(path)
    if _CODE_NAME.search(name) or _CODE_DESC.search(desc or ""):
        return "code"
    mx = s.get("maxLength")
    if _TEXT_NAME.search(name) or mx is None or int(mx) >= 1000:
        return "text"
    return "short"


def string_token_cap(cls: str, s: dict, opts: "GrammarOptions") -> int:
    """Token cap of a free-string hole (including its closing quote) before the maxLength bound."""
    cap = {"code": opts.max_string_tokens, "text": opts.text_string_tokens,
           "short": opts.short_string_tokens}[cls]
    fmt = s.get("format")
    if isinstance(fmt, str) and fmt in FORMAT_TOKENS and cls != "code":
        cap = FORMAT_TOKENS[fmt]
    return int(min(cap, opts.max_string_tokens))


def _compact(s: dict) -> str:
    keep = {k: v for k, v in s.items() if k not in ("description", "properties", "items", "additionalProperties",
                                                      "required")}
    return json.dumps(keep, sort_keys=True, separators=(",", ":"), default=str)[:200]


def _number_node(t: str, path: str, desc: str, st: str, lo, hi, opts: GrammarOptions, red: list) -> Node:
    integer = t == "integer"
    b = opts.default_num_bound
    missing = [n for n, v in (("minimum", lo), ("maximum", hi)) if v is None or not math.isfinite(float(v))]
    if missing:
        red.append({"path": path, "kind": "default_bound", "detail": f"{'/'.join(missing)} -> +-{b:g}"})
    lo = -b if "minimum" in missing else lo
    hi = b if "maximum" in missing else hi
    if integer:
        lo, hi = int(math.ceil(lo)), int(math.floor(hi))
    if lo > hi:
        raise UnsupportedSchema(f"{path}: empty numeric range")
    if integer and NumberGrid.fits(lo, hi, opts):
        num = NumberGrid(lo, hi, opts)
    else:
        num = DecimalSpec(lo, hi, integer, opts)
        if num.clamped:
            red.append({"path": path, "kind": "bound_clamped", "detail": "magnitude capped so every value is exact "
                                                                         "(integers < 10^max_digits; numbers: <= 2^53, "
                                                                         "or <= 15 significant digits)"})
        if not integer:
            red.append({"path": path, "kind": "fraction_digits", "detail": f"<= {opts.float_digits} fractional digits"})
    return Node(t, path, desc, st, num=num)


def _untyped(path: str, depth: int, opts: GrammarOptions, red: list) -> Node:
    red.append({"path": path, "kind": "untyped", "detail": f"any JSON value: nesting <= {depth}, numbers within "
                                                           f"+-{opts.default_num_bound:g}"})
    sink: list = []                         # the untyped members' own default bounds are covered by the line above
    b = opts.default_num_bound
    opt = {"null": Node("null", path), "boolean": Node("boolean", path),
           "integer": _number_node("integer", path, "", "", -b, b, opts, sink),
           "number": _number_node("number", path, "", "", -b, b, opts, sink),
           "string": Node("string", path, str_class="text",
                          tok_cap=min(opts.text_string_tokens, opts.max_string_tokens))}
    if depth > 0:
        opt["object"] = Node("object", path, extra=_untyped(path + ".*", depth - 1, opts, sink))
        opt["array"] = Node("array", path, item=_untyped(path + "[]", depth - 1, opts, sink), lo=0,
                            hi=opts.max_array_items)
    return Node("union", path, desc="any JSON value", options=opt)


def compile_schema(s: Any, path: str, opts: GrammarOptions, depth: int, red: Optional[list] = None) -> Node:
    """Compile a JSON schema (Crucible subset) into a Node tree. Raises UnsupportedSchema when no schema-valid value
    can be generated under the options (budget, caps, regex features). Capability reductions are appended to `red`
    as {"path", "kind", "detail"}."""
    red = [] if red is None else red
    if s is True or s == {}:
        return _untyped(path, depth, opts, red)
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
            sub_red: list = []
            try:
                opt[ti] = compile_schema({**s, "type": ti}, path, opts, depth, sub_red)
                red.extend(sub_red)
            except UnsupportedSchema as e:
                red.append({"path": path, "kind": "dropped_type", "detail": f"{ti}: {e}"})
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
            u = _untyped(path, depth, opts, red)
            u.desc, u.schema_text = desc, st
            return u
    if t == "object":
        props = s.get("properties") or {}
        req = list(s.get("required") or [])
        addl = s.get("additionalProperties", False)
        extra = None
        if addl is True:
            extra = _untyped(path + ".*", max(depth - 1, 0), opts, red)
        elif isinstance(addl, dict):
            extra = compile_schema(addl, path + ".*", opts, depth, red)
        if extra is not None:
            red.append({"path": path, "kind": "map_cap", "detail": f"at most {opts.max_map_keys} free keys"})
        node = Node("object", path, desc, st, extra=extra)
        names = list(props) + [r for r in req if r not in props]
        for name in names:
            sub = props.get(name)
            if sub is None:
                if extra is None:
                    raise UnsupportedSchema(f"{path}.{name}: required but not allowed")
                sub = addl if isinstance(addl, dict) else True
            required = name in req
            sub_red: list = []
            try:
                child = compile_schema(sub, f"{path}.{name}", opts, depth, sub_red)
            except UnsupportedSchema as e:
                if required:
                    raise
                node.dropped.append(name)
                red.append({"path": f"{path}.{name}", "kind": "dropped_optional", "detail": str(e)})
                continue
            red.extend(sub_red)
            node.props.append((name, child, required))
        return node
    if t == "array":
        lo = int(s.get("minItems", 0))
        mx = int(s.get("maxItems", 10_000))
        hi = min(mx, opts.max_array_items)
        if lo > hi:
            raise UnsupportedSchema(f"{path}: minItems {lo} exceeds the generator cap {hi}")
        if mx > hi:
            red.append({"path": path, "kind": "array_cap", "detail": f"maxItems {mx} -> {hi}"})
        items = s.get("items")
        sub_red: list = []
        try:
            item = compile_schema(items, path + "[]", opts, depth, sub_red) if items is not None else \
                _untyped(path + "[]", max(depth - 1, 0), opts, sub_red)
            red.extend(sub_red)
        except UnsupportedSchema as e:
            if lo > 0:
                raise
            red.append({"path": path, "kind": "array_empty_only", "detail": f"items not generable: {e}"})
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
            if mx > cap - 1:
                red.append({"path": path, "kind": "string_cap", "detail": f"pattern string <= {cap - 1} chars"})
        else:
            if not JsonStringTables.feasible(mn, mx, opts.max_string_tokens):
                raise UnsupportedSchema(f"{path}: length bounds infeasible")
            cls = string_class(path, desc, s)
            # a token adds >= 1 decoded character except a lone backslash (which the next token completes), so a
            # string of at most mx characters never needs more than 2*mx + 1 tokens (incl. the quote); minLength
            # characters always fit in mn + 1 tokens
            full = 2 * mx + 1
            cap = max(min(string_token_cap(cls, s, opts), full), mn + 1)
            node.str_class, node.tok_cap = cls, cap
            if cap < full:
                fmt = s.get("format")
                red.append({"path": path, "kind": "string_cap",
                            "detail": f"{cls} string <= {cap - 1} tokens (maxLength {mx}"
                                      f"{', format ' + fmt if isinstance(fmt, str) and fmt in FORMAT_TOKENS else ''})"})
        return node
    if t in ("integer", "number"):
        return _number_node(t, path, desc, st, s.get("minimum"), s.get("maximum"), opts, red)
    if t == "boolean":
        return Node("boolean", path, desc, st)
    if t == "null":
        return Node("null", path, desc, st)
    raise UnsupportedSchema(f"{path}: unknown type {t!r}")


class ToolGrammar:
    """Compiled argument grammar of one tool (cached by ToolSchema.key) with its capability reductions."""

    def __init__(self, tool, opts: GrammarOptions):
        self.tool = tool
        self.key = tool.key
        self.error: Optional[str] = None
        self.root: Optional[Node] = None
        self.reductions: list = []
        params = tool.parameters or {"type": "object", "properties": {}}
        try:
            red: list = []
            root = compile_schema(params, "", opts, opts.untyped_depth, red)
            if any(r["kind"] == "untyped" for r in red) and \
                    min(opts.text_string_tokens, opts.max_string_tokens) < opts.max_string_tokens:
                red.append({"path": "", "kind": "string_cap", "detail": f"untyped strings <= "
                            f"{min(opts.text_string_tokens, opts.max_string_tokens) - 1} tokens"})
            if root.kind != "object":
                raise UnsupportedSchema("top-level arguments schema must be an object")
            self.root = root
            self.reductions = red
        except UnsupportedSchema as e:
            self.error = str(e)

    @property
    def generable(self) -> bool:
        return self.root is not None

    @property
    def may_hole(self) -> bool:
        return self.root is not None and self.root.may_hole()

    @property
    def uses_evidence(self) -> bool:
        """Evidence feeds the string decoder and the number-copy pointer."""
        return self.root is not None and (self.root.may_hole() or self.root.has_number())


# ============================================================================ walking
@dataclass(eq=False)
class Decision:
    kind: str
    node: Node
    mask: list
    values: Optional[list] = None
    aux: int = -1                   # 'dec': place value (number of digits after this one)


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


def _walk_number(node: Node, opts: GrammarOptions, cands: Optional[list]):
    if cands:
        adm = [c.admissible(node) for c in cands]
        if any(adm):
            c = yield from _ask(Decision("numsrc", node, [True] + adm, cands))
            if c > 0:
                return cands[c - 1].value_for(node)
    spec = node.num
    if isinstance(spec, NumberGrid):
        g = 0
        for l in range(opts.num_levels):
            d = yield from _ask(Decision(f"digit{l}", node, spec.digit_mask(l, g)))
            g = g * opts.num_base + d
        return spec.value(g)
    k = (yield from _ask(Decision("fdig", node, spec.k_mask()))) if spec.F > 0 else 0
    s = yield from _ask(Decision("sign", node, spec.sign_mask(k)))
    L = 1 + (yield from _ask(Decision("ndig", node, spec.len_mask(k, s))))
    n = 0
    for pos in range(L):
        d = yield from _ask(Decision("dec", node, spec.digit_mask(k, s, L, pos, n), aux=L - 1 - pos))
        n = n * 10 + d
    return spec.value(k, s, n)


def walk(node: Node, opts: GrammarOptions, cands: Optional[list] = None):
    """Generator: yields Decisions, receives chosen indices, returns the template value (with Holes). `cands` are
    the row's evidence numbers (NumCand) offered to the copy pointer of every numeric slot."""
    k = node.kind
    if k == "object":
        entries = []
        for name, child, req in node.props:
            if not req:
                if (yield from _ask(Decision("presence", child, [True, True]))) == 0:
                    continue
            entries.append((name, (yield from walk(child, opts, cands))))
        if node.extra is not None:
            n = yield from _ask(Decision("mapcount", node, [True] * (opts.max_map_keys + 1)))
            for _ in range(n):
                key = Hole("key", node, 0, MAX_STRING)
                entries.append((key, (yield from walk(node.extra, opts, cands))))
        return TObject(node, entries)
    if k == "array":
        n = yield from _ask(Decision("count", node, [node.lo <= i <= node.hi for i in range(opts.max_array_items + 1)]))
        out = []
        for _ in range(n):
            out.append((yield from walk(node.item, opts, cands)))
        return out
    if k == "string":
        return Hole("pattern" if node.dfa is not None else "str", node, node.min_len, node.max_len, node.dfa)
    if k == "enum":
        return node.values[(yield from _ask(Decision("enum", node, [True] * len(node.values), node.values)))]
    if k == "const":
        return node.values[0]
    if k in ("integer", "number"):
        return (yield from _walk_number(node, opts, cands))
    if k == "boolean":
        return bool((yield from _ask(Decision("bool", node, [True, True]))))
    if k == "null":
        return None
    if k == "union":
        c = yield from _ask(Decision("union", node, [t in node.options for t in JSON_TYPES]))
        return (yield from walk(node.options[JSON_TYPES[c]], opts, cands))
    raise GrammarError(f"unknown node kind {k}")


def encode_value(node: Node, value, opts: GrammarOptions, choices: list, holes: list, flags: dict,
                 cands: Optional[list] = None) -> None:
    """Inverse of `walk` for behaviour cloning: append the choices (same auto-resolution rule) and hole strings that
    make `walk` reproduce `value`. A number equal to an admissible evidence number is encoded as a copy of the first
    such candidate; otherwise as its exact digits (more than float_digits fractional digits round and are counted in
    flags['approx']; flags['numbers'] / flags['copied'] count the numbers seen / copied)."""
    def put(dec: Decision, c: int):
        allowed = [i for i, m in enumerate(dec.mask) if m]
        if c not in allowed:
            raise GrammarError(f"{node.path}: value {value!r} not representable ({dec.kind}={c})")
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
                encode_value(child, value[name], opts, choices, holes, flags, cands)
        extra_keys = [kk for kk in value if kk not in declared]
        if node.extra is not None:
            if len(extra_keys) > opts.max_map_keys:
                raise GrammarError(f"{node.path}: {len(extra_keys)} map keys exceed the cap {opts.max_map_keys}")
            put(Decision("mapcount", node, [True] * (opts.max_map_keys + 1)), len(extra_keys))
            for kk in extra_keys:
                holes.append(("key", str(kk)))
                encode_value(node.extra, value[kk], opts, choices, holes, flags, cands)
        elif extra_keys:
            raise GrammarError(f"{node.path}: unexpected properties {extra_keys}")
        return
    if k == "array":
        if not isinstance(value, list) or not (node.lo <= len(value) <= node.hi):
            raise GrammarError(f"{node.path}: array length outside [{node.lo}, {node.hi}]")
        put(Decision("count", node, [node.lo <= i <= node.hi for i in range(opts.max_array_items + 1)]), len(value))
        for v in value:
            encode_value(node.item, v, opts, choices, holes, flags, cands)
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
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise GrammarError(f"{node.path}: expected a finite number")
        flags["numbers"] = flags.get("numbers", 0) + 1
        if cands:
            adm = [c.admissible(node) for c in cands]
            if any(adm):
                j = next((i for i, c in enumerate(cands) if adm[i] and c.value_for(node) == value), None)
                put(Decision("numsrc", node, [True] + adm), 0 if j is None else j + 1)
                if j is not None:
                    flags["copied"] = flags.get("copied", 0) + 1
                    return
        spec = node.num
        if isinstance(spec, NumberGrid):
            if isinstance(value, float) and not value.is_integer():
                flags["approx"] = flags.get("approx", 0) + 1
            g = spec.index(round(value))
            if not 0 <= g < spec.n:
                raise GrammarError(f"{node.path}: {value!r} outside [{spec.lo}, {spec.hi}]")
            pre = 0
            for l, d in enumerate(spec.to_digits(g)):
                put(Decision(f"digit{l}", node, spec.digit_mask(l, pre)), d)
                pre = pre * opts.num_base + d
            return
        kk, s, n, approx = spec.encode(value)
        if approx:
            flags["approx"] = flags.get("approx", 0) + 1
        if spec.F > 0:
            put(Decision("fdig", node, spec.k_mask()), kk)
        put(Decision("sign", node, spec.sign_mask(kk)), s)
        digits = str(n)
        L = len(digits)
        put(Decision("ndig", node, spec.len_mask(kk, s)), L - 1)
        pre = 0
        for pos, ch in enumerate(digits):
            put(Decision("dec", node, spec.digit_mask(kk, s, L, pos, pre)), int(ch))
            pre = pre * 10 + int(ch)
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
                encode_value(sub, value, opts, choices, holes, flags, cands)
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


def drive_walks(roots: list, opts: GrammarOptions, decide: Callable[[list, list], list],
                cands: Optional[list] = None) -> list:
    """Run one walk per row in lockstep; `decide(rows, decisions) -> choices` is called once per round with the
    pending decision of every unfinished row. cands[r] are row r's evidence numbers. Returns the templates (None for
    rows whose root is None)."""
    gens, pending, result = {}, {}, [None] * len(roots)
    for r, root in enumerate(roots):
        if root is None:
            continue
        g = walk(root, opts, cands[r] if cands is not None else None)
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
POINTER_KINDS = ("enum", "numsrc")


class ArgHeads(nn.Module):
    """Typed decision heads. A decision's query is q = h_walk + E_slot + E_kind + tool_token (+ E_place for decimal
    digits: the number of digits after this one); it reads the decision context through a ContextReader; categorical
    kinds use per-kind linear outputs, enums a pointer onto the encoded enum values, `numsrc` a pointer onto the
    encoded evidence numbers (context text around each number) plus a "write the digits" logit. After each choice the
    GRU walk state is updated with (E_slot + E_choice)."""

    def __init__(self, d: int, n_heads: int, sem_dim: int, opts: GrammarOptions):
        super().__init__()
        self.opts = opts
        self.kinds = opts.kinds
        self.kind_index = {k: i for i, k in enumerate(self.kinds)}
        self.slot_proj = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.kind_emb = nn.Embedding(len(self.kinds), d)
        self.place = nn.Embedding(opts.max_digits + 1, d)
        nn.init.normal_(self.place.weight, std=0.02)
        self.reader = ContextReader(d, n_heads, 1)
        self.out = nn.ModuleDict({k: nn.Linear(d, opts.n_options(k)) for k in self.kinds if k not in POINTER_KINDS})
        self.choice_emb = nn.ModuleDict({k: nn.Embedding(opts.n_options(k), d) for k in self.kinds
                                         if k not in POINTER_KINDS})
        self.enum_q = nn.Linear(d, d)
        self.enum_k = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.copy_q = nn.Linear(d, d)
        self.copy_k = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.gen_logit = nn.Linear(d, 1)
        self.gen_emb = nn.Parameter(torch.randn(d) * 0.02)
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
        aux = torch.tensor([max(d.aux, 0) for d in decisions], device=dev)
        has_aux = torch.tensor([d.aux >= 0 for d in decisions], device=dev).unsqueeze(1)
        q = h + slot + self.kind_emb(kind) + tool_tok + self.place(aux) * has_aux.to(slot.dtype)
        f = self.reader(q.unsqueeze(1), ctx_x, ctx_mask).squeeze(1)                   # [n, D]
        choices, logps, ents, cembs = [], [], [], []
        for i, d in enumerate(decisions):
            vk = None
            if d.kind == "enum":
                vk = self.enum_k(enc.embed([json.dumps(v, ensure_ascii=False) for v in d.values]).to(dev))  # [m, D]
                logits = (self.enum_q(f[i]) @ vk.t()) / math.sqrt(self.d)
            elif d.kind == "numsrc":
                vk = self.copy_k(enc.embed([c.key[: self.opts.slot_chars] for c in d.values]).to(dev))     # [m, D]
                logits = torch.cat([self.gen_logit(f[i]).float(),
                                    ((self.copy_q(f[i]) @ vk.t()) / math.sqrt(self.d)).float()])
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
            if d.kind == "enum":
                cembs.append(vk[c])
            elif d.kind == "numsrc":
                cembs.append(self.gen_emb.to(vk.dtype) if c == 0 else vk[c - 1])
            else:
                cembs.append(self.choice_emb[d.kind].weight[c])
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
    cls: list = field(default_factory=list)   # per-hole string class index (STRING_CLASSES) of the length prior
    hz_off: Any = None             # [n_holes] float tensor: learned per-hole stop-logit offsets (None = no prior)


def hole_class(h: Hole) -> str:
    """Length-prior class of a hole: keys and pattern strings are short; free strings carry their slot's class."""
    if h.kind == "str":
        return h.node.str_class
    return "short"


class LengthHead(nn.Module):
    """Learned stop hazard of free-string holes (the length prior of the untrained decoder; trainable end to end).

    At every token of a hole whose constraint allows both closing and continuing, the decoder's next-token
    distribution is the mixture

        p(t) = lam * [t closes] p_LM(t) / P_close  +  (1 - lam) * p_LM(t)          (P_close = sum over closing tokens)

    i.e. with probability lam the string is closed now (choosing among the closing tokens - a lone '"' or 'abc"' - by
    the LM's own relative probabilities); otherwise the LM's distribution is used unchanged, so the LM can still close
    whenever it wants: P(stop) = lam + (1 - lam) P_close >= P_close, and lam -> 0 recovers the pure LM. The hazard is
    a soft length limit with a per-class ceiling

        lam(n) = sigmoid(m_c) * sigmoid((n - L_c) / w_c + h_theta(slot schema embedding, tool token, CAM summary))

    with n = tokens already in the hole, per-class learned ceiling m_c (logit), location L_c and scale w_c (stored as
    logs) and a zero-initialised MLP h_theta (so lam depends on CAM's latent state through the tool-pointer summary of
    [R; G; B] and on the field's schema). Initial values (tokens): short (L 32, w 3, ceiling 0.999), text (192, 24,
    0.999), code (256, 32, 1/192). An untrained decoder that never closes a string therefore stops after about 28
    (short), 104 (text) or 441 (code) tokens in expectation (`expected_length`) instead of running to the hard cap
    (1,536 tokens for code before), while the prior cuts a string the LM would close itself after 12 tokens with
    probability 0.3% (short) / 0.5% (text) and after 40 tokens 3% (text). The code ceiling keeps long code possible at
    init: a never-closing code string survives 1,068 tokens (a measured code.commit) with probability 1.4%, i.e. the
    prior costs such a file ~4 nats. Every parameter
    receives gradients from the exact log-probabilities (closing and continuing tokens both carry the stop factor), so
    behaviour cloning / RL move L_c, w_c, m_c and h_theta when other lengths are needed; the hard caps stay the
    schema-derived ones (string_token_cap)."""

    PRIOR = {"short": (32.0, 3.0, 0.999), "text": (192.0, 24.0, 0.999), "code": (256.0, 32.0, 1.0 / 192.0)}

    def __init__(self, d: int, sem_dim: int):
        super().__init__()
        self.slot_proj = nn.Sequential(RMSNorm(sem_dim), nn.Linear(sem_dim, d))
        self.ff = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
        pr = [self.PRIOR[c] for c in STRING_CLASSES]
        self.log_L = nn.Parameter(torch.tensor([p[0] for p in pr]).log())
        self.log_w = nn.Parameter(torch.tensor([p[1] for p in pr]).log())
        self.ceil_logit = nn.Parameter(torch.tensor([math.log(p[2] / (1 - p[2])) for p in pr]))

    def offsets(self, slot_emb: torch.Tensor, tool_tok: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """Per-hole offsets [n] float32 from slot embeddings [n, H], tool tokens [n, D], decision summaries [n, D]."""
        x = self.slot_proj(slot_emb.to(tool_tok.device)) + tool_tok + summary
        return self.ff(x).squeeze(-1).float()

    def log_hazard(self, off: torch.Tensor, cls: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """log lam [m] for holes with offsets off [m], class indices cls [m] and n [m] tokens so far."""
        L = self.log_L.float().exp()[cls]
        w = self.log_w.float().exp()[cls]
        return F.logsigmoid(self.ceil_logit.float()[cls]) + F.logsigmoid(off.float() + (n.float() - L) / w)

    def expected_length(self, off: float = 0.0, cls: str = "short", horizon: int = 8192) -> float:
        """Expected tokens until the prior alone stops a string (an LM that never closes), truncated at `horizon`."""
        with torch.no_grad():
            dev = self.log_L.device
            c = torch.full((horizon,), STRING_CLASSES.index(cls), dtype=torch.long, device=dev)
            n = torch.arange(horizon, device=dev, dtype=torch.float32)
            lam = self.log_hazard(torch.full((horizon,), float(off), device=dev), c, n).exp()
            surv = torch.cumprod(torch.cat([torch.ones(1, device=dev), 1 - lam[:-1]]), 0)
            return float((surv * lam * (n + 1)).sum() + surv[-1] * (1 - lam[-1]) * horizon)


def stop_mixture_log_softmax(logits: torch.Tensor, allowed: torch.Tensor, close: torch.Tensor, temperature,
                             log_lam: Optional[torch.Tensor]) -> torch.Tensor:
    """Log-probabilities [P, V] of the stop-hazard mixture (LengthHead) over the allowed tokens; log_lam [P] the log
    stop hazards (None = the pure tempered LM). Rows that cannot close (or must close) are the pure LM restricted to
    the mask."""
    lp = masked_log_softmax(logits, allowed, temperature)
    if log_lam is None:
        return lp
    close = close & allowed
    has_c = close.any(-1)
    mix = has_c & (allowed & ~close).any(-1)
    lse_c = torch.logsumexp(lp.masked_fill(~close, -torch.inf), -1)
    lse_c = torch.where(has_c, lse_c, torch.zeros_like(lse_c))           # no NaN in the unused branch / backward
    log_lam = log_lam.float().clamp(max=math.log1p(-1e-6))
    log_1m = torch.log1p(-log_lam.exp())
    adj_c = torch.logaddexp(log_lam, log_1m + lse_c) - lse_c             # log(lam + (1-lam) P_c) - log P_c
    zero = torch.zeros_like(adj_c)
    adj_c = torch.where(mix, adj_c, zero)
    adj_n = torch.where(mix, log_1m, zero)
    out = lp + torch.where(close, adj_c.unsqueeze(1), adj_n.unsqueeze(1))
    return out.masked_fill(~allowed, -torch.inf)


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
        self.length_head: Optional[LengthHead] = None       # set by the action generator (opts.length_prior)
        self.merged_acting = True                           # sample() decodes with merged LoRA weights (enc.acting)

    # ---------------------------------------------------------------- prompts and programs
    def _clip_ids(self, ids: list, keep: int) -> tuple:
        """Middle-clip a token list to <= keep tokens (head 2/3, marker, tail). Returns (ids, dropped originals)."""
        if len(ids) <= keep:
            return ids, 0
        n_drop = len(ids) - keep
        for _ in range(3):                                  # the marker's own length depends on the number printed
            marker = self.enc.tok(f" ...[{n_drop} tokens omitted]... ", add_special_tokens=False)["input_ids"]
            content = max(keep - len(marker), 0)
            n_new = len(ids) - content
            if n_new == n_drop:
                break
            n_drop = n_new
        content = max(keep - len(marker), 0)
        h = (2 * content) // 3
        out = ids[:h] + marker + (ids[len(ids) - (content - h):] if content - h > 0 else [])
        return out, len(ids) - content

    def fit_pieces(self, pieces: list, budget: Optional[int] = None) -> tuple:
        """pieces: [(text, priority)] (priority 0 = never shortened; 1 = evidence, shortened first; 2 = description;
        3 = arguments schema, shortened last) -> (ids, truncated original tokens). Each piece is tokenised on its own
        and the over-budget group of the lowest priority is shortened by water-filling: its longest pieces are
        middle-clipped (with an explicit "[n tokens omitted]" marker) to a common cap of >= 24 tokens."""
        budget = self.max_prompt_tokens if budget is None else budget
        toks = [self.enc.tok(t, add_special_tokens=False)["input_ids"] if t else [] for t, _ in pieces]
        prio = [p for _, p in pieces]
        truncated = 0
        for level in (1, 2, 3):
            excess = sum(len(t) for t in toks) - budget
            if excess <= 0:
                break
            idx = [i for i, p in enumerate(prio) if p == level and toks[i]]
            if not idx:
                continue
            lens = [len(toks[i]) for i in idx]
            target = max(sum(lens) - excess, 24 * len(idx))
            lo_c, hi_c = 24, max(lens)                         # largest cap c with sum(min(len, c)) <= target
            while lo_c < hi_c:
                mid = (lo_c + hi_c + 1) // 2
                if sum(min(l, mid) for l in lens) <= target:
                    lo_c = mid
                else:
                    hi_c = mid - 1
            for i in idx:
                if len(toks[i]) > lo_c:
                    toks[i], dropped = self._clip_ids(toks[i], lo_c)
                    truncated += dropped
        ids = [t for tk in toks for t in tk]
        if len(ids) > budget:                                    # only the never-shortened parts remain too long
            ids, dropped = self._clip_ids(ids, budget)
            truncated += dropped
        return ids, truncated

    def context_ids(self, prompt) -> tuple:
        """prompt: str or [(text, priority)] -> (chat-formatted ids, truncated prompt tokens)."""
        pieces = [(prompt, 3)] if isinstance(prompt, str) else list(prompt)
        ids, truncated = self.fit_pieces(pieces)
        head = self.enc.tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        tail = self.enc.tok("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        return head + ids + tail, truncated

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
            # string of at most max_len characters never needs more than 2*max_len + 1 tokens (incl. the quote);
            # free strings carry their schema-derived cap (string_class / string_token_cap)
            if h.kind == "pattern":
                cap.append(min(self.opts.max_string_tokens, h.max_len + 1))
            elif h.kind == "key":
                cap.append(min(self.opts.max_key_tokens, 2 * h.max_len + 1))
            elif h.node.tok_cap > 0:
                cap.append(h.node.tok_cap)
            else:
                cap.append(min(self.opts.max_string_tokens, 2 * h.max_len + 1))
        return Program(forced, holes, reserve, cap, [STRING_CLASSES.index(hole_class(h)) for h in holes])

    # ---------------------------------------------------------------- masks
    def _pattern_masks(self, h: Hole) -> PatternMasks:
        pm = self._pmask.get(id(h.dfa))
        if pm is None:
            pm = PatternMasks(h.dfa, self.tables)
            self._pmask[id(h.dfa)] = pm
        return pm

    def masks(self, specs: list) -> tuple:
        """specs: list of (hole, state, count, remaining) -> (allowed-token masks [P, V], closing-token masks [P, V]:
        the allowed tokens that end the string, i.e. carry its closing quote)."""
        T = self.tables
        P = len(specs)
        out = torch.zeros(P, T.V, dtype=torch.bool, device=T.valid.device)
        close = torch.zeros(P, T.V, dtype=torch.bool, device=T.valid.device)
        srows = [i for i, sp in enumerate(specs) if sp[0].kind != "pattern"]
        for i, sp in enumerate(specs):
            if sp[0].kind == "pattern":
                out[i] = self._pattern_masks(sp[0]).mask(sp[1], sp[2], sp[3], sp[0].min_len, sp[0].max_len)
                close[i, T.quote_id] = out[i, T.quote_id]
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
            rt = torch.tensor(rows, device=dev)
            out[rt] = close_ok | (v & ~cl & (cont0 | cont1))
            close[rt] = close_ok
        return out, close

    def hazard(self, programs: list, items: list) -> Optional[torch.Tensor]:
        """Log stop hazards [m] of (row, hole index, tokens so far) items (None when no program carries a length prior)."""
        head = self.length_head
        if head is None or not items or all(programs[r].hz_off is None for r, _, _ in items):
            return None
        dev = self.tables.valid.device
        offs = [programs[r].hz_off[k] if programs[r].hz_off is not None else None for r, k, _ in items]
        zero = torch.zeros((), device=dev)
        off = torch.stack([o.float() if o is not None else zero for o in offs])
        cls = torch.tensor([programs[r].cls[k] for r, k, _ in items], dtype=torch.long, device=dev)
        n = torch.tensor([float(u) for _, _, u in items], device=dev)
        hz = head.log_hazard(off, cls, n)
        none = [o is None for o in offs]
        if any(none):                                      # rows without a prior: no stop mixing (lam = e^-1e4)
            hz = torch.where(torch.tensor(none, device=dev), torch.full_like(hz, -1e4), hz)
        return hz

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
        n_steps). Decodes on the acting path (`enc.acting`: merged LoRA weights when `merged_acting`); the returned
        logp is the incremental diagnostic value - the behaviour log-probability is the learner-path `score`. One
        device->host transfer per step (the sampled tokens and the empty-mask flag together)."""
        with self.enc.acting(self.merged_acting):
            return self._sample(programs, prefix, ctx, temperature, greedy, generator)

    def _sample(self, programs, prefix, ctx, temperature, greedy, generator):
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
                m, mc = self.masks(specs)
                empty = ~m.any(1)
                hz = self.hazard(programs, [(r, st[r]["hi"], st[r]["used"]) for r in rows])
                lp = stop_mixture_log_softmax(logits[rows], m | empty.unsqueeze(1), mc, temperature, hz)
                if greedy:
                    tok = lp.argmax(-1)
                else:
                    tok = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
                chosen = lp.gather(1, tok.unsqueeze(1)).squeeze(1)
                e = masked_entropy(lp, m)
                rr = torch.tensor(rows, device=dev)
                logp.index_add_(0, rr, chosen)
                ent.index_add_(0, rr, e)
                host = torch.cat([tok, empty.any().long().view(1)]).tolist()      # the step's only sync
                if host[-1]:
                    raise GrammarError("empty token mask during hole sampling")
                tok_l = host[:-1]
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

        exact_rows=True evaluates every row alone without padding. In bf16 a padded batched forward depends on the
        batch's padding (measured on Qwen3-0.6B free strings: median ~0.19, max ~0.4 nats per action vs row-wise), so
        row-wise scoring is what makes a row's value independent of the minibatch. Per-row sums are plain
        reductions (no atomic index_add), so the same row gives bit-identical values in every call."""
        if exact_rows and len(programs) > 1:
            outs = []
            for r in range(len(programs)):
                tau = temperature.reshape(-1)[r: r + 1] if isinstance(temperature, torch.Tensor) else temperature
                outs.append(self.score(programs[r: r + 1], prefix[r: r + 1], ctx[r: r + 1], tau, chunk))
            return tuple(torch.cat([o[k] for o in outs]) for k in range(3))
        enc = self.enc
        R = len(programs)
        dev = enc.device
        tails, positions, targets, specs, owner, items = [], [], [], [], [], []
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
                    items.append((r, k, u))
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
        hz_all = self.hazard(programs, items)
        for c0 in range(0, len(targets), chunk):
            sl = slice(c0, c0 + chunk)
            hsel = hid[own[sl], abs_pos[sl]]
            lg = enc.lm_logits(hsel)
            m, mc = self.masks(specs[sl])
            bad = ~m.gather(1, tgt[sl].unsqueeze(1)).squeeze(1)
            if bool(bad.any()):
                raise GrammarError("recorded token outside the replayed constraint mask")
            lp = stop_mixture_log_softmax(lg, m, mc, tau_all[sl] if tau_all is not None else temperature,
                                          None if hz_all is None else hz_all[sl])
            lps.append(lp.gather(1, tgt[sl].unsqueeze(1)).squeeze(1))
            ents.append(masked_entropy(lp, m))
        # per-row sums over contiguous blocks (owners are in row order): a plain reduction is deterministic, unlike
        # index_add's atomic float adds, so the same row always gets bit-identical values
        counts = [0] * R
        for o in owner:
            counts[o] += 1
        zero = torch.zeros((), device=dev, dtype=torch.float32)
        if targets:
            lp_all, en_all = torch.cat(lps), torch.cat(ents)
            logp = torch.stack([c.sum() if c.numel() else zero for c in torch.split(lp_all, counts)])
            ent = torch.stack([c.sum() if c.numel() else zero for c in torch.split(en_all, counts)])
        else:
            logp = torch.zeros(R, device=dev, dtype=torch.float32)
            ent = torch.zeros(R, device=dev, dtype=torch.float32)
        n_tok = torch.tensor(counts, dtype=torch.long, device=dev)
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
