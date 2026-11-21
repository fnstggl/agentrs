"""Featurisation shared by the controller's inference and its training batches.  [candidate-editable]

Everything the controller scores - context tokens, action heads, argument options - is an *item* with the same
four fields, so that tools, entities and documents the controller has never seen need no new output units:

    ek    int64      key of a text embedded by the frozen language model (0 = none): tool descriptions, the goal,
                     documents, error feedback
    tok   int16[10]  hashed symbolic tokens (tool name parts, argument name, option source, identifier kind, ...)
    num   float16[10] small numeric features (recency rank, magnitudes, flags)
    typ   int8       item type (goal, history step, document, tool, wait, argument option, ...)

`build_slots` turns a tool's JSON schema into argument slots whose options come from what the actor knows
(identifiers by key, numbers it has seen and simple multiples of them), from the schema (enum values, booleans, a
log-spaced grid), from the current draft program, from what worked before, and - for free text, code and
structured values - from the frozen language model ("gen" options, one per focus document and variant). Nothing
in the option set names a strategy.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

N_BUCKETS = 32768
T_TOK = 10
N_NUM = 10
OBS_DIM = 512
MAX_SLOTS = 8
MAX_OPTS = 24

# item types
C_GOAL, C_STATE, C_HIST, C_ERR, C_DOC, C_DRAFT, C_MEM, C_PLAN, C_LIB = range(9)
H_TOOL, H_WAIT, H_PROPOSE, H_DRAFT_RUN, H_DRAFT_SKIP, H_DEFINE, H_USE = range(10, 17)
O_OPT, P_SLOT = 20, 21
N_TYPES = 24

INT_GRID = (0, 1, 2, 3, 5, 10, 20, 50, 100, 1_000, 10_000, 100_000)
NUM_GRID = (0.0, 0.1, 0.5, 0.8, 1.0, 2.0, 10.0)
WAITS = (600, 3600, 6 * 3600, 86400, 3 * 86400)
TEXT_KEYS = {"text", "body", "description", "memo", "note", "message", "title", "bio", "q", "comment",
             "reason", "name", "content", "subject", "purpose"}
OMIT = "<omit>"
WHOLE = "*"       # pseudo-argument: the whole argument object of an endpoint that declares no arguments


@functools.lru_cache(maxsize=200_000)
def bucket(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big") % (N_BUCKETS - 1) + 1


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(float(x))), x)


class Items:
    """A list of items in array form; `meta[i]` is the Python object item i stands for."""

    def __init__(self):
        self.ek: list[int] = []
        self.tok: list[list[int]] = []
        self.num: list[list[float]] = []
        self.typ: list[int] = []
        self.meta: list[Any] = []

    def add(self, typ: int, tokens: list[str], num: list[float] | None = None, ek: int = 0, meta: Any = None) -> int:
        t = [bucket(x) for x in tokens[:T_TOK]]
        self.tok.append(t + [0] * (T_TOK - len(t)))
        n = list(num or [])[:N_NUM]
        self.num.append(n + [0.0] * (N_NUM - len(n)))
        self.ek.append(int(ek))
        self.typ.append(int(typ))
        self.meta.append(meta)
        return len(self.ek) - 1

    def __len__(self) -> int:
        return len(self.ek)

    def arrays(self) -> dict:
        n = len(self.ek)
        return {"ek": np.asarray(self.ek, np.int64).reshape(n),
                "tok": np.asarray(self.tok, np.int16).reshape(n, T_TOK),
                "num": np.clip(np.asarray(self.num, np.float32).reshape(n, N_NUM), -30, 30).astype(np.float16),
                "typ": np.asarray(self.typ, np.int8).reshape(n)}


def tool_tokens(name: str, spec: dict | None = None) -> list[str]:
    host, _, ep = name.partition(".")
    ann = (spec or {}).get("annotations") or {}
    toks = [f"T:{name}", f"H:{host}", f"E:{ep}", "RO" if ann.get("readOnlyHint") else "RW",
            "B" if ann.get("builtin", True) else "D"]
    toks += [f"W:{w}" for w in ep.split("_") if w][: T_TOK - len(toks)]
    return toks


# ================================================================================================ argument slots
@dataclass
class Opt:
    label: str
    value: Any
    source: str               # omit | key | recent | self | enum | bool | grid | seen | rel | fresh | text | gen | draft | prev
    rank: int = 0
    doc: Any = None           # focus document for "gen"
    variant: int = 0
    nkey: str = ""            # for numbers: the key the base number was seen under
    xform: float = 1.0
    mine: bool = False


@dataclass
class Slot:
    param: str
    kind: str                 # enum | bool | int | num | id | handle | text | list | object
    required: bool
    schema: dict
    options: list = field(default_factory=list)


@dataclass
class SlotContext:
    actor: str
    know: Any                                 # grounding.knowledge.Knowledge
    mine: set = field(default_factory=set)    # identifiers the actor created or controls
    draft_args: dict | None = None            # arguments the current draft suggests for this tool
    prev_args: dict | None = None             # arguments of the last successful call of this tool (memory)
    docs: list = field(default_factory=list)  # focus documents, most relevant first
    llm: bool = True
    fresh_n: int = 0
    gen_variants: int = 2


def _ptype(p: dict) -> str:
    t = p.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    return t or "any"


def _jkey(v: Any) -> str:
    try:
        return json.dumps(v, sort_keys=True, default=str)[:300]
    except (TypeError, ValueError):
        return repr(v)[:300]


def _id_opts(param: str, p: dict, sc: SlotContext) -> list[Opt]:
    ents = sc.know.ents
    out = [Opt(v, v, "key", i, mine=v in sc.mine) for i, (v, _) in enumerate(ents.ids_for(param, 6))]
    seen = {o.value for o in out}
    out += [Opt(v, v, "recent", i, mine=v in sc.mine)
            for i, (v, _) in enumerate(ents.recent_ids(4 if len(out) >= 2 else 6, exclude=seen))]
    if sc.actor not in {o.value for o in out}:
        out.append(Opt(sc.actor, sc.actor, "self", 0, mine=True))
    mx, pat = p.get("maxLength", 4000), p.get("pattern")
    return [o for o in out if len(o.value) <= mx and (not pat or re.fullmatch(pat, o.value))]


def _num_opts(param: str, p: dict, sc: SlotContext, integer: bool) -> list[Opt]:
    lo, hi = p.get("minimum", -math.inf), p.get("maximum", math.inf)
    ents = sc.know.ents
    out: list[Opt] = []

    def put(v, source, rank=0, nkey="", xform=1.0):
        v = int(round(v)) if integer else float(v)
        if lo <= v <= hi:
            out.append(Opt(str(v), v, source, rank, nkey=nkey, xform=xform))
    for i, v in enumerate(ents.numbers_for(param, 3)):
        for x in (1.0, 0.9, 1.5):
            put(v * x, "seen", i, param, x)
    recent = sorted(((d[v], k, v) for k, d in ents.numbers.items() for v in d if abs(v) >= 10), reverse=True)
    used: set = set()
    for _, k, v in recent:
        if v in used or len(used) >= 4:
            continue
        used.add(v)
        for x in (1.0, 0.7, 0.3):
            put(v * x, "rel", len(used) - 1, k, x)
    for i, v in enumerate(INT_GRID if integer else NUM_GRID):
        put(v, "grid", i)
    return out


def _gen_opts(sc: SlotContext, with_docs: bool) -> list[Opt]:
    if not sc.llm:
        return []
    if not with_docs or not sc.docs:
        return [Opt(f"<gen:{v}>", None, "gen", 0, None, v) for v in range(1)]
    out = []
    for r, d in enumerate(sc.docs[:5]):
        for v in range(sc.gen_variants if r < 2 else 1):
            out.append(Opt(f"<gen:{d.title[:40]}:{v}>", None, "gen", r, d, v))
    return out


def slot_for(tool: str, param: str, p: dict, required: bool, sc: SlotContext) -> Slot:
    t = _ptype(p)
    opts: list[Opt] = [] if required else [Opt(OMIT, OMIT, "omit")]
    kind = "object"
    if "enum" in p:
        kind = "enum"
        opts += [Opt(str(v), v, "enum", i) for i, v in enumerate(p["enum"])]
    elif t == "boolean":
        kind = "bool"
        opts += [Opt("true", True, "bool"), Opt("false", False, "bool", 1)]
    elif t in ("integer", "number"):
        kind = "int" if t == "integer" else "num"
        opts += _num_opts(param, p, sc, t == "integer")
    elif t == "string":
        pat = p.get("pattern")
        if pat:
            kind = "handle"
            opts += [o for o in _id_opts(param, p, sc) if o.source in ("key", "self")]
            base = re.sub(r"[^a-z0-9]", "", sc.actor.split(":", 1)[-1].lower())[:10] or "a"
            for j in range(2):
                h = f"{base}-{re.sub(r'[^a-z]', '', param.lower())[:4] or 'x'}{sc.fresh_n + j + 1}"
                if re.fullmatch(pat, h) and len(h) <= p.get("maxLength", 4000):
                    opts.append(Opt(h, h, "fresh", j))
        elif p.get("maxLength", 4000) > 200 or param in TEXT_KEYS:
            kind = "text"
            opts += _gen_opts(sc, with_docs=True)[:4]
            opts += [o for o in _id_opts(param, p, sc) if o.source == "key"][:2]
            txt = f"{param} {sc.actor.split(':', 1)[-1]}"[: p.get("maxLength", 4000)]
            opts.append(Opt(txt, txt, "text"))
        else:
            kind = "id"
            opts += _id_opts(param, p, sc)
    elif t == "array" and _ptype(p.get("items") or {}) == "string":
        kind = "list"
        base = slot_for(tool, param, dict(p.get("items") or {}, type="string"), True, sc)
        opts += [Opt(f"[{o.label}]", [o.value], o.source, o.rank, mine=o.mine) for o in base.options
                 if o.source not in ("gen",)][:8]
        opts += _gen_opts(sc, with_docs=True)[:3]
    else:
        opts += _gen_opts(sc, with_docs=True)
    extra = []
    if sc.draft_args and param in sc.draft_args:
        extra.append(Opt(_jkey(sc.draft_args[param])[:60], sc.draft_args[param], "draft"))
    if sc.prev_args and param in sc.prev_args:
        extra.append(Opt(_jkey(sc.prev_args[param])[:60], sc.prev_args[param], "prev"))
    seen: dict[str, Opt] = {}
    merged: list[Opt] = []
    for o in extra + opts:
        k = _jkey(o.value) if o.source not in ("gen", "omit") else f"{o.source}:{o.label}"
        if k in seen:
            continue
        seen[k] = o
        merged.append(o)
    return Slot(param, kind, required, p, merged[:MAX_OPTS])


def build_slots(tool: str, schema: dict, sc: SlotContext) -> list[Slot] | None:
    """Slots of a tool call (required arguments first). None if a required argument has no option at all."""
    props = (schema or {}).get("properties") or {}
    if not props and (schema or {}).get("additionalProperties"):
        s = slot_for(tool, WHOLE, {"type": "object", "additionalProperties": True}, False, sc)
        return [s] if s.options else []
    req = [k for k in (schema or {}).get("required") or [] if k in props]
    order = req + [k for k in props if k not in req]
    out: list[Slot] = []
    for name in order:
        is_req = name in req
        if len(out) >= MAX_SLOTS:
            if is_req:
                return None
            break
        s = slot_for(tool, name, props[name], is_req, sc)
        if not s.options:
            if is_req:
                return None
            continue
        out.append(s)
    return out


def slot_items(tool: str, slot: Slot, index: int) -> tuple[Items, Items]:
    """(parameter item, option items) for one slot."""
    par = Items()
    par.add(P_SLOT, [f"P:{slot.param}", f"PT:{tool}.{slot.param}", f"K:{slot.kind}", "REQ" if slot.required else "OPT"],
            [index / MAX_SLOTS, float(slot.required), len(slot.options) / MAX_OPTS])
    it = Items()
    for o in slot.options:
        v = o.value
        isnum = isinstance(v, (int, float)) and not isinstance(v, bool)
        toks = [f"P:{slot.param}", f"PT:{tool}.{slot.param}", f"K:{slot.kind}", f"SRC:{o.source}"]
        if o.source in ("enum", "bool", "grid", "fresh", "text"):
            toks.append(f"V:{slot.param}={o.label}"[:80])
            toks.append(f"TV:{tool}.{slot.param}={o.label}"[:100])
        if isinstance(v, str) and o.source in ("key", "recent", "self", "draft", "prev"):
            from ..grounding.knowledge import alias_kind
            toks.append(f"IK:{alias_kind(v) or 'word'}")
            toks.append(f"PIK:{slot.param}:{alias_kind(v) or 'word'}")
        if o.nkey:
            toks.append(f"NK:{o.nkey}")
            toks.append(f"NKX:{o.nkey}:{o.xform}")
            toks.append(f"PNK:{slot.param}:{o.nkey}:{o.xform}")
        if o.source == "gen":
            toks.append(f"GV:{o.variant}")
            toks.append(f"GR:{min(o.rank, 4)}")
        num = [min(o.rank, 8) / 8.0, float(o.source == "omit"), symlog(v) / 12.0 if isnum else 0.0,
               float(o.source in ("key", "seen")), float(o.source == "draft"), float(o.source == "prev"),
               float(o.source == "gen"), float(o.variant), float(o.mine), o.xform if o.nkey else 0.0]
        it.add(O_OPT, toks, num, ek=(o.doc.key if o.doc is not None else 0), meta=o)
    return par, it
