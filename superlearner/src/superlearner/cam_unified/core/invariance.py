"""Invariant representations across surface presentation (workstream B; requirement W02, spec 13.3).

Three parts:

1. `SurfaceRandomizer` builds training pairs: from the records of one experience (observations, tools, actions, goal)
   it produces a dynamically equivalent view with a different surface (spec 13.3's list):
   * entity identities: consistent bijection to fresh ids (Crucible-style prefixes such as "p:" / "o:" kept), applied
     wherever the id occurs as a whole token — text, JSON keys/values, paths, URLs, action arguments — so anything
     an agent copies from the surface view maps back exactly (`SurfaceMap.invert_action`);
   * tool names: consistent bijection (tool schemas, observation sources, actions);
   * units: NUMERIC observations rescaled with matching unit strings (e.g. 12_345 "cents" -> 123.45 "dollars",
     2000 "ms" -> 2 "s", 0.25 "fraction" -> 25 "percent"; unknown units -> "milli-<unit>" x1000), and
     "<number> <unit>" phrases in natural-language text; action arguments keep the environment's units;
   * observation (and tool) ordering: permuted;
   * wording: case-preserving, meaning-preserving synonym substitution in natural-language spans only. Text is split
     into whitespace-delimited chunks; identifier-like chunks are never touched — anything with _ / \\ @ # = ? & < > { }
     [ ] | $ ^ ~ ` *, dotted or colon names (main.py, p:nova, https://...), letter-digit mixes (v2, prod1),
     camelCase, CLI flags, quoted single tokens ('price', "build_ok":) and renamed ids. Kinds: TEXT / MESSAGE /
     EVENT / CONSTRAINT / GOAL text / tool descriptions are prose; TOOL_RESULT is reworded only inside JSON string
     values that are prose (keys and identifier values never); EXEC_RESULT / ERROR / HTML / A11Y / CODE / FILE are
     machine output and get identity renames only (no rewording, no unit rewriting);
   * interface presentation: JSON tool results are re-rendered in another presentation with identical content —
     compact / pretty / key-sorted JSON, "path: value" lines, or a markdown table for lists of records.

2. `InvarianceLoss`: projected, normalised pooled latent states of the two views; NT-Xent with 2B-2 candidates per
   anchor where the positive is the other view of the same experience and negatives are views of experiences from
   **different worlds** only — other experiences of the same world (same mechanism, possibly similar dynamics) are
   excluded rather than pushed apart, and distinct mechanisms are never pulled together.

3. `PredictionEquivarianceLoss`: predictions of the same outcome queries in the two views must transform by the
   view's unit map — Gaussian predictive N(m, v) in view 1 must equal N(f m, f^2 v) in view 2 (equivariance;
   f = 1 gives invariance) — and categorical / Bernoulli predictions must be identical (invariance).
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import Action, GoalSpec, Observation, ObsKind, TokenSet, ToolSchema
from .blocks import RMSNorm, count_parameters

# ---------------------------------------------------------------------------- surface randomisation

#: unit -> list of (new unit, factor): value_new = value * factor
UNIT_CONVERSIONS = {
    "cents": [("dollars", 0.01), ("millicents", 1000.0)],
    "dollars": [("cents", 100.0)], "usd": [("cents", 100.0)], "USD": [("cents", 100.0)],
    "ms": [("s", 1e-3), ("us", 1e3)], "s": [("ms", 1e3), ("min", 1 / 60)], "seconds": [("minutes", 1 / 60)],
    "min": [("s", 60.0)], "minutes": [("seconds", 60.0)], "h": [("min", 60.0)], "hours": [("minutes", 60.0)],
    "days": [("hours", 24.0)], "fraction": [("percent", 100.0)], "percent": [("fraction", 0.01)],
    "%": [("fraction", 0.01)], "bytes": [("KB", 1e-3)], "KB": [("bytes", 1e3)], "MB": [("KB", 1e3)],
    "GB": [("MB", 1e3)],
}
UNCHANGED_UNITS = {"count", "units", "items", "", None}

#: meaning-preserving substitutions for natural-language spans. Words whose synonyms change meaning in some context
#: are deliberately absent (e.g. by, did, can, long, your, set, return, cost, account, program, review, network).
SYNONYMS = {
    "customers": ["clients", "buyers"], "customer": ["client", "buyer"], "complain": ["grumble", "object"],
    "takes": ["requires", "needs"], "too": ["overly", "excessively"], "deliver": ["ship", "supply"],
    "units": ["pieces", "items"], "search": ["look up"], "simulated": ["virtual", "emulated"], "internet": ["web"],
    "pages": ["documents", "web pages"], "matching": ["fitting", "satisfying"], "price": ["charge", "list price"],
    "monthly": ["per-month"], "products": ["offerings", "goods"], "product": ["offering", "item"],
    "execute": ["run"], "send": ["transmit", "dispatch"], "message": ["note"], "participant": ["party"],
    "hire": ["recruit", "employ"], "workers": ["staff", "employees"], "error": ["failure", "fault"],
    "respond": ["answer", "reply"], "revenue": ["sales", "turnover"], "increase": ["raise", "grow"],
    "decrease": ["lower", "reduce"], "upstream": ["backend"], "segment": ["group", "cohort"],
    "onboarding": ["setup", "sign-up"], "balance": ["funds"], "deploy": ["release"], "request": ["call"],
    "rating": ["score", "grade"], "one": ["a single"], "another": ["a different"], "quickly": ["fast", "rapidly"],
    "buy": ["purchase"], "begin": ["start"], "help": ["assist"], "show": ["display"], "big": ["large"],
    "small": ["little"],
}

#: observation kinds whose content is natural language
PROSE_KINDS = {ObsKind.TEXT, ObsKind.MESSAGE, ObsKind.EVENT, ObsKind.CONSTRAINT, ObsKind.GOAL, ObsKind.TOOL_SCHEMA}
#: machine output: identity renames only
MACHINE_KINDS = {ObsKind.EXEC_RESULT, ObsKind.ERROR, ObsKind.HTML, ObsKind.A11Y, ObsKind.CODE, ObsKind.FILE,
                 ObsKind.IMAGE}
PRESENTATIONS = ("json_compact", "json_pretty", "json_sorted", "kv_lines", "table")

_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_CHUNK = re.compile(r"\S+")
_LEAD = "(\"'«“‘"
_TRAIL = ".,;:!?)\"'»”’"
_ID_CHARS = re.compile(r"[_/\\@#=?&<>{}\[\]|$^~`*]")
_ID_SHAPES = re.compile(r"[A-Za-z][.:][A-Za-z0-9]|[0-9][.:][A-Za-z]|[A-Za-z][0-9]|[0-9][A-Za-z]|[a-z][A-Z]|^-+[A-Za-z]")


def _case_like(src: str, word: str) -> str:
    if src.isupper() and len(src) > 1:
        return word.upper()
    if src[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def is_identifier_chunk(chunk: str, protected: frozenset = frozenset()) -> bool:
    """Whether a whitespace-delimited chunk is (or contains) an identifier, path, URL, literal or quoted token that a
    surface transformation must never reword (see module docstring)."""
    core = chunk.strip(_LEAD + _TRAIL)
    if not core:
        return False
    if core in protected or _ID_CHARS.search(core) or _ID_SHAPES.search(core):
        return True
    quoted = chunk.rstrip(".,;:)]}").lstrip("([{")
    if len(quoted) >= 2 and quoted[0] in "\"'`" and quoted[-1] == quoted[0] and " " not in quoted:
        return True                                     # a quoted single token: a literal / key, not prose
    return False


def map_prose(text: str, fn, protected: frozenset = frozenset()) -> str:
    """Apply fn to the maximal natural-language spans of `text` (runs of non-identifier chunks with their
    whitespace); identifier chunks and the whitespace layout are kept verbatim."""
    if not text:
        return text
    out, last, span_start = [], 0, None
    for m in _CHUNK.finditer(text):
        ident = is_identifier_chunk(m.group(0), protected)
        if ident:
            if span_start is not None:
                out.append(fn(text[span_start:m.start()]))
                span_start = None
            else:
                out.append(text[last:m.start()])
            out.append(m.group(0))
            last = m.end()
        elif span_start is None:
            out.append(text[last:m.start()])
            span_start = m.start()
    if span_start is not None:
        out.append(fn(text[span_start:]))
    else:
        out.append(text[last:])
    return "".join(out)


def _parse_json(content: str):
    st = content.strip()
    if not st or st[0] not in "{[":
        return None
    try:
        return json.loads(st)
    except (ValueError, TypeError):
        return None


def _is_prose_value(v: str, protected: frozenset) -> bool:
    chunks = v.split()
    return len(chunks) >= 2 and sum(not is_identifier_chunk(c, protected) for c in chunks) >= 2


def _scalar_str(v) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v)


def _flatten(obj, prefix: str = "") -> list:
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out or [(prefix, "{}")]
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out += _flatten(v, f"{prefix}[{i}]")
        return out or [(prefix, "[]")]
    return [(prefix, _scalar_str(obj))]


def _records(obj):
    """(key path, list of flat records with identical keys) when obj is (or holds under one key) a table."""
    def ok(lst):
        return (isinstance(lst, list) and lst and all(isinstance(r, dict) for r in lst)
                and all(list(r) == list(lst[0]) for r in lst)
                and all(not isinstance(v, (dict, list)) for r in lst for v in r.values()))
    if ok(obj):
        return None, obj
    if isinstance(obj, dict):
        tabs = [k for k, v in obj.items() if ok(v)]
        if len(tabs) == 1:
            return tabs[0], obj[tabs[0]]
    return None


def render_json(obj, style: str) -> str:
    """Render a parsed JSON value in an interface presentation (content identical, layout differs)."""
    if style == "json_compact":
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if style == "json_pretty":
        return json.dumps(obj, indent=2, ensure_ascii=False)
    if style == "json_sorted":
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    if style == "table":
        rec = _records(obj)
        if rec is not None:
            key, rows = rec
            cols = list(rows[0])
            lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(_scalar_str(r[c]) for c in cols) + " |" for r in rows]
            rest = []
            if key is not None:
                rest = [f"{p}: {v}" for p, v in _flatten({k: v for k, v in obj.items() if k != key})]
                lines = [f"{key}:"] + lines
            return "\n".join(rest + lines)
        style = "kv_lines"
    if style == "kv_lines":
        return "\n".join(f"{p}: {v}" for p, v in _flatten(obj))
    raise ValueError(style)


@dataclass
class SurfaceMap:
    """A consistent surface transformation of one experience (and its inverse for actions)."""
    entity: dict = field(default_factory=dict)       # original id -> surface id
    tool: dict = field(default_factory=dict)         # original tool name -> surface name
    unit: dict = field(default_factory=dict)         # original unit -> (surface unit, factor)
    words: dict = field(default_factory=dict)        # lowercase word -> replacement
    order_seed: Optional[int] = None                 # None = keep order
    presentation: Optional[str] = None               # rendering of JSON tool results (None = unchanged)

    # ---- text
    def _protected(self) -> frozenset:
        return frozenset(self.entity.values()) | frozenset(self.tool.values())

    def _replace_ids(self, text: str) -> str:
        return _replace_with(text, self._fwd())

    def _units_in_text(self, text: str) -> str:
        units = [u for u in self.unit if u]
        if not units or not text:
            return text
        pat = re.compile(r"(?<![\w.:/=-])(-?\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
                         + r")(?![\w/])")

        def sub(m):
            nu, f = self.unit[m.group(2)]
            return f"{_fmt(float(m.group(1)) * f)} {nu}"
        return pat.sub(sub, text)

    def _reword_span(self, text: str) -> str:
        if not self.words:
            return text

        def sub(m):
            w = m.group(0)
            r = self.words.get(w.lower())
            return _case_like(w, r) if r else w
        return _WORD.sub(sub, text)

    def _reword(self, text: str) -> str:
        if not self.words or not text:
            return text
        return map_prose(text, self._reword_span, self._protected())

    def text(self, s: str) -> str:
        """Natural-language text: identity renames everywhere, then unit phrases and rewording in prose spans."""
        s = self._replace_ids(s)
        return map_prose(s, lambda span: self._reword_span(self._units_in_text(span)), self._protected())

    def _json_value(self, v):
        if isinstance(v, dict):
            return {self._replace_ids(str(k)): self._json_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._json_value(x) for x in v]
        if isinstance(v, str):
            r = self._replace_ids(v)
            return self._reword(r) if _is_prose_value(r, self._protected()) else r
        return v

    def structured(self, content: str) -> str:
        """Tool-result content: JSON is parsed — keys and identifier values only get identity renames, prose string
        values are reworded — and re-rendered in the map's presentation; non-JSON gets identity renames only."""
        obj = _parse_json(content)
        if obj is None:
            return self._replace_ids(content)
        obj = self._json_value(obj)
        if self.presentation is None:
            return json.dumps(obj, ensure_ascii=False)
        return render_json(obj, self.presentation)

    # ---- records
    def apply_observation(self, o: Observation) -> Observation:
        o = copy.deepcopy(o)
        if o.kind in PROSE_KINDS:
            o.content = self.text(o.content)
        elif o.kind == ObsKind.TOOL_RESULT:
            o.content = self.structured(o.content)
        else:                                                  # machine output, numeric, relation, transaction
            o.content = self._replace_ids(o.content)
        o.entity = self.entity.get(o.entity, o.entity)
        o.target = self.entity.get(o.target, o.target)
        if o.source in self.tool:
            o.source = self.tool[o.source]
        if o.relation:
            o.relation = self._reword(self._replace_ids(o.relation))
        if o.kind == ObsKind.NUMERIC and o.value is not None and o.unit in self.unit:
            nu, f = self.unit[o.unit]
            o.value, o.unit = o.value * f, nu
        return o

    def apply_observations(self, obs: Sequence[Observation]) -> list:
        out = [self.apply_observation(o) for o in obs]
        if self.order_seed is not None:
            random.Random(self.order_seed).shuffle(out)
        return out

    def apply_tool(self, t: ToolSchema) -> ToolSchema:
        t = copy.deepcopy(t)
        t.name = self.tool.get(t.name, t.name)
        t.description = self.text(t.description)
        return t

    def apply_tools(self, tools: Sequence[ToolSchema]) -> list:
        out = [self.apply_tool(t) for t in tools]
        if self.order_seed is not None:
            random.Random(self.order_seed + 1).shuffle(out)
        return out

    def _map_args(self, v, names: dict):
        if isinstance(v, str):
            return names[v] if v in names else _replace_with(v, names)
        if isinstance(v, list):
            return [self._map_args(x, names) for x in v]
        if isinstance(v, dict):
            return {k: self._map_args(x, names) for k, x in v.items()}
        return v

    def _fwd(self) -> dict:
        return {**self.entity, **self.tool}

    def apply_action(self, a: Action) -> Action:
        a = copy.deepcopy(a)
        names = self._fwd()
        a.tool = self.tool.get(a.tool, a.tool)
        a.args = self._map_args(a.args, names)
        a.entities = [self.entity.get(e, e) for e in a.entities]
        if a.code:
            a.code = _replace_with(a.code, names)
        return a

    def invert_action(self, a: Action) -> Action:
        a = copy.deepcopy(a)
        inv_e = {v: k for k, v in self.entity.items()}
        inv_t = {v: k for k, v in self.tool.items()}
        names = {**inv_e, **inv_t}
        a.tool = inv_t.get(a.tool, a.tool)
        a.args = self._map_args(a.args, names)
        a.entities = [inv_e.get(e, e) for e in a.entities]
        if a.code:
            a.code = _replace_with(a.code, names)
        return a

    def apply_goal(self, g: GoalSpec) -> GoalSpec:
        g = copy.deepcopy(g)
        g.text = self.text(g.text)
        return g

    def unit_factor(self, unit: Optional[str]) -> tuple:
        """(surface unit, factor) for an outcome / observation unit (identity for units the map leaves alone)."""
        return self.unit.get(unit, (unit, 1.0))


def _replace_with(text: str, names: dict) -> str:
    if not names or not text:
        return text
    keys = sorted(names, key=len, reverse=True)
    pat = re.compile(r"(?<![\w:])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")
    return pat.sub(lambda m: names[m.group(1)], text)


def _fmt(x: float) -> str:
    """Plain decimal formatting of a converted value (integers without a fraction, no exponent when avoidable)."""
    if math.isfinite(x) and x == int(x) and abs(x) < 1e15:
        return str(int(x))
    if 1e-4 <= abs(x) < 1e15:
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return f"{x:.6g}"


_NAME_WORDS = ["alder", "birch", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "iris", "juniper", "kestrel",
               "lumen", "maple", "nimbus", "onyx", "pylon", "quartz", "raven", "sable", "tundra", "umber", "vale",
               "willow", "xenon", "yarrow", "zephyr"]
_TOOL_VERBS = ["do", "run", "make", "use", "call", "apply", "issue", "perform"]


class SurfaceRandomizer:
    """Draws SurfaceMaps. Each switch enables one surface factor (spec 13.3)."""

    def __init__(self, seed: int = 0, rename_entities: bool = True, rename_tools: bool = True,
                 rescale_units: bool = True, permute: bool = True, reword: bool = True, reword_prob: float = 0.7,
                 presentation: bool = True):
        self.rng = random.Random(seed)
        self.rename_entities, self.rename_tools = rename_entities, rename_tools
        self.rescale_units, self.permute, self.reword = rescale_units, permute, reword
        self.reword_prob = reword_prob
        self.presentation = presentation

    def _new_id(self, old: str, taken: set) -> str:
        prefix = ""
        m = re.match(r"^([a-z]{1,4}:)", old)
        if m:
            prefix = m.group(1)
        while True:
            name = f"{prefix}{self.rng.choice(_NAME_WORDS)}_{self.rng.randrange(16 ** 4):04x}"
            if name not in taken:
                taken.add(name)
                return name

    def make_map(self, obs: Sequence[Observation] = (), tools: Sequence[ToolSchema] = (),
                 actions: Sequence[Action] = ()) -> SurfaceMap:
        sm = SurfaceMap()
        taken: set = set()
        if self.rename_entities:
            ids = []
            for o in obs:
                ids += [o.entity, o.target]
            for a in actions:
                ids += list(a.entities)
            for e in dict.fromkeys(i for i in ids if i):
                sm.entity[e] = self._new_id(e, taken)
        if self.rename_tools:
            names = [t.name for t in tools] + [a.tool for a in actions]
            for n in dict.fromkeys(names):
                if n in ("noop", "advance_time", "wait"):
                    continue
                sm.tool[n] = f"{self.rng.choice(_TOOL_VERBS)}_{self.rng.randrange(16 ** 6):06x}"
        if self.rescale_units:
            for o in obs:
                u = o.unit
                if o.kind != ObsKind.NUMERIC or u in UNCHANGED_UNITS or u in sm.unit:
                    continue
                sm.unit[u] = self._convert_unit(u)
        if self.permute:
            sm.order_seed = self.rng.randrange(2 ** 31)
        if self.reword:
            for w, alts in SYNONYMS.items():
                if self.rng.random() < self.reword_prob:
                    sm.words[w] = self.rng.choice(alts)
        if self.presentation:
            sm.presentation = self.rng.choice(PRESENTATIONS)
        return sm

    def _convert_unit(self, unit: str) -> tuple:
        if unit in UNIT_CONVERSIONS:
            return self.rng.choice(UNIT_CONVERSIONS[unit])
        if "/" in unit:                                          # convert the numerator of a rate
            num, den = unit.split("/", 1)
            if num in UNIT_CONVERSIONS:
                nu, f = self.rng.choice(UNIT_CONVERSIONS[num])
                return f"{nu}/{den}", f
        return (f"milli-{unit}", 1000.0) if self.rng.random() < 0.5 else (f"kilo-{unit}", 1e-3)


# ---------------------------------------------------------------------------- invariance objective

class InvarianceLoss(nn.Module):
    """forward(Z1 [B,S,D], Z2 [B,S,D], world [B] long) -> (loss scalar, stats dict).

    Pooled representation: concat(mean over state tokens, attention pooling with a learned query) -> MLP ->
    L2-normalised embedding [B,P]. NT-Xent over the 2B embeddings at temperature tau: the positive of view-1 row i is
    view-2 row i (and vice versa); candidates of a different world are negatives; other rows of the same world are
    masked out (neither positive nor negative)."""

    def __init__(self, d_model: int, proj_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature
        self.norm = RMSNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, proj_dim))

    def embed(self, Z: Tensor) -> Tensor:
        h = self.norm(Z.float())
        a = torch.softmax(h @ self.query / h.shape[-1] ** 0.5, dim=1)          # [B,S]
        pooled = torch.cat([h.mean(1), (a.unsqueeze(-1) * h).sum(1)], -1)
        return F.normalize(self.proj(pooled), dim=-1)

    def forward(self, Z1: Tensor, Z2: Tensor, world: Tensor) -> tuple:
        B = Z1.shape[0]
        e = torch.cat([self.embed(Z1), self.embed(Z2)], 0)                    # [2B,P]
        w = torch.cat([world, world], 0).to(e.device)
        sim = e @ e.t()                                                        # cosine similarities
        idx = torch.arange(2 * B, device=e.device)
        pos = torch.cat([idx[B:], idx[:B]])
        same_world = w.unsqueeze(0) == w.unsqueeze(1)
        excluded = same_world.clone()
        excluded[idx, pos] = False                                             # keep the positive
        logits = (sim / self.tau).masked_fill(excluded, float("-inf"))
        loss = F.cross_entropy(logits, pos)
        with torch.no_grad():
            neg = ~same_world
            other_same = same_world.clone()
            other_same[idx, idx] = False
            other_same[idx, pos] = False
            stats = {
                "pos_cos": sim[idx, pos].mean().item(),
                "neg_cos": (sim[neg].mean().item() if bool(neg.any()) else float("nan")),
                "same_world_cos": (sim[other_same].mean().item() if bool(other_same.any()) else float("nan")),
                "top1": (logits.argmax(1) == pos).float().mean().item(),
            }
        return loss, stats

    def count_parameters(self) -> int:
        return count_parameters(self)


class PredictionEquivarianceLoss(nn.Module):
    """Equivariance / invariance of outcome predictions across two surface views of one experience.

    gaussian(mean1, var1, mean2, var2, factor, mask) -> scalar: the view-2 predictive N(m2, v2) must equal the view-1
    predictive transformed by the view's unit map, N(f m1, f^2 v1) (f = 1 for unit-free outcomes: invariance).
    Loss = mean over masked queries of the symmetric KL between the two Gaussians (scale-free: KL is unchanged when
    both are rescaled, so cents and dollars outcomes weigh the same). All [B,Q]; factor [B,Q] or scalar.

    categorical(logits1, logits2, mask) -> scalar: symmetric KL between the two categorical predictions [B,Q,C]
    (Bernoulli: pass logits of shape [B,Q,2] or use `bernoulli`).

    `target` ("none" | "view1" | "view2") stops the gradient through one view so it acts as the teacher (e.g. the
    view whose outcomes are labelled); "none" trains both views symmetrically. `robust` uses log(1 + KL) per query
    (still zero iff the views agree; gradients bounded when they are far apart, e.g. early in training, so the
    term cannot swamp the prediction loss).
    """

    def __init__(self, min_var: float = 1e-8, target: str = "none", robust: bool = True):
        super().__init__()
        assert target in ("none", "view1", "view2"), target
        self.min_var = min_var
        self.target = target
        self.robust = robust

    def _reduce(self, sym: Tensor, mask: Optional[Tensor]) -> Tensor:
        if self.robust:
            sym = torch.log1p(sym)
        if mask is None:
            return sym.mean()
        m = mask.to(sym.dtype)
        return (sym * m).sum() / m.sum().clamp_min(1.0)

    def _sg(self, a: Tensor, b: Tensor) -> tuple:
        if self.target == "view1":
            return a.detach(), b
        if self.target == "view2":
            return a, b.detach()
        return a, b

    @staticmethod
    def _kl_gauss(m_a, v_a, m_b, v_b):
        return 0.5 * (torch.log(v_b / v_a) + (v_a + (m_a - m_b) ** 2) / v_b - 1.0)

    def gaussian(self, mean1: Tensor, var1: Tensor, mean2: Tensor, var2: Tensor, factor,
                 mask: Optional[Tensor] = None) -> Tensor:
        f = torch.as_tensor(factor, dtype=mean1.dtype, device=mean1.device)
        mean1, mean2 = self._sg(mean1, mean2)
        var1, var2 = self._sg(var1, var2)
        m1, v1 = f * mean1, (f * f) * var1.clamp_min(self.min_var)
        v2 = var2.clamp_min(self.min_var)
        sym = self._kl_gauss(m1, v1, mean2, v2) + self._kl_gauss(mean2, v2, m1, v1)
        return self._reduce(sym, mask)

    def categorical(self, logits1: Tensor, logits2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits1, logits2 = self._sg(logits1, logits2)
        lp1, lp2 = F.log_softmax(logits1.float(), -1), F.log_softmax(logits2.float(), -1)
        sym = ((lp1.exp() * (lp1 - lp2)).sum(-1) + (lp2.exp() * (lp2 - lp1)).sum(-1))
        return self._reduce(sym, mask)

    def bernoulli(self, logit1: Tensor, logit2: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z = torch.zeros_like(logit1)
        return self.categorical(torch.stack([z, logit1], -1), torch.stack([z, logit2], -1), mask)


def unit_factors(sm: SurfaceMap, units: Sequence[Sequence[Optional[str]]], device=None) -> Tensor:
    """Per-query rescaling factors [B,Q] for outcome queries with the given units (view 1 -> view 2 of `sm`)."""
    return torch.tensor([[sm.unit_factor(u)[1] for u in row] for row in units], dtype=torch.float32, device=device)


def invariance_loss_from_views(core, loss_fn: InvarianceLoss, ctx1: TokenSet, ctx2: TokenSet, world: Tensor,
                               Z1_prev: Optional[Tensor] = None, Z2_prev: Optional[Tensor] = None,
                               code_rows1: Optional[Tensor] = None, code_rows2: Optional[Tensor] = None) -> tuple:
    """Run the backbone on both views (from their own previous states; default the learned initial state) and return
    (loss, stats, Z1, Z2)."""
    B = ctx1.B
    dev = ctx1.x.device
    Z1_prev = core.init_state(B, dev) if Z1_prev is None else Z1_prev
    Z2_prev = core.init_state(B, dev) if Z2_prev is None else Z2_prev
    Z1, _ = core(Z1_prev, ctx1, code_rows=code_rows1)
    Z2, _ = core(Z2_prev, ctx2, code_rows=code_rows2)
    loss, stats = loss_fn(Z1, Z2, world)
    return loss, stats, Z1, Z2
