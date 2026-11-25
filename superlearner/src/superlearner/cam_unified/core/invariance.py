"""Invariant representations across surface presentation (workstream B; requirement W02, spec 13.3).

Two parts:

1. `SurfaceRandomizer` builds training pairs: from the records of one experience (observations, tools, actions, goal)
   it produces a dynamically equivalent view with a different surface — renamed entity identities (consistent bijection,
   Crucible-style prefixes such as "p:" / "o:" / "svc:" kept), renamed tools, rescaled units with matching unit
   strings (e.g. 12_345 "cents" -> 123.45 "dollars", 2000 "ms" -> 2 "s", 0.25 "fraction" -> 25 "percent"; unknown
   units -> "milli-<unit>" x1000), permuted observation and tool order, and reworded text (case-preserving synonym
   substitution that never touches identifiers or numbers). `SurfaceMap.apply_action` / `invert_action` translate
   actions between the views, so a policy acting in the surface view executes the same intervention.
   Action arguments keep their original units (tool semantics are defined by the environment).

2. `InvarianceLoss`: projected, normalised pooled latent states of the two views; NT-Xent with 2B-2 candidates per
   anchor where the positive is the other view of the same experience and negatives are views of experiences from
   **different worlds** only — other experiences of the same world (same mechanism, possibly similar dynamics) are
   excluded rather than pushed apart, and distinct mechanisms are never pulled together.
"""
from __future__ import annotations

import copy
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

SYNONYMS = {
    "customers": ["clients", "buyers"], "customer": ["client", "buyer"], "complain": ["grumble", "object"],
    "takes": ["requires", "needs"], "too": ["overly", "excessively"], "long": ["lengthy", "slow"],
    "deliver": ["ship", "supply"], "units": ["pieces", "items"], "search": ["query", "look up"],
    "simulated": ["virtual", "emulated"], "internet": ["web", "network"], "pages": ["documents", "sites"],
    "matching": ["fitting", "satisfying"], "set": ["assign", "configure"], "price": ["cost", "charge"],
    "monthly": ["per-month", "each month's"], "products": ["offerings", "goods"], "product": ["offering", "item"],
    "execute": ["run", "launch"], "program": ["script", "routine"], "return": ["give back", "report"],
    "send": ["transmit", "dispatch"], "message": ["note", "communication"], "participant": ["party", "actor"],
    "hire": ["recruit", "employ"], "workers": ["staff", "employees"], "role": ["position", "function"],
    "error": ["failure", "fault"], "did": ["could"], "respond": ["answer", "reply"], "revenue": ["income", "sales"],
    "increase": ["raise", "grow"], "decrease": ["lower", "reduce"], "service": ["system", "app"],
    "upstream": ["backend", "dependency"], "segment": ["group", "cohort"], "onboarding": ["setup", "sign-up"],
    "balance": ["funds", "holdings"], "account": ["ledger", "wallet"], "build": ["compile", "assemble"],
    "deploy": ["release", "ship"], "request": ["query", "call"], "rating": ["score", "review"],
    "can": ["could"], "you": ["you"], "by": ["before"], "your": ["the"], "of": ["of"], "one": ["a single"],
    "another": ["a different"], "workers.": ["staff."],
}
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _case_like(src: str, word: str) -> str:
    if src.isupper() and len(src) > 1:
        return word.upper()
    if src[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


@dataclass
class SurfaceMap:
    """A consistent surface transformation of one experience (and its inverse for actions)."""
    entity: dict = field(default_factory=dict)       # original id -> surface id
    tool: dict = field(default_factory=dict)         # original tool name -> surface name
    unit: dict = field(default_factory=dict)         # original unit -> (surface unit, factor)
    words: dict = field(default_factory=dict)        # lowercase word -> replacement
    order_seed: Optional[int] = None                 # None = keep order

    # ---- text
    def _replace_ids(self, text: str) -> str:
        return _replace_with(text, self._fwd())

    def _units_in_text(self, text: str) -> str:
        units = [u for u in self.unit if u]
        if not units or not text:
            return text
        pat = re.compile(r"(-?\d+(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
                         + r")(?![\w])")

        def sub(m):
            nu, f = self.unit[m.group(2)]
            return f"{_fmt(float(m.group(1)) * f)} {nu}"
        return pat.sub(sub, text)

    def _reword(self, text: str) -> str:
        if not self.words or not text:
            return text
        protected = set(self.entity.values()) | set(self.tool.values())

        def sub(m):
            w = m.group(0)
            if w in protected:
                return w
            r = self.words.get(w.lower())
            return _case_like(w, r) if r else w
        return _WORD.sub(sub, text)

    def text(self, s: str) -> str:
        return self._reword(self._units_in_text(self._replace_ids(s)))

    # ---- records
    def apply_observation(self, o: Observation) -> Observation:
        o = copy.deepcopy(o)
        if o.kind in (ObsKind.CODE, ObsKind.FILE):
            o.content = self._replace_ids(o.content)          # code: identifiers only, never reworded
        else:
            o.content = self.text(o.content)
        o.entity = self.entity.get(o.entity, o.entity)
        o.target = self.entity.get(o.target, o.target)
        if o.source in self.tool:
            o.source = self.tool[o.source]
        if o.relation:
            o.relation = self._reword(o.relation)
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
        return a

    def invert_action(self, a: Action) -> Action:
        a = copy.deepcopy(a)
        inv_e = {v: k for k, v in self.entity.items()}
        inv_t = {v: k for k, v in self.tool.items()}
        names = {**inv_e, **inv_t}
        a.tool = inv_t.get(a.tool, a.tool)
        a.args = self._map_args(a.args, names)
        a.entities = [inv_e.get(e, e) for e in a.entities]
        return a

    def apply_goal(self, g: GoalSpec) -> GoalSpec:
        g = copy.deepcopy(g)
        g.text = self.text(g.text)
        return g


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
                 rescale_units: bool = True, permute: bool = True, reword: bool = True, reword_prob: float = 0.7):
        self.rng = random.Random(seed)
        self.rename_entities, self.rename_tools = rename_entities, rename_tools
        self.rescale_units, self.permute, self.reword = rescale_units, permute, reword
        self.reword_prob = reword_prob

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
