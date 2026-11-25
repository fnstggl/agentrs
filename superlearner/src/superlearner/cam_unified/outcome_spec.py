"""Generic typed outcomes of executed actions (spec Parts 5.3, 6.2; architecture §5, §6).

Every decision produces an `OutcomeBatch` whose keys are derived from what the environment actually returned, not
from a fixed list of levers or gauges:

=============================  ===========  ==========================================================================
key                            likelihood   target y
=============================  ===========  ==========================================================================
``ok``                         BERNOULLI    1.0 if the tool call succeeded else 0.0
``error_code``                 CATEGORICAL  0 = no error, else 1 + sha256(code) mod (n_error_classes - 1)
``cost``                       GAUSSIAN     symlog(cost) in the world's money unit (Crucible: integer cents)
``duration``                   GAUSSIAN     symlog(duration) in simulated days
``reward``                     GAUSSIAN     the verified return increment (raw)
``num:<metric>@<entity>[unit]`` GAUSSIAN    symlog(v_new - v_prev) for a numeric observation whose key was seen before
``lvl:<metric>@<entity>[unit]`` GAUSSIAN    symlog(v_new) when the (metric, entity, unit) key is seen for the first time
``cnt:<kind>/<type>``          COUNT        number of new observations of that event type in the step (0 included)
=============================  ===========  ==========================================================================

Query tokens are built from a natural-language *description* of the key (metric, entity type, unit; never the
persistent entity id, which is bound through the episode-local entity slot instead) passed through the observation
tokenizer, plus two fixed-size embeddings (likelihood family and target space). A new outcome type therefore needs no
new parameters. Channel slots of the Kalman/NIG last layer are bound per environment in ``FastState.channels``.

``symlog(x) = sign(x) * log1p(|x|)`` keeps cents-scale money and fraction-scale readings on one numerical scale.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from .interfaces import (Action, EntityRegistry, ExecutionResult, FastState, Likelihood, Observation, ObsKind,
                         OutcomeBatch, TokenKind, TokenSet)

CORE_KEYS = ("ok", "error_code", "cost", "duration", "reward")
CORE_INDEX = {k: i for i, k in enumerate(CORE_KEYS)}
SPACES = ("binary", "class", "symlog", "raw", "symlog_delta", "symlog_level", "count")
SPACE_INDEX = {s: i for i, s in enumerate(SPACES)}
CONTINUOUS = (int(Likelihood.GAUSSIAN), int(Likelihood.STUDENT_T))


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def symexp(y: float) -> float:
    return math.copysign(math.expm1(abs(y)), y)


def stable_hash(s: str) -> int:
    """Process-independent hash (Python's ``hash`` is salted per process)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def error_class(code: Optional[str], n_classes: int) -> int:
    """0 = no error; other codes hash into 1..n_classes-1 (a small shared vocabulary, so unseen codes need no new
    parameters; collisions are possible and harmless for a likelihood target)."""
    if not code:
        return 0
    return 1 + stable_hash(str(code)) % max(1, n_classes - 1)


def error_code_of(result: Optional[ExecutionResult]) -> Optional[str]:
    """The machine-readable error code of a failed call: ``meta['error_code']`` when the adapter provides it (Crucible
    ``error.code``), otherwise the first token of ``error`` (``"invalid_args: price out of range"`` -> ``invalid_args``)."""
    if result is None or result.ok:
        return None
    code = result.meta.get("error_code") if isinstance(result.meta, dict) else None
    if code:
        return str(code)
    if result.error:
        head = str(result.error).strip().split(":", 1)[0].split()
        return head[0] if head else "error"
    return "error"


def numeric_identity(o: Observation) -> tuple:
    """(key, metric, unit, entity) of a NUMERIC observation; metric = meta['metric'|'field'] or the source."""
    meta = o.meta if isinstance(o.meta, dict) else {}
    metric = str(meta.get("metric") or meta.get("field") or o.source or "value")
    unit = o.unit or ""
    ent = o.entity or "-"
    return f"{metric}@{ent}[{unit}]", metric, unit, o.entity


def event_type_of(o: Observation) -> Optional[str]:
    """Event type of a non-numeric observation (``meta['type']`` for Crucible events, else the observation kind).
    Tool results are excluded (their count is always one per call; ``ok`` covers them)."""
    if o.kind in (ObsKind.NUMERIC, ObsKind.TOOL_RESULT):
        return None
    meta = o.meta if isinstance(o.meta, dict) else {}
    et = meta.get("type") or meta.get("event_type") or o.kind.value
    return f"{o.kind.value}/{et}"


@dataclass
class OutcomeSpec:
    """One typed outcome query (and, once realised, its observed target)."""
    key: str
    text: str                      # description fed through the tokenizer (no persistent ids)
    lik: int
    space: str
    y: float = 0.0
    observed: bool = False
    n_classes: int = 0
    unit: Optional[str] = None
    entity: Optional[str] = None   # persistent id (bound to an episode-local slot when the batch is built)
    horizon_days: float = 0.0
    censored: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OutcomeSpec":
        return OutcomeSpec(**d)


def _core_specs(money_unit: str, n_error_classes: int) -> list:
    return [OutcomeSpec("ok", "whether the action succeeded", int(Likelihood.BERNOULLI), "binary"),
            OutcomeSpec("error_code", "error code returned by the action", int(Likelihood.CATEGORICAL), "class",
                        n_classes=n_error_classes),
            OutcomeSpec("cost", f"resource cost charged for the action, in {money_unit}", int(Likelihood.GAUSSIAN),
                        "symlog", unit=money_unit),
            OutcomeSpec("duration", "simulated duration charged for the action, in days", int(Likelihood.GAUSSIAN),
                        "symlog", unit="days"),
            OutcomeSpec("reward", "verified goal progress reward after the action", int(Likelihood.GAUSSIAN), "raw")]


class OutcomeTracker:
    """Per-environment bookkeeping that turns raw results into typed outcomes.

    Keeps the last observed value of every numeric (metric, entity, unit) key (for symlog deltas), the recently seen
    event types (so zero counts are observed outcomes too) and the recency of every key (for channel recycling).
    Bounded working sets (``max_numeric``, ``max_event_types``) are per-step prediction budgets, not data limits: the
    archive keeps every raw observation.
    """

    def __init__(self, money_unit: str = "cents", n_error_classes: int = 16, max_numeric: int = 32,
                 max_event_types: int = 16):
        self.money_unit = money_unit
        self.n_error_classes = n_error_classes
        self.max_numeric = max_numeric
        self.max_event_types = max_event_types
        self.reset()

    def reset(self) -> None:
        self.last_value: "OrderedDict[str, tuple]" = OrderedDict()   # key -> (value, t, metric, unit, entity, etype)
        self.event_types: "OrderedDict[str, int]" = OrderedDict()   # type -> last step seen
        self.last_seen: dict = {}                                   # outcome key -> step
        self.step = 0

    # ---- state ingestion
    def ingest(self, obs: Sequence[Observation], etypes: Optional[dict] = None) -> None:
        """Record numeric levels and event types of observations seen without being an action outcome (e.g. the
        observations delivered at reset)."""
        for o in sorted((o for o in obs if o.kind == ObsKind.NUMERIC and o.value is not None),
                        key=lambda o: (o.t, -1 if o.seq is None else o.seq)):
            k, metric, unit, ent = numeric_identity(o)
            self.last_value[k] = (float(o.value), o.t, metric, unit, ent, o.entity_type)
            self.last_value.move_to_end(k)
        for o in obs:
            et = event_type_of(o)
            if et is not None:
                self.event_types[et] = self.step
                self.event_types.move_to_end(et)
        while len(self.event_types) > 4 * self.max_event_types:
            self.event_types.popitem(last=False)

    # ---- query sets
    def _num_spec(self, k: str, delta: bool, metric: str, unit: str, ent, etype) -> OutcomeSpec:
        what = etype or "entity"
        if delta:
            return OutcomeSpec("num:" + k, f"change in {metric} of {what}, in {unit or 'units'}",
                               int(Likelihood.GAUSSIAN), "symlog_delta", unit=unit or None, entity=ent)
        return OutcomeSpec("lvl:" + k, f"newly observed {metric} of {what}, in {unit or 'units'}",
                           int(Likelihood.GAUSSIAN), "symlog_level", unit=unit or None, entity=ent)

    @staticmethod
    def _cnt_spec(et: str) -> OutcomeSpec:
        kind, typ = et.split("/", 1)
        return OutcomeSpec("cnt:" + et, f"number of new {typ} {kind} observations", int(Likelihood.COUNT), "count")

    def prospective(self) -> list:
        """Outcome queries for the next action before it is executed (``EventRecord.predicted`` keys)."""
        out = _core_specs(self.money_unit, self.n_error_classes)
        for k in list(self.last_value)[::-1][: self.max_numeric]:
            _, _, metric, unit, ent, etype = self.last_value[k]
            out.append(self._num_spec(k, True, metric, unit, ent, etype))
        for et in list(self.event_types)[::-1][: self.max_event_types]:
            out.append(self._cnt_spec(et))
        return out

    def realize(self, prev_obs: Sequence[Observation], action: Optional[Action], result: Optional[ExecutionResult],
                next_obs: Sequence[Observation], reward: float) -> list:
        """Observed outcomes of ``action``: the five core keys, every numeric observation in ``next_obs`` (delta to its
        previous value, or level when first seen) and event-type counts. Updates the tracker state."""
        self.ingest(prev_obs)          # values visible at decision time are the "before" values
        specs = _core_specs(result.meta.get("cost_unit", self.money_unit) if result is not None else self.money_unit,
                            self.n_error_classes)
        if result is not None:
            vals = [1.0 if result.ok else 0.0, float(error_class(error_code_of(result), self.n_error_classes)),
                    symlog(float(result.cost)), symlog(float(result.duration))]
            for s, v in zip(specs[:4], vals):
                s.y, s.observed = v, True
        specs[4].y, specs[4].observed = float(reward), True
        # numeric observations: latest value per key within this step
        latest: "OrderedDict[str, Observation]" = OrderedDict()
        for o in sorted((o for o in next_obs if o.kind == ObsKind.NUMERIC and o.value is not None
                         and math.isfinite(float(o.value))), key=lambda o: (o.t, -1 if o.seq is None else o.seq)):
            latest[numeric_identity(o)[0]] = o
        num_specs = []
        for k, o in list(latest.items())[::-1][: self.max_numeric]:
            _, metric, unit, ent = numeric_identity(o)
            prev = self.last_value.get(k)
            etype = o.entity_type or (prev[5] if prev else None)
            if prev is not None:
                s = self._num_spec(k, True, metric, unit, ent, etype)
                s.y = symlog(float(o.value) - prev[0])
            else:
                s = self._num_spec(k, False, metric, unit, ent, etype)
                s.y = symlog(float(o.value))
            s.observed = True
            num_specs.append(s)
        specs.extend(num_specs)
        for k, o in latest.items():
            _, metric, unit, ent = numeric_identity(o)
            self.last_value[k] = (float(o.value), o.t, metric, unit, ent, o.entity_type)
            self.last_value.move_to_end(k)
        # event-type counts (types tracked before this step get explicit zero counts)
        counts: dict = {}
        for o in next_obs:
            et = event_type_of(o)
            if et is not None:
                counts[et] = counts.get(et, 0) + 1
        tracked = list(self.event_types)[::-1][: self.max_event_types]
        order = list(dict.fromkeys(list(counts) + tracked))[: self.max_event_types]
        for et in order:
            s = self._cnt_spec(et)
            s.y, s.observed = float(counts.get(et, 0)), True
            specs.append(s)
        self.step += 1
        for et in counts:
            self.event_types[et] = self.step
            self.event_types.move_to_end(et)
        while len(self.event_types) > 4 * self.max_event_types:
            self.event_types.popitem(last=False)
        for s in specs:
            self.last_seen[s.key] = self.step
        return specs


# ============================================================================ query tokens

class TokenizerTextEncoder:
    """Text -> one d_model vector through the injected ObservationTokenizer: every description becomes one TEXT
    observation in its own batch row; its tokens are mask-mean pooled. With gradients disabled, results are cached
    per (text, version) and never served across versions (the tokenizer's projections change with updates)."""

    def __init__(self, tokenizer, version_fn: Callable[[], int] = lambda: 0, cache_size: int = 4096):
        self.tokenizer = tokenizer
        self.version_fn = version_fn
        self.cache_size = cache_size
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self.calls = 0

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        use_cache = not torch.is_grad_enabled()
        ver = self.version_fn()
        uniq = list(dict.fromkeys(texts))
        todo = [t for t in uniq if not (use_cache and (t, ver) in self._cache)]
        fresh = {}
        if todo:
            self.calls += 1
            obs_lists = [[Observation(ObsKind.TEXT, content=t, source="outcome_query")] for t in todo]
            regs = [EntityRegistry(1, seed=0) for _ in todo]
            ts = self.tokenizer.observations(obs_lists, regs, [0.0] * len(todo))
            m = ts.mask.to(ts.x.dtype).unsqueeze(-1)
            pooled = (ts.x * m).sum(1) / m.sum(1).clamp(min=1.0)
            for t, v in zip(todo, pooled):
                fresh[t] = v
                if use_cache:
                    self._cache[(t, ver)] = v.detach()
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        rows = [fresh[t] if t in fresh else self._cache[(t, ver)] for t in texts]
        return torch.stack(rows, 0)


class SemanticTextEncoder(nn.Module):
    """Text -> d_model through a SemanticEncoder's pooled ``embed`` and a learned projection (alternative to the
    tokenizer path when only the semantic encoder is available)."""

    def __init__(self, semantic, d_model: int):
        super().__init__()
        self.semantic = [semantic]                     # not registered: pretrained parameters are counted separately
        self.proj = nn.Linear(int(semantic.sem_dim), d_model)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        e = self.semantic[0].embed(list(texts))
        return self.proj(e.to(self.proj.weight.dtype).to(self.proj.weight.device))


class OutcomeQueryEncoder(nn.Module):
    """QUERY token = text_encoder(description) + E_lik[likelihood] + E_space[target space]; RMS-normalised.
    Parameters: two fixed tables (6 + 7 rows) and a gain, independent of how many outcome types exist."""

    def __init__(self, d_model: int, text_encoder: Callable[[Sequence[str]], torch.Tensor]):
        super().__init__()
        self.text_encoder = text_encoder
        self.lik_emb = nn.Embedding(len(Likelihood), d_model)
        self.space_emb = nn.Embedding(len(SPACES), d_model)
        nn.init.normal_(self.lik_emb.weight, std=0.02)
        nn.init.normal_(self.space_emb.weight, std=0.02)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, texts: Sequence[str], liks: torch.Tensor, spaces: torch.Tensor) -> torch.Tensor:
        """texts [N]; liks [N] long; spaces [N] long -> [N, D]."""
        if len(texts) == 0:
            return self.lik_emb.weight.new_zeros(0, self.lik_emb.embedding_dim)
        t = self.text_encoder(texts).to(self.lik_emb.weight.device, self.lik_emb.weight.dtype)
        return self.norm(t + self.lik_emb(liks) + self.space_emb(spaces))


def make_text_encoder(obj, d_model: int, version_fn: Callable[[], int] = lambda: 0):
    """Dispatch: an ObservationTokenizer (has ``observations``), a SemanticEncoder (has ``embed`` and ``sem_dim``) or
    a plain callable texts -> [N, D]."""
    if hasattr(obj, "observations"):
        return TokenizerTextEncoder(obj, version_fn)
    if hasattr(obj, "embed") and hasattr(obj, "sem_dim"):
        return SemanticTextEncoder(obj, d_model)
    if callable(obj):
        return obj
    raise TypeError(f"cannot build outcome query text encoder from {type(obj).__name__}")


# ============================================================================ channels and batches

def bind_channels(fast: Optional[FastState], b: int, specs: Sequence[OutcomeSpec], last_seen: Optional[dict] = None,
                  prior_var: float = 1.0, obs_var: float = 1.0) -> list:
    """Bind every outcome key of env ``b`` to a Kalman/NIG channel slot in ``fast.channels[b]``. A full table recycles
    the least recently seen key's slot and resets that slot's posterior (mean 0, covariance prior_var * I, noise
    obs_var, count 0). Returns the slot per spec (-1 when ``fast`` is None or has no channels)."""
    if fast is None or fast.bayes_mu is None or fast.bayes_mu.shape[1] == 0:
        return [-1] * len(specs)
    C = int(fast.bayes_mu.shape[1])
    table: dict = fast.channels[b]
    last_seen = last_seen or {}
    out = []
    protected = {s.key for s in specs}
    for s in specs:
        if s.key in table:
            out.append(int(table[s.key]))
            continue
        used = set(table.values())
        free = [c for c in range(C) if c not in used]
        if free:
            c = free[0]
        else:
            victims = [k for k in table if k not in protected]
            if not victims:
                out.append(-1)
                continue
            victim = min(victims, key=lambda k: last_seen.get(k, -1))
            c = int(table.pop(victim))
            with torch.no_grad():
                fast.bayes_mu[b, c].zero_()
                F = fast.bayes_P.shape[-1]
                fast.bayes_P[b, c].copy_(torch.eye(F, dtype=fast.bayes_P.dtype, device=fast.bayes_P.device) * prior_var)
                fast.bayes_noise[b, c] = obs_var
                fast.bayes_n[b, c] = 0
        table[s.key] = c
        out.append(c)
    return out


def build_outcome_batch(spec_lists: Sequence[Sequence[OutcomeSpec]], query_encoder: OutcomeQueryEncoder,
                        registries: Optional[Sequence[EntityRegistry]] = None,
                        channels: Optional[Sequence[Sequence[int]]] = None, device=None) -> OutcomeBatch:
    """Pad [B] lists of specs into an OutcomeBatch. q.x [B,Q,D]; padded entries have q.mask = y_mask = False and
    channel = -1. ``refs`` and ``keys`` carry the outcome keys."""
    B = len(spec_lists)
    Q = max([len(s) for s in spec_lists] + [0])
    D = query_encoder.lik_emb.embedding_dim
    dev = device if device is not None else query_encoder.lik_emb.weight.device
    flat = [s for specs in spec_lists for s in specs]
    liks = torch.tensor([s.lik for s in flat], dtype=torch.long, device=dev)
    spaces = torch.tensor([SPACE_INDEX[s.space] for s in flat], dtype=torch.long, device=dev)
    qx_flat = query_encoder([s.text for s in flat], liks, spaces)
    x = qx_flat.new_zeros(B, Q, D)
    mask = torch.zeros(B, Q, dtype=torch.bool, device=dev)
    lik = torch.zeros(B, Q, dtype=torch.long, device=dev)
    ncls = torch.zeros(B, Q, dtype=torch.long, device=dev)
    y = torch.zeros(B, Q, device=dev)
    ymask = torch.zeros(B, Q, dtype=torch.bool, device=dev)
    cens = torch.zeros(B, Q, dtype=torch.bool, device=dev)
    chan = torch.full((B, Q), -1, dtype=torch.long, device=dev)
    time = torch.zeros(B, Q, device=dev)
    ent = torch.full((B, Q), -1, dtype=torch.long, device=dev)
    keys = []
    i = 0
    rows_b, rows_q = [], []
    for b, specs in enumerate(spec_lists):
        keys.append([s.key for s in specs])
        for q, s in enumerate(specs):
            rows_b.append(b)
            rows_q.append(q)
            lik[b, q] = s.lik
            ncls[b, q] = s.n_classes
            y[b, q] = float(s.y)
            ymask[b, q] = bool(s.observed)
            cens[b, q] = bool(s.censored)
            time[b, q] = float(s.horizon_days)
            if channels is not None:
                chan[b, q] = int(channels[b][q])
            if registries is not None and s.entity is not None:
                ent[b, q] = int(registries[b].slot.get(s.entity, -1))
            i += 1
        mask[b, : len(specs)] = True
    if flat:
        idx_b = torch.tensor(rows_b, device=dev)
        idx_q = torch.tensor(rows_q, device=dev)
        x = x.index_put((idx_b, idx_q), qx_flat)
    q = TokenSet(x, mask, torch.full((B, Q), int(TokenKind.QUERY), dtype=torch.long, device=dev), time, ent,
                 [list(k) + [None] * (Q - len(k)) for k in keys])
    return OutcomeBatch(q, lik, ncls, y, cens, ymask, chan, keys)


def select_outcomes(ob: OutcomeBatch, rows) -> OutcomeBatch:
    """Batch rows of an OutcomeBatch (replication allowed, e.g. one row per imagined branch)."""
    rows_l = [int(r) for r in rows]
    idx = torch.tensor(rows_l, dtype=torch.long, device=ob.lik.device)
    return OutcomeBatch(ob.q.select(idx), ob.lik[idx], ob.n_classes[idx], ob.y[idx], ob.censored[idx], ob.y_mask[idx],
                        ob.channel[idx], None if ob.keys is None else [list(ob.keys[r]) for r in rows_l])


def with_targets(ob: OutcomeBatch, y: torch.Tensor, y_mask: Optional[torch.Tensor] = None) -> OutcomeBatch:
    """Same queries, new targets (imagined samples or Kalman residuals)."""
    return OutcomeBatch(ob.q, ob.lik, ob.n_classes, y, ob.censored, ob.q.mask.clone() if y_mask is None else y_mask,
                        ob.channel, ob.keys)


def outcomes_from_results(prev_obs: Sequence[Sequence[Observation]], actions: Sequence[Optional[Action]],
                          results: Sequence[Optional[ExecutionResult]], next_obs: Sequence[Sequence[Observation]],
                          reward: Sequence[float], encoder, *, trackers: Optional[Sequence[OutcomeTracker]] = None,
                          registries: Optional[Sequence[EntityRegistry]] = None, fast: Optional[FastState] = None,
                          d_model: Optional[int] = None, prior_var: float = 1.0, obs_var: float = 1.0,
                          device=None, return_specs: bool = False):
    """[B] step data -> OutcomeBatch with y filled (spec 6.2).

    ``encoder``: an OutcomeQueryEncoder, or an ObservationTokenizer / SemanticEncoder / callable from which one is
    built (``d_model`` required then). ``trackers`` carry numeric history across steps (fresh trackers when None:
    deltas are then relative to ``prev_obs`` only). With ``fast`` the keys are bound to channel slots in
    ``fast.channels``."""
    B = len(actions)
    if not isinstance(encoder, OutcomeQueryEncoder):
        if d_model is None:
            raise ValueError("d_model is required when `encoder` is not an OutcomeQueryEncoder")
        encoder = OutcomeQueryEncoder(d_model, make_text_encoder(encoder, d_model))
    trackers = list(trackers) if trackers is not None else [OutcomeTracker() for _ in range(B)]
    specs = [trackers[b].realize(prev_obs[b], actions[b], results[b], next_obs[b], float(reward[b])) for b in range(B)]
    chans = [bind_channels(fast, b, specs[b], trackers[b].last_seen, prior_var, obs_var) for b in range(B)]
    ob = build_outcome_batch(specs, encoder, registries, chans, device)
    return (ob, specs) if return_specs else ob
