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
``reward``                     GAUSSIAN     the verified return increment (raw); unobserved in observation_only
``num:<metric>@<entity>[unit]`` GAUSSIAN    symlog(v_new - v_prev) for a numeric observation whose key was seen before
``lvl:<metric>@<entity>[unit]`` GAUSSIAN    symlog(v_new) when the (metric, entity, unit) key is seen for the first time
``cnt:<kind>/<type>``          COUNT        number of new observations of that event type in the step (0 included)
=============================  ===========  ==========================================================================

Query tokens are built from a natural-language *description* of the key (metric, entity type, unit; never the
persistent entity id, which is bound through the episode-local entity slot instead) passed through the observation
tokenizer, plus two fixed-size embeddings (likelihood family and target space). A new outcome type therefore needs no
new parameters. Channel slots of the Kalman/NIG last layer are bound per environment in ``FastState.channels`` for
the *continuous* keys that were actually observed (``make_room`` / ``bind_channels``): prospective queries only look
up existing bindings, realised keys are never evicted in the step they are realised (stable least-recently-realised
recycling), and every eviction or key that does not fit is counted.

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
    event types (so zero counts are observed outcomes too) and the recency of every realised key (for channel
    recycling). Bounded working sets (``max_numeric``, ``max_event_types``) are per-step prediction budgets, not data
    limits: the archive keeps every raw observation, so a dropped target can always be recomputed from it. Every target
    left out by a budget is counted in ``dropped`` (never silent).
    """

    def __init__(self, money_unit: str = "cents", n_error_classes: int = 16, max_numeric: int = 32,
                 max_event_types: int = 16):
        self.money_unit = money_unit
        self.n_error_classes = n_error_classes
        self.max_numeric = max_numeric
        self.max_event_types = max_event_types
        self.dropped = {"numeric_targets": 0, "event_type_targets": 0, "numeric_queries": 0, "event_type_queries": 0}
        self.reset()

    def reset(self) -> None:
        self.last_value: "OrderedDict[str, tuple]" = OrderedDict()   # key -> (value, t, metric, unit, entity, etype)
        self.event_types: "OrderedDict[str, int]" = OrderedDict()   # type -> last step seen
        self.last_seen: dict = {}                                   # outcome key -> step it was last realised
        self.step = 0
        self.last_dropped = {"numeric_targets": 0, "event_type_targets": 0}

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
        self.dropped["numeric_queries"] += max(0, len(self.last_value) - self.max_numeric)
        self.dropped["event_type_queries"] += max(0, len(self.event_types) - self.max_event_types)
        for k in list(self.last_value)[::-1][: self.max_numeric]:
            _, _, metric, unit, ent, etype = self.last_value[k]
            out.append(self._num_spec(k, True, metric, unit, ent, etype))
        for et in list(self.event_types)[::-1][: self.max_event_types]:
            out.append(self._cnt_spec(et))
        return out

    def realize(self, prev_obs: Sequence[Observation], action: Optional[Action], result: Optional[ExecutionResult],
                next_obs: Sequence[Observation], reward: Optional[float]) -> list:
        """Observed outcomes of ``action``: the five core keys, every numeric observation in ``next_obs`` (delta to its
        previous value, or level when first seen) and event-type counts. Updates the tracker state. ``reward`` None
        (reward_mode "observation_only"): the core 'reward' outcome is unobserved."""
        self.ingest(prev_obs)          # values visible at decision time are the "before" values
        specs = _core_specs(result.meta.get("cost_unit", self.money_unit) if result is not None else self.money_unit,
                            self.n_error_classes)
        if result is not None:
            vals = [1.0 if result.ok else 0.0, float(error_class(error_code_of(result), self.n_error_classes)),
                    symlog(float(result.cost)), symlog(float(result.duration))]
            for s, v in zip(specs[:4], vals):
                s.y, s.observed = v, True
        if reward is None:
            # observation_only (plan 2026-11-26 §2): the verified return is not an actor-side observation; the
            # 'reward' query stays in the set (stable CORE_INDEX) but is never a belief / Kalman / world-model target
            specs[4].y, specs[4].observed = 0.0, False
        else:
            specs[4].y, specs[4].observed = float(reward), True
        # numeric observations: latest value per key within this step
        latest: "OrderedDict[str, Observation]" = OrderedDict()
        for o in sorted((o for o in next_obs if o.kind == ObsKind.NUMERIC and o.value is not None
                         and math.isfinite(float(o.value))), key=lambda o: (o.t, -1 if o.seq is None else o.seq)):
            latest[numeric_identity(o)[0]] = o
        num_specs = []
        n_num_drop = max(0, len(latest) - self.max_numeric)
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
        order_all = list(dict.fromkeys(list(counts) + tracked))
        order = order_all[: self.max_event_types]
        n_evt_drop = len(order_all) - len(order)
        self.last_dropped = {"numeric_targets": n_num_drop, "event_type_targets": n_evt_drop}
        self.dropped["numeric_targets"] += n_num_drop
        self.dropped["event_type_targets"] += n_evt_drop
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

def params_fingerprint(module) -> int:
    """Monotone counter of in-place modifications of `module`'s parameters and buffers (sum of tensor version
    counters): every optimiser step, ``load_state_dict`` copy or manual in-place edit increases it."""
    if not isinstance(module, nn.Module):
        return 0
    return int(sum(t._version for t in module.parameters()) + sum(t._version for t in module.buffers()))


class TokenizerTextEncoder:
    """Text -> one d_model vector through the injected ObservationTokenizer: every description becomes one TEXT
    observation in its own batch row; its tokens are mask-mean pooled. With gradients disabled, results are cached
    per (text, weights key). The key is ``(epoch_fn(), params_fingerprint(tokenizer))``: ``epoch_fn`` is the owner's
    monotone weights epoch (bumped on every optimiser step, weight load and version assignment), and the fingerprint
    catches any in-place parameter change the owner was not told about, so a cached embedding is never served for
    weights other than the ones that produced it (spec Part 22)."""

    def __init__(self, tokenizer, epoch_fn: Callable[[], int] = lambda: 0, cache_size: int = 4096):
        self.tokenizer = tokenizer
        self.epoch_fn = epoch_fn
        self.cache_size = cache_size
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._key = None
        self.calls = 0
        self.hits = 0

    def weights_key(self) -> tuple:
        return (self.epoch_fn(), params_fingerprint(self.tokenizer))

    def clear(self) -> None:
        self._cache.clear()

    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        texts = list(texts)
        use_cache = not torch.is_grad_enabled()
        key = self.weights_key()
        if key != self._key:                       # weights changed: nothing cached is valid any more
            self._cache.clear()
            self._key = key
        uniq = list(dict.fromkeys(texts))
        todo = [t for t in uniq if not (use_cache and t in self._cache)]
        self.hits += len(uniq) - len(todo)
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
                    self._cache[t] = v.detach()
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        rows = [fresh[t] if t in fresh else self._cache[t] for t in texts]
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


def make_text_encoder(obj, d_model: int, epoch_fn: Callable[[], int] = lambda: 0):
    """Dispatch: an ObservationTokenizer (has ``observations``), a SemanticEncoder (has ``embed`` and ``sem_dim``) or
    a plain callable texts -> [N, D]."""
    if hasattr(obj, "observations"):
        return TokenizerTextEncoder(obj, epoch_fn)
    if hasattr(obj, "embed") and hasattr(obj, "sem_dim"):
        return SemanticTextEncoder(obj, d_model)
    if callable(obj):
        return obj
    raise TypeError(f"cannot build outcome query text encoder from {type(obj).__name__}")


# ============================================================================ channels and batches

def is_kalman_spec(s: OutcomeSpec) -> bool:
    """Continuous outcomes are the ones the Kalman/NIG last layer adapts (Bernoulli / categorical / count / survival
    outcomes are adapted by the particle beliefs only)."""
    return int(s.lik) in CONTINUOUS


def lookup_channels(fast: Optional[FastState], b: int, specs: Sequence[OutcomeSpec]) -> list:
    """Existing channel slot per spec of env ``b`` (-1 = unbound or not continuous). Never allocates: prospective
    (decision-time) queries only use bindings made by realised outcomes."""
    if fast is None or fast.bayes_mu is None or fast.bayes_mu.shape[1] == 0:
        return [-1] * len(specs)
    table = fast.channels[b]
    return [int(table.get(s.key, -1)) if is_kalman_spec(s) else -1 for s in specs]


def _reset_slot(fast: FastState, b: int, c: int, prior_var, obs_var) -> None:
    with torch.no_grad():
        F_ = fast.bayes_P.shape[-1]
        pv = torch.as_tensor(prior_var, dtype=fast.bayes_P.dtype, device=fast.bayes_P.device).detach()
        pv = pv.expand(F_) if pv.dim() == 0 else pv
        fast.bayes_mu[b, c].zero_()
        fast.bayes_P[b, c].copy_(torch.diag(pv))
        fast.bayes_noise[b, c] = float(torch.as_tensor(obs_var).detach())
        fast.bayes_n[b, c] = 0


def make_room(fast: Optional[FastState], b: int, specs: Sequence[OutcomeSpec], last_seen: Optional[dict] = None,
              prior_var=1.0, obs_var=1.0, stats: Optional[dict] = None) -> list:
    """Free channel slots for the *observed continuous* specs of env ``b`` that are not bound yet: when fewer slots are
    free than needed, unbind the least recently realised keys that are not realised in this step (keys realised now
    are never evicted, so a set of keys that keeps being observed keeps its posteriors). A freed slot's posterior is
    reset to the prior (mean 0, covariance diag(prior_var), noise obs_var, count 0). Returns the evicted keys;
    ``stats['evicted']`` counts them."""
    if fast is None or fast.bayes_mu is None or fast.bayes_mu.shape[1] == 0:
        return []
    C = int(fast.bayes_mu.shape[1])
    table: dict = fast.channels[b]
    last_seen = last_seen or {}
    realised = [s.key for s in specs if s.observed and is_kalman_spec(s)]
    now = set(realised)
    need = len([k for k in dict.fromkeys(realised) if k not in table])
    free = C - len(set(table.values()))
    evicted = []
    if need > free:
        victims = sorted((k for k in table if k not in now), key=lambda k: last_seen.get(k, -1))
        for k in victims[: need - free]:
            c = int(table.pop(k))
            _reset_slot(fast, b, c, prior_var, obs_var)
            evicted.append(k)
    if stats is not None:
        stats["evicted"] = stats.get("evicted", 0) + len(evicted)
    return evicted


def bind_channels(fast: Optional[FastState], b: int, specs: Sequence[OutcomeSpec], last_seen: Optional[dict] = None,
                  prior_var=1.0, obs_var=1.0, stats: Optional[dict] = None) -> list:
    """Model-side allocation (for a Bayes layer that does not allocate its own slots): `make_room`, then bind every
    observed continuous key of env ``b`` to a free slot in ``fast.channels[b]`` (slot posterior reset to the prior).
    Unobserved or non-continuous specs only use existing bindings. Returns the slot per spec (-1 = none); keys that
    still do not fit are counted in ``stats['unbound']``."""
    if fast is None or fast.bayes_mu is None or fast.bayes_mu.shape[1] == 0:
        return [-1] * len(specs)
    C = int(fast.bayes_mu.shape[1])
    make_room(fast, b, specs, last_seen, prior_var, obs_var, stats)
    table: dict = fast.channels[b]
    out, unbound = [], 0
    for s in specs:
        if not is_kalman_spec(s):
            out.append(-1)
            continue
        if s.key not in table and s.observed:
            used = set(table.values())
            free = [c for c in range(C) if c not in used]
            if free:
                table[s.key] = free[0]
                _reset_slot(fast, b, free[0], prior_var, obs_var)
            else:
                unbound += 1
        out.append(int(table.get(s.key, -1)))
    if stats is not None:
        stats["unbound"] = stats.get("unbound", 0) + unbound
    return out


def build_outcome_batch(spec_lists: Sequence[Sequence[OutcomeSpec]], query_encoder: OutcomeQueryEncoder,
                        registries: Optional[Sequence[EntityRegistry]] = None,
                        channels: Optional[Sequence[Sequence[int]]] = None, device=None) -> OutcomeBatch:
    """Pad [B] lists of specs into an OutcomeBatch. q.x [B,Q,D]; padded entries have q.mask = y_mask = False and
    channel = -1. ``refs`` and ``keys`` carry the outcome keys ([B][Q], None on padded entries)."""
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
    keys = [list(k) + [None] * (Q - len(k)) for k in keys]          # contract: keys [B][Q] (None = padding)
    q = TokenSet(x, mask, torch.full((B, Q), int(TokenKind.QUERY), dtype=torch.long, device=dev), time, ent,
                 [list(k) for k in keys])
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
                          reward: Optional[Sequence[Optional[float]]], encoder, *, trackers: Optional[Sequence[OutcomeTracker]] = None,
                          registries: Optional[Sequence[EntityRegistry]] = None, fast: Optional[FastState] = None,
                          d_model: Optional[int] = None, prior_var: float = 1.0, obs_var: float = 1.0,
                          device=None, return_specs: bool = False, stats: Optional[dict] = None):
    """[B] step data -> OutcomeBatch with y filled (spec 6.2).

    ``encoder``: an OutcomeQueryEncoder, or an ObservationTokenizer / SemanticEncoder / callable from which one is
    built (``d_model`` required then). ``trackers`` carry numeric history across steps (fresh trackers when None:
    deltas are then relative to ``prev_obs`` only). With ``fast`` the keys are bound to channel slots in
    ``fast.channels``. ``reward`` None (or None entries): the 'reward' outcome is unobserved (observation_only)."""
    B = len(actions)
    if not isinstance(encoder, OutcomeQueryEncoder):
        if d_model is None:
            raise ValueError("d_model is required when `encoder` is not an OutcomeQueryEncoder")
        encoder = OutcomeQueryEncoder(d_model, make_text_encoder(encoder, d_model))
    trackers = list(trackers) if trackers is not None else [OutcomeTracker() for _ in range(B)]
    rew = [None if (reward is None or reward[b] is None) else float(reward[b]) for b in range(B)]
    specs = [trackers[b].realize(prev_obs[b], actions[b], results[b], next_obs[b], rew[b]) for b in range(B)]
    chans = [bind_channels(fast, b, specs[b], trackers[b].last_seen, prior_var, obs_var, stats) for b in range(B)]
    ob = build_outcome_batch(specs, encoder, registries, chans, device)
    return (ob, specs) if return_specs else ob
