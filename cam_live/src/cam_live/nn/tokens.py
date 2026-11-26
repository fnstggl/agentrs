"""cam_live.nn.tokens — experience tokens (spec Part 3.1).

* :func:`to_items` turns every contract record into :class:`~cam_live.contracts.ExperienceItem` objects (text for
  the semantic backbone + exact :class:`~cam_live.contracts.Quantity` values).
* :class:`ExperienceTokenizer` fuses, per item, four separately scaled channels into one token:

      content   RMS-normalised projection of the multi-depth semantic embedding            (unit RMS)
      value     the exact quantities of the item, through their own channel                (unit RMS when present)
      kind      item-kind and trust embeddings                                             (small)
      time      features of the gap to the previous token                                  (small)

  The value channel never goes through the language embedding. Each quantity is encoded from sign and several
  symmetric-log magnitude scales plus Fourier terms of asinh(value); the odd (sign-carrying) features are bound
  multiplicatively to a hashed, learned per-metric-name embedding, so ``+1`` and ``-1`` of the same metric map to
  opposite directions and different metrics to near-orthogonal ones. It is normalised on its own before fusion, so
  a long text cannot swamp a number (an earlier design gave cos(+1, -1) = 0.98). Raw values ride along unchanged in
  ``TokenBatch.values`` (float64).

  Time: ``TokenBatch.time`` keeps absolute float64 seconds. A token only encodes the (log-scaled) gap to its
  predecessor; the age of every earlier event relative to the newest one is supplied to the core's attention as
  ``t_i - t_j`` (see :mod:`cam_live.nn.core`), which keeps ``h[:, i]`` causal while resolving 1 second and 30 days.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..contracts import (ActionResult, ActionSpec, Cost, Decision, ExperienceItem, ExperimentRecord, FailureClass,
                         ItemKind, Limits, Observation, ObstacleRecord, OutcomeKind, OutcomeObservation,
                         OutcomeQuery, OutcomeStatus, PredictionRecord, Quantity, ResultStatus, TaskSpec,
                         ToolDescriptor, Trust, VerifiedOutcome)
from .core import RMSNorm

KIND_ID: Dict[str, int] = {k: i for i, k in enumerate(ItemKind.ALL)}
TRUST_ALL: Tuple[str, ...] = (Trust.UNTRUSTED, Trust.OPERATOR, Trust.VERIFIER, Trust.SELF)
TRUST_ID: Dict[str, int] = {k: i for i, k in enumerate(TRUST_ALL)}
QKIND_ID: Dict[str, int] = {k: i for i, k in enumerate(OutcomeKind.ALL)}

_ASINH_SCALES = (1e-3, 1.0, 1e3, 1e6)
_VALUE_FREQS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_ODD = 1 + len(_ASINH_SCALES) + len(_VALUE_FREQS)
N_EVEN = len(_VALUE_FREQS) + 3
N_META = 2 + len(OutcomeKind.ALL)
_GAP_SCALES = (1.0, 60.0, 3600.0, 86400.0)
_GAP_FREQS = (0.5, 1.0, 2.0, 4.0, 8.0)
N_GAP = len(_GAP_SCALES) + 2 * len(_GAP_FREQS) + 2


def stable_hash(text: str, salt: str = "") -> int:
    """Process-independent 63-bit hash (Python's ``hash`` is salted per process)."""
    d = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8, person=salt.encode()[:16]).digest()
    return int.from_bytes(d, "little") >> 1


def value_features(v: Tensor) -> Tuple[Tensor, Tensor]:
    """(odd [..., N_ODD], even [..., N_EVEN]) float32 features of exact values ``v`` (computed in float64).
    odd(-v) = -odd(v); even(-v) = even(v)."""
    v = v.to(torch.float64)
    u = torch.asinh(v)
    freqs = torch.tensor(_VALUE_FREQS, dtype=torch.float64, device=v.device)
    scales = torch.tensor(_ASINH_SCALES, dtype=torch.float64, device=v.device)
    ang = u.unsqueeze(-1) * freqs
    odd = torch.cat([torch.sign(v).unsqueeze(-1), torch.tanh(torch.asinh(v.unsqueeze(-1) / scales) / 4.0),
                     torch.sin(ang)], -1)
    even = torch.cat([torch.cos(ang), (v == 0).to(torch.float64).unsqueeze(-1),
                      (v == torch.round(v)).to(torch.float64).unsqueeze(-1), torch.ones_like(v).unsqueeze(-1)], -1)
    return odd.to(torch.float32), even.to(torch.float32)


def gap_features(gap: Tensor, is_first: Tensor) -> Tensor:
    """float32 [..., N_GAP] features of the signed gap (seconds) between a token and its predecessor."""
    g = gap.to(torch.float64)
    mag = g.abs()
    scales = torch.tensor(_GAP_SCALES, dtype=torch.float64, device=g.device)
    freqs = torch.tensor(_GAP_FREQS, dtype=torch.float64, device=g.device)
    u = torch.log1p(mag).unsqueeze(-1)
    f = torch.cat([torch.tanh(torch.log1p(mag.unsqueeze(-1) / scales) / 4.0), torch.sin(u * freqs),
                   torch.cos(u * freqs), (g < 0).to(torch.float64).unsqueeze(-1),
                   is_first.to(torch.float64).unsqueeze(-1)], -1)
    return f.to(torch.float32)


class ExperienceTokenizer(nn.Module):
    """Args: d_model; d_sem (= n_taps * backbone hidden width); n_taps; max_values (V); value_dim (0 = auto);
    n_buckets / unit_buckets (hash tables for metric names / units); sem_rank (0 = full projection)."""

    def __init__(self, d_model: int, d_sem: int, n_taps: int = 1, *, max_values: int = 4, value_dim: int = 0,
                 n_buckets: int = 2048, unit_buckets: int = 256, sem_rank: int = 0):
        super().__init__()
        if d_sem % n_taps:
            raise ValueError("d_sem must be a multiple of n_taps")
        self.d_model, self.d_sem, self.n_taps, self.max_values = d_model, d_sem, n_taps, max_values
        self.n_buckets, self.unit_buckets = n_buckets, unit_buckets
        dv = self.value_dim = value_dim or max(16, d_model // 4)
        # content
        if sem_rank:
            self.sem_proj: nn.Module = nn.Sequential(nn.Linear(d_sem, sem_rank, bias=False),
                                                     nn.Linear(sem_rank, d_model, bias=False))
        else:
            self.sem_proj = nn.Linear(d_sem, d_model, bias=False)
        self.no_text = nn.Parameter(torch.randn(d_model) * 0.5)
        self.content_norm = RMSNorm(d_model)
        self.kind_emb = nn.Embedding(len(ItemKind.ALL), d_model)
        self.trust_emb = nn.Embedding(len(TRUST_ALL), d_model)
        # value channel
        self.w_odd = nn.Linear(N_ODD, dv, bias=False)
        self.w_even = nn.Linear(N_EVEN + N_META, dv, bias=False)
        self.even_gain = nn.Parameter(torch.tensor(0.3))
        self.name_a = nn.Embedding(n_buckets, dv)
        self.name_b = nn.Embedding(n_buckets, dv)
        self.unit_emb = nn.Embedding(unit_buckets, dv)
        self.value_out = nn.Linear(dv, d_model, bias=False)
        self.value_norm = RMSNorm(d_model)
        self.value_gain = nn.Parameter(torch.tensor(1.0))
        # time gap
        self.gap_proj = nn.Linear(N_GAP, d_model, bias=False)
        self.stats = {"dropped_values": 0, "nonfinite_values": 0, "truncated_items": 0}
        for emb, std in ((self.kind_emb, 0.3), (self.trust_emb, 0.15), (self.name_a, 0.67), (self.name_b, 0.67),
                         (self.unit_emb, 0.3)):
            nn.init.normal_(emb.weight, std=std)
        nn.init.normal_(self.w_odd.weight, std=1.0 / math.sqrt(N_ODD))
        nn.init.normal_(self.w_even.weight, std=1.0 / math.sqrt(N_EVEN + N_META))
        nn.init.normal_(self.value_out.weight, std=1.0 / math.sqrt(dv))
        nn.init.normal_(self.gap_proj.weight, std=0.3 / math.sqrt(N_GAP))

    # ---- hashing ---------------------------------------------------------------------------------------------------
    def name_ids(self, name: str) -> Tuple[int, int]:
        return stable_hash(name, "cam.name.a") % self.n_buckets, stable_hash(name, "cam.name.b") % self.n_buckets

    def unit_id(self, unit: str) -> int:
        return stable_hash(unit, "cam.unit") % self.unit_buckets

    def name_embedding(self, names: Sequence[str]) -> Tensor:
        """[n, value_dim] learned hashed embedding of metric names (shared with the value channel)."""
        dev = self.name_a.weight.device
        ids = torch.tensor([self.name_ids(n) for n in names], dtype=torch.long, device=dev).view(-1, 2)
        return self.name_a(ids[:, 0]) + self.name_b(ids[:, 1])

    # ---- collation (python records -> index / value tensors) ------------------------------------------------------
    def collate(self, contexts: Sequence[Sequence[ExperienceItem]], max_context: Optional[int] = None
                ) -> Dict[str, Any]:
        """Right-padded tensors (on CPU) for a batch of contexts. Each context keeps its newest ``max_context`` items.
        T >= 1 always; an empty context is one padded position."""
        ctxs = []
        for c in contexts:
            c = list(c)
            if max_context is not None and len(c) > max_context:
                self.stats["truncated_items"] += len(c) - max_context
                c = c[-max_context:]
            ctxs.append(c)
        B, V = len(ctxs), self.max_values
        T = max(1, max((len(c) for c in ctxs), default=1))
        pad = torch.zeros(B, T, dtype=torch.bool)
        has_text = torch.zeros(B, T, dtype=torch.bool)
        time = torch.zeros(B, T, dtype=torch.float64)
        kind_ids = torch.zeros(B, T, dtype=torch.long)
        trust_ids = torch.zeros(B, T, dtype=torch.long)
        values = torch.zeros(B, T, V, dtype=torch.float64)
        vmask = torch.zeros(B, T, V, dtype=torch.bool)
        name_ids = torch.zeros(B, T, V, 2, dtype=torch.long)
        unit_ids = torch.zeros(B, T, V, dtype=torch.long)
        qkind = torch.zeros(B, T, V, dtype=torch.long)
        exposure = torch.ones(B, T, V, dtype=torch.float64)
        texts: List[List[str]] = []
        refs: List[List[str]] = []
        for bi, c in enumerate(ctxs):
            row_t, row_r = [""] * T, [""] * T
            for ti, it in enumerate(c):
                if it.kind not in KIND_ID:
                    raise ValueError(f"unknown item kind {it.kind!r}")
                pad[bi, ti] = True
                time[bi, ti] = float(it.ts)
                kind_ids[bi, ti] = KIND_ID[it.kind]
                trust_ids[bi, ti] = TRUST_ID.get(it.trust, 0)
                row_t[ti], row_r[ti] = it.text or "", it.ref or ""
                has_text[bi, ti] = bool(it.text)
                vi = 0
                for q in it.quantities:
                    val = float(q.value)
                    if not math.isfinite(val):
                        self.stats["nonfinite_values"] += 1
                        continue
                    if vi >= V:
                        self.stats["dropped_values"] += 1
                        continue
                    values[bi, ti, vi], vmask[bi, ti, vi] = val, True
                    name_ids[bi, ti, vi, 0], name_ids[bi, ti, vi, 1] = self.name_ids(q.name)
                    unit_ids[bi, ti, vi] = self.unit_id(q.unit)
                    qkind[bi, ti, vi] = QKIND_ID.get(q.kind, 0)
                    exposure[bi, ti, vi] = float(q.exposure) if math.isfinite(float(q.exposure)) else 1.0
                    vi += 1
            if c:
                time[bi, len(c):] = float(c[-1].ts)
            texts.append(row_t)
            refs.append(row_r)
        return {"texts": texts, "refs": refs, "pad_mask": pad, "has_text": has_text, "time": time,
                "kind_ids": kind_ids, "trust_ids": trust_ids, "values": values, "value_mask": vmask,
                "name_ids": name_ids, "unit_ids": unit_ids, "qkind": qkind, "exposure": exposure}

    # ---- channels --------------------------------------------------------------------------------------------------
    def content(self, sem: Tensor, has_text: Tensor) -> Tensor:
        """Unit-RMS content vector [..., D] from semantic embeddings [..., d_sem] (each depth tap normalised on its
        own); items without text use a learned embedding."""
        s = sem.to(torch.float32)
        taps = s.view(*s.shape[:-1], self.n_taps, -1)
        taps = taps * torch.rsqrt(taps.pow(2).mean(-1, keepdim=True) + 1e-6)
        c = self.sem_proj(taps.flatten(-2))
        c = torch.where(has_text.unsqueeze(-1), c, self.no_text.expand_as(c))
        return self.content_norm(c)

    def value_channel(self, values: Tensor, value_mask: Tensor, name_ids: Tensor, unit_ids: Tensor, qkind: Tensor,
                      exposure: Tensor) -> Tensor:
        """[..., D] value vector of each item from its exact quantities [..., V] (zero where an item has none)."""
        odd, even = value_features(values)
        ex = exposure.to(torch.float64)
        meta = torch.cat([(torch.log1p(ex.clamp_min(0.0)) / 10.0).to(torch.float32).unsqueeze(-1),
                          (ex != 1.0).to(torch.float32).unsqueeze(-1),
                          torch.nn.functional.one_hot(qkind, len(OutcomeKind.ALL)).to(torch.float32)], -1)
        name = self.name_a(name_ids[..., 0]) + self.name_b(name_ids[..., 1]) + self.unit_emb(unit_ids)
        per_q = (self.w_odd(odd) + self.even_gain * self.w_even(torch.cat([even, meta], -1))) * name
        m = value_mask.unsqueeze(-1).to(per_q.dtype)
        v = self.value_out((per_q * m).sum(-2))
        any_v = value_mask.any(-1, keepdim=True).to(v.dtype)
        return self.value_gain * self.value_norm(v) * any_v

    def encode(self, sem: Tensor, batch: Dict[str, Tensor], *, with_time: bool = True) -> Tensor:
        """Fuse the channels -> [..., D]. ``batch`` holds the tensors of :meth:`collate` (on the model device)."""
        x = self.content(sem, batch["has_text"])
        x = x + self.value_channel(batch["values"], batch["value_mask"], batch["name_ids"], batch["unit_ids"],
                                   batch["qkind"], batch["exposure"])
        x = x + self.kind_emb(batch["kind_ids"]) + self.trust_emb(batch["trust_ids"])
        if with_time:
            t = batch["time"]
            gap = torch.zeros_like(t)
            gap[..., 1:] = t[..., 1:] - t[..., :-1]
            first = torch.zeros_like(t, dtype=torch.bool)
            first[..., 0] = True
            x = x + self.gap_proj(gap_features(gap, first))
        return x * batch["pad_mask"].unsqueeze(-1).to(x.dtype)


# --------------------------------------------------------------------------------------------------------------------
# records -> items
# --------------------------------------------------------------------------------------------------------------------
_OBS_KIND = {"document": ItemKind.DOCUMENT, "event": ItemKind.EVENT, "error": ItemKind.FAILURE,
             "software": ItemKind.SOFTWARE, "code": ItemKind.SOFTWARE, "file": ItemKind.DOCUMENT}


def _q(name: str, value: Any, unit: str = "", kind: str = OutcomeKind.REAL, exposure: float = 1.0) -> Quantity:
    return Quantity(name=name, value=float(value), unit=unit, kind=kind, exposure=float(exposure))


def _cost_quantities(cost: Cost, prefix: str) -> List[Quantity]:
    out = []
    for name, unit, kind in (("seconds", "s", OutcomeKind.REAL), ("money", "currency", OutcomeKind.REAL),
                             ("compute_s", "s", OutcomeKind.REAL), ("primitive_ops", "ops", OutcomeKind.COUNT),
                             ("tokens", "tokens", OutcomeKind.COUNT)):
        v = getattr(cost, name)
        if v:
            out.append(_q(f"{prefix}.{name}", v, unit, kind))
    return out


def _chunk(item: ExperienceItem, max_values: int) -> List[ExperienceItem]:
    """Split an item whose quantities exceed one token's value slots into continuation items (same text / ref)."""
    qs = item.quantities
    if max_values <= 0 or len(qs) <= max_values:
        return [item]
    return [ExperienceItem(kind=item.kind, text=item.text, ts=item.ts, quantities=qs[i:i + max_values],
                           ref=item.ref, trust=item.trust) for i in range(0, len(qs), max_values)]


def relation_item(subject: str, predicate: str, obj: str, ts: float = 0.0, ref: str = "",
                  trust: str = Trust.UNTRUSTED, quantities: Optional[List[Quantity]] = None) -> ExperienceItem:
    return ExperienceItem(kind=ItemKind.RELATION, text=f"{subject} | {predicate} | {obj}", ts=ts,
                          quantities=list(quantities or []), ref=ref, trust=trust)


def time_item(now: float, started: Optional[float] = None, deadline: Optional[float] = None,
              ref: str = "") -> ExperienceItem:
    """A clock token: exact elapsed / remaining seconds."""
    qs = []
    if started is not None:
        qs.append(_q("time.elapsed", now - started, "s"))
    if deadline is not None:
        qs.append(_q("time.remaining", deadline - now, "s"))
    return ExperienceItem(kind=ItemKind.TIME, text="clock", ts=now, quantities=qs, ref=ref, trust=Trust.OPERATOR)


def resource_items(state: Dict[str, Any], ts: float = 0.0, ref: str = "", prefix: str = "resource",
                   max_values: int = 4) -> List[ExperienceItem]:
    """Numeric leaves of a resource-state mapping (``EnvironmentAdapter.get_resource_state``) as exact quantities."""
    qs: List[Quantity] = []

    def walk(d: Dict[str, Any], pre: str) -> None:
        for k in sorted(d):
            v = d[k]
            if isinstance(v, dict):
                walk(v, f"{pre}.{k}")
            elif isinstance(v, bool):
                qs.append(_q(f"{pre}.{k}", float(v), kind=OutcomeKind.BINARY))
            elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                qs.append(_q(f"{pre}.{k}", v))

    walk(state, prefix)
    return _chunk(ExperienceItem(kind=ItemKind.RESOURCE, text="resource state", ts=ts, quantities=qs, ref=ref,
                                 trust=Trust.OPERATOR), max_values)


def to_items(record: Any, **kw: Any) -> List[ExperienceItem]:
    """Convert a contract record into experience items.

    Keyword options: ``ts`` (override / supply the timestamp), ``trust`` (override), ``max_values`` (quantity slots
    per token, default 4: longer quantity lists continue in further items), ``max_entities`` (default 4),
    ``query`` (the :class:`OutcomeQuery` an :class:`OutcomeObservation` answers), ``kind`` (item kind for a bare
    string or :class:`Quantity`), ``include_observations`` (ActionResult, default True).

    Supported: Observation, ActionSpec, Decision, ActionResult, OutcomeQuery, OutcomeObservation (with ``query=``)
    or a ``(query, observation)`` pair, PredictionRecord, ToolDescriptor, TaskSpec, Limits, Cost, VerifiedOutcome,
    ObstacleRecord, ExperimentRecord, Quantity, ExperienceItem, str, and lists of these. Values of pending or
    unobservable outcomes are never invented: such items carry no value quantity.
    """
    mv = int(kw.get("max_values", 4))
    ts = kw.get("ts")
    trust = kw.get("trust")

    def item(kind: str, text: str, t: float, qs: Sequence[Quantity], ref: str, tr: str) -> List[ExperienceItem]:
        return _chunk(ExperienceItem(kind=kind, text=text, ts=float(ts if ts is not None else t),
                                     quantities=list(qs), ref=ref, trust=trust or tr), mv)

    if isinstance(record, ExperienceItem):
        return _chunk(record, mv)
    if isinstance(record, (list, tuple)) and len(record) == 2 and isinstance(record[0], OutcomeQuery) \
            and isinstance(record[1], OutcomeObservation):
        return to_items(record[1], **{**kw, "query": record[0]})
    if isinstance(record, (list, tuple)):
        return [it for r in record for it in to_items(r, **kw)]
    if isinstance(record, str):
        return item(kw.get("kind", ItemKind.OBSERVATION), record, 0.0, [], kw.get("ref", ""), Trust.UNTRUSTED)
    if isinstance(record, Quantity):
        return item(kw.get("kind", ItemKind.OBSERVATION), record.name, 0.0, [record], kw.get("ref", ""),
                    Trust.UNTRUSTED)
    if isinstance(record, Observation):
        out = item(_OBS_KIND.get(record.kind, ItemKind.OBSERVATION), record.text, record.ts, record.quantities,
                   record.obs_id, record.provenance.trust)
        for e in record.entities[: int(kw.get("max_entities", 4))]:
            out += item(ItemKind.ENTITY, str(e), record.ts, [], record.obs_id, record.provenance.trust)
        return out
    if isinstance(record, ActionSpec):
        qs = _cost_quantities(record.est_cost, "est_cost")
        if record.steps:
            qs.append(_q("action.steps", len(record.steps), kind=OutcomeKind.COUNT))
        return item(ItemKind.ACTION, record.render(), 0.0, qs, record.action_id, Trust.SELF)
    if isinstance(record, Decision):
        qs = [_q("decision.propensity", record.propensity), _q("decision.candidates", len(record.candidates),
                                                                kind=OutcomeKind.COUNT)]
        return item(ItemKind.ACTION, record.chosen.render(), record.ts, qs + _cost_quantities(
            record.chosen.est_cost, "est_cost"), record.decision_id, Trust.SELF)
    if isinstance(record, ActionResult):
        qs = [_q("result.ok", record.status == ResultStatus.OK, kind=OutcomeKind.BINARY),
              _q("result.duration", max(0.0, record.ended - record.started), "s")]
        if record.verified is not None:
            qs.append(_q("result.verified", record.verified, kind=OutcomeKind.BINARY))
        if record.retries:
            qs.append(_q("result.retries", record.retries, kind=OutcomeKind.COUNT))
        qs += _cost_quantities(record.cost, "cost")
        out = item(ItemKind.ATTEMPT, f"result {record.status}", record.ended, qs, record.action_id, Trust.UNTRUSTED)
        if record.failure_class != FailureClass.NONE or record.status not in (ResultStatus.OK,):
            out += item(ItemKind.FAILURE, f"{record.status} {record.failure_class}: {record.failure_detail}".strip(),
                        record.ended, [], record.action_id, Trust.UNTRUSTED)
        if kw.get("include_observations", True):
            for o in record.observations:
                out += to_items(o, **kw)
        return out
    if isinstance(record, OutcomeQuery):
        text = f"{record.metric}: {record.description} (pending)" if record.description else f"{record.metric} (pending)"
        return item(ItemKind.OUTCOME, text, record.issued_at, [_q("outcome.horizon", record.horizon_s, "s")],
                    record.query_id, Trust.SELF)
    if isinstance(record, OutcomeObservation):
        q: Optional[OutcomeQuery] = kw.get("query")
        metric = q.metric if q is not None else record.query_id
        desc = f": {q.description}" if q is not None and q.description else ""
        qs = []
        if record.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) and record.value is not None:
            qs.append(_q(metric, record.value, kind=q.kind if q is not None else OutcomeKind.REAL,
                         exposure=q.exposure if q is not None else 1.0))
            if record.status == OutcomeStatus.CENSORED:
                qs.append(_q("outcome.censored", 1.0, kind=OutcomeKind.BINARY))
            if record.uncertainty:
                qs.append(_q("outcome.stderr", record.uncertainty))
        if q is not None:
            qs.append(_q("outcome.horizon", q.horizon_s, "s"))
        return item(ItemKind.OUTCOME, f"{metric}{desc} ({record.status})", record.observed_at, qs, record.query_id,
                    Trust.VERIFIER if record.verified else Trust.UNTRUSTED)
    if isinstance(record, PredictionRecord):
        d = record.dist
        qs = [_q(f"pred.{n}", v) for n, v in (("mean", d.mean), ("std", d.std)) if v is not None]
        qs.append(_q("pred.horizon", record.query.horizon_s, "s"))
        if record.resolved_value is not None and record.resolved_status in (OutcomeStatus.OBSERVED,
                                                                            OutcomeStatus.CENSORED):
            qs.append(_q(record.query.metric, record.resolved_value, kind=record.query.kind))
        out = item(ItemKind.PREDICTION, f"prediction of {record.query.metric} [{d.family}]", record.made_at, qs,
                   record.prediction_id, Trust.SELF)
        if d.epistemic_std is not None and kw.get("with_uncertainty", True):
            out += item(ItemKind.UNCERTAINTY, f"uncertainty about {record.query.metric}", record.made_at,
                        [_q("pred.epistemic_std", d.epistemic_std)], record.prediction_id, Trust.SELF)
        return out
    if isinstance(record, ToolDescriptor):
        args = ", ".join(sorted((record.arg_schema.get("properties") or {}).keys()))
        text = f"tool {record.name} [{record.modality}/{record.level}] {record.description} args: {args}" \
               f" effects: {record.side_effects}"
        if not record.available:
            text += f" unavailable: {record.unavailable_reason}"
        qs = [_q("tool.available", record.available, kind=OutcomeKind.BINARY)] + _cost_quantities(record.est_cost,
                                                                                                    "est_cost")
        return item(ItemKind.TOOL, text, 0.0, qs, record.name, Trust.OPERATOR)
    if isinstance(record, TaskSpec):
        out = item(ItemKind.GOAL, record.objective, 0.0, [], record.objective_id, Trust.OPERATOR)
        return out + to_items(record.limits, **{**kw, "ref": record.objective_id})
    if isinstance(record, Limits):
        qs = [_q(f"limit.{n}", getattr(record, n)) for n in ("duration_s", "money", "compute_s",
                                                             "max_external_actions", "max_decisions")
              if getattr(record, n)]
        return item(ItemKind.RESOURCE, "limits", 0.0, qs, kw.get("ref", ""), Trust.OPERATOR) if qs else []
    if isinstance(record, Cost):
        return item(ItemKind.COST, "cost", 0.0, _cost_quantities(record, "cost"), kw.get("ref", ""), Trust.UNTRUSTED)
    if isinstance(record, VerifiedOutcome):
        qs = [_q("objective.value", record.value), _q("objective.target", record.target)]
        if record.uncertainty:
            qs.append(_q("objective.stderr", record.uncertainty))
        return item(ItemKind.OUTCOME, f"objective {record.status}", record.evaluated_at,
                    qs + list(record.secondary), record.objective_id, Trust.VERIFIER)
    if isinstance(record, ObstacleRecord):
        return item(ItemKind.FAILURE, f"obstacle {record.failure_class}: {record.description}", record.ts, [],
                    record.obstacle_id, Trust.SELF)
    if isinstance(record, ExperimentRecord):
        text = f"experiment [{record.status}] question: {record.question} hypothesis: {record.hypothesis}"
        qs = [_q("experiment.arms", len(record.arms), kind=OutcomeKind.COUNT),
              _q("experiment.horizon", record.horizon_s, "s")]
        if record.supported is not None:
            qs.append(_q("experiment.supported", record.supported, kind=OutcomeKind.BINARY))
        return item(ItemKind.EVENT, text, record.created_at, qs, record.experiment_id, Trust.SELF)
    raise TypeError(f"to_items: unsupported record type {type(record).__name__}")


def action_meta(action: ActionSpec) -> Dict[str, Any]:
    """Structured (non-text) fields of an action used by the action encoder."""
    return {"modality": action.modality, "level": action.level, "timeout_s": float(action.timeout_s),
            "n_steps": len(action.steps), "args": json.dumps(action.arguments, sort_keys=True, default=str)}
