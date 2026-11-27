"""cam_live.predict.providers — Workstream B's trainable objectives as ``learning_api.LossProvider``s.

    WorldModelLoss        NLL of the matured outcomes of the executed action through
                          encode_decisions -> z -> encode_queries -> features -> WorldHeads, plus the Bayesian fast
                          layer's prior-predictive NLL of the same outcomes (head as offset) and the latent-consistency
                          term of the transition head. Pending and unobservable outcomes never reach the loss
                          (``learning_api.matured_outcomes``); kinds are balanced by a running scale.
    FastPriorLoss         ALPaCA-style: within one world context (objective) the outcomes are scored prequentially
                          through Workstream A's differentiable posterior (``nn.fast.online_nll``), training the
                          feature map and the learned prior so that a few observations adapt predictions.
    EvaluatorLoss         BCE-with-logits (success) + magnitude / time / resource likelihoods against *signed
                          verified* objective outcomes, applied to every earlier prefix (V on the belief and Q on the
                          executed candidate). Unsigned outcomes are refused; pending objectives give no target.
    WorldModelAdapterLoss / EvaluatorAdapterLoss
                          the same objectives on the slow timescale with a graph into the semantic adapters
                          (``tokenize(with_grad=True, grad_last=k)``), so LoRA receives gradient from prediction.

Providers hold no optimizer. They never read or write a reward and never touch the ``reward`` parameter group.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..contracts import (ExperienceRecord, ExperienceStore, GoalStatus, OutcomeObservation, OutcomeQuery,
                         VerifiedOutcome)
from ..learning_api import (TIMESCALE_FREQUENT, TIMESCALE_SLOW, DecisionBatch, LossProvider, matured_outcomes)
from ..nn.api import LatentModelAPI
from .evaluator import EvalTargets, UnverifiedOutcomeError, evaluator_loss, prefix_targets_from_outcome
from .heads import (F64, FAST_KINDS, PredictHeads, WorldBatch, canonical_query, fast_inputs, fast_prior_module,
                    head_offsets, world_batch)
from .imagination import refuse_imagined
from .world import KIND_INDEX, bootstrap_weights, transition_nll, world_nll

Row = List[Tuple[OutcomeQuery, Optional[OutcomeObservation]]]


def record_rows(records: Sequence[ExperienceRecord]) -> List[Row]:
    """Matured (query, observation) pairs of each record's executed action, with canonical metric names."""
    rows: List[Row] = []
    for r in records:
        ids = [r.decision.chosen.action_id] + [c.action.action_id for c in r.decision.candidates]
        rows.append([(canonical_query(q, ids), o) for q, o in matured_outcomes(r)])
    return rows


def _balanced(x: Tensor) -> Tensor:
    """x / max(1, |x|): a numerically large likelihood cannot own the shared trunk's gradient."""
    return x / max(1.0, abs(float(x.detach())))


def _next_belief(model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore) -> Tuple[Tensor, Tensor]:
    """(b_next [B, D] detached, has_next [B]) — from the batch when it was encoded ``with_next``, else computed."""
    if batch.b_next is not None and batch.has_next is not None:
        return batch.b_next.detach(), batch.has_next
    has = [bool(r.post_item_refs) for r in batch.records]
    mask = torch.tensor(has, device=batch.b.device)
    if not any(has):
        return torch.zeros_like(batch.b), mask
    with torch.no_grad():
        tok = model.tokenize([store.items(r.post_item_refs if h else r.context_item_refs)
                              for r, h in zip(batch.records, has)])
        b_next = model.core_forward(tok, None).b
    return b_next, mask


def encode_with_adapters(model: LatentModelAPI, store: ExperienceStore, records: Sequence[ExperienceRecord], *,
                         grad_last: int = 2, with_next: bool = True) -> DecisionBatch:
    """A ``DecisionBatch`` of the *executed* action of each record (N = 1) whose graph reaches the semantic
    adapters: the newest ``grad_last`` context items, the action and the goal are embedded with gradient; older
    items use the cached, detached embeddings (bounded memory)."""
    records = list(records)
    contexts = [store.items(r.context_item_refs) for r in records]
    try:
        tokens = model.tokenize(contexts, with_grad=True, grad_last=grad_last)      # CamLiveModel
    except TypeError:
        tokens = model.tokenize(contexts, with_grad=True)                            # plain LatentModelAPI
    out = model.core_forward(tokens, None)
    e_a, a_mask = model.encode_actions([[r.decision.chosen] for r in records], with_grad=True)
    e_g = model.encode_goals([r.goal_text for r in records], with_grad=True)
    z = model.decision_latent(out.b, e_a, e_g)
    dev = out.b.device
    prop = torch.tensor([r.decision.propensity for r in records], device=dev, dtype=torch.float32)
    batch = DecisionBatch(records=records, b=out.b, h=out.h, pad_mask=tokens.pad_mask, e_a=e_a, a_mask=a_mask,
                          e_g=e_g, z=z, chosen=torch.zeros(len(records), dtype=torch.long, device=dev),
                          propensity=prop, behaviour_probs=prop[:, None],
                          duration_s=torch.tensor([r.duration_s for r in records], dtype=torch.float64, device=dev),
                          ts=torch.tensor([r.decision.ts for r in records], dtype=torch.float64, device=dev))
    if with_next:
        batch.b_next, batch.has_next = _next_belief(model, batch, store)
    batch.with_grad_semantic = True                # type: ignore[attr-defined]
    return batch


class _Base:
    name = "predict"
    timescale = TIMESCALE_FREQUENT
    weight = 1.0
    #: how the learner should encode batches for this provider (``encode_decisions`` keyword arguments)
    batch_kwargs: Dict[str, Any] = {"with_next": False, "with_grad_semantic": False}

    def __init__(self, heads: PredictHeads, weight: Optional[float] = None,
                 allowed_provenance: Optional[Sequence[str]] = None):
        self.heads = heads
        if weight is not None:
            self.weight = float(weight)
        self.allowed_provenance = tuple(allowed_provenance or heads.cfg.allowed_provenance)

    def _permitted(self, record: ExperienceRecord) -> bool:
        return record.provenance in self.allowed_provenance and record.eligible.get(self.name, True)


# ----------------------------------------------------------------------------------------------------------------
# world model
# ----------------------------------------------------------------------------------------------------------------
class WorldModelLoss(_Base):
    name = "world_model"
    batch_kwargs = {"with_next": True, "with_grad_semantic": False}

    def __init__(self, heads: PredictHeads, weight: Optional[float] = None, *,
                 kind_weights: Optional[Dict[str, float]] = None, fast_weight: Optional[float] = None,
                 transition_weight: Optional[float] = None, allowed_provenance: Optional[Sequence[str]] = None):
        super().__init__(heads, weight, allowed_provenance)
        self.kind_weights = dict(kind_weights or {})
        self.fast_weight = heads.cfg.fast_prior_weight if fast_weight is None else float(fast_weight)
        self.transition_weight = heads.cfg.transition_weight if transition_weight is None else float(transition_weight)

    def eligible(self, record: ExperienceRecord) -> bool:
        if not self._permitted(record) or record.result is None:
            return False
        return bool(matured_outcomes(record)) or bool(record.post_item_refs)

    def fast_term(self, model: LatentModelAPI, wb: WorldBatch) -> Tuple[Optional[Tensor], Dict[str, float]]:
        """Prior-predictive NLL of the matured outcomes under the learned fast prior (no within-context update)."""
        prior_mod = fast_prior_module(model)
        if prior_mod is None:
            return None, {}
        from ..nn.fast import online_nll
        off, mask = head_offsets(wb), wb.mask()
        total: Optional[Tensor] = None
        metrics: Dict[str, float] = {}
        for kind in FAST_KINDS:
            if kind not in prior_mod.kinds:
                continue
            km = mask & (wb.kind_ids == KIND_INDEX[kind])
            if not bool(km.any()):
                continue
            y, ex, ce = fast_inputs(kind, wb, off)
            nll = online_nll(prior_mod.prior(kind), wb.phi[:, 0][km].to(F64)[:, None, :], y[km][:, None], kind,
                             exposure=None if ex is None else ex[km][:, None],
                             censored=None if ce is None else ce[km][:, None], noise_mode=prior_mod.noise_mode)
            v = nll.mean()
            metrics[f"fast_prior_nll_{kind}"] = float(v.detach())
            total = _balanced(v) if total is None else total + _balanced(v)
        return total, metrics

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore
             ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        recs = batch.records
        rows = record_rows(recs)
        n_targets = sum(len(r) for r in rows)
        b_next, has_next = _next_belief(model, batch, store)
        if n_targets == 0 and not bool(has_next.any()):
            return None, {"world_n_targets": 0.0}
        z = batch.z_chosen
        ids = [r.exp_id for r in recs]
        seed = self.heads.cfg.seed
        total = z.sum() * 0.0
        metrics: Dict[str, float] = {"world_n_targets": float(n_targets)}
        if n_targets:
            wb = world_batch(model, self.heads, z, rows, update_scaler=torch.is_grad_enabled())
            mw = bootstrap_weights(ids, self.heads.world.cfg.n_members, seed).to(z.device)
            l_w, m = world_nll(wb.out, wb.targets, q_mask=wb.q_mask, member_weights=mw,
                               kind_weights=self.kind_weights, normaliser=self.heads.normaliser)
            total = total + l_w
            metrics.update({f"world_{k}": v for k, v in m.items()})
            if self.fast_weight > 0:
                l_f, mf = self.fast_term(model, wb)
                if l_f is not None:
                    total = total + self.fast_weight * l_f.to(total.dtype)
                    metrics.update(mf)
        if self.transition_weight > 0 and bool(has_next.any()):
            dt = batch.duration_s if batch.duration_s is not None else torch.zeros(len(recs), device=z.device)
            tout = self.heads.transition(z, dt, batch.b)
            tw = bootstrap_weights(ids, self.heads.transition.n_members, seed + 17).to(z.device)
            l_t, mt = transition_nll(tout, b_next, mask=has_next, member_weights=tw)
            total = total + self.transition_weight * _balanced(l_t)
            metrics.update(mt)
        metrics["world_model_loss"] = float(total.detach())
        return total, metrics


class FastPriorLoss(_Base):
    """Prequential NLL through the differentiable fast posterior, one sequence per (objective, outcome kind).
    A batch should contain several records of the same objective: with a single row per objective only the
    prior-predictive term remains."""
    name = "fast_prior"
    weight = 0.5

    def __init__(self, heads: PredictHeads, weight: Optional[float] = None, *, max_rows: int = 48,
                 allowed_provenance: Optional[Sequence[str]] = None):
        super().__init__(heads, weight, allowed_provenance)
        self.max_rows = int(max_rows)

    def eligible(self, record: ExperienceRecord) -> bool:
        return self._permitted(record) and any(q.kind in FAST_KINDS for q, _ in matured_outcomes(record))

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore
             ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        prior_mod = fast_prior_module(model)
        if prior_mod is None:
            return None, {"fast_prior_available": 0.0}
        from ..nn.fast import online_nll
        recs = batch.records
        rows = record_rows(recs)
        if not any(rows):
            return None, {"fast_prior_n": 0.0}
        wb = world_batch(model, self.heads, batch.z_chosen, rows)
        off, mask = head_offsets(wb), wb.mask()
        groups: Dict[str, List[int]] = {}
        for i, r in enumerate(recs):
            groups.setdefault(r.objective_id or r.window_id or r.exp_id, []).append(i)
        dev = wb.phi.device
        total: Optional[Tensor] = None
        metrics: Dict[str, float] = {}
        n_all = 0
        for kind in FAST_KINDS:
            if kind not in prior_mod.kinds:
                continue
            km = (mask & (wb.kind_ids == KIND_INDEX[kind])).cpu()
            seqs: List[List[Tuple[int, int]]] = []
            for idxs in groups.values():
                ent = sorted((rows[i][j][1].observed_at, i, j) for i in idxs for j in range(len(rows[i]))
                             if bool(km[i, j]))
                if ent:
                    seqs.append([(i, j) for _, i, j in ent[:self.max_rows]])
            if not seqs:
                continue
            g, t = len(seqs), max(len(s) for s in seqs)
            bi = torch.zeros(g, t, dtype=torch.long)
            qi = torch.zeros(g, t, dtype=torch.long)
            m = torch.zeros(g, t, dtype=torch.bool)
            for a, s in enumerate(seqs):
                for c, (i, j) in enumerate(s):
                    bi[a, c], qi[a, c], m[a, c] = i, j, True
            bi, qi, m = bi.to(dev), qi.to(dev), m.to(dev)
            y, ex, ce = fast_inputs(kind, wb, off)
            nll = online_nll(prior_mod.prior(kind), wb.phi[:, 0][bi, qi].to(F64), y[bi, qi], kind, mask=m,
                             exposure=None if ex is None else ex[bi, qi],
                             censored=None if ce is None else ce[bi, qi], noise_mode=prior_mod.noise_mode)
            n = int(m.sum())
            v = nll.sum() / max(n, 1)
            n_all += n
            metrics[f"fast_online_nll_{kind}"] = float(v.detach())
            metrics[f"fast_online_nll_first_{kind}"] = float(nll[:, 0].sum().detach() / max(1, int(m[:, 0].sum())))
            if t > 1 and bool(m[:, 1:].any()):
                metrics[f"fast_online_nll_later_{kind}"] = float(nll[:, 1:].sum().detach() / int(m[:, 1:].sum()))
            total = _balanced(v) if total is None else total + _balanced(v)
        metrics["fast_prior_n"] = float(n_all)
        if total is None:
            return None, metrics
        return total.to(batch.z.dtype), metrics


# ----------------------------------------------------------------------------------------------------------------
# evaluator
# ----------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class _Objective:
    outcome: VerifiedOutcome
    deadline: float
    achieved_at: Optional[float]
    magnitude_is_count: bool
    spent: Dict[str, Tuple[float, ...]]          # exp_id -> cumulative (seconds, money, compute) before the decision
    final: Optional[Tuple[float, ...]]


class EvaluatorLoss(_Base):
    """``check_signature`` is the independent verifier's check (``OutcomeVerifier.check_signature``). An outcome
    that fails it is refused with :class:`UnverifiedOutcomeError` and supervises nothing."""
    name = "evaluator"
    batch_kwargs = {"with_next": False, "with_grad_semantic": False}

    def __init__(self, heads: PredictHeads, check_signature: Callable[[VerifiedOutcome], bool],
                 weight: Optional[float] = None, *, allowed_provenance: Optional[Sequence[str]] = None,
                 term_weights: Optional[Dict[str, float]] = None):
        super().__init__(heads, weight, allowed_provenance)
        self._check = check_signature
        self._objectives: Dict[str, _Objective] = {}
        self.term_weights = dict(term_weights or {})

    def add_outcome(self, outcome: VerifiedOutcome, *, deadline: float, achieved_at: Optional[float] = None,
                    trajectory: Optional[Sequence[ExperienceRecord]] = None, magnitude_is_count: bool = True
                    ) -> bool:
        """Register the verified outcome of an objective. Returns True when it supervises (achieved / failed);
        a pending outcome is remembered as pending and supervises nothing. ``trajectory`` (the objective's
        records in order) adds remaining-expenditure targets for the resource head."""
        refuse_imagined(outcome, "objective outcome")
        if not isinstance(outcome, VerifiedOutcome):
            raise UnverifiedOutcomeError("only a contracts.VerifiedOutcome can supervise the evaluator")
        if not outcome.signature or not self._check(outcome):
            raise UnverifiedOutcomeError("the verifier's signature on this outcome did not check")
        spent: Dict[str, Tuple[float, ...]] = {}
        final: Optional[Tuple[float, ...]] = None
        if trajectory:
            acc = [0.0, 0.0, 0.0]
            for r in sorted(trajectory, key=lambda r: r.decision.ts):
                spent[r.exp_id] = tuple(acc)
                if r.result is not None:
                    acc = [acc[0] + r.result.cost.seconds, acc[1] + r.result.cost.money,
                           acc[2] + r.result.cost.compute_s]
            final = tuple(acc)
        self._objectives[outcome.objective_id] = _Objective(outcome, float(deadline), achieved_at,
                                                            magnitude_is_count, spent, final)
        return outcome.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED)

    def objective(self, objective_id: str) -> Optional[VerifiedOutcome]:
        o = self._objectives.get(objective_id)
        return None if o is None else o.outcome

    def eligible(self, record: ExperienceRecord) -> bool:
        o = self._objectives.get(record.objective_id)
        return (self._permitted(record) and o is not None
                and o.outcome.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED)
                and record.decision.ts < o.outcome.evaluated_at)

    def targets(self, records: Sequence[ExperienceRecord]) -> Tuple[Tensor, EvalTargets, Tensor]:
        """(horizon_s [B], targets [B], scale [B]) for a batch of prefixes (records)."""
        nres = self.heads.evaluator.cfg.n_resources
        hs, ts, sc = [], [], []
        per_obj: Dict[str, int] = {}
        for r in records:
            if self.eligible(r):
                per_obj[r.objective_id] = per_obj.get(r.objective_id, 0) + 1
        for r in records:
            o = self._objectives.get(r.objective_id)
            if o is None or not self.eligible(r):
                h, t = torch.zeros(1, dtype=torch.float64), _empty_targets(nres)
                scale = 1.0
            else:
                kw: Dict[str, Any] = {}
                if o.final is not None and r.exp_id in o.spent:
                    kw = {"resources_final": o.final, "resources_at_prefix": [o.spent[r.exp_id]]}
                h, t = prefix_targets_from_outcome(o.outcome, [r.decision.ts], deadline=o.deadline,
                                                   signature_checked=True, achieved_at=o.achieved_at,
                                                   magnitude_is_count=o.magnitude_is_count, n_resources=nres, **kw)
                if o.outcome.status == GoalStatus.ACHIEVED and o.achieved_at is None:
                    t.time_mask = torch.zeros_like(t.time_mask)        # achievement time unknown: no time target
                t.weight = t.weight / per_obj[r.objective_id]          # one objective = one unit of evidence
                scale = max(float(o.outcome.target), 1.0)
            hs.append(h)
            ts.append(t)
            sc.append(scale)
        return torch.cat(hs), EvalTargets.cat(ts), torch.tensor(sc, dtype=torch.float32)

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore
             ) -> Tuple[Optional[Tensor], Dict[str, float]]:
        horizon, tgt, scale = self.targets(batch.records)
        if not bool(tgt.success_mask.any()):
            return None, {"eval_success_n": 0.0}
        dev = batch.b.device
        tgt = EvalTargets(*(None if getattr(tgt, n) is None else getattr(tgt, n).to(dev) for n in EvalTargets._FIELDS))
        horizon, scale = horizon.to(dev), scale.to(dev)
        res_scale = torch.tensor(self.heads.cfg.resource_scale, device=dev)
        ev = self.heads.evaluator
        v_out = ev(batch.b, batch.e_g, horizon, scale=scale)
        l_v, m_v = evaluator_loss(v_out, tgt, self.term_weights, res_scale)
        q_out = ev.forward_candidates(batch.z_chosen.unsqueeze(1), batch.e_g, horizon, scale=scale)
        q_tgt = EvalTargets(*(None if getattr(tgt, n) is None else getattr(tgt, n).unsqueeze(1)
                              for n in EvalTargets._FIELDS))
        l_q, m_q = evaluator_loss(q_out, q_tgt, self.term_weights, res_scale)
        metrics = dict(m_v)
        metrics.update({f"q_{k}": v for k, v in m_q.items()})
        with torch.no_grad():
            m = tgt.success_mask
            p = v_out.p_success[m]
            metrics["eval_brier"] = float(((p - tgt.success[m].to(p.dtype)) ** 2).mean())
        loss = 0.5 * (l_v + l_q)
        metrics["evaluator_loss"] = float(loss.detach())
        return loss, metrics


def _empty_targets(n_resources: int) -> EvalTargets:
    z, f = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    return EvalTargets(success=z.clone(), success_mask=f.clone(), magnitude=z.clone(), magnitude_mask=f.clone(),
                       magnitude_is_count=torch.ones(1, dtype=torch.bool), time=z.clone(), time_event=f.clone(),
                       time_mask=f.clone(), resources=torch.zeros(1, n_resources),
                       resources_mask=torch.zeros(1, n_resources, dtype=torch.bool), weight=z.clone())


# ----------------------------------------------------------------------------------------------------------------
# slow timescale: the same objectives with gradient into the semantic adapters
# ----------------------------------------------------------------------------------------------------------------
class _AdapterMixin:
    timescale = TIMESCALE_SLOW
    batch_kwargs = {"with_next": True, "with_grad_semantic": True}

    def _adapter_batch(self, model: LatentModelAPI, batch: DecisionBatch, store: ExperienceStore) -> DecisionBatch:
        cfg = self.heads.cfg                                        # type: ignore[attr-defined]
        if getattr(batch, "with_grad_semantic", False):
            return batch
        return encode_with_adapters(model, store, batch.records[:cfg.adapter_max_records],
                                    grad_last=cfg.adapter_grad_last)


class WorldModelAdapterLoss(_AdapterMixin, WorldModelLoss):
    """World-model NLL on the slow timescale. If the batch was not encoded with a graph into the adapters it is
    re-encoded here for at most ``adapter_max_records`` records with ``adapter_grad_last`` items of gradient."""
    name = "world_model_adapters"
    weight = 0.5

    def loss(self, model, batch, store):
        return super().loss(model, self._adapter_batch(model, batch, store), store)


class EvaluatorAdapterLoss(_AdapterMixin, EvaluatorLoss):
    name = "evaluator_adapters"
    weight = 0.5

    def loss(self, model, batch, store):
        return super().loss(model, self._adapter_batch(model, batch, store), store)


def default_providers(heads: PredictHeads, check_signature: Optional[Callable[[VerifiedOutcome], bool]] = None, *,
                      adapters: bool = True) -> List[LossProvider]:
    """The providers the integrated agent registers with the continual learner. The evaluator providers are
    included only when the verifier's signature check is supplied."""
    out: List[Any] = [WorldModelLoss(heads), FastPriorLoss(heads)]
    if check_signature is not None:
        out.append(EvaluatorLoss(heads, check_signature))
    if adapters:
        out.append(WorldModelAdapterLoss(heads))
        if check_signature is not None:
            ev = EvaluatorAdapterLoss(heads, check_signature)
            ev._objectives = out[2]._objectives                    # one shared table of verified outcomes
            out.append(ev)
    return out
