"""cam_live.predict.predictor — the runtime-facing prediction API of Workstream B (spec Parts 4, 6, 8).

    Predictor(model, fast_state, registry, clock, cfg)
        .evaluate_candidates(context_items, goal_text, candidates, queries) -> List[CandidateEvaluation]
        .observe_outcomes(outcomes)   immediate fast-belief update from matured outcomes -> surprise, belief change
        .resolve(outcomes)            fills the registered prediction records
        .resolve_objective(verified)  resolves the prospectively registered success predictions

Every prediction is written to the hash-chained ``PredictionRegistry`` *before* ``evaluate_candidates`` returns,
with ``made_at`` taken from the clock; the registry refuses any resolution that is not strictly later.

A prediction for (candidate, query) combines the world-model heads (a bootstrapped ensemble: amortised knowledge,
epistemic = member spread) with Workstream A's Bayesian fast layer (within-lifetime knowledge, epistemic =
posterior variance) as defined in :mod:`cam_live.predict.heads`; the two epistemic parts are added as variances of
independent factors and reported next to the aleatoric part. Value of information is the decision-relevant
quantity  E_y[max_a' Q(b_updated(a, y), a')] - max_a' Q(b, a')  computed by sampling outcomes from the predictive
and applying the fast layer's own update to a copy (optionally also the experiential core's update, by letting the
imagined result enter a copy of the context). It is estimated as the Jensen gap  E_y[max_a' Q_y] - max_a' E_y[Q_y]
over the same samples, which equals the formula above for an exact Bayesian update and does not mistake the bias of
an approximate update for information. Nothing here returns or consumes a reward.
"""
from __future__ import annotations

import collections
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..contracts import (ActionSpec, CandidateEvaluation, Clock, Cost, Distribution, ExperienceItem, GoalStatus,
                         ItemKind, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus, PredictionRecord,
                         Quantity, Trust, VerifiedOutcome, new_id)
from ..nn.api import LatentModelAPI
from . import dists as D
from .errors import ChangeDetector, classify_error
from .evaluator import UnverifiedOutcomeError
from .heads import (ACTION_PLACEHOLDER, F64, K_BINARY, K_COUNT, K_REAL, K_TTE, KIND_OF_ID, PredictConfig,
                    PredictHeads, VOI_SIGNALS, WorldBatch, attach_heads, canonical_query, world_batch)
from .imagination import refuse_imagined
from .registry import AlreadyResolvedError, PredictionRegistry

QuerySpec = Union[Sequence[OutcomeQuery], Mapping[str, Sequence[OutcomeQuery]],
                  Callable[[ActionSpec], Sequence[OutcomeQuery]]]
THOMPSON_KEY = "ts_"                      # extras["ts_<i>"]: joint posterior samples of the candidate's goal value


def at_horizons(query: OutcomeQuery, horizons_s: Sequence[float]) -> List[OutcomeQuery]:
    """The same question asked at several horizons (templates; ids are assigned when they are instantiated)."""
    return [dataclasses.replace(query, query_id="", horizon_s=float(h)) for h in horizons_s]


@dataclass
class _Cached:
    """What a later outcome needs in order to update the fast belief and be attributed (one per prediction)."""
    prediction_id: str
    query: OutcomeQuery
    metric: str                      # canonical
    action_id: str
    decision_id: str
    phi: Tensor                      # [F] float64, cpu
    head_count: float                # head mean count (includes exposure): the fast layer's exposure offset
    head_loc: float                  # head location, normalised units (real)
    head_time: float                 # head expected time, seconds
    real_loc: float
    real_scale: float
    mean: float
    aleatoric_std: float
    epistemic_std: float


@dataclass
class _View:
    """Head-side tensors of a world batch, float64, candidate axis removed: [M, R, Q] / [R, Q] / [R, Q, F]."""
    count: Tensor
    loc: Tensor
    logit: Tensor
    time: Tensor
    cat: Tensor
    phi: Tensor
    kind_ids: Tensor
    real_loc: Tensor
    real_scale: Tensor


class Predictor:
    def __init__(self, model: LatentModelAPI, fast_state: Any, registry: PredictionRegistry, clock: Clock,
                 cfg: Any = None, *, heads: Optional[PredictHeads] = None, model_version: str = "",
                 checkpoint_id: str = "", max_cache: int = 50_000):
        self.model, self.fast, self.registry, self.clock = model, fast_state, registry, clock
        pcfg = cfg if isinstance(cfg, PredictConfig) else PredictConfig.from_cfg(cfg or getattr(model, "cfg", None))
        self.heads = heads or attach_heads(model, pcfg)
        self.cfg = self.heads.cfg
        self.model_version, self.checkpoint_id = model_version, checkpoint_id
        self.max_cache = int(max_cache)
        self._cache: Dict[str, _Cached] = {}                # query id -> cached prediction
        self._alias: Dict[str, str] = {}                    # environment query id -> predicted query id
        self._absorbed: set = set()                         # dedup keys already used for a fast update
        self._detectors: Dict[str, ChangeDetector] = {}
        self._success: Dict[str, List[str]] = {}            # objective id -> prediction ids of success forecasts
        self._ref = None
        self.set_fast_state(fast_state)
        self.last: Dict[str, Any] = {}                      # tensors of the latest evaluation (diagnostics)
        # LRN-3: per decision, the decision-time head view of the goal-relevant predictions (chosen row after
        # commit), so the goal-value epistemic std can be recomputed after the fast layer absorbed the decision's
        # own outcomes without another network pass (bounded; diagnostics / reward inputs only)
        self._goal_views: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        self.max_goal_views = 512

    def set_fast_state(self, fast_state: Any) -> None:
        """Use another fast belief state (e.g. a fresh one at the start of a new lifetime)."""
        self.fast = fast_state
        self._ref = None
        if fast_state is not None:
            self._ref = fast_state.clone()
            self._ref.reset()                               # the prior, for the binary logit shift

    # ---- queries ------------------------------------------------------------------------------------------------
    def _instantiate(self, queries: Optional[QuerySpec], candidates: Sequence[ActionSpec], now: float,
                     decision_id: str) -> List[List[OutcomeQuery]]:
        out: List[List[OutcomeQuery]] = []
        for a in candidates:
            if queries is None:
                tmpl: Sequence[OutcomeQuery] = ()
            elif callable(queries):
                tmpl = queries(a)
            elif isinstance(queries, Mapping):
                tmpl = queries.get(a.action_id, ())
            else:
                tmpl = queries
            row = []
            for t in tmpl:
                issued = t.issued_at if t.issued_at > 0 else now + max(float(a.est_cost.seconds), 0.0)
                row.append(dataclasses.replace(
                    t, query_id=new_id("q"), issued_at=issued, decision_id=decision_id,
                    experiment_id=t.experiment_id or a.experiment_id,
                    metric=t.metric.replace(ACTION_PLACEHOLDER, a.action_id), categories=list(t.categories)))
            out.append(row)
        return out

    # ---- head + fast layer --------------------------------------------------------------------------------------
    def _view(self, wb: WorldBatch) -> _View:
        o = wb.out
        sq = lambda t: t[:, :, 0].to(F64)                                                  # noqa: E731
        edges = o._edges()[:, :, 0]
        return _View(count=torch.exp(sq(o.log_rate)) * wb.meta.exposure.to(F64)[None], loc=sq(o.loc),
                     logit=sq(o.logit), time=D.hazard_expected_time(o.hazard_logits[:, :, 0], edges).to(F64),
                     cat=o._cat_mode_prob()[:, :, 0].to(F64), phi=wb.phi[:, 0].to(F64), kind_ids=wb.kind_ids,
                     real_loc=wb.real_loc.to(F64), real_scale=wb.real_scale.to(F64))

    def _supports(self, fast: Any, kind: str) -> bool:
        return fast is not None and fast.supports(kind)

    def _means(self, v: _View, fast: Any) -> Tuple[Tensor, Tensor]:
        """Combined predictive mean and epistemic variance, [R, Q] float64, in each query's own units."""
        var0 = lambda t: t.var(0) if t.shape[0] > 1 else torch.zeros_like(t[0])            # noqa: E731
        kid, phi = v.kind_ids, v.phi
        one, zero = torch.ones_like(v.time[0]), torch.zeros_like(v.time[0])
        present = {KIND_OF_ID[int(k)] for k in torch.unique(kid)}
        sup = lambda kind: kind in present and self._supports(fast, kind)                  # noqa: E731
        # real
        f_m, f_v = (zero, zero)
        if sup(OutcomeKind.REAL):
            f_m, f_v, _ = fast.predict(phi, OutcomeKind.REAL)
        mean = (v.loc.mean(0) + f_m) * v.real_scale + v.real_loc
        epi = (var0(v.loc) + f_v) * v.real_scale ** 2
        # count: head mean x fast multiplier (independent factors)
        f_m, f_v = (one, zero)
        if sup(OutcomeKind.COUNT):
            f_m, f_v, _ = fast.predict(phi, OutcomeKind.COUNT)
        hm, hv = v.count.mean(0), var0(v.count)
        mean = torch.where(kid == K_COUNT, hm * f_m, mean)
        epi = torch.where(kid == K_COUNT, hv * f_m ** 2 + hm ** 2 * f_v + hv * f_v, epi)
        # binary: posterior shift of the fast logit added to the head logit
        shift, lin_v = zero, zero
        if sup(OutcomeKind.BINARY):
            lin_m, lin_v = fast.linear(phi, OutcomeKind.BINARY)
            shift = lin_m - self._ref.linear(phi, OutcomeKind.BINARY)[0]
        p_m = torch.sigmoid(v.logit + shift[None])
        p = p_m.mean(0)
        mean = torch.where(kid == K_BINARY, p, mean)
        epi = torch.where(kid == K_BINARY, var0(p_m) + (p * (1 - p)) ** 2 * lin_v, epi)
        # time to event: head expected time x fast expected rescaled time
        f_m, f_v = (one, zero)
        if sup(OutcomeKind.TIME_TO_EVENT):
            f_m, f_v, _ = fast.predict(phi, OutcomeKind.TIME_TO_EVENT)
        tm, tv = v.time.mean(0), var0(v.time)
        mean = torch.where(kid == K_TTE, tm * f_m, mean)
        epi = torch.where(kid == K_TTE, tv * f_m ** 2 + tm ** 2 * f_v + tv * f_v, epi)
        # categorical: head only
        cat = kid == max(KIND_OF_ID)
        mean = torch.where(cat, v.cat.mean(0), mean)
        epi = torch.where(cat, var0(v.cat), epi)
        return mean, epi.clamp_min(0.0)

    def _samples(self, v: _View, fast: Any, n: int, generator: Optional[torch.Generator]) -> Tensor:
        """[n, R, Q] joint posterior samples of the predictive means: one ensemble member and one draw of the
        fast-layer weights per sample, shared by every candidate (coherent comparisons)."""
        m = v.count.shape[0]
        idx = torch.randint(0, m, (n,), generator=generator)
        kid, phi = v.kind_ids, v.phi
        w: Dict[str, Tensor] = {}
        for kind in (OutcomeKind.REAL, OutcomeKind.COUNT, OutcomeKind.BINARY, OutcomeKind.TIME_TO_EVENT):
            if self._supports(fast, kind):
                w[kind] = fast.sample_weights(n, kind, generator).to(phi.device)                 # [n, F]
        lin = lambda kind: torch.einsum("rqf,sf->srq", phi, w[kind]) if kind in w else None      # noqa: E731
        out = v.loc[idx] + (lin(OutcomeKind.REAL) if OutcomeKind.REAL in w else 0.0)
        out = out * v.real_scale + v.real_loc
        mult = torch.exp(lin(OutcomeKind.COUNT).clamp(max=30.0)) if OutcomeKind.COUNT in w else 1.0
        out = torch.where(kid == K_COUNT, v.count[idx] * mult, out)
        shift = 0.0
        if OutcomeKind.BINARY in w:
            shift = lin(OutcomeKind.BINARY) - self._ref.linear(phi, OutcomeKind.BINARY)[0]
        out = torch.where(kid == K_BINARY, torch.sigmoid(v.logit[idx] + shift), out)
        mult = torch.exp(-lin(OutcomeKind.TIME_TO_EVENT).clamp(min=-30.0)) if OutcomeKind.TIME_TO_EVENT in w else 1.0
        out = torch.where(kid == K_TTE, v.time[idx] * mult, out)
        return torch.where(kid == max(KIND_OF_ID), v.cat[idx], out)

    def _distribution(self, wb: WorldBatch, v: _View, r: int, q: int, mean: float, epi_var: float
                      ) -> Tuple[Distribution, float]:
        """The registered distribution of entry (r, q) and its aleatoric standard deviation."""
        o, kid = wb.out, int(wb.kind_ids[r, q])
        sel = (slice(None), r, 0, q)
        epi = math.sqrt(max(epi_var, 0.0))
        if kid == K_COUNT:
            expo = max(float(wb.meta.exposure[r, q]), 1e-12)
            conc = float(torch.exp(o.log_conc[sel].double()).mean())
            alea = mean + mean * mean / conc
            conc_tot = min(max(mean * mean / max(alea + epi_var - mean, 1e-9), 1e-3), 1e6)
            d: D.Dist = D.NegBinomial(torch.tensor(max(mean, 1e-12) / expo, dtype=F64),
                                      torch.tensor(conc_tot, dtype=F64), expo)
            return d.to_record(epistemic_std=epi), math.sqrt(max(alea, 0.0))
        if kid == K_REAL:
            s = float(v.real_scale[r, q])
            sc2 = float((o.scale[sel].double() ** 2).mean())
            df = float(o.df[sel].double().mean())
            fam_normal = o.cfg.real_family == D.FAMILY_NORMAL
            scale = math.sqrt(sc2 + epi_var / (s * s)) * s
            d = D.Normal(torch.tensor(mean, dtype=F64), torch.tensor(scale, dtype=F64)) if fam_normal else \
                D.StudentT(torch.tensor(mean, dtype=F64), torch.tensor(scale, dtype=F64), torch.tensor(df, dtype=F64))
            alea = math.sqrt(sc2 * (1.0 if fam_normal else df / (df - 2.0))) * s
            return d.to_record(epistemic_std=epi), alea
        if kid == K_BINARY:
            p = min(max(mean, 1e-9), 1 - 1e-9)
            return D.Bernoulli.from_prob(torch.tensor(p, dtype=F64)).to_record(epistemic_std=epi), math.sqrt(p * (1 - p))
        if kid == K_TTE:
            pmf = D.hazard_pmf(o.hazard_logits[sel].detach().double()).mean(0)               # exact mixture
            surv_before = 1.0 - torch.cumsum(pmf[:-1], 0) + pmf[:-1]
            h = (pmf[:-1] / surv_before.clamp_min(1e-300)).clamp(1e-12, 1 - 1e-12)
            ratio = 1.0
            if self._supports(self.fast, OutcomeKind.TIME_TO_EVENT):                         # proportional hazards
                ratio = float(torch.exp(self.fast.linear(v.phi[r, q], OutcomeKind.TIME_TO_EVENT)[0].clamp(-10, 10)))
            h = (1.0 - (1.0 - h) ** ratio).clamp(1e-12, 1 - 1e-12)
            edges = D.uniform_edges(wb.horizon_s[r, q].double().clamp_min(1e-6), o.cfg.n_hazard_bins)
            d = D.DiscreteHazard(torch.log(h) - torch.log1p(-h), edges)
            return d.to_record(epistemic_std=epi), float(d.std)
        rec = o.to_distribution(r, 0, q)
        p = float(rec.mean if rec.mean is not None else mean)
        return rec, math.sqrt(max(p * (1 - p), 0.0))

    # ---- evaluation -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_candidates(self, context_items: Sequence[ExperienceItem], goal_text: str,
                            candidates: Sequence[ActionSpec], queries: Optional[QuerySpec] = None, *,
                            decision_id: Optional[str] = None, objective_id: str = "",
                            deadline: Optional[float] = None, objective_scale: float = 1.0,
                            validate: Optional[Callable[[ActionSpec], List[str]]] = None,
                            compute_voi: bool = True, generator: Optional[torch.Generator] = None
                            ) -> List[CandidateEvaluation]:
        """Predict the consequences of every candidate and register the predictions.

        ``queries``: templates asked of every candidate, or a mapping ``action_id -> templates``, or a callable
        ``action -> templates``. ``{action}`` in a template's metric is replaced by the candidate's action id; a
        template with ``issued_at == 0`` is issued at ``now + action.est_cost.seconds``. ``deadline`` (absolute
        time of the objective's end) sets the evaluator's horizon and, with ``objective_id``, registers the
        state-level success forecast. ``validate`` is ``EnvironmentAdapter.validate_action``."""
        cands = list(candidates)
        if not cands:
            return []
        now = float(self.clock.now())
        decision_id = decision_id or new_id("dec")
        model, heads, cfg = self.model, self.heads, self.cfg
        n = len(cands)
        tok = model.tokenize([list(context_items)])
        b = model.core_forward(tok, None).b
        e_a, _ = model.encode_actions([cands])
        e_g = model.encode_goals([goal_text])
        z = model.decision_latent(b, e_a, e_g)                                               # [1, N, D]
        reasons = [(validate(a) if validate is not None else []) for a in cands]
        feasible = torch.tensor([not r for r in reasons], device=z.device)
        if not bool(feasible.any()):
            feasible = torch.ones_like(feasible)
        a_mask = feasible.unsqueeze(0)
        horizon = torch.tensor([max((deadline - now) if deadline is not None else cfg.default_horizon_s, 1.0)],
                               dtype=torch.float64, device=z.device)
        scale = torch.tensor([max(float(objective_scale), 1.0)], device=z.device)
        ev_q = heads.evaluator.forward_candidates(z, e_g, horizon, scale=scale)
        ev_v = heads.evaluator(b, e_g, horizon, scale=scale)
        pol = heads.policy(z, a_mask, b)
        res_scale = torch.tensor(cfg.resource_scale, device=z.device)
        res = ev_q.resources_expected(res_scale)[0]                                           # [N, R]
        # ---- world model -----------------------------------------------------------------------------------------
        concrete = self._instantiate(queries, cands, now, decision_id)
        canon = [[canonical_query(q, [a.action_id]) for q in row] for a, row in zip(cands, concrete)]
        has_q = any(concrete)
        mean = epi = None
        goal = torch.zeros(n, dtype=F64)
        goal_epi = torch.zeros(n, dtype=F64)
        ts = None
        any_goal = False
        if has_q:
            wb = world_batch(model, heads, z[0], [[(q, None) for q in row] for row in canon])
            view = self._view(wb)
            mean, epi = self._means(view, self.fast)
            gmask = self._goal_mask(canon, wb).to(mean.device)
            any_goal = bool(gmask.any())
            goal = (mean * gmask).sum(-1).cpu()
            goal_epi = (epi * gmask).sum(-1).sqrt().cpu()
            if any_goal and cfg.thompson_samples > 0:
                ts = (self._samples(view, self.fast, cfg.thompson_samples, generator) * gmask).sum(-1).cpu()   # [S, N]
        money = torch.tensor([a.est_cost.money for a in cands], dtype=F64)
        seconds = torch.tensor([a.est_cost.seconds for a in cands], dtype=F64)
        net_cost = cfg.cost_weight * money + cfg.time_weight * seconds
        voi = info = None
        if compute_voi and has_q and any_goal and self.fast is not None:
            voi, info = self._voi(wb, view, gmask, feasible.cpu(), net_cost, context_items, goal_text, cands,
                                  e_a, e_g, generator, means=(mean, epi))
        # ---- records ---------------------------------------------------------------------------------------------
        out: List[CandidateEvaluation] = []
        p_succ, q_logit = ev_q.p_success[0].double().cpu(), ev_q.success_logit[0].double().cpu()
        z_norm = z[0].norm(dim=-1).double().cpu()
        for i, a in enumerate(cands):
            preds: List[PredictionRecord] = []
            for j, q in enumerate(concrete[i]):
                m_ij, e_ij = float(mean[i, j]), float(epi[i, j])
                dist, alea = self._distribution(wb, view, i, j, m_ij, e_ij)
                rec = PredictionRecord(prediction_id=new_id("pred"), query=q, dist=dist, made_at=now,
                                       decision_id=decision_id, action_id=a.action_id,
                                       model_version=self.model_version, checkpoint_id=self.checkpoint_id)
                preds.append(self.registry.register(rec))
                self._remember(_Cached(rec.prediction_id, q, canon[i][j].metric, a.action_id, decision_id,
                                       view.phi[i, j].detach().cpu().clone(), float(view.count[:, i, j].mean()),
                                       float(view.loc[:, i, j].mean()), float(view.time[:, i, j].mean()),
                                       float(view.real_loc[i, j]), float(view.real_scale[i, j]), m_ij, alea,
                                       math.sqrt(max(e_ij, 0.0))))
            extras = {"policy_logit": float(pol.raw_logits[0, i]), "policy_q": float(pol.q[0, i]),
                      "policy_prob": float(pol.probs[0, i]), "v_state": float(ev_v.p_success[0]),
                      "eval_logit": float(q_logit[i]), "expected_time_s": float(ev_q.expected_time()[0, i]),
                      "expected_magnitude": float(torch.exp(ev_q.mag_log_rate[0, i]) * scale[0]),
                      "expected_money": float(res[i, 1]) if res.shape[-1] > 1 else 0.0,
                      "z_norm": float(z_norm[i]), "b_norm": float(b[0].norm()), "net_cost": float(net_cost[i])}
            if any_goal:
                extras["q_goal"], extras["q_goal_epistemic_std"] = float(goal[i]), float(goal_epi[i])
            if ts is not None:
                extras.update({f"{THOMPSON_KEY}{s}": float(ts[s, i]) for s in range(ts.shape[0])})
            if info is not None:
                extras.update({k: float(val[i]) for k, val in info.items()})
            out.append(CandidateEvaluation(
                action=a, predictions=preds, p_success=float(p_succ[i]), value=float(pol.raw_logits[0, i]),
                voi=None if voi is None else float(voi[i]), cost=Cost.from_dict(a.est_cost.to_dict()),
                uncertainty=float(goal_epi[i]) if any_goal else None,
                feasible=not reasons[i], infeasible_reason="; ".join(reasons[i]), extras=extras))
        if has_q and any_goal:
            self._keep_goal_view(decision_id, cands, concrete, view, gmask, goal_epi)
        if objective_id and deadline is not None and deadline > now:
            self._register_success(objective_id, decision_id, float(ev_v.p_success[0]), now, deadline)
        self.last = {"decision_id": decision_id, "b": b.detach(), "z": z.detach(), "e_a": e_a.detach(),
                     "e_g": e_g.detach(), "a_mask": a_mask, "policy_logits": pol.raw_logits.detach(),
                     "p_success": ev_q.p_success.detach(), "v_state": float(ev_v.p_success[0])}
        return out

    @staticmethod
    def _goal_mask(canon: List[List[OutcomeQuery]], wb: WorldBatch) -> Tensor:
        """[N, Q] 1.0 at each candidate's goal-relevant queries of the longest horizon it was asked about."""
        m = torch.zeros(wb.q_mask.shape, dtype=F64)
        for i, row in enumerate(canon):
            rel = [(q.horizon_s, j) for j, q in enumerate(row) if q.goal_relevant]
            if rel:
                top = max(h for h, _ in rel)
                for h, j in rel:
                    if h == top:
                        m[i, j] = 1.0
        return m

    def _remember(self, c: _Cached) -> None:
        if len(self._cache) >= self.max_cache:
            for k in list(self._cache)[: self.max_cache // 10]:
                del self._cache[k]
        self._cache[c.query.query_id] = c

    def _register_success(self, objective_id: str, decision_id: str, p: float, now: float, deadline: float) -> None:
        q = OutcomeQuery(query_id=new_id("q"), metric=f"objective.{objective_id}.achieved", kind=OutcomeKind.BINARY,
                         horizon_s=deadline - now, issued_at=now, description="the objective is verified as achieved",
                         decision_id=decision_id, goal_relevant=True)
        dist = D.Bernoulli.from_prob(torch.tensor(min(max(p, 1e-9), 1 - 1e-9), dtype=F64)).to_record()
        rec = self.registry.register(PredictionRecord(prediction_id=new_id("pred"), query=q, dist=dist, made_at=now,
                                                      decision_id=decision_id, model_version=self.model_version,
                                                      checkpoint_id=self.checkpoint_id))
        self._success.setdefault(objective_id, []).append(rec.prediction_id)

    # ---- value of information -------------------------------------------------------------------------------------
    def _voi(self, wb: WorldBatch, view: _View, gmask: Tensor, feasible: Tensor, net_cost: Tensor,
             context_items: Sequence[ExperienceItem], goal_text: str, cands: Sequence[ActionSpec],
             e_a: Tensor, e_g: Tensor, generator: Optional[torch.Generator],
             means: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Decision-relevant value of observing each candidate's own outcomes once (module docstring).

        ``means``: ``self._means(view, self.fast)`` when the caller already has it (the same values; INT3A item 2:
        it was computed three times per decision). The sampled-outcome loop runs on a copy of the head view on the
        fast state's device: the fast layer computes there anyway, and with the view on the GPU every one of the
        ``voi_max_candidates x voi_samples`` re-evaluations moved the features of every candidate to it and back."""
        cfg, fast = self.cfg, self.fast
        n = len(cands)
        s_n = max(1, cfg.voi_samples)

        def values(v: _View, f: Any) -> Tensor:
            m, _ = self._means(v, f)
            return (m * gmask.to(m.device)).sum(-1).cpu() - net_cost

        mean0, epi0 = means if means is not None else self._means(view, fast)
        base_vals = (mean0 * gmask.to(mean0.device)).sum(-1).cpu() - net_cost       # = values(view, fast)
        base = base_vals[feasible].max()
        kinds = [[KIND_OF_ID[int(k)] for k in row] for row in wb.kind_ids.cpu()]
        qm = wb.q_mask.cpu()
        dev = getattr(fast, "device", None)
        if dev is not None and view.phi.device != torch.device(dev):
            view = _View(**{f.name: getattr(view, f.name).to(dev) for f in dataclasses.fields(_View)})
        gross = torch.zeros(n, dtype=F64)
        stderr = torch.zeros(n, dtype=F64)
        gain = torch.zeros(n, dtype=F64)
        plain = torch.zeros(n, dtype=F64)
        order = sorted(range(n), key=lambda i: -float(epi0[i].sum()))[: cfg.voi_max_candidates]
        slow = self._slow_views(wb, context_items, goal_text, cands, e_a, e_g, order, mean0, epi0, s_n, generator) \
            if cfg.voi_slow_path else {}
        for i in order:
            js = [j for j in range(qm.shape[1]) if bool(qm[i, j]) and self._supports(fast, kinds[i][j])]
            if not bool(feasible[i]) or (not js and i not in slow):
                continue
            per_sample = []
            for s in range(s_n):
                f = fast.clone()
                kl = 0.0
                for j in js:
                    kind = kinds[i][j]
                    w = f.sample_weights(1, kind, generator)[0].to(view.phi.device)
                    lin = float((view.phi[i, j] * w).sum())
                    if kind == OutcomeKind.COUNT:
                        hc = max(float(view.count[:, i, j].mean()), 1e-9)
                        y = float(torch.poisson(torch.tensor(min(hc * math.exp(min(lin, 30.0)), 1e9), dtype=F64),
                                                generator=generator))
                        info = f.update(view.phi[i, j], y, kind, exposure=hc)
                    elif kind == OutcomeKind.REAL:
                        noise = float(f.posterior(kind)["noise"])
                        y = lin + math.sqrt(max(noise, 0.0)) * float(torch.randn((), generator=generator))
                        info = f.update(view.phi[i, j], y, kind)
                    elif kind == OutcomeKind.BINARY:
                        y = float(torch.rand((), generator=generator) < 1.0 / (1.0 + math.exp(-lin)))
                        info = f.update(view.phi[i, j], y, kind)
                    else:                                                                   # time to event
                        u = max(float(torch.rand((), generator=generator)), 1e-12)
                        info = f.update(view.phi[i, j], -math.log(u) * math.exp(-max(min(lin, 30.0), -30.0)), kind)
                    kl += float(info["kl"])
                v_after = slow[i][s] if i in slow else view
                vals = values(v_after, f)
                per_sample.append(vals)
                gain[i] += kl / s_n
            stack = torch.stack(per_sample)[:, feasible]                              # [S, feasible]
            t = stack.max(-1).values
            # Value of *knowing* the outcome before the next choice: E_y[max_a' Q_y(a')] - max_a' E_y[Q_y(a')], both
            # over the same samples. With an exact Bayesian update E_y[Q_y] = Q(b), i.e. this is the usual
            # E_y[max Q(b_updated)] - max Q(b); the approximate (assumed-density) update is not a martingale, and
            # taking max Q(b) as the baseline then credits that bias as information (kept as ``voi_vs_current``).
            max_e = stack.mean(0).max()
            g_i = float(t.mean() - max_e)
            gross[i] = 0.0 if abs(g_i) <= 1e-12 * max(1.0, abs(float(max_e))) else g_i
            plain[i] = float(t.mean() - base)
            stderr[i] = t.std() / math.sqrt(len(t)) if len(t) > 1 else 0.0
        voi = gross.clamp_min(0.0)
        return voi, {"voi_fast": voi, "voi_stderr": stderr, "info_gain_nats": gain, "voi_vs_current": plain,
                     "voi_net": gross - net_cost, "value_net": base_vals}

    def _slow_views(self, wb: WorldBatch, context_items: Sequence[ExperienceItem], goal_text: str,
                    cands: Sequence[ActionSpec], e_a: Tensor, e_g: Tensor, order: Sequence[int], mean0: Tensor,
                    epi0: Tensor, s_n: int, generator: Optional[torch.Generator]) -> Dict[int, List[_View]]:
        """For each candidate in ``order`` with predicted outcomes: ``s_n`` head views recomputed after an
        *imagined* result (sampled from its predictive) has entered a copy of the context. The imagined items
        never leave this function."""
        model, heads = self.model, self.heads
        now = float(self.clock.now())
        ctx = list(context_items)
        contexts, owners = [], []
        qm = wb.q_mask.cpu()
        for i in order:
            js = [j for j in range(qm.shape[1]) if bool(qm[i, j])]
            if not js:
                continue
            a = cands[i]
            for s in range(s_n):
                items = [ExperienceItem(kind=ItemKind.ACTION, text=a.render(), ts=now, ref=a.action_id,
                                        trust=Trust.SELF)]
                for j in js:
                    q = wb.queries[i][j]
                    sd = math.sqrt(max(float(epi0[i, j]), 0.0) + self._alea2(wb, i, j, float(mean0[i, j])))
                    y = float(mean0[i, j]) + sd * float(torch.randn((), generator=generator))
                    if q.kind in (OutcomeKind.COUNT, OutcomeKind.TIME_TO_EVENT):
                        y = max(0.0, round(y) if q.kind == OutcomeKind.COUNT else y)
                    elif q.kind == OutcomeKind.BINARY:
                        y = float(torch.rand((), generator=generator) < min(max(float(mean0[i, j]), 0.0), 1.0))
                    name = q.metric.replace(ACTION_PLACEHOLDER, "").rstrip(".")
                    # an immediate reading enters the context as an observation, a delayed consequence as an outcome
                    immediate = q.kind == OutcomeKind.REAL and not q.goal_relevant and q.horizon_s <= 3600.0
                    items.append(ExperienceItem(kind=ItemKind.OBSERVATION if immediate else ItemKind.OUTCOME,
                                                text=q.description or name, ts=now + q.horizon_s,
                                                ref=f"imagined:{i}:{s}:{j}", trust=Trust.UNTRUSTED,
                                                quantities=[Quantity(name, y, kind=q.kind, exposure=q.exposure)]))
                contexts.append(ctx + items)
                owners.append(i)
        if not contexts:
            return {}
        b2 = model.core_forward(model.tokenize(contexts), None).b                              # [R, D]
        r, n = b2.shape[0], e_a.shape[1]
        z2 = model.decision_latent(b2, e_a.expand(r, n, -1), e_g.expand(r, -1))                # [R, N, D]
        out: Dict[int, List[_View]] = {}
        rep = lambda t: t                                                                       # noqa: E731
        for k, i in enumerate(owners):
            zk = z2[k].unsqueeze(1)                                                             # [N, 1, D]
            phi = model.features(zk, wb.e_q)
            o = heads.world(zk, wb.e_q, wb.kind_ids, wb.horizon_s, phi if heads.world.in_phi is not None else None,
                            exposure=wb.meta.exposure, n_categories=wb.meta.n_categories, real_loc=wb.real_loc,
                            real_scale=wb.real_scale)
            out.setdefault(i, []).append(self._view(dataclasses.replace(wb, phi=rep(phi), out=o)))
        return out

    @staticmethod
    def _alea2(wb: WorldBatch, i: int, j: int, mean: float) -> float:
        kid = int(wb.kind_ids[i, j])
        if kid == K_COUNT:
            conc = float(torch.exp(wb.out.log_conc[:, i, 0, j].double()).mean())
            return max(mean, 0.0) + mean * mean / conc
        if kid == K_BINARY:
            return max(mean * (1 - mean), 0.0)
        return float(wb.out.member_variances()[:, i, 0, j].double().mean())

    # ---- outcomes ---------------------------------------------------------------------------------------------------
    def adopt_queries(self, action_id: str, env_queries: Sequence[OutcomeQuery], rel_tol: float = 0.05
                      ) -> Dict[str, str]:
        """Link queries the environment issued after executing ``action_id`` to the predictions registered for
        that action beforehand (same kind and canonical metric, nearest horizon within ``rel_tol``). Returns
        {environment query id: predicted query id}; outcomes may then arrive under either id."""
        mine = [c for c in self._cache.values() if c.action_id == action_id]
        linked: Dict[str, str] = {}
        for eq in env_queries:
            metric = canonical_query(eq, [action_id]).metric
            best = None
            for c in mine:
                if c.query.kind != eq.kind or c.metric != metric or c.query.query_id in linked.values():
                    continue
                gap = abs(c.query.horizon_s - eq.horizon_s)
                if gap <= rel_tol * max(eq.horizon_s, 1.0) and (best is None or gap < best[0]):
                    best = (gap, c)
            if best is not None:
                linked[eq.query_id] = best[1].query.query_id
        self._alias.update(linked)
        return linked

    def commit(self, decision_id: str, chosen_action_id: str) -> int:
        """Forget cached features of the candidates of ``decision_id`` that were not executed."""
        drop = [k for k, c in self._cache.items() if c.decision_id == decision_id and c.action_id != chosen_action_id]
        for k in drop:
            del self._cache[k]
        gv = self._goal_views.get(decision_id)
        if gv is not None and "rows" in gv:
            row = gv["rows"].get(chosen_action_id)
            if row is None:
                del self._goal_views[decision_id]
            else:
                self._goal_views[decision_id] = {"action_id": chosen_action_id, **row}
        return len(drop)

    # ---- goal-value epistemic uncertainty after the decision's own outcomes (LRN-3) -------------------------------
    def _keep_goal_view(self, decision_id: str, cands: Sequence[ActionSpec], concrete: List[List[OutcomeQuery]],
                        view: _View, gmask: Tensor, goal_epi: Tensor) -> None:
        rows: Dict[str, Dict[str, Any]] = {}
        for i, a in enumerate(cands):
            cols = [j for j in range(len(concrete[i])) if float(gmask[i, j]) > 0]
            if not cols:
                continue
            sl = lambda t: t[:, i:i + 1].detach().cpu().clone()                                # noqa: E731
            r = lambda t: t[i:i + 1].detach().cpu().clone()                                    # noqa: E731
            rows[a.action_id] = {
                "view": _View(count=sl(view.count), loc=sl(view.loc), logit=sl(view.logit), time=sl(view.time),
                              cat=sl(view.cat), phi=r(view.phi), kind_ids=r(view.kind_ids),
                              real_loc=r(view.real_loc), real_scale=r(view.real_scale)),
                "gmask": gmask[i:i + 1].detach().cpu().clone(),
                "queries": [concrete[i][j].query_id for j in cols], "before": float(goal_epi[i])}
        if not rows:
            return
        self._goal_views[decision_id] = {"rows": rows}
        self._goal_views.move_to_end(decision_id)
        while len(self._goal_views) > self.max_goal_views:
            self._goal_views.popitem(last=False)

    def goal_epistemic_after(self, decision_id: str) -> Optional[Dict[str, float]]:
        """The executed candidate's goal-value epistemic std recomputed with the *current* fast belief state on the
        decision-time head view (the same quantity as its ``CandidateEvaluation.uncertainty``), once every one of its
        goal-relevant predictions has absorbed its own matured outcome; None before that (or when the decision had
        no goal-relevant prediction). Returns {"epistemic_before", "epistemic_after"}."""
        gv = self._goal_views.get(decision_id)
        if gv is None or "rows" in gv:
            return None
        if not all(f"query:{q}" in self._absorbed for q in gv["queries"]):
            return None
        v: _View = gv["view"]
        dev = self.fast.device if hasattr(self.fast, "device") else None
        if dev is not None:
            v = _View(**{f.name: getattr(v, f.name).to(dev) for f in dataclasses.fields(_View)})
        _, epi = self._means(v, self.fast)
        after = float((epi.cpu() * gv["gmask"]).sum(-1).sqrt()[0])
        return {"epistemic_before": float(gv["before"]), "epistemic_after": after}

    def forget_goal_view(self, decision_id: str) -> None:
        self._goal_views.pop(decision_id, None)

    def goal_views_state(self) -> List[Dict[str, Any]]:
        """Checkpointable goal views of executed decisions (a restored agent writes the same ``epistemic_after``).
        Views of decisions not yet committed are left out (they are re-made by the next evaluation)."""
        out = []
        for did, gv in self._goal_views.items():
            if "rows" in gv:
                continue
            v: _View = gv["view"]
            out.append({"decision_id": did, "action_id": gv["action_id"], "queries": list(gv["queries"]),
                        "before": float(gv["before"]), "gmask": gv["gmask"].detach().cpu(),
                        "view": {f.name: getattr(v, f.name).detach().cpu() for f in dataclasses.fields(_View)}})
        return out

    def load_goal_views_state(self, rows: Sequence[Dict[str, Any]]) -> None:
        self._goal_views.clear()
        for r in rows:
            self._goal_views[r["decision_id"]] = {"action_id": r["action_id"], "queries": list(r["queries"]),
                                                  "before": float(r["before"]), "gmask": r["gmask"],
                                                  "view": _View(**r["view"])}

    def _own(self, o: OutcomeObservation) -> OutcomeObservation:
        qid = self._alias.get(o.query_id, o.query_id)
        return o if qid == o.query_id else dataclasses.replace(o, query_id=qid)

    def observe_outcomes(self, outcomes: Sequence[OutcomeObservation]) -> Dict[str, Any]:
        """Immediate fast-belief update (no gradient step) from matured outcomes of predicted queries.

        Returns {"n": updates applied, "surprise": summed predictive NLL (nats), "kl": summed belief change
        KL(posterior || previous posterior) (nats), "change_detected", "details": per outcome (ids, kind,
        surprise, kl, error attribution), "skipped": reason -> count}. Pending and unobservable outcomes, outcomes
        of unknown queries and repeated reports of the same event change nothing. Imagined values are refused."""
        res: Dict[str, Any] = {"n": 0, "surprise": 0.0, "kl": 0.0, "change_detected": False, "details": [],
                               "skipped": {}}

        def skip(why: str) -> None:
            res["skipped"][why] = res["skipped"].get(why, 0) + 1

        for raw in outcomes:
            refuse_imagined(raw, "outcome")
            if not isinstance(raw, OutcomeObservation):
                raise TypeError("observe_outcomes() takes contracts.OutcomeObservation records")
            if str(raw.source).startswith("imagin"):
                refuse_imagined({"imagined": True}, "outcome")
            o = self._own(raw)
            c = self._cache.get(o.query_id)
            if o.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) or o.value is None:
                skip("not_matured")
                continue
            if c is None:
                skip("unknown_query")
                continue
            key = o.dedup_key or f"{o.query_id}@{o.observed_at}"
            if key in self._absorbed or f"query:{c.query.query_id}" in self._absorbed:
                skip("duplicate")                              # one prediction absorbs one outcome, once
                continue
            if not (o.observed_at > self.registry.registered_at(c.prediction_id)):
                skip("not_after_prediction")
                continue
            kind = c.query.kind
            if o.status == OutcomeStatus.CENSORED and kind != OutcomeKind.TIME_TO_EVENT:
                skip("censored_non_time")
                continue
            self._absorbed.update((key, f"query:{c.query.query_id}"))
            det = self._detectors.setdefault(c.metric, ChangeDetector())
            total = math.sqrt(c.aleatoric_std ** 2 + c.epistemic_std ** 2)
            det.update((float(o.value) - c.mean) / max(total, 1e-9))
            att = classify_error(o.status, o.value, c.mean, c.aleatoric_std, c.epistemic_std, det)
            detail: Dict[str, Any] = {"query_id": c.query.query_id, "prediction_id": c.prediction_id, "kind": kind,
                                      "action_id": c.action_id, "decision_id": c.decision_id, "error": att.label,
                                      "z_total": att.z_total,
                                      "predicted": c.mean, "observed": float(o.value)}
            if self._supports(self.fast, kind):
                y = float(o.value)
                if kind == OutcomeKind.COUNT:
                    info = self.fast.update(c.phi, y, kind, exposure=max(c.head_count, 1e-9))
                elif kind == OutcomeKind.REAL:
                    info = self.fast.update(c.phi, (y - c.real_loc) / c.real_scale - c.head_loc, kind)
                elif kind == OutcomeKind.TIME_TO_EVENT:
                    info = self.fast.update(c.phi, max(y / max(c.head_time, 1e-6), 1e-9), kind,
                                            censored=o.status == OutcomeStatus.CENSORED)
                else:
                    info = self.fast.update(c.phi, min(max(y, 0.0), 1.0), kind)
                res["n"] += 1
                res["surprise"] += float(info["surprise"])
                res["kl"] += float(info["kl"])
                res["change_detected"] = res["change_detected"] or bool(info["change_detected"])
                detail.update(surprise=float(info["surprise"]), kl=float(info["kl"]),
                              change_detected=bool(info["change_detected"]))
            else:
                skip("kind_not_in_fast_layer")
            res["details"].append(detail)
        return res

    def resolve(self, outcomes: Sequence[OutcomeObservation]) -> List[PredictionRecord]:
        """Fill the registered prediction records these outcomes answer (each exactly once). Pending outcomes
        are ignored; an unobservable one closes its prediction without a score."""
        done: List[PredictionRecord] = []
        for raw in outcomes:
            refuse_imagined(raw, "outcome of a prediction")
            if raw.status == OutcomeStatus.PENDING:
                continue
            o = self._own(raw)
            for rec in self.registry.by_query(o.query_id):
                if rec.resolved_status != OutcomeStatus.PENDING:
                    continue
                try:
                    done.append(self.registry.resolve(rec.prediction_id, o))
                except AlreadyResolvedError:
                    continue
        return done

    def success_predictions(self, objective_id: str) -> List[PredictionRecord]:
        return [self.registry.get(p) for p in self._success.get(objective_id, [])]

    def resolve_objective(self, outcome: VerifiedOutcome, check_signature: Callable[[VerifiedOutcome], bool]
                          ) -> List[PredictionRecord]:
        """Resolve the success forecasts of an objective with its *verified* outcome. A pending objective
        resolves nothing; an outcome whose signature does not check is refused."""
        refuse_imagined(outcome, "objective outcome")
        if not isinstance(outcome, VerifiedOutcome) or not outcome.signature or not check_signature(outcome):
            raise UnverifiedOutcomeError("the verifier's signature on this outcome did not check")
        if outcome.status not in (GoalStatus.ACHIEVED, GoalStatus.FAILED):
            return []
        done: List[PredictionRecord] = []
        for pid in self._success.get(outcome.objective_id, []):
            rec = self.registry.get(pid)
            if rec.resolved_status != OutcomeStatus.PENDING or not (outcome.evaluated_at > rec.made_at):
                continue
            obs = OutcomeObservation(query_id=rec.query.query_id, status=OutcomeStatus.OBSERVED,
                                     value=1.0 if outcome.status == GoalStatus.ACHIEVED else 0.0,
                                     observed_at=outcome.evaluated_at, verified=True, source="verifier",
                                     dedup_key=f"objective:{outcome.objective_id}:{pid}")
            done.append(self.registry.resolve(pid, obs))
        return done

    # ---- signals for the reward workstream (not rewards) --------------------------------------------------------------
    @torch.no_grad()
    def belief_value(self, context_items: Sequence[ExperienceItem], goal_text: str, *,
                     deadline: Optional[float] = None, objective_scale: float = 1.0) -> Dict[str, float]:
        """V(b, g, horizon) of a context: {"p_success", "success_logit", "b_norm"}. The difference between two
        calls is the evaluator's changed prospects (a signal, see ``evaluator.delta_v``), never a reward."""
        now = float(self.clock.now())
        b = self.model.core_forward(self.model.tokenize([list(context_items)]), None).b
        hz = torch.tensor([max((deadline - now) if deadline is not None else self.cfg.default_horizon_s, 1.0)],
                          dtype=torch.float64, device=b.device)
        out = self.heads.evaluator(b, self.model.encode_goals([goal_text]), hz,
                                   scale=torch.tensor([max(objective_scale, 1.0)], device=b.device))
        return {"p_success": float(out.p_success[0]), "success_logit": float(out.success_logit[0]),
                "b_norm": float(b[0].norm())}

    def downstream_signals(self, ev: CandidateEvaluation) -> List[float]:
        """The signal vector of ``heads.downstream`` for one evaluation (order: ``heads.VOI_SIGNALS``)."""
        x = ev.extras
        vals = {"voi_fast": x.get("voi_fast", 0.0), "info_gain_nats": x.get("info_gain_nats", 0.0),
                "epistemic_std": ev.uncertainty or 0.0, "cost_money": ev.cost.money, "cost_seconds": ev.cost.seconds}
        return [float(vals[k]) for k in VOI_SIGNALS]
