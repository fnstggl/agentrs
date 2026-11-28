"""cam_live.credit.providers — RUDDER and COCOA on the real model's latents, the slow bridge objective, and the
credit assigner (spec 9.2-9.4, 12A.9; RC-02, RC-03, RC-05, RC-06, RC-08, CU-20).

* :class:`TrajectoryEncoder` turns the stored decisions of one objective into decision-level latents with the
  *current* model (``learning_api.encode_decisions``), cached exactly: the cache is dropped as soon as any tensor
  the latents depend on changes.
* :class:`ReturnDecompositionProvider` (``LossProvider`` "rudder"): squared-error regression of the matured return
  of *closed* objectives from every prefix of their decision latents.
* :class:`HindsightProvider` (``LossProvider`` "cocoa"): the hindsight model over the recorded candidate set of a
  decision given an outcome that a *later* decision of the same objective produced.
* :class:`BridgeGroundingProvider` (``LossProvider`` "bridge_grounding", slow timescale): the backbone, conditioned
  on CAM's belief through the latent bridge, has to assign likelihood to the text of the consequence that was
  actually observed. It is the experience-grounded loss that reaches both adapter parts (LoRA and bridge).
* :class:`CreditAssigner` computes, stores and versions per-decision credit for every estimator when outcomes
  mature, withholds learned credit until the return is final, exposes the credited returns of the estimator that
  the evidence-based selector currently trusts, and compares the estimators with randomised arm differences.

Return convention (the same in all three estimators): an outcome of value v observed at time tau is worth
``v * exp(-beta * (tau - t))`` at a decision made at time t. A trajectory return is referenced at its first
decision. An outcome event (dedup key) enters a return exactly once, whatever number of records it is attached to.

Nothing here produces a reward: credit is bookkeeping about verified / measured outcomes, written to
``ExperienceRecord.credit`` and read by whoever is allowed to use it.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

from ..contracts import Clock, ExperienceRecord, ExperimentRecord, OutcomeStatus, Record, WallClock
from ..learning_api import TIMESCALE_FREQUENT, TIMESCALE_SLOW, DecisionBatch, encode_decisions
from ..nn.api import LatentModelAPI
from .cocoa import SUPPORTED, UNSUPPORTED, HindsightModel, check_support
from .links import OutcomeLink, record_arm
from .replay import SqliteExperienceStore, outcome_key
from .rudder import ReturnDecomposer, duration_discount
from .select import ARM_DIFFERENCE, UNIT_MEAN, EstimatorSelector, ExperimentEvidence, SelectionRecord, paired_comparison
from .simple import NAME as SIMPLE, arm_difference, credit_from_links

RUDDER, COCOA = "rudder", "cocoa"
KEY_RTG, KEY_CONTRIB = "rudder_rtg", "rudder_contribution"
#: ``record.credit`` key holding the *learning signal* of each selectable estimator
LEARNING_KEY = {SIMPLE: SIMPLE, RUDDER: RUDDER, COCOA: COCOA}
#: ``record.credit`` key compared with randomised arm differences (for RUDDER the prediction difference: the
#: return-to-go signal equals the measured difference by construction and would validate nothing)
VALIDATION_KEY = {SIMPLE: SIMPLE, RUDDER: KEY_CONTRIB}
U_DIM = 10
#: candidate generators whose proposals are open-ended (free-form text of the backbone, random sampling, mutation):
#: the set of alternatives has no enumerable support unless the proposal probabilities were tracked
OPEN_ENDED_GENERATORS = ("backbone", "schema_sampler", "mutation", "generated", "llm", "sampled")
#: generators that enumerate a fixed set: a candidate they produced would be in the set whatever was proposed
CLOSED_GENERATORS = ("schema_walker", "skill_library", "builtin", "fixed", "enumerated", "experiment")


def candidate_set_open_ended(rec: ExperienceRecord) -> bool:
    """Whether the decision's candidate set came from an open-ended generator (spec 9.3: no unbiased counterfactual
    claim over arbitrary open-ended actions). True if the decision says so (``diagnostics['open_ended']``) or if some
    candidate's provenance names only open-ended generators (``ActionSpec.provenance``, parts joined by '+'); an
    explicit ``open_ended: False`` does not override the candidates' provenance."""
    if rec.decision.diagnostics.get("open_ended"):
        return True
    for c in rec.decision.candidates:
        parts = {x for x in str(c.action.provenance or "").split("+") if x}
        if parts & set(OPEN_ENDED_GENERATORS) and not parts & set(CLOSED_GENERATORS):
            return True
    return False


def proposals_tracked(rec: ExperienceRecord) -> bool:
    """Whether the proposal probability of every candidate was recorded (``diagnostics['tracked_proposals']`` or
    ``diagnostics['proposal_probs']``). Default False: nothing is assumed."""
    d = rec.decision.diagnostics
    pp = d.get("proposal_probs")
    return bool(d.get("tracked_proposals", False)) or (isinstance(pp, (list, tuple)) and
                                                       len(pp) == len(rec.decision.candidates) and len(pp) > 0)


# ------------------------------------------------------------------------------------------------ returns
@dataclass
class OutcomeEvent:
    exp_id: str                  # the decision that declared the measurement
    key: str                     # outcome identity (dedup key)
    value: float
    observed_at: float
    verified: bool
    index: int = -1              # position of the owning top-level decision in the trajectory (-1: not in it)


@dataclass
class TrajectoryReturn:
    objective_id: str
    exp_ids: List[str]           # top-level decisions in order
    times: List[float]
    events: List[OutcomeEvent]
    G: float                     # sum of event values discounted to the first decision
    closed: bool                 # an objective outcome was recorded: no further decision will be added
    n_pending: int               # goal-relevant measurements still outstanding
    matured: bool                # the return is final
    reason: str = ""


def parent_id(rec: ExperienceRecord) -> str:
    """Decision id of the higher-level decision this one was issued under ('' = top level; ``credit.smdp``)."""
    return str(rec.decision.chosen.parent_decision_id or "")


def is_top_level(rec: ExperienceRecord) -> bool:
    return not parent_id(rec)


def top_index(records: Sequence[ExperienceRecord]) -> Dict[str, int]:
    """exp_id -> index of the top-level decision that owns it (a lower-level record maps to its ancestor)."""
    top = [r for r in records if is_top_level(r)]
    pos = {r.exp_id: i for i, r in enumerate(top)}
    by_dec = {r.decision.decision_id: r for r in records}
    out: Dict[str, int] = {}
    for r in records:
        cur, hops = r, 0
        while not is_top_level(cur) and hops < 16:
            parent = by_dec.get(parent_id(cur))
            if parent is None:
                break
            cur, hops = parent, hops + 1
        out[r.exp_id] = pos.get(cur.exp_id, -1)
    return out


def trajectory_return(store: SqliteExperienceStore, records: Sequence[ExperienceRecord], *, beta: float = 0.0,
                      require_closed: bool = True, verified_only: bool = False, goal_relevant_only: bool = True
                      ) -> TrajectoryReturn:
    """The measured return of one objective's decisions and whether it is final.

    Events are the *observed* answers to declared goal-relevant queries, one per dedup key. The return is final
    when the objective was closed (``store.put_objective_outcome``) and no goal-relevant query is outstanding."""
    records = list(records)
    top = [r for r in records if is_top_level(r)]
    oid = records[0].objective_id if records else ""
    owner = top_index(records)
    events: List[OutcomeEvent] = []
    seen: set = set()
    n_pending = 0
    for r in records:
        queries = {q.query_id: q for q in r.pending_queries}
        if r.result is not None:
            for q in r.result.expected_outcomes:
                queries.setdefault(q.query_id, q)
        final = {o.query_id for o in r.outcomes
                 if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED, OutcomeStatus.UNOBSERVABLE)}
        for qid, q in queries.items():
            if (q.goal_relevant or not goal_relevant_only) and qid not in final:
                n_pending += 1
        for o in r.outcomes:
            q = queries.get(o.query_id)
            if o.status != OutcomeStatus.OBSERVED or o.value is None or q is None:
                continue
            if (goal_relevant_only and not q.goal_relevant) or (verified_only and not o.verified):
                continue
            k = outcome_key(o)
            if k in seen:
                continue
            seen.add(k)
            events.append(OutcomeEvent(r.exp_id, k, float(o.value), float(o.observed_at or 0.0), bool(o.verified),
                                       owner.get(r.exp_id, -1)))
    events.sort(key=lambda e: (e.observed_at, e.key))
    times = [float(r.decision.ts) for r in top]
    t0 = times[0] if times else 0.0
    G = sum(e.value * math.exp(-beta * max(e.observed_at - t0, 0.0)) for e in events)
    closed = bool(oid) and store.objective_outcome(oid) is not None
    reason = ""
    if not top:
        reason = "no decisions"
    elif require_closed and not closed:
        reason = "objective still open"
    elif n_pending:
        reason = f"{n_pending} goal-relevant measurements pending"
    return TrajectoryReturn(oid, [r.exp_id for r in top], times, events, float(G), closed, n_pending, not reason, reason)


def return_to_go_at(tr: TrajectoryReturn, t: float, beta: float) -> float:
    """Measured return-to-go at time ``t``: events observed at or after ``t``, discounted to ``t``."""
    return float(sum(e.value * math.exp(-beta * max(e.observed_at - t, 0.0)) for e in tr.events if e.observed_at >= t))


def _training_call() -> bool:
    """A loss computed with gradients enabled may be stepped on: its data counts as training data of the estimator
    (``validate`` and credit assignment run without gradients)."""
    return torch.is_grad_enabled()


def outcome_features(values: Tensor) -> Tensor:
    """[K] outcome values -> [K, U_DIM] bounded features (symlog magnitude, sign, radial bumps). The outcome
    representation of the hindsight model: it describes the outcome, not the action that preceded it."""
    v = values.float().reshape(-1)
    x = torch.sign(v) * torch.log1p(v.abs())
    centers = torch.linspace(-1.0, 5.0, U_DIM - 2, device=v.device)
    return torch.cat([x[:, None] / 3.0, (v > 0).float()[:, None],
                      torch.exp(-0.5 * ((x[:, None] - centers) / 0.75) ** 2)], -1)


# ------------------------------------------------------------------------------------------------ latents
@dataclass
class EncodedTrajectory:
    objective_id: str
    exp_ids: List[str]
    b: Tensor                    # [T, D] belief at each decision
    z: Tensor                    # [T, D] decision latent of the executed candidate
    e_a: Tensor                  # [T, N, D]
    a_mask: Tensor               # [T, N]
    probs: Tensor                # [T, N] logged behaviour probabilities
    has_probs: List[bool]        # False where the decision recorded none
    chosen: Tensor               # [T]
    e_g: Tensor                  # [D]
    ts: Tensor                   # [T] float64
    open_ended: List[bool] = field(default_factory=list)
    tracked: List[bool] = field(default_factory=list)
    arms: List[Dict[str, int]] = field(default_factory=list)       # arm name -> candidate index, per decision

    def __len__(self) -> int:
        return len(self.exp_ids)


def candidate_arms(rec: ExperienceRecord) -> Dict[str, int]:
    """arm name -> candidate index: ``decision.diagnostics['arms']`` or the candidates' ``arguments['arm']``."""
    arms = rec.decision.diagnostics.get("arms")
    if isinstance(arms, dict):
        return {str(k): int(v) for k, v in arms.items()}
    out: Dict[str, int] = {}
    for i, c in enumerate(rec.decision.candidates):
        a = c.action.arguments.get("arm")
        if a not in (None, ""):
            out.setdefault(str(a), i)
    return out


class TrajectoryEncoder:
    """Decision-level latents of stored trajectories under the current model, with an exact cache.

    The cache key includes the version counters of every experiential and adapter tensor, so a cached latent is
    never used after the parameters it was computed from changed."""

    def __init__(self, model: LatentModelAPI, store: SqliteExperienceStore, *, max_len: int = 64, chunk: int = 32,
                 max_cached: int = 4096):
        self.model, self.store = model, store
        self.max_len, self.chunk, self.max_cached = int(max_len), int(chunk), int(max_cached)
        self._cache: Dict[Tuple[str, int], EncodedTrajectory] = {}
        self._fp: Tuple = ()
        self.stats = {"encoded": 0, "hits": 0, "invalidations": 0}

    def fingerprint(self) -> Tuple:
        groups = self.model.param_groups()
        return tuple(p._version for g in ("experiential", "semantic_adapters") for p in groups.get(g, []))

    def records(self, objective_id: str) -> List[ExperienceRecord]:
        return self.store.trajectory(objective_id)

    def encode(self, objective_id: str, records: Optional[Sequence[ExperienceRecord]] = None, *, grad: bool = False,
               full: bool = False, semantic: bool = False, last: Optional[int] = None,
               grad_last: Optional[int] = None) -> Optional[EncodedTrajectory]:
        """Latents of the objective's top-level decisions: the newest ``max_len`` (training) or all (``full``:
        credit assignment, so that no decision of a long trajectory is left without credit). ``semantic``: with a
        graph into the semantic adapters (newest ``grad_last`` context items) for the newest ``last`` decisions,
        never cached (the slow step's protection of readers of the representation, ``learner``)."""
        recs = [r for r in (records if records is not None else self.records(objective_id)) if is_top_level(r)]
        if not full:
            recs = recs[-(int(last) if last else self.max_len):]
        if not recs:
            return None
        key = (objective_id, len(recs))
        if semantic:
            grad = True
        if not grad:
            fp = self.fingerprint()
            if fp != self._fp:
                if self._cache:
                    self.stats["invalidations"] += 1
                self._cache, self._fp = {}, fp
            hit = self._cache.get(key)
            if hit is not None and hit.exp_ids == [r.exp_id for r in recs]:
                self.stats["hits"] += 1
                return hit
        parts: List[DecisionBatch] = []
        kw: Dict[str, Any] = {"with_grad_semantic": True, "grad_last": grad_last} if semantic else {}
        for i in range(0, len(recs), self.chunk):
            parts.append(encode_decisions(self.model, self.store, recs[i:i + self.chunk], grad=grad, **kw))
        n = max(p.e_a.shape[1] for p in parts)

        def pad(t: Tensor) -> Tensor:
            return t if t.shape[1] == n else torch.cat([t, t.new_zeros((t.shape[0], n - t.shape[1]) + t.shape[2:])], 1)

        enc = EncodedTrajectory(
            objective_id, [r.exp_id for r in recs], b=torch.cat([p.b for p in parts]),
            z=torch.cat([p.z_chosen for p in parts]), e_a=torch.cat([pad(p.e_a) for p in parts]),
            a_mask=torch.cat([pad(p.a_mask) for p in parts]), probs=torch.cat([pad(p.behaviour_probs) for p in parts]),
            has_probs=[bool(r.decision.probabilities) for r in recs], chosen=torch.cat([p.chosen for p in parts]),
            e_g=parts[0].e_g[0], ts=torch.cat([p.ts for p in parts]),
            open_ended=[candidate_set_open_ended(r) for r in recs], tracked=[proposals_tracked(r) for r in recs],
            arms=[candidate_arms(r) for r in recs])
        self.stats["encoded"] += len(recs)
        if not grad and not semantic:
            if len(self._cache) >= self.max_cached:
                self._cache.clear()
            self._cache[key] = enc
        return enc


def _batch_objectives(batch: DecisionBatch, limit: int) -> List[str]:
    out: List[str] = []
    for r in batch.records:
        if r.objective_id and r.objective_id not in out:
            out.append(r.objective_id)
        if len(out) >= limit:
            break
    return out


class _ReturnCache:
    """Final returns of closed objectives (a matured return never changes; anything else is recomputed)."""

    def __init__(self, store: SqliteExperienceStore, beta: float, require_closed: bool, verified_only: bool):
        self.store, self.beta, self.require_closed, self.verified_only = store, beta, require_closed, verified_only
        self._final: Dict[str, TrajectoryReturn] = {}

    def get(self, objective_id: str, records: Optional[Sequence[ExperienceRecord]] = None) -> TrajectoryReturn:
        hit = self._final.get(objective_id)
        if hit is not None:
            return hit
        if self.require_closed and records is None and self.store.objective_outcome(objective_id) is None:
            return TrajectoryReturn(objective_id, [], [], [], 0.0, False, 0, False, "objective still open")
        recs = list(records) if records is not None else self.store.trajectory(objective_id)
        tr = trajectory_return(self.store, recs, beta=self.beta, require_closed=self.require_closed,
                               verified_only=self.verified_only)
        if tr.matured:
            self._final[objective_id] = tr
        return tr

    def forget(self, objective_id: str = "") -> None:
        if objective_id:
            self._final.pop(objective_id, None)
        else:
            self._final.clear()


def _seeded(seed: int, build: Callable[[], Any]) -> Any:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return build()


# ------------------------------------------------------------------------------------------------ RUDDER provider
@dataclass
class RudderCredit:
    objective_id: str
    exp_ids: List[str]
    signal: List[float]          # learning signal: redistributed return-to-go minus the prefix prediction
    return_to_go: List[float]
    contribution: List[float]    # prediction differences (+ offset on the first, residual on the last decision)
    G: float
    residual: float              # G - F(whole trajectory): what the decomposer failed to explain
    conservation_error: float    # |sum_t Gamma_t c_t - G|


class ReturnDecompositionProvider:
    """``LossProvider`` for the return decomposer on decision-level latents of matured trajectories.

    beta            rate of the duration-aware discount (per second)
    use_belief      per-decision input is [z_chosen, b] (True) or z_chosen alone
    detach_latents  True: only the decomposer (a head) is trained by this loss; False: the regression also shapes
                    the experiential representation
    """
    timescale = TIMESCALE_FREQUENT
    adapter_grad = False             # latents are re-encoded without an adapter graph: nothing for the slow step
    needs_latents = False            # reads ``batch.records`` only and encodes whole trajectories itself

    def __init__(self, model: LatentModelAPI, store: SqliteExperienceStore, *, beta: float = 0.0, weight: float = 1.0,
                 d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_trajectories: int = 8, max_len: int = 64,
                 use_belief: bool = True, detach_latents: bool = True, require_closed: bool = True,
                 verified_only: bool = False, head_name: str = "credit_rudder", name: str = RUDDER, seed: int = 0,
                 encoder: Optional[TrajectoryEncoder] = None):
        self.name, self.weight = name, float(weight)
        self.model, self.store, self.beta = model, store, float(beta)
        self.max_trajectories, self.use_belief, self.detach_latents = int(max_trajectories), use_belief, detach_latents
        self.encoder = encoder or TrajectoryEncoder(model, store, max_len=max_len)
        self.returns = _ReturnCache(store, self.beta, require_closed, verified_only)
        d = model.d_model
        self.decomposer: ReturnDecomposer = model.register_head(head_name, _seeded(
            7_000_001 + seed, lambda: ReturnDecomposer(d * (2 if use_belief else 1), d_model=d_model, n_layers=n_layers,
                                                       n_heads=n_heads, d_goal=d, beta=self.beta)))
        self.decomposer.register_buffer("scale_fitted", torch.zeros((), dtype=torch.bool))
        self.decomposer.to(model.device)
        #: objectives whose trajectories entered a training loss of the decomposer (held out from its validation)
        self.trained_objectives: Set[str] = set()

    def state_dict(self) -> Dict[str, Any]:
        return {"trained_objectives": sorted(self.trained_objectives)}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.trained_objectives = set(st.get("trained_objectives", []))

    # ---- LossProvider -------------------------------------------------------------------------------------
    def eligible(self, record: ExperienceRecord) -> bool:
        return is_top_level(record) and self.returns.get(record.objective_id).matured

    def _inputs(self, enc: EncodedTrajectory) -> Tensor:
        x = torch.cat([enc.z, enc.b], -1) if self.use_belief else enc.z
        return x.detach() if self.detach_latents else x

    def _stack(self, encs: Sequence[EncodedTrajectory], G: Sequence[float]) -> Tuple[Tensor, ...]:
        dev = self.model.device
        T = max(len(e) for e in encs)
        B = len(encs)
        x = torch.zeros(B, T, self.decomposer.inp.in_features, device=dev)
        mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
        times = torch.zeros(B, T, dtype=torch.float64, device=dev)
        rows = []
        for i, e in enumerate(encs):
            xi = self._inputs(e)
            rows.append(torch.cat([xi, xi.new_zeros(T - len(e), xi.shape[1])]) if len(e) < T else xi)
            mask[i, :len(e)] = True
            times[i, :len(e)] = e.ts
            times[i, len(e):] = e.ts[-1]
        x = torch.stack(rows)
        e_g = torch.stack([e.e_g for e in encs])
        e_g = e_g.detach() if self.detach_latents else e_g
        return x, mask, torch.tensor(list(G), dtype=torch.float64, device=dev), e_g, times

    def _matured(self, objective_ids: Sequence[str], grad: bool, full: bool = False
                 ) -> Tuple[List[EncodedTrajectory], List[TrajectoryReturn]]:
        encs, rets = [], []
        for oid in objective_ids:
            tr = self.returns.get(oid)
            if not tr.matured:
                continue
            enc = self.encoder.encode(oid, grad=grad, full=full)
            if enc is None:
                continue
            encs.append(enc)
            rets.append(tr)
        return encs, rets

    def window_target(self, enc: EncodedTrajectory, tr: TrajectoryReturn) -> float:
        """Regression target of an encoded window: G for a whole trajectory; for a window of its newest decisions
        the measured return-to-go at the window's first decision, referenced there (the decomposer's time origin is
        its first input), so the target and the discount convention agree."""
        if len(enc) >= len(tr.exp_ids) or not len(enc):
            return tr.G
        return return_to_go_at(tr, float(enc.ts[0]), self.beta)

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Optional[Tensor], Dict[str, float]]:
        encs, rets = self._matured(_batch_objectives(batch, self.max_trajectories), grad=not self.detach_latents)
        if not encs:
            return None, {"n_trajectories": 0.0}
        x, mask, G, e_g, times = self._stack(encs, [self.window_target(e, t) for e, t in zip(encs, rets)])
        matured = torch.ones(len(encs), dtype=torch.bool, device=x.device)
        training = _training_call()
        if training:
            self.trained_objectives.update(t.objective_id for t in rets)
            with torch.no_grad():                   # RUDDER's linear return scale: the largest |G| trained on so far
                top = float(G.abs().max())
                if not bool(self.decomposer.scale_fitted) or top > float(self.decomposer.scale):
                    self.decomposer.scale.fill_(max(top, 1e-8))
                    self.decomposer.scale_fitted.fill_(True)
        loss, m = self.decomposer.loss(x, mask, G, matured, e_g, times)
        m["n_trajectories"] = float(len(encs))
        m["n_windows_truncated"] = float(sum(len(e) < len(t.exp_ids) for e, t in zip(encs, rets)))
        m["return_scale"] = float(self.decomposer.scale)
        return loss, m

    # ---- credit ---------------------------------------------------------------------------------------------
    def redistribute(self, objective_ids: Sequence[str]) -> Dict[str, RudderCredit]:
        """Redistributed credit of every *matured* objective in ``objective_ids`` (others are absent), over the whole
        trajectory (every decision gets credit; G and every Gamma_t are referenced at the trajectory's first decision).
        Training uses windows of the newest ``max_len`` decisions; positions beyond ``max_len`` are extrapolated."""
        encs, rets = self._matured(objective_ids, grad=False, full=True)
        out: Dict[str, RudderCredit] = {}
        for i in range(0, len(encs), 64):
            ee, rr = encs[i:i + 64], rets[i:i + 64]
            x, mask, G, e_g, times = self._stack(ee, [t.G for t in rr])
            red = self.decomposer.redistribute(x, mask, G, torch.ones(len(ee), dtype=torch.bool, device=x.device),
                                               e_g, times)
            err = red.conservation_error(G)
            for j, (e, t) in enumerate(zip(ee, rr)):
                n = len(e)
                out[t.objective_id] = RudderCredit(
                    t.objective_id, list(e.exp_ids), red.signal[j, :n].tolist(), red.return_to_go[j, :n].tolist(),
                    red.contrib[j, :n].tolist(), float(G[j]), float(red.residual[j]), float(err[j]))
        return out


# ------------------------------------------------------------------------------------------------ COCOA provider
@dataclass
class CocoaCredit:
    exp_id: str
    status: str                              # "supported" | "unsupported"
    reasons: List[str]
    value: Optional[float]                   # credit of the executed candidate (None when unsupported)
    uncertainty: Optional[float]
    own: float                               # the decision's own declared outcomes (attributed directly)
    per_candidate: Optional[List[float]]     # counterfactual estimate for every recorded candidate
    per_candidate_std: Optional[List[float]] = None
    n_later_outcomes: int = 0
    clipped: bool = False


class HindsightProvider:
    """``LossProvider`` for the hindsight model h(a | b_t, u, elapsed) on stored trajectories.

    A training pair is a decision t and an outcome event of the same objective observed after t that t did not
    declare itself: declared by a later decision, or by an earlier one and observed after t (Meulemans et al. 2023,
    Eq. 4 sums over every later reward R_{t+k}). The decision's own outcomes need no hindsight: they are attributed to
    the executed candidate directly. Beliefs and candidate embeddings are detached (the hindsight model is a head)."""
    timescale = TIMESCALE_FREQUENT
    adapter_grad = False
    needs_latents = False

    def __init__(self, model: LatentModelAPI, store: SqliteExperienceStore, *, beta: float = 0.0, weight: float = 1.0,
                 d_hidden: int = 128, d_key: int = 64, n_members: int = 4, max_trajectories: int = 8, max_pairs: int = 512,
                 max_len: int = 64, min_prob: float = 1e-3, clip: float = 10.0, require_closed: bool = True,
                 verified_only: bool = False, head_name: str = "credit_cocoa", name: str = COCOA, seed: int = 0,
                 encoder: Optional[TrajectoryEncoder] = None):
        self.name, self.weight = name, float(weight)
        self.model, self.store, self.beta = model, store, float(beta)
        self.max_trajectories, self.max_pairs = int(max_trajectories), int(max_pairs)
        self.min_prob, self.clip = float(min_prob), float(clip)
        self.encoder = encoder or TrajectoryEncoder(model, store, max_len=max_len)
        self.returns = _ReturnCache(store, self.beta, require_closed, verified_only)
        d = model.d_model
        self.hindsight: HindsightModel = model.register_head(head_name, HindsightModel(
            d, U_DIM, d, d_hidden=d_hidden, d_key=d_key, n_members=n_members, n_elapsed=2, seed=7_000_003 + seed))
        self.hindsight.to(model.device)
        #: objectives whose trajectories entered a training loss of the hindsight model
        self.trained_objectives: Set[str] = set()

    def state_dict(self) -> Dict[str, Any]:
        return {"trained_objectives": sorted(self.trained_objectives)}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.trained_objectives = set(st.get("trained_objectives", []))

    def eligible(self, record: ExperienceRecord) -> bool:
        return is_top_level(record) and bool(record.decision.probabilities) and \
            self.returns.get(record.objective_id).matured

    # ---- pairs ----------------------------------------------------------------------------------------------
    def _pairs(self, enc: EncodedTrajectory, tr: TrajectoryReturn) -> List[Tuple[int, int]]:
        """(decision index, event index) for every event of the trajectory that decision t did not declare and that
        was observed after t: declared by a later decision (observed at or after t), or declared by an earlier
        decision and observed strictly after t (it is a later reward R_{t+k} of t as well)."""
        pos = {e: i for i, e in enumerate(enc.exp_ids)}
        full = {e: i for i, e in enumerate(tr.exp_ids)}
        out = []
        for k, ev in enumerate(tr.events):
            if ev.index < 0:
                continue
            for e, t in pos.items():
                own, ts = full.get(e), float(enc.ts[t])
                if own is None or own == ev.index:
                    continue
                if (own < ev.index and ts <= ev.observed_at) or (own > ev.index and ts < ev.observed_at):
                    out.append((t, k))
        return out

    def _rows(self, enc: EncodedTrajectory, tr: TrajectoryReturn, pairs: Sequence[Tuple[int, int]],
              detach: bool = True) -> Dict[str, Tensor]:
        dev = self.model.device
        t = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
        vals = torch.tensor([tr.events[p[1]].value for p in pairs], dtype=torch.float64, device=dev)
        when = torch.tensor([tr.events[p[1]].observed_at for p in pairs], dtype=torch.float64, device=dev)
        # decisions taken after t up to the observation (k - 1 in R_{t+k}): the same definition for events declared
        # by later and by earlier decisions
        times = sorted(tr.times)
        steps = torch.tensor([bisect.bisect_right(times, tr.events[p[1]].observed_at) -
                              bisect.bisect_right(times, float(enc.ts[p[0]])) for p in pairs], dtype=torch.float64,
                             device=dev).clamp_min(0.0)
        secs = (when - enc.ts[t]).clamp_min(0.0)
        probs = enc.probs[t]
        b, cand = (enc.b[t].detach(), enc.e_a[t].detach()) if detach else (enc.b[t], enc.e_a[t])
        return {"b": b, "u": outcome_features(vals).to(dev), "cand": cand,
                "mask": enc.a_mask[t], "logp": probs.clamp_min(1e-30).log(), "taken": enc.chosen[t],
                "elapsed": torch.stack([secs, steps], -1), "value": vals, "secs": secs, "t": t}

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Optional[Tensor], Dict[str, float]]:
        rows: List[Dict[str, Tensor]] = []
        n_traj = 0
        for oid in _batch_objectives(batch, self.max_trajectories):
            tr = self.returns.get(oid)
            if not tr.matured:
                continue
            enc = self.encoder.encode(oid)
            if enc is None:
                continue
            pairs = [p for p in self._pairs(enc, tr) if enc.has_probs[p[0]]]
            if pairs:
                rows.append(self._rows(enc, tr, pairs))
                n_traj += 1
                if _training_call():
                    self.trained_objectives.add(oid)
        if not rows:
            return None, {"n_pairs": 0.0}
        b, u, cand, mask, logp, taken, el = self._stack_rows(rows)
        ok = (logp.masked_fill(~mask, float("-inf")).exp().sum(1) - 1.0).abs() < 1e-3      # a recorded distribution
        if int(ok.sum()) == 0:
            return None, {"n_pairs": 0.0}
        idx = torch.nonzero(ok).flatten()
        if idx.numel() > self.max_pairs:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:self.max_pairs]]
        # bootstrap weights (Poisson draws from the global generator) only for a training step: a held-out
        # evaluation is deterministic, so the guard's paired before / after comparison measures the step (DX round 2)
        loss, m = self.hindsight.loss(b[idx], u[idx], cand[idx], mask[idx], taken[idx], logp[idx], el[idx],
                                      bootstrap=_training_call())
        m.update({"n_pairs": float(idx.numel()), "n_trajectories": float(n_traj)})
        return loss / len(self.hindsight.members), m

    @staticmethod
    def _stack_rows(rows: Sequence[Dict[str, Tensor]]) -> Tuple[Tensor, ...]:
        n = max(r["cand"].shape[1] for r in rows)

        def cat(k: str) -> Tensor:
            parts = []
            for r in rows:
                t = r[k]
                if k in ("cand", "mask", "logp") and t.shape[1] < n:
                    fill = float("-inf") if k == "logp" else 0
                    t = torch.cat([t, t.new_full((t.shape[0], n - t.shape[1]) + t.shape[2:], fill)], 1)
                parts.append(t)
            return torch.cat(parts)

        return tuple(cat(k) for k in ("b", "u", "cand", "mask", "logp", "taken", "elapsed"))

    def adapter_sensitivity_loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any, *,
                                 train_ok: Optional[Callable[[str], bool]] = None, max_trajectories: int = 2,
                                 max_len: int = 8, grad_last: Optional[int] = 8) -> Optional[Tensor]:
        """The hindsight loss with a graph into the semantic adapters (beliefs and candidate embeddings attached),
        on the newest ``max_len`` decisions of at most ``max_trajectories`` matured objectives of ``batch``, over
        pairs whose decision is training data (``train_ok(exp_id)``). It never trains the hindsight model and is not
        training data of COCOA: the slow step uses its gradient only to keep its update from raising this loss (the
        hindsight model reads the representation the adapters change; DX round 2). No random draw."""
        rows: List[Dict[str, Tensor]] = []
        for oid in _batch_objectives(batch, max_trajectories):
            tr = self.returns.get(oid)
            if not tr.matured:
                continue
            enc = self.encoder.encode(oid, semantic=True, last=max_len, grad_last=grad_last)
            if enc is None:
                continue
            pairs = [p for p in self._pairs(enc, tr) if enc.has_probs[p[0]]
                     and (train_ok is None or train_ok(enc.exp_ids[p[0]]))]
            if pairs:
                rows.append(self._rows(enc, tr, pairs, detach=False))
        if not rows:
            return None
        b, u, cand, mask, logp, taken, el = self._stack_rows(rows)
        ok = (logp.masked_fill(~mask, float("-inf")).exp().sum(1) - 1.0).abs() < 1e-3
        idx = torch.nonzero(ok).flatten()[:self.max_pairs]
        if idx.numel() == 0:
            return None
        loss, _ = self.hindsight.loss(b[idx], u[idx], cand[idx], mask[idx], taken[idx], logp[idx], el[idx],
                                      bootstrap=False)
        return loss / len(self.hindsight.members)

    # ---- credit ---------------------------------------------------------------------------------------------
    def credit(self, objective_ids: Sequence[str]) -> Dict[str, CocoaCredit]:
        """exp_id -> counterfactual credit for every top-level decision of the *matured* objectives.

        credit(a) = [a is the executed candidate] * own outcomes + sum over later outcomes of
        w(b_t, a, u_k) * v_k * exp(-beta * (tau_k - t)). Decisions without usable propensities are reported
        ``unsupported`` with the reasons and carry no number. Every top-level decision of the trajectory is covered
        (``max_len`` bounds training windows only)."""
        out: Dict[str, CocoaCredit] = {}
        for oid in objective_ids:
            tr = self.returns.get(oid)
            if not tr.matured:
                continue
            enc = self.encoder.encode(oid, full=True)
            if enc is None:
                continue
            T, N = enc.a_mask.shape
            own = [0.0] * T
            pos = {e: i for i, e in enumerate(enc.exp_ids)}
            full = {e: i for i, e in enumerate(tr.exp_ids)}
            for ev in tr.events:
                for e, t in pos.items():
                    if full.get(e) == ev.index:
                        own[t] += ev.value * math.exp(-self.beta * max(ev.observed_at - float(enc.ts[t]), 0.0))
            later = torch.zeros(T, N, dtype=torch.float64)
            var = torch.zeros(T, N, dtype=torch.float64)
            n_later = [0] * T
            clipped = [False] * T
            status, reasons = [SUPPORTED] * T, [[] for _ in range(T)]
            pairs = self._pairs(enc, tr)
            # support is a property of the decision's recorded propensities, with or without later outcomes
            logp_all = torch.where(enc.a_mask, enc.probs.clamp_min(1e-30).log(), torch.zeros_like(enc.probs))
            for t in range(T):
                rep = check_support(logp_all[t:t + 1].cpu() if enc.has_probs[t] else None, enc.a_mask[t:t + 1].cpu(),
                                    min_prob=self.min_prob, open_ended=enc.open_ended[t],
                                    tracked_proposals=enc.tracked[t])
                status[t], reasons[t] = rep.status[0], list(rep.reasons[0])
            good = [p for p in pairs if status[p[0]] == SUPPORTED]
            if good:
                r = self._rows(enc, tr, good)
                cc = self.hindsight.contribution(r["b"], r["u"], r["cand"], r["mask"], r["logp"], r["elapsed"],
                                                 clip=self.clip, min_prob=self.min_prob)
                gamma = duration_discount(r["secs"], self.beta).cpu()
                amount = (r["value"].cpu() * gamma)[:, None]
                w = torch.nan_to_num(cc.w, nan=0.0)
                ws = torch.nan_to_num(cc.w_std, nan=0.0)
                tt = r["t"].cpu()
                later.index_add_(0, tt, w * amount)
                var.index_add_(0, tt, (ws * amount) ** 2)
                for i, t in enumerate(tt.tolist()):
                    n_later[t] += 1
                    clipped[t] = clipped[t] or bool(cc.clipped[i].any())
            for t, e in enumerate(enc.exp_ids):
                c = int(enc.chosen[t])
                if status[t] != SUPPORTED:
                    out[e] = CocoaCredit(e, UNSUPPORTED, reasons[t], None, None, own[t], None, None, n_later[t])
                    continue
                n = int(enc.a_mask[t].sum())
                per = later[t, :n].clone()
                per[c] += own[t]
                out[e] = CocoaCredit(e, SUPPORTED, [], float(per[c]), float(var[t, c].sqrt()), own[t], per.tolist(),
                                     var[t, :n].sqrt().tolist(), n_later[t], clipped[t])
        return out


# ------------------------------------------------------------------------------------------------ slow provider
def consequence_text(rec: ExperienceRecord, max_chars: int = 240) -> str:
    """The observed consequence of a decision as plain text (first result observation and its exact quantities)."""
    if rec.result is None or not rec.result.observations:
        return ""
    o = rec.result.observations[0]
    q = "; ".join(f"{x.name}={x.value:g}" for x in o.quantities[:4])
    return (o.text + (f" [{q}]" if q else ""))[:max_chars]


class BridgeGroundingProvider:
    """Slow-timescale ``LossProvider``: -log p_LM(observed consequence text | executed action, belief via bridge).

    The belief is detached (this loss trains the semantic adapters, not the experiential core). The consequence
    text is only ever a *target*: it is never part of a prompt."""
    timescale = TIMESCALE_SLOW

    def __init__(self, model: LatentModelAPI, *, weight: float = 1.0, max_rows: int = 4, max_chars: int = 240,
                 name: str = "bridge_grounding"):
        self.name, self.weight, self.model = name, float(weight), model
        self.max_rows, self.max_chars = int(max_rows), int(max_chars)

    def eligible(self, record: ExperienceRecord) -> bool:
        return bool(consequence_text(record, self.max_chars))

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Optional[Tensor], Dict[str, float]]:
        rows = [(i, consequence_text(r, self.max_chars)) for i, r in enumerate(batch.records)]
        rows = [(i, t) for i, t in rows if t][:self.max_rows]
        if not rows:
            return None, {"n_rows": 0.0}
        idx = torch.tensor([i for i, _ in rows], device=batch.b.device)
        prompts = [f"Action: {batch.records[i].decision.chosen.render()}\nObserved:" for i, _ in rows]
        targets = [" " + t for _, t in rows]
        lp = model.generation_logprob(prompts, targets, latent=batch.b.index_select(0, idx).detach())
        words = torch.tensor([max(1, len(t.split())) for t in targets], dtype=lp.dtype, device=lp.device)
        loss = -(lp / words).mean()
        return loss, {"n_rows": float(len(rows)), "nll_per_word": float(loss.detach())}


# ------------------------------------------------------------------------------------------------ assigner
@dataclass
class CreditUpdate:
    version: int
    reason: str
    objectives: List[str] = field(default_factory=list)
    matured: List[str] = field(default_factory=list)
    withheld: Dict[str, str] = field(default_factory=dict)            # objective -> why learned credit is withheld
    written: Dict[str, int] = field(default_factory=dict)             # estimator key -> records written
    changes: Dict[str, float] = field(default_factory=dict)           # estimator key -> mean |change| of stored credit
    conservation_error: float = 0.0                                   # max over matured objectives (RUDDER)
    unsupported: Dict[str, List[str]] = field(default_factory=dict)   # exp_id -> COCOA refusal reasons
    exp_ids: List[str] = field(default_factory=list)                  # records whose credit changed


@dataclass
class CreditedReturns:
    estimator: str
    key: str
    values: Tensor               # [B] float64; 0 where ``mask`` is False (masked, never a target)
    mask: Tensor                 # [B] bool: the record carries credit from this estimator
    uncertainty: Tensor          # [B] float64 (0 where unknown)


@dataclass
class ExperimentEvaluation(Record):
    """One pass of :meth:`CreditAssigner.evaluate_experiments`: which randomised experiments could be compared with
    the estimators on held-out units, why the others could not, and whether the estimator selection ran.

    ``ran`` is True only when at least one held-out randomised comparison exists and ``EstimatorSelector.select`` was
    evaluated on it; otherwise ``reason`` says why the selection did not run (it is never run on in-sample data)."""
    ts: float
    ran: bool
    reason: str
    default_before: str
    default_after: str
    n_experiments: int = 0                 # experiments in the experience store
    n_randomized: int = 0                  # of them randomised with >= 2 arms
    n_recorded: int = 0                    # experiment records written to the store by this pass
    n_compared: int = 0                    # comparisons that produced held-out evidence in this pass
    n_evidence: int = 0                    # held-out randomised comparisons the selector holds in total
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    selection: Optional[Dict[str, Any]] = None


class CreditAssigner:
    """Retrospective credit for stored decisions (spec 9.1-9.4).

    ``assign`` is called when outcomes matured (with the affected records, e.g. ``ReconcileReport.retrain_marked``
    or ``store.retrain_queue()``). It recomputes, for every affected objective:

    * ``simple``                credit from evidence links (available as soon as a linked outcome matured);
    * ``rudder`` / ``rudder_rtg`` / ``rudder_contribution``   only once the objective's return is final;
    * ``cocoa``                 only once the return is final and only where propensities support it.

    Every changed value is appended to the store's credit history under a new version; the record keeps the
    newest value. Credit that is no longer warranted is withdrawn, not zeroed."""

    def __init__(self, model: LatentModelAPI, store: SqliteExperienceStore, *,
                 rudder: Optional[ReturnDecompositionProvider] = None, cocoa: Optional[HindsightProvider] = None,
                 selector: Optional[EstimatorSelector] = None, beta: float = 0.0, clock: Optional[Clock] = None,
                 goal_relevant_only: bool = True):
        self.model, self.store, self.rudder, self.cocoa = model, store, rudder, cocoa
        self.beta, self.clock = float(beta), clock or WallClock()
        self.selector = selector or EstimatorSelector(clock=self.clock)
        self.goal_relevant_only = goal_relevant_only
        self.last: Optional[CreditUpdate] = None
        self._learners: List[Any] = []
        self._experiment_source: Any = None
        self._eval_fp: Optional[str] = None
        self._evidence_fp: Optional[str] = None
        self.last_evaluation: Optional[ExperimentEvaluation] = None
        self.min_units_per_arm = 2

    def attach_learner(self, learner: Any) -> None:
        """Called by ``ContinualLearner(credit=self)``: what the learner trained on is excluded from validation."""
        if all(l is not learner for l in self._learners):
            self._learners.append(learner)

    def trained_objectives(self) -> Set[str]:
        """Lifetimes (objective ids) that the estimators or the representation they read were trained on: RUDDER's
        and COCOA's own training trajectories and every objective of a learner update batch."""
        out: Set[str] = set()
        for prov in (self.rudder, self.cocoa):
            if prov is not None:
                out |= set(getattr(prov, "trained_objectives", ()))
        for l in self._learners:
            out |= set(getattr(l, "trained_objectives", ()))
        return out

    @property
    def default(self) -> str:
        """The estimator whose credit is used for learning right now (chosen by evidence, ``credit.select``)."""
        return self.selector.default

    @property
    def version(self) -> int:
        return int(self.store.kv_get("credit_version", 0))

    # ---- assignment -----------------------------------------------------------------------------------------
    def assign(self, exp_ids: Optional[Sequence[str]] = None, *, objective_ids: Optional[Sequence[str]] = None,
               reason: str = "outcome_matured") -> CreditUpdate:
        st = self.store
        if objective_ids is None:
            if exp_ids is None:
                objective_ids = st.objective_ids()
            else:
                seen = st.objective_of(exp_ids)
                objective_ids = list(dict.fromkeys(seen[e] for e in exp_ids if e in seen))
        objective_ids = [o for o in objective_ids if o]
        version = self.version + 1
        up = CreditUpdate(version=version, reason=reason, objectives=list(objective_ids))
        now = self.clock.now()
        for prov in (self.rudder, self.cocoa):
            if prov is not None:
                for o in objective_ids:
                    if not prov.returns.get(o).matured:
                        prov.returns.forget(o)
        rud = self.rudder.redistribute(objective_ids) if self.rudder is not None else {}
        coc = self.cocoa.credit(objective_ids) if self.cocoa is not None else {}
        totals: Dict[str, List[float]] = {}
        with st.batch():
            for oid in objective_ids:
                recs = st.trajectory(oid)
                ids = [r.exp_id for r in recs]
                links = [OutcomeLink.from_dict(l) for e in ids for l in st.links(exp_id=e)]
                simple = credit_from_links(links, beta=self.beta, goal_relevant_only=self.goal_relevant_only)
                rc = rud.get(oid)
                if rc is not None:
                    up.matured.append(oid)
                    up.conservation_error = max(up.conservation_error, rc.conservation_error)
                else:
                    src = self.rudder or self.cocoa
                    up.withheld[oid] = src.returns.get(oid, recs).reason if src is not None else "no learned estimator"
                rmap = {e: i for i, e in enumerate(rc.exp_ids)} if rc is not None else {}
                for r in recs:
                    vals: Dict[str, Optional[Tuple[float, Optional[float]]]] = {}
                    cv = simple.get(r.exp_id)
                    vals[SIMPLE] = None if cv is None else (cv.value, cv.uncertainty)
                    if self.rudder is not None:
                        i = rmap.get(r.exp_id)
                        if i is None:
                            vals[RUDDER] = vals[KEY_RTG] = vals[KEY_CONTRIB] = None
                        else:
                            vals[RUDDER] = (rc.signal[i], abs(rc.residual))
                            vals[KEY_RTG] = (rc.return_to_go[i], None)
                            vals[KEY_CONTRIB] = (rc.contribution[i], abs(rc.residual))
                    if self.cocoa is not None:
                        c = coc.get(r.exp_id)
                        if c is None or c.value is None:
                            vals[COCOA] = None
                            if c is not None:
                                up.unsupported[r.exp_id] = c.reasons
                        else:
                            vals[COCOA] = (c.value, c.uncertainty)
                    ch = st.set_credits(r.exp_id, vals, version=version, reason=reason, ts=now)
                    if ch:
                        up.exp_ids.append(r.exp_id)
                    for k, d in ch.items():
                        totals.setdefault(k, []).append(abs(d))
                    for k, v in vals.items():
                        if v is not None:
                            up.written[k] = up.written.get(k, 0) + 1
            st.kv_put("credit_version", version)
        up.changes = {k: sum(v) / len(v) for k, v in totals.items()}
        self.last = up
        return up

    def on_reconcile(self, report: Any) -> CreditUpdate:
        """Retrospective update after ``OutcomeLinker.reconcile``: the linked decisions and the earlier decisions of
        their objectives get their credit recomputed."""
        return self.assign(list(report.retrain_marked), reason="outcome_matured")

    # ---- reading --------------------------------------------------------------------------------------------
    def credited_returns(self, records: Sequence[ExperienceRecord], estimator: Optional[str] = None) -> CreditedReturns:
        """Per-decision credited return under ``estimator`` (default: the evidence-selected one). Records without
        credit from that estimator are masked; nothing is filled in."""
        est = estimator or self.default
        key = LEARNING_KEY.get(est, est)
        dev = self.model.device
        vals = [r.credit.get(key) for r in records]
        mask = torch.tensor([v is not None and math.isfinite(v) for v in vals], dtype=torch.bool, device=dev)
        v = torch.tensor([float(x) if m else 0.0 for x, m in zip(vals, mask.tolist())], dtype=torch.float64, device=dev)
        u = torch.tensor([float(r.credit_uncertainty.get(key, 0.0)) for r in records], dtype=torch.float64, device=dev)
        return CreditedReturns(est, key, v, mask, u)

    # ---- validation against randomised experiments ----------------------------------------------------------
    def _return_to_go(self, rec: ExperienceRecord, tr: TrajectoryReturn) -> float:
        t = float(rec.decision.ts)
        return sum(e.value * math.exp(-self.beta * max(e.observed_at - t, 0.0)) for e in tr.events if e.observed_at >= t)

    def compare_with_experiment(self, experiment_id: str, *, treatment: str, control: str, confidence: float = 0.95,
                                exclude_objectives: Optional[Iterable[str]] = None) -> Optional[ExperimentEvidence]:
        """Measured arm difference of a registered experiment (matured return-to-go of its randomised decisions)
        next to each estimator's estimate of it, added to the selector's evidence. ``None`` when either arm has
        fewer than two matured held-out units.

        Held out by lifetime (enforced here): a unit whose objective any estimator or attached learner trained on
        (:meth:`trained_objectives`, plus ``exclude_objectives``) is left out of the measured difference *and* of
        every estimate; the evidence records how many were excluded and is marked ``held_out``. Each estimator is
        compared on the same units with the paired standard error (``select.paired_comparison``)."""
        st = self.store
        exp = st.experiment(experiment_id)
        recs = st.experiment_records(experiment_id)
        src = self.rudder or self.cocoa
        returns = src.returns if src is not None else _ReturnCache(st, self.beta, True, False)
        trained = self.trained_objectives() | set(exclude_objectives or ())
        arms: Dict[str, List[ExperienceRecord]] = {treatment: [], control: []}
        G: Dict[str, float] = {}
        excluded: List[str] = []
        randomized = exp is not None and exp.assignment == "randomized"
        for r in recs:
            arm = record_arm(r)
            if arm not in arms:
                continue
            tr = returns.get(r.objective_id)
            if not tr.matured:
                continue
            if r.objective_id in trained:
                excluded.append(r.objective_id)
                continue
            arms[arm].append(r)
            G[r.exp_id] = self._return_to_go(r, tr)
            randomized = randomized and r.decision.exploration == "randomized" and bool(r.decision.probabilities)
        measured = arm_difference([G[r.exp_id] for r in arms[treatment]], [G[r.exp_id] for r in arms[control]],
                                  confidence=confidence, randomized=randomized, treatment_name=treatment,
                                  control_name=control, experiment_id=experiment_id, metric="return_to_go")
        if measured is None:
            return None
        est: Dict[str, float] = {}
        se: Dict[str, float] = {}
        pd: Dict[str, float] = {}
        ps: Dict[str, float] = {}
        pn: Dict[str, int] = {}
        for name, key in VALIDATION_KEY.items():
            a = [r.credit[key] for r in arms[treatment] if key in r.credit]
            b = [r.credit[key] for r in arms[control] if key in r.credit]
            d = arm_difference(a, b)
            if d is not None:
                est[name], se[name] = d.diff, d.se
            units = [(arm, float(r.credit[key]), G[r.exp_id]) for arm in (treatment, control) for r in arms[arm]
                     if key in r.credit]
            pc = paired_comparison(units, treatment, control, kind=ARM_DIFFERENCE)
            if pc is not None:
                pd[name], ps[name], pn[name] = pc
        if self.cocoa is not None:
            units_r = arms[treatment] + arms[control]
            cc = self.cocoa.credit(list(dict.fromkeys(r.objective_id for r in units_r)))
            diffs, cunits = [], []
            for r in units_r:
                c, idx = cc.get(r.exp_id), candidate_arms(r)
                if c is None or c.per_candidate is None or treatment not in idx or control not in idx:
                    continue
                # contrast of the *later* outcomes only: the decision's own outcome belongs to the executed arm
                per = list(c.per_candidate)
                per[r.decision.chosen_index] -= c.own
                d = per[idx[treatment]] - per[idx[control]]
                own_t = c.own if r.decision.chosen_index == idx[treatment] else 0.0
                own_c = c.own if r.decision.chosen_index == idx[control] else 0.0
                p = r.decision.probabilities
                x = d + own_t / max(p[idx[treatment]], 1e-12) - own_c / max(p[idx[control]], 1e-12)
                diffs.append(x)
                cunits.append((record_arm(r), x, G[r.exp_id]))
            if len(diffs) >= 2:
                m = sum(diffs) / len(diffs)
                sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
                est[COCOA], se[COCOA] = m, sd / math.sqrt(len(diffs))
                pc = paired_comparison(cunits, treatment, control, kind=UNIT_MEAN)
                if pc is not None:
                    pd[COCOA], ps[COCOA], pn[COCOA] = pc
        n_units = len(arms[treatment]) + len(arms[control])
        ev = ExperimentEvidence.from_difference(
            measured, est, se, held_out=True, n_excluded_trained=len(excluded), paired_diff=pd, paired_se=ps,
            paired_n=pn, holdout={"split": "lifetime", "units": n_units, "excluded_trained": len(excluded),
                                  "trained_objectives_known": len(trained)})
        self.selector.add_evidence(ev)
        return ev

    def select(self) -> SelectionRecord:
        """Re-evaluate which estimator is the training default from all randomised evidence added so far."""
        return self.selector.select()

    # ---- experiments recorded elsewhere (the agent's registry) -----------------------------------------------
    def attach_experiments(self, source: Any) -> None:
        """Where the caller records its experiments: an object with ``records()`` (``predict.registry.
        ExperimentRegistry``), a callable returning ``ExperimentRecord`` objects, or an iterable of them. Once a
        source is attached, an attached :class:`~cam_live.credit.learner.ContinualLearner` copies the experiments
        into the experience store before every update (so randomised lifetimes are known to the held-out split) and
        runs :meth:`evaluate_experiments` after it (RC-08, G2L-6)."""
        self._experiment_source = source

    @property
    def evaluates_experiments(self) -> bool:
        return self._experiment_source is not None

    def _source_records(self) -> List[ExperimentRecord]:
        src = self._experiment_source
        if src is None:
            return []
        if hasattr(src, "records"):
            got = src.records()
        elif callable(src):
            got = src()
        else:
            got = src
        return [e for e in (got or []) if isinstance(e, ExperimentRecord)]

    def record_experiments(self, experiments: Optional[Iterable[ExperimentRecord]] = None) -> int:
        """Copy experiment records (default: the attached source) into the experience store; a record is written
        when it is new or changed (status, outcomes ...). Returns the number written."""
        exps = list(experiments) if experiments is not None else self._source_records()
        st, n = self.store, 0
        for e in exps:
            old = st.experiment(e.experiment_id)
            if old is None or json.dumps(old.to_dict(), sort_keys=True, default=str) != \
                    json.dumps(e.to_dict(), sort_keys=True, default=str):
                st.put_experiment(e)
                n += 1
        return n

    def _units(self, exp: ExperimentRecord, recs: Sequence[ExperienceRecord], trained: Set[str], arm: str,
               returns: "_ReturnCache") -> Dict[str, int]:
        """Unit accounting of one arm, with the filters of :meth:`compare_with_experiment`."""
        out = {"assigned": 0, "return_not_final": 0, "lifetime_trained_on": 0, "not_randomized": 0, "held_out": 0}
        for r in recs:
            if record_arm(r) != arm:
                continue
            out["assigned"] += 1
            if not returns.get(r.objective_id).matured:
                out["return_not_final"] += 1
            elif r.objective_id in trained:
                out["lifetime_trained_on"] += 1
            elif not (exp.assignment == "randomized" and r.decision.exploration == "randomized"
                      and bool(r.decision.probabilities)):
                out["not_randomized"] += 1
            else:
                out["held_out"] += 1
        return out

    def evaluate_experiments(self, experiments: Optional[Iterable[ExperimentRecord]] = None, *,
                             confidence: float = 0.95, force_log: bool = False) -> ExperimentEvaluation:
        """Compare the estimators with every registered randomised experiment whose arms have at least
        ``min_units_per_arm`` matured held-out units, then re-run the evidence-based selection when the held-out
        evidence changed (spec 9.4; RC-08). Experiments given here or by the attached source are recorded in the
        store first. Comparisons are held out by lifetime exactly as in :meth:`compare_with_experiment`; nothing
        in-sample is ever scored. Every pass whose outcome differs from the previous one is logged to the
        selector's log (with the selection when it ran, else with the reason it did not)."""
        st = self.store
        n_rec = self.record_experiments(experiments)
        before = self.selector.default
        trained = self.trained_objectives()
        src = self.rudder or self.cocoa
        returns = src.returns if src is not None else _ReturnCache(st, self.beta, True, False)
        rows: List[Dict[str, Any]] = []
        exps = st.experiments()
        n_rand = n_cmp = 0
        for exp in exps:
            arms = [str(a) for a in exp.arms]
            row: Dict[str, Any] = {"experiment_id": exp.experiment_id, "assignment": exp.assignment, "arms": arms,
                                   "status": exp.status}
            if exp.assignment != "randomized" or len(arms) < 2:
                row.update(compared=False, reason="not a randomised experiment with two or more arms")
                rows.append(row)
                continue
            n_rand += 1
            recs = st.experiment_records(exp.experiment_id)
            row["decisions"] = len(recs)
            pairs = []
            for other in arms[1:]:
                treatment, control = arms[0], other
                units = {a: self._units(exp, recs, trained, a, returns) for a in (treatment, control)}
                pr: Dict[str, Any] = {"treatment": treatment, "control": control, "units": units}
                short = [a for a, u in units.items() if u["held_out"] < self.min_units_per_arm]
                if short:
                    pr.update(compared=False, reason="; ".join(
                        f"arm {a!r}: {units[a]['held_out']} matured held-out units (< {self.min_units_per_arm}; "
                        f"assigned {units[a]['assigned']}, return not final {units[a]['return_not_final']}, lifetime "
                        f"trained on {units[a]['lifetime_trained_on']}, not randomised {units[a]['not_randomized']})"
                        for a in short))
                else:
                    ev = self.compare_with_experiment(exp.experiment_id, treatment=treatment, control=control,
                                                      confidence=confidence)
                    if ev is None:
                        pr.update(compared=False, reason="the measured arm difference is undefined")
                    else:
                        n_cmp += 1
                        pr.update(compared=True, held_out=ev.held_out, measured=ev.diff, se=ev.se,
                                  estimates=dict(ev.estimates), paired_n=dict(ev.paired_n),
                                  n_excluded_trained=ev.n_excluded_trained)
                pairs.append(pr)
            row["comparisons"] = pairs
            row["compared"] = any(p.get("compared") for p in pairs)
            if not row["compared"]:
                row["reason"] = " | ".join(p["reason"] for p in pairs)
            rows.append(row)
        used = self.selector._used()
        # the selection is re-run when the held-out *measurement* changes (a new comparison, more or other units),
        # not each time a still-training estimator's estimate moves (that is picked up with the next measurement)
        ev_fp = hashlib.sha256(json.dumps(sorted(
            (e.key, e.diff, e.se, e.n_treatment, e.n_control, e.n_excluded_trained, sorted(e.estimates))
            for e in used), default=str).encode()).hexdigest()
        ran, reason, sel = False, "", None
        if not exps:
            reason = "no experiment is recorded in the experience store"
        elif not n_rand:
            reason = "no randomised experiment with two or more arms"
        elif not used:
            reason = (f"no randomised experiment has {self.min_units_per_arm} or more matured held-out units in each "
                      f"arm (held out by lifetime: units whose lifetime an estimator or the learner trained on are "
                      f"excluded); the default stays {before}")
        report = ExperimentEvaluation(ts=self.clock.now(), ran=False, reason=reason, default_before=before,
                                      default_after=before, n_experiments=len(exps), n_randomized=n_rand,
                                      n_recorded=n_rec, n_compared=n_cmp, n_evidence=len(used), experiments=rows)
        if used and (ev_fp != self._evidence_fp or force_log):
            sel = self.selector.select(context=report.to_dict())
            self._evidence_fp = ev_fp
            report.ran, report.default_after = True, sel.default_after
            report.reason = "; ".join(sel.reasons) or "selection ran on held-out randomised evidence"
            report.selection = sel.to_dict()
            self._eval_fp = None
        elif used:                 # the last selection stands; nothing new to log
            report.ran, report.reason = False, "held-out evidence unchanged since the last selection"
            report.default_after = self.selector.default
        fp = hashlib.sha256(json.dumps({k: v for k, v in report.to_dict().items() if k not in ("ts", "n_recorded")},
                                       sort_keys=True, default=str).encode()).hexdigest()
        if not used and (fp != self._eval_fp or force_log):
            self.selector.log_row({"evaluation": report.to_dict(), "selection": None})
            self._eval_fp = fp
        self.last_evaluation = report
        return report

    # ---- persistence ----------------------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"selector": self.selector.state_dict(), "credit_version": self.version, "beta": self.beta}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.selector.load_state_dict(st.get("selector", {}))
