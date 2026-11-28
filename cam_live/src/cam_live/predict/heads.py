"""cam_live.predict.heads — Workstream B's heads on the real model, and the tensors they share (stage 2).

``attach_heads(model)`` builds the world-model heads, the latent transition, the goal evaluator, the candidate
policy and the downstream-usefulness estimator at the model's width and registers them with
``model.register_head`` (group ``heads``), so they are counted, optimised, checkpointed and hashed with the model.
It is idempotent: a second call returns the modules that are already registered.

``world_batch`` is the single place where stored queries and outcomes become world-model tensors
(``encode_queries`` -> ``features`` -> ``WorldHeads``), used by the loss providers and by the runtime ``Predictor``
alike, so training and prediction cannot drift apart. ``fast_inputs`` defines how the Bayesian fast layer of
Workstream A is combined with the heads:

    count           the head's rate is an *offset*: the fast layer models  log(actual / head)  through the exposure
    real            the fast layer models the residual  (y - head location)  in the metric's normalised units
    time_to_event   the fast layer models the log hazard ratio against the head's expected time (time rescaled)
    binary          no offset is available in the fast layer: it models the logit itself and the *shift* of its
                    posterior from its prior is added to the head's logit (an approximation, stated as such)
    categorical     head only (the fast layer has no categorical likelihood)
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..config import CamLiveConfig
from ..contracts import OutcomeKind, OutcomeObservation, OutcomeQuery
from ..nn.api import GROUP_HEADS, LatentModelAPI
from .evaluator import EvaluatorConfig, GoalEvaluator
from .policy import CandidatePolicy, PolicyConfig
from .voi import DownstreamValueEstimator
from .world import (K_BINARY, K_COUNT, K_REAL, K_TTE, KIND_INDEX, KindNormaliser, LatentTransition, QueryMeta,
                    WorldConfig, WorldHeads, WorldOutput, WorldTargets, build_targets)

F64 = torch.float64
HEAD_WORLD = "predict_world"
HEAD_TRANSITION = "predict_transition"
HEAD_EVALUATOR = "predict_evaluator"
HEAD_POLICY = "predict_policy"
HEAD_DOWNSTREAM = "predict_downstream"
HEAD_NAMES = (HEAD_WORLD, HEAD_TRANSITION, HEAD_EVALUATOR, HEAD_POLICY, HEAD_DOWNSTREAM)
#: outcome kinds the fast layer can adapt (mirrors ``cam_live.nn.fast.FAST_KINDS``)
FAST_KINDS = (OutcomeKind.REAL, OutcomeKind.COUNT, OutcomeKind.BINARY, OutcomeKind.TIME_TO_EVENT)
ACTION_PLACEHOLDER = "{action}"
#: signals fed to the downstream-usefulness estimator next to the decision latent
VOI_SIGNALS = ("voi_fast", "info_gain_nats", "epistemic_std", "cost_money", "cost_seconds")


@dataclass
class PredictConfig:
    """Settings of ``cam_live.predict`` (section ``predict`` of the configuration)."""
    hidden: int = 192
    n_members: int = 5
    transition_hidden: int = 256
    transition_members: int = 3
    n_hazard_bins: int = 12
    max_categories: int = 16
    use_phi: bool = True             # the heads also read the fast-layer features phi
    prior_scale: float = 1.0         # randomised prior functions of the ensemble (0 disables)
    # selection
    eps_floor: float = 0.02
    temperature: float = 1.0
    ucb_c: float = 1.0
    ig_weight: float = 1.0
    thompson_samples: int = 32
    cost_weight: float = 0.0         # value units per unit of money
    time_weight: float = 0.0         # value units per second (opportunity cost)
    # value of information
    voi_samples: int = 8
    voi_max_candidates: int = 16
    voi_slow_path: bool = False      # also imagine the result entering the context (one core pass per sample)
    # horizons
    default_horizon_s: float = 14 * 86400.0
    # losses
    fast_prior_weight: float = 0.25
    transition_weight: float = 0.25
    allowed_provenance: Tuple[str, ...] = ("live", "staging", "diagnostic")
    adapter_grad_last: int = 2       # newest context items that carry gradient into the semantic adapters
    adapter_max_records: int = 4
    resource_scale: Tuple[float, ...] = (3600.0, 1.0, 60.0)       # seconds, money, compute seconds
    # imagination
    imagination_particles: int = 8
    max_imagination_horizon: int = 4
    imagination_threshold: float = 0.5
    imagination_criterion: str = "value"      # what must be measured accurate for k imagined steps: "value" (the
                                             # world model's values of the candidates read at the imagined belief
                                             # vs at the real one, relative to their spread) or "latent" (belief)
    imagination_select: bool = True          # imagined continuation values enter rule thompson_policy, weighted by
                                             # the ImaginationCalibration (False: ablation, weight 0; still recorded)
    imagination_continuation: str = "search"  # how imagined steps k >= 1 continue: "search" (every feasible action
                                             # at each imagined belief while imagination_max_rows allows, else the
                                             # best-valued one plus a random subset; the best sequence counts),
                                             # "greedy" (best value at each imagined belief) or "policy" (the
                                             # policy's most probable action)
    imagination_max_rows: int = 0            # world-model rows per imagined level (bounds the search's breadth);
                                             # 0: 32768 on a GPU, 8192 on the CPU (measured at debug scale on one
                                             # CPU thread: 32768 rows ~4 s per level, 8192 ~0.5 s)
    imagination_aleatoric: bool = False      # add the transition's aleatoric noise to imagined beliefs
    imagination_prior_sd: float = 0.5        # prior sd of beta_im (and beta_q) of the ImaginationCalibration
    seed: int = 0

    @classmethod
    def from_cfg(cls, cfg: Optional[CamLiveConfig]) -> "PredictConfig":
        sec = {} if cfg is None else cfg.section("predict")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(sec) - known)
        if unknown:
            raise KeyError(f"unknown keys in the 'predict' configuration section: {unknown}")
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sec.items()}
        if cfg is not None and cfg.core.d_model < 128:           # debug-scale heads unless stated otherwise
            for k, v in (("hidden", 64), ("n_members", 3), ("transition_hidden", 96), ("n_hazard_bins", 6),
                         ("max_categories", 8)):
                kw.setdefault(k, v)
        if cfg is not None:
            kw.setdefault("seed", cfg.seed)
        return cls(**kw)


class RealScaler:
    """Running location / scale of every real-valued metric, so the heads work in normalised units and report in
    the metric's own units. Plain numbers (not parameters); saved with :meth:`state`."""

    def __init__(self, momentum: float = 0.98):
        self.momentum = momentum
        self.stats: Dict[str, List[float]] = {}                   # metric -> [n, mean, var]

    def update(self, metric: str, value: float) -> None:
        v = float(value)
        s = self.stats.get(metric)
        if s is None:
            self.stats[metric] = [1.0, v, 0.0]
            return
        n = s[0] + 1.0
        w = max(1.0 / n, 1.0 - self.momentum)
        d = v - s[1]
        s[1] += w * d
        s[2] = (1.0 - w) * (s[2] + w * d * d)                      # exponentially weighted variance
        s[0] = n

    def lookup(self, metric: str) -> Tuple[float, float]:
        s = self.stats.get(metric)
        if s is None or s[0] < 2:
            return (0.0 if s is None else s[1]), 1.0
        return s[1], max(math.sqrt(max(s[2], 0.0)), 1e-6)

    def tensors(self, rows: Sequence[Sequence[OutcomeQuery]], device: torch.device) -> Tuple[Tensor, Tensor]:
        b, q = len(rows), max(1, max((len(r) for r in rows), default=1))
        loc, scale = torch.zeros(b, q), torch.ones(b, q)
        for i, row in enumerate(rows):
            for j, query in enumerate(row):
                if query.kind == OutcomeKind.REAL:
                    loc[i, j], scale[i, j] = self.lookup(query.metric)
        return loc.to(device), scale.to(device)

    def state(self) -> Dict[str, Any]:
        return {"momentum": self.momentum, "stats": {k: list(v) for k, v in self.stats.items()}}

    def load_state(self, s: Dict[str, Any]) -> None:
        self.momentum = float(s.get("momentum", self.momentum))
        self.stats = {k: list(v) for k, v in s.get("stats", {}).items()}


@dataclass
class PredictHeads:
    cfg: PredictConfig
    world: WorldHeads
    transition: LatentTransition
    evaluator: GoalEvaluator
    policy: CandidatePolicy
    downstream: DownstreamValueEstimator
    scaler: RealScaler = field(default_factory=RealScaler)

    @property
    def normaliser(self) -> KindNormaliser:
        return self.world.normaliser                              # buffers saved with the world head

    def modules(self) -> Dict[str, nn.Module]:
        return {HEAD_WORLD: self.world, HEAD_TRANSITION: self.transition, HEAD_EVALUATOR: self.evaluator,
                HEAD_POLICY: self.policy, HEAD_DOWNSTREAM: self.downstream}

    def parameters(self) -> List[nn.Parameter]:
        return [p for m in self.modules().values() for p in m.parameters()]


def _existing(model: LatentModelAPI, name: str) -> Optional[nn.Module]:
    try:
        return model.head(name)
    except (KeyError, AttributeError):
        return None


def attach_heads(model: LatentModelAPI, pcfg: Optional[PredictConfig] = None) -> PredictHeads:
    """Build (deterministically from ``pcfg.seed``) and register B's heads on ``model``; reuse them if present."""
    pcfg = pcfg or PredictConfig.from_cfg(getattr(model, "cfg", None))
    have = {n: _existing(model, n) for n in HEAD_NAMES}
    if all(m is not None for m in have.values()):
        heads = PredictHeads(pcfg, have[HEAD_WORLD], have[HEAD_TRANSITION], have[HEAD_EVALUATOR], have[HEAD_POLICY],
                             have[HEAD_DOWNSTREAM])
        heads.scaler = getattr(have[HEAD_WORLD], "_real_scaler", heads.scaler)
        have[HEAD_WORLD]._real_scaler = heads.scaler
        return heads
    if any(m is not None for m in have.values()):
        raise RuntimeError("only some of the predict heads are registered on this model")
    d = model.d_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2_000_003 + int(pcfg.seed))
        world = WorldHeads(WorldConfig(d_model=d, n_features=model.n_features if pcfg.use_phi else 0,
                                       hidden=pcfg.hidden, n_members=pcfg.n_members,
                                       n_hazard_bins=pcfg.n_hazard_bins, max_categories=pcfg.max_categories,
                                       prior_scale=pcfg.prior_scale, prior_hidden=max(16, pcfg.hidden // 3),
                                       seed=pcfg.seed))
        world.normaliser = KindNormaliser()
        transition = LatentTransition(d, hidden=pcfg.transition_hidden, n_members=pcfg.transition_members,
                                      seed=pcfg.seed + 1)
        evaluator = GoalEvaluator(EvaluatorConfig(d_model=d, hidden=pcfg.hidden, n_time_bins=pcfg.n_hazard_bins,
                                                  n_resources=len(pcfg.resource_scale)))
        policy = CandidatePolicy(PolicyConfig(d_model=d, hidden=pcfg.hidden))
        downstream = DownstreamValueEstimator(d, n_signals=len(VOI_SIGNALS), hidden=max(32, pcfg.hidden // 2))
    heads = PredictHeads(pcfg, world, transition, evaluator, policy, downstream)
    for name, mod in heads.modules().items():
        model.register_head(name, mod, GROUP_HEADS)
    world._real_scaler = heads.scaler
    return heads


# ----------------------------------------------------------------------------------------------------------------
# queries and outcomes -> tensors
# ----------------------------------------------------------------------------------------------------------------
def canonical_query(q: OutcomeQuery, action_ids: Iterable[str] = ()) -> OutcomeQuery:
    """The query with every occurrence of an action id in its metric replaced by a placeholder, so the metric
    name (and its hashed embedding) is shared by all actions instead of being unique per action."""
    metric = q.metric
    for aid in action_ids:
        if aid and aid in metric:
            metric = metric.replace(aid, ACTION_PLACEHOLDER)
    return q if metric == q.metric else dataclasses.replace(q, metric=metric)


@dataclass
class WorldBatch:
    """World-model tensors for B rows of one executed (or candidate) action each: N = 1 everywhere."""
    queries: List[List[OutcomeQuery]]        # canonical queries
    meta: QueryMeta
    targets: WorldTargets
    e_q: Tensor                              # [B, Q, D]
    q_mask: Tensor                           # [B, Q]
    kind_ids: Tensor                         # [B, Q]
    horizon_s: Tensor                        # [B, Q]
    phi: Tensor                              # [B, 1, Q, F]
    out: WorldOutput                         # members' parameters, [M, B, 1, Q, ...]
    real_loc: Tensor
    real_scale: Tensor

    def mask(self) -> Tensor:
        return self.targets.mask(self.kind_ids) & self.q_mask


def world_batch(model: LatentModelAPI, heads: PredictHeads, z: Tensor,
                rows: Sequence[Sequence[Tuple[OutcomeQuery, Optional[OutcomeObservation]]]], *,
                update_scaler: bool = False) -> WorldBatch:
    """``z`` [B, D] decision latent of the action of each row; ``rows[i]`` its (canonical query, observation or
    None) pairs. Pending / unobservable / missing observations are masked by ``build_targets``."""
    dev = z.device
    queries = [[q for q, _ in row] for row in rows]
    if update_scaler:
        for row in rows:
            for q, o in row:
                if q.kind == OutcomeKind.REAL and o is not None and o.value is not None and o.status == "observed":
                    heads.scaler.update(q.metric, float(o.value))
    meta, tgt = build_targets(rows, dev)
    e_q, q_mask, kind_ids, horizon = model.encode_queries(queries)
    z1 = z.unsqueeze(1)
    phi = model.features(z1, e_q)
    rl, rs = heads.scaler.tensors(queries, dev)
    out = heads.world(z1, e_q, kind_ids, horizon, phi if heads.world.in_phi is not None else None,
                      exposure=meta.exposure, n_categories=meta.n_categories, real_loc=rl, real_scale=rs)
    return WorldBatch(queries, meta, tgt, e_q, q_mask, kind_ids, horizon, phi, out, rl, rs)


def head_offsets(wb: WorldBatch) -> Dict[str, Tensor]:
    """Detached head summaries the fast layer is combined with, each [B, Q] float64:
    ``log_rate`` (count, per unit exposure), ``loc`` (real, normalised units), ``time`` (expected seconds)."""
    out = wb.out
    with torch.no_grad():
        log_rate = torch.logsumexp(out.log_rate[:, :, 0], 0) - math.log(out.n_members)      # log of the mean rate
        loc = out.loc[:, :, 0].mean(0)
        from . import dists as D
        time = D.hazard_expected_time(out.hazard_logits[:, :, 0], out._edges()[:, :, 0]).mean(0)
    return {"log_rate": log_rate.to(F64), "loc": loc.to(F64), "time": time.to(F64).clamp_min(1e-6)}


def fast_inputs(kind: str, wb: WorldBatch, offsets: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """(y, exposure, censored) [B, Q] float64 as the fast layer sees outcomes of ``kind`` (module docstring)."""
    off = offsets or head_offsets(wb)
    y = wb.targets.y.to(F64)
    if kind == OutcomeKind.COUNT:
        return y, (wb.meta.exposure.to(F64) * torch.exp(off["log_rate"].clamp(-20.0, 20.0))).clamp_min(1e-9), None
    if kind == OutcomeKind.REAL:
        return (y - wb.real_loc.to(F64)) / wb.real_scale.to(F64) - off["loc"], None, None
    if kind == OutcomeKind.TIME_TO_EVENT:
        return (y / off["time"]).clamp_min(1e-9), None, wb.targets.censored.to(F64)
    if kind == OutcomeKind.BINARY:
        return y.clamp(0.0, 1.0), None, None
    raise ValueError(f"the fast layer does not model outcome kind {kind!r}")


def fast_prior_module(model: LatentModelAPI) -> Optional[nn.Module]:
    """Workstream A's learned fast prior (``model.exp.fast_prior``) if this model has one."""
    return getattr(getattr(model, "exp", None), "fast_prior", None)


KIND_OF_ID = {v: k for k, v in KIND_INDEX.items()}
__all__ = ["PredictConfig", "PredictHeads", "RealScaler", "WorldBatch", "attach_heads", "canonical_query",
           "fast_inputs", "fast_prior_module", "head_offsets", "world_batch", "FAST_KINDS", "HEAD_NAMES",
           "K_BINARY", "K_COUNT", "K_REAL", "K_TTE", "KIND_OF_ID", "VOI_SIGNALS"]
