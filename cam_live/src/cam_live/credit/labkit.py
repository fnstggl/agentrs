"""cam_live.credit.labkit — what Workstream D needs to exercise its code end to end in the LabWorld diagnostic
environment on the real model (debug scale on CPU, the real configuration on GPU).

* :func:`collect_lifetime` runs a behaviour policy in ``diagnostics.world.LabWorld`` and writes honest experience
  through D's own path: records into the durable store, measurements and events through
  ``links.OutcomeLinker.reconcile`` (so links, de-duplication and the re-training queue are exercised), and the
  verifier's objective outcome when the lifetime closes. It reuses the item builders of the master's
  ``diagnostics.collect`` and adds what that collector does not do: the linker, randomised experiment arms on a
  decision, and store-assigned sequence numbers (change request D-7).
* :class:`WorldPredictionProvider` is a *local wrapper* of Workstream B's stage-1 ``predict.world`` heads as a
  ``LossProvider`` (B's own providers are what the integrated agent registers; this one exists so that D's learner
  can be shown to improve held-out prediction NLL without depending on B's stage-2 files).
* :func:`build_lab` assembles model + heads + store + providers + learner for tests and ``credit.diagnose``.

Everything written here has ``provenance="diagnostic"``. LabWorld returns observations and consequences only.
"""
from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..config import CamLiveConfig
from ..contracts import (ActionSpec, CandidateEvaluation, Decision, ExperienceItem, ExperienceRecord, ExperimentRecord,
                         ItemKind, Level, Modality, OutcomeQuery, OutcomeStatus, SimClock, TaskSpec, Trust,
                         VerifiedOutcome, new_id)
from ..diagnostics.collect import action_item, lab_candidates, obs_item, outcome_item, tool_item
from ..diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
from ..learning_api import TIMESCALE_FREQUENT, DecisionBatch, matured_outcomes
from ..nn.api import LatentModelAPI
from ..predict.policy import CandidatePolicy, PolicyConfig
from ..predict.world import KindNormaliser, WorldConfig, WorldHeads, bootstrap_weights, build_targets, world_nll
from .learner import ContinualLearner, LearnerSettings
from .links import OutcomeLinker
from .providers import (BridgeGroundingProvider, CreditAssigner, HindsightProvider, ReturnDecompositionProvider,
                        TrajectoryEncoder)
from .replay import SqliteExperienceStore
from .retention import FunctionSpec, latent_functions
from .select import EstimatorSelector

PolicyFn = Callable[[List[ActionSpec], Dict[str, Any]], Tuple[int, List[float]]]
CandidatesFn = Callable[[LabWorld, random.Random], List[ActionSpec]]


def mk_action(op: str, modality: str = Modality.API, **args: Any) -> ActionSpec:
    return ActionSpec(action_id=new_id("act"), operation=op, modality=modality, level=Level.INTERVENTION,
                      arguments=args, provenance="schema_walker")


@dataclass
class Arms:
    """A randomised assignment on one decision of a lifetime: ``arms`` maps arm name -> action factory."""
    experiment_id: str
    arms: Dict[str, Callable[[], ActionSpec]]
    probs: Dict[str, float]
    step: int = 0


@dataclass
class LabLifetime:
    task: TaskSpec
    exp_ids: List[str]
    verified: Optional[VerifiedOutcome]
    world: LabWorld
    arm: str = ""
    reports: List[Any] = field(default_factory=list)          # ReconcileReports, in order


class LabRun:
    """One LabWorld lifetime written through the store and the linker. Use :meth:`decide` step by step (tests that
    look at the state before outcomes mature) or :func:`collect_lifetime` for the whole thing."""

    def __init__(self, store: SqliteExperienceStore, linker: OutcomeLinker, world: LabWorld, *,
                 task: Optional[TaskSpec] = None, seed: int = 0, max_context: int = 40, window_id: str = "",
                 on_outcome: Optional[Callable[[ExperienceRecord, OutcomeQuery, Any], None]] = None):
        self.store, self.linker, self.world = store, linker, world
        self.task = task or lab_task(world)
        self.rng = random.Random(seed)
        self.verifier, self.policy = LabVerifier(world), AllowAllPolicy()
        self.max_context, self.window_id, self.on_outcome = max_context, window_id, on_outcome
        self.ctx: List[str] = []
        self.cursor = ""
        self.exp_ids: List[str] = []
        self.reports: List[Any] = []
        self.arm = ""
        now = world.clock.now()
        self._put(ExperienceItem(kind=ItemKind.GOAL, text=self.task.objective, ts=now,
                                 ref=f"goal:{self.task.objective_id}", trust=Trust.OPERATOR))
        for t in world.discover_tools():
            item = tool_item(t, now)
            item.ref = f"{item.ref}:{self.task.objective_id}"
            self._put(item)
        for o in world.observe():
            self._put(obs_item(o))

    def _put(self, item: ExperienceItem) -> None:
        self.store.put_item(item)
        self.ctx.append(item.ref)

    def _window(self) -> List[str]:
        c = self.ctx
        return list(c) if len(c) <= self.max_context else [c[0]] + c[-(self.max_context - 1):]

    def reconcile(self) -> Any:
        """Poll events and due measurements and pass them through the linker. Returns the ``ReconcileReport``."""
        events, self.cursor = self.world.poll_events(self.cursor)
        for e in events:
            self._put(obs_item(e))
        due = self.store.pending(self.world.clock.now())
        mine = [(r, q) for r, q in due if r.objective_id == self.task.objective_id]
        obs = self.verifier.measure([q for _, q in mine]) if mine else []
        final = [(r, q, o) for (r, q), o in zip(mine, obs) if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED)]
        rep = self.linker.reconcile(outcomes=[o for _, _, o in final], events=events)
        for r, q, o in final:
            self._put(outcome_item(q, o))
            if self.on_outcome is not None:
                self.on_outcome(r, q, o)
        self.reports.append(rep)
        return rep

    def decide(self, policy_fn: PolicyFn, *, candidates_fn: Optional[CandidatesFn] = None, arms: Optional[Arms] = None
               ) -> ExperienceRecord:
        world, task = self.world, self.task
        diagnostics: Dict[str, Any] = {}
        experiment_id, exploration = "", "behaviour"
        if arms is not None:
            names = list(arms.arms)
            cands = [arms.arms[n]() for n in names]
            probs = [float(arms.probs[n]) for n in names]
            idx = self.rng.choices(range(len(names)), weights=probs)[0]
            self.arm = names[idx]
            diagnostics = {"arm": names[idx], "arms": {n: i for i, n in enumerate(names)}}
            experiment_id, exploration = arms.experiment_id, "randomized"
        else:
            cands = (candidates_fn or lab_candidates)(world, self.rng)
            cands = [c for c in cands if not world.validate_action(c)]
            idx, probs = policy_fn(cands, {"world": world, "task": task, "context_refs": self._window()})
        chosen, h_t, ts = cands[idx], self._window(), world.clock.now()
        k = len(self.exp_ids)
        dec = Decision(decision_id=f"dec_{task.objective_id}_{k}", ts=ts, objective_id=task.objective_id, chosen=chosen,
                       candidates=[CandidateEvaluation(action=c) for c in cands], probabilities=list(probs),
                       chosen_index=idx, exploration=exploration, context_obs_ids=list(h_t),
                       experiment_id=experiment_id, diagnostics=diagnostics)
        res = world.execute(chosen, self.policy.authorize(chosen, task, {}))
        self._put(action_item(chosen, ts))
        for o in res.observations:
            self._put(obs_item(o))
        rec = ExperienceRecord(exp_id=f"exp_{task.objective_id}_{k}", objective_id=task.objective_id, decision=dec, result=res,
                               context_item_refs=list(h_t), post_item_refs=self._window(),
                               pending_queries=list(res.expected_outcomes), goal_text=task.objective,
                               duration_s=res.ended - res.started, window_id=self.window_id,
                               objective_family=task.family, provenance="diagnostic")
        self.store.append(rec)                       # seq is assigned by the store
        self.exp_ids.append(rec.exp_id)
        self.reconcile()
        return rec

    def mature(self) -> None:
        """Let every delayed consequence of this lifetime arrive and be measured (simulated time only)."""
        world = self.world
        for _ in range(64):
            nxt = [q.due_at for r, q in self.store.pending(1e18) if r.objective_id == self.task.objective_id]
            if not nxt:
                break
            world.clock.advance(max(0.0, min(nxt) - world.clock.now()) + 1.0)
            self.reconcile()

    def close(self) -> VerifiedOutcome:
        """The independent verifier's statement about the objective; a checked signature closes the objective."""
        v = self.verifier.verify(self.task)
        self.store.put_objective_outcome(self.task.objective_id, status=v.status, value=v.value,
                                         evaluated_at=v.evaluated_at, verified=self.verifier.check_signature(v),
                                         body={"target": v.target, "notes": v.notes})
        return v

    def result(self, verified: Optional[VerifiedOutcome] = None) -> LabLifetime:
        return LabLifetime(self.task, list(self.exp_ids), verified, self.world, self.arm, list(self.reports))


def collect_lifetime(store: SqliteExperienceStore, linker: OutcomeLinker, world: LabWorld, policy_fn: PolicyFn,
                     n_decisions: int, *, seed: int = 0, candidates_fn: Optional[CandidatesFn] = None,
                     arms: Optional[Arms] = None, mature: bool = True, close: bool = True, tag: str = "",
                     **run_kw: Any) -> LabLifetime:
    """One whole lifetime. Record, decision and objective ids are deterministic functions of the family and
    ``tag`` (default: the seed), so a seeded collection is reproducible (the held-out split hashes these ids)."""
    run_kw.setdefault("task", lab_task(world, objective_id=f"obj_{world.cfg.family.replace('/', '-')}_{tag or seed}"))
    run = LabRun(store, linker, world, seed=seed, **run_kw)
    for t in range(n_decisions):
        if world.get_resource_state()["time_remaining_s"] <= 0:
            break
        run.decide(policy_fn, candidates_fn=candidates_fn, arms=arms if (arms is not None and arms.step == t) else None)
    if mature:
        run.mature()
    return run.result(run.close() if close else None)


def uniform_policy(seed: int = 0) -> PolicyFn:
    rng = random.Random(seed)

    def fn(cands: List[ActionSpec], info: Dict[str, Any]) -> Tuple[int, List[float]]:
        n = len(cands)
        return rng.randrange(n), [1.0 / n] * n
    return fn


def weighted_policy(weight: Callable[[ActionSpec, LabWorld], float], seed: int = 0) -> PolicyFn:
    """A fixed stochastic behaviour policy with logged probabilities proportional to ``weight(action, world)``."""
    rng = random.Random(seed)

    def fn(cands: List[ActionSpec], info: Dict[str, Any]) -> Tuple[int, List[float]]:
        w = [max(float(weight(c, info["world"])), 1e-9) for c in cands]
        tot = sum(w)
        probs = [x / tot for x in w]
        return rng.choices(range(len(cands)), weights=probs)[0], probs
    return fn


def collect_family(store: SqliteExperienceStore, linker: OutcomeLinker, base: LabConfig, n_worlds: int, n_decisions: int,
                   *, seed: int = 0, clock: Optional[SimClock] = None, policy_fn: Optional[PolicyFn] = None,
                   **kw: Any) -> List[LabLifetime]:
    """Independent lifetimes in worlds of one family (a different hidden mechanism per world), on one clock."""
    clock = clock or SimClock()
    out = []
    for i in range(n_worlds):
        world = LabWorld(dataclasses.replace(base, seed=seed * 1000 + i), clock)
        out.append(collect_lifetime(store, linker, world, policy_fn or uniform_policy(seed * 1000 + i), n_decisions,
                                    seed=seed * 1000 + i, **kw))
    return out


# ------------------------------------------------------------------------------------------------ B's heads, wrapped
def _canonical(q: OutcomeQuery) -> OutcomeQuery:
    """LabWorld puts the action id into the metric name; the metric *kind* is what a query embedding should see."""
    return dataclasses.replace(q, metric=q.metric.rsplit(".act_", 1)[0])


def _queries(rec: ExperienceRecord) -> List[OutcomeQuery]:
    qs = {q.query_id: q for q in rec.pending_queries}
    if rec.result is not None:
        for q in rec.result.expected_outcomes:
            qs.setdefault(q.query_id, q)
    return list(qs.values())


class WorldPredictionProvider:
    """Local ``LossProvider`` over ``predict.world.WorldHeads`` (Workstream B, stage 1): masked NLL of the matured
    outcomes of the executed candidate. Also exposes predictions for probes and forgetting measurements."""
    timescale = TIMESCALE_FREQUENT

    def __init__(self, model: LatentModelAPI, *, weight: float = 1.0, hidden: int = 64, n_members: int = 3,
                 use_phi: bool = True, head_name: str = "world", name: str = "world", seed: int = 0):
        self.name, self.weight, self.use_phi = name, float(weight), use_phi
        cfg = WorldConfig(d_model=model.d_model, n_features=model.n_features if use_phi else 0, hidden=hidden,
                          n_members=n_members, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8_000_009 + seed)
            self.heads: WorldHeads = model.register_head(head_name, WorldHeads(cfg))
        self.norm: KindNormaliser = model.register_head(head_name + "_norm", KindNormaliser())
        self.n_members = n_members

    def eligible(self, record: ExperienceRecord) -> bool:
        return bool(matured_outcomes(record))

    def _forward(self, model: LatentModelAPI, batch: DecisionBatch, rows: Sequence[Sequence[Tuple[OutcomeQuery, Any]]]):
        dev = batch.b.device
        meta, tgt = build_targets(rows, device=dev)
        e_q, q_mask, kind_ids, horizon = model.encode_queries([[_canonical(q) for q, _ in row] for row in rows])
        zc = batch.z_chosen[:, None]
        phi = model.features(zc, e_q) if self.use_phi else None
        out = self.heads(zc, e_q, kind_ids, horizon, phi, exposure=meta.exposure, n_categories=meta.n_categories)
        return out, meta, tgt

    def loss(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Optional[Tensor], Dict[str, float]]:
        rows = [matured_outcomes(r) for r in batch.records]
        if not any(rows):
            return None, {"n_targets": 0.0}
        out, meta, tgt = self._forward(model, batch, rows)
        w = bootstrap_weights([r.exp_id for r in batch.records], self.n_members) if model.training else None
        return world_nll(out, tgt, q_mask=meta.q_mask, member_weights=w, normaliser=self.norm if model.training else None)

    @torch.no_grad()
    def predict(self, model: LatentModelAPI, batch: DecisionBatch) -> Dict[str, Tensor]:
        """Predictions for every declared query of each record: mean / std [B, Q], mask [B, Q], and — where the
        outcome has matured — per-record NLL and standardised residual [B] (NaN elsewhere)."""
        matured = [dict((q.query_id, o) for q, o in matured_outcomes(r)) for r in batch.records]
        rows = [[(q, matured[i].get(q.query_id)) for q in _queries(r)] for i, r in enumerate(batch.records)]
        out, meta, tgt = self._forward(model, batch, rows)
        pred = out.predictive()
        mean, std = pred["mean"][:, 0], pred["total_std"][:, 0]
        mask = tgt.mask(meta.kind_ids) & meta.q_mask
        lp = torch.logsumexp(out.log_prob(torch.where(mask, tgt.y, torch.zeros_like(tgt.y)), event=tgt.observed)[:, :, 0],
                             0) - torch.log(torch.tensor(float(out.n_members)))
        n = mask.sum(1).clamp_min(1)
        nll = torch.where(mask.any(1), -(lp * mask).sum(1) / n, torch.full_like(n, float("nan"), dtype=lp.dtype))
        z = torch.where(mask, (tgt.y - mean) / std.clamp_min(1e-6), torch.zeros_like(mean)).sum(1) / n
        z = torch.where(mask.any(1), z, torch.full_like(z, float("nan")))
        return {"mean": mean, "std": std, "mask": meta.q_mask, "nll": nll, "z": z}

    def probe(self, model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Dict[str, Tensor]:
        p = self.predict(model, batch)
        return {"pred_mean": torch.where(p["mask"], torch.log1p(p["mean"].clamp_min(0)), torch.zeros_like(p["mean"]))}


def register_policy(model: LatentModelAPI, head_name: str = "policy", seed: int = 0) -> CandidatePolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8_000_017 + seed)
        return model.register_head(head_name, CandidatePolicy(PolicyConfig(d_model=model.d_model, hidden=64)))


def policy_probe(policy: CandidatePolicy) -> Callable[[LatentModelAPI, DecisionBatch, Any], Dict[str, Tensor]]:
    def fn(model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Dict[str, Tensor]:
        return {"action_probs": policy(batch.z, batch.a_mask, batch.b).probs}
    return fn


def lab_eval_fn(world: WorldPredictionProvider, policy: Optional[CandidatePolicy] = None):
    """``retention.EvalFn``: prediction NLL, standardised residual and action probabilities per record."""
    def fn(model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Dict[str, Tensor]:
        p = world.predict(model, batch)
        out = {"nll": p["nll"], "z": p["z"]}
        if policy is not None:
            out["action_probs"] = policy(batch.z, batch.a_mask, batch.b).probs
        return out
    return fn


def lab_functions(world: WorldPredictionProvider, policy: Optional[CandidatePolicy] = None) -> List[FunctionSpec]:
    """Functions whose anchor outputs the retention regulariser preserves: latents, predicted (log) outcome means
    and action probabilities."""
    def pred(model: LatentModelAPI, batch: DecisionBatch, store: Any) -> Tuple[Tensor, Tensor]:
        rows = [[(q, None) for q in _queries(r)] for r in batch.records]
        out, meta, _ = world._forward(model, batch, rows)
        return torch.log1p(out.member_means().mean(0)[:, 0].clamp_min(0)), meta.q_mask

    fns = latent_functions() + [FunctionSpec("world_log_mean", pred)]
    if policy is not None:
        fns.append(FunctionSpec("action_probs", lambda m, b, s: (policy(b.z, b.a_mask, b.b).probs, b.a_mask), kind="kl"))
    return fns


# ------------------------------------------------------------------------------------------------ assembly
@dataclass
class Lab:
    cfg: CamLiveConfig
    model: Any
    store: SqliteExperienceStore
    linker: OutcomeLinker
    learner: ContinualLearner
    clock: SimClock
    world: Optional[WorldPredictionProvider] = None
    policy: Optional[CandidatePolicy] = None
    rudder: Optional[ReturnDecompositionProvider] = None
    cocoa: Optional[HindsightProvider] = None
    bridge: Optional[BridgeGroundingProvider] = None
    assigner: Optional[CreditAssigner] = None

    def close(self) -> None:
        self.store.close()


def register_lab_heads(model: Any, store: Optional[SqliteExperienceStore] = None, *, world: bool = True,
                       policy: bool = True, credit: bool = True, beta: float = 0.0, seed: int = 0) -> Dict[str, Any]:
    """Register the heads used in D's evidence on ``model`` (the same call rebuilds them in a fresh process before a
    weights-only load). Returns the provider objects by name."""
    out: Dict[str, Any] = {}
    if world:
        out["world"] = WorldPredictionProvider(model, seed=seed)
    if policy:
        out["policy"] = register_policy(model, seed=seed)
    if credit:
        enc = TrajectoryEncoder(model, store)
        small = model.d_model < 128
        out["rudder"] = ReturnDecompositionProvider(model, store, beta=beta, seed=seed, encoder=enc,
                                                    d_model=64 if small else 128)
        out["cocoa"] = HindsightProvider(model, store, beta=beta, seed=seed, encoder=enc,
                                         d_hidden=64 if small else 128, d_key=32 if small else 64)
    return out


def build_lab(store_path: str, cfg: Optional[CamLiveConfig] = None, *, world: bool = True, policy: bool = True,
              credit: bool = True, bridge: bool = False, beta: float = 0.0, settings: Optional[LearnerSettings] = None,
              clock: Optional[SimClock] = None, log_path: Optional[str] = None, durable: bool = False,
              selector: Optional[EstimatorSelector] = None, model: Any = None, **store_kw: Any) -> Lab:
    """Model (``CamLiveConfig.debug()`` unless ``cfg`` is given) + heads + durable store + linker + providers +
    learner, wired the way the integrated agent wires them."""
    from ..nn.model import build_model
    cfg = cfg or CamLiveConfig.debug()
    clock = clock or SimClock()
    model = model if model is not None else build_model(cfg)
    store = SqliteExperienceStore(store_path, durable=durable, seed=cfg.seed, **store_kw)
    linker = OutcomeLinker(store, clock=clock, retro_predecessors=64)
    parts = register_lab_heads(model, store, world=world, policy=policy, credit=credit, beta=beta, seed=cfg.seed)
    assigner = None
    if credit:
        assigner = CreditAssigner(model, store, rudder=parts["rudder"], cocoa=parts["cocoa"], beta=beta, clock=clock,
                                  selector=selector or EstimatorSelector(clock=clock))
    learner = ContinualLearner(model, store, cfg, clock=clock, settings=settings, log_path=log_path, credit=assigner)
    for k in ("world", "rudder", "cocoa"):
        if k in parts:
            learner.register(parts[k])
    bp = None
    if bridge:
        bp = learner.register(BridgeGroundingProvider(model))
    if "world" in parts:
        learner.add_probe("world", parts["world"].probe)
    if "policy" in parts:
        learner.add_probe("policy", policy_probe(parts["policy"]))
    return Lab(cfg, model, store, linker, learner, clock, parts.get("world"), parts.get("policy"), parts.get("rudder"),
               parts.get("cocoa"), bp, assigner)


def register_experiment(store: SqliteExperienceStore, experiment_id: str, arms: Sequence[str], now: float) -> None:
    store.put_experiment(ExperimentRecord(
        experiment_id=experiment_id, created_at=now, question="does the first decision change the later return?",
        hypothesis="the arms differ in matured return", arms=list(arms), assignment="randomized",
        comparison=list(arms)[-1], status="running"))
