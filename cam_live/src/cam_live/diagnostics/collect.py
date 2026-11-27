"""cam_live.diagnostics.collect — run a behaviour policy in LabWorld and store honest experience. MASTER-OWNED.

Produces exactly the records the live agent produces (items forming H_t and H_{t+1}, the full candidate set with
logged probabilities, results, pending queries, matured outcomes, verified objective outcome), so prediction,
credit, reward and learner code can be exercised on real stored experience before the integrated agent exists.
Everything written here has ``provenance="diagnostic"``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import (ActionSpec, CandidateEvaluation, Decision, ExperienceItem, ExperienceRecord,
                         ExperienceStore, GoalStatus, ItemKind, Level, Modality, Observation, OutcomeObservation,
                         OutcomeQuery, OutcomeStatus, Quantity, ResultStatus, TaskSpec, ToolDescriptor, Trust,
                         VerifiedOutcome, new_id)
from .world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task

#: policy_fn(candidates, info) -> (chosen index, probability of every candidate)
PolicyFn = Callable[[List[ActionSpec], Dict[str, Any]], Tuple[int, List[float]]]


def lab_candidates(world: LabWorld, rng: random.Random, max_n: int = 12) -> List[ActionSpec]:
    """A varied, schema-valid candidate set for the current LabWorld state (order is shuffled)."""
    c = world.cfg
    out: List[ActionSpec] = []

    def mk(op: str, modality: str = Modality.API, **args: Any) -> ActionSpec:
        return ActionSpec(action_id=new_id("act"), operation=op, modality=modality, level=Level.INTERVENTION,
                          arguments=args, provenance="schema_walker")

    chans = list(range(c.n_channels))
    rng.shuffle(chans)
    for ch in chans:
        out.append(mk("lab.act", channel=ch, effort=rng.randint(1, c.max_effort)))
    for ch in chans[: max(1, c.n_channels // 2)]:
        out.append(mk("lab.probe", channel=ch))
    out.append(mk("wait", Modality.INTERNAL, seconds=float(rng.choice([3600, 6 * 3600, 86400]))))
    if c.key_door:
        out.append(mk("lab.unlock"))
    if c.noisy_tv:
        out.append(mk("lab.noise"))
    rng.shuffle(out)
    return out[:max_n]


def uniform_policy(seed: int = 0) -> PolicyFn:
    rng = random.Random(seed)

    def fn(cands: List[ActionSpec], info: Dict[str, Any]) -> Tuple[int, List[float]]:
        n = len(cands)
        return rng.randrange(n), [1.0 / n] * n
    return fn


def obs_item(o: Observation) -> ExperienceItem:
    kind = {"event": ItemKind.EVENT, "document": ItemKind.DOCUMENT, "error": ItemKind.FAILURE}.get(o.kind,
                                                                                                 ItemKind.OBSERVATION)
    return ExperienceItem(kind=kind, text=o.text, ts=o.ts, quantities=list(o.quantities), ref=o.obs_id,
                          trust=o.provenance.trust)


def action_item(a: ActionSpec, ts: float) -> ExperienceItem:
    return ExperienceItem(kind=ItemKind.ACTION, text=a.render(), ts=ts, ref=a.action_id, trust=Trust.SELF)


def outcome_item(q: OutcomeQuery, o: OutcomeObservation) -> ExperienceItem:
    return ExperienceItem(kind=ItemKind.OUTCOME, text=q.description or q.metric, ts=o.observed_at,
                          quantities=[Quantity(q.metric.rsplit(".act_", 1)[0], float(o.value or 0.0), kind=q.kind,
                                               exposure=q.exposure)],
                          ref=f"{q.query_id}:out", trust=Trust.VERIFIER if o.verified else Trust.UNTRUSTED)


def tool_item(t: ToolDescriptor, ts: float) -> ExperienceItem:
    return ExperienceItem(kind=ItemKind.TOOL, text=f"{t.name}: {t.description}", ts=ts, ref=f"tool:{t.name}",
                          quantities=[Quantity("tool.cost.money", t.est_cost.money),
                                      Quantity("tool.cost.seconds", t.est_cost.seconds)], trust=Trust.OPERATOR)


@dataclass
class Lifetime:
    task: TaskSpec
    records: List[ExperienceRecord]
    verified: VerifiedOutcome
    world: LabWorld
    item_refs: List[str] = field(default_factory=list)


class LabCollector:
    """Collects one LabWorld lifetime into ``store``. ``max_context`` bounds |H_t| (goal item always kept).

    ``seq_start=0`` (default): when the store keeps a lifetime head (``head_seq()``, e.g.
    ``credit.replay.SqliteExperienceStore``) every record gets the next position after it, so a store that assigns
    positions itself never sees a position twice (change request D-7a); otherwise positions count from 0."""

    def __init__(self, store: ExperienceStore, *, max_context: int = 40, verified_measurements: bool = True,
                 window_id: str = "", seq_start: int = 0):
        self.store, self.max_context = store, max_context
        self.verified_measurements, self.window_id, self.seq = verified_measurements, window_id, seq_start

    def _put(self, ctx: List[str], item: ExperienceItem) -> None:
        self.store.put_item(item)
        ctx.append(item.ref)

    def _window(self, ctx: List[str]) -> List[str]:
        return list(ctx) if len(ctx) <= self.max_context else [ctx[0]] + ctx[-(self.max_context - 1):]

    def run(self, world: LabWorld, policy_fn: PolicyFn, n_decisions: int, *, task: Optional[TaskSpec] = None,
            seed: int = 0, finish: bool = True, on_step: Optional[Callable[[ExperienceRecord], None]] = None
            ) -> Lifetime:
        rng = random.Random(seed)
        task = task or lab_task(world)
        verifier, policy = LabVerifier(world), AllowAllPolicy()
        ctx: List[str] = []
        now = world.clock.now()
        self._put(ctx, ExperienceItem(kind=ItemKind.GOAL, text=task.objective, ts=now, ref=f"goal:{task.objective_id}",
                                      trust=Trust.OPERATOR))
        for t in world.discover_tools():
            item = tool_item(t, now)
            item.ref = f"{item.ref}:{task.objective_id}"
            self._put(ctx, item)
        for o in world.observe():
            self._put(ctx, obs_item(o))
        cursor, records, open_recs = "", [], []

        def reconcile() -> None:
            nonlocal cursor
            events, cursor = world.poll_events(cursor)
            for e in events:
                self._put(ctx, obs_item(e))
            still = []
            for rec in open_recs:
                src = verifier if self.verified_measurements else world
                left = []
                for q, o in zip(rec.pending_queries, src.measure(rec.pending_queries)):
                    if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                        rec.outcomes.append(o)
                        self._put(ctx, outcome_item(q, o))
                    elif o.status == OutcomeStatus.PENDING:
                        left.append(q)
                rec.pending_queries = left
                self.store.update(rec)
                if left:
                    still.append(rec)
            open_recs[:] = still

        for _ in range(n_decisions):
            if world.get_resource_state()["time_remaining_s"] <= 0:
                break
            cands = lab_candidates(world, rng)
            cands = [c for c in cands if not world.validate_action(c)]
            idx, probs = policy_fn(cands, {"world": world, "task": task, "context_refs": self._window(ctx)})
            chosen = cands[idx]
            h_t = self._window(ctx)
            ts = world.clock.now()
            dec = Decision(decision_id=new_id("dec"), ts=ts, objective_id=task.objective_id, chosen=chosen,
                           candidates=[CandidateEvaluation(action=c) for c in cands], probabilities=list(probs),
                           chosen_index=idx, exploration="behaviour", context_obs_ids=list(h_t))
            res = world.execute(chosen, policy.authorize(chosen, task, {}))
            self._put(ctx, action_item(chosen, ts))
            for o in res.observations:
                self._put(ctx, obs_item(o))
            seq = self.seq
            if hasattr(self.store, "head_seq") and seq <= int(self.store.head_seq()):
                seq = int(self.store.head_seq()) + 1
            rec = ExperienceRecord(exp_id=new_id("exp"), objective_id=task.objective_id, decision=dec, result=res,
                                   context_item_refs=list(h_t), pending_queries=list(res.expected_outcomes),
                                   goal_text=task.objective, duration_s=res.ended - res.started,
                                   window_id=self.window_id, objective_family=task.family, provenance="diagnostic",
                                   seq=seq)
            self.seq = seq + 1
            self.store.append(rec)
            reconcile()
            rec.post_item_refs = self._window(ctx)
            self.store.update(rec)
            records.append(rec)
            if rec.pending_queries:
                open_recs.append(rec)
            if on_step is not None:
                on_step(rec)
            if res.status != ResultStatus.OK and res.failure_detail == "budget exhausted":
                break
        if finish:                                   # let every delayed consequence mature, then verify the objective
            left = world.get_resource_state()["time_remaining_s"]
            while left > 0:
                world.clock.advance(min(left, 86400.0))
                reconcile()
                left = world.get_resource_state()["time_remaining_s"]
            world.clock.advance(world.cfg.outcome_horizon_s + 1.0)
            reconcile()
        verified = verifier.verify(task)
        self.store.flush()
        return Lifetime(task=task, records=records, verified=verified, world=world, item_refs=list(ctx))


def collect_family(store: ExperienceStore, n_worlds: int, n_decisions: int, *, base: Optional[LabConfig] = None,
                   policy_fn: Optional[PolicyFn] = None, seed: int = 0, **collector_kw: Any) -> List[Lifetime]:
    """Several independent lifetimes in worlds drawn from one family (different hidden mechanism per world)."""
    import dataclasses
    base = base or LabConfig()
    out, col = [], LabCollector(store, **collector_kw)
    for i in range(n_worlds):
        world = LabWorld(dataclasses.replace(base, seed=seed * 1000 + i))
        out.append(col.run(world, policy_fn or uniform_policy(seed * 1000 + i), n_decisions, seed=seed * 1000 + i))
    return out


def verified_return(lt: Lifetime) -> float:
    """Scalar verified utility of a finished lifetime: 1 if achieved else the verified fraction of the target."""
    v = lt.verified
    return 1.0 if v.status == GoalStatus.ACHIEVED else min(1.0, v.value / max(v.target, 1.0)) * 0.5
