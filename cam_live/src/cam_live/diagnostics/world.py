"""cam_live.diagnostics.world — LabWorld: a small controlled world with known ground truth. MASTER-OWNED.

It exists to test the *mathematics and plumbing* of CAM Live (prediction, fast adaptation, delayed credit,
information value, reward integrity, meta-learning windows, the runtime loop). It is not a simulator of Discord or
of any real task, results on it are labelled ``provenance="diagnostic"``, and it is never a substitute for live
experience.

Mechanism (hidden from the agent):
* ``n_channels`` interventions. Channel c has an observable feature vector f_c and a hidden rate
  ``lambda_c = rate_scale * softplus(w . f_c)`` adopters per unit effort, where the weight vector ``w`` is drawn per
  world from a family prior — so features are learnable across worlds and ``w`` must be inferred within one.
* Adopters produced by an intervention arrive after exponential delays (channel-specific mean, hours to days).
* ``lab.probe`` returns a noisy measurement of a channel's rate (information with no direct effect).
* ``lab.unlock`` has no immediate effect but multiplies the last channel's later yield (delayed dependency).
* ``lab.noise`` returns a large unpredictable number and never affects anything (a "noisy TV").
* Optionally the mechanism changes at ``change_at_s``.

The environment returns observations and consequences only. There is no reward.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, Cost, EXEC_INTERFACE_VERSION,
                         FailureClass, GoalStatus, Level, Limits, Modality, Observation, OutcomeKind,
                         OutcomeObservation, OutcomeQuery, OutcomeStatus, PrimitiveOp, Provenance, Quantity,
                         ResultStatus, SimClock, TaskSpec, ToolDescriptor, Trust, VerifiedOutcome, new_id)

DAY = 86400.0


@dataclass
class LabConfig:
    n_channels: int = 6
    n_feat: int = 3
    horizon_s: float = 14 * DAY
    target: int = 30
    budget: float = 100.0
    act_cost: float = 2.0            # per unit effort
    probe_cost: float = 0.3
    act_duration_s: float = 3600.0
    probe_duration_s: float = 600.0
    mean_delay_s: Tuple[float, float] = (3600.0, 3 * DAY)
    outcome_horizon_s: float = 2 * DAY
    family: str = "lab/base"
    w_mean: Tuple[float, ...] = (1.0, -0.5, 0.3)
    w_std: float = 0.5
    rate_scale: float = 2.0
    probe_noise: float = 0.5
    key_door: bool = True
    door_boost: float = 4.0
    door_locked: float = 0.1
    noisy_tv: bool = True
    change_at_s: float = 0.0         # 0 = the mechanism never changes
    max_effort: int = 3
    seed: int = 0


def _softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 50:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


class LabWorld:
    """Implements ``contracts.EnvironmentAdapter``."""
    interface_version = EXEC_INTERFACE_VERSION

    def __init__(self, cfg: Optional[LabConfig] = None, clock: Optional[SimClock] = None):
        self.cfg = cfg or LabConfig()
        self.clock = clock or SimClock()
        self.rng = random.Random(self.cfg.seed)
        c = self.cfg
        self.t0 = self.clock.now()
        self._w = [random.Random(c.seed * 7919 + 1).gauss(m, c.w_std) for m in c.w_mean[:c.n_feat]]
        self._w += [0.0] * (c.n_feat - len(self._w))
        frng = random.Random(c.seed * 104729 + 2)
        self.features = [[round(frng.gauss(0, 1), 3) for _ in range(c.n_feat)] for _ in range(c.n_channels)]
        lo, hi = c.mean_delay_s
        self._delay = [math.exp(frng.uniform(math.log(lo), math.log(hi))) for _ in range(c.n_channels)]
        words = ["amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper"]
        self.names = [f"channel-{i} ({words[(i * 3 + c.seed) % len(words)]})" for i in range(c.n_channels)]
        self._arrivals: List[Tuple[float, str, int]] = []     # (arrival time, action id, channel)
        self._actions: Dict[str, Dict[str, Any]] = {}
        self._pending: Dict[str, OutcomeQuery] = {}
        self._unlocked_at: Optional[float] = None
        self._spent = 0.0
        self._n_exec = 0
        self._first_observe = True
        self._events_seen = 0

    # ---- ground truth (tests only; never reachable through the adapter API) -------------------------------
    def _rate(self, ch: int, at: Optional[float] = None) -> float:
        c = self.cfg
        w = self._w
        at = self.clock.now() if at is None else at
        if c.change_at_s and at - self.t0 >= c.change_at_s:
            w = [-x for x in w]
        lam = c.rate_scale * _softplus(sum(a * b for a, b in zip(w, self.features[ch])))
        if c.key_door and ch == c.n_channels - 1:
            lam *= c.door_boost if self._unlocked_at is not None and self._unlocked_at <= at else c.door_locked
        return lam

    def _truth(self) -> Dict[str, Any]:
        return {"w": list(self._w), "rates": [self._rate(i) for i in range(self.cfg.n_channels)],
                "delays": list(self._delay), "unlocked_at": self._unlocked_at}

    # ---- EnvironmentAdapter ---------------------------------------------------------------------------------
    def _obs(self, kind: str, text: str, q: Sequence[Quantity] = (), parent: str = "", data=None) -> Observation:
        now = self.clock.now()
        return Observation(obs_id=new_id("obs"), ts=now, kind=kind, text=text, quantities=list(q),
                           provenance=Provenance(source="labworld", retrieved_at=now, trust=Trust.UNTRUSTED),
                           parent_action_id=parent, data=data or {})

    def total(self, at: Optional[float] = None) -> int:
        at = self.clock.now() if at is None else at
        return sum(1 for t, _, _ in self._arrivals if t <= at)

    def observe(self) -> List[Observation]:
        out: List[Observation] = []
        if self._first_observe:
            self._first_observe = False
            for i, name in enumerate(self.names):
                out.append(self._obs("document", f"{name}: an available intervention channel.",
                                     [Quantity(f"channel.{i}.f{j}", v) for j, v in enumerate(self.features[i])]))
        out.append(self._obs("metric", "current state of the lab world", self.get_observable_metrics()))
        return out

    def discover_tools(self) -> List[ToolDescriptor]:
        c = self.cfg
        ch = {"type": "integer", "minimum": 0, "maximum": c.n_channels - 1}
        tools = [
            ToolDescriptor("lab.act", Modality.API, "Run an intervention on a channel with a chosen effort. Adopters, "
                           "if any, arrive later.", {"type": "object", "properties": {
                               "channel": ch, "effort": {"type": "integer", "minimum": 1, "maximum": c.max_effort}},
                               "required": ["channel", "effort"]}, Level.INTERVENTION, "lab.write", "external_write",
                           Cost(seconds=c.act_duration_s, money=c.act_cost)),
            ToolDescriptor("lab.probe", Modality.API, "Measure a channel. Returns a noisy reading; changes nothing.",
                           {"type": "object", "properties": {"channel": ch}, "required": ["channel"]},
                           Level.INTERVENTION, "lab.read", "external_read",
                           Cost(seconds=c.probe_duration_s, money=c.probe_cost)),
            ToolDescriptor("wait", Modality.INTERNAL, "Do nothing until later.", {"type": "object", "properties": {
                "seconds": {"type": "number", "minimum": 60, "maximum": 3 * DAY}}, "required": ["seconds"]},
                Level.INTERVENTION, "", "none", Cost()),
        ]
        if c.key_door:
            tools.append(ToolDescriptor("lab.unlock", Modality.API, "Prepare the last channel. No immediate effect.",
                                        {"type": "object", "properties": {}}, Level.INTERVENTION, "lab.write",
                                        "external_write", Cost(seconds=c.act_duration_s, money=c.act_cost)))
        if c.noisy_tv:
            tools.append(ToolDescriptor("lab.noise", Modality.API, "Read the static gauge.",
                                        {"type": "object", "properties": {}}, Level.INTERVENTION, "lab.read",
                                        "external_read", Cost(seconds=c.probe_duration_s)))
        return tools

    def validate_action(self, action: ActionSpec) -> List[str]:
        c, a, errs = self.cfg, action.arguments, []
        names = {t.name for t in self.discover_tools()}
        if action.operation not in names:
            return [f"unknown operation {action.operation!r}"]
        if action.operation in ("lab.act", "lab.probe"):
            ch = a.get("channel")
            if not isinstance(ch, int) or not 0 <= ch < c.n_channels:
                errs.append("channel out of range")
        if action.operation == "lab.act":
            e = a.get("effort")
            if not isinstance(e, int) or not 1 <= e <= c.max_effort:
                errs.append("effort out of range")
        if action.operation == "wait":
            s = a.get("seconds")
            if not isinstance(s, (int, float)) or not 60 <= s <= 3 * DAY:
                errs.append("seconds out of range")
        return errs

    def _cost_of(self, action: ActionSpec) -> float:
        c = self.cfg
        return {"lab.act": c.act_cost * int(action.arguments.get("effort", 1) or 1), "lab.probe": c.probe_cost,
                "lab.unlock": c.act_cost}.get(action.operation, 0.0)

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        c, now = self.cfg, self.clock.now()
        res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now)
        if authorization is None or authorization.verdict != AuthVerdict.ALLOW:
            needs = authorization is not None and authorization.verdict == AuthVerdict.NEEDS_APPROVAL
            res.status = ResultStatus.NEEDS_APPROVAL if needs else ResultStatus.DENIED
            res.failure_class = FailureClass.NONE if needs else FailureClass.MISSING_PERMISSION
            res.failure_detail = "; ".join(authorization.reasons) if authorization else "no authorization"
            return res
        errs = self.validate_action(action)
        if errs:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                                 "; ".join(errs))
            return res
        cost = self._cost_of(action)
        if self._spent + cost > c.budget + 1e-9:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT,
                                                                 "budget exhausted")
            return res
        self._spent += cost
        self._n_exec += 1
        op, a = action.operation, action.arguments
        dur = {"lab.act": c.act_duration_s, "lab.unlock": c.act_duration_s, "lab.probe": c.probe_duration_s,
               "lab.noise": c.probe_duration_s, "wait": float(a.get("seconds", 0.0))}[op]
        obs: List[Observation] = []
        if op == "lab.act":
            ch, effort = int(a["channel"]), int(a["effort"])
            n = _poisson(self.rng, self._rate(ch, now) * effort)
            for _ in range(n):
                self._arrivals.append((now + dur + self.rng.expovariate(1.0 / self._delay[ch]), action.action_id, ch))
            self._arrivals.sort()
            self._actions[action.action_id] = {"channel": ch, "effort": effort, "at": now, "n": n}
            q = OutcomeQuery(query_id=new_id("q"), metric=f"lab.adopters.from.{action.action_id}",
                             kind=OutcomeKind.COUNT, horizon_s=c.outcome_horizon_s, issued_at=now + dur,
                             description="adopters attributed to this intervention within the attribution window",
                             exposure=float(effort), goal_relevant=True)
            self._pending[q.query_id] = q
            res.expected_outcomes.append(q)
            obs.append(self._obs("tool_result", f"intervention started on {self.names[ch]} with effort {effort}",
                                 [Quantity("lab.effort", effort, kind=OutcomeKind.COUNT)], action.action_id))
        elif op == "lab.probe":
            ch = int(a["channel"])
            reading = self._rate(ch, now) + self.rng.gauss(0, c.probe_noise)
            obs.append(self._obs("tool_result", f"probe reading for {self.names[ch]}",
                                 [Quantity(f"lab.probe.{ch}", round(reading, 4))], action.action_id))
        elif op == "lab.unlock":
            if self._unlocked_at is None:
                self._unlocked_at = now + dur
            obs.append(self._obs("tool_result", "the last channel was prepared", [], action.action_id))
        elif op == "lab.noise":
            obs.append(self._obs("tool_result", "static gauge reading",
                                 [Quantity("lab.static", round(self.rng.gauss(0, 100.0), 3))], action.action_id))
        elif op == "wait":
            obs.append(self._obs("tool_result", f"waited {dur:.0f} seconds", [], action.action_id))
        self.clock.advance(dur)
        res.ended = self.clock.now()
        res.observations = obs
        res.cost = Cost(seconds=dur, money=cost, primitive_ops=1)
        res.ops = [PrimitiveOp(op=op, modality=action.modality, arguments=dict(a), started=now, ended=res.ended)]
        res.verified = True
        return res

    def poll_events(self, cursor: str) -> Tuple[List[Observation], str]:
        now, start = self.clock.now(), int(cursor or 0)
        ready = [x for x in self._arrivals if x[0] <= now]
        new = ready[start:]
        if not new:
            return [], str(len(ready))
        by: Dict[str, int] = {}
        for _, aid, _ in new:
            by[aid] = by.get(aid, 0) + 1
        ev = self._obs("event", f"{len(new)} new adopters arrived",
                       [Quantity("lab.adopters.new", len(new), kind=OutcomeKind.COUNT),
                        Quantity("lab.adopters.total", len(ready), kind=OutcomeKind.COUNT)], data={"by_action": by})
        return [ev], str(len(ready))

    def get_resource_state(self) -> Dict[str, Any]:
        el = self.clock.now() - self.t0
        return {"money_spent": self._spent, "money_remaining": self.cfg.budget - self._spent, "time_elapsed_s": el,
                "time_remaining_s": max(0.0, self.cfg.horizon_s - el), "external_actions": self._n_exec}

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        return list(self._pending.values())

    def get_observable_metrics(self) -> List[Quantity]:
        rs = self.get_resource_state()
        return [Quantity("lab.adopters.total", self.total(), kind=OutcomeKind.COUNT),
                Quantity("lab.budget.remaining", round(rs["money_remaining"], 4), unit="credits"),
                Quantity("lab.time.remaining", rs["time_remaining_s"], unit="s")]

    def get_capabilities(self) -> Dict[str, Any]:
        return {"name": "labworld", "interface_version": self.interface_version, "provenance": "diagnostic",
                "modalities": [Modality.API, Modality.INTERNAL], "simulated_time": True, "live": False}

    def measure(self, queries: Sequence[OutcomeQuery], *, _verified: bool = False) -> List[OutcomeObservation]:
        now, out = self.clock.now(), []
        for q in queries:
            if now < q.due_at:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.PENDING, source="labworld"))
                continue
            if q.metric.startswith("lab.adopters.from."):
                aid = q.metric[len("lab.adopters.from."):]
                v = float(sum(1 for t, a, _ in self._arrivals if a == aid and t <= q.due_at))
            elif q.metric == "lab.adopters.total":
                v = float(self.total(q.due_at))
            else:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, source="labworld"))
                continue
            self._pending.pop(q.query_id, None)
            out.append(OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, v, observed_at=now, verified=_verified,
                                          source="labworld", dedup_key=f"{q.metric}@{q.due_at:.0f}"))
        return out


class LabVerifier:
    """Independent verifier for LabWorld objectives (implements ``contracts.OutcomeVerifier``). Holds its own key."""

    def __init__(self, world: LabWorld, secret: bytes = b"lab-verifier-key"):
        self._world, self._secret = world, secret

    def _sign(self, o: VerifiedOutcome) -> str:
        d = o.to_dict()
        d.pop("signature", None)
        return hmac.new(self._secret, json.dumps(d, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        w, now = self._world, self._world.clock.now()
        target = float(task.success.get("target", w.cfg.target))
        deadline = w.t0 + float(task.limits.duration_s or w.cfg.horizon_s)
        reached = [t for t, _, _ in w._arrivals][int(target) - 1] if len(w._arrivals) >= target else None
        if reached is not None and reached <= min(now, deadline):
            status = GoalStatus.ACHIEVED
        elif now >= deadline:
            status = GoalStatus.FAILED
        else:
            status = GoalStatus.PENDING
        o = VerifiedOutcome(objective_id=task.objective_id, status=status, value=float(w.total(min(now, deadline))),
                            target=target, evaluated_at=now, notes="labworld ground truth",
                            secondary=[Quantity("lab.budget.spent", w._spent)])
        o.signature = self._sign(o)
        return o

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return bool(outcome.signature) and hmac.compare_digest(outcome.signature, self._sign(outcome))

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return self._world.measure(queries, _verified=True)


def lab_task(world: LabWorld, objective_id: str = "") -> TaskSpec:
    c = world.cfg
    return TaskSpec(objective_id=objective_id or new_id("obj"),
                    objective=f"Reach {c.target} adopters in the lab world before time runs out.",
                    success={"metric": "lab.adopters.total", "target": c.target}, family=c.family,
                    interfaces=[Modality.API, Modality.INTERNAL],
                    limits=Limits(duration_s=c.horizon_s, money=c.budget, max_decisions=10_000),
                    measurement={"source": "LabVerifier"}, mode="diagnostic")


class AllowAllPolicy:
    """Diagnostic-only ``PolicyEngine``: allows everything in LabWorld. Refuses to authorize a live task, so it can
    never stand in for the real authorization service."""

    def __init__(self):
        self._stopped, self.log = False, []

    def authorize(self, action, task, context=None):
        if self._stopped or task.mode == "live":
            return AuthorizationDecision(AuthVerdict.DENY, action.content_hash(),
                                         ["stopped" if self._stopped else "diagnostic policy cannot authorize live tasks"])
        return AuthorizationDecision(AuthVerdict.ALLOW, action.content_hash(), ["diagnostic allow-all"], grant="diag")

    def check_grant(self, action, decision):
        return decision.verdict == AuthVerdict.ALLOW and decision.action_hash == action.content_hash()

    def check_primitive(self, op, action):
        return AuthorizationDecision(AuthVerdict.DENY if self._stopped else AuthVerdict.ALLOW)

    def record(self, action, result):
        self.log.append((action.action_id, result.status))

    def approvals(self):
        return []

    def resolve_approval(self, approval_id, approve, by, note=""):
        return AuthorizationDecision(AuthVerdict.DENY, reasons=["no approvals in diagnostic policy"])

    def emergency_stop(self, reason, by):
        self._stopped = True

    def stopped(self):
        return self._stopped

    def revoke(self, scope, by):
        self._stopped = True

    def limits_state(self):
        return {}
