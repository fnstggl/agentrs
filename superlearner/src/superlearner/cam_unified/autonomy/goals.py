"""Self-generated, evidence-grounded learning goals (meta-reward spec 9.1-9.3, 12.2; manifest AU-01, AU-02, AU-06).

A `SelfGoal` is what CAM's curriculum policy proposes when no objective is assigned (goal_mode =
"autonomous_learning"): a semantic statement plus a list of *predicates over observable quantities* that make it
checkable, the entities involved, the tools it may use, a resource limit, a horizon, the evidence that would verify it,
the curriculum's predicted learning benefit and its uncertainty.

* `GoalCompiler.candidates(obs)` instantiates predicate templates over the numeric quantities the agent currently
  observes (paths of NUMERIC observations: balances, readings, counts, analytics): reach / exceed / reduce / maintain a
  quantity relative to its current value, in units of its observed variability. The candidate *set* is generated from
  the world's own observation, not from a fixed task menu; the curriculum policy (C_chi, `autonomy/curriculum.py`)
  scores candidates and their parameters with its own learned network.
* `GoalCompiler.compile_text(text)` parses a decoder-generated goal statement in the strict goal grammar
  (`"<op> <path> <number> within <days> days"`, ops: reach, exceed, reduce, maintain); anything else is
  UNVERIFIABLE — it may still motivate investigation but can never be counted as achieved.
* `SelfGoalVerifier` lives on the operator side. It registers a goal with the world's own observation at registration
  time, then checks predicates on later observations the *operator* reads from the world (never CAM's claims), and
  resolves SUCCESS / FAILURE at the deadline. A goal already satisfied at registration is TRIVIAL and resolves with
  zero credit (anti-trivial-goal guard, 12.2); duplicate goals within a lifetime are collapsed.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..interfaces import ObsKind, Observation, new_id

OPS = ("reach", "exceed", "reduce", "maintain")
PENDING, SUCCESS, FAILURE, TRIVIAL, UNVERIFIABLE = "pending", "success", "failure", "trivial", "unverifiable"


@dataclass
class GoalPredicate:
    path: str                 # observable quantity (NUMERIC observation source path)
    op: str                   # reach (>= thr), exceed (> thr, strictly), reduce (<= thr), maintain (>= thr at every check)
    threshold: float

    def holds(self, v: Optional[float]) -> bool:
        if v is None or not math.isfinite(v):
            return False
        if self.op == "reach" or self.op == "maintain":
            return v >= self.threshold
        if self.op == "exceed":
            return v > self.threshold
        if self.op == "reduce":
            return v <= self.threshold
        raise ValueError(self.op)

    def text(self) -> str:
        return f"{self.op} {self.path} {self.threshold:.6g}"


@dataclass
class SelfGoal:
    goal_id: str
    text: str
    predicates: list                                  # list[GoalPredicate]; all must hold
    horizon_days: float
    entities: list = field(default_factory=list)
    authorized_tools: Optional[list] = None           # None = every tool the agent may use anyway
    resource_limit: Optional[float] = None            # spend limit in the world's money unit
    evidence: str = ""
    expected_benefit: float = 0.0                     # curriculum's predicted learning gain (KV-02 units)
    uncertainty: float = 0.0
    created_t: float = 0.0
    verifiable: bool = True
    features: list = field(default_factory=list)      # numeric descriptor for the curriculum network

    def key(self) -> str:
        return "|".join(sorted(p.text() for p in self.predicates)) + f"@{self.horizon_days:.3g}"


def numeric_view(obs: Sequence[Observation]) -> dict:
    """Latest value per observable numeric path."""
    out = {}
    for o in obs:
        if o.kind == ObsKind.NUMERIC and o.value is not None and o.source:
            out[o.source] = float(o.value)
    return out


class GoalCompiler:
    """Builds candidate self-goals from what the agent observes; parses decoder-generated goal statements."""

    GRAMMAR = re.compile(r"^\s*(reach|exceed|reduce|maintain)\s+([A-Za-z0-9_.:\-]+)\s+(-?[0-9.eE+]+)\s+within\s+"
                         r"([0-9.]+)\s+days?\s*$")

    def __init__(self, scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0), horizons: Sequence[float] = (1.0, 3.0, 7.0)):
        self.scales = tuple(scales)
        self.horizons = tuple(horizons)
        self.history: dict = defaultdict(list)

    def observe(self, obs: Sequence[Observation]) -> None:
        for k, v in numeric_view(obs).items():
            h = self.history[k]
            h.append(v)
            if len(h) > 256:
                del h[:-256]

    def _spread(self, path: str, v: float) -> float:
        h = self.history.get(path, [])
        if len(h) >= 3:
            m = sum(h) / len(h)
            sd = math.sqrt(sum((x - m) ** 2 for x in h) / (len(h) - 1))
            if sd > 0:
                return sd
        return max(abs(v) * 0.1, 1.0)

    def candidates(self, obs: Sequence[Observation], t_days: float, max_n: int = 64) -> list:
        nv = numeric_view(obs)
        out = []
        for path, v in sorted(nv.items()):
            s = self._spread(path, v)
            for k in self.scales:
                for hz in self.horizons:
                    for op, thr in (("exceed", v + k * s), ("reduce", v - k * s)):
                        p = GoalPredicate(path, op, thr)
                        out.append(SelfGoal(new_id("g_"), f"{p.text()} within {hz:g} days", [p], hz,
                                            evidence=f"operator reads {path} from the world's observation",
                                            created_t=t_days,
                                            features=[math.log1p(abs(v)) * (1 if v >= 0 else -1), math.log1p(s), k,
                                                      math.log(hz), 1.0 if op == "exceed" else -1.0]))
        if len(out) > max_n:
            stride = len(out) / max_n
            out = [out[int(i * stride)] for i in range(max_n)]
        return out

    def compile_text(self, text: str, t_days: float) -> SelfGoal:
        m = self.GRAMMAR.match(text or "")
        if not m:
            return SelfGoal(new_id("g_"), text, [], 0.0, created_t=t_days, verifiable=False)
        op, path, thr, hz = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
        return SelfGoal(new_id("g_"), text, [GoalPredicate(path, op, thr)], hz, created_t=t_days)


@dataclass
class _Reg:
    goal: SelfGoal
    t0: float
    deadline: float
    status: str = PENDING
    checks: int = 0
    held: int = 0
    resolved_t: Optional[float] = None


class SelfGoalVerifier:
    """Operator-side checker of self-generated goals on the world's own observations."""

    def __init__(self):
        self.regs: dict = {}
        self.keys: dict = {}

    def register(self, goal: SelfGoal, world_obs: Sequence[Observation], t_days: float) -> str:
        if not goal.verifiable or not goal.predicates:
            self.regs[goal.goal_id] = _Reg(goal, t_days, t_days, status=UNVERIFIABLE, resolved_t=t_days)
            return UNVERIFIABLE
        k = goal.key()
        if k in self.keys and self.regs[self.keys[k]].status == PENDING:
            return "duplicate:" + self.keys[k]
        nv = numeric_view(world_obs)
        r = _Reg(goal, t_days, t_days + goal.horizon_days)
        if all(p.holds(nv.get(p.path)) for p in goal.predicates) and not any(p.op == "maintain" for p in goal.predicates):
            r.status, r.resolved_t = TRIVIAL, t_days
        self.regs[goal.goal_id] = r
        self.keys[k] = goal.goal_id
        return r.status

    def check(self, world_obs: Sequence[Observation], t_days: float) -> list:
        """Update with the operator's latest view of the world; returns goals resolved by this call."""
        nv = numeric_view(world_obs)
        done = []
        for r in self.regs.values():
            if r.status != PENDING:
                continue
            ok = all(p.holds(nv.get(p.path)) for p in r.goal.predicates)
            r.checks += 1
            r.held += int(ok)
            maintain = any(p.op == "maintain" for p in r.goal.predicates)
            if maintain:
                if not ok:
                    r.status, r.resolved_t = FAILURE, t_days
                elif t_days >= r.deadline:
                    r.status, r.resolved_t = SUCCESS, t_days
            elif ok:
                r.status, r.resolved_t = SUCCESS, t_days
            elif t_days >= r.deadline:
                r.status, r.resolved_t = FAILURE, t_days
            if r.status != PENDING:
                done.append((r.goal.goal_id, r.status, t_days - r.t0))
        return done

    def status(self, goal_id: str) -> str:
        return self.regs[goal_id].status

    def summary(self) -> dict:
        c = defaultdict(int)
        for r in self.regs.values():
            c[r.status] += 1
        return dict(c)
