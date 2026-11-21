"""Adaptive curriculum over task arms (task family x level x start state).  [candidate-editable]

The curriculum chooses *situations*, never strategies. For every arm it tracks an exponential moving average of
verified success, a history of progress, how uncertain the controller's own critic was there (spread of the
value ensemble) and which failure classes dominated. An arm's sampling weight is

    floor + learning progress + c_unc * binomial uncertainty + c_v * critic spread + c_nov / sqrt(1 + visits)

It responds to competence, not to a schedule:
* prerequisites - an arm is eligible only when every arm listed as its prerequisite has reached `unlock` success
  (or it has no prerequisite); while nothing harder is unlocked, the unlocked arms share the budget;
* mastery       - arms above `mastered` keep a reduced weight, so earlier skills are rehearsed (retention) without
  consuming the budget;
* failure patterns - when an arm mostly fails with prerequisite / permission / reference errors, part of its
  weight is moved to its prerequisites; when it mostly fails with valid actions but no outcome (strategic), the
  weight stays on the arm itself.

Worlds are drawn uniformly from the split's world families, with a fresh seed every time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..world.tasks import SPLITS, TaskSpec, check_split


@dataclass
class Arm:
    family: str
    level: int = 0
    start: str = "scratch"
    prereq: tuple = ()
    n: int = 0
    succ: float = 0.0
    prog: list = field(default_factory=list)
    vstd: float = 0.0
    fails: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.family}/{self.level}/{self.start}"

    def lp(self, w: int = 20) -> float:
        p = self.prog
        if len(p) < 4:
            return 0.5
        h = min(w, len(p) // 2)
        return abs(float(np.mean(p[-h:])) - float(np.mean(p[-2 * h:-h])))

    def uncertainty(self) -> float:
        return math.sqrt(max(self.succ * (1 - self.succ), 0.0) / (self.n + 1)) + 0.5 / math.sqrt(self.n + 1)


DEFAULT_ARMS = [
    {"family": "found"}, {"family": "apicall"},
    {"family": "fund", "prereq": ["found/0/scratch"]},
    {"family": "code"},
    {"family": "list", "start": "deployed"},
    {"family": "deploy", "start": "coded", "prereq": ["fund/0/scratch"]},
    {"family": "service", "start": "listed"},
    {"family": "repair", "prereq": ["service/0/listed"]},
    {"family": "service", "start": "deployed", "prereq": ["list/0/deployed"]},
    {"family": "service", "start": "coded", "prereq": ["deploy/0/coded", "service/0/deployed"]},
    {"family": "deploy", "prereq": ["deploy/0/coded", "code/0/scratch"]},
    {"family": "service", "prereq": ["service/0/coded", "deploy/0/scratch"]},
    {"family": "revenue", "prereq": ["service/0/scratch"]},
]


class Curriculum:
    def __init__(self, arms: list[dict] | None = None, world_families: list[str] | None = None, split: str = "train",
                 seed: int = 0, ema: float = 0.1, unlock: float = 0.3, mastered: float = 0.9, floor: float = 0.05,
                 c_unc: float = 0.5, c_v: float = 0.5, c_nov: float = 0.3, adaptive: bool = True, gate: bool = True):
        self.arms = [Arm(a["family"], int(a.get("level", 0)), a.get("start", "scratch"), tuple(a.get("prereq", ())))
                     for a in (arms or DEFAULT_ARMS)]
        self.by_key = {a.key: a for a in self.arms}
        self.split = split
        self.world_families = list(world_families or SPLITS[split][2])
        self.rng = np.random.default_rng(seed)
        self.ema, self.unlock, self.mastered, self.floor = ema, unlock, mastered, floor
        self.c_unc, self.c_v, self.c_nov, self.adaptive = c_unc, c_v, c_nov, adaptive
        self.gate = gate              # False: prerequisites do not lock arms (use when a warm start already covers them)
        self.counter = 0

    def eligible(self, a: Arm) -> bool:
        if not self.gate:
            return True
        return all(self.by_key[p].succ >= self.unlock for p in a.prereq if p in self.by_key)

    def weights(self) -> np.ndarray:
        if not self.adaptive:
            return np.ones(len(self.arms)) / len(self.arms)
        w = np.zeros(len(self.arms))
        for i, a in enumerate(self.arms):
            if not self.eligible(a):
                continue
            x = self.floor + a.lp() + self.c_unc * a.uncertainty() + self.c_v * a.vstd + self.c_nov / math.sqrt(1 + a.n)
            if a.succ >= self.mastered:
                x *= 0.3
            w[i] = x
        for i, a in enumerate(self.arms):          # failure patterns: missing prerequisites send weight upstream
            tot = sum(a.fails.values())
            if w[i] > 0 and tot >= 5 and a.prereq:
                up = sum(a.fails.get(k, 0) for k in ("prerequisite", "permission", "unresolved_reference",
                                                     "nonexistent_resource")) / tot
                if up > 0.4:
                    move = 0.3 * up * w[i]
                    w[i] -= move
                    for p in a.prereq:
                        if p in self.by_key:
                            w[self.arms.index(self.by_key[p])] += move / len(a.prereq)
        if w.sum() <= 0:
            w[:] = 1.0
        return w / w.sum()

    def sample(self, n: int) -> list[TaskSpec]:
        w = self.weights()
        out = []
        for i in self.rng.choice(len(self.arms), size=n, p=w):
            a = self.arms[int(i)]
            self.counter += 1
            wf = self.world_families[int(self.rng.integers(0, len(self.world_families)))]
            spec = TaskSpec(a.family, wf, SPLITS[self.split][0] + self.counter, a.level, a.start, self.split)
            check_split(spec)
            out.append(spec)
        return out

    def update(self, summary: dict, vstd: float = 0.0) -> None:
        arm = summary.get("arm") or []
        a = self.by_key.get(f"{arm[0]}/{arm[1]}/{arm[2]}") if len(arm) == 3 else None
        if a is None:
            return
        a.n += 1
        e = self.ema if a.n > 10 else 1.0 / a.n
        a.succ += e * (float(summary["success"]) - a.succ)
        a.prog.append(float(summary.get("best_progress", 0.0)))
        a.prog = a.prog[-200:]
        a.vstd += e * (float(vstd) - a.vstd)
        for k, v in ((summary.get("agent") or {}).get("failures") or {}).items():
            a.fails[k] = a.fails.get(k, 0) * 0.98 + v
        if not summary["success"] and (summary.get("agent") or {}).get("valid_rate", 0) > 0.8:
            a.fails["strategic"] = a.fails.get("strategic", 0) * 0.98 + 1

    def table(self) -> list[dict]:
        w = self.weights()
        return [{"arm": a.key, "n": a.n, "success": round(a.succ, 3), "lp": round(a.lp(), 3), "weight": round(float(w[i]), 3),
                 "eligible": self.eligible(a), "vstd": round(a.vstd, 3),
                 "top_failure": max(a.fails, key=a.fails.get) if a.fails else ""} for i, a in enumerate(self.arms)]

    def state_dict(self) -> dict:
        return {"arms": [{"key": a.key, "n": a.n, "succ": a.succ, "prog": a.prog, "vstd": a.vstd, "fails": a.fails}
                         for a in self.arms], "counter": self.counter, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, s: dict) -> None:
        for d in s.get("arms", []):
            a = self.by_key.get(d["key"])
            if a is not None:
                a.n, a.succ, a.prog, a.vstd, a.fails = d["n"], d["succ"], list(d["prog"]), d["vstd"], dict(d["fails"])
        self.counter = s.get("counter", 0)
        if "rng" in s:
            self.rng.bit_generator.state = s["rng"]
