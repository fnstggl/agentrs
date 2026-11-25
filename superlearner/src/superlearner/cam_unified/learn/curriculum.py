"""Learning-progress curriculum over (task family, difficulty) arms (spec 13.5).

Each arm keeps two exponential moving averages of its normalised competence score (IndependentGoalResult.score in
[0, 1]): a fast one and a slow one. Absolute learning progress LP = |fast - slow| is large exactly where competence is
changing — challenging but learnable — and near zero both on mastered arms and on arms that give no usable signal.
Sampling mass is

    p = (1 - F * n_families - eps) * normalise(LP + c_nov / sqrt(1 + n))  +  F per family (uniform over its arms)
        + eps uniform over all arms,

so the curriculum follows learning progress but can never collapse onto one family (diversity floor F) and keeps
rehearsing mastered arms at the floor (retention). A split guard is called for every arm on construction, on
`add_arm`, on every `update` and on every sampled arm: held-out validation / test families raise immediately, so
held-out results can neither be sampled nor used to steer the curriculum.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence


class HeldOutFamilyError(ValueError):
    """A held-out (validation/test) family reached the training curriculum."""


class SplitGuard:
    """Callable guard: raises `HeldOutFamilyError` for held-out families (and, if ``allowed`` is given, for any family
    not in the declared training list)."""

    def __init__(self, held_out: Iterable[str], allowed: Optional[Iterable[str]] = None):
        self.held_out = frozenset(held_out)
        self.allowed = None if allowed is None else frozenset(allowed)

    def __call__(self, family: str) -> None:
        if family in self.held_out:
            raise HeldOutFamilyError(f"family {family!r} is held out from training")
        if self.allowed is not None and family not in self.allowed:
            raise HeldOutFamilyError(f"family {family!r} is not a declared training family")


@dataclass
class _Arm:
    family: str
    difficulty: float
    n: int = 0
    fast: float = 0.0
    slow: float = 0.0
    last: float = float("nan")

    @property
    def key(self) -> tuple:
        return (self.family, self.difficulty)

    @property
    def lp(self) -> float:
        return abs(self.fast - self.slow) if self.n >= 2 else 0.0


class LearningProgressCurriculum:
    """Args: arms [(family, difficulty)]; split_guard(family) raising for held-out families (required);
    family_floor F; eps uniform mass; fast/slow EMA rates; c_nov novelty bonus; seed."""

    def __init__(self, arms: Sequence[tuple], split_guard: Callable[[str], None], family_floor: float = 0.05,
                 eps: float = 0.02, fast: float = 0.25, slow: float = 0.03, c_nov: float = 0.05, seed: int = 0):
        if split_guard is None or not callable(split_guard):
            raise ValueError("a split guard is required (it must raise for held-out families)")
        if not 0 < slow < fast <= 1:
            raise ValueError("need 0 < slow < fast <= 1")
        self.guard = split_guard
        self.F, self.eps, self.a_fast, self.a_slow, self.c_nov = family_floor, eps, fast, slow, c_nov
        self.rng = random.Random(seed)
        self.arms: list[_Arm] = []
        self.by_key: dict = {}
        for fam, diff in arms:
            self.add_arm(fam, diff)
        if not self.arms:
            raise ValueError("no arms")

    def add_arm(self, family: str, difficulty=0) -> None:
        self.guard(family)
        key = (family, float(difficulty))
        if key in self.by_key:
            raise ValueError(f"duplicate arm {key}")
        a = _Arm(family, float(difficulty))
        self.arms.append(a)
        self.by_key[key] = a
        nf = len(self.families())
        if self.F * nf + self.eps >= 1.0:
            raise ValueError("diversity floor too large for the number of families")

    def families(self) -> list:
        return sorted({a.family for a in self.arms})

    def weights(self) -> list:
        fams = self.families()
        N = len(self.arms)
        base = [a.lp + self.c_nov / math.sqrt(1 + a.n) for a in self.arms]
        tot = sum(base)
        p_lp = [b / tot for b in base] if tot > 0 else [1.0 / N] * N
        free = 1.0 - self.F * len(fams) - self.eps
        per_fam = {f: sum(1 for a in self.arms if a.family == f) for f in fams}
        return [free * p_lp[i] + self.F / per_fam[a.family] + self.eps / N for i, a in enumerate(self.arms)]

    def sample(self, n: int = 1) -> list:
        w = self.weights()
        out = []
        for _ in range(n):
            a = self.rng.choices(self.arms, weights=w, k=1)[0]
            self.guard(a.family)                     # defensive: never emit a held-out family
            out.append(a.key)
        return out

    def update(self, family: str, difficulty, score: float) -> None:
        """Record a normalised competence score (clipped to [0, 1]) for a training arm."""
        self.guard(family)
        a = self.by_key.get((family, float(difficulty)))
        if a is None:
            raise KeyError(f"unknown arm {(family, difficulty)}")
        s = min(1.0, max(0.0, float(score)))
        if a.n == 0:
            a.fast = a.slow = s
        else:
            a.fast += self.a_fast * (s - a.fast)
            a.slow += self.a_slow * (s - a.slow)
        a.n += 1
        a.last = s

    def table(self) -> list:
        w = self.weights()
        return [{"family": a.family, "difficulty": a.difficulty, "n": a.n, "competence": round(a.fast, 4),
                 "lp": round(a.lp, 4), "p": round(w[i], 4)} for i, a in enumerate(self.arms)]

    def state_dict(self) -> dict:
        return {"arms": [(a.family, a.difficulty, a.n, a.fast, a.slow, a.last) for a in self.arms],
                "rng": self.rng.getstate()}

    def load_state_dict(self, s: dict) -> None:
        for fam, diff, n, f, sl, last in s["arms"]:
            self.guard(fam)
            a = self.by_key.get((fam, float(diff)))
            if a is None:
                self.add_arm(fam, diff)
                a = self.by_key[(fam, float(diff))]
            a.n, a.fast, a.slow, a.last = int(n), float(f), float(sl), float(last)
        st = s["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
