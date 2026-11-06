"""Competence-based curriculum: difficulty advances only on measured, verified success."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from pydantic import ConfigDict, BaseModel, Field

from agentrs.environments.miniweb.families import DIFFICULTY, TRAIN_FAMILIES


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: list[list[str]] = Field(default_factory=lambda: [
        [f for f in TRAIN_FAMILIES if DIFFICULTY[f] <= 2],
        [f for f in TRAIN_FAMILIES if DIFFICULTY[f] == 3],
        [f for f in TRAIN_FAMILIES if DIFFICULTY[f] >= 4],
    ])
    advance_threshold: float = 0.6  # success rate on the current frontier stage
    window: int = 40  # recent verified episodes per stage used for the estimate
    min_episodes: int = 20
    replay_fraction: float = 0.3  # fraction of tasks drawn from already-mastered stages


class Curriculum:
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.stage = 0
        self.recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.window))
        self.log: list[dict] = []

    def stage_of(self, family: str) -> int:
        for i, fams in enumerate(self.cfg.stages):
            if family in fams:
                return i
        return len(self.cfg.stages) - 1

    def active_families(self) -> list[str]:
        return [f for s in self.cfg.stages[: self.stage + 1] for f in s]

    def frontier_families(self) -> list[str]:
        return list(self.cfg.stages[self.stage])

    def record(self, family: str, success: bool) -> None:
        self.recent[self.stage_of(family)].append(float(success))

    def competence(self, stage: int) -> float:
        r = self.recent[stage]
        return sum(r) / len(r) if r else 0.0

    def maybe_advance(self) -> bool:
        if self.stage >= len(self.cfg.stages) - 1:
            return False
        r = self.recent[self.stage]
        if len(r) >= self.cfg.min_episodes and self.competence(self.stage) >= self.cfg.advance_threshold:
            self.log.append({"advanced_from": self.stage, "competence": self.competence(self.stage), "n": len(r)})
            self.stage += 1
            return True
        return False

    def sample_families(self, n: int, rng) -> list[str]:
        out = []
        mastered = [f for s in self.cfg.stages[: self.stage] for f in s]
        for _ in range(n):
            if mastered and rng.random() < self.cfg.replay_fraction:
                out.append(rng.choice(mastered))
            else:
                out.append(rng.choice(self.frontier_families()))
        return out

    def state(self) -> dict:
        return {"stage": self.stage, "competence": {i: self.competence(i) for i in range(len(self.cfg.stages))},
                "log": self.log}

    def load_state(self, st: dict) -> None:
        self.stage = int(st.get("stage", 0))
        self.log = list(st.get("log", []))


def update_from_episodes(cur: Curriculum, eps: Sequence) -> None:
    for e in eps:
        cur.record(e.family, bool(e.success))
