"""The agent's non-parametric memory.  [candidate-editable]

Three kinds of memory are kept apart so that experiments can reset one and keep the others:

* working memory  - `WorkingMemory`: this episode's steps and outcomes, the current plan and draft program,
                    subgoals and notes. Cleared at every episode start.
* episodic memory - `EpisodicMemory`: one record per finished episode (goal embedding, outcome, which tools were
                    called successfully, literal argument values that worked). Retrieved by goal similarity. Together
                    with the agent's tool library it survives episodes and world resets, and is saved in checkpoints
                    separately from the weights.
* neural memory   - the controller's weights and its recurrent state (`cam/model.py`); not in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HistStep:
    kind: str                  # tool | wait | propose | draft_run | draft_skip | define | use
    tool: str
    args: dict
    ok: bool
    fclass: str = ""
    feedback: str = ""
    result: Any = None
    cost_s: int = 0
    n_calls: int = 1
    t: int = 0


@dataclass
class WorkingMemory:
    steps: list = field(default_factory=list)       # HistStep, one per decision
    calls: list = field(default_factory=list)       # successful primitive calls: (tool, args, result)
    plan: str = ""
    draft: list = field(default_factory=list)       # remaining steps of the current draft program
    subgoals: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tool_uses: dict = field(default_factory=dict)   # tool -> [n_ok, n_fail, last_ok]
    last_ok_args: dict = field(default_factory=dict)
    since_define: int = 0
    fresh_n: int = 0
    n_propose: int = 0
    n_gen: int = 0
    last_gen: dict = field(default_factory=dict)    # (tool, param) -> text of the last LLM-written value

    def note_use(self, tool: str, ok: bool) -> None:
        u = self.tool_uses.setdefault(tool, [0, 0, 0])
        u[0 if ok else 1] += 1
        u[2] = 1 if ok else -1

    def consecutive_failures(self) -> int:
        n = 0
        for s in reversed(self.steps):
            if s.ok:
                break
            n += 1
        return n


class EpisodicMemory:
    def __init__(self, capacity: int = 4000):
        self.capacity = capacity
        self.items: list[dict] = []
        self._mat: np.ndarray | None = None

    def add(self, goal_key: int, goal_vec: np.ndarray, success: bool, ret: float, tools_ok: list[str],
            literal_args: dict, n_decisions: int, tag: str = "") -> None:
        self.items.append({"goal_key": int(goal_key), "vec": np.asarray(goal_vec, np.float32), "success": bool(success),
                           "ret": float(ret), "tools": list(tools_ok)[:40], "args": literal_args,
                           "n": int(n_decisions), "tag": tag})
        if len(self.items) > self.capacity:
            # keep successes preferentially: drop the oldest failure, else the oldest item
            for i, it in enumerate(self.items):
                if not it["success"]:
                    del self.items[i]
                    break
            else:
                del self.items[0]
        self._mat = None

    def retrieve(self, goal_vec: np.ndarray, k: int = 3) -> list[tuple[float, dict]]:
        if not self.items:
            return []
        if self._mat is None:
            self._mat = np.stack([it["vec"] for it in self.items])
        s = self._mat @ np.asarray(goal_vec, np.float32)
        s = s + 0.05 * np.array([it["success"] for it in self.items], np.float32)
        # most similar first; among equally similar goals prefer recent ones
        idx = np.lexsort((-np.arange(len(s)), -s))[:k]
        return [(float(s[i]), self.items[i]) for i in idx]

    def __len__(self) -> int:
        return len(self.items)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "items": list(self.items)}

    def load_state_dict(self, s: dict) -> None:
        self.capacity = s.get("capacity", self.capacity)
        self.items = list(s.get("items", []))
        self._mat = None
