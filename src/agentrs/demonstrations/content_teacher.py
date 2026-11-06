"""Scripted content demonstrator ("another agent" writing content) for ContentEnv.

Uses only the public goal and observation. Deliberately imperfect: it proposes a random listed
meeting time (it does not know the recipient's preference), sometimes adds hype to forum
answers (which raises engagement but not acceptance), and does not know the held-out document
domain. Outcomes are always decided by the environment's verifier."""

from __future__ import annotations

import random
import re
from typing import Optional

from agentrs.actions.schema import Action, ActionType
from agentrs.environments.base import Observation, PublicTask
from agentrs.environments.content.env import OPERATIONS


class ContentTeacher:
    source_id = "teacher:content-v1"

    def __init__(self, seed: int = 0, hype_prob: float = 0.4):
        self.rng = random.Random(seed)
        self.hype_prob = hype_prob
        self.tried: list[str] = []

    def reset(self) -> None:
        self.tried = []

    def act(self, task: PublicTask, obs: Observation, candidates=None) -> Action:
        if not any(c.type == ActionType.SUBMIT for c in (candidates or [])):
            return Action.finish()
        content = self.write(task, obs)
        return Action.submit(OPERATIONS[task.family], content) if content is not None else Action.finish()

    def write(self, task: PublicTask, obs: Observation) -> Optional[str]:
        g = task.goal
        texts = [e.name for e in obs.elements]
        if task.family == "content_config":
            pairs = re.findall(r"(\w+) is (\d+)", g)
            return "; ".join(f"{k}={v}" for k, v in pairs)
        if task.family == "content_email":
            line = next(t for t in texts if t.startswith("Available times:"))
            slots = [s.strip() for s in line.split(":", 1)[1].split(",")]
            left = [s for s in slots if s not in self.tried] or slots
            slot = self.rng.choice(left)
            self.tried.append(slot)
            who = re.findall(r"(?:Email|to|Ask) (\w+)", g)
            name = who[0] if who else "there"
            return f"Hi {name}, can we meet {slot}?"
        if task.family == "content_forum":
            place = re.findall(r"capital of (\w+)", g)[0]
            ans = next(t.split(": ")[-1] for t in texts if t.startswith(f"Fact: {place} capital"))
            body = f"The capital of {place} is {ans}."
            if self.rng.random() < self.hype_prob:
                body += " AMAZING question!!!"
            return body
        if task.family == "content_code":
            ys = dict((int(a), int(b)) for a, b in re.findall(r"f\((\d+)\)=(\d+)", g))
            b = ys[0]
            a = ys[1] - ys[0]
            return f"{a}*x+{b}"
        return None
