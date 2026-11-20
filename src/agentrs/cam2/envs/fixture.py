"""TEST FIXTURE ONLY — a tiny in-process web-form-like environment for fast unit tests of the CAM 2.0 plumbing
(gradient flow, PPO replay exactness, retention mechanics). It is never used for reported experiments; reported
results come from BrowserGym/MiniWoB++ and AppWorld.

Families: ``press:<k>`` (click the button whose label the goal names; k buttons), ``type`` (type the quoted word into
the text field, then click Submit). Labels are random words; the instance is fixed by the seed.
"""
from __future__ import annotations

import random

from agentrs.cam2.envs.base import EnvironmentAdapter, EvalResult, StepResult, TaskRef
from agentrs.cam2.envs.miniwob import miniwob_tools
from agentrs.cam2.tools import Element, Observation, ToolCall, validate_call

WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "grape", "heron", "ivory", "jolly", "koala", "lemon",
         "mango", "nylon", "olive", "pearl", "quail", "raven", "salsa", "tango"]


class LookupFixtureAdapter(EnvironmentAdapter):
    name = "fixture"

    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps
        self.tools = miniwob_tools()
        self.tmap = {t.name: t for t in self.tools}

    def reset(self, task: TaskRef) -> Observation:
        rng = random.Random(f"{task.family}:{task.seed}")
        self.family, self.k, self.done, self.success, self.typed = task.family, 0, False, False, None
        if task.family.startswith("press"):
            n = int(task.family.split(":")[1]) if ":" in task.family else 3
            self.labels = rng.sample(WORDS, n)
            self.target = rng.choice(self.labels)
            self.goal = f'Click the "{self.target}" button.'
            self.els = [Element(str(10 + i), "button", f"button '{w}'") for i, w in enumerate(self.labels)]
        else:
            self.target = rng.choice(WORDS)
            self.goal = f'Enter "{self.target}" into the text field and press Submit.'
            self.els = [Element("20", "textbox", "textbox ''"), Element("21", "button", "button 'Submit'")]
        self.text = ""
        return self.observe()

    def observe(self) -> Observation:
        return Observation(self.goal, list(self.els), self.text, self.k)

    def available_tools(self) -> list:
        return list(self.tools)

    def step(self, a: ToolCall) -> StepResult:
        err = validate_call(a, self.tmap, self.observe())
        self.k += 1
        r, term = 0.0, False
        self.text = ("invalid action: " + err) if err else ""
        if not err and a.tool == "click":
            el = next(e for e in self.els if e.eid == a.element)
            if self.family.startswith("press"):
                term, r = True, float(el.text == f"button '{self.target}'")
            elif el.eid == "21":
                term, r = True, float((self.typed or "").strip() == self.target)
        if not err and a.tool == "fill" and not self.family.startswith("press"):
            self.typed = a.text
            self.els[0] = Element("20", "textbox", f"textbox '{a.text}'")
        trunc = not term and self.k >= self.max_steps
        self.done, self.success = term or trunc, r > 0
        info = {"invalid": err is not None}
        if self.done:
            info.update(success=self.success, score=float(self.success))
        return StepResult(self.observe(), r, term, trunc, info)

    def evaluate(self) -> EvalResult:
        return EvalResult(self.success, float(self.success))
