"""`ToolValueLearner`: a statistical reference learner for tool worlds (Clarification 1 names statistical learners
beside small neural RL agents; `ReferenceLearner` is the neural one).

It learns how much each tool is worth to it from its own valuation of what it observes. After an action it takes
the change of its signal between the observation it acted on and the next one - by default its own money as the
observation shows it, the sum of the ``bank.balances.*`` numbers (the same `observed` signal `ReferenceLearner`
uses in observation-only mode). At `update` it folds the buffered changes into

* a baseline: the median change per step over its last 256 steps, whatever it did (what its money does on its own:
  an operator's revenue and running costs, salaries, ...; a median, so one windfall does not shift it), and
* per tool: the recency-weighted mean of the change minus the baseline at the time (the tool's marginal effect),
  shrunk toward 0 by ``n / (n + shrink)`` for a tool tried n times.

It acts by a softmax over the tools' values (``wait`` is an option with its own value) at a temperature of
`temperature` or the median absolute deviation of those last 256 changes from the baseline, whichever is larger
(so the noise of its money's own movements is not mistaken for a tool's effect, and a rare windfall does not flatten
its preferences); a tool it never tried has the prior value 0. A copy with no values (the frozen initial copy) acts
uniformly at random over the offered tools. Arguments come from the tool's JSON schema and what the learner was
shown (`ArgumentMemory`, as in `ReferenceLearner`): enums and bounded numbers uniformly, ids and name tokens it saw under the argument's name,
its own accounts for account arguments.

It is deliberately simple and transparent: values do not depend on the state, credit reaches only the step of the
action, and a one-off windfall keeps a tool's value high. It exists so that improvement over a frozen copy can be
measured in kernel worlds with hundreds of tools, where the neural learner's near-uniform policy rarely repeats a
tool (MASTER_SPEC 11.6: separate algorithmic limitations from environment failures). Crucible computes nothing for
it: the signal is its own reading of legitimate observations, and the environment's reward (0.0 in
observation-only mode) is used only when ``signal="reward"`` is chosen explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

import numpy as np

from .modes import OBSERVATION_ONLY, check_mode
from .reference import ArgumentMemory, string_candidates

VERSION = "tool-value/1"
WAIT = "<wait>"
MONEY_KEYS = r"^bank\.balances\."
RESET = "<reset>"


class ToolValueLearner(ArgumentMemory):
    name = "tool-value"
    kind = "statistical"
    version = VERSION

    def __init__(self, seed: int = 0, reward_mode: str | None = None, *, temperature: float = 0.5,
                 signal: str = "auto", signal_keys: str = MONEY_KEYS, signal_scale: float = 0.01,
                 recency: float = 0.05, shrink: float = 1.0, credit_horizon: int = 1, credit_decay: float = 0.9):
        """`temperature` is in signal units (default: dollars, `signal_scale` 0.01 per cent): a tool that loses
        50 cents a call more than the baseline is e times less likely than an untried one (measured in a
        civilisation: a person's fees are worth $0.7-$6 a call on average - a $5 listing, a $20 domain, a $30
        subscription that succeeds one call in ten - and at $10 the trained values hardly changed what it did).
        `recency` is the smallest weight of a new sample in a running mean (1/n before that). `credit_horizon` > 1
        also credits a change to the actions of the previous steps, the j-th back with weight ``credit_decay ** j`` (for consequences that
        arrive later than the next observation); 1 credits only the step's own action."""
        self.seed = int(seed)
        self.credit_horizon, self.credit_decay = max(1, int(credit_horizon)), float(credit_decay)
        self.trace: list[str] = []                      # the tools of the last credit_horizon - 1 steps
        self.reward_mode = check_mode(reward_mode)
        self.temperature = float(temperature)
        self.recency, self.shrink = float(recency), float(shrink)
        self.recent: list[float] = []                   # the signal changes of the last WINDOW steps
        self.signal, self.signal_keys, self.signal_scale = signal, signal_keys, float(signal_scale)
        self.values: dict[str, list[float]] = {}        # tool -> [n, mean change of the signal]
        self.memory: dict[str, dict] = {}
        self.ids: dict[str, list[str]] = {}
        self.rng = np.random.default_rng([self.seed, 23])
        self.frozen = False
        self.updates = 0
        self.steps = 0
        self.buf: list[tuple[str, float]] = []
        self._pending: dict | None = None
        self.actor: str | None = None

    # ------------------------------------------------------------------ context
    def reset(self, actor: str | None = None, rng=None) -> None:
        """A new world, episode or agent: no credit crosses the boundary (marked in the buffer when transitions are
        still waiting for `update`)."""
        self.actor = actor
        self._pending = None
        if self.buf:
            self.buf.append((RESET, 0.0))
        else:
            self.trace = []

    def signal_mode(self) -> str:
        if self.signal == "auto":
            return "observed" if self.reward_mode == OBSERVATION_ONLY else "reward"
        return self.signal

    def _signal_value(self, obs: dict) -> float:
        pat = re.compile(self.signal_keys)
        hits = [float(x) for k, x in (obs.get("numbers") or {}).items() if pat.search(k)]
        if hits or self.signal_keys != MONEY_KEYS:
            return float(sum(hits))              # a chosen signal that is absent (nothing posted yet) is 0
        res = obs.get("resources") or {}
        for k in ("balance_cents", "cash_cents"):
            v = res.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.0

    WINDOW = 256

    def baseline(self) -> float:
        return float(np.median(self.recent)) if self.recent else 0.0

    def temperature_now(self) -> float:
        b = self.baseline()
        mad = float(np.median(np.abs(np.asarray(self.recent) - b))) if self.recent else 0.0
        return max(self.temperature, mad, 1e-9)

    def value(self, tool: str) -> float:
        v = self.values.get(tool)
        return float(v[1]) * v[0] / (v[0] + self.shrink) if v else 0.0

    # ------------------------------------------------------------------ acting
    def act(self, obs: dict, tools: list[dict] | None = None, explore: bool = True) -> dict | None:
        tools = list(tools if tools is not None else obs.get("tools") or [])
        self._harvest(obs.get("last_result"))
        self._harvest_events(obs)
        names = [WAIT] + [t.get("name", "") for t in tools]
        q = np.array([self.value(n) for n in names], np.float64) / self.temperature_now()
        if explore:
            p = np.exp(q - q.max())
            p /= p.sum()
            k = int(self.rng.choice(len(names), p=p))
        else:
            k = int(np.argmax(q))
        self._pending = {"tool": names[k], "signal0": self._signal_value(obs)}
        self.steps += 1
        if k == 0:
            return None
        tool = tools[k - 1]
        return {"tool": tool["name"], "args": self._obj(tool.get("inputSchema") or {}, obs, 0)}

    def _obj(self, schema: dict, obs: dict, depth: int) -> dict:
        props = schema.get("properties") or {}
        out = {}
        for k in sorted(schema.get("required") or []):
            v = self._value(props.get(k) or {}, k, obs, depth)
            if v is not _SKIP:
                out[k] = v
        addl = schema.get("additionalProperties")
        if not props and isinstance(addl, dict) and addl.get("type") in ("number", "integer"):
            for key in sorted(self.memory):
                sp = self.memory[key]
                integer = sp.get("integer", False) or addl.get("type") == "integer"
                out[key] = self._uniform(sp["min"], sp["max"], integer)
        return out

    def _uniform(self, lo: float, hi: float, integer: bool):
        if integer:
            return int(self.rng.integers(int(math.ceil(lo)), int(math.floor(hi)) + 1)) if hi >= lo else int(lo)
        return float(self.rng.uniform(lo, hi)) if hi > lo else float(lo)

    def _value(self, s: dict, key: str, obs: dict, depth: int):
        t = s.get("type")
        t = t[0] if isinstance(t, list) else t
        if "enum" in s and s["enum"]:
            vals = list(s["enum"])
            return vals[int(self.rng.integers(len(vals)))]
        if t in ("number", "integer"):
            lo, hi = s.get("minimum"), s.get("maximum")
            if lo is not None and hi is not None:
                return self._uniform(float(lo), float(hi), t == "integer")
            if "default" in s:
                return s["default"]
            v = float(math.exp(self.rng.uniform(0.0, math.log(1000.0))))     # 1 .. 1000, log-uniform
            if lo is not None:
                v = max(float(lo), v)
            if hi is not None:
                v = min(float(hi), v)
            return int(round(v)) if t == "integer" else v
        if t == "boolean":
            return bool(self.rng.integers(2))
        if t == "string":
            cands = string_candidates(self, obs, key)
            if cands:
                return cands[int(self.rng.integers(len(cands)))]
            return s.get("default") or self.actor or "x"
        if t == "object":
            return self._obj(s, obs, depth + 1) if depth < 4 else {}
        if t == "array":
            return []
        return _SKIP

    def observe(self, reward: float, result: dict | None = None, next_obs: dict | None = None,
                done: bool = False) -> None:
        p, self._pending = self._pending, None
        if result is not None:
            self._harvest(result.get("result") if isinstance(result, dict) else None)
        if p is None or self.frozen:
            return
        mode = self.signal_mode()
        if mode == "reward":
            sig = float(reward)
        elif mode == "observed" and next_obs is not None:
            sig = (self._signal_value(next_obs) - p["signal0"]) * self.signal_scale
        else:
            return
        self.buf.append((p["tool"], float(sig)))

    def observe_result(self, action: dict, result: dict) -> None:
        self._harvest(result.get("result") if isinstance(result, dict) else None)

    def freeze(self, frozen: bool = True) -> None:
        self.frozen = bool(frozen)

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        """Fold the buffered (tool, signal change) pairs into the per-tool running means; clears the buffer."""
        if self.frozen or not self.buf:
            return {"updates": 0, "transitions": len(self.buf)}
        before = self.params_hash()
        n_t = sum(1 for t, _ in self.buf if t != RESET)
        total = 0.0
        for tool, sig in self.buf:
            if tool == RESET:
                self.trace = []
                continue
            adv = sig - self.baseline()                 # against what the money did on its own before this step
            self.recent.append(sig)
            if len(self.recent) > self.WINDOW:
                del self.recent[0]
            for j, t in enumerate([tool] + self.trace[::-1]):
                w = self.credit_decay ** j
                v = self.values.setdefault(t, [0.0, 0.0])
                v[0] += w
                v[1] += (adv - v[1]) * max(w / v[0], self.recency * w)
            if self.credit_horizon > 1:
                self.trace = (self.trace + [tool])[-(self.credit_horizon - 1):]
            total += sig
        self.buf = []
        self.updates += 1
        after = self.params_hash()
        return {"updates": 1, "transitions": n_t, "signal_sum": total, "params_hash_before": before,
                "params_hash_after": after, "changed": before != after}

    # ------------------------------------------------------------------ parameters and checkpoints
    def params(self) -> int:
        return 2 * len(self.values)

    def params_blob(self) -> bytes:
        return json.dumps({"format": VERSION, "values": self.values, "recent": self.recent}, sort_keys=True).encode()

    def load_params(self, blob: bytes) -> None:
        if not blob:
            return
        d = json.loads(blob.decode())
        if d.get("format") != VERSION:
            raise ValueError(f"not a {VERSION} parameter blob")
        self.values = {k: [float(a), float(b)] for k, (a, b) in d["values"].items()}
        self.recent = [float(x) for x in d.get("recent", [])]

    def params_hash(self) -> str:
        return hashlib.sha256(self.params_blob()).hexdigest()

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "params": self.params(), "params_hash": self.params_hash(),
                "updates": self.updates}

    def state(self) -> dict:
        return {"format": VERSION, "seed": self.seed, "reward_mode": self.reward_mode, "temperature": self.temperature,
                "recency": self.recency, "shrink": self.shrink, "recent": self.recent,
                "credit_horizon": self.credit_horizon, "credit_decay": self.credit_decay, "trace": self.trace,
                "signal": self.signal, "signal_keys": self.signal_keys, "signal_scale": self.signal_scale,
                "values": self.values, "memory": self.memory, "ids": self.ids, "rng": self.rng.bit_generator.state,
                "updates": self.updates, "steps": self.steps, "buf": [list(b) for b in self.buf],
                "pending": self._pending, "actor": self.actor, "frozen": self.frozen}

    def save(self, path: str) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.state(), f, sort_keys=True)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path) as f:
            st = json.load(f)
        if st.get("format") != VERSION:
            raise ValueError(f"not a {VERSION} checkpoint")
        self.seed, self.reward_mode = int(st["seed"]), st["reward_mode"]
        self.temperature, self.signal = float(st["temperature"]), st["signal"]
        self.signal_keys, self.signal_scale = st["signal_keys"], float(st["signal_scale"])
        self.values = {k: [float(a), float(b)] for k, (a, b) in st["values"].items()}
        self.recency, self.shrink = float(st["recency"]), float(st["shrink"])
        self.credit_horizon, self.credit_decay = int(st["credit_horizon"]), float(st["credit_decay"])
        self.trace = [str(x) for x in st["trace"]]
        self.recent = [float(x) for x in st["recent"]]
        self.memory, self.ids = dict(st["memory"]), {k: list(v) for k, v in st["ids"].items()}
        self.rng.bit_generator.state = st["rng"]
        self.updates, self.steps = int(st["updates"]), int(st["steps"])
        self.buf = [(str(a), float(b)) for a, b in st["buf"]]
        self._pending, self.actor, self.frozen = st["pending"], st["actor"], bool(st["frozen"])


_SKIP = object()


def describe_values(L: ToolValueLearner, top: int = 8) -> dict:
    """The learner's most and least valued tools (for reports)."""
    items = sorted(((k, L.value(k), int(v[0])) for k, v in L.values.items()), key=lambda x: x[1])
    return {"lowest": [(k, round(m, 3), n) for k, m, n in items[:top]],
            "highest": [(k, round(m, 3), n) for k, m, n in items[::-1][:top]], "tools_valued": len(items),
            "baseline": round(L.baseline(), 3), "temperature": round(L.temperature_now(), 3)}
