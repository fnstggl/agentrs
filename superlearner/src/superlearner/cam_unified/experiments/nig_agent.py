"""Knowledge-gradient / Thompson / greedy / random experimenters over the MechWorld tools (Experiment 2 baselines).

The existing mathematical exploration baselines (spec 8.5, Experiment 2: "Compare learned exploration with the
existing mathematical knowledge-gradient and Thompson-sampling baselines"): D's `learn.math_assist` (the NIG
posterior, knowledge gradient and Thompson draws of CAM 7.0) on the reward regression of `baselines.NIGPredictor`
(per-lever features [d, d^2, e, e^2] + intercept), choosing among discrete single-lever interventions: hold, and for
every lever `grid` evenly spaced values of its public range. Rules:

* ``kg``       online knowledge gradient (Ryzhov-Powell-Frazier): argmax mean + (decisions left - 1) * KG;
* ``thompson`` one posterior draw of the weights, argmax of its predicted reward;
* ``greedy``   argmax posterior mean;
* ``random``   a uniformly random single-lever move.

The agent is driven by `mechrun.run_batch` through the same tools, clock and verified rewards as CAM Unified (one
decision = one lever setting = one tick), and reads only public information: the tool schemas (ranges), the setting
observations at reset and its own rewards. Actions are labelled ``ActionSource.BASELINE``.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import torch

from ..interfaces import Action, ActionSource, ObsKind
from ..learn import math_assist as MA
from .baselines import NIGPredictor, query_features


class NIGToolAgent:
    def __init__(self, rule: str = "kg", prior: Optional[NIGPredictor] = None, grid: int = 5, seed: int = 0,
                 ema: float = 0.5):
        if rule not in ("kg", "thompson", "greedy", "random"):
            raise ValueError(rule)
        self.rule, self.grid, self.ema = rule, int(grid), float(ema)
        self.prior = prior or NIGPredictor(meta=False)
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.env_ids: list = []
        self.compute = {"act_s": 0.0, "decisions": 0, "candidates_scored": 0}

    # ---- CamUnifiedAgent-compatible episode control (what mechrun.run_batch calls)
    def reset(self, env_ids) -> None:
        self.env_ids = list(env_ids)
        self.state = [None] * len(self.env_ids)

    def set_goals(self, goals) -> None:
        self.goals = list(goals)
        self.horizon = [int((g.metrics or {}).get("horizon_ticks", 100)) for g in goals]

    def flush(self, reason: str = "") -> list:
        return []

    def update(self) -> dict:
        return {"updated": False}

    def _init_env(self, b: int, obs: Sequence, tools: Sequence) -> dict:
        levers = []
        for t in tools:
            if not t.name.startswith("set_"):
                continue
            p = t.parameters["properties"]["value"]
            levers.append((t.name, float(p["minimum"]), float(p["maximum"]), p["type"] == "integer"))
        cur = {}
        for o in obs:
            if o.kind == ObsKind.NUMERIC and (o.meta or {}).get("field") == "setting" and (o.entity or "").startswith("lever:"):
                cur["set_" + o.entity[6:]] = float(o.value)
        raw = np.array([cur.get(n, (lo + hi) / 2) for n, lo, hi, _ in levers])
        lo = np.array([x[1] for x in levers])
        hi = np.array([x[2] for x in levers])
        x0 = np.where(hi > lo, 2 * (raw - lo) / np.maximum(hi - lo, 1e-12) - 1, 0.0)
        return {"levers": levers, "lo": lo, "hi": hi, "raw": raw, "x0": x0, "e": x0.copy(), "rows": [], "ys": [],
                "pending": None, "t": 0}

    def _norm(self, s: dict, raw: np.ndarray) -> np.ndarray:
        return np.where(s["hi"] > s["lo"], 2 * (raw - s["lo"]) / np.maximum(s["hi"] - s["lo"], 1e-12) - 1, 0.0)

    def _candidates(self, s: dict) -> list:
        out = [(0, s["raw"][0])]                                    # hold: re-set lever 0 to its current value
        for j, (_, lo, hi, integer) in enumerate(s["levers"]):
            for v in np.linspace(lo, hi, self.grid):
                v = float(round(v)) if integer else float(v)
                if abs(v - s["raw"][j]) > 1e-9:
                    out.append((j, v))
        return out

    def act(self, ob, tools) -> list:
        t0 = time.perf_counter()
        acts = []
        for b in range(len(self.env_ids)):
            if ob.done[b]:
                acts.append(None)
                continue
            if self.state[b] is None:
                self.state[b] = self._init_env(b, ob.observations[b], tools[b])
            s = self.state[b]
            cands = self._candidates(s)
            Xq = []
            for j, v in cands:
                raw = s["raw"].copy()
                raw[j] = v
                Xq.append(self._norm(s, raw))
            Xq = np.array(Xq)
            if self.rule == "random":
                c = int(self.rng.integers(1, len(cands))) if len(cands) > 1 else 0
            else:
                L = len(s["levers"])
                phi_s = np.array(s["rows"]) if s["rows"] else np.zeros((1, 1 + 4 * L))
                y_s = np.array(s["ys"]) if s["ys"] else np.zeros(1)
                w_s = np.ones(len(s["rows"])) if s["rows"] else np.zeros(1)
                post = self.prior._post(phi_s, y_s, w_s, L)
                pq = torch.as_tensor(query_features(Xq, s["e"], s["x0"], self.ema), dtype=torch.float64).unsqueeze(0)
                noise = 1e-9 * self.rng.random(len(cands))
                if self.rule == "thompson":
                    wdraw = MA.sample_weights(post, 1, self.gen)[0, 0, 0]          # [D]
                    score = (pq[0] @ wdraw).numpy()
                else:
                    mean, cov = MA.mean_cov(post, pq)
                    m = mean[0, :, 0]
                    if self.rule == "greedy":
                        score = m.numpy()
                    else:
                        kg = MA.knowledge_gradient(m.unsqueeze(0), cov[:, 0], post.s2.expand(1, len(cands)))[0]
                        left = max(1, self.horizon[b] - s["t"])
                        score = (m + (left - 1) * kg).numpy()
                c = int(np.argmax(score + noise))
                self.compute["candidates_scored"] += len(cands)
            j, v = cands[c]
            name, lo, hi, integer = s["levers"][j]
            s["pending"] = (j, v, Xq[c])
            acts.append(Action(name, {"value": int(round(v)) if integer else float(v)}, source=ActionSource.BASELINE,
                               logp=0.0, logp_parts={"tool": 0.0, "rule": self.rule}))
        self.compute["act_s"] += time.perf_counter() - t0
        self.compute["decisions"] += sum(a is not None for a in acts)
        return acts

    def observe(self, results, rewards, nob) -> list:
        for b in range(len(self.env_ids)):
            s = self.state[b]
            if s is None or s["pending"] is None:
                continue
            j, v, x = s["pending"]
            s["raw"][j] = v
            phi = query_features(x[None], s["e"], s["x0"], self.ema)[0]
            s["e"] = (1 - self.ema) * s["e"] + self.ema * x
            s["rows"].append(phi)
            s["ys"].append(float(rewards[b]))
            s["pending"] = None
            s["t"] += 1
        return []
