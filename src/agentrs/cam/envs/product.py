"""Product adoption / early-stage company simulator.

Hidden market: 4 user segments with hidden sizes, needs over 6 product features, price sensitivity
and channel affinities. The agent can research a segment (noisy interviews), build features (cost +
multi-week delay before shipping), set the price, run acquisition campaigns on 4 channels (spend
continues for 4 weeks; signups arrive with a 1-2 week lag), improve onboarding, or wait. Users churn
according to product-need fit and onboarding; satisfied users refer others; revenue funds the
company. Goal: at least ``target`` *retained active* users at the deadline without running out of
money — repeatedly spending on one channel without product fit buys signups that churn away.

Train/val markets: segment needs ~ Dirichlet(0.6), consumer-like channel affinities. Held-out
(test): B2B-like markets (one dominant niche segment, partner/content channels dominate, higher
price tolerance) with different needs structure.
"""
from __future__ import annotations

import copy

import numpy as np

from agentrs.cam.envs.base import CamEnv, Grammar, StepResult, logb, nb

NF, NS, NC = 6, 4, 4
FEATS = [f"F{i}" for i in range(NF)]
SEGS = [f"S{i}" for i in range(NS)]
CHANNELS = ["ads", "content", "partners", "referral"]
PRICES = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
SPEND = [0.0, 1000.0, 3000.0, 8000.0, 20000.0]
WEEKS = 26


class ProductEnv(CamEnv):
    family = "product"
    keywords = ["GOAL", "GROW", "USERS", "BY", "WEEK", "TARGET", "BUDGET", "ACTIVE", "SIGNUPS", "CHURN", "REVENUE",
                "PRICE", "ONB", "BUILT", "BUILDING", "CH", "RESEARCH", "BUILD", "SETPRICE", "CAMPAIGN", "ONBOARD",
                "WAIT", "NEEDS", "SENS", "HIGH", "MED", "LOW", "P0", "P1", "P2", "P3", "P4", "P5", "B1", "B2", "B3",
                "B4", "W1", "W2", "W3", "W4", *FEATS, *SEGS, *CHANNELS]
    events = ["users_up", "churn_spike", "budget_low", "shipped", "research_done", "bankrupt"]
    n_features = 22
    max_steps = 26

    def reset(self, seed: int, split: str = "train") -> None:
        self._seed(seed, split)
        g = self.np
        test = split == "test"
        if test:
            sizes = g.dirichlet([4.0, 0.6, 0.6, 0.6]) * g.uniform(20_000, 60_000)
            self.needs = g.dirichlet(np.full(NF, 0.35), size=NS)
            self.aff = g.dirichlet([0.4, 1.5, 2.5, 0.8], size=NS)
            self.sens = g.uniform(0.01, 0.04, size=NS)
        else:
            sizes = g.dirichlet(np.ones(NS)) * g.uniform(60_000, 200_000)
            self.needs = g.dirichlet(np.full(NF, 0.6), size=NS)
            self.aff = g.dirichlet([2.0, 1.0, 0.6, 1.0], size=NS)
            self.sens = g.uniform(0.03, 0.12, size=NS)
        self.sizes = sizes
        self.eff = np.array([0.020, 0.010, 0.014, 0.0]) * g.uniform(0.7, 1.3, size=NC)
        self.week = 0
        self.budget = float(g.uniform(60_000, 120_000))
        self.built = np.zeros(NF, dtype=bool)
        self.building: dict[int, int] = {}
        self.price = 2
        self.onb = 0
        self.spend = np.zeros(NC)
        self.spend_left = np.zeros(NC, dtype=int)
        self.active = np.zeros(NS)
        self.pipeline = np.zeros((3, NS))  # signups arriving in 1, 2, 3 weeks
        self.reached = np.zeros(NS)
        self.last = {"signups": 0.0, "churn": 0.0, "revenue": 0.0, "by_ch": np.zeros(NC)}
        self.research: dict[int, tuple] = {}
        self.answer = None
        self.target = 1.0
        self.target = max(500.0, self._expert_users() * self.rng.uniform(0.45, 0.7))
        if split == "train" and self.difficulty < 1.0:  # curriculum: easier goals during training only
            self.target = max(50.0, self.target * self.difficulty)

    # ------------------------------------------------------------------ dynamics
    def _fit(self):
        return (self.needs * self.built[None, :]).sum(-1)  # share of each segment's needs met

    def _accept(self):
        value = 60.0 * self._fit()
        return 1 / (1 + np.exp(-(value * 0.15 - self.sens * PRICES[self.price] * 3 - 1.0)))

    def _advance(self, weeks: int, ev: dict) -> None:
        for _ in range(weeks):
            if self.week >= WEEKS:
                break
            self.week += 1
            for f in list(self.building):
                self.building[f] -= 1
                if self.building[f] <= 0:
                    self.built[f] = True
                    del self.building[f]
                    ev["shipped"] = True
            acc = self._accept()
            fit = self._fit()
            remaining = np.maximum(0.0, self.sizes - self.reached)
            by_ch = np.zeros(NC)
            new = np.zeros(NS)
            for c in range(NC):
                if c == 3:  # referral: satisfied users bring others
                    reach = self.active * 0.04 * fit
                else:
                    reach = self.spend[c] * self.eff[c] * self.aff[:, c] * NS * (remaining / np.maximum(1, self.sizes))
                    if c == 1:  # content compounds slowly with the weeks it has been running
                        reach = reach * min(1.0, 0.3 + 0.1 * (4 - self.spend_left[c]))
                s = reach * acc * self.np.uniform(0.8, 1.2, size=NS)
                new += s
                by_ch[c] = s.sum()
            new = np.minimum(new, remaining)
            self.reached += new
            self.pipeline[0] += new * 0.5
            self.pipeline[1] += new * 0.5
            arrived = self.pipeline[0].copy()
            self.pipeline = np.roll(self.pipeline, -1, axis=0)
            self.pipeline[-1] = 0
            churn_rate = 0.02 + 0.30 * (1 - fit) * (1 - 0.22 * self.onb)
            churned = self.active * churn_rate
            before = self.active.sum()
            self.active = self.active + arrived - churned
            revenue = float((self.active * PRICES[self.price] / 4.3 * 0.6).sum())
            spent = float(self.spend.sum())
            self.budget += revenue - spent
            self.spend_left = np.maximum(0, self.spend_left - 1)
            self.spend = np.where(self.spend_left > 0, self.spend, 0.0)
            self.last = {"signups": float(arrived.sum()), "churn": float(churned.sum()), "revenue": revenue,
                         "by_ch": by_ch}
            if self.active.sum() > 1.1 * before + 10:
                ev["users_up"] = True
            if churned.sum() > 0.15 * max(1.0, before):
                ev["churn_spike"] = True
            if self.budget < 10_000:
                ev["budget_low"] = True
            if self.budget < 0:
                ev["bankrupt"] = True
                break

    def _expert_users(self) -> float:
        """Achievable reference: a heuristic expert that knows the hidden market (used only to set a
        solvable but demanding target)."""
        e = copy.deepcopy(self)
        w = (e.needs * e.sizes[:, None]).sum(0)
        for f in np.argsort(-w)[:3]:
            e.building[int(f)] = 2
        ev = {}
        e._advance(2, ev)
        best_p = max(range(len(PRICES)), key=lambda p: (setattr(e, "price", p), (e._accept() * e.sizes).sum() * max(1.0, PRICES[p]) ** 0.3)[1])
        e.price = best_p
        c = int(np.argmax((e.aff[:, :3] * e.sizes[:, None]).sum(0) * e.eff[:3]))
        e.onb = 2
        while e.week < WEEKS and e.budget > 0:
            if e.spend_left[c] == 0:
                e.spend[c], e.spend_left[c] = SPEND[2], 4
            e._advance(1, ev)
        return float(e.active.sum())

    # ------------------------------------------------------------------ interface
    def goal_tokens(self):
        return ["GOAL", "GROW", "USERS", "TARGET", logb(self.target, 200_000), "BY", "WEEK", nb(WEEKS, 0, 26)]

    def obs_tokens(self):
        t = ["WEEK", nb(self.week, 0, 26), "BUDGET", logb(max(0, self.budget), 300_000), "ACTIVE",
             logb(self.active.sum(), 200_000), "SIGNUPS", logb(self.last["signups"], 20_000), "CHURN",
             logb(self.last["churn"], 20_000), "REVENUE", logb(self.last["revenue"], 100_000), "PRICE",
             f"P{self.price}", "ONB", nb(self.onb, 0, 3)]
        t += ["BUILT"] + [FEATS[i] for i in range(NF) if self.built[i]]
        t += ["BUILDING"] + [FEATS[i] for i in self.building]
        for c in range(NC):
            t += ["CH", CHANNELS[c], logb(self.last["by_ch"][c], 20_000)]
        if self.answer is not None:
            t += list(self.answer)
        for s, (top, sens) in sorted(self.research.items()):
            t += ["NEEDS", SEGS[s], *top, "SENS", sens]
        return t

    def features(self):
        f = np.zeros(self.n_features, dtype=np.float32)
        f[0] = self.week / WEEKS
        f[1] = np.log1p(self.active.sum()) / 12
        f[2] = np.log1p(self.last["signups"]) / 10
        f[3] = self.last["churn"] / max(1.0, self.active.sum() + self.last["churn"])
        f[4] = np.log1p(max(0, self.budget)) / 13
        f[5] = self.price / 5
        f[6:12] = self.built
        f[12] = self.onb / 3
        f[13:17] = self.spend / SPEND[-1]
        f[17:21] = np.log1p(self.last["by_ch"]) / 10
        f[21] = min(1.0, self.active.sum() / self.target)
        return f

    def grammar(self):
        return GRAMMAR

    def progress(self):
        return float(min(1.0, self.active.sum() / self.target))

    def step(self, action):
        ev = {k: False for k in self.events}
        typ = action[0]
        self.answer = None
        weeks, valid = 1, True
        if typ == "RESEARCH":
            s = SEGS.index(action[1])
            noisy = self.needs[s] + self.np.normal(0, 0.05, NF)
            top = tuple(FEATS[i] for i in np.argsort(-noisy)[:2])
            sens = "HIGH" if self.sens[s] > 0.08 else ("MED" if self.sens[s] > 0.04 else "LOW")
            self.research[s] = (top, sens)
            self.budget -= 2000
            ev["research_done"] = True
        elif typ == "BUILD":
            f = FEATS.index(action[1])
            if self.built[f] or f in self.building:
                valid = False
            else:
                self.building[f] = self.rng.randint(2, 4)
                self.budget -= 15_000
        elif typ == "SETPRICE":
            self.price = int(action[1][1])
        elif typ == "CAMPAIGN":
            c = CHANNELS.index(action[1])
            self.spend[c] = SPEND[int(action[2][1])]
            self.spend_left[c] = 4
        elif typ == "ONBOARD":
            if self.onb >= 3:
                valid = False
            else:
                self.onb += 1
                self.budget -= 8000
                weeks = 2
        elif typ == "WAIT":
            weeks = int(action[1][1])
        self._advance(weeks, ev)
        if self.week >= WEEKS or self.budget < 0:
            self.done = True
            self.success = bool(self.budget >= 0 and self.active.sum() >= self.target)
        return StepResult(elapsed_h=168.0 * weeks, events=ev, done=self.done,
                          success=self.success if self.done else None, valid=valid)

    def hidden(self):
        return {"sizes": self.sizes.round().tolist(), "target": round(self.target), "sens": self.sens.round(3).tolist()}


GRAMMAR = Grammar({"RESEARCH": [SEGS], "BUILD": [FEATS], "SETPRICE": [["P0", "P1", "P2", "P3", "P4", "P5"]],
                        "CAMPAIGN": [CHANNELS[:3], ["B1", "B2", "B3", "B4"]], "ONBOARD": [],
                        "WAIT": [["W1", "W2", "W3", "W4"]]}, name="product")
