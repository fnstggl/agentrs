"""Search-process controllers ("discovery of discovery").

All controllers make the same three decisions, so they can be compared at matched budgets:
  * ``decide_job``  — budget allocation: promote an existing candidate to the next ASHA rung, or explore (new child);
  * ``propose``     — candidate selection (which parent) and mutation-operator choice;
  * ``order``       — evaluation/promotion priority among promotable candidates.

``UniformController``  — uniform parent among evaluated candidates, uniform operator over all operators; ASHA's
                          default allocation (promote whenever something is promotable).
``PBTController``      — PBT-style truncation selection (parent from the top quartile) and hyper-parameter
                          perturbation only (fixed operator distribution over ``PBT_OPERATORS``). No weight copying,
                          so it follows the same scratch-training protocol as the other arms.
``AdaptiveController`` — contextual Thompson-sampling bandit over operators (Bayesian linear regression per operator
                          on parent features), rewarded by the *measured* outer-validation improvement of each child
                          over its parent at the same budget, normalised by the child's compute cost; diversity-aware
                          Boltzmann parent selection; learned promote-vs-explore allocation from the empirical gain
                          per GPU-hour of past promotions and past explorations; an exploration floor.

Rewards come only from independent outer-validation measurements recorded in ``Population.hypotheses``; nothing a
candidate predicts about itself is ever used. The controller state is a deterministic function of that history, so
a resumed search reconstructs it exactly.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from agentrs.cam2.search.population import ASHA, Candidate, Population
from agentrs.cam2.search.spec import OPERATORS, PBT_OPERATORS


def _evaluated(pop: Population) -> list:
    return [c for c in pop.cands.values() if c.rung >= 0 and c.status in ("ready", "stopped")]


class Controller:
    name = "base"

    def __init__(self, operators: Optional[list] = None, seed: int = 0):
        self.ops = list(operators or OPERATORS)
        self.rng = random.Random(seed)

    def decide_job(self, pop: Population, asha: ASHA) -> tuple:
        pro = self.order(pop, asha.promotable(pop))
        if pro:
            return "promote", pro[0]
        return "new", None

    def order(self, pop: Population, promotable: list) -> list:
        return promotable

    def propose(self, pop: Population) -> tuple[Candidate, str]:
        raise NotImplementedError

    def probs(self, pop: Population) -> dict:
        return {o: 1 / len(self.ops) for o in self.ops}

    def context(self, pop: Population, parent: Candidate) -> np.ndarray:
        """Parent features recorded with every hypothesis (all controllers record them; only the adaptive one
        learns from them): bias, rung-0 validation score, log10 params / 8, lineage depth / 10, budget fraction."""
        s0 = parent.score(0) if parent.score(0) is not None else 0.0
        n_params = parent.validation.get("params", {}).get("total", 5e7)
        depth = len(parent.history)
        used = pop.budget.used_gpu_hours / max(1e-6, pop.budget.gpu_hours)
        return np.array([1.0, s0, math.log10(max(1.0, n_params)) / 8.0, min(depth, 10) / 10.0, used], dtype=np.float64)


class UniformController(Controller):
    name = "uniform"

    def propose(self, pop):
        parent = self.rng.choice(_evaluated(pop))
        return parent, self.rng.choice(self.ops)


class PBTController(Controller):
    name = "pbt"

    def __init__(self, operators=None, seed: int = 0, quantile: float = 0.25):
        super().__init__([o for o in (operators or PBT_OPERATORS) if o in PBT_OPERATORS] or PBT_OPERATORS, seed)
        self.q = quantile

    def propose(self, pop):
        ev = sorted(_evaluated(pop), key=lambda c: -c.score(0))
        top = ev[: max(1, int(math.ceil(len(ev) * self.q)))]
        return self.rng.choice(top), self.rng.choice(self.ops)

    def probs(self, pop):
        return {o: (1 / len(self.ops) if o in self.ops else 0.0) for o in OPERATORS}


class AdaptiveController(Controller):
    name = "adaptive"

    def __init__(self, operators=None, seed: int = 0, eps: float = 0.15, prior_var: float = 0.02,
                 noise_var: float = 0.01, temp: float = 0.05, max_rung: int = 1, fail_reward: float = -0.05):
        super().__init__(operators, seed)
        self.eps, self.prior_var, self.noise_var, self.temp = eps, prior_var, noise_var, temp
        self.max_rung, self.fail_reward = max_rung, fail_reward

    # ---------------------------------------------------------------- rewards
    @staticmethod
    def ref_cost(pop: Population) -> float:
        c = [h["cost_h"] for h in pop.hypotheses if h.get("cost_h") and h.get("outcome") == "evaluated"]
        return float(np.median(c)) if c else 0.08

    def reward(self, pop: Population, h: dict) -> float:
        if h["outcome"] != "evaluated":
            return self.fail_reward
        return float(h["delta"]) / max(0.25, h["cost_h"] / self.ref_cost(pop))

    def posterior(self, pop: Population, op: str) -> tuple[np.ndarray, np.ndarray]:
        dim = 5
        A = np.eye(dim) / self.prior_var
        b = np.zeros(dim)
        for h in pop.hypotheses:
            if h["op"] != op or "context" not in h:
                continue
            x = np.asarray(h["context"], dtype=np.float64)
            y = self.reward(pop, h)
            A += np.outer(x, x) / self.noise_var
            b += x * y / self.noise_var
        cov = np.linalg.inv(A)
        return cov @ b, cov

    # ---------------------------------------------------------------- decisions
    def probs(self, pop: Population, parent: Optional[Candidate] = None, n_samples: int = 400) -> dict:
        """Monte-Carlo estimate of the Thompson-sampling operator distribution (mixed with the exploration floor)."""
        ev = _evaluated(pop)
        if not ev:
            return {o: 1 / len(self.ops) for o in self.ops}
        parent = parent or max(ev, key=lambda c: c.score(0))
        x = self.context(pop, parent)
        rng = np.random.default_rng(len(pop.hypotheses))
        post = {o: self.posterior(pop, o) for o in self.ops}
        wins = {o: 0 for o in self.ops}
        for _ in range(n_samples):
            best = max(self.ops, key=lambda o: float(x @ rng.multivariate_normal(*post[o])))
            wins[best] += 1
        return {o: (1 - self.eps) * wins[o] / n_samples + self.eps / len(self.ops) for o in self.ops}

    def select_parent(self, pop: Population) -> Candidate:
        ev = _evaluated(pop)
        fams: dict = {}
        for c in ev:
            fams[c.family()] = fams.get(c.family(), 0) + 1
        kids: dict = {}
        for c in pop.cands.values():
            if c.parent:
                kids[c.parent] = kids.get(c.parent, 0) + 1
        best = max(c.score(0) for c in ev)
        w = []
        for c in ev:
            # Boltzmann over measured rung-0 score; families that dominate the population and parents that already
            # have many children are down-weighted (keeps the search from collapsing onto one lineage)
            share = fams[c.family()] / len(ev)
            w.append(math.exp((c.score(0) - best) / self.temp) / (1 + kids.get(c.cid, 0)) * (1.0 / (0.5 + share)))
        return self.rng.choices(ev, weights=w, k=1)[0]

    def propose(self, pop):
        parent = self.select_parent(pop)
        if self.rng.random() < self.eps:
            return parent, self.rng.choice(self.ops)
        x = self.context(pop, parent)
        nrng = np.random.default_rng(self.rng.randrange(2**31))
        scores = {}
        for o in self.ops:
            mu, cov = self.posterior(pop, o)
            scores[o] = float(x @ nrng.multivariate_normal(mu, cov))
        return parent, max(scores, key=scores.get)

    def order(self, pop, promotable):
        # promote under-represented families first among equally ranked candidates
        fams: dict = {}
        for c in _evaluated(pop):
            fams[c.family()] = fams.get(c.family(), 0) + 1
        return sorted(promotable, key=lambda rc: (-rc[0], -(rc[1].score(rc[0]) - 0.02 * fams.get(rc[1].family(), 0))))

    def decide_job(self, pop: Population, asha: ASHA) -> tuple:
        pro = self.order(pop, asha.promotable(pop))
        if not pro:
            return "new", None
        if not pop.budget.can_create(len(pop.order)):
            return "promote", pro[0]
        # learned allocation: expected gain of the population's best validation score per GPU-hour
        best = max(c.score(c.rung) for c in _evaluated(pop))
        r, cand = pro[0]
        rung_gain = [h for h in pop.events if h["kind"] == "rung_gain" and h["rung"] == r + 1]
        explore = [h for h in pop.hypotheses if h.get("outcome") == "evaluated"]
        if len(rung_gain) < 1 or len(explore) < 3:
            return "promote", pro[0]       # ASHA default until both kinds of evidence exist
        exp_promote = np.mean([max(0.0, cand.score(r) + g["gain"] - best) for g in rung_gain])
        cost_promote = np.mean([g["cost_h"] for g in rung_gain])
        exp_new = np.mean([max(0.0, h["child_score"] - best) for h in explore])
        cost_new = self.ref_cost(pop)
        pop.log("allocation", promote_gain_per_h=float(exp_promote / max(1e-6, cost_promote)),
                new_gain_per_h=float(exp_new / max(1e-6, cost_new)), cand=cand.cid, rung=r)
        if exp_promote / max(1e-6, cost_promote) >= exp_new / max(1e-6, cost_new):
            return "promote", pro[0]
        return "new", None


def make_controller(name: str, seed: int = 0, operators: Optional[list] = None, **kw) -> Controller:
    if name == "uniform":
        return UniformController(operators, seed)
    if name == "pbt":
        return PBTController(operators, seed)
    if name == "adaptive":
        return AdaptiveController(operators, seed, **kw)
    raise ValueError(name)
