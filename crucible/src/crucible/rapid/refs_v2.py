"""A generic adaptive public-information reference learner (evaluator side; scripted, labelled as such in evidence).

`adaptive(view, env)` plays one episode through a `PublicView` - actions, observations, schemas, nothing hidden -
and learns while it plays from what it observes, with no knowledge of any mechanism:

* continuous levers: simultaneous-perturbation extremum seeking (SPSA). Every step it holds u + d*p and then
  u - d*p (p a random sign vector) for a few ticks each, compares the observed net income of the later ticks of
  each hold (so effects that take a tick or two to show are seen), and moves u by a sign-adaptive step (Rprop:
  the step grows while the estimated direction is stable and shrinks when it flips). It never stops perturbing,
  so it keeps tracking an optimum that moves;
* discrete decisions: a discounted bandit over each decision's options (old observations are forgotten, so a
  change of which option is best is followed);
* allocations: the same extremum seeking on the allocation's weights (a softmax of free coordinates).

Several learner seats each run their own learner on their own observed net income. This is the reference
admission compares simple policies against when a family has no mechanism-specific reference: a world where it
cannot beat the best constant setting, threshold rule or explore-then-commit policy by a clear margin does not
reward adaptation, whatever its mechanism.
"""
from __future__ import annotations

import math

import numpy as np

from . import dominance as D


class _SeatLearner:
    def __init__(self, coords: D.Coords, rng, hold: int):
        self.c, self.rng, self.hold = coords, rng, hold
        L = len(coords.levers)
        self.u = np.full(L, 0.5)
        self.step = np.full(L, 0.12)
        self.last_sign = np.zeros(L)
        nA = len(coords.alloc)
        self.z = np.zeros(nA)                     # allocation logits
        self.zstep = np.full(nA, 0.5)
        self.f = 1.0
        self.q = {d: {o: 0.0 for o in opts} for d, opts in coords.decisions.items()}
        self.n = {d: {o: 0.0 for o in opts} for d, opts in coords.decisions.items()}
        self.choice = {d: opts[0] for d, opts in coords.decisions.items()}
        self.phase, self.k, self.buf = 0, 0, []
        self.p = np.ones(L)
        self.pz = np.ones(nA)
        self.y = [0.0, 0.0]
        self.delta = 0.15

    def _setting(self, sign: int) -> dict:
        # the probe size follows the step size: large while the optimum is far or moving, small near it
        d = np.clip(self.step, 0.04, self.delta) if len(self.u) else 0.0
        u = np.clip(self.u + sign * d * self.p, 0.0, 1.0) if len(self.u) else self.u
        z = self.z + sign * 0.6 * self.pz if len(self.z) else self.z
        w = np.exp(z - z.max()) if len(z) else z
        return {"u": [float(x) for x in u], "d": dict(self.choice),
                "w": [float(x) for x in (w / w.sum())] if len(w) else [], "f": self.f, "e": None,
                "p": 0 if self.c.plans else None}

    def act(self, t: int) -> list[dict]:
        sign = 1 if self.phase == 0 else -1
        return self.c.actions(self._setting(sign))

    def observe(self, net: float) -> None:
        self.k += 1
        if self.k > self.hold // 2:                 # the later ticks of a hold: lagged effects have arrived
            self.buf.append(net)
        if self.k < self.hold:
            return
        self.y[self.phase] = float(np.mean(self.buf)) if self.buf else 0.0
        self.buf, self.k = [], 0
        for d in self.q:                            # discounted credit for the options held during this half
            for o in self.q[d]:
                self.q[d][o] *= 0.85
                self.n[d][o] *= 0.85
            o = self.choice[d]
            self.q[d][o] += self.y[self.phase]
            self.n[d][o] += 1.0
        if self.phase == 0:
            self.phase = 1
            return
        self.phase = 0
        g = self.y[0] - self.y[1]
        if len(self.u):
            s = np.sign(g) * self.p
            same = s * self.last_sign
            self.step = np.clip(np.where(same > 0, self.step * 1.4, np.where(same < 0, self.step * 0.5,
                                                                              self.step)), 0.02, 0.3)
            self.u = np.clip(self.u + self.step * s, 0.0, 1.0)
            self.last_sign = s
            self.p = self.rng.choice([-1.0, 1.0], size=len(self.u))
        if len(self.z):
            self.z = self.z + 0.4 * np.sign(g) * self.pz
            self.pz = self.rng.choice([-1.0, 1.0], size=len(self.z))
        for d, opts in self.c.decisions.items():    # discounted UCB over the options of each decision
            tot = sum(self.n[d].values()) + 1.0
            best, bv = opts[0], -math.inf
            for o in opts:
                n = self.n[d][o]
                v = (self.q[d][o] / n if n > 0 else math.inf) + 0.5 * (abs(g) + 1.0) * math.sqrt(math.log(tot + 1)
                                                                                                / (n + 1e-9))
                if v > bv:
                    best, bv = o, v
            self.choice[d] = best


def adaptive(view, env=None, hold: int = 3, seed: int = 0) -> None:
    """Play the rest of the episode with one adaptive learner per learner seat (public information only)."""
    agents = view.agents()
    rng = np.random.default_rng(seed)
    learners = {a: _SeatLearner(D.Coords(view.action_space(a), a, agents), rng, hold) for a in agents}
    obs = {a: view.observe(a) for a in agents}
    while not view.done:
        acts = {}
        for a, L in learners.items():
            atoms = L.act(view.t)
            if not atoms:
                continue
            sp = view.action_space(a)
            if len(atoms) == 1:
                acts[a] = atoms[0]
            elif "sequence" in sp.kinds:
                inner = sp.limits.get("sequence_kinds")
                steps = [x for x in atoms if inner is None or x["kind"] in inner]
                acts[a] = {"kind": "sequence", "steps": steps} if steps else atoms[view.t % len(atoms)]
            else:
                acts[a] = atoms[view.t % len(atoms)]
        r = view.step(acts if len(agents) > 1 else acts.get(agents[0]))
        obs = r.get("observations") or {agents[0]: r["observation"]}
        for a, L in learners.items():
            L.observe(D._net(obs[a]))
