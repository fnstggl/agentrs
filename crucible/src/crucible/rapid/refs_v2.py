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


# ====================================================================== law discovery by model selection
def _basis(form: str, X: np.ndarray, q: float) -> np.ndarray:
    a, b = X[:, 0], X[:, 1]
    one = np.ones(len(X))
    sat = lambda v: np.maximum(v, 0) / (np.maximum(v, 0) + q)                 # noqa: E731
    if form == "linear":
        return np.c_[a, b, one]
    if form == "quadratic":
        return np.c_[a, a * a, b, b * b, a * b, one]
    if form == "sat_a+lin_b":
        return np.c_[sat(a), b, one]
    if form == "lin_a+sat_b":
        return np.c_[a, sat(b), one]
    if form == "sat_a+sat_b":
        return np.c_[sat(a), sat(b), one]
    if form == "sat_a*lin_b":
        return np.c_[sat(a) * b, sat(a), one]
    if form == "lin_a*sat_b":
        return np.c_[sat(b) * a, sat(b), one]
    if form == "power_a+lin_b":
        return np.c_[np.abs(a) ** (q / 4.0), b, one]
    if form == "lin_a+power_b":
        return np.c_[a, np.abs(b) ** (q / 4.0), one]
    if form == "step_a+lin_b":
        return np.c_[(a >= q).astype(float), a, b, one]
    if form == "lin_a+step_b":
        return np.c_[a, (b >= q).astype(float), b, one]
    if form == "bump_a+lin_b":
        return np.c_[a * np.exp(-a / max(q, 1e-6)), b, one]
    if form == "lin_a+bump_b":
        return np.c_[a, b * np.exp(-b / max(q, 1e-6)), one]
    if form == "step_a*lin_b":
        return np.c_[(a >= q) * b, (a >= q).astype(float), b, one]
    raise ValueError(form)


FORMS = ("linear", "quadratic", "sat_a+lin_b", "lin_a+sat_b", "sat_a+sat_b", "sat_a*lin_b", "lin_a*sat_b",
         "power_a+lin_b", "lin_a+power_b", "step_a+lin_b", "lin_a+step_b", "bump_a+lin_b", "lin_a+bump_b",
         "step_a*lin_b")
Q_GRID = np.linspace(0.3, 6.0, 20)


def fit_law(X: np.ndarray, y: np.ndarray):
    """Choose a law from a fixed library of forms (each with one shape parameter on a grid) by how well it
    predicts the measurements at the edge of the measured range from the rest (an extrapolation check), then
    refit it on everything. Returns (predict, form, q)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    edge = (X[:, 0] >= np.max(X[:, 0]) - 1e-9) | (X[:, 1] >= np.max(X[:, 1]) - 1e-9)
    inner = ~edge
    best = None
    for form in FORMS:
        for q in (Q_GRID if form not in ("linear", "quadratic") else [1.0]):
            F = _basis(form, X, q)
            if inner.sum() <= F.shape[1]:
                continue
            coef, *_ = np.linalg.lstsq(F[inner], y[inner], rcond=None)
            err = float(np.mean((F[edge] @ coef - y[edge]) ** 2)) + 1e-3 * F.shape[1]
            if best is None or err < best[0]:
                best = (err, form, q)
    _, form, q = best
    coef, *_ = np.linalg.lstsq(_basis(form, X, q), y, rcond=None)
    return (lambda Xq: _basis(form, np.asarray(Xq, dtype=float), q) @ coef), form, q


def measure_grid(view, names: list[str], lo: float, hi: float, n: int = 5, hold: int = 1, settle: bool = False):
    """Set two levers over an n x n grid of their range, `hold` ticks each, and record the reading (the mean of
    the later half of the hold; with `settle`, extrapolated from the geometric approach of the last ticks)."""
    X, y = [], []
    for a in np.linspace(lo, hi, n):
        for b in np.linspace(lo, hi, n):
            vals = []
            for _ in range(hold):
                if view.done:
                    return np.asarray(X), np.asarray(y)
                r = view.step({"kind": "levers", "values": {names[0]: float(a), names[1]: float(b)}})
                nums = r["observation"]["numbers"]
                if nums:
                    vals.append(list(nums.values())[0])
            if not vals:
                continue
            if settle and len(vals) >= 4:
                v = np.asarray(vals[-4:])
                dv = np.diff(v)
                r_ = float(np.clip(dv[1:].sum() / dv[:-1].sum(), 0.0, 0.9)) if abs(dv[:-1].sum()) > 1e-9 else 0.0
                val = float(v[-1] + dv[-1] * r_ / (1 - r_)) if r_ > 0 else float(v[-1])
            else:
                val = float(np.mean(vals[len(vals) // 2:]))
            X.append([a, b])
            y.append(val)
    return np.asarray(X), np.asarray(y)


def discover_and_predict(view, env, n: int = 5, hold: int = 1, settle: bool = False) -> None:
    """Scripted public reference for prediction objectives: measure the reading over a grid of the two settings,
    select a law from the fixed library by extrapolation error, and file its predictions for the held-out
    conditions (public in the objective). Knows no family; only the objective's format."""
    obj = env._config["objective"]
    held = obj["heldout"]
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = max(float(props[m]["minimum"]) for m in names)
    hi = min(float(props[m]["maximum"]) for m in names)
    X, y = measure_grid(view, names, lo, hi, n=n, hold=hold, settle=settle)
    if len(y) >= 8 and not view.done:
        pred, form, q = fit_law(X, y)
        Xh = np.asarray([[c[names[0]], c[names[1]]] for c in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def discover_settled(view, env) -> None:
    """`discover_and_predict` for slow processes: each grid point is held long enough to extrapolate where the
    reading settles."""
    n = 4
    hold = max(6, (view.horizon - 2) // (n * n))
    discover_and_predict(view, env, n=n, hold=hold, settle=True)


def discover_averaged(view, env) -> None:
    """`discover_and_predict` for noisy, confounded readings: each grid point is held several ticks."""
    n = 4
    hold = max(2, (view.horizon - 2) // (n * n))
    discover_and_predict(view, env, n=n, hold=hold)
