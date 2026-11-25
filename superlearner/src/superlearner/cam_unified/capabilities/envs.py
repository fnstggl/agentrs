"""Small diagnostic environments for the integrated capability tests (workstream Y).

Each is a pure-python `EnvAdapter` over B independent rows (episodes) with a fixed number of decisions per episode
and a hidden per-episode mechanism drawn from the row's seed. The agent sees only records: text / numeric
observations, tool schemas (no string arguments, so decisions are made by the action generator's tool pointer and
typed heads, never by free text), the goal text and scalar verified returns (through `ScoreEvaluator`, whose score is
the episode's cumulative return computed from the hidden state). Observation texts are real natural language and go
through A's tokenizer and the pretrained semantic encoder like any other observation.

These are diagnostic tasks, not OpenWorld: they isolate one capability each (spec Parts 4, 5, 7, 8, 10) so that a
failure of the full stack can be attributed. Hidden mechanism fields (`hidden`) are evaluator-side only.

* `ChainEnv` (A03): follow arrows k times through a random permutation of colours; k varies (1..max_hops), so some
  decisions need more latent composition steps than others.
* `MechanismEnv` (A05/A06): a slot machine whose payout distribution is one of several competing mechanisms (plus
  optional unfamiliar mechanisms that never occur in training: misspecification).
* `ProbeEnv` (A08/A09/C05, spec 7.4 / 8.4): a costly probe reveals whether a risky high-value action pays; without the
  probe the safe action is better, with it the agent can exploit only when it pays.
* `DelayedCueEnv` (E03): a cue colour shown at the first decision must be reported many decisions later, after
  interfering observations.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from ..interfaces import (ExecutionResult, GoalSpec, IndependentGoalResult, Observation, ObservationBatch, ObsKind,
                          ToolSchema)

NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}
COLOURS = ("red", "green", "blue", "yellow", "purple", "orange")


def tool(name: str, description: str) -> ToolSchema:
    return ToolSchema(name, description, dict(NO_ARGS), returns="a short confirmation", duration_hint=0.0)


class DiagnosticEnv:
    """Base EnvAdapter: B rows, `horizon` decisions per episode, hidden state per row from its seed.

    Subclasses implement `_new(rng, b) -> hidden dict`, `_tools(h) -> [ToolSchema]`, `_obs0(h, b) -> [Observation]`,
    `_goal(h, b) -> GoalSpec` and `_act(h, action, rng, b) -> (ok, [Observation], reward, error)`. The clock moves
    only in `advance_time` (days); results carry duration 0 (the harness adds the think time)."""

    name = "diagnostic"
    horizon = 1

    def __init__(self, B: int, seed: int = 0):
        self.B = B
        self.seed = seed
        self.env_ids = [f"{self.name}{b}" for b in range(B)]
        self.reset()

    # ---- hooks
    def _new(self, rng: random.Random, b: int) -> dict:
        raise NotImplementedError

    def _tools(self, h: dict) -> list:
        raise NotImplementedError

    def _obs0(self, h: dict, b: int) -> list:
        return []

    def _goal(self, h: dict, b: int) -> GoalSpec:
        return GoalSpec(f"{self.name}_goal", "")

    def _act(self, h: dict, action, rng: random.Random, b: int) -> tuple:
        raise NotImplementedError

    # ---- EnvAdapter
    def reset(self, seeds: Optional[Sequence[int]] = None) -> ObservationBatch:
        seeds = list(seeds) if seeds is not None else [self.seed * 100_003 + b for b in range(self.B)]
        self.seeds = seeds
        self.rng = [random.Random(int(s)) for s in seeds]
        self.t = [0.0] * self.B
        self.seq = [0] * self.B
        self.k = [0] * self.B
        self.returns = [0.0] * self.B
        self.hidden = [self._new(self.rng[b], b) for b in range(self.B)]
        self._pending = [[] for _ in range(self.B)]
        obs = [[self._emit(b, o) for o in self._obs0(self.hidden[b], b)] for b in range(self.B)]
        return ObservationBatch(list(self.env_ids), obs, list(self.t), [False] * self.B)

    def _emit(self, b: int, o: Observation) -> Observation:
        self.seq[b] += 1
        o.seq = self.seq[b]
        o.t = self.t[b]
        return o

    def cursors(self) -> list:
        return list(self.seq)

    def done(self) -> list:
        return [self.k[b] >= self.horizon for b in range(self.B)]

    def observe(self) -> ObservationBatch:
        return ObservationBatch(list(self.env_ids), [[] for _ in range(self.B)], list(self.t), self.done())

    def available_tools(self) -> list:
        return [list(self._tools(self.hidden[b])) for b in range(self.B)]

    def goals(self) -> list:
        return [self._goal(self.hidden[b], b) for b in range(self.B)]

    def execute(self, actions) -> list:
        out = []
        for b, a in enumerate(actions):
            if a is None or self.k[b] >= self.horizon:
                out.append(None)
                continue
            names = {t.name for t in self._tools(self.hidden[b])}
            if a.tool not in names:
                ok, obs, r, err = False, [Observation(ObsKind.ERROR, f"unknown tool {a.tool}", source=a.tool)], 0.0, \
                    "unknown_tool"
            else:
                ok, obs, r, err = self._act(self.hidden[b], a, self.rng[b], b)
            self.k[b] += 1
            self.returns[b] += float(r)
            obs = [self._emit(b, o) for o in obs]
            out.append(ExecutionResult(ok=ok, observations=obs, error=err, cost=0.0, duration=0.0,
                                       started_t=self.t[b], finished_t=self.t[b],
                                       meta={} if err is None else {"error_code": err}))
        return out

    def collect_events(self, since: Sequence[int]) -> tuple:
        return [[] for _ in range(self.B)], [max(int(s), self.seq[b]) for b, s in enumerate(since)]

    def advance_time(self, duration: Sequence[float]) -> ObservationBatch:
        for b, d in enumerate(duration):
            self.t[b] += float(d)
        return ObservationBatch(list(self.env_ids), [[] for _ in range(self.B)], list(self.t), self.done())

    def close(self) -> None:
        pass


class ScoreEvaluator:
    """GoalEvaluator: score = the row's cumulative return in the current episode (from the hidden mechanism); the
    harness turns score differences into verified returns. `details` (the hidden mechanism) never reaches the agent."""

    def __init__(self, env: DiagnosticEnv):
        self.env = env

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        e = self.env
        return IndependentGoalResult(e.env_ids[env_index] + ":goal", achieved=False, score=e.returns[env_index],
                                     metrics={"verified_return": e.returns[env_index]}, t=e.t[env_index],
                                     evaluator=type(e).__name__, details=dict(e.hidden[env_index]))


# ============================================================================ A03: k-hop composition


class ChainEnv(DiagnosticEnv):
    """One decision per episode. A random permutation f of `n_colours` colours is shown as arrows ("Arrow: red points
    to blue."); the goal names a start colour and a hop count k in 1..max_hops; pressing the button of f^k(start)
    returns 1, anything else 0. The answer is a function of observable inputs only; harder rows need k sequential
    lookups."""

    name = "chain"
    horizon = 1
    HOP_WORDS = {1: "once", 2: "twice", 3: "three times", 4: "four times"}

    def __init__(self, B: int, seed: int = 0, n_colours: int = 4, hops: Sequence[int] = (1, 2), derangement=True):
        self.colours = COLOURS[:n_colours]
        self.hops = tuple(hops)
        self.derangement = derangement
        super().__init__(B, seed)

    def _new(self, rng, b):
        cs = list(self.colours)
        while True:
            perm = cs[:]
            rng.shuffle(perm)
            if not self.derangement or all(a != p for a, p in zip(cs, perm)):
                break
        f = dict(zip(cs, perm))
        start = rng.choice(cs)
        k = rng.choice(self.hops)
        x = start
        for _ in range(k):
            x = f[x]
        order = cs[:]
        rng.shuffle(order)
        return {"f": f, "start": start, "k": k, "answer": x, "order": order}

    def _tools(self, h):
        return [tool(f"press_{c}", f"Press the {c} button.") for c in self.colours]

    def _obs0(self, h, b):
        return [Observation(ObsKind.TEXT, f"Arrow: {c} points to {h['f'][c]}.", source="map") for c in h["order"]]

    def _goal(self, h, b):
        return GoalSpec("chain", f"Start at {h['start']}. Follow the arrows {self.HOP_WORDS[h['k']]}. "
                                 f"Press the button of the colour you reach.")

    def _act(self, h, a, rng, b):
        c = a.tool[len("press_"):]
        r = 1.0 if c == h["answer"] else 0.0
        return True, [Observation(ObsKind.TEXT, "Correct." if r else "Wrong.", source="judge")], r, None


# ============================================================================ A05/A06: competing mechanisms


class MechanismEnv(DiagnosticEnv):
    """A slot machine pulled `horizon` times. The episode's hidden mechanism m (uniform over `mechanisms`) fixes the
    payout distribution N(mu_m, sigma_m^2); every pull pays a verified return and shows the machine's cumulative payout
    counter (a NUMERIC observation, so the outcome tracker also sees the payout as a counter delta). `mechanisms` maps a
    name to (mu, sigma); `fixed` forces one mechanism for every row (evaluation / misspecification), which may be one
    of `unfamiliar` (mechanisms never drawn in training)."""

    name = "slot"

    def __init__(self, B: int, seed: int = 0, horizon: int = 8, mechanisms: Optional[dict] = None,
                 fixed: Optional[str] = None, unfamiliar: Optional[dict] = None):
        self.horizon = horizon
        self.mechanisms = dict(mechanisms or {"low": (-1.0, 0.25), "high": (1.0, 0.25)})
        self.fixed = fixed
        self.all_mechanisms = {**self.mechanisms, **(unfamiliar or {})}
        super().__init__(B, seed)

    def _new(self, rng, b):
        m = self.fixed if self.fixed is not None else rng.choice(sorted(self.mechanisms))
        mu, sd = self.all_mechanisms[m]
        return {"mechanism": m, "mu": mu, "sigma": sd, "total": 0.0}

    def _tools(self, h):
        return [tool("pull_lever", "Pull the lever of the slot machine and collect the payout.")]

    def _obs0(self, h, b):
        return [Observation(ObsKind.TEXT, "A slot machine. Each pull pays out an amount that depends on how the "
                                          "machine is set up; the setup is unknown.", source="brief"),
                Observation(ObsKind.NUMERIC, value=0.0, unit="coins", entity="machine", entity_type="slot machine",
                            source="counter", meta={"metric": "total payout"})]

    def _goal(self, h, b):
        return GoalSpec("slot", "Collect payouts from the slot machine.")

    def _act(self, h, a, rng, b):
        y = rng.gauss(h["mu"], h["sigma"])
        h["total"] += y
        return True, [Observation(ObsKind.NUMERIC, value=round(h["total"], 4), unit="coins", entity="machine",
                                  entity_type="slot machine", source="counter",
                                  meta={"metric": "total payout"})], y, None


# ============================================================================ A08/A09/C05: costly probe


class ProbeEnv(DiagnosticEnv):
    """Spec 7.4 / 8.4 diagnostic. Hidden state good with probability `p_good`. Tools: `run_test` (costs `probe_cost`
    now and reports the state as a numeric test signal, +1 good / -1 bad, plus a sentence), `launch` (+`win` if good,
    -`loss` if bad), `play_safe` (+`safe`). `horizon` decisions. The test signal is a NUMERIC observation that is
    already present (value 0, "not tested") at the start, so it is an outcome the world model and the beliefs
    predict for every action (the run_test outcome is a +-1 change of it). Without information the risky launch is
    worth p*win - (1-p)*loss < safe; the probe costs now but lets every later decision launch only when good.
    `informative=False` makes the signal a coin flip independent of the state (the probe is then worthless)."""

    name = "probe"

    def __init__(self, B: int, seed: int = 0, horizon: int = 2, p_good: float = 0.5, probe_cost: float = 1.0,
                 win: float = 10.0, loss: float = 10.0, safe: float = 1.0, noise: float = 0.0,
                 informative: bool = True):
        self.horizon = horizon
        self.p_good, self.probe_cost, self.win, self.loss, self.safe, self.noise = p_good, probe_cost, win, loss, safe, \
            noise
        self.informative = informative
        super().__init__(B, seed)

    def _new(self, rng, b):
        good = rng.random() < self.p_good
        return {"good": good, "signal": (good if self.informative else rng.random() < 0.5)}

    def _tools(self, h):
        return [tool("run_test", "Run a costly test that shows whether launching will succeed."),
                tool("launch", "Launch the product: a large gain if conditions are right, a large loss otherwise."),
                tool("play_safe", "Take the small, certain gain.")]

    def _signal(self, v: float) -> Observation:
        return Observation(ObsKind.NUMERIC, value=v, unit="signal", entity="market", entity_type="market",
                           source="test", meta={"metric": "test signal"})

    def _obs0(self, h, b):
        return [Observation(ObsKind.TEXT, "Launching pays off only under the right conditions, which are not known "
                                          "yet. A test can reveal them.", source="brief"), self._signal(0.0)]

    def _goal(self, h, b):
        return GoalSpec("probe", "Earn as much as possible over the next few decisions.")

    def value_bounds(self) -> dict:
        """Expected returns of the reference strategies (evaluator-side; for reporting)."""
        p, H = self.p_good, self.horizon
        best_after = (p * self.win + (1 - p) * self.safe) if self.informative else \
            max(self.safe, p * self.win - (1 - p) * self.loss)
        return {"always_safe": H * self.safe, "probe_then_best": -self.probe_cost + (H - 1) * best_after,
                "always_launch": H * (p * self.win - (1 - p) * self.loss)}

    def _act(self, h, a, rng, b):
        eps = rng.gauss(0.0, self.noise) if self.noise else 0.0
        if a.tool == "run_test":
            pos = h["signal"]
            txt = "Test result: conditions are right, launching will succeed." if pos else \
                "Test result: conditions are wrong, launching will fail."
            return True, [Observation(ObsKind.TEXT, txt, source="test"), self._signal(1.0 if pos else -1.0)], \
                -self.probe_cost, None
        if a.tool == "launch":
            r = (self.win if h["good"] else -self.loss) + eps
            return True, [Observation(ObsKind.TEXT, "Launch succeeded." if h["good"] else "Launch failed.",
                                      source="market")], r, None
        return True, [Observation(ObsKind.TEXT, "Small gain collected.", source="market")], self.safe + eps, None


# ============================================================================ E03: delayed cue


class DelayedCueEnv(DiagnosticEnv):
    """Decision 0 shows a cue colour; the next `delay` decisions show interfering observations (each a random colour
    statement unrelated to the cue) and the agent may only `wait`; at the last decision it must press the cue's
    button (reward 1, else 0). Chance = 1 / n_colours."""

    name = "cue"

    def __init__(self, B: int, seed: int = 0, delay: int = 8, n_colours: int = 4, distractors: int = 2):
        self.delay = delay
        self.horizon = delay + 2
        self.colours = COLOURS[:n_colours]
        self.distractors = distractors
        super().__init__(B, seed)

    def _new(self, rng, b):
        return {"cue": rng.choice(self.colours)}

    def _phase(self, b: int) -> str:
        k = self.k[b]
        return "cue" if k == 0 else ("answer" if k >= self.horizon - 1 else "wait")

    def _tools(self, h):
        return [tool("wait_turn", "Do nothing this turn.")] + \
            [tool(f"press_{c}", f"Press the {c} button.") for c in self.colours]

    def _obs0(self, h, b):
        return [Observation(ObsKind.TEXT, f"Remember this: the vault colour is {h['cue']}.", source="cue")]

    def _goal(self, h, b):
        return GoalSpec("cue", "When asked, press the button of the vault colour you were told at the start.")

    def _act(self, h, a, rng, b):
        k = self.k[b]
        last = k >= self.horizon - 1
        if not last:
            r = 0.0 if a.tool == "wait_turn" else -0.1
            obs = [Observation(ObsKind.TEXT, f"Noise: someone mentions the colour {rng.choice(self.colours)}.",
                               source="chatter") for _ in range(self.distractors)]
            if k + 1 >= self.horizon - 1:
                obs.append(Observation(ObsKind.TEXT, "Question: which colour is the vault? Press its button now.",
                                       source="question"))
            return True, obs, r, None
        r = 1.0 if a.tool == f"press_{h['cue']}" else 0.0
        return True, [Observation(ObsKind.TEXT, "Correct." if r else "Wrong.", source="judge")], r, None
