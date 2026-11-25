"""Small, real diagnostic environments behind the contract's `EnvAdapter` / `GoalEvaluator` (pure python).

They isolate the decision problems workstream D must solve and verification must check (spec 7.4, 8.4, 9.4):

(a) `ProbeBanditAdapter` — an immediately costly **probe** reveals which of ``n_arms`` arms pays; pulling the paying arm
    earns ``pay_cents`` per decision. A useful experiment with immediate cost that creates the opportunity for later
    actions. With ``noise_tv=True`` a free ``watch`` tool returns pure noise (high surprise, zero information).
(b) `DelayedRewardAdapter` — ``invest`` costs now and pays back ``delay_days`` later through an asynchronous EVENT
    observation attributed (``caused_by``) to the investing action; ``safe`` pays a little immediately; ``wait`` lets
    time pass. Credit must cross several decisions and real elapsed time.
(c) `ChangingMechanismAdapter` — a hidden actuator gain whose sign and magnitude change at a hidden decision; only
    noisy readings are observed. The agent must notice and re-adapt.

Agent side: observations, tool schemas, goals and the SMDP clock (executing never moves time; ``advance_time`` does).
Evaluator side: `DiagnosticEvaluator.evaluate_goal(i)` returns the verified cumulative return (``metrics["return"]``)
and keeps hidden mechanism truth in ``details`` only. No observation ever carries the hidden state except through the
mechanism's genuine consequences (the probe's report, the arms' payouts, the gauge readings).

Event cursors are per-environment sequence numbers (contract decision 6); ``collect_events(since)`` is lossless within
an episode; ``reset`` starts a new episode (new log, cursor 0). ``reset(seeds)`` is deterministic in the seed, so
re-resetting with the same seeds replays the same hidden mechanism — this is how multi-trial meta-RL episodes repeat a
task (`meta.TrialRunner`).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..interfaces import (Action, EvidenceKind, ExecutionResult, GoalSpec, IndependentGoalResult, Observation,
                          ObservationBatch, ObsKind, ToolSchema, new_id)


@dataclass
class _Env:
    seed: int
    rng: random.Random
    t: float = 0.0                      # simulated days
    seq: int = 0
    log: list = field(default_factory=list)        # every observation (seq-ordered)
    inbox: list = field(default_factory=list)      # world events not yet delivered by observe/advance_time
    pending: list = field(default_factory=list)    # (due_t, Observation, reward) scheduled world events
    decisions: int = 0
    horizon: int = 1
    done: bool = False
    ret: float = 0.0                    # verified cumulative return (evaluator side)
    hidden: dict = field(default_factory=dict)     # mechanism truth (evaluator side only)
    goal: Optional[GoalSpec] = None


class DiagnosticAdapter:
    """Common machinery: B independent environments, seq-cursor event logs, SMDP clock, verified ledger."""

    family = "diagnostic"
    money_unit = "cents"

    def __init__(self, B: int, decision_days: float = 1.0 / 24):
        if B < 1:
            raise ValueError("B must be >= 1")
        self.B = B
        self.decision_days = float(decision_days)
        self.envs: list[_Env] = []
        self.evaluator = DiagnosticEvaluator(self)
        self._closed = False

    # ---------------------------------------------------------------- subclass hooks
    def _init_env(self, e: _Env) -> list:          # -> initial observations
        raise NotImplementedError

    def _tools(self, e: _Env) -> list:
        raise NotImplementedError

    def _act(self, e: _Env, a: Action) -> ExecutionResult:
        raise NotImplementedError

    def _oracle_return(self, e: _Env) -> float:
        raise NotImplementedError

    def _on_time(self, e: _Env) -> None:            # world dynamics on clock movement (optional)
        return None

    # ---------------------------------------------------------------- helpers
    def _emit(self, e: _Env, obs: Observation, world_event: bool = False) -> Observation:
        e.seq += 1
        obs.seq = e.seq
        obs.t = e.t
        e.log.append(obs)
        if world_event:
            e.inbox.append(obs)
        return obs

    def _batch(self, per_env: list) -> ObservationBatch:
        return ObservationBatch(env_ids=[f"{self.family}:{i}" for i in range(self.B)], observations=per_env,
                                t=[e.t for e in self.envs], done=[e.done for e in self.envs])

    def _check(self):
        if self._closed:
            raise RuntimeError("adapter closed")
        if not self.envs:
            raise RuntimeError("call reset() first")

    @staticmethod
    def _schema(name, description, props=None, required=None, cost_hint=None) -> ToolSchema:
        return ToolSchema(name, description, {"type": "object", "properties": props or {}, "required": required or []},
                          cost_hint=cost_hint)

    def _fail(self, e: _Env, a: Action, code: str, msg: str) -> ExecutionResult:
        o = self._emit(e, Observation(ObsKind.ERROR, f"{code}: {msg}", source=a.tool, evidence=EvidenceKind.CONTROLLED,
                                      caused_by=a.action_id))
        return ExecutionResult(False, [o], error=code, duration=self.decision_days, started_t=e.t, finished_t=e.t,
                               meta={"error_code": code})

    # ---------------------------------------------------------------- EnvAdapter
    def reset(self, seeds: Sequence[int] | None = None) -> ObservationBatch:
        self._closed = False
        seeds = list(range(self.B)) if seeds is None else [int(s) for s in seeds]
        if len(seeds) != self.B:
            raise ValueError(f"need {self.B} seeds")
        self.envs = []
        first = []
        for s in seeds:
            e = _Env(seed=s, rng=random.Random(f"{self.family}:{s}"))
            obs = self._init_env(e)
            first.append([self._emit(e, o) for o in obs])
            self.envs.append(e)
        return self._batch(first)

    def observe(self) -> ObservationBatch:
        self._check()
        out = []
        for e in self.envs:
            out.append(list(e.inbox))
            e.inbox.clear()
        return self._batch(out)

    def available_tools(self) -> list:
        self._check()
        return [self._tools(e) for e in self.envs]

    def execute(self, actions: Sequence[Optional[Action]]) -> list:
        """[B] ExecutionResult; a ``None`` action skips that environment and yields ``None``."""
        self._check()
        if len(actions) != self.B:
            raise ValueError(f"need {self.B} actions (None = skip)")
        res = []
        for e, a in zip(self.envs, actions):
            if a is None:
                res.append(None)
                continue
            if e.done or e.decisions >= e.horizon:
                res.append(self._fail(e, a, "episode_done", "no decisions left in this episode"))
                continue
            r = self._act(e, a)
            e.decisions += 1
            res.append(r)
        return res

    def collect_events(self, since: Sequence[int]) -> tuple:
        self._check()
        obs, cur = [], []
        for e, s in zip(self.envs, since):
            s = int(s)
            new = [o for o in e.log if o.seq > s]
            obs.append(new)
            cur.append(e.seq)
        return obs, cur

    def advance_time(self, duration: Sequence[float]) -> ObservationBatch:
        """Operator side: move each clock by ``duration[b]`` days, release due world events, then mark episodes whose
        decision budget is exhausted as done. Returns the world events delivered (same as a following observe())."""
        self._check()
        if len(duration) != self.B:
            raise ValueError(f"need {self.B} durations")
        for e, d in zip(self.envs, duration):
            d = float(d)
            if not math.isfinite(d) or d < 0:
                raise ValueError("duration must be a finite non-negative number of days")
            t_end = e.t + d
            due = sorted([p for p in e.pending if p[0] <= t_end], key=lambda p: p[0])
            e.pending = [p for p in e.pending if p[0] > t_end]
            for due_t, o, rew in due:
                e.t = max(e.t, due_t)
                self._emit(e, o, world_event=True)
                e.ret += rew
            e.t = t_end
            self._on_time(e)
            if e.decisions >= e.horizon:
                e.done = True
        return self.observe()

    def goals(self) -> list:
        self._check()
        return [e.goal for e in self.envs]

    def close(self) -> None:
        self._closed = True


class DiagnosticEvaluator:
    """GoalEvaluator for the diagnostic adapters (evaluator side; may read hidden state, which stays in `details`)."""

    def __init__(self, adapter: DiagnosticAdapter):
        self.adapter = adapter

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        e = self.adapter.envs[env_index]
        oracle = self.adapter._oracle_return(e)
        score = 0.0 if oracle <= 0 else max(0.0, min(1.0, e.ret / oracle))
        return IndependentGoalResult(goal_id=e.goal.goal_id, achieved=bool(e.done and e.ret >= 0.9 * oracle),
                                     score=score, metrics={"return": e.ret}, t=e.t, evaluator=self.adapter.family,
                                     details={"hidden": dict(e.hidden), "oracle_return": oracle})


# ================================================================================================ (a) probe bandit

class ProbeBanditAdapter(DiagnosticAdapter):
    """Informative-probe bandit. Hidden: the paying arm g (uniform). Tools: ``probe`` (costs ``probe_cost_cents``;
    reports a signal per arm, 1 for the paying arm, each flipped with prob. ``probe_noise``), ``pull`` {arm: int}
    (pays ``pay_cents`` if arm == g, plus Gaussian noise ``pay_noise * pay_cents``), and with ``noise_tv`` ``watch``
    (free; returns an integer 0..99 independent of everything). Decision budget H per episode (``horizon`` int or an
    inclusive (lo, hi) range drawn per seed); the budget is public (goal meta and a CONSTRAINT observation)."""

    family = "probe_bandit"

    def __init__(self, B: int, n_arms: int = 6, horizon=6, probe_cost_cents: float = 30.0, pay_cents: float = 100.0,
                 probe_noise: float = 0.0, pay_noise: float = 0.0, noise_tv: bool = False,
                 decision_days: float = 1.0 / 24):
        super().__init__(B, decision_days)
        self.n_arms, self.horizon = n_arms, horizon
        self.probe_cost, self.pay = float(probe_cost_cents), float(pay_cents)
        self.probe_noise, self.pay_noise, self.noise_tv = probe_noise, pay_noise, noise_tv

    def _init_env(self, e: _Env) -> list:
        e.hidden["good_arm"] = e.rng.randrange(self.n_arms)
        if isinstance(self.horizon, (tuple, list)):
            e.horizon = e.rng.randint(int(self.horizon[0]), int(self.horizon[1]))
        else:
            e.horizon = int(self.horizon)
        e.goal = GoalSpec(goal_id=new_id("g_"), text=(f"Earn as many cents as possible within {e.horizon} decisions. "
                                                      f"Exactly one of {self.n_arms} arms pays."),
                          metrics={"decisions": e.horizon}, deadline_days=e.horizon * self.decision_days,
                          family=self.family, meta={"decision_budget": e.horizon, "money_unit": "cents"})
        return [Observation(ObsKind.GOAL, e.goal.text, source="operator"),
                Observation(ObsKind.CONSTRAINT, "decision budget", value=float(e.horizon), unit="decisions",
                            entity="budget", source="operator")]

    def _tools(self, e: _Env) -> list:
        tools = [self._schema("probe", f"Run a diagnostic that reports, for every arm, whether it is the paying arm. "
                                       f"Costs {self.probe_cost:g} cents.", cost_hint=self.probe_cost),
                 self._schema("pull", "Pull one arm and receive its payout in cents.",
                              {"arm": {"type": "integer", "minimum": 0, "maximum": self.n_arms - 1}}, ["arm"],
                              cost_hint=0.0)]
        if self.noise_tv:
            tools.append(self._schema("watch", "Watch a channel and report its current reading.", cost_hint=0.0))
        return tools

    def _act(self, e: _Env, a: Action) -> ExecutionResult:
        g = e.hidden["good_arm"]
        if a.tool == "probe":
            obs = []
            for i in range(self.n_arms):
                s = 1.0 if i == g else 0.0
                if self.probe_noise > 0 and e.rng.random() < self.probe_noise:
                    s = 1.0 - s
                obs.append(self._emit(e, Observation(ObsKind.NUMERIC, value=s, unit="signal", entity=f"arm_{i}",
                                                     entity_type="arm", source="probe",
                                                     evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id)))
            e.ret -= self.probe_cost
            return ExecutionResult(True, obs, cost=self.probe_cost, duration=self.decision_days, started_t=e.t,
                                   finished_t=e.t)
        if a.tool == "pull":
            arm = a.args.get("arm")
            if isinstance(arm, bool) or not isinstance(arm, int) or not 0 <= arm < self.n_arms:
                return self._fail(e, a, "invalid_argument", f"arm must be an integer in [0, {self.n_arms - 1}]")
            payout = self.pay if arm == g else 0.0
            if self.pay_noise > 0:
                payout += e.rng.gauss(0.0, self.pay_noise * self.pay)
            o = self._emit(e, Observation(ObsKind.NUMERIC, value=payout, unit="cents", entity=f"arm_{arm}",
                                          entity_type="arm", source="pull", evidence=EvidenceKind.CONTROLLED,
                                          caused_by=a.action_id))
            e.ret += payout
            return ExecutionResult(True, [o], duration=self.decision_days, started_t=e.t, finished_t=e.t)
        if a.tool == "watch" and self.noise_tv:
            o = self._emit(e, Observation(ObsKind.NUMERIC, value=float(e.rng.randrange(100)), unit="count",
                                          entity="channel", source="watch", evidence=EvidenceKind.CONTROLLED,
                                          caused_by=a.action_id))
            return ExecutionResult(True, [o], duration=self.decision_days, started_t=e.t, finished_t=e.t)
        return self._fail(e, a, "unknown_tool", f"no tool named {a.tool!r}")

    def _oracle_return(self, e: _Env) -> float:
        return e.horizon * self.pay

    def bayes_optimal_value(self, horizon: int) -> float:
        """Expected return of the Bayes-optimal policy (noise-free probe and payouts) by dynamic programming over
        (decisions left r, candidate arms m): probe -> -cost + (r-1) pay; pull a candidate ->
        (1/m)(pay + (r-1) pay) + ((m-1)/m) V(r-1, m-1); V(r, 1) = r pay."""
        if self.probe_noise or self.pay_noise:
            raise ValueError("closed form only for the noise-free bandit")
        memo: dict = {}

        def V(r, m):
            if r <= 0:
                return 0.0
            if m == 1:
                return r * self.pay
            if (r, m) in memo:
                return memo[(r, m)]
            probe = -self.probe_cost + (r - 1) * self.pay
            pull = (self.pay + (r - 1) * self.pay) / m + (m - 1) / m * V(r - 1, m - 1)
            memo[(r, m)] = max(probe, pull)
            return memo[(r, m)]
        return V(horizon, self.n_arms)

    def elimination_value(self, horizon: int) -> float:
        """Expected return of never probing and pulling untried arms until the paying one is found (greedy optimum)."""
        n, H = self.n_arms, horizon
        tot = 0.0
        for j in range(1, n + 1):           # paying arm found at pull j (prob 1/n); the last one is known
            if j <= H:
                tot += self.pay * (H - j + 1) / n
        return tot


# ================================================================================================ (b) delayed reward

class DelayedRewardAdapter(DiagnosticAdapter):
    """Delayed reward. Tools: ``invest`` (pays ``invest_cost_cents`` now; ``invest_return_cents`` arrive after
    ``delay_days`` as an EVENT observation with ``caused_by`` = the investing action), ``safe`` (``safe_cents`` now),
    ``wait`` {days: number in [0, max_wait_days]} (only lets time pass; its duration is the chosen days).
    Decisions per episode: ``horizon``; a payoff counts only if it arrives before the episode ends (the episode ends at
    the clock movement after the last decision)."""

    family = "delayed_reward"

    def __init__(self, B: int, horizon: int = 8, delay_days: float = 1.0, invest_cost_cents: float = 50.0,
                 invest_return_cents: float = 150.0, safe_cents: float = 10.0, decision_days: float = 0.25,
                 max_wait_days: float = 2.0):
        super().__init__(B, decision_days)
        self.horizon, self.delay = int(horizon), float(delay_days)
        self.cost, self.back, self.safe = float(invest_cost_cents), float(invest_return_cents), float(safe_cents)
        self.max_wait = float(max_wait_days)

    def _init_env(self, e: _Env) -> list:
        e.horizon = self.horizon
        e.hidden["delay_days"] = self.delay
        e.goal = GoalSpec(goal_id=new_id("g_"), text=f"Maximise cents earned within {self.horizon} decisions.",
                          metrics={"decisions": self.horizon}, family=self.family,
                          meta={"decision_budget": self.horizon, "money_unit": "cents"})
        return [Observation(ObsKind.GOAL, e.goal.text, source="operator"),
                Observation(ObsKind.CONSTRAINT, "decision budget", value=float(self.horizon), unit="decisions",
                            entity="budget", source="operator")]

    def _tools(self, e: _Env) -> list:
        return [self._schema("invest", f"Invest {self.cost:g} cents in a project that may pay back later.",
                             cost_hint=self.cost),
                self._schema("safe", "Do a small safe job that pays immediately.", cost_hint=0.0),
                self._schema("wait", "Let time pass.", {"days": {"type": "number", "minimum": 0,
                                                                  "maximum": self.max_wait}}, ["days"])]

    def _act(self, e: _Env, a: Action) -> ExecutionResult:
        if a.tool == "invest":
            e.ret -= self.cost
            o = self._emit(e, Observation(ObsKind.TRANSACTION, "investment placed", value=-self.cost, unit="cents",
                                          entity="project", source="invest", evidence=EvidenceKind.CONTROLLED,
                                          caused_by=a.action_id))
            pay = Observation(ObsKind.EVENT, "project paid back", value=self.back, unit="cents", entity="project",
                              source="world", evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id)
            e.pending.append((e.t + self.delay, pay, self.back))
            return ExecutionResult(True, [o], cost=self.cost, duration=self.decision_days, started_t=e.t,
                                   finished_t=e.t)
        if a.tool == "safe":
            e.ret += self.safe
            o = self._emit(e, Observation(ObsKind.TRANSACTION, "job paid", value=self.safe, unit="cents", entity="job",
                                          source="safe", evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id))
            return ExecutionResult(True, [o], duration=self.decision_days, started_t=e.t, finished_t=e.t)
        if a.tool == "wait":
            d = a.args.get("days")
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 <= d <= self.max_wait:
                return self._fail(e, a, "invalid_argument", f"days must be a number in [0, {self.max_wait:g}]")
            return ExecutionResult(True, [], duration=float(d), started_t=e.t, finished_t=e.t)
        return self._fail(e, a, "unknown_tool", f"no tool named {a.tool!r}")

    def _oracle_return(self, e: _Env) -> float:
        n_inv = max(0, min(self.horizon, math.ceil((self.horizon * self.decision_days - self.delay)
                                                   / self.decision_days + 1e-9)))
        return n_inv * (self.back - self.cost) + (self.horizon - n_inv) * self.safe


# ================================================================================================ (c) changing mechanism

class ChangingMechanismAdapter(DiagnosticAdapter):
    """Changing mechanism. Hidden actuator gain k (sign +-1 times magnitude in [0.5, 1.5]); at a hidden decision
    c ~ U[H/3, 2H/3] the gain is redrawn with the opposite sign. Tool ``set_level`` {level in [-1, 1]} yields a gauge
    reading y = k * level + N(0, noise^2) (NUMERIC, unit "units"); verified reward per decision is -(y - target)^2
    with the public target from the goal. With ``announce`` an EVENT observation reports maintenance at the change."""

    family = "changing_mechanism"

    def __init__(self, B: int, horizon: int = 30, noise: float = 0.1, target: float = 0.8, announce: bool = False,
                 decision_days: float = 1.0 / 24):
        super().__init__(B, decision_days)
        self.horizon, self.noise, self.target, self.announce = int(horizon), float(noise), float(target), announce

    def _init_env(self, e: _Env) -> list:
        e.horizon = self.horizon
        sign = 1.0 if e.rng.random() < 0.5 else -1.0
        e.hidden.update(gain=sign * e.rng.uniform(0.5, 1.5),
                        change_at=e.rng.randint(self.horizon // 3, (2 * self.horizon) // 3))
        e.hidden["gain_after"] = -math.copysign(1.0, e.hidden["gain"]) * e.rng.uniform(0.5, 1.5)
        e.goal = GoalSpec(goal_id=new_id("g_"), text=f"Hold the gauge reading at {self.target:g}.",
                          metrics={"target_reading": self.target}, family=self.family,
                          meta={"decision_budget": self.horizon})
        return [Observation(ObsKind.GOAL, e.goal.text, source="operator"),
                Observation(ObsKind.CONSTRAINT, "target reading", value=self.target, unit="units", entity="gauge",
                            source="operator")]

    def _gain(self, e: _Env) -> float:
        return e.hidden["gain"] if e.decisions < e.hidden["change_at"] else e.hidden["gain_after"]

    def _tools(self, e: _Env) -> list:
        return [self._schema("set_level", "Set the actuator level; the gauge reading responds.",
                             {"level": {"type": "number", "minimum": -1.0, "maximum": 1.0}}, ["level"])]

    def _act(self, e: _Env, a: Action) -> ExecutionResult:
        if a.tool != "set_level":
            return self._fail(e, a, "unknown_tool", f"no tool named {a.tool!r}")
        lv = a.args.get("level")
        if isinstance(lv, bool) or not isinstance(lv, (int, float)) or not -1.0 <= lv <= 1.0:
            return self._fail(e, a, "invalid_argument", "level must be a number in [-1, 1]")
        if self.announce and e.decisions == e.hidden["change_at"]:
            self._emit(e, Observation(ObsKind.EVENT, "maintenance performed on the actuator", entity="actuator",
                                      source="world"), world_event=True)
        y = self._gain(e) * float(lv) + e.rng.gauss(0.0, self.noise)
        e.ret += -(y - self.target) ** 2
        o = self._emit(e, Observation(ObsKind.NUMERIC, value=y, unit="units", entity="gauge", entity_type="gauge",
                                      source="set_level", evidence=EvidenceKind.CONTROLLED, caused_by=a.action_id,
                                      meta={"level": float(lv)}))
        return ExecutionResult(True, [o], duration=self.decision_days, started_t=e.t, finished_t=e.t)

    def _oracle_return(self, e: _Env) -> float:
        return 0.0          # the best expected return is -H * noise^2 (scores are not normalised for this task)
