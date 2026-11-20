"""Goal contracts: outcomes to achieve in a world, with independent verification. Never strategies.

A `GoalContract` specifies
  * the desired outcome as a typed predicate over authoritative state (goals/metrics.py), optionally *sustained*
    (true at k consecutive daily checks),
  * initial conditions (world family, seed, the agent's endowment / starting organisation),
  * budgets and horizon (simulated time; optionally action and compute budgets),
  * hard constraints (integrity checks that void success, e.g. related-party revenue is not counted at all
    because the metric excludes it; breaching contracts or operating with sock-puppet organisations is reported),
  * a progress measure used for shaping, which must be a verified metric (never an agent claim).

It does not say how. The verifier (`GoalVerifier`) reads state through metric functions; the agent never sees
evaluator-only metrics or hidden parameters.

Reward contract (`RewardContract`): r_t = λ·(Φ(s_t) − Φ(s_{t−1})) + R_success·[first success] where Φ is the
progress potential in [0, 1]. Potential-based shaping is policy-invariant and cannot be farmed by cycles: returning
to a state returns the potential. There is no reward for named behaviours (inventing, negotiating, planning).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.canonical import digest
from .metrics import METRICS, check_predicate, evaluate_metric, evaluate_predicate, related_parties

DAY = 86400


@dataclass
class GoalContract:
    family: str
    description: str
    world: dict                     # {"family": ..., "seed": ..., "overrides": {...}}
    principal: str                  # role binding: the actor's handle (resolved at instantiation)
    success: dict                   # predicate (may reference "$self" / "$org" / "$host" placeholders)
    progress: dict                  # {"metric", "args", "target"} -> potential = clip(metric/target, 0, 1)
    horizon_s: int
    sustain_checks: int = 1         # consecutive daily checks the predicate must hold
    endowment: int = 1_000_000
    budget: dict = field(default_factory=dict)   # e.g. {"max_actions": 5000}
    split: str = "train"
    difficulty: float = 0.5
    params: dict = field(default_factory=dict)
    tags: tuple = ()

    @property
    def id(self) -> str:
        return "g-" + digest([self.family, self.world, self.success, self.horizon_s, self.endowment, self.sustain_checks,
                              self.params], 6)

    def to_dict(self) -> dict:
        return {"id": self.id, "family": self.family, "description": self.description, "world": self.world,
                "principal": self.principal, "success": self.success, "progress": self.progress,
                "horizon_s": self.horizon_s, "sustain_checks": self.sustain_checks, "endowment": self.endowment,
                "budget": self.budget, "split": self.split, "difficulty": self.difficulty, "params": self.params,
                "tags": list(self.tags)}

    def validate(self) -> list[str]:
        errs = check_predicate(_bind(self.success, {"$self": "p:x", "$org": "o:x", "$host": "h"}))
        if self.progress.get("metric") not in METRICS:
            errs.append(f"unknown progress metric {self.progress.get('metric')}")
        if self.horizon_s < DAY:
            errs.append("horizon below a day")
        return errs


def _bind(x: Any, env: dict) -> Any:
    if isinstance(x, str):
        return env.get(x, x)
    if isinstance(x, dict):
        return {k: _bind(v, env) for k, v in x.items()}
    if isinstance(x, list):
        return [_bind(v, env) for v in x]
    return x


def bindings(world, actor: str) -> dict:
    """Resolve $self/$org/$host for an actor: its first owned organisation and that organisation's best host."""
    st = world.state
    env = {"$self": actor}
    orgs = sorted(m["org"] for m in st.find("members", "person", actor) if m["role"] == "owner")
    env["$org"] = orgs[0] if orgs else "o:none"
    best, best_ok = "none", -1.0
    if "deployments" in st.tables:
        for o in orgs:
            for did in st.find_ids("deployments", "owner", o):
                d = st.tables["deployments"][did]
                okn = METRICS["service_requests_ok"](world, d["host"], 7 * DAY)
                if okn > best_ok:
                    best, best_ok = d["host"], okn
    env["$host"] = best
    env["$orgs"] = orgs
    return env


class GoalVerifier:
    """Independent verification of one goal for one actor. Holds evaluator-side memory (sustain counters) outside
    the world, so the agent cannot read or alter it."""

    def __init__(self, goal: GoalContract, actor: str):
        self.goal = goal
        self.actor = actor
        self.start_t: int | None = None
        self.checks: list[tuple[int, bool]] = []
        self.streak = 0
        self.success_t: int | None = None
        self.initially_true = False

    def begin(self, world) -> None:
        self.start_t = world.now
        self.initially_true = self.predicate(world)

    def predicate(self, world) -> bool:
        best = False
        env = bindings(world, self.actor)
        for org in env["$orgs"] or ["o:none"]:  # success may be achieved through any organisation the actor owns
            e = dict(env, **{"$org": org})
            try:
                if evaluate_predicate(world, _bind(self.goal.success, e)):
                    best = True
                    break
            except Exception:  # noqa: BLE001 - missing entities -> predicate false
                continue
        return best

    def progress(self, world) -> float:
        env = bindings(world, self.actor)
        p = self.goal.progress
        vals = []
        for org in env["$orgs"] or ["o:none"]:
            try:
                vals.append(evaluate_metric(world, p["metric"], _bind(p.get("args", {}), dict(env, **{"$org": org}))))
            except Exception:  # noqa: BLE001
                vals.append(0.0)
        v = max(vals) if vals else 0.0
        if p.get("baseline") is not None:
            v = v - p["baseline"]
        return float(np.clip(v / max(1e-9, p["target"]), 0.0, 1.0))

    def daily_check(self, world) -> None:
        if self.success_t is not None:
            return
        okv = self.predicate(world)
        self.checks.append((world.now, okv))
        self.streak = self.streak + 1 if okv else 0
        if self.streak >= self.goal.sustain_checks and not self.initially_true:
            self.success_t = world.now

    def status(self, world) -> dict:
        elapsed = world.now - (self.start_t or 0)
        return {"success": self.success_t is not None, "failed": self.success_t is None and elapsed >= self.goal.horizon_s,
                "progress": self.progress(world), "streak": self.streak, "elapsed_s": elapsed,
                "success_t": self.success_t, "initially_true": self.initially_true}


class RewardContract:
    """Gym/PettingZoo objective protocol: begin / step_reward / status. Rewards come only from verified state."""

    def __init__(self, goal: GoalContract, shaping: float = 1.0, success_bonus: float = 1.0, check_every_s: int = DAY):
        self.goal = goal
        self.shaping = shaping
        self.success_bonus = success_bonus
        self.check_every_s = check_every_s
        self.v: dict[str, GoalVerifier] = {}
        self.phi: dict[str, float] = {}
        self.next_check: dict[str, int] = {}
        self.paid: set[str] = set()

    def begin(self, world, actor):
        v = GoalVerifier(self.goal, actor)
        v.begin(world)
        self.v[actor] = v
        self.phi[actor] = v.progress(world)
        self.next_check[actor] = world.now + self.check_every_s

    def step_reward(self, world, actor) -> float:
        v = self.v[actor]
        while world.now >= self.next_check[actor]:
            v.daily_check(world)
            self.next_check[actor] += self.check_every_s
        phi = v.progress(world)
        r = self.shaping * (phi - self.phi[actor])
        self.phi[actor] = phi
        if v.success_t is not None and actor not in self.paid:
            self.paid.add(actor)
            r += self.success_bonus
        return float(r)

    def status(self, world, actor) -> dict:
        return self.v[actor].status(world)


# ============================================================================ goal families (procedural)
def _revenue_goal(rng, world, split, level):
    target = int([100_000, 1_000_000, 3_000_000, 10_000_000][level])      # cents per 30 days: $1k ... $100k
    sustain = [1, 3, 7, 14][level]
    horizon = [21, 45, 90, 180][level] * DAY
    return GoalContract("revenue", f"Earn at least ${target // 100:,} of independent revenue over a 30-day window, "
                        f"held for {sustain} consecutive daily checks, within {horizon // DAY} days.",
                        world, "$self", {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": 30 * DAY},
                                         "op": ">=", "value": target},
                        {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": 30 * DAY},
                         "target": target}, horizon, sustain, endowment=int(rng.choice([500_000, 1_000_000, 3_000_000])),
                        split=split, difficulty=level / 3, params={"target": target, "level": level})


def _recruit_goal(rng, world, split, level):
    n = [1, 3, 6, 10][level]
    tenure = [3, 7, 14, 30][level] * DAY
    horizon = [14, 30, 60, 120][level] * DAY
    return GoalContract("recruit", f"Have {n} independent people (not agents) as members of your organisation, each "
                        f"for at least {tenure // DAY} days, with no unpaid employment contracts.",
                        world, "$self", {"metric": "independent_members", "args": {"org": "$org", "min_tenure_s": tenure},
                                         "op": ">=", "value": n},
                        {"metric": "independent_members", "args": {"org": "$org", "min_tenure_s": 0}, "target": n},
                        horizon, 1, endowment=int(rng.choice([2_000_000, 5_000_000, 10_000_000])), split=split,
                        difficulty=level / 3, params={"n": n, "level": level})


def _service_goal(rng, world, split, level):
    k = [500, 5_000, 30_000, 200_000][level]
    horizon = [10, 21, 45, 90][level] * DAY
    return GoalContract("service", f"Operate a service that successfully serves at least {k:,} requests in a 7-day "
                        f"window with availability >= 0.95.",
                        world, "$self", {"all": [{"metric": "service_requests_ok", "args": {"host": "$host", "window_s": 7 * DAY},
                                                  "op": ">=", "value": k},
                                                 {"metric": "service_availability", "args": {"host": "$host", "window_s": 7 * DAY},
                                                  "op": ">=", "value": 0.95}]},
                        {"metric": "service_requests_ok", "args": {"host": "$host", "window_s": 7 * DAY}, "target": k},
                        horizon, 1, endowment=int(rng.choice([300_000, 1_000_000])), split=split, difficulty=level / 3,
                        params={"k": k, "level": level})


def _solvency_goal(rng, world, split, level):
    days = [7, 21, 45, 90][level]
    return GoalContract("solvency", f"Stay solvent for {days} days while your organisation earns independent revenue "
                        f"every week (at least $50 per 7 days at every daily check).",
                        world, "$self", {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": 7 * DAY},
                                         "op": ">=", "value": 5_000},
                        {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": 7 * DAY}, "target": 5_000},
                        (days + 14) * DAY, days, endowment=int(rng.choice([200_000, 500_000])), split=split,
                        difficulty=level / 3, params={"days": days, "level": level})


GOAL_FAMILIES = {"revenue": _revenue_goal, "recruit": _recruit_goal, "service": _service_goal,
                 "solvency": _solvency_goal}
HELDOUT_GOAL_FAMILIES = {"outperform", "lab_improvement"}


def generate_goal(family: str, world_family: str, seed: int, level: int, split: str | None = None,
                  overrides: dict | None = None) -> GoalContract:
    """A goal instance = goal family x difficulty level x world family x seed (each changes the challenge)."""
    from ..worldgen.generator import FAMILIES
    rng = np.random.default_rng([seed, level, int(digest([family, world_family], 4), 16)])
    split = split or FAMILIES[world_family].split
    world = {"family": world_family, "seed": int(seed), "overrides": copy.deepcopy(overrides or {})}
    return GOAL_FAMILIES[family](rng, world, split, int(level))


def goal_suite(split: str, n_per: int = 2, levels=(0, 1), families=None, world_families=None, seed0: int = 0) -> list[GoalContract]:
    from ..worldgen.generator import families as wfams
    out = []
    for gf in sorted(families or GOAL_FAMILIES):
        for wf in sorted(world_families or wfams(split)):
            for lvl in levels:
                for i in range(n_per):
                    out.append(generate_goal(gf, wf, seed0 + 1000 * i + lvl, lvl, split))
    return out


def instantiate(goal: GoalContract, handle: str = "agent", store=None, strict: bool = False):
    """Create the goal's world and admit the agent with the goal's endowment. Returns (world, actor)."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..worldgen import generate
    spec = generate({"family": goal.world["family"], "overrides": goal.world.get("overrides", {})}, goal.world["seed"])
    w = World.create(spec, world_id=f"{goal.id}-{handle}", store=store, strict=strict)
    r = w.system_call("admin.join", {"handle": handle, "endowment": goal.endowment, "kind": "agent"})
    return w, r["principal"]
