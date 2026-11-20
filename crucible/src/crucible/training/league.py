"""Opponent pools and league management for multi-agent training in shared persistent worlds.

A `League` holds policy entries (checkpoint id, factory, lineage parent, tags, generation), plays *matches* -
several policies acting as different actors in one freshly generated world, competing (or cooperating) for the
same simulated customers - and keeps:

* **matchup records** with outcomes measured independently from authoritative state per agent: independent revenue
  of the organisations it owns (related-party flows excluded), customers and market share of its services, cash
  change, and optional goal success (`GoalVerifier` predicate). Agents' own claims never enter an outcome.
* **paired worlds**: every pairing is played on the same world spec + seed list, once in each seating (seat
  handles swapped), so differences between policies are not differences between worlds or seats.
* **ratings**: Elo and a TrueSkill-style Gaussian rating (mu, sigma; conservative mu - 3 sigma), updated pairwise
  from each match's ranking.
* **opponent / collaborator sampling**: uniform, PFSP (prioritised fictitious self-play: weight f(P[learner beats
  opponent]), "hard" = (1-p)^2 or "variance" = p(1-p)) and recency-biased.
* a **held-out evaluation population** that is never sampled for training; evaluation matches against it are
  recorded separately and excluded from the statistics that drive training-time sampling and training ratings
  (leakage prevention). See docs/SELF_PLAY.md.

Policies are given by factories (zero-argument callables) or by spec strings (`policy_from_spec`):
``"builder:need=route.shortest:undercut=0.8:advertise=0"``.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

DAY = 86400
MODES = ("competition", "cooperation", "mixed")


# ====================================================================== policy specs
def _policy_classes() -> dict:
    from ..agents.scripted import BountyHunter, Builder, Composer, Employer, IdlePolicy, RandomPolicy
    return {"builder": Builder, "employer": Employer, "composer": Composer, "bounty": BountyHunter,
            "bounty_hunter": BountyHunter, "random": RandomPolicy, "idle": IdlePolicy}


def _val(s: str):
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_spec(spec: str) -> tuple[str, dict]:
    """``"name:key=value:key=value"`` -> (name, kwargs)."""
    parts = [p for p in str(spec).strip().split(":") if p]
    if not parts:
        raise ValueError("empty policy spec")
    name, kw = parts[0].lower(), {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"bad policy option {p!r} in {spec!r} (expected key=value)")
        k, v = p.split("=", 1)
        kw[k.strip()] = _val(v.strip())
    if name not in _policy_classes():
        raise ValueError(f"unknown policy {name!r}; known: {sorted(_policy_classes())}")
    return name, kw


def policy_from_spec(spec: str):
    """Build a fresh policy instance from a spec string (see module docstring)."""
    name, kw = parse_spec(spec)
    cls = _policy_classes()[name]
    pol = cls(**kw) if kw else cls()
    pol.spec = spec
    return pol


def policy_factory(spec: str) -> Callable:
    parse_spec(spec)  # validate eagerly
    return lambda: policy_from_spec(spec)


def split_specs(text: str) -> list[str]:
    """Split a comma-separated list of policy specs."""
    return [s.strip() for s in str(text).split(",") if s.strip()]


# ====================================================================== records
@dataclass
class PolicyEntry:
    id: str
    spec: str | None = None
    parent: str | None = None
    tags: tuple = ()
    generation: int = 0
    added: int = 0
    heldout: bool = False
    elo: float = 1000.0
    mu: float = 25.0
    sigma: float = 25.0 / 3
    games: int = 0
    factory: Callable | None = field(default=None, repr=False, compare=False)

    def make(self):
        if self.factory is not None:
            pol = self.factory()
        elif self.spec:
            pol = policy_from_spec(self.spec)
        else:
            raise ValueError(f"policy {self.id} has neither a factory nor a spec")
        pol.name = self.id      # label trajectories with the checkpoint id
        return pol

    @property
    def conservative(self) -> float:
        return self.mu - 3 * self.sigma

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("factory", None)
        d["tags"] = list(self.tags)
        return d


@dataclass
class MatchRecord:
    id: str
    world_family: str
    seed: int
    spec_hash: str
    mode: str
    days: float
    participants: list[str]
    seats: dict                    # policy id -> actor principal
    outcomes: dict                 # policy id -> measured outcomes
    scores: dict                   # policy id -> score used for ranking
    results: list                  # [a, b, s] pairwise: s = 1 (a wins), 0 (b wins), 0.5 (draw)
    heldout_eval: bool = False
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ====================================================================== rating math
def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def elo_update(ra: float, rb: float, s: float, k: float) -> tuple[float, float]:
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (s - ea), rb - k * (s - ea)


def trueskill_update(a: tuple[float, float], b: tuple[float, float], s: float, beta: float = 25 / 6,
                     tau: float = 25 / 300, eps: float = 0.5) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two-player TrueSkill update (with draw margin `eps`). a, b = (mu, sigma); s = 1 a wins, 0 b wins, 0.5 draw."""
    (m1, s1), (m2, s2) = a, b
    s1, s2 = math.sqrt(s1 * s1 + tau * tau), math.sqrt(s2 * s2 + tau * tau)
    if s == 0:
        (m2, s2), (m1, s1) = trueskill_update((m2, s2), (m1, s1), 1, beta, 0.0, eps)
        return (m1, s1), (m2, s2)
    c = math.sqrt(2 * beta * beta + s1 * s1 + s2 * s2)
    t = (m1 - m2) / c
    e = eps / c
    if s == 1:
        d = max(_cdf(t - e), 1e-12)
        v = _pdf(t - e) / d
        w = v * (v + t - e)
    else:
        a_, b_ = e - t, -e - t
        d = max(_cdf(a_) - _cdf(b_), 1e-12)
        v = (_pdf(b_) - _pdf(a_)) / d
        w = v * v + (a_ * _pdf(a_) - b_ * _pdf(b_)) / d
    m1 += s1 * s1 / c * v
    m2 -= s2 * s2 / c * v
    s1 *= math.sqrt(max(1e-6, 1 - s1 * s1 / (c * c) * w))
    s2 *= math.sqrt(max(1e-6, 1 - s2 * s2 / (c * c) * w))
    return (m1, s1), (m2, s2)


# ====================================================================== outcome measurement
def measure_agent(world, actor: str, endowment: int, t0: int, goal=None) -> dict:
    """Independent outcomes of one agent from authoritative state (never from the agent's claims)."""
    from ..analytics import hosts_of, owned_orgs
    from ..goals.metrics import m_cash, m_independent_revenue
    orgs = owned_orgs(world, actor)
    window = max(DAY, world.now - t0 + DAY)
    revenue = int(sum(m_independent_revenue(world, o, window) for o in orgs))
    cash = int(m_cash(world, actor) + sum(m_cash(world, o) for o in orgs))
    hosts = [h for o in orgs for h in hosts_of(world, o)]
    cons = world.modules.get("consumers")
    customers = 0
    need_total = 0
    needs = set()
    if cons is not None and "listings" in world.state.tables:
        for h in hosts:
            for lid in world.state.find_ids("listings", "host", h):
                if world.state.tables["listings"][lid]["need"]:
                    needs.add(world.state.tables["listings"][lid]["need"])
        all_shares = cons.market_shares(world)
        customers = int(sum(all_shares.get(h, 0) for h in hosts))
        need_total = int(sum(sum(cons.market_shares(world, n).values()) for n in sorted(needs)))
    requests_ok = 0
    if "svc_days" in world.state.tables:
        d0 = t0 // DAY
        for h in hosts:
            for k in world.state.find_ids("svc_days", "host", h):
                if int(k.rsplit("|", 1)[1]) >= d0:
                    requests_ok += int(world.state.tables["svc_days"][k]["ok"])
    out = {"actor": actor, "orgs": orgs, "hosts": hosts, "needs": sorted(needs), "revenue": revenue,
           "cash": cash, "cash_delta": cash - int(endowment), "customers": customers,
           "share": round(customers / need_total, 4) if need_total else 0.0, "requests_ok": requests_ok,
           "actions": int((world.state.get("principals", actor) or {}).get("actions", 0))}
    if goal is not None:
        from ..goals.contracts import GoalVerifier
        out["goal_success"] = bool(GoalVerifier(goal, actor).predicate(world))
    return out


# ====================================================================== league
class League:
    """Policy pool + match scheduler + ratings (see module docstring)."""

    def __init__(self, *, world: str = "tiny", days: float = 3.0, seed: int = 0, mode: str = "competition",
                 overrides: dict | None = None, endowment: int = 2_000_000, dt: int = 3600, k_elo: float = 24.0,
                 coop_weight: float = 0.5, draw_tol: float = 0.01, goal=None, deterministic: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.world = world
        self.days = float(days)
        self.seed = int(seed)
        self.mode = mode
        self.overrides = dict(overrides or {})
        if deterministic:
            self.overrides.setdefault("params", {})
            self.overrides["params"] = dict(self.overrides["params"], cost_model="deterministic")
        self.endowment = int(endowment)
        self.dt = int(dt)
        self.k_elo = float(k_elo)
        self.coop_weight = float(coop_weight)
        self.draw_tol = float(draw_tol)
        self.goal = goal
        self.entries: dict[str, PolicyEntry] = {}
        self.records: list[MatchRecord] = []
        self._rng = np.random.default_rng([self.seed, 7919])
        self._n = 0

    # ------------------------------------------------------------------ pool
    def add_policy(self, pid: str, factory: Callable | None = None, *, spec: str | None = None,
                   parent: str | None = None, tags=(), heldout: bool = False,
                   generation: int | None = None) -> PolicyEntry:
        if pid in self.entries:
            raise ValueError(f"policy {pid} already in the league")
        if factory is None and spec is None:
            spec = pid
        if spec is not None:
            parse_spec(spec)
        if generation is None:
            generation = self.entries[parent].generation + 1 if parent in self.entries else 0
        e = PolicyEntry(pid, spec, parent, tuple(tags), int(generation), len(self.entries), bool(heldout),
                        factory=factory)
        if parent in self.entries:   # a child starts from its parent's rating (wider uncertainty)
            p = self.entries[parent]
            e.elo, e.mu, e.sigma = p.elo, p.mu, min(25 / 3, p.sigma * 1.5)
        self.entries[pid] = e
        return e

    def training_pool(self) -> list[str]:
        return [e.id for e in sorted(self.entries.values(), key=lambda e: e.added) if not e.heldout]

    def heldout_pool(self) -> list[str]:
        return [e.id for e in sorted(self.entries.values(), key=lambda e: e.added) if e.heldout]

    def lineage(self, pid: str) -> list[str]:
        out = [pid]
        while self.entries[out[-1]].parent in self.entries:
            out.append(self.entries[out[-1]].parent)
        return out

    # ------------------------------------------------------------------ statistics
    def _pair_results(self, a: str, b: str, include_heldout_eval: bool = False):
        for r in self.records:
            if r.heldout_eval and not include_heldout_eval:
                continue
            for x, y, s in r.results:
                if (x, y) == (a, b):
                    yield s
                elif (x, y) == (b, a):
                    yield 1 - s

    def win_rate(self, a: str, b: str, prior: float = 1.0, include_heldout_eval: bool = False) -> float:
        """P[a beats b] from recorded training matches (draws count half), with a Beta(prior, prior) prior."""
        res = list(self._pair_results(a, b, include_heldout_eval))
        return (sum(res) + prior) / (len(res) + 2 * prior)

    # ------------------------------------------------------------------ sampling
    def sample_opponents(self, learner: str, k: int = 1, method: str = "pfsp", weighting: str = "hard",
                         decay: float = 0.7, exclude=(), rng: np.random.Generator | None = None) -> list[str]:
        """Opponents for `learner` from the training pool (never held-out entries). Methods: "uniform",
        "pfsp" (prioritised by how often the learner loses), "recent" (newer checkpoints more likely)."""
        pool = [p for p in self.training_pool() if p != learner and p not in set(exclude)]
        if not pool:
            return []
        rng = rng or self._rng
        if method == "uniform":
            w = np.ones(len(pool))
        elif method == "pfsp":
            p = np.array([self.win_rate(learner, o) for o in pool])
            w = (1 - p) ** 2 if weighting == "hard" else p * (1 - p)
            w = np.maximum(w, 1e-6)
        elif method == "recent":
            order = sorted(pool, key=lambda x: -self.entries[x].added)
            rank = {x: i for i, x in enumerate(order)}
            w = np.array([decay ** rank[x] for x in pool])
        else:
            raise ValueError(f"unknown sampling method {method!r}")
        w = w / w.sum()
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False, p=w)
        return [pool[i] for i in sorted(idx)]

    def sample_collaborators(self, learner: str, k: int = 1, method: str = "best", exclude=(),
                             rng: np.random.Generator | None = None) -> list[str]:
        """Partners for cooperative / mixed-motive training: "best" weights partners by the mean team score of
        past matches with the learner (softmax over ranks), otherwise as `sample_opponents`."""
        if method != "best":
            return self.sample_opponents(learner, k, method, exclude=exclude, rng=rng)
        pool = [p for p in self.training_pool() if p != learner and p not in set(exclude)]
        if not pool:
            return []
        rng = rng or self._rng
        team = []
        for o in pool:
            vals = [sum(r.scores.values()) for r in self.records if not r.heldout_eval
                    and learner in r.participants and o in r.participants]
            team.append(float(np.mean(vals)) if vals else 0.0)
        order = np.argsort(np.argsort(-np.asarray(team)))          # rank 0 = best partner
        w = np.exp(-order.astype(float))
        w = w / w.sum()
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False, p=w)
        return [pool[i] for i in sorted(idx)]

    # ------------------------------------------------------------------ matches
    def play(self, participants: list[str], seed: int, *, heldout_eval: bool = False, days: float | None = None,
             record_trajectories: bool = False):
        """One match: a fresh world (league family, `seed`), each participant as its own actor, run for `days`.
        Returns the MatchRecord (and the Runner's trajectories when `record_trajectories`)."""
        import crucible
        crucible.load_all_modules()
        from ..agents.base import Runner
        from ..core.kernel import World
        from ..worldgen import generate
        if len(set(participants)) != len(participants):
            raise ValueError("a policy cannot occupy two seats in one match")
        for p in participants:
            if self.entries[p].heldout and not heldout_eval:
                raise ValueError(f"held-out policy {p} cannot play a training match")
        days = self.days if days is None else float(days)
        t_wall = time.perf_counter()
        spec = generate({"family": self.world, "overrides": self.overrides}, int(seed))
        self._n += 1
        mid = f"m{self._n:05d}"
        w = World.create(spec, world_id=f"league-{mid}-s{seed}")
        try:
            seats, agents = {}, []
            for i, pid in enumerate(participants):
                # neutral seat handles; the policy id is NOT written into the world (person records are public)
                r = w.system_call("admin.join", {"handle": f"seat{i + 1}", "endowment": self.endowment,
                                                 "kind": "agent"})
                seats[pid] = r["principal"]
                agents.append((r["principal"], self.entries[pid].make()))
            t0 = w.now
            run = Runner(w, agents, dt=self.dt, seed=int(seed), record=record_trajectories)
            run.run(int(round(days * DAY / self.dt)))
            outcomes = {pid: measure_agent(w, seats[pid], self.endowment, t0, self.goal) for pid in participants}
            trajs = run.trajs if record_trajectories else None
            spec_hash = spec.hash()
        finally:
            w.close()
        scores = self._scores(outcomes)
        results = []
        for i, a in enumerate(participants):
            for b in participants[i + 1:]:
                sa, sb = scores[a], scores[b]
                tol = self.draw_tol * max(abs(sa), abs(sb), 1.0)
                results.append([a, b, 0.5 if abs(sa - sb) <= tol else (1.0 if sa > sb else 0.0)])
        rec = MatchRecord(mid, self.world, int(seed), spec_hash, self.mode, days, list(participants), seats,
                          outcomes, scores, results, heldout_eval, round(time.perf_counter() - t_wall, 3))
        self.records.append(rec)
        if not heldout_eval:
            self._update_ratings(rec)
        return (rec, trajs) if record_trajectories else rec

    def _scores(self, outcomes: dict) -> dict:
        rev = {p: float(o["revenue"]) for p, o in outcomes.items()}
        if self.mode == "competition":
            # independent revenue first, customers as a tie-breaker (a fraction of a cent per customer)
            return {p: rev[p] + 1e-3 * outcomes[p]["customers"] for p in outcomes}
        total = sum(rev.values())
        if self.mode == "cooperation":
            return {p: total for p in outcomes}
        return {p: rev[p] + self.coop_weight * (total - rev[p]) for p in outcomes}

    def _update_ratings(self, rec: MatchRecord) -> None:
        n = len(rec.participants)
        k = self.k_elo / max(1, n - 1)
        for a, b, s in rec.results:
            ea, eb = self.entries[a], self.entries[b]
            ea.elo, eb.elo = elo_update(ea.elo, eb.elo, s, k)
            (ea.mu, ea.sigma), (eb.mu, eb.sigma) = trueskill_update((ea.mu, ea.sigma), (eb.mu, eb.sigma), s)
        for p in rec.participants:
            self.entries[p].games += 1

    def play_pairing(self, a: str, b: str, seeds, both_seatings: bool = True, **kw) -> list[MatchRecord]:
        """Paired worlds: the pairing is played on every seed, in both seatings."""
        out = []
        for s in seeds:
            out.append(self.play([a, b], s, **kw))
            if both_seatings:
                out.append(self.play([b, a], s, **kw))
        return out

    def round_robin(self, seeds, ids=None, both_seatings: bool = True) -> list[MatchRecord]:
        ids = list(ids or self.training_pool())
        out = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                out += self.play_pairing(a, b, seeds, both_seatings)
        return out

    def train_schedule(self, learner: str, seeds, n_matches: int, method: str = "pfsp", n_opponents: int = 1):
        """Matches for one learner against sampled opponents (PFSP / uniform / recent), on the paired seeds."""
        out = []
        seeds = list(seeds)
        for i in range(n_matches):
            opp = self.sample_opponents(learner, n_opponents, method)
            if not opp:
                break
            out.append(self.play([learner, *opp], seeds[i % len(seeds)]))
        return out

    def evaluate_heldout(self, pid: str, seeds, both_seatings: bool = True) -> dict:
        """Play `pid` against every held-out policy on the paired seeds. Recorded with heldout_eval=True, which
        keeps them out of training ratings and of the win rates used for PFSP sampling."""
        rows = {}
        for h in self.heldout_pool():
            recs = []
            for s in seeds:
                recs.append(self.play([pid, h], s, heldout_eval=True))
                if both_seatings:
                    recs.append(self.play([h, pid], s, heldout_eval=True))
            res = [x[2] if x[0] == pid else 1 - x[2] for r in recs for x in r.results]
            rows[h] = {"games": len(res), "win_rate": float(np.mean(res)) if res else None,
                       "mean_score": float(np.mean([r.scores[pid] for r in recs])),
                       "opponent_mean_score": float(np.mean([r.scores[h] for r in recs]))}
        allres = [v["win_rate"] for v in rows.values() if v["win_rate"] is not None]
        return {"policy": pid, "vs": rows, "mean_win_rate": float(np.mean(allres)) if allres else None}

    # ------------------------------------------------------------------ reports
    def ratings(self) -> list[dict]:
        rows = [{"id": e.id, "generation": e.generation, "parent": e.parent, "heldout": e.heldout,
                 "games": e.games, "elo": round(e.elo, 1), "mu": round(e.mu, 3), "sigma": round(e.sigma, 3),
                 "conservative": round(e.conservative, 3), "tags": list(e.tags)} for e in self.entries.values()]
        return sorted(rows, key=lambda r: (-r["conservative"], r["id"]))

    def matchup_table(self, include_heldout_eval: bool = False) -> list[dict]:
        pairs: dict[tuple, dict] = {}
        for r in self.records:
            if r.heldout_eval and not include_heldout_eval:
                continue
            for a, b, s in r.results:
                key = tuple(sorted((a, b)))
                x = pairs.setdefault(key, {"a": key[0], "b": key[1], "games": 0, "a_wins": 0, "b_wins": 0,
                                           "draws": 0, "a_revenue": [], "b_revenue": [], "a_customers": [],
                                           "b_customers": [], "a_share": [], "b_share": []})
                sa = s if (a, b) == key else 1 - s
                x["games"] += 1
                x["a_wins" if sa == 1 else "b_wins" if sa == 0 else "draws"] += 1
                for side, pid in (("a", key[0]), ("b", key[1])):
                    o = r.outcomes[pid]
                    x[f"{side}_revenue"].append(o["revenue"])
                    x[f"{side}_customers"].append(o["customers"])
                    x[f"{side}_share"].append(o["share"])
        out = []
        for key in sorted(pairs):
            x = pairs[key]
            for f in ("revenue", "customers", "share"):
                for side in ("a", "b"):
                    v = x[f"{side}_{f}"]
                    x[f"{side}_{f}"] = round(float(np.mean(v)), 4) if v else None
            x["a_win_rate"] = round((x["a_wins"] + 0.5 * x["draws"]) / x["games"], 4) if x["games"] else None
            out.append(x)
        return out

    def generation_compare(self, gen_a: int, gen_b: int) -> dict:
        """Head-to-head of generation `gen_b` against `gen_a` (training matches) and their mean ratings."""
        A = {e.id for e in self.entries.values() if e.generation == gen_a and not e.heldout}
        B = {e.id for e in self.entries.values() if e.generation == gen_b and not e.heldout}
        res, diffs = [], []
        for r in self.records:
            if r.heldout_eval:
                continue
            for a, b, s in r.results:
                if a in B and b in A:
                    res.append(s)
                    diffs.append(r.scores[a] - r.scores[b])
                elif a in A and b in B:
                    res.append(1 - s)
                    diffs.append(r.scores[b] - r.scores[a])
        mean = lambda ids, f: float(np.mean([getattr(self.entries[i], f) for i in ids])) if ids else None  # noqa: E731
        return {"gen_a": gen_a, "gen_b": gen_b, "games": len(res),
                "b_win_rate": float(np.mean(res)) if res else None,
                "b_minus_a_score": float(np.mean(diffs)) if diffs else None,
                "elo_a": mean(A, "elo"), "elo_b": mean(B, "elo"), "mu_a": mean(A, "mu"), "mu_b": mean(B, "mu")}

    def format_tables(self) -> str:
        """Markdown matchup table and ratings."""
        lines = [f"League: world={self.world} mode={self.mode} days={self.days:g} matches={len(self.records)}", "",
                 "| A | B | games | A wins | B wins | draws | A win rate | A revenue | B revenue | A cust | B cust "
                 "| A share | B share |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in self.matchup_table():
            lines.append(f"| {x['a']} | {x['b']} | {x['games']} | {x['a_wins']} | {x['b_wins']} | {x['draws']} | "
                         f"{x['a_win_rate']} | {_c(x['a_revenue'])} | {_c(x['b_revenue'])} | {x['a_customers']} | "
                         f"{x['b_customers']} | {x['a_share']} | {x['b_share']} |")
        lines += ["", "| policy | gen | games | Elo | mu | sigma | mu-3sigma | held-out |", "|---|---|---|---|---|---|---|---|"]
        for r in self.ratings():
            lines.append(f"| {r['id']} | {r['generation']} | {r['games']} | {r['elo']} | {r['mu']} | {r['sigma']} | "
                         f"{r['conservative']} | {'yes' if r['heldout'] else ''} |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"format": "crucible-league/1", "world": self.world, "days": self.days, "seed": self.seed,
                "mode": self.mode, "overrides": self.overrides, "endowment": self.endowment, "dt": self.dt,
                "entries": [e.to_dict() for e in self.entries.values()],
                "records": [r.to_dict() for r in self.records], "ratings": self.ratings(),
                "matchups": self.matchup_table()}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)

    @classmethod
    def load(cls, path_or_dict, factories: dict | None = None) -> "League":
        d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
        lg = cls(world=d["world"], days=d["days"], seed=d["seed"], mode=d["mode"], overrides=d.get("overrides"),
                 endowment=d.get("endowment", 2_000_000), dt=d.get("dt", 3600), deterministic=False)
        for e in sorted(d["entries"], key=lambda e: e["added"]):
            ent = PolicyEntry(e["id"], e.get("spec"), e.get("parent"), tuple(e.get("tags", ())),
                              e.get("generation", 0), e.get("added", 0), e.get("heldout", False), e["elo"], e["mu"],
                              e["sigma"], e.get("games", 0), factory=(factories or {}).get(e["id"]))
            lg.entries[ent.id] = ent
        lg.records = [MatchRecord(**r) for r in d.get("records", [])]
        lg._n = len(lg.records)
        return lg


def _c(x) -> str:
    return "-" if x is None else f"${x / 100:,.2f}"
