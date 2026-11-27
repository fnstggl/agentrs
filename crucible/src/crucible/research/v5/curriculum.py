"""Adaptive curriculum generation (MASTER_SPEC Part 14.3; T.5, T.6, T.8, SP-14.3).

The curriculum measures where a learner struggles from its own measured behaviour - verified, normalised episode
scores (`episodes.score_episode` or the evaluator's results) on training and validation items - and generates new
training worlds aimed at those weaknesses through the structural generator (`compose`), while keeping a
substantial share of uniformly random draws from the whole training distribution.

Weakness categories (Part 14.3), each with the generator structures that exercise it (`TEMPLATES`):

    causal_discovery        confounded histories, aggregated readings, competing pathways
    memory                  hidden drifting state or a wandering market taste, missing and delayed readings
    delayed_credit          construction delays, lagged pathways, stocks
    experimentation         pilot experiments, on-demand measurements
    strategic_flexibility   regime shifts, wear latches, drifting and wandering optima (a strategy that worked
                            stops working)
    adaptive_competitors    adaptive rivals that learn the participant's price
    skill_combination       two drivers and complementary or multiplicative pathways
    forgetting              exact revisits (same structure, same world seed) of items measured earlier, drawn
                            uniformly among items at least `REVISIT_GAP` measurements old (spaced review)
    generalization          structures measured earlier, at new world seeds (seen structure, unseen world)
    long_horizon            long horizons with drivers that pay off late or keep moving

Measures per category (`WeaknessProfiler`): learning progress (slope of recent scores, with its standard error),
regret (1 - mean score), uncertainty (standard error of the mean score) and task success (share of episodes at or
above the competence threshold). The two categories that are properties of a learner's history have their own
statistics (wave 3b repair, SP-14.3: both were reported as measured while their statistics were never computed):
forgetting is the mean paired drop from an item's first score to its exact revisit's score, beyond twice its
standard error (`forgetting_drop`; revisits are drawn without regard to the first score, so a learner that does not
change has an expected drop of zero - revisiting items selected for scoring well would make it positive by
regression to the mean); generalization is the gap between the scores of seen structures at new seeds and of
first encounters of structures, within each structural category and the recent window, pooled over categories and
beyond the 95% t quantile of its standard error (`seen_unseen_gap`). A category's weakness is its regret in excess of a reference
profile (category difficulty is not a learner weakness), plus a learning-progress bonus where scores are moving by
more than twice their standard error (practice pays there), plus a small uncertainty bonus; forgetting's weakness is
its drop and generalization's its gap (each at least 0).

Selection (`Curriculum.next`): with probability `random_share` (default 0.3) a uniform draw from the training
distribution or the structural grammar; otherwise, with probability `review_share` (default 0.1) a review item (an
exact revisit or a seen structure at a new seed, alternately: forgetting and generalization are only measurable
from revisits, so they are sampled whatever their weights), else a category drawn from the targeted weights (softmax of weakness),
mixed with uniform until its entropy is at least `entropy_floor` (collapse guard), then a structure from that
category's templates with every other field random. Partition guard: `observe` accepts only results of the
train and validation partitions; evaluation and sealed results are refused, counted and never stored.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np

from . import compose as C
from .distribution import PartitionError, Registry, build_registry, generated_entry

WEAKNESSES = ("causal_discovery", "memory", "delayed_credit", "experimentation", "strategic_flexibility",
              "adaptive_competitors", "skill_combination", "forgetting", "generalization", "long_horizon")
MEASURES = ("learning_progress", "regret", "uncertainty", "success")
# the category-specific statistic without which a category is not measured (the four MEASURES for the others)
HISTORY_STATISTIC = {"forgetting": "forgetting_drop", "generalization": "seen_unseen_gap"}
ALLOWED_PARTITIONS = ("train", "validation")
Q = 0.3
REVISIT_GAP = 8                 # an exact revisit is of an item at least this many measurements old


def _base(rng: np.random.Generator) -> dict:
    return {"topology": str(rng.choice(C.TOPOLOGIES)),
            "motifs": [str(x) for x in rng.choice(["saturation", "threshold", "non_monotonic", "linear_nonlinear",
                                                   "feedback"], size=int(rng.integers(1, 3)))],
            "combiner": str(rng.choice(C.COMBINERS)), "temporal": "instant", "channels": [],
            "n_levers": int(rng.integers(1, 3)), "drivers": [], "population": {"kind": "none"}, "horizon": 60}


def _t_causal(rng, s):
    s["drivers"] = ["confounded_history"]
    s["channels"] = ["aggregated"]
    s["topology"] = str(rng.choice(["collider", "fork", "diamond"]))
    return s


def _t_memory(rng, s):
    # a hidden drifting factor, or a market taste that wanders (following it means remembering how recent settings
    # were received)
    s["drivers"] = [str(rng.choice(["hidden_state", "wandering_optimum"]))]
    s["channels"] = sorted({"missing", str(rng.choice(["delayed", "missing"]))})
    if s["drivers"] == ["wandering_optimum"]:
        s["n_levers"] = 1
    return s


def _t_delayed(rng, s):
    s["drivers"] = ["delayed_payoff"]
    s["temporal"] = str(rng.choice(["lagged", "stock"]))
    return s


def _t_experiment(rng, s):
    s["drivers"] = ["pilot_experiments"]
    s["channels"] = ["on_demand"]
    return s


def _t_flex(rng, s):
    s["drivers"] = [str(rng.choice(["regime_shift", "wear_latch", "drifting_optimum", "wandering_optimum"]))]
    if s["drivers"] == ["wandering_optimum"]:
        s["n_levers"] = 1
    return s


def _t_rivals(rng, s):
    s["population"] = {"kind": str(rng.choice(["rivals", "mixed"])), "n": int(rng.integers(1, 3)),
                       "capability": float(rng.uniform(0.1, 0.5)), "mode": str(rng.choice(["undercut", "match"]))}
    s["n_levers"] = 1
    return s


def _t_combine(rng, s):
    s["drivers"] = [str(x) for x in rng.choice(["delayed_payoff", "hidden_state", "regime_shift", "wear_latch"],
                                               size=2, replace=False)]
    s["combiner"] = str(rng.choice(["complementarity", "multiplicative"]))
    s["n_levers"] = int(rng.integers(2, 4))
    s["topology"] = str(rng.choice(["layered", "fork", "collider"]))
    return s


def _t_long(rng, s):
    s["drivers"] = ["long_horizon", str(rng.choice(["delayed_payoff", "regime_shift", "wandering_optimum"]))]
    s["horizon"] = int(rng.integers(120, 241))
    if "wandering_optimum" in s["drivers"]:
        s["n_levers"] = 1
    return s


TEMPLATES = {"causal_discovery": _t_causal, "memory": _t_memory, "delayed_credit": _t_delayed,
             "experimentation": _t_experiment, "strategic_flexibility": _t_flex, "adaptive_competitors": _t_rivals,
             "skill_combination": _t_combine, "long_horizon": _t_long}


def targeted_structure(weakness: str, rng: np.random.Generator) -> dict:
    """A generator structure aimed at `weakness` (forgetting and generalization reuse earlier structures and are
    produced by the curriculum itself)."""
    return C.normalize(TEMPLATES[weakness](rng, _base(rng)))


def random_structure(rng: np.random.Generator) -> dict:
    """A uniform draw from the structural grammar (rejection on grammar errors)."""
    for _ in range(50):
        s = {"topology": str(rng.choice(C.TOPOLOGIES)),
             "motifs": [str(x) for x in rng.choice(C.MOTIFS, size=int(rng.integers(1, 4)))],
             "combiner": str(rng.choice(C.COMBINERS)), "temporal": str(rng.choice(C.TEMPORAL)),
             "channels": [str(x) for x in rng.choice(["direct", "delayed", "missing", "aggregated", "selected",
                                                       "on_demand"], size=int(rng.integers(0, 3)), replace=False)],
             "n_levers": int(rng.integers(1, 4)),
             "drivers": [str(x) for x in rng.choice(C.DRIVERS, size=int(rng.integers(0, 3)), replace=False)],
             "population": {"kind": str(rng.choice(C.POPULATIONS))}, "horizon": int(rng.integers(40, 121))}
        try:
            return C.normalize(s)
        except C.CompositionError:
            continue
    raise RuntimeError("grammar rejected 50 random structures")


def category_of(structure: Mapping) -> str:
    """The primary weakness category a structure exercises (measured from its fields, not declared)."""
    s = C.normalize(dict(structure))
    d, pop = set(s["drivers"]), s["population"]["kind"]
    if len(d) >= 2 and "long_horizon" not in d:
        return "skill_combination"
    if "long_horizon" in d or s["horizon"] >= 120:
        return "long_horizon"
    if pop in ("rivals", "mixed"):
        return "adaptive_competitors"
    if "pilot_experiments" in d or "on_demand" in s["channels"]:
        return "experimentation"
    if "delayed_payoff" in d or s["temporal"] in ("lagged", "stock"):
        return "delayed_credit"
    if "hidden_state" in d or "missing" in s["channels"]:
        return "memory"
    if d & {"regime_shift", "wear_latch", "drifting_optimum", "wandering_optimum"}:
        return "strategic_flexibility"
    if "confounded_history" in d or "aggregated" in s["channels"]:
        return "causal_discovery"
    return "generalization"


class WeaknessProfiler:
    """Per-category measures from measured learner behaviour (train/validation results only)."""

    def __init__(self, window: int = 24, q: float = Q):
        self.window, self.q = int(window), float(q)
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        # per structural category, the recent window of (score, seen) of non-exact items (first encounters and seen
        # structures at new seeds)
        self.fam: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.drops: list[float] = []                     # first score - exact revisit score
        self.n = 0

    def update(self, category: str, score: float, *, seen: bool = False, structural: str | None = None,
               first_score: float | None = None) -> None:
        """One measured item: `category` is what the item was drawn for, `structural` the category its structure
        exercises (`category_of`), `seen` whether its structure was measured before, `first_score` the first
        score of the same item when this is an exact revisit."""
        if category not in WEAKNESSES:
            raise ValueError(f"unknown weakness category {category}")
        self.n += 1
        self.hist[category].append(float(score))
        if first_score is not None:
            self.drops.append(float(first_score) - float(score))
        elif structural is not None:
            self.fam[structural].append((float(score), bool(seen)))

    def measures(self, category: str) -> dict:
        h = np.array(self.hist.get(category) or [])
        if len(h) == 0:
            return {"n": 0, "regret": None, "success": None, "uncertainty": None, "learning_progress": None}
        k = len(h)
        lp, lp_se = 0.0, 0.0
        if k >= 3:
            x = np.arange(k, dtype=float)
            b, a = np.polyfit(x, h, 1)
            resid = h - (a + b * x)
            sxx = float(((x - x.mean()) ** 2).sum())
            s = float(np.sqrt((resid ** 2).sum() / max(k - 2, 1)))
            lp, lp_se = float(b) * k, s / math.sqrt(sxx) * k
        return {"n": k, "regret": float(1.0 - h.mean()), "success": float((h >= self.q).mean()),
                "uncertainty": float(h.std(ddof=1) / math.sqrt(k)) if k > 1 else 1.0, "learning_progress": lp,
                "learning_progress_se": lp_se}

    def forgetting(self) -> dict:
        d = np.array(self.drops)
        if len(d) < 3:
            return {"forgetting_drop": None, "revisits": int(len(d)), "mean_drop": float(d.mean()) if len(d) else None}
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        return {"forgetting_drop": float(d.mean() - 2.0 * se), "revisits": int(len(d)), "mean_drop": float(d.mean()),
                "drop_se": se}

    def generalization(self) -> dict:
        """Seen-structure-at-a-new-seed minus first-encounter scores within each structural category (at least 2
        seen and 3 first encounters in its recent window), pooled; counted beyond the two-sided 95% t quantile of its standard error
        (df = items - 2 x categories): with a few items per category a 2-se margin gave a frozen learner a
        spurious gap in a development run (raw 0.36, se 0.13)."""
        from scipy.stats import t as student_t
        diffs, var, n_seen, n_all = [], [], 0, 0
        for c, rows in self.fam.items():
            a = np.array([sc for sc, sn in rows if sn])
            b = np.array([sc for sc, sn in rows if not sn])
            if len(a) >= 2 and len(b) >= 3:
                diffs.append(float(a.mean() - b.mean()))
                var.append(float(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
                n_seen += len(a)
                n_all += len(a) + len(b)
        if not diffs:
            return {"seen_unseen_gap": None, "categories_compared": 0, "seen_items": 0}
        g = float(np.mean(diffs))
        se = float(math.sqrt(sum(var)) / len(diffs))
        q = float(student_t.ppf(0.975, max(1, n_all - 2 * len(diffs))))
        return {"seen_unseen_gap": float(g - q * se), "raw_gap": g, "gap_se": se, "t_quantile": q,
                "categories_compared": len(diffs), "seen_items": n_seen}

    def profile(self) -> dict:
        out = {c: self.measures(c) for c in WEAKNESSES}
        out["forgetting"] = dict(out["forgetting"], **self.forgetting())
        out["generalization"] = dict(out["generalization"], **self.generalization())
        return out


class Curriculum:
    def __init__(self, registry: Registry | None = None, *, seed: int = 0, random_share: float = 0.3,
                 entropy_floor: float = 0.6, temperature: float = 0.1, reference: Mapping | None = None,
                 q: float = Q, review_share: float = 0.1):
        if not 0.2 <= random_share < 1.0:
            raise ValueError("the random-diversity share must be at least 0.2")
        self.reg = registry or build_registry()
        self.rng = np.random.default_rng(seed)
        self.random_share, self.entropy_floor, self.temperature = float(random_share), float(entropy_floor), \
            float(temperature)
        self.review_share = float(review_share)
        self._review = 0
        self.reference = dict(reference or {})
        self.prof = WeaknessProfiler(q=q)
        self.refused = {"evaluation": 0, "sealed": 0, "other": 0}
        self.draws = {"random": 0, "targeted": 0}
        self.by_category: dict[str, int] = defaultdict(int)
        self.history: list[dict] = []
        self.good: list[dict] = []
        self.items: list[dict] = []                     # measured items: structure, seed, first score, id
        self.seen_ids: set[str] = set()

    # ---------------------------------------------------------------- measurement
    def observe(self, result: Mapping, item: Mapping | None = None) -> bool:
        """Feed one measured result {partition, category | structure, score, seed?, revisit_of?}; `item` (the
        `next()` item the result measures) supplies structure, category, seed and revisit link when given.
        Whether the structure was seen before is the curriculum's own record (`seen_ids`), not the caller's claim.
        Results of the evaluation or sealed partitions are refused (returns False) and never stored."""
        part = result.get("partition")
        if part not in ALLOWED_PARTITIONS:
            self.refused[part if part in ("evaluation", "sealed") else "other"] += 1
            return False
        src = dict(result)
        for k in ("structure", "category", "seed", "revisit_of"):
            if item is not None and item.get(k) is not None:
                src[k] = item[k]
        score = float(src["score"])
        struct = src.get("structure")
        cat = src.get("category") or (category_of(struct) if struct is not None else None)
        if cat is None:
            raise ValueError("a result needs a category or a structure")
        sid = C.structure_id(struct) if struct is not None else None
        rev = src.get("revisit_of")
        first = self.items[int(rev)]["score"] if rev is not None and 0 <= int(rev) < len(self.items) else None
        seen = bool(sid is not None and sid in self.seen_ids)
        self.prof.update(cat, score, seen=seen, structural=category_of(struct) if struct is not None else None,
                         first_score=first)
        if struct is not None:
            if first is None:
                self.items.append({"structure": dict(struct), "seed": src.get("seed"), "score": score,
                                   "n": self.prof.n})
            self.seen_ids.add(sid)
            if score >= self.prof.q:
                self.good.append(dict(struct))
        return True

    def weakness(self) -> dict[str, float]:
        p = self.prof.profile()
        w = {}
        for c in WEAKNESSES:
            m = p[c]
            if m["n"] == 0:
                w[c] = 0.5                               # unmeasured: worth measuring
                continue
            if c in HISTORY_STATISTIC:
                v = m.get(HISTORY_STATISTIC[c])
                w[c] = 0.5 if v is None else max(0.0, v)    # not yet computable: worth measuring
                continue
            excess = m["regret"] - float(self.reference.get(c, 0.0))
            # learning progress counts where it is distinguishable from noise (beyond twice its standard error);
            # the raw |slope| of a frozen learner's noisy scores swamped excess regret (T.5/T.8 development,
            # 2026-11-26: the planted price-blind learner's competitor weakness, 0.31 excess regret, was never targeted)
            lp = max(0.0, abs(m["learning_progress"] or 0.0) - 2.0 * (m.get("learning_progress_se") or 0.0))
            w[c] = max(0.0, excess) + 0.5 * lp + 0.25 * (m["uncertainty"] or 0.0)
        return w

    def weights(self) -> dict[str, float]:
        w = self.weakness()
        x = np.array([w[c] for c in WEAKNESSES]) / max(self.temperature, 1e-6)
        p = np.exp(x - x.max())
        p /= p.sum()
        uni = np.full(len(p), 1.0 / len(p))
        lam = 0.0
        while _entropy(p * (1 - lam) + uni * lam) < self.entropy_floor * math.log(len(p)) and lam < 1.0:
            lam = min(1.0, lam + 0.05)
        p = p * (1 - lam) + uni * lam
        return {c: float(v) for c, v in zip(WEAKNESSES, p)}

    # ---------------------------------------------------------------- generation
    def next(self) -> dict:
        """The next training item: a generated structure (registry entry) or a uniform draw from the registry."""
        if self.rng.random() < self.random_share:
            self.draws["random"] += 1
            if self.rng.random() < 0.5:
                fams = [w["id"] for w in self.reg.worlds(tier=1, family_partition="train", source="crucible.rapid")
                        if w.get("lever_view")]
                eid = fams[int(self.rng.integers(len(fams)))]
                item = {"kind": "random", "entry": eid, "category": None}
            else:
                s = random_structure(self.rng)
                item = {"kind": "random", "structure": s, "category": category_of(s)}
        else:
            self.draws["targeted"] += 1
            wts = self.weights()
            cat = WEAKNESSES[int(self.rng.choice(len(WEAKNESSES), p=[wts[c] for c in WEAKNESSES]))]
            if self.rng.random() < self.review_share:    # review item (module doc), alternating the two kinds
                cat = ("forgetting", "generalization")[self._review % 2]
                self._review += 1
            old = [i for i, it in enumerate(self.items) if self.prof.n - it["n"] >= REVISIT_GAP and it["seed"] is not None]
            revisit = None
            if cat == "forgetting" and old:              # exact revisit, drawn without regard to its score
                revisit = old[int(self.rng.integers(len(old)))]
                s = self.items[revisit]["structure"]
            elif cat == "generalization" and self.items:  # a seen structure at a new world seed
                s = self.items[int(self.rng.integers(len(self.items)))]["structure"]
            elif cat in ("forgetting", "generalization"):  # nothing to revisit yet: a first encounter
                s = targeted_structure(str(self.rng.choice(sorted(TEMPLATES))), self.rng)
            else:
                s = targeted_structure(cat, self.rng)
            item = {"kind": "targeted", "structure": s, "category": cat}
            if revisit is not None:
                item["revisit_of"] = revisit
        if item.get("category"):
            self.by_category[item["category"]] += 1
        item["seed"] = int(self.rng.integers(0, 1_000_000))
        if item.get("revisit_of") is not None:
            item["seed"] = int(self.items[item["revisit_of"]]["seed"])
        self.history.append({k: item.get(k) for k in ("kind", "category")})
        return item

    def entry_for(self, item: Mapping) -> dict:
        if "entry" in item:
            return dict(self.reg[item["entry"]])
        return generated_entry(item["structure"], proposer="curriculum")

    def stats(self) -> dict:
        tot = sum(self.draws.values())
        cats = np.array([self.by_category.get(c, 0) for c in WEAKNESSES], dtype=float)
        ent = _entropy(cats / cats.sum()) / math.log(len(WEAKNESSES)) if cats.sum() else 0.0
        return {"random_share": self.draws["random"] / tot if tot else None, "draws": dict(self.draws),
                "category_counts": {c: int(self.by_category.get(c, 0)) for c in WEAKNESSES},
                "normalised_entropy": float(ent), "refused_results": dict(self.refused),
                "sealed_or_evaluation_results_used": 0}


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def goal_possible(env, goal_type: str) -> bool:
    """Whether the independent goal generator can draw a goal of `goal_type` from this world's public face."""
    from ...rapid import goals as G
    try:
        G.generate(env, 0.4, 5, goal_type)
        return True
    except ValueError:
        return False
