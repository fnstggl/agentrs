"""Experiments on the adaptive curriculum (MASTER_SPEC Part 14.3; T.5, T.6, T.8, SP-14.3).

Planted weaknesses are real algorithmic deficiencies of otherwise identical within-episode adapters - each one a
different within-episode search strategy (`learners.META_ARMS`), so what the learner does badly is a property of
its behaviour, not a label:

    lag_blind          judges a change on the very next tick (settle 0, one-tick measurements): misattributes
                       delayed and accumulated effects
    price_blind        never moves a lever whose range starts above zero (a price): cannot answer rivals that
                       learn its price
    early_committer    stops exploring after a third of the horizon: cannot follow regime shifts and long
                       horizons
    single_lever       only ever moves its first lever: cannot combine several levers

`measure` runs a learner on curriculum-chosen probe worlds (validation seeds) and feeds the measured, verified
scores to the curriculum; `selection_experiment` asks whether the curriculum's targeted weights follow each learner's
independently measured weakness profile more closely than other learners' profiles (a learner-label permutation
test, `T5_DESIGN`; wave 3b repair: the earlier rank test against uniform ranks on 1..10 had an invalid null -
forgetting and generalization always ranked 9-10 for frozen learners, and two of its four learners shared their
weakest category); `targeted_training_experiment` trains a
learner whose search strategy is learnable (`TargetableMeta`) either on curriculum-targeted worlds or on uniformly
drawn worlds with the same number of episodes and compares the improvement on held-out probes of the weakness the
curriculum targeted.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import learners as LR
from .curriculum import TEMPLATES, WEAKNESSES, Curriculum, category_of, random_structure, targeted_structure
from .distribution import Registry, build_registry, generated_entry
from .episodes import references, normalised, run_episode, score_episode
from .sampler import build_env

REFERENCE_ARM = LR.META_ARMS[1]
# planted deficiency: (search strategy, lever mask) - the mask names levers the learner never moves
PLANTED = {"lag_blind": ((0.3, 0, 1, 0.7), None), "price_blind": (REFERENCE_ARM, "priced"),
           "early_committer": ((0.25, 1, 2, 0.35), None), "single_lever": (REFERENCE_ARM, "first_only")}


def lever_mask(kind: str | None, ctx: Mapping) -> np.ndarray:
    """Levers a planted learner never moves: ``priced`` = levers whose range starts above 0 (prices), ``first_only``
    = every lever but the first. Computed from the public lever ranges."""
    n = int(ctx["n"])
    if kind == "priced":
        return np.asarray(ctx.get("lo", np.zeros(n)))[:n] > 0
    if kind == "first_only":
        m = np.ones(n, dtype=bool)
        m[0] = False
        return m
    return np.zeros(n, dtype=bool)


class PlantedAdapter(LR.Adapter):
    """An adapter with a planted deficiency (search strategy and lever mask); it does not learn."""
    arch = "planted_adapter"

    def __init__(self, seed=0, arm=REFERENCE_ARM, mask: str | None = None, **cfg):
        super().__init__(seed, **cfg)
        self.ARM, self.mask_kind = tuple(arm), mask

    @classmethod
    def planted(cls, name: str, seed: int = 0) -> "PlantedAdapter":
        arm, mask = PLANTED[name]
        return cls(seed, arm=arm, mask=mask)

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        self._mask = lever_mask(self.mask_kind, ctx)
        self._u0 = np.asarray(ctx["u0"], dtype=float)

    def act(self, k):
        u = np.array(self._search.act(k), dtype=float)
        u[self._mask] = self._u0[self._mask]
        return u


class TargetableMeta(LR.MetaLearner):
    """The meta learner over (search strategy, lever mask) arms; untrained behaviour = a planted deficiency;
    training learns which arm to use from its own observed episode gain (`EpisodeSearch.episode_gain`).

    Allocation (wave 3b, D-69 addendum): one warm-up episode per arm, then UCB1 on the mean episode gain
    (exploration constant `UCB_C`, on the gain's own scale of about +-0.5); frozen, it uses the arm with the best
    mean among arms tried at least twice. The earlier epsilon-greedy allocation locked onto an early leader: with
    96 episodes its final choice on the same training stream varied with the learner seed, and the targeted and
    untargeted arms of T.8 differed by noise (development run 2026-11-26: mean difference +0.02, p 0.34)."""
    arch = "targetable_meta"
    ARMS = tuple((a, None) for a in LR.META_ARMS[1:]) + tuple(PLANTED.values())
    UCB_C = 0.25

    def __init__(self, seed=0, prior: str = "lag_blind", **cfg):
        super().__init__(seed, **cfg)
        self.bandit = LR._Bandit(len(self.ARMS))
        self.prior = self.ARMS.index(PLANTED[prior])

    def _ucb(self) -> int:
        n, tot = self.bandit.n, float(self.bandit.n.sum())
        if (n == 0).any():
            return int(np.argmax(n == 0))
        return int(np.argmax(self.bandit.sum / n + self.UCB_C * np.sqrt(np.log(tot) / n)))

    def _best(self) -> int:
        n = self.bandit.n
        if n.sum() == 0:
            return self.prior
        m = np.where(n >= min(2.0, float(n.max())), self.bandit.sum / np.maximum(n, 1), -np.inf)
        return int(np.argmax(m))

    def begin(self, ctx, rng):
        LR.LeverLearner.begin(self, ctx, rng)
        if self.frozen:
            self.arm = self._best()
        else:
            self.arm = self._ucb()
        (st, se, me, co), mk = self.ARMS[self.arm]
        self._search = LR.EpisodeSearch(ctx["u0"], ctx["horizon"], st, se, me, co)
        self._mask = lever_mask(mk, ctx)
        self._u0 = np.asarray(ctx["u0"], dtype=float)

    def act(self, k):
        u = np.array(self._search.act(k), dtype=float)
        u[self._mask] = self._u0[self._mask]
        return u


def probe_worlds(category: str, n: int, seed: int) -> list[tuple[dict, int]]:
    """Generated worlds aimed at `category` on validation seeds (structures drawn from the category template)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = targeted_structure(category, rng)
        e = generated_entry(s, proposer="probe")
        sd = int(1_000_000 + rng.integers(0, 900_000))
        try:
            r = references(e, sd)
        except Exception:  # noqa: BLE001 - an unbuildable draw is skipped (the grammar admits it, the world not)
            continue
        if r["upper"] - r["wait"] > 0.02 * max(abs(r["wait"]), 1.0):
            out.append((e, sd))
    return out


def _score(e: Mapping, seed: int, learner, rng) -> float:
    env = build_env(e, seed, {}, {"mode": "assigned"})
    was = learner.frozen
    learner.frozen = True
    rec = run_episode(env, learner, learn=False, rng=rng)
    learner.frozen = was
    return normalised(rec["value"], references(e, seed))


def reference_profile(n_per: int = 12, seed: int = 0) -> dict[str, float]:
    """Regret of the reference adapter per weakness category (category difficulty, not a learner weakness)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in TEMPLATES:
        sc = [_score(e, s, PlantedAdapter(0), rng) for e, s in probe_worlds(c, n_per, 900 + seed)]
        out[c] = float(1.0 - np.mean(sc))
    return out


def measure(curriculum: Curriculum, learner, n_items: int, seed: int) -> list[dict]:
    """Let the curriculum choose `n_items` probe worlds (validation seeds of generated structures) and feed it the
    learner's measured scores."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_items):
        it = curriculum.next()
        if "structure" not in it:
            continue
        e = curriculum.entry_for(it)
        sd = 1_000_000 + int(it["seed"]) % 900_000
        try:
            sc = _score(e, sd, learner, rng)
        except Exception:  # noqa: BLE001 - a world that cannot be built gives no measurement
            continue
        rec = {"partition": "validation", "structure": it["structure"], "category": it["category"], "score": sc,
               "seed": it["seed"], "revisit_of": it.get("revisit_of")}
        curriculum.observe(rec, item=it)
        out.append(rec)
    return out


# T.5 design (wave 3b repair, D-69 addendum; fixed before the confirmation run, whose seeds were never used in
# development): eight frozen learners with different within-episode search strategies or lever masks; the truth is
# each learner's excess regret over the reference adapter per template category, paired on independent probe worlds
# (`truth_probes` per category, probe seed `truth_seed`); the curriculum (reference profile from `reference_probes`
# per category, seed `reference_seed`) measures each learner for `n_items` items at each curriculum seed. Statistic
# (primary): the mean over runs of the Spearman correlation between the run's targeted weights and its learner's
# truth profile over the eight template categories (forgetting and generalization have no truth for a frozen
# learner and are excluded); p-value: exact learner-label permutation test (the curriculum's weights of learner i
# against the truth of learner pi(i), all 8! relabellings). Secondary, reported: the same test with the rank of the
# learner's measured weakest category.
T5_LEARNERS = {"lag_blind": PLANTED["lag_blind"], "price_blind": PLANTED["price_blind"],
               "early_committer": PLANTED["early_committer"], "single_lever": PLANTED["single_lever"],
               "frozen_defaults": (LR.META_ARMS[0], None), "big_steps": (LR.META_ARMS[5], None),
               "late_small_steps": (LR.META_ARMS[6], None), "slow_settler": (LR.META_ARMS[3], None)}
T5_DESIGN = {"learners": tuple(T5_LEARNERS), "curriculum_seeds": (20, 21, 22), "n_items": 80, "truth_probes": 8,
             "truth_seed": 4900, "reference_probes": 12, "reference_seed": 7, "categories": tuple(TEMPLATES)}


def t5_learner(name: str, seed: int = 0) -> "PlantedAdapter":
    arm, mask = T5_LEARNERS[name]
    return PlantedAdapter(seed, arm=arm, mask=mask)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def truth_profiles(learners: Sequence[str], n: int, seed: int) -> dict[str, dict[str, float]]:
    """Each learner's excess regret over the reference adapter per template category, paired on the same `n`
    independent probe worlds per category (probe seed `seed`), learner rng seeded by the probe (common random
    numbers)."""
    out = {p: {} for p in learners}
    for c in TEMPLATES:
        ws = probe_worlds(c, n, seed)
        ref = [_probe_score(e, sd, PlantedAdapter(0)) for e, sd in ws]
        for p in learners:
            out[p][c] = float(np.mean([r - _probe_score(e, sd, t5_learner(p)) for r, (e, sd) in zip(ref, ws)]))
    return out


def _curriculum_run(args) -> dict:
    p, sd, n_items, ref = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cur = Curriculum(seed=sd, reference=ref)
    measure(cur, t5_learner(p), n_items, seed=sd)
    return {"learner": p, "seed": sd, "weights": cur.weights(), "draws": cur.stats()["category_counts"]}


def permutation_pvalue(M: np.ndarray) -> tuple[float, float, int]:
    """Exact one-sided permutation test on M[i, j] = sum over learner i's runs of the statistic of its weights
    against learner j's truth: (observed sum over i of M[i, i], p = share of relabellings pi with sum M[i, pi(i)]
    >= observed, number of relabellings)."""
    import itertools
    L = M.shape[0]
    obs = float(np.trace(M))
    cnt = tot = 0
    idx = np.arange(L)
    for perm in itertools.permutations(range(L)):
        tot += 1
        cnt += int(M[idx, list(perm)].sum() >= obs - 1e-12)
    return obs, cnt / tot, tot


def selection_experiment_v2(design: Mapping = T5_DESIGN, jobs: int = 1, reference: Mapping | None = None) -> dict:
    """T.5 (design `T5_DESIGN`, module doc)."""
    learners = list(design["learners"])
    cats = list(design["categories"])
    ref = dict(reference or reference_profile(int(design["reference_probes"]), int(design["reference_seed"])))
    truth = truth_profiles(learners, int(design["truth_probes"]), int(design["truth_seed"]))
    tasks = [(p, int(sd), int(design["n_items"]), ref) for p in learners for sd in design["curriculum_seeds"]]
    if jobs > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            runs = list(ex.map(_curriculum_run, tasks))
    else:
        runs = [_curriculum_run(t) for t in tasks]
    L = len(learners)
    Ms, Mr = np.zeros((L, L)), np.zeros((L, L))
    weakest = {q: max(cats, key=lambda c: truth[q][c]) for q in learners}
    for r in runs:
        i = learners.index(r["learner"])
        w = [r["weights"][c] for c in cats]
        order = sorted(cats, key=lambda c: -r["weights"][c])
        r["rank_of_weakest"] = order.index(weakest[r["learner"]]) + 1
        r["spearman"] = _spearman(w, [truth[r["learner"]][c] for c in cats])
        for j, q in enumerate(learners):
            Ms[i, j] += _spearman(w, [truth[q][c] for c in cats])
            Mr[i, j] -= order.index(weakest[q]) + 1                  # higher is better (minus the rank)
    obs_s, p_s, n_perm = permutation_pvalue(Ms)
    obs_r, p_r, _ = permutation_pvalue(Mr)
    off = ~np.eye(L, dtype=bool)
    per = len(design["curriculum_seeds"])
    return {"runs": runs, "truth": truth, "weakest": weakest, "reference_regret": ref,
            "p_value": p_s, "mean_spearman": obs_s / len(runs),
            "mean_spearman_other_learners": float(Ms[off].mean() / per),
            "rank_p_value": p_r, "mean_rank_of_weakest": -obs_r / len(runs),
            "mean_rank_of_other_learners_weakest": float(-Mr[off].mean() / per),
            "distinct_weakest_categories": len(set(weakest.values())), "permutations": n_perm,
            "design": {k: list(v) if isinstance(v, tuple) else v for k, v in design.items()}}


def selection_experiment(planted: Sequence[str] = tuple(PLANTED), seeds: Sequence[int] = (0, 1, 2),
                         n_items: int = 48, reference: Mapping | None = None) -> dict:
    """Superseded for T.5 by `selection_experiment_v2` (kept for the wave-3b record; its p-value against uniform
    ranks on 1..10 is not a valid null, see the module doc). For each planted weakness and curriculum seed: the
    category in which the planted learner's excess regret over the reference adapter is largest versus the
    curriculum's targeted weights after `n_items` measured items."""
    ref = dict(reference or reference_profile())
    runs = []
    for p in planted:
        truth = {}
        rng = np.random.default_rng(77)
        for c in TEMPLATES:                      # paired: planted versus reference adapter on the same worlds
            ws = probe_worlds(c, 5, 1900)
            truth[c] = float(np.mean([_score(e, s, PlantedAdapter(0), rng) - _score(e, s, PlantedAdapter.planted(p), rng)
                                      for e, s in ws]))
        weakest = max(truth, key=truth.get)
        for sd in seeds:
            cur = Curriculum(seed=sd, reference=ref)
            measure(cur, PlantedAdapter.planted(p), n_items, seed=sd)
            w = cur.weights()
            order = sorted(WEAKNESSES, key=lambda c: -w[c])
            runs.append({"planted": p, "seed": sd, "weakest_measured": weakest, "rank": order.index(weakest) + 1,
                         "weights": w, "truth_excess_regret": truth})
    ranks = [r["rank"] for r in runs]
    # adversarial check of the statistic (not the preregistered test): the rank, in the same run's weights, of the
    # weakest category of the OTHER planted learners. A curriculum that favours some categories whatever the
    # learner does would rank those as well as the matched one; one that reads the learner ranks the matched lower.
    weakest = {r["planted"]: r["weakest_measured"] for r in runs}
    mism = []
    for r in runs:
        order = sorted(WEAKNESSES, key=lambda c: -r["weights"][c])
        mism += [order.index(w) + 1 for p, w in weakest.items() if p != r["planted"] and w != r["weakest_measured"]]
    return {"runs": runs, "mean_rank": float(np.mean(ranks)), "p_value": rank_pvalue(ranks, len(WEAKNESSES)),
            "reference_regret": ref, "mismatched_mean_rank": float(np.mean(mism)) if mism else None,
            "mismatched_pairs": len(mism)}


def rank_pvalue(ranks: Sequence[int], k: int) -> float:
    """P(sum of len(ranks) independent uniform ranks on 1..k <= observed sum): exact by convolution."""
    dist = np.zeros(1)
    dist[0] = 1.0
    for _ in ranks:
        new = np.zeros(len(dist) + k)
        for r in range(1, k + 1):
            new[r: r + len(dist)] += dist / k
        dist = new
    s = int(sum(ranks))
    return float(dist[: s + 1].sum())


T8_DESIGN = {"budget": 160, "n_probe": 16, "measure_items": 80, "seeds": (0, 1, 2, 3, 4)}


def _probe_score(e: Mapping, sd: int, learner) -> float:
    """A held-out probe's normalised score with common random numbers: the learner's rng is seeded by the probe, so
    scores before and after training, and of the two training arms, differ only by what the learner does."""
    return _score(e, sd, learner, np.random.default_rng([sd, 7]))


def _training_pair(args) -> dict:
    p, sd, budget, n_probe, n_measure, ref = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cur = Curriculum(seed=100 + sd, reference=ref, random_share=0.2)
    probe_l = TargetableMeta(sd, prior=p)
    measure(cur, probe_l, n_measure, seed=sd)
    w = cur.weights()
    target = max((c for c in TEMPLATES), key=lambda c: w[c])
    probes = probe_worlds(target, n_probe, 5000 + sd)
    res, arms_used = {}, {}
    for arm in ("targeted", "untargeted"):
        L = TargetableMeta(sd, prior=p)
        rng = np.random.default_rng(sd)
        before = np.mean([_probe_score(e, s, L) for e, s in probes])
        r2 = np.random.default_rng(300 + sd)
        for i in range(budget):
            if arm == "targeted":            # worlds the curriculum's generator makes for its target
                s = targeted_structure(target, r2)
            else:                            # uniformly drawn worlds of the grammar
                s = random_structure(r2)
            e = generated_entry(s, proposer="training")
            try:
                run_episode(build_env(e, int(r2.integers(0, 1_000_000)), {}, {"mode": "assigned"}), L,
                            learn=True, rng=rng, verify=False)
            except Exception:  # noqa: BLE001 - an unbuildable draw costs its episode
                continue
        after = np.mean([_probe_score(e, s, L) for e, s in probes])
        res[arm] = float(after - before)
        L.frozen = True
        arms_used[arm] = int(L._best())
    return {"planted": p, "seed": sd, "target": target, **res, "difference": res["targeted"] - res["untargeted"],
            "arm_after": arms_used, "prior_arm": TargetableMeta(sd, prior=p).prior}


def targeted_training_experiment(planted: Sequence[str] = tuple(PLANTED),
                                 seeds: Sequence[int] = T8_DESIGN["seeds"], budget: int = T8_DESIGN["budget"],
                                 n_probe: int = T8_DESIGN["n_probe"], reference: Mapping | None = None,
                                 n_measure: int = T8_DESIGN["measure_items"], jobs: int = 1) -> dict:
    """For each planted weakness: a `TargetableMeta` learner starting with that strategy is trained for `budget`
    episodes either on worlds generated for the category the curriculum targets after measuring it (the
    curriculum's targeted generator, `targeted_structure`; its random share is not part of this arm) or on uniformly
    random generated worlds; both are scored before and after on held-out probes of that category (validation
    seeds, common random numbers). Statistic: improvement(targeted) - improvement(untargeted), paired by (weakness,
    seed); p-value of a one-sided sign-flip permutation test over all pairs. Design `T8_DESIGN`, fixed before the
    gate run (D-69 addendum)."""
    ref = dict(reference or reference_profile())
    tasks = [(p, int(sd), int(budget), int(n_probe), int(n_measure), ref) for p in planted for sd in seeds]
    if jobs > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            pairs = list(ex.map(_training_pair, tasks))
    else:
        pairs = [_training_pair(t) for t in tasks]
    d = np.array([x["difference"] for x in pairs])
    return {"pairs": pairs, "mean_difference": float(d.mean()), "p_value": signflip_pvalue(d),
            "targets": sorted({x["target"] for x in pairs}), "design": {k: list(v) if isinstance(v, tuple) else v
                                                                     for k, v in T8_DESIGN.items()}}


def signflip_pvalue(d: np.ndarray, trials: int = 20000, seed: int = 0) -> float:
    """One-sided p-value of mean(d) > 0 by random sign flips (exact enumeration when len(d) <= 16)."""
    d = np.asarray(d, dtype=float)
    obs = d.mean()
    n = len(d)
    if n <= 16:
        import itertools
        cnt = tot = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            tot += 1
            cnt += int((d * np.array(signs)).mean() >= obs - 1e-12)
        return cnt / tot
    rng = np.random.default_rng(seed)
    s = rng.choice((1.0, -1.0), size=(trials, n))
    return float(((s * d).mean(axis=1) >= obs - 1e-12).mean())


def score_or_none(e: Mapping, seed: int, learner, rng) -> float | None:
    """`_score`, or None when the generated world cannot be built (reported by the caller, never hidden)."""
    try:
        return _score(e, seed, learner, rng)
    except (ValueError, KeyError, IndexError, FloatingPointError, ZeroDivisionError):
        return None
