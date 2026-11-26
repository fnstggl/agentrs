"""The A/B/C/D training-diversity comparison and its controls (MASTER_SPEC Part 15.5; U.5, SP-15.5, SP-14.2).

Arms (one identical learner, identical seeds, matched budgets; the lever learners act only in Tier 1, so arms C and D
are run with the architecture-neutral `crucible.experience.ReferenceLearner` (``learner="reference"``), which acts
in every tier through the tools of the world it is in):

    A  the original six 4.0 training families of CAM 7.0 (abm_congestion, abm_utility, abm_wom, scm_additive,
       scm_delay, scm_gate; Tier 1 4.0 mechanism tasks, the rig.* interface)
    B  the new diverse Tier 1 rapid families (training split)
    C  Tier 1 rapid families and Tier 2 semantic digital worlds (domain configurations)
    D  all three tiers (rapid, domains, civilisations) - the default 50/30/20 compute mixture

Budgets: ``experience`` (every arm trains until the same number of world ticks has been experienced) or
``compute`` (every arm trains until the same measured CPU seconds of the training process are spent: the stop
is enforced before each episode, and the realised overshoot - at most one episode - is reported). Experience and
compute are both recorded for every arm whichever budget is matched.

Evaluation: a common set of worlds unfamiliar to every arm - held-out rapid primitive-graph families (validation
seeds) - scored by the evaluator-side references; differences between arms use the crossed learner-seed x world
interval (`hstats.crossed_diff`). Nothing assumes that D beats A: the result is whatever is measured.

Controls (`CONTROL_KINDS`, all run on the same evaluation set):

    untrained                the identical learner before any training
    frozen_pretrained        a weights network pretrained on arm B and frozen (`learners.Pretrained`)
    scripted_reference       a hand-written explore-then-commit grid policy (LABELLED SCRIPTED)
    adaptive_statistical     the bandit-over-search-strategies learner (`learners.MetaLearner`) and the
                             coordinate-search adapter
    model_based              the existing 4.0 model-based learner (`research.learners.model_based`, Bayesian last
                             layer with Thompson sampling) through a lever-interface adapter
    external_checkpoint      a learner restored from a checkpoint file written by another process, run through the
                             architecture-neutral `crucible.experience` interface (Clarification 1: CAM is replaced
                             by architecture-independent learners; any CAM checkpoint would enter the same way)
    reference_learner        `crucible.experience.ReferenceLearner` (the lightweight neural reference learner)
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np

from . import hstats as HS
from . import learners as LR
from .distribution import Registry, build_registry
from .episodes import lever_entries, references, normalised, run_episode, score_episode
from .sampler import Mixture, Sampler, TierUnavailable, build_env, make_env

ORIGINAL_SIX = ("abm_congestion", "abm_utility", "abm_wom", "scm_additive", "scm_delay", "scm_gate")
ARMS = ("A", "B", "C", "D")
BUDGET_MODES = ("experience", "compute")
CONTROL_KINDS = ("untrained", "frozen_pretrained", "scripted_reference", "adaptive_statistical", "model_based",
                 "external_checkpoint", "reference_learner")


def arm_mixture(arm: str, registry: Registry, eval_families: Sequence[str] = ()) -> Mixture:
    if arm == "A":
        return Mixture(tiers={1: 1.0}, entries=tuple(f"t1:mech:{n}" for n in ORIGINAL_SIX), name="A",
                       objectives={"assigned": 1.0}, difficulty={"0": 1.0})
    rapid = tuple(sorted(e for e in lever_entries(registry, "train") if e not in set(eval_families)))
    if arm == "B":
        return Mixture(tiers={1: 1.0}, entries=rapid, name="B", objectives={"assigned": 1.0}, difficulty={"0": 1.0})
    if arm == "C":
        return Mixture(tiers={1: 0.6, 2: 0.4}, name="C", objectives={"assigned": 1.0}, difficulty={"0": 1.0},
                       entries=rapid + tuple(w["id"] for w in registry.worlds(tier=2, family_partition="train")))
    if arm == "D":
        return Mixture(tiers={1: 0.5, 2: 0.3, 3: 0.2}, name="D", objectives={"assigned": 1.0},
                       difficulty={"0": 1.0}, entries=rapid + tuple(
                           w["id"] for w in registry.worlds(family_partition="train") if w["tier"] in (2, 3)))
    raise ValueError(arm)


def train_arm(arm: str, learner, *, mode: str, budget: float, seed: int, registry: Registry,
              eval_families: Sequence[str]) -> dict:
    """Train `learner` on arm `arm` until the matched budget is spent. Raises TierUnavailable for an arm whose tiers
    do not exist in this checkout."""
    if mode not in BUDGET_MODES:
        raise ValueError(mode)
    s = Sampler(registry, arm_mixture(arm, registry, eval_families), seed=seed)
    rng = np.random.default_rng(seed)
    ticks, cpu, episodes, skipped = 0, 0.0, 0, 0
    while (ticks if mode == "experience" else cpu) < budget:
        spec = s.next()
        c0 = time.process_time()
        try:
            env = make_env(spec, registry)
            rec = run_episode(env, learner, learn=True, rng=rng, verify=False)
        except Exception as e:  # noqa: BLE001 - a world the learner cannot drive is reported, not hidden
            from .episodes import InterfaceError
            if isinstance(e, InterfaceError):
                skipped += 1
                cpu += time.process_time() - c0
                if skipped > 50:
                    raise
                continue
            raise
        used = time.process_time() - c0
        s.report(spec, cpu_s=used, interactions=rec["interactions"], wall_s=rec["wall_s"])
        ticks += rec["ticks"]
        cpu += used
        episodes += 1
    return {"arm": arm, "mode": mode, "budget": budget, "ticks": ticks, "cpu_s": cpu, "episodes": episodes,
            "overshoot": (ticks if mode == "experience" else cpu) - budget, "skipped_uncontrollable": skipped,
            "sampler": s.shares()}


NEUTRAL = "reference"          # the architecture-neutral learner (`crucible.experience.ReferenceLearner`)
NEUTRAL_STEPS = 48            # decisions per episode of the neutral learner in a world without a tick horizon


def _neutral_steps(env) -> int:
    a = env.agents()[0]
    left = (env.observe(a).get("numbers") or {}).get("world.ticks_left")
    return int(left) if isinstance(left, (int, float)) and left > 0 else NEUTRAL_STEPS


def train_arm_neutral(arm: str, learner, *, mode: str, budget: float, seed: int, registry: Registry,
                      eval_families: Sequence[str]) -> dict:
    """`train_arm` for the architecture-neutral learner, which acts in every tier through the tool schemas of the
    world it is put in (rapid levers, the 4.0 rig, kernel tools of domains and civilisations): arms C and D train
    one identical learner across tiers. Experience is counted in decisions (one per tick in Tier 1)."""
    from ...experience.loop import run_learner
    if mode not in BUDGET_MODES:
        raise ValueError(mode)
    s = Sampler(registry, arm_mixture(arm, registry, eval_families), seed=seed)
    ticks, cpu, episodes, per_tier = 0, 0.0, 0, {}
    while (ticks if mode == "experience" else cpu) < budget:
        spec = s.next()
        c0 = time.process_time()
        env = make_env(spec, registry)
        steps = _neutral_steps(env)
        summ = run_learner(env, learner, steps)
        used = time.process_time() - c0
        s.report(spec, cpu_s=used, interactions=int(summ.get("world_changes", 0)), wall_s=float(summ.get("wall_s",
                                                                                                         0.0)))
        ticks += steps
        cpu += used
        episodes += 1
        per_tier[spec["tier"]] = per_tier.get(spec["tier"], 0) + 1
    return {"arm": arm, "mode": mode, "budget": budget, "ticks": ticks, "cpu_s": cpu, "episodes": episodes,
            "overshoot": (ticks if mode == "experience" else cpu) - budget, "episodes_per_tier": per_tier,
            "sampler": s.shares()}


def score_set_neutral(registry: Registry, learner, worlds) -> list[float]:
    """Normalised verified scores of the frozen neutral learner on Tier 1 evaluation worlds."""
    from ...experience.loop import run_learner
    from .episodes import verified_value
    learner.freeze(True)
    out = []
    for f, sd in worlds:
        env = build_env(registry[f], sd, {}, {"mode": "assigned"})
        run_learner(env, learner, _neutral_steps(env), learn=False)
        out.append(normalised(verified_value(env.verify()), references(registry[f], sd)))
    learner.freeze(False)
    return out


def eval_set(registry: Registry, families: Sequence[str], seeds: Sequence[int] = (1_000_901, 1_000_902)) -> list:
    return [(f, s) for f in families for s in seeds]


def score_set(registry: Registry, learner, worlds, rng) -> list[float]:
    was = learner.frozen
    learner.frozen = True
    out = [score_episode(registry, f, s, learner, learn=False, rng=rng)["score"] for f, s in worlds]
    learner.frozen = was
    return out


def compare(arms: Sequence[str] = ARMS, *, learner: str = "meta", mode: str = "experience", budget: float = 3000,
            seeds: Sequence[int] = (0, 1, 2), registry: Registry | None = None, n_eval: int = 4) -> dict:
    """Run the comparison for every arm that exists; arms whose tiers are missing are reported as not executed
    (with the reason), never replaced by another arm."""
    reg = registry or build_registry()
    fams = lever_entries(reg, "train")
    rng0 = np.random.default_rng(424242)
    held = tuple(sorted(fams[i] for i in rng0.permutation(len(fams))[:n_eval]))
    worlds = eval_set(reg, held)
    out = {"learner": learner, "mode": mode, "budget": budget, "seeds": list(seeds), "eval_families": list(held),
           "arms": {}, "differences": {}}
    mats = {}
    for arm in arms:
        rows, tr = [], []
        try:
            for sd in seeds:
                if learner == NEUTRAL:
                    from ...experience.reference import ReferenceLearner
                    L = ReferenceLearner(seed=sd)
                    tr.append(train_arm_neutral(arm, L, mode=mode, budget=budget, seed=sd, registry=reg,
                                                eval_families=held))
                    rows.append(score_set_neutral(reg, L, worlds))
                    continue
                L = LR.make(learner, sd)
                tr.append(train_arm(arm, L, mode=mode, budget=budget, seed=sd, registry=reg, eval_families=held))
                rows.append(score_set(reg, L, worlds, np.random.default_rng(77 + sd)))
        except TierUnavailable as e:
            out["arms"][arm] = {"executed": False, "why": str(e)}
            continue
        mats[arm] = np.array(rows)
        out["arms"][arm] = {"executed": True, "training": tr, "mean_score": float(np.mean(rows)),
                            "ticks": [t["ticks"] for t in tr], "cpu_s": [t["cpu_s"] for t in tr]}
    done = sorted(mats)
    for i, a in enumerate(done):
        for b in done[i + 1:]:
            out["differences"][f"{b}-{a}"] = HS.crossed_diff(mats[a], mats[b], paired_seeds=True)
    return out


# ====================================================================== controls
class ScriptedExploreCommit(LR.LeverLearner):
    """LABELLED SCRIPTED. Tries a fixed grid of settings for two ticks each during the first third of the episode,
    then holds the best one observed. Not a learner."""
    arch = "scripted_explore_commit"
    scripted = True

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        n = ctx["n"]
        grid = [np.full(n, f) for f in (0.2, 0.5, 0.8)] + [np.asarray(ctx["u0"])]
        self.grid, self.vals, self.H = grid, [[] for _ in grid], ctx["horizon"]

    def act(self, k):
        span = max(1, (self.H // 3) // len(self.grid))
        i = k // span
        if i < len(self.grid):
            return self.grid[i]
        return self.grid[int(np.argmax([np.mean(v) if v else -np.inf for v in self.vals]))]

    def observe(self, k, s):
        span = max(1, (self.H // 3) // len(self.grid))
        i = k // span
        if i < len(self.grid) and k % span >= span // 2:
            self.vals[i].append(s)


class ModelBased40(LR.LeverLearner):
    """The 4.0 `ModelBasedLearner` behind the lever interface: the driver's public context and own signal are
    packed into the 4.0 padded observation (settings in force, lever mask, own net per tick as revenue, time)."""
    arch = "model_based_4_0"

    def __init__(self, seed=0, **cfg):
        super().__init__(seed, **cfg)
        from ..learner import EnvSpec
        from ..learners.model_based import ModelBasedLearner
        self.spec = EnvSpec()
        self.m = ModelBasedLearner(self.spec, seed=seed)
        self.m.reset_context(1)

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        self.m.reset_context(1)
        self.u = np.asarray(ctx["u0"], dtype=float)
        self.last_s, self.scale, self.k = 0.0, None, 0
        self.n = ctx["n"]

    def _obs(self, k):
        sp = self.spec
        o = np.zeros(sp.obs_dim, dtype=np.float32)
        la = np.zeros(sp.act_dim)
        la[: self.n] = self.u * 2 - 1
        o[sp.last_action] = la
        m = np.zeros(sp.act_dim)
        m[: self.n] = 1.0
        o[sp.lever_mask] = m
        sc = np.zeros(5)
        sc[0] = sc[1] = self.last_s / (self.scale or 1.0)
        sc[3] = 1.0
        sc[4] = k / max(self.ctx["horizon"], 1)
        o[sp.scalars] = sc
        return o[None, :]

    def act(self, k):
        a = self.m.act(self._obs(k), explore=False)[0][: self.n]
        self.u = np.clip((np.asarray(a, dtype=float) + 1) / 2, 0, 1)
        return self.u

    def observe(self, k, s):
        if self.scale is None and s != 0:
            self.scale = abs(s)
        self.last_s = s


def external_checkpoint_run(registry: Registry, worlds, *, seed: int, steps: int = 60) -> dict:
    """Train a `crucible.experience.ReferenceLearner` briefly, write its checkpoint, restore it in a new learner
    object from the file alone (as an external checkpoint would arrive) and score it frozen through the
    architecture-neutral interface (`run_learner`), alongside the trained in-process learner."""
    from ...experience.loop import run_learner
    from ...experience.reference import ReferenceLearner
    L = ReferenceLearner(seed=seed)
    for f, s in worlds[:2]:
        run_learner(build_env(registry[f], 300 + s % 1000, {}, {"mode": "assigned"}), L, steps)
    d = tempfile.mkdtemp(prefix="crucible-v5-extckpt-")
    path = os.path.join(d, "ckpt.pt")
    L.save(path)
    X = ReferenceLearner(seed=seed + 1000)
    X.load(path)
    X.freeze(True)
    vals = []
    for f, s in worlds:
        env = build_env(registry[f], s, {}, {"mode": "assigned"})
        run_learner(env, X, steps)
        from .episodes import verified_value
        vals.append(normalised(verified_value(env.verify()), references(registry[f], s)))
    return {"scores": vals, "params_hash_equal": X.params_hash() == L.params_hash(), "checkpoint_bytes":
            os.path.getsize(path)}


def run_controls(registry: Registry | None = None, *, seeds: Sequence[int] = (0, 1, 2), n_eval: int = 4,
                 pretrain_episodes: int = 60) -> dict:
    """Every control kind on the common evaluation set (scores per seed x world)."""
    reg = registry or build_registry()
    fams = lever_entries(reg, "train")
    rng0 = np.random.default_rng(424242)
    held = tuple(sorted(fams[i] for i in rng0.permutation(len(fams))[:n_eval]))
    worlds = eval_set(reg, held)
    out: dict = {}
    rows: dict[str, list] = {k: [] for k in CONTROL_KINDS}
    for sd in seeds:
        rng = np.random.default_rng(31 + sd)
        rows["untrained"].append(score_set(reg, LR.make("meta", sd), worlds, rng))
        w = LR.make("weights", sd)
        s = Sampler(reg, arm_mixture("B", reg, held), seed=sd)
        for _ in range(pretrain_episodes):
            run_episode(make_env(s.next(), reg), w, learn=True, rng=rng, verify=False)
        p = LR.make("pretrained", sd)
        p.load_parameters(w.parameters())
        rows["frozen_pretrained"].append(score_set(reg, p, worlds, rng))
        rows["scripted_reference"].append(score_set(reg, ScriptedExploreCommit(sd), worlds, rng))
        rows["adaptive_statistical"].append(score_set(reg, LR.make("adapter", sd), worlds, rng))
        rows["model_based"].append(score_set(reg, ModelBased40(sd), worlds, rng))
        ext = external_checkpoint_run(reg, worlds, seed=sd)
        rows["external_checkpoint"].append(ext["scores"])
        out.setdefault("external_checkpoint_hash_preserved", []).append(ext["params_hash_equal"])
    from ...experience.loop import run_learner
    from ...experience.reference import ReferenceLearner
    from .episodes import verified_value
    for sd in seeds:
        R = ReferenceLearner(seed=sd)
        for f, s in worlds[:2]:
            run_learner(build_env(reg[f], 500 + s % 1000, {}, {"mode": "assigned"}), R, 60)
        R.freeze(True)
        vals = []
        for f, s in worlds:
            env = build_env(reg[f], s, {}, {"mode": "assigned"})
            run_learner(env, R, 60)
            vals.append(normalised(verified_value(env.verify()), references(reg[f], s)))
        rows["reference_learner"].append(vals)
    out["kinds"] = {k: {"executed": len(v) == len(seeds) and all(len(r) == len(worlds) for r in v),
                        "mean_score": float(np.mean(v)) if v else None,
                        "labels": {"scripted": k == "scripted_reference", "learner": k not in ("scripted_reference",)}}
                    for k, v in rows.items()}
    out["eval_worlds"] = [list(w) for w in worlds]
    out["matrices"] = {k: v for k, v in rows.items()}
    return out
