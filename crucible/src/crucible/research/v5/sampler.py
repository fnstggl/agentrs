"""The training sampler (MASTER_SPEC Parts 14.1 and 14.2; SP-14.1, SP-14.1b, SP-14.2, SP-14.2b).

    reg = distribution.build_registry()
    s = Sampler(reg, DEFAULT_MIXTURE, seed=0)          # 50/30/20 of measured compute over Tiers 1/2/3
    spec = s.next()                                    # a JSON episode spec (training items only)
    env = make_env(spec)                               # an ExperienceEnv, observation_only by default
    ... run any learner ...
    s.report(spec, cpu_s=..., interactions=..., wall_s=...)   # compute and experience, accounted separately
    st = s.state(); s2 = Sampler.resume(reg, st)        # resumable; refuses another distribution version

Episode specs are plain JSON (`EPISODE_SCHEMA`): any learner infrastructure (another process, another language)
can build the world from one with `make_env` / `python -m crucible.research.v5.sampler make <spec.json>`, or ignore
Crucible's drivers entirely and use `crucible.experience.make(**spec["make"])`.

Mixtures. `Mixture.tiers` is the share of the chosen `basis` - ``compute`` (measured CPU seconds; the default, as
Part 14.2 states shares of training compute), ``experience`` (interactions) or ``episodes`` - per tier. The sampler
picks the tier whose realised share is furthest below its target (deficit scheduling), so realised shares converge
to the targets whatever each tier's episodes cost. Within a tier, `within` is ``balanced`` (every family equally
often), ``engine_balanced`` (engines equally often, then families) or explicit weights per entry id. Difficulty
levels and objective modes (``assigned`` or ``generated`` by the independent goal generator) are drawn from their
own weights. The default mixture requires all three tiers (Part 14.2: every tier available from the start); a tier
without training families makes it raise `TierUnavailable` instead of silently training on fewer tiers.

Every spec the sampler emits is a training item of the registry (`Registry.check_trainable`: training family,
training seed range, structurally distinct from every evaluation family); `make_env` re-derives the partition from
the registry and refuses anything else, whatever the spec claims.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .distribution import (TRAIN_SEEDS, PartitionError, Registry, build_registry, canonical, difficulty_for,
                           sha, thaw)

EPISODE_SCHEMA = "crucible.v5.episode/1"
SAMPLER_SCHEMA = "crucible.v5.sampler/1"
BASES = ("compute", "experience", "episodes")


class TierUnavailable(LookupError):
    """A mixture names a tier with no training family in the registry."""


@dataclass(frozen=True)
class Mixture:
    tiers: Mapping[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.3, 3: 0.2})
    basis: str = "compute"
    within: Any = "balanced"                         # "balanced" | "engine_balanced" | {entry_id: weight}
    difficulty: Mapping[str, float] = field(default_factory=lambda: {"0": 0.5, "1": 0.3, "2": 0.2})
    objectives: Mapping[str, float] = field(default_factory=lambda: {"assigned": 0.6, "generated": 0.4})
    entries: tuple[str, ...] | None = None          # restrict to these entry ids (e.g. arm A's six families)
    sources: tuple[str, ...] | None = None          # restrict to these sources (crucible.rapid, generated, ...)
    lever_only: bool = False                        # only families with a continuous lever interface
    require_all_tiers: bool = True
    name: str = "default"

    def to_json(self) -> dict:
        d = asdict(self)
        d["tiers"] = {str(k): float(v) for k, v in self.tiers.items()}
        d["entries"] = list(self.entries) if self.entries is not None else None
        d["sources"] = list(self.sources) if self.sources is not None else None
        return d

    @classmethod
    def from_json(cls, d: Mapping) -> "Mixture":
        d = dict(d)
        d["tiers"] = {int(k): float(v) for k, v in d["tiers"].items()}
        for k in ("entries", "sources"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)


DEFAULT_MIXTURE = Mixture()


def _weights(d: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    keys = sorted(d)
    w = np.array([float(d[k]) for k in keys])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"bad weights {dict(d)}")
    return keys, w / w.sum()


class Sampler:
    def __init__(self, registry: Registry | None = None, mixture: Mixture = DEFAULT_MIXTURE, seed: int = 0):
        self.registry = registry or build_registry()
        self.mixture = mixture
        self.seed = int(seed)
        if mixture.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}")
        tot = sum(float(v) for v in mixture.tiers.values())
        if tot <= 0 or any(float(v) < 0 for v in mixture.tiers.values()):
            raise ValueError("tier shares must be non-negative and not all zero")
        self.targets = {int(k): float(v) / tot for k, v in mixture.tiers.items() if float(v) > 0}
        self.pools: dict[int, list[str]] = {}
        missing = []
        for t in sorted(self.targets):
            pool = [w["id"] for w in self.registry.worlds(tier=t, family_partition="train")
                    if (mixture.entries is None or w["id"] in mixture.entries)
                    and (mixture.sources is None or w["source"] in mixture.sources)
                    and (not mixture.lever_only or w.get("lever_view"))]
            if pool:
                self.pools[t] = sorted(pool)
            else:
                missing.append(t)
        if missing:
            if mixture.require_all_tiers:
                raise TierUnavailable(f"mixture {mixture.name!r} needs tiers {missing}, which have no training "
                                      f"world family in {self.registry.version} (training families per tier: "
                                      f"{self.registry.tiers()})")
            for t in missing:
                self.targets.pop(t)
            z = sum(self.targets.values())
            self.targets = {k: v / z for k, v in self.targets.items()}
        if not self.targets:
            raise TierUnavailable("no tier of the mixture has a training family")
        self.missing_tiers = missing
        self.draws = 0
        self.used = {t: {"cpu_s": 0.0, "interactions": 0, "episodes": 0, "wall_s": 0.0} for t in self.targets}
        self._refused: dict[str, str] = {}

    # ---------------------------------------------------------------- drawing
    def _rng(self, k: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(k), 0x5A3]))

    def _basis_used(self, t: int) -> float:
        u = self.used[t]
        return {"compute": u["cpu_s"], "experience": float(u["interactions"]), "episodes": float(u["episodes"])}[
            self.mixture.basis]

    def _pick_tier(self) -> int:
        tot = sum(self._basis_used(t) for t in self.targets)
        deficit = {t: self.targets[t] * tot - self._basis_used(t) for t in self.targets}
        return max(sorted(self.targets), key=lambda t: (deficit[t], self.targets[t]))

    def _family_weights(self, pool: list[str]) -> np.ndarray:
        w = self.mixture.within
        if isinstance(w, Mapping):
            arr = np.array([float(w.get(e, 0.0)) for e in pool])
        elif w == "engine_balanced":
            eng = [self.registry[e]["engine"] for e in pool]
            n = {x: eng.count(x) for x in set(eng)}
            arr = np.array([1.0 / n[x] for x in eng])
        elif w == "balanced":
            arr = np.ones(len(pool))
        else:
            raise ValueError(f"unknown within-tier weighting {w!r}")
        for i, e in enumerate(pool):
            if e in self._refused:
                arr[i] = 0.0
        if arr.sum() <= 0:
            raise TierUnavailable(f"no admissible training family left in pool {pool[:5]}...")
        return arr / arr.sum()

    def next(self) -> dict:
        """The next episode spec (JSON). Deterministic in (registry, mixture, seed, draw index, accounting)."""
        t = self._pick_tier()
        rng = self._rng(self.draws)
        pool = self.pools[t]
        for _ in range(len(pool) + 1):
            e = pool[int(rng.choice(len(pool), p=self._family_weights(pool)))]
            seed = int(rng.integers(TRAIN_SEEDS[0], TRAIN_SEEDS[1]))
            try:
                self.registry.check_trainable(e, seed)
                break
            except PartitionError as err:
                self._refused[e] = str(err)
        else:
            raise TierUnavailable(f"tier {t}: every family was refused")
        spec = episode_spec(self.registry, e, seed, rng, self.mixture, draw=self.draws)
        self.draws += 1
        return spec

    def report(self, spec: Mapping, *, cpu_s: float, interactions: int, wall_s: float = 0.0) -> None:
        """Account an executed episode: measured CPU seconds (compute) and interactions (experience) separately."""
        t = int(spec["tier"])
        if t not in self.used:
            raise KeyError(f"tier {t} is not in this mixture")
        u = self.used[t]
        u["cpu_s"] += max(0.0, float(cpu_s))
        u["interactions"] += max(0, int(interactions))
        u["episodes"] += 1
        u["wall_s"] += max(0.0, float(wall_s))

    def shares(self) -> dict:
        """Realised shares of compute, experience and episodes per tier, and the targets."""
        out = {"targets": {str(k): v for k, v in self.targets.items()}}
        for key, name in (("cpu_s", "compute"), ("interactions", "experience"), ("episodes", "episodes")):
            tot = sum(float(self.used[t][key]) for t in self.used)
            out[name] = {str(t): (float(self.used[t][key]) / tot if tot else 0.0) for t in self.used}
        out["totals"] = {str(t): dict(self.used[t]) for t in self.used}
        return out

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"schema": SAMPLER_SCHEMA, "registry": self.registry.version, "registry_digest": self.registry.digest,
                "mixture": self.mixture.to_json(), "seed": self.seed, "draws": self.draws,
                "used": {str(t): dict(u) for t, u in self.used.items()}, "refused": dict(self._refused)}

    @classmethod
    def resume(cls, registry: Registry, state: Mapping) -> "Sampler":
        if state.get("schema") != SAMPLER_SCHEMA:
            raise ValueError("not a sampler state")
        if state["registry_digest"] != registry.digest:
            raise ValueError(f"sampler state was made from distribution {state['registry']}, not "
                             f"{registry.version}: resuming would change the training distribution")
        s = cls(registry, Mixture.from_json(state["mixture"]), int(state["seed"]))
        s.draws = int(state["draws"])
        s.used = {int(t): dict(u) for t, u in state["used"].items()}
        s._refused = dict(state.get("refused") or {})
        return s


def episode_spec(registry: Registry, entry_id: str, seed: int, rng: np.random.Generator | None = None,
                 mixture: Mixture = DEFAULT_MIXTURE, *, draw: int | None = None, level: str | None = None,
                 objective: Mapping | None = None) -> dict:
    """The JSON spec of one episode of `entry_id` (any partition: `make_env` decides who may build it)."""
    e = registry[entry_id]
    rng = rng if rng is not None else np.random.default_rng(int(seed))
    if level is None:
        ks, w = _weights(mixture.difficulty)
        level = ks[int(rng.choice(len(ks), p=w))]
    diff = difficulty_for(e, level)
    if objective is None:
        ks, w = _weights(mixture.objectives)
        mode = ks[int(rng.choice(len(ks), p=w))]
        if mode == "generated" and e["source"] in ("crucible.rapid", "generated"):
            from ...rapid.goals import GOAL_TYPES
            objective = {"mode": "generated", "goal_type": None, "goal_difficulty": float(rng.uniform(0.2, 0.8)),
                         "goal_seed": int(rng.integers(1 << 30)), "candidates": list(GOAL_TYPES)}
        else:
            objective = {"mode": "assigned"}
    mk = dict(thaw(e["make"]))
    mk["seed"] = int(seed)
    kw = dict(mk.get("kw") or {})
    kw.update(diff)
    mk["kw"] = kw
    spec = {"schema": EPISODE_SCHEMA, "registry": registry.version, "registry_digest": registry.digest,
            "draw": draw, "entry": entry_id, "tier": int(e["tier"]), "seed": int(seed), "difficulty_level": level,
            "difficulty": diff, "objective": dict(objective), "make": mk,
            "partition": registry.partition_of(entry_id, seed)}
    spec["item"] = sha({k: spec[k] for k in ("entry", "seed", "difficulty", "objective")})[:16]
    return spec


# ====================================================================== building worlds from specs
def make_env(spec: Mapping, registry: Registry | None = None, reward_mode: str | None = None, *,
             _evaluator: bool = False):
    """The ExperienceEnv of an episode spec. Outside the evaluator only training items are built: the partition is
    re-derived from the registry (the spec's own `partition` field is never trusted)."""
    reg = registry or build_registry()
    if spec.get("schema") != EPISODE_SCHEMA:
        raise ValueError("not an episode spec")
    if spec.get("registry_digest") not in (None, reg.digest):
        raise ValueError(f"spec was drawn from distribution {spec.get('registry')}, not {reg.version}")
    eid, seed = str(spec["entry"]), int(spec["seed"])
    if eid not in reg:
        raise PartitionError(f"{eid!r} is not in distribution {reg.version}")
    if not _evaluator:
        reg.check_trainable(eid, seed)
    return build_env(reg[eid], seed, dict(spec.get("difficulty") or {}), dict(spec.get("objective") or {}),
                     reward_mode)


def build_env(entry: Mapping, seed: int, difficulty: dict, objective: dict, reward_mode: str | None = None):
    from ...experience import make as xmake
    from ...experience.tier1 import RapidExperienceEnv
    e = thaw(entry)
    if e["source"] in ("crucible.rapid", "generated"):
        raw = rapid_world(e, seed, difficulty, objective)
        return RapidExperienceEnv(e["name"], seed, reward_mode=reward_mode, rapid_env=raw)
    mk = dict(e["make"])
    kw = dict(mk.get("kw") or {})
    kw.update(difficulty)
    return xmake(int(mk["tier"]), str(mk["name"]), int(seed), reward_mode, **kw)


def rapid_world(entry: Mapping, seed: int, difficulty: dict, objective: dict | None = None):
    """The raw rapid world of a Tier 1 entry, with an assigned or independently generated objective."""
    from ...rapid import families as RF
    from ...rapid.env import Env
    e = thaw(entry)

    def build(obj):
        if e["source"] == "generated":
            from . import compose as C
            from ...rapid.families import family_rng
            cfg = C.compose(family_rng(e["name"], seed), e["structure"], difficulty)
            cfg["family"] = e["name"]
            return Env(e["name"], seed, cfg, difficulty, meta={"category": e["category"], "split": "train"},
                       objective=obj)
        return RF.make(e["name"], int(seed), objective=obj, **difficulty)
    objective = objective or {"mode": "assigned"}
    if objective.get("mode") != "generated":
        return build("assigned")
    from ...rapid import goals as G
    probe = build("assigned")
    cands = list(objective.get("candidates") or G.GOAL_TYPES)
    gt = objective.get("goal_type")
    if gt is None:                      # draw among the goal types this world's public face makes possible
        rng = np.random.default_rng(int(objective.get("goal_seed", 0)))
        possible = []
        for t in cands:
            try:
                G.generate(probe, float(objective.get("goal_difficulty", 0.5)), int(objective.get("goal_seed", 0)), t)
                possible.append(t)
            except ValueError:
                continue
        gt = possible[int(rng.integers(len(possible)))] if possible else None
    if gt is None:
        return probe
    goal = G.generate(probe, float(objective.get("goal_difficulty", 0.5)), int(objective.get("goal_seed", 0)), gt)
    return build(goal)


def _main(argv: list[str]) -> int:
    """`python -m crucible.research.v5.sampler make <spec.json>`: build the world of a spec and print its identity
    (digest of the first observation, tools, interface) - the round trip an external learner stack performs."""
    if len(argv) != 2 or argv[0] != "make":
        print("usage: python -m crucible.research.v5.sampler make <spec.json>", file=sys.stderr)
        return 2
    spec = json.load(open(argv[1]))
    env = make_env(spec)
    a = env.agents()[0]
    obs = env.observe(a)
    out = {"describe": env.describe(), "env_digest": env.digest(), "observation_digest": sha(obs),
           "tools": sorted(t["name"] for t in env.tools(a)), "reward_mode": env.reward_mode}
    print(canonical(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
