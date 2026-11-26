"""The Tier 1 family registry (INTERFACE_CONTRACTS section 8, MASTER_SPEC Parts 7.5, 8 and 14.1).

A family is a structural class of worlds: one engine, one composition of mechanisms, one observation structure, one
action interface, one objective type and one set of participant policies. Its builder draws an instance from a seed
(names, coefficients, random graph instances) and a difficulty setting; nothing a seed or a coefficient changes can
make a new family. Every family states its capability category (Part 8.1-8.10), its split and why it is distinct;
its fingerprint is measured from instantiated worlds (`fingerprint.family_fingerprint`), not declared.

Splits: `train` families are the v5 training partition; `val` families form the separately partitioned v5
evaluation set `v5-eval-1` (new families, held out from training by structure, never 4.0's val/test families,
which stay in `crucible.mechanisms` untouched).

    REGISTRY[name] = {"engine", "category", "item", "split", "partition", "fingerprint", "doc", "distinct_because",
                      "objective", "axes", "participants", "action_kinds", "lever_view", "version"}
    make(name, seed, **difficulty) -> Env
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.canonical import digest

DIFFICULTY_AXES = {
    "n_variables": "number of variables in the mechanism (distractors and latent variables added)",
    "graph_size": "number of nodes of the network or stations of the system",
    "horizon": "episode length in ticks",
    "observation_noise": "standard deviation of measurement noise on readings",
    "experiment_cost": "cost of one query, probe, test or experiment replicate-tick",
    "hidden_state_complexity": "number of hidden states, regimes or latent factors",
    "opponent_capability": "learning rate / responsiveness of adaptive opponents",
    "resources": "budget or capacity available to the participant",
    "action_complexity": "number of options, targets or levers to choose among",
}

CATEGORIES = {
    "8.1": "Hidden causal systems", "8.2": "Experimental science", "8.3": "Complex engineering",
    "8.4": "Resource allocation", "8.5": "Uncertain discovery", "8.6": "Delayed consequences",
    "8.7": "Network systems", "8.8": "Strategic games", "8.9": "Cooperation", "8.10": "Changing environments",
}

PARTITIONS = {"train": "v5-train", "val": "v5-eval-1", "test": "v5-eval-1"}


@dataclass
class Family:
    name: str
    engine: str
    category: str
    item: str
    split: str
    doc: str
    distinct_because: str
    build: Callable[[np.random.Generator, dict], dict]
    axes: dict[str, object]                 # supported difficulty axis -> default value
    objective: str
    participants: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()            # "module:function" intervention tests (evaluator side)
    identify: str | None = None             # "module:function" public-interface identification procedure
    reference: str | None = None            # "module:function" public-interface scripted reference policy
    oracle: str | None = None               # "module:function" evaluator-side oracle
    lever_view: bool = False
    version: str = "1"
    extra: dict = field(default_factory=dict)


FAMILIES: dict[str, Family] = {}
REGISTRY: dict[str, dict] = {}


def family(name: str, *, engine: str, category: str, item: str, split: str, doc: str, distinct_because: str,
           axes: dict, objective: str, participants: tuple = (), claims: tuple = (), identify: str | None = None,
           reference: str | None = None, oracle: str | None = None, lever_view: bool = False, version: str = "1"):
    if split not in PARTITIONS:
        raise ValueError(f"bad split {split}")
    if category not in CATEGORIES:
        raise ValueError(f"bad category {category}")
    bad = set(axes) - set(DIFFICULTY_AXES)
    if bad:
        raise ValueError(f"unknown difficulty axes {bad}")
    if name in FAMILIES:
        raise ValueError(f"family {name} registered twice")

    def deco(fn):
        FAMILIES[name] = Family(name, engine, category, item, split, doc, distinct_because, fn, dict(axes), objective,
                                tuple(participants), tuple(claims), identify, reference, oracle, lever_view, version)
        return fn
    return deco


def family_rng(name: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(["rapid", name])[:8], 16)]))


def build_config(name: str, seed: int, **difficulty) -> dict:
    fam = FAMILIES[name]
    bad = set(difficulty) - set(fam.axes)
    if bad:
        raise ValueError(f"{name} has no difficulty axes {sorted(bad)}; it supports {sorted(fam.axes)}")
    d = dict(fam.axes)
    d.update(difficulty)
    cfg = fam.build(family_rng(name, seed), d)
    cfg["family"] = name
    return cfg


def make(name: str, seed: int, *, reward_mode: str | None = None, objective="assigned", **difficulty):
    """A fresh world of family `name` drawn with `seed` at the given difficulty (axes of `DIFFICULTY_AXES` the
    family supports; the others are refused).

    `reward_mode` defaults to "observation_only" (Clarification 2; a rapid world never returns a reward in any
    mode). `objective` is "assigned" (the family's own objective is shown), None (objective-free: no objective
    is shown and the participant may pursue its own) or a goal drawn by `crucible.rapid.goals.generate`."""
    from .env import Env
    fam = FAMILIES[name]
    cfg = build_config(name, seed, **difficulty)
    return Env(name, seed, cfg, difficulty, meta={"category": fam.category, "split": fam.split},
               reward_mode=reward_mode, objective=objective)


def families(split: str | None = None, engine: str | None = None, category: str | None = None) -> list[str]:
    return sorted(n for n, f in FAMILIES.items() if (split is None or f.split == split)
                  and (engine is None or f.engine == engine) and (category is None or f.category == category))


def _entry(f: Family) -> dict:
    from .engines import build
    eng = build(build_config(f.name, 0))
    kinds = sorted({k for a in eng.agents for k in eng.action_space(a).kinds})
    return {"engine": f.engine, "category": f.category, "item": f.item, "split": f.split,
            "partition": PARTITIONS[f.split], "doc": f.doc, "distinct_because": f.distinct_because,
            "objective": f.objective, "axes": dict(f.axes), "participants": list(f.participants),
            "action_kinds": kinds, "lever_view": f.lever_view, "version": f.version}


class _Registry(dict):
    """name -> entry; the fingerprint of an entry is measured the first time it is read."""

    def __getitem__(self, k):
        e = dict.__getitem__(self, k)
        if "fingerprint" not in e:
            from .fingerprints import family_fingerprint
            e["fingerprint"] = family_fingerprint(k)
        return e

    def get(self, k, default=None):
        return self[k] if k in self else default

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


def _load() -> None:
    from . import families_des, families_net, families_pg, families_posm, families_search  # noqa: F401
    from . import families_h1  # noqa: F401
    from . import families_h2  # noqa: F401
    from . import families_h3  # noqa: F401


_load()
REGISTRY = _Registry({n: _entry(f) for n, f in sorted(FAMILIES.items())})
