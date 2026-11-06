"""Deterministic MiniWeb task generation with leakage-safe splits.

Held-out axes:
* instances  - disjoint seed ranges per split
* families   - HELDOUT_FAMILIES only appear in the "test_family" split
* layouts    - the "distractors" layout only appears in test splits
* templates  - the last goal template of each family is reserved for test splits
* vocabulary - test instances draw labels/names from a disjoint vocabulary
"""

from __future__ import annotations

import random
from typing import Iterable, Optional

from agentrs.environments.base import TaskSpec
from agentrs.environments.miniweb.families import (
    DIFFICULTY,
    FAMILIES,
    HELDOUT_FAMILIES,
    MAX_STEPS,
    TRAIN_FAMILIES,
)

SEED_BASE = {"train": 0, "val": 1_000_000, "test": 2_000_000, "test_family": 3_000_000}
TRAIN_LAYOUTS = ("default", "reversed")
TEST_LAYOUTS = ("default", "reversed", "distractors")
SPLITS = tuple(SEED_BASE)


def _vocab_split(split: str) -> str:
    return "test" if split in ("test", "test_family") else "train"


def make_task(family: str, split: str, index: int, layout: Optional[str] = None,
              template: Optional[int] = None) -> TaskSpec:
    if split not in SEED_BASE:
        raise ValueError(f"unknown split {split!r}")
    if split == "test_family" and family not in HELDOUT_FAMILIES:
        raise ValueError(f"{family} is not a held-out family")
    if split != "test_family" and family in HELDOUT_FAMILIES:
        raise ValueError(f"held-out family {family} may only be used in split 'test_family'")
    cls = FAMILIES[family]
    seed = SEED_BASE[split] + index * 101 + sum(map(ord, family))
    rng = random.Random(seed)
    private = cls.make(rng, _vocab_split(split))
    n_t = len(cls.TEMPLATES)
    is_test = split in ("test", "test_family")
    if template is None:
        template = rng.randrange(n_t) if is_test else rng.randrange(n_t - 1)
    if not is_test and template == n_t - 1:
        raise ValueError("last template is reserved for test splits")
    if layout is None:
        layout = rng.choice(TEST_LAYOUTS if is_test else TRAIN_LAYOUTS)
    if not is_test and layout not in TRAIN_LAYOUTS:
        raise ValueError(f"layout {layout} is reserved for test splits")
    fmt = {k: v for k, v in private.items() if isinstance(v, str)}
    goal = cls.TEMPLATES[template].format(**fmt)
    return TaskSpec(
        task_id=f"miniweb/{family}/{split}/{index}",
        env_id="miniweb",
        family=family,
        goal=goal,
        seed=seed,
        split=split,
        max_steps=MAX_STEPS[family],
        category=cls.category,
        layout=layout,
        difficulty=DIFFICULTY[family],
        private=private,
    )


def task_set(split: str, families: Optional[Iterable[str]] = None, n_per_family: int = 10,
             start: int = 0) -> list[TaskSpec]:
    if families is None:
        families = HELDOUT_FAMILIES if split == "test_family" else TRAIN_FAMILIES
    out = []
    for fam in families:
        for i in range(start, start + n_per_family):
            out.append(make_task(fam, split, i))
    return out


def all_families() -> list[str]:
    return list(FAMILIES)
