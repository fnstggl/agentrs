"""Registry of world families. Each family provides a world factory and a task sampler with disjoint
train / val / test task (seed) ranges; held-out *families* are simply never put in a run's training mix.

Adapters for external environments register themselves only if their package is importable in the running
interpreter (e.g. Melting Pot lives in its own virtualenv and is driven through worker processes)."""
from __future__ import annotations

import random
from typing import Callable

from agentrs.sl.world import Task

FAMILIES: dict = {}
SEEDS = {"train": (0, 1_000_000), "val": (0, 1_000), "test": (0, 1_000)}


def register(name: str, factory: Callable, sampler: Callable = None, meta: dict = None) -> None:
    FAMILIES[name] = {"make": factory, "sample": sampler or default_sampler(name), "meta": meta or {}}


def default_sampler(name: str):
    def sample(rng: random.Random, split: str) -> Task:
        lo, hi = SEEDS[split]
        return Task(family=name, seed=rng.randrange(lo, hi), split=split)
    return sample


def resolve(family: str) -> bool:
    """Families created on demand: ``mp/<substrate>`` and ``mpsc/<scenario>`` (Melting Pot)."""
    if family in FAMILIES:
        return True
    if family.startswith(("mp/", "mpsc/")):
        from agentrs.sl.worlds import meltingpot as M

        if not M.available():
            return False
        kind, name = ("substrate", family[3:]) if family.startswith("mp/") else ("scenario", family[5:])
        register(family, lambda k=kind, n=name: M.MeltingPotWorld(k, n),
                 meta={"source": "google-deepmind/meltingpot 2.4.1 (Apache-2.0)", "kind": kind})
        return True
    return False


def make_world(family: str):
    resolve(family)
    return FAMILIES[family]["make"]()


def sample_task(family: str, rng: random.Random, split: str) -> Task:
    resolve(family)
    return FAMILIES[family]["sample"](rng, split)


def eval_tasks(family: str, split: str, n: int) -> list:
    """Fixed, reproducible evaluation tasks: seeds 0..n-1 of the split (never used for training: the training
    split has its own seed range / simulator seed offset)."""
    return [Task(family=family, seed=i, split=split) for i in range(n)]


def _register_builtin() -> None:
    from agentrs.sl.worlds import cam_worlds

    for n in ("negotiation", "product", "bizdev"):
        register(f"cam/{n}", cam_worlds.make(n), meta={"source": "AGENTRS CAM simulator", "agents": 1})
    from agentrs.sl.worlds import vending

    if vending.available():
        register("ext/vending", vending.VendingWorld, meta={"source": "ProsusAI/vending-bench (Apache-2.0)", "agents": 1})


_register_builtin()
