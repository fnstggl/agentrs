"""The fast batched path (MASTER_SPEC Part 8.12): many seeds of one family stepped together with vectorised NumPy.

Engines that implement `batch_step` (`primitive_graph`, `graph_dynamical` routing) run a batch of instances of
one structure as one array computation: parameters are stacked per instance, the state has a leading batch
dimension, and every instance draws exactly the random numbers its own world would draw - from its own tick
generator, `tick_rng(world_seed, "r1", t)` - so instance b of the batch equals the single world `make(family,
seeds[b])` and the kernel-hosted facility of the same world, bit for bit.

    sim = BatchRapid("net_routing", seeds=range(256))
    out = sim.step(np.full((256, sim.n_levers), 0.5))      # readings (B, G), revenue (B,), cost (B,)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..mechanisms.base import tick_rng
from ..mechanisms.sim import TickRng
from .engines import ENGINE_KINDS
from .env import world_seed_of
from .families import FAMILIES, build_config


class BatchRapid:
    def __init__(self, family: str, seeds: Sequence[int], **difficulty):
        self.family, self.seeds = family, [int(s) for s in seeds]
        cfgs = [build_config(family, s, **difficulty) for s in self.seeds]
        cls = ENGINE_KINDS[cfgs[0]["engine"]]
        if not hasattr(cls, "batch"):
            raise ValueError(f"engine {cls.kind} has no vectorised batch path")
        self.engine = cls.batch(cfgs)
        if not getattr(self.engine, "batched", False):
            raise ValueError(f"{family} ({cls.kind}) is not batchable")
        self.singles = [cls(c) for c in cfgs]
        self.world_seeds = [world_seed_of(family, s) for s in self.seeds]
        self.rngs = [TickRng(w, "r1") for w in self.world_seeds]
        self.B = len(self.seeds)
        self.lever_names = [[x["name"] for x in e.public_levers()] for e in self.singles]
        self.n_levers = len(self.lever_names[0])
        self.lo = np.array([[x["min"] for x in e.public_levers()] for e in self.singles])
        self.hi = np.array([[x["max"] for x in e.public_levers()] for e in self.singles])
        self.reset()

    def reset(self) -> None:
        # an initial state (including any pre-arrival history) is a pure function of the world seed, so it is built
        # once and copied on every later reset
        if getattr(self, "_init", None) is None:
            sts = [e.init_state(tick_rng(w, "r1", "init")) for e, w in zip(self.singles, self.world_seeds)]
            self._init = {k: np.concatenate([np.asarray(s[k]) for s in sts], axis=0) for k in sts[0]}
        self.state = {k: np.array(v, copy=True) for k, v in self._init.items()}
        self.t = 0

    def step(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Set every instance's levers (B, L) in lever units (clipped to range) and run one tick for all."""
        v = np.clip(np.asarray(values, dtype=float), self.lo, self.hi)
        eng = self.engine
        if eng.kind == "primitive_graph":
            ints = np.array([lv.get("integer", False) for lv in eng.lev])
            v = np.where(ints[None, :], np.round(v), v)
            self.state["lev"] = v
            self.state["levset"] = np.ones_like(v)
        else:
            self.state["w"] = v
        Z, U = [], []
        for r in self.rngs:
            z, u = eng.draw(r(self.t))
            Z.append(z)
            U.append(u)
        out = eng.batch_step(self.state, np.concatenate(Z), np.concatenate(U), self.t)
        self.t += 1
        return {"readings": out["readings"], "revenue": out["revenue"], "cost": out["cost"],
                "net": out["revenue"] - out["cost"]}


def batchable(family: str) -> bool:
    eng = FAMILIES[family].engine
    if eng == "primitive_graph":
        return True
    if eng == "graph_dynamical":
        return build_config(family, 0)["scenario"] == "routing"
    return False
