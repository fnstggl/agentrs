"""The rest of an individual's life in a standalone Tier 2 world (kernel module `elsewhere`, host `elsewhere`).

A Tier 2 domain world composes only the subsystems its configuration needs (D-17), so its population has fewer
registered activities than a civilisation: without an alternative, every session an individual starts would go to
the domain's own activity (everybody a full-time student or job seeker). This module registers the population
activity `elsewhere`: time spent on everything the world does not model (paid work outside it, family, leisure).
Its utility is the individual's own running satisfaction plus noise, i.e. neutral: the population's habit learning
then moves an individual towards a domain activity only when that activity pays off for it above what it gets
anyway, and away from it after disappointments (participation and dropout emerge from experienced outcomes).

Fidelity record `FR-ELSEWHERE-1` (deliberate simplification): the outside option of discrete choice models; its
value is not modelled beyond satisfaction. Civilisations do not use it (their activities are the outside options).
"""
from __future__ import annotations

import numpy as np

from ..core.errors import ProtocolError
from ..core.kernel import Module, register_module
from ..substrate import columns


@register_module
class Elsewhere(Module):
    name = "elsewhere"
    host = "elsewhere"
    version = "1"
    depends = ("pop",)

    def install(self, world):
        from .. import population as P
        for k in self.config or {}:
            if k not in ("prior", "noise"):
                raise ProtocolError(f"elsewhere: unknown config key {k!r}")
        P.register_activity(world, "elsewhere", self.act, owner=self.name,
                            prior=float((self.config or {}).get("prior", 0.6)))

    def act(self, ctx, idx, now):
        idx = np.asarray(idx, np.int64)
        sat = columns.col(ctx.world, "pop", "satisfaction")[idx].astype(np.float64)
        rng = ctx.rng("elsewhere", int(now))
        return sat + rng.normal(0.0, float((self.config or {}).get("noise", 0.1)), idx.size)
