"""A shared integrated world of the one population for the SP-3.x / SP-4.1 gate tests (cached per process): the
development profile (1,500 lightweight + 32 adaptive individuals of crucible.population), the 4.0 web with generated
sites browsed by popweb, and the whole social internet on the same population, 6 simulated days. Nothing scripted
acts in it."""
from __future__ import annotations

import functools
import time

import crucible
from crucible.core.kernel import World
from crucible.population import analysis as A
from crucible.social.worlds import social_world

crucible.load_all_modules()
DAY = 86400
PLATFORMS = ["posts", "blog", "community", "letters", "pics", "kb", "dev", "launch"]


@functools.lru_cache(maxsize=1)
def integrated(days: int = 6, seed: int = 41):
    t0 = time.time()
    spec = social_world("development", seed, n=1500, web=True, sites=14, adaptive=True,
                        classes={"adaptive": 32, "strategic": 0, "semantic": 0}, platforms=PLATFORMS)
    w = World.create(spec, strict=True)
    genesis = {c: A.columns.col(w, "pop", c).copy() for c in
               sorted({c for cs in A.schema.FIELDS.values() for c in cs if c in A.columns.tables(w)["pop"]["columns"]})}
    w.advance_time(days * DAY)
    return {"world": w, "genesis": genesis, "wall_s": time.time() - t0}
