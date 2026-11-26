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


def launch_product(w, topic: str = "recipes", price: int = 400) -> str:
    """A maker (scripted, labelled in evidence) launches a content product with a launch post, through ordinary
    tools; whoever buys it is the population's own decision."""
    import numpy as np
    from crucible.population import semantics, writing
    maker = w.system_call("admin.join", {"handle": "maker", "endowment": 10_000, "password": "pw-m"})["principal"]
    call = lambda tool, **a: w.execute(maker, {"tool": tool, "args": a}).result  # noqa: E731
    rng = np.random.default_rng(3)
    t = semantics.topic_names().index(topic)
    _, guide = writing.compose(rng, t, 12, 1.0)
    doc = call("docs.create", title="A practical guide", body=guide)["doc"]
    prod = call("prod.create", name="A practical guide", kind="media", description="a guide",
                delivery={"type": "content", "doc": doc})["product"]
    call("prod.update", product=prod, status="launched")
    call("prod.offer", product=prod, kind="one_time", price_cents=price)
    _, body = writing.compose(rng, t, 4, 1.0)
    ref = call("studio.write", format="text", title="New: a practical guide", body=body)["ref"]
    return call("launch.post", ref=ref, title="New: a practical guide", product=prod)["id"]


@functools.lru_cache(maxsize=1)
def integrated(days: int = 6, seed: int = 41):
    """The world after `days` simulated days, its genesis state of the Part 3.1 columns, and the runtime decision
    models' usage of this world's run (`runtime.usage()`, reset before the world is built) for in-world audits."""
    from crucible.calibration import runtime
    runtime.reset_usage()
    t0 = time.time()
    spec = social_world("development", seed, n=1500, web=True, sites=14, adaptive=True,
                        classes={"adaptive": 32, "strategic": 0, "semantic": 0}, platforms=PLATFORMS,
                        extra=[{"name": "docs", "config": {}}, {"name": "products", "config": {}},
                               {"name": "subs", "config": {}}])
    w = World.create(spec, strict=True)
    launch_product(w)
    genesis = {c: A.columns.col(w, "pop", c).copy() for c in
               sorted({c for cs in A.schema.FIELDS.values() for c in cs if c in A.columns.tables(w)["pop"]["columns"]})}
    w.advance_time(days * DAY)
    return {"world": w, "genesis": genesis, "wall_s": time.time() - t0, "usage": dict(runtime.usage())}
