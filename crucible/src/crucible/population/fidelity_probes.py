"""In-world probes of the population's default use of fitted runtime models (FIDELITY_REGISTER `default:` claims,
`crucible.calibration.fidelity.INWORLD_PROBES`; workstream E registers them - population code imports no calibration
module other than `runtime`).

`PROBES[(module, fitted model id)]` is a callable returning the set of fitted model ids that a default world of the
population evaluated through `crucible.calibration.runtime`: a population world (development profile, 300
individuals, the population's generated web with shops, 3 simulated days) built after `runtime.reset_usage()`. The
result is computed once per process."""
from __future__ import annotations

import functools

DAYS = 3


@functools.lru_cache(maxsize=1)
def fitted_models_evaluated() -> frozenset:
    import crucible
    from ..calibration import runtime
    from ..core.kernel import World
    from .worlds import population_world
    crucible.load_all_modules()
    runtime.reset_usage()
    w = World.create(population_world("development", 11, n=300, sites=8, shop_share=0.8, tick_s=1800,
                                      classes={"adaptive": 0, "strategic": 0, "semantic": 0}))
    try:
        w.advance_time(DAYS * 86400)
        used = {sid for sid, n in runtime.usage().items() if n > 0}
    finally:
        w.close()
    specs = runtime.specs()
    return frozenset(specs[s]["model"] for s in used if s in specs)


PROBES = {("pop", "return_daily@1"): fitted_models_evaluated, ("pop", "session_rate@1"): fitted_models_evaluated,
          ("popweb", "news_click_semantic@1"): fitted_models_evaluated,
          ("popweb", "customer_return@1"): fitted_models_evaluated}
