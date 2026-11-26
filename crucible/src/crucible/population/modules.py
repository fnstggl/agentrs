"""Kernel modules of the population package (imported by `crucible.load_all_modules()`).

`pop` (core), `popweb` (resource-visit engine), `popsites` (genesis web sites), `pop_adaptive` (class 2) and
`pop_strategic` (class 3). The torch networks are imported only when a world uses a learning class.

The kernel module `space` that population worlds list is the ONE shared semantic space's record
(`crucible.semantic.space.store`, CORE-SEM); `population.semantics` delegates to it and registers no module.

Gate E.3 probe (`crucible.semantic.encoders.effects`): `population_reading` - 3,000 individuals of the population
read a fixed catalogue (six real news abstracts of the population's corpus plus the probed artifact) through the
shared space (CORE-SEM's `space_read` world: MIND-fitted click calibration on the interest-item cosine plus the
space's comprehension index, common random numbers between the arms) for four simulated days; the outcome is each
individual's number of opens of the probed artifact. Both arms differ only in the artifact's bytes.
"""
from . import core, sites, web, adaptive, strategic  # noqa: F401
from ..semantic.space import store as _space_store  # noqa: F401  (module `space`, registered by CORE-SEM)

PROBE_N, PROBE_DAYS = 3000, 4.0
PROBE_TOPICS = ("finance-real-estate", "recipes", "weathertopstories", "movies-celebrity", "travelarticle",
                "baseball_mlb")


def _probe_catalogue() -> list[tuple[bytes, str]]:
    from . import semantics
    names, corpus = semantics.topic_names(), semantics.corpus()
    return [(corpus[names.index(t)][0].encode("utf-8"), "text/plain") for t in PROBE_TOPICS]


def _reading_world(data: bytes, mime: str, seed: int):
    from ..core.kernel import World
    from ..semantic.space import probe
    cat = _probe_catalogue() + [(bytes(data), mime)]
    return World.create(probe.world_spec(cat, len(cat) - 1, int(seed), n=PROBE_N))


def _reading_run(world):
    import numpy as np
    from ..semantic.space import probe
    from ..substrate import columns
    world.advance_time(world.now + int(PROBE_DAYS * 86400))
    return np.array(columns.col(world, probe.TABLE, "target_opens"), dtype=np.float64)


def _register_probes() -> None:
    from ..semantic.encoders import effects
    effects.register_probe("population_reading", _reading_world, _reading_run, modality="text",
                           doc="population individuals choose what to read from a catalogue through the shared "
                               "space (semantic.space.probe); outcome: opens of the probed artifact per individual")


_register_probes()
