"""Shared invention experiment for Gates J and L, SP-6.3, SP-6.3b, SP-6.4b and SP-6.5b (one run per test session).

`A()` runs the four arms of `crucible.innovation.civexp` (factual, garbage, ban_novel, ban_published) in the
integrated civilisation with the population's autonomous developers (`innovation.inventworld.civ_spec`: profile
balanced, seed 3, development population, 30 simulated days) and caches them for the session, so the gate files
measure the same worlds. Nobody is scripted.

Diagnostics only: with `G_INVENTION_DUMP=<dir>` the arms' histories, outcomes and chains and the factual world (state
snapshot and event log) are written there once, for offline analysis; nothing reads them back in a gate test.
"""
from __future__ import annotations

import os

import crucible

crucible.load_all_modules()
from crucible.innovation import gates as G  # noqa: E402

KIND, SEED, DAYS, PROFILE = "civ", 3, 30, "balanced"
_DUMPED: list[bool] = []


def _dump(arms: dict, d: str) -> None:
    import pickle
    import zlib
    os.makedirs(d, exist_ok=True)
    slim = {a: {k: v for k, v in r.items() if k != "world"} for a, r in arms.items()}
    with open(os.path.join(d, "arms.pkl"), "wb") as fh:
        pickle.dump(slim, fh, protocol=5)
    w = arms["factual"]["world"]
    with open(os.path.join(d, "factual.blob"), "wb") as fh:
        fh.write(w.snapshot_blob())
    with open(os.path.join(d, "factual.log.z"), "wb") as fh:
        fh.write(zlib.compress(pickle.dumps([e.to_dict() for e in w.log], protocol=5), 1))


def A() -> dict:
    arms = G.arms(KIND, SEED, DAYS, PROFILE)
    d = os.environ.get("G_INVENTION_DUMP")
    if d and not _DUMPED:
        _DUMPED.append(True)
        try:
            _dump(arms, d)
        except Exception as e:                      # diagnostics never decide a gate
            print(f"invention dump failed: {e!r}")
    return arms
