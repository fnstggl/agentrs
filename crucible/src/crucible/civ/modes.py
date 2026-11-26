"""Mode C provider (MASTER_SPEC Part 2.1; crucible.substrate.modes): integrated persistent civilisations.

`substrate.modes.discover()` imports this file. The provider lists one environment per civilisation profile
(`civ:<profile>`) and builds it with `crucible.civ.generate`. Its check does not accept any world that runs: it
builds the reference profile twice from one seed, runs one simulated day, and requires

* determinism (equal event hash and state digest) and exact replay of the log;
* integration measured on data flow (crucible.civ.integration): at least `MIN_PRESENT` of the 17 Mode C
  subsystems present, at least `MIN_CONNECTED` active AND connected to another subsystem by provenance, and at
  least `MIN_COUPLINGS` coupling records.
"""
from __future__ import annotations

import time

from ..substrate import modes as M
from . import profiles

MIN_PRESENT = 15
MIN_CONNECTED = 8
MIN_COUPLINGS = 3
CHECK_PROFILE = "balanced"


def _names() -> list[str]:
    return [f"civ:{p}" for p in profiles.PROFILES]


def _make(name: str, seed: int = 0, **kw):
    from .generator import generate
    prof = name.split(":", 1)[1] if name.startswith("civ:") else name
    return generate(prof, int(seed), kw.pop("population", "development"), **kw)


def check(p) -> dict:
    import torch
    torch.set_num_threads(1)
    from ..core.kernel import World
    from . import chains, integration
    t = time.time()
    hashes, digests, w = [], [], None
    for _ in range(2):
        w = _make(f"civ:{CHECK_PROFILE}", 0)
        w.advance_time(w.now + 86400)
        hashes.append(w.last_hash)
        digests.append(w.digest())
    replay_ok = World.replay(w.spec, [e.to_dict() for e in w.log], world_id="modeC-replay").digest() == w.digest() \
        if w.log_base == 0 else None
    pv = chains.build(w, 0)
    sub = integration.subsystems(w, pv)
    n_c = len(integration._rows(w.state, "civ_links"))
    det = hashes[0] == hashes[1] and digests[0] == digests[1]
    ok = bool(det and replay_ok and sub["present"] >= MIN_PRESENT and sub["strict"] >= MIN_CONNECTED
              and n_c >= MIN_COUPLINGS)
    return {"ok": ok, "name": f"civ:{CHECK_PROFILE}", "deterministic": det, "replay_exact": replay_ok,
            "subsystems_present": sub["present"], "subsystems_active_connected": sub["strict"],
            "not_connected": sub["missing"], "couplings": n_c, "events": w.seq, "modules": len(w.modules),
            "check_wall_s": round(time.time() - t, 1)}


M.register("C", "civilisations", package="crucible.civ", names=_names, make=_make, kind="world", check=check,
           doc="Tier 3 civilisations from crucible.civ.generate: every subsystem in one kernel world with the one "
               "population, couplings between domains, profiles that differ in structure")
