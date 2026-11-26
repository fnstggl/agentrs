"""SP-3.3 (MASTER_SPEC Part 3.3): population profiles Development, Standard, Large and Internet with the specified
sizes, each created and run (benchmarked) as a world of the one population (crucible.population).

profiles_matching_spec >= 4: a profile counts when its class counts match the Part 3.3 table AND a world of it was
created and ran one simulated daytime hour with sessions (the internet profile: 10,000,000 lightweight individuals,
compact dimensions, derived initial state)."""
import resource
import time

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World, WorldSpec
from crucible.population import profiles
from crucible.substrate import columns

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m crucible.scale.slots --cpu 1 -- python -m pytest -q tests/v5/test_gate_SP_3_3.py"
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]


def _bench(name: str) -> dict:
    from crucible.social import cascade
    # the profile's population with the social internet's posts platform as its activity (classes 2-4 rows are
    # created with the profile's counts; their policies are benchmarked in Gate G)
    p = profiles.PROFILES[name]
    spec = cascade.pop_internet_spec(None, 1, positive_control=False, profile=name,
                                     classes={k: int(p[k]) for k in profiles.CLASS_KEYS[1:]})
    t0 = time.time()
    w = World.create(spec)
    t_create = time.time() - t0
    t1 = time.time()
    w.advance_time(9 * 3600)            # the morning: individuals wake up and start sessions
    sess = sum(int(e.payload["extra"]["sessions"]) for e in w.log if e.type == "cols.batch"
               and e.payload["kind"] == "pop.session")
    cls = columns.col(w, "pop", "class")
    out = {"rows": int(cls.size), "counts": {k: int(np.sum(cls == c)) for c, k in enumerate(profiles.CLASS_KEYS, 1)},
           "create_s": round(t_create, 1), "run_9h_s": round(time.time() - t1, 1), "sessions_9h": sess,
           "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
    w.close()
    return out


def test_SP_3_3_population_profiles():
    rep = {}
    for name in ("development", "standard", "large", "internet"):
        rep[name] = _bench(name)
        rep[name]["matches_spec"] = bool(profiles.matches_spec(name))
    ok = [n for n, r in rep.items() if r["matches_spec"] and r["sessions_9h"] > 0]
    m = {"profiles_matching_spec": len(ok), "profiles": ",".join(ok)}
    for n, r in rep.items():
        m[f"{n}_rows"] = r["rows"]
        m[f"{n}_create_s"] = r["create_s"]
        m[f"{n}_run_9h_s"] = r["run_9h_s"]
        m[f"{n}_sessions_9h"] = r["sessions_9h"]
    evidence.record("SP-3.3", m, command=CMD, notes=f"Benchmarks (one process, 1 thread): {rep}")
    assert rep["internet"]["rows"] >= 10_000_000
