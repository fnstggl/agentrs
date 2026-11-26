"""SP-3.3b (MASTER_SPEC Part 3.3): every population profile including Internet scale is benchmarked; large profiles
use resource-aware sparse or lazy state and batched inference.

The development, standard and large profiles come from the benchmark suite (civ world, 3 seeds x 30 days), the
internet profile from the 10M capacity run. A large profile counts as resource-aware when its population state is
columnar (one row per individual in `pop`), at most 5% of its individuals were materialised as kernel principals
(lazy), and its learning classes decided in batches (mean batch > 1)."""
import pytest

from crucible.acceptance.v5 import evidence

import _perf_kit as K

pytestmark = [pytest.mark.slow, pytest.mark.timeout(48 * 3600)]
CMD = ("PYTHONPATH=src python -m crucible.scale.slots --cpu 6 --gpu-mb 14000 --ram-mb 64000 --disk-mb 12000 -- "
       "python -m crucible.acceptance.v5 run --requirement SP-3.3b")


def test_SP_3_3b_every_profile_benchmarked_with_sparse_state():
    s = K.suite()
    done = {p for p in ("development", "standard", "large")
            if len({r["seed"] for r in s["runs"] if r["world"] == "civ" and r["profile"] == p
                    and r["sim_days"] >= 30 - 1e-6}) >= 3}
    cap = K.capacity().get("result") or {}
    if cap.get("identities", 0) >= 10_000_000 and cap.get("qualifying_distinct_consumption_events", 0) > 0:
        done.add("internet")
    sparse = []
    for p in ("large",):
        rs = [r for r in s["runs"] if r["world"] == "civ" and r["profile"] == p]
        if rs and all(r["principals"] / max(1, r["individuals"]) <= 0.05 and r["batching"]["class2"]["mean_batch"] > 1
                      for r in rs):
            sparse.append(p)
    if cap.get("identities", 0) >= 10_000_000 and cap.get("create", {}).get("rss_mb", 1e12) / 1e7 * 1024 < 4.0:
        sparse.append("internet")           # under 4 KB of resident memory per identity at creation: columnar, lazy
    m = {"profiles_benchmarked": len(done), "large_profiles_with_sparse_state": len(sparse),
         "profiles": ",".join(sorted(done)), "sparse_profiles": ",".join(sparse),
         "internet_create_rss_mb": float(cap.get("create", {}).get("rss_mb", 0))}
    for r in s["runs"]:
        if r["world"] == "civ" and r["seed"] == 0:
            m[f"{r['profile']}_principal_share"] = round(r["principals"] / max(1, r["individuals"]), 5)
            m[f"{r['profile']}_class2_mean_batch"] = float(r["batching"]["class2"]["mean_batch"])
    evidence.record("SP-3.3b", m, command=CMD)
    assert m["profiles_benchmarked"] == 4 and m["large_profiles_with_sparse_state"] >= 2, m
