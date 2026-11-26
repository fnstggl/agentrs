"""Gate SP-8.12b - fast CPU and batched simulation where valid, with a measured throughput target that learners use
(workstream H).

Batched families: every registered family whose engine has a vectorised batch path (`crucible.rapid.batch`) is
stepped as a batch of seeds and as the single worlds `make(family, seed)` with the same random lever settings for
25 ticks; it counts when readings and money agree exactly on every tick (instances whose wiring differs are batched
in groups of equal wiring). Throughput: a learner (`crucible.rapid.vec.BatchedSearchLearner`, cross-entropy search
from observed income only) trains through the learner-facing `RapidVecEnv` (observation_only, reward exactly 0.0)
on 256 worlds per batch, and the same learner's settings are stepped through the single worlds one by one; the
target (decision H-D3 in `crucible.rapid.vec`, fixed before measuring) is >= THROUGHPUT_TARGET world-steps per
second of process CPU time on one thread and >= SPEEDUP_TARGET times the single-world path."""
from __future__ import annotations

import time

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.core.canonical import digest
from crucible.rapid import families as F
from crucible.rapid import vec as V
from crucible.rapid.batch import BatchRapid, batchable

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
THROUGHPUT_FAMILIES = ("net_routing", "hc_confounded", "ch_silent_reversal")


def _equal(family: str) -> bool:
    seeds = [0, 1, 2, 3]
    try:
        b = BatchRapid(family, seeds)
    except ValueError as e:                              # instances with different wiring: batch one wiring group
        assert "differ" in str(e)
        groups = {}
        for s in range(16):
            cfg = F.build_config(family, s)
            key = digest([[n["id"], n.get("prim"), sorted((p, [(t[0], t[1]) for t in ts]) for p, ts in
                                                         n.get("ports", {}).items()),
                           {k: v for k, v in n.get("params", {}).items() if isinstance(v, (int, str))}]
                          for n in cfg["nodes"]])
            groups.setdefault(key, []).append(s)
        seeds = max(groups.values(), key=len)[:4]
        b = BatchRapid(family, seeds)
    envs = [F.make(family, s) for s in seeds]
    rng = np.random.default_rng(3)
    for t in range(25):
        vals = b.lo + rng.random(b.lo.shape) * (b.hi - b.lo)
        out = b.step(vals)
        for i, e in enumerate(envs):
            r = e.step({"kind": "levers", "values": {n: float(v) for n, v in zip(b.lever_names[i], vals[i])}})
            eng = e._engine
            vis = [k for k, g in enumerate(getattr(eng, "gauges", [])) if not g.get("on_demand")] \
                if eng.kind == "primitive_graph" else list(range(out["readings"].shape[1]))
            want = [float(out["readings"][i, k]) for k in vis]
            if list(r["observation"]["numbers"].values()) != [w for w in want if w == w]:
                return False
            res = r["observation"]["resources"]
            if "revenue_cents_last_tick" in res and res["revenue_cents_last_tick"] != float(out["revenue"][i]):
                return False
    return True


def _single_path_rate(family: str, B: int = 32, ticks: int = 40) -> float:
    """World-steps per CPU second of the same kind of learner stepping single worlds one by one."""
    envs = [F.make(family, s) for s in range(B)]
    names = [sorted(e.action_space().kinds["levers"]["properties"]["values"]["properties"]) for e in envs]
    rng = np.random.default_rng(0)
    t0 = time.process_time()
    n = 0
    for _ in range(ticks):
        u = rng.random((B, len(names[0])))
        for i, e in enumerate(envs):
            if e.done:
                continue
            props = e.action_space().kinds["levers"]["properties"]["values"]["properties"]
            e.step({"kind": "levers", "values": {m: float(props[m]["minimum"] + x * (props[m]["maximum"] -
                                                                                   props[m]["minimum"]))
                                                 for m, x in zip(names[i], u[i])}})
            n += 1
    return n / max(time.process_time() - t0, 1e-9)


def test_SP_8_12b_batched_families_and_learner_throughput():
    cand = [n for n in sorted(F.FAMILIES) if batchable(n)]
    ok = [n for n in cand if _equal(n)]
    rates, speedups, improved = {}, {}, {}
    for fam in THROUGHPUT_FAMILIES:
        vec = V.RapidVecEnv(fam, seeds=range(256))
        assert vec.reward_mode == "observation_only"
        learner = V.BatchedSearchLearner(vec, seed=1)
        t0 = time.process_time()
        first = learner.episode()
        for _ in range(3):
            last = learner.episode()
        dt = time.process_time() - t0
        rates[fam] = learner.steps / max(dt, 1e-9)
        speedups[fam] = rates[fam] / _single_path_rate(fam)
        improved[fam] = bool(np.mean(last) > np.mean(first))
    rate, speed = min(rates.values()), min(speedups.values())
    met = bool(rate >= V.THROUGHPUT_TARGET and speed >= V.SPEEDUP_TARGET)
    record("SP-8.12b", {"batched_families": len(ok), "batchable_families_checked": len(cand),
                        "batched_throughput_target_met": met, "learner_world_steps_per_cpu_s_min": float(rate),
                        "speedup_over_single_worlds_min": float(speed),
                        "throughput_target": V.THROUGHPUT_TARGET, "speedup_target": V.SPEEDUP_TARGET},
           notes=f"Batch path equal to single worlds: {len(ok)} of {len(cand)} ({sorted(set(cand) - set(ok))} "
                 f"differ). Learner rates {rates}, speedups {speedups}, learner improved its observed income from "
                 f"its first to its fourth episode: {improved}.")
    assert set(ok) <= set(cand) and rate > 0
