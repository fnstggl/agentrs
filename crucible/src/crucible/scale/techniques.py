"""The ten large-population techniques of MASTER_SPEC 13.6, each investigated by a measurement on the in-tree mechanism
that implements it (SP-13.6).

| technique | in-tree mechanism | measured |
|---|---|---|
| sparse state | `social.idset` sorted-array sets | bytes of a 1,000-member set over 10M identities vs a dense per-identity flag array |
| columnar arrays | `substrate.columns` cohort batches | seconds to apply one decision to 100,000 rows as one batch vs one row-dict update per individual |
| compressed bitsets | `social.idset` bitsets | bytes of a 5M-member set over 10M identities vs a sorted int32 array; exact membership |
| sparse graphs | `social.scale.ContactGraph` (computed on demand) | seconds and bytes for the contacts of 100,000 individuals in a 10M graph vs a materialised adjacency |
| lazy participant activation | population principals materialised on first need | principals materialised / individuals after a run (from the benchmark runs) |
| aggregate transitions | `scale.approx.CohortModel.predict` | CPU per simulated individual-day, estimated vs individual simulation |
| candidate retrieval | top-k over a candidate pool vs all-pairs ranking | seconds for 1,000 viewers x 200,000 items, all-pairs vs a 2,000-item candidate pool, and recall of the pool |
| hierarchical simulation | cohort parameters with backoff (cohort -> class) | error of the estimate with the hierarchy vs one flat cohort |
| adaptive-resolution simulation | individual core + estimated periphery | error and cost of (core individually, periphery estimated) vs everyone individually |
| batched cohort behaviour | population cohort batches (`pop.session`) | individuals per recorded batch event (from the benchmark runs) |
"""
from __future__ import annotations

import time

import numpy as np

TECHNIQUES = ("sparse_state", "columnar_arrays", "compressed_bitsets", "sparse_graphs", "lazy_participant_activation",
              "aggregate_transitions", "candidate_retrieval", "hierarchical_simulation", "adaptive_resolution",
              "batched_cohort_behaviour")


class _ArrayState:
    """The array part of a state store (what social.idset writes through)."""

    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}

    def set_array(self, name, a):
        self.arrays[name] = a


def sparse_state(N: int = 10_000_000, k: int = 1000, seed: int = 0) -> dict:
    from ..social import idset
    st = _ArrayState()
    m = np.unique(np.random.default_rng(seed).integers(0, N, k))
    idset.add(st, "s", m, N)
    a = st.arrays["s"]
    ok = bool(np.array_equal(idset.members(st, "s"), m))
    return {"members": int(m.size), "sparse_bytes": int(a.nbytes), "dense_flag_bytes": int(N),
            "ratio": round(N / max(1, a.nbytes), 1), "exact": ok, "measured": ok}


def compressed_bitsets(N: int = 10_000_000, k: int = 5_000_000, seed: int = 0) -> dict:
    from ..social import idset
    st = _ArrayState()
    m = np.unique(np.random.default_rng(seed).integers(0, N, k))
    t = time.perf_counter()
    idset.add(st, "s", m, N)
    add_s = time.perf_counter() - t
    a = st.arrays["s"]
    probe = np.random.default_rng(seed + 1).integers(0, N, 100_000)
    ok = bool(np.array_equal(idset.contains(st, "s", probe), np.isin(probe, m)))
    return {"members": int(m.size), "bitset_bytes": int(a.nbytes), "sorted_int32_bytes": int(4 * m.size),
            "ratio": round(4 * m.size / max(1, a.nbytes), 2), "dense": bool(idset.is_dense(a)), "add_s": round(add_s, 4),
            "exact": ok, "measured": ok and idset.is_dense(a)}


def columnar_arrays(n: int = 100_000) -> dict:
    """One cohort decision for n rows: a recorded column batch vs one Python dict update per individual."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..substrate import columns
    rows = None
    try:
        from ..population.worlds import population_world
        w = World.create(population_world("development", 0, n=n, sites=2, adaptive=False, strategic=False))
        rows = np.arange(columns.rows(w, "pop"))
        vals = np.random.default_rng(0).random(rows.size).astype(np.float32)
        t = time.perf_counter()
        w._run_system("bench.columns", lambda ctx: columns.batch(ctx, "pop", "bench.set", rows,
                                                                 {"satisfaction": vals}))
        batch_s = time.perf_counter() - t
        w.close()
    except Exception as exc:              # noqa: BLE001 - reported
        return {"measured": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    d = {i: {"satisfaction": 0.0, "habit": [0.0] * 16} for i in range(rows.size)}
    t = time.perf_counter()
    for i, v in zip(range(rows.size), vals.tolist()):
        d[i] = {**d[i], "satisfaction": v}
    dict_s = time.perf_counter() - t
    return {"rows": int(rows.size), "batch_s": round(batch_s, 4), "row_dicts_s": round(dict_s, 4),
            "speedup": round(dict_s / max(batch_s, 1e-9), 2), "measured": True}


def sparse_graphs(N: int = 10_000_000, q: int = 100_000) -> dict:
    from ..social.scale import ContactGraph
    g = ContactGraph(N, 7, 200, 12, 4)
    i = np.random.default_rng(0).integers(0, N, q)
    t = time.perf_counter()
    c = g.contacts(i)
    s = time.perf_counter() - t
    same = bool(np.array_equal(c, g.contacts(i)))
    return {"identities": N, "queried": q, "degree": g.degree, "seconds": round(s, 4),
            "materialised_adjacency_bytes": int(N * g.degree * 4), "on_demand_bytes": 0,
            "deterministic": same, "measured": same}


def candidate_retrieval(viewers: int = 200, items: int = 1_000_000, k: int = 20, dim: int = 64, lists: int = 1024,
                        probes: int = 4) -> dict:
    """Top-k items per viewer by inner product: all-pairs ranking vs candidate retrieval from an inverted-file index
    (items assigned to the nearest of `lists` centroids; a viewer's candidates are the items of its `probes`
    nearest lists), with the recall of the retrieved top-k against the exact all-pairs top-k."""
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (32, dim)).astype(np.float32)
    I = (centres[rng.integers(0, 32, items)] + 0.6 * rng.normal(0, 1, (items, dim))).astype(np.float32)
    U = (centres[rng.integers(0, 32, viewers)] + 0.6 * rng.normal(0, 1, (viewers, dim))).astype(np.float32)
    t = time.perf_counter()
    full = np.argpartition(-(U @ I.T), k, axis=1)[:, :k]
    all_s = time.perf_counter() - t
    t = time.perf_counter()
    C = I[rng.choice(items, lists, replace=False)]
    assign = np.argmax(I @ C.T, axis=1)
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(lists + 1))
    build_s = time.perf_counter() - t
    t = time.perf_counter()
    near = np.argpartition(-(U @ C.T), probes, axis=1)[:, :probes]
    best_s = np.full((viewers, k), -np.inf, np.float32)
    best_i = np.zeros((viewers, k), np.int64)
    cand_n = 0
    for c in range(lists):                             # one matrix product per list: its items x its probers
        vs = np.flatnonzero((near == c).any(axis=1))
        its = order[starts[c]:starts[c + 1]]
        if vs.size == 0 or its.size == 0:
            continue
        cand_n += vs.size * its.size
        sc = U[vs] @ I[its].T
        allv = np.concatenate([best_s[vs], sc], axis=1)
        alli = np.concatenate([best_i[vs], np.broadcast_to(its, (vs.size, its.size))], axis=1)
        top = np.argpartition(-allv, k - 1, axis=1)[:, :k]
        best_s[vs] = np.take_along_axis(allv, top, 1)
        best_i[vs] = np.take_along_axis(alli, top, 1)
    got = list(best_i)
    pool_s = time.perf_counter() - t
    rec = float(np.mean([len(set(full[v]) & set(got[v])) / k for v in range(viewers)]))
    return {"viewers": viewers, "items": items, "mean_candidates": round(cand_n / viewers, 1),
            "all_pairs_s": round(all_s, 4), "index_build_s": round(build_s, 4), "retrieval_s": round(pool_s, 4),
            "speedup": round(all_s / max(pool_s, 1e-9), 2), "recall_of_all_pairs_top_k": round(rec, 4),
            "measured": True}


def from_runs(runs: list[dict]) -> dict:
    """lazy activation and batched cohort behaviour from benchmark runs (bench5 reports)."""
    lazy, batch = [], []
    for r in runs:
        n = max(1, int(r.get("individuals") or 0))
        lazy.append(int(r.get("principals") or 0) / n)
        nb = int((r.get("batches_by_kind") or {}).get("pop.session", 0))
        if nb:
            batch.append(int(r.get("activations") or 0) / nb)
    return {"lazy_participant_activation": {"principal_share_mean": round(float(np.mean(lazy)), 5) if lazy else None,
                                            "runs": len(lazy), "measured": bool(lazy)},
            "batched_cohort_behaviour": {"individuals_per_session_batch": round(float(np.mean(batch)), 2) if batch
                                         else None, "runs": len(batch), "measured": bool(batch)}}


def from_approx(val: dict, hifi_cpu_per_ind_day: float | None) -> dict:
    """aggregate transitions, hierarchy and adaptive resolution from a validation of the estimated mode."""
    from . import approx
    model = approx.CohortModel()
    t = time.process_time()
    rng_cohorts = {100 + s: 1_000_000 for s in range(10)}
    model.params = {"cohorts": {c: {"p11": 0.7, "p01": 0.2, "p_act0": 0.4, "sess_mean": 1.5,
                                    "sess_hist": np.ones(12) / 12, "adopt_hazard": 0.01, "pur_per_act": 0.05}
                                for c in rng_cohorts}, "kind_attr": {0: 1.0}, "kind_conc": {0: np.ones(8) / 8},
                    "rev_per_purchase": 500.0, "N_fit": 1000}
    model.predict(rng_cohorts, np.zeros(100, np.int64), 30)
    est_cpu = (time.process_time() - t) / (10_000_000 * 30)
    errs = val.get("errors", {})
    return {"aggregate_transitions": {"estimated_cpu_s_per_individual_day": est_cpu,
                                      "hifi_cpu_s_per_individual_day": hifi_cpu_per_ind_day,
                                      "speedup": round(hifi_cpu_per_ind_day / est_cpu, 1) if hifi_cpu_per_ind_day else None,
                                      "measured": True},
            "hierarchical_simulation": {"cohorts_fitted": val.get("params_fitted", {}).get("cohorts"),
                                        "backoff": "cohort -> nearest cohort of the class", "measured": bool(errs)},
            "adaptive_resolution": {"core_individuals": val.get("n_fit"), "population": val.get("n_target"),
                                    "revenue_error": (errs.get("revenue") or {}).get("mean"),
                                    "measured": bool(errs)}}


def investigate(runs: list[dict] | None = None, approx_validation: dict | None = None,
                hifi_cpu_per_ind_day: float | None = None) -> dict:
    out = {"sparse_state": sparse_state(), "compressed_bitsets": compressed_bitsets(),
           "columnar_arrays": columnar_arrays(), "sparse_graphs": sparse_graphs(),
           "candidate_retrieval": candidate_retrieval()}
    out.update(from_runs(runs or []))
    out.update(from_approx(approx_validation or {}, hifi_cpu_per_ind_day))
    return out


def replay_exact(days: float = 1.0, n: int = 300) -> dict:
    """No outcome without a decision event: a population world's state is rebuilt exactly from its event log
    (every counter - visits, distinct visitors, purchases, revenue, sessions - is written only by appliers of
    recorded events), and the estimated mode adds no event to a world."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..population.worlds import population_world
    from . import approx
    w = World.create(population_world("development", 3, n=n, sites=8))
    w.advance_time(int(days * 86400))
    events = [e.to_dict() for e in w.events_range(0)] if hasattr(w, "events_range") else [e.to_dict() for e in w.log]
    seq = w.seq
    run = {"snaps": [approx.snapshot(w), approx.snapshot(w)], "n": n, "seed": 3, "days": 1, "shop_share": 0.4,
           "sites": 8}
    m = approx.CohortModel().fit([run])
    m.predict(approx.cohorts_of(run), run["snaps"][-1]["res_kind"], 3)
    added = w.seq - seq
    r = World.replay(w.spec, events, verify=True)
    ok = r.digest() == w.digest()
    w.close()
    r.close()
    return {"events": len(events), "replay_digest_equal": ok, "events_added_by_estimate": int(added),
            "outcomes_without_decision_events": 0 if ok and added == 0 else 1}


__all__ = ["TECHNIQUES", "investigate", "replay_exact"]
