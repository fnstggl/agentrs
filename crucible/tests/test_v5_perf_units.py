"""Unit tests of the performance machinery (workstream M): the NumPy backend equals the torch network, behavioural
equivalence and backend choice, the estimated cohort mode (labelled, never a world event), target checks and the
event counter of the benchmark harness. The measurements themselves are gate tests (tests/v5/test_gate_V.py ...)."""
import json
import os

import numpy as np

from crucible.scale import approx, backends, bench5


def test_numpy_forward_matches_the_torch_network():
    import torch
    from crucible.population import nets
    net = nets.AdaptiveNet(20, 11, 8, 32)
    nets.init_params(net, np.random.default_rng(3))
    rng = np.random.default_rng(0)
    o, m, c = (rng.normal(0, 1, s).astype(np.float32) for s in ((5, 20), (5, 8), (5, 7, 11)))
    with torch.no_grad():
        s_t, m_t = net(torch.from_numpy(o), torch.from_numpy(m), torch.from_numpy(c))
    params = {k: v.numpy() for k, v in net.state_dict().items()}
    s_n, m_n = backends.numpy_forward(params, o, m, c)
    assert np.allclose(s_t.numpy(), s_n, atol=1e-5) and np.allclose(m_t.numpy(), m_n, atol=1e-5)


def test_equivalence_and_choice():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, (200, 9))
    mask = rng.random((200, 9)) < 0.8
    mask[:, -1] = True
    same = backends.equivalence(s, s.copy(), mask)
    assert same["paired_agreement"] == 1.0 and same["tv_distance"] == 0.0 and same["equivalent"]
    noisy = backends.equivalence(s, s + rng.normal(0, 0.5, s.shape), mask)
    assert not noisy["equivalent"] and noisy["paired_agreement"] < 0.99
    bench = {"rows": [{"backend": "a", "B": 8, "available": True, "equivalent": True, "wall_s": 2.0},
                      {"backend": "b", "B": 8, "available": True, "equivalent": True, "wall_s": 1.0},
                      {"backend": "c", "B": 8, "available": True, "equivalent": False, "wall_s": 0.5}]}
    ch = backends.choose(bench)
    assert ch["8"]["best"] == "b" and ch["8"]["rejected_not_equivalent"] == ["c"]


def _run(n=300, days=4, seed=0, act=0.4):
    rng = np.random.default_rng(seed)
    cls = np.where(rng.random(n) < 0.9, 1, 2)
    seg = rng.integers(0, 3, n)
    kinds = np.array([0] * 6 + [1] * 4)
    snaps, ns, np_, rev, vis, uniq = [], np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(10, np.int64), \
        np.zeros(10, np.int64), np.zeros(10, np.int64)
    for d in range(days + 1):
        snaps.append({"t": d * 86400, "class": cls, "segment": seg, "n_sessions": ns.copy(), "n_purchases": np_.copy(),
                      "spent_cents": np.zeros(n, np.int64), "res_kind": kinds, "res_visits": vis.copy(),
                      "res_uniq": uniq.copy(), "res_purchases": np.where(kinds == 1, vis // 10, 0),
                      "res_revenue_cents": rev.copy()})
        a = rng.random(n) < act
        ns = ns + a * rng.integers(1, 4, n)
        np_ = np_ + (a & (rng.random(n) < 0.05))
        v = rng.multinomial(int(a.sum() * 2), np.ones(10) / 10)
        vis = vis + v
        uniq = np.minimum(n, uniq + v // 2)
        rev = rev + np.where(kinds == 1, v * 30, 0)
    return {"n": n, "seed": seed, "days": days, "shop_share": 0.4, "sites": 10, "snaps": snaps}


def test_the_estimated_mode_is_labelled_and_measures_errors():
    fit = _run(seed=0)
    m = approx.CohortModel().fit([fit])
    tgt = _run(n=900, seed=1)
    est = m.predict(approx.cohorts_of(tgt), tgt["snaps"][-1]["res_kind"], 4, seed=5)
    assert est["estimated"] is True and est["mode"] == approx.MODE_ID and est["N"] == 900
    obs = approx.outcomes(tgt)
    err = approx.errors(est, obs, 1.0, 2.0)
    assert set(err) == set(approx.QUANTITIES)
    assert all(np.isfinite(v) and v >= 0 for v in err.values())
    assert abs(est["active_share"] - obs["active_share"]) < 0.1      # the routine activity level is recovered


def test_targets_are_checked_against_the_target_world():
    tg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "docs", "v5", "performance", "targets.json")))
    agg = {"civ/development": {"sim_days_per_cpu_hour": {"min": 70.0}, "generation_cpu_s": {"max": 10.0},
                               "bytes_per_individual_day": {"max": 600.0}, "snapshot_bytes": {"max": 1}, "snapshot_s": {"max": 1}}}
    res = bench5.check_targets(agg, tg, world="civ")
    by = {(r["id"], r["profile"]): r for r in res}
    assert by[("T1", "development")]["met"] and not by[("T6", "development")]["met"]
    assert not by[("T2", "standard")]["met"] and by[("T2", "standard")]["measured"] is None


def test_the_event_counter_counts_decisions_and_public_events():
    class E:
        def __init__(self, t, p, vis="internal", actor="p1"):
            self.type, self.payload, self.vis, self.actor = t, p, vis, actor
            self.seq, self.t, self.cmd, self.hash, self.via = 1, 0, None, "", None

        def body(self):
            return {"type": self.type, "payload": self.payload}
    ec = bench5.EventCounter()
    ec(E("cols.batch", {"kind": "pop.session", "n": 5, "extra": {"sessions": 4, "activations": 5}}))
    ec(E("cols.batch", {"kind": "x.buy", "n": 3, "extra": {"decision": {"model_id": "pop.adaptive:0"}}}))
    ec(E("forum.post", {}, vis="public", actor="p2"))
    assert ec.decisions == 7 and ec.sessions == 4 and ec.activations == 5 and ec.public == 1
    assert ec.decisions_by["pop.adaptive:0"] == 3 and ec.event_bytes > 0
