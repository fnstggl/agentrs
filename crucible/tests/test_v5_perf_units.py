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


def test_rollout_world_env_defaults_to_observation_only_regression_def_m_5():
    """DEF-M-5 (audit M_performance finding 4): the 4.0 farm's WorldEnv returned the account-balance change as its
    reward by default (12345.0 for an exogenous deposit with no action). The default is observation_only: the scalar
    reward is 0.0 while the balance change stays observable; the shaped reward exists only when selected."""
    import crucible
    from crucible.scale.rollout import WorldEnv
    crucible.load_all_modules()
    out = {}
    for mode in (None, "legacy_shaped_reward"):
        env = WorldEnv({"family": "tiny", "seed": 0, "overrides": {"n_consumers": 40}}, reward_mode=mode)
        env.reset()
        before = env._balance()
        env.world.system_call("admin.endow", {"principal": env.actor, "amount": 12345})
        obs, r, done, info = env.step(None)
        out[mode] = (env.reward_mode, r, env._balance() - before)
    assert out[None][0] == "observation_only" and out[None][1] == 0.0 and out[None][2] >= 12345
    assert out["legacy_shaped_reward"][1] == float(out["legacy_shaped_reward"][2]) >= 12345


def test_semantic_batches_are_counted_per_reply():
    class E:
        def __init__(self, t, p):
            self.type, self.payload, self.vis, self.actor = t, p, "internal", "sys"

        def body(self):
            return {"type": self.type, "payload": self.payload}
    ec = bench5.EventCounter(count_bytes=False)
    ec(E("ext", {"outcomes": [{"kind": "infer.semantic.choose", "out": {"output": [{"choice": 0}] * 5}},
                              {"kind": "infer.space.items", "out": {"output_ref": "sha256:x"}}]}))
    ec(E("ext", {"outcomes": [{"kind": "infer.semantic.generate", "out": {"output": [{"text": "a"}] * 3}}]}))
    assert ec.semantic_calls == 2 and ec.semantic_rows == 8 and ec.infer["infer.space.items"] == 1


def test_outcomes_handle_resources_created_during_the_run():
    """Web resources appear during a run (the res table grows): daily reach is padded, not a shape error."""
    from crucible.scale import approx
    def snap(n_res, sess):
        return {"class": np.ones(4, np.int64), "segment": np.zeros(4, np.int64), "n_sessions": np.asarray(sess),
                "n_purchases": np.zeros(4, np.int64), "spent_cents": np.zeros(4, np.int64),
                "res_kind": np.zeros(n_res, np.int64), "res_visits": np.arange(n_res), "res_uniq": np.arange(n_res),
                "res_purchases": np.zeros(n_res, np.int64), "res_revenue_cents": np.zeros(n_res, np.int64)}
    o = approx.outcomes({"snaps": [snap(2, [0, 0, 0, 0]), snap(3, [1, 0, 2, 0]), snap(5, [2, 1, 2, 0])]})
    assert o["days"] == 2 and len(o["reach_top"]) == 2 and o["audience_shares"][0] == 1.0


def test_batch_sizes_are_read_from_the_recorded_outputs():
    """bench5 sizes every class-2/3 recorded call from its recorded output (V.4 uses the exact in-world batches, not
    per-run means): real AdaptivePolicy and StrategicGroup calls recorded the way the population records them."""
    import crucible
    crucible.load_all_modules()
    import numpy as np
    from crucible.core.kernel import World, WorldSpec
    from crucible.population.adaptive import AdaptivePolicy
    from crucible.population.strategic import HIST, OBS_DIM, StrategicGroup
    from crucible.substrate import inference
    pol = AdaptivePolicy(0, 12, 9, 8, hidden=32, seed=1)
    grp = StrategicGroup({"group": "g", "arch": "mlp", "hidden": 32, "learning": False, "code": 0}, seed=2)
    w = World.create(WorldSpec("bench-sizes", 1, [{"name": m, "config": {}} for m in
                                                  ("identity", "bank", "admin", "artifacts")]))
    ec = bench5.EventCounter(count_bytes=False)
    w.listeners.append(ec)
    rng = np.random.default_rng(0)

    def run(ctx):
        for n in (3, 3, 7):
            x = {"obs": rng.normal(size=(n, 12)).astype(np.float32), "mem": rng.normal(size=(n, 8)).astype(np.float32),
                 "cand": rng.normal(size=(n, 5, 9)).astype(np.float32)}
            inference.recorded(ctx, "pop.adaptive.0", "adaptive:0", pol.weights_hash(), x,
                               lambda x=x: {"scores": pol.scores(x["obs"], x["mem"], x["cand"])[0]})
        O = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
        H = rng.normal(size=(2, HIST, OBS_DIM)).astype(np.float32)
        inference.recorded(ctx, "pop.strategic.g", "strategic:g", grp.weights_hash(), {"obs": O, "hist": H},
                           lambda: grp.logits(O, H))
        xd = {"obs": rng.normal(size=(4, 12)).astype(np.float32), "mem": rng.normal(size=(4, 8)).astype(np.float32)}
        cd = rng.normal(size=(4, 5, 9)).astype(np.float32)      # day review: memory step only (pop.adaptive_day.*)
        inference.recorded(ctx, "pop.adaptive_day.0", "adaptive:0", pol.weights_hash(), xd,
                           lambda: {"memory": pol.scores(xd["obs"], xd["mem"], cd)[1]})
        inference.recorded(ctx, "other.model", "m", "h", {"x": O}, lambda: {"y": O})   # not a class-2/3 call
    w._run_system("bench.sizes", run)
    ec.size_batches(w)
    assert dict(ec.batch_sizes["class2"]) == {3: 2, 7: 1, 4: 1} and dict(ec.batch_sizes["class3"]) == {2: 1}
    assert ec.model_refs == []
    b = bench5.batching(w, ec)
    assert b["class2"]["sizes"] == {"3": 2, "4": 1, "7": 1} and b["class3"]["sizes"] == {"2": 1}
    w.close()


def test_resident_store_size_is_measured_without_a_second_copy(tmp_path):
    """bench5 measures the resident store as the database pages in use after a WAL checkpoint (no VACUUM, which
    needs a second copy of the store on disk): deleting rows frees pages, which the measure excludes, and it never
    exceeds the file."""
    from crucible.storage.sqlite import SQLiteStore
    db = str(tmp_path / "s.sqlite")
    st = SQLiteStore(db)
    c = st._conn()
    c.execute("CREATE TABLE junk (k INTEGER PRIMARY KEY, v BLOB)")
    c.executemany("INSERT INTO junk VALUES (?, ?)", [(i, os.urandom(4000)) for i in range(2000)])
    c.commit()
    bench5._settle(st)
    full = bench5._live_bytes(st)
    assert full >= 2000 * 4000 and full <= os.path.getsize(db)
    assert not os.path.exists(db + "-wal") or os.path.getsize(db + "-wal") == 0     # checkpoint folded the WAL
    c.execute("DELETE FROM junk WHERE k >= 500")
    c.commit()
    bench5._settle(st)
    after = bench5._live_bytes(st)
    assert after < 0.4 * full and os.path.getsize(db) >= full                     # freed pages excluded, file kept
    st.close()
