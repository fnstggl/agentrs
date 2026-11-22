"""The facility module `rig`: bit-identity with the kernel-free simulator, real ledger money (demand budget,
insufficient funds), seats, tool validation, secrecy of the mechanism, determinism (replay, regeneration, forks) in
both state-recording modes, engine-agnosticism, metrics, admission, the mechanism world families, and a deployed
controller program operating a facility through brokered tool calls."""
from __future__ import annotations

import copy
import json
import math
import sys
import time

import numpy as np
import pytest

import crucible.services  # noqa: F401  (registers identity, bank, admin, code, cloud, wf)
from crucible.core.kernel import World, WorldSpec
from crucible.goals.metrics import METRICS, evaluate_metric
from crucible.mechanisms import base as B
from crucible.mechanisms import module as M
from crucible.mechanisms import scm
from crucible.mechanisms import sim as S
from crucible.sandbox.runner import capabilities
from crucible.worldgen import mech_families as MF
from crucible.worldgen.admission import admit
from crucible.worldgen.generator import FAMILIES, generate

TICK = 3600
A, BOB = "p:agent", "p:bob"
SCM_FAMILIES = ["scm_additive", "scm_delay", "scm_gate", "scm_delay_gate", "scm_invu", "scm_stock_gate",
                "scm_feedback", "scm_regime"]


def make_world(mech, seed=11, holders=("agent",), endow=5_000_000, join=("agent",), **entry):
    cfg = {"rigs": [M.rig_entry(mech, holders=list(holders), **entry)]}
    spec = WorldSpec("mech-test", seed, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"},
                                         {"name": "rig", "config": cfg}], {})
    w = World.create(spec, strict=True)
    for h in join:
        w.system_call("admin.join", {"handle": h, "endowment": endow})
    return w, spec


def act(w, actor, tool_, **args):
    return w.execute(actor, {"tool": tool_, "args": args})


def ok(w, actor, tool_, **args):
    r = act(w, actor, tool_, **args)
    assert r.ok, (tool_, r.error)
    return r.result


def tick(w, n=1):
    w.advance_time(w.now + n * TICK)


def log_dicts(w):
    return copy.deepcopy([e.to_dict() for e in w.events_range(0)])


def bal(w, principal):
    return w.state.tables["accounts"][w.modules["bank"].primary_account(w.state, principal)]["balance"]


def schedule(eng, n, seed=0, every=3, seat=0):
    """A seeded settings schedule {tick: partial settings} that also goes out of range now and then."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(0, n, every):
        out[t] = {lv.name: float(rng.uniform(lv.lo - 0.2 * (lv.hi - lv.lo), lv.hi + 0.2 * (lv.hi - lv.lo)))
                  for lv in eng.levers(seat) if rng.random() < 0.7}
    return out


def same(report, rec):
    """A hosted report against the simulator's record of the same tick."""
    return (report["readings"] == rec["readings"] and report["revenue_cents"] == rec["revenue"]
            and report["cost_cents"] == rec["cost"] and report["score"] == rec["score"]
            and report["status"] == rec["status"] and report["shortfall_cents"] == rec["shortfall"]
            and report["settings"] == rec["settings"])


# ====================================================================== the per-tick generator
def test_tick_rng_fast_path_equals_the_contract_generator():
    assert S.FAST_RNG, "this numpy does not reproduce the fast seeding path (the slow path is still correct)"
    for seed, rig in ((0, "r1"), (11, "r1"), (2**40 + 7, 'odd "id"/é')):
        fast = S.TickRng(seed, rig)
        for t in list(range(130)) + ["init", 5, 4000, 4001, 3, 10**9]:
            a, b = B.tick_rng(seed, rig, t), fast(t)
            assert a.bit_generator.state == b.bit_generator.state
            assert a.standard_normal(3).tobytes() == b.standard_normal(3).tobytes()
    fast = S.TickRng(5, "r1")
    fast.prefetch(40, 20)
    assert fast(47).bit_generator.state == B.tick_rng(5, "r1", 47).bit_generator.state


def test_clip_value_equals_lever_clip():
    eng = B.build_engine(B.make_mechanism("scm_gate", 0))
    for lv in eng.levers(0):
        for v in (lv.lo - 1, lv.hi + 1, lv.lo + 0.37 * (lv.hi - lv.lo), float("nan"), float("inf"), "x", None, True):
            assert S.clip_value(v, lv.lo, lv.hi, lv.default, lv.integer) == lv.clip(v)


def test_simulator_settings_persist_policies_snapshots_and_pickling():
    import pickle
    mech = B.make_mechanism("scm_delay", 4)
    sim = S.MechSim(mech, 7)
    eng = sim.engine
    lv = eng.levers(0)[0]
    assert sim.settings[0] == eng.defaults(0) and sim.t == 0
    sim.step({lv.name: lv.hi})
    sim.step(None)
    assert sim.settings[0][lv.name] == lv.hi and sim.t == 2 and sim.last[0]["settings"][lv.name] == lv.hi
    with pytest.raises(ValueError, match="valid names"):
        sim.step({"nosuch": 1.0})
    # a snapshot is a branch point: both branches see the same hidden noise
    snap = sim.snapshot()
    a = [(r.readings[0], r.cost[0]) for r in (sim.step(None) for _ in range(5))]
    sim.restore(snap)
    assert [(r.readings[0], r.cost[0]) for r in (sim.step(None) for _ in range(5))] == a
    sim.restore(snap)
    all_hi = {x.name: x.hi for x in eng.levers(0)}
    other = [(r.readings[0], r.cost[0]) for r in (sim.step(all_hi if k == 0 else None) for k in range(5))]
    assert [c for _, c in other] != [c for _, c in a]              # ... and differ only through the settings
    # run(): constant, callable and "keep" policies; arrays of shape (ticks, seats)
    r1 = S.MechSim(mech, 7).run({lv.name: lv.hi}, 30)
    r2 = S.MechSim(mech, 7).run(lambda t, last: {lv.name: lv.hi} if t == 0 else None, 30)
    assert r1["net"].shape == (30, 1) and np.array_equal(r1["net"], r2["net"]) and r1["readings"] == r2["readings"]
    assert np.array_equal(r1["net"], r1["revenue"] - r1["cost"]) and set(r1["status"][0]) == {"ok"}
    seen = []
    S.MechSim(mech, 7).run(lambda t, last: seen.append((t, None if last is None else last[0]["status"])), 3)
    assert seen == [(0, None), (1, "ok"), (2, "ok")]
    # reset() starts the same episode again; a pickled simulator continues identically
    sim.reset()
    assert sim.t == 0 and sim.settings[0] == eng.defaults(0)
    first = [sim.step(None).readings[0] for _ in range(4)]
    twin = pickle.loads(pickle.dumps(sim))
    assert [sim.step(None).readings[0] for _ in range(6)] == [twin.step(None).readings[0] for _ in range(6)]
    sim.reset()
    assert [sim.step(None).readings[0] for _ in range(4)] == first


def test_batch_simulator_equals_individual_simulators():
    mechs = [B.make_mechanism(f, s) for f, s in (("scm_additive", 1), ("scm_gate", 2), ("scm_delay", 3))]
    mechs.append(B.make_mechanism("scm_additive", 1, {"seats": 2}))
    batch = S.BatchSim(mechs, world_seeds=[5, 6, 7, 8])
    solo = [S.MechSim(m, s) for m, s in zip(mechs, (5, 6, 7, 8))]
    assert batch.n == 4 and batch.lo.shape == batch.hi.shape == batch.lever_mask.shape == (4, batch.max_levers)
    assert batch.gauge_mask.shape == (4, batch.max_gauges)
    for i, sim in enumerate(solo):
        nl, ng = len(sim.engine.levers(0)), len(sim.engine.gauges(0))
        assert batch.lever_mask[i].sum() == nl and batch.gauge_mask[i].sum() == ng
        assert batch.lever_names[i] == [lv.name for lv in sim.engine.levers(0)]
        assert list(batch.default[i, :nl]) == [lv.default for lv in sim.engine.levers(0)]
    rng = np.random.default_rng(0)
    for t in range(20):
        u = rng.random((4, batch.max_levers))
        out = batch.step(u, normalized=True)
        assert out["readings"].shape == (4, batch.max_gauges) and (out["t"] == t + 1).all()
        for i, sim in enumerate(solo):
            x = {lv.name: lv.lo + u[i, j] * (lv.hi - lv.lo) for j, lv in enumerate(sim.engine.levers(0))}
            r = sim.step([x] + [None] * (sim.n_seats - 1))
            ng = len(sim.engine.gauges(0))
            assert list(out["readings"][i, :ng]) == [r.readings[0][g] for g in batch.gauge_names[i]]
            assert (out["readings"][i, ng:] == 0).all()
            assert (out["revenue"][i], out["cost"][i], out["score"][i]) == (r.revenue[0], r.cost[0], r.score[0])
            assert out["net"][i] == r.revenue[0] - r.cost[0]
    # lever units and mappings are accepted too; None keeps the current settings
    raw = batch.lo + 0.5 * (batch.hi - batch.lo)
    o1 = batch.step(raw)
    for i, sim in enumerate(solo):
        r = sim.step([{lv.name: float(raw[i, j]) for j, lv in enumerate(sim.engine.levers(0))}] +
                     [None] * (sim.n_seats - 1))
        assert o1["revenue"][i] == r.revenue[0]
    o2 = batch.step(None)
    assert [o2["cost"][i] for i in range(4)] == [sim.step(None).cost[0] for sim in solo]
    batch.step([{batch.lever_names[i][0]: float(batch.hi[i, 0])} for i in range(4)])
    assert all(batch.sims[i].settings[0][batch.lever_names[i][0]] == batch.hi[i, 0] for i in range(4))
    # replace a slot with a new task, reset some slots
    batch.replace(1, B.make_mechanism("scm_invu", 9), 99)
    assert batch.sims[1].t == 0 and batch.lever_names[1] == [lv.name for lv in batch.sims[1].engine.levers(0)]
    batch.reset([0])
    assert batch.sims[0].t == 0 and batch.sims[2].t > 0
    with pytest.raises(ValueError):
        S.BatchSim(mechs, world_seeds=[1, 2])


# ====================================================================== hosted facility == simulator
@pytest.mark.parametrize("fam", ["scm_additive", "scm_delay", "scm_gate", "scm_feedback", "scm_regime"])
def test_hosted_world_family_and_simulator_agree_tick_for_tick(fam):
    spec = generate({"family": f"mech_{fam}"}, 23)
    assert [m["name"] for m in spec.modules] == ["identity", "bank", "admin", "rig"]
    w = World.create(spec, strict=True)
    endow = 20_000_000
    w.system_call("admin.join", {"handle": "agent", "endowment": endow})
    sim = MF.sim_of(spec, funds=[endow])
    eng = sim.engine
    n = min(eng.horizon, 120) if fam != "scm_regime" else eng.horizon       # run through the regime switch
    plan = schedule(eng, n, seed=3)
    for t in range(n):
        if t in plan:
            ok(w, A, "rig.set", settings=plan[t])
        tick(w)
        sim.step(plan.get(t))
        rep = ok(w, A, "rig.read")["report"]
        assert rep["t"] == t and same(rep, sim.last[0]), (fam, t, rep, sim.last[0])
    assert bal(w, A) == sim.funds[0]                                # the ledger moved exactly these amounts
    assert w.check_invariants() == []
    hist = ok(w, A, "rig.history", n=n)["reports"]
    assert [r["t"] for r in hist] == list(range(n))
    assert sum(r["net_cents"] for r in hist) == bal(w, A) - endow


def test_throughput_of_both_hosts():
    """Measured, not assumed. CPU time is asserted (the machine is shared; wall clock is reported)."""
    mech = B.make_mechanism("scm_delay", 3)
    sim = S.MechSim(mech, 5)
    lv = sim.engine.levers(0)
    rng = np.random.default_rng(0)
    xs = [{l.name: float(rng.uniform(l.lo, l.hi)) for l in lv} for _ in range(40)]
    n = 6000
    w0, c0 = time.perf_counter(), time.process_time()
    for i in range(n):
        sim.step(xs[i % 40] if i % 5 == 0 else None)
    sim_wall, sim_cpu = n / (time.perf_counter() - w0), n / (time.process_time() - c0)
    w, _ = make_world(mech, endow=10**9)
    m = 1500
    w0, c0 = time.perf_counter(), time.process_time()
    tick(w, m)
    k_wall, k_cpu = m / (time.perf_counter() - w0), m / (time.process_time() - c0)
    print(f"\nMechSim: {sim_wall:.0f} ticks/s wall, {sim_cpu:.0f} ticks/s of CPU time; "
          f"kernel-hosted: {k_wall:.0f} ticks/s wall, {k_cpu:.0f} ticks/s of CPU time", file=sys.stderr)
    assert w.state.tables["rigs"]["r1"]["tick"] == m
    assert sim_cpu >= 5000 and k_cpu >= 400


def test_demand_budget_caps_revenue_and_the_shortfall_is_reported_not_created():
    mech = B.make_mechanism("scm_additive", 4)
    income = 700                                                    # far below what the facility earns
    w, _ = make_world(mech, income_cents_per_tick=income)
    sim = S.MechSim(mech, 11, income_cents_per_tick=income, funds=[5_000_000])
    demand = w.state.tables["rigs"]["r1"]["account"]
    short = 0
    for t in range(40):
        tick(w)
        sim.step(None)
        rep = ok(w, A, "rig.read")["report"]
        assert same(rep, sim.last[0])
        assert rep["revenue_cents"] <= income and rep["revenue_cents"] + rep["shortfall_cents"] == sim.last[0]["earned"]
        short += rep["shortfall_cents"]
        assert w.state.tables["accounts"][demand]["balance"] == 0   # everything it had was paid out, never more
    assert short > 0 and w.check_invariants() == []
    # an empty seat earns nothing for anyone: the budget just accumulates
    w2, _ = make_world(mech, holders=(None,), join=(), income_cents_per_tick=income)
    tick(w2, 10)
    rec = w2.state.tables["rigs"]["r1"]
    assert rec["tick"] == 10 and rec["paid"] == 0 and rec["charged"] == 0
    assert w2.state.tables["accounts"][rec["account"]]["balance"] == 10 * income
    assert w2.state.count("rig_reports") == 0 and w2.check_invariants() == []


def test_a_holder_who_cannot_pay_falls_back_to_defaults_then_idles():
    mech = B.make_mechanism("scm_additive", 4)
    eng = B.build_engine(mech)
    probe = S.MechSim(mech, 11)
    cost_default = int(probe.step(None).cost[0])
    assert cost_default > 0
    costly = max(eng.levers(0), key=lambda lv: next(n["cost"] for n in mech["config"]["nodes"]
                                                    if n.get("public") == lv.name))
    endow = int(1.2 * cost_default)                                 # one tick at defaults, not two
    w, _ = make_world(mech, endow=endow, income_cents_per_tick=300)
    sim = S.MechSim(mech, 11, income_cents_per_tick=300, funds=[endow])
    x = {costly.name: costly.hi}
    ok(w, A, "rig.set", settings=x)
    statuses = []
    for t in range(4):
        tick(w)
        sim.step(x if t == 0 else None)
        rep = ok(w, A, "rig.read")["report"]
        assert same(rep, sim.last[0])
        statuses.append(rep["status"])
    assert statuses == ["fallback", "idle", "idle", "idle"]
    first, idle = ok(w, A, "rig.history", n=4)["reports"][0], ok(w, A, "rig.read")["report"]
    assert first["settings"] == eng.defaults(0) and first["cost_cents"] == cost_default   # ran at defaults
    assert idle["cost_cents"] == 0 and idle["revenue_cents"] == 0 and idle["score"] == 0.0
    assert bal(w, A) == sim.funds[0] >= 0
    # the operator refunds the holder: the seat runs at its own settings again
    w.system_call("admin.endow", {"principal": A, "amount": 10_000_000})
    sim.funds[0] += 10_000_000
    tick(w)
    sim.step(None)
    rep = ok(w, A, "rig.read")["report"]
    assert rep["status"] == "ok" and rep["settings"][costly.name] == costly.hi and same(rep, sim.last[0])
    assert w.check_invariants() == []


# ====================================================================== determinism
class _ScmRecompute(scm.ScmEngine):
    """The same engine declaring a large state: the facility then leaves the state out of its events."""
    engine = "scm_recompute_test"
    state_in_event = False


@pytest.fixture(scope="module", autouse=True)
def _test_engines():
    """Engines that exist only for this module (registered while it runs, removed afterwards)."""
    B.register_engine(_ScmRecompute)
    B.register_engine(_Toy)
    yield
    for name in ("scm_recompute_test", "toy_test"):
        B.ENGINES.pop(name, None)


@pytest.mark.parametrize("mode", ["state_in_event", "recompute"])
def test_replay_regeneration_and_forks_in_both_state_modes(mode):
    mech = B.make_mechanism("scm_stock_gate", 6)
    if mode == "recompute":
        mech = dict(mech, engine="scm_recompute_test")
    w, spec = make_world(mech)
    plan = schedule(B.build_engine(mech), 40, seed=1)
    for t in range(40):
        if t in plan:
            ok(w, A, "rig.set", settings=plan[t])
        tick(w)
    ticks = [e for e in w.log if e.type == "rig.tick"]
    assert len(ticks) == 40 and all(("state" in e.payload) == (mode == "state_in_event") for e in ticks)
    evs = log_dicts(w)
    assert World.replay(spec, evs).digest() == w.digest()
    assert World.replay(spec, json.loads(json.dumps(evs))).digest() == w.digest()   # from stored JSON
    g = World.regenerate(spec, evs)
    assert g.last_hash == w.last_hash and [e.hash for e in g.log] == [e.hash for e in w.log]
    # two forks advanced identically stay identical; a fork that acts differently diverges; the original is intact
    d0 = w.digest()
    f1, f2, f3 = w.fork("f1"), w.fork("f2"), w.fork("f3")
    lv = B.build_engine(mech).levers(0)[0]
    for f in (f1, f2):
        ok(f, A, "rig.set", settings={lv.name: lv.hi})
        tick(f, 5)
    ok(f3, A, "rig.set", settings={lv.name: lv.lo})
    tick(f3, 5)
    assert f1.digest() == f2.digest() != f3.digest() and w.digest() == d0
    assert ok(f1, A, "rig.read")["report"] == ok(f2, A, "rig.read")["report"]
    assert f1.check_invariants() == [] and f3.check_invariants() == []
    # the fork is a controlled counterfactual: same hidden noise, different settings
    sim = S.MechSim(mech, spec.seed, income_cents_per_tick=mech["config"]["income_cents_per_tick"], funds=[5_000_000])
    for t in range(40):
        sim.step(plan.get(t))
    sim.step({lv.name: lv.lo})
    for _ in range(4):
        sim.step(None)
    assert same(ok(f3, A, "rig.read")["report"], sim.last[0])


def test_both_state_modes_hold_the_same_state():
    mech = B.make_mechanism("scm_delay", 2)
    w1, _ = make_world(mech)
    w2, _ = make_world(dict(mech, engine="scm_recompute_test"))
    for _ in range(25):
        tick(w1)
        tick(w2)
        assert w1.state.arrays["rig:r1:h"].tobytes() == w2.state.arrays["rig:r1:h"].tobytes()
        assert ok(w1, A, "rig.read")["report"] == ok(w2, A, "rig.read")["report"]
    size = lambda w: sum(len(json.dumps(e.payload)) for e in w.log if e.type == "rig.tick") / 25   # noqa: E731
    assert size(w2) < size(w1)                                       # leaving the state out makes events smaller


# ====================================================================== engine-agnostic hosting
class _Toy(B.Engine):
    """A second, unrelated engine: two seats, integer / float32 / boolean state, fractional and occasionally
    absurd money figures, no declared demand income. Hosting it uses nothing but the contract."""
    engine = "toy_test"

    def levers(self, seat):
        return [B.Lever(f"p{seat}", 0.0, 5.0, 1.0), B.Lever(f"k{seat}", 0, 3, 0, integer=True)]

    def gauges(self, seat):
        return [B.Gauge(f"q{seat}"), B.Gauge(f"n{seat}")]

    def brief(self, seat):
        return "toy"

    def init_state(self, rng):
        return {"n": np.zeros(2, dtype=np.int64), "x": rng.standard_normal(3).astype(np.float32),
                "flag": np.array([False])}

    def step(self, state, settings, rng, t):
        z = rng.standard_normal(3)
        n = state["n"] + rng.integers(0, 3, size=2)
        x = (0.9 * state["x"] + 0.1 * z).astype(np.float32)
        readings, rev, cost, score = [], [], [], []
        for s in range(2):
            p, k = settings[s][f"p{s}"], settings[s][f"k{s}"]
            readings.append({f"q{s}": float(x[s]) + p, f"n{s}": float(n[s])})
            rev.append(37.5 * p * (1.0 + 0.2 * float(z[s])) if t % 7 else -5.0)       # sometimes negative: clamped
            cost.append(3.25 * p * p + 11.0 * k)
            score.append(p * (k + 1))
        return B.StepResult({"n": n, "x": x, "flag": np.array([bool(t % 2)])}, readings, rev, cost, score,
                            {"secret": 12345.678})

    def truth(self):
        return {"engine": "toy_test", "family": None, "nodes": [], "edges": [], "regimes": [], "motifs": [], "notes": ""}


@pytest.mark.parametrize("in_event", [True, False])
def test_any_engine_can_be_hosted_through_the_contract_alone(in_event):
    _Toy.state_in_event = in_event
    try:
        mech = {"engine": "toy_test", "family": None, "seed": 0, "config": {"n_seats": 2, "horizon": 30}}
        entry = M.rig_entry(mech, holders=["agent", "bob"])
        assert entry["income_cents_per_tick"] == S.suggest_income(mech) > 0     # derived: the engine declares none
        w, spec = make_world(mech, holders=("agent", "bob"), join=("agent", "bob"), endow=1_000_000)
        sim = S.MechSim(mech, spec.seed, income_cents_per_tick=entry["income_cents_per_tick"],
                        funds=[1_000_000, 1_000_000])
        rng = np.random.default_rng(2)
        for t in range(30):
            xa = {"p0": float(rng.uniform(-1, 6)), "k0": float(rng.uniform(0, 3))}
            xb = {"p1": float(rng.uniform(0, 5))}
            ok(w, A, "rig.set", settings=xa)
            ok(w, BOB, "rig.set", settings=xb)
            tick(w)
            sim.step([xa, xb])
            for who, seat in ((A, 0), (BOB, 1)):
                rep = ok(w, who, "rig.read")["report"]
                assert same(rep, sim.last[seat]), (t, seat)
                assert rep["revenue_cents"] >= 0 and isinstance(rep["revenue_cents"], int)
        st = w.state.arrays
        assert st["rig:r1:n"].dtype == np.int64 and st["rig:r1:x"].dtype == np.float32
        assert st["rig:r1:flag"].dtype == np.bool_
        for k in ("n", "x", "flag"):
            assert st[f"rig:r1:{k}"].tobytes() == sim.state[k].tobytes()
        assert bal(w, A) == sim.funds[0] and bal(w, BOB) == sim.funds[1]
        assert w.check_invariants() == []
        evs = log_dicts(w)
        assert World.replay(spec, json.loads(json.dumps(evs))).digest() == w.digest()
        assert World.regenerate(spec, evs).last_hash == w.last_hash
        assert "12345.678" not in json.dumps(evs)                    # private diagnostics are not even recorded
    finally:
        _Toy.state_in_event = True


def test_every_importable_engine_hosts_identically_to_its_simulator():
    """The other engines are written independently against the same contract; whichever are importable now must
    behave under this module exactly as under the simulator."""
    done = []
    for name in S.ensure_engines():
        fams = B.mech_families(engine=name)
        if name in ("toy_test", "scm_recompute_test") or not fams:
            continue
        mech = B.make_mechanism(fams[0], 1)
        eng = B.build_engine(mech)
        entry = M.rig_entry(mech, holders=["agent"])
        w, spec = make_world(mech, endow=10**9)
        held = [True] + [False] * (eng.n_seats - 1)
        sim = S.MechSim(mech, spec.seed, income_cents_per_tick=entry["income_cents_per_tick"],
                        funds=[10**9] + [None] * (eng.n_seats - 1), held=held)
        plan = schedule(eng, 25, seed=4, every=2)
        for t in range(25):
            if t in plan:
                ok(w, A, "rig.set", settings=plan[t])
            tick(w)
            sim.step([plan.get(t)] + [None] * (eng.n_seats - 1))
            assert same(ok(w, A, "rig.read")["report"], sim.last[0]), (name, fams[0], t)
        assert w.check_invariants() == []
        assert World.replay(spec, log_dicts(w)).digest() == w.digest()
        done.append(name)
    assert "scm" in done


# ====================================================================== seats
def two_seat_world(lease=0, **kw):
    mech = B.make_mechanism("scm_additive", 1, {"seats": 2})
    return make_world(mech, holders=("agent",), join=("agent", "bob", "carol"), lease_cents=lease, **kw) + (mech,)


def test_seats_assignment_take_release_and_lease():
    w, spec, mech = two_seat_world(lease=2500)
    eng = B.build_engine(mech)
    seats = ok(w, A, "rig.list")[0]["seats"]
    assert [(s["status"], s["holder"], s["yours"]) for s in seats] == [("held", A, True), ("free", None, False)]
    assert act(w, A, "rig.take").error["code"] == "conflict"         # one seat per principal per facility
    assert act(w, BOB, "rig.read").error["code"] == "forbidden"      # no seat, no report
    assert act(w, BOB, "rig.set", settings={}).error["code"] == "forbidden"
    b0, fees0 = bal(w, BOB), w.state.tables["accounts"]["acct_fees"]["balance"]
    r = ok(w, BOB, "rig.take")
    assert r["seat"] == 1 and r["lease_cents"] == 2500
    assert bal(w, BOB) == b0 - 2500 and w.state.tables["accounts"]["acct_fees"]["balance"] == fees0 + 2500
    assert act(w, "p:carol", "rig.take").error["code"] == "conflict"  # no free seat left
    assert act(w, "p:carol", "rig.take", seat=7).error["code"] == "invalid_action"
    # each holder sees its own seat's levers and reports, never the other's
    sa, sb = ok(w, A, "rig.spec"), ok(w, BOB, "rig.spec")
    assert (sa["seat"], sb["seat"]) == (0, 1)
    assert not {x["name"] for x in sa["levers"]} & {x["name"] for x in sb["levers"]}
    lb = eng.levers(1)[0]
    ok(w, BOB, "rig.set", settings={lb.name: lb.hi})
    tick(w, 3)
    ra, rb = ok(w, A, "rig.read")["report"], ok(w, BOB, "rig.read")["report"]
    assert set(ra["readings"]) == {g.name for g in eng.gauges(0)}
    assert set(rb["readings"]) == {g.name for g in eng.gauges(1)} and rb["settings"][lb.name] == lb.hi
    assert act(w, A, "rig.read", seat=1).error["code"] == "forbidden"
    # both seats match the simulator, including the money
    sim = S.MechSim(mech, spec.seed, income_cents_per_tick=mech["config"]["income_cents_per_tick"],
                    funds=[5_000_000, 5_000_000 - 2500])
    for t in range(3):
        sim.step([None, {lb.name: lb.hi}] if t == 0 else None)
    assert same(ra, sim.last[0]) and same(rb, sim.last[1])
    assert bal(w, A) == sim.funds[0] and bal(w, BOB) == sim.funds[1]
    # release: the seat is free again, its settings are back at the defaults, the old holder is out
    assert ok(w, BOB, "rig.release") == {"rig": "r1", "seat": 1, "released": True}
    assert w.state.tables["rig_seats"]["r1|1"]["settings"] == eng.defaults(1)
    assert act(w, BOB, "rig.read").error["code"] == "forbidden"
    assert act(w, BOB, "rig.release").error["code"] == "not_found"
    b1 = bal(w, BOB)
    tick(w, 2)
    assert bal(w, BOB) == b1                                         # nothing charged or paid after release
    # a new holder does not inherit the previous holder's reports
    ok(w, "p:carol", "rig.take", seat=1)
    assert ok(w, "p:carol", "rig.read")["report"] is None
    assert ok(w, "p:carol", "rig.history", n=50)["reports"] == []
    tick(w)
    assert [r["t"] for r in ok(w, "p:carol", "rig.history", n=50)["reports"]] == [5]
    assert w.check_invariants() == []
    assert World.replay(spec, log_dicts(w)).digest() == w.digest()


def test_a_reserved_seat_waits_for_its_holder():
    mech = B.make_mechanism("scm_additive", 1, {"seats": 2})
    w, spec = make_world(mech, holders=(None, "dana"), join=("bob",))
    assert [s["status"] for s in ok(w, BOB, "rig.list")[0]["seats"]] == ["free", "reserved"]
    assert act(w, BOB, "rig.take", seat=1).error["code"] == "conflict"
    assert ok(w, BOB, "rig.take")["seat"] == 0
    tick(w, 2)
    w.system_call("admin.join", {"handle": "dana", "endowment": 1_000_000})
    assert ok(w, "p:dana", "rig.list")[0]["seats"][1] == {"seat": 1, "status": "held", "holder": "p:dana",
                                                         "yours": True}
    tick(w)
    assert ok(w, "p:dana", "rig.read")["report"]["t"] == 2
    assert World.regenerate(spec, log_dicts(w)).last_hash == w.last_hash


def test_a_holder_can_delegate_operation_with_a_grant():
    w, spec, mech = two_seat_world()
    lv = B.build_engine(mech).levers(0)[0]
    assert act(w, BOB, "rig.set", settings={lv.name: lv.hi}).error["code"] == "forbidden"
    ok(w, A, "id.grant", grantee=BOB, cap="rig.operate", scope="r1")
    assert ok(w, BOB, "rig.set", settings={lv.name: lv.hi})["seat"] == 0
    tick(w)
    assert ok(w, BOB, "rig.read")["report"]["settings"][lv.name] == lv.hi
    assert act(w, "p:carol", "rig.read").error["code"] == "forbidden"
    b = bal(w, BOB)
    tick(w)
    assert bal(w, BOB) == b                                          # the money stays with the holder


# ====================================================================== tool behaviour
def test_set_validates_names_clips_and_applies_from_the_next_tick():
    mech = B.make_mechanism("scm_gate", 0)
    eng = B.build_engine(mech)
    w, _ = make_world(mech)
    names = sorted(lv.name for lv in eng.levers(0))
    r = act(w, A, "rig.set", settings={"nosuch": 1.0, names[0]: 0.0})
    assert r.error["code"] == "invalid_action" and "nosuch" in r.error["message"]
    assert all(n in r.error["message"] for n in names)              # the valid names are listed
    assert w.state.tables["rig_seats"]["r1|0"]["settings"] == eng.defaults(0)     # and nothing was applied
    assert act(w, A, "rig.set", settings={names[0]: "high"}).error["code"] == "invalid_action"
    assert act(w, A, "rig.set", settings={names[0]: float("nan")}).error["code"] == "invalid_action"
    assert act(w, A, "rig.set").error["code"] == "invalid_action"   # settings is required
    assert act(w, A, "rig.set", settings={}, rig="zz").error["code"] == "not_found"
    cont = next(lv for lv in eng.levers(0) if not lv.integer)
    integ = next(lv for lv in eng.levers(0) if lv.integer)
    r = ok(w, A, "rig.set", settings={cont.name: cont.hi + 100, integ.name: integ.lo + 1.4})
    assert r["settings"][cont.name] == cont.hi and r["settings"][integ.name] == integ.lo + 1
    assert sorted(r["clipped"]) == sorted([cont.name, integ.name]) and r["applies_at_tick"] == 0
    assert r["applies_at"] == TICK
    others = {k: v for k, v in r["settings"].items() if k not in (cont.name, integ.name)}
    assert others == {k: v for k, v in eng.defaults(0).items() if k in others}      # a partial mapping
    tick(w)
    first = ok(w, A, "rig.read")
    assert first["report"]["t"] == 0 and first["report"]["settings"] == r["settings"]
    assert first["next_settings"] == r["settings"] and first["ticks_done"] == 1 and first["next_tick_at"] == 2 * TICK
    r2 = ok(w, A, "rig.set", settings={cont.name: cont.lo})        # set after tick 0: only later ticks change
    assert r2["applies_at_tick"] == 1
    assert ok(w, A, "rig.read")["report"]["settings"][cont.name] == cont.hi
    tick(w)
    h = ok(w, A, "rig.history", n=5)["reports"]
    assert [x["settings"][cont.name] for x in h] == [cont.hi, cont.lo]
    assert w.check_invariants() == []


def test_spec_is_the_public_spec_plus_tick_length_and_money_rules():
    mech = B.make_mechanism("scm_delay", 1)
    eng = B.build_engine(mech)
    w, _ = make_world(mech, lease_cents=40)
    sp = ok(w, A, "rig.spec")
    pub = eng.public_spec(0)
    assert {k: sp[k] for k in pub} == pub
    assert sorted(sp) == sorted(list(pub) + ["rig", "seat", "tick_s", "money", "targets"])
    assert sp["tick_s"] == TICK and sp["money"]["lease_cents"] == 40 and sp["money"]["unit"] == "cents"
    assert sp["targets"] == [g["name"] for g in pub["gauges"]] + ["revenue", "cost", "net", "score"]
    assert ok(w, A, "rig.spec", seat=0) == sp
    assert act(w, A, "rig.spec", seat=5).error["code"] == "invalid_action"


def _floats(x, out):
    if isinstance(x, float):
        out.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            _floats(v, out)
    elif isinstance(x, list):
        for v in x:
            _floats(v, out)
    return out


def _keys(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(x, list):
        for v in x:
            _keys(v, out)
    return out


@pytest.mark.parametrize("fam", ["scm_gate", "scm_feedback", "scm_regime"])
def test_nothing_a_participant_can_reach_exposes_the_mechanism(fam):
    mech = B.make_mechanism(fam, 3)
    eng = B.build_engine(mech)
    truth = eng.truth()
    w, spec = make_world(mech, join=("agent", "bob"))
    names = [lv.name for lv in eng.levers(0)]
    gauge = eng.gauges(0)[0].name
    seen, own = [], []                   # everything any call returned / what the facility itself produced
    calls = {
        "rig.list": [{}], "rig.spec": [{}, {"seat": 0}, {"rig": "r1"}],
        "rig.set": [{"settings": {names[0]: 1.0}}, {"settings": {"zzz": 1}}, {"settings": {}}],
        "rig.read": [{}, {"seat": 0}], "rig.history": [{"n": 500}, {}],
        "rig.hypothesize": [{"effects": [{"lever": names[0], "target": gauge, "sign": 1, "lag": 0, "confidence": 0.9}],
                             "predictions": [{"settings": {names[0]: 1.0}, "target": "net", "value": 5.0}]},
                            {"effects": [{"lever": "zzz", "target": gauge, "sign": 1}]},
                            {"effects": [{"lever": names[0], "target": "zzz", "sign": 0}]}],
        "rig.take": [{}, {"seat": 0}], "rig.release": [{}],
    }
    assert set(calls) == {n for n in w.endpoints if n.startswith("rig.")}      # every tool of the host is probed
    for rnd in range(3):
        tick(w, 4)
        for who in (A, BOB):
            for tool, argsets in calls.items():
                if tool in ("rig.release", "rig.take") and who == A and rnd < 2:
                    continue                                         # the agent keeps its seat until the last round
                for args in argsets:
                    own.append(w.execute(who, {"tool": tool, "args": args}).to_dict())
            own.append(w.observe(who).get("rig"))
            own.append([e.to_dict() for e in w.visible_events(who, 0, limit=10**6)[0] if e.type.startswith("rig.")])
            seen.append(w.observe(who))
            seen.append(w.execute(who, {"tool": "world.describe", "args": {"host": "rig"}}).to_dict())
            seen.append([t for t in w.available_tools(who) if t["name"].startswith("rig.")])
            seen.append([e.to_dict() for e in w.visible_events(who, 0, limit=10**6)[0]])
    seen.append(own)
    text = json.dumps(seen)
    assert '"ok": true' in text and "invalid_action" in text and "forbidden" in text
    # 1. no internal node id (their numbering alone would reveal roles and creation order)
    for n in truth["nodes"]:
        assert n["id"] not in text, n["id"]
    # 2. no hidden coefficient: every distinctive number of the config that is not part of the public spec
    public = _floats(eng.public_spec(0), set())
    hidden = {v for v in _floats(mech["config"], set()) if v not in public and len(repr(v)) >= 9}
    assert len(hidden) > 30
    leaked = [v for v in hidden if repr(v) in text]
    assert not leaked, leaked[:5]
    # 3. none of the structures that carry the truth
    assert not _keys(own, set()) & {"truth", "mech", "config", "info", "state", "nodes", "edges", "motifs", "calib",
                                    "overhead", "price", "used", "funded", "income_cents_per_tick", "regime"}
    # 4. the other participant sees nothing of the holder's operation
    bob_events = [e for e in w.visible_events(BOB, 0, limit=10**6)[0] if e.type.startswith("rig.")]
    assert all(e.type == "rig.seat" or e.actor == BOB or BOB in e.vis for e in bob_events)
    assert not [e for e in bob_events if e.type == "rig.report" and e.payload["holder"] == A]
    internal = {e.type for e in w.log if e.vis == "internal"}
    assert {"rig.tick", "rig.init", "rig.claims"} <= internal


def test_hypotheses_are_validated_stored_internally_and_give_no_feedback():
    mech = B.make_mechanism("scm_additive", 2)
    eng = B.build_engine(mech)
    w, spec = make_world(mech, join=("agent", "bob"))
    lever, gauge = eng.levers(0)[0].name, eng.gauges(0)[0].name
    tick(w, 3)
    assert act(w, A, "rig.hypothesize").error["code"] == "invalid_action"           # nothing claimed
    assert act(w, A, "rig.hypothesize", effects=[{"lever": lever, "target": gauge, "sign": 2}]).error["code"] == \
        "invalid_action"
    assert act(w, A, "rig.hypothesize", effects=[{"lever": lever, "target": gauge, "sign": 1, "confidence": 1.5}]) \
        .error["code"] == "invalid_action"
    bad = act(w, A, "rig.hypothesize", effects=[{"lever": lever, "target": "profitt", "sign": 1}])
    assert bad.error["code"] == "invalid_action" and gauge in bad.error["message"] and "net" in bad.error["message"]
    assert act(w, BOB, "rig.hypothesize", effects=[{"lever": lever, "target": gauge, "sign": 1}]).error["code"] == \
        "forbidden"
    claim = {"effects": [{"lever": lever, "target": gauge, "sign": 1, "lag": 2, "confidence": 0.8},
                         {"lever": lever, "target": "net", "sign": -1}],
             "predictions": [{"settings": {lever: eng.levers(0)[0].hi}, "target": "revenue", "value": 1234.5}],
             "note": "first guess"}
    r = w.execute(A, {"tool": "rig.hypothesize", "args": claim})
    assert r.ok and sorted(r.result) == ["claim", "effects", "filed_at_tick", "predictions", "rig", "seat"]
    assert r.result["effects"] == 2 and r.result["predictions"] == 1 and r.result["filed_at_tick"] == 3
    ev = next(e for e in w.log if e.type == "rig.claims")
    assert ev.vis == "internal" and not w.visible(ev, {A})
    stored = w.modules["rig"].claims(w, "r1", A)
    assert len(stored) == 1 and stored[0]["effects"] == claim["effects"] and stored[0]["t"] == 3
    assert stored[0]["predictions"] == claim["predictions"] and stored[0]["by"] == A
    assert World.replay(spec, log_dicts(w)).digest() == w.digest()


# ====================================================================== metrics, invariants, admission
def test_metrics_follow_the_reports_and_the_ledger():
    assert {"rig_net", "rig_score", "rig_ticks"} <= set(METRICS)
    mech = B.make_mechanism("scm_additive", 3)
    w, _ = make_world(mech, history=64)
    plan = schedule(B.build_engine(mech), 30, seed=5)
    start = bal(w, A)
    for t in range(30):
        if t in plan:
            ok(w, A, "rig.set", settings=plan[t])
        tick(w)
    h = ok(w, A, "rig.history", n=30)["reports"]
    args = {"principal": A, "rig": "r1"}
    assert evaluate_metric(w, "rig_ticks", {"rig": "r1"}) == 30
    assert evaluate_metric(w, "rig_net", args) == sum(r["net_cents"] for r in h) == bal(w, A) - start
    assert evaluate_metric(w, "rig_score", args) == pytest.approx(sum(r["score"] for r in h))
    last10 = h[-10:]
    win = dict(args, window_s=10 * TICK)
    assert evaluate_metric(w, "rig_net", win) == sum(r["net_cents"] for r in last10)
    assert evaluate_metric(w, "rig_score", win) == pytest.approx(sum(r["score"] for r in last10))
    assert evaluate_metric(w, "rig_net", dict(args, principal=BOB)) == 0.0
    assert evaluate_metric(w, "rig_net", {"principal": A, "rig": "nope"}) == 0.0
    # the report history is bounded; lifetime totals stay exact and a window beyond it is flagged, not invented
    tick(w, 100)
    assert w.state.count("rig_reports") == 64
    assert evaluate_metric(w, "rig_net", args) == bal(w, A) - start
    assert M.rig_window(w, A, "r1", 100 * TICK)["truncated"] and not M.rig_window(w, A, "r1", 60 * TICK)["truncated"]
    assert M.rig_window(w, A, "r1", 60 * TICK)["ticks"] == 60
    assert w.check_invariants() == []


def test_invariants_catch_broken_money_state_and_settings():
    mech = B.make_mechanism("scm_additive", 3)
    w, _ = make_world(mech)
    tick(w, 5)
    assert w.check_invariants() == []
    st = w.state
    lv = B.build_engine(mech).levers(0)[0]
    acct = st.tables["rigs"]["r1"]["account"]
    st.tables["accounts"][acct]["balance"] += 7                      # money from nowhere in the demand account
    assert any("demand balance" in e for e in w.check_invariants())
    st.tables["accounts"][acct]["balance"] -= 7
    st.tables["rigs"]["r1"]["charged"] += 1
    assert any("charged" in e for e in w.check_invariants())
    st.tables["rigs"]["r1"]["charged"] -= 1
    good = st.arrays["rig:r1:h"].copy()
    st.arrays["rig:r1:h"][0, 0] = math.nan
    assert any("not finite" in e for e in w.check_invariants())
    st.arrays["rig:r1:h"] = good * 0 + 1e15
    assert any("exceeds the bound" in e for e in w.check_invariants())
    st.arrays["rig:r1:h"] = good
    st.tables["rig_seats"]["r1|0"]["settings"][lv.name] = lv.hi + 1
    assert any("outside" in e for e in w.check_invariants())
    st.tables["rig_seats"]["r1|0"]["settings"][lv.name] = lv.default
    assert w.check_invariants() == []


def test_module_passes_the_admission_harness():
    mech = B.make_mechanism("scm_gate", 2, {"seats": 2})
    cfg = {"rigs": [M.rig_entry(mech, tick_s=1800, holders=["adm0"], lease_cents=300)]}
    rep = admit(["rig"], steps=300, configs={"rig": cfg})
    assert rep.passed, (rep.checks, rep.errors[:5])
    assert set(rep.checks) >= {"schemas", "no_crashes", "invariants", "replay", "regeneration", "snapshot_fork",
                               "visibility", "protected_hooks"}
    # with no facility configured the module is inert but still admissible
    assert admit(["rig"], steps=60).passed


def test_bad_configuration_is_refused():
    mech = B.make_mechanism("scm_additive", 0)
    for bad in ({"id": "a|b", "mech": mech}, {"id": "r1", "mech": {"engine": "nope", "config": {}}},
                {"id": "r1", "mech": mech, "tick_s": 0}, {"id": "r1", "mech": mech, "seats": [{}, {}, {}]}):
        with pytest.raises(Exception, match="rig"):
            M.Rig({"rigs": [bad]})
    with pytest.raises(Exception, match="duplicate"):
        M.Rig({"rigs": [M.rig_entry(mech), M.rig_entry(mech)]})


# ====================================================================== world families
def test_world_families_exist_for_every_mechanism_family():
    names = MF.register_mech_world_families()
    for f in SCM_FAMILIES:
        for suffix, mods in (("", MF.MECH_MODULES), ("_tools", MF.MECH_TOOL_MODULES)):
            wname = f"mech_{f}{suffix}"
            assert wname in names and FAMILIES[wname].split == B.MECH_FAMILIES[f].split
            spec = generate({"family": wname}, 5)
            assert [m["name"] for m in spec.modules] == mods and spec.params["family_split"] == FAMILIES[wname].split
            assert spec.hash() == generate({"family": wname}, 5).hash() != generate({"family": wname}, 6).hash()
            mech = MF.mech_of(spec)
            assert mech["family"] == f and mech["engine"] == "scm"
            rig = next(m for m in spec.modules if m["name"] == "rig")["config"]["rigs"][0]
            assert rig["id"] == "r1" and rig["seats"] == [{"holder": "agent"}]
            assert rig["income_cents_per_tick"] == mech["config"]["income_cents_per_tick"]
    assert set(MF.mech_world_families("train", tools=False)) >= {"mech_scm_additive", "mech_scm_delay", "mech_scm_gate"}
    assert "mech_scm_regime" in MF.mech_world_families("test") and "mech_scm_invu" in MF.mech_world_families("val")
    before = dict(FAMILIES)
    assert MF.register_mech_world_families() == names               # idempotent
    assert all(FAMILIES[k] is v for k, v in before.items())


def test_world_family_overrides():
    spec = generate({"family": "mech_scm_delay", "overrides": {"mech_seed": 7, "tick_s": 600, "holders": [None],
                                                                 "lease_cents": 5, "history": 10}}, 1)
    rig = next(m for m in spec.modules if m["name"] == "rig")["config"]["rigs"][0]
    assert rig["mech"] == B.make_mechanism("scm_delay", 7) and rig["tick_s"] == 600
    assert rig["seats"] == [{"holder": None}] and rig["lease_cents"] == 5 and rig["history"] == 10
    given = B.make_mechanism("scm_gate", 9)
    spec = generate({"family": "mech_scm_delay", "overrides": {"mech": given}}, 1)
    assert MF.mech_of(spec) == given
    two = generate({"family": "mech_scm_additive", "overrides": {"mech_overrides": {"seats": 2},
                                                                    "holders": ["agent", "bob"]}}, 1)
    assert MF.mech_of(two)["config"]["n_seats"] == 2
    w = World.create(two, strict=True)
    for h in ("agent", "bob"):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
    assert [s["holder"] for s in ok(w, A, "rig.list")[0]["seats"]] == [A, BOB]


# ====================================================================== a deployed controller operates the facility
CONTROLLER = '''
import crucible_sdk as sdk


def tick(args, caller):
    """Read the latest report of the seat and set one lever from one gauge reading."""
    report = sdk.call("rig.read")["report"]
    if report is None:
        return {"set": None}
    spec = sdk.call("rig.spec")
    lever = next(l for l in spec["levers"] if l["name"] == args["lever"])
    frac = (abs(report["readings"][args["gauge"]]) * 7.0) % 1.0
    out = sdk.call("rig.set", settings={lever["name"]: lever["min"] + frac * (lever["max"] - lever["min"])})
    return {"set": out["settings"][lever["name"]], "after_tick": report["t"]}
'''
caps = capabilities()


@pytest.mark.sandbox
@pytest.mark.skipif(caps["landlock_abi"] < 1 or not caps["seccomp"] or not caps["root"],
                    reason=f"needs Landlock+seccomp and root to drop privileges: {caps}")
def test_a_deployed_controller_program_operates_the_facility_through_tools():
    spec = generate({"family": "mech_scm_additive_tools"}, 5)
    eng = B.build_engine(MF.mech_of(spec))
    lever = next(lv for lv in eng.levers(0) if not lv.integer)
    gauge = eng.gauges(0)[0].name

    def world(with_controller: bool):
        w = World.create(spec, strict=True)
        w.system_call("admin.join", {"handle": "agent", "endowment": 3_000_000})
        if with_controller:
            acct = w.modules["bank"].primary_account(w.state, A)
            manifest = json.dumps({"entry": "service.py", "endpoints": {"tick": {
                "handler": "tick", "params": {"type": "object", "additionalProperties": True}}}})
            repo = ok(w, A, "code.create_repo", name="ctl")["repo"]
            ok(w, A, "code.commit", repo=repo, files={"crucible.json": manifest, "service.py": CONTROLLER})
            inst = ok(w, A, "cloud.rent", provider="o:cloud1", cores=1, account=acct)["instance"]
            assert ok(w, A, "cloud.deploy", repo=repo, host="ctl", instance=inst)["status"] == "live"
            # without the grant the program is refused: it acts as its own service principal
            assert ok(w, A, "ctl.tick", lever=lever.name, gauge=gauge)["status"] == 500
            ok(w, A, "id.grant", grantee="svc:ctl", cap="rig.operate", scope="r1")
            ok(w, A, "wf.trigger", tool="ctl.tick", args={"lever": lever.name, "gauge": gauge}, every_s=TICK)
        return w

    w, plain = world(True), world(False)
    try:
        n = 10
        tick(w, n)
        tick(plain, n)
        h = ok(w, A, "rig.history", n=n)["reports"]
        assert len(h) == n
        for prev, cur in zip(h, h[1:]):                             # the program's rule, tick after tick
            frac = (abs(prev["readings"][gauge]) * 7.0) % 1.0
            assert cur["settings"][lever.name] == lever.clip(lever.lo + frac * (lever.hi - lever.lo))
        sets = [e for e in w.log if e.type == "rig.set"]
        assert len(sets) == n and {e.payload["by"] for e in sets} == {"svc:ctl"}   # nobody else touched a lever
        trig = ok(w, A, "wf.triggers")[0]
        assert trig["runs"] == n and trig["fails"] == 0
        # the same world without the program stays at defaults and earns differently
        hp = ok(plain, A, "rig.history", n=n)["reports"]
        assert all(r["settings"] == eng.defaults(0) for r in hp)
        assert h[0]["readings"] == hp[0]["readings"] and h[-1]["readings"] != hp[-1]["readings"]
        assert sum(r["net_cents"] for r in h) != sum(r["net_cents"] for r in hp)
        assert bal(w, A) < 3_000_000 + sum(r["net_cents"] for r in h)               # compute rent was paid too
        assert w.check_invariants() == []
        evs = log_dicts(w)
        assert World.replay(spec, evs).digest() == w.digest()
        g = World.regenerate(spec, evs)                              # recorded program outcomes are re-injected
        assert g.last_hash == w.last_hash
        g.close()
    finally:
        w.close()
        plain.close()
