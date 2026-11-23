"""The agent-based market engine (`crucible.mechanisms.abm`), driven directly through `Engine.step` with
`base.tick_rng` (no kernel, no simulator module).

What is established here:

* contract: registration, JSON config that rebuilds the engine, opaque public surface;
* purity: no mutation of the input state, fresh arrays, bit-identical repetition, no global randomness;
* money: budgets never overspent, revenue never above customer income, exact running totals;
* the mechanisms the families claim (awareness sets, network-only word of mouth, capacity, the trust gate,
  segments, bundles, the attention pool) hold customer by customer;
* incumbents adapt: price to lost share, capacity to queues, exit when unprofitable and a replacement later;
* `truth()` is true of the simulation: paired interventions with common random numbers reproduce the declared
  reachability, minimal lags and signs for every single-seat family;
* calibration over 20 seeds per family: defaults earn about nothing, reference settings earn clearly, random
  settings lose, no corner of the lever box is near the best found, and the best settings differ across families
  the way the causal story predicts;
* the strategic structure of the two-seat families.

Money is reported in units of N * p0 per tick (customers times the reference price), so numbers compare across
mechanisms. Simulation tests use 300 customers; the throughput test uses 500.
"""
from __future__ import annotations

import functools
import itertools
import json
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest

from crucible.mechanisms import abm  # noqa: F401  (importing registers the engine and its families)
from crucible.mechanisms.base import (ENGINES, MECH_FAMILIES, build_engine, copy_state, make_mechanism,
                                      mech_families, tick_rng)

SINGLE = ["abm_utility", "abm_wom", "abm_congestion", "abm_trust_gate", "abm_network_fx", "abm_segments"]
TWO = ["abm_duopoly", "abm_commons", "abm_complement", "abm_duopoly_fx"]
ALL = SINGLE + TWO
N_SIM = 300
SEEDS20 = range(20)

# Reference settings per family in natural units: price in p0, quality as effort, outreach in N*p0 per tick,
# capacity in N, referral in p0. Found by random search plus coordinate refinement on seeds 0-2 (scratch work,
# not shipped); they are the "best found" the calibration tests compare against.
REF = {
    "abm_utility": dict(price=1.0, quality=0.16, outreach=0.011),
    "abm_wom": dict(price=1.05, quality=0.65, outreach=0.001, referral=0.45),
    "abm_congestion": dict(price=0.96, quality=0.17, capacity=0.2, outreach=0.008),
    "abm_trust_gate": dict(price=1.0, quality=0.58, outreach=0.014),
    "abm_network_fx": dict(price=0.85, quality=0.2, outreach=0.04),
    "abm_segments": dict(price=1.0, quality=0.58, outreach=0.007, referral=0.3),
    "abm_duopoly": dict(price=1.1, quality=0.1, outreach=0.008),
    "abm_commons": dict(price=1.05, quality=0.25, outreach=0.004),
    "abm_complement": dict(price=0.52, quality=0.35, outreach=0.010),
    "abm_duopoly_fx": dict(price=0.85, quality=0.16, outreach=0.027),
}


# ====================================================================== helpers
EVIDENCE: dict[str, dict] = {}


def note(key: str, **vals) -> None:
    """Record a measured number; the whole table is printed once at the end (shown with `pytest -s`)."""
    EVIDENCE[key] = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in vals.items()}


@pytest.fixture(scope="module", autouse=True)
def _print_evidence():
    yield
    if EVIDENCE:
        print("\n=== measured (net in N*p0 per tick) ===")
        for key in sorted(EVIDENCE):
            print(f"{key}: {json.dumps(EVIDENCE[key])}")


@functools.lru_cache(maxsize=None)
def eng_of(family: str, seed: int, n: int = N_SIM, over: str = "{}"):
    o = json.loads(over)
    o.setdefault("n_customers", n)
    return build_engine(make_mechanism(family, seed, o))


def unit(eng) -> float:
    return float(eng.config["n_customers"] * eng.config["p0"])


def nat(eng, **kw) -> dict:
    """Dial settings from natural multiples (see REF)."""
    n, p0 = eng.config["n_customers"], eng.config["p0"]
    scale = {"price": p0, "quality": 1.0, "outreach": n * p0, "capacity": n, "referral": p0}
    return eng.encode({k: v * scale[k] for k, v in kw.items()})


def rollout(eng, wseed: int, policy, ticks: int | None = None, state=None, t0: int = 0, keep: bool = False):
    """Drive the engine exactly as a host would: init from tick_rng(.., "init"), tick t from tick_rng(.., t)."""
    st = eng.init_state(tick_rng(wseed, "r1", "init")) if state is None else state
    ticks = eng.horizon if ticks is None else ticks
    score = np.zeros((ticks, eng.n_seats))
    res_all = []
    for i in range(ticks):
        t = t0 + i
        res = eng.step(st, policy(t) if callable(policy) else policy, tick_rng(wseed, "r1", t), t)
        st = res.state
        score[i] = res.score
        if keep:
            res_all.append(res)
    return SimpleNamespace(score=score, state=st, res=res_all)


def net(eng, wseed: int, policy, ticks: int | None = None) -> np.ndarray:
    """Mean net per tick per seat, in N*p0."""
    return rollout(eng, wseed, policy, ticks).score.mean(0) / unit(eng)


def both(eng, settings: dict) -> list[dict]:
    return [settings] * eng.n_seats


@functools.lru_cache(maxsize=None)
def _net_nat(family: str, seed: int, profile: str, ticks: int | None) -> tuple:
    eng = eng_of(family, seed)
    return tuple(net(eng, 7000 + seed, [nat(eng, **s) for s in json.loads(profile)], ticks))


def net_nat(family: str, seed: int, per_seat: list[dict], ticks: int | None = None) -> np.ndarray:
    """Mean net per tick per seat (N*p0) of constant natural settings, on mechanism `seed` with world seed
    7000 + seed. Cached: many tests compare against the same reference run."""
    return np.array(_net_nat(family, seed, json.dumps(per_seat, sort_keys=True), ticks))


def mean_se(x) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)


def random_settings(eng, rng) -> dict:
    return {lv.name: float(rng.uniform(lv.lo, lv.hi)) for lv in eng.levers(0)}


# ====================================================================== contract
def test_engine_and_families_registered():
    assert ENGINES["abm"] is abm.AbmEngine
    assert abm.AbmEngine.state_in_event is False
    assert abm.AbmEngine.claims == ("default_zero", "random_loses", "corners_bad")
    want = {"abm_utility": ("train", "base", 1), "abm_wom": ("train", "base", 1),
            "abm_congestion": ("train", "base", 1), "abm_trust_gate": ("val", "structural", 1),
            "abm_network_fx": ("test", "structural", 1), "abm_segments": ("test", "compositional", 1),
            "abm_duopoly": ("train", "base", 2), "abm_commons": ("val", "structural", 2),
            "abm_complement": ("val", "structural", 2), "abm_duopoly_fx": ("test", "structural", 2)}
    assert sorted(mech_families(engine="abm")) == sorted(want)
    for name, (split, transfer, seats) in want.items():
        fam = MECH_FAMILIES[name]
        assert (fam.split, fam.transfer, fam.seats, fam.engine) == (split, transfer, seats, "abm")
        assert fam.motifs and len(fam.doc) > 60
        assert build_engine(make_mechanism(name, 3, {"n_customers": 120})).n_seats == seats
    # structural families really switch on a rule no training family has
    flags = {f: {k for k in ("wom", "congest", "gate", "nfx", "bundle", "commons") if eng_of(f, 0).config["rules"][k]}
             for f in ALL}
    train = set().union(*(flags[f] for f in ALL if MECH_FAMILIES[f].split == "train"))
    for f in ALL:
        if MECH_FAMILIES[f].transfer == "structural":
            assert flags[f] - train, f
        if MECH_FAMILIES[f].transfer == "compositional":
            assert flags[f] <= train and len(eng_of(f, 0).config["pop"]["segments"]) == 2


@pytest.mark.parametrize("family", ALL)
def test_config_is_plain_json_and_rebuilds_the_engine(family):
    mech = make_mechanism(family, 11, {"n_customers": 200})
    assert make_mechanism(family, 11, {"n_customers": 200}) == mech           # deterministic in (family, seed)
    assert make_mechanism(family, 12, {"n_customers": 200})["config"] != mech["config"]
    text = json.dumps(mech, allow_nan=False)
    assert json.loads(text) == mech                                           # nothing but plain JSON types

    def plain(x):
        if isinstance(x, dict):
            return all(isinstance(k, str) and plain(v) for k, v in x.items())
        if isinstance(x, list):
            return len(x) <= 16 and all(plain(v) for v in x)                  # no adjacency, no per-customer data
        return x is None or type(x) in (str, int, float, bool)
    assert plain(mech["config"]) and len(text) < 8000
    a, b = build_engine(mech), build_engine(json.loads(text))
    assert np.array_equal(a._k.esrc, b._k.esrc) and np.array_equal(a._k.edst, b._k.edst)   # network from its seed
    ra = rollout(a, 5, both(a, nat(a, **REF[family])), 25, keep=True)
    rb = rollout(b, 5, both(b, nat(b, **REF[family])), 25, keep=True)
    assert state_equal(ra.state, rb.state) and np.array_equal(ra.score, rb.score)
    assert [r.readings for r in ra.res] == [r.readings for r in rb.res]


def test_overrides_change_only_what_they_name():
    a = make_mechanism("abm_utility", 4, {})["config"]
    b = make_mechanism("abm_utility", 4, {"n_customers": 1000, "horizon": 77})["config"]
    assert (b["n_customers"], b["horizon"]) == (1000, 77)
    for key in a:
        if key not in ("n_customers", "horizon"):
            assert a[key] == b[key], key
    c = make_mechanism("abm_utility", 4, {"rules": {"phi": 0.5}, "n_inc": 1})["config"]
    assert c["rules"]["phi"] == 0.5 and len(c["inc"]) == 1 and c["levers"][0]["name"] == a["levers"][0]["name"]
    with pytest.raises(ValueError):
        make_mechanism("abm_utility", 4, {"n_customers": 5})
    with pytest.raises(ValueError):
        abm.AbmEngine.generate(np.random.default_rng(0), {"kind": "nope"})


@pytest.mark.parametrize("family", ALL)
def test_public_surface_is_opaque(family):
    words = ("price", "quality", "capacity", "outreach", "referral", "advert", "customer", "market", "rival",
             "rating", "review", "queue", "complain", "demand", "order", "network", "segment", "bundle", "noop",
             "firm", "incumbent", "volume", "cent")
    pat = re.compile(r"^[bdfgjklmnprstvz][aeiou]\d\d$")
    eng = eng_of(family, 0)
    for seat in range(eng.n_seats):
        spec = eng.public_spec(seat)
        names = [lv["name"] for lv in spec["levers"]] + [g["name"] for g in spec["gauges"]]
        assert len(set(names)) == len(names) and all(pat.match(nm) for nm in names)
        # every lever looks the same from outside: a dial 0..100 with no description
        assert all((lv["min"], lv["max"], lv["integer"], lv["doc"]) == (0.0, 100.0, False, "") for lv in spec["levers"])
        assert all(g["doc"] == "" for g in spec["gauges"])
        text = json.dumps(spec).lower()
        assert not [w for w in words if w in text], [w for w in words if w in text]
        assert set(eng.defaults(seat)) == {lv["name"] for lv in spec["levers"]}
    # the order of roles on the panel is seeded: the same role sits at different positions across mechanisms
    pos = {tuple(lv["role"] for lv in eng_of(family, s).config["levers"]).index("price") for s in range(12)}
    gpos = {tuple(g["role"] for g in eng_of(family, s).config["gauges"]).index("orders") for s in range(12)}
    assert len(pos) > 1 and len(gpos) > 2
    # so is the direction of every dial: the same role runs forwards in some mechanisms and backwards in others
    for role in (lv["role"] for lv in eng.config["levers"] if not lv["role"].startswith("noop")):
        turns = {next(lv["flip"] for lv in eng_of(family, s).config["levers"] if lv["role"] == role) for s in range(12)}
        rests = {next(lv["default"] for lv in eng_of(family, s).config["levers"] if lv["role"] == role)
                 for s in range(12)}
        assert turns == {True, False} and len(rests) > 1, (role, turns, rests)
    # a flipped dial is the same lever read backwards
    plain = build_engine(make_mechanism(family, 0, {"n_customers": N_SIM, "flip": False}))
    assert not any(lv["flip"] for lv in plain.config["levers"])
    a = rollout(eng, 6, both(eng, nat(eng, **REF[family])), 15)
    b = rollout(plain, 6, both(plain, nat(plain, **REF[family])), 15)
    # (the two decodings can differ by float rounding, i.e. by one cent or one part in 1e15 of a natural value)
    assert np.allclose(a.score, b.score, rtol=0, atol=3e-4 * unit(eng))
    assert np.mean(a.state["cur"] == b.state["cur"]) > 0.97
    # nothing private in a reading: exactly the public gauge names, all finite numbers
    res = eng.step(eng.init_state(tick_rng(1, "r1", "init")), both(eng, eng.defaults(0)), tick_rng(1, "r1", 0), 0)
    for seat in range(eng.n_seats):
        assert set(res.readings[seat]) == {g.name for g in eng.gauges(seat)}
        assert all(isinstance(v, float) and np.isfinite(v) for v in res.readings[seat].values())


def test_population_is_heterogeneous_and_network_generated():
    for n in (300, 3000):
        eng = eng_of("abm_wom", 2, n)
        st = eng.init_state(tick_rng(0, "r1", "init"))
        tr = st["traits"]
        assert tr.shape == (abm.N_TRAITS, n) and len(abm.TRAIT_ROWS) == abm.N_TRAITS
        for row in (abm.T_INC, abm.T_WTP, abm.T_WP, abm.T_WQ, abm.T_HAB, abm.T_NEED, abm.T_STD, abm.T_SUS):
            assert tr[row].std() > 0.05 * abs(tr[row].mean()) > 0, abm.TRAIT_ROWS[row]
        assert st["aware"].shape == (eng._k.F, n) and st["mem_q"].shape == (eng._k.O, n)
        assert st["cur"].shape == st["tenure"].shape == st["budget"].shape == (n,)
        res = eng.step(st, [eng.defaults(0)], tick_rng(0, "r1", 0), 0)
        assert res.state["budget"].shape == (n,)
    # three generators, all symmetric, loop-free and deterministic in the stored seed
    for kind, extra in (("ring", {"k": 4, "rewire": 0.1}), ("pa", {"m": 4}), ("blocks", {"blocks": 6, "k_in": 4})):
        net_cfg = {"kind": kind, "seed": 9, **extra}
        src, dst, block = abm._net_edges(net_cfg, 600)
        src2, dst2, _ = abm._net_edges(net_cfg, 600)
        assert np.array_equal(src, src2) and np.array_equal(dst, dst2)
        assert np.all(src != dst)
        pairs = set(zip(src.tolist(), dst.tolist()))
        assert len(pairs) == src.size and all((b, a) in pairs for a, b in pairs)
        deg = np.bincount(src, minlength=600)
        assert 5 <= deg.mean() <= 11
        if kind == "ring":
            gap = np.minimum((src - dst) % 600, (dst - src) % 600)
            assert (gap <= 4).mean() > 0.8                       # mostly the lattice, a few rewired ties
        if kind == "pa":
            assert deg.max() > 4 * deg.mean()                    # hubs
        if kind == "blocks":
            assert (block[src] == block[dst]).mean() > 0.7       # communities
    assert not np.array_equal(abm._net_edges({"kind": "pa", "seed": 10, "m": 4}, 600)[1],
                              abm._net_edges({"kind": "pa", "seed": 9, "m": 4}, 600)[1])


def test_calibration_register_is_honest():
    cal = abm.CALIBRATION
    assert "not calibrated" in cal["_scope"]["status"]
    rules = [k for k in cal if not k.startswith("_")]
    assert len(rules) >= 14
    for k in rules:
        assert cal[k]["status"] in ("plausible hypothesis", "inspired by a published model"), k
        assert ("reference" in cal[k]) == (cal[k]["status"] != "plausible hypothesis"), k
        assert cal[k]["note"]
    assert "Bass 1969" in cal["word_of_mouth"]["reference"] and "Granovetter 1978" in cal["trust_gate"]["reference"]
    assert eng_of("abm_utility", 0).truth()["calibration"] is cal


# ====================================================================== purity
@pytest.mark.parametrize("family", ALL)
def test_step_is_pure_and_bit_identical(family):
    eng = eng_of(family, 1)
    st0 = eng.init_state(tick_rng(3, "r1", "init"))
    assert state_equal(st0, eng.init_state(tick_rng(3, "r1", "init")))
    rng = np.random.default_rng(0)
    st = st0
    for t in range(12):                                  # reach a busy state with random settings
        st = eng.step(st, [random_settings(eng, rng) for _ in range(eng.n_seats)], tick_rng(3, "r1", t), t).state
    sets = [random_settings(eng, rng) for _ in range(eng.n_seats)]
    frozen, frozen_sets = copy_state(st), [dict(s) for s in sets]
    np.random.seed(1)
    r1 = eng.step(st, sets, tick_rng(3, "r1", 12), 12)
    assert state_equal(st, frozen) and sets == frozen_sets           # inputs untouched
    np.random.seed(2)
    np.random.random(100)                                            # global randomness is irrelevant
    r2 = eng.step(st, sets, tick_rng(3, "r1", 12), 12)
    r3 = build_engine(json.loads(json.dumps({"engine": "abm", "config": eng.config}))).step(
        copy_state(st), sets, tick_rng(3, "r1", 12), 12)             # another instance, copied state
    for r in (r2, r3):
        assert state_equal(r1.state, r.state)
        assert (r1.readings, r1.revenue, r1.cost, r1.score) == (r.readings, r.revenue, r.cost, r.score)
        assert json.dumps(r1.info, sort_keys=True) == json.dumps(r.info, sort_keys=True)
    for key, arr in r1.state.items():                                # fresh arrays, nothing aliased
        assert isinstance(arr, np.ndarray)
        assert not any(np.shares_memory(arr, old) for old in st.values()), key
        assert not any(np.shares_memory(arr, other) for k2, other in r1.state.items() if k2 != key), key
    r4 = eng.step(st, sets, tick_rng(3, "r1", 13), 12)               # a different draw does change the outcome
    assert not state_equal(r1.state, r4.state)
    json.dumps(r1.info)                                              # private diagnostics are plain data too


@pytest.mark.parametrize("family", ["abm_utility", "abm_commons"])
def test_settings_are_robust_to_garbage(family):
    eng = eng_of(family, 0)
    st = eng.init_state(tick_rng(0, "r1", "init"))
    d = eng.defaults(0)
    ref = eng.step(st, both(eng, d), tick_rng(0, "r1", 0), 0)
    name = next(lv["name"] for lv in eng.config["levers"] if lv["role"] == "price")
    for bad in (float("nan"), "x", None):
        r = eng.step(st, both(eng, {**d, name: bad}), tick_rng(0, "r1", 0), 0)
        assert state_equal(r.state, ref.state)                       # unusable -> the default position
    lo = eng.step(st, both(eng, {**d, name: -1e9}), tick_rng(0, "r1", 0), 0)
    assert state_equal(lo.state, eng.step(st, both(eng, {**d, name: 0.0}), tick_rng(0, "r1", 0), 0).state)
    missing = eng.step(st, both(eng, {}), tick_rng(0, "r1", 0), 0)
    assert state_equal(missing.state, ref.state)
    assert eng.clip_settings(0, {name: 1e9, "zz99": 3.0})[name] == 100.0


# ====================================================================== money
@pytest.mark.parametrize("family", ALL)
def test_money_is_conserved_and_budgets_never_overspent(family):
    eng = eng_of(family, 2)
    k = eng._k
    rng = np.random.default_rng(7)
    st = eng.init_state(tick_rng(9, "r1", "init"))
    income_cap = float(st["traits"][abm.T_INC].sum())
    assert st["totals"][0] == st["budget"].sum() and st["totals"][1] == 0.0
    sets = [random_settings(eng, rng) for _ in range(eng.n_seats)]
    seat_rev = 0.0
    for t in range(90):
        if t % 6 == 0:                                   # hold each random setting for a few ticks
            sets = [random_settings(eng, rng) for _ in range(eng.n_seats)]
            if t % 18 == 0:                              # and visit cheap, attractive settings too, so people buy
                sets = [nat(eng, **{**REF[family], "price": 0.4}) for _ in range(eng.n_seats)]
        before = st
        res = eng.step(st, sets, tick_rng(9, "r1", t), t)
        st = res.state
        b = st["budget"]
        assert b.min() >= 0.0 and np.all(b <= st["traits"][abm.T_BCAP])          # never overspent, never over cap
        assert np.array_equal(b, np.round(b))                                    # whole cents
        tot = st["totals"]
        assert tot[0] - tot[1] == b.sum()                                        # exact running identity
        assert tot[1] <= tot[0] and tot[2] <= tot[1] and tot[3] == t + 1
        credited = tot[0] - before["totals"][0]
        assert 0.0 <= credited <= income_cap and credited == res.info["income_credited"]
        paid = (np.minimum(before["budget"] + before["traits"][abm.T_INC], before["traits"][abm.T_BCAP]) - b).sum()
        assert paid == sum(res.info["revenue_all"]) == tot[1] - before["totals"][1]   # customers paid = firms got
        for s in range(eng.n_seats):
            assert res.revenue[s] == res.info["price"][s] * res.info["served"][s] == res.info["revenue_all"][s]
            assert res.cost[s] >= 0.0 and res.score[s] == res.revenue[s] - res.cost[s]
            assert res.revenue[s] == round(res.revenue[s]) and res.cost[s] == round(res.cost[s])
            seat_rev += res.revenue[s]
        assert all(0 <= u <= o for u, o in zip(res.info["unserved"], res.info["orders"]))
        if k.finite_cap:
            assert all(s_ <= c for s_, c in zip(res.info["served"], res.info["capacity"]))
    assert seat_rev == st["totals"][2] > 0.0
    # the bound a host can rely on: all revenue ever paid <= all income ever credited <= endowment + ticks * income
    assert st["totals"][1] <= st["totals"][0] <= eng.revenue_bound(90) <= 2.0 * income_cap + 90 * income_cap
    assert income_cap == k.income_total == eng.income_cents_per_tick


def test_without_an_endowment_revenue_never_exceeds_income_paid_so_far():
    """`pop.endow_ticks = 0`: customers start with nothing, so a host that funds one tick of customer income before
    each tick can always pay the revenue."""
    eng = eng_of("abm_utility", 0, N_SIM, json.dumps({"pop": {"endow_ticks": 0.0}}))
    st = eng.init_state(tick_rng(0, "r1", "init"))
    assert st["budget"].sum() == 0.0 and st["totals"][0] == 0.0 and eng.revenue_bound(0) == 0.0
    s = [nat(eng, price=0.5, quality=0.5, outreach=0.05)]
    paid = 0.0
    for t in range(40):
        res = eng.step(st, s, tick_rng(0, "r1", t), t)
        paid += sum(res.info["revenue_all"])
        assert paid <= (t + 1) * eng.income_cents_per_tick == eng.revenue_bound(t + 1)
        st = res.state
    assert paid > 0.2 * 40 * eng.income_cents_per_tick       # and people do spend a good part of it


def test_market_with_many_incumbents_keeps_every_invariant():
    """Five options exercise the general choice path (the running maximum is used for up to four)."""
    eng = eng_of("abm_utility", 3, N_SIM, json.dumps({"n_inc": 4}))
    k = eng._k
    assert k.O == 5 and len(eng.gauges(0)) == 6 + 4
    r = rollout(eng, 0, [nat(eng, price=0.9, quality=0.3, outreach=0.02)], 40, keep=True)
    sold = np.zeros(k.F)
    for res in r.res:
        st = res.state
        assert st["budget"].min() >= 0.0 and st["totals"][0] - st["totals"][1] == st["budget"].sum()
        has = st["cur"] < k.O
        assert st["aware"][st["cur"][has], np.flatnonzero(has)].all()
        sold += res.info["served"]
    assert (sold > 0).all() and sold[0] > 300              # every firm sells; the seat wins a real share


def test_cost_is_what_the_settings_consume():
    eng = eng_of("abm_congestion", 0)
    k, st = eng._k, eng.init_state(tick_rng(0, "r1", "init"))
    s = nat(eng, price=1.0, quality=0.5, capacity=0.3, outreach=0.02)
    r = eng.step(st, [s], tick_rng(0, "r1", 0), 0)
    want = k.c_q * 0.5 + k.c_cap * round(0.3 * k.N) + round(0.02 * k.N * k.p0) + k.c_unit * r.info["served"][0]
    assert abs(r.cost[0] - want) <= 1.0
    eng = eng_of("abm_wom", 0)
    k = eng._k
    hi = nat(eng, **{**REF["abm_wom"], "referral": 1.0})
    lo = nat(eng, **{**REF["abm_wom"], "referral": 0.0})
    w = rollout(eng, 3, [hi], 60)
    a = eng.step(w.state, [hi], tick_rng(3, "r1", 60), 60)
    b = eng.step(w.state, [lo], tick_rng(3, "r1", 60), 60)
    # the reward is paid per referred first purchase: a whole number of rewards, and none when the reward is zero
    base_cost = k.c_q * 0.65 + round(0.001 * k.N * k.p0) + k.c_unit * a.info["served"][0]
    n_rewards = (a.cost[0] - base_cost) / k.p0
    assert abs(n_rewards - round(n_rewards)) < 0.01
    assert abs(b.cost[0] - (k.c_q * 0.65 + round(0.001 * k.N * k.p0) + k.c_unit * b.info["served"][0])) <= 1.0
    total_rewards = 0.0
    st = eng.init_state(tick_rng(3, "r1", "init"))
    for t in range(60):
        r = eng.step(st, [hi], tick_rng(3, "r1", t), t)
        total_rewards += r.cost[0] - (k.c_q * 0.65 + round(0.001 * k.N * k.p0) + k.c_unit * r.info["served"][0])
        st = r.state
    ever = int(st["tried"][0].sum())
    assert 0.3 * ever <= total_rewards / k.p0 <= ever + 0.5          # most first purchases came by referral


# ====================================================================== the rules, customer by customer
@pytest.mark.parametrize("family", ALL)
def test_customers_only_buy_from_firms_they_know(family):
    eng = eng_of(family, 3)
    k = eng._k
    r = rollout(eng, 4, both(eng, nat(eng, **{**REF[family], "price": 0.5})), 40, keep=True)
    bought = 0
    for res in r.res:
        cur, aware = res.state["cur"], res.state["aware"]
        for o in range(k.O):
            mine = cur == o
            bought += int(mine.sum())
            for f in {k.oa[o], k.ob[o]}:                 # every member firm of the option is known to its buyer
                assert aware[f][mine].all()
    assert bought > 1000
    # with no way to hear of the seat's firm, nobody ever buys from it, however cheap and good it is
    over = json.dumps({"rules": {"org": 0.0, "beta0": 0.0}, "seat": {**eng.config["seat"], "aware0": 0.0}})
    blind = eng_of(family, 3, N_SIM, over)
    s = nat(blind, **{**REF[family], "price": 0.35, "quality": 1.0, "outreach": 0.0})
    rr = rollout(blind, 4, both(blind, s), 40, keep=True)
    assert all(res.revenue == [0.0] * blind.n_seats for res in rr.res)
    assert not rr.state["aware"][:blind.n_seats].any()
    loud = rollout(blind, 4, both(blind, {**s, **nat(blind, outreach=0.02)}), 40)
    heard = loud.state["aware"][0].mean()            # outreach reaches whoever is receptive to it
    assert heard > {"wom": -1.0, "segments": 0.3}.get(blind.config["kind"], 0.5), heard


def test_word_of_mouth_travels_only_along_edges_from_satisfied_customers():
    over = json.dumps({"rules": {"org": 0.0}})
    eng = eng_of("abm_wom", 1, N_SIM, over)
    k = eng._k
    nbrs = [k.edst[k.esrc == i] for i in range(k.N)]
    s = nat(eng, price=0.9, quality=0.8, outreach=0.0, referral=0.5)
    st = eng.init_state(tick_rng(2, "r1", "init"))
    seeded = int(st["aware"][0].sum())
    assert 0 < seeded < 0.1 * k.N
    new_total = 0
    for t in range(80):
        res = eng.step(st, [s], tick_rng(2, "r1", t), t)
        newly = np.flatnonzero(res.state["aware"][0] & ~st["aware"][0])
        advocates = st["happy"] & (st["cur"] == 0)
        for i in newly:                                  # every new knower has a satisfied customer next door
            assert advocates[nbrs[i]].any()
        new_total += len(newly)
        st = res.state
    assert new_total > 0.4 * k.N                         # and that channel alone carries the firm to most people
    # the same firm with poor quality stays where it was seeded: no satisfied customers, no telling
    poor = rollout(eng, 2, [nat(eng, price=0.9, quality=0.0, outreach=0.0, referral=0.5)], 80)
    assert poor.state["aware"][0].sum() < 0.5 * st["aware"][0].sum()
    # in abm_utility the network carries nothing: awareness is identical whatever the neighbours do
    u = eng_of("abm_utility", 1)
    assert "happy" not in u.init_state(tick_rng(2, "r1", "init")) and u.config["rules"]["wom"] is False


def test_capacity_rations_service_and_waits_cause_churn_and_complaints():
    eng = eng_of("abm_congestion", 0)
    k = eng._k
    g = {role: v["public"] for role, v in eng.roles()["gauges"].items()}
    tight = nat(eng, price=1.0, quality=0.4, capacity=0.03, outreach=0.03)
    ample = nat(eng, price=1.0, quality=0.4, capacity=0.5, outreach=0.03)
    a = rollout(eng, 6, [tight], 80, keep=True)
    b = rollout(eng, 6, [ample], 80, keep=True)
    cap = round(0.03 * k.N)
    for res in a.res:
        assert res.info["served"][0] <= cap == res.info["capacity"][0]
        assert res.readings[0][g["queue"]] == res.info["orders"][0] - res.info["served"][0] == res.info["unserved"][0]
        assert res.revenue[0] == res.info["price"][0] * res.info["served"][0]        # the unserved pay nothing
    assert all(res.info["unserved"][0] == 0 for res in b.res)
    late = slice(40, 80)
    queue = np.mean([r.info["unserved"][0] for r in a.res[late]])
    assert queue > 5
    assert np.mean([r.readings[0][g["complaints"]] for r in a.res[late]]) > \
        3 * np.mean([r.readings[0][g["complaints"]] for r in b.res[late]])
    # customers who were turned away remember it and stop ordering: demand itself is lower, not just service
    mean_of = lambda run, key: np.mean([r.info[key][0] for r in run.res[late]])          # noqa: E731
    assert mean_of(a, "orders") < 0.6 * mean_of(b, "orders")
    assert a.state["mem_w"][0].mean() > 0.05 and b.state["mem_w"][0].max() == 0.0
    assert mean_of(a, "active") < 0.4 * mean_of(b, "active")


def test_incumbents_add_capacity_after_queues_and_shed_idle_capacity():
    eng = eng_of("abm_congestion", 1)
    k = eng._k
    w = rollout(eng, 2, [eng.defaults(0)], 30)
    f = k.S
    st = copy_state(w.state)
    st["firm"][abm.R_CAP, f] = 3.0                       # squeeze one incumbent
    caps, uns = [], []
    for t in range(30, 45):
        res = eng.step(st, [eng.defaults(0)], tick_rng(2, "r1", t), t)
        caps.append(res.state["firm"][abm.R_CAP, f])
        uns.append(res.info["unserved"][f])
        st = res.state
    assert uns[0] > 0 and caps[0] > 3.0 and caps[-1] > 5 * 3.0
    assert all(c2 > c1 for c1, c2, q in zip([3.0] + caps, caps, uns) if q > 0)      # every queue adds capacity
    st = copy_state(w.state)
    st["firm"][abm.R_CAP, f] = float(k.N)                # and one with far too much sheds it tick by tick
    res = eng.step(st, [eng.defaults(0)], tick_rng(2, "r1", 30), 30)
    assert res.info["unserved"][f] == 0 and res.state["firm"][abm.R_CAP, f] < k.N


def test_trust_gate_blocks_trial_until_rating_and_reviews_pass():
    eng = eng_of("abm_trust_gate", 0)
    k = eng._k
    st = eng.init_state(tick_rng(8, "r1", "init"))
    tr = st["traits"]
    pioneers = (tr[abm.T_GTH] == 0.0) & (tr[abm.T_GM] == 0.0)
    assert 0.02 < pioneers.mean() < 0.12
    s = nat(eng, **REF["abm_trust_gate"])
    first_by_gate = 0
    orders = []
    for t in range(70):
        f = st["firm"]
        rating = (f[abm.R_RS] + k.rating_r0 * k.rating_n0) / (f[abm.R_RN] + k.rating_n0)
        res = eng.step(st, [s], tick_rng(8, "r1", t), t)
        for firm in range(k.F):                          # every first purchase passed that customer's gate
            new = np.flatnonzero(res.state["tried"][firm] & ~st["tried"][firm])
            assert np.all(tr[abm.T_GTH][new] <= rating[firm]) and np.all(tr[abm.T_GM][new] <= f[abm.R_RN, firm])
            if firm == 0:
                first_by_gate += int((~pioneers[new]).sum())
        if t < 3:
            assert pioneers[res.state["cur"] == 0].all()                 # cold start: only pioneers at first
        orders.append(res.info["orders"][0])
        st = res.state
    assert first_by_gate > 30 and np.mean(orders[50:]) > 5 * max(1.0, np.mean(orders[:8]))
    # poor quality never opens the gate: the rating stays under almost every threshold and only pioneers buy
    poor = rollout(eng, 8, [nat(eng, price=1.1, quality=0.05, outreach=0.014)], 70, keep=True)
    assert poor.res[-1].info["rating"][0] < np.quantile(tr[abm.T_GTH][~pioneers], 0.05)
    assert int((poor.state["tried"][0] & ~pioneers).sum()) == 0


def test_network_effects_raise_value_with_neighbours_usage():
    eng = eng_of("abm_network_fx", 0)
    k = eng._k
    w = rollout(eng, 1, [nat(eng, price=0.9, quality=0.4, outreach=0.03)], 5)
    nbr_of = [k.edst[k.esrc == i] for i in range(k.N)]
    s = [nat(eng, price=1.0, quality=0.4, outreach=0.03)]

    def orders_with_everyone_on(option: int) -> list:
        st = copy_state(w.state)
        st["aware"][:] = True
        st["cur"][:] = option
        st["tenure"][:] = 0.0                            # no habit: isolate the neighbours' effect
        return eng.step(st, s, tick_rng(1, "r1", 5), 5).info["orders"]

    on_seat, on_rival = orders_with_everyone_on(0), orders_with_everyone_on(1)
    assert on_seat[0] > 3 * max(1, on_rival[0])          # same prices, same draws: only the installed base differs
    assert on_rival[1] > 3 * max(1, on_seat[1])
    # and adoption clusters: after a run, users of the seat have far more seat-using neighbours than non-users
    r = rollout(eng, 1, [nat(eng, price=0.9, quality=0.4, outreach=0.02)], 100)
    cur = r.state["cur"]
    share = np.array([np.mean(cur[nb] == 0) if len(nb) else 0.0 for nb in nbr_of])
    users = cur == 0
    assert 0.05 < users.mean() < 0.99
    assert share[users].mean() > share[~users].mean() + 0.15


def test_segments_obey_two_different_training_rules():
    eng = eng_of("abm_segments", 0, N_SIM, json.dumps({"rules": {"org": 0.0}, "seat": {"aware0": 0.08}}))
    k = eng._k
    seg = k.traits[abm.T_SEG]
    assert 0.3 < (seg == 0).mean() < 0.7
    assert np.all(k.traits[abm.T_SUS][seg == 0] == 0.0) and np.all(k.traits[abm.T_SUS][seg == 1] > 0.0)
    assert k.traits[abm.T_RECV][seg == 0].mean() > 20 * k.traits[abm.T_RECV][seg == 1].mean()
    same = (seg[k.esrc] == seg[k.edst]).mean()
    assert same > 0.7                                    # segments live in different communities
    quiet = rollout(eng, 5, [nat(eng, price=0.95, quality=0.6, outreach=0.0, referral=0.4)], 120)
    loud = rollout(eng, 5, [nat(eng, price=0.95, quality=0.6, outreach=0.01, referral=0.4)], 120)
    init = eng.init_state(tick_rng(5, "r1", "init"))["aware"][0]
    aw_q, aw_l = quiet.state["aware"][0], loud.state["aware"][0]
    # without outreach segment A (utility rule) learns nothing new; segment B (word of mouth) still spreads
    assert aw_q[seg == 0].sum() <= init[seg == 0].sum()
    assert aw_q[seg == 1].mean() > 0.3
    # outreach is what reaches segment A
    assert aw_l[seg == 0].mean() > 0.5 > 5 * aw_q[seg == 0].mean()


def test_bundle_needs_both_components_known_and_pays_both_seats():
    eng = eng_of("abm_complement", 0)
    k = eng._k
    assert k.O == k.F + 1 and (k.oa[0], k.ob[0]) == (0, 1)
    s = nat(eng, **REF["abm_complement"])
    r = rollout(eng, 3, [s, s], 60, keep=True)
    bundles = 0
    for res in r.res:
        cur, aware = res.state["cur"], res.state["aware"]
        b = cur == 0
        assert (aware[0] & aware[1])[b].all()
        bundles += int(b.sum())
        # a bundle is one unit for each seat at that seat's own price
        assert res.revenue[0] == res.info["price"][0] * res.info["served"][0]
        assert res.revenue[1] == res.info["price"][1] * res.info["served"][1]
    assert bundles > 2000
    last = r.res[-1]
    solo0, solo1 = int((last.state["cur"] == 1).sum()), int((last.state["cur"] == 2).sum())
    assert last.info["active"][0] - solo0 == last.info["active"][1] - solo1 == int((last.state["cur"] == 0).sum())
    # weakest link with a mismatch penalty: raising one side's quality above the other's lowers bundle sales
    matched = rollout(eng, 3, [nat(eng, price=0.55, quality=0.3, outreach=0.012)] * 2, 60, keep=True)
    lopsided = rollout(eng, 3, [nat(eng, price=0.55, quality=0.3, outreach=0.012),
                                nat(eng, price=0.55, quality=1.0, outreach=0.012)], 60, keep=True)
    sold = lambda run: sum(r_.info["served"][0] for r_ in run.res[20:])          # noqa: E731
    assert sold(lopsided) < 0.9 * sold(matched)


def test_attention_pool_is_depleted_by_outreach_and_recovers():
    eng = eng_of("abm_commons", 0)
    heavy = nat(eng, price=1.05, quality=0.25, outreach=0.03)
    none = nat(eng, price=1.05, quality=0.25, outreach=0.0)
    r = rollout(eng, 2, [heavy, none], 40, keep=True)        # one seat alone drains the pool both draw on
    pool = [float(x.state["pool"][0]) for x in r.res]
    assert pool[0] < 1.0 and pool[-1] < 0.65 and all(b <= a + 1e-12 for a, b in zip(pool, pool[1:]))
    rec = rollout(eng, 2, [none, none], 60, state=r.state, t0=40, keep=True)
    pool2 = [float(x.state["pool"][0]) for x in rec.res]
    assert all(b >= a for a, b in zip(pool2, pool2[1:])) and pool2[-1] > 0.95
    assert "pool" not in eng_of("abm_duopoly", 0).init_state(tick_rng(0, "r1", "init"))
    # a drained pool means customers shop less often and outreach reaches fewer: same market, same draws, only the
    # stock differs
    full = copy_state(r.state)
    full["pool"][0] = 1.0
    a = eng.step(r.state, [heavy, heavy], tick_rng(2, "r1", 40), 40)
    b = eng.step(full, [heavy, heavy], tick_rng(2, "r1", 40), 40)
    assert a.info["volume"] < 0.9 * b.info["volume"]
    assert a.state["aware"][1].sum() < b.state["aware"][1].sum()


# ====================================================================== incumbents
def test_incumbent_price_reacts_to_lost_share():
    for family in ("abm_utility", "abm_congestion"):
        drops = []
        for sd in range(4):
            eng = eng_of(family, sd)
            k = eng._k
            quiet = rollout(eng, 3, [eng.defaults(0)], 60, keep=True)
            attack = rollout(eng, 3, [nat(eng, price=0.75, quality=0.6, outreach=0.04, capacity=0.5)], 60, keep=True)
            pq = np.mean([np.mean(r.info["price"][k.S:]) for r in quiet.res[20:]])
            pa = np.mean([np.mean(r.info["price"][k.S:]) for r in attack.res[20:]])
            assert pa < pq
            assert min(min(r.info["price"][k.S:]) for r in attack.res) >= k.p_floor     # never below their floor
            drops.append((pq - pa) / k.p0)
        assert np.mean(drops) > 0.03, (family, drops)


def test_incumbent_exits_when_unprofitable_and_a_replacement_enters_after_the_delay():
    eng = eng_of("abm_utility", 0)
    k = eng._k
    g = {role: v["public"] for role, v in eng.roles()["gauges"].items()}
    w = rollout(eng, 1, [eng.defaults(0)], 10)
    f, j = k.S, 0
    slot = k.rivals[0].index(f)
    st = copy_state(w.state)
    st["firm"][abm.R_PEMA, f] = -1e12                    # a firm deep in the red
    had = int((st["cur"] == f).sum())
    assert had > 10 and st["aware"][f].mean() > 0.4
    res = eng.step(st, [eng.defaults(0)], tick_rng(1, "r1", 10), 10)
    assert res.info["events"] == [{"t": 10, "what": "exit", "firm": f}]
    s1 = res.state
    assert s1["firm"][abm.R_ALIVE, f] == 0.0 and s1["firm"][abm.R_TIMER, f] == k.delay
    assert not (s1["cur"] == f).any() and not s1["aware"][f].any()       # customers released, firm forgotten
    assert np.all(s1["mem_q"][f] == k.q_prior)
    # while it is gone nobody orders from it, its public price reads 0 (after the lag), and then a new firm enters
    t, st, entered = 11, s1, None
    while entered is None and t < 11 + k.delay + 3:
        res = eng.step(st, [eng.defaults(0)], tick_rng(1, "r1", t), t)
        if res.info["events"]:
            entered = t
        else:
            assert res.info["orders"][f] == 0 and res.info["alive"][f] == 0.0
            if t >= 11 + k.L:
                assert res.readings[0][g[f"rival_price_{slot}"]] == -1.0
        st = res.state
        t += 1
    assert entered == 10 + k.delay and res.info["events"] == [{"t": entered, "what": "entry", "firm": f}]
    firm = st["firm"]
    assert firm[abm.R_ALIVE, f] == 1.0 and firm[abm.R_AGE, f] == 0.0
    assert k.ent_p[0] * k.p0 - 1 <= firm[abm.R_PRICE, f] <= k.ent_p[1] * k.p0 + 1
    assert k.ent_q[0] <= firm[abm.R_QUAL, f] <= k.ent_q[1]
    assert firm[abm.R_QUAL, f] != w.state["firm"][abm.R_QUAL, f]                     # a different firm
    assert abs(st["aware"][f].mean() - k.launch_aware) < 0.08
    later = rollout(eng, 1, [eng.defaults(0)], 30, state=st, t0=t, keep=True)
    assert later.res[-1].info["served"][f] > 0                                       # and it wins customers
    # it also happens without any tampering: a seat that takes the whole market starves an incumbent out
    eng = eng_of("abm_network_fx", 0)
    r = rollout(eng, 1, [nat(eng, price=0.7, quality=0.6, outreach=0.04)], 160, keep=True)
    events = [e for x in r.res for e in x.info["events"]]
    exits = [e for e in events if e["what"] == "exit"]
    assert exits and any(e["what"] == "entry" and e["firm"] == exits[0]["firm"] and
                         e["t"] == exits[0]["t"] + eng._k.delay for e in events)


# ====================================================================== truth() against the simulation
def _paths(truth: dict, src: str, dst: str) -> list[tuple[int, int]]:
    """Every simple directed path src -> dst as (product of signs, 0 if any edge is unsigned; sum of lags)."""
    adj: dict[str, list[dict]] = {}
    for e in truth["edges"]:
        adj.setdefault(e["src"], []).append(e)
    out: list[tuple[int, int]] = []

    def walk(node: str, seen: frozenset, sign: int, lag: int) -> None:
        if node == dst:
            out.append((sign, lag))
            return
        for e in adj.get(node, ()):
            if e["dst"] not in seen:
                walk(e["dst"], seen | {e["dst"]}, sign * e["sign"], lag + e["lag"])
    walk(src, frozenset([src]), 1, 0)
    return out


@pytest.mark.parametrize("family", ALL)
def test_truth_is_well_formed(family):
    eng = eng_of(family, 0)
    tr = eng.truth()
    assert tr["engine"] == "abm" and tr["family"] == family and tr["regimes"] == []
    assert tuple(tr["motifs"]) == MECH_FAMILIES[family].motifs
    ids = [n["id"] for n in tr["nodes"]]
    assert len(set(ids)) == len(ids)
    by_id = {n["id"]: n for n in tr["nodes"]}
    assert {n["role"] for n in tr["nodes"]} <= {"lever", "latent", "gauge", "outcome"}
    for e in tr["edges"]:
        assert e["src"] in by_id and e["dst"] in by_id and e["sign"] in (-1, 0, 1) and e["lag"] >= 0 and e["kind"]
        assert by_id[e["dst"]]["role"] != "lever"                       # nothing causes a lever
    for seat in range(eng.n_seats):
        levers = {n["public"] for n in tr["nodes"] if n["role"] == "lever" and n["seat"] == seat}
        gauges = {n["public"] for n in tr["nodes"] if n["role"] == "gauge" and n["seat"] == seat}
        assert levers == {lv.name for lv in eng.levers(seat)} and gauges == {g.name for g in eng.gauges(seat)}
        assert sum(n["role"] == "outcome" and n["seat"] == seat for n in tr["nodes"]) == 1
    assert all(n["public"] is None for n in tr["nodes"] if n["role"] in ("latent", "outcome"))
    assert set(tr["chain"]) == {"awareness", "trial", "retention", "feedback"} and all(tr["chain"].values())
    srcs = {e["src"] for e in tr["edges"]}
    for n in tr["nodes"]:
        if n["role"] == "lever":
            assert (n["id"] in srcs) != ("noop" in n["id"])             # distractors cause nothing; the rest do
    json.dumps(tr)


TRUTH_SEEDS = range(5)
WARM, HOLD, SHIFT = 45, 40, 40.0
WARM_BY_FAMILY = {"abm_network_fx": 12}     # under lock-in the market is only contested early on
# Where the interventions start from: a point at which every mechanism of the family is at work (some bad
# experiences, a capacity that binds, a gate that is partly open, a market that is still contested), so that an
# edge that exists has something to move.
PROBE = {
    "abm_utility": dict(price=1.0, quality=0.12, outreach=0.011),
    "abm_wom": dict(price=1.0, quality=0.25, outreach=0.004, referral=0.3),
    "abm_congestion": dict(price=0.96, quality=0.15, capacity=0.07, outreach=0.012),
    "abm_trust_gate": dict(price=1.0, quality=0.3, outreach=0.014),
    "abm_network_fx": dict(price=1.0, quality=0.1, outreach=0.015),
    "abm_segments": dict(price=1.0, quality=0.25, outreach=0.007, referral=0.3),
}


@functools.lru_cache(maxsize=None)
def _interventions(family: str):
    """For every seed, lever and target: the paired response (shifted minus base, common random numbers) to moving
    the lever by SHIFT dial points for HOLD ticks from a warmed-up market, oriented so that + means 'lever up'."""
    out = []
    for sd in TRUTH_SEEDS:
        eng = eng_of(family, sd)
        tr = eng.truth()
        probe = nat(eng, **PROBE[family])
        w = WARM_BY_FAMILY.get(family, WARM)
        warm = rollout(eng, 50 + sd, [probe], w)
        base = rollout(eng, 50 + sd, [probe], HOLD, state=warm.state, t0=w, keep=True)
        targets = [n for n in tr["nodes"] if n["role"] in ("gauge", "outcome")]
        resp = {}
        for lv in (n for n in tr["nodes"] if n["role"] == "lever"):
            step = SHIFT if probe[lv["public"]] <= 100.0 - SHIFT else -SHIFT
            alt = rollout(eng, 50 + sd, [{**probe, lv["public"]: probe[lv["public"]] + step}], HOLD,
                          state=warm.state, t0=w, keep=True)
            for tg in targets:
                if tg["role"] == "gauge":
                    d = [a.readings[0][tg["public"]] - b.readings[0][tg["public"]] for a, b in zip(alt.res, base.res)]
                else:
                    d = [a.score[0] - b.score[0] for a, b in zip(alt.res, base.res)]
                resp[(lv["id"], tg["id"])] = np.sign(step) * np.array(d)
        out.append((tr, resp))
    return out


@pytest.mark.parametrize("family", SINGLE)
def test_truth_reachability_and_lags_match_interventions(family):
    """No declared path: the response is exactly zero, bit for bit. A declared path: nothing moves before the
    smallest total lag along any path, and it does move (in most seeds) once that lag has passed."""
    moved, total = {}, {}
    n_none = n_path = 0
    for tr, resp in _interventions(family):
        for (lv, tg), d in resp.items():
            paths = _paths(tr, lv, tg)
            key = (lv.split(".", 1)[1], re.sub(r"_\d+$", "", tg.split(".", 1)[1]))
            if not paths:
                assert not d.any(), (family, lv, tg)
                n_none += 1
                continue
            m = min(lag for _, lag in paths)
            assert m < HOLD and not d[:m].any(), (family, lv, tg, m, d[:m])
            total[key] = total.get(key, 0) + 1
            moved[key] = moved.get(key, 0) + bool(d[m:].any())
            n_path += 1
    assert n_path >= len(TRUTH_SEEDS) * 15
    if family != "abm_congestion":                       # (there every gauge is reachable unless a distractor exists)
        assert n_none >= len(TRUTH_SEEDS) * 3
    for key in total:                                    # (a few responses are too faint to show in every market)
        assert moved[key] >= 0.4 * total[key], (family, key, moved[key], total[key])


@pytest.mark.parametrize("family", SINGLE)
def test_truth_signs_match_interventions(family):
    """Signs, two ways. (a) First arrival: when every signed path of minimal lag agrees, the response at exactly
    that lag never has the opposite sign in any market (with common random numbers this is exact, not statistical);
    whether it is visible at that very tick depends on a marginal customer existing, so strict visibility is
    required of the price lever everywhere and of at least 40% of the checked pairs. (b) Overall: when every signed
    path of any length agrees, the response summed over the hold has that sign in total, with at most one market in
    five against. Paths through an unsigned (gated or state-dependent) edge make no prediction. The outcome is
    excluded: every costly lever is non-monotone on net, which the calibration tests establish."""
    first: dict = {}
    overall: dict = {}
    for tr, resp in _interventions(family):
        role = {n["id"]: n["role"] for n in tr["nodes"]}
        for (lv, tg), d in resp.items():
            if role[tg] != "gauge":
                continue
            paths = _paths(tr, lv, tg)
            signed = [(s, lag) for s, lag in paths if s != 0]
            if not signed:
                continue
            key = (lv.split(".", 1)[1], re.sub(r"_\d+$", "", tg.split(".", 1)[1]))
            m = min(lag for _, lag in paths)
            at_m = {s for s, lag in signed if lag == m}
            if len(at_m) == 1:
                first.setdefault(key, []).append(next(iter(at_m)) * d[m])
            every = {s for s, _ in signed}
            if len(every) == 1:
                overall.setdefault(key, []).append(next(iter(every)) * d.sum())
    assert len(first) >= 10 and len(overall) >= 8, (family, len(first), len(overall))
    for key, vals in first.items():
        assert min(vals) >= 0, (family, "first arrival", key, vals)
    visible = [key for key, vals in first.items() if sum(vals) > 0]
    assert len(visible) >= 0.4 * len(first) and ("price", "g.orders") in visible, (family, sorted(visible))
    for key, vals in overall.items():
        assert sum(vals) > 0 and np.mean(np.array(vals) < 0) <= 0.2, (family, "overall", key, vals)


def test_declared_sign_tables_are_not_vacuous():
    """The pairs the sign test actually checks include the ones each causal story rests on."""
    def checked(family):
        tr = eng_of(family, 0).truth()
        out = {}
        for lv in (n["id"] for n in tr["nodes"] if n["role"] == "lever"):
            for tg in (n["id"] for n in tr["nodes"] if n["role"] == "gauge"):
                signed = {s for s, _ in _paths(tr, lv, tg) if s != 0}
                if len(signed) == 1:
                    out[(lv[3:], re.sub(r"_\d+$", "", tg[5:]))] = next(iter(signed))
        return out
    # declared signs are in dial terms; `natural` turns them back into the direction of the underlying quantity
    def natural(family, table, lever, gauge):
        flip = eng_of(family, 0).roles()["levers"]
        s = table[(lever, gauge)] * (-1 if flip[lever]["flip"] else 1)
        return s * (-1 if gauge == "rival_price" and flip["price"]["flip"] else 1)
    u = checked("abm_utility")
    assert natural("abm_utility", u, "price", "orders") == -1 and natural("abm_utility", u, "outreach", "orders") == 1
    assert natural("abm_utility", u, "quality", "rating") == 1 and natural("abm_utility", u, "quality", "orders") == 1
    assert natural("abm_utility", u, "price", "rival_price") == 1
    assert natural("abm_utility", u, "outreach", "rival_price") == -1
    w = checked("abm_wom")
    assert natural("abm_wom", w, "referral", "orders") == 1 and natural("abm_wom", w, "quality", "active") == 1
    c = checked("abm_congestion")
    assert natural("abm_congestion", c, "price", "queue") == -1
    assert natural("abm_congestion", c, "outreach", "queue") == 1
    assert ("outreach", "volume") not in c            # rationed orders divert buyers: no declared direction
    assert ("capacity", "queue") not in c             # overall mixed (kept customers order again) ...
    tr = eng_of("abm_congestion", 0).truth()
    cap_queue = _paths(tr, "s0.capacity", "s0.g.queue")
    turned = -1 if eng_of("abm_congestion", 0).roles()["levers"]["capacity"]["flip"] else 1
    assert {s * turned for s, lag in cap_queue if lag == 0} == {-1}      # ... but its first arrival is declared
    n = checked("abm_network_fx")
    assert ("price", "rival_price") not in n                             # rivals tip among themselves
    assert natural("abm_network_fx", n, "price", "active") == -1
    assert ("price", "queue") not in u                                   # no queue without finite capacity


# ====================================================================== calibration (20 seeds per family)
@functools.lru_cache(maxsize=None)
def _calibration(family: str):
    rng = np.random.default_rng(2024)
    out = SimpleNamespace(default=[], ref=[], rand=[], corners={})
    for sd in SEEDS20:
        eng = eng_of(family, sd)
        out.default.append(net(eng, 7000 + sd, both(eng, eng.defaults(0))).mean())
        out.ref.append(net_nat(family, sd, [REF[family]] * eng.n_seats).mean())
        for _ in range(2):
            out.rand.append(net(eng, 7000 + sd, [random_settings(eng, rng) for _ in range(eng.n_seats)]).mean())
    # corners of the box of the levers that do something (distractors stay at their default), on two mechanisms;
    # a corner is named in natural terms (each lever at the bottom or the top of its range), because which end of
    # a dial that is differs from mechanism to mechanism
    out.roles = sorted(lv["role"] for lv in eng_of(family, 0).config["levers"] if not lv["role"].startswith("noop"))
    for combo in itertools.product(("natural_lo", "natural_hi"), repeat=len(out.roles)):
        vals = []
        for sd in (0, 1):
            eng = eng_of(family, sd)
            ends = eng.roles()["levers"]
            corner = eng.encode({ro: ends[ro][end] for ro, end in zip(out.roles, combo)})
            assert all(corner[ends[ro]["public"]] in (0.0, 100.0) for ro in out.roles)
            vals.append(net(eng, 7000 + sd, both(eng, corner)).mean())
        out.corners[combo] = float(np.mean(vals))
    return out


@pytest.mark.parametrize("family", ALL)
def test_calibration_default_reference_random_corners(family):
    c = _calibration(family)
    d, d_se = mean_se(c.default)
    r, r_se = mean_se(c.ref)
    x, x_se = mean_se(c.rand)
    # some settings earn clearly positive net ...
    assert r > 0.012 and r / r_se > 4.0, (family, r, r_se)
    assert np.mean(np.array(c.ref) > 0) >= 0.8, (family, c.ref)
    # ... the default earns about nothing ...
    assert abs(d) < 0.08 * r and abs(d) < 0.003, (family, d, r)
    assert max(abs(v) for v in c.default) < 0.25 * r, (family, c.default)
    # ... uniformly random settings lose money on average ...
    assert x < -0.02 and x / x_se < -3.0 and np.mean(np.array(c.rand) < 0) >= 0.7, (family, x, x_se)
    # ... and no corner of the lever box is anywhere near the best found
    best_corner = max(c.corners.values())
    note(f"calibration/{family}", default=d, reference=r, reference_se=r_se, random=x, random_se=x_se,
         best_corner=best_corner, reference_positive_share=float(np.mean(np.array(c.ref) > 0)),
         random_negative_share=float(np.mean(np.array(c.rand) < 0)))
    assert len(c.corners) == 2 ** len(c.roles) >= 8
    assert best_corner < 0.3 * r, (family, best_corner, r, max(c.corners, key=c.corners.get))


@pytest.mark.parametrize("family", ALL)
def test_constructive_optimum_is_the_reference_these_tests_measure(family):
    """The engine's own statement of a good setting (a private hint for the operator's reference search) is exactly
    the reference whose earnings the calibration test establishes, in the schedule form the evaluator expects."""
    for sd in (0, 7):
        eng = eng_of(family, sd)
        for seat in range(eng.n_seats):
            hint = eng.constructive_optimum(seat)
            assert [e["from"] for e in hint["schedule"]] == [0]
            got, want = hint["schedule"][0]["settings"], nat(eng, **REF[family])
            assert got.keys() == want.keys() == {lv.name for lv in eng.levers(seat)}
            assert all(abs(got[nm] - want[nm]) < 1e-9 and 0.0 <= got[nm] <= 100.0 for nm in got)
    assert abm._GOOD[eng_of(family, 0).config["kind"]] == REF[family]


def _paired(family: str, a: dict, b: dict, seeds=SEEDS20, ticks=None) -> tuple[float, float, float, float]:
    """Mean net of settings a and of b (same mechanisms, same random numbers), mean of a - b and its t statistic."""
    va, vb = [], []
    seats = MECH_FAMILIES[family].seats
    for sd in seeds:
        va.append(net_nat(family, sd, [a] * seats, ticks).mean())
        vb.append(net_nat(family, sd, [b] * seats, ticks).mean())
    diff = np.array(va) - np.array(vb)
    m, se = mean_se(diff)
    return float(np.mean(va)), float(np.mean(vb)), m, m / se


def test_prediction_outreach_pays_in_utility_and_quality_pays_in_wom():
    """Same total spend, split two ways. Where awareness comes from outreach (abm_utility) the outreach-heavy split
    earns more; where it only travels from satisfied customers (abm_wom) moving that money into quality earns more."""
    o_heavy = dict(price=1.05, outreach=0.030, quality=0.10)        # 0.030 + 0.10 * 0.10 = 0.040 N*p0 per tick
    q_heavy = dict(price=1.05, outreach=0.002, quality=0.38)        # 0.002 + 0.10 * 0.38 = 0.040 N*p0 per tick
    for family in ("abm_utility", "abm_wom"):
        k = eng_of(family, 0)._k
        assert abs((0.030 * k.N * k.p0 + 0.10 * k.c_q) - (0.002 * k.N * k.p0 + 0.38 * k.c_q)) < 1e-6 * k.N * k.p0
    q_u, o_u, d_u, t_u = _paired("abm_utility", q_heavy, o_heavy)
    q_w, o_w, d_w, t_w = _paired("abm_wom", q_heavy, o_heavy)
    note("prediction/outreach_vs_quality", utility_quality_heavy=q_u, utility_outreach_heavy=o_u, utility_t=t_u,
         wom_quality_heavy=q_w, wom_outreach_heavy=o_w, wom_t=t_w)
    assert d_u < -0.005 and t_u < -4, (q_u, o_u, t_u)
    assert d_w > 0.02 and t_w > 4, (q_w, o_w, t_w)
    assert q_w > 0.02 > o_w and o_u > 0.012


def test_prediction_under_provisioning_is_the_trap_in_congestion():
    """With too little capacity, buying more demand makes things worse (queues, complaints, churn); with adequate
    capacity the same outreach pays. In abm_utility, where nothing is rationed, outreach simply pays."""
    ref = REF["abm_congestion"]
    lean = {**ref, "outreach": 0.002}
    tight, tight_lean = {**ref, "capacity": 0.02}, {**lean, "capacity": 0.02}
    a, b, d_ok, t_ok = _paired("abm_congestion", ref, lean)
    c, e, d_tight, t_tight = _paired("abm_congestion", tight, tight_lean)
    note("prediction/congestion_trap", adequate_capacity_outreach=a, adequate_capacity_little_outreach=b,
         t_adequate=t_ok, tight_capacity_outreach=c, tight_capacity_little_outreach=e, t_tight=t_tight)
    assert d_ok > 0.004 and t_ok > 3, (a, b, t_ok)
    assert d_tight < -0.004 and t_tight < -4, (c, e, t_tight)
    assert a > 0.012 and c < -0.01                                   # the trap loses money outright
    u, v, d_u, t_u = _paired("abm_utility", REF["abm_utility"], {**REF["abm_utility"], "outreach": 0.002},
                             seeds=range(10))
    assert d_u > 0.01 and t_u > 4
    # the best capacity is interior: neither the least nor the most the lever allows
    _, _, d_hi, t_hi = _paired("abm_congestion", ref, {**ref, "capacity": 0.6}, seeds=range(10))
    assert d_hi > 0.008 and t_hi > 4


def test_prediction_cold_start_and_quality_threshold_in_trust_gate():
    """The gate makes quality a threshold: below it nothing ever happens; above it there is a cold start during
    which the right settings lose money, then a take-off. abm_utility has neither."""
    ref = REF["abm_trust_gate"]
    early, total, low_total, late_ratio = [], [], [], []
    for sd in SEEDS20:
        eng = eng_of("abm_trust_gate", sd)
        good = rollout(eng, 7000 + sd, [nat(eng, **ref)], keep=True)
        poor = rollout(eng, 7000 + sd, [nat(eng, **{**ref, "quality": 0.1})], keep=True)
        early.append(good.score[:12].mean() / unit(eng))
        total.append(good.score.mean() / unit(eng))
        low_total.append(poor.score.mean() / unit(eng))
        late_ratio.append(np.mean([r.info["orders"][0] for r in poor.res[-40:]]) /
                          max(1.0, np.mean([r.info["orders"][0] for r in good.res[-40:]])))
    note("prediction/trust_gate", first_12_ticks=np.mean(early), whole_run=np.mean(total),
         whole_run_low_quality=np.mean(low_total), late_orders_low_over_good=np.mean(late_ratio))
    assert np.mean(early) < -0.01 and np.mean(np.array(early) < 0) >= 0.9          # cold start loses money
    assert np.mean(total) > 0.012 and np.mean(low_total) < -0.005
    assert np.mean(late_ratio) < 0.25                                              # the gate never opened
    # orders respond to quality as a threshold in trust_gate and smoothly in utility
    def late_orders(family, q):
        vals = []
        for sd in range(8):
            eng = eng_of(family, sd)
            r = rollout(eng, 7000 + sd, [nat(eng, **{**REF[family], "quality": q})], keep=True)
            vals.append(np.mean([x.info["orders"][0] for x in r.res[-40:]]) / eng._k.N)
        return float(np.mean(vals))
    g_lo, g_hi = late_orders("abm_trust_gate", 0.05), late_orders("abm_trust_gate", 0.65)
    u_lo, u_hi = late_orders("abm_utility", 0.05), late_orders("abm_utility", 0.65)
    note("prediction/quality_threshold", gate_orders_low_q=g_lo, gate_orders_high_q=g_hi, utility_orders_low_q=u_lo,
         utility_orders_high_q=u_hi)
    assert g_hi > 6 * g_lo and u_hi < 3 * u_lo and u_lo > 0.03


def test_prediction_early_penetration_pays_and_late_price_cuts_do_not_under_network_effects():
    """Two campaigns with the same prices and the same length: a low price for 30 ticks, the high price otherwise.
    Run it at the start (the market is open) or from tick 60 (the incumbents have built their bases). Under network
    effects the early one captures nearly the whole market and pays; the late one wins far fewer customers, keeps
    none of them and does not pay; without network effects (abm_utility) timing hardly matters."""
    lo, hi = 0.7, 1.2
    base = dict(quality=0.4, outreach=0.02)

    def campaign(family, start, seeds=SEEDS20):
        tot, share_end, share_peak = [], [], []
        for sd in seeds:
            eng = eng_of(family, sd)
            s_lo, s_hi = [nat(eng, price=lo, **base)], [nat(eng, price=hi, **base)]
            r = rollout(eng, 7000 + sd, lambda t: s_lo if start <= t < start + 30 else s_hi, 160, keep=True)
            tot.append(r.score.mean() / unit(eng))
            share_peak.append(r.res[min(start + 29, 159)].info["active"][0] / eng._k.N)
            share_end.append(r.res[-1].info["active"][0] / eng._k.N)
        return np.array(tot), float(np.mean(share_peak)), float(np.mean(share_end))

    early, early_peak, _ = campaign("abm_network_fx", 0)
    late, late_peak, late_end = campaign("abm_network_fx", 60)
    never, _, _ = campaign("abm_network_fx", 10 ** 6)                 # the high price throughout
    d, se = mean_se(early - late)
    note("prediction/network_fx_timing", early=early.mean(), late=late.mean(), never=never.mean(), t=d / se,
         share_after_early_campaign=early_peak, share_after_late_campaign=late_peak, late_share_at_end=late_end)
    assert early.mean() > 0.01 and late.mean() < -0.02 and d / se > 4, (early.mean(), late.mean())
    assert early_peak > 0.8 and late_peak < 0.5 * early_peak and late_end < 0.1    # locked-in customers stay put
    assert abs(late.mean() - never.mean()) < 0.35 * abs(early.mean() - never.mean())
    u_early, u_peak_e, _ = campaign("abm_utility", 0, range(10))
    u_late, u_peak_l, _ = campaign("abm_utility", 60, range(10))
    note("prediction/utility_timing", early=u_early.mean(), late=u_late.mean(), share_after_early_campaign=u_peak_e,
         share_after_late_campaign=u_peak_l)
    assert abs((u_early - u_late).mean()) < 0.25 * d                  # no lock-in, no path dependence
    assert u_peak_l > 0.5 * u_peak_e                 # there a late cut wins customers much as an early one does


def test_prediction_segments_need_both_channels():
    ref = REF["abm_segments"]
    both_, no_out, d1, t1 = _paired("abm_segments", ref, {**ref, "outreach": 0.0})
    _, low_q, d2, t2 = _paired("abm_segments", ref, {**ref, "quality": 0.08})
    note("prediction/segments", both_channels=both_, no_outreach=no_out, low_quality=low_q, t_outreach=t1,
         t_quality=t2)
    assert d1 > 0.008 and t1 > 4 and d2 > 0.008 and t2 > 4, (both_, no_out, low_q)
    # in the pure word-of-mouth market dropping outreach costs next to nothing, in the pure utility market it is fatal
    w_ref = REF["abm_wom"]
    _, _, d_w, _ = _paired("abm_wom", w_ref, {**w_ref, "outreach": 0.0}, seeds=range(10))
    u_ref = REF["abm_utility"]
    u, u0, d_u, _ = _paired("abm_utility", u_ref, {**u_ref, "outreach": 0.0}, seeds=range(10))
    note("prediction/outreach_dropped", wom_change=-d_w, utility_change=-d_u, utility_without_outreach=u0)
    assert abs(d_w) < 0.006 and d_u > 0.03 and u0 < 0.004


# ====================================================================== strategic structure of two-seat families
def _profile(family: str, s0: dict, s1: dict, seeds=SEEDS20) -> np.ndarray:
    """Net per seat (rows: seeds) when seat 0 plays s0 and seat 1 plays s1."""
    return np.array([net_nat(family, sd, [s0, s1]) for sd in seeds])


@pytest.mark.parametrize("family", ["abm_duopoly", "abm_duopoly_fx"])
def test_duopoly_is_competitive(family):
    ref = REF[family]
    base = _profile(family, ref, ref)
    cut = _profile(family, {**ref, "price": ref["price"] - 0.2}, ref)
    d, se = mean_se(cut[:, 1] - base[:, 1])
    note(f"strategic/{family}", both_reference=base.mean(0).round(4).tolist(),
         seat0_cuts_price=cut.mean(0).round(4).tolist(), other_seat_change=d, t=d / se)
    assert d < -0.008 and d / se < -4, (family, base.mean(0), cut.mean(0))       # your cut is my loss
    assert np.mean(cut[:, 1] < base[:, 1]) >= 0.9
    loud = _profile(family, {**ref, "outreach": 3 * ref["outreach"]}, ref, seeds=range(10))
    assert (loud[:, 1] - base[:10, 1]).mean() < 0                                # so is your outreach
    # the seats are symmetric: with equal settings neither seat is systematically ahead
    gap, gap_se = mean_se(base[:, 0] - base[:, 1])
    assert abs(gap) < 3 * gap_se, (family, gap, gap_se)


def test_duopoly_with_network_effects_tips():
    ref = REF["abm_duopoly_fx"]
    ten = range(10)
    base = _profile("abm_duopoly_fx", ref, ref, ten)
    cut = _profile("abm_duopoly_fx", {**ref, "price": ref["price"] - 0.1}, ref, ten)
    plain_ref = REF["abm_duopoly"]
    plain = _profile("abm_duopoly", plain_ref, plain_ref, ten)
    plain_cut = _profile("abm_duopoly", {**plain_ref, "price": plain_ref["price"] - 0.1}, plain_ref, ten)
    swing_fx = (cut[:, 0] - cut[:, 1]).mean() - (base[:, 0] - base[:, 1]).mean()
    swing = (plain_cut[:, 0] - plain_cut[:, 1]).mean() - (plain[:, 0] - plain[:, 1]).mean()
    note("strategic/tipping", payoff_gap_from_small_price_edge_fx=swing_fx, same_in_plain_duopoly=swing)
    assert swing_fx > 2 * swing > 0, (swing_fx, swing)       # the same small price edge decides far more


def test_commons_is_a_mixed_motive_dilemma():
    """Restraint R and heavy outreach H. Joint restraint beats joint heavy outreach, yet whatever the other seat
    does, heavy outreach pays the seat that does it: T > R > P > S."""
    b = dict(price=1.05, quality=0.25)
    lo, hi = {**b, "outreach": 0.004}, {**b, "outreach": 0.012}
    rr, hh = _profile("abm_commons", lo, lo), _profile("abm_commons", hi, hi)
    hr, rh = _profile("abm_commons", hi, lo), _profile("abm_commons", lo, hi)
    reward, punish = (rr[:, 0] + rr[:, 1]) / 2, (hh[:, 0] + hh[:, 1]) / 2
    tempt, sucker = (hr[:, 0] + rh[:, 1]) / 2, (hr[:, 1] + rh[:, 0]) / 2
    note("strategic/abm_commons", T_heavy_vs_restraint=tempt.mean(), R_both_restrain=reward.mean(),
         P_both_heavy=punish.mean(), S_restraint_vs_heavy=sucker.mean())
    for name, hi_, lo_ in (("R>P", reward, punish), ("T>R", tempt, reward), ("P>S", punish, sucker)):
        d, se = mean_se(hi_ - lo_)
        assert d > 0.002 and d / se > 3.0, (name, d, se, reward.mean(), punish.mean(), tempt.mean(), sucker.mean())
    # the shared stock is why: the pool is lower, and total shopping with it, when both push
    def pool_and_volume(s0, s1):
        p, v = [], []
        for sd in range(6):
            eng = eng_of("abm_commons", sd)
            r = rollout(eng, 7000 + sd, [nat(eng, **s0), nat(eng, **s1)], keep=True)
            p.append(float(r.state["pool"][0]))
            v.append(np.mean([x.info["volume"] for x in r.res[40:]]) / eng._k.N)
        return float(np.mean(p)), float(np.mean(v))
    p_rr, v_rr = pool_and_volume(lo, lo)
    p_hr, v_hr = pool_and_volume(hi, lo)
    p_hh, v_hh = pool_and_volume(hi, hi)
    note("strategic/abm_commons_pool", pool_RR=p_rr, pool_HR=p_hr, pool_HH=p_hh, volume_RR=v_rr, volume_HH=v_hh)
    assert p_rr > p_hr > p_hh and p_rr - p_hh > 0.15 and v_hh < v_rr


def test_complement_rewards_matched_effort_and_a_jointly_sustainable_price():
    ref = REF["abm_complement"]
    idle = dict(price=0.8, quality=0.0, outreach=0.0)                 # a seat that does nothing
    matched = _profile("abm_complement", ref, ref)
    alone = _profile("abm_complement", ref, idle)
    joint, solo = matched.sum(1), alone.sum(1)
    note("strategic/abm_complement", matched_per_seat=matched.mean(0).round(4).tolist(),
         effort_alone_seat0_then_idle_seat1=alone.mean(0).round(4).tolist())
    assert joint.mean() > 0.03 and matched[:, 0].mean() > 0.012 and matched[:, 1].mean() > 0.012
    assert alone[:, 0].mean() < 0.2 * matched[:, 0].mean()            # effort alone earns little (here: loses)
    assert solo.mean() < 0.2 * joint.mean()
    d, se = mean_se(joint - solo)
    assert d / se > 6
    # the best a seat can do alone is small: a cheap component with no effort wasted on a bundle nobody can buy
    best_alone = max(_profile("abm_complement", dict(price=p, quality=q, outreach=0.006), idle, range(6))[:, 0].mean()
                     for p in (0.4, 0.5) for q in (0.0, 0.3))
    note("strategic/abm_complement_alone", best_found_alone=best_alone)
    assert best_alone < 0.25 * matched[:, 0].mean()
    # quality must be matched: a lopsided pair with the same total effort earns less
    ten = range(10)
    lopsided = _profile("abm_complement", {**ref, "quality": 0.65}, {**ref, "quality": 0.05}, ten)
    note("strategic/abm_complement_quality", matched_total=joint[:10].mean(), lopsided_total=lopsided.sum(1).mean())
    assert lopsided.sum(1).mean() < 0.6 * joint[:10].mean()
    # the price is joint: customers pay the sum, so if both charge what looks fine alone the bundle is unsaleable
    greedy = _profile("abm_complement", {**ref, "price": 0.85}, {**ref, "price": 0.85}, ten)
    assert greedy.sum(1).mean() < 0 < joint.mean()
    one_greedy = _profile("abm_complement", {**ref, "price": 0.85}, ref, ten)
    note("strategic/abm_complement_price", both_high_price_total=greedy.sum(1).mean(),
         one_high_price_per_seat=one_greedy.mean(0).round(4).tolist())
    assert one_greedy[:, 1].mean() < 0.6 * matched[:10, 1].mean()      # one seat's price hike is paid by the other


# ====================================================================== throughput
def test_throughput_n500_one_seat():
    eng = build_engine(make_mechanism("abm_utility", 1, {"n_customers": 500}))
    assert eng.n_seats == 1 and eng.config["n_customers"] == 500
    s = [nat(eng, **REF["abm_utility"])]
    ticks = 400
    best_cpu = best_wall = 0.0
    for rep in range(3):
        st = eng.init_state(tick_rng(rep, "r1", "init"))
        rngs = [tick_rng(rep, "r1", t) for t in range(ticks)]
        w0, c0 = time.perf_counter(), time.process_time()
        for t in range(ticks):
            st = eng.step(st, s, rngs[t], t).state
        best_cpu = max(best_cpu, ticks / (time.process_time() - c0))
        best_wall = max(best_wall, ticks / (time.perf_counter() - w0))
    note("throughput/n500_one_seat", ticks_per_cpu_second=round(best_cpu), ticks_per_wall_second=round(best_wall))
    # CPU time, because the container's CPU quota is shared and wall-clock here measures the neighbours
    assert best_cpu >= 1000.0
