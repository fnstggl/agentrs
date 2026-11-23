"""The `techsys` mechanism engine: executable technical systems with hidden operational constraints.

Everything here drives `Engine.step` directly with `base.tick_rng` (no kernel, no simulator module). The tests
establish, by simulation:

* the contract: JSON configs, deterministic generation, opaque public surface, purity and bit-identical steps;
* invariants: exact job conservation every tick, exact value accounting, the revenue bound, bounded state under
  any settings for 2,000 ticks;
* the truth graph: every monotone effect it implies holds under common random numbers, every declared
  non-monotone lever response really changes direction across the lever's range, lags and regimes are exact;
* calibration over 20 seeds per family, and the strategic structure of the two-seat families.

The numbers printed with `-s` are the ones quoted in the work-package report.
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache

import numpy as np
import pytest

from crucible.core.canonical import digest
from crucible.mechanisms import techsys
from crucible.mechanisms.base import (ENGINES, MECH_FAMILIES, Engine, StepResult, build_engine, make_mechanism,
                                      mech_families, tick_rng)
from crucible.mechanisms.techsys import TechSys, implied_effects, node_value, reference_settings

FAMILIES = ("tech_pipeline", "tech_contention", "tech_cascade", "tech_redundant", "tech_lostwork", "tech_handoff",
            "tech_shared_pool")
SINGLE_SEAT = FAMILIES[:5]
N_SEEDS = 20
RIG = "r1"
NAME = re.compile(r"^[bdfgjklmnprstvz][aeiou]\d\d$")


# ====================================================================== helpers
@lru_cache(maxsize=None)
def mech(family: str, seed: int):
    m = make_mechanism(family, seed)
    return m, build_engine(m)


def defaults(eng: Engine) -> list[dict]:
    return [eng.defaults(j) for j in range(eng.n_seats)]


_STATES: dict = {}
_BITS = np.random.PCG64(0)
_GEN = np.random.Generator(_BITS)


def rng_at(world: int, t) -> np.random.Generator:
    """`tick_rng(world, RIG, t)` without paying for its derivation again: the bit-generator state it starts from
    is remembered per (world, t) and loaded into one reused generator (the streams are identical, see
    `test_remembered_generators_are_tick_rng`). The caller must draw before asking for the next one."""
    state = _STATES.get((world, t))
    if state is None:
        state = _STATES[(world, t)] = tick_rng(world, RIG, t).bit_generator.state
    _BITS.state = state
    return _GEN


def simulate(eng: Engine, settings=None, ticks: int | None = None, world: int = 1, policy=None, keep: bool = False,
             state=None, t0: int = 0) -> dict:
    """Run `ticks` ticks from `state` (or the initial state) with constant `settings` or `policy(t) -> settings`.
    Returns totals, the final state and (with `keep`) every StepResult."""
    ticks = eng.horizon if ticks is None else ticks
    ns = eng.n_seats
    st = eng.init_state(rng_at(world, "init")) if state is None else state
    out = {"arrivals": 0, "completed": 0, "late": 0, "dropped": 0, "shed": 0, "value": 0.0, "value_done": 0.0,
           "lost": 0.0, "revenue": [0.0] * ns, "cost": [0.0] * ns, "score": [0.0] * ns, "rows": [],
           "reasons": {"overload": 0.0, "timeout": 0.0, "failure": 0.0, "shed": 0.0}}
    for t in range(t0, t0 + ticks):
        r = eng.step(st, policy(t) if policy is not None else settings, rng_at(world, t), t)
        st = r.state
        i = r.info
        for k in ("arrivals", "completed", "late", "dropped", "shed"):
            out[k] += i[k]
        out["value"] += i["value_arrived"]
        out["value_done"] += i["value_completed"]
        out["lost"] += i["value_lost"]
        for k in out["reasons"]:
            out["reasons"][k] += i["lost_by_reason"][k]
        for j in range(ns):
            out["revenue"][j] += r.revenue[j]
            out["cost"][j] += r.cost[j]
            out["score"][j] += r.score[j]
        if keep:
            out["rows"].append(r)
    out["state"] = st
    out["net"] = sum(out["revenue"]) - sum(out["cost"])
    out["net_frac"] = out["net"] / out["value"]          # net as a fraction of the value that arrived
    out["lost_frac"] = out["lost"] / out["value"]
    return out


@lru_cache(maxsize=None)
def outcome(family: str, seed: int, style: str) -> tuple:
    """(net / arriving value, lost value / arriving value, value lost, net, score) of a reference style, on the
    world seed that belongs to this mechanism seed."""
    m, eng = mech(family, seed)
    s = defaults(eng) if style == "default" else reference_settings(m["config"], style)
    r = simulate(eng, s, world=1000 + seed)
    return r["net_frac"], r["lost_frac"], r["lost"], r["net"], sum(r["score"])


def levers_of(cfg: dict, kind: str) -> list[dict]:
    return [lv for lv in cfg["levers"] if lv["kind"] == kind]


def with_lever(settings: list[dict], lv: dict, value: float) -> list[dict]:
    out = [dict(s) for s in settings]
    out[lv["seat"]][lv["name"]] = float(min(lv["hi"], max(lv["lo"], round(value) if lv["integer"] else value)))
    return out


def state_digest(state: dict) -> str:
    return digest({k: [str(v.dtype), list(v.shape), v.tolist()] for k, v in sorted(state.items())})


def result_digest(r: StepResult) -> str:
    return digest([state_digest(r.state), r.readings, r.revenue, r.cost, r.score, r.info])


def random_settings(eng: Engine, g: np.random.Generator) -> list[dict]:
    return [{lv.name: lv.clip(g.uniform(lv.lo, lv.hi)) for lv in eng.levers(j)} for j in range(eng.n_seats)]


def corner_settings(eng: Engine, g: np.random.Generator | None = None, which: str | None = None) -> list[dict]:
    out = []
    for j in range(eng.n_seats):
        d = {}
        for lv in eng.levers(j):
            top = (which == "hi") if which is not None else bool(g.integers(2))
            d[lv.name] = lv.hi if top else lv.lo
        out.append(d)
    return out


def is_plain_json(x) -> bool:
    if x is None or type(x) in (str, int, float, bool):
        return not (type(x) is float and not math.isfinite(x))
    if type(x) is list:
        return all(is_plain_json(v) for v in x)
    if type(x) is dict:
        return all(type(k) is str and is_plain_json(v) for k, v in x.items())
    return False


# ====================================================================== contract
def test_remembered_generators_are_tick_rng():
    _, eng = mech("tech_cascade", 0)
    for world, t in ((1, 0), (1, 7), (1042, 143), (3, "init")):
        a = [tick_rng(world, RIG, t).standard_normal(9).tolist(), tick_rng(world, RIG, t).random(),
             int(tick_rng(world, RIG, t).poisson(37.5))]
        b = [rng_at(world, t).standard_normal(9).tolist(), rng_at(world, t).random(), int(rng_at(world, t).poisson(37.5))]
        assert a == b == [rng_at(world, t).standard_normal(9).tolist(), rng_at(world, t).random(),
                          int(rng_at(world, t).poisson(37.5))]
    st = eng.init_state(tick_rng(8, RIG, "init"))
    for t in range(30):
        r1 = eng.step(st, defaults(eng), tick_rng(8, RIG, t), t)
        r2 = eng.step(st, defaults(eng), rng_at(8, t), t)
        assert result_digest(r1) == result_digest(r2)
        st = r1.state


def test_engine_and_families_are_registered():
    assert ENGINES["techsys"] is TechSys
    assert set(mech_families(engine="techsys")) == set(FAMILIES)
    split = {"tech_pipeline": "val", "tech_handoff": "val", "tech_contention": "test", "tech_cascade": "test",
             "tech_redundant": "test", "tech_lostwork": "test", "tech_shared_pool": "test"}
    for name in FAMILIES:
        fam = MECH_FAMILIES[name]
        assert fam.engine == "techsys" and fam.transfer == "cross_impl" and fam.split == split[name]
        assert fam.seats == (2 if name in ("tech_handoff", "tech_shared_pool") else 1)
        m, eng = mech(name, 0)
        assert eng.n_seats == fam.seats and m["config"]["family"] == name
        assert set(m["config"]["motifs"]) == set(fam.motifs)
    with pytest.raises(ValueError):
        TechSys.generate(np.random.default_rng(0), {"scenario": "no_such_scenario"})


@pytest.mark.parametrize("family", FAMILIES)
def test_config_is_plain_json_and_generation_is_deterministic(family):
    m, eng = mech(family, 3)
    assert is_plain_json(m)
    assert json.loads(json.dumps(m)) == m
    again = make_mechanism(family, 3)
    assert digest(again) == digest(m)
    assert digest(make_mechanism(family, 4)) != digest(m)
    # overrides reach the generator, and a config rebuilt from JSON drives an identical trajectory
    short = make_mechanism(family, 3, {"horizon": 48})
    assert build_engine(short).horizon == 48
    twin = build_engine(json.loads(json.dumps(m)))
    a = simulate(eng, defaults(eng), ticks=40, world=5, keep=True)
    b = simulate(twin, defaults(twin), ticks=40, world=5, keep=True)
    assert [result_digest(r) for r in a["rows"]] == [result_digest(r) for r in b["rows"]]


@pytest.mark.parametrize("family", FAMILIES)
def test_public_surface_is_opaque(family):
    forbidden = ("queue", "batch", "cache", "worker", "retry", "retri", "pool", "rout", "timeout", "deadline",
                 "stage", "pipeline", "admission", "limit", "latency", "capacity", "thrash", "memory", "shed",
                 "backlog", "depend", "path", "split", "buffer")
    for seed in range(3):
        m, eng = mech(family, seed)
        cfg = m["config"]
        names = []
        for j in range(eng.n_seats):
            spec = eng.public_spec(j)
            assert is_plain_json(spec)
            for lv in eng.levers(j):
                assert NAME.match(lv.name) and lv.doc == "" and lv.lo <= lv.default <= lv.hi and lv.lo < lv.hi
                names.append(lv.name)
            for g in eng.gauges(j):
                assert NAME.match(g.name) and g.doc == ""
                names.append(g.name)
            brief = eng.brief(j).lower()
            assert not any(w in brief for w in forbidden), brief
            assert set(spec["levers"][0]) == {"name", "min", "max", "default", "integer", "doc"}
        assert len(names) == len(set(names)), "a name is used twice"
        assert len(names) == len(cfg["levers"]) + len(cfg["gauges"])
    # the order of levers is seeded: across seeds the same kind does not sit in the same position
    first = {mech(family, s)[0]["config"]["levers"][0]["kind"] for s in range(12)}
    assert len(first) > 1


def test_hooks_for_the_host_modules():
    """What the facility module, the validity checks and the oracle look for on an engine: a declared demand
    income, the properties the engine claims, and a constructive optimum (private, like `truth`)."""
    for family in FAMILIES:
        for seed in range(2):
            m, eng = mech(family, seed)
            cfg = m["config"]
            assert isinstance(cfg["income_cents_per_tick"], int) and cfg["income_cents_per_tick"] > 0
            assert eng.claims == (("default_zero", "random_loses", "corners_bad") if eng.n_seats == 1
                                  else ("default_zero", "corners_bad"))
            cuts = sorted({r["at"] for r in eng.truth()["regimes"]})
            for seat in range(eng.n_seats):
                sched = eng.constructive_optimum(seat)["schedule"]
                assert [e["from"] for e in sched] == [0] + cuts
                for e in sched:
                    assert e["settings"] == eng.clip_settings(seat, e["settings"]), "complete and inside the ranges"
                    assert e["settings"] == reference_settings(cfg, "tuned", at=e["from"])[seat]
                assert sched[-1]["settings"] == reference_settings(cfg, "tuned")[seat]
            # nothing public leaks it
            assert not any(k in eng.public_spec(0) for k in ("income_cents_per_tick", "claims", "schedule"))


def test_redundant_constructive_schedule_switches_paths_and_beats_any_constant():
    better = 0
    for seed in range(10):
        m, eng = mech("tech_redundant", seed)
        cfg = m["config"]
        main, split = cfg["design"]["main"], cfg["design"]["split"]
        (route,) = levers_of(cfg, "route")
        to_main = (lambda x: x) if cfg["stages"][split]["next"][0] == main else (lambda x: 1.0 - x)
        before, after = eng.constructive_optimum(0)["schedule"]
        assert to_main(before["settings"][route["name"]]) >= 0.8 and to_main(after["settings"][route["name"]]) <= 0.2
        const = simulate(eng, reference_settings(cfg, "tuned"), world=1000 + seed)
        piecewise = simulate(eng, None, world=1000 + seed,
                             policy=lambda t: [before["settings"]] if t < after["from"] else [after["settings"]])
        stale = simulate(eng, [before["settings"]], world=1000 + seed)
        better += piecewise["net"] > const["net"] > stale["net"]
    assert better >= 9


def test_families_differ_in_what_can_be_seen():
    kinds = {f: {g["kind"] for g in mech(f, 0)[0]["config"]["gauges"]} for f in FAMILIES}
    assert "pool" not in kinds["tech_pipeline"] and "pool" in kinds["tech_contention"]
    assert "late" not in kinds["tech_contention"] and "offered" not in kinds["tech_contention"]
    assert "late" not in kinds["tech_cascade"] and "late" in kinds["tech_lostwork"]
    assert "offered" not in kinds["tech_redundant"]
    # distractors exist exactly where the family says so
    for f in FAMILIES:
        has = bool(levers_of(mech(f, 0)[0]["config"], "noop"))
        assert has == ("distractor" in MECH_FAMILIES[f].motifs)
    # hand-off: each seat sees only its own half; only the downstream seat sees deliveries
    cfg = mech("tech_handoff", 0)[0]["config"]
    by_seat = [{(g["kind"], tuple(g["targets"])) for g in cfg["gauges"] if g["seat"] == j} for j in (0, 1)]
    assert not by_seat[0] & by_seat[1] - {("err", ())}
    assert {k for k, _ in by_seat[0]} == {"shed", "offered", "queue", "err", "handoff"}
    assert {k for k, _ in by_seat[1]} == {"done", "late", "lat50", "lat95", "tout", "queue", "err"}
    assert {t for k, t in by_seat[0] if k == "queue"} == {(0,), (1,)}
    assert {t for k, t in by_seat[1] if k == "queue"} == {(2,), (3,)}
    for lv in cfg["levers"]:
        if lv["kind"] in ("workers", "batch"):
            assert lv["seat"] == cfg["stages"][lv["targets"][0]]["seat"]
    # shared pool: each seat controls and sees only its own line, and both see the pool
    cfg = mech("tech_shared_pool", 0)[0]["config"]
    for g in cfg["gauges"]:
        if g["kind"] == "queue":
            assert cfg["stages"][g["targets"][0]]["seat"] == g["seat"]
        elif g["kind"] not in ("pool", "err"):
            assert g["targets"][0] == g["seat"]
    assert sum(g["kind"] == "pool" for g in cfg["gauges"]) == 2
    assert all(ln["share"] == [1.0 if j == li else 0.0 for j in range(2)] for li, ln in enumerate(cfg["lines"]))


# ====================================================================== purity
@pytest.mark.parametrize("family", FAMILIES)
def test_step_is_pure_and_bit_identical(family):
    _, eng = mech(family, 1)
    g = np.random.default_rng(7)
    st = eng.init_state(tick_rng(9, RIG, "init"))
    assert state_digest(st) == state_digest(eng.init_state(tick_rng(9, RIG, "init")))
    for t in range(60):
        s = random_settings(eng, g) if t % 7 == 0 else (defaults(eng) if t % 7 == 3 else s)
        before, s_before = state_digest(st), json.dumps(s, sort_keys=True)
        r1 = eng.step(st, s, tick_rng(9, RIG, t), t)
        assert state_digest(st) == before and json.dumps(s, sort_keys=True) == s_before, "inputs were mutated"
        r2 = eng.step(st, [dict(reversed(list(x.items()))) for x in s], tick_rng(9, RIG, t), t)
        assert result_digest(r1) == result_digest(r2), "step is not a pure function of its inputs"
        for k, v in r1.state.items():
            assert isinstance(v, np.ndarray) and v is not st[k] and not np.shares_memory(v, st[k])
        assert is_plain_json(r1.info) and is_plain_json(r1.readings)
        assert set(r1.readings[0]) == {gg.name for gg in eng.gauges(0)}
        assert len(r1.revenue) == len(r1.cost) == len(r1.score) == len(r1.readings) == eng.n_seats
        st = r1.state
    # a different draw gives a different tick; a fork advanced identically stays identical
    assert (result_digest(eng.step(st, s, tick_rng(9, RIG, 60), 60))
            != result_digest(eng.step(st, s, tick_rng(10, RIG, 60), 60)))
    a = simulate(eng, defaults(eng), ticks=30, world=3, state=st, t0=60)
    b = simulate(eng, defaults(eng), ticks=30, world=3, state={k: v.copy() for k, v in st.items()}, t0=60)
    assert state_digest(a["state"]) == state_digest(b["state"]) and a["net"] == b["net"]


@pytest.mark.parametrize("family", FAMILIES)
def test_unclipped_settings_behave_like_clipped_ones(family):
    """`step` re-clips what it is given exactly as `Engine.clip_settings` would."""
    _, eng = mech(family, 2)
    g = np.random.default_rng(3)
    st = simulate(eng, defaults(eng), ticks=20, world=4)["state"]
    junk = [float("nan"), float("inf"), -float("inf"), -1e12, 1e12, "seven", None, 3.7, -0.2, [1]]
    for t in range(20, 60):
        raw = []
        for j in range(eng.n_seats):
            d = {}
            for lv in eng.levers(j):
                u = g.random()
                if u < 0.5:
                    d[lv.name] = junk[int(g.integers(len(junk)))]
                elif u < 0.8:
                    d[lv.name] = float(g.uniform(lv.lo - 5, lv.hi + 5))
            d["no_such_lever"] = 1.0
            raw.append(d)
        clean = [eng.clip_settings(j, raw[j]) for j in range(eng.n_seats)]
        r_raw = eng.step(st, raw, tick_rng(4, RIG, t), t)
        r_clean = eng.step(st, clean, tick_rng(4, RIG, t), t)
        assert result_digest(r_raw) == result_digest(r_clean)
        st = r_raw.state


# ====================================================================== invariants
@pytest.mark.parametrize("family", FAMILIES)
def test_jobs_and_value_are_conserved_every_tick(family):
    for seed in range(3):
        m, eng = mech(family, seed)
        cfg = m["config"]
        g = np.random.default_rng([seed, 5])
        st = eng.init_state(tick_rng(seed, RIG, "init"))
        value = [ln["value"] for ln in cfg["lines"]]
        tot = {"arrivals": 0, "completed": 0, "late": 0, "dropped": 0, "shed": 0}
        s = defaults(eng)
        seen = {"late": 0, "dropped": 0, "shed": 0, "completed": 0}
        money = {"value_arrived": 0.0, "value_completed": 0.0, "value_lost": 0.0}
        for t in range(200):
            if t % 9 == 0:
                u = g.random()
                s = random_settings(eng, g) if u < 0.5 else (corner_settings(eng, g) if u < 0.75 else defaults(eng))
            queued_before = int(st["q"].sum())
            r = eng.step(st, s, tick_rng(seed, RIG, t), t)
            st, i = r.state, r.info
            assert int(st["q"].min()) >= 0, "negative queue"
            queued = int(st["q"].sum())
            assert i["queued"] == queued and i["queued_before"] == queued_before
            assert i["queue"] == [int(x) for x in st["q"].sum(axis=1)]
            # jobs: exactly, this tick
            assert i["arrivals"] == i["completed"] + i["late"] + i["dropped"] + i["shed"] + queued - queued_before
            for k in tot:
                assert isinstance(i[k], int) and i[k] >= 0
                tot[k] += i[k]
            assert tot["arrivals"] == tot["completed"] + tot["late"] + tot["dropped"] + tot["shed"] + queued
            # per line, and the reasons for loss
            lost = 0.0
            for li, ln in enumerate(i["lines"]):
                assert ln["dropped"] == ln["overflow"] + ln["timed_out"] + ln["failed"] + ln["abandoned"]
                assert ln["arrivals"] == ln["admitted"] + ln["shed"] + ln["overflow"]
                by = ln["lost_by_reason"]
                assert by["shed"] == value[li] * ln["shed"] and by["timeout"] == value[li] * ln["timed_out"]
                assert by["failure"] == value[li] * ln["failed"]
                assert by["overload"] == value[li] * (ln["overflow"] + ln["abandoned"] + ln["late"])
                assert ln["value_lost"] == pytest.approx(sum(by.values()), rel=1e-12, abs=1e-9)
                lost += ln["value_lost"]
            assert set(i["lost_by_reason"]) == {"overload", "timeout", "failure", "shed"}
            assert i["value_lost"] == pytest.approx(lost, rel=1e-12, abs=1e-9)
            assert sum(i["lost_by_reason"].values()) == pytest.approx(i["value_lost"], rel=1e-12, abs=1e-9)
            assert sum(ln[k] for ln in i["lines"] for k in ("completed",)) == i["completed"]
            for k in seen:
                seen[k] += i[k]
            for k in money:
                money[k] += i[k]
            assert i["value_completed"] == sum(r.revenue)
            # value: what has arrived so far is paid for, lost, or still inside
            inside = sum(value[cfg["stages"][j]["line"]] * int(st["q"][j].sum()) for j in range(len(cfg["stages"])))
            assert money["value_arrived"] == pytest.approx(
                money["value_completed"] + money["value_lost"] + inside, rel=1e-9, abs=1e-6)
        assert seen["completed"] > 0
        assert seen["dropped"] + seen["shed"] + seen["late"] > 0, "the random schedule never lost a job"


def test_more_than_one_service_round_per_tick_keeps_the_invariants():
    """`substeps` > 1 (not used by the shipped families) runs several service rounds per tick."""
    m = make_mechanism("tech_redundant", 0, {"substeps": 3})
    eng = build_engine(m)
    assert m["config"]["substeps"] == 3
    g = np.random.default_rng(0)
    st = eng.init_state(tick_rng(1, RIG, "init"))
    total = 0
    for t in range(150):
        s = random_settings(eng, g) if t % 10 < 4 else defaults(eng)
        before = int(st["q"].sum())
        r = eng.step(st, s, tick_rng(1, RIG, t), t)
        assert result_digest(r) == result_digest(eng.step(st, s, tick_rng(1, RIG, t), t))
        st, i = r.state, r.info
        assert int(st["q"].min()) >= 0
        assert i["arrivals"] == i["completed"] + i["late"] + i["dropped"] + i["shed"] + int(st["q"].sum()) - before
        total += i["completed"]
    assert total > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_revenue_is_paid_by_customers_and_bounded_by_arrivals(family):
    for seed in range(2):
        m, eng = mech(family, seed)
        cfg = m["config"]
        value = [ln["value"] for ln in cfg["lines"]]
        share = [ln["share"] for ln in cfg["lines"]]
        for style in ("default", "tuned", "hi"):
            s = {"default": defaults(eng), "tuned": reference_settings(cfg, "tuned"),
                 "hi": corner_settings(eng, which="hi")}[style]
            run = simulate(eng, s, world=20 + seed, keep=True)
            cum_rev = [0.0] * eng.n_seats
            cum_cap = [0.0] * eng.n_seats
            for r in run["rows"]:
                for j in range(eng.n_seats):
                    paid = sum(value[li] * ln["completed"] * share[li][j] for li, ln in enumerate(r.info["lines"]))
                    assert r.revenue[j] == pytest.approx(paid, rel=1e-12, abs=1e-9) and r.revenue[j] >= 0.0
                    assert r.cost[j] > 0.0
                    cum_rev[j] += r.revenue[j]
                    cum_cap[j] += sum(value[li] * ln["arrivals"] * share[li][j]
                                      for li, ln in enumerate(r.info["lines"]))
                    assert cum_rev[j] <= cum_cap[j] + 1e-6, "revenue exceeds what the arrivals could pay"
                assert sum(r.revenue) <= cfg["income_cents_per_tick"], "one tick earned more than the declared income"
                if cfg["objective"] == "net":
                    assert r.score == [r.revenue[j] - r.cost[j] for j in range(eng.n_seats)]
                elif cfg["objective"] == "team_net":
                    assert r.score[0] == r.score[1] == (sum(r.revenue) - sum(r.cost)) / 2
                    assert r.revenue[0] == r.revenue[1]
            assert sum(cum_rev) <= run["value"] + 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_state_stays_bounded_for_2000_ticks_under_any_settings(family):
    m, eng = mech(family, 0)
    cfg = m["config"]
    buf = np.array([s["buf"] for s in cfg["stages"]])
    wmax = max(lv["hi"] for lv in cfg["levers"] if lv["kind"] == "workers")
    bmax = max([lv["hi"] for lv in cfg["levers"] if lv["kind"] == "batch"] + [1.0])
    cap_bound = max(wmax * s["mu"] * 8.0 * 12.0 for s in cfg["stages"])      # workers x rate x batch and cache gains
    per_tick = sum(ln["arrival"]["base"] * ln["value"] for ln in cfg["lines"])
    g = np.random.default_rng(11)
    st = eng.init_state(tick_rng(2, RIG, "init"))
    junk = [float("nan"), float("inf"), -1e300, 1e300, "x", None]
    s = defaults(eng)
    peak_q = 0
    peak_rev = 0.0
    for t in range(2000):
        if t % 23 == 0:
            mode = int(g.integers(6))
            if mode == 0:
                s = corner_settings(eng, which="hi")
            elif mode == 1:
                s = corner_settings(eng, which="lo")
            elif mode == 2:
                s = corner_settings(eng, g)
            elif mode == 3:
                s = random_settings(eng, g)
            elif mode == 4:
                s = [{lv.name: junk[int(g.integers(len(junk)))] for lv in eng.levers(j)} for j in range(eng.n_seats)]
            else:
                s = [{} for _ in range(eng.n_seats)]
        r = eng.step(st, s, tick_rng(2, RIG, t), t)
        st = r.state
        q = st["q"]
        assert q.dtype == np.int64 and int(q.min()) >= 0
        assert (q.sum(axis=1) <= buf).all(), "a queue outgrew its buffer"
        for k, v in st.items():
            assert np.isfinite(v).all(), k
        assert 0.0 <= st["wh"].min() and st["wh"].max() <= wmax
        assert st["car"][0].min() >= 0.0 and st["car"][0].max() <= cap_bound + bmax
        assert st["car"][1:].min() >= 0.0 and st["car"][1:].max() < 1.0 + 1e-9
        assert np.abs(st["aux"]).max() <= max(40.0 * per_tick, cfg["ages"] + 1) + 1e-6
        for j in range(eng.n_seats):
            assert all(math.isfinite(v) and v >= 0.0 for v in r.readings[j].values())
            assert math.isfinite(r.revenue[j]) and math.isfinite(r.cost[j]) and math.isfinite(r.score[j])
        assert max(r.info["capacity"]) <= cap_bound
        assert sum(r.revenue) <= cfg["income_cents_per_tick"], "one tick earned more than the declared income"
        peak_q = max(peak_q, int(q.sum()))
        peak_rev = max(peak_rev, sum(r.revenue))
    assert peak_q > 0
    # the declared income is a real ceiling, not an astronomically loose one: it is of the order of what the
    # facility earns in its best tick (a backlog flushed at the delivery ceiling)
    assert cfg["income_cents_per_tick"] < 12 * per_tick and peak_rev > 0.05 * cfg["income_cents_per_tick"]


# ====================================================================== truth
@pytest.mark.parametrize("family", FAMILIES)
def test_truth_graph_is_well_formed(family):
    for seed in range(3):
        m, eng = mech(family, seed)
        cfg = m["config"]
        tr = eng.truth()
        assert set(tr) == {"engine", "family", "nodes", "edges", "regimes", "motifs", "notes"}
        assert tr["engine"] == "techsys" and tr["family"] == family and is_plain_json(tr)
        ids = [n["id"] for n in tr["nodes"]]
        assert len(ids) == len(set(ids))
        for n in tr["nodes"]:
            assert set(n) == {"id", "role", "seat", "public"}
            assert n["role"] in ("lever", "latent", "gauge", "outcome", "exogenous")
            assert (n["public"] is not None) == (n["role"] in ("lever", "gauge"))
        public = {(n["role"], n["seat"], n["public"]) for n in tr["nodes"] if n["public"] is not None}
        assert public == ({("lever", lv["seat"], lv["name"]) for lv in cfg["levers"]}
                          | {("gauge", g["seat"], g["name"]) for g in cfg["gauges"]})
        assert sum(n["role"] == "outcome" for n in tr["nodes"]) == eng.n_seats
        for e in tr["edges"]:
            assert set(e) == {"src", "dst", "sign", "lag", "kind"}
            assert e["src"] in ids and e["dst"] in ids and e["sign"] in (-1, 0, 1) and e["lag"] >= 0
        role = {n["id"]: n["role"] for n in tr["nodes"]}
        assert not any(role[e["dst"]] in ("lever", "exogenous") for e in tr["edges"])
        assert not any(role[e["src"]] in ("gauge", "outcome") for e in tr["edges"])
        # every gauge reads exactly one quantity; every working lever acts on something; distractors on nothing
        for n in tr["nodes"]:
            if n["role"] == "gauge":
                assert sum(e["dst"] == n["id"] for e in tr["edges"]) == 1
            if n["role"] == "lever":
                assert bool(sum(e["src"] == n["id"] for e in tr["edges"])) != n["id"].startswith("lever.noop")
        # scheduled changes are exactly the configured ones
        assert tr["regimes"] == sorted(tr["regimes"], key=lambda r: (r["at"], r["what"]))
        ats = sorted({r["at"] for r in tr["regimes"]})
        want = sorted({r["at"] for r in cfg["regimes"]} | {r["until"] for r in cfg["regimes"] if r["until"] is not None})
        assert ats == want and all(0 < a < eng.horizon for a in ats)
        assert bool(ats) == ("regime_change" in MECH_FAMILIES[family].motifs)


SCALE_FREE = {"pool": 1.0, "errors": 1.0, "latency": 1.0}


def node_scale(cfg: dict, node: str, a: float, b: float) -> float:
    head = node.split(".")[0]
    if head in SCALE_FREE:
        return SCALE_FREE[head]
    if head == "capacity":
        return max(abs(a), abs(b), 1e-9)
    rate = sum(ln["arrival"]["base"] for ln in cfg["lines"])
    if head in ("revenue", "cost", "score"):
        return max(ln["value"] for ln in cfg["lines"]) * rate
    return rate


def mean_nodes(eng: Engine, settings: list[dict], nodes: list[str], ticks: int, world: int, skip: int = 12) -> dict:
    st = eng.init_state(rng_at(world, "init"))
    acc = dict.fromkeys(nodes, 0.0)
    for t in range(ticks):
        r = eng.step(st, settings, rng_at(world, t), t)
        st = r.state
        if t >= skip:
            for n in nodes:
                acc[n] += node_value(r, n)
    return {n: v / (ticks - skip) for n, v in acc.items()}


@pytest.mark.parametrize("family", FAMILIES)
def test_implied_monotone_effects_hold_in_simulation(family):
    """Every (lever, node, sign) the truth graph implies is checked with common random numbers: moving the lever
    up against moving it down, from two operating points and with two step sizes, never moves the node the wrong
    way, and most implied effects are visibly active."""
    total = active = 0
    for seed in (0, 1):
        m, eng = mech(family, seed)
        cfg = m["config"]
        tr = eng.truth()
        claims = implied_effects(tr)
        assert claims, "the graph implies no monotone effect at all"
        lever = {n["id"]: (n["seat"], n["public"]) for n in tr["nodes"] if n["role"] == "lever"}
        by_name = {lv["name"]: lv for lv in cfg["levers"]}
        bases = [defaults(eng), reference_settings(cfg, "tuned")]
        for lid in sorted({c[0] for c in claims}):
            seat, name = lever[lid]
            lv = by_name[name]
            mine = [(n, sg) for (l, n, sg) in claims if l == lid]
            nodes = [n for n, _ in mine]
            best = dict.fromkeys(nodes, 0.0)
            for base_settings in bases:
                for frac in (0.15, 0.4):
                    span = lv["hi"] - lv["lo"]
                    up = with_lever(base_settings, lv, base_settings[seat][name] + frac * span)
                    dn = with_lever(base_settings, lv, base_settings[seat][name] - frac * span)
                    if up[seat][name] == dn[seat][name]:
                        continue
                    a = mean_nodes(eng, up, nodes, 96, 31 + seed)
                    b = mean_nodes(eng, dn, nodes, 96, 31 + seed)
                    for n, sg in mine:
                        eff = sg * (a[n] - b[n]) / node_scale(cfg, n, a[n], b[n])
                        assert eff > -0.02, f"{family}/{seed}: {lid} -> {n} declared {sg:+d}, measured {eff:+.3f}"
                        best[n] = max(best[n], eff)
            total += len(mine)
            active += sum(v > 0.01 for v in best.values())
    print(f"\n[truth] {family}: {total} implied monotone effects over 2 seeds, {active} active, 0 violated")
    assert active >= 0.25 * total and active >= 8


def test_every_family_implies_the_basic_effects():
    """The implied effects are not vacuous: in every family each worker lever raises its stage's capacity and its
    seat's cost, and the admission limit raises what is admitted."""
    for family in FAMILIES:
        m, eng = mech(family, 0)
        cfg = m["config"]
        claims = set(implied_effects(eng.truth()))
        for lv in levers_of(cfg, "workers"):
            i = lv["targets"][0]
            assert (f"lever.workers.s{i}", f"cost.seat{lv['seat']}", +1) in claims
            if cfg["pool"] is None:
                assert (f"lever.workers.s{i}", f"capacity.s{i}", +1) in claims
            else:       # with a shared pool the same lever also slows the stage: no monotone claim, by design
                assert not any(c[0] == f"lever.workers.s{i}" and c[1] == f"capacity.s{i}" for c in claims)
                assert (f"lever.workers.s{i}", "pool", +1) in claims
        storm = any(s["stale"] is not None for s in cfg["stages"])
        for lv in levers_of(cfg, "admit"):
            claim = (f"lever.admit.l{lv['targets'][0]}", f"admitted.l{lv['targets'][0]}", +1)
            # where attempts can go stale, everything in the flow sits inside a loop through a non-monotone
            # link (the storm), and nothing about it is claimed to be monotone
            assert (claim in claims) != storm
        if storm:
            assert {c[1].split(".")[0] for c in claims} == {"capacity", "cost"}


@pytest.mark.parametrize("family", FAMILIES)
def test_worker_start_up_lag_is_the_declared_lag(family):
    """Scale-up shows in capacity exactly `lag` ticks after the request, scale-down at once; the lag is the one
    on the lever->capacity edge of the truth graph."""
    m, eng = mech(family, 1)
    cfg = m["config"]
    lag_of = {e["src"]: e["lag"] for e in eng.truth()["edges"] if e["kind"] == "delayed"}
    base_s = defaults(eng)
    assert {cfg["stages"][lv["targets"][0]]["lag"] for lv in levers_of(cfg, "workers")} != {0}
    for lv in levers_of(cfg, "workers"):
        i = lv["targets"][0]
        lag = lag_of[f"lever.workers.s{i}"]
        assert lag == cfg["stages"][i]["lag"]
        more = with_lever(base_s, lv, lv["default"] + 3)
        less = with_lever(base_s, lv, max(lv["lo"], lv["default"] - 2))
        t_up, t_down = 20, 40
        st = eng.init_state(tick_rng(6, RIG, "init"))
        eff = []
        for t in range(60):
            s = base_s if t < t_up else (more if t < t_down else less)
            r = eng.step(st, s, tick_rng(6, RIG, t), t)
            st = r.state
            eff.append(r.info["workers_effective"][i])
        d, up, dn = lv["default"], more[lv["seat"]][lv["name"]], less[lv["seat"]][lv["name"]]
        assert eff[:t_up + lag] == [d] * (t_up + lag), "capacity moved before the lag had passed"
        assert eff[t_up + lag:t_down] == [up] * (t_down - t_up - lag)
        assert eff[t_down:] == [dn] * 20, "scale-down is immediate"


NON_MONOTONE_KINDS = ("knee", "thrash", "expiry", "storm", "split")
PEAKED = ("knee", "thrash", "split")        # must rise and fall along the lever at a single operating point


def response_curve(eng, base_settings, lv, node, ticks, world, points=11) -> np.ndarray:
    grid = sorted({with_lever(base_settings, lv, lv["lo"] + (lv["hi"] - lv["lo"]) * k / (points - 1))[lv["seat"]][lv["name"]]
                   for k in range(points)})
    return np.array([mean_nodes(eng, with_lever(base_settings, lv, x), [node], ticks, world)[node] for x in grid])


def curve_shape(ys: np.ndarray) -> tuple[float, float, float]:
    """(largest rise, largest fall, peakedness) of a response along a lever, each relative to the largest value.
    Peakedness is the smaller of the rise to an interior peak and the fall after it (or the same for a valley)."""
    scale = max(float(np.abs(ys).max()), 1e-9)
    up = max(float(ys[j] - ys[:j + 1].min()) for j in range(len(ys))) / scale
    down = max(float(ys[:j + 1].max() - ys[j]) for j in range(len(ys))) / scale
    k = int(ys.argmax())
    peak = min(ys[k] - ys[:k + 1].min(), ys[k] - ys[k:].min())
    k = int(ys.argmin())
    valley = min(ys[:k + 1].max() - ys[k], ys[k:].max() - ys[k])
    return up, down, float(max(peak, valley)) / scale


@pytest.mark.parametrize("family", FAMILIES)
def test_declared_non_monotone_responses_change_direction(family):
    """Every sign-0 edge that starts at a lever is swept across the lever's whole range from three operating
    points (the defaults, the tuned settings, the defaults with the admission limit lifted).

    * batch knee, pool thrash, traffic split: at one operating point the response rises to an interior peak and
      falls again, each by more than 3% of its largest value;
    * expiry (timeout) and retry storm, whose direction depends on the state of the system: the response rises by
      more than 3% along the lever at some operating point and falls by more than 3% at some operating point
      (often the same one)."""
    seen: dict[str, list[float]] = {}
    for seed in (0, 1):
        m, eng = mech(family, seed)
        cfg = m["config"]
        tr = eng.truth()
        lever = {n["id"]: n["public"] for n in tr["nodes"] if n["role"] == "lever"}
        by_name = {lv["name"]: lv for lv in cfg["levers"]}
        declared = [e for e in tr["edges"] if e["sign"] == 0 and e["src"] in lever]
        assert declared and all(e["kind"] in NON_MONOTONE_KINDS for e in declared)
        loose = defaults(eng)
        for lv in levers_of(cfg, "admit"):
            loose = with_lever(loose, lv, lv["hi"])
        points = {"default": defaults(eng), "tuned": reference_settings(cfg, "tuned"), "loose": loose}
        for e in declared:
            lv = by_name[lever[e["src"]]]
            up = down = peaked = 0.0
            for name in (("default", "tuned", "loose") if e["kind"] in PEAKED else ("loose", "tuned", "default")):
                u, d, pk = curve_shape(response_curve(eng, points[name], lv, e["dst"], eng.horizon, 41 + seed))
                up, down, peaked = max(up, u), max(down, d), max(peaked, pk)
                if (peaked if e["kind"] in PEAKED else min(up, down)) > 0.1:
                    break                   # clear enough; the other operating points are not needed
            got = peaked if e["kind"] in PEAKED else min(up, down)
            assert got > 0.03, (f"{family}/{seed}: {e['src']} -> {e['dst']} ({e['kind']}) looks monotone: "
                                f"rise {up:.3f} fall {down:.3f} peaked {peaked:.3f}")
            seen.setdefault(e["kind"], []).append(got)
    print(f"\n[non-monotone] {family}: " + ", ".join(
        f"{k} x{len(v)} min {min(v):.2f} median {float(np.median(v)):.2f}" for k, v in sorted(seen.items())))
    want = {"tech_pipeline": {"knee", "expiry"}, "tech_contention": {"knee", "thrash", "expiry"},
            "tech_cascade": {"knee", "storm", "expiry"}, "tech_redundant": {"split", "expiry"},
            "tech_lostwork": {"knee", "split", "expiry"}, "tech_handoff": {"knee", "expiry"},
            "tech_shared_pool": {"thrash", "expiry"}}[family]
    assert want <= set(seen)


def test_thrash_is_declared_exactly_where_one_lever_can_thrash_the_pool():
    for family in ("tech_contention", "tech_shared_pool", "tech_lostwork"):
        for seed in range(4):
            m, eng = mech(family, seed)
            cfg = m["config"]
            tr = eng.truth()
            thrash = {e["src"] for e in tr["edges"] if e["kind"] == "thrash"}
            lever = {n["public"]: n["id"] for n in tr["nodes"] if n["role"] == "lever"}
            d = defaults(eng)
            for lv in cfg["levers"]:
                if lv["kind"] not in ("workers", "cache"):
                    continue
                r = simulate(eng, with_lever(d, lv, lv["hi"]), ticks=3, world=1, keep=True)["rows"][-1]
                assert (r.info["slowdown"] < 0.85) == (lever[lv["name"]] in thrash)
            if family == "tech_shared_pool":       # the stage that draws on the pool, not the ones that hardly do
                work = {int(k) for k in cfg["design"]["grab"]}
                assert {f"lever.workers.s{i}" for i in work} | {f"lever.cache.s{i}" for i in work} == thrash


def test_batch_knee_is_where_the_config_says():
    """Capacity as a function of batch size peaks exactly at the hidden knee, and beyond it jobs also wait."""
    for family, seed in (("tech_pipeline", 0), ("tech_pipeline", 1), ("tech_handoff", 0), ("tech_cascade", 0)):
        m, eng = mech(family, seed)
        cfg = m["config"]
        for lv in levers_of(cfg, "batch"):
            i = lv["targets"][0]
            knee = cfg["stages"][i]["batch"]["knee"]
            base_s = reference_settings(cfg, "tuned")
            caps = {}
            for b in (1, knee - 1, knee, knee + 1, int(lv["hi"])):
                r = simulate(eng, with_lever(base_s, lv, b), ticks=30, world=8, keep=True)
                caps[b] = float(np.mean([x.info["capacity"][i] for x in r["rows"]]))
            assert caps[knee] == max(caps.values())
            assert caps[1] < caps[knee - 1] < caps[knee] > caps[knee + 1] > caps[int(lv["hi"])]
            assert caps[knee] > 1.5 * caps[1] and caps[int(lv["hi"])] < 0.5 * caps[knee]


def test_distractor_levers_do_nothing():
    for family in ("tech_contention", "tech_cascade"):
        for seed in range(3):
            m, eng = mech(family, seed)
            noop = levers_of(m["config"], "noop")
            assert len(noop) == 1
            lv = noop[0]
            base_s = defaults(eng)
            a = simulate(eng, with_lever(base_s, lv, lv["lo"]), ticks=60, world=9, keep=True)
            b = simulate(eng, with_lever(base_s, lv, lv["hi"]), ticks=60, world=9, keep=True)
            assert [result_digest(r) for r in a["rows"]] == [result_digest(r) for r in b["rows"]]


# ====================================================================== mechanics, one at a time
def quiet(cfg: dict, **changes) -> TechSys:
    """The same system with every source of failure and service noise removed (and optional other changes)."""
    c = json.loads(json.dumps(cfg))
    for s in c["stages"]:
        s["err"] = 0.0
        s["jitter"] = 0.0
    c.update(changes)
    return TechSys(c)


def test_deadlines_delivery_delay_and_timeout_at_the_last_stage():
    """A cohort of known age is placed in front of the last stage with all arrivals shed. It is paid for in
    full up to the deadline, in the hidden fractions one and two ticks after it, not at all later; its latency
    is its queue age plus the delivery delay; and a timeout shorter than that latency discards it."""
    for seed in range(4):
        m, _ = mech("tech_pipeline", seed)
        cfg = m["config"]
        eng = quiet(cfg)
        (ln,) = cfg["lines"]
        settle, deadline, tail, value = ln["settle"], ln["deadline"], ln["tail"], ln["value"]
        assert 2 <= settle <= 3 and deadline - settle >= 2
        sink = ln["sink"]
        (admit,) = levers_of(cfg, "admit")
        (timeout,) = levers_of(cfg, "timeout")
        s = with_lever(defaults(eng), admit, 0.0)
        for lv in levers_of(cfg, "batch"):
            s = with_lever(s, lv, 1)
        n = 20
        for age in range(0, cfg["ages"] - settle + 1):
            latency = age + settle
            for ttl in (timeout["hi"], latency, latency - 1):
                if ttl < 1:
                    continue
                st = eng.init_state(tick_rng(1, RIG, "init"))
                st["q"][sink, age] = n
                r = eng.step(st, with_lever(s, timeout, ttl), tick_rng(1, RIG, 0), 0)
                i = r.info["lines"][0]
                assert i["shed"] == i["arrivals"] and r.info["queued"] == 0
                if ttl < latency:
                    assert (i["completed"], i["late"], i["timed_out"]) == (0, 0, n) and r.revenue == [0.0]
                    assert r.info["lost_by_reason"]["timeout"] == value * n
                    continue
                k = latency - deadline
                paid = n if k <= 0 else (int(n * tail[k - 1]) if k <= len(tail) else 0)
                assert (i["completed"], i["late"], i["timed_out"]) == (paid, n - paid, 0)
                assert r.revenue == [value * paid]
                assert i["latency_p50"] == pytest.approx(latency + 0.5) and i["latency_p95"] == pytest.approx(latency + 0.95)
        # a job never waits in a queue beyond what the timeout allows: after the end-of-tick sweep nothing is older
        for ttl in (settle, settle + 1, deadline, timeout["hi"]):
            run = simulate(eng, with_lever(with_lever(defaults(eng), admit, admit["hi"]), timeout, ttl), ticks=60,
                           world=3, keep=True)
            oldest = max(int(np.nonzero(r.state["q"].sum(axis=0))[0].max()) if r.state["q"].sum() else 0
                         for r in run["rows"])
            assert oldest <= max(0, ttl - settle)
            assert all(r.info["lines"][0]["latency_p95"] <= ttl + 1.0 for r in run["rows"]
                       if r.info["completed"] + r.info["late"])
        # and a timeout below the delivery delay delivers nothing at all
        dead = simulate(eng, with_lever(defaults(eng), timeout, settle - 1), ticks=40, world=3)
        assert dead["completed"] == dead["late"] == 0 and dead["dropped"] + dead["shed"] > 0


def test_arrivals_are_exogenous_and_follow_cycle_bursts_and_drift():
    for family, seed in (("tech_pipeline", 0), ("tech_contention", 1), ("tech_redundant", 2)):
        m, eng = mech(family, seed)
        cfg = m["config"]
        a = cfg["lines"][0]["arrival"]
        g = np.random.default_rng(5)
        ticks = 1440
        base_run = simulate(eng, defaults(eng), ticks=ticks, world=77, keep=True)
        other = simulate(eng, None, ticks=200, world=77, keep=True, policy=lambda t: random_settings(eng, g))
        arr = np.array([r.info["arrivals"] for r in base_run["rows"]], dtype=float)
        assert [r.info["arrivals"] for r in other["rows"]] == [int(x) for x in arr[:200]], "arrivals depend on the settings"
        burst = np.array([r.state["aux"][0] for r in base_run["rows"]]) > 0.5
        t = np.arange(ticks)
        expected = a["base"] * (1 + a["amp"] * np.sin(2 * np.pi * (t + a["phase"]) / a["period"])) \
            * (1 + a["drift"] * np.minimum(t / eng.horizon, 1.5)) * np.where(burst, a["burst_x"], 1.0)
        assert np.corrcoef(arr, expected)[0, 1] > 0.8
        assert arr.sum() == pytest.approx(expected.sum(), rel=0.03)
        # the daily cycle, the bursts, and the on/off rates of the burst chain
        phase = np.sin(2 * np.pi * (t + a["phase"]) / a["period"])
        calm = ~burst & (t >= 1.5 * eng.horizon)         # drift has stopped by then
        ratio = arr[calm & (phase > 0.5)].mean() / arr[calm & (phase < -0.5)].mean()
        assert ratio == pytest.approx((1 + 0.83 * a["amp"]) / (1 - 0.83 * a["amp"]), rel=0.12)
        assert 0.3 * a["burst_on"] < np.mean(burst[1:] & ~burst[:-1]) < 3 * a["burst_on"] + 0.01
        assert arr[burst].mean() / arr[~burst].mean() == pytest.approx(a["burst_x"], rel=0.25)
        if abs(a["drift"]) > 0.05:
            early, late_ = arr[~burst & (t < 48)].mean(), arr[~burst & (t > 1.5 * eng.horizon)].mean()
            assert (late_ - early) * a["drift"] > 0


def test_routing_weight_splits_traffic_exactly_and_a_starved_path_holds_up_the_split():
    for seed in range(3):
        m, _ = mech("tech_redundant", seed)
        cfg = m["config"]
        eng = quiet(cfg, regimes=[])
        split = cfg["design"]["split"]
        ja = cfg["stages"][split]["next"][0]
        (route,) = levers_of(cfg, "route")
        staff = defaults(eng)
        for lv in levers_of(cfg, "workers"):
            staff = with_lever(staff, lv, lv["hi"])
        for w in (0.0, 0.25, 0.6, 1.0):
            run = simulate(eng, with_lever(staff, route, w), ticks=80, world=4, keep=True)
            rows = run["rows"][12:]
            out = sum(r.info["throughput"][split] for r in rows)
            to_a = (sum(r.info["throughput"][ja] for r in rows) + rows[-1].info["queue"][ja]
                    - run["rows"][11].info["queue"][ja])
            assert abs(to_a - w * out) <= 2.0, "the split is exact up to the carried fraction"
            assert run["completed"] > 0.97 * run["arrivals"]
        # starve one path with a long timeout: its buffer fills, a backlog forms in front of the split, and the
        # traffic of the healthy path is held up behind it, so less than the healthy half arrives in time
        (timeout,) = levers_of(cfg, "timeout")
        starved = next(lv for lv in levers_of(cfg, "workers") if lv["targets"] == [ja])
        lone = with_lever(with_lever(with_lever(staff, route, 0.5), starved, 1), timeout, timeout["hi"])
        run = simulate(eng, lone, ticks=100, world=4, keep=True)
        rows = run["rows"][40:]
        arrived = sum(r.info["arrivals"] for r in rows)
        assert np.mean([r.info["queue"][ja] for r in rows]) >= 0.6 * cfg["stages"][ja]["buf"]
        assert np.mean([r.info["queue"][split] for r in rows]) > 0.5 * arrived / len(rows)
        assert sum(r.info["completed"] for r in rows) < 0.47 * arrived
        assert sum(r.info["late"] for r in rows) > 0.1 * arrived
        # the same starved path with no traffic sent to it costs nothing
        away = simulate(eng, with_lever(lone, route, 0.0), ticks=100, world=4, keep=True)["rows"][40:]
        assert sum(r.info["completed"] for r in away) > 0.97 * sum(r.info["arrivals"] for r in away)
        assert max(r.info["queue"][split] for r in away) == 0


def test_cache_has_diminishing_returns_and_draws_on_the_pool():
    for seed in range(3):
        m, eng = mech("tech_contention", seed)
        cfg = m["config"]
        free = quiet(cfg, pool={**cfg["pool"], "theta": 0.999, "size": cfg["pool"]["size"] * 1000})
        for lv in levers_of(cfg, "cache"):
            i = lv["targets"][0]
            half = cfg["stages"][i]["cache"]["half"]
            sizes = [0, round(half), round(2 * half), round(4 * half), int(lv["hi"])]
            caps, utils = [], []
            for c in sizes:
                s = with_lever(defaults(eng), lv, c)
                caps.append(simulate(free, s, ticks=3, world=1, keep=True)["rows"][-1].info["capacity"][i])
                utils.append(simulate(eng, s, ticks=3, world=1, keep=True)["rows"][-1].info["pool_util"])
            gains = np.diff(caps) / np.diff(sizes)
            assert (gains > 0).all() and (np.diff(gains) < 0).all(), "each further unit of cache adds less"
            ch = cfg["stages"][i]["cache"]
            assert caps[-1] / caps[0] == pytest.approx(techsys.cache_gain(ch, sizes[-1]), rel=1e-9)
            assert caps[-1] / caps[0] < 1.0 / (1.0 - ch["save"] * ch["hmax"])
            assert np.diff(utils) == pytest.approx(np.diff(sizes) * ch["mem"] / cfg["pool"]["size"], rel=1e-6)


# ====================================================================== calibration (20 seeds per family)
@lru_cache(maxsize=None)
def calibration(family: str) -> dict:
    rows = []
    for seed in range(N_SEEDS):
        _, eng = mech(family, seed)
        g = np.random.default_rng([seed, 99])
        world = 1000 + seed
        net = lambda s: simulate(eng, s, world=world)["net_frac"]  # noqa: E731
        rnd = [net(random_settings(eng, g)) for _ in range(4)]
        cor = [net(corner_settings(eng, g)) for _ in range(8)]
        hi = simulate(eng, corner_settings(eng, which="hi"), world=world)
        lo = net(corner_settings(eng, which="lo"))
        rows.append({"default": outcome(family, seed, "default")[0], "default_lost": outcome(family, seed, "default")[1],
                     "tuned": outcome(family, seed, "tuned")[0], "tuned_lost": outcome(family, seed, "tuned")[1],
                     "random": float(np.mean(rnd)), "random_best": max(rnd), "corner_best": max(cor + [hi["net_frac"], lo]),
                     "all_hi": hi["net_frac"], "all_hi_lost": hi["lost_frac"]})
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


@pytest.mark.parametrize("family", FAMILIES)
def test_calibration_over_twenty_seeds(family):
    """Net result as a fraction of the value that arrived, constant settings, one episode per mechanism seed:
    defaults about nothing or a small loss; settings chosen with knowledge of the mechanism clearly positive;
    uniformly random settings a loss; no corner of the lever box near the best found."""
    c = calibration(family)
    best = np.maximum(c["tuned"], c["random_best"])
    print(f"\n[calibration] {family} ({N_SEEDS} seeds), net / arriving value:"
          f"\n   defaults     mean {c['default'].mean():+.3f}  min {c['default'].min():+.3f}  max {c['default'].max():+.3f}"
          f"   (value lost at defaults: mean {c['default_lost'].mean():.2f})"
          f"\n   tuned        mean {c['tuned'].mean():+.3f}  min {c['tuned'].min():+.3f}  max {c['tuned'].max():+.3f}"
          f"   (value lost: mean {c['tuned_lost'].mean():.2f})"
          f"\n   random       mean {c['random'].mean():+.3f}  seeds with negative mean {int((c['random'] < 0).sum())}/{N_SEEDS}"
          f"   best random draw: mean {c['random_best'].mean():+.3f}"
          f"\n   best corner  mean {c['corner_best'].mean():+.3f}  max {c['corner_best'].max():+.3f}"
          f"   largest corner/best ratio {float((c['corner_best'] / best).max()):+.2f}"
          f"\n   all levers at maximum: mean {c['all_hi'].mean():+.3f}  max {c['all_hi'].max():+.3f}"
          f"  (value lost: mean {c['all_hi_lost'].mean():.2f})")
    assert -0.08 <= c["default"].mean() <= 0.0 and -0.06 <= float(np.median(c["default"])) <= 0.0
    assert c["default"].min() >= -0.15 and c["default"].max() <= 0.05
    assert (c["tuned"] > 0.1).sum() >= N_SEEDS - 1 and c["tuned"].mean() > 0.2 and c["tuned"].min() > 0.0
    assert (c["tuned"] > c["default"] + 0.1).all()
    assert c["random"].mean() < -0.1 and (c["random"] < 0).sum() >= N_SEEDS - 2
    assert (c["corner_best"] <= 0.5 * best).all(), "a corner of the lever box is close to the best found"
    assert (c["all_hi"] < c["default"]).all()


def test_more_of_everything_is_bad_where_resources_interact():
    for family in ("tech_contention", "tech_cascade"):
        c = calibration(family)
        assert (c["all_hi"] < -0.5).all() and (c["all_hi"] < c["default"] - 0.4).all()
        assert c["all_hi_lost"].mean() > c["default_lost"].mean() + 0.2
    assert calibration("tech_contention")["all_hi_lost"].min() > 0.9, "the thrashing pool should lose nearly all work"


# ---------------------------------------------------------------------- contention
def test_contention_adding_resources_makes_it_worse_until_some_are_freed():
    """At rest the pool sits just below its threshold. More workers, or more cache, for the stage that is short
    slows every stage and delivers less; taking the hoarding stage's surplus away first makes the same addition
    pay."""
    worse = better = cache_worse = 0
    for seed in range(N_SEEDS):
        m, eng = mech("tech_contention", seed)
        cfg = m["config"]
        weak, hog = cfg["design"]["weak"], cfg["design"]["hog"]
        lev = {(lv["kind"], lv["targets"][0]): lv for lv in cfg["levers"] if lv["targets"]}
        base_s = defaults(eng)
        w_weak, w_hog, c_hog, c_weak = (lev[("workers", weak)], lev[("workers", hog)], lev[("cache", hog)],
                                        lev[("cache", weak)])
        world = 1000 + seed
        run0 = simulate(eng, base_s, world=world, keep=True)
        add = with_lever(base_s, w_weak, w_weak["default"] + 8)
        run1 = simulate(eng, add, world=world, keep=True)
        util0 = np.mean([r.info["pool_util"] for r in run0["rows"]])
        util1 = np.mean([r.info["pool_util"] for r in run1["rows"]])
        assert util0 < cfg["pool"]["theta"] < util1
        assert all(r.info["slowdown"] == 1.0 for r in run0["rows"])
        slow = np.mean([r.info["slowdown"] for r in run1["rows"][12:]])
        assert slow < 0.9
        # every other stage lost capacity although nothing about it was changed
        for i in range(len(cfg["stages"])):
            if i != weak:
                c0 = np.mean([r.info["capacity"][i] for r in run0["rows"][12:]])
                c1 = np.mean([r.info["capacity"][i] for r in run1["rows"][12:]])
                assert c1 < 0.95 * c0
        worse += run1["completed"] < run0["completed"] and run1["net"] < run0["net"]
        freed = with_lever(with_lever(add, c_hog, 0.25 * c_hog["default"]), w_hog, 0.6 * w_hog["default"])
        run2 = simulate(eng, freed, world=world)
        better += run2["completed"] > run0["completed"] and run2["net"] > run0["net"]
        run3 = simulate(eng, with_lever(base_s, c_weak, c_weak["hi"]), world=world)
        cache_worse += run3["completed"] < run0["completed"]
    print(f"\n[contention] +8 workers at the short stage delivers less and earns less in {worse}/{N_SEEDS} seeds; "
          f"the same addition after freeing the hoarder's surplus beats the defaults in {better}/{N_SEEDS}; "
          f"maximum cache at the short stage delivers less in {cache_worse}/{N_SEEDS}")
    assert worse >= N_SEEDS - 2 and better >= N_SEEDS - 2 and cache_worse >= N_SEEDS - 2


# ---------------------------------------------------------------------- cascade
def test_cascade_is_metastable_and_recovers_only_by_shedding_load():
    """With the resting settings the system is healthy until the scheduled slowdown, collapses during it and stays
    collapsed after the dependency has recovered. From that same collapsed state: unchanged settings and maximum
    workers leave it collapsed; shedding all load for a few ticks and then restoring the *same* settings brings
    it back for good. Without the slowdown it never collapses."""
    tally = {"healthy_before": 0, "collapsed_after": 0, "capacity_fails": 0, "shed_recovers": 0, "no_trigger": 0}
    for seed in range(N_SEEDS):
        m, eng = mech("tech_cascade", seed)
        cfg = m["config"]
        (reg,) = cfg["regimes"]
        at, until, dep = reg["at"], reg["until"], cfg["design"]["dependency"]
        assert cfg["stages"][dep]["stale"] is not None and until < eng.horizon
        d = defaults(eng)
        (admit,) = levers_of(cfg, "admit")
        world = 1000 + seed
        act, end = until + 10, until + 60

        def goodput(rows, a, b):
            return sum(r.info["completed"] for r in rows[a:b]) / max(1, sum(r.info["arrivals"] for r in rows[a:b]))

        run = simulate(eng, d, ticks=act, world=world, keep=True)
        tally["healthy_before"] += goodput(run["rows"], at - 20, at) > 0.9
        collapsed = run["state"]
        # the cause is gone: the dependency is back at full rate ...
        tail = simulate(eng, d, ticks=end - act, world=world, state=collapsed, t0=act, keep=True)
        cap = [r.info["capacity"][dep] for r in tail["rows"]]
        before = [r.info["capacity"][dep] for r in run["rows"][at - 20:at]]
        assert np.mean(cap) > 0.9 * np.mean(before)
        # ... and yet nothing gets through, because almost every attempt is stale
        tally["collapsed_after"] += goodput(tail["rows"], 20, 50) < 0.2
        assert np.mean([r.info["failure_prob"][dep] for r in tail["rows"][20:]]) > 0.8
        more = [dict(d[0])]
        for lv in levers_of(cfg, "workers"):
            more[0][lv["name"]] = lv["hi"]
        cap_run = simulate(eng, more, ticks=end - act, world=world, state=collapsed, t0=act, keep=True)
        tally["capacity_fails"] += goodput(cap_run["rows"], 20, 50) < 0.2
        shed = with_lever(d, admit, 0.0)
        shed_run = simulate(eng, None, ticks=end - act, world=world, state=collapsed, t0=act, keep=True,
                            policy=lambda t: shed if t < act + 8 else d)
        tally["shed_recovers"] += goodput(shed_run["rows"], 20, 50) > 0.9
        calm = build_engine({"engine": "techsys", "config": {**cfg, "regimes": []}})
        calm_run = simulate(calm, d, ticks=end, world=world, keep=True)
        tally["no_trigger"] += goodput(calm_run["rows"], act + 20, end) > 0.9
    print(f"\n[cascade] over {N_SEEDS} seeds: healthy before the slowdown {tally['healthy_before']}, still collapsed "
          f"20-50 ticks after it ended {tally['collapsed_after']}, still collapsed with every worker lever at "
          f"maximum {tally['capacity_fails']}, recovered after 8 ticks of shedding then the same settings "
          f"{tally['shed_recovers']}, never collapsed without the slowdown {tally['no_trigger']}")
    assert all(v >= N_SEEDS - 1 for v in tally.values()), tally


def test_cascade_retry_limit_times_error_rate_decides_the_storm():
    """The retry lever is what arms the trap: with few retries the same slowdown passes; and the give-up rule
    reproduces the loss e^(R+1) and the load (1 - e^(R+1)) / (1 - e) of a retry limit R."""
    for e, lim in ((0.1, 0), (0.1, 3), (0.5, 2), (0.9, 5)):
        d = techsys.giveup_probability(e, lim)
        lost, load = e * d / (1 - e * (1 - d)), 1 / (1 - e * (1 - d))
        assert lost == pytest.approx(e ** (lim + 1), rel=1e-9)
        assert load == pytest.approx((1 - e ** (lim + 1)) / (1 - e), rel=1e-9)
    survive = collapse = 0
    for seed in range(N_SEEDS):
        m, eng = mech("tech_cascade", seed)
        cfg = m["config"]
        (retry,) = levers_of(cfg, "retry")
        (reg,) = cfg["regimes"]
        d = defaults(eng)
        assert retry["default"] >= 5
        after = slice(reg["until"] + 25, eng.horizon)
        few = simulate(eng, with_lever(d, retry, 0), world=1000 + seed, keep=True)
        many = simulate(eng, with_lever(d, retry, retry["hi"]), world=1000 + seed, keep=True)
        frac = lambda run: (sum(r.info["completed"] for r in run["rows"][after])  # noqa: E731
                            / max(1, sum(r.info["arrivals"] for r in run["rows"][after])))
        survive += frac(few) > 0.8
        collapse += frac(many) < 0.2
        # without retries the base failures are simply lost
        e = cfg["stages"][cfg["design"]["dependency"]]["err"]
        pre = few["rows"][20:reg["at"]]
        failed = sum(r.info["lines"][0]["failed"] for r in pre) / sum(r.info["arrivals"] for r in pre)
        assert failed == pytest.approx(e, abs=0.03)
    print(f"\n[cascade] retry limit 0: healthy after the slowdown in {survive}/{N_SEEDS} seeds; retry limit at "
          f"maximum: collapsed in {collapse}/{N_SEEDS}")
    assert survive >= N_SEEDS - 1 and collapse >= N_SEEDS - 1


# ---------------------------------------------------------------------- redundant paths
def test_redundant_paths_differ_in_reliability_and_the_better_split_flips_at_the_regime_change():
    flips = cheaper = 0
    for seed in range(N_SEEDS):
        m, eng = mech("tech_redundant", seed)
        cfg = m["config"]
        main, alt, split = cfg["design"]["main"], cfg["design"]["alt"], cfg["design"]["split"]
        at = cfg["regimes"][0]["at"]
        assert {r["target"] for r in cfg["regimes"]} == {main} and all(r["until"] is None for r in cfg["regimes"])
        assert {r["at"] for r in eng.truth()["regimes"]} == {at}
        (route,) = levers_of(cfg, "route")
        to_main = (lambda x: x) if cfg["stages"][split]["next"][0] == main else (lambda x: 1.0 - x)
        d = defaults(eng)
        assert to_main(route["default"]) >= 0.7
        # hidden reliabilities: measured failure rates per path, before and after
        run = simulate(eng, d, world=1000 + seed, keep=True)
        def rate(rows, i):
            return sum(r.info["failures"][i] for r in rows) / max(1, sum(r.info["attempts"][i] for r in rows))

        e_main, e_alt = rate(run["rows"][:at], main), rate(run["rows"][:at], alt)
        assert e_main == pytest.approx(cfg["stages"][main]["err"], abs=0.03) and e_main > e_alt + 0.02
        assert rate(run["rows"][at:], main) > e_main + 0.25
        assert rate(run["rows"][at:], alt) == pytest.approx(e_alt, abs=0.03)
        # the same staffing, traffic mostly on the cheap path against mostly on the dear one (common random numbers)
        staff = reference_settings(cfg, "capacity")
        for lv in levers_of(cfg, "workers"):
            if lv["targets"][0] in (main, alt):          # both paths able to carry everything, so only routing differs
                need = cfg["lines"][0]["arrival"]["base"] * 1.6 / cfg["stages"][lv["targets"][0]]["mu"]
                staff = with_lever(staff, lv, need)
        on_main = simulate(eng, with_lever(staff, route, to_main(0.9)), world=1000 + seed, keep=True)
        on_alt = simulate(eng, with_lever(staff, route, to_main(0.1)), world=1000 + seed, keep=True)
        done = lambda run, sl: sum(r.info["completed"] for r in run["rows"][sl])  # noqa: E731
        before, after = slice(12, at), slice(at + 6, eng.horizon)
        flips += (done(on_main, before) >= done(on_alt, before) - 0.01 * done(on_alt, before)
                  and done(on_main, after) < 0.95 * done(on_alt, after))
        # before the change the cheap path is the economical one at the resting staffing
        cheap = simulate(eng, d, ticks=at, world=1000 + seed)
        dear = simulate(eng, with_lever(d, route, to_main(0.1)), ticks=at, world=1000 + seed)
        cheaper += cheap["net"] > dear["net"]
    print(f"\n[redundant] with both paths fully staffed, routing 90% to the cheap path delivers as much before the "
          f"change and clearly less after it in {flips}/{N_SEEDS} seeds; before the change the resting split "
          f"out-earns the reverse split in {cheaper}/{N_SEEDS}")
    assert flips >= N_SEEDS - 2 and cheaper >= N_SEEDS - 2


# ---------------------------------------------------------------------- lost work
LOSTWORK_STYLES = ("capacity", "batching", "cache", "shedding", "routing")


def test_lostwork_admits_several_materially_different_solutions():
    """Each of five strategies moves one class of levers only, and each alone cuts the value lost to unreliability
    by at least 30% relative to the defaults while keeping the net result non-negative, in most seeds."""
    kinds = {"capacity": {"workers"}, "batching": {"batch"}, "cache": {"cache"}, "shedding": {"admit", "timeout"},
             "routing": {"route"}}
    cuts = {s: [] for s in LOSTWORK_STYLES + ("retry", "tuned")}
    nets = {s: [] for s in cuts}
    for seed in range(N_SEEDS):
        m, eng = mech("tech_lostwork", seed)
        cfg = m["config"]
        d = defaults(eng)[0]
        lost0 = outcome("tech_lostwork", seed, "default")[2]
        for style in cuts:
            s = reference_settings(cfg, style)[0]
            if style in kinds:
                changed = {lv["kind"] for lv in cfg["levers"] if s[lv["name"]] != d[lv["name"]]}
                assert changed and changed <= kinds[style], (style, changed)
            _, _, lost, net, _ = outcome("tech_lostwork", seed, style)
            cuts[style].append(1.0 - lost / lost0)
            nets[style].append(net)
    lost_default = np.mean([outcome("tech_lostwork", s, "default")[1] for s in range(N_SEEDS)])
    print(f"\n[lostwork] value lost at defaults: mean {lost_default:.2f} of arriving value; cut in lost value by "
          f"strategy over {N_SEEDS} seeds:")
    for style in cuts:
        c, n = np.array(cuts[style]), np.array(nets[style])
        ok = int(((c >= 0.3) & (n >= 0)).sum())
        print(f"   {style:9s} median cut {float(np.median(c)):+.2f}  min {c.min():+.2f}  "
              f"cut>=30% and net>=0 in {ok}/{N_SEEDS} seeds")
        if style in LOSTWORK_STYLES:
            assert ok >= N_SEEDS - 2, style
    # retries alone are not one of the solutions here: they add load to a path that is already short
    assert float(np.median(cuts["retry"])) < 0.1
    # the strategies are different operating points, not one point reached five ways
    m, eng = mech("tech_lostwork", 0)
    points = [json.dumps(reference_settings(m["config"], s), sort_keys=True) for s in LOSTWORK_STYLES]
    assert len(set(points)) == len(points)


def test_lostwork_score_is_value_delivered_while_the_operation_pays_its_way():
    for seed in range(4):
        m, eng = mech("tech_lostwork", seed)
        cfg = m["config"]
        assert cfg["objective"] == "ontime_value" and cfg["reserve"]["start"] > 0
        value = cfg["lines"][0]["value"]
        good = simulate(eng, reference_settings(cfg, "tuned"), world=50 + seed, keep=True)
        for r in good["rows"]:
            assert r.score == [value * r.info["completed"]] and r.info["reserve"] >= 0
        assert good["score"][0] == pytest.approx(good["value_done"]) and good["net"] > 0
        # a setting that delivers work at a loss stops scoring once the reserve is gone, and not before
        waste = [dict(reference_settings(cfg, "tuned")[0])]
        for lv in levers_of(cfg, "workers"):
            waste[0][lv["name"]] = lv["hi"]
        bad = simulate(eng, waste, world=50 + seed, keep=True)
        gone = [r.info["reserve"] < 0 for r in bad["rows"]]
        assert bad["net"] < 0 and any(gone) and not gone[0]
        first = gone.index(True)
        balance, bound = cfg["reserve"]["start"], cfg["reserve"]["cap"]
        for r in bad["rows"]:
            balance = max(-bound, min(bound, balance + r.revenue[0] - r.cost[0]))
            assert r.info["reserve"] == pytest.approx(balance, rel=1e-9, abs=1e-6)
            assert r.score[0] == (value * r.info["completed"] if balance >= 0 else 0.0)
        assert sum(r.info["completed"] for r in bad["rows"][first:]) > 0, "work was still being delivered"
        assert bad["score"][0] < 0.8 * bad["value_done"]
        # the constraint is stated to the seat in outcome terms only
        brief = eng.brief(0)
        assert "running balance" in brief and "reserve" in brief


# ---------------------------------------------------------------------- two seats
def scaled_workers(cfg: dict, settings: list[dict], seat: int, factor: float) -> list[dict]:
    out = settings
    for lv in levers_of(cfg, "workers"):
        if lv["seat"] == seat:
            out = with_lever(out, lv, max(1.0, round(settings[seat][lv["name"]] * factor)))
    return out


def test_handoff_value_needs_matched_capacity():
    """Cooperative pair. Matched halves beat every mismatch for the team (and both seats get the team score);
    over-provisioning a half only adds its cost, starving a half caps the whole line; a starved downstream half
    blocks the upstream half, which sees its own queues grow and its hand-off rate fall without having changed
    anything. The block is partial: work that has aged in the upstream queues expires soon after it is handed
    over, which keeps freeing room in the hand-off buffer."""
    wins = {"up x2": 0, "down x2": 0, "up x0.5": 0, "down x0.5": 0}
    means = {k: [] for k in ("matched", "default", *wins)}
    blocked = 0
    ratios, growth = [], []
    for seed in range(N_SEEDS):
        m, eng = mech("tech_handoff", seed)
        cfg = m["config"]
        world = 1000 + seed
        matched = reference_settings(cfg, "tuned")
        run_m = simulate(eng, matched, world=world, keep=True)
        assert all(r.score[0] == r.score[1] for r in run_m["rows"])
        means["matched"].append(run_m["net_frac"])
        means["default"].append(outcome("tech_handoff", seed, "default")[0])
        for name, (seat, f) in {"up x2": (0, 2.0), "down x2": (1, 2.0), "up x0.5": (0, 0.5), "down x0.5": (1, 0.5)}.items():
            run = simulate(eng, scaled_workers(cfg, matched, seat, f), world=world)
            means[name].append(run["net_frac"])
            wins[name] += run["net"] < run_m["net"]
            if f > 1:       # the extra capacity bought nothing: same deliveries, more cost
                assert run["completed"] <= run_m["completed"] * 1.02 and run["cost"][seat] > 1.5 * run_m["cost"][seat]
            else:           # the starved half caps what both are paid for
                assert run["completed"] < 0.95 * run_m["completed"]
        starved = simulate(eng, scaled_workers(cfg, matched, 1, 0.3), world=world, keep=True)
        handoff = lambda run: np.mean([r.info["throughput"][1] for r in run["rows"][12:]])  # noqa: E731
        upstream_q = lambda run: np.mean([r.info["queue"][0] + r.info["queue"][1] for r in run["rows"][12:]])  # noqa: E731
        # seat 0's own readings show it: its hand-off gauge and its queue gauges
        g_hand = next(g["name"] for g in cfg["gauges"] if g["kind"] == "handoff")
        assert np.mean([r.readings[0][g_hand] for r in starved["rows"][12:]]) == pytest.approx(handoff(starved))
        blocked += handoff(starved) < 0.95 * handoff(run_m) and upstream_q(starved) > 3 * upstream_q(run_m) + 20
        growth.append(upstream_q(starved) / max(1.0, upstream_q(run_m)))
        ratios.append(handoff(starved) / handoff(run_m))
        assert g_hand not in starved["rows"][0].readings[1]
    print(f"\n[handoff] team net / arriving value over {N_SEEDS} seeds: " + ", ".join(
        f"{k} {float(np.mean(v)):+.3f}" for k, v in means.items()) + f"; matched beats " + ", ".join(
        f"{k} in {v}/{N_SEEDS}" for k, v in wins.items()) + f"; with the downstream half at 0.3x the upstream "
        f"hand-off rate falls to {float(np.mean(ratios)):.2f} of matched and its own queues grow "
        f"{float(np.median(growth)):.0f}-fold (median); both in {blocked}/{N_SEEDS} seeds")
    assert all(v >= N_SEEDS - 1 for v in wins.values()) and blocked >= N_SEEDS - 1
    assert np.mean(means["matched"]) > np.mean(means["default"]) + 0.2


def test_shared_pool_is_a_commons():
    """Mixed motive pair. R: both keep the small cache; T: I take the large one alone; S: the other takes it
    alone; P: both take it. For each seat T > R (taking more pays), T > S (the one who takes does better than the
    one who refrains), and R > P (when both take, the pool thrashes and both do worse than under restraint)."""
    rows = []
    slow = {"RR": [], "GR": [], "GG": []}
    for seed in range(N_SEEDS):
        m, eng = mech("tech_shared_pool", seed)
        cfg = m["config"]
        world = 1000 + seed
        prof = {"R": reference_settings(cfg, "restraint"), "G": reference_settings(cfg, "grab")}
        cache = {lv["seat"]: lv for lv in levers_of(cfg, "cache")}
        for j in (0, 1):
            assert prof["G"][j][cache[j]["name"]] > 3 * prof["R"][j][cache[j]["name"]]
            assert {k for k in prof["R"][j] if prof["R"][j][k] != prof["G"][j][k]} <= {
                cache[j]["name"], *(lv["name"] for lv in levers_of(cfg, "admit"))}
        net = {}
        for a in "RG":
            for b in "RG":
                run = simulate(eng, [prof[a][0], prof[b][1]], world=world, keep=True)
                arrived = [sum(r.info["lines"][li]["arrivals"] for r in run["rows"]) * cfg["lines"][li]["value"]
                           for li in (0, 1)]
                net[a + b] = [(run["revenue"][j] - run["cost"][j]) / arrived[j] for j in (0, 1)]
                if a + b in slow:
                    slow[a + b].append(float(np.mean([r.info["slowdown"] for r in run["rows"]])))
        rows.append((net["RR"][0], net["GR"][0], net["RG"][0], net["GG"][0]))
        rows.append((net["RR"][1], net["RG"][1], net["GR"][1], net["GG"][1]))
    a = np.array(rows)
    R, T, S, P = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    n = len(rows)
    print(f"\n[shared pool] own net / own arriving value over {N_SEEDS} seeds x 2 seats: R {R.mean():+.3f}  T {T.mean():+.3f}  "
          f"S {S.mean():+.3f}  P {P.mean():+.3f};  T>R {int((T > R).sum())}/{n}  T>S {int((T > S).sum())}/{n}  "
          f"R>P {int((R > P).sum())}/{n}  S>P {int((S > P).sum())}/{n};  mean pool slowdown: both small "
          f"{np.mean(slow['RR']):.2f}, one large {np.mean(slow['GR']):.2f}, both large {np.mean(slow['GG']):.2f}")
    assert (T > R).sum() >= n - 2 and (T > S).sum() >= n - 2 and (R > P).sum() >= n - 2
    assert T.mean() > R.mean() + 0.08 and R.mean() > P.mean() + 0.15 and R.mean() > 0.05
    assert np.mean(slow["RR"]) > 0.99 and np.mean(slow["GG"]) < 0.75 < np.mean(slow["GR"]) < 0.99


# ====================================================================== performance
def test_single_seat_throughput():
    """Ticks per second of `step` on one core, single-seat families. Wall-clock on this shared machine swings
    by several times with what else is running, so the assertion uses processor time and the best of several
    short runs; the wall-clock figures (with and without deriving `tick_rng`) are printed."""
    lines = []
    for family in SINGLE_SEAT:
        m, eng = mech(family, 0)
        s = defaults(eng)
        g = np.random.default_rng(0)
        best_cpu = best_wall = best_rng = float("inf")
        n = 400
        for _ in range(8):
            st = eng.init_state(g)
            c0, w0 = time.process_time(), time.perf_counter()
            for t in range(n):
                st = eng.step(st, s, g, t).state
            best_cpu = min(best_cpu, (time.process_time() - c0) / n)
            best_wall = min(best_wall, (time.perf_counter() - w0) / n)
        for _ in range(3):
            st = eng.init_state(g)
            w0 = time.perf_counter()
            for t in range(n):
                st = eng.step(st, s, tick_rng(1, RIG, t), t).state
            best_rng = min(best_rng, (time.perf_counter() - w0) / n)
        lines.append(f"   {family:16s} {len(m['config']['stages'])} stages, {len(m['config']['levers'])} levers: "
                     f"{1 / best_cpu:7.0f} ticks/s (cpu)  {1 / best_wall:7.0f} ticks/s (wall)  "
                     f"{1 / best_rng:6.0f} ticks/s (wall, incl. tick_rng)")
        assert 1 / best_cpu >= 3000, f"{family}: {1 / best_cpu:.0f} ticks/s"
    print("\n[throughput] step on one core, best of 8 runs of 400 ticks:\n" + "\n".join(lines))
