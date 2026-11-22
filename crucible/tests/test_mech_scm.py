"""The `scm` mechanism engine: determinism and purity, every edge function, composed motifs and the family
partition (structural and compositional held-out families), opacity of the public surface, calibrated economics,
boundedness and conservation, the hidden confounder, the regime switch, and the accuracy of `truth()`."""
from __future__ import annotations

import copy
import functools
import json
import re

import numpy as np
import pytest

from crucible.core.canonical import digest
from crucible.mechanisms import base as B
from crucible.mechanisms import evaluator as E
from crucible.mechanisms import scm
from crucible.mechanisms.sim import MechSim

TRAIN = ["scm_additive", "scm_delay", "scm_gate"]
COMPOSITIONAL = {"scm_delay_gate": "val", "scm_stock_gate": "test"}
STRUCTURAL = {"scm_invu": "val", "scm_feedback": "test", "scm_regime": "test"}
ALL = TRAIN + list(COMPOSITIONAL) + list(STRUCTURAL)
N_SEEDS = 24


@functools.lru_cache(maxsize=None)
def mech(family: str, seed: int) -> dict:
    return B.make_mechanism(family, seed)


def engine(family: str, seed: int) -> scm.ScmEngine:
    return B.build_engine(mech(family, seed))


def steady_out(m: dict, settings: dict, regime: int = 0) -> float:
    eng = B.build_engine(m)
    return eng.steady([dict(eng.defaults(0), **settings)], regime)[eng._idx["out_0"]]


def run(m: dict, schedule: dict[int, dict], ticks: int, seed: int = 0):
    """Readings and scores of a hand-made mechanism under a settings schedule {tick: partial settings}."""
    sim = MechSim(m, seed)
    out = []
    for t in range(ticks):
        r = sim.step(schedule.get(t))
        out.append((r.readings[0], r.score[0], r.revenue[0], r.cost[0]))
    return out


def one(term: dict, **kw) -> dict:
    """lever a (and b) -> latent x -> outcome, without noise."""
    return scm.handmade(levers=[{"name": "a"}, {"name": "b"}],
                        latents=[dict({"id": "x", "terms": [dict(term, src=term.get("src", "a"))]}, **kw)],
                        outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "cap": 50.0})


# ====================================================================== registration and determinism
def test_families_are_registered_with_the_required_partition():
    for f in TRAIN:
        fam = B.MECH_FAMILIES[f]
        assert (fam.engine, fam.split, fam.transfer) == ("scm", "train", "base")
    for f, split in COMPOSITIONAL.items():
        assert (B.MECH_FAMILIES[f].split, B.MECH_FAMILIES[f].transfer) == (split, "compositional")
    for f, split in STRUCTURAL.items():
        assert (B.MECH_FAMILIES[f].split, B.MECH_FAMILIES[f].transfer) == (split, "structural")
    assert set(B.mech_families(engine="scm")) == set(ALL)
    assert B.ENGINES["scm"] is scm.ScmEngine


def test_generation_is_deterministic_json_and_seed_dependent():
    for f in ALL:
        a, b = B.make_mechanism(f, 3), B.make_mechanism(f, 3)
        assert digest(a) == digest(b)
        assert a["config"]["calibrated"] and a["config"]["family"] == f and a["engine"] == "scm"
        assert json.loads(json.dumps(a)) == a                     # plain JSON, nothing lost
        assert digest(B.make_mechanism(f, 4)) != digest(a)


def test_a_config_restored_from_json_reproduces_the_trajectory_bit_for_bit():
    for f in ("scm_delay", "scm_feedback", "scm_regime"):
        m = mech(f, 1)
        m2 = json.loads(json.dumps(m))
        s1, s2 = MechSim(m, 9), MechSim(m2, 9)
        rng = np.random.default_rng(0)
        lv = s1.engine.levers(0)
        for t in range(60):
            x = {l.name: float(rng.uniform(l.lo, l.hi)) for l in lv} if t % 4 == 0 else None
            r1, r2 = s1.step(x), s2.step(x)
            assert r1.readings == r2.readings and r1.revenue == r2.revenue and r1.cost == r2.cost
            assert r1.state["h"].tobytes() == r2.state["h"].tobytes()


def test_step_is_pure_and_never_mutates_its_input():
    for f in ALL:
        eng = engine(f, 2)
        st = eng.init_state(B.tick_rng(5, "r1", "init"))
        x = [eng.defaults(0)]
        frozen = {k: v.copy() for k, v in st.items()}
        for v in st.values():
            v.flags.writeable = False                              # a step that wrote into its input would raise
        a = eng.step(st, x, B.tick_rng(5, "r1", 0), 0)
        b = eng.step(st, x, B.tick_rng(5, "r1", 0), 0)
        assert all(np.array_equal(st[k], frozen[k]) for k in st)
        assert a.state["h"].tobytes() == b.state["h"].tobytes() and a.readings == b.readings
        assert a.revenue == b.revenue and a.cost == b.cost and a.score == b.score
        c = eng.step(st, x, B.tick_rng(5, "r1", 1), 0)             # other random numbers: another outcome
        assert c.readings != a.readings


# ====================================================================== edge functions (noise-free, hand-made)
def test_linear_and_saturating_edges():
    lin = one({"kind": "linear", "w": 2.0})
    assert [steady_out(lin, {"a": u}) for u in (0.0, 0.25, 1.0)] == pytest.approx([0.0, 0.5, 2.0])
    sat = one({"kind": "saturating", "w": 2.0, "k": 4.0})
    y = [steady_out(sat, {"a": u}) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert y[0] == 0.0 and y[-1] == pytest.approx(2.0)
    gains = np.diff(y)
    assert (gains > 0).all() and (np.diff(gains) < 0).all()        # monotone with diminishing returns
    assert y[1] == pytest.approx(2.0 * (1 - np.exp(-1.0)) / (1 - np.exp(-4.0)))


def test_threshold_edge_is_a_step_at_the_hidden_level():
    m = one({"kind": "threshold", "w": 1.5, "theta": 0.6})
    assert [steady_out(m, {"a": u}) for u in (0.0, 0.59, 0.61, 1.0)] == pytest.approx([0.0, 0.0, 1.5, 1.5])


def test_inverted_u_edge_peaks_inside_the_range_and_is_not_monotone():
    m = one({"kind": "invu", "w": 1.0, "peak": 0.4, "width": 0.4}, bias=1.0)
    y = {u: steady_out(m, {"a": u}) for u in (0.0, 0.2, 0.4, 0.6, 1.0)}
    assert y[0.4] == pytest.approx(2.0) and y[0.4] > y[0.2] > y[0.0] and y[0.4] > y[0.6] > y[1.0]
    assert y[0.2] == pytest.approx(y[0.6])                          # symmetric about the peak
    assert B.build_engine(m).truth()["edges"][0]["sign"] == 0      # declared non-monotone


def test_product_edge_needs_both_sources():
    m = one({"kind": "product", "src2": "b", "w": 3.0})
    assert steady_out(m, {"a": 1.0, "b": 0.0}) == 0.0 and steady_out(m, {"a": 0.0, "b": 1.0}) == 0.0
    assert steady_out(m, {"a": 0.5, "b": 0.5}) == pytest.approx(0.75)
    assert steady_out(m, {"a": 1.0, "b": 1.0}) == pytest.approx(3.0)


def test_gated_edge_is_active_only_while_the_third_node_is_above_its_level():
    m = one({"kind": "linear", "w": 2.0, "gate": {"node": "b", "level": 0.7}})
    assert steady_out(m, {"a": 1.0, "b": 0.69}) == 0.0
    assert steady_out(m, {"a": 1.0, "b": 0.71}) == pytest.approx(2.0)
    assert steady_out(m, {"a": 0.0, "b": 1.0}) == 0.0
    edges = B.build_engine(m).truth()["edges"]
    gated = next(e for e in edges if e["kind"] == "gate")
    assert gated["sign"] == 0 and gated["open_sign"] == 1 and gated["gate"] == "lev_01"
    assert any(e["kind"] == "gate_control" and e["src"] == "lev_01" for e in edges)


def test_pure_delay_shows_exactly_k_ticks_later():
    m = one({"kind": "delay", "w": 1.0, "lag": 4})
    rows = run(m, {5: {"a": 1.0}}, 14)                              # the lever moves at tick 5
    x = [r[0]["g_x"] for r in rows]
    assert x[:9] == [0.0] * 9 and x[9:] == [1.0] * 5                # first visible at tick 5 + 4
    assert [r[1] for r in rows][8:10] == [0.0, 100.0]               # and in the output sold at the same tick


def test_stock_accumulates_and_decays_geometrically():
    d = 0.25
    m = one({"kind": "linear", "w": 2.0}, decay=d)
    rows = run(m, {0: {"a": 1.0}, 12: {"a": 0.0}}, 20)
    x = [r[0]["g_x"] for r in rows]
    assert x[:12] == pytest.approx([2.0 * (1 - (1 - d) ** (n + 1)) for n in range(12)], abs=1e-4)
    assert x[12:] == pytest.approx([x[11] * (1 - d) ** (n + 1) for n in range(8)], abs=1e-4)
    assert steady_out(m, {"a": 1.0}) == pytest.approx(2.0)          # the level it relaxes to


def test_feedback_loop_through_a_lag_converges_to_the_fixed_point():
    phi = -0.5
    m = scm.handmade(levers=[{"name": "a"}],
                     latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 1.0},
                                                    {"kind": "linear", "src": "y", "w": phi, "lag": 1}]},
                              {"id": "y", "terms": [{"kind": "linear", "src": "x", "w": 1.0}]}],
                     outcome={"terms": [{"kind": "linear", "src": "y", "w": 1.0}]})
    assert "feedback" in m["config"]["motifs"]
    rows = run(m, {0: {"a": 1.0}}, 30)
    y = [r[0]["g_y"] for r in rows]
    assert y[0] == 1.0 and y[1] == pytest.approx(0.5) and y[2] == pytest.approx(0.75)   # it rings
    assert y[-1] == pytest.approx(1.0 / (1.0 - phi), abs=1e-3)
    assert steady_out(m, {"a": 1.0}) == pytest.approx(1.0 / (1.0 - phi), abs=1e-6)


def test_regime_switch_changes_a_weight_at_the_hidden_tick():
    m = scm.handmade(levers=[{"name": "a"}], latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 1.0}]}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0, "w2": -1.0}], "bias": 1.0},
                     regime_at=7)
    rows = run(m, {0: {"a": 0.5}}, 12)
    assert [r[1] for r in rows] == [150.0] * 7 + [50.0] * 5         # a rule that was true stops being true
    truth = B.build_engine(m).truth()
    assert truth["regimes"][0]["at"] == 7 and truth["regimes"][0]["changes"][0]["w_after"] == -1.0
    assert "regime" in truth["motifs"]


def test_economics_formula_and_integer_levers():
    m = scm.handmade(levers=[{"name": "a", "lo": 0, "hi": 4, "integer": True, "cost": 80.0}],
                     latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 1.0}]}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "bias": 0.2}, price=3.0, units=50.0,
                     overhead=40.0)
    sim = MechSim(m, 0)
    r = sim.step({"a": 2.6})                                        # rounds to 3 -> u = 0.75
    assert sim.settings[0]["a"] == 3.0
    assert r.score[0] == pytest.approx(50.0 * (0.2 + 0.75))
    assert r.revenue[0] == int(3.0 * 50.0 * 0.95 + 0.5) and r.cost[0] == 40.0 + 80.0 * 0.75   # half cents round up
    r = sim.step({"a": -9})                                         # clipped to the minimum
    assert r.cost[0] == 40.0 and r.revenue[0] == round(3.0 * 50.0 * 0.2)


# ====================================================================== motifs and the family partition
@functools.lru_cache(maxsize=None)
def motif_draws(family: str) -> tuple:
    return tuple(frozenset(B.build_engine(mech(family, s)).truth()["motifs"]) for s in range(N_SEEDS))


def test_truth_motifs_stay_inside_what_the_family_declares():
    for f in ALL:
        declared = set(B.MECH_FAMILIES[f].motifs)
        for s, ms in enumerate(motif_draws(f)):
            assert ms <= declared, (f, s, sorted(ms - declared))


def test_structural_families_contain_a_motif_no_training_draw_contains():
    train = set().union(*[ms for f in TRAIN for ms in motif_draws(f)])
    novel = {"scm_invu": {"invu"}, "scm_feedback": {"feedback", "confounder"}, "scm_regime": {"regime"}}
    for f, must in novel.items():
        assert not (must & train), f"{must & train} appears in a training draw"
        for s, ms in enumerate(motif_draws(f)):
            assert must <= ms, (f, s, sorted(ms))
            assert ms - train, (f, s)                               # at least one motif unseen in training


def test_compositional_families_use_only_training_motifs_in_an_unseen_cooccurrence():
    train_draws = [ms for f in TRAIN for ms in motif_draws(f)]
    train = set().union(*train_draws)
    pair = {"scm_delay_gate": ("delay", "gate"), "scm_stock_gate": ("stock", "gate")}
    for f, (a, b) in pair.items():
        assert a in train and b in train
        assert not any(a in ms and b in ms for ms in train_draws), f"{a}+{b} co-occur in training"
        for s, ms in enumerate(motif_draws(f)):
            assert ms <= train, (f, s, sorted(ms - train))          # nothing structurally new
            assert a in ms and b in ms, (f, s, sorted(ms))          # ... in a combination training never shows
    # the two held-out combinations are different from each other too
    assert not any("stock" in ms for ms in motif_draws("scm_delay_gate"))
    assert not any("delay" in ms for ms in motif_draws("scm_stock_gate"))


def test_training_families_differ_from_each_other_in_structure():
    dyn, cond = {"delay", "stock", "lagged_gauge"}, {"gate", "threshold", "interaction"}
    for ms in motif_draws("scm_additive"):
        assert not ms & (dyn | cond)
    for ms in motif_draws("scm_delay"):
        assert {"delay", "stock"} <= ms and not ms & cond
    for ms in motif_draws("scm_gate"):
        assert "gate" in ms and ms & {"threshold", "interaction"} and not ms & dyn
    # every common motif is really exercised somewhere in training
    train = set().union(*[ms for f in TRAIN for ms in motif_draws(f)])
    assert {"chain", "fork", "collider", "mediator", "saturation", "distractor", "costly"} <= train


def _shape(cfg: dict) -> str:
    """The wiring of a mechanism with every coefficient and name removed."""
    idx = {n["id"]: i for i, n in enumerate(cfg["nodes"])}
    rows = []
    for n in cfg["nodes"]:
        ts = sorted((t["kind"], idx[t["src"]], t.get("lag", 0), "gate" in t, "w2" in t, t["w"] > 0)
                    for t in n.get("terms", []))
        rows.append((n["role"], bool(n.get("decay")), ts))
    return digest([rows, sorted((idx[g["src"]], g["lag"]) for g in cfg["gauges"])])


def test_seeds_draw_different_graphs_not_only_different_coefficients():
    for f in ALL:
        cfgs = [mech(f, s)["config"] for s in range(N_SEEDS)]
        assert len({_shape(c) for c in cfgs}) >= N_SEEDS - 2, f     # (almost) every seed is a different graph
        assert len({sum(n["role"] == "lever" for n in c["nodes"]) for c in cfgs}) >= 3
        assert len({len(c["nodes"]) for c in cfgs}) >= 4
        kinds = [tuple(sorted(t["kind"] for n in c["nodes"] for t in n.get("terms", []))) for c in cfgs]
        assert len(set(kinds)) >= N_SEEDS // 3                      # how many edges of which kind varies too


# ====================================================================== opacity
def test_public_names_are_neutral_and_say_nothing_about_role():
    pat = re.compile(r"[bdfgjklmnprstvz][aeiou]\d\d")
    pos_distractor, pos_costly, first_is_gauge = [], [], []
    for f in ALL:
        for s in range(N_SEEDS):
            cfg = mech(f, s)["config"]
            eng = engine(f, s)
            spec = eng.public_spec(0)
            names = [x["name"] for x in spec["levers"]] + [x["name"] for x in spec["gauges"]]
            assert len(set(names)) == len(names) and all(pat.fullmatch(n) for n in names)
            assert all(x["doc"] == "" for x in spec["levers"] + spec["gauges"])
            text = json.dumps(spec)
            assert not re.search(r"lev_\d|lat_\d|exo_\d|gau_\d|out_\d", text)
            by = {n["public"]: n for n in cfg["nodes"] if n["role"] == "lever"}
            kids = {t["src"] for n in cfg["nodes"] for t in n.get("terms", [])} | \
                   {t["src2"] for n in cfg["nodes"] for t in n.get("terms", []) if "src2" in t} | \
                   {t["gate"]["node"] for n in cfg["nodes"] for t in n.get("terms", []) if "gate" in t}
            order = [x["name"] for x in spec["levers"]]
            for i, n in enumerate(order):
                rel = i / (len(order) - 1)
                if by[n]["id"] not in kids:
                    pos_distractor.append(rel)
                if by[n]["cost"] > 0:
                    pos_costly.append(rel)
            first_is_gauge.append(sorted(names)[0] in {x["name"] for x in spec["gauges"]})
            assert "brief" in spec and not re.search(r"\d", spec["brief"])    # no numbers, no structure
    # position in the list is uninformative: distractors and costly levers sit anywhere
    assert 0.4 < np.mean(pos_distractor) < 0.6 and 0.4 < np.mean(pos_costly) < 0.6
    assert min(pos_distractor) == 0.0 and max(pos_distractor) == 1.0
    assert 0.25 < np.mean(first_is_gauge) < 0.75                    # levers and gauges share one name pool


def test_defaults_ranges_and_settings_clip():
    eng = engine("scm_gate", 0)
    for lv in eng.levers(0):
        assert lv.lo <= lv.default <= lv.hi
        assert lv.clip(lv.hi + 5) == lv.hi and lv.clip(lv.lo - 5) == lv.lo and lv.clip("x") == lv.default
        if lv.integer:
            assert lv.default == round(lv.default) and lv.clip(lv.lo + 0.4) == lv.lo
    full = eng.clip_settings(0, {eng.levers(0)[0].name: 1e9, "nope": 3})
    assert set(full) == {lv.name for lv in eng.levers(0)} and full[eng.levers(0)[0].name] == eng.levers(0)[0].hi


# ====================================================================== economics
def test_calibration_holds_on_the_noise_free_steady_state():
    for f in ALL:
        with_free = 0
        for s in range(10):
            cfg = mech(f, s)["config"]
            assert cfg["calibrated"] and cfg["attempts"] <= scm.MAX_ATTEMPTS
            sd, cal = cfg["seats"][0], cfg["seats"][0]["calib"]
            r0 = cal["revenue_default"]
            assert 1500 <= r0 <= 6000 and sd["price"] > 0 and sd["overhead"] >= 0
            assert cal["net_opt"] >= scm.G_MIN * r0                 # good settings: clearly positive
            assert cal["net_random"] <= -scm.R_MIN * r0             # random settings: a loss on average
            assert cal["net_max"] <= scm.CORNER * cal["net_opt"]    # neither corner is near the best
            assert cal["net_min"] <= scm.CORNER * cal["net_opt"]
            levers = [n for n in cfg["nodes"] if n["role"] == "lever"]
            assert all(n["cost"] >= 0 for n in levers) and any(n["cost"] > 0 for n in levers)
            with_free += any(n["cost"] == 0 for n in levers)
        assert with_free >= 5, f                                    # costly and free levers both occur


def test_default_settings_earn_about_nothing_and_the_constructive_optimum_earns_clearly_more():
    for f in ("scm_additive", "scm_delay", "scm_gate", "scm_invu", "scm_feedback"):
        m = mech(f, 5)
        eng = B.build_engine(m)
        r0 = m["config"]["seats"][0]["calib"]["revenue_default"]
        seeds = list(range(700, 724))
        d, _ = E.mean_net(m, [{"from": 0, "settings": eng.defaults(0)}], seeds, eng.horizon, engine=eng)
        o, _ = E.mean_net(m, eng.constructive_optimum(0)["schedule"], seeds, eng.horizon, engine=eng)
        assert abs(d) < 0.05 * r0, (f, d / r0)
        assert o > 0.15 * r0 and o - d > 0.15 * r0, (f, o / r0)


def test_bounded_state_and_conservation_under_random_and_extreme_settings():
    for f in ALL:
        m = mech(f, 7)
        eng = B.build_engine(m)
        income = m["config"]["income_cents_per_tick"]
        lv = eng.levers(0)
        rng = np.random.default_rng(1)
        hi, lo = {l.name: l.hi for l in lv}, {l.name: l.lo for l in lv}
        for mode in ("random", "hi", "lo", "flip"):
            sim = MechSim(m, 3)
            for t in range(2 * eng.horizon):
                x = {"random": {l.name: float(rng.uniform(l.lo, l.hi)) for l in lv}, "hi": hi, "lo": lo,
                     "flip": hi if t % 2 else lo}[mode]
                r = sim.step(x)
                h = r.state["h"]
                assert np.isfinite(h).all() and np.abs(h).max() <= scm.CLIP
                assert 0 <= r.revenue[0] <= income and r.cost[0] >= 0 and r.score[0] >= 0
                assert all(np.isfinite(v) for v in r.readings[0].values())


# ====================================================================== the causal claims
def test_truth_matches_measured_effects_in_every_family():
    """Wherever the declared graph implies an unambiguous sign, the simulated intervention never shows the
    opposite sign and never responds earlier than the shortest declared path; where the graph has no path, the
    effect is exactly zero."""
    seen = agree = 0
    for f in ALL:
        m = mech(f, 2)
        eng = B.build_engine(m)
        em = E.effect_matrix(m, n_base=10, lags=(0, 1, 2, 4, 8), engine=eng)
        implied = E.implied_signs(eng.truth())
        for (lever, target), (sign, lag) in implied.items():
            i, j = em["levers"].index(lever), em["targets"].index(target)
            got, first = int(em["sign"][i, j]), int(em["first_lag"][i, j])
            if sign == 0:
                assert np.all(em["response"][i, j] == 0.0), (f, lever, target)      # no path: exactly nothing
            elif sign is not None:
                seen += 1
                assert got in (0, sign), (f, lever, target, got, sign)
                agree += got == sign
                if got != 0:
                    assert first >= lag, (f, lever, target, first, lag)
    assert seen >= 40 and agree / seen >= 0.8, (seen, agree)


def test_a_delay_is_measured_at_its_declared_lag():
    hits = 0
    for s in range(6):
        m = mech("scm_delay", s)
        eng = B.build_engine(m)
        truth = eng.truth()
        em = E.effect_matrix(m, n_base=10, lags=(0, 1, 2, 3, 4, 5, 6, 8), engine=eng)
        pub = {n["id"]: n["public"] for n in truth["nodes"]}
        gauge_of = {e["src"]: (pub[e["dst"]], e["lag"]) for e in truth["edges"] if e["kind"] == "observe"}
        for e in truth["edges"]:
            if e["kind"] == "delay" and e["dst"] in gauge_of and pub.get(e["src"]) in em["levers"]:
                g, glag = gauge_of[e["dst"]]
                i, j = em["levers"].index(pub[e["src"]]), em["targets"].index(g)
                if len([x for x in truth["edges"] if x["dst"] == e["dst"]]) == 1:
                    assert em["first_lag"][i, j] == e["lag"] + glag, (s, e, glag)
                    hits += 1
    assert hits >= 3


def test_hidden_confounder_passive_correlation_misleads_and_intervention_reveals_it():
    checked = 0
    for s in range(8):
        m = mech("scm_feedback", s)
        eng = B.build_engine(m)
        truth = eng.truth()
        role = {n["id"]: n["role"] for n in truth["nodes"]}
        pub = {n["id"]: n["public"] for n in truth["nodes"]}
        kids: dict[str, list] = {}
        for e in truth["edges"]:
            kids.setdefault(e["src"], []).append(e)
        for u in (n["id"] for n in truth["nodes"] if n["role"] == "exogenous"):
            to_out = [e for e in kids.get(u, []) if role[e["dst"]] == "outcome"]
            to_lat = [e for e in kids.get(u, []) if role[e["dst"]] == "latent"]
            if not to_out or not to_lat:
                continue
            g_lat = to_lat[0]["dst"]
            obs = next(e for e in truth["edges"] if e["src"] == g_lat and e["kind"] == "observe")
            lever = next(e["src"] for e in truth["edges"] if e["dst"] == g_lat and role[e["src"]] == "lever")
            assert not [e for e in kids.get(g_lat, []) if e["kind"] != "observe"]    # the gauge causes nothing
            # passive data: the gauge and the revenue move together (through the hidden driver)
            sim = MechSim(m, 100 + s)
            rows = [sim.step(None) for _ in range(600)]
            g = np.array([r.readings[0][pub[obs["dst"]]] for r in rows[obs["lag"]:]])
            rev = np.array([r.revenue[0] for r in rows[:len(rows) - obs["lag"]]])
            corr = float(np.corrcoef(g, rev)[0, 1])
            expected = np.sign(to_lat[0]["w"] * obs["sign"] * to_out[0]["w"])
            assert corr * expected > 0.25, (s, corr)
            # intervention: the lever moves the gauge and leaves the revenue exactly where it was
            em = E.effect_matrix(m, n_base=8, engine=eng)
            i = em["levers"].index(pub[lever])
            assert em["sign"][i, em["targets"].index(pub[obs["dst"]])] != 0
            assert np.all(em["response"][i, em["targets"].index("revenue")] == 0.0)
            assert np.all(em["response"][i, em["targets"].index("score")] == 0.0)
            checked += 1
    assert checked >= 6


def test_regime_family_the_best_settings_change_at_the_switch():
    for s in range(4):
        m = mech("scm_regime", s)
        eng = B.build_engine(m)
        cut = eng.truth()["regimes"][0]["at"]
        assert 0.35 * eng.horizon <= cut <= 0.65 * eng.horizon
        sched = eng.constructive_optimum(0)["schedule"]
        before, after = sched[0]["settings"], sched[1]["settings"]
        assert before != after
        seeds, h = list(range(300, 308)), eng.horizon

        def net(settings, window):
            return E.mean_net(m, [{"from": 0, "settings": settings}], seeds, h, window=window, engine=eng)[0]
        assert net(before, (0, cut)) > net(after, (0, cut))         # right before the switch ...
        assert net(after, (cut, h)) > net(before, (cut, h))         # ... wrong after it
        # and a measured effect really changes: some lever's effect on the output differs between regimes
        e1 = E.effect_matrix(m, n_base=8, t0_range=(8, 16), engine=eng)
        e2 = E.effect_matrix(m, n_base=8, t0_range=(cut + 8, cut + 16), engine=eng)
        j = e1["targets"].index("score")
        assert np.abs(e1["effect"][:, j, -1] - e2["effect"][:, j, -1]).max() > 1.0


def test_two_seat_mechanisms_have_separate_levers_and_coupled_outcomes():
    m = B.make_mechanism("scm_additive", 1, {"seats": 2})
    eng = B.build_engine(m)
    assert eng.n_seats == 2 and "coupling" in eng.truth()["motifs"]
    a, b = {lv.name for lv in eng.levers(0)}, {lv.name for lv in eng.levers(1)}
    assert a and b and not a & b
    assert not {g.name for g in eng.gauges(0)} & {g.name for g in eng.gauges(1)}
    # what one seat does reaches the other's output one tick later (and never in the same tick)
    truth = eng.truth()
    seat_of = {n["id"]: n["seat"] for n in truth["nodes"]}
    cross = [e for e in truth["edges"] if seat_of[e["src"]] is not None and seat_of[e["dst"]] is not None
             and seat_of[e["src"]] != seat_of[e["dst"]]]
    assert cross and all(e["lag"] == 1 for e in cross)
    src, dst = seat_of[cross[0]["src"]], seat_of[cross[0]["dst"]]
    far = {lv.name: (lv.lo if lv.default > (lv.lo + lv.hi) / 2 else lv.hi) for lv in eng.levers(src)}
    base, moved = MechSim(m, 4), MechSim(m, 4)
    d = []
    for t in range(6):
        r1, r2 = base.step(None), moved.step([far if k == src else None for k in range(2)] if t == 2 else None)
        d.append(r2.score[dst] - r1.score[dst])
    assert d[:3] == [0.0, 0.0, 0.0] and d[3] != 0.0
    assert copy.deepcopy(m) == m
