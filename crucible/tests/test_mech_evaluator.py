"""Private evaluation and validity of mechanisms: effect matrices against hand-made and generated ground truth,
scoring of hypotheses, the reference policy and regret, information efficiency, the operator-side pipeline for
claims filed in a world, and the validity checker (what it accepts, what it rejects and why, solvability)."""
from __future__ import annotations

import functools
import math
import subprocess
import sys

import numpy as np
import pytest

import crucible.services  # noqa: F401
from crucible.core.kernel import World, WorldSpec
from crucible.mechanisms import base as B
from crucible.mechanisms import evaluator as E
from crucible.mechanisms import module as M
from crucible.mechanisms import scm
from crucible.mechanisms import validity as V
from crucible.mechanisms.sim import MechSim

FAST = {"n_seeds": 4, "budget": 64, "n_random": 96}


@functools.lru_cache(maxsize=None)
def mech(family: str, seed: int) -> dict:
    return B.make_mechanism(family, seed)


def chain(w1=1.0, w2=0.8, w3=0.5, lag=3, noise=0.01) -> dict:
    """a -> x -> y -> outcome; b -> z (after `lag` ticks) -> outcome; d does nothing; a costs money."""
    return scm.handmade(
        levers=[{"name": "a", "cost": 50.0}, {"name": "b"}, {"name": "d"}],
        latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": w1}], "noise": noise, "bias": 1.0},
                 {"id": "y", "terms": [{"kind": "saturating", "src": "x", "w": w2, "k": 1.5}], "bias": 1.0, "noise": noise},
                 {"id": "z", "terms": [{"kind": "delay", "src": "b", "w": 0.5, "lag": lag}], "bias": 1.0, "noise": noise}],
        outcome={"terms": [{"kind": "linear", "src": "y", "w": w3}, {"kind": "linear", "src": "z", "w": 0.4}],
                 "bias": 2.0, "noise": noise},
        gauges=[{"name": "gx", "src": "x"}, {"name": "gy", "src": "y", "a": -2.0, "c": 5.0},
                {"name": "gz", "src": "z", "lag": 1}])


# ====================================================================== effects
@pytest.mark.parametrize("w1,w2,w3", [(1.0, 0.8, 0.5), (-1.0, 0.8, 0.5), (1.0, -0.8, 0.5), (-1.0, -0.8, -0.5)])
def test_effect_matrix_equals_the_signs_and_lags_implied_by_truth_on_chains(w1, w2, w3):
    m = chain(w1, w2, w3)
    eng = B.build_engine(m)
    em = E.effect_matrix(m, n_base=8, lags=(0, 1, 2, 4, 8), engine=eng)
    implied = E.implied_signs(eng.truth())
    assert len(implied) == 3 * 4 and all(s is not None for s, _ in implied.values())
    for (lever, target), (sign, lag) in implied.items():
        i, j = em["levers"].index(lever), em["targets"].index(target)
        assert em["sign"][i, j] == sign, (lever, target)
        assert em["first_lag"][i, j] == (lag if sign else -1), (lever, target)
    a, b, d = (em["levers"].index(x) for x in "abd")
    col = {t: j for j, t in enumerate(em["targets"])}
    s_a = int(np.sign(w1 * w2 * w3))
    assert em["sign"][a, col["gx"]] == np.sign(w1) and em["sign"][a, col["gy"]] == -np.sign(w1 * w2)   # gy reads -2y
    assert em["sign"][a, col["score"]] == em["sign"][a, col["revenue"]] == s_a
    assert em["sign"][a, col["cost"]] == 1 and em["first_lag"][a, col["cost"]] == 0     # the costly lever
    assert em["sign"][b, col["cost"]] == 0 and (em["sign"][d] == 0).all()               # free lever, distractor
    assert em["first_lag"][b, col["gz"]] == 4 and em["first_lag"][b, col["score"]] == 3  # delay 3 (+1 gauge lag)
    assert em["effect"].shape == em["se"].shape == em["significant"].shape == (3, 7, 5)
    assert (em["response"][d] == 0).all()                           # common random numbers: exactly nothing


def test_effect_sizes_are_right_and_carry_standard_errors():
    # x = 2 a b: the effect of a depends on where b happens to be, so it has a spread across base conditions
    m = scm.handmade(levers=[{"name": "a"}, {"name": "b"}],
                     latents=[{"id": "x", "terms": [{"kind": "product", "src": "a", "src2": "b", "w": 2.0}]}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}]})
    em = E.effect_matrix(m, n_base=64, lags=(0, 2), seed=3)
    i, j = em["levers"].index("a"), em["targets"].index("g_x")
    eff, se = em["effect"][i, j, 0], em["se"][i, j, 0]
    assert se > 0 and abs(eff - 2.0 * 0.5 * 0.5) < 3 * se           # E[2 * (0.75 - 0.25) * b] = 0.5
    assert em["significant"][i, j].all() and em["sign"][i, j] == 1
    k = em["targets"].index("score")
    assert em["effect"][i, k, 0] == pytest.approx(100.0 * eff, rel=1e-3)                # 100 units per unit of x
    assert E.effect_rows(em)[0].keys() >= {"lever", "target", "sign", "first_lag", "effect", "se"}


# ====================================================================== claims
def test_true_claims_score_one_and_random_claims_score_at_chance():
    rng = np.random.default_rng(0)
    acc_random = []
    for fam, seed in (("scm_additive", 1), ("scm_delay", 2), ("scm_gate", 3)):
        m = mech(fam, seed)
        eng = B.build_engine(m)
        em = E.effect_matrix(m, n_base=10, engine=eng)
        truth = E.true_claims(em)
        for c in truth["effects"]:
            c["confidence"] = 1.0
        s = E.score_claims(m, truth, effects=em, engine=eng)
        assert s["sign_accuracy"] == 1.0 and s["lag_accuracy"] == 1.0 and s["lag_mae"] == 0.0
        assert s["coverage"] == 1.0 and s["identification"] == 1.0 and s["brier"] == 0.0 and s["invalid"] == 0
        assert s["n_claims"] == s["n_pairs"] == len(em["levers"]) * len(em["targets"]) and s["n_true_effects"] > 5
        for _ in range(12):
            guess = {"effects": [{"lever": lv, "target": tg, "sign": int(rng.integers(-1, 2)), "confidence": 0.5}
                                 for lv in em["levers"] for tg in em["targets"]]}
            r = E.score_claims(m, guess, effects=em, engine=eng)
            acc_random.append(r["sign_accuracy"])
            assert r["brier"] == pytest.approx(0.25)
        # stating the opposite of every true effect is worse than chance on the pairs that have an effect
        flipped = {"effects": [dict(c, sign=-c["sign"]) for c in truth["effects"] if c["sign"] != 0]}
        assert E.score_claims(m, flipped, effects=em, engine=eng)["sign_accuracy"] == 0.0
    assert abs(float(np.mean(acc_random)) - 1 / 3) < 0.04           # three possible signs: chance is one third


def test_claim_scoring_details():
    m = chain()
    em = E.effect_matrix(m, n_base=8)
    claims = {"effects": [{"lever": "a", "target": "gx", "sign": -1},            # wrong, then corrected below
                          {"lever": "a", "target": "gx", "sign": 1, "confidence": 0.9},
                          {"lever": "b", "target": "score", "sign": 1, "lag": 5, "confidence": 0.6},   # lag is 3
                          {"lever": "d", "target": "net", "sign": 1, "confidence": 0.7},              # no effect
                          {"lever": "nope", "target": "gx", "sign": 1}, {"lever": "a", "target": "gx", "sign": 3}]}
    s = E.score_claims(m, claims, effects=em)
    assert s["n_claims"] == 3 and s["invalid"] == 2                 # the later claim on (a, gx) replaced the first
    assert s["sign_accuracy"] == pytest.approx(2 / 3)
    assert s["lag_accuracy"] == 0.0 and s["lag_mae"] == 2.0
    assert E.score_claims(m, claims, effects=em, lag_tolerance=2)["lag_accuracy"] == 1.0
    assert s["brier"] == pytest.approx(((0.9 - 1) ** 2 + (0.6 - 1) ** 2 + (0.7 - 0) ** 2) / 3)
    assert s["coverage"] == pytest.approx(2 / s["n_true_effects"]) and s["identification"] == pytest.approx(2 / 21)
    empty = E.score_claims(m, {}, effects=em)
    assert empty["sign_accuracy"] is None and empty["coverage"] == 0.0 and empty["identification"] == 0.0


def test_prediction_error_is_near_zero_for_the_truth_and_large_for_a_bad_guess():
    m = chain(noise=0.0)
    eng = B.build_engine(m)
    settings = {"a": 0.7, "b": 0.4}
    v = eng.steady([dict(eng.defaults(0), **settings)])
    out = v[eng._idx["out_0"]]
    good = [{"settings": settings, "target": "score", "value": 100.0 * out},
            {"settings": settings, "target": "gx", "value": v[eng._idx["x"]]},
            {"settings": settings, "target": "cost", "value": 35.0}]
    r = E.prediction_error(m, good)
    assert r["n_valid"] == 3 and r["nmae"] < 0.01 and r["mae"] < 0.01
    bad = [{"settings": settings, "target": "score", "value": 100.0 * out + 60.0},
           {"settings": {"zzz": 1.0}, "target": "score", "value": 1.0}]
    r = E.prediction_error(m, bad)
    assert r["n"] == 2 and r["n_valid"] == 1 and r["nmae"] > 2.0 and r["items"][1]["valid"] is False
    s = E.score_claims(m, {"predictions": good})
    assert s["prediction"]["nmae"] < 0.01 and s["sign_accuracy"] is None


def test_discriminate_log_score():
    cands = [mech("scm_additive", s) for s in range(4)]
    sure = E.discriminate(cands, [0, 0, 1, 0], 2)
    assert sure["prob_true"] == pytest.approx(1.0) and sure["normalized"] == pytest.approx(1.0, abs=1e-6)
    assert sure["top1"] and sure["rank"] == 1 and sure["brier"] == pytest.approx(0.0, abs=1e-9)
    flat = E.discriminate(cands, [1, 1, 1, 1], 2)
    assert flat["log_score"] == pytest.approx(math.log(0.25)) and flat["normalized"] == pytest.approx(0.0)
    wrong = E.discriminate(cands, [0.97, 0.01, 0.01, 0.01], 2)
    assert wrong["normalized"] < 0 and not wrong["top1"] and wrong["rank"] == 2
    certain_wrong = E.discriminate(cands, [1, 0, 0, 0], 2)
    assert math.isfinite(certain_wrong["log_score"]) and certain_wrong["log_score"] < math.log(1e-11)
    better = E.discriminate(cands, [0.1, 0.1, 0.7, 0.1], 2)
    assert flat["log_score"] < better["log_score"] < sure["log_score"]
    for bad in ([1, 1, 1], [1, -1, 1, 1], [0, 0, 0, 0], [1, float("nan"), 1, 1]):
        with pytest.raises(ValueError):
            E.discriminate(cands, bad, 0)
    with pytest.raises(ValueError):
        E.discriminate(cands, [1, 1, 1, 1], 4)


# ====================================================================== the reference policy and regret
def test_oracle_finds_a_known_optimum_without_hints():
    # net(a) = P * sat(a) - c * a has one interior maximum; b only costs money; d only hurts
    k, p_units, c = 4.0, 3000.0, 2400.0
    m = scm.handmade(levers=[{"name": "a", "cost": c}, {"name": "b", "cost": 900.0, "default": 0.5},
                             {"name": "d", "default": 0.5}],
                     latents=[{"id": "x", "terms": [{"kind": "saturating", "src": "a", "w": 1.0, "k": k}], "noise": 0.01},
                              {"id": "h", "terms": [{"kind": "linear", "src": "d", "w": 1.0}]}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}, {"kind": "linear", "src": "h", "w": -0.3}],
                              "bias": 0.5, "noise": 0.02}, price=30.0, units=100.0, overhead=1000.0, horizon=40)
    a_star = math.log(p_units * k / (1 - math.exp(-k)) / c) / k
    o = E.oracle(m, budget=160, seeds=2, seed=1)
    assert abs(o["settings"]["a"] - a_star) < 0.08 and o["settings"]["b"] < 0.08 and o["settings"]["d"] < 0.08
    best = p_units * (0.5 + (1 - math.exp(-k * a_star)) / (1 - math.exp(-k))) - c * a_star - 1000.0
    assert o["expected_net"] == pytest.approx(best, rel=0.03)
    assert o["expected_net"] > o["default_net"] + 500 and o["evals"] <= 160
    assert o["schedule"] == [{"from": 0, "settings": o["settings"]}] and len(o["segments"]) == 1
    again = E.oracle(m, budget=160, seeds=2, seed=1)
    assert again["settings"] == o["settings"] and again["expected_net"] == o["expected_net"]     # seeded


@pytest.mark.parametrize("fam,seed", [("scm_additive", 3), ("scm_stock_gate", 1), ("scm_invu", 2)])
def test_oracle_beats_defaults_and_random_settings(fam, seed):
    m = mech(fam, seed)
    eng = B.build_engine(m)
    o = E.oracle(m, budget=96, seeds=1, engine=eng)
    r0 = m["config"]["seats"][0]["calib"]["revenue_default"]
    assert o["expected_net"] > o["default_net"] + 0.15 * r0
    rng = np.random.default_rng(seed)
    rand = [E.mean_net(m, [{"from": 0, "settings": E._u_to_settings(eng, 0, rng.random(len(eng.levers(0))))}],
                       [900 + k], eng.horizon, engine=eng)[0] for k in range(40)]
    assert o["expected_net"] > max(rand) and np.mean(rand) < o["default_net"]
    con, _ = E.mean_net(m, eng.constructive_optimum(0)["schedule"], list(range(950, 956)), eng.horizon, engine=eng)
    assert o["expected_net"] > 0.9 * con                            # at least as good as the generator's own optimum
    for corner in ({lv.name: lv.hi for lv in eng.levers(0)}, {lv.name: lv.lo for lv in eng.levers(0)}):
        assert E.mean_net(m, [{"from": 0, "settings": corner}], [960, 961], eng.horizon, engine=eng)[0] < \
            0.75 * o["expected_net"]


def test_oracle_follows_a_regime_switch_and_regret_is_normalised():
    m = mech("scm_regime", 0)
    eng = B.build_engine(m)
    cut = eng.truth()["regimes"][0]["at"]
    o = E.oracle(m, budget=120, seeds=1, engine=eng)
    assert [s["from"] for s in o["schedule"]] == [0, cut] and o["schedule"][0]["settings"] != o["schedule"][1]["settings"]
    assert [(s["from"], s["to"]) for s in o["segments"]] == [(0, cut), (cut, eng.horizon)]
    assert o["expected_net"] >= o["net_constant"] > o["default_net"]
    h = eng.horizon

    def play(schedule, seed=77):
        sim = MechSim(m, seed)
        plan = {e["from"]: e["settings"] for e in schedule}
        reports = []
        for t in range(h):
            sim.step(plan.get(t))
            reports.append({"revenue_cents": sim.last[0]["revenue"], "cost_cents": sim.last[0]["cost"]})
        return reports
    r_ref = E.regret(m, play(o["schedule"]), oracle_result=o)
    r_def = E.regret(m, play([{"from": 0, "settings": eng.defaults(0)}]), oracle_result=o)
    r_hi = E.regret(m, play([{"from": 0, "settings": {lv.name: lv.hi for lv in eng.levers(0)}}]), oracle_result=o)
    assert abs(r_ref["normalized"]) < 0.3 and 0.7 < r_def["normalized"] < 1.3 and r_hi["normalized"] > 1.3
    assert r_def["ticks"] == h and r_def["regret_total"] == pytest.approx(r_def["regret_mean"] * h)
    assert r_def["regret_total"] == pytest.approx(r_def["reference_mean"] * h - r_def["net_total"])
    # a trajectory may also be plain numbers, and may start later than tick 0
    late = E.regret(m, [o["segments"][1]["net"]] * 10, oracle_result=o, t0=cut)
    assert late["regret_total"] == pytest.approx(0.0) and late["normalized"] == pytest.approx(0.0)


def test_information_efficiency_curve():
    m = chain()
    em = E.effect_matrix(m, n_base=8)
    full = E.true_claims(em)["effects"]
    half = full[:len(full) // 2]
    r = E.information_efficiency(m, [{"ticks": 100, "claims": {"effects": full}}, {"ticks": 10, "claims": {}},
                                     {"ticks": 50, "claims": {"effects": half}}], effects=em, threshold=0.9)
    ident = [p["identification"] for p in r["curve"]]
    assert [p["ticks"] for p in r["curve"]] == [10, 50, 100]
    assert ident == pytest.approx([0.0, len(half) / len(full), 1.0])
    assert r["auc"] == pytest.approx(ident[1] * 50 / 100)           # beliefs hold until the next checkpoint
    assert r["ticks_to_threshold"] == 100 and r["final"] == 1.0
    assert E.information_efficiency(m, [], effects=em)["auc"] is None
    quick = E.information_efficiency(m, [{"ticks": 20, "claims": {"effects": full}},
                                         {"ticks": 100, "claims": {"effects": full}}], effects=em)
    assert quick["auc"] > r["auc"] and quick["ticks_to_threshold"] == 20     # the same knowledge, earlier


# ====================================================================== the operator-side pipeline in a world
def test_claims_filed_in_a_world_are_scored_outside_it():
    """An agent that only uses tools experiments on its facility and files what it concludes; the operator then
    scores the filings against the hidden mechanism. Honest experiments beat chance, and more of them beat fewer."""
    m = mech("scm_additive", 4)
    spec = WorldSpec("mech-eval", 31, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"},
                                       {"name": "rig", "config": {"rigs": [M.rig_entry(m, holders=["agent"])]}}], {})
    w = World.create(spec, strict=True)
    w.system_call("admin.join", {"handle": "agent", "endowment": 50_000_000})
    agent = "p:agent"

    def call(tool_, **args):
        r = w.execute(agent, {"tool": tool_, "args": args})
        assert r.ok, (tool_, r.error)
        return r.result

    pub = call("rig.spec")
    arm = 6
    for n_levers in (2, len(pub["levers"])):                         # an early filing, then a complete one
        effects = []
        for lv in pub["levers"][:n_levers]:
            means = []
            for value in (lv["min"], lv["max"]):
                call("rig.set", settings={lv["name"]: value})
                w.advance_time(w.now + arm * pub["tick_s"])
                rows = call("rig.history", n=arm)["reports"]
                means.append({t: float(np.mean([r["readings"].get(t, r.get(f"{t}_cents", r.get(t))) for r in rows]))
                              for t in pub["targets"]})
            call("rig.set", settings={lv["name"]: lv["default"]})
            for t in pub["targets"]:
                d = means[1][t] - means[0][t]
                scale = abs(means[0][t]) + abs(means[1][t]) + 1e-9
                effects.append({"lever": lv["name"], "target": t, "sign": 0 if abs(d) < 0.04 * scale else int(np.sign(d)),
                                "confidence": 0.7})
        call("rig.hypothesize", effects=effects, note=f"{n_levers} levers tested")
    rig = w.modules["rig"]
    filed = rig.claims(w, "r1", agent)
    assert len(filed) == 2 and filed[0]["t"] < filed[1]["t"]
    truth = rig.mech("r1")
    assert truth == m
    em = E.effect_matrix(truth, n_base=12)
    scores = [E.score_claims(truth, c, effects=em) for c in filed]
    assert scores[1]["n_claims"] == scores[1]["n_pairs"] > scores[0]["n_claims"]
    assert scores[1]["sign_accuracy"] > 0.6 and scores[1]["identification"] > 0.6 > scores[0]["identification"]
    assert scores[1]["brier"] < 0.25                                 # better than a coin at 0.5 confidence
    eff = E.information_efficiency(truth, [{"ticks": c["t"], "claims": c} for c in filed], effects=em, threshold=0.6)
    assert eff["ticks_to_threshold"] == filed[1]["t"] and eff["curve"][0]["identification"] < eff["final"]
    reg = E.regret(truth, rig.reports(w, "r1"), oracle_result=E.oracle(truth, budget=64, seeds=1))
    assert reg["ticks"] == filed[1]["t"] and reg["normalized"] > 0.5      # experimenting at the extremes is costly


def test_world_code_paths_do_not_import_the_evaluator():
    code = ("import sys; sys.path.insert(0, 'src'); import crucible.mechanisms.module, crucible.mechanisms.sim, "
            "crucible.worldgen.mech_families; "
            "bad = [m for m in sys.modules if m.endswith(('mechanisms.evaluator', 'mechanisms.validity'))]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(
        __file__).resolve().parents[1]))
    assert r.returncode == 0, r.stdout + r.stderr


# ====================================================================== validity
def test_validity_accepts_generated_mechanisms_and_reports_its_evidence():
    rep = V.check_mechanism(mech("scm_additive", 0), **FAST)
    assert rep["ok"] and rep["reasons"] == [] and rep["solvable"] == "yes"
    assert set(rep["checks"]) == {"purity", "bounded", "conservation", "exploits", "consequences", "discoverability",
                                  "difficulty", "claims"}
    assert all(c["ok"] for c in rep["checks"].values())
    c = rep["checks"]
    assert c["purity"]["two_engines_identical"] and c["purity"]["repeat_identical"] and c["purity"]["input_not_mutated"]
    assert c["conservation"]["min_revenue"] >= 0 and c["conservation"]["min_cost"] >= 0
    assert c["conservation"]["max_total_revenue"] <= c["conservation"]["income_cents_per_tick"]
    assert len(c["consequences"]["levers_with_effect_on_net"]) >= 2 and len(c["discoverability"]["detectable"]) >= 2
    assert (c["discoverability"]["ticks_per_arm"], c["discoverability"]["snr_required"]) == (V.ARM_TICKS, V.SNR_MIN)
    assert c["claims"]["declared"] == ["default_zero", "random_loses", "corners_bad"]
    e = rep["economics"][0]
    assert e["reference_net"] > e["default_net"] > e["random_net"] and e["random_net"] < 0
    assert abs(e["default_net"]) < 0.2 * e["gap_reference_default"]
    assert max(e["corner_max_net"], e["corner_min_net"]) - e["default_net"] < V.CORNER_MAX * e["gap_reference_default"]
    ev = rep["evidence"]
    assert ev["search"]["positive"] and ev["constructive"]["positive"] and set(ev["baselines"]) == {
        "explore_commit", "coordinate"}
    again = V.check_mechanism(mech("scm_additive", 0), **FAST)
    again["cpu_s"] = rep["cpu_s"]
    assert again == rep                                              # deterministic in its arguments


def test_validity_checks_dynamism_of_regime_mechanisms():
    rep = V.check_mechanism(mech("scm_regime", 1), **FAST)
    assert rep["ok"], rep["reasons"]
    d = rep["checks"]["dynamism"]
    assert d["ok"] and d["after_minus_before_settings_after_switch"] > 0
    assert d["before_minus_after_settings_before_switch"] > 0
    assert d["switch_at"] == B.build_engine(mech("scm_regime", 1)).truth()["regimes"][0]["at"]


class _Flawed(scm.ScmEngine):
    """A sound hand-made mechanism with one deliberate defect switched on by its config."""
    engine = "flawed_test"
    claims = ()

    def __init__(self, config):
        super().__init__(config)
        self._calls = 0

    @classmethod
    def generate(cls, rng, params):
        cfg = sound()["config"]
        cfg = dict(cfg, flaw=("negative_cost" if rng.random() < 0.5 else None))
        return cfg

    def step(self, state, settings, rng, t):
        flaw = self.config.get("flaw")
        res = super().step(state, settings, rng, t)
        self._calls += 1
        if flaw == "hidden_memory":
            res.readings[0] = {k: v + self._calls for k, v in res.readings[0].items()}
        elif flaw == "global_random":
            res.revenue[0] += float(np.random.random() > 0.5)
        elif flaw == "mutates_input":
            if state["h"].flags.writeable:
                state["h"][0, 0] += 1.0
        elif flaw == "negative_cost":
            if settings[0]["b"] > 0.9:
                res.cost[0] = -50.0
        elif flaw == "unbounded":
            res.state["h"] = res.state["h"] + 10.0 ** (t / 8.0)
        elif flaw == "revenue_above_income":
            res.revenue[0] = res.revenue[0] * 100.0
        elif flaw == "nan":
            if t == 20:
                res.readings[0] = {k: float("nan") for k in res.readings[0]}
        return res


def sound(**kw) -> dict:
    """A small mechanism that passes every universal check: a pays with diminishing returns and costs money, b
    only costs, c pays behind a threshold, d hurts."""
    return scm.handmade(
        levers=[{"name": "a", "cost": 2600.0, "default": 0.0}, {"name": "b", "cost": 1500.0, "default": 0.1},
                {"name": "c", "cost": 700.0, "default": 0.0}, {"name": "d", "default": 0.3}],
        latents=[{"id": "x", "terms": [{"kind": "saturating", "src": "a", "w": 1.0, "k": 5.0}], "noise": 0.01},
                 {"id": "y", "terms": [{"kind": "threshold", "src": "c", "w": 0.6, "theta": 0.55}], "noise": 0.01},
                 {"id": "z", "terms": [{"kind": "linear", "src": "d", "w": 1.0}], "noise": 0.01}],
        outcome={"terms": [{"kind": "linear", "src": "x", "w": 0.5}, {"kind": "linear", "src": "y", "w": 0.5},
                           {"kind": "linear", "src": "z", "w": -0.4}], "bias": 0.62, "noise": 0.02, "cap": 2.0},
        price=40.0, units=100.0, overhead=1850.0, horizon=48, **kw)


@pytest.fixture(scope="module")
def flawed_engine():
    B.register_engine(_Flawed)
    B.mech_family("flawed_family_test", "flawed_test", "train", "base", "test only")(lambda rng, o: {})
    yield
    B.ENGINES.pop("flawed_test", None)
    B.MECH_FAMILIES.pop("flawed_family_test", None)


def flawed(flaw: str | None) -> dict:
    m = sound()
    return {"engine": "flawed_test", "family": None, "seed": 0, "config": dict(m["config"], flaw=flaw)}


def test_validity_accepts_the_sound_hand_made_mechanism(flawed_engine):
    rep = V.check_mechanism(flawed(None), **FAST)
    assert rep["ok"], (rep["reasons"], rep["checks"])
    assert rep["solvable"] == "yes" and "claims" not in rep["checks"]       # this engine declares none


@pytest.mark.parametrize("flaw,check,detail", [
    ("hidden_memory", "purity", "repeat_identical"), ("global_random", "purity", "two_engines_identical"),
    ("mutates_input", "purity", "input_not_mutated"), ("negative_cost", "exploits", None),
    ("unbounded", "bounded", None), ("revenue_above_income", "conservation", None), ("nan", "bounded", None)])
def test_validity_rejects_broken_engines_for_the_right_reason(flawed_engine, flaw, check, detail):
    rep = V.check_mechanism(flawed(flaw), n_seeds=2, budget=24, n_random=16)
    assert not rep["ok"] and check in rep["reasons"], (flaw, rep["reasons"])
    if detail:
        assert rep["checks"][check][detail] is False
    if flaw == "negative_cost":
        assert rep["checks"]["exploits"]["negative_cost"] and rep["checks"]["conservation"]["min_cost"] < 0
    if flaw == "revenue_above_income":
        assert rep["checks"]["exploits"]["revenue_above_income"]


def test_a_mechanism_whose_levers_do_nothing_is_unsolvable():
    m = scm.handmade(levers=[{"name": "a"}, {"name": "b"}],
                     latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 0.0}], "noise": 0.01}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "bias": 0.5, "noise": 0.02}, horizon=40)
    rep = V.check_mechanism(m, n_seeds=2, budget=24, n_random=16)
    assert rep["solvable"] == "no" and "unsolvable" in rep["reasons"] and "consequences" in rep["reasons"]
    assert rep["checks"]["consequences"]["levers_with_effect_on_net"] == []


def test_defaults_already_best_gives_unknown_which_alone_does_not_reject():
    # every lever only costs money: nothing beats the defaults, yet the levers do matter
    m = scm.handmade(levers=[{"name": "a", "cost": 500.0}, {"name": "b", "cost": 300.0}, {"name": "c", "cost": 200.0}],
                     latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 0.0}], "noise": 0.01}],
                     outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "bias": 0.5, "noise": 0.02}, horizon=40)
    rep = V.check_mechanism(m, n_seeds=2, budget=24, n_random=16)
    assert rep["solvable"] == "unknown" and "unsolvable" not in rep["reasons"]
    assert "difficulty" in rep["reasons"] and len(rep["checks"]["consequences"]["levers_with_effect_on_net"]) == 3
    assert not any(b["positive"] for b in rep["evidence"]["baselines"].values())


class _Known(scm.ScmEngine):
    """An engine that can state its own optimum."""
    engine = "known_test"
    claims = ()

    def constructive_optimum(self, seat):
        return {"schedule": [{"from": 0, "settings": dict(self.config["best"])}]}


def test_scripted_operators_failing_does_not_make_a_mechanism_unsolvable():
    # revenue needs a in a narrow window AND b above a level, and a open gate with a outside the window is a
    # disaster: both scripted operators stay at the defaults. The engine's own optimum settles the question.
    m = scm.handmade(
        levers=[{"name": "a", "cost": 300.0}, {"name": "b", "cost": 300.0}, {"name": "c", "cost": 1500.0},
                {"name": "d", "cost": 1500.0}],
        latents=[{"id": "x", "terms": [{"kind": "invu", "src": "a", "w": 1.0, "peak": 0.42, "width": 0.07,
                                         "gate": {"node": "b", "level": 0.45}}], "noise": 0.005}],
        outcome={"terms": [{"kind": "linear", "src": "x", "w": 0.6}], "bias": 0.4, "noise": 0.01, "cap": 2.0},
        price=40.0, units=100.0, overhead=1600.0, horizon=48)
    B.register_engine(_Known)
    try:
        known = {"engine": "known_test", "family": None, "seed": 0,
                 "config": dict(m["config"], best={"a": 0.42, "b": 0.5, "c": 0.0, "d": 0.0})}
        rep = V.check_mechanism(known, n_seeds=3, budget=48, n_random=32)
        ev = rep["evidence"]
        assert not any(b["positive"] for b in ev["baselines"].values()), ev["baselines"]
        assert all(abs(b["gain_over_default"]) < ev["margin"] for b in ev["baselines"].values())
        assert ev["constructive"]["positive"] and ev["constructive"]["gap"] > 1500
        assert rep["solvable"] == "yes" and "unsolvable" not in rep["reasons"]
        # without the engine's statement and with a search too small to find the window, the verdict is
        # "unknown" (the levers do matter), never "no"
        blind = V.check_mechanism(m, n_seeds=3, budget=8, n_random=32)
        assert blind["evidence"]["constructive"] is None
        if not blind["evidence"]["search"]["positive"]:
            assert blind["solvable"] == "unknown" and "unsolvable" not in blind["reasons"]
    finally:
        B.ENGINES.pop("known_test", None)


def test_sample_valid_redraws_deterministically_and_records_why(flawed_engine):
    kw = {"n_seeds": 2, "budget": 24, "n_random": 16}
    seeds = [s for s in range(12)
             if B.make_mechanism("flawed_family_test", s)["config"]["flaw"] == "negative_cost"]
    assert 2 <= len(seeds) <= 10                                     # the test family is flawed about half the time
    got = V.sample_valid("flawed_family_test", seeds[0], max_tries=6, **kw)
    assert got["mech"] is not None and got["tries"] >= 2 and got["report"]["ok"]
    assert got["rejected"][0]["seed"] == seeds[0] and got["rejected"][0]["solvable"] == "yes"
    assert {"conservation", "exploits"} <= set(got["rejected"][0]["reasons"])
    assert got["seed"] == V.redraw_seed("flawed_family_test", seeds[0], got["tries"] - 1)
    assert got["mech"] == B.make_mechanism("flawed_family_test", got["seed"]) and got["mech"]["config"]["flaw"] is None
    again = V.sample_valid("flawed_family_test", seeds[0], max_tries=6, **kw)
    assert (again["seed"], again["tries"], again["rejected"]) == (got["seed"], got["tries"], got["rejected"])
    none = V.sample_valid("flawed_family_test", seeds[0], max_tries=1, **kw)
    assert none["mech"] is None and len(none["rejected"]) == 1
    good = next(s for s in range(12) if s not in seeds)
    first = V.sample_valid("flawed_family_test", good, max_tries=3, **kw)
    assert first["tries"] == 1 and first["rejected"] == [] and first["seed"] == good


def test_acceptance_summary_over_real_families():
    acc = V.acceptance(["scm_delay", "scm_feedback"], n=3, first_seed=40, **FAST)
    for fam in ("scm_delay", "scm_feedback"):
        a = acc[fam]
        assert a["n"] == 3 and a["accepted"] == round(a["rate"] * 3) and sum(a["solvable"].values()) == 3
        assert a["accepted"] >= 2 and a["solvable"].get("no", 0) == 0, a
