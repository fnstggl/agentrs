"""Procedural structural composition of Tier 1 worlds (MASTER_SPEC Parts 7, 14.3; Gate T items 1-4).

A *structure* is a JSON genome in the vocabulary of `crucible.rapid.generator` (topologies, temporal classes,
observation channels) plus the composer's own structural modules:

    {"topology":  one of generator.TOPOLOGIES       how the lever pathways meet
     "motifs":    [primitive, ...] (1-4)            what sits on the pathways (crucible.rapid.primitives)
     "combiner":  sum | complementarity | multiplicative | substitution
     "temporal":  one of generator.TEMPORAL         lags and accumulation on the pathways
     "channels":  [generator.CHANNELS ...]          how the participant observes the world
     "n_levers":  1-3
     "drivers":   [DRIVERS ...] (0-2)               dynamic structure (regime shifts, delays, latches, hidden state,
                                                    confounded records, pilot experiments, long horizons)
     "population": {"kind": POPULATIONS, "n", "capability", "mode"}   other participants living in the world
     "horizon":   ticks}

`compose(rng, structure, difficulty)` builds one `primitive_graph` world from it with
`crucible.rapid.generator.GraphBuilder` (the generator's exact construction kit) and `generator.topo_order`; the seed
draws names and coefficients only. Then the world's economics are calibrated: lever costs are set to the marginal
revenue of the lever at a drawn interior target setting (measured on the noise-free mechanism with constant
settings, vectorised over a grid), so that the best constant setting is interior (drawn uniformly in 0.15-0.85 of
each lever's range), the levers' defaults sit at least 0.3 of the range from it on a side drawn independently of
the optimum (so neither the default nor its position predicts where the optimum is), and no canonical setting (all
levers at minimum, middle or maximum) is close to the best. Nothing in the
calibration is visible to a participant except through ordinary consequences (lever costs show up as costs).

Why not `generator.compose` directly: its draws have revenue far above costs, so a canonical constant (all levers
at their maximum) is almost optimal and admission rejects them (1 of 40 random recipes admitted, measured on this
branch); the calibration layer is what makes generated worlds pass the Part 7.4 checks at a useful rate.

Generated structures become ordinary Tier 1 families of `crucible.rapid.families` (split `train`, partition
`v5-train`) only inside `registered(...)`, so the registry other workstreams count is never changed behind their back.
"""
from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from typing import Any

import numpy as np

from ...core.canonical import digest
from ...rapid import families as RF
from ...rapid.generator import CHANNELS, TEMPORAL, TOPOLOGIES, GraphBuilder, topo_order

COMPOSE_VERSION = "n-compose/5"
MOTIFS = ("saturation", "threshold", "non_monotonic", "linear_nonlinear", "decay", "delay", "hysteresis", "feedback",
          "path_dependence")
COMBINERS = ("sum", "complementarity", "multiplicative", "substitution")
DRIVERS = ("regime_shift", "delayed_payoff", "wear_latch", "hidden_state", "confounded_history", "pilot_experiments",
           "long_horizon", "drifting_optimum", "wandering_optimum")
POPULATIONS = ("none", "rivals", "learning_customers", "word_of_mouth", "mixed")
# the Part 8 category a generated family is filed under, by its first driver (or population)
CATEGORY_OF = {"regime_shift": "8.10", "drifting_optimum": "8.10", "wandering_optimum": "8.10", "delayed_payoff": "8.6",
               "wear_latch": "8.6",
               "hidden_state": "8.5",
               "confounded_history": "8.1", "pilot_experiments": "8.2", "long_horizon": "8.6", "rivals": "8.8",
               "mixed": "8.8", "learning_customers": "8.10", "word_of_mouth": "8.4", "none": "8.1"}
GEN_PREFIX = "gen_"
LEVER_HI = 10.0


class CompositionError(ValueError):
    """A structure that is not in the generator's grammar."""


# ====================================================================== the genome
def normalize(structure: dict) -> dict:
    """Validate a structure against the grammar and return its canonical form (sorted sets, defaults filled).
    Raises CompositionError naming every problem."""
    if not isinstance(structure, dict):
        raise CompositionError("a structure is a JSON object")
    s = json.loads(json.dumps(structure))               # JSON-native copy: no objects, no shared references
    errs = []
    allowed = {"topology", "motifs", "combiner", "temporal", "channels", "n_levers", "drivers", "population",
               "horizon", "version"}
    extra = sorted(set(s) - allowed)
    if extra:
        errs.append(f"unknown structure fields {extra}")
    s.setdefault("topology", "chain")
    s.setdefault("motifs", ["saturation"])
    s.setdefault("combiner", "sum")
    s.setdefault("temporal", "instant")
    s.setdefault("channels", ["noisy"])
    s.setdefault("n_levers", 2)
    s.setdefault("drivers", [])
    s.setdefault("population", {"kind": "none"})
    s.setdefault("horizon", 60)
    if s["topology"] not in TOPOLOGIES:
        errs.append(f"topology {s['topology']!r} not in {TOPOLOGIES}")
    if not isinstance(s["motifs"], list) or not 1 <= len(s["motifs"]) <= 4 or \
            any(m not in MOTIFS for m in s["motifs"]):
        errs.append(f"motifs must be 1-4 of {MOTIFS}")
    if s["combiner"] not in COMBINERS:
        errs.append(f"combiner {s['combiner']!r} not in {COMBINERS}")
    if s["temporal"] not in TEMPORAL:
        errs.append(f"temporal {s['temporal']!r} not in {TEMPORAL}")
    if not isinstance(s["channels"], list) or any(c not in CHANNELS for c in s["channels"]):
        errs.append(f"channels must be a subset of {CHANNELS}")
    if not isinstance(s["n_levers"], int) or isinstance(s["n_levers"], bool) or not 1 <= s["n_levers"] <= 3:
        errs.append("n_levers must be 1, 2 or 3")
    if not isinstance(s["drivers"], list) or len(s["drivers"]) > 2 or any(d not in DRIVERS for d in s["drivers"]) \
            or len(set(s["drivers"])) != len(s["drivers"]):
        errs.append(f"drivers must be at most two distinct of {DRIVERS}")
    pop = s["population"]
    if not isinstance(pop, dict) or pop.get("kind") not in POPULATIONS:
        errs.append(f"population.kind must be one of {POPULATIONS}")
    else:
        bad = sorted(set(pop) - {"kind", "n", "capability", "mode"})
        if bad:
            errs.append(f"unknown population fields {bad}")
        if pop["kind"] in ("rivals", "mixed"):
            pop.setdefault("n", 1)
            pop.setdefault("capability", 0.2)
            pop.setdefault("mode", "undercut")
            if not isinstance(pop["n"], int) or not 1 <= pop["n"] <= 3:
                errs.append("population.n must be 1-3")
            if not isinstance(pop["capability"], (int, float)) or not 0.01 <= float(pop["capability"]) <= 0.8:
                errs.append("population.capability must be in [0.01, 0.8]")
            if pop["mode"] not in ("undercut", "match"):
                errs.append("population.mode must be undercut or match")
            if s.get("n_levers", 1) > 2:
                errs.append("worlds with rivals have at most two levers besides the price")
        elif pop["kind"] in ("learning_customers", "word_of_mouth"):
            pop.setdefault("capability", 0.15)
            if not isinstance(pop["capability"], (int, float)) or not 0.01 <= float(pop["capability"]) <= 0.8:
                errs.append("population.capability must be in [0.01, 0.8]")
    if not isinstance(s["horizon"], int) or isinstance(s["horizon"], bool) or not 20 <= s["horizon"] <= 240:
        errs.append("horizon must be an integer in [20, 240]")
    if "long_horizon" in s.get("drivers", []) and isinstance(s["horizon"], int) and s["horizon"] < 100:
        s["horizon"] = 120
    if errs:
        raise CompositionError("; ".join(errs))
    if "noisy" not in s["channels"]:
        s["channels"] = ["noisy"] + s["channels"]
    s["channels"] = sorted(set(s["channels"]))
    s["version"] = COMPOSE_VERSION
    return s


def structure_id(structure: dict) -> str:
    return GEN_PREFIX + digest(normalize(structure))[:12]


# ====================================================================== composition
def _u(rng, lo, hi) -> float:
    return float(rng.uniform(lo, hi))


def _lags(rng, temporal: str):
    def lag():
        if temporal == "lagged":
            return int(rng.integers(1, 4))
        if temporal == "mixed":
            return int(rng.integers(0, 4))
        return 0
    return lag


def _motif(gb: GraphBuilder, rng, name: str, inputs: list) -> str:
    """One pathway node of primitive `name`; inputs are lever-scale (0..10) and so is the output (about)."""
    if name == "saturation":
        return gb.prim(name, {"cap": _u(rng, 11, 15), "half": _u(rng, 1.5, 6.0)}, x=inputs)
    if name == "threshold":
        return gb.prim(name, {"theta": _u(rng, 3.0, 7.0), "lo": _u(rng, 0.0, 2.0), "hi": 10.0}, x=inputs)
    if name == "non_monotonic":
        return gb.prim(name, {"peak": _u(rng, 3.0, 7.5), "height": 10.0}, x=inputs)
    if name == "linear_nonlinear":
        return gb.prim(name, {"form": "logistic", "a": 10.0, "b": 0.0, "c": _u(rng, 0.7, 1.5),
                              "m": _u(rng, 3.0, 7.0)}, x=inputs)
    if name == "decay":
        r = _u(rng, 0.15, 0.4)
        return gb.prim(name, {"rate": r, "init": 0.0}, x=[[s, lg, w * r] for s, lg, w in inputs])
    if name == "delay":
        return gb.prim(name, {"lag": int(rng.integers(2, 6)), "init": 0.0}, x=inputs)
    if name == "hysteresis":
        lo = _u(rng, 2.0, 4.0)
        return gb.prim(name, {"lo": lo, "hi": lo + _u(rng, 1.5, 3.0), "off": _u(rng, 0.0, 2.0), "on": 10.0,
                              "init": 0.0}, x=inputs)
    if name == "feedback":
        g = _u(rng, 0.3, 0.6)
        return gb.prim(name, {"g": g}, x=[[s, lg, w * (1.0 - g)] for s, lg, w in inputs])
    if name == "path_dependence":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        k = _u(rng, 1.5, 3.0)
        pd = gb.prim(name, {"eta": _u(rng, 0.15, 0.35), "k": _u(rng, 5.0, 7.0), "init": 0.5},
                     bias={"x": -sign * k}, x=[[s, lg, sign * w * k / 5.0] for s, lg, w in inputs])
        return gb.prim("linear_nonlinear", {"form": "linear", "a": 10.0, "b": 0.0}, x=[(pd, 0, 1.0)])
    raise CompositionError(f"no placement for motif {name}")


def _combine(gb: GraphBuilder, rng, combiner: str, a: str, b: str) -> str:
    if combiner == "complementarity":
        return gb.prim("complementarity", {"scale": 1.0, "rho": -2.0, "w": _u(rng, 0.3, 0.7)},
                       a=[(a, 0, 1.0)], b=[(b, 0, 1.0)])
    if combiner == "multiplicative":
        return gb.prim("multiplicative", {"k": 0.1}, a=[(a, 0, 1.0)], b=[(b, 0, 1.0)])
    if combiner == "substitution":
        return gb.prim("substitution", {"cap": 12.0, "k": _u(rng, 4.0, 8.0), "w": _u(rng, 0.5, 2.0)},
                       a=[(a, 0, 1.0)], b=[(b, 0, 1.0)])
    return gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, x=[(a, 0, 0.5), (b, 0, 0.5)])


def _pathways(gb: GraphBuilder, rng, s: dict, levers: list[str]) -> tuple[str, list[str], str]:
    """Wire the levers through the motifs by topology. Returns (drive node, pathway nodes, key lever)."""
    topo, motifs, comb = s["topology"], list(s["motifs"]), s["combiner"]
    lag = _lags(rng, s["temporal"])
    src = list(levers)
    if s["temporal"] == "stock":                         # every lever accumulates into a leaky stock first
        src = []
        for lv in levers:
            r = _u(rng, 0.2, 0.45)
            src.append(gb.prim("decay", {"rate": r, "init": 0.0}, x=[(lv, 0, r)]))
    path: list[str] = []
    if topo == "chain":
        x = _motif(gb, rng, motifs[0], [(v, lag(), 1.0 / len(src)) for v in src])
        path.append(x)
        for m in motifs[1:]:
            x = _motif(gb, rng, m, [(x, lag(), 1.0)])
            path.append(x)
        drive = x
    elif topo == "fork":
        a = _motif(gb, rng, motifs[0], [(src[0], lag(), 1.0)])
        b = _motif(gb, rng, motifs[1 % len(motifs)], [(src[0], lag(), 0.6)] + [(v, lag(), 0.4) for v in src[1:]])
        path += [a, b]
        drive = _combine(gb, rng, comb, a, b)
    elif topo == "collider":
        a = _motif(gb, rng, motifs[0], [(src[0], lag(), 1.0)])
        b = _motif(gb, rng, motifs[1 % len(motifs)], [(src[-1], lag(), 1.0)])
        path += [a, b]
        drive = _combine(gb, rng, comb if comb != "sum" else "complementarity", a, b)
    elif topo == "diamond":
        med = _motif(gb, rng, motifs[0], [(src[0], lag(), 1.0)])
        path.append(med)
        drive = _motif(gb, rng, motifs[-1] if len(motifs) > 1 else "saturation",
                       [(med, lag(), 1.0), (src[0], 0, -0.25)] + [(v, 0, 0.3) for v in src[1:]])
        path.append(drive)
    elif topo == "layered":
        layer = [_motif(gb, rng, motifs[k % len(motifs)], [(v, lag(), _u(rng, 0.3, 0.7)) for v in src])
                 for k in range(2)]
        path += layer
        drive = _combine(gb, rng, comb, layer[0], layer[1])
    elif topo == "hub":
        x = _motif(gb, rng, motifs[0], [(v, lag(), 1.0 / len(src)) for v in src])
        path.append(x)
        for m in motifs[1:]:
            x = _motif(gb, rng, m, [(x, 0, 1.0)])
            path.append(x)
        drive = x
    elif topo == "feedback":
        g = _u(rng, 0.25, 0.5)
        loop = gb.prim("feedback", {"g": g}, x=[(src[0], 0, 1.0 - g)] + [(v, 0, (1.0 - g) * 0.5) for v in src[1:]])
        x = _motif(gb, rng, motifs[0], [(loop, lag(), 1.0)])
        path += [loop, x]
        for m in motifs[1:]:
            x = _motif(gb, rng, m, [(x, 0, 1.0)])
            path.append(x)
        gb.nodes[int(loop[1:])]["ports"]["x"].append([x, 1, -_u(rng, 0.05, 0.15)])     # outcome feeds back, lagged
        drive = x
    else:                                                # random_dag over the motifs
        made = list(src)
        for m in motifs:
            k = int(rng.integers(1, min(2, len(made)) + 1))
            pick = [made[int(i)] for i in rng.choice(len(made), size=k, replace=False)]
            made.append(_motif(gb, rng, m, [(p, lag(), 1.0 / k) for p in pick]))
            path.append(made[-1])
        unused = [v for v in src if not any(v == t[0] for n in gb.nodes for terms in n.get("ports", {}).values()
                                            for t in terms)]
        drive = made[-1]
        if unused:
            drive = _combine(gb, rng, comb, drive, _motif(gb, rng, "saturation", [(v, 0, 1.0 / len(unused))
                                                                                   for v in unused]))
    return drive, path, levers[0]


# ranges of the wandering-optimum driver's hidden parameters (lever units of a 0-10 lever; fidelity gen.wandering_optimum)
WANDER = {"half_width": (1.0, 1.5), "path_sd": (2.0, 2.6), "shock_ar": (0.85, 0.9), "lowpass_rate": (0.08, 0.14),
          "centre_low": (2.0, 3.0), "centre_high": (7.0, 8.0)}


def _wandering(gb, rng, drive: str, key: str) -> tuple[str, str]:
    """Driver `wandering_optimum`: the key-lever setting at which what the facility makes fits the market (a taste,
    a position, a specification) follows a hidden, smooth random path for the whole episode, and sales fall off
    quickly away from it. The path is a low-pass filtered AR(1) process (hidden_confounding -> decay, unit gain, so
    it is smooth on the scale of a few ticks and wanders by `A` lever units around its centre c0); the fit is a
    bump of half-width about `w` around c0 + path (the product of two logistic edges, always in (0, 1)), which
    multiplies the drive. A constant setting is right only while the path passes it, so the best constant in
    hindsight earns far less than following the path, and a two-setting rule covers at most two places on it;
    nothing observable reveals the path except how sales respond to the setting. The key lever carries no cost (its
    economics are the fit). Its public default is drawn uniformly over the lever's range, independently of every
    hidden parameter (c0, w, the path): the default carries no information about where the market's taste is
    (DEF-N-8: a default at c0 disclosed the hidden centre, and admission's dominance check then measured gains
    over a waiting policy that already sat at the centre)."""
    w = _u(rng, *WANDER["half_width"])                   # half-width of the fit, lever units (range 0-10)
    A = _u(rng, *WANDER["path_sd"])                      # stationary sd of the path, lever units
    a = _u(rng, *WANDER["shock_ar"])                     # AR(1) coefficient of the shocks
    r = _u(rng, *WANDER["lowpass_rate"])                 # low-pass rate (decay: level' = (1 - r) level + r v)
    b = 1.0 - r
    gain = r * math.sqrt((1 + a * b) / ((1 - a * b) * (1 - a * a) * (1 - b * b)))
    shocks = gb.prim("hidden_confounding", {"mean": 0.0, "sd": A / gain, "ar": a}, role="taste_shock")
    path = gb.prim("decay", {"rate": r, "init": 0.0}, role="taste", x=[(shocks, 0, r)])
    c0 = _u(rng, *WANDER["centre_low"]) if rng.random() < 0.5 else _u(rng, *WANDER["centre_high"])
    k = 2.0 / w
    lo_edge = gb.prim("linear_nonlinear", {"form": "logistic", "a": 1.0, "b": 0.0, "c": k, "m": 0.0},
                      bias={"x": -(c0 - w)}, x=[(key, 0, 1.0), (path, 0, -1.0)])
    hi_edge = gb.prim("linear_nonlinear", {"form": "logistic", "a": 1.0, "b": 0.0, "c": -k, "m": 0.0},
                      bias={"x": -(c0 + w)}, x=[(key, 0, 1.0), (path, 0, -1.0)])
    fit = gb.prim("multiplicative", {"k": 1.0}, role="ideal", a=[(lo_edge, 0, 1.0)], b=[(hi_edge, 0, 1.0)])
    nd = gb.nodes[int(key[1:])]
    lo, hi = float(nd["lever"]["lo"]), float(nd["lever"]["hi"])
    nd["lever"]["default"] = float(rng.uniform(lo, hi))  # independent of c0, w and the path (DEF-N-8)
    nd["lever"]["cost"] = 0.0
    out = gb.prim("multiplicative", {"k": 1.0}, a=[(drive, 0, 1.0)], b=[(fit, 0, 1.0)])
    return out, key


def compose(rng: np.random.Generator, structure: dict, difficulty: dict | None = None) -> dict:
    """One world (a `primitive_graph` config) of `structure`, coefficients drawn from `rng`, at `difficulty`
    (horizon, observation_noise, n_variables, opponent_capability, experiment_cost). Calibrated (module doc)."""
    s = normalize(structure)
    d = dict(difficulty or {})
    for attempt in range(8):
        cfg = _compose_once(rng, s, d)
        ok = calibrate(cfg, rng)
        if ok:
            break
    cfg["structure"] = s
    cfg["calibrated"] = bool(ok)
    return cfg


def _compose_once(rng, s: dict, d: dict) -> dict:
    H = int(d.get("horizon", s["horizon"]))
    noise = float(d.get("observation_noise", 0.3))
    drivers = list(s["drivers"])
    pop = s["population"]
    gb = GraphBuilder(rng)
    nl = int(s["n_levers"])
    levers = [gb.lever(0.0, LEVER_HI, 0.0, cost=0.0) for _ in range(nl)]
    warmup = 0
    confounder = None
    if "confounded_history" in drivers:                  # the previous operator spent with a hidden demand cycle
        confounder = gb.prim("hidden_confounding", {"mean": 0.0, "sd": _u(rng, 0.4, 0.5), "ar": 0.9},
                             role="confounder")
        nd = gb.nodes[int(levers[0][1:])]
        nd["natural"] = {"ports": {"x": [[confounder, 0, 2.0]]}, "bias": {"x": 5.0}, "sd": 0.3}
        warmup = int(rng.integers(30, 41))
    drive, path, key = _pathways(gb, rng, s, levers)
    if "regime_shift" in drivers:                        # at a hidden tick, pushing harder starts to hurt
        at = float(int(rng.integers(int(0.3 * H), int(0.6 * H) + 1)))
        m0 = _u(rng, 3.0, 6.0)
        flip = gb.prim("regime_change", {"before": 1.0, "after": -_u(rng, 0.4, 0.9), "at": at, "trig": 1e18},
                       role="regime", bias={"x": -m0}, x=[(drive, 0, 1.0)], trigger=[])
        drive = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": m0}, x=[(flip, 0, 1.0)])
    if "drifting_optimum" in drivers:                    # the drive level the market likes best drifts during the
        up = bool(rng.random() < 0.5)                    # episode (up in some worlds, down in others): a setting that
        shift = _u(rng, 4.0, 6.0)                        # was right early is wrong late, gradually (no abrupt switch)
        d0 = _u(rng, 1.5, 3.0) if up else _u(rng, 7.0, 8.5)
        rate = _u(rng, 0.03, 0.05)
        reach = 1.0 - (1.0 - rate) ** H                  # share of the drift stock's asymptote reached by the end
        D = gb.prim("decay", {"rate": rate, "init": 0.0}, role="taste", bias={"x": shift * rate / reach}, x=[])
        peak = _u(rng, 2.5, 4.0)
        drive = gb.prim("non_monotonic", {"peak": peak, "height": 10.0}, role="ideal", bias={"x": peak - d0},
                        x=[(drive, 0, 1.0), (D, 0, -1.0 if up else 1.0)])
    if "delayed_payoff" in drivers:                      # the drive builds capacity through a construction delay
        lagd = int(rng.integers(max(3, H // 12), max(4, H // 6) + 1))
        pipe = gb.prim("delay", {"lag": lagd, "init": 0.0}, x=[(drive, 0, 1.0)])
        r = _u(rng, 0.08, 0.2)
        drive = gb.prim("decay", {"rate": r, "init": 0.0}, role="capacity", x=[(pipe, 0, r)])
    if "hidden_state" in drivers:                        # an unobserved, slowly drifting factor scales the effect
        fac = gb.prim("hidden_confounding", {"mean": 1.0, "sd": _u(rng, 0.12, 0.2), "ar": _u(rng, 0.9, 0.97)},
                      role="hidden_state")
        drive = gb.prim("multiplicative", {"k": 1.0}, a=[(drive, 0, 1.0)], b=[(fac, 0, 1.0)])
        gb.gauge(fac, [("measurement_noise", {"sd": 0.1 + noise * 0.2}), ("missing_observation", {"p": 0.6})])
    if pop["kind"] in ("learning_customers", "mixed"):  # customers tire of a strong stimulus and come back slowly
        fatigue = gb.prim("linear_nonlinear", {"form": "linear", "a": -_u(rng, 0.08, 0.12), "b": 1.0},
                          x=[(drive, 1, 1.0)])
        drive = gb.prim("learning_behaviour", {"lr": float(pop.get("capability", 0.15)), "prior": 1.0},
                        role="customers", x=[(drive, 0, 1.0)], o=[(fatigue, 0, 1.0)])
    wander = None
    if "wandering_optimum" in drivers:                   # the key-lever setting the market wants wanders (module doc)
        drive, wander = _wandering(gb, rng, drive, key)
    units = gb.prim("saturation", {"cap": _u(rng, 60.0, 100.0), "half": _u(rng, 4.0, 12.0)}, role="output",
                    x=[(drive, 0, 1.0)])
    if pop["kind"] == "word_of_mouth":                   # demand grows with how well it was served
        dem = gb.prim("endogenous_demand", {"base": 1.0, "elasticity": float(pop.get("capability", 0.15)) * 0.1,
                                            "rate": _u(rng, 0.1, 0.25), "s0": _u(rng, 20.0, 40.0), "init": 1.0},
                      role="demand", served=[(units, 1, 1.0)])
        units = gb.prim("multiplicative", {"k": 1.0}, a=[(units, 0, 1.0)], b=[(dem, 0, 1.0)])
    if "wear_latch" in drivers:                          # pushing the key lever wears the plant; past a hidden level
        wear = gb.prim("stock", {"init": 0.0, "cap": 1e6}, role="wear", inflow=[(key, 0, 1.0)], outflow=[],
                       bias={"outflow": _u(rng, 2.0, 4.0)})
        hi = _u(rng, 0.25, 0.45) * H * 3.0
        health = gb.prim("hysteresis", {"lo": -1.0, "hi": hi, "off": 1.0, "on": _u(rng, 0.3, 0.5), "init": 0.0},
                         role="health", x=[(wear, 0, 1.0)])
        units = gb.prim("multiplicative", {"k": 1.0}, a=[(units, 0, 1.0)], b=[(health, 0, 1.0)])
    price_lever = None
    rev_node = units
    unit_value = _u(rng, 80.0, 130.0)
    money = {"revenue_node": units, "price": unit_value, "cap": 200.0, "fixed_cost": 0.0}
    participants: list[str] = []
    if pop["kind"] in ("rivals", "mixed"):               # adaptive rivals learn the participant's price
        price_lever = gb.lever(1.0, LEVER_HI, _u(rng, 4.0, 6.0), cost=0.0)
        n = int(pop["n"])
        lr = float(d.get("opponent_capability", pop["capability"]))
        opps = [gb.prim("adaptive_opponent", {"lr": lr, "lm": lr, "margin": _u(rng, 0.4, 1.0), "lo": 1.0,
                                              "hi": LEVER_HI, "init": _u(rng, 4.0, 6.5), "mode": pop["mode"]},
                        role="rival", x=[(price_lever, 1, 1.0)]) for _ in range(n)]
        share = gb.prim("strategic_interaction", {"market": 1.0, "beta": _u(rng, 0.5, 0.9), "outside": 0.3},
                        role="share", own=[(price_lever, 0, 1.0)], rival=[(o, 0, 1.0 / n) for o in opps])
        sold = gb.prim("multiplicative", {"k": 1.0}, role="sold", a=[(units, 0, 1.0)], b=[(share, 0, 1.0)])
        rev_node = gb.prim("multiplicative", {"k": unit_value / 5.0}, role="revenue", a=[(sold, 0, 1.0)],
                           b=[(price_lever, 0, 1.0)])
        money = {"revenue_node": rev_node, "price": 1.0, "cap": 200.0 * unit_value * 2.0, "fixed_cost": 0.0}
        gb.gauge(opps[0], [("measurement_noise", {"sd": 0.05})])
        participants += [f"adaptive_opponent:{pop['mode']}"] * n
        units = sold
    if pop["kind"] in ("learning_customers", "mixed"):
        participants.append("learning_customers")
    if pop["kind"] == "word_of_mouth":
        participants.append("endogenous_demand")
    # ---- observation channels
    ch = s["channels"]
    out_chain = [("measurement_noise", {"sd": noise * 3.0})]
    if "selected" in ch:
        out_chain = [("selection_effect", {"n": 16, "sd": max(noise * 3.0, 1.0), "c": 0.0})] + out_chain
    if "missing" in ch:
        out_chain = out_chain + [("missing_observation", {"p": _u(rng, 0.15, 0.3), "k": 0.0, "m": 0.0})]
    gb.gauge(units, out_chain, lag=1 if "delayed" in ch else 0)
    gb.gauge(path[0], [("measurement_noise", {"sd": noise})], lag=int(rng.integers(1, 3)) if "delayed" in ch else 0)
    if "aggregated" in ch and len(path) > 1:
        gb.gauge([path[0], path[1]], [("partial_observability", {"mode": "sum", "weights": [1.0, 1.0]}),
                                      ("measurement_noise", {"sd": noise})])
    query_cost = 0.0
    if "on_demand" in ch:
        query_cost = float(d.get("experiment_cost", 1.0))
        gb.gauge(drive, [("measurement_noise", {"sd": noise / 4})], on_demand=True, query_cost=query_cost)
    for _ in range(int(d.get("n_variables", 0))):
        dv = gb.prim("hidden_confounding", {"mean": _u(rng, 20, 60), "sd": _u(rng, 1, 3), "ar": 0.8},
                     role="distractor")
        gb.gauge(dv, [("measurement_noise", {"sd": noise * 5})])
    if confounder is not None:                           # the confounder also drives the output (passive correlation)
        scale = 1.0 if price_lever is None else unit_value
        with_conf = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                            x=[(rev_node, 0, 1.0), (confounder, 0, _u(rng, 6.0, 10.0) * scale)],
                            bias={"x": _u(rng, 8.0, 12.0) * scale})
        money["revenue_node"] = with_conf
        rev_node = with_conf
    afford = ["levers"]
    experiment = None
    if "pilot_experiments" in drivers:
        afford.append("experiment")
        names = [gb.name_of(lv) for lv in levers]
        experiment = {"budget": float(d.get("resources", 60.0)), "cost": float(d.get("experiment_cost", 1.0)),
                      "vars": names, "measure": [g["name"] for g in gb.gauges if not g.get("on_demand")][:2],
                      "max_ticks": 4, "max_replicates": 3}
    endowment = int(money["price"] * money["cap"] * 30)
    lever_names = [gb.name_of(lv) for lv in levers] + ([gb.name_of(price_lever)] if price_lever else [])
    cfg = gb.build(horizon=H, money=money, affordances=afford, warmup=warmup, experiment=experiment,
                   endowment=endowment,
                   objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                   brief="You operate a facility with spending levers" + (" and a price" if price_lever else "") +
                         ". You are paid for what it sells; the levers cost money.",
                   claim={"lever": key, "outcome": rev_node},
                   calib={"cost_levers": [gb.name_of(lv) for lv in levers if lv != wander],
                          "all_levers": lever_names, "wander_lever": gb.name_of(wander) if wander else None})
    cfg["order"] = topo_order(cfg["nodes"])
    return cfg


# ====================================================================== calibration of the economics
def _noise_free(eng, H: int):
    z = np.zeros((1, eng.n_normal))
    u = np.ones((1, eng.n_uniform))
    return [(z, u)] * H


def schedule_ledgers(eng, state: dict, X: np.ndarray, t0: int = 0, noise=None) -> tuple[np.ndarray, np.ndarray]:
    """Total revenue and cost of P lever schedules X (P, H, L) (lever units) from `state`, vectorised, on `noise`
    (default: the noise-free mechanism). World-side engine code only (`PrimitiveGraph.batch_step`)."""
    P, H, _ = X.shape
    noise = noise or _noise_free(eng, H)
    st = {k: np.repeat(v, P, axis=0) if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1 else
          np.array(v, copy=True) for k, v in eng.copy_state(state).items()}
    lo = np.array([x["min"] for x in eng.public_levers()])
    hi = np.array([x["max"] for x in eng.public_levers()])
    rev, cost = np.zeros(P), np.zeros(P)
    for k in range(H):
        st["lev"] = np.clip(X[:, k, :], lo, hi)
        st["levset"] = np.ones_like(st["lev"])
        z, u = noise[k]
        r = eng.batch_step(st, np.repeat(z, P, axis=0), np.repeat(u, P, axis=0), t0 + k)
        rev += r["revenue"]
        cost += r["cost"]
    return rev, cost


def _grid(L: int) -> np.ndarray:
    g = np.linspace(0.0, 1.0, 21 if L <= 2 else 11)
    return np.array(np.meshgrid(*([g] * L), indexing="ij")).reshape(L, -1).T


def calibrate(cfg: dict, rng: np.random.Generator) -> bool:
    """Set lever costs and defaults (in place) so that the best constant setting is interior and far from every
    canonical setting. Returns False when no interior calibration exists for this draw."""
    from ...rapid.engines import build
    names = cfg["calib"]["all_levers"]
    costed = cfg["calib"]["cost_levers"]
    lev_nodes = {nd["lever"]["name"]: nd for nd in cfg["nodes"] if nd["type"] == "lever"}
    for nm in costed:
        lev_nodes[nm]["lever"]["cost"] = 0.0
    eng = build(cfg)
    H = eng.horizon
    pub = [x["name"] for x in eng.public_levers()]
    L = len(pub)
    F = _grid(L)
    lo = np.array([x["min"] for x in eng.public_levers()])
    hi = np.array([x["max"] for x in eng.public_levers()])
    Xc = lo + F * (hi - lo)
    st = eng.init_state(np.random.default_rng(0))
    rev, _ = schedule_ledgers(eng, st, np.repeat(Xc[:, None, :], H, axis=1))
    cpos = [pub.index(n) for n in costed]
    span = (hi - lo)
    d0 = np.array([(lev_nodes[n]["lever"]["default"] - lo[j]) / span[j] for j, n in enumerate(pub)])
    if not cpos:
        # only a wandering-optimum lever (and possibly a price): no cost to set; the noise-free best constant must
        # still be interior and away from the canonical settings
        if not cfg["calib"].get("wander_lever"):
            return False
        best = F[int(np.argmax(rev))]
        j = pub.index(cfg["calib"]["wander_lever"])
        return bool(0.1 <= best[j] <= 0.9 and abs(best[j] - 0.5) >= 0.08)
    for _ in range(10):
        # interior optimum drawn continuously (a fixed set of fractions was a learnable generator artifact)
        target = rng.uniform(0.15, 0.85, size=L)
        tgt = lo + target * span
        costs = np.zeros(L)
        ok = True
        for j in cpos:
            # marginal revenue per unit of lever j at the target, others at their targets (central difference)
            step = span[j] * 0.1
            a, b = tgt.copy(), tgt.copy()
            a[j] -= step
            b[j] += step
            ra, rb = _interp_rev(F, rev, (a - lo) / span), _interp_rev(F, rev, (b - lo) / span)
            mr = (rb - ra) / (2 * step) / H
            if mr <= 1e-6:
                ok = False
                break
            costs[j] = mr
        if not ok:
            continue
        net = rev - H * (Xc * costs).sum(axis=1)
        best = F[int(np.argmax(net))]
        # the best constant must sit near the target on the costed levers and away from the canonical settings
        if np.max(np.abs(best[cpos] - target[cpos])) > 0.151:
            continue
        canon = [np.full(L, f) for f in (0.0, 0.5, 1.0)]
        nb = float(net.max())
        dflt = d0.copy()
        for j in cpos:                                   # defaults away from the best constant setting, on a side
            side = rng.random() < 0.5                    # drawn independently of where the optimum lies
            lo_ok, hi_ok = target[j] - 0.3 >= 0.0, target[j] + 0.3 <= 1.0
            if side and lo_ok or not hi_ok:
                dflt[j] = float(rng.uniform(0.0, target[j] - 0.3))
            else:
                dflt[j] = float(rng.uniform(target[j] + 0.3, 1.0))
        nd_ = float(_interp_rev(F, net, dflt))
        if nb - nd_ <= 1e-6:
            continue
        shares = [(float(_interp_rev(F, net, c)) - nd_) / (nb - nd_) for c in canon]
        if max(shares) > 0.6:
            continue
        for j, nm in enumerate(pub):
            if nm in costed:
                lev_nodes[nm]["lever"]["cost"] = float(costs[j])
                lev_nodes[nm]["lever"]["default"] = float(lo[j] + dflt[j] * span[j])
        return True
    for nm in costed:                                    # no interior calibration: a plain, uncalibrated draw
        lev_nodes[nm]["lever"]["cost"] = float(rng.uniform(5.0, 50.0))
        lev_nodes[nm]["lever"]["default"] = float(rng.uniform(0.0, LEVER_HI))
    return False


def _interp_rev(F: np.ndarray, v: np.ndarray, f: np.ndarray) -> float:
    """Value at fractional setting f from the grid (nearest grid point)."""
    i = int(np.argmin(np.abs(F - np.clip(f, 0.0, 1.0)).sum(axis=1)))
    return float(v[i])


# ====================================================================== families of generated structures
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 4096


def axes_for(structure: dict) -> dict:
    s = normalize(structure)
    ax: dict[str, Any] = {"horizon": int(s["horizon"]), "observation_noise": 0.3, "n_variables": 0}
    if s["population"]["kind"] in ("rivals", "mixed"):
        ax["opponent_capability"] = float(s["population"]["capability"])
    if "pilot_experiments" in s["drivers"] or "on_demand" in s["channels"]:
        ax["experiment_cost"] = 1.0
    if "pilot_experiments" in s["drivers"]:
        ax["resources"] = 60.0
    return ax


def _builder(structure: dict, name: str):
    s = normalize(structure)

    def build(rng, d):
        key = digest([name, rng.bit_generator.state, sorted(d.items())])
        hit = _CACHE.get(key)
        if hit is None:
            hit = compose(rng, s, d)
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = hit
        return copy.deepcopy(hit)
    return build


def family_spec(structure: dict, *, proposer: str | None = None) -> dict:
    """The rapid family definition of a generated structure (not registered)."""
    s = normalize(structure)
    name = structure_id(s)
    first = (s["drivers"] or [s["population"]["kind"]])[0]
    parts = []
    pop = s["population"]
    if pop["kind"] in ("rivals", "mixed"):
        parts += [f"adaptive_opponent:{pop['mode']}"] * int(pop["n"])
    if pop["kind"] in ("learning_customers", "mixed"):
        parts.append("learning_customers")
    if pop["kind"] == "word_of_mouth":
        parts.append("endogenous_demand")
    if pop["kind"] in ("rivals", "mixed") and int(pop["n"]) > 1:
        parts.append(f"rivals:{int(pop['n'])}")
    return {"name": name, "structure": s, "category": CATEGORY_OF[first], "axes": axes_for(s),
            "participants": tuple(parts), "proposer": proposer,
            "doc": f"generated structure {json.dumps(s, sort_keys=True)}",
            "distinct_because": "measured by fingerprint distance (generated)"}


_REFS: dict[str, int] = {}


def reference_for(structure: dict) -> str | None:
    """The family's mechanism-specific public reference ("module:function"), or None for the generic adaptive
    learner (`rapid.refs_v2.adaptive`): generated worlds whose optimum wanders get the perturbation tracker of
    `genref` (it knows that the optimum moves, nothing hidden)."""
    return "crucible.research.v5.genref:play" if "wandering_optimum" in normalize(structure)["drivers"] else None


@contextmanager
def registered(structures: list[dict] | dict):
    """Register generated structures as rapid families (split train) for the duration of the block; restores the
    rapid registry (and fingerprint cache) exactly on exit. Yields the family names in input order."""
    if isinstance(structures, dict):
        structures = [structures]
    specs = [family_spec(s) for s in structures]
    added = []
    try:
        for sp in specs:
            n = sp["name"]
            if n in RF.FAMILIES and n not in _REFS:
                raise CompositionError(f"{n} already names a rapid family that this module did not register")
            if n not in RF.FAMILIES:
                from .genadmit import claims_for
                RF.FAMILIES[n] = RF.Family(n, "primitive_graph", sp["category"], "generated", "train", sp["doc"],
                                           sp["distinct_because"], _builder(sp["structure"], n), dict(sp["axes"]),
                                           "maximize_net", sp["participants"], claims_for(sp["structure"]),
                                           "crucible.rapid.identify:lever_sign", reference_for(sp["structure"]),
                                           None, True, "1")
            _REFS[n] = _REFS.get(n, 0) + 1
            added.append(n)
        yield [sp["name"] for sp in specs]
    finally:
        from ...rapid import fingerprints as FP
        for n in added:
            _REFS[n] -= 1
            if _REFS[n] <= 0:
                _REFS.pop(n, None)
                RF.FAMILIES.pop(n, None)
                for k in [k for k in FP._CACHE if k[0] == n]:
                    FP._CACHE.pop(k, None)


def is_generated(name: str) -> bool:
    return isinstance(name, str) and name.startswith(GEN_PREFIX)
