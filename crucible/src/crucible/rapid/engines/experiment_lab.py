"""Engine `experiment_lab`: scientific experiment environments (MASTER_SPEC Part 7.3, Part 8.2, Part 8.6).

A laboratory: an apparatus whose settable conditions span a public box, instruments that differ in what they
measure, what they cost and how they err (noise, heavy tails, drift, saturation, calibration bias), an experiment
budget, and a hidden law of nature behind every reading. A designed experiment names the conditions to impose, the
quantities to measure, the instrument, the number of replicates and (where the law unfolds in time) when to sample;
each replicate is an independent preparation, so replication averages measurement noise but not what the
instrument shares across replicates (its drift, its bias). Results come back as the raw replicate readings.

Scientific success is judged by the evaluator on conditions the participant never observed: predictions filed
with the journal (`message` to `journal`, content {"predictions": [{"set": condition, "value": v}, ...]}; only the
last filing counts) are scored against the noise-free law at held-out conditions outside the settable box (or, in
the delayed-consequence worlds, the participant's own money over a long horizon is the measure). Scores use a
tolerance: each held-out condition contributes exp(-((prediction - truth) / tol)^2 / 2), a missing one 0.

The transition code is a set of laws and instruments evaluated on designs, and a slowly evolving apparatus state
(drifting baselines, soil, pilot plants), unlike any other engine here. Scenarios:

    kinetics       a reaction network of unknown scheme (sequential or autocatalytic) with temperature-dependent
                   rates; predict concentrations at temperatures above the settable range
    drift_blanks   a small dose-response effect measured on an instrument whose baseline drifts and whose noise has
                   heavy tails; blanks and replication are needed; predict the effect at unobserved doses
"""
from __future__ import annotations

import math

import numpy as np

from .. import actions as A
from .base import ActionError, RapidEngine, register

MAX_TRIALS_PER_DESIGN = 8


def tolerance_score(preds: list | None, conds: list[dict], truth: list[float], tol: float) -> dict:
    """Evaluator: mean over held-out conditions of exp(-((p - truth) / tol)^2 / 2); a missing prediction is 0."""
    got = []
    for cond in conds:
        v = None
        for p in preds or []:
            if isinstance(p, dict) and p.get("set") == cond and isinstance(p.get("value"), (int, float)) \
                    and not isinstance(p.get("value"), bool) and math.isfinite(float(p["value"])):
                v = float(p["value"])
        got.append(v)
    s = [0.0 if v is None else math.exp(-0.5 * ((v - y) / tol) ** 2) for v, y in zip(got, truth)]
    return {"prediction_score": float(np.mean(s)) if s else 0.0,
            "predictions_filed": sum(v is not None for v in got), "heldout": len(conds)}


def last_predictions(journal: list[dict]) -> list | None:
    preds = None
    for m in journal:
        c = (m.get("content") or {}).get("predictions")
        if isinstance(c, list):
            preds = c
    return preds


def experiment_kind(variables: list[dict], measures: list[str], instruments: dict, max_replicates: int,
                    max_time: int | None, cost_doc: str) -> dict:
    """The schema of one designed experiment."""
    props = {v["name"]: {"type": "number", "minimum": v["min"], "maximum": v["max"]} for v in variables}
    design = {"set": A.S.obj(props, description="conditions imposed in the trial (unset ones take the "
                                                 "apparatus defaults)"),
              "measure": {"type": "array", "items": A.S.s_enum(measures), "minItems": 1, "maxItems": len(measures),
                          "description": "the quantities to measure"},
              "instrument": A.S.s_enum(sorted(instruments), "; ".join(
                  f"{k}: measures {v['measures']}, nominal sd {v['sd']}, cost x{v['cost']}"
                  for k, v in sorted(instruments.items()))),
              "replicates": A.S.s_int("independent preparations", minimum=1, maximum=max_replicates)}
    req = ["set", "measure", "instrument"]
    if max_time:
        design["time"] = A.S.s_int("when the sample is taken (time units after the start)", minimum=1,
                                   maximum=max_time)
        req.append("time")
    return A.S.obj({"kind": {"type": "string", "enum": ["experiment"],
                             "description": f"a designed trial on fresh preparations; {cost_doc}"},
                    "design": A.S.obj(design, req)}, ["kind", "design"])


# ====================================================================== scenarios
class Lab:
    """One laboratory world; `self.c` is the config, `self.h` its hidden part (evaluator data)."""
    name = ""
    motifs_: tuple = ()
    edges_: tuple = ()
    time_axis = False            # designs carry a sampling time

    def __init__(self, cfg: dict):
        self.c, self.h = cfg, cfg["hidden"]

    # public surface
    def manual(self, state) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    def init_state(self, rng) -> dict:
        return {}

    def apply(self, st, action, rng, t) -> dict:
        raise ActionError(f"{action['kind']} is not available in this world")

    def query(self, st, target, args, rng, t) -> dict:
        raise ActionError(f"unknown query target {target}")

    # experiments
    def cost(self, design) -> float:
        inst = self.c["instruments"][design["instrument"]]
        return float(self.c["base_cost"]) * int(design.get("replicates", 1)) * float(inst["cost"])

    def readings(self, st, cond: dict, design: dict, q: str, rng, t) -> list[float]:
        raise NotImplementedError

    def advance(self, st, rng, t) -> dict:
        raise NotImplementedError

    def outcome(self, st, ledger, net) -> dict:
        raise NotImplementedError

    def conservation(self, before, after, out) -> list[str]:
        return []

    def latent(self, st):
        return None

    def truth(self) -> dict:
        nodes = sorted({e[0] for e in self.edges_} | {e[1] for e in self.edges_})
        return {"nodes": nodes, "edges": [{"src": a, "dst": b, "kind": k, "lag": lag} for a, b, k, lag in self.edges_]}

    # helpers
    def defaults(self) -> dict:
        return {v["name"]: float(v["default"]) for v in self.c["vars"]}

    def heldout_truth(self) -> list[float]:
        raise NotImplementedError

    def predict_outcome(self, ledger) -> dict:
        c = self.c
        sc = tolerance_score(last_predictions(ledger.get("journal", [])), c["heldout"], self.heldout_truth(),
                             float(c["tol"]))
        return dict(sc, value=sc["prediction_score"])


LABS: dict[str, type[Lab]] = {}


def lab(cls):
    LABS[cls.name] = cls
    return cls


# ---------------------------------------------------------------------- kinetics (8.2.4)
KIN_STEPS = 20          # integration steps per time unit (sampling times are whole time units)


def kinetics_solve(scheme: str, k30, q, seed_b: float, T, A0, times) -> np.ndarray:
    """Concentrations (n, 3) of the three species at whole-unit `times` for rows of temperature T and loading A0
    (classical Runge-Kutta, all rows integrated together)."""
    T, A0, times = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (T, A0, times))
    n = max(len(T), len(A0), len(times))
    T, A0, times = (np.broadcast_to(x, (n,)).copy() for x in (T, A0, times))
    a = k30[0] * q[0] ** ((T - 30.0) / 10.0)
    b = k30[1] * q[1] ** ((T - 30.0) / 10.0)
    auto = scheme != "sequential"

    def f(y):
        r1 = a * y[:, 0] * (y[:, 1] if auto else 1.0)
        r2 = b * y[:, 1]
        return np.stack([-r1, r1 - r2, r2], axis=1)
    y = np.stack([A0, np.full(n, float(seed_b)), np.zeros(n)], axis=1)
    out = np.zeros((n, 3))
    stops = np.rint(times * KIN_STEPS).astype(int)
    h = 1.0 / KIN_STEPS
    for s in range(1, int(stops.max()) + 1):
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        m = stops == s
        if m.any():
            out[m] = y[m]
    return out


@lab
class Kinetics(Lab):
    """Three species; the participant loads the first and sets the temperature. The hidden scheme is either
    sequential first-order conversion (first -> second -> third) or autocatalytic (first + second -> 2 second,
    second -> third, started by a trace of the second species); both rates rise with temperature by hidden factors
    per 10 degrees. Instruments: a cheap probe of the second species and a costly chromatograph of all three."""
    name = "kinetics"
    time_axis = True
    edges_ = (("loading", "first", "causal", 0), ("first", "second", "transformation", 1),
              ("second", "second", "autocatalysis", 1), ("second", "third", "transformation", 1),
              ("temperature", "rates", "multiplicative", 0), ("rates", "second", "modifies", 0))

    @property
    def motifs_(self):
        base = ["stock", "multiplicative", "linear_nonlinear", "measurement_noise", "delay"]
        if self.h["scheme"] == "autocatalytic":
            base += ["feedback", "threshold"]
        return tuple(base)

    def manual(self, state):
        return {"levers": A.levers_schema(self.c["vars"])}, {}, {}

    def init_state(self, rng):
        return {"budget": float(self.c["budget"]), "live": self.defaults(), "trials": 0}

    def apply(self, st, action, rng, t):
        if action["kind"] == "levers":
            st["live"] = dict(st["live"], **{k: float(v) for k, v in action["values"].items()})
            return {"set": dict(st["live"])}
        return super().apply(st, action, rng, t)

    def conc(self, cond: dict, time) -> np.ndarray:
        nT, nA = self.c["vars"][0]["name"], self.c["vars"][1]["name"]
        h = self.h
        return kinetics_solve(h["scheme"], h["k30"], h["q"], h["seed_b"], cond[nT], cond[nA], time)

    def readings(self, st, cond, design, q, rng, t):
        inst = self.c["instruments"][design["instrument"]]
        j = self.c["species"].index(q)
        y = float(self.conc(cond, design["time"])[0, j])
        r = int(design.get("replicates", 1))
        return [max(0.0, y + inst["sd"] * float(z)) for z in rng.standard_normal(r)]

    def advance(self, st, rng, t):
        c = self.c
        y = float(self.conc(st["live"], c["monitor_time"])[0, 1]) + c["instruments"]["probe"]["sd"] * \
            float(rng.standard_normal())
        return {"obs": {"numbers": {c["monitor_name"]: round(max(0.0, y), 6)},
                        "resources": {"experiment_budget": st["budget"]}}, "revenue": 0.0, "cost": 0.0}

    def heldout_truth(self):
        nT, nA = self.c["vars"][0]["name"], self.c["vars"][1]["name"]
        h = self.h
        conds = self.c["heldout"]
        y = kinetics_solve(h["scheme"], h["k30"], h["q"], h["seed_b"], [x[nT] for x in conds],
                           [x[nA] for x in conds], [x["time"] for x in conds])
        return [float(v) for v in y[:, 1]]

    def outcome(self, st, ledger, net):
        return self.predict_outcome(ledger)

    def latent(self, st):
        return [float(self.conc(st["live"], self.c["monitor_time"])[0, 1])]


# ---------------------------------------------------------------------- drift_blanks (8.2.5)
def hill(x, emax, ec50, n):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return emax * x ** n / (x ** n + ec50 ** n)


@lab
class DriftBlanks(Lab):
    """A small dose-response effect (a Hill curve with hidden maximum, half-effect dose and steepness) read on a
    meter whose baseline drifts as a hidden random walk from tick to tick and whose noise has heavy tails
    (occasional gross outliers). Replicates in one tick share the baseline. The live channel reads a blank every
    tick."""
    name = "drift_blanks"
    motifs_ = ("saturation", "measurement_noise", "decay", "hidden_confounding", "partial_observability",
               "linear_nonlinear")
    edges_ = (("dose", "effect", "saturation", 0), ("effect", "reading", "sensor", 0),
              ("baseline", "baseline", "random_walk", 1), ("baseline", "reading", "confounds", 0),
              ("outliers", "reading", "noise", 0))

    def manual(self, state):
        return {}, {}, {"blank": "a free blank reading of the live channel (same as the gauge)"}

    def init_state(self, rng):
        return {"budget": float(self.c["budget"]), "base": float(self.h["base0"]), "trials": 0}

    def noise(self, rng, n: int) -> np.ndarray:
        h = self.h
        z = rng.standard_normal(n) * self.c["instruments"]["meter"]["sd"]
        out = rng.random(n) < h["out_p"]
        return z + out * rng.standard_normal(n) * h["out_scale"]

    def effect(self, x) -> np.ndarray:
        h = self.h
        return hill(x, h["emax"], h["ec50"], h["n"])

    def readings(self, st, cond, design, q, rng, t):
        nx = self.c["vars"][0]["name"]
        r = int(design.get("replicates", 1))
        y = float(self.effect(cond[nx])) + st["base"]
        return [float(v) for v in y + self.noise(rng, r)]

    def query(self, st, target, args, rng, t):
        return {"blank": round(st["base"] + float(self.noise(rng, 1)[0]), 6)}

    def advance(self, st, rng, t):
        c, h = self.c, self.h
        z = rng.standard_normal(2)
        st["base"] = st["base"] + h["drift_sd"] * float(z[0])
        reading = st["base"] + c["instruments"]["meter"]["sd"] * float(z[1])
        return {"obs": {"numbers": {c["monitor_name"]: round(reading, 6)},
                        "resources": {"experiment_budget": st["budget"]}}, "revenue": 0.0, "cost": 0.0}

    def heldout_truth(self):
        nx = self.c["vars"][0]["name"]
        return [float(v) for v in self.effect([x[nx] for x in self.c["heldout"]])]

    def outcome(self, st, ledger, net):
        return self.predict_outcome(ledger)

    def latent(self, st):
        return [float(st["base"])]


# ====================================================================== the engine
@register
class ExperimentLab(RapidEngine):
    kind = "experiment_lab"
    SECRET_KEYS = ("hidden",)

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = LABS[config["scenario"]](config)
        if type(self.sc).latent is not Lab.latent:
            self.latent = self._latent

    def action_space(self, agent, state=None):
        c = self.config
        kinds, limits, q = self.sc.manual(state)
        kinds, limits, q = dict(kinds), dict(limits), dict(q)
        if c.get("instruments"):
            measures = sorted({m for v in c["instruments"].values() for m in v["measures"]})
            kinds["experiment"] = experiment_kind(c["vars"], measures, c["instruments"], int(c["max_replicates"]),
                                                  int(c["max_time"]) if self.sc.time_axis else None,
                                                  f"costs {c['base_cost']} budget per replicate times the "
                                                  "instrument's cost factor")
        kinds["message"] = A.message_schema(["journal"])
        q.update({"tools": "the action kinds available to you", "history": "your last observations (args: n)",
                  "spec": "the public description of this world"})
        kinds["query"] = A.query_schema(q)
        inner = sorted(kinds)
        kinds["sequence"] = A.sequence_schema(inner)
        limits["sequence_kinds"] = inner
        return A.ActionSpace(kinds, limits)

    def init_state(self, rng):
        return self.sc.init_state(rng)

    def apply(self, st, agent, action, rng, t):
        k = action["kind"]
        if k == "experiment":
            return self._experiment(st, action["design"], rng, t)
        if k == "query":
            return self.sc.query(st, action["target"], action.get("args") or {}, rng, t)
        return self.sc.apply(st, action, rng, t)

    def _experiment(self, st, d, rng, t):
        c = self.config
        inst = c["instruments"][d["instrument"]]
        bad = [q for q in d["measure"] if q not in inst["measures"]]
        if bad:
            raise ActionError(f"instrument {d['instrument']} does not measure {bad}; it measures {inst['measures']}")
        cost = self.sc.cost(d) * len(d["measure"])
        if st["budget"] < cost - 1e-12:
            raise ActionError(f"not enough experiment budget ({st['budget']:.4g} left, this design costs {cost:g})")
        st["budget"] -= cost
        st["trials"] = st.get("trials", 0) + 1
        cond = dict(self.sc.defaults(), **{k: float(v) for k, v in d["set"].items()})
        res = {}
        for q in d["measure"]:
            vals = self.sc.readings(st, cond, d, q, rng, t)
            res[q] = {"values": [round(v, 6) for v in vals], "mean": round(float(np.mean(vals)), 6)}
        return {"design": d, "measured": res, "cost": cost, "budget_left": st["budget"]}

    def advance(self, st, rng, t):
        out = self.sc.advance(st, rng, t)
        a = self.agents[0]
        return {"obs": {a: out["obs"]}, "revenue": {a: float(out["revenue"])}, "cost": {a: float(out["cost"])},
                "info": {}}

    # evaluator side
    def truth(self):
        tr = self.sc.truth()
        tr.update({"engine": self.kind, "scenario": self.sc.name, "motifs": self.motifs()})
        return tr

    def motifs(self):
        return sorted(set(self.sc.motifs_))

    def _latent(self, state):
        return np.asarray(self.sc.latent(state), dtype=float)

    def outcome(self, state, ledger):
        a = self.agents[0]
        rev, cost = float(ledger["revenue"].get(a, 0.0)), float(ledger["cost"].get(a, 0.0))
        out = {"net": rev - cost, "revenue": rev, "cost": cost}
        out.update(self.sc.outcome(state, ledger, rev - cost))
        return out

    def conservation(self, before, after, out):
        errs = []
        for a in self.agents:
            if out["revenue"][a] < 0 or out["cost"][a] < 0:
                errs.append("negative money flow")
        if after.get("budget", 0.0) > before.get("budget", 0.0) + 1e-9:
            errs.append("experiment budget grew")
        return errs + self.sc.conservation(before, after, out)

    def check_state(self, state):
        errs = []
        for k, v in state.items():
            if isinstance(v, float) and (not math.isfinite(v) or abs(v) > 1e12):
                errs.append(f"state {k} is not finite or unbounded")
        return errs
