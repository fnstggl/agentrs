"""Versioned registry of the 28 composable causal primitives (MASTER_SPEC Part 7.1).

A primitive is a reusable mechanism component with a version, a JSON parameter schema and a vectorised
implementation. It maps named input ports to one output over a batch of independent instances (arrays of shape
(B,), or (B, m) for the vector primitives), may keep its own state between ticks, and declares how many standard
normal and uniform draws per instance it needs each tick, so that a composition knows exactly which random numbers
each component consumes (this is what makes a batched simulation bit-identical to single-instance runs).

Parameters may be scalars, shared by the batch, or arrays of shape (B,), one per instance, so one composition can
run many seeds of a family at once. `apply` never mutates its input state: it returns a new state dict.

Kinds of primitive:

    node     a causal function of its ports (most primitives)
    source   an exogenous or latent process with no ports (hidden confounding)
    channel  an observation channel applied to a reading on its way to a participant (measurement noise, missing
             observations, partial observability, selection effects)
    vector   operates on a vector of claimants or components (resource competition, cascading failures); each also
             has a scalar node form for use inside a causal graph

`REGISTRY[name]` is `{"version", "doc", "params" (JSON schema), "ports", "kind", "stateful", "implementation",
"check"}`; `check()` runs the primitive's characteristic-behaviour experiment (in `primitive_checks`) and returns
what it measured. `get(name, version)` refuses a version that is not the registered one, so a stored mechanism
config that names `name@version` cannot silently run under different code.
"""
from __future__ import annotations

from typing import Any

import numpy as np

REGISTRY_VERSION = "rapid-primitives/1"

_NUM = {"type": "number"}


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": list(required or props), "additionalProperties": False}


def _p(p: dict, k: str, default: Any = None) -> Any:
    v = p.get(k, default)
    return v


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Primitive:
    name = ""
    version = "1.0"
    kind = "node"
    ports: tuple[str, ...] = ("x",)
    stateful = False
    doc = ""
    params: dict = _schema({})

    def init(self, p: dict, B: int) -> dict:
        return {}

    def draws(self, p: dict) -> tuple[int, int]:
        """(standard normal draws, uniform draws) per instance per tick."""
        return 0, 0

    def apply(self, p: dict, x: dict[str, np.ndarray], s: dict, z: np.ndarray, u: np.ndarray, t: int
              ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


# ====================================================================== functional forms
class LinearNonlinear(Primitive):
    name = "linear_nonlinear"
    doc = ("Linear and nonlinear relationships: out = b + a*f(x) with f linear, quadratic (x + c*x^2), a signed "
           "power |x|^k or logistic 1/(1+exp(-c(x-m))).")
    params = _schema({"form": {"type": "string", "enum": ["linear", "quadratic", "power", "logistic"]},
                      "a": _NUM, "b": _NUM, "c": _NUM, "m": _NUM, "k": _NUM}, ["form", "a", "b"])

    def apply(self, p, x, s, z, u, t):
        v, form = x["x"], p["form"]
        a, b = p["a"], p["b"]
        if form == "linear":
            f = v
        elif form == "quadratic":
            f = v + _p(p, "c", 0.0) * v * v
        elif form == "power":
            f = np.sign(v) * np.abs(v) ** _p(p, "k", 2.0)
        elif form == "logistic":
            f = sigmoid(_p(p, "c", 1.0) * (v - _p(p, "m", 0.0)))
        else:
            raise ValueError(f"unknown form {form}")
        return b + a * f, s


class Threshold(Primitive):
    name = "threshold"
    doc = "Thresholds: out = hi when x >= theta, else lo (a hard step)."
    params = _schema({"theta": _NUM, "lo": _NUM, "hi": _NUM})

    def apply(self, p, x, s, z, u, t):
        return np.where(x["x"] >= p["theta"], p["hi"] + 0.0 * x["x"], p["lo"] + 0.0 * x["x"]), s


class Saturation(Primitive):
    name = "saturation"
    doc = "Saturation: out = cap * x / (x + half) for x >= 0 (diminishing returns, bounded by cap)."
    params = _schema({"cap": _NUM, "half": _NUM})

    def apply(self, p, x, s, z, u, t):
        v = np.maximum(x["x"], 0.0)
        return p["cap"] * v / (v + p["half"]), s


class NonMonotonic(Primitive):
    name = "non_monotonic"
    doc = ("Non-monotonic response: out = height * r * exp(1 - r), r = max(x,0)/peak; rises to `height` at x = peak "
           "and falls beyond it.")
    params = _schema({"peak": _NUM, "height": _NUM})

    def apply(self, p, x, s, z, u, t):
        r = np.maximum(x["x"], 0.0) / p["peak"]
        return p["height"] * r * np.exp(1.0 - r), s


class Complementarity(Primitive):
    name = "complementarity"
    ports = ("a", "b")
    doc = ("Complementarity: CES aggregate with rho < 0, out = scale*(w a^rho + (1-w) b^rho)^(1/rho); either input "
           "alone is worth almost nothing.")
    params = _schema({"scale": _NUM, "rho": _NUM, "w": _NUM})

    def apply(self, p, x, s, z, u, t):
        a = np.maximum(x["a"], 1e-9)
        b = np.maximum(x["b"], 1e-9)
        rho, w = p["rho"], p["w"]
        return p["scale"] * (w * a ** rho + (1.0 - w) * b ** rho) ** (1.0 / rho), s


class Substitution(Primitive):
    name = "substitution"
    ports = ("a", "b")
    doc = ("Substitution: the inputs are interchangeable at rate w and share one saturating response, out = "
           "cap*(1 - exp(-(a + w b)/k)); more of one lowers the marginal value of the other.")
    params = _schema({"cap": _NUM, "k": _NUM, "w": _NUM})

    def apply(self, p, x, s, z, u, t):
        e = np.maximum(x["a"], 0.0) + p["w"] * np.maximum(x["b"], 0.0)
        return p["cap"] * (1.0 - np.exp(-e / p["k"])), s


class Multiplicative(Primitive):
    name = "multiplicative"
    ports = ("a", "b")
    doc = "Multiplicative interaction: out = k * a * b (the effect of one input scales with the other)."
    params = _schema({"k": _NUM})

    def apply(self, p, x, s, z, u, t):
        return p["k"] * x["a"] * x["b"], s


# ====================================================================== latent and temporal
class HiddenConfounding(Primitive):
    name = "hidden_confounding"
    kind = "source"
    ports = ()
    stateful = True
    doc = ("Hidden confounding: a latent AR(1) common cause v' = mean + ar (v - mean) + sd z that drives several "
           "variables and is never observable; it makes passive correlation differ from the effect of intervening.")
    params = _schema({"mean": _NUM, "sd": _NUM, "ar": _NUM})

    def init(self, p, B):
        return {"v": np.zeros(B) + p["mean"]}

    def draws(self, p):
        return 1, 0

    def apply(self, p, x, s, z, u, t):
        v = p["mean"] + p["ar"] * (s["v"] - p["mean"]) + p["sd"] * z[:, 0]
        return v, {"v": v}


class Delay(Primitive):
    name = "delay"
    stateful = True
    doc = "Delayed effect: out_t = x_(t-lag) (a pipeline of `lag` ticks)."
    params = _schema({"lag": {"type": "integer", "minimum": 1, "maximum": 64}, "init": _NUM}, ["lag"])

    def init(self, p, B):
        return {"buf": np.zeros((B, int(p["lag"]))) + _p(p, "init", 0.0)}

    def apply(self, p, x, s, z, u, t):
        buf = s["buf"]
        out = buf[:, 0].copy()
        nb = np.concatenate([buf[:, 1:], x["x"][:, None]], axis=1)
        return out, {"buf": nb}


class Stock(Primitive):
    name = "stock"
    ports = ("inflow", "outflow")
    stateful = True
    doc = ("Accumulating stock: level' = min(level + inflow - taken, cap), taken = min(outflow, level + inflow); the "
           "output is the level. Flows are conserved: level change = inflow - taken - spill.")
    params = _schema({"init": _NUM, "cap": _NUM})

    def init(self, p, B):
        return {"level": np.zeros(B) + p["init"], "inflow": np.zeros(B), "taken": np.zeros(B), "spill": np.zeros(B)}

    def apply(self, p, x, s, z, u, t):
        inflow = np.maximum(x["inflow"], 0.0)
        avail = s["level"] + inflow
        taken = np.minimum(np.maximum(x["outflow"], 0.0), avail)
        rest = avail - taken
        level = np.minimum(rest, p["cap"])
        return level, {"level": level, "inflow": inflow, "taken": taken, "spill": rest - level}


class Decay(Primitive):
    name = "decay"
    stateful = True
    doc = "Decay: level' = (1 - rate) * level + x; an impulse fades geometrically with half-life ln2/-ln(1-rate)."
    params = _schema({"rate": _NUM, "init": _NUM})

    def init(self, p, B):
        return {"level": np.zeros(B) + p["init"]}

    def apply(self, p, x, s, z, u, t):
        level = (1.0 - p["rate"]) * s["level"] + x["x"]
        return level, {"level": level}


class Hysteresis(Primitive):
    name = "hysteresis"
    stateful = True
    doc = ("Hysteresis: a latch that switches on when x >= hi and off when x <= lo (lo < hi); between the two the "
           "output depends on the path that led there.")
    params = _schema({"lo": _NUM, "hi": _NUM, "off": _NUM, "on": _NUM, "init": _NUM}, ["lo", "hi", "off", "on"])

    def init(self, p, B):
        return {"on": np.zeros(B) + _p(p, "init", 0.0)}

    def apply(self, p, x, s, z, u, t):
        v = x["x"]
        st = np.where(v >= p["hi"], 1.0, np.where(v <= p["lo"], 0.0, s["on"]))
        return p["off"] + (p["on"] - p["off"]) * st, {"on": st}


class Feedback(Primitive):
    name = "feedback"
    stateful = True
    doc = "Feedback: y' = x + g * y; a constant input settles at x / (1 - g) (amplified for g > 0, damped for g < 0)."
    params = _schema({"g": _NUM})

    def init(self, p, B):
        return {"y": np.zeros(B)}

    def apply(self, p, x, s, z, u, t):
        y = x["x"] + p["g"] * s["y"]
        return y, {"y": y}


class PathDependence(Primitive):
    name = "path_dependence"
    stateful = True
    doc = ("Path dependence by increasing returns: a share s moves toward sigmoid(k (s - 0.5) + x) at rate eta; with "
           "k > 4 the dynamics have two attractors, so the order of early pushes decides where it locks in.")
    params = _schema({"eta": _NUM, "k": _NUM, "init": _NUM}, ["eta", "k"])

    def init(self, p, B):
        return {"s": np.zeros(B) + _p(p, "init", 0.5)}

    def apply(self, p, x, s, z, u, t):
        sh = s["s"] + p["eta"] * (sigmoid(p["k"] * (s["s"] - 0.5) + x["x"]) - s["s"])
        return sh, {"s": sh}


# ====================================================================== interaction and populations
class NetworkEffect(Primitive):
    name = "network_effect"
    ports = ("x", "base")
    doc = ("Network effect: the value of a unit rises with the installed base, out = x * (1 + strength * base); "
           "total value base * out grows faster than the base (Metcalfe-like).")
    params = _schema({"strength": _NUM})

    def apply(self, p, x, s, z, u, t):
        return x["x"] * (1.0 + p["strength"] * x["base"]), s


class Congestion(Primitive):
    name = "congestion"
    ports = ("load", "cap")
    doc = "Congestion: latency = t0 * (1 + alpha * (load / cap)^beta) (BPR form): flat when idle, steep near capacity."
    params = _schema({"t0": _NUM, "alpha": _NUM, "beta": _NUM})

    def apply(self, p, x, s, z, u, t):
        r = np.maximum(x["load"], 0.0) / np.maximum(x["cap"], 1e-9)
        return p["t0"] * (1.0 + p["alpha"] * r ** p["beta"]), s


def ration(claims: np.ndarray, supply: np.ndarray) -> np.ndarray:
    """Proportional rationing of `supply` (B,) over `claims` (B, m): every claim is met when they fit, otherwise all
    are scaled down by the same factor. The allocations never sum to more than the supply."""
    c = np.maximum(claims, 0.0)
    tot = c.sum(axis=-1)
    f = np.where(tot > supply, supply / np.maximum(tot, 1e-300), 1.0)
    return c * f[..., None]


class ResourceCompetition(Primitive):
    name = "resource_competition"
    kind = "vector"
    ports = ("own", "rival", "supply")
    doc = ("Resource competition: claimants share a fixed supply by proportional rationing (`ration`); the node form "
           "returns the own claimant's allocation, which falls as the rival claims more once supply binds.")
    params = _schema({})

    def apply(self, p, x, s, z, u, t):
        claims = np.stack([x["own"], x["rival"]], axis=-1)
        return ration(claims, np.maximum(x["supply"], 0.0))[:, 0], s


class StrategicInteraction(Primitive):
    name = "strategic_interaction"
    ports = ("own", "rival")
    doc = ("Strategic interaction: logit competition between two prices with an outside option, own demand = "
           "market * e^(-beta own) / (e^(-beta own) + e^(-beta rival) + outside); the best own price depends on the "
           "rival's.")
    params = _schema({"market": _NUM, "beta": _NUM, "outside": _NUM})

    def apply(self, p, x, s, z, u, t):
        eo = np.exp(-p["beta"] * x["own"])
        er = np.exp(-p["beta"] * x["rival"])
        return p["market"] * eo / (eo + er + p["outside"]), s


class AdaptiveOpponent(Primitive):
    name = "adaptive_opponent"
    stateful = True
    doc = ("Adaptive opponent: a rival that learns the agent's typical action (memory m' = m + lm (x - m)) and moves "
           "its own action toward a best response (undercut: m - margin; match: m) at rate lr; `lr` is its "
           "capability.")
    params = _schema({"lr": _NUM, "lm": _NUM, "margin": _NUM, "lo": _NUM, "hi": _NUM, "init": _NUM,
                      "mode": {"type": "string", "enum": ["undercut", "match"]}})

    def init(self, p, B):
        return {"o": np.zeros(B) + p["init"], "m": np.zeros(B) + p["init"]}

    def apply(self, p, x, s, z, u, t):
        m = s["m"] + p["lm"] * (x["x"] - s["m"])
        target = m - p["margin"] if p["mode"] == "undercut" else m
        target = np.clip(target, p["lo"], p["hi"])
        o = s["o"] + p["lr"] * (target - s["o"])
        return o, {"o": o, "m": m}


class StateDependentEffect(Primitive):
    name = "state_dependent_effect"
    ports = ("a", "s")
    doc = "State-dependent action effect: out = gain * a * (setpoint - s); the same action helps below the setpoint and hurts above it."
    params = _schema({"gain": _NUM, "setpoint": _NUM})

    def apply(self, p, x, s, z, u, t):
        return p["gain"] * x["a"] * (p["setpoint"] - x["s"]), s


class RegimeChange(Primitive):
    name = "regime_change"
    ports = ("x", "trigger")
    stateful = True
    doc = ("Changing causal regime: out = x * before until the switch, x * after from then on; the switch happens at "
           "tick `at` (if >= 0) or when the trigger first reaches `trig`, and is permanent.")
    params = _schema({"before": _NUM, "after": _NUM, "at": _NUM, "trig": _NUM}, ["before", "after", "at", "trig"])

    def init(self, p, B):
        return {"on": np.zeros(B)}

    def apply(self, p, x, s, z, u, t):
        at = p["at"]
        timed = (np.zeros_like(x["x"]) + at >= 0) & (t >= np.zeros_like(x["x"]) + at)
        trig = x["trigger"] >= p["trig"]
        on = np.where(timed | trig, 1.0, s["on"])
        return x["x"] * np.where(on > 0, p["after"], p["before"]), {"on": on}


class EndogenousDemand(Primitive):
    name = "endogenous_demand"
    ports = ("served",)
    stateful = True
    doc = ("Endogenous demand: demand moves toward base * (1 + elasticity * (served - s0)) at rate `rate`; how well "
           "demand was served now changes demand later (never in the same tick).")
    params = _schema({"base": _NUM, "elasticity": _NUM, "rate": _NUM, "s0": _NUM, "init": _NUM})

    def init(self, p, B):
        return {"d": np.zeros(B) + p["init"]}

    def apply(self, p, x, s, z, u, t):
        d = s["d"]
        nd = d + p["rate"] * (p["base"] * (1.0 + p["elasticity"] * (x["served"] - p["s0"])) - d)
        nd = np.maximum(nd, 0.0)
        return d, {"d": nd}                  # this tick's demand was fixed by the past; the update shows next tick


class LearningBehaviour(Primitive):
    name = "learning_behaviour"
    ports = ("x", "o")
    stateful = True
    doc = ("Learning-driven behavioural change: participants hold a belief w updated from experienced outcomes "
           "(w' = w + lr (o - w)); their response to a stimulus is x * w, so the same stimulus draws a different "
           "response after experience.")
    params = _schema({"lr": _NUM, "prior": _NUM})

    def init(self, p, B):
        return {"w": np.zeros(B) + p["prior"]}

    def apply(self, p, x, s, z, u, t):
        out = x["x"] * s["w"]
        w = s["w"] + p["lr"] * (x["o"] - s["w"])
        return out, {"w": w}


class CascadingFailure(Primitive):
    name = "cascading_failure"
    kind = "vector"
    ports = ("stress",)
    doc = ("Cascading failure: components with loads and capacities on a ring; an overloaded component fails and its "
           "load moves to its surviving neighbours, which may fail in turn. `cascade(load, cap, adj)` is the vector "
           "form; the node form applies a uniform extra stress to m components and returns the failed fraction.")
    params = _schema({"m": {"type": "integer", "minimum": 3, "maximum": 256}, "load": _NUM, "spread": _NUM,
                      "cap": _NUM})

    def apply(self, p, x, s, z, u, t):
        m = int(p["m"])
        B = x["stress"].shape[0]
        base = p["load"] * (1.0 + p["spread"] * np.cos(np.linspace(0.0, 2 * np.pi, m, endpoint=False)))
        load = np.zeros((B, m)) + base[None, :] + x["stress"][:, None]
        cap = np.zeros((B, m)) + p["cap"]
        adj = ring(m)
        failed = cascade(load, cap, adj)
        return failed.mean(axis=1), s


def ring(m: int) -> np.ndarray:
    a = np.zeros((m, m))
    for i in range(m):
        a[i, (i + 1) % m] = a[(i + 1) % m, i] = 1.0
    return a


def cascade(load: np.ndarray, cap: np.ndarray, adj: np.ndarray, max_rounds: int | None = None) -> np.ndarray:
    """Load redistribution on failure. `load`, `cap` (B, m); `adj` (m, m) or (B, m, m) 0/1. Returns the failed mask
    (B, m) as floats. Each round every newly overloaded component fails and its load is split equally among its
    alive neighbours (lost if it has none). Exact and deterministic."""
    load = np.array(load, dtype=float)
    B, m = load.shape
    A = np.broadcast_to(adj, (B, m, m))
    failed = np.zeros((B, m), dtype=bool)
    for _ in range(max_rounds or m + 1):
        new = (load > cap) & ~failed
        if not new.any():
            break
        failed |= new
        alive = ~failed
        nb = A * alive[:, None, :]                      # neighbours of each component that are still alive
        deg = nb.sum(axis=2)
        share = np.where(new & (deg > 0), load / np.maximum(deg, 1), 0.0)
        load = load + (nb * share[:, :, None]).sum(axis=1)
        load = np.where(failed, 0.0, load)
    return failed.astype(float)


# ====================================================================== observation channels
class PartialObservability(Primitive):
    name = "partial_observability"
    kind = "channel"
    ports = ("xs",)
    doc = ("Partial observability: the reading is a non-injective function of several hidden quantities (a weighted "
           "sum, a maximum, or a sum rounded to `step`), so different states can look identical.")
    params = _schema({"mode": {"type": "string", "enum": ["sum", "max", "quantize"]}, "weights": {"type": "array"},
                      "step": _NUM}, ["mode", "weights"])

    def apply(self, p, x, s, z, u, t):
        xs = x["xs"]                                    # (B, m)
        w = np.asarray(p["weights"], dtype=float)
        if p["mode"] == "max":
            return (xs * w).max(axis=-1), s
        tot = (xs * w).sum(axis=-1)
        if p["mode"] == "quantize":
            st = p.get("step", 1.0)
            return np.floor(tot / st) * st, s
        return tot, s


class MeasurementNoise(Primitive):
    name = "measurement_noise"
    kind = "channel"
    doc = "Measurement noise: reading = x + sd z (additive) or x (1 + sd z) (multiplicative); repeated readings average out."
    params = _schema({"sd": _NUM, "mult": {"type": "boolean"}}, ["sd"])

    def draws(self, p):
        return 1, 0

    def apply(self, p, x, s, z, u, t):
        if p.get("mult"):
            return x["x"] * (1.0 + p["sd"] * z[:, 0]), s
        return x["x"] + p["sd"] * z[:, 0], s


class MissingObservation(Primitive):
    name = "missing_observation"
    kind = "channel"
    doc = ("Missing observations: the reading is absent (NaN) with probability p, or, when k != 0, with probability "
           "sigmoid(k (x - m)) (missing not at random, which biases what remains).")
    params = _schema({"p": _NUM, "k": _NUM, "m": _NUM}, ["p"])

    def draws(self, p):
        return 0, 1

    def apply(self, p, x, s, z, u, t):
        v = x["x"]
        k = p.get("k", 0.0)
        prob = (np.zeros_like(v) + p["p"]) if np.all(np.asarray(k) == 0) else sigmoid(k * (v - p.get("m", 0.0)))
        return np.where(u[:, 0] < prob, np.nan, v), s


class SelectionEffect(Primitive):
    name = "selection_effect"
    kind = "channel"
    doc = ("Selection effect: each tick n units with outcomes x + sd z are drawn and only those above the cut c are "
           "reported; the reading is the mean of the reported units (NaN when none survive), biased upward and "
           "flattened against changes in x.")
    params = _schema({"n": {"type": "integer", "minimum": 1, "maximum": 256}, "sd": _NUM, "c": _NUM})

    def draws(self, p):
        return int(p["n"]), 0

    def apply(self, p, x, s, z, u, t):
        y = x["x"][:, None] + p["sd"] * z[:, :int(p["n"])]
        keep = y > (np.zeros_like(x["x"]) + p["c"])[:, None]
        cnt = keep.sum(axis=1)
        tot = np.where(keep, y, 0.0).sum(axis=1)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), s


PRIMITIVES: dict[str, Primitive] = {}
for _cls in (LinearNonlinear, Threshold, Saturation, NonMonotonic, Complementarity, Substitution, Multiplicative,
             HiddenConfounding, Delay, Stock, Decay, Hysteresis, Feedback, PathDependence, NetworkEffect, Congestion,
             ResourceCompetition, StrategicInteraction, PartialObservability, StateDependentEffect, RegimeChange,
             AdaptiveOpponent, MeasurementNoise, MissingObservation, SelectionEffect, CascadingFailure,
             EndogenousDemand, LearningBehaviour):
    _inst = _cls()
    PRIMITIVES[_inst.name] = _inst

# The Part 7.1 list, in its order: spec wording -> primitive name.
SPEC_LIST = {
    "Linear and nonlinear relationships": "linear_nonlinear", "Thresholds": "threshold", "Saturation": "saturation",
    "Non-monotonic responses": "non_monotonic", "Complementarity": "complementarity", "Substitution": "substitution",
    "Multiplicative interactions": "multiplicative", "Hidden confounding": "hidden_confounding",
    "Delayed effects": "delay", "Accumulating stocks": "stock", "Decay": "decay", "Hysteresis": "hysteresis",
    "Feedback": "feedback", "Path dependence": "path_dependence", "Network effects": "network_effect",
    "Congestion": "congestion", "Resource competition": "resource_competition",
    "Strategic interaction": "strategic_interaction", "Partial observability": "partial_observability",
    "State-dependent action effects": "state_dependent_effect", "Changing causal regimes": "regime_change",
    "Adaptive opponents": "adaptive_opponent", "Measurement noise": "measurement_noise",
    "Missing observations": "missing_observation", "Selection effects": "selection_effect",
    "Cascading failures": "cascading_failure", "Endogenous demand": "endogenous_demand",
    "Learning-driven behavioral changes": "learning_behaviour",
}


def _check(name: str):
    def run() -> dict:
        from . import primitive_checks
        return primitive_checks.CHECKS[name]()
    return run


REGISTRY: dict[str, dict] = {
    n: {"version": pr.version, "doc": pr.doc, "params": pr.params, "ports": list(pr.ports), "kind": pr.kind,
        "stateful": pr.stateful, "implementation": pr, "check": _check(n)}
    for n, pr in PRIMITIVES.items()}


def get(name: str, version: str | None = None) -> Primitive:
    """The implementation of `name`; with `version`, refuses any other registered version."""
    if name not in PRIMITIVES:
        raise KeyError(f"unknown primitive {name!r}")
    pr = PRIMITIVES[name]
    if version is not None and version != pr.version:
        raise ValueError(f"primitive {name} is at version {pr.version}, the config asks for {version}")
    return pr


def ref(name: str) -> str:
    """`name@version`, how configs name a primitive."""
    return f"{name}@{get(name).version}"


def parse_ref(r: str) -> Primitive:
    name, _, ver = r.partition("@")
    return get(name, ver or None)


def validate_params(name: str, params: dict) -> list[str]:
    from ..core.schema import validate
    sch = dict(REGISTRY[name]["params"])
    return validate(sch, params, f"{name}.params")


def run(name: str, params: dict, inputs: dict[str, Any], ticks: int | None = None, seed: int = 0,
        state: dict | None = None) -> tuple[np.ndarray, dict]:
    """Run a primitive on its own. `inputs[port]` is an array (B,) for one tick, or (ticks, B) with `ticks`; noise
    comes from `default_rng(seed)`. Returns (outputs (B,) or (ticks, B), final state). For experiments and tests."""
    pr = get(name)
    rng = np.random.default_rng(seed)
    if ticks is None:
        xs = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in inputs.items()}
        B = len(next(iter(xs.values()))) if xs else 1
        st = state if state is not None else pr.init(params, B)
        nn, nu = pr.draws(params)
        out, st = pr.apply(params, xs, st, rng.standard_normal((B, nn)), rng.random((B, nu)), 0)
        return out, st
    arr = {k: np.asarray(v, dtype=float) for k, v in inputs.items()}
    B = next(iter(arr.values())).shape[1] if arr else 1
    st = state if state is not None else pr.init(params, B)
    nn, nu = pr.draws(params)
    outs = []
    for t in range(ticks):
        xs = {k: v[t] for k, v in arr.items()}
        o, st = pr.apply(params, xs, st, rng.standard_normal((B, nn)), rng.random((B, nu)), t)
        outs.append(o)
    return np.stack(outs), st
