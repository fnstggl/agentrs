"""The structural generator (MASTER_SPEC Part 7.2): composes registered primitives into causal graphs.

Two layers:

* `GraphBuilder` - an exact construction kit for one `primitive_graph` mechanism: levers (with a natural mechanism
  for the passive regime), discrete decisions, primitive nodes wired through lagged, weighted ports, observation
  channels, money and affordances. Public names are neutral (`neutral_names`), drawn from the seed.
* `compose(rng, recipe)` - draws a whole mechanism from a structural recipe: a causal-graph topology (chain, fork,
  collider, diamond, layered, hub, feedback, random DAG), which primitives sit on the path from the levers to the
  outcome, temporal structure (instantaneous, fixed lags, mixed lags, stock/decay dynamics), observation channels
  (direct, noisy, delayed, missing, aggregated, selected, on demand), action affordances (levers, discrete,
  experiment, query, program), distractor variables and historical trajectories.

`lookalike_market(rng, rule, difficulty)` builds the Part 7.2 example: four markets with the same public interface
(two levers, the same gauges) that obey different causal rules - lower prices raise purchases (`elastic`), lower
prices lower perceived value (`signal`), price matters only once a compatibility requirement is met (`gate`), the
price effect changes as competition evolves (`drift`). The learner has to infer which rule applies.

Seeds change names and coefficients; recipes change structure. A family is a recipe (plus its interface), never a
seed or a coefficient (Part 7.5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..mechanisms.base import neutral_names
from .primitives import ref, validate_params

TOPOLOGIES = ("chain", "fork", "collider", "diamond", "layered", "hub", "feedback", "random_dag")
CHANNELS = ("direct", "noisy", "delayed", "missing", "aggregated", "selected", "on_demand")
TEMPORAL = ("instant", "lagged", "mixed", "stock")


class GraphBuilder:
    def __init__(self, rng: np.random.Generator, n_names: int = 64):
        self.rng = rng
        self.names = neutral_names(rng, n_names)
        self._k = 0
        self.nodes: list[dict] = []
        self.gauges: list[dict] = []
        self.cfg: dict[str, Any] = {}

    def name(self) -> str:
        n = self.names[self._k]
        self._k += 1
        return n

    def _id(self) -> str:
        return f"n{len(self.nodes):02d}"

    def lever(self, lo: float, hi: float, default: float, cost: float = 0.0, natural: dict | None = None,
              role: str = "lever", integer: bool = False) -> str:
        nid = self._id()
        nd = {"id": nid, "type": "lever", "role": role,
              "lever": {"name": self.name(), "lo": float(lo), "hi": float(hi), "default": float(default),
                        "cost": float(cost), "integer": bool(integer)}}
        if natural:
            nd["natural"] = {"ports": {"x": [[s, int(l), float(w)] for s, l, w in natural.get("x", [])]},
                             "bias": {"x": float(natural.get("bias", 0.0))}, "sd": float(natural.get("sd", 0.0))}
        self.nodes.append(nd)
        return nid

    def discrete(self, values: list[float], cost: list[float] | None = None, default: int = 0,
                 role: str = "decision") -> str:
        nid = self._id()
        self.nodes.append({"id": nid, "type": "discrete", "role": role,
                           "options": {"name": self.name(), "choices": [self.name() for _ in values],
                                       "values": [float(v) for v in values],
                                       "cost": [float(c) for c in (cost or [0.0] * len(values))],
                                       "default": int(default)}})
        return nid

    def prim(self, name: str, params: dict, role: str = "latent", hidden: bool = True, bias: dict | None = None,
             **ports: list) -> str:
        errs = validate_params(name, params)
        if errs:
            raise ValueError(errs)
        nid = self._id()
        self.nodes.append({"id": nid, "type": "prim", "prim": ref(name), "params": params, "role": role,
                           "hidden": bool(hidden),
                           "ports": {p: [[s, int(l), float(w)] for s, l, w in terms] for p, terms in ports.items()},
                           "bias": {p: float(v) for p, v in (bias or {}).items()}})
        return nid

    def gauge(self, src: str | list[str], chain: list[tuple[str, dict]] = (), lag: int = 0, on_demand: bool = False,
              query_cost: float = 0.0) -> str:
        name = self.name()
        g = {"name": name, "src": [src] if isinstance(src, str) else list(src),
             "chain": [{"prim": ref(p), "params": par} for p, par in chain], "lag": int(lag)}
        if on_demand:
            g["on_demand"], g["query_cost"] = True, float(query_cost)
        self.gauges.append(g)
        return name

    def name_of(self, nid: str) -> str:
        nd = next(n for n in self.nodes if n["id"] == nid)
        return (nd.get("lever") or nd.get("options"))["name"]

    def build(self, *, horizon: int, money: dict | None = None, affordances=("levers",), objective: dict | None = None,
              warmup: int = 0, experiment: dict | None = None, endowment: int = 0, brief: str = "",
              **extra) -> dict:
        cfg = {"engine": "primitive_graph", "horizon": int(horizon), "nodes": self.nodes, "gauges": self.gauges,
               "order": topo_order(self.nodes), "affordances": list(affordances), "warmup": int(warmup),
               "objective": objective or {"type": "maximize_net"}, "endowment": int(endowment), "brief": brief}
        if money:
            cfg["money"] = money
            cfg["income_cents_per_tick"] = int(np.ceil(money["price"] * money["cap"])) + 1
        if experiment:
            cfg["experiment"] = experiment
        cfg.update(extra)
        return cfg


def topo_order(nodes: list[dict]) -> list[str]:
    """Evaluation order: every lag-0 input before its consumer. Lagged edges may form cycles (feedback)."""
    deps = {}
    for nd in nodes:
        ports = nd.get("ports", {}) if nd["type"] != "lever" else nd.get("natural", {}).get("ports", {})
        deps[nd["id"]] = {s for terms in ports.values() for s, lag, _ in terms if int(lag) == 0}
    order, done = [], set()
    pending = [nd["id"] for nd in nodes]
    while pending:
        ready = [n for n in pending if deps[n] <= done]
        if not ready:
            raise ValueError(f"instantaneous cycle among {pending}")
        for n in ready:
            order.append(n)
            done.add(n)
        pending = [n for n in pending if n not in done]
    return order


# ====================================================================== drawing a mechanism from a recipe
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def compose(rng: np.random.Generator, recipe: dict) -> dict:
    """Draw a `primitive_graph` mechanism from a structural recipe::

        {"topology": one of TOPOLOGIES, "motifs": [primitive names placed on the lever->outcome paths, in order],
         "n_levers": 2, "n_distractors": 0, "temporal": one of TEMPORAL, "channels": [CHANNELS...],
         "affordances": ["levers", ...], "horizon": 60, "noise": 0.1, "warmup": 0, "price": 20.0}

    The topology decides how lever paths meet: `chain` (one path through every motif), `fork` (one lever drives
    two branches that both reach the outcome), `collider` (paths from different levers meet in an interaction),
    `diamond` (a direct and a mediated path), `layered` (two layers fully wired), `hub` (all levers into one
    node), `feedback` (the outcome feeds back into the first motif through a lag), `random_dag` (a random DAG over
    the motifs). Every lever has a cost, the outcome node earns `price` per unit."""
    topo = recipe.get("topology", "chain")
    if topo not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topo}")
    motifs = list(recipe.get("motifs", ["saturation"]))
    nl = int(recipe.get("n_levers", 2))
    temporal = recipe.get("temporal", "instant")
    noise = float(recipe.get("noise", 0.1))
    gb = GraphBuilder(rng)
    levers = [gb.lever(0.0, 10.0, _u(rng, 1.0, 3.0), cost=_u(rng, 1.0, 3.0)) for _ in range(nl)]

    def lag():
        if temporal == "instant":
            return 0
        if temporal == "lagged":
            return int(rng.integers(1, 4))
        if temporal == "mixed":
            return int(rng.integers(0, 4))
        return 0

    def node(name: str, inputs: list[tuple[str, int, float]]) -> str:
        return _motif_node(gb, rng, name, inputs)

    if topo == "chain":
        x = node(motifs[0], [(lv, lag(), 1.0) for lv in levers])
        for m in motifs[1:]:
            x = node(m, [(x, lag(), 1.0)])
        out = x
    elif topo == "fork":
        a = node(motifs[0], [(levers[0], lag(), 1.0)])
        b = node(motifs[1 % len(motifs)], [(levers[0], lag(), 1.0)] + [(lv, lag(), 0.5) for lv in levers[1:]])
        out = node("linear_nonlinear", [(a, 0, 1.0), (b, 0, 1.0)])
    elif topo == "collider":
        a = node(motifs[0], [(levers[0], lag(), 1.0)])
        b = node(motifs[1 % len(motifs)], [(levers[-1], lag(), 1.0)])
        out = _pair_node(gb, rng, "complementarity" if "complementarity" in motifs else "multiplicative", a, b)
    elif topo == "diamond":
        med = node(motifs[0], [(levers[0], lag(), 1.0)])
        out = node(motifs[-1], [(med, lag(), 1.0), (levers[0], 0, -0.3)] + [(lv, 0, 0.4) for lv in levers[1:]])
    elif topo == "layered":
        layer1 = [node(motifs[k % len(motifs)], [(lv, lag(), _u(rng, 0.3, 1.0)) for lv in levers])
                  for k in range(2)]
        out = node(motifs[-1], [(x, 0, 1.0) for x in layer1])
    elif topo == "hub":
        out = node(motifs[0], [(lv, lag(), _u(rng, 0.5, 1.5)) for lv in levers])
        for m in motifs[1:]:
            out = node(m, [(out, 0, 1.0)])
    elif topo == "feedback":
        fb = node("feedback", [(levers[0], 0, 1.0)])
        x = node(motifs[0], [(fb, lag(), 1.0)] + [(lv, 0, 0.5) for lv in levers[1:]])
        for m in motifs[1:]:
            x = node(m, [(x, 0, 1.0)])
        # the outcome feeds back into the loop through a lag of one tick
        gb.nodes[int(fb[1:])]["ports"]["x"].append([x, 1, -0.2])
        out = x
    else:                                                    # random_dag
        made = list(levers)
        for m in motifs:
            k = int(rng.integers(1, min(3, len(made)) + 1))
            srcs = list(rng.choice(made, size=k, replace=False))
            made.append(node(m, [(s, lag(), _u(rng, 0.4, 1.2)) for s in srcs]))
        out = made[-1]
    sold = gb.prim("saturation", {"cap": 100.0, "half": _u(rng, 5.0, 15.0)}, role="outcome",
                   x=[(out, 0, 1.0)])
    for _ in range(int(recipe.get("n_distractors", 0))):
        d = gb.prim("hidden_confounding", {"mean": _u(rng, 2, 8), "sd": 1.0, "ar": 0.7}, role="distractor")
        gb.gauge(d, [("measurement_noise", {"sd": noise})])
    channels = recipe.get("channels", ["noisy"])
    gb.gauge(sold, _channel(channels, noise, rng))
    gb.gauge(out, _channel(channels, noise, rng), lag=1 if "delayed" in channels else 0)
    if "aggregated" in channels and len(levers) > 1:
        gb.gauge([levers[0], levers[1]], [("partial_observability", {"mode": "sum", "weights": [1.0, 1.0]})])
    if "on_demand" in channels:
        gb.gauge(out, [("measurement_noise", {"sd": noise / 4})], on_demand=True, query_cost=1.0)
    price = float(recipe.get("price", 20.0))
    return gb.build(horizon=int(recipe.get("horizon", 60)),
                    money={"revenue_node": sold, "price": price, "cap": 100.0, "fixed_cost": 0.0},
                    affordances=recipe.get("affordances", ["levers"]), warmup=int(recipe.get("warmup", 0)),
                    experiment={"budget": 1e9, "cost": 0.0, "vars": [gb.name_of(lv) for lv in levers],
                                "measure": [g["name"] for g in gb.gauges if not g.get("on_demand")],
                                "max_ticks": 4, "max_replicates": 4} if "experiment" in recipe.get(
                        "affordances", []) else None,
                    endowment=int(price * 100 * 20),
                    brief="You operate a facility. Set its levers; you are paid for what it sells.",
                    recipe=recipe)


def _channel(channels: list[str], noise: float, rng) -> list[tuple[str, dict]]:
    ch: list[tuple[str, dict]] = []
    if "selected" in channels:
        ch.append(("selection_effect", {"n": 16, "sd": max(noise, 0.5), "c": 0.0}))
    if "noisy" in channels or not ch:
        ch.append(("measurement_noise", {"sd": noise}))
    if "missing" in channels:
        ch.append(("missing_observation", {"p": 0.15, "k": 0.0, "m": 0.0}))
    return ch


def _pair_node(gb: GraphBuilder, rng, name: str, a: str, b: str) -> str:
    if name == "complementarity":
        return gb.prim(name, {"scale": _u(rng, 1.5, 3.0), "rho": -2.0, "w": 0.5}, a=[(a, 0, 1.0)], b=[(b, 0, 1.0)])
    if name == "substitution":
        return gb.prim(name, {"cap": 20.0, "k": _u(rng, 4.0, 8.0), "w": _u(rng, 0.5, 2.0)}, a=[(a, 0, 1.0)],
                       b=[(b, 0, 1.0)])
    return gb.prim("multiplicative", {"k": _u(rng, 0.1, 0.3)}, a=[(a, 0, 1.0)], b=[(b, 0, 1.0)])


def _motif_node(gb: GraphBuilder, rng, name: str, inputs: list[tuple[str, int, float]]) -> str:
    """One node of primitive `name` fed by `inputs` (summed into its main port)."""
    if name == "saturation":
        return gb.prim(name, {"cap": _u(rng, 15.0, 30.0), "half": _u(rng, 3.0, 8.0)}, x=inputs)
    if name == "threshold":
        return gb.prim(name, {"theta": _u(rng, 4.0, 7.0), "lo": 0.0, "hi": _u(rng, 10.0, 20.0)}, x=inputs)
    if name == "non_monotonic":
        return gb.prim(name, {"peak": _u(rng, 3.0, 7.0), "height": _u(rng, 10.0, 20.0)}, x=inputs)
    if name == "linear_nonlinear":
        return gb.prim(name, {"form": "linear", "a": _u(rng, 0.8, 1.5), "b": 0.0}, x=inputs)
    if name == "decay":
        return gb.prim(name, {"rate": _u(rng, 0.2, 0.5), "init": 0.0}, x=inputs)
    if name == "delay":
        return gb.prim(name, {"lag": int(rng.integers(2, 5)), "init": 0.0}, x=inputs)
    if name == "hysteresis":
        lo = _u(rng, 2.0, 4.0)
        return gb.prim(name, {"lo": lo, "hi": lo + _u(rng, 2.0, 3.0), "off": 2.0, "on": _u(rng, 12.0, 20.0),
                              "init": 0.0}, x=inputs)
    if name == "feedback":
        return gb.prim(name, {"g": _u(rng, 0.3, 0.6)}, x=inputs)
    if name in ("complementarity", "multiplicative", "substitution"):
        a = inputs[0]
        b = inputs[1] if len(inputs) > 1 else inputs[0]
        pr = {"complementarity": {"scale": _u(rng, 1.5, 3.0), "rho": -2.0, "w": 0.5},
              "multiplicative": {"k": _u(rng, 0.2, 0.5)},
              "substitution": {"cap": 20.0, "k": _u(rng, 4.0, 8.0), "w": _u(rng, 0.5, 2.0)}}[name]
        return gb.prim(name, pr, a=[a], b=[b])
    if name == "regime_change":
        return gb.prim(name, {"before": 1.0, "after": _u(rng, -0.6, -0.2), "at": float(rng.integers(20, 40)),
                              "trig": 1e18}, x=inputs, trigger=[])
    raise ValueError(f"compose cannot place primitive {name}")


# ====================================================================== the look-alike markets of Part 7.2
LOOKALIKE_RULES = ("elastic", "signal", "gate", "drift")


def lookalike_market(rng: np.random.Generator, rule: str, difficulty: dict) -> dict:
    """One market with levers (price, promotion) and gauges (units sold, store visits, a rival's price index)
    whose causal rule is `rule`. All four rules share this public interface exactly; their mechanisms differ:

    elastic  units fall smoothly with price (logistic demand); promotion adds visits with saturation
    signal   buyers read price as quality: units rise with price up to a peak and fall beyond it
    gate     units ignore price until the compatibility requirement (a discrete choice) is met; then elastic
    drift    the price response is elastic until the rival's share crosses a level, then its sign flips
             (a regime change triggered by the competition's evolution)
    """
    if rule not in LOOKALIKE_RULES:
        raise ValueError(rule)
    noise = float(difficulty.get("observation_noise", 0.15))
    horizon = int(difficulty.get("horizon", 60))
    gb = GraphBuilder(rng)
    price = gb.lever(1.0, 10.0, _u(rng, 4.0, 6.0), cost=0.0)
    promo = gb.lever(0.0, 10.0, _u(rng, 1.0, 3.0), cost=_u(rng, 60.0, 120.0))
    seg = gb.prim("hidden_confounding", {"mean": 0.0, "sd": 0.5, "ar": 0.8}, role="exogenous")
    visits = gb.prim("saturation", {"cap": _u(rng, 60.0, 90.0), "half": _u(rng, 2.0, 5.0)}, role="intermediate",
                     bias={"x": 1.0}, x=[(promo, 0, 1.0), (seg, 0, 1.0)])
    mid = _u(rng, 4.0, 6.5)
    if rule == "elastic":
        conv = gb.prim("linear_nonlinear", {"form": "logistic", "a": 0.9, "b": 0.02, "c": -_u(rng, 0.8, 1.4),
                                            "m": mid}, role="conversion", x=[(price, 0, 1.0)])
    elif rule == "signal":
        conv = gb.prim("non_monotonic", {"peak": _u(rng, 6.5, 8.5), "height": _u(rng, 0.7, 0.9)}, role="conversion",
                       x=[(price, 0, 1.0)])
    elif rule == "gate":
        compat = gb.discrete([0.0, 1.0], cost=[0.0, _u(rng, 40.0, 80.0)], default=0, role="compatibility")
        el = gb.prim("linear_nonlinear", {"form": "logistic", "a": 0.9, "b": 0.02, "c": -_u(rng, 0.8, 1.4),
                                          "m": mid}, role="conversion_if_compatible", x=[(price, 0, 1.0)])
        conv = gb.prim("complementarity", {"scale": 1.0, "rho": -8.0, "w": 0.5}, role="conversion",
                       a=[(el, 0, 1.0)], b=[(compat, 0, 1.0)], bias={"b": 0.03})
    else:                                                    # drift: the effect changes as competition evolves
        rival_share = gb.prim("decay", {"rate": 0.08, "init": 0.0}, role="rival_share", x=[(price, 1, 0.012)])
        el = gb.prim("linear_nonlinear", {"form": "linear", "a": -0.08, "b": 0.0}, role="price_effect",
                     x=[(price, 0, 1.0)], bias={"x": -mid})
        flip = gb.prim("regime_change", {"before": 1.0, "after": -1.0, "at": -1.0, "trig": _u(rng, 0.45, 0.6)},
                       role="regime", x=[(el, 0, 1.0)], trigger=[(rival_share, 1, 1.0)])
        conv = gb.prim("linear_nonlinear", {"form": "logistic", "a": 0.9, "b": 0.02, "c": 6.0, "m": 0.0},
                       role="conversion", x=[(flip, 0, 1.0)])
    units = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(visits, 0, 1.0)], b=[(conv, 0, 1.0)])
    gb.gauge(units, [("measurement_noise", {"sd": noise * 4})])
    gb.gauge(visits, [("measurement_noise", {"sd": noise * 4})])
    rival = gb.prim("linear_nonlinear", {"form": "linear", "a": 0.3, "b": _u(rng, 3.0, 5.0)}, role="rival_price",
                    x=[(seg, 0, 1.0)])
    gb.gauge(rival, [("measurement_noise", {"sd": noise})])
    unit_price = 100.0                                  # cents per unit at price lever 1 (scaled by the lever)
    # revenue = units * price: units are the outcome node; the price lever multiplies via a revenue node
    rev = gb.prim("multiplicative", {"k": unit_price / 10.0}, role="revenue", a=[(units, 0, 1.0)],
                  b=[(price, 0, 1.0)])
    afford = ["levers", "discrete"] if rule == "gate" else ["levers"]
    if rule != "gate":                                   # every market offers the same decision interface
        gb.discrete([0.0, 0.0], cost=[0.0, _u(rng, 40.0, 80.0)], default=0, role="inert_option")
        afford = ["levers", "discrete"]
    return gb.build(horizon=horizon, money={"revenue_node": rev, "price": 1.0, "cap": 100.0 * unit_price,
                                            "fixed_cost": 0.0},
                    affordances=afford, warmup=int(difficulty.get("warmup", 0)), endowment=200_000,
                    objective={"type": "maximize_net", "text": "Run the shop: your net income over the horizon is "
                                                               "what counts."},
                    brief="You run a shop. Levers: your price and your promotion spend. Gauges: units sold, store "
                          "visits, a rival's price index. Decision: an optional certification.",
                    rule=rule)
