"""Structural extensions of the generator's `compose()` recipes (workstream H3, world side).

`compose(rng, recipe)` draws the core of a mechanism - a causal-graph topology, the primitives on the lever paths,
their lags and the observation channels. Some structural classes need more than a recipe can say: a previous
operator whose settings followed a hidden process or reacted to the outcome (historical records with confounding
or reverse causation), a regime switch spliced into a path, a drifting set point, a capacity input, a penalty
rule. `Composed` opens a composed mechanism for such edits through the same construction kit (`GraphBuilder`), so
every family built here is still a `compose()` recipe plus named structural operations - never a hand-written
mechanism - and the result is an ordinary `primitive_graph` config.

    c = Composed(rng, {"topology": "chain", "motifs": ["saturation", "linear_nonlinear"], "n_levers": 2})
    a, b = c.levers
    c.drop_input(c.motifs[0], b)                       # lever b has no causal path
    u = c.prim("hidden_confounding", {...}, role="confounder")
    c.natural(a, [(u, 0, 2.0)], bias=5.0, sd=0.3)       # the previous operator followed u
    cfg = c.finish(horizon=60, warmup=40, claim={...})

Seeds change names and coefficients; the recipe and the operations are the structure.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..mechanisms.base import neutral_names
from .generator import GraphBuilder, compose, topo_order


class Composed:
    def __init__(self, rng: np.random.Generator, recipe: dict):
        self.rng = rng
        self.recipe = dict(recipe)
        cfg = compose(rng, recipe)
        self.base = cfg
        gb = GraphBuilder.__new__(GraphBuilder)
        gb.rng = rng
        gb.nodes = cfg["nodes"]
        gb.gauges = cfg["gauges"]
        gb.cfg = {}
        used = {g["name"] for g in cfg["gauges"]}
        for nd in cfg["nodes"]:
            pub = nd.get("lever") or nd.get("options")
            if pub:
                used.add(pub["name"])
                used |= set(pub.get("choices", []))
        gb.names = neutral_names(rng, 48, used)
        gb._k = 0
        self.gb = gb
        for nd in gb.nodes:                                  # plain str ids (compose may hand back numpy strings)
            for terms in nd.get("ports", {}).values():
                for tm in terms:
                    tm[0] = str(tm[0])

    # ------------------------------------------------------------------ what compose drew
    @property
    def nodes(self) -> list[dict]:
        return self.gb.nodes

    @property
    def levers(self) -> list[str]:
        return [n["id"] for n in self.nodes if n["type"] == "lever"]

    @property
    def motifs(self) -> list[str]:
        """The primitive nodes compose placed on the lever paths, in creation order."""
        return [n["id"] for n in self.nodes if n["type"] == "prim" and n.get("role") == "latent"]

    @property
    def sold(self) -> str:
        return next(n["id"] for n in self.nodes if n.get("role") == "outcome")

    @property
    def core(self) -> str:
        """The node whose value the outcome saturates (the end of the lever paths)."""
        return self.node(self.sold)["ports"]["x"][0][0]

    def node(self, nid: str) -> dict:
        return self.nodes[int(nid[1:])]

    def prim_of(self, nid: str) -> str:
        return self.node(nid).get("prim", "").split("@")[0]

    def name_of(self, nid: str) -> str:
        return self.gb.name_of(nid)

    def gauge_of(self, nid: str) -> str | None:
        return next((g["name"] for g in self.gb.gauges if g["src"] == [nid] and not g.get("on_demand")), None)

    # ------------------------------------------------------------------ edits
    def set_lever(self, nid: str, **kw) -> None:
        lv = self.node(nid)["lever"]
        for k, v in kw.items():
            lv[{"lo": "lo", "hi": "hi", "default": "default", "cost": "cost"}[k]] = float(v)

    def set_params(self, nid: str, **params) -> None:
        self.node(nid)["params"].update({k: (v if isinstance(v, (str, bool, int)) and not isinstance(v, float)
                                             else float(v)) for k, v in params.items()})

    def add_input(self, nid: str, port: str, src: str, lag: int, w: float) -> None:
        self.node(nid).setdefault("ports", {}).setdefault(port, []).append([src, int(lag), float(w)])

    def drop_input(self, nid: str, src: str) -> None:
        nd = self.node(nid)
        for p, terms in nd.get("ports", {}).items():
            nd["ports"][p] = [tm for tm in terms if tm[0] != src]

    def set_input(self, nid: str, port: str, terms: list[tuple[str, int, float]]) -> None:
        self.node(nid)["ports"][port] = [[s, int(lag), float(w)] for s, lag, w in terms]

    def scale_inputs(self, nid: str, src: str, f: float) -> None:
        for terms in self.node(nid).get("ports", {}).values():
            for tm in terms:
                if tm[0] == src:
                    tm[2] = float(tm[2] * f)

    def natural(self, nid: str, terms: list[tuple[str, int, float]], bias: float, sd: float) -> None:
        """The lever's passive regime (followed until the participant first sets it; the previous operator)."""
        self.node(nid)["natural"] = {"ports": {"x": [[s, int(lag), float(w)] for s, lag, w in terms]},
                                     "bias": {"x": float(bias)}, "sd": float(sd)}

    def prim(self, name: str, params: dict, role: str = "latent_ext", hidden: bool = True, bias: dict | None = None,
             **ports) -> str:
        return self.gb.prim(name, params, role=role, hidden=hidden, bias=bias, **ports)

    def lever(self, *a, **kw) -> str:
        return self.gb.lever(*a, **kw)

    def discrete(self, *a, **kw) -> str:
        return self.gb.discrete(*a, **kw)

    def gauge(self, *a, **kw) -> str:
        return self.gb.gauge(*a, **kw)

    def splice(self, nid: str, name: str, params: dict, port: str = "x", role: str = "latent_ext",
               **extra_ports) -> str:
        """Insert a new primitive node between `nid` and every node that reads it; gauges keep reading `nid`."""
        consumers = [(nd, p, tm) for nd in self.nodes for p, terms in nd.get("ports", {}).items() for tm in terms
                     if tm[0] == nid]
        new = self.prim(name, params, role=role, **{port: [(nid, 0, 1.0)]}, **extra_ports)
        for nd, p, tm in consumers:
            if nd["id"] != new:
                tm[0] = new
        return new

    def drop_gauge(self, nid: str) -> None:
        self.gb.gauges[:] = [g for g in self.gb.gauges if g["src"] != [nid]]

    # ------------------------------------------------------------------ the config
    def finish(self, *, horizon: int | None = None, price: float | None = None, warmup: int | None = None,
               affordances=None, objective: dict | None = None, brief: str | None = None,
               endowment: int | None = None, experiment: dict | None = None, **extra) -> dict:
        b = self.base
        money = dict(b["money"])
        if price is not None:
            money["price"] = float(price)
        cfg = self.gb.build(horizon=int(horizon if horizon is not None else b["horizon"]), money=money,
                            affordances=list(affordances if affordances is not None else b["affordances"]),
                            objective=objective or {"type": "maximize_net",
                                                    "text": "Maximise your net income over the horizon."},
                            warmup=int(warmup if warmup is not None else b["warmup"]),
                            experiment=experiment if experiment is not None else b.get("experiment"),
                            endowment=int(endowment if endowment is not None else b["endowment"]),
                            brief=brief if brief is not None else b["brief"], recipe=self.recipe, **extra)
        cfg["order"] = topo_order(cfg["nodes"])
        return cfg


def u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


# ====================================================================== economics from the mechanism itself
def _settle(cfg: dict, settings: dict[str, float], ticks: int, t0: int):
    from .engines.primitive_graph import PrimitiveGraph
    eng = PrimitiveGraph(cfg)
    st = eng.empty_state(1)
    eng.set_levers(st, {eng.config["nodes"][eng.idx[k]]["lever"]["name"]: float(v) for k, v in settings.items()},
                   clip=False)
    z = np.zeros((1, eng.n_normal))
    uu = np.ones((1, eng.n_uniform))
    cur = None
    for k in range(ticks):
        cur, _ = eng.tick(st, z, uu, t0 + k)
    return eng, st, cur


def settled_revenue(cfg: dict, settings: dict[str, float], ticks: int = 60, t0: int = -100_000) -> float:
    """Noise-free revenue per tick after holding lever `settings` (node id -> value) for `ticks` ticks, starting at
    tick `t0` (negative: before any timed regime switch). Every other lever keeps its default. Runs the engine's
    own tick on a private state; nothing here is observable."""
    from .engines.primitive_graph import PrimitiveGraph
    eng = PrimitiveGraph(cfg)
    st = eng.empty_state(1)
    eng.set_levers(st, {eng.config["nodes"][eng.idx[k]]["lever"]["name"]: float(v) for k, v in settings.items()},
                   clip=False)
    z = np.zeros((1, eng.n_normal))
    uu = np.ones((1, eng.n_uniform))
    revs = []
    for k in range(ticks):
        cur, _ = eng.tick(st, z, uu, t0 + k)
        revs.append(float(eng.money_flows(st, cur)[0][0]))
    return float(np.mean(revs[-10:]))           # the mean of the last ticks (robust to a settled oscillation)


def settled_value(cfg: dict, node: str, settings: dict[str, float], ticks: int = 60, t0: int = -100_000) -> float:
    """Noise-free settled value of one node under held settings."""
    eng, st, cur = _settle(cfg, settings, ticks, t0)
    return float(cur[0, eng.idx[node]])


def settled_port(cfg: dict, node: str, settings: dict[str, float], port: str = "x", ticks: int = 60,
                 t0: int = -100_000) -> float:
    """Noise-free settled input (weighted port sum) of one node under held settings."""
    eng, st, cur = _settle(cfg, settings, ticks, t0)
    return float(eng._port(eng.idx[node], port, cur, st["h"])[0])


def marginal_revenue(cfg: dict, lever: str, at: dict[str, float], delta: float = 0.25, **kw) -> float:
    """d(settled revenue)/d(lever) at the settings `at` (central difference)."""
    hi = dict(at, **{lever: at[lever] + delta})
    lo = dict(at, **{lever: at[lever] - delta})
    return (settled_revenue(cfg, hi, **kw) - settled_revenue(cfg, lo, **kw)) / (2 * delta)


def price_levers_at(cfg: dict, target: dict[str, float], scale: dict[str, float] | None = None,
                    floor: float = 0.0, **kw) -> dict[str, float]:
    """Set each lever's unit cost to its marginal revenue at `target` (times `scale`), so that `target` is a
    stationary point of net income: the economic optimum sits at a hidden interior setting drawn by the seed,
    not at a corner or the middle of the ranges. Returns the costs set."""
    scale = scale or {}
    costs = {}
    for lv, _ in target.items():
        m = marginal_revenue(cfg, lv, target, **kw)
        costs[lv] = max(float(m) * float(scale.get(lv, 1.0)), floor)
    for lv, c in costs.items():
        cfg["nodes"][int(lv[1:])]["lever"]["cost"] = c
    return costs


def price_key_and_costly(cfg: dict, key: str, key_at: float, costly: str, excess: float, floor: float = 0.0,
                         others: dict[str, float] | None = None, **kw) -> dict[str, float]:
    """The usual economics of a two-lever family: the key lever's unit cost equals its marginal revenue at the
    hidden setting `key_at` (with the costly lever off), and the costly lever's unit cost exceeds its own marginal
    revenue by the factor `excess` everywhere (it is worth less than it costs, so the optimum leaves it off). A
    constant action that moves every lever together - all low, all middle, all high - is then wrong on one lever
    or the other."""
    at = dict(others or {}, **{key: key_at, costly: 0.0})
    ck = max(marginal_revenue(cfg, key, at, **kw), floor)
    cb = max((settled_revenue(cfg, dict(at, **{costly: 0.5}), **kw) - settled_revenue(cfg, at, **kw)) / 0.5, 0.0)
    cb = max(cb * excess, floor, 0.02 * ck)
    cfg["nodes"][int(key[1:])]["lever"]["cost"] = float(ck)
    cfg["nodes"][int(costly[1:])]["lever"]["cost"] = float(cb)
    return {key: float(ck), costly: float(cb)}


def jsonable_ports(cfg: dict) -> dict[str, Any]:
    """Every port term as plain JSON (ids as str, lags as int, weights as float)."""
    for nd in cfg["nodes"]:
        for terms in list(nd.get("ports", {}).values()) + list(nd.get("natural", {}).get("ports", {}).values()):
            for tm in terms:
                tm[0], tm[1], tm[2] = str(tm[0]), int(tm[1]), float(tm[2])
    return cfg
