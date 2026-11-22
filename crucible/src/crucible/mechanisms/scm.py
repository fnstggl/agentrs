"""Engine "scm": rule-compositional structural causal models.

A mechanism is a directed graph over five kinds of node:

    lever       set by a seat each tick (normalised internally to u in [0, 1])
    exogenous   hidden AR(1) driver nobody controls
    latent      hidden quantity: bias + sum of edge terms + noise (optionally a *stock*: accumulation with decay)
    gauge       what a seat reads: an affine, noisy, possibly lagged view of one latent
    outcome     units sold this tick: clip(bias + terms + noise, 0, demand cap)

Edge function kinds: `linear`, `saturating` (diminishing returns), `threshold` (a step at a hidden level), `invu`
(inverted U: non-monotone, a hidden peak), `product` (two sources multiply), `delay` (the source value k >= 2 ticks
ago), any of them optionally behind a `gate` (active only while a third node is above a hidden level), and inflows
into a `stock` node. In regime families some weights change at a hidden tick.

The graph is *composed from motifs* by the generator: chain, fork, collider, mediator, hidden confounder (an
exogenous driver moves both a gauge and the outcome, the gauge itself causes nothing), feedback through a lag,
gate, threshold, interaction, delay, stock, inverted U, regime switch, distractor levers, costly levers. A family
fixes which motif blocks may be composed; the seed draws how many, how they are wired (in parallel into the
outcome, in series, sharing levers), every functional form and parameter, the lever types and defaults, which
latents are observed and under which opaque names.

Economics per tick and seat: revenue = unit price x outcome units; cost = overhead + sum of (cost rate x lever
level). Generation calibrates these on the noise-free steady state so that default settings earn about nothing,
the best settings earn clearly positive net, uniformly random settings lose money and neither corner (everything
max / everything min) is close to the best; a draw that cannot be calibrated is redrawn. `mechanisms.validity`
verifies the same properties independently by stochastic simulation.

The per-tick update is generated as straight-line Python from the config (`_compile`), so a step costs a few
microseconds and is a pure function of (config, state, settings, the tick's normal draws, t).
"""
from __future__ import annotations

import math

import numpy as np

from .base import Engine, Gauge, Lever, StepResult, mech_family, neutral_names, register_engine

CLIP = 6.0                      # hard bound on every latent (a safety net: calibrated mechanisms stay well inside)
G_MIN = 0.20                    # calibration: best net >= G_MIN x default revenue
R_MIN = 0.18                    # calibration: mean net of uniformly random settings <= -R_MIN x default revenue
CORNER = 0.5                    # calibration: net of either corner <= CORNER x best net
SENS_MIN = 0.05                 # calibration: at least two levers move net by >= SENS_MIN x default revenue
MAX_ATTEMPTS = 40
R_TARGET = 0.30                 # calibration: among feasible cost levels pick the one whose random loss is nearest
LAMBDAS = (0.4, 0.5, 0.62, 0.78, 0.97, 1.2, 1.5, 1.9, 2.35, 2.9)

#                (lo,   hi,    integer)
LEVER_TYPES = [(0.0, 1.0, False), (0.0, 10.0, False), (0.0, 100.0, False), (-1.0, 1.0, False), (-5.0, 5.0, False),
               (0.0, 4.0, True), (0.0, 8.0, True), (1.0, 5.0, True)]
LEVER_W = [3, 3, 2, 1, 1, 1, 1, 1]
PHI = {"linear": (0.3, 0.6), "saturating": (0.9, 1.8), "threshold": (0.3, 0.7), "delay": (0.3, 0.6),
       "product": (0.2, 0.4), "invu": (0.0, 0.25)}
SERIES_OK = ("direct", "chain", "threshold", "delay", "stock", "invu", "gate")


def _sat(z: float, k: float, c: float) -> float:
    """Saturating response, 0 at z <= 0 and 1 at z = 1 (c = 1 / (1 - exp(-k)))."""
    return (1.0 - math.exp(-k * z)) * c if z > 0.0 else 0.0


def _invu(s: float, m: float, iw: float) -> float:
    """Inverted U: 1 at the peak m, falling quadratically, floored at -1."""
    d = (s - m) * iw
    d = 1.0 - d * d
    return d if d > -1.0 else -1.0


def _pick(rng, items, weights=None):
    if weights is None:
        return items[int(rng.integers(len(items)))]
    w = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _u(rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


# ====================================================================== structure draft
class _Draft:
    """Draws one causal graph from a family recipe. Nodes are created in an order that is topological for
    same-tick edges (every lag-0 source exists before its target), which the compiler relies on."""

    def __init__(self, rng: np.random.Generator, recipe: dict):
        self.rng, self.r = rng, recipe
        self.levers: list[dict] = []
        self.exos: list[dict] = []
        self.lats: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.range: dict[str, tuple[float, float]] = {}     # noise-free value range of every node
        self.out_terms: dict[int, list[dict]] = {}
        self.blocks: dict[int, list[dict]] = {}             # seat -> [{"node", "term", "pol"}]
        self.force_gauge: list[str] = []
        self.regime_terms: list[dict] = []

    # ---- nodes -----------------------------------------------------------------------------------
    def lever(self, seat: int) -> str:
        lo, hi, integer = _pick(self.rng, LEVER_TYPES, LEVER_W)
        node = {"id": f"lev_{len(self.levers):02d}", "role": "lever", "seat": seat, "lo": lo, "hi": hi,
                "integer": integer, "c": 0.0, "u_def": None, "far_from": None}
        self.levers.append(node)
        self.by_id[node["id"]] = node
        self.range[node["id"]] = (0.0, 1.0)
        return node["id"]

    def lever_active(self, seat: int) -> str:
        lv = self.lever(seat)
        self.by_id[lv]["active"] = True
        self.by_id[lv]["c"] = _u(self.rng, 0.1, 0.3)
        return lv

    def exo(self, rho: float | None = None, sd: float = 1.0) -> str:
        node = {"id": f"exo_{len(self.exos):02d}", "role": "exogenous", "seat": None,
                "rho": _u(self.rng, 0.6, 0.95) if rho is None else float(rho), "sd": float(sd)}
        self.exos.append(node)
        self.by_id[node["id"]] = node
        self.range[node["id"]] = (0.0, 0.0)
        return node["id"]

    def latent(self, seat: int, terms: list[dict], decay: float | None = None) -> str:
        lo = hi = 0.0
        for t in terms:
            a, b = self._term_range(t)
            lo += a
            hi += b
        bias = _u(self.rng, 0.0, 0.12) - min(lo, 0.0)          # keeps the noise-free value non-negative
        node = {"id": f"lat_{len(self.lats):02d}", "role": "latent", "seat": seat, "bias": bias,
                "noise": _u(self.rng, *self.r["latent_noise"]), "decay": decay, "terms": list(terms)}
        self.lats.append(node)
        self.by_id[node["id"]] = node
        self.range[node["id"]] = (bias + lo, bias + hi)
        return node["id"]

    def _term_range(self, t: dict) -> tuple[float, float]:
        a, b = self.range[t["src"]]
        k = t["kind"]
        ws = [t["w"]] + ([t["w2"]] if "w2" in t else [])
        if k in ("linear", "delay"):
            f = [a, b]
        elif k == "saturating":
            f = [_sat((a - t.get("x0", 0.0)) * t.get("ix", 1.0), t["k"], 1.0 / (1.0 - math.exp(-t["k"]))),
                 _sat((b - t.get("x0", 0.0)) * t.get("ix", 1.0), t["k"], 1.0 / (1.0 - math.exp(-t["k"])))]
        elif k == "threshold":
            f = [0.0, 1.0]
        elif k == "invu":
            f = [_invu(a, t["peak"], 1.0 / t["width"]), _invu(b, t["peak"], 1.0 / t["width"])]
            if a <= t["peak"] <= b:
                f.append(1.0)
        else:                                                  # product
            a2, b2 = self.range[t["src2"]]
            f = [a * a2, a * b2, b * a2, b * b2]
        vals = [w * x for w in ws for x in f]
        if "gate" in t:
            vals.append(0.0)
        return min(vals), max(vals)

    # ---- edges -----------------------------------------------------------------------------------
    def _kind(self) -> str:
        ks = sorted(self.r["kinds"])
        return _pick(self.rng, ks, [self.r["kinds"][k] for k in ks])

    def edge(self, src: str, kind: str | None = None, scale: float = 1.0, sign: int | None = None) -> dict:
        """A term reading `src`. Functional parameters are placed inside the source's reachable range, so a
        threshold can actually be crossed and a peak actually reached."""
        rng = self.rng
        kind = kind or self._kind()
        if self.by_id[src]["role"] == "exogenous":
            kind = "linear"
        w = _u(rng, 0.7, 1.0) * scale
        if sign is None:
            sign = -1 if rng.random() < self.r["neg_edge_p"] else 1
        t: dict = {"kind": kind, "src": src, "lag": 0, "w": w * sign}
        a, b = self.range[src]
        span = max(b - a, 1e-3)
        if kind == "saturating":
            t["k"] = _u(rng, 2.5, 6.0)
            if self.by_id[src]["role"] != "lever":
                t["x0"], t["ix"] = a, 1.0 / span
        elif kind == "threshold":
            t["theta"] = a + _u(rng, 0.35, 0.75) * span
        elif kind == "invu":
            t["peak"] = a + _u(rng, 0.3, 0.7) * span
            t["width"] = _u(rng, 0.35, 0.5) * span
        elif kind == "delay":
            t["lag"] = int(rng.integers(2, 6))
        return t

    def to_outcome(self, seat: int, src: str, w: float) -> dict:
        t = {"kind": "linear", "src": src, "lag": 0, "w": float(w)}
        self.out_terms.setdefault(seat, []).append(t)
        return t

    def _omega(self, lo: float = 0.3, hi: float = 0.6) -> float:
        return _u(self.rng, lo, hi)

    def _cost(self, levers: list[str], om: float, kind: str) -> None:
        for lv in levers:
            if self.rng.random() < self.r["free_p"]:
                continue
            lo, hi = PHI[kind]
            self.by_id[lv]["c"] += _u(self.rng, lo, hi) * abs(om)

    def _driver(self, seat: int, src: str | None) -> tuple[str, list[str]]:
        """The node that drives a block: a given latent (series composition), an existing lever of this seat
        (shared lever: the lever then feeds several latents), or a new lever."""
        if src is not None:
            return src, []
        mine = [x["id"] for x in self.levers if x["seat"] == seat and x.get("active")]
        if len(mine) >= 2 and self.rng.random() < self.r["share_p"]:
            return _pick(self.rng, mine), []
        lv = self.lever(seat)
        self.by_id[lv]["active"] = True
        return lv, [lv]

    def _pol(self, *terms: dict) -> int:
        p = 1
        for t in terms:
            p *= 1 if t["w"] > 0 else -1
        return p

    def _register(self, seat: int, node: str, term: dict | None, pol: int) -> None:
        self.blocks.setdefault(seat, []).append({"node": node, "term": term, "pol": pol})

    # ---- motif blocks ----------------------------------------------------------------------------
    def blk_direct(self, seat, src=None, harmful=False):
        s, new = self._driver(seat, src)
        e = self.edge(s)
        a = self.latent(seat, [e])
        om = self._omega()
        if harmful:                      # a lever that only hurts: free to move, best left at its low end
            t = self.to_outcome(seat, a, -om * _u(self.rng, 0.5, 1.0) * self._pol(e))
        else:
            t = self.to_outcome(seat, a, om * self._pol(e))
            self._cost(new, om, e["kind"])
        self._register(seat, a, t, self._pol(e))

    def blk_chain(self, seat, src=None):
        s, new = self._driver(seat, src)
        e1 = self.edge(s)
        a = self.latent(seat, [e1])
        e2 = self.edge(a)
        b = self.latent(seat, [e2])
        es = [e1, e2]
        if self.rng.random() < 0.3:
            e3 = self.edge(b)
            b = self.latent(seat, [e3])
            es.append(e3)
        om = self._omega()
        t = self.to_outcome(seat, b, om * self._pol(*es))
        self._cost(new, om, e1["kind"])
        self._register(seat, b, t, self._pol(*es))

    def blk_fork(self, seat, src=None):
        s, new = self._driver(seat, src)
        e1, e2 = self.edge(s), self.edge(s)
        a, b = self.latent(seat, [e1]), self.latent(seat, [e2])
        om = self._omega()
        t = self.to_outcome(seat, a, om * self._pol(e1))
        mode = _pick(self.rng, ["oppose", "gauge_only", "assist"], [5, 3, 2])
        if mode == "oppose":             # the same lever helps along one branch and hurts along the other
            self.to_outcome(seat, b, -om * _u(self.rng, 0.4, 0.9) * self._pol(e2))
        elif mode == "assist":
            self.to_outcome(seat, b, om * _u(self.rng, 0.3, 0.6) * self._pol(e2))
        for lv in new:
            if self.rng.random() >= self.r["free_p"]:
                self.by_id[lv]["c"] += _u(self.rng, 0.05, 0.3) * om
        self.force_gauge += [a, b]
        self._register(seat, a, t, self._pol(e1))

    def blk_collider(self, seat, src=None):
        s1, new = self._driver(seat, src)
        if self.rng.random() < 0.7:
            s2 = self.lever(seat)
            self.by_id[s2]["active"] = True
            new = new + [s2]
            e2 = self.edge(s2, scale=0.6, sign=1)
        else:                            # one of the two causes is a hidden driver
            s2 = self.exo()
            e2 = {"kind": "linear", "src": s2, "lag": 0, "w": _u(self.rng, 0.05, 0.12)}
        e1 = self.edge(s1, scale=0.6, sign=1)
        c = self.latent(seat, [e1, e2])
        om = self._omega()
        t = self.to_outcome(seat, c, om)
        for lv in new:
            if self.rng.random() >= self.r["free_p"]:
                self.by_id[lv]["c"] += _u(self.rng, *PHI[e1["kind"]]) * om * 0.6
        self.force_gauge.append(c)
        self._register(seat, c, t, 1)

    def blk_mediator(self, seat, src=None):
        s, new = self._driver(seat, src)
        e1 = self.edge(s, sign=1)
        m = self.latent(seat, [e1])
        e2 = self.edge(m, sign=1)
        direct = {"kind": "linear", "src": s, "lag": 0, "w": -_u(self.rng, 0.25, 0.5)}     # opposes the indirect path
        o = self.latent(seat, [e2, direct])
        om = self._omega()
        t = self.to_outcome(seat, o, om)
        for lv in new:
            if self.rng.random() >= self.r["free_p"]:
                self.by_id[lv]["c"] += _u(self.rng, 0.0, 0.3) * om
        self.force_gauge.append(m)
        self._register(seat, o, t, 1)

    def blk_gate(self, seat, src=None):
        rng = self.rng
        s, new = self._driver(seat, src)
        levers = [x["id"] for x in self.levers if x["seat"] == seat and x["id"] != s and x.get("active")]
        lats = [x["id"] for x in self.lats if x["seat"] == seat and x["id"] != s
                and self.range[x["id"]][1] - self.range[x["id"]][0] >= 0.4]
        mode = _pick(rng, ["new", "lever", "latent"], [55, 25 if levers else 0, 20 if lats else 0])
        om = self._omega(0.4, 0.7)
        if mode == "new":
            g = self.lever(seat)
            self.by_id[g]["active"] = True
            self.by_id[g]["c"] += _u(rng, 0.15, 0.4) * om      # holding the gate open costs money
        else:
            g = _pick(rng, levers if mode == "lever" else lats)
        a, b = self.range[g]
        e = self.edge(s, kind=_pick(rng, ["linear", "saturating"]))
        e["gate"] = {"node": g, "level": a + _u(rng, 0.35, 0.7) * (b - a), "lag": 0}
        node = self.latent(seat, [e])
        t = self.to_outcome(seat, node, om * self._pol(e))
        self._cost(new, om, e["kind"])
        self.force_gauge.append(node)
        self._register(seat, node, t, self._pol(e))

    def _single(self, seat, src, kind, decay=None, om_range=(0.3, 0.6)):
        s, new = self._driver(seat, src)
        e = self.edge(s, kind=kind)
        a = self.latent(seat, [e], decay=decay)
        om = self._omega(*om_range)
        t = self.to_outcome(seat, a, om * self._pol(e))
        self._cost(new, om, e["kind"])
        self.force_gauge.append(a)
        self._register(seat, a, t, self._pol(e))
        return s, new, e

    def blk_threshold(self, seat, src=None):
        self._single(seat, src, "threshold")

    def blk_delay(self, seat, src=None):
        self._single(seat, src, "delay")

    def blk_stock(self, seat, src=None):
        self._single(seat, src, None, decay=_u(self.rng, 0.08, 0.3))

    def blk_invu(self, seat, src=None):
        s, new, e = self._single(seat, src, "invu")
        for lv in new:                   # the default must not already sit on the peak
            self.by_id[lv]["far_from"] = e["peak"]

    def blk_interaction(self, seat, src=None):
        rng = self.rng
        s1, new = self._driver(seat, src)
        others = [x["id"] for x in self.levers if x["seat"] == seat and x["id"] != s1 and x.get("active")]
        if others and rng.random() < 0.3:
            s2 = _pick(rng, others)
        else:
            s2 = self.lever(seat)
            self.by_id[s2]["active"] = True
            new = new + [s2]
        e = {"kind": "product", "src": s1, "src2": s2, "lag": 0, "lag2": 0, "w": _u(rng, 0.8, 1.1)}
        a = self.latent(seat, [e])
        om = self._omega(0.4, 0.7)
        t = self.to_outcome(seat, a, om)
        self._cost(new, om, "product")
        self.force_gauge.append(a)
        self._register(seat, a, t, 1)

    def blk_feedback(self, seat, src=None):
        """lever -> A -> B, and B at an earlier tick feeds back into A (loop gain below 1, so one stable
        operating point per setting; the response to a change rings or builds up over several ticks)."""
        rng = self.rng
        s, new = self._driver(seat, src)
        e1 = self.edge(s, sign=1)
        a = self.latent(seat, [e1])
        e2 = {"kind": "linear", "src": a, "lag": 0, "w": _u(rng, 0.7, 1.0)}
        b = self.latent(seat, [e2])
        phi = _u(rng, 0.3, 0.55) * (-1 if rng.random() < 0.6 else 1)
        self.by_id[a]["terms"].append({"kind": "linear", "src": b, "lag": int(rng.integers(1, 3)), "w": phi})
        om = self._omega()
        t = self.to_outcome(seat, b, om)
        self._cost(new, om, e1["kind"])
        self.force_gauge += [a, b]
        self._register(seat, b, t, 1)

    def blk_confounder(self, seat, src=None):
        """A hidden driver U moves both a gauge and the outcome; a lever also moves that gauge. The gauge and the
        revenue are correlated in passive data, yet moving the gauge through the lever earns nothing."""
        rng = self.rng
        u = self.exo(rho=_u(rng, 0.8, 0.95), sd=1.0)
        lv = self.lever(seat)            # dedicated: never shared with a block that feeds the outcome
        sgn = -1 if rng.random() < 0.3 else 1
        g = self.latent(seat, [{"kind": "linear", "src": lv, "lag": 0, "w": _u(rng, 0.5, 0.9)},
                               {"kind": "linear", "src": u, "lag": 0, "w": sgn * _u(rng, 0.25, 0.4)}])
        self.to_outcome(seat, u, _u(rng, 0.03, 0.06))
        if rng.random() < 0.7:
            self.by_id[lv]["c"] += _u(rng, 0.1, 0.3)
        self.force_gauge.append(g)

    def blk_dormant(self, seat, src=None):
        """Regime families: a lever that is worthless (slightly harmful) before the switch and valuable after."""
        s, new = self._driver(seat, None)
        e = self.edge(s, sign=1)
        a = self.latent(seat, [e])
        om = self._omega(0.45, 0.7)
        t = self.to_outcome(seat, a, -_u(self.rng, 0.03, 0.12))
        t["w2"] = om
        self.regime_terms.append(t)
        self._cost(new, om * 0.6, e["kind"])
        self.force_gauge.append(a)

    # ---- one seat --------------------------------------------------------------------------------
    def build_seat(self, seat: int) -> None:
        rng, r = self.rng, self.r
        n = int(rng.integers(r["n_blocks"][0], r["n_blocks"][1] + 1))
        blocks = list(r["required"])
        pool = sorted(r["pool"])
        while len(blocks) < n:
            blocks.append(_pick(rng, pool, [r["pool"][b] for b in pool]))
        order = [blocks[i] for i in rng.permutation(len(blocks))]
        self.out_terms.setdefault(seat, [])
        for b in order:
            src = None
            mine = self.blocks.get(seat, [])
            if len(mine) >= 2 and b in SERIES_OK and rng.random() < r["series_p"]:
                up = _pick(rng, mine)                          # series composition: feed on an earlier block
                src = up["node"]
                if up["term"] is not None and rng.random() < 0.5 and "w2" not in up["term"]:
                    self.out_terms[seat] = [t for t in self.out_terms[seat] if t is not up["term"]]
                    up["term"] = None
            getattr(self, "blk_" + b)(seat, src)
        while sum(1 for x in self.levers if x["seat"] == seat and x.get("active")) < 3:
            self.blk_direct(seat, self.lever_active(seat))     # never fewer than three levers that matter
        if rng.random() < r["harmful_p"]:
            self.blk_direct(seat, None, harmful=True)
        for _ in range(int(rng.integers(r["distractors"][0], r["distractors"][1] + 1))):
            lv = self.lever(seat)
            if rng.random() < 0.75:
                self.by_id[lv]["c"] = _u(rng, 0.25, 0.6)
        if r["regime"]:
            self.blk_dormant(seat)
            live = [t for t in self.out_terms[seat] if "w2" not in t and t["w"] > 0
                    and self.by_id[t["src"]]["role"] == "latent"]
            if live:                                           # the best-paying path stops paying or turns harmful
                t = max(live, key=lambda x: x["w"])
                t["w2"] = t["w"] * _pick(rng, [-0.5, -0.25, 0.0, 0.1])
                self.regime_terms.append(t)

    def couple(self, n_seats: int) -> None:
        """Seats share the world: what one seat produces spills into the other's demand one tick later."""
        rng = self.rng
        for a in range(n_seats):
            for b in range(n_seats):
                mine = [x for x in self.blocks.get(a, []) if x["node"]]
                if a == b or not mine or rng.random() >= 0.8:
                    continue
                src = _pick(rng, mine)["node"]
                w = _u(rng, 0.1, 0.25) * (-1 if rng.random() < 0.7 else 1)
                self.out_terms[b].append({"kind": "linear", "src": src, "lag": 1, "w": w, "coupling": True})

    # ---- assemble --------------------------------------------------------------------------------
    def finish(self, n_seats: int, horizon: int) -> dict:
        rng, r = self.rng, self.r
        demand = self.exo()
        outs = []
        for seat in range(n_seats):
            terms = list(self.out_terms[seat])
            terms.append({"kind": "linear", "src": demand, "lag": 0, "w": _u(rng, 0.02, 0.05)})
            outs.append({"id": f"out_{seat}", "role": "outcome", "seat": seat, "bias": 0.0,
                         "noise": _u(rng, 0.01, 0.03), "decay": None, "terms": terms, "cap": 1e9})
        for lv in self.levers:                                 # defaults: costly levers rest low
            n_lv = int(lv["hi"] - lv["lo"]) if lv["integer"] else 20
            if lv["far_from"] is not None:
                d = _u(rng, 0.25, 0.4)
                u = lv["far_from"] - d if lv["far_from"] - d >= 0.0 else min(1.0, lv["far_from"] + d)
            elif lv["c"] > 0:
                u = _pick(rng, [0.0, 0.1, 0.2, 0.3], [5, 2, 2, 1])
            else:
                u = _pick(rng, [0.0, 0.5, 1.0, float(rng.integers(0, 21)) / 20.0], [4, 3, 1, 2])
            lv["u_def"] = round(u * n_lv) / n_lv
            lv.pop("far_from")
            lv.pop("active", None)
        # which latents are observed
        gauges = []
        forced = set(self.force_gauge)
        for lat in self.lats:
            if lat["id"] in forced or rng.random() < 0.7:
                lag = int(rng.integers(1, 4)) if r["gauge_lag_p"] and rng.random() < r["gauge_lag_p"] else 0
                a = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0)))) * (-1 if rng.random() < 0.2 else 1)
                gauges.append({"id": f"gau_{len(gauges):02d}", "seat": lat["seat"], "src": lat["id"], "lag": lag,
                               "a": a, "c": _u(rng, 0.0, 50.0), "noise": _u(rng, *r["gauge_noise"])})
        for seat in range(n_seats):                            # an unrelated reading (pure hidden noise)
            if rng.random() < 0.4 or sum(g["seat"] == seat for g in gauges) < 2:
                x = self.exo()
                gauges.append({"id": f"gau_{len(gauges):02d}", "seat": seat, "src": x, "lag": 0,
                               "a": _u(rng, 1.0, 8.0), "c": _u(rng, 0.0, 50.0), "noise": _u(rng, *r["gauge_noise"])})
        names = neutral_names(rng, len(self.levers) + len(gauges))
        perm = rng.permutation(len(names))
        pub = self.levers + gauges
        for i, x in enumerate(pub):
            x["public"] = names[int(perm[i])]
        seats = []
        for seat in range(n_seats):
            lv = [x["id"] for x in self.levers if x["seat"] == seat]
            gg = [g["id"] for g in gauges if g["seat"] == seat]
            seats.append({"levers": [lv[i] for i in rng.permutation(len(lv))],
                          "gauges": [gg[i] for i in rng.permutation(len(gg))], "outcome": f"out_{seat}"})
        nodes = self.levers + self.exos + self.lats + outs
        lags = [g["lag"] for g in gauges] + [1]
        for nd in self.lats + outs:
            for t in nd["terms"]:
                lags += [t.get("lag", 0), t.get("lag2", 0), t.get("gate", {}).get("lag", 0)]
        cfg = {"engine": "scm", "n_seats": n_seats, "horizon": int(horizon), "K": int(max(lags)), "nodes": nodes,
               "gauges": gauges, "seats": seats, "regime": None}
        if self.regime_terms:
            cfg["regime"] = {"at": int(round(_u(rng, 0.4, 0.6) * horizon))}
        return cfg


# ====================================================================== compilation
def _compile(cfg: dict, want_step: bool = True, clip_outcome: bool = True):
    """Generate the per-tick update (`step(h, u, z, t) -> (values, gauge readings)`) and the noise-free steady
    state map (`steady(u, r, p) -> values`, one fixed-point sweep for regime r from the previous iterate p) as
    straight-line Python. `clip_outcome=False` leaves the outcome unclipped (calibration works on the raw sum,
    before the outcome bias is known)."""
    nodes = cfg["nodes"]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    regime_at = (cfg.get("regime") or {}).get("at")
    rows: set[int] = {0}

    def src(i: int, lag: int, j: int, steady: bool) -> str:
        if steady:
            return f"v{i}" if i < j else f"p[{i}]"
        if lag == 0:
            if i >= j:
                raise ValueError("same-tick edge against the node order")
            return f"v{i}"
        rows.add(lag - 1)
        return f"h{lag - 1}[{i}]"

    def wexpr(t: dict, steady: bool) -> str:
        if "w2" in t:
            cond = "r == 0" if steady else f"t < {int(regime_at)}"
            return f"({float(t['w'])!r} if {cond} else {float(t['w2'])!r})"
        return repr(float(t["w"]))

    def term(t: dict, j: int, steady: bool) -> str:
        s = src(idx[t["src"]], int(t.get("lag", 0)), j, steady)
        k, w = t["kind"], wexpr(t, steady)
        if k in ("linear", "delay"):
            e = f"{w} * {s}"
        elif k == "saturating":
            z = f"({s} - {float(t['x0'])!r}) * {float(t['ix'])!r}" if "x0" in t else s
            e = f"{w} * _sat({z}, {float(t['k'])!r}, {1.0 / (1.0 - math.exp(-float(t['k'])))!r})"
        elif k == "threshold":
            e = f"({w} if {s} > {float(t['theta'])!r} else 0.0)"
        elif k == "invu":
            e = f"{w} * _invu({s}, {float(t['peak'])!r}, {1.0 / float(t['width'])!r})"
        elif k == "product":
            s2 = src(idx[t["src2"]], int(t.get("lag2", 0)), j, steady)
            e = f"{w} * {s} * {s2}"
        else:
            raise ValueError(f"unknown edge kind {k!r}")
        if "gate" in t:
            g = t["gate"]
            gs = src(idx[g["node"]], int(g.get("lag", 0)), j, steady)
            e = f"(({e}) if {gs} > {float(g['level'])!r} else 0.0)"
        return e

    def body(steady: bool) -> tuple[list[str], int]:
        out: list[str] = []
        nz = 0
        for j, nd in enumerate(nodes):
            role = nd["role"]
            if role == "lever":
                out.append(f"v{j} = u[{j}]")
            elif role == "exogenous":
                if steady:
                    out.append(f"v{j} = 0.0")
                else:
                    sd, rho = float(nd["sd"]), float(nd["rho"])
                    out.append(f"v{j} = {rho!r} * h0[{j}] + {sd * math.sqrt(1.0 - rho * rho)!r} * z[{nz}]")
                    out.append(f"if v{j} > {4.0 * sd!r}: v{j} = {4.0 * sd!r}")
                    out.append(f"elif v{j} < {-4.0 * sd!r}: v{j} = {-4.0 * sd!r}")
                    nz += 1
            else:
                inner = " + ".join([repr(float(nd["bias"]))] + [term(t, j, steady) for t in nd["terms"]])
                d = nd.get("decay")
                if steady:
                    out.append(f"v{j} = {inner}")
                else:
                    if d:
                        inner = f"{1.0 - float(d)!r} * h0[{j}] + {float(d)!r} * ({inner})"
                    out.append(f"v{j} = {inner} + {float(nd['noise'])!r} * z[{nz}]")
                    nz += 1
                if role == "outcome" and not clip_outcome:
                    continue
                lo, hi = (0.0, float(nd["cap"])) if role == "outcome" else (-CLIP, CLIP)
                out.append(f"if v{j} > {hi!r}: v{j} = {hi!r}")
                out.append(f"elif v{j} < {lo!r}: v{j} = {lo!r}")
        return out, nz

    ns = {"_sat": _sat, "_invu": _invu}
    vals = "[" + ", ".join(f"v{j}" for j in range(len(nodes))) + "]"
    lines, _ = body(True)
    code = "def _steady(u, r, p):\n" + "".join(f"    {ln}\n" for ln in lines) + f"    return {vals}\n"
    exec(compile(code, "<scm-steady>", "exec"), ns)                        # noqa: S102 - generated arithmetic only
    step_fn, nz = None, 0
    if want_step:
        lines, nz = body(False)
        for k, g in enumerate(cfg["gauges"]):
            s = src(idx[g["src"]], int(g["lag"]), len(nodes), False)
            a = float(g["a"])
            lines.append(f"g{k} = round({a!r} * {s} + {float(g['c'])!r} + {abs(a) * float(g['noise'])!r} * z[{nz}], 4)")
            nz += 1
        head = "".join(f"    h{k} = h[{k}]\n" for k in sorted(rows))
        gl = "[" + ", ".join(f"g{k}" for k in range(len(cfg["gauges"]))) + "]"
        code = "def _step(h, u, z, t):\n" + head + "".join(f"    {ln}\n" for ln in lines) + f"    return {vals}, {gl}\n"
        exec(compile(code, "<scm-step>", "exec"), ns)                      # noqa: S102
        step_fn = ns["_step"]
    feedback = any(idx[t["src"]] >= j or ("gate" in t and idx[t["gate"]["node"]] >= j)
                   for j, nd in enumerate(nodes) for t in nd.get("terms", []))
    return step_fn, ns["_steady"], {"nz": nz, "feedback": feedback, "idx": idx}


def _steady_values(steady_fn, n: int, u: list[float], r: int, feedback: bool) -> list[float]:
    p = [0.0] * n
    v = steady_fn(u, r, p)
    if feedback:
        for _ in range(80):
            p = v
            v = steady_fn(u, r, p)
            if max(abs(a - b) for a, b in zip(v, p)) < 1e-10:
                break
    return v


# ====================================================================== calibration
def _levels(lv: dict) -> list[float]:
    if lv["integer"]:
        n = int(lv["hi"] - lv["lo"])
        return [i / n for i in range(n + 1)]
    return [i / 10 for i in range(11)]


def _calibrate(rng: np.random.Generator, cfg: dict, diag: list | None = None) -> bool:
    """Set the outcome bias, demand cap, price, lever cost rates and overhead of every seat so that the economic
    properties in the module docstring hold on the noise-free steady state. Returns False if this structure
    cannot satisfy them (the caller redraws)."""
    nodes = cfg["nodes"]
    _, steady_fn, meta = _compile(cfg, want_step=False, clip_outcome=False)
    idx, fb = meta["idx"], meta["feedback"]
    n = len(nodes)
    levers = [nd for nd in nodes if nd["role"] == "lever"]
    regimes = [0, 1] if cfg["regime"] else [0]
    f1 = cfg["regime"]["at"] / cfg["horizon"] if cfg["regime"] else 1.0
    fr = [f1, 1.0 - f1] if cfg["regime"] else [1.0]
    udef = [lv["u_def"] for lv in levers]
    calib = []
    for seat, sd in enumerate(cfg["seats"]):
        mine = [i for i, lv in enumerate(levers) if lv["seat"] == seat]
        out = idx[sd["outcome"]]
        cost = [levers[i]["c"] for i in mine]
        d0 = [udef[i] for i in mine]
        cache: dict[tuple, list[float]] = {}

        def raw(us: tuple) -> list[float]:
            y = cache.get(us)
            if y is None:
                u = list(udef)
                for i, x in zip(mine, us):
                    u[i] = x
                y = cache[us] = [_steady_values(steady_fn, n, u, r, fb)[out] for r in regimes]
            return y

        y0 = raw(tuple(d0))
        target = _u(rng, 0.4, 0.6)
        bias = target - sum(f * y for f, y in zip(fr, y0))
        ydef = [max(0.0, y + bias) for y in y0]
        ybar = sum(f * y for f, y in zip(fr, ydef))
        grids = [_levels(levers[i]) for i in mine]
        pool = [tuple(d0), tuple(0.0 for _ in mine), tuple(1.0 for _ in mine)]
        # "uniformly random settings" as an operator produces them: a uniform draw over the range, which the
        # facility rounds to the nearest admissible value of an integer lever
        rand = [tuple(round(float(rng.random()) * (len(g) - 1)) / (len(g) - 1) if levers[i]["integer"]
                      else float(rng.random()) for i, g in zip(mine, grids)) for _ in range(256)]
        pool += rand
        kd = sum(c * d for c, d in zip(cost, d0))

        def net(us: tuple, lam: float, r: int | None = None) -> float:
            ys = raw(us)
            k = lam * (sum(c * x for c, x in zip(cost, us)) - kd)
            if r is not None:
                return max(0.0, ys[r] + bias) - ydef[r] - k
            return sum(f * (max(0.0, y + bias) - yd) for f, y, yd in zip(fr, ys, ydef)) - k

        def climb(start: tuple, lam: float, r: int | None) -> tuple[tuple, float]:
            best, val = start, net(start, lam, r)
            for sweep in range(3):
                moved = False
                for j, g in enumerate(grids):
                    cand = g if sweep < 2 or levers[mine[j]]["integer"] else \
                        [min(1.0, max(0.0, best[j] + d)) for d in (-0.06, -0.03, 0.03, 0.06)]
                    for x in cand:
                        us = best[:j] + (x,) + best[j + 1:]
                        v = net(us, lam, r)
                        if v > val + 1e-12:
                            best, val, moved = us, v, True
                if not moved and sweep >= 1:
                    break
            return best, val

        choice = None
        for lam in LAMBDAS:
            if ybar - lam * kd < 0.02:                          # overhead would be negative
                continue
            opts, g = [], 0.0
            for r, f in zip(regimes, fr):
                rr = r if cfg["regime"] else None
                starts = sorted(pool[:99], key=lambda us: -net(us, lam, rr))[:3]
                o, v = max((climb(s, lam, rr) for s in starts), key=lambda x: x[1])
                opts.append(o)
                g += f * v if cfg["regime"] else v
            r_mean = sum(net(us, lam) for us in rand) / len(rand)
            cmax, cmin = net(pool[2], lam), net(pool[1], lam)
            sens = 0                                            # levers that move the outcome and the net
            for j, gr in enumerate(grids):
                o, rr = opts[0], (0 if cfg["regime"] else None)
                base, yb = net(o, lam, rr), raw(o)[0]
                alt = [o[:j] + (x,) + o[j + 1:] for x in gr]
                if max(abs(net(a, lam, rr) - base) for a in alt) >= SENS_MIN * ybar and \
                        max(abs(raw(a)[0] - yb) for a in alt) >= SENS_MIN * ybar:
                    sens += 1
            ok = (g >= G_MIN * ybar and r_mean <= -R_MIN * ybar and cmax <= CORNER * g and cmin <= CORNER * g
                  and sens >= 2)
            if cfg["regime"] and ok:                            # the best settings before and after must differ
                a = net(opts[0], lam, 1) <= 0.6 * net(opts[1], lam, 1)
                b = net(opts[1], lam, 0) <= 0.6 * net(opts[0], lam, 0)
                ok = a and b
            if diag is not None:
                diag.append({"seat": seat, "lam": lam, "g": g / ybar, "rand": r_mean / ybar, "cmax": cmax / ybar,
                             "cmin": cmin / ybar, "sens": sens, "ok": ok})
            if ok:                                              # among feasible cost levels: a moderate loss at random
                score = -abs(r_mean / ybar + R_TARGET)
                if choice is None or score > choice[0]:
                    choice = (score, lam, opts, g, r_mean, cmax, cmin)
        if choice is None:
            return False
        _, lam, opts, g, r_mean, cmax, cmin = choice
        ymax = max(max(0.0, y + bias) for ys in cache.values() for y in ys)
        calib.append({"lam": lam, "bias": bias, "cap": 1.2 * ymax + 0.15, "ybar": ybar, "opts": opts, "g": g,
                      "rand": r_mean, "cmax": cmax, "cmin": cmin, "mine": mine})
    # commit: price, costs, overhead (in cents)
    income = 0
    for seat, (sd, cb) in enumerate(zip(cfg["seats"], calib)):
        out = nodes[idx[sd["outcome"]]]
        out["bias"], out["cap"] = cb["bias"], cb["cap"]
        p_y = _u(rng, 1500.0, 6000.0) / cb["ybar"]             # cents per unit of the outcome variable
        units = float(_pick(rng, [10.0, 50.0, 100.0, 250.0]))
        for i in cb["mine"]:
            levers[i]["cost"] = cb["lam"] * levers[i]["c"] * p_y
        overhead = p_y * cb["ybar"] - sum(levers[i]["cost"] * levers[i]["u_def"] for i in cb["mine"])
        sd.update({"units": units, "price": p_y / units, "overhead": overhead})
        income += int(math.ceil(p_y * cb["cap"])) + 1
        names = [levers[i]["public"] for i in cb["mine"]]
        vals = [[_public_value(levers[i], x) for i, x in zip(cb["mine"], o)] for o in cb["opts"]]
        sd["calib"] = {"lambda": cb["lam"], "optimum": [dict(zip(names, v)) for v in vals],
                       "net_opt": cb["g"] * p_y, "net_random": cb["rand"] * p_y, "net_max": cb["cmax"] * p_y,
                       "net_min": cb["cmin"] * p_y, "revenue_default": cb["ybar"] * p_y}
    for lv in levers:
        lv.pop("c", None)
        lv["default"] = _public_value(lv, lv["u_def"])
    cfg["income_cents_per_tick"] = income
    return True


def _public_value(lv: dict, u: float) -> float:
    x = lv["lo"] + u * (lv["hi"] - lv["lo"])
    return float(round(x)) if lv["integer"] else float(round(x, 4))


class _Mirror:
    """A generator whose normal draws are the negatives of another generator's (an antithetic twin)."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def standard_normal(self, n):
        return -self._rng.standard_normal(n)


def _settle_overhead(cfg: dict) -> None:
    """Replace the steady-state overhead by the one that makes default settings break even under the real noisy
    dynamics (noise passing through thresholds, gates and clips shifts mean revenue a little). Antithetic pairs
    of runs cancel the part of the noise that enters linearly, which is most of it."""
    eng = ScmEngine(cfg)
    dflt = [eng.defaults(s) for s in range(eng.n_seats)]
    tot = [0.0] * eng.n_seats
    pairs, ticks = 4, int(cfg["horizon"])
    for k in range(pairs):
        for mirror in (False, True):
            rng = np.random.default_rng([977, k, int(cfg["horizon"]), len(cfg["nodes"])])
            rng = _Mirror(rng) if mirror else rng
            st = eng.init_state(rng)
            for t in range(ticks):
                res = eng.step(st, dflt, rng, t)
                st = res.state
                for s in range(eng.n_seats):
                    tot[s] += res.revenue[s] - res.cost[s]
    for s, sd in enumerate(cfg["seats"]):
        sd["overhead"] = max(0.0, sd["overhead"] + tot[s] / (2 * pairs * ticks))


def _recipe(params: dict) -> dict:
    r = {"required": [], "pool": {"direct": 1}, "n_blocks": [3, 5], "kinds": {"linear": 4, "saturating": 6},
         "neg_edge_p": 0.2, "share_p": 0.25, "series_p": 0.3, "free_p": 0.15, "harmful_p": 0.5,
         "distractors": [1, 2], "gauge_lag_p": 0.0, "regime": False, "latent_noise": [0.005, 0.03],
         "gauge_noise": [0.005, 0.04], "horizon": 96, "seats": 1}
    r.update({k: v for k, v in params.items() if k in r})
    return r


# ====================================================================== the engine
@register_engine
class ScmEngine(Engine):
    engine = "scm"
    version = "1"
    state_in_event = True                      # the state is one small array: events carry it
    claims = ("default_zero", "random_loses", "corners_bad")    # economic properties validity must confirm

    def __init__(self, config: dict):
        super().__init__(config)
        self._step_fn, self._steady_fn, meta = _compile(config)
        self._nz, self._feedback, self._idx = meta["nz"], meta["feedback"], meta["idx"]
        nodes = config["nodes"]
        self._n, self._k = len(nodes), int(config["K"])
        by = {nd["id"]: nd for nd in nodes}
        gby = {g["id"]: (k, g) for k, g in enumerate(config["gauges"])}
        self._lev = [(i, nd["seat"], nd["public"], float(nd["lo"]), 1.0 / (float(nd["hi"]) - float(nd["lo"])),
                      float(nd.get("cost", 0.0)), float(nd["default"]))
                     for i, nd in enumerate(nodes) if nd["role"] == "lever"]
        self._levers = [[Lever(by[i]["public"], float(by[i]["lo"]), float(by[i]["hi"]), float(by[i]["default"]),
                               bool(by[i]["integer"])) for i in sd["levers"]] for sd in config["seats"]]
        self._gauges = [[Gauge(gby[i][1]["public"]) for i in sd["gauges"]] for sd in config["seats"]]
        self._gread = [[(gby[i][1]["public"], gby[i][0]) for i in sd["gauges"]] for sd in config["seats"]]
        self._seat = [(self._idx[sd["outcome"]], float(sd.get("units", 1.0)), float(sd.get("price", 0.0)),
                       float(sd.get("overhead", 0.0))) for sd in config["seats"]]
        self._regime_at = (config.get("regime") or {}).get("at")

    def __reduce__(self):
        return (ScmEngine, (self.config,))

    # ---- construction ------------------------------------------------------------------------------
    @classmethod
    def generate(cls, rng: np.random.Generator, params: dict) -> dict:
        r = _recipe(params)
        n_seats = int(r["seats"])
        cfg, ok, attempt = None, False, 0
        for attempt in range(MAX_ATTEMPTS):
            d = _Draft(rng, r)
            for seat in range(n_seats):
                d.build_seat(seat)
            if n_seats > 1:
                d.couple(n_seats)
            cfg = d.finish(n_seats, r["horizon"])
            ok = _calibrate(rng, cfg)
            if ok:
                break
        if not ok:                             # leave a runnable but uncalibrated mechanism; validity rejects it
            for nd in cfg["nodes"]:
                if nd["role"] == "lever":
                    nd["default"] = _public_value(nd, nd["u_def"])
                    nd.pop("c", None)
            for sd in cfg["seats"]:
                sd.update({"units": 100.0, "price": 30.0, "overhead": 0.0})
            cfg["income_cents_per_tick"] = 1_000_000
        cfg["calibrated"] = bool(ok)
        cfg["attempts"] = attempt + 1
        cfg["motifs"] = _motifs(cfg)
        if ok:
            _settle_overhead(cfg)
        return cfg

    # ---- public surface ----------------------------------------------------------------------------
    def levers(self, seat: int) -> list[Lever]:
        return self._levers[seat]

    def gauges(self, seat: int) -> list[Gauge]:
        return self._gauges[seat]

    def brief(self, seat: int) -> str:
        return ("You operate a production facility. Every tick it turns out some quantity of output and you are "
                "paid a fixed price per unit of output sold; running it costs money, and the cost depends on your "
                "settings. Your objective is net income (revenue minus cost) accumulated over time. Each tick's "
                "report gives your readings, the output sold (score), revenue and cost.")

    # ---- dynamics ----------------------------------------------------------------------------------
    def _u_of(self, settings: list[dict[str, float]]) -> list[float]:
        u = []
        for _, seat, name, lo, inv, _, dflt in self._lev:
            x = (settings[seat].get(name, dflt) - lo) * inv
            u.append(0.0 if x < 0.0 else (1.0 if x > 1.0 else x))
        return u

    def steady(self, settings: list[dict[str, float]], regime: int = 0) -> list[float]:
        """Noise-free steady-state value of every node under constant settings (evaluator / generator use)."""
        return _steady_values(self._steady_fn, self._n, self._u_of(settings), regime, self._feedback)

    def init_state(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        v = self.steady([self.defaults(s) for s in range(self.n_seats)], 0)
        z = rng.standard_normal(self._n)
        for i, nd in enumerate(self.config["nodes"]):
            if nd["role"] == "exogenous":
                v[i] = float(nd["sd"]) * float(z[i])
        return {"h": np.tile(np.asarray(v, dtype=np.float64), (self._k, 1))}

    def step(self, state, settings, rng, t):
        h = state["h"]
        u = self._u_of(settings)
        vals, g = self._step_fn(h.tolist(), u, rng.standard_normal(self._nz).tolist(), t)
        new = np.empty_like(h)
        new[1:] = h[:-1]
        new[0] = vals
        cost = [sd[3] for sd in self._seat]
        for (i, seat, _, _, _, c, _), x in zip(self._lev, u):
            if c:
                cost[seat] += c * x
        readings, revenue, score = [], [], []
        for s, (out, units, price, _) in enumerate(self._seat):
            q = round(units * vals[out], 4)
            score.append(q)
            revenue.append(float(int(price * q + 0.5)))
            cost[s] = float(int(cost[s] + 0.5))
            readings.append({name: g[k] for name, k in self._gread[s]})
        regime = int(self._regime_at is not None and t >= self._regime_at)
        return StepResult({"h": new}, readings, revenue, cost, score, {"regime": regime})

    # ---- private (evaluator only) ------------------------------------------------------------------
    def truth(self) -> dict:
        cfg = self.config
        nodes = [{"id": nd["id"], "role": nd["role"], "seat": nd["seat"], "public": nd.get("public")}
                 for nd in cfg["nodes"]]
        nodes += [{"id": g["id"], "role": "gauge", "seat": g["seat"], "public": g["public"]} for g in cfg["gauges"]]
        edges, changes = [], []
        for nd in cfg["nodes"]:
            for t in nd.get("terms", []):
                sgn = 1 if t["w"] > 0 else (-1 if t["w"] < 0 else 0)
                kind = {"invu": "inverted_u"}.get(t["kind"], t["kind"])
                e = {"src": t["src"], "dst": nd["id"], "sign": 0 if t["kind"] == "invu" else sgn,
                     "lag": int(t.get("lag", 0)), "kind": kind, "w": t["w"]}
                if nd.get("decay"):
                    e.update({"kind": "stock", "form": kind, "decay": nd["decay"]})
                if "gate" in t:
                    e.update({"sign": 0, "kind": "gate", "form": kind, "open_sign": sgn, "gate": t["gate"]["node"],
                              "level": t["gate"]["level"]})
                    edges.append({"src": t["gate"]["node"], "dst": nd["id"], "sign": sgn,
                                  "lag": int(t["gate"].get("lag", 0)), "kind": "gate_control"})
                if t["kind"] == "invu":
                    e["peak"] = t["peak"]
                if "w2" in t:
                    e["w_after"] = t["w2"]
                    e["sign_after"] = 1 if t["w2"] > 0 else (-1 if t["w2"] < 0 else 0)
                    changes.append({"src": t["src"], "dst": nd["id"], "w_before": t["w"], "w_after": t["w2"]})
                edges.append(e)
                if t["kind"] == "product":
                    edges.append({"src": t["src2"], "dst": nd["id"], "sign": sgn, "lag": int(t.get("lag2", 0)),
                                  "kind": "product", "w": t["w"]})
        for g in cfg["gauges"]:
            edges.append({"src": g["src"], "dst": g["id"], "sign": 1 if g["a"] > 0 else -1, "lag": int(g["lag"]),
                          "kind": "observe"})
        regimes = []
        if cfg.get("regime"):
            what = "; ".join(f"{c['src']}->{c['dst']} weight {c['w_before']:.3f} -> {c['w_after']:.3f}"
                             for c in changes)
            regimes.append({"at": int(cfg["regime"]["at"]), "what": what, "changes": changes})
        econ = [{"seat": s, "price": sd.get("price"), "units_per_y": sd.get("units"), "overhead": sd.get("overhead"),
                 "cost_rates": {nd["public"]: nd.get("cost", 0.0) for nd in cfg["nodes"]
                                if nd["role"] == "lever" and nd["seat"] == s},
                 "calibration": sd.get("calib")} for s, sd in enumerate(cfg["seats"])]
        return {"engine": "scm", "family": cfg.get("family"), "nodes": nodes, "edges": edges, "regimes": regimes,
                "motifs": list(cfg.get("motifs", [])), "economics": econ,
                "notes": "lever values act through u = (x - min) / (max - min); stock nodes relax to bias + inflow "
                         "at rate `decay`; sign 0 = non-monotone or gated."}

    def constructive_optimum(self, seat: int) -> dict | None:
        """The generator's own best settings (noise-free steady state): one entry per regime."""
        cal = self.config["seats"][seat].get("calib")
        if not cal:
            return None
        at = [0] + ([int(self._regime_at)] if self._regime_at is not None else [])
        return {"schedule": [{"from": a, "settings": dict(o)} for a, o in zip(at, cal["optimum"])],
                "expected_net": cal["net_opt"]}


def handmade(levers: list[dict], latents: list[dict], outcome: dict, gauges: list[dict] | None = None,
             exogenous: list[dict] | None = None, price: float = 10.0, units: float = 100.0, overhead: float = 0.0,
             horizon: int = 60, regime_at: int | None = None, income_cents_per_tick: int | None = None) -> dict:
    """A one-seat mechanism written out by hand: known ground truth for tests of evaluators and learners.

        levers     [{"name", "lo"=0, "hi"=1, "default"=lo, "integer"=False, "cost"=0 (cents per tick at full level)}]
        exogenous  [{"id", "rho", "sd"}]
        latents    [{"id", "terms": [...], "bias"=0, "noise"=0, "decay"=None}]   in dependency order
        outcome    {"terms": [...], "bias"=0, "noise"=0, "cap"=10}               (revenue = price x units x value)
        gauges     [{"name", "src", "lag"=0, "a"=1, "c"=0, "noise"=0}]           (default: one per latent)

    A term is `{"kind": linear | saturating (k) | threshold (theta) | invu (peak, width) | product (src2) |
    delay (lag), "src": lever name or node id, "w": weight, "lag"=0, "w2": weight after `regime_at`,
    "gate": {"node", "level"}}`. Returns the same structure as `make_mechanism`."""
    names = {lv["name"]: f"lev_{i:02d}" for i, lv in enumerate(levers)}

    def ref(x: str) -> str:
        return names.get(x, x)

    def terms(ts: list[dict]) -> list[dict]:
        out = []
        for t in ts:
            t = dict(t, src=ref(t["src"]), lag=int(t.get("lag", 0)), w=float(t["w"]))
            if "src2" in t:
                t["src2"] = ref(t["src2"])
            if "gate" in t:
                t["gate"] = dict(t["gate"], node=ref(t["gate"]["node"]), lag=int(t["gate"].get("lag", 0)))
            out.append(t)
        return out

    nodes: list[dict] = []
    for i, lv in enumerate(levers):
        lo, hi = float(lv.get("lo", 0.0)), float(lv.get("hi", 1.0))
        nodes.append({"id": names[lv["name"]], "role": "lever", "seat": 0, "public": lv["name"], "lo": lo, "hi": hi,
                      "integer": bool(lv.get("integer", False)), "default": float(lv.get("default", lo)),
                      "u_def": (float(lv.get("default", lo)) - lo) / (hi - lo), "cost": float(lv.get("cost", 0.0))})
    for x in exogenous or []:
        nodes.append({"id": x["id"], "role": "exogenous", "seat": None, "rho": float(x.get("rho", 0.8)),
                      "sd": float(x.get("sd", 1.0))})
    for lt in latents:
        nodes.append({"id": lt["id"], "role": "latent", "seat": 0, "bias": float(lt.get("bias", 0.0)),
                      "noise": float(lt.get("noise", 0.0)), "decay": lt.get("decay"), "terms": terms(lt["terms"])})
    nodes.append({"id": "out_0", "role": "outcome", "seat": 0, "bias": float(outcome.get("bias", 0.0)),
                  "noise": float(outcome.get("noise", 0.0)), "decay": None, "terms": terms(outcome["terms"]),
                  "cap": float(outcome.get("cap", 10.0))})
    if gauges is None:
        gauges = [{"name": f"g_{lt['id']}", "src": lt["id"]} for lt in latents]
    gs = [{"id": f"gau_{i:02d}", "seat": 0, "public": g["name"], "src": ref(g["src"]), "lag": int(g.get("lag", 0)),
           "a": float(g.get("a", 1.0)), "c": float(g.get("c", 0.0)), "noise": float(g.get("noise", 0.0))}
          for i, g in enumerate(gauges)]
    lags = [1] + [g["lag"] for g in gs]
    for nd in nodes:
        for t in nd.get("terms", []):
            lags += [t["lag"], int(t.get("lag2", 0)), t.get("gate", {}).get("lag", 0)]
    cap_cents = price * units * float(outcome.get("cap", 10.0))
    cfg = {"engine": "scm", "n_seats": 1, "horizon": int(horizon), "K": int(max(lags)), "nodes": nodes, "gauges": gs,
           "seats": [{"levers": [n["id"] for n in nodes if n["role"] == "lever"], "gauges": [g["id"] for g in gs],
                      "outcome": "out_0", "units": float(units), "price": float(price), "overhead": float(overhead)}],
           "regime": {"at": int(regime_at)} if regime_at is not None else None, "calibrated": False, "attempts": 0,
           "family": None,
           "income_cents_per_tick": int(income_cents_per_tick if income_cents_per_tick is not None
                                        else math.ceil(cap_cents) + 1)}
    cfg["motifs"] = _motifs(cfg)
    return {"engine": "scm", "family": None, "seed": 0, "config": cfg}


# ====================================================================== motif census
def _motifs(cfg: dict) -> list[str]:
    """Which causal motifs this draw really contains, read off the graph (not off the recipe)."""
    nodes = {nd["id"]: nd for nd in cfg["nodes"]}
    kids: dict[str, set[str]] = {i: set() for i in nodes}
    par0: dict[str, set[str]] = {i: set() for i in nodes}       # same-tick, non-gate parents (one per term)
    m: set[str] = set()
    for nd in cfg["nodes"]:
        if nd.get("decay"):
            m.add("stock")
        for t in nd.get("terms", []):
            srcs = [t["src"]] + ([t["src2"]] if t["kind"] == "product" else [])
            for s in srcs:
                kids[s].add(nd["id"])
            if int(t.get("lag", 0)) == 0 and t["kind"] != "product":
                par0[nd["id"]].add(t["src"])
            if "gate" in t:
                kids[t["gate"]["node"]].add(nd["id"])
                m.add("gate")
            if "w2" in t:
                m.add("regime")
            if t.get("coupling"):
                m.add("coupling")
            m.add({"saturating": "saturation", "threshold": "threshold", "invu": "invu", "product": "interaction",
                   "delay": "delay", "linear": ""}[t["kind"]])
    m.discard("")

    def desc(i: str) -> set[str]:
        seen, stack = set(), [i]
        while stack:
            for k in kids[stack.pop()]:
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        return seen

    d = {i: desc(i) for i in nodes}
    outs = {i for i, nd in nodes.items() if nd["role"] == "outcome"}
    gauged = {g["src"] for g in cfg["gauges"]}
    for i, nd in nodes.items():
        role = nd["role"]
        if role == "lever":
            m.add("costly" if nd.get("cost", nd.get("c", 0.0)) > 0 else "")
            if not kids[i]:
                m.add("distractor")
            if any(nodes[a]["role"] == "latent" and any(nodes[b]["role"] == "latent" for b in kids[a])
                   for a in kids[i]):
                m.add("chain")
        if role in ("lever", "latent") and len(kids[i]) >= 2:
            m.add("fork")
        if role == "latent":
            ps = sorted(par0[i])
            if any(b not in d[a] and a not in d[b] for x, a in enumerate(ps) for b in ps[x + 1:]):
                m.add("collider")
            if i in d[i]:
                m.add("feedback")
        if role in ("lever", "latent"):
            if any(z in kids[i] and z in d[mid] for mid in kids[i] for z in kids[mid] if mid != z):
                m.add("mediator")
        if role == "exogenous" and (kids[i] & outs or any(d[k] & outs for k in kids[i])):
            if any(k in gauged and not (d[k] & outs) and k not in outs for k in kids[i]):
                m.add("confounder")
    m.discard("")
    if any(int(g["lag"]) > 0 for g in cfg["gauges"]):
        m.add("lagged_gauge")
    return sorted(m)


# ====================================================================== families
_COMMON = ("chain", "fork", "collider", "mediator", "saturation", "distractor", "costly")


@mech_family("scm_additive", "scm", "train", "base",
             "Additive structure: chains, forks, colliders and mediators with linear and saturating responses, "
             "distractor levers, costly levers. No dynamics: the response to a setting is immediate.",
             motifs=_COMMON)
def fam_additive(rng, o):
    return {"required": ["chain", "fork"], "pool": {"direct": 3, "chain": 2, "fork": 2, "collider": 2, "mediator": 2},
            "n_blocks": [3, 5], "horizon": 96, "seats": int(o.get("seats", 1))}


@mech_family("scm_delay", "scm", "train", "base",
             "Dynamics: pure delays of several ticks, stocks that accumulate and decay, lagged gauges.",
             motifs=_COMMON + ("delay", "stock", "lagged_gauge"))
def fam_delay(rng, o):
    return {"required": ["delay", "stock"], "pool": {"delay": 2, "stock": 2, "direct": 2, "chain": 1},
            "n_blocks": [3, 4], "gauge_lag_p": 0.35, "horizon": 144, "seats": int(o.get("seats", 1))}


@mech_family("scm_gate", "scm", "train", "base",
             "Conditional structure: gated edges, thresholds, product interactions. No dynamics.",
             motifs=_COMMON + ("gate", "threshold", "interaction"))
def fam_gate(rng, o):
    second = "threshold" if rng.random() < 0.5 else "interaction"
    return {"required": ["gate", second], "pool": {"gate": 2, "threshold": 2, "interaction": 2, "direct": 2, "fork": 1},
            "n_blocks": [3, 5], "horizon": 96, "seats": int(o.get("seats", 1))}


@mech_family("scm_delay_gate", "scm", "val", "compositional",
             "Delays together with gates and thresholds: every motif occurs in training, never in this combination.",
             motifs=_COMMON + ("delay", "gate", "threshold", "lagged_gauge"))
def fam_delay_gate(rng, o):
    return {"required": ["delay", "gate"], "pool": {"delay": 2, "gate": 2, "threshold": 1, "direct": 2, "chain": 1},
            "n_blocks": [3, 5], "gauge_lag_p": 0.25, "horizon": 144, "seats": int(o.get("seats", 1))}


@mech_family("scm_invu", "scm", "val", "structural",
             "Inverted-U responses (too much is as bad as too little): a functional form absent from training.",
             motifs=_COMMON + ("invu",))
def fam_invu(rng, o):
    return {"required": ["invu", "invu"], "pool": {"invu": 2, "direct": 2, "chain": 1, "fork": 1, "collider": 1},
            "n_blocks": [3, 5], "harmful_p": 0.8, "distractors": [1, 3], "horizon": 96,
            "seats": int(o.get("seats", 1))}


@mech_family("scm_stock_gate", "scm", "test", "compositional",
             "Stocks together with gates, thresholds and interactions: training motifs in an unseen combination.",
             motifs=_COMMON + ("stock", "gate", "threshold", "interaction", "lagged_gauge"))
def fam_stock_gate(rng, o):
    return {"required": ["stock", "gate"], "pool": {"stock": 2, "gate": 2, "threshold": 1, "interaction": 1,
                                                     "direct": 2, "fork": 1},
            "n_blocks": [3, 5], "gauge_lag_p": 0.25, "horizon": 144, "seats": int(o.get("seats", 1))}


@mech_family("scm_feedback", "scm", "test", "structural",
             "Feedback loops through a lag and hidden confounders: passive correlation misleads, only intervention "
             "reveals what causes revenue. Both motifs are absent from training.",
             motifs=_COMMON + ("feedback", "confounder", "lagged_gauge"))
def fam_feedback(rng, o):
    return {"required": ["feedback", "confounder"], "pool": {"feedback": 1, "direct": 3, "chain": 2, "fork": 1,
                                                             "mediator": 1},
            "n_blocks": [3, 5], "gauge_lag_p": 0.2, "horizon": 144, "seats": int(o.get("seats", 1))}


@mech_family("scm_regime", "scm", "test", "structural",
             "A regime switch at a hidden tick: some causal weights change strength or sign mid-episode, so settings "
             "that were the best stop being the best.",
             motifs=_COMMON + ("regime",))
def fam_regime(rng, o):
    return {"required": ["direct", "chain"], "pool": {"direct": 3, "chain": 2, "fork": 2, "collider": 1, "mediator": 1},
            "n_blocks": [3, 4], "regime": True, "horizon": 192, "seats": int(o.get("seats", 1))}
