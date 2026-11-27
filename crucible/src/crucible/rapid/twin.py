"""An independent scalar interpreter of `primitive_graph` mechanisms (MASTER_SPEC Part 8.12: the fast simulator and
a second representation must be demonstrably consistent where they are claimed to represent the same mechanism).

`ScalarGraph(config)` re-implements, from the documented contracts alone, what a composed causal graph does in one
tick: node values in evaluation order from weighted, lagged input ports; levers (set, default or the natural
regime); discrete decisions; each primitive's documented formula written again here in plain Python floats (no
NumPy, no code shared with `crucible.rapid.primitives`); observation channels; revenue and cost. It consumes the
world's documented random-number contract - per tick, standard normals and uniforms in the order: natural lever
noise, node primitives, gauge channels, each taking the counts its primitive documents, then one normal for the
realised revenue when the money has a `realisation_sd` - so given the same
tick draws it must reproduce the vectorised batch engine tick for tick. A disagreement is a bug in one of them.

Supported primitives are listed in `SUPPORTED`; `supports(config)` says whether a mechanism uses only those.
"""
from __future__ import annotations

import math
from collections import deque

SUPPORTED = ("linear_nonlinear", "threshold", "saturation", "non_monotonic", "multiplicative", "hidden_confounding",
             "delay", "decay", "stock", "regime_change", "state_dependent_effect", "measurement_noise",
             "feedback", "complementarity", "substitution")
DRAWS = {"hidden_confounding": (1, 0), "measurement_noise": (1, 0)}       # (normals, uniforms) per tick


def _name(ref: str) -> str:
    return ref.split("@")[0]


def supports(config: dict) -> bool:
    if config.get("engine") != "primitive_graph":
        return False
    prims = {_name(n["prim"]) for n in config["nodes"] if n["type"] == "prim"}
    prims |= {_name(c["prim"]) for g in config.get("gauges", []) for c in g.get("chain", [])}
    single = all(len(g["src"]) == 1 and not g.get("on_demand") for g in config.get("gauges", []))
    return prims <= set(SUPPORTED) and single and not config.get("experiment")


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class ScalarGraph:
    def __init__(self, config: dict):
        if not supports(config):
            raise ValueError("this mechanism uses primitives the scalar twin does not implement")
        self.c = config
        self.nodes = {n["id"]: n for n in config["nodes"]}
        self.order = list(config.get("order", [n["id"] for n in config["nodes"]]))
        lags = [1] + [int(t[1]) for n in config["nodes"]
                      for ts in (n.get("natural", {}) if n["type"] == "lever" else n).get("ports", {}).values()
                      for t in ts] + [int(g.get("lag", 0)) + 1 for g in config.get("gauges", [])]
        self.K = max(lags)
        self.levers = [n for n in config["nodes"] if n["type"] == "lever"]
        self.decisions = [n for n in config["nodes"] if n["type"] == "discrete"]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.hist: deque = deque([{i: 0.0 for i in self.nodes} for _ in range(self.K)], maxlen=self.K)
        self.lev = {n["id"]: None for n in self.levers}                  # None: not set (default or natural)
        self.choice = {n["id"]: int(n["options"].get("default", 0)) for n in self.decisions}
        self.mem: dict = {}
        for n in self.c["nodes"]:
            if n["type"] != "prim":
                continue
            p, k = n["params"], _name(n["prim"])
            if k == "hidden_confounding":
                self.mem[n["id"]] = p["mean"]
            elif k == "delay":
                self.mem[n["id"]] = deque([p.get("init", 0.0)] * int(p["lag"]))
            elif k in ("decay", "stock"):
                self.mem[n["id"]] = p.get("init", 0.0)
            elif k == "regime_change":
                self.mem[n["id"]] = False
            elif k == "feedback":
                self.mem[n["id"]] = 0.0

    def set_lever(self, name: str, value: float) -> None:
        for n in self.levers:
            if n["lever"]["name"] == name:
                lv = n["lever"]
                v = min(max(float(value), lv["lo"]), lv["hi"])
                self.lev[n["id"]] = float(round(v)) if lv.get("integer") else v
                return
        raise KeyError(name)

    # ------------------------------------------------------------------ one tick
    def _port(self, n: dict, port: str, cur: dict, natural: bool = False) -> float:
        src = n.get("natural", {}) if natural else n
        v = float(src.get("bias", {}).get(port, 0.0))
        for s, lag, w in src.get("ports", {}).get(port, []):
            v += w * (cur[s] if int(lag) == 0 else self.hist[-int(lag)][s])
        return v

    def step(self, z: list[float], u: list[float], t: int) -> dict:
        """One tick with this tick's normal draws `z` and uniform draws `u`. Returns node values, readings,
        revenue and cost."""
        zi = 0
        cur: dict = {}
        for nid in self.order:
            n = self.nodes[nid]
            if n["type"] == "lever":
                lv = n["lever"]
                if "natural" in n:
                    nat = self._port(n, "x", cur, natural=True) + n["natural"].get("sd", 0.0) * z[zi]
                    zi += 1
                    nat = min(max(nat, lv["lo"]), lv["hi"])
                else:
                    nat = lv["default"]
                cur[nid] = self.lev[nid] if self.lev[nid] is not None else nat
                continue
            if n["type"] == "discrete":
                cur[nid] = n["options"]["values"][self.choice[nid]]
                continue
            k, p = _name(n["prim"]), n["params"]
            x = {pn: self._port(n, pn, cur) for pn in self._ports_of(k)}
            if k == "linear_nonlinear":
                v = x["x"]
                f = {"linear": v, "quadratic": v + p.get("c", 0.0) * v * v,
                     "power": math.copysign(abs(v) ** p.get("k", 2.0), v) if v != 0 else 0.0,
                     "logistic": _sig(p.get("c", 1.0) * (v - p.get("m", 0.0)))}[p["form"]]
                y = p["b"] + p["a"] * f
            elif k == "threshold":
                y = p["hi"] if x["x"] >= p["theta"] else p["lo"]
            elif k == "saturation":
                v = max(x["x"], 0.0)
                y = p["cap"] * v / (v + p["half"])
            elif k == "non_monotonic":
                r = max(x["x"], 0.0) / p["peak"]
                y = p["height"] * r * math.exp(1.0 - r)
            elif k == "multiplicative":
                y = p.get("k", 1.0) * x["a"] * x["b"]
            elif k == "complementarity":
                a, b = max(x["a"], 1e-9), max(x["b"], 1e-9)
                y = p["scale"] * (p["w"] * a ** p["rho"] + (1 - p["w"]) * b ** p["rho"]) ** (1.0 / p["rho"])
            elif k == "substitution":
                e = max(x["a"], 0.0) + p["w"] * max(x["b"], 0.0)
                y = p["cap"] * (1.0 - math.exp(-e / p["k"]))
            elif k == "hidden_confounding":
                y = p["mean"] + p["ar"] * (self.mem[nid] - p["mean"]) + p["sd"] * z[zi]
                zi += 1
                self.mem[nid] = y
            elif k == "delay":
                buf = self.mem[nid]
                y = buf.popleft()
                buf.append(x["x"])
            elif k == "decay":
                y = (1.0 - p["rate"]) * self.mem[nid] + x["x"]
                self.mem[nid] = y
            elif k == "stock":
                inflow = max(x["inflow"], 0.0)
                avail = self.mem[nid] + inflow
                taken = min(max(x["outflow"], 0.0), avail)
                y = min(avail - taken, p["cap"])
                self.mem[nid] = y
            elif k == "regime_change":
                on = self.mem[nid] or (p["at"] >= 0 and t >= p["at"]) or x["trigger"] >= p["trig"]
                self.mem[nid] = on
                y = x["x"] * (p["after"] if on else p["before"])
            elif k == "state_dependent_effect":
                y = p["gain"] * x["a"] * (p["setpoint"] - x["s"])
            elif k == "feedback":
                y = x["x"] + p["g"] * self.mem[nid]
                self.mem[nid] = y
            else:                                                     # pragma: no cover - guarded by supports()
                raise ValueError(k)
            cur[nid] = y
        self.hist.append(cur)
        reads = []
        for g in self.c.get("gauges", []):
            lag = int(g.get("lag", 0))
            vals = [cur[s] if lag == 0 else self.hist[-1 - lag][s] for s in g["src"]]
            v = vals[0] if len(vals) == 1 else sum(vals) / len(vals)
            for item in g.get("chain", []):
                k, p = _name(item["prim"]), item["params"]
                if k == "measurement_noise":
                    v = v * (1.0 + p["sd"] * z[zi]) if p.get("mult") else v + p["sd"] * z[zi]
                    zi += 1
            reads.append(v)
        m = self.c.get("money") or {}
        rev = cost = 0.0
        if m:
            q = cur[m["revenue_node"]]
            if m.get("realisation_sd", 0.0) > 0.0:        # realised sales: the last normal draw of the tick
                q = q * (1.0 + m["realisation_sd"] * z[zi])
                zi += 1
            rev = m["price"] * min(max(q, 0.0), m["cap"])
            cost = m.get("fixed_cost", 0.0) + sum(cur[n["id"]] * n["lever"].get("cost", 0.0) for n in self.levers)
            for n in self.decisions:
                cost += n["options"].get("cost", [0.0] * len(n["options"]["values"]))[self.choice[n["id"]]]
            cost = max(cost, 0.0)
        return {"values": cur, "readings": reads, "revenue": rev, "cost": cost}

    def draws(self) -> tuple[int, int]:
        """(normals, uniforms) this mechanism takes per tick, counted from the documented contract."""
        nn = sum(1 for n in self.c["nodes"] if n["type"] == "lever" and "natural" in n)
        nn += sum(DRAWS.get(_name(n["prim"]), (0, 0))[0] for n in self.c["nodes"] if n["type"] == "prim")
        nn += sum(DRAWS.get(_name(c["prim"]), (0, 0))[0] for g in self.c.get("gauges", []) for c in g.get("chain", []))
        m = self.c.get("money") or {}
        nn += 1 if m and m.get("realisation_sd", 0.0) > 0.0 else 0
        return nn, 0

    def warm_up(self, rng) -> None:
        """The pre-arrival history: `warmup` ticks under the levers' natural regime, drawing from the world's
        `init` generator (as the documented contract says)."""
        W = int(self.c.get("warmup", 0))
        nn, nu = self.draws()
        for w in range(W):
            z = list(rng.standard_normal(nn))
            u = list(rng.random(nu))
            self.step(z, u, -W + w)

    @staticmethod
    def _ports_of(kind: str) -> tuple:
        return {"multiplicative": ("a", "b"), "complementarity": ("a", "b"), "substitution": ("a", "b"),
                "stock": ("inflow", "outflow"), "regime_change": ("x", "trigger"),
                "state_dependent_effect": ("a", "s"), "hidden_confounding": ()}.get(kind, ("x",))


def compare(family: str, seed: int, ticks: int = 40, action_seed: int = 0) -> dict:
    """Run a world of `family` (the vectorised engine behind `RapidEnv`) and the scalar twin in lockstep on the
    same tick draws, with random lever settings, and count ticks whose readings, revenue or cost differ (beyond
    1e-9 relative: the two implementations compute in a different order)."""
    import numpy as np

    from ..mechanisms.base import tick_rng
    from ..mechanisms.sim import TickRng
    from .families import make
    env = make(family, seed)
    tw = ScalarGraph(env._config)
    tw.warm_up(tick_rng(env.world_seed, env.env_id, "init"))
    rng_t = TickRng(env.world_seed, env.env_id)
    nn, nu = tw.draws()
    if (nn, nu) != (env._engine.n_normal, env._engine.n_uniform):
        return {"ok": False, "ticks": 0, "mismatches": ticks, "reason": "draw counts differ"}
    arng = np.random.default_rng(action_seed)
    levers = [n["lever"] for n in tw.levers]
    mism, worst = 0, 0.0
    for t in range(ticks):
        vals = {lv["name"]: float(lv["lo"] + arng.random() * (lv["hi"] - lv["lo"])) for lv in levers}
        r = env.step({"kind": "levers", "values": vals})
        for k, v in vals.items():
            tw.set_lever(k, v)
        g = rng_t(t)
        z = list(g.standard_normal(nn))
        u = list(g.random(nu))
        out = tw.step(z, u, t)
        got = [out["readings"][k] for k, gg in enumerate(tw.c.get("gauges", []))]
        want = [r["observation"]["numbers"].get(gg["name"], float("nan")) for gg in tw.c.get("gauges", [])]
        res = r["observation"]["resources"]
        pairs = list(zip(got, want)) + [(out["revenue"], res.get("revenue_cents_last_tick", 0.0)),
                                         (out["cost"], res.get("cost_cents_last_tick", 0.0))]
        bad = False
        for a, b in pairs:
            d = abs(a - b) / max(1.0, abs(b))
            worst = max(worst, d)
            bad |= d > 1e-9
        mism += int(bad)
    return {"ok": mism == 0, "ticks": ticks, "mismatches": mism, "max_relative_difference": worst}
