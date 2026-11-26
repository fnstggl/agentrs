"""Engine `primitive_graph`: causal graphs composed from the registered primitives (Part 7.1-7.2).

Every node is a lever (set by the participant; when it has never been set it follows its natural mechanism, which is
how historical, passively observed data arise), a discrete decision, or a primitive applied to named input ports.
A port is a weighted sum of other nodes' values, each at a lag: lag 0 reads a node earlier in the evaluation order
in the same tick, lag L >= 1 reads its value L ticks ago, so feedback loops are cycles through lagged edges.
Readings pass through observation channels (chains of channel primitives: measurement noise, missing observations,
partial observability over several nodes, selection effects) and may be reported late or only on demand.

The engine is vectorised over a batch of instances with one structure: `PrimitiveGraph.batch(configs)` stacks the
numeric parameters of several seeds of one family and steps them together. A single instance is a batch of one, and
both paths run the same code on the same per-instance random numbers, which is why they agree bit for bit.

State is a flat mapping of arrays (leading dimension = batch), so a kernel facility can store it as is.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import actions as A
from ..primitives import get as get_prim
from ..primitives import parse_ref
from .base import ActionError, RapidEngine, register


def _f(x) -> float:
    return float(x)


@register
class PrimitiveGraph(RapidEngine):
    kind = "primitive_graph"
    batched = True

    def __init__(self, config: dict, _stack: list[dict] | None = None):
        super().__init__(config)
        self._compile(_stack)

    # ------------------------------------------------------------------ compilation
    def _compile(self, stack: list[dict] | None) -> None:
        c = self.config
        cfgs = stack or [c]
        self.B = len(cfgs)
        nodes = c["nodes"]
        self.ids = [n["id"] for n in nodes]
        self.idx = {k: i for i, k in enumerate(self.ids)}
        self.n = len(nodes)
        self.order = [self.idx[k] for k in c.get("order", self.ids)]
        lags = [1]
        for nd in nodes:
            for terms in list(nd.get("ports", {}).values()) + list(nd.get("natural", {}).get("ports", {}).values()):
                lags += [int(tm[1]) for tm in terms]
        # a gauge with lag L reads the value of L ticks ago from the history after this tick's values are appended,
        # so the history must hold L + 1 ticks
        lags += [int(g.get("lag", 0)) + 1 for g in c.get("gauges", []) if int(g.get("lag", 0)) > 0]
        self.K = max(lags) if max(lags) >= 1 else 1
        self.lev_nodes = [i for i, nd in enumerate(nodes) if nd["type"] == "lever"]
        self.dec_nodes = [i for i, nd in enumerate(nodes) if nd["type"] == "discrete"]
        self.lev_pos = {i: j for j, i in enumerate(self.lev_nodes)}
        self.dec_pos = {i: j for j, i in enumerate(self.dec_nodes)}
        self.prims = {i: parse_ref(nd["prim"]) for i, nd in enumerate(nodes) if nd["type"] == "prim"}
        # numeric parameters: scalars for one instance, (B,) arrays for a stacked batch
        self.P: dict[int, dict] = {}
        self.W: dict[int, dict] = {}            # node -> port -> [(src index, lag, weight array)]
        self.Bias: dict[int, dict] = {}
        for i, nd in enumerate(nodes):
            nds = [cc["nodes"][i] for cc in cfgs]
            if any(x["id"] != nd["id"] or x["type"] != nd["type"] or x.get("prim") != nd.get("prim") for x in nds):
                raise ValueError("batched configs differ in structure")
            shape = lambda x: {p: [(tm[0], int(tm[1])) for tm in ts] for p, ts in  # noqa: E731
                               (x.get("natural", {}) if x["type"] == "lever" else x).get("ports", {}).items()}
            if any(shape(x) != shape(nd) for x in nds):
                raise ValueError("batched configs differ in their wiring or lags")
            if nd["type"] == "prim":
                self.P[i] = _stack_params([x.get("params", {}) for x in nds], self.B, self.prims[i])
            src = "natural" if nd["type"] == "lever" else None
            ports = (nd.get("natural", {}) if src else nd).get("ports", {})
            biases = [(x.get("natural", {}) if src else x).get("bias", {}) for x in nds]
            self.W[i] = {}
            self.Bias[i] = {}
            for pn, terms in ports.items():
                ws = []
                for k, tm in enumerate(terms):
                    w = _stack([(x.get("natural", {}) if src else x)["ports"][pn][k][2] for x in nds], self.B)
                    ws.append((self.idx[tm[0]], int(tm[1]), w))
                self.W[i][pn] = ws
                self.Bias[i][pn] = _stack([b.get(pn, 0.0) for b in biases], self.B)
        self.lev = [nodes[i]["lever"] for i in self.lev_nodes]
        self.lev_lo = np.array([lv["lo"] for lv in self.lev], dtype=float)
        self.lev_hi = np.array([lv["hi"] for lv in self.lev], dtype=float)
        self.lev_default = _stack_rows([[cc["nodes"][i]["lever"]["default"] for i in self.lev_nodes] for cc in cfgs])
        self.lev_cost = _stack_rows([[cc["nodes"][i]["lever"].get("cost", 0.0) for i in self.lev_nodes]
                                     for cc in cfgs])
        self.nat_sd = {i: _stack([cc["nodes"][i].get("natural", {}).get("sd", 0.0) for cc in cfgs], self.B)
                       for i in self.lev_nodes if "natural" in nodes[i]}
        self.dec = [nodes[i]["options"] for i in self.dec_nodes]
        self.dec_values = [_stack_rows([cc["nodes"][i]["options"]["values"] for cc in cfgs]) for i in self.dec_nodes]
        self.dec_cost = [_stack_rows([cc["nodes"][i]["options"].get("cost", [0.0] * len(nodes[i]["options"]["values"]))
                                      for cc in cfgs]) for i in self.dec_nodes]
        # random draws per tick: lever natural noise, node primitives, then gauge channels
        nn = nu = 0
        self.draw_at: dict[Any, tuple[int, int, int, int]] = {}
        for i in self.order:
            nd = nodes[i]
            if nd["type"] == "lever" and "natural" in nd:
                self.draw_at[i] = (nn, 1, nu, 0)
                nn += 1
            elif nd["type"] == "prim":
                a, b = self.prims[i].draws(self.P[i])
                self.draw_at[i] = (nn, a, nu, b)
                nn, nu = nn + a, nu + b
        self.gauges = c.get("gauges", [])
        self.gauge_chain = []
        for gi, g in enumerate(self.gauges):
            chain = []
            for ci, item in enumerate(g.get("chain", [])):
                pr = parse_ref(item["prim"])
                par = _stack_params([cc["gauges"][gi]["chain"][ci].get("params", {}) for cc in cfgs], self.B, pr)
                a, b = pr.draws(par)
                chain.append((pr, par, (nn, a, nu, b)))
                nn, nu = nn + a, nu + b
            self.gauge_chain.append(chain)
        self.n_normal, self.n_uniform = nn, nu
        self.gauge_names = [g["name"] for g in self.gauges]
        m = c.get("money") or {}
        self.money = bool(m)
        self.rev_node = self.idx[m["revenue_node"]] if m else None
        self.price = _stack([cc.get("money", {}).get("price", 0.0) for cc in cfgs], self.B)
        self.cap = _stack([cc.get("money", {}).get("cap", 0.0) for cc in cfgs], self.B)
        self.fixed = _stack([cc.get("money", {}).get("fixed_cost", 0.0) for cc in cfgs], self.B)

    @classmethod
    def batch(cls, configs: list[dict]) -> "PrimitiveGraph":
        """One engine stepping several instances of one structure (several seeds of a family) together."""
        return cls(configs[0], _stack=list(configs))

    # ------------------------------------------------------------------ public surface
    def public_levers(self) -> list[dict]:
        return [{"name": lv["name"], "min": _f(lv["lo"]), "max": _f(lv["hi"]),
                 "default": _f(self.lev_default[0, j]), "integer": bool(lv.get("integer", False))}
                for j, lv in enumerate(self.lev)]

    def visible_gauges(self) -> list[str]:
        return [g["name"] for g in self.gauges if not g.get("on_demand")]

    def action_space(self, agent: str) -> A.ActionSpace:
        c = self.config
        kinds, limits = {}, {}
        allowed = c.get("affordances", ["levers"])
        if self.lev and "levers" in allowed:
            kinds["levers"] = A.levers_schema(self.public_levers())
        if self.dec and "discrete" in allowed:
            decs = {d["name"]: list(d["choices"]) for d in self.dec}
            kinds["discrete"] = A.discrete_schema(decs)
            limits["decisions"] = decs
        q = self.query_targets(agent)
        if q:
            kinds["query"] = A.query_schema(dict(q, tools="the action kinds available to you and their schemas",
                                                 history="your last observations (args: n)",
                                                 spec="the public description of this world"))
        ex = c.get("experiment")
        if ex and "experiment" in allowed:
            var = [d for d in self.public_levers() if d["name"] in ex["vars"]]
            box = ex.get("box", {})
            var = [dict(d, min=box.get(d["name"], [d["min"], d["max"]])[0], max=box.get(d["name"], [d["min"],
                                                                                         d["max"]])[1]) for d in var]
            kinds["experiment"] = A.experiment_schema(var, list(ex["measure"]), int(ex["max_ticks"]),
                                                      int(ex["max_replicates"]), _f(ex["cost"]),
                                                      sorted(ex.get("instruments", {})) or None)
        if c.get("journal") or "message" in allowed:
            kinds["message"] = A.message_schema(["journal"])
        if "program" in allowed:
            kinds["program"] = A.program_schema(int(c.get("program_max_ticks", self.horizon)),
                                                "Allowed actions inside: levers, discrete.")
            limits["program_kinds"] = ["levers", "discrete"]
        if len(kinds) > 1:
            inner = [k for k in kinds if k not in ("program",)]
            kinds["sequence"] = A.sequence_schema(inner)
            limits["sequence_kinds"] = inner
        return A.ActionSpace(kinds, limits)

    def query_targets(self, agent: str) -> dict[str, str]:
        out = {}
        for g in self.gauges:
            if g.get("on_demand"):
                out[g["name"]] = f"reading of sensor {g['name']} (costs {g.get('query_cost', 0)} budget)"
        return out

    # ------------------------------------------------------------------ state
    def empty_state(self, B: int | None = None) -> dict:
        B = self.B if B is None else B
        st = {"h": np.zeros((B, self.K, self.n)), "lev": np.zeros((B, len(self.lev))) + self.lev_default,
              "levset": np.zeros((B, len(self.lev))), "choice": np.zeros((B, len(self.dec))),
              "budget": np.zeros(B) + _f(self.config.get("experiment", {}).get("budget", 0.0)),
              "read": np.zeros((B, len(self.gauges)))}
        for j, d in enumerate(self.dec):
            st["choice"][:, j] = int(d.get("default", 0))
        for i, pr in self.prims.items():
            for k, v in pr.init(self.P[i], B).items():
                st[f"ps:{i}:{k}"] = np.array(v, dtype=float)
        if self.config.get("warmup"):                # keyed by position, so instances of one family stack
            for k, g in enumerate(self.gauges):
                if not g.get("on_demand"):
                    st[f"hist:g{k}"] = np.zeros((B, int(self.config["warmup"])))
            for j in range(len(self.lev)):
                st[f"hist:l{j}"] = np.zeros((B, int(self.config["warmup"])))
        return st

    def init_state(self, rng: np.random.Generator) -> dict:
        """The state at t=0. With `warmup` W, the world first runs W ticks with every lever at its natural
        mechanism (the previous operator's regime); what was observable then becomes the participant's history.
        All of it draws from the one `init` generator."""
        st = self.empty_state()
        W = int(self.config.get("warmup", 0))
        vis = [k for k, g in enumerate(self.gauges) if not g.get("on_demand")]
        for w in range(W):
            z, u = self.draw(rng)
            cur, read = self.tick(st, z, u, -W + w)
            for k in vis:
                st[f"hist:g{k}"][:, w] = read[:, k]
            for j, i in enumerate(self.lev_nodes):
                st[f"hist:l{j}"][:, w] = cur[:, i]
        return st

    def history_of(self, state: dict, name: str) -> np.ndarray:
        """The recorded pre-arrival series of a gauge or lever, by public name."""
        if name in self.gauge_names:
            return state[f"hist:g{self.gauge_names.index(name)}"][0]
        return state[f"hist:l{[lv['name'] for lv in self.lev].index(name)}"][0]

    def draw(self, rng: np.random.Generator, B: int = 1) -> tuple[np.ndarray, np.ndarray]:
        z = rng.standard_normal(self.n_normal * B).reshape(B, self.n_normal)
        u = rng.random(self.n_uniform * B).reshape(B, self.n_uniform)
        return z, u

    # ------------------------------------------------------------------ the tick (vectorised)
    def _port(self, i: int, pn: str, cur: np.ndarray, h: np.ndarray) -> np.ndarray:
        v = np.zeros(cur.shape[0]) + self.Bias[i][pn]
        for src, lag, w in self.W[i][pn]:
            v = v + w * (cur[:, src] if lag == 0 else h[:, self.K - lag, src])
        return v

    def tick(self, st: dict, z: np.ndarray, u: np.ndarray, t: int, params: dict | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
        """Advance `st` in place by one tick with draws z (B, n_normal), u (B, n_uniform). Returns (node values,
        gauge readings (B, n_gauges), NaN where missing)."""
        B = z.shape[0]
        h = st["h"]
        cur = np.zeros((B, self.n))
        for i in self.order:
            nd = self.config["nodes"][i]
            typ = nd["type"]
            if typ == "lever":
                j = self.lev_pos[i]
                if i in self.nat_sd:
                    a = self.draw_at[i][0]
                    nat = self._port(i, "x", cur, h) + self.nat_sd[i] * z[:, a]
                    nat = np.clip(nat, self.lev_lo[j], self.lev_hi[j])
                else:
                    nat = self.lev_default[:, j] + 0.0 * cur[:, 0]
                v = np.where(st["levset"][:, j] > 0, st["lev"][:, j], nat)
            elif typ == "discrete":
                j = self.dec_pos[i]
                ch = st["choice"][:, j].astype(int)
                v = np.take_along_axis(self.dec_values[j] + np.zeros((B, 1)), ch[:, None], axis=1)[:, 0]
            else:
                pr = self.prims[i]
                ports = {pn: self._port(i, pn, cur, h) for pn in pr.ports}
                pre = f"ps:{i}:"
                ps = {k[len(pre):]: st[k] for k in st if k.startswith(pre)}
                a, na, b, nb = self.draw_at[i]
                v, ps2 = pr.apply(self.P[i], ports, ps, z[:, a:a + na], u[:, b:b + nb], t)
                for k, x in ps2.items():
                    st[pre + k] = x
            cur[:, i] = v
        nh = np.empty_like(h)
        nh[:, :-1] = h[:, 1:]
        nh[:, -1] = cur
        st["h"] = nh
        read = self.readings(cur, nh, z, u, t, params)
        st["read"] = read
        return cur, read

    def readings(self, cur: np.ndarray, h: np.ndarray, z: np.ndarray, u: np.ndarray, t: int,
                 params: dict | None = None) -> np.ndarray:
        B = cur.shape[0]
        out = np.zeros((B, len(self.gauges)))
        for k, g in enumerate(self.gauges):
            lag = int(g.get("lag", 0))
            srcs = [self.idx[s] for s in g["src"]]
            # `h` already holds this tick's values at its last position: L ticks ago is position K - 1 - L
            col = (lambda s: cur[:, s]) if lag == 0 else (lambda s: h[:, self.K - 1 - lag, s])
            if len(srcs) > 1:
                x: Any = {"xs": np.stack([col(s) for s in srcs], axis=1)}
            else:
                x = {"x": col(srcs[0])}
            val = None
            for pr, par, (a, na, b, nb) in self.gauge_chain[k]:
                if params and (k, pr.name) in params:
                    par = dict(par, **params[(k, pr.name)])
                val, _ = pr.apply(par, x, {}, z[:, a:a + na], u[:, b:b + nb], t)
                x = {"x": val}
            out[:, k] = x["x"] if val is None else val
        return out

    def money_flows(self, st: dict, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B = cur.shape[0]
        if not self.money:
            return np.zeros(B), np.zeros(B)
        rev = self.price * np.clip(cur[:, self.rev_node], 0.0, self.cap)
        levv = np.stack([cur[:, i] for i in self.lev_nodes], axis=1) if self.lev_nodes else np.zeros((B, 0))
        cost = self.fixed + (levv * self.lev_cost).sum(axis=1)
        for j in range(len(self.dec)):
            ch = st["choice"][:, j].astype(int)
            cost = cost + np.take_along_axis(self.dec_cost[j] + np.zeros((B, 1)), ch[:, None], axis=1)[:, 0]
        return rev, np.maximum(cost, 0.0)

    def batch_step(self, st: dict, z: np.ndarray, u: np.ndarray, t: int) -> dict:
        """Vectorised tick for a batch (state changed in place): node values, readings, revenue and cost."""
        cur, read = self.tick(st, z, u, t)
        rev, cost = self.money_flows(st, cur)
        return {"values": cur, "readings": read, "revenue": rev, "cost": cost}

    def set_levers(self, st: dict, values: dict[str, float], clip: bool = True) -> None:
        """Set levers (clipped to their public ranges; the evaluator may set them outside with clip=False)."""
        names = [lv["name"] for lv in self.lev]
        for k, v in values.items():
            j = names.index(k)
            x = float(np.clip(float(v), self.lev_lo[j], self.lev_hi[j])) if clip else float(v)
            if self.lev[j].get("integer"):
                x = float(round(x))
            st["lev"][:, j] = x
            st["levset"][:, j] = 1.0

    # ------------------------------------------------------------------ actions (single instance)
    def apply(self, state, agent, action, rng, t):
        k = action["kind"]
        if k == "levers":
            self.set_levers(state, action["values"])
            return {"set": {n: _f(state["lev"][0, [lv["name"] for lv in self.lev].index(n)])
                            for n in action["values"]}}
        if k == "discrete":
            j = [d["name"] for d in self.dec].index(action["decision"])
            state["choice"][:, j] = self.dec[j]["choices"].index(action["choice"])
            return {"decision": action["decision"], "choice": action["choice"]}
        if k == "query":
            return self._query(state, action, rng)
        if k == "experiment":
            return self._experiment(state, action["design"], rng, t)
        raise ActionError(f"{k} is not available in this world")

    def _query(self, state, action, rng):
        name = action["target"]
        if name not in self.gauge_names:
            raise ActionError(f"unknown query target {name}")
        g = self.gauges[self.gauge_names.index(name)]
        cost = float(g.get("query_cost", 0.0))
        if state["budget"][0] < cost - 1e-12:
            raise ActionError(f"not enough budget ({state['budget'][0]:.3g} left, the query costs {cost:g})")
        state["budget"] = state["budget"] - cost
        v = float(state["read"][0, self.gauge_names.index(name)])
        return {"target": name, "value": None if v != v else v, "cost": cost}

    def _experiment(self, state, d, rng, t):
        ex = self.config["experiment"]
        reps, ticks = int(d.get("replicates", 1)), int(d.get("ticks", 1))
        inst = ex.get("instruments", {}).get(d.get("instrument"), {"sd_mult": 1.0, "cost_mult": 1.0})
        cost = float(ex["cost"]) * reps * ticks * float(inst.get("cost_mult", 1.0))
        if state["budget"][0] < cost - 1e-12:
            raise ActionError(f"not enough experiment budget ({state['budget'][0]:.3g} left, this design costs "
                              f"{cost:g})")
        box = ex.get("box", {})
        for k, v in d["set"].items():
            if k not in ex["vars"]:
                raise ActionError(f"{k} cannot be set in a trial")
            lo, hi = box.get(k, [None, None])
            if lo is not None and not (lo - 1e-12 <= v <= hi + 1e-12):
                raise ActionError(f"{k}={v} is outside the pilot plant's range [{lo}, {hi}]")
        state["budget"] = state["budget"] - cost
        st = {k: np.repeat(v, reps, axis=0) for k, v in state.items()}
        self.set_levers(st, d["set"])
        over = {}
        for k in range(len(self.gauges)):
            for pr, par, _ in self.gauge_chain[k]:
                if pr.name == "measurement_noise":
                    over[(k, pr.name)] = {"sd": par["sd"] * float(inst.get("sd_mult", 1.0))}
        sums: dict[str, list] = {m: [] for m in d["measure"]}
        for k in range(ticks):
            z, u = self.draw(rng, reps)
            cur, read = self.tick(st, z, u, t + k, over)
            for m in d["measure"]:
                sums[m].append(read[:, self.gauge_names.index(m)])
        res = {}
        for m, vals in sums.items():
            a = np.concatenate(vals)
            a = a[~np.isnan(a)]
            res[m] = {"mean": float(a.mean()) if a.size else None,
                      "se": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else None, "n": int(a.size)}
        return {"design": d, "measured": res, "cost": cost, "budget_left": float(state["budget"][0])}

    def advance(self, state, rng, t):
        z, u = self.draw(rng)
        cur, read = self.tick(state, z, u, t)
        rev, cost = self.money_flows(state, cur)
        nums = {}
        for k, g in enumerate(self.gauges):
            if not g.get("on_demand"):
                nums[g["name"]] = float(read[0, k])
        a = self.agents[0]
        res = {"numbers": nums, "resources": {"experiment_budget": float(state["budget"][0])}}
        return {"obs": {a: res}, "revenue": {a: float(rev[0])}, "cost": {a: float(cost[0])},
                "info": {"values": cur[0].tolist()}}

    def history(self, state) -> dict:
        W = int(self.config.get("warmup", 0))
        if not W:
            return {}
        names = {f"g{k}": g["name"] for k, g in enumerate(self.gauges)}
        names.update({f"l{j}": lv["name"] for j, lv in enumerate(self.lev)})
        arr = {f"history.{names[k[5:]]}": state[k][0].tolist() for k in sorted(state) if k.startswith("hist:")}
        return {"arrays": arr, "text": [f"Records of the {W} ticks before you took over: the settings the previous "
                                        "operator used and what the sensors showed."]}

    # ------------------------------------------------------------------ evaluator side
    def truth(self) -> dict:
        c = self.config
        nodes = [{"id": n["id"], "type": n["type"], "prim": n.get("prim"), "role": n.get("role", ""),
                  "public": (n.get("lever") or n.get("options") or {}).get("name"), "hidden": bool(n.get("hidden"))}
                 for n in c["nodes"]]
        edges = []
        for nd in c["nodes"]:
            ports = nd.get("ports", {}) if nd["type"] != "lever" else nd.get("natural", {}).get("ports", {})
            for pn, terms in ports.items():
                for src, lag, w in terms:
                    edges.append({"src": src, "dst": nd["id"], "port": pn, "lag": int(lag),
                                  "sign": int(np.sign(w)), "kind": "natural" if nd["type"] == "lever" else "causal"})
        for g in c.get("gauges", []):
            for s in g["src"]:
                edges.append({"src": s, "dst": "gauge:" + g["name"], "port": "obs", "lag": int(g.get("lag", 0)),
                              "sign": 1, "kind": "observation"})
        return {"engine": self.kind, "nodes": nodes, "edges": edges,
                "gauges": [{"name": g["name"], "src": g["src"], "chain": [x["prim"] for x in g.get("chain", [])],
                            "lag": int(g.get("lag", 0)), "on_demand": bool(g.get("on_demand"))} for g in c.get("gauges", [])],
                "regimes": c.get("regimes", []), "motifs": self.motifs()}

    def motifs(self) -> list[str]:
        """Primitives in use, read off the graph. A source counts as hidden confounding only when it is hidden and
        drives at least two other nodes (a common cause nobody can observe)."""
        out = set()
        tr_children: dict[str, int] = {}
        for nd in self.config["nodes"]:
            ports = nd.get("ports", {}) if nd["type"] != "lever" else nd.get("natural", {}).get("ports", {})
            for terms in ports.values():
                for src, _, w in terms:
                    if w != 0:
                        tr_children[src] = tr_children.get(src, 0) + 1
        for nd in self.config["nodes"]:
            if nd["type"] != "prim":
                continue
            name = nd["prim"].split("@")[0]
            if name == "hidden_confounding":
                if nd.get("hidden") and tr_children.get(nd["id"], 0) >= 2:
                    out.add(name)
            else:
                out.add(name)
        for g in self.config.get("gauges", []):
            for x in g.get("chain", []):
                out.add(x["prim"].split("@")[0])
            if g.get("lag", 0) > 0:
                out.add("delay")
        for nd in self.config["nodes"]:
            for terms in nd.get("ports", {}).values():
                if any(int(tm[1]) > 0 for tm in terms):
                    out.add("delay")
        return sorted(out)

    def outcome(self, state, ledger):
        a = self.agents[0]
        obj = self.config.get("objective", {})
        net = float(ledger["revenue"].get(a, 0.0) - ledger["cost"].get(a, 0.0))
        out = {"net": net, "revenue": float(ledger["revenue"].get(a, 0.0)), "cost": float(ledger["cost"].get(a, 0.0))}
        if obj.get("type") == "predict_heldout":
            out.update(score_predictions(self, ledger.get("journal", [])))
            out["value"] = out["prediction_score"]
        else:
            out["value"] = net
        return out

    def conservation(self, before, after, out):
        errs = []
        a = self.agents[0]
        if out["revenue"][a] < 0 or out["cost"][a] < 0:
            errs.append("negative money flow")
        if self.money and out["revenue"][a] > float(np.max(self.price * self.cap)) + 1e-6:
            errs.append("revenue above what demand can pay")
        if after["budget"][0] > before["budget"][0] + 1e-9:
            errs.append("experiment budget grew")
        for k in after:
            if k.startswith("ps:") and k.endswith(":taken"):
                i = k.split(":")[1]
                lv0, lv1 = before[f"ps:{i}:level"], after[f"ps:{i}:level"]
                bal = lv1 - lv0 - (after[f"ps:{i}:inflow"] - after[k] - after[f"ps:{i}:spill"])
                if not np.all(lv1 >= -1e-9):
                    errs.append(f"stock {i} went negative")
                if np.max(np.abs(bal)) > 1e-6 * (1.0 + float(np.max(np.abs(lv1)))):
                    errs.append(f"stock {i} does not balance its flows")
        return errs


def _stack(vals: list, B: int):
    if B == 1:
        return float(vals[0]) if not isinstance(vals[0], (list, tuple)) else np.asarray(vals[0], dtype=float)
    return np.asarray(vals, dtype=float)


def _stack_rows(rows: list[list]) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(len(rows), -1)


def _stack_params(ps: list[dict], B: int, prim=None) -> dict:
    """Parameters of one node across a batch: numbers become (B,) arrays; strings, booleans and integer-typed
    parameters (lags, counts) are structure and must agree."""
    out = {}
    ints = {k for k, s in (prim.params.get("properties", {}) if prim is not None else {}).items()
            if s.get("type") == "integer"}
    for k, v in ps[0].items():
        if isinstance(v, str) or isinstance(v, bool):
            if any(p.get(k) != v for p in ps):
                raise ValueError(f"batched configs differ in structural parameter {k}")
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v) if B == 1 else np.asarray([p[k] for p in ps], dtype=float)
        elif k in ints:
            if any(p.get(k) != v for p in ps):
                raise ValueError(f"batched configs differ in structural parameter {k}")
            out[k] = v
        else:
            out[k] = float(v) if B == 1 else np.asarray([float(p[k]) for p in ps])
    return out


def score_predictions(eng: PrimitiveGraph, journal: list[dict]) -> dict:
    """Evaluator: the latest predictions filed with the journal, scored against the true noise-free mechanism at
    the held-out conditions of the objective. Score = 1 - MSE / MSE(predicting the mean of the true values)."""
    obj = eng.config["objective"]
    conds = obj["heldout"]
    gauge = obj["gauge"]
    truth = true_response(eng, conds, gauge, ticks=int(obj.get("settle_ticks", 1)))
    preds = None
    for m in journal:
        c = (m.get("content") or {}).get("predictions")
        if isinstance(c, list):
            preds = c
    if preds is None:
        return {"prediction_score": 0.0, "predictions_filed": 0, "heldout": len(conds)}
    got = []
    for k, cond in enumerate(conds):
        v = None
        for p in preds:
            if isinstance(p, dict) and p.get("set") == cond and isinstance(p.get("value"), (int, float)):
                v = float(p["value"])
        got.append(v)
    filled = np.array([np.nan if v is None else v for v in got])
    tv = np.asarray(truth)
    base = float(np.mean((tv - tv.mean()) ** 2)) or 1.0
    err = np.where(np.isnan(filled), (tv - tv.mean()) ** 2 * 4.0, (filled - tv) ** 2)
    return {"prediction_score": float(1.0 - err.mean() / base), "predictions_filed": int((~np.isnan(filled)).sum()),
            "heldout": len(conds)}


def true_response(eng: PrimitiveGraph, conds: list[dict], gauge: str, ticks: int = 1) -> list[float]:
    """The noise-free steady response of a node behind `gauge` with the levers set to each condition (evaluator)."""
    out = []
    g = eng.gauges[eng.gauge_names.index(gauge)]
    src = eng.idx[g["src"][0]]
    dnames = [d["name"] for d in eng.dec]
    for cond in conds:
        st = eng.empty_state(1)
        eng.set_levers(st, {k: v for k, v in cond.items() if k not in dnames}, clip=False)
        for k, v in cond.items():                 # a condition may also fix a discrete decision (by its option)
            if k in dnames:
                j = dnames.index(k)
                st["choice"][:, j] = list(eng.dec[j]["choices"]).index(v)
        z = np.zeros((1, eng.n_normal))
        u = np.ones((1, eng.n_uniform))           # no missing readings, no selection draws below the cut
        val = None
        for k in range(max(ticks, eng.K + 1)):
            cur, _ = eng.tick(st, z, u, 10_000 + k)
            val = float(cur[0, src])
        out.append(val)
    return out


def get_primitive(name: str):
    return get_prim(name)
