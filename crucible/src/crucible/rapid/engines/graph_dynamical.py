"""Engine `graph_dynamical`: dynamical systems on explicit networks (Part 7.3, Part 8.7).

The state lives on the nodes and links of a real graph and evolves by local update rules applied to the whole graph
at once: fluid queues on links that pass flow downstream, threshold contagion that reads each node's neighbourhood,
load redistribution after failures, connectivity of origin-destination pairs over the surviving links. Graph
algorithms (neighbour sums over adjacency, breadth-first search) are part of the transition. This is independent of
the causal-graph engine: there is no node function graph here, only network state.

Scenarios:

    routing    flow from a source to a sink over K routes whose second hops share hidden bottlenecks; fluid
               queues with one-tick transit; route latency is measured with congestion; optionally the topology
               changes at a hidden time (8.7.1, 8.7.6). Vectorised over a batch of instances (`batch_step`).
    diffusion  complex contagion (node thresholds) on a clustered graph; the participant places seeds by
               cluster (`allocate`); value grows with adoption (network effect) (8.7.3)
    cascade    a power layer whose overloaded nodes shed load onto neighbours (cascading failure) and a dependent
               control layer; failures propagate between layers; the participant reinforces zones (`allocate`)
               and adds backup supply links (`graph_edit`) (8.7.4)
    repair     links fail at hidden rates; demand between node pairs is served only while they are connected; the
               participant dispatches a repair crew (`discrete`) or builds new links (`graph_edit`) (8.7.2)
    balance    several participants, one per node of a small grid, each with its own fluctuating demand (seen only
               by that participant) and its own generation cost; power flows over capacity-limited, lossy links
               when a participant exports to a neighbour (`allocate`); meeting every node's demand cheaply needs
               the participants to coordinate through messages (distributed resource coordination, 8.7.5)

State is a flat mapping of arrays (leading dimension = batch for routing), so a kernel facility can store it.
"""
from __future__ import annotations

import numpy as np

from .. import actions as A
from ..primitives import cascade as cascade_fn
from ..primitives import get as prim
from .base import ActionError, RapidEngine, register

_CONG = {"t0": 1.0, "alpha": 0.15, "beta": 4.0}


def _arr(x, B=None):
    a = np.asarray(x, dtype=float)
    return a[None] if B is not None and a.ndim and a.shape[0] != B else a


@register
class GraphDynamical(RapidEngine):
    kind = "graph_dynamical"

    def __init__(self, config: dict, _stack: list[dict] | None = None):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in ("routing", "diffusion", "cascade", "repair", "balance"):
            raise ValueError(f"unknown scenario {self.sc}")
        self.batched = self.sc == "routing"
        self.dynamic_space = False
        if self.sc == "routing":
            self._compile_routing(_stack or [config])
        elif self.sc == "diffusion":
            self.adj = np.asarray(config["adj"], dtype=float)
            self.deg = np.maximum(self.adj.sum(axis=1), 1.0)
        elif self.sc == "cascade":
            self.adj = np.asarray(config["adj"], dtype=float)
        elif self.sc == "balance":
            self.N = len(config["agents"])
        else:
            self.N = int(config["n_nodes"])

    # ================================================================== routing (vectorised)
    def _compile_routing(self, cfgs):
        c = cfgs[0]
        self.B = len(cfgs)
        self.K, self.J = int(c["K"]), int(c["J"])
        for x in cfgs:
            if int(x["K"]) != self.K or int(x["J"]) != self.J:
                raise ValueError("batched routing instances differ in size")
        st = lambda key: np.asarray([x[key] for x in cfgs], dtype=float)  # noqa: E731
        self.c1, self.c2, self.toll = st("c1"), st("c2"), st("toll")
        self.M1 = np.zeros((self.B, self.K, self.J))
        self.M2 = np.zeros((self.B, self.K, self.J))
        for b, x in enumerate(cfgs):
            for k, j in enumerate(x["assign"]):
                self.M1[b, k, j] = 1.0
            for k, j in enumerate(x.get("assign2") or x["assign"]):
                self.M2[b, k, j] = 1.0
        self.rewire_at = np.asarray([x.get("rewire_at", -1) for x in cfgs], dtype=float)
        self.d_mean = st("demand_mean")
        self.d_sd = st("demand_sd")
        self.d_ar = st("demand_ar")
        self.price = st("price")
        self.hold = st("hold")
        self.noise = st("noise")
        self.w_default = np.asarray([x["w_default"] for x in cfgs], dtype=float)
        self.n_normal = self.K + 2
        agg = {bool(x.get("aggregate_only")) for x in cfgs}
        if len(agg) > 1:
            raise ValueError("batched routing instances differ in their measurement channel")
        # aggregate_only: end-to-end measurement only - delivered flow and the flow-weighted mean latency, never a
        # route's own latency (network tomography); computed here so the batch and single paths agree
        self.aggregate = agg.pop()
        self.n_read = 2 if self.aggregate else self.K + 1

    @classmethod
    def batch(cls, configs: list[dict]) -> "GraphDynamical":
        return cls(configs[0], _stack=list(configs))

    def empty_state(self, B=None):
        B = self.B if B is None else B
        return {"q1": np.zeros((B, self.K)), "q2": np.zeros((B, self.J)), "o1": np.zeros((B, self.K)),
                "dem": np.zeros(B) + self.d_mean, "w": np.zeros((B, self.K)) + self.w_default,
                "delivered": np.zeros(B), "arrived": np.zeros(B), "read": np.zeros((B, self.n_read))}

    def draw(self, rng, B=1):
        return rng.standard_normal(self.n_normal * B).reshape(B, self.n_normal), np.zeros((B, 0))

    def batch_step(self, st: dict, z: np.ndarray, u: np.ndarray, t: int) -> dict:
        """One tick for every instance at once (state changed in place)."""
        dem = self.d_mean + self.d_ar * (st["dem"] - self.d_mean) + self.d_sd * z[:, 0]
        st["dem"] = dem
        D = np.maximum(dem, 0.0)
        w = np.maximum(st["w"], 0.0)
        ws = w.sum(axis=1)
        # route k carries the share w_k of demand; when the shares add up to more than one they are scaled down,
        # when to less, the rest of the demand is turned away (admission control)
        flow = D[:, None] * w / np.maximum(ws, 1.0)[:, None]
        q1 = st["q1"] + flow
        o1 = np.minimum(q1, self.c1)
        q1 = q1 - o1
        M = np.where((self.rewire_at >= 0)[:, None, None] & (t >= self.rewire_at)[:, None, None], self.M2, self.M1)
        in2 = (M * st["o1"][:, :, None]).sum(axis=1)
        q2 = st["q2"] + in2
        o2 = np.minimum(q2, self.c2)
        q2 = q2 - o2
        deliv = o2.sum(axis=1)
        arrived = st["arrived"] + flow.sum(axis=1)
        cg = prim("congestion")
        lat1, _ = cg.apply(_CONG, {"load": flow + q1, "cap": self.c1}, {}, None, None, t)
        load2 = in2 + q2
        lat2_j, _ = cg.apply(_CONG, {"load": load2, "cap": self.c2}, {}, None, None, t)
        lat2 = (M * lat2_j[:, None, :]).sum(axis=2)
        read = np.empty((dem.shape[0], self.K + 1))
        read[:, 0] = deliv + self.noise * 10.0 * z[:, 1]
        read[:, 1:] = lat1 + lat2 + self.noise[:, None] * z[:, 2:]
        if self.aggregate:
            fs = flow.sum(axis=1)
            mean_lat = np.where(fs > 0, (flow * read[:, 1:]).sum(axis=1) / np.where(fs > 0, fs, 1.0),
                                read[:, 1:].mean(axis=1))
            read = np.stack([read[:, 0], mean_lat], axis=1)
        st.update(q1=q1, q2=q2, o1=o1, delivered=st["delivered"] + deliv, arrived=arrived, read=read)
        rev = self.price * deliv
        cost = (self.toll * flow).sum(axis=1) + self.hold * (q1.sum(axis=1) + q2.sum(axis=1))
        return {"readings": read, "revenue": rev, "cost": cost, "delivered": deliv, "flow": flow}

    def public_levers(self):
        c = self.config
        return [{"name": n, "min": 0.0, "max": 1.0, "default": float(self.w_default[0, k]), "integer": False}
                for k, n in enumerate(c["route_names"])]

    def set_levers(self, st, values):
        names = self.config["route_names"]
        for k, v in values.items():
            st["w"][:, names.index(k)] = float(np.clip(float(v), 0.0, 1.0))

    # ================================================================== public surface
    def action_space(self, agent: str, state: dict | None = None) -> A.ActionSpace:
        c = self.config
        kinds, limits = {}, {}
        if self.sc == "routing":
            kinds["levers"] = A.levers_schema(self.public_levers())
        elif self.sc == "diffusion":
            kinds["allocate"] = A.allocate_schema(list(c["cluster_names"]), float(c["seed_budget"]), "seeds")
            limits["allocate_total"] = float(c["seed_budget"])
        elif self.sc == "cascade":
            kinds["allocate"] = A.allocate_schema(list(c["zone_names"]), float(c["reinforce_budget"]),
                                                  "reinforcement crews this tick")
            limits["allocate_total"] = float(c["reinforce_budget"])
            kinds["graph_edit"] = A.graph_edit_schema(["add_edge", "remove_edge"], list(c["node_names"]))
            limits["edge_ops"] = ("add_edge", "remove_edge")
        elif self.sc == "balance":
            i = self.agents.index(agent)
            kinds["levers"] = A.levers_schema([{"name": c["gen_names"][i], "min": 0.0, "max": float(c["gen_max"]),
                                                "default": 0.0}])
            nb = self.neighbours(i)
            kinds["allocate"] = A.allocate_schema([c["node_names"][j] for j in nb],
                                                  float(sum(self.link_cap(i, j) for j in nb)),
                                                  "power sent to each neighbour this tick")
            limits["allocate_total"] = float(sum(self.link_cap(i, j) for j in nb))
            kinds["message"] = A.message_schema([a for a in self.agents if a != agent])
        else:
            opts = ["none"] + list(c["edge_names"])
            kinds["discrete"] = A.discrete_schema({"repair": opts})
            limits["decisions"] = {"repair": opts}
            kinds["graph_edit"] = A.graph_edit_schema(["add_edge"], list(c["node_names"]))
            limits["edge_ops"] = ("add_edge",)
        if len(kinds) > 1:
            kinds["sequence"] = A.sequence_schema(sorted(kinds))
        return A.ActionSpace(kinds, limits)

    # ================================================================== state
    def init_state(self, rng):
        c = self.config
        if self.sc == "routing":
            return self.empty_state(1)
        if self.sc == "diffusion":
            return {"adopted": np.zeros(len(c["theta"])), "seeds_left": np.array([float(c["seed_total"])]),
                    "pending": np.zeros(len(c["cluster_names"])), "seeded": np.zeros(1)}
        if self.sc == "balance":
            N = self.N
            return {"x": np.zeros(N), "gen": np.zeros(N), "exp": np.zeros((N, N)), "served": np.zeros(1),
                    "demanded": np.zeros(1)}
        if self.sc == "cascade":
            P = len(c["cap"])
            return {"down": np.zeros(P), "backup": np.full(len(c["dep"]), -1.0), "reinf": np.zeros(len(c["zone_names"])),
                    "dem": np.ones(1), "edits": np.zeros(1), "cdown": np.zeros(len(c["dep"]))}
        E = len(c["edges"])
        return {"alive": np.ones(E), "extra": np.zeros((0, 2)), "extra_alive": np.zeros(0), "repairs": np.zeros(1),
                "pending_repair": np.full(1, -1.0), "built_seen": np.zeros(1)}

    def copy_state(self, state):
        return {k: np.array(v, copy=True) for k, v in state.items()}

    # ================================================================== actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        k = action["kind"]
        if self.sc == "routing" and k == "levers":
            self.set_levers(state, action["values"])
            return {"set": {n: float(state["w"][0, c["route_names"].index(n)]) for n in action["values"]}}
        if self.sc == "diffusion" and k == "allocate":
            amt = np.array([np.floor(float(action["amounts"].get(n, 0.0)) + 1e-9) for n in c["cluster_names"]])
            op = c.get("open_at")
            if op:                                        # a community that is not online yet takes no seeds
                amt = np.where(np.asarray(op, dtype=float) <= float(t), amt, 0.0)
            if amt.sum() > state["seeds_left"][0] + 1e-9:
                raise ActionError(f"only {int(state['seeds_left'][0])} seeds left")
            state["pending"] = state["pending"] + amt
            state["seeds_left"] = state["seeds_left"] - amt.sum()
            return {"seeds": {n: int(a) for n, a in zip(c["cluster_names"], amt)},
                    "seeds_left": int(state["seeds_left"][0])}
        if self.sc == "cascade" and k == "allocate":
            state["reinf"] = np.array([float(action["amounts"].get(n, 0.0)) for n in c["zone_names"]])
            return {"reinforcement": {n: float(a) for n, a in zip(c["zone_names"], state["reinf"])}}
        if self.sc == "cascade" and k == "graph_edit":
            names = c["node_names"]
            src, dst = names.index(action["src"]), names.index(action["dst"])
            P = len(c["cap"])
            if not (src < P <= dst):
                raise ActionError("a backup supply link goes from a power node to a control node")
            ci = dst - P
            if action["op"] == "add_edge":
                if src == c["dep"][ci]:
                    raise ActionError("that is already the node's primary supply")
                state["backup"][ci] = src
                state["edits"] = state["edits"] + 1
                return {"backup": {action["dst"]: action["src"]}, "cost": float(c["edit_cost"])}
            if state["backup"][ci] != src:
                raise ActionError("no such backup link")
            state["backup"][ci] = -1.0
            return {"removed": [action["src"], action["dst"]]}
        if self.sc == "balance":
            i = self.agents.index(agent)
            if k == "levers":
                state["gen"] = state["gen"].copy()
                state["gen"][i] = float(np.clip(action["values"].get(c["gen_names"][i], state["gen"][i]), 0.0,
                                                float(c["gen_max"])))
                return {"generation": float(state["gen"][i])}
            if k == "allocate":
                state["exp"] = state["exp"].copy()
                for j in self.neighbours(i):
                    v = float(action["amounts"].get(c["node_names"][j], 0.0))
                    state["exp"][i, j] = float(np.clip(v, 0.0, self.link_cap(i, j)))
                return {"exports": {c["node_names"][j]: float(state["exp"][i, j]) for j in self.neighbours(i)}}
        if self.sc == "repair" and k == "discrete":
            if action["choice"] == "none":
                return {"repair": None}
            e = c["edge_names"].index(action["choice"])
            if state["alive"][e] > 0.5:
                raise ActionError(f"{action['choice']} is not broken")
            state["pending_repair"] = np.array([float(e)])
            return {"repair": action["choice"]}
        if self.sc == "repair" and k == "graph_edit":
            names = c["node_names"]
            a, b = names.index(action["src"]), names.index(action["dst"])
            state["extra"] = np.concatenate([state["extra"], [[a, b]]], axis=0)
            state["extra_alive"] = np.concatenate([state["extra_alive"], [1.0]])
            return {"built": [action["src"], action["dst"]], "cost": float(c["build_cost"])}
        raise ActionError(f"{k} is not available in this world")

    # ================================================================== dynamics
    def advance(self, state, rng, t):
        return getattr(self, f"_adv_{self.sc}")(state, rng, t)

    def _adv_routing(self, st, rng, t):
        z, u = self.draw(rng)
        r = self.batch_step(st, z, u, t)
        c = self.config
        a = self.agents[0]
        nums = {c["gauge_names"][k]: float(r["readings"][0, k]) for k in range(self.n_read)}
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: float(r["revenue"][0])},
                "cost": {a: float(r["cost"][0])}, "info": {"delivered": float(r["delivered"][0])}}

    def _adv_diffusion(self, st, rng, t):
        c = self.config
        theta = np.asarray(c["theta"])
        cl = np.asarray(c["cluster"], dtype=int)
        adopted = st["adopted"].copy()
        seeded = 0
        for k in range(len(c["cluster_names"])):
            n = int(st["pending"][k])
            if n <= 0:
                continue
            cand = [i for i in c["by_degree"][k] if adopted[i] < 0.5][:n]
            adopted[cand] = 1.0
            seeded += len(cand)
        st["pending"] = np.zeros_like(st["pending"])
        frac = (self.adj @ adopted) / self.deg
        th = prim("threshold")
        newly, _ = th.apply({"theta": 0.0, "lo": 0.0, "hi": 1.0}, {"x": frac - theta}, {}, None, None, t)
        churn = rng.random(len(theta)) < c["churn"]
        adopted = np.where(adopted > 0.5, np.where(churn, 0.0, 1.0), newly)
        op = c.get("open_at")
        online = np.asarray([float(op[int(k)]) <= float(t) for k in cl]) if op else np.ones(len(theta), dtype=bool)
        adopted = np.where(online, adopted, 0.0)          # communities come online at hidden times
        st["adopted"] = adopted
        st["seeded"] = st["seeded"] + seeded
        n_ad = float(adopted.sum())
        ne = prim("network_effect")
        val, _ = ne.apply({"strength": c["strength"]}, {"x": np.array([n_ad]), "base": np.array([n_ad / len(theta)])},
                          {}, None, None, t)
        rev = float(c["value"] * val[0])
        cost = float(c["seed_cost"] * seeded)
        counts = [float(adopted[cl == k].sum()) for k in range(len(c["cluster_names"]))]
        ents = [{"kind": "community", "id": n, "adopters": counts[k],
                 "members": int((cl == k).sum()) if (not op or float(op[k]) <= float(t)) else 0}
                for k, n in enumerate(c["cluster_names"])]
        a = self.agents[0]
        return {"obs": {a: {"numbers": {c["gauge_names"][0]: n_ad}, "entities": ents,
                            "resources": {"seeds_left": float(st["seeds_left"][0])}}},
                "revenue": {a: rev}, "cost": {a: cost}, "info": {"seeded": seeded}}

    # ---------------------------------------------------------------- balance (8.7.5)
    def neighbours(self, i: int) -> list[int]:
        return sorted({b for a, b, _ in self.config["links"] if a == i} | {a for a, b, _ in self.config["links"]
                                                                          if b == i})

    def link_cap(self, i: int, j: int) -> float:
        for a, b, cap in self.config["links"]:
            if {a, b} == {i, j}:
                return float(cap)
        return 0.0

    def demand(self, st, i: int) -> float:
        return max(0.0, float(self.config["d_mean"][i]) + float(st["x"][i]))

    def gen_cost(self, i: int, g: float) -> float:
        c = self.config
        return float(c["c_lin"][i]) * g + float(c["c_quad"][i]) * g * g

    def _adv_balance(self, st, rng, t):
        c = self.config
        N = self.N
        z = rng.standard_normal(N)
        # power available at each node: its generation minus what it sends plus what arrives (after line losses);
        # a node cannot send more than it generates
        gen = st["gen"].copy()
        exp = st["exp"].copy()
        for i in range(N):
            out = exp[i].sum()
            if out > gen[i] + 1e-12:
                exp[i] *= gen[i] / out if out > 0 else 0.0
        avail = np.maximum(0.0, gen - exp.sum(axis=1) + (1.0 - float(c["loss"])) * exp.sum(axis=0))
        obs, rev, cost = {}, {}, {}
        tot_served = tot_dem = 0.0
        for i, a in enumerate(self.agents):
            d = self.demand(st, i)
            served = min(d, avail[i])
            short = d - served
            tot_served += served
            tot_dem += d
            rev[a] = float(c["price"]) * served
            cost[a] = self.gen_cost(i, float(gen[i])) + float(c["shortage_cost"]) * short
            obs[a] = {"numbers": {"my_demand": d, "my_served": served, "my_shortfall": short,
                                  "my_generation": float(gen[i]),
                                  "power_received": float((1.0 - float(c["loss"])) * exp[:, i].sum()),
                                  "my_cost_at_half": self.gen_cost(i, 0.5 * float(c["gen_max"])) / (0.5 * float(c["gen_max"])),
                                  "my_cost_at_full": self.gen_cost(i, float(c["gen_max"])) / float(c["gen_max"])}}
        st["served"] = st["served"] + tot_served
        st["demanded"] = st["demanded"] + tot_dem
        phi = float(np.clip(c["phi"], 0.0, 0.99))
        st["x"] = phi * st["x"] + float(c["d_sd"]) * np.sqrt(1 - phi ** 2) * z
        st["exp"] = np.zeros((N, N))                   # exports are set tick by tick
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    def propagate(self, load, cap, alive0, backup):
        """Failures within the power layer (load shed onto alive neighbours) and then between layers (a control node
        without supply fails; a grid node without its control node fails), until nothing changes. Returns which
        grid nodes are still alive."""
        c = self.config
        P = len(c["cap"])
        dep = np.asarray(c["dep"], dtype=int)
        ctrl = np.asarray(c["ctrl"], dtype=int)
        ld = np.where(alive0, load, 0.0)[None, :]
        adj = self.adj * alive0[None, :] * alive0[:, None]
        failed = cascade_fn(ld, np.where(alive0, cap, np.inf)[None, :], adj, share=c["shed_share"])[0] > 0.5
        p_alive = alive0 & ~failed
        for _ in range(P + len(dep)):
            c_alive = p_alive[dep] | ((backup >= 0) & p_alive[np.maximum(backup, 0)])
            lost = p_alive & ~c_alive[ctrl]
            if not lost.any():
                break
            p_alive = p_alive & ~lost
        return p_alive

    def _adv_cascade(self, st, rng, t):
        c = self.config
        P = len(c["cap"])
        dep = np.asarray(c["dep"], dtype=int)
        zone = np.asarray(c["zone"], dtype=int)
        z = rng.standard_normal(1)
        uu = rng.random(2)
        dem = 1.0 + c["dem_ar"] * (st["dem"][0] - 1.0) + c["dem_sd"] * z[0]
        st["dem"] = np.array([dem])
        load = np.asarray(c["base"]) * max(dem, 0.0)
        spike = int(uu[1] * P) if uu[0] < c["p_shock"] else -1
        if spike >= 0:
            load = load.copy()
            load[spike] += c["shock"]
        cap = np.asarray(c["cap"]) + c["reinf_gain"] * st["reinf"][zone]
        down = st["down"].copy()
        alive0 = down <= 0
        backup = st["backup"].astype(int)
        p_alive = self.propagate(load, cap, alive0, backup)
        newly = alive0 & ~p_alive
        down = np.where(newly, float(c["repair_ticks"]), np.maximum(down - 1.0, 0.0))
        st["down"] = down
        c_alive = p_alive[dep] | ((backup >= 0) & p_alive[np.maximum(backup, 0)])
        st["cdown"] = (~c_alive).astype(float)
        served = float(np.where(p_alive, load, 0.0).sum())
        rev = c["price"] * served
        cost = c["reinf_cost"] * float(st["reinf"].sum()) + c["edit_cost"] * float(st["edits"][0]) \
            + c["backup_cost"] * float((backup >= 0).sum())
        st["edits"] = np.zeros(1)
        zones = [float(p_alive[zone == k].mean()) for k in range(len(c["zone_names"]))]
        ent = [{"kind": "supply", "power": c["node_names"][int(b)], "control": c["node_names"][P + i]}
               for i, b in enumerate(backup) if b >= 0]
        a = self.agents[0]
        return {"obs": {a: {"numbers": {c["gauge_names"][0]: served, c["gauge_names"][1]: float(newly.sum())},
                            "arrays": {"alive_by_zone": zones}, "entities": ent}},
                "revenue": {a: float(rev)}, "cost": {a: float(cost)}, "info": {"failed": int(newly.sum())}}

    def _adv_repair(self, st, rng, t):
        c = self.config
        E = len(c["edges"])
        cost = 0.0
        e = int(st["pending_repair"][0])
        if e >= 0:
            st["alive"] = st["alive"].copy()
            st["alive"][e] = 1.0
            st["repairs"] = st["repairs"] + 1
            cost += c["repair_cost"]
            st["pending_repair"] = np.full(1, -1.0)
        nx = st["extra"].shape[0]
        cost += c["build_cost"] * float(nx - int(st["built_seen"][0]))
        st["built_seen"] = np.array([float(nx)])
        uu = rng.random(E + nx)
        haz = np.asarray(c["hazard"])
        st["alive"] = np.where((st["alive"] > 0.5) & (uu[:E] < haz), 0.0, st["alive"])
        if nx:
            st["extra_alive"] = np.where((st["extra_alive"] > 0.5) & (uu[E:] < c["extra_hazard"]), 0.0,
                                         st["extra_alive"])
        comp = _components(self.N, [ed for k, ed in enumerate(c["edges"]) if st["alive"][k] > 0.5] +
                           [tuple(int(v) for v in st["extra"][k]) for k in range(nx) if st["extra_alive"][k] > 0.5])
        served = float(sum(d for (a, b), d in zip(c["pairs"], c["pair_demand"]) if comp[a] == comp[b]))
        rev = c["price"] * served
        ent = [{"kind": "link", "id": c["edge_names"][k], "ends": [c["node_names"][a], c["node_names"][b]],
                "up": bool(st["alive"][k] > 0.5)} for k, (a, b) in enumerate(c["edges"])]
        ent += [{"kind": "link", "id": f"new{k}", "ends": [c["node_names"][int(st["extra"][k, 0])],
                                                         c["node_names"][int(st["extra"][k, 1])]],
                 "up": bool(st["extra_alive"][k] > 0.5)} for k in range(nx)]
        a = self.agents[0]
        return {"obs": {a: {"numbers": {c["gauge_names"][0]: served,
                                        c["gauge_names"][1]: float(E - st["alive"].sum())}, "entities": ent}},
                "revenue": {a: float(rev)}, "cost": {a: float(cost)}, "info": {}}

    # ================================================================== evaluator side
    def truth(self):
        c = self.config
        if self.sc == "routing":
            edges = [{"src": f"r{k}", "dst": f"h{j}", "lag": 1, "sign": 1, "kind": "transit"}
                     for k, j in enumerate(c["assign"])]
            nodes = [{"id": f"r{k}", "cap": c["c1"][k]} for k in range(self.K)] + \
                [{"id": f"h{j}", "cap": c["c2"][j]} for j in range(self.J)]
            hidden = {"bottleneck_capacity": c["c2"], "assign": c["assign"], "rewire_at": c.get("rewire_at", -1)}
            regimes = [{"at": c["rewire_at"], "what": "second-hop assignment changes"}] if c.get("rewire_at", -1) >= 0 \
                else []
        elif self.sc == "diffusion":
            nodes = [{"id": i, "theta": c["theta"][i], "cluster": c["cluster"][i]} for i in range(len(c["theta"]))]
            edges = [{"src": i, "dst": j} for i in range(len(nodes)) for j in range(len(nodes)) if c["adj"][i][j]]
            hidden, regimes = {}, []
        elif self.sc == "cascade":
            P = len(c["cap"])
            nodes = [{"id": i, "cap": c["cap"][i], "base": c["base"][i]} for i in range(P)]
            edges = [{"src": i, "dst": j, "kind": "grid"} for i in range(P) for j in range(P) if c["adj"][i][j]]
            edges += [{"src": int(p), "dst": P + i, "kind": "supply"} for i, p in enumerate(c["dep"])]
            edges += [{"src": P + int(ci), "dst": i, "kind": "control"} for i, ci in enumerate(c["ctrl"])]
            hidden, regimes = {}, []
        elif self.sc == "balance":
            nodes = [{"id": i, "c_lin": c["c_lin"][i], "c_quad": c["c_quad"][i]} for i in range(self.N)]
            edges = [{"src": a, "dst": b, "cap": cap} for a, b, cap in c["links"]]
            hidden, regimes = {"d_mean": c["d_mean"]}, []
        else:
            nodes = [{"id": i} for i in range(self.N)]
            edges = [{"src": a, "dst": b, "hazard": h} for (a, b), h in zip(c["edges"], c["hazard"])]
            hidden, regimes = {"hazard": c["hazard"]}, []
        return {"engine": self.kind, "scenario": self.sc, "nodes": nodes, "edges": edges, "hidden": hidden,
                "regimes": regimes, "motifs": self.motifs(), "graph": self.graph_stats()}

    def graph_stats(self) -> dict:
        """Size, density, clustering and number of connected components of the network (evaluator)."""
        c = self.config
        if self.sc == "routing":
            n = self.K + self.J + 2
            m = 2 * self.K + self.J
            return {"nodes": n, "edges": m, "shared_hops": int(self.K - len(set(c["assign"]))), "clustering": 0.0}
        if self.sc in ("diffusion", "cascade"):
            A_ = np.asarray(c["adj"], dtype=float)
            n = A_.shape[0]
            tri = float(np.trace(A_ @ A_ @ A_))
            deg = A_.sum(axis=1)
            trip = float((deg * (deg - 1)).sum())
            return {"nodes": n, "edges": int(A_.sum() / 2), "clustering": tri / trip if trip else 0.0}
        if self.sc == "balance":
            return {"nodes": self.N, "edges": len(c["links"]), "clustering": 0.0}
        return {"nodes": self.N, "edges": len(c["edges"]), "clustering": 0.0}

    def motifs(self):
        c = self.config
        if self.sc == "routing":
            # latency rises with load; routes compete for shared second hops; flow is in transit for a tick and
            # waits in queues (stocks); the hidden demand process drives every reading at once
            m = {"congestion", "resource_competition", "delay", "stock", "hidden_confounding"}
            if c.get("rewire_at", -1) >= 0:
                m.add("regime_change")
        elif self.sc == "diffusion":
            m = {"threshold", "network_effect", "partial_observability"}
            if c["churn"] > 0:
                m.add("decay")
        elif self.sc == "cascade":
            m = {"cascading_failure", "resource_competition", "delay"}
        elif self.sc == "balance":
            # private fluctuating demand at each node, convex generation cost, transfers over capacity-limited lossy
            # links: the participants' choices interact through the shared network
            m = {"resource_competition", "strategic_interaction", "partial_observability", "saturation",
                 "hidden_confounding"}
        else:
            # service between a pair is all or nothing on connectivity; a repair is worth something only when it
            # reconnects a pair given the other failures
            m = {"threshold", "state_dependent_effect"}
        return sorted(m)

    def observation_channels(self):
        c = self.config
        if self.sc == "routing":
            # route latencies (or, with aggregate_only, only the flow-weighted mean over the routes) carry noise
            return sorted({"noisy"} | ({"aggregated"} if c.get("aggregate_only") else set()))
        return []

    def outcome(self, state, ledger):
        if self.sc == "balance":                       # a team objective: the whole grid's net income
            rev = sum(ledger["revenue"].values())
            cost = sum(ledger["cost"].values())
            return {"net": float(rev - cost), "value": float(rev - cost), "revenue": float(rev), "cost": float(cost),
                    "per_agent": {a: float(ledger["revenue"][a] - ledger["cost"][a]) for a in self.agents},
                    "served_share": float(state["served"][0] / max(state["demanded"][0], 1e-9))}
        a = self.agents[0]
        net = float(ledger["revenue"][a] - ledger["cost"][a])
        out = {"net": net, "value": net, "revenue": float(ledger["revenue"][a]), "cost": float(ledger["cost"][a])}
        if self.sc == "routing":
            out["delivered"] = float(state["delivered"][0])
        if self.sc == "diffusion":
            out["adopters"] = float(state["adopted"].sum())
        return out

    def conservation(self, before, after, out):
        errs = []
        for a in self.agents:
            if out["revenue"].get(a, 0.0) < 0 or out["cost"].get(a, 0.0) < 0:
                errs.append("negative money flow")
        if self.sc == "balance" and after["served"][0] > after["demanded"][0] + 1e-6:
            errs.append("more power served than demanded")
        if self.sc == "routing":
            inside = after["q1"].sum() + after["q2"].sum() + after["o1"].sum()
            bal = after["arrived"][0] - after["delivered"][0] - inside
            if abs(bal) > 1e-6 * max(1.0, after["arrived"][0]):
                errs.append(f"flow not conserved (imbalance {bal:.3g})")
            if (after["q1"] < -1e-12).any() or (after["q2"] < -1e-12).any():
                errs.append("negative queue")
        if self.sc == "diffusion":
            if after["seeds_left"][0] < -1e-9 or after["seeds_left"][0] > before["seeds_left"][0] + 1e-9:
                errs.append("seed budget not conserved")
        return errs


def _components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def components(n: int, edges) -> list[int]:
    return _components(n, list(edges))
