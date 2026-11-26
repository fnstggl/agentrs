"""Engine `planning`: planning and resource systems (Part 7.3 "planning and resource systems"; Part 8.4, 8.9.1,
8.9.4).

The world is a ledger of discrete units moving through a planned system in periods: stocks at locations, orders in
pipelines with replenishment lags, lots that age and perish, production that consumes inputs in fixed proportions,
capacity that is bought in lumps and arrives after a build lag, staff spread over projects with diminishing returns,
and shared machines booked by several participants. Each tick has one planning period: the participants' orders,
shipments, production plans, bookings and investments take effect, then the period's exogenous draws (demand,
reports) are applied, units move, and money is booked.

Unlike the continuous-time event queue of `discrete_event`, time here is the planning period and every quantity is
an integer number of units with an exact unit ledger (what entered = what is held + in transit + what left), checked
every tick by `conservation`. Every random draw of a period is taken up front with a fixed shape (inverse-CDF Poisson
from uniforms), so two copies of a world that differ only in decisions see exactly the same demand and noise.

Scenarios (one per family):

    distribution  a depot replenished after a hidden lead time ships stock to sites whose hidden demand rates shift
                  at hidden times (local regimes: a site heats up or goes quiet); sales are censored by stock
                  (inventory allocation with delayed replenishment, 8.4.1)
    perishable    daily production of a good with a hidden shelf life; customers who take only fresh units,
                  flexible ones and bargain hunters drawn by a markdown; issuing order fifo/lifo (8.4.2)
    supply_chain  a serial chain of stages with hidden shipping and production lags and backlogs; customer demand
                  shifts at a hidden time (multi-stage supply chain, 8.4.3)
    capacity      lumpy, irreversible capacity bought before demand of a hidden growth scenario is known; build lag,
                  maintenance, salvage on retirement (capacity investment under uncertain demand, 8.4.4)
    complements   products made from several inputs in hidden fixed proportions (Leontief); inputs ordered with
                  lead times, machine hours allocated to products (complementary resource requirements, 8.4.5)
    portfolio     staff allocated over projects with hidden work content, productivity, diminishing returns and
                  duds revealed at a milestone; payment on completion, decaying after deadlines (8.4.6)
    assembly      two participants with private, complementary production costs feed one assembly line; output
                  needs both components (complementary production capabilities, 8.9.1)
    schedule      three participants with private job queues share two machines; double bookings waste the period,
                  joint plans reserve machines (multi-agent scheduling, 8.9.4)
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .. import actions as A
from .base import ActionError, RapidEngine, register

SCENARIOS = ("distribution", "perishable", "supply_chain", "capacity", "complements", "portfolio", "assembly",
             "schedule")
_ND = NormalDist()


def pois(u: float, lam: float) -> int:
    """Inverse-CDF Poisson draw from one uniform (a fixed number of random numbers per draw, whatever `lam`)."""
    lam = float(lam)
    if lam <= 0.0:
        return 0
    if lam > 80.0:
        return max(0, int(round(lam + math.sqrt(lam) * _ND.inv_cdf(min(max(u, 1e-12), 1 - 1e-12)))))
    p = math.exp(-lam)
    c, k = p, 0
    while u > c and k < 1000:
        k += 1
        p *= lam / k
        c += p
    return k


def _i(x) -> int:
    return int(max(0, math.floor(float(x) + 1e-9)))


@register
class Planning(RapidEngine):
    kind = "planning"
    SECRET_KEYS = ("lam", "lam_sched", "lead", "shelf", "lam_bar", "amp", "phase", "fresh_frac", "bargain", "lags", "mu0", "mu1", "mu2", "t_change2",
                   "t_change", "path", "t_jump", "build_lag", "recipe", "dem", "work", "prod", "beta", "dud",
                   "mu_amp", "mu_phase", "mu_mean", "true_time", "speed")

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.sc}")

    # ================================================================== public surface
    def action_space(self, agent: str, state: dict | None = None) -> A.ActionSpace:
        c = self.config
        kinds: dict = {}
        limits: dict = {}
        q = {"tools": "the action kinds available to you", "history": "your last observations (args: n)",
             "spec": "the public description of this world"}
        sc = self.sc
        if sc == "distribution":
            kinds["levers"] = A.levers_schema([{"name": c["order_name"], "min": 0.0, "max": float(c["max_order"]),
                                                "default": 0.0, "integer": True}])
            kinds["allocate"] = A.allocate_schema(c["site_names"], float(c["ship_cap"]), "units shipped from the depot")
            limits["allocate_total"] = float(c["ship_cap"])
        elif sc == "perishable":
            kinds["levers"] = A.levers_schema([
                {"name": c["bake_name"], "min": 0.0, "max": float(c["max_bake"]), "default": 0.0, "integer": True},
                {"name": c["markdown_name"], "min": 0.0, "max": 0.5, "default": 0.0}])
            kinds["discrete"] = A.discrete_schema({"issue": ["fifo", "lifo"]})
            limits["decisions"] = {"issue": ["fifo", "lifo"]}
        elif sc == "supply_chain":
            kinds["levers"] = A.levers_schema([{"name": n, "min": 0.0, "max": float(c["max_order"]), "default": 0.0,
                                                "integer": True} for n in c["order_names"]])
            kinds["program"] = A.program_schema(int(c["horizon"]), "Programs may return levers actions.")
            limits["program_kinds"] = ["levers"]
        elif sc == "capacity":
            opts = ["none", "small", "large", "retire"]
            kinds["discrete"] = A.discrete_schema({"build": opts})
            limits["decisions"] = {"build": opts}
        elif sc == "complements":
            kinds["levers"] = A.levers_schema([{"name": n, "min": 0.0, "max": float(c["max_order"]), "default": 0.0,
                                                "integer": True} for n in c["order_names"]])
            kinds["allocate"] = A.allocate_schema(c["prod_names"], float(c["hours"]), "machine hours per period")
            limits["allocate_total"] = float(c["hours"])
        elif sc == "portfolio":
            kinds["allocate"] = A.allocate_schema(c["proj_names"], float(c["staff"]), "staff")
            limits["allocate_total"] = float(c["staff"])
            q["audit"] = (f"an exact count of a project's completed work, in staff-periods (args: {{\"project\": "
                          f"name}}); costs {c['audit_cost']}")
        elif sc == "assembly":
            kinds["levers"] = A.levers_schema([{"name": n, "min": 0.0, "max": float(c["cap"]), "default": 0.0,
                                                "integer": True} for n in c["make_names"]])
            others = [a for a in self.agents if a != agent]
            kinds["message"] = A.message_schema(others)
        elif sc == "schedule":
            opts = ["idle"] + list(c["machine_names"])
            kinds["discrete"] = A.discrete_schema({"book": opts})
            limits["decisions"] = {"book": opts}
            others = [a for a in self.agents if a != agent]
            kinds["message"] = A.message_schema(others)
            plan = {m: A.S.s_enum(["none"] + list(self.agents), f"who runs machine {m} this period")
                    for m in c["machine_names"]}
            kinds["coordinate"] = A.coordinate_schema(others, plan)
            limits["parties"] = others
        kinds["query"] = A.query_schema(q)
        if sc in ("distribution", "perishable", "complements", "portfolio", "schedule"):
            kinds["sequence"] = A.sequence_schema(sorted(k for k in kinds if k not in ("sequence", "program")))
        return A.ActionSpace(kinds, limits)

    # Private information of one participant (own unit costs, own jobs) is never put in `brief`: the public spec
    # carries every participant's brief. It reaches only its owner, in the owner's observations.

    # ================================================================== state
    def init_state(self, rng) -> dict:
        c = self.config
        sc = self.sc
        if sc == "distribution":
            M = len(c["site_names"])
            st = {"depot": float(c["init_depot"]), "sites": [float(x) for x in c["init_sites"]], "pipe": [],
                  "transit": [], "pend_cost": 0.0, "pend_rev": 0.0}
            st["units0"] = st["depot"] + sum(st["sites"])
            st["in"], st["out"] = 0.0, 0.0
            st["M"] = M
            return st
        if sc == "perishable":
            lots = [0.0] * 6
            lots[0] = float(c["init_stock"])
            return {"lots": lots, "issue": 0, "markdown": 0.0, "pend_cost": 0.0, "units0": float(c["init_stock"]),
                    "in": 0.0, "out": 0.0}
        if sc == "supply_chain":
            K = len(c["stage_names"])
            st = {"inv": [float(x) for x in c["init_stock"]], "backlog": [0.0] * K, "pipe": [], "orders": [0.0] * K,
                  "pend_cost": 0.0, "in": 0.0, "out": 0.0}
            st["units0"] = sum(st["inv"])
            return st
        if sc == "capacity":
            return {"cap": float(c["cap0"]), "pending": [], "pend_cost": 0.0, "pend_rev": 0.0, "built": 0.0,
                    "retired": 0.0}
        if sc == "complements":
            R, P = len(c["res_names"]), len(c["prod_names"])
            return {"res": [float(x) for x in c["init_res"]], "fg": [0.0] * P, "pipe": [], "plan": [0.0] * P,
                    "pend_cost": 0.0, "res_in": 0.0, "res_used": 0.0, "fg_in": 0.0, "fg_out": 0.0,
                    "res0": float(sum(c["init_res"])), "R": R}
        if sc == "portfolio":
            N = len(c["proj_names"])
            return {"work": [0.0] * N, "status": [0] * N, "alloc": [0.0] * N, "pend_cost": 0.0, "paid": [0.0] * N}
        if sc == "assembly":
            return {"buf": [0.0, 0.0], "make": [[0.0, 0.0], [0.0, 0.0]], "pend_cost": [0.0, 0.0], "in": 0.0,
                    "out": 0.0}
        # schedule
        nA = len(self.agents)
        return {"work": [[0.0] * len(c["jobs"][i]) for i in range(nA)],
                "done": [[0] * len(c["jobs"][i]) for i in range(nA)], "book": [-1] * nA, "plan": None,
                "pend_cost": [0.0] * nA, "last_use": [-1] * len(c["machine_names"])}

    # ================================================================== actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        k = action["kind"]
        sc = self.sc
        if k == "query" and action["target"] == "audit" and sc == "portfolio":
            name = str((action.get("args") or {}).get("project", ""))
            if name not in c["proj_names"]:
                raise ActionError("audit needs args {\"project\": <project name>}")
            j = c["proj_names"].index(name)
            state["pend_cost"] += float(c["audit_cost"])
            return {"project": name, "work_done": round(float(state["work"][j]), 6),
                    "status": ["active", "completed", "abandoned"][state["status"][j]], "cost": float(c["audit_cost"])}
        if sc == "distribution":
            if k == "levers":
                q = _i(action["values"].get(c["order_name"], 0.0))
                if q > 0:
                    state["pipe"].append([t + int(c["lead"]), float(q)])
                    state["pend_cost"] += q * float(c["unit_cost"])
                    state["in"] += q
                return {"ordered": q}
            if k == "allocate":
                req = [_i(action["amounts"].get(n, 0.0)) for n in c["site_names"]]
                tot = sum(req)
                avail = int(state["depot"])
                if tot > avail and tot > 0:
                    req = [int(math.floor(r * avail / tot)) for r in req]
                    tot = sum(req)
                state["depot"] -= tot
                for i, r in enumerate(req):
                    if r > 0:
                        state["transit"].append([t + 1, i, float(r)])
                state["pend_cost"] += tot * float(c["ship_cost"])
                return {"shipped": {n: r for n, r in zip(c["site_names"], req)}}
        if sc == "perishable":
            if k == "levers":
                out = {}
                if c["bake_name"] in action["values"]:
                    q = _i(action["values"][c["bake_name"]])
                    state["lots"][0] += q
                    state["pend_cost"] += q * float(c["unit_cost"])
                    state["in"] += q
                    out["made"] = q
                if c["markdown_name"] in action["values"]:
                    state["markdown"] = float(action["values"][c["markdown_name"]])
                    out["markdown"] = state["markdown"]
                return out
            if k == "discrete":
                state["issue"] = 0 if action["choice"] == "fifo" else 1
                return {"issue": action["choice"]}
        if sc == "supply_chain" and k == "levers":
            for j, n in enumerate(c["order_names"]):
                if n in action["values"]:
                    state["orders"][j] = float(_i(action["values"][n]))
            return {"orders": {n: state["orders"][j] for j, n in enumerate(c["order_names"])}}
        if sc == "capacity" and k == "discrete":
            ch = action["choice"]
            s = float(c["unit_block"])
            if ch == "small":
                state["pending"].append([t + int(c["build_lag"]), s])
                state["pend_cost"] += s * float(c["cost_small"])
                state["built"] += s
            elif ch == "large":
                state["pending"].append([t + int(c["build_lag"]), 3 * s])
                state["pend_cost"] += 3 * s * float(c["cost_large"])
                state["built"] += 3 * s
            elif ch == "retire":
                if state["cap"] < s:
                    raise ActionError("not enough installed capacity to retire a block")
                state["cap"] -= s
                state["pend_rev"] += s * float(c["salvage"])
                state["retired"] += s
            return {"build": ch}
        if sc == "complements":
            if k == "levers":
                out = {}
                for i, n in enumerate(c["order_names"]):
                    if n in action["values"]:
                        q = _i(action["values"][n])
                        if q > 0:
                            state["pipe"].append([t + int(c["lead_time"][i]), i, float(q)])
                            state["pend_cost"] += q * float(c["res_cost"][i])
                            state["res_in"] += q
                        out[n] = q
                return {"ordered": out}
            if k == "allocate":
                state["plan"] = [float(action["amounts"].get(n, 0.0)) for n in c["prod_names"]]
                return {"hours": {n: h for n, h in zip(c["prod_names"], state["plan"])}}
        if sc == "portfolio" and k == "allocate":
            state["alloc"] = [float(action["amounts"].get(n, 0.0)) for n in c["proj_names"]]
            return {"staffing": {n: x for n, x in zip(c["proj_names"], state["alloc"])}}
        if sc == "assembly" and k == "levers":
            i = self.agents.index(agent)
            x = [float(_i(action["values"].get(n, state["make"][i][j]))) for j, n in enumerate(c["make_names"])]
            tot = sum(x)
            if tot > c["cap"]:
                x = [float(math.floor(v * c["cap"] / tot)) for v in x]
            state["make"][i] = x
            return {"making": {n: v for n, v in zip(c["make_names"], x)}}
        if sc == "schedule" and k == "discrete":
            i = self.agents.index(agent)
            ch = action["choice"]
            state["book"][i] = -1 if ch == "idle" else c["machine_names"].index(ch)
            return {"booked": ch}
        raise ActionError(f"{k} is not available in this world")

    def coordinate(self, state, plan, parties, rng, t):
        if self.sc != "schedule":
            raise ActionError("this world has no joint plans")
        if sorted(parties) != sorted(self.agents):
            raise ActionError("a machine plan needs every participant")
        c = self.config
        used = [plan[m] for m in c["machine_names"] if plan[m] != "none"]
        if len(used) != len(set(used)):
            raise ActionError("a participant can run only one machine at a time")
        state["plan"] = {m: plan[m] for m in c["machine_names"]}
        return {"reserved": dict(state["plan"])}

    # ================================================================== dynamics
    def advance(self, st, rng, t):
        return getattr(self, "_adv_" + self.sc)(st, rng, t)

    def _adv_distribution(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        M = st["M"]
        u = rng.random(M)
        arrived = sum(q for ta, q in st["pipe"] if ta == t)
        st["pipe"] = [x for x in st["pipe"] if x[0] != t]
        st["depot"] += arrived
        recv = [0.0] * M
        for ta, i, q in st["transit"]:
            if ta == t:
                st["sites"][i] += q
                recv[i] += q
        st["transit"] = [x for x in st["transit"] if x[0] != t]
        sold, short = [], []
        for i in range(M):
            d = pois(float(u[i]), self.site_rate(i, t))
            s = min(float(d), st["sites"][i])
            st["sites"][i] -= s
            sold.append(s)
            short.append(1.0 if d > s else 0.0)
        st["out"] += sum(sold)
        rev = float(c["price"]) * sum(sold)
        cost = st["pend_cost"] + float(c["hold_depot"]) * st["depot"] + float(c["hold_site"]) * sum(st["sites"])
        st["pend_cost"] = 0.0
        nums = {"depot_stock": st["depot"], "on_order": float(sum(q for _, q in st["pipe"])),
                "depot_received": float(arrived)}
        for i, n in enumerate(c["site_names"]):
            nums[f"{n}.stock"] = st["sites"][i]
            nums[f"{n}.sold"] = sold[i]
            nums[f"{n}.stockout"] = short[i]
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def _adv_perishable(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        u = rng.random(3)
        lam = float(c["lam_bar"]) * (1.0 + float(c["amp"]) * math.sin(2 * math.pi * (t + float(c["phase"])) / 7.0))
        d = st["markdown"]
        n_fresh = pois(float(u[0]), lam * float(c["fresh_frac"]))
        n_flex = pois(float(u[1]), lam * (1.0 - float(c["fresh_frac"])))
        n_barg = pois(float(u[2]), float(c["lam_bar"]) * float(c["bargain"]) * d)
        lots = st["lots"]
        p = float(c["unit_price"])
        rev = 0.0
        sold_fresh = sold_aged = 0.0
        x = min(float(n_fresh), lots[0])
        lots[0] -= x
        sold_fresh += x
        rev += p * x
        order = list(range(5, -1, -1)) if st["issue"] == 0 else list(range(6))
        need = float(n_flex)
        for age in order:
            if need <= 0:
                break
            x = min(need, lots[age])
            lots[age] -= x
            need -= x
            if age == 0:
                sold_fresh += x
                rev += p * x
            else:
                sold_aged += x
                rev += p * (1.0 - d) * x
        need = float(n_barg)
        for age in range(5, 0, -1):
            if need <= 0:
                break
            x = min(need, lots[age])
            lots[age] -= x
            need -= x
            sold_aged += x
            rev += p * (1.0 - d) * x
        S = int(c["shelf"])
        wasted = float(sum(lots[S - 1:]))                  # units on their last day that did not sell expire
        st["lots"] = [0.0] + lots[:S - 1] + [0.0] * (6 - S)
        st["out"] += sold_fresh + sold_aged + wasted
        cost = st["pend_cost"] + float(c["disposal_cost"]) * wasted + float(c["hold_cost"]) * sum(st["lots"])
        st["pend_cost"] = 0.0
        nums = {"weekday": float((t + 1) % 7), "sold_fresh": sold_fresh, "sold_older": sold_aged,
                "discarded": wasted, "carried_over": float(sum(st["lots"]))}
        return {"obs": {a: {"numbers": nums, "arrays": {"stock_by_age": [float(v) for v in st["lots"]]}}},
                "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def _adv_supply_chain(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        K = len(c["stage_names"])
        u = rng.random(1)
        mu = self.chain_mu(t)
        D = float(pois(float(u[0]), mu))
        recv = [0.0] * K
        keep = []
        for ta, k, q in st["pipe"]:
            if ta == t:
                st["inv"][k] += q
                recv[k] += q
            else:
                keep.append([ta, k, q])
        st["pipe"] = keep
        shipped = [0.0] * K
        s = min(st["inv"][0], D)                          # customers who find no stock are lost
        lost = D - s
        st["inv"][0] -= s
        shipped[0] = s
        st["out"] += s
        rev = float(c["unit_price"]) * s
        orders = list(st["orders"])
        for k in range(1, K):
            owed = st["backlog"][k] + orders[k - 1]
            s = min(st["inv"][k], owed)
            st["inv"][k] -= s
            st["backlog"][k] = owed - s
            shipped[k] = s
            if s > 0:
                st["pipe"].append([t + int(c["lags"][k - 1]), k - 1, s])
        qf = orders[K - 1]
        if qf > 0:
            st["pipe"].append([t + int(c["lags"][K - 1]), K - 1, qf])
            st["in"] += qf
        cost = st["pend_cost"] + float(c["raw_cost"]) * qf + float(c["lost_cost"]) * lost + \
            sum(float(h) * x for h, x in zip(c["hold"], st["inv"]))
        st["pend_cost"] = 0.0
        st["orders"] = [0.0] * K
        nums = {"customer_orders": D, "lost_sales": lost}
        for k, n in enumerate(c["stage_names"]):
            nums[f"{n}.on_hand"] = st["inv"][k]
            nums[f"{n}.backlog"] = st["backlog"][k]
            nums[f"{n}.received"] = recv[k]
            nums[f"{n}.shipped"] = shipped[k]
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def chain_mu(self, t: int) -> float:
        """Mean customer demand of period t in the supply-chain world (evaluator side): it shifts at a hidden
        period and, when the world has a second shift, again later."""
        c = self.config
        if c.get("t_change2") is not None and t >= int(c["t_change2"]):
            return float(c["mu2"])
        return float(c["mu0"]) if t < int(c["t_change"]) else float(c["mu1"])

    def site_rate(self, i: int, t: int) -> float:
        """Mean demand of site i in period t in the distribution world (evaluator side): its base rate times the
        multiplier of the local regime in force (`lam_sched[i]`: [[start period, multiplier], ...])."""
        c = self.config
        m = 1.0
        for t0, x in (c.get("lam_sched") or [[]] * (i + 1))[i]:
            if t >= t0:
                m = float(x)
        return float(c["lam"][i]) * m

    def demand_path(self, t: int) -> float:
        """Mean demand of period t in the capacity world (evaluator side; the hidden growth scenario)."""
        c = self.config
        H = float(c["horizon"])
        kind = c["path"]
        D0 = float(c["d0"])
        if kind == "boom":
            return D0 * (1.0 + float(c["growth"]) * t / H)
        if kind == "bust":
            return D0 * max(0.2, 1.0 - float(c["growth"]) * 0.35 * t / H)
        return D0 if t < int(c["t_jump"]) else D0 * (1.0 + float(c["growth"]))

    def _adv_capacity(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        u = rng.random(1)
        D = float(pois(float(u[0]), self.demand_path(t)))
        keep, arrived = [], 0.0
        for ta, x in st["pending"]:
            if ta == t:
                arrived += x
            else:
                keep.append([ta, x])
        st["pending"] = keep
        st["cap"] += arrived
        sales = min(D, st["cap"])
        rev = st["pend_rev"] + float(c["margin"]) * sales
        cost = st["pend_cost"] + float(c["upkeep"]) * st["cap"]
        st["pend_rev"], st["pend_cost"] = 0.0, 0.0
        nums = {"capacity": st["cap"], "sales": sales, "turned_away": D - sales,
                "under_construction": float(sum(x for _, x in st["pending"])), "capacity_added": arrived}
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def _adv_complements(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        R, P = len(c["res_names"]), len(c["prod_names"])
        u = rng.random(P)
        keep = []
        for ta, i, q in st["pipe"]:
            if ta == t:
                st["res"][i] += q
            else:
                keep.append([ta, i, q])
        st["pipe"] = keep
        Arec = c["recipe"]
        planned = [float(math.floor(h * float(c["rate"]) + 1e-9)) for h in st["plan"]]
        claim = [sum(Arec[i][j] * planned[j] for j in range(P)) for i in range(R)]
        scale = [1.0 if claim[i] <= st["res"][i] or claim[i] == 0 else st["res"][i] / claim[i] for i in range(R)]
        out = []
        for j in range(P):
            f = min([scale[i] for i in range(R) if Arec[i][j] > 0] or [1.0])
            out.append(float(math.floor(planned[j] * f + 1e-9)))
        used = [sum(Arec[i][j] * out[j] for j in range(P)) for i in range(R)]
        for i in range(R):
            st["res"][i] -= used[i]
        st["res_used"] += sum(used)
        st["fg_in"] += sum(out)
        rev = 0.0
        sales = []
        for j in range(P):
            d = pois(float(u[j]), c["dem"][j])
            s = min(float(d), st["fg"][j] + out[j])
            st["fg"][j] += out[j] - s
            sales.append(s)
            rev += float(c["price"][j]) * s
        st["fg_out"] += sum(sales)
        cost = st["pend_cost"] + float(c["hold_res"]) * sum(st["res"]) + float(c["hold_fg"]) * sum(st["fg"])
        st["pend_cost"] = 0.0
        nums = {}
        for i, n in enumerate(c["res_names"]):
            nums[f"{n}.stock"] = st["res"][i]
            nums[f"{n}.used"] = used[i]
            nums[f"{n}.on_order"] = float(sum(q for _, ii, q in st["pipe"] if ii == i))
        for j, n in enumerate(c["prod_names"]):
            nums[f"{n}.made"] = out[j]
            nums[f"{n}.sold"] = sales[j]
            nums[f"{n}.in_store"] = st["fg"][j]
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def _adv_portfolio(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        N = len(c["proj_names"])
        z = rng.standard_normal(N)
        rev = 0.0
        events = []
        for j in range(N):
            x = st["alloc"][j]
            if st["status"][j] != 0 or x <= 0:
                continue
            st["work"][j] += float(c["prod"][j]) * x ** float(c["beta"])
            W = float(c["work"][j])
            if c["dud"][j] and st["work"][j] >= 0.5 * W:
                st["status"][j] = 2
                events.append({"type": "project_abandoned", "project": c["proj_names"][j], "vis": "public"})
            elif st["work"][j] >= W:
                st["status"][j] = 1
                late = max(0, t - int(c["deadline"][j]))
                pay = float(c["value"][j]) * max(0.0, 1.0 - late / float(c["late_window"]))
                st["paid"][j] = pay
                rev += pay
                events.append({"type": "project_completed", "project": c["proj_names"][j], "paid": pay,
                               "vis": "public"})
        cost = st["pend_cost"] + float(c["wage"]) * sum(st["alloc"])
        st["pend_cost"] = 0.0
        nums = {}
        for j, n in enumerate(c["proj_names"]):
            W = float(c["work"][j])
            if st["status"][j] == 1:
                frac = 1.0
            else:
                frac = float(min(0.99, max(0.0, st["work"][j] / W + float(c["report_sd"]) * z[j])))
            nums[f"{n}.reported_done"] = round(frac, 4)
            nums[f"{n}.status"] = float(st["status"][j])
            nums[f"{n}.periods_to_deadline"] = float(int(c["deadline"][j]) - t)
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: rev}, "cost": {a: cost}, "info": {},
                "events": events}

    def _adv_assembly(self, st, rng, t):
        c = self.config
        ags = self.agents
        u = rng.random(1)
        mu = float(c["mu_mean"]) * (1.0 + float(c["mu_amp"]) * math.sin(2 * math.pi * (t + float(c["mu_phase"]))
                                                                          / float(c["mu_period"])))
        D = float(pois(float(u[0]), mu))
        cost = {a: st["pend_cost"][i] for i, a in enumerate(ags)}
        made = [0.0, 0.0]
        for i, a in enumerate(ags):
            x = st["make"][i]
            cost[a] += float(c["unit_cost"][i][0]) * x[0] + float(c["unit_cost"][i][1]) * x[1]
            made[0] += x[0]
            made[1] += x[1]
        st["buf"][0] += made[0]
        st["buf"][1] += made[1]
        st["in"] += made[0] + made[1]
        sold = min(st["buf"][0], st["buf"][1], D)
        st["buf"][0] -= sold
        st["buf"][1] -= sold
        st["out"] += 2 * sold
        hold = float(c["hold"]) * (st["buf"][0] + st["buf"][1])
        rev = {a: float(c["unit_price"]) * sold / len(ags) for a in ags}
        for a in ags:
            cost[a] += hold / len(ags)
        st["pend_cost"] = [0.0] * len(ags)
        X, Y = c["comp_names"]
        obs = {}
        for i, a in enumerate(ags):
            obs[a] = {"numbers": {f"{X}.buffer": st["buf"][0], f"{Y}.buffer": st["buf"][1], "assembled_sold": sold,
                                  "orders": D, f"my_{X}_made": st["make"][i][0], f"my_{Y}_made": st["make"][i][1]},
                      "resources": {f"unit_cost_{X}": float(c["unit_cost"][i][0]),
                                    f"unit_cost_{Y}": float(c["unit_cost"][i][1])}}
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    def current_job(self, st, i: int, t: int) -> int:
        """The participant's job worked on next: its released, unfinished job with the earliest due period."""
        jobs = self.config["jobs"][i]
        cand = [j for j in range(len(jobs)) if not st["done"][i][j] and jobs[j]["release"] <= t]
        if not cand:
            return -1
        return min(cand, key=lambda j: (jobs[j]["deadline"], j))

    def _adv_schedule(self, st, rng, t):
        c = self.config
        ags = self.agents
        nA, nM = len(ags), len(c["machine_names"])
        zw = rng.standard_normal(nM)                     # how much work a machine gets through varies by period
        rev = {a: 0.0 for a in ags}
        cost = {a: st["pend_cost"][i] for i, a in enumerate(ags)}
        users: list = [None] * nM
        conflict = [0] * nM
        if st["plan"] is not None:
            for m, who in st["plan"].items():
                if who != "none":
                    users[c["machine_names"].index(m)] = ags.index(who)
        else:
            for mi in range(nM):
                who = [i for i in range(nA) if st["book"][i] == mi]
                if len(who) == 1:
                    users[mi] = who[0]
                elif len(who) > 1:
                    conflict[mi] = 1
                    for i in who:
                        cost[ags[i]] += float(c["conflict_cost"])
        done_now = {a: [] for a in ags}
        worked = [0.0] * nA
        for mi, i in enumerate(users):
            if i is None:
                continue
            cost[ags[i]] += float(c["machine_cost"])
            j = self.current_job(st, i, t)
            if j < 0:
                continue
            w = float(c["speed"][i][mi]) * math.exp(float(c["work_sd"]) * float(zw[mi])
                                                     - 0.5 * float(c["work_sd"]) ** 2)
            st["work"][i][j] += w
            worked[i] += w
            if st["work"][i][j] >= float(c["true_time"][i][j]) - 1e-9:
                st["done"][i][j] = 1
                job = c["jobs"][i][j]
                late = max(0, t - int(job["deadline"]))
                pay = float(job["value"]) * max(0.0, 1.0 - late / float(c["late_window"]))
                rev[ags[i]] += pay
                done_now[ags[i]].append(job["name"])
        st["last_use"] = [-1 if u_ is None else u_ for u_ in users]
        st["book"] = [-1] * nA
        st["plan"] = None
        st["pend_cost"] = [0.0] * nA
        obs = {}
        for i, a in enumerate(ags):
            nums = {"work_done_this_period": worked[i], "jobs_finished": float(sum(st["done"][i])),
                    "jobs_waiting": float(sum(1 for j, jb in enumerate(c["jobs"][i])
                                              if not st["done"][i][j] and jb["release"] <= t + 1))}
            for mi, m in enumerate(c["machine_names"]):
                nums[f"{m}.user"] = float(-1 if users[mi] is None else users[mi])
                nums[f"{m}.double_booked"] = float(conflict[mi])
            ents = [{"kind": "job", "id": jb["name"], "value": float(jb["value"]), "due": int(jb["deadline"]),
                     "release": int(jb["release"]), "estimated_work": int(jb["est"]),
                     "finished": bool(st["done"][i][j]), "worked": round(float(st["work"][i][j]), 6)}
                    for j, jb in enumerate(c["jobs"][i])]
            obs[a] = {"numbers": nums, "entities": ents,
                      "text": [f"finished: {', '.join(done_now[a])}"] if done_now[a] else []}
        return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}

    # ================================================================== evaluator side
    def truth(self) -> dict:
        c = self.config
        sc = self.sc
        nodes, edges = [], []
        hidden: dict = {}
        if sc == "distribution":
            nodes = ["supplier", "depot"] + [f"site{i}" for i in range(len(c["site_names"]))]
            edges = [{"src": "supplier", "dst": "depot", "kind": "replenish", "lag": int(c["lead"])}]
            edges += [{"src": "depot", "dst": f"site{i}", "kind": "ship", "lag": 1} for i in range(len(c["site_names"]))]
            hidden = {"lam": c["lam"], "lam_sched": c.get("lam_sched"), "lead": c["lead"]}
        elif sc == "perishable":
            nodes = ["oven", "shelf", "fresh_buyers", "flexible_buyers", "bargain_buyers", "bin"]
            edges = [{"src": "oven", "dst": "shelf", "kind": "make", "lag": 0},
                     {"src": "shelf", "dst": "fresh_buyers", "kind": "sell", "lag": 0},
                     {"src": "shelf", "dst": "flexible_buyers", "kind": "sell", "lag": 0},
                     {"src": "shelf", "dst": "bargain_buyers", "kind": "markdown_sell", "lag": 0},
                     {"src": "shelf", "dst": "bin", "kind": "expire", "lag": int(c["shelf"])}]
            hidden = {"shelf": c["shelf"], "fresh_frac": c["fresh_frac"], "bargain": c["bargain"]}
        elif sc == "supply_chain":
            K = len(c["stage_names"])
            nodes = ["raw"] + [f"stage{k}" for k in range(K)] + ["customers"]
            edges = [{"src": "raw", "dst": f"stage{K - 1}", "kind": "produce", "lag": int(c["lags"][K - 1])}]
            edges += [{"src": f"stage{k}", "dst": f"stage{k - 1}", "kind": "ship", "lag": int(c["lags"][k - 1])}
                      for k in range(1, K)]
            edges += [{"src": "stage0", "dst": "customers", "kind": "sell", "lag": 0}]
            hidden = {"lags": c["lags"], "t_change": c["t_change"], "t_change2": c.get("t_change2")}
        elif sc == "capacity":
            nodes = ["builder", "capacity", "market"]
            edges = [{"src": "builder", "dst": "capacity", "kind": "build", "lag": int(c["build_lag"])},
                     {"src": "capacity", "dst": "market", "kind": "serve", "lag": 0}]
            hidden = {"path": c["path"], "build_lag": c["build_lag"]}
        elif sc == "complements":
            nodes = ["suppliers"] + [f"r{i}" for i in range(len(c["res_names"]))] + \
                [f"p{j}" for j in range(len(c["prod_names"]))] + ["market"]
            edges = [{"src": "suppliers", "dst": f"r{i}", "kind": "order", "lag": int(c["lead_time"][i])}
                     for i in range(len(c["res_names"]))]
            edges += [{"src": f"r{i}", "dst": f"p{j}", "kind": "input", "lag": 0}
                      for i in range(len(c["res_names"])) for j in range(len(c["prod_names"])) if c["recipe"][i][j]]
            edges += [{"src": f"p{j}", "dst": "market", "kind": "sell", "lag": 0} for j in range(len(c["prod_names"]))]
            hidden = {"recipe": c["recipe"]}
        elif sc == "portfolio":
            nodes = ["staff"] + [f"proj{j}" for j in range(len(c["proj_names"]))] + ["client"]
            edges = [{"src": "staff", "dst": f"proj{j}", "kind": "work", "lag": 0} for j in range(len(c["proj_names"]))]
            edges += [{"src": f"proj{j}", "dst": "client", "kind": "deliver", "lag": 2} for j in range(len(c["proj_names"]))]
            hidden = {"work": c["work"], "prod": c["prod"], "dud": c["dud"]}
        elif sc == "assembly":
            nodes = ["a0", "a1", "comp0", "comp1", "assembly", "market"]
            edges = [{"src": a, "dst": f"comp{k}", "kind": "make", "lag": 0} for a in ("a0", "a1") for k in (0, 1)]
            edges += [{"src": f"comp{k}", "dst": "assembly", "kind": "complement", "lag": 0} for k in (0, 1)]
            edges += [{"src": "assembly", "dst": "market", "kind": "sell", "lag": 0}]
            hidden = {"mu_mean": c["mu_mean"]}
        else:
            nodes = list(self.agents) + [f"m{k}" for k in range(len(c["machine_names"]))]
            edges = [{"src": a, "dst": f"m{k}", "kind": "book", "lag": 0} for a in self.agents
                     for k in range(len(c["machine_names"]))]
            hidden = {"speed": c["speed"], "true_time": c["true_time"]}
        return {"engine": self.kind, "scenario": sc, "nodes": nodes, "edges": edges, "hidden": hidden,
                "motifs": self.motifs(), "regimes": sc in ("supply_chain", "capacity")}

    def motifs(self) -> list[str]:
        return {
            "distribution": ["delay", "stock", "resource_competition", "selection_effect", "partial_observability",
                             "regime_change"],
            "perishable": ["stock", "decay", "substitution", "endogenous_demand", "non_monotonic"],
            "supply_chain": ["delay", "stock", "feedback", "regime_change", "partial_observability"],
            "capacity": ["delay", "stock", "path_dependence", "regime_change", "saturation"],
            "complements": ["complementarity", "delay", "stock", "resource_competition"],
            "portfolio": ["saturation", "threshold", "resource_competition", "measurement_noise", "selection_effect"],
            "assembly": ["complementarity", "stock", "partial_observability", "multiplicative"],
            "schedule": ["congestion", "resource_competition", "partial_observability", "threshold"],
        }[self.sc]

    def latent(self, state) -> np.ndarray:
        """Hidden state for the informativeness check (evaluator side)."""
        if self.sc == "portfolio":
            return np.asarray([w / float(W) for w, W in zip(state["work"], self.config["work"])], dtype=float)
        if self.sc == "supply_chain":
            return np.asarray([float(sum(q for _, _, q in state["pipe"]))], dtype=float)
        if self.sc == "distribution":
            return np.asarray([float(sum(q for _, q in state["pipe"])) + float(sum(q for *_, q in state["transit"]))])
        return np.asarray([0.0])

    def outcome(self, state, ledger) -> dict:
        rev = sum(ledger["revenue"].values())
        cost = sum(ledger["cost"].values())
        out = {"net": float(rev - cost), "value": float(rev - cost), "revenue": float(rev), "cost": float(cost)}
        if len(self.agents) > 1:
            out["per_agent"] = {a: float(ledger["revenue"][a] - ledger["cost"][a]) for a in self.agents}
        if self.sc == "portfolio":
            out["completed"] = int(sum(1 for s in state["status"] if s == 1))
        if self.sc == "schedule":
            out["jobs_finished"] = int(sum(sum(d) for d in state["done"]))
        return out

    def conservation(self, before, after, out) -> list[str]:
        errs = []
        for a in self.agents:
            if out["revenue"].get(a, 0.0) < -1e-9 or out["cost"].get(a, 0.0) < -1e-9:
                errs.append("negative money flow")
        sc = self.sc
        st = after
        tol = 1e-6
        if sc == "distribution":
            held = st["depot"] + sum(st["sites"]) + sum(q for _, q in st["pipe"]) + sum(q for *_, q in st["transit"])
            if abs(held - (st["units0"] + st["in"] - st["out"])) > tol:
                errs.append("distribution units not conserved")
            if st["depot"] < -tol or min(st["sites"]) < -tol:
                errs.append("negative stock")
        elif sc == "perishable":
            if abs(sum(st["lots"]) - (st["units0"] + st["in"] - st["out"])) > tol or min(st["lots"]) < -tol:
                errs.append("perishable units not conserved")
        elif sc == "supply_chain":
            held = sum(st["inv"]) + sum(q for *_, q in st["pipe"])
            if abs(held - (st["units0"] + st["in"] - st["out"])) > tol or min(st["inv"]) < -tol:
                errs.append("supply-chain units not conserved")
        elif sc == "capacity":
            pend = sum(x for _, x in st["pending"])
            if abs(st["cap"] + pend - (float(self.config["cap0"]) + st["built"] - st["retired"])) > tol \
                    or st["cap"] < -tol:
                errs.append("capacity not conserved")
        elif sc == "complements":
            held = sum(st["res"]) + sum(q for *_, q in st["pipe"])
            if abs(held - (st["res0"] + st["res_in"] - st["res_used"])) > tol or min(st["res"]) < -tol:
                errs.append("input units not conserved")
            if abs(sum(st["fg"]) - (st["fg_in"] - st["fg_out"])) > tol or min(st["fg"]) < -tol:
                errs.append("finished goods not conserved")
        elif sc == "assembly":
            if abs(sum(st["buf"]) - (st["in"] - st["out"])) > tol or min(st["buf"]) < -tol:
                errs.append("components not conserved")
        elif sc == "portfolio":
            if any(w2 < w1 - tol for w1, w2 in zip(before["work"], after["work"])):
                errs.append("completed work decreased")
        elif sc == "schedule":
            for i in range(len(self.agents)):
                if any(w2 < w1 - tol for w1, w2 in zip(before["work"][i], after["work"][i])):
                    errs.append("completed work decreased")
            if sum(1 for u_ in after["last_use"] if u_ >= 0) != len({u_ for u_ in after["last_use"] if u_ >= 0}):
                errs.append("a participant ran two machines")
        return errs
