"""Engine `discrete_event`: discrete-event simulation (Part 7.3).

Time is continuous. A tick is the interval [t, t+1); inside it, events are taken from a priority queue in time
order and handled one at a time, and each handler may schedule further events (an arrival schedules the next
arrival, a service start schedules its completion, a delivery is scheduled a random lead time after its order).
Nothing is stepped on a grid: queues, lots and machine states change only at event times, and time-weighted
quantities (work in progress, stock held) are integrated exactly between events. This transition code shares
nothing with the causal-graph engine.

Scenarios (one per family structure):

    line         tandem stations with hidden service rates; the participant allocates servers (`allocate`);
                 the bottleneck is hidden (8.3.1)
    maintenance  machines with hidden Weibull wear; preventive service (`discrete`) versus run-to-failure and
                 long repairs (8.3.2)
    inventory    stock replenished by orders arriving after a hidden random lead time; Poisson demand with hidden
                 seasonality; optionally perishable lots that expire after a hidden shelf life (8.4.1, 8.4.2)
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .. import actions as A
from ..primitives import get as prim
from .base import ActionError, RapidEngine, register


def _exp(rng, rate: float) -> float:
    return float(rng.exponential(1.0 / rate)) if rate > 0 else math.inf


@register
class DiscreteEvent(RapidEngine):
    kind = "discrete_event"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in ("line", "maintenance", "inventory"):
            raise ValueError(f"unknown scenario {self.sc}")

    # ------------------------------------------------------------------ public surface
    def action_space(self, agent: str) -> A.ActionSpace:
        c = self.config
        kinds, limits = {}, {}
        if self.sc == "line":
            kinds["allocate"] = A.allocate_schema(list(c["station_names"]), float(c["servers"]), "servers")
            limits["allocate_total"] = float(c["servers"])
        elif self.sc == "maintenance":
            opts = ["none"] + list(c["machine_names"])
            kinds["discrete"] = A.discrete_schema({"service": opts})
            limits["decisions"] = {"service": opts}
        else:
            kinds["levers"] = A.levers_schema([{"name": c["order_name"], "min": 0.0, "max": float(c["max_order"]),
                                                "default": 0.0, "integer": True}])
        q = self.query_targets(agent)
        if q:
            kinds["query"] = A.query_schema(dict(q, tools="the action kinds available to you",
                                                 history="your last observations (args: n)",
                                                 spec="the public description of this world"))
        return A.ActionSpace(kinds, limits)

    def query_targets(self, agent):
        c = self.config
        if self.sc == "line":
            return {f"station:{n}": f"queue length and busy servers of station {n} (costs {c['query_cost']})"
                    for n in c["station_names"]}
        return {}

    # ------------------------------------------------------------------ state
    def init_state(self, rng):
        c = self.config
        st = {"heap": [], "seq": 0, "clock": 0.0, "stats": {}}
        if self.sc == "line":
            S = len(c["mu"])
            st.update(q=[0] * S, busy=[0] * S, servers=list(c["default_alloc"]), arrived=0, completed=0)
            self._push(st, _exp(rng, c["arrival_rate"]), "arr", None)
        elif self.sc == "maintenance":
            M = len(c["eta"])
            st.update(up=[True] * M, mode=["run"] * M, age=[0.0] * M, ver=[0] * M, started=[0.0] * M,
                      produced=0.0, failures=0, services=0)
            for j in range(M):
                self._schedule_failure(st, j, 0.0, rng)
        else:
            st.update(lots=[[float(c["init_stock"]), math.inf]], pipeline=[], sold=0, lost=0, expired=0,
                      delivered=float(c["init_stock"]), ordered=0.0)
            self._push(st, _exp(rng, self._lam(0.0)), "cust", None)
        return st

    def _push(self, st, time, typ, data):
        st["seq"] += 1
        heapq.heappush(st["heap"], (float(time), st["seq"], typ, data))

    def _lam(self, time):
        c = self.config
        return c["demand"] * (1.0 + c["season_amp"] * math.sin(2 * math.pi * time / c["season_period"] +
                                                               c["season_phase"]))

    def _schedule_failure(self, st, j, now, rng):
        """Next failure of machine j from its current age (Weibull, sampled by inverting the conditional survival)."""
        c = self.config
        eta, k = c["eta"][j], c["shape"]
        a0 = st["age"][j]
        af = eta * ((a0 / eta) ** k - math.log(1.0 - float(rng.random()))) ** (1.0 / k)
        st["ver"][j] += 1
        st["started"][j] = now
        self._push(st, now + (af - a0), "fail", [j, st["ver"][j]])

    # ------------------------------------------------------------------ actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        k = action["kind"]
        if k == "allocate" and self.sc == "line":
            names = c["station_names"]
            alloc = [int(math.floor(float(action["amounts"].get(n, 0.0)) + 1e-9)) for n in names]
            if sum(alloc) > int(c["servers"]):
                raise ActionError("more servers than you have")
            state["servers"] = alloc
            state["_restart"] = True
            return {"servers": dict(zip(names, alloc))}
        if k == "discrete" and self.sc == "maintenance":
            ch = action["choice"]
            if ch == "none":
                return {"service": None}
            j = c["machine_names"].index(ch)
            if state["mode"][j] != "run":
                raise ActionError(f"{ch} is not running (it is {state['mode'][j]})")
            state["pending_service"] = j
            return {"service": ch}
        if k == "levers" and self.sc == "inventory":
            q = int(round(float(action["values"].get(c["order_name"], 0.0))))
            state["order_now"] = max(0, min(q, int(c["max_order"])))
            return {"ordered": state["order_now"]}
        if k == "query" and self.sc == "line":
            name = action["target"].split(":", 1)[1]
            j = c["station_names"].index(name)
            return {"station": name, "queue": int(state["q"][j]), "busy": int(state["busy"][j]),
                    "servers": int(state["servers"][j]), "cost": float(c["query_cost"])}
        raise ActionError(f"{k} is not available in this world")

    # ------------------------------------------------------------------ dynamics
    def advance(self, st, rng, t):
        c = self.config
        end = float(t + 1)
        stats = {"area": 0.0, "last": float(t), "rev": 0.0, "cost": 0.0, "events": 0, "done": 0, "sold": 0,
                 "lost": 0, "expired": 0, "delivered": 0, "failures": 0, "uptime": 0.0}
        self._tick_start(st, rng, float(t), stats)
        while st["heap"] and st["heap"][0][0] < end:
            time, _, typ, data = heapq.heappop(st["heap"])
            self._integrate(st, stats, time)
            st["clock"] = time
            stats["events"] += 1
            getattr(self, f"_on_{self.sc}")(st, typ, data, time, rng, stats)
        self._integrate(st, stats, end)
        st["clock"] = end
        return self._tick_end(st, stats, t)

    def _integrate(self, st, stats, time):
        dt = time - stats["last"]
        if dt <= 0:
            return
        if self.sc == "line":
            stats["area"] += dt * (sum(st["q"]) + sum(st["busy"]))
        elif self.sc == "maintenance":
            up = sum(1 for m in st["mode"] if m == "run")
            stats["uptime"] += dt * up
            stats["prod"] = stats.get("prod", 0.0) + dt * sum(r for r, m in zip(self.config["rate"], st["mode"])
                                                              if m == "run")
        else:
            stats["area"] += dt * sum(q for q, _ in st["lots"])
        stats["last"] = time

    def _tick_start(self, st, rng, now, stats):
        c = self.config
        if self.sc == "line" and st.pop("_restart", False):
            for k in range(len(c["mu"])):
                self._start(st, k, now, rng)
        if self.sc == "maintenance" and st.get("pending_service") is not None:
            j = st.pop("pending_service")
            st["mode"][j] = "service"
            st["ver"][j] += 1                                # the pending failure no longer applies
            st["services"] += 1
            stats["cost"] += c["service_cost"]
            self._push(st, now + c["service_time"], "back", [j, st["ver"][j]])
        if self.sc == "inventory":
            q = st.pop("order_now", 0)
            if q > 0:
                lead = c["lead_min"] + int(rng.geometric(c["lead_p"])) - 1
                st["seq"] += 1
                oid = st["seq"]
                st["pipeline"].append([oid, q])
                st["ordered"] += q
                stats["cost"] += c["order_fixed"] + c["unit_cost"] * q
                self._push(st, now + lead + 0.5, "deliver", [oid, q])

    def mu_at(self, k: int, now: float) -> float:
        """Service rate of station k at time `now` (line scenario; evaluator data): after a hidden shift time the
        stations run at their second set of rates (`mu2`), so the bottleneck moves."""
        c = self.config
        if c.get("mu2") is not None and now >= float(c["shift_at"]):
            return float(c["mu2"][k])
        return float(c["mu"][k])

    def _start(self, st, k, now, rng):
        while st["busy"][k] < st["servers"][k] and st["q"][k] > 0:
            st["q"][k] -= 1
            st["busy"][k] += 1
            self._push(st, now + _exp(rng, self.mu_at(k, now)), "done", k)

    def _on_line(self, st, typ, data, time, rng, stats):
        c = self.config
        S = len(c["mu"])
        if typ == "arr":
            st["arrived"] += 1
            st["q"][0] += 1
            self._start(st, 0, time, rng)
            self._push(st, time + _exp(rng, c["arrival_rate"]), "arr", None)
        elif typ == "done":
            k = data
            st["busy"][k] -= 1
            if k + 1 < S:
                st["q"][k + 1] += 1
                self._start(st, k + 1, time, rng)
            else:
                st["completed"] += 1
                stats["done"] += 1
                stats["rev"] += c["price"]
            self._start(st, k, time, rng)

    def _on_maintenance(self, st, typ, data, time, rng, stats):
        c = self.config
        j, ver = data
        if ver != st["ver"][j]:
            return                                           # superseded (the machine was serviced first)
        if typ == "fail":
            st["mode"][j] = "repair"
            st["failures"] += 1
            stats["failures"] += 1
            stats["cost"] += c["repair_cost"]
            st["ver"][j] += 1
            self._push(st, time + _exp(rng, 1.0 / c["repair_time"]), "back", [j, st["ver"][j]])
        elif typ == "back":
            st["mode"][j] = "run"
            st["age"][j] = 0.0
            self._schedule_failure(st, j, time, rng)

    def _on_inventory(self, st, typ, data, time, rng, stats):
        c = self.config
        if typ == "cust":
            got = False
            for lot in st["lots"]:
                if lot[0] >= 1:
                    lot[0] -= 1
                    got = True
                    break
            st["lots"] = [lot for lot in st["lots"] if lot[0] > 0]
            if got:
                st["sold"] += 1
                stats["sold"] += 1
                stats["rev"] += c["price"]
            else:
                st["lost"] += 1
                stats["lost"] += 1
            self._push(st, time + _exp(rng, max(self._lam(time), 1e-6)), "cust", None)
        elif typ == "deliver":
            oid, q = data
            exp_at = time + c["shelf_life"] if c.get("shelf_life") else math.inf
            st["lots"].append([float(q), exp_at])
            st["delivered"] += q
            stats["delivered"] += q
            st["pipeline"] = [p for p in st["pipeline"] if p[0] != oid]
            if exp_at < math.inf:
                self._push(st, exp_at, "expire", exp_at)
        elif typ == "expire":
            keep = []
            for lot in st["lots"]:
                if abs(lot[1] - data) < 1e-9:
                    st["expired"] += int(lot[0])
                    stats["expired"] += int(lot[0])
                else:
                    keep.append(lot)
            st["lots"] = keep

    def _tick_end(self, st, stats, t):
        c = self.config
        a = self.agents[0]
        if self.sc == "line":
            cost = stats["cost"] + c["server_cost"] * sum(st["servers"]) + c["holding_cost"] * stats["area"]
            wip = sum(st["q"]) + sum(st["busy"])
            nums = {c["gauge_names"][0]: float(stats["done"]), c["gauge_names"][1]: float(wip)}
            ob = {"numbers": nums, "arrays": {"servers": [float(x) for x in st["servers"]]}}
            rev = stats["rev"]
        elif self.sc == "maintenance":
            rev = c["price"] * stats.get("prod", 0.0)
            cost = stats["cost"]
            for j, m in enumerate(st["mode"]):
                if m == "run":
                    st["age"][j] += float(t + 1) - max(st["started"][j], float(t))
                    st["started"][j] = float(t + 1)
            ent = [{"kind": "machine", "id": n, "status": m, "runtime": round(st["age"][j], 6)}
                   for j, (n, m) in enumerate(zip(c["machine_names"], st["mode"]))]
            nums = {c["gauge_names"][0]: float(stats.get("prod", 0.0)), c["gauge_names"][1]: float(stats["failures"])}
            ob = {"numbers": nums, "entities": ent}
        else:
            cost = stats["cost"] + c["holding_cost"] * stats["area"]
            rev = stats["rev"]
            stock = sum(q for q, _ in st["lots"])
            nums = {c["gauge_names"][0]: float(stock), c["gauge_names"][1]: float(stats["sold"]),
                    c["gauge_names"][2]: float(stats["lost"])}
            if c.get("shelf_life"):
                nums[c["gauge_names"][3]] = float(stats["expired"])
            ob = {"numbers": nums, "resources": {"on_order_units": float(sum(q for _, q in st["pipeline"]))}}
        return {"obs": {a: ob}, "revenue": {a: float(rev)}, "cost": {a: float(cost)},
                "info": {"events": stats["events"]}}

    # ------------------------------------------------------------------ evaluator side
    def truth(self):
        c = self.config
        if self.sc == "line":
            nodes = [{"id": f"s{k}", "role": "station", "mu": c["mu"][k]} for k in range(len(c["mu"]))]
            edges = [{"src": f"s{k}", "dst": f"s{k + 1}", "lag": 0, "sign": 1, "kind": "flow"}
                     for k in range(len(c["mu"]) - 1)]
            hidden = {"bottleneck": int(np.argmin(c["mu"]))}
            if c.get("mu2") is not None:
                hidden.update(bottleneck_after=int(np.argmin(c["mu2"])), shift_at=c["shift_at"])
        elif self.sc == "maintenance":
            nodes = [{"id": f"m{j}", "role": "machine", "eta": c["eta"][j]} for j in range(len(c["eta"]))]
            edges = []
            hidden = {"fragile": int(np.argmin(c["eta"]))}
        else:
            nodes = [{"id": "orders"}, {"id": "pipeline"}, {"id": "stock"}, {"id": "demand"}, {"id": "sales"}]
            edges = [{"src": "orders", "dst": "pipeline", "lag": c["lead_min"], "sign": 1, "kind": "delay"},
                     {"src": "pipeline", "dst": "stock", "lag": 0, "sign": 1, "kind": "flow"},
                     {"src": "stock", "dst": "sales", "lag": 0, "sign": 1, "kind": "gate"},
                     {"src": "demand", "dst": "sales", "lag": 0, "sign": 1, "kind": "flow"}]
            hidden = {"lead_min": c["lead_min"], "shelf_life": c.get("shelf_life")}
        return {"engine": self.kind, "scenario": self.sc, "nodes": nodes, "edges": edges, "hidden": hidden,
                "motifs": self.motifs()}

    def motifs(self):
        c = self.config
        if self.sc == "line":
            m = {"congestion", "resource_competition"}         # queues at finite servers; servers shared by stations
        elif self.sc == "maintenance":
            m = {"state_dependent_effect", "stock"}            # hazard grows with accumulated runtime (age)
            if c["shape"] > 1:
                m.add("linear_nonlinear")
        else:
            m = {"delay", "stock"}                           # lead-time pipeline into a stock
            if c.get("shelf_life"):
                m.add("decay")                               # lots lose their value: they expire after a shelf life
        return sorted(m)

    def outcome(self, state, ledger):
        a = self.agents[0]
        net = float(ledger["revenue"][a] - ledger["cost"][a])
        out = {"net": net, "value": net, "revenue": float(ledger["revenue"][a]), "cost": float(ledger["cost"][a])}
        if self.sc == "line":
            out.update(arrived=state["arrived"], completed=state["completed"])
        elif self.sc == "maintenance":
            out.update(failures=state["failures"], services=state["services"])
        else:
            out.update(sold=state["sold"], lost=state["lost"], expired=state["expired"])
        return out

    def conservation(self, before, after, out):
        errs = []
        a = self.agents[0]
        if out["revenue"][a] < 0 or out["cost"][a] < 0:
            errs.append("negative money flow")
        if self.sc == "line":
            inside = sum(after["q"]) + sum(after["busy"])
            if after["arrived"] != after["completed"] + inside:
                errs.append(f"jobs not conserved: {after['arrived']} arrived, {after['completed']} done, {inside} in")
            if any(x < 0 for x in after["q"] + after["busy"]):
                errs.append("negative queue")
        elif self.sc == "inventory":
            stock = sum(q for q, _ in after["lots"])
            bal = after["delivered"] - after["sold"] - after["expired"] - stock
            if abs(bal) > 1e-9:
                errs.append(f"stock not conserved (imbalance {bal})")
        else:
            if any(m not in ("run", "service", "repair") for m in after["mode"]):
                errs.append("unknown machine mode")
        return errs

    def check_state(self, state):
        return [] if all(math.isfinite(x[0]) for x in state["heap"]) else ["event time not finite"]


def hazard_curve(eta: float, shape: float, ages: np.ndarray) -> np.ndarray:
    """The machines' wear law as a registered primitive (power form of `linear_nonlinear`): hazard(age) =
    (shape/eta) (age/eta)^(shape-1). Used by the evaluator to compute the private optimal service age."""
    p = {"form": "power", "a": shape / eta, "b": 0.0, "k": shape - 1.0}
    out, _ = prim("linear_nonlinear").apply(p, {"x": np.asarray(ages, dtype=float) / eta}, {}, None, None, 0)
    return out
