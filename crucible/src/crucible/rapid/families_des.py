"""Families on the `discrete_event` engine: processing bottlenecks, equipment maintenance, inventory with delayed
replenishment and perishable inventory."""
from __future__ import annotations

import math

from ..mechanisms.base import neutral_names
from .families import family

C = "crucible.rapid.claims:"
I = "crucible.rapid.identify:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


@family("des_bottleneck", engine="discrete_event", category="8.3", item="8.3.1", split="train",
        doc="A tandem processing line with one hidden slow station; at a hidden time (a tooling change) another "
            "station becomes the slow one and the first recovers. Servers are allocated across stations every tick; "
            "jobs queue, wait and are paid on completion.",
        distinct_because="event-driven tandem queues with exponential service; the hidden bottleneck is found by "
                         "reallocating a fixed pool of servers (allocate) and probing station queues (query), and "
                         "it moves at a hidden time (shifting bottleneck)",
        axes={"graph_size": 4, "horizon": 60, "resources": 12, "experiment_cost": 20.0}, objective="maximize_net",
        claims=(C + "bottleneck_station", "crucible.rapid.claims_w3b:bottleneck_moves"), identify=I + "bottleneck",
        reference="crucible.rapid.claims_w3b:queue_tracker", oracle="crucible.rapid.claims_w3b:oracle_shifting_line",
        version="2")
def des_bottleneck(rng, d):
    S = int(d["graph_size"])
    R = int(d["resources"])
    H = int(d["horizon"])
    names = neutral_names(rng, S + 2)
    mu = [u(rng, 0.8, 1.2) for _ in range(S)]
    b = int(rng.integers(S))
    mu[b] = u(rng, 0.35, 0.5)
    b2 = int(rng.choice([k for k in range(S) if k != b]))
    mu2 = [u(rng, 0.8, 1.2) for _ in range(S)]
    mu2[b2] = u(rng, 0.35, 0.5)

    def best_cap(m):
        inv = [1.0 / x for x in m]
        best = [max(1, int(round(R * x / sum(inv)))) for x in inv]
        while sum(best) > R:
            best[best.index(max(best))] -= 1
        return min(c * x for c, x in zip(best, m))
    cap = min(best_cap(mu), best_cap(mu2))
    base = max(1, R // S)
    default = [base] * S
    return {"engine": "discrete_event", "scenario": "line", "horizon": H, "mu": mu, "mu2": mu2,
            "shift_at": float(int(rng.integers(int(0.35 * H), int(0.6 * H) + 1))),
            "arrival_rate": 0.85 * cap, "servers": R, "default_alloc": default, "station_names": names[:S],
            "gauge_names": names[S:S + 2], "price": 600.0, "server_cost": 25.0, "holding_cost": 8.0,
            "query_cost": float(d["experiment_cost"]), "endowment": 300_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income: completed jobs pay, servers "
                                                          "and waiting jobs cost."},
            "brief": f"A processing line with {S} stations in sequence and {R} servers to allocate between them. "
                     "Gauges: jobs completed this tick and jobs in the system."}


@family("des_maintenance", engine="discrete_event", category="8.3", item="8.3.2", split="train",
        doc="Machines wear out at hidden rates (Weibull failures). Preventive service takes a machine down briefly "
            "and cheaply; a failure takes it down for a long, expensive repair.",
        distinct_because="event-driven machine life cycles with age-dependent hazard; the action (service one "
                         "machine) pays off only for machines whose wear state warrants it",
        axes={"graph_size": 3, "horizon": 80}, objective="maximize_net",
        claims=(C + "maintenance_pays",), identify=I + "fragile_machine")
def des_maintenance(rng, d):
    M = int(d["graph_size"])
    names = neutral_names(rng, M + 2)
    eta = [u(rng, 18.0, 40.0) for _ in range(M)]
    eta[int(rng.integers(M))] = u(rng, 7.0, 11.0)
    return {"engine": "discrete_event", "scenario": "maintenance", "horizon": int(d["horizon"]), "eta": eta,
            "shape": u(rng, 2.5, 3.5), "rate": [1.0] * M, "price": 300.0, "service_cost": 120.0,
            "service_time": 0.3, "repair_cost": 900.0, "repair_time": 5.0, "machine_names": names[:M],
            "gauge_names": names[M:M + 2], "endowment": 200_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income from the machines' output."},
            "brief": f"{M} machines. Each tick you may send the service crew to one machine. You see each machine's "
                     "status and runtime since its last service."}


def inventory_terms_text(price: float, unit_cost: float, order_fixed: float, holding: float) -> str:
    """The shop's trading terms as the brief states them (public: a shop knows its prices and costs)."""
    return (f"Terms: each unit sold pays {price:g} cents; each order costs {order_fixed:g} cents plus {unit_cost:g} "
            f"cents per unit; every unit on hand costs {holding:g} cents per tick.")


def _inventory(rng, d, perishable: bool):
    names = neutral_names(rng, 5)
    price, unit_cost, order_fixed, holding = 500.0, 250.0, 300.0, 10.0 if not perishable else 5.0
    cfg = {"engine": "discrete_event", "scenario": "inventory", "horizon": int(d["horizon"]),
           "demand": u(rng, 4.0, 8.0), "season_amp": u(rng, 0.2, 0.4), "season_period": u(rng, 15.0, 30.0),
           "season_phase": u(rng, 0.0, 2 * math.pi), "lead_min": int(rng.integers(2, 6)), "lead_p": u(rng, 0.4, 0.8),
           "price": price, "unit_cost": unit_cost, "order_fixed": order_fixed, "holding_cost": holding,
           "max_order": 40, "init_stock": 20, "order_name": names[0], "gauge_names": names[1:5],
           "endowment": 300_000,
           "objective": {"type": "maximize_net", "text": "Maximise your net income from sales."},
           "brief": "A shop with one product. Each tick you may order stock; deliveries arrive later. Gauges: stock "
                    "on hand, units sold, customers turned away" + (", units expired. " if perishable else ". ") +
                    inventory_terms_text(price, unit_cost, order_fixed, holding)}
    if perishable:
        cfg["shelf_life"] = float(rng.integers(3, 7))
    return cfg


@family("des_inventory", engine="discrete_event", category="8.4", item="8.4.1", split="train",
        doc="Inventory with delayed, random replenishment lead times and seasonal Poisson demand; orders are the "
            "action, lost sales and holding are the costs.",
        distinct_because="an order pipeline with hidden random lead times into a stock drained by event-level "
                         "Poisson demand",
        axes={"horizon": 60}, objective="maximize_net", claims=(C + "lead_time",), identify=I + "lever_sign",
        lever_view=True)
def des_inventory(rng, d):
    return _inventory(rng, d, False)


@family("des_perishable", engine="discrete_event", category="8.4", item="8.4.2", split="val",
        doc="The inventory world with perishable lots: each delivery expires a hidden number of ticks after it "
            "arrives, so over-ordering is wasted.",
        distinct_because="adds expiry events (decay of lots after a hidden shelf life) and an expiry gauge to the "
                         "lead-time inventory structure",
        axes={"horizon": 60}, objective="maximize_net", claims=(C + "perishing",), identify=I + "lever_sign",
        lever_view=True)
def des_perishable(rng, d):
    return _inventory(rng, d, True)
