"""Families on the `graph_dynamical` engine: routing under congestion, changing topology, diffusion on clustered
networks, interdependent infrastructure failures, and network reliability with repair."""
from __future__ import annotations

import numpy as np

from ..mechanisms.base import neutral_names
from .families import family

C = "crucible.rapid.claims:"
I = "crucible.rapid.identify:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _routing(rng, d, rewire: bool):
    K = int(d["graph_size"])
    J = max(2, K // 2)
    names = neutral_names(rng, 2 * K + 2)
    assign = [int(j) for j in rng.integers(J, size=K)]
    for j in range(J):                                   # every second hop carries at least one route
        assign[j % K] = j
    rng.shuffle(assign)
    c1 = [u(rng, 8.0, 15.0) for _ in range(K)]
    c2 = [u(rng, 6.0, 14.0) for _ in range(J)]
    cfg = {"engine": "graph_dynamical", "scenario": "routing", "horizon": int(d["horizon"]), "K": K, "J": J,
           "assign": assign, "c1": c1, "c2": c2, "toll": [u(rng, 5.0, 25.0) for _ in range(K)],
           "demand_mean": 0.8 * sum(c2), "demand_sd": 0.1 * sum(c2) * 0.3, "demand_ar": 0.8, "price": 100.0,
           "hold": 15.0, "noise": float(d["observation_noise"]), "w_default": [0.5] * K,
           "route_names": names[:K], "gauge_names": names[K:2 * K + 1], "endowment": 300_000,
           "objective": {"type": "maximize_net", "text": "Maximise your net income: delivered flow pays, route "
                                                         "tolls and queued flow cost."},
           "brief": f"Traffic from one source to one sink over {K} routes. Levers: the weight of each route in your "
                    "split. Gauges: flow delivered, and the latency of each route."}
    cfg["income_cents_per_tick"] = int(cfg["price"] * (sum(c2) + 1)) + 1
    if rewire:
        a2 = list(assign)
        while a2 == assign:
            a2 = [int(j) for j in rng.permutation(assign)]
        cfg["assign2"] = a2
        cfg["rewire_at"] = int(rng.integers(int(d["horizon"]) // 3, 2 * int(d["horizon"]) // 3))
    return cfg


@family("net_routing", engine="graph_dynamical", category="8.7", item="8.7.1", split="train",
        doc="Split traffic over routes whose second hops share hidden bottlenecks; fluid queues build when a hop is "
            "overloaded and route latency rises with congestion.",
        distinct_because="flow dynamics on a two-hop network with shared hidden bottleneck links and congestion "
                         "latency; vectorised batch path and a kernel-hosted twin",
        axes={"graph_size": 4, "horizon": 60, "observation_noise": 0.2}, objective="maximize_net",
        claims=(C + "shared_bottleneck",), identify=I + "lever_sign", lever_view=True)
def net_routing(rng, d):
    return _routing(rng, d, False)


@family("net_topology_shift", engine="graph_dynamical", category="8.7", item="8.7.6", split="val",
        doc="The routing network, but at a hidden time the second hops are rewired, so the routes that were best "
            "become congested.",
        distinct_because="adds a hidden topology change (regime change of the network structure) to the routing "
                         "dynamics",
        axes={"graph_size": 4, "horizon": 60, "observation_noise": 0.2}, objective="maximize_net",
        claims=(C + "topology_shift",), identify=I + "lever_sign", lever_view=True)
def net_topology_shift(rng, d):
    return _routing(rng, d, True)


@family("net_diffusion", engine="graph_dynamical", category="8.7", item="8.7.3", split="train",
        doc="Adoption spreads by complex contagion (each node adopts once enough of its neighbours have) on a "
            "clustered network; seeds are placed by cluster and adopters are worth more as adoption grows.",
        distinct_because="threshold contagion over a stochastic block graph with network effects in value; only "
                         "cluster-level adoption counts are visible",
        axes={"graph_size": 60, "hidden_state_complexity": 3, "resources": 8, "horizon": 40},
        objective="maximize_net", claims=(C + "concentrated_seeding",), identify=I + "best_cluster")
def net_diffusion(rng, d):
    N, Cn = int(d["graph_size"]), int(d["hidden_state_complexity"])
    names = neutral_names(rng, Cn + 1)
    cluster = [int(i * Cn // N) for i in range(N)]
    p_in = [u(rng, 0.15, 0.45) for _ in range(Cn)]
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in[cluster[i]] if cluster[i] == cluster[j] else 0.01
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    th_mean = [u(rng, 0.15, 0.4) for _ in range(Cn)]
    theta = [float(np.clip(th_mean[cluster[i]] + rng.normal(0, 0.05), 0.05, 0.9)) for i in range(N)]
    deg = A.sum(axis=1)
    by_degree = [[int(i) for i in sorted((i for i in range(N) if cluster[i] == k), key=lambda i: (-deg[i], i))]
                 for k in range(Cn)]
    return {"engine": "graph_dynamical", "scenario": "diffusion", "horizon": int(d["horizon"]),
            "adj": A.astype(int).tolist(), "cluster": cluster, "theta": theta, "by_degree": by_degree,
            "seed_total": int(d["resources"]), "seed_budget": float(d["resources"]), "seed_cost": 300.0,
            "value": 40.0, "strength": 1.0, "churn": 0.01, "cluster_names": names[:Cn], "gauge_names": names[Cn:],
            "endowment": 100_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income: adopters pay every tick, more "
                                                          "when more have adopted; seeds cost money."},
            "brief": f"A population of {N} in {Cn} communities. You may seed adopters in a community (at most "
                     f"{int(d['resources'])} seeds in total). Gauges: adopters in total and per community."}


@family("net_cascade", engine="graph_dynamical", category="8.7", item="8.7.4", split="val",
        doc="A power grid whose overloaded nodes shed load onto neighbours, and a control layer that depends on the "
            "grid and controls it; failures cascade within and between layers.",
        distinct_because="load-redistribution cascades on a grid coupled to a dependency layer (interdependent "
                         "failure); graph edits add backup supply links",
        axes={"graph_size": 20, "horizon": 60, "resources": 4}, objective="maximize_net",
        claims=(C + "backup_links_contain_cascades",), identify=I + "lever_sign")
def net_cascade(rng, d):
    P = int(d["graph_size"])
    Cm = max(2, P // 2)
    names = neutral_names(rng, P + Cm + 6)
    A = np.zeros((P, P))
    for i in range(P):
        A[i, (i + 1) % P] = A[(i + 1) % P, i] = 1.0
    for _ in range(P // 3):
        i, j = (int(x) for x in rng.integers(P, size=2))
        if i != j:
            A[i, j] = A[j, i] = 1.0
    base = [u(rng, 0.5, 0.8) for _ in range(P)]
    cap = [b * u(rng, 1.12, 1.35) for b in base]
    dep = [int(x) for x in rng.integers(P, size=Cm)]
    ctrl = [int(x) for x in rng.integers(Cm, size=P)]
    zones = 4
    zone = [int(i * zones // P) for i in range(P)]
    return {"engine": "graph_dynamical", "scenario": "cascade", "horizon": int(d["horizon"]),
            "adj": A.astype(int).tolist(), "base": base, "cap": cap, "dep": dep, "ctrl": ctrl, "zone": zone,
            "zone_names": names[P + Cm:P + Cm + zones], "node_names": names[:P + Cm],
            "gauge_names": names[P + Cm + zones:P + Cm + zones + 2], "dem_ar": 0.8, "dem_sd": 0.05, "p_shock": 0.25,
            "shock": u(rng, 0.5, 0.8), "repair_ticks": 5.0, "reinforce_budget": float(d["resources"]),
            "reinf_gain": 0.12, "reinf_cost": 60.0, "edit_cost": 400.0, "backup_cost": 15.0, "price": 300.0,
            "endowment": 200_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income from the load you serve."},
            "brief": f"A grid of {P} power nodes and {Cm} control nodes. Each control node draws power from a grid "
                     "node and each grid node is run by a control node. You may reinforce zones every tick and add "
                     "backup supply links. Gauges: load served and nodes lost this tick."}


@family("net_repair", engine="graph_dynamical", category="8.7", item="8.7.2", split="train",
        doc="Links of a network fail at hidden rates; demand between node pairs is served only while they are "
            "connected. A crew repairs one link per tick; new links can be built.",
        distinct_because="connectivity of origin-destination pairs over a failing graph (breadth of the surviving "
                         "network), repaired by discrete dispatch and graph edits",
        axes={"graph_size": 12, "horizon": 60}, objective="maximize_net", claims=(C + "repair_restores",),
        identify=I + "lever_sign")
def net_repair(rng, d):
    N = int(d["graph_size"])
    names = neutral_names(rng, N + 3 * N + 2)
    edges = []
    order = [int(i) for i in rng.permutation(N)]
    for k in range(1, N):
        edges.append((order[k], order[int(rng.integers(k))]))
    have = {tuple(sorted(e)) for e in edges}
    while len(edges) < N + N // 2:
        a, b = (int(x) for x in rng.integers(N, size=2))
        if a != b and tuple(sorted((a, b))) not in have:
            have.add(tuple(sorted((a, b))))
            edges.append((a, b))
    haz = [u(rng, 0.005, 0.02) for _ in edges]
    for k in rng.choice(len(edges), size=max(2, len(edges) // 5), replace=False):
        haz[int(k)] = u(rng, 0.08, 0.15)
    pairs = []
    while len(pairs) < 6:
        a, b = (int(x) for x in rng.integers(N, size=2))
        if a != b:
            pairs.append((a, b))
    return {"engine": "graph_dynamical", "scenario": "repair", "horizon": int(d["horizon"]), "n_nodes": N,
            "edges": [list(e) for e in edges], "hazard": haz, "pairs": [list(p) for p in pairs],
            "pair_demand": [u(rng, 1.0, 5.0) for _ in pairs], "price": 150.0, "repair_cost": 200.0,
            "build_cost": 2500.0, "extra_hazard": 0.005, "node_names": names[:N],
            "edge_names": names[N:N + len(edges)], "gauge_names": names[-2:], "endowment": 200_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income from the demand you serve."},
            "brief": f"A network of {N} nodes and {len(edges)} links. Links break; each tick your crew can repair one "
                     "broken link, and you may build new links. You see which links are up and how much demand is "
                     "served."}
