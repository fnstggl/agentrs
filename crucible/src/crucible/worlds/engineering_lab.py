"""A computational engineering laboratory: improve the routing controller of a production network.

This world exists to show that the kernel is not hard-wired to economics. Its ground truth is *executable
measurement*: a deterministic flow model of a real (synthetic) network, driven by a hidden traffic process, scores
whatever routing program is deployed. No judge, no rubric.

Module `lab` (host `lab`) writes at genesis (seeded, through ordinary events):

* a production network: 12-40 nodes, undirected links with capacity and base latency, built by a configurable
  generator (`grid` with express links, `scale_free` preferential attachment, or `waxman` geometric);
* a hidden traffic process (gravity or datacenter-hub structure, diurnal cycle, hotspot pairs, per-snapshot noise,
  daily drift and shocks). Its parameters live only in `internal` events and in state no endpoint exposes;
* the lab operator organisation with a bank account, and a public repository `<operator>/controllers` holding the
  incumbent controller (`baseline.py`: lowest-latency single path, real Python);
* the baseline deployed in production.

Controllers are programs in repositories with the signature

    route(topology, demands) -> {"routes": {demand_id: [[path, fraction], ...]}}

`topology = {"nodes": [...], "links": [{"id", "u", "v", "capacity", "latency"}]}`,
`demands = [{"id": "src>dst", "src", "dst", "volume"}]`; a demand may be split across up to 3 simple paths whose
fractions sum to 1. Agent controllers run in the sandbox (no world access); their output is recorded.

Physics (`evaluate`): link load = sum of routed flow; link delay = latency + queue_scale * load/(capacity - load)
(i.e. rho/(1-rho), rho capped at 0.99); above capacity a link drops the excess fraction (load - capacity)/load of
everything crossing it. Dropped traffic costs `drop_penalty_ms` (a timeout) in the *effective* mean delay, so a
controller cannot look fast by losing traffic. Invalid output for a demand drops all of it.

Production runs hourly on a fresh sample of the hidden process; exact metrics are internal, a probe-sampled noisy
dashboard is published with a delay, and SLA breaches cost the operator money. The paired metric
`lab_improvement_vs_baseline` re-runs the baseline on the recorded snapshot seeds of a window.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ..core import schema as S
from ..core.canonical import digest, to_jsonable
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..goals.metrics import metric
from ..services.identity import ident

DAY = 86400
HOUR = 3600
RHO_MAX = 0.99
MAX_PATHS = 3
FRAC_TOL = 1e-6
HARNESS_FILE = "_crucible_lab_harness.py"
HARNESS_TARGET = "_crucible_lab_harness:run"
BASELINE_TARGETS = ("baseline:route", "baseline.py:route")
TARGET_RE = r"[A-Za-z_][A-Za-z0-9_./-]{0,119}(:[A-Za-z_][A-Za-z0-9_]{0,63})?"

DEFAULTS: dict[str, Any] = {
    "topology": "grid",            # grid | scale_free | waxman
    "n_nodes": 16,
    "shortcuts": 3,                # grid: express links (low latency, attract shortest paths)
    "ba_m": 2,                     # scale_free: links per new node
    "traffic": "gravity",          # gravity | hub
    "n_demands": 36,
    "n_hubs": 3,                   # hub traffic: datacenter nodes
    "hotspots": 1,
    "hotspot_mult": [2.5, 4.0],
    "noise_sigma": 0.12,           # per-demand lognormal noise of one snapshot (hidden)
    "peak_util": 1.05,             # calibration: baseline max utilisation at the noise-free daily peak (hidden)
    "drift_sigma": 0.06,           # daily lognormal drift of pair rates (hidden)
    "shock_p": 0.1,                # daily probability of a random shock (hidden)
    "shocks": [],                  # scheduled shocks [{"day": d, "kind": hotspot|hotspot_end|surge|shift, ...}]
    "queue_scale": 1.0,
    "drop_penalty_ms": 250.0,
    "sla_factor": 1.2,             # SLA = factor x baseline mean delay over the calibration day
    "sla_penalty_cents": 5_000,
    "sla_penalty_per_ms": 200,
    "sla_penalty_cap": 100_000,
    "experiment_fee_cents": 1_000,
    "scenario_fee_cents": 200,
    "max_scenarios": 24,
    "maintenance_hour": 2,
    "dashboard_delay_h": 2,
    "probe_fraction": 0.5,
    "measure_sigma": 0.04,
    "controller_timeout_s": 5.0,
    "operator": {"handle": "netops", "name": "NetOps", "lead": "netops-lead", "cash": 5_000_000},
    "engineer_seat": None,         # handle that receives a seat (member + lab.operate) when admitted via admin.join
}

# hidden regime fields (never in a non-internal event; tests check this)
HIDDEN_KEYS = ("base_rates", "hotspots", "diurnal", "volume_scale", "noise_sigma")


# ============================================================================ reference controllers (real Python)
BASELINE_SOURCE = '''"""Baseline routing controller: every demand on its single lowest-latency path (Dijkstra).

route(topology, demands) -> {"routes": {demand_id: [[path, 1.0]]}}
"""
import heapq


def _graph(topology):
    adj = {n: [] for n in topology["nodes"]}
    for link in topology["links"]:
        adj[link["u"]].append((link["v"], float(link["latency"])))
        adj[link["v"]].append((link["u"], float(link["latency"])))
    for n in adj:
        adj[n].sort()
    return adj


def shortest_path(adj, src, dst):
    dist = {src: 0.0}
    prev = {}
    done = set()
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")) - 1e-12:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in done:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def route(topology, demands):
    adj = _graph(topology)
    routes = {}
    for d in demands:
        p = shortest_path(adj, d["src"], d["dst"])
        if p is not None:
            routes[d["id"]] = [[p, 1.0]]
    return {"routes": routes}
'''

GREEDY_SOURCE = '''"""Congestion-aware greedy controller (reference).

Demands are placed largest first, each in three equal chunks; every chunk takes the path with the lowest marginal
cost given the load placed so far (queueing delay of the new load plus a penalty for traffic above capacity).
A demand therefore ends up on 1-3 paths with fractions in thirds.
"""
import heapq

RHO_MAX = 0.99
DROP_MS = 250.0


def _delay(lat, load, cap):
    r = min(load / cap, RHO_MAX)
    return lat + r / (1.0 - r)


def _marginal(link, load, x):
    lat, cap = float(link["latency"]), float(link["capacity"])
    new = (load + x) * _delay(lat, load + x, cap) - load * _delay(lat, load, cap)
    over = max(0.0, load + x - cap) - max(0.0, load - cap)
    return (new + DROP_MS * over) / x


def _best_path(adj, links, load, src, dst, x):
    dist = {src: 0.0}
    prev = {}
    done = set()
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dst:
            break
        for v, lid in adj[u]:
            nd = d + _marginal(links[lid], load[lid], x)
            if nd < dist.get(v, float("inf")) - 1e-12:
                dist[v] = nd
                prev[v] = (u, lid)
                heapq.heappush(heap, (nd, v))
    if dst not in done:
        return None, None
    nodes, lids = [dst], []
    while nodes[-1] != src:
        u, lid = prev[nodes[-1]]
        nodes.append(u)
        lids.append(lid)
    return nodes[::-1], lids


def route(topology, demands, chunks=3):
    links = {link["id"]: link for link in topology["links"]}
    adj = {n: [] for n in topology["nodes"]}
    for link in topology["links"]:
        adj[link["u"]].append((link["v"], link["id"]))
        adj[link["v"]].append((link["u"], link["id"]))
    for n in adj:
        adj[n].sort()
    load = {lid: 0.0 for lid in links}
    routes = {}
    for d in sorted(demands, key=lambda d: (-float(d["volume"]), d["id"])):
        vol = float(d["volume"])
        if vol <= 0:
            continue
        x = vol / chunks
        used = {}
        order = []
        for _ in range(chunks):
            nodes, lids = _best_path(adj, links, load, d["src"], d["dst"], x)
            if nodes is None:
                break
            key = tuple(nodes)
            if key not in used:
                used[key] = 0
                order.append(key)
            used[key] += 1
            for lid in lids:
                load[lid] += x
        if order:
            n = sum(used.values())
            routes[d["id"]] = [[list(k), used[k] / n] for k in order]
    return {"routes": routes}
'''

INVALID_SOURCE = '''"""A plausible but broken controller: assumes a full mesh and sends every demand over the
direct link. Wherever src and dst are not adjacent the path is invalid, so the lab drops that demand entirely."""


def route(topology, demands):
    return {"routes": {d["id"]: [[[d["src"], d["dst"]], 1.0]] for d in demands}}
'''

HARNESS_SOURCE = '''"""Installed by the Crucible lab next to your files (not part of your repository): calls your
controller once per scenario, seeding `random` (and numpy, if imported) with the scenario seed so that runs
are reproducible.
Exceptions raised while routing are reported as type and location only (messages could carry the traffic)."""
import importlib
import json
import os
import random
import sys
import traceback


def _where(e):
    tb = traceback.extract_tb(e.__traceback__)
    return " at %s:%d" % (os.path.basename(tb[-1].filename), tb[-1].lineno) if tb else ""


def run(job, caller=None):
    mod_name, _, fn_name = job["target"].partition(":")
    if mod_name.endswith(".py"):
        mod_name = mod_name[:-3]
    mod_name = mod_name.replace("/", ".")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name or "route")
    except BaseException as e:  # noqa: BLE001
        msg = str(e).replace(os.getcwd() + os.sep, "")
        return {"load_error": (type(e).__name__ + ": " + msg)[:300], "outputs": []}
    topo = json.dumps(job["topology"])
    outputs = []
    for sc in job["scenarios"]:
        random.seed(sc["seed"])
        np = sys.modules.get("numpy")
        if np is not None:
            np.random.seed(sc["seed"] % (2 ** 32))
        try:
            out = fn(json.loads(topo), sc["demands"])
            json.dumps(out)
            outputs.append({"ok": True, "out": out})
        except BaseException as e:  # noqa: BLE001
            outputs.append({"ok": False, "error": (type(e).__name__ + _where(e))[:300]})
    return {"outputs": outputs}
'''

CONTROLLER_README = """# Production routing controllers

`baseline.py` is the controller currently trusted in production: every demand on its lowest-latency path.

A controller is a function `route(topology, demands)` returning `{"routes": {demand_id: [[path, fraction], ...]}}`:

* `topology`: `{"nodes": [...], "links": [{"id", "u", "v", "capacity", "latency"}]}` (links are undirected,
  capacity is shared by both directions);
* `demands`: `[{"id": "src>dst", "src", "dst", "volume"}]`;
* each demand may use up to 3 distinct simple paths (node lists from src to dst over existing links); fractions
  must be positive and sum to 1. Anything else drops the whole demand.

Link delay is `latency + load/(capacity - load)` below capacity; above capacity the excess is dropped. Dropped
traffic counts as a 250 ms timeout in the mean delay. Test candidates with `lab.experiment`; deploy with
`lab.propose_change` (needs the `lab.operate` grant of the operator) at the next maintenance window.
"""

_COMPILED: dict[str, Any] = {}


def compile_controller(source: str, name: str = "route"):
    """Compile trusted controller source in-process (lab-owned reference code only)."""
    key = digest(source)
    if key not in _COMPILED:
        ns: dict = {"__name__": "crucible_lab_controller"}
        exec(compile(source, "<lab-controller>", "exec"), ns)  # trusted, lab-owned code only
        _COMPILED[key] = ns
    return _COMPILED[key][name]


def baseline_route(topology: dict, demands: list) -> dict:
    return compile_controller(BASELINE_SOURCE)(topology, demands)


# ============================================================================ physics
def link_delay(latency: float, load: float, capacity: float, queue_scale: float = 1.0) -> float:
    """Base latency plus M/M/1-style queueing load/(capacity - load) = rho/(1 - rho), rho capped at RHO_MAX
    (above capacity the queue is full and the excess is dropped instead)."""
    if capacity <= 0:
        rho = RHO_MAX
    else:
        rho = min(max(float(load), 0.0) / float(capacity), RHO_MAX)
    return float(latency) + float(queue_scale) * rho / (1.0 - rho)


def link_drop(load: float, capacity: float) -> float:
    """Fraction of the traffic crossing a link that is dropped: the excess over capacity, proportionally."""
    load = float(load)
    return (load - float(capacity)) / load if load > capacity else 0.0


class Network:
    """Index of a topology: node set, links by id, adjacency (u, v) -> link id in both directions."""

    def __init__(self, topology: dict):
        self.topology = topology
        self.nodes = [str(n) for n in topology["nodes"]]
        self.node_set = frozenset(self.nodes)
        self.links = {str(link["id"]): link for link in topology["links"]}
        self.link_ids = sorted(self.links)
        self.adj: dict[tuple[str, str], str] = {}
        self.neighbors: dict[str, list[str]] = {n: [] for n in self.nodes}
        for lid in self.link_ids:
            u, v = self.links[lid]["u"], self.links[lid]["v"]
            self.adj[(u, v)] = lid
            self.adj[(v, u)] = lid
            self.neighbors[u].append(v)
            self.neighbors[v].append(u)
        for n in self.nodes:
            self.neighbors[n].sort()

    def connected(self) -> bool:
        if not self.nodes:
            return True
        seen = {self.nodes[0]}
        stack = [self.nodes[0]]
        while stack:
            u = stack.pop()
            for v in self.neighbors[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(self.nodes)


class _Invalid(Exception):
    pass


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _validate_demand(net: Network, d: dict, entry) -> list[tuple[tuple[str, ...], float]]:
    if entry is None:
        raise _Invalid("no route")
    if not isinstance(entry, list) or not entry:
        raise _Invalid("route must be a non-empty list of [path, fraction]")
    if len(entry) > MAX_PATHS:
        raise _Invalid(f"more than {MAX_PATHS} paths")
    out: list[tuple[tuple[str, ...], float]] = []
    seen: set[tuple[str, ...]] = set()
    total = 0.0
    for item in entry:
        if not isinstance(item, list) or len(item) != 2:
            raise _Invalid("each path entry must be [path, fraction]")
        path, frac = item
        if not _is_num(frac) or frac <= 0 or frac > 1 + FRAC_TOL:
            raise _Invalid("bad fraction (must be a number in (0, 1])")
        if not isinstance(path, list) or len(path) < 2 or not all(isinstance(n, str) for n in path):
            raise _Invalid("path must be a list of at least two node ids")
        if path[0] != d["src"] or path[-1] != d["dst"]:
            raise _Invalid("path must start at src and end at dst")
        if not all(n in net.node_set for n in path):
            raise _Invalid("unknown node in path")
        if len(set(path)) != len(path):
            raise _Invalid("path is not simple (repeats a node)")
        links = []
        for a, b in zip(path, path[1:]):
            lid = net.adj.get((a, b))
            if lid is None:
                raise _Invalid(f"no link {a}-{b}")
            links.append(lid)
        key = tuple(path)
        if key in seen:
            raise _Invalid("duplicate path")
        seen.add(key)
        out.append((tuple(links), float(frac)))
        total += float(frac)
    if abs(total - 1.0) > FRAC_TOL:
        raise _Invalid("fractions do not sum to 1")
    return [(links, f / total) for links, f in out]


def validate_routes(net: Network, demands: list[dict], output) -> tuple[dict, dict]:
    """Strict validation of a controller's output. Returns ({demand_id: [(link_ids, fraction)]},
    {demand_id: reason}) - every demand is in exactly one of the two."""
    table = output.get("routes") if isinstance(output, dict) else None
    if not isinstance(table, dict):
        reason = "output must be {'routes': {demand_id: [[path, fraction], ...]}}"
        return {}, {d["id"]: reason for d in demands}
    ok, bad = {}, {}
    for d in demands:
        try:
            ok[d["id"]] = _validate_demand(net, d, table.get(d["id"]))
        except _Invalid as e:
            bad[d["id"]] = str(e)
    return ok, bad


def evaluate(net: Network, demands: list[dict], output, queue_scale: float = 1.0,
             drop_penalty_ms: float = 250.0) -> dict:
    """Deterministic flow model. Returns aggregate metrics, per-demand (volume, cost, dropped) and link utilisation.

    `cost` is the effective delay-volume: delivered volume x path delay + dropped volume x drop penalty."""
    demands = sorted(demands, key=lambda d: d["id"])
    routes, errors = validate_routes(net, demands, output)
    load = {lid: 0.0 for lid in net.link_ids}
    for d in demands:
        vol = float(d["volume"])
        for links, f in routes.get(d["id"], ()):
            for lid in links:
                load[lid] += vol * f
    delay, drop = {}, {}
    for lid in net.link_ids:
        link = net.links[lid]
        delay[lid] = link_delay(link["latency"], load[lid], link["capacity"], queue_scale)
        drop[lid] = link_drop(load[lid], link["capacity"])
    per: dict[str, list[float]] = {}
    tot_vol = tot_cost = tot_drop = deliv = deliv_cost = 0.0
    for d in demands:
        vol = float(d["volume"])
        if d["id"] in errors:
            cost, dropped = vol * drop_penalty_ms, vol
        else:
            cost = dropped = 0.0
            for links, f in routes[d["id"]]:
                x = vol * f
                keep, dl = 1.0, 0.0
                for lid in links:
                    keep *= 1.0 - drop[lid]
                    dl += delay[lid]
                dv = x * keep
                cost += dv * dl + (x - dv) * drop_penalty_ms
                dropped += x - dv
                deliv += dv
                deliv_cost += dv * dl
        per[d["id"]] = [vol, cost, dropped]
        tot_vol += vol
        tot_cost += cost
        tot_drop += dropped
    util = {lid: load[lid] / float(net.links[lid]["capacity"]) for lid in net.link_ids}
    return {"volume": tot_vol, "cost": tot_cost,
            "mean_delay": tot_cost / tot_vol if tot_vol > 0 else 0.0,
            "delivered_delay": deliv_cost / deliv if deliv > 0 else 0.0,
            "max_util": max(util.values()) if util else 0.0,
            "dropped": tot_drop, "dropped_fraction": tot_drop / tot_vol if tot_vol > 0 else 0.0,
            "invalid": len(errors), "errors": [f"{k}: {v}" for k, v in sorted(errors.items())[:5]],
            "per_demand": per, "util": util}


def summary(res: dict, nd: int = 6) -> dict:
    return {k: round(float(res[k]), nd) for k in ("mean_delay", "delivered_delay", "max_util", "dropped_fraction",
                                                   "volume", "cost")} | {"invalid": int(res["invalid"])}


# ============================================================================ network and traffic generators
def _node(i: int) -> str:
    return f"n{i}"


def make_topology(rng: np.random.Generator, cfg: dict) -> dict:
    n = int(min(40, max(12, cfg["n_nodes"])))
    kind = cfg["topology"]
    edges: dict[tuple[int, int], tuple[float, float]] = {}   # (i<j) -> (capacity, latency)

    def add(i, j, cap, lat):
        a, b = min(i, j), max(i, j)
        if a != b and (a, b) not in edges:
            edges[(a, b)] = (float(cap), round(float(lat), 2))

    if kind == "grid":
        rows = int(math.floor(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        pos = {i: (i // cols, i % cols) for i in range(n)}
        for i in range(n):
            r, c = pos[i]
            for j in (i + 1, i + cols):
                if j < n and (j == i + cols or pos[j][0] == r):
                    add(i, j, rng.choice([8, 10, 12, 14]), rng.uniform(2.0, 5.0))
        k = 0
        tries = 0
        while k < int(cfg.get("shortcuts", 3)) and tries < 200:
            tries += 1
            i, j = (int(x) for x in rng.choice(n, 2, replace=False))
            dist = abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1])
            if dist < 3 or (min(i, j), max(i, j)) in edges:
                continue
            add(i, j, rng.choice([10, 12]), 0.55 * dist * 3.5)
            k += 1
    elif kind == "scale_free":
        m = int(cfg.get("ba_m", 2))
        deg = [0] * n
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                add(i, j, 0, 0)
                deg[i] += 1
                deg[j] += 1
        for i in range(m + 1, n):
            w = np.array(deg[:i], dtype=float)
            targets = rng.choice(i, size=m, replace=False, p=w / w.sum())
            for j in sorted(int(t) for t in targets):
                add(i, j, 0, 0)
                deg[i] += 1
                deg[j] += 1
        for (a, b) in sorted(edges):
            cap = 4.0 * math.sqrt(deg[a] * deg[b]) * float(rng.uniform(0.8, 1.2))
            edges[(a, b)] = (float(round(cap)), round(float(rng.uniform(1.0, 8.0)), 2))
    elif kind == "waxman":
        xy = rng.uniform(0, 1, (n, 2))
        for i in range(n):
            d = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
            for j in [int(x) for x in np.argsort(d, kind="stable")[1:4]]:
                add(i, j, rng.choice([8, 10, 12, 16]), 1.0 + 12.0 * float(d[j]))
    else:
        raise InvalidAction(f"unknown topology generator {kind!r}")
    # make sure the network is connected (link components along a chain)
    topo = _topo_from(n, edges)
    net = Network(topo)
    while not net.connected():
        comp = _component(net, net.nodes[0])
        outside = sorted((int(x[1:]) for x in net.nodes if x not in comp))
        inside = sorted(int(x[1:]) for x in comp)
        add(inside[-1], outside[0], 10, rng.uniform(2.0, 6.0))
        topo = _topo_from(n, edges)
        net = Network(topo)
    topo["kind"] = kind
    return topo


def _topo_from(n: int, edges: dict) -> dict:
    links = []
    for k, (a, b) in enumerate(sorted(edges)):
        cap, lat = edges[(a, b)]
        links.append({"id": f"l{k}", "u": _node(a), "v": _node(b), "capacity": cap, "latency": lat})
    return {"nodes": [_node(i) for i in range(n)], "links": links}


def _component(net: Network, start: str) -> set:
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in net.neighbors[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def make_regime(rng: np.random.Generator, topo: dict, cfg: dict) -> dict:
    """Hidden traffic parameters (unscaled; `volume_scale` is set by calibration)."""
    nodes = topo["nodes"]
    n = len(nodes)
    k = int(min(cfg["n_demands"], n * (n - 1)))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if cfg["traffic"] == "gravity":
        mass = rng.lognormal(0.0, 0.7, n)
        w = np.array([mass[i] * mass[j] for i, j in pairs])
        diurnal = {"kind": "sine", "amp": round(float(rng.uniform(0.3, 0.5)), 4),
                   "peaks": [round(float(rng.uniform(13.0, 20.0)), 3)], "width": 0.0}
    elif cfg["traffic"] == "hub":
        nh = int(min(max(1, cfg.get("n_hubs", 3)), n - 1))
        hubs = set(int(x) for x in rng.choice(n, nh, replace=False))
        w = np.array([(3.0 if i in hubs and j not in hubs else 1.0 if j in hubs and i not in hubs else
                       2.0 if i in hubs and j in hubs else 0.05) for i, j in pairs])
        w = w * rng.lognormal(0.0, 0.4, len(pairs))
        diurnal = {"kind": "bimodal", "amp": round(float(rng.uniform(0.35, 0.55)), 4),
                   "peaks": [round(float(rng.uniform(8.0, 10.5)), 3), round(float(rng.uniform(19.0, 22.0)), 3)],
                   "width": round(float(rng.uniform(2.0, 3.0)), 3)}
    else:
        raise InvalidAction(f"unknown traffic structure {cfg['traffic']!r}")
    idx = rng.choice(len(pairs), size=k, replace=False, p=w / w.sum())
    chosen = sorted(int(x) for x in idx)
    mean_w = float(np.mean([w[c] for c in chosen]))
    rates = [[nodes[pairs[c][0]], nodes[pairs[c][1]], round(float(w[c] / mean_w), 6)] for c in chosen]
    lo, hi = cfg.get("hotspot_mult", [2.5, 4.0])
    hot = []
    for c in sorted(int(x) for x in rng.choice(len(rates), size=min(int(cfg["hotspots"]), len(rates)), replace=False)):
        hot.append([rates[c][0], rates[c][1], round(float(rng.uniform(lo, hi)), 4)])
    return {"kind": cfg["traffic"], "base_rates": rates, "hotspots": hot, "diurnal": diurnal,
            "volume_scale": 1.0, "noise_sigma": float(cfg["noise_sigma"])}


def diurnal_mult(d: dict, hod: float) -> float:
    if d["kind"] == "sine":
        return 1.0 + d["amp"] * math.cos(2.0 * math.pi * (hod - d["peaks"][0]) / 24.0)
    bump = 0.0
    for p in d["peaks"]:
        dh = abs(hod - p) % 24.0
        dh = min(dh, 24.0 - dh)
        bump = max(bump, math.exp(-dh * dh / (2.0 * d["width"] ** 2)))
    return 1.0 - d["amp"] / 2.0 + d["amp"] * bump


def sample_demands(regime: dict, hod: int, seed: int | None) -> list[dict]:
    """One demand snapshot of the hidden process at hour-of-day `hod`. `seed=None` gives the noise-free mean."""
    rates = regime["base_rates"]
    mult = diurnal_mult(regime["diurnal"], float(hod))
    hot = {(s, t): m for s, t, m in regime["hotspots"]}
    if seed is None:
        noise = [1.0] * len(rates)
    else:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        noise = [float(z) for z in rng.lognormal(0.0, regime["noise_sigma"], len(rates))]
    scale = regime["volume_scale"]
    out = []
    for (s, t, r), z in zip(rates, noise):
        out.append({"id": f"{s}>{t}", "src": s, "dst": t,
                    "volume": round(scale * r * mult * hot.get((s, t), 1.0) * z, 4)})
    return out


def calibrate(net: Network, regime: dict, cfg: dict) -> tuple[float, float]:
    """(volume_scale, sla_ms): scale so that the baseline's worst noise-free hour reaches `peak_util`; SLA is
    `sla_factor` x the baseline's volume-weighted mean delay over that calibrated day."""
    topo = net.topology
    peak = 0.0
    for hod in range(24):
        r = evaluate(net, sample_demands(regime, hod, None), baseline_route(topo, sample_demands(regime, hod, None)))
        peak = max(peak, r["max_util"])
    scale = round(float(cfg["peak_util"]) / peak, 6) if peak > 0 else 1.0
    scaled = dict(regime, volume_scale=scale)
    cost = vol = 0.0
    for hod in range(24):
        dem = sample_demands(scaled, hod, None)
        r = evaluate(net, dem, baseline_route(topo, dem), cfg["queue_scale"], cfg["drop_penalty_ms"])
        cost += r["cost"]
        vol += r["volume"]
    return scale, round(float(cfg["sla_factor"]) * cost / vol, 3)


def drift_regime(rng: np.random.Generator, regime: dict, nodes: list[str], cfg: dict, shocks: list[dict]) -> dict:
    """Daily evolution of the hidden process: lognormal drift of pair rates, small phase wander, then shocks."""
    sig = float(cfg["drift_sigma"])
    z = rng.lognormal(0.0, sig, len(regime["base_rates"])) if sig > 0 else [1.0] * len(regime["base_rates"])
    rates = [[s, t, round(r * float(f), 6)] for (s, t, r), f in zip(regime["base_rates"], z)]
    d = dict(regime["diurnal"])
    d["peaks"] = [round(p + float(rng.normal(0.0, 0.3)), 3) % 24.0 for p in d["peaks"]]
    new = dict(regime, base_rates=rates, diurnal=d, hotspots=[list(h) for h in regime["hotspots"]])
    lo, hi = cfg.get("hotspot_mult", [2.5, 4.0])
    for sh in shocks:
        kind = sh.get("kind")
        if kind == "hotspot":
            s, t = sh.get("src"), sh.get("dst")
            if s is None or t is None or s == t:
                i, j = (int(x) for x in rng.choice(len(nodes), 2, replace=False))
                s, t = nodes[i], nodes[j]
            mult = float(sh.get("mult", rng.uniform(lo * 1.2, hi * 1.5)))
            if not any(r[0] == s and r[1] == t for r in new["base_rates"]):
                med = float(np.median([r[2] for r in new["base_rates"]]))
                new["base_rates"] = sorted(new["base_rates"] + [[s, t, round(med, 6)]])
            new["hotspots"] = sorted([h for h in new["hotspots"] if (h[0], h[1]) != (s, t)] +
                                     [[s, t, round(mult, 4)]])
        elif kind == "hotspot_end" and new["hotspots"]:
            k = int(rng.integers(0, len(new["hotspots"])))
            new["hotspots"] = [h for i, h in enumerate(new["hotspots"]) if i != k]
        elif kind == "surge":
            new["volume_scale"] = round(new["volume_scale"] * float(sh.get("factor", rng.uniform(1.08, 1.2))), 6)
        elif kind == "shift":
            delta = float(sh.get("hours", rng.uniform(-4.0, 4.0)))
            d2 = dict(new["diurnal"])
            d2["peaks"] = [round((p + delta) % 24.0, 3) for p in d2["peaks"]]
            new["diurnal"] = d2
    return new


def _target_file(target: str) -> str:
    mod = target.partition(":")[0]
    return mod if mod.endswith(".py") else mod.replace(".", "/") + ".py"


# ============================================================================ the module
@register_module
class Lab(Module):
    name = "lab"
    host = "lab"
    depends = ("identity", "bank", "code")

    def __init__(self, config=None):
        super().__init__(config)
        cfg = dict(DEFAULTS)
        cfg.update(self.config)
        cfg["operator"] = dict(DEFAULTS["operator"], **(self.config.get("operator") or {}))
        self.cfg = cfg

    def install(self, world):
        st = world.state
        st.declare_table("lab")                       # conf, topology, deployed, pending, regime, totals
        st.declare_table("lab_regimes")               # hidden traffic parameters, by version
        st.declare_table("lab_hours")                 # production hour records (exact metrics are internal)
        st.declare_table("lab_experiments", index=("by",))
        st.declare_table("lab_changes", index=("by",))
        prev = world.hooks.get("on_join")

        def on_join(ctx, pid, kind):
            if prev is not None:
                prev(ctx, pid, kind)
            self._seat(ctx, pid)
        world.hooks["on_join"] = on_join

    # ------------------------------------------------------------------ helpers
    def net(self, world) -> Network:
        cache = world.runtime.get("lab_net")
        if cache is None:
            cache = Network(to_jsonable(world.state.get("lab", "topology")["topology"]))
            world.runtime["lab_net"] = cache
        return cache

    def conf(self, state) -> dict:
        c = state.get("lab", "conf")
        if c is None:
            raise NotFound("no lab in this world")
        return c

    def regime(self, state, rid: str | None = None) -> tuple[str, dict]:
        rid = rid or state.get("lab", "regime")["id"]
        return rid, to_jsonable(state.get("lab_regimes", rid)["params"])

    def _seat(self, ctx: Ctx, pid: str) -> None:
        seat = self.cfg.get("engineer_seat")
        if not seat or pid != f"p:{seat}" or not ctx.state.has("lab", "conf"):
            return
        conf = self.conf(ctx.state)
        ident(ctx.world).add_member(ctx, conf["operator"], pid, "member", "network engineer")
        ctx.call("id.grant", {"grantee": pid, "cap": "lab.operate", "owner": conf["operator"], "scope": "*"},
                 as_principal=conf["lead"])

    def _next_maintenance(self, state) -> int | None:
        j = state.get("sched", "lab_maint")
        return j["at"] if j else None

    def _run(self, ctx: Ctx, repo: str, ref: str, target: str, principal: str, scenarios: list[dict],
             trusted: bool, recorded: str) -> dict:
        """Run a controller on scenarios [{"seed", "demands"}]. Returns {"ok", "outputs": [{"ok", "out"|"error"}],
        "load_error", "error", "cpu_ms"}. `recorded` = "cmd" (bridge: full outcome incl. measured CPU, injected on
        regeneration) or "proc" (only the deterministic part of the outcome is recorded: scheduled processes are
        re-executed during regeneration)."""
        net = self.net(ctx.world)
        topo = {"nodes": net.nodes, "links": [dict(net.links[lid]) for lid in net.link_ids]}
        if trusted:
            outs = [{"ok": True, "out": baseline_route(topo, sc["demands"])} for sc in scenarios]
            return {"ok": True, "outputs": outs, "load_error": None, "error": None, "cpu_ms": 0.0}
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo, principal):
            raise Forbidden(f"{principal} cannot read {repo}")
        files = dict(code.files_with_deps(ctx, repo, ref, principal))
        files[HARNESS_FILE] = HARNESS_SOURCE
        pool = code.sandbox_pool(ctx.world)
        job = {"kind": "call", "target": HARNESS_TARGET, "caller": principal,
               "args": {"target": target, "topology": topo, "scenarios": scenarios}}
        timeout = min(30.0, float(self.cfg["controller_timeout_s"]) * max(1, len(scenarios)) + 2.0)
        from ..sandbox.runner import Worker
        if recorded == "cmd":
            from ..sandbox.bridge import run_in_ctx
            res = run_in_ctx(ctx, "lab.controller", lambda: Worker(files, pool.cfg, "lab").start(), job, principal,
                             timeout=timeout, close_after=True, world_access=False)
        else:
            def go():
                w = Worker(files, pool.cfg, "lab").start()
                try:
                    r = w.run(job, rpc=None, timeout=timeout)
                finally:
                    w.close()
                err = r.get("error")
                return {"ok": bool(r.get("ok")), "result": r.get("result"),
                        "error": {"type": err.get("type", "Error")} if isinstance(err, dict) else None, "cpu_ms": 0.0}
            res = ctx.external("lab.controller", go)
        result = res.get("result") if isinstance(res.get("result"), dict) else {}
        raw = result.get("outputs") if isinstance(result.get("outputs"), list) else []
        outs = [raw[i] if i < len(raw) and isinstance(raw[i], dict) else {"ok": False, "error": "no output"}
                for i in range(len(scenarios))]
        return {"ok": bool(res.get("ok")), "outputs": outs, "load_error": result.get("load_error"),
                "error": res.get("error"), "cpu_ms": float(res.get("cpu_ms") or 0.0)}

    @staticmethod
    def _controller_error(run: dict) -> dict | None:
        if run["load_error"]:
            return {"type": "LoadError", "message": str(run["load_error"])[:300]}
        if not run["ok"]:
            e = run["error"] or {}
            return {"type": e.get("type", "Error"), "message": str(e.get("message", ""))[:300]}
        return None

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        cfg = self.cfg
        w = ctx.world
        idm, bank, code = ident(w), w.modules["bank"], w.modules["code"]
        rng = w.rng_for("lab-genesis")
        topo = make_topology(rng, cfg)
        net = Network(topo)
        regime = make_regime(rng, topo, cfg)
        scale, sla = calibrate(net, regime, cfg)
        regime["volume_scale"] = scale
        op = cfg["operator"]
        lead = idm.create_person(ctx, op["lead"], f"{op['name']} operations lead", "npc", {"role": "lab operator"})
        bank.open_account(ctx, lead, "checking")
        org = idm.create_org(ctx, op["handle"], op["name"], lead, "company", {"lab": True})
        acct = bank.open_account(ctx, org, "operating")
        if int(op.get("cash", 0)) > 0:
            bank.move(ctx, "acct_genesis", acct, int(op["cash"]), "endowment", f"initial capital of {org}")
        rid = code.create_repo(ctx, org, "controllers", "public", f"Production routing controllers of {op['name']}",
                               "open")
        commit = code.commit_files(ctx, rid, {"baseline.py": BASELINE_SOURCE, "README.md": CONTROLLER_README},
                                   "baseline controller (lowest-latency single path)", lead)
        ctx.emit("lab.topology", {"topology": topo}, vis="public")
        ctx.emit("lab.conf", {"operator": org, "lead": lead, "account": acct, "repo": rid, "baseline_ref": commit,
                              "sla_ms": sla, "queue_scale": float(cfg["queue_scale"]),
                              "drop_penalty_ms": float(cfg["drop_penalty_ms"]),
                              "experiment_fee_cents": int(cfg["experiment_fee_cents"]),
                              "scenario_fee_cents": int(cfg["scenario_fee_cents"]),
                              "max_scenarios": int(cfg["max_scenarios"]),
                              "maintenance_hour": int(cfg["maintenance_hour"]) % 24,
                              "dashboard_delay_h": int(cfg["dashboard_delay_h"])}, vis="public")
        rgid = ctx.state.new_id("rg")
        ctx.emit("lab.regime", {"id": rgid, "day": 0, "cause": "genesis", "params": regime}, vis="internal")
        ctx.emit("lab.deployed", {"id": "dep_genesis", "repo": rid, "ref": commit, "target": "baseline:route",
                                  "by": org, "principal": lead, "trusted": True, "change": None}, vis="public")
        docs = w.modules.get("docs")
        if docs is not None:
            docs.create_doc(ctx, org, f"{op['name']} network lab: how to work with production",
                            CONTROLLER_README + "\nEndpoints: lab.topology, lab.measure (delayed, sampled, noisy "
                            "dashboard), lab.experiment (paid offline tests on fresh traffic samples), "
                            f"lab.propose_change, lab.status. Production SLA: mean delay <= {sla} ms.\n", "public")
        mh = int(cfg["maintenance_hour"]) % 24
        ctx.schedule(HOUR, "lab.hour", {}, prio=40, period=HOUR, key="lab_hour")
        ctx.schedule(DAY, "lab.daily", {}, prio=45, period=DAY, key="lab_daily")
        ctx.schedule(mh * HOUR if mh else DAY, "lab.maintenance", {}, prio=60, period=DAY, key="lab_maint")

    # ------------------------------------------------------------------ endpoints
    @endpoint("topology", S.obj(), read_only=True, cost_s=5)
    def ep_topology(self, ctx):
        """The production network (public): nodes, undirected links with capacity and base latency (ms)."""
        t = ctx.state.get("lab", "topology")
        if t is None:
            raise NotFound("no lab in this world")
        return to_jsonable(t["topology"])

    @endpoint("status", S.obj(), read_only=True)
    def ep_status(self, ctx):
        """Operator, deployed controller, pending change (operators only), next maintenance window, SLA, prices."""
        st = ctx.state
        conf = self.conf(st)
        dep = st.get("lab", "deployed")
        pend = st.get("lab", "pending")
        operates = ident(ctx.world).allowed(ctx, ctx.actor, conf["operator"], "lab.operate")
        hours = st.ids("lab_hours")
        last = st.tables["lab_hours"][hours[-1]]["hour"] if hours else None
        totals = st.get("lab", "totals") or {}
        return {"operator": conf["operator"], "sla_ms": conf["sla_ms"],
                "deployed": {k: dep[k] for k in ("repo", "ref", "target", "by", "since")},
                "pending": (None if pend is None else {k: pend[k] for k in ("change", "repo", "ref", "target", "by")})
                if operates else (pend is not None),
                "next_maintenance": self._next_maintenance(st), "maintenance_hour": conf["maintenance_hour"],
                "dashboard_delay_h": conf["dashboard_delay_h"],
                "last_published_hour": None if last is None else last - conf["dashboard_delay_h"],
                "sla_penalties_cents": int(totals.get("penalties", 0)),
                "prices": {"experiment_fee_cents": conf["experiment_fee_cents"],
                           "scenario_fee_cents": conf["scenario_fee_cents"],
                           "max_scenarios": conf["max_scenarios"]},
                "you_operate": operates}

    @endpoint("measure", S.obj({"window_h": S.s_int("hours to look back", minimum=1, maximum=24 * 14)}),
              read_only=True, cost_s=5)
    def ep_measure(self, ctx, window_h=24):
        """Production dashboard: per-hour metrics measured by probes on a sample of demands, with measurement noise,
        published `dashboard_delay_h` hours late. Mean delay includes dropped traffic at the timeout penalty."""
        st = ctx.state
        conf = self.conf(st)
        hours = st.ids("lab_hours")
        if not hours:
            return {"rows": [], "summary": None, "delay_h": conf["dashboard_delay_h"]}
        latest = st.tables["lab_hours"][hours[-1]]["hour"] - conf["dashboard_delay_h"]
        lo = ctx.now // HOUR - window_h
        rows = []
        for hid in hours:
            r = st.tables["lab_hours"][hid]
            if lo <= r["hour"] <= latest:
                rows.append({"hour": r["hour"], "hour_of_day": r["hour"] % 24, **to_jsonable(r["measured"])})
        if not rows:
            return {"rows": [], "summary": None, "delay_h": conf["dashboard_delay_h"]}
        n = len(rows)
        summ = {k: round(sum(r[k] for r in rows) / n, 4) for k in ("mean_delay", "max_util", "dropped_fraction")}
        summ["peak_util"] = round(max(r["max_util"] for r in rows), 4)
        summ["sla_breach_hours"] = sum(1 for r in rows if r["mean_delay"] > conf["sla_ms"])
        return {"rows": rows, "summary": summ, "delay_h": conf["dashboard_delay_h"], "sla_ms": conf["sla_ms"]}

    @endpoint("experiment", S.obj({
        "repo": S.s_str("repository holding the controller", maxLength=81),
        "target": S.s_str("module:function, e.g. controller.py:route", maxLength=184),
        "scenarios": S.s_int("number of fresh traffic snapshots to test on", minimum=1, maximum=48),
        "account": S.s_str("account paying the lab fee (default: your primary account)", maxLength=60),
        "ref": S.s_str("commit (default: head)", maxLength=40),
        "hours": S.s_arr(S.s_int(minimum=0, maximum=23), "hours of day to sample, one scenario each (overrides "
                         "scenarios; default: random hours)", maxItems=48)},
        ["repo", "target"]), cost_s=300)
    def ep_experiment(self, ctx, repo, target, scenarios=4, account=None, ref=None, hours=None):
        """Offline test of a controller on freshly sampled demand snapshots of the current traffic process (not
        production's). Returns measured metrics for the candidate and, on the same snapshots, the baseline.
        Costs a lab fee (paid to the operator) plus sandbox compute."""
        st = ctx.state
        conf = self.conf(st)
        if not re.fullmatch(TARGET_RE, target):
            raise InvalidAction("target must look like module.py:function")
        k = len(hours) if hours else int(scenarios)
        if k > conf["max_scenarios"]:
            raise InvalidAction(f"at most {conf['max_scenarios']} scenarios per experiment")
        bank = ctx.world.modules["bank"]
        acct = bank.resolve_account(st, account) if account else bank.primary_account(st, ctx.actor)
        if acct is None:
            raise NotFound("no account to pay from")
        payer = bank.account_owner(st, acct)
        fee = int(conf["experiment_fee_cents"]) + int(conf["scenario_fee_cents"]) * k
        ident(ctx.world).authorize(ctx, payer, "bank.spend", amount=fee, scope=acct)
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        r = code.repo(st, repo)
        cid = ref or r["head"]
        if cid is None:
            raise InvalidAction(f"{repo} is empty")
        if _target_file(target) not in code.tree(st, repo, cid):
            raise NotFound(f"{_target_file(target)} not in {repo}@{cid}")
        exp = st.new_id("labx")
        tid = bank.move(ctx, acct, conf["account"], fee, "lab_fee", f"lab experiment {exp}", exp)
        rng = ctx.rng("lab-experiment")
        hods = [int(h) for h in hours] if hours else [int(x) for x in rng.integers(0, 24, k)]
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, k)]
        rgid, regime = self.regime(st)
        scen = [{"seed": s, "demands": sample_demands(regime, h, s)} for h, s in zip(hods, seeds)]
        run = self._run(ctx, repo, cid, target, ctx.actor, scen, False, "cmd")
        compute = ctx.world.modules["code"].charge_compute(ctx, payer, run["cpu_ms"], f"lab experiment {repo}")
        net = self.net(ctx.world)
        qs, pen = conf["queue_scale"], conf["drop_penalty_ms"]
        per, errors = [], []
        tot = {"cost": 0.0, "volume": 0.0, "dropped": 0.0, "bcost": 0.0, "bdropped": 0.0}
        max_u = bmax_u = 0.0
        invalid = 0
        topo = net.topology
        for h, sc, o in zip(hods, scen, run["outputs"]):
            res = evaluate(net, sc["demands"], o.get("out") if o.get("ok") else None, qs, pen)
            base = evaluate(net, sc["demands"], baseline_route(topo, sc["demands"]), qs, pen)
            tot["cost"] += res["cost"]
            tot["volume"] += res["volume"]
            tot["dropped"] += res["dropped"]
            tot["bcost"] += base["cost"]
            tot["bdropped"] += base["dropped"]
            max_u, bmax_u = max(max_u, res["max_util"]), max(bmax_u, base["max_util"])
            invalid += res["invalid"]
            if not o.get("ok"):
                errors.append(f"controller raised: {o.get('error', '')}"[:300])
            errors.extend(res["errors"][: max(0, 5 - len(errors))])
            per.append({"hour_of_day": h, "mean_delay": round(res["mean_delay"], 4),
                        "baseline_mean_delay": round(base["mean_delay"], 4), "max_util": round(res["max_util"], 4),
                        "baseline_max_util": round(base["max_util"], 4),
                        "dropped_fraction": round(res["dropped_fraction"], 4), "invalid": res["invalid"]})
        vol = tot["volume"] or 1.0
        metrics = {"mean_delay": round(tot["cost"] / vol, 4), "max_util": round(max_u, 4),
                   "dropped_fraction": round(tot["dropped"] / vol, 4), "invalid_demands": invalid}
        baseline = {"mean_delay": round(tot["bcost"] / vol, 4), "max_util": round(bmax_u, 4),
                    "dropped_fraction": round(tot["bdropped"] / vol, 4)}
        improvement = round((tot["bcost"] - tot["cost"]) / tot["bcost"], 6) if tot["bcost"] > 0 else 0.0
        cerr = self._controller_error(run)
        out = {"experiment": exp, "repo": repo, "ref": cid, "target": target, "scenarios": k, "metrics": metrics,
               "baseline": baseline, "improvement_vs_baseline": improvement, "per_scenario": per,
               "errors": errors[:5], "controller_error": cerr, "cpu_ms": round(run["cpu_ms"], 3),
               "fee_cents": fee, "compute_cents": compute, "txn": tid}
        ctx.emit("lab.experiment", {"id": exp, "by": ctx.actor, "payer": payer, "repo": repo, "ref": cid,
                                    "target": target, "scenarios": k, "metrics": metrics, "baseline": baseline,
                                    "improvement": improvement, "fee": fee}, vis=(ctx.actor, payer))
        ctx.emit("lab.experiment_samples", {"id": exp, "regime": rgid, "hours": hods, "seeds": seeds},
                 vis="internal")
        return out

    @endpoint("propose_change", S.obj({
        "repo": S.s_str(maxLength=81), "target": S.s_str("module:function", maxLength=184),
        "ref": S.s_str("commit to deploy (default: current head, pinned now)", maxLength=40),
        "note": S.s_str(maxLength=500)}, ["repo", "target"]), cost_s=60)
    def ep_propose_change(self, ctx, repo, target, ref=None, note=""):
        """Schedule deployment of a controller to production at the next maintenance window (replaces any pending
        change). Requires the `lab.operate` capability of the operator organisation."""
        st = ctx.state
        conf = self.conf(st)
        ident(ctx.world).authorize(ctx, conf["operator"], "lab.operate")
        if not re.fullmatch(TARGET_RE, target):
            raise InvalidAction("target must look like module.py:function")
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        cid = ref or code.repo(st, repo)["head"]
        if cid is None:
            raise InvalidAction(f"{repo} is empty")
        if _target_file(target) not in code.tree(st, repo, cid):
            raise NotFound(f"{_target_file(target)} not in {repo}@{cid}")
        chg = st.new_id("chg")
        at = self._next_maintenance(st)
        ctx.emit("lab.change", {"id": chg, "repo": repo, "ref": cid, "target": target, "by": ctx.actor,
                                "note": note[:500], "deploy_at": at}, vis=(ctx.actor, conf["operator"]))
        return {"change": chg, "ref": cid, "deploy_at": at}

    # ------------------------------------------------------------------ processes
    @process("hour")
    def pr_hour(self, ctx):
        st = ctx.state
        h = ctx.now // HOUR - 1
        if h < 0 or not st.has("lab", "conf"):
            return
        conf = self.conf(st)
        dep = st.get("lab", "deployed")
        rgid, regime = self.regime(st)
        rng = ctx.rng("lab-hour", h)
        seed = int(rng.integers(0, 2**31 - 1))
        demands = sample_demands(regime, h % 24, seed)
        try:
            run = self._run(ctx, dep["repo"], dep["ref"], dep["target"], dep["principal"],
                            [{"seed": seed, "demands": demands}], bool(dep["trusted"]), "proc")
        except CrucibleError as e:     # repository gone / access revoked: production loses its controller
            run = {"ok": False, "outputs": [{"ok": False, "error": e.message}], "load_error": None,
                   "error": {"type": e.code, "message": e.message}, "cpu_ms": 0.0}
        o = run["outputs"][0]
        net = self.net(ctx.world)
        res = evaluate(net, demands, o.get("out") if o.get("ok") else None, conf["queue_scale"],
                       conf["drop_penalty_ms"])
        # probe-sampled, noisy measurement (what the public dashboard shows)
        frac = float(self.cfg["probe_fraction"])
        sig = float(self.cfg["measure_sigma"])
        ids = sorted(res["per_demand"])
        pick = rng.random(len(ids)) < frac
        probe = [i for i, p in zip(ids, pick) if p] or ids[:1]
        pv = sum(res["per_demand"][i][0] for i in probe) or 1.0
        pc = sum(res["per_demand"][i][1] for i in probe)
        pdrop = sum(res["per_demand"][i][2] for i in probe)
        z = rng.normal(0.0, 1.0, 3)
        measured = {"mean_delay": round(pc / pv * math.exp(sig * float(z[0])), 4),
                    "max_util": round(res["max_util"] * math.exp(sig / 2 * float(z[1])), 4),
                    "dropped_fraction": round(min(1.0, pdrop / pv), 4),
                    "volume": round(res["volume"] * math.exp(sig * float(z[2])), 3)}
        exact = summary(res)
        penalty = due = 0
        if exact["mean_delay"] > conf["sla_ms"]:
            c = self.cfg
            due = int(min(c["sla_penalty_cap"], c["sla_penalty_cents"] +
                          c["sla_penalty_per_ms"] * (exact["mean_delay"] - conf["sla_ms"])))
            bank = ctx.world.modules["bank"]
            penalty = max(0, min(due, bank.available(st, conf["account"])))
            if penalty > 0:
                bank.move(ctx, conf["account"], "acct_external", penalty, "sla_penalty",
                          f"SLA penalty for hour {h}: mean delay {exact['mean_delay']:.2f} > {conf['sla_ms']} ms")
        ctx.emit("lab.hour", {"id": f"h{h:06d}", "hour": h, "regime": rgid, "seed": seed, "deployment": dep["id"],
                              "exact": exact, "measured": measured, "controller_ok": bool(o.get("ok")),
                              "controller_error": self._controller_error(run), "errors": res["errors"][:3],
                              "demands": digest(demands), "penalty": penalty, "penalty_due": due}, vis="internal")
        old = st.get("lab_hours", f"h{h - conf['dashboard_delay_h']:06d}")
        if old is not None:
            ctx.emit("lab.dashboard", {"hour": old["hour"], "hour_of_day": old["hour"] % 24,
                                       **to_jsonable(old["measured"])}, vis="public")

    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        if not st.has("lab", "conf"):
            return
        day = ctx.now // DAY
        rgid, regime = self.regime(st)
        rng = ctx.rng("lab-drift", day)
        shocks = [dict(s) for s in self.cfg.get("shocks", []) if int(s.get("day", -1)) == day]
        if float(rng.random()) < float(self.cfg["shock_p"]):
            shocks.append({"kind": str(rng.choice(["hotspot", "hotspot", "hotspot_end", "surge", "shift"]))})
        new = drift_regime(rng, regime, self.net(ctx.world).nodes, self.cfg, shocks)
        cause = "drift" + "".join(f"+{s['kind']}" for s in shocks)
        ctx.emit("lab.regime", {"id": st.new_id("rg"), "day": day, "cause": cause, "params": new}, vis="internal")

    @process("maintenance")
    def pr_maintenance(self, ctx):
        st = ctx.state
        pend = st.get("lab", "pending")
        if pend is None:
            return
        conf = self.conf(st)
        trusted = (pend["repo"] == conf["repo"] and pend["ref"] == conf["baseline_ref"]
                   and pend["target"] in BASELINE_TARGETS)
        ctx.emit("lab.deployed", {"id": st.new_id("dep"), "repo": pend["repo"], "ref": pend["ref"],
                                  "target": pend["target"], "by": pend["by"], "principal": pend["by"],
                                  "trusted": trusted, "change": pend["change"]}, vis="public")

    # ------------------------------------------------------------------ appliers
    @applier("lab.topology")
    def ap_topology(self, ev):
        self.world.state.put("lab", "topology", {"topology": to_jsonable(ev.payload["topology"])})
        self.world.runtime.pop("lab_net", None)

    @applier("lab.conf")
    def ap_conf(self, ev):
        self.world.state.put("lab", "conf", to_jsonable(ev.payload))
        self.world.state.put("lab", "totals", {"penalties": 0, "fees": 0, "experiments": 0})

    @applier("lab.regime")
    def ap_regime(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("lab_regimes", p["id"], {"day": p["day"], "cause": p["cause"], "params": to_jsonable(p["params"]),
                                           "t": ev.t})
        st.put("lab", "regime", {"id": p["id"]})

    @applier("lab.deployed")
    def ap_deployed(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("lab", "deployed", dict(to_jsonable(p), since=ev.t))
        if p.get("change"):
            st.delete("lab", "pending")
            st.patch("lab_changes", p["change"], status="deployed", deployed_at=ev.t)

    @applier("lab.change")
    def ap_change(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.get("lab", "pending")
        if old is not None:
            st.patch("lab_changes", old["change"], status="superseded")
        st.insert("lab_changes", p["id"], dict(to_jsonable(p), t=ev.t, status="pending"))
        st.put("lab", "pending", {"change": p["id"], "repo": p["repo"], "ref": p["ref"], "target": p["target"],
                                  "by": p["by"]})

    @applier("lab.hour")
    def ap_hour(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("lab_hours", p["id"], dict(to_jsonable(p), t=ev.t, published=False))
        tot = st.get("lab", "totals")
        st.patch("lab", "totals", penalties=tot["penalties"] + p["penalty"])

    @applier("lab.dashboard")
    def ap_dashboard(self, ev):
        self.world.state.patch("lab_hours", f"h{ev.payload['hour']:06d}", published=True)

    @applier("lab.experiment")
    def ap_experiment(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("lab_experiments", p["id"], dict(to_jsonable(p), t=ev.t))
        tot = st.get("lab", "totals")
        st.patch("lab", "totals", fees=tot["fees"] + p["fee"], experiments=tot["experiments"] + 1)

    @applier("lab.experiment_samples")
    def ap_experiment_samples(self, ev):
        p = ev.payload
        self.world.state.patch("lab_experiments", p["id"], regime=p["regime"], hours=list(p["hours"]),
                               seeds=list(p["seeds"]))

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        if not st.has("lab", "conf"):
            return []
        errs = []
        conf = st.get("lab", "conf")
        dep = st.get("lab", "deployed")
        if dep is None or not st.has("repos", dep["repo"]) or not st.has("commits", dep["ref"]):
            errs.append("deployed controller missing")
        if not st.has("lab_regimes", st.get("lab", "regime")["id"]):
            errs.append("current regime missing")
        if not Network(to_jsonable(st.get("lab", "topology")["topology"])).connected():
            errs.append("network disconnected")
        hours = [st.tables["lab_hours"][h]["hour"] for h in st.ids("lab_hours")]
        if hours and hours != list(range(hours[0], hours[0] + len(hours))):
            errs.append("production hours not contiguous")
        if hours and hours[-1] != world.now // HOUR - 1:
            errs.append(f"production behind the clock: last hour {hours[-1]}, now {world.now // HOUR}")
        pen = 0
        for hid in st.ids("lab_hours"):
            r = st.tables["lab_hours"][hid]
            e = r["exact"]
            pen += r["penalty"]
            if not (0.0 <= e["dropped_fraction"] <= 1.0 + 1e-9) or not math.isfinite(e["mean_delay"]):
                errs.append(f"{hid}: bad metrics")
            if r["penalty"] > r["penalty_due"]:
                errs.append(f"{hid}: penalty above amount due")
            if not st.has("lab_regimes", r["regime"]):
                errs.append(f"{hid}: regime missing")
        if pen != st.get("lab", "totals")["penalties"]:
            errs.append("penalty total mismatch")
        pend = st.get("lab", "pending")
        if pend is not None and st.get("lab_changes", pend["change"])["status"] != "pending":
            errs.append("pending change not pending")
        if conf["sla_ms"] <= 0:
            errs.append("non-positive SLA")
        return errs

    def observe(self, world, actor):
        st = world.state
        if not st.has("lab", "conf"):
            return None
        dep = st.get("lab", "deployed")
        conf = st.get("lab", "conf")
        out = {"deployed": f"{dep['repo']}@{dep['ref']}:{dep['target']}", "sla_ms": conf["sla_ms"]}
        hours = st.ids("lab_hours")
        if hours:
            h = st.tables["lab_hours"][hours[-1]]["hour"] - conf["dashboard_delay_h"]
            r = st.get("lab_hours", f"h{h:06d}")
            if r is not None:
                out["dashboard"] = {"hour": h, **to_jsonable(r["measured"])}
        return out


# ============================================================================ metrics (exact, evaluator-side)
def _window_hours(world, window_h: int) -> list:
    st = world.state
    if "lab_hours" not in st.tables:
        return []
    lo = world.now // HOUR - int(window_h)
    return [st.tables["lab_hours"][hid] for hid in st.ids("lab_hours") if st.tables["lab_hours"][hid]["hour"] >= lo]


@metric("lab_mean_delay", "Volume-weighted exact mean effective delay (ms; dropped traffic at the timeout penalty) "
        "of production over the last window_h hours.")
def m_lab_mean_delay(world, window_h: int = 24) -> float:
    rows = _window_hours(world, window_h)
    vol = sum(r["exact"]["volume"] for r in rows)
    return float(sum(r["exact"]["cost"] for r in rows) / vol) if vol > 0 else 0.0


@metric("lab_dropped_fraction", "Fraction of production traffic dropped (overload or invalid routes) over the last "
        "window_h hours.")
def m_lab_dropped(world, window_h: int = 24) -> float:
    rows = _window_hours(world, window_h)
    vol = sum(r["exact"]["volume"] for r in rows)
    return float(sum(r["exact"]["dropped_fraction"] * r["exact"]["volume"] for r in rows) / vol) if vol > 0 else 0.0


def baseline_cost(world, rec) -> float:
    """Effective delay-volume of the baseline controller on a recorded production snapshot (re-sampled from the
    stored seed and traffic regime, so the comparison with the deployed controller is paired)."""
    cache = world.runtime.setdefault("lab_baseline_cost", {})
    key = (rec["regime"], rec["hour"], rec["seed"])
    if key not in cache:
        mod: Lab = world.modules["lab"]  # type: ignore[assignment]
        st = world.state
        conf = st.get("lab", "conf")
        net = mod.net(world)
        regime = to_jsonable(st.get("lab_regimes", rec["regime"])["params"])
        dem = sample_demands(regime, rec["hour"] % 24, rec["seed"])
        if digest(dem) != rec["demands"]:
            raise InvalidAction(f"demand snapshot of hour {rec['hour']} does not reproduce")
        res = evaluate(net, dem, baseline_route(net.topology, dem), conf["queue_scale"], conf["drop_penalty_ms"])
        cache[key] = summary(res)["cost"]          # same rounding as the recorded production hour
    return cache[key]


@metric("lab_improvement_vs_baseline", "(baseline delay - deployed delay) / baseline delay over the last window_h "
        "production hours, paired: the baseline is re-run on the same recorded demand snapshots.")
def m_lab_improvement(world, window_h: int = 24) -> float:
    rows = _window_hours(world, window_h)
    base = sum(baseline_cost(world, r) for r in rows)
    dep = sum(r["exact"]["cost"] for r in rows)
    return float((base - dep) / base) if base > 0 else 0.0


def lab_spec(family: str = "engineering_lab", seed: int = 0, overrides: dict | None = None):
    """WorldSpec of an engineering-lab family with the family's own module list (independent of whether
    `worldgen.generate` already honours a family's `module_list`)."""
    import copy

    from ..core.kernel import WorldSpec
    from ..worldgen import generator as G
    config = {"family": family, "overrides": dict(overrides or {})}
    spec = G.generate(config, seed)
    fam = G.FAMILIES[family]
    ov = dict(config["overrides"])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(digest(family)[:8], 16)]))
    r = fam.build(rng, ov)
    names = list(r.get("module_list") or G.BASE_MODULES)
    if [m["name"] for m in spec.modules] == names:
        return spec
    mods = []
    for m in names:
        cfg = copy.deepcopy(r["modules"].get(m, {}))
        if m in ov.get("module_config", {}):
            cfg.update(ov["module_config"][m])
        mods.append({"name": m, "config": cfg})
    params = dict(r["params"])
    params.update(ov.get("params", {}))
    params["family_split"] = fam.split
    return WorldSpec(family=family, seed=int(seed), modules=mods, params=params)


__all__ = ["Lab", "Network", "evaluate", "validate_routes", "link_delay", "link_drop", "sample_demands",
           "make_topology", "make_regime", "baseline_route", "compile_controller", "lab_spec", "BASELINE_SOURCE",
           "GREEDY_SOURCE", "INVALID_SOURCE", "HIDDEN_KEYS"]
