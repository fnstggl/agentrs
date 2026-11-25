"""Structural fingerprints of world families (MASTER_SPEC Part 7.5) and a distance between them.

A fingerprint is measured from instantiated worlds - the engine's hidden structure (`truth`, `motifs`) and a short
run through the public interface - never from a family's own labels. It has eight components:

    motifs            mechanism motifs present (primitive vocabulary), and for causal graphs which primitive feeds
                      which (`motif_edges`)
    topology          graph class and size buckets (nodes, edges, in-degree, depth, cycles, clustering)
    temporal          lag buckets, number of stateful mechanisms, regime changes, horizon bucket
    observation       observation channels, number of readings, history, on-demand sensors, arrays, entities
    action_interface  the action kinds and the kinds of query target offered
    objective         the objective type
    participants      other participant policy families and the number of participants
    engine            the transition implementation

Names, seeds and numerical coefficients never enter a fingerprint, so a renamed family or a parameter variant has
distance 0 to its original and does not count as a new family (`counted`).
"""
from __future__ import annotations

import math
from typing import Any

FP_VERSION = "rapid-fp/1"
WEIGHTS = {"engine": 1.0, "motifs": 2.0, "motif_edges": 1.0, "topology": 1.0, "temporal": 1.0, "observation": 1.0,
           "action_interface": 1.5, "objective": 1.0, "participants": 0.5}
PROBE_SEEDS = (0, 1, 2)


def _bucket(n: float) -> str:
    n = int(n)
    if n <= 0:
        return "0"
    return f"2^{int(math.log2(n))}"


def _query_kind(target: str) -> str:
    if target in ("tools", "history", "spec"):
        return f"query:{target}"
    if ":" in target:
        return "query:" + target.split(":", 1)[0].rstrip("0123456789") + "*"
    if target in ("probe", "catalogue"):
        return f"query:{target}"
    return "query:sensor"


def _graph_shape(nodes: list, edges: list[dict]) -> dict:
    """Topology descriptors of a directed graph given as node ids and edges {src, dst, lag?}."""
    ids = [n if not isinstance(n, dict) else n["id"] for n in nodes]
    succ: dict = {i: set() for i in ids}
    indeg: dict = {i: 0 for i in ids}
    for e in edges:
        if e["src"] in succ and e["dst"] in succ and e["src"] != e["dst"]:
            if e["dst"] not in succ[e["src"]]:
                succ[e["src"]].add(e["dst"])
                indeg[e["dst"]] += 1
    self_loops = any(e["src"] == e["dst"] for e in edges)
    # cycle detection and longest path (on the condensation-free graph when acyclic)
    color: dict = {}
    cyc = [False]

    def dfs(v):
        color[v] = 1
        for w in succ[v]:
            if color.get(w) == 1:
                cyc[0] = True
            elif w not in color:
                dfs(w)
        color[v] = 2
    for v in ids:
        if v not in color:
            dfs(v)
    depth = 0
    if not cyc[0]:
        memo: dict = {}

        def longest(v):
            if v not in memo:
                memo[v] = 1 + max((longest(w) for w in succ[v]), default=0)
            return memo[v]
        depth = max((longest(v) for v in ids), default=0)
    m = sum(len(s) for s in succ.values())
    maxin = max(indeg.values(), default=0)
    outdeg = {v: len(s) for v, s in succ.items()}
    if cyc[0] or self_loops:
        cls = "cyclic"
    elif maxin <= 1 and max(outdeg.values(), default=0) <= 1:
        cls = "chain"
    elif maxin <= 1:
        cls = "tree"
    else:
        cls = "dag"
    return {"class": cls, "nodes": _bucket(len(ids)), "edges": _bucket(m), "max_in": _bucket(maxin),
            "depth": _bucket(depth), "feedback": bool(cyc[0] or self_loops)}


def instance_fingerprint(env) -> dict:
    """Fingerprint components of one instantiated world (evaluator side: reads the hidden structure)."""
    eng = env._engine
    cfg = env._config
    tr = eng.truth()
    motifs = set(eng.motifs())
    medges: set[str] = set()
    temporal: dict[str, Any] = {"horizon": "<=40" if eng.horizon <= 40 else ("<=80" if eng.horizon <= 80 else ">80")}
    if eng.kind == "primitive_graph":
        kinds = {n["id"]: (n["prim"].split("@")[0] if n.get("prim") else n["type"]) for n in tr["nodes"]}
        causal = [e for e in tr["edges"] if e["kind"] in ("causal", "natural")]
        for e in causal:
            medges.add(f"{kinds[e['src']]}>{kinds[e['dst']]}")
        topo = _graph_shape([n["id"] for n in tr["nodes"]], causal)
        lags = sorted({_lag_bucket(e["lag"]) for e in causal})
        stateful = sum(1 for n in tr["nodes"] if n.get("prim") and n["prim"].split("@")[0] in STATEFUL)
        temporal.update(lags=lags, stateful=_bucket(stateful), regimes="regime_change" in motifs)
        if topo["feedback"]:
            motifs.add("feedback")
        obs = {"channels": sorted({p.split("@")[0] for g in tr["gauges"] for p in g["chain"]}),
               "readings": _bucket(sum(1 for g in tr["gauges"] if not g["on_demand"])),
               "on_demand": any(g["on_demand"] for g in tr["gauges"]),
               "delayed": any(g["lag"] > 0 for g in tr["gauges"]),
               "aggregated": any(len(g["src"]) > 1 for g in tr["gauges"]), "history": bool(cfg.get("warmup"))}
    else:
        nodes = tr.get("nodes") or [e["src"] for e in tr.get("edges", [])] + [e["dst"] for e in tr.get("edges", [])]
        nodes = list(dict.fromkeys(n if not isinstance(n, dict) else n["id"] for n in nodes))
        edges = [e for e in tr.get("edges", []) if "src" in e and "dst" in e]
        if eng.kind == "graph_dynamical":
            # a drawn random network: its class and size, not the particular edges a seed produced
            topo = {"class": "network", "nodes": _bucket(tr["graph"]["nodes"])}
        else:
            topo = _graph_shape(nodes, edges) if nodes else {"class": "none"}
        topo["scenario_structure"] = tr.get("scenario")
        if eng.kind == "state_machine":
            topo["states"] = _bucket(tr["states"])
        for e in edges:
            medges.add(f"{tr.get('scenario')}:{e.get('kind', 'edge')}")
        lags = sorted({_lag_bucket(e.get("lag", 0)) for e in edges}) or ["0"]
        temporal.update(lags=lags, stateful=_bucket(len(motifs & STATEFUL)),
                        regimes=bool(tr.get("regimes")) or "regime_change" in motifs)
        obs = _observed_structure(env)
    kinds: set[str] = set()
    for a in eng.agents:
        sp = eng.action_space(a)
        for k, sch in sp.kinds.items():
            if k == "query":
                kinds |= {_query_kind(t) for t in sch["properties"]["target"]["enum"]}
            else:
                kinds.add(k)
    if getattr(eng, "dynamic_space", False):
        kinds.add("dynamic_actions")
    if (cfg.get("experiment") or {}).get("instruments"):
        kinds.add("experiment:instrument_choice")
    parts = sorted(set(env_family_participants(env)) | {f"agents:{len(eng.agents)}"})
    return {"engine": eng.kind, "motifs": sorted(motifs), "motif_edges": sorted(medges), "topology": topo,
            "temporal": temporal, "observation": obs, "action_interface": sorted(kinds),
            "objective": (cfg.get("objective") or {}).get("type", "maximize_net"), "participants": parts}


STATEFUL = {"hidden_confounding", "delay", "stock", "decay", "hysteresis", "feedback", "path_dependence",
            "adaptive_opponent", "regime_change", "endogenous_demand", "learning_behaviour"}


def _lag_bucket(lag: int) -> str:
    """Instantaneous, one tick, or several ticks: how long a delay is is a coefficient, whether there is one is
    structure."""
    lag = int(lag)
    return "0" if lag == 0 else ("1" if lag == 1 else "2+")


def _observed_structure(env) -> dict:
    """What the public observations of a world contain, from a reset and two waiting ticks."""
    import copy
    e = copy.deepcopy(env)
    obs = [e.reset()]
    for _ in range(2):
        if not e.done:
            obs.append(e.step(None)["observation"])
    nums = max(len(o["numbers"]) for o in obs)
    return {"readings": _bucket(nums), "arrays": any(o["arrays"] for o in obs),
            "entities": sorted({x.get("kind", "?") for o in obs for x in o["entities"]}),
            "text": any(len(o["text"]) > 1 for o in obs[:1]) or any(o["text"] for o in obs[1:]),
            "resources": sorted({k for o in obs for k in o["resources"]})}


def env_family_participants(env) -> list[str]:
    from .families import FAMILIES
    f = FAMILIES.get(env.family)
    return list(f.participants) if f else []


def merge(fps: list[dict]) -> dict:
    """A family's fingerprint from its instances: sets are united, scalar descriptors collected as sorted sets."""
    out: dict[str, Any] = {"version": FP_VERSION}
    for k in WEIGHTS:
        vals = [fp[k] for fp in fps]
        if isinstance(vals[0], list):
            out[k] = sorted({x for v in vals for x in v})
        elif isinstance(vals[0], dict):
            keys = sorted({kk for v in vals for kk in v})
            out[k] = {kk: sorted({_h(v.get(kk)) for v in vals}) for kk in keys}
        else:
            out[k] = sorted({_h(v) for v in vals}) if len({_h(v) for v in vals}) > 1 else vals[0]
    return out


def _h(x):
    if isinstance(x, list):
        return ",".join(map(str, x))
    return str(x)


_CACHE: dict[tuple, dict] = {}


def family_fingerprint(name: str, seeds=PROBE_SEEDS, **difficulty) -> dict:
    from .families import FAMILIES, make
    key = (name, tuple(seeds), tuple(sorted(difficulty.items())), FAMILIES[name].version)
    if key not in _CACHE:
        _CACHE[key] = merge([instance_fingerprint(make(name, s, **difficulty)) for s in seeds])
    return _CACHE[key]


def _jac(a, b) -> float:
    a, b = set(map(str, a)), set(map(str, b))
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def components(fa: dict, fb: dict) -> dict[str, float]:
    """Per-component distance in [0, 1]."""
    out = {}
    for k in WEIGHTS:
        a, b = fa.get(k), fb.get(k)
        if isinstance(a, dict) or isinstance(b, dict):
            a, b = a or {}, b or {}
            keys = set(a) | set(b)
            out[k] = sum(1.0 for kk in keys if a.get(kk) != b.get(kk)) / len(keys) if keys else 0.0
        elif isinstance(a, list) or isinstance(b, list):
            out[k] = _jac(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b])
        else:
            out[k] = 0.0 if a == b else 1.0
    return out


def distance(fa: dict, fb: dict) -> float:
    """Weighted mean of the component distances, in [0, 1]; 0 means structurally identical."""
    c = components(fa, fb)
    return sum(WEIGHTS[k] * v for k, v in c.items()) / sum(WEIGHTS.values())


def differing(fa: dict, fb: dict) -> list[str]:
    return [k for k, v in components(fa, fb).items() if v > 0]


def counted(fps: dict[str, dict], order: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """Families that count as distinct: walking in `order`, a family whose fingerprint has distance 0 to one already
    counted is a cosmetic variant of it. Returns (counted names, variant -> original)."""
    names = order or sorted(fps)
    kept: list[str] = []
    dup: dict[str, str] = {}
    for n in names:
        same = next((k for k in kept if distance(fps[n], fps[k]) == 0.0), None)
        if same is None:
            kept.append(n)
        else:
            dup[n] = same
    return kept, dup


def nearest(name: str, fps: dict[str, dict]) -> dict:
    """The nearest other family, the distance and which components differ (the measured reason it is distinct)."""
    best = None
    for k, fp in fps.items():
        if k == name:
            continue
        d = distance(fps[name], fp)
        if best is None or d < best[1]:
            best = (k, d)
    if best is None:
        return {"nearest": None, "distance": None, "differs_in": []}
    return {"nearest": best[0], "distance": best[1], "differs_in": differing(fps[name], fps[best[0]])}


def legacy_fingerprint(family: str, seeds=(0, 1)) -> dict:
    """Fingerprint of a Crucible 4.0 mechanism family (scm, abm, techsys) from its `truth()` and public spec, so
    distinctness from the existing training and evaluation worlds can be measured."""
    from ..mechanisms.base import build_engine, make_mechanism
    from ..mechanisms.sim import ensure_engines
    ensure_engines()
    fps = []
    for s in seeds:
        eng = build_engine(make_mechanism(family, s))
        tr = eng.truth()
        nodes = [n["id"] for n in tr["nodes"]]
        topo = _graph_shape(nodes, tr["edges"])
        spec = eng.public_spec(0)
        lags = sorted({_lag_bucket(e.get("lag", 0)) for e in tr["edges"]})
        fps.append({"engine": eng.engine, "motifs": sorted(set(tr.get("motifs", []))),
                    "motif_edges": sorted({f"{e.get('kind', 'edge')}" for e in tr["edges"]}), "topology": topo,
                    "temporal": {"horizon": "<=40" if eng.horizon <= 40 else ("<=80" if eng.horizon <= 80 else ">80"),
                                 "lags": lags, "regimes": bool(tr.get("regimes"))},
                    "observation": {"readings": _bucket(len(spec["gauges"]))},
                    "action_interface": ["levers"], "objective": "maximize_net",
                    "participants": [f"agents:{eng.n_seats}"]})
    return merge(fps)
