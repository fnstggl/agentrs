"""Structural fingerprints of world families (MASTER_SPEC Part 7.5) and a distance between them.

A fingerprint is measured from instantiated worlds - the engine's hidden structure (`truth`, `motifs`), its action
spaces and a short run through the public interface - never from a family's free-text labels. It is purely
structural: every component is a set of categorical descriptors or a categorical value, and no component holds a
count, a size, a coefficient, a horizon or a bucket of any of them. Nine components:

    engine            the transition implementation (engine kind, and the scenario code path inside it)
    motifs            mechanism motifs of the core mechanism (primitive vocabulary)
    motif_edges       which kind of mechanism feeds which, with the sign of the influence (core mechanism only)
    topology          the class of the core causal graph (chain / tree / dag / cyclic / network) and its structural
                      properties (feedback, common causes, colliders) - never its size
    temporal          whether influences are instantaneous or lagged, whether the mechanism is stateful, whether the
                      causal regime changes, whether a pre-arrival history exists
    observation       the observation channels (noise, delay, missingness, selection, aggregation, on demand), what
                      kinds of observation are present (readings, arrays, entity kinds, resource kinds, text)
    action_interface  the action kinds and the kinds of query target offered
    objective         the objective type
    participants      the policy families of the other participants and whether there are several learner seats

The core mechanism excludes distractor variables (nodes whose role is `distractor`, which the `n_variables` axis
adds) and their gauges. Hence a renamed family, a reseeded family, a parameter variant, a horizon variant, a size
variant (`graph_size`, `hidden_state_complexity`, `action_complexity`, `resources`) and a distractor variant all
have distance exactly 0 to their family (`counted` drops them), and the distance between two families is positive
only when one of the nine structural components differs. Claimed differences are checked with controlled
interventions on the public interface by `crucible.rapid.distinct` (evaluator side).
"""
from __future__ import annotations

from typing import Any

FP_VERSION = "rapid-fp/2"
WEIGHTS = {"engine": 1.0, "motifs": 2.0, "motif_edges": 1.0, "topology": 1.0, "temporal": 1.0, "observation": 1.0,
           "action_interface": 1.5, "objective": 1.0, "participants": 0.5}
PROBE_SEEDS = tuple(range(8))      # instances whose structures are united into the family's fingerprint
STATEFUL = {"hidden_confounding", "delay", "stock", "decay", "hysteresis", "feedback", "path_dependence",
            "adaptive_opponent", "regime_change", "endogenous_demand", "learning_behaviour"}
CHANNEL_PRIMS = {"measurement_noise", "missing_observation", "selection_effect", "partial_observability"}
EXCLUDED_ROLES = {"distractor"}


def _query_kind(target: str) -> str:
    """The kind of a query target, without its instance name: `station:s3` -> `query:station*`."""
    if target in ("tools", "history", "spec", "probe", "catalogue"):
        return f"query:{target}"
    if ":" in target:
        return "query:" + target.split(":", 1)[0].rstrip("0123456789") + "*"
    return "query:sensor"


def graph_class(nodes: list, edges: list[dict]) -> dict:
    """Structural class of a directed graph (node ids, edges {src, dst}): chain / tree / dag / cyclic, plus whether
    it has feedback, a common cause (a node with two or more children) and a collider (two or more parents)."""
    ids = list(dict.fromkeys(n if not isinstance(n, dict) else n["id"] for n in nodes))
    succ: dict = {i: set() for i in ids}
    indeg: dict = {i: 0 for i in ids}
    self_loop = False
    for e in edges:
        s, d = e["src"], e["dst"]
        if s == d:
            self_loop = True
            continue
        if s in succ and d in succ and d not in succ[s]:
            succ[s].add(d)
            indeg[d] += 1
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
    maxin = max(indeg.values(), default=0)
    maxout = max((len(s) for s in succ.values()), default=0)
    if not edges:
        cls = "none"
    elif cyc[0] or self_loop:
        cls = "cyclic"
    elif maxin <= 1 and maxout <= 1:
        cls = "chain"
    elif maxin <= 1:
        cls = "tree"
    else:
        cls = "dag"
    return {"class": cls, "feedback": bool(cyc[0] or self_loop), "common_cause": maxout >= 2,
            "collider": maxin >= 2}


def _pg_core(tr: dict) -> tuple[set, list[dict], list[dict]]:
    """Core node ids, core causal edges and core gauges of a primitive graph (distractors removed)."""
    excluded = {n["id"] for n in tr["nodes"] if n.get("role") in EXCLUDED_ROLES}
    ids = {n["id"] for n in tr["nodes"] if n["id"] not in excluded}
    edges = [e for e in tr["edges"] if e["kind"] in ("causal", "natural") and e["src"] in ids and e["dst"] in ids]
    gauges = [g for g in tr["gauges"] if any(s in ids for s in g["src"])]
    return ids, edges, gauges


def _kind_of(n: dict) -> str:
    return n["prim"].split("@")[0] if n.get("prim") else n["type"]


def instance_fingerprint(env) -> dict:
    """Fingerprint components of one instantiated world (evaluator side: reads the hidden structure)."""
    eng = env._engine
    cfg = env._config
    tr = eng.truth()
    motifs = set(eng.motifs())
    medges: set[str] = set()
    temporal: dict[str, Any] = {"history": bool(cfg.get("warmup")) or bool(eng.history(env._state))}
    if eng.kind == "primitive_graph":
        ids, causal, gauges = _pg_core(tr)
        kinds = {n["id"]: _kind_of(n) for n in tr["nodes"]}
        # motifs of the core: primitives on core nodes (hidden confounding as the engine measures it) and the
        # channel primitives of core gauges; a distractor's primitive never counts
        core_kinds = {kinds[i] for i in ids if kinds[i] not in ("lever", "discrete")}
        chans = {p.split("@")[0] for g in gauges for p in g["chain"]}
        motifs = {m for m in motifs if m in core_kinds or m in chans}
        for e in causal:
            sign = "+" if e.get("sign", 1) >= 0 else "-"
            medges.add(f"{kinds[e['src']]}>{kinds[e['dst']]}{sign}")
        topo = graph_class(sorted(ids), causal)
        if topo["feedback"]:
            motifs.add("feedback")
        lags = {("lagged" if e["lag"] > 0 else "instant") for e in causal}
        temporal.update(lags=sorted(lags), stateful=bool(core_kinds & STATEFUL),
                        regimes=bool(tr.get("regimes")) or "regime_change" in motifs)
        obs = {"channels": sorted(chans | ({"delayed"} if any(g["lag"] > 0 for g in gauges) else set())
                                  | ({"on_demand"} if any(g["on_demand"] for g in gauges) else set())
                                  | ({"aggregated"} if any(len(g["src"]) > 1 for g in gauges) else set())),
               **_observed_structure(env)}
    else:
        nodes = tr.get("nodes") or [e["src"] for e in tr.get("edges", [])] + [e["dst"] for e in tr.get("edges", [])]
        nodes = list(dict.fromkeys(n if not isinstance(n, dict) else n["id"] for n in nodes))
        edges = [e for e in tr.get("edges", []) if "src" in e and "dst" in e]
        if eng.kind == "graph_dynamical":
            topo = {"class": "network", "generator": str(tr.get("graph", {}).get("kind", "drawn"))}
        elif eng.kind == "state_machine":
            # the transition structure of a hidden-state machine: its class, not how many states it has (more
            # states add branching, which is the hidden_state_complexity axis)
            g = graph_class(nodes, edges) if nodes else {"class": "none", "feedback": False}
            topo = {"class": g["class"], "feedback": g["feedback"]}
        else:
            topo = graph_class(nodes, edges) if nodes else {"class": "none"}
        for e in edges:
            medges.add(str(e.get("kind", "edge")))
        lags = {("lagged" if int(e.get("lag", 0) or 0) > 0 else "instant") for e in edges} or {"instant"}
        temporal.update(lags=sorted(lags), stateful=bool(motifs & STATEFUL),
                        regimes=bool(tr.get("regimes")) or "regime_change" in motifs)
        obs = {"channels": sorted(eng.observation_channels()), **_observed_structure(env)}
    kinds_offered: set[str] = set()
    for a in eng.agents:
        sp = eng.action_space(a)
        for k, sch in sp.kinds.items():
            if k == "query":
                kinds_offered |= {_query_kind(t) for t in sch["properties"]["target"]["enum"]}
            else:
                kinds_offered.add(k)
    if getattr(eng, "dynamic_space", False):
        kinds_offered.add("dynamic_actions")
    if (cfg.get("experiment") or {}).get("instruments") or cfg.get("instruments"):
        kinds_offered.add("experiment:instrument_choice")
    parts = sorted(set(participant_policies(env)) | {"seats:multi" if len(eng.agents) > 1 else "seats:single"})
    engine = eng.kind if eng.kind == "primitive_graph" else f"{eng.kind}/{tr.get('scenario') or cfg.get('scenario')}"
    return {"engine": engine, "motifs": sorted(motifs), "motif_edges": sorted(medges), "topology": topo,
            "temporal": temporal, "observation": obs, "action_interface": sorted(kinds_offered),
            "objective": (cfg.get("objective") or {}).get("type", "maximize_net"), "participants": parts}


def _observed_structure(env) -> dict:
    """What kinds of observation a world's public observations contain (from a reset and six waiting ticks):
    whether there are readings and arrays, which entity kinds and which resource kinds. Kinds only, never how many
    (event text depends on what happened in the first ticks, so it is not part of the structure)."""
    import copy
    e = copy.deepcopy(env)
    obs = [e.reset()]
    for _ in range(6):
        if not e.done:
            obs.append(e.step(None)["observation"])
    return {"readings": any(o["numbers"] for o in obs), "arrays": any(o["arrays"] for o in obs),
            # entity kinds listed from the start (a catalogue the world shows); kinds that come and go with events
            # (a job waiting, a trip) depend on what happens, so only whether entities appear at all counts
            "entities": sorted({str(x.get("kind") or "entity") for x in obs[0]["entities"]}),
            "event_entities": any(o["entities"] for o in obs[1:]),
            "resources": sorted({_resource_kind(k) for o in obs for k in o["resources"]})}


def _resource_kind(key: str) -> str:
    """Resource keys name kinds (`cash_cents`, `experiment_budget`); a key that carries an instance name (a neutral
    public name such as `ra67`, drawn by the seed) is reduced to its kind (`unit_cost_ra67` -> `unit_cost_*`)."""
    import re
    return re.sub(r"(?<![a-z])[a-z]{2}\d{2}(?!\d)", "*", key.split(":", 1)[0])


def participant_policies(env) -> list[str]:
    """Policy families of the other participants: an engine that measures them (`participants()`) is asked; else
    the family's declaration."""
    fn = getattr(env._engine, "participants", None)
    if callable(fn):
        return sorted(str(x) for x in fn())
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
            out[k] = {}
            for kk in keys:
                xs = [v.get(kk) for v in vals]
                if all(isinstance(x, list) or x is None for x in xs):
                    out[k][kk] = sorted({str(e) for x in xs if x for e in x})     # kinds present in any instance
                else:
                    out[k][kk] = sorted({_h(x) for x in xs})
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
    key = (name, tuple(seeds), tuple(sorted(difficulty.items())), FAMILIES[name].version, id(FAMILIES[name].build))
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
        return {"nearest": None, "distance": None, "differs_in": [], "difference": {}}
    diff = differing(fps[name], fps[best[0]])
    return {"nearest": best[0], "distance": best[1], "differs_in": diff,
            "difference": {c: {"this": fps[name].get(c), "nearest": fps[best[0]].get(c)} for c in diff}}


# ====================================================================== 4.0 families in the v5 vocabulary
# 4.0 motif names mapped onto the primitive vocabulary, so the mechanisms of the 4.0 validation and sealed test
# families can be compared with v5 training families (D-9, mechanism-level non-contamination). Motifs with no
# primitive counterpart (graph-shape words, costs, distractors, deadlines) map to nothing.
LEGACY_VOCAB = {
    "confounder": "hidden_confounding", "feedback": "feedback", "regime": "regime_change",
    "regime_change": "regime_change", "local_network_effects": "network_effect",
    "network_word_of_mouth": "network_effect", "lock_in": "path_dependence", "path_dependence": "path_dependence",
    "shared_pool": "resource_competition", "shared_attention_pool": "resource_competition",
    "commons": "resource_competition", "gate": "threshold", "rating_threshold_gate": "threshold",
    "thrash_threshold": "threshold", "delay": "delay", "lagged_gauge": "delay", "startup_lag": "delay",
    "stock": "stock", "saturation": "saturation", "batch_knee": "saturation", "invu": "non_monotonic",
    "interaction": "multiplicative", "bundle_of_complements": "complementarity",
    "weakest_link_quality": "complementarity", "queue": "congestion", "bottleneck": "congestion",
    "back_pressure": "congestion", "capacity_rationing": "congestion", "metastable_collapse": "cascading_failure",
    "retry_storm": "cascading_failure", "adaptive_incumbents": "adaptive_opponent",
    "two_seat_rivalry": "strategic_interaction", "switching_cost": "hysteresis",
    "private_gauges": "partial_observability", "experience_memory": "learning_behaviour",
    "satisfaction_referral_loop": "feedback", "wait_churn_loop": "feedback", "cold_start": "threshold",
    "two_segments": "hidden_confounding", "retry": "feedback", "redundant_paths": "substitution",
    "unequal_reliability": "state_dependent_effect", "cache": "state_dependent_effect",
}


def legacy_fingerprint(family: str, seeds=(0, 1)) -> dict:
    """Fingerprint of a Crucible 4.0 mechanism family (scm, abm, techsys) in the v5 vocabulary, from its `truth()`
    and public spec."""
    from ..mechanisms.base import build_engine, make_mechanism
    from ..mechanisms.sim import ensure_engines
    ensure_engines()
    fps, raw_all = [], set()
    for s in seeds:
        eng = build_engine(make_mechanism(family, s))
        tr = eng.truth()
        raw = set(tr.get("motifs", []))
        raw_all |= raw
        motifs = sorted({LEGACY_VOCAB[m] for m in raw if m in LEGACY_VOCAB})
        nodes = [n["id"] for n in tr["nodes"]]
        topo = graph_class(nodes, tr["edges"])
        lags = sorted({("lagged" if int(e.get("lag", 0) or 0) > 0 else "instant") for e in tr["edges"]})
        fps.append({"engine": f"4.0:{eng.engine}", "motifs": motifs,
                    "motif_edges": sorted({f"{e.get('kind', 'edge')}" for e in tr["edges"]}), "topology": topo,
                    "temporal": {"lags": lags, "regimes": bool(tr.get("regimes")) or "regime_change" in motifs,
                                 "stateful": bool(set(motifs) & STATEFUL), "history": False},
                    "observation": {"readings": True}, "action_interface": ["levers"], "objective": "maximize_net",
                    "participants": ["seats:multi" if eng.n_seats > 1 else "seats:single"]})
    out = merge(fps)
    out["legacy_motifs"] = sorted(raw_all)
    return out


_LEGACY_TEST: dict | None = None


def legacy_sealed() -> dict:
    """The 4.0 validation and sealed test families in the v5 vocabulary: name -> {split, motifs, novel}, where
    `novel` are the mapped motifs absent from every 4.0 training family (what the 4.0 split holds out)."""
    global _LEGACY_TEST
    if _LEGACY_TEST is None:
        from ..mechanisms.sim import ensure_engines
        ensure_engines()
        from ..mechanisms.base import mech_families
        train = set()
        for f in mech_families(split="train"):
            train |= set(legacy_fingerprint(f)["motifs"])
        out = {}
        for split in ("val", "test"):
            for f in mech_families(split=split):
                fp = legacy_fingerprint(f)
                out[f] = {"split": split, "motifs": fp["motifs"], "novel": sorted(set(fp["motifs"]) - train),
                          "fingerprint": fp}
        _LEGACY_TEST = out
    return _LEGACY_TEST


LEVER_LIKE = {"levers", "sequence", "discrete"}


def legacy_recreation(fp: dict) -> dict:
    """D-9 at the mechanism level, for a v5 training family's fingerprint. `recreates`: 4.0 validation and sealed
    test families whose mechanism the v5 family reproduces - the same mechanism motif set in the shared vocabulary
    (observation channels aside) offered through a lever-like interface (levers, discrete settings, sequences and
    queries only) - so a learner trained on it would meet that held-out mechanism again (D-9 covers both 4.0
    held-out splits). `exposes_novel`: 4.0 validation or test
    families whose held-out motifs the v5 family contains at all (recorded: the 4.0 split no longer holds those
    motifs out for a learner trained on v5-train)."""
    mech = set(fp["motifs"]) - CHANNEL_PRIMS
    iface = {k.split(":")[0] for k in fp["action_interface"]} - {"query"}
    recreates, exposes = [], []
    for name, rec in legacy_sealed().items():
        lm = set(rec["motifs"]) - CHANNEL_PRIMS
        if rec["novel"] and set(rec["novel"]) <= mech:
            exposes.append(name)
        if lm and mech == lm and iface <= LEVER_LIKE:
            recreates.append(name)
    return {"recreates": sorted(recreates), "exposes_novel": sorted(exposes)}
