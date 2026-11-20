"""Emergence metrics over a world's event log and state (read-only, evaluator side).

* `tool_reuse`          - for every agent-deployed host: distinct principals calling it directly (recorded `cmd`
                          events) or nested (service principals calling it from their own code: priced calls are
                          attributed through the ledger, unpriced ones through the recorded RPC list of the
                          calling request), and how many of them are independent of the operator.
* `diffusion_curves`    - customers per host over time (consumers.hour slot summaries) and adoption statistics.
* `market_concentration`- Herfindahl-Hirschman index of customer shares per need over time.
* `specialization`      - roles and service categories per organisation, and world-level concentration of them.
* `strategy_diversity`  - per-agent tool-category usage histograms -> clusters (Jensen-Shannon leader
                          clustering), cluster entropy, mean pairwise divergence.
* `dependency_graph`    - artifacts and who depends on whom: repo -> repo (manifest deps), host -> repo
                          (deploys), host -> host (service calls service), operator -> host.
* `useful_novelty`      - an artifact counts as useful novelty only if a counterfactual removal (fork, undeploy,
                          advance) *reduces* some outcome other than its owner's: households' served requests,
                          served consumers, or the requests its dependents serve.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..analytics import DAY, has_table, slot_map, world_events

BUILTIN_CATEGORIES = {"bank": "finance", "biz": "finance", "contracts": "contracts", "market": "market",
                      "ads": "marketing", "search": "discovery", "world": "discovery", "forum": "social",
                      "mail": "communication", "docs": "communication", "cal": "communication",
                      "issues": "engineering", "code": "engineering", "cloud": "infrastructure",
                      "support": "operations", "jobs": "labor", "id": "organisation", "research": "research",
                      "goals": "planning"}


def _deployed_hosts(world) -> dict[str, dict]:
    st = world.state
    if not has_table(world, "deployments"):
        return {}
    out = {}
    for did, d in st.scan("deployments"):
        out.setdefault(d["host"], {"owner": d["owner"], "trusted": d["trusted"], "created": d["created"],
                                   "deployments": []})["deployments"].append(did)
    return out


def agent_hosts(world) -> list[str]:
    """Hosts deployed by actors (not by the generator's simulated incumbent firms)."""
    st = world.state
    npc = has_table(world, "npc_firms")
    return sorted(h for h, v in _deployed_hosts(world).items()
                  if not (st.has("npc_firms", v["owner"]) if npc else v["trusted"]))


# ====================================================================== calls
def call_edges(world, since: int = 0, detail: bool = False):
    """Counter {(caller principal, host): calls} over deployed hosts. Direct calls come from `cmd` events, nested
    calls from the ledger (priced calls carry the calling service principal in `by`) and from the recorded RPCs
    of the outermost request of a command. With `detail`, returns (direct Counter, nested Counter)."""
    hosts = _deployed_hosts(world)
    direct: Counter = Counter()
    nested: Counter = Counter()
    cmd_tool: dict[int, str] = {}
    for ev in world_events(world, since, types=("cmd", "bank.txn", "ext")):
        p = ev.payload
        if ev.type == "cmd":
            tool = str(p.get("tool") or "")
            h = tool.split(".", 1)[0]
            if p.get("n") is not None:
                cmd_tool[p["n"]] = tool
            if h in hosts and not p.get("system"):
                direct[(ev.actor, h)] += 1
        elif ev.type == "bank.txn":
            ref = p.get("ref") or ""
            by = p.get("by") or ""
            if p.get("kind") == "sale" and ref.startswith("svc:") and by.startswith("svc:"):
                h = ref.split(":", 1)[1]
                if h in hosts:
                    nested[(by, h)] += 1
        elif ev.type == "ext" and ev.cmd is not None:
            outer = cmd_tool.get(ev.cmd, "").split(".", 1)[0]
            outs = p.get("outcomes") or []
            if outer in hosts and outs and isinstance(outs[0], dict):
                rp = (outs[0].get("out") or {}).get("rpcs") or []
                for r in rp:
                    h = str(r.get("tool", "")).split(".", 1)[0]
                    if h in hosts and h != outer and \
                            _endpoint_price(world, h, str(r.get("tool", "")).split(".", 1)[-1]) == 0:
                        nested[(f"svc:{outer}", h)] += 1   # priced calls are already counted through the ledger
    if detail:
        return direct, nested
    return direct + nested


def _endpoint_price(world, host: str, ep: str) -> int:
    st = world.state
    h = st.get("hosts", host)
    if h is None or not h.get("deployment"):
        return 0
    d = st.get("deployments", h["deployment"])
    spec = (d["manifest"].get("endpoints") or {}).get(ep) or {}
    return int(spec.get("price_cents", 0))


def tool_reuse(world, since: int = 0) -> dict:
    from ..goals.metrics import related_parties
    direct, nested = call_edges(world, since, detail=True)
    hosts = _deployed_hosts(world)
    rows = {}
    for h in agent_hosts(world):
        owner = hosts[h]["owner"]
        rel = related_parties(world, owner) | {f"svc:{x}" for x in hosts if hosts[x]["owner"] == owner}
        dc = {c for (c, x) in direct if x == h}
        nc = {c for (c, x) in nested if x == h}
        callers = dc | nc
        rows[h] = {"owner": owner, "direct_callers": sorted(dc), "nested_callers": sorted(nc),
                   "distinct_callers": len(callers), "independent_callers": len(callers - rel),
                   "calls_direct": sum(v for (c, x), v in direct.items() if x == h),
                   "calls_nested": sum(v for (c, x), v in nested.items() if x == h)}
    reused = [h for h, r in rows.items() if r["independent_callers"] > 0]
    return {"hosts": rows, "n_agent_hosts": len(rows), "n_reused": len(reused), "reused": reused,
            "mean_distinct_callers": float(np.mean([r["distinct_callers"] for r in rows.values()])) if rows else 0.0}


# ====================================================================== diffusion & concentration
def _market_buckets(world, since: int, every_s: int):
    """Yield (bucket_time, {slot: customers}, public) at the last consumers.hour event of each bucket."""
    last = None
    for ev in world_events(world, since, types=("consumers.hour",)):
        p = ev.payload
        t = int(p.get("t", ev.t))
        b = (t - 1) // every_s
        cust = {int(k): int(v.get("customers", 0)) for k, v in (p.get("slots") or {}).items()}
        pub = {int(k): v for k, v in (p.get("public") or {}).items()}
        if last is not None and b != last[0]:
            yield last[1], last[2], last[3]
        last = (b, t, cust, pub)
    if last is not None:
        yield last[1], last[2], last[3]


def diffusion_curves(world, since: int = 0, every_s: int = DAY) -> dict:
    smap = slot_map(world)
    curves: dict[str, list] = defaultdict(list)
    for t, cust, pub in _market_buckets(world, since, every_s):
        for s in sorted(set(cust) | set(pub)):
            h = smap.get(s, {}).get("host")
            if h:
                curves[h].append((t, cust.get(s, 0)))
    stats = {}
    for h, c in curves.items():
        vals = [v for _, v in c]
        peak = max(vals) if vals else 0
        t_first = c[0][0] if c else None

        def reach(frac):
            for t, v in c:
                if peak and v >= frac * peak:
                    return t
            return None
        stats[h] = {"peak": peak, "t_peak": c[vals.index(peak)][0] if vals else None, "current": vals[-1] if vals else 0,
                    "t_first": t_first, "t10": reach(0.1), "t50": reach(0.5),
                    "need": next((v["need"] for v in smap.values() if v["host"] == h), None)}
    return {"curves": {h: [[t, v] for t, v in c] for h, c in sorted(curves.items())}, "stats": stats}


def hhi(shares) -> float:
    tot = float(sum(shares))
    return float(sum((x / tot) ** 2 for x in shares)) if tot else 0.0


def market_concentration(world, since: int = 0, every_s: int = DAY) -> dict:
    """HHI (0..1; x10,000 for the antitrust convention) of customer shares per need over time."""
    smap = slot_map(world)
    series: dict[str, list] = defaultdict(list)
    for t, cust, pub in _market_buckets(world, since, every_s):
        by_need: dict[str, dict[str, int]] = defaultdict(dict)
        for s in sorted(set(cust) | set(pub)):
            v = smap.get(s)
            if v is None:
                continue
            by_need[v["need"]][v["host"]] = by_need[v["need"]].get(v["host"], 0) + cust.get(s, 0)
        for need, hs in sorted(by_need.items()):
            tot = sum(hs.values())
            top = max(hs, key=lambda k: (hs[k], k)) if hs else None
            series[need].append({"t": t, "hhi": round(hhi(list(hs.values())), 4), "customers": tot,
                                 "n_active": sum(1 for v in hs.values() if v > 0), "top": top,
                                 "top_share": round(hs[top] / tot, 4) if tot else 0.0})
    current = {}
    cons = world.modules.get("consumers")
    if cons is not None and has_table(world, "cons_offers"):
        for need in sorted({o["need"] for _, o in world.state.scan("cons_offers")}):
            current[need] = round(hhi(list(cons.market_shares(world, need).values())), 4)
    return {"series": dict(series), "current": current}


# ====================================================================== specialization
def _entropy(counts) -> float:
    c = np.asarray([x for x in counts if x > 0], float)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def specialization(world) -> dict:
    st = world.state
    rows = {}
    if not has_table(world, "orgs"):
        return {"orgs": {}, "summary": {}}
    for org, o in st.scan("orgs"):
        roles = Counter(m["role"] for m in st.find("members", "org", org)) if has_table(world, "members") else Counter()
        titles = Counter((m.get("title") or "") for m in st.find("members", "org", org)) \
            if has_table(world, "members") else Counter()
        cats: Counter = Counter()
        if has_table(world, "listings"):
            for lid in st.find_ids("listings", "seller", org):
                l = st.tables["listings"][lid]
                cats[f"{l['kind']}:{l['need']}" if l["need"] else l["kind"]] += 1
        sector = (st.get("companies", org) or {}).get("sector") if has_table(world, "companies") else None
        if has_table(world, "providers") and st.has("providers", org):
            cats["infrastructure:compute"] += 1
        if has_table(world, "npc_lenders") and st.has("npc_lenders", org):
            cats["finance:lending"] += 1
        rows[org] = {"status": o["status"], "members": sum(roles.values()), "roles": dict(roles),
                     "titles": {k: v for k, v in titles.items() if k}, "categories": dict(cats), "sector": sector,
                     "n_categories": len(cats), "role_entropy": round(_entropy(roles.values()), 4)}
    active = [r for r in rows.values() if r["n_categories"]]
    allcats = Counter()
    for r in active:
        allcats.update({c: 1 for c in r["categories"]})
    summary = {"orgs": len(rows), "orgs_with_offerings": len(active),
               "specialists": sum(1 for r in active if r["n_categories"] == 1),
               "mean_categories": float(np.mean([r["n_categories"] for r in active])) if active else 0.0,
               "distinct_categories": len(allcats),
               "category_entropy_bits": round(_entropy(allcats.values()), 4),
               "mean_role_entropy_bits": float(np.mean([r["role_entropy"] for r in rows.values()])) if rows else 0.0}
    return {"orgs": rows, "summary": summary}


# ====================================================================== strategy diversity
def tool_category(world, tool: str, hosts=None) -> str:
    h = str(tool).split(".", 1)[0]
    if h in BUILTIN_CATEGORIES:
        return BUILTIN_CATEGORIES[h]
    hosts = _deployed_hosts(world) if hosts is None else hosts
    if h in hosts:
        return "deployed_api"
    return "other"


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def strategy_diversity(world, actors=None, since: int = 0, threshold: float = 0.25, min_actions: int = 3) -> dict:
    """Per-agent tool-category histograms (from recorded commands, accepted and rejected) clustered with a
    leader algorithm on Jensen-Shannon divergence (bits, in [0, 1])."""
    hosts = _deployed_hosts(world)
    st = world.state
    if actors is None and has_table(world, "persons"):
        actors = [pid for pid, p in st.scan("persons") if p["kind"] in ("agent", "human")]
    want = set(actors) if actors is not None else None
    counts: dict[str, Counter] = defaultdict(Counter)
    for ev in world_events(world, since, types=("cmd", "cmd.rejected")):
        p = ev.payload
        if ev.type == "cmd" and p.get("n") is None:
            continue
        a = p.get("actor") if ev.type == "cmd.rejected" else ev.actor
        if not a or (want is not None and a not in want):
            continue
        counts[a][tool_category(world, p.get("tool", ""), hosts)] += 1
    actors_ok = sorted(a for a, c in counts.items() if sum(c.values()) >= min_actions)
    cats = sorted({k for a in actors_ok for k in counts[a]})
    if not actors_ok:
        return {"agents": {}, "n_clusters": 0, "cluster_entropy_bits": 0.0, "mean_pairwise_js": 0.0,
                "categories": cats}
    X = np.array([[counts[a][c] for c in cats] for a in actors_ok], float)
    P = X / X.sum(axis=1, keepdims=True)
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    assign = []
    for i in range(len(actors_ok)):
        best, bd = None, None
        for j, c in enumerate(centroids):
            d = _js(P[i], c)
            if d <= threshold and (bd is None or d < bd):
                best, bd = j, d
        if best is None:
            centroids.append(P[i].copy())
            members.append([i])
            assign.append(len(centroids) - 1)
        else:
            members[best].append(i)
            centroids[best] = P[members[best]].mean(axis=0)
            assign.append(best)
    pair = [_js(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    sizes = [len(m) for m in members]
    return {"agents": {a: {"histogram": dict(counts[a]), "cluster": assign[i]} for i, a in enumerate(actors_ok)},
            "categories": cats, "n_clusters": len(members), "cluster_sizes": sizes,
            "cluster_entropy_bits": round(_entropy(sizes), 4),
            "mean_pairwise_js": round(float(np.mean(pair)), 4) if pair else 0.0, "threshold": threshold}


# ====================================================================== dependency graph
def dependency_graph(world, since: int = 0) -> dict:
    st = world.state
    nodes: dict[str, dict] = {}
    edges: Counter = Counter()

    def node(nid, kind, **kw):
        nodes.setdefault(nid, {"id": nid, "kind": kind, **kw})
    code = world.modules.get("code")
    if has_table(world, "repos") and code is not None:
        for rid, r in st.scan("repos"):
            node(f"repo:{rid}", "repo", owner=r["owner"])
            try:
                files = code.files(st, rid)
            except Exception:  # noqa: BLE001
                files = {}
            import json
            try:
                man = json.loads(files.get("crucible.json") or "{}")
            except ValueError:
                man = {}
            for dep in (man.get("deps") or [])[:20] if isinstance(man, dict) else []:
                d = str(dep).split("@", 1)[0]
                node(f"repo:{d}", "repo")
                edges[(f"repo:{rid}", f"repo:{d}", "depends_on")] += 1
    for h, v in _deployed_hosts(world).items():
        node(f"host:{h}", "host", owner=v["owner"], trusted=v["trusted"])
        node(v["owner"], "org" if v["owner"].startswith("o:") else "principal")
        edges[(v["owner"], f"host:{h}", "operates")] += 1
        for did in v["deployments"]:
            d = st.tables["deployments"][did]
            node(f"repo:{d['repo']}", "repo")
            edges[(f"host:{h}", f"repo:{d['repo']}", "deploys")] += 1
    for (caller, h), n in call_edges(world, since).items():
        if caller.startswith("svc:"):
            edges[(f"host:{caller[4:]}", f"host:{h}", "calls")] += n
        else:
            node(caller, "principal")
            edges[(caller, f"host:{h}", "uses")] += n
    calls = defaultdict(set)
    for (a, b, k) in edges:
        if k == "calls":
            calls[a].add(b)

    def depth(n, seen=()):
        if n in seen:
            return 0
        return 1 + max((depth(m, seen + (n,)) for m in calls.get(n, ())), default=0)
    max_chain = max((depth(n) for n in calls), default=0)
    return {"nodes": sorted(nodes.values(), key=lambda x: x["id"]),
            "edges": [{"src": a, "dst": b, "kind": k, "n": n} for (a, b, k), n in sorted(edges.items())],
            "n_nodes": len(nodes), "n_edges": len(edges), "max_service_chain": max_chain,
            "service_call_edges": sum(1 for (a, b, k) in edges if k == "calls")}


# ====================================================================== useful novelty
def useful_novelty(world, horizon_s: int = 2 * DAY, hosts=None, agents=None, tol: float = 0.0, dt: int = 3600) -> dict:
    """For each candidate artifact (default: agent-deployed live hosts), remove it in a fork and compare with a
    baseline fork. Novelty is *useful* only if the removal reduces an outcome beyond its owner's own revenue:
    total requests served to households, consumers served (fewer unserved), or requests served by dependents."""
    from .counterfactual import fork_compare, undeploy_host
    cands = hosts if hosts is not None else [h for h in agent_hosts(world)
                                             if (world.state.get("deployments", world.state.tables["hosts"][h]
                                                                 ["deployment"]) or {}).get("status") == "live"]
    out = {}
    for h in cands:
        r = fork_compare(world, undeploy_host(h), horizon_s, agents, dt=dt)
        b, t = r["baseline"], r["treatment"]
        others_ok_b = sum(v["ok"] for k, v in b.get("requests", {}).items() if k != h)
        others_ok_t = sum(v["ok"] for k, v in t.get("requests", {}).items() if k != h)
        total_ok_b = sum(v["ok"] for v in b.get("requests", {}).values())
        total_ok_t = sum(v["ok"] for v in t.get("requests", {}).values())
        dep_loss = sum(-x.get("delta_ok", 0) for rows in r["dependents"].values() for x in rows.values()
                       if x.get("delta_ok", 0) < 0)
        reductions = {"households_served": total_ok_b - total_ok_t,
                      "consumers_served": t.get("unserved", 0) - b.get("unserved", 0),
                      "dependents_ok": dep_loss}
        useful = any(v > tol for v in reductions.values())
        out[h] = {"useful": useful, "reductions": reductions, "others_ok_delta": others_ok_t - others_ok_b,
                  "owner_revenue_delta": {k: v for k, v in r["delta"].get("revenue", {}).items()
                                          if k == world.state.tables["hosts"][h]["owner"]},
                  "exact": r["exact"], "summary": r["text"]}
    return {"hosts": out, "n_candidates": len(out), "n_useful": sum(1 for v in out.values() if v["useful"])}


def emergence_report(world, since: int = 0, every_s: int = DAY) -> dict:
    """All log/state metrics (not the counterfactual novelty check, which runs forks)."""
    return {"tool_reuse": tool_reuse(world, since), "diffusion": diffusion_curves(world, since, every_s),
            "concentration": market_concentration(world, since, every_s), "specialization": specialization(world),
            "strategy_diversity": strategy_diversity(world, since=since), "dependencies": dependency_graph(world, since)}


def entropy_bits(counts) -> float:
    return _entropy(counts)


__all__ = ["call_edges", "tool_reuse", "diffusion_curves", "market_concentration", "specialization",
           "strategy_diversity", "dependency_graph", "useful_novelty", "emergence_report", "hhi", "agent_hosts",
           "tool_category", "entropy_bits"]
