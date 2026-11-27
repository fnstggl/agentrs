"""Independent measurement of open-ended structure in a world (read-only; computed from authoritative state).

`census(world)` returns structural metrics, never counts of decorative content:

* capabilities: published; with independent adopters / payers; verified trial outcomes (positive / not);
  delay from a problem report to the first adoption that addresses it
* markets: emergent capability markets (>= 2 independent buyers), product clusters that sold, B2B trade volume,
  problem requests and responses
* technology graph: nodes and edges by kind (observed service-to-service calls, uses, improvement lineage), the
  longest composition chain, process knowledge traded
* knowledge: evidence shared by mail, beliefs held from others, process notes applied
* institutions: constitutions, proposals decided, contracts by outcome, negotiations by outcome
* organisations: created after genesis (by agents or simulated entrepreneurs), dissolved
* economy: households' purchases (API requests served, goods units), firms' operating result, losses by kind

`ablation(...)` runs the same world configuration and agents with mechanisms disabled through world parameters
(`ablate_adoption`, `ablate_sharing`, `ablate_requests`, `ablate_entry`, `ablate_diffusion`, `ablate_negotiation`)
and independent seeds, so the contribution of a mechanism is measured rather than assumed.
"""
from __future__ import annotations

import math
from typing import Callable

DAY = 86400
ABLATIONS = ("ablate_adoption", "ablate_sharing", "ablate_requests", "ablate_entry", "ablate_diffusion",
             "ablate_negotiation")


def _longest_chain(edges: list[tuple[str, str]]) -> int:
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    memo: dict[str, int] = {}

    def depth(n, seen):
        if n in memo:
            return memo[n]
        best = 0
        for m in adj.get(n, []):
            if m not in seen:
                best = max(best, 1 + depth(m, seen | {m}))
        memo[n] = best
        return best
    return max((depth(n, {n}) for n in adj), default=0)


def census(world) -> dict:
    st = world.state
    out: dict = {"t": world.now, "day": world.now // DAY, "events": world.seq}
    has = lambda t: t in st.tables  # noqa: E731
    if has("capabilities"):
        caps = world.modules["caps"]
        sums = [caps.summary(world, c) for c in st.ids("capabilities")]
        outs = [e for _, e in st.scan("cap_evidence") if e["kind"] == "outcome"]
        req_t = sorted(r["t"] for _, r in st.scan("ops_requests"))
        first_adopt = sorted(i["t"] for _, i in st.scan("cap_installs") if i["status"] == "active" and i["independent"])
        out["capabilities"] = {
            "published": len(sums), "active": sum(1 for _, c in st.scan("capabilities") if c["status"] == "active"),
            "with_independent_adopters": sum(1 for s in sums if s["independent_adopters"] > 0),
            "with_independent_payers": sum(1 for s in sums if s["independent_payers"] > 0),
            "installs": st.count("cap_installs"),
            "active_installs": sum(1 for _, i in st.scan("cap_installs") if i["status"] == "active"),
            "trials_verified": len(outs),
            "trials_positive_significant": sum(1 for e in outs if e["data"].get("effect_per_day", 0) > 2 * e["data"].get("se", 0)),
            "days_first_request_to_first_adoption": round((first_adopt[0] - req_t[0]) / DAY, 2)
            if first_adopt and req_t else None}
        g = caps.graph(world)
        calls = [(e["src"], e["dst"]) for e in g["edges"] if e["kind"] == "calls"]
        out["technology"] = {"nodes": len(g["nodes"]), "call_edges": len(calls),
                             "uses_edges": sum(1 for e in g["edges"] if e["kind"] == "uses"),
                             "improvement_edges": sum(1 for e in g["edges"] if e["kind"] == "improves"),
                             "longest_composition_chain": _longest_chain(calls),
                             "emergent_capability_markets": len(g["markets"]),
                             "categories": [c["label"] for c in caps.categories(world, 1)]}
    if has("ops_requests"):
        tags = sorted({t for _, r in st.scan("ops_requests") for t in r["tags"]})
        out["problems"] = {"requests": st.count("ops_requests"), "responses": st.count("ops_responses"),
                           "distinct_tags": tags}
        losses: dict[str, int] = {}
        for _, r in st.scan("ops_days"):
            for k, v in r["loss_n"].items():
                losses[k] = losses.get(k, 0) + v
        out["losses_events"] = losses
    if has("runs"):
        ind = world.modules["industry"]
        shop = world.modules.get("shoppers")
        out["industry"] = {"designs": st.count("designs"), "runs": st.count("runs"),
                           "units_made": sum((r["made"] or 0) for _, r in st.scan("runs")),
                           "b2b_trades": st.count("trade_orders"),
                           "b2b_units": sum(t["qty"] for _, t in st.scan("trade_orders")),
                           "retail_units_sold": sum(r.get("sold_total", 0) for _, r in st.scan("retail")),
                           "product_clusters_selling": len(shop.census(world)) if shop else None,
                           "machines": st.count("machines"), "process_notes": st.count("process_notes"),
                           "process_notes_sold": st.count("note_rights"),
                           "substitutions": sum(1 for _, m in st.scan("minds") for e in m["memory"]
                                                if e["kind"] == "substitute")}
        del ind
    if has("minds"):
        ms = [m for _, m in st.scan("minds")]
        out["knowledge"] = {"minds": len(ms), "firms": sum(1 for m in ms if m["kind"] == "firm"),
                            "people": sum(1 for m in ms if m["kind"] == "person"),
                            "evidence_messages_sent": sum(m["stats"]["shared"] for m in ms),
                            "beliefs_from_others": sum(1 for m in ms for b in m["beliefs"].values() if b.get("src") == "contact"),
                            "processes_learned": sum(len(m["routines"].get("process_learning", {}).get("applied", []))
                                                     for m in ms),
                            "mean_contacts": round(sum(len(m["contacts"]) for m in ms) / max(1, len(ms)), 2)}
    if has("contracts"):
        from ..services.governance import constitutions_view, proposals_view
        cs = [c["status"] for _, c in st.scan("contracts")]
        out["institutions"] = {"contracts": {k: cs.count(k) for k in sorted(set(cs))},
                               "deals": ({k: sum(1 for _, d in st.scan("deals") if d["status"] == k)
                                          for k in ("accepted", "rejected", "countered", "expired", "open")}
                                         if has("deals") else None),
                               # (workstream F, D-40: gov records are inst charters/decisions where inst is installed)
                               "constitutions": len(constitutions_view(world)),
                               "proposals_decided": sum(1 for p in proposals_view(world) if p["status"] != "open")}
    if has("orgs"):
        created = [o for _, o in st.scan("orgs") if o.get("created", 0) > 0]
        out["organisations"] = {"total": st.count("orgs"), "created_after_genesis": len(created),
                                "dissolved_or_bankrupt": sum(1 for _, o in st.scan("orgs") if o["status"] != "active")}
    if has("ops_days"):
        from ..economy.operations import operating_result
        firms = [m["org"] for _, m in st.scan("minds") if m["kind"] == "firm"] if has("minds") else []
        out["economy"] = {"firm_operating_result_total": sum(operating_result(r) for k, r in st.scan("ops_days")
                                                             if r["owner"] in set(firms)),
                          "api_requests_served": sum(r["ok"] for _, r in st.scan("svc_days")) if has("svc_days") else 0}
    return out


def ablation(make_world: Callable[[dict, int], object], make_agents: Callable[[object], list], days: int,
             seeds: list[int], flags: list[str | None]) -> dict:
    """Run every (flag, seed) and return the census per run plus per-flag means of key metrics."""
    from ..agents.base import Runner
    runs = []
    for flag in flags:
        for seed in seeds:
            params = {flag: True} if flag else {}
            w = make_world(params, seed)
            ag = make_agents(w)
            Runner(w, ag, dt=3600, record=False, seed=seed).run(24 * days)
            c = census(w)
            runs.append({"flag": flag or "none", "seed": seed, "census": c})
    keys = [("capabilities", "active_installs"), ("capabilities", "trials_verified"),
            ("capabilities", "days_first_request_to_first_adoption"), ("technology", "emergent_capability_markets"),
            ("industry", "retail_units_sold"), ("industry", "process_notes_sold"), ("knowledge", "beliefs_from_others"),
            ("economy", "firm_operating_result_total"), ("organisations", "created_after_genesis"),
            ("institutions", "deals")]
    summary = {}
    for flag in [f or "none" for f in flags]:
        rs = [r["census"] for r in runs if r["flag"] == flag]
        summary[flag] = {}
        for a, b in keys:
            vals = [r.get(a, {}).get(b) for r in rs]
            num = [v for v in vals if isinstance(v, (int, float))]
            summary[flag][f"{a}.{b}"] = {"mean": round(sum(num) / len(num), 2) if num else None,
                                         "sd": round(math.sqrt(sum((v - sum(num) / len(num)) ** 2 for v in num) /
                                                               max(1, len(num) - 1)), 2) if len(num) > 1 else None,
                                         "values": vals}
    return {"runs": runs, "summary": summary}
