"""Scientific evaluation of open-endedness in the synthetic internet (Part 21).

`measure(world, t0)` computes the 15 measures from authoritative state since `t0`. None of them is an artifact
count: each is grounded in an independent party's behaviour (an install kept after the installer's own trial, a
payment, a verified trial outcome, a loss that did not exist before) or in a counterfactual.

`ablate(make_world, make_agents, days, seeds)` runs the factual world and four counterfactual conditions with the
same seeds - **invention removed** (the inventing agents present but idle), **communication off** (evidence
sharing, forum posts, mail of outcomes and negotiation disabled), **dynamic demand off** (no problem reports:
firms cannot publish needs), **propagation off** (no technique diffusion and no imitating entrants) - and reports
per-measure differences, so the dependence of the world's complexity on each mechanism is measured rather than
asserted. Measures 4 (second-order opportunities) and 12 (causal impact) are only *established* through the
`invention_removed` comparison; `measure` alone reports them as candidates.

Measures 13 (transfer) and 15 (learning across lifetimes) need several worlds or lifetimes; helpers `transfer`
and `learning_curve` compute them from per-run results.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import numpy as np

DAY = 86400
CONDITIONS = {
    "factual": {},
    "invention_removed": None,           # same world; the inventing agents idle
    "communication_off": {"ablate_sharing": True, "ablate_negotiation": True},
    "demand_off": {"ablate_requests": True},
    "propagation_off": {"ablate_diffusion": True, "ablate_entry": True},
}


def _t(st, name):
    return name in st.tables


def _firms(st):
    return sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm") if _t(st, "minds") else []


def _related(world, a, b):
    from ...goals.metrics import related_parties
    return b in related_parties(world, a)


# ============================================================================ the 15 measures
def m1_useful_capabilities(world, t0):
    """Capabilities published after t0 with at least one *independent* verified trial outcome whose effect is
    positive and exceeds 2 standard errors (the installer's own books)."""
    st = world.state
    if not _t(st, "capabilities"):
        return {"value": 0}
    useful = {}
    for _, e in st.scan("cap_evidence"):
        c = st.tables["capabilities"][e["capability"]]
        if e["kind"] == "outcome" and e["independent"] and c["t"] >= t0:
            d = e["data"]
            if d.get("effect_per_day", 0) > 2 * d.get("se", 0) > 0:
                useful.setdefault(e["capability"], []).append(e["by"])
    return {"value": len(useful), "capabilities": {k: sorted(set(v)) for k, v in sorted(useful.items())},
            "published": sum(1 for _, c in st.scan("capabilities") if c["t"] >= t0)}


def m2_independent_adoption(world, t0):
    st = world.state
    if not _t(st, "cap_installs"):
        return {"value": 0}
    act = [i for _, i in st.scan("cap_installs") if i["status"] == "active" and i["independent"] and i["t"] >= t0]
    tenure = [(world.now - int(i.get("status_t") or i["t"])) / DAY for i in act]
    return {"value": len({i["installer"] for i in act}), "active_installs": len(act),
            "kept_14_days": sum(1 for x in tenure if x >= 14),
            "tried": len({i["installer"] for _, i in st.scan("cap_installs") if i["t"] >= t0})}


def m3_new_markets(world, t0):
    """Emergent capability markets (>= 2 independent buyers of substitutable capabilities), improvement entrants,
    and organisations created after t0 that earned independent revenue."""
    st = world.state
    from ...goals.metrics import m_independent_revenue
    markets = [m for _, m in st.scan("cap_markets")] if _t(st, "cap_markets") else []
    entrants = [c for _, c in st.scan("capabilities") if c.get("improves") and c["t"] >= t0] \
        if _t(st, "capabilities") else []
    new_orgs = [k for k, o in st.scan("orgs") if o.get("created", 0) > t0 and o["status"] == "active"
                and m_independent_revenue(world, k, world.now - t0 + 1) > 0]
    return {"value": len(markets) + len(new_orgs), "capability_markets": len(markets),
            "improvement_entrants": len(entrants), "new_orgs_with_independent_revenue": len(new_orgs)}


def m4_second_order_candidates(world, t0):
    """Loss kinds that first occur after the first independent adoption after t0 (temporal precedence only; the
    counterfactual comparison in `ablate` establishes or rejects them)."""
    st = world.state
    if not (_t(st, "ops_days") and _t(st, "cap_installs")):
        return {"value": 0, "kinds": []}
    adopt = sorted(i["t"] for _, i in st.scan("cap_installs") if i["independent"] and i["t"] >= t0)
    first: dict[str, int] = {}
    for _, r in st.scan("ops_days"):
        for k, n in r["loss_n"].items():
            if n:
                first[k] = min(first.get(k, 10 ** 12), r["day"] * DAY)
    kinds = sorted(k for k, t in first.items() if adopt and t > adopt[0] and t > t0)
    return {"value": len(kinds), "kinds": kinds, "first_day": {k: first[k] // DAY for k in kinds}}


def m5_persistence(world, t0):
    st = world.state
    if not _t(st, "cap_installs"):
        return {"value": None}
    ever = [i for _, i in st.scan("cap_installs") if i["t"] >= t0 and i["independent"] and
            (i["status"] == "active" or i.get("reason") == "adopted")]
    still = [i for i in ever if i["status"] == "active"]
    ten = sorted((world.now - int(i.get("status_t") or i["t"])) / DAY for i in still)
    return {"value": round(len(still) / len(ever), 3) if ever else None, "adopted": len(ever), "still_active": len(still),
            "median_active_days": ten[len(ten) // 2] if ten else None}


def m6_information_propagation(world, t0):
    st = world.state
    if not _t(st, "minds"):
        return {"value": 0}
    ms = [m for _, m in st.scan("minds")]
    beliefs = sum(1 for m in ms for b in m["beliefs"].values() if b.get("src") == "contact")
    delays = []
    if _t(st, "cap_evidence"):
        pub = {}
        for _, e in st.scan("cap_evidence"):
            if e["kind"] == "outcome" and e.get("public"):
                pub.setdefault(e["capability"], []).append((e["t"], e["by"]))
        for cid, evs in pub.items():
            t_pub, by = min(evs)
            later = sorted(i["t"] for i in st.find("cap_installs", "capability", cid) if i["t"] > t_pub and
                           i["installer"] != by)
            if later:
                delays.append((later[0] - t_pub) / DAY)
    return {"value": beliefs, "beliefs_from_others": beliefs,
            "evidence_messages": sum(m["stats"]["shared"] for m in ms),
            "median_days_public_evidence_to_next_trial": float(np.median(delays)) if delays else None}


def m7_social_coordination(world, t0):
    st = world.state
    deals = sum(1 for _, d in st.scan("deals") if d["status"] == "accepted" and d["t"] >= t0) if _t(st, "deals") else 0
    contracts = sum(1 for _, c in st.scan("contracts") if c["status"] in ("completed", "fulfilled", "active")
                    and c.get("t", 0) >= t0) if _t(st, "contracts") else 0
    grants = sum(1 for _, g in st.scan("grants") if g.get("t", 0) >= t0 and g.get("status", "active") == "active"
                 and not _related(world, g["owner"], g["grantee"])) if _t(st, "grants") else 0
    return {"value": deals + contracts + grants, "deals_accepted": deals, "contracts": contracts,
            "independent_delegations": grants}


def m8_strategic_competition(world, t0):
    """Problem classes served by >= 2 independent providers with installs; HHI of installs per class; provider
    price moves."""
    st = world.state
    out = {"contested_classes": 0, "hhi": {}, "provider_price_changes": 0}
    if _t(st, "capabilities"):
        by_cls: dict[str, Counter] = {}
        for _, i in st.scan("cap_installs"):
            c = st.tables["capabilities"][i["capability"]]
            root = c
            while root.get("improves") and root["improves"] in st.tables["capabilities"]:
                root = st.tables["capabilities"][root["improves"]]
            by_cls.setdefault(root["host"], Counter())[c["provider"]] += 1
        for cls, cnt in sorted(by_cls.items()):
            tot = sum(cnt.values())
            out["hhi"][cls] = round(sum((v / tot) ** 2 for v in cnt.values()), 3)
            out["contested_classes"] += len(cnt) >= 2
    if _t(st, "providers"):
        out["provider_price_changes"] = sum(1 for e in world.log if e.type == "cloud.provider_price" and e.t >= t0) \
            if hasattr(world, "log") else None
    out["value"] = out["contested_classes"]
    return out


def m9_institutional_evolution(world, t0):
    # (workstream F, D-40: gov proposals/constitutions are inst decisions/charters where inst is installed; gov and
    # inst both record a passed decision as "executed", which this metric did not count before)
    from ...services.governance import constitutions_view, proposals_view
    props = [p for p in proposals_view(world) if p.get("t", 0) >= t0]
    decided = sum(1 for p in props if p["status"] != "open")
    passed = sum(1 for p in props if p["status"] in ("executed", "passed", "approved", "enacted"))
    consts = len(constitutions_view(world))
    return {"value": passed, "proposals_decided": decided, "passed": passed, "constitutions": consts}


def m10_viability(world, t0):
    st = world.state
    from ...economy.operations import operating_result
    firms = _firms(st)
    if not firms or not _t(st, "ops_days"):
        return {"value": None}
    d0, d1 = t0 // DAY, world.now // DAY
    act = sum(1 for f in firms if st.tables["orgs"].get(f, {}).get("status", "active") == "active")
    span = max(1, min(14, (d1 - d0) // 2))
    first = sum(operating_result(st.tables["ops_days"].get(f"{f}|{d}")) for f in firms for d in range(d0, d0 + span))
    last = sum(operating_result(st.tables["ops_days"].get(f"{f}|{d}")) for f in firms for d in range(d1 - span, d1))
    pos = sum(1 for f in firms if sum(operating_result(st.tables["ops_days"].get(f"{f}|{d}"))
                                    for d in range(d1 - span, d1)) > 0)
    return {"value": round(pos / len(firms), 3), "firms": len(firms), "active": act,
            "firms_with_positive_recent_result": pos, "result_first_window": first, "result_last_window": last}


def m11_novel_problem_solving(world, t0):
    """Problem reports after t0 whose reporter adopted (after its own trial) a capability offered in a response
    to that report, and whose losses of the reported kind then fell below the reported rate."""
    st = world.state
    if not (_t(st, "ops_requests") and _t(st, "ops_responses")):
        return {"value": 0}
    solved = []
    for rid, r in st.scan("ops_requests"):
        if r["t"] < t0:
            continue
        offered = {x["capability"] for x in st.find("ops_responses", "request", rid) if x.get("capability")}
        kept = [i for i in st.find("cap_installs", "installer", r["by"]) if i["status"] == "active" and
                i["capability"] in offered]
        ev = r.get("evidence") or {}
        if not kept or not ev.get("loss_events"):
            continue
        days = max(1, min(int(ev.get("days", 7)), r["t"] // DAY))
        today = world.now // DAY
        for kind, n in ev["loss_events"].items():
            now = sum((st.tables["ops_days"].get(f"{r['by']}|{d}") or {"loss_n": {}})["loss_n"].get(kind, 0)
                      for d in range(today - 7, today))
            if n and now < n * 7 / days:
                solved.append({"request": rid, "by": r["by"], "kind": kind, "reported_per_7d": round(n * 7 / days, 1),
                               "last_7d": now})
    return {"value": len({s["request"] for s in solved}), "solved": solved}


def m14_successful_strategy_diversity(world, t0):
    from ...goals.metrics import m_independent_revenue
    from ..emergence import strategy_diversity
    st = world.state
    agents = [pid for pid, p in st.scan("persons") if p["kind"] in ("agent", "human")]
    ok = []
    for a in agents:
        orgs = [m["org"] for m in st.find("members", "person", a) if m["role"] == "owner"]
        if any(m_independent_revenue(world, o, world.now - t0 + 1) > 0 for o in orgs):
            ok.append(a)
    if len(ok) < 1:
        return {"value": 0, "successful_agents": []}
    sd = strategy_diversity(world, actors=ok, since=0)
    return {"value": sd.get("n_clusters", 0), "successful_agents": ok, "mean_pairwise_js": sd.get("mean_pairwise_js"),
            "cluster_sizes": sd.get("cluster_sizes")}


def m15_within_lifetime_improvement(world, t0):
    """Per agent: slope of weekly independent revenue over the lifetime (a learner's improvement shows here; a
    fixed script's does not need to)."""
    from ...goals.metrics import counterparty_flows, related_parties
    st = world.state
    out = {}
    for a, p in st.scan("persons"):
        if p["kind"] not in ("agent", "human"):
            continue
        orgs = [m["org"] for m in st.find("members", "person", a) if m["role"] == "owner"]
        weeks = []
        for w0 in range(t0, world.now - 7 * DAY + 1, 7 * DAY):
            tot = 0
            for o in orgs:   # flows inside [w0, w0 + 7d) = flows since w0 minus flows since w0 + 7d
                rel = related_parties(world, o)
                since = counterparty_flows(world, o, world.now - w0)
                after = counterparty_flows(world, o, world.now - w0 - 7 * DAY)
                for cp, (i_s, o_s) in since.items():
                    if cp in rel or cp in ("sys", "escrow"):
                        continue
                    i_a, o_a = after.get(cp, [0, 0])
                    tot += max(0, (i_s - i_a) - (o_s - o_a))
            weeks.append(tot)
        if len(weeks) >= 3:
            x = np.arange(len(weeks))
            out[a] = {"weekly_independent_revenue": weeks, "slope_per_week": round(float(np.polyfit(x, weeks, 1)[0]), 1)}
    return {"value": len([v for v in out.values() if v["slope_per_week"] > 0]), "agents": out}


MEASURES = {
    "1_new_useful_capability_creation": m1_useful_capabilities,
    "2_independent_adoption": m2_independent_adoption,
    "3_new_markets": m3_new_markets,
    "4_second_order_opportunities": m4_second_order_candidates,
    "5_persistence_of_technological_change": m5_persistence,
    "6_information_propagation": m6_information_propagation,
    "7_social_coordination": m7_social_coordination,
    "8_strategic_competition": m8_strategic_competition,
    "9_institutional_evolution": m9_institutional_evolution,
    "10_long_term_economic_viability": m10_viability,
    "11_novel_problem_solving": m11_novel_problem_solving,
    "14_diversity_of_successful_strategies": m14_successful_strategy_diversity,
    "15_agent_learning_within_lifetime": m15_within_lifetime_improvement,
}


def measure(world, t0: int = 0) -> dict:
    out = {"t0_day": t0 // DAY, "day": world.now // DAY}
    for k, fn in MEASURES.items():
        try:
            out[k] = fn(world, t0)
        except Exception as e:  # noqa: BLE001 - a missing module makes a measure inapplicable, not an error
            out[k] = {"value": None, "error": repr(e)[:160]}
    out["12_causal_impact_of_invented_artifacts"] = {"value": None, "requires": "ablate(): invention_removed"}
    out["13_transfer_across_world_structures"] = {"value": None, "requires": "transfer(): several world families"}
    out["outcomes"] = outcomes(world, t0)
    return out


def outcomes(world, t0) -> dict:
    """Welfare-relevant totals used for causal comparison (not novelty measures)."""
    st = world.state
    from ...economy.operations import operating_result
    firms = set(_firms(st))
    o = {"requests_served": sum(r["ok"] for k, r in st.scan("svc_days") if int(k.rsplit("|", 1)[1]) * DAY >= t0)
         if _t(st, "svc_days") else 0}
    if _t(st, "ops_days"):
        loss: Counter = Counter()
        res = 0
        for _, r in st.scan("ops_days"):
            if r["owner"] in firms and r["day"] * DAY >= t0:
                res += operating_result(r)
                loss.update(r["loss_n"])
        o["firm_operating_result"] = res
        o["loss_events"] = dict(sorted(loss.items()))
    if _t(st, "retail"):
        o["retail_units_sold"] = sum(r.get("sold_total", 0) for _, r in st.scan("retail"))
    return o


# ============================================================================ counterfactual ablations
def _scalar(m: dict) -> float | None:
    v = m.get("value") if isinstance(m, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def ablate(make_world: Callable[[dict, int], object], make_agents: Callable[[object, bool], list], days: int,
           seeds: list[int], conditions: dict | None = None, warmup_days: int = 0) -> dict:
    """make_world(params, seed) -> World; make_agents(world, inventing: bool) -> [(actor, policy)]: with
    inventing=False the same actors are admitted but act as idle (invention removed)."""
    from ..open_endedness import census
    from ...agents.base import Runner
    conds = conditions or CONDITIONS
    runs = []
    for name, params in conds.items():
        for seed in seeds:
            w = make_world(params or {}, seed)
            ag = make_agents(w, params is not None)
            if warmup_days:
                from ...agents.scripted import IdlePolicy
                Runner(w, [(a, IdlePolicy()) for a, _ in ag], dt=3600, record=False).run(24 * warmup_days)
            t0 = w.now
            Runner(w, ag, dt=3600, record=False, seed=seed).run(24 * days)
            runs.append({"condition": name, "seed": seed, "measures": measure(w, t0), "census": census(w),
                         "invariants": w.check_invariants()})
    keys = [k for k in MEASURES] + ["outcomes.requests_served", "outcomes.firm_operating_result"]
    summary: dict = {}
    for name in conds:
        rs = [r for r in runs if r["condition"] == name]
        summary[name] = {}
        for k in keys:
            if k.startswith("outcomes."):
                vals = [r["measures"]["outcomes"].get(k.split(".", 1)[1]) for r in rs]
            else:
                vals = [_scalar(r["measures"][k]) for r in rs]
            num = [float(v) for v in vals if isinstance(v, (int, float))]
            summary[name][k] = {"mean": round(sum(num) / len(num), 3) if num else None, "values": vals}
    # differences vs factual, paired by seed
    diff: dict = {}
    for name in conds:
        if name == "factual":
            continue
        diff[name] = {}
        for k in keys:
            fv, cv = summary["factual"][k]["values"], summary[name][k]["values"]
            pairs = [(a, b) for a, b in zip(fv, cv) if isinstance(a, (int, float)) and isinstance(b, (int, float))]
            if pairs:
                d = [a - b for a, b in pairs]
                diff[name][k] = {"mean_factual_minus_condition": round(sum(d) / len(d), 3),
                                 "sd": round(float(np.std(d, ddof=1)), 3) if len(d) > 1 else None, "n": len(d),
                                 "positive_in_every_seed": all(x > 0 for x in d)}
    # second-order kinds established by the invention_removed counterfactual
    so = {}
    for seed in seeds:
        f = next((r for r in runs if r["condition"] == "factual" and r["seed"] == seed), None)
        c = next((r for r in runs if r["condition"] == "invention_removed" and r["seed"] == seed), None)
        if f and c:
            fk = set(f["measures"]["outcomes"].get("loss_events", {}))
            ck = {k for k, v in c["measures"]["outcomes"].get("loss_events", {}).items() if v}
            so[seed] = sorted(k for k in fk if k not in ck and f["measures"]["outcomes"]["loss_events"][k] > 0)
    return {"runs": runs, "summary": summary, "differences": diff, "second_order_established": so,
            "complexity_dependence": _dependence(diff)}


def _dependence(diff: dict) -> dict:
    """For each condition: which grounded measures drop when the mechanism is removed. 'depends' requires
    factual - condition > 0 in every paired seed, not just on average."""
    out = {}
    for name, d in diff.items():
        out[name] = sorted(k for k, v in d.items() if not k.startswith("outcomes.") and v["positive_in_every_seed"])
    return out


# ============================================================================ multi-run measures
def transfer(results: dict[str, list[float]]) -> dict:
    """results: world family -> per-run scores of one agent/policy. Reports mean per family and the ratio of the
    worst family to the best (1 = transfers evenly)."""
    means = {k: float(np.mean(v)) for k, v in results.items() if v}
    if not means:
        return {"value": None}
    hi, lo = max(means.values()), min(means.values())
    return {"value": round(lo / hi, 3) if hi > 0 else None, "per_family": means}


def learning_curve(episode_scores: list[float]) -> dict:
    """Slope over successive lifetimes with a bootstrap 90% interval."""
    y = np.asarray(episode_scores, dtype=float)
    if len(y) < 3:
        return {"value": None}
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(idx)) > 1:
            bs.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    return {"value": round(slope, 4), "ci90": [round(float(np.percentile(bs, 5)), 4),
                                               round(float(np.percentile(bs, 95)), 4)]}


__all__ = ["measure", "ablate", "transfer", "learning_curve", "MEASURES", "CONDITIONS", "outcomes"]
del math
