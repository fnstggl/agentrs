"""Gate SP-4.8b: population-level social-to-economy chains, with infrastructure cost, retention and demand effects,
through ordinary mechanisms only. Owner: workstream C (the economy side is workstream ECON's `economy.work`,
`commerce`).

One civilisation (`crucible.civ.generate`, profile balanced, development population, 14 simulated days, nothing
scripted): the one population works, shops for and relies on the incumbents' SaaS products (`economy.work`), and
uses the social internet. A member who relies on a service that works for it sometimes recommends it in a post that
names the service's address (`social.popcreate`); feeds distribute the post; a reader with that task in its own work
looks into it - a `shop` session of `economy.work` (`social.popinbox._service_leads`), where it searches, weighs and
maybe subscribes by work's own rules. A chain is read from the recorded log, in causal order, for one reader B and one
product P:

  1. social exposure  - B viewed (qualifying consumption) an item that links P (a launch) or names P's host;
  2. discovery        - afterwards B considered P in a shop session (`work.considered`);
  3. adoption         - afterwards B relied on P (`work.rely` active) through a subscription of its own;
  4. use              - afterwards B's requests ran on P's live deployment (`prod.used`).

Effect kinds observed on completed chains (each counted when at least one chain shows it):
  infrastructure_cost - the reader's requests after its adoption ran on the seller's live deployment and consumed
                        compute there (`cloud.usage` compute units: load on infrastructure the seller rents);
  retention           - the chain's subscription was renewed (a paid period after the first) or ended by B's own
                        decision (a cancellation): the experience decided whether B stayed;
  demand              - B rated P (`prod.rate`, read by later shoppers' priors) or itself recommended P in a post;
  revenue             - B paid an invoice of P's subscription from its own account.

No bonus: no code of the social package moves money (static scan of SP-4.8), and every step is a recorded command
or decision of the members themselves.

Metrics (ACCEPTANCE_MANIFEST): population_social_economy_chains >= 3, chain_effect_kinds >= 3.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.social import experiments as X
from crucible.substrate import ids
from social_helpers import assert_met, failed_ticks  # noqa: E402

crucible.load_all_modules()
CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_8b.py"
DAY = 86400
DAYS = 14
pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def _ev(e):
    return e if isinstance(e, dict) else e.to_dict()


def test_SP_4_8b_population_social_economy_chains():
    from crucible.civ import generate
    from crucible.social.popinbox import _hosts_of
    w = generate("balanced", 48, "development")
    for d in range(DAYS):
        w.advance_time((d + 1) * DAY)
    st = w.state
    events = [_ev(e) for e in w.events_range(0)]
    host_of = {pid: (p["delivery"] or {}).get("host") for pid, p in st.tables.get("products", {}).items()}
    by_host = defaultdict(set)
    for pid, h in host_of.items():
        if h:
            by_host[h].add(pid)
    # items that link or name a product
    item_products: dict[int, set] = {}
    for it, info in st.tables["aud_item_info"].items():
        cid = info.get("content_id")
        if not cid:
            continue
        prods = set()
        L = st.get("launches", cid) if "launches" in st.tables else None
        if L is not None and L["link"] == "product":
            prods.add(L["target"])
        for h in _hosts_of(w, cid):
            prods |= by_host.get(h, set())
        if prods:
            item_products[int(it)] = prods
    con = X.decision_rows(w, "consume", 0)
    first_view: dict[tuple, int] = {}
    for v, it, t, ok in zip(con["viewer"].tolist(), con["item"].tolist(), con["t"].tolist(), con["is_view"].tolist()):
        if ok and it in item_products:
            for pid in item_products[it]:
                k = (int(v), pid)
                if k not in first_view or t < first_view[k]:
                    first_view[k] = int(t)
    considered, rely, used, rated, cancels = {}, {}, defaultdict(list), set(), set()
    compute = defaultdict(list)                       # reader -> [(t, compute units, outcome)] of its requests
    for d in events:
        p = d["payload"]
        if d["type"] == "cloud.usage":
            i = ids.index(d.get("actor") or "")
            if i is not None:
                compute[i].append((int(d["t"]), float(p.get("cu") or 0.0), p.get("outcome")))
            continue
        if d["type"] == "work.considered":
            for i, prov in p["rows"]:
                considered.setdefault((int(i), prov), []).append(int(d["t"]))
        elif d["type"] == "work.rely" and p.get("active"):
            rely.setdefault((int(p["i"]), p["prov"]), int(d["t"]))
        elif d["type"] == "prod.used":
            i = ids.index(p.get("holder") or "")
            if i is not None:
                used[(i, p["product"])].append(int(d["t"]))
        elif d["type"] == "prod.rating":
            i = ids.index(p.get("rater") or "")
            if i is not None:
                rated.add((i, p.get("product")))
        elif d["type"] == "subs.status" and (p.get("changes") or {}).get("cancel_at_period_end") or \
                d["type"] == "subs.status" and (p.get("changes") or {}).get("status") == "cancelled":
            i = ids.index(d.get("actor") or "")
            if i is not None:
                cancels.add(i)
    chains = []
    for (i, pid), t1 in sorted(first_view.items()):
        t2 = [t for t in considered.get((i, pid), []) if t >= t1]
        if not t2:
            continue
        t3 = rely.get((i, pid))
        if t3 is None or t3 < min(t2):
            continue
        t4 = [t for t in used.get((i, pid), []) if t >= t3]
        if not t4:
            continue
        chains.append({"reader": i, "product": pid, "viewed": t1, "considered": min(t2), "adopted": t3,
                       "first_use": min(t4), "uses": len(t4)})
    # effects of the chains
    kinds = defaultdict(int)
    subs = st.tables.get("sub_subs", {})
    invoices = st.tables.get("sub_invoices", {})
    cu_total = overloads = 0.0
    recommended_back = 0
    rec_posts = {}
    for d in events:
        if d["type"] == "content.published" and ids.index(d["payload"].get("creator") or "") is not None:
            rec_posts.setdefault(ids.index(d["payload"]["creator"]), []).append((int(d["t"]), d["payload"]["id"]))
    for c in chains:
        p = ids.pid(c["reader"])
        req = [(t, cu, o) for t, cu, o in compute.get(c["reader"], []) if t >= c["adopted"]]
        cu = sum(x for _, x, _ in req)
        cu_total += cu
        overloads += sum(1 for _, _, o in req if o == "overload")
        if cu > 0:
            kinds["infrastructure_cost"] += 1
        mine = {sid: s_ for sid, s_ in subs.items() if s_.get("customer") == p and s_.get("product") == c["product"]}
        if any(int(s_.get("periods_paid", 0)) >= 2 for s_ in mine.values()) or \
                any(s_.get("status") == "cancelled" or s_.get("cancel_at_period_end") for s_ in mine.values()):
            kinds["retention"] += 1
        host = host_of.get(c["product"])
        again = [cid for t, cid in rec_posts.get(c["reader"], []) if t > c["first_use"] and host and
                 host in _hosts_of(w, cid)]
        recommended_back += len(again)
        if (c["reader"], c["product"]) in rated or again:
            kinds["demand"] += 1
        paid = [v for v in invoices.values() if v.get("subscription") in mine and v.get("status") == "paid"]
        if paid:
            kinds["revenue"] += 1
    sellers = {(st.get("products", c["product"]) or {}).get("owner") for c in chains}
    observed = sorted(k for k, v in kinds.items() if v > 0)
    metrics = {"population_social_economy_chains": len(chains), "chain_effect_kinds": len(observed),
               "effect_kinds": ",".join(observed), **{f"chains_with_{k}": int(v) for k, v in kinds.items()},
               "readers": len({c["reader"] for c in chains}), "products": len({c["product"] for c in chains}),
               "items_naming_products": len(item_products), "social_exposures": len(first_view),
               "sellers": len(sellers), "chain_compute_units": round(cu_total, 3), "chain_overloaded_requests":
               int(overloads), "chain_readers_who_recommended_it_in_turn": recommended_back,
               "cancellations_by_readers": len(cancels & {c["reader"] for c in chains}), "days": DAYS}
    metrics.update(failed_ticks(w))
    evidence.record("SP-4.8b", metrics, command=CMD, notes=(
        "A balanced civilisation (development population, 14 days). Chains read from the log: a qualifying view of "
        "an item linking or naming a product, then the reader's own shop consideration of it, its adoption "
        "(subscription) and its use on the seller's deployment, with the effects that followed. "
        f"Chains: {json.dumps(chains[:20])}"))
    assert_met('SP-4.8b', metrics)
    assert metrics["failed_ticks"] == 0, metrics["failed_tick_errors"]
    assert w.check_invariants() == []
