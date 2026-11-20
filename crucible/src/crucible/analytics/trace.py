"""Causal debugging over the event log: timelines, "why did X happen" explanations, causal chains.

* `timeline(world, entity, since=0)` - the events that mention an entity (as actor, in the payload, or through its
  aliases: an organisation's accounts, hosts, deployments, instances and listings), oldest first.
* `explain(world, entity)` - structured findings plus readable text answering the natural questions:
    - organisations: why did the company fail (or how is it doing)? preceding claims, breached contracts, unpaid
      fees/invoices, failed instance bills (suspensions), provider outages, lost customers (from the consumer
      market's hourly slot summaries), service failure rates, revenue and cost trend;
    - hosts: why is the service (un)popular? customers over time, ratings, ad impressions, search appearances
      (organic / sponsored), price versus rivals, reliability; and for every drop in customers, what preceded it
      (failure rates, overload, unavailability, price changes, outages, rival entries);
    - accounts and persons: flows by counterparty, organisations, rejected actions.
* `causal_chain(world, seq)` - follows an event backwards: the transaction that produced it (the command and its
  actor, or the scheduled process), who scheduled that process, and the creation events of every entity referenced
  in the payload or the command arguments.

All functions are read-only. They use authoritative state for current facts and the event log for history
(`crucible.analytics.world_events`, which follows fork lineage in the store).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np

from . import (DAY, HOUR, NOISE_TYPES, accounts_of, day_of, has_table, hosts_of, money, owned_orgs,
               resolve_entity, slot_map, world_events)

REVENUE_KINDS = {"sale", "invoice", "contract", "claim", "bounty", "salary_in"}
COST_LABELS = {"compute": "compute", "fee": "fees", "opex": "opex", "ads": "advertising", "salary": "wages",
               "wage": "wages", "sale": "purchases", "refund": "refunds", "contract": "contract payments",
               "interest": "interest", "loan": "loan repayments"}
_ID_RE = re.compile(r"^(acct|adc|bounty|cev|chg|claim|ctr|dep|doc|fpost|freply|grant|hw|icmt|inst|inv|iss|job|labx|"
                    r"lst|mail|mandate|ord|proj|rg|sg|tkt|tx|app|sub|post)_[0-9]+$")


# ====================================================================== event text
def describe_event(ev) -> str:
    """One readable line for an event (operator view: payload contents are shown)."""
    p = ev.payload if isinstance(ev.payload, dict) else {}
    t = ev.type
    a = ev.actor
    try:
        if t == "cmd":
            if p.get("system"):
                return f"operator system call {p.get('tool')} as {p.get('as')}"
            return f"{a} called {p.get('tool')}({_short(p.get('args'), 120)})"
        if t == "cmd.rejected":
            err = p.get("error") or {}
            return f"{p.get('actor') or a} was refused {p.get('tool')}: {err.get('code')}: {str(err.get('message', ''))[:100]}"
        if t == "bank.txn":
            legs = ", ".join(f"{acct} {'+' if d > 0 else ''}{money(d)}" for acct, d in p.get("postings", []))
            return f"{p.get('kind')} txn {p.get('id')}: {legs} ({p.get('memo', '')[:60]})"
        if t == "cloud.instance_status":
            return f"instance {p.get('id')} -> {p.get('status')} ({p.get('reason', '')})"
        if t == "cloud.dep_status":
            return f"deployment {p.get('id')} -> {p.get('status')} ({p.get('reason', '')})"
        if t == "cloud.outage":
            return f"provider {p.get('id')} outage day {day_of(p.get('start', 0))} {(p.get('until', 0) - p.get('start', 0)) // HOUR}h"
        if t == "cloud.deployment":
            return f"{p.get('owner')} deployed {p.get('repo')}@{str(p.get('commit'))[:8]} as {p.get('host')}.crucible"
        if t == "cloud.instance":
            return f"{p.get('owner')} rented {p.get('cores')} cores at {p.get('provider')} ({p.get('id')})"
        if t == "contracts.claim":
            return f"claim {p.get('id')}: {p.get('debtor')} owes {p.get('creditor')} {money(p.get('amount'))} ({p.get('source')})"
        if t == "contracts.claim_status":
            return f"claim {p.get('id')} -> {p.get('status')} (paid {money(p.get('paid', 0))})"
        if t == "contracts.status":
            return f"contract {p.get('id')} -> {p.get('status')} {p.get('reason', '') or ''}".strip()
        if t == "biz.distress":
            return (f"{p.get('org')} in distress (owes {money(p.get('owed'))})" if p.get("since") is not None
                    else f"{p.get('org')} left distress")
        if t == "biz.status":
            return f"{p.get('org')} -> {p.get('status')} (cash {money(p.get('cash'))}, owed {money(p.get('owed'))})"
        if t == "biz.fee":
            return f"{p.get('org')} {'FAILED to pay' if p.get('unpaid') else 'paid'} the registration fee"
        if t == "market.listing":
            return f"{p.get('seller')} listed {p.get('kind')} '{str(p.get('title'))[:50]}' at {p.get('price')}c ({p.get('id')})"
        if t == "market.listing_update":
            return f"listing {p.get('id')} changed {_short(p.get('changes'), 80)}"
        if t == "market.rating":
            return f"{p.get('buyer')} rated {p.get('listing')} {p.get('stars')}/5"
        if t == "consumers.hour":
            n = sum(s.get("customers", 0) for s in p.get("slots", {}).values())
            r = sum(s.get("revenue", 0) for s in p.get("slots", {}).values())
            return f"consumer market hour: {len(p.get('slots', {}))} offerings served {n} customers, revenue {money(r)}"
        if t == "support.open":
            return f"ticket {p.get('id')} opened against {p.get('host')} by {p.get('customer')}: {str(p.get('subject'))[:50]}"
        if t == "forum.post":
            return f"{p.get('author')} posted '{str(p.get('title'))[:60]}' on {p.get('board')}"
        if t == "mail.sent":
            return f"{p.get('from')} mailed {len(p.get('to', []))} recipient(s)"
        if t == "proc":
            return f"scheduled process {p.get('proc')} ({p.get('id')})"
        if t == "proc.failed":
            return f"process {p.get('proc')} FAILED: {str(p.get('error'))[:100]}"
        if t == "sched.add":
            return f"scheduled {p.get('proc')} at day {day_of(p.get('at', 0))} ({p.get('id')})" + \
                   (f" every {p['period']}s" if p.get("period") else "")
        if t == "ads.campaign":
            return f"{p.get('advertiser')} started ad campaign {p.get('id')} on {p.get('keywords')}"
        if t == "ads.impressions":
            return f"campaign {p.get('id')}: {p.get('n')} impressions, {money(p.get('amount'))}"
        if t == "id.org":
            return f"{p.get('founder')} created organisation {p.get('id')}"
        if t == "id.member":
            return f"{p.get('person')} joined {p.get('org')} as {p.get('role')}"
        if t == "principal.update":
            return f"principal {p.get('id')} changed {_short(p.get('changes'), 80)}"
    except Exception:  # noqa: BLE001 - descriptions must never break analytics
        pass
    return f"{t} by {a}: {_short(p, 140)}"


def _short(x: Any, n: int) -> str:
    s = json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def event_row(ev, payload: bool = False) -> dict:
    d = {"seq": ev.seq, "t": ev.t, "day": round(ev.t / DAY, 3), "type": ev.type, "actor": ev.actor, "cmd": ev.cmd,
         "vis": list(ev.vis) if isinstance(ev.vis, tuple) else ev.vis, "text": describe_event(ev)}
    if payload:
        d["payload"] = ev.payload
    return d


# ====================================================================== timeline
def entity_aliases(world, entity: str, expand: bool = True) -> set[str]:
    """Identifiers that stand for the entity in event payloads: the id itself and, with `expand`, its accounts,
    hosts (and their service principals, deployments, listings), instances; a person also includes the
    organisations it owns."""
    st = world.state
    eid, kind = resolve_entity(world, entity)
    out = {eid}
    if not expand:
        return out
    owners = [eid]
    if kind == "person":
        owners += owned_orgs(world, eid)
    for o in owners:
        out.add(o)
        out.update(accounts_of(world, o))
        for h in hosts_of(world, o):
            out.update(_host_aliases(world, h))
        if has_table(world, "instances"):
            out.update(st.find_ids("instances", "owner", o))
        if has_table(world, "listings"):
            out.update(st.find_ids("listings", "seller", o))
    if kind == "host":
        out.update(_host_aliases(world, eid))
    if kind == "service":
        h = eid.split(":", 1)[1]
        out.update(_host_aliases(world, h))
    return {x for x in out if x}


def _host_aliases(world, host: str) -> set[str]:
    st = world.state
    out = {host, f"svc:{host}"}
    if has_table(world, "deployments"):
        out.update(st.find_ids("deployments", "host", host))
    if has_table(world, "listings"):
        out.update(st.find_ids("listings", "host", host))
    return out


def _pattern(aliases: set[str]) -> re.Pattern:
    alts = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![\w:-])(?:{alts})(?![\w-])")


def _mentions(ev, pat: re.Pattern, aliases: set[str]) -> bool:
    if ev.actor in aliases:
        return True
    if isinstance(ev.vis, tuple) and any(v in aliases for v in ev.vis):
        return True
    return pat.search(json.dumps(ev.payload, separators=(",", ":"), default=str)) is not None


def timeline(world, entity: str, since: int = 0, *, expand: bool = True, include_noise: bool = False,
             types=None, limit: int | None = None, payload: bool = False) -> list[dict]:
    """Events (oldest first) with seq > `since` that mention `entity` or, with `expand`, one of its aliases.
    Bookkeeping events (counters, scheduling, per-request metering) are left out unless `include_noise`."""
    aliases = entity_aliases(world, entity, expand)
    pat = _pattern(aliases)
    tset = set(types) if types else None
    rows = []
    for ev in world_events(world, since):
        if tset is not None and ev.type not in tset:
            continue
        if tset is None and not include_noise and ev.type in NOISE_TYPES:
            continue
        if ev.type == "consumers.hour":
            continue  # aggregated into explain(); mentions are by slot number
        if _mentions(ev, pat, aliases):
            rows.append(event_row(ev, payload))
    if limit is not None and len(rows) > limit:
        rows = rows[-limit:]
    return rows


# ====================================================================== shared collectors
def _flows(world, accounts: list[str], related: set[str] | None = None, t0: int = 0) -> dict:
    """Per-day inflows / outflows by transaction kind for a set of accounts (from ledger state)."""
    st = world.state
    mine = set(accounts)
    inflow: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    outflow: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_cp: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if not has_table(world, "acct_txns"):
        return {"in": {}, "out": {}, "counterparties": {}}
    for a in sorted(mine):
        for lid in st.find_ids("acct_txns", "acct", a):
            e = st.tables["acct_txns"][lid]
            if e["t"] < t0:
                continue
            tx = st.tables["txns"][e["txn"]]
            others = [x for x, _ in tx["postings"] if x not in mine]
            if not others:
                continue  # internal transfer between the entity's own accounts
            cp = st.tables["accounts"][others[0]]["owner"] if st.has("accounts", others[0]) else "?"
            if cp == "sys":
                cp = st.tables["accounts"][others[0]].get("name") or others[0]
            d = day_of(e["t"])
            if e["amount"] > 0:
                inflow[d][tx["kind"]] += e["amount"]
                by_cp[cp][0] += e["amount"]
            else:
                outflow[d][tx["kind"]] += -e["amount"]
                by_cp[cp][1] += -e["amount"]
    return {"in": {d: dict(v) for d, v in sorted(inflow.items())},
            "out": {d: dict(v) for d, v in sorted(outflow.items())},
            "counterparties": {k: {"in": v[0], "out": v[1]} for k, v in sorted(by_cp.items())}}


def _cash(world, owner: str) -> int:
    st = world.state
    return int(sum(st.tables["accounts"][a]["balance"] for a in accounts_of(world, owner)))


def market_series(world, since: int = 0, hosts: set[str] | None = None) -> dict:
    """From consumers.hour events: per host hourly series of customers, requests, ok, errors, unavailable,
    overload, revenue, and search appearances (organic / sponsored)."""
    smap = slot_map(world)
    slot_host = {s: v["host"] for s, v in smap.items()}
    want = None if hosts is None else {s for s, h in slot_host.items() if h in hosts}
    out: dict[str, dict[str, list]] = {}

    def row(h):
        return out.setdefault(h, {"t": [], "customers": [], "requests": [], "ok": [], "errors": [], "unavailable": [],
                                  "overload": [], "revenue": [], "q": [], "seen": [], "sponsored": [], "price": [],
                                  "rating": []})
    for ev in world_events(world, since, types=("consumers.hour",)):
        p = ev.payload
        seen = defaultdict(int)
        spons = defaultdict(int)
        for _, lst in (p.get("seen") or {}).items():
            for s in lst:
                seen[int(s)] += 1
        for _, lst in (p.get("sponsored") or {}).items():
            for s in lst:
                spons[int(s)] += 1
        public = {int(k): v for k, v in (p.get("public") or {}).items()}
        slots = {int(k): v for k, v in (p.get("slots") or {}).items()}
        for s in sorted(set(public) | set(slots)):
            if want is not None and s not in want:
                continue
            h = slot_host.get(s)
            if h is None:
                continue
            sm = slots.get(s, {})
            r = row(h)
            r["t"].append(int(p.get("t", ev.t)))
            r["customers"].append(int(sm.get("customers", 0)))
            for k in ("requests", "ok", "errors", "unavailable", "overload", "revenue"):
                r[k].append(int(sm.get(k, 0)))
            r["q"].append(float(sm.get("q", 0.0)))
            r["seen"].append(seen.get(s, 0))
            r["sponsored"].append(spons.get(s, 0))
            r["price"].append(public.get(s, {}).get("price"))
            r["rating"].append(public.get(s, {}).get("rating"))
    return out


def daily(series: dict, key: str, how: str = "last") -> dict[int, float]:
    """Aggregate an hourly series to days: last value (levels) or sum (flows)."""
    out: dict[int, float] = {}
    for t, v in zip(series.get("t", []), series.get(key, [])):
        if v is None:
            continue
        d = day_of(t - 1) if t % DAY == 0 else day_of(t)  # the hour ending at midnight belongs to the day before
        out[d] = v if how == "last" else out.get(d, 0) + v
    return out


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _finding(kind: str, text: str, t: int | None = None, severity: int = 1, **evidence) -> dict:
    return {"kind": kind, "t": t, "day": None if t is None else round(t / DAY, 2), "severity": severity,
            "text": text, "evidence": evidence}


# ====================================================================== explain
def explain(world, entity: str, since: int = 0) -> dict:
    """Structured explanation for an organisation, host, account or person (see module docstring).
    Returns {"entity", "kind", "headline", "facts", "findings": [...], "series": {...}, "text"}."""
    eid, kind = resolve_entity(world, entity)
    if kind == "org":
        out = _explain_org(world, eid, since)
    elif kind in ("host", "service", "listing", "deployment"):
        host = _host_of(world, eid, kind)
        out = _explain_host(world, host, since) if host else _unknown(eid, kind)
    elif kind == "account":
        out = _explain_account(world, eid)
    elif kind == "person":
        out = _explain_person(world, eid, since)
    else:
        out = _unknown(eid, kind)
    out["text"] = _render(out)
    return out


def _unknown(eid, kind):
    return {"entity": eid, "kind": kind, "headline": f"No explanation model for {eid} ({kind}).", "facts": {},
            "findings": [], "series": {}}


def _host_of(world, eid, kind):
    st = world.state
    if kind == "host":
        return eid
    if kind == "service":
        return eid.split(":", 1)[1]
    if kind == "listing":
        return st.tables["listings"][eid]["host"]
    if kind == "deployment":
        return st.tables["deployments"][eid]["host"]
    return None


def _org_events(world, org: str, aliases: set[str], since: int):
    """One pass over the log collecting the events relevant to an organisation's fate."""
    st = world.state
    instances = set(st.find_ids("instances", "owner", org)) if has_table(world, "instances") else set()
    providers = {st.tables["instances"][i]["provider"] for i in instances}
    deps = set(st.find_ids("deployments", "owner", org)) if has_table(world, "deployments") else set()
    listings = set(st.find_ids("listings", "seller", org)) if has_table(world, "listings") else set()
    needs = {st.tables["listings"][l]["need"] for l in listings if st.tables["listings"][l]["need"]}
    first_listing = min((st.tables["listings"][l].get("created", 0) for l in listings), default=0)
    got = defaultdict(list)
    for ev in world_events(world, since):
        p = ev.payload if isinstance(ev.payload, dict) else {}
        t = ev.type
        if t == "cloud.instance_status" and p.get("id") in instances:
            got["instance"].append(ev)
        elif t == "cloud.dep_status" and p.get("id") in deps:
            got["deployment"].append(ev)
        elif t == "cloud.outage" and p.get("id") in providers:
            got["outage"].append(ev)
        elif t in ("biz.distress", "biz.status", "biz.fee") and p.get("org") == org:
            got[t].append(ev)
        elif t == "contracts.claim" and p.get("debtor") == org:
            got["claim"].append(ev)
        elif t == "contracts.claim" and p.get("creditor") == org:
            got["claim_in"].append(ev)
        elif t == "contracts.status" and p.get("status") in ("breached", "terminated") and \
                has_table(world, "contracts") and st.has("contracts", p.get("id", "")) and \
                org in st.tables["contracts"][p["id"]]["parties"]:
            got["contract"].append(ev)
        elif t == "market.listing_update" and p.get("id") in listings:
            got["listing"].append(ev)
        elif t == "consumers.offer" and p.get("need") in needs and p.get("id") not in listings and \
                ev.t > first_listing:
            got["rival"].append(ev)
        elif t == "support.open" and p.get("owner") == org:
            got["ticket"].append(ev)
        elif t == "cmd.rejected" and ev.actor in aliases:
            got["rejected"].append(ev)
    return got


def _explain_org(world, org: str, since: int) -> dict:
    st = world.state
    o = st.get("orgs", org) if has_table(world, "orgs") else None
    comp = st.get("companies", org) if has_table(world, "companies") else None
    status = (o or {}).get("status", "unknown")
    aliases = entity_aliases(world, org)
    accts = accounts_of(world, org)
    hosts = hosts_of(world, org)
    fl = _flows(world, accts)
    got = _org_events(world, org, aliases, since)
    ms = market_series(world, since, set(hosts))
    findings: list[dict] = []
    end_t = world.now
    if status in ("bankrupt", "dissolved"):
        st_ev = (got.get("biz.status") or [None])[-1]
        end_t = st_ev.t if st_ev is not None else world.now
    # --- money: revenue and costs by day
    days = sorted(set(fl["in"]) | set(fl["out"]))
    rev = {d: sum(v for k, v in fl["in"].get(d, {}).items() if k in REVENUE_KINDS) for d in days}
    cost = {d: sum(fl["out"].get(d, {}).values()) for d in days}
    cost_kinds: dict[str, int] = defaultdict(int)
    for d in days:
        for k, v in fl["out"].get(d, {}).items():
            cost_kinds[COST_LABELS.get(k, k)] += v
    end_d = day_of(end_t)
    win = [d for d in days if d <= end_d]
    last7 = [rev.get(d, 0) for d in range(end_d - 6, end_d + 1)]
    prev7 = [rev.get(d, 0) for d in range(end_d - 13, end_d - 6)]
    c7 = sum(cost.get(d, 0) for d in range(end_d - 6, end_d + 1))
    r7 = sum(last7)
    if win:
        tr = _trend([rev.get(d, 0) for d in range(min(win), end_d + 1)])
        if sum(prev7) > 0 and r7 < 0.7 * sum(prev7):
            findings.append(_finding("revenue_decline", f"revenue fell from {money(sum(prev7))} to {money(r7)} "
                                     f"over the last two weeks", end_t, 2, trend_per_day=tr))
        elif r7 == 0 and c7 > 0:
            findings.append(_finding("no_revenue", f"no revenue in the last 7 days while spending {money(c7)}",
                                     end_t, 2))
        if c7 > r7 and c7 > 0:
            top = sorted(cost_kinds.items(), key=lambda kv: -kv[1])[:3]
            findings.append(_finding("burn", f"costs exceeded revenue over the last 7 days ({money(c7)} vs "
                                     f"{money(r7)}); biggest cost lines overall: " +
                                     ", ".join(f"{k} {money(v)}" for k, v in top), end_t, 2 if r7 < 0.5 * c7 else 1))
    # --- obligations
    for ev in got.get("claim", []):
        p = ev.payload
        cl = st.get("claims", p["id"]) if has_table(world, "claims") else None
        stt = cl["status"] if cl else "?"
        findings.append(_finding("claim", f"claim {p['id']}: owes {p['creditor']} {money(p['amount'])} for "
                                 f"{p.get('source')} (now {stt})", ev.t, 3 if stt in ("open", "written_off") else 1,
                                 seq=ev.seq))
    for ev in got.get("contract", []):
        findings.append(_finding("contract", f"contract {ev.payload['id']} {ev.payload['status']}", ev.t, 2,
                                 seq=ev.seq))
    for ev in got.get("biz.fee", []):
        if ev.payload.get("unpaid"):
            findings.append(_finding("unpaid_fee", "could not pay the annual registration fee", ev.t, 2, seq=ev.seq))
    for ev in got.get("biz.distress", []):
        if ev.payload.get("since") is not None:
            findings.append(_finding("distress", f"declared in distress owing {money(ev.payload.get('owed'))}",
                                     ev.t, 3, seq=ev.seq))
    if has_table(world, "invoices"):
        for iid in st.find_ids("invoices", "payer", org):
            inv = st.tables["invoices"][iid]
            if inv.get("status") not in ("paid", "cancelled"):
                findings.append(_finding("unpaid_invoice", f"invoice {iid} of {money(inv.get('amount'))} is "
                                         f"{inv.get('status')}", inv.get("t"), 2))
    # --- infrastructure
    sus = [ev for ev in got.get("instance", []) if ev.payload.get("status") == "suspended"]
    term = [ev for ev in got.get("instance", []) if ev.payload.get("status") == "terminated"]
    for ev in sus:
        findings.append(_finding("instance_suspended", f"instance {ev.payload['id']} suspended: "
                                 f"{ev.payload.get('reason')}", ev.t, 3, seq=ev.seq))
    for ev in term:
        findings.append(_finding("instance_terminated", f"instance {ev.payload['id']} terminated: "
                                 f"{ev.payload.get('reason')}", ev.t, 2, seq=ev.seq))
    for ev in got.get("deployment", []):
        if ev.payload.get("status") in ("stopped", "failed"):
            findings.append(_finding("service_down", f"deployment {ev.payload['id']} {ev.payload['status']}: "
                                     f"{ev.payload.get('reason')}", ev.t, 2, seq=ev.seq))
    for ev in got.get("outage", []):
        p = ev.payload
        findings.append(_finding("outage", f"provider {p['id']} hosting its instances had a "
                                 f"{(p['until'] - p['start']) // HOUR}h outage", p["start"], 2, seq=ev.seq))
    # --- customers and service quality
    for h, s in sorted(ms.items()):
        cust = daily(s, "customers")
        if cust:
            peak_d = max(cust, key=lambda d: (cust[d], -d))
            last_d = max(d for d in cust if d <= end_d) if any(d <= end_d for d in cust) else max(cust)
            peak, now = cust[peak_d], cust[last_d]
            if peak >= 10 and now < 0.6 * peak:
                findings.append(_finding("lost_customers", f"{h} lost customers: {int(peak)} on day {peak_d} -> "
                                         f"{int(now)} on day {last_d}", last_d * DAY, 3 if now < 0.3 * peak else 2,
                                         host=h))
        tot = sum(s["ok"]) + sum(s["errors"]) + sum(s["unavailable"]) + sum(s["overload"])
        bad = tot - sum(s["ok"])
        if tot and bad / tot > 0.05:
            findings.append(_finding("failures", f"{h}: {bad / tot:.0%} of consumer requests failed "
                                     f"(errors {sum(s['errors'])}, unavailable {sum(s['unavailable'])}, "
                                     f"overload {sum(s['overload'])})", None, 2 if bad / tot > 0.2 else 1, host=h))
    rivals = got.get("rival", [])
    if rivals:
        findings.append(_finding("rivals", f"{len(rivals)} rival offering(s) entered its markets", rivals[0].t, 1,
                                 listings=[e.payload["id"] for e in rivals][:10]))
    tickets = got.get("ticket", [])
    if tickets:
        findings.append(_finding("tickets", f"{len(tickets)} support tickets opened against its services",
                                 tickets[0].t, 1))
    findings.sort(key=lambda f: (f["t"] if f["t"] is not None else 10**12, -f["severity"]))
    # --- headline
    total_rev = sum(rev.values())
    if status in ("bankrupt", "dissolved"):
        causes = sorted((f for f in findings if f["t"] is None or f["t"] <= end_t),
                        key=lambda f: (-f["severity"], f["t"] or 0))[:4]
        headline = (f"{org} {'went bankrupt' if status == 'bankrupt' else 'was dissolved'} on day "
                    f"{end_t / DAY:.1f}. Preceding causes: " + ("; ".join(f["text"] for f in causes) or "none recorded"))
    else:
        dist = comp and comp.get("distress_since") is not None
        headline = (f"{org} is {status}{' (IN DISTRESS)' if dist else ''}: cash {money(_cash(world, org))}, revenue "
                    f"{money(r7)} in the last 7 days ({money(total_rev)} total)")
        cust_now = {h: (list(v.values())[-1] if v else 0) for h, v in
                    ((h, daily(s, "customers")) for h, s in sorted(ms.items()))}
        if cust_now:
            headline += "; customers " + ", ".join(f"{h} {int(c)}" for h, c in cust_now.items())
        risks = [f for f in findings if f["severity"] >= 2]
        if risks:
            headline += "; risks: " + "; ".join(f["text"] for f in risks[-3:])
    facts = {"status": status, "company": dict(comp) if comp else None, "cash": _cash(world, org),
             "accounts": accts, "hosts": hosts, "revenue_total": total_rev, "revenue_7d": r7, "costs_7d": c7,
             "cost_breakdown": dict(cost_kinds), "members": [dict(m) for m in st.find("members", "org", org)]
             if has_table(world, "members") else [],
             "counterparties": fl["counterparties"], "rejected_actions": len(got.get("rejected", []))}
    series = {"revenue_by_day": rev, "costs_by_day": cost,
              "customers_by_day": {h: daily(s, "customers") for h, s in ms.items()}}
    return {"entity": org, "kind": "org", "headline": headline, "facts": facts, "findings": findings,
            "series": series}


def _explain_host(world, host: str, since: int) -> dict:
    st = world.state
    owner = st.tables["hosts"][host]["owner"] if has_table(world, "hosts") and st.has("hosts", host) else None
    smap = slot_map(world)
    my_slots = [s for s, v in smap.items() if v["host"] == host]
    need = smap[my_slots[0]]["need"] if my_slots else None
    rival_hosts = {v["host"] for v in smap.values() if v["need"] == need and v["host"] != host and v["host"]}
    ms = market_series(world, since, {host} | rival_hosts)
    s = ms.get(host, {"t": []})
    findings: list[dict] = []
    cust = daily(s, "customers") if s["t"] else {}
    listing_ids = st.find_ids("listings", "host", host) if has_table(world, "listings") else []
    listings = [dict(st.tables["listings"][l], id=l) for l in listing_ids]
    ratings = []
    if has_table(world, "ratings"):
        for l in listing_ids:
            ratings += [dict(r) for r in st.find("ratings", "listing", l)]
    ratings.sort(key=lambda r: r.get("t", 0))
    rating_avg = (sum(r["stars"] for r in ratings) / len(ratings)) if ratings else None
    # advertising
    camps = []
    if has_table(world, "ads_campaigns"):
        for cid, c in st.scan("ads_campaigns"):
            land = c.get("landing") or {}
            if host in (land.get("hosts") or []) or land.get("ref") in listing_ids:
                camps.append(dict(c, id=cid))
    impressions = sum(c.get("impressions", 0) for c in camps)
    # current shares among rivals
    shares = {}
    cons = world.modules.get("consumers")
    if cons is not None and need:
        shares = cons.market_shares(world, need)
    total = sum(shares.values())
    mine = shares.get(host, 0)
    rank = 1 + sum(1 for h, v in shares.items() if v > mine)
    price = listings[-1]["price"] if listings else None
    rival_prices = {}
    for v in smap.values():
        if v["host"] in rival_hosts and v["listing"] and st.has("listings", v["listing"]):
            l = st.tables["listings"][v["listing"]]
            if l["status"] == "active":
                rival_prices[v["host"]] = l["price"]
    seen_total = sum(s.get("seen", [])) if s["t"] else 0
    spons_total = sum(s.get("sponsored", [])) if s["t"] else 0
    rival_seen = {h: sum(ms[h]["seen"]) for h in rival_hosts if h in ms}
    tot_req = sum(s.get("requests", [])) if s["t"] else 0
    ok = sum(s.get("ok", [])) if s["t"] else 0
    fail = (tot_req - ok) / tot_req if tot_req else 0.0
    # drivers of popularity relative to rivals
    if total:
        findings.append(_finding("share", f"{host} has {mine} of {total} customers for {need} ({mine / total:.0%}, "
                                 f"rank {rank} of {len(shares)})", world.now, 1))
    if price is not None and rival_prices:
        cheaper = sum(1 for p in rival_prices.values() if p < price)
        findings.append(_finding("price", f"price {price}c vs rivals {dict(sorted(rival_prices.items()))} "
                                 f"({'cheapest' if cheaper == 0 else f'{cheaper} rival(s) cheaper'})", None,
                                 1 if cheaper == 0 else 2))
    if ratings:
        findings.append(_finding("ratings", f"{len(ratings)} verified ratings, average {rating_avg:.2f}/5",
                                 ratings[-1].get("t"), 1 if rating_avg >= 3.5 else 2))
    if s["t"]:
        findings.append(_finding("search", f"appeared in organic search results {seen_total} times and in "
                                 f"sponsored results {spons_total} times (rivals organic: "
                                 f"{dict(sorted(rival_seen.items()))})", None, 1))
    if camps:
        findings.append(_finding("ads", f"{len(camps)} ad campaign(s), {impressions} impressions, spent "
                                 f"{money(sum(c.get('total_spent', 0) for c in camps))}", camps[0].get("created"), 1))
    if tot_req:
        findings.append(_finding("reliability", f"{fail:.1%} of {tot_req} consumer requests failed", None,
                                 1 if fail < 0.05 else (2 if fail < 0.2 else 3)))
    # churn episodes: daily drops and what preceded them
    churn = []
    ds = sorted(cust)
    for a, b in zip(ds, ds[1:]):
        drop = cust[a] - cust[b]
        if drop >= max(5, 0.1 * cust[a]):
            ctx_ = _churn_context(world, host, s, b, owner, listing_ids, rival_hosts, smap)
            churn.append({"day": b, "from": cust[a], "to": cust[b], **ctx_})
            findings.append(_finding("customers_left", f"day {b}: customers {int(cust[a])} -> {int(cust[b])}; "
                                     f"before: {ctx_['why']}", b * DAY, 2))
    findings.sort(key=lambda f: (f["t"] if f["t"] is not None else -1, -f["severity"]))
    if cust:
        peak_d = max(cust, key=lambda d: (cust[d], -d))
        headline = (f"{host}.crucible ({need}, operated by {owner}): {mine} customers now, peak {int(cust[peak_d])} "
                    f"on day {peak_d}")
    else:
        headline = f"{host}.crucible (operated by {owner}) has no consumer-market history"
    if total:
        headline += f"; market share {mine / total:.0%} (rank {rank}/{len(shares)})"
    if churn:
        headline += f"; {len(churn)} customer-loss episode(s), latest day {churn[-1]['day']}: {churn[-1]['why']}"
    facts = {"owner": owner, "need": need, "slots": my_slots, "listings": listings, "price": price,
             "rival_prices": rival_prices, "shares": shares, "rating_avg": rating_avg, "n_ratings": len(ratings),
             "ad_campaigns": [c["id"] for c in camps], "impressions": impressions, "search_organic": seen_total,
             "search_sponsored": spons_total, "requests": tot_req, "failure_rate": fail, "churn": churn}
    series = {"customers_by_day": cust, "rivals_customers_by_day": {h: daily(ms[h], "customers") for h in
                                                                     sorted(rival_hosts) if h in ms},
              "seen_by_day": daily(s, "seen", "sum") if s["t"] else {},
              "failures_by_day": {d: v for d, v in _fail_by_day(s).items()},
              "ratings": [(r.get("t"), r["stars"]) for r in ratings]}
    return {"entity": host, "kind": "host", "headline": headline, "facts": facts, "findings": findings,
            "series": series}


def _fail_by_day(s) -> dict[int, float]:
    req = daily(s, "requests", "sum")
    ok = daily(s, "ok", "sum")
    return {d: round((req[d] - ok.get(d, 0)) / req[d], 4) for d in req if req[d]}


def _churn_context(world, host, s, day, owner, listing_ids, rival_hosts, smap) -> dict:
    """What happened in the 24 hours before customers were lost on `day`."""
    t1 = day * DAY
    t0 = t1 - 2 * DAY
    idx = [i for i, t in enumerate(s["t"]) if t0 < t <= t1]
    req = sum(s["requests"][i] for i in idx)
    ok = sum(s["ok"][i] for i in idx)
    err = sum(s["errors"][i] for i in idx)
    una = sum(s["unavailable"][i] for i in idx)
    ovl = sum(s["overload"][i] for i in idx)
    why = []
    if req:
        fr = (req - ok) / req
        if fr > 0.05:
            why.append(f"{fr:.0%} of requests failed (errors {err}, unavailable {una}, overload {ovl})")
    prices = [p for i, p in zip(range(len(s["t"])), s["price"]) if i in set(idx) and p is not None]
    if prices and prices[0] != prices[-1]:
        why.append(f"price changed {prices[0]}c -> {prices[-1]}c")
    rival_new = [v["host"] for v in smap.values() if v["host"] in rival_hosts and t0 - DAY < v["since"] <= t1]
    if rival_new:
        why.append(f"new rival(s) {rival_new}")
    st = world.state
    if has_table(world, "instances") and owner:
        provs = {st.tables["instances"][i]["provider"] for i in st.find_ids("instances", "owner", owner)}
        for pv in sorted(provs):
            for start in st.tables["providers"][pv]["outages"] if st.has("providers", pv) else []:
                if t0 <= start <= t1:
                    why.append(f"outage at provider {pv}")
    return {"requests": req, "ok": ok, "errors": err, "unavailable": una, "overload": ovl,
            "why": "; ".join(why) or "no service failure, price change, outage or rival entry recorded "
                                     "(normal turnover or rivals' improvements)"}


def _explain_account(world, acct: str) -> dict:
    st = world.state
    a = st.tables["accounts"][acct]
    fl = _flows(world, [acct])
    cps = sorted(fl["counterparties"].items(), key=lambda kv: -(kv[1]["in"] + kv[1]["out"]))
    findings = [_finding("counterparty", f"{cp}: in {money(v['in'])}, out {money(v['out'])}", None, 1)
                for cp, v in cps[:10]]
    tin = sum(v["in"] for _, v in cps)
    tout = sum(v["out"] for _, v in cps)
    headline = (f"account {acct} of {a['owner']} ({a['kind']}, {a['status']}): balance {money(a['balance'])}; "
                f"received {money(tin)}, paid {money(tout)}")
    return {"entity": acct, "kind": "account", "headline": headline, "facts": {"account": dict(a), "flows": fl},
            "findings": findings, "series": {"in_by_day": {d: sum(v.values()) for d, v in fl["in"].items()},
                                             "out_by_day": {d: sum(v.values()) for d, v in fl["out"].items()}}}


def _explain_person(world, pid: str, since: int) -> dict:
    st = world.state
    p = st.get("principals", pid)
    person = st.get("persons", pid) if has_table(world, "persons") else None
    orgs = owned_orgs(world, pid, roles=("owner", "admin", "member"))
    rej: dict[str, int] = defaultdict(int)
    tools: dict[str, int] = defaultdict(int)
    for ev in world_events(world, since, types=("cmd", "cmd.rejected")):
        if ev.type == "cmd" and ev.actor == pid and ev.payload.get("n") is not None:
            tools[str(ev.payload.get("tool"))] += 1
        elif ev.type == "cmd.rejected" and (ev.payload.get("actor") == pid):
            rej[(ev.payload.get("error") or {}).get("code", "?")] += 1
    findings = [_finding("orgs", f"member of {orgs}", None, 1)] if orgs else []
    if rej:
        findings.append(_finding("rejected", f"rejected actions by error code: {dict(rej)}", None,
                                 2 if sum(rej.values()) > 10 else 1))
    sub = [_explain_org(world, o, since)["headline"] for o in owned_orgs(world, pid)]
    for h in sub:
        findings.append(_finding("org", h, None, 1))
    headline = (f"{pid} ({(person or {}).get('kind', p['kind'] if p else '?')}): {p.get('actions', 0) if p else 0} "
                f"actions, cash {money(_cash(world, pid))}, organisations {orgs}")
    return {"entity": pid, "kind": "person", "headline": headline,
            "facts": {"principal": dict(p) if p else None, "orgs": orgs, "tools": dict(tools), "rejected": dict(rej)},
            "findings": findings, "series": {}}


def _render(out: dict) -> str:
    lines = [out["headline"]]
    if out.get("findings"):
        lines.append("Findings (chronological):")
        for f in out["findings"]:
            when = f"day {f['day']:>6}" if f["day"] is not None else "         "
            lines.append(f"  [{'!' * f['severity']:<3}] {when}  {f['kind']}: {f['text']}")
    return "\n".join(lines)


# ====================================================================== causal chain
_STARTERS = ("proc", "cmd", "time")


def causal_chain(world, seq: int, max_depth: int = 12, max_links: int = 60) -> dict:
    """Follow event `seq` backwards through command ids, scheduling and references.

    Links (newest first): {"seq", "t", "type", "actor", "relation", "text", "depth"}. Relations:
      * ``transaction``   - the command (actor, tool, args) or scheduled process run that emitted the event;
      * ``scheduled_by``  - the event that scheduled that process (periodic jobs stop here: "periodic");
      * ``references``    - the event that created an entity referenced in the payload or command arguments.
    """
    events = list(world_events(world, 0, seq))
    by_seq = {e.seq: e for e in events}
    if seq not in by_seq:
        return {"seq": seq, "links": [], "text": f"event {seq} is not available"}
    created: dict[str, int] = {}
    for e in events:
        p = e.payload if isinstance(e.payload, dict) else {}
        i = p.get("id")
        if isinstance(i, str) and i not in created and e.type not in ("sched.del", "proc", "idem", "know"):
            created[i] = e.seq
    links: list[dict] = []
    seen: set[int] = set()

    def add(ev, relation, depth, note=""):
        links.append({"seq": ev.seq, "t": ev.t, "type": ev.type, "actor": ev.actor, "relation": relation,
                      "depth": depth, "text": describe_event(ev) + (f" [{note}]" if note else "")})

    def tx_start(ev):
        if ev.type in _STARTERS or (ev.type == "note" and ev.payload.get("what") == "genesis"):
            return ev
        s = ev.seq - 1
        while s in by_seq:
            e = by_seq[s]
            if ev.cmd is not None:
                if e.type in ("cmd", "cmd.rejected") and e.payload.get("n") == ev.cmd:
                    return e
                if e.cmd != ev.cmd:
                    return None
            elif e.type in _STARTERS or (e.type == "note" and e.payload.get("what") == "genesis"):
                return e
            s -= 1
        return None

    def walk(ev, depth):
        if depth > max_depth or len(links) >= max_links or ev.seq in seen:
            return
        seen.add(ev.seq)
        start = tx_start(ev)
        if start is not None and start.seq != ev.seq and start.seq not in seen:
            add(start, "transaction", depth)
            seen.add(start.seq)
        root = start or ev
        if root.type == "proc":
            jid = root.payload.get("id")
            sch = None
            for s in range(root.seq - 1, 0, -1):
                e = by_seq.get(s)
                if e is not None and e.type == "sched.add" and e.payload.get("id") == jid:
                    sch = e
                    break
            if sch is not None and sch.seq not in seen:
                if sch.payload.get("period"):
                    add(sch, "scheduled_by", depth, f"periodic every {sch.payload['period']}s")
                    seen.add(sch.seq)
                else:
                    add(sch, "scheduled_by", depth)
                    walk(sch, depth + 1)
        refs = set()
        for src in (ev.payload, root.payload.get("args") if root.type == "cmd" else None):
            _collect_ids(src, refs)
        own = ev.payload.get("id") if isinstance(ev.payload, dict) else None
        for r in sorted(refs):
            if r == own:
                continue
            cs = created.get(r)
            if cs is None or cs >= ev.seq or cs in seen:
                continue
            add(by_seq[cs], "references", depth, f"created {r}")
            walk(by_seq[cs], depth + 1)

    target = by_seq[seq]
    walk(target, 0)
    lines = [f"#{target.seq} day {target.t / DAY:.2f} {describe_event(target)}"]
    for l in links:
        lines.append(f"{'  ' * (l['depth'] + 1)}<- {l['relation']}: #{l['seq']} day {l['t'] / DAY:.2f} {l['text']}")
    return {"seq": seq, "event": event_row(target, True), "links": links, "text": "\n".join(lines)}


def _collect_ids(x, out: set, depth: int = 0):
    if depth > 4 or x is None:
        return
    if isinstance(x, str):
        if _ID_RE.match(x):
            out.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            _collect_ids(v, out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for v in x[:50]:
            _collect_ids(v, out, depth + 1)
