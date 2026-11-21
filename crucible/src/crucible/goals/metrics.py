"""Metrics over authoritative world state, and a small typed predicate language over them.

Used by (a) contract deliverable clauses inside the world, (b) goal contracts and the independent evaluator outside
it. Metrics read the kernel state (and, for windows, the ledger postings in state); they never trust what an agent
says. Modules can register more metrics with `@metric("name")`.

Predicate JSON:
    {"metric": "<name>", "args": {...}, "op": ">=", "value": 100000}
    {"all": [p, ...]} | {"any": [p, ...]} | {"not": p}
"""
from __future__ import annotations

import operator
from typing import Any, Callable

from ..core.errors import InvalidAction

METRICS: dict[str, Callable[..., float]] = {}
METRIC_DOCS: dict[str, str] = {}
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq, "!=": operator.ne}


def metric(name: str, doc: str = ""):
    def deco(fn):
        METRICS[name] = fn
        METRIC_DOCS[name] = doc or (fn.__doc__ or "").strip()
        return fn
    return deco


def evaluate_metric(world, name: str, args: dict | None = None) -> float:
    fn = METRICS.get(name)
    if fn is None:
        raise InvalidAction(f"unknown metric {name!r}")
    return float(fn(world, **(args or {})))


def check_predicate(p: Any, depth: int = 0) -> list[str]:
    if depth > 6:
        return ["predicate nested too deeply"]
    if not isinstance(p, dict):
        return ["predicate must be an object"]
    if "all" in p or "any" in p:
        xs = p.get("all", p.get("any"))
        if not isinstance(xs, list) or not xs or len(xs) > 20:
            return ["all/any needs 1-20 predicates"]
        return [e for x in xs for e in check_predicate(x, depth + 1)]
    if "not" in p:
        return check_predicate(p["not"], depth + 1)
    errs = []
    if p.get("metric") not in METRICS:
        errs.append(f"unknown metric {p.get('metric')!r}")
    if p.get("op") not in OPS:
        errs.append(f"bad op {p.get('op')!r}")
    if not isinstance(p.get("value"), (int, float)):
        errs.append("value must be a number")
    if not isinstance(p.get("args", {}), dict):
        errs.append("args must be an object")
    return errs


def evaluate_predicate(world, p: dict) -> bool:
    if "all" in p:
        return all(evaluate_predicate(world, x) for x in p["all"])
    if "any" in p:
        return any(evaluate_predicate(world, x) for x in p["any"])
    if "not" in p:
        return not evaluate_predicate(world, p["not"])
    v = evaluate_metric(world, p["metric"], p.get("args"))
    return bool(OPS[p["op"]](v, p["value"]))


# ============================================================================ relatedness (anti self-dealing)
def related_parties(world, principal: str) -> set[str]:
    """Principals whose payments to `principal` are not independent revenue: itself, organisations it belongs to,
    members of it (if an organisation), organisations sharing a member with it, and service principals they
    operate. Conservative by design: any shared membership makes parties related."""
    st = world.state
    out = {principal}
    persons: set[str] = set()
    orgs: set[str] = set()
    if principal.startswith("p:"):
        persons.add(principal)
    elif principal.startswith("o:"):
        orgs.add(principal)
        persons.update(m["person"] for m in st.find("members", "org", principal))
    for p in list(persons):
        orgs.update(m["org"] for m in st.find("members", "person", p))
    for o in list(orgs):
        persons.update(m["person"] for m in st.find("members", "org", o))
    out |= persons | orgs
    for pid, rec in st.scan("principals"):
        if rec["kind"] == "service" and rec["info"].get("operator") in out:
            out.add(pid)
    return out


def accounts_of(world, principals: set[str]) -> set[str]:
    st = world.state
    return {a for p in principals for a in st.find_ids("accounts", "owner", p)}


def counterparty_flows(world, principal: str, window_s: int) -> dict[str, list[int]]:
    """{counterparty_owner: [inflow, outflow]} over the window, for the principal's own accounts."""
    st = world.state
    t0 = world.now - window_s
    mine = set(st.find_ids("accounts", "owner", principal))
    flows: dict[str, list[int]] = {}
    for acct in sorted(mine):
        for lid in st.find_ids("acct_txns", "acct", acct):
            e = st.tables["acct_txns"][lid]
            if e["t"] <= t0:
                continue
            tx = st.tables["txns"][e["txn"]]
            if tx["kind"] in ("escrow",):
                continue
            others = [(a, d) for a, d in tx["postings"] if a not in mine]
            for a, d in others:
                owner = st.tables["accounts"][a]["owner"]
                if a == "acct_households":
                    owner = "households"
                elif a == "acct_escrow":
                    owner = "escrow"
                f = flows.setdefault(owner, [0, 0])
                if d < 0:      # money left the counterparty -> came to us
                    f[0] += -d
                else:
                    f[1] += d
    return flows


@metric("independent_revenue", "Net inflow (cents) over a window from counterparties unrelated to the principal "
        "(households count as independent; circular flows net out per counterparty; escrow excluded until released).")
def m_independent_revenue(world, principal: str, window_s: int = 30 * 86400) -> float:
    rel = related_parties(world, principal)
    total = 0
    for cp, (inflow, outflow) in counterparty_flows(world, principal, window_s).items():
        if cp in rel or cp in ("sys", "escrow"):
            continue
        total += max(0, inflow - outflow)
    return float(total)


@metric("cash", "Sum of balances of the principal's open accounts (cents).")
def m_cash(world, principal: str) -> float:
    st = world.state
    return float(sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", principal)
                     if st.tables["accounts"][a]["status"] == "open"))


@metric("independent_members", "Members of an organisation who are independent simulated persons (not "
        "agent-controlled), have been members for at least min_tenure_s, and whose employment payments are not in breach.")
def m_independent_members(world, org: str, min_tenure_s: int = 0) -> float:
    st = world.state
    n = 0
    for m in st.find("members", "org", org):
        p = st.get("persons", m["person"])
        if p is None or p["kind"] != "npc" or world.now - m["since"] < min_tenure_s:
            continue
        ok = True
        if "contracts" in st.tables:
            for c in st.find("contracts", "employee", m["person"]):
                if c["employer"] == org and c["status"] == "breached":
                    ok = False
        n += ok
    return float(n)


@metric("service_requests_ok", "Successful requests served by a host over a window.")
def m_requests_ok(world, host: str, window_s: int = 7 * 86400) -> float:
    st = world.state
    if "svc_days" not in st.tables:
        return 0.0
    d1 = world.now // 86400
    d0 = (world.now - window_s) // 86400
    return float(sum(st.tables["svc_days"][k]["ok"] for k in st.find_ids("svc_days", "host", host)
                     if d0 < int(k.split("|")[1]) <= d1))


@metric("service_availability", "Fraction of requests to a host over a window that were served (not unavailable, "
        "overloaded or errors); 0 if no requests.")
def m_availability(world, host: str, window_s: int = 7 * 86400) -> float:
    st = world.state
    if "svc_days" not in st.tables:
        return 0.0
    d1 = world.now // 86400
    d0 = (world.now - window_s) // 86400
    ok = tot = 0
    for k in st.find_ids("svc_days", "host", host):
        if d0 < int(k.split("|")[1]) <= d1:
            r = st.tables["svc_days"][k]
            ok += r["ok"]
            tot += r["ok"] + r["errors"] + r["unavailable"] + r["overload"]
    return float(ok / tot) if tot else 0.0


@metric("repo_commits", "Number of commits in a repository.")
def m_repo_commits(world, repo: str) -> float:
    r = world.state.get("repos", repo) if "repos" in world.state.tables else None
    return float(r["commits"]) if r else 0.0


@metric("org_status_active", "1 if the organisation is active (not bankrupt or dissolved).")
def m_org_active(world, org: str) -> float:
    o = world.state.get("orgs", org)
    return 1.0 if o and o["status"] == "active" else 0.0


@metric("quality_margin_vs_incumbents", "EVALUATOR-ONLY. Mean verified quality of the principal's best live service "
        "for a need minus the best incumbent's, on a hidden battery of requests executed in an isolated fork. "
        "-1 if the principal has no live listed service for the need.")
def m_quality_margin(world, org: str, need: str, n: int = 6, seed: int = 0) -> float:
    import numpy as np
    from ..core.canonical import digest as _dg
    from ..economy.needs import NEEDS
    if world.state.in_tx:
        raise InvalidAction("evaluator-only metric (not available inside world transactions)")
    st = world.state
    if "listings" not in st.tables or need not in NEEDS:
        return -1.0
    mine, inc = [], []
    for lid, l in st.scan("listings"):
        if l["kind"] != "api" or l["need"] != need or l["status"] != "active":
            continue
        if l["seller"] == org:
            mine.append((l["host"], l["endpoint"]))
        elif "npc_firms" in st.tables and st.has("npc_firms", l["seller"]):
            inc.append((l["host"], l["endpoint"]))
    if not mine:
        return -1.0
    nd = NEEDS[need]
    rng = np.random.default_rng([int(seed), int(_dg(["battery", need, world.spec.seed], 4), 16)])
    cases = [nd.generate(rng, 0.7) for _ in range(int(n))]
    f = world.fork(f"eval-{world.seq}-{_dg([org, need], 3)}")
    try:
        cloud = f.modules["cloud"]

        def quality(host, ep):
            out = {"q": 0.0}

            def run(ctx):
                qs = []
                dep = cloud.live_deployment(f.state, host)
                if dep is None or cloud.availability(f.state, dep):
                    out["q"] = 0.0
                    return
                did = f.state.get("hosts", host)["deployment"]
                for inp, hidden in cases:
                    r = cloud.execute(ctx, did, ep, inp, "evaluator", nd.work_scale,
                                      post=lambda b, inp=inp, hidden=hidden: {"q": float(nd.score(inp, hidden, b))})
                    qs.append(float(r["body"]["q"]) if r["status"] == 200 and isinstance(r.get("body"), dict) else 0.0)
                out["q"] = sum(qs) / len(qs)
            f._run_system("evaluator.battery", run)
            return out["q"]
        best_mine = max(quality(h, e) for h, e in mine)
        best_inc = max((quality(h, e) for h, e in inc), default=0.0)
        return float(best_mine - best_inc)
    finally:
        f.close()


# ============================================================================ open-world outcome metrics
@metric("revenue_from", "Net inflow (cents) to the principal's accounts from one payer's own accounts over a window "
        "(0 if the payer is a related party).")
def m_revenue_from(world, principal: str, payer: str, window_s: int = 14 * 86400) -> float:
    if payer in related_parties(world, principal):
        return 0.0
    inflow, outflow = counterparty_flows(world, principal, window_s).get(payer, [0, 0])
    return float(max(0, inflow - outflow))


@metric("loss_events", "Number of operating loss events of one kind (e.g. capacity_overload) an organisation "
        "recorded over the last complete days of a window.")
def m_loss_events(world, org: str, kind: str, window_s: int = 7 * 86400) -> float:
    st = world.state
    if "ops_days" not in st.tables:
        return 0.0
    today = world.now // 86400
    n = 0
    for d in range(today - max(1, window_s // 86400), today):
        r = st.tables["ops_days"].get(f"{org}|{d}")
        if r:
            n += int(r["loss_n"].get(kind, 0))
    return float(n)


@metric("capability_adopters", "Independent organisations (unrelated to the principal) that have kept a capability "
        "published by the principal (or a party related to it) active - adopted after their own trial - for at least "
        "min_active_s.")
def m_capability_adopters(world, principal: str, min_active_s: int = 0) -> float:
    st = world.state
    if "cap_installs" not in st.tables:
        return 0.0
    rel = related_parties(world, principal)
    mine = {k for k, c in st.scan("capabilities") if c["provider"] in rel}
    since: dict[str, int] = {}
    for _, i in st.scan("cap_installs"):
        if i["capability"] in mine and i["status"] == "active" and i["installer"] not in rel:
            t = int(i.get("status_t") or i.get("t") or 0)
            if world.now - t >= min_active_s:
                since[i["installer"]] = min(t, since.get(i["installer"], t))
    return float(len(since))


@metric("search_best_rank", "1-based rank of the best organic search result owned by the principal (or a related "
        "party) for a query, among pages on the synthetic internet; 99 when none is in the first 20.")
def m_search_best_rank(world, principal: str, q: str, kind: str = "page") -> float:
    from ..services.search import search_module
    mod = search_module(world)
    if mod is None:
        return 99.0
    rel = related_parties(world, principal)
    _, organic, _ = mod.search(world, q, kind, 20)
    for k, r in enumerate(organic):
        if r["owner"] in rel:
            return float(k + 1)
    return 99.0


@metric("search_visibility", "1 / search_best_rank (0 when the principal has no page in the first 20 results).")
def m_search_visibility(world, principal: str, q: str, kind: str = "page") -> float:
    r = m_search_best_rank(world, principal, q, kind)
    return 0.0 if r >= 99 else 1.0 / r
