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
# Whose data a metric reads: {argument name: "principal" | "host" | "repo"}. A metric without an entry cannot be
# used in an agreement between participants (contract deliverables): it is evaluator-only, or its author has not
# declared a subject. Register with `@metric(name, doc, subject={...})`.
METRIC_SUBJECTS: dict[str, dict[str, str]] = {}
# Metrics that describe one instant (a balance, a rank) rather than a window of time. A goal that must be
# *sustained* on such a metric is checked at every harness step between daily checks (goals/contracts.py).
INSTANT_METRICS: set[str] = set()


def metric(name: str, doc: str = "", subject: dict | None = None, instant: bool = False):
    def deco(fn):
        METRICS[name] = fn
        METRIC_DOCS[name] = doc or (fn.__doc__ or "").strip()
        if subject is not None:
            METRIC_SUBJECTS[name] = dict(subject)
        if instant:
            INSTANT_METRICS.add(name)
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


def predicate_leaves(p: Any) -> list[dict]:
    """The metric comparisons inside a (validated) predicate."""
    if not isinstance(p, dict):
        return []
    if "all" in p or "any" in p:
        return [lf for x in (p.get("all") or []) + (p.get("any") or []) for lf in predicate_leaves(x)]
    if "not" in p:
        return predicate_leaves(p["not"])
    return [p]


def uses_instant_metric(p: Any) -> bool:
    return any(lf.get("metric") in INSTANT_METRICS for lf in predicate_leaves(p))


def check_predicate_scope(world, p: Any, parties) -> list[str]:
    """Errors if a predicate reads anything that is not the business of `parties`: every metric must declare whose
    data it reads, and that principal / host / repository must be a party or belong to one (an organisation a
    party owns or administers, a host or repository of a party or of such an organisation). This is the rule a
    contract deliverable must satisfy - otherwise any metric of anyone could be read by signing a contract with
    oneself and watching which deliverables hold."""
    from ..core import evaluator_only as EO
    st = world.state
    mine: set[str] = set()
    for q in parties:
        mine |= world.principals_of(q)
    errs = EO.metric_errors(world, predicate_leaves(p))    # 5.0 (D-19): evaluator-only metrics are never readable
    for lf in predicate_leaves(p):
        name = lf.get("metric")
        subj = METRIC_SUBJECTS.get(name)
        if subj is None:
            errs.append(f"metric {name!r} cannot be used in an agreement between participants")
            continue
        args = lf.get("args") or {}
        for arg, kind in sorted(subj.items()):
            v = args.get(arg)
            if not isinstance(v, str):
                errs.append(f"{name}: argument {arg!r} must name a party")
                continue
            owner = v
            if kind == "host":
                h = st.get("hosts", v) if "hosts" in st.tables else None
                owner = h["owner"] if h else None
            elif kind == "repo":
                r = st.get("repos", v) if "repos" in st.tables else None
                owner = r["owner"] if r else None
            if owner not in mine:
                errs.append(f"{name}: {v!r} does not belong to a party of this agreement")
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
def _circle(st, principal: str) -> tuple[set[str], set[str]]:
    """(persons, organisations) tied to a principal by membership or founding."""
    persons: set[str] = set()
    orgs: set[str] = set()
    if principal.startswith("p:"):
        persons.add(principal)
    elif principal.startswith("o:"):
        orgs.add(principal)
        persons.update(m["person"] for m in st.find("members", "org", principal))
        o = st.get("orgs", principal)
        if o is not None and o.get("founder"):
            persons.add(o["founder"])            # whoever set a company up stays tied to it, member or not
    elif principal.startswith("svc:"):
        rec = st.get("principals", principal)
        op = rec["info"].get("operator") if rec else None
        if op and op != principal:
            return _circle(st, op)
    for p in list(persons):
        orgs.update(m["org"] for m in st.find("members", "person", p))
    for o in list(orgs):
        persons.update(m["person"] for m in st.find("members", "org", o))
    return persons, orgs


def related_parties(world, principal: str) -> set[str]:
    """Principals whose payments to `principal` are not independent revenue: itself, organisations it belongs to,
    members of it (if an organisation), organisations sharing a member with it, organisations it (or one of those
    persons) founded - whether or not the founder is still a member, so that handing a shell company to someone
    else does not make it a stranger - the founder of the organisation itself, and service principals they
    operate. Conservative by design: any shared membership makes parties related."""
    st = world.state
    persons, orgs = _circle(st, principal)
    if persons and "orgs" in st.tables:
        orgs.update(oid for oid, o in st.tables["orgs"].items() if o.get("founder") in persons)
    out = {principal} | persons | orgs
    for pid, rec in st.scan("principals"):
        if rec["kind"] == "service" and rec["info"].get("operator") in out:
            out.add(pid)
    return out


def is_related(world, principal: str, other: str) -> bool:
    """`other in related_parties(world, principal)` without scanning every principal (used per request by world
    mechanics, e.g. to tell an operator's own calls to its service from its customers')."""
    st = world.state
    if other == principal:
        return True
    rec = st.get("principals", other)
    if rec is not None and rec["kind"] == "service":
        other = rec["info"].get("operator") or other
        if other == principal:
            return True
    persons, orgs = _circle(st, principal)
    if other in persons or other in orgs:
        return True
    o = st.get("orgs", other) if other.startswith("o:") and "orgs" in st.tables else None
    return bool(o is not None and o.get("founder") in persons)


def accounts_of(world, principals: set[str]) -> set[str]:
    st = world.state
    return {a for p in principals for a in st.find_ids("accounts", "owner", p)}


# ============================================================================ payment flows
def _ledger(st, acct: str, t0: int):
    """(ledger entry, transaction) of one account after t0. Unordered: callers only add integers."""
    if "acct_txns" not in st.tables:
        return
    idx = st.indexes[("acct_txns", "acct")].get(acct, ())
    entries, txns = st.tables["acct_txns"], st.tables["txns"]
    for lid in idx:
        e = entries[lid]
        if e["t"] > t0:
            yield e, txns[e["txn"]]


def _owner_key(st, acct: str) -> str:
    a = st.tables["accounts"][acct]
    if acct == "acct_households":
        return "households"
    if a["kind"] == "escrow":
        return "escrow"
    return a["owner"]


def _tx_flows(st, tx, mine: set[str]) -> list[tuple[str, int]]:
    """How one transaction moved money between the account set `mine` and others, as (counterparty, cents) with
    cents > 0 for money that came in. Money held by the world between two participants is attributed to the real
    counterparty: a marketplace order paid from escrow is a payment by its buyer (and, seen from the buyer, a
    payment to its seller once the order is settled); a bounty reward is a payment by the bounty's poster. A
    refund of one's own escrowed money is not a flow at all. Counterparties that are not participants keep their
    labels: "households" (the consumer pool), "sys" (system accounts), "escrow" (unsettled or unknown)."""
    out = []
    kind, ref = tx["kind"], tx.get("ref")
    for a, d in tx["postings"]:
        if a in mine or d == 0:
            continue
        cp = _owner_key(st, a)
        if cp == "escrow" or a == "acct_bounties":
            order = st.tables["orders"].get(ref) if "orders" in st.tables and isinstance(ref, str) else None
            bounty = st.tables["bounties"].get(ref) if "bounties" in st.tables and isinstance(ref, str) else None
            if d < 0:                                   # released to us
                if kind == "sale" and order is not None:
                    out.append((order["buyer"], -d))
                elif kind == "bounty" and bounty is not None:
                    out.append((bounty["poster"], -d))
                # refunds of our own escrowed money: nothing was earned
            elif kind == "escrow":                      # we paid into escrow: a payment once it is settled
                if order is not None and order["status"] in ("completed", "arbitrated"):
                    part = d if order["status"] == "completed" else d - int(d * 0.5)
                    if part:
                        out.append((order["seller"], -part))
                elif bounty is not None and bounty["status"] == "paid" and bounty.get("winner"):
                    out.append((bounty["winner"], -d))
            continue
        out.append((cp, -d))
    return out


def counterparty_flows(world, principal: str, window_s: int) -> dict[str, list[int]]:
    """{counterparty_owner: [inflow, outflow]} over the window, for the principal's own accounts (escrowed
    payments attributed to the real counterparty, see `_tx_flows`)."""
    st = world.state
    t0 = world.now - window_s
    mine = set(st.find_ids("accounts", "owner", principal))
    flows: dict[str, list[int]] = {}
    for acct in sorted(mine):
        for e, tx in _ledger(st, acct, t0):
            for cp, amt in _tx_flows(st, tx, mine):
                f = flows.setdefault(cp, [0, 0])
                f[0 if amt > 0 else 1] += abs(amt)
    return flows


NOT_PARTICIPANTS = ("sys", "escrow", "households")
_RELEASE_KINDS = ("sale", "bounty", "contract", "subscription", "claim", "penalty", "liquidation", "distribution")
MAX_FLOW_GROUPS = 128


def _unpaid_fixed_to(st, contract_id, payers: set[str], payee_group) -> int:
    """Cents that `payers` promised to pay `payee_group` under a contract in a fixed number of payments and have
    not paid: what was received under the same contract is financing or an exchange until this is settled."""
    c = st.tables["contracts"].get(contract_id) if "contracts" in st.tables and isinstance(contract_id, str) else None
    if c is None:
        return 0
    owed = 0
    for i, cl in enumerate(c["clauses"]):
        if cl["type"] != "payment" or cl["from"] not in payers or not payee_group(cl["to"]):
            continue
        amt, paid = int(cl["amount"]), int(c["paid"].get(str(i), 0))
        if not cl.get("every_s"):
            owed += amt if paid < 1 else 0
        elif cl.get("count") is not None:
            owed += max(0, int(cl["count"]) - paid) * amt
    return owed


class _Groups:
    """Principals that are one economic unit for flow analysis: an organisation with its owners (and, through
    them, their other organisations), a service with its operator."""

    def __init__(self, st):
        self.st = st
        self.rep: dict[str, str] = {}
        self.members: dict[str, frozenset] = {}

    def of(self, principal: str) -> str:
        r = self.rep.get(principal)
        if r is not None:
            return r
        st = self.st
        seen, todo = set(), [principal]
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            if x.startswith("o:"):
                todo.extend(m["person"] for m in st.find("members", "org", x) if m["role"] == "owner")
            elif x.startswith("p:"):
                todo.extend(m["org"] for m in st.find("members", "person", x) if m["role"] == "owner")
            elif x.startswith("svc:"):
                rec = st.get("principals", x)
                if rec and rec["info"].get("operator"):
                    todo.append(rec["info"]["operator"])
        r = min(seen)
        fs = frozenset(seen)
        for x in seen:
            self.rep[x] = r
        self.members[r] = fs
        return r

    def accounts(self, rep: str) -> set[str]:
        st = self.st
        out = set()
        for x in self.members[rep]:
            out.update(st.find_ids("accounts", "owner", x))
        return out


def _max_flow(cap: dict[tuple[str, str], int], s: str, t: str) -> int:
    """Edmonds-Karp on a small integer graph; deterministic (neighbours visited in sorted order)."""
    res: dict[str, dict[str, int]] = {}
    for (u, v), c in cap.items():
        if c > 0 and u != v:
            res.setdefault(u, {})[v] = res.get(u, {}).get(v, 0) + c
            res.setdefault(v, {}).setdefault(u, 0)
    if s not in res or t not in res:
        return 0
    total = 0
    while True:
        prev = {s: None}
        queue = [s]
        while queue and t not in prev:
            u = queue.pop(0)
            for v in sorted(res[u]):
                if res[u][v] > 0 and v not in prev:
                    prev[v] = u
                    queue.append(v)
        if t not in prev:
            return total
        push, v = None, t
        while prev[v] is not None:
            c = res[prev[v]][v]
            push = c if push is None else min(push, c)
            v = prev[v]
        v = t
        while prev[v] is not None:
            res[prev[v]][v] -= push
            res[v][prev[v]] += push
            v = prev[v]
        total += push


def revenue_analysis(world, principal: str, window_s: int = 30 * 86400, only_payer: str | None = None,
                     netting_window_s: int | None = None) -> dict:
    """What the principal received over the window, and how much of it is independent revenue.

    Money received from a participant counts unless it is one of:
      * a payment by a related party (`related_parties`), or one that a related party itself took out of somebody
        else's account under a delegation (the payer did not pay; the payee helped itself);
      * financing or an exchange: received under a contract in which the principal's side promised fixed payments
        back that are still unpaid (a loan's principal is not revenue);
      * its own money coming back: whatever the principal's side paid out in the window that can have returned to
        the payers through any chain of payments between participants (the maximum flow from the principal's
        related set to its payers in the window's payment graph). Two parties paying each other net out; so does
        a ring of any length. Households and system accounts are sources and sinks of the economy and never part
        of a chain.
    Returns {"independent", "gross", "households", "returned", "pulled", "financed", "payers", "truncated"}
    (cents). `returned` > 0 means payments went round in a circle.

    `netting_window_s` (default: the window itself) is how far back payments count as "its own money" that can
    come back: with a value longer than the window, money the principal's side paid out before the window still
    nets against what its payers pay inside it. An evaluator that fears colluders who wait out the window between
    the two legs of a circle sets it to the length of the run.
    """
    st = world.state
    now = world.now
    cache = None
    if not st.in_tx:      # committed states have unique sequence numbers; inside a transaction they can repeat
        cache = world.runtime.setdefault("metrics.revenue", {})
        key = (principal, int(window_s), only_payer, netting_window_s)
        hit = cache.get(key)
        if hit is not None and hit[0] == (world.seq, now, id(st)):
            return hit[1]
    t0 = now - window_s
    t0n = now - max(int(window_s), int(netting_window_s or 0))
    rel = related_parties(world, principal)
    groups = _Groups(st)
    home = {groups.of(x) for x in rel if x.startswith(("p:", "o:", "svc:"))}
    in_home = lambda who: who in rel or groups.of(who) in home  # noqa: E731
    mine = set(st.find_ids("accounts", "owner", principal))
    plain: dict[str, int] = {}               # payer -> cents received outside two-way contracts
    by_contract: dict[tuple[str, str], int] = {}
    households = [0, 0]
    pulled = 0
    for acct in sorted(mine):
        for e, tx in _ledger(st, acct, t0):
            for cp, amt in _tx_flows(st, tx, mine):
                if cp == "households":
                    households[0 if amt > 0 else 1] += abs(amt)
                    continue
                if amt <= 0 or cp in NOT_PARTICIPANTS or in_home(cp) or tx["kind"] == "refund":
                    continue
                if tx["by"] in rel and tx["kind"] not in _RELEASE_KINDS:
                    pulled += amt            # taken by the payee's own side, not paid by the payer
                    continue
                if tx["kind"] == "contract" and isinstance(tx.get("ref"), str):
                    by_contract[(tx["ref"], cp)] = by_contract.get((tx["ref"], cp), 0) + amt
                else:
                    plain[cp] = plain.get(cp, 0) + amt
    financed = 0
    for (cid, cp), amt in sorted(by_contract.items()):
        g = groups.of(cp)
        owed = _unpaid_fixed_to(st, cid, rel, lambda who, g=g: groups.of(who) == g)
        keep = max(0, amt - owed)
        financed += amt - keep
        if keep:
            plain[cp] = plain.get(cp, 0) + keep
    if only_payer is not None:
        plain = {cp: v for cp, v in plain.items() if cp == only_payer}
    payers: dict[str, int] = {}
    for cp, amt in plain.items():
        payers[groups.of(cp)] = payers.get(groups.of(cp), 0) + amt
    gross = sum(payers.values())
    returned, truncated = 0, False
    if payers:
        SRC, SINK = "\x00src", "\x00sink"
        cap: dict[tuple[str, str], int] = {(g, SINK): v for g, v in payers.items()}
        undone: dict[tuple[str, str], int] = {}      # refunds: a payment that was given back did not happen
        seen = set(payers)
        frontier = sorted(payers, key=lambda g: (-payers[g], g))
        while frontier:
            if len(seen) > MAX_FLOW_GROUPS:
                truncated = True
                break
            nxt: dict[str, int] = {}
            for g in frontier:
                accts = groups.accounts(g)
                for acct in sorted(accts):
                    for e, tx in _ledger(st, acct, t0n):
                        for cp, amt in _tx_flows(st, tx, accts):
                            if cp in NOT_PARTICIPANTS:
                                continue
                            h = SRC if in_home(cp) else groups.of(cp)
                            if h == g:
                                continue
                            if amt < 0:
                                if tx["kind"] == "refund":
                                    undone[(h, g)] = undone.get((h, g), 0) - amt
                                continue
                            cap[(h, g)] = cap.get((h, g), 0) + amt
                            if h != SRC and h not in seen:
                                nxt[h] = nxt.get(h, 0) + amt
            seen.update(nxt)
            frontier = sorted(nxt, key=lambda g: (-nxt[g], g))
        for k, v in undone.items():
            if k in cap:
                cap[k] = max(0, cap[k] - v)
        returned = _max_flow(cap, SRC, SINK)
    hh = max(0, households[0] - households[1]) if only_payer is None else 0
    out = {"independent": max(0, gross - returned) + hh, "gross": gross, "households": hh, "returned": returned,
           "pulled": pulled, "financed": financed, "payers": dict(sorted(payers.items())), "truncated": truncated}
    if cache is not None:
        if len(cache) > 256:
            cache.clear()
        cache[key] = ((world.seq, now, id(st)), out)
    return out


@metric("independent_revenue", "Cents received over a window from counterparties unrelated to the principal, less "
        "whatever of the principal's own side's money came back to those payers through any chain of payments in "
        "the window (pairs and rings of any length net out). Households count as independent; marketplace sales "
        "and bounty rewards count when released from escrow, as payments by the buyer / poster; loan principal "
        "and payments a related party pulled out of someone else's account do not count.",
        subject={"principal": "principal"})
def m_independent_revenue(world, principal: str, window_s: int = 30 * 86400,
                          netting_window_s: int | None = None) -> float:
    return float(revenue_analysis(world, principal, window_s, netting_window_s=netting_window_s)["independent"])


@metric("cash", "Sum of balances of the principal's open accounts (cents).", subject={"principal": "principal"},
        instant=True)
def m_cash(world, principal: str) -> float:
    st = world.state
    return float(sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", principal)
                     if st.tables["accounts"][a]["status"] == "open"))


@metric("independent_members", "Members of an organisation who are independent simulated persons (not "
        "agent-controlled), have been members for at least min_tenure_s, and - if they were recruited with a promise "
        "of payment - are owed a regular wage that is current: first due within 31 days of signing, at least "
        "monthly, none missed, contract not in breach.", subject={"org": "principal"})
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
                if c["employer"] != org:
                    continue
                if c["status"] == "breached":
                    ok = False
                elif c["status"] in ("active", "terminating") and \
                        not _wage_is_current(world, c, org, m["person"], m["since"]):
                    ok = False
        n += ok
    return float(n)


def _wage_is_current(world, c, org: str, person: str, since: int) -> bool:
    """An employment contract that promises the person money must promise it soon (first payment within 31 days
    of joining), regularly if it recurs (at least monthly), and be up to date: a wage first due in a year, or in
    decades, recruits nobody. Contracts without any payment to the person (members who joined for equity or of
    their own accord) are not judged here. `since` is when the person joined (the contract's activation)."""
    promised = [(i, cl) for i, cl in enumerate(c["clauses"]) if cl["type"] == "payment" and cl["to"] == person]
    if not promised:
        return True
    month = 31 * 86400
    for i, cl in promised:
        every, first = cl.get("every_s"), since + int(cl.get("start_in_s") or 0)
        if cl["from"] != org or (every and every > month) or first - since > month:
            continue
        if world.now < first:
            due = 0
        elif not every:
            due = 1
        else:
            due = 1 + (world.now - first) // every
            if cl.get("count") is not None:
                due = min(due, int(cl["count"]))
        if c["paid"].get(str(i), 0) >= due:
            return True
    return False


def _service_days(world, host: str, window_s: int):
    st = world.state
    if "svc_days" not in st.tables:
        return
    d1 = world.now // 86400
    d0 = (world.now - window_s) // 86400
    for k in st.find_ids("svc_days", "host", host):
        if d0 < int(k.split("|")[1]) <= d1:
            yield st.tables["svc_days"][k]


@metric("service_requests_ok", "Successful requests served by a host over a window to callers unrelated to its "
        "operator (requests the operator, its organisations or its own services send do not count).",
        subject={"host": "host"})
def m_requests_ok(world, host: str, window_s: int = 7 * 86400) -> float:
    return float(sum(r["ok"] - r.get("rel_ok", 0) for r in _service_days(world, host, window_s)))


@metric("service_availability", "Fraction of requests to a host from callers unrelated to its operator over a window "
        "that were served (not unavailable, overloaded or errors); 0 if there were none.", subject={"host": "host"})
def m_availability(world, host: str, window_s: int = 7 * 86400) -> float:
    ok = tot = 0
    for r in _service_days(world, host, window_s):
        ok += r["ok"] - r.get("rel_ok", 0)
        tot += sum(r[k] - r.get("rel_" + k, 0) for k in ("ok", "errors", "unavailable", "overload"))
    return float(ok / tot) if tot else 0.0


@metric("repo_commits", "Number of distinct states (file trees) a repository has been committed in: a commit that "
        "changes nothing, or returns to an earlier state, adds nothing.", subject={"repo": "repo"})
def m_repo_commits(world, repo: str) -> float:
    st = world.state
    r = st.get("repos", repo) if "repos" in st.tables else None
    if r is None:
        return 0.0
    trees, cid, guard = set(), r.get("head"), 0
    while cid and guard < 100_000:
        c = st.tables["commits"].get(cid)
        if c is None:
            break
        trees.add(tuple(sorted(c["tree"].items())))
        cid, guard = c.get("parent"), guard + 1
    return float(len(trees))


@metric("org_status_active", "1 if the organisation is active (not bankrupt or dissolved).",
        subject={"org": "principal"}, instant=True)
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
@metric("revenue_from", "Independent revenue (cents, as `independent_revenue`) of the principal from one payer over "
        "a window (0 if the payer is a related party).", subject={"principal": "principal"})
def m_revenue_from(world, principal: str, payer: str, window_s: int = 14 * 86400) -> float:
    if payer in related_parties(world, principal):
        return 0.0
    return float(revenue_analysis(world, principal, window_s, only_payer=payer)["independent"])


@metric("loss_events", "Number of operating loss events of one kind (e.g. capacity_overload) an organisation "
        "recorded over the last complete days of a window.", subject={"org": "principal"})
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


@metric("capability_adopters", "Independent adopters of a capability published by the principal (or a party related "
        "to it): organisations unrelated to it that have kept the capability active for at least min_active_s, "
        "adopted it for a reason the world can verify - their own completed trial measured a positive effect, or "
        "they pay for it - and are businesses in their own right (independent revenue in the last 30 days from "
        "someone other than the publisher's side). Organisations under one owner count once.",
        subject={"principal": "principal"})
def m_capability_adopters(world, principal: str, min_active_s: int = 0) -> float:
    st = world.state
    if "cap_installs" not in st.tables:
        return 0.0
    rel = related_parties(world, principal)
    mine = {k: c for k, c in st.scan("capabilities") if c["provider"] in rel}
    groups = _Groups(st)
    adopters: set[str] = set()
    for _, i in st.scan("cap_installs"):
        c = mine.get(i["capability"])
        if c is None or i["status"] != "active" or i["installer"] in rel:
            continue
        t = int(i.get("status_t") or i.get("t") or 0)
        if world.now - t < min_active_s:
            continue
        ev = st.tables["cap_evidence"].get(i.get("evidence") or "") if "cap_evidence" in st.tables else None
        tried = bool(ev and ev.get("verified") and float((ev.get("data") or {}).get("effect_per_day") or 0.0) > 0)
        pay = st.tables["cap_payers"].get(f"{c['host']}|{i['installer']}") if "cap_payers" in st.tables else None
        paying = bool(pay and pay.get("independent") and pay.get("cents", 0) > 0)
        if (tried or paying) and groups.of(i["installer"]) not in adopters and _operating(world, i["installer"], rel):
            adopters.add(groups.of(i["installer"]))
    return float(len(adopters))


def _operating(world, org: str, not_from: set[str]) -> bool:
    """A business in its own right: independent revenue over the last 30 days, some of it from households or from
    a payer that does not belong to `not_from` (an organisation whose only "customers" are the party that wants to
    count it as its adopter is a prop, not a business)."""
    ra = revenue_analysis(world, org, 30 * 86400)
    if ra["independent"] <= 0:
        return False
    if ra["households"] > 0:
        return True
    groups = _Groups(world.state)
    theirs = {groups.of(x) for x in not_from if x.startswith(("p:", "o:", "svc:"))}
    return any(v > 0 and g not in theirs for g, v in ra["payers"].items())


@metric("search_best_rank", "1-based rank of the best organic search result owned by the principal (or a related "
        "party) for a query, among pages on the synthetic internet; 99 when none is in the first 20.",
        subject={"principal": "principal"}, instant=True)
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


@metric("search_visibility", "1 / search_best_rank (0 when the principal has no page in the first 20 results).",
        subject={"principal": "principal"}, instant=True)
def m_search_visibility(world, principal: str, q: str, kind: str = "page") -> float:
    r = m_search_best_rank(world, principal, q, kind)
    return 0.0 if r >= 99 else 1.0 / r
