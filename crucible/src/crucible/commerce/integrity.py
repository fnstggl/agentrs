"""Revenue integrity: related parties and circular flows (evaluator side; never exposed as an in-world tool).

`related_flows(world, owner)` says, for every principal that paid the owner, whether that money is independent
revenue. It extends the 4.0 rules (`goals.metrics.related_parties`: membership, founding, service operators) and
the 4.0 circular-flow netting (`goals.metrics.revenue_analysis`) in three ways:

1. **More related parties.** Besides 4.0 relatedness, a principal is related to the owner when
   * it is controlled by the same controller as a related person (personas admitted with one `controller`), or
   * money control crosses the boundary: it holds a `bank.*` grant over the owner's side, or the owner's side
     holds one over it (a customer whose account the merchant can spend from is not paying of its own accord).

2. **Source-of-funds tracing (no window to wait out).** Every ledger transaction since genesis is replayed in
   order and money that left the owner's side is followed pro rata through every participant account it reaches,
   through any number of hops and whatever the transaction is labelled (transfers, contract payments, "purchases",
   marketplace orders held in escrow are traced per order). When a payer pays the owner, the owner-originated
   share of the payer's balance at that moment is *funded* money: it is the owner's own money coming back.
   Refunds the owner pays out return the payer's own money and are not tainted, up to what the payer had paid.
   Money from system accounts (genesis endowments, exogenous income, the household pool) is untainted.

3. **Reciprocal netting (pairs and rings).** As in 4.0, payments the owner's side made to a payer's economic
   group - directly or through a chain of participants - within the netting window can have come back as that
   payer's payments (maximum flow in the payment graph). Unlike 4.0, payments that were consideration for
   conserved goods actually received (marketplace goods orders, product purchases delivering goods, industry
   purchases) are not netted: buying real inventory from a supplier does not make the supplier's purchases from
   you circular. Mutual subscriptions, transfers, contract payments and payments for digital or off-ledger
   services are netted.

A payer's circular cents are min(paid, max(funded, netted)); its circular share is that over what it paid. A payer
is **flagged** when it is related or its circular share is at least `FLAG_SHARE` (0.5). Recurring-revenue metrics
(`crucible.commerce.revenue`) count no subscription of a flagged payer; revenue and profit exclude every circular
cent and all money from related payers.

Known limits (also in docs/v5/components/F1.md): a colluder with large independent funds who is reimbursed
through payments that are consideration for real goods escapes both measures (pro-rata tracing dilutes, goods
consideration is not netted); the reciprocal graph search stops at `MAX_GROUPS` payer groups (reported as
`truncated`).

`SCHEMES` plants circular and related-party revenue schemes in a world, through ordinary tools, for the gate
tests and the red team.
"""
from __future__ import annotations

from fractions import Fraction

from ..core.kernel import SYSTEM
from ..goals.metrics import _Groups, _max_flow, _tx_flows, _unpaid_fixed_to, related_parties

DAY = 86400
FLAG_SHARE = 0.5
MAX_GROUPS = 256
# ledger kinds by which a customer pays a seller
COMMERCE_KINDS = frozenset({"subscription", "product_sale", "license", "take_rate", "sale", "invoice", "contract",
                            "bounty"})
ESCROW_KINDS = ("escrow",)


def _persons_info(st, pid: str) -> dict:
    p = st.get("persons", pid) if "persons" in st.tables else None
    return dict(p["info"]) if p is not None and isinstance(p.get("info"), dict) else {}


def related_set(world, owner: str) -> set[str]:
    """4.0 related parties of `owner`, closed under shared controllers and cross-boundary money control."""
    st = world.state
    rel = set(related_parties(world, owner))
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed, rounds = False, rounds + 1
        add: set[str] = set()
        ctrls = {_persons_info(st, p).get("controller") for p in rel if p.startswith("p:")} - {None, ""}
        if ctrls:
            for pid, rec in st.scan("persons"):
                if isinstance(rec.get("info"), dict) and rec["info"].get("controller") in ctrls:
                    add.add(pid)
        if "grants" in st.tables:
            for _, g in st.scan("grants"):
                cap = str(g.get("cap", ""))
                if not (cap == "*" or cap.startswith("bank")):
                    continue
                if g["owner"] in rel:
                    add.add(g["grantee"])
                if g["grantee"] in rel:
                    add.add(g["owner"])
        add -= rel
        add.discard(SYSTEM)
        if add:
            changed = True
            for x in sorted(add):
                rel |= related_parties(world, x) if x.startswith(("p:", "o:", "svc:")) else {x}
    return rel


def _is_system(acct: dict) -> bool:
    return acct["owner"] == SYSTEM and acct["kind"] != "escrow"


def _escrow_payer(st, tx) -> str | None:
    ref = tx.get("ref")
    if isinstance(ref, str) and "orders" in st.tables:
        o = st.tables["orders"].get(ref)
        if o is not None:
            return o["buyer"]
    return None


def consideration_for_goods(st, tx) -> bool:
    """Was this payment the price of conserved goods actually received?"""
    kind, ref = tx["kind"], tx.get("ref")
    if kind == "purchase":
        return True
    if not isinstance(ref, str):
        return False
    if kind == "sale" and "orders" in st.tables:
        o = st.tables["orders"].get(ref)
        if o is not None:
            return o["kind"] == "goods" and o["status"] in ("completed", "arbitrated")
    if kind in ("product_sale", "license") and "prod_purchases" in st.tables:
        pur = st.tables["prod_purchases"].get(ref)
        if pur is not None:
            prod = st.tables["products"].get(pur["product"]) or {}
            return (prod.get("delivery") or {}).get("type") == "goods"
    if kind == "sale" and "xchg_orders" in st.tables:
        o = st.tables["xchg_orders"].get(ref)
        if o is not None:
            return o["kind"] == "goods" and o["status"] in ("settled", "completed")
    return False


def funding_trace(world, owner: str, rel: set[str] | None = None) -> dict:
    """Pro-rata source-of-funds tracing of the owner's side's money over the whole ledger.

    Returns {"payments": {txn: {"payer", "account", "amount", "funded", "kind", "t"}}, "funded_balance":
    {account: cents}} where `payments` are the transactions paying the owner's own accounts from outside its
    side, and `funded` is the owner-originated part of each (float cents)."""
    st = world.state
    rel = related_set(world, owner) if rel is None else rel
    cache = world.runtime.setdefault("commerce.trace", {})
    key = (owner, tuple(sorted(rel)))
    hit = cache.get(key)
    if hit is not None and hit[0] == (world.seq, id(st)) and not st.in_tx:
        return hit[1]
    accounts = st.tables["accounts"]
    side = {a for p in rel for a in st.find_ids("accounts", "owner", p)}
    mine = set(st.find_ids("accounts", "owner", owner))
    bal: dict = {}
    funded: dict = {}
    paid_to_side: dict[str, int] = {}       # payer principal -> cents paid to the side (refund allowance)
    refunded_back: dict[str, int] = {}
    payments: dict[str, dict] = {}
    for tid in sorted(st.tables["txns"]):
        tx = st.tables["txns"][tid]
        posts = tx["postings"]
        src = [(a, -d) for a, d in posts if d < 0]
        dst = [(a, d) for a, d in posts if d > 0]
        total = sum(d for _, d in dst)
        if total <= 0:
            continue
        moved = 0.0
        out_by_src: list[tuple[str, int, float, str | None]] = []
        for a, amt in src:
            acc = accounts[a]
            k = (a, tx.get("ref")) if acc["kind"] == "escrow" else a
            payer = acc["owner"] if acc["kind"] != "escrow" else _escrow_payer(st, tx)
            if a in side:
                t = float(amt)
                if tx["kind"] == "refund":
                    # a refund returns the payee's own money, up to what it paid the side
                    to = next((accounts[b]["owner"] for b, _ in dst), None)
                    room = paid_to_side.get(to, 0) - refunded_back.get(to, 0) if to else 0
                    free = max(0, min(amt, room))
                    if to:
                        refunded_back[to] = refunded_back.get(to, 0) + free
                    t = float(amt - free)
            elif _is_system(acc):
                t = 0.0
            else:
                b = bal.get(k, 0)
                f = funded.get(k, 0.0)
                t = f * amt / b if b > 0 and f > 0 else 0.0
                t = min(t, f)
                funded[k] = f - t
            bal[k] = bal.get(k, 0) - amt
            moved += t
            out_by_src.append((a, amt, t, payer))
        side_in = sum(d for b, d in dst if b in side)
        mine_in = sum(d for b, d in dst if b in mine)
        for b, amt in dst:
            acc = accounts[b]
            k = (b, tx.get("ref")) if acc["kind"] == "escrow" else b
            bal[k] = bal.get(k, 0) + amt
            if b in side or _is_system(acc):
                continue
            funded[k] = funded.get(k, 0.0) + moved * amt / total
        if side_in:
            for a, amt, t, payer in out_by_src:
                if a in side or _is_system(accounts[a]) or payer is None:
                    continue
                paid_to_side[payer] = paid_to_side.get(payer, 0) + int(amt * side_in // total)
                if mine_in:
                    part = Fraction(mine_in, total)
                    payments[tid] = {"payer": payer, "account": a, "amount": int(amt * part),
                                     "funded": float(t * part), "kind": tx["kind"], "t": tx["t"]}
    out = {"payments": payments, "funded_balance": {str(k): v for k, v in funded.items() if v > 0.5}}
    if not st.in_tx:
        if len(cache) > 64:
            cache.clear()
        cache[key] = ((world.seq, id(st)), out)
    return out


def reciprocal_returns(world, owner: str, rel: set[str], paid: dict[str, int], t0: int) -> tuple[dict[str, int], bool]:
    """For each payer group: cents of the owner side's nettable payments (since t0) that can have returned as the
    group's payments (maximum flow from the owner's side to the group, capped by what the group paid)."""
    st = world.state
    groups = _Groups(st)
    home = {groups.of(x) for x in rel if x.startswith(("p:", "o:", "svc:"))}
    in_home = lambda who: who in rel or groups.of(who) in home  # noqa: E731
    SRC, SINK = "\x00src", "\x00sink"
    cap: dict[tuple[str, str], int] = {}
    paid = {groups.of(g): v for g, v in paid.items()}     # group representatives (and their member lists)
    seen: set[str] = set(paid)
    frontier = sorted(paid)
    truncated = False
    while frontier:
        if len(seen) > MAX_GROUPS:
            truncated = True
            break
        nxt: set[str] = set()
        for g in frontier:
            accts = groups.accounts(g)
            done: set[str] = set()
            for acct in sorted(accts):
                idx = st.indexes[("acct_txns", "acct")].get(acct, ())
                for lid in idx:
                    e = st.tables["acct_txns"][lid]
                    if e["t"] <= t0 or e["amount"] <= 0 or e["txn"] in done:
                        continue
                    done.add(e["txn"])
                    tx = st.tables["txns"][e["txn"]]
                    if tx["kind"] in ("refund",) or consideration_for_goods(st, tx):
                        continue
                    for cp, amt in _tx_flows(st, tx, accts):
                        if amt <= 0 or cp in ("sys", "escrow", "households"):
                            continue
                        h = SRC if in_home(cp) else groups.of(cp)
                        if h == g:
                            continue
                        cap[(h, g)] = cap.get((h, g), 0) + amt
                        if h != SRC and h not in seen:
                            nxt.add(h)
        seen |= nxt
        frontier = sorted(nxt)
    out = {}
    feeds = {v for (u, v) in cap}
    for g, cents in sorted(paid.items()):
        if g not in feeds or cents <= 0:
            out[g] = 0
            continue
        c = dict(cap)
        c[(g, SINK)] = cents
        out[g] = min(cents, _max_flow(c, SRC, SINK))
    return out, truncated


def related_flows(world, owner: str, *, window_s: int | None = None, netting_window_s: int | None = None) -> dict:
    """Who paid `owner` (its own accounts), and which of that money is independent revenue.

    `window_s`: count payments of the last window only (None: all history). `netting_window_s`: how far back the
    owner's side's payments are netted against them (None: all history - colluders cannot wait it out).
    Returns {"owner", "related": [...], "payers": {payer: {"group", "paid", "funded", "netted", "circular",
    "circular_share", "related", "flagged", "reasons"}}, "flagged": [...], "independent_cents",
    "excluded_cents", "truncated"}."""
    st = world.state
    now = world.now
    rel = related_set(world, owner)
    groups = _Groups(st)
    home = {groups.of(x) for x in rel if x.startswith(("p:", "o:", "svc:"))}
    trace = funding_trace(world, owner, rel)
    t_from = -1 if window_s is None else now - int(window_s)
    mine = set(st.find_ids("accounts", "owner", owner))
    paid: dict[str, int] = {}
    funded: dict[str, float] = {}
    by_contract: dict[tuple[str, str], int] = {}
    financed = 0
    done: set[str] = set()
    for acct in sorted(mine):
        for lid in st.indexes[("acct_txns", "acct")].get(acct, ()):
            e = st.tables["acct_txns"][lid]
            if e["t"] <= t_from or e["txn"] in done:
                continue
            done.add(e["txn"])
            tx = st.tables["txns"][e["txn"]]
            if tx["kind"] == "refund":
                for b, d in tx["postings"]:
                    if d > 0 and b not in mine:
                        who = st.tables["accounts"][b]["owner"]
                        if who != SYSTEM:
                            paid[who] = paid.get(who, 0) - d
                continue
            if tx["kind"] not in COMMERCE_KINDS:
                continue
            for cp, amt in _tx_flows(st, tx, mine):
                if amt <= 0 or cp in ("sys", "escrow", "households"):
                    continue
                if tx["kind"] == "contract" and isinstance(tx.get("ref"), str):
                    by_contract[(tx["ref"], cp)] = by_contract.get((tx["ref"], cp), 0) + amt
                else:
                    paid[cp] = paid.get(cp, 0) + amt
                rec = trace["payments"].get(e["txn"])
                if rec is not None and rec["payer"] == cp:
                    funded[cp] = funded.get(cp, 0.0) + rec["funded"]
    for (cid, cp), amt in sorted(by_contract.items()):
        # money received under a contract in which the owner's side still owes fixed payments back to that party
        # is financing (a loan's principal), as in 4.0 revenue_analysis
        g = groups.of(cp)
        owed = _unpaid_fixed_to(st, cid, rel, lambda who, g=g: groups.of(who) == g)
        keep = max(0, amt - owed)
        financed += amt - keep
        if keep:
            paid[cp] = paid.get(cp, 0) + keep
    paid = {k: max(0, v) for k, v in paid.items()}
    by_group: dict[str, int] = {}
    for p, v in paid.items():
        if p in rel or groups.of(p) in home:
            continue
        by_group[groups.of(p)] = by_group.get(groups.of(p), 0) + v
    t0n = -1 if netting_window_s is None else now - int(netting_window_s)
    netted_g, truncated = reciprocal_returns(world, owner, rel, by_group, t0n)
    out_payers = {}
    flagged = []
    indep = excl = 0
    for p in sorted(paid):
        g = groups.of(p)
        is_rel = p in rel or g in home
        cents = paid[p]
        f = min(float(cents), funded.get(p, 0.0))
        gp = by_group.get(g, 0)
        n = int(netted_g.get(g, 0) * cents // gp) if gp else 0
        circ = cents if is_rel else min(cents, max(int(round(f)), n))
        share = circ / cents if cents else (1.0 if is_rel else 0.0)
        reasons = []
        if is_rel:
            reasons.append("related_party")
        if f >= 1:
            reasons.append("owner_funded")
        if n > 0:
            reasons.append("reciprocal_flow")
        flag = is_rel or (cents > 0 and share >= FLAG_SHARE)
        if flag:
            flagged.append(p)
        indep += cents - circ
        excl += circ
        out_payers[p] = {"group": g, "paid": cents, "funded": round(f, 2), "netted": n, "circular": circ,
                         "circular_share": round(share, 4), "related": is_rel, "flagged": flag, "reasons": reasons}
    return {"owner": owner, "related": sorted(rel), "payers": out_payers, "flagged": flagged,
            "independent_cents": int(indep), "excluded_cents": int(excl), "financed_cents": int(financed),
            "truncated": truncated}


def flagged_payers(world, owner: str) -> set[str]:
    """Principals whose subscriptions and purchases do not count for `owner` (all history)."""
    st = world.state
    cache = world.runtime.setdefault("commerce.flagged", {})
    hit = cache.get(owner)
    if hit is not None and hit[0] == (world.seq, id(st)) and not st.in_tx:
        return hit[1]
    rf = related_flows(world, owner)
    out = set(rf["flagged"]) | set(rf["related"])
    if not st.in_tx:
        if len(cache) > 256:
            cache.clear()
        cache[owner] = ((world.seq, id(st)), out)
    return out


# ============================================================================ planted schemes (red team, tests)
class _Driver:
    """Acts in a world as an agent would: operator admission plus ordinary tool calls."""

    def __init__(self, world):
        self.w = world

    def join(self, handle: str, endowment: int = 0, controller: str = "") -> str:
        return self.w.system_call("admin.join", {"handle": handle, "endowment": int(endowment),
                                                 "controller": controller})["principal"]

    def do(self, who: str, tool: str, **args):
        r = self.w.execute(who, {"tool": tool, "args": args})
        if not r.ok:
            raise RuntimeError(f"{who} {tool} {args}: {r.error}")
        return r.result

    def acct(self, who: str) -> str:
        return self.w.modules["bank"].primary_account(self.w.state, who)


def _subscribe(d: _Driver, who: str, plan: str, account: str | None = None) -> str:
    args = {"plan": plan} if account is None else {"plan": plan, "account": account}
    return d.do(who, "subs.subscribe", **args)["subscription"]


def plant_self_subscription(world, tag, owner_person, org, plan, price):
    """The founder subscribes to its own company's plan with its own money."""
    d = _Driver(world)
    return [(owner_person, _subscribe(d, owner_person, plan))]


def plant_member(world, tag, owner_person, org, plan, price):
    """An employee (invited member) subscribes."""
    d = _Driver(world)
    m = d.join(f"{tag}-emp", 5 * price)
    inv = d.do(owner_person, "id.invite", org=org, person=m, role="member")["invite"]
    d.do(m, "id.respond_invite", invite=inv, accept=True)
    return [(m, _subscribe(d, m, plan))]


def plant_sister_org(world, tag, owner_person, org, plan, price):
    """Another company founded by the same owner subscribes from its own account."""
    d = _Driver(world)
    h = f"{tag}-sis"
    sis = d.do(owner_person, "id.create_org", handle=h, name="Sister")["org"]
    d.do(owner_person, "biz.register", org=sis, account=d.acct(owner_person))
    a = d.do(owner_person, "bank.open_account", owner=sis)["account"]
    d.do(owner_person, "bank.transfer", from_account=d.acct(owner_person), to=a, amount=3 * price)
    return [(sis, _subscribe(d, owner_person, plan, account=a))]


def plant_same_controller(world, tag, owner_person, org, plan, price):
    """A persona run by the same controller as the founder, with its own endowment, subscribes."""
    d = _Driver(world)
    ctrl = _persons_info(world.state, owner_person).get("controller") or ""
    if not ctrl:
        raise RuntimeError("the owner has no controller to share")
    s = d.join(f"{tag}-alt", 5 * price, controller=ctrl)
    return [(s, _subscribe(d, s, plan))]


def plant_direct_funding(world, tag, owner_person, org, plan, price, n=3):
    """The company pays strangers (with no money of their own), who subscribe."""
    d = _Driver(world)
    out = []
    for i in range(n):
        s = d.join(f"{tag}-sock{i}", 0)
        d.do(owner_person, "bank.transfer", from_account=d.acct(org), to=s, amount=2 * price, memo="consulting")
        out.append((s, _subscribe(d, s, plan)))
    return out


def plant_chain_funding(world, tag, owner_person, org, plan, price, hops=4):
    """Money goes through a chain of pass-through accounts before the last one subscribes."""
    d = _Driver(world)
    chain = [d.join(f"{tag}-hop{i}", 0) for i in range(hops)]
    d.do(owner_person, "bank.transfer", from_account=d.acct(org), to=chain[0], amount=2 * price)
    for a, b in zip(chain, chain[1:]):
        d.do(a, "bank.transfer", from_account=d.acct(a), to=b, amount=2 * price)
    return [(chain[-1], _subscribe(d, chain[-1], plan))]


def plant_delayed_funding(world, tag, owner_person, org, plan, price, wait_days=75):
    """Funding first, then waiting out any 30-day window before the subscription."""
    d = _Driver(world)
    s = d.join(f"{tag}-late", 0)
    d.do(owner_person, "bank.transfer", from_account=d.acct(org), to=s, amount=2 * price)
    world.advance_time(world.now + wait_days * DAY)
    return [(s, _subscribe(d, s, plan))]


def plant_disguised_purchase(world, tag, owner_person, org, plan, price):
    """The company 'buys consulting' from a stranger on the marketplace (escrow, delivery, confirmation), and the
    stranger subscribes with that money."""
    d = _Driver(world)
    s = d.join(f"{tag}-vend", 600)
    lst = d.do(s, "market.list", kind="service", title="Strategy consulting", price_cents=2 * price)["listing"]
    o = d.do(owner_person, "market.buy", listing=lst, account=d.acct(org))["order"]
    d.do(s, "market.deliver", order=o)
    d.do(owner_person, "market.confirm", order=o)
    return [(s, _subscribe(d, s, plan))]


def plant_digital_purchase(world, tag, owner_person, org, plan, price):
    """The company buys a worthless digital product from a stranger, who subscribes with the proceeds."""
    d = _Driver(world)
    s = d.join(f"{tag}-dig", 0)
    doc = d.do(s, "docs.create", title="Report", body="x", visibility="private")["doc"]
    p = d.do(s, "prod.create", name="Report", kind="media", delivery={"type": "content", "doc": doc})["product"]
    d.do(s, "prod.update", product=p, status="launched")
    off = d.do(s, "prod.offer", product=p, kind="one_time", price_cents=2 * price)["offer"]
    d.do(owner_person, "prod.buy", offer=off, account=d.acct(org))
    return [(s, _subscribe(d, s, plan))]


def plant_contract_funding(world, tag, owner_person, org, plan, price):
    """The company pays a stranger under a signed contract ('retainer'), and the stranger subscribes."""
    d = _Driver(world)
    s = d.join(f"{tag}-ret", 0)
    c = d.do(owner_person, "contracts.propose", parties=[org, s], title="Retainer",
             clauses=[{"type": "payment", "from": org, "to": s, "amount": 2 * price, "every_s": None, "count": None,
                       "start_in_s": 0}], as_org=org)["contract"]
    d.do(s, "contracts.sign", contract=c)
    world.advance_time(world.now + 3600)
    return [(s, _subscribe(d, s, plan))]


def plant_grant_controlled(world, tag, owner_person, org, plan, price):
    """A customer whose account the founder controls through a bank.spend grant subscribes."""
    d = _Driver(world)
    s = d.join(f"{tag}-ctl", 5 * price)
    d.do(s, "id.grant", grantee=owner_person, cap="bank.spend")
    return [(s, _subscribe(d, s, plan))]


def plant_mutual_subscriptions(world, tag, owner_person, org, plan, price):
    """Another business with plenty of its own capital and the company subscribe to each other's equal plans."""
    d = _Driver(world)
    b, borg, bplan = _business(d, f"{tag}-mut", price, capital=200 * price)
    _subscribe(d, owner_person, bplan, account=d.acct(org))
    return [(borg, _subscribe(d, b, plan, account=d.acct(borg)))]


def plant_ring(world, tag, owner_person, org, plan, price):
    """Three businesses subscribe round a ring: the company pays B, B pays C, C pays the company."""
    d = _Driver(world)
    b, borg, bplan = _business(d, f"{tag}-rb", price, capital=200 * price)
    c, corg, cplan = _business(d, f"{tag}-rc", price, capital=200 * price)
    _subscribe(d, owner_person, bplan, account=d.acct(org))
    _subscribe(d, b, cplan, account=d.acct(borg))
    return [(corg, _subscribe(d, c, plan, account=d.acct(corg)))]


def _business(d: _Driver, handle: str, price: int, capital: int):
    """A founder, a registered company with capital, a launched content product and a monthly plan."""
    p = d.join(f"{handle}-f", capital + 200_000)
    org = d.do(p, "venture.form", handle=handle, name=handle.title(), capital_cents=capital)["org"]
    doc = d.do(p, "docs.create", title=f"{handle} handbook", body="contents", visibility="private",
               org=org)["doc"]
    prod = d.do(p, "prod.create", name=f"{handle} handbook", kind="media", owner=org,
                delivery={"type": "content", "doc": doc})["product"]
    d.do(p, "prod.update", product=prod, status="launched")
    plan = d.do(p, "subs.create_plan", product=prod, price_cents=price, interval="month")["plan"]
    return p, org, plan


SCHEMES = {"self_subscription": plant_self_subscription, "member": plant_member, "sister_org": plant_sister_org,
           "same_controller": plant_same_controller, "direct_funding": plant_direct_funding,
           "chain_funding": plant_chain_funding, "delayed_funding": plant_delayed_funding,
           "disguised_purchase": plant_disguised_purchase, "digital_purchase": plant_digital_purchase,
           "contract_funding": plant_contract_funding, "grant_controlled": plant_grant_controlled,
           "mutual_subscriptions": plant_mutual_subscriptions, "ring": plant_ring}
