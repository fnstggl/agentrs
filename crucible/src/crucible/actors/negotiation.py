"""How Tier-2 minds negotiate (installed into `society`): evaluation of offers from the recipient's own position,
private reservation values, trust, public reputation, memory, attention-limited and delayed replies, concession
over rounds, counteroffers and responses to broken commitments.

Valuation (money-equivalent cents over a horizon, from the recipient's point of view):

* payments to me count at face value times the probability I believe the payer will pay (trust blended with its
  public record); payments I make count fully against;
* joining an organisation costs a person the value of its time (its private reservation wage, or its current
  salary if employed) and is worth a private affinity for that organisation;
* receiving shares is worth the stake times a valuation of the company from its *verified* operating results,
  discounted by risk aversion; giving away shares of one's own company costs that plus a control premium;
* delegating a capability costs its sensitivity times distrust; a deliverable I owe costs effort; one owed to me
  is worth its penalty (what I can recover) times trust.

Decision: reject outright counterparties with a bad record or broken promises to me; accept if the value beats
an aspiration level that starts above the reservation and decays with each round (patience); otherwise counter by
moving the main payment toward the aspiration; give up after a few rounds. Replies take time (attention).
"""
from __future__ import annotations

import math

from ..core.errors import CrucibleError

DAY = 86400
MONTH = 30 * DAY


def _subst(x, me, you):
    from ..services.deals import _subst as s
    return s(x, me, you)


def blended_trust(soc, ctx, m, who) -> float:
    deals = ctx.world.modules.get("deal")
    a, b = (m["trust"].get(who) or [1.0, 1.0])
    if deals is not None:
        rep = deals.reputation(ctx.world, who)
        a += 0.5 * rep["contracts_completed"]
        b += 3.0 * rep["contracts_breached_by"] + 1.0 * rep["claims_owed"] + 2.0 * rep.get("debts_written_off", 0)
    return a / (a + b)


def org_value(ctx, org) -> float:
    """A conservative valuation of a company from its verified books: 12 months of positive operating result
    (last 30 days) plus cash."""
    st = ctx.state
    ops = ctx.world.modules.get("ops")
    bank = ctx.world.modules["bank"]
    acct = bank.primary_account(st, org)
    cash = bank.balance(st, acct) if acct else 0
    op = ops.window(st, org, 30, ctx.now // DAY + 1)["operating"] if ops else 0
    return max(0.0, cash + 12 * max(0, op))


def total_shares(st, org) -> int:
    return sum(s["shares"] for s in st.find("shares", "org", org)) or 1


def evaluate(soc, ctx, m, me: str, d: dict) -> tuple[float, dict]:
    st = ctx.state
    you = d["from"]
    clauses = _subst(d["terms"], d["from"], d["to"])
    tr = blended_trust(soc, ctx, m, you)
    horizon = 3 * MONTH
    spans = [int(c["count"]) * int(c["every_s"]) for c in clauses
             if c.get("type") == "payment" and c.get("every_s") and c.get("count")]
    if spans:            # a fixed-term agreement: value time commitments over its actual duration
        horizon = min(horizon, max(spans))
    u = 0.0
    parts = []
    for i, cl in enumerate(clauses):
        t = cl.get("type")
        v = 0.0
        if t == "payment":
            every = cl.get("every_s")
            n = int(cl.get("count") or (horizon // int(every) if every else 1)) if every else 1
            disc = sum(m["traits"]["patience"] ** (k * (int(every or MONTH)) / MONTH) for k in range(n))
            amt = float(cl["amount"]) * disc
            if cl["to"] == me:      # a payer that defaults stops being worked for: exposure is limited
                v = amt * (0.6 + 0.4 * tr)
            elif cl["from"] == me:
                v = -amt
        elif t == "membership":
            if cl["person"] == me:
                months = horizon / MONTH
                v = -float(m.get("reservation", 200_000)) * months
                v += m.get("affinity", {}).get(cl["org"], 0.0) * float(m.get("reservation", 200_000)) * months
                if m.get("employer") and m["employer"] != cl["org"]:
                    v -= 0.1 * float(m.get("reservation", 200_000))      # switching cost
            elif cl["org"] == me or cl["org"] == m.get("org"):
                v = float(m.get("labor_value", 0.0)) * horizon / MONTH
        elif t == "shares":
            frac = cl["shares"] / max(1, total_shares(st, cl["org"]))
            val = frac * org_value(ctx, cl["org"]) * (1 - 0.5 * (1 - m["traits"]["risk"]))
            if cl["to"] == me:
                v = val * tr
            elif cl["from"] == me:
                v = -val * (1.5 if cl["org"] in (me, m.get("org")) else 1.0)
        elif t == "grant":
            from .minds import SENSITIVITY
            if cl["owner"] in (me, m.get("org")):
                v = -SENSITIVITY.get(cl["cap"], 1.0) * 50_000 * (1 - tr) * (3 if cl.get("scope", "*") == "*" else 1)
        elif t == "deliverable":
            pen = cl.get("penalty") or {}
            if cl["by"] in (me, m.get("org")):
                v = -float(m.get("effort_cost", 100_000)) - (pen.get("amount", 0) * 0.3 if pen.get("from") == me else 0)
            else:
                v = float(pen.get("amount", 0)) * 0.5 * tr if pen.get("to") == me else 0.0
        parts.append((i, t, round(v, 1)))
        u += v
    return u, {"trust": round(tr, 3), "parts": parts}


def decide(soc, ctx, mid, m, me, d, rng) -> tuple[str, list | None, str]:
    u, info = evaluate(soc, ctx, m, me, d)
    you = d["from"]
    rnd = int(d["round"])
    if info["trust"] < 0.25:
        return "reject", None, "We do not do business with parties that have broken agreements."
    give = sum(-p[2] for p in info["parts"] if p[2] < 0)
    stake = sum(abs(p[2]) for p in info["parts"]) or 1.0
    greed = m.get("greed", 0.15)
    # aspiration: a premium over what we give up, conceded with every round (patience)
    asp = greed * give * (m["traits"]["patience"] ** (2 * rnd)) - 0.03 * give * rnd
    if u >= asp:
        return "accept", None, "Agreed."
    max_rounds = 2 + int(m["traits"]["patience"] * 4)
    if rnd >= max_rounds or u < -0.6 * give:
        return "reject", None, "This does not work for us."
    # counter: move the largest payment toward what we would accept
    clauses = _subst(d["terms"], d["from"], d["to"])
    pays = [(abs(c["amount"]), i) for i, c in enumerate(clauses) if c.get("type") == "payment" and me in (c["to"], c["from"])]
    if not pays:
        return "reject", None, "Not interesting for us."
    _, i = max(pays)
    c = dict(clauses[i])
    every = c.get("every_s")
    n = int(c.get("count") or (3 * MONTH // int(every) if every else 1)) if every else 1
    gap = asp - u + 0.03 * give
    if c["to"] == me:
        c["amount"] = int(math.ceil(c["amount"] + gap / max(1, n) / max(0.2, info["trust"])))
    else:
        c["amount"] = max(1, int(c["amount"] - gap / max(1, n)))
        if c["amount"] <= 1:
            return "reject", None, "We cannot make this work."
    new = list(clauses)
    new[i] = c
    return "counter", new, f"We could do it at {c['amount']} cents" + (" per period." if every else ".")


def on_deal_offered(soc, ctx, deal_id, recipient):
    st = ctx.state
    mid = _mind_for(st, recipient)
    if mid is None:
        return
    m = st.tables["minds"][mid]
    rng = ctx.world.rng_for("deal-delay", deal_id)
    hours = 1 + int(rng.integers(0, max(1, 24 // max(1, int(m["traits"]["attention"])))))
    ctx.schedule(ctx.now + hours * 3600, "society.deal_reply", {"deal": deal_id, "mind": mid}, prio=57)


def _mind_for(st, principal):
    if "minds" not in st.tables:
        return None
    if st.has("minds", principal):
        return principal
    return None


def deal_reply(soc, ctx, deal, mind):
    st = ctx.state
    d = st.get("deals", deal)
    m = st.get("minds", mind)
    if d is None or m is None or d["status"] != "open":
        return
    m = dict(m)
    me = d["to"]
    rng = ctx.rng("deal-reply", deal)
    if rng.random() > 0.35 + 0.65 * m["traits"]["sociability"] and d["round"] == 0 and not soc.p("always_reply", False):
        soc.remember(ctx, mind, m, "ignored_offer", d["from"], {"deal": deal})
        return                                           # ignored (attention elsewhere); the offer expires
    action, terms, msg = decide(soc, ctx, mind, m, me, d, rng)
    actor = m["owner"]
    try:
        args = {"deal": deal, "action": action, "message": msg}
        if terms is not None:
            args["terms"] = terms
        ctx.call("deal.respond", args, as_principal=actor)
    except CrucibleError as e:
        ctx.emit("society.failed", {"who": actor, "tool": "deal.respond", "error": e.as_dict()}, vis="internal")
        action = "failed"
    m = soc.add_contact(ctx, mind, m, d["from"])
    if action == "accept":
        m = soc.update_trust(ctx, mind, m, d["from"], 0.3, 0.0)
    soc.remember(ctx, mind, m, f"deal_{action}", d["from"], {"deal": deal, "round": d["round"]})


def on_breach(soc, ctx, cid, who):
    """Counterparties of a broken agreement lose trust in the breaker and (if sociable) warn others publicly."""
    st = ctx.state
    c = st.tables["contracts"][cid]
    for p in c["parties"]:
        if p == who:
            continue
        mid = _mind_for(st, p)
        if mid is None:
            continue
        m = dict(st.tables["minds"][mid])
        m = soc.update_trust(ctx, mid, m, who, 0.0, 4.0)
        soc.remember(ctx, mid, m, "breach_suffered", who, {"contract": cid})
        if m["traits"]["sociability"] > 0.5 and "forum" in ctx.world.modules:
            try:
                ctx.call("forum.post", {"board": "general", "title": f"Warning about {who}",
                                        "body": f"{who} broke our agreement {cid}. Be careful. #breach {who}"},
                         as_principal=m["owner"])
            except CrucibleError:
                pass


# ====================================================================== institutional behaviour
def on_gov_proposal(soc, ctx, pid, voter):
    """Members with a mind read a proposal after an attention-dependent delay and vote from their own interest."""
    st = ctx.state
    mid = _mind_for(st, voter)
    if mid is None:
        return
    m = st.tables["minds"][mid]
    rng = ctx.world.rng_for("gov-delay", pid, voter)
    hours = 1 + int(rng.integers(0, max(1, 36 // max(1, int(m["traits"]["attention"])))))
    ctx.schedule(ctx.now + hours * 3600, "society.gov_vote", {"proposal": pid, "mind": mid}, prio=57)


def gov_vote(soc, ctx, proposal, mind):
    st = ctx.state
    p = st.get("proposals", proposal)
    m = st.get("minds", mind)
    if p is None or m is None or p["status"] != "open":
        return
    m = dict(m)
    me = m["owner"]
    rng = ctx.rng("gov-vote", proposal, mind)
    if rng.random() > 0.3 + 0.7 * m["traits"]["sociability"]:
        return                                                  # abstains (did not get round to it)
    args = p["args"] or {}
    to = args.get("to") or args.get("to_account") or ""
    bank = ctx.world.modules["bank"]
    to_owner = None
    if isinstance(to, str) and to.startswith("acct_") and st.has("accounts", to):
        to_owner = bank.account_owner(st, to)
    elif isinstance(to, str) and to:
        to_owner = to
    tr = blended_trust(soc, ctx, m, p["by"])
    if to_owner == me:
        approve = True                                          # it pays me
    elif to_owner is not None and to_owner == p["by"]:
        approve = tr > 0.8                                      # it pays the proposer: conflict of interest
    else:
        approve = rng.random() < 0.35 + 0.6 * tr
    try:
        ctx.call("gov.vote", {"proposal": proposal, "approve": bool(approve)}, as_principal=me)
    except CrucibleError as e:
        ctx.emit("society.failed", {"who": me, "tool": "gov.vote", "error": e.as_dict()}, vis="internal")
        return
    soc.remember(ctx, mind, m, "voted", p["by"], {"proposal": proposal, "approve": bool(approve)})
