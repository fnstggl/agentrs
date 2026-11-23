"""Contracts (host `contracts`): enforceable agreements composed from primitive clauses.

Clause types (composable; none encodes a strategy):
* payment      {from, to, amount, every_s | null, count | null, start_in_s}  - scheduled debits from the account the
                payer designates when signing; a failed payment is a breach and creates a claim.
* membership   {person, org, role, title}  - the person joins the organisation while the contract is active.
* grant        {owner, grantee, cap, scope, limit}  - a delegation that exists while the contract is active.
* shares       {org, from, to, shares}  - equity transfer at activation (company cap table).
* deliverable  {by, predicate, due_in_s, penalty: {from, to, amount} | null}  - a predicate over authoritative
                state (see goals/metrics.py) checked at the due time; failure is a breach (penalty applies).
* notice       {seconds}  - any party may terminate with this notice.

A contract becomes active when every party has signed; offers expire. Breaches are public facts (they affect
counterparties' decisions) and unpaid amounts become claims that the bankruptcy process honours.

What a contract can and cannot do (validated when it is proposed; validation is the only gate, because the
scheduled processes that execute clauses move money without asking anybody again):

* money only ever moves between parties: payer and payee of a payment, and payer and payee of a penalty, have all
  signed. A penalty is owed for a failed deliverable by the party that must deliver, or by another signatory acting
  as guarantor - never to the party that failed (nobody is paid for its own failure);
* every number is a bounded integer (amounts 1..10^12 cents like the ledger, periods of at least an hour, a first
  payment at most a year out), so nothing that cannot be executed can be signed;
* a deliverable predicate may only read metrics about the parties themselves and what they own (their companies,
  hosts, repositories): a contract is not a way to measure somebody who never signed it;
* a public breach record says who breached what and how much is owed - not the debtor's account or balance.
"""
from __future__ import annotations

import numbers
import re

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InsufficientFunds, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

CLAUSE = {"type": "object", "additionalProperties": True}
MAX_CLAUSES = 12
MAX_AMOUNT = 10**12                 # the ledger's own maximum (services/bank.py)
MAX_START_S = 366 * 86400           # first payment / notice at most a year out
MAX_TERM_S = 10 * 365 * 86400       # periods and deliverable deadlines
CAP_RE = r"\*|[a-z]+(\.[a-z_*]+)?"


def _int(x, lo: int, hi: int, what: str) -> int:
    """A clause number: a real integer (never a bool, float or string) inside [lo, hi]."""
    if isinstance(x, bool) or not isinstance(x, numbers.Integral):
        raise InvalidAction(f"{what} must be an integer")
    x = int(x)
    if not lo <= x <= hi:
        raise InvalidAction(f"{what} must be between {lo} and {hi}")
    return x


def remaining_fixed(c, i: int) -> int:
    """Cents still to be paid under payment clause `i` of contract record `c` when the number of payments is
    fixed (a single payment, or a recurring one with a count); 0 for open-ended recurring payments."""
    cl = c["clauses"][i]
    paid = c["paid"].get(str(i), 0)
    if not cl.get("every_s"):
        return 0 if paid >= 1 else int(cl["amount"])
    if cl.get("count") is None:
        return 0
    return max(0, int(cl["count"]) - paid) * int(cl["amount"])


def public_view(cid: str, c) -> dict:
    """What a non-party may see of a breached contract: who, what status, what is owed - no text, no clauses,
    no accounts, no reasons that quote the bank."""
    reason = str(c.get("status_reason") or "")
    if reason.startswith("payment failed"):
        reason = "payment failed"
    owed = [{"from": cl["from"], "to": cl["to"], "amount": cl["amount"]} for cl in c["clauses"]
            if cl["type"] == "payment" and cl["from"] == c.get("breached_by")]
    return {"id": cid, "title": c["title"], "parties": list(c["parties"]), "status": c["status"],
            "breached_by": c.get("breached_by"), "status_reason": reason, "updated": c.get("updated"),
            "payment_obligations_of_the_breaching_party": owed, "redacted": True}


@register_module
class Contracts(Module):
    name = "contracts"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("contracts", index=("status", "employee", "employer"))
        st.declare_table("contract_parties", index=("party", "contract"))
        st.declare_table("claims", index=("debtor", "creditor", "status"))
        world.hooks["contracts"] = self

    # ------------------------------------------------------------------ validation
    def validate(self, ctx: Ctx, parties: list[str], clauses: list[dict]) -> list[dict]:
        from ..goals.metrics import check_predicate, check_predicate_scope, is_related
        st = ctx.state
        out = []
        if not clauses or len(clauses) > MAX_CLAUSES:
            raise InvalidAction(f"1-{MAX_CLAUSES} clauses required")
        R = lambda x: ident(ctx.world).resolve_handle(st, x)  # noqa: E731
        for i, c in enumerate(clauses):
            t = c.get("type") if isinstance(c, dict) else None
            try:
                if t == "payment":
                    cl = {"type": t, "from": R(c["from"]), "to": R(c["to"]), "amount": c["amount"],
                          "every_s": c.get("every_s"), "count": c.get("count"), "start_in_s": c.get("start_in_s", 0)}
                    cl["amount"] = _int(cl["amount"], -MAX_AMOUNT, MAX_AMOUNT, f"clause {i}: amount")
                    if cl["amount"] <= 0 or cl["from"] == cl["to"]:
                        raise InvalidAction("payment needs a positive amount between different parties")
                    cl["start_in_s"] = _int(cl["start_in_s"], 0, MAX_START_S, f"clause {i}: start_in_s")
                    if cl["every_s"] is not None:
                        cl["every_s"] = _int(cl["every_s"], -MAX_TERM_S, MAX_TERM_S, f"clause {i}: every_s")
                        if cl["every_s"] < 3600:
                            raise InvalidAction("recurring payments at most hourly")
                    if cl["count"] is not None:
                        if isinstance(cl["count"], bool) or not isinstance(cl["count"], numbers.Integral) \
                                or not 1 <= cl["count"] <= 10_000:
                            raise InvalidAction("bad count")
                        cl["count"] = int(cl["count"])
                    for k in ("from", "to"):
                        if cl[k] not in parties:
                            raise InvalidAction(f"clause {i}: {k} must be a party")
                elif t == "membership":
                    cl = {"type": t, "person": R(c["person"]), "org": R(c["org"]), "role": c.get("role", "member"),
                          "title": str(c.get("title", ""))[:80]}
                    if cl["role"] not in ("member", "admin") or not cl["person"].startswith("p:") or not cl["org"].startswith("o:"):
                        raise InvalidAction("membership needs a person, an organisation and role member/admin")
                    if cl["person"] not in parties or cl["org"] not in parties:
                        raise InvalidAction("membership parties must sign")
                elif t == "grant":
                    cl = {"type": t, "owner": R(c["owner"]), "grantee": R(c["grantee"]), "cap": str(c["cap"]),
                          "scope": str(c.get("scope", "*")), "limit": c.get("limit")}
                    if cl["owner"] not in parties:
                        raise InvalidAction("the granting owner must be a party")
                    if not isinstance(c["cap"], str) or not re.fullmatch(CAP_RE, cl["cap"]) or len(cl["cap"]) > 40 \
                            or len(cl["scope"]) > 80:
                        raise InvalidAction(f"clause {i}: bad capability or scope")
                    if cl["limit"] is not None:
                        cl["limit"] = _int(cl["limit"], 0, MAX_AMOUNT, f"clause {i}: limit")
                elif t == "shares":
                    cl = {"type": t, "org": R(c["org"]), "from": R(c["from"]), "to": R(c["to"]),
                          "shares": _int(c["shares"], -MAX_AMOUNT, MAX_AMOUNT, f"clause {i}: shares")}
                    if cl["from"] not in parties or cl["to"] not in parties or cl["shares"] <= 0:
                        raise InvalidAction("share transfers between parties, positive amounts")
                elif t == "deliverable":
                    errs = check_predicate(c.get("predicate"))
                    if errs:
                        raise InvalidAction("; ".join(errs))
                    pen = c.get("penalty")
                    cl = {"type": t, "by": R(c["by"]), "predicate": c["predicate"],
                          "due_in_s": _int(c["due_in_s"], -MAX_TERM_S, MAX_TERM_S, f"clause {i}: due_in_s"),
                          "penalty": None if not pen else {"from": R(pen["from"]), "to": R(pen["to"]),
                                                           "amount": _int(pen["amount"], 1, MAX_AMOUNT,
                                                                          f"clause {i}: penalty amount")}}
                    if cl["due_in_s"] < 3600:
                        raise InvalidAction("deliverables due at least an hour out")
                    if cl["by"] not in parties:
                        raise InvalidAction(f"clause {i}: the party that must deliver (`by`) must be a party")
                    p = cl["penalty"]
                    if p is not None:
                        if p["from"] not in parties or p["to"] not in parties or p["from"] == p["to"]:
                            raise InvalidAction(f"clause {i}: a penalty is paid by one party to another party")
                        if p["to"] == cl["by"] or (p["from"] != cl["by"] and is_related(ctx.world, cl["by"], p["to"])):
                            # "if I fail, you pay me" - or my company, my owner, my employee
                            raise InvalidAction(f"clause {i}: a penalty cannot be paid to the party whose own "
                                                "deliverable failed, or to a party related to it")
                    errs = check_predicate_scope(ctx.world, cl["predicate"], parties)
                    if errs:
                        raise Forbidden(f"clause {i}: " + "; ".join(errs[:3]))
                elif t == "notice":
                    cl = {"type": t, "seconds": _int(c["seconds"], 0, MAX_START_S, f"clause {i}: seconds")}
                else:
                    raise InvalidAction(f"unknown clause type {t!r}")
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                raise InvalidAction(f"clause {i} malformed: {e}")
            out.append(cl)
        return out

    # ------------------------------------------------------------------ internal API
    def propose(self, ctx: Ctx, proposer: str, parties: list[str], clauses: list[dict], title: str, text: str = "",
                expires_in_s: int = 7 * 86400, tags: tuple = ()) -> str:
        st = ctx.state
        parties = sorted({ident(ctx.world).resolve_handle(st, p) for p in parties})
        if proposer not in parties:
            raise InvalidAction("the proposer must be a party")
        if not 2 <= len(parties) <= 8:
            raise InvalidAction("2-8 parties")
        cls = self.validate(ctx, parties, clauses)
        cid = st.new_id("ctr")
        emp = next((c for c in cls if c["type"] == "membership"), None)
        ctx.emit("contracts.proposed", {"id": cid, "proposer": proposer, "parties": parties, "clauses": cls,
                                        "title": title[:120], "text": text[:4000], "expires": ctx.now + expires_in_s,
                                        "tags": list(tags), "employee": emp["person"] if emp else None,
                                        "employer": emp["org"] if emp else None}, vis=tuple(parties))
        hook = ctx.world.hooks.get("on_contract_proposed")
        if hook:
            for p in parties:
                if p != proposer:
                    hook(ctx, cid, p)
        return cid

    def sign(self, ctx: Ctx, cid: str, party: str, accounts: dict | None = None) -> dict:
        st = ctx.state
        c = st.get("contracts", cid)
        if c is None or c["status"] != "proposed":
            raise NotFound("no open proposal")
        if party not in c["parties"]:
            raise Forbidden("not a party")
        if party in c["signed"]:
            raise Conflict("already signed")
        if ctx.now > c["expires"]:
            raise Conflict("offer expired")
        bank = ctx.world.modules["bank"]
        accts = {}
        for i, cl in enumerate(c["clauses"]):
            if cl["type"] == "payment" and cl["from"] == party:
                a = (accounts or {}).get(str(i)) or bank.primary_account(st, party)
                if a is None or bank.account_owner(st, a) != party:
                    raise InvalidAction(f"clause {i}: name one of your accounts to pay from")
                # a schedule of future debits is a standing order: it needs full authority over the account, not a
                # delegation capped at an amount per month (the cap could never be enforced payment by payment)
                if ident(ctx.world).authorize_standing(ctx, party, a) is not None:
                    raise Forbidden(f"clause {i}: a spending grant with a limit cannot commit {a} to scheduled "
                                    "contract payments")
                accts[str(i)] = a
        ctx.emit("contracts.signed", {"id": cid, "party": party, "accounts": accts}, vis=tuple(c["parties"]))
        c = st.get("contracts", cid)
        if set(c["signed"]) == set(c["parties"]):
            self._activate(ctx, cid)
        return {"status": st.get("contracts", cid)["status"]}

    def _activate(self, ctx: Ctx, cid: str) -> None:
        st = ctx.state
        c = st.get("contracts", cid)
        ctx.emit("contracts.status", {"id": cid, "status": "active", "reason": "all parties signed"}, vis=tuple(c["parties"]))
        self._activate_clauses(ctx, cid)
        hook = ctx.world.hooks.get("on_contract_activated")
        if hook:
            hook(ctx, cid)

    def _activate_clauses(self, ctx: Ctx, cid: str) -> None:
        st = ctx.state
        c = st.get("contracts", cid)
        idm = ident(ctx.world)
        for i, cl in enumerate(c["clauses"]):
            t = cl["type"]
            if t == "payment":
                ctx.schedule(ctx.now + cl["start_in_s"], "contracts.pay", {"contract": cid, "clause": i}, prio=35,
                             key=f"pay|{cid}|{i}")
            elif t == "membership":
                idm.add_member(ctx, cl["org"], cl["person"], cl["role"], cl["title"])
            elif t == "grant":
                gid = st.new_id("grant")
                ctx.emit("id.grant", {"id": gid, "grantee": cl["grantee"], "owner": cl["owner"], "cap": cl["cap"],
                                      "scope": cl["scope"], "limit": cl["limit"], "expires": None, "by": cid},
                         vis=(cl["grantee"], cl["owner"]))
                ctx.emit("contracts.note", {"id": cid, "clause": i, "grant": gid}, vis="internal")
            elif t == "shares":
                biz = ctx.world.modules.get("business")
                if biz is None:
                    raise InvalidAction("no company registry in this world")
                biz.transfer_shares(ctx, cl["org"], cl["from"], cl["to"], cl["shares"])
            elif t == "deliverable":
                ctx.schedule(ctx.now + cl["due_in_s"], "contracts.check", {"contract": cid, "clause": i}, prio=36,
                             key=f"chk|{cid}|{i}")

    def end(self, ctx: Ctx, cid: str, status: str, reason: str, by: str | None = None) -> None:
        st = ctx.state
        c = st.get("contracts", cid)
        if c["status"] not in ("active", "terminating", "proposed"):
            return
        was_active = c["status"] in ("active", "terminating")
        ctx.emit("contracts.status", {"id": cid, "status": status, "reason": reason[:300],
                                      **({"by": by} if by else {})},
                 vis="public" if status == "breached" else tuple(c["parties"]))
        for i, cl in enumerate(c["clauses"]):
            ctx.unschedule(f"pay|{cid}|{i}")
            ctx.unschedule(f"chk|{cid}|{i}")
            if not was_active:
                continue
            if cl["type"] == "membership":
                idm = ident(ctx.world)
                if idm.role(st, cl["org"], cl["person"]) in ("member", "admin"):
                    idm.remove_member(ctx, cl["org"], cl["person"], f"contract {status}")
            elif cl["type"] == "grant":
                gid = c["grants"].get(str(i))
                if gid and st.has("grants", gid):
                    ctx.emit("id.revoke", {"id": gid}, vis=(cl["grantee"], cl["owner"]))
        ctx.unschedule(f"end|{cid}")
        hook = ctx.world.hooks.get("on_contract_ended")
        if hook and was_active:
            hook(ctx, cid, status)

    def add_claim(self, ctx: Ctx, debtor: str, creditor: str, amount: int, source: str) -> str:
        if isinstance(amount, bool) or not isinstance(amount, numbers.Integral) or not 0 < amount <= MAX_AMOUNT:
            raise InvalidAction("a claim is a positive amount within the ledger's range")
        clid = ctx.state.new_id("claim")
        ctx.emit("contracts.claim", {"id": clid, "debtor": debtor, "creditor": creditor, "amount": int(amount),
                                     "source": source}, vis=(debtor, creditor))
        return clid

    # ------------------------------------------------------------------ processes
    def _unexecutable(self, ctx, contract, clause, err) -> None:
        """A clause that cannot be executed at all (a record written before clause numbers were validated) ends
        the contract instead of leaving it active for ever with a dead schedule."""
        c = ctx.state.get("contracts", contract)
        if c is not None and c["status"] in ("active", "terminating"):
            ctx.emit("contracts.clause_failed", {"id": contract, "clause": clause,
                                                 "error": f"{type(err).__name__}: {err}"[:200]}, vis=tuple(c["parties"]))
            self.end(ctx, contract, "terminated", f"clause {clause} could not be executed")

    @process("pay")
    def pr_pay(self, ctx, contract, clause):
        err = contain(ctx, self._pay, ctx, contract, clause)
        if err is not None:
            self._unexecutable(ctx, contract, clause, err)

    def _pay(self, ctx, contract, clause):
        st = ctx.state
        c = st.get("contracts", contract)
        if c is None or c["status"] not in ("active", "terminating"):
            return
        cl = c["clauses"][clause]
        bank = ctx.world.modules["bank"]
        src = c["accounts"].get(str(clause))
        dst = bank.primary_account(st, cl["to"])
        paid = c["paid"].get(str(clause), 0)
        try:
            if src is None or dst is None:
                raise Forbidden("missing account")
            bank.move(ctx, src, dst, cl["amount"], "contract", f"{c['title']} (clause {clause})", ref=contract)
        except CrucibleError as e:
            self.add_claim(ctx, cl["from"], cl["to"], cl["amount"] + self._accelerated(c, clause),
                           f"contract {contract} clause {clause}")
            # the breach is a public record: it says that the payment failed and (coarsely) why, never the bank's
            # own message, which names the debtor's account and its available funds
            why = "insufficient funds" if isinstance(e, InsufficientFunds) else "the paying account is not available"
            self.end(ctx, contract, "breached", f"payment failed: {why}", by=cl["from"])
            self._penalties(ctx, contract, breacher=cl["from"])
            return
        ctx.emit("contracts.paid", {"id": contract, "clause": clause, "n": paid + 1}, vis=tuple(c["parties"]))
        if cl["every_s"] and (cl["count"] is None or paid + 1 < cl["count"]):
            ctx.schedule(ctx.now + int(cl["every_s"]), "contracts.pay", {"contract": contract, "clause": clause},
                         prio=35, key=f"pay|{contract}|{clause}")
        self._maybe_complete(ctx, contract)

    @staticmethod
    def _accelerated(c, clause: int) -> int:
        """What else falls due when the payment of `clause` is missed. Nothing, for a one-way obligation (a fee,
        a retainer: the missed instalment is claimed and the contract ends). But a party that has RECEIVED money
        from its creditor under the same contract (a loan's disbursement) owes all of it back: every remaining
        instalment it promised that creditor in a fixed number of payments falls due with the one it missed -
        otherwise missing the first repayment would be the cheapest way to settle a loan."""
        cl = c["clauses"][clause]
        got = any(x["type"] == "payment" and x["from"] == cl["to"] and x["to"] == cl["from"]
                  and c["paid"].get(str(i), 0) > 0 for i, x in enumerate(c["clauses"]))
        if not got:
            return 0
        rest = sum(remaining_fixed(c, i) for i, x in enumerate(c["clauses"])
                   if x["type"] == "payment" and x["from"] == cl["from"] and x["to"] == cl["to"])
        return max(0, rest - int(cl["amount"])) if remaining_fixed(c, clause) else rest

    @process("check")
    def pr_check(self, ctx, contract, clause):
        err = contain(ctx, self._check, ctx, contract, clause)
        if err is not None:
            self._unexecutable(ctx, contract, clause, err)

    def _check(self, ctx, contract, clause):
        from ..goals.metrics import evaluate_predicate
        st = ctx.state
        c = st.get("contracts", contract)
        if c is None or c["status"] not in ("active", "terminating"):
            return
        cl = c["clauses"][clause]
        try:
            ok = bool(evaluate_predicate(ctx.world, cl["predicate"]))
        except Exception:  # noqa: BLE001 - a predicate that cannot be evaluated has not been met
            ok = False
        ctx.emit("contracts.deliverable", {"id": contract, "clause": clause, "ok": ok}, vis=tuple(c["parties"]))
        if not ok:
            self.end(ctx, contract, "breached", f"deliverable {clause} not met", by=cl["by"])
            self._penalties(ctx, contract, only=clause)
        else:
            self._maybe_complete(ctx, contract)

    def _penalties(self, ctx, cid, only=None, breacher=None):
        """Charge the penalty of failed deliverable `only`; or, when `breacher` stopped paying, the penalties of
        the deliverables that party itself still owed. Money moves only between parties (validation guarantees it
        for new contracts; records written before that rule are checked again here)."""
        st = ctx.state
        c = st.get("contracts", cid)
        bank = ctx.world.modules["bank"]
        for i, cl in enumerate(c["clauses"]):
            if cl["type"] != "deliverable" or not cl.get("penalty") or (only is not None and i != only):
                continue
            if breacher is not None and (cl["by"] != breacher or str(i) in c["delivered"]):
                continue
            pen = cl["penalty"]
            amt = pen.get("amount")
            if pen["from"] not in c["parties"] or pen["to"] not in c["parties"] or isinstance(amt, bool) \
                    or not isinstance(amt, numbers.Integral) or not 0 < amt <= MAX_AMOUNT:
                continue
            src, dst = bank.primary_account(st, pen["from"]), bank.primary_account(st, pen["to"])
            try:
                if src is None or dst is None:
                    raise Forbidden("missing account")
                bank.move(ctx, src, dst, pen["amount"], "penalty", f"contract {cid} clause {i}", ref=cid)
            except CrucibleError:
                self.add_claim(ctx, pen["from"], pen["to"], pen["amount"], f"penalty {cid} clause {i}")

    def _maybe_complete(self, ctx, cid):
        c = ctx.state.get("contracts", cid)
        if c["status"] != "active":
            return
        for i, cl in enumerate(c["clauses"]):
            if cl["type"] == "payment":
                total = 1 if not cl["every_s"] else cl["count"]
                if total is None or c["paid"].get(str(i), 0) < total:
                    return
            if cl["type"] == "deliverable" and str(i) not in c["delivered"]:
                return
            if cl["type"] in ("membership", "grant"):
                return  # ongoing relationships end by termination
        self.end(ctx, cid, "completed", "all obligations met")

    @process("end")
    def pr_end(self, ctx, contract):
        c = ctx.state.get("contracts", contract)
        if c is not None and c["status"] == "terminating":
            self.end(ctx, contract, "terminated", "notice period over")

    # ------------------------------------------------------------------ endpoints
    @endpoint("propose", S.obj({"parties": S.s_arr(S.s_str(maxLength=40), maxItems=8, minItems=2),
                                "clauses": S.s_arr(CLAUSE, maxItems=MAX_CLAUSES, minItems=1),
                                "title": S.s_str(maxLength=120), "text": S.s_str(maxLength=4000),
                                "as_org": S.s_str("propose on behalf of an org you administer", maxLength=40),
                                "expires_in_s": S.s_int(minimum=3600, maximum=90 * 86400)},
                               ["parties", "clauses", "title"]), cost_s=120)
    def ep_propose(self, ctx, parties, clauses, title, text="", as_org=None, expires_in_s=7 * 86400):
        """Propose a contract built from clauses (payment, membership, grant, shares, deliverable, notice). It takes
        effect when every party signs. Payment clauses debit the account each payer designates when signing."""
        proposer = ctx.actor
        if as_org:
            proposer = ident(ctx.world).resolve_handle(ctx.state, as_org)
            ident(ctx.world).authorize(ctx, proposer, "org.admin")
        cid = self.propose(ctx, proposer, parties, clauses, title, text, expires_in_s)
        self.sign(ctx, cid, proposer, None)
        return {"contract": cid}

    @endpoint("sign", S.obj({"contract": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40),
                             "accounts": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 40},
                                          "description": "clause index -> account to pay from"}}, ["contract"]), cost_s=60)
    def ep_sign(self, ctx, contract, as_org=None, accounts=None):
        """Sign a proposed contract (designate accounts for the payment clauses you owe). It activates when all parties signed."""
        party = ctx.actor
        if as_org:
            party = ident(ctx.world).resolve_handle(ctx.state, as_org)
            ident(ctx.world).authorize(ctx, party, "org.admin")
        return self.sign(ctx, contract, party, accounts)

    @endpoint("reject", S.obj({"contract": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40)}, ["contract"]))
    def ep_reject(self, ctx, contract, as_org=None):
        """Reject a contract proposal addressed to you."""
        party = ident(ctx.world).resolve_handle(ctx.state, as_org) if as_org else ctx.actor
        if as_org:
            ident(ctx.world).authorize(ctx, party, "org.admin")
        c = ctx.state.get("contracts", contract)
        if c is None or c["status"] != "proposed" or party not in c["parties"]:
            raise NotFound("no open proposal for you")
        self.end(ctx, contract, "rejected", f"rejected by {party}")
        return {"ok": True}

    @endpoint("terminate", S.obj({"contract": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40)}, ["contract"]))
    def ep_terminate(self, ctx, contract, as_org=None):
        """Terminate with the contract's notice period (if it has one); otherwise all parties must request it."""
        party = ident(ctx.world).resolve_handle(ctx.state, as_org) if as_org else ctx.actor
        if as_org:
            ident(ctx.world).authorize(ctx, party, "org.admin")
        c = ctx.state.get("contracts", contract)
        if c is None or party not in c["parties"] or c["status"] != "active":
            raise NotFound("no active contract for you")
        notice = next((cl["seconds"] for cl in c["clauses"] if cl["type"] == "notice"), None)
        if notice is not None:
            ctx.emit("contracts.status", {"id": contract, "status": "terminating", "reason": f"notice by {party}"},
                     vis=tuple(c["parties"]))
            ctx.schedule(ctx.now + notice, "contracts.end", {"contract": contract}, prio=37, key=f"end|{contract}")
            return {"status": "terminating", "ends": ctx.now + notice}
        ctx.emit("contracts.term_request", {"id": contract, "party": party}, vis=tuple(c["parties"]))
        c = ctx.state.get("contracts", contract)
        if set(c["term_requests"]) == set(c["parties"]):
            self.end(ctx, contract, "terminated", "mutual")
            return {"status": "terminated"}
        return {"status": "termination requested"}

    @endpoint("list", S.obj({"status": S.s_enum(["proposed", "active", "terminating", "completed", "breached",
                                                 "terminated", "rejected"])}), read_only=True)
    def ep_list(self, ctx, status=None):
        """Contracts you are a party to, optionally filtered by status."""
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for row in ctx.state.find("contract_parties", "party", p):
                c = ctx.state.tables["contracts"][row["contract"]]
                if status is None or c["status"] == status:
                    out.append({"contract": row["contract"], "title": c["title"], "status": c["status"],
                                "parties": c["parties"], "signed": c["signed"]})
        return out[-50:]

    @endpoint("view", S.obj({"contract": S.s_str(maxLength=40)}, ["contract"]), read_only=True)
    def ep_view(self, ctx, contract):
        """Full text and state of a contract you are a party to. A breach is a public fact: anyone may read who
        breached which contract and what that party owed under it (no text, clauses or accounts)."""
        c = ctx.state.get("contracts", contract)
        if c is None:
            raise NotFound("no contract")
        if not (ctx.world.principals_of(ctx.actor) & set(c["parties"])):
            if c["status"] != "breached":
                raise Forbidden("not a party")
            return public_view(contract, c)
        return dict(c, id=contract)

    @endpoint("claims", S.obj(), read_only=True)
    def ep_claims(self, ctx):
        """Open and settled claims (unpaid obligations) where you are debtor or creditor."""
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for f in ("debtor", "creditor"):
                for cid in ctx.state.find_ids("claims", f, p):
                    out.append(dict(ctx.state.tables["claims"][cid], id=cid))
        return out

    @endpoint("settle_claim", S.obj({"claim": S.s_str(maxLength=40), "account": S.s_str(maxLength=40)},
                                    ["claim", "account"]))
    def ep_settle(self, ctx, claim, account):
        """Pay an open claim you owe from one of your accounts."""
        cl = ctx.state.get("claims", claim)
        if cl is None or cl["status"] != "open":
            raise NotFound("no open claim")
        bank = ctx.world.modules["bank"]
        if bank.account_owner(ctx.state, account) != cl["debtor"]:
            raise Forbidden("pay from the debtor's account")
        ident(ctx.world).authorize(ctx, cl["debtor"], "bank.spend", amount=cl["amount"], scope=account)
        bank.move(ctx, account, bank.primary_account(ctx.state, cl["creditor"]), cl["amount"], "claim", f"claim {claim}")
        ctx.emit("contracts.claim_status", {"id": claim, "status": "settled", "paid": cl["amount"]},
                 vis=(cl["debtor"], cl["creditor"]))
        return {"ok": True}

    # ------------------------------------------------------------------ appliers
    @applier("contracts.proposed")
    def ap_proposed(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("contracts", p["id"], {"proposer": p["proposer"], "parties": p["parties"], "clauses": p["clauses"],
                                        "title": p["title"], "text": p["text"], "expires": p["expires"],
                                        "tags": p["tags"], "status": "proposed", "signed": [], "accounts": {},
                                        "paid": {}, "delivered": [], "grants": {}, "term_requests": [],
                                        "t": ev.t, "employee": p["employee"], "employer": p["employer"]})
        for party in p["parties"]:
            st.insert("contract_parties", f"{p['id']}|{party}", {"party": party, "contract": p["id"]})

    @applier("contracts.signed")
    def ap_signed(self, ev):
        st = self.world.state
        c = st.tables["contracts"][ev.payload["id"]]
        st.patch("contracts", ev.payload["id"], signed=sorted(set(c["signed"]) | {ev.payload["party"]}),
                 accounts={**c["accounts"], **ev.payload["accounts"]})

    @applier("contracts.status")
    def ap_status(self, ev):
        self.world.state.patch("contracts", ev.payload["id"], status=ev.payload["status"],
                               status_reason=ev.payload["reason"], updated=ev.t,
                               **({"breached_by": ev.payload["by"]} if ev.payload.get("by") else {}))

    @applier("contracts.paid")
    def ap_paid(self, ev):
        st = self.world.state
        c = st.tables["contracts"][ev.payload["id"]]
        st.patch("contracts", ev.payload["id"], paid={**c["paid"], str(ev.payload["clause"]): ev.payload["n"]})

    @applier("contracts.deliverable")
    def ap_deliverable(self, ev):
        st = self.world.state
        c = st.tables["contracts"][ev.payload["id"]]
        if ev.payload["ok"]:
            st.patch("contracts", ev.payload["id"], delivered=sorted(set(c["delivered"]) | {str(ev.payload["clause"])}))

    @applier("contracts.clause_failed")
    def ap_clause_failed(self, ev):
        pass

    @applier("contracts.note")
    def ap_note(self, ev):
        st = self.world.state
        c = st.tables["contracts"][ev.payload["id"]]
        st.patch("contracts", ev.payload["id"], grants={**c["grants"], str(ev.payload["clause"]): ev.payload["grant"]})

    @applier("contracts.term_request")
    def ap_term_request(self, ev):
        st = self.world.state
        c = st.tables["contracts"][ev.payload["id"]]
        st.patch("contracts", ev.payload["id"], term_requests=sorted(set(c["term_requests"]) | {ev.payload["party"]}))

    @applier("contracts.claim")
    def ap_claim(self, ev):
        p = ev.payload
        self.world.state.insert("claims", p["id"], {"debtor": p["debtor"], "creditor": p["creditor"], "amount": p["amount"],
                                                   "source": p["source"], "status": "open", "t": ev.t, "paid": 0})

    @applier("contracts.claim_status")
    def ap_claim_status(self, ev):
        self.world.state.patch("claims", ev.payload["id"], status=ev.payload["status"], paid=ev.payload.get("paid", 0))

    def check_invariants(self, world):
        errs = []
        st = world.state
        for cid, c in st.tables["contracts"].items():
            if c["status"] == "active" and set(c["signed"]) != set(c["parties"]):
                errs.append(f"{cid} active without all signatures")
        return errs
