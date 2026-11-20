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
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

CLAUSE = {"type": "object", "additionalProperties": True}
MAX_CLAUSES = 12


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
        from ..goals.metrics import check_predicate
        st = ctx.state
        out = []
        if not clauses or len(clauses) > MAX_CLAUSES:
            raise InvalidAction(f"1-{MAX_CLAUSES} clauses required")
        R = lambda x: ident(ctx.world).resolve_handle(st, x)  # noqa: E731
        for i, c in enumerate(clauses):
            t = c.get("type")
            try:
                if t == "payment":
                    cl = {"type": t, "from": R(c["from"]), "to": R(c["to"]), "amount": int(c["amount"]),
                          "every_s": c.get("every_s"), "count": c.get("count"), "start_in_s": int(c.get("start_in_s", 0))}
                    if cl["amount"] <= 0 or cl["from"] == cl["to"]:
                        raise InvalidAction("payment needs a positive amount between different parties")
                    if cl["every_s"] is not None and int(cl["every_s"]) < 3600:
                        raise InvalidAction("recurring payments at most hourly")
                    if cl["count"] is not None and not (1 <= int(cl["count"]) <= 10_000):
                        raise InvalidAction("bad count")
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
                elif t == "shares":
                    cl = {"type": t, "org": R(c["org"]), "from": R(c["from"]), "to": R(c["to"]), "shares": int(c["shares"])}
                    if cl["from"] not in parties or cl["to"] not in parties or cl["shares"] <= 0:
                        raise InvalidAction("share transfers between parties, positive amounts")
                elif t == "deliverable":
                    errs = check_predicate(c.get("predicate"))
                    if errs:
                        raise InvalidAction("; ".join(errs))
                    pen = c.get("penalty")
                    cl = {"type": t, "by": R(c["by"]), "predicate": c["predicate"], "due_in_s": int(c["due_in_s"]),
                          "penalty": None if not pen else {"from": R(pen["from"]), "to": R(pen["to"]),
                                                           "amount": int(pen["amount"])}}
                    if cl["due_in_s"] < 3600:
                        raise InvalidAction("deliverables due at least an hour out")
                elif t == "notice":
                    cl = {"type": t, "seconds": int(c["seconds"])}
                else:
                    raise InvalidAction(f"unknown clause type {t!r}")
            except (KeyError, TypeError, ValueError) as e:
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
                ident(ctx.world).authorize(ctx, party, "bank.spend", scope=a)
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

    def end(self, ctx: Ctx, cid: str, status: str, reason: str) -> None:
        st = ctx.state
        c = st.get("contracts", cid)
        if c["status"] not in ("active", "terminating", "proposed"):
            return
        was_active = c["status"] in ("active", "terminating")
        ctx.emit("contracts.status", {"id": cid, "status": status, "reason": reason[:300]},
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
        clid = ctx.state.new_id("claim")
        ctx.emit("contracts.claim", {"id": clid, "debtor": debtor, "creditor": creditor, "amount": int(amount),
                                     "source": source}, vis=(debtor, creditor))
        return clid

    # ------------------------------------------------------------------ processes
    @process("pay")
    def pr_pay(self, ctx, contract, clause):
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
            self.add_claim(ctx, cl["from"], cl["to"], cl["amount"], f"contract {contract} clause {clause}")
            self.end(ctx, contract, "breached", f"payment failed: {e.message}")
            self._penalties(ctx, contract)
            return
        ctx.emit("contracts.paid", {"id": contract, "clause": clause, "n": paid + 1}, vis=tuple(c["parties"]))
        if cl["every_s"] and (cl["count"] is None or paid + 1 < cl["count"]):
            ctx.schedule(ctx.now + int(cl["every_s"]), "contracts.pay", {"contract": contract, "clause": clause},
                         prio=35, key=f"pay|{contract}|{clause}")
        self._maybe_complete(ctx, contract)

    @process("check")
    def pr_check(self, ctx, contract, clause):
        from ..goals.metrics import evaluate_predicate
        st = ctx.state
        c = st.get("contracts", contract)
        if c is None or c["status"] not in ("active", "terminating"):
            return
        cl = c["clauses"][clause]
        ok = evaluate_predicate(ctx.world, cl["predicate"])
        ctx.emit("contracts.deliverable", {"id": contract, "clause": clause, "ok": ok}, vis=tuple(c["parties"]))
        if not ok:
            self.end(ctx, contract, "breached", f"deliverable {clause} not met")
            self._penalties(ctx, contract, only=clause)
        else:
            self._maybe_complete(ctx, contract)

    def _penalties(self, ctx, cid, only=None):
        st = ctx.state
        c = st.get("contracts", cid)
        bank = ctx.world.modules["bank"]
        for i, cl in enumerate(c["clauses"]):
            if cl["type"] != "deliverable" or not cl.get("penalty") or (only is not None and i != only):
                continue
            pen = cl["penalty"]
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
        party = ctx.actor
        if as_org:
            party = ident(ctx.world).resolve_handle(ctx.state, as_org)
            ident(ctx.world).authorize(ctx, party, "org.admin")
        return self.sign(ctx, contract, party, accounts)

    @endpoint("reject", S.obj({"contract": S.s_str(maxLength=40), "as_org": S.s_str(maxLength=40)}, ["contract"]))
    def ep_reject(self, ctx, contract, as_org=None):
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
        c = ctx.state.get("contracts", contract)
        if c is None:
            raise NotFound("no contract")
        if not (ctx.world.principals_of(ctx.actor) & set(c["parties"])) and c["status"] != "breached":
            raise Forbidden("not a party")
        return dict(c, id=contract)

    @endpoint("claims", S.obj(), read_only=True)
    def ep_claims(self, ctx):
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for f in ("debtor", "creditor"):
                for cid in ctx.state.find_ids("claims", f, p):
                    out.append(dict(ctx.state.tables["claims"][cid], id=cid))
        return out

    @endpoint("settle_claim", S.obj({"claim": S.s_str(maxLength=40), "account": S.s_str(maxLength=40)},
                                    ["claim", "account"]))
    def ep_settle(self, ctx, claim, account):
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
                               status_reason=ev.payload["reason"], updated=ev.t)

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
