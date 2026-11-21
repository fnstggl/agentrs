"""Negotiation (module `deal`, host `deal`): structured offers, counteroffers and binding agreements, plus public
reputation derived from how agreements actually ended.

A *deal* is an offer from one principal to another: a set of contract clauses (the same composable vocabulary as
`contracts`: payment, membership, grant, shares, deliverable, notice) written with the placeholders `$me` (the
proposer) and `$you` (the recipient), plus a free-text message. The recipient may accept, reject or counter with
different terms; a counter is a new offer in the same thread, so negotiations have a persistent history. Accepting
creates and activates the contract immediately - the proposer consented when it made the offer (and named the
accounts it pays from), the acceptor consents by accepting. Nothing is agreed in text alone: only accepted terms
become commitments, and the contract service enforces them (scheduled payments, deliverable checks, breaches,
claims).

Reputation (`deal.reputation`) is computed from authoritative records: contracts completed, breached (and as which
party), claims outstanding, deals accepted/rejected, account age. Simulated people and firms consult it, alongside
their own private memory of each counterparty.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

CLAUSE = {"type": "object", "additionalProperties": True}


def _subst(x, me, you):
    if isinstance(x, str):
        return me if x == "$me" else you if x == "$you" else x
    if isinstance(x, dict):
        return {k: _subst(v, me, you) for k, v in x.items()}
    if isinstance(x, list):
        return [_subst(v, me, you) for v in x]
    return x


@register_module
class Deals(Module):
    name = "deal"
    depends = ("identity", "bank", "contracts")

    def install(self, world):
        st = world.state
        st.declare_table("deals", index=("from", "to", "status", "thread"))
        world.hooks["deals"] = self
        prev = world.hooks.get("on_contract_ended")

        def ended(ctx, cid, status):
            if prev:
                prev(ctx, cid, status)
            self.on_contract_ended(ctx, cid, status)
        world.hooks["on_contract_ended"] = ended

    # ------------------------------------------------------------------ helpers
    def party(self, ctx, as_org):
        if not as_org:
            return ctx.actor
        org = ident(ctx.world).resolve_handle(ctx.state, as_org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        return org

    def concrete(self, ctx, terms: list, me: str, you: str) -> list:
        cl = _subst(terms, me, you)
        ctx.world.modules["contracts"].validate(ctx, sorted({me, you} | self._extra_parties(cl)), cl)
        return cl

    @staticmethod
    def _extra_parties(clauses) -> set:
        out = set()
        for c in clauses:
            if c.get("type") == "membership":
                out |= {c.get("person"), c.get("org")}
        return {x for x in out if isinstance(x, str)}

    # ------------------------------------------------------------------ endpoints
    @endpoint("propose", S.obj({
        "to": S.s_str("recipient principal or handle", maxLength=60),
        "terms": S.s_arr(CLAUSE, "contract clauses using $me / $you", minItems=1, maxItems=12),
        "title": S.s_str(maxLength=120), "message": S.s_str(maxLength=4000),
        "as_org": S.s_str("make the offer for an organisation you administer", maxLength=40),
        "accounts": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 40},
                     "description": "clause index -> your account for payments you owe"},
        "expires_in_s": S.s_int(minimum=3600, maximum=30 * 86400)}, ["to", "terms", "title"]), cost_s=60)
    def ep_propose(self, ctx, to, terms, title, message="", as_org=None, accounts=None, expires_in_s=3 * 86400):
        """Make an offer. The terms are validated now (as concrete contract clauses); nothing is binding until the
        recipient accepts."""
        st = ctx.state
        me = self.party(ctx, as_org)
        you = ident(ctx.world).resolve_handle(st, to)
        if not st.has("principals", you) and not st.has("orgs", you):
            raise NotFound("no such recipient " + to)
        if you == me:
            raise InvalidAction("cannot negotiate with yourself")
        self.concrete(ctx, terms, me, you)
        return {"deal": self._offer(ctx, me, you, terms, title, message, accounts or {}, expires_in_s, None)}

    def _offer(self, ctx, me, you, terms, title, message, accounts, expires_in_s, parent) -> str:
        st = ctx.state
        did = st.new_id("deal")
        p = st.get("deals", parent) if parent else None
        thread = p["thread"] if p else did
        rnd = (p["round"] + 1) if p else 0
        ctx.emit("deal.offer", {"id": did, "from": me, "to": you, "terms": terms, "title": title[:120],
                                "message": message[:4000], "accounts": accounts, "expires": ctx.now + int(expires_in_s),
                                "parent": parent, "thread": thread, "round": rnd, "by": ctx.actor},
                 vis=(me, you))
        hook = ctx.world.hooks.get("on_deal_offered")
        if hook:
            hook(ctx, did, you)
        return did

    def _mine(self, ctx, d, side="to"):
        who = d[side]
        if ctx.actor != who and not ctx.world.may_act_as(ctx.actor, who):
            raise Forbidden("not your deal")
        return who

    @endpoint("respond", S.obj({
        "deal": S.s_str(maxLength=40), "action": S.s_enum(["accept", "reject", "counter"]),
        "terms": S.s_arr(CLAUSE, "for a counter: your terms ($me = you now)", maxItems=12),
        "message": S.s_str(maxLength=4000),
        "accounts": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 40}}},
        ["deal", "action"]), cost_s=60)
    def ep_respond(self, ctx, deal, action, terms=None, message="", accounts=None):
        """Accept (creates the binding contract now), reject, or counter with different terms."""
        st = ctx.state
        d = st.get("deals", deal)
        if d is None or d["status"] != "open":
            raise NotFound("no open deal " + deal)
        who = self._mine(ctx, d)
        if ctx.now > d["expires"]:
            ctx.emit("deal.status", {"id": deal, "status": "expired"}, vis=(d["from"], d["to"]))
            raise Conflict("offer expired")
        if action == "reject":
            ctx.emit("deal.status", {"id": deal, "status": "rejected", "message": message[:1000]}, vis=(d["from"], d["to"]))
            return {"status": "rejected"}
        if action == "counter":
            if not terms:
                raise InvalidAction("a counter needs terms")
            self.concrete(ctx, terms, who, d["from"])
            ctx.emit("deal.status", {"id": deal, "status": "countered", "message": ""}, vis=(d["from"], d["to"]))
            nd = self._offer(ctx, who, d["from"], terms, d["title"], message, accounts or {}, 3 * 86400, deal)
            return {"status": "countered", "deal": nd}
        cid = self.accept(ctx, deal, who, accounts or {})
        return {"status": "accepted", "contract": cid}

    def accept(self, ctx, deal, who, accounts) -> str:
        st = ctx.state
        d = st.tables["deals"][deal]
        contracts = ctx.world.modules["contracts"]
        clauses = _subst(d["terms"], d["from"], d["to"])
        parties = sorted({d["from"], d["to"]} | self._extra_parties(clauses))

        def as_proposer(sub):
            cid = contracts.propose(sub, d["from"], parties, clauses, d["title"], d["message"], 7 * 86400, ("deal",))
            contracts.sign(sub, cid, d["from"], d["accounts"])
            return cid
        cid = ctx.run_as(d["from"], as_proposer)
        acc = {}
        for i, cl in enumerate(st.tables["contracts"][cid]["clauses"]):
            if cl["type"] == "payment" and cl["from"] == who:
                acc[str(i)] = accounts.get(str(i)) or ctx.world.modules["bank"].primary_account(st, who)
        ctx.run_as(who, lambda sub: contracts.sign(sub, cid, who, acc))
        for p in parties:            # any further party (e.g. the person in a membership clause) signs as well
            c = st.tables["contracts"][cid]
            if p not in c["signed"] and c["status"] == "proposed" and (p == ctx.actor or ctx.world.may_act_as(ctx.actor, p)):
                ctx.run_as(p, lambda sub, p=p: contracts.sign(sub, cid, p, {}))
        ctx.emit("deal.status", {"id": deal, "status": "accepted", "contract": cid}, vis=(d["from"], d["to"]))
        return cid

    @endpoint("withdraw", S.obj({"deal": S.s_str(maxLength=40)}, ["deal"]))
    def ep_withdraw(self, ctx, deal):
        """Withdraw an open offer you made."""
        d = ctx.state.get("deals", deal)
        if d is None or d["status"] != "open":
            raise NotFound("no open deal")
        self._mine(ctx, d, "from")
        ctx.emit("deal.status", {"id": deal, "status": "withdrawn"}, vis=(d["from"], d["to"]))
        return {"ok": True}

    @endpoint("inbox", S.obj({"status": S.s_enum(["open", "any"]), "as_org": S.s_str(maxLength=40)}), read_only=True)
    def ep_inbox(self, ctx, status="open", as_org=None):
        """Deals offered to you (or your organisation) and by you."""
        st = ctx.state
        me = self.party(ctx, as_org)
        out = []
        for side in ("to", "from"):
            for did in st.find_ids("deals", side, me):
                d = st.tables["deals"][did]
                if status == "open" and d["status"] != "open":
                    continue
                out.append({"deal": did, "direction": "received" if side == "to" else "sent",
                            **{k: d[k] for k in ("from", "to", "title", "terms", "message", "status", "round",
                                                 "thread", "expires", "contract")}})
        return out

    @endpoint("thread", S.obj({"deal": S.s_str(maxLength=40)}, ["deal"]), read_only=True)
    def ep_thread(self, ctx, deal):
        """The full negotiation history of a deal's thread."""
        st = ctx.state
        d = st.get("deals", deal)
        if d is None:
            raise NotFound("no deal")
        mine = ctx.world.principals_of(ctx.actor)
        if d["from"] not in mine and d["to"] not in mine:
            raise Forbidden("not your negotiation")
        return [{"deal": k, **{x: st.tables["deals"][k][x] for x in ("from", "to", "terms", "message", "status",
                                                                      "round", "t")}}
                for k in st.find_ids("deals", "thread", d["thread"])]

    def reputation(self, world, who: str) -> dict:
        st = world.state
        if world.spec.params.get("public_reputation", True) is False:
            return {"principal": who, "contracts_completed": 0, "contracts_breached_by": 0, "contracts_active": 0,
                    "claims_owed": 0, "debts_written_off": 0, "deals_made": 0, "deals_accepted_by_others": 0,
                    "deals_rejected_by_others": 0, "age_days": None, "breaches_public": [], "unavailable": True}
        out = {"principal": who, "contracts_completed": 0, "contracts_breached_by": 0, "contracts_active": 0,
               "claims_owed": 0, "deals_made": 0, "deals_accepted_by_others": 0, "deals_rejected_by_others": 0}
        for cp in st.find("contract_parties", "party", who):
            c = st.tables["contracts"].get(cp["contract"])
            if c is None:
                continue
            if c["status"] == "completed" or c["status"] == "terminated":
                out["contracts_completed"] += 1
            elif c["status"] == "active":
                out["contracts_active"] += 1
            elif c["status"] == "breached" and c.get("breached_by") == who:
                out["contracts_breached_by"] += 1
        out["claims_owed"] = sum(1 for c in st.find("claims", "debtor", who) if c["status"] == "open")
        out["debts_written_off"] = sum(1 for c in st.find("claims", "debtor", who) if c["status"] == "written_off")
        for did in st.find_ids("deals", "from", who):
            d = st.tables["deals"][did]
            out["deals_made"] += 1
            out["deals_accepted_by_others"] += d["status"] == "accepted"
            out["deals_rejected_by_others"] += d["status"] == "rejected"
        p = st.get("principals", who)
        out["age_days"] = round((world.now - p["created"]) / 86400, 1) if p else None
        out["breaches_public"] = list(st.meta.get(f"breaches:{who}", []))[-10:]
        return out

    @endpoint("reputation", S.obj({"principal": S.s_str(maxLength=60)}, ["principal"]), read_only=True)
    def ep_reputation(self, ctx, principal):
        """Public track record of a principal: agreements completed and breached, claims it owes, offers made."""
        return self.reputation(ctx.world, ident(ctx.world).resolve_handle(ctx.state, principal))

    def on_contract_ended(self, ctx, cid, status):
        c = ctx.state.tables["contracts"][cid]
        if status == "breached":
            who = c.get("breached_by")
            if who:
                ctx.emit("deal.breach", {"who": who, "contract": cid}, vis="public")
                hook = ctx.world.hooks.get("on_breach")
                if hook:
                    hook(ctx, cid, who)

    @process("expire")
    def pr_expire(self, ctx):
        for did in ctx.state.find_ids("deals", "status", "open"):
            d = ctx.state.tables["deals"][did]
            if d["expires"] <= ctx.now:
                ctx.emit("deal.status", {"id": did, "status": "expired"}, vis=(d["from"], d["to"]))

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 3600, "deal.expire", {}, prio=58, period=3600, key="deal.expire")

    # ------------------------------------------------------------------ appliers
    @applier("deal.offer")
    def ap_offer(self, ev):
        p = dict(ev.payload)
        p.update(status="open", t=ev.t, contract=None)
        self.world.state.insert("deals", p.pop("id"), p)

    @applier("deal.status")
    def ap_status(self, ev):
        p = ev.payload
        ch = {"status": p["status"]}
        if p.get("contract"):
            ch["contract"] = p["contract"]
        self.world.state.patch("deals", p["id"], **ch)

    @applier("deal.breach")
    def ap_breach(self, ev):
        k = f"breaches:{ev.payload['who']}"
        self.world.state.meta_set(k, list(self.world.state.meta.get(k, [])) + [{"contract": ev.payload["contract"],
                                                                                   "t": ev.t}])
