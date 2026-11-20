"""Company registry and administration (host `biz`): registration, cap tables, recurring fees, insolvency.

* `biz.register` turns an organisation into a registered company (fee; public registry entry indexed by search;
  1,000,000 shares issued to the registering owner). Only registered companies can open bank accounts, list on
  the market as organisations, or operate compute.
* An annual registration fee is debited from the company's primary account.
* Insolvency: a company with open claims (unpaid contract payments or penalties, unpaid fees) that it cannot pay
  for `insolvency_grace_s` is declared bankrupt. Bankruptcy is a real procedure: accounts are frozen, the remaining
  cash is distributed pro rata to claimants, services and instances are terminated, contracts end, listings close,
  non-owner members are released, and the event is public.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, process, register_module
from .identity import ident

SHARES = 1_000_000


@register_module
class Business(Module):
    name = "business"
    host = "biz"
    depends = ("identity", "bank", "contracts")

    def install(self, world):
        st = world.state
        st.declare_table("companies", index=("status",))
        st.declare_table("shares", index=("org", "holder"))
        world.hooks["is_registered_company"] = self.is_registered

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 86400, "business.daily", {}, prio=60, period=86400, key="business.daily")

    def p(self, k, d):
        return self.world.spec.params.get(k, self.config.get(k, d))

    def is_registered(self, state, org: str) -> bool:
        c = state.get("companies", org)
        return c is not None and c["status"] == "registered"

    # ------------------------------------------------------------------ internal API
    def register_company(self, ctx, org: str, owner: str | None, sector: str = "", fee: int | None = None,
                         pay_from: str | None = None) -> None:
        fee = int(self.p("registration_fee_cents", 50_000)) if fee is None else fee
        if fee and pay_from:
            ctx.world.modules["bank"].move(ctx, pay_from, "acct_fees", fee, "fee", f"company registration {org}")
        o = ctx.state.get("orgs", org)
        ctx.emit("biz.registered", {"org": org, "sector": sector[:80], "name": o["name"], "holder": owner},
                 vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            search.index(ctx, f"company:{org}", "company", o["name"], f"{o['name']} {sector} {o.get('bio', '')}", org, ref=org)

    def transfer_shares(self, ctx, org, src, dst, n):
        have = ctx.state.get("shares", f"{org}|{src}")
        if have is None or have["n"] < n:
            raise InvalidAction(f"{src} holds fewer than {n} shares of {org}")
        ctx.emit("biz.shares", {"org": org, "from": src, "to": dst, "n": int(n)}, vis="public")
        hold = ctx.state.get("shares", f"{org}|{dst}")
        if dst.startswith("p:") and hold and hold["n"] * 2 > SHARES:
            idm = ident(ctx.world)
            if idm.role(ctx.state, org, dst) != "owner":   # a majority shareholder controls the company
                if idm.role(ctx.state, org, dst) is None:
                    idm.add_member(ctx, org, dst, "owner", "majority shareholder")
                else:
                    ctx.emit("id.role", {"org": org, "person": dst, "role": "owner"}, vis=(dst, org))

    # ------------------------------------------------------------------ endpoints
    @endpoint("register", S.obj({"org": S.s_str(maxLength=40), "account": S.s_str("pays the fee", maxLength=40),
                                 "sector": S.s_str(maxLength=80)}, ["org", "account"]), cost_s=600)
    def ep_register(self, ctx, org, account, sector=""):
        """Register an organisation as a company (fee from your account). Required to trade as an organisation."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        if ident(ctx.world).role(ctx.state, org, ctx.actor) != "owner":
            raise Forbidden("only owners register companies")
        if ctx.state.has("companies", org):
            raise Conflict("already registered")
        bank = ctx.world.modules["bank"]
        ident(ctx.world).authorize(ctx, bank.account_owner(ctx.state, account), "bank.spend", scope=account)
        self.register_company(ctx, org, ctx.actor, sector, pay_from=account)
        return {"company": org, "shares_issued": SHARES, "holder": ctx.actor}

    @endpoint("company", S.obj({"org": S.s_str(maxLength=40)}, ["org"]), read_only=True)
    def ep_company(self, ctx, org):
        """Public registry entry: status, age, sector, shareholders, public breaches and bankruptcy."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        c = ctx.state.get("companies", org)
        if c is None:
            raise NotFound("not a registered company")
        holders = sorted(((r["holder"], r["n"]) for r in ctx.state.find("shares", "org", org) if r["n"] > 0),
                         key=lambda x: -x[1])
        breaches = sum(1 for _, k in ctx.state.scan("contracts") if k["status"] == "breached" and org in k["parties"]
                       and any(cl.get("from") == org for cl in k["clauses"] if cl["type"] == "payment"))
        return {"org": org, "status": c["status"], "since": c["t"], "sector": c["sector"],
                "shareholders": [{"holder": h, "shares": n} for h, n in holders[:10]], "payment_breaches": breaches}

    @endpoint("registry", S.obj({"status": S.s_enum(["registered", "bankrupt", "dissolved"]),
                                 "limit": S.s_int(minimum=1, maximum=100)}), read_only=True)
    def ep_registry(self, ctx, status="registered", limit=50):
        rows = []
        for org in ctx.state.find_ids("companies", "status", status)[-limit:]:
            c = ctx.state.tables["companies"][org]
            rows.append({"org": org, "name": c["name"], "sector": c["sector"], "since": c["t"]})
        return rows

    @endpoint("dissolve", S.obj({"org": S.s_str(maxLength=40)}, ["org"]), cost_s=600)
    def ep_dissolve(self, ctx, org):
        """Voluntarily wind up a solvent company: claims are paid first, the rest goes to shareholders."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        if ident(ctx.world).role(ctx.state, org, ctx.actor) != "owner":
            raise Forbidden("only owners dissolve")
        if not self.is_registered(ctx.state, org):
            raise NotFound("not a registered company")
        self._wind_up(ctx, org, "dissolved")
        return {"ok": True}

    # ------------------------------------------------------------------ insolvency
    def open_claims(self, state, org) -> list:
        return [(cid, c) for cid in state.find_ids("claims", "debtor", org)
                for c in [state.tables["claims"][cid]] if c["status"] == "open"]

    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        grace = int(self.p("insolvency_grace_s", 14 * 86400))
        annual = int(self.p("annual_fee_cents", 20_000))
        contracts = ctx.world.modules["contracts"]
        for org in st.find_ids("companies", "status", "registered"):
            c = st.tables["companies"][org]
            acct = bank.primary_account(st, org)
            if annual and (ctx.now - c["t"]) // (365 * 86400) > c["fees_paid"] - 1 and ctx.now - c["t"] >= 365 * 86400:
                try:
                    if acct is None:
                        raise Forbidden("no account")
                    bank.move(ctx, acct, "acct_fees", annual, "fee", "annual registration fee")
                    ctx.emit("biz.fee", {"org": org}, vis=(org,))
                except CrucibleError:
                    contracts.add_claim(ctx, org, "sys", annual, "annual registration fee")
                    ctx.emit("biz.fee", {"org": org, "unpaid": True}, vis=(org,))
            claims = self.open_claims(st, org)
            if not claims:
                if c["distress_since"] is not None:
                    ctx.emit("biz.distress", {"org": org, "since": None}, vis="public")
                continue
            owed = sum(cl["amount"] for _, cl in claims)
            cash = bank.available(st, acct) if acct else 0
            if cash >= owed and acct is not None:   # pay claims automatically when cash exists (creditor priority)
                for cid, cl in claims:
                    dst = bank.primary_account(st, cl["creditor"]) if cl["creditor"] != "sys" else "acct_fees"
                    if dst is None:
                        continue
                    bank.move(ctx, acct, dst, cl["amount"], "claim", f"claim {cid}")
                    ctx.emit("contracts.claim_status", {"id": cid, "status": "settled", "paid": cl["amount"]},
                             vis=(org, cl["creditor"]))
                continue
            if c["distress_since"] is None:
                ctx.emit("biz.distress", {"org": org, "since": ctx.now, "owed": owed}, vis="public")
            elif ctx.now - c["distress_since"] >= grace:
                self._wind_up(ctx, org, "bankrupt")

    def _wind_up(self, ctx, org, status):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        accts = [a for a in st.find_ids("accounts", "owner", org) if st.tables["accounts"][a]["status"] == "open"]
        cash = sum(max(0, st.tables["accounts"][a]["balance"]) for a in accts)
        claims = self.open_claims(st, org)
        owed = sum(c["amount"] for _, c in claims)
        payout_ratio = min(1.0, cash / owed) if owed else 0.0
        # pay claimants pro rata from the accounts in id order
        for cid, cl in claims:
            due = int(cl["amount"] * payout_ratio)
            dst = bank.primary_account(st, cl["creditor"]) if cl["creditor"] != "sys" else "acct_fees"
            for a in accts:
                if due <= 0 or dst is None:
                    break
                bal = st.tables["accounts"][a]["balance"]
                x = min(bal, due)
                if x > 0:
                    bank.move(ctx, a, dst, x, "liquidation", f"claim {cid}")
                    due -= x
            ctx.emit("contracts.claim_status", {"id": cid, "status": "settled" if payout_ratio >= 1 else "written_off",
                                                "paid": int(cl["amount"] * payout_ratio)}, vis=(org, cl["creditor"]))
        if status == "dissolved":   # remainder to shareholders pro rata
            holders = [(r["holder"], r["n"]) for r in st.find("shares", "org", org) if r["n"] > 0]
            total = sum(n for _, n in holders) or 1
            remaining = sum(max(0, st.tables["accounts"][a]["balance"]) for a in accts)
            for h, n in holders:
                share = remaining * n // total
                dst = bank.primary_account(st, h)
                for a in accts:
                    bal = st.tables["accounts"][a]["balance"]
                    x = min(bal, share)
                    if x > 0 and dst is not None and dst != a:
                        bank.move(ctx, a, dst, x, "distribution", f"wind-up of {org}")
                        share -= x
        contracts = ctx.world.modules["contracts"]
        for row in st.find("contract_parties", "party", org):
            k = st.tables["contracts"][row["contract"]]
            if k["status"] in ("active", "terminating", "proposed"):
                contracts.end(ctx, row["contract"], "terminated", f"{org} {status}")
        cloud = ctx.world.modules.get("cloud")
        if cloud is not None:
            cloud.on_bankrupt(ctx, org)
        market = ctx.world.modules.get("market")
        if market is not None:
            for lid in st.find_ids("listings", "seller", org):
                if st.tables["listings"][lid]["status"] == "active":
                    ctx.emit("market.listing_update", {"id": lid, "changes": {"status": "closed"}}, vis="public")
        idm = ident(ctx.world)
        for m in st.find("members", "org", org):
            if m["role"] != "owner":
                idm.remove_member(ctx, org, m["person"], f"company {status}")
        for a in accts:
            ctx.emit("bank.account_status", {"id": a, "status": "frozen"}, vis=(org,))
        ctx.emit("biz.status", {"org": org, "status": status, "cash": cash, "owed": owed}, vis="public")
        ctx.emit("id.org_status", {"id": org, "status": status}, vis="public")
        hook = ctx.world.hooks.get("on_company_ended")
        if hook:
            hook(ctx, org, status)

    # ------------------------------------------------------------------ appliers
    @applier("biz.registered")
    def ap_registered(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("companies", p["org"], {"name": p["name"], "sector": p["sector"], "status": "registered", "t": ev.t,
                                         "fees_paid": 1, "distress_since": None})
        if p["holder"]:
            st.put("shares", f"{p['org']}|{p['holder']}", {"org": p["org"], "holder": p["holder"], "n": SHARES})

    @applier("biz.shares")
    def ap_shares(self, ev):
        p = ev.payload
        st = self.world.state
        a = st.tables["shares"][f"{p['org']}|{p['from']}"]
        st.patch("shares", f"{p['org']}|{p['from']}", n=a["n"] - p["n"])
        b = st.tables["shares"].get(f"{p['org']}|{p['to']}")
        st.put("shares", f"{p['org']}|{p['to']}", {"org": p["org"], "holder": p["to"], "n": (b["n"] if b else 0) + p["n"]})

    @applier("biz.fee")
    def ap_fee(self, ev):
        c = self.world.state.tables["companies"][ev.payload["org"]]
        self.world.state.patch("companies", ev.payload["org"], fees_paid=c["fees_paid"] + 1)

    @applier("biz.distress")
    def ap_distress(self, ev):
        self.world.state.patch("companies", ev.payload["org"], distress_since=ev.payload["since"])

    @applier("biz.status")
    def ap_status(self, ev):
        self.world.state.patch("companies", ev.payload["org"], status=ev.payload["status"], ended=ev.t)

    def check_invariants(self, world):
        st = world.state
        errs = []
        tot: dict[str, int] = {}
        for r in st.tables["shares"].values():
            if r["n"] < 0:
                errs.append(f"negative shares {r}")
            tot[r["org"]] = tot.get(r["org"], 0) + r["n"]
        for org, n in tot.items():
            if n != SHARES:
                errs.append(f"{org} has {n} shares outstanding, expected {SHARES}")
        return errs
