"""Business lifecycle (module `ventures`, host `venture`): formation, books, orderly wind-down, insolvency of
businesses with customers, and world-generated competitors.

Formation (`venture.form`) is the 4.0 path done in one step through the ordinary tools, as the founder: create the
organisation (`id.create_org`), register the company (`biz.register`, fee from the founder's account), open its
operating account and pay in the founder's capital.

Wind-down. A business that ends must not walk away from customers who paid in advance:
* `venture.wind_down` (owners): stops all sales (products retired, offers and plans closed, exchanges it
  operates closed with open orders refunded), then - if the company can pay every customer the unused part of
  what they prepaid plus everything else it owes - refunds them, ends the subscriptions and dissolves the company
  through `biz.dissolve` (creditors first, the rest to shareholders). If it cannot, each customer's unused
  prepaid amount becomes a claim against the company and the 4.0 insolvency procedure takes over: after the
  grace period the company is declared bankrupt and claimants share its cash pro rata.
* Involuntary insolvency: the daily run (before the 4.0 company run on the same day) finds companies whose
  insolvency grace has expired and whose cash cannot cover their claims, turns their customers' unused prepaid
  subscription time into claims, ends those subscriptions and retires the products - so the bankruptcy that
  follows treats customers as creditors.
* When any company ends (hook `on_company_ended`), its remaining subscriptions as merchant or customer end, its
  products are retired and the exchanges it operates close.

World generation (config): `providers` - compute providers ({handle, name, cores, price, reliability, cash});
`incumbents` - established software businesses ({handle, name, need, tier, price_month, price_year?,
trial_days?, cores?, cash?, provider?}) that run a kernel-authored solver of a public need as a SaaS product with
subscription plans, created through the same tools an agent would use.
"""
from __future__ import annotations

from fractions import Fraction

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import HANDLE, ident

DAY = 86400


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


@register_module
class Ventures(Module):
    name = "ventures"
    host = "venture"
    depends = ("identity", "bank", "business", "contracts", "products", "subs")

    def install(self, world):
        st = world.state
        st.declare_table("ventures", index=("founder", "status"))
        prev = world.hooks.get("on_company_ended")

        def ended(ctx, org, status, _prev=prev):
            if _prev is not None:
                _prev(ctx, org, status)
            self._company_ended(ctx, org, status)
        world.hooks["on_company_ended"] = ended

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY, "ventures.daily", {}, prio=55, period=DAY, key="ventures.daily")
        self._genesis_world(ctx)

    def p(self, k, d):
        return self.world.spec.params.get(k, self.config.get(k, d))

    # ------------------------------------------------------------------ liabilities
    def prepaid_unused(self, state, org: str) -> list[tuple[str, str, int]]:
        """(subscription, customer, cents) of unused prepaid time of every live paid subscription to `org`."""
        out = []
        now = self.world.now
        for sid in state.find_ids("sub_subs", "merchant", org):
            s = state.tables["sub_subs"][sid]
            if s["status"] != "active" or not s["period_paid"] or s["period_end"] <= now:
                continue
            inv = state.get("sub_invoices", s["last_invoice"] or "")
            if inv is None or inv["status"] != "paid":
                continue
            net = int(inv["amount"]) - int(inv["refunded"])
            span = int(s["period_end"]) - int(s["period_start"])
            due = _floor(Fraction(net * (int(s["period_end"]) - now), span)) if span > 0 and net > 0 else 0
            if due > 0:
                out.append((sid, s["customer"], due))
        return out

    def _cash(self, state, org: str) -> int:
        return sum(max(0, state.tables["accounts"][a]["balance"]) for a in state.find_ids("accounts", "owner", org)
                   if state.tables["accounts"][a]["status"] == "open")

    def _stop_sales(self, ctx: Ctx, org: str) -> None:
        st = ctx.state
        prods = ctx.world.modules["products"]
        for pid in st.find_ids("products", "owner", org):
            if st.tables["products"][pid]["status"] != "retired":
                prods.retire(ctx, pid)
        xchg = ctx.world.modules.get("xchg")
        if xchg is not None:
            xchg.close_for_wind_up(ctx, org)

    def _claims_for_prepaid(self, ctx: Ctx, org: str) -> int:
        """Unused prepaid time becomes claims; the subscriptions end. Returns the number of claims."""
        contracts = ctx.world.modules["contracts"]
        subs = ctx.world.modules["subs"]
        n = 0
        for sid, customer, due in self.prepaid_unused(ctx.state, org):
            contracts.add_claim(ctx, org, customer, due, f"unused prepaid subscription {sid}")
            n += 1
        for sid in ctx.state.find_ids("sub_subs", "merchant", org):
            if ctx.state.tables["sub_subs"][sid]["status"] in ("trialing", "active", "past_due"):
                subs._end(ctx, sid, "merchant_ended")
        return n

    # ------------------------------------------------------------------ endpoints
    @endpoint("form", S.obj({"handle": S.s_str("company handle", pattern=HANDLE, maxLength=31),
                             "name": S.s_str(minLength=1, maxLength=120), "sector": S.s_str(maxLength=80),
                             "capital_cents": S.s_int("paid in from your account", minimum=0, maximum=10**11)},
                            ["handle", "name"]), cost_s=900)
    def ep_form(self, ctx, handle, name, sector="", capital_cents=0):
        """Found a company: organisation, company registration (fee), operating account, founder capital."""
        if not ctx.actor.startswith("p:"):
            raise Forbidden("only persons found companies")
        bank = ctx.world.modules["bank"]
        me = bank.primary_account(ctx.state, ctx.actor)
        if me is None:
            raise Forbidden("you need an account")
        org = ctx.call("id.create_org", {"handle": handle, "name": name, "kind": "company"})["org"]
        ctx.call("biz.register", {"org": org, "account": me, "sector": sector})
        acct = ctx.call("bank.open_account", {"name": "operating", "owner": org})["account"]
        if capital_cents:
            ctx.call("bank.transfer", {"from_account": me, "to": acct, "amount": int(capital_cents),
                                       "memo": "founder capital"})
        ctx.emit("venture.formed", {"org": org, "founder": ctx.actor, "capital": int(capital_cents),
                                    "sector": sector[:80]}, vis="public")
        return {"org": org, "account": acct}

    @endpoint("overview", S.obj({"org": S.s_str(maxLength=40), "days": S.s_int(minimum=1, maximum=365)}, ["org"]),
              read_only=True)
    def ep_overview(self, ctx, org, days=30):
        """Your company's own books: cash, ledger by kind, products, subscriptions by status, booked recurring
        revenue (all of your subscriptions, as billed), support load."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        t0 = ctx.now - int(days) * DAY
        accts = set(st.find_ids("accounts", "owner", org))
        by_kind: dict[str, int] = {}
        seen: set[str] = set()
        for a in sorted(accts):
            for lid in st.find_ids("acct_txns", "acct", a):
                e = st.tables["acct_txns"][lid]
                if e["t"] <= t0 or e["txn"] in seen:
                    continue
                seen.add(e["txn"])
                tx = st.tables["txns"][e["txn"]]
                net = sum(d for x, d in tx["postings"] if x in accts)
                by_kind[tx["kind"]] = by_kind.get(tx["kind"], 0) + net
        subs = {"trialing": 0, "active": 0, "past_due": 0, "cancelled": 0}
        booked = Fraction(0)
        sm = ctx.world.modules["subs"]
        for s in st.find("sub_subs", "merchant", org):
            subs[s["status"]] = subs.get(s["status"], 0) + 1
            if sm.recurring_valid(s, ctx.now):
                booked += sm.monthly_recurring(s)
        prods = [{"product": pid, "name": p["name"], "status": p["status"], "kind": p["kind"]}
                 for pid in st.find_ids("products", "owner", org) for p in [st.tables["products"][pid]]]
        sup = ctx.world.modules.get("support")
        return {"org": org, "cash": self._cash(st, org), "ledger_by_kind": dict(sorted(by_kind.items())),
                "products": prods, "subscriptions": subs, "booked_mrr_cents": float(booked),
                "support": sup.stats(st, org) if sup is not None else None,
                "status": (st.get("companies", org) or {}).get("status")}

    @endpoint("wind_down", S.obj({"org": S.s_str(maxLength=40), "reason": S.s_str(maxLength=200)}, ["org"]),
              cost_s=1800)
    def ep_wind_down(self, ctx, org, reason=""):
        """Orderly wind-down of your company: stop sales, repay customers' unused prepaid time, then dissolve; if
        the company cannot pay everyone, customers' prepaid time becomes claims for the insolvency procedure."""
        st = ctx.state
        org = ident(ctx.world).resolve_handle(st, org)
        if ident(ctx.world).role(st, org, ctx.actor) != "owner":
            raise Forbidden("only owners wind a company down")
        biz = ctx.world.modules["business"]
        if not biz.is_registered(st, org):
            raise NotFound("not a registered company")
        self._stop_sales(ctx, org)
        prepaid = self.prepaid_unused(st, org)
        owed_prepaid = sum(c for _, _, c in prepaid)
        other = sum(c["amount"] for _, c in biz.open_claims(st, org)) + \
            sum(x[3] for x in biz.fixed_obligations(st, org))
        cash = self._cash(st, org)
        subs = ctx.world.modules["subs"]
        if cash >= owed_prepaid + other:
            refunded = 0
            for sid in st.find_ids("sub_subs", "merchant", org):
                if st.tables["sub_subs"][sid]["status"] in ("trialing", "active", "past_due"):
                    refunded += subs._end(ctx, sid, "merchant_ended", refund_unused=True)
            for sid in st.find_ids("sub_subs", "customer", org):
                if st.tables["sub_subs"][sid]["status"] in ("trialing", "active", "past_due"):
                    subs._end(ctx, sid, "customer_ended")
            ctx.emit("venture.wind_down", {"org": org, "mode": "dissolve", "refunded": refunded,
                                           "reason": reason[:200]}, vis="public")
            ctx.call("biz.dissolve", {"org": org})
            return {"status": "dissolved", "refunded_cents": refunded}
        n = self._claims_for_prepaid(ctx, org)
        ctx.emit("venture.wind_down", {"org": org, "mode": "insolvent", "claims": n, "owed": owed_prepaid + other,
                                       "cash": cash, "reason": reason[:200]}, vis="public")
        return {"status": "insolvent", "customer_claims": n, "owed_cents": owed_prepaid + other, "cash_cents": cash}

    # ------------------------------------------------------------------ processes and hooks
    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        if "companies" not in st.tables:
            return
        biz = ctx.world.modules["business"]
        grace = int(biz.p("insolvency_grace_s", 14 * DAY))
        for org in st.find_ids("companies", "status", "registered"):
            c = st.tables["companies"][org]
            if c["distress_since"] is None or ctx.now - c["distress_since"] < grace:
                continue
            owed = sum(cl["amount"] for _, cl in biz.open_claims(st, org) if biz._valid(cl["amount"]))
            bank = ctx.world.modules["bank"]
            acct = bank.primary_account(st, org)
            cash = bank.available(st, acct) if acct else 0
            if cash >= owed:
                continue
            err = contain(ctx, self._pre_bankruptcy, ctx, org)
            if err is not None:
                ctx.emit("venture.daily_failed", {"org": org, "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def _pre_bankruptcy(self, ctx, org):
        self._stop_sales(ctx, org)
        n = self._claims_for_prepaid(ctx, org)
        ctx.emit("venture.insolvent", {"org": org, "customer_claims": n}, vis="public")

    def _company_ended(self, ctx, org, status):
        st = ctx.state
        if "products" not in st.tables:
            return
        subs = ctx.world.modules["subs"]
        for role in ("merchant", "customer"):
            for sid in st.find_ids("sub_subs", role, org):
                if st.tables["sub_subs"][sid]["status"] in ("trialing", "active", "past_due"):
                    subs._end(ctx, sid, f"{role}_ended")
        prods = ctx.world.modules["products"]
        for pid in st.find_ids("products", "owner", org):
            if st.tables["products"][pid]["status"] != "retired":
                prods.retire(ctx, pid)
        xchg = ctx.world.modules.get("xchg")
        if xchg is not None:
            xchg.close_for_wind_up(ctx, org)
        ctx.emit("venture.ended", {"org": org, "status": status}, vis="public")

    # ------------------------------------------------------------------ world generation
    def _genesis_world(self, ctx: Ctx) -> None:
        w = ctx.world
        cfg = self.config
        if not cfg.get("providers") and not cfg.get("incumbents"):
            return
        idm, bank, biz = w.modules["identity"], w.modules["bank"], w.modules["business"]
        cloud = w.modules.get("cloud")
        if cloud is None:
            raise InvalidAction("providers and incumbents need the cloud module")

        def company(handle, name, cash, sector, trusted=False):
            owner = idm.create_person(ctx, f"{handle}-ceo", f"{name} CEO", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, handle, name, owner, "company", {"trusted_runtime": trusted})
            biz.register_company(ctx, oid, owner, sector, fee=0)
            acct = bank.open_account(ctx, oid, "operating")
            if cash:
                bank.move(ctx, "acct_genesis", acct, int(cash), "endowment", f"initial capital of {oid}")
            return oid, owner, acct

        for p in cfg.get("providers", []):
            oid, _, _ = company(p["handle"], p["name"], p.get("cash", 5_000_000), "compute")
            cloud.add_provider(ctx, oid, p["name"], int(p["price"]), int(p["cores"]), float(p.get("reliability", 0.99)))
        from ..economy.needs import NEEDS
        from ..worlds.digital_civilization import service_repo_files
        code = w.modules["code"]
        provs = sorted(pid for pid, p in ctx.state.scan("providers") if p["status"] == "active")
        for f in cfg.get("incumbents", []):
            need = f["need"]
            if need not in NEEDS:
                raise InvalidAction(f"unknown need {need!r}")
            oid, owner, acct = company(f["handle"], f["name"], f.get("cash", 3_000_000), need, trusted=True)
            rid = code.create_repo(ctx, oid, f"{need.replace('.', '-')}-saas", "private", f"{f['name']} service")
            # direct calls are priced so that the subscription is the way to use it
            code.commit_files(ctx, rid, service_repo_files(need, int(f.get("tier", 0)), int(f.get("call_price", 500))),
                              "initial", owner, trusted=True)
            prov = f.get("provider") or provs[0]
            inst = ctx.call("cloud.rent", {"provider": prov, "cores": int(f.get("cores", 1)), "account": acct,
                                           "owner": oid}, as_principal=owner)
            host = f.get("host") or f["handle"][:24] + "-api"
            ctx.call("cloud.deploy", {"repo": rid, "host": host, "instance": inst["instance"]}, as_principal=owner)
            prod = ctx.call("prod.create", {"name": f["name"], "kind": "saas", "owner": oid,
                                            "description": f"{NEEDS[need].title} as a service. "
                                                           f"Keywords: {' '.join(NEEDS[need].keywords)}",
                                            "delivery": {"type": "service", "host": host, "endpoint": "solve",
                                                         "need": need}}, as_principal=owner)["product"]
            ctx.call("prod.release", {"product": prod, "version": "1.0"}, as_principal=owner)
            ctx.call("prod.update", {"product": prod, "status": "launched"}, as_principal=owner)
            ctx.call("subs.create_plan", {"product": prod, "price_cents": int(f["price_month"]), "interval": "month",
                                          "trial_days": int(f.get("trial_days", 0))}, as_principal=owner)
            if f.get("price_year"):
                ctx.call("subs.create_plan", {"product": prod, "price_cents": int(f["price_year"]),
                                              "interval": "year", "trial_days": int(f.get("trial_days", 0))},
                         as_principal=owner)

    # ------------------------------------------------------------------ appliers
    @applier("venture.formed")
    def ap_formed(self, ev):
        p = ev.payload
        self.world.state.put("ventures", p["org"], {"founder": p["founder"], "capital": p["capital"],
                                                   "sector": p["sector"], "status": "operating", "t": ev.t})

    @applier("venture.wind_down")
    def ap_wind_down(self, ev):
        st = self.world.state
        if st.has("ventures", ev.payload["org"]):
            st.patch("ventures", ev.payload["org"], status=f"wind_down:{ev.payload['mode']}")

    @applier("venture.insolvent")
    def ap_insolvent(self, ev):
        st = self.world.state
        if st.has("ventures", ev.payload["org"]):
            st.patch("ventures", ev.payload["org"], status="insolvent")

    @applier("venture.ended")
    def ap_ended(self, ev):
        st = self.world.state
        if st.has("ventures", ev.payload["org"]):
            st.patch("ventures", ev.payload["org"], status=ev.payload["status"], ended=ev.t)

    @applier("venture.daily_failed")
    def ap_daily_failed(self, ev):
        pass

    def check_invariants(self, world):
        st = world.state
        errs = []
        for org, v in st.tables["ventures"].items():
            c = st.get("companies", org)
            if c is None:
                errs.append(f"venture {org} has no company record")
            elif c["status"] in ("bankrupt", "dissolved"):
                live = [s for s in st.find("sub_subs", "merchant", org) if s["status"] in ("trialing", "active",
                                                                                           "past_due")]
                if live:
                    errs.append(f"ended company {org} still has {len(live)} live subscriptions")
        return errs


__all__ = ["Ventures", "CrucibleError", "Conflict"]
