"""Subscriptions (module `subs`, host `subs`): plans, trials, discounts, recurring billing through the bank,
failed payments, cancellation, plan changes with proration, and refunds after settlement.

Billing periods: a `month` is 30 days and a `year` 365 days of world time, counted from the moment a period
starts. Recurring-revenue metrics normalise by the plan's interval, not by days: a monthly plan contributes its
price to MRR, an annual plan one twelfth of its price (`crucible.commerce.revenue`).

Who pays: `subs.subscribe` is paid from the caller's own account (or an account of an organisation the caller
administers). Subscribing sets up a direct-debit mandate (`bank.authorize_debit`) from that account to the
merchant, limited to the plan's list price per billing period, so every later charge is checked by the bank
against the customer's own authority (including the spending limit of a delegation the customer acted under). A
customer who cancels the mandate in the bank stops paying: the next charge fails.

Life cycle (`status`):
  trialing  - plan with trial days, first subscription of this customer to this product: no charge until the
              trial ends; the first period is charged then.
  active    - the current period is paid. Renewal is charged at the period end (process `subs.bill`, hourly).
  past_due  - a charge failed (insufficient funds, cancelled mandate, frozen account). Access is suspended.
              Retries after 1, 3 and 7 days; a successful retry starts a new period at the time of payment. When
              the last retry fails the subscription is cancelled for non-payment (involuntary churn).
  cancelled - ended. `cancel_reason` says why: `customer`, `merchant`, `trial_cancelled`, `nonpayment`,
              `merchant_ended`, `customer_ended`.

Cancellation: `at_period_end=true` keeps the paid period (or the trial) and stops renewal; `subs.resume` undoes
it before the period ends. `at_period_end=false` ends the subscription now.

Proration rules (all amounts integer cents; u = unused fraction of the current paid period
= (period_end - now) / (period_end - period_start)):
  * immediate cancellation by the CUSTOMER: access ends now, nothing is refunded automatically (the merchant can
    refund with `subs.refund`);
  * immediate cancellation by the MERCHANT, or because the merchant's business ends: the unused part of what the
    customer paid for the current period, floor(paid_net x u), is refunded at once;
  * plan change within the same interval: credit = floor(paid_net x u); due = ceil(new_recurring_price x u) -
    credit. A positive amount is charged now (an invoice of kind `proration` for the rest of the period, which
    keeps its end); a negative amount is kept as account credit and taken off the next invoices;
  * plan change to another interval: the new period starts now; the new recurring price minus the same credit is
    charged now, any excess credit is carried forward;
  * trialing subscriptions change plan without any charge;
  * a coupon (`subs.create_coupon`) takes `percent_off` off the list price for `periods` paid periods (0 = for as
    long as the subscription lasts); it belongs to its plan and ends when the subscription changes plan; credit
    never makes an invoice negative.

Refunds (`subs.refund`) are made by the merchant after settlement: a reversing ledger entry (kind `refund`,
`ref` = the invoice, memo names the reversed transaction) back to the account that paid. Refunding the whole
invoice of the current period makes that period unpaid, so the subscription stops counting as recurring revenue.
"""
from __future__ import annotations

from fractions import Fraction

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
PERIOD_S = {"month": 30 * DAY, "year": 365 * DAY}
MONTHS = {"month": 1, "year": 12}
RETRY_S = (1 * DAY, 3 * DAY, 7 * DAY)
LIVE = ("trialing", "active", "past_due")


def subs_module(world) -> "Subscriptions | None":
    return world.modules.get("subs")  # type: ignore[return-value]


def _ceil(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _floor(x: Fraction) -> int:
    return x.numerator // x.denominator


@register_module
class Subscriptions(Module):
    name = "subs"
    host = "subs"
    depends = ("identity", "bank", "products")

    def install(self, world):
        st = world.state
        st.declare_table("sub_plans", index=("product", "merchant", "status"))
        st.declare_table("sub_coupons", index=("plan",))
        st.declare_table("sub_subs", index=("plan", "product", "customer", "merchant", "status"))
        st.declare_table("sub_invoices", index=("subscription", "customer", "merchant", "status"))

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 3600, "subs.bill", {}, prio=45, period=3600, key="subs.bill")

    # ------------------------------------------------------------------ internal API (other modules, metrics)
    def public_plans(self, state, product: str) -> list[dict]:
        out = []
        for plid in state.find_ids("sub_plans", "product", product):
            pl = state.tables["sub_plans"][plid]
            if pl["status"] == "active":
                out.append({"plan": plid, "name": pl["name"], "price_cents": pl["price"], "interval": pl["interval"],
                            "trial_days": pl["trial_days"],
                            "monthly_equivalent_cents": round(pl["price"] / MONTHS[pl["interval"]], 2)})
        return out

    def paid_by(self, state, product: str, principals) -> bool:
        for c in principals:
            for inv in state.find("sub_invoices", "customer", c):
                if inv["product"] == product and inv["status"] == "paid" and inv["amount"] - inv["refunded"] > 0:
                    return True
        return False

    def live_subscription(self, state, product: str, customer: str) -> str | None:
        for sid in state.find_ids("sub_subs", "customer", customer):
            s = state.tables["sub_subs"][sid]
            if s["product"] == product and s["status"] in LIVE:
                return sid
        return None

    def archive_product_plans(self, ctx: Ctx, product: str) -> None:
        for plid in ctx.state.find_ids("sub_plans", "product", product):
            if ctx.state.tables["sub_plans"][plid]["status"] == "active":
                ctx.emit("subs.plan_status", {"id": plid, "status": "archived"}, vis="public")

    def monthly_recurring(self, s: dict) -> Fraction:
        """Normalised monthly amount of one subscription's current recurring charge (Fraction of cents)."""
        return Fraction(int(s["current_amount"]), MONTHS[s["interval"]])

    def recurring_valid(self, s: dict, now: int) -> bool:
        """The subscription is paid up for a period that has not ended: active, current period paid and not
        refunded in full."""
        return s["status"] == "active" and bool(s["period_paid"]) and s["period_end"] is not None \
            and s["period_end"] > now

    # ------------------------------------------------------------------ helpers
    def _plan(self, state, plid: str) -> dict:
        pl = state.get("sub_plans", plid)
        if pl is None:
            raise NotFound(f"no plan {plid}")
        return pl

    def _sub(self, state, sid: str) -> dict:
        s = state.get("sub_subs", sid)
        if s is None:
            raise NotFound(f"no subscription {sid}")
        return s

    def _side(self, ctx: Ctx, s: dict) -> str:
        """'customer' or 'merchant' for the caller, with the authority that side needs."""
        mine = ctx.world.principals_of(ctx.actor)
        if s["customer"] in mine:
            ident(ctx.world).authorize(ctx, s["customer"], "bank.spend")
            return "customer"
        if s["merchant"] in mine:
            ident(ctx.world).authorize(ctx, s["merchant"], "org.admin")
            return "merchant"
        raise Forbidden("not a party to this subscription")

    def _vis(self, s: dict) -> tuple:
        return tuple(sorted({s["customer"], s["merchant"]}))

    @staticmethod
    def _try(ctx: Ctx, fn, *a, **kw):
        """Run fn in a savepoint; an in-world refusal (CrucibleError) is rolled back and returned. Bugs raise."""
        state, world = ctx.state, ctx.world
        mark, n_ev, seq0 = state.savepoint(), len(ctx.events), world.seq
        try:
            return fn(*a, **kw), None
        except CrucibleError as e:
            state.rollback_to(mark)
            del ctx.events[n_ev:]
            world.seq = seq0
            return None, e

    def _price_now(self, s: dict) -> tuple[int, int]:
        """(list price, discount) of the next period's charge."""
        price = int(s["price"])
        if s["percent_off"] and (s["discount_left"] is None or s["discount_left"] > 0):
            return price, price * int(s["percent_off"]) // 100
        return price, 0

    def _mandate(self, ctx: Ctx, account: str, merchant: str, price: int, interval: str, sid: str) -> str:
        return ctx.call("bank.authorize_debit", {"from_account": account, "payee": merchant,
                                                 "max_per_period": int(price), "period_s": PERIOD_S[interval],
                                                 "purpose": f"subscription {sid}"})["mandate"]

    def _cancel_mandate(self, ctx: Ctx, s: dict) -> None:
        m = ctx.state.get("mandates", s["mandate"] or "")
        if m is not None and m["status"] == "active":
            ctx.emit("bank.mandate_status", {"id": s["mandate"], "status": "cancelled"}, vis=(m["payer"], m["payee"]))

    def _charge_period(self, ctx: Ctx, sid: str, kind: str, start: int) -> str:
        """Charge one billing period under the mandate. Raises CrucibleError if the payment fails (nothing is
        recorded then). Returns the invoice id."""
        st = ctx.state
        s = st.tables["sub_subs"][sid]
        bank = ctx.world.modules["bank"]
        price, discount = self._price_now(s)
        gross = price - discount
        credit = min(int(s["credit"]), gross)
        amount = gross - credit
        dst = bank.primary_account(st, s["merchant"])
        if dst is None:
            raise Forbidden("the merchant cannot be paid")
        iid = st.new_id("sinv")
        txn = bank.debit(ctx, s["mandate"], dst, amount, "subscription", f"{s['interval']}ly {s['product']}",
                         ref=iid) if amount > 0 else None
        end = start + PERIOD_S[s["interval"]]
        ctx.emit("subs.invoice", {"id": iid, "subscription": sid, "plan": s["plan"], "product": s["product"],
                                  "customer": s["customer"], "merchant": s["merchant"], "account": s["account"],
                                  "kind": kind, "status": "paid", "list_price": price, "discount": discount,
                                  "credit_applied": credit, "amount": amount, "txn": txn, "interval": s["interval"],
                                  "period_start": start, "period_end": end}, vis=self._vis(s))
        left = s["discount_left"]
        if discount and left is not None:
            left -= 1
        ch = {"status": "active", "period_start": start, "period_end": end, "period_paid": True,
              "current_amount": gross, "credit": int(s["credit"]) - credit, "failures": 0, "next_retry": None,
              "last_invoice": iid, "periods_paid": int(s["periods_paid"]) + 1, "discount_left": left}
        ctx.emit("subs.status", {"id": sid, "changes": ch}, vis=self._vis(s))
        if s["status"] != "active" or s.get("entitlement") is None:
            self._ensure_entitlement(ctx, sid)
        return iid

    def _ensure_entitlement(self, ctx: Ctx, sid: str) -> None:
        st = ctx.state
        s = st.tables["sub_subs"][sid]
        e = st.get("prod_entitlements", s.get("entitlement") or "")
        if e is not None and e["status"] == "active":
            return
        eid = ctx.world.modules["products"].grant(ctx, s["product"], s["customer"], "subscription", sid)
        ctx.emit("subs.status", {"id": sid, "changes": {"entitlement": eid}}, vis=self._vis(s))

    def _suspend_entitlement(self, ctx: Ctx, sid: str, status: str) -> None:
        s = ctx.state.tables["sub_subs"][sid]
        if s.get("entitlement"):
            ctx.world.modules["products"].revoke(ctx, s["entitlement"], status)

    def _attempt(self, ctx: Ctx, sid: str, kind: str, start: int) -> bool:
        """One billing attempt from the scheduler: a failure is recorded as a failed invoice and starts or
        continues dunning."""
        _, err = self._try(ctx, self._charge_period, ctx, sid, kind, start)
        if err is None:
            return True
        st = ctx.state
        s = st.tables["sub_subs"][sid]
        price, discount = self._price_now(s)
        iid = st.new_id("sinv")
        ctx.emit("subs.invoice", {"id": iid, "subscription": sid, "plan": s["plan"], "product": s["product"],
                                  "customer": s["customer"], "merchant": s["merchant"], "account": s["account"],
                                  "kind": kind, "status": "failed", "list_price": price, "discount": discount,
                                  "credit_applied": 0, "amount": 0, "txn": None, "interval": s["interval"],
                                  "period_start": None, "period_end": None, "reason": f"{err.code}: {err.message}"[:200]},
                 vis=self._vis(s))
        failures = int(s["failures"]) + 1
        if failures > len(RETRY_S):
            self._end(ctx, sid, "nonpayment")
        else:
            ctx.emit("subs.status", {"id": sid, "changes": {"status": "past_due", "failures": failures,
                                                            "period_paid": False,
                                                            "next_retry": ctx.now + RETRY_S[failures - 1]}},
                     vis=self._vis(s))
            self._suspend_entitlement(ctx, sid, "suspended")
        return False

    def _end(self, ctx: Ctx, sid: str, reason: str, refund_unused: bool = False) -> int:
        """End a subscription now. Returns the cents refunded for unused time (merchant-side endings)."""
        st = ctx.state
        s = st.tables["sub_subs"][sid]
        refunded = 0
        if refund_unused and s["status"] == "active" and s["period_paid"] and s["period_end"] > ctx.now:
            inv = st.get("sub_invoices", s["last_invoice"] or "")
            if inv is not None and inv["status"] == "paid":
                net = int(inv["amount"]) - int(inv["refunded"])
                span = int(s["period_end"]) - int(s["period_start"])
                due = _floor(Fraction(net * (int(s["period_end"]) - ctx.now), span)) if span > 0 else 0
                if due > 0:
                    self._refund(ctx, s["last_invoice"], due, "unused time (cancelled by the merchant)")
                    refunded = due
        self._cancel_mandate(ctx, s)
        ctx.emit("subs.status", {"id": sid, "changes": {"status": "cancelled", "cancelled_at": ctx.now,
                                                        "cancel_reason": reason, "cancel_at_period_end": False,
                                                        "next_retry": None}}, vis=self._vis(s))
        self._suspend_entitlement(ctx, sid, "ended")
        return refunded

    def _refund(self, ctx: Ctx, invoice: str, amount: int, reason: str) -> str:
        st = ctx.state
        inv = st.tables["sub_invoices"][invoice]
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, inv["merchant"])
        a = st.get("accounts", inv["account"])
        dst = inv["account"] if a is not None and a["status"] == "open" else bank.primary_account(st, inv["customer"])
        if src is None or dst is None:
            raise Forbidden("the refund cannot be paid (an account is closed)")
        txn = bank.move(ctx, src, dst, int(amount), "refund", f"reversal of {inv['txn']}: {reason}"[:300], ref=invoice)
        ctx.emit("subs.refund", {"invoice": invoice, "subscription": inv["subscription"], "amount": int(amount),
                                 "txn": txn, "reverses": inv["txn"], "reason": reason[:300]},
                 vis=tuple(sorted({inv["customer"], inv["merchant"]})))
        s = st.tables["sub_subs"][inv["subscription"]]
        inv = st.tables["sub_invoices"][invoice]
        if s["last_invoice"] == invoice and int(inv["refunded"]) >= int(inv["amount"]):
            ctx.emit("subs.status", {"id": inv["subscription"], "changes": {"period_paid": False}}, vis=self._vis(s))
        return txn

    # ------------------------------------------------------------------ endpoints: merchants
    @endpoint("create_plan", S.obj({"product": S.s_str(maxLength=40), "price_cents": S.s_int(minimum=1, maximum=10**9),
                                    "interval": S.s_enum(["month", "year"]),
                                    "trial_days": S.s_int(minimum=0, maximum=90), "name": S.s_str(maxLength=80)},
                                   ["product", "price_cents", "interval"]), cost_s=30)
    def ep_create_plan(self, ctx, product, price_cents, interval, trial_days=0, name=""):
        """Create a subscription plan for a product you sell (monthly or annual, optional free trial)."""
        st = ctx.state
        prods = ctx.world.modules["products"]
        p = prods.product(st, product)
        ident(ctx.world).authorize(ctx, p["owner"], "org.admin")
        if p["status"] == "retired":
            raise Conflict("product is retired")
        if (p["delivery"] or {}).get("type") == "goods":
            raise InvalidAction("goods are sold with offers (prod.offer), not subscriptions")
        if len([x for x in st.find("sub_plans", "product", product) if x["status"] == "active"]) >= 10:
            raise Conflict("plan limit reached")
        plid = st.new_id("plan")
        ctx.emit("subs.plan", {"id": plid, "product": product, "merchant": p["owner"], "price": int(price_cents),
                               "interval": interval, "trial_days": int(trial_days),
                               "name": name or f"{p['name']} ({interval}ly)"}, vis="public")
        return {"plan": plid, "monthly_equivalent_cents": round(price_cents / MONTHS[interval], 2)}

    @endpoint("archive_plan", S.obj({"plan": S.s_str(maxLength=40)}, ["plan"]))
    def ep_archive_plan(self, ctx, plan):
        """Stop selling a plan (existing subscriptions continue)."""
        pl = self._plan(ctx.state, plan)
        ident(ctx.world).authorize(ctx, pl["merchant"], "org.admin")
        if pl["status"] != "active":
            raise Conflict("plan is archived")
        ctx.emit("subs.plan_status", {"id": plan, "status": "archived"}, vis="public")
        return {"ok": True}

    @endpoint("create_coupon", S.obj({"plan": S.s_str(maxLength=40), "percent_off": S.s_int(minimum=1, maximum=100),
                                      "periods": S.s_int("paid periods discounted (0: always)", minimum=0, maximum=36),
                                      "max_redemptions": S.s_int(minimum=1, maximum=100_000)},
                                     ["plan", "percent_off"]), cost_s=10)
    def ep_create_coupon(self, ctx, plan, percent_off, periods=1, max_redemptions=100):
        """Create a discount coupon for a plan. Give its id to customers; they pass it to subs.subscribe."""
        pl = self._plan(ctx.state, plan)
        ident(ctx.world).authorize(ctx, pl["merchant"], "org.admin")
        cid = ctx.state.new_id("cpn")
        ctx.emit("subs.coupon", {"id": cid, "plan": plan, "merchant": pl["merchant"], "percent_off": int(percent_off),
                                 "periods": int(periods), "max_redemptions": int(max_redemptions)},
                 vis=(pl["merchant"],))
        return {"coupon": cid}

    @endpoint("refund", S.obj({"invoice": S.s_str(maxLength=40), "amount_cents": S.s_int(minimum=1, maximum=10**12),
                               "reason": S.s_str(maxLength=300)}, ["invoice", "reason"]), cost_s=10)
    def ep_refund(self, ctx, invoice, reason, amount_cents=None):
        """Merchant refunds a paid invoice (all or part) after settlement: a reversing ledger entry to the
        account that paid. Refunding the whole current invoice makes the current period unpaid."""
        st = ctx.state
        inv = st.get("sub_invoices", invoice)
        if inv is None:
            raise NotFound("no invoice")
        ident(ctx.world).authorize(ctx, inv["merchant"], "bank.spend")
        if inv["status"] != "paid":
            raise Conflict(f"invoice is {inv['status']}")
        left = int(inv["amount"]) - int(inv["refunded"])
        if left <= 0:
            raise Conflict("nothing left to refund")
        amount = left if amount_cents is None else int(amount_cents)
        if amount > left:
            raise InvalidAction(f"at most {left} cents can still be refunded")
        txn = self._refund(ctx, invoice, amount, reason)
        inv = st.tables["sub_invoices"][invoice]
        return {"txn": txn, "refunded": inv["refunded"], "remaining": inv["amount"] - inv["refunded"]}

    # ------------------------------------------------------------------ endpoints: customers
    @endpoint("subscribe", S.obj({"plan": S.s_str(maxLength=40), "coupon": S.s_str(maxLength=40),
                                  "account": S.s_str("your account, or one of an organisation you administer "
                                                     "(default: your primary account)", maxLength=40)},
                                 ["plan"]), cost_s=20)
    def ep_subscribe(self, ctx, plan, coupon=None, account=None):
        """Subscribe to a plan, paid from your own account. With a trial (first subscription to this product
        only) nothing is charged until it ends; otherwise the first period is charged now."""
        st = ctx.state
        pl = self._plan(st, plan)
        if pl["status"] != "active":
            raise Conflict("plan is not on sale")
        prods = ctx.world.modules["products"]
        p = prods.product(st, pl["product"])
        if p["status"] != "launched":
            raise Conflict("product is not on sale")
        acct, customer = prods._account_of_caller(ctx, account, 0)
        if customer == pl["merchant"]:
            raise Forbidden("cannot subscribe to your own plan")
        if self.live_subscription(st, pl["product"], customer) is not None:
            raise Conflict("already subscribed to this product (use subs.change_plan)")
        percent, periods = 0, None
        if coupon is not None:
            c = st.get("sub_coupons", coupon)
            if c is None or c["plan"] != plan or c["status"] != "active":
                raise NotFound("no such coupon for this plan")
            if c["redeemed"] >= c["max_redemptions"]:
                raise Conflict("coupon fully redeemed")
            percent, periods = int(c["percent_off"]), (int(c["periods"]) or None)
        before = any(st.tables["sub_subs"][x]["product"] == pl["product"]
                     for x in st.find_ids("sub_subs", "customer", customer))
        trial = int(pl["trial_days"]) > 0 and not before
        sid = st.new_id("sub")
        mandate = self._mandate(ctx, acct, pl["merchant"], pl["price"], pl["interval"], sid)
        ctx.emit("subs.subscription", {"id": sid, "plan": plan, "product": pl["product"], "merchant": pl["merchant"],
                                       "customer": customer, "account": acct, "mandate": mandate,
                                       "interval": pl["interval"], "price": pl["price"], "coupon": coupon,
                                       "percent_off": percent, "discount_left": periods,
                                       "trial_end": ctx.now + int(pl["trial_days"]) * DAY if trial else None},
                 vis=tuple(sorted({customer, pl["merchant"]})))
        if coupon is not None:
            ctx.emit("subs.coupon_used", {"id": coupon}, vis=(pl["merchant"],))
        if trial:
            self._ensure_entitlement(ctx, sid)
            out = {"subscription": sid, "status": "trialing", "trial_end": ctx.now + int(pl["trial_days"]) * DAY}
        else:
            iid = self._charge_period(ctx, sid, "initial", ctx.now)
            inv = st.tables["sub_invoices"][iid]
            out = {"subscription": sid, "status": "active", "invoice": iid, "charged_cents": inv["amount"],
                   "period_end": inv["period_end"]}
        return out

    @endpoint("cancel", S.obj({"subscription": S.s_str(maxLength=40),
                               "at_period_end": S.s_bool("keep the paid period (or trial) and stop renewal")},
                              ["subscription", "at_period_end"]))
    def ep_cancel(self, ctx, subscription, at_period_end):
        """Cancel a subscription, as its customer or its merchant: at the end of the current period, or now
        (a merchant's immediate cancellation refunds the unused paid time)."""
        st = ctx.state
        s = self._sub(st, subscription)
        side = self._side(ctx, s)
        if s["status"] not in LIVE:
            raise Conflict(f"subscription is {s['status']}")
        if at_period_end and s["status"] != "past_due":
            if s["cancel_at_period_end"]:
                raise Conflict("already set to cancel at period end")
            ctx.emit("subs.status", {"id": subscription, "changes": {"cancel_at_period_end": True,
                                                                     "cancel_requested_by": side}}, vis=self._vis(s))
            end = s["trial_end"] if s["status"] == "trialing" else s["period_end"]
            return {"status": s["status"], "ends_at": end}
        reason = side if s["status"] != "trialing" else "trial_cancelled"
        refunded = self._end(ctx, subscription, reason, refund_unused=side == "merchant")
        return {"status": "cancelled", "refunded_cents": refunded}

    @endpoint("resume", S.obj({"subscription": S.s_str(maxLength=40)}, ["subscription"]))
    def ep_resume(self, ctx, subscription):
        """Undo a pending cancellation at period end (before the period ends)."""
        s = self._sub(ctx.state, subscription)
        self._side(ctx, s)
        if s["status"] not in ("trialing", "active") or not s["cancel_at_period_end"]:
            raise Conflict("nothing to resume")
        ctx.emit("subs.status", {"id": subscription, "changes": {"cancel_at_period_end": False,
                                                                 "cancel_requested_by": None}}, vis=self._vis(s))
        return {"ok": True}

    @endpoint("change_plan", S.obj({"subscription": S.s_str(maxLength=40), "plan": S.s_str(maxLength=40)},
                                   ["subscription", "plan"]), cost_s=20)
    def ep_change_plan(self, ctx, subscription, plan):
        """Switch to another plan of the same product, prorated (see the module documentation)."""
        st = ctx.state
        s = self._sub(st, subscription)
        if self._side(ctx, s) != "customer":
            raise Forbidden("only the customer changes plans")
        if s["status"] not in ("trialing", "active"):
            raise Conflict(f"subscription is {s['status']}")
        new = self._plan(st, plan)
        if new["product"] != s["product"] or new["status"] != "active":
            raise InvalidAction("choose an active plan of the same product")
        if plan == s["plan"]:
            raise Conflict("already on this plan")
        bank = ctx.world.modules["bank"]
        mandate = self._mandate(ctx, s["account"], s["merchant"], new["price"], new["interval"], subscription)
        self._cancel_mandate(ctx, s)
        # a coupon belongs to the plan it was issued for: it does not follow the subscription to another plan
        base = {"plan": plan, "price": new["price"], "interval": new["interval"], "mandate": mandate,
                "percent_off": 0, "discount_left": None}
        if s["status"] == "trialing":
            ctx.emit("subs.status", {"id": subscription, "changes": base}, vis=self._vis(s))
            return {"status": "trialing", "charged_cents": 0, "credit_cents": int(s["credit"])}
        inv = st.tables["sub_invoices"][s["last_invoice"]]
        paid_net = int(inv["amount"]) - int(inv["refunded"]) if s["period_paid"] else 0
        span = int(s["period_end"]) - int(s["period_start"])
        u = Fraction(max(0, int(s["period_end"]) - ctx.now), span) if span > 0 else Fraction(0)
        credit = _floor(paid_net * u)
        lp, disc = int(new["price"]), 0
        gross_new = lp
        if new["interval"] == s["interval"]:
            gross_due = _ceil(gross_new * u)
            start, end = ctx.now, int(s["period_end"])
        else:
            gross_due = gross_new
            start, end = ctx.now, ctx.now + PERIOD_S[new["interval"]]
        used = min(credit, gross_due)
        charge = gross_due - used
        carried = int(s["credit"]) + (credit - used)
        iid = st.new_id("sinv")
        txn = None
        if charge:
            ident(ctx.world).authorize(ctx, s["customer"], "bank.spend", amount=charge, scope=s["account"])
            dst = bank.primary_account(st, s["merchant"])
            if dst is None:
                raise Forbidden("the merchant cannot be paid")
            txn = bank.move(ctx, s["account"], dst, charge, "subscription", f"proration {subscription}", ref=iid)
        ctx.emit("subs.invoice", {"id": iid, "subscription": subscription, "plan": plan, "product": s["product"],
                                  "customer": s["customer"], "merchant": s["merchant"], "account": s["account"],
                                  "kind": "proration", "status": "paid", "list_price": lp, "discount": disc,
                                  "credit_applied": used, "prorated_cents": gross_due, "amount": charge, "txn": txn, "interval": new["interval"],
                                  "period_start": start, "period_end": end}, vis=self._vis(s))
        ch = dict(base, period_start=start, period_end=end, current_amount=gross_new, credit=carried,
                  last_invoice=iid, period_paid=True)
        if new["interval"] != s["interval"]:
            ch["periods_paid"] = int(s["periods_paid"]) + 1
        ctx.emit("subs.status", {"id": subscription, "changes": ch}, vis=self._vis(s))
        return {"status": "active", "charged_cents": charge, "credit_cents": carried, "period_end": end}

    @endpoint("plans", S.obj({"product": S.s_str(maxLength=40), "merchant": S.s_str(maxLength=40),
                              "limit": S.s_int(minimum=1, maximum=100)}), read_only=True)
    def ep_plans(self, ctx, product=None, merchant=None, limit=50):
        """Plans on sale (optionally for one product or merchant)."""
        st = ctx.state
        if product is not None:
            ids = st.find_ids("sub_plans", "product", product)
        elif merchant is not None:
            ids = st.find_ids("sub_plans", "merchant", ident(ctx.world).resolve_handle(st, merchant))
        else:
            ids = st.find_ids("sub_plans", "status", "active")
        out = []
        for plid in ids:
            pl = st.tables["sub_plans"][plid]
            if pl["status"] != "active":
                continue
            out.append({"plan": plid, "product": pl["product"], "merchant": pl["merchant"], "name": pl["name"],
                        "price_cents": pl["price"], "interval": pl["interval"], "trial_days": pl["trial_days"],
                        "monthly_equivalent_cents": round(pl["price"] / MONTHS[pl["interval"]], 2)})
            if len(out) >= limit:
                break
        return out

    @endpoint("mine", S.obj({"role": S.s_enum(["customer", "merchant"]), "status": S.s_enum(list(LIVE) + ["cancelled"]),
                             "limit": S.s_int(minimum=1, maximum=500)}), read_only=True)
    def ep_mine(self, ctx, role="customer", status=None, limit=100):
        """Your subscriptions as a customer (default) or as a merchant (yours and your organisations')."""
        st = ctx.state
        out = []
        for who in sorted(ctx.world.principals_of(ctx.actor)):
            for sid in st.find_ids("sub_subs", role, who):
                s = st.tables["sub_subs"][sid]
                if status is not None and s["status"] != status:
                    continue
                out.append({"subscription": sid, "plan": s["plan"], "product": s["product"], "customer": s["customer"],
                            "merchant": s["merchant"], "status": s["status"], "interval": s["interval"],
                            "price_cents": s["price"], "current_amount_cents": s["current_amount"],
                            "trial_end": s["trial_end"], "period_end": s["period_end"],
                            "cancel_at_period_end": s["cancel_at_period_end"], "credit_cents": s["credit"],
                            "cancel_reason": s["cancel_reason"], "periods_paid": s["periods_paid"],
                            "mandate": s["mandate"], "account": s["account"]})
        return out[-limit:]

    @endpoint("invoices", S.obj({"subscription": S.s_str(maxLength=40)}, ["subscription"]), read_only=True)
    def ep_invoices(self, ctx, subscription):
        """Invoices of a subscription you are a party to."""
        st = ctx.state
        s = self._sub(st, subscription)
        mine = ctx.world.principals_of(ctx.actor)
        if s["customer"] not in mine and s["merchant"] not in mine:
            raise Forbidden("not a party to this subscription")
        return [dict(st.tables["sub_invoices"][i], id=i) for i in st.find_ids("sub_invoices", "subscription", subscription)]

    # ------------------------------------------------------------------ billing
    @process("bill")
    def pr_bill(self, ctx):
        st = ctx.state
        due = []
        for status in LIVE:
            for sid in st.find_ids("sub_subs", "status", status):
                s = st.tables["sub_subs"][sid]
                when = {"trialing": s["trial_end"], "active": s["period_end"], "past_due": s["next_retry"]}[status]
                if when is not None and when <= ctx.now:
                    due.append(sid)
        for sid in sorted(due):
            err = contain(ctx, self._bill_one, ctx, sid)
            if err is not None:
                ctx.emit("subs.bill_failed", {"id": sid, "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def _bill_one(self, ctx: Ctx, sid: str) -> None:
        s = ctx.state.tables["sub_subs"][sid]
        if s["status"] == "trialing":
            if s["cancel_at_period_end"]:
                self._end(ctx, sid, "trial_cancelled")
            else:
                self._attempt(ctx, sid, "initial", int(s["trial_end"]))
        elif s["status"] == "active":
            if s["cancel_at_period_end"]:
                self._end(ctx, sid, s.get("cancel_requested_by") or "customer")
            else:
                self._attempt(ctx, sid, "renewal", int(s["period_end"]))
        elif s["status"] == "past_due":
            if s["cancel_at_period_end"]:
                self._end(ctx, sid, s.get("cancel_requested_by") or "customer")
            else:
                self._attempt(ctx, sid, "retry", ctx.now)

    def end_for_party(self, ctx: Ctx, principal: str, why: str) -> int:
        """End every live subscription in which `principal` is the merchant (unused time refunded where it can
        be) or the customer. Used when a business ends. Returns the number ended."""
        st = ctx.state
        n = 0
        for role in ("merchant", "customer"):
            for sid in st.find_ids("sub_subs", role, principal):
                if st.tables["sub_subs"][sid]["status"] in LIVE:
                    _, err = self._try(ctx, self._end, ctx, sid, f"{role}_ended", role == "merchant")
                    if err is not None:     # the refund could not be paid: end it without one
                        self._end(ctx, sid, f"{role}_ended")
                    n += 1
        return n

    # ------------------------------------------------------------------ appliers
    @applier("subs.plan")
    def ap_plan(self, ev):
        p = ev.payload
        self.world.state.insert("sub_plans", p["id"], {"product": p["product"], "merchant": p["merchant"],
                                                      "price": p["price"], "interval": p["interval"],
                                                      "trial_days": p["trial_days"], "name": p["name"],
                                                      "status": "active", "t": ev.t})

    @applier("subs.plan_status")
    def ap_plan_status(self, ev):
        self.world.state.patch("sub_plans", ev.payload["id"], status=ev.payload["status"])

    @applier("subs.coupon")
    def ap_coupon(self, ev):
        p = ev.payload
        self.world.state.insert("sub_coupons", p["id"], {"plan": p["plan"], "merchant": p["merchant"],
                                                        "percent_off": p["percent_off"], "periods": p["periods"],
                                                        "max_redemptions": p["max_redemptions"], "redeemed": 0,
                                                        "status": "active", "t": ev.t})

    @applier("subs.coupon_used")
    def ap_coupon_used(self, ev):
        c = self.world.state.tables["sub_coupons"][ev.payload["id"]]
        self.world.state.patch("sub_coupons", ev.payload["id"], redeemed=c["redeemed"] + 1)

    @applier("subs.subscription")
    def ap_subscription(self, ev):
        p = ev.payload
        trial = p["trial_end"] is not None
        self.world.state.insert("sub_subs", p["id"], {
            "plan": p["plan"], "product": p["product"], "merchant": p["merchant"], "customer": p["customer"],
            "account": p["account"], "mandate": p["mandate"], "interval": p["interval"], "price": p["price"],
            "coupon": p["coupon"], "percent_off": p["percent_off"], "discount_left": p["discount_left"],
            "status": "trialing" if trial else "incomplete", "created": ev.t, "trial_end": p["trial_end"],
            "period_start": None, "period_end": None, "period_paid": False, "current_amount": 0, "credit": 0,
            "failures": 0, "next_retry": None, "cancel_at_period_end": False, "cancel_requested_by": None,
            "cancelled_at": None, "cancel_reason": None, "last_invoice": None, "periods_paid": 0,
            "entitlement": None})

    @applier("subs.status")
    def ap_status(self, ev):
        self.world.state.patch("sub_subs", ev.payload["id"], **ev.payload["changes"])

    @applier("subs.invoice")
    def ap_invoice(self, ev):
        p = ev.payload
        rec = {k: p[k] for k in ("subscription", "plan", "product", "customer", "merchant", "account", "kind",
                                 "status", "list_price", "discount", "credit_applied", "amount", "txn", "interval",
                                 "period_start", "period_end")}
        rec.update(refunded=0, refunds=[], t=ev.t, reason=p.get("reason"))
        self.world.state.insert("sub_invoices", p["id"], rec)

    @applier("subs.refund")
    def ap_refund(self, ev):
        p = ev.payload
        st = self.world.state
        inv = st.tables["sub_invoices"][p["invoice"]]
        st.patch("sub_invoices", p["invoice"], refunded=inv["refunded"] + p["amount"],
                 refunds=list(inv["refunds"]) + [{"txn": p["txn"], "amount": p["amount"], "reverses": p["reverses"],
                                                  "t": ev.t}])

    @applier("subs.bill_failed")
    def ap_bill_failed(self, ev):
        pass

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        for iid, inv in st.tables["sub_invoices"].items():
            if not 0 <= inv["refunded"] <= inv["amount"]:
                errs.append(f"invoice {iid} refunded {inv['refunded']} of {inv['amount']}")
            if inv["status"] == "paid" and inv["amount"] > 0:
                tx = st.get("txns", inv["txn"] or "")
                if tx is None or sum(d for a, d in tx["postings"] if d > 0) != inv["amount"] or tx["ref"] != iid:
                    errs.append(f"invoice {iid} does not match its ledger transaction")
        for sid, s in st.tables["sub_subs"].items():
            if s["status"] == "incomplete":
                errs.append(f"subscription {sid} left incomplete")
            if s["status"] == "active" and (s["period_end"] is None or s["last_invoice"] is None):
                errs.append(f"active subscription {sid} has no paid period")
            if s["credit"] < 0:
                errs.append(f"subscription {sid} has negative credit")
        return errs

    def observe(self, world, actor):
        st = world.state
        mine = world.principals_of(actor)
        n_c = sum(1 for c in mine for s in st.find("sub_subs", "customer", c) if s["status"] in LIVE)
        n_m = sum(1 for c in mine for s in st.find("sub_subs", "merchant", c) if s["status"] in LIVE)
        return {"subscriptions": n_c, "subscribers": n_m} if (n_c or n_m) else None
