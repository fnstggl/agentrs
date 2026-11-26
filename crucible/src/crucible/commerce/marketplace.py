"""Digital marketplaces run by participants (module `xchg`, host `xchg`).

Anyone with an account (a person, or a company it administers) can open an exchange and set its take rate. Sellers
list their own launched products on any exchange (`products` decides what a buyer receives: goods units, a
repository licence, document access, service entitlement). Buyers choose the exchange and the listing.

Settlement is immediate: the buyer pays the seller (ledger kind `sale`, ref = the order) and the seller pays the
operator its commission (kind `take_rate`, same ref) in the same transaction as the delivery. Buyer protection:
during the exchange's dispute window the buyer can dispute an order; the operator resolves it by deciding the
refunded share (`xchg.resolve`), which the seller pays back (the operator returns the matching part of its
commission). If the seller cannot pay, the operator pays the buyer from its own account (it guarantees its
market). A dispute the operator leaves unresolved for `resolve_days` is refunded in full at the operator's
expense. A full refund revokes the entitlement (goods come back if the buyer still holds them).

Public market information (what any participant can see, as on a real marketplace): each exchange's take rate,
its active listings and sellers, the distinct buyers and orders of the last 30 days, and its dispute record;
each listing's price and its seller's rating on that exchange (ratings only from buyers of settled orders).
Two-sided network effects, trust and pricing are not scripted: they follow from which exchanges buyers and
sellers choose on that information (see `crucible.commerce.customers`).

Passes (wave 3b): a listing may sell a *term licence* (`term_days`): the buyer's entitlement expires after the term
(a 30-day pass to a service, a course access period), so a marketplace sells one-time passes beside the sellers' own
subscriptions - a one-time sale, never recurring revenue (`commerce.revenue` counts it as a sale).

World-generated venues (config `venues`): in a world with the population (`pop`), the default `"auto"` founds two
incumbent marketplaces at genesis with drawn take rates and buyer-protection terms (operator companies with an
endowment that guarantees their buyer protection); a list of venue dicts founds exactly those; `[]` none. A venue
with `policy: "buyer_first"` has a labelled scripted operating rule: its operator decides every dispute after one
day in the buyer's favour (a full refund the seller pays, the operator's commission returned) - the policy a
large real marketplace runs for low-value orders. Who buys there, and from which venue, is the population's own
decision (`economy.work`, exchange channel); which venue a seller lists on is the seller's (agents; scripted
incumbents follow `commerce.rivals`).
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident

DAY = 86400
MAX_TAKE_BPS = 5000
# venues drawn for a population world (config venues="auto"): handle, name, take-rate range (bps), dispute days,
# operating policy (labelled scripted rule, see the module docstring)
AUTO_VENUES = (("bazaarx", "BazaarX", (300, 900), 14, "buyer_first"),
               ("dealmart", "DealMart", (1200, 2500), 7, None))


@register_module
class Exchanges(Module):
    name = "xchg"
    host = "xchg"
    depends = ("identity", "bank", "products")

    def install(self, world):
        st = world.state
        st.declare_table("xchg_exchanges", index=("operator", "status"))
        st.declare_table("xchg_listings", index=("exchange", "seller", "status", "product"))
        st.declare_table("xchg_orders", index=("exchange", "buyer", "seller", "status", "listing"))
        st.declare_table("xchg_ratings", index=("exchange", "seller"))

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY, "xchg.sweep", {}, prio=46, period=DAY, key="xchg.sweep")
        for v in self.venues(ctx.world):
            self._found_venue(ctx, v)

    def venues(self, world) -> list[dict]:
        """The world-generated venues of this world (config `venues`, see the module docstring)."""
        v = self.config.get("venues", "auto")
        if v == "auto":
            if "pop" not in world.modules:
                return []
            rng = world.rng_for("xchg-venues")
            return [{"handle": h, "name": n, "take_rate_bps": int(rng.integers(lo, hi)), "dispute_days": d,
                     "resolve_days": 7, "policy": pol, "cash": 2_000_000}
                    for h, n, (lo, hi), d, pol in AUTO_VENUES]
        if not isinstance(v, list):
            raise InvalidAction("xchg venues must be 'auto' or a list of venues")
        return [dict(x) for x in v]

    def _found_venue(self, ctx: Ctx, v: dict) -> str:
        """World generation: an operator company and its open exchange."""
        idm, bank = ident(ctx.world), ctx.world.modules["bank"]
        oid = idm.create_org(ctx, v["handle"], f"{v['name']} Ltd", None, "company", {"marketplace_operator": True})
        acct = bank.open_account(ctx, oid, "operating")
        if int(v.get("cash", 0)):
            bank.move(ctx, "acct_genesis", acct, int(v["cash"]), "endowment", f"initial capital of {oid}")
        eid = ctx.state.new_id("xch")
        ctx.emit("xchg.exchange", {"id": eid, "name": v["name"], "operator": oid,
                                   "take_rate_bps": int(v["take_rate_bps"]),
                                   "description": f"{v['name']}: buy digital products from independent sellers",
                                   "dispute_days": int(v.get("dispute_days", 7)),
                                   "resolve_days": int(v.get("resolve_days", 7)), "policy": v.get("policy")},
                 vis="public")
        return eid

    # ------------------------------------------------------------------ helpers
    def _ex(self, state, eid: str) -> dict:
        e = state.get("xchg_exchanges", eid)
        if e is None:
            raise NotFound(f"no exchange {eid}")
        return e

    def _order(self, state, oid: str) -> dict:
        o = state.get("xchg_orders", oid)
        if o is None:
            raise NotFound(f"no order {oid}")
        return o

    def _vis(self, o) -> tuple:
        return tuple(sorted({o["buyer"], o["seller"], o["operator"]}))

    def stats(self, state, eid: str, now: int) -> dict:
        """Public 30-day statistics of an exchange. Buyers and orders are counted under the platform trust rule
        (`commerce.platform.counted_buyer`): buyers related to, or recently funded by, the exchange's operator or
        the order's seller are not counted (wash trading with shells does not move the statistics)."""
        from .platform import counted_buyer, side_accounts
        t0 = now - 30 * DAY
        ex = state.tables["xchg_exchanges"][eid]
        op_side = side_accounts(self.world, ex["operator"])
        first: dict[str, int] = {}
        for o in state.find("xchg_orders", "exchange", eid):
            first[o["buyer"]] = min(first.get(o["buyer"], o["t"]), o["t"])
        ok_buyer: dict[tuple[str, str], bool] = {}

        def counted(o):
            k = (o["buyer"], o["seller"])
            if k not in ok_buyer:
                ok_buyer[k] = counted_buyer(self.world, ex["operator"], o["buyer"], first[o["buyer"]], op_side) and \
                    counted_buyer(self.world, o["seller"], o["buyer"], first[o["buyer"]])
            return ok_buyer[k]
        orders = [o for o in state.find("xchg_orders", "exchange", eid) if o["t"] > t0 and counted(o)]
        lst = [x for x in state.find("xchg_listings", "exchange", eid) if x["status"] == "active"]
        disputes = [o for o in orders if o["disputed"]]
        return {"active_listings": len(lst), "sellers": len({x["seller"] for x in lst}),
                "buyers_30d": len({o["buyer"] for o in orders}), "orders_30d": len(orders),
                "disputes_30d": len(disputes),
                "disputes_refunded_30d": sum(1 for o in disputes if o["refunded"] > 0)}

    def close_for_wind_up(self, ctx: Ctx, org: str) -> None:
        st = ctx.state
        for eid in st.find_ids("xchg_exchanges", "operator", org):
            if st.tables["xchg_exchanges"][eid]["status"] == "open":
                self._close(ctx, eid, f"operator {org} ended")
        for lid in st.find_ids("xchg_listings", "seller", org):
            if st.tables["xchg_listings"][lid]["status"] == "active":
                ctx.emit("xchg.listing_status", {"id": lid, "status": "closed"}, vis="public")

    def _close(self, ctx: Ctx, eid: str, why: str) -> None:
        st = ctx.state
        ctx.emit("xchg.exchange_update", {"id": eid, "changes": {"status": "closed"}, "why": why[:200]}, vis="public")
        for lid in st.find_ids("xchg_listings", "exchange", eid):
            if st.tables["xchg_listings"][lid]["status"] == "active":
                ctx.emit("xchg.listing_status", {"id": lid, "status": "closed"}, vis="public")
        for oid in st.find_ids("xchg_orders", "exchange", eid):
            if st.tables["xchg_orders"][oid]["status"] == "disputed":
                contain(ctx, self._refund, ctx, oid, 100, "exchange closed")

    def _refund(self, ctx: Ctx, oid: str, pct: int, note: str, by_operator_only: bool = False) -> int:
        """Refund `pct` % of an order: the seller pays the buyer back (the operator returns its share of the
        commission to the seller); if the seller cannot, the operator pays. Returns the cents refunded."""
        st = ctx.state
        o = st.tables["xchg_orders"][oid]
        bank = ctx.world.modules["bank"]
        amount = int(o["amount"]) * int(pct) // 100 - int(o["refunded"])
        fee_back = int(o["fee"]) * int(pct) // 100 - int(o["fee_refunded"])
        buyer_acct = o["account"] if (st.get("accounts", o["account"]) or {}).get("status") == "open" \
            else bank.primary_account(st, o["buyer"])
        seller_acct = bank.primary_account(st, o["seller"])
        op_acct = bank.primary_account(st, o["operator"])
        if amount <= 0 or buyer_acct is None:
            return 0
        payer = "seller"
        if fee_back > 0 and seller_acct is not None and op_acct is not None:
            bank.move(ctx, op_acct, seller_acct, fee_back, "refund", f"commission returned, order {oid}", ref=oid)
        if by_operator_only or seller_acct is None or bank.available(st, seller_acct) < amount:
            if op_acct is None or bank.available(st, op_acct) < amount:
                raise Forbidden("neither the seller nor the operator can pay the refund")
            bank.move(ctx, op_acct, buyer_acct, amount, "refund", f"buyer protection, order {oid}: {note}"[:300],
                      ref=oid)
            payer = "operator"
        else:
            bank.move(ctx, seller_acct, buyer_acct, amount, "refund", f"reversal of order {oid}: {note}"[:300], ref=oid)
        ctx.emit("xchg.order_status", {"id": oid, "status": "refunded" if pct >= 100 else "resolved",
                                       "refunded": int(o["refunded"]) + amount,
                                       "fee_refunded": int(o["fee_refunded"]) + max(0, fee_back), "paid_by": payer,
                                       "note": note[:200]}, vis=self._vis(o))
        if pct >= 100 and o.get("entitlement"):
            ctx.world.modules["products"].revoke(ctx, o["entitlement"], "refunded", return_goods=True)
        return amount

    # ------------------------------------------------------------------ endpoints: operators
    @endpoint("create", S.obj({"name": S.s_str(minLength=1, maxLength=80),
                               "take_rate_bps": S.s_int("commission in basis points of each sale", minimum=0,
                                                        maximum=MAX_TAKE_BPS),
                               "operator": S.s_str("organisation that runs it (default: you)", maxLength=40),
                               "description": S.s_str(maxLength=2000),
                               "dispute_days": S.s_int(minimum=0, maximum=60),
                               "resolve_days": S.s_int(minimum=1, maximum=60)}, ["name", "take_rate_bps"]), cost_s=300)
    def ep_create(self, ctx, name, take_rate_bps, operator=None, description="", dispute_days=7, resolve_days=7):
        """Open a marketplace you operate. Sellers list there, buyers buy there, you earn the take rate."""
        st = ctx.state
        op = ident(ctx.world).resolve_handle(st, operator) if operator else ctx.actor
        if op != ctx.actor:
            ident(ctx.world).authorize(ctx, op, "org.admin")
        if ctx.world.modules["bank"].primary_account(st, op) is None:
            raise Forbidden("the operator needs an account")
        if len([e for e in st.find("xchg_exchanges", "operator", op) if e["status"] == "open"]) >= 5:
            raise Conflict("exchange limit reached")
        eid = st.new_id("xch")
        ctx.emit("xchg.exchange", {"id": eid, "name": name, "operator": op, "take_rate_bps": int(take_rate_bps),
                                   "description": description[:2000], "dispute_days": int(dispute_days),
                                   "resolve_days": int(resolve_days)}, vis="public")
        return {"exchange": eid}

    @endpoint("set_fee", S.obj({"exchange": S.s_str(maxLength=40),
                                "take_rate_bps": S.s_int(minimum=0, maximum=MAX_TAKE_BPS)},
                               ["exchange", "take_rate_bps"]))
    def ep_set_fee(self, ctx, exchange, take_rate_bps):
        """Change an exchange's take rate (applies to new orders)."""
        e = self._ex(ctx.state, exchange)
        ident(ctx.world).authorize(ctx, e["operator"], "org.admin")
        if e["status"] != "open":
            raise Conflict("exchange is closed")
        ctx.emit("xchg.exchange_update", {"id": exchange, "changes": {"take_rate_bps": int(take_rate_bps)}, "why": ""},
                 vis="public")
        return {"ok": True}

    @endpoint("close", S.obj({"exchange": S.s_str(maxLength=40)}, ["exchange"]))
    def ep_close(self, ctx, exchange):
        """Close an exchange you operate (listings close; open disputes are refunded)."""
        e = self._ex(ctx.state, exchange)
        ident(ctx.world).authorize(ctx, e["operator"], "org.admin")
        if e["status"] != "open":
            raise Conflict("already closed")
        self._close(ctx, exchange, "closed by operator")
        return {"ok": True}

    @endpoint("resolve", S.obj({"order": S.s_str(maxLength=40), "refund_pct": S.s_int(minimum=0, maximum=100),
                                "note": S.s_str(maxLength=500)}, ["order", "refund_pct"]))
    def ep_resolve(self, ctx, order, refund_pct, note=""):
        """Operator decides a dispute: the share of the price refunded to the buyer (0 rejects the dispute)."""
        st = ctx.state
        o = self._order(st, order)
        ident(ctx.world).authorize(ctx, o["operator"], "org.admin")
        if o["status"] != "disputed":
            raise Conflict(f"order is {o['status']}")
        if refund_pct == 0:
            ctx.emit("xchg.order_status", {"id": order, "status": "settled", "refunded": o["refunded"],
                                           "fee_refunded": o["fee_refunded"], "paid_by": None,
                                           "note": f"dispute rejected: {note}"[:200]}, vis=self._vis(o))
            return {"refunded_cents": 0}
        return {"refunded_cents": self._refund(ctx, order, refund_pct, note or "dispute upheld")}

    # ------------------------------------------------------------------ endpoints: sellers
    @endpoint("list", S.obj({"exchange": S.s_str(maxLength=40), "product": S.s_str(maxLength=40),
                             "price_cents": S.s_int(minimum=1, maximum=10**9), "title": S.s_str(maxLength=120),
                             "term_days": S.s_int("sell a pass: the buyer's access expires after this many days "
                                                  "(default: perpetual)", minimum=1, maximum=3650)},
                            ["exchange", "product", "price_cents"]), cost_s=30)
    def ep_list(self, ctx, exchange, product, price_cents, title="", term_days=None):
        """List one of your launched products on an exchange at a price (per unit for goods), optionally as a pass
        whose access expires after `term_days`."""
        st = ctx.state
        e = self._ex(st, exchange)
        if e["status"] != "open":
            raise Conflict("exchange is closed")
        prods = ctx.world.modules["products"]
        p = prods.product(st, product)
        ident(ctx.world).authorize(ctx, p["owner"], "org.admin")
        if p["status"] != "launched":
            raise Conflict("launch the product first")
        for x in st.find("xchg_listings", "product", product):
            if x["exchange"] == exchange and x["status"] == "active":
                raise Conflict("already listed on this exchange (delist first)")
        if term_days is not None and (p["delivery"] or {}).get("type") == "goods":
            raise InvalidAction("goods are sold outright (no term)")
        lid = st.new_id("xl")
        ctx.emit("xchg.listing", {"id": lid, "exchange": exchange, "product": product, "seller": p["owner"],
                                  "price": int(price_cents), "title": (title or p["name"])[:120],
                                  "delivery": (p["delivery"] or {}).get("type"),
                                  "term_days": None if term_days is None else int(term_days)}, vis="public")
        return {"listing": lid}

    @endpoint("delist", S.obj({"listing": S.s_str(maxLength=40)}, ["listing"]))
    def ep_delist(self, ctx, listing):
        """Withdraw a listing."""
        x = ctx.state.get("xchg_listings", listing)
        if x is None:
            raise NotFound("no listing")
        ident(ctx.world).authorize(ctx, x["seller"], "org.admin")
        if x["status"] != "active":
            raise Conflict("not active")
        ctx.emit("xchg.listing_status", {"id": listing, "status": "closed"}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ endpoints: buyers
    @endpoint("buy", S.obj({"listing": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=1000),
                            "account": S.s_str(maxLength=40)}, ["listing"]), cost_s=20)
    def ep_buy(self, ctx, listing, qty=1, account=None):
        """Buy from a listing: you pay the seller now and receive the product; dispute within the exchange's
        window if it is not what was promised."""
        st = ctx.state
        x = st.get("xchg_listings", listing)
        if x is None or x["status"] != "active":
            raise NotFound("no active listing")
        e = self._ex(st, x["exchange"])
        if e["status"] != "open":
            raise Conflict("exchange is closed")
        prods = ctx.world.modules["products"]
        p = prods.product(st, x["product"])
        if p["status"] != "launched":
            raise Conflict("product is not on sale")
        if qty != 1 and x["delivery"] != "goods":
            raise InvalidAction("qty applies to goods")
        amount = int(x["price"]) * int(qty)
        acct, buyer = prods._account_of_caller(ctx, account, amount)
        if buyer == x["seller"]:
            raise Forbidden("cannot buy your own listing")
        bank = ctx.world.modules["bank"]
        seller_acct = bank.primary_account(st, x["seller"])
        op_acct = bank.primary_account(st, e["operator"])
        if seller_acct is None or op_acct is None:
            raise Forbidden("this listing cannot be settled")
        oid = st.new_id("xo")
        txn = bank.move(ctx, acct, seller_acct, amount, "sale", f"{x['title']} on {e['name']}", ref=oid)
        fee = amount * int(e["take_rate_bps"]) // 10_000
        if fee and x["seller"] != e["operator"]:
            bank.move(ctx, seller_acct, op_acct, fee, "take_rate", f"commission {e['name']} order {oid}", ref=oid)
        elif x["seller"] == e["operator"]:
            fee = 0
        term = x.get("term_days")
        eid = prods.grant(ctx, x["product"], buyer, "exchange", oid, qty=qty,
                          expires=None if term is None else ctx.now + int(term) * DAY)
        ctx.emit("xchg.order", {"id": oid, "exchange": x["exchange"], "listing": listing, "product": x["product"],
                                "buyer": buyer, "seller": x["seller"], "operator": e["operator"], "account": acct,
                                "qty": int(qty), "amount": amount, "fee": fee, "txn": txn, "entitlement": eid,
                                "kind": x["delivery"], "dispute_until": ctx.now + int(e["dispute_days"]) * DAY},
                 vis=tuple(sorted({buyer, x["seller"], e["operator"]})))
        return {"order": oid, "amount": amount, "entitlement": eid}

    @endpoint("dispute", S.obj({"order": S.s_str(maxLength=40), "reason": S.s_str(minLength=1, maxLength=1000)},
                               ["order", "reason"]))
    def ep_dispute(self, ctx, order, reason):
        """Dispute an order within the exchange's dispute window; the operator decides."""
        st = ctx.state
        o = self._order(st, order)
        if o["buyer"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("only the buyer disputes")
        if o["status"] != "settled" or o["disputed"]:
            raise Conflict("this order cannot be disputed")
        if ctx.now > o["dispute_until"]:
            raise Conflict("the dispute window has passed")
        ctx.emit("xchg.order_status", {"id": order, "status": "disputed", "refunded": o["refunded"],
                                       "fee_refunded": o["fee_refunded"], "paid_by": None, "note": reason[:200]},
                 vis=self._vis(o))
        return {"ok": True}

    @endpoint("rate", S.obj({"order": S.s_str(maxLength=40), "stars": S.s_int(minimum=1, maximum=5),
                             "text": S.s_str(maxLength=500)}, ["order", "stars"]))
    def ep_rate(self, ctx, order, stars, text=""):
        """Rate the seller of an order you bought (settled, not refunded in full); one rating per order."""
        st = ctx.state
        o = self._order(st, order)
        if o["buyer"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("only the buyer rates")
        if o["status"] not in ("settled", "resolved") or o["refunded"] >= o["amount"]:
            raise Conflict("only settled orders can be rated")
        if st.has("xchg_ratings", order):
            raise Conflict("already rated")
        ctx.emit("xchg.rating", {"id": order, "exchange": o["exchange"], "seller": o["seller"], "stars": int(stars),
                                 "text": text[:500]}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ endpoints: discovery
    def seller_rating(self, state, eid: str, seller: str) -> tuple[float | None, int]:
        rs = [r["stars"] for r in state.find("xchg_ratings", "seller", seller) if r["exchange"] == eid]
        return (round(sum(rs) / len(rs), 2) if rs else None), len(rs)

    @endpoint("exchanges", S.obj({"limit": S.s_int(minimum=1, maximum=50)}), read_only=True)
    def ep_exchanges(self, ctx, limit=20):
        """Open exchanges with their take rate and public 30-day statistics."""
        st = ctx.state
        out = []
        for eid in st.find_ids("xchg_exchanges", "status", "open")[:limit]:
            e = st.tables["xchg_exchanges"][eid]
            out.append(dict({"exchange": eid, "name": e["name"], "operator": e["operator"],
                             "take_rate_bps": e["take_rate_bps"], "dispute_days": e["dispute_days"]},
                            **self.stats(st, eid, ctx.now)))
        return out

    @endpoint("browse", S.obj({"exchange": S.s_str(maxLength=40), "product": S.s_str(maxLength=40),
                               "delivery": S.s_enum(["service", "repo", "content", "goods"]),
                               "sort": S.s_enum(["price", "rating", "new"]), "limit": S.s_int(minimum=1, maximum=50)}),
              read_only=True)
    def ep_browse(self, ctx, exchange=None, product=None, delivery=None, sort="price", limit=20):
        """Active listings (on one exchange or all), with their sellers' ratings on that exchange."""
        st = ctx.state
        if exchange is not None:
            ids = st.find_ids("xchg_listings", "exchange", exchange)
        elif product is not None:
            ids = st.find_ids("xchg_listings", "product", product)
        else:
            ids = st.find_ids("xchg_listings", "status", "active")
        rows = []
        for lid in ids:
            x = st.tables["xchg_listings"][lid]
            if x["status"] != "active" or (delivery is not None and x["delivery"] != delivery):
                continue
            if product is not None and x["product"] != product:
                continue
            avg, n = self.seller_rating(st, x["exchange"], x["seller"])
            rows.append({"listing": lid, "exchange": x["exchange"], "product": x["product"], "seller": x["seller"],
                         "title": x["title"], "price_cents": x["price"], "delivery": x["delivery"],
                         "seller_rating": avg, "seller_ratings": n, "term_days": x.get("term_days"),
                         "t": x["t"]})
        if sort == "price":
            rows.sort(key=lambda r: (r["price_cents"], r["listing"]))
        elif sort == "rating":
            rows.sort(key=lambda r: (-(r["seller_rating"] or 0), r["listing"]))
        else:
            rows.sort(key=lambda r: r["listing"], reverse=True)
        return rows[:limit]

    @endpoint("mine", S.obj({"role": S.s_enum(["buyer", "seller", "operator"])}), read_only=True)
    def ep_mine(self, ctx, role="buyer"):
        """Your orders as buyer or seller, or the orders on exchanges you operate."""
        st = ctx.state
        out = []
        for who in sorted(ctx.world.principals_of(ctx.actor)):
            if role == "operator":
                for eid in st.find_ids("xchg_exchanges", "operator", who):
                    out.extend(dict(o, id=oid) for oid in st.find_ids("xchg_orders", "exchange", eid)
                               for o in [st.tables["xchg_orders"][oid]])
            else:
                out.extend(dict(o, id=oid) for oid in st.find_ids("xchg_orders", role, who)
                           for o in [st.tables["xchg_orders"][oid]])
        return out[-100:]

    # ------------------------------------------------------------------ process
    @process("sweep")
    def pr_sweep(self, ctx):
        st = ctx.state
        for oid in st.find_ids("xchg_orders", "status", "disputed"):
            o = st.tables["xchg_orders"][oid]
            e = st.tables["xchg_exchanges"][o["exchange"]]
            if e.get("policy") == "buyer_first" and e["status"] == "open" and ctx.now - o["updated"] >= DAY:
                # the venue's labelled scripted operating rule: disputes are decided for the buyer after a day
                err = contain(ctx, self._refund, ctx, oid, 100, "buyer protection (venue policy)")
                if err is not None:
                    ctx.emit("xchg.sweep_failed", {"id": oid, "error": f"{type(err).__name__}: {err}"[:200]},
                             vis="internal")
                continue
            if ctx.now - o["updated"] >= int(e["resolve_days"]) * DAY:
                err = contain(ctx, self._refund, ctx, oid, 100, "unresolved by the operator", True)
                if err is not None:
                    ctx.emit("xchg.sweep_failed", {"id": oid, "error": f"{type(err).__name__}: {err}"[:200]},
                             vis="internal")

    # ------------------------------------------------------------------ appliers
    @applier("xchg.exchange")
    def ap_exchange(self, ev):
        p = ev.payload
        self.world.state.insert("xchg_exchanges", p["id"], {"name": p["name"], "operator": p["operator"],
                                                           "take_rate_bps": p["take_rate_bps"],
                                                           "description": p["description"],
                                                           "dispute_days": p["dispute_days"],
                                                           "resolve_days": p["resolve_days"], "status": "open",
                                                           "policy": p.get("policy"), "t": ev.t})

    @applier("xchg.exchange_update")
    def ap_exchange_update(self, ev):
        self.world.state.patch("xchg_exchanges", ev.payload["id"], **ev.payload["changes"])

    @applier("xchg.listing")
    def ap_listing(self, ev):
        p = ev.payload
        self.world.state.insert("xchg_listings", p["id"], {"exchange": p["exchange"], "product": p["product"],
                                                          "seller": p["seller"], "price": p["price"],
                                                          "title": p["title"], "delivery": p["delivery"],
                                                          "status": "active", "sold": 0,
                                                          "term_days": p.get("term_days"), "t": ev.t})

    @applier("xchg.listing_status")
    def ap_listing_status(self, ev):
        self.world.state.patch("xchg_listings", ev.payload["id"], status=ev.payload["status"])

    @applier("xchg.order")
    def ap_order(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("xchg_orders", p["id"], {k: p[k] for k in ("exchange", "listing", "product", "buyer", "seller",
                                                              "operator", "account", "qty", "amount", "fee", "txn",
                                                              "entitlement", "kind", "dispute_until")})
        st.patch("xchg_orders", p["id"], status="settled", disputed=False, refunded=0, fee_refunded=0, t=ev.t,
                 updated=ev.t, paid_by=None, note="")
        x = st.tables["xchg_listings"][p["listing"]]
        st.patch("xchg_listings", p["listing"], sold=x["sold"] + p["qty"])

    @applier("xchg.order_status")
    def ap_order_status(self, ev):
        p = ev.payload
        st = self.world.state
        o = st.tables["xchg_orders"][p["id"]]
        st.patch("xchg_orders", p["id"], status=p["status"], refunded=p["refunded"], fee_refunded=p["fee_refunded"],
                 paid_by=p["paid_by"] or o["paid_by"], note=p["note"], updated=ev.t,
                 disputed=o["disputed"] or p["status"] == "disputed")

    @applier("xchg.rating")
    def ap_rating(self, ev):
        p = ev.payload
        self.world.state.insert("xchg_ratings", p["id"], {"exchange": p["exchange"], "seller": p["seller"],
                                                         "stars": p["stars"], "text": p["text"], "t": ev.t})

    @applier("xchg.sweep_failed")
    def ap_sweep_failed(self, ev):
        pass

    def check_invariants(self, world):
        st = world.state
        errs = []
        for oid, o in st.tables["xchg_orders"].items():
            if not 0 <= o["refunded"] <= o["amount"] or not 0 <= o["fee_refunded"] <= o["fee"]:
                errs.append(f"order {oid} refunds out of range")
            if o["fee"] > o["amount"]:
                errs.append(f"order {oid} commission exceeds the price")
        return errs
