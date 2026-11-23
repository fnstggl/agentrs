"""Marketplace (host `market`): listings, escrowed orders, deliveries, disputes, verified ratings, inventory.

Listing kinds:
* `api`     - a live service endpoint sold per request. Simulated customers find these (search, browse, word of
              mouth, ads) and are billed per request by the demand mechanism; direct callers pay the endpoint price.
* `code`    - a licence to a repository: buying grants `code.read` (lets the buyer depend on / fork it).
* `goods`   - inventory items (conserved; escrow prevents double sale).
* `service` - labour or consulting delivered off-ledger; escrow, confirmation, dispute and arbitration.

Ratings are only accepted from principals that actually paid the seller money (completed orders of a positive
amount, or paid API calls), so reputation cannot be manufactured without spending money - and self-dealing is
excluded (nobody related to the seller can rate it).

Settlement never depends on one order: the hourly sweep settles every order on its own, and an order whose seller
or buyer has ceased to exist (company wound up, accounts frozen) is settled to whoever can still receive the money
- the buyer first, then the seller - or, if nobody can, forfeited to the fee account with a record. A company that
is wound up settles its own open orders first (`settle_for_wind_up`).
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InsufficientFunds, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from .identity import ident

ESCROW = "acct_escrow"
AUTO_CONFIRM_S = 3 * 86400
ARBITRATION_S = 7 * 86400


@register_module
class Market(Module):
    name = "market"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("listings", index=("seller", "kind", "need", "status", "host"))
        st.declare_table("orders", index=("buyer", "seller", "status", "listing"))
        st.declare_table("ratings", index=("listing", "seller"))
        st.declare_table("inventory", index=("owner", "sku"))
        st.declare_table("api_customers", index=("host",))
        world.hooks["reputation"] = self.reputation
        world.hooks["on_api_purchase"] = self.on_api_purchase
        prev = world.hooks.get("rank_signal")
        world.hooks["rank_signal"] = lambda w, doc: self.rank_signal(w, doc) * (prev(w, doc) if prev else 1.0)

    def genesis(self, ctx):
        ctx.emit("bank.account", {"id": ESCROW, "owner": SYSTEM, "kind": "escrow", "name": "market escrow",
                                  "overdraft": 0}, vis="internal")
        ctx.schedule(ctx.now + 3600, "market.sweep", {}, prio=40, period=3600, key="market.sweep")

    # ------------------------------------------------------------------ reputation
    def reputation(self, world, principal: str) -> dict | None:
        st = world.state
        n, s = 0, 0
        for lid in st.find_ids("listings", "seller", principal):
            l = st.tables["listings"][lid]
            n += l["rating_n"]
            s += l["rating_sum"]
        return {"ratings": n, "avg": round(s / n, 3) if n else None}

    def rank_signal(self, world, doc) -> float:
        if doc.get("kind") != "listing":
            return 1.0
        l = world.state.get("listings", doc.get("ref") or "")
        if l is None or l["status"] != "active":
            return 0.0
        if l["rating_n"] == 0:
            return 1.0
        avg = l["rating_sum"] / l["rating_n"]
        conf = l["rating_n"] / (l["rating_n"] + 10)
        return 0.5 + conf * (avg / 5.0) + (1 - conf) * 0.5

    def on_api_purchase(self, ctx, host: str, payer: str, cents: int) -> None:
        key = f"{host}|{payer}"
        ctx.emit("market.api_customer", {"id": key, "host": host, "payer": payer, "cents": cents}, vis="internal")

    # ------------------------------------------------------------------ endpoints
    @endpoint("list", S.obj({"kind": S.s_enum(["api", "code", "goods", "service"]), "title": S.s_str(maxLength=120),
                             "description": S.s_str(maxLength=3000), "price_cents": S.s_int(minimum=0, maximum=10**9),
                             "seller": S.s_str("org handle (default you)", maxLength=40),
                             "host": S.s_str("api: deployed host", maxLength=31),
                             "endpoint": S.s_str("api: endpoint name", maxLength=31),
                             "need": S.s_str("api: the public need format served, e.g. route.tsp", maxLength=40),
                             "repo": S.s_str("code: repository licensed", maxLength=81),
                             "sku": S.s_str("goods: item type", maxLength=40)},
                            ["kind", "title", "price_cents"]), cost_s=60)
    def ep_list(self, ctx, kind, title, price_cents, description="", seller=None, host=None, endpoint=None,
                need=None, repo=None, sku=None):
        """Publish a listing (indexed by search). A listing fee is charged to deter spam."""
        st = ctx.state
        seller = ident(ctx.world).resolve_handle(st, seller) if seller else ctx.actor
        if seller != ctx.actor:
            ident(ctx.world).authorize(ctx, seller, "org.admin")
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(st, seller)
        if acct is None:
            raise Forbidden("sellers need a bank account")
        hosts = []
        if kind == "api":
            cloud = ctx.world.modules.get("cloud")
            dep = cloud.live_deployment(st, host or "") if cloud else None
            if dep is None or dep["owner"] != seller:
                raise InvalidAction("api listings need a live deployment you operate")
            if endpoint not in dep["manifest"]["endpoints"]:
                raise InvalidAction(f"{host} has no endpoint {endpoint!r}")
            hosts = [host]
        if need is not None:
            from ..economy.needs import NEEDS
            if need not in NEEDS:
                raise InvalidAction(f"unknown need {need!r}")
        if kind == "api":
            pass
        elif kind == "code":
            code = ctx.world.modules["code"]
            r = code.repo(st, repo or "")
            if r["owner"] != seller:
                raise Forbidden("you can only license your own repositories")
        elif kind == "goods":
            inv = st.get("inventory", f"{seller}|{sku}")
            if inv is None or inv["qty"] <= 0:
                raise InvalidAction("you have no such goods")
        fee = int(ctx.world.spec.params.get("listing_fee_cents", 500))
        if fee:
            bank.move(ctx, acct, "acct_fees", fee, "fee", "listing fee")
        lid = st.new_id("lst")
        ctx.emit("market.listing", {"id": lid, "seller": seller, "kind": kind, "title": title, "description": description,
                                    "price": price_cents, "host": host, "endpoint": endpoint, "need": need,
                                    "repo": repo, "sku": sku}, vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            search.index(ctx, f"listing:{lid}", "listing", title, f"{title} {description} {need or ''}", seller,
                         ref=lid, hosts=hosts)
        return {"listing": lid}

    @endpoint("update", S.obj({"listing": S.s_str(maxLength=40), "price_cents": S.s_int(minimum=0, maximum=10**9),
                               "description": S.s_str(maxLength=3000), "status": S.s_enum(["active", "closed"])},
                              ["listing"]))
    def ep_update(self, ctx, listing, price_cents=None, description=None, status=None):
        """Change a listing's price, description or status (seller's admins)."""
        l = ctx.state.get("listings", listing)
        if l is None:
            raise NotFound("no listing")
        # owners/admins always may; a delegate needs `market.manage` scoped to this listing (e.g. a pricing service)
        ident(ctx.world).authorize(ctx, l["seller"], "market.manage", scope=listing)
        ch = {k: v for k, v in (("price", price_cents), ("description", description), ("status", status)) if v is not None}
        if not ch:
            raise InvalidAction("nothing to change")
        ctx.emit("market.listing_update", {"id": listing, "changes": ch}, vis="public")
        search = ctx.world.modules.get("search")
        if search is not None:
            if ch.get("status") == "closed":
                search.unindex(ctx, f"listing:{listing}")
            elif description is not None:
                search.index(ctx, f"listing:{listing}", "listing", l["title"], f"{l['title']} {description} {l['need'] or ''}",
                             l["seller"], ref=listing, hosts=[l["host"]] if l["host"] else [])
        return {"ok": True}

    def _public(self, lid, l):
        return {"listing": lid, "seller": l["seller"], "kind": l["kind"], "title": l["title"], "price_cents": l["price"],
                "need": l["need"], "host": l["host"], "endpoint": l["endpoint"], "status": l["status"],
                "rating": round(l["rating_sum"] / l["rating_n"], 2) if l["rating_n"] else None,
                "ratings": l["rating_n"], "sales": l["sales"]}

    @endpoint("browse", S.obj({"kind": S.s_enum(["api", "code", "goods", "service"]), "need": S.s_str(maxLength=40),
                               "sort": S.s_enum(["new", "price", "rating"]), "limit": S.s_int(minimum=1, maximum=30)}),
              read_only=True)
    def ep_browse(self, ctx, kind=None, need=None, sort="new", limit=10):
        """Browse active listings (by kind, need, price, rating or recency). Seen hosts become known to you."""
        st = ctx.state
        ids = st.find_ids("listings", "need", need) if need else (st.find_ids("listings", "kind", kind) if kind else st.ids("listings"))
        rows = [(lid, st.tables["listings"][lid]) for lid in ids]
        rows = [(i, l) for i, l in rows if l["status"] == "active" and (kind is None or l["kind"] == kind)]
        if sort == "price":
            rows.sort(key=lambda r: (r[1]["price"], r[0]))
        elif sort == "rating":
            rows.sort(key=lambda r: (-(r[1]["rating_sum"] / r[1]["rating_n"]) if r[1]["rating_n"] else 0, r[0]))
        else:
            rows.sort(key=lambda r: r[0], reverse=True)
        out = [self._public(i, l) for i, l in rows[:limit]]
        for r in out:
            if r["host"]:
                ctx.world.learn(ctx, ctx.actor, "host", r["host"])
        return out

    @endpoint("view", S.obj({"listing": S.s_str(maxLength=40)}, ["listing"]), read_only=True)
    def ep_view(self, ctx, listing):
        """Details of a listing with recent verified reviews; its host becomes known to you."""
        l = ctx.state.get("listings", listing)
        if l is None:
            raise NotFound("no listing")
        out = self._public(listing, l)
        out["description"] = l["description"]
        out["recent_reviews"] = [{"stars": r["stars"], "text": r["text"], "by": r["buyer"], "t": r["t"]}
                                 for r in ctx.state.find("ratings", "listing", listing)[-5:]]
        if l["host"]:
            ctx.world.learn(ctx, ctx.actor, "host", l["host"])
        return out

    @endpoint("buy", S.obj({"listing": S.s_str(maxLength=40), "qty": S.s_int(minimum=1, maximum=10_000),
                            "account": S.s_str(maxLength=40)}, ["listing", "account"]), cost_s=20)
    def ep_buy(self, ctx, listing, account, qty=1):
        """Buy a code licence, goods or a service. Payment is held in escrow until delivery is confirmed."""
        st = ctx.state
        l = st.get("listings", listing)
        if l is None or l["status"] != "active":
            raise NotFound("no active listing")
        if l["kind"] == "api":
            raise InvalidAction(f"api listings are used by calling {l['host']}.{l['endpoint']}")
        bank = ctx.world.modules["bank"]
        buyer = bank.account_owner(st, account)
        ident(ctx.world).authorize(ctx, buyer, "bank.spend", amount=l["price"] * qty, scope=account)
        if buyer == l["seller"]:
            raise Forbidden("cannot buy from yourself")
        if l["kind"] in ("code",) and qty != 1:
            raise InvalidAction("licences are bought one at a time")
        amount = l["price"] * qty
        oid = st.new_id("ord")
        if amount:
            bank.move(ctx, account, ESCROW, amount, "escrow", f"order {oid}", ref=oid)
        if l["kind"] == "goods":
            inv = st.get("inventory", f"{l['seller']}|{l['sku']}")
            if inv is None or inv["qty"] < qty:
                raise InsufficientFunds("seller does not have that many items in stock")
            ctx.emit("market.stock", {"owner": l["seller"], "sku": l["sku"], "delta": -qty}, vis=(l["seller"],))
        ctx.emit("market.order", {"id": oid, "listing": listing, "buyer": buyer, "seller": l["seller"], "qty": qty,
                                  "amount": amount, "account": account, "kind": l["kind"]}, vis=(buyer, l["seller"]))
        if l["kind"] == "code":
            ctx.emit("id.grant", {"id": st.new_id("grant"), "grantee": buyer, "owner": l["seller"], "cap": "code.read",
                                  "scope": l["repo"], "limit": None, "expires": None, "by": SYSTEM},
                     vis=(buyer, l["seller"]))
            self._complete(ctx, oid)
        return {"order": oid, "amount": amount}

    @endpoint("orders", S.obj({"role": S.s_enum(["buyer", "seller"])}), read_only=True)
    def ep_orders(self, ctx, role="buyer"):
        """Your orders as buyer or seller."""
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for oid in ctx.state.find_ids("orders", role, p):
                out.append(dict(ctx.state.tables["orders"][oid], id=oid))
        return out[-50:]

    def _order_for(self, ctx, order, role):
        o = ctx.state.get("orders", order)
        if o is None:
            raise NotFound("no order")
        ident(ctx.world).authorize(ctx, o[role], "org.admin" if o[role].startswith("o:") else "bank.read")
        return o

    @endpoint("deliver", S.obj({"order": S.s_str(maxLength=40), "note": S.s_str(maxLength=1000)}, ["order"]))
    def ep_deliver(self, ctx, order, note=""):
        """Seller marks an order delivered (goods move to the buyer)."""
        o = self._order_for(ctx, order, "seller")
        if o["status"] != "paid":
            raise Conflict(f"order is {o['status']}")
        if o["kind"] == "goods":
            ctx.emit("market.stock", {"owner": o["buyer"], "sku": ctx.state.tables["listings"][o["listing"]]["sku"],
                                      "delta": o["qty"]}, vis=(o["buyer"],))
        ctx.emit("market.order_status", {"id": order, "status": "delivered", "note": note[:1000]},
                 vis=(o["buyer"], o["seller"]))
        return {"ok": True}

    @endpoint("confirm", S.obj({"order": S.s_str(maxLength=40)}, ["order"]))
    def ep_confirm(self, ctx, order):
        """Buyer confirms delivery; escrow is released to the seller."""
        o = self._order_for(ctx, order, "buyer")
        if o["status"] not in ("delivered", "paid"):
            raise Conflict(f"order is {o['status']}")
        if o["status"] == "paid" and o["kind"] == "goods":
            raise Conflict("goods must be delivered first")
        self._complete(ctx, order)
        return {"ok": True}

    @endpoint("dispute", S.obj({"order": S.s_str(maxLength=40), "reason": S.s_str(maxLength=1000)}, ["order", "reason"]))
    def ep_dispute(self, ctx, order, reason):
        """Buyer disputes an order; unresolved disputes are arbitrated 50/50 after 7 days."""
        o = self._order_for(ctx, order, "buyer")
        if o["status"] not in ("paid", "delivered"):
            raise Conflict(f"order is {o['status']}")
        ctx.emit("market.order_status", {"id": order, "status": "disputed", "note": reason[:1000]},
                 vis=(o["buyer"], o["seller"]))
        return {"ok": True, "arbitration_after_s": ARBITRATION_S}

    @endpoint("refund", S.obj({"order": S.s_str(maxLength=40)}, ["order"]))
    def ep_refund(self, ctx, order):
        """Seller refunds an order from escrow."""
        o = self._order_for(ctx, order, "seller")
        if o["status"] not in ("paid", "delivered", "disputed"):
            raise Conflict(f"order is {o['status']}")
        self._refund(ctx, order, 1.0)
        return {"ok": True}

    @endpoint("rate", S.obj({"listing": S.s_str(maxLength=40), "stars": S.s_int(minimum=1, maximum=5),
                             "text": S.s_str(maxLength=1000)}, ["listing", "stars"]))
    def ep_rate(self, ctx, listing, stars, text=""):
        """Rate a listing you paid for (completed order, or at least one paid API call). One rating per buyer."""
        st = ctx.state
        l = st.get("listings", listing)
        if l is None:
            raise NotFound("no listing")
        buyers = ctx.world.principals_of(ctx.actor)
        from ..goals.metrics import is_related
        if l["seller"] in buyers or is_related(ctx.world, l["seller"], ctx.actor):
            raise Forbidden("sellers cannot rate themselves")
        paid = any(o["buyer"] in buyers and o["status"] == "completed" and o["amount"] > 0
                   for o in st.find("orders", "listing", listing))
        if not paid and l["kind"] == "api":
            paid = any((st.get("api_customers", f"{l['host']}|{b}") or {}).get("cents", 0) > 0 for b in buyers)
        if not paid:
            raise Forbidden("only paying customers can rate")
        rid = f"{listing}|{ctx.actor}"
        if st.has("ratings", rid):
            raise Conflict("already rated")
        self.add_rating(ctx, rid, listing, ctx.actor, stars, text)
        return {"ok": True}

    def add_rating(self, ctx, rid, listing, buyer, stars, text=""):
        l = ctx.state.tables["listings"][listing]
        ctx.emit("market.rating", {"id": rid, "listing": listing, "seller": l["seller"], "buyer": buyer,
                                   "stars": int(stars), "text": text[:1000]}, vis="public")

    @endpoint("inventory", S.obj(), read_only=True)
    def ep_inventory(self, ctx):
        """Goods held by you or organisations you administer."""
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for iid in ctx.state.find_ids("inventory", "owner", p):
                r = ctx.state.tables["inventory"][iid]
                out.append({"owner": p, "sku": r["sku"], "qty": r["qty"]})
        return out

    # ------------------------------------------------------------------ settlement
    def _complete(self, ctx, oid):
        o = ctx.state.tables["orders"][oid]
        bank = ctx.world.modules["bank"]
        if o["amount"]:
            dst = bank.primary_account(ctx.state, o["seller"])
            if dst is None:
                raise Forbidden("seller has no account")
            bank.move(ctx, ESCROW, dst, o["amount"], "sale", f"order {oid}", ref=oid)
        ctx.emit("market.order_status", {"id": oid, "status": "completed", "note": ""}, vis=(o["buyer"], o["seller"]))

    def _buyer_account(self, st, o) -> str | None:
        """Where a refund can go: the account the order was paid from if it is still open, else the buyer's
        primary account, else nowhere."""
        a = st.get("accounts", o["account"])
        if a is not None and a["status"] == "open":
            return o["account"]
        return self.world.modules["bank"].primary_account(st, o["buyer"])

    def _refund(self, ctx, oid, frac):
        o = ctx.state.tables["orders"][oid]
        bank = ctx.world.modules["bank"]
        back = int(o["amount"] * frac)
        if back:
            bank.move(ctx, ESCROW, o["account"], back, "refund", f"order {oid}", ref=oid)
        rest = o["amount"] - back
        if rest:
            bank.move(ctx, ESCROW, bank.primary_account(ctx.state, o["seller"]), rest, "sale", f"order {oid} (arbitrated)", ref=oid)
        if o["kind"] == "goods" and not o.get("shipped"):
            ctx.emit("market.stock", {"owner": o["seller"], "sku": ctx.state.tables["listings"][o["listing"]]["sku"],
                                      "delta": o["qty"]}, vis=(o["seller"],))
        ctx.emit("market.order_status", {"id": oid, "status": "refunded" if frac == 1.0 else "arbitrated", "note": ""},
                 vis=(o["buyer"], o["seller"]))

    def _settle_orphan(self, ctx, oid, why: str) -> None:
        """An open order one of whose sides can no longer receive money. The whole amount goes to the side that
        still can (the buyer gets its money back; failing that the seller is paid); if neither can, it is
        forfeited to the fee account. Goods not yet shipped return to the seller's stock. Never raises for a
        missing account."""
        st = ctx.state
        o = st.tables["orders"][oid]
        bank = ctx.world.modules["bank"]
        back, seller = self._buyer_account(st, o), bank.primary_account(st, o["seller"])
        status = "refunded"
        if o["amount"]:
            if back is not None:
                bank.move(ctx, ESCROW, back, o["amount"], "refund", f"order {oid} ({why})", ref=oid)
            elif seller is not None:
                bank.move(ctx, ESCROW, seller, o["amount"], "sale", f"order {oid} ({why})", ref=oid)
                status = "completed"
            else:
                bank.move(ctx, ESCROW, "acct_fees", o["amount"], "forfeit", f"order {oid} ({why})", ref=oid)
                status = "void"
        if status != "completed" and o["kind"] == "goods" and not o.get("shipped"):
            ctx.emit("market.stock", {"owner": o["seller"], "sku": st.tables["listings"][o["listing"]]["sku"],
                                      "delta": o["qty"]}, vis=(o["seller"],))
        ctx.emit("market.order_status", {"id": oid, "status": status, "note": why[:200]}, vis=(o["buyer"], o["seller"]))

    def _settle_due(self, ctx, oid, frac) -> None:
        """One order of the sweep: automatic confirmation (frac None) or arbitration (frac = buyer's share)."""
        st = ctx.state
        o = st.tables["orders"][oid]
        bank = ctx.world.modules["bank"]
        seller_ok = bank.primary_account(st, o["seller"]) is not None
        a = st.get("accounts", o["account"])
        buyer_ok = a is not None and a["status"] == "open"
        if o["amount"] and (not seller_ok or (frac is not None and not buyer_ok)):
            self._settle_orphan(ctx, oid, "seller no longer exists" if not seller_ok else "buyer no longer exists")
        elif frac is None:
            self._complete(ctx, oid)
        else:
            self._refund(ctx, oid, frac)

    @process("sweep")
    def pr_sweep(self, ctx):
        st = ctx.state
        due = [(oid, None) for oid in st.find_ids("orders", "status", "delivered")
               if ctx.now - st.tables["orders"][oid]["updated"] >= AUTO_CONFIRM_S]
        due += [(oid, 0.5) for oid in st.find_ids("orders", "status", "disputed")
                if ctx.now - st.tables["orders"][oid]["updated"] >= ARBITRATION_S]
        for oid, frac in due:        # every order on its own: one that cannot be settled never blocks the others
            err = contain(ctx, self._settle_due, ctx, oid, frac)
            if err is not None:
                ctx.emit("market.settle_failed", {"id": oid, "error": f"{type(err).__name__}: {err}"[:200]},
                         vis="internal")

    def settle_for_wind_up(self, ctx, org: str) -> None:
        """Called before a company's accounts are frozen (dissolution or bankruptcy): its open orders are settled
        while both sides can still be paid. As seller: orders the buyer has not confirmed are refunded (the seller
        will not be there to answer a dispute). As buyer: delivered orders are paid, the others refunded."""
        st = ctx.state
        for oid in st.find_ids("orders", "seller", org):
            if st.tables["orders"][oid]["status"] in ("paid", "delivered", "disputed"):
                if contain(ctx, self._settle_orphan_as, ctx, oid, "seller") is not None:
                    ctx.emit("market.settle_failed", {"id": oid, "error": "wind-up of the seller"}, vis="internal")
        for oid in st.find_ids("orders", "buyer", org):
            if st.tables["orders"][oid]["status"] in ("paid", "delivered", "disputed"):
                if contain(ctx, self._settle_orphan_as, ctx, oid, "buyer") is not None:
                    ctx.emit("market.settle_failed", {"id": oid, "error": "wind-up of the buyer"}, vis="internal")

    def _settle_orphan_as(self, ctx, oid, side: str) -> None:
        st = ctx.state
        o = st.tables["orders"][oid]
        bank = ctx.world.modules["bank"]
        if side == "buyer" and o["status"] == "delivered" and bank.primary_account(st, o["seller"]) is not None:
            self._complete(ctx, oid)                         # the buyer received it: the seller is paid
        elif self._buyer_account(st, o) is not None or not o["amount"]:
            back = self._buyer_account(st, o)
            if o["amount"]:
                bank.move(ctx, ESCROW, back, o["amount"], "refund", f"order {oid} ({side} wound up)", ref=oid)
            if o["kind"] == "goods" and not o.get("shipped"):
                ctx.emit("market.stock", {"owner": o["seller"], "sku": st.tables["listings"][o["listing"]]["sku"],
                                          "delta": o["qty"]}, vis=(o["seller"],))
            ctx.emit("market.order_status", {"id": oid, "status": "refunded", "note": f"{side} wound up"},
                     vis=(o["buyer"], o["seller"]))
        else:
            self._settle_orphan(ctx, oid, f"{side} wound up")

    def mint_goods(self, ctx, owner, sku, qty):
        """Only world generation / production mechanisms create goods (a declared source)."""
        ctx.emit("market.stock", {"owner": owner, "sku": sku, "delta": qty, "minted": True}, vis=(owner,))

    # ------------------------------------------------------------------ appliers
    @applier("market.listing")
    def ap_listing(self, ev):
        p = ev.payload
        self.world.state.insert("listings", p["id"], {"seller": p["seller"], "kind": p["kind"], "title": p["title"],
                                                     "description": p["description"], "price": p["price"],
                                                     "host": p["host"], "endpoint": p["endpoint"], "need": p["need"],
                                                     "repo": p["repo"], "sku": p["sku"], "status": "active",
                                                     "rating_sum": 0, "rating_n": 0, "sales": 0, "created": ev.t})

    @applier("market.listing_update")
    def ap_listing_update(self, ev):
        self.world.state.patch("listings", ev.payload["id"], **ev.payload["changes"])

    @applier("market.order")
    def ap_order(self, ev):
        p = ev.payload
        self.world.state.insert("orders", p["id"], {"listing": p["listing"], "buyer": p["buyer"], "seller": p["seller"],
                                                   "qty": p["qty"], "amount": p["amount"], "account": p["account"],
                                                   "kind": p["kind"], "status": "paid", "t": ev.t, "updated": ev.t})

    @applier("market.order_status")
    def ap_order_status(self, ev):
        st = self.world.state
        o = st.tables["orders"][ev.payload["id"]]
        st.patch("orders", ev.payload["id"], status=ev.payload["status"], updated=ev.t, note=ev.payload.get("note", ""),
                 shipped=o.get("shipped", False) or ev.payload["status"] == "delivered")
        if ev.payload["status"] == "completed":
            l = st.tables["listings"][o["listing"]]
            st.patch("listings", o["listing"], sales=l["sales"] + o["qty"])

    @applier("market.settle_failed")
    def ap_settle_failed(self, ev):
        pass

    @applier("market.rating")
    def ap_rating(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("ratings", p["id"], {"listing": p["listing"], "seller": p["seller"], "buyer": p["buyer"],
                                       "stars": p["stars"], "text": p["text"], "t": ev.t})
        l = st.tables["listings"][p["listing"]]
        st.patch("listings", p["listing"], rating_sum=l["rating_sum"] + p["stars"], rating_n=l["rating_n"] + 1)

    @applier("market.stock")
    def ap_stock(self, ev):
        p = ev.payload
        st = self.world.state
        key = f"{p['owner']}|{p['sku']}"
        cur = st.tables["inventory"].get(key)
        q = (cur["qty"] if cur else 0) + p["delta"]
        st.put("inventory", key, {"owner": p["owner"], "sku": p["sku"], "qty": q})
        if p.get("minted"):
            st.meta_set("goods_minted:" + p["sku"], st.meta_get("goods_minted:" + p["sku"], 0) + p["delta"])

    @applier("market.api_customer")
    def ap_api_customer(self, ev):
        p = ev.payload
        st = self.world.state
        cur = st.tables["api_customers"].get(p["id"])
        st.put("api_customers", p["id"], {"host": p["host"], "payer": p["payer"],
                                          "calls": (cur["calls"] if cur else 0) + 1,
                                          "cents": (cur["cents"] if cur else 0) + p["cents"]})

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        held = sum(o["amount"] for o in st.tables["orders"].values() if o["status"] in ("paid", "delivered", "disputed"))
        bal = st.tables["accounts"][ESCROW]["balance"]
        if held != bal:
            errs.append(f"escrow {bal} != open orders {held}")
        stock: dict[str, int] = {}
        for r in st.tables["inventory"].values():
            if r["qty"] < 0:
                errs.append(f"negative inventory {r}")
            stock[r["sku"]] = stock.get(r["sku"], 0) + r["qty"]
        for o in st.tables["orders"].values():
            if o["kind"] == "goods" and o["status"] in ("paid", "disputed") and not o.get("shipped"):
                sku = st.tables["listings"][o["listing"]]["sku"]
                stock[sku] = stock.get(sku, 0) + o["qty"]
        for sku, q in stock.items():
            minted = st.meta_get("goods_minted:" + sku, 0)
            if q != minted:
                errs.append(f"goods {sku} not conserved: {q} in existence, {minted} minted")
        return errs
