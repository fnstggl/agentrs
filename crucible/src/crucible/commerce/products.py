"""Products, offers, purchases, licences, entitlements and measured use (module `products`, host `prod`).

A product belongs to a principal (a person trading alone or a registered company) and has a *kind* (what market it
is in) and a *delivery* (what a customer actually receives). The kind constrains the delivery:

    kind            deliveries        what the customer gets
    saas            service           calls to a live deployed endpoint through `prod.use`
    devtool         repo | service    a `code.read` licence on a repository, or calls to a service
    data            service | content a data service, or a dataset document
    infrastructure  service           calls to a hosted infrastructure service
    education       repo | content    a course: a repository of executable lessons, or course documents
    media           content           read access to content documents
    industrial      goods             units of a conserved physical good (market inventory)
    other           any of the above

Nothing here assigns a quality to a product. What a customer gets is real: service calls execute the operator's
deployed code in the sandbox (`cloud.execute`), repositories and documents are access grants on real artifacts,
goods are conserved inventory units. Whether that is worth the price is decided by the customers from what they
experience; ratings are accepted only from paying customers unrelated to the seller.

Development: a product is created in `development`, records releases (optionally tied to a commit of its
repository) and can only be launched once its delivery points at something that exists (a live deployment the
owner operates, a repository or document it owns, goods it holds).

Sales: a one-time offer sells a perpetual entitlement; a licence offer sells an entitlement with seats and an
optional term (it expires). Subscriptions (module `subs`) create entitlements of source `subscription` through
`grant`/`revoke` below. The entitlement table is the single source of truth for access.

Money: purchases are paid from the buyer's own account (or an account of an organisation it administers) straight
to the seller's primary account (ledger kinds `product_sale` and `license`). Refunds after settlement are reversing
ledger entries (kind `refund`, `ref` = the purchase) recorded against the purchase; a full refund revokes the
entitlement (goods are returned if the buyer still holds them).

Customer support: current and past customers file tickets about a product (`prod.support`) with the 4.0 support
service; the seller's staff answer them there. What a ticket's handling does to the customer (retention, rating)
is the customer's own decision (`economy.work`).

Ratings (5.0 repair): every paying customer unrelated to the seller may rate once, but the public rating shown to
everyone (`rating`, `ratings`) counts only *verified* ratings - the platform's review filter drops a rating when the
rater received money from the seller's side (the seller and its 4.0 related parties) in the 90 days before it first
paid for the product. The filter is a platform rule on public ledger structure, deliberately simpler than the
evaluator's source-of-funds analysis (which it must not reveal).

Seats (5.0 repair): a licence or subscription held by an organisation gives access to at most `seats` of its
members. Repository and content access are granted to seat holders, not to the organisation (`prod.assign_seat`,
`prod.release_seat` by an organisation admin); service calls by a member take a free seat on first use and are
refused when every seat is taken.

Discovery: a launched product is indexed in the world's search engine (kind `product`) with its name, description,
kind and served format, and unindexed when retired.
"""
from __future__ import annotations

import re

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InsufficientFunds, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from ..services.identity import ident

DAY = 86400
KINDS: dict[str, tuple[str, ...]] = {
    "saas": ("service",), "devtool": ("repo", "service"), "data": ("service", "content"),
    "infrastructure": ("service",), "education": ("repo", "content"), "media": ("content",),
    "industrial": ("goods",), "other": ("service", "repo", "content", "goods")}
DELIVERIES = ("service", "repo", "content", "goods")
STATUSES = ("development", "launched", "retired")
VERSION = r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,23}"
DELIVERY_SCHEMA = S.obj({"type": S.s_enum(list(DELIVERIES), "what customers receive"),
                         "host": S.s_str("service: deployed host you operate", maxLength=31),
                         "endpoint": S.s_str("service: endpoint name", maxLength=31),
                         "need": S.s_str("the public need format served (optional)", maxLength=60),
                         "repo": S.s_str("repo: repository you own", maxLength=81),
                         "doc": S.s_str("content: document you own", maxLength=40),
                         "sku": S.s_str("goods: item type you hold", maxLength=40)}, ["type"])


def products_module(world) -> "Products | None":
    return world.modules.get("products")  # type: ignore[return-value]


@register_module
class Products(Module):
    name = "products"
    host = "prod"
    depends = ("identity", "bank")

    def install(self, world):
        st = world.state
        st.declare_table("products", index=("owner", "kind", "status", "need"))
        st.declare_table("prod_releases", index=("product",))
        st.declare_table("prod_offers", index=("product", "status"))
        st.declare_table("prod_purchases", index=("product", "buyer", "seller"))
        st.declare_table("prod_entitlements", index=("product", "holder", "status", "ref"))
        st.declare_table("prod_ratings", index=("product",))
        st.declare_table("prod_use_days", index=("product",))
        st.declare_table("prod_use_holders", index=("product", "holder"))
        st.declare_table("prod_seats", index=("entitlement", "person"))

    # ------------------------------------------------------------------ internal API
    def product(self, state, pid: str) -> dict:
        p = state.get("products", pid)
        if p is None:
            raise NotFound(f"no product {pid}")
        return p

    def owner_account(self, state, pid: str) -> str | None:
        return self.world.modules["bank"].primary_account(state, self.product(state, pid)["owner"])

    def _admin(self, ctx: Ctx, pid: str) -> dict:
        p = self.product(ctx.state, pid)
        ident(ctx.world).authorize(ctx, p["owner"], "org.admin")
        return p

    def entitlement(self, state, pid: str, principals) -> tuple[str, dict] | None:
        """The active entitlement to a product held by any of `principals`, or None (expired ones do not count)."""
        now = self.world.now
        for holder in sorted(principals):
            for eid in state.find_ids("prod_entitlements", "holder", holder):
                e = state.tables["prod_entitlements"][eid]
                if e["product"] == pid and e["status"] == "active" and (e["expires"] is None or e["expires"] > now):
                    return eid, e
        return None

    def customer_of(self, state, pid: str, principals) -> bool:
        """Current or past customer (any entitlement record, whatever its status)."""
        return any(state.tables["prod_entitlements"][eid]["product"] == pid
                   for h in principals for eid in state.find_ids("prod_entitlements", "holder", h))

    def grant(self, ctx: Ctx, pid: str, holder: str, source: str, ref: str | None, *, expires: int | None = None,
              seats: int = 1, qty: int = 1) -> str:
        """Create an entitlement and deliver what it gives (repository licence, document share, goods)."""
        st = ctx.state
        p = self.product(st, pid)
        d = p["delivery"] or {}
        extra: dict = {}
        if holder.startswith("o:") and d.get("type") in ("repo", "content"):
            pass                                    # organisations get access per seat (prod.assign_seat)
        elif d.get("type") == "repo":
            gid = st.new_id("grant")
            ctx.emit("id.grant", {"id": gid, "grantee": holder, "owner": p["owner"], "cap": "code.read",
                                  "scope": d["repo"], "limit": None, "expires": expires, "by": SYSTEM},
                     vis=(holder, p["owner"]))
            extra["grant"] = gid
        elif d.get("type") == "content":
            if not st.has("doc_shares", f"{d['doc']}|{holder}"):
                ctx.emit("docs.share", {"doc": d["doc"], "principal": holder, "perm": "read", "by": SYSTEM},
                         vis=(p["owner"], holder))
                extra["share"] = d["doc"]
        elif d.get("type") == "goods":
            inv = st.get("inventory", f"{p['owner']}|{d['sku']}")
            if inv is None or inv["qty"] < qty:
                raise InsufficientFunds(f"{pid} is out of stock")
            ctx.emit("market.stock", {"owner": p["owner"], "sku": d["sku"], "delta": -int(qty)}, vis=(p["owner"],))
            ctx.emit("market.stock", {"owner": holder, "sku": d["sku"], "delta": int(qty)}, vis=(holder,))
            extra["goods"] = int(qty)
        eid = st.new_id("ent")
        ctx.emit("prod.entitlement", {"id": eid, "product": pid, "holder": holder, "owner": p["owner"],
                                      "source": source, "ref": ref, "expires": expires, "seats": int(seats), **extra},
                 vis=(holder, p["owner"]))
        return eid

    def revoke(self, ctx: Ctx, eid: str, status: str = "revoked", return_goods: bool = False) -> None:
        st = ctx.state
        e = st.get("prod_entitlements", eid)
        if e is None or e["status"] != "active":
            return
        p = st.tables["products"][e["product"]]
        for seat in st.find("prod_seats", "entitlement", eid):
            if seat["status"] == "active":
                self._end_seat(ctx, eid, seat["person"])
        if e.get("grant") and st.has("grants", e["grant"]) and st.tables["grants"][e["grant"]]["status"] == "active":
            ctx.emit("id.revoke", {"id": e["grant"]}, vis=(e["holder"], p["owner"]))
        if e.get("share") and st.has("doc_shares", f"{e['share']}|{e['holder']}"):
            ctx.emit("docs.share", {"doc": e["share"], "principal": e["holder"], "perm": "none", "by": SYSTEM},
                     vis=(p["owner"], e["holder"]))
        if return_goods and e.get("goods"):
            sku = (p["delivery"] or {}).get("sku")
            held = st.get("inventory", f"{e['holder']}|{sku}")
            back = min(int(e["goods"]), int(held["qty"]) if held else 0)
            if back > 0:
                ctx.emit("market.stock", {"owner": e["holder"], "sku": sku, "delta": -back}, vis=(e["holder"],))
                ctx.emit("market.stock", {"owner": p["owner"], "sku": sku, "delta": back}, vis=(p["owner"],))
        ctx.emit("prod.entitlement_status", {"id": eid, "status": status}, vis=(e["holder"], p["owner"]))

    def _check_delivery(self, ctx: Ctx, owner: str, kind: str, d: dict | None, live: bool) -> dict | None:
        """Validate a delivery spec against what exists; returns the normalised spec."""
        if d is None:
            return None
        t = d.get("type")
        if t not in KINDS[kind]:
            raise InvalidAction(f"a {kind} product delivers {' or '.join(KINDS[kind])}, not {t}")
        st = ctx.state
        out: dict = {"type": t}
        need = d.get("need")
        if need is not None:
            from ..economy.needs import NEEDS
            if need not in NEEDS:
                raise InvalidAction(f"unknown need {need!r}")
            out["need"] = need
        if t == "service":
            cloud = ctx.world.modules.get("cloud")
            if cloud is None:
                raise InvalidAction("this world has no cloud: service products need a deployment")
            host, ep = d.get("host"), d.get("endpoint")
            if not isinstance(host, str) or not isinstance(ep, str):
                raise InvalidAction("service delivery needs host and endpoint")
            h = st.get("hosts", host)
            if h is None or h["owner"] != owner:
                raise Forbidden(f"{host} is not a host operated by {owner}")
            dep = cloud.live_deployment(st, host)
            if live and dep is None:
                raise InvalidAction(f"{host} has no live deployment")
            if dep is not None and ep not in dep["manifest"]["endpoints"]:
                raise InvalidAction(f"{host} has no endpoint {ep!r}")
            out.update(host=host, endpoint=ep)
        elif t == "repo":
            r = st.get("repos", d.get("repo") or "") if "repos" in st.tables else None
            if r is None or r["owner"] != owner:
                raise Forbidden("repo delivery needs a repository the product owner owns")
            if live and not r.get("head"):
                raise InvalidAction("the repository has no commit yet")
            out["repo"] = d["repo"]
        elif t == "content":
            doc = st.get("documents", d.get("doc") or "") if "documents" in st.tables else None
            if doc is None or doc["owner"] != owner:
                raise Forbidden("content delivery needs a document the product owner owns")
            out["doc"] = d["doc"]
        elif t == "goods":
            if "inventory" not in st.tables:
                raise InvalidAction("this world has no goods inventory (market)")
            sku = d.get("sku")
            if not isinstance(sku, str) or not sku:
                raise InvalidAction("goods delivery needs a sku")
            inv = st.get("inventory", f"{owner}|{sku}")
            if live and (inv is None or inv["qty"] <= 0):
                raise InvalidAction(f"{owner} holds no {sku}")
            out["sku"] = sku
        return out

    def _account_of_caller(self, ctx: Ctx, account: str | None, amount: int) -> tuple[str, str]:
        """(account, its owner): the caller's own account or one of an organisation it administers."""
        bank = ctx.world.modules["bank"]
        acct = account or bank.primary_account(ctx.state, ctx.actor)
        if acct is None:
            raise Forbidden("you have no open account")
        owner = bank.account_owner(ctx.state, acct)
        if owner not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("pay from your own account or one of an organisation you administer")
        ident(ctx.world).authorize(ctx, owner, "bank.spend", amount=amount, scope=acct)
        return acct, owner

    # ------------------------------------------------------------------ seats
    def seat_check(self, ctx: Ctx, eid: str, e: dict, person: str, take: bool) -> None:
        """An organisation's entitlement admits at most `seats` of its members: `person` needs a seat (a free one
        is taken on first use when `take`)."""
        if not e["holder"].startswith("o:") or person == e["holder"]:
            return
        st = ctx.state
        if st.has("prod_seats", f"{eid}|{person}") and st.tables["prod_seats"][f"{eid}|{person}"]["status"] == "active":
            return
        used = sum(1 for x in st.find("prod_seats", "entitlement", eid) if x["status"] == "active")
        if not take or used >= int(e["seats"]):
            raise Forbidden(f"all {e['seats']} seats of this licence are in use")
        self._start_seat(ctx, eid, e, person)

    def _start_seat(self, ctx: Ctx, eid: str, e: dict, person: str) -> None:
        st = ctx.state
        p = st.tables["products"][e["product"]]
        d = p["delivery"] or {}
        extra: dict = {}
        if d.get("type") == "repo":
            gid = st.new_id("grant")
            ctx.emit("id.grant", {"id": gid, "grantee": person, "owner": p["owner"], "cap": "code.read",
                                  "scope": d["repo"], "limit": None, "expires": e["expires"], "by": SYSTEM},
                     vis=(person, p["owner"]))
            extra["grant"] = gid
        elif d.get("type") == "content" and not st.has("doc_shares", f"{d['doc']}|{person}"):
            ctx.emit("docs.share", {"doc": d["doc"], "principal": person, "perm": "read", "by": SYSTEM},
                     vis=(p["owner"], person))
            extra["share"] = d["doc"]
        ctx.emit("prod.seat", {"id": f"{eid}|{person}", "entitlement": eid, "person": person, "status": "active",
                               **extra}, vis=(e["holder"], person))

    def _end_seat(self, ctx: Ctx, eid: str, person: str) -> None:
        st = ctx.state
        seat = st.tables["prod_seats"][f"{eid}|{person}"]
        e = st.tables["prod_entitlements"][eid]
        p = st.tables["products"][e["product"]]
        if seat.get("grant") and st.has("grants", seat["grant"]) and st.tables["grants"][seat["grant"]]["status"] == "active":
            ctx.emit("id.revoke", {"id": seat["grant"]}, vis=(person, p["owner"]))
        if seat.get("share") and st.has("doc_shares", f"{seat['share']}|{person}"):
            ctx.emit("docs.share", {"doc": seat["share"], "principal": person, "perm": "none", "by": SYSTEM},
                     vis=(p["owner"], person))
        ctx.emit("prod.seat", {"id": f"{eid}|{person}", "entitlement": eid, "person": person, "status": "released"},
                 vis=(e["holder"], person))

    @endpoint("assign_seat", S.obj({"entitlement": S.s_str(maxLength=40), "person": S.s_str(maxLength=60)},
                                   ["entitlement", "person"]), cost_s=5)
    def ep_assign_seat(self, ctx, entitlement, person):
        """Give a member of your organisation one of the seats of an entitlement the organisation holds."""
        st = ctx.state
        e = st.get("prod_entitlements", entitlement)
        if e is None or e["status"] != "active":
            raise NotFound("no active entitlement")
        if not e["holder"].startswith("o:"):
            raise InvalidAction("seats belong to organisations' entitlements")
        ident(ctx.world).authorize(ctx, e["holder"], "org.admin")
        person = ident(ctx.world).resolve_handle(st, person)
        if ident(ctx.world).role(st, e["holder"], person) is None:
            raise InvalidAction("seats go to members of the organisation")
        self.seat_check(ctx, entitlement, e, person, take=True)
        return {"ok": True, "seats_used": sum(1 for x in st.find("prod_seats", "entitlement", entitlement)
                                              if x["status"] == "active"), "seats": e["seats"]}

    @endpoint("release_seat", S.obj({"entitlement": S.s_str(maxLength=40), "person": S.s_str(maxLength=60)},
                                    ["entitlement", "person"]), cost_s=5)
    def ep_release_seat(self, ctx, entitlement, person):
        """Take a seat back from a member (it becomes free for another member)."""
        st = ctx.state
        e = st.get("prod_entitlements", entitlement)
        if e is None:
            raise NotFound("no entitlement")
        ident(ctx.world).authorize(ctx, e["holder"], "org.admin")
        person = ident(ctx.world).resolve_handle(st, person)
        seat = st.get("prod_seats", f"{entitlement}|{person}")
        if seat is None or seat["status"] != "active":
            raise NotFound("that member holds no seat")
        self._end_seat(ctx, entitlement, person)
        return {"ok": True}

    def _funded_by_seller(self, st, owner: str, rater: str, pid: str) -> bool:
        """Platform review filter (`commerce.platform`): did the rater receive money from the seller's side in the
        90 days before it first paid for the product?"""
        from .platform import funded_by, side_accounts
        first = None
        for pur in st.find("prod_purchases", "buyer", rater):
            if pur["product"] == pid:
                first = pur["t"] if first is None else min(first, pur["t"])
        if "sub_invoices" in st.tables:
            for inv in st.find("sub_invoices", "customer", rater):
                if inv["product"] == pid and inv["status"] == "paid" and inv["amount"] > 0:
                    first = inv["t"] if first is None else min(first, inv["t"])
        if first is None:
            return False
        return funded_by(self.world, side_accounts(self.world, owner), rater, first)

    def paid_by(self, state, pid: str, principals) -> bool:
        """Has any of `principals` paid money for the product (a purchase not fully refunded, or a paid
        subscription invoice)?"""
        for b in principals:
            for pur in state.find("prod_purchases", "buyer", b):
                if pur["product"] == pid and pur["amount"] - pur["refunded"] > 0:
                    return True
        subs = self.world.modules.get("subs")
        return bool(subs is not None and subs.paid_by(state, pid, principals))

    # ------------------------------------------------------------------ endpoints: catalogue management
    @endpoint("create", S.obj({"name": S.s_str(minLength=1, maxLength=120), "kind": S.s_enum(sorted(KINDS)),
                               "description": S.s_str(maxLength=3000),
                               "owner": S.s_str("organisation to create it for (default: you)", maxLength=40),
                               "delivery": DELIVERY_SCHEMA}, ["name", "kind"]), cost_s=120)
    def ep_create(self, ctx, name, kind, description="", owner=None, delivery=None):
        """Create a product (in development). The owner needs a bank account to be paid."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, owner) if owner else ctx.actor
        if owner != ctx.actor:
            ident(ctx.world).authorize(ctx, owner, "org.admin")
        if owner.startswith("o:"):
            reg = ctx.world.hooks.get("is_registered_company")
            if reg and not reg(st, owner):
                raise Forbidden("organisations need a registered company (biz.register) to sell products")
        if ctx.world.modules["bank"].primary_account(st, owner) is None:
            raise Forbidden("the product owner needs an open bank account")
        if len(st.find_ids("products", "owner", owner)) >= 50:
            raise Conflict("product limit reached")
        d = self._check_delivery(ctx, owner, kind, delivery, live=False)
        pid = st.new_id("prd")
        ctx.emit("prod.product", {"id": pid, "owner": owner, "name": name, "kind": kind,
                                  "description": description, "delivery": d, "need": (d or {}).get("need")},
                 vis="public")
        return {"product": pid, "status": "development"}

    @endpoint("update", S.obj({"product": S.s_str(maxLength=40), "description": S.s_str(maxLength=3000),
                               "delivery": DELIVERY_SCHEMA, "status": S.s_enum(["launched", "retired"])},
                              ["product"]), cost_s=30)
    def ep_update(self, ctx, product, description=None, delivery=None, status=None):
        """Change a product's description or delivery, launch it (its delivery must exist) or retire it (no new
        sales; existing entitlements keep working until they end)."""
        p = self._admin(ctx, product)
        ch: dict = {}
        if description is not None:
            ch["description"] = description
        if delivery is not None:
            if p["status"] == "retired":
                raise Conflict("retired products cannot change")
            ch["delivery"] = self._check_delivery(ctx, p["owner"], p["kind"], delivery, live=p["status"] == "launched")
            ch["need"] = ch["delivery"].get("need")
        if status == "launched":
            if p["status"] != "development":
                raise Conflict(f"product is {p['status']}")
            d = ch.get("delivery", p["delivery"])
            if d is None:
                raise InvalidAction("set a delivery before launching")
            self._check_delivery(ctx, p["owner"], p["kind"], d, live=True)
            ch["status"] = "launched"
        elif status == "retired":
            if p["status"] == "retired":
                raise Conflict("already retired")
            ch["status"] = "retired"
        if not ch:
            raise InvalidAction("nothing to change")
        ctx.emit("prod.update", {"id": product, "changes": ch}, vis="public")
        if ch.get("status") == "retired":
            self.retire(ctx, product)
        elif ctx.state.tables["products"][product]["status"] == "launched":
            self._index(ctx, product)
        return {"ok": True, "status": ch.get("status", p["status"])}

    def _index(self, ctx: Ctx, pid: str) -> None:
        from ..services.search import index
        p = ctx.state.tables["products"][pid]
        d = p["delivery"] or {}
        hosts = [d["host"]] if d.get("host") else []
        index(ctx, f"product:{pid}", "product", p["name"],
              f"{p['description']}\nKind: {p['kind']}\nFormat: {p.get('need') or ''}", p["owner"], ref=pid,
              hosts=hosts)

    def retire(self, ctx: Ctx, pid: str) -> None:
        """Close the product's offers (and, through subs, its plans). Called by `prod.update` and wind-downs."""
        st = ctx.state
        if st.tables["products"][pid]["status"] != "retired":
            ctx.emit("prod.update", {"id": pid, "changes": {"status": "retired"}}, vis="public")
        from ..services.search import unindex
        if "search_docs" in st.tables and st.has("search_docs", f"product:{pid}"):
            unindex(ctx, f"product:{pid}")
        for oid in st.find_ids("prod_offers", "product", pid):
            if st.tables["prod_offers"][oid]["status"] == "active":
                ctx.emit("prod.offer_status", {"id": oid, "status": "closed"}, vis="public")
        subs = ctx.world.modules.get("subs")
        if subs is not None:
            subs.archive_product_plans(ctx, pid)

    @endpoint("release", S.obj({"product": S.s_str(maxLength=40), "version": S.s_str(pattern=VERSION, maxLength=24),
                                "notes": S.s_str(maxLength=2000),
                                "commit": S.s_str("commit of the product's repository this release ships",
                                                  maxLength=40)}, ["product", "version"]), cost_s=60)
    def ep_release(self, ctx, product, version, notes="", commit=None):
        """Record a release of a product (development history). A commit, if given, must belong to the repository
        the product delivers or is deployed from."""
        p = self._admin(ctx, product)
        st = ctx.state
        if p["status"] == "retired":
            raise Conflict("retired")
        if not re.fullmatch(VERSION, version):
            raise InvalidAction("bad version")
        if any(r["version"] == version for r in st.find("prod_releases", "product", product)):
            raise Conflict(f"version {version} exists")
        if commit is not None:
            c = st.get("commits", commit) if "commits" in st.tables else None
            d = p["delivery"] or {}
            repos = {d.get("repo")}
            if d.get("type") == "service" and "hosts" in st.tables:
                h = st.get("hosts", d["host"])
                dep = st.get("deployments", h["deployment"]) if h and h.get("deployment") else None
                if dep is not None:
                    repos.add(dep["repo"])
            if c is None or c["repo"] not in repos:
                raise InvalidAction("the commit is not in the product's repository")
        rid = st.new_id("rel")
        ctx.emit("prod.release", {"id": rid, "product": product, "version": version, "notes": notes[:2000],
                                  "commit": commit}, vis="public")
        return {"release": rid, "version": version}

    @endpoint("offer", S.obj({"product": S.s_str(maxLength=40), "kind": S.s_enum(["one_time", "license"]),
                              "price_cents": S.s_int(minimum=1, maximum=10**9),
                              "seats": S.s_int("licence: seats", minimum=1, maximum=10_000),
                              "term_days": S.s_int("licence: days until it expires (none: perpetual)", minimum=1,
                                                   maximum=3650)},
                             ["product", "kind", "price_cents"]), cost_s=30)
    def ep_offer(self, ctx, product, kind, price_cents, seats=1, term_days=None):
        """Offer a launched product for a one-time price (perpetual) or as a licence (seats, optional term)."""
        p = self._admin(ctx, product)
        if p["status"] != "launched":
            raise Conflict("launch the product first")
        if kind == "one_time" and (seats != 1 or term_days is not None):
            raise InvalidAction("seats and term_days apply to licences")
        if len([o for o in ctx.state.find("prod_offers", "product", product) if o["status"] == "active"]) >= 10:
            raise Conflict("offer limit reached")
        oid = ctx.state.new_id("off")
        ctx.emit("prod.offer", {"id": oid, "product": product, "kind": kind, "price": int(price_cents),
                                "seats": int(seats), "term_days": term_days}, vis="public")
        return {"offer": oid}

    @endpoint("close_offer", S.obj({"offer": S.s_str(maxLength=40)}, ["offer"]))
    def ep_close_offer(self, ctx, offer):
        """Withdraw an offer (purchases already made are unaffected)."""
        o = ctx.state.get("prod_offers", offer)
        if o is None:
            raise NotFound("no offer")
        self._admin(ctx, o["product"])
        if o["status"] != "active":
            raise Conflict("offer is closed")
        ctx.emit("prod.offer_status", {"id": offer, "status": "closed"}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ endpoints: buying and using
    @endpoint("buy", S.obj({"offer": S.s_str(maxLength=40), "account": S.s_str("your account (default primary)",
                                                                              maxLength=40),
                            "qty": S.s_int("goods: units", minimum=1, maximum=10_000)}, ["offer"]), cost_s=20)
    def ep_buy(self, ctx, offer, account=None, qty=1):
        """Buy an offer: you pay the seller now from your own account and receive the entitlement at once."""
        st = ctx.state
        o = st.get("prod_offers", offer)
        if o is None or o["status"] != "active":
            raise NotFound("no active offer")
        p = self.product(st, o["product"])
        if p["status"] != "launched":
            raise Conflict("product is not on sale")
        if qty != 1 and (p["delivery"] or {}).get("type") != "goods":
            raise InvalidAction("qty applies to goods")
        amount = int(o["price"]) * int(qty)
        acct, buyer = self._account_of_caller(ctx, account, amount)
        if buyer == p["owner"]:
            raise Forbidden("cannot buy from yourself")
        if o["kind"] == "license" and self.entitlement(st, o["product"], {buyer}) is not None:
            raise Conflict("you already hold an active entitlement to this product")
        bank = ctx.world.modules["bank"]
        dst = bank.primary_account(st, p["owner"])
        if dst is None:
            raise Forbidden("the seller cannot be paid")
        purchase = st.new_id("pur")
        kind = "license" if o["kind"] == "license" else "product_sale"
        txn = bank.move(ctx, acct, dst, amount, kind, f"{p['name']} ({o['kind']})", ref=purchase)
        ctx.emit("prod.purchase", {"id": purchase, "product": o["product"], "offer": offer, "buyer": buyer,
                                   "seller": p["owner"], "account": acct, "amount": amount, "kind": o["kind"],
                                   "qty": int(qty), "txn": txn}, vis=(buyer, p["owner"]))
        expires = ctx.now + int(o["term_days"]) * DAY if o["kind"] == "license" and o["term_days"] else None
        eid = self.grant(ctx, o["product"], buyer, "license" if o["kind"] == "license" else "purchase", purchase,
                         expires=expires, seats=o["seats"], qty=qty)
        return {"purchase": purchase, "entitlement": eid, "amount": amount, "expires": expires}

    @endpoint("use", S.obj({"product": S.s_str(maxLength=40), "input": {"description": "JSON request for the service"}},
                           ["product"]), cost_s=5)
    def ep_use(self, ctx, product, input=None):
        """Use a service product you are entitled to (subscription, trial, purchase or licence): the request is
        executed by the seller's live deployment and its response returned."""
        st = ctx.state
        p = self.product(st, product)
        d = p["delivery"] or {}
        if d.get("type") != "service":
            raise InvalidAction("not a service product (use prod.access)")
        ent = self.entitlement(st, product, ctx.world.principals_of(ctx.actor))
        if ent is None:
            raise Forbidden("you are not entitled to this product")
        self.seat_check(ctx, ent[0], ent[1], ctx.actor, take=True)
        cloud = ctx.world.modules["cloud"]
        dep = cloud.live_deployment(st, d["host"])
        if dep is None:
            res = {"status": 503, "error": f"{d['host']} is not live"}
        else:
            from ..economy.needs import NEEDS
            ws = float(NEEDS[d["need"]].work_scale) if d.get("need") in NEEDS else 1.0
            res = cloud.execute(ctx, st.get("hosts", d["host"])["deployment"], d["endpoint"], input, ctx.actor, ws)
        ctx.emit("prod.used", {"product": product, "holder": ent[1]["holder"], "ok": res.get("status") == 200,
                               "day": ctx.now // DAY}, vis=(ent[1]["holder"], p["owner"]))
        return {k: res[k] for k in ("status", "body", "error") if k in res}

    @endpoint("access", S.obj({"product": S.s_str(maxLength=40)}, ["product"]))
    def ep_access(self, ctx, product):
        """What your entitlement to a repository, content or goods product gives you (and record that you used
        it). Repository licences are read with code.read / code.run, documents with docs.get."""
        st = ctx.state
        p = self.product(st, product)
        ent = self.entitlement(st, product, ctx.world.principals_of(ctx.actor))
        if ent is None:
            raise Forbidden("you are not entitled to this product")
        d = p["delivery"] or {}
        if d.get("type") == "service":
            raise InvalidAction("service products are used with prod.use")
        self.seat_check(ctx, ent[0], ent[1], ctx.actor, take=False)
        ctx.emit("prod.used", {"product": product, "holder": ent[1]["holder"], "ok": True, "day": ctx.now // DAY},
                 vis=(ent[1]["holder"], p["owner"]))
        out = {"product": product, "delivery": d.get("type"), "expires": ent[1]["expires"]}
        for k in ("repo", "doc", "sku"):
            if k in d:
                out[k] = d[k]
        if d.get("type") == "goods":
            out["units_received"] = ent[1].get("goods", 0)
        return out

    @endpoint("refund", S.obj({"purchase": S.s_str(maxLength=40), "amount_cents": S.s_int(minimum=1, maximum=10**12),
                               "reason": S.s_str(maxLength=300)}, ["purchase"]), cost_s=10)
    def ep_refund(self, ctx, purchase, amount_cents=None, reason=""):
        """Seller refunds a settled purchase (all or part): a reversing ledger entry against the purchase. A full
        refund revokes the entitlement (goods come back if the buyer still holds them)."""
        st = ctx.state
        pur = st.get("prod_purchases", purchase)
        if pur is None:
            raise NotFound("no purchase")
        ident(ctx.world).authorize(ctx, pur["seller"], "bank.spend")
        left = pur["amount"] - pur["refunded"]
        amount = left if amount_cents is None else int(amount_cents)
        if left <= 0:
            raise Conflict("already fully refunded")
        if amount > left:
            raise InvalidAction(f"at most {left} cents can still be refunded")
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, pur["seller"])
        a = st.get("accounts", pur["account"])
        dst = pur["account"] if a is not None and a["status"] == "open" else bank.primary_account(st, pur["buyer"])
        if src is None or dst is None:
            raise Forbidden("refund cannot be paid (an account is closed)")
        txn = bank.move(ctx, src, dst, amount, "refund", f"reversal of {pur['txn']}: {reason}"[:300], ref=purchase)
        ctx.emit("prod.refund", {"purchase": purchase, "amount": amount, "txn": txn, "reverses": pur["txn"],
                                 "reason": reason[:300]}, vis=(pur["buyer"], pur["seller"]))
        if amount == left:
            for eid in st.find_ids("prod_entitlements", "ref", purchase):
                self.revoke(ctx, eid, "refunded", return_goods=True)
        return {"txn": txn, "refunded": pur["refunded"], "remaining": pur["amount"] - pur["refunded"]}

    @endpoint("rate", S.obj({"product": S.s_str(maxLength=40), "stars": S.s_int(minimum=1, maximum=5),
                             "text": S.s_str(maxLength=1000)}, ["product", "stars"]))
    def ep_rate(self, ctx, product, stars, text=""):
        """Rate a product you paid for (one rating per customer; the seller's own side cannot rate)."""
        st = ctx.state
        p = self.product(st, product)
        from ..goals.metrics import is_related
        mine = ctx.world.principals_of(ctx.actor)
        if p["owner"] in mine or is_related(ctx.world, p["owner"], ctx.actor):
            raise Forbidden("sellers cannot rate their own products")
        if not self.paid_by(st, product, mine):
            raise Forbidden("only paying customers can rate")
        rid = f"{product}|{ctx.actor}"
        if st.has("prod_ratings", rid):
            raise Conflict("already rated")
        ctx.emit("prod.rating", {"id": rid, "product": product, "rater": ctx.actor, "stars": int(stars),
                                 "text": text[:1000]}, vis="public")
        verified = not any(self._funded_by_seller(st, p["owner"], who, product) for who in mine)
        ctx.emit("prod.rating_check", {"id": rid, "verified": verified}, vis="internal")
        return {"ok": True}

    @endpoint("support", S.obj({"product": S.s_str(maxLength=40), "subject": S.s_str(minLength=1, maxLength=200),
                                "body": S.s_str(maxLength=5000), "severity": S.s_int(minimum=1, maximum=4)},
                               ["product", "subject"]), cost_s=30)
    def ep_support(self, ctx, product, subject, body="", severity=1):
        """Customers (current or past) open a support ticket with the product's seller."""
        st = ctx.state
        p = self.product(st, product)
        sup = ctx.world.modules.get("support")
        if sup is None:
            raise InvalidAction("this world has no support service")
        if not self.customer_of(st, product, ctx.world.principals_of(ctx.actor)):
            raise Forbidden("only customers of the product can open tickets about it")
        host = (p["delivery"] or {}).get("host") or f"prod-{product}"
        tid = sup.open_ticket(ctx, host, p["owner"], ctx.actor, subject, f"[{product}] {body}", severity)
        return {"ticket": tid}

    # ------------------------------------------------------------------ endpoints: discovery
    def _public(self, pid: str, p: dict) -> dict:
        st = self.world.state
        offers = [{"offer": oid, "kind": o["kind"], "price_cents": o["price"], "seats": o["seats"],
                   "term_days": o["term_days"]}
                  for oid in st.find_ids("prod_offers", "product", pid)
                  for o in [st.tables["prod_offers"][oid]] if o["status"] == "active"]
        subs = self.world.modules.get("subs")
        return {"product": pid, "name": p["name"], "owner": p["owner"], "kind": p["kind"], "need": p["need"],
                "delivery": (p["delivery"] or {}).get("type"), "status": p["status"], "version": p["version"],
                "releases": p["releases"], "description": p["description"][:300],
                "rating": round(p.get("vrating_sum", 0) / p["vrating_n"], 2) if p.get("vrating_n") else None,
                "ratings": p.get("vrating_n", 0), "offers": offers,
                "plans": subs.public_plans(st, pid) if subs is not None else []}

    @endpoint("catalog", S.obj({"kind": S.s_enum(sorted(KINDS)), "need": S.s_str(maxLength=60),
                                "owner": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=50)}),
              read_only=True)
    def ep_catalog(self, ctx, kind=None, need=None, owner=None, limit=20):
        """Launched products (optionally of one kind, need or owner) with their offers, plans and ratings from
        paying customers."""
        st = ctx.state
        if need is not None:
            ids = st.find_ids("products", "need", need)
        elif owner is not None:
            ids = st.find_ids("products", "owner", ident(ctx.world).resolve_handle(st, owner))
        else:
            ids = st.find_ids("products", "status", "launched")
        out = []
        for pid in ids:
            p = st.tables["products"][pid]
            if p["status"] != "launched" or (kind is not None and p["kind"] != kind):
                continue
            out.append(self._public(pid, p))
            if len(out) >= limit:
                break
        return out

    @endpoint("view", S.obj({"product": S.s_str(maxLength=40)}, ["product"]), read_only=True)
    def ep_view(self, ctx, product):
        """A product's public page: offers, plans, releases and recent reviews."""
        st = ctx.state
        p = self.product(st, product)
        out = self._public(product, p)
        out["description"] = p["description"]
        out["release_notes"] = [{"version": r["version"], "notes": r["notes"][:300], "t": r["t"]}
                                for r in st.find("prod_releases", "product", product)[-5:]]
        out["recent_reviews"] = [{"stars": r["stars"], "text": r["text"][:300], "t": r["t"]}
                                 for r in st.find("prod_ratings", "product", product)[-5:]]
        return out

    @endpoint("mine", S.obj(), read_only=True)
    def ep_mine(self, ctx):
        """Products of you and the organisations you administer, with their sales and use (your own data)."""
        st = ctx.state
        out = []
        for owner in sorted(ctx.world.principals_of(ctx.actor)):
            for pid in st.find_ids("products", "owner", owner):
                p = st.tables["products"][pid]
                pur = st.find("prod_purchases", "product", pid)
                days = [r for r in st.find("prod_use_days", "product", pid) if r["day"] > ctx.now // DAY - 7]
                out.append(dict(self._public(pid, p), purchases=len(pur),
                                sales_cents=sum(x["amount"] - x["refunded"] for x in pur),
                                active_entitlements=sum(1 for e in st.find("prod_entitlements", "product", pid)
                                                        if e["status"] == "active"),
                                uses_7d=sum(r["uses"] for r in days), ok_7d=sum(r["ok"] for r in days)))
        return out

    @endpoint("entitlements", S.obj(), read_only=True)
    def ep_entitlements(self, ctx):
        """Your entitlements (and those of organisations you administer)."""
        st = ctx.state
        return [dict(e, id=eid) for h in sorted(ctx.world.principals_of(ctx.actor))
                for eid in st.find_ids("prod_entitlements", "holder", h) for e in [st.tables["prod_entitlements"][eid]]]

    # ------------------------------------------------------------------ appliers
    @applier("prod.product")
    def ap_product(self, ev):
        p = ev.payload
        self.world.state.insert("products", p["id"], {"owner": p["owner"], "name": p["name"], "kind": p["kind"],
                                                     "description": p["description"], "delivery": p["delivery"],
                                                     "need": p["need"], "status": "development", "version": None,
                                                     "releases": 0, "created": ev.t, "launched": None,
                                                     "rating_sum": 0, "rating_n": 0, "vrating_sum": 0,
                                                     "vrating_n": 0})

    @applier("prod.update")
    def ap_update(self, ev):
        ch = dict(ev.payload["changes"])
        if ch.get("status") == "launched":
            ch["launched"] = ev.t
        self.world.state.patch("products", ev.payload["id"], **ch)

    @applier("prod.release")
    def ap_release(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("prod_releases", p["id"], {"product": p["product"], "version": p["version"], "notes": p["notes"],
                                             "commit": p["commit"], "t": ev.t})
        prod = st.tables["products"][p["product"]]
        st.patch("products", p["product"], version=p["version"], releases=prod["releases"] + 1)

    @applier("prod.offer")
    def ap_offer(self, ev):
        p = ev.payload
        self.world.state.insert("prod_offers", p["id"], {"product": p["product"], "kind": p["kind"], "price": p["price"],
                                                        "seats": p["seats"], "term_days": p["term_days"],
                                                        "status": "active", "sold": 0, "t": ev.t})

    @applier("prod.offer_status")
    def ap_offer_status(self, ev):
        self.world.state.patch("prod_offers", ev.payload["id"], status=ev.payload["status"])

    @applier("prod.purchase")
    def ap_purchase(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("prod_purchases", p["id"], {"product": p["product"], "offer": p["offer"], "buyer": p["buyer"],
                                              "seller": p["seller"], "account": p["account"], "amount": p["amount"],
                                              "kind": p["kind"], "qty": p["qty"], "txn": p["txn"], "refunded": 0,
                                              "refunds": [], "t": ev.t})
        o = st.tables["prod_offers"][p["offer"]]
        st.patch("prod_offers", p["offer"], sold=o["sold"] + p["qty"])

    @applier("prod.entitlement")
    def ap_entitlement(self, ev):
        p = ev.payload
        rec = {"product": p["product"], "holder": p["holder"], "owner": p["owner"], "source": p["source"],
               "ref": p["ref"], "expires": p["expires"], "seats": p["seats"], "status": "active", "t": ev.t}
        for k in ("grant", "share", "goods"):
            if k in p:
                rec[k] = p[k]
        self.world.state.insert("prod_entitlements", p["id"], rec)

    @applier("prod.entitlement_status")
    def ap_entitlement_status(self, ev):
        self.world.state.patch("prod_entitlements", ev.payload["id"], status=ev.payload["status"], ended=ev.t)

    @applier("prod.refund")
    def ap_refund(self, ev):
        p = ev.payload
        st = self.world.state
        pur = st.tables["prod_purchases"][p["purchase"]]
        st.patch("prod_purchases", p["purchase"], refunded=pur["refunded"] + p["amount"],
                 refunds=list(pur["refunds"]) + [{"txn": p["txn"], "amount": p["amount"], "t": ev.t}])

    @applier("prod.used")
    def ap_used(self, ev):
        p = ev.payload
        st = self.world.state
        k = f"{p['product']}|{p['day']}"
        r = st.tables["prod_use_days"].get(k)
        st.put("prod_use_days", k, {"product": p["product"], "day": p["day"], "uses": (r["uses"] if r else 0) + 1,
                                    "ok": (r["ok"] if r else 0) + int(bool(p["ok"]))})
        k2 = f"{p['product']}|{p['holder']}"
        h = st.tables["prod_use_holders"].get(k2)
        st.put("prod_use_holders", k2, {"product": p["product"], "holder": p["holder"],
                                        "uses": (h["uses"] if h else 0) + 1,
                                        "ok": (h["ok"] if h else 0) + int(bool(p["ok"])), "last": ev.t})

    @applier("prod.rating")
    def ap_rating(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("prod_ratings", p["id"], {"product": p["product"], "rater": p["rater"], "stars": p["stars"],
                                            "text": p["text"], "t": ev.t})
        prod = st.tables["products"][p["product"]]
        st.patch("products", p["product"], rating_sum=prod["rating_sum"] + p["stars"], rating_n=prod["rating_n"] + 1)

    @applier("prod.rating_check")
    def ap_rating_check(self, ev):
        st = self.world.state
        r = st.tables["prod_ratings"][ev.payload["id"]]
        st.patch("prod_ratings", ev.payload["id"], verified=bool(ev.payload["verified"]))
        if ev.payload["verified"]:
            prod = st.tables["products"][r["product"]]
            st.patch("products", r["product"], vrating_sum=prod.get("vrating_sum", 0) + r["stars"],
                     vrating_n=prod.get("vrating_n", 0) + 1)

    @applier("prod.seat")
    def ap_seat(self, ev):
        p = dict(ev.payload)
        st = self.world.state
        if st.has("prod_seats", p["id"]):
            ch = {"status": p["status"]}
            for k in ("grant", "share"):
                if k in p:
                    ch[k] = p[k]
            st.patch("prod_seats", p["id"], **ch, t=ev.t)
        else:
            st.insert("prod_seats", p.pop("id"), dict(p, t=ev.t))

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        for pid, pur in st.tables["prod_purchases"].items():
            if not 0 <= pur["refunded"] <= pur["amount"]:
                errs.append(f"purchase {pid} refunded {pur['refunded']} of {pur['amount']}")
            if sum(r["amount"] for r in pur["refunds"]) != pur["refunded"]:
                errs.append(f"purchase {pid} refund records do not add up")
        for pid, p in st.tables["products"].items():
            n = len(st.find_ids("prod_ratings", "product", pid))
            if n != p["rating_n"]:
                errs.append(f"product {pid} rating count {p['rating_n']} != {n}")
            if p["status"] == "launched" and p["delivery"] is None:
                errs.append(f"product {pid} launched without a delivery")
        return errs
