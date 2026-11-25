"""Generic ownable assets (kernel module `assets`, host `asset`; shared kind `asset`).

An asset is a non-money thing of value that exactly one principal owns at a time and that can change hands: a licence,
a right, a digital good (optionally backed by an artifact REF), a title. Asset classes that 4.0 modules already own
(company equity `business.shares`, physical capital `industry.machines`/`facilities`, design rights
`industry.design_rights`, web domains `net.domains`, money and accounts in `bank`) are refused here, so no fact is
kept twice.

Ownership changes only by recorded events: a gift (`asset.transfer`) or a sale (`asset.offer` to one buyer, then
`asset.accept` by that buyer). A sale is one transaction across modules: the buyer's payment through `bank` and the
change of owner are applied together or not at all (a buyer who cannot pay leaves the asset where it was).
"""
from __future__ import annotations

import re

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from . import artifacts, bus

CLASS_RE = r"[a-z][a-z0-9_-]{1,31}"
RESERVED = {"equity": "business.shares", "share": "business.shares", "shares": "business.shares",
            "machine": "industry.machines", "facility": "industry.facilities", "design": "industry.design_rights",
            "design_right": "industry.design_rights", "domain": "net.domains", "money": "bank.txns",
            "account": "bank.accounts", "cash": "bank.txns", "currency": "bank.txns"}

_A = {"type": "string", "pattern": r"ast_[0-9]{6,}"}
_P = {"type": "string", "minLength": 1, "maxLength": 120}
bus.event_type("asset.registered", S.obj({"id": _A, "class": {"type": "string", "pattern": CLASS_RE}, "owner": _P,
                                          "title": {"type": "string", "maxLength": 200},
                                          "ref": {"type": ["string", "null"]},
                                          "meta": {"type": "object", "additionalProperties": True}},
                                         ["id", "class", "owner", "title", "ref", "meta"]), owner="assets",
               doc="a new asset and its first owner")
bus.event_type("asset.transferred", S.obj({"id": _A, "from": _P, "to": _P, "offer": {"type": ["string", "null"]},
                                           "price_cents": {"type": "integer", "minimum": 0},
                                           "txn": {"type": ["string", "null"]}},
                                          ["id", "from", "to", "offer", "price_cents", "txn"]), owner="assets",
               doc="ownership changed (gift or sale; a sale carries its bank transaction)")
bus.event_type("asset.offered", S.obj({"offer": {"type": "string"}, "id": _A, "from": _P, "to": _P,
                                       "price_cents": {"type": "integer", "minimum": 0}},
                                      ["offer", "id", "from", "to", "price_cents"]), owner="assets",
               doc="the owner offered an asset to one buyer at a price")
bus.event_type("asset.offer_closed", S.obj({"offer": {"type": "string"}, "status": {"type": "string",
                                                                                      "enum": ["accepted", "withdrawn",
                                                                                               "lapsed"]}},
                                           ["offer", "status"]), owner="assets", doc="an offer ended")


@register_module
class Assets(Module):
    name = "assets"
    host = "asset"
    version = "1"
    depends = ("identity",)
    shared_facts = {"asset": "assets"}

    def install(self, world):
        st = world.state
        st.declare_table("assets", index=("owner", "class"))
        st.declare_table("asset_offers", index=("asset", "to", "status"))
        bus.enforce(world, owner="assets")

    # ------------------------------------------------------------------ internal API
    def register(self, ctx: Ctx, owner: str, cls: str, title: str, ref: str | None = None,
                 meta: dict | None = None) -> str:
        if not isinstance(cls, str) or not re.fullmatch(CLASS_RE, cls):
            raise InvalidAction(f"bad asset class {cls!r}")
        if cls in RESERVED:
            raise InvalidAction(f"asset class {cls!r} is owned by {RESERVED[cls]}; use that module")
        if not ctx.state.has("principals", owner):
            raise NotFound(f"no principal {owner}")
        if ref is not None:
            if "artifacts" not in ctx.world.modules or not artifacts.readable(ctx.world, owner, ref):
                raise NotFound(f"no artifact {ref}")
        aid = ctx.state.new_id("ast")
        ctx.emit("asset.registered", {"id": aid, "class": cls, "owner": owner, "title": str(title)[:200], "ref": ref,
                                      "meta": dict(meta or {})}, vis=(owner,))
        return aid

    def _asset(self, ctx: Ctx, aid: str):
        a = ctx.state.get("assets", aid)
        if a is None or not (a["owner"] in ctx.world.principals_of(ctx.actor) or ctx.actor == SYSTEM
                             or any(o["to"] in ctx.world.principals_of(ctx.actor)
                                    for o in ctx.state.find("asset_offers", "asset", aid) if o["status"] == "open")):
            raise NotFound(f"no asset {aid}")
        return a

    def _close_open_offers(self, ctx: Ctx, aid: str, status: str, keep: str | None = None) -> None:
        for oid in ctx.state.find_ids("asset_offers", "asset", aid):
            if oid != keep and ctx.state.tables["asset_offers"][oid]["status"] == "open":
                ctx.emit("asset.offer_closed", {"offer": oid, "status": status}, vis="internal")

    def transfer(self, ctx: Ctx, aid: str, to: str, offer: str | None = None, price: int = 0,
                 txn: str | None = None) -> None:
        a = ctx.state.tables["assets"][aid]
        if not ctx.state.has("principals", to):
            raise NotFound(f"no principal {to}")
        if to == a["owner"]:
            raise InvalidAction("the asset already belongs to them")
        self._close_open_offers(ctx, aid, "lapsed", keep=offer)
        ctx.emit("asset.transferred", {"id": aid, "from": a["owner"], "to": to, "offer": offer, "price_cents": price,
                                       "txn": txn}, vis=tuple(sorted({a["owner"], to})))

    # ------------------------------------------------------------------ appliers
    @applier("asset.registered")
    def ap_registered(self, ev):
        p = ev.payload
        self.world.state.insert("assets", p["id"], {"class": p["class"], "owner": p["owner"], "title": p["title"],
                                                    "ref": p["ref"], "meta": p["meta"], "created": ev.t,
                                                    "transfers": 0})

    @applier("asset.transferred")
    def ap_transferred(self, ev):
        p = ev.payload
        a = self.world.state.tables["assets"][p["id"]]
        self.world.state.patch("assets", p["id"], owner=p["to"], transfers=a["transfers"] + 1)

    @applier("asset.offered")
    def ap_offered(self, ev):
        p = ev.payload
        self.world.state.insert("asset_offers", p["offer"], {"asset": p["id"], "from": p["from"], "to": p["to"],
                                                             "price_cents": p["price_cents"], "status": "open",
                                                             "created": ev.t})

    @applier("asset.offer_closed")
    def ap_offer_closed(self, ev):
        self.world.state.patch("asset_offers", ev.payload["offer"], status=ev.payload["status"])

    # ------------------------------------------------------------------ tools
    @endpoint("register", S.obj({"class": S.s_str("what kind of asset: licence, right, title, digital-good, ...",
                                                  pattern=CLASS_RE, maxLength=32),
                                 "title": S.s_str(maxLength=200),
                                 "ref": S.s_str("artifact REF backing it (optional)", maxLength=71),
                                 "meta": S.s_any_obj("small JSON description"),
                                 "owner": S.s_str("organisation you administer (default: you)", maxLength=60)},
                                ["class", "title"]), cost_s=5)
    def ep_register(self, ctx, title, ref=None, meta=None, owner=None, **kw):
        """Register a new asset that you (or an organisation you administer) own."""
        cls = kw["class"]
        owner = owner or ctx.actor
        if owner not in ctx.world.principals_of(ctx.actor):
            raise Forbidden(f"you cannot register assets for {owner}")
        if meta is not None and len(str(meta)) > 2000:
            raise InvalidAction("meta too large")
        return {"asset": self.register(ctx, owner, cls, title, ref, meta)}

    @endpoint("transfer", S.obj({"asset": S.s_str(maxLength=40), "to": S.s_str("new owner", maxLength=60)},
                                ["asset", "to"]), cost_s=5)
    def ep_transfer(self, ctx, asset, to):
        """Give an asset you own to another principal (no payment)."""
        a = self._asset(ctx, asset)
        if a["owner"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("only the owner can transfer an asset")
        self.transfer(ctx, asset, to)
        return {"asset": asset, "owner": to}

    @endpoint("offer", S.obj({"asset": S.s_str(maxLength=40), "to": S.s_str("the one buyer", maxLength=60),
                             "price_cents": S.s_int(minimum=0, maximum=10**12)}, ["asset", "to", "price_cents"]),
              cost_s=5)
    def ep_offer(self, ctx, asset, to, price_cents):
        """Offer an asset you own to one buyer at a price; the sale happens when the buyer accepts and pays."""
        a = self._asset(ctx, asset)
        if a["owner"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("only the owner can offer an asset")
        if not ctx.state.has("principals", to) or to == a["owner"]:
            raise InvalidAction("the buyer must be another principal")
        if price_cents and "bank" not in ctx.world.modules:
            raise InvalidAction("priced offers need the bank")
        if any(o["status"] == "open" for o in ctx.state.find("asset_offers", "asset", asset)):
            raise Conflict("this asset already has an open offer (withdraw it first)")
        oid = ctx.state.new_id("aoff")
        ctx.emit("asset.offered", {"offer": oid, "id": asset, "from": a["owner"], "to": to,
                                   "price_cents": int(price_cents)}, vis=tuple(sorted({a["owner"], to})))
        return {"offer": oid}

    @endpoint("accept", S.obj({"offer": S.s_str(maxLength=40)}, ["offer"]), cost_s=5)
    def ep_accept(self, ctx, offer):
        """Accept an offer made to you: your payment and the change of owner happen together (or not at all)."""
        o = ctx.state.get("asset_offers", offer)
        if o is None or o["to"] not in ctx.world.principals_of(ctx.actor):
            raise NotFound(f"no offer {offer}")
        if o["status"] != "open":
            raise Conflict(f"offer {offer} is {o['status']}")
        a = ctx.state.tables["assets"][o["asset"]]
        if a["owner"] != o["from"]:
            raise Conflict("the asset changed hands since the offer")
        txn = None
        if o["price_cents"]:
            bank = ctx.world.modules["bank"]
            src = bank.primary_account(ctx.state, o["to"])
            dst = bank.primary_account(ctx.state, o["from"])
            if src is None or dst is None:
                raise NotFound("buyer and seller need open accounts")
            txn = bank.move(ctx, src, dst, int(o["price_cents"]), "asset", f"asset {o['asset']}", ref=o["asset"])
        ctx.emit("asset.offer_closed", {"offer": offer, "status": "accepted"}, vis=tuple(sorted({o["from"], o["to"]})))
        self.transfer(ctx, o["asset"], o["to"], offer=offer, price=int(o["price_cents"]), txn=txn)
        return {"asset": o["asset"], "owner": o["to"], "txn": txn}

    @endpoint("withdraw", S.obj({"offer": S.s_str(maxLength=40)}, ["offer"]), cost_s=1)
    def ep_withdraw(self, ctx, offer):
        """Withdraw an open offer you made."""
        o = ctx.state.get("asset_offers", offer)
        if o is None or o["from"] not in ctx.world.principals_of(ctx.actor):
            raise NotFound(f"no offer {offer}")
        if o["status"] != "open":
            raise Conflict(f"offer {offer} is {o['status']}")
        ctx.emit("asset.offer_closed", {"offer": offer, "status": "withdrawn"}, vis=tuple(sorted({o["from"], o["to"]})))
        return {"offer": offer, "status": "withdrawn"}

    @endpoint("mine", S.obj({"owner": S.s_str(maxLength=60)}), read_only=True, cost_s=0)
    def ep_mine(self, ctx, owner=None):
        """Assets you (or an organisation you administer) own, and open offers made to you."""
        owner = owner or ctx.actor
        if owner not in ctx.world.principals_of(ctx.actor):
            raise Forbidden(f"you cannot list the assets of {owner}")
        st = ctx.state
        rows = [dict(st.tables["assets"][i], id=i) for i in st.find_ids("assets", "owner", owner)]
        offers = [dict(st.tables["asset_offers"][i], id=i) for i in st.find_ids("asset_offers", "to", owner)
                  if st.tables["asset_offers"][i]["status"] == "open"]
        return {"owner": owner, "assets": rows, "offers_to_you": offers}

    @endpoint("info", S.obj({"asset": S.s_str(maxLength=40)}, ["asset"]), read_only=True, cost_s=0)
    def ep_info(self, ctx, asset):
        """An asset you own or have an open offer for."""
        a = self._asset(ctx, asset)
        return dict(a, id=asset)

    def check_invariants(self, world):
        errs = []
        st = world.state
        for aid, a in st.tables["assets"].items():
            if not st.has("principals", a["owner"]):
                errs.append(f"asset {aid} owned by unknown principal {a['owner']}")
            if a["class"] in RESERVED:
                errs.append(f"asset {aid} has reserved class {a['class']}")
        open_by_asset: dict[str, int] = {}
        for oid, o in st.tables["asset_offers"].items():
            if o["status"] != "open":
                continue
            open_by_asset[o["asset"]] = open_by_asset.get(o["asset"], 0) + 1
            a = st.tables["assets"].get(o["asset"])
            if a is None or a["owner"] != o["from"]:
                errs.append(f"open offer {oid} is not from the asset's owner")
        errs += [f"asset {a} has {n} open offers" for a, n in open_by_asset.items() if n > 1]
        return errs
