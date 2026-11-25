"""Product discovery (module `launch`, host `launch`): launch posts that point at real products.

A launch is a catalogue item (a post, an app demo, a short video or an image) on platform `launch` that links to
something a participant can actually visit and buy through ordinary mechanisms:

    listing   a 4.0 market listing (market.view / market.buy / market.rate)
    product   a product of the commerce workstream (prod.view / prod.buy / subs.subscribe / prod.rate)
    host      a deployed service or web site (world.describe / net.fetch / its own endpoints)

Anyone may post a launch (the maker or a "hunter"); the launch records whether the poster is on the seller side.
Participants upvote launches (one vote each, never their own), browse the day's launches ranked by those votes,
comment, and `launch.visit` a launch to get the product's public page (and learn its host). Whatever happens next -
a trial, a purchase, a review, infrastructure load - happens in the market, commerce and cloud modules exactly as
if the participant had found the product any other way. Nothing here pays, credits or ranks a seller by sales:
there is no social-to-revenue path except participants' own decisions.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, applier, endpoint, register_module
from ..substrate import bus
from . import content as C
from .platforms import PLATFORMS
from .sites import Site, _site_endpoints

DAY = 86400
_ID = S.s_str("content id c<n>", pattern=C.CID_RE.pattern, maxLength=14)
_S = {"type": "string"}
bus.event_type("launch.product", S.obj({"id": _S, "link": {"type": "string", "enum": ["listing", "product", "host"]},
                                        "target": _S, "maker": {"type": "boolean"}, "by": _S},
                                       ["id", "link", "target", "maker", "by"]), owner="launch",
               doc="a launch post links to a product a participant can visit and buy")
bus.event_type("launch.vote", S.obj({"id": _S, "voter": _S, "up": {"type": "boolean"}}, ["id", "voter", "up"]),
               owner="launch", doc="an upvote (or its withdrawal) on a launch")


class LaunchSite(Site):
    def install(self, world):
        super().install(world)
        st = world.state
        st.declare_table("launches", index=("link_key",))
        st.declare_table("launch_votes", index=("item", "voter"))
        bus.enforce(world, owner="launch")

    def _target(self, ctx: Ctx, listing, product, host) -> tuple[str, str, str]:
        """(link kind, target id, seller principal) of a launch's product - it must exist and be on offer."""
        given = [(k, v) for k, v in (("listing", listing), ("product", product), ("host", host)) if v]
        if len(given) != 1:
            raise InvalidAction("link exactly one of listing, product or host")
        kind, tgt = given[0]
        st = ctx.state
        if kind == "listing":
            if "listings" not in st.tables:
                raise NotFound("this world has no market")
            l = st.get("listings", tgt)
            if l is None or l["status"] != "active":
                raise NotFound(f"no active listing {tgt}")
            return kind, tgt, l["seller"]
        if kind == "product":
            if "products" not in st.tables:
                raise NotFound("this world has no product catalogue")
            p = st.get("products", tgt)
            if p is None or p.get("status") != "launched":
                raise NotFound(f"no launched product {tgt}")
            return kind, tgt, p["owner"]
        from ..services.search import is_host
        if not is_host(ctx.world, tgt):
            raise NotFound(f"no service at {tgt}.crucible")
        owner_fn = ctx.world.hooks.get("host_owner")
        return kind, tgt, (owner_fn(st, tgt) if owner_fn else None) or "sys"

    @endpoint("post", S.obj({"ref": S.s_str("artifact you own: a text/html post, an app demo, a short video or an "
                                            "image", maxLength=71),
                             "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE),
                             "kind": S.s_enum(list(PLATFORMS["launch"]["kinds"])),
                             "listing": S.s_str("a market listing id", maxLength=40),
                             "product": S.s_str("a product id of the product catalogue", maxLength=40),
                             "host": S.s_str("a deployed service host", maxLength=41),
                             "tags": C._TAGS, "description": S.s_str(maxLength=C.MAX_DESC)},
                            ["ref", "title"]), cost_s=60)
    def ep_post(self, ctx, ref, title, kind="post", listing=None, product=None, host=None, tags=(), description=""):
        """Launch a product: a post that links to a real listing, product or service."""
        link, tgt, seller = self._target(ctx, listing, product, host)
        cid = C.publish(ctx, ctx.actor, ref, kind, "launch", title, tags=tags, description=description)
        maker = seller in ctx.world.principals_of(ctx.actor) or seller == ctx.actor
        bus.emit(ctx, "launch.product", {"id": cid, "link": link, "target": tgt, "maker": bool(maker),
                                         "by": ctx.actor}, vis="public")
        return {"id": cid, "link": link, "target": tgt, "maker": bool(maker)}

    @endpoint("upvote", S.obj({"id": _ID, "up": S.s_bool("false withdraws your vote", default=True)}, ["id"]),
              cost_s=2)
    def ep_upvote(self, ctx, id, up=True):
        """Upvote a launch (one vote per person; not your own)."""
        rec = self._mine(ctx, id)
        L = ctx.state.get("launches", id)
        if L is None:
            raise NotFound(f"{id} is not a launch")
        if rec["creator"] in ctx.world.principals_of(ctx.actor):
            raise Forbidden("you cannot upvote your own launch")
        has = ctx.state.has("launch_votes", f"{id}|{ctx.actor}")
        if has == bool(up):
            raise Conflict("nothing changes")
        bus.emit(ctx, "launch.vote", {"id": id, "voter": ctx.actor, "up": bool(up)}, vis=(ctx.actor,))
        return {"id": id, "upvotes": ctx.state.tables["launches"][id]["upvotes"]}

    def _rows(self, ctx: Ctx, since: int, until: int):
        st = ctx.state
        out = []
        for cid, L in st.scan("launches"):
            rec = st.tables["content"][cid]
            if rec["removed"] or not since <= rec["published_at"] < until:
                continue
            out.append(dict(C.module(ctx.world).view(ctx.world, rec, ctx.actor), link=L["link"], target=L["target"],
                            maker=L["maker"], upvotes=L["upvotes"]))
        out.sort(key=lambda r: (-r["upvotes"], -r["n"]))
        return out

    @endpoint("today", S.obj({"days_ago": S.s_int(minimum=0, maximum=365, default=0),
                              "limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True, cost_s=2)
    def ep_today(self, ctx, days_ago=0, limit=20):
        """The launches of a day, ranked by upvotes."""
        day0 = (ctx.now // DAY - int(days_ago)) * DAY
        return {"day": day0 // DAY, "launches": self._rows(ctx, day0, day0 + DAY)[:limit]}

    @endpoint("top", S.obj({"days": S.s_int(minimum=1, maximum=365, default=7),
                            "limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True, cost_s=2)
    def ep_top(self, ctx, days=7, limit=20):
        """The most upvoted launches of the last days."""
        return {"days": days, "launches": self._rows(ctx, ctx.now - int(days) * DAY, ctx.now + 1)[:limit]}

    @endpoint("visit", S.obj({"id": _ID}, ["id"]), read_only=True, cost_s=5)
    def ep_visit(self, ctx, id):
        """Open a launch's product page: the listing, product or service it links to (you learn its host). Buying,
        trying or reviewing it is done with the market, product and subscription tools themselves."""
        rec = self._mine(ctx, id)
        L = ctx.state.get("launches", id)
        if L is None:
            raise NotFound(f"{id} is not a launch")
        out = {"launch": id, "title": rec["title"], "link": L["link"], "target": L["target"]}
        if L["link"] == "listing":
            out["page"] = ctx.call("market.view", {"listing": L["target"]})
            out["next"] = {"buy": "market.buy", "rate": "market.rate"}
        elif L["link"] == "product":
            out["page"] = ctx.call("prod.view", {"product": L["target"]})
            out["next"] = {"plans": "subs.plans", "subscribe": "subs.subscribe", "buy": "prod.buy",
                           "rate": "prod.rate"}
        else:
            out["page"] = ctx.call("world.describe", {"host": L["target"]})
            out["next"] = {"call": f"{L['target']}.<endpoint>", "browse": f"http://{L['target']}.crucible/"}
        return out

    @applier("launch.product")
    def ap_product(self, ev):
        p = ev.payload
        self.world.state.insert("launches", p["id"], {"link": p["link"], "target": p["target"],
                                                      "link_key": f"{p['link']}:{p['target']}", "maker": p["maker"],
                                                      "by": p["by"], "upvotes": 0, "t": ev.t})

    @applier("launch.vote")
    def ap_vote(self, ev):
        p = ev.payload
        st = self.world.state
        L = st.tables["launches"][p["id"]]
        key = f"{p['id']}|{p['voter']}"
        if p["up"]:
            st.insert("launch_votes", key, {"item": p["id"], "voter": p["voter"], "t": ev.t})
            st.patch("launches", p["id"], upvotes=L["upvotes"] + 1)
        else:
            st.delete("launch_votes", key)
            st.patch("launches", p["id"], upvotes=L["upvotes"] - 1)

    def check_invariants(self, world):
        st = world.state
        errs = []
        cnt: dict[str, int] = {}
        for k, v in st.tables["launch_votes"].items():
            cnt[v["item"]] = cnt.get(v["item"], 0) + 1
        for cid, L in st.tables["launches"].items():
            if L["upvotes"] != cnt.get(cid, 0):
                errs.append(f"launch {cid} counts {L['upvotes']} upvotes, {cnt.get(cid, 0)} recorded")
        return errs


Launch = register_module(_site_endpoints(type("Site_launch", (LaunchSite,), {
    "name": "launch", "host": "launch", "platform": "launch",
    "__doc__": f"{PLATFORMS['launch']['label']}: {PLATFORMS['launch']['doc']}."})))
