"""Content platforms as kernel modules over the shared catalogue: short-form video (`shorts`), long-form video
(`tube`), text posts (`posts`), images (`pics`), articles and blogs (`blog`), audio and podcasts (`pods`).

Each platform is a site with its own host and tools, carrying the catalogue kinds of its registry entry
(crucible.social.platforms). What differs between platforms is real: what may be published there and how a
participant discovers things there. All of them share the population, the catalogue, the follow graph and the
identity space.

Tools of every site `<host>`:

    <host>.publish   publish an artifact you own on this platform (content.publish with the platform fixed)
    <host>.latest    newest items (optionally one tag or kind)
    <host>.channel   one creator's items here
    <host>.feed      your feed. For a population individual in a world with workstream C2's distribution module
                     (`feed`), the platform's own recommender mixture (`feed.home`: the impressions are recorded by
                     the audience module); otherwise the chronological following feed - newest items of the accounts
                     you follow - then the newest items of everyone else
    <host>.trending  items of the last hours ranked by recorded comments and public shares in that window
    <host>.view      an item of this platform with its comments
    <host>.comment   comment on an item of this platform

Consumption (reading, watching, listening, liking, sharing with contacts) is recorded by the audience module
only - population individuals use `feed.watch` (workstream C2) on items they were shown; no tool here changes an
audience figure.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import NotFound
from ..core.kernel import Ctx, Module, register_module
from ..substrate import ids
from . import content as C
from . import graph as G
from . import nodes
from .platforms import PLATFORMS

_ID = S.s_str("content id c<n>", pattern=C.CID_RE.pattern, maxLength=14)
DAY = 86400


class Site(Module):
    """Base class of a content platform. Subclasses set `name`/`host` (= the platform name) and may add tools."""
    platform: str = ""
    version = "1"
    depends = ("identity", "content", "social")

    def __init__(self, config=None):
        super().__init__(config)
        self.spec = PLATFORMS[self.platform]

    # ------------------------------------------------------------------ helpers
    def _mine(self, ctx: Ctx, cid: str):
        rec = C.module(ctx.world)._rec(ctx, cid)
        if rec["platform"] != self.platform:
            raise NotFound(f"{cid} is not on {self.platform}")
        return rec

    def _items(self, world, actor: str, mask: np.ndarray) -> list[str]:
        tbl = world.state.tables["content"]
        return [C.cid(i) for i in np.flatnonzero(mask) if C.readable(world, actor, tbl[C.cid(i)])]

    def _here(self, world) -> np.ndarray:
        cat = C.catalogue(world)
        return (cat["platform"] == C.PLATFORM_CODE[self.platform]) & ~cat["removed"]

    # ------------------------------------------------------------------ tools (attached by _site_endpoints)
    def do_publish(self, ctx, ref, title, kind=None, tags=(), description="", duration_s=None, access="public",
                   reply_to=None, as_org=None):
        cm = C.module(ctx.world)
        creator = cm._creator(ctx, as_org)
        kind = kind or self.spec["kinds"][0]
        c = cm.publish(ctx, creator, ref, kind, self.platform, title, tags=tags, duration_s=duration_s,
                       description=description, access=access, reply_to=reply_to)
        return {"id": c, "n": C.index_of(c), "platform": self.platform, "kind": kind,
                "url": f"http://{self.host}.crucible/item?id={c}"}

    def do_latest(self, ctx, limit=20, offset=0, tag=None, kind=None):
        m = self._here(ctx.world)
        if kind is not None:
            m &= C.catalogue(ctx.world)["kind"] == C.KIND_CODE[kind]
        ids_ = self._items(ctx.world, ctx.actor, m)
        if tag is not None:
            tbl = ctx.state.tables["content"]
            ids_ = [i for i in ids_ if tag in tbl[i]["tags"]]
        items = C.module(ctx.world).listing(ctx.world, ctx.actor, ids_, limit, offset)
        return {"items": items, "total": len(ids_)}

    def do_channel(self, ctx, creator, limit=20):
        from ..services.identity import ident
        p = creator if ids.index(creator) is not None else ident(ctx.world).resolve_handle(ctx.state, creator)
        node = nodes.node_of(ctx.world, p)
        if node is None:
            return {"creator": p, "items": [], "followers": 0}
        m = self._here(ctx.world) & (C.catalogue(ctx.world)["creator"] == node)
        items = C.module(ctx.world).listing(ctx.world, ctx.actor, self._items(ctx.world, ctx.actor, m), limit)
        return {"creator": p, "items": items, "followers": int(G.followers(ctx.world, node).size)}

    def _distributed(self, world) -> bool:
        aud = world.modules.get("aud")
        return "feed" in world.modules and aud is not None and self.platform in getattr(aud, "platforms", ())

    def feed_items(self, ctx, limit: int) -> tuple[list[dict], str]:
        world = ctx.world
        cm = C.module(world)
        tbl = world.state.tables["content"]
        if ids.index(ctx.actor) is not None and self._distributed(world):
            got = ctx.call("feed.home", {"platform": self.platform, "k": int(limit)})
            out = []
            for it in got.get("items", []):
                cid = (world.state.tables["aud_item_info"].get(str(int(it["item"]))) or {}).get("content_id") \
                    if isinstance(it, dict) and "item" in it else None
                if cid and cid in tbl:
                    out.append(dict(cm.view(world, tbl[cid], ctx.actor), **{"from": "recommended"}))
            return out, "recommended"
        cat = C.catalogue(world)
        here = self._here(world)
        me = nodes.node_of(world, ctx.actor)
        fol = G.following(world, me) if me is not None else np.zeros(0, np.int64)
        followed = here & np.isin(cat["creator"], fol)
        first = cm.listing(world, ctx.actor, self._items(world, ctx.actor, followed), limit)
        rest = []
        if len(first) < limit:
            others = here & ~np.isin(cat["creator"], fol)
            if me is not None:
                others &= cat["creator"] != me
            rest = cm.listing(world, ctx.actor, self._items(world, ctx.actor, others), limit - len(first))
        for it in first:
            it["from"] = "following"
        for it in rest:
            it["from"] = "latest"
        return first + rest, "chronological"

    def do_trending(self, ctx, limit=20, hours=24):
        world = ctx.world
        since = ctx.now - int(hours) * 3600
        st = world.state
        here = self._here(world)
        score: dict[int, int] = {}
        for k, c in st.tables["content_comments"].items():
            if c["t"] >= since and not c["removed"]:
                n = st.tables["content"][c["item"]]["n"]
                if here[n]:
                    score[n] = score.get(n, 0) + 1
        sh = G.SHARE.view(world)
        m = (sh["t"] >= since) & (sh["to"] == -1)
        for n in sh["item"][m]:
            if here[int(n)]:
                score[int(n)] = score.get(int(n), 0) + 1
        order = sorted(score, key=lambda n: (-score[n], -n))
        tbl = st.tables["content"]
        cm = C.module(world)
        out = []
        for n in order:
            rec = tbl[C.cid(n)]
            if C.readable(world, ctx.actor, rec):
                out.append(dict(cm.view(world, rec, ctx.actor), interactions=score[n]))
            if len(out) >= limit:
                break
        return {"items": out, "window_hours": int(hours)}

    def do_view(self, ctx, id):
        rec = self._mine(ctx, id)
        cm = C.module(ctx.world)
        out = cm.view(ctx.world, rec, ctx.actor)
        out["comments_list"] = cm.ep_comments(ctx, id, limit=30)
        out["creator_followers"] = int(G.followers(ctx.world, rec["creator_node"]).size)
        return out


def _site_endpoints(cls):
    """Attach the standard site tools to a Site subclass (the endpoint decorator needs per-class schemas)."""
    spec = PLATFORMS[cls.platform]
    label = spec["label"].lower()
    kinds = list(spec["kinds"])

    pub_schema = S.obj({
        "ref": S.s_str("artifact reference (create one with the studio.* tools)", maxLength=71),
        "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE),
        "kind": S.s_enum(kinds, f"default {kinds[0]}"), "tags": C._TAGS,
        "description": S.s_str(maxLength=C.MAX_DESC), "duration_s": S.s_num(minimum=0),
        "access": S.s_enum(list(C.ACCESS)), "reply_to": S.s_str(maxLength=14),
        "as_org": S.s_str(maxLength=40)}, ["ref", "title"])

    def ep_publish(self, ctx, ref, title, kind=None, tags=(), description="", duration_s=None, access="public",
                   reply_to=None, as_org=None):
        return self.do_publish(ctx, ref, title, kind, tags, description, duration_s, access, reply_to, as_org)
    ep_publish.__doc__ = f"Publish an artifact you own on {label} ({', '.join(kinds)}). Returns {{id, url}}."

    def ep_latest(self, ctx, limit=20, offset=0, tag=None, kind=None):
        return self.do_latest(ctx, limit, offset, tag, kind)
    ep_latest.__doc__ = f"Newest items on {label}."

    def ep_channel(self, ctx, creator, limit=20):
        return self.do_channel(ctx, creator, limit)
    ep_channel.__doc__ = f"One creator's items on {label} and how many follow them."

    def ep_feed(self, ctx, limit=20):
        items, how = self.feed_items(ctx, limit)
        return {"items": items, "ranking": how}
    ep_feed.__doc__ = (f"Your {label} feed: the platform's recommendations (population members, when the platform "
                       f"distributes), else newest from accounts you follow, then newest of everyone else.")

    def ep_trending(self, ctx, limit=20, hours=24):
        return self.do_trending(ctx, limit, hours)
    ep_trending.__doc__ = f"Items on {label} with the most comments and public shares in the last hours."

    def ep_view(self, ctx, id):
        return self.do_view(ctx, id)
    ep_view.__doc__ = f"An item on {label} with its comments."

    def ep_comment(self, ctx, id, text, parent=None):
        self._mine(ctx, id)
        return {"comment": C.module(ctx.world).add_comment(ctx, ctx.actor, id, text, parent)}
    ep_comment.__doc__ = f"Comment on an item on {label}."

    table = [
        ("publish", ep_publish, pub_schema, 30, False),
        ("latest", ep_latest, S.obj({"limit": S.s_int(minimum=1, maximum=50, default=20),
                                     "offset": S.s_int(minimum=0, maximum=10_000, default=0),
                                     "tag": S.s_str(pattern=C.TAG_RE, maxLength=32),
                                     "kind": S.s_enum(kinds)}), 2, True),
        ("channel", ep_channel, S.obj({"creator": S.s_str(maxLength=60),
                                       "limit": S.s_int(minimum=1, maximum=50, default=20)}, ["creator"]), 2, True),
        ("feed", ep_feed, S.obj({"limit": S.s_int("attention budget", minimum=1, maximum=50, default=20)}), 5, True),
        ("trending", ep_trending, S.obj({"limit": S.s_int(minimum=1, maximum=50, default=20),
                                         "hours": S.s_int(minimum=1, maximum=24 * 30, default=24)}), 2, True),
        ("view", ep_view, S.obj({"id": _ID}, ["id"]), 5, True),
        ("comment", ep_comment, S.obj({"id": _ID, "text": S.s_str(minLength=1, maxLength=C.MAX_COMMENT),
                                       "parent": S.s_str(maxLength=20)}, ["id", "text"]), 20, False),
    ]
    for name, fn, schema, cost, ro in table:
        fn.__crucible_endpoint__ = dict(name=name, params=schema, doc=fn.__doc__, cost_s=cost, read_only=ro,
                                        returns="", version=None)
        setattr(cls, f"ep_{name}", fn)
    return cls


def _make(platform: str):
    spec = PLATFORMS[platform]
    cls = type(f"Site_{platform}", (Site,), {"name": platform, "host": spec["host"], "platform": platform,
                                             "__doc__": f"{spec['label']}: {spec['doc']}."})
    return register_module(_site_endpoints(cls))


SITES = {p: _make(p) for p in ("shorts", "tube", "posts", "pics", "blog", "pods")}
