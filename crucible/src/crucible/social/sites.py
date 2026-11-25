"""Content platforms as kernel modules over the shared catalogue: short-form video (`shorts`), long-form video
(`tube`), text posts (`posts`), images (`pics`), articles and blogs (`blog`), audio and podcasts (`pods`).

Each platform is a site with its own host and tools, carrying the catalogue kinds of its registry entry
(crucible.social.platforms). What differs between platforms is real: what may be published there, how a
participant discovers things (a chronological following feed, latest, tags, channels, trending by recorded
interactions, and - when the consumption engine of workstream C2 is in the world - the platform's own
recommender mixture), and how consumption is recorded. All of them share the population, the catalogue, the
follow graph and the identity space.

Tools of every site `<host>`:

    <host>.publish   publish an artifact you own on this platform (content.publish with the platform fixed)
    <host>.latest    newest items (optionally one tag or kind)
    <host>.channel   one creator's items here
    <host>.feed      your feed: the platform recommender mixture when the consumption engine is present,
                     otherwise newest items from accounts you follow, then the newest of everyone else
    <host>.trending  items of the last hours ranked by recorded comments and public shares in that window
    <host>.view      open an item of this platform with its comments
    <host>.watch     consume an item (watch/listen/read for some seconds; like, share, follow) - recorded by the
                     consumption engine (crucible.social.consumption), the only way audience state changes
    <host>.comment   comment on an item of this platform

The platform's recommender configuration (`recommenders`, `schedule` in the module config) is evaluator-only: it
is moved to `world.runtime["recsys_private"]` at install and never appears in an event or a tool result.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, endpoint, register_module
from ..substrate import ids
from . import content as C
from . import graph as G
from . import nodes
from .platforms import PLATFORMS

_ID = S.s_str("content id c<n>", pattern=C.CID_RE.pattern, maxLength=14)
DAY = 86400


def consumption_engine():
    """The consumption engine of workstream C2 (crucible.social.consumption), or None when it is not installed."""
    try:
        from . import consumption as cons      # noqa: F401
    except ImportError:
        return None
    return cons


def recommenders():
    try:
        from . import recsys
    except ImportError:
        return None
    return recsys


class Site(Module):
    """Base class of a content platform. Subclasses set `name`/`host` (= the platform name) and may add tools."""
    platform: str = ""
    version = "1"
    depends = ("identity", "content", "social")

    def __init__(self, config=None):
        cfg = dict(config or {})
        self._recsys = {k: cfg.pop(k) for k in ("recommenders", "schedule") if k in cfg}
        super().__init__(cfg)
        self.spec = PLATFORMS[self.platform]

    def install(self, world):
        if self._recsys:
            world.runtime.setdefault("recsys_private", {})[self.platform] = dict(self._recsys)

    def on_load(self, world):
        if self._recsys:
            world.runtime.setdefault("recsys_private", {})[self.platform] = dict(self._recsys)

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

    def _pop_viewer(self, ctx: Ctx) -> int | None:
        return ids.index(ctx.actor)

    def _impressions(self, ctx: Ctx, items: list[dict]) -> None:
        """Items shown to a population individual acting through tools are impressions of the consumption engine."""
        i = self._pop_viewer(ctx)
        cons = consumption_engine()
        if i is None or cons is None or not items or ctx.read_only and ctx.cmd is None:
            return
        n = np.array([it["n"] for it in items], dtype=np.int64)
        cons.impressions(ctx, self.platform, np.full(n.size, i, dtype=np.int64), n, np.arange(n.size, dtype=np.int64))

    # ------------------------------------------------------------------ tools
    def _publish_schema(self):
        return S.obj({
            "ref": S.s_str("artifact reference (see studio.*)", maxLength=71),
            "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE),
            "kind": S.s_enum(list(self.spec["kinds"]), f"default {self.spec['kinds'][0]}"),
            "tags": C._TAGS, "description": S.s_str(maxLength=C.MAX_DESC),
            "duration_s": S.s_num(minimum=0), "access": S.s_enum(list(C.ACCESS)),
            "reply_to": S.s_str(maxLength=14), "as_org": S.s_str(maxLength=40)}, ["ref", "title"])

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

    def feed_items(self, ctx, limit: int) -> tuple[list[dict], str]:
        world = ctx.world
        rs = recommenders()
        conf = world.runtime.get("recsys_private", {}).get(self.platform)
        cat = C.catalogue(world)
        here = self._here(world)
        cm = C.module(world)
        if rs is not None and conf and conf.get("recommenders") and self._pop_viewer(ctx) is not None:
            mix = rs.mixture(world, self.platform) if hasattr(rs, "mixture") else None
            if mix is None:
                raise ProtocolError("recsys has no mixture() for platform feeds")
            cand = np.flatnonzero(here)
            ranked = mix.rank(world, self.platform, np.array([self._pop_viewer(ctx)]), cand, limit,
                              ctx.rng("feed", self.platform))
            tbl = world.state.tables["content"]
            items = [cm.view(world, tbl[C.cid(int(i))], ctx.actor) for i in ranked[0] if int(i) >= 0]
            return items, "recommended"
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

    def do_watch(self, ctx, id, seconds, liked=False, shared=False, follow=False):
        rec = self._mine(ctx, id)
        cons = consumption_engine()
        i = self._pop_viewer(ctx)
        if cons is None:
            raise NotFound("this world has no consumption engine (crucible.social.consumption)")
        if i is None:
            raise InvalidAction("consumption is recorded for population individuals (p<i>) only")
        dur = rec["duration_s"]
        sec = float(min(seconds, dur)) if dur else float(seconds)
        thr = min(30.0, 0.5 * dur) if dur else 30.0
        one = lambda v, dt: np.array([v], dtype=dt)  # noqa: E731
        cons.consumption(ctx, self.platform, one(i, np.int64), one(rec["n"], np.int64), one(sec, np.float32),
                         one(bool(dur) and sec >= dur - 0.5, bool), one(bool(shared), bool), one(bool(follow), bool),
                         one(bool(liked), bool))
        return {"id": id, "watched_s": sec, "counts_as_view": sec >= thr, "view_threshold_s": thr}


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
        self._impressions(ctx, items)
        return {"items": items, "ranking": how}
    ep_feed.__doc__ = (f"Your {label} feed: ranked by the platform's recommenders, or newest from accounts you "
                       f"follow, then newest of everyone else.")

    def ep_trending(self, ctx, limit=20, hours=24):
        return self.do_trending(ctx, limit, hours)
    ep_trending.__doc__ = f"Items on {label} with the most comments and public shares in the last hours."

    def ep_view(self, ctx, id):
        return self.do_view(ctx, id)
    ep_view.__doc__ = f"An item on {label} with its comments."

    def ep_watch(self, ctx, id, seconds, liked=False, shared=False, follow=False):
        return self.do_watch(ctx, id, seconds, liked, shared, follow)
    ep_watch.__doc__ = ("Consume an item for some seconds (and optionally like it, share it, follow its creator); "
                        "a view needs at least min(30 s, half the duration).")

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
        ("watch", ep_watch, S.obj({"id": _ID, "seconds": S.s_num(minimum=0, maximum=36_000),
                                   "liked": S.s_bool(default=False), "shared": S.s_bool(default=False),
                                   "follow": S.s_bool(default=False)}, ["id", "seconds"]), 30, False),
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
