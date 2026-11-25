"""The social network (kernel module `social`, host `social`): the follow graph, shares, profiles and timelines,
shared by every platform.

State (columnar, crucible.social.edges):

    follow  src, dst (social node ids), t, t_end (-1 while active), active, via (how the follow came about)
    share   sharer (node), item (catalogue index), t, to (-1: a public repost on the sharer's timeline; else the
            node it was sent to), via

so a population of millions can follow and share through one recorded `cols.batch` per cohort decision
(`follow_batch`, `unfollow_batch`, `share_batch`), while agents use the endpoints. A follow is one directed edge;
following again while the edge is active changes nothing, unfollowing closes the edge (its history stays).

What follows and shares do is decided elsewhere, by ordinary mechanisms: the chronological following feed
(`timeline`), the platforms' following feeds, the consumption engine's social-graph recommenders and the
notifications that reach the followed. Nothing here multiplies a view or a revenue figure.

Endpoints: social.follow / unfollow / followers / following / profile / set_profile / timeline / share / shares /
suggestions / graph_stats.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from ..substrate import artifacts, bus
from . import content as C
from . import nodes
from .edges import EdgeTable, pair_keys

VIA = ("profile", "content", "feed", "suggestion", "community", "share", "search", "population", "message",
       "subscription", "channel", "other")
VIA_CODE = {v: i for i, v in enumerate(VIA)}
FOLLOW = EdgeTable("follow", {"src": ("int64", (), -1), "dst": ("int64", (), -1), "t": ("int64", (), -1),
                              "t_end": ("int64", (), -1), "active": ("bool", (), False), "via": ("uint8", (), 0)})
SHARE = EdgeTable("share", {"sharer": ("int64", (), -1), "item": ("int64", (), -1), "t": ("int64", (), -1),
                            "to": ("int64", (), -1), "via": ("uint8", (), 0)})
MAX_FOLLOWING = 5000
MAX_BIO = 2000


def _mod(world) -> "Social":
    m = world.modules.get("social")
    if m is None:
        raise NotFound("this world has no social network")
    return m  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------------- queries
def followers(world, node: int) -> np.ndarray:
    """Nodes that currently follow `node` (sorted)."""
    v = FOLLOW.view(world)
    return np.unique(v["src"][(v["dst"] == int(node)) & v["active"]])


def following(world, node: int) -> np.ndarray:
    """Nodes `node` currently follows (sorted)."""
    v = FOLLOW.view(world)
    return np.unique(v["dst"][(v["src"] == int(node)) & v["active"]])


def follower_counts(world, nodes_: np.ndarray) -> np.ndarray:
    v = FOLLOW.view(world)
    d = v["dst"][v["active"]]
    nodes_ = np.asarray(nodes_, dtype=np.int64)
    if not d.size or not nodes_.size:
        return np.zeros(nodes_.size, dtype=np.int64)
    u, c = np.unique(d, return_counts=True)
    pos = np.searchsorted(u, nodes_)
    pos = np.clip(pos, 0, len(u) - 1)
    return np.where(u[pos] == nodes_, c[pos], 0).astype(np.int64)


def is_following(world, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Vectorised: does src[i] currently follow dst[i]?"""
    v = FOLLOW.view(world)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if not src.size:
        return np.zeros(0, dtype=bool)
    act = v["active"]
    if not act.any():
        return np.zeros(src.size, dtype=bool)
    have = np.unique(pair_keys(v["src"][act], v["dst"][act]))
    return np.isin(pair_keys(src, dst), have)


def shares_of(world, item: int | None = None, sharer: int | None = None) -> dict[str, np.ndarray]:
    v = SHARE.view(world)
    m = np.ones(v["item"].shape, dtype=bool)
    if item is not None:
        m &= v["item"] == int(item)
    if sharer is not None:
        m &= v["sharer"] == int(sharer)
    return {k: a[m] for k, a in v.items()}


# ---------------------------------------------------------------------------------------------------- population API
def follow_batch(ctx: Ctx, src: np.ndarray, dst: np.ndarray, *, via: str = "population") -> int:
    """Record that each src[i] now follows dst[i] (one cohort decision). Self-follows, repeats and pairs already
    following are dropped. Returns the number of new edges."""
    mod = _mod(ctx.world)
    return mod.add_follows(ctx, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), via)


def unfollow_batch(ctx: Ctx, src: np.ndarray, dst: np.ndarray) -> int:
    return _mod(ctx.world).end_follows(ctx, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64))


def share_batch(ctx: Ctx, sharers: np.ndarray, items: np.ndarray, *, to: np.ndarray | None = None,
                via: str = "population") -> int:
    """Record shares: sharers[i] shares catalogue item items[i] publicly (to=-1) or with node to[i]."""
    return _mod(ctx.world).add_shares(ctx, np.asarray(sharers, dtype=np.int64), np.asarray(items, dtype=np.int64),
                                      None if to is None else np.asarray(to, dtype=np.int64), via)


bus.event_type("social.notice", S.obj({"what": {"type": "string", "enum": ["follow", "share", "mention"]},
                                       "by": {"type": "string"}, "to": {"type": "string"},
                                       "item": {"type": ["string", "null"]}, "note": {"type": "string"}},
                                      ["what", "by", "to"]), owner="social",
               doc="a notification for a principal: someone followed it, shared an item with it or mentioned it")
bus.event_type("social.profile", S.obj({"principal": {"type": "string"}, "display_name": {"type": "string"},
                                        "bio": {"type": "string"}, "avatar": {"type": ["string", "null"]}},
                                       ["principal"]), owner="social", doc="a public profile was set")

_TARGET = S.s_str("a principal (p:<handle>, o:<handle>, p<i>) or a handle", maxLength=60)


@register_module
class Social(Module):
    name = "social"
    host = "social"
    version = "1"
    depends = ("identity", "content", "social_ids", "social_rows", "cols")
    DEFAULTS = {"capacity": 4096}

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(self.DEFAULTS, **self.config)

    def install(self, world):
        FOLLOW.declare(world, int(self.cfg["capacity"]))
        SHARE.declare(world, int(self.cfg["capacity"]))
        world.state.declare_table("social_profiles")
        world.hooks["social.creator_stats"] = self.creator_stats
        bus.enforce(world, owner="social")

    # ------------------------------------------------------------------ internal API
    def add_follows(self, ctx: Ctx, src: np.ndarray, dst: np.ndarray, via: str) -> int:
        if via not in VIA_CODE:
            raise InvalidAction(f"via must be one of {VIA}")
        if src.shape != dst.shape or src.ndim != 1:
            raise InvalidAction("src and dst must be equal-length vectors")
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if not src.size:
            return 0
        k = pair_keys(src, dst)
        _, first = np.unique(k, return_index=True)
        first = np.sort(first)
        src, dst, k = src[first], dst[first], k[first]
        v = FOLLOW.view(ctx.world)
        if v["active"].any():
            have = pair_keys(v["src"][v["active"]], v["dst"][v["active"]])
            new = ~np.isin(k, have)
            src, dst = src[new], dst[new]
        if not src.size:
            return 0
        n = src.size
        FOLLOW.append(ctx, "social.follow", {"src": src, "dst": dst, "t": np.full(n, ctx.now, np.int64),
                                             "t_end": np.full(n, -1, np.int64), "active": np.ones(n, bool),
                                             "via": np.full(n, VIA_CODE[via], np.uint8)}, extra={"via": via})
        return int(n)

    def end_follows(self, ctx: Ctx, src: np.ndarray, dst: np.ndarray) -> int:
        v = FOLLOW.view(ctx.world)
        act = np.flatnonzero(v["active"])
        if not act.size or not src.size:
            return 0
        rows = act[np.isin(pair_keys(v["src"][act], v["dst"][act]), pair_keys(src, dst))]
        if not rows.size:
            return 0
        FOLLOW.update(ctx, "social.unfollow", rows, {"active": np.zeros(rows.size, bool),
                                                     "t_end": np.full(rows.size, ctx.now, np.int64)})
        return int(rows.size)

    def add_shares(self, ctx: Ctx, sharers: np.ndarray, items: np.ndarray, to: np.ndarray | None, via: str) -> int:
        if via not in VIA_CODE:
            raise InvalidAction(f"via must be one of {VIA}")
        if sharers.shape != items.shape:
            raise InvalidAction("sharers and items must be equal-length vectors")
        if not sharers.size:
            return 0
        n_items = C.n_items(ctx.world)
        if int(items.min()) < 0 or int(items.max()) >= n_items:
            raise InvalidAction("share of an unknown item")
        removed = C.catalogue(ctx.world)["removed"]
        if removed[items].any():
            raise InvalidAction("a removed item cannot be shared")
        n = sharers.size
        SHARE.append(ctx, "social.share", {"sharer": sharers, "item": items, "t": np.full(n, ctx.now, np.int64),
                                           "to": np.full(n, -1, np.int64) if to is None else to,
                                           "via": np.full(n, VIA_CODE[via], np.uint8)}, extra={"via": via})
        return int(n)

    def creator_stats(self, world, principal: str) -> dict:
        node = nodes.node_of(world, principal)
        if node is None:
            return {"followers": 0, "following": 0, "shares_of_items": 0}
        tbl = world.state.tables["content"]
        mine = {tbl[i]["n"] for i in world.state.find_ids("content", "creator", principal)}
        sh = SHARE.view(world)
        return {"followers": int(followers(world, node).size), "following": int(following(world, node).size),
                "shares_of_items": int(np.isin(sh["item"], list(mine)).sum()) if mine else 0}

    def _resolve(self, ctx: Ctx, target: str) -> str:
        from ..services.identity import ident
        if nodes.ids.index(target) is not None:
            if not ctx.state.has("principals", target):
                raise NotFound(f"no principal {target}")
            return target
        return ident(ctx.world).resolve_handle(ctx.state, target)

    # ------------------------------------------------------------------ endpoints: the graph
    @endpoint("follow", S.obj({"target": _TARGET, "via": S.s_enum(list(VIA), "where you found them")}, ["target"]),
              cost_s=2)
    def ep_follow(self, ctx, target, via="profile"):
        """Follow a person or organisation: their publications enter your timeline and following feeds."""
        tgt = self._resolve(ctx, target)
        if tgt == ctx.actor:
            raise InvalidAction("you cannot follow yourself")
        a, b = nodes.ensure(ctx, ctx.actor), nodes.ensure(ctx, tgt)
        if following(ctx.world, a).size >= MAX_FOLLOWING:
            raise Conflict(f"you follow {MAX_FOLLOWING} accounts already")
        if not self.add_follows(ctx, np.array([a]), np.array([b]), via):
            raise Conflict(f"you already follow {tgt}")
        bus.emit(ctx, "social.notice", {"what": "follow", "by": ctx.actor, "to": tgt, "item": None, "note": ""},
                 vis=(ctx.actor, tgt))
        return {"following": tgt, "followers_of_target": int(followers(ctx.world, b).size)}

    @endpoint("unfollow", S.obj({"target": _TARGET}, ["target"]), cost_s=2)
    def ep_unfollow(self, ctx, target):
        """Stop following a person or organisation."""
        tgt = self._resolve(ctx, target)
        a, b = nodes.node_of(ctx.world, ctx.actor), nodes.node_of(ctx.world, tgt)
        if a is None or b is None or not self.end_follows(ctx, np.array([a]), np.array([b])):
            raise NotFound(f"you do not follow {tgt}")
        return {"unfollowed": tgt}

    def _people(self, world, arr: np.ndarray, limit: int) -> list[str]:
        return [p for p in nodes.principals_of_nodes(world, arr[:limit]) if p]

    @endpoint("followers", S.obj({"principal": _TARGET, "limit": S.s_int(minimum=1, maximum=500, default=100)}),
              read_only=True, cost_s=1)
    def ep_followers(self, ctx, principal=None, limit=100):
        """Who follows a principal (default: you), and how many."""
        p = self._resolve(ctx, principal) if principal else ctx.actor
        n = nodes.node_of(ctx.world, p)
        arr = followers(ctx.world, n) if n is not None else np.zeros(0, np.int64)
        return {"principal": p, "count": int(arr.size), "followers": self._people(ctx.world, arr, limit)}

    @endpoint("following", S.obj({"principal": _TARGET, "limit": S.s_int(minimum=1, maximum=500, default=100)}),
              read_only=True, cost_s=1)
    def ep_following(self, ctx, principal=None, limit=100):
        """Whom a principal (default: you) follows."""
        p = self._resolve(ctx, principal) if principal else ctx.actor
        n = nodes.node_of(ctx.world, p)
        arr = following(ctx.world, n) if n is not None else np.zeros(0, np.int64)
        return {"principal": p, "count": int(arr.size), "following": self._people(ctx.world, arr, limit)}

    @endpoint("profile", S.obj({"principal": _TARGET, "items": S.s_int(minimum=0, maximum=50, default=10)}),
              read_only=True, cost_s=2)
    def ep_profile(self, ctx, principal=None, items=10):
        """A public profile: name, bio, avatar, followers, following and recent public items."""
        p = self._resolve(ctx, principal) if principal else ctx.actor
        prof = ctx.state.get("social_profiles", p) or {}
        n = nodes.node_of(ctx.world, p)
        cm = C.module(ctx.world)
        ids = ctx.state.find_ids("content", "creator", p)
        tbl = ctx.state.tables["content"]
        visible = [i for i in ids if C.readable(ctx.world, ctx.actor, tbl[i])]
        me = nodes.node_of(ctx.world, ctx.actor)
        return {"principal": p, "display_name": prof.get("display_name", p), "bio": prof.get("bio", ""),
                "avatar": prof.get("avatar"),
                "followers": int(followers(ctx.world, n).size) if n is not None else 0,
                "following": int(following(ctx.world, n).size) if n is not None else 0,
                "you_follow": bool(me is not None and n is not None and is_following(ctx.world, [me], [n])[0]),
                "items": cm.listing(ctx.world, ctx.actor, visible, items)}

    @endpoint("set_profile", S.obj({"display_name": S.s_str(minLength=1, maxLength=80),
                                    "bio": S.s_str(maxLength=MAX_BIO),
                                    "avatar": S.s_str("an image artifact you own", maxLength=71)}), cost_s=10)
    def ep_set_profile(self, ctx, display_name=None, bio=None, avatar=None):
        """Set your public profile (display name, bio, avatar image). The avatar becomes public."""
        old = ctx.state.get("social_profiles", ctx.actor) or {}
        if avatar is not None:
            artifacts.parse_ref(avatar)
            a = ctx.state.get("artifacts", avatar)
            if a is None or a["owner"] != ctx.actor or a["kind"] != "image":
                raise NotFound(f"no image artifact {avatar} of yours")
            artifacts.publish(ctx, avatar)
        nodes.ensure(ctx, ctx.actor)
        bus.emit(ctx, "social.profile", {"principal": ctx.actor,
                                         "display_name": display_name or old.get("display_name", ctx.actor),
                                         "bio": old.get("bio", "") if bio is None else bio,
                                         "avatar": avatar if avatar is not None else old.get("avatar")}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ endpoints: timelines and shares
    @endpoint("timeline", S.obj({"limit": S.s_int("attention budget", minimum=1, maximum=50, default=20),
                                 "before": S.s_int("only entries before this time", minimum=0),
                                 "platform": S.s_str(maxLength=20)}), read_only=True, cost_s=5)
    def ep_timeline(self, ctx, limit=20, before=None, platform=None):
        """Your chronological timeline: items published by people you follow and items they shared publicly,
        newest first (on every platform, or one)."""
        me = nodes.node_of(ctx.world, ctx.actor)
        if me is None:
            return {"entries": [], "following": 0}
        return {"entries": self.timeline(ctx.world, ctx.actor, me, limit, before, platform),
                "following": int(following(ctx.world, me).size)}

    def timeline(self, world, actor: str, me: int, limit: int, before=None, platform=None) -> list[dict]:
        fol = following(world, me)
        if not fol.size:
            return []
        cat = C.catalogue(world)
        cut = np.iinfo(np.int64).max if before is None else int(before)
        pcode = C.PLATFORM_CODE.get(platform) if platform else None
        m = np.isin(cat["creator"], fol) & ~cat["removed"] & (cat["published_at"] < cut)
        if pcode is not None:
            m &= cat["platform"] == pcode
        own = np.flatnonzero(m)
        sh = SHARE.view(world)
        sm = np.isin(sh["sharer"], fol) & (sh["to"] == -1) & (sh["t"] < cut)
        entries = [(int(cat["published_at"][i]), int(i), "post", int(cat["creator"][i])) for i in own]
        for t, it, s in zip(sh["t"][sm], sh["item"][sm], sh["sharer"][sm]):
            if not cat["removed"][it] and (pcode is None or cat["platform"][it] == pcode):
                entries.append((int(t), int(it), "share", int(s)))
        entries.sort(key=lambda e: (-e[0], -e[1], e[2]))
        cm = C.module(world)
        tbl = world.state.tables["content"]
        out = []
        for t, it, typ, by in entries:
            rec = tbl[C.cid(it)]
            if not C.readable(world, actor, rec):
                continue
            out.append({"type": typ, "t": t, "by": nodes.principal_of(world, by), "item": cm.view(world, rec, actor)})
            if len(out) >= limit:
                break
        return out

    @endpoint("share", S.obj({"id": S.s_str("content id c<n>", pattern=C.CID_RE.pattern, maxLength=14),
                              "to": S.s_arr(_TARGET, "send it to these principals instead of reposting publicly",
                                            maxItems=20),
                              "note": S.s_str(maxLength=500)}, ["id"]), cost_s=5)
    def ep_share(self, ctx, id, to=None, note=""):
        """Share an item: a public repost on your timeline (seen by your followers), or sent to people."""
        rec = C.module(ctx.world)._rec(ctx, id)
        if rec["removed"]:
            raise NotFound(f"no content {id}")
        me = nodes.ensure(ctx, ctx.actor)
        if to:
            tgts = []
            for t in to:
                p = self._resolve(ctx, t)
                if p == ctx.actor:
                    raise InvalidAction("you cannot share with yourself")
                if not C.readable(ctx.world, p, rec):
                    raise Forbidden(f"{p} cannot open {id}")
                tgts.append(p)
            tn = np.array([nodes.ensure(ctx, p) for p in tgts], dtype=np.int64)
            self.add_shares(ctx, np.full(tn.size, me), np.full(tn.size, rec["n"]), tn, "message")
            for p in tgts:
                bus.emit(ctx, "social.notice", {"what": "share", "by": ctx.actor, "to": p, "item": id,
                                                "note": str(note)[:500]}, vis=(ctx.actor, p))
            msg = ctx.world.modules.get("msg")
            if msg is not None:
                msg.send_share(ctx, ctx.actor, tgts, id, note)
            return {"shared": id, "to": tgts}
        if rec["access"] != "public":
            raise Forbidden("only public items can be reposted publicly")
        self.add_shares(ctx, np.array([me]), np.array([rec["n"]]), None, "feed")
        if rec["creator"] != ctx.actor:
            bus.emit(ctx, "social.notice", {"what": "share", "by": ctx.actor, "to": rec["creator"], "item": id,
                                            "note": str(note)[:500]}, vis=(ctx.actor, rec["creator"]))
        return {"shared": id, "public": True}

    @endpoint("shares", S.obj({"id": S.s_str(pattern=C.CID_RE.pattern, maxLength=14),
                               "limit": S.s_int(minimum=1, maximum=200, default=50)}, ["id"]), read_only=True)
    def ep_shares(self, ctx, id, limit=50):
        """Public reposts of an item (who and when)."""
        rec = C.module(ctx.world)._rec(ctx, id)
        s = shares_of(ctx.world, item=rec["n"])
        pub = s["to"] == -1
        order = np.argsort(-s["t"][pub], kind="stable")
        who = s["sharer"][pub][order][:limit]
        return {"id": id, "public_shares": int(pub.sum()),
                "recent": [{"by": nodes.principal_of(ctx.world, int(b)), "t": int(t)}
                           for b, t in zip(who, s["t"][pub][order][:limit])]}

    @endpoint("suggestions", S.obj({"limit": S.s_int(minimum=1, maximum=50, default=10)}), read_only=True, cost_s=2)
    def ep_suggestions(self, ctx, limit=10):
        """People followed by the people you follow (friends of friends), by how many of them follow each."""
        me = nodes.node_of(ctx.world, ctx.actor)
        if me is None:
            return []
        fol = following(ctx.world, me)
        if not fol.size:
            return []
        v = FOLLOW.view(ctx.world)
        m = np.isin(v["src"], fol) & v["active"]
        cand = v["dst"][m]
        cand = cand[(cand != me) & ~np.isin(cand, fol)]
        if not cand.size:
            return []
        u, c = np.unique(cand, return_counts=True)
        order = np.lexsort((u, -c))[:limit]
        return [{"principal": nodes.principal_of(ctx.world, int(u[i])), "followed_by_people_you_follow": int(c[i])}
                for i in order]

    @endpoint("graph_stats", S.obj(), read_only=True, cost_s=0)
    def ep_graph_stats(self, ctx):
        """Size of the public follow graph and share record."""
        v = FOLLOW.view(ctx.world)
        s = SHARE.view(ctx.world)
        return {"edges_active": int(v["active"].sum()), "edges_ever": int(v["active"].size),
                "public_shares": int((s["to"] == -1).sum()), "direct_shares": int((s["to"] != -1).sum())}

    # ------------------------------------------------------------------ appliers
    @applier("social.notice")
    def ap_notice(self, ev):
        return None                      # a notification only: no state changes

    @applier("social.profile")
    def ap_profile(self, ev):
        p = ev.payload
        self.world.state.put("social_profiles", p["principal"], {"display_name": p.get("display_name", ""),
                                                                 "bio": p.get("bio", ""), "avatar": p.get("avatar"),
                                                                 "t": ev.t})

    def check_invariants(self, world):
        errs = []
        v = FOLLOW.view(world)
        act = v["active"]
        if act.any():
            k = pair_keys(v["src"][act], v["dst"][act])
            if np.unique(k).size != k.size:
                errs.append("duplicate active follow edges")
            if np.any(v["src"][act] == v["dst"][act]):
                errs.append("self-follow edge")
        if np.any(v["t_end"][act] != -1) or np.any(v["t_end"][~act] < v["t"][~act]):
            errs.append("follow edge end times are inconsistent")
        s = SHARE.view(world)
        if s["item"].size and int(s["item"].max()) >= C.n_items(world):
            errs.append("share of an unknown item")
        return errs

    def observe(self, world, actor):
        n = nodes.node_of(world, actor)
        if n is None:
            return None
        return {"followers": int(followers(world, n).size), "following": int(following(world, n).size)}
