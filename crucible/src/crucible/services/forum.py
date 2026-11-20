"""Public forum (host `forum`): word of mouth, reviews, announcements, help requests.

Boards are created at genesis from `config["boards"]` (default general/dev/market/jobs/help). Posts and replies
are public (`vis="public"`) and indexed in search as kind `post`, with the service hosts they mention
(`<name>.crucible`, kept only if the `is_host` hook accepts the name). Reading a post makes the reader learn
those hosts. Votes are +1/-1, one per principal per post (changeable), never on one's own posts (or posts of
an organisation one belongs to). `feed` is attention-limited (<= 30 items): hot posts from followed principals,
followed boards and the most active boards.

Internal API: `post(ctx, author, board, title, body, tags=(), hosts=()) -> post_id` (used by simulated
consumers to publish reviews), `reply(ctx, author, post_id, body, hosts=()) -> reply_id`,
`create_board(ctx, name, title, description)`.

Invariants: post score/up/down equal the recorded votes, reply counts match, no self-votes, vote ids are
`<post>|<voter>` (uniqueness by construction), board post counters match.
"""
from __future__ import annotations

import re

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, register_module
from .identity import ident
from .search import extract_hosts, index as search_index, unindex as search_unindex

DEFAULT_BOARDS = {"general": "General discussion", "dev": "Software development and tools",
                  "market": "Products, services, reviews and deals", "jobs": "Hiring and looking for work",
                  "help": "Questions and support"}
BOARD_RE = r"[a-z][a-z0-9-]{1,30}"
TAG_RE = r"[a-z0-9][a-z0-9-]{0,23}"
MAX_BODY = 20_000
FEED_MAX = 30
POPULAR_BOARDS = 3
FEED_SOURCE_DEPTH = 50
MAX_FOLLOWS = 200


def _num(rid: str) -> int:
    try:
        return int(rid.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


@register_module
class Forum(Module):
    name = "forum"
    host = "forum"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("forum_boards")
        st.declare_table("forum_posts", index=("board", "author"))
        st.declare_table("forum_replies", index=("post", "author"))
        st.declare_table("forum_votes", index=("post", "voter"))
        st.declare_table("forum_follows", index=("follower", "target"))

    def genesis(self, ctx):
        boards = self.config.get("boards") or list(DEFAULT_BOARDS)
        for b in boards:
            self.create_board(ctx, b, b.capitalize(), DEFAULT_BOARDS.get(b, ""))

    # ------------------------------------------------------------------ internal API
    def create_board(self, ctx: Ctx, name: str, title: str, description: str = "") -> None:
        if not re.fullmatch(BOARD_RE, name):
            raise InvalidAction(f"bad board name {name!r}")
        if ctx.state.has("forum_boards", name):
            raise Conflict(f"board {name} exists")
        ctx.emit("forum.board", {"id": name, "title": title[:80], "description": description[:500]}, vis="public")

    def post(self, ctx: Ctx, author: str, board: str, title: str, body: str, tags=(), hosts=()) -> str:
        """Publish a post as `author` (no authorisation check: callers authorise). Indexed in search."""
        st = ctx.state
        if not st.has("forum_boards", board):
            raise NotFound(f"no board {board}")
        title, body = str(title)[:200], str(body)[:MAX_BODY]
        tags = sorted({str(t).lower()[:24] for t in tags if t})[:5]
        hs = extract_hosts(ctx.world, title + "\n" + body, hosts)
        pid = st.new_id("fpost")
        ctx.emit("forum.post", {"id": pid, "board": board, "author": author, "title": title, "body": body,
                                "tags": tags, "hosts": hs}, vis="public")
        search_index(ctx, pid, "post", title, body + ("\n" + " ".join(tags) if tags else ""), author, ref=pid,
                     hosts=hs)
        return pid

    def reply(self, ctx: Ctx, author: str, post_id: str, body: str, hosts=()) -> str:
        st = ctx.state
        p = st.get("forum_posts", post_id)
        if p is None or p["status"] != "visible":
            raise NotFound(f"no post {post_id}")
        body = str(body)[:MAX_BODY]
        hs = extract_hosts(ctx.world, body, hosts)
        rid = st.new_id("freply")
        ctx.emit("forum.reply", {"id": rid, "post": post_id, "author": author, "body": body, "hosts": hs},
                 vis="public")
        search_index(ctx, rid, "post", "Re: " + p["title"], body, author, ref=post_id, hosts=hs)
        return rid

    def _post(self, ctx, post_id: str):
        p = ctx.state.get("forum_posts", post_id)
        if p is None or p["status"] != "visible":
            raise NotFound(f"no post {post_id}")
        return p

    def _author(self, ctx, as_org: str | None) -> str:
        if not as_org:
            return ctx.actor
        org = ident(ctx.world).resolve_handle(ctx.state, as_org)
        if not org.startswith("o:"):
            raise InvalidAction("as_org must be an organisation")
        ident(ctx.world).authorize(ctx, org, "forum.post")
        return org

    def _own(self, ctx, author: str) -> bool:
        """Is `author` the actor or an organisation the actor belongs to (any role)?"""
        if author == ctx.actor:
            return True
        return author.startswith("o:") and ctx.state.has("members", f"{author}|{ctx.actor}")

    # ------------------------------------------------------------------ endpoints
    @endpoint("boards", S.obj(), read_only=True)
    def ep_boards(self, ctx):
        """All boards with their post counts."""
        return [{"board": b, "title": r["title"], "description": r["description"], "posts": r["posts"]}
                for b, r in ctx.state.scan("forum_boards")]

    @endpoint("post", S.obj({"board": S.s_str(pattern=BOARD_RE, maxLength=31),
                             "title": S.s_str(minLength=1, maxLength=200),
                             "body": S.s_str("text; mention services as <host>.crucible", minLength=1,
                                             maxLength=MAX_BODY),
                             "tags": S.s_arr(S.s_str(pattern=TAG_RE, maxLength=24), maxItems=5),
                             "as_org": S.s_str("post on behalf of an organisation (needs forum.post)",
                                               maxLength=40)},
                            ["board", "title", "body"]), cost_s=120)
    def ep_post(self, ctx, board, title, body, tags=(), as_org=None):
        """Publish a public post. It is indexed by search; services you mention as <host>.crucible become
        discoverable to readers."""
        author = self._author(ctx, as_org)
        pid = self.post(ctx, author, board, title, body, tags)
        return {"post": pid, "hosts": list(ctx.state.tables["forum_posts"][pid]["hosts"])}

    @endpoint("reply", S.obj({"post": S.s_str(maxLength=40), "body": S.s_str(minLength=1, maxLength=MAX_BODY),
                              "as_org": S.s_str(maxLength=40)}, ["post", "body"]), cost_s=60)
    def ep_reply(self, ctx, post, body, as_org=None):
        """Reply publicly to a post."""
        author = self._author(ctx, as_org)
        return {"reply": self.reply(ctx, author, post, body)}

    @endpoint("read", S.obj({"post": S.s_str(maxLength=40),
                             "replies": S.s_int("max replies to return (oldest first)", minimum=0, maximum=100,
                                                default=50)}, ["post"]), cost_s=10, read_only=True)
    def ep_read(self, ctx, post, replies=50):
        """Read a post with its replies. You learn the service hosts mentioned in what you read."""
        p = self._post(ctx, post)
        st = ctx.state
        rids = sorted(st.find_ids("forum_replies", "post", post), key=lambda r: (st.tables["forum_replies"][r]["t"],
                                                                              _num(r)))
        reps = []
        hosts = list(p["hosts"])
        for rid in rids[:replies]:
            r = st.tables["forum_replies"][rid]
            reps.append({"reply": rid, "author": r["author"], "body": r["body"], "t": r["t"],
                         "hosts": list(r["hosts"])})
            hosts.extend(r["hosts"])
        for h in sorted(set(hosts)):
            ctx.world.learn(ctx, ctx.actor, "host", h)
        my = st.get("forum_votes", f"{post}|{ctx.actor}")
        return {"post": post, "board": p["board"], "author": p["author"], "title": p["title"], "body": p["body"],
                "tags": list(p["tags"]), "hosts": list(p["hosts"]), "t": p["t"], "score": p["score"],
                "n_replies": p["replies"], "replies": reps, "your_vote": my["value"] if my else 0}

    @endpoint("list", S.obj({"board": S.s_str(pattern=BOARD_RE, maxLength=31),
                             "sort": S.s_enum(["new", "top", "active"], "ordering"),
                             "limit": S.s_int(minimum=1, maximum=FEED_MAX, default=20),
                             "offset": S.s_int(minimum=0, maximum=1000, default=0)}, ["board"]), read_only=True)
    def ep_list(self, ctx, board, sort="new", limit=20, offset=0):
        """List posts of a board (titles and snippets)."""
        st = ctx.state
        if not st.has("forum_boards", board):
            raise NotFound(f"no board {board}")
        rows = [(pid, st.tables["forum_posts"][pid]) for pid in st.find_ids("forum_posts", "board", board)]
        rows = [(pid, p) for pid, p in rows if p["status"] == "visible"]
        if sort == "top":
            rows.sort(key=lambda x: (-x[1]["score"], -x[1]["t"], _num(x[0])))
        elif sort == "active":
            rows.sort(key=lambda x: (-x[1]["last_activity"], -_num(x[0])))
        else:
            rows.sort(key=lambda x: (-x[1]["t"], -_num(x[0])))
        return [_summary(pid, p) for pid, p in rows[offset:offset + limit]]

    @endpoint("vote", S.obj({"post": S.s_str(maxLength=40),
                             "value": {"type": "integer", "enum": [-1, 1], "description": "+1 or -1"}},
                            ["post", "value"]), cost_s=2)
    def ep_vote(self, ctx, post, value):
        """Up- or down-vote a post (one vote per principal; voting again with the other value changes it)."""
        p = self._post(ctx, post)
        if self._own(ctx, p["author"]):
            raise Forbidden("you cannot vote on your own posts")
        old = ctx.state.get("forum_votes", f"{post}|{ctx.actor}")
        prev = old["value"] if old else 0
        if prev == value:
            raise Conflict("you already cast this vote")
        ctx.emit("forum.vote", {"post": post, "voter": ctx.actor, "value": value, "prev": prev}, vis=(ctx.actor,))
        return {"score": ctx.state.tables["forum_posts"][post]["score"]}

    def _target(self, ctx, target: str) -> str:
        if target.startswith("board:"):
            if not ctx.state.has("forum_boards", target[6:]):
                raise NotFound(f"no board {target[6:]}")
            return target
        return ident(ctx.world).resolve_handle(ctx.state, target)

    @endpoint("follow", S.obj({"target": S.s_str("principal/handle, or board:<name>", maxLength=60)}, ["target"]),
              cost_s=2)
    def ep_follow(self, ctx, target):
        """Follow a person, organisation or board: their posts enter your feed."""
        tgt = self._target(ctx, target)
        if tgt == ctx.actor:
            raise InvalidAction("you cannot follow yourself")
        fid = f"{ctx.actor}|{tgt}"
        if ctx.state.has("forum_follows", fid):
            raise Conflict("already following")
        if len(ctx.state.find_ids("forum_follows", "follower", ctx.actor)) >= MAX_FOLLOWS:
            raise Conflict("follow limit reached")
        vis = (ctx.actor,) if tgt.startswith("board:") else (ctx.actor, tgt)
        ctx.emit("forum.follow", {"id": fid, "follower": ctx.actor, "target": tgt}, vis=vis)
        return {"following": tgt}

    @endpoint("unfollow", S.obj({"target": S.s_str(maxLength=60)}, ["target"]), cost_s=2)
    def ep_unfollow(self, ctx, target):
        """Stop following a person, organisation or board."""
        tgt = self._target(ctx, target)
        fid = f"{ctx.actor}|{tgt}"
        if not ctx.state.has("forum_follows", fid):
            raise NotFound("not following")
        ctx.emit("forum.unfollow", {"id": fid}, vis=(ctx.actor,))
        return {"ok": True}

    @endpoint("feed", S.obj({"limit": S.s_int("attention budget", minimum=1, maximum=FEED_MAX, default=20)}),
              cost_s=5, read_only=True)
    def ep_feed(self, ctx, limit=20):
        """Your feed: hot posts from people/organisations and boards you follow, plus the most active boards."""
        return self.feed(ctx.world, ctx.actor, limit)

    def feed(self, world, actor: str, limit: int = 20) -> list[dict]:
        st = world.state
        limit = max(1, min(int(limit), FEED_MAX))
        people, boards = [], []
        for fid in st.find_ids("forum_follows", "follower", actor):
            t = st.tables["forum_follows"][fid]["target"]
            (boards if t.startswith("board:") else people).append(t[6:] if t.startswith("board:") else t)
        popular = sorted(st.tables["forum_boards"], key=lambda b: (-st.tables["forum_boards"][b]["posts"], b))
        boards = sorted(set(boards) | set(popular[:POPULAR_BOARDS]))
        cand: dict[str, bool] = {}
        for src, field, followed in [(p, "author", True) for p in sorted(people)] + \
                                    [(b, "board", False) for b in boards]:
            ids = st.find_ids("forum_posts", field, src)
            ids = sorted(ids, key=lambda r: (st.tables["forum_posts"][r]["t"], _num(r)))[-FEED_SOURCE_DEPTH:]
            for pid in ids:
                cand[pid] = cand.get(pid, False) or followed
        now = world.now
        scored = []
        for pid in sorted(cand):
            p = st.tables["forum_posts"][pid]
            if p["status"] != "visible":
                continue
            age_h = max(0, now - p["t"]) / 3600.0
            hot = (max(p["score"], 0) + 1 + 0.5 * p["replies"]) / (age_h + 2.0) ** 1.5
            if cand[pid]:
                hot *= 2.0
            scored.append((-hot, -p["t"], pid))
        scored.sort()
        return [dict(_summary(pid, st.tables["forum_posts"][pid]), followed=cand[pid])
                for _, _, pid in scored[:limit]]

    @endpoint("delete", S.obj({"post": S.s_str(maxLength=40)}, ["post"]), cost_s=5)
    def ep_delete(self, ctx, post):
        """Delete your post (or your organisation's, with forum.post). Its replies disappear from search too."""
        p = self._post(ctx, post)
        if p["author"] != ctx.actor:
            ident(ctx.world).authorize(ctx, p["author"], "forum.post")
        ctx.emit("forum.delete", {"post": post, "by": ctx.actor}, vis="public")
        search_unindex(ctx, post)
        for rid in ctx.state.find_ids("forum_replies", "post", post):
            search_unindex(ctx, rid)
        return {"ok": True}

    # ------------------------------------------------------------------ appliers
    @applier("forum.board")
    def ap_board(self, ev):
        p = ev.payload
        self.world.state.insert("forum_boards", p["id"], {"title": p["title"], "description": p["description"],
                                                          "posts": 0, "created": ev.t})

    @applier("forum.post")
    def ap_post(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("forum_posts", p["id"], {"board": p["board"], "author": p["author"], "title": p["title"],
                                           "body": p["body"], "tags": list(p["tags"]), "hosts": list(p["hosts"]),
                                           "t": ev.t, "score": 0, "up": 0, "down": 0, "replies": 0,
                                           "last_activity": ev.t, "status": "visible"})
        b = st.tables["forum_boards"][p["board"]]
        st.patch("forum_boards", p["board"], posts=b["posts"] + 1)

    @applier("forum.reply")
    def ap_reply(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("forum_replies", p["id"], {"post": p["post"], "author": p["author"], "body": p["body"],
                                             "hosts": list(p["hosts"]), "t": ev.t})
        post = st.tables["forum_posts"][p["post"]]
        st.patch("forum_posts", p["post"], replies=post["replies"] + 1, last_activity=ev.t)

    @applier("forum.vote")
    def ap_vote(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("forum_votes", f"{p['post']}|{p['voter']}", {"post": p["post"], "voter": p["voter"],
                                                             "value": p["value"], "t": ev.t})
        post = st.tables["forum_posts"][p["post"]]
        up = post["up"] + (p["value"] == 1) - (p["prev"] == 1)
        down = post["down"] + (p["value"] == -1) - (p["prev"] == -1)
        st.patch("forum_posts", p["post"], score=up - down, up=up, down=down)

    @applier("forum.follow")
    def ap_follow(self, ev):
        p = ev.payload
        self.world.state.insert("forum_follows", p["id"], {"follower": p["follower"], "target": p["target"],
                                                           "t": ev.t})

    @applier("forum.unfollow")
    def ap_unfollow(self, ev):
        self.world.state.delete("forum_follows", ev.payload["id"])

    @applier("forum.delete")
    def ap_delete(self, ev):
        self.world.state.patch("forum_posts", ev.payload["post"], status="deleted")

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        tally: dict[str, list[int]] = {}
        for vid, v in st.tables["forum_votes"].items():
            if vid != f"{v['post']}|{v['voter']}":
                errs.append(f"vote {vid} has a non-canonical id")
            if v["value"] not in (-1, 1):
                errs.append(f"vote {vid} has value {v['value']}")
            post = st.tables["forum_posts"].get(v["post"])
            if post is None:
                errs.append(f"vote {vid} on a missing post")
                continue
            if post["author"] == v["voter"]:
                errs.append(f"self-vote {vid}")
            t = tally.setdefault(v["post"], [0, 0])
            t[0 if v["value"] == 1 else 1] += 1
        replies: dict[str, int] = {}
        for rid, r in st.tables["forum_replies"].items():
            replies[r["post"]] = replies.get(r["post"], 0) + 1
            if r["post"] not in st.tables["forum_posts"]:
                errs.append(f"reply {rid} to a missing post")
        per_board: dict[str, int] = {}
        for pid, p in st.tables["forum_posts"].items():
            up, down = tally.get(pid, [0, 0])
            if (p["up"], p["down"], p["score"]) != (up, down, up - down):
                errs.append(f"post {pid} score {p['score']} ({p['up']}/{p['down']}) != votes {up}/{down}")
            if p["replies"] != replies.get(pid, 0):
                errs.append(f"post {pid} reply count {p['replies']} != {replies.get(pid, 0)}")
            per_board[p["board"]] = per_board.get(p["board"], 0) + 1
        for b, r in st.tables["forum_boards"].items():
            if r["posts"] != per_board.get(b, 0):
                errs.append(f"board {b} post counter {r['posts']} != {per_board.get(b, 0)}")
        return errs

    def observe(self, world, actor):
        st = world.state
        posts = [st.tables["forum_posts"][pid] for pid in st.find_ids("forum_posts", "author", actor)]
        if not posts:
            return None
        return {"my_posts": len(posts), "replies_received": sum(p["replies"] for p in posts),
                "score": sum(p["score"] for p in posts)}


def _summary(pid: str, p) -> dict:
    return {"post": pid, "board": p["board"], "author": p["author"], "title": p["title"],
            "snippet": p["body"][:200], "tags": list(p["tags"]), "hosts": list(p["hosts"]), "t": p["t"],
            "score": p["score"], "replies": p["replies"]}
