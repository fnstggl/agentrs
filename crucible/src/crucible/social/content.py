"""The shared content catalogue (kernel module `content`, host `content`; INTERFACE_CONTRACTS section 3).

Every publication on every platform is one row of table `content`:

    {id: "c<n>", n, ref (artifact REF), kind, platform, creator (principal), published_at, duration_s, title, tags,
     removed, ...}

where `kind` is one of video_short | video_long | post | image | article | newsletter | audio | thread | app |
package and `n` (0, 1, 2, ...) is the item index used by population-scale arrays (consumption, audience,
recommenders). The bytes are an actual artifact in the substrate artifact store; `publish` inspects them
(`crucible.social.formats`) and records only facts any reader could establish (format, media type, size,
dimensions, duration, page count, entry point). Nobody can attach a quality, relevance or virality figure.

`publish(ctx, creator, ref, kind, platform, title, *, tags=(), duration_s=None, ...)` is the one way content enters
the catalogue (agents call it through `content.publish` or a platform's `<host>.publish`; the forum bridge for 4.0
forum posts). It validates the artifact against the kind (a short video lasts at most 3 minutes; a stated
`duration_s` must agree with the measured one), publishes the artifact (public content) and indexes it in search
(kind = content kind) so every content kind is findable. `update` publishes a new version (new REF, the history is
kept), `remove` takes an item out of every listing, feed and search (the published bytes stay where they are: a
publication cannot be recalled), `comment` adds a reply to an item.

Access: `public` items are readable by everyone; `subscribers` items (creator memberships, module `creators`) and
`members` items (communities) are readable only through `content.open`, which asks the access providers registered
in `world.hooks["content.access"]`.

Dense per-item arrays (`catalogue(world)`: kind, platform, creator node, published_at, duration_s, removed,
community, access, version) are state written by the same appliers, so recommenders and the consumption engine
can work on whole candidate sets without per-item Python.

Typed events (crucible.substrate.bus): content.published, content.updated, content.removed, content.comment,
content.comment_removed - other modules subscribe to them (audience columns, recommenders, notifications).
"""
from __future__ import annotations

import base64
import re
from typing import Any, Iterable

import numpy as np

from ..core import schema as S
from ..core.canonical import to_jsonable
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from ..substrate import artifacts, bus
from ..substrate.ids import index as ids_index
from . import formats as F
from . import nodes
from .platforms import KIND_FORMATS, KINDS, LONG_MIN_S, PLATFORMS, SHORT_MAX_S

CID_RE = re.compile(r"c(0|[1-9][0-9]{0,11})")
TAG_RE = r"[a-z0-9][a-z0-9-]{0,31}"
ACCESS = ("public", "subscribers", "members")
MAX_TAGS = 10
MAX_TITLE = 200
MAX_DESC = 4000
MAX_COMMENT = 4000
DURATION_TOL = 0.5          # seconds, or 2 %
SEARCH_TEXT = 8000
FORMAT_CODE = {f: i for i, f in enumerate(F.FORMATS)}
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
PLATFORM_CODE = {p: i for i, p in enumerate(PLATFORMS)}
ACCESS_CODE = {a: i for i, a in enumerate(ACCESS)}
ARRAYS = {"kind": ("uint8", 255), "platform": ("uint8", 255), "creator": ("int64", -1), "published_at": ("int64", -1),
          "duration_s": ("float32", 0.0), "removed": ("bool", True), "community": ("int32", -1),
          "access": ("uint8", 0), "format": ("uint8", 255), "version": ("int32", 0), "reply_to": ("int64", -1)}
AKEY = "catalogue."


def cid(n: int) -> str:
    return f"c{int(n)}"


def index_of(content_id: str) -> int:
    m = CID_RE.fullmatch(content_id or "")
    if not m:
        raise InvalidAction(f"not a content id: {str(content_id)[:40]!r} (expected c<n>)")
    return int(m.group(1))


def module(world) -> "Content":
    m = world.modules.get("content")
    if m is None:
        raise ProtocolError("this world has no 'content' module")
    return m  # type: ignore[return-value]


def item(world, content_id: str | int) -> dict:
    """The catalogue record (evaluator/world side: no permission check)."""
    k = cid(content_id) if isinstance(content_id, (int, np.integer)) else content_id
    rec = world.state.tables["content"].get(k)
    if rec is None:
        raise NotFound(f"no content {k}")
    return dict(rec)


def n_items(world) -> int:
    return int(world.state.meta_get("content:n", 0))


def catalogue(world) -> dict[str, np.ndarray]:
    """Read-only per-item arrays of length n_items(world) (see ARRAYS)."""
    n = n_items(world)
    return {k: world.state.array(AKEY + k)[:n] for k in ARRAYS}


def text_of(world, content_id: str) -> str:
    """Searchable text of an item: title, description, tags and the artifact's own text."""
    rec = item(world, content_id)
    try:
        data = artifacts.get(world, rec["ref"])
        body = F.extract_text(data, rec["format"], rec["mime"], limit=SEARCH_TEXT)
    except NotFound:
        body = ""
    return "\n".join(x for x in (rec["title"], rec.get("description", ""), " ".join(rec["tags"]), body) if x)


def publish(ctx: Ctx, creator: str, ref: str, kind: str, platform: str, title: str, *, tags: Iterable[str] = (),
            duration_s: float | None = None, description: str = "", community: str | None = None,
            access: str = "public", reply_to: str | None = None) -> str:
    """Publish the artifact `ref` as `kind` on `platform` for `creator`; returns the content id "c<n>"."""
    return module(ctx.world).publish(ctx, creator, ref, kind, platform, title, tags=tags, duration_s=duration_s,
                                     description=description, community=community, access=access,
                                     reply_to=reply_to)


def readable(world, principal: str, rec) -> bool:
    """May `principal` open this item? Public items: everyone; others: creator side or an access provider."""
    if principal == SYSTEM:
        return True
    if rec["removed"]:
        return rec["creator"] in world.principals_of(principal)
    if rec["access"] == "public":
        return True
    if rec["creator"] in world.principals_of(principal):
        return True
    for name, fn in sorted((world.hooks.get("content.access") or {}).items()):
        if fn(world, principal, rec):
            return True
    return False


# ---------------------------------------------------------------------------------------------------- typed events
_S, _I, _N = {"type": "string"}, {"type": "integer"}, {"type": "number"}
_SN = {"type": ["string", "null"]}
bus.event_type("content.published", S.obj({
    "id": {"type": "string", "pattern": CID_RE.pattern}, "n": {"type": "integer", "minimum": 0},
    "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern}, "kind": {"type": "string", "enum": list(KINDS)},
    "platform": {"type": "string", "enum": list(PLATFORMS)}, "creator": _S, "creator_node": _I, "title": _S,
    "tags": {"type": "array", "items": _S}, "duration_s": {"type": ["number", "null"]}, "description": _S,
    "community": _SN, "community_n": _I, "access": {"type": "string", "enum": list(ACCESS)},
    "format": {"type": "string", "enum": list(F.FORMATS)}, "mime": _S, "size": _I,
    "facts": {"type": "object", "additionalProperties": True}, "reply_to": _SN, "by": _S},
    ["id", "n", "ref", "kind", "platform", "creator", "creator_node", "title", "tags", "duration_s", "access",
     "format", "mime", "size", "facts"]), owner="content",
    doc="an artifact was published as a catalogue item (facts are measured from the bytes, never claimed)")
bus.event_type("content.updated", S.obj({"id": _S, "ref": _S, "version": _I, "title": _S,
                                         "tags": {"type": "array", "items": _S}, "description": _S,
                                         "duration_s": {"type": ["number", "null"]}, "mime": _S, "size": _I,
                                         "format": _S, "facts": {"type": "object", "additionalProperties": True},
                                         "by": _S},
                                        ["id", "ref", "version", "by"]), owner="content",
               doc="a new version of a catalogue item (new artifact; the history is kept)")
bus.event_type("content.removed", S.obj({"id": _S, "by": _S, "reason": _S, "role": _S}, ["id", "by", "reason", "role"]),
               owner="content", doc="an item was taken out of every listing, feed and search")
bus.event_type("content.comment", S.obj({"id": _S, "item": _S, "author": _S, "text": _S, "parent": _SN},
                                        ["id", "item", "author", "text"]), owner="content",
               doc="a comment on a catalogue item (or a reply to a comment)")
bus.event_type("content.comment_removed", S.obj({"id": _S, "by": _S, "reason": _S}, ["id", "by", "reason"]),
               owner="content", doc="a comment was removed by its author, the item's creator or a moderator")

_ID = S.s_str("content id c<n>", pattern=CID_RE.pattern, maxLength=14)
_TAGS = S.s_arr(S.s_str(pattern=TAG_RE, maxLength=32), "topic tags (lowercase)", maxItems=MAX_TAGS)


@register_module
class Content(Module):
    name = "content"
    host = "content"
    version = "1"
    depends = ("identity", "artifacts", "social_ids")

    DEFAULTS = {"capacity": 1024, "bridge_forum": True}

    def __init__(self, config=None):
        super().__init__(config)
        for k in self.config:
            if k not in self.DEFAULTS:
                raise ProtocolError(f"content: unknown config key {k!r}")
        self.cfg = dict(self.DEFAULTS, **self.config)

    # ------------------------------------------------------------------ install
    def install(self, world):
        st = world.state
        st.declare_table("content", index=("creator", "platform", "kind", "community", "ref"))
        st.declare_table("content_comments", index=("item", "author"))
        st.declare_table("content_forum")          # 4.0 forum post id -> catalogue id (forum bridge)
        cap = max(16, int(self.cfg["capacity"]))
        for k, (dt, fill) in ARRAYS.items():
            if AKEY + k not in st.arrays:
                st.set_array(AKEY + k, np.full(cap, fill, dtype=np.dtype(dt)))
        world.hooks.setdefault("content.access", {})
        world.hooks.setdefault("content.post_check", {})
        if self.cfg["bridge_forum"] and any(m["name"] == "forum" for m in world.spec.modules):
            bus.subscribe(world, "forum.post", self._on_forum_post, owner="content")
            bus.subscribe(world, "forum.delete", self._on_forum_delete, owner="content")
        bus.enforce(world, owner="content")

    # ------------------------------------------------------------------ helpers
    def _creator_side(self, ctx: Ctx, principal: str) -> bool:
        return ctx.actor == SYSTEM or principal in ctx.world.principals_of(ctx.actor)

    def _creator(self, ctx: Ctx, as_org: str | None) -> str:
        if not as_org:
            return ctx.actor
        from ..services.identity import ident
        org = ident(ctx.world).resolve_handle(ctx.state, as_org)
        if not org.startswith("o:"):
            raise InvalidAction("as_org must be an organisation")
        ident(ctx.world).authorize(ctx, org, "content.publish")
        return org

    def _rec(self, ctx: Ctx, content_id: str):
        index_of(content_id)
        rec = ctx.state.get("content", content_id)
        if rec is None or not readable(ctx.world, ctx.actor, rec):
            raise NotFound(f"no content {content_id}")
        return rec

    def _check_artifact(self, ctx: Ctx, creator: str, ref: str, kind: str, duration_s) -> dict:
        artifacts.parse_ref(ref)
        if not artifacts.readable(ctx.world, creator, ref) or not artifacts.readable(ctx.world, ctx.actor, ref):
            raise NotFound(f"no artifact {ref}")
        a = ctx.state.tables["artifacts"][ref]
        owner_side = ctx.world.principals_of(ctx.actor) | {creator}
        holder = any(ctx.state.has("artifact_holders", f"{ref}|{p}") for p in owner_side)
        if ctx.actor != SYSTEM and a["owner"] not in owner_side and not holder:
            raise Forbidden("you can only publish artifacts you own")
        fmt = a["kind"]
        allowed = KIND_FORMATS[kind]
        if fmt not in allowed:
            raise InvalidAction(f"a {kind} carries a {' or '.join(allowed)} artifact, not {fmt!r} "
                                f"(the artifact's kind)")
        try:
            facts = F.inspect(artifacts.get(ctx.world, ref), fmt, a["mime"])
        except F.FormatError as e:
            raise InvalidAction(f"artifact {ref[:20]}... is not a valid {fmt}: {e}") from None
        measured = facts.get("duration_s")
        if kind in ("video_short", "video_long", "audio"):
            if measured is None:
                raise InvalidAction(f"a {kind} needs media with a measurable duration")
            if kind == "video_short" and measured > SHORT_MAX_S + DURATION_TOL:
                raise InvalidAction(f"a short video lasts at most {SHORT_MAX_S:g} s (this one {measured:g} s)")
            if kind == "video_long" and measured < LONG_MIN_S - DURATION_TOL:
                raise InvalidAction(f"a long-form video lasts at least {LONG_MIN_S:g} s (this one {measured:g} s)")
        if duration_s is not None:
            if measured is None:
                raise InvalidAction(f"{fmt} artifacts have no duration")
            if abs(float(duration_s) - measured) > max(DURATION_TOL, 0.02 * measured):
                raise InvalidAction(f"stated duration {duration_s} s does not match the measured {measured} s")
        return {"format": fmt, "mime": a["mime"], "size": a["size"], "facts": facts,
                "duration_s": float(measured) if measured is not None else None}

    def _community(self, ctx: Ctx, creator: str, community: str | None, action: str) -> int:
        if community is None:
            return -1
        chk = (ctx.world.hooks.get("content.post_check") or {}).get("community")
        if chk is None:
            raise NotFound(f"no community {community} (this world has no communities)")
        return int(chk(ctx, creator, community, action))

    def _index(self, ctx: Ctx, cid_: str) -> None:
        from ..services import search
        rec = ctx.state.tables["content"][cid_]
        if rec["platform"] == "forum":
            return                       # 4.0 forum posts are indexed by the forum itself (kind post)
        if rec["access"] == "public":
            text = text_of(ctx.world, cid_)
        else:
            text = "\n".join(x for x in (rec["title"], rec.get("description", ""), " ".join(rec["tags"])) if x)
        hosts = search.extract_hosts(ctx.world, text)
        search.index(ctx, cid_, rec["kind"], rec["title"], text, rec["creator"], ref=cid_, hosts=hosts)

    # ------------------------------------------------------------------ internal API
    def publish(self, ctx: Ctx, creator: str, ref: str, kind: str, platform: str, title: str, *, tags=(),
                duration_s=None, description: str = "", community: str | None = None, access: str = "public",
                reply_to: str | None = None) -> str:
        if kind not in KINDS:
            raise InvalidAction(f"kind must be one of {', '.join(KINDS)}")
        p = PLATFORMS.get(platform)
        if p is None:
            raise NotFound(f"no platform {platform!r} (platforms: {', '.join(PLATFORMS)})")
        if p.get("publish") is False:
            raise InvalidAction(f"{platform} carries no public publications")
        if p.get("bridged") and ctx.actor != SYSTEM:
            raise InvalidAction(f"publish on {platform} with its own tools ({p['host']}.post)")
        if kind not in p["kinds"]:
            raise InvalidAction(f"{platform} does not carry {kind} (it carries {', '.join(p['kinds'])})")
        if not p.get("bridged") and not any(e.host == p["host"] for e in ctx.world.endpoints.values()):
            raise NotFound(f"platform {platform} is not part of this world")
        if not isinstance(creator, str) or not ctx.state.has("principals", creator):
            raise NotFound(f"no principal {creator}")
        if not self._creator_side(ctx, creator):
            raise Forbidden(f"you cannot publish as {creator}")
        if access not in ACCESS:
            raise InvalidAction(f"access must be one of {ACCESS}")
        if access == "members" and community is None:
            raise InvalidAction("members-only content belongs to a community")
        if access == "subscribers" and "subscribers" not in (ctx.world.hooks.get("content.access") or {}):
            raise InvalidAction("subscribers-only content needs creator subscriptions (module creators)")
        title = str(title).strip()[:MAX_TITLE]
        if not title:
            raise InvalidAction("a title is required")
        tags = sorted({str(t).lower()[:32] for t in tags if t and re.fullmatch(TAG_RE, str(t).lower())})[:MAX_TAGS]
        if reply_to is not None:
            index_of(reply_to)
            r = ctx.state.get("content", reply_to)
            if r is None or r["removed"] or not readable(ctx.world, creator, r):
                raise NotFound(f"no content {reply_to}")
        a = self._check_artifact(ctx, creator, ref, kind, duration_s)
        comm_n = self._community(ctx, creator, community, "post")
        node = nodes.ensure(ctx, creator)
        n = ctx.state.next_counter("content") - 1
        content_id = cid(n)
        if access == "public":
            artifacts.publish(ctx, ref)
        bus.emit(ctx, "content.published", {
            "id": content_id, "n": n, "ref": ref, "kind": kind, "platform": platform, "creator": creator,
            "creator_node": node, "title": title, "tags": tags, "duration_s": a["duration_s"],
            "description": str(description)[:MAX_DESC], "community": community, "community_n": comm_n,
            "access": access, "format": a["format"], "mime": a["mime"], "size": a["size"],
            "facts": to_jsonable(a["facts"]), "reply_to": reply_to, "by": ctx.actor}, vis="public")
        self._index(ctx, content_id)
        return content_id

    def update(self, ctx: Ctx, content_id: str, ref: str, *, title=None, tags=None, description=None,
               duration_s=None) -> int:
        rec = self._rec(ctx, content_id)
        if rec["removed"]:
            raise Conflict(f"{content_id} was removed")
        if not self._creator_side(ctx, rec["creator"]):
            raise Forbidden("only the creator can update an item")
        if ref == rec["ref"] and title is None and tags is None and description is None:
            raise Conflict("nothing changes")
        a = self._check_artifact(ctx, rec["creator"], ref, rec["kind"], duration_s)
        if rec["access"] == "public":
            artifacts.publish(ctx, ref)
        tags = rec["tags"] if tags is None else \
            sorted({str(t).lower()[:32] for t in tags if t and re.fullmatch(TAG_RE, str(t).lower())})[:MAX_TAGS]
        v = rec["version"] + 1
        bus.emit(ctx, "content.updated", {"id": content_id, "ref": ref, "version": v,
                                          "title": rec["title"] if title is None else str(title).strip()[:MAX_TITLE],
                                          "tags": list(tags),
                                          "description": rec["description"] if description is None
                                          else str(description)[:MAX_DESC],
                                          "duration_s": a["duration_s"], "mime": a["mime"], "size": a["size"],
                                          "format": a["format"], "facts": to_jsonable(a["facts"]), "by": ctx.actor},
                 vis="public")
        self._index(ctx, content_id)
        return v

    def remove(self, ctx: Ctx, content_id: str, reason: str = "", *, role: str | None = None) -> None:
        """Take an item out of listings, feeds and search. `role`: who removes it ("creator" by default when the
        actor is on the creator side; "moderator"/"platform" when a moderation module calls this)."""
        index_of(content_id)
        rec = ctx.state.get("content", content_id)
        if rec is None:
            raise NotFound(f"no content {content_id}")
        if rec["removed"]:
            raise Conflict(f"{content_id} is already removed")
        if role is None:
            if not self._creator_side(ctx, rec["creator"]):
                if not readable(ctx.world, ctx.actor, rec):
                    raise NotFound(f"no content {content_id}")
                raise Forbidden("only the creator (or a moderator) can remove an item")
            role = "creator"
        bus.emit(ctx, "content.removed", {"id": content_id, "by": ctx.actor, "reason": str(reason)[:500],
                                          "role": role}, vis="public")
        from ..services import search
        search.unindex(ctx, content_id)

    def add_comment(self, ctx: Ctx, author: str, content_id: str, text: str, parent: str | None = None) -> str:
        rec = self._rec(ctx, content_id)
        if rec["removed"]:
            raise NotFound(f"no content {content_id}")
        text = str(text).strip()[:MAX_COMMENT]
        if not text:
            raise InvalidAction("an empty comment")
        if parent is not None:
            pc = ctx.state.get("content_comments", parent)
            if pc is None or pc["item"] != content_id or pc["removed"]:
                raise NotFound(f"no comment {parent} on {content_id}")
        if rec["community"] is not None:
            self._community(ctx, author, rec["community"], "comment")
        nodes.ensure(ctx, author)
        mid = ctx.state.new_id("cm")
        vis = "public" if rec["access"] == "public" else tuple(sorted({author, rec["creator"]}))
        bus.emit(ctx, "content.comment", {"id": mid, "item": content_id, "author": author, "text": text,
                                          "parent": parent}, vis=vis)
        return mid

    def remove_comment(self, ctx: Ctx, comment_id: str, reason: str = "", *, role: str | None = None) -> None:
        c = ctx.state.get("content_comments", comment_id)
        if c is None or c["removed"]:
            raise NotFound(f"no comment {comment_id}")
        if role is None:
            rec = ctx.state.tables["content"][c["item"]]
            if c["author"] not in ctx.world.principals_of(ctx.actor) and not self._creator_side(ctx, rec["creator"]):
                raise Forbidden("only the author or the item's creator can remove a comment")
        bus.emit(ctx, "content.comment_removed", {"id": comment_id, "by": ctx.actor, "reason": str(reason)[:300]},
                 vis="public")

    # ------------------------------------------------------------------ forum bridge (4.0 forum -> catalogue)
    def _on_forum_post(self, ctx: Ctx, ev) -> None:
        p = ev.payload
        author = p["author"]
        if not ctx.state.has("principals", author):
            return
        body = f"{p['title']}\n\n{p['body']}".encode("utf-8")
        ref = artifacts.put(ctx, body, "text/plain", owner=author, meta={"title": p["title"][:200], "forum_post": p["id"]},
                            kind="text")
        c = self.publish(ctx, author, ref, "thread", "forum", p["title"] or p["id"], tags=p.get("tags") or ())
        ctx.emit("content.forum_link", {"post": p["id"], "id": c}, vis="internal")

    def _on_forum_delete(self, ctx: Ctx, ev) -> None:
        link = ctx.state.get("content_forum", ev.payload["post"])
        if link is None:
            return
        rec = ctx.state.tables["content"].get(link["id"])
        if rec is not None and not rec["removed"]:
            self.remove(ctx, link["id"], "forum post deleted", role="creator")

    # ------------------------------------------------------------------ appliers
    def _grow(self, n: int) -> None:
        st = self.world.state
        cap = st.arrays[AKEY + "kind"].shape[0]
        if n < cap:
            return
        new = max(cap * 2, n + 1)
        for k, (dt, fill) in ARRAYS.items():
            a = st.arrays[AKEY + k]
            st.set_array(AKEY + k, np.concatenate([a, np.full(new - cap, fill, dtype=a.dtype)]))

    def _set(self, n: int, **vals) -> None:
        st = self.world.state
        rows = np.array([n], dtype=np.int64)
        for k, v in vals.items():
            st.array_rows_mut(AKEY + k, rows)[n] = v

    @applier("content.published")
    def ap_published(self, ev):
        p = ev.payload
        st = self.world.state
        if p["n"] != st.meta_get("content:n", 0):
            raise InvalidAction(f"content index {p['n']} out of order")
        st.insert("content", p["id"], {
            "id": p["id"], "n": p["n"], "ref": p["ref"], "kind": p["kind"], "platform": p["platform"],
            "creator": p["creator"], "creator_node": p["creator_node"], "published_at": ev.t,
            "duration_s": p["duration_s"], "title": p["title"], "tags": list(p["tags"]), "removed": False,
            "description": p.get("description", ""), "community": p.get("community"), "access": p["access"],
            "format": p["format"], "mime": p["mime"], "size": p["size"], "facts": p["facts"],
            "reply_to": p.get("reply_to"), "version": 1, "versions": [{"ref": p["ref"], "t": ev.t}],
            "updated_at": ev.t, "removed_at": None, "removed_by": None, "removal_reason": None, "removal_role": None,
            "comments": 0})
        st.meta_set("content:n", p["n"] + 1)
        self._grow(p["n"])
        self._set(p["n"], kind=KIND_CODE[p["kind"]], platform=PLATFORM_CODE[p["platform"]],
                  creator=p["creator_node"], published_at=ev.t, duration_s=p["duration_s"] or 0.0, removed=False,
                  community=p.get("community_n", -1), access=ACCESS_CODE[p["access"]],
                  format=FORMAT_CODE[p["format"]], version=1,
                  reply_to=index_of(p["reply_to"]) if p.get("reply_to") else -1)

    @applier("content.updated")
    def ap_updated(self, ev):
        p = ev.payload
        st = self.world.state
        rec = st.tables["content"][p["id"]]
        vers = list(rec["versions"]) + [{"ref": p["ref"], "t": ev.t}]
        st.patch("content", p["id"], ref=p["ref"], version=p["version"], versions=vers[-50:],
                 title=p.get("title", rec["title"]), tags=list(p.get("tags", rec["tags"])),
                 description=p.get("description", rec["description"]), duration_s=p.get("duration_s"),
                 mime=p.get("mime", rec["mime"]), size=p.get("size", rec["size"]), facts=p.get("facts", rec["facts"]),
                 updated_at=ev.t)
        self._set(rec["n"], version=p["version"], duration_s=p.get("duration_s") or 0.0)

    @applier("content.removed")
    def ap_removed(self, ev):
        p = ev.payload
        st = self.world.state
        rec = st.tables["content"][p["id"]]
        st.patch("content", p["id"], removed=True, removed_at=ev.t, removed_by=p["by"], removal_reason=p["reason"],
                 removal_role=p["role"])
        self._set(rec["n"], removed=True)

    @applier("content.forum_link")
    def ap_forum_link(self, ev):
        self.world.state.insert("content_forum", ev.payload["post"], {"id": ev.payload["id"]})

    @applier("content.comment")
    def ap_comment(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("content_comments", p["id"], {"item": p["item"], "author": p["author"], "text": p["text"],
                                                "parent": p.get("parent"), "t": ev.t, "removed": False})
        rec = st.tables["content"][p["item"]]
        st.patch("content", p["item"], comments=rec["comments"] + 1)

    @applier("content.comment_removed")
    def ap_comment_removed(self, ev):
        st = self.world.state
        c = st.tables["content_comments"][ev.payload["id"]]
        st.patch("content_comments", ev.payload["id"], removed=True, removed_by=ev.payload["by"])
        rec = st.tables["content"][c["item"]]
        st.patch("content", c["item"], comments=rec["comments"] - 1)

    # ------------------------------------------------------------------ views
    def view(self, world, rec, actor: str) -> dict:
        own = rec["creator"] in world.principals_of(actor)
        d = {"id": rec["id"], "n": rec["n"], "kind": rec["kind"], "platform": rec["platform"], "creator": rec["creator"],
             "title": rec["title"], "tags": list(rec["tags"]), "description": rec["description"],
             "published_at": rec["published_at"], "updated_at": rec["updated_at"], "duration_s": rec["duration_s"],
             "format": rec["format"], "mime": rec["mime"], "size": rec["size"], "access": rec["access"],
             "community": rec["community"], "reply_to": rec["reply_to"], "version": rec["version"],
             "comments": rec["comments"], "removed": rec["removed"], "facts": to_jsonable(rec["facts"])}
        if rec["access"] == "public" or own or readable(world, actor, rec):
            d["ref"] = rec["ref"]
        if rec["removed"]:
            d["removal"] = {"at": rec["removed_at"], "role": rec["removal_role"], "reason": rec["removal_reason"]}
        if own:
            d["versions"] = [dict(v) for v in rec["versions"]]
        return d

    def listing(self, world, actor: str, ids: Iterable[str], limit: int, offset: int = 0, newest=True) -> list[dict]:
        tbl = world.state.tables["content"]
        rows = [tbl[i] for i in ids]
        rows = [r for r in rows if not r["removed"]]
        rows.sort(key=lambda r: (-r["published_at"], -r["n"]) if newest else (r["published_at"], r["n"]))
        return [self.view(world, r, actor) for r in rows[offset:offset + limit]]

    # ------------------------------------------------------------------ endpoints
    @endpoint("publish", S.obj({
        "ref": S.s_str("artifact reference sha256:<hex> (create one with the studio.* tools or artifact.put)",
                       maxLength=71),
        "kind": S.s_enum(list(KINDS), "what it is on the platform"),
        "platform": S.s_enum([p for p, v in PLATFORMS.items() if v.get("publish") is not False and not v.get("bridged")],
                             "where it appears"),
        "title": S.s_str(minLength=1, maxLength=MAX_TITLE), "tags": _TAGS,
        "description": S.s_str("what it is about", maxLength=MAX_DESC),
        "duration_s": S.s_num("optional; must agree with the media's measured duration", minimum=0),
        "community": S.s_str("publish inside a community you may post in", maxLength=40),
        "access": S.s_enum(list(ACCESS), "who can open it (default public)"),
        "reply_to": S.s_str("content id this responds to (a response, remix or follow-up)", maxLength=14),
        "as_org": S.s_str("publish for an organisation you may publish for", maxLength=40)},
        ["ref", "kind", "platform", "title"]), cost_s=30)
    def ep_publish(self, ctx, ref, kind, platform, title, tags=(), description="", duration_s=None, community=None,
                   access="public", reply_to=None, as_org=None):
        """Publish an artifact you own as a catalogue item on a platform. The artifact is inspected (format, size,
        dimensions, duration); public items become readable by everyone and searchable. Returns {id}."""
        creator = self._creator(ctx, as_org)
        c = self.publish(ctx, creator, ref, kind, platform, title, tags=tags, duration_s=duration_s,
                         description=description, community=community, access=access, reply_to=reply_to)
        return {"id": c, "n": index_of(c), "platform": platform, "kind": kind}

    @endpoint("update", S.obj({"id": _ID, "ref": S.s_str("the new version's artifact", maxLength=71),
                               "title": S.s_str(minLength=1, maxLength=MAX_TITLE), "tags": _TAGS,
                               "description": S.s_str(maxLength=MAX_DESC),
                               "duration_s": S.s_num(minimum=0)}, ["id", "ref"]), cost_s=20)
    def ep_update(self, ctx, id, ref, title=None, tags=None, description=None, duration_s=None):
        """Publish a new version of your item (a new artifact; earlier versions stay in its history)."""
        v = self.update(ctx, id, ref, title=title, tags=tags, description=description, duration_s=duration_s)
        return {"id": id, "version": v}

    @endpoint("remove", S.obj({"id": _ID, "reason": S.s_str(maxLength=500)}, ["id"]), cost_s=5)
    def ep_remove(self, ctx, id, reason=""):
        """Remove your item from every listing, feed and search. (A publication cannot be recalled: copies of the
        published bytes stay readable by reference.)"""
        self.remove(ctx, id, reason)
        return {"id": id, "removed": True}

    @endpoint("get", S.obj({"id": _ID}, ["id"]), read_only=True, cost_s=1)
    def ep_get(self, ctx, id):
        """Catalogue record of an item: kind, platform, creator, title, tags, measured facts, version, comments."""
        return self.view(ctx.world, self._rec(ctx, id), ctx.actor)

    @endpoint("open", S.obj({"id": _ID}, ["id"]), read_only=True, cost_s=2)
    def ep_open(self, ctx, id):
        """The item's bytes (base64) and media type - if you may open it (public, your own, or members/subscribers
        content you have access to). You learn the service hosts its text mentions."""
        rec = self._rec(ctx, id)
        if rec["removed"] and not self._creator_side(ctx, rec["creator"]):
            raise NotFound(f"no content {id}")
        data = artifacts.get(ctx.world, rec["ref"])
        from ..services import search
        text = F.extract_text(data, rec["format"], rec["mime"], limit=SEARCH_TEXT)
        for h in search.extract_hosts(ctx.world, text):
            ctx.world.learn(ctx, ctx.actor, "host", h)
        out = self.view(ctx.world, rec, ctx.actor)
        out.update(data_b64=base64.b64encode(data).decode("ascii"), text=text[:SEARCH_TEXT])
        return out

    @endpoint("list", S.obj({"creator": S.s_str(maxLength=60), "platform": S.s_enum(list(PLATFORMS)),
                             "kind": S.s_enum(list(KINDS)), "community": S.s_str(maxLength=40),
                             "tag": S.s_str(pattern=TAG_RE, maxLength=32),
                             "limit": S.s_int(minimum=1, maximum=50, default=20),
                             "offset": S.s_int(minimum=0, maximum=10_000, default=0)}), read_only=True, cost_s=2)
    def ep_list(self, ctx, creator=None, platform=None, kind=None, community=None, tag=None, limit=20, offset=0):
        """Browse the catalogue, newest first, filtered by creator, platform, kind, community or tag."""
        st = ctx.state
        if creator is not None:
            from ..services.identity import ident
            if ids_index(creator) is None:
                creator = ident(ctx.world).resolve_handle(st, creator)
            ids = st.find_ids("content", "creator", creator)
        elif community is not None:
            ids = st.find_ids("content", "community", community)
        elif platform is not None:
            ids = st.find_ids("content", "platform", platform)
        elif kind is not None:
            ids = st.find_ids("content", "kind", kind)
        else:
            ids = st.ids("content")
        tbl = st.tables["content"]
        ids = [i for i in ids if (platform is None or tbl[i]["platform"] == platform)
               and (kind is None or tbl[i]["kind"] == kind) and (community is None or tbl[i]["community"] == community)
               and (tag is None or tag in tbl[i]["tags"]) and readable(ctx.world, ctx.actor, tbl[i])]
        return {"items": self.listing(ctx.world, ctx.actor, ids, limit, offset), "total": len(ids)}

    @endpoint("mine", S.obj({"removed": S.s_bool("include removed items", default=False),
                             "limit": S.s_int(minimum=1, maximum=200, default=50)}), read_only=True, cost_s=1)
    def ep_mine(self, ctx, removed=False, limit=50):
        """Your items (and your organisations'), newest first, with version history."""
        st = ctx.state
        ids = sorted({i for p in ctx.world.principals_of(ctx.actor) for i in st.find_ids("content", "creator", p)})
        tbl = st.tables["content"]
        rows = sorted((tbl[i] for i in ids if removed or not tbl[i]["removed"]),
                      key=lambda r: (-r["published_at"], -r["n"]))
        return {"items": [self.view(ctx.world, r, ctx.actor) for r in rows[:limit]], "total": len(rows)}

    @endpoint("inspect", S.obj({"id": _ID}, ["id"]), read_only=True, cost_s=2)
    def ep_inspect(self, ctx, id):
        """Observable performance of your item: its catalogue record and history, comments, and the audience
        analytics the platform reports to creators (delayed and noisy, as on real platforms), plus your
        followers and subscribers."""
        rec = self._rec(ctx, id)
        if not self._creator_side(ctx, rec["creator"]):
            raise Forbidden("only the creator can inspect an item's performance")
        out = {"item": self.view(ctx.world, rec, ctx.actor), "comments": rec["comments"]}
        w = ctx.world
        try:
            from . import audience as _aud          # workstream C2
            out["audience"] = _aud.audience(w, rec["n"], principal=ctx.actor)
        except ImportError:
            out["audience"] = None
            out["audience_note"] = "audience analytics are reported by the consumption engine (not in this world)"
        hook = w.hooks.get("social.creator_stats")
        if hook is not None:
            out["creator"] = hook(w, rec["creator"])
        return out

    @endpoint("comment", S.obj({"id": _ID, "text": S.s_str(minLength=1, maxLength=MAX_COMMENT),
                                "parent": S.s_str("comment id to reply to", maxLength=20)}, ["id", "text"]), cost_s=20)
    def ep_comment(self, ctx, id, text, parent=None):
        """Comment on an item you can open (or reply to a comment on it)."""
        return {"comment": self.add_comment(ctx, ctx.actor, id, text, parent)}

    @endpoint("comments", S.obj({"id": _ID, "limit": S.s_int(minimum=1, maximum=100, default=30)}, ["id"]),
              read_only=True, cost_s=2)
    def ep_comments(self, ctx, id, limit=30):
        """Comments on an item, oldest first."""
        self._rec(ctx, id)
        st = ctx.state
        rows = sorted(((cid_, st.tables["content_comments"][cid_]) for cid_ in st.find_ids("content_comments", "item", id)),
                      key=lambda x: (x[1]["t"], x[0]))
        return [{"comment": k, "author": c["author"], "text": c["text"], "parent": c["parent"], "t": c["t"]}
                for k, c in rows if not c["removed"]][:limit]

    @endpoint("delete_comment", S.obj({"comment": S.s_str(maxLength=20), "reason": S.s_str(maxLength=300)},
                                      ["comment"]), cost_s=2)
    def ep_delete_comment(self, ctx, comment, reason=""):
        """Delete your comment, or a comment on your item."""
        self.remove_comment(ctx, comment, reason)
        return {"comment": comment, "removed": True}

    @endpoint("kinds", S.obj(), read_only=True, cost_s=0)
    def ep_kinds(self, ctx):
        """Catalogue kinds, the artifact formats each carries, and the platforms that carry them."""
        return {"kinds": {k: {"formats": list(KIND_FORMATS[k]),
                              "platforms": [p for p, v in PLATFORMS.items() if k in v["kinds"]]} for k in KINDS},
                "formats": F.formats_doc(), "short_video_max_s": SHORT_MAX_S, "long_video_min_s": LONG_MIN_S}

    @endpoint("platforms", S.obj(), read_only=True, cost_s=0)
    def ep_platforms(self, ctx):
        """The platforms of this world: host, what they carry, how things are discovered there."""
        out = []
        for name, p in PLATFORMS.items():
            present = p.get("bridged") and "forum" in ctx.world.modules or \
                any(e.host == p["host"] for e in ctx.world.endpoints.values())
            out.append({"platform": name, "host": p["host"], "label": p["label"], "kinds": list(p["kinds"]),
                        "discovery": list(p["discovery"]), "doc": p["doc"], "present": bool(present)})
        return out

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        st = world.state
        n = n_items(world)
        if len(st.tables["content"]) != n:
            errs.append(f"{len(st.tables['content'])} catalogue rows but content:n = {n}")
        arr = catalogue(world)
        comments: dict[str, int] = {}
        for k, c in st.tables["content_comments"].items():
            if not c["removed"]:
                comments[c["item"]] = comments.get(c["item"], 0) + 1
            if c["item"] not in st.tables["content"]:
                errs.append(f"comment {k} on a missing item")
        for k, r in st.tables["content"].items():
            i = r["n"]
            if k != cid(i) or i >= n:
                errs.append(f"item {k} has index {i}")
                continue
            if not artifacts.exists(world, r["ref"]):
                errs.append(f"item {k} references a missing artifact")
            if r["kind"] not in KINDS or r["format"] not in KIND_FORMATS[r["kind"]]:
                errs.append(f"item {k}: format {r['format']} is not allowed for {r['kind']}")
            if arr["removed"][i] != r["removed"] or arr["kind"][i] != KIND_CODE[r["kind"]] or \
                    arr["published_at"][i] != r["published_at"] or arr["version"][i] != r["version"] or \
                    arr["creator"][i] != r["creator_node"]:
                errs.append(f"item {k}: dense arrays disagree with the record")
            if r["comments"] != comments.get(k, 0):
                errs.append(f"item {k}: comment count {r['comments']} != {comments.get(k, 0)}")
            if r["access"] == "public" and not st.tables["artifacts"][r["ref"]]["published"]:
                errs.append(f"public item {k} has an unpublished artifact")
        return errs

    def observe(self, world, actor):
        st = world.state
        ids = st.find_ids("content", "creator", actor)
        if not ids:
            return None
        tbl = st.tables["content"]
        live = [tbl[i] for i in ids if not tbl[i]["removed"]]
        return {"items": len(live), "removed": len(ids) - len(live), "comments_received": sum(r["comments"] for r in live)}
