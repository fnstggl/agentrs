"""Messaging (kernel module `msg`, host `msg`): private direct and group conversations.

A conversation has members (principals); every message is visible to the members of the conversation at the time
it is sent and to nobody else (event visibility), so messaging is a private channel: what people share there is
discovered only by those they share it with. A message may carry a catalogue item (the recipients must be able to
open it) or an artifact of the sender (the recipients are granted read access to it).

Rules: a direct conversation is found or created by `msg.send` to one person; groups are created with members and
managed by their admins; people can block senders (a blocked sender's messages are refused); each member's read
position is kept, so unread counts are exact; at most `max_per_day` messages per sender per day (spam bound).

Population individuals share items with each other through `crucible.social.graph.share_batch(..., to=...)` (the
direct-share relation); agents and tool-using population members use these endpoints.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound, RateLimited
from ..core.kernel import Ctx, Module, applier, endpoint, register_module
from ..substrate import artifacts, bus, ids
from . import content as C
from . import nodes

DAY = 86400
MAX_TEXT = 10_000
MAX_GROUP = 256

_S = {"type": "string"}
bus.event_type("msg.conversation", S.obj({"id": _S, "kind": {"type": "string", "enum": ["dm", "group"]},
                                          "title": _S, "members": {"type": "array", "items": _S}, "by": _S},
                                         ["id", "kind", "members", "by"]), owner="msg", doc="a conversation started")
bus.event_type("msg.message", S.obj({"id": _S, "conversation": _S, "sender": _S, "text": _S,
                                     "item": {"type": ["string", "null"]}, "ref": {"type": ["string", "null"]},
                                     "reply_to": {"type": ["string", "null"]}},
                                    ["id", "conversation", "sender", "text"]), owner="msg",
               doc="a message in a conversation (visible to its members only)")
bus.event_type("msg.members", S.obj({"conversation": _S, "add": {"type": "array", "items": _S},
                                     "remove": {"type": "array", "items": _S}, "by": _S},
                                    ["conversation", "add", "remove", "by"]), owner="msg",
               doc="people joined or left a group conversation")
bus.event_type("msg.read", S.obj({"conversation": _S, "member": _S, "upto": {"type": "integer"}},
                                 ["conversation", "member", "upto"]), owner="msg", doc="a member read up to a message")
bus.event_type("msg.block", S.obj({"by": _S, "who": _S, "blocked": {"type": "boolean"}}, ["by", "who", "blocked"]),
               owner="msg", doc="someone blocked or unblocked a sender")

_WHO = S.s_str("a principal (p:<handle>, p<i>, o:<handle>) or a handle", maxLength=60)


@register_module
class Messaging(Module):
    name = "msg"
    host = "msg"
    version = "1"
    depends = ("identity", "content", "social_ids")
    DEFAULTS = {"max_per_day": 500}

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(self.DEFAULTS, **self.config)

    def install(self, world):
        st = world.state
        st.declare_table("msg_conversations", index=("dm_key",))
        st.declare_table("msg_members", index=("member", "conversation"))
        st.declare_table("msg_messages", index=("conversation", "sender"))
        st.declare_table("msg_blocks", index=("by",))
        st.declare_table("msg_daily")
        bus.enforce(world, owner="msg")

    # ------------------------------------------------------------------ helpers
    def _who(self, ctx: Ctx, who: str) -> str:
        from ..services.identity import ident
        if ids.index(who) is not None:
            if not ctx.state.has("principals", who):
                raise NotFound(f"no principal {who}")
            return who
        return ident(ctx.world).resolve_handle(ctx.state, who)

    def _members(self, st, conv: str) -> list[str]:
        return sorted(st.tables["msg_members"][k]["member"] for k in st.find_ids("msg_members", "conversation", conv)
                      if st.tables["msg_members"][k]["active"])

    def _conv(self, ctx: Ctx, conv: str) -> dict:
        c = ctx.state.get("msg_conversations", conv)
        if c is None or ctx.actor not in self._members(ctx.state, conv):
            raise NotFound(f"no conversation {conv}")
        return dict(c)

    def _blocked(self, st, by: str, who: str) -> bool:
        b = st.get("msg_blocks", f"{by}|{who}")
        return bool(b and b["blocked"])

    def dm(self, ctx: Ctx, a: str, b: str) -> str:
        key = "|".join(sorted((a, b)))
        found = ctx.state.find_ids("msg_conversations", "dm_key", key)
        if found:
            return found[0]
        cid = ctx.state.new_id("conv")
        nodes.ensure(ctx, a)
        nodes.ensure(ctx, b)
        bus.emit(ctx, "msg.conversation", {"id": cid, "kind": "dm", "title": "", "members": sorted((a, b)), "by": a},
                 vis=tuple(sorted((a, b))))
        return cid

    def post(self, ctx: Ctx, sender: str, conv: str, text: str, item: str | None = None, ref: str | None = None,
             reply_to: str | None = None) -> str:
        st = ctx.state
        mem = self._members(st, conv)
        if sender not in mem:
            raise NotFound(f"no conversation {conv}")
        text = str(text)[:MAX_TEXT]
        if not text.strip() and item is None and ref is None:
            raise InvalidAction("an empty message")
        others = [m for m in mem if m != sender]
        for m in others:
            if self._blocked(st, m, sender):
                raise Forbidden(f"{m} does not accept your messages")
        day = ctx.now // DAY
        used = st.get("msg_daily", f"{sender}|{day}")
        if used and used["n"] >= int(self.cfg["max_per_day"]):
            raise RateLimited(f"at most {self.cfg['max_per_day']} messages per day")
        if item is not None:
            rec = C.module(ctx.world)._rec(ctx, item)
            if rec["removed"]:
                raise NotFound(f"no content {item}")
            for m in others:
                if not C.readable(ctx.world, m, rec):
                    raise Forbidden(f"{m} cannot open {item}")
        if ref is not None:
            artifacts.parse_ref(ref)
            a = st.get("artifacts", ref)
            if a is None or a["owner"] not in ctx.world.principals_of(sender):
                raise NotFound(f"no artifact {ref} of yours")
            for m in others:
                if not artifacts.readable(ctx.world, m, ref):
                    ctx.call("artifact.grant", {"ref": ref, "grantee": m}, as_principal=sender)
        if reply_to is not None:
            r = st.get("msg_messages", reply_to)
            if r is None or r["conversation"] != conv:
                raise NotFound(f"no message {reply_to} here")
        mid = st.new_id("msg")
        bus.emit(ctx, "msg.message", {"id": mid, "conversation": conv, "sender": sender, "text": text, "item": item,
                                      "ref": ref, "reply_to": reply_to}, vis=tuple(mem))
        return mid

    def send_share(self, ctx: Ctx, sender: str, recipients: list[str], item: str, note: str = "") -> list[str]:
        """Deliver a shared catalogue item to each recipient in the direct conversation (social.share with `to`)."""
        out = []
        for p in recipients:
            conv = self.dm(ctx, sender, p)
            out.append(self.post(ctx, sender, conv, note or "", item=item))
        return out

    # ------------------------------------------------------------------ endpoints
    @endpoint("send", S.obj({"to": _WHO, "conversation": S.s_str("an existing conversation", maxLength=20),
                             "text": S.s_str(maxLength=MAX_TEXT),
                             "item": S.s_str("a catalogue item to share (c<n>)", pattern=C.CID_RE.pattern, maxLength=14),
                             "ref": S.s_str("an artifact of yours to attach", maxLength=71),
                             "reply_to": S.s_str(maxLength=20)}), cost_s=20)
    def ep_send(self, ctx, to=None, conversation=None, text="", item=None, ref=None, reply_to=None):
        """Send a private message to a person (direct conversation) or into a conversation you are in, optionally
        with a catalogue item or an artifact of yours (recipients get read access to it)."""
        if (to is None) == (conversation is None):
            raise InvalidAction("give either to or conversation")
        if to is not None:
            p = self._who(ctx, to)
            if p == ctx.actor:
                raise InvalidAction("you cannot message yourself")
            conversation = self.dm(ctx, ctx.actor, p)
        else:
            self._conv(ctx, conversation)
        mid = self.post(ctx, ctx.actor, conversation, text, item, ref, reply_to)
        return {"conversation": conversation, "message": mid}

    @endpoint("group", S.obj({"members": S.s_arr(_WHO, minItems=1, maxItems=MAX_GROUP - 1),
                              "title": S.s_str(minLength=1, maxLength=120)}, ["members", "title"]), cost_s=10)
    def ep_group(self, ctx, members, title):
        """Start a group conversation (you are its admin)."""
        ps = sorted({self._who(ctx, m) for m in members} | {ctx.actor})
        if len(ps) < 2:
            raise InvalidAction("a group needs someone besides you")
        for p in ps:
            if p != ctx.actor and self._blocked(ctx.state, p, ctx.actor):
                raise Forbidden(f"{p} does not accept your messages")
            nodes.ensure(ctx, p)
        cid = ctx.state.new_id("conv")
        bus.emit(ctx, "msg.conversation", {"id": cid, "kind": "group", "title": str(title)[:120], "members": ps,
                                           "by": ctx.actor}, vis=tuple(ps))
        return {"conversation": cid, "members": ps}

    @endpoint("add", S.obj({"conversation": S.s_str(maxLength=20), "member": _WHO}, ["conversation", "member"]),
              cost_s=5)
    def ep_add(self, ctx, conversation, member):
        """Group admin: add someone."""
        c = self._conv(ctx, conversation)
        if c["kind"] != "group" or c["admin"] != ctx.actor:
            raise Forbidden("only the group's admin adds members")
        p = self._who(ctx, member)
        mem = self._members(ctx.state, conversation)
        if p in mem:
            raise Conflict(f"{p} is in the group")
        if len(mem) >= MAX_GROUP:
            raise Conflict("the group is full")
        if self._blocked(ctx.state, p, ctx.actor):
            raise Forbidden(f"{p} does not accept your messages")
        nodes.ensure(ctx, p)
        bus.emit(ctx, "msg.members", {"conversation": conversation, "add": [p], "remove": [], "by": ctx.actor},
                 vis=tuple(sorted(set(mem) | {p})))
        return {"conversation": conversation, "added": p}

    @endpoint("leave", S.obj({"conversation": S.s_str(maxLength=20)}, ["conversation"]), cost_s=2)
    def ep_leave(self, ctx, conversation):
        """Leave a group conversation."""
        c = self._conv(ctx, conversation)
        if c["kind"] != "group":
            raise InvalidAction("direct conversations cannot be left; block the person instead")
        mem = self._members(ctx.state, conversation)
        bus.emit(ctx, "msg.members", {"conversation": conversation, "add": [], "remove": [ctx.actor],
                                      "by": ctx.actor}, vis=tuple(mem))
        return {"conversation": conversation, "left": True}

    @endpoint("conversations", S.obj({"limit": S.s_int(minimum=1, maximum=200, default=50)}), read_only=True,
              cost_s=1)
    def ep_conversations(self, ctx, limit=50):
        """Your conversations, most recent first, with unread counts."""
        st = ctx.state
        out = []
        for k in st.find_ids("msg_members", "member", ctx.actor):
            m = st.tables["msg_members"][k]
            if not m["active"]:
                continue
            c = st.tables["msg_conversations"][m["conversation"]]
            out.append({"conversation": m["conversation"], "kind": c["kind"], "title": c["title"],
                        "members": self._members(st, m["conversation"]), "last_t": c["last_t"],
                        "messages": c["n"], "unread": c["n"] - m["read"]})
        out.sort(key=lambda r: (-r["last_t"], r["conversation"]))
        return out[:limit]

    def history(self, ctx: Ctx, conversation: str, limit: int) -> dict:
        c = self._conv(ctx, conversation)
        st = ctx.state
        rows = sorted((st.tables["msg_messages"][k] | {"message": k}
                       for k in st.find_ids("msg_messages", "conversation", conversation)),
                      key=lambda r: (r["k"], r["message"]))
        mine = st.tables["msg_members"][f"{conversation}|{ctx.actor}"]
        rows = [r for r in rows if r["k"] > mine["since"]]
        return {"conversation": conversation, "kind": c["kind"], "title": c["title"], "n": c["n"],
                "unread": c["n"] - mine["read"], "members": self._members(st, conversation),
                "messages": [{"message": r["message"], "sender": r["sender"], "text": r["text"], "item": r["item"],
                              "ref": r["ref"], "reply_to": r["reply_to"], "t": r["t"]} for r in rows[-limit:]]}

    @endpoint("history", S.obj({"conversation": S.s_str(maxLength=20),
                                "limit": S.s_int(minimum=1, maximum=200, default=50)}, ["conversation"]),
              read_only=True, cost_s=2)
    def ep_history(self, ctx, conversation, limit=50):
        """The latest messages of a conversation, without marking them read."""
        return self.history(ctx, conversation, limit)

    @endpoint("read", S.obj({"conversation": S.s_str(maxLength=20),
                             "limit": S.s_int(minimum=1, maximum=200, default=50)}, ["conversation"]), cost_s=5)
    def ep_read(self, ctx, conversation, limit=50):
        """Read the latest messages of a conversation (marks them read)."""
        c = self._conv(ctx, conversation)
        st = ctx.state
        rows = sorted((st.tables["msg_messages"][k] | {"message": k}
                       for k in st.find_ids("msg_messages", "conversation", conversation)),
                      key=lambda r: (r["k"], r["message"]))
        mine = st.tables["msg_members"][f"{conversation}|{ctx.actor}"]
        rows = [r for r in rows if r["k"] > mine["since"]]
        if c["n"] > mine["read"]:
            bus.emit(ctx, "msg.read", {"conversation": conversation, "member": ctx.actor, "upto": c["n"]},
                     vis=(ctx.actor,))
        return {"conversation": conversation, "kind": c["kind"], "title": c["title"],
                "members": self._members(st, conversation),
                "messages": [{"message": r["message"], "sender": r["sender"], "text": r["text"], "item": r["item"],
                              "ref": r["ref"], "reply_to": r["reply_to"], "t": r["t"]} for r in rows[-limit:]]}

    @endpoint("block", S.obj({"who": _WHO, "blocked": S.s_bool(default=True)}, ["who"]), cost_s=2)
    def ep_block(self, ctx, who, blocked=True):
        """Refuse (or accept again) messages from someone."""
        p = self._who(ctx, who)
        if p == ctx.actor:
            raise InvalidAction("you cannot block yourself")
        bus.emit(ctx, "msg.block", {"by": ctx.actor, "who": p, "blocked": bool(blocked)}, vis=(ctx.actor,))
        return {"who": p, "blocked": bool(blocked)}

    # ------------------------------------------------------------------ appliers
    @applier("msg.conversation")
    def ap_conversation(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("msg_conversations", p["id"], {"kind": p["kind"], "title": p.get("title", ""), "admin": p["by"],
                                                 "dm_key": "|".join(p["members"]) if p["kind"] == "dm" else None,
                                                 "created": ev.t, "last_t": ev.t, "n": 0})
        for m in p["members"]:
            st.insert("msg_members", f"{p['id']}|{m}", {"conversation": p["id"], "member": m, "active": True,
                                                        "read": 0, "since": 0, "t": ev.t})

    @applier("msg.message")
    def ap_message(self, ev):
        p = ev.payload
        st = self.world.state
        c = st.tables["msg_conversations"][p["conversation"]]
        k = c["n"] + 1
        st.insert("msg_messages", p["id"], {"conversation": p["conversation"], "sender": p["sender"], "text": p["text"],
                                            "item": p.get("item"), "ref": p.get("ref"),
                                            "reply_to": p.get("reply_to"), "t": ev.t, "k": k})
        st.patch("msg_conversations", p["conversation"], n=k, last_t=ev.t)
        st.patch("msg_members", f"{p['conversation']}|{p['sender']}", read=k)
        day = ev.t // DAY
        key = f"{p['sender']}|{day}"
        old = st.tables["msg_daily"].get(key)
        st.put("msg_daily", key, {"n": (old["n"] if old else 0) + 1})

    @applier("msg.members")
    def ap_members(self, ev):
        p = ev.payload
        st = self.world.state
        c = st.tables["msg_conversations"][p["conversation"]]
        for m in p["add"]:
            st.put("msg_members", f"{p['conversation']}|{m}", {"conversation": p["conversation"], "member": m,
                                                              "active": True, "read": c["n"], "since": c["n"],
                                                              "t": ev.t})
        for m in p["remove"]:
            st.patch("msg_members", f"{p['conversation']}|{m}", active=False)

    @applier("msg.read")
    def ap_read(self, ev):
        p = ev.payload
        self.world.state.patch("msg_members", f"{p['conversation']}|{p['member']}", read=p["upto"])

    @applier("msg.block")
    def ap_block(self, ev):
        p = ev.payload
        self.world.state.put("msg_blocks", f"{p['by']}|{p['who']}", {"by": p["by"], "who": p["who"],
                                                                    "blocked": p["blocked"], "t": ev.t})

    def check_invariants(self, world):
        errs = []
        st = world.state
        counts: dict[str, int] = {}
        for k, m in st.tables["msg_messages"].items():
            counts[m["conversation"]] = counts.get(m["conversation"], 0) + 1
        for cid, c in st.tables["msg_conversations"].items():
            if c["n"] != counts.get(cid, 0):
                errs.append(f"conversation {cid} counts {c['n']} messages, {counts.get(cid, 0)} recorded")
        for k, m in st.tables["msg_members"].items():
            c = st.tables["msg_conversations"].get(m["conversation"])
            if c is None or m["read"] > c["n"]:
                errs.append(f"member record {k} reads beyond its conversation")
        return errs

    def observe(self, world, actor):
        st = world.state
        unread = 0
        for k in st.find_ids("msg_members", "member", actor):
            m = st.tables["msg_members"][k]
            if m["active"]:
                unread += st.tables["msg_conversations"][m["conversation"]]["n"] - m["read"]
        return {"unread_messages": unread} if unread else None
