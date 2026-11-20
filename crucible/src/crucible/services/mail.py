"""Mail (host `mail`): private, asynchronous messaging between principals.

Sending records the message (`mail.sent`, visible to the sender only) and schedules one delivery job per
recipient after a latency drawn from `world.spec.params["mail_latency_s"]` (default 60 s, uniform jitter
x0.5-1.5; the drawn delays are in the event payload). At delivery (`mail.deliver` process) the message bounces
if the recipient principal is inactive or a dissolved organisation, or with probability
`params["mail_failure_p"]` (default 0); otherwise a mailbox entry appears (`mail.delivered`, visible to the
recipient only, sender/subject but no body) and `world.hooks["on_mail_delivered"](ctx, msg_id, recipient)` is
called (NPC actors wake up on it). A bounce puts an "Undeliverable" notice from the mail daemon in the sender's
inbox.

Organisation recipients get one org mailbox; owners/admins see it (via `principals_of`), others need a
`mail.read` grant. Sending as an organisation needs `mail.send` on it. Reading a message makes the reader learn
the service hosts it mentions.

Internal API: `send(ctx, sender, to, subject, body, in_reply_to=None) -> msg_id` (no authorisation check).

Invariants: every pending recipient has its delivery job; delivery/bounce never precedes sending; mailbox
entries reference existing messages; per-mailbox unread counters equal a recount.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident
from .search import extract_hosts, tokenize

MAX_RCPT = 20
MAX_BODY = 20_000
DAEMON = "sys"


def _job_key(msg: str, to: str) -> str:
    return f"mail|{msg}|{to}"


@register_module
class Mail(Module):
    name = "mail"
    host = "mail"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("mail_msgs", index=("from",))
        st.declare_table("mail_box", index=("owner", "msg"))
        st.declare_table("mail_unread")

    # ------------------------------------------------------------------ internal API
    def send(self, ctx: Ctx, sender: str, to, subject: str, body: str, in_reply_to: str | None = None) -> str:
        st = ctx.state
        idm = ident(ctx.world)
        rcpts: list[str] = []
        for x in to:
            pid = idm.resolve_handle(st, str(x))
            if pid not in rcpts:
                rcpts.append(pid)
        if not rcpts:
            raise InvalidAction("no recipients")
        if len(rcpts) > MAX_RCPT:
            raise InvalidAction(f"at most {MAX_RCPT} recipients")
        subject, body = str(subject)[:200], str(body)[:MAX_BODY]
        mid = st.new_id("mail")
        base = int(ctx.world.spec.params.get("mail_latency_s", 60))
        delays = {}
        rng = ctx.rng("mail.latency", mid)
        for r in rcpts:
            delays[r] = 0 if base <= 0 else max(1, int(round(base * (0.5 + float(rng.random())))))
        hosts = extract_hosts(ctx.world, subject + "\n" + body)
        ctx.emit("mail.sent", {"id": mid, "from": sender, "to": rcpts, "subject": subject, "body": body,
                               "hosts": hosts, "in_reply_to": in_reply_to, "delays": delays, "by": ctx.actor},
                 vis={sender, ctx.actor})
        for r in rcpts:
            ctx.schedule(ctx.now + delays[r], "mail.deliver", {"msg": mid, "to": r}, prio=40, key=_job_key(mid, r))
        return mid

    def unread(self, state, owner: str) -> int:
        r = state.get("mail_unread", owner)
        return r["n"] if r else 0

    # ------------------------------------------------------------------ helpers
    def _owners(self, ctx, mailbox: str | None) -> list[str]:
        if mailbox:
            pid = ident(ctx.world).resolve_handle(ctx.state, mailbox)
            if pid not in ctx.world.principals_of(ctx.actor):
                ident(ctx.world).authorize(ctx, pid, "mail.read")
            return [pid]
        return sorted(ctx.world.principals_of(ctx.actor))

    def _entry(self, ctx, msg: str, mailbox: str | None):
        """The mailbox entry through which the actor may see `msg` (inbox copies first, then sent copies)."""
        owners = self._owners(ctx, mailbox)
        for folder in ("inbox", "sent"):
            for o in owners:
                eid = f"{msg}|{o}|{folder}"
                e = ctx.state.get("mail_box", eid)
                if e is not None:
                    return eid, e
        raise NotFound(f"no message {msg} in your mailbox")

    # ------------------------------------------------------------------ endpoints
    @endpoint("send", S.obj({"to": S.s_arr(S.s_str("principal or handle", maxLength=60), "recipients", minItems=1,
                                           maxItems=MAX_RCPT),
                             "subject": S.s_str(maxLength=200), "body": S.s_str(maxLength=MAX_BODY),
                             "in_reply_to": S.s_str("message id you are answering", maxLength=40),
                             "from_org": S.s_str("send from an organisation's mailbox (needs mail.send)",
                                                 maxLength=40)},
                            ["to", "subject", "body"]), cost_s=30)
    def ep_send(self, ctx, to, subject, body, in_reply_to=None, from_org=None):
        """Send a private message. Delivery takes simulated time (about a minute) and can bounce; you get an
        'Undeliverable' notice if it does. Mention services as <host>.crucible to introduce them."""
        sender = ctx.actor
        if from_org:
            sender = ident(ctx.world).resolve_handle(ctx.state, from_org)
            ident(ctx.world).authorize(ctx, sender, "mail.send")
        if in_reply_to is not None:
            self._entry(ctx, in_reply_to, None)
        mid = self.send(ctx, sender, to, subject, body, in_reply_to)
        m = ctx.state.tables["mail_msgs"][mid]
        return {"msg": mid, "to": list(m["to"]), "eta": {r: m["t"] + d for r, d in sorted(m["delays"].items())}}

    @endpoint("inbox", S.obj({"folder": S.s_enum(["inbox", "sent"]), "unread_only": S.s_bool(default=False),
                              "limit": S.s_int(minimum=1, maximum=50, default=20),
                              "mailbox": S.s_str("an organisation mailbox you administer or hold mail.read on",
                                                 maxLength=40)}), read_only=True)
    def ep_inbox(self, ctx, folder="inbox", unread_only=False, limit=20, mailbox=None):
        """List messages (newest first): id, mailbox, sender, recipients, subject, time, read flag."""
        st = ctx.state
        rows = []
        for o in self._owners(ctx, mailbox):
            for eid in st.find_ids("mail_box", "owner", o):
                e = st.tables["mail_box"][eid]
                if e["folder"] != folder or (unread_only and e["read"]):
                    continue
                rows.append((-e["t"], eid, e))
        rows.sort(key=lambda x: (x[0], x[1]))
        out = []
        for _, eid, e in rows[:limit]:
            m = st.tables["mail_msgs"][e["msg"]]
            out.append({"msg": e["msg"], "mailbox": e["owner"], "from": m["from"], "to": list(m["to"]),
                        "subject": m["subject"], "t": e["t"], "read": e["read"], "kind": m.get("kind", "mail")})
        return out

    @endpoint("read", S.obj({"msg": S.s_str(maxLength=40), "mailbox": S.s_str(maxLength=40)}, ["msg"]), cost_s=5)
    def ep_read(self, ctx, msg, mailbox=None):
        """Open a message (marks it read). You learn the service hosts it mentions."""
        eid, e = self._entry(ctx, msg, mailbox)
        m = ctx.state.tables["mail_msgs"][msg]
        if e["folder"] == "inbox" and not e["read"]:
            ctx.emit("mail.read", {"id": eid}, vis=(e["owner"],))
        for h in m["hosts"]:
            ctx.world.learn(ctx, ctx.actor, "host", h)
        out = {"msg": msg, "from": m["from"], "to": list(m["to"]), "subject": m["subject"], "body": m["body"],
               "t": m["t"], "in_reply_to": m["in_reply_to"], "hosts": list(m["hosts"]), "mailbox": e["owner"],
               "folder": e["folder"], "kind": m.get("kind", "mail")}
        if e["folder"] == "sent":
            out["delivery"] = {r: {"status": s, "t": m["dtime"].get(r)} for r, s in sorted(m["status"].items())}
        else:
            out["delivered"] = e["t"]
        return out

    @endpoint("search", S.obj({"q": S.s_str(maxLength=200), "limit": S.s_int(minimum=1, maximum=20, default=10)},
                              ["q"]), cost_s=3, read_only=True)
    def ep_search(self, ctx, q, limit=10):
        """Search your own mail (all words must appear in subject or body), newest first."""
        st = ctx.state
        want = set(tokenize(q))
        rows = []
        for o in sorted(ctx.world.principals_of(ctx.actor)):
            for eid in st.find_ids("mail_box", "owner", o):
                e = st.tables["mail_box"][eid]
                m = st.tables["mail_msgs"][e["msg"]]
                if want and not want <= set(tokenize(m["subject"] + " " + m["body"])):
                    continue
                rows.append((-e["t"], eid, e, m))
        rows.sort(key=lambda x: (x[0], x[1]))
        return [{"msg": e["msg"], "mailbox": e["owner"], "folder": e["folder"], "from": m["from"],
                 "subject": m["subject"], "t": e["t"], "snippet": m["body"][:160]} for _, _, e, m in rows[:limit]]

    @endpoint("delete", S.obj({"msg": S.s_str(maxLength=40), "folder": S.s_enum(["inbox", "sent"]),
                               "mailbox": S.s_str(maxLength=40)}, ["msg"]), cost_s=2)
    def ep_delete(self, ctx, msg, folder="inbox", mailbox=None):
        """Delete a message from one of your folders."""
        for o in self._owners(ctx, mailbox):
            eid = f"{msg}|{o}|{folder}"
            if ctx.state.has("mail_box", eid):
                ctx.emit("mail.delete", {"id": eid}, vis=(o,))
                return {"ok": True}
        raise NotFound(f"no message {msg} in your {folder}")

    # ------------------------------------------------------------------ delivery process
    @process("deliver")
    def pr_deliver(self, ctx, msg, to):
        st = ctx.state
        m = st.get("mail_msgs", msg)
        if m is None or m["status"].get(to) != "pending":
            return
        p = st.get("principals", to)
        reason = None
        if p is None or p.get("status") != "active":
            reason = "recipient_inactive"
        elif to.startswith("o:") and (st.get("orgs", to) or {}).get("status", "active") != "active":
            reason = "organisation_dissolved"
        else:
            fp = float(ctx.world.spec.params.get("mail_failure_p", 0.0))
            if fp > 0 and float(ctx.rng("mail.fail", msg, to).random()) < fp:
                reason = "transient_failure"
        if reason is None:
            ctx.emit("mail.delivered", {"msg": msg, "to": to, "from": m["from"], "subject": m["subject"],
                                        "delay": ctx.now - m["t"]}, vis=(to,))
            hook = ctx.world.hooks.get("on_mail_delivered")
            if hook is not None:
                hook(ctx, msg, to)
        else:
            notice = None
            if m["from"] != SYSTEM:
                notice = st.new_id("mail")
            ctx.emit("mail.bounced", {"msg": msg, "to": to, "reason": reason, "notice": notice}, vis=(m["from"],))

    # ------------------------------------------------------------------ appliers
    def _bump(self, owner: str, d: int) -> None:
        st = self.world.state
        r = st.tables["mail_unread"].get(owner)
        st.put("mail_unread", owner, {"n": (r["n"] if r else 0) + d})

    @applier("mail.sent")
    def ap_sent(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("mail_msgs", p["id"], {"from": p["from"], "to": list(p["to"]), "subject": p["subject"],
                                         "body": p["body"], "hosts": list(p["hosts"]), "in_reply_to": p["in_reply_to"],
                                         "t": ev.t, "delays": dict(p["delays"]),
                                         "status": {r: "pending" for r in p["to"]}, "dtime": {}, "by": p["by"],
                                         "kind": "mail"})
        if p["from"] != SYSTEM:
            st.insert("mail_box", f"{p['id']}|{p['from']}|sent", {"msg": p["id"], "owner": p["from"],
                                                                  "folder": "sent", "read": True, "t": ev.t})

    @applier("mail.delivered")
    def ap_delivered(self, ev):
        p = ev.payload
        st = self.world.state
        m = st.tables["mail_msgs"][p["msg"]]
        st.patch("mail_msgs", p["msg"], status={**m["status"], p["to"]: "delivered"},
                 dtime={**m["dtime"], p["to"]: ev.t})
        st.insert("mail_box", f"{p['msg']}|{p['to']}|inbox", {"msg": p["msg"], "owner": p["to"], "folder": "inbox",
                                                              "read": False, "t": ev.t})
        self._bump(p["to"], 1)

    @applier("mail.bounced")
    def ap_bounced(self, ev):
        p = ev.payload
        st = self.world.state
        m = st.tables["mail_msgs"][p["msg"]]
        st.patch("mail_msgs", p["msg"], status={**m["status"], p["to"]: "bounced"},
                 dtime={**m["dtime"], p["to"]: ev.t})
        if p["notice"]:
            sender = m["from"]
            st.insert("mail_msgs", p["notice"], {
                "from": DAEMON, "to": [sender], "subject": f"Undeliverable: {m['subject']}"[:200],
                "body": f"Your message {p['msg']} to {p['to']} could not be delivered ({p['reason']}).",
                "hosts": [], "in_reply_to": p["msg"], "t": ev.t, "delays": {sender: 0},
                "status": {sender: "delivered"}, "dtime": {sender: ev.t}, "by": DAEMON, "kind": "bounce"})
            st.insert("mail_box", f"{p['notice']}|{sender}|inbox", {"msg": p["notice"], "owner": sender,
                                                                    "folder": "inbox", "read": False, "t": ev.t})
            self._bump(sender, 1)

    @applier("mail.read")
    def ap_read(self, ev):
        st = self.world.state
        e = st.tables["mail_box"][ev.payload["id"]]
        st.patch("mail_box", ev.payload["id"], read=True)
        self._bump(e["owner"], -1)

    @applier("mail.delete")
    def ap_delete(self, ev):
        st = self.world.state
        e = st.tables["mail_box"][ev.payload["id"]]
        if e["folder"] == "inbox" and not e["read"]:
            self._bump(e["owner"], -1)
        st.delete("mail_box", ev.payload["id"])

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        for mid, m in st.tables["mail_msgs"].items():
            for r, s in m["status"].items():
                if s == "pending":
                    if not st.has("sched", _job_key(mid, r)):
                        errs.append(f"{mid} to {r} pending without a delivery job")
                elif s in ("delivered", "bounced"):
                    if m["dtime"].get(r) is None or m["dtime"][r] < m["t"]:
                        errs.append(f"{mid} to {r} {s} before it was sent")
                else:
                    errs.append(f"{mid} to {r} has bad status {s}")
        unread: dict[str, int] = {}
        for eid, e in st.tables["mail_box"].items():
            m = st.tables["mail_msgs"].get(e["msg"])
            if m is None:
                errs.append(f"mailbox entry {eid} references a missing message")
                continue
            if e["t"] < m["t"]:
                errs.append(f"mailbox entry {eid} predates its message")
            if e["folder"] == "inbox" and m["status"].get(e["owner"]) != "delivered":
                errs.append(f"mailbox entry {eid} without a delivery")
            if e["folder"] == "inbox" and not e["read"]:
                unread[e["owner"]] = unread.get(e["owner"], 0) + 1
        for owner in sorted(set(unread) | set(st.tables["mail_unread"])):
            if unread.get(owner, 0) != self.unread(st, owner):
                errs.append(f"unread counter of {owner} is {self.unread(st, owner)}, recount {unread.get(owner, 0)}")
        return errs

    def observe(self, world, actor):
        if not actor.startswith("p:"):
            return None
        st = world.state
        n = sum(self.unread(st, o) for o in sorted(world.principals_of(actor)))
        return {"unread": n}
