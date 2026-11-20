"""Customer support (host `support`): the feedback channel between customers and service operators.

A ticket is filed against a service host owned by an organisation (or person) by a customer, which is either a
principal or an opaque customer reference of the simulated population (e.g. `seg:12`). Tickets come from real
service failures: other modules (cloud, market, consumers) call the internal API
`open_ticket(ctx, host, owner, customer, subject, body, severity=1) -> ticket_id`; agents file them with
`support.open` (the host's owner comes from `world.hooks["host_owner"](state, host)`).

Status: `open` (waiting for the operator) -> `answered` (operator replied; waiting for the customer) ->
`resolved`. A customer reply re-opens a ticket. Staff are principals with `support.reply` on the owner
(organisation members hold it). `resolve` calls `world.hooks["on_ticket_resolved"](ctx, ticket_id)` (the
economy reads satisfaction from it). A daily process (`support.daily`) emits SLA metrics per owner (open and
answered counts, median/max age of unresolved tickets, tickets opened/resolved and mean first-response time
over the day) as an internal `support.stats` event, stored in `sup_stats` and returned by `support.stats`.

Ticket contents are visible only to the owner and the customer (if a principal).

Invariants: per-owner live counters equal a recount; resolved tickets (and only those) have a resolution time;
first response never precedes opening; message counters match.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from .identity import ident
from .search import HOST_NAME

DAY = 86400
STATUSES = ("open", "answered", "resolved")
MAX_BODY = 10_000
TODAY = ("opened_today", "resolved_today", "first_resp_n", "first_resp_sum")


@register_module
class Support(Module):
    name = "support"
    host = "support"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("sup_tickets", index=("owner", "customer", "status"))
        st.declare_table("sup_msgs", index=("ticket",))
        st.declare_table("sup_owners")
        st.declare_table("sup_stats", index=("owner",))

    def genesis(self, ctx):
        ctx.schedule(DAY, "support.daily", {}, prio=20, period=DAY, key="support.daily")

    # ------------------------------------------------------------------ internal API
    def open_ticket(self, ctx: Ctx, host: str, owner: str, customer: str, subject: str, body: str,
                    severity: int = 1) -> str:
        """File a ticket (no authorisation check: callers decide who may complain about what)."""
        if not 1 <= int(severity) <= 4:
            raise InvalidAction("severity must be 1..4")
        tid = ctx.state.new_id("tkt")
        ctx.emit("support.open", {"id": tid, "host": host, "owner": owner, "customer": customer,
                                  "subject": str(subject)[:200], "body": str(body)[:MAX_BODY],
                                  "severity": int(severity)}, vis=self._parties(ctx.state, owner, customer))
        return tid

    def resolve(self, ctx: Ctx, ticket_id: str, note: str = "") -> None:
        """Resolve a ticket (no authorisation check) and fire the on_ticket_resolved hook."""
        t = ctx.state.get("sup_tickets", ticket_id)
        if t is None:
            raise NotFound(f"no ticket {ticket_id}")
        if t["status"] == "resolved":
            raise Conflict("already resolved")
        ctx.emit("support.status", {"id": ticket_id, "status": "resolved", "by": ctx.actor, "note": note[:500]},
                 vis=self._parties(ctx.state, t["owner"], t["customer"]))
        hook = ctx.world.hooks.get("on_ticket_resolved")
        if hook is not None:
            hook(ctx, ticket_id)

    def _parties(self, state, owner: str, customer: str) -> tuple:
        return tuple(sorted({owner, customer} if state.has("principals", customer) else {owner}))

    def _role(self, ctx, t) -> str | None:
        """'staff', 'customer' or None for the acting principal on ticket record t."""
        if t["customer"] != ctx.actor and ident(ctx.world).allowed(ctx, ctx.actor, t["owner"], "support.reply"):
            return "staff"
        if t["customer"] == ctx.actor or t["customer"] in ctx.world.principals_of(ctx.actor):
            return "customer"
        return None

    def _ticket(self, ctx, tid: str):
        t = ctx.state.get("sup_tickets", tid)
        role = self._role(ctx, t) if t is not None else None
        if role is None:
            raise NotFound(f"no ticket {tid}")
        return t, role

    # ------------------------------------------------------------------ endpoints
    @endpoint("open", S.obj({"host": S.s_str("service host the ticket is about", pattern=HOST_NAME, maxLength=41),
                             "subject": S.s_str(minLength=1, maxLength=200),
                             "body": S.s_str(maxLength=MAX_BODY),
                             "severity": S.s_int("1 (question) .. 4 (outage)", minimum=1, maximum=4, default=1)},
                            ["host", "subject"]), cost_s=30)
    def ep_open(self, ctx, host, subject, body="", severity=1):
        """Open a support ticket with the operator of a service."""
        fn = ctx.world.hooks.get("host_owner")
        owner = fn(ctx.state, host) if fn else None
        if not owner:
            raise NotFound(f"no operated service at host {host}")
        if owner == ctx.actor:
            raise InvalidAction("you operate this service")
        return {"ticket": self.open_ticket(ctx, host, owner, ctx.actor, subject, body, severity)}

    @endpoint("list", S.obj({"owner": S.s_str("operator view: tickets of an organisation you support",
                                              maxLength=40),
                             "status": S.s_enum(list(STATUSES) + ["unresolved", "any"]),
                             "limit": S.s_int(minimum=1, maximum=50, default=20)}), read_only=True)
    def ep_list(self, ctx, owner=None, status="unresolved", limit=20):
        """Your tickets as a customer, or (with owner) an operator's queue: most severe and oldest first."""
        st = ctx.state
        if owner:
            own = ident(ctx.world).resolve_handle(st, owner)
            ident(ctx.world).authorize(ctx, own, "support.reply")
            ids = st.find_ids("sup_tickets", "owner", own)
        else:
            ids = sorted({i for c in sorted(ctx.world.principals_of(ctx.actor))
                          for i in st.find_ids("sup_tickets", "customer", c)})
        rows = []
        for tid in ids:
            t = st.tables["sup_tickets"][tid]
            if status == "unresolved" and t["status"] == "resolved":
                continue
            if status in STATUSES and t["status"] != status:
                continue
            rows.append((-t["severity"], t["opened"], tid, t))
        rows.sort(key=lambda x: x[:3])
        return [_summary(tid, t) for _, _, tid, t in rows[:limit]]

    @endpoint("get", S.obj({"ticket": S.s_str(maxLength=40)}, ["ticket"]), read_only=True)
    def ep_get(self, ctx, ticket):
        """A ticket with its full conversation."""
        t, role = self._ticket(ctx, ticket)
        st = ctx.state
        msgs = [st.tables["sup_msgs"][m] for m in st.find_ids("sup_msgs", "ticket", ticket)]
        return dict(_summary(ticket, t), body=t["body"], your_role=role,
                    messages=[{"author": m["author"], "role": m["role"], "body": m["body"], "t": m["t"]}
                              for m in msgs])

    @endpoint("reply", S.obj({"ticket": S.s_str(maxLength=40), "body": S.s_str(minLength=1, maxLength=MAX_BODY)},
                             ["ticket", "body"]), cost_s=30)
    def ep_reply(self, ctx, ticket, body):
        """Reply on a ticket: as operator staff (support.reply) it becomes 'answered'; as the customer it is
        (re)opened."""
        t, role = self._ticket(ctx, ticket)
        new_status = "answered" if role == "staff" else "open"
        n = t["msgs"] + 1
        ctx.emit("support.msg", {"ticket": ticket, "n": n, "author": ctx.actor, "role": role, "body": body,
                                 "status": new_status},
                 vis={*self._parties(ctx.state, t["owner"], t["customer"]), ctx.actor})
        return {"ticket": ticket, "status": new_status}

    @endpoint("resolve", S.obj({"ticket": S.s_str(maxLength=40), "note": S.s_str(maxLength=500)}, ["ticket"]),
              cost_s=10)
    def ep_resolve(self, ctx, ticket, note=""):
        """Mark a ticket resolved (operator staff with support.reply)."""
        t, role = self._ticket(ctx, ticket)
        if role != "staff":
            raise Forbidden("only the operator's staff resolve tickets")
        self.resolve(ctx, ticket, note)
        return {"ticket": ticket, "status": "resolved"}

    @endpoint("stats", S.obj({"owner": S.s_str("service operator", maxLength=40),
                              "days": S.s_int(minimum=1, maximum=30, default=7)}, ["owner"]), read_only=True)
    def ep_stats(self, ctx, owner, days=7):
        """Public service-level metrics of an operator: live queue and daily history (newest first)."""
        own = ident(ctx.world).resolve_handle(ctx.state, owner)
        st = ctx.state
        live = st.get("sup_owners", own)
        rows = [st.tables["sup_stats"][s] for s in st.find_ids("sup_stats", "owner", own)][-days:]
        return {"owner": own,
                "live": {k: (live[k] if live else 0) for k in ("open", "answered", "resolved", "total")},
                "daily": [dict(r) for r in reversed(rows)]}

    def stats(self, state, owner: str) -> dict | None:
        """Latest stored daily SLA row for an owner (for other modules)."""
        ids = state.find_ids("sup_stats", "owner", owner)
        return dict(state.tables["sup_stats"][ids[-1]]) if ids else None

    # ------------------------------------------------------------------ daily SLA metrics
    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        now = ctx.now
        ages: dict[str, list[int]] = {}
        for status in ("open", "answered"):
            for tid in st.find_ids("sup_tickets", "status", status):
                t = st.tables["sup_tickets"][tid]
                ages.setdefault(t["owner"], []).append(now - t["opened"])
        day = now // DAY
        for owner, o in st.scan("sup_owners"):
            a = sorted(ages.get(owner, []))
            if not a and not any(o[k] for k in TODAY):
                continue
            n = len(a)
            median = 0 if not n else (a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) // 2)
            ctx.emit("support.stats", {
                "owner": owner, "day": day, "open": o["open"], "answered": o["answered"], "unresolved": n,
                "median_age_s": median, "max_age_s": a[-1] if a else 0, "opened": o["opened_today"],
                "resolved": o["resolved_today"],
                "mean_first_response_s": (o["first_resp_sum"] // o["first_resp_n"]) if o["first_resp_n"] else None},
                vis="internal")

    # ------------------------------------------------------------------ appliers
    def _owner_patch(self, owner: str, **delta: int) -> None:
        st = self.world.state
        o = st.tables["sup_owners"].get(owner)
        base = dict(o) if o else {"open": 0, "answered": 0, "resolved": 0, "total": 0,
                                  **{k: 0 for k in TODAY}}
        for k, d in delta.items():
            base[k] = base[k] + d
        st.put("sup_owners", owner, base)

    @applier("support.open")
    def ap_open(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("sup_tickets", p["id"], {"host": p["host"], "owner": p["owner"], "customer": p["customer"],
                                           "subject": p["subject"], "body": p["body"], "severity": p["severity"],
                                           "status": "open", "opened": ev.t, "first_response": None,
                                           "resolved": None, "msgs": 0, "last_activity": ev.t})
        self._owner_patch(p["owner"], open=1, total=1, opened_today=1)

    def _transition(self, tid: str, t, new: str, t_now: int, **extra) -> None:
        st = self.world.state
        old = t["status"]
        changes = dict(extra, status=new, last_activity=t_now)
        if new == "resolved":
            changes["resolved"] = t_now
        elif old == "resolved":
            changes["resolved"] = None
        st.patch("sup_tickets", tid, **changes)
        delta = {}
        if old != new:
            delta[old] = -1
            delta[new] = 1
        if new == "resolved" and old != "resolved":
            delta["resolved_today"] = 1
        if delta:
            self._owner_patch(t["owner"], **delta)

    @applier("support.msg")
    def ap_msg(self, ev):
        p = ev.payload
        st = self.world.state
        t = st.tables["sup_tickets"][p["ticket"]]
        st.insert("sup_msgs", f"{p['ticket']}|{p['n']:05d}", {"ticket": p["ticket"], "author": p["author"],
                                                              "role": p["role"], "body": p["body"], "t": ev.t})
        extra = {"msgs": t["msgs"] + 1}
        if p["role"] == "staff" and t["first_response"] is None:
            extra["first_response"] = ev.t
            self._owner_patch(t["owner"], first_resp_n=1, first_resp_sum=ev.t - t["opened"])
        self._transition(p["ticket"], t, p["status"], ev.t, **extra)

    @applier("support.status")
    def ap_status(self, ev):
        p = ev.payload
        t = self.world.state.tables["sup_tickets"][p["id"]]
        self._transition(p["id"], t, p["status"], ev.t)

    @applier("support.stats")
    def ap_stats(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("sup_stats", f"{p['owner']}|{p['day']:06d}", dict(p, t=ev.t))
        o = st.tables["sup_owners"][p["owner"]]
        st.put("sup_owners", p["owner"], {**o, **{k: 0 for k in TODAY}})

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        counts: dict[str, dict[str, int]] = {}
        msgs: dict[str, int] = {}
        for mid, m in st.tables["sup_msgs"].items():
            msgs[m["ticket"]] = msgs.get(m["ticket"], 0) + 1
        for tid, t in st.tables["sup_tickets"].items():
            if t["status"] not in STATUSES:
                errs.append(f"ticket {tid} has bad status {t['status']}")
                continue
            c = counts.setdefault(t["owner"], {"open": 0, "answered": 0, "resolved": 0, "total": 0})
            c[t["status"]] += 1
            c["total"] += 1
            if (t["status"] == "resolved") != (t["resolved"] is not None):
                errs.append(f"ticket {tid} status/resolution time mismatch")
            if t["first_response"] is not None and t["first_response"] < t["opened"]:
                errs.append(f"ticket {tid} answered before it was opened")
            if t["msgs"] != msgs.get(tid, 0):
                errs.append(f"ticket {tid} message counter {t['msgs']} != {msgs.get(tid, 0)}")
        for owner in sorted(set(counts) | set(st.tables["sup_owners"])):
            o = st.tables["sup_owners"].get(owner) or {}
            c = counts.get(owner, {"open": 0, "answered": 0, "resolved": 0, "total": 0})
            if any(o.get(k, 0) != v for k, v in c.items()):
                errs.append(f"owner {owner} counters {dict(o)} != recount {c}")
        return errs

    def observe(self, world, actor):
        st = world.state
        queue = 0
        orgs = [m["org"] for m in st.find("members", "person", actor)] if "members" in st.tables else []
        for o in sorted(set(orgs) | {actor}):
            r = st.tables["sup_owners"].get(o)
            if r:
                queue += r["open"]
        waiting = sum(1 for t in st.find("sup_tickets", "customer", actor) if t["status"] == "answered")
        return {"open_tickets": queue, "answered_for_you": waiting} if queue or waiting else None


def _summary(tid: str, t) -> dict:
    return {"ticket": tid, "host": t["host"], "owner": t["owner"], "customer": t["customer"],
            "subject": t["subject"], "severity": t["severity"], "status": t["status"], "opened": t["opened"],
            "first_response": t["first_response"], "resolved": t["resolved"], "messages": t["msgs"]}
