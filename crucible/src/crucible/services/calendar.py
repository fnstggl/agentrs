"""Calendar (host `cal`): meetings and coordination in simulated time.

An event has an organiser (a person, or an organisation with `cal.write`), a creator, a start time, a duration,
a title and attendees (persons). The creator attends automatically (accepted); every other attendee receives
an invitation that needs a response (`respond`), and `world.hooks["on_calendar_invite"](ctx, event_id,
attendee)` is called so that NPC actors can react. Accepting an event that overlaps another accepted event of
the same attendee fails with `conflict` (listing the overlapping events) unless `allow_conflict` is set; the
agenda marks overlaps. Events are visible only to the organiser and the attendees.

Module name and host are both `cal` (`calendar` is registered as an alias module name).

Invariants: invitations reference existing events, responses are valid, the creator is an attendee, every
attendee has exactly one invitation, durations are positive.
"""
from __future__ import annotations

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import MODULES, Ctx, Module, applier, endpoint, register_module
from .identity import ident

DAY = 86400
MAX_ATTENDEES = 20
MAX_DURATION = 14 * DAY
MAX_WINDOW = 90 * DAY
RESPONSES = ("pending", "accepted", "declined")


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


@register_module
class Calendar(Module):
    name = "cal"
    host = "cal"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("cal_events", index=("organiser",))
        st.declare_table("cal_invites", index=("attendee", "event"))

    # ------------------------------------------------------------------ queries
    def conflicts(self, state, attendee: str, start: int, end: int, exclude: str | None = None) -> list[str]:
        """Active events the attendee has accepted that overlap [start, end)."""
        out = []
        for iid in state.find_ids("cal_invites", "attendee", attendee):
            inv = state.tables["cal_invites"][iid]
            if inv["response"] != "accepted" or inv["event"] == exclude:
                continue
            ev = state.tables["cal_events"][inv["event"]]
            if ev["status"] == "active" and _overlap(start, end, ev["start"], ev["start"] + ev["duration"]):
                out.append(inv["event"])
        return out

    def _event(self, ctx, eid: str):
        ev = ctx.state.get("cal_events", eid)
        if ev is None:
            raise NotFound(f"no event {eid}")
        return ev

    def _party(self, ctx, eid: str, ev) -> bool:
        if ctx.state.has("cal_invites", f"{eid}|{ctx.actor}"):
            return True
        return ev["organiser"] in ctx.world.principals_of(ctx.actor)

    # ------------------------------------------------------------------ endpoints
    @endpoint("create", S.obj({"title": S.s_str(minLength=1, maxLength=200),
                               "start": S.s_int("absolute simulated time in seconds (>= now)", minimum=0),
                               "duration": S.s_int("seconds", minimum=60, maximum=MAX_DURATION),
                               "attendees": S.s_arr(S.s_str("person or handle", maxLength=60), maxItems=MAX_ATTENDEES),
                               "description": S.s_str(maxLength=2000),
                               "org": S.s_str("organise on behalf of an organisation (needs cal.write)", maxLength=40),
                               "allow_conflict": S.s_bool("create even if it overlaps your accepted events",
                                                          default=False)},
                              ["title", "start", "duration"]), cost_s=20)
    def ep_create(self, ctx, title, start, duration, attendees=(), description="", org=None, allow_conflict=False):
        """Schedule an event and invite attendees (each must accept or decline). You attend automatically."""
        st = ctx.state
        idm = ident(ctx.world)
        if not ctx.actor.startswith("p:"):
            raise Forbidden("only persons create calendar events")
        if start < ctx.now:
            raise InvalidAction("start is in the past")
        organiser = ctx.actor
        if org:
            organiser = idm.resolve_handle(st, org)
            if not organiser.startswith("o:"):
                raise InvalidAction("org must be an organisation")
            idm.authorize(ctx, organiser, "cal.write")
        others: list[str] = []
        for a in attendees:
            pid = idm.resolve_handle(st, a)
            if not pid.startswith("p:"):
                raise InvalidAction(f"attendees must be persons ({pid})")
            if pid != ctx.actor and pid not in others:
                others.append(pid)
        clash = self.conflicts(st, ctx.actor, start, start + duration)
        if clash and not allow_conflict:
            raise Conflict("overlaps events you accepted", conflicts=clash)
        eid = st.new_id("cev")
        ctx.emit("cal.create", {"id": eid, "organiser": organiser, "creator": ctx.actor, "title": title,
                                "description": description, "start": start, "duration": duration,
                                "attendees": [ctx.actor] + others},
                 vis={organiser, ctx.actor, *others})
        hook = ctx.world.hooks.get("on_calendar_invite")
        if hook is not None:
            for a in others:
                hook(ctx, eid, a)
        return {"event": eid, "invited": others, "conflicts": clash}

    @endpoint("respond", S.obj({"event": S.s_str(maxLength=40), "accept": S.s_bool(),
                                "allow_conflict": S.s_bool(default=False)}, ["event", "accept"]), cost_s=5)
    def ep_respond(self, ctx, event, accept, allow_conflict=False):
        """Accept or decline an invitation. Accepting fails if it overlaps another event you accepted, unless
        allow_conflict is true."""
        ev = self._event(ctx, event)
        inv = ctx.state.get("cal_invites", f"{event}|{ctx.actor}")
        if inv is None:
            raise NotFound(f"no invitation to {event}")
        if ev["status"] != "active":
            raise InvalidAction("event was cancelled")
        response = "accepted" if accept else "declined"
        if inv["response"] == response:
            raise Conflict(f"already {response}")
        clash = []
        if accept:
            clash = self.conflicts(ctx.state, ctx.actor, ev["start"], ev["start"] + ev["duration"], exclude=event)
            if clash and not allow_conflict:
                raise Conflict("overlaps events you accepted", conflicts=clash)
        ctx.emit("cal.respond", {"event": event, "attendee": ctx.actor, "response": response},
                 vis={ev["organiser"], ev["creator"], ctx.actor})
        return {"event": event, "response": response, "conflicts": clash}

    @endpoint("cancel", S.obj({"event": S.s_str(maxLength=40), "reason": S.s_str(maxLength=300)}, ["event"]),
              cost_s=5)
    def ep_cancel(self, ctx, event, reason=""):
        """Cancel an event you created (or that your organisation organises, with cal.write)."""
        ev = self._event(ctx, event)
        if ctx.actor != ev["creator"]:
            ident(ctx.world).authorize(ctx, ev["organiser"], "cal.write")
        if ev["status"] != "active":
            raise Conflict("already cancelled")
        ctx.emit("cal.cancel", {"event": event, "reason": reason, "by": ctx.actor},
                 vis={ev["organiser"], *ev["attendees"]})
        return {"ok": True}

    @endpoint("agenda", S.obj({"from_t": S.s_int("window start (default now)", minimum=0),
                               "to_t": S.s_int("window end (default now + 7 days)", minimum=0),
                               "include_declined": S.s_bool(default=False)}), read_only=True)
    def ep_agenda(self, ctx, from_t=None, to_t=None, include_declined=False):
        """Your events in a time window (at most 90 days), with your response and overlaps with accepted ones."""
        st = ctx.state
        t0 = ctx.now if from_t is None else from_t
        t1 = t0 + 7 * DAY if to_t is None else to_t
        if t1 <= t0 or t1 - t0 > MAX_WINDOW:
            raise InvalidAction("window must be positive and at most 90 days")
        rows = []
        for iid in st.find_ids("cal_invites", "attendee", ctx.actor):
            inv = st.tables["cal_invites"][iid]
            ev = st.tables["cal_events"][inv["event"]]
            if ev["status"] != "active" or not _overlap(t0, t1, ev["start"], ev["start"] + ev["duration"]):
                continue
            if inv["response"] == "declined" and not include_declined:
                continue
            rows.append((ev["start"], inv["event"], ev, inv["response"]))
        rows.sort(key=lambda x: (x[0], x[1]))
        out = []
        for start, eid, ev, resp in rows:
            clash = self.conflicts(st, ctx.actor, start, start + ev["duration"], exclude=eid)
            out.append({"event": eid, "title": ev["title"], "start": start, "duration": ev["duration"],
                        "organiser": ev["organiser"], "response": resp, "conflicts": clash})
        return out

    @endpoint("get", S.obj({"event": S.s_str(maxLength=40)}, ["event"]), read_only=True)
    def ep_get(self, ctx, event):
        """Event details with every attendee's response (attendees and organisers only)."""
        ev = self._event(ctx, event)
        if not self._party(ctx, event, ev):
            raise NotFound(f"no event {event}")
        resp = {a: ctx.state.tables["cal_invites"][f"{event}|{a}"]["response"] for a in ev["attendees"]}
        return {"event": event, "title": ev["title"], "description": ev["description"], "start": ev["start"],
                "duration": ev["duration"], "organiser": ev["organiser"], "creator": ev["creator"],
                "status": ev["status"], "responses": resp}

    # ------------------------------------------------------------------ appliers
    @applier("cal.create")
    def ap_create(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("cal_events", p["id"], {"organiser": p["organiser"], "creator": p["creator"], "title": p["title"],
                                          "description": p["description"], "start": p["start"],
                                          "duration": p["duration"], "attendees": list(p["attendees"]),
                                          "status": "active", "created": ev.t})
        for a in p["attendees"]:
            st.insert("cal_invites", f"{p['id']}|{a}", {"event": p["id"], "attendee": a,
                                                         "response": "accepted" if a == p["creator"] else "pending",
                                                         "t": ev.t})

    @applier("cal.respond")
    def ap_respond(self, ev):
        p = ev.payload
        self.world.state.patch("cal_invites", f"{p['event']}|{p['attendee']}", response=p["response"], t=ev.t)

    @applier("cal.cancel")
    def ap_cancel(self, ev):
        self.world.state.patch("cal_events", ev.payload["event"], status="cancelled")

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        errs = []
        for iid, inv in st.tables["cal_invites"].items():
            ev = st.tables["cal_events"].get(inv["event"])
            if ev is None:
                errs.append(f"invite {iid} for a missing event")
            elif inv["attendee"] not in ev["attendees"]:
                errs.append(f"invite {iid} for a non-attendee")
            if inv["response"] not in RESPONSES:
                errs.append(f"invite {iid} has bad response {inv['response']}")
        for eid, ev in st.tables["cal_events"].items():
            if ev["duration"] <= 0:
                errs.append(f"event {eid} has non-positive duration")
            if ev["creator"] not in ev["attendees"]:
                errs.append(f"event {eid} creator is not an attendee")
            for a in ev["attendees"]:
                if f"{eid}|{a}" not in st.tables["cal_invites"]:
                    errs.append(f"event {eid} attendee {a} has no invitation")
        return errs

    def observe(self, world, actor):
        st = world.state
        now = world.now
        pending, nxt = 0, None
        for iid in st.find_ids("cal_invites", "attendee", actor):
            inv = st.tables["cal_invites"][iid]
            ev = st.tables["cal_events"][inv["event"]]
            if ev["status"] != "active" or ev["start"] + ev["duration"] <= now:
                continue
            if inv["response"] == "pending":
                pending += 1
            elif inv["response"] == "accepted" and ev["start"] < now + DAY:
                key = (ev["start"], inv["event"])
                if nxt is None or key < nxt[0]:
                    nxt = (key, {"event": inv["event"], "title": ev["title"], "start": ev["start"]})
        if not pending and nxt is None:
            return None
        return {"pending_invites": pending, "next_24h": nxt[1] if nxt else None}


MODULES.setdefault("calendar", Calendar)
