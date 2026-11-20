"""Identity, organisations, roles, delegation (grants) and authorisation.

Principals: persons (`p:<handle>`), organisations (`o:<handle>`), service principals of deployed software
(`svc:<host>`), and the system. Every other module asks `authorize(ctx, owner, cap, amount)` before touching a
resource owned by someone else. The rules:

1. A principal may do anything with its own resources.
2. Owners and admins of an organisation hold every capability over the organisation's resources.
3. Members hold the capabilities in `MEMBER_CAPS` over their organisation's resources.
4. Anyone else needs an explicit grant (delegation) from an owner/admin: capability, scope, optional spending
   limit per rolling 30-day window, optional expiry. Spending against a limit is recorded.
5. A service principal acts with the grants its operating organisation gave it - nothing more.
"""
from __future__ import annotations

import hashlib
import re

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module

ROLES = ("owner", "admin", "member")
MEMBER_CAPS = {"code.read", "code.write", "docs.read", "docs.write", "issues.write", "support.reply", "cal.write",
               "cloud.read", "analytics.read"}
HANDLE = r"[a-z][a-z0-9-]{1,30}"
WINDOW = 30 * 86400


@register_module
class Identity(Module):
    name = "identity"
    host = "id"

    def install(self, world):
        st = world.state
        st.declare_table("persons", index=("handle", "kind"))
        st.declare_table("orgs", index=("handle", "status"))
        st.declare_table("members", index=("org", "person"))
        st.declare_table("invites", index=("person", "org"))
        st.declare_table("grants", index=("grantee", "owner"))
        st.declare_table("credentials")
        world.hooks["principals_of"] = principals_of
        world.hooks["may_act_as"] = may_act_as

    # ------------------------------------------------------------------ internal API (used by other modules)
    def create_person(self, ctx: Ctx, handle: str, name: str, kind: str, info: dict | None = None,
                      password: str | None = None) -> str:
        if not re.fullmatch(HANDLE, handle):
            raise InvalidAction(f"bad handle {handle!r}")
        pid = f"p:{handle}"
        if ctx.state.has("principals", pid) or ctx.state.find_ids("orgs", "handle", handle):
            raise Conflict(f"handle {handle} taken")
        ctx.emit("principal.add", {"id": pid, "kind": "person", "info": {"name": name}}, vis="internal")
        ctx.emit("id.person", {"id": pid, "handle": handle, "name": name, "kind": kind, "info": info or {}}, vis="public")
        if password:
            salt = ctx.world.rng_for("salt", pid).bytes(8).hex()
            ctx.emit("id.credential", {"principal": pid, "salt": salt,
                                       "hash": hashlib.sha256((salt + password).encode()).hexdigest()}, vis="internal")
        return pid

    def create_org(self, ctx: Ctx, handle: str, name: str, founder: str | None, kind: str = "group",
                   info: dict | None = None) -> str:
        if not re.fullmatch(HANDLE, handle):
            raise InvalidAction(f"bad handle {handle!r}")
        oid = f"o:{handle}"
        if ctx.state.has("principals", oid) or ctx.state.find_ids("persons", "handle", handle):
            raise Conflict(f"handle {handle} taken")
        ctx.emit("principal.add", {"id": oid, "kind": "org", "info": {"name": name}}, vis="internal")
        ctx.emit("id.org", {"id": oid, "handle": handle, "name": name, "kind": kind, "founder": founder,
                            "info": info or {}}, vis="public")
        if founder:
            ctx.emit("id.member", {"org": oid, "person": founder, "role": "owner", "title": "founder"},
                     vis=(founder, oid))
        return oid

    def check_password(self, world, pid: str, password: str) -> bool:
        c = world.state.get("credentials", pid)
        return bool(c) and hashlib.sha256((c["salt"] + password).encode()).hexdigest() == c["hash"]

    def role(self, state, org: str, person: str) -> str | None:
        m = state.get("members", f"{org}|{person}")
        return m["role"] if m else None

    def org_owner_persons(self, state, org: str) -> list[str]:
        return [m["person"] for m in state.find("members", "org", org) if m["role"] == "owner"]

    def authorize(self, ctx: Ctx, owner: str, cap: str, amount: int = 0, scope: str | None = None) -> None:
        if not self.allowed(ctx, ctx.actor, owner, cap, amount, scope, record=True):
            raise Forbidden(f"{ctx.actor} lacks {cap} on {owner}" + (f" ({scope})" if scope else ""))

    def allowed(self, ctx: Ctx, actor: str, owner: str, cap: str, amount: int = 0, scope: str | None = None,
                record: bool = False) -> bool:
        st = ctx.state
        if actor == owner or actor == SYSTEM:
            return True
        o = st.get("orgs", owner)
        if o is not None and o["status"] == "active":
            r = self.role(st, owner, actor)
            if r in ("owner", "admin") or (r == "member" and cap in MEMBER_CAPS):
                return True
        now = ctx.now
        for gid in st.find_ids("grants", "grantee", actor):
            g = st.tables["grants"][gid]
            if g["owner"] != owner or g["status"] != "active" or not _cap_match(g["cap"], cap):
                continue
            if g["scope"] not in ("*", None) and scope is not None and g["scope"] != scope:
                continue
            if g["expires"] is not None and g["expires"] <= now:
                continue
            if amount and g["limit"] is not None:
                used = g["used"] if now - g["window_start"] < WINDOW else 0
                if used + amount > g["limit"]:
                    continue
                if record:
                    ctx.emit("id.grant_used", {"id": gid, "amount": amount}, vis=(owner, actor))
            return True
        return False

    # ------------------------------------------------------------------ endpoints
    @endpoint("profile", S.obj({"principal": S.s_str("p:<handle>, o:<handle> or a bare handle", maxLength=80)},
                               ["principal"]), read_only=True)
    def ep_profile(self, ctx, principal):
        """Public profile of a person or organisation: name, kind, members (for orgs), age, reputation."""
        pid = self.resolve_handle(ctx.state, principal)
        if pid.startswith("p:"):
            p = ctx.state.get("persons", pid)
            orgs = [m["org"] for m in ctx.state.find("members", "person", pid)]
            rep = ctx.world.hooks.get("reputation", lambda w, x: None)(ctx.world, pid)
            return {"id": pid, "handle": p["handle"], "name": p["name"], "since": p["created"], "orgs": orgs,
                    "bio": p.get("bio", ""), "reputation": rep}
        o = ctx.state.get("orgs", pid)
        members = [{"person": m["person"], "role": m["role"], "title": m["title"]}
                   for m in ctx.state.find("members", "org", pid)]
        rep = ctx.world.hooks.get("reputation", lambda w, x: None)(ctx.world, pid)
        return {"id": pid, "handle": o["handle"], "name": o["name"], "kind": o["kind"], "status": o["status"],
                "since": o["created"], "members": members, "bio": o.get("bio", ""), "reputation": rep}

    def resolve_handle(self, state, x: str) -> str:
        if x.startswith(("p:", "o:", "svc:")):
            if x.startswith("svc:") and state.has("principals", x):
                return x
            if state.has("persons", x) or state.has("orgs", x):
                return x
            raise NotFound(f"no principal {x}")
        for table, prefix in (("persons", "p:"), ("orgs", "o:")):
            if state.find_ids(table, "handle", x):
                return prefix + x
        raise NotFound(f"no principal with handle {x!r}")

    @endpoint("update_profile", S.obj({"bio": S.s_str("public bio", maxLength=2000),
                                       "org": S.s_str("organisation to update instead of yourself", maxLength=40)},
                                      ["bio"]))
    def ep_update_profile(self, ctx, bio, org=None):
        target = ctx.actor
        if org:
            target = self.resolve_handle(ctx.state, org)
            self.authorize(ctx, target, "org.admin")
        ctx.emit("id.bio", {"id": target, "bio": bio}, vis="public")
        return {"ok": True}

    @endpoint("create_org", S.obj({"handle": S.s_str("lowercase handle", pattern=HANDLE, maxLength=31),
                                   "name": S.s_str("display name", maxLength=120),
                                   "kind": S.s_enum(["group", "company", "cooperative", "association"], "type")},
                                  ["handle", "name"]), cost_s=60)
    def ep_create_org(self, ctx, handle, name, kind="group"):
        """Create an organisation; you become its owner. Companies must also register with biz to trade."""
        if not ctx.actor.startswith("p:"):
            raise Forbidden("only persons can found organisations")
        oid = self.create_org(ctx, handle, name, ctx.actor, kind)
        return {"org": oid}

    @endpoint("invite", S.obj({"org": S.s_str(maxLength=40), "person": S.s_str(maxLength=40),
                               "role": S.s_enum(list(ROLES)), "title": S.s_str(maxLength=80)},
                              ["org", "person"]))
    def ep_invite(self, ctx, org, person, role="member", title=""):
        """Invite a person into an organisation. They join only if they accept."""
        org = self.resolve_handle(ctx.state, org)
        person = self.resolve_handle(ctx.state, person)
        self.authorize(ctx, org, "org.admin")
        if not person.startswith("p:"):
            raise InvalidAction("only persons can be members")
        if ctx.state.has("members", f"{org}|{person}"):
            raise Conflict("already a member")
        iid = ctx.state.new_id("inv")
        ctx.emit("id.invite", {"id": iid, "org": org, "person": person, "role": role, "title": title,
                               "by": ctx.actor}, vis=(person, org))
        return {"invite": iid}

    @endpoint("respond_invite", S.obj({"invite": S.s_str(maxLength=40), "accept": S.s_bool()}, ["invite", "accept"]))
    def ep_respond_invite(self, ctx, invite, accept):
        inv = ctx.state.get("invites", invite)
        if inv is None or inv["person"] != ctx.actor or inv["status"] != "pending":
            raise NotFound("no such pending invite")
        ctx.emit("id.invite_response", {"id": invite, "accept": accept}, vis=(ctx.actor, inv["org"]))
        if accept:
            self.add_member(ctx, inv["org"], ctx.actor, inv["role"], inv["title"])
        return {"joined": bool(accept), "org": inv["org"]}

    def add_member(self, ctx, org, person, role="member", title=""):
        if not ctx.state.has("members", f"{org}|{person}"):
            ctx.emit("id.member", {"org": org, "person": person, "role": role, "title": title}, vis=(person, org))

    def remove_member(self, ctx, org, person, reason=""):
        if ctx.state.has("members", f"{org}|{person}"):
            ctx.emit("id.member_removed", {"org": org, "person": person, "reason": reason}, vis=(person, org))

    @endpoint("invites", S.obj(), read_only=True)
    def ep_invites(self, ctx):
        return [dict(i, id=iid) for iid in ctx.state.find_ids("invites", "person", ctx.actor)
                for i in [ctx.state.tables["invites"][iid]] if i["status"] == "pending"]

    @endpoint("set_role", S.obj({"org": S.s_str(maxLength=40), "person": S.s_str(maxLength=40),
                                 "role": S.s_enum(list(ROLES))}, ["org", "person", "role"]))
    def ep_set_role(self, ctx, org, person, role):
        org = self.resolve_handle(ctx.state, org)
        person = self.resolve_handle(ctx.state, person)
        if self.role(ctx.state, org, ctx.actor) != "owner":
            raise Forbidden("only owners change roles")
        if not ctx.state.has("members", f"{org}|{person}"):
            raise NotFound("not a member")
        if role != "owner" and person == ctx.actor and len(self.org_owner_persons(ctx.state, org)) == 1:
            raise InvalidAction("an organisation needs at least one owner")
        ctx.emit("id.role", {"org": org, "person": person, "role": role}, vis=(person, org))
        return {"ok": True}

    @endpoint("leave", S.obj({"org": S.s_str(maxLength=40)}, ["org"]))
    def ep_leave(self, ctx, org):
        org = self.resolve_handle(ctx.state, org)
        if not ctx.state.has("members", f"{org}|{ctx.actor}"):
            raise NotFound("not a member")
        if self.role(ctx.state, org, ctx.actor) == "owner" and len(self.org_owner_persons(ctx.state, org)) == 1:
            raise InvalidAction("transfer ownership before leaving")
        self.remove_member(ctx, org, ctx.actor, "left")
        ctx.world.hooks.get("on_member_left", lambda *a: None)(ctx, org, ctx.actor)
        return {"ok": True}

    @endpoint("remove", S.obj({"org": S.s_str(maxLength=40), "person": S.s_str(maxLength=40)}, ["org", "person"]))
    def ep_remove(self, ctx, org, person):
        org = self.resolve_handle(ctx.state, org)
        person = self.resolve_handle(ctx.state, person)
        self.authorize(ctx, org, "org.admin")
        if self.role(ctx.state, org, person) == "owner":
            raise Forbidden("owners cannot be removed")
        self.remove_member(ctx, org, person, "removed")
        ctx.world.hooks.get("on_member_left", lambda *a: None)(ctx, org, person)
        return {"ok": True}

    @endpoint("grant", S.obj({"grantee": S.s_str(maxLength=60), "cap": S.s_str("capability, e.g. bank.spend or code.*", maxLength=40),
                              "owner": S.s_str("whose resources (default: you)", maxLength=40),
                              "scope": S.s_str("resource id or *", maxLength=80),
                              "limit": S.s_int("spending limit per 30 days in cents", minimum=0),
                              "expires_in": S.s_int("seconds", minimum=60)}, ["grantee", "cap"]))
    def ep_grant(self, ctx, grantee, cap, owner=None, scope="*", limit=None, expires_in=None):
        """Delegate a capability over your (or your organisation's) resources to another principal."""
        owner = self.resolve_handle(ctx.state, owner) if owner else ctx.actor
        if owner != ctx.actor:
            self.authorize(ctx, owner, "org.admin")
        grantee = self.resolve_handle(ctx.state, grantee)
        if not re.fullmatch(r"[a-z]+(\.[a-z_*]+)?", cap):
            raise InvalidAction("bad capability")
        gid = ctx.state.new_id("grant")
        ctx.emit("id.grant", {"id": gid, "grantee": grantee, "owner": owner, "cap": cap, "scope": scope,
                              "limit": limit, "expires": None if expires_in is None else ctx.now + expires_in,
                              "by": ctx.actor}, vis=(grantee, owner))
        return {"grant": gid}

    @endpoint("revoke", S.obj({"grant": S.s_str(maxLength=40)}, ["grant"]))
    def ep_revoke(self, ctx, grant):
        g = ctx.state.get("grants", grant)
        if g is None:
            raise NotFound("no grant")
        if g["owner"] != ctx.actor:
            self.authorize(ctx, g["owner"], "org.admin")
        ctx.emit("id.revoke", {"id": grant}, vis=(g["grantee"], g["owner"]))
        return {"ok": True}

    @endpoint("me", S.obj(), read_only=True)
    def ep_me(self, ctx):
        """Your organisations (with role), grants you hold and grants you issued."""
        orgs = [{"org": m["org"], "role": m["role"], "title": m["title"]}
                for m in ctx.state.find("members", "person", ctx.actor)]
        held = [dict(g, id=gid) for gid in ctx.state.find_ids("grants", "grantee", ctx.actor)
                for g in [ctx.state.tables["grants"][gid]] if g["status"] == "active"]
        return {"principal": ctx.actor, "orgs": orgs, "grants_held": held}

    @endpoint("directory", S.obj({"kind": S.s_enum(["persons", "orgs"]), "limit": S.s_int(minimum=1, maximum=200)}),
              read_only=True)
    def ep_directory(self, ctx, kind="orgs", limit=50):
        """Public directory listing (most recently created first)."""
        rows = []
        table = "orgs" if kind == "orgs" else "persons"
        for rid in reversed(ctx.state.ids(table)):
            r = ctx.state.tables[table][rid]
            rows.append({"id": rid, "handle": r["handle"], "name": r["name"], "kind": r["kind"]})
            if len(rows) >= limit:
                break
        return rows

    # ------------------------------------------------------------------ appliers
    @applier("id.person")
    def ap_person(self, ev):
        p = ev.payload
        self.world.state.insert("persons", p["id"], {"handle": p["handle"], "name": p["name"], "kind": p["kind"],
                                                    "info": p["info"], "created": ev.t, "bio": ""})

    @applier("id.org")
    def ap_org(self, ev):
        p = ev.payload
        self.world.state.insert("orgs", p["id"], {"handle": p["handle"], "name": p["name"], "kind": p["kind"],
                                                 "founder": p["founder"], "info": p["info"], "created": ev.t,
                                                 "status": "active", "bio": ""})

    @applier("id.credential")
    def ap_cred(self, ev):
        self.world.state.put("credentials", ev.payload["principal"], {"salt": ev.payload["salt"], "hash": ev.payload["hash"]})

    @applier("id.bio")
    def ap_bio(self, ev):
        t = "persons" if ev.payload["id"].startswith("p:") else "orgs"
        self.world.state.patch(t, ev.payload["id"], bio=ev.payload["bio"])

    @applier("id.member")
    def ap_member(self, ev):
        p = ev.payload
        self.world.state.put("members", f"{p['org']}|{p['person']}", {"org": p["org"], "person": p["person"],
                                                                     "role": p["role"], "title": p["title"], "since": ev.t})

    @applier("id.member_removed")
    def ap_member_removed(self, ev):
        self.world.state.delete("members", f"{ev.payload['org']}|{ev.payload['person']}")

    @applier("id.role")
    def ap_role(self, ev):
        p = ev.payload
        self.world.state.patch("members", f"{p['org']}|{p['person']}", role=p["role"])

    @applier("id.invite")
    def ap_invite(self, ev):
        p = ev.payload
        self.world.state.insert("invites", p["id"], {"org": p["org"], "person": p["person"], "role": p["role"],
                                                    "title": p["title"], "by": p["by"], "status": "pending", "t": ev.t})

    @applier("id.invite_response")
    def ap_invite_response(self, ev):
        self.world.state.patch("invites", ev.payload["id"], status="accepted" if ev.payload["accept"] else "declined")

    @applier("id.grant")
    def ap_grant(self, ev):
        p = ev.payload
        self.world.state.insert("grants", p["id"], {"grantee": p["grantee"], "owner": p["owner"], "cap": p["cap"],
                                                   "scope": p["scope"], "limit": p["limit"], "expires": p["expires"],
                                                   "by": p["by"], "status": "active", "used": 0, "window_start": ev.t,
                                                   "t": ev.t})

    @applier("id.grant_used")
    def ap_grant_used(self, ev):
        g = self.world.state.tables["grants"][ev.payload["id"]]
        if ev.t - g["window_start"] >= WINDOW:
            self.world.state.patch("grants", ev.payload["id"], used=ev.payload["amount"], window_start=ev.t)
        else:
            self.world.state.patch("grants", ev.payload["id"], used=g["used"] + ev.payload["amount"])

    @applier("id.revoke")
    def ap_revoke(self, ev):
        self.world.state.patch("grants", ev.payload["id"], status="revoked")

    @applier("id.org_status")
    def ap_org_status(self, ev):
        self.world.state.patch("orgs", ev.payload["id"], status=ev.payload["status"])
        self.world.state.patch("principals", ev.payload["id"], status="active" if ev.payload["status"] == "active" else "inactive")

    def check_invariants(self, world):
        errs = []
        st = world.state
        for oid, o in st.tables["orgs"].items():
            if o["status"] == "active" and not any(m["role"] == "owner" for m in st.find("members", "org", oid)):
                if o["founder"] is not None:
                    errs.append(f"org {oid} has no owner")
        for mid, m in st.tables["members"].items():
            if not st.has("persons", m["person"]) or not st.has("orgs", m["org"]):
                errs.append(f"dangling membership {mid}")
        return errs

    def observe(self, world, actor):
        st = world.state
        if not actor.startswith("p:"):
            return None
        return {"orgs": {m["org"]: m["role"] for m in st.find("members", "person", actor)},
                "pending_invites": sum(1 for i in st.find("invites", "person", actor) if i["status"] == "pending")}


def _cap_match(pattern: str, cap: str) -> bool:
    if pattern == cap or pattern == "*":
        return True
    if pattern.endswith(".*"):
        return cap.startswith(pattern[:-1])
    return False


def principals_of(world, actor: str) -> set[str]:
    out = {actor}
    st = world.state
    if "members" in st.tables:
        for m in st.find("members", "person", actor):
            if m["role"] in ("owner", "admin"):
                out.add(m["org"])
    return out


def may_act_as(world, actor: str, principal: str) -> bool:
    st = world.state
    if principal.startswith("o:"):
        m = st.get("members", f"{principal}|{actor}")
        return bool(m and m["role"] in ("owner", "admin"))
    if principal.startswith("svc:"):
        p = st.get("principals", principal)
        op = p["info"].get("operator") if p else None
        return op is not None and (op == actor or may_act_as(world, actor, op))
    return False


def ident(world) -> Identity:
    return world.modules["identity"]  # type: ignore[return-value]
