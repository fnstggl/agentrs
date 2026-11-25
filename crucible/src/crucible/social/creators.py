"""Creator subscriptions (module `creators`, host `creators`) and newsletters (module `letters`, host `letters`).

Creator memberships: a creator offers tiers (a monthly price and perks, written by the creator); a fan joins a tier
and pays from its OWN bank account (a population individual is given its own account through
`substrate.ids.ensure_principal(..., account=True)`; there is no pooled payer). The first period is charged at once,
renewals by the hourly process `creators.bill` when a paid period ends; a payment that fails ends access (status
`lapsed`), cancellation stops renewal at the period end (or at once). Every charge is one `bank.move` of ledger kind
`subscription` from the fan's account to the creator's primary account, so revenue is the sum of independent fans'
real payments and nothing else. Members-only items (`access: subscribers`) are readable by fans with a paid-up
membership (the `content.access` provider `subscribers`).

When workstream F's subscription module (`subs`, with products `prod`) is in the world it is the billing system
of record for recurring revenue metrics; this module then remains the creator-side membership catalogue (see the
component guide for the connection).

Newsletters: a creator's mailing list. Free subscribers receive every issue; issues marked paid are members-only
items for the newsletter's tier. Sending an issue publishes it as a catalogue item of kind `newsletter` on platform
`letters` and delivers it: by mail (4.0 `mail` module, with its delivery delays) to subscribers with mailboxes, and
as impressions of the consumption engine (when present) to population subscribers.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InsufficientFunds, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..substrate import bus, ids
from . import content as C
from . import nodes
from .edges import EdgeTable

DAY = 86400
PERIOD = 30 * DAY
LIVE = ("active",)
NLSUB = EdgeTable("nlsub", {"node": ("int64", (), -1), "letter": ("int32", (), -1), "t": ("int64", (), -1),
                            "t_end": ("int64", (), -1), "active": ("bool", (), False), "via": ("uint8", (), 0)})

_S, _I = {"type": "string"}, {"type": "integer"}
for _n, _p, _r, _d in [
    ("creators.tier", {"id": _S, "creator": _S, "name": _S, "price_cents": _I, "perks": _S}, ["id", "creator", "name",
                                                                                              "price_cents"],
     "a creator offered a membership tier"),
    ("creators.tier_status", {"id": _S, "status": {"type": "string", "enum": ["active", "archived"]}}, ["id", "status"],
     "a tier was archived or reopened"),
    ("creators.membership", {"id": _S, "tier": _S, "creator": _S, "fan": _S, "account": _S}, ["id", "tier", "creator",
                                                                                              "fan", "account"],
     "a fan joined a tier"),
    ("creators.paid", {"id": _S, "txn": _S, "amount": _I, "period_end": _I}, ["id", "txn", "amount", "period_end"],
     "a membership period was paid (the bank transaction is the payment)"),
    ("creators.status", {"id": _S, "status": {"type": "string", "enum": ["active", "cancelled", "lapsed"]},
                         "reason": _S, "at_period_end": {"type": "boolean"}}, ["id", "status", "reason"],
     "a membership changed status"),
    ("letters.newsletter", {"id": _S, "n": _I, "owner": _S, "name": _S, "description": _S,
                            "tier": {"type": ["string", "null"]}}, ["id", "n", "owner", "name"],
     "a newsletter was created"),
    ("letters.issue", {"id": _S, "newsletter": _S, "item": _S, "paid": {"type": "boolean"}, "mailed": _I,
                       "population": _I}, ["id", "newsletter", "item", "paid", "mailed", "population"],
     "an issue was sent"),
]:
    bus.event_type(_n, S.obj(_p, _r), owner=_n.split(".")[0], doc=_d)


def _bank(world):
    b = world.modules.get("bank")
    if b is None:
        raise NotFound("creator memberships need the bank")
    return b


@register_module
class Creators(Module):
    name = "creators"
    host = "creators"
    version = "1"
    depends = ("identity", "bank", "content", "social_ids")

    def install(self, world):
        st = world.state
        st.declare_table("creator_tiers", index=("creator", "status"))
        st.declare_table("creator_members", index=("creator", "fan", "status", "tier"))
        world.hooks.setdefault("content.access", {})["subscribers"] = self.access
        bus.enforce(world, owner="creators")

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 3600, "creators.bill", {}, prio=45, period=3600, key="creators.bill")

    # ------------------------------------------------------------------ internal API
    def live(self, st, creator: str, fan: str) -> str | None:
        for mid in st.find_ids("creator_members", "fan", fan):
            m = st.tables["creator_members"][mid]
            if m["creator"] == creator and m["status"] == "active":
                return mid
        return None

    def access(self, world, principal: str, rec) -> bool:
        if rec["access"] != "subscribers":
            return False
        return any(self.live(world.state, rec["creator"], p) is not None for p in world.principals_of(principal))

    def _charge(self, ctx: Ctx, mid: str) -> str:
        st = ctx.state
        m = st.tables["creator_members"][mid]
        t = st.tables["creator_tiers"][m["tier"]]
        bank = _bank(ctx.world)
        dst = bank.primary_account(st, m["creator"])
        if dst is None:
            raise NotFound(f"{m['creator']} has no account to be paid into")
        txn = bank.move(ctx, m["account"], dst, int(t["price_cents"]), "subscription",
                        f"membership {t['name']} of {m['creator']}", ref=mid)
        start = max(ctx.now, m["paid_until"] or 0)
        bus.emit(ctx, "creators.paid", {"id": mid, "txn": txn, "amount": int(t["price_cents"]),
                                        "period_end": start + PERIOD}, vis=tuple(sorted({m["fan"], m["creator"]})))
        return txn

    def join(self, ctx: Ctx, fan: str, tier: str, account: str | None = None) -> str:
        st = ctx.state
        t = st.get("creator_tiers", tier)
        if t is None or t["status"] != "active":
            raise NotFound(f"no tier {tier} on offer")
        if fan in ctx.world.principals_of(t["creator"]) or fan == t["creator"]:
            raise Forbidden("you cannot join your own tier")
        if self.live(st, t["creator"], fan) is not None:
            raise Conflict(f"you already support {t['creator']} (cancel first to change tier)")
        bank = _bank(ctx.world)
        acct = account or bank.primary_account(st, fan)
        if acct is None:
            raise NotFound(f"{fan} has no bank account")
        a = st.get("accounts", acct)
        if a is None or (a["owner"] != fan and a["owner"] not in ctx.world.principals_of(fan)):
            raise Forbidden("pay from your own account")
        mid = st.new_id("mbr")
        nodes.ensure(ctx, fan)
        bus.emit(ctx, "creators.membership", {"id": mid, "tier": tier, "creator": t["creator"], "fan": fan,
                                              "account": acct}, vis=tuple(sorted({fan, t["creator"]})))
        self._charge(ctx, mid)
        return mid

    # ------------------------------------------------------------------ endpoints
    @endpoint("create_tier", S.obj({"name": S.s_str(minLength=1, maxLength=80),
                                    "price_cents": S.s_int("per 30 days", minimum=1, maximum=10**7),
                                    "perks": S.s_str("what members get, in your words", maxLength=2000)},
                                   ["name", "price_cents"]), cost_s=30)
    def ep_create_tier(self, ctx, name, price_cents, perks=""):
        """Offer a monthly membership tier to your audience (members can open your members-only items)."""
        if _bank(ctx.world).primary_account(ctx.state, ctx.actor) is None:
            raise Forbidden("you need a bank account to be paid")
        if len(ctx.state.find_ids("creator_tiers", "creator", ctx.actor)) >= 10:
            raise Conflict("at most 10 tiers")
        tid = ctx.state.new_id("tier")
        nodes.ensure(ctx, ctx.actor)
        bus.emit(ctx, "creators.tier", {"id": tid, "creator": ctx.actor, "name": str(name)[:80],
                                        "price_cents": int(price_cents), "perks": str(perks)[:2000]}, vis="public")
        return {"tier": tid}

    @endpoint("archive_tier", S.obj({"tier": S.s_str(maxLength=20)}, ["tier"]), cost_s=5)
    def ep_archive_tier(self, ctx, tier):
        """Stop offering a tier (current members keep it until they cancel or a payment fails)."""
        t = ctx.state.get("creator_tiers", tier)
        if t is None or t["creator"] != ctx.actor:
            raise NotFound(f"no tier {tier} of yours")
        bus.emit(ctx, "creators.tier_status", {"id": tier, "status": "archived"}, vis="public")
        return {"tier": tier, "status": "archived"}

    @endpoint("tiers", S.obj({"creator": S.s_str(maxLength=60)}, ["creator"]), read_only=True)
    def ep_tiers(self, ctx, creator):
        """A creator's tiers on offer, with prices, perks and how many members each has."""
        from ..services.identity import ident
        p = creator if ids.index(creator) is not None else ident(ctx.world).resolve_handle(ctx.state, creator)
        st = ctx.state
        out = []
        for tid in st.find_ids("creator_tiers", "creator", p):
            t = st.tables["creator_tiers"][tid]
            if t["status"] != "active":
                continue
            n = sum(1 for m in st.find_ids("creator_members", "tier", tid)
                    if st.tables["creator_members"][m]["status"] == "active")
            out.append({"tier": tid, "name": t["name"], "price_cents": t["price_cents"], "perks": t["perks"],
                        "members": n})
        return {"creator": p, "tiers": out}

    @endpoint("join", S.obj({"tier": S.s_str(maxLength=20), "account": S.s_str("your account", maxLength=40)},
                            ["tier"]), cost_s=20)
    def ep_join(self, ctx, tier, account=None):
        """Join a creator's tier; the first 30 days are charged now from your account, then every 30 days."""
        mid = self.join(ctx, ctx.actor, tier, account)
        m = ctx.state.tables["creator_members"][mid]
        return {"membership": mid, "paid_until": m["paid_until"], "status": m["status"]}

    @endpoint("cancel", S.obj({"membership": S.s_str(maxLength=20),
                               "at_period_end": S.s_bool("keep access until the paid period ends", default=True)},
                              ["membership"]), cost_s=5)
    def ep_cancel(self, ctx, membership, at_period_end=True):
        """Cancel your membership (no renewal; at once if at_period_end is false - nothing is refunded)."""
        m = ctx.state.get("creator_members", membership)
        if m is None or m["fan"] != ctx.actor:
            raise NotFound(f"no membership {membership} of yours")
        if m["status"] != "active":
            raise Conflict(f"membership is {m['status']}")
        if at_period_end:
            if m["cancel_at_period_end"]:
                raise Conflict("already cancelled at period end")
            bus.emit(ctx, "creators.status", {"id": membership, "status": "active", "reason": "cancel_at_period_end",
                                              "at_period_end": True}, vis=tuple(sorted({m["fan"], m["creator"]})))
        else:
            bus.emit(ctx, "creators.status", {"id": membership, "status": "cancelled", "reason": "fan",
                                              "at_period_end": False}, vis=tuple(sorted({m["fan"], m["creator"]})))
        return {"membership": membership, "at_period_end": bool(at_period_end)}

    @endpoint("mine", S.obj({"role": S.s_enum(["fan", "creator"])}), read_only=True)
    def ep_mine(self, ctx, role="fan"):
        """Memberships you pay for (fan), or your members (creator) with this month's recurring revenue."""
        st = ctx.state
        field = "fan" if role == "fan" else "creator"
        rows = [dict(st.tables["creator_members"][m], membership=m) for m in st.find_ids("creator_members", field,
                                                                                         ctx.actor)]
        out = {"memberships": rows}
        if role == "creator":
            live = [r for r in rows if r["status"] == "active" and not r["cancel_at_period_end"]]
            out["active"] = sum(1 for r in rows if r["status"] == "active")
            out["monthly_cents"] = sum(st.tables["creator_tiers"][r["tier"]]["price_cents"] for r in live)
        return out

    @endpoint("publish", S.obj({"ref": S.s_str(maxLength=71), "kind": S.s_enum(list(C.KINDS)),
                                "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE), "tags": C._TAGS,
                                "description": S.s_str("public teaser", maxLength=C.MAX_DESC)},
                               ["ref", "kind", "title"]), cost_s=30)
    def ep_publish(self, ctx, ref, kind, title, tags=(), description=""):
        """Publish a members-only item (title and teaser are public; the artifact opens for paid-up members)."""
        if not ctx.state.find_ids("creator_tiers", "creator", ctx.actor):
            raise Conflict("offer a tier first (creators.create_tier)")
        cid = C.publish(ctx, ctx.actor, ref, kind, "creators", title, tags=tags, description=description,
                        access="subscribers")
        return {"id": cid}

    # ------------------------------------------------------------------ billing
    @process("bill")
    def pr_bill(self, ctx):
        st = ctx.state
        for mid in st.find_ids("creator_members", "status", "active"):
            m = st.tables["creator_members"][mid]
            if m["paid_until"] > ctx.now:
                continue
            if m["cancel_at_period_end"]:
                bus.emit(ctx, "creators.status", {"id": mid, "status": "cancelled", "reason": "fan",
                                                  "at_period_end": True}, vis=tuple(sorted({m["fan"], m["creator"]})))
                continue
            err = contain(ctx, self._charge, ctx, mid)
            if err is not None:
                why = "payment_failed" if isinstance(err, (InsufficientFunds, CrucibleError)) else type(err).__name__
                bus.emit(ctx, "creators.status", {"id": mid, "status": "lapsed", "reason": why, "at_period_end": False},
                         vis=tuple(sorted({m["fan"], m["creator"]})))

    # ------------------------------------------------------------------ appliers
    @applier("creators.tier")
    def ap_tier(self, ev):
        p = ev.payload
        self.world.state.insert("creator_tiers", p["id"], {"creator": p["creator"], "name": p["name"],
                                                           "price_cents": p["price_cents"], "perks": p.get("perks", ""),
                                                           "status": "active", "t": ev.t})

    @applier("creators.tier_status")
    def ap_tier_status(self, ev):
        self.world.state.patch("creator_tiers", ev.payload["id"], status=ev.payload["status"])

    @applier("creators.membership")
    def ap_membership(self, ev):
        p = ev.payload
        self.world.state.insert("creator_members", p["id"], {"tier": p["tier"], "creator": p["creator"],
                                                             "fan": p["fan"], "account": p["account"],
                                                             "status": "active", "started": ev.t, "paid_until": 0,
                                                             "periods": 0, "paid_cents": 0,
                                                             "cancel_at_period_end": False, "ended": None,
                                                             "reason": None})

    @applier("creators.paid")
    def ap_paid(self, ev):
        p = ev.payload
        st = self.world.state
        m = st.tables["creator_members"][p["id"]]
        st.patch("creator_members", p["id"], paid_until=p["period_end"], periods=m["periods"] + 1,
                 paid_cents=m["paid_cents"] + p["amount"], last_txn=p["txn"])

    @applier("creators.status")
    def ap_status(self, ev):
        p = ev.payload
        st = self.world.state
        if p["status"] == "active":
            st.patch("creator_members", p["id"], cancel_at_period_end=bool(p.get("at_period_end")))
        else:
            st.patch("creator_members", p["id"], status=p["status"], ended=ev.t, reason=p.get("reason"))

    def check_invariants(self, world):
        errs = []
        st = world.state
        if "txns" not in st.tables:
            return errs
        paid: dict[str, int] = {}
        for tid, t in st.tables["txns"].items():
            if t["kind"] == "subscription" and t.get("ref") in st.tables["creator_members"]:
                pos = sum(a for _, a in t["postings"] if a > 0)
                paid[t["ref"]] = paid.get(t["ref"], 0) + pos
        for mid, m in st.tables["creator_members"].items():
            if m["paid_cents"] != paid.get(mid, 0):
                errs.append(f"membership {mid}: recorded {m['paid_cents']} paid, ledger has {paid.get(mid, 0)}")
            acct = st.tables["accounts"].get(m["account"])
            if acct is None or (acct["owner"] != m["fan"] and not acct["owner"].startswith("o:")):
                errs.append(f"membership {mid} is paid from someone else's account")
        return errs


@register_module
class Letters(Module):
    name = "letters"
    host = "letters"
    version = "1"
    depends = ("identity", "content", "social_ids", "social_rows", "cols")

    def install(self, world):
        st = world.state
        st.declare_table("newsletters", index=("owner",))
        st.declare_table("newsletter_index")
        st.declare_table("newsletter_issues", index=("newsletter",))
        NLSUB.declare(world, 1024)
        bus.enforce(world, owner="letters")

    def _nl(self, ctx: Ctx, nid: str) -> dict:
        n = ctx.state.get("newsletters", nid)
        if n is None:
            raise NotFound(f"no newsletter {nid}")
        return dict(n)

    def subscribers(self, world, letter_n: int) -> np.ndarray:
        v = NLSUB.view(world)
        return v["node"][(v["letter"] == int(letter_n)) & v["active"]]

    def subscribe_many(self, ctx: Ctx, node_arr: np.ndarray, letter_arr: np.ndarray, via: int = 0) -> int:
        from .edges import pair_keys
        v = NLSUB.view(ctx.world)
        k = pair_keys(node_arr, letter_arr.astype(np.int64))
        _, first = np.unique(k, return_index=True)
        first = np.sort(first)
        node_arr, letter_arr, k = node_arr[first], letter_arr[first], k[first]
        if v["active"].any():
            have = pair_keys(v["node"][v["active"]], v["letter"][v["active"]].astype(np.int64))
            new = ~np.isin(k, have)
            node_arr, letter_arr = node_arr[new], letter_arr[new]
        if not node_arr.size:
            return 0
        n = node_arr.size
        NLSUB.append(ctx, "letters.subscribe", {"node": node_arr, "letter": letter_arr.astype(np.int32),
                                                "t": np.full(n, ctx.now, np.int64), "t_end": np.full(n, -1, np.int64),
                                                "active": np.ones(n, bool), "via": np.full(n, via, np.uint8)})
        return int(n)

    @endpoint("create", S.obj({"name": S.s_str(minLength=1, maxLength=80), "description": S.s_str(maxLength=2000),
                               "tier": S.s_str("your creator tier whose members receive paid issues", maxLength=20)},
                              ["name"]), cost_s=30)
    def ep_create(self, ctx, name, description="", tier=None):
        """Start a newsletter."""
        if tier is not None:
            t = ctx.state.get("creator_tiers", tier) if "creator_tiers" in ctx.state.tables else None
            if t is None or t["creator"] != ctx.actor:
                raise NotFound(f"no tier {tier} of yours")
        n = ctx.state.next_counter("newsletter") - 1
        nid = f"nl{n}"
        nodes.ensure(ctx, ctx.actor)
        bus.emit(ctx, "letters.newsletter", {"id": nid, "n": n, "owner": ctx.actor, "name": str(name)[:80],
                                             "description": str(description)[:2000], "tier": tier}, vis="public")
        from ..services import search
        search.index(ctx, f"newsletter:{nid}", "newsletter", str(name)[:200], str(description), ctx.actor, ref=nid)
        return {"newsletter": nid}

    @endpoint("subscribe", S.obj({"newsletter": S.s_str(maxLength=20)}, ["newsletter"]), cost_s=2)
    def ep_subscribe(self, ctx, newsletter):
        """Subscribe (free) to a newsletter: its issues reach you by mail."""
        n = self._nl(ctx, newsletter)
        if n["owner"] == ctx.actor:
            raise InvalidAction("you write this newsletter")
        node = nodes.ensure(ctx, ctx.actor)
        if not self.subscribe_many(ctx, np.array([node]), np.array([n["n"]])):
            raise Conflict("already subscribed")
        return {"newsletter": newsletter, "subscribed": True}

    @endpoint("unsubscribe", S.obj({"newsletter": S.s_str(maxLength=20)}, ["newsletter"]), cost_s=2)
    def ep_unsubscribe(self, ctx, newsletter):
        """Stop receiving a newsletter."""
        n = self._nl(ctx, newsletter)
        node = nodes.node_of(ctx.world, ctx.actor)
        v = NLSUB.view(ctx.world)
        rows = np.flatnonzero((v["node"] == (node if node is not None else -2)) & (v["letter"] == n["n"]) & v["active"])
        if not rows.size:
            raise NotFound("not subscribed")
        NLSUB.update(ctx, "letters.unsubscribe", rows, {"active": np.zeros(rows.size, bool),
                                                        "t_end": np.full(rows.size, ctx.now, np.int64)})
        return {"newsletter": newsletter, "subscribed": False}

    @endpoint("send", S.obj({"newsletter": S.s_str(maxLength=20), "ref": S.s_str("an HTML or text artifact you own",
                                                                                 maxLength=71),
                             "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE), "tags": C._TAGS,
                             "paid": S.s_bool("members-only issue (the newsletter's tier)", default=False)},
                            ["newsletter", "ref", "title"]), cost_s=60)
    def ep_send(self, ctx, newsletter, ref, title, tags=(), paid=False):
        """Send an issue: it is published (kind newsletter on letters) and delivered to the subscribers."""
        n = self._nl(ctx, newsletter)
        if n["owner"] != ctx.actor:
            raise Forbidden("only the newsletter's author sends issues")
        if paid and not n["tier"]:
            raise InvalidAction("paid issues need a newsletter with a tier")
        cid = C.publish(ctx, ctx.actor, ref, "newsletter", "letters", title, tags=tags,
                        description=f"{n['name']}", access="subscribers" if paid else "public")
        subs = self.subscribers(ctx.world, n["n"])
        pop = subs[subs < nodes.OFFSET]
        agents = [p for p in nodes.principals_of_nodes(ctx.world, subs[subs >= nodes.OFFSET]) if p]
        rec = ctx.state.tables["content"][cid]
        mailed = 0
        mail = ctx.world.modules.get("mail")
        if mail is not None:
            from ..services.mail import MAX_RCPT
            to = [p for p in agents if C.readable(ctx.world, p, rec)]
            for k in range(0, len(to), MAX_RCPT):
                mail.send(ctx, ctx.actor, to[k:k + MAX_RCPT], f"[{n['name']}] {title}",
                          f"A new issue of {n['name']}: {title}\nOpen it: content {cid} "
                          f"(http://letters.crucible/item?id={cid})")
            mailed = len(to)
        from . import integration
        reached = 0 if paid else integration.deliver_impressions(ctx, "letters", pop,
                                                                 ctx.state.tables["content"][cid].get("aud_item"))
        iid = ctx.state.new_id("issue")
        bus.emit(ctx, "letters.issue", {"id": iid, "newsletter": newsletter, "item": cid, "paid": bool(paid),
                                        "mailed": mailed, "population": reached}, vis="public")
        return {"issue": iid, "item": cid, "mailed": mailed, "population_reached": reached}

    @endpoint("list", S.obj({"limit": S.s_int(minimum=1, maximum=100, default=30)}), read_only=True)
    def ep_list(self, ctx, limit=30):
        """Newsletters with their subscriber counts, largest first."""
        out = []
        for nid, n in ctx.state.scan("newsletters"):
            out.append({"newsletter": nid, "name": n["name"], "owner": n["owner"], "description": n["description"],
                        "subscribers": int(self.subscribers(ctx.world, n["n"]).size), "issues": n["issues"]})
        out.sort(key=lambda r: (-r["subscribers"], r["newsletter"]))
        return out[:limit]

    @endpoint("issues", S.obj({"newsletter": S.s_str(maxLength=20)}, ["newsletter"]), read_only=True)
    def ep_issues(self, ctx, newsletter):
        """A newsletter's issues, newest first."""
        self._nl(ctx, newsletter)
        st = ctx.state
        rows = [dict(st.tables["newsletter_issues"][i], issue=i) for i in st.find_ids("newsletter_issues",
                                                                                       "newsletter", newsletter)]
        return sorted(rows, key=lambda r: (-r["t"], r["issue"]))

    @applier("letters.newsletter")
    def ap_newsletter(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("newsletters", p["id"], {"n": p["n"], "owner": p["owner"], "name": p["name"],
                                           "description": p.get("description", ""), "tier": p.get("tier"),
                                           "issues": 0, "t": ev.t})
        st.insert("newsletter_index", str(p["n"]), {"id": p["id"]})

    @applier("letters.issue")
    def ap_issue(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("newsletter_issues", p["id"], {"newsletter": p["newsletter"], "item": p["item"], "paid": p["paid"],
                                                 "mailed": p["mailed"], "population": p["population"], "t": ev.t})
        n = st.tables["newsletters"][p["newsletter"]]
        st.patch("newsletters", p["newsletter"], issues=n["issues"] + 1)

    def check_invariants(self, world):
        st = world.state
        errs = []
        counts: dict[str, int] = {}
        for _, i in st.tables["newsletter_issues"].items():
            counts[i["newsletter"]] = counts.get(i["newsletter"], 0) + 1
        for nid, n in st.tables["newsletters"].items():
            if n["issues"] != counts.get(nid, 0):
                errs.append(f"newsletter {nid} counts {n['issues']} issues, {counts.get(nid, 0)} recorded")
        return errs


__all__ = ["Creators", "Letters", "SYSTEM"]
