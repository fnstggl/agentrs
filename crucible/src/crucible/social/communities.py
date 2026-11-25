"""Communities (kernel module `community`, host `community`; MASTER_SPEC Parts 4.7 and 9.4).

A community is founded by a participant, with a subject the participant writes (nothing is predefined), and lives
as long as people keep taking part in it.

* **Persistent membership** - one row per (person, community) in the column table `cmember` (crucible.social.edges):
  status (member, left, banned, pending, invited), role (member, moderator, owner), first and latest join, leave
  time, number of membership spells, last activity, and the participation history counters (posts, comments,
  kudos given and received, events attended, items removed by moderators, warnings). Every counter moves only
  through the recorded decision that caused it (a post, a comment, a kudos, an attendance, a moderation action).
* **Entry and exit** - open communities admit on request; `approval` communities queue requests for moderators;
  `invite` communities admit invited people only. Leaving keeps the history; rejoining opens a new spell.
* **Inactivity** - a member is inactive after `inactive_days` without taking part (derived from last activity,
  never assigned); a community may prune members inactive for `prune_days` (daily process). Joining does not imply
  engagement: participation is measured, membership is not participation.
* **Content and member interactions** - posts are catalogue items on platform `community` (members-only content
  for private communities), comments are catalogue comments; kudos are member-to-member endorsements.
* **Shared activities** - scheduled events with RSVPs and attendance during the event window.
* **Trust** - reputation points and trust levels computed from the recorded history (Discourse/Stack Exchange
  style); trust levels gate proposals, kudos and founding spin-offs.
* **Governance** - who decides is itself a rule: `owner`, `moderators` or `vote` (members vote on proposals with a
  quorum of active members and a threshold within a window). Proposals change rules, name, description and
  subject, appoint or remove moderators, ban or unban, transfer ownership, merge into another community or archive.
* **Moderation** - members report items, comments and members; moderators remove content (catalogue removal with
  role `moderator`), warn, mute and ban; every action is in the public moderation log.
* **Organizational change** - rule versions, renaming, moderator turnover, merges (members move over), spin-offs
  (`found` with a parent) and archiving.
* **Success** - `community.health`: active members over 7/30 days, weekly contributors over the last weeks, and the
  retention of members who joined at least 30 days ago - sustained voluntary participation, not head counts.

Population individuals take part through the same state with batch decisions (`join_batch`, `leave_batch`,
`touch_batch`); agents through the endpoints.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..substrate import bus, ids
from . import content as C
from . import nodes
from .edges import EdgeTable

DAY = 86400
WEEK = 7 * DAY
SLUG_RE = r"[a-z][a-z0-9-]{2,39}"
NONE, MEMBER, LEFT, BANNED, PENDING, INVITED = 0, 1, 2, 3, 4, 5
STATUS = {NONE: "none", MEMBER: "member", LEFT: "left", BANNED: "banned", PENDING: "pending", INVITED: "invited"}
ROLE = {0: "member", 1: "moderator", 2: "owner"}
ROLE_CODE = {v: k for k, v in ROLE.items()}
CMEMBER = EdgeTable("cmember", {
    "node": ("int64", (), -1), "comm": ("int32", (), -1), "status": ("uint8", (), 0), "role": ("uint8", (), 0),
    "first_joined": ("int64", (), -1), "joined": ("int64", (), -1), "left": ("int64", (), -1),
    "spells": ("int32", (), 0), "last_active": ("int64", (), -1), "posts": ("int32", (), 0),
    "comments": ("int32", (), 0), "kudos_given": ("int32", (), 0), "kudos_recv": ("int32", (), 0),
    "attended": ("int32", (), 0), "removed": ("int32", (), 0), "warnings": ("int32", (), 0),
    "muted_until": ("int64", (), -1)})
DEFAULT_RULES = {"join": "open", "visibility": "public", "post": "members", "comment": "members",
                 "governance": "owner", "vote_threshold": 0.5, "quorum": 0.2, "vote_window_s": 3 * DAY,
                 "inactive_days": 30, "prune_days": 0, "min_trust_propose": 1, "min_trust_kudos": 0,
                 "max_members": 1_000_000}
RULE_SCHEMA = S.obj({
    "join": S.s_enum(["open", "approval", "invite"], "who may join"),
    "visibility": S.s_enum(["public", "private"], "private: posts are readable by members only"),
    "post": S.s_enum(["members", "anyone"], "who may post"), "comment": S.s_enum(["members", "anyone"]),
    "governance": S.s_enum(["owner", "moderators", "vote"], "who decides proposals"),
    "vote_threshold": S.s_num(minimum=0.5, maximum=1.0), "quorum": S.s_num(minimum=0.0, maximum=1.0),
    "vote_window_s": S.s_int(minimum=3600, maximum=30 * DAY), "inactive_days": S.s_int(minimum=1, maximum=365),
    "prune_days": S.s_int("0: never prune inactive members", minimum=0, maximum=3650),
    "min_trust_propose": S.s_int(minimum=0, maximum=3), "min_trust_kudos": S.s_int(minimum=0, maximum=3),
    "max_members": S.s_int(minimum=2, maximum=10_000_000)})
ACTIONS = ("set_rules", "rename", "describe", "add_moderator", "remove_moderator", "ban", "unban", "transfer",
           "merge", "archive")
TRUST_LEVELS = (0, 5, 25, 100)          # reputation needed for trust levels 0..3
MOD_ACTIONS = ("remove", "warn", "mute", "ban", "dismiss")


def reputation(row: dict) -> float:
    """Reputation points from a member's recorded history."""
    return (1.0 * row["posts"] + 0.5 * row["comments"] + 5.0 * row["kudos_recv"] + 2.0 * row["attended"]
            - 15.0 * row["removed"] - 10.0 * row["warnings"])


def trust_level(rep: float) -> int:
    lvl = 0
    for k, need in enumerate(TRUST_LEVELS):
        if rep >= need:
            lvl = k
    return lvl


def _mod(world) -> "Communities":
    m = world.modules.get("community")
    if m is None:
        raise NotFound("this world has no communities")
    return m  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------------- queries
def rows_of(world, comm_n: int | None = None, node: int | None = None) -> np.ndarray:
    v = CMEMBER.view(world)
    m = np.ones(v["node"].shape, dtype=bool)
    if comm_n is not None:
        m &= v["comm"] == int(comm_n)
    if node is not None:
        m &= v["node"] == int(node)
    return np.flatnonzero(m)


def member_row(world, node: int, comm_n: int) -> int | None:
    r = rows_of(world, comm_n, node)
    return int(r[0]) if r.size else None


def row_dict(world, r: int) -> dict:
    v = CMEMBER.view(world)
    return {k: (a[r].item() if hasattr(a[r], "item") else a[r]) for k, a in v.items()}


def members(world, comm_n: int, *, active_days: int | None = None) -> np.ndarray:
    """Nodes that are members now (optionally: active within the last `active_days`)."""
    v = CMEMBER.view(world)
    m = (v["comm"] == int(comm_n)) & (v["status"] == MEMBER)
    if active_days is not None:
        m &= v["last_active"] >= world.now - int(active_days) * DAY
    return v["node"][m]


def memberships(world, node: int) -> np.ndarray:
    """Community indices `node` is a member of now."""
    v = CMEMBER.view(world)
    return v["comm"][(v["node"] == int(node)) & (v["status"] == MEMBER)]


def by_index(world, comm_n: int) -> dict:
    cid = world.state.tables["community_index"].get(str(int(comm_n)))
    if cid is None:
        raise NotFound(f"no community #{comm_n}")
    return dict(world.state.tables["communities"][cid["id"]])


# ---------------------------------------------------------------------------------------------------- population API
def join_batch(ctx: Ctx, nodes_: np.ndarray, comm_n: np.ndarray, *, via: str = "population") -> int:
    """Each nodes_[i] asks to join community comm_n[i] (open: admitted; approval: pending; invite-only: admitted
    only when invited; banned: refused). Returns the number admitted or queued."""
    return _mod(ctx.world).join_many(ctx, np.asarray(nodes_, np.int64), np.asarray(comm_n, np.int64), via)


def leave_batch(ctx: Ctx, nodes_: np.ndarray, comm_n: np.ndarray, *, reason: str = "left") -> int:
    return _mod(ctx.world).leave_many(ctx, np.asarray(nodes_, np.int64), np.asarray(comm_n, np.int64), reason)


def touch_batch(ctx: Ctx, nodes_: np.ndarray, comm_n: np.ndarray) -> int:
    """Record that members visited (read) their communities now (last activity) - reading counts as taking part,
    posting and commenting are counted by the content they create."""
    return _mod(ctx.world).touch_many(ctx, np.asarray(nodes_, np.int64), np.asarray(comm_n, np.int64))


# ---------------------------------------------------------------------------------------------------- typed events
_S, _I = {"type": "string"}, {"type": "integer"}
_O = {"type": "object", "additionalProperties": True}
for _name, _props, _req, _doc in [
    ("community.founded", {"id": _S, "n": _I, "name": _S, "subject": _S, "description": _S,
                           "tags": {"type": "array", "items": _S}, "founder": _S, "rules": _O,
                           "parent": {"type": ["string", "null"]}}, ["id", "n", "name", "subject", "founder", "rules"],
     "a participant founded a community on a subject of its own choosing"),
    ("community.rules", {"id": _S, "rules": _O, "version": _I, "by": _S, "proposal": {"type": ["string", "null"]}},
     ["id", "rules", "version", "by"], "a community's rules changed"),
    ("community.info", {"id": _S, "name": _S, "description": _S, "subject": _S, "tags": {"type": "array"}, "by": _S},
     ["id", "by"], "a community was renamed or re-described"),
    ("community.status", {"id": _S, "status": {"type": "string", "enum": ["active", "archived", "merged"]},
                          "merged_into": {"type": ["string", "null"]}, "by": _S, "reason": _S},
     ["id", "status", "by"], "a community was archived, merged or reactivated"),
    ("community.proposal", {"id": _S, "community": _S, "action": {"type": "string", "enum": list(ACTIONS)},
                            "args": _O, "by": _S, "reason": _S, "deadline": _I, "electorate": _I},
     ["id", "community", "action", "args", "by", "deadline", "electorate"], "a governance proposal"),
    ("community.vote", {"proposal": _S, "voter": _S, "approve": {"type": "boolean"}}, ["proposal", "voter", "approve"],
     "a member voted on a proposal"),
    ("community.decided", {"proposal": _S, "outcome": {"type": "string", "enum": ["approved", "rejected", "expired",
                                                                                  "failed"]},
                           "yes": _I, "no": _I, "electorate": _I, "note": _S}, ["proposal", "outcome"],
     "a proposal was decided (and, if approved, executed)"),
    ("community.report", {"id": _S, "community": _S, "target": _S, "target_kind": _S, "reason": _S, "by": _S},
     ["id", "community", "target", "target_kind", "by"], "a member reported content or a member"),
    ("community.moderation", {"community": _S, "action": _S, "target": _S, "reason": _S, "by": _S,
                              "report": {"type": ["string", "null"]}, "until": {"type": ["integer", "null"]}},
     ["community", "action", "target", "by"], "a moderation action (public log)"),
    ("community.event", {"id": _S, "community": _S, "title": _S, "description": _S, "at": _I, "duration_s": _I,
                         "by": _S}, ["id", "community", "title", "at", "duration_s", "by"], "a scheduled shared activity"),
    ("community.rsvp", {"event": _S, "member": _S, "going": {"type": "boolean"}}, ["event", "member", "going"],
     "a member answered an event invitation"),
    ("community.attend", {"event": _S, "member": _S}, ["event", "member"], "a member attended an event"),
    ("community.kudos", {"community": _S, "from": _S, "to": _S, "note": _S}, ["community", "from", "to"],
     "a member endorsed another member"),
    ("community.notice", {"community": _S, "what": _S, "who": _S, "by": _S, "note": _S}, ["community", "what", "who"],
     "a membership notice (request, invitation, approval, removal) for the people concerned"),
]:
    bus.event_type(_name, S.obj(_props, _req), owner="community", doc=_doc)

_COMM = S.s_str("community id", pattern=SLUG_RE, maxLength=40)
_WHO = S.s_str("a principal (p:<handle>, p<i>) or handle", maxLength=60)


@register_module
class Communities(Module):
    name = "community"
    host = "community"
    version = "1"
    depends = ("identity", "content", "social", "social_ids", "social_rows", "cols")
    DEFAULTS = {"capacity": 4096, "max_founded_per_day": 3}

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(self.DEFAULTS, **self.config)

    def install(self, world):
        st = world.state
        st.declare_table("communities", index=("status", "founder"))
        st.declare_table("community_index")         # str(n) -> {id}
        st.declare_table("community_proposals", index=("community", "status"))
        st.declare_table("community_votes", index=("proposal",))
        st.declare_table("community_reports", index=("community", "status"))
        st.declare_table("community_modlog", index=("community",))
        st.declare_table("community_events", index=("community",))
        st.declare_table("community_rsvps", index=("event",))
        st.declare_table("community_history", index=("community",))
        CMEMBER.declare(world, int(self.cfg["capacity"]))
        world.hooks.setdefault("content.post_check", {})["community"] = self.post_check
        world.hooks.setdefault("content.access", {})["members"] = self.access
        bus.subscribe(world, "content.published", self._on_published, owner="community")
        bus.subscribe(world, "content.comment", self._on_comment, owner="community")
        bus.enforce(world, owner="community")

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY, "community.daily", {}, prio=60, period=DAY, key="community.daily")
        ctx.schedule(ctx.now + 3600, "community.resolve", {}, prio=55, period=3600, key="community.resolve")

    # ------------------------------------------------------------------ lookups
    def _c(self, ctx_or_world, cid: str) -> dict:
        st = ctx_or_world.state
        c = st.get("communities", cid) if isinstance(cid, str) else None
        if c is None:
            raise NotFound(f"no community {cid}")
        return dict(c)

    def _active(self, c: dict) -> None:
        if c["status"] != "active":
            raise Conflict(f"community {c['id']} is {c['status']}")

    def _who(self, ctx: Ctx, who: str) -> str:
        from ..services.identity import ident
        if ids.index(who) is not None:
            if not ctx.state.has("principals", who):
                raise NotFound(f"no principal {who}")
            return who
        return ident(ctx.world).resolve_handle(ctx.state, who)

    def _row(self, world, principal: str, c: dict) -> dict | None:
        node = nodes.node_of(world, principal)
        if node is None:
            return None
        r = member_row(world, node, c["n"])
        return None if r is None else dict(row_dict(world, r), row=r)

    def _role(self, world, principal: str, c: dict) -> str | None:
        if principal == SYSTEM:
            return "owner"
        for p in sorted(world.principals_of(principal)):
            r = self._row(world, p, c)
            if r is not None and r["status"] == MEMBER:
                return ROLE[r["role"]]
        return None

    def _require(self, ctx: Ctx, c: dict, roles: tuple[str, ...]) -> str:
        role = self._role(ctx.world, ctx.actor, c)
        if role not in roles:
            raise Forbidden(f"you need to be {' or '.join(roles)} of {c['id']}")
        return role

    def trust(self, world, principal: str, c: dict) -> dict:
        r = self._row(world, principal, c)
        if r is None:
            return {"reputation": 0.0, "level": 0}
        rep = reputation(r)
        return {"reputation": rep, "level": trust_level(rep)}

    def access(self, world, principal: str, rec) -> bool:
        """content.access provider: members-only items are readable by current members of their community."""
        if rec["access"] != "members" or rec["community"] is None:
            return False
        c = world.state.tables["communities"].get(rec["community"])
        if c is None:
            return False
        return self._role(world, principal, dict(c)) is not None

    def post_check(self, ctx: Ctx, author: str, community: str, action: str) -> int:
        """content.post_check provider: may `author` post/comment in `community`? Returns its index."""
        c = self._c(ctx, community)
        self._active(c)
        if author == SYSTEM:
            return c["n"]
        r = self._row(ctx.world, author, c)
        if r is not None and r["status"] == BANNED:
            raise Forbidden(f"you are banned from {community}")
        if r is not None and r["muted_until"] > ctx.now:
            raise Forbidden(f"you are muted in {community} until {r['muted_until']}")
        need = c["rules"]["post" if action == "post" else "comment"]
        if need == "members" and (r is None or r["status"] != MEMBER):
            raise Forbidden(f"only members can {action} in {community}")
        return c["n"]

    # ------------------------------------------------------------------ membership (internal, vectorised)
    def _upsert(self, ctx: Ctx, kind: str, node_arr: np.ndarray, comm_arr: np.ndarray, vals_new: dict,
                vals_existing: dict, existing_rows: np.ndarray, new_mask: np.ndarray) -> None:
        if existing_rows.size:
            CMEMBER.update(ctx, kind, existing_rows, vals_existing)
        if new_mask.any():
            k = int(new_mask.sum())
            base = {"node": node_arr[new_mask], "comm": comm_arr[new_mask].astype(np.int32)}
            for c, v in vals_new.items():
                base[c] = v[new_mask] if isinstance(v, np.ndarray) and v.shape == new_mask.shape else \
                    np.full(k, v)
            CMEMBER.append(ctx, kind, base)

    def _lookup(self, world, node_arr: np.ndarray, comm_arr: np.ndarray) -> np.ndarray:
        """Row of each (node, community) pair, -1 when none."""
        from .edges import pair_keys
        v = CMEMBER.view(world)
        out = np.full(node_arr.size, -1, dtype=np.int64)
        if not v["node"].size or not node_arr.size:
            return out
        have = pair_keys(v["node"], v["comm"].astype(np.int64))
        want = pair_keys(node_arr, comm_arr.astype(np.int64))
        order = np.argsort(have, kind="stable")
        pos = np.searchsorted(have[order], want)
        pos = np.clip(pos, 0, max(0, have.size - 1))
        hit = have[order][pos] == want
        out[hit] = order[pos[hit]]
        return out

    def join_many(self, ctx: Ctx, node_arr: np.ndarray, comm_arr: np.ndarray, via: str) -> int:
        st = ctx.state
        if node_arr.shape != comm_arr.shape:
            raise InvalidAction("nodes and communities must be equal-length vectors")
        if not node_arr.size:
            return 0
        from .edges import pair_keys
        _, first = np.unique(pair_keys(node_arr, comm_arr), return_index=True)
        first = np.sort(first)
        node_arr, comm_arr = node_arr[first], comm_arr[first]
        rules = []
        for cn in comm_arr:
            c = by_index(ctx.world, int(cn))
            rules.append((c["status"] == "active", c["rules"]["join"], c["rules"]["max_members"]))
        active = np.array([r[0] for r in rules], bool)
        policy = np.array([r[1] for r in rules])
        rows = self._lookup(ctx.world, node_arr, comm_arr)
        v = CMEMBER.view(ctx.world)
        status = np.where(rows >= 0, v["status"][np.maximum(rows, 0)], NONE) if v["status"].size else \
            np.full(rows.size, NONE)
        ok = active & np.isin(status, [NONE, LEFT, INVITED])
        ok &= ~((policy == "invite") & (status != INVITED))
        # capacity
        for cn in np.unique(comm_arr[ok]):
            cap = by_index(ctx.world, int(cn))["rules"]["max_members"]
            have = members(ctx.world, int(cn)).size
            idx = np.flatnonzero(ok & (comm_arr == cn))
            if have + idx.size > cap:
                ok[idx[max(0, cap - have):]] = False
        if not ok.any():
            return 0
        admit = ok & ((policy != "approval") | (status == INVITED))
        queue = ok & ~admit
        now = ctx.now
        for mask, new_status, kind in ((admit, MEMBER, "community.join"), (queue, PENDING, "community.request")):
            if not mask.any():
                continue
            ex = mask & (rows >= 0)
            exr = rows[ex]
            if new_status == MEMBER:
                vals_ex = {"status": np.full(exr.size, MEMBER, np.uint8), "joined": np.full(exr.size, now, np.int64),
                           "left": np.full(exr.size, -1, np.int64), "last_active": np.full(exr.size, now, np.int64),
                           "spells": v["spells"][exr] + 1,
                           "first_joined": np.where(v["first_joined"][exr] < 0, now, v["first_joined"][exr])}
                vals_new = {"status": MEMBER, "role": 0, "first_joined": now, "joined": now, "left": -1, "spells": 1,
                            "last_active": now}
            else:
                vals_ex = {"status": np.full(exr.size, PENDING, np.uint8)}
                vals_new = {"status": PENDING, "role": 0}
            self._upsert(ctx, kind, node_arr[mask], comm_arr[mask], vals_new, vals_ex, exr,
                         ~(rows[mask] >= 0))
        del st
        return int(ok.sum())

    def leave_many(self, ctx: Ctx, node_arr: np.ndarray, comm_arr: np.ndarray, reason: str) -> int:
        rows = self._lookup(ctx.world, node_arr, comm_arr)
        v = CMEMBER.view(ctx.world)
        rows = rows[rows >= 0]
        rows = rows[np.isin(v["status"][rows], [MEMBER, PENDING])]
        if not rows.size:
            return 0
        rows = np.unique(rows)
        CMEMBER.update(ctx, "community.leave", rows, {"status": np.full(rows.size, LEFT, np.uint8),
                                                      "left": np.full(rows.size, ctx.now, np.int64),
                                                      "role": np.zeros(rows.size, np.uint8)},
                       extra={"reason": str(reason)[:60]})
        return int(rows.size)

    def touch_many(self, ctx: Ctx, node_arr: np.ndarray, comm_arr: np.ndarray) -> int:
        rows = self._lookup(ctx.world, node_arr, comm_arr)
        v = CMEMBER.view(ctx.world)
        rows = rows[rows >= 0]
        rows = np.unique(rows[v["status"][rows] == MEMBER]) if rows.size else rows
        if not rows.size:
            return 0
        CMEMBER.update(ctx, "community.visit", rows, {"last_active": np.full(rows.size, ctx.now, np.int64)},
                       mode="max")
        return int(rows.size)

    def _bump(self, ctx: Ctx, principal: str, c: dict, kind: str, **add) -> None:
        node = nodes.node_of(ctx.world, principal)
        if node is None:
            return
        r = member_row(ctx.world, node, c["n"])
        if r is None:
            return
        if add:
            vals = {k: np.array([v], dtype=np.int32) for k, v in add.items()}
            CMEMBER.update(ctx, kind, np.array([r]), vals, mode="add")
        if row_dict(ctx.world, r)["status"] == MEMBER:
            CMEMBER.update(ctx, kind + ".active", np.array([r]), {"last_active": np.array([ctx.now], np.int64)},
                           mode="max")

    def _set_row(self, ctx: Ctx, kind: str, r: int, **vals) -> None:
        dt = {k: CMEMBER.cols[k][0] for k in vals}
        CMEMBER.update(ctx, kind, np.array([r]), {k: np.array([v], dtype=dt[k]) for k, v in vals.items()})

    # ------------------------------------------------------------------ subscribers: participation history
    def _on_published(self, ctx: Ctx, ev) -> None:
        p = ev.payload
        if p.get("community"):
            c = dict(ctx.state.tables["communities"][p["community"]])
            self._bump(ctx, p["creator"], c, "community.posted", posts=1)

    def _on_comment(self, ctx: Ctx, ev) -> None:
        p = ev.payload
        rec = ctx.state.tables["content"][p["item"]]
        if rec["community"]:
            c = dict(ctx.state.tables["communities"][rec["community"]])
            self._bump(ctx, p["author"], c, "community.commented", comments=1)

    # ------------------------------------------------------------------ internal API
    def found(self, ctx: Ctx, founder: str, slug: str, name: str, subject: str, description: str = "", tags=(),
              rules: dict | None = None, parent: str | None = None) -> str:
        if not re.fullmatch(SLUG_RE, slug):
            raise InvalidAction(f"bad community id {slug!r} (lowercase letters, digits and -, 3..40)")
        if ctx.state.has("communities", slug):
            raise Conflict(f"community {slug} exists")
        subject = str(subject).strip()
        if len(subject) < 3:
            raise InvalidAction("describe the subject in your own words (at least 3 characters)")
        r = dict(DEFAULT_RULES)
        for k, v in (rules or {}).items():
            if k not in DEFAULT_RULES:
                raise InvalidAction(f"unknown rule {k}")
            r[k] = v
        errs = S.validate(RULE_SCHEMA, r)
        if errs:
            raise InvalidAction("; ".join(errs[:3]))
        if parent is not None:
            pc = self._c(ctx, parent)
            self._active(pc)
            if self._role(ctx.world, founder, pc) is None:
                raise Forbidden(f"a spin-off of {parent} is founded by one of its members")
            if founder != SYSTEM and self.trust(ctx.world, founder, pc)["level"] < 2:
                raise Forbidden(f"founding a spin-off of {parent} needs trust level 2 there")
        day = ctx.now // DAY
        today = [c for c in ctx.state.find_ids("communities", "founder", founder)
                 if ctx.state.tables["communities"][c]["created"] // DAY == day]
        if len(today) >= int(self.cfg["max_founded_per_day"]):
            raise Conflict("you founded enough communities today")
        n = ctx.state.next_counter("community") - 1
        node = nodes.ensure(ctx, founder)
        bus.emit(ctx, "community.founded", {"id": slug, "n": n, "name": str(name)[:80], "subject": subject[:500],
                                            "description": str(description)[:4000],
                                            "tags": sorted({str(t)[:32].lower() for t in tags})[:10],
                                            "founder": founder, "rules": r, "parent": parent}, vis="public")
        CMEMBER.append(ctx, "community.found", {"node": np.array([node]), "comm": np.array([n], np.int32),
                                                "status": np.array([MEMBER], np.uint8),
                                                "role": np.array([ROLE_CODE["owner"]], np.uint8),
                                                "first_joined": np.array([ctx.now]), "joined": np.array([ctx.now]),
                                                "left": np.array([-1]), "spells": np.array([1], np.int32),
                                                "last_active": np.array([ctx.now])})
        from ..services import search
        search.index(ctx, f"community:{slug}", "community", str(name)[:200],
                     f"{subject}\n{description}\n{' '.join(tags)}", founder, ref=slug)
        return slug

    def _execute(self, ctx: Ctx, c: dict, action: str, args: dict, proposal: str | None) -> None:
        """Carry out a governance action (after its decision rule was satisfied)."""
        st = ctx.state
        if action == "set_rules":
            r = dict(c["rules"])
            for k, v in (args.get("rules") or {}).items():
                if k not in DEFAULT_RULES:
                    raise InvalidAction(f"unknown rule {k}")
                r[k] = v
            errs = S.validate(RULE_SCHEMA, r)
            if errs:
                raise InvalidAction("; ".join(errs[:3]))
            bus.emit(ctx, "community.rules", {"id": c["id"], "rules": r, "version": c["rules_version"] + 1,
                                              "by": ctx.actor, "proposal": proposal}, vis="public")
        elif action in ("rename", "describe"):
            pay = {"id": c["id"], "by": ctx.actor}
            for k in ("name", "description", "subject", "tags"):
                if k in args:
                    pay[k] = args[k] if k == "tags" else str(args[k])[:4000 if k == "description" else 500]
            if len(pay) == 2:
                raise InvalidAction("nothing to change")
            bus.emit(ctx, "community.info", pay, vis="public")
        elif action in ("add_moderator", "remove_moderator", "transfer"):
            who = self._who(ctx, str(args.get("member", "")))
            r = self._row(ctx.world, who, c)
            if r is None or r["status"] != MEMBER:
                raise NotFound(f"{who} is not a member of {c['id']}")
            if action == "add_moderator":
                if r["role"] >= 1:
                    raise Conflict(f"{who} already moderates")
                self._set_row(ctx, "community.role", r["row"], role=1)
            elif action == "remove_moderator":
                if r["role"] != 1:
                    raise Conflict(f"{who} is not a moderator")
                self._set_row(ctx, "community.role", r["row"], role=0)
            else:
                for rr in rows_of(ctx.world, c["n"]):
                    d = row_dict(ctx.world, int(rr))
                    if d["role"] == 2 and d["status"] == MEMBER:
                        self._set_row(ctx, "community.role", int(rr), role=1)
                self._set_row(ctx, "community.role", r["row"], role=2)
        elif action in ("ban", "unban"):
            who = self._who(ctx, str(args.get("member", "")))
            node = nodes.ensure(ctx, who)
            r = member_row(ctx.world, node, c["n"])
            if action == "ban":
                if r is None:
                    CMEMBER.append(ctx, "community.ban", {"node": np.array([node]), "comm": np.array([c["n"]], np.int32),
                                                          "status": np.array([BANNED], np.uint8)})
                else:
                    self._set_row(ctx, "community.ban", r, status=BANNED, role=0, left=ctx.now)
            else:
                if r is None or row_dict(ctx.world, r)["status"] != BANNED:
                    raise Conflict(f"{who} is not banned")
                self._set_row(ctx, "community.unban", r, status=LEFT)
            bus.emit(ctx, "community.moderation", {"community": c["id"], "action": action, "target": who,
                                                   "reason": str(args.get("reason", ""))[:300], "by": ctx.actor,
                                                   "report": None, "until": None}, vis="public")
        elif action == "merge":
            into = self._c(ctx, str(args.get("into", "")))
            self._active(into)
            if into["id"] == c["id"]:
                raise InvalidAction("a community cannot merge into itself")
            mem = members(ctx.world, c["n"])
            if mem.size:
                self.join_many(ctx, mem, np.full(mem.size, into["n"], np.int64), "community")
            bus.emit(ctx, "community.status", {"id": c["id"], "status": "merged", "merged_into": into["id"],
                                               "by": ctx.actor, "reason": str(args.get("reason", ""))[:300]},
                     vis="public")
        elif action == "archive":
            bus.emit(ctx, "community.status", {"id": c["id"], "status": "archived", "merged_into": None,
                                               "by": ctx.actor, "reason": str(args.get("reason", ""))[:300]},
                     vis="public")
        else:
            raise InvalidAction(f"unknown action {action}")
        del st

    def decide_by(self, c: dict) -> str:
        return c["rules"]["governance"]

    def health(self, world, c: dict, weeks: int = 8) -> dict:
        """Sustained voluntary participation, measured from the recorded history."""
        st = world.state
        now = world.now
        v = CMEMBER.view(world)
        mine = v["comm"] == c["n"]
        cur = mine & (v["status"] == MEMBER)
        act7 = cur & (v["last_active"] >= now - 7 * DAY)
        act30 = cur & (v["last_active"] >= now - 30 * DAY)
        inactive = cur & (v["last_active"] < now - c["rules"]["inactive_days"] * DAY)
        old = mine & (v["first_joined"] >= 0) & (v["first_joined"] <= now - 30 * DAY)
        retained = old & (v["status"] == MEMBER) & (v["last_active"] >= now - 30 * DAY)
        contrib: list[set] = [set() for _ in range(weeks)]
        counts = [0] * weeks
        for cid_ in st.find_ids("content", "community", c["id"]):
            r = st.tables["content"][cid_]
            k = (now - r["published_at"]) // WEEK
            if 0 <= k < weeks and not r["removed"]:
                contrib[k].add(r["creator"])
                counts[k] += 1
        items = set(st.find_ids("content", "community", c["id"]))
        for _, cm in st.tables["content_comments"].items():
            if cm["item"] in items and not cm["removed"]:
                k = (now - cm["t"]) // WEEK
                if 0 <= k < weeks:
                    contrib[k].add(cm["author"])
                    counts[k] += 1
        sustained = set.intersection(*contrib[:4]) if weeks >= 4 and all(contrib[:4]) else set()
        return {"members": int(cur.sum()), "active_7d": int(act7.sum()), "active_30d": int(act30.sum()),
                "inactive_members": int(inactive.sum()), "ever_members": int((mine & (v["first_joined"] >= 0)).sum()),
                "left": int((mine & (v["status"] == LEFT)).sum()), "banned": int((mine & (v["status"] == BANNED)).sum()),
                "weekly_contributors": [len(s) for s in contrib], "weekly_contributions": counts,
                "contributors_every_week_4w": len(sustained),
                "retention_30d": round(float(retained.sum()) / float(old.sum()), 4) if old.any() else None,
                "age_days": int((now - c["created"]) // DAY)}

    # ------------------------------------------------------------------ endpoints: founding and membership
    @endpoint("found", S.obj({"id": _COMM, "name": S.s_str(minLength=1, maxLength=80),
                              "subject": S.s_str("what the community is about, in your own words", minLength=3,
                                                 maxLength=500),
                              "description": S.s_str(maxLength=4000),
                              "tags": S.s_arr(S.s_str(pattern=C.TAG_RE, maxLength=32), maxItems=10),
                              "rules": RULE_SCHEMA,
                              "parent": S.s_str("found a spin-off of a community you belong to", maxLength=40)},
                             ["id", "name", "subject"]), cost_s=120)
    def ep_found(self, ctx, id, name, subject, description="", tags=(), rules=None, parent=None):
        """Found a community on a subject you choose. You become its owner; the rules (joining, visibility,
        posting, governance, voting, inactivity) are yours to set and change later through its governance."""
        self.found(ctx, ctx.actor, id, name, subject, description, tags, rules, parent)
        return {"community": id, "url": f"http://community.crucible/c?id={id}"}

    @endpoint("join", S.obj({"community": _COMM, "message": S.s_str(maxLength=500)}, ["community"]), cost_s=5)
    def ep_join(self, ctx, community, message=""):
        """Join a community (open), ask to join (approval) or accept an invitation (invite-only)."""
        c = self._c(ctx, community)
        self._active(c)
        node = nodes.ensure(ctx, ctx.actor)
        r = member_row(ctx.world, node, c["n"])
        st = row_dict(ctx.world, r)["status"] if r is not None else NONE
        if st == MEMBER:
            raise Conflict(f"you are a member of {community}")
        if st == BANNED:
            raise Forbidden(f"you are banned from {community}")
        if st == PENDING:
            raise Conflict("your request is waiting for a moderator")
        if c["rules"]["join"] == "invite" and st != INVITED:
            raise Forbidden(f"{community} admits invited people only")
        if not self.join_many(ctx, np.array([node]), np.array([c["n"]]), "agent"):
            raise Conflict(f"{community} is full")
        new = STATUS[row_dict(ctx.world, member_row(ctx.world, node, c["n"]))["status"]]
        if new == "pending":
            for m in self._mods(ctx.world, c):
                bus.emit(ctx, "community.notice", {"community": community, "what": "request", "who": ctx.actor,
                                                   "by": ctx.actor, "note": str(message)[:500]}, vis=(ctx.actor, m))
        return {"community": community, "status": new}

    def _mods(self, world, c: dict) -> list[str]:
        v = CMEMBER.view(world)
        m = (v["comm"] == c["n"]) & (v["status"] == MEMBER) & (v["role"] >= 1)
        return [p for p in nodes.principals_of_nodes(world, v["node"][m]) if p]

    @endpoint("leave", S.obj({"community": _COMM}, ["community"]), cost_s=2)
    def ep_leave(self, ctx, community):
        """Leave a community (your history stays; you can rejoin)."""
        c = self._c(ctx, community)
        node = nodes.node_of(ctx.world, ctx.actor)
        r = self._row(ctx.world, ctx.actor, c)
        if r is None or r["status"] not in (MEMBER, PENDING):
            raise NotFound(f"you are not a member of {community}")
        if r["role"] == 2 and r["status"] == MEMBER:
            others = [x for x in rows_of(ctx.world, c["n"]) if row_dict(ctx.world, int(x))["role"] == 2 and
                      row_dict(ctx.world, int(x))["status"] == MEMBER and int(x) != r["row"]]
            if not others and members(ctx.world, c["n"]).size > 1:
                raise Conflict("transfer ownership before you leave (community.transfer, or a proposal)")
        self.leave_many(ctx, np.array([node]), np.array([c["n"]]), "left")
        return {"community": community, "status": "left"}

    @endpoint("invite", S.obj({"community": _COMM, "member": _WHO}, ["community", "member"]), cost_s=5)
    def ep_invite(self, ctx, community, member):
        """Invite someone (moderators; any member in open communities)."""
        c = self._c(ctx, community)
        self._active(c)
        role = self._role(ctx.world, ctx.actor, c)
        if role is None or (c["rules"]["join"] != "open" and role == "member"):
            raise Forbidden("only moderators can invite here")
        who = self._who(ctx, member)
        node = nodes.ensure(ctx, who)
        r = member_row(ctx.world, node, c["n"])
        st = row_dict(ctx.world, r)["status"] if r is not None else NONE
        if st in (MEMBER, BANNED, INVITED):
            raise Conflict(f"{who} is {STATUS[st]}")
        if r is None:
            CMEMBER.append(ctx, "community.invite", {"node": np.array([node]), "comm": np.array([c["n"]], np.int32),
                                                     "status": np.array([INVITED], np.uint8)})
        else:
            self._set_row(ctx, "community.invite", r, status=INVITED)
        bus.emit(ctx, "community.notice", {"community": community, "what": "invitation", "who": who, "by": ctx.actor,
                                           "note": ""}, vis=(ctx.actor, who))
        return {"community": community, "invited": who}

    @endpoint("approve", S.obj({"community": _COMM, "member": _WHO, "accept": S.s_bool()},
                               ["community", "member", "accept"]), cost_s=5)
    def ep_approve(self, ctx, community, member, accept):
        """Moderators: admit or refuse a pending join request."""
        c = self._c(ctx, community)
        self._require(ctx, c, ("moderator", "owner"))
        who = self._who(ctx, member)
        r = self._row(ctx.world, who, c)
        if r is None or r["status"] != PENDING:
            raise NotFound(f"{who} has no pending request")
        if accept:
            self._set_row(ctx, "community.invite", r["row"], status=INVITED)
            self.join_many(ctx, np.array([nodes.node_of(ctx.world, who)]), np.array([c["n"]]), "community")
        else:
            self._set_row(ctx, "community.refuse", r["row"], status=LEFT)
        bus.emit(ctx, "community.notice", {"community": community, "what": "approved" if accept else "refused",
                                           "who": who, "by": ctx.actor, "note": ""}, vis=(ctx.actor, who))
        return {"community": community, "member": who, "admitted": bool(accept)}

    @endpoint("transfer", S.obj({"community": _COMM, "member": _WHO}, ["community", "member"]), cost_s=10)
    def ep_transfer(self, ctx, community, member):
        """Owner: hand ownership to another member (you become a moderator)."""
        c = self._c(ctx, community)
        self._require(ctx, c, ("owner",))
        self._execute(ctx, c, "transfer", {"member": member}, None)
        return {"community": community, "owner": self._who(ctx, member)}

    # ------------------------------------------------------------------ endpoints: content and interactions
    @endpoint("post", S.obj({"community": _COMM, "ref": S.s_str("artifact you own", maxLength=71),
                             "title": S.s_str(minLength=1, maxLength=C.MAX_TITLE),
                             "kind": S.s_enum(["post", "image", "article", "thread", "video_short", "audio"]),
                             "tags": C._TAGS, "description": S.s_str(maxLength=C.MAX_DESC)},
                            ["community", "ref", "title"]), cost_s=60)
    def ep_post(self, ctx, community, ref, title, kind="post", tags=(), description=""):
        """Post in a community (a catalogue item on platform `community`; members-only in private communities)."""
        c = self._c(ctx, community)
        access = "members" if c["rules"]["visibility"] == "private" else "public"
        cid = C.publish(ctx, ctx.actor, ref, kind, "community", title, tags=tags, description=description,
                        community=community, access=access)
        return {"id": cid, "community": community}

    @endpoint("feed", S.obj({"community": _COMM, "limit": S.s_int(minimum=1, maximum=50, default=20),
                             "offset": S.s_int(minimum=0, maximum=10_000, default=0)}, ["community"]),
              read_only=True, cost_s=5)
    def ep_feed(self, ctx, community, limit=20, offset=0):
        """A community's posts, newest first (private communities: members only)."""
        c = self._c(ctx, community)
        if c["rules"]["visibility"] == "private" and self._role(ctx.world, ctx.actor, c) is None:
            raise Forbidden(f"{community} is private")
        cm = C.module(ctx.world)
        ids_ = [i for i in ctx.state.find_ids("content", "community", community)
                if C.readable(ctx.world, ctx.actor, ctx.state.tables["content"][i])]
        return {"community": community, "items": cm.listing(ctx.world, ctx.actor, ids_, limit, offset),
                "total": len(ids_)}

    @endpoint("visit", S.obj({"community": _COMM}, ["community"]), cost_s=60)
    def ep_visit(self, ctx, community):
        """Spend time in a community you belong to (counts as taking part; see its feed with community.feed)."""
        c = self._c(ctx, community)
        node = nodes.node_of(ctx.world, ctx.actor)
        if node is None or not self.touch_many(ctx, np.array([node]), np.array([c["n"]])):
            raise NotFound(f"you are not a member of {community}")
        return {"community": community, "last_active": ctx.now}

    @endpoint("kudos", S.obj({"community": _COMM, "member": _WHO, "note": S.s_str(maxLength=500)},
                             ["community", "member"]), cost_s=5)
    def ep_kudos(self, ctx, community, member, note=""):
        """Endorse another member for their contribution (raises their reputation). One per pair per day."""
        c = self._c(ctx, community)
        self._active(c)
        who = self._who(ctx, member)
        if who == ctx.actor:
            raise InvalidAction("you cannot endorse yourself")
        me = self._row(ctx.world, ctx.actor, c)
        them = self._row(ctx.world, who, c)
        if me is None or me["status"] != MEMBER or them is None or them["status"] != MEMBER:
            raise Forbidden("kudos are between current members")
        if trust_level(reputation(me)) < c["rules"]["min_trust_kudos"]:
            raise Forbidden("your trust level is too low for kudos here")
        key = f"{c['id']}|{ctx.actor}|{who}|{ctx.now // DAY}"
        if ctx.state.has("community_history", key):
            raise Conflict("you already endorsed this member today")
        bus.emit(ctx, "community.kudos", {"community": c["id"], "from": ctx.actor, "to": who,
                                          "note": str(note)[:500]}, vis="public")
        CMEMBER.update(ctx, "community.kudos", np.array([me["row"], them["row"]]),
                       {"kudos_given": np.array([1, 0], np.int32), "kudos_recv": np.array([0, 1], np.int32)},
                       mode="add")
        self._bump(ctx, ctx.actor, c, "community.kudos_given")
        return {"to": who, "their_reputation": reputation(dict(row_dict(ctx.world, them["row"])))}

    @endpoint("event", S.obj({"community": _COMM, "title": S.s_str(minLength=1, maxLength=200),
                              "description": S.s_str(maxLength=4000),
                              "in_s": S.s_int("starts this many seconds from now", minimum=0, maximum=90 * DAY),
                              "duration_s": S.s_int(minimum=600, maximum=7 * DAY, default=3600)},
                             ["community", "title", "in_s"]), cost_s=30)
    def ep_event(self, ctx, community, title, in_s, description="", duration_s=3600):
        """Members: schedule a shared activity (a meeting, a session, a challenge window) for the community."""
        c = self._c(ctx, community)
        self._active(c)
        r = self._row(ctx.world, ctx.actor, c)
        if r is None or r["status"] != MEMBER:
            raise Forbidden("members schedule events")
        eid = ctx.state.new_id("cev")
        bus.emit(ctx, "community.event", {"id": eid, "community": community, "title": str(title)[:200],
                                          "description": str(description)[:4000], "at": ctx.now + int(in_s),
                                          "duration_s": int(duration_s), "by": ctx.actor}, vis="public")
        return {"event": eid, "at": ctx.now + int(in_s)}

    @endpoint("events", S.obj({"community": _COMM, "upcoming": S.s_bool(default=True)}, ["community"]),
              read_only=True)
    def ep_events(self, ctx, community, upcoming=True):
        """A community's events with RSVP and attendance numbers."""
        self._c(ctx, community)
        st = ctx.state
        out = []
        for eid in st.find_ids("community_events", "community", community):
            e = st.tables["community_events"][eid]
            if upcoming and e["at"] + e["duration_s"] < ctx.now:
                continue
            out.append({"event": eid, "title": e["title"], "description": e["description"], "at": e["at"],
                        "duration_s": e["duration_s"], "by": e["by"], "going": e["going"], "attended": e["attended"]})
        return sorted(out, key=lambda e: (e["at"], e["event"]))

    @endpoint("rsvp", S.obj({"event": S.s_str(maxLength=20), "going": S.s_bool(default=True)}, ["event"]), cost_s=2)
    def ep_rsvp(self, ctx, event, going=True):
        """Say whether you will take part in an event."""
        e = ctx.state.get("community_events", event)
        if e is None:
            raise NotFound(f"no event {event}")
        c = self._c(ctx, e["community"])
        r = self._row(ctx.world, ctx.actor, c)
        if r is None or r["status"] != MEMBER:
            raise Forbidden("members answer event invitations")
        bus.emit(ctx, "community.rsvp", {"event": event, "member": ctx.actor, "going": bool(going)}, vis="public")
        return {"event": event, "going": bool(going)}

    @endpoint("attend", S.obj({"event": S.s_str(maxLength=20)}, ["event"]), cost_s=1800)
    def ep_attend(self, ctx, event):
        """Take part in an event while it runs (it costs your time)."""
        e = ctx.state.get("community_events", event)
        if e is None:
            raise NotFound(f"no event {event}")
        if not e["at"] <= ctx.now < e["at"] + e["duration_s"]:
            raise Conflict("the event is not running now")
        if ctx.state.has("community_rsvps", f"{event}|{ctx.actor}|attended"):
            raise Conflict("you attended already")
        c = self._c(ctx, e["community"])
        r = self._row(ctx.world, ctx.actor, c)
        if r is None or r["status"] != MEMBER:
            raise Forbidden("members attend events")
        bus.emit(ctx, "community.attend", {"event": event, "member": ctx.actor}, vis="public")
        self._bump(ctx, ctx.actor, c, "community.attended", attended=1)
        return {"event": event, "attended": True}

    # ------------------------------------------------------------------ endpoints: governance
    @endpoint("propose", S.obj({"community": _COMM, "action": S.s_enum(list(ACTIONS)),
                                "args": S.s_any_obj("rules: {...} | name/description/subject/tags | member | into | "
                                                    "reason"),
                                "reason": S.s_str(maxLength=2000)}, ["community", "action"]), cost_s=60)
    def ep_propose(self, ctx, community, action, args=None, reason=""):
        """Propose a change. Under `owner` governance the owner's proposal takes effect at once; under
        `moderators` any moderator's does; under `vote` members vote (quorum of active members, threshold, window).
        Members below the community's trust level for proposals cannot propose."""
        c = self._c(ctx, community)
        self._active(c)
        role = self._role(ctx.world, ctx.actor, c)
        if role is None:
            raise Forbidden("members make proposals")
        if self.trust(ctx.world, ctx.actor, c)["level"] < c["rules"]["min_trust_propose"] and role == "member":
            raise Forbidden("your trust level is too low to make proposals here")
        mode = self.decide_by(c)
        args = dict(args or {})
        immediate = (mode == "owner" and role == "owner") or (mode == "moderators" and role in ("moderator", "owner"))
        pid = ctx.state.new_id("cprop")
        electorate = int(members(ctx.world, c["n"]).size)
        bus.emit(ctx, "community.proposal", {"id": pid, "community": community, "action": action, "args": args,
                                             "by": ctx.actor, "reason": str(reason)[:2000],
                                             "deadline": ctx.now + int(c["rules"]["vote_window_s"]),
                                             "electorate": electorate}, vis="public")
        if immediate:
            self._execute(ctx, c, action, args, pid)
            bus.emit(ctx, "community.decided", {"proposal": pid, "outcome": "approved", "yes": 1, "no": 0,
                                                "electorate": 1, "note": f"decided by the {role}"}, vis="public")
            return {"proposal": pid, "status": "approved"}
        if mode != "vote":
            bus.emit(ctx, "community.decided", {"proposal": pid, "outcome": "rejected", "yes": 0, "no": 0,
                                                "electorate": 0,
                                                "note": f"under {mode} governance only the {mode} decide"},
                     vis="public")
            return {"proposal": pid, "status": "rejected"}
        bus.emit(ctx, "community.vote", {"proposal": pid, "voter": ctx.actor, "approve": True}, vis="public")
        self._try_decide(ctx, pid)
        return {"proposal": pid, "status": ctx.state.tables["community_proposals"][pid]["status"]}

    @endpoint("vote", S.obj({"proposal": S.s_str(maxLength=20), "approve": S.s_bool()}, ["proposal", "approve"]),
              cost_s=5)
    def ep_vote(self, ctx, proposal, approve):
        """Vote on an open proposal of a community you belong to (one vote per member; changeable while open)."""
        p = ctx.state.get("community_proposals", proposal)
        if p is None:
            raise NotFound(f"no proposal {proposal}")
        if p["status"] != "open":
            raise Conflict(f"proposal is {p['status']}")
        c = self._c(ctx, p["community"])
        if self._role(ctx.world, ctx.actor, c) is None:
            raise Forbidden("members vote")
        bus.emit(ctx, "community.vote", {"proposal": proposal, "voter": ctx.actor, "approve": bool(approve)},
                 vis="public")
        self._try_decide(ctx, proposal)
        return {"proposal": proposal, "status": ctx.state.tables["community_proposals"][proposal]["status"]}

    def _tally(self, world, pid: str) -> tuple[int, int]:
        yes = no = 0
        for vid in world.state.find_ids("community_votes", "proposal", pid):
            if world.state.tables["community_votes"][vid]["approve"]:
                yes += 1
            else:
                no += 1
        return yes, no

    def _try_decide(self, ctx: Ctx, pid: str, final: bool = False) -> None:
        p = dict(ctx.state.tables["community_proposals"][pid])
        if p["status"] != "open":
            return
        c = self._c(ctx, p["community"])
        rules = c["rules"]
        yes, no = self._tally(ctx.world, pid)
        active = int(members(ctx.world, c["n"], active_days=rules["inactive_days"]).size)
        quorum = max(1, math.ceil(rules["quorum"] * max(1, active)))
        total = int(members(ctx.world, c["n"]).size)
        decided = None
        if yes + no >= quorum and yes > rules["vote_threshold"] * max(1, total):
            decided = "approved"                          # a majority of all members: no later vote can change it
        elif no >= (1 - rules["vote_threshold"]) * max(1, total) and no > 0:
            decided = "rejected"
        elif final or ctx.now >= p["deadline"]:
            if yes + no < quorum:
                decided = "expired"
            else:
                decided = "approved" if yes > rules["vote_threshold"] * (yes + no) else "rejected"
        if decided is None:
            return
        note = f"{yes} for, {no} against, quorum {quorum} of {active} active"
        if decided == "approved":
            from ..core.errors import CrucibleError, contain
            err = contain(ctx, self._execute, ctx, c, p["action"], dict(p["args"]), pid) if c["status"] == "active" \
                else Conflict("community no longer active")
            if err is not None:
                decided, note = "failed", f"{note}; not carried out: {err if isinstance(err, CrucibleError) else type(err).__name__}"
        bus.emit(ctx, "community.decided", {"proposal": pid, "outcome": decided, "yes": yes, "no": no,
                                            "electorate": total, "note": note[:300]}, vis="public")

    @endpoint("proposals", S.obj({"community": _COMM, "status": S.s_enum(["open", "any"])}, ["community"]),
              read_only=True)
    def ep_proposals(self, ctx, community, status="open"):
        """A community's proposals with their tallies."""
        self._c(ctx, community)
        st = ctx.state
        out = []
        for pid in st.find_ids("community_proposals", "community", community):
            p = st.tables["community_proposals"][pid]
            if status == "open" and p["status"] != "open":
                continue
            yes, no = self._tally(ctx.world, pid)
            out.append({"proposal": pid, "action": p["action"], "args": dict(p["args"]), "by": p["by"],
                        "reason": p["reason"], "status": p["status"], "deadline": p["deadline"], "yes": yes,
                        "no": no, "note": p.get("note", "")})
        return out

    # ------------------------------------------------------------------ endpoints: moderation
    @endpoint("report", S.obj({"community": _COMM, "target": S.s_str("content id c<n>, comment id cm_..., or a "
                                                                      "member principal", maxLength=60),
                               "reason": S.s_str(minLength=1, maxLength=1000)}, ["community", "target", "reason"]),
              cost_s=10)
    def ep_report(self, ctx, community, target, reason):
        """Report a post, a comment or a member to the community's moderators."""
        c = self._c(ctx, community)
        if self._role(ctx.world, ctx.actor, c) is None:
            raise Forbidden("members report to moderators")
        kind = self._target_kind(ctx, c, target)
        rid = ctx.state.new_id("crep")
        bus.emit(ctx, "community.report", {"id": rid, "community": community, "target": target, "target_kind": kind,
                                           "reason": str(reason)[:1000], "by": ctx.actor},
                 vis=tuple(sorted(set(self._mods(ctx.world, c)) | {ctx.actor})))
        return {"report": rid}

    def _target_kind(self, ctx: Ctx, c: dict, target: str) -> str:
        st = ctx.state
        if C.CID_RE.fullmatch(target):
            rec = st.get("content", target)
            if rec is None or rec["community"] != c["id"]:
                raise NotFound(f"{target} is not in {c['id']}")
            return "item"
        if target.startswith("cm_"):
            cm = st.get("content_comments", target)
            if cm is None or st.tables["content"][cm["item"]]["community"] != c["id"]:
                raise NotFound(f"{target} is not in {c['id']}")
            return "comment"
        who = self._who(ctx, target)
        if self._row(ctx.world, who, c) is None:
            raise NotFound(f"{who} is not in {c['id']}")
        return "member"

    @endpoint("reports", S.obj({"community": _COMM, "status": S.s_enum(["open", "any"])}, ["community"]),
              read_only=True)
    def ep_reports(self, ctx, community, status="open"):
        """Moderators: the report queue."""
        c = self._c(ctx, community)
        self._require(ctx, c, ("moderator", "owner"))
        st = ctx.state
        return [dict(st.tables["community_reports"][r], report=r) for r in st.find_ids("community_reports",
                                                                                         "community", community)
                if status == "any" or st.tables["community_reports"][r]["status"] == "open"]

    @endpoint("moderate", S.obj({"community": _COMM, "action": S.s_enum(list(MOD_ACTIONS)),
                                 "target": S.s_str("content id, comment id or member", maxLength=60),
                                 "reason": S.s_str(maxLength=1000), "report": S.s_str(maxLength=20),
                                 "days": S.s_int("mute length", minimum=1, maximum=365, default=7)},
                                ["community", "action", "target"]), cost_s=10)
    def ep_moderate(self, ctx, community, action, target, reason="", report=None, days=7):
        """Moderators: remove a post or comment, warn, mute or ban a member, or dismiss a report. Public log."""
        c = self._c(ctx, community)
        self._require(ctx, c, ("moderator", "owner"))
        if report is not None:
            rp = ctx.state.get("community_reports", report)
            if rp is None or rp["community"] != community or rp["status"] != "open":
                raise NotFound(f"no open report {report}")
        until = None
        if action == "dismiss":
            if report is None:
                raise InvalidAction("dismiss needs a report")
        else:
            kind = self._target_kind(ctx, c, target)
            if action == "remove":
                if kind == "item":
                    rec = ctx.state.tables["content"][target]
                    C.module(ctx.world).remove(ctx, target, reason, role="moderator")
                    self._bump(ctx, rec["creator"], c, "community.removed", removed=1)
                elif kind == "comment":
                    cm = ctx.state.tables["content_comments"][target]
                    C.module(ctx.world).remove_comment(ctx, target, reason, role="moderator")
                    self._bump(ctx, cm["author"], c, "community.removed", removed=1)
                else:
                    raise InvalidAction("remove applies to posts and comments; ban a member")
            else:
                if kind != "member":
                    raise InvalidAction(f"{action} applies to members")
                who = self._who(ctx, target)
                r = self._row(ctx.world, who, c)
                if r["role"] >= ROLE_CODE[self._role(ctx.world, ctx.actor, c)] and ctx.actor != SYSTEM:
                    raise Forbidden("you cannot moderate someone of your rank or above")
                if action == "warn":
                    self._bump(ctx, who, c, "community.warned", warnings=1)
                elif action == "mute":
                    until = ctx.now + int(days) * DAY
                    self._set_row(ctx, "community.mute", r["row"], muted_until=until)
                elif action == "ban":
                    self._set_row(ctx, "community.ban", r["row"], status=BANNED, role=0, left=ctx.now)
        bus.emit(ctx, "community.moderation", {"community": community, "action": action, "target": target,
                                               "reason": str(reason)[:1000], "by": ctx.actor, "report": report,
                                               "until": until}, vis="public")
        return {"community": community, "action": action, "target": target}

    @endpoint("modlog", S.obj({"community": _COMM, "limit": S.s_int(minimum=1, maximum=200, default=50)},
                              ["community"]), read_only=True)
    def ep_modlog(self, ctx, community, limit=50):
        """The public moderation log of a community, newest first."""
        self._c(ctx, community)
        st = ctx.state
        rows = [dict(st.tables["community_modlog"][k], entry=k) for k in st.find_ids("community_modlog", "community",
                                                                                      community)]
        rows.sort(key=lambda r: (-r["t"], r["entry"]))
        return rows[:limit]

    # ------------------------------------------------------------------ endpoints: views
    def _summary(self, world, c: dict) -> dict:
        v = CMEMBER.view(world)
        m = (v["comm"] == c["n"]) & (v["status"] == MEMBER)
        act = m & (v["last_active"] >= world.now - c["rules"]["inactive_days"] * DAY)
        return {"community": c["id"], "name": c["name"], "subject": c["subject"], "tags": list(c["tags"]),
                "status": c["status"], "members": int(m.sum()), "active_members": int(act.sum()),
                "join": c["rules"]["join"], "visibility": c["rules"]["visibility"], "created": c["created"],
                "founder": c["founder"], "parent": c["parent"]}

    @endpoint("directory", S.obj({"q": S.s_str("words of the name, subject or tags", maxLength=200),
                                  "sort": S.s_enum(["active", "members", "new"]),
                                  "limit": S.s_int(minimum=1, maximum=100, default=20)}), read_only=True, cost_s=2)
    def ep_directory(self, ctx, q="", sort="active", limit=20):
        """Active communities, by recent activity, size or age (optionally matching words)."""
        from ..services.search import tokenize
        want = set(tokenize(q))
        rows = []
        for cid_ in ctx.state.find_ids("communities", "status", "active"):
            c = dict(ctx.state.tables["communities"][cid_])
            if want and not want & set(tokenize(f"{c['name']} {c['subject']} {' '.join(c['tags'])}")):
                continue
            rows.append(self._summary(ctx.world, c))
        key = {"active": lambda r: (-r["active_members"], -r["members"], r["community"]),
               "members": lambda r: (-r["members"], r["community"]),
               "new": lambda r: (-r["created"], r["community"])}[sort]
        return sorted(rows, key=key)[:limit]

    @endpoint("view", S.obj({"community": _COMM}, ["community"]), read_only=True, cost_s=2)
    def ep_view(self, ctx, community):
        """A community: subject, rules, owners and moderators, health, and your membership and trust."""
        c = self._c(ctx, community)
        out = self._summary(ctx.world, c)
        out.update(description=c["description"], rules=dict(c["rules"]), rules_version=c["rules_version"],
                   merged_into=c["merged_into"], moderators=self._mods(ctx.world, c),
                   health=self.health(ctx.world, c))
        r = self._row(ctx.world, ctx.actor, c)
        if r is not None:
            rep = reputation(r)
            out["you"] = {"status": STATUS[r["status"]], "role": ROLE[r["role"]], "reputation": rep,
                          "trust_level": trust_level(rep), "posts": r["posts"], "comments": r["comments"],
                          "kudos_received": r["kudos_recv"], "attended": r["attended"], "warnings": r["warnings"]}
        return out

    @endpoint("members", S.obj({"community": _COMM, "active_only": S.s_bool(default=False),
                                "limit": S.s_int(minimum=1, maximum=500, default=100)}, ["community"]),
              read_only=True, cost_s=2)
    def ep_members(self, ctx, community, active_only=False, limit=100):
        """Members with role, tenure, last activity, contributions and trust level."""
        c = self._c(ctx, community)
        if c["rules"]["visibility"] == "private" and self._role(ctx.world, ctx.actor, c) is None:
            raise Forbidden(f"{community} is private")
        out = []
        for r in rows_of(ctx.world, c["n"]):
            d = row_dict(ctx.world, int(r))
            if d["status"] != MEMBER:
                continue
            if active_only and d["last_active"] < ctx.now - c["rules"]["inactive_days"] * DAY:
                continue
            rep = reputation(d)
            out.append({"member": nodes.principal_of(ctx.world, d["node"]), "role": ROLE[d["role"]],
                        "joined": d["joined"], "first_joined": d["first_joined"], "spells": d["spells"],
                        "last_active": d["last_active"], "posts": d["posts"], "comments": d["comments"],
                        "kudos_received": d["kudos_recv"], "reputation": rep, "trust_level": trust_level(rep),
                        "inactive": d["last_active"] < ctx.now - c["rules"]["inactive_days"] * DAY})
        out.sort(key=lambda m: (-m["reputation"], m["member"] or ""))
        return {"community": community, "members": out[:limit], "count": len(out)}

    @endpoint("mine", S.obj(), read_only=True, cost_s=1)
    def ep_mine(self, ctx):
        """Communities you belong to (or have asked to join, or are invited to)."""
        node = nodes.node_of(ctx.world, ctx.actor)
        if node is None:
            return []
        out = []
        for r in rows_of(ctx.world, node=node):
            d = row_dict(ctx.world, int(r))
            if d["status"] in (MEMBER, PENDING, INVITED):
                c = by_index(ctx.world, d["comm"])
                out.append(dict(self._summary(ctx.world, c), your_status=STATUS[d["status"]], role=ROLE[d["role"]]))
        return out

    @endpoint("health", S.obj({"community": _COMM, "weeks": S.s_int(minimum=4, maximum=52, default=8)},
                              ["community"]), read_only=True, cost_s=2)
    def ep_health(self, ctx, community, weeks=8):
        """Sustained participation: active members, weekly contributors, retention of members after 30 days."""
        return self.health(ctx.world, self._c(ctx, community), weeks)

    # ------------------------------------------------------------------ processes
    @process("daily")
    def pr_daily(self, ctx):
        """Prune members inactive for longer than a community's `prune_days` (when it sets one)."""
        v = CMEMBER.view(ctx.world)
        for cid_ in ctx.state.find_ids("communities", "status", "active"):
            c = dict(ctx.state.tables["communities"][cid_])
            pd = int(c["rules"].get("prune_days") or 0)
            if pd <= 0:
                continue
            m = (v["comm"] == c["n"]) & (v["status"] == MEMBER) & (v["role"] == 0) & \
                (v["last_active"] < ctx.now - pd * DAY)
            if m.any():
                rows = np.flatnonzero(m)
                CMEMBER.update(ctx, "community.prune", rows, {"status": np.full(rows.size, LEFT, np.uint8),
                                                              "left": np.full(rows.size, ctx.now, np.int64)},
                               extra={"community": cid_, "reason": "inactive"})

    @process("resolve")
    def pr_resolve(self, ctx):
        """Close proposals whose voting window has ended."""
        for pid in ctx.state.find_ids("community_proposals", "status", "open"):
            if ctx.state.tables["community_proposals"][pid]["deadline"] <= ctx.now:
                self._try_decide(ctx, pid, final=True)

    # ------------------------------------------------------------------ appliers
    @applier("community.founded")
    def ap_founded(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("communities", p["id"], {"id": p["id"], "n": p["n"], "name": p["name"], "subject": p["subject"],
                                           "description": p.get("description", ""), "tags": list(p.get("tags", [])),
                                           "founder": p["founder"], "rules": dict(p["rules"]), "rules_version": 1,
                                           "parent": p.get("parent"), "status": "active", "merged_into": None,
                                           "created": ev.t})
        st.insert("community_index", str(p["n"]), {"id": p["id"]})
        st.insert("community_history", f"{p['id']}|{ev.seq}", {"community": p["id"], "what": "founded", "t": ev.t,
                                                               "by": p["founder"]})

    @applier("community.rules")
    def ap_rules(self, ev):
        p = ev.payload
        st = self.world.state
        st.patch("communities", p["id"], rules=dict(p["rules"]), rules_version=p["version"])
        st.insert("community_history", f"{p['id']}|{ev.seq}", {"community": p["id"], "what": "rules", "t": ev.t,
                                                               "by": p["by"], "version": p["version"]})

    @applier("community.info")
    def ap_info(self, ev):
        p = ev.payload
        st = self.world.state
        st.patch("communities", p["id"], **{k: (list(p[k]) if k == "tags" else p[k])
                                            for k in ("name", "description", "subject", "tags") if k in p})
        st.insert("community_history", f"{p['id']}|{ev.seq}", {"community": p["id"], "what": "info", "t": ev.t,
                                                               "by": p["by"]})

    @applier("community.status")
    def ap_status(self, ev):
        p = ev.payload
        st = self.world.state
        st.patch("communities", p["id"], status=p["status"], merged_into=p.get("merged_into"))
        st.insert("community_history", f"{p['id']}|{ev.seq}", {"community": p["id"], "what": p["status"],
                                                               "t": ev.t, "by": p["by"]})

    @applier("community.proposal")
    def ap_proposal(self, ev):
        p = ev.payload
        self.world.state.insert("community_proposals", p["id"], {
            "community": p["community"], "action": p["action"], "args": dict(p["args"]), "by": p["by"],
            "reason": p.get("reason", ""), "deadline": p["deadline"], "electorate": p["electorate"],
            "status": "open", "t": ev.t, "note": ""})

    @applier("community.vote")
    def ap_vote(self, ev):
        p = ev.payload
        self.world.state.put("community_votes", f"{p['proposal']}|{p['voter']}",
                             {"proposal": p["proposal"], "voter": p["voter"], "approve": p["approve"], "t": ev.t})

    @applier("community.decided")
    def ap_decided(self, ev):
        p = ev.payload
        self.world.state.patch("community_proposals", p["proposal"], status=p["outcome"], decided=ev.t,
                               note=p.get("note", ""))

    @applier("community.report")
    def ap_report(self, ev):
        p = ev.payload
        self.world.state.insert("community_reports", p["id"], {"community": p["community"], "target": p["target"],
                                                               "target_kind": p["target_kind"], "reason": p["reason"],
                                                               "by": p["by"], "status": "open", "t": ev.t})

    @applier("community.moderation")
    def ap_moderation(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("community_modlog", f"cmod_{ev.seq}", {"community": p["community"], "action": p["action"],
                                                         "target": p["target"], "reason": p.get("reason", ""),
                                                         "by": p["by"], "report": p.get("report"),
                                                         "until": p.get("until"), "t": ev.t})
        if p.get("report"):
            st.patch("community_reports", p["report"], status="dismissed" if p["action"] == "dismiss" else "acted",
                     handled_by=p["by"])

    @applier("community.event")
    def ap_event(self, ev):
        p = ev.payload
        self.world.state.insert("community_events", p["id"], {"community": p["community"], "title": p["title"],
                                                              "description": p.get("description", ""), "at": p["at"],
                                                              "duration_s": p["duration_s"], "by": p["by"],
                                                              "going": 0, "attended": 0, "t": ev.t})

    @applier("community.rsvp")
    def ap_rsvp(self, ev):
        p = ev.payload
        st = self.world.state
        key = f"{p['event']}|{p['member']}"
        old = st.tables["community_rsvps"].get(key)
        e = st.tables["community_events"][p["event"]]
        going = e["going"] - (1 if old and old["going"] else 0) + (1 if p["going"] else 0)
        st.put("community_rsvps", key, {"event": p["event"], "member": p["member"], "going": p["going"], "t": ev.t})
        st.patch("community_events", p["event"], going=going)

    @applier("community.attend")
    def ap_attend(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("community_rsvps", f"{p['event']}|{p['member']}|attended", {"event": p["event"],
                                                                              "member": p["member"], "t": ev.t,
                                                                              "attended": True})
        e = st.tables["community_events"][p["event"]]
        st.patch("community_events", p["event"], attended=e["attended"] + 1)

    @applier("community.kudos")
    def ap_kudos(self, ev):
        p = ev.payload
        self.world.state.insert("community_history", f"{p['community']}|{p['from']}|{p['to']}|{ev.t // DAY}",
                                {"community": p["community"], "what": "kudos", "t": ev.t, "by": p["from"],
                                 "to": p["to"]})

    @applier("community.notice")
    def ap_notice(self, ev):
        return None

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        from .edges import pair_keys
        v = CMEMBER.view(world)
        if v["node"].size:
            k = pair_keys(v["node"], v["comm"].astype(np.int64))
            if np.unique(k).size != k.size:
                errs.append("more than one membership row for a (person, community) pair")
        n_c = len(world.state.tables["communities"])
        if v["comm"].size and (int(v["comm"].min()) < 0 or int(v["comm"].max()) >= n_c):
            errs.append("membership of an unknown community")
        if np.any((v["status"] == MEMBER) & (v["joined"] < 0)):
            errs.append("a member without a join time")
        if np.any((v["status"] == MEMBER) & (v["last_active"] < v["joined"])):
            errs.append("a member active before joining")
        # participation counters equal the recorded content and comments
        st = world.state
        posts: dict[tuple, int] = {}
        for _, r in st.tables["content"].items():
            if r["community"]:
                key = (r["creator_node"], st.tables["communities"][r["community"]]["n"])
                posts[key] = posts.get(key, 0) + 1
        for (node, cn), cnt in posts.items():
            row = member_row(world, node, cn)
            if row is not None and v["posts"][row] > cnt:     # posts made before joining are not member posts
                errs.append(f"posts counter of node {node} in community #{cn} is {v['posts'][row]}, recorded {cnt}")
        if v["posts"].size and (np.any(v["posts"] < 0) or np.any(v["comments"] < 0) or np.any(v["kudos_recv"] < 0)):
            errs.append("negative participation counter")
        for cid_, c in st.tables["communities"].items():
            owners = (v["comm"] == c["n"]) & (v["status"] == MEMBER) & (v["role"] == 2)
            if c["status"] == "active" and int(owners.sum()) > 1:
                errs.append(f"community {cid_} has {int(owners.sum())} owners")
        return errs

    def observe(self, world, actor):
        node = nodes.node_of(world, actor)
        if node is None:
            return None
        m = memberships(world, node)
        return {"communities": int(m.size)} if m.size else None
