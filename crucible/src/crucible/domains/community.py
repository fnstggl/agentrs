"""Tier 2 domain 9.4 - Community ecosystem (MASTER_SPEC Part 9.4; workstream I-1).

Example objective: build a community with sustained voluntary participation.

A configuration is the civilisation's communities slice (D-17, `base.civ_recipe`): the community module
(`social.communities`: persistent memberships, posts and comments, kudos, events, trust levels, moderation,
governance, rule versions, spin-offs, merges), the content catalogue and its distribution (`content`, `aud`, `feed`
on the community platform and the short-post platform), the follow/share graph, optionally direct messaging (`msg`),
and the ONE population, which takes part through two activities:

* `social.community` (`social.popsocial`, CORE-POP): visit one's communities, read, post, reply, report, moderate,
  join from the directory, found communities about what one has been reading, leave when it stops paying off;
* `community.civic` (module `commlife`, defined here; fidelity record FR-COMMLIFE-1, plausible, unvalidated): the
  civic side of membership - vote on open proposals by one's own experienced value and preferences, propose changes
  when one sees a problem one can name (a moderation backlog, removals, an absent owner, an exclusive door that
  keeps people out), schedule shared activities (owners and moderators of lively communities), answer invitations to
  them and attend while they run, invite and message one's contacts about a community one values, and - when the
  owner of an owner-ruled community has been absent for a week, so that nobody else can change anything - found a
  new home for its members and move there (members of the abandoned community follow). Every choice is a recorded
  population decision (model `commlife@1`).

Configurations differ in structure measured on the built world: whether communities exist at genesis (founded by
population members through the ordinary API) or only grow from the population's own founding, their governance
modes and join policies, whether spin-offs nest under parents, and whether direct messaging exists.

Evaluator (hidden): for each community the seat founded, `community.health`-style measures computed from the
recorded membership history: members active in the last 7 days, weekly contributors over the last weeks, retention
of members who joined at least 14 days before the end (still members and active in the last 7 days), and the
share of the seat's own activity (participation must be voluntary: the founder's own posts do not count).

Reference (hidden, LABELLED SCRIPTED): a tool-only organiser that reads the directory and the feed, founds a
community about the most-read subject it observes, writes regular posts there, schedules events, answers members
and moderates reports.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, register_module
from ..substrate import columns, ids
from . import _t2kit as K
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, civ_recipe, register

DAY = 86400
WEEK = 7 * DAY
DECISION = {"model_id": "commlife", "model_version": "1", "fidelity_id": "FR-COMMLIFE-1"}
CHOICE = {"none": 0, "vote": 1, "propose": 2, "event": 3, "rsvp": 4, "attend": 5, "invite": 6, "message": 7}
LIFE_DEFAULTS = {"prior": -0.6, "seed": None, "invite_p": 0.25, "event_min_active": 4, "event_every_days": 7,
                 "propose_backlog": 3, "propose_removals": 2}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


@register_module
class CommunityLife(Module):
    """The population's civic participation in communities (see the module doc)."""
    name = "commlife"
    host = "commlife"
    depends = ("identity", "pop", "cols", "community", "social_ids")

    def cfg(self, k):
        return self.config.get(k, LIFE_DEFAULTS[k])

    def install(self, world):
        from .. import population
        n = population.n_individuals(world)
        columns.declare(world, "commlife", n, {"choice": ("int16", (), -1), "acts": ("int32", (), 0)})
        world.state.declare_table("commlife_succ")
        population.register_activity(world, "community.civic", self.act_civic, owner="commlife",
                                     prior=float(self.cfg("prior")), goals=(3,))

    # ------------------------------------------------------------------ genesis: communities founded by members
    def genesis(self, ctx):
        """Seeded configurations: communities founded at genesis by population members through the ordinary API
        (`community.found`), on the topic nearest to the founder's own interests, with the configuration's rules
        (and, nested: spin-offs under a parent)."""
        seed = self.cfg("seed")
        if not seed:
            return
        from .. import population
        from ..population import semantics
        w = ctx.world
        n = population.n_individuals(w)
        rng = w.rng_for("commlife-genesis")
        f = population.features(w, np.arange(n))
        prefs, skills = f["prefs"].astype(np.float64), f["skills"].astype(np.float64)
        score = prefs[:, 6] + skills[:, 1] + skills[:, 5] + rng.normal(0, 0.1, n)
        k = int(seed.get("communities", 4))
        founders = [int(i) for i in np.argsort(-score, kind="stable")[: k * 3]]
        names = semantics.topic_names()
        coords = population.interest_coords(w, np.array(founders, np.int64))
        topics = semantics.nearest_topic(coords)
        made, used = [], set()
        for j, i in enumerate(founders):
            if len(made) >= k:
                break
            topic = str(names[int(topics[j]) % len(names)])
            if topic in used:
                continue
            used.add(topic)
            p = ids.ensure_principal(ctx, i)
            slug = "".join(ch if ch.isalnum() else "-" for ch in topic.lower())[:30].strip("-") + "-club"
            tag = "".join(ch for ch in topic.lower() if ch.isalnum() or ch == "-")[:30] or "topic"
            ctx.call("community.found", {"id": slug, "name": f"{topic.replace('_', ' ').title()} club"[:80],
                                         "subject": f"people who care about {topic}",
                                         "description": f"A community about {topic}.", "tags": [tag],
                                         "rules": dict(seed.get("rules") or {})}, as_principal=p)
            made.append((slug, p))

    # ------------------------------------------------------------------ helpers
    def _call(self, ctx, p, tool, args):
        try:
            return ctx.call(tool, args, as_principal=p)
        except CrucibleError:
            return None

    # ------------------------------------------------------------------ the civic session
    def act_civic(self, ctx, idx: np.ndarray, now: int):
        from ..social import communities as CM
        w = ctx.world
        st = ctx.state
        util = np.full(idx.size, -0.05)
        if "communities" not in st.tables:
            return util
        rng = ctx.rng("commlife", now)
        prefs = columns.col(w, "pop", "prefs")[idx].astype(np.float64)
        contacts = columns.col(w, "pop", "contacts")[idx].astype(np.int64)
        left = columns.col(w, "pop", "attention_left_min")[idx].astype(np.float64)
        v = CM.CMEMBER.view(w)
        dec_i, dec_c = [], []
        for j, i in enumerate(idx.tolist()):
            rows = np.flatnonzero((v["node"] == int(i)) & (v["status"] == CM.MEMBER))
            if rows.size == 0:
                continue
            vals = v["value"][rows].astype(np.float64) + rng.gumbel(size=rows.size) * 0.1
            r = int(rows[int(np.argmax(vals))])
            try:
                c = CM.by_index(w, int(v["comm"][r]))
            except CrucibleError:
                continue
            if c.get("status") != "active":
                continue
            p = ids.ensure_principal(ctx, int(i))
            value = float(v["value"][r])
            role = int(v["role"][r])
            choice, u = self._civic(ctx, p, int(i), c, role, value, prefs[j], contacts[j], float(left[j]), rng, now)
            util[j] = u
            dec_i.append(int(i))
            dec_c.append(CHOICE[choice])
        if dec_i:
            from .. import population
            population.record_decisions(ctx, "commlife.decide", np.array(dec_i, np.int64),
                                        {"choice": np.array(dec_c, np.int16)}, table="commlife", **DECISION)
        return util

    def _civic(self, ctx, p, i, c, role, value, pr, contacts, left, rng, now) -> tuple[str, float]:
        cid = c["id"]
        # 1. a running event one said yes to: take part
        evs = self._call(ctx, p, "community.events", {"community": cid, "upcoming": True}) or []
        for e in evs:
            rs = ctx.state.get("community_rsvps", f"{e['event']}|{p}")
            if e["at"] <= now < e["at"] + e["duration_s"] and rs is not None and rs.get("going") and left >= 30 \
                    and not ctx.state.has("community_rsvps", f"{e['event']}|{p}|attended"):
                if self._call(ctx, p, "community.attend", {"event": e["event"]}) is not None:
                    return "attend", 0.35 + 0.3 * float(pr[6])
        # 2. open proposals: vote by one's own judgement
        props = self._call(ctx, p, "community.proposals", {"community": cid, "status": "open"}) or []
        for pp in props:
            if ctx.state.has("community_votes", f"{pp['proposal']}|{p}"):
                continue
            yes = self._judge(ctx, p, c, pp, value, pr, rng)
            if self._call(ctx, p, "community.vote", {"proposal": pp["proposal"], "approve": bool(yes)}) is not None:
                return "vote", 0.1
        # 3. an absent owner: under owner rule nobody else can change anything, so members found a new home and
        #    move there (community forks after absentee owners); members of an abandoned community follow
        succ = ctx.state.get("commlife_succ", cid)
        if succ is not None and succ["to"] != cid and value <= 0.6 and rng.random() < 0.5 + 0.3 * float(pr[4]):
            if self._call(ctx, p, "community.join", {"community": succ["to"]}) is not None:
                self._call(ctx, p, "community.leave", {"community": cid})
                return "propose", 0.05
        if succ is None and self._owner_absent(ctx, c, now) and c["rules"]["governance"] == "owner" \
                and value > 0 and rng.random() < 0.15 * (0.5 + float(pr[1])):
            cm = ctx.world.modules["community"]
            if cm.trust(ctx.world, p, c)["level"] >= 1:
                new_id = f"{cid[:30]}-new"
                res = self._call(ctx, p, "community.found", {
                    "id": new_id, "name": f"{c['name']} (new home)"[:80],
                    "subject": str(c["subject"])[:400] or "the same subject",
                    "description": f"A new home for the members of {c['name']}: its owner is gone."})
                if res is not None:
                    ctx.emit("commlife.successor", {"from": cid, "to": new_id, "by": p}, vis="public")
                    self._call(ctx, p, "community.leave", {"community": cid})
                    return "propose", 0.15
        # 4. invitations to upcoming events
        for e in evs:
            if e["at"] > now and not ctx.state.has("community_rsvps", f"{e['event']}|{p}"):
                going = rng.random() < _sig(-0.5 + 2.5 * value + 1.5 * (float(pr[6]) - 0.5))
                if self._call(ctx, p, "community.rsvp", {"event": e["event"], "going": bool(going)}) is not None:
                    if going:
                        from .. import population
                        population.wake(ctx, np.array([i]), int(e["at"]) + 60, "community.civic")
                    return "rsvp", 0.05
        # 5. organisers schedule shared activities in lively communities
        if role >= 1 and not evs:
            from ..social import communities as CM
            act7 = int(CM.members(ctx.world, int(c["n"]), active_days=7).size)
            past = [x for x in ctx.state.find_ids("community_events", "community", cid)
                    if ctx.state.tables["community_events"][x]["at"] > now - int(self.cfg("event_every_days")) * DAY]
            if act7 >= int(self.cfg("event_min_active")) and not past:
                in_s = int(rng.integers(1, 4)) * DAY + int(rng.integers(17, 21)) * 3600 - (now % DAY)
                res = self._call(ctx, p, "community.event", {
                    "community": cid, "title": f"{c['name']} meetup", "in_s": max(3600, in_s),
                    "description": f"A get-together of {c['name']}: {c['subject']}", "duration_s": 2 * 3600})
                if res is not None:
                    return "event", 0.15
        # 6. a trusted member of a large community starts a local chapter (a spin-off under it)
        if role >= 1 or value > 0.2:
            cm = ctx.world.modules["community"]
            from ..social import communities as CM
            if int(CM.members(ctx.world, int(c["n"])).size) >= 25 and cm.trust(ctx.world, p, c)["level"] >= 2 \
                    and not c.get("parent") and rng.random() < 0.05 * (0.5 + float(pr[1])):
                kids = [x for x, cc in ctx.state.scan("communities") if cc.get("parent") == cid]
                if len(kids) < 2:
                    slug = f"{cid[:28]}-local{len(kids) + 1}"
                    res = self._call(ctx, p, "community.found", {
                        "id": slug, "name": f"{c['name']} (local chapter {len(kids) + 1})",
                        "subject": f"local chapter of {c['name']}: {c['subject']}"[:400], "parent": cid})
                    if res is not None:
                        return "propose", 0.2
        # 7. propose a change one can name
        prop = self._proposal(ctx, p, c, role, value, pr, now)
        if prop is not None:
            res = self._call(ctx, p, "community.propose", {"community": cid, **prop})
            if res is not None:
                return "propose", 0.1
        # 8. bring a contact along (an invitation, and a message where messaging exists)
        if value > 0 and rng.random() < float(self.cfg("invite_p")) * (0.5 + float(pr[6])):
            cs = [int(x) for x in contacts if x >= 0]
            if cs:
                from ..social import communities as CM
                mem = set(int(x) for x in CM.members(ctx.world, int(c["n"])))
                cand = [x for x in cs if x not in mem]
                if cand:
                    who = ids.pid(int(cand[int(rng.integers(0, len(cand)))]))
                    ids.ensure_principal(ctx, ids.index(who))
                    inv = self._call(ctx, p, "community.invite", {"community": cid, "member": who})
                    if "msg" in ctx.world.modules:
                        m = self._call(ctx, p, "msg.send", {"to": who, "text": f"Come and join {c['name']} - "
                                                                              f"{c['subject']}."})
                        if m is not None:
                            return "message", 0.05
                    if inv is not None:
                        return "invite", 0.05
        return "none", 0.0

    def _owner_absent(self, ctx, c, now) -> bool:
        """The owner has not taken part for a week in a community older than a week (public: the member list shows
        when members were last active)."""
        from ..social import communities as CM
        from ..social import nodes
        if now - int(c.get("created", now)) < 7 * DAY:
            return False
        v = CM.CMEMBER.view(ctx.world)
        rows = np.flatnonzero((v["comm"] == int(c["n"])) & (v["status"] == CM.MEMBER) & (v["role"] == 2))
        if rows.size == 0:
            return True
        return int(v["last_active"][rows].max()) < now - 7 * DAY

    def _judge(self, ctx, p, c, pp, value, pr, rng) -> bool:
        """A member's vote: its own experienced value of the community and preferences, never a scripted outcome."""
        act = pp.get("action")
        args = pp.get("args") or {}
        conform, tol, novelty = float(pr[4]), float(pr[2]), float(pr[1])
        x = 0.3 + 1.5 * value
        if act == "set_rules":
            j = (args.get("rules") or {}).get("join")
            if j in ("approval", "invite"):
                x += 1.5 * (0.5 - tol) - 1.0 * (novelty - 0.5)
            elif j == "open":
                x += 1.0 * (novelty - 0.5) - 1.0 * (0.5 - tol)
        elif act == "add_moderator":
            x += 0.8
        elif act in ("transfer", "remove_moderator"):
            x -= 0.3
        yes, no = int(pp.get("yes", 0)), int(pp.get("no", 0))
        if yes + no:
            x += 2.0 * conform * ((yes - no) / (yes + no))
        return bool(rng.random() < _sig(x))

    def _proposal(self, ctx, p, c, role, value, pr, now) -> dict | None:
        """A change the member can name from what it sees in public (the moderation log, the member list)."""
        from ..social import communities as CM
        if value <= 0:
            return None
        cm = ctx.world.modules["community"]
        if cm.trust(ctx.world, p, c)["level"] < int(c["rules"]["min_trust_propose"]) and role == 0:
            return None
        if ctx.state.find_ids("community_proposals", "community", c["id"]) and any(
                ctx.state.tables["community_proposals"][x]["status"] == "open"
                for x in ctx.state.find_ids("community_proposals", "community", c["id"])):
            return None
        log = self._call(ctx, p, "community.modlog", {"community": c["id"], "limit": 50}) or []
        recent = [x for x in (log if isinstance(log, list) else log.get("log", []))
                  if int(x.get("t", 0)) > now - 14 * DAY]
        removals = sum(1 for x in recent if x.get("action") in ("remove", "ban", "mute", "warn"))
        if c["rules"]["join"] == "open" and removals >= int(self.cfg("propose_removals")):
            return {"action": "set_rules", "args": {"rules": {"join": "approval"}},
                    "reason": f"{removals} moderation actions in two weeks: screen newcomers"}
        act7 = CM.members(ctx.world, int(c["n"]), active_days=7)
        if c["rules"]["join"] in ("approval", "invite") and removals == 0 and act7.size < 3:
            return {"action": "set_rules", "args": {"rules": {"join": "open"}},
                    "reason": "hardly anyone takes part: open the door"}
        mods = [r for r in CM.rows_of(ctx.world, int(c["n"]))
                if CM.row_dict(ctx.world, r)["status"] == CM.MEMBER and CM.row_dict(ctx.world, r)["role"] >= 1]
        if act7.size >= 8 and len(mods) <= 1:
            best = None
            for node in act7.tolist():
                if int(node) == ids.index(p):
                    continue
                r = CM.member_row(ctx.world, int(node), int(c["n"]))
                d = CM.row_dict(ctx.world, r)
                rep = CM.reputation(d)
                if best is None or rep > best[0]:
                    best = (rep, int(node))
            if best is not None and best[0] > 0:
                from ..social import nodes
                return {"action": "add_moderator", "args": {"member": nodes.principal_of(ctx.world, best[1])},
                        "reason": "the community has grown: one more moderator"}
        return None

    @applier("commlife.successor")
    def ap_successor(self, ev):
        self.world.state.put("commlife_succ", ev.payload["from"], {"to": ev.payload["to"], "by": ev.payload["by"],
                                                                   "t": ev.t})


# ====================================================================== configurations (world generation)
STRUCTURES = {
    "grassroots": {"seeded": False, "governance": [], "join": [], "nested": False, "messaging": True},
    "assemblies": {"seeded": True, "governance": ["vote"], "join": ["open"], "nested": False, "messaging": False},
    "curated": {"seeded": True, "governance": ["moderators"], "join": ["approval"], "nested": False,
                "messaging": True},
    "curated_assemblies": {"seeded": True, "governance": ["vote"], "join": ["approval"], "nested": False,
                           "messaging": False},
    "federated": {"seeded": True, "governance": ["vote"], "join": ["invite"], "nested": False, "messaging": True},
}
DOCS = {
    "grassroots": "no community at genesis: the population founds them; owner governance; messaging",
    "assemblies": "communities founded at genesis by members, open doors, members vote on every change; no messaging",
    "curated": "communities founded at genesis with moderator governance and approval to join; messaging",
    "curated_assemblies": "compositional: approval to join (curated) AND member votes (assemblies), no messaging",
    "federated": "structural: invite-only communities (newcomers arrive only through members' invitations), member "
                 "votes, messaging",
}
PLATFORMS = ["community", "posts"]


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    from ..social import worlds as SW
    st = STRUCTURES[config.name]
    ov = dict(overrides or {})
    n_pop = int(ov.pop("population", 1500))
    if ov:
        raise ValueError(f"community domain: unknown overrides {sorted(ov)}")
    modules = ["community", "content", "social", "social_ids", "social_rows", "aud", "feed", "posts", "studio",
               "pop_social", "commlife", "search", "docs"]
    if st["messaging"]:
        modules.append("msg")
    life = {}
    if st["seeded"]:
        rules = {"governance": st["governance"][0], "join": st["join"][0], "quorum": 0.2, "min_trust_propose": 1}
        life["seed"] = {"communities": 4, "rules": rules}
    cfg = {"aud": {"platforms": PLATFORMS, "max_items": 16384, "max_creators": 4096},
           "feed": {"platforms": {p: {"recommenders": SW.MIXTURES[p]} for p in PLATFORMS}},
           "pop_social": {"platforms": {"short": "posts", "long": "posts"}, "community_prior": 0.3},
           "commlife": life,
           "pop": {"profile": "development", "n": n_pop, "tick_s": 900,
                   "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}}
    return civ_recipe(int(seed), modules=modules, population_modules=("pop",), module_config=cfg,
                      params={"log_retain": 3_000_000})


# ====================================================================== objective, evaluator
HORIZON_DAYS = 28.0
TARGET = {1: {"active": 12, "contributors": 3, "retention": 0.3},
          2: {"active": 40, "contributors": 6, "retention": 0.4},
          3: {"active": 100, "contributors": 12, "retention": 0.5}}


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = int(min(3, max(1, difficulty)))
    t = TARGET[d]
    text = (f"Build a community with sustained voluntary participation: within {HORIZON_DAYS:.0f} days found a "
            f"community (community.found) so that, at the end, at least {t['active']} members other than you took "
            f"part in the last 7 days, at least {t['contributors']} other members posted or commented in each of "
            f"the last two weeks, and at least {t['retention']:.0%} of the members who joined two or more weeks "
            f"before the end are still members and took part in the last week.")
    return {"text": text, "description": text, "domain": "community", "difficulty": d, **t,
            "window_days": HORIZON_DAYS}


def _contributors(world, cid: str, exclude: set, t0: int, t1: int) -> int:
    st = world.state
    who = set()
    for xid in st.find_ids("content", "community", cid):
        it = st.tables["content"][xid]
        if t0 <= int(it.get("published_at", it.get("t", -1))) < t1 and it.get("creator") not in exclude:
            who.add(it.get("creator"))
    for cid_, cm in (st.scan("content_comments") if "content_comments" in st.tables else []):
        item = st.tables["content"].get(cm.get("item")) if cm.get("item") else None
        if item is not None and item.get("community") == cid and t0 <= int(cm.get("t", -1)) < t1 \
                and cm.get("author") not in exclude:
            who.add(cm.get("author"))
    return len(who)


def community_health(world, cid: str, exclude: set) -> dict:
    """Evaluator-side participation measures of one community from the recorded membership and content."""
    from ..social import communities as CM
    from ..social import nodes
    c = world.state.tables["communities"][cid]
    v = CM.CMEMBER.view(world)
    now = int(world.now)
    ex_nodes = {nodes.node_of(world, p) for p in exclude}
    m = (v["comm"] == int(c["n"])) & (v["status"] == CM.MEMBER) & ~np.isin(v["node"], list(ex_nodes - {None}) or [-9])
    act = m & (v["last_active"] >= now - WEEK)
    old = (v["comm"] == int(c["n"])) & (v["first_joined"] >= 0) & (v["first_joined"] <= now - 2 * WEEK) & \
        ~np.isin(v["node"], list(ex_nodes - {None}) or [-9])
    kept = old & (v["status"] == CM.MEMBER) & (v["last_active"] >= now - WEEK)
    weeks = [_contributors(world, cid, exclude, now - (k + 1) * WEEK, now - k * WEEK) for k in range(2)]
    return {"members": int(m.sum()), "active_7d": int(act.sum()), "weekly_contributors": weeks,
            "joined_2w_ago": int(old.sum()), "retained": int(kept.sum()),
            "retention": float(kept.sum() / old.sum()) if old.sum() else 0.0}


def evaluate(world, principal: str, obj: dict) -> dict:
    st = world.state
    mine = set(world.principals_of(principal)) | {principal} if world.principal(principal) else {principal}
    rows = {}
    if "communities" in st.tables:
        for cid, c in st.scan("communities"):
            if c.get("founder") in mine:
                rows[cid] = community_health(world, cid, mine)
    ok = any(r["active_7d"] >= int(obj["active"]) and min(r["weekly_contributors"]) >= int(obj["contributors"])
             and r["joined_2w_ago"] > 0 and r["retention"] >= float(obj["retention"]) for r in rows.values())
    best = max(rows.values(), key=lambda r: r["active_7d"], default=None)
    score = 0.0 if best is None else min(1.0, best["active_7d"] / float(obj["active"]))
    return {"success": bool(ok), "score": round(score, 5), "details": {"communities": rows}}


# ====================================================================== the hidden reference (LABELLED SCRIPTED)
_STOP = frozenset("the a an and or of to in on for with at by from is are was were be it its this that as new how "
                  "why what who says said will about after over into more than".split())


def reference_script(post_every_s: int = DAY):
    """A tool-only organiser: subject from what people read, a community about it, daily posts written about that
    subject, weekly events, moderation of reports."""
    def script(pol):
        from ..population import semantics, writing
        items = K.ok((yield {"tool": "content.list", "args": {"platform": "posts", "limit": 50}})) or []
        rows = items if isinstance(items, list) else items.get("items", [])
        words: dict = {}
        for it in rows:
            for wd in str(it.get("title", "")).lower().replace(",", " ").split():
                if len(wd) > 3 and wd not in _STOP and wd.isalpha():
                    words[wd] = words.get(wd, 0) + 1
        names = semantics.topic_names()
        # the handbook-free choice: the corpus topic whose name shares most words with what people read
        best_t, best_s = 0, -1
        for t, nm in enumerate(names):
            sc = sum(words.get(x, 0) for x in str(nm).lower().replace("_", " ").split())
            if sc > best_s:
                best_t, best_s = t, sc
        topic = str(names[best_t])
        slug = ("".join(ch if ch.isalnum() else "-" for ch in topic.lower())[:24].strip("-") or "topic") + "-circle"
        top = sorted(words, key=lambda x: (-words[x], x))[:3]
        res = K.ok((yield {"tool": "community.found", "args": {
            "id": slug, "name": f"{topic.replace('_', ' ').title()} circle",
            "subject": f"{topic.replace('_', ' ')}: {' '.join(top)}".strip()[:400],
            "description": f"Talk about {topic.replace('_', ' ')} with people who follow it.",
            "tags": [x[:30] for x in top if x.isalpha()][:5]}}))
        if not res:
            while True:
                yield None
        cid = res["community"]
        rng = np.random.default_rng(abs(hash(cid)) % (2 ** 31))
        k, last_event = 0, -10 ** 9
        while True:
            title, body = writing.compose(rng, best_t, 6, 0.9)
            ref = (K.ok((yield {"tool": "studio.write", "args": {"format": "text", "title": title[:180],
                                                                 "body": body}})) or {}).get("ref")
            if ref:
                yield {"tool": "community.post", "args": {"community": cid, "ref": ref, "title": title[:180]}}
            k += 1
            t = pol.t
            if t - last_event >= 7 * DAY:
                ev = K.ok((yield {"tool": "community.event", "args": {
                    "community": cid, "title": f"Weekly {topic.replace('_', ' ')} chat", "in_s": DAY,
                    "description": "Bring what you read this week.", "duration_s": 3 * 3600}}))
                if ev:
                    last_event = t
            reps = K.ok((yield {"tool": "community.reports", "args": {"community": cid, "status": "open"}})) or []
            for rp in (reps if isinstance(reps, list) else []):
                yield {"tool": "community.moderate", "args": {"community": cid, "action": "remove",
                                                              "target": rp.get("target"), "report": rp.get("report"),
                                                              "reason": "reported by members"}}
            for _ in range(max(1, int(post_every_s // (6 * 3600))) - 1):
                yield None
    return script


def reference(principal: str, obj: dict):
    return K.ScriptPolicy(reference_script(), label="community-reference")


def _found_and_idle_script():
    def script(pol):
        yield {"tool": "community.found", "args": {"id": "my-place", "name": "My place",
                                                   "subject": "anything and everything"}}
        while True:
            yield None
    return script


def found_and_idle(principal: str, obj: dict):
    """Degenerate: found a community and never take part."""
    return K.ScriptPolicy(_found_and_idle_script(), label="found-and-idle")


def _talk_to_myself_script():
    def script(pol):
        from ..population import writing
        res = K.ok((yield {"tool": "community.found", "args": {"id": "my-stage", "name": "My stage",
                                                              "subject": "my thoughts, many times a day"}}))
        rng = np.random.default_rng(5)
        while res:
            for _ in range(6):
                title, body = writing.compose(rng, int(rng.integers(0, 10)), 3, 0.5)
                ref = (K.ok((yield {"tool": "studio.write", "args": {"format": "text", "title": title[:180],
                                                                     "body": body}})) or {}).get("ref")
                if ref:
                    yield {"tool": "community.post", "args": {"community": "my-stage", "ref": ref,
                                                              "title": title[:180]}}
            yield None
        while True:
            yield None
    return script


def talk_to_myself(principal: str, obj: dict):
    """Degenerate: the founder's own activity, many posts a day (participation that is not voluntary)."""
    return K.ScriptPolicy(_talk_to_myself_script(), label="talk-to-myself")


def null_policy(principal=None, obj=None):
    def script(pol):
        while True:
            yield None
    return K.ScriptPolicy(script, label="null")


# ====================================================================== mechanics (recorded-event probes)
def _ev(world, since, types):
    return [e for e in world.events_range(int(since), world.seq) if e.type in types]


def count_network_joins(world, since: int) -> int:
    """Population members who joined a community after one of their contacts was already a member there (the
    social network carrying membership)."""
    from ..social import communities as CM
    v = CM.CMEMBER.view(world)
    if not v["node"].size or "pop" not in world.modules:
        return 0
    contacts = columns.col(world, "pop", "contacts")
    t_since = world.events_range(int(since), int(since) + 1)[0].t if world.seq > since else world.now
    n = 0
    first = {}
    for r in range(v["node"].size):
        first[(int(v["node"][r]), int(v["comm"][r]))] = int(v["first_joined"][r])
    for (node, comm), t in first.items():
        if t < t_since or node >= contacts.shape[0] or node < 0:
            continue
        cs = contacts[node]
        if any(first.get((int(x), comm), 10 ** 18) < t for x in cs if x >= 0):
            n += 1
    return n


def count_persistent_members(world, since: int) -> int:
    from ..social import communities as CM
    v = CM.CMEMBER.view(world)
    now = int(world.now)
    return int(np.sum((v["status"] == CM.MEMBER) & (v["first_joined"] >= 0) & (v["first_joined"] <= now - WEEK)))


def count_independent_participants(world, since: int) -> int:
    """Distinct population individuals who posted, commented, voted, attended or reported in a community they did
    not found."""
    st = world.state
    founders = {c.get("founder") for _, c in st.scan("communities")} if "communities" in st.tables else set()
    who = set()
    for e in _ev(world, since, ("community.vote", "community.attend", "community.report", "content.published",
                                "content.comment")):
        p = e.payload.get("voter") or e.payload.get("member") or e.payload.get("by") or e.payload.get("creator") \
            or e.payload.get("author") or e.actor
        if e.type == "content.published" and e.payload.get("platform") != "community":
            continue
        if K.is_population(p) and p not in founders:
            who.add(p)
    return len(who)


def count_communication(world, since: int) -> int:
    return sum(1 for e in _ev(world, since, ("content.published", "content.comment", "msg.message"))
               if e.type != "content.published" or e.payload.get("platform") == "community")


def count_retained(world, since: int) -> int:
    """Members who joined at least a week ago and took part in the last week (retention measured, not assigned)."""
    from ..social import communities as CM
    v = CM.CMEMBER.view(world)
    now = int(world.now)
    return int(np.sum((v["status"] == CM.MEMBER) & (v["first_joined"] >= 0) & (v["first_joined"] <= now - WEEK) &
                      (v["last_active"] >= now - WEEK)))


MECHANICS = (
    Mechanic("Social networks", ("cols.batch:social.follow", "community.notice"),
             "follows, invitations to contacts, joins carried by contacts", count=count_network_joins),
    Mechanic("Persistent memberships", ("cols.batch:community.join", "cols.batch:community.found"),
             "membership rows with spells, roles and history", count=count_persistent_members),
    Mechanic("Independent participants", (), "population individuals taking part in communities they did not found",
             count=count_independent_participants),
    Mechanic("Shared activities", ("community.event", "community.rsvp", "community.attend"),
             "scheduled events, RSVPs and attendance while they run"),
    Mechanic("Communication", (), "posts and comments in communities, direct messages", count=count_communication),
    Mechanic("Moderation", ("community.report", "community.moderation"), "reports and moderators' actions"),
    Mechanic("Governance", ("community.proposal", "community.vote", "community.decided"),
             "proposals, votes and decisions under each community's governance rule"),
    Mechanic("Retention", ("cols.batch:community.leave",), "members who stay and take part, members who leave",
             count=count_retained),
    Mechanic("Community evolution", ("community.rules", "community.info", "community.founded", "community.status"),
             "rule changes, renames, new communities and spin-offs, archiving"),
)


def structure(world) -> dict:
    st = world.state
    comms = [c for _, c in st.scan("communities")] if "communities" in st.tables else []
    return {"seeded": bool(comms), "governance": sorted({c["rules"]["governance"] for c in comms}),
            "join": sorted({c["rules"]["join"] for c in comms}), "messaging": "msg" in world.modules}


CONFIGS = tuple(DomainConfig(n, s, dict(STRUCTURES[n]), DOCS[n]) for n, s in (
    ("grassroots", "train"), ("assemblies", "train"), ("curated", "train"),
    ("curated_assemblies", "eval_compositional"), ("federated", "eval_structural")))

DOMAIN = register(Domain(
    name="community", title="Community ecosystem", spec_part="9.4",
    required_mechanics=tuple(m.name for m in MECHANICS), configs=CONFIGS, build=build, objective=objective,
    evaluate=evaluate, reference=reference, mechanics=MECHANICS,
    seats=(Seat("organiser", 200_000, "Organiser", "someone who wants to start a community"),),
    actors=(Actor("population members", ("cols.batch:commlife.decide", "cols.batch:community.experience"),
                  "individuals who visit, post, vote, attend and invite"),
            Actor("population founders and creators", ("community.founded", "cols.batch:pop_social.created"),
                  "individuals who found communities and write")),
    null_baseline=null_policy,
    degenerate=(Probe("found_and_idle", "found a community and never take part", found_and_idle),
                Probe("talk_to_myself", "the founder posts many times a day alone", talk_to_myself)),
    structure=structure, horizon_days=HORIZON_DAYS, slowest_effect_days=14.0, dt=6 * 3600,
    reference_seeds=(0, 1), sources=("community.py",),
    doc="Communities on the one population: memberships, posts, events, moderation, governance and evolution."))
