"""The population's own contributions to the social internet (kernel module `pop_social`, host `pop_social`).

Members of the one population (`crucible.population`) create content through the same path as everyone else
(`content.publish`: catalogue, search index, distribution and exact audience accounting), as an activity that
competes for their attention and time with reading, browsing, searching and the communities:

* `social.community` - take part in communities (4.7): members visit one of their communities (the one whose
  experienced value is highest, with noise), read its recent posts through the consumption engine (exposure,
  reading time, views: ordinary audience accounting), and may post on its subject, reply to what they read, report
  unreadable posts to its moderators, or leave when taking part stopped paying off (the membership's experienced
  value). The author of a post that gets a reply is notified and comes back to its communities about half an hour
  later. Owners and moderators act on reports. Non-members and explorers look through the directory: a community
  fits by the similarity of its subject to one's interests (shared space), its recent activity and how many of
  one's contacts take part; one may join it. Someone who finds nothing that fits may found a community whose
  subject is what they have been reading (the most frequent words of the titles of their recent views) - subjects
  are never predefined. An owner who leaves hands the community to its most active member.
* product discovery (4.8): after a session on the product-discovery platform (`launch`), a member who viewed a
  launch post that interests it may visit the product it links to (`launch.visit`), weigh the price against what it
  is willing to pay and the reviews of paying customers, buy it from its own bank account (`prod.buy`), use it
  (`prod.access`, then reading the delivered document: its value is the document's fit to the buyer's interests and
  its legibility, from the bytes), and rate it (`prod.rate`) - which later visitors read. Everything goes through
  the commerce tools as the member's own principal; no bonus anywhere.
* `social.create` - make something for an audience on every platform (`social.popcreate`): a text post, a status
  update, an article (sent as an issue of one's newsletter once one has followers), an image card, a narrated audio
  piece, a narrated short or long slide video (`social.popmedia`, recorded rendering), an answer or a question on the
  knowledge and developer platforms, or a recommendation of a service that works for one's own work (naming its
  address). The medium is chosen by taste and ability plus what one's earlier pieces of each medium earned (the
  platforms' delayed, noisy analytics), so creators move to where they find an audience. No piece carries a quality,
  relevance or virality figure.
* `social.inbox` - messages (`social.popinbox`): read one's conversations, open linked items, answer, recommend what
  one liked to the contact it suits best; review paid creator memberships. After every feed session viewers may
  subscribe to newsletters, join creator memberships, vote and accept answers on the Q&A platforms, unsubscribe
  from newsletters that stopped paying off, and look into a service for their own work they read about (a `shop`
  session of `economy.work`).

Decision provenance: every decision is recorded through `population.record_decisions` (`pop_social.created`, model
`pop-create@2`; `pop_social.inbox`, `pop-inbox@1`; `pop_social.followup`, `pop-followup@1`; communities
`pop-community@1`; product discovery `pop-shop@1`). Config keys `platforms`, `minutes` and `topic_noise` are kept for
compatibility (the media table of `social.popcreate` sets platforms and effort per medium).

Fidelity: plausible, unvalidated (FIDELITY_REGISTER: population creator model).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import CrucibleError, ProtocolError
from ..core.kernel import Ctx, Module, register_module
from ..substrate import artifacts, columns, ids
from .. import population
from ..population import semantics, writing
from . import audience as AUD
from . import content as C

from .popcreate import CREATE_MODEL  # noqa: E402,F401  (pop-create@2: the creation of every medium)
COMMUNITY_MODEL = {"model_id": "pop-community", "model_version": "1", "fidelity_id": "population-community-model"}
SHOP_MODEL = {"model_id": "pop-shop", "model_version": "1", "fidelity_id": "population-product-discovery-model"}
DEFAULTS = {"prior": -1.2, "platforms": {"short": "posts", "long": "blog"}, "minutes": [10.0, 45.0],
            "topic_noise": 0.08, "community_prior": -0.2, "explore": 0.15, "found_threshold": 0.32,
            "found_prob": 0.10, "leave_value": -0.04, "creator_share": 1.0, "inbox_prior": -0.6,
            "newsletter_followers": 5, "tier_followers": 15}
_STOP = frozenset("""the a an and or of to in on for with at by from is are was were be been it its this that as
after over about into new how why what who says said will would could more most than his her their they you your
our we he she not but have has had up out off one two first last year years day week according monday tuesday
wednesday thursday friday saturday sunday just also only when where which while there here them then these those
some many much very still back being after before during people time made make makes says told""".split())


@register_module
class PopSocial(Module):
    name = "pop_social"
    host = "pop_social"
    version = "1"
    depends = ("pop", "space", "content", "aud")

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"pop_social: unknown config key {k!r}")
            self.cfg[k] = v

    def install(self, world):
        n = columns.rows(world, "pop")
        import functools
        from . import popcreate, popinbox
        from .platforms import PLATFORMS
        names = {m["name"] for m in world.spec.modules}
        cols = {"soc_items": ("int32", (), 0), "soc_seen_views": ("int64", (), 0), "soc_last_item": ("int32", (), -1)}
        if int(popcreate.available(world).sum()) > 1:      # a choice of media to learn about (not the 1-platform
            cols.update(popcreate.columns_spec())           # internet profile: 10M rows stay lean)
        if "msg" in names or "letters" in names:
            cols.update(popinbox.columns_spec())
        columns.declare(world, "pop", n, cols)
        population.register_activity(world, "social.create", self.act_create, owner="pop_social",
                                     prior=float(self.cfg["prior"]), goals=(3, 5))
        if "msg" in names:
            population.register_activity(world, "social.inbox", self.act_inbox, owner="pop_social",
                                         prior=float(self.cfg["inbox_prior"]), goals=(3,))
        # what viewers do next with what they saw, on every platform (subscriptions, memberships, Q&A votes;
        # product discovery on launch)
        hooks = world.hooks.setdefault("feed.after_session", {})
        for plat in PLATFORMS:
            hooks[plat] = functools.partial(self.followup, platform=plat)
        if "community" in world.modules or any(m["name"] == "community" for m in world.spec.modules):
            population.register_activity(world, "social.community", self.act_community, owner="pop_social",
                                         prior=float(self.cfg["community_prior"]), goals=(3, 1))

    # the activities: act_create (crucible.social.popcreate), act_inbox and followup (crucible.social.popinbox),
    # act_community and after_launch (below)


# ================================================================================================ communities
def _community_vectors(ctx: Ctx, recs: list[dict], k: int) -> np.ndarray:
    texts = [f"{r['name']}. {r['subject']}. {r.get('description', '')}".strip() for r in recs]
    if not texts:
        return np.zeros((0, k), np.float32)
    return semantics.project(semantics.text_vectors(ctx.world, texts, ctx=ctx), k)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _act_community(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
    from . import communities as CM
    from .recsys import Mixture
    w = ctx.world
    cm = w.modules["community"]
    idx = np.asarray(idx, np.int64)
    n = idx.size
    rng = ctx.rng("pop_social.community")
    U = np.full(n, -0.02)
    st = ctx.state
    active_ids = sorted(st.find_ids("communities", "status", "active"))
    recs = [dict(st.tables["communities"][c]) for c in active_ids]
    k = population.interest_coords(w, idx[:1]).shape[1] if n else 0
    cvec = _community_vectors(ctx, recs, k)
    v = CM.CMEMBER.view(w)
    mem_mask = (v["status"] == CM.MEMBER) & np.isin(v["node"], idx)
    mrows = np.flatnonzero(mem_mask)
    by_ind: dict[int, list[int]] = {}
    for r in mrows:
        by_ind.setdefault(int(v["node"][r]), []).append(int(r))
    coords = population.interest_coords(w, idx)
    prefs = columns.col(w, "pop", "prefs")[idx].astype(np.float64)
    skills = columns.col(w, "pop", "skills")[idx].astype(np.float64)
    explore = rng.random(n) < float(self.cfg["explore"])
    visit = {}          # community n -> list of cohort positions
    joiners, join_comm = [], []
    founders = []
    contacts = columns.col(w, "pop", "contacts")[idx].astype(np.int64)
    n_by_comm = {int(r["n"]): r for r in recs}
    pos_of = {int(r["n"]): j for j, r in enumerate(recs)}
    act7 = np.array([CM.members(w, int(rc["n"]), active_days=7).size for rc in recs], np.float64)
    mem_of = [np.sort(CM.members(w, int(rc["n"]))) for rc in recs]
    for j in range(n):
        i = int(idx[j])
        mine = [r for r in by_ind.get(i, []) if int(v["comm"][r]) in pos_of]
        if mine and not explore[j]:
            vals = np.array([float(v["value"][r]) for r in mine]) + rng.gumbel(size=len(mine)) * 0.15
            r = mine[int(np.argmax(vals))]
            visit.setdefault(int(v["comm"][r]), []).append(j)
            continue
        # the directory: fit of subject, recent activity, contacts taking part
        if recs:
            aff = cvec @ coords[j]
            cj = contacts[j][contacts[j] >= 0]
            cmem = np.array([np.isin(cj, m).mean() if cj.size else 0.0 for m in mem_of])
            already = np.array([int(rc["n"]) in {int(v["comm"][r]) for r in mine} for rc in recs])
            score = 8.0 * aff + 0.15 * np.log1p(act7) + 1.5 * cmem + rng.gumbel(size=len(recs)) * 0.5
            score = np.where(already, -np.inf, score)
            best = int(np.argmax(score))
            if np.isfinite(score[best]):
                p_join = _sig(-4.0 + 8.0 * aff[best] + 0.15 * np.log1p(act7[best]) + 1.5 * cmem[best] +
                              1.0 * prefs[j, 6] - 0.8 * len(mine))
                if rng.random() < p_join:
                    joiners.append(j)
                    join_comm.append(int(recs[best]["n"]))
                    continue
            fits = float(aff.max())
        else:
            fits = -1.0
        if fits < float(self.cfg["found_threshold"]) and \
                rng.random() < float(self.cfg["found_prob"]) * (0.5 + prefs[j, 6]) * (0.5 + skills[j, 1]):
            founders.append(j)
    # joins (one recorded decision batch through the community module)
    if joiners:
        jn = np.array([int(idx[j]) for j in joiners], np.int64)
        for i in jn:
            ids.ensure_principal(ctx, int(i))
        cm.join_many(ctx, jn, np.array(join_comm, np.int64), "population")
        U[np.array(joiners)] = 0.02
    # founding: the subject is what one has been reading
    for j in founders:
        self._found(ctx, int(idx[j]), rng)
    # visits: read the community's recent posts, then post, reply, report, or leave
    notified: set[int] = set()
    for cn, js in sorted(visit.items()):
        js = np.array(js, np.int64)
        c = n_by_comm[cn]
        items = [int(st.tables["content"][cid]["aud_item"]) for cid in st.find_ids("content", "community", c["id"])
                 if st.tables["content"][cid].get("aud_item") is not None and not st.tables["content"][cid]["removed"]
                 and st.tables["content"][cid]["published_at"] >= now - 14 * 86400]
        util = np.zeros(js.size)
        out = None
        if items:
            out = self.engine().run(ctx, "community", idx[js], pool=np.array(items, np.int64),
                                    mixture=Mixture([{"name": "fresh", "weight": 1.0}]), activity="social.community")
            uniq = out["idx"]
            util = np.asarray(out["utility"], np.float64)[np.searchsorted(uniq, idx[js])]
        # write: post on the subject
        topic = semantics.nearest_topic(cvec[pos_of[cn]][None, :])[0] if cn in pos_of else 0
        p_post = _sig(-3.2 + 2.5 * skills[js, 1] + 1.5 * prefs[js, 6] + 2.0 * np.maximum(util, 0))
        posters = js[rng.random(js.size) < p_post]
        for j in posters:
            i = int(idx[j])
            p = ids.ensure_principal(ctx, i)
            title, body = writing.compose(rng, int(topic), int(rng.integers(1, 4)), float(skills[j, 1]))
            try:
                ref = artifacts.put(ctx, f"{title}\n\n{body}\n".encode("utf-8"), "text/plain", owner=p,
                                    meta={"title": title[:80]}, kind="text")
                C.module(w).publish(ctx, p, ref, "post", "community", title, community=c["id"],
                                    access="public" if c["rules"]["visibility"] == "public" else "members")
            except CrucibleError:
                pass
        # reply to and report what one read
        if out is not None and np.any(out["view"]):
            vv, pp = np.nonzero(out["view"])
            legib = AUD.item_col(w, "legib")
            for a, b in zip(vv, pp):
                i = int(out["idx"][a])
                item = int(out["feed"][a, b])
                info = AUD.info(w, item)
                cid = info.get("content_id")
                if not cid:
                    continue
                if float(legib[item]) < 0.3 and rng.random() < 0.5:
                    try:
                        ctx.call("community.report", {"community": c["id"], "target": cid,
                                                      "reason": "unreadable"}, as_principal=ids.pid(i))
                    except CrucibleError:
                        pass
                elif rng.random() < 0.08 + 0.2 * float(columns.col(w, "pop", "prefs")[i, 6]):
                    _, reply = writing.compose(rng, int(topic), 1, float(columns.col(w, "pop", "skills")[i, 1]))
                    try:
                        C.module(w).add_comment(ctx, ids.pid(i), cid, reply)
                    except CrucibleError:
                        continue
                    # the author is notified of the reply and comes back to the community (member-to-member
                    # interaction: a conversation in a community brings its members back, like a message does)
                    a = ids.index(info.get("creator") or "")
                    if a is not None and a != i:
                        notified.add(int(a))
        # moderators act on open reports
        for j in js:
            p = ids.pid(int(idx[j]))
            if cm._role(w, p, c) not in ("moderator", "owner"):
                continue
            for rid in st.find_ids("community_reports", "community", c["id"]):
                rep = st.tables["community_reports"][rid]
                if rep.get("status") != "open" or not C.CID_RE.fullmatch(rep["target"]):
                    continue
                ai = st.tables["content"][rep["target"]].get("aud_item")
                action = "remove" if ai is not None and float(AUD.item_col(w, "legib")[int(ai)]) < 0.3 else "dismiss"
                try:
                    ctx.call("community.moderate", {"community": c["id"], "action": action, "target": rep["target"],
                                                    "reason": "unreadable" if action == "remove" else "fine",
                                                    "report": rid}, as_principal=p)
                except CrucibleError:
                    pass
        # the experienced value of taking part, and leaving when it stopped paying off
        cm.experience_many(ctx, idx[js], np.full(js.size, cn, np.int64), util, decision=COMMUNITY_MODEL)
        U[js] = util
        v2 = CM.CMEMBER.view(w)
        rows = cm._lookup(w, idx[js], np.full(js.size, cn, np.int64))
        low = (rows >= 0) & (v2["value"][np.maximum(rows, 0)] < float(self.cfg["leave_value"])) & \
            (v2["visits"][np.maximum(rows, 0)] >= 3)
        leavers = idx[js][low & (rng.random(js.size) < 0.5)]
        for i in leavers:
            p = ids.pid(int(i))
            if cm._role(w, p, c) == "owner":
                others = [int(x) for x in CM.members(w, cn, active_days=30) if int(x) != int(i)]
                if not others:
                    continue
                act = CM.CMEMBER.view(w)
                best = max(others, key=lambda x: int(act["posts"][cm._lookup(w, np.array([x]), np.array([cn]))[0]])
                           + int(act["comments"][cm._lookup(w, np.array([x]), np.array([cn]))[0]]))
                cm._execute(ctx, dict(st.tables["communities"][c["id"]]), "transfer", {"member": ids.pid(best)}, None)
            cm.leave_many(ctx, np.array([int(i)]), np.array([cn]), "no longer worth it")
    if notified:
        from .popinbox import NOTE_DELAY_S
        population.wake(ctx, np.array(sorted(notified), np.int64), ctx.now + NOTE_DELAY_S, "social.community")
    return U


def _found(self, ctx: Ctx, i: int, rng) -> str | None:
    """Found a community about what individual i has been reading: the titles of the items it viewed lately."""
    import collections
    import re
    from . import idset
    w = ctx.world
    st = ctx.state
    ni = AUD.n_items(w)
    pub = AUD.item_col(w, "published_at")
    recent = np.flatnonzero(pub >= ctx.now - 7 * 86400)[-400:]
    seen = [int(x) for x in recent if idset.contains(st, idset.key("vw", int(x)), np.array([i]))[0]]
    if len(seen) < 3:
        return None
    # the subject: the words that distinguish what one read (frequent in one's views, rare in the catalogue)
    tok = lambda t: {x for x in re.findall(r"[a-z][a-z'-]{3,}", t.lower()) if x not in _STOP and "'" not in x}  # noqa
    mine = collections.Counter()
    for it in seen[-30:]:
        mine.update(tok(AUD.info(w, it)["title"]))
    df = collections.Counter()
    sample = recent[-400:]
    for it in sample:
        df.update(tok(AUD.info(w, int(it))["title"]))
    score = {x: c * math.log((1 + len(sample)) / (1 + df[x])) for x, c in mine.items()}
    top = [x for x, _ in sorted(score.items(), key=lambda kv: (-(mine[kv[0]] >= 2), -kv[1], kv[0]))[:3]]
    if len(top) < 2:
        return None
    E = AUD.item_col(w, "emb")[seen].astype(np.float32).mean(axis=0, keepdims=True)
    topic = semantics.topic_names()[int(semantics.nearest_topic(E)[0])]
    base = "-".join(re.sub(r"[^a-z0-9]", "", x)[:12] for x in top[:2]).strip("-")
    slug = f"{base}-{int(rng.integers(100, 999))}"[:40]
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,39}", slug) or st.has("communities", slug):
        return None
    p = ids.ensure_principal(ctx, i)
    name = " ".join(x.capitalize() for x in top)
    subject = f"{', '.join(top)} ({topic.replace('_', ' ')})"
    try:
        return w.modules["community"].found(ctx, p, slug, name, subject, f"People who read about {' and '.join(top)}.",
                                            tags=[re.sub(r"[^a-z0-9-]", "", x)[:30] for x in top])
    except CrucibleError:
        return None


from . import popcreate as _popcreate  # noqa: E402
from . import popinbox as _popinbox  # noqa: E402


def _followup(self, ctx: Ctx, out: dict, platform: str) -> np.ndarray:
    u = _popinbox.after_session(self, ctx, out, platform)
    if platform == "launch":
        u = u + np.asarray(self.after_launch(ctx, out), np.float64)
    return u


PopSocial.act_create = _popcreate.act_create
PopSocial.act_inbox = _popinbox.act_inbox
PopSocial.followup = _followup
PopSocial.creator_channels = _popinbox.creator_channels
PopSocial.act_community = _act_community
PopSocial._found = _found
PopSocial.engine = lambda self: self.world.modules["feed"].engine


def _sigm(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def _after_launch(self, ctx: Ctx, out: dict) -> np.ndarray:
    """Members who viewed launch posts: visit, decide, buy, use and rate products (ordinary commerce tools)."""
    from ..population.core import money_deposit
    w = ctx.world
    st = ctx.state
    idx = out["idx"]
    U = np.zeros(idx.size)
    if not np.any(out["view"]) or "products" not in w.modules:
        return U
    rng = ctx.rng("pop_social.shop")
    vv, pp = np.nonzero(out["view"])
    coords = population.interest_coords(w, idx)
    wtp = columns.col(w, "pop", "wtp_scale")[idx].astype(np.float64)
    deciders, bought_n, spent = [], [], []
    for a, b in zip(vv.tolist(), pp.tolist()):
        i = int(idx[a])
        item = int(out["feed"][a, b])
        cid = AUD.info(w, item).get("content_id")
        L = st.get("launches", cid) if cid else None
        if L is None or L["link"] != "product":
            continue
        aff = float(coords[a] @ AUD.item_col(w, "emb")[item].astype(np.float32))
        if rng.random() >= _sigm(-1.5 + 6.0 * aff):
            continue
        p = ids.ensure_principal(ctx, i)
        try:
            page = ctx.call("launch.visit", {"id": cid}, as_principal=p)["page"]
        except CrucibleError:
            continue
        deciders.append(a)
        offers = sorted(page.get("offers") or [], key=lambda o: (o["price_cents"], o["offer"]))
        if not offers or page.get("status") != "launched":
            continue
        price = int(offers[0]["price_cents"])
        willing = 2500.0 * wtp[a] * (1.0 + 1.5 * max(aff, 0.0))
        stars = page.get("rating")
        rv = 0.0 if stars is None else (float(stars) - 3.0) / 2.0
        if rng.random() >= _sigm(-1.0 + 3.0 * (willing - price) / max(willing, 1.0) + 1.2 * rv):
            U[a] -= 0.02
            continue
        if int(population.spendable(w, np.array([i]))[0]) < price:
            continue
        try:
            acct = money_deposit(ctx, i)
            ctx.call("prod.buy", {"offer": offers[0]["offer"], "account": acct}, as_principal=p)
            got = ctx.call("prod.access", {"product": page["product"]}, as_principal=p)
        except CrucibleError:
            continue
        bought_n.append(a)
        spent.append(price)
        # using it: read the delivered document; its value for this buyer comes from its bytes
        value = -0.1
        if got.get("doc"):
            try:
                doc = ctx.call("docs.get", {"doc": got["doc"]}, as_principal=p)
                text = str(doc.get("body") or doc.get("text") or doc.get("content") or "")
            except CrucibleError:
                text = ""
            if text.strip():
                v = semantics.project(semantics.text_vectors(w, [text], ctx=ctx), coords.shape[1])[0]
                legib = semantics.legibility(semantics.text_features(text))
                value = float(coords[a] @ v) * (0.3 + 0.7 * legib) - 0.1
        U[a] += value - 0.3 * price / max(willing, 1.0)
        stars_given = int(np.clip(round(3 + 6.0 * value), 1, 5))
        try:
            ctx.call("prod.rate", {"product": page["product"], "stars": stars_given,
                                   "text": "worth it" if stars_given >= 4 else ("ok" if stars_given == 3 else
                                                                                "not worth it")}, as_principal=p)
        except CrucibleError:
            pass
    if deciders:
        rows = np.unique(np.array(deciders, np.int64))
        nb = columns.col(w, "pop", "n_purchases")[idx[rows]].astype(np.int64)
        sp = columns.col(w, "pop", "spent_cents")[idx[rows]].astype(np.int64)
        for a, c in zip(bought_n, spent):
            k = int(np.searchsorted(rows, a))
            nb[k] += 1
            sp[k] += c
        population.record_decisions(ctx, "pop_social.shop", idx[rows], {"n_purchases": nb.astype(np.int32),
                                                                        "spent_cents": sp},
                                    **SHOP_MODEL, extra={"visits": len(deciders), "purchases": len(bought_n),
                                                         "revenue_cents": int(sum(spent))})
    return U


PopSocial.after_launch = _after_launch
