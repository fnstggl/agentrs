"""The population on the private and subscription channels of the social internet (module `pop_social`).

* activity `social.inbox` (messaging, 4.1/4.7): a member opens its messages. It reads the unread messages of its
  conversations (`msg.read` as its own principal: exact read positions); a message that links an item puts that item
  at the top of its feed on the item's platform (the feed's share inbox - it opens it there, through the ordinary
  consumption engine, if it wants to); it may answer (sociable people more often). It may also recommend the last
  thing it read and liked to the contact whose interests fit it best (`msg.send` with the item; the recipient is
  notified and comes back to its messages). Members review their paid creator memberships here and cancel the ones
  that stopped paying off (they no longer follow the creator, the creator stopped publishing, or money is short).
* after every feed session (hook `feed.after_session` of every platform; 4.4/4.5): what viewers do next with what
  they saw - subscribe to the newsletter of a creator they follow or whose piece they read through and liked
  (`letters.subscribe` internals: one subscription edge), join the paid membership of a creator they follow and
  already knew (`creators.join`: the first period is charged from their OWN bank account at once, renewals monthly),
  on the Q&A platforms (kb, dev) vote on questions and answers they read (`<platform>.vote`; never their own) and, as
  a question's author, accept an answer that helped; after reading a newsletter issue that was not worth it,
  unsubscribe. Launch sessions keep their product-discovery follow-up (`popsocial._after_launch`).

Every step goes through the platforms' own tools or internal APIs as the member's own principal; nothing is
credited, ranked or paid by this module. The decisions are recorded with provenance (`pop_social.inbox`, model
`pop-inbox@1`; `pop_social.followup`, model `pop-followup@1`). Fidelity: plausible, unvalidated
(FIDELITY_REGISTER `population-messaging-model`, `population-subscription-model`).
"""
from __future__ import annotations

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Ctx
from ..substrate import attention, columns, ids
from .. import population
from ..population import semantics, writing
from . import audience as AUD
from . import nodes

DAY = 86400
INBOX_MODEL = {"model_id": "pop-inbox", "model_version": "1", "fidelity_id": "population-messaging-model"}
FOLLOWUP_MODEL = {"model_id": "pop-followup", "model_version": "1", "fidelity_id": "population-subscription-model"}
NOTE_DELAY_S = 1800         # a message notification brings the recipient back to its messages after ~30 min
MEMBERSHIP_WTP_CENTS = 800  # monthly support a member with wtp_scale 1 would pay a creator it values


def columns_spec() -> dict:
    return {"soc_msgs_sent": ("int32", (), 0), "soc_msgs_read": ("int32", (), 0), "soc_subs": ("int16", (), 0)}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _item_topic(world, item: int) -> int:
    return int(semantics.nearest_topic(AUD.item_col(world, "emb")[[int(item)]].astype(np.float32))[0])


def act_inbox(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
    w = ctx.world
    st = ctx.state
    idx = np.asarray(idx, np.int64)
    n = idx.size
    U = np.full(n, -0.01)
    if "msg" not in w.modules or n == 0:
        return U
    rng = ctx.rng("pop_social.inbox")
    prefs = columns.col(w, "pop", "prefs")[idx].astype(np.float64)
    skills = columns.col(w, "pop", "skills")[idx].astype(np.float64)
    coords = population.interest_coords(w, idx)
    sent = columns.col(w, "pop", "soc_msgs_sent")[idx].astype(np.int64).copy()
    readn = columns.col(w, "pop", "soc_msgs_read")[idx].astype(np.int64).copy()
    minutes = np.zeros(n)
    feed = w.modules.get("feed")
    last_c = columns.col(w, "pop", "aud_c_item")[idx].astype(np.int64)
    last_t = columns.col(w, "pop", "aud_c_t")[idx].astype(np.int64)
    last_ws = columns.col(w, "pop", "aud_c_watch_s")[idx].astype(np.float64)
    contacts = columns.col(w, "pop", "contacts")[idx].astype(np.int64)
    emb = AUD.item_col(w, "emb")
    dur_all = AUD.item_col(w, "duration_s")
    for j in range(n):
        i = int(idx[j])
        p = ids.pid(i)
        soc = float(prefs[j, 6])
        if st.has("principals", p):
            # 1. read: the unread messages of one's conversations (most recent first, at most three)
            unread = []
            for k in st.find_ids("msg_members", "member", p):
                m = st.tables["msg_members"][k]
                c = st.tables["msg_conversations"].get(m["conversation"])
                if m["active"] and c is not None and c["n"] > m["read"]:
                    unread.append((-int(c["last_t"]), m["conversation"]))
            for _, conv in sorted(unread)[:3]:
                try:
                    got = ctx.call("msg.read", {"conversation": conv, "limit": 10}, as_principal=p)
                except CrucibleError:
                    continue
                new = [x for x in got["messages"] if x["sender"] != p][-5:]
                readn[j] += len(new)
                minutes[j] += 0.5 * len(new)
                U[j] += 0.03 * len(new) * (0.5 + soc)
                linked = []
                for x in new:
                    cid = x.get("item")
                    rec = st.get("content", cid) if cid else None
                    if rec is not None and rec.get("aud_item") is not None and not rec["removed"]:
                        linked.append(int(rec["aud_item"]))
                if linked and feed is not None:
                    for it in sorted(set(linked)):
                        plat = st.tables["aud_item_info"][str(it)]["platform"]
                        if plat in feed.platforms:
                            # the link opens the item's platform with it on top of the feed (the share inbox)
                            feed.engine.deliver(ctx, np.array([i]), np.array([it]), np.array([1], np.uint8),
                                                platform=plat)
                if new and rng.random() < float(_sig(-1.5 + 2.5 * soc)):
                    topic = _item_topic(w, linked[0]) if linked else int(rng.integers(0, len(semantics.topic_names())))
                    _, text = writing.compose(rng, topic, 1, float(skills[j, 1]))
                    try:
                        ctx.call("msg.send", {"conversation": conv, "text": text[:400]}, as_principal=p)
                        sent[j] += 1
                        minutes[j] += 1.0
                        others = [m_ for m_ in got["members"] if m_ != p and ids.index(m_) is not None]
                        if others:
                            population.wake(ctx, np.array([ids.index(o) for o in others], np.int64),
                                            now + NOTE_DELAY_S, "social.inbox")
                    except CrucibleError:
                        pass
            # memberships: keep paying only while it is worth it
            if "creator_members" in st.tables:
                _review_memberships(ctx, i, p, rng)
        # 2. recommend: the last thing one read through and liked, to the contact it suits best
        it = int(last_c[j])
        if it >= 0 and now - int(last_t[j]) < 3 * DAY and last_ws[j] >= float(AUD.view_threshold(dur_all[it])):
            info = st.tables["aud_item_info"].get(str(it))
            cid = None if info is None else info.get("content_id")
            cj = contacts[j][(contacts[j] >= 0) & (contacts[j] != i)]
            aff = float(coords[j] @ emb[it].astype(np.float32))
            if cid and cj.size and info["creator"] != p and \
                    rng.random() < float(_sig(-3.0 + 2.5 * soc + 3.0 * aff)):
                cc = population.interest_coords(w, cj)
                fit = cc @ emb[it].astype(np.float32) + rng.gumbel(size=cj.size) * 0.1
                to = int(cj[int(np.argmax(fit))])
                me = ids.ensure_principal(ctx, i)
                ids.ensure_principal(ctx, to)
                note = ["you might like this", "thought of you", "worth a look", "have you seen this?"][
                    int(rng.integers(0, 4))]
                try:
                    ctx.call("msg.send", {"to": ids.pid(to), "text": note, "item": cid}, as_principal=me)
                    sent[j] += 1
                    minutes[j] += 1.0
                    U[j] += 0.02 * (0.5 + soc)
                    population.wake(ctx, np.array([to], np.int64), now + NOTE_DELAY_S, "social.inbox")
                except CrucibleError:
                    pass
    left = columns.col(w, "pop", "attention_left_min")[idx].astype(np.float64)
    minutes = np.minimum(minutes, left)
    sp = np.flatnonzero(minutes > 0)
    if sp.size:
        attention.spend(ctx, idx[sp], minutes[sp], activity="social.inbox")
    population.record_decisions(ctx, "pop_social.inbox", idx, {"soc_msgs_sent": sent.astype(np.int32),
                                                              "soc_msgs_read": readn.astype(np.int32)},
                                **INBOX_MODEL, extra={"t": int(now), "sent": int(sent.sum() -
                                                      columns.col(w, "pop", "soc_msgs_sent")[idx].sum())})
    return U


def _review_memberships(ctx: Ctx, i: int, p: str, rng) -> None:
    st = ctx.state
    from . import graph
    for mid in st.find_ids("creator_members", "fan", p) if ("creator_members", "fan") in st.indexes else []:
        m = st.tables["creator_members"][mid]
        if m.get("status") != "active" or m.get("cancel_at_end"):
            continue
        creator = m["creator"]
        t = st.tables["creator_tiers"].get(m["tier"], {})
        a = nodes.node_of(ctx.world, p)
        b = nodes.node_of(ctx.world, creator)
        follows = bool(a is not None and b is not None and graph.is_following(ctx.world, np.array([a]),
                                                                              np.array([b]))[0])
        recent = [c for c in st.find_ids("content", "creator", creator)
                  if st.tables["content"][c]["published_at"] >= ctx.now - 30 * DAY]
        money = int(population.spendable(ctx.world, np.array([i]))[0])
        reason = None
        if not follows:
            reason = "no longer following"
        elif not recent:
            reason = "the creator stopped publishing"
        elif money < 2 * int(t.get("price_cents", 0)):
            reason = "cannot afford it"
        if reason and rng.random() < 0.5:
            try:
                ctx.call("creators.cancel", {"membership": mid, "at_period_end": True}, as_principal=p)
            except CrucibleError:
                pass


# ================================================================================================ after sessions
def after_session(self, ctx: Ctx, out: dict, platform: str) -> np.ndarray:
    """What viewers do next with what they saw on `platform` (see the module docstring)."""
    w = ctx.world
    st = ctx.state
    idx = out["idx"]
    U = np.zeros(idx.size)
    if not np.any(out.get("view", False)):
        return U
    rng = ctx.rng("pop_social.followup", platform)
    vv, pp = np.nonzero(out["view"])
    feed, watch = out["feed"], out["watch"]
    coords = population.interest_coords(w, idx)
    emb = AUD.item_col(w, "emb")
    dur = AUD.item_col(w, "duration_s")
    legib = AUD.item_col(w, "legib")
    wtp = columns.col(w, "pop", "wtp_scale")[idx].astype(np.float64)
    subs_n = columns.col(w, "pop", "soc_subs")[idx].astype(np.int64).copy()
    letters = w.modules.get("letters")
    cm = w.modules.get("creators")
    from . import graph
    deciders = set()
    n_sub = n_join = n_vote = n_accept = n_unsub = 0
    for a, b in zip(vv.tolist(), pp.tolist()):
        i = int(idx[a])
        item = int(feed[a, b])
        info = st.tables["aud_item_info"].get(str(item))
        if info is None:
            continue
        creator = info["creator"]
        p = ids.pid(i)
        if creator == p:
            continue
        aff = float(coords[a] @ emb[item].astype(np.float32))
        frac = float(watch[a, b]) / max(float(dur[item]), 1.0)
        na, nb = nodes.node_of(w, p), nodes.node_of(w, creator)
        follows = bool(na is not None and nb is not None and graph.is_following(w, np.array([na]),
                                                                               np.array([nb]))[0])
        # newsletters: subscribe to the creator's newsletter
        if letters is not None and platform != "letters":
            nls = [x for x in st.find_ids("newsletters", "owner", creator)]
            if nls and rng.random() < float(_sig(-4.5 + 2.5 * follows + 5.0 * aff + 1.5 * frac)):
                nl = st.tables["newsletters"][nls[0]]
                node = nodes.ensure(ctx, ids.ensure_principal(ctx, i))
                if letters.subscribe_many(ctx, np.array([node], np.int64), np.array([int(nl["n"])], np.int64),
                                          via=2):
                    n_sub += 1
                    subs_n[a] += 1
                    deciders.add(a)
        # memberships: support a creator one follows and keeps coming back to
        if cm is not None and follows and "creator_tiers" in st.tables:
            tiers = [t for t in st.find_ids("creator_tiers", "creator", creator)
                     if st.tables["creator_tiers"][t]["status"] == "active"] \
                if ("creator_tiers", "creator") in st.indexes else []
            if tiers and cm.live(st, creator, p) is None:
                t = st.tables["creator_tiers"][sorted(tiers)[0]]
                willing = MEMBERSHIP_WTP_CENTS * wtp[a] * (0.5 + 2.0 * max(aff, 0.0))
                if rng.random() < float(_sig(-5.0 + 4.0 * aff + 2.0 * frac + 2.0 * (willing - t["price_cents"]) /
                                              max(willing, 1.0))) and \
                        int(population.spendable(w, np.array([i]))[0]) >= 3 * int(t["price_cents"]):
                    from ..population.core import money_deposit
                    try:
                        acct = money_deposit(ctx, i)
                        cm.join(ctx, ids.pid(i), t["id"] if "id" in t else sorted(tiers)[0], account=acct)
                        n_join += 1
                        deciders.add(a)
                    except CrucibleError:
                        pass
        # Q&A: vote on what one read; accept an answer to one's own question that helped
        if platform in ("kb", "dev") and f"{platform}_posts" in st.tables:
            cid = info.get("content_id")
            post = st.get(f"{platform}_posts", cid) if cid else None
            if post is not None:
                vote = 0
                if float(legib[item]) < 0.3:
                    vote = -1 if rng.random() < 0.3 else 0
                elif frac >= 0.6 and aff > 0.05 and rng.random() < 0.35:
                    vote = 1
                if vote:
                    try:
                        ctx.call(f"{platform}.vote", {"id": cid, "value": vote}, as_principal=ids.ensure_principal(
                            ctx, i))
                        n_vote += 1
                        deciders.add(a)
                    except CrucibleError:
                        pass
                q = post.get("question")
                if post["type"] == "answer" and q and st.get("content", q) is not None and \
                        st.tables["content"][q]["creator"] == p and \
                        not st.tables[f"{platform}_posts"][q].get("accepted") and aff > 0.0 and rng.random() < 0.5:
                    try:
                        ctx.call(f"{platform}.accept", {"question": q, "answer": cid}, as_principal=p)
                        n_accept += 1
                        deciders.add(a)
                    except CrucibleError:
                        pass
        # newsletter issues that were not worth it: unsubscribe
        if platform == "letters" and letters is not None and frac < 0.3 and aff < 0.05:
            desc = st.tables["content"].get(info.get("content_id") or "", {}).get("description")
            nl = next((x for x in st.find_ids("newsletters", "owner", creator)
                       if st.tables["newsletters"][x]["name"] == desc), None)
            if nl is not None and rng.random() < 0.4:
                try:
                    ctx.call("letters.unsubscribe", {"newsletter": nl}, as_principal=p)
                    n_unsub += 1
                    subs_n[a] = max(0, subs_n[a] - 1)
                    deciders.add(a)
                except CrucibleError:
                    pass
    if deciders:
        rows = np.array(sorted(deciders), np.int64)
        population.record_decisions(ctx, "pop_social.followup", idx[rows],
                                    {"soc_subs": subs_n[rows].astype(np.int16)}, **FOLLOWUP_MODEL,
                                    extra={"platform": platform, "subscribed": n_sub, "joined": n_join,
                                           "votes": n_vote, "accepted": n_accept, "unsubscribed": n_unsub})
    return U


def creator_channels(self, ctx: Ctx, i: int, p: str, rng, title: str) -> str | None:
    """A creator with an audience of followers opens a newsletter and a paid membership tier (ordinary tools as its
    own principal); returns the newsletter id it may send this piece to (None: publish it as a blog article)."""
    w = ctx.world
    st = ctx.state
    from . import graph
    node = nodes.node_of(w, p)
    followers = int(graph.follower_counts(w, np.array([node]))[0]) if node is not None else 0
    nl = None
    if "letters" in w.modules:
        mine = st.find_ids("newsletters", "owner", p)
        if mine:
            nl = mine[0]
        elif followers >= int(self.cfg["newsletter_followers"]) and rng.random() < 0.3:
            name = " ".join(title.split()[:3]) or "Notes"
            try:
                nl = ctx.call("letters.create", {"name": f"{name} - a letter", "description":
                                                 f"Pieces by {p} about what they follow"}, as_principal=p)["newsletter"]
            except CrucibleError:
                nl = None
    if "creators" in w.modules and "creator_tiers" in st.tables and \
            followers >= int(self.cfg["tier_followers"]) and rng.random() < 0.15:
        have = [t for t in st.find_ids("creator_tiers", "creator", p)] \
            if ("creator_tiers", "creator") in st.indexes else []
        if not have:
            price = int(300 + 100 * np.log2(1 + followers)) // 50 * 50
            try:
                ctx.call("creators.create_tier", {"name": "Supporters", "price_cents": price,
                                                  "perks": "support my writing"}, as_principal=p)
            except CrucibleError:
                pass
    if nl is not None and rng.random() < 0.6:
        return nl
    return None
