"""The consumption engine: platform sessions of the population (workstream C2; MASTER_SPEC Parts 4.1, 4.4, 4.5).

`run_session(ctx, platform, idx)` runs one platform session for every individual in `idx` (a cohort the population
activated and that chose this platform). Per session:

1. **Candidates** come from the platform: up to `pool` live items (newest and most viewed), plus items contacts
   shared to the viewer (its inbox).
2. **Distribution**: up to `inbox_slots` unseen shared items open the feed (source "share"); the platform's current
   recommender mixture fills the other slots (`recsys.Mixture`, evaluator-side parameters). Exposure is finite:
   a session has `k` slots and a viewer only reaches the slots it scrolls to.
3. **Decisions** (the behaviour model `BUILTIN`, parameters in `behaviour`), slot by slot, vectorised over the
   cohort: open the item or scroll on (a choice among the item, the next `window` items and not opening: competing
   content), how long to read or watch (bounce, else an exponential abandonment hazard that falls with
   individual-content affinity, creator familiarity and satisfaction and rises for content already seen, truncated
   by the item's duration and the attention left), whether to like, share (scaled by the individual's share
   propensity) or follow the creator, whether to continue the session (satisfaction, minutes spent) - every one a
   seeded random draw from the individual's own state. Affinity is the cosine between the individual's need-shifted
   interests and the representation computed from the artifact's bytes; nothing reads a creator-assigned score.
4. **Recording**: impressions (reached slots) and consumption rows (opened items) through `audience.impressions` /
   `audience.consumption` (one `cols.batch` each), the attention actually used through `attention.spend` (never
   more than is left), the individuals' new state (satisfaction, next activation, platform habit, consumed inbox
   entries, interest drift) in one batch, and shares delivered to the sharers' contacts' inboxes (a wake-up for
   the recipients) and likes pushed to the contacts' activity rings, each one batch.

Fidelity: `BUILTIN` is "plausible, unvalidated" - its functional forms follow the consumption literature (choice
among competing items, a hazard model of watch time, social contagion through shares), its parameters are not
fitted. Workstream E's calibrated consumption model replaces the watch/choice draws when it is merged
(docs/v5/components/C2.md, integration notes).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.errors import InvalidAction
from ..substrate import attention, columns
from . import audience as AUD
from . import idset
from .recsys import features as F

BUILTIN = {
    "id": "c2-consumption", "version": "1",
    # opening an item (logit utility), compared with the next `window` items and not opening (utility `outside`)
    "open_bias": -0.6, "open_affinity": 6.0, "open_familiar": 0.6, "open_follow": 0.9, "open_share": 1.2,
    "open_seen": -1.8, "open_sat": 1.2, "open_len": -0.25, "outside": 0.0, "window": 3,
    # bounce (leave within seconds) and the abandonment hazard (per second) of those who stay
    "bounce_bias": -0.8, "bounce_affinity": -5.0,
    "hazard_per_s": 1.0 / 60.0, "hazard_affinity": -4.0, "hazard_familiar": -0.3, "hazard_sat": -0.8,
    "hazard_seen": 0.8,
    # decisions after a qualifying view
    "like_bias": -2.8, "like_affinity": 4.0, "like_complete": 0.8,
    "share_bias": -6.6, "share_affinity": 5.0, "share_complete": 1.0, "share_sat": 1.0,
    "follow_bias": -4.6, "follow_affinity": 4.0, "follow_familiar": 1.2, "follow_complete": 0.8,
    # continuing the session
    "continue_bias": 2.2, "continue_sat": 2.0, "continue_view": 0.3, "continue_minutes": -0.12,
    # state updates
    "sat_lr": 0.12, "sat_ref_affinity": 0.2, "sat_skip": -0.004, "habit_lr": 0.5, "habit_decay": 0.02,
    "taste_drift": 0.02, "scroll_s": 2.0, "notice_prob": 0.3, "gap_mean_s": 6 * 3600.0, "gap_sat": 1.0, "need_weight": 0.3,
}
DEFAULT_PLATFORM = {"k": 10, "pool": 300, "pool_days": 7, "inbox_slots": 2}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def candidate_pool(world, platform: str, size: int, days: float) -> np.ndarray:
    """Live items of the platform published in the last `days`: half newest, half most viewed (sorted ids)."""
    items = AUD.items_on(world, platform)
    if items.size == 0:
        return items
    pub = columns.col(world, AUD.ITEMS, "published_at")[items]
    items = items[pub >= world.now - days * 86400]
    if items.size <= size:
        return items
    pub = columns.col(world, AUD.ITEMS, "published_at")[items]
    views = columns.col(world, AUD.ITEMS, "views")[items]
    newest = items[np.argsort(-pub, kind="stable")[:size // 2]]
    rest = np.setdiff1d(items, newest)
    rv = columns.col(world, AUD.ITEMS, "views")[rest]
    top = rest[np.argsort(-rv, kind="stable")[:size - newest.size]]
    return np.unique(np.concatenate([newest, top]))


def rowwise_member(world, kind: str, keys: np.ndarray, viewers: np.ndarray) -> np.ndarray:
    """[V, k] bool: viewers[v] in set (kind, keys[v, j]) (keys < 0 -> False)."""
    out = np.zeros(keys.shape, dtype=bool)
    vv = np.broadcast_to(viewers[:, None], keys.shape)
    flat = keys.reshape(-1)
    vf = vv.reshape(-1)
    of = out.reshape(-1)
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]]) if fs.size else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], fs.size] if fs.size else np.zeros(0, np.int64)
    st = world.state
    for s0, e0 in zip(starts, ends):
        k = int(fs[s0])
        if k < 0:
            continue
        rows = order[s0:e0]
        of[rows] = idset.contains(st, idset.key(kind, k), vf[rows])
    return of.reshape(keys.shape)


def ring_push(old: np.ndarray, new_items: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Push new entries (newest first) into ring rows: new_items [U, R] holds each row's m[u] new entries in its
    first columns; older entries shift right and the oldest fall off."""
    U, R = old.shape
    J = np.arange(R)[None, :]
    mm = np.minimum(m, R)[:, None]
    src = np.clip(J - mm, 0, R - 1)
    return np.where(J < mm, new_items, np.take_along_axis(old, src, axis=1))


def group_into_rows(recipients: np.ndarray, values: list[np.ndarray], R: int):
    """Group deliveries by recipient: (unique recipients, counts, [U, R] arrays holding each recipient's first R
    values in delivery order)."""
    order = np.argsort(recipients, kind="stable")
    r = recipients[order]
    u, start, cnt = np.unique(r, return_index=True, return_counts=True)
    rank = np.arange(r.size) - np.repeat(start, cnt)
    keep = rank < R
    outs = []
    for vals in values:
        a = np.full((u.size, R), -1 if vals.dtype.kind == "i" else 0, dtype=vals.dtype)
        a[np.searchsorted(u, r[keep]), rank[keep]] = vals[order][keep]
        outs.append(a)
    return u, cnt, outs


def item_topics(world, items: np.ndarray) -> np.ndarray:
    """Each item's topic: the topic vocabulary nearest to its representation (computed from the artifact)."""
    from . import textrep
    E = F.item_emb(world, items).astype(np.float64)
    return np.argmax(E @ textrep.topic_centroids(E.shape[1]).T, axis=1)


class CalibratedOpen:
    """The decision to open an item shown in a slate, from workstream E's `news_click@1` (gradient-boosted trees
    fitted on MIND-small, Microsoft News, US, November 2019; held-out AUC 0.78 vs 0.73 for the popularity baseline;
    card: docs/v5/calibration/models/news_click/model_card.json). Its card allows exactly this use (opening an
    article shown in a slate of article recommendations) and nothing about dwell time, which stays built-in.

    Feature mapping (the card's `feature_map`; approximations are named):
      text_cos       cosine between the item and the individual's interests (which drift toward what it consumed),
                     quantile-mapped onto the card's text_cos quantiles through its rank among this session batch's
                     slots (different encoder: the card says to map by quantile)
      text_cos_z     that cosine standardised within the individual's slate (scale-free, as the card says)
      cat_share      (individual's past views in the item's topic + s) / (past views + 1), s = the platform's share
                     of views in that topic; topic = nearest topic vocabulary of the item's representation
      subcat_share   cat_share quantile-mapped onto the card's subcat_share quantiles (approximation: this world has
                     no finer taxonomy, and a subcategory share is much smaller than a category share)
      log_hist_len   log1p(individual's past views)
      pop_ctr_24h    (views + 10 m) / (impressions + 10) of the item on the platform, m = card prior_rate, quantile-
                     mapped onto the card's quantiles (MIND logs whole slates as impressions, this world only the
                     slots an individual reached, so the raw rates are not on the same scale; all-time counters,
                     pool items being at most a few days old)
      pop_log_clicks_per_1k  log1p(1000 * views in the last 24 h / individuals with an impression in the last 24 h);
                     views in the last 24 h approximated by views x min(1, 24 h / age)
      age_log_hours  log1p(hours since the item was published)
      slate_size_log log1p(slots in the slate)
    """

    def __init__(self, name: str):
        from .. import calibration
        self.model = calibration.load(name)
        self.name = name
        feats = list(self.model.features)
        if feats != ["text_cos", "text_cos_z", "cat_share", "subcat_share", "log_hist_len", "pop_ctr_24h",
                     "pop_log_clicks_per_1k", "age_log_hours", "slate_size_log"]:
            raise ValueError(f"{name}: unexpected features {feats}")
        q = self.model.card["feature_quantiles"]
        self.levels = np.asarray(q["levels"], dtype=np.float64)
        self.quantiles = {f: np.asarray(q[f], dtype=np.float64) for f in feats}
        self.prior = float(self.model.card["prior_rate"])

    def qmap(self, values: np.ndarray, feature: str) -> np.ndarray:
        """Map values onto the card's distribution of `feature` by their rank in this batch (ties share a rank)."""
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return v
        u, inv = np.unique(v, return_inverse=True)
        cnt = np.bincount(inv)
        mid = (np.cumsum(cnt) - 0.5 * cnt) / v.size                  # mid-rank of each distinct value
        return np.interp(mid[inv], self.levels, self.quantiles[feature])

    def p_open(self, world, platform: str, idx: np.ndarray, feed: np.ndarray, aff: np.ndarray,
               has_item: np.ndarray) -> np.ndarray:
        V, k = feed.shape
        out = np.zeros((V, k))
        if not np.any(has_item):
            return out
        rows, cols_ = np.nonzero(has_item)
        items = feed[rows, cols_]
        a = aff[rows, cols_]
        text_cos = self.qmap(a, "text_cos")
        am = np.where(has_item, aff, np.nan)
        mu = np.nanmean(am, axis=1, keepdims=True)
        sd = np.nanstd(am, axis=1, keepdims=True)
        z = ((am - mu) / np.where(sd > 1e-9, sd, 1.0))[rows, cols_]
        topic_of = item_topics(world, AUD.items_on(world, platform, live_only=False))
        on = AUD.items_on(world, platform, live_only=False)
        views_on = columns.col(world, AUD.ITEMS, "views")[on].astype(np.float64)
        T = int(columns.col(world, "pop", "feed_topic_views").shape[1])
        share = (np.bincount(topic_of, weights=views_on, minlength=T) + 1.0) / (views_on.sum() + T)
        t_item = item_topics(world, items)
        tv = columns.col(world, "pop", "feed_topic_views")[idx[rows]].astype(np.float64)
        hist = tv.sum(axis=1)
        cat = (tv[np.arange(rows.size), t_item] + share[t_item]) / (hist + 1.0)
        views = columns.col(world, AUD.ITEMS, "views")[items].astype(np.float64)
        imp = columns.col(world, AUD.ITEMS, "imp")[items].astype(np.float64)
        age = np.maximum(F.age_s(world, items), 0.0)
        last_imp = columns.col(world, "pop", "aud_i_t")
        active = max(1, int(np.count_nonzero(last_imp >= world.now - 86400)))
        v24 = views * np.minimum(1.0, 86400.0 / np.maximum(age, 1.0))
        ctr = (views + 10 * self.prior) / (imp + 10)
        X = np.stack([text_cos, z, cat, self.qmap(cat, "subcat_share"), np.log1p(hist), self.qmap(ctr, "pop_ctr_24h"),
                      np.log1p(1000.0 * v24 / active), np.log1p(age / 3600.0), np.full(rows.size, math.log1p(k))],
                     axis=1)
        out[rows, cols_] = self.model.predict(X)
        return out


CALIBRATED_OPEN = {"news_click@1": CalibratedOpen}


class Engine:
    """Bound to the `feed` module: platform configs, mixtures, ring sizes and the behaviour parameters."""

    def __init__(self, feed_module):
        self.feed = feed_module
        self.b = dict(BUILTIN)
        self.b.update(feed_module.cfg.get("behaviour") or {})
        self.calibrated = {p: CALIBRATED_OPEN[c["open"]](c["open"])
                           for p, c in sorted((feed_module.cfg.get("calibrated") or {}).items())}

    # ------------------------------------------------------------------ distribution
    def build_feed(self, ctx, platform: str, idx: np.ndarray, rng):
        w = ctx.world
        pc = {**DEFAULT_PLATFORM, **self.feed.platform_cfg(platform)}
        k = int(pc["k"])
        V = idx.size
        pool = candidate_pool(w, platform, int(pc["pool"]), float(pc["pool_days"]))
        feed = np.full((V, k), -1, np.int64)
        src = np.zeros((V, k), np.int8)
        gen = np.zeros((V, k), np.uint8)
        used_inbox = np.zeros((V, self.feed.R), dtype=bool)
        n_in = np.zeros(V, np.int64)
        islots = int(pc["inbox_slots"])
        if islots > 0:
            ring = columns.col(w, "pop", "feed_inbox")[idx].astype(np.int64)
            rgen = columns.col(w, "pop", "feed_inbox_gen")[idx]
            ok = ring >= 0
            ni = AUD.n_items(w)
            ringc = np.where(ok & (ring < ni), ring, 0)
            okp = ok & (ring < ni) & \
                (columns.col(w, AUD.ITEMS, "platform")[ringc] == self.feed.aud.platform_code(platform)) & \
                ~columns.col(w, AUD.ITEMS, "removed")[ringc]
            seen = rowwise_member(w, "vw", np.where(okp, ring, -1), idx)
            okp &= ~seen
            # duplicates inside one ring: keep the first (newest) occurrence
            for r in range(1, ring.shape[1]):
                okp[:, r] &= ~np.any((ring[:, :r] == ring[:, r:r + 1]) & okp[:, :r], axis=1)
            take = okp & (np.cumsum(okp, axis=1) <= min(islots, k))
            used_inbox = take
            n_in = take.sum(axis=1)
            rr, cc = np.nonzero(take)
            slot = (np.cumsum(take, axis=1) - 1)[rr, cc]
            feed[rr, slot] = ring[rr, cc]
            gen[rr, slot] = rgen[rr, cc]
            src[rr, slot] = AUD.SOURCE_CODE["share"]
        mix = self.feed.mixture(platform, ctx.now)
        if pool.size and k > 0:
            recs, rsrc = mix.rank(w, platform, idx, pool, k + islots, rng)
            # drop recommended items already placed from the inbox, then fill the remaining slots in order
            dup = np.zeros(recs.shape, dtype=bool)
            for j in range(islots):
                dup |= (recs == feed[:, j:j + 1]) & (feed[:, j:j + 1] >= 0)
            recs = np.where(dup, -1, recs)
            valid = recs >= 0
            order = np.argsort(~valid, axis=1, kind="stable")
            recs = np.take_along_axis(recs, order, axis=1)
            rsrc = np.take_along_axis(rsrc, order, axis=1)
            J = np.arange(k)[None, :]
            srcpos = J - n_in[:, None]
            fill = (srcpos >= 0)
            sp = np.clip(srcpos, 0, recs.shape[1] - 1)
            feed = np.where(fill, np.take_along_axis(recs, sp, axis=1), feed)
            src = np.where(fill, np.take_along_axis(rsrc, sp, axis=1), src)
            gen = np.where(fill, 0, gen).astype(np.uint8)
        return feed, src, gen, used_inbox

    # ------------------------------------------------------------------ the session
    def run(self, ctx, platform: str, idx: np.ndarray) -> dict:
        w = ctx.world
        b = self.b
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        V = idx.size
        if V == 0:
            return {"sessions": 0}
        rng = ctx.rng("feed.session", platform)
        feed, src, gen, used_inbox = self.build_feed(ctx, platform, idx, rng)
        k = feed.shape[1]
        has_item = feed >= 0
        fe = np.where(has_item, feed, 0)
        u = F.interests(w, idx, need_weight=float(b["need_weight"]))
        E = F.item_emb(w, fe.reshape(-1)).reshape(V, k, -1)
        aff = np.einsum("vd,vkd->vk", u, E).astype(np.float64)
        dur = columns.col(w, AUD.ITEMS, "duration_s")[fe].astype(np.float64)
        cre = columns.col(w, AUD.ITEMS, "creator")[fe].astype(np.int64)
        cpop = columns.col(w, AUD.CREATORS, "pop_index")[cre]
        self_item = (cpop == idx[:, None]) & has_item
        fam = rowwise_member(w, "cv", np.where(has_item, cre, -1), idx)
        fol = rowwise_member(w, "fo", np.where(has_item, cre, -1), idx)
        seen = rowwise_member(w, "vw", np.where(has_item, feed, -1), idx)
        from_share = src == AUD.SOURCE_CODE["share"]
        sat = columns.col(w, "pop", "satisfaction")[idx].astype(np.float64)
        left_min = columns.col(w, "pop", "attention_left_min")[idx].astype(np.float64)
        L = left_min * 60.0
        try:
            sprop = columns.col(w, "pop", "share_prop")[idx].astype(np.float64)
        except Exception:
            sprop = np.ones(V)
        util = (b["open_bias"] + b["open_affinity"] * aff + b["open_familiar"] * fam + b["open_follow"] * fol +
                b["open_share"] * from_share + b["open_seen"] * seen + b["open_sat"] * (sat[:, None] - 0.5) +
                b["open_len"] * np.log(np.maximum(dur, 1.0) / 60.0))
        eu = np.where(has_item & ~self_item, np.exp(np.clip(util, -40, 40)), 0.0)
        W_ = int(b["window"])
        csum = np.concatenate([np.zeros((V, 1)), np.cumsum(eu, axis=1)], axis=1)
        J = np.arange(k)
        win = csum[:, np.minimum(J + W_, k)] - csum[:, J]
        p_open = eu / (math.exp(b["outside"]) + win)
        if platform in self.calibrated:
            p_open = np.where(self_item, 0.0, self.calibrated[platform].p_open(w, platform, idx, feed, aff, has_item))
        p_bounce = _sigmoid(b["bounce_bias"] + b["bounce_affinity"] * aff)
        lam = b["hazard_per_s"] * np.exp(b["hazard_affinity"] * aff + b["hazard_familiar"] * fam +
                                         b["hazard_sat"] * (sat[:, None] - 0.5) + b["hazard_seen"] * seen)
        thr = AUD.view_threshold(dur)
        R = rng.random((V, k, 8))
        active = np.ones(V, dtype=bool)
        elapsed = np.zeros(V)
        reached = np.zeros((V, k), dtype=bool)
        opened = np.zeros((V, k), dtype=bool)
        watch = np.zeros((V, k))
        comp = np.zeros((V, k), dtype=bool)
        view = np.zeros((V, k), dtype=bool)
        liked = np.zeros((V, k), dtype=bool)
        shared = np.zeros((V, k), dtype=bool)
        followed = np.zeros((V, k), dtype=bool)
        value = np.zeros(V)
        n_open = np.zeros(V)
        scroll = float(b["scroll_s"])
        for p in range(k):
            hp = active & has_item[:, p] & (elapsed + scroll <= L)
            reached[:, p] = hp
            elapsed = np.where(hp, elapsed + scroll, elapsed)
            o = hp & (R[:, p, 0] < p_open[:, p])
            bnc = R[:, p, 1] < p_bounce[:, p]
            T = np.where(bnc, 1.0 + 5.0 * R[:, p, 2], -np.log(np.maximum(R[:, p, 2], 1e-300)) / lam[:, p])
            T = np.minimum(np.minimum(T, dur[:, p]), np.maximum(L - elapsed, 0.0))
            T = np.where(o, T, 0.0)
            T = T.astype(np.float32).astype(np.float64)              # what is recorded (float32)
            watch[:, p] = T
            elapsed = elapsed + T
            opened[:, p] = o & (T > 0)
            c = opened[:, p] & (T >= AUD.COMPLETE_FRAC * dur[:, p])
            comp[:, p] = c
            vw = opened[:, p] & (T >= thr[:, p])
            view[:, p] = vw
            a = aff[:, p]
            liked[:, p] = vw & (R[:, p, 3] < _sigmoid(b["like_bias"] + b["like_affinity"] * a + b["like_complete"] * c))
            shared[:, p] = vw & (R[:, p, 4] < np.minimum(1.0, sprop * _sigmoid(
                b["share_bias"] + b["share_affinity"] * a + b["share_complete"] * c + b["share_sat"] * (sat - 0.5))))
            followed[:, p] = vw & ~fol[:, p] & (R[:, p, 5] < _sigmoid(
                b["follow_bias"] + b["follow_affinity"] * a + b["follow_familiar"] * fam[:, p] +
                b["follow_complete"] * c))
            frac = np.where(opened[:, p], T / np.maximum(dur[:, p], 1e-9), 0.0)
            gain = np.where(opened[:, p], frac * (a - b["sat_ref_affinity"]), np.where(hp, b["sat_skip"], 0.0))
            sat = np.clip(sat + b["sat_lr"] * gain, 0.0, 1.0)
            value += np.where(opened[:, p], frac * (a - b["sat_ref_affinity"]), 0.0)
            n_open += opened[:, p]
            cont = R[:, p, 6] < _sigmoid(b["continue_bias"] + b["continue_sat"] * (sat - 0.5) +
                                         b["continue_view"] * vw + b["continue_minutes"] * elapsed / 60.0)
            active = active & np.where(hp, cont, True) & (elapsed < L - 1e-9)
        return self._record(ctx, platform, idx, feed, src, gen, used_inbox, reached, opened, watch, comp, liked,
                            shared, followed, view, elapsed, L, sat, value, n_open, rng)

    def _record(self, ctx, platform, idx, feed, src, gen, used_inbox, reached, opened, watch, comp, liked, shared,
                followed, view, elapsed, L, sat, value, n_open, rng) -> dict:
        w = ctx.world
        b = self.b
        V, k = feed.shape
        vr, pr = np.nonzero(reached)
        AUD.impressions(ctx, platform, idx[vr], feed[vr, pr], pr, source=src[vr, pr], gen=gen[vr, pr])
        vo, po = np.nonzero(opened)
        AUD.consumption(ctx, platform, idx[vo], feed[vo, po], watch[vo, po], comp[vo, po], shared[vo, po],
                        followed[vo, po], liked[vo, po])
        spent = elapsed > 0
        if np.any(spent):
            mins = elapsed[spent] / 60.0
            left = columns.col(w, "pop", "attention_left_min")[idx[spent]].astype(np.float64)
            mins = np.minimum(mins, left)                         # elapsed <= left by construction
            attention.spend(ctx, idx[spent], mins, activity=f"feed.{platform}")
        # the individuals' new state
        st_vals = {"satisfaction": sat.astype(np.float32)}
        left_after = (L - elapsed) / 60.0
        gap = float(b["gap_mean_s"]) * np.exp(-float(b["gap_sat"]) * (sat - 0.5))
        nxt = ctx.now + np.maximum(60.0, -np.log(np.maximum(rng.random(V), 1e-300)) * gap)
        day_end = (ctx.now // 86400 + 1) * 86400
        tired = left_after < 1.0
        nxt = np.where(tired, day_end + rng.random(V) * 8 * 3600, nxt)
        st_vals["next_active_at"] = nxt.astype(np.int64)
        P = len(self.feed.platforms)
        pcode = self.feed.platform_index(platform)
        hab = columns.col(w, "pop", "feed_habit")[idx].astype(np.float64)
        sess_val = np.where(n_open > 0, value / np.maximum(n_open, 1), b["sat_skip"])
        hab = hab * (1 - float(b["habit_decay"]))
        hab[:, pcode] += float(b["habit_lr"]) * sess_val
        st_vals["feed_habit"] = hab.astype(np.float32)
        ring = columns.col(w, "pop", "feed_inbox")[idx].copy()
        ring[used_inbox] = -1
        st_vals["feed_inbox"] = ring
        if self.calibrated and np.any(view):
            tv = columns.col(w, "pop", "feed_topic_views")[idx].astype(np.int64)
            vv, pv = np.nonzero(view)
            np.add.at(tv, (vv, item_topics(w, feed[vv, pv])), 1)
            st_vals["feed_topic_views"] = np.minimum(tv, 65535).astype(np.uint16)
        drift = float(b["taste_drift"])
        if drift > 0 and np.any(view):
            u = columns.col(w, "pop", "interests")[idx].astype(np.float64)
            fe = np.where(view, feed, 0)
            E = F.item_emb(w, fe.reshape(-1)).reshape(V, k, -1).astype(np.float64)
            pull = np.einsum("vk,vkd->vd", view * (watch / np.maximum(columns.col(w, AUD.ITEMS, "duration_s")[fe], 1.0)), E)
            u2 = u + drift * pull
            st_vals["interests"] = (u2 / np.maximum(np.linalg.norm(u2, axis=1, keepdims=True), 1e-9)
                                    ).astype(np.float16)
        columns.batch(ctx, "pop", f"feed.session.{platform}", idx, st_vals, mode="set",
                      extra={"platform": platform, "sessions": int(V)})
        # shares: to every contact of the sharer, as an inbox entry one generation further; contacts are woken
        graph = w.runtime.get("crowd.graph")
        n_deliv = 0
        vs, ps = np.nonzero(shared)
        if vs.size and graph is not None:
            sharers = idx[vs]
            items = feed[vs, ps]
            g = np.minimum(gen[vs, ps].astype(np.int64) + 1, 255)
            con = graph.contacts(sharers)
            rec = con.reshape(-1)
            it = np.repeat(items, con.shape[1])
            gg = np.repeat(g, con.shape[1])
            okr = rec != np.repeat(sharers, con.shape[1])
            rec, it, gg = rec[okr], it[okr], gg[okr]
            n_deliv = int(rec.size)
            self.deliver(ctx, rec, it, gg)
        # likes: into the contacts' activity rings (what friends liked)
        vl, pl = np.nonzero(liked)
        if vl.size and graph is not None:
            likers = idx[vl]
            con = graph.contacts(likers)
            rec = con.reshape(-1)
            it = np.repeat(feed[vl, pl], con.shape[1])
            okr = rec != np.repeat(likers, con.shape[1])
            u_, cnt, (newi,) = group_into_rows(rec[okr], [it[okr].astype(np.int32)], self.feed.R)
            old = columns.col(w, "pop", "feed_social")[u_]
            columns.batch(ctx, "pop", "feed.activity", u_, {"feed_social": ring_push(old, newi, cnt)}, mode="set")
        return {"sessions": int(V), "impressions": int(vr.size), "opened": int(vo.size), "views": int(view.sum()),
                "shares": int(shared.sum()), "deliveries": n_deliv}

    def deliver(self, ctx, recipients: np.ndarray, items: np.ndarray, gens: np.ndarray) -> None:
        """Put shared items into the recipients' inboxes (newest first); each recipient notices the notification
        with probability `notice_prob` and then comes back after `share_delay_s` (unless it was coming back sooner)."""
        w = ctx.world
        u_, cnt, (newi, newg) = group_into_rows(recipients, [items.astype(np.int32), gens.astype(np.uint8)],
                                               self.feed.R)
        old_i = columns.col(w, "pop", "feed_inbox")[u_]
        old_g = columns.col(w, "pop", "feed_inbox_gen")[u_]
        columns.batch(ctx, "pop", "share.deliver", u_, {"feed_inbox": ring_push(old_i, newi, cnt),
                                                         "feed_inbox_gen": ring_push(old_g, newg, cnt)}, mode="set",
                      extra={"deliveries": int(recipients.size)})
        noticed = u_[ctx.rng("feed.notice").random(u_.size) < float(self.b["notice_prob"])]
        if noticed.size:
            if "crowd" in w.modules:
                from .scale import wake
            else:
                from ..population import wake
            wake(ctx, noticed, ctx.now + int(self.feed.cfg["share_delay_s"]), activity="share")
