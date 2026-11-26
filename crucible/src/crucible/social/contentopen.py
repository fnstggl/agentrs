"""The decision to keep watching a short video shown in a feed, from workstream E's fitted KuaiRand model, and the
platform's random-exposure experiment (MASTER_SPEC 5.5, SP-5.5b; contract in docs/v5/calibration/RUNTIME.md).

`ContentOpen` replaces the consumption engine's built-in open decision on a short-video platform (feed config
``calibrated: {"shorts": {"open": "content_open@1", "content_kind": "video_short"}}``) with the runtime decision
model `content_open@1` (fitted model content_consumption@1: KuaiRand-Pure, random-exposure rows), used through its
scoped handle (`calibration.runtime`, content kind ``video_short``). Watch time, likes, shares and follows stay the
engine's own (built-in) decisions. Every input is computed row by row from recorded state of the individual, the
item and the platform, in the contract's units:

    own_views_7d, own_opens_before, own_impressions_before, own_views_before  the viewer's counts on this platform
        (opens = items it decided to keep watching), from day-start snapshots: "before today" exactly;
    own_topic_views   the viewer's views on this platform, before today, of items whose primary topic (nearest topic
                      of the item's vector in the shared space) is this item's;
    item_views_7d, item_opens_before, item_impressions_before  the item's counts from day-start snapshots;
    active_individuals_7d  individuals with a feed impression on this platform in the previous 7 days (day start);
    topic_population_share (topic views + 1) / (views + topics) over the platform's items before today;
    item_age_days, item_duration_s (measured from the bytes), local_hour (simulated UTC hour);
    own_creator_views, own_item_exposures, own_item_consumed  flags (0/1) from the exact audience sets: the world
                      keeps sets, not counts, of these pairs; for own_creator_views and own_item_exposures the flag
                      is a lower bound of the count and covers today as well (documented deviation).

Each session's open decisions on the platform are recorded as one batch per arm (`open.<platform>`, provenance
`content_open@1`, columns `opened` and `own_topic` = the item's primary topic is one the viewer has viewed more
than its platform share, the in-world equivalent of KuaiRand's user_tag_lift > 0), inside the population's session.

Random exposure (feed config ``random_insert: {"rate": r, "max_age_days": 30}``, evaluator-side): with probability
r per feed a slot after the share inbox holds an item drawn uniformly at random from the platform's live items of
the last `max_age_days` days instead of a recommendation (impression source ``random``). The open decisions on
inserted slots carry ``extra.experiment = {"id": "random_exposure", "arm": "random"}``, the others arm
``recommended``; `crucible.calibration.inworld.random_exposure_experiment` compares the random arm with KuaiRand's
random-exposure log. Fidelity: FIDELITY_REGISTER `runtime:content_open@1`, `social-random-exposure`.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import ProtocolError
from ..substrate import columns
from . import audience as AUD

DAY = 86400
RING = 8
EXPERIMENT = "random_exposure"


def viewers_table(platform: str) -> str:
    return f"feed_co_{platform}_v"


def items_table(platform: str) -> str:
    return f"feed_co_{platform}_i"


def _topics(world, items: np.ndarray) -> np.ndarray:
    from .consumption import item_topics
    return item_topics(world, np.asarray(items, np.int64)) if np.size(items) else np.zeros(0, np.int64)


def _rows(a: np.ndarray, n: int) -> np.ndarray:
    """`a` padded with zero rows to at least n rows (items indexed after the last day-start snapshot)."""
    if a.shape[0] >= n:
        return a
    return np.concatenate([a, np.zeros((n - a.shape[0],) + a.shape[1:], a.dtype)])


def n_topics(world) -> int:
    from ..population import semantics
    return len(semantics.topic_names())


def declare(world, platform: str, max_items: int) -> None:
    """Column tables of the platform's history counters (owned by `feed`; never on the population table)."""
    n = columns.rows(world, "pop")
    T = n_topics(world)
    # the open decision's columns on the population table, named as the in-world contract requires (RUNTIME.md):
    # the individual's latest decision on a calibrated platform (every row is in the decision batch's blob)
    columns.declare(world, "pop", n, {"opened": ("bool", (), False), "own_topic": ("bool", (), False)})
    columns.declare(world, viewers_table(platform), n, {
        "imp": ("int32", (), 0), "opens": ("int32", (), 0), "views": ("int32", (), 0),
        "ds_imp": ("int32", (), 0), "ds_opens": ("int32", (), 0), "vring": ("int32", (RING,), 0),
        "topic_views": ("uint16", (T,), 0), "topic_ds": ("uint16", (T,), 0), "last_imp_day": ("int32", (), -1)})
    columns.declare(world, items_table(platform), int(max_items), {
        "opens": ("int64", (), 0), "ds_imp": ("int64", (), 0), "ds_opens": ("int64", (), 0),
        "vring": ("int64", (RING,), 0)})


def daystart(ctx, platform: str) -> None:
    """Day-start snapshots of the platform's counters (process `feed.daystart`): what 'before today' means."""
    w = ctx.world
    d = int(ctx.now // DAY)
    vt, it = viewers_table(platform), items_table(platform)
    nv = columns.rows(w, vt)
    v = {c: columns.col(w, vt, c) for c in ("imp", "opens", "views", "vring", "topic_views", "last_imp_day")}
    ring = v["vring"].copy()
    ring[:, d % RING] = v["views"]
    act = int(np.count_nonzero((v["last_imp_day"] >= d - 7) & (v["last_imp_day"] <= d - 1)))
    columns.batch(ctx, vt, "feed.daystart", np.arange(nv, dtype=np.int64),
                  {"ds_imp": v["imp"].copy(), "ds_opens": v["opens"].copy(), "vring": ring,
                   "topic_ds": v["topic_views"].copy()}, mode="set",
                  extra={"platform": platform, "day": d, "active_individuals_7d": act})
    ni = AUD.n_items(w)
    if ni:
        if ni > columns.rows(w, it):
            columns.grow(ctx, it, max(256, ni - columns.rows(w, it)))
        rows = np.arange(ni, dtype=np.int64)
        iring = columns.col(w, it, "vring")[rows].copy()
        iring[:, d % RING] = columns.col(w, AUD.ITEMS, "views")[rows]
        columns.batch(ctx, it, "feed.daystart", rows, {"ds_imp": columns.col(w, AUD.ITEMS, "imp")[rows].copy(),
                                                       "ds_opens": columns.col(w, it, "opens")[rows].copy(),
                                                       "vring": iring}, mode="set", extra={"platform": platform,
                                                                                           "day": d})
    w.state.meta_set(f"feed.co.{platform}.active7", act)


class ContentOpen:
    """content_open@1 on one short-video platform (see the module docstring)."""

    def __init__(self, decision: str, affinity_space: str | None = None, content_kind: str | None = None):
        from ..calibration import runtime
        if decision != "content_open@1":
            raise ProtocolError(f"ContentOpen serves content_open@1, not {decision}")
        self.dm = runtime.get(decision)
        self.handle = self.dm.scoped(content_kind=content_kind or "video_short")
        self.name = decision
        self.model = self.dm.fitted
        self.last = None

    def inputs(self, world, platform: str, idx: np.ndarray, feed: np.ndarray, has_item: np.ndarray):
        vt, it = viewers_table(platform), items_table(platform)
        rows, cols_ = np.nonzero(has_item)
        items = feed[rows, cols_]
        viewers = idx[rows]
        d = int(world.now // DAY)
        V = {c: columns.col(world, vt, c) for c in ("ds_imp", "ds_opens", "vring", "topic_ds")}
        I = {c: _rows(columns.col(world, it, c), AUD.n_items(world)) for c in ("ds_imp", "ds_opens", "vring")}
        own_before = V["vring"][viewers, d % RING].astype(np.float64)
        own_7d = np.maximum(own_before - V["vring"][viewers, (d - 7) % RING].astype(np.float64), 0.0)
        if d < 7:
            own_7d = own_before
        it_before = I["vring"][items, d % RING].astype(np.float64)
        it_7d = it_before if d < 7 else np.maximum(it_before - I["vring"][items, (d - 7) % RING], 0.0)
        t_item = _topics(world, items)
        tds = V["topic_ds"][viewers, t_item].astype(np.float64)
        on = AUD.items_on(world, platform, live_only=False)
        T = V["topic_ds"].shape[1]
        on_views = I["vring"][on, d % RING].astype(np.float64)
        share_t = (np.bincount(_topics(world, on), weights=on_views, minlength=T) + 1.0) / (on_views.sum() + T)
        from .consumption import rowwise_member
        st = world.state
        keys = np.where(has_item, feed, -1)
        cre = np.where(has_item, columns.col(world, AUD.ITEMS, "creator")[np.where(has_item, feed, 0)], -1)
        flag = lambda kind, kk: rowwise_member(world, kind, kk, idx)[rows, cols_].astype(np.float64)  # noqa: E731
        age = np.maximum(world.now - columns.col(world, AUD.ITEMS, "published_at")[items], 0) / DAY
        inp = {"own_views_7d": own_7d, "own_opens_before": V["ds_opens"][viewers].astype(np.float64),
               "own_impressions_before": V["ds_imp"][viewers].astype(np.float64),
               "item_views_7d": it_7d.astype(np.float64),
               "active_individuals_7d": np.full(rows.size, float(st.meta.get(f"feed.co.{platform}.active7", 0))),
               "item_opens_before": I["ds_opens"][items].astype(np.float64),
               "item_impressions_before": I["ds_imp"][items].astype(np.float64),
               "item_age_days": age.astype(np.float64),
               "item_duration_s": columns.col(world, AUD.ITEMS, "duration_s")[items].astype(np.float64),
               "own_topic_views": tds, "own_views_before": own_before,
               "topic_population_share": share_t[t_item],
               "own_creator_views": flag("cv", cre), "own_item_exposures": flag("ex", keys),
               "own_item_consumed": flag("vw", keys),
               "local_hour": np.full(rows.size, float((world.now % DAY) // 3600))}
        own_share = np.where(own_before > 0, tds / np.maximum(own_before, 1.0), 0.0)
        own_topic = (own_before > 0) & (own_share > share_t[t_item])
        return inp, (rows, cols_), t_item, own_topic

    def p_open(self, world, platform: str, idx: np.ndarray, feed: np.ndarray, aff: np.ndarray,
               has_item: np.ndarray) -> np.ndarray:
        out = np.zeros(feed.shape)
        self.last = None
        if not np.any(has_item):
            return out
        inp, (rows, cols_), t_item, own_topic = self.inputs(world, platform, idx, feed, has_item)
        out[rows, cols_] = self.handle.evaluate(inp).values
        ot = np.zeros(feed.shape, bool)
        ot[rows, cols_] = own_topic
        tt = np.full(feed.shape, -1, np.int64)
        tt[rows, cols_] = t_item
        self.last = {"own_topic": ot, "topic": tt}
        return out

    def record(self, ctx, platform: str, idx: np.ndarray, feed: np.ndarray, src: np.ndarray, reached: np.ndarray,
               decided: np.ndarray, view: np.ndarray) -> None:
        """The session's open decisions (one batch per arm) and the platform's history counters."""
        from .. import population
        w = ctx.world
        if self.last is None or not np.any(reached):
            return
        vr, pr = np.nonzero(reached)
        rnd = src[vr, pr] == AUD.SOURCE_CODE["random"]
        prov = {"model_id": "content_open", "model_version": "1", "fidelity_id": self.dm.fidelity_id}
        for arm, m in (("random", rnd), ("recommended", ~rnd)):
            if not m.any():
                continue
            population.record_decisions(ctx, f"open.{platform}", idx[vr[m]],
                                        {"opened": decided[vr[m], pr[m]], "own_topic": self.last["own_topic"][vr[m],
                                                                                                          pr[m]]},
                                        **prov, extra={"platform": platform, "rows": int(m.sum()),
                                                       "experiment": {"id": EXPERIMENT, "arm": arm}})
        # history counters of the viewers and the items (consequences of the decisions just recorded)
        vt, it = viewers_table(platform), items_table(platform)
        rows = np.flatnonzero(reached.any(axis=1))
        imp = columns.col(w, vt, "imp")[idx[rows]].astype(np.int64) + reached[rows].sum(axis=1)
        opens = columns.col(w, vt, "opens")[idx[rows]].astype(np.int64) + decided[rows].sum(axis=1)
        views = columns.col(w, vt, "views")[idx[rows]].astype(np.int64) + view[rows].sum(axis=1)
        tv = columns.col(w, vt, "topic_views")[idx[rows]].astype(np.int64)
        vv, pv = np.nonzero(view[rows])
        if vv.size:
            np.add.at(tv, (vv, self.last["topic"][rows][vv, pv]), 1)
        d = int(ctx.now // DAY)
        columns.batch(ctx, vt, f"feed.co.{platform}", idx[rows],
                      {"imp": imp.astype(np.int32), "opens": opens.astype(np.int32), "views": views.astype(np.int32),
                       "topic_views": np.minimum(tv, 65535).astype(np.uint16),
                       "last_imp_day": np.full(rows.size, d, np.int32)}, mode="set")
        vd, pd_ = np.nonzero(decided)
        if vd.size:
            its = feed[vd, pd_]
            u, c = np.unique(its, return_counts=True)
            if int(u.max()) >= columns.rows(w, it):
                columns.grow(ctx, it, max(256, int(u.max()) + 1 - columns.rows(w, it)))
            columns.batch(ctx, it, f"feed.co.{platform}", u,
                          {"opens": columns.col(w, it, "opens")[u].astype(np.int64) + c}, mode="set")


def insert_random(world, platform: str, feed: np.ndarray, src: np.ndarray, gen: np.ndarray, n_in: np.ndarray,
                  rng, rate: float, max_age_days: float) -> None:
    """In place: with probability `rate` per feed, one slot after the share inbox holds an item drawn uniformly at
    random from the platform's live items of the last `max_age_days` days (impression source 'random')."""
    cand = AUD.items_on(world, platform)
    if cand.size:
        pub = columns.col(world, AUD.ITEMS, "published_at")[cand]
        cand = cand[pub >= world.now - max_age_days * DAY]
    insert_into(feed, src, gen, n_in, cand, rng, rate)


def insert_into(feed: np.ndarray, src: np.ndarray, gen: np.ndarray, n_in: np.ndarray, cand: np.ndarray, rng,
                rate: float) -> None:
    """The insertion itself (pure): three uniform draws per feed (insert?, slot, item) whatever the outcome."""
    V, k = feed.shape
    if V == 0 or k == 0 or rate <= 0:
        return
    draw = rng.random(V)
    pos = rng.random(V)
    pick = rng.random(V)
    if cand.size == 0:
        return
    for v in np.flatnonzero(draw < rate):
        lo = int(n_in[v])
        if lo >= k:
            continue
        item = int(cand[min(int(pick[v] * cand.size), cand.size - 1)])
        if np.any(feed[v] == item):
            continue
        s = lo + min(int(pos[v] * (k - lo)), k - lo - 1)
        feed[v, s] = item
        src[v, s] = AUD.SOURCE_CODE["random"]
        gen[v, s] = 0
