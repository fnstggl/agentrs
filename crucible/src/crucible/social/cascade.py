"""Large-scale reach experiments (workstream C2; MASTER_SPEC Part 4.6, Gate I). Evaluator side.

* `internet_spec(n, seed, positive_control=...)` - an internet-profile world (10M+ identities in columns) on the
  ordinary engine: crowd population, population creators, one posts platform, ordinary recommenders and sharing.
  With `positive_control=True` the experiment is designed so that a cascade is expected: one topic is popular with
  a third of the communities, the creator most interested in it publishes a focused post on it at the start
  (item 0, the control item), the platform's mixture gives weight to trending, and sharing is more common. Nothing
  assigns a view: the item reaches people only through feeds and shares, and each person decides.
* `measure(world, item)` - the Gate I measurements, every one recomputed from the recorded batches:
  distinct identities, exact exposure and distinct-viewer recounts against the state, attention ledger and daily
  consumption against budgets, views without a prior exposure (forced views), share generations, and the reach of
  one item.

No realism is claimed: these runs demonstrate computational capacity and that large reach can arise from the
ordinary mechanisms, while the behaviour, interest and graph models are unvalidated (FIDELITY_REGISTER), so the size
and shape of the cascades are not claimed to be empirically realistic.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.kernel import World, WorldSpec
from ..substrate import attention, columns, ids
from . import audience as AUD

DAY = 86400

SCALE_BEHAVIOUR = {"gap_mean_s": 24 * 3600.0, "taste_drift": 0.0}


def internet_spec(n: int, seed: int, *, positive_control: bool, creators: int = 500, k: int = 8,
                  pool: int = 200) -> WorldSpec:
    mix = [{"name": "interest", "weight": 1.0}, {"name": "social", "weight": 0.8}, {"name": "fresh", "weight": 0.4},
           {"name": "explore", "weight": 0.3}]
    crowd = {"n": n, "creators": creators, "posts_per_day": 0.5, "community_size": 500, "init_chunk": 1_000_000,
             "first_active_spread_s": DAY, "share_propensity": [1.0, 0.6], "creator_platforms": {"posts": 1.0}}
    if positive_control:
        mix = [{"name": "trending", "weight": 1.0}, {"name": "interest", "weight": 1.0},
               {"name": "social", "weight": 0.8}, {"name": "explore", "weight": 0.3}]
        others = 0.67 / 23.0
        crowd.update({"topic_popularity": {"football": 0.33, **{t: others for t in _other_topics("football")}},
                      "share_propensity": [2.5, 0.6],
                      "seed_posts": [{"topic": "football", "platform": "posts", "focus": 0.95}]})
    platforms = {"posts": {"k": k, "pool": pool, "pool_days": 3, "inbox_slots": 2, "recommenders": mix}}
    mods = [{"name": "artifacts"}, {"name": "cols"}, {"name": "crowd", "config": crowd},
            {"name": "aud", "config": {"platforms": ["posts"], "max_items": 8192, "max_creators": creators + 64,
                                       "analytics_interval_s": 3 * 3600}},
            {"name": "feed", "config": {"platforms": platforms, "behaviour": SCALE_BEHAVIOUR,
                                        "train_interval_s": DAY}}]
    return WorldSpec("internet-positive-control" if positive_control else "internet-organic", seed, mods,
                     {"log_retain": 10_000_000})


def _other_topics(t):
    from .vocab import TOPIC_NAMES
    return [x for x in TOPIC_NAMES if x != t]


def run(world: World, until: int, *, step: int = 3600, progress=None) -> list[dict]:
    """Advance to `until`; after every `step` record progress (time, wall seconds, impressions, views, reach of the
    most viewed item)."""
    out = []
    t0 = time.time()
    t = world.now
    while t < until:
        t = min(until, t + step)
        s0 = world.seq
        world.advance_time(t)
        failed = [e.payload for e in world.events_range(s0) if e.type == "proc.failed"]
        if failed:
            raise RuntimeError(f"a process failed during the run: {failed[0]['proc']}: {failed[0]['error'][:300]}")
        ni = AUD.n_items(world)
        viewers = AUD.item_col(world, "viewers")
        rec = {"t": int(t), "wall_s": round(time.time() - t0, 1), "items": ni,
               "impressions": int(AUD.item_col(world, "imp").sum()), "views": int(AUD.item_col(world, "views").sum()),
               "distinct_views": int(viewers.sum()), "item0_unique_viewers": int(viewers[0]) if ni else 0,
               "top_item": int(np.argmax(viewers)) if ni else -1,
               "top_unique_viewers": int(viewers.max()) if ni else 0}
        out.append(rec)
        if progress is not None:
            progress(rec)
    return out


def share_generations(world: World) -> dict:
    """Generations of share-driven exposure that led to a qualifying view: an impression from a share carries the
    generation of the share (1 = shared by someone who found the item through a feed, 2 = shared by someone who got
    it through a share, ...). Returns {generation: qualifying views}."""
    from .experiments import decision_rows
    imp = decision_rows(world, "imp", 0)
    con = decision_rows(world, "consume", 0)
    n = AUD.n_people(world)
    share = imp["src"] == AUD.SOURCE_CODE["share"]
    k_imp = imp["item"][share] * n + imp["viewer"][share]
    g_imp = imp["gen"][share]
    order = np.lexsort((g_imp, k_imp))              # the lowest generation per (item, viewer)
    k_s, g_s = k_imp[order], g_imp[order]
    first = np.r_[True, k_s[1:] != k_s[:-1]] if k_s.size else np.zeros(0, bool)
    k_u, g_u = k_s[first], g_s[first]
    kv = con["item"][con["is_view"]] * n + con["viewer"][con["is_view"]]
    kv = np.unique(kv)
    pos = np.searchsorted(k_u, kv)
    pos_c = np.minimum(pos, max(k_u.size - 1, 0))
    hit = (k_u.size > 0) & (k_u[pos_c] == kv) if k_u.size else np.zeros(kv.size, bool)
    gens = g_u[pos_c[hit]] if k_u.size else np.zeros(0, np.int64)
    return {int(g): int(c) for g, c in zip(*np.unique(gens, return_counts=True))}


def distinct_state_rows(world: World, chunk: int = 1_000_000) -> int:
    """Number of distinct individual state rows (interests and attention budget), by a 64-bit hash per row."""
    from .scale import splitmix64
    n = AUD.n_people(world)
    hs = []
    for s in range(0, n, chunk):
        u = columns.col(world, "pop", "interests")[s:s + chunk].view(np.uint16).astype(np.uint64)
        b = columns.col(world, "pop", "attention_budget_min")[s:s + chunk].view(np.uint32).astype(np.uint64)
        h = splitmix64(b)
        with np.errstate(over="ignore"):
            for d in range(u.shape[1]):
                h = splitmix64(h ^ (u[:, d] + np.uint64(d * 65537)))
        hs.append(h)
    return int(np.unique(np.concatenate(hs)).size)


def measure(world: World, item: int | None = None) -> dict:
    """Gate I measurements of a run (see the module docstring)."""
    from .experiments import decision_rows
    t0 = time.time()
    rc = AUD.recount(world)
    n = AUD.n_people(world)
    att = attention.check(world)
    con = decision_rows(world, "consume", 0)
    day = con["t"] // DAY
    key = day * n + con["viewer"]
    uk, inv = np.unique(key, return_inverse=True)
    watched_min = np.bincount(inv, weights=con["watch_s"]) / 60.0
    budget = columns.col(world, "pop", "attention_budget_min").astype(np.float64)
    over = int(np.sum(watched_min > budget[uk % n] * (1 + 1e-5) + 1e-3))
    gens = share_generations(world)
    viewers = AUD.item_col(world, "viewers")
    top = int(np.argmax(viewers))
    item = top if item is None else int(item)
    exposed_or_viewed = np.unique(np.concatenate([decision_rows(world, "imp", 0)["viewer"], con["viewer"]]))
    out = {"distinct_identities": n, "identities_with_decisions": int(exposed_or_viewed.size),
           "exposures_checked": rc["exposures_checked"], "exposure_recount_mismatches": rc["exposure_recount_mismatches"],
           "distinct_viewers_checked": rc["distinct_viewers_checked"],
           "distinct_viewer_recount_mismatches": rc["distinct_viewer_recount_mismatches"],
           "total_recount_mismatches": rc["total_mismatches"], "impression_rows": rc["impression_rows"],
           "consumption_rows": rc["consumption_rows"], "qualifying_views": rc["views"],
           "qualifying_distinct_consumption_events": rc["distinct_viewers_checked"],
           "attention_ledger_violations": len(att), "daily_consumption_over_budget": over,
           "attention_budget_violations": len(att) + over,
           "forced_view_assignments": rc["views_without_prior_exposure"],
           "share_generations_observed": max(gens) if gens else 0,
           "share_generation_views": {str(k): v for k, v in gens.items()},
           "item": item, "item_unique_viewers": int(viewers[item]),
           "item_unique_exposed": int(AUD.item_col(world, "exposed")[item]),
           "top_item": top, "top_item_unique_viewers": int(viewers[top]), "items": AUD.n_items(world),
           "measure_wall_s": round(time.time() - t0, 1)}
    return out


def materialise_sample(world: World, k: int = 1000, seed: int = 0) -> int:
    """Turn k random individuals into kernel principals (each has its own row; nothing is copied). Returns how many
    principals of kind person exist afterwards for those rows."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(AUD.n_people(world), size=k, replace=False))
    world._run_system("materialise.sample", lambda ctx: [ids.ensure_principal(ctx, int(i)) for i in pick])
    return sum(1 for i in pick if ids.principal_of(world, int(i)) is not None)
