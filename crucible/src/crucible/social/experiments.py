"""Evaluator-side experiment drivers for the social internet (workstream C2; Gates D, I, SP-4.5, SP-4.6).

Builds worlds, runs them, forks controlled arms and reads the recorded decision batches back out of the event log.
Nothing here is reachable from world code; it only reads what the world recorded.
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import World, WorldSpec
from ..substrate import artifacts
from . import audience as AUD

DAY = 86400

# a mixture per platform with every recommender in use somewhere, and one scheduled change (day 3 on posts)
GATE_D_PLATFORMS = {
    "posts": {"k": 10, "pool": 240,
              "recommenders": [{"name": "interest", "weight": 1.0, "params": {"need_weight": 0.3173}},
                               {"name": "social", "weight": 0.9}, {"name": "chronological", "weight": 0.6},
                               {"name": "engagement", "weight": 0.8, "params": {"delay_s": 21611}},
                               {"name": "cf", "weight": 0.6173}, {"name": "explore", "weight": 0.3}],
              "schedule": [{"day": 3, "recommenders": [{"name": "fresh", "weight": 1.0,
                                                        "params": {"half_life_s": 41111}},
                                                       {"name": "trending", "weight": 0.9},
                                                       {"name": "explore", "weight": 0.6},
                                                       {"name": "social", "weight": 0.5}]}]},
    "articles": {"k": 8, "pool": 200,
                 "recommenders": [{"name": "search", "weight": 1.0, "params": {"query_terms": 3}},
                                  {"name": "diversity", "weight": 0.8, "params": {"lam": 0.6299}},
                                  {"name": "reputation", "weight": 0.6}, {"name": "interest", "weight": 0.5}]},
}
# values that appear only in the evaluator-side configuration (hidden-parameter probes look for them)
# (numbers set only in this configuration, parameter names, and the names of the algorithms in use)
SECRET_MARKERS = ("0.3173", "21611", "0.6173", "41111", "0.6299", "half_life_s", "need_weight", "velocity_weight",
                  "query_terms", "rrf_k", "explore", "engagement", "trending", "chronological", "diversity",
                  "reputation")


def spec(family: str, seed: int, *, n: int = 3000, creators: int = 40, posts_per_day: float = 2.0,
         platforms: dict | None = None, crowd: dict | None = None, feed: dict | None = None,
         aud: dict | None = None, identity: bool = True) -> WorldSpec:
    platforms = GATE_D_PLATFORMS if platforms is None else platforms
    mods = ([{"name": "identity"}, {"name": "bank"}] if identity else []) + [
        {"name": "artifacts"}, {"name": "cols"},
        {"name": "crowd", "config": {"n": n, "creators": creators, "posts_per_day": posts_per_day,
                                     **(crowd or {})}},
        {"name": "aud", "config": {"platforms": sorted(platforms), **(aud or {})}},
        {"name": "feed", "config": {"platforms": platforms, **(feed or {})}}]
    return WorldSpec(family, seed, mods)


def run(world: World, until: int, *, step: int = 900, check=None) -> list:
    """Advance to `until` in steps; `check(world)` (if given) runs after every step and its results are returned."""
    out = []
    t = world.now
    while t < until:
        t = min(until, t + step)
        world.advance_time(t)
        if check is not None:
            out.append(check(world))
    return out


def set_mixture(world: World, platform: str, entries: list[dict]) -> None:
    world.system_call("feed.set_mixture", {"platform": platform, "recommenders": entries})


def decision_rows(world: World, kind: str, since_seq: int = 0, platform: str | None = None) -> dict:
    """Rows of the recorded impression (kind 'imp') or consumption ('consume') batches after `since_seq`."""
    et = "aud.impressed" if kind == "imp" else "aud.consumed"
    parts: dict[str, list] = {}
    for e in world.log:
        if e.type != et or e.seq <= since_seq or (platform is not None and e.payload["platform"] != platform):
            continue
        a = artifacts.get_array_blob(world, e.payload["ref"])
        cols = {"viewer": a["idx"].astype(np.int64), "seq": np.full(a["idx"].size, e.seq, np.int64),
                "t": np.full(a["idx"].size, e.t, np.int64)}
        if kind == "imp":
            cols.update(item=a["v.aud_i_item"].astype(np.int64), slot=a["v.aud_i_slot"].astype(np.int64),
                        src=a["v.aud_i_src"].astype(np.int64), gen=a["v.aud_i_gen"].astype(np.int64))
        else:
            cols.update(item=a["v.aud_c_item"].astype(np.int64), watch_s=a["v.aud_c_watch_s"].astype(np.float64),
                        flags=a["v.aud_c_flags"].astype(np.int64))
        for k, v in cols.items():
            parts.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if not out:
        z = np.zeros(0, np.int64)
        out = {"viewer": z, "item": z, "seq": z, "t": z}
    if kind == "consume" and out["item"].size:
        dur = AUD.item_col(world, "duration_s")[out["item"]].astype(np.float64)
        cre = AUD.item_col(world, "creator")[out["item"]].astype(np.int64)
        cpop = AUD.creator_col(world, "pop_index")[cre]
        out["is_view"] = (out["watch_s"] >= AUD.view_threshold(dur)) & (cpop != out["viewer"])
    return out


def pair_keys(world: World, rows: dict, mask=None) -> np.ndarray:
    n = AUD.n_people(world)
    it, v = rows["item"], rows["viewer"]
    if mask is not None:
        it, v = it[mask], v[mask]
    return np.unique(it * n + v)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / float(a.size + b.size - inter)


def mixture_changes(world: World) -> int:
    """Distribution changes recorded during the run (scheduled versions after the initial one, and overrides)."""
    return sum(int(v) for k, v in world.state.meta.items() if k.startswith("feed.changes."))
