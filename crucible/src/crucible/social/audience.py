"""Audience accounting (workstream C2; INTERFACE_CONTRACTS section 3, MASTER_SPEC Part 4.6, DECISION_LOG D-5).

Kernel module `aud` (host `aud`). It keeps, for every content item, exact audience state that changes only through
recorded decisions of the deciding individuals:

* `impressions(ctx, platform, viewers, items, slot)` - items placed in front of individuals (one row per impression);
* `consumption(ctx, platform, viewers, items, watch_s, completed, shared, followed, liked)` - what individuals did
  with items they were exposed to (one row per viewing);

each recorded as ONE `cols.batch` of kind `impress.<platform>` / `consume.<platform>` on the population table (its
blob holds every decision row: viewer, item, slot or watch time and decision flags; the batch also sets each
individual's "last impression / last consumption" columns), followed by one `aud.impressed` / `aud.consumed` event
that references the same blob. This module's appliers read that blob and update the per-item exact sets and counters.
Nothing else writes them: there is no tool or function through which a creator (or anyone) can set a counter.

Audience quantities per item (`QUANTITIES`):

    impressions        impression rows
    unique_exposed     distinct identities with at least one impression (exact set)
    views              consumption rows that qualify as a view (watch_s >= min(30 s, 0.5 x duration_s))
    unique_viewers     distinct identities with at least one qualifying view (exact set)
    watch_s            seconds watched, all consumption rows (qualifying or not)
    completions        qualifying views that the viewer completed
    repeat_views       qualifying views by someone who had already viewed the item (views - unique_viewers)
    shares             voluntary share decisions
    subscribers        follows (subscriptions to the creator) newly made from this item
    returning_viewers  unique viewers who, at their first view of this item, had already viewed another item of the
                       same creator

THE VIEW THRESHOLD (`VIEW_RULE`: a view requires watch_s >= min(30 s, 0.5 x duration_s)): a consumption row is a view iff `watch_s >= min(VIEW_MIN_S, VIEW_MIN_FRAC * duration_s)` with
VIEW_MIN_S = 30 and VIEW_MIN_FRAC = 0.5, and the viewer is not the item's creator (self-consumption is recorded and
spends attention but never counts). An exposure is never a view; a view requires a prior exposure of the same
individual to the same item (enforced: `consumption` refuses rows without one). Likes count once, at the first view.

`audience(world, item, principal=None)` returns the exact values for the evaluator. A creator (principal = the
item's creator) sees the analytics the platform publishes: values staged at one analytics tick and published with
multiplicative log-normal noise at the next (so they are between one and two `analytics_interval_s` old); the noisy
value is drawn once per publication, so asking again gives the same number. Anyone else sees the public view count
only (same delay and noise).

`recount(world)` re-derives every quantity from the recorded batch blobs alone, with an independent implementation,
and compares it with the state (evaluator side; Gate I.2, I.3, I.7, D.5).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, InvariantViolation, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..substrate import artifacts, bus, columns, ids
from . import idset, textrep

VIEW_MIN_S = 30.0
VIEW_MIN_FRAC = 0.5
VIEW_RULE = "a view requires watch_s >= min(30 s, 0.5 x duration_s)"
COMPLETE_FRAC = 0.95            # `completed` is accepted only for rows that watched at least 95% of the duration
WORDS_PER_S = 4.0               # reading speed used for the duration of text items (240 words per minute)
QUANTITIES = ("impressions", "unique_exposed", "views", "unique_viewers", "watch_s", "completions", "repeat_views",
              "shares", "subscribers", "returning_viewers")
# counter column of each quantity in table aud_items
_QCOL = {"impressions": "imp", "unique_exposed": "exposed", "views": "views", "unique_viewers": "viewers",
         "watch_s": "watch_s", "completions": "completions", "repeat_views": "repeats", "shares": "shares",
         "subscribers": "follows", "returning_viewers": "returning"}
KINDS = ("video_short", "video_long", "post", "image", "article", "newsletter", "audio", "thread", "app", "package")
F_COMPLETED, F_SHARED, F_FOLLOWED, F_LIKED = 1, 2, 4, 8

# impression sources (recorded with every impression row; evaluator side only)
SOURCES = ("unknown", "direct", "search_tool", "share", "chronological", "interest", "cf", "engagement", "explore",
           "social", "diversity", "reputation", "search", "fresh", "trending", "random")
SOURCE_CODE = {s: i for i, s in enumerate(SOURCES)}

POP = "pop"
ITEMS = "aud_items"
CREATORS = "aud_creators"
_POP_COLS = {
    # last impression (set by every impress.<platform> batch; the blob keeps every row)
    "aud_i_item": ("int32", (), -1), "aud_i_slot": ("int16", (), -1), "aud_i_src": ("int8", (), 0),
    "aud_i_gen": ("uint8", (), 0), "aud_i_t": ("int64", (), -1),
    # last consumption (set by every consume.<platform> batch; the blob keeps every row)
    "aud_c_item": ("int32", (), -1), "aud_c_watch_s": ("float32", (), 0.0), "aud_c_flags": ("uint8", (), 0),
    "aud_c_t": ("int64", (), -1),
}
_COUNTERS = {"imp": "int64", "exposed": "int64", "views": "int64", "viewers": "int64", "watch_s": "float64",
             "completions": "int64", "repeats": "int64", "shares": "int64", "follows": "int64", "returning": "int64",
             "likes": "int64", "self_rows": "int64"}


def _item_cols(dim: int) -> dict:
    cols = {"creator": ("int32", (), -1), "platform": ("int16", (), -1), "kind": ("int8", (), -1),
            "duration_s": ("float32", (), 0.0), "published_at": ("int64", (), -1), "removed": ("bool", (), False),
            "emb": ("float16", (dim,), 0.0), "pub_t": ("int64", (), -1), "stg_t": ("int64", (), -1),
            "prev_views": ("int64", (), 0), "vel_views": ("float32", (), 0.0),
            # artifact-intrinsic legibility in [0, 1] (population.semantics.legibility; 1.0 for the lexical path)
            "legib": ("float32", (), 1.0), "version": ("int32", (), 0)}
    for c, dt in _COUNTERS.items():
        cols[c] = (dt, (), 0)
    for q in QUANTITIES:
        cols[f"pub_{_QCOL[q]}"] = ("float64", (), 0.0)
        cols[f"stg_{_QCOL[q]}"] = ("float64", (), 0.0)
    return cols


_CREATOR_COLS = {"pop_index": ("int64", (), -1), "followers": ("int64", (), 0), "viewers": ("int64", (), 0),
                 "items": ("int32", (), 0), "imp": ("int64", (), 0), "views": ("int64", (), 0),
                 "completions": ("int64", (), 0), "follows": ("int64", (), 0), "shares": ("int64", (), 0),
                 "watch_s": ("float64", (), 0.0)}

DEFAULTS = {"platforms": ["posts", "articles"], "max_items": 4096, "max_creators": 1024,
            "analytics_interval_s": 21600, "analytics_noise": 0.05, "batch_retention_days": 14,
            "encoder": "auto", "dim": textrep.DIM}
#: decision provenance of the consumption engine's built-in behaviour model (FIDELITY_REGISTER: plausible, unvalidated)
ENGINE_DECISION = {"model_id": "c2-consumption", "model_version": "2", "fidelity_id": "social-consumption-builtin"}


def view_threshold(duration_s) -> np.ndarray:
    """Seconds a consumption row must reach to count as a view: min(30 s, 50% of the duration)."""
    return np.minimum(VIEW_MIN_S, VIEW_MIN_FRAC * np.asarray(duration_s, dtype=np.float64))


def _mod(world) -> "Audience":
    m = world.modules.get("aud")
    if m is None:
        raise ProtocolError("this world has no 'aud' module")
    return m  # type: ignore[return-value]


def n_items(world) -> int:
    return int(world.state.meta.get("aud.n_items", 0))


def n_people(world) -> int:
    return columns.rows(world, POP)


def item_col(world, name: str) -> np.ndarray:
    return columns.col(world, ITEMS, name)[:n_items(world)]


#: what the content catalogue owns about a distributed item (`content.content`); the audience module keeps a copy only
#: for items without a catalogue record (the crowd stand-in's scale path)
CATALOGUE_FIELDS = ("ref", "kind", "platform", "creator", "title", "tags", "published_at")


def info(world, item) -> dict | None:
    """An item's facts: the audience module's own attributes (content_id, creator code, duration, terms, words,
    removed) and, for a catalogue item, its catalogue record's metadata (ref, kind, platform, creator, title, tags,
    published_at) read from `content.content` - the one owner (D-21 content_artifact migration; no copy here)."""
    rec = world.state.get("aud_item_info", str(int(item)))
    if rec is None:
        return None
    cid = rec.get("content_id")
    if cid is not None and "title" not in rec:
        c = world.state.tables["content"][cid]
        rec = {**rec, "ref": c["ref"], "kind": c["kind"], "platform": c["platform"], "creator": c["creator"],
               "title": c["title"], "tags": list(c["tags"]), "published_at": int(c["published_at"])}
    return dict(rec)


def items_of(world, principal: str) -> list[int]:
    """Item ids of a creator (from the per-item creator codes)."""
    code = creator_code(world, principal)
    if code is None:
        return []
    return [int(i) for i in np.flatnonzero(item_col(world, "creator") == int(code))]


def creator_col(world, name: str) -> np.ndarray:
    return columns.col(world, CREATORS, name)[:int(world.state.meta.get("aud.n_creators", 0))]


# ---------------------------------------------------------------------------------------------------- catalogue
def creator_code(world, principal: str) -> int | None:
    rec = world.state.find_one("aud_creator_info", "principal", principal)
    return None if rec is None else int(rec["code"])


def creator_principal(world, code: int) -> str:
    rec = world.state.get("aud_creator_info", str(int(code)))
    if rec is None:
        raise NotFound(f"no creator {code}")
    return rec["principal"]


def ensure_creator(ctx: Ctx, principal: str) -> int:
    """The creator code of `principal` (registered on first use)."""
    c = creator_code(ctx.world, principal)
    if c is not None:
        return c
    if not ctx.state.has("principals", principal):
        raise NotFound(f"no principal {principal}")
    code = int(ctx.state.meta.get("aud.n_creators", 0))
    if code >= columns.rows(ctx.world, CREATORS):
        columns.grow(ctx, CREATORS, max(64, columns.rows(ctx.world, CREATORS)))
    ctx.emit("aud.creator", {"code": code, "principal": principal, "pop_index": ids.index(principal)}, vis="public")
    return code


def index_item(ctx: Ctx, ref: str, kind: str, platform: str, creator: str, title: str, *, tags: Iterable[str] = (),
               duration_s: float | None = None, content_id: str | None = None) -> int:
    """Index a published content artifact for distribution and audience accounting; returns its item id.

    The representation is computed once here from the artifact's bytes (never from a creator-supplied score). The
    content catalogue (workstream C1's `content.publish`) calls this for every published item; population creators
    of the scale path call it directly after storing and publishing their artifact."""
    mod = _mod(ctx.world)
    if kind not in KINDS:
        raise InvalidAction(f"kind must be one of {KINDS}")
    pcode = mod.platform_code(platform)
    a = artifacts.info(ctx.world, ref)
    if not a["published"]:
        raise InvalidAction("only published artifacts can be distributed (artifact.publish first)")
    if not artifacts.readable(ctx.world, creator, ref):
        raise Forbidden(f"{creator} cannot distribute {ref}")
    emb, terms, words, legib, rep = mod.encode(ctx.world, ref, a["mime"], ctx=ctx,
                                               caption=" ".join(x for x in (str(title), " ".join(tags)) if x))
    if duration_s is None:
        if not a["mime"].startswith("text/"):
            raise InvalidAction("duration_s is required for non-text content")
        duration_s = max(5.0, words / WORDS_PER_S)
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0 or duration_s > 86400:
        raise InvalidAction("duration_s must be in (0, 86400]")
    title = str(title)[:200]
    tags = [str(t)[:40] for t in list(tags)[:16]]
    code = ensure_creator(ctx, creator)
    item = n_items(ctx.world)
    if item >= columns.rows(ctx.world, ITEMS):
        columns.grow(ctx, ITEMS, max(256, columns.rows(ctx.world, ITEMS)))
    ctx.emit("aud.item", {"item": item, "ref": ref, "kind": kind, "platform": platform, "creator": creator,
                          "creator_code": code, "title": title, "tags": tags, "duration_s": duration_s,
                          "content_id": content_id, "terms": terms, "words": int(words),
                          "emb": [round(float(x), 6) for x in emb], "legib": round(float(legib), 6), "rep": rep},
             vis="internal")         # the platform's representation of the item (emb, terms) is never observable
    return item


def reindex_item(ctx: Ctx, item: int, ref: str, *, title: str | None = None, tags=None,
                 duration_s: float | None = None) -> None:
    """A new version of a distributed item (content.update): its representation, terms, words, legibility and
    duration are recomputed from the new bytes, so distribution and choice follow what the item is now."""
    mod = _mod(ctx.world)
    rec = info(ctx.world, item)
    if rec is None:
        raise NotFound(f"no item {item}")
    a = artifacts.info(ctx.world, ref)
    title = rec["title"] if title is None else str(title)[:200]
    tags = list(rec["tags"]) if tags is None else [str(t)[:40] for t in list(tags)[:16]]
    emb, terms, words, legib, rep = mod.encode(ctx.world, ref, a["mime"], ctx=ctx,
                                               caption=" ".join(x for x in (title, " ".join(tags)) if x))
    if duration_s is None:
        duration_s = max(5.0, words / WORDS_PER_S) if a["mime"].startswith("text/") else float(rec["duration_s"])
    ctx.emit("aud.item_updated", {"item": int(item), "ref": ref, "title": title, "tags": tags,
                                  "duration_s": float(duration_s), "terms": terms, "words": int(words),
                                  "emb": [round(float(x), 6) for x in emb], "legib": round(float(legib), 6),
                                  "rep": rep}, vis="internal")


def remove_item(ctx: Ctx, item: int) -> None:
    rec = info(ctx.world, item)
    if rec is None:
        raise NotFound(f"no item {item}")
    if rec["removed"]:
        return
    ctx.emit("aud.item_removed", {"item": int(item)}, vis="public")


def items_on(world, platform: str, *, live_only: bool = True) -> np.ndarray:
    """Item ids of a platform (not removed)."""
    m = _mod(world)
    pc = m.platform_code(platform)
    p = item_col(world, "platform")
    mask = p == pc
    if live_only:
        mask &= ~item_col(world, "removed")
    return np.flatnonzero(mask).astype(np.int64)


# ---------------------------------------------------------------------------------------------------- decisions
def _arr(x, name, dtype, n=None):
    a = np.asarray(x)
    if a.ndim != 1:
        raise InvalidAction(f"{name} must be one-dimensional")
    if n is not None and a.shape[0] != n:
        raise InvalidAction(f"{name} has {a.shape[0]} rows, viewers has {n}")
    if dtype == "int":
        if a.size and a.dtype.kind not in "iu":
            raise InvalidAction(f"{name} must be integers")
        return a.astype(np.int64, copy=False)
    if dtype == "bool":
        if a.size and a.dtype.kind not in "bui":
            raise InvalidAction(f"{name} must be booleans")
        return a.astype(bool, copy=False)
    a = a.astype(np.float64, copy=False)
    if a.size and not np.all(np.isfinite(a)):
        raise InvalidAction(f"{name} must be finite")
    return a


def _check_items(world, platform: str, items: np.ndarray) -> np.ndarray:
    mod = _mod(world)
    pc = mod.platform_code(platform)
    ni = n_items(world)
    if items.size and (int(items.min()) < 0 or int(items.max()) >= ni):
        raise InvalidAction(f"item ids must be in [0, {ni})")
    if items.size:
        if np.any(item_col(world, "platform")[items] != pc):
            raise InvalidAction(f"an item is not distributed on {platform}")
        if np.any(item_col(world, "removed")[items]):
            raise InvalidAction("a removed item cannot be shown or consumed")
    return item_col(world, "duration_s")[items].astype(np.float64)


def _decision_batch(ctx: Ctx, kind: str, v: np.ndarray, vals: dict, extra: dict, decision: dict | None) -> str:
    """Record audience decision rows: through `population.record_decisions` (decision provenance) when the world's
    population is `pop`; the crowd stand-in of the historical tests records plain batches."""
    if "pop" not in ctx.world.modules:
        return columns.batch(ctx, POP, kind, v, vals, mode="set", extra=extra)
    from .. import population
    if decision is None:
        decision = population.participant_decision(ctx) if ids.index(ctx.actor) is not None else ENGINE_DECISION
    return population.record_decisions(ctx, kind, v, vals, model_id=decision["model_id"],
                                       model_version=decision["model_version"],
                                       fidelity_id=decision["fidelity_id"], extra=extra)


def impressions(ctx: Ctx, platform: str, viewers: np.ndarray, items: np.ndarray, slot: np.ndarray, *,
                source: np.ndarray | None = None, gen: np.ndarray | None = None,
                decision: dict | None = None) -> str | None:
    """Record that each viewer was shown each item at a feed slot (one row per impression). `source` (int8 codes of
    `SOURCES`) and `gen` (share generation: 0 = not via a share) are optional provenance. Returns the blob REF."""
    v = _arr(viewers, "viewers", "int")
    n = v.size
    it = _arr(items, "items", "int", n)
    sl = _arr(slot, "slot", "int", n)
    if n == 0:
        return None
    src = np.zeros(n, np.int8) if source is None else _arr(source, "source", "int", n)
    g = np.zeros(n, np.uint8) if gen is None else _arr(gen, "gen", "int", n)
    if np.any(sl < 0) or np.any(sl > 32767):
        raise InvalidAction("slot must be in [0, 32767]")
    if np.any(src < 0) or np.any(src >= len(SOURCES)):
        raise InvalidAction("unknown impression source")
    if np.any(g < 0) or np.any(g > 255):
        raise InvalidAction("gen must be in [0, 255]")
    _check_items(ctx.world, platform, it)
    ref = _decision_batch(ctx, f"impress.{platform}", v,
                          {"aud_i_item": it, "aud_i_slot": sl, "aud_i_src": src, "aud_i_gen": g,
                           "aud_i_t": np.full(n, ctx.now, np.int64)}, {"platform": platform, "rows": int(n)}, decision)
    ctx.emit("aud.impressed", {"platform": platform, "ref": ref, "n": int(n)}, vis="internal")
    return ref


def consumption(ctx: Ctx, platform: str, viewers: np.ndarray, items: np.ndarray, watch_s: np.ndarray,
                completed: np.ndarray, shared: np.ndarray, followed: np.ndarray, liked: np.ndarray, *,
                decision: dict | None = None) -> str | None:
    """Record viewing decisions (one row per viewing): seconds watched and whether the viewer completed, shared,
    followed (subscribed to the creator) and liked. The ONLY way audience state changes. A row needs a prior
    exposure of that viewer to that item; whether it is a view is decided by the applier from watch_s and the item's
    duration (`view_threshold`). Returns the blob REF."""
    v = _arr(viewers, "viewers", "int")
    n = v.size
    it = _arr(items, "items", "int", n)
    w = _arr(watch_s, "watch_s", "float", n)
    c, sh, fo, li = (_arr(x, nm, "bool", n) for x, nm in ((completed, "completed"), (shared, "shared"),
                                                         (followed, "followed"), (liked, "liked")))
    if n == 0:
        return None
    dur = _check_items(ctx.world, platform, it)
    if np.any(w < 0):
        raise InvalidAction("watch_s must be >= 0")
    if np.any(w > dur * (1 + 1e-6) + 1e-3):
        raise InvalidAction("a consumption row is one viewing: watch_s cannot exceed the item's duration")
    if np.any(c & (w < COMPLETE_FRAC * dur - 1e-6)):
        raise InvalidAction(f"completed needs watch_s >= {COMPLETE_FRAC} x duration")
    if np.any((sh | fo | li) & (w <= 0)):
        raise InvalidAction("share, follow and like decisions follow consumption (watch_s > 0)")
    npop = n_people(ctx.world)
    if int(v.min()) < 0 or int(v.max()) >= npop:
        raise InvalidAction(f"viewer index out of range [0, {npop})")
    # no view without exposure: every (viewer, item) row must have been shown to that viewer before
    order = np.argsort(it, kind="stable")
    its = it[order]
    starts = np.flatnonzero(np.r_[True, its[1:] != its[:-1]])
    ends = np.r_[starts[1:], its.size]
    st = ctx.state
    for s0, e0 in zip(starts, ends):
        rows = order[s0:e0]
        if not np.all(idset.contains(st, idset.key("ex", int(its[s0])), v[rows])):
            raise InvalidAction(f"item {int(its[s0])} was not shown to every viewer of this batch (no view without "
                                f"exposure)")
    flags = (c * F_COMPLETED + sh * F_SHARED + fo * F_FOLLOWED + li * F_LIKED).astype(np.uint8)
    ref = _decision_batch(ctx, f"consume.{platform}", v,
                          {"aud_c_item": it, "aud_c_watch_s": w.astype(np.float32), "aud_c_flags": flags,
                           "aud_c_t": np.full(n, ctx.now, np.int64)}, {"platform": platform, "rows": int(n)}, decision)
    ctx.emit("aud.consumed", {"platform": platform, "ref": ref, "n": int(n)}, vis="internal")
    return ref


# ---------------------------------------------------------------------------------------------------- reading
def audience(world, item: int, *, principal: str | None = None) -> dict:
    """Audience of one item. Exact for the evaluator (principal None); for the item's creator the delayed, noisy
    published analytics; for anyone else the public (delayed, noisy) view count only."""
    item = int(item)
    rec = info(world, item)
    if rec is None:
        raise NotFound(f"no item {item}")
    if principal is None:
        out = {q: _exact(world, item, q) for q in QUANTITIES}
        out.update(item=item, likes=int(columns.col(world, ITEMS, "likes")[item]), exact=True, as_of=int(world.now))
        return out
    pub_t = int(columns.col(world, ITEMS, "pub_t")[item])
    if rec["creator"] in world.principals_of(principal) or principal == SYSTEM:
        out = {q: _pub(world, item, q) for q in QUANTITIES}
        out.update(item=item, exact=False, noisy=True, as_of=pub_t if pub_t >= 0 else None)
        return out
    return {"item": item, "views": _pub(world, item, "views"), "exact": False, "as_of": pub_t if pub_t >= 0 else None}


def _exact(world, item, q):
    v = columns.col(world, ITEMS, _QCOL[q])[item]
    return float(v) if q == "watch_s" else int(v)


def _pub(world, item, q):
    v = float(columns.col(world, ITEMS, f"pub_{_QCOL[q]}")[item])
    return round(v, 1) if q == "watch_s" else int(v)


def exposed(world, item: int) -> np.ndarray:
    return idset.members(world.state, idset.key("ex", int(item)))


def viewers_of(world, item: int) -> np.ndarray:
    return idset.members(world.state, idset.key("vw", int(item)))


def followers(world, creator: str | int) -> np.ndarray:
    code = creator if isinstance(creator, (int, np.integer)) else creator_code(world, creator)
    if code is None:
        return np.zeros(0, dtype=np.int64)
    return idset.members(world.state, idset.key("fo", int(code)))


def is_follower(world, codes: np.ndarray, viewers: np.ndarray) -> np.ndarray:
    """Membership of viewers[r] in the follower set of creator codes[r] (row-wise)."""
    codes = np.asarray(codes, np.int64)
    viewers = np.asarray(viewers, np.int64)
    out = np.zeros(codes.shape, dtype=bool)
    for c in np.unique(codes):
        if c < 0:
            continue
        m = codes == c
        out[m] = idset.contains(world.state, idset.key("fo", int(c)), viewers[m])
    return out


def batches(world, kind: str | None = None, platform: str | None = None, since_t: int | None = None) -> list[dict]:
    """Recorded decision batches still in the retention window (evaluator and platform side), oldest first."""
    out = []
    for rid, r in world.state.tables["aud_batches"].items():
        if kind is not None and r["kind"] != kind:
            continue
        if platform is not None and r["platform"] != platform:
            continue
        if since_t is not None and r["t"] < since_t:
            continue
        out.append(dict(r, seq=int(rid)))
    out.sort(key=lambda r: r["seq"])
    return out


# ---------------------------------------------------------------------------------------------------- events
_PLAT = {"type": "string", "pattern": r"[a-z][a-z0-9_]{0,31}"}
bus.event_type("aud.impressed", S.obj({"platform": _PLAT, "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern},
                                       "n": {"type": "integer", "minimum": 1}}, ["platform", "ref", "n"]), owner="aud",
               doc="impression rows recorded in the cols.batch blob `ref` (kind impress.<platform>)")
bus.event_type("aud.consumed", S.obj({"platform": _PLAT, "ref": {"type": "string", "pattern": artifacts.REF_RE.pattern},
                                      "n": {"type": "integer", "minimum": 1}}, ["platform", "ref", "n"]), owner="aud",
               doc="consumption rows recorded in the cols.batch blob `ref` (kind consume.<platform>)")


@register_module
class Audience(Module):
    """Content index and exact audience accounting for every platform of the world (see the module docstring)."""
    name = "aud"
    host = "aud"
    version = "1"
    depends = ("cols",)

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"aud: unknown config key {k!r}")
            self.cfg[k] = v
        plats = list(self.cfg["platforms"])
        if not plats or len(set(plats)) != len(plats):
            raise ProtocolError("aud: platforms must be a non-empty list of distinct names")
        import re
        for p in plats:
            if not isinstance(p, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", p):
                raise ProtocolError(f"aud: bad platform name {p!r}")
        self.platforms = plats
        if self.cfg["encoder"] not in ("auto", "space", "c2-random-indexing", "semantic"):
            raise ProtocolError("aud: encoder must be 'auto', 'space' (the shared semantic space, D-16), "
                                "'c2-random-indexing' or 'semantic' (historical paths of the crowd population)")

    def platform_code(self, platform: str) -> int:
        try:
            return self.platforms.index(platform)
        except ValueError:
            raise NotFound(f"no platform {platform!r}") from None

    def install(self, world):
        if POP not in columns.tables(world):
            raise ProtocolError("aud needs the population table 'pop' (list the population module before aud)")
        n = columns.rows(world, POP)
        if self.cfg["encoder"] == "auto":
            # the one population lives in the shared semantic space; the lexical path stays for the crowd only
            self.cfg["encoder"] = "space" if "pop" in world.modules else "c2-random-indexing"
        if self.cfg["encoder"] == "space":
            if "space" not in world.modules:
                raise ProtocolError("aud: the shared-space encoder needs the kernel module 'space'")
            self.cfg["dim"] = int(columns.col(world, POP, "interests").shape[1])
        columns.declare(world, POP, n, _POP_COLS)
        columns.declare(world, ITEMS, int(self.cfg["max_items"]), _item_cols(int(self.cfg["dim"])))
        columns.declare(world, CREATORS, int(self.cfg["max_creators"]), _CREATOR_COLS)
        st = world.state
        st.declare_table("aud_item_info", index=("creator", "platform"))
        st.declare_table("aud_creator_info", index=("principal",))
        st.declare_table("aud_batches", index=("kind",))
        bus.enforce(world, owner="aud")

    def genesis(self, ctx):
        ctx.schedule(int(self.cfg["analytics_interval_s"]), "aud.analytics", {}, prio=60,
                     period=int(self.cfg["analytics_interval_s"]), key="aud.analytics")

    # ------------------------------------------------------------------ representation
    def encode(self, world, ref: str, mime: str, *, ctx: Ctx | None = None, caption: str = "") -> tuple:
        """(representation, BM25 terms, words, legibility, how it was represented)."""
        dim = int(self.cfg["dim"])
        if self.cfg["encoder"] == "space":
            from ..population import semantics
            terms, words = {}, 0
            if mime.startswith("text/"):
                _, terms, words = textrep.encode_bytes(artifacts.get(world, ref), dim)
            try:
                # one recorded representation of the shared space (crucible.semantic.space): text, image, audio,
                # video and code from their bytes; text gains the population's reader model (legibility)
                vec, feats = semantics.item_rep(world, ref, ctx=ctx)
                legib, rep = float(feats.get("legibility", 1.0)), "bytes"
            except semantics.EncodingError:
                # undecodable media (the space's MediaError): the item is represented by the text a viewer sees
                # with it, labelled rep="caption"
                if not caption.strip():
                    raise InvalidAction(f"no representation for {mime} without a caption in this world") from None
                vec = semantics.text_vector(world, caption, ctx=ctx)
                legib, rep = 1.0, "caption"
                terms = dict(textrep.term_counts(caption))
            e = semantics.project(vec[None, :], dim)[0]
            return e, terms, words, legib, rep
        if self.cfg["encoder"] == "semantic":
            from .. import semantic                       # workstream D (raises if it is not merged)
            modality = mime.split("/", 1)[0]
            modality = {"text": "text", "image": "image", "audio": "audio", "video": "video"}.get(modality, "code")
            vec = np.asarray(semantic.encode(world, ref, modality), dtype=np.float64)
            proj = world.rng_for("aud.semantic_projection", vec.size, dim).standard_normal((vec.size, dim))
            e = vec @ proj
            e = e / max(float(np.linalg.norm(e)), 1e-12)
            terms, words = {}, 0
            if mime.startswith("text/"):
                _, terms, words = textrep.encode_bytes(artifacts.get(world, ref), dim)
            return e, terms, words, 1.0, "bytes"
        if not mime.startswith("text/"):
            raise InvalidAction(f"no encoder for {mime} in this world (configure aud.encoder='semantic' once "
                                f"crucible.semantic is available)")
        e, terms, words = textrep.encode_bytes(artifacts.get(world, ref), dim)
        return e, terms, words, 1.0, "bytes"

    # ------------------------------------------------------------------ appliers: catalogue
    @applier("aud.creator")
    def ap_creator(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("aud_creator_info", str(p["code"]), {"code": p["code"], "principal": p["principal"],
                                                       "pop_index": p["pop_index"], "t": ev.t})
        st.array_rows_mut(f"{CREATORS}.pop_index", np.array([p["code"]]))[p["code"]] = \
            -1 if p["pop_index"] is None else p["pop_index"]
        st.meta_set("aud.n_creators", p["code"] + 1)

    @applier("aud.item")
    def ap_item(self, ev):
        p = ev.payload
        st = self.world.state
        i = p["item"]
        if i != n_items(self.world):
            raise InvariantViolation(f"aud.item {i} out of order")
        own = {"creator_code": p["creator_code"], "duration_s": p["duration_s"], "content_id": p["content_id"],
               "terms": p["terms"], "words": p["words"], "removed": False}
        if p["content_id"] is None or not st.has("content", p["content_id"]):
            # no catalogue record (the crowd stand-in's scale path): this is the only record of the item's metadata
            own.update({"ref": p["ref"], "kind": p["kind"], "platform": p["platform"], "creator": p["creator"],
                        "title": p["title"], "tags": p["tags"], "published_at": ev.t})
        st.insert("aud_item_info", str(i), own)
        r = np.array([i])
        for c, v in (("creator", p["creator_code"]), ("platform", self.platform_code(p["platform"])),
                     ("kind", KINDS.index(p["kind"])), ("duration_s", p["duration_s"]), ("published_at", ev.t)):
            st.array_rows_mut(f"{ITEMS}.{c}", r)[i] = v
        st.array_rows_mut(f"{ITEMS}.emb", r)[i] = np.asarray(p["emb"], dtype=np.float16)
        st.array_rows_mut(f"{ITEMS}.legib", r)[i] = float(p.get("legib", 1.0))
        st.array_rows_mut(f"{CREATORS}.items", np.array([p["creator_code"]]))[p["creator_code"]] += 1
        st.meta_set("aud.n_items", i + 1)

    @applier("aud.item_updated")
    def ap_item_updated(self, ev):
        p = ev.payload
        st = self.world.state
        i = int(p["item"])
        if "title" in st.tables["aud_item_info"][str(i)]:          # an item without a catalogue record
            st.patch("aud_item_info", str(i), ref=p["ref"], title=p["title"], tags=p["tags"],
                     duration_s=p["duration_s"], terms=p["terms"], words=p["words"])
        else:                                                        # the catalogue holds ref, title and tags
            st.patch("aud_item_info", str(i), duration_s=p["duration_s"], terms=p["terms"], words=p["words"])
        r = np.array([i])
        st.array_rows_mut(f"{ITEMS}.emb", r)[i] = np.asarray(p["emb"], dtype=np.float16)
        st.array_rows_mut(f"{ITEMS}.legib", r)[i] = float(p.get("legib", 1.0))
        st.array_rows_mut(f"{ITEMS}.duration_s", r)[i] = float(p["duration_s"])
        st.array_rows_mut(f"{ITEMS}.version", r)[i] += 1

    @applier("aud.item_removed")
    def ap_removed(self, ev):
        i = ev.payload["item"]
        self.world.state.patch("aud_item_info", str(i), removed=True)
        self.world.state.array_rows_mut(f"{ITEMS}.removed", np.array([i]))[i] = True

    # ------------------------------------------------------------------ appliers: decisions
    def _record_batch(self, ev, kind):
        self.world.state.insert("aud_batches", str(ev.seq), {"kind": kind, "platform": ev.payload["platform"],
                                                             "ref": ev.payload["ref"], "n": ev.payload["n"], "t": ev.t})

    def _blob(self, ev, prefix):
        p = ev.payload
        try:
            a = artifacts.get_array_blob(self.world, p["ref"])
        except NotFound as e:
            raise InvariantViolation(f"{ev.type} seq {ev.seq}: {e}") from None
        v = a["idx"].astype(np.int64)
        if v.size != p["n"]:
            raise InvariantViolation(f"{ev.type} seq {ev.seq}: blob has {v.size} rows, event says {p['n']}")
        return v, a

    @applier("aud.impressed")
    def ap_impressed(self, ev):
        w = self.world
        st = w.state
        v, a = self._blob(ev, "aud_i_")
        it = a["v.aud_i_item"].astype(np.int64)
        npop = n_people(w)
        order = np.argsort(it, kind="stable")
        its, vs = it[order], v[order]
        starts = np.flatnonzero(np.r_[True, its[1:] != its[:-1]])
        ends = np.r_[starts[1:], its.size]
        uitems = its[starts]
        new_exposed = np.zeros(uitems.size, np.int64)
        for j, (s0, e0) in enumerate(zip(starts, ends)):
            g = vs[s0:e0]                                   # sorted by viewer within the item (blob order)
            u = g[np.r_[True, g[1:] != g[:-1]]]
            new_exposed[j] = int(idset.add(st, idset.key("ex", int(its[s0])), u, npop).sum())
        imp = st.array_rows_mut(f"{ITEMS}.imp", uitems)
        imp[uitems] += (ends - starts)
        ex = st.array_rows_mut(f"{ITEMS}.exposed", uitems)
        ex[uitems] += new_exposed
        cre = st.arrays[f"{ITEMS}.creator"][uitems].astype(np.int64)
        cu = np.unique(cre)
        ci = st.array_rows_mut(f"{CREATORS}.imp", cu)
        np.add.at(ci, cre, (ends - starts))
        self._record_batch(ev, "imp")

    @applier("aud.consumed")
    def ap_consumed(self, ev):
        w = self.world
        st = w.state
        v, a = self._blob(ev, "aud_c_")
        it = a["v.aud_c_item"].astype(np.int64)
        ws = a["v.aud_c_watch_s"].astype(np.float64)
        fl = a["v.aud_c_flags"]
        npop = n_people(w)
        dur = st.arrays[f"{ITEMS}.duration_s"][it].astype(np.float64)
        cre = st.arrays[f"{ITEMS}.creator"][it].astype(np.int64)
        self_row = st.arrays[f"{CREATORS}.pop_index"][cre] == v
        is_view = (ws >= view_threshold(dur)) & ~self_row
        comp = ((fl & F_COMPLETED) > 0) & is_view
        shared = ((fl & F_SHARED) > 0) & ~self_row
        foll = ((fl & F_FOLLOWED) > 0) & ~self_row
        liked = ((fl & F_LIKED) > 0) & is_view
        n = v.size
        # first qualifying view of (item, viewer): row order is the blob order (viewer, then decision order)
        first_view = np.zeros(n, dtype=bool)
        order = np.argsort(it, kind="stable")
        its = it[order]
        starts = np.flatnonzero(np.r_[True, its[1:] != its[:-1]])
        ends = np.r_[starts[1:], its.size]
        uitems = its[starts]
        for s0, e0 in zip(starts, ends):
            rows = order[s0:e0]
            rv = rows[is_view[rows]]
            if rv.size == 0:
                continue
            g = v[rv]
            first_in = np.r_[True, g[1:] != g[:-1]]
            u = g[first_in]
            new = idset.add(st, idset.key("vw", int(its[s0])), u, npop)
            first_view[rv[first_in][new]] = True
        # returning viewers and follows, per creator
        returning = np.zeros(n, dtype=bool)
        new_follow = np.zeros(n, dtype=bool)
        corder = np.argsort(cre, kind="stable")
        cs = cre[corder]
        cst = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        cen = np.r_[cst[1:], cs.size]
        ucre = cs[cst]
        new_cviewers = np.zeros(ucre.size, np.int64)
        new_followers = np.zeros(ucre.size, np.int64)
        for j, (s0, e0) in enumerate(zip(cst, cen)):
            c = int(cs[s0])
            rows = corder[s0:e0]
            rv = rows[is_view[rows]]
            if rv.size:
                g = v[rv]
                first_in = np.r_[True, g[1:] != g[:-1]]
                prior = idset.contains(st, idset.key("cv", c), g) | ~first_in
                returning[rv[prior & first_view[rv]]] = True
                new_cviewers[j] = int(idset.add(st, idset.key("cv", c), g[first_in], npop).sum())
            rf = rows[foll[rows]]
            if rf.size:
                g = v[rf]
                first_in = np.r_[True, g[1:] != g[:-1]]
                newm = idset.add(st, idset.key("fo", c), g[first_in], npop)
                new_follow[rf[first_in][newm]] = True
                new_followers[j] = int(newm.sum())

        def add_items(col, vals, dtype=np.int64):
            arr = st.array_rows_mut(f"{ITEMS}.{col}", uitems)
            tot = np.zeros(uitems.size, dtype=dtype)
            np.add.at(tot, np.searchsorted(uitems, it), vals)
            arr[uitems] += tot.astype(arr.dtype)
            return tot
        nv = add_items("views", is_view.astype(np.int64))
        nu = add_items("viewers", first_view.astype(np.int64))
        arr = st.array_rows_mut(f"{ITEMS}.repeats", uitems)
        arr[uitems] += nv - nu
        add_items("watch_s", np.where(self_row, 0.0, ws), np.float64)
        add_items("completions", comp.astype(np.int64))
        add_items("shares", shared.astype(np.int64))
        add_items("follows", new_follow.astype(np.int64))
        add_items("returning", returning.astype(np.int64))
        add_items("likes", (liked & first_view).astype(np.int64))
        add_items("self_rows", self_row.astype(np.int64))

        def add_creators(col, vals, dtype=np.int64):
            arr = st.array_rows_mut(f"{CREATORS}.{col}", ucre)
            tot = np.zeros(ucre.size, dtype=dtype)
            np.add.at(tot, np.searchsorted(ucre, cre), vals)
            arr[ucre] += tot.astype(arr.dtype)
        add_creators("views", is_view.astype(np.int64))
        add_creators("completions", comp.astype(np.int64))
        add_creators("shares", shared.astype(np.int64))
        add_creators("follows", new_follow.astype(np.int64))
        add_creators("watch_s", np.where(self_row, 0.0, ws), np.float64)
        arr = st.array_rows_mut(f"{CREATORS}.viewers", ucre)
        arr[ucre] += new_cviewers
        arr = st.array_rows_mut(f"{CREATORS}.followers", ucre)
        arr[ucre] += new_followers
        self._record_batch(ev, "consume")

    # ------------------------------------------------------------------ analytics publication (delayed, noisy)
    @process("analytics")
    def pr_analytics(self, ctx):
        ni = n_items(ctx.world)
        if ni:
            idx = np.arange(ni, dtype=np.int64)
            vals = {}
            sigma = float(self.cfg["analytics_noise"])
            rng = ctx.rng("aud.analytics", ctx.now)
            for q in QUANTITIES:
                c = _QCOL[q]
                staged = columns.col(ctx.world, ITEMS, f"stg_{c}")[:ni].astype(np.float64)
                noise = np.exp(sigma * rng.standard_normal(ni) - 0.5 * sigma * sigma) if sigma > 0 else 1.0
                pub = staged * noise
                vals[f"pub_{c}"] = pub if q == "watch_s" else np.round(pub)
                vals[f"stg_{c}"] = columns.col(ctx.world, ITEMS, c)[:ni].astype(np.float64)
            vals["pub_t"] = columns.col(ctx.world, ITEMS, "stg_t")[:ni].copy()
            vals["stg_t"] = np.full(ni, ctx.now, np.int64)
            views = columns.col(ctx.world, ITEMS, "views")[:ni].astype(np.int64)
            vals["vel_views"] = (views - columns.col(ctx.world, ITEMS, "prev_views")[:ni]).astype(np.float32)
            vals["prev_views"] = views
            columns.batch(ctx, ITEMS, "analytics.publish", idx, vals, mode="set", extra={"t": int(ctx.now)})
        cutoff = ctx.now - int(self.cfg["batch_retention_days"]) * 86400
        old = [rid for rid, r in ctx.state.tables["aud_batches"].items() if r["t"] < cutoff]
        if old:
            ctx.emit("aud.batches_pruned", {"before_t": int(cutoff), "n": len(old)}, vis="internal")

    @applier("aud.batches_pruned")
    def ap_pruned(self, ev):
        st = self.world.state
        for rid in [rid for rid, r in st.tables["aud_batches"].items() if r["t"] < ev.payload["before_t"]]:
            st.delete("aud_batches", rid)

    # ------------------------------------------------------------------ tools
    def _item_rec(self, ctx, item):
        rec = info(ctx.world, item)
        if rec is None or int(item) >= n_items(ctx.world):
            raise NotFound(f"no item {item}")
        return rec

    def public_item(self, world, item: int) -> dict:
        rec = info(world, item)
        return {"item": int(item), "title": rec["title"], "creator": rec["creator"], "platform": rec["platform"],
                "kind": rec["kind"], "ref": rec["ref"], "duration_s": rec["duration_s"], "tags": list(rec["tags"]),
                "published_at": rec["published_at"], "removed": rec["removed"], "views": _pub(world, int(item), "views")}

    @endpoint("item", S.obj({"item": S.s_int("item id", minimum=0, maximum=2**31 - 1)}, ["item"]), read_only=True,
              cost_s=0)
    def ep_item(self, ctx, item):
        """Public facts of one content item: title, creator, platform, kind, artifact reference, duration and its
        public view count (published with the platform's analytics delay)."""
        self._item_rec(ctx, item)
        return self.public_item(ctx.world, item)

    @endpoint("analytics", S.obj({"item": S.s_int("item id", minimum=0, maximum=2**31 - 1)}, ["item"]),
              read_only=True, cost_s=1)
    def ep_analytics(self, ctx, item):
        """Creator analytics of one of your items: impressions, unique exposed, views, unique viewers, watch time,
        completions, repeat views, shares, subscribers gained and returning viewers, as published by the platform
        (delayed and with measurement noise; `as_of` is the time the figures refer to)."""
        rec = self._item_rec(ctx, item)
        if rec["creator"] not in ctx.world.principals_of(ctx.actor):
            raise NotFound(f"no item {item} of yours")          # others' analytics look like a missing item
        return audience(ctx.world, int(item), principal=ctx.actor)

    @endpoint("mine", S.obj({"limit": S.s_int(minimum=1, maximum=200, default=50),
                             "offset": S.s_int(minimum=0, default=0)}), read_only=True, cost_s=1)
    def ep_mine(self, ctx, limit=50, offset=0):
        """Your distributed items with their published analytics and your follower count (delayed like the
        item analytics)."""
        mine = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            mine += items_of(ctx.world, p)
        mine.sort()
        page = mine[offset:offset + limit]
        return {"total": len(mine), "items": [dict(self.public_item(ctx.world, i),
                                                   analytics=audience(ctx.world, i, principal=ctx.actor))
                                              for i in page]}

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = []
        ni = n_items(world)
        copies = [k for k, r in world.state.tables.get("aud_item_info", {}).items()
                  if r.get("content_id") is not None and world.state.has("content", r["content_id"])
                  and any(f in r for f in CATALOGUE_FIELDS)]
        if copies:
            errs.append(f"aud: {len(copies)} catalogue items keep a copy of the catalogue's metadata (e.g. {copies[0]})")
        if ni == 0:
            return errs
        st = world.state
        c = {k: columns.col(world, ITEMS, k)[:ni] for k in ("imp", "exposed", "views", "viewers", "completions",
                                                             "repeats", "returning", "follows", "likes")}
        if np.any(c["exposed"] > c["imp"]):
            errs.append("unique exposed above impressions")
        if np.any(c["viewers"] > c["views"]):
            errs.append("unique viewers above views")
        if np.any(c["viewers"] > c["exposed"]):
            errs.append("unique viewers above unique exposed")
        if np.any(c["completions"] > c["views"]):
            errs.append("completions above views")
        if np.any(c["repeats"] != c["views"] - c["viewers"]):
            errs.append("repeat views != views - unique viewers")
        if np.any(c["returning"] > c["viewers"]) or np.any(c["likes"] > c["viewers"]):
            errs.append("returning viewers or likes above unique viewers")
        # the exact sets agree with the counters (sizes); full membership checks are done by recount()
        touched = np.flatnonzero(c["imp"] > 0)
        for i in touched[:2000]:
            if idset.size(st, idset.key("ex", int(i))) != int(c["exposed"][i]):
                errs.append(f"item {i}: exposed set size != counter")
                break
            if idset.size(st, idset.key("vw", int(i))) != int(c["viewers"][i]):
                errs.append(f"item {i}: viewer set size != counter")
                break
        return errs


# ---------------------------------------------------------------------------------------------------- recount
def decision_events(world, events=None) -> list:
    """The aud.impressed / aud.consumed events of a world's log (in-memory window, or a given event list)."""
    evs = world.log if events is None else events
    out = []
    for e in evs:
        d = e if isinstance(e, dict) else e.to_dict()
        if d["type"] in ("aud.impressed", "aud.consumed"):
            out.append(d)
    return out


def recount(world, events=None, *, check_members: bool = True) -> dict:
    """Re-derive every audience quantity of every item from the recorded batch blobs alone (independently of the
    appliers) and compare with the world state. Evaluator side. Returns counts checked and mismatches."""
    st = world.state
    ni = n_items(world)
    npop = n_people(world)
    evs = decision_events(world, events)
    imp_keys, imp_ord, con = [], [], []
    for d in evs:
        a = artifacts.get_array_blob(world, d["payload"]["ref"])
        v = a["idx"].astype(np.int64)
        if d["type"] == "aud.impressed":
            it = a["v.aud_i_item"].astype(np.int64)
            imp_keys.append(it * npop + v)
            imp_ord.append(np.full(v.size, d["seq"], np.int64))
        else:
            con.append((d["seq"], v, a["v.aud_c_item"].astype(np.int64), a["v.aud_c_watch_s"].astype(np.float64),
                        a["v.aud_c_flags"]))
    dur = columns.col(world, ITEMS, "duration_s")[:ni].astype(np.float64)
    cre = columns.col(world, ITEMS, "creator")[:ni].astype(np.int64)
    cpop = columns.col(world, CREATORS, "pop_index").astype(np.int64)
    ik = np.concatenate(imp_keys) if imp_keys else np.zeros(0, np.int64)
    io = np.concatenate(imp_ord) if imp_ord else np.zeros(0, np.int64)
    rc = {q: np.zeros(ni, np.float64 if q == "watch_s" else np.int64) for q in QUANTITIES}
    np.add.at(rc["impressions"], ik // npop, 1)
    uk, first_idx = np.unique(ik, return_index=True)
    np.add.at(rc["unique_exposed"], uk // npop, 1)
    first_imp_seq = io[first_idx]
    forced = 0
    if con:
        seqs = np.concatenate([np.full(c[1].size, c[0], np.int64) for c in con])
        rowno = np.concatenate([np.arange(c[1].size) for c in con])
        v = np.concatenate([c[1] for c in con])
        it = np.concatenate([c[2] for c in con])
        ws = np.concatenate([c[3] for c in con])
        fl = np.concatenate([c[4] for c in con])
        self_row = cpop[cre[it]] == v
        view = (ws >= np.minimum(VIEW_MIN_S, VIEW_MIN_FRAC * dur[it])) & ~self_row
        key = it * npop + v
        # every consumption row needs an earlier impression of the same (item, viewer)
        if uk.size:
            pos_c = np.minimum(np.searchsorted(uk, key), uk.size - 1)
            forced = int(np.sum((uk[pos_c] != key) | (first_imp_seq[pos_c] >= seqs)))
        else:
            forced = int(key.size)
        np.add.at(rc["watch_s"], it, np.where(self_row, 0.0, ws))
        np.add.at(rc["views"], it[view], 1)
        np.add.at(rc["completions"], it[view & ((fl & F_COMPLETED) > 0)], 1)
        np.add.at(rc["shares"], it[~self_row & ((fl & F_SHARED) > 0)], 1)
        gorder = np.lexsort((rowno, seqs))            # global decision order: event seq, then blob row
        vk = key[gorder][view[gorder]]
        vrow = np.flatnonzero(view[gorder])
        uvk, fvi = np.unique(vk, return_index=True)
        np.add.at(rc["unique_viewers"], uvk // npop, 1)
        rc["repeat_views"] = rc["views"] - rc["unique_viewers"]
        # returning: first view of (item, viewer) after an earlier qualifying view of (creator, viewer)
        cv = cre[it[gorder]][view[gorder]] * npop + v[gorder][view[gorder]]
        ucv, fci = np.unique(cv, return_index=True)
        first_cv_pos = fci[np.searchsorted(ucv, cv[fvi])]
        ret = first_cv_pos < fvi
        np.add.at(rc["returning_viewers"], uvk[ret] // npop, 1)
        # subscribers: first follow decision of (creator, viewer)
        fol = (~self_row & ((fl & F_FOLLOWED) > 0))[gorder]
        fk = cre[it[gorder]][fol] * npop + v[gorder][fol]
        _, ffi = np.unique(fk, return_index=True)
        np.add.at(rc["subscribers"], it[gorder][fol][ffi], 1)
        viewers_by_item = uvk
    else:
        viewers_by_item = np.zeros(0, np.int64)
    mism = {}
    for q in QUANTITIES:
        s = columns.col(world, ITEMS, _QCOL[q])[:ni]
        if q == "watch_s":
            bad = int(np.sum(~np.isclose(s.astype(np.float64), rc[q], rtol=1e-9, atol=1e-3)))
        else:
            bad = int(np.sum(s.astype(np.int64) != rc[q]))
        mism[q] = bad
    member_mism = {"exposed": 0, "viewers": 0}
    if check_members:
        for kind, keys, nm in (("ex", uk, "exposed"), ("vw", viewers_by_item, "viewers")):
            items_k = keys // npop
            st_ = np.flatnonzero(np.r_[True, items_k[1:] != items_k[:-1]]) if keys.size else np.zeros(0, np.int64)
            en_ = np.r_[st_[1:], keys.size] if keys.size else np.zeros(0, np.int64)
            seen = set()
            for s0, e0 in zip(st_, en_):
                i = int(items_k[s0])
                seen.add(i)
                if not np.array_equal(idset.members(st, idset.key(kind, i)), keys[s0:e0] % npop):
                    member_mism[nm] += 1
            for i in range(ni):
                if i not in seen and idset.size(st, idset.key(kind, i)):
                    member_mism[nm] += 1
    prov = None
    if "pop" in world.modules:
        from .. import population
        prov = population.check_provenance(world, events)
    return {"items": ni, "impression_rows": int(ik.size), "exposures_checked": int(uk.size),
            "provenance": prov,
            "views_without_decision_provenance": None if prov is None else int(prov["forced_rows"]),
            "distinct_viewers_checked": int(viewers_by_item.size),
            "consumption_rows": int(sum(c[1].size for c in con)), "views": int(rc["views"].sum()),
            "mismatches": mism, "member_mismatches": member_mism, "views_without_prior_exposure": forced,
            "exposure_recount_mismatches": mism["impressions"] + mism["unique_exposed"] + member_mism["exposed"],
            "distinct_viewer_recount_mismatches": mism["unique_viewers"] + mism["views"] + member_mism["viewers"],
            "total_mismatches": sum(mism.values()) + sum(member_mism.values()), "recount": rc}
