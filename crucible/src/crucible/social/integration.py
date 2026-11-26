"""Connections from the C1 social modules to the other workstreams' modules, used only when those are in the world.

* Workstream C2 (audience accounting `aud`, distribution `feed`): every public catalogue item on a platform the
  audience module knows is indexed for distribution and exact audience accounting with
  `crucible.social.audience.index_item(...)` in the publishing transaction; its audience item id is kept in the
  catalogue record (`aud_item`). A removal removes it from distribution. Impressions and consumption of tool-using
  population individuals on the C1 sites are recorded through `audience.impressions` / `audience.consumption` - the
  only way audience state changes.
* Workstream B (population `pop`): the social activities of `crucible.social.behaviour` are registered with
  `crucible.population.core.register_activity` (see that module).

Nothing here runs when the other modules are absent; nothing here invents a number they would have produced.
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import Ctx

#: seconds a viewer needs for one look at an image (there is no duration in the bytes); the audience module needs
#: a duration for the view threshold of every item it distributes. Plausible, unvalidated (FIDELITY_REGISTER).
IMAGE_VIEW_S = 10.0
WORDS_PER_S = 4.0


def aud_module(world):
    return world.modules.get("aud")


def audience_api():
    from . import audience        # workstream C2
    return audience


def index_published(ctx: Ctx, rec: dict, text_words: int | None) -> int | None:
    """Index a public catalogue item for distribution (C2) and return its audience item id, or None when the world
    has no audience module, the platform is not one it distributes, or the item is not public."""
    m = aud_module(ctx.world)
    if m is None or rec["access"] != "public" or rec["platform"] not in getattr(m, "platforms", ()):
        return None
    AUD = audience_api()
    dur = rec["duration_s"]
    if dur is None and not str(rec["mime"]).startswith("text/"):
        if rec["format"] == "image":
            dur = IMAGE_VIEW_S
        elif text_words:
            dur = max(5.0, text_words / WORDS_PER_S)
        else:
            return None
    return int(AUD.index_item(ctx, rec["ref"], rec["kind"], rec["platform"], rec["creator"], rec["title"],
                              tags=rec["tags"], duration_s=dur, content_id=rec["id"]))


def unindex(ctx: Ctx, aud_item: int | None) -> None:
    if aud_item is None or aud_module(ctx.world) is None:
        return
    audience_api().remove_item(ctx, int(aud_item))


def record_impressions(ctx: Ctx, platform: str, viewer: int, aud_items: list[int]) -> int:
    """Items shown on a C1 site to population individual `viewer` (tool use) as audience impressions."""
    if aud_module(ctx.world) is None or not aud_items:
        return 0
    m = aud_module(ctx.world)
    if platform not in getattr(m, "platforms", ()):
        return 0
    items = np.asarray(aud_items, dtype=np.int64)
    audience_api().impressions(ctx, platform, np.full(items.size, int(viewer), np.int64), items,
                               np.arange(items.size, dtype=np.int64))
    return int(items.size)


def deliver_impressions(ctx: Ctx, platform: str, viewers: np.ndarray, aud_item: int | None) -> int:
    """One item delivered to many population individuals (a newsletter issue in their inboxes) as impressions."""
    m = aud_module(ctx.world)
    viewers = np.asarray(viewers, dtype=np.int64)
    if m is None or aud_item is None or not viewers.size or platform not in getattr(m, "platforms", ()):
        return 0
    audience_api().impressions(ctx, platform, viewers, np.full(viewers.size, int(aud_item), np.int64),
                               np.zeros(viewers.size, np.int64))
    return int(viewers.size)


def record_consumption(ctx: Ctx, platform: str, viewer: int, aud_item: int, watch_s: float, completed: bool,
                       shared: bool, followed: bool, liked: bool) -> None:
    one = lambda v, dt: np.array([v], dtype=dt)  # noqa: E731
    audience_api().consumption(ctx, platform, one(int(viewer), np.int64), one(int(aud_item), np.int64),
                               one(float(watch_s), np.float32), one(bool(completed), bool), one(bool(shared), bool),
                               one(bool(followed), bool), one(bool(liked), bool))


def audience_view(world, aud_item: int | None, principal: str) -> dict | None:
    if aud_item is None or aud_module(world) is None:
        return None
    return audience_api().audience(world, int(aud_item), principal=principal)


def reindex(ctx: Ctx, aud_item: int, ref: str, *, title=None, tags=None, duration_s=None) -> None:
    """content.update of a distributed item: recompute its distributed representation from the new bytes."""
    if aud_module(ctx.world) is None:
        return
    audience_api().reindex_item(ctx, int(aud_item), ref, title=title, tags=tags, duration_s=duration_s)


def record_open(ctx: Ctx, platform: str, viewer: int, aud_item: int) -> None:
    """A population member opened an item with content.open: an impression of source "direct"."""
    m = aud_module(ctx.world)
    if m is None or platform not in getattr(m, "platforms", ()):
        return
    from ..substrate import columns
    if int(viewer) >= columns.rows(ctx.world, "pop"):
        return
    AUD = audience_api()
    AUD.impressions(ctx, platform, np.array([int(viewer)]), np.array([int(aud_item)]), np.array([0]),
                    source=np.array([AUD.SOURCE_CODE["direct"]]))


def record_read(ctx: Ctx, platform: str, viewer: int, aud_item: int, seconds: float, like: bool, share: bool,
                follow: bool) -> dict:
    """Reading time a population member spends on an item it was shown (content.read): attention is spent, the
    consumption row is the member's own decision (participant provenance)."""
    from ..core.errors import InvalidAction as _IA
    from ..substrate import attention, columns
    from . import idset
    AUD = audience_api()
    if aud_module(ctx.world) is None or platform not in getattr(aud_module(ctx.world), "platforms", ()):
        raise _IA("this item is not distributed here")
    if not idset.contains(ctx.state, idset.key("ex", int(aud_item)), np.array([int(viewer)]))[0]:
        raise _IA("open the item first (content.open)")
    dur = float(AUD.item_col(ctx.world, "duration_s")[aud_item])
    left = float(columns.col(ctx.world, "pop", "attention_left_min")[viewer]) * 60.0
    got = float(np.float32(min(max(seconds, 0.0), dur, left)))
    if got <= 0:
        return {"read_s": 0.0, "counted_as_view": False}
    attention.spend(ctx, np.array([int(viewer)]), np.array([got / 60.0]), activity=f"read.{platform}")
    comp = got >= AUD.COMPLETE_FRAC * dur
    AUD.consumption(ctx, platform, np.array([int(viewer)]), np.array([int(aud_item)]), np.array([got]),
                    np.array([comp]), np.array([share]), np.array([follow]), np.array([like]))
    return {"read_s": round(got, 3), "counted_as_view": bool(got >= float(AUD.view_threshold(dur))),
            "completed": bool(comp)}
