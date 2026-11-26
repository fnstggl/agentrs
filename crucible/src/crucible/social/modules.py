"""Kernel modules of the social internet; `crucible.load_all_modules()` imports this file.

Workstream C1: social_ids, social_rows, content (catalogue), studio (creator tools), social (follow graph), the
content sites (shorts, tube, posts, pics, blog, pods), community, msg, creators, letters, launch, dev, kb.
Workstream C2: `crowd` (columnar population of the scale path), `aud` (content index and exact audience
accounting), `feed` (recommendation and the consumption engine).
"""
from . import nodes, edges, content, studio, graph, sites  # noqa: F401  (C1, registration on import)
from . import communities, messaging, creators, qa, launch  # noqa: F401  (C1)
from . import scale  # noqa: F401  (C2: module crowd)
from . import audience  # noqa: F401  (C2: module aud)
from .recsys import service  # noqa: F401  (C2: module feed)
from . import popsocial  # noqa: F401  (the population creates content: module pop_social)
from . import probes  # noqa: F401  (Gate E.3 probes on the feeds: social_feed, social_feed_image)

#: every C1 module in dependency order (the social internet of one world)
C1_MODULES = ["social_ids", "social_rows", "content", "studio", "social", "shorts", "tube", "posts", "pics", "blog",
              "pods", "community", "msg", "creators", "letters", "launch", "dev", "kb"]
#: what they need from the 4.0 services and the substrate
BASE_MODULES = ["identity", "bank", "admin", "artifacts", "cols", "search", "forum", "mail"]


def social_internet_modules(extra: list[str] | None = None) -> list[dict]:
    """A world-spec module list with the whole social internet (and `extra` modules) on the 4.0 base services."""
    names = BASE_MODULES + C1_MODULES + [m for m in (extra or []) if m not in BASE_MODULES + C1_MODULES]
    return [{"name": n, "config": {}} for n in names]
