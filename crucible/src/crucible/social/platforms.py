"""The platforms of the social internet (MASTER_SPEC Part 4): one population, one content catalogue, one identity
space, many sites.

`PLATFORMS` is the registry: platform name -> host (the kernel module that serves it), what it is, which content
kinds it carries and how participants discover things on it. A platform's publications are rows of the shared
content catalogue (`crucible.social.content`); its audience state is the consumption engine's
(`crucible.social.consumption`, workstream C2); its social state (follows, communities, messages, subscriptions)
belongs to the social modules. Nothing here holds a quality, relevance or virality figure for any item.

Platform kinds (SP-4): short-form video, long-form video, text posts, images, articles/blogs, discussion forums,
social network, communities, messaging, search, creator subscriptions, newsletters, product discovery, developer
communities, knowledge sharing, plus audio (podcasts) so that every catalogue kind has a home.
"""
from __future__ import annotations

KINDS = ("video_short", "video_long", "post", "image", "article", "newsletter", "audio", "thread", "app", "package")

#: content kind -> artifact formats (crucible.social.formats) it may carry
KIND_FORMATS: dict[str, tuple[str, ...]] = {
    "video_short": ("video",),
    "video_long": ("video",),
    "post": ("text", "html", "image"),
    "image": ("image",),
    "article": ("html", "text", "document", "course"),
    "newsletter": ("html", "text"),
    "audio": ("audio",),
    "thread": ("text", "html"),
    "app": ("app", "page", "course"),
    "package": ("package",),
}
SHORT_MAX_S = 180.0          # a short video lasts at most three minutes
LONG_MIN_S = 60.0            # a long-form video lasts at least one minute

#: name -> {host, label, kinds, doc, discovery, publish}; `publish` False: the platform carries no public
#: publications of its own (messaging is private, search indexes the others).
PLATFORMS: dict[str, dict] = {
    "shorts": {"host": "shorts", "label": "Short-form video", "kinds": ("video_short",),
               "discovery": ("recommended", "following", "trending", "search", "share"),
               "doc": "vertical short videos (at most 3 minutes), a recommended feed and a following feed"},
    "tube": {"host": "tube", "label": "Long-form video", "kinds": ("video_long",),
             "discovery": ("recommended", "subscriptions", "search", "channel", "share"),
             "doc": "long videos on creator channels; subscriptions, search and recommendations"},
    "posts": {"host": "posts", "label": "Text posts", "kinds": ("post", "thread"),
              "discovery": ("following", "recommended", "reposts", "search", "share"),
              "doc": "short public text posts and threads, reposts and replies"},
    "pics": {"host": "pics", "label": "Images", "kinds": ("image", "post"),
             "discovery": ("following", "explore", "tags", "search", "share"),
             "doc": "photo and image sharing with tags and an explore page"},
    "blog": {"host": "blog", "label": "Articles and blogs", "kinds": ("article",),
             "discovery": ("following", "tags", "search", "share", "newsletter"),
             "doc": "long-form articles on personal blogs"},
    "pods": {"host": "pods", "label": "Audio and podcasts", "kinds": ("audio",),
             "discovery": ("subscriptions", "recommended", "search", "share"),
             "doc": "audio episodes and podcasts"},
    "forum": {"host": "forum", "label": "Discussion forums", "kinds": ("thread",),
              "discovery": ("boards", "following", "search"),
              "doc": "the 4.0 forum (boards, posts, replies, votes); its posts enter the catalogue as threads",
              "bridged": True},
    "social": {"host": "social", "label": "Social network", "kinds": ("post", "image", "video_short"),
               "discovery": ("following", "friends", "shares", "suggestions"),
               "doc": "profiles, the follow graph, timelines of followed people, shares"},
    "community": {"host": "community", "label": "Communities", "kinds": ("post", "image", "article", "thread",
                                                                          "video_short", "audio"),
                  "discovery": ("membership", "directory", "search", "invites"),
                  "doc": "participant-founded communities with membership, governance, moderation and activities"},
    "msg": {"host": "msg", "label": "Messaging", "kinds": (), "publish": False,
            "discovery": ("share",),
            "doc": "direct and group conversations; links to content shared privately"},
    "search": {"host": "search", "label": "Search", "kinds": (), "publish": False,
               "discovery": ("search",),
               "doc": "the 4.0 search engine; every public catalogue item is indexed by kind"},
    "creators": {"host": "creators", "label": "Creator subscriptions", "kinds": KINDS,
                 "discovery": ("subscriptions", "search"),
                 "doc": "paid creator memberships through the bank; members-only content"},
    "letters": {"host": "letters", "label": "Newsletters", "kinds": ("newsletter",),
                "discovery": ("subscriptions", "search", "share"),
                "doc": "newsletters delivered by mail to free or paid subscribers"},
    "launch": {"host": "launch", "label": "Product discovery", "kinds": ("post", "app", "video_short", "image"),
               "discovery": ("launches", "upvotes", "search", "share"),
               "doc": "product launches that link to real product pages; upvotes and comments"},
    "dev": {"host": "dev", "label": "Developer communities", "kinds": ("package", "app", "article", "thread"),
            "discovery": ("tags", "search", "following", "dependencies"),
            "doc": "software packages and apps, technical articles and developer Q&A"},
    "kb": {"host": "kb", "label": "Knowledge sharing", "kinds": ("thread", "article", "app"),
           "discovery": ("questions", "tags", "search"),
           "doc": "questions and answers with accepted answers, and educational products"},
}
PLATFORM_KINDS = tuple(PLATFORMS)


def platform(name: str) -> dict:
    p = PLATFORMS.get(name)
    if p is None:
        raise KeyError(name)
    return dict(p, name=name)


def accepts(name: str, kind: str) -> bool:
    p = PLATFORMS.get(name)
    return bool(p) and kind in p["kinds"]


def hosts() -> dict[str, str]:
    return {n: p["host"] for n, p in PLATFORMS.items()}
