"""Tier 2 domain 9.3 - Content ecosystem (MASTER_SPEC Part 9.3; workstream I-1).

Example objective: create original content that reaches a large independent audience.

A configuration is the civilisation's social-media slice (D-17, `base.civ_recipe`): the content catalogue and the
creator studio (actual artifacts: text, documents, images, audio, video, apps), content representations in the ONE
shared semantic space (`space`: pretrained encoders, recorded inference, D-16), the exact audience accounting with
distinct-viewer bitsets (`aud`), one recommender mixture per platform (`feed`: social, chronological, engagement,
interest, trending, collaborative, search, reputation, diversity, explore - genuinely different mechanisms), the
follow/share graph, and the ONE population, which browses its feeds (consumption decided by each individual's own
interests in the space, the item's artifact-intrinsic features, attention and social signals: views, completions,
likes, shares, follows) and creates (`pop_social`: members write pieces about what they care about in their own
words and keep creating while their analytics reward it - the competing creators).

Configurations differ in which platforms (and so which recommender mechanisms and media) the world has, measured on
the built world (`structure`). Difficulty levels scale the population and the audience target up to the internet
profile: level 4 is a world of 10,000,000 individuals and a target of 10,000,000 distinct verified viewers
(`POPULATION_FOR` / `TARGET`; build with `overrides={"population": POPULATION_FOR[d]}`).

Evaluator (hidden): the exact union of the viewer bitsets of the seat's original items (distinct verified viewers:
an individual counts once however many items it viewed; a view needs the platform's minimum watch time), excluding
items whose artifact or text the seat did not originate (an artifact first published by someone else, or a text
whose five-word shingles appear almost entirely (90%) in an earlier item by another creator).

Reference (hidden, LABELLED SCRIPTED): a tool-only creator that reads what is being read on its platforms, writes
original pieces on the most-read subjects (composed from the real news corpus the population also writes from) and
publishes regularly on every text platform the world has.
"""
from __future__ import annotations

import re

import numpy as np

from . import _t2kit as K
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, civ_recipe, register

DAY = 86400
HORIZON_DAYS = 10.0
TARGET = {1: 500, 2: 5_000, 3: 100_000, 4: 10_000_000}          # distinct verified viewers
POPULATION_FOR = {1: "development", 2: "standard", 3: "large", 4: "internet"}

STRUCTURES = {
    "short_social": {"platforms": ["posts"], "media": ["text"]},
    "visual_trending": {"platforms": ["pics", "posts"], "media": ["image", "text"]},
    "longform": {"platforms": ["blog"], "media": ["text"]},
    "visual_longform": {"platforms": ["blog", "pics"], "media": ["image", "text"]},
    "knowledge_base": {"platforms": ["kb", "posts"], "media": ["text"]},
}
DOCS = {
    "short_social": "short posts only: social, chronological, engagement, interest and exploration feeds",
    "visual_trending": "image cards and short posts: trending and interest feeds for images",
    "longform": "long articles only: interest, engagement, search, reputation and chronological feeds",
    "visual_longform": "compositional: image cards AND long articles (each seen in training, never together)",
    "knowledge_base": "structural: a search-driven knowledge platform with diversity re-ranking (a discovery "
                      "mechanism no training world has) beside short posts",
}


def _text_platforms(plats: list[str]) -> dict:
    short = "posts" if "posts" in plats else ("blog" if "blog" in plats else plats[0])
    long_ = "blog" if "blog" in plats else short
    return {"short": short, "long": long_}


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    from ..population import profiles as PP
    from ..social import worlds as SW
    st = STRUCTURES[config.name]
    ov = dict(overrides or {})
    population = str(ov.pop("population", "development"))
    n_pop = ov.pop("n", None)
    if ov:
        raise ValueError(f"content domain: unknown overrides {sorted(ov)}")
    plats = list(st["platforms"])
    n = int(PP.resolve({"profile": population, **({"n": int(n_pop)} if n_pop else {})})["n"])
    modules = ["content", "social", "social_ids", "social_rows", "aud", "feed", "studio", "pop_social", "search",
               "docs"] + plats
    pop = {"profile": population, "tick_s": 900, "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}
    if n_pop:
        pop["n"] = int(n_pop)
    elif population == "development":
        pop["n"] = 1500
    cfg = {"aud": {"platforms": plats, "max_items": min(1 << 20, max(8192, 4 * n)),
                   "max_creators": min(1 << 20, max(2048, n + 256))},
           "feed": {"platforms": {p: {"recommenders": SW.MIXTURES[p]} for p in plats}},
           "pop_social": {"platforms": _text_platforms(plats)},
           "pop": pop}
    return civ_recipe(int(seed), modules=modules, population_modules=("pop",), module_config=cfg,
                      population=population if population in ("development", "standard", "large", "internet")
                      else "development", params={"log_retain": 3_000_000})


# ====================================================================== objective, evaluator
def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = int(min(4, max(1, difficulty)))
    text = (f"Create original content that reaches a large independent audience: within {HORIZON_DAYS:.0f} days "
            f"your own original items must be viewed by at least {TARGET[d]:,} distinct people (each person counts "
            f"once; a view needs the platform's minimum watch time). Content first published by someone else does "
            f"not count.")
    return {"text": text, "description": text, "domain": "content", "difficulty": d,
            "target_unique_viewers": TARGET[d], "population_profile": POPULATION_FOR[d], "window_days": HORIZON_DAYS}


_WORD = re.compile(r"[a-z]{3,}")


def _shingles(text: str, k: int = 5) -> set:
    w = _WORD.findall(str(text).lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def _text_of(world, ref: str) -> str:
    from ..substrate import artifacts
    try:
        info = artifacts.info(world, ref)
        if not str(info.get("mime", "")).startswith("text/"):
            return ""
        return artifacts.get(world, ref).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreadable artifact has no text to compare
        return ""


COPY_SHARE = 0.9          # share of a text's five-word shingles found in ONE earlier item of another creator


class Originality:
    """Evaluator-side originality over the whole catalogue, in publication order: an item is a copy when its artifact
    was first published by another creator, or when at least `COPY_SHARE` of its five-word shingles appear in one
    earlier item of another creator (a re-typed text). Exact against every earlier item (an inverted shingle index;
    no sample or cap: the newest items, which a copycat sees first, are compared like the oldest)."""

    def __init__(self):
        self.first_ref: dict[str, str] = {}
        self.index: dict[str, list[int]] = {}
        self.n = 0

    def add_other(self, ref: str, creator: str, text: str) -> None:
        self.first_ref.setdefault(ref, creator)
        k = self.n
        self.n += 1
        for s_ in _shingles(text):
            self.index.setdefault(s_, []).append(k)

    def is_copy(self, ref: str, mine: set, text: str) -> bool:
        first = self.first_ref.get(ref)
        if first is not None and first not in mine:
            return True
        sh = _shingles(text)
        if not sh:
            return False
        hits: dict[int, int] = {}
        for s_ in sh:
            for k in self.index.get(s_, ()):
                hits[k] = hits.get(k, 0) + 1
        return bool(hits) and max(hits.values()) >= COPY_SHARE * len(sh)


def original_items(world, principal: str) -> tuple[list[dict], list[dict]]:
    """(original, copied) catalogue items of the principal (and organisations it administers)."""
    st = world.state
    mine = set(world.principals_of(principal)) | {principal} if world.principal(principal) else {principal}
    items = sorted(((c["published_at"], cid, c) for cid, c in st.scan("content")), key=lambda x: (x[0], x[1]))
    og = Originality()
    orig, copied = [], []
    for t, cid, c in items:
        if c["creator"] not in mine:
            og.add_other(c["ref"], c["creator"], _text_of(world, c["ref"]))
        elif og.is_copy(c["ref"], mine, _text_of(world, c["ref"])):
            copied.append(c)
        else:
            orig.append(c)
    return orig, copied


def evaluate(world, principal: str, obj: dict) -> dict:
    from ..social import audience as AUD
    orig, copied = original_items(world, principal)
    viewers = np.zeros(0, np.int64)
    per = {}
    for c in orig:
        if c.get("aud_item") is None:
            continue
        v = AUD.viewers_of(world, int(c["aud_item"]))
        per[c["id"]] = int(v.size)
        viewers = np.union1d(viewers, v)
    n = int(viewers.size)
    t = int(obj["target_unique_viewers"])
    return {"success": n >= t, "score": round(min(1.0, n / t), 6),
            "details": {"distinct_viewers": n, "target": t, "items": per, "original_items": len(orig),
                        "copied_items": len(copied)}}


# ====================================================================== the hidden reference (LABELLED SCRIPTED)
def _tag(name) -> str:
    t = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-")[:32]
    return t if t and t[0].isalnum() else "topic"


_STOP = frozenset("the a an and or of to in on for with at by from is are was were be it its this that as new how "
                  "why what who says said will about after over into more than".split())


def reference_script(per_day: int = 3):
    """A tool-only creator (see the module doc)."""
    def script(pol):
        from ..population import semantics, writing
        feeds = K.ok((yield {"tool": "feed.platforms", "args": {}})) or {}
        fnames = [p.get("platform") for p in (feeds.get("platforms") if isinstance(feeds, dict) else feeds) or []]
        text_plats = [p for p in ("posts", "blog", "kb") if p in fnames]
        image_plats = [p for p in ("pics",) if p in fnames]
        names = semantics.topic_names()
        rng = np.random.default_rng(11)
        k = 0
        while True:
            words: dict = {}
            for p in text_plats + image_plats:
                items = K.ok((yield {"tool": "content.list", "args": {"platform": p, "limit": 50}})) or []
                for it in (items if isinstance(items, list) else items.get("items", [])):
                    for wd in str(it.get("title", "")).lower().replace(",", " ").split():
                        if len(wd) > 3 and wd not in _STOP and wd.isalpha():
                            words[wd] = words.get(wd, 0) + 1
            scores = []
            for t, nm in enumerate(names):
                scores.append(sum(words.get(x, 0) for x in str(nm).lower().replace("_", " ").split()))
            order = list(np.argsort(-np.array(scores, float), kind="stable"))
            for j in range(per_day):
                topic = int(order[(k + j) % max(1, min(5, len(order)))]) if order else int(rng.integers(0, 10))
                title, body = writing.compose(rng, topic, 10 if "blog" in text_plats else 5, 0.95)
                for p in text_plats:
                    ref = (K.ok((yield {"tool": "studio.write", "args": {"format": "text", "title": title[:180],
                                                                         "body": body}})) or {}).get("ref")
                    if ref:
                        kind = "article" if p in ("blog", "kb") else "post"
                        yield {"tool": "content.publish", "args": {"ref": ref, "kind": kind, "platform": p,
                                                                   "title": title[:180],
                                                                   "tags": [_tag(names[topic])]}}
                for p in image_plats:
                    img = (K.ok((yield {"tool": "studio.image", "args": {"text": f"{title}\n\n{body[:300]}",
                                                                         "title": title[:180]}})) or {}).get("ref")
                    if img:
                        yield {"tool": "content.publish", "args": {"ref": img, "kind": "image", "platform": p,
                                                                   "title": title[:180]}}
            k += per_day
            for _ in range(3):
                yield None
    return script


def reference(principal: str, obj: dict):
    return K.ScriptPolicy(reference_script(), label="content-reference")


def _copycat_script():
    """Degenerate: republish the most viewed items of other creators (their artifacts and texts)."""
    def script(pol):
        while True:
            for p in ("posts", "blog", "pics", "kb"):
                items = K.ok((yield {"tool": "content.list", "args": {"platform": p, "limit": 20}})) or []
                for it in (items if isinstance(items, list) else items.get("items", []))[:3]:
                    if it.get("creator") == pol.actor or not it.get("id"):
                        continue
                    got = K.ok((yield {"tool": "content.open", "args": {"id": it["id"]}})) or {}
                    text = got.get("text") or ""
                    if not text:
                        continue
                    ref = (K.ok((yield {"tool": "studio.write", "args": {"format": "text",
                                                                         "title": str(it.get("title", "x"))[:180],
                                                                         "body": text}})) or {}).get("ref")
                    if ref:
                        yield {"tool": "content.publish", "args": {"ref": ref, "kind": it.get("kind", "post"),
                                                                   "platform": p,
                                                                   "title": str(it.get("title", "x"))[:180]}}
            for _ in range(3):
                yield None
    return script


def copycat(principal: str, obj: dict):
    return K.ScriptPolicy(_copycat_script(), label="copycat")


def _spam_script():
    """Degenerate: many short, meaningless items an hour (keyword stuffing)."""
    def script(pol):
        rng = np.random.default_rng(3)
        while True:
            for _ in range(6):
                body = " ".join(str(x) for x in rng.choice(["best", "viral", "free", "top", "news", "wow", "now"],
                                                           size=30))
                ref = (K.ok((yield {"tool": "studio.write", "args": {"format": "text", "title": "top viral news now",
                                                                     "body": body}})) or {}).get("ref")
                if ref:
                    yield {"tool": "content.publish", "args": {"ref": ref, "kind": "post", "platform": "posts",
                                                               "title": "top viral news now"}}
            yield None
    return script


def spam(principal: str, obj: dict):
    return K.ScriptPolicy(_spam_script(), label="spam")


def null_policy(principal=None, obj=None):
    def script(pol):
        while True:
            yield None
    return K.ScriptPolicy(script, label="null")


# ====================================================================== mechanics (recorded-event probes)
def _ev(world, since, types):
    return [e for e in world.events_range(int(since), world.seq) if e.type in types]


def count_artifacts(world, since: int) -> int:
    """Published catalogue items whose bytes are stored artifacts (text, documents, images, audio, video)."""
    st = world.state
    from ..substrate import artifacts
    n = 0
    for e in _ev(world, since, ("content.published",)):
        try:
            n += int(artifacts.info(world, e.payload["ref"])["size"] > 0)
        except Exception:  # noqa: BLE001 - a missing artifact does not count
            continue
    return n


def count_representations(world, since: int) -> int:
    """Items represented in the shared space by pretrained encoders through recorded inference."""
    return sum(1 for e in world.events_range(int(since), world.seq)
               if e.type in ("space.recorded", "aud.item") or (e.type == "ext" and "space" in str(e.payload)[:80]))


def count_competing_creators(world, since: int) -> int:
    """Distinct creators with published items after `since` beyond the first (competition for the same feeds)."""
    who = {e.payload.get("creator") for e in _ev(world, since, ("content.published",))}
    return max(0, len(who) - 1)


def count_distinct_viewers(world, since: int) -> int:
    from ..social import audience as AUD
    st = world.state
    allv = np.zeros(0, np.int64)
    for _, c in st.scan("content"):
        if c.get("aud_item") is not None:
            allv = np.union1d(allv, AUD.viewers_of(world, int(c["aud_item"])))
    return int(allv.size)


MECHANICS = (
    Mechanic("Actual content artifacts", ("content.published",), "stored artifacts published as catalogue items",
             count=count_artifacts),
    Mechanic("Learned content representations", ("space.recorded",),
             "pretrained-encoder representations in the shared space", count=count_representations),
    Mechanic("Autonomous consumption", ("cols.batch:consume.*",), "the population's own consumption decisions"),
    Mechanic("Recommendation systems", ("cols.batch:impress.*",), "feed impressions ranked by recommender mixtures"),
    Mechanic("Competing creators", ("cols.batch:pop_social.created",), "population creators publishing",
             count=count_competing_creators),
    Mechanic("Social sharing", ("cols.batch:social.share", "cols.batch:share.deliver"), "shares and their delivery"),
    Mechanic("Audience feedback", ("cols.batch:social.follow", "content.comment", "cols.batch:analytics.publish"),
             "follows, comments, likes and the creators' delayed analytics"),
    Mechanic("Distinct verified viewers", (), "exact distinct-viewer bitsets", count=count_distinct_viewers),
)


def structure(world) -> dict:
    feed = world.modules["feed"].config.get("platforms") or {}
    recs = sorted({r["name"] for p in feed.values() for r in (p.get("recommenders") or [])})
    return {"platforms": sorted(feed), "recommenders": recs}


CONFIGS = tuple(DomainConfig(n, s, dict(STRUCTURES[n]), DOCS[n]) for n, s in (
    ("short_social", "train"), ("visual_trending", "train"), ("longform", "train"),
    ("visual_longform", "eval_compositional"), ("knowledge_base", "eval_structural")))

DOMAIN = register(Domain(
    name="content", title="Content ecosystem", spec_part="9.3", required_mechanics=tuple(m.name for m in MECHANICS),
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=reference, mechanics=MECHANICS,
    seats=(Seat("creator", 100_000, "Creator", "an independent creator"),),
    actors=(Actor("population audience", ("cols.batch:consume.*", "cols.batch:impress.*"),
                  "individuals browsing their feeds and consuming by their own interests"),
            Actor("population creators", ("cols.batch:pop_social.created",), "individuals writing and publishing")),
    null_baseline=null_policy,
    degenerate=(Probe("copycat", "republish other creators' most viewed items", copycat),
                Probe("spam", "many meaningless keyword-stuffed posts an hour", spam)),
    structure=structure, horizon_days=HORIZON_DAYS, slowest_effect_days=3.0, dt=6 * 3600,
    difficulties=(1, 2, 3, 4), reference_seeds=(0, 1), sources=("content_domain.py",),
    doc="The social internet's content economy on the one population: artifacts, representations, feeds, "
        "creators, sharing, feedback and exact distinct viewers."))
