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
whose words lie almost entirely (90%) in five-word passages of one earlier item by another creator, in any order;
exact against every earlier item).

Reference (hidden, LABELLED SCRIPTED): a tool-only creator that reads what is being read on its platforms, writes
original pieces on the most-read subjects (sentences of the real news corpus the population also writes from, in its
own arrangement) and publishes regularly on every text platform the world has.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, register_module
from . import _t2kit as K
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, Seat, civ_recipe, events_where, register

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


CATALOGUE_DEFAULTS = {"platforms": [], "per_platform": 6}


@register_module
class BackCatalogue(Module):
    """The platforms' back catalogue at genesis (kernel module `catalogue`, workstream I-1): a platform is not empty
    when the run starts. For every configured platform a few individuals of the one population (drawn per world
    from its own stream) have published pieces about what they care about: the nearest corpus topic to their
    interests in the shared space, in their own words (`population.writing`, skill shapes length), stored as their
    artifacts - a text, or on an image platform a text card drawn by the studio - and published through the
    ordinary catalogue path (`content.publish`: search index, distribution, the shared space's representation,
    exact audience accounting). Nothing about these items is assigned: they compete for the same feeds as everyone
    else's. Fidelity: plausible, unvalidated (FR-CATALOGUE-1)."""
    name = "catalogue"
    host = "catalogue"
    depends = ("identity", "pop", "cols", "content", "studio", "aud", "feed", "social_ids")

    def cfg(self, k):
        return self.config.get(k, CATALOGUE_DEFAULTS[k])

    def genesis(self, ctx):
        from .. import population
        from ..population import semantics, writing
        from ..social import content as C
        from ..substrate import artifacts, columns, ids
        w = ctx.world
        n = population.n_individuals(w)
        plats = [p for p in self.cfg("platforms") if p in C.PLATFORMS]
        per = int(self.cfg("per_platform"))
        if not n or not plats or per <= 0:
            return
        rng = w.rng_for("catalogue-genesis")
        T = len(semantics.topic_names())
        for plat in plats:
            idx = np.sort(rng.choice(n, size=min(per, n), replace=False)).astype(np.int64)
            coords = population.interest_coords(w, idx)
            tc = semantics.topic_coords(coords.shape[1])
            topic = np.argmax(coords @ tc.T + rng.gumbel(size=(idx.size, T)) * 0.08, axis=1)
            skill = columns.col(w, "pop", "skills")[idx, 1].astype(np.float64)
            for j, i in enumerate(idx.tolist()):
                longform = plat in ("blog", "kb")
                n_sent = int(4 + rng.integers(0, 8) * (0.5 + skill[j])) if longform else int(1 + rng.integers(0, 3))
                title, body = writing.compose(rng, int(topic[j]), n_sent, float(skill[j]))
                p = ids.ensure_principal(ctx, int(i))
                try:
                    if plat == "pics":
                        ref = ctx.call("studio.image", {"text": body[:600], "title": title[:180]},
                                       as_principal=p)["ref"]
                        kind = "image"
                    else:
                        data = f"{title}\n\n{body}\n".encode("utf-8")
                        ref = artifacts.put(ctx, data, "text/plain", owner=p, meta={"title": title[:80]}, kind="text")
                        kind = "article" if longform else "post"
                    C.module(w).publish(ctx, p, ref, kind, plat, title[:180])
                except CrucibleError:
                    continue


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
               "docs", "catalogue"] + plats
    pop = {"profile": population, "tick_s": 900, "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}
    if n_pop:
        pop["n"] = int(n_pop)
    elif population == "development":
        pop["n"] = 1500
    cfg = {"aud": {"platforms": plats, "max_items": min(1 << 20, max(8192, 4 * n)),
                   "max_creators": min(1 << 20, max(2048, n + 256))},
           "feed": {"platforms": {p: {"recommenders": SW.MIXTURES[p]} for p in plats}},
           "pop_social": {"platforms": _text_platforms(plats)},
           "catalogue": {"platforms": plats, "per_platform": 6},
           "pop": pop}
    return civ_recipe(int(seed), modules=modules, population_modules=("pop",), module_config=cfg,
                      population=population if population in ("development", "standard", "large", "internet")
                      else "development", params={"log_retain": 3_000_000})


# ====================================================================== objective, evaluator
def _goal_draw(rng, difficulty: int, world) -> dict:
    """The goal generator's draw (`crucible.goals`, its own stream): the level's distinct-viewer target (varied by
    +-10% per instance and rounded below the top level; the top level is the spec's 10,000,000 viewers of a
    10,000,000-individual internet population) and the population profile its world is built with."""
    d = int(min(4, max(1, difficulty)))
    t = TARGET[d] * float(rng.uniform(0.9, 1.1)) if d < max(TARGET) else float(TARGET[d])   # the top level: 10M
    step = 10 ** max(1, int(math.log10(TARGET[d])) - 1)
    return {"target_unique_viewers": int(round(t / step) * step), "population_profile": POPULATION_FOR[d],
            "window_days": HORIZON_DAYS}


def _goal_text(p: dict) -> str:
    return (f"Create original content that reaches a large independent audience: within {p['window_days']:.0f} days "
            f"your own original items must be viewed by at least {p['target_unique_viewers']:,} distinct people "
            f"(each person counts once; a view needs the platform's minimum watch time). Content first published "
            f"by someone else does not count.")


GOAL = GoalFamily(
    "original_audience", draw=_goal_draw, text=_goal_text,
    success=lambda p: {"metric": "content_original_viewers", "args": {"principal": "$self"}, "op": ">=",
                       "value": int(p["target_unique_viewers"])},
    progress=lambda p: {"metric": "content_original_viewers", "args": {"principal": "$self"},
                        "target": int(p["target_unique_viewers"])},
    window_days=lambda p: float(p["window_days"]),
    doc="the exact union of the viewer bitsets of the seat's original items (aud distinct verified viewers)")


def objective(world, principal: str, difficulty: int = 1) -> dict:
    """The objective generated for a participant (`Domain.make_objective`: the goal generator above)."""
    return DOMAIN.make_objective(world, principal, difficulty)


_WORD = re.compile(r"[a-z]{3,}")


def _shingles(text: str, k: int = 5) -> set:
    w = _WORD.findall(str(text).lower())
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")           # sentence and line boundaries (the copycat probe's rearranging)
COPY_SHARE = 0.9          # share of a text's words covered by five-word shingles of ONE earlier item of another creator


class Originality:
    """Evaluator-side originality over the whole catalogue, in publication order: an item is a copy when its artifact
    was first published by another creator, or when at least `COPY_SHARE` of its words lie in five-word shingles
    that also occur in one earlier item of another creator - a re-typed text, the same text with a few words added,
    or its sentences rearranged (coverage by shingles does not depend on the order of the passages, unlike the share
    of shingles, which loses those across passage boundaries). Exact against every earlier item (an inverted shingle
    index; no sample or cap: the newest items, which a copycat sees first, are compared like the oldest)."""

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

    def coverage(self, text: str) -> float:
        """The largest share of the text's words covered by shingles of a single earlier item."""
        w = _WORD.findall(str(text).lower())
        if len(w) < 5:
            return 0.0
        pos: dict[int, list[int]] = {}
        for i in range(len(w) - 4):
            for k in set(self.index.get(" ".join(w[i:i + 5]), ())):
                pos.setdefault(k, []).append(i)
        best = 0
        for k, ps in pos.items():
            if 5 * len(ps) < best:               # cannot cover more words than the best so far
                continue
            cov = np.zeros(len(w), bool)
            for i in ps:
                cov[i:i + 5] = True
            best = max(best, int(cov.sum()))
        return best / len(w)

    def is_copy(self, ref: str, mine: set, text: str) -> bool:
        first = self.first_ref.get(ref)
        if first is not None and first not in mine:
            return True
        return self.coverage(text) >= COPY_SHARE


def _text_of(world, ref: str) -> str:
    from ..substrate import artifacts
    try:
        info = artifacts.info(world, ref)
        if not str(info.get("mime", "")).startswith("text/"):
            return ""
        return artifacts.get(world, ref).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unreadable artifact has no text to compare
        return ""


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


def verdict(world, principal: str, target_unique_viewers: int) -> dict:
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
    t = int(target_unique_viewers)
    return {"success": n >= t, "score": round(min(1.0, n / t), 6),
            "details": {"distinct_viewers": n, "target": t, "items": per, "original_items": len(orig),
                        "copied_items": len(copied)}}


def evaluate(world, principal: str, obj: dict) -> dict:
    """The evaluator (hidden): `verdict` with the generated goal's parameters."""
    p = obj.get("params") or {}
    return verdict(world, principal, int(p.get("target_unique_viewers", TARGET[1])))


def _register_metrics() -> None:
    """The goal's metric (`crucible.goals.metrics`, evaluator side; no in-world subject, evaluator-only)."""
    from ..goals import metrics as GM

    @GM.metric("content_original_viewers", "EVALUATOR-ONLY. Distinct verified viewers of the principal's original "
               "items (exact union of their viewer bitsets).")
    def m_viewers(world, principal: str) -> float:
        return float(verdict(world, principal, 1)["details"]["distinct_viewers"])

    GM.EVALUATOR_ONLY_METRICS.add("content_original_viewers")


_register_metrics()


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
                # a piece of its own: sentences on the subject in its own order (one sentence at a time)
                parts = [writing.compose(rng, topic, 1, 0.95) for _ in range(10 if "blog" in text_plats else 5)]
                title, body = parts[0][0], " ".join(b for _, b in parts)
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
    """Degenerate: republish the newest items of other creators (their texts re-typed, every other one with its
    sentences rearranged)."""
    def script(pol):
        rng = np.random.default_rng(7)
        n_copied = 0
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
                    n_copied += 1
                    if n_copied % 2 == 0:
                        sents = [x for x in _SPLIT.split(text) if x.strip()]
                        text = " ".join(sents[int(j)] for j in rng.permutation(len(sents)))
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
# Evidence functions return the SEQUENCE NUMBERS of the recorded events that exercise a mechanic (D-48 /3).
def stored_artifacts(world, since: int) -> list[int]:
    """Published catalogue items whose bytes are stored artifacts (text, documents, images, audio, video)."""
    from ..substrate import artifacts

    def stored(e):
        try:
            return int(artifacts.info(world, e.payload["ref"])["size"]) > 0
        except Exception:  # noqa: BLE001 - a missing artifact does not count
            return False
    return events_where(world, since, ("content.published",), stored)


def population_publications(world, since: int) -> list[int]:
    """Items published by individuals of the population (the creators competing for the same feeds)."""
    return events_where(world, since, ("content.published",), lambda e: K.is_population(e.payload.get("creator")))


MECHANICS = (
    Mechanic("Actual content artifacts", (), "stored artifacts published as catalogue items",
             count=stored_artifacts),
    Mechanic("Learned content representations", ("aud.item",),
             "items registered for distribution with their representation in the shared space (pretrained "
             "encoders, recorded inference)"),
    Mechanic("Autonomous consumption", ("cols.batch:consume.*",), "the population's own consumption decisions"),
    Mechanic("Recommendation systems", ("cols.batch:impress.*",), "feed impressions ranked by recommender mixtures"),
    Mechanic("Competing creators", ("cols.batch:pop_social.created",), "population creators publishing",
             count=population_publications),
    Mechanic("Social sharing", ("cols.batch:social.share", "cols.batch:share.deliver"), "shares and their delivery"),
    Mechanic("Audience feedback", ("cols.batch:social.follow", "content.comment", "cols.batch:analytics.publish"),
             "follows, comments, likes and the creators' delayed analytics"),
    Mechanic("Distinct verified viewers", ("aud.consumed",),
             "viewings whose applier decides views from watch time and sets the exact distinct-viewer bitsets"),
)


def structure(world) -> dict:
    """Built state only (`base.BuiltWorld`): the platforms that carry a catalogue at genesis and the media they
    carry (the back catalogue, module `catalogue`)."""
    st = world.state
    items = [c for _, c in st.scan("content")] if "content" in st.tables else []
    return {"platforms": sorted({str(c.get("platform")) for c in items}),
            "media": sorted({str(c.get("kind")) for c in items})}


CONFIGS = tuple(DomainConfig(n, s, dict(STRUCTURES[n]), DOCS[n]) for n, s in (
    ("short_social", "train"), ("visual_trending", "train"), ("longform", "train"),
    ("visual_longform", "eval_compositional"), ("knowledge_base", "eval_structural")))

DOMAIN = register(Domain(
    name="content", title="Content ecosystem", spec_part="9.3", required_mechanics=tuple(m.name for m in MECHANICS),
    configs=CONFIGS, build=build, goal=GOAL, evaluate=evaluate, reference=reference, mechanics=MECHANICS,
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
