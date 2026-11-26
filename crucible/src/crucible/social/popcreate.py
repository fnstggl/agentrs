"""The population's creation on every platform of the social internet (activity `social.create` of `pop_social`).

A member who chooses to create picks a MEDIUM - what to make and where to publish it - by its own taste and
abilities and by what its earlier pieces of each medium earned: the views its pieces of that medium gained since it
last looked, as the platforms publish them to their creator (delayed, noisy analytics), against the effort. So
creators drift towards the media where they find an audience, and away from the ones where they do not.

    medium   platform  kind          made of (crucible.social.popmedia, recorded rendering)
    post     posts     post          1-3 sentences of one's own writing (text)
    status   social    post          a 1-2 sentence status update on the social network (text)
    article  blog      article       4-12 sentences (text)
    image    pics      image         a poster card of the piece's title and lead sentence (PNG)
    audio    pods      audio         the piece narrated by a synthetic voice after a jingle (Ogg Opus)
    short    shorts    video_short   2-4 narrated slides with subtitles (MP4, at most 3 minutes)
    video    tube      video_long    8-14 narrated slides with subtitles (MP4, at least one minute)
    answer   kb        thread        answer an open question that fits one's interests (kb.answer), or ask one
    tech     dev       thread        the same on the developer platform (technical topics, technical skill)

Only media whose platform is part of the world are available. Writing is `population.writing` (real news text of the
creator's topic: skill shapes coherence); the topic is the nearest corpus topic to one's interests with personal
variation (technical topics for `tech`). Every artifact is the creator's own (artifact store), published through
`content.publish` (catalogue, search, distribution, exact audience accounting) or the Q&A tools as the creator's
principal. No piece carries a quality, relevance or virality figure; how far it travels is decided by the platforms'
recommenders and the audience's own decisions on its bytes.

Decision provenance: one `pop_social.created` batch per cohort (model `pop-create@2`), with the media chosen.
Fidelity: plausible, unvalidated (FIDELITY_REGISTER `population-creator-model`, `population-media-creation`).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Ctx
from ..substrate import artifacts, attention, columns, ids
from .. import population
from ..population import semantics, writing
from . import audience as AUD
from . import content as C

CREATE_MODEL = {"model_id": "pop-create", "model_version": "2", "fidelity_id": "population-creator-model"}
#: medium -> (platform, kind, minutes lo/hi, sentences lo/hi)
MEDIA = {
    "post": ("posts", "post", (5, 15), (1, 3)),
    "status": ("social", "post", (1, 5), (1, 2)),
    "article": ("blog", "article", (20, 60), (4, 12)),
    "image": ("pics", "image", (5, 20), (1, 1)),
    "audio": ("pods", "audio", (20, 60), (3, 7)),
    "short": ("shorts", "video_short", (15, 40), (2, 4)),
    "video": ("tube", "video_long", (40, 120), (9, 14)),
    "answer": ("kb", "thread", (10, 30), (2, 5)),
    "tech": ("dev", "thread", (15, 45), (2, 6)),
}
MEDIUM_NAMES = tuple(MEDIA)
NM = len(MEDIUM_NAMES)
# taste and ability per medium: logit = base + sum(weight x skill/pref) (schema.SKILL_NAMES / PREF_NAMES indices)
_SK = {"searching": 0, "writing": 1, "evaluating": 2, "shopping": 3, "technical": 4, "social": 5, "reading": 6,
       "planning": 7}
_PF = {"novelty": 1, "sociability": 6}
PRIORS = {
    "post": (0.0, {"writing": 1.0}, {}),
    "status": (-0.3, {"social": 0.6}, {"sociability": 0.8}),
    "article": (-1.0, {"writing": 2.0}, {}),
    "image": (-0.7, {"social": 1.0}, {"novelty": 0.5}),
    "audio": (-1.5, {"social": 0.8, "technical": 0.8}, {}),
    "short": (-1.2, {"technical": 0.8}, {"novelty": 0.9}),
    "video": (-2.0, {"technical": 1.0, "writing": 1.0}, {}),
    "answer": (-1.2, {"reading": 1.0, "evaluating": 0.6}, {}),
    "tech": (-2.6, {"technical": 2.8}, {}),
}
TECH_TOPICS = ("newsscienceandtechnology", "science", "gaming", "finance-companies", "markets", "medical")
VALUE_WEIGHT = 2.0          # how much a medium's experienced value counts against taste
VALUE_RATE = 0.3            # learning rate of the per-medium value
_SENT = re.compile(r"(?<=[.!?])\s+")


def columns_spec() -> dict:
    return {"soc_med_value": ("float32", (NM,), 0.0), "soc_med_seen": ("int64", (NM,), 0)}


def medium_of(platform: str, kind: str) -> int | None:
    for j, (p, k, _, _) in enumerate(MEDIA.values()):
        if p == platform and (k == kind or p in ("kb", "dev")):
            return j
    if platform in ("blog", "letters"):
        return MEDIUM_NAMES.index("article")
    return None


def available(world) -> np.ndarray:
    """Media whose platform is served in this world and distributed to the population (an audience platform)."""
    hosts = {e.host for e in world.endpoints.values()}
    aud = world.modules.get("aud")
    dist = set(getattr(aud, "platforms", ()))
    from .platforms import PLATFORMS
    return np.array([PLATFORMS[p]["host"] in hosts and p in dist for p, _, _, _ in MEDIA.values()], bool)


def views_by_medium(world, i: int) -> np.ndarray:
    """Views of individual i's items per medium, as the platforms publish them to their creator."""
    p = ids.pid(i)
    out = np.zeros(NM, np.int64)
    st = world.state
    for r in st.find_ids("aud_item_info", "creator", p):
        rec = st.tables["aud_item_info"][r]
        m = medium_of(rec["platform"], rec["kind"])
        if m is not None:
            out[m] += int(AUD.audience(world, int(r), principal=p)["views"])
    return out


def _priors(skills: np.ndarray, prefs: np.ndarray) -> np.ndarray:
    out = np.zeros((skills.shape[0], NM))
    for j, m in enumerate(MEDIUM_NAMES):
        base, sk, pf = PRIORS[m]
        v = np.full(skills.shape[0], base)
        for k, w in sk.items():
            v += w * skills[:, _SK[k]]
        for k, w in pf.items():
            v += w * prefs[:, _PF[k]]
        out[:, j] = v
    return out


def _sentences(body: str) -> list[str]:
    return [s.strip() for s in _SENT.split(body) if s.strip()]


def _render(ctx: Ctx, medium: str, p: str, i: int, title: str, body: str) -> tuple[str, str, str]:
    """(artifact REF, mime, artifact kind) of the creator's piece in `medium` (recorded rendering)."""
    from . import popmedia as PM
    from . import render
    w = ctx.world
    if medium in ("post", "status", "article"):
        data = f"{title}\n\n{body}\n".encode("utf-8")
        return artifacts.put(ctx, data, "text/plain", owner=p, meta={"title": title[:80]}, kind="text"), \
            "text/plain", "text"
    style = PM.style_of(w.rng_for("pop_social.style", i).integers(0, 2 ** 31))
    sents = _sentences(body)
    inputs = {"medium": medium, "title": title, "body": body, "style": style}
    if medium == "image":
        mime, akind = "image/png", "image"
        produce = lambda: PM.image_card(title, sents[0] if sents else "", style)       # noqa: E731
    elif medium == "audio":
        mime, akind = "audio/ogg", "audio"
        produce = lambda: PM.narration(f"{title}. {body}", style)                     # noqa: E731
    elif medium == "short":
        mime, akind = "video/mp4", "video"
        produce = lambda: PM.slideshow(title, sents, style, max_s=170.0)              # noqa: E731
    elif medium == "video":
        mime, akind = "video/mp4", "video"
        produce = lambda: PM.slideshow(title, sents, style, min_s=62.0, max_s=900.0)  # noqa: E731
    else:
        raise ValueError(medium)
    ref, _ = render.recorded_bytes(ctx, f"pop_social.{medium}", inputs, produce, mime=mime, owner=p,
                                   meta={"title": title[:80], "made_by": "population", "medium": medium},
                                   akind=akind)
    return ref, mime, akind


def _qa(ctx: Ctx, platform: str, p: str, coords_j: np.ndarray, rng, skill: float, k: int) -> str | None:
    """Answer the open question of `platform` that fits one's interests best (with noise), or ask one."""
    w = ctx.world
    st = ctx.state
    T = f"{platform}_posts"
    if T not in st.tables:
        return None
    open_q = []
    for qid in st.find_ids(T, "type", "question")[-200:]:
        q = st.tables[T][qid]
        c = st.get("content", qid)
        if c is None or c["removed"] or c["creator"] == p or c.get("aud_item") is None or q.get("accepted"):
            continue
        open_q.append((qid, int(c["aud_item"])))
    if open_q and rng.random() < 0.7:
        E = AUD.item_col(w, "emb")[[it for _, it in open_q]].astype(np.float32)
        fit = E @ coords_j.astype(np.float32) + rng.gumbel(size=len(open_q)) * 0.05
        b = int(np.argmax(fit))
        if fit[b] > 0.05:
            topic = int(semantics.nearest_topic(E[b:b + 1])[0])
            _, body = writing.compose(rng, topic, int(rng.integers(2, 6)), skill)
            try:
                return ctx.call(f"{platform}.answer", {"question": open_q[b][0], "body": body},
                                as_principal=p)["answer"]
            except CrucibleError:
                return None
    topic = _topic_for(coords_j[None, :], rng, platform == "dev", k)[0]
    title, body = writing.compose(rng, topic, int(rng.integers(1, 3)), skill)
    q = f"{title}: what do you know about this?"[:C.MAX_TITLE]
    try:
        return ctx.call(f"{platform}.ask", {"title": q, "body": body}, as_principal=p)["question"]
    except CrucibleError:
        return None


def _topic_for(coords: np.ndarray, rng, tech: bool, k: int) -> np.ndarray:
    names = semantics.topic_names()
    tc = semantics.topic_coords(k)
    score = coords @ tc.T + rng.gumbel(size=(coords.shape[0], len(names))) * 0.08
    if tech:
        mask = np.array([n in TECH_TOPICS for n in names])
        if mask.any():
            score = np.where(mask[None, :], score, -np.inf)
    return np.argmax(score, axis=1)


REVIEW_P = 0.35            # a member who relies on a provider that works for it recommends it this often


def _recommendation(ctx: Ctx, i: int, rng) -> tuple[str, str] | None:
    """(title, body) of a recommendation of a service the member relies on in its work and that works for it (its
    own adoption record of `economy.work`: at least 3 uses, at least 2/3 accepted), naming the service's address -
    or None. Read-only use of the member's own experience; the text is the member's (no score, no bonus)."""
    st = ctx.state
    work = ctx.world.modules.get("work")
    if work is None or "work_exp" not in st.tables:
        return None
    good = []
    for key in st.find_ids("work_exp", "i", int(i)):
        r = st.tables["work_exp"][key]
        if r["active"] and r["uses"] >= 3 and r["ok"] >= (2.0 / 3.0) * r["uses"]:
            prov = work._provider(st, r["prov"])
            if prov is not None and prov.get("host"):
                good.append((r["prov"], r, prov))
    if not good or rng.random() >= REVIEW_P:
        return None
    pid, r, prov = good[int(rng.integers(0, len(good)))]
    task = work.task_text(r["step_need"])
    name = (st.get("products", pid) or {}).get("name") or prov["host"]
    title = f"How I get my {task.split()[0].lower()} work done"[:C.MAX_TITLE]
    body = (f"For my work I need {task}. I use {name} (http://{prov['host']}.crucible/): it handled {int(r['ok'])} "
            f"of my last {int(r['uses'])} tasks well enough for my employer. Recommended if you do the same work.")
    return title, body


def act_create(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
    w = ctx.world
    idx = np.asarray(idx, np.int64)
    n = idx.size
    rng = ctx.rng("pop_social.create")
    coords = population.interest_coords(w, idx)
    k = coords.shape[1]
    skills = columns.col(w, "pop", "skills")[idx].astype(np.float64)
    prefs = columns.col(w, "pop", "prefs")[idx].astype(np.float64)
    left = columns.col(w, "pop", "attention_left_min")[idx].astype(np.float64)
    learn = "soc_med_value" in columns.tables(w)["pop"]["columns"]
    value = columns.col(w, "pop", "soc_med_value")[idx].astype(np.float64).copy() if learn else np.zeros((n, NM))
    seen = columns.col(w, "pop", "soc_med_seen")[idx].astype(np.int64).copy() if learn else np.zeros((n, NM), np.int64)
    seen_tot = columns.col(w, "pop", "soc_seen_views")[idx].astype(np.int64).copy()
    last_item = columns.col(w, "pop", "soc_last_item")[idx].astype(np.int64).copy()
    nitems = columns.col(w, "pop", "soc_items")[idx].astype(np.int64).copy()
    avail = available(w)
    logits = _priors(skills, prefs) + VALUE_WEIGHT * value + rng.gumbel(size=(n, NM)) * 0.7
    logits[:, ~avail] = -np.inf
    choice = np.argmax(logits, axis=1)
    share = float(self.cfg["creator_share"])
    if share < 1.0:
        from scipy.stats import beta as _beta              # skills are Beta(2.5, 2.5) (population genesis)
        writer = skills[:, 1] >= min(0.975, float(_beta.ppf(1.0 - share, 2.5, 2.5)))
    else:
        writer = np.ones(n, bool)
    topics = _topic_for(coords, rng, False, k)
    tech_topics = _topic_for(coords, rng, True, k)
    made = np.zeros(n, bool)
    spent = np.zeros(n)
    u = np.zeros(n)
    media_made: list[str] = []
    for j in range(n):
        i = int(idx[j])
        m = MEDIUM_NAMES[int(choice[j])]
        plat, kind, (lo, hi), (s_lo, s_hi) = MEDIA[m]
        minutes = min(left[j], lo + (hi - lo) * rng.random() * (0.5 + skills[j, 1]))
        if minutes < lo * 0.5 or not writer[j]:
            u[j] = -0.3 if not writer[j] else -0.05
            continue
        # what the creator learns from the platforms about its earlier pieces, per medium
        if nitems[j] and not learn:
            p_ = ids.pid(i)
            tot = int(sum(int(AUD.audience(w, int(r), principal=p_)["views"])
                          for r in w.state.find_ids("aud_item_info", "creator", p_)))
            g_all = max(0, tot - int(seen_tot[j]))
            seen_tot[j] = tot
        elif nitems[j]:
            views = views_by_medium(w, i)
            gained = np.maximum(0, views - seen[j])
            seen[j] = views
            for mm in np.flatnonzero(gained > 0):
                value[j, mm] += VALUE_RATE * (0.5 * math.log1p(float(gained[mm])) / math.log(30.0) - value[j, mm])
            tot = int(views.sum())
            g_all = max(0, tot - int(seen_tot[j]))
            seen_tot[j] = tot
        else:
            g_all = 0
        p = ids.ensure_principal(ctx, i)
        skill = float(skills[j, 1])
        n_sent = int(rng.integers(s_lo, s_hi + 1))
        cid = None
        rec_post = _recommendation(ctx, i, rng) if m in ("post", "status", "article") else None
        try:
            if rec_post is not None:
                # a recommendation of a service that works for the member (text naming its address)
                title, body = rec_post
                data = f"{title}\n\n{body}\n".encode("utf-8")
                ref = artifacts.put(ctx, data, "text/plain", owner=p, meta={"title": title[:80]}, kind="text")
                cid = C.module(w).publish(ctx, p, ref, kind, plat, title)
                media_made.append("recommendation")
            elif m in ("answer", "tech"):
                cid = _qa(ctx, plat, p, coords[j], rng, skill, k)
            else:
                topic = int(tech_topics[j] if m == "tech" else topics[j])
                title, body = writing.compose(rng, topic, n_sent, skill)
                ref, _, _ = _render(ctx, m, p, i, title, body)
                letter = self.creator_channels(ctx, i, p, rng, title) if m == "article" else None
                if letter is not None:
                    # the article goes out as an issue of the creator's newsletter (published on `letters` and
                    # delivered to its subscribers) instead of the blog
                    cid = ctx.call("letters.send", {"newsletter": letter, "ref": ref, "title": title},
                                   as_principal=p)["item"]
                else:
                    cid = C.module(w).publish(ctx, p, ref, kind, plat, title)
        except CrucibleError:
            cid = None
        if cid is None:
            u[j] = -0.1
            value[j, int(choice[j])] += VALUE_RATE * (-0.2 - value[j, int(choice[j])])
            continue
        rec = ctx.state.tables["content"][cid]
        if rec.get("aud_item") is not None:
            last_item[j] = int(rec["aud_item"])
        made[j] = True
        spent[j] = minutes
        nitems[j] += 1
        media_made.append(m)
        # the creator's experienced value: the audience its earlier pieces found, against the effort
        u[j] = 0.5 * math.log1p(g_all) / math.log(30.0) - 0.004 * minutes
    sp = np.flatnonzero(made & (spent > 0))
    if sp.size:
        attention.spend(ctx, idx[sp], spent[sp], activity="social.create")
    vals = {"soc_items": nitems.astype(np.int32), "soc_seen_views": seen_tot, "soc_last_item": last_item.astype(np.int32)}
    if learn:
        vals.update(soc_med_value=value.astype(np.float32), soc_med_seen=seen)
    population.record_decisions(ctx, "pop_social.created", idx, vals,
                                **CREATE_MODEL, extra={"t": int(now), "created": int(made.sum()),
                                                       "topics": [int(t) for t in topics[made]], "media": media_made})
    return u
