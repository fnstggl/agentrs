"""Does the audience respond to meaning or to tokens? Evaluator-side experiments on the one population (MASTER_SPEC
4.3, 4.4; audit C_social p3, D_semantic E.3).

`token_gaming(seed, ...)`: one world of the social internet on `pop` warms up with its own creators; then it is forked
into arms that differ in ONE artifact a scripted experimenter (`p:lab`, labelled scripted in evidence) publishes on the
same platform at the same moment, about the same topic:

* `informative` - natural English: real news abstracts of the topic (MIND-small corpus);
* `stuffed` - keyword stuffing: the topic's characteristic words repeated to the same length;
* `gibberish` - token optimisation: pseudo-words chosen by an adversary WITH the platform's encoder and the
  population's interests, greedily, for the highest cosine between the post's vector and the mean interest.

Each arm runs the same simulated hours; the outcome is the item's exact audience (unique viewers, views, impressions,
opens) from the audience recount. `artifact_change(seed, ...)`: an item is published about one topic and then
`content.update`d to another; the outcome is the shift in the topical make-up of its later viewers (through the
re-indexed representation).
"""
from __future__ import annotations

import numpy as np

import crucible
from ..core.kernel import World
from ..population import semantics, writing
from . import audience as AUD
from .worlds import social_world

DAY = 86400


def _pseudo_words(rng, n: int) -> list[str]:
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiouy"
    out = []
    for _ in range(n):
        L = int(rng.integers(2, 4))
        out.append("".join(cons[int(rng.integers(0, len(cons)))] + vow[int(rng.integers(0, len(vow)))]
                           for _ in range(L)) + cons[int(rng.integers(0, len(cons)))])
    return out


def adversarial_gibberish(world, target: np.ndarray, n_words: int, rng, *, pool: int = 240) -> str:
    """Greedy token optimisation against the platform's encoder: add the pseudo-word that most raises the affinity
    the platform computes - the cosine of the post's topic block (crucible.semantic.space) with `target` (a unit
    space vector, here a topic centroid) - `n_words` times (the adversary runs the public encoder itself; nothing is
    recorded in the world)."""
    cand = _pseudo_words(rng, pool)
    V = semantics.topic(semantics.text_vectors(world, cand))
    order = np.argsort(-(V @ target))[:40]
    words: list[str] = []
    best = [cand[int(j)] for j in order[:8]]
    while len(words) < n_words:
        trials = [" ".join(words + [b]) for b in best]
        tv = semantics.topic(semantics.text_vectors(world, trials))
        words.append(best[int(np.argmax(tv @ target))])
    return " ".join(words)


TOPICS = ("football_nfl", "recipes", "newspolitics", "finance-real-estate", "movies-celebrity", "weathertopstories",
          "travelarticle", "baseball_mlb")


def arm_texts(world, topic: str, seed: int) -> dict[str, tuple[str, str]]:
    names = semantics.topic_names()
    t = names.index(topic) if topic in names else int(np.argmax(semantics.popularity()))  # not-a-content-score: real MIND topic shares (fallback topic), not an item score
    rng = np.random.default_rng(seed)
    title, body = writing.compose(rng, t, 5, 1.0)
    n_words = len(body.split())
    words = semantics.topic_words(t)
    stuffed = " ".join(words[j % len(words)] for j in range(n_words))
    target = semantics.asset()["centroids"][t]
    gib = adversarial_gibberish(world, target, min(n_words, 40), np.random.default_rng(seed + 1))
    reps = max(1, n_words // max(1, len(gib.split())))
    return {"informative": (title, body), "stuffed": (" ".join(words[:5]).capitalize(), stuffed),
            "gibberish": (" ".join(gib.split()[:5]).capitalize(), " ".join([gib] * reps))}


def _publish(w, lab: str, title: str, body: str, platform: str = "posts") -> tuple[str, int]:
    """The experimenter's account writes and publishes one text post (ordinary tools: studio.write,
    <platform>.publish)."""
    a = w.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": title, "body": body}})
    assert a.ok, a.error
    r = w.execute(lab, {"tool": f"{platform}.publish", "args": {"ref": a.result["ref"], "title": title}})
    assert r.ok, r.error
    cid = r.result["id"]
    return cid, int(w.state.tables["content"][cid]["aud_item"])


def token_gaming(seed: int, *, n: int = 1500, warm_h: int = 24, run_h: int = 72, topics=TOPICS,
                 platforms=("posts", "blog"), platform: str = "blog", creators_prior: float = -1.2) -> dict:
    """Arms differ only in the kind of text `len(topics)` experimenter accounts publish (one post each, the same
    topics in every arm, at the same moment); outcome: the posts' exact audience after `run_h` hours."""
    crucible.load_all_modules()
    spec = social_world("development", seed, n=n, platforms=list(platforms), creators={"prior": creators_prior})
    base = World.create(spec)
    labs = [base.system_call("admin.join", {"handle": f"lab{j}", "endowment": 10_000, "password": f"pw-lab{j}"}
                             )["principal"] for j in range(len(topics))]
    base.advance_time(base.now + warm_h * 3600)
    texts = [arm_texts(base, t, seed * 100 + j) for j, t in enumerate(topics)]
    out = {"seed": seed, "n": n, "topics": list(topics), "arms": {},
           "scripted": "experimenter accounts p:lab<j> publish one post each (the arms' only difference)"}
    for arm in ("informative", "stuffed", "gibberish"):
        w = base.fork(f"tg-{seed}-{arm}")
        items = [_publish(w, lab, *texts[j][arm], platform=platform)[1] for j, lab in enumerate(labs)]
        w.advance_time(w.now + run_h * 3600)
        aud = [AUD.audience(w, i) for i in items]
        tot = {q: int(sum(a[q] for a in aud)) for q in ("unique_viewers", "views", "impressions", "unique_exposed",
                                                        "shares", "subscribers")}
        tot["view_rate"] = tot["unique_viewers"] / max(1, tot["unique_exposed"])
        tot["per_post_unique_viewers"] = [int(a["unique_viewers"]) for a in aud]
        tot["legibility"] = [round(float(AUD.item_col(w, "legib")[i]), 3) for i in items]
        out["arms"][arm] = tot
    return out


def artifact_change(seed: int, *, n: int = 1500, warm_h: int = 24, run_h: int = 24, a_topic: str = "football_nfl",
                    b_topic: str = "recipes", posts: int = 8) -> dict:
    """`posts` experimenter accounts publish about `a_topic`, then update the bytes to `b_topic` (same titles): who
    views the posts before and after the change? (The re-indexed representation should move the audience.)"""
    crucible.load_all_modules()
    spec = social_world("development", seed, n=n, platforms=["posts", "blog"], creators={"prior": -1.2})
    w = World.create(spec)
    labs = [w.system_call("admin.join", {"handle": f"lab{j}", "endowment": 10_000, "password": f"pw-lab{j}"}
                          )["principal"] for j in range(posts)]
    w.advance_time(w.now + warm_h * 3600)
    names = semantics.topic_names()
    rng = np.random.default_rng(seed)
    ta, tb = names.index(a_topic), names.index(b_topic)
    pub = []
    for j, lab in enumerate(labs):
        _, body_a = writing.compose(rng, ta, 5, 1.0)
        pub.append((lab,) + _publish(w, lab, f"Notes for this week {j}", body_a, platform="blog"))
    w.advance_time(w.now + run_h * 3600)
    from ..population import api as P
    ca, cb = semantics.asset()["centroids"][ta], semantics.asset()["centroids"][tb]
    before = {item: AUD.viewers_of(w, item) for _, _, item in pub}
    for j, (lab, cid, item) in enumerate(pub):
        _, body_b = writing.compose(rng, tb, 5, 1.0)
        a = w.execute(lab, {"tool": "studio.write", "args": {"format": "text", "title": f"Notes for this week {j}",
                                                              "body": body_b}})
        r = w.execute(lab, {"tool": "content.update", "args": {"id": cid, "ref": a.result["ref"]}})
        assert r.ok, r.error
    w.advance_time(w.now + run_h * 3600)
    v1 = np.unique(np.concatenate([before[i] for _, _, i in pub]))
    v2 = np.unique(np.concatenate([np.setdiff1d(AUD.viewers_of(w, i), before[i]) for _, _, i in pub]))

    def lean(v):
        if v.size == 0:
            return float("nan")
        u = P.interests(w, v)
        return float(np.mean(u @ cb - u @ ca))
    return {"seed": seed, "posts": posts, "viewers_before": int(v1.size), "viewers_after": int(v2.size),
            "lean_to_b_before": lean(v1), "lean_to_b_after": lean(v2), "population_lean": lean(np.arange(n)),
            "scripted": "experimenter accounts publish and update the posts (the only difference is their bytes)"}
