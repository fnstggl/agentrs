"""The internet-scale social path (workstream C2; MASTER_SPEC Parts 4.6, 13.6, Gate I).

Kernel module `crowd` (host `crowd`): a columnar population for the social engine, from development size up to the
internet profile (10M+ identities). It exists because workstream B's population module (`pop`) is not merged in this
worktree; it declares the population table `pop` with the column names of INTERFACE_CONTRACTS section 2 (interests,
attention ledger, satisfaction, next_active_at) so the consumption engine reads the same columns from either module.
A world lists `crowd` OR B's `pop`, never both. Everything population-scale goes through `columns.batch`.

What it holds and does:

* identities are rows `0..n-1` (principal `p<i>` only on first need, `ids.ensure_principal`); interests are
  generated in the content representation space (`textrep`) from community-level topic mixtures plus individual
  variation, so neighbours tend to share interests (homophily);
* a **contact graph** that is never stored: `contacts(i)` computes each individual's contacts on demand from a
  counter-based hash of (seed, i, k) - `local` contacts inside the individual's community block and `long`
  contacts anywhere (a small-world, stochastic-block-like directed graph). 10M individuals x 12 contacts cost no
  memory; the graph is part of the world spec (config + seed), so replay, regeneration and forks see the same graph;
* lazy activation: an individual acts only when its `next_active_at` has come (sessions it chose, or a wake-up by a
  share); `register_activity(world, name, handler, owner)` has the signature of B's population API;
* the daily attention reset (`crucible.substrate.attention`);
* **population creators**: a configured set of individuals who write and publish real text artifacts (posts,
  articles) about what they themselves are interested in (topic weights derived from their own interest vector),
  stored with `artifacts.put`, published and indexed with `audience.index_item`. No item carries a quality score.

Fidelity: the interest model, graph model and creator writing model are "plausible, unvalidated" simplifications
(docs/v5/FIDELITY_REGISTER.md); none of them is fitted to data.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..substrate import artifacts, attention, columns, ids
from . import audience as AUD
from . import textrep
from .vocab import FUNCTION_WORDS, TOPIC_NAMES, TOPICS

DAY = 86400
DEFAULTS = {
    "n": 2000, "interest_dim": textrep.DIM, "community_size": 200, "community_topics": 2, "interest_noise": 0.45,
    "personal_topic_weight": 0.35, "local_contacts": 8, "long_contacts": 4, "attention_min": [20.0, 120.0],
    "first_active_spread_s": 86400, "tick_s": 900, "init_chunk": 1_000_000,
    "creators": 40, "posts_per_day": 1.0, "creator_platforms": {"posts": 0.7, "articles": 0.3},
    "publish_interval_s": 3600, "topic_popularity": None, "share_propensity": [0.6, 0.6],
}
POP_COLS_BASE = {
    "satisfaction": ("float32", (), 0.5), "next_active_at": ("int64", (), 2 ** 62), "share_prop": ("float16", (), 1.0),
}


# ---------------------------------------------------------------------------------------------------- hashing
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Counter-based 64-bit hash (vectorised SplitMix64 finaliser)."""
    with np.errstate(over="ignore"):
        z = (np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _hash3(seed: int, i: np.ndarray, k: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        base = splitmix64(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ np.uint64(k * 0x632BE59BD9B4E019 & 0xFFFFFFFFFFFFFFFF))
        return splitmix64(np.asarray(i, dtype=np.uint64) ^ base)


class ContactGraph:
    """Directed contact graph computed on demand (never materialised). Row i's contacts are `local` individuals of
    its community block [b*size, (b+1)*size) and `long` individuals drawn uniformly from the whole population."""

    def __init__(self, n: int, seed: int, size: int, local: int, long: int):
        self.n, self.seed, self.size, self.local, self.long = int(n), int(seed), int(size), int(local), int(long)
        self.degree = self.local + self.long

    def community(self, i: np.ndarray) -> np.ndarray:
        return np.asarray(i, dtype=np.int64) // self.size

    def contacts(self, i: np.ndarray) -> np.ndarray:
        """[len(i), degree] int64 contact indices (a contact may repeat; never the individual itself)."""
        i = np.asarray(i, dtype=np.int64)
        out = np.empty((i.size, self.degree), dtype=np.int64)
        base = (i // self.size) * self.size
        csize = np.minimum(self.size, self.n - base)
        pos = i - base
        for k in range(self.local):
            h = _hash3(self.seed, i, k)
            off = (h % np.maximum(csize - 1, 1).astype(np.uint64)).astype(np.int64) + 1
            out[:, k] = np.where(csize > 1, base + (pos + off) % csize, i)
        for k in range(self.long):
            h = _hash3(self.seed, i, self.local + k)
            j = (h % np.uint64(self.n)).astype(np.int64)
            out[:, self.local + k] = np.where(j == i, (j + 1) % self.n, j)
        return out


# ---------------------------------------------------------------------------------------------------- module
def _mod(world) -> "Crowd":
    m = world.modules.get("crowd")
    if m is None:
        raise ProtocolError("this world has no 'crowd' module")
    return m  # type: ignore[return-value]


def register_activity(world, name: str, handler, *, owner: str) -> None:
    """handler(ctx, idx: np.ndarray, now: int) -> None is called with the cohort that activates (B's signature)."""
    acts = world.runtime.setdefault("crowd.activities", {})
    acts[name] = (owner, handler)


def wake(ctx: Ctx, idx: np.ndarray, at, activity: str = "") -> None:
    """Schedule activations no later than `at` (messages, shares, notifications): next_active_at = min(current, at)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return
    u = np.unique(idx)
    at_a = np.broadcast_to(np.asarray(at, dtype=np.int64), idx.shape)
    best = np.full(u.size, 2 ** 62, np.int64)
    np.minimum.at(best, np.searchsorted(u, idx), at_a)
    columns.batch(ctx, "pop", f"wake.{activity or 'any'}", u, {"next_active_at": best}, mode="min")


def interest_space(world) -> np.ndarray:
    return textrep.topic_centroids(int(_mod(world).cfg["interest_dim"]))


@register_module
class Crowd(Module):
    """Columnar population of the social scale path (see the module docstring)."""
    name = "crowd"
    host = "crowd"
    version = "1"
    depends = ("cols",)

    def __init__(self, config=None):
        super().__init__(config)
        self.cfg = dict(DEFAULTS)
        for k, v in self.config.items():
            if k not in DEFAULTS:
                raise ProtocolError(f"crowd: unknown config key {k!r}")
            self.cfg[k] = v
        n = int(self.cfg["n"])
        if n < 2 or n > 2 ** 31 - 1:
            raise ProtocolError("crowd: n must be in [2, 2**31)")
        self.n = n
        self.dim = int(self.cfg["interest_dim"])
        if self.dim != textrep.DIM:
            raise ProtocolError(f"crowd: interest_dim must equal the content representation dim {textrep.DIM}")

    def install(self, world):
        if any(m["name"] == "pop" for m in world.spec.modules):
            raise ProtocolError("crowd and pop both declare the population table: list one of them")
        cols = {"interests": ("float16", (self.dim,), 0.0), **POP_COLS_BASE, **attention.COLUMNS}
        columns.declare(world, "pop", self.n, cols)
        self.graph = ContactGraph(self.n, world.spec.seed, int(self.cfg["community_size"]),
                                  int(self.cfg["local_contacts"]), int(self.cfg["long_contacts"]))
        world.runtime["crowd.graph"] = self.graph
        world.state.declare_table("crowd_creators")

    def on_load(self, world):
        world.runtime["crowd.graph"] = self.graph

    # ------------------------------------------------------------------ genesis
    def _interests(self, world, idx: np.ndarray) -> np.ndarray:
        """Initial interests (unit vectors in the content representation space) of individuals `idx`."""
        C = textrep.topic_centroids(self.dim)
        T = C.shape[0]
        comm = idx // int(self.cfg["community_size"])
        ncomm = int(comm.max()) + 1 if idx.size else 0
        pop_w = self.cfg["topic_popularity"]
        pw = np.ones(T) / T if pop_w is None else np.asarray([float(pop_w.get(t, 0.0)) for t in TOPIC_NAMES])
        pw = pw / pw.sum()
        crng = world.rng_for("crowd.communities", ncomm)
        k = int(self.cfg["community_topics"])
        ctop = np.stack([crng.choice(T, size=k, replace=False, p=pw) for _ in range(ncomm)]) if ncomm else \
            np.zeros((0, k), np.int64)
        cw = crng.dirichlet(np.ones(k) * 2.0, size=ncomm) if ncomm else np.zeros((0, k))
        rng = world.rng_for("crowd.interests", int(idx[0]) if idx.size else 0, idx.size)
        v = np.einsum("nk,nkd->nd", cw[comm], C[ctop[comm]])
        personal = rng.choice(T, size=idx.size, p=pw)
        v = v + float(self.cfg["personal_topic_weight"]) * C[personal]
        v = v + float(self.cfg["interest_noise"]) * rng.standard_normal((idx.size, self.dim)) / math.sqrt(self.dim)
        return textrep.normalise_rows(v)

    def genesis(self, ctx):
        w = ctx.world
        n = self.n
        chunk = int(self.cfg["init_chunk"])
        lo, hi = (float(x) for x in self.cfg["attention_min"])
        sp_mu, sp_sigma = (float(x) for x in self.cfg["share_propensity"])
        spread = int(self.cfg["first_active_spread_s"])
        for s in range(0, n, chunk):
            idx = np.arange(s, min(n, s + chunk), dtype=np.int64)
            rng = w.rng_for("crowd.init", s)
            vals = {"interests": self._interests(w, idx).astype(np.float16),
                    "satisfaction": np.full(idx.size, 0.5, np.float32),
                    "next_active_at": rng.integers(0, max(1, spread), idx.size).astype(np.int64),
                    "share_prop": (sp_mu * np.exp(sp_sigma * rng.standard_normal(idx.size) - 0.5 * sp_sigma ** 2)
                                   ).astype(np.float16)}
            columns.batch(ctx, "pop", "crowd.init", idx, vals, mode="set", extra={"from": int(s)})
            attention.reset(ctx, idx, budget=rng.uniform(lo, hi, idx.size).astype(np.float32))
        nc = int(self.cfg["creators"])
        if nc:
            crng = w.rng_for("crowd.creators")
            chosen = np.sort(crng.choice(n, size=min(nc, n), replace=False))
            for i in chosen:
                p = ids.ensure_principal(ctx, int(i))
                ctx.emit("crowd.creator", {"index": int(i), "principal": p}, vis="public")
            ctx.schedule(int(self.cfg["publish_interval_s"]), "crowd.publish", {}, prio=40,
                         period=int(self.cfg["publish_interval_s"]), key="crowd.publish")
        ctx.schedule(int(self.cfg["tick_s"]), "crowd.tick", {}, prio=50, period=int(self.cfg["tick_s"]),
                     key="crowd.tick")
        ctx.schedule(DAY, "crowd.daily", {}, prio=10, period=DAY, key="crowd.daily")

    @applier("crowd.creator")
    def ap_creator(self, ev):
        p = ev.payload
        self.world.state.insert("crowd_creators", str(p["index"]), {"principal": p["principal"], "posts": 0})

    @applier("crowd.posted")
    def ap_posted(self, ev):
        p = ev.payload
        rec = self.world.state.tables["crowd_creators"][str(p["index"])]
        self.world.state.patch("crowd_creators", str(p["index"]), posts=rec["posts"] + 1)

    # ------------------------------------------------------------------ processes
    @process("daily")
    def pr_daily(self, ctx):
        n = self.n
        chunk = int(self.cfg["init_chunk"])
        for s in range(0, n, chunk):
            attention.reset(ctx, np.arange(s, min(n, s + chunk), dtype=np.int64))

    @process("tick")
    def pr_tick(self, ctx):
        nxt = columns.col(ctx.world, "pop", "next_active_at")
        active = np.flatnonzero(nxt <= ctx.now).astype(np.int64)
        acts = ctx.world.runtime.get("crowd.activities", {})
        if active.size == 0 or not acts:
            return
        names = sorted(acts)
        if len(names) == 1:
            acts[names[0]][1](ctx, active, ctx.now)
            return
        pick = ctx.rng("crowd.dispatch").integers(0, len(names), active.size)
        for j, nm in enumerate(names):
            sel = active[pick == j]
            if sel.size:
                acts[nm][1](ctx, sel, ctx.now)

    @process("publish")
    def pr_publish(self, ctx):
        recs = sorted(ctx.state.tables["crowd_creators"].items(), key=lambda kv: int(kv[0]))
        if not recs:
            return
        w = ctx.world
        rng = ctx.rng("crowd.publish")
        rate = float(self.cfg["posts_per_day"]) * int(self.cfg["publish_interval_s"]) / DAY
        plats = self.cfg["creator_platforms"]
        pnames = sorted(plats)
        pw = np.array([float(plats[p]) for p in pnames])
        pw = pw / pw.sum()
        aud = AUD._mod(w)
        for rid, rec in recs:
            for _ in range(int(rng.poisson(rate))):
                platform = pnames[int(rng.choice(len(pnames), p=pw))]
                if platform not in aud.platforms:
                    raise ProtocolError(f"crowd: creator platform {platform!r} is not an aud platform")
                publish_post(ctx, int(rid), rec["principal"], platform, rng)


# ---------------------------------------------------------------------------------------------------- writing
# population creators write text: posts and articles (reading time is their duration). Video and audio platforms
# carry media produced by the content tools of workstream C1 and encoded by workstream D.
KIND_OF_PLATFORM = {"posts": "post", "articles": "article"}
LENGTH_WORDS = {"post": (20, 140), "article": (300, 1400)}


def creator_topic_weights(world, i: int, tau: float = 0.08) -> np.ndarray:
    """What an individual writes about: a softmax of its own interests against the topic vocabularies."""
    v = columns.col(world, "pop", "interests")[i].astype(np.float64)
    s = textrep.topic_centroids(v.size) @ v / tau
    s = np.exp(s - s.max())
    return s / s.sum()


def write_text(rng, topic_weights: np.ndarray, n_words: int) -> tuple[str, str, list[str]]:
    """A post: a title and a body written mostly from one topic (Zipf-weighted words), some from a second topic,
    with function words between. Returns (title, body, tags)."""
    T = len(TOPIC_NAMES)
    main = int(rng.choice(T, p=topic_weights))
    second = int(rng.choice(T, p=topic_weights))
    words_main = TOPICS[TOPIC_NAMES[main]]
    words_second = TOPICS[TOPIC_NAMES[second]]
    zipf = 1.0 / np.arange(1, len(words_main) + 1) ** 0.8
    perm_m = rng.permutation(len(words_main))           # each post has its own word ranking
    perm_s = rng.permutation(len(words_second))
    zp = zipf / zipf.sum()
    focus = float(rng.uniform(0.55, 0.95))              # how focused this post is on its main subject
    out = []
    for _ in range(n_words):
        r = rng.random()
        if r < 0.38:
            out.append(FUNCTION_WORDS[int(rng.integers(0, len(FUNCTION_WORDS)))])
        elif r < 0.38 + 0.62 * focus:
            out.append(words_main[perm_m[int(rng.choice(len(zp), p=zp))]])
        else:
            out.append(words_second[perm_s[int(rng.choice(len(zp), p=zp))]])
    title_words = [words_main[perm_m[j]] for j in range(min(4, len(words_main)))]
    title = " ".join(title_words).capitalize()
    body = []
    for s in range(0, len(out), 12):
        body.append(" ".join(out[s:s + 12]).capitalize() + ".")
    return title, " ".join(body), title_words[:3]


def publish_post(ctx: Ctx, i: int, principal: str, platform: str, rng, *, text: str | None = None,
                 title: str | None = None, tags=None) -> int:
    """Individual i writes, stores, publishes and distributes one text artifact on `platform`; returns the item id."""
    kind = KIND_OF_PLATFORM.get(platform)
    if kind is None:
        raise InvalidAction(f"population creators write text; {platform!r} is not a text platform")
    if text is None:
        lo, hi = LENGTH_WORDS[kind]
        n_words = int(rng.integers(lo, hi + 1))
        title, text, tags = write_text(rng, creator_topic_weights(ctx.world, i), n_words)
    data = f"{title}\n\n{text}\n".encode("utf-8")
    ref = artifacts.put(ctx, data, "text/plain", owner=principal, meta={"title": title[:80]}, kind="text")
    artifacts.publish(ctx, ref)
    item = AUD.index_item(ctx, ref, kind, platform, principal, title, tags=tags or ())
    ctx.emit("crowd.posted", {"index": int(i), "item": int(item)}, vis="public")
    return item
