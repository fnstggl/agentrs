"""The recommendation mechanisms (workstream C2). See the package docstring for the list.

Each class has `name`, `version`, `DEFAULTS` (its parameters) and `rank(world, platform, viewers, candidates, k,
rng)`. Most compute a [viewers, candidates] score matrix (`scores`, -inf = not eligible) and take the row-wise top
k; diversity-aware ranking selects greedily instead. Learned recommenders also have `fit(ctx, platform)`, run by
the `feed` module's training process; they read their fitted parameters from world state (`training.model`).
"""
from __future__ import annotations

import math

import numpy as np

from .. import audience as AUD
from .. import textrep
from ..vocab import TOPICS, TOPIC_NAMES
from ...substrate import columns
from . import features as F
from . import register, training


def _sha(*arrays) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Base:
    name = ""
    version = "1"
    DEFAULTS: dict = {}
    learned = False

    def __init__(self, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.p = {**self.DEFAULTS, **params}

    def _exclude_seen(self, world, viewers, cand, s):
        if self.p.get("exclude_seen"):
            s[F.membership(world, "vw", cand, viewers)] = -np.inf
        return s

    def rank(self, world, platform, viewers, candidates, k, rng) -> np.ndarray:
        """Row-wise top-k of `self.scores` (every subclass but Diversity defines it), in viewer chunks."""
        viewers = np.asarray(viewers, dtype=np.int64)
        cand = np.unique(np.asarray(candidates, dtype=np.int64))
        out = []
        step = int(getattr(self, "CHUNK", F.CHUNK))
        for s in range(0, viewers.size, step):
            v = viewers[s:s + step]
            out.append(F.topk(self.scores(world, platform, v, cand, rng), cand, k))
        return np.vstack(out) if out else np.full((0, k), -1, np.int64)


@register
class Chronological(Base):
    """Following feed: items of creators the viewer follows, newest first, within a time window."""
    name = "chronological"
    DEFAULTS = {"window_s": 7 * 86400, "exclude_seen": False}

    def scores(self, world, platform, viewers, cand, rng):
        fol = F.creator_membership(world, "fo", cand, viewers)
        pub = columns.col(world, AUD.ITEMS, "published_at")[cand].astype(np.float64)
        ok = (world.now - pub) <= self.p["window_s"]
        s = np.where(fol & ok[None, :], pub[None, :], -np.inf)
        return self._exclude_seen(world, viewers, cand, s)


@register
class Interest(Base):
    """Content-based retrieval and ranking: cosine between the viewer's need-shifted interests and the item
    representations computed from the artifacts."""
    name = "interest"
    DEFAULTS = {"need_weight": 0.3, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        u = F.interests(world, viewers, need_weight=self.p["need_weight"])
        s = (u @ F.item_emb(world, cand).T).astype(np.float64)
        return self._exclude_seen(world, viewers, cand, s)


@register
class CollaborativeFiltering(Base):
    """Implicit-feedback matrix factorisation (ALS) fitted on the platform's recorded qualifying views; a viewer's
    vector is folded in from the items it has viewed."""
    name = "cf"
    learned = True
    DEFAULTS = {"factors": 16, "reg": 0.1, "alpha": 8.0, "iters": 6, "max_items": 400, "max_users": 20000,
                "window_s": 7 * 86400, "min_pairs": 50, "exclude_seen": True}

    def fit(self, ctx, platform):
        w = ctx.world
        rows = training.consumption_rows(w, platform, ctx.now - int(self.p["window_s"]))
        ok = training.is_view(w, rows["item"], rows["watch_s"]) if rows["item"].size else np.zeros(0, bool)
        v, it = rows["viewer"][ok], rows["item"][ok]
        if v.size == 0:
            return None
        npop = AUD.n_people(w)
        key = np.unique(it * npop + v)
        it, v = key // npop, key % npop
        if key.size < int(self.p["min_pairs"]):
            return None
        ui, ic = np.unique(it, return_counts=True)
        keep = ui[np.argsort(-ic, kind="stable")[:int(self.p["max_items"])]]
        keep.sort()
        m = np.isin(it, keep)
        it, v = it[m], v[m]
        users = np.unique(v)
        rng = ctx.rng("cf.fit", platform)
        if users.size > int(self.p["max_users"]):
            users = np.sort(rng.choice(users, size=int(self.p["max_users"]), replace=False))
            m = np.isin(v, users)
            it, v = it[m], v[m]
        P = np.zeros((users.size, keep.size), dtype=np.float32)
        P[np.searchsorted(users, v), np.searchsorted(keep, it)] = 1.0
        seed = int(rng.integers(0, 2 ** 31))
        p = self.p

        def run():
            X, Y = training.implicit_als(P, p["factors"], p["reg"], p["alpha"], p["iters"],
                                         np.random.default_rng(seed))
            G = Y.T @ Y + np.eye(Y.shape[1]) * p["reg"]
            return {"items": keep.astype(np.int64), "Y": Y.astype(np.float32),
                    "Ginv": np.linalg.inv(G).astype(np.float32), "pop": P.sum(0).astype(np.float32)}
        return run, {"data_sha256": _sha(P, keep, users), "seed": seed, "pairs": int(P.sum()),
                     "users": int(users.size), "items": int(keep.size),
                     **{k: p[k] for k in ("factors", "reg", "alpha", "iters")}}

    def scores(self, world, platform, viewers, cand, rng):
        mdl = training.model(world, platform, self.name)
        s = np.full((viewers.size, cand.size), -np.inf)
        if mdl is None:
            return s
        mi, Y, Ginv, pop = mdl["items"], mdl["Y"].astype(np.float32), mdl["Ginv"], mdl["pop"]
        B = F.membership(world, "vw", mi, viewers).astype(np.float32)
        U = (B @ Y) * np.float32(1.0 + self.p["alpha"]) @ Ginv
        inm = np.isin(cand, mi)
        pos = np.searchsorted(mi, cand[inm])
        sc = (U @ Y[pos].T).astype(np.float64) + 1e-6 * np.log1p(pop[pos].astype(np.float64))[None, :]
        s[:, inm] = sc
        return self._exclude_seen(world, viewers, cand, s)


def _engagement_features(world, viewers: np.ndarray, items: np.ndarray, *, pairwise: bool) -> np.ndarray:
    """Features of (viewer, item): pairwise=True gives [V, C, d] for all pairs; False gives [n, d] row-wise."""
    pub_views = columns.col(world, AUD.ITEMS, "pub_views")[items].astype(np.float64)
    pub_imp = columns.col(world, AUD.ITEMS, "pub_imp")[items].astype(np.float64)
    pub_comp = columns.col(world, AUD.ITEMS, "pub_completions")[items].astype(np.float64)
    dur = columns.col(world, AUD.ITEMS, "duration_s")[items].astype(np.float64)
    age = np.clip(F.age_s(world, items) / 86400.0, 0, 14)
    item_f = np.stack([np.log(np.maximum(dur, 1) / 60.0), age, np.log1p(pub_views) / 10.0,
                       pub_comp / (pub_views + 5.0), pub_views / (pub_imp + 20.0)], axis=-1)
    if pairwise:
        u = F.interests(world, viewers)
        aff = (u @ F.item_emb(world, items).T).astype(np.float64)
        fol = F.creator_membership(world, "fo", items, viewers).astype(np.float64)
        fam = F.creator_membership(world, "cv", items, viewers).astype(np.float64)
        seen = F.membership(world, "vw", items, viewers).astype(np.float64)
        V, C = aff.shape
        return np.concatenate([np.ones((V, C, 1)), aff[..., None], (aff ** 2)[..., None],
                               np.broadcast_to(item_f[None], (V, C, item_f.shape[1])),
                               fol[..., None], fam[..., None], seen[..., None]], axis=-1)
    u = F.interests(world, viewers)
    aff = np.einsum("nd,nd->n", u, F.item_emb(world, items)).astype(np.float64)
    cre = F.creators_of(world, items)
    fol = np.zeros(items.size)
    fam = np.zeros(items.size)
    seen = np.zeros(items.size)
    from .. import idset
    st = world.state
    fol = F.follows_matrix(world, viewers, cre[:, None])[:, 0].astype(np.float64)
    for c in np.unique(cre):
        m = cre == c
        fam[m] = idset.contains(st, idset.key("cv", int(c)), viewers[m])
    for i in np.unique(items):
        m = items == i
        seen[m] = idset.contains(st, idset.key("vw", int(i)), viewers[m])
    return np.concatenate([np.ones((items.size, 1)), aff[:, None], (aff ** 2)[:, None], item_f,
                           fol[:, None], fam[:, None], seen[:, None]], axis=-1)


@register
class Engagement(Base):
    """Learned engagement prediction: a logistic model of P(qualifying view | viewer, item), refitted online from
    impressions whose outcome has matured (only impressions at least `delay_s` old are used, the label is whether
    the viewer is now a viewer of the item) with `label_noise` of the labels flipped (logging noise)."""
    name = "engagement"
    learned = True
    CHUNK = 2048                                       # [chunk, candidates, features] stays below ~60 MB
    DEFAULTS = {"delay_s": 6 * 3600, "window_s": 3 * 86400, "label_noise": 0.05, "l2": 1e-3, "iters": 12,
                "max_examples": 50000, "min_examples": 200, "exclude_seen": True}

    def fit(self, ctx, platform):
        w = ctx.world
        until = ctx.now - int(self.p["delay_s"])
        rows = training.impression_rows(w, platform, until - int(self.p["window_s"]), until)
        n = rows["viewer"].size
        if n < int(self.p["min_examples"]):
            return None
        rng = ctx.rng("engagement.fit", platform)
        sel = np.sort(rng.choice(n, size=min(n, int(self.p["max_examples"])), replace=False))
        v, it = rows["viewer"][sel], rows["item"][sel]
        X = _engagement_features(w, v, it, pairwise=False)
        from .. import idset
        y = np.zeros(v.size)
        for i in np.unique(it):
            m = it == i
            y[m] = idset.contains(w.state, idset.key("vw", int(i)), v[m])
        true_pos = float(y.sum())
        flip = rng.random(v.size) < float(self.p["label_noise"])
        y = np.where(flip, 1.0 - y, y)
        prev = training.model(w, platform, self.name)
        w0 = None if prev is None else prev["w"].astype(np.float64)
        p = self.p

        def run():
            return {"w": training.logistic_fit(X, y, p["l2"], p["iters"], w0).astype(np.float64),
                    "n": np.array([v.size], np.int64), "pos": np.array([y.sum()], np.float64)}
        return run, {"data_sha256": _sha(X, y, w0 if w0 is not None else np.zeros(0)), "until": int(until),
                     "newest_example_t": int(rows["t"][sel].max()), "examples": int(v.size),
                     "flipped": int(flip.sum()), "true_positives": true_pos, "delay_s": int(self.p["delay_s"])}

    def scores(self, world, platform, viewers, cand, rng):
        mdl = training.model(world, platform, self.name)
        if mdl is None:
            return np.full((viewers.size, cand.size), -np.inf)
        X = _engagement_features(world, viewers, cand, pairwise=True)
        s = X @ mdl["w"].astype(np.float64)
        return self._exclude_seen(world, viewers, cand, s)


@register
class Explore(Base):
    """Exploration and discovery: batched Thompson sampling. Each item's view rate has a Beta posterior from the
    platform's published (delayed, noisy) analytics; one posterior draw per item is shared by `batch` consecutive
    viewers, and the draws rank the candidates."""
    name = "explore"
    DEFAULTS = {"a0": 1.0, "b0": 30.0, "batch": 256, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        views = np.maximum(columns.col(world, AUD.ITEMS, "pub_views")[cand].astype(np.float64), 0)
        imp = np.maximum(columns.col(world, AUD.ITEMS, "pub_imp")[cand].astype(np.float64), 0)
        a = self.p["a0"] + views
        b = self.p["b0"] + np.maximum(imp - views, 0)
        nb = -(-viewers.size // int(self.p["batch"]))
        draws = rng.beta(a[None, :], b[None, :], size=(nb, cand.size))
        s = np.repeat(draws, int(self.p["batch"]), axis=0)[:viewers.size].copy()
        return self._exclude_seen(world, viewers, cand, s)


@register
class Social(Base):
    """Social-graph distribution: items shared to the viewer by its contacts (inbox) and items its contacts liked
    (activity ring), scored by how many times each reached the viewer."""
    name = "social"
    DEFAULTS = {"inbox_weight": 2.0, "friends_weight": 1.0, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        s = np.zeros((viewers.size, cand.size))
        for colname, wgt in (("feed_inbox", self.p["inbox_weight"]), ("feed_social", self.p["friends_weight"])):
            try:
                ring = columns.col(world, "pop", colname)[viewers].astype(np.int64)
            except Exception:       # the column exists only when the feed module is installed
                continue
            for r in range(ring.shape[1]):
                it = ring[:, r]
                pos = np.searchsorted(cand, it)
                pos_c = np.minimum(pos, cand.size - 1)
                ok = (it >= 0) & (cand[pos_c] == it)
                np.add.at(s, (np.flatnonzero(ok), pos_c[ok]), wgt)
        s[s <= 0] = -np.inf
        return self._exclude_seen(world, viewers, cand, s)


@register
class Diversity(Base):
    """Diversity-aware ranking: maximal marginal relevance (Carbonell & Goldstein 1998). Each next slot maximises
    lam * relevance - (1 - lam) * (max similarity to the items already selected)."""
    name = "diversity"
    DEFAULTS = {"lam": 0.6, "need_weight": 0.3, "exclude_seen": True}

    def rank(self, world, platform, viewers, candidates, k, rng):
        viewers = np.asarray(viewers, dtype=np.int64)
        cand = np.unique(np.asarray(candidates, dtype=np.int64))
        out = []
        E = F.item_emb(world, cand)
        sim = (E @ E.T).astype(np.float64)
        lam = float(self.p["lam"])
        for s0 in range(0, viewers.size, F.CHUNK):
            v = viewers[s0:s0 + F.CHUNK]
            rel = (F.interests(world, v, need_weight=self.p["need_weight"]) @ E.T).astype(np.float64)
            rel = self._exclude_seen(world, v, cand, rel)
            V, C = rel.shape
            sel = np.full((V, k), -1, np.int64)
            maxsim = np.full((V, C), -np.inf)
            avail = np.isfinite(rel)
            rows = np.arange(V)
            for t in range(min(k, C)):
                pen = np.where(np.isfinite(maxsim), maxsim, 0.0)
                mmr = np.where(avail, lam * rel - (1 - lam) * pen, -np.inf)
                j = np.argmax(mmr, axis=1)
                ok = np.isfinite(mmr[rows, j])
                sel[ok, t] = cand[j[ok]]
                avail[rows[ok], j[ok]] = False
                maxsim = np.where(ok[:, None], np.maximum(maxsim, sim[j]), maxsim)
            out.append(sel)
        return np.vstack(out) if out else np.full((0, k), -1, np.int64)


@register
class Reputation(Base):
    """Reputation-sensitive distribution: a creator's reputation is earned from recorded outcomes (smoothed
    completion rate and follow rate of its audience); ranking blends it with the viewer's relevance."""
    name = "reputation"
    DEFAULTS = {"prior": 20.0, "prior_completion": 0.3, "prior_follow": 0.02, "follow_weight": 5.0, "weight": 0.5,
                "need_weight": 0.2, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        cre = F.creators_of(world, cand)
        views = columns.col(world, AUD.CREATORS, "views")[cre].astype(np.float64)
        comp = columns.col(world, AUD.CREATORS, "completions")[cre].astype(np.float64)
        fol = columns.col(world, AUD.CREATORS, "follows")[cre].astype(np.float64)
        pr = self.p["prior"]
        rep = (comp + pr * self.p["prior_completion"]) / (views + pr) + \
            self.p["follow_weight"] * (fol + pr * self.p["prior_follow"]) / (views + pr)
        rz = (rep - rep.mean()) / (rep.std() + 1e-9)
        rel = (F.interests(world, viewers, need_weight=self.p["need_weight"]) @ F.item_emb(world, cand).T
               ).astype(np.float64)
        relz = (rel - rel.mean(axis=1, keepdims=True)) / (rel.std(axis=1, keepdims=True) + 1e-9)
        w = float(self.p["weight"])
        s = w * rz[None, :] + (1 - w) * relz
        return self._exclude_seen(world, viewers, cand, s)


def _bm25_matrix(world, cand: np.ndarray, vocab: list[str], k1: float, b: float) -> np.ndarray:
    """[len(vocab), len(cand)] BM25 term weights of the candidates' text (terms recorded at indexing)."""
    def run():
        widx = {w: j for j, w in enumerate(vocab)}
        tf = np.zeros((len(vocab), cand.size))
        dl = np.zeros(cand.size)
        for c, it in enumerate(cand):
            rec = world.state.tables["aud_item_info"][str(int(it))]
            dl[c] = max(1, rec["words"])
            for t, n in rec["terms"].items():
                j = widx.get(t)
                if j is not None:
                    tf[j, c] = n
        df = (tf > 0).sum(axis=1)
        N = cand.size
        idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        avg = dl.mean() if dl.size else 1.0
        denom = tf + k1 * (1 - b + b * dl[None, :] / avg)
        return idf[:, None] * np.where(tf > 0, tf * (k1 + 1) / np.maximum(denom, 1e-9), 0.0)
    return F.memo(world, "bm25", run, cand, k1, b, "|".join(vocab))


_VOCAB = sorted({w for words in TOPICS.values() for w in words})
_TOPIC_WORD_IDX = [np.array([_VOCAB.index(w) for w in TOPICS[t]]) for t in TOPIC_NAMES]


@register
class Search(Base):
    """Search-driven discovery: each viewer issues a query of `query_terms` words about a subject drawn from its
    current needs (softmax of need-shifted interests against subject vocabularies); candidates are ranked by
    BM25 over their text. Items that match no query word are not retrieved."""
    name = "search"
    DEFAULTS = {"k1": 1.2, "b": 0.75, "query_terms": 3, "tau": 0.08, "need_weight": 0.3, "exclude_seen": True}

    def queries(self, world, viewers, rng) -> np.ndarray:
        u = F.interests(world, viewers, need_weight=self.p["need_weight"]).astype(np.float64)
        logits = u @ textrep.topic_centroids(u.shape[1]).T / self.p["tau"]
        logits -= logits.max(axis=1, keepdims=True)
        pr = np.exp(logits)
        pr /= pr.sum(axis=1, keepdims=True)
        cum = np.cumsum(pr, axis=1)
        topic = (rng.random((viewers.size, 1)) > cum).sum(axis=1)
        topic = np.minimum(topic, len(TOPIC_NAMES) - 1)
        q = int(self.p["query_terms"])
        out = np.empty((viewers.size, q), dtype=np.int64)
        pick = rng.integers(0, 1 << 30, size=(viewers.size, q))
        for t in np.unique(topic):
            m = topic == t
            words = _TOPIC_WORD_IDX[int(t)]
            out[m] = words[pick[m] % words.size]
        return out

    def space_queries(self, world, viewers, rng) -> tuple[list[str], np.ndarray]:
        """In the shared space: each viewer types the query-vocabulary words whose embeddings best match its
        need-shifted interests (population.semantics.query_words). Returns (vocabulary, [V, q] word indices)."""
        from ...population import semantics
        u = F.interests(world, viewers, need_weight=self.p["need_weight"]).astype(np.float32)
        qs = semantics.query_words(u, rng, np.full(viewers.size, 0.15), k=int(self.p["query_terms"]))
        vocab = sorted({w for q in qs for w in q.split()})
        widx = {w: j for j, w in enumerate(vocab)}
        out = np.array([[widx[w] for w in q.split()] for q in qs], dtype=np.int64)
        return vocab, out

    def scores(self, world, platform, viewers, cand, rng):
        if F.space_world(world):
            vocab, q = self.space_queries(world, viewers, rng)
            W = _bm25_matrix(world, cand, vocab, self.p["k1"], self.p["b"])
        else:
            W = _bm25_matrix(world, cand, _VOCAB, self.p["k1"], self.p["b"])
            q = self.queries(world, viewers, rng)
        s = W[q].sum(axis=1)                           # [V, C]
        s[s <= 0] = -np.inf
        return self._exclude_seen(world, viewers, cand, s)


@register
class Fresh(Base):
    """Recency and freshness: exponential decay with item age, multiplied by the item's recent view velocity
    (views in the last analytics interval), the same for every viewer."""
    name = "fresh"
    DEFAULTS = {"half_life_s": 12 * 3600, "velocity_weight": 1.0, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        age = np.maximum(F.age_s(world, cand), 0)
        vel = np.maximum(columns.col(world, AUD.ITEMS, "vel_views")[cand].astype(np.float64), 0)
        g = -math.log(2) * age / self.p["half_life_s"] + np.log1p(self.p["velocity_weight"] * np.log1p(vel))
        s = np.broadcast_to(g[None, :], (viewers.size, cand.size)).copy()
        return self._exclude_seen(world, viewers, cand, s)


@register
class Trending(Base):
    """Global popularity: recent view velocity plus a share of all-time views."""
    name = "trending"
    DEFAULTS = {"views_weight": 0.1, "exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        vel = np.maximum(columns.col(world, AUD.ITEMS, "vel_views")[cand].astype(np.float64), 0)
        views = columns.col(world, AUD.ITEMS, "views")[cand].astype(np.float64)
        g = np.log1p(vel) + self.p["views_weight"] * np.log1p(views)
        s = np.broadcast_to(g[None, :], (viewers.size, cand.size)).copy()
        return self._exclude_seen(world, viewers, cand, s)


@register
class Random(Base):
    """Uniformly random order (control)."""
    name = "random"
    DEFAULTS = {"exclude_seen": True}

    def scores(self, world, platform, viewers, cand, rng):
        s = rng.random((viewers.size, cand.size))
        return self._exclude_seen(world, viewers, cand, s)
