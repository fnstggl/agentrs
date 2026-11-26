"""Interest prior of the shared space from a real topic corpus: MIND-small news reading histories.

Build (`python -m crucible.semantic.space build-prior`, once per space version; GPU slot recommended):

1. every MIND news item (title + abstract, ~65k) is represented through the space's own text path
   (`compute.text_reps`: chunked MiniLM topic, LM valence and cues, quantities) - interests and items live in exactly
   the same space;
2. each MIND *train* user's interest profile is the normalised mean of the vectors of the news in their click
   history (users with at least `MIN_HISTORY` clicks; one profile per user; user ids are not stored);
3. validation on MIND *dev* impressions (users mostly unseen in train): a dev user's profile from their own history
   scores each impression's candidates by cosine similarity; we report the mean per-impression AUC against clicks,
   for the full space and for its topic block alone, and fit a one-feature logistic calibration
   ``P(click) = sigmoid(a + b * cos)`` on half of the dev impressions, evaluated on the other half.

`interest_prior(rng, n)` (exposed as `crucible.semantic.space.interest_prior`) draws profiles with the caller's RNG.
For n beyond the number of profiles, repeated draws are blended with a second random profile (weight `BLEND`) so
that no two individuals share an identical vector.

Licence: MIND is MSR research-licensed (non-commercial); the built file lives under `$CRUCIBLE_DATA`, never in the
repository.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time

import numpy as np

from . import corpus as C

MIN_HISTORY = 3
BLEND = 0.3
FILE = "mind_prior.npz"
_LOCK = threading.RLock()
_LOADED: dict[str, dict] = {}


def prior_dir(version: str) -> str:
    return os.path.join(C.data_root(), "semantic", "space", version)


def prior_path(version: str) -> str:
    return os.path.join(prior_dir(version), FILE)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks of ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _profiles(hist_lists: list[list[str]], index: dict[str, int], vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prof = np.zeros((len(hist_lists), vecs.shape[1]), dtype=np.float32)
    n = np.zeros(len(hist_lists), dtype=np.int32)
    for i, h in enumerate(hist_lists):
        ix = [index[x] for x in h if x in index]
        if ix:
            m = vecs[ix].mean(0)
            prof[i] = m / max(1e-12, float(np.linalg.norm(m)))
            n[i] = len(ix)
    return prof, n


def build(version: str, *, report: str | None = None, batch: int = 512, max_eval_impressions: int = 20000) -> dict:
    from . import DIM, TOPIC, compute
    t0 = time.time()
    news = C.news()
    ids = sorted(news)
    texts = [C.news_text(news[i]) for i in ids]
    vecs = np.zeros((len(ids), DIM), dtype=np.float32)
    lp_novel = np.zeros(len(ids), dtype=np.float64)
    validity = np.zeros(len(ids), dtype=np.float64)
    for s in range(0, len(ids), batch):
        for j, r in enumerate(compute.text_reps(texts[s:s + batch])):
            vecs[s + j] = r.vec
            lp_novel[s + j] = r.features["lm_novel_logprob_per_token"]
            validity[s + j] = r.features["language_validity"]
    ok = np.isfinite(lp_novel)
    norms = {"lm_novel_logprob_mean": round(float(lp_novel[ok].mean()), 6),
             "lm_novel_logprob_sd": round(float(lp_novel[ok].std()), 6),
             "lm_novel_logprob_p05": round(float(np.quantile(lp_novel[ok], 0.05)), 6),
             "language_validity_mean": round(float(validity.mean()), 6),
             "language_validity_p05": round(float(np.quantile(validity, 0.05)), 6), "texts": int(ok.sum())}
    index = {n: i for i, n in enumerate(ids)}
    t_vec = time.time() - t0
    # train users: last history per user
    hist: dict[str, list[str]] = {}
    for user, _, h, _ in C.behaviors("train"):
        if len(h) >= len(hist.get(user, [])):
            hist[user] = h
    users = sorted(u for u, h in hist.items() if len(h) >= MIN_HISTORY)
    prof, nh = _profiles([hist[u] for u in users], index, vecs)
    keep = nh >= MIN_HISTORY
    prof, nh = prof[keep], nh[keep]
    # dev validation
    rows = []
    for k, (user, _, h, imps) in enumerate(C.behaviors("dev")):
        if len(h) < MIN_HISTORY or len(rows) >= max_eval_impressions:
            continue
        lab = np.array([c for _, c in imps])
        if lab.all() or not lab.any():
            continue
        rows.append((h, [n for n, _ in imps], lab))
    dprof, _ = _profiles([r[0] for r in rows], index, vecs)
    topic_vecs = vecs[:, TOPIC] / np.maximum(np.linalg.norm(vecs[:, TOPIC], axis=1, keepdims=True), 1e-12)
    tprof, _ = _profiles([r[0] for r in rows], index, topic_vecs)
    # popularity baseline: clicks per news in train impressions
    pop = np.zeros(len(ids))
    for _, _, _, imps in C.behaviors("train"):
        for n, c in imps:
            if c and n in index:
                pop[index[n]] += 1
    rng = np.random.default_rng(0)
    aucs = {"space": [], "topic_block": [], "popularity": [], "random": []}  # not-a-content-score: name of the click-count baseline in the prior's validation
    xs, ys, half = [], [], []
    for r, (h, cands, lab), p, tp in zip(range(len(rows)), rows, dprof, tprof):
        ix = np.array([index[c] for c in cands if c in index])
        if len(ix) != len(cands):
            continue
        s = vecs[ix] @ p
        for name, sc in (("space", s), ("topic_block", topic_vecs[ix] @ tp), ("popularity", pop[ix]),
                         ("random", rng.random(len(ix)))):
            a = _auc(np.asarray(sc, dtype=np.float64), lab)
            if a is not None:
                aucs[name].append(a)
        xs.append(s)
        ys.append(lab)
        half.append(r % 2)
    xs_a = np.concatenate(xs)
    ys_a = np.concatenate(ys).astype(np.float64)
    hf = np.concatenate([np.full(len(x), hh) for x, hh in zip(xs, half)])
    a, b = _fit_logit(xs_a[hf == 0], ys_a[hf == 0])
    pt = 1 / (1 + np.exp(-(a + b * xs_a[hf == 1])))
    yt = ys_a[hf == 1]
    ll = float(-np.mean(yt * np.log(pt + 1e-12) + (1 - yt) * np.log(1 - pt + 1e-12)))
    base = float(yt.mean())
    ll0 = float(-(base * math.log(base) + (1 - base) * math.log(1 - base)))
    calib = {"intercept": round(a, 6), "slope_cos": round(b, 6), "heldout_logloss": round(ll, 6),
             "heldout_logloss_constant": round(ll0, 6), "heldout_rows": int(len(yt)), "click_rate": round(base, 6),
             "fit": "MIND dev impressions, even rows fit / odd rows evaluated"}
    summary = {name: {"mean_auc": round(float(np.mean(v)), 4), "impressions": len(v)} for name, v in aucs.items()}
    os.makedirs(prior_dir(version), exist_ok=True)
    meta = {"version": version, "users": int(len(prof)), "news": len(ids), "min_history": MIN_HISTORY,
            "click_auc": summary, "calibration": calib, "text_norms": norms, "corpus": "MIND-small train/dev (MSR research licence)",
            "corpus_sha256": {k: _sha(p) for k, p in C.corpus_files().items()}, "vector_seconds": round(t_vec, 1),
            "seconds": round(time.time() - t0, 1)}
    tmp = prior_path(version) + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, profiles=prof.astype(np.float16), history=nh, meta=np.frombuffer(
        json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8))
    os.replace(tmp, prior_path(version))
    meta["sha256"] = _sha(prior_path(version))
    with _LOCK:
        _LOADED.pop(version, None)
    if report:
        with open(report, "w") as fh:
            json.dump(meta, fh, indent=1)
    return meta


def _fit_logit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    a, b = math.log(max(1e-6, y.mean()) / max(1e-6, 1 - y.mean())), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum(y - p), np.sum((y - p) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        step = np.linalg.solve(H, g)
        a, b = a + float(step[0]), b + float(step[1])
        if np.abs(step).max() < 1e-9:
            break
    return a, b


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load(version: str) -> dict:
    """{profiles [U, DIM] float32 unit rows, history [U], meta, sha256} of the built prior for `version`."""
    with _LOCK:
        hit = _LOADED.get(version)
        if hit is not None:
            return hit
        p = prior_path(version)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"the interest prior for space version {version} is not built ({p}); run `PYTHONPATH=src python -m "
                "crucible.scale.slots --cpu 2 --gpu-mb 3000 -- python -m crucible.semantic.space build-prior`")
        z = np.load(p)
        prof = z["profiles"].astype(np.float32)
        prof /= np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-12)
        prof.setflags(write=False)
        out = {"profiles": prof, "history": z["history"], "meta": json.loads(bytes(z["meta"]).decode()),
               "sha256": _sha(p)}
        _LOADED[version] = out
        return out


def draw(version: str, rng: np.random.Generator, n: int) -> tuple[np.ndarray, dict]:
    pr = load(version)
    P = pr["profiles"]
    U = len(P)
    n = int(n)
    if n <= U:
        idx = rng.choice(U, size=n, replace=False)
        return P[idx].copy(), {"profiles_sha256": pr["sha256"], "n": n, "blended": 0}
    idx = rng.choice(U, size=n, replace=True)
    other = rng.integers(0, U, size=n)
    first = np.zeros(n, dtype=bool)
    _, fi = np.unique(idx, return_index=True)
    first[fi] = True
    out = P[idx].copy()
    rep = ~first
    out[rep] = out[rep] + BLEND * P[other[rep]]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32), {"profiles_sha256": pr["sha256"], "n": n, "blended": int(rep.sum())}
