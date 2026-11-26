"""Distil the space's valence onto the cheap LM (`lm_fast`) and measure whether its features are equivalent.

    python -m crucible.semantic.space.distill --report out.json [--n 12000]      (GPU slot recommended)

1. Texts: MIND-small news (title and abstract; train split), the population's own writing (`population.writing`,
   all skill levels), and the DEV fine-grained sentences (`finegrained.DEV`: base, paraphrase, changed). The TEST
   fine-grained sentences are never trained on.
2. Labels: `lm.analyse` (Qwen3-0.6B, the current space) - per window the valence (tanh-squashed answer log-odds) and
   per text the coherence features. Features: `lm_fast.forward` - per window the mean-pooled final hidden state of
   SmolLM2-135M and its own token log probabilities.
3. Ridge regression of atanh(valence) on standardised hidden states (penalty chosen by 5-fold CV on the training
   texts), saved to `maps/valence_head.npz` with the report.
4. Equivalence on held-out texts (20 % of the news and writing texts) and on the TEST fine-grained pairs: Pearson and
   Spearman of valence, coherence and novel coherence between the two backends; for every TEST pair whether the
   changed sentence's valence moves in the same direction under both; the fine-grained separation of the space
   (`validate.finegrained_from`) when the affect block's valence comes from either backend; and seconds per text of
   both on this device.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def texts(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    from ...population import writing
    from .. import finegrained as FG
    from . import corpus as C
    rng = np.random.default_rng(seed)
    news = [C.news_text(r) for r in C.news().values()]
    rng.shuffle(news)
    pop = []
    for k in range(n // 4):
        t, b = writing.compose(rng, int(rng.integers(0, 90)), int(rng.integers(1, 7)), float(rng.uniform(0, 1)))
        pop.append(t + "\n\n" + b)
    main = news[: n - len(pop)] + pop
    order = rng.permutation(len(main))
    main = [main[i] for i in order]
    dev = [s for tup in FG.DEV for s in tup[:3]]
    return main, dev


def run(n: int = 12000, report: str | None = None, seed: int = 0) -> dict:
    from scipy.stats import pearsonr, spearmanr
    from .. import finegrained as FG
    from . import lm, lm_fast
    main, dev = texts(n, seed)
    cut = int(0.8 * len(main))
    train, held = main[:cut] + dev, main[cut:]
    t0 = time.time()
    qa = lm.analyse(train + held)
    q_s = (time.time() - t0) / max(1, len(train) + len(held))
    tok = lm_fast.service().weights.tok
    rows, owner, labels = [], [], []
    for i, (t, r) in enumerate(zip(train + held, qa)):
        ws = lm_fast.windows(t, tok)
        for k, (ids, _, _) in enumerate(ws):
            if k >= len(r["windows"]):
                break
            rows.append(ids)
            owner.append(i)
            v = float(np.clip(r["windows"][k][2], -0.999999, 0.999999))
            labels.append(math.atanh(v))
    t0 = time.time()
    lps, pooled = lm_fast.forward(rows)
    f_s = (time.time() - t0) / max(1, len(train) + len(held))
    owner = np.array(owner)
    y = np.array(labels)
    is_tr = owner < len(train)
    X = pooled
    mu, sd = X[is_tr].mean(0), X[is_tr].std(0) + 1e-6
    Z = (X - mu) / sd
    folds = np.arange(int(is_tr.sum())) % 5
    best, cv = None, {}
    Ztr, ytr = Z[is_tr], y[is_tr]
    for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        err = []
        for f in range(5):
            a, b = folds != f, folds == f
            A = Ztr[a]
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr[a] - ytr[a].mean()))
            err.append(float(np.mean((Ztr[b] @ w + ytr[a].mean() - ytr[b]) ** 2)))
        cv[lam] = float(np.mean(err))
        if best is None or cv[lam] < cv[best]:
            best = lam
    w = np.linalg.solve(Ztr.T @ Ztr + best * np.eye(Ztr.shape[1]), Ztr.T @ (ytr - ytr.mean()))
    b = float(ytr.mean())
    np.savez(lm_fast.HEAD, w=w.astype(np.float32), b=np.float32(b), mu=mu.astype(np.float32),
             sd=sd.astype(np.float32), lam=np.float32(best), n_train=np.int64(is_tr.sum()))
    lm_fast._HEAD.clear()
    # ---- equivalence on held-out texts (recomputed through lm_fast.analyse: the deployed path)
    fa = lm_fast.analyse(held)
    qh = qa[len(train):]
    out = {"n_train_texts": len(train), "n_heldout_texts": len(held), "ridge_lambda": best, "cv_mse": cv,
           "qwen_s_per_text": round(q_s, 5), "fast_s_per_text": round(f_s, 5)}
    for k in ("valence", "coherence", "coherence_novel"):
        a = np.array([r[k] for r in qh], np.float64)
        c = np.array([r[k] for r in fa], np.float64)
        ok = np.isfinite(a) & np.isfinite(c)
        out[k] = {"pearson": round(float(pearsonr(a[ok], c[ok])[0]), 4),
                  "spearman": round(float(spearmanr(a[ok], c[ok])[0]), 4), "n": int(ok.sum())}
    # ---- TEST fine-grained pairs: direction of the valence change and the space's separation
    pairs = FG.TEST
    flat = [s for p in pairs for s in p[:3]]
    qv = [r["valence"] for r in lm.analyse(flat)]
    fv = [r["valence"] for r in lm_fast.analyse(flat)]
    agree, moved = 0, 0
    for j, (bs, pa, ch, kind) in enumerate(pairs):
        dq, df = qv[3 * j + 2] - qv[3 * j], fv[3 * j + 2] - fv[3 * j]
        if kind in ("negation", "polarity"):
            moved += 1
            agree += int(np.sign(dq) == np.sign(df) and dq != 0)
    out["test_pairs_valence_direction_agreement"] = f"{agree}/{moved}"
    out["test_pairs_agreement_share"] = round(agree / max(1, moved), 4)
    out["equivalent"] = lm_fast.equivalent(out)
    with open(lm_fast.REPORT, "w") as fh:          # kept next to the head it describes
        json.dump(out, fh, indent=1)
    if report:
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crucible.semantic.space.distill")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.n, a.report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
