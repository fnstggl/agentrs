"""Cheaper language-model features of text (candidate replacement of `lm`, adopted only with measured equivalence).

One forward pass of a small causal LM (SmolLM2-135M-Instruct, 135M parameters: about a quarter of Qwen3-0.6B, with
a vocabulary a third the size) over each window of the text, with no instruction frame:

* **coherence** exactly as `lm` defines it (mean log-likelihood per token; *novel* excludes tokens completing a
  4-gram already seen in the window), from this model's next-token distribution after a beginning-of-text token;
* **valence** from a linear head on the window's mean-pooled final hidden state, distilled from `lm`'s Qwen3-0.6B
  valence (the answer log-odds of "positive" vs "negative", squashed with tanh(x / 4)) on real news windows, the
  population's own writing and the DEV fine-grained sentences (`distill.py`; the head and its held-out report live
  in `space/maps/valence_head.npz`).

`analyse(texts)` returns the same structure as `lm.analyse`. Selective GPU as in `lm` (`lm.device()`).

Adoption rule. The space switches to this backend only when the distillation report (`maps/valence_head_report.json`,
written by `distill.py` on held-out texts) shows equivalence: valence Pearson >= 0.85, coherence and novel-coherence
Spearman >= 0.9, and the valence change of at least 11 of 12 TEST negation/polarity pairs in Qwen's direction
(`ADOPTION`, `equivalent`). The report of 2026-11-26 (12,054 texts: 9,654 train, 2,400 held out; GPU) gave valence
Pearson 0.752, coherence Spearman 0.887, novel coherence Spearman 0.886, TEST pairs 12/12, 2.6x faster per text on
the GPU (about 7x on one CPU thread): not equivalent, so the space keeps `lm` (Qwen3-0.6B) and this module stays a
measured, unadopted candidate.
"""
from __future__ import annotations

import os
import threading

import numpy as np

MODEL_ID = os.environ.get("CRUCIBLE_SPACE_FAST_LM", "HuggingFaceTB/SmolLM2-135M-Instruct")
WINDOW_TOKENS = 256
MAX_WINDOWS = 8
BATCH = 16
NGRAM = 4
HEAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "valence_head.npz")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "valence_head_report.json")
ADOPTION = {"valence_pearson": 0.85, "coherence_spearman": 0.9, "coherence_novel_spearman": 0.9,
            "test_pairs_agreement_share": 11 / 12}


def equivalent(report: dict) -> bool:
    """Whether a distillation report meets the adoption thresholds (`ADOPTION`)."""
    return (report["valence"]["pearson"] >= ADOPTION["valence_pearson"]
            and report["coherence"]["spearman"] >= ADOPTION["coherence_spearman"]
            and report["coherence_novel"]["spearman"] >= ADOPTION["coherence_novel_spearman"]
            and report["test_pairs_agreement_share"] >= ADOPTION["test_pairs_agreement_share"])

_LOCK = threading.RLock()
_SVC: dict = {}
_HEAD: dict = {}
_STATS = {"windows": 0, "forward_passes": 0, "tokens": 0}


def service():
    from ..service import SemanticService
    from .lm import device
    dev = device()
    with _LOCK:
        s = _SVC.get(dev)
        if s is None:
            s = _SVC[dev] = SemanticService(MODEL_ID, device=dev, max_batch=BATCH)
        return s


def head() -> dict:
    with _LOCK:
        if not _HEAD:
            z = np.load(HEAD)
            _HEAD.update({"w": z["w"].astype(np.float64), "b": float(z["b"]), "mu": z["mu"].astype(np.float64),
                          "sd": z["sd"].astype(np.float64)})
        return _HEAD


def windows(text: str, tok, size: int = WINDOW_TOKENS, max_windows: int = MAX_WINDOWS):
    from .lm import windows as _w
    return _w(text, tok, size, max_windows)


def forward(rows: list[list[int]]) -> tuple[list[np.ndarray], np.ndarray]:
    """Per row (token ids): log probabilities of its tokens after a beginning-of-text token, and the mean-pooled
    final hidden state [n, d] (float64)."""
    import torch
    svc = service()
    w = svc.weights
    tok, model = w.tok, w.model
    bos = tok.bos_token_id if tok.bos_token_id is not None else w.eos_ids[0]
    lps: list = [None] * len(rows)
    pooled = None
    order = np.argsort([len(r) for r in rows], kind="stable")
    for s in range(0, len(order), BATCH):
        idx = [int(j) for j in order[s:s + BATCH]]
        seqs = [[bos] + list(rows[j]) for j in idx]
        L = max(len(x) for x in seqs)
        inp = torch.full((len(seqs), L), w.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            inp[r, :len(x)] = torch.tensor(x)
            att[r, :len(x)] = 1
        with torch.inference_mode():
            out = model(input_ids=inp.to(svc.device), attention_mask=att.to(svc.device), output_hidden_states=True)
            logits = out.logits.float()
            hid = out.hidden_states[-1].float()
        with _LOCK:
            _STATS["forward_passes"] += 1
            _STATS["windows"] += len(idx)
            _STATS["tokens"] += int(att.sum())
        for r, j in enumerate(idx):
            n = len(rows[j])
            lg = logits[r, :n]
            tgt = inp[r, 1:n + 1].to(lg.device)
            lp = (lg.gather(1, tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(lg, dim=-1)).cpu().numpy()
            lps[j] = lp.astype(np.float64)
            hv = hid[r, 1:n + 1].mean(0).cpu().numpy().astype(np.float64)
            if pooled is None:
                pooled = np.zeros((len(rows), hv.shape[0]), np.float64)
            pooled[j] = hv
    return lps, (pooled if pooled is not None else np.zeros((0, 1)))


def valence_raw(pooled: np.ndarray) -> np.ndarray:
    """The head's prediction of Qwen's valence log-odds / VALENCE_SCALE, before tanh."""
    h = head()
    return ((pooled - h["mu"]) / h["sd"]) @ h["w"] + h["b"]


def analyse(texts: list[str]) -> list[dict]:
    svc = service()
    tok = svc.weights.tok
    rows, owner, spans = [], [], []
    for i, t in enumerate(texts):
        ws = windows(t, tok)
        spans.append([(a, b) for _, a, b in ws])
        for ids, _, _ in ws:
            rows.append(ids)
            owner.append(i)
    res = [{"lp": [], "novel": [], "val": []} for _ in texts]
    if rows:
        lps, pooled = forward(rows)
        vals = np.tanh(valence_raw(pooled))
        for j, (ids, lp, v) in enumerate(zip(rows, lps, vals)):
            seen, novel = set(), np.ones(len(ids), dtype=bool)
            for k in range(len(ids)):
                if k >= NGRAM - 1:
                    g = tuple(ids[k - NGRAM + 1:k + 1])
                    if g in seen:
                        novel[k] = False
                    seen.add(g)
            r = res[owner[j]]
            r["lp"].append(lp)
            r["novel"].append(lp[novel])
            r["val"].append(round(float(v), 6))
    out = []
    for i, t in enumerate(texts):
        r = res[i]
        if not r["lp"]:
            out.append({"coherence": float("nan"), "coherence_novel": float("nan"), "novel_share": float("nan"),
                        "n_tokens": 0, "valence": 0.0, "windows": []})
            continue
        lp = np.concatenate(r["lp"])
        nv = np.concatenate(r["novel"])
        wv = [(a, b, v) for (a, b), v in zip(spans[i], r["val"])]
        out.append({"coherence": float(lp.mean()), "coherence_novel": float(nv.mean()) if len(nv) else float(lp.mean()),
                    "novel_share": float(len(nv) / max(1, len(lp))), "n_tokens": int(len(lp)),
                    "valence": float(np.mean([x[2] for x in wv])), "windows": wv})
    return out


def identity() -> dict:
    from ..service import local_revision
    return {"model_id": MODEL_ID, "revision": local_revision(MODEL_ID)}

