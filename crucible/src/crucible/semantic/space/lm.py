"""Language-model features of text, computed once per artifact with the shared small LM (Qwen3-0.6B by default).

One forward pass per text window gives two things:

* **coherence**: the log-likelihood of each token of the window given the tokens before it. The window is placed in
  a fixed frame (``Statement: <window>``), so the first tokens are conditioned on the same neutral prefix for every
  artifact. Besides the mean log probability per token, the *novel* mean excludes tokens that complete a 4-gram
  already seen in the window: a model predicts repeated keyword runs almost perfectly, so stuffing would otherwise
  look fluent; its novel tokens are a word salad and score low. Gibberish scores low on both.
* **valence**: after the window the frame asks whether the statement is positive or negative (good or bad news);
  the log-odds of the answers ``positive`` and ``negative`` at the answer position, squashed with ``tanh(x / 4)``.
  On the DEV fine-grained pairs this separates negation and polarity flips in every pair (probe in the workstream
  report: Qwen3-0.6B ranks the changed sentence lower than its base in 12/12 negation and polarity pairs; SmolLM2-135M
  and -360M did not, and a MiniLM sentiment axis failed on negation), so the smallest effective model is used.

The weights are the shared weights of `crucible.semantic.service` (one copy per process, also used by semantic
participants). Windows are batched (right padding, attention mask). Selective GPU: the model runs on the GPU only
when the slot granted one (`CUDA_VISIBLE_DEVICES` non-empty and CUDA available) or `CRUCIBLE_LM_DEVICE` says so.
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np

MODEL_ID = os.environ.get("CRUCIBLE_SPACE_LM", "Qwen/Qwen3-0.6B")
WINDOW_TOKENS = 256
MAX_WINDOWS = 8
BATCH = 16
FRAME_HEAD = "Statement: "
FRAME_TAIL = ("\nIs this statement good news or bad news, positive or negative? "
              "Answer with one word: positive or negative.")
ANSWERS = ("positive", "negative")
VALENCE_SCALE = 4.0
NGRAM = 4

_LOCK = threading.RLock()
_SVC = {}
_STATS = {"windows": 0, "forward_passes": 0, "tokens": 0}   # run-time counters (private: not a parameter)


def device() -> str:
    want = os.environ.get("CRUCIBLE_LM_DEVICE", "auto")
    if want != "auto":
        return want
    import torch
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return "cuda" if torch.cuda.is_available() and visible not in ("", "-1") else "cpu"


def service():
    """The shared `SemanticService` whose weights compute the features (created on first use)."""
    from ..service import SemanticService
    dev = device()
    with _LOCK:
        s = _SVC.get(dev)
        if s is None:
            s = _SVC[dev] = SemanticService(MODEL_ID, device=dev, max_batch=BATCH)
        return s


def identity() -> dict:
    """Model id, revision and dtype of the feature model (without loading it)."""
    from ..service import local_revision
    dev = device()
    return {"model_id": MODEL_ID, "revision": local_revision(MODEL_ID),
            "dtype": "bfloat16" if dev.startswith("cuda") else "float32", "device_class": dev.split(":")[0]}


def _frame(tok):
    """Token ids of the frame before and after the window (chat template of the model, no thinking)."""
    key = id(tok)
    cached = getattr(_frame, "_c", {})
    if key in cached:
        return cached[key]
    marker = "⁣⁣WINDOW⁣⁣"
    msgs = [{"role": "user", "content": FRAME_HEAD + marker + FRAME_TAIL}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in MODEL_ID.lower():
        kw["enable_thinking"] = False
    txt = tok.apply_chat_template(msgs, **kw)
    head, tail = txt.split(marker)
    h = tok(head, add_special_tokens=False)["input_ids"]
    t = tok(tail, add_special_tokens=False)["input_ids"]
    ans = [tok(a, add_special_tokens=False)["input_ids"][0] for a in ANSWERS]
    cached[key] = (h, t, ans)
    _frame._c = cached
    return cached[key]


def windows(text: str, tok, size: int = WINDOW_TOKENS, max_windows: int = MAX_WINDOWS) -> list[tuple[list[int], int, int]]:
    """(token ids, char start, char end) of at most `max_windows` evenly spread windows of `size` tokens."""
    if not text.strip():
        return []
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    starts = list(range(0, len(ids), size))
    if len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[int(i)] for i in pick]
    return [(ids[s:s + size], offs[s][0], offs[min(len(ids), s + size) - 1][1]) for s in starts]


def analyse(texts: list[str]) -> list[dict]:
    """Per text: {coherence, coherence_novel, novel_share, n_tokens, valence, windows: [(start, end, valence)]}.
    Empty texts get NaN coherence and 0 valence."""
    svc = service()
    w = svc.weights
    tok, model = w.tok, w.model
    head, tail, ans = _frame(tok)
    rows, owner = [], []
    spans: list[list[tuple[int, int]]] = []
    for i, t in enumerate(texts):
        ws = windows(t, tok)
        spans.append([(a, b) for _, a, b in ws])
        for ids, _, _ in ws:
            rows.append(ids)
            owner.append(i)
    results = [{"lp": [], "novel": [], "val": []} for _ in texts]
    if rows:
        import torch
        order = np.argsort([len(r) for r in rows], kind="stable")
        dev = svc.device
        for s in range(0, len(order), BATCH):
            idx = [int(j) for j in order[s:s + BATCH]]
            seqs = [head + rows[j] + tail for j in idx]
            L = max(len(x) for x in seqs)
            inp = torch.full((len(seqs), L), w.pad_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for r, x in enumerate(seqs):
                inp[r, :len(x)] = torch.tensor(x)
                att[r, :len(x)] = 1
            with torch.inference_mode():
                hidden = model.base_model(input_ids=inp.to(dev), attention_mask=att.to(dev)).last_hidden_state
            with _LOCK:
                _STATS["forward_passes"] += 1
                _STATS["windows"] += len(idx)
                _STATS["tokens"] += int(att.sum())
            head_w = model.get_output_embeddings()
            for r, j in enumerate(idx):
                ids = rows[j]
                n0 = len(head)
                last = len(seqs[r]) - 1
                with torch.inference_mode():
                    tgt = torch.tensor(ids, device=hidden.device)
                    lps = []
                    for c in range(0, len(ids), 64):           # logits of 64 positions at a time (vocab ~152k)
                        pos = torch.arange(n0 - 1 + c, n0 - 1 + min(len(ids), c + 64), device=hidden.device)
                        lg = head_w(hidden[r, pos]).float()
                        lps.append((lg.gather(1, tgt[c:c + 64].unsqueeze(1)).squeeze(1) -
                                    torch.logsumexp(lg, dim=-1)).cpu().numpy())
                    lg_last = head_w(hidden[r, last:last + 1]).float()[0]
                lp = np.concatenate(lps).astype(np.float64)
                seen, novel = set(), np.ones(len(ids), dtype=bool)
                for k in range(len(ids)):
                    if k >= NGRAM - 1:
                        g = tuple(ids[k - NGRAM + 1:k + 1])
                        if g in seen:
                            novel[k] = False
                        seen.add(g)
                v = float(lg_last[ans[0]] - lg_last[ans[1]])
                res = results[owner[j]]
                res["lp"].append(lp)
                res["novel"].append(lp[novel])
                res["val"].append((j, v))
    out = []
    for i, t in enumerate(texts):
        res = results[i]
        if not res["lp"]:
            out.append({"coherence": float("nan"), "coherence_novel": float("nan"), "novel_share": float("nan"),
                        "n_tokens": 0, "valence": 0.0, "windows": []})
            continue
        lp = np.concatenate(res["lp"])
        nv = np.concatenate(res["novel"])
        vals = [v for _, v in sorted(res["val"])]
        wv = [(a, b, round(math.tanh(v / VALENCE_SCALE), 6)) for (a, b), v in zip(spans[i], vals)]
        out.append({"coherence": float(lp.mean()), "coherence_novel": float(nv.mean()) if len(nv) else float(lp.mean()),
                    "novel_share": float(len(nv) / max(1, len(lp))), "n_tokens": int(len(lp)),
                    "valence": float(np.mean([x[2] for x in wv])), "windows": wv})
    return out
