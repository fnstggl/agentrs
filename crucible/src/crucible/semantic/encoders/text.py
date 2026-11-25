"""Text encoder: sentence-transformers/all-MiniLM-L6-v2 run through `transformers` with attention-masked mean pooling.

Input bytes are decoded as UTF-8 (a BOM is dropped, invalid bytes replaced); an HTML document is reduced to its title
and visible text (scripts, styles and markup removed: `media.html_text`). Text is NFC-normalised and whitespace is
collapsed. Long text is split into windows of 254 word pieces (256 with [CLS]/[SEP], the model's training length);
each window is mean-pooled and L2-normalised, and the document vector is the token-count-weighted mean of its windows
(at most 16 windows, i.e. about 4,000 word pieces), L2-normalised. Output: float32 [384].
"""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import numpy as np

from .. import media
from . import models as M
from .base import Encoder, register

MAX_CHARS = 200_000


def looks_like_html(s: str) -> bool:
    head = s[:2048].lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html") or \
        bool(re.search(r"<(body|head|p|div|h[1-6]|article)[\s>]", head))


def to_text(data: bytes) -> str:
    s = data[:MAX_CHARS * 4].decode("utf-8", errors="replace")
    if s.startswith("﻿"):
        s = s[1:]
    if looks_like_html(s):
        title, body = media.html_text(s)
        s = (title + "\n" + body) if title else body
    s = unicodedata.normalize("NFC", s)[:MAX_CHARS]
    return " ".join(s.split())


class TextEncoder(Encoder):
    modality = "text"
    id = "crucible.text.minilm-l6-v2"
    roles = ("text",)
    dim = 384
    window = 254
    max_windows = 16
    batch_size = 64
    preprocessing = {"decode": "utf-8/replace, html->title+visible text", "unicode": "NFC", "whitespace": "collapse",
                     "window_wordpieces": 254, "max_windows": 16, "pooling": "attention-masked mean",
                     "window_norm": "l2", "combine": "token-weighted mean, l2", "max_chars": MAX_CHARS}

    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        return self.embed_texts([to_text(b) for b in items])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """[n, 384] for already-decoded strings (also used for transcripts and code summaries)."""
        import torch
        ld = M.load("text")
        tok = ld.processor
        ids_list = tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
        windows, owner, weight = [], [], []
        for i, ids in enumerate(ids_list):
            chunks = [ids[k:k + self.window] for k in range(0, len(ids), self.window)][:self.max_windows] or [[]]
            for ch in chunks:
                windows.append([tok.cls_token_id] + list(ch) + [tok.sep_token_id])
                owner.append(i)
                weight.append(max(1, len(ch)))
        vecs = np.zeros((len(windows), self.dim), dtype=np.float32)
        order = np.argsort([len(w) for w in windows], kind="stable")      # similar lengths together: less padding
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            L = max(len(windows[j]) for j in idx)
            inp = np.full((len(idx), L), tok.pad_token_id, dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.int64)
            for r, j in enumerate(idx):
                inp[r, :len(windows[j])] = windows[j]
                mask[r, :len(windows[j])] = 1
            with torch.inference_mode():
                ti = torch.from_numpy(inp).to(ld.device)
                tm = torch.from_numpy(mask).to(ld.device)
                out = ld.model(input_ids=ti, attention_mask=tm, token_type_ids=torch.zeros_like(ti)).last_hidden_state
                m = tm.unsqueeze(-1).to(out.dtype)
                pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vecs[idx] = pooled.float().cpu().numpy()
        out = np.zeros((len(ids_list), self.dim), dtype=np.float32)
        for w_i, (o, wt) in enumerate(zip(owner, weight)):
            out[o] += wt * vecs[w_i]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-12)).astype(np.float32)


TEXT = register(TextEncoder())
