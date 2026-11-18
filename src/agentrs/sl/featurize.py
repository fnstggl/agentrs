"""Parameter-free preprocessing of observations and action labels into hashed ids / numbers (done at collection
time, stored compactly, collated into padded tensors for the network). No vocabulary is shared by hand between
worlds: every symbol and field name is hashed twice, once qualified by the world family (family-specific
meaning) and once bare (meaning shared across worlds when the same symbol appears)."""
from __future__ import annotations

import zlib

import numpy as np
import torch

from agentrs.sl.world import Obs

NB = 1 << 16          # hash buckets (0 = padding)
MAX_TOK, MAX_NUM, MAX_ENT, ENT_F = 64, 48, 112, 16
CELL_F = 12           # image cells: 2x2 RGB summary per grid cell
MAX_CAND, LABEL_IDS = 24, 5


def hid(s: str) -> int:
    return zlib.crc32(s.encode()) % (NB - 1) + 1


def label_ids(family: str, label: str) -> np.ndarray:
    words = [w for w in label.replace("_", " ").replace("/", " ").replace(":", " ").split() if w][:3]
    ids = [hid(f"{family}|a|{label}"), hid(f"a|{label}")] + [hid(f"w|{w.lower()}") for w in words]
    out = np.zeros(LABEL_IDS, np.int32)
    out[:len(ids)] = ids[:LABEL_IDS]
    return out


def featurize(family: str, o: Obs) -> dict:
    toks = o.tokens[:MAX_TOK]
    tok = np.array([[hid(f"{family}|t|{t}"), hid(f"t|{t}")] for t in toks], np.int32).reshape(-1, 2)
    nums = list(o.numbers.items())[:MAX_NUM]
    numk = np.array([[hid(f"{family}|n|{k}"), hid(f"n|{k}")] for k, _ in nums], np.int32).reshape(-1, 2)
    numv = np.array([float(v) for _, v in nums], np.float32)
    ents = o.entities[:MAX_ENT]
    entk = np.array([[hid(f"{family}|e|{t}"), hid(f"e|{t}")] for t, _ in ents], np.int32).reshape(-1, 2)
    entf = np.zeros((len(ents), ENT_F), np.float32)
    for i, (_, f) in enumerate(ents):
        f = np.asarray(f, np.float32)[:ENT_F]
        entf[i, :len(f)] = f
    out = {"tok": tok, "numk": numk, "numv": numv, "entk": entk, "entf": entf, "fam": hid(f"family|{family}")}
    if o.image is not None:
        out["img"] = o.image  # (G, G, CELL_F) uint8, produced by the world adapter
    return out


def _pad(arrs, shape_tail, dtype, maxlen):
    n = len(arrs)
    L = max(1, min(maxlen, max((len(a) for a in arrs), default=1)))
    out = np.zeros((n, L) + shape_tail, dtype)
    mask = np.zeros((n, L), bool)
    for i, a in enumerate(arrs):
        k = min(len(a), L)
        if k:
            out[i, :k] = a[:k]
            mask[i, :k] = True
    return out, mask


def collate(feats: list, device) -> dict:
    """Pad a list of featurized observations into tensors."""
    tok, tm = _pad([f["tok"] for f in feats], (2,), np.int64, MAX_TOK)
    numk, nm = _pad([f["numk"] for f in feats], (2,), np.int64, MAX_NUM)
    numv, _ = _pad([f["numv"] for f in feats], (), np.float32, MAX_NUM)
    entk, em = _pad([f["entk"] for f in feats], (2,), np.int64, MAX_ENT)
    entf, _ = _pad([f["entf"] for f in feats], (ENT_F,), np.float32, MAX_ENT)
    b = {"tok": tok, "tm": tm, "numk": numk, "nm": nm, "numv": numv, "entk": entk, "em": em, "entf": entf,
         "fam": np.array([f["fam"] for f in feats], np.int64)}
    has_img = [("img" in f) for f in feats]
    if any(has_img):
        g = next(f["img"] for f in feats if "img" in f).shape
        img = np.zeros((len(feats),) + g, np.uint8)
        for i, f in enumerate(feats):
            if "img" in f and f["img"].shape == g:
                img[i] = f["img"]
        b["img"], b["imgm"] = img, np.array(has_img, bool)
    return {k: torch.as_tensor(v, device=device) for k, v in b.items()}


def image_cells(rgb: np.ndarray, cell: int = 8) -> np.ndarray:
    """(H, W, 3) uint8 egocentric image -> (H/cell, W/cell, 12) uint8: the 2x2 average colour of each cell."""
    H, W, _ = rgb.shape
    g = rgb[:H - H % cell, :W - W % cell].reshape(H // cell, 2, cell // 2, W // cell, 2, cell // 2, 3)
    return g.mean(axis=(2, 5)).transpose(0, 2, 1, 3, 4).reshape(H // cell, W // cell, 12).astype(np.uint8)
