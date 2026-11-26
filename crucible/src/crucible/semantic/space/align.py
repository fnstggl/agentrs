"""Alignment of image, audio, video and code representations with the text topic space (MiniLM, 384-d).

The shared space is anchored on the text encoder because most artifacts, every query, need and description, and the
real topic corpus that seeds interests (MIND news) are text. Other modalities are mapped into it:

* **Images** (CLIP ViT-B/32 image features, 512-d). Two maps were fitted and compared by retrieval accuracy
  (`crucible.semantic.space.validate.alignment`, recorded in the workstream report):
  - ``ridge``: a linear ridge map from CLIP's *text* features to MiniLM features, fitted on pairs (CLIP-text(t),
    MiniLM(t)) of real news titles and abstracts (MIND), applied to the image features after removing CLIP's
    modality gap (image mean replaced by text mean);
  - ``concepts``: CLIP zero-shot reading of the image against a vocabulary of concept prompts (ImageNet-1k classes,
    scenes, everyday internet image genres), the top concepts' softmax weights (CLIP's logit scale) averaged over the
    MiniLM vectors of the concept phrases.
  The image topic is a fixed blend of the two (weights chosen on DEV images, `IMAGE_BLEND`).
* **Audio** (AST AudioSet posteriors): the sound topic is the posterior-weighted mean of the MiniLM vectors of the
  527 AudioSet label phrases (``the sound of <label>``); speech is transcribed (Whisper-tiny) and goes through the text
  path. The speech posterior mixes the two softly (no hard gate).
* **Video**: the image map on sampled frames, the audio map on the audio track, embedded subtitle streams through the
  text path.
* **Code**: the code encoder's declaration summary (docstrings, split identifiers, endpoints, imports, comments,
  paths) through the text path.

Everything here is computed from pretrained models and fixed vocabularies; nothing is creator-declared.
"""
from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from ..encoders import models as M

IMAGE_BLEND = {"ridge": 0.75, "concepts": 0.25}     # chosen on DEV by mean MRR (fit.run, workstream report)
CONCEPT_TOPK = 8
CLIP_PROMPT = "a photo of {}."
CONCEPT_PHRASE = "{}"
AUDIO_PHRASE = "the sound of {}"
RIDGE_LAMBDA = 0.05
MAP_DIR = os.path.join(os.path.dirname(__file__), "maps")
RIDGE_FILE = os.path.join(MAP_DIR, "clip_text_to_minilm.npz")

# scenes, everyday objects and internet image genres missing from ImageNet's classes
EXTRA_CONCEPTS = (
    "a city street", "a skyline", "a beach", "a forest", "a mountain landscape", "a lake", "a desert", "a farm field",
    "a garden", "a park", "a river", "snow", "a rainy day", "a sunset", "the night sky", "outer space", "a kitchen",
    "a living room", "a bedroom", "an office", "a classroom", "a hospital", "a laboratory", "a factory", "a warehouse",
    "a shop", "a supermarket", "a restaurant", "a cafe", "a bar", "a hotel room", "an airport", "a train station",
    "a highway", "a parking lot", "a construction site", "a stadium", "a football match", "a basketball game",
    "a tennis match", "a swimming pool", "a gym", "a concert", "a festival", "a wedding", "a protest", "a crowd of people",
    "a meeting", "a press conference", "a courtroom", "a parliament", "a church", "a temple", "a mosque", "a school",
    "a university campus", "a family", "a child", "a baby", "an elderly person", "a doctor", "a nurse", "a police officer",
    "a soldier", "a politician", "a businessman", "a chef", "a farmer", "a worker", "an athlete", "a musician",
    "a portrait of a man", "a portrait of a woman", "a selfie", "a group photo", "a dog", "a cat", "a horse", "a bird",
    "fish", "flowers", "a tree", "fruit", "vegetables", "bread", "a cake", "a pizza", "a burger", "pasta", "a salad",
    "soup", "coffee", "wine", "beer", "a cocktail", "a car", "a truck", "a bus", "a train", "an airplane", "a bicycle",
    "a motorcycle", "a boat", "a ship", "a house", "an apartment building", "a bridge", "a castle", "money", "coins",
    "a credit card", "a smartphone", "a laptop", "a computer screen", "a keyboard", "a television", "a camera",
    "headphones", "a watch", "jewellery", "clothes", "shoes", "a dress", "a suit", "furniture", "a lamp", "a book",
    "a newspaper", "a magazine cover", "a document", "a letter", "a form", "a receipt", "an invoice", "a contract",
    "a chart", "a bar chart", "a line graph", "a pie chart", "a table of numbers", "a spreadsheet", "a map",
    "a diagram", "a flowchart", "a slide presentation", "a poster", "an advertisement", "a sale sign", "a logo",
    "an icon", "a screenshot", "a screenshot of a website", "a screenshot of source code", "a meme", "a cartoon",
    "a comic strip", "an illustration", "a painting", "a drawing", "a sculpture", "a photograph of text",
    "a handwritten note", "a sign", "a warning sign", "a traffic sign", "a flag", "a medal", "a trophy",
    "a video game", "a board game", "toys", "a medicine bottle", "pills", "a syringe", "a hospital bed",
    "an x-ray", "fire", "smoke", "a flood", "a storm", "an earthquake damage", "a car accident", "a weapon",
    "a robot", "solar panels", "a wind turbine", "a power plant", "an oil rig", "a circuit board", "a data centre",
    "geometric shapes", "a red circle", "a blue square", "a green triangle", "colourful abstract art", "a pattern",
    "a black and white photo", "an empty white image", "a blurry photo",
)

_LOCK = threading.RLock()
_CACHE: dict[str, np.ndarray] = {}


def _norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def image_concepts() -> list[str]:
    from torchvision.models import ResNet50_Weights
    names = [c for c in ResNet50_Weights.IMAGENET1K_V1.meta["categories"]]
    seen, out = set(), []
    for c in list(names) + list(EXTRA_CONCEPTS):
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def audio_labels() -> list[str]:
    ld = M.load("audio")
    cfg = ld.model.config
    return [cfg.id2label[i] for i in range(cfg.num_labels)]


def clip_text(texts: list[str], batch: int = 128) -> np.ndarray:
    """CLIP ViT-B/32 text features (projected, L2-normalised) [n, 512]."""
    import torch
    ld = M.load("image")
    tok = ld.processor.tokenizer
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for s in range(0, len(texts), batch):
        b = tok(list(texts[s:s + batch]), padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.inference_mode():
            f = ld.model.get_text_features(input_ids=b["input_ids"].to(ld.device),
                                           attention_mask=b["attention_mask"].to(ld.device))
            if not torch.is_tensor(f):
                f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
            f = torch.nn.functional.normalize(f, dim=-1)
        out[s:s + len(f)] = f.float().cpu().numpy()
    return out


def _cached(name: str, fn) -> np.ndarray:
    with _LOCK:
        v = _CACHE.get(name)
        if v is None:
            v = _CACHE[name] = np.asarray(fn(), dtype=np.float32)
        return v


def concept_tables() -> tuple[np.ndarray, np.ndarray]:
    """(CLIP prompt features [K, 512], MiniLM phrase vectors [K, 384]) of the image concept vocabulary."""
    from ..encoders.text import TEXT
    cs = image_concepts()
    clip = _cached("concepts.clip", lambda: clip_text([CLIP_PROMPT.format(c) for c in cs]))
    mini = _cached("concepts.minilm", lambda: TEXT.embed_texts([CONCEPT_PHRASE.format(c) for c in cs]))
    return clip, mini


def audio_label_table() -> np.ndarray:
    from ..encoders.text import TEXT
    return _cached("audio.minilm", lambda: TEXT.embed_texts([AUDIO_PHRASE.format(l.lower()) for l in audio_labels()]))


# ---------------------------------------------------------------------------------------------- ridge map
def fit_ridge(clip_txt: np.ndarray, mini: np.ndarray, lam: float = RIDGE_LAMBDA) -> dict:
    """Ridge regression MiniLM ~ CLIP-text (both centred), float64 normal equations; returns the map arrays."""
    X = np.asarray(clip_txt, dtype=np.float64)
    Y = np.asarray(mini, dtype=np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]) / X.shape[1]
    W = np.linalg.solve(A, Xc.T @ Yc)
    return {"W": W.astype(np.float32), "mx": mx.astype(np.float32), "my": my.astype(np.float32)}


def save_ridge(maps: dict, image_mean: np.ndarray, meta: dict) -> str:
    os.makedirs(MAP_DIR, exist_ok=True)
    np.savez(RIDGE_FILE, W=maps["W"].astype(np.float16), mx=maps["mx"], my=maps["my"],
             image_mean=np.asarray(image_mean, dtype=np.float32), meta=np.frombuffer(
                 __import__("json").dumps(meta, sort_keys=True).encode(), dtype=np.uint8))
    return file_sha256(RIDGE_FILE)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ridge() -> dict | None:
    """The committed ridge map (None if it has not been fitted yet)."""
    with _LOCK:
        if "ridge" in _CACHE:
            return _CACHE["ridge"]  # type: ignore[return-value]
        if not os.path.exists(RIDGE_FILE):
            return None
        z = np.load(RIDGE_FILE)
        out = {"W": z["W"].astype(np.float32), "mx": z["mx"], "my": z["my"], "image_mean": z["image_mean"],
               "sha256": file_sha256(RIDGE_FILE)}
        _CACHE["ridge"] = out  # type: ignore[assignment]
        return out


def ridge_apply(clip_vecs: np.ndarray, *, image: bool) -> np.ndarray:
    """CLIP features -> MiniLM space via the ridge map; image features first get CLIP's text mean instead of the
    image mean (modality-gap correction). Returns L2-normalised [n, 384]."""
    r = ridge()
    if r is None:
        raise RuntimeError("the CLIP->MiniLM ridge map has not been fitted (python -m crucible.semantic.space fit)")
    X = np.asarray(clip_vecs, dtype=np.float32)
    if image:
        X = _norm_rows(X - r["image_mean"] + r["mx"])
    return _norm_rows((X - r["mx"]) @ r["W"] + r["my"])


# ---------------------------------------------------------------------------------------------- concept map
def concepts_apply(clip_img: np.ndarray, topk: int = CONCEPT_TOPK) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
    """CLIP zero-shot concepts -> MiniLM space. Returns (L2-normalised [n, 384], top concepts per image)."""
    clip, mini = concept_tables()
    names = image_concepts()
    ld = M.load("image")
    scale = float(ld.model.logit_scale.exp().item())
    logits = scale * (np.asarray(clip_img, dtype=np.float32) @ clip.T)
    out = np.zeros((len(logits), mini.shape[1]), dtype=np.float32)
    tops = []
    for i, lg in enumerate(logits):
        idx = np.argsort(-lg)[:topk]
        w = np.exp(lg[idx] - lg[idx].max())
        w = w / w.sum()
        out[i] = (w[:, None] * mini[idx]).sum(0)
        tops.append([(names[int(j)], round(float(x), 4)) for j, x in zip(idx, w)])
    return _norm_rows(out), tops


def image_topics(clip_img: np.ndarray, blend: dict | None = None) -> tuple[np.ndarray, list]:
    """Image topic vectors [n, 384] in the text space (blend of the ridge and concept maps)."""
    blend = blend or IMAGE_BLEND
    c, tops = concepts_apply(clip_img)
    if blend.get("ridge", 0.0) > 0 and ridge() is not None:
        r = ridge_apply(clip_img, image=True)
        v = blend["ridge"] * r + blend.get("concepts", 0.0) * c
    else:
        v = c
    return _norm_rows(v), tops


def audio_topics(probs: np.ndarray, topk: int = 10) -> tuple[np.ndarray, list]:
    """AST AudioSet posteriors [n, 527] -> sound topic vectors [n, 384] (top-k posterior-weighted label phrases)."""
    tab = audio_label_table()
    labels = audio_labels()
    out = np.zeros((len(probs), tab.shape[1]), dtype=np.float32)
    tops = []
    for i, p in enumerate(np.asarray(probs, dtype=np.float32)):
        idx = np.argsort(-p)[:topk]
        w = p[idx] / max(1e-6, float(p[idx].sum()))
        out[i] = (w[:, None] * tab[idx]).sum(0)
        tops.append([(labels[int(j)], round(float(x), 4)) for j, x in zip(idx, p[idx])])
    return _norm_rows(out), tops


def maps_fingerprint() -> dict:
    """What identifies the alignment maps (enters SPACE_VERSION): blend, vocabulary, prompts and the ridge file."""
    vocab = hashlib.sha256("\n".join(EXTRA_CONCEPTS).encode()).hexdigest()[:16]
    return {"blend": IMAGE_BLEND, "topk": CONCEPT_TOPK, "clip_prompt": CLIP_PROMPT, "concept_phrase": CONCEPT_PHRASE,
            "audio_phrase": AUDIO_PHRASE, "extra_vocab": vocab, "imagenet": "torchvision IMAGENET1K_V1 categories",
            "ridge_sha256": file_sha256(RIDGE_FILE) if os.path.exists(RIDGE_FILE) else None}
