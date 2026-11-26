"""Computing an artifact's representation in the shared space from its bytes: vector, chunk vectors and intrinsic
features, for text, image, audio, video and code. Everything is computed by pretrained models and fixed procedures
from the bytes; no metadata enters (there is no argument through which it could).

Layout of a space vector (`DIM` = 384 + 4 + 34):

    [ sqrt(W_TOPIC) * topic (384, unit) | sqrt(W_AFFECT) * affect (4) | sqrt(W_QUANT) * quantity (34, unit or 0) ]

A document's topic block pools its chunk vectors with salience weights (word pieces x exp(SALIENCE x the chunk's
cosine distance from the plain mean)), so one distinctive passage in a long text moves the item vector instead of
being averaged away; the chunk vectors themselves are kept (`item_chunks`).

L2-normalised. topic = MiniLM text topic (other modalities mapped into it, `align`); affect = [valence, negation,
hedging, questions] (`lm` valence, `textual` cues); quantity = `textual.quantity_block`.
"""
from __future__ import annotations

import io
import json
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from .. import media as MD
from ..encoders import models as EM
from ..encoders.text import TEXT, looks_like_html, to_text
from . import align as AL
from . import lm as LM
from . import textual as TX

TOPIC_DIM, AFFECT_DIM = 384, 4
QUANT_DIM = TX.QUANT_DIM
DIM = TOPIC_DIM + AFFECT_DIM + QUANT_DIM
W_TOPIC, W_AFFECT, W_QUANT = 0.78, 0.10, 0.12   # chosen on the DEV fine-grained pairs (workstream report)
SALIENCE = 4.0                 # chunk pooling: weight ~ pieces * exp(SALIENCE * distance from the plain mean)
TOPIC, AFFECT, QUANT = slice(0, TOPIC_DIM), slice(TOPIC_DIM, TOPIC_DIM + AFFECT_DIM), slice(TOPIC_DIM + AFFECT_DIM, DIM)
MODALITIES = ("text", "image", "audio", "video", "code")
SR = 16000
SPEECH_FLOOR = 0.15            # below this AudioSet speech posterior nothing is transcribed
AUDIO_MAX_S = 600.0
VIDEO_WEIGHTS = {"visual": 0.5, "audio": 0.3, "subtitles": 0.2}
MAX_SOURCE_CHARS = 60_000      # code text read by the LM (naturalness)


@dataclass
class Rep:
    modality: str
    vec: np.ndarray
    chunks: np.ndarray
    features: dict = field(default_factory=dict)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def assemble(topic: np.ndarray, affect: np.ndarray, quant: np.ndarray) -> np.ndarray:
    v = np.concatenate([math.sqrt(W_TOPIC) * _unit(topic), math.sqrt(W_AFFECT) * np.asarray(affect, np.float32),
                        math.sqrt(W_QUANT) * np.asarray(quant, np.float32)])
    return _unit(v).astype(np.float32)


# ============================================================================================== detection
def detect(data: bytes) -> str:
    """Modality from the bytes (magic numbers; containers are probed for a video stream)."""
    h = data[:16]
    if h.startswith(b"\x89PNG") or h.startswith(b"\xff\xd8\xff") or h.startswith(b"GIF8") or h.startswith(b"BM") \
            or (h[:4] == b"RIFF" and h[8:12] == b"WEBP"):
        return "image"
    if (h[:4] == b"RIFF" and h[8:12] == b"WAVE") or h.startswith(b"ID3") or h[:2] in (b"\xff\xfb", b"\xff\xf3") \
            or h.startswith(b"fLaC"):
        return "audio"
    if h[4:8] == b"ftyp" or h.startswith(b"\x1aE\xdf\xa3") or h.startswith(b"OggS"):
        try:
            return "video" if MD.probe(data)["has_video"] else "audio"
        except MD.MediaError:
            return "text"
    if h.startswith(b"PK\x03\x04") or h.startswith(b"\x1f\x8b") or (len(data) > 262 and data[257:262] == b"ustar"):
        return "code"
    if h.lstrip()[:1] == b"{":
        try:
            j = json.loads(data[:4_000_000].decode("utf-8"))
            if isinstance(j, dict) and isinstance(j.get("files"), dict):
                return "code"
        except (ValueError, UnicodeDecodeError):
            pass
    return "text"


# ============================================================================================== text
def text_stats(text: str) -> dict:
    words = TX.WORD.findall(text)
    low = [w.lower() for w in words]
    nw = len(words)
    sents = [s for s in TX.SENT.findall(text) if s.strip()]
    vocab = TEXT_VOCAB()
    alpha = [w for w in low if len(w) >= 2]
    valid = sum(1 for w in alpha if w in vocab) / max(1, len(alpha))
    raw = text.encode("utf-8")
    comp = len(zlib.compress(raw, 9)) / max(1, len(raw)) if len(raw) >= 64 else float("nan")
    grams = [tuple(low[i:i + 4]) for i in range(max(0, nw - 3))]
    rep = 1.0 - len(set(grams)) / max(1, len(grams)) if grams else 0.0
    return {"chars": float(len(text)), "words": float(nw), "sentences": float(len(sents)),
            "mean_sentence_words": float(nw / max(1, len(sents))),
            "distinct_word_share": float(len(set(low[:2000])) / max(1, min(nw, 2000))),
            "repeated_4gram_share": float(rep), "language_validity": float(valid),
            "compression_ratio": float(comp), "numbers": float(TX.count_quantities(text)),
            "reading_time_s": float(nw / 238.0 * 60.0)}


_VOCAB: set | None = None


def TEXT_VOCAB() -> set:
    """Whole English words of the MiniLM WordPiece vocabulary (language-validity reference)."""
    global _VOCAB
    if _VOCAB is None:
        tok = EM.load("text").processor
        _VOCAB = {w for w in tok.get_vocab() if w.isalpha() and len(w) >= 2}
    return _VOCAB


def text_reps(texts: list[str]) -> list[Rep]:
    """Representations of already-decoded texts (batched over the chunks of all texts)."""
    tok = EM.load("text").processor
    chs = [TX.chunks(t, tok) for t in texts]
    flat = [c.text for cs in chs for c in cs]
    tv = TEXT.embed_texts(flat) if flat else np.zeros((0, TOPIC_DIM), np.float32)
    lmr = LM.analyse(list(texts))
    out, k = [], 0
    for t, cs, lr in zip(texts, chs, lmr):
        topics = tv[k:k + len(cs)]
        k += len(cs)
        wins = lr["windows"]
        doc_val = float(lr["valence"])
        rows = []
        for c, tp in zip(cs, topics):
            if wins:
                mid = (c.start + c.end) / 2.0
                val = min(wins, key=lambda w: (0 if w[0] <= mid <= w[1] else 1, abs((w[0] + w[1]) / 2 - mid)))[2]
            else:
                val = doc_val
            aff = np.concatenate([[val], TX.cues(c.text)]).astype(np.float32)
            rows.append(assemble(tp, aff, TX.quantity_block(c.text)))
        weights = np.array([max(1, c.n_pieces) for c in cs], dtype=np.float32)
        plain = _unit((weights[:, None] * topics).sum(0))
        weights = weights * np.exp(SALIENCE * (1.0 - topics @ plain))   # distinctive passages are not averaged away
        topic = _unit((weights[:, None] * topics).sum(0))
        affect = np.concatenate([[doc_val], TX.cues(t)]).astype(np.float32)
        vec = assemble(topic, affect, TX.quantity_block(t))
        feats = text_stats(t)
        feats.update({"lm_logprob_per_token": lr["coherence"], "lm_novel_logprob_per_token": lr["coherence_novel"],
                      "lm_novel_token_share": lr["novel_share"], "lm_tokens_scored": float(lr["n_tokens"]),
                      "lm_perplexity": float(math.exp(-lr["coherence"])) if lr["n_tokens"] else float("nan"),
                      "valence": doc_val, "negation": float(affect[1]), "hedging": float(affect[2]),
                      "question_share": float(affect[3]), "chunks": float(len(cs))})
        out.append(Rep("text", vec, np.stack(rows).astype(np.float32), feats))
    return out


def decode_text(data: bytes) -> tuple[str, dict]:
    s = data[:4 * 200_000].decode("utf-8", errors="replace")
    html = 1.0 if looks_like_html(s) else 0.0
    links = float(s.lower().count("<a ")) if html else 0.0
    return to_text(data), {"html": html, "links": links, "bytes": float(len(data))}


# ============================================================================================== image
def image_stats(im) -> dict:
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    g = a.mean(axis=2)
    if max(g.shape) > 512:
        from PIL import Image
        s = 512 / max(g.shape)
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((max(1, int(g.shape[1] * s)),
                                                                     max(1, int(g.shape[0] * s)))), dtype=np.float32)
    hist = np.bincount(np.clip(g, 0, 255).astype(np.int64).ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 8.0)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colorful = float((math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 100)
    lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]) if min(g.shape) > 2 else np.zeros(1)
    gx = np.abs(np.diff(g, axis=1))[:-1, :] if min(g.shape) > 1 else np.zeros(1)
    gy = np.abs(np.diff(g, axis=0))[:, :-1] if min(g.shape) > 1 else np.zeros(1)
    edges = float(np.mean(np.hypot(gx, gy) > 40.0)) if min(g.shape) > 1 else 0.0
    return {"width": float(im.width), "height": float(im.height), "megapixels": float(im.width * im.height / 1e6),
            "aspect_ratio": float(im.width / max(1, im.height)), "entropy": ent, "colorfulness": colorful,
            "sharpness": float(math.log1p(float(lap.var()))), "edge_density": edges,
            "brightness": float(g.mean() / 255.0), "contrast": float(g.std() / 128.0)}


def image_reps(datas: list[bytes]) -> list[Rep]:
    from ..encoders.image import IMAGE
    ims = [MD.open_image(d) for d in datas]
    clip = IMAGE.embed_images(ims)
    topics, tops = AL.image_topics(clip)
    out = []
    for d, im, tp, top in zip(datas, ims, topics, tops):
        vec = assemble(tp, np.zeros(AFFECT_DIM, np.float32), np.zeros(QUANT_DIM, np.float32))
        f = image_stats(im)
        f.update({"bytes": float(len(d)), "top_concept_weight": float(top[0][1]) if top else 0.0})
        out.append(Rep("image", vec, vec[None, :].copy(), f))
    return out


# ============================================================================================== audio
def audio_signal_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"duration_s": 0.0, "rms_dbfs": -120.0, "peak_dbfs": -120.0, "clipping_share": 0.0, "silence_share": 1.0}
    rms = math.sqrt(float(np.mean(x * x)))
    fr = SR // 20
    n = max(1, len(x) // fr)
    frames = x[:n * fr].reshape(n, fr) if len(x) >= fr else x.reshape(1, -1)
    frms = np.sqrt(np.mean(frames ** 2, axis=1))
    return {"duration_s": float(len(x) / SR), "rms_dbfs": float(20 * math.log10(max(rms, 1e-6))),
            "peak_dbfs": float(20 * math.log10(max(float(np.max(np.abs(x))), 1e-6))),
            "clipping_share": float(np.mean(np.abs(x) >= 0.999)),
            "silence_share": float(np.mean(frms < 10 ** (-50 / 20)))}


def _ast(waves: list[np.ndarray]) -> tuple[np.ndarray, list[float], list[float]]:
    """AudioSet posteriors [n, 527] (max over windows), speech and music posteriors."""
    import torch
    import warnings
    from ..encoders.audio import normalise_loudness, windows
    ld = EM.load("audio")
    flat, owner = [], []
    for i, x in enumerate(waves):
        x = np.asarray(x, dtype=np.float32)
        if len(x) < SR // 10:
            x = np.concatenate([x, np.zeros(SR // 10 - len(x), dtype=np.float32)])
        for w in windows(normalise_loudness(x)):
            flat.append(w)
            owner.append(i)
    probs = np.zeros((len(flat), ld.model.config.num_labels), dtype=np.float32)
    for s in range(0, len(flat), 8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = ld.processor(flat[s:s + 8], sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            iv = feats["input_values"].to(ld.device)
            logits = ld.model(input_values=iv).logits
        probs[s:s + len(iv)] = torch.sigmoid(logits).float().cpu().numpy()
    out = np.zeros((len(waves), probs.shape[1]), dtype=np.float32)
    for i in range(len(waves)):
        out[i] = probs[[j for j, o in enumerate(owner) if o == i]].max(axis=0)
    sp, mu = ld.model.config.label2id["Speech"], ld.model.config.label2id["Music"]
    return out, [float(p[sp]) for p in out], [float(p[mu]) for p in out]


def audio_reps_from_waves(waves: list[np.ndarray]) -> list[Rep]:
    from ..encoders.audio import AUDIO
    probs, speech, music = _ast(waves)
    sound, tops = AL.audio_topics(probs)
    need = [i for i, p in enumerate(speech) if p >= SPEECH_FLOOR]
    trans = [""] * len(waves)
    if need:
        for i, t in zip(need, AUDIO.transcribe([np.asarray(waves[i], np.float32) for i in need])):
            trans[i] = t
    tr = {i: r for i, r in zip([i for i in range(len(waves)) if trans[i]],
                                text_reps([t for t in trans if t]))}
    out = []
    for i, x in enumerate(waves):
        p = speech[i]
        f = audio_signal_stats(x)
        f.update({"speech_prob": p, "music_prob": music[i], "transcript_words": 0.0})
        if i in tr:
            r = tr[i]
            topic = _unit(p * r.vec[TOPIC] / math.sqrt(W_TOPIC) + (1 - p) * sound[i])
            affect = p * r.vec[AFFECT] / math.sqrt(W_AFFECT)
            quant = r.vec[QUANT] / math.sqrt(W_QUANT)
            f.update({"transcript_words": r.features["words"],
                      "transcript_lm_logprob_per_token": r.features["lm_logprob_per_token"],
                      "valence": float(affect[0])})
        else:
            topic, affect, quant = sound[i], np.zeros(AFFECT_DIM, np.float32), np.zeros(QUANT_DIM, np.float32)
        vec = assemble(topic, affect, quant)
        out.append(Rep("audio", vec, vec[None, :].copy(), f))
    return out


def audio_reps(datas: list[bytes]) -> list[Rep]:
    waves = [MD.decode_audio(d, SR, max_s=AUDIO_MAX_S) for d in datas]
    reps = audio_reps_from_waves(waves)
    for r, d in zip(reps, datas):
        r.features["bytes"] = float(len(d))
    return reps


# ============================================================================================== video
def video_rep(data: bytes) -> Rep:
    from ..encoders.image import IMAGE
    from ..encoders.video import MAX_S, N_FRAMES, temporal_features
    info = MD.probe(data)
    if not info["has_video"]:
        raise MD.MediaError("not a video (no video stream)")
    duration = min(float(info["duration_s"] or 0.0), MAX_S)
    frames = MD.extract_frames(data, N_FRAMES, max_side=448, info=info)
    fv = IMAGE.embed_images(frames)
    ftop, _ = AL.image_topics(fv)
    visual = _unit(ftop.mean(0))
    parts = [(VIDEO_WEIGHTS["visual"], visual, None)]
    feats = {"duration_s": duration, "width": float(info["width"] or 0), "height": float(info["height"] or 0),
             "fps": float(info["fps"] or 0.0), "has_audio": 1.0 if info["has_audio"] else 0.0, "bytes": float(len(data)),
             "speech_prob": 0.0, "music_prob": 0.0, "transcript_words": 0.0, "subtitle_words": 0.0}
    affect, quant, wsum = np.zeros(AFFECT_DIM, np.float32), np.zeros(QUANT_DIM, np.float32), 0.0
    if info["has_audio"]:
        wave = MD.decode_audio(data, SR, max_s=AUDIO_MAX_S)
        if len(wave):
            ar = audio_reps_from_waves([wave])[0]
            parts.append((VIDEO_WEIGHTS["audio"], ar.vec[TOPIC], None))
            feats.update({k: ar.features[k] for k in ("speech_prob", "music_prob", "transcript_words")})
            feats.update({"rms_dbfs": ar.features["rms_dbfs"], "silence_share": ar.features["silence_share"]})
            a = ar.vec[AFFECT] / math.sqrt(W_AFFECT)
            if np.any(a):
                affect, wsum = affect + a, wsum + 1.0
            quant = quant + ar.vec[QUANT]
    subs = MD.extract_subtitles(data, info)
    if subs:
        sr = text_reps([subs])[0]
        parts.append((VIDEO_WEIGHTS["subtitles"], sr.vec[TOPIC], None))
        feats["subtitle_words"] = sr.features["words"]
        affect, wsum = affect + sr.vec[AFFECT] / math.sqrt(W_AFFECT), wsum + 1.0
        quant = quant + sr.vec[QUANT]
    tw = sum(w for w, _, _ in parts)
    topic = _unit(sum(w * _unit(t) for w, t, _ in parts) / tw)
    if wsum:
        affect = affect / wsum
    qn = float(np.linalg.norm(quant))
    quant = quant / qn if qn > 1e-9 else quant
    gray = MD.gray_sequence(data, fps=4.0, size=64, max_frames=int(4 * MAX_S))
    tf = temporal_features(gray, duration, fv, int(info["width"] or 1), int(info["height"] or 1), info["has_audio"])
    feats.update({"motion_mean": float(tf[1]), "motion_spread": float(tf[2]), "cut_rate": float(tf[3]),
                  "static_share": float(tf[4]), "frame_change": float(tf[5])})
    vec = assemble(topic, affect, quant)
    segs = np.stack([assemble(t, affect, quant) for t in ftop])
    return Rep("video", vec, segs, feats)


# ============================================================================================== code
def code_reps(datas: list[bytes]) -> list[Rep]:
    import ast as _ast
    from ..encoders.code import facts, read_tree, summary
    trees = [read_tree(d) for d in datas]
    fs = [facts(t) for t in trees]
    reps = text_reps([summary(f) for f in fs])
    srcs = ["\n\n".join(f"# {p}\n{s}" for p, s in sorted(t.items()))[:MAX_SOURCE_CHARS] for t in trees]
    nat = LM.analyse(srcs)
    out = []
    for d, t, f, r, n in zip(datas, trees, fs, reps, nat):
        py = [s for p, s in t.items() if p.endswith(".py")]
        ok = 0
        for s in py:
            try:
                _ast.parse(s)
                ok += 1
            except (SyntaxError, ValueError):
                pass
        lines = max(1, f["lines"])
        feats = {"bytes": float(len(d)), "files": float(len(t)), "lines": float(f["lines"]),
                 "functions": float(f["funcs"]), "classes": float(len(f["classes"])),
                 "tests_present": 1.0 if f["tests"] else 0.0, "manifest_present": 1.0 if f["manifest"] else 0.0,
                 "comment_ratio": float(f["comment_lines"] / lines), "endpoints": float(len(f["routes"])),
                 "imports": float(len(set(f["imports"]))), "python_parse_share": float(ok / len(py)) if py else float("nan"),
                 "branches_per_function": float(f["branches"] / max(1, f["funcs"])),
                 "lm_logprob_per_token": n["coherence"], "lm_novel_token_share": n["novel_share"],
                 "summary_words": r.features["words"]}
        out.append(Rep("code", r.vec, r.chunks, feats))
    return out


# ============================================================================================== dispatch
def reps(datas: list[bytes], modalities: list[str]) -> list[Rep]:
    """Representations of artifacts (grouped and batched by modality, order kept)."""
    out: list[Rep | None] = [None] * len(datas)
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(modalities):
        groups.setdefault(m, []).append(i)
    for m, idx in groups.items():
        ds = [datas[i] for i in idx]
        if m == "text":
            dec = [decode_text(d) for d in ds]
            rs = text_reps([t for t, _ in dec])
            for r, (_, extra) in zip(rs, dec):
                r.features.update(extra)
        elif m == "image":
            rs = image_reps(ds)
        elif m == "audio":
            rs = audio_reps(ds)
        elif m == "video":
            rs = [video_rep(d) for d in ds]
        elif m == "code":
            rs = code_reps(ds)
        else:
            raise ValueError(f"unknown modality {m!r}")
        for i, r in zip(idx, rs):
            r.features["modality"] = m
            out[i] = r
    return out  # type: ignore[return-value]
