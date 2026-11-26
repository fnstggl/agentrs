"""Audio encoder: Audio Spectrogram Transformer (AudioSet) embedding plus, for speech, a Whisper transcript embedded
with the text encoder.

Preprocessing: decode the first audio stream with FFmpeg to 16 kHz mono float32 (at most 600 s); normalise loudness to
an RMS of -20 dBFS (peak-limited, digital silence left as is) so that gain changes are not content changes. AST reads
10.24 s windows (1,024 fbank frames x 128 mel bins): a clip up to 10.24 s is one window, a longer one is covered by up
to three evenly spaced windows. The AST block is the mean of the windows' L2-normalised pooled outputs ([768]). The
AudioSet "Speech" posterior (max over windows) gates the transcript: at >= 0.5 Whisper-tiny transcribes the first 60 s
(greedy, English), otherwise the transcript is empty (Whisper invents words for music and tones). The transcript block
is the text encoder's vector of the transcript ([384]; the empty transcript has its own fixed vector).

Output: concat(0.8 * AST block, 0.6 * transcript block): unit norm, float32 [1152], so that the cosine similarity of
two clips is 0.64 x (sound similarity) + 0.36 x (spoken-content similarity).
"""
from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np

from .. import media
from . import models as M
from .base import Encoder, block, register
from .text import TEXT

SR = 16000
WINDOW = int(10.24 * SR)
TARGET_RMS = 0.1          # -20 dBFS
W_AST, W_TEXT = 0.8, 0.6


def normalise_loudness(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    if rms < 1e-5:
        return x
    y = x * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def windows(x: np.ndarray) -> list[np.ndarray]:
    if len(x) <= WINDOW:
        return [x]
    k = min(3, int(math.ceil(len(x) / WINDOW)))
    starts = np.linspace(0, len(x) - WINDOW, k).round().astype(int)
    return [x[s:s + WINDOW] for s in starts]


class AudioEncoder(Encoder):
    modality = "audio"
    id = "crucible.audio.ast-audioset+whisper-tiny"
    roles = ("audio", "speech", "text")
    dim = 768 + 384
    batch_size = 8
    speech_gate = 0.5
    preprocessing = {"decode": "ffmpeg mono f32", "sample_rate": SR, "max_s": 600, "loudness": "rms -20 dBFS, peak<=0.99",
                     "ast_window_s": 10.24, "ast_windows": "1 or up to 3 evenly spaced", "ast_block": "mean of l2(pooler)",
                     "speech_label": "AudioSet Speech", "speech_gate": 0.5, "transcript": "whisper-tiny greedy en, first 60 s",
                     "weights": {"ast": W_AST, "transcript": W_TEXT}}

    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        waves = []
        for b in items:
            try:
                waves.append(media.decode_audio(b, SR, max_s=600.0))
            except media.MediaError:
                raise
        return self.embed_waves(waves)

    def embed_waves(self, waves: Sequence[np.ndarray]) -> np.ndarray:
        an = self.analyze(waves)
        tvec = TEXT.embed_texts([a["transcript"] for a in an])
        return np.stack([np.concatenate([block(a["ast"], W_AST), block(t, W_TEXT)])
                         for a, t in zip(an, tvec)]).astype(np.float32)

    def analyze(self, waves: Sequence[np.ndarray], transcribe: bool = True) -> list[dict]:
        """Per clip: AST block [768], speech posterior, transcript, top AudioSet labels."""
        import torch
        ld = M.load("audio")
        prepared = []
        for x in waves:
            x = np.asarray(x, dtype=np.float32)
            if len(x) < SR // 10:
                x = np.concatenate([x, np.zeros(SR // 10 - len(x), dtype=np.float32)])
            prepared.append(normalise_loudness(x))
        flat, owner = [], []
        for i, x in enumerate(prepared):
            for w in windows(x):
                flat.append(w)
                owner.append(i)
        pooled_all = np.zeros((len(flat), 768), dtype=np.float32)
        probs_all = np.zeros((len(flat), ld.model.config.num_labels), dtype=np.float32)
        for s in range(0, len(flat), self.batch_size):
            with warnings.catch_warnings():       # transformers' numpy fbank warns about an empty top mel filter
                warnings.simplefilter("ignore")
                feats = ld.processor(flat[s:s + self.batch_size], sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                iv = feats["input_values"].to(ld.device)
                pooled = ld.model.audio_spectrogram_transformer(input_values=iv).pooler_output
                logits = ld.model.classifier(pooled)
            pooled_all[s:s + len(iv)] = pooled.float().cpu().numpy()
            probs_all[s:s + len(iv)] = torch.sigmoid(logits).float().cpu().numpy()
        speech_id = ld.model.config.label2id["Speech"]
        id2label = ld.model.config.id2label
        out = []
        for i in range(len(prepared)):
            rows = [j for j, o in enumerate(owner) if o == i]
            ast = np.mean([M.l2n(pooled_all[j]) for j in rows], axis=0)
            probs = probs_all[rows].max(axis=0)
            top = np.argsort(-probs)[:5]
            out.append({"ast": ast.astype(np.float32), "speech": float(probs[speech_id]), "transcript": "",
                        "labels": [(id2label[int(k)], round(float(probs[k]), 4)) for k in top]})
        if transcribe:
            need = [i for i, a in enumerate(out) if a["speech"] >= self.speech_gate]
            if need:
                texts = self.transcribe([prepared[i] for i in need])
                for i, t in zip(need, texts):
                    out[i]["transcript"] = t
        return out

    def transcribe(self, waves: Sequence[np.ndarray], max_s: float = 60.0) -> list[str]:
        import torch
        ld = M.load("speech")
        chunks, owner = [], []
        for i, x in enumerate(waves):
            x = x[:int(max_s * SR)]
            for s in range(0, max(1, len(x)), 30 * SR):
                chunks.append(x[s:s + 30 * SR])
                owner.append(i)
        texts = [""] * len(chunks)
        for s in range(0, len(chunks), self.batch_size):
            inp = ld.processor(chunks[s:s + self.batch_size], sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                ids = ld.model.generate(inp["input_features"].to(ld.device), language="en", task="transcribe",
                                        max_new_tokens=160, do_sample=False, num_beams=1)
            for k, t in enumerate(ld.processor.batch_decode(ids, skip_special_tokens=True)):
                texts[s + k] = " ".join(t.split())
        joined = [""] * len(waves)
        for i, t in zip(owner, texts):
            joined[i] = (joined[i] + " " + t).strip()
        return joined


AUDIO = register(AudioEncoder())
