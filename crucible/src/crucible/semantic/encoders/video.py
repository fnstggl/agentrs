"""Video encoder: sampled frames through the image encoder, the audio track through the audio encoder (sound and
transcript), and temporal features.

* Frames: FFmpeg samples 8 frames at the midpoints of 8 equal segments of the video (longest side scaled to 448 px).
  Each goes through CLIP (the image encoder's preprocessing). The visual block keeps temporal order: the frames are
  grouped into 4 consecutive segments, each segment's mean embedding is L2-normalised, and the 4 are concatenated
  ([2048]; its cosine is the mean of the per-segment cosines).
* Audio: the first audio stream through the audio encoder ([1152] = AST sound block + Whisper transcript block). A video
  without audio gets the audio encoder's vector of one second of digital silence.
* Temporal features ([8], each in about [0, 1]): log duration, mean and spread of frame-to-frame change and the rate
  of hard cuts (from 4 fps 64x64 grayscale frames), the share of static intervals, the mean CLIP distance between
  consecutive sampled frames, the aspect ratio and whether there is an audio track.

Output: concat(sqrt(0.55) * visual, sqrt(0.35) * audio, sqrt(0.10) * temporal), L2-normalised, float32 [3208].
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .. import media
from . import models as M
from .audio import AUDIO, SR
from .base import Encoder, register
from .image import IMAGE

N_FRAMES, N_SEGMENTS = 8, 4
W_VIS, W_AUD, W_TMP = math.sqrt(0.55), math.sqrt(0.35), math.sqrt(0.10)
MAX_S = 1800.0
CUT_THRESHOLD = 30.0      # mean absolute grey-level change between consecutive 4 fps frames that counts as a cut
STATIC_THRESHOLD = 2.0


def temporal_features(gray: np.ndarray, duration: float, frame_vecs: np.ndarray, width: int, height: int,
                      has_audio: bool) -> np.ndarray:
    if len(gray) >= 2:
        d = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))
    else:
        d = np.zeros(1, dtype=np.float32)
    minutes = max(duration, 1e-3) / 60.0
    cuts = float(np.sum(d > CUT_THRESHOLD))
    sims = np.sum(frame_vecs[1:] * frame_vecs[:-1], axis=1) if len(frame_vecs) > 1 else np.ones(1)
    f = np.array([
        math.log1p(max(duration, 0.0)) / math.log1p(MAX_S),
        min(1.0, float(d.mean()) / 32.0),
        min(1.0, float(d.std()) / 32.0),
        math.tanh(cuts / minutes / 20.0),
        float(np.mean(d < STATIC_THRESHOLD)),
        min(1.0, float(np.mean(1.0 - sims)) * 3.0),
        0.5 + 0.5 * max(-1.0, min(1.0, math.log(max(width, 1) / max(height, 1)) / math.log(4.0))),
        1.0 if has_audio else 0.0,
    ], dtype=np.float32)
    return f


class VideoEncoder(Encoder):
    modality = "video"
    id = "crucible.video.clip-frames+ast+whisper+temporal"
    roles = ("image", "audio", "speech", "text")
    dim = N_SEGMENTS * 512 + 1152 + 8
    batch_size = 4
    preprocessing = {"frames": N_FRAMES, "frame_times": "segment midpoints (one fps-filter pass up to 180 s, seeks beyond)",
                     "frame_max_side": 448,
                     "segments": N_SEGMENTS, "image_preprocessing": IMAGE.preprocessing,
                     "audio_preprocessing": AUDIO.preprocessing, "no_audio": "audio vector of 1 s silence",
                     "motion": {"fps": 4, "size": 64, "cut_threshold": CUT_THRESHOLD,
                                "static_threshold": STATIC_THRESHOLD},
                     "weights": {"visual": 0.55, "audio": 0.35, "temporal": 0.10}, "max_s": MAX_S}

    _silence: np.ndarray | None = None

    def silence_vector(self) -> np.ndarray:
        if self._silence is None:
            self._silence = AUDIO.embed_waves([np.zeros(SR, dtype=np.float32)])[0]
        return self._silence

    def embed(self, items: Sequence[bytes]) -> np.ndarray:
        return np.stack([self.embed_one(b) for b in items]).astype(np.float32)

    def embed_one(self, data: bytes) -> np.ndarray:
        parts = self.analyze(data)
        return parts["vector"]

    def analyze(self, data: bytes) -> dict:
        info = media.probe(data)
        if not info["has_video"]:
            raise media.MediaError("not a video (no video stream)")
        duration = min(float(info["duration_s"] or 0.0), MAX_S)
        if duration <= 0.0 or info["format"] in ("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe"):
            raise media.MediaError("not a video (a still image or no duration)")
        frames = media.extract_frames(data, N_FRAMES, max_side=448, info=info)
        if len(frames) != N_FRAMES:
            raise media.MediaError(f"decoded {len(frames)} of {N_FRAMES} frames")
        fv = IMAGE.embed_images(frames)                                          # [8, 512], unit rows
        per = len(fv) // N_SEGMENTS
        segs = [M.l2n(fv[k * per:(k + 1) * per].mean(axis=0)) for k in range(N_SEGMENTS)]
        visual = np.concatenate(segs) / np.float32(math.sqrt(N_SEGMENTS))
        if info["has_audio"]:
            wave = media.decode_audio(data, SR, max_s=600.0)
            avec = AUDIO.embed_waves([wave])[0] if len(wave) else self.silence_vector()
        else:
            avec = self.silence_vector()
        gray = media.gray_sequence(data, fps=4.0, size=64, max_frames=int(4 * MAX_S))
        tf = temporal_features(gray, duration, fv, int(info["width"] or 1), int(info["height"] or 1), info["has_audio"])
        vec = np.concatenate([W_VIS * visual, W_AUD * M.l2n(avec), W_TMP * tf / np.float32(math.sqrt(len(tf)))])
        return {"vector": M.l2n(vec), "frames": fv, "audio": avec, "temporal": tf, "info": info}


VIDEO = register(VideoEncoder())
