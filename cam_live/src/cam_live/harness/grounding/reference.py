"""A non-neural reference grounder — a BASELINE FOR COMPARISON ONLY (workstream EC).

No model, no learning: it looks for the words of the target description as rendered text in the screenshot
(normalised cross-correlation of an edge map against the label rendered with the fonts installed on this host,
at several sizes), and, when the description names a plain colour and no label matches, for the largest blob of
that colour. It can only find controls that carry the literal label; icon-only controls, paraphrases and
look-alike distractors are exactly what it cannot do, which is why it is the yardstick the neural grounders are
compared against and never the default. It also produces false positives: a short word can match part of a
different label ("Gear" on "Search"), which the benchmark counts as incorrect actions.
"""
from __future__ import annotations

import glob
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import GroundingError, decode_png, point_result

_STOP = {"the", "a", "an", "button", "link", "icon", "element", "field", "input", "box", "tab", "menu", "item",
         "click", "on", "press", "tap", "select", "open", "labelled", "labeled", "called", "named", "that", "says",
         "with", "text", "label", "of", "to", "in", "for", "control", "checkbox", "option", "at", "which", "is",
         "circle", "square", "dot", "tile", "rectangle", "area", "bar", "row", "small", "large", "big"}
_COLOURS = {"red": (220, 40, 40), "green": (40, 160, 70), "blue": (40, 100, 220), "yellow": (240, 200, 40),
            "orange": (240, 140, 30), "purple": (140, 60, 190), "black": (20, 20, 20), "grey": (128, 128, 128),
            "gray": (128, 128, 128), "pink": (240, 110, 170)}
_FONT_GLOBS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
               "/usr/share/fonts/truetype/ubuntu/Ubuntu*.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")


def candidate_labels(instruction: str) -> List[str]:
    """Strings that may be printed on the target: quoted text first, then the description without filler words."""
    out: List[str] = []
    for q in re.findall(r"[\"'“‘]([^\"'”’]{1,60})[\"'”’]", instruction):
        out.append(q.strip())
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+&.\-]*", instruction) if w.lower() not in _STOP
             and w.lower() not in _COLOURS]
    if words:
        out.append(" ".join(words))
        if len(words) > 1:
            out.extend(sorted(set(words), key=len, reverse=True)[:2])
    seen, uniq = set(), []
    for s in out:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    return uniq[:4]


class ReferenceGrounder:
    """Text-template and colour-blob matching. Implements ``contracts.VisualGrounder``."""

    name = "reference-template"
    revision = "builtin-1"
    baseline = True
    heavy = False
    noncommercial = False

    def __init__(self, sizes: Sequence[int] = (12, 13, 14, 15, 16, 18, 20), min_score: float = 0.68,
                 max_fonts: int = 2):
        self.sizes, self.min_score = tuple(sizes), float(min_score)
        self._fonts: Optional[List[str]] = None
        self._max_fonts = max_fonts
        self.calls = 0
        self.total_latency_s = 0.0

    # ------------------------------------------------------------------------------------------ pieces
    def _font_files(self) -> List[str]:
        if self._fonts is None:
            found: List[str] = []
            for pat in _FONT_GLOBS:
                for f in sorted(glob.glob(pat)):
                    if os.path.isfile(f) and f not in found and "Italic" not in f and "Mono" not in f:
                        found.append(f)
            self._fonts = found[:self._max_fonts]
        return self._fonts

    @staticmethod
    def _edges(gray: Any) -> Any:
        import numpy as np
        g = gray.astype(np.float32)
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        return np.minimum(np.abs(gx) + np.abs(gy), 255.0) / 255.0

    def _templates(self, label: str) -> List[Any]:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        out = []
        for path in self._font_files():
            for size in self.sizes:
                try:
                    font = ImageFont.truetype(path, size)
                except OSError:
                    continue
                l, t, r, b = font.getbbox(label)
                w, h = r - l + 6, b - t + 6
                if w < 8 or h < 6 or w > 600:
                    continue
                img = Image.new("L", (w, h), 255)
                ImageDraw.Draw(img).text((3 - l, 3 - t), label, fill=0, font=font)
                out.append(self._edges(np.asarray(img)))
        return out

    @staticmethod
    def _prepare(image: Any) -> Dict[str, Any]:
        """Everything about the screenshot that is shared by all templates: its spectrum and integral images."""
        import numpy as np
        H, W = image.shape
        shape = (H + 64, W + 640)                        # room for templates up to 64 x 640
        return {"image": image, "shape": shape, "fft": np.fft.rfft2(image, shape),
                "ii": np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1),
                "ii2": np.pad(image * image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)}

    @staticmethod
    def _ncc(prep: Dict[str, Any], templ: Any) -> Tuple[float, int, int]:
        """Best zero-mean normalised cross-correlation of ``templ`` over the image -> (score, x, y of the centre)."""
        import numpy as np
        image, shape = prep["image"], prep["shape"]
        H, W = image.shape
        h, w = templ.shape
        if h > min(H, 64) or w > min(W, 640):
            return 0.0, 0, 0
        t = templ - templ.mean()
        tn = float(np.sqrt((t * t).sum()))
        if tn < 1e-6:
            return 0.0, 0, 0
        corr = np.fft.irfft2(prep["fft"] * np.fft.rfft2(t[::-1, ::-1], shape), shape)[h - 1:H, w - 1:W]
        ii, ii2 = prep["ii"], prep["ii2"]
        s = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
        s2 = ii2[h:, w:] - ii2[:-h, w:] - ii2[h:, :-w] + ii2[:-h, :-w]
        var = np.maximum(s2 - s * s / (h * w), 1e-6)
        score = corr / (np.sqrt(var) * tn)
        score[var < 1e-3] = 0.0
        idx = int(np.argmax(score))
        y, x = divmod(idx, score.shape[1])
        return float(score[y, x]), x + w // 2, y + h // 2

    def _by_text(self, gray_edges: Any, labels: Sequence[str]) -> Optional[Tuple[float, int, int, str]]:
        best: Optional[Tuple[float, int, int, str]] = None
        prep = self._prepare(gray_edges)
        for label in labels:
            for text in dict.fromkeys((label, label.capitalize(), label.upper())):
                for templ in self._templates(text):
                    score, x, y = self._ncc(prep, templ)
                    score *= min(1.0, 0.6 + 0.08 * len(text))        # very short strings match by accident
                    if best is None or score > best[0]:
                        best = (score, x, y, text)
            if best is not None and best[0] >= 0.75:
                break                                    # a clear match for an earlier (more specific) label
        return best

    @staticmethod
    def _by_colour(rgb: Any, instruction: str) -> Optional[Tuple[float, int, int, str]]:
        import numpy as np
        names = [c for c in _COLOURS if re.search(rf"\b{c}\b", instruction.lower())]
        if not names:
            return None
        target = np.array(_COLOURS[names[0]], dtype=np.float32)
        dist = np.abs(rgb.astype(np.float32) - target).sum(-1)
        mask = dist < 110
        if mask.sum() < 30:
            return None
        ys, xs = np.nonzero(mask)
        # densest 64-px cell, then the centroid of the colour inside it and its neighbours
        cell = 64
        hist: Dict[Tuple[int, int], int] = {}
        for cy, cx in zip((ys // cell).tolist(), (xs // cell).tolist()):
            hist[(cy, cx)] = hist.get((cy, cx), 0) + 1
        (cy, cx), n = max(hist.items(), key=lambda kv: kv[1])
        near = (np.abs(ys // cell - cy) <= 1) & (np.abs(xs // cell - cx) <= 1)
        return min(0.9, 0.4 + n / (cell * cell)), int(xs[near].mean()), int(ys[near].mean()), names[0]

    # ------------------------------------------------------------------------------------------ contract
    def ground(self, image_png: bytes, instruction: str) -> Dict[str, Any]:
        import numpy as np
        if not isinstance(instruction, str) or not instruction.strip():
            raise GroundingError("empty target description")
        t0 = time.monotonic()
        image = decode_png(image_png)
        rgb = np.asarray(image)
        gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        found = self._by_text(self._edges(gray), candidate_labels(instruction))
        method = "text-template"
        if found is None or found[0] < self.min_score:
            colour = self._by_colour(rgb, instruction)
            if colour is not None:
                found, method = colour, "colour-blob"
        dt = time.monotonic() - t0
        self.calls += 1
        self.total_latency_s += dt
        if found is None or (method == "text-template" and found[0] < self.min_score):
            raise GroundingError("no label of the description was found in the screenshot"
                                 + (f" (best match {found[0]:.2f} for {found[3]!r})" if found else ""))
        score, x, y, what = found
        out = point_result(x, y, image.width, image.height, score, self.name, self.revision, raw=what, method=method,
                           confidence_kind="normalised cross-correlation" if method == "text-template" else "colour density")
        out["latency_s"] = dt
        return out

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "repo": "", "revision": self.revision, "license": "part of cam_live",
                "architecture": "rendered-text template matching (numpy FFT) + colour blobs; no model",
                "noncommercial": False, "baseline": True, "needs_gpu_lock": False, "trust_remote_code": False,
                "frozen": True, "available": True, "unavailable_reason": "", "loaded": True, "device": "cpu",
                "params": 0, "load_s": 0.0, "gpu_mem_mb": 0.0, "gpu_peak_mb": 0.0, "calls": self.calls,
                "mean_latency_s": self.total_latency_s / self.calls if self.calls else None,
                "note": "baseline for comparison only; finds literal labels and plain colours, nothing else"}

    def close(self) -> None:
        self._fonts = None                               # nothing else is held: no model, no process
