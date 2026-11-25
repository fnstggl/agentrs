"""Media tooling for content creators and for the encoders: render text and HTML to images, process images, synthesise
audio (speech with flite, tones, melodies, noise), assemble images and audio into videos, transcode, probe, and decode
audio and video frames for the encoders.

Everything runs locally with bounded resources:

* FFmpeg and ffprobe run as child processes in a private temporary directory with a wall-clock timeout, CPU-time,
  address-space and file-size rlimits, one thread, and input restricted to the `file`/`pipe` protocols and an
  allow-list of demuxers (no playlists, concat lists or network URLs, so a crafted file cannot make FFmpeg read other
  files or open connections).
* Images are decoded with Pillow under a pixel limit; text and HTML inputs have length limits.
* Outputs are bounded in size (`MediaLimits.max_output_bytes`).

These functions are pure computations on bytes. Kernel worlds call them through the `media` module (`media_tools`
below), which records each outcome with `ctx.external` (or `substrate.inference.recorded` once it exists) so that
replay and regeneration use the recorded bytes, and stores the result as an artifact.
"""
from __future__ import annotations

import html as _html
import html.parser as _hp
import io
import json
import os
import re
import resource
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"

# demuxers FFmpeg may use on untrusted input (no hls/dash/concat/playlists/network formats)
INPUT_FORMATS = ("mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,ogg,mp3,wav,flac,aac,avi,mpegts,gif,image2,png_pipe,"
                 "jpeg_pipe,webp_pipe,bmp_pipe,gif_pipe,flv,asf,s16le,f32le,u8,lavfi")
TOOL_VERSION = "crucible-media/1"


class MediaError(ValueError):
    """A media input could not be processed (bad data, over a limit, tool failure)."""


@dataclass(frozen=True)
class MediaLimits:
    timeout_s: float = 120.0
    cpu_s: int = 240
    mem_mb: int = 3072
    max_output_bytes: int = 32 * 1024 * 1024     # = the artifact size limit (INTERFACE_CONTRACTS 1.1)
    max_input_bytes: int = 64 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_duration_s: float = 1800.0
    max_text_chars: int = 50_000
    threads: int = 1


LIMITS = MediaLimits()


def _tmpdir() -> str:
    d = os.environ.get("CRUCIBLE_MEDIA_TMP") or None
    if d:
        os.makedirs(d, exist_ok=True)
    return tempfile.mkdtemp(prefix="crucible-media-", dir=d)


def _limiter(lim: MediaLimits):
    def pre():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_s, lim.cpu_s + 5))
        mem = lim.mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        fs = lim.max_output_bytes * 4
        resource.setrlimit(resource.RLIMIT_FSIZE, (fs, fs))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return pre


def run_tool(argv: Sequence[str], *, cwd: str, lim: MediaLimits = LIMITS, stdin: bytes | None = None) -> bytes:
    """Run FFmpeg/ffprobe/sox with the limits; returns stdout or raises MediaError with the tool's last error line."""
    try:
        p = subprocess.run(list(argv), cwd=cwd, input=stdin, capture_output=True, timeout=lim.timeout_s,
                           preexec_fn=_limiter(lim),
                           env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "MALLOC_ARENA_MAX": "2"})
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{os.path.basename(argv[0])} exceeded {lim.timeout_s}s") from e
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"{os.path.basename(argv[0])} failed ({p.returncode}): {(err[-1] if err else '')[:300]}")
    if len(p.stdout) > lim.max_output_bytes * 4:
        raise MediaError("tool output over the limit")
    return p.stdout


def _ff(*args: str) -> list[str]:
    return [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(LIMITS.threads),
            "-filter_threads", "1", *args]


def _safe_input(path: str) -> list[str]:
    return ["-protocol_whitelist", "file,pipe", "-format_whitelist", INPUT_FORMATS, "-i", path]


class _Work:
    """A private temporary directory holding the inputs of one tool run."""

    def __init__(self, inputs: dict[str, bytes] | None = None, lim: MediaLimits = LIMITS):
        self.lim = lim
        self.dir = _tmpdir()
        for name, data in (inputs or {}).items():
            if len(data) > lim.max_input_bytes:
                self.close()
                raise MediaError(f"input {name} is {len(data)} bytes (limit {lim.max_input_bytes})")
            with open(os.path.join(self.dir, name), "wb") as fh:
                fh.write(data)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def read(self, name: str) -> bytes:
        p = self.path(name)
        if not os.path.exists(p):
            raise MediaError(f"tool produced no {name}")
        if os.path.getsize(p) > self.lim.max_output_bytes:
            raise MediaError(f"output {os.path.getsize(p)} bytes is over the limit {self.lim.max_output_bytes}")
        with open(p, "rb") as fh:
            return fh.read()

    def run(self, argv: Sequence[str], stdin: bytes | None = None) -> bytes:
        return run_tool(argv, cwd=self.dir, lim=self.lim, stdin=stdin)

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================================== probing and decoding
def probe(data: bytes) -> dict:
    """Container and stream facts of an audio/video/image file (ffprobe)."""
    with _Work({"in.bin": data}) as w:
        out = w.run([FFPROBE, "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
                     "-format_whitelist", INPUT_FORMATS, "-show_format", "-show_streams", "-of", "json", "in.bin"])
    try:
        j = json.loads(out.decode("utf-8", "replace") or "{}")
    except ValueError as e:
        raise MediaError("unreadable probe output") from e
    fmt = j.get("format") or {}
    streams = []
    for s in j.get("streams") or []:
        fps = None
        r = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/0"
        try:
            a, b = r.split("/")
            fps = float(a) / float(b) if float(b) else None
        except (ValueError, ZeroDivisionError):
            fps = None
        streams.append({"type": s.get("codec_type"), "codec": s.get("codec_name"),
                        "width": s.get("width"), "height": s.get("height"), "fps": fps,
                        "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
                        "channels": s.get("channels"),
                        "duration_s": float(s["duration"]) if s.get("duration") not in (None, "N/A") else None,
                        "attached_pic": bool((s.get("disposition") or {}).get("attached_pic"))})
    video = [s for s in streams if s["type"] == "video" and not s["attached_pic"]]
    audio = [s for s in streams if s["type"] == "audio"]
    dur = fmt.get("duration")
    duration = float(dur) if dur not in (None, "N/A") else max([s["duration_s"] or 0.0 for s in streams] or [0.0])
    v = video[0] if video else {}
    return {"format": fmt.get("format_name"), "duration_s": duration, "bytes": len(data), "streams": streams,
            "has_video": bool(video), "has_audio": bool(audio), "width": v.get("width"), "height": v.get("height"),
            "fps": v.get("fps"), "sample_rate": audio[0]["sample_rate"] if audio else None}


def decode_audio(data: bytes, sr: int = 16000, *, max_s: float = 600.0, offset_s: float = 0.0) -> np.ndarray:
    """Decode the first audio stream to mono float32 PCM at `sr` Hz (at most `max_s` seconds)."""
    with _Work({"in.bin": data}) as w:
        out = w.run(_ff(*(["-ss", f"{offset_s:.3f}"] if offset_s else []), *_safe_input("in.bin"),
                        "-map", "0:a:0", "-t", f"{max_s:.3f}", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"))
    x = np.frombuffer(out, dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)
    return x


def extract_frames(data: bytes, n: int, *, max_side: int = 448, info: dict | None = None) -> list[np.ndarray]:
    """`n` RGB frames (uint8 [h, w, 3]) at evenly spaced times (segment midpoints) across the video."""
    info = info or probe(data)
    if not info["has_video"]:
        raise MediaError("no video stream")
    w0, h0 = int(info["width"] or 0), int(info["height"] or 0)
    if w0 <= 0 or h0 <= 0 or w0 * h0 > LIMITS.max_pixels:
        raise MediaError(f"bad frame size {w0}x{h0}")
    s = min(1.0, max_side / max(w0, h0))
    w, h = max(2, int(round(w0 * s / 2)) * 2), max(2, int(round(h0 * s / 2)) * 2)
    dur = min(float(info["duration_s"] or 0.0), LIMITS.max_duration_s)
    frames = []
    with _Work({"in.bin": data}) as wk:
        if dur <= 0:
            times = [0.0]
        else:
            times = [dur * (i + 0.5) / n for i in range(n)]
        for t in times:
            out = wk.run(_ff("-ss", f"{t:.3f}", *_safe_input("in.bin"), "-map", "0:v:0", "-frames:v", "1",
                             "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"))
            if len(out) < w * h * 3:   # seeking past the last decodable frame: take the last one instead
                out = wk.run(_ff("-sseof", "-0.5", *_safe_input("in.bin"), "-map", "0:v:0", "-frames:v", "1",
                                 "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"))
            if len(out) < w * h * 3:
                raise MediaError("could not decode a frame")
            frames.append(np.frombuffer(out[:w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy())
    return frames


def gray_sequence(data: bytes, *, fps: float = 4.0, size: int = 64, max_frames: int = 1200) -> np.ndarray:
    """Low-resolution grayscale frames [T, size, size] (uint8) at `fps`, for motion and cut statistics."""
    with _Work({"in.bin": data}) as w:
        out = w.run(_ff(*_safe_input("in.bin"), "-map", "0:v:0", "-vf", f"fps={fps},scale={size}:{size}:flags=area",
                        "-frames:v", str(max_frames), "-pix_fmt", "gray", "-f", "rawvideo", "-"))
    n = len(out) // (size * size)
    return np.frombuffer(out[:n * size * size], dtype=np.uint8).reshape(n, size, size)


# ============================================================================================== audio synthesis
def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """16-bit PCM mono WAV."""
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes((x * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def synth_speech(text: str, *, voice: str = "slt", sr: int = 16000, rate: float = 1.0) -> bytes:
    """Speech for `text` with the flite synthesiser built into FFmpeg (voices awb, kal, kal16, rms, slt); WAV bytes.
    `rate` > 1 speaks faster (tempo change without pitch change)."""
    if voice not in ("awb", "kal", "kal16", "rms", "slt"):
        raise MediaError(f"unknown voice {voice!r}")
    text = " ".join(str(text).split())[:LIMITS.max_text_chars]
    if not text:
        raise MediaError("empty text")
    with _Work({"speech.txt": text.encode("utf-8")}) as w:
        af = [] if abs(rate - 1.0) < 1e-6 else ["-af", f"atempo={min(2.0, max(0.5, rate)):.3f}"]
        w.run(_ff("-f", "lavfi", "-i", f"flite=textfile=speech.txt:voice={voice}", *af, "-ac", "1", "-ar", str(sr),
                  "-c:a", "pcm_s16le", "out.wav"))
        return w.read("out.wav")


def synth_tone(freq: float, seconds: float, *, sr: int = 16000, amp: float = 0.5, harmonics: Sequence[float] = (1.0,),
               fade_s: float = 0.01) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for k, a in enumerate(harmonics, start=1):
        x += a * np.sin(2 * np.pi * freq * k * t)
    x *= amp / max(1e-9, float(np.max(np.abs(x))) if len(x) else 1.0)
    return _fade(x, int(fade_s * sr))


def _fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        r = np.linspace(0.0, 1.0, n)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


NOTE_NAMES = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


def note_freq(name: str) -> float:
    """'A4' -> 440.0; accepts sharps/flats ('C#5', 'Eb3')."""
    m = re.fullmatch(r"([A-G])([#b]?)(-?\d)", name)
    if not m:
        raise MediaError(f"bad note {name!r}")
    semis = NOTE_NAMES[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) - 4)
    return 440.0 * 2 ** (semis / 12)


def synth_melody(notes: Sequence[str | Sequence[str] | None], *, bpm: float = 120.0, beats: float = 1.0,
                 sr: int = 16000, timbre: Sequence[float] = (1.0, 0.5, 0.33, 0.25), amp: float = 0.5,
                 decay: float = 3.0, drums: bool = False) -> np.ndarray:
    """A melody (or chord sequence: a list of note names per step; None = rest) with a plucked-string envelope and an
    optional kick/hi-hat pattern. Returns mono float32 PCM."""
    step = int(sr * 60.0 / bpm * beats)
    out = np.zeros(step * len(notes), dtype=np.float64)
    t = np.arange(step) / sr
    env = np.exp(-decay * t)
    for i, nt in enumerate(notes):
        if nt is None:
            continue
        chord = [nt] if isinstance(nt, str) else list(nt)
        seg = np.zeros(step)
        for name in chord:
            f = note_freq(name)
            for k, a in enumerate(timbre, start=1):
                seg += a * np.sin(2 * np.pi * f * k * t)
        out[i * step:(i + 1) * step] += _fade(seg * env, int(0.004 * sr))
    if drums:
        beat = int(sr * 60.0 / bpm)
        kick_t = np.arange(int(0.12 * sr)) / sr
        kick = np.sin(2 * np.pi * (60 + 90 * np.exp(-30 * kick_t)) * kick_t) * np.exp(-25 * kick_t)
        rng = np.random.default_rng(7)
        hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr)) * 0.3
        for b in range(0, len(out) - len(kick), beat):
            out[b:b + len(kick)] += kick * 1.5
            h = b + beat // 2
            if h + len(hat) < len(out):
                out[h:h + len(hat)] += hat
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * amp).astype(np.float32)


def synth_noise(seconds: float, *, color: str = "pink", sr: int = 16000, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """White, pink (1/f) or brown (1/f^2) noise, seeded."""
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if color != "white":
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        f[0] = f[1] if len(f) > 1 else 1.0
        X /= f ** ({"pink": 0.5, "brown": 1.0}[color])
        x = np.fft.irfft(X, n)
    return (x / (float(np.max(np.abs(x))) or 1.0) * amp).astype(np.float32)


def mix(*tracks: np.ndarray, gains: Sequence[float] | None = None) -> np.ndarray:
    n = max(len(t) for t in tracks)
    out = np.zeros(n, dtype=np.float64)
    for i, t in enumerate(tracks):
        out[:len(t)] += (gains[i] if gains else 1.0) * t
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / max(1.0, peak)).astype(np.float32)


# ============================================================================================== images
_FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/freefont")
_FONTS = {("sans", False, False): "DejaVuSans.ttf", ("sans", True, False): "DejaVuSans-Bold.ttf",
          ("sans", False, True): "DejaVuSans-Oblique.ttf", ("sans", True, True): "DejaVuSans-BoldOblique.ttf",
          ("serif", False, False): "DejaVuSerif.ttf", ("serif", True, False): "DejaVuSerif-Bold.ttf",
          ("serif", False, True): "DejaVuSerif-Italic.ttf", ("serif", True, True): "DejaVuSerif-BoldItalic.ttf",
          ("mono", False, False): "DejaVuSansMono.ttf", ("mono", True, False): "DejaVuSansMono-Bold.ttf",
          ("mono", False, True): "DejaVuSansMono-Oblique.ttf", ("mono", True, True): "DejaVuSansMono-BoldOblique.ttf"}
_font_cache: dict = {}


def font(family: str = "sans", size: int = 24, bold: bool = False, italic: bool = False):
    from PIL import ImageFont
    key = (family, int(size), bool(bold), bool(italic))
    if key in _font_cache:
        return _font_cache[key]
    name = _FONTS.get((family, bool(bold), bool(italic))) or _FONTS[("sans", False, False)]
    f = None
    for d in _FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            f = ImageFont.truetype(p, int(size))
            break
    if f is None:
        f = ImageFont.load_default(size=int(size))
    _font_cache[key] = f
    return f


def parse_color(c: Any, default=(255, 255, 255)) -> tuple[int, int, int]:
    from PIL import ImageColor
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(int(max(0, min(255, v))) for v in c[:3])
    try:
        rgb = ImageColor.getrgb(str(c).strip())
        return tuple(int(v) for v in rgb[:3])
    except (ValueError, AttributeError):
        return default


def open_image(data: bytes):
    """Decode an image with Pillow under the pixel limit; returns an RGB image (first frame of animations)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = LIMITS.max_pixels
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > LIMITS.max_pixels:
            raise MediaError(f"image {im.width}x{im.height} over the pixel limit")
        im.seek(0)
        im.load()
    except MediaError:
        raise
    except Exception as e:  # PIL raises many types for corrupt data, including DecompressionBombError
        raise MediaError(f"not a decodable image: {type(e).__name__}: {str(e)[:120]}") from e
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


def image_bytes(im, fmt: str = "PNG", codec_quality: int = 90) -> bytes:
    """Encode a PIL image. `codec_quality` is the JPEG/WebP compression level (1-95), a codec setting."""
    buf = io.BytesIO()
    fmt = fmt.upper().replace("JPG", "JPEG")
    kw = {}
    if fmt in ("JPEG", "WEBP"):
        kw["quality"] = int(codec_quality)  # not-a-content-score: Pillow's name for the JPEG/WebP compression level
    if fmt == "PNG":
        kw["optimize"] = False
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _wrap(draw, text: str, fnt, max_w: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        words = para.split(" ")
        cur = ""
        for wd in words:
            cand = wd if not cur else cur + " " + wd
            if draw.textlength(cand, font=fnt) <= max_w or not cur:
                if draw.textlength(cand, font=fnt) > max_w and not cur:   # a single word longer than the line
                    while wd and draw.textlength(wd, font=fnt) > max_w:
                        k = max(1, int(len(wd) * max_w / max(1.0, draw.textlength(wd, font=fnt))))
                        lines.append(wd[:k])
                        wd = wd[k:]
                    cur = wd
                else:
                    cur = cand
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
    return lines


def render_text_image(text: str, *, width: int = 800, height: int | None = None, font_size: int = 32,
                      color: Any = "#111111", background: Any = "#ffffff", padding: int = 40, align: str = "left",
                      family: str = "sans", bold: bool = False, line_spacing: float = 1.3, fmt: str = "PNG") -> bytes:
    """Word-wrapped text on a solid background. With `height` None the image grows to fit (at most 4x the width)."""
    from PIL import Image, ImageDraw
    text = str(text)[:LIMITS.max_text_chars]
    width = int(max(64, min(4096, width)))
    fnt = font(family, max(6, min(400, int(font_size))), bold)
    probe_im = Image.new("RGB", (8, 8))
    d = ImageDraw.Draw(probe_im)
    lines = _wrap(d, text, fnt, width - 2 * padding)
    lh = int(font_size * line_spacing)
    h = int(height or min(4 * width, 2 * padding + lh * len(lines)))
    h = max(32, min(8192, h))
    im = Image.new("RGB", (width, h), parse_color(background))
    d = ImageDraw.Draw(im)
    y = padding
    for ln in lines:
        if y + lh > h - padding // 2 and height is not None:
            break
        tw = d.textlength(ln, font=fnt)
        x = {"left": padding, "center": (width - tw) / 2, "right": width - padding - tw}.get(align, padding)
        d.text((x, y), ln, fill=parse_color(color, (0, 0, 0)), font=fnt)
        y += lh
    return image_bytes(im, fmt)


def draw_scene(spec: dict, *, width: int = 512, height: int = 512) -> bytes:
    """Draw a simple vector scene: {"background": color, "shapes": [{"kind": circle|rect|triangle|line|text,
    "xy": [...], "fill": color, "outline": color, "width": n, "text": str, "size": n}], "fmt": "PNG"}.
    Coordinates are fractions of the canvas (0..1)."""
    from PIL import Image, ImageDraw
    width, height = int(max(16, min(4096, width))), int(max(16, min(4096, height)))
    im = Image.new("RGB", (width, height), parse_color(spec.get("background", "#ffffff")))
    d = ImageDraw.Draw(im)
    for s in list(spec.get("shapes", []))[:500]:
        k = s.get("kind")
        xy = [float(v) for v in s.get("xy", [])]
        pts = [(xy[i] * width, xy[i + 1] * height) for i in range(0, len(xy) - 1, 2)]
        fill = parse_color(s["fill"]) if s.get("fill") is not None else None
        outline = parse_color(s["outline"]) if s.get("outline") is not None else None
        lw = int(s.get("width", 1))
        if k == "circle" and len(pts) >= 2:
            d.ellipse([pts[0], pts[1]], fill=fill, outline=outline, width=lw)
        elif k == "rect" and len(pts) >= 2:
            d.rectangle([pts[0], pts[1]], fill=fill, outline=outline, width=lw)
        elif k in ("triangle", "polygon") and len(pts) >= 3:
            d.polygon(pts, fill=fill, outline=outline)
        elif k == "line" and len(pts) >= 2:
            d.line(pts, fill=fill or outline or (0, 0, 0), width=lw)
        elif k == "text" and pts:
            d.text(pts[0], str(s.get("text", ""))[:2000], fill=fill or (0, 0, 0),
                   font=font(s.get("family", "sans"), int(s.get("size", 24)), bool(s.get("bold"))))
    return image_bytes(im, spec.get("fmt", "PNG"))


def process_image(data: bytes, ops: Iterable[dict]) -> bytes:
    """Apply image operations in order and encode the result. Ops: {"op": "resize", "w", "h"} | {"op": "scale", "f"}
    | {"op": "crop", "box": [l, t, r, b] fractions} | {"op": "rotate", "deg"} | {"op": "flip"} | {"op": "mirror"}
    | {"op": "grayscale"} | {"op": "brightness", "f"} | {"op": "contrast", "f"} | {"op": "blur", "r"}
    | {"op": "noise", "sigma", "seed"} | {"op": "text", "text", "xy", "size", "color"}
    | {"op": "encode", "fmt": PNG|JPEG|WEBP, "codec_quality"} (last encode wins; default PNG)."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
    im = open_image(data)
    fmt, codec_q = "PNG", 90
    for op in list(ops)[:64]:
        k = op.get("op")
        if k == "resize":
            im = im.resize((int(max(1, min(8192, op["w"]))), int(max(1, min(8192, op["h"])))), Image.BICUBIC)
        elif k == "scale":
            f = float(op["f"])
            im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.BICUBIC)
        elif k == "crop":
            l, t, r, b = [float(v) for v in op["box"]]
            im = im.crop((int(l * im.width), int(t * im.height), int(r * im.width), int(b * im.height)))
        elif k == "rotate":
            im = im.rotate(float(op["deg"]), resample=Image.BICUBIC, expand=bool(op.get("expand", False)),
                           fillcolor=(255, 255, 255))
        elif k == "flip":
            im = ImageOps.flip(im)
        elif k == "mirror":
            im = ImageOps.mirror(im)
        elif k == "grayscale":
            im = ImageOps.grayscale(im).convert("RGB")
        elif k == "brightness":
            im = ImageEnhance.Brightness(im).enhance(float(op["f"]))
        elif k == "contrast":
            im = ImageEnhance.Contrast(im).enhance(float(op["f"]))
        elif k == "blur":
            im = im.filter(ImageFilter.GaussianBlur(float(op["r"])))
        elif k == "noise":
            rng = np.random.default_rng(int(op.get("seed", 0)))
            a = np.asarray(im, dtype=np.float32)
            a = np.clip(a + rng.normal(0.0, float(op["sigma"]), a.shape), 0, 255).astype(np.uint8)
            im = Image.fromarray(a)
        elif k == "text":
            d = ImageDraw.Draw(im)
            x, y = op.get("xy", [0.05, 0.05])
            d.text((float(x) * im.width, float(y) * im.height), str(op.get("text", ""))[:2000],
                   fill=parse_color(op.get("color", "#000000"), (0, 0, 0)), font=font(size=int(op.get("size", 24))))
        elif k == "encode":
            fmt, codec_q = str(op.get("fmt", "PNG")).upper(), int(op.get("codec_quality", 90))
        else:
            raise MediaError(f"unknown image op {k!r}")
        if im.width * im.height > LIMITS.max_pixels:
            raise MediaError("image over the pixel limit")
    return image_bytes(im, fmt, codec_q)


# ---------------------------------------------------------------------------------------------- HTML rendering
_BLOCK = {"p", "div", "section", "article", "header", "footer", "main", "nav", "aside", "h1", "h2", "h3", "h4", "h5",
          "h6", "ul", "ol", "li", "blockquote", "pre", "table", "tr", "form", "figure", "figcaption", "hr", "br",
          "title", "dl", "dt", "dd"}
_SKIP = {"script", "style", "head", "noscript", "template", "svg", "iframe", "object", "canvas"}
_HEAD_SIZE = {"h1": 2.0, "h2": 1.6, "h3": 1.3, "h4": 1.15, "h5": 1.0, "h6": 0.9}


class _Layout(_hp.HTMLParser):
    """Collects (text, style) runs grouped into blocks from a subset of HTML: headings, paragraphs, lists,
    blockquotes, preformatted text, links, bold/italic/code, line breaks, horizontal rules and inline colours."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self.stack: list[tuple[str, dict]] = []
        self.skip = 0
        self.title = ""
        self.in_title = False
        self.body_style: dict = {}
        self.list_stack: list[list] = []
        self._new_block("p")

    def _style(self) -> dict:
        st = {"bold": False, "italic": False, "mono": False, "color": None, "link": False, "size": 1.0}
        for tag, css in self.stack:
            if tag in ("b", "strong", "th") or tag in _HEAD_SIZE:
                st["bold"] = True
            if tag in ("i", "em", "cite"):
                st["italic"] = True
            if tag in ("code", "pre", "kbd", "samp", "tt"):
                st["mono"] = True
            if tag == "a":
                st["link"] = True
            if tag in _HEAD_SIZE:
                st["size"] = _HEAD_SIZE[tag]
            if css.get("color"):
                st["color"] = css["color"]
        return st

    def _new_block(self, kind: str, **kw) -> None:
        if self.blocks and not self.blocks[-1]["runs"] and self.blocks[-1]["kind"] not in ("hr",):
            self.blocks[-1].update(kind=kind, indent=len(self.list_stack), **kw)
            return
        self.blocks.append({"kind": kind, "runs": [], "indent": len(self.list_stack), **kw})

    @staticmethod
    def _css(attrs) -> dict:
        out = {}
        for k, v in attrs:
            if k == "style" and v:
                for decl in v.split(";"):
                    if ":" in decl:
                        a, b = decl.split(":", 1)
                        out[a.strip().lower()] = b.strip()
            if k in ("bgcolor",) and v:
                out["background-color"] = v
            if k == "color" and v:
                out["color"] = v
        return out

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
            return
        if tag == "title":
            self.in_title = True
            return
        css = self._css(attrs)
        if tag == "body":
            self.body_style = css
        if tag in ("br",):
            self._new_block(self.blocks[-1]["kind"] if self.blocks else "p")
            return
        if tag == "hr":
            self.blocks.append({"kind": "hr", "runs": [], "indent": 0})
            self._new_block("p")
            return
        if tag == "img":
            alt = dict(attrs).get("alt") or ""
            self._new_block("p")
            self.blocks[-1]["runs"].append((f"[image: {alt}]" if alt else "[image]", {**self._style(), "italic": True}))
            self._new_block("p")
            return
        if tag in ("ul", "ol"):
            self.list_stack.append([tag, 0])
        if tag == "li":
            mark = "-"
            if self.list_stack:
                self.list_stack[-1][1] += 1
                if self.list_stack[-1][0] == "ol":
                    mark = f"{self.list_stack[-1][1]}."
            self._new_block("li", mark=mark)
        elif tag in _BLOCK:
            self._new_block(tag)
        self.stack.append((tag, css))

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "title":
            self.in_title = False
            return
        if tag in ("ul", "ol") and self.list_stack:
            self.list_stack.pop()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag in _BLOCK or tag == "li":
            self._new_block("p")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.skip:
            return
        pre = any(t == "pre" for t, _ in self.stack)
        txt = data if pre else re.sub(r"\s+", " ", data)
        if not txt.strip() and not pre:
            if self.blocks[-1]["runs"] and not self.blocks[-1]["runs"][-1][0].endswith(" "):
                self.blocks[-1]["runs"].append((" ", self._style()))
            return
        if pre:
            parts = txt.split("\n")
            for j, part in enumerate(parts):
                if j:
                    self._new_block("pre")
                if part:
                    self.blocks[-1]["runs"].append((part, self._style()))
            return
        self.blocks[-1]["runs"].append((txt, self._style()))


def html_text(html: str) -> tuple[str, str]:
    """(title, visible text) of an HTML document, blocks separated by newlines."""
    lay = _Layout()
    lay.feed(str(html)[:LIMITS.max_text_chars * 4])
    lay.close()
    lines = []
    for b in lay.blocks:
        t = "".join(r[0] for r in b["runs"]).strip()
        if t:
            lines.append((b.get("mark", "") + " " + t).strip() if b["kind"] == "li" else t)
    return " ".join(lay.title.split()), "\n".join(lines)


def render_html_image(html: str, *, width: int = 800, max_height: int = 4000, base_size: int = 20,
                      engine: str = "lite", fmt: str = "PNG") -> bytes:
    """Render an HTML document to an image.

    engine "lite" (default): a bounded, deterministic layout of headings, paragraphs, lists, quotes, code blocks,
    links, emphasis, rules, inline text colours and the body background colour (no scripts, no external resources).
    engine "chromium": a headless Chromium screenshot (Playwright) with JavaScript disabled and every network request
    refused; one browser at a time (file lock), 30 s limit."""
    if engine == "chromium":
        return _render_html_chromium(html, width=width, max_height=max_height)
    from PIL import Image, ImageDraw
    width = int(max(200, min(2400, width)))
    lay = _Layout()
    lay.feed(str(html)[:LIMITS.max_text_chars * 4])
    lay.close()
    bg = parse_color(lay.body_style.get("background-color") or lay.body_style.get("background") or "#ffffff")
    fg_default = parse_color(lay.body_style.get("color") or "#1a1a1a", (26, 26, 26))
    pad = max(16, width // 25)
    probe_d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    # layout pass: words with styles into lines
    lines: list[tuple[int, list[tuple[float, str, Any, tuple]], int, dict]] = []   # (y_advance, items, x0, block)
    for b in lay.blocks:
        if b["kind"] == "hr":
            lines.append((int(base_size * 1.2), [], pad, b))
            continue
        runs = [r for r in b["runs"] if r[0]]
        if not runs or not "".join(r[0] for r in runs).strip():
            continue
        indent = pad + int(base_size * 1.5 * b.get("indent", 0)) + (int(base_size * 1.2)
                                                                    if b["kind"] == "blockquote" else 0)
        x_max = width - pad
        if b["kind"] == "li":
            runs = [(b.get("mark", "-") + " ", runs[0][1])] + runs
        cur: list = []
        x = indent
        line_h = 0
        words = []
        for text, st in runs:
            for tok in re.split(r"(\s+)", text):
                if tok:
                    words.append((tok, st))
        for tok, st in words:
            size = int(base_size * st["size"])
            fnt = font("mono" if st["mono"] or b["kind"] == "pre" else "sans", size, st["bold"], st["italic"])
            if tok.isspace():
                if cur:
                    x += probe_d.textlength(" ", font=fnt)
                elif b["kind"] == "pre":
                    x += probe_d.textlength(tok.replace("\t", "    "), font=fnt)
                continue
            tw = probe_d.textlength(tok, font=fnt)
            if x + tw > x_max and cur:
                lines.append((line_h, cur, indent, b))
                cur, x, line_h = [], indent, 0
            color = parse_color(st["color"], fg_default) if st["color"] else ((30, 80, 200) if st["link"]
                                                                              else fg_default)
            cur.append((x, tok, fnt, color))
            x += tw
            line_h = max(line_h, int(size * 1.35))
        if cur:
            lines.append((line_h, cur, indent, b))
        lines.append((int(base_size * 0.5), [], indent, {"kind": "gap"}))
    height = min(int(max_height), max(64, pad * 2 + sum(h for h, *_ in lines)))
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    y = pad
    for h, items, x0, b in lines:
        if y + h > height:
            break
        if b.get("kind") == "hr":
            d.line([(pad, y + h // 2), (width - pad, y + h // 2)], fill=(160, 160, 160), width=2)
        if b.get("kind") == "blockquote" and items:
            d.line([(x0 - int(base_size * 0.7), y), (x0 - int(base_size * 0.7), y + h)], fill=(180, 180, 180), width=3)
        if b.get("kind") == "pre" and items:
            d.rectangle([(x0 - 6, y), (width - pad + 6, y + h)], fill=tuple(int(v * 0.94) for v in bg))
        for x, tok, fnt, color in items:
            d.text((x, y), tok, fill=color, font=fnt)
        y += h
    return image_bytes(im, fmt)


def _render_html_chromium(html: str, *, width: int, max_height: int) -> bytes:
    import fcntl
    from playwright.sync_api import sync_playwright
    lock_path = os.path.join(tempfile.gettempdir(), "crucible-media-chromium.lock")
    with open(lock_path, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox",
                                                                  "--renderer-process-limit=1"])
                try:
                    page = browser.new_page(viewport={"width": int(width), "height": 600}, java_script_enabled=False)
                    page.route("**/*", lambda route: route.abort())
                    page.set_default_timeout(30000)
                    page.set_content(str(html)[:LIMITS.max_text_chars * 4], wait_until="domcontentloaded")
                    png = page.screenshot(full_page=True, type="png", timeout=30000)
                finally:
                    browser.close()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    im = open_image(png)
    if im.height > max_height:
        im = im.crop((0, 0, im.width, int(max_height)))
    return image_bytes(im, "PNG")


# ============================================================================================== video and transcoding
AUDIO_FORMATS = {"wav": ("wav", ["-c:a", "pcm_s16le"]), "flac": ("flac", ["-c:a", "flac"]),
                 "mp3": ("mp3", ["-c:a", "libmp3lame"]), "ogg": ("ogg", ["-c:a", "libvorbis"]),
                 "opus": ("ogg", ["-c:a", "libopus"]), "m4a": ("ipod", ["-c:a", "aac"])}
# format -> (muxer, video codec args, audio codec args)
VIDEO_FORMATS = {"mp4": ("mp4", ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags",
                                 "+faststart"], ["-c:a", "aac"]),
                 "webm": ("webm", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "0",
                                   "-pix_fmt", "yuv420p"], ["-c:a", "libopus"]),
                 "mkv": ("matroska", ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                         ["-c:a", "aac"])}


def transcode(data: bytes, fmt: str, *, crf: int | None = None, audio_bitrate: str | None = None,
              width: int | None = None, height: int | None = None, fps: float | None = None,
              sample_rate: int | None = None, start_s: float | None = None, duration_s: float | None = None,
              audio: bool = True, video: bool = True) -> bytes:
    """Convert audio or video to `fmt` (wav, flac, mp3, ogg, opus, m4a, mp4, webm, mkv) with optional compression, size,
    frame rate, sample rate and trimming. Scaling keeps the aspect ratio when only one side is given."""
    fmt = fmt.lower()
    if fmt not in AUDIO_FORMATS and fmt not in VIDEO_FORMATS:
        raise MediaError(f"unsupported format {fmt!r}")
    args = []
    if start_s:
        args += ["-ss", f"{float(start_s):.3f}"]
    args += _safe_input("in.bin")
    if duration_s:
        args += ["-t", f"{min(float(duration_s), LIMITS.max_duration_s):.3f}"]
    else:
        args += ["-t", f"{LIMITS.max_duration_s:.0f}"]
    if fmt in AUDIO_FORMATS:
        muxer, codec = AUDIO_FORMATS[fmt]
        args += ["-vn", "-map", "0:a:0", *codec]
        if audio_bitrate and fmt not in ("wav", "flac"):
            args += ["-b:a", audio_bitrate]
        if sample_rate:
            args += ["-ar", str(int(sample_rate))]
    else:
        muxer, vcodec, acodec = VIDEO_FORMATS[fmt]
        args += ["-map", "0:v:0"] if video else ["-vn"]
        args += ["-map", "0:a:0?"] if audio else ["-an"]
        args += (vcodec if video else []) + (acodec if audio else [])
        if crf is not None:
            args += ["-crf", str(int(crf))] + (["-b:v", "0"] if fmt == "webm" else [])
        vf = []
        if width or height:
            vf.append(f"scale={int(width) if width else -2}:{int(height) if height else -2}:flags=bicubic")
        if fps:
            vf.append(f"fps={float(fps)}")
        if vf:
            args += ["-vf", ",".join(vf)]
        if audio_bitrate:
            args += ["-b:a", audio_bitrate]
        if sample_rate:
            args += ["-ar", str(int(sample_rate))]
    with _Work({"in.bin": data}) as w:
        w.run(_ff(*args, "-f", muxer, "-y", "out.bin"))
        return w.read("out.bin")


def assemble_video(images: Sequence[bytes], audio: bytes | None = None, *, seconds_per_image: float | Sequence[float] = 2.0,
                   size: tuple[int, int] = (640, 360), fps: int = 25, crf: int = 28, fade_s: float = 0.0,
                   background: Any = "#000000", fmt: str = "mp4") -> bytes:
    """A video from still images (each shown for `seconds_per_image`, letterboxed to `size`, optional cross-fade
    to black) with an optional audio track (trimmed or padded with silence to the video's length)."""
    from PIL import Image
    if not images:
        raise MediaError("no images")
    if len(images) > 600:
        raise MediaError("at most 600 images")
    W, H = int(size[0]) // 2 * 2, int(size[1]) // 2 * 2
    if W * H > 1920 * 1088:
        raise MediaError("video larger than 1920x1088")
    durs = [float(seconds_per_image)] * len(images) if isinstance(seconds_per_image, (int, float)) \
        else [float(s) for s in seconds_per_image]
    if len(durs) != len(images) or min(durs) <= 0 or sum(durs) > LIMITS.max_duration_s:
        raise MediaError("bad image durations")
    inputs = {}
    bgc = parse_color(background, (0, 0, 0))
    for i, data in enumerate(images):
        im = open_image(data)
        s = min(W / im.width, H / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC)
        canvas = Image.new("RGB", (W, H), bgc)
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        inputs[f"img{i:04d}.png"] = image_bytes(canvas, "PNG")
    if audio is not None:
        inputs["audio.bin"] = audio
    total = sum(durs)
    with _Work(inputs) as w:
        args = []
        for i, dsec in enumerate(durs):
            args += ["-loop", "1", "-framerate", str(int(fps)), "-t", f"{dsec:.3f}",
                     "-protocol_whitelist", "file,pipe", "-format_whitelist", "image2,png_pipe", "-i",
                     f"img{i:04d}.png"]
        n = len(durs)
        chains = []
        for i, dsec in enumerate(durs):
            ch = f"[{i}:v]format=yuv420p,setsar=1"
            if fade_s > 0 and dsec > 2 * fade_s:
                ch += f",fade=t=in:st=0:d={fade_s:.3f},fade=t=out:st={dsec - fade_s:.3f}:d={fade_s:.3f}"
            chains.append(ch + f"[v{i}]")
        graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]"
        if fmt not in VIDEO_FORMATS:
            raise MediaError(f"unsupported video format {fmt!r}")
        muxer, vcodec, acodec = VIDEO_FORMATS[fmt]
        if audio is not None:
            args += _safe_input("audio.bin")            # every input before the first output option
        args += ["-filter_complex", graph, "-map", "[vout]"]
        if audio is not None:
            args += ["-map", f"{n}:a:0", "-af", f"apad,atrim=0:{total:.3f}", *acodec, "-b:a", "128k"]
        args += [*vcodec, "-crf", str(int(crf)), "-r", str(int(fps)), "-t", f"{total:.3f}"]
        w.run(_ff(*args, "-f", muxer, "-y", "out.bin"))
        return w.read("out.bin")


def media_info() -> dict:
    """Tool versions (part of the media tools' recorded identity)."""
    def ver(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, timeout=10).stdout.decode().splitlines()[0][:120]
        except Exception:
            return None
    import PIL
    return {"tool_version": TOOL_VERSION, "ffmpeg": ver([FFMPEG, "-version"]), "pillow": PIL.__version__}
