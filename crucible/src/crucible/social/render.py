"""Creator media tooling: render, process and assemble actual media bytes (MASTER_SPEC Part 4.2).

Lightweight, local tools only (no generative model is needed for an ordinary action):

* images: text cards and vector scenes drawn with Pillow; image processing (resize, crop, rotate, flip, grayscale,
  blur, sharpen, brightness, contrast, text overlay, thumbnail);
* audio: tones, chords, melodies and noise synthesised with numpy into PCM WAV; effects (fade, reverb, tempo,
  pitch, gain, reverse) applied by SoX;
* video: images (and an optional sound track) assembled into H.264/AAC MP4 by FFmpeg;
* documents: PDF written directly (text pages, standard Helvetica font, uncompressed content streams);
* previews: any text-like artifact rendered as an image.

External programs (FFmpeg, SoX) run as child processes in a private temporary directory with a wall-clock
timeout, CPU-time, address-space and file-size rlimits, one thread, a minimal environment and no network inputs
(`-protocol_whitelist file,pipe`). Their outputs are not guaranteed to be bit-identical across tool versions, so the
studio records every produced artifact through `recorded_bytes` (the `crucible.substrate.inference` record format:
tool id, tool hash, inputs digest, output REF): regeneration re-stores the recorded bytes instead of re-running a
tool, and replay never runs anything.

When workstream D's `crucible.semantic.media` is available its functions can be used instead (same inputs, bytes
in, bytes out); this module keeps the studio working without it.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import resource
import shutil
import subprocess
import tempfile
import wave
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..core.errors import InvalidAction, ProtocolError
from ..core.kernel import Ctx

FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
SOX = shutil.which("sox") or "/usr/bin/sox"
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/freefont")
TOOL_ID = "crucible-studio/1"
TIMEOUT_S = 120.0
CPU_S = 240
MEM_MB = 3072
MAX_OUT = 32 * 1024 * 1024
MAX_SIDE = 4096
MAX_AUDIO_S = 600.0
MAX_VIDEO_S = 3600.0
SR = 22050


class MediaError(InvalidAction):
    """A media input could not be processed (bad data, over a limit, tool failure)."""
    code = "media_error"


# ---------------------------------------------------------------------------------------------------- processes
def _limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S + 5))
    mem = MEM_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUT * 4, MAX_OUT * 4))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _jailed(argv: Sequence[str], cwd: str, timeout: float):
    from ..sandbox import isolation
    return isolation.run(list(argv), cwd=cwd, timeout=timeout, preexec_fn=_limits,
                         env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "MALLOC_ARENA_MAX": "2"})


def run_tool(argv: Sequence[str], cwd: str) -> bytes:
    """Run FFmpeg/SoX on participant bytes in `cwd`, jailed (`crucible.sandbox.isolation.run`: own uid, Landlock with
    read-write access to `cwd` only, seccomp without sockets or new processes) in addition to the rlimits and the
    timeout (Part 16, Gate W.1). Where the jail cannot be applied this fails closed (MediaError) unless
    CRUCIBLE_ALLOW_UNJAILED_TOOLS=1."""
    from ..sandbox import isolation
    try:
        p = _jailed(argv, cwd, TIMEOUT_S)
    except isolation.JailUnavailable as e:
        raise MediaError(f"untrusted media cannot be processed here: {e}") from None
    except subprocess.TimeoutExpired:
        raise MediaError(f"{os.path.basename(argv[0])} exceeded {TIMEOUT_S}s") from None
    except OSError as e:
        raise MediaError(f"{os.path.basename(argv[0])} is unavailable: {e}") from None
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"{os.path.basename(argv[0])} failed ({p.returncode}): {(err[-1] if err else '')[:300]}")
    return p.stdout


class _Work:
    def __init__(self, inputs: dict[str, bytes]):
        self.dir = tempfile.mkdtemp(prefix="crucible-studio-", dir=os.environ.get("CRUCIBLE_MEDIA_TMP") or None)
        for name, data in inputs.items():
            with open(os.path.join(self.dir, name), "wb") as fh:
                fh.write(data)

    def read(self, name: str) -> bytes:
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            raise MediaError("tool produced no output")
        if os.path.getsize(p) > MAX_OUT:
            raise MediaError(f"output larger than {MAX_OUT} bytes")
        with open(p, "rb") as fh:
            return fh.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)


def tool_versions() -> dict:
    out = {}
    for name, argv in (("ffmpeg", [FFMPEG, "-version"]), ("sox", [SOX, "--version"])):
        # the version query starts the same native binaries, so it runs in the same jail as every studio launch
        with _Work({}) as w:
            try:
                out[name] = _jailed(argv, w.dir, 10.0).stdout.decode("utf-8", "replace").splitlines()[0][:120]
            except Exception:  # noqa: BLE001 - a missing tool is recorded as None in the tool-chain identity
                out[name] = None
    try:
        import PIL
        out["pillow"] = PIL.__version__
    except ImportError:
        out["pillow"] = None
    out["numpy"] = np.__version__
    return out


_HASH: list[str] = []


def tool_hash() -> str:
    """Identity of the tool chain recorded with every produced artifact."""
    if not _HASH:
        _HASH.append(hashlib.sha256(repr(sorted(tool_versions().items())).encode()).hexdigest())
    return _HASH[0]


# ---------------------------------------------------------------------------------------------------- recording
def recorded_bytes(ctx: Ctx, kind: str, inputs: Any, produce: Callable[[], bytes], *, mime: str, owner: str,
                   parents: Sequence[str] = (), meta: dict | None = None, akind: str = "file") -> tuple[str, bytes]:
    """Produce bytes once (recorded like `substrate.inference.recorded`) and store them as an artifact of `owner`.
    Regeneration re-stores the recorded bytes (found by their SHA-256) without running `produce`; a recorded
    failure is re-raised the same way. Returns (REF, bytes)."""
    from ..substrate import artifacts, inference
    dig = inference.inputs_digest(inputs)
    th = tool_hash()
    slot, rec = ctx.reserve()
    if rec is not None:
        if rec.get("model_id") != TOOL_ID or rec.get("inputs_digest") != dig:
            raise ProtocolError(f"recorded studio output {kind} diverged")
        ctx.fill(slot, inference.PREFIX + kind, rec)
        out = rec.get("output") or {}
        if "error" in out:
            raise MediaError(out["error"])
        data = artifacts._read_bytes(ctx.world, artifacts.parse_ref(out["ref"]))
        if data is None:
            raise ProtocolError(f"recorded studio output {out['ref']} is not available in this world's store")
    else:
        record = {"model_id": TOOL_ID, "model_hash": th, "inputs_digest": dig, "output_kind": "json"}
        try:
            data = produce()
        except MediaError as e:
            record["output"] = {"error": str(e)[:500]}
            ctx.fill(slot, inference.PREFIX + kind, record)
            raise
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise MediaError("tool produced no output")
        data = bytes(data)
        record["output"] = {"ref": artifacts.ref_of(data), "size": len(data)}
        ctx.fill(slot, inference.PREFIX + kind, record)
    ref = artifacts.put(ctx, data, mime, owner=owner, parents=parents, meta=meta, kind=akind)
    return ref, data


# ---------------------------------------------------------------------------------------------------- images
def font(size: int = 24, bold: bool = False, mono: bool = False):
    from PIL import ImageFont
    names = (["DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"] if mono else
             ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "FreeSansBold.ttf" if bold else "FreeSans.ttf"])
    for d in FONT_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, int(size))
    return ImageFont.load_default(size=int(size))


def color(c: Any, default=(255, 255, 255)) -> tuple[int, int, int]:
    if c is None:
        return default
    if isinstance(c, (list, tuple)) and len(c) == 3 and all(isinstance(x, int) for x in c):
        return tuple(max(0, min(255, x)) for x in c)  # type: ignore[return-value]
    s = str(c).strip()
    named = {"black": "#000000", "white": "#ffffff", "red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4",
             "yellow": "#ffd92f", "orange": "#ff7f0e", "purple": "#9467bd", "gray": "#7f7f7f", "grey": "#7f7f7f",
             "pink": "#e377c2", "brown": "#8c564b", "cyan": "#17becf", "navy": "#1b2a4a", "teal": "#1a9e8f"}
    s = named.get(s.lower(), s)
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    if re.fullmatch(r"#[0-9a-fA-F]{3}", s):
        return tuple(int(ch * 2, 16) for ch in s[1:])  # type: ignore[return-value]
    raise MediaError(f"bad colour {s[:20]!r} (use #rrggbb or a colour name)")


def _side(v: Any, name: str) -> int:
    v = int(v)
    if not 1 <= v <= MAX_SIDE:
        raise MediaError(f"{name} must be 1..{MAX_SIDE}")
    return v


def png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def _wrap(draw, text: str, fnt, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).split("\n"):
        words = para.split(" ")
        cur = ""
        for wd in words:
            t = (cur + " " + wd).strip() if cur else wd
            if draw.textlength(t, font=fnt) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
    return lines


def text_card(text: str, *, width: int = 800, height: int | None = None, font_size: int = 32, fg="#111111",
              bg="#ffffff", title: str | None = None, mono: bool = False) -> bytes:
    from PIL import Image, ImageDraw
    width = _side(width, "width")
    if len(text) > 20_000:
        raise MediaError("text longer than 20000 characters")
    fnt, tf = font(font_size, mono=mono), font(int(font_size * 1.4), bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    margin = max(8, font_size // 2)
    lines = _wrap(probe, text, fnt, width - 2 * margin)
    tlines = _wrap(probe, title, tf, width - 2 * margin) if title else []
    lh, th = int(font_size * 1.35), int(font_size * 1.9)
    need = 2 * margin + len(lines) * lh + len(tlines) * th
    height = _side(height or min(MAX_SIDE, need), "height")
    im = Image.new("RGB", (width, height), color(bg))
    d = ImageDraw.Draw(im)
    y = margin
    for ln in tlines:
        d.text((margin, y), ln, font=tf, fill=color(fg))
        y += th
    for ln in lines:
        if y > height - lh // 2:
            break
        d.text((margin, y), ln, font=fnt, fill=color(fg))
        y += lh
    return png(im)


SHAPES = ("rect", "ellipse", "line", "polygon", "text")


def draw_scene(shapes: Iterable[dict], *, width: int = 512, height: int = 512, bg="#ffffff") -> bytes:
    """A vector scene: [{"shape": rect|ellipse|line|polygon|text, "xy": [...], "fill", "outline", "width", "text",
    "size"}]; coordinates in pixels."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (_side(width, "width"), _side(height, "height")), color(bg))
    d = ImageDraw.Draw(im)
    n = 0
    for s in shapes:
        n += 1
        if n > 500:
            raise MediaError("at most 500 shapes")
        if not isinstance(s, dict) or s.get("shape") not in SHAPES:
            raise MediaError(f"shape {n}: 'shape' must be one of {SHAPES}")
        xy = s.get("xy") or []
        if not isinstance(xy, list) or not all(isinstance(v, (int, float)) for v in xy) or len(xy) % 2 or \
                len(xy) > 200:
            raise MediaError(f"shape {n}: 'xy' must be a flat list of coordinate pairs")
        fill = color(s["fill"]) if s.get("fill") is not None else None
        outline = color(s["outline"]) if s.get("outline") is not None else None
        lw = max(1, min(50, int(s.get("width", 2))))
        kind = s["shape"]
        if kind in ("rect", "ellipse"):
            if len(xy) != 4:
                raise MediaError(f"shape {n}: {kind} needs xy=[x0, y0, x1, y1]")
            box = [min(xy[0], xy[2]), min(xy[1], xy[3]), max(xy[0], xy[2]), max(xy[1], xy[3])]
            (d.rectangle if kind == "rect" else d.ellipse)(box, fill=fill, outline=outline, width=lw)
        elif kind == "line":
            if len(xy) < 4:
                raise MediaError(f"shape {n}: line needs at least two points")
            d.line(xy, fill=fill or outline or (0, 0, 0), width=lw)
        elif kind == "polygon":
            if len(xy) < 6:
                raise MediaError(f"shape {n}: polygon needs at least three points")
            d.polygon(xy, fill=fill, outline=outline)
        else:
            if len(xy) != 2:
                raise MediaError(f"shape {n}: text needs xy=[x, y]")
            d.text((xy[0], xy[1]), str(s.get("text", ""))[:500], font=font(int(s.get("size", 24))),
                   fill=fill or (0, 0, 0))
    return png(im)


IMAGE_OPS = ("resize", "crop", "rotate", "flip", "mirror", "grayscale", "blur", "sharpen", "brightness", "contrast",
             "text", "thumbnail", "border", "invert")


def decode_pixels(data: bytes, max_pixels: int = 40_000_000) -> dict:
    """Decode participant image bytes outside the world process (`crucible.sandbox.isolation.decode_image`: Pillow
    in the jail; Gate W.1 / SP-16b): {"format", "source_mode", "mode", "size", "frames", "raw"}. Raises MediaError
    when the bytes do not decode or the jail cannot be applied (fail closed)."""
    from ..sandbox import isolation
    try:
        meta = isolation.decode_image(bytes(data), max_pixels=int(max_pixels))
    except isolation.JailUnavailable as e:
        raise MediaError(f"untrusted images cannot be decoded here: {e}") from None
    if not meta.get("ok"):
        raise MediaError(f"not a decodable image: {str(meta.get('error', ''))[:200]}")
    return meta


def open_image(data: bytes):
    """A participant image as an RGB Pillow image, decoded in the jail (only raw pixels cross into this process)."""
    from PIL import Image
    meta = decode_pixels(data)
    im = Image.frombytes(meta["mode"], tuple(meta["size"]), meta["raw"])
    return im.convert("RGB")


def process_image(data: bytes, ops: Iterable[dict]) -> bytes:
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageOps
    im = open_image(data)
    for k, op in enumerate(ops):
        if k >= 50:
            raise MediaError("at most 50 operations")
        name = op.get("op") if isinstance(op, dict) else None
        if name not in IMAGE_OPS:
            raise MediaError(f"operation {k}: 'op' must be one of {IMAGE_OPS}")
        if name == "resize":
            im = im.resize((_side(op.get("width", im.width), "width"), _side(op.get("height", im.height), "height")))
        elif name == "thumbnail":
            s = _side(op.get("size", 256), "size")
            im = im.copy()
            im.thumbnail((s, s))
        elif name == "crop":
            box = op.get("box")
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)):
                raise MediaError("crop needs box=[left, top, right, bottom]")
            l, t, r, b = max(0, box[0]), max(0, box[1]), min(im.width, box[2]), min(im.height, box[3])
            if r <= l or b <= t:
                raise MediaError("crop box is empty")
            im = im.crop((l, t, r, b))
        elif name == "rotate":
            im = im.rotate(float(op.get("degrees", 90)), expand=True, fillcolor=color(op.get("fill"), (0, 0, 0)))
        elif name == "flip":
            im = ImageOps.flip(im)
        elif name == "mirror":
            im = ImageOps.mirror(im)
        elif name == "grayscale":
            im = ImageOps.grayscale(im).convert("RGB")
        elif name == "invert":
            im = ImageOps.invert(im)
        elif name == "blur":
            im = im.filter(ImageFilter.GaussianBlur(max(0.0, min(50.0, float(op.get("radius", 2))))))
        elif name == "sharpen":
            im = im.filter(ImageFilter.SHARPEN)
        elif name in ("brightness", "contrast"):
            f = max(0.0, min(10.0, float(op.get("factor", 1.2))))
            im = (ImageEnhance.Brightness if name == "brightness" else ImageEnhance.Contrast)(im).enhance(f)
        elif name == "border":
            im = ImageOps.expand(im, border=max(1, min(500, int(op.get("width", 10)))), fill=color(op.get("color"),
                                                                                                  (0, 0, 0)))
        elif name == "text":
            d = ImageDraw.Draw(im)
            xy = op.get("xy") or [10, 10]
            d.text((float(xy[0]), float(xy[1])), str(op.get("text", ""))[:500], font=font(int(op.get("size", 24))),
                   fill=color(op.get("color"), (255, 255, 255)))
        if im.width * im.height > 40_000_000:
            raise MediaError("image too large")
    return png(im)


# ---------------------------------------------------------------------------------------------------- audio
NOTE_RE = re.compile(r"([A-Ga-g])([#b]?)(-?\d)")


def note_freq(name: str) -> float:
    m = NOTE_RE.fullmatch(str(name))
    if not m:
        raise MediaError(f"bad note {name!r} (e.g. C4, F#3, Bb5)")
    semis = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[m.group(1).lower()]
    semis += {"#": 1, "b": -1, "": 0}[m.group(2)]
    midi = 12 * (int(m.group(3)) + 1) + semis
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, sr: int, attack=0.01, release=0.05) -> np.ndarray:
    e = np.ones(n, dtype=np.float64)
    a, r = min(n, int(attack * sr)), min(n, int(release * sr))
    if a:
        e[:a] = np.linspace(0, 1, a, endpoint=False)
    if r:
        e[n - r:] *= np.linspace(1, 0, r)
    return e


def _tone(freqs: Sequence[float], seconds: float, sr: int, timbre: str) -> np.ndarray:
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n)
    harm = {"sine": (1.0,), "organ": (1.0, 0.5, 0.25, 0.125), "bright": (1.0, 0.7, 0.5, 0.35, 0.2, 0.1),
            "soft": (1.0, 0.2)}.get(timbre)
    if harm is None:
        raise MediaError("timbre must be sine, organ, bright or soft")
    for f in freqs:
        for k, a in enumerate(harm, start=1):
            if f * k < sr / 2:
                out += a * np.sin(2 * math.pi * f * k * t)
    if len(freqs):
        out /= max(1.0, np.abs(out).max() if n else 1.0)
    return out * _env(n, sr)


def synth(kind: str, *, seconds: float = 3.0, freq: float = 440.0, notes: Sequence | None = None, bpm: float = 120.0,
          timbre: str = "organ", seed: int = 0, sr: int = SR) -> np.ndarray:
    """float64 samples in [-1, 1]: tone | chord | melody | noise | silence."""
    seconds = float(seconds)
    if not 0.05 <= seconds <= MAX_AUDIO_S:
        raise MediaError(f"seconds must be 0.05..{MAX_AUDIO_S}")
    if kind == "tone":
        if not 20 <= freq <= 12000:
            raise MediaError("freq must be 20..12000 Hz")
        return 0.6 * _tone([freq], seconds, sr, timbre)
    if kind == "chord":
        fs = [note_freq(x) for x in (notes or ["C4", "E4", "G4"])][:8]
        return 0.6 * _tone(fs, seconds, sr, timbre)
    if kind == "melody":
        if not notes:
            raise MediaError("a melody needs notes, e.g. ['C4', 'E4', 'G4', null, 'C5']")
        if not 20 <= bpm <= 400:
            raise MediaError("bpm must be 20..400")
        beat = 60.0 / bpm
        parts = []
        for x in list(notes)[:2000]:
            if x is None or x == "-":
                parts.append(np.zeros(int(round(beat * sr))))
            elif isinstance(x, list):
                parts.append(0.6 * _tone([note_freq(v) for v in x][:8], beat, sr, timbre))
            else:
                parts.append(0.6 * _tone([note_freq(x)], beat, sr, timbre))
        y = np.concatenate(parts)
        if y.size / sr > MAX_AUDIO_S:
            raise MediaError("melody too long")
        return y
    if kind == "noise":
        rng = np.random.default_rng(int(seed))
        w = rng.standard_normal(int(round(seconds * sr)))
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(w.size, 1 / sr)
        spec[1:] /= np.sqrt(f[1:])                      # pink
        y = np.fft.irfft(spec, n=w.size)
        return 0.4 * y / max(1e-9, np.abs(y).max())
    if kind == "silence":
        return np.zeros(int(round(seconds * sr)))
    raise MediaError("kind must be tone, chord, melody, noise or silence")


def wav(samples: np.ndarray, sr: int = SR) -> bytes:
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


SOX_EFFECTS = ("fade", "reverb", "tempo", "pitch", "gain", "reverse", "echo", "lowpass", "highpass", "trim")


def sox_effects(data: bytes, effects: Iterable[dict]) -> bytes:
    """Apply SoX effects to a WAV: [{"effect": fade|reverb|tempo|pitch|gain|reverse|echo|lowpass|highpass|trim,
    ...numeric parameters}]. Each effect is one SoX run on the previous output (SoX misjudges a fade's stop
    position after an effect that changes the channel layout, so effects are not chained in one process)."""
    chain: list[list[str]] = []
    for k, e in enumerate(effects):
        if k >= 20:
            raise MediaError("at most 20 effects")
        name = e.get("effect") if isinstance(e, dict) else None
        if name not in SOX_EFFECTS:
            raise MediaError(f"effect {k}: one of {SOX_EFFECTS}")

        def num(key, lo, hi, default):
            try:
                v = float(e.get(key, default))
            except (TypeError, ValueError):
                raise MediaError(f"{name}.{key} must be a number") from None
            if not lo <= v <= hi:
                raise MediaError(f"{name}.{key} must be {lo}..{hi}")
            return f"{v:g}"
        if name == "fade":
            chain.append(["fade", "t", num("in_s", 0, 60, 0.5), "-0", num("out_s", 0, 60, 0.5)])
        elif name == "reverb":
            chain.append(["reverb", num("amount", 0, 100, 50), "channels", "1"])     # reverb makes 2 channels
        elif name == "tempo":
            chain.append(["tempo", num("factor", 0.25, 4, 1.25)])
        elif name == "pitch":
            chain.append(["pitch", num("cents", -2400, 2400, 300)])
        elif name == "gain":
            chain.append(["gain", num("db", -40, 20, -3)])
        elif name == "reverse":
            chain.append(["reverse"])
        elif name == "echo":
            chain.append(["echo", "0.8", "0.88", num("delay_ms", 1, 2000, 60), num("decay", 0, 1, 0.4)])
        elif name == "lowpass":
            chain.append(["lowpass", num("hz", 20, 20000, 2000)])
        elif name == "highpass":
            chain.append(["highpass", num("hz", 20, 20000, 200)])
        elif name == "trim":
            chain.append(["trim", num("start_s", 0, MAX_AUDIO_S, 0), num("length_s", 0.05, MAX_AUDIO_S, 5)])
    if not chain:
        raise MediaError("no effects given")
    for argv in chain:
        with _Work({"in.wav": data}) as w:
            run_tool([SOX, "-q", "-V0", "in.wav", "-b", "16", "out.wav", *argv], w.dir)
            data = w.read("out.wav")
    return data


# ---------------------------------------------------------------------------------------------------- video
def assemble_video(images: Sequence[bytes], audio: bytes | None = None, *, seconds_per_image: float = 2.0,
                   width: int = 640, height: int = 360, fps: int = 24) -> bytes:
    """Images (each shown for `seconds_per_image`, letterboxed to width x height) plus an optional WAV sound track
    -> H.264 + AAC MP4 (`+faststart`). The video lasts as long as the slideshow (audio is cut or padded)."""
    if not images:
        raise MediaError("a video needs at least one image")
    if len(images) > 600:
        raise MediaError("at most 600 images")
    spi = float(seconds_per_image)
    if not 0.1 <= spi <= 60:
        raise MediaError("seconds_per_image must be 0.1..60")
    width, height = _side(width, "width") // 2 * 2, _side(height, "height") // 2 * 2
    if not 1 <= int(fps) <= 60:
        raise MediaError("fps must be 1..60")
    total = spi * len(images)
    if total > MAX_VIDEO_S:
        raise MediaError(f"video longer than {MAX_VIDEO_S}s")
    from PIL import Image
    frames = {}
    for i, data in enumerate(images):
        im = open_image(data)
        im.thumbnail((width, height))
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
        frames[f"f{i:04d}.png"] = png(canvas)
    inputs = dict(frames)
    if audio is not None:
        inputs["a.wav"] = audio
    with _Work(inputs) as w:
        lst = "".join(f"file '{name}'\nduration {spi:g}\n" for name in sorted(frames)) + \
              f"file '{sorted(frames)[-1]}'\n"
        with open(os.path.join(w.dir, "list.txt"), "w") as fh:
            fh.write(lst)
        argv = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                "-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "0", "-i", "list.txt"]
        if audio is not None:
            argv += ["-protocol_whitelist", "file,pipe", "-i", "a.wav"]
        argv += ["-vf", f"fps={int(fps)},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                 "-x264-params", "threads=1", "-t", f"{total:g}"]
        argv += (["-c:a", "aac", "-b:a", "96k", "-af", "apad"] if audio is not None else ["-an"])
        argv += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1", "-movflags", "+faststart",
                 "out.mp4"]
        run_tool(argv, w.dir)
        return w.read("out.mp4")


def slides_video(slides: Sequence[bytes], durations: Sequence[float], audio: bytes | None = None, *,
                 fps: int = 5) -> bytes:
    """H.264/AAC MP4 of equally sized PNG slides, each shown for its own duration (FFmpeg's concat demuxer, jailed),
    with an optional audio track. The slides are handed to FFmpeg as they are (no decoding in this process): the
    caller renders them at the video's size (population creators, `crucible.social.popmedia`)."""
    if not slides or len(slides) != len(durations) or len(slides) > 600:
        raise MediaError("1 to 600 slides, one duration each")
    durs = [float(d) for d in durations]
    if min(durs) < 0.1 or max(durs) > 600:
        raise MediaError("slide durations must be 0.1..600 s")
    total = sum(durs)
    if total > MAX_VIDEO_S:
        raise MediaError(f"video longer than {MAX_VIDEO_S}s")
    if not 1 <= int(fps) <= 60:
        raise MediaError("fps must be 1..60")
    inputs = {f"s{i:04d}.png": bytes(b) for i, b in enumerate(slides)}
    if audio is not None:
        inputs["a.wav"] = audio
    with _Work(inputs) as w:
        names = sorted(n for n in inputs if n.endswith(".png"))
        lst = "".join(f"file '{n}'\nduration {d:.3f}\n" for n, d in zip(names, durs)) + f"file '{names[-1]}'\n"
        with open(os.path.join(w.dir, "list.txt"), "w") as fh:
            fh.write(lst)
        argv = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                "-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "0", "-i", "list.txt"]
        if audio is not None:
            argv += ["-protocol_whitelist", "file,pipe", "-i", "a.wav"]
        argv += ["-vf", f"fps={int(fps)},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                 "-x264-params", "threads=1", "-t", f"{total:.3f}"]
        argv += (["-c:a", "aac", "-b:a", "32k", "-af", "apad"] if audio is not None else ["-an"])
        argv += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1", "-movflags", "+faststart",
                 "out.mp4"]
        run_tool(argv, w.dir)
        return w.read("out.mp4")


def edit_video(data: bytes, *, start_s: float = 0.0, duration_s: float | None = None, width: int | None = None,
               height: int | None = None, mute: bool = False) -> bytes:
    """Trim (start, duration), rescale and optionally mute a video; re-encoded to H.264/AAC MP4."""
    start_s = float(start_s)
    if not 0 <= start_s <= MAX_VIDEO_S:
        raise MediaError("start_s out of range")
    if duration_s is not None and not 0.1 <= float(duration_s) <= MAX_VIDEO_S:
        raise MediaError("duration_s out of range")
    vf = ["format=yuv420p"]
    if width or height:
        w_ = _side(width or 2, "width") // 2 * 2 if width else -2
        h_ = _side(height or 2, "height") // 2 * 2 if height else -2
        vf.insert(0, f"scale={w_}:{h_}")
    with _Work({"in.mp4": data}) as w:
        argv = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                "-protocol_whitelist", "file,pipe", "-ss", f"{start_s:g}", "-i", "in.mp4"]
        if duration_s is not None:
            argv += ["-t", f"{float(duration_s):g}"]
        argv += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-x264-params",
                 "threads=1"]
        argv += ["-an"] if mute else ["-c:a", "aac", "-b:a", "96k"]
        argv += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1", "-movflags", "+faststart",
                 "out.mp4"]
        run_tool(argv, w.dir)
        return w.read("out.mp4")


def video_frame(data: bytes, at_s: float = 0.5, width: int = 480) -> bytes:
    """One frame of a video as a PNG (a thumbnail)."""
    with _Work({"in.mp4": data}) as w:
        run_tool([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                  "-protocol_whitelist", "file,pipe", "-ss", f"{max(0.0, float(at_s)):g}", "-i", "in.mp4",
                  "-frames:v", "1", "-vf", f"scale={_side(width, 'width')}:-2", "out.png"], w.dir)
        return w.read("out.png")


def waveform(data: bytes, width: int = 800, height: int = 200) -> bytes:
    """A waveform picture of a WAV (min/max envelope per pixel column)."""
    from PIL import Image, ImageDraw
    try:
        with wave.open(io.BytesIO(data)) as wf:
            n, ch, sw = wf.getnframes(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(n)
    except (wave.Error, EOFError) as e:
        raise MediaError(f"not a WAV: {e}") from None
    if sw != 2:
        raise MediaError("waveforms are drawn for 16-bit PCM")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, ch).mean(axis=1) / 32768.0
    width, height = _side(width, "width"), _side(height, "height")
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    if x.size:
        cols = np.array_split(x, width)
        mid = height / 2
        for i, c in enumerate(cols):
            if c.size:
                d.line([(i, mid - c.max() * mid), (i, mid - c.min() * mid)], fill=(31, 119, 180))
    return png(im)


# ---------------------------------------------------------------------------------------------------- documents
def _pdf_escape(s: str) -> str:
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_document(title: str, body: str, *, author: str = "", page_chars: int = 90, lines_per_page: int = 48) -> bytes:
    """A real PDF: title page header, wrapped Helvetica text, one content stream per page (uncompressed, so its
    text can be extracted). Deterministic (no dates)."""
    if len(body) > 200_000:
        raise MediaError("document text longer than 200000 characters")
    lines: list[str] = []
    for para in str(body).split("\n"):
        words, cur = para.split(), ""
        for wd in words:
            if len(cur) + len(wd) + 1 > page_chars and cur:
                lines.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
    pages = [lines[i:i + lines_per_page] for i in range(0, max(1, len(lines)), lines_per_page)] or [[]]
    objs: list[bytes] = []

    def add(b: bytes) -> int:
        objs.append(b)
        return len(objs)
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bold_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    pages_id = add(b"")                 # filled below
    page_ids = []
    for k, pl in enumerate(pages):
        ops = ["BT"]
        y = 800
        if k == 0 and title:
            ops += [f"/F2 18 Tf 50 {y} Td ({_pdf_escape(title[:120])}) Tj", f"0 -28 Td"]
            y -= 28
            ops += ["/F1 11 Tf"]
        else:
            ops += [f"/F1 11 Tf 50 {y} Td"]
        for ln in pl:
            ops.append(f"({_pdf_escape(ln)}) Tj 0 -15 Td")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        cid = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] /Contents %d 0 R "
                            b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> >>" % (pages_id, cid, font_id,
                                                                                         bold_id)))
    objs[pages_id - 1] = (b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids) +
                          b"] /Count %d >>" % len(page_ids))
    info_id = add(b"<< /Title (" + _pdf_escape(title[:200]).encode("latin-1") + b") /Author (" +
                  _pdf_escape(author[:100]).encode("latin-1") + b") /Producer (crucible-studio) >>")
    cat_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offs = []
    for i, ob in enumerate(objs, start=1):
        offs.append(out.tell())
        out.write(b"%d 0 obj\n" % i + ob + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for o in offs:
        out.write(b"%010d 00000 n \n" % o)
    out.write(b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(objs) + 1, cat_id, info_id, xref))
    return out.getvalue()
