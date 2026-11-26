"""Content artifact formats (MASTER_SPEC Part 4.2): what an artifact IS, checked from its bytes.

Ten formats, each with a media type, a deterministic validator that inspects the actual bytes and returns the
facts any reader could establish (dimensions, duration, page count, entry point, ...), and a text extractor used by
search and by encoders. No format carries a quality, relevance or virality figure: such a number would be the
author's claim, and the world never accepts one.

    text        UTF-8 text (text/plain, text/markdown)
    html        an HTML document (text/html)
    image       PNG / JPEG / GIF / WebP raster image (decoded with Pillow under a pixel limit)
    audio       RIFF/WAVE PCM audio (duration from the header) or other audio identified by FFmpeg
    video       MP4 / QuickTime / WebM video (duration and streams from the container)
    app         an executable application: JSON bundle {"entry": "module:function", "files": {...}} run in the
                4.0 program sandbox (crucible.sandbox)
    page        an interactive web page: HTML with a script or a form
    document    a PDF document (application/pdf)
    course      a digital educational product: JSON {"title", "lessons": [...], "exercises": [...]}
    package     a software package: JSON {"name", "version", "files", "deps"?, "tests"?, "entry"?}

`inspect(data, fmt, mime)` -> {"format", "mime", "size", ...facts}; raises FormatError when the bytes are not
what they claim to be. Everything here is pure, deterministic Python (no external process), so validation can run
inside a kernel handler and give identical results during regeneration.
"""
from __future__ import annotations

import io
import json
import re
import struct
import wave
from html.parser import HTMLParser

FORMATS = ("text", "html", "image", "audio", "video", "app", "page", "document", "course", "package")
MIMES = {
    "text": ("text/plain", "text/markdown"),
    "html": ("text/html",),
    "image": ("image/png", "image/jpeg", "image/gif", "image/webp"),
    "audio": ("audio/wav", "audio/x-wav", "audio/wave"),
    "video": ("video/mp4", "video/quicktime", "video/webm"),
    "app": ("application/vnd.crucible.app+json",),
    "page": ("text/html",),
    "document": ("application/pdf",),
    "course": ("application/vnd.crucible.course+json",),
    "package": ("application/vnd.crucible.package+json",),
}
DOC = {
    "text": "plain or markdown text",
    "html": "an HTML document",
    "image": "a raster image (PNG, JPEG, GIF, WebP)",
    "audio": "PCM audio in a WAV container",
    "video": "an MP4/QuickTime or WebM video",
    "app": "an executable application run in the program sandbox",
    "page": "an interactive web page (HTML with a script or a form)",
    "document": "a PDF document",
    "course": "a digital educational product (lessons and exercises)",
    "package": "a software package (files, version, dependencies, tests)",
}
MAX_PIXELS = 40_000_000
MAX_TEXT = 200_000          # characters of extracted text kept for search / encoders
ENTRY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*")
PKG_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{1,40}")
VERSION_RE = re.compile(r"\d{1,4}\.\d{1,4}\.\d{1,4}")
PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,120}")
MAX_BUNDLE_FILES = 200


class FormatError(ValueError):
    """The bytes are not a valid artifact of the claimed format."""


# ---------------------------------------------------------------------------------------------------- helpers
def _utf8(data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise FormatError(f"{what} is not valid UTF-8") from None


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "head", "title", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.tags: dict[str, int] = {}
        self._stack: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1
        self._stack.append(tag)
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v[:300])
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        while self._stack:
            t = self._stack.pop()
            if t == tag:
                break

    def handle_data(self, data):
        cur = self._stack[-1] if self._stack else ""
        if cur == "title":
            self.title.append(data)
        elif not (set(self._stack) & self.SKIP):
            self.parts.append(data)


def html_text(html: str) -> dict:
    p = _HTMLText()
    try:
        p.feed(html)
        p.close()
    except Exception as e:  # noqa: BLE001 - html.parser raises plain exceptions on pathological input
        raise FormatError(f"unparseable HTML: {e}") from None
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return {"text": text[:MAX_TEXT], "title": " ".join("".join(p.title).split())[:200], "tags": p.tags,
            "links": p.links[:100]}


def _json_obj(data: bytes, what: str) -> dict:
    try:
        d = json.loads(_utf8(data, what))
    except json.JSONDecodeError as e:
        raise FormatError(f"{what} is not JSON: {e.msg}") from None
    if not isinstance(d, dict):
        raise FormatError(f"{what} must be a JSON object")
    return d


def _files(d: dict, what: str) -> dict[str, str]:
    files = d.get("files")
    if not isinstance(files, dict) or not files:
        raise FormatError(f"{what} needs a non-empty 'files' object")
    if len(files) > MAX_BUNDLE_FILES:
        raise FormatError(f"{what} has more than {MAX_BUNDLE_FILES} files")
    for path, src in files.items():
        if not isinstance(path, str) or not PATH_RE.fullmatch(path) or ".." in path.split("/"):
            raise FormatError(f"{what}: bad file path {str(path)[:60]!r}")
        if not isinstance(src, str):
            raise FormatError(f"{what}: file {path} must be text")
    return files


# ---------------------------------------------------------------------------------------------------- media facts
def image_info(data: bytes) -> dict:
    sig = data[:12]
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt = "png"
    elif sig.startswith(b"\xff\xd8\xff"):
        fmt = "jpeg"
    elif sig[:6] in (b"GIF87a", b"GIF89a"):
        fmt = "gif"
    elif sig[:4] == b"RIFF" and sig[8:12] == b"WEBP":
        fmt = "webp"
    else:
        raise FormatError("not a PNG, JPEG, GIF or WebP image")
    # every pixel is decoded (a truncated or corrupt file fails), in the jail: participant bytes never reach a native
    # decoder inside the world process (crucible.sandbox.isolation.decode_image; Gate W.1 / SP-16b)
    from ..sandbox import isolation
    try:
        meta = isolation.decode_image(bytes(data), max_pixels=MAX_PIXELS)
    except isolation.JailUnavailable as e:
        raise FormatError(f"untrusted images cannot be decoded here: {e}") from None
    if not meta.get("ok"):
        err = str(meta.get("error", ""))
        if "pixel limit" in err or "DecompressionBomb" in err:
            raise FormatError(f"image exceeds {MAX_PIXELS} pixels: {err[:160]}")
        raise FormatError(f"image does not decode: {err[:200]}")
    w, h = (int(x) for x in meta["size"])
    return {"image_format": fmt, "width": w, "height": h, "mode": meta["source_mode"], "frames": int(meta["frames"])}


def wav_info(data: bytes) -> dict:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise FormatError("not a RIFF/WAVE file")
    try:
        with wave.open(io.BytesIO(data)) as wf:
            sr, ch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            got = len(wf.readframes(n)) // max(1, ch * sw)
    except (wave.Error, EOFError, struct.error) as e:
        raise FormatError(f"bad WAV: {e}") from None
    if sr <= 0 or ch <= 0 or got <= 0:
        raise FormatError("WAV has no audio")
    return {"audio_format": "wav", "sample_rate": int(sr), "channels": int(ch), "sample_bits": 8 * int(sw),
            "frames": int(got), "duration_s": round(got / sr, 3)}


def _mp4_boxes(data: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", data[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            return
        yield typ.decode("latin-1"), i + hdr, i + size
        i += size


def mp4_info(data: bytes) -> dict:
    """Duration and track kinds of an ISO-BMFF (MP4/QuickTime) file, read from its boxes (moov/mvhd, trak/mdia/hdlr)."""
    top = {t: (a, b) for t, a, b in _mp4_boxes(data, 0, len(data))}
    if "ftyp" not in top or "moov" not in top:
        raise FormatError("not an MP4/QuickTime file (no ftyp/moov box)")
    a, b = top["moov"]
    duration = None
    tracks: list[str] = []
    for t, s, e in _mp4_boxes(data, a, b):
        if t == "mvhd":
            ver = data[s]
            if ver == 1:
                timescale, dur = struct.unpack(">IQ", data[s + 20:s + 32])
            else:
                timescale, dur = struct.unpack(">II", data[s + 12:s + 20])
            if timescale:
                duration = dur / timescale
        elif t == "trak":
            for t2, s2, e2 in _mp4_boxes(data, s, e):
                if t2 != "mdia":
                    continue
                for t3, s3, e3 in _mp4_boxes(data, s2, e2):
                    if t3 == "hdlr" and e3 - s3 >= 12:
                        tracks.append(data[s3 + 8:s3 + 12].decode("latin-1"))
    if "mdat" not in top:
        raise FormatError("MP4 has no media data (mdat)")
    if duration is None or duration <= 0:
        raise FormatError("MP4 has no duration")
    if "vide" not in tracks:
        raise FormatError("MP4 has no video track")
    return {"video_format": "mp4", "duration_s": round(float(duration), 3), "tracks": sorted(set(tracks)),
            "has_audio": "soun" in tracks}


def webm_info(data: bytes) -> dict:
    if data[:4] != b"\x1a\x45\xdf\xa3":
        raise FormatError("not a WebM/Matroska file")
    # the EBML Duration element (0x4489) holds a float in timecode-scale units (default 1 ms)
    k = data.find(b"\x44\x89")
    if k < 0 or k + 3 > len(data):
        raise FormatError("WebM without a duration element")
    n = data[k + 2] & 0x7F if data[k + 2] & 0x80 else None
    if n not in (4, 8):
        raise FormatError("WebM duration element has an unexpected size")
    raw = data[k + 3:k + 3 + n]
    dur = struct.unpack(">f" if n == 4 else ">d", raw)[0] / 1000.0
    if not dur > 0:
        raise FormatError("WebM has no duration")
    return {"video_format": "webm", "duration_s": round(float(dur), 3), "tracks": ["vide"], "has_audio": None}


def pdf_info(data: bytes) -> dict:
    if not data.startswith(b"%PDF-"):
        raise FormatError("not a PDF (no %PDF- header)")
    if b"%%EOF" not in data[-2048:]:
        raise FormatError("truncated PDF (no %%EOF trailer)")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    if pages <= 0:
        raise FormatError("PDF without pages")
    return {"pdf_version": data[5:8].decode("latin-1"), "pages": pages}


# ---------------------------------------------------------------------------------------------------- validators
def inspect(data: bytes, fmt: str, mime: str) -> dict:
    """Facts about `data` as an artifact of format `fmt` (raises FormatError if it is not one)."""
    if fmt not in FORMATS:
        raise FormatError(f"unknown format {fmt!r} (formats: {', '.join(FORMATS)})")
    if mime not in MIMES[fmt]:
        raise FormatError(f"format {fmt} needs media type {' or '.join(MIMES[fmt])}, not {mime}")
    if not data:
        raise FormatError("empty artifact")
    out: dict = {"format": fmt, "mime": mime, "size": len(data)}
    if fmt == "text":
        t = _utf8(data, "text")
        out.update(chars=len(t), words=len(t.split()), lines=t.count("\n") + 1)
    elif fmt in ("html", "page"):
        h = html_text(_utf8(data, "HTML"))
        if not any(t in h["tags"] for t in ("html", "body", "p", "div", "h1", "h2", "article", "main", "form",
                                             "section", "span", "ul", "table")):
            raise FormatError("HTML without any document element")
        out.update(chars=len(h["text"]), words=len(h["text"].split()), title=h["title"], links=len(h["links"]))
        interactive = bool(h["tags"].get("script") or h["tags"].get("form") or h["tags"].get("button")
                           or h["tags"].get("input"))
        if fmt == "page" and not interactive:
            raise FormatError("an interactive page needs a script, a form, a button or an input")
        out["interactive"] = interactive
    elif fmt == "image":
        out.update(image_info(data))
    elif fmt == "audio":
        out.update(wav_info(data))
    elif fmt == "video":
        out.update(webm_info(data) if mime == "video/webm" else mp4_info(data))
    elif fmt == "document":
        out.update(pdf_info(data))
    elif fmt == "app":
        d = _json_obj(data, "app bundle")
        files = _files(d, "app bundle")
        entry = d.get("entry")
        if not isinstance(entry, str) or not ENTRY_RE.fullmatch(entry):
            raise FormatError("app bundle needs 'entry': 'module:function'")
        mod = entry.split(":", 1)[0].replace(".", "/") + ".py"
        if mod not in files:
            raise FormatError(f"app entry module {mod} is not among its files")
        out.update(entry=entry, files=len(files), lines=sum(s.count("\n") + 1 for s in files.values()),
                   description=str(d.get("description", ""))[:500])
    elif fmt == "course":
        d = _json_obj(data, "course")
        lessons, exercises = d.get("lessons"), d.get("exercises", [])
        if not isinstance(d.get("title"), str) or not d["title"].strip():
            raise FormatError("course needs a title")
        if not isinstance(lessons, list) or not lessons:
            raise FormatError("course needs at least one lesson")
        for i, ls in enumerate(lessons):
            if not isinstance(ls, dict) or not isinstance(ls.get("title"), str) or not isinstance(ls.get("body"), str):
                raise FormatError(f"lesson {i} needs a title and a body")
        if not isinstance(exercises, list):
            raise FormatError("exercises must be a list")
        for i, ex in enumerate(exercises):
            if not isinstance(ex, dict) or not isinstance(ex.get("question"), str):
                raise FormatError(f"exercise {i} needs a question")
        out.update(title=d["title"][:200], lessons=len(lessons), exercises=len(exercises),
                   words=sum(len(ls["body"].split()) for ls in lessons))
    elif fmt == "package":
        d = _json_obj(data, "package")
        name, ver = d.get("name"), d.get("version")
        if not isinstance(name, str) or not PKG_NAME_RE.fullmatch(name):
            raise FormatError("package needs a lowercase 'name'")
        if not isinstance(ver, str) or not VERSION_RE.fullmatch(ver):
            raise FormatError("package needs a 'version' like 1.0.0")
        files = _files(d, "package")
        deps = d.get("deps", [])
        if not isinstance(deps, list) or not all(isinstance(x, str) and len(x) <= 80 for x in deps) or len(deps) > 20:
            raise FormatError("deps must be a list of at most 20 package names")
        tests = [p for p in files if p.startswith("tests/") and p.endswith(".py")]
        out.update(name=name, version=ver, files=len(files), deps=sorted(set(deps)), tests=len(tests),
                   entry=d.get("entry") if isinstance(d.get("entry"), str) else None)
    return out


def extract_text(data: bytes, fmt: str, mime: str, *, limit: int = MAX_TEXT) -> str:
    """The searchable/encodable text of an artifact (empty for pure media)."""
    try:
        if fmt == "text":
            return _utf8(data, "text")[:limit]
        if fmt in ("html", "page"):
            h = html_text(_utf8(data, "HTML"))
            return ((h["title"] + "\n") if h["title"] else "") + h["text"][:limit]
        if fmt == "course":
            d = _json_obj(data, "course")
            parts = [d.get("title", "")] + [f"{ls.get('title', '')}\n{ls.get('body', '')}" for ls in d.get("lessons", [])]
            parts += [ex.get("question", "") for ex in d.get("exercises", []) if isinstance(ex, dict)]
            return "\n".join(str(p) for p in parts)[:limit]
        if fmt in ("app", "package"):
            d = _json_obj(data, fmt)
            readme = d.get("files", {}).get("README.md", "") if isinstance(d.get("files"), dict) else ""
            return "\n".join(str(x) for x in (d.get("name", ""), d.get("description", ""), readme))[:limit]
        if fmt == "document":
            # text drawn by our own PDF writer is kept in the document's /Subject and the uncompressed page streams
            out = re.findall(rb"\(((?:[^()\\]|\\.){1,2000})\)\s*Tj", data)
            txt = " ".join(re.sub(r"\\(.)", r"\1", x.decode("latin-1")) for x in out)
            return txt[:limit]
    except FormatError:
        return ""
    return ""


def formats_doc() -> dict:
    return {f: {"mime": list(MIMES[f]), "doc": DOC[f]} for f in FORMATS}
