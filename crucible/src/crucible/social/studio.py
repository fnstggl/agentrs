"""The creator studio (kernel module `studio`, host `studio`): sandboxed tools to create, edit and render actual
artifacts of the ten formats of MASTER_SPEC Part 4.2, run executable applications and package tests in the program
sandbox, and inspect artifacts. Publishing, updating, removing and performance inspection are the catalogue's
tools (`content.publish`, `content.update`, `content.remove`, `content.inspect`).

    create   studio.write (text, markdown, HTML, PDF document), studio.page (interactive web page), studio.image
             (text card or vector scene), studio.audio (tones, chords, melodies, noise), studio.video (images and a
             sound track assembled by FFmpeg), studio.app (executable application), studio.package (software
             package), studio.course (educational product)
    edit     studio.edit (text-like artifacts: replace/append/prepend), studio.edit_image (Pillow operations),
             studio.edit_audio (SoX effects), studio.revise (new version of an app/package/course bundle)
    render   studio.preview (any text-like artifact rendered to an image), studio.video
    run      studio.run (an app's entry point in the program sandbox), studio.test (a package's tests there)
    inspect  studio.probe (format facts measured from the bytes), studio.drafts (your artifacts)

Every produced artifact belongs to the caller (or an organisation it may publish for), records its inputs as
provenance (`parents`), and is produced once: the bytes are recorded (`render.recorded_bytes`), so replay and
regeneration never re-run FFmpeg, SoX or Pillow. Programs run in the 4.0 program sandbox (`crucible.sandbox`:
separate process, uid nobody, Landlock, seccomp, rlimits, wall-clock timeout, no network, numpy only) and their
outcomes are recorded by the sandbox bridge. Media tools run as rlimited child processes without network inputs.
"""
from __future__ import annotations

import html as _html
import json
import re

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, NotFound, ProtocolError, ServiceUnavailable
from ..core.kernel import Ctx, Module, endpoint, register_module
from ..substrate import artifacts
from . import formats as F
from . import render as R

_REF = S.s_str("artifact reference sha256:<64 hex>", maxLength=71)
_FILES = {"type": "object", "description": "path -> source text", "additionalProperties": {"type": "string",
                                                                                            "maxLength": 65536}}
_OWNER = S.s_str("store it for an organisation you administer (default: you)", maxLength=40)
MAX_FILES_BYTES = 1 << 20


def _json_bytes(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _html_doc(title: str, body_html: str, head: str = "") -> str:
    return (f"<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{_html.escape(title)}</title>{head}</head>"
            f"<body>\n<h1>{_html.escape(title)}</h1>\n{body_html}\n</body></html>\n")


def _markdown_html(md: str) -> str:
    """A small markdown subset to HTML (headings, paragraphs, lists, emphasis, code, links); text is escaped."""
    out, para, lst = [], [], []

    def inline(s):
        s = _html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+|/[^)\s]*)\)", r'<a href="\2">\1</a>', s)
        return s

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if lst:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in lst) + "</ul>")
            lst.clear()
    for line in md.split("\n"):
        s = line.rstrip()
        m = re.match(r"(#{1,4})\s+(.*)", s)
        if m:
            flush()
            k = len(m.group(1)) + 1
            out.append(f"<h{k}>{inline(m.group(2))}</h{k}>")
        elif re.match(r"\s*[-*]\s+", s):
            if para:
                out.append("<p>" + inline(" ".join(para)) + "</p>")
                para.clear()
            lst.append(re.sub(r"^\s*[-*]\s+", "", s))
        elif not s.strip():
            flush()
        else:
            if lst:
                flush()
            para.append(s.strip())
    flush()
    return "\n".join(out)


@register_module
class Studio(Module):
    name = "studio"
    host = "studio"
    version = "1"
    depends = ("identity", "artifacts")

    # ------------------------------------------------------------------ helpers
    def _owner(self, ctx: Ctx, owner: str | None) -> str:
        if not owner:
            return ctx.actor
        from ..services.identity import ident
        o = ident(ctx.world).resolve_handle(ctx.state, owner)
        if o not in ctx.world.principals_of(ctx.actor):
            raise Forbidden(f"you cannot store artifacts for {o}")
        return o

    def _read(self, ctx: Ctx, ref: str, formats: tuple[str, ...] | None = None) -> tuple[bytes, dict]:
        artifacts.parse_ref(ref)
        if not artifacts.readable(ctx.world, ctx.actor, ref):
            raise NotFound(f"no artifact {ref}")
        a = ctx.state.tables["artifacts"][ref]
        if formats is not None and a["kind"] not in formats:
            raise InvalidAction(f"this tool works on {', '.join(formats)} artifacts, not {a['kind']}")
        return artifacts.get(ctx.world, ref), dict(a)

    def _store(self, ctx: Ctx, tool: str, fmt: str, mime: str, owner: str, inputs: dict, produce, *,
               parents=(), title: str = "") -> dict:
        def checked():
            data = produce()
            try:
                facts = F.inspect(data, fmt, mime)
            except F.FormatError as e:
                raise R.MediaError(f"{tool} produced an invalid {fmt}: {e}") from None
            del facts
            return data
        meta = {"title": str(title)[:200], "tool": tool} if title else {"tool": tool}
        ref, data = R.recorded_bytes(ctx, tool, {"tool": tool, "inputs": inputs, "parents": sorted(parents)},
                                     checked, mime=mime, owner=owner, parents=parents, meta=meta, akind=fmt)
        facts = F.inspect(data, fmt, mime)
        return {"ref": ref, "format": fmt, "mime": mime, "size": len(data), "facts": facts}

    def _pool(self, world):
        code = world.modules.get("code")
        if code is not None:
            return code.sandbox_pool(world)
        pool = world.runtime.get("studio.sandbox_pool")
        if pool is None:
            from ..sandbox.runner import SandboxConfig, WorkerPool, capabilities
            caps = capabilities()
            p = world.spec.params
            if (caps["landlock_abi"] < 1 or not caps["seccomp"]) and not p.get("allow_weak_sandbox"):
                raise ServiceUnavailable("program execution is disabled: Landlock/seccomp unavailable on this host")
            # the same metering as services.code.sandbox_pool (A.json, D-20): work measured per world lineage, the
            # deterministic memo shared by every process of the machine (a rerun, resume or replication in a new
            # process bills identical computations alike), one tenant per world
            from ..sandbox import metering
            det = p.get("cost_model") == "deterministic"
            memo = metering.default_memo_path() if det and p.get("meter_memo", "machine") == "machine" else None
            cfg = SandboxConfig(mem_mb=int(p.get("sandbox_mem_mb", 1024)), job_wall_s=float(p.get("sandbox_job_s", 10)),
                                allow_weak=bool(p.get("allow_weak_sandbox", False)), count_lines=det,
                                meter_scope=str(p.get("meter_lineage") or world.spec.hash()[:16]), memo_path=memo,
                                tenant=f"world:{world.spec.hash()[:16]}")
            pool = world.runtime["studio.sandbox_pool"] = WorkerPool(int(p.get("sandbox_max_workers", 24)), cfg)
        return pool

    def close(self):
        w = self.world
        if w is not None and "studio.sandbox_pool" in w.runtime:
            w.runtime.pop("studio.sandbox_pool").close()

    # ------------------------------------------------------------------ create: text, html, documents, pages
    @endpoint("write", S.obj({"format": S.s_enum(["text", "markdown", "html", "document"],
                                                 "text/markdown are plain text, html an HTML document (markdown "
                                                 "body converted unless html is given), document a PDF"),
                              "title": S.s_str(minLength=1, maxLength=200),
                              "body": S.s_str("the text (markdown for html/document)", maxLength=200_000),
                              "html": S.s_str("raw HTML body for format html (escaped nowhere: you write it)",
                                              maxLength=200_000),
                              "owner": _OWNER}, ["format", "title"]), cost_s=60)
    def ep_write(self, ctx, format, title, body="", html=None, owner=None):
        """Write a text, markdown, HTML or PDF artifact. Returns {ref, format, mime, size, facts}."""
        owner = self._owner(ctx, owner)
        if format in ("text", "markdown"):
            if not body:
                raise InvalidAction("body is required")
            mime = "text/markdown" if format == "markdown" else "text/plain"
            return self._store(ctx, "studio.write", "text", mime, owner, {"format": format, "body": body},
                               lambda: body.encode("utf-8"), title=title)
        if format == "html":
            if not body and not html:
                raise InvalidAction("body or html is required")
            doc = html if html and re.search(r"<html[\s>]", html, re.I) else \
                _html_doc(title, html if html is not None else _markdown_html(body))
            return self._store(ctx, "studio.write", "html", "text/html", owner, {"format": format, "doc": doc},
                               lambda: doc.encode("utf-8"), title=title)
        if not body:
            raise InvalidAction("body is required")
        return self._store(ctx, "studio.write", "document", "application/pdf", owner,
                           {"format": format, "title": title, "body": body, "author": owner},
                           lambda: R.pdf_document(title, body, author=owner), title=title)

    @endpoint("page", S.obj({"title": S.s_str(minLength=1, maxLength=200),
                             "html": S.s_str("page body HTML", maxLength=200_000),
                             "script": S.s_str("inline JavaScript run by browsers that open the page", maxLength=100_000),
                             "style": S.s_str("inline CSS", maxLength=50_000), "owner": _OWNER},
                            ["title", "html"]), cost_s=60)
    def ep_page(self, ctx, title, html, script="", style="", owner=None):
        """Create an interactive web page (HTML with a script, a form, buttons or inputs)."""
        owner = self._owner(ctx, owner)
        head = (f"<style>{style}</style>" if style else "")
        doc = _html_doc(title, html + (f"\n<script>{script}</script>" if script else ""), head)
        return self._store(ctx, "studio.page", "page", "text/html", owner, {"doc": doc}, lambda: doc.encode("utf-8"),
                           title=title)

    # ------------------------------------------------------------------ create/edit: images
    @endpoint("image", S.obj({"text": S.s_str("text for a text card", maxLength=20_000),
                              "title": S.s_str(maxLength=200),
                              "shapes": S.s_arr(S.s_any_obj("{shape: rect|ellipse|line|polygon|text, xy, fill, "
                                                            "outline, width, text, size}"), maxItems=500),
                              "width": S.s_int(minimum=1, maximum=R.MAX_SIDE, default=800),
                              "height": S.s_int(minimum=1, maximum=R.MAX_SIDE),
                              "font_size": S.s_int(minimum=6, maximum=200, default=32),
                              "color": S.s_str("text colour #rrggbb or a name", maxLength=20),
                              "background": S.s_str("#rrggbb or a name", maxLength=20), "owner": _OWNER}),
              cost_s=60)
    def ep_image(self, ctx, text=None, title=None, shapes=None, width=800, height=None, font_size=32, color=None,
                 background=None, owner=None):
        """Draw a PNG image: a text card (text, optional title) or a vector scene (shapes)."""
        owner = self._owner(ctx, owner)
        if shapes:
            inp = {"shapes": shapes, "width": width, "height": height or width, "bg": background}
            make = lambda: R.draw_scene(shapes, width=width, height=height or width, bg=background or "#ffffff")  # noqa: E731
        elif text:
            inp = {"text": text, "title": title, "width": width, "height": height, "fs": font_size, "fg": color,
                   "bg": background}
            make = lambda: R.text_card(text, width=width, height=height, font_size=font_size,  # noqa: E731
                                       fg=color or "#111111", bg=background or "#ffffff", title=title)
        else:
            raise InvalidAction("give text (a card) or shapes (a scene)")
        return self._store(ctx, "studio.image", "image", "image/png", owner, inp, make, title=title or "")

    @endpoint("edit_image", S.obj({"ref": _REF, "ops": S.s_arr(S.s_any_obj(
        "{op: resize|crop|rotate|flip|mirror|grayscale|blur|sharpen|brightness|contrast|text|thumbnail|border|invert,"
        " ...}"), minItems=1, maxItems=50), "owner": _OWNER}, ["ref", "ops"]), cost_s=30)
    def ep_edit_image(self, ctx, ref, ops, owner=None):
        """Process an image you can read; the result is a new PNG artifact derived from it."""
        owner = self._owner(ctx, owner)
        data, _ = self._read(ctx, ref, ("image",))
        return self._store(ctx, "studio.edit_image", "image", "image/png", owner, {"ops": ops},
                           lambda: R.process_image(data, ops), parents=[ref])

    # ------------------------------------------------------------------ create/edit: audio, video
    @endpoint("audio", S.obj({"kind": S.s_enum(["tone", "chord", "melody", "noise", "silence"]),
                              "seconds": S.s_num(minimum=0.05, maximum=R.MAX_AUDIO_S, default=3.0),
                              "freq": S.s_num(minimum=20, maximum=12000, default=440.0),
                              "notes": S.s_arr({"type": ["string", "array", "null"]},
                                               "notes like C4, F#3, Bb5; null = rest; a list = chord", maxItems=2000),
                              "bpm": S.s_num(minimum=20, maximum=400, default=120.0),
                              "timbre": S.s_enum(["sine", "organ", "bright", "soft"]),
                              "seed": S.s_int(minimum=0, maximum=2**31 - 1, default=0),
                              "effects": S.s_arr(S.s_any_obj("{effect: fade|reverb|tempo|pitch|gain|reverse|echo|"
                                                             "lowpass|highpass|trim, ...}"), maxItems=20),
                              "title": S.s_str(maxLength=200), "owner": _OWNER}, ["kind"]), cost_s=60)
    def ep_audio(self, ctx, kind, seconds=3.0, freq=440.0, notes=None, bpm=120.0, timbre="organ", seed=0,
                 effects=None, title="", owner=None):
        """Synthesise audio (PCM WAV), optionally through SoX effects."""
        owner = self._owner(ctx, owner)

        def make():
            data = R.wav(R.synth(kind, seconds=seconds, freq=freq, notes=notes, bpm=bpm, timbre=timbre or "organ",
                                 seed=seed))
            return R.sox_effects(data, effects) if effects else data
        return self._store(ctx, "studio.audio", "audio", "audio/wav", owner,
                           {"kind": kind, "s": seconds, "f": freq, "notes": notes, "bpm": bpm, "timbre": timbre,
                            "seed": seed, "effects": effects}, make, title=title)

    @endpoint("edit_audio", S.obj({"ref": _REF, "effects": S.s_arr(S.s_any_obj(), minItems=1, maxItems=20),
                                   "owner": _OWNER}, ["ref", "effects"]), cost_s=30)
    def ep_edit_audio(self, ctx, ref, effects, owner=None):
        """Apply SoX effects (fade, reverb, tempo, pitch, gain, reverse, echo, lowpass, highpass, trim) to a WAV."""
        owner = self._owner(ctx, owner)
        data, _ = self._read(ctx, ref, ("audio",))
        return self._store(ctx, "studio.edit_audio", "audio", "audio/wav", owner, {"effects": effects},
                           lambda: R.sox_effects(data, effects), parents=[ref])

    @endpoint("video", S.obj({"images": S.s_arr(_REF, "image artifacts, in order", minItems=1, maxItems=600),
                              "audio": S.s_str("optional WAV sound track", maxLength=71),
                              "seconds_per_image": S.s_num(minimum=0.1, maximum=60, default=2.0),
                              "width": S.s_int(minimum=16, maximum=1920, default=640),
                              "height": S.s_int(minimum=16, maximum=1920, default=360),
                              "fps": S.s_int(minimum=1, maximum=60, default=24),
                              "title": S.s_str(maxLength=200), "owner": _OWNER}, ["images"]), cost_s=300)
    def ep_video(self, ctx, images, audio=None, seconds_per_image=2.0, width=640, height=360, fps=24, title="",
                 owner=None):
        """Assemble images (and an optional sound track) into an H.264/AAC MP4 video with FFmpeg."""
        owner = self._owner(ctx, owner)
        imgs = [self._read(ctx, r, ("image",))[0] for r in images]
        snd = self._read(ctx, audio, ("audio",))[0] if audio else None
        parents = sorted(set(images) | ({audio} if audio else set()))
        return self._store(ctx, "studio.video", "video", "video/mp4", owner,
                           {"images": list(images), "audio": audio, "spi": seconds_per_image, "w": width, "h": height,
                            "fps": fps},
                           lambda: R.assemble_video(imgs, snd, seconds_per_image=seconds_per_image, width=width,
                                                    height=height, fps=fps), parents=parents, title=title)

    @endpoint("edit_video", S.obj({"ref": _REF, "start_s": S.s_num(minimum=0, maximum=R.MAX_VIDEO_S, default=0.0),
                                   "duration_s": S.s_num(minimum=0.1, maximum=R.MAX_VIDEO_S),
                                   "width": S.s_int(minimum=16, maximum=1920), "height": S.s_int(minimum=16,
                                                                                                 maximum=1920),
                                   "mute": S.s_bool(default=False), "owner": _OWNER}, ["ref"]), cost_s=120)
    def ep_edit_video(self, ctx, ref, start_s=0.0, duration_s=None, width=None, height=None, mute=False, owner=None):
        """Trim, rescale or mute a video (FFmpeg); the result is a new MP4 derived from it."""
        owner = self._owner(ctx, owner)
        data, _ = self._read(ctx, ref, ("video",))
        return self._store(ctx, "studio.edit_video", "video", "video/mp4", owner,
                           {"start": start_s, "dur": duration_s, "w": width, "h": height, "mute": mute},
                           lambda: R.edit_video(data, start_s=start_s, duration_s=duration_s, width=width,
                                                height=height, mute=mute), parents=[ref])

    # ------------------------------------------------------------------ create: apps, packages, courses
    def _bundle_size(self, files: dict) -> None:
        if sum(len(v.encode("utf-8")) for v in files.values()) > MAX_FILES_BYTES:
            raise InvalidAction(f"files larger than {MAX_FILES_BYTES} bytes in total")

    @endpoint("app", S.obj({"name": S.s_str(pattern=F.PKG_NAME_RE.pattern, maxLength=41),
                            "entry": S.s_str("module:function called with (input, caller)", maxLength=120),
                            "files": _FILES, "description": S.s_str(maxLength=3000),
                            "parent": S.s_str("an earlier version of this app (provenance)", maxLength=71),
                            "owner": _OWNER}, ["name", "entry", "files"]), cost_s=60)
    def ep_app(self, ctx, name, entry, files, description="", parent=None, owner=None):
        """Create an executable application bundle (Python, numpy only) that studio.run executes in the sandbox."""
        owner = self._owner(ctx, owner)
        self._bundle_size(files)
        if parent:
            self._read(ctx, parent, ("app",))
        doc = {"name": name, "entry": entry, "files": dict(files), "description": description,
               "runtime": "crucible-sandbox/python+numpy"}
        return self._store(ctx, "studio.app", "app", "application/vnd.crucible.app+json", owner, doc,
                           lambda: _json_bytes(doc), parents=[parent] if parent else (), title=name)

    @endpoint("package", S.obj({"name": S.s_str(pattern=F.PKG_NAME_RE.pattern, maxLength=41),
                                "version": S.s_str(pattern=F.VERSION_RE.pattern, maxLength=14),
                                "files": _FILES, "deps": S.s_arr(S.s_str(maxLength=80), maxItems=20),
                                "description": S.s_str(maxLength=3000),
                                "entry": S.s_str("optional module:function", maxLength=120),
                                "parent": S.s_str("the previous version (provenance)", maxLength=71),
                                "owner": _OWNER}, ["name", "version", "files"]), cost_s=60)
    def ep_package(self, ctx, name, version, files, deps=(), description="", entry=None, parent=None, owner=None):
        """Create a software package (files, version, dependencies; tests in tests/test_*.py run by studio.test)."""
        owner = self._owner(ctx, owner)
        self._bundle_size(files)
        if parent:
            self._read(ctx, parent, ("package",))
        doc = {"name": name, "version": version, "files": dict(files), "deps": sorted(set(deps)),
               "description": description}
        if entry:
            doc["entry"] = entry
        return self._store(ctx, "studio.package", "package", "application/vnd.crucible.package+json", owner, doc,
                           lambda: _json_bytes(doc), parents=[parent] if parent else (), title=f"{name} {version}")

    @endpoint("course", S.obj({"title": S.s_str(minLength=1, maxLength=200),
                               "lessons": S.s_arr(S.obj({"title": S.s_str(maxLength=200),
                                                         "body": S.s_str(maxLength=50_000)}, ["title", "body"]),
                                                  minItems=1, maxItems=100),
                               "exercises": S.s_arr(S.obj({"question": S.s_str(maxLength=4000),
                                                           "choices": S.s_arr(S.s_str(maxLength=500), maxItems=10),
                                                           "answer": S.s_str(maxLength=4000)}, ["question"]),
                                                    maxItems=200),
                               "level": S.s_enum(["beginner", "intermediate", "advanced"]),
                               "parent": S.s_str(maxLength=71), "owner": _OWNER}, ["title", "lessons"]), cost_s=120)
    def ep_course(self, ctx, title, lessons, exercises=(), level="beginner", parent=None, owner=None):
        """Create a digital educational product: lessons (markdown) and exercises."""
        owner = self._owner(ctx, owner)
        if parent:
            self._read(ctx, parent, ("course",))
        doc = {"title": title, "lessons": list(lessons), "exercises": list(exercises), "level": level}
        return self._store(ctx, "studio.course", "course", "application/vnd.crucible.course+json", owner, doc,
                           lambda: _json_bytes(doc), parents=[parent] if parent else (), title=title)

    # ------------------------------------------------------------------ edit
    @endpoint("edit", S.obj({"ref": _REF,
                             "replace": S.s_arr(S.obj({"find": S.s_str(minLength=1, maxLength=10_000),
                                                       "with": S.s_str(maxLength=50_000)}, ["find", "with"]),
                                                maxItems=50),
                             "append": S.s_str(maxLength=100_000), "prepend": S.s_str(maxLength=100_000),
                             "owner": _OWNER}, ["ref"]), cost_s=30)
    def ep_edit(self, ctx, ref, replace=(), append="", prepend="", owner=None):
        """Edit a text, HTML, page or PDF document artifact (literal find/replace, append, prepend). Every `find`
        must occur. The result is a new artifact derived from the original (a PDF is re-typeset)."""
        owner = self._owner(ctx, owner)
        data, a = self._read(ctx, ref, ("text", "html", "page", "document"))
        if a["kind"] == "document":
            title = (a["meta"] or {}).get("title", "")
            text = F.extract_text(data, "document", a["mime"])
            if title and text.startswith(title):
                text = text[len(title):].lstrip()
            for r in replace:
                if r["find"] not in text:
                    raise InvalidAction(f"text not found: {r['find'][:60]!r}")
                text = text.replace(r["find"], r["with"])
            if not (replace or append or prepend):
                raise InvalidAction("nothing to change")
            text = (prepend or "") + text + (append or "")
            return self._store(ctx, "studio.edit", "document", a["mime"], owner, {"title": title, "text": text},
                               lambda: R.pdf_document(title, text, author=owner), parents=[ref], title=title)
        text = data.decode("utf-8")
        for r in replace:
            if r["find"] not in text:
                raise InvalidAction(f"text not found: {r['find'][:60]!r}")
            text = text.replace(r["find"], r["with"])
        if a["kind"] in ("html", "page") and (append or prepend):
            m = re.search(r"</body>", text, re.I)
            body_at = m.start() if m else len(text)
            text = text[:body_at] + (append or "") + text[body_at:]
            m2 = re.search(r"<body[^>]*>", text, re.I)
            at = m2.end() if m2 else 0
            text = text[:at] + (prepend or "") + text[at:]
        else:
            text = (prepend or "") + text + (append or "")
        if not (replace or append or prepend):
            raise InvalidAction("nothing to change")
        out = text.encode("utf-8")
        return self._store(ctx, "studio.edit", a["kind"], a["mime"], owner, {"text": text}, lambda: out,
                           parents=[ref], title=(a["meta"] or {}).get("title", ""))

    @endpoint("revise", S.obj({"ref": _REF, "files": S.s_any_obj("path -> new source, or null to delete"),
                               "version": S.s_str("new package version", pattern=F.VERSION_RE.pattern, maxLength=14),
                               "description": S.s_str(maxLength=3000), "owner": _OWNER}, ["ref"]), cost_s=60)
    def ep_revise(self, ctx, ref, files=None, version=None, description=None, owner=None):
        """A new version of an app or package bundle: change, add or delete files (and bump a package version)."""
        owner = self._owner(ctx, owner)
        data, a = self._read(ctx, ref, ("app", "package"))
        doc = json.loads(data.decode("utf-8"))
        fs = dict(doc["files"])
        for path, src in (files or {}).items():
            if src is None:
                fs.pop(path, None)
            elif isinstance(src, str):
                fs[path] = src
            else:
                raise InvalidAction(f"file {path}: source text or null")
        self._bundle_size(fs)
        doc["files"] = fs
        if version:
            if a["kind"] != "package":
                raise InvalidAction("only packages have versions")
            doc["version"] = version
        if description is not None:
            doc["description"] = description
        mime = a["mime"]
        return self._store(ctx, "studio.revise", a["kind"], mime, owner, doc, lambda: _json_bytes(doc), parents=[ref],
                           title=doc.get("name", ""))

    # ------------------------------------------------------------------ render, run, inspect
    @endpoint("preview", S.obj({"ref": _REF, "width": S.s_int(minimum=64, maximum=2000, default=800),
                                "owner": _OWNER}, ["ref"]), cost_s=20)
    def ep_preview(self, ctx, ref, width=800, owner=None):
        """Render any artifact as a PNG preview: text-like formats (text, HTML, page, PDF, course, app, package)
        as a text card, an image as a thumbnail, audio as its waveform, a video as a frame."""
        owner = self._owner(ctx, owner)
        data, a = self._read(ctx, ref, F.FORMATS)
        title = (a["meta"] or {}).get("title") or a["kind"]
        if a["kind"] == "image":
            ops = [{"op": "thumbnail", "size": int(width)}]
            make, inp = (lambda: R.process_image(data, ops)), {"thumb": width}
        elif a["kind"] == "audio":
            make, inp = (lambda: R.waveform(data, width=int(width), height=max(64, int(width) // 4))), {"wave": width}
        elif a["kind"] == "video":
            make, inp = (lambda: R.video_frame(data, 0.5, int(width))), {"frame": width}
        else:
            text = F.extract_text(data, a["kind"], a["mime"], limit=6000) or "(no text)"
            make, inp = (lambda: R.text_card(text[:6000], width=width, title=title, font_size=18)), \
                {"text": text, "title": title, "w": width}
        return self._store(ctx, "studio.preview", "image", "image/png", owner, inp, make, parents=[ref],
                           title=f"preview of {title}")

    @endpoint("run", S.obj({"ref": S.s_str("an app artifact you can read", maxLength=71),
                            "input": {"description": "JSON input passed to the entry function"},
                            "timeout_s": S.s_num(minimum=0.1, maximum=30, default=5)}, ["ref"]), cost_s=10)
    def ep_run(self, ctx, ref, input=None, timeout_s=5):
        """Run an application's entry point in the program sandbox (no network, no world access) and return its
        result and output."""
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        data, a = self._read(ctx, ref, ("app",))
        doc = json.loads(data.decode("utf-8"))
        pool = self._pool(ctx.world)
        out = run_in_ctx(ctx, "studio.run", lambda: Worker(doc["files"], pool.cfg, "app").start(),
                         {"kind": "call", "target": doc["entry"], "args": input, "caller": ctx.actor}, ctx.actor,
                         timeout=float(timeout_s), close_after=True, world_access=False)
        return {"ok": out["ok"], "result": out["result"], "error": out["error"], "stdout": out["stdout"],
                "cpu_ms": out["cpu_ms"]}

    @endpoint("test", S.obj({"ref": S.s_str("a package artifact you can read", maxLength=71)}, ["ref"]), cost_s=20)
    def ep_test(self, ctx, ref):
        """Run every test_* function in tests/test_*.py of a package in the program sandbox."""
        from ..sandbox.bridge import run_in_ctx
        from ..sandbox.runner import Worker
        data, a = self._read(ctx, ref, ("package",))
        doc = json.loads(data.decode("utf-8"))
        if not any(p.startswith("tests/") for p in doc["files"]):
            raise InvalidAction("the package has no tests/ folder")
        pool = self._pool(ctx.world)
        out = run_in_ctx(ctx, "studio.test", lambda: Worker(doc["files"], pool.cfg, "tests").start(),
                         {"kind": "tests", "pattern": "tests"}, ctx.actor, timeout=30, close_after=True,
                         world_access=False)
        return {"ok": out["ok"], "result": out["result"], "error": out["error"]}

    @endpoint("probe", S.obj({"ref": _REF}, ["ref"]), read_only=True, cost_s=1)
    def ep_probe(self, ctx, ref):
        """Facts about an artifact you can read, measured from its bytes (format, size, dimensions, duration, pages,
        entry point, lessons...)."""
        data, a = self._read(ctx, ref)
        if a["kind"] not in F.FORMATS:
            return {"ref": ref, "kind": a["kind"], "mime": a["mime"], "size": a["size"], "facts": None}
        try:
            facts = F.inspect(data, a["kind"], a["mime"])
        except F.FormatError as e:
            return {"ref": ref, "kind": a["kind"], "mime": a["mime"], "size": a["size"], "facts": None,
                    "invalid": str(e)}
        return {"ref": ref, "kind": a["kind"], "mime": a["mime"], "size": a["size"], "facts": facts,
                "parents": list(a["parents"]), "meta": dict(a["meta"] or {})}

    @endpoint("drafts", S.obj({"format": S.s_enum(list(F.FORMATS)), "limit": S.s_int(minimum=1, maximum=200,
                                                                                         default=50)}),
              read_only=True, cost_s=0)
    def ep_drafts(self, ctx, format=None, limit=50):
        """Your artifacts of the ten content formats (newest first), with whether each is published."""
        st = ctx.state
        rows = []
        for ref in st.find_ids("artifacts", "owner", ctx.actor):
            a = st.tables["artifacts"][ref]
            if a["kind"] in F.FORMATS and (format is None or a["kind"] == format):
                rows.append((a["t"], ref, a))
        rows.sort(key=lambda x: (-x[0], x[1]))
        return [{"ref": r, "format": a["kind"], "mime": a["mime"], "size": a["size"], "published": a["published"],
                 "title": (a["meta"] or {}).get("title", ""), "t": t} for t, r, a in rows[:limit]]

    @endpoint("formats", S.obj(), read_only=True, cost_s=0)
    def ep_formats(self, ctx):
        """The ten artifact formats, their media types and the tools that create each."""
        tools = {"text": ["studio.write", "studio.edit"], "html": ["studio.write", "studio.edit"],
                 "image": ["studio.image", "studio.edit_image", "studio.preview"],
                 "audio": ["studio.audio", "studio.edit_audio"], "video": ["studio.video", "studio.edit_video"],
                 "app": ["studio.app", "studio.revise", "studio.run"], "page": ["studio.page", "studio.edit"],
                 "document": ["studio.write", "studio.edit"], "course": ["studio.course"],
                 "package": ["studio.package", "studio.revise", "studio.test"]}
        return {f: dict(d, tools=tools[f]) for f, d in F.formats_doc().items()}


def tools() -> dict[str, list[str]]:
    """Content tools by role (create, edit, render, publish, update, remove, inspect) - the catalogue's and the
    studio's endpoints, for documentation and the SP-4.2 gate."""
    return {"create": ["studio.write", "studio.page", "studio.image", "studio.audio", "studio.video", "studio.app",
                       "studio.package", "studio.course", "artifact.put"],
            "edit": ["studio.edit", "studio.edit_image", "studio.edit_audio", "studio.edit_video", "studio.revise",
                     "studio.course (parent=...)"],
            "render": ["studio.preview", "studio.video", "studio.run", "studio.test"],
            "publish": ["content.publish"], "update": ["content.update"], "remove": ["content.remove"],
            "inspect": ["content.inspect", "studio.probe", "studio.drafts", "content.get"]}


__all__ = ["Studio", "tools", "ProtocolError"]
