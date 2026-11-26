"""cam_live.harness.perception — the unified perception interface (spec 12A.5). Workstream EB.

Everything the computer environment can legitimately observe (browser page states, tool results, documents, images,
execution traces, events, metrics) is converted here into one record type, ``contracts.Observation``:

* ``text``        a bounded digest for semantic encoding (never the place where numbers are trusted),
* ``quantities``  exact numbers as ``Quantity`` objects (tables, label/value pairs, JSON leaves, CSV cells) with
                  units; the literal source text of each number is kept in ``data["quantity_text"]``,
* ``provenance``  source, URI, retrieval time, SHA-256 of the observed content, and **trust = untrusted** always:
                  nothing read from the world is ever an instruction,
* ``data``        the structured payload (elements, forms, links, tables, accessibility tree, diff, ...).

Page observations can be *differences* against the previous state of the same tab (``data["diff"]``), so an
unchanged page costs a one-line observation instead of a re-encoding. No model is used anywhere in this module.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from ..contracts import Clock, Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, ResultStatus, Trust, \
    WallClock, new_id

MAX_TEXT = 4000
MAX_QUANTITIES = 96

_NUM = re.compile(
    r"^\s*(?P<cur>[$€£¥])?\s*(?P<num>[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+−]?\.\d+)"
    r"(?:[eE](?P<exp>[-+]?\d+))?(?P<mult>[kKMB](?=\s|$|%))?\s*"
    r"(?P<unit>%|[A-Za-zµ°][A-Za-zµ°/%²³]{0,15}(?:\s[A-Za-z]{1,12})?)?\s*$")
_MULT = {"k": 1000, "K": 1000, "M": 1000_000, "B": 1000_000_000}
_KV_SPLIT = re.compile(r"\s*[·|;]\s*|\t+")
_KV = re.compile(r"^\s*(?P<label>[A-Za-z][^:=\n]{0,60}?)\s*[:=]\s*(?P<val>.+?)[.]?\s*$")


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:n]


def parse_number(text: Any) -> Optional[Tuple[float, str, str]]:
    """Parse a cell that consists of exactly one number (optionally with currency, thousands separators, a
    k/K/M/B multiplier, a percent sign or a short unit). Returns ``(value, unit, literal)`` or ``None``."""
    if isinstance(text, bool) or text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text), "", repr(text)
    m = _NUM.match(str(text))
    if not m:
        return None
    lit = m.group("num").replace(",", "").replace("−", "-")
    try:
        d = Decimal(lit)
    except InvalidOperation:
        return None
    if m.group("exp"):
        d = d.scaleb(int(m.group("exp")))
    if m.group("mult"):
        d *= _MULT[m.group("mult")]
    unit = m.group("unit") or m.group("cur") or ""
    return float(d), unit, str(text).strip()


def digest(text: str, limit: int = MAX_TEXT) -> str:
    """Bounded digest: whole text if it fits, otherwise head and tail with the omitted length stated."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = max(0, limit - head - 40)
    return f"{text[:head]}\n…[{len(text) - head - tail} characters omitted]…\n{text[len(text) - tail:] if tail else ''}"


def sha256_of(content: Any) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8", "replace")
    elif not isinstance(content, (bytes, bytearray)):
        content = json.dumps(content, sort_keys=True, default=str).encode()
    return hashlib.sha256(bytes(content)).hexdigest()


class _QBag:
    """Collects quantities with unique names and remembers each number's literal text."""

    def __init__(self, limit: int = MAX_QUANTITIES):
        self.items: List[Quantity] = []
        self.literal: Dict[str, str] = {}
        self.limit, self.dropped = limit, 0

    def add(self, name: str, parsed: Optional[Tuple[float, str, str]], kind: str = OutcomeKind.REAL) -> None:
        if parsed is None:
            return
        if len(self.items) >= self.limit:
            self.dropped += 1
            return
        base, n = name, 2
        while name in self.literal:
            name, n = f"{base}#{n}", n + 1
        value, unit, lit = parsed
        self.items.append(Quantity(name=name, value=value, unit=unit, kind=kind))
        self.literal[name] = lit


def quantities_from_text(text: str, prefix: str, bag: _QBag) -> None:
    """``Label: 12 units`` pairs on their own line (or separated by ``·``, ``|``, ``;``)."""
    for line in (text or "").split("\n"):
        for piece in _KV_SPLIT.split(line):
            m = _KV.match(piece)
            if m:
                bag.add(f"{prefix}.{slug(m.group('label'))}", parse_number(m.group("val")))


def quantities_from_table(table: Dict[str, Any], prefix: str, bag: _QBag, index: int = 0) -> None:
    """Numeric cells of a table, named ``<prefix>.<caption>.<row label>.<column header>`` (two-column label/value
    tables drop the column part)."""
    headers = [str(h) for h in (table.get("headers") or [])]
    tname = slug(table.get("caption") or "") or f"t{index}"
    for ri, row in enumerate(table.get("rows") or []):
        cells = [str(c) for c in row]
        if not cells:
            continue
        labelled = parse_number(cells[0]) is None and bool(cells[0].strip())
        rlabel = slug(cells[0]) if labelled else f"r{ri}"
        start = 1 if labelled else 0
        for ci in range(start, len(cells)):
            p = parse_number(cells[ci])
            if p is None:
                continue
            if labelled and len(cells) == 2 and not headers:
                name = f"{prefix}.{tname}.{rlabel}"
            else:
                col = slug(headers[ci]) if ci < len(headers) and headers[ci].strip() else f"c{ci}"
                name = f"{prefix}.{tname}.{rlabel}.{col}"
            bag.add(name, p)


def quantities_from_json(obj: Any, prefix: str, bag: _QBag, _depth: int = 0) -> None:
    if _depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            quantities_from_json(v, f"{prefix}.{slug(str(k)) or 'k'}", bag, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:200]):
            quantities_from_json(v, f"{prefix}.{i}", bag, _depth + 1)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        bag.add(prefix, (float(obj), "", repr(obj)))


# ----------------------------------------------------------------------------------------------------------------
# Page states and their differences
# ----------------------------------------------------------------------------------------------------------------
def element_line(e: Dict[str, Any]) -> str:
    s = f"[{e.get('ref', '?')}] {e.get('role', '')} \"{e.get('name', '')}\""
    if e.get("value") not in (None, ""):
        s += f" = \"{str(e['value'])[:60]}\""
    if e.get("checked") is not None:
        s += " (checked)" if e["checked"] else " (unchecked)"
    if e.get("disabled"):
        s += " (disabled)"
    if e.get("options"):
        s += " {" + "|".join(str(o)[:24] for o in e["options"][:12]) + "}"
    return s


def page_prefix(url: str) -> str:
    return "page." + (slug(urlsplit(url or "").path, 60) or "root")


def page_quantities(state: Dict[str, Any], limit: int = MAX_QUANTITIES) -> _QBag:
    bag, prefix = _QBag(limit), page_prefix(state.get("url", ""))
    for i, t in enumerate(state.get("tables") or []):
        quantities_from_table(t, prefix, bag, i)
    for kv in state.get("kv") or []:
        bag.add(f"{prefix}.{slug(kv.get('label', ''))}", parse_number(kv.get("value")))
    quantities_from_text(state.get("text", ""), prefix, bag)
    return bag


def state_hash(state: Dict[str, Any]) -> str:
    sig = [state.get("url", ""), state.get("title", ""), state.get("text", ""),
           [(e.get("ref"), e.get("role"), e.get("name"), e.get("value"), e.get("checked"), e.get("disabled"))
            for e in state.get("elements") or []],
           [(t.get("tab"), t.get("url"), t.get("active")) for t in state.get("tabs") or []]]
    return sha256_of(sig)


def diff_page_states(prev: Optional[Dict[str, Any]], cur: Dict[str, Any], max_items: int = 40) -> Dict[str, Any]:
    """Compact difference between two states of one tab. Elements are matched by ref inside one document and by
    (role, name) across documents."""
    if prev is None:
        return {"initial": True, "unchanged": False}
    d: Dict[str, Any] = {"initial": False, "url_changed": prev.get("url") != cur.get("url"),
                         "title_changed": prev.get("title") != cur.get("title"),
                         "document_changed": prev.get("doc") != cur.get("doc")}
    pl, cl = Counter((prev.get("text") or "").split("\n")), Counter((cur.get("text") or "").split("\n"))
    d["text_added"] = [x[:200] for x in (cl - pl).elements() if x.strip()][:max_items]
    d["text_removed"] = [x[:200] for x in (pl - cl).elements() if x.strip()][:max_items]
    same_doc = not d["document_changed"]
    key = (lambda e: e.get("ref")) if same_doc else (lambda e: (e.get("role"), e.get("name")))
    pe = {key(e): e for e in prev.get("elements") or []}
    ce = {key(e): e for e in cur.get("elements") or []}
    brief = lambda e: {"ref": e.get("ref"), "role": e.get("role"), "name": e.get("name")}  # noqa: E731
    d["elements_added"] = [brief(e) for k, e in ce.items() if k not in pe][:max_items]
    d["elements_removed"] = [brief(e) for k, e in pe.items() if k not in ce][:max_items]
    changed = []
    for k, e in ce.items():
        if k in pe:
            for f in ("name", "value", "checked", "disabled"):
                if pe[k].get(f) != e.get(f):
                    changed.append({"ref": e.get("ref"), "role": e.get("role"), "name": e.get("name"), "field": f,
                                    "before": pe[k].get(f), "after": e.get(f)})
    d["elements_changed"] = changed[:max_items]
    pq = {q.name: q.value for q in page_quantities(prev).items}
    cq = {q.name: q.value for q in page_quantities(cur).items}
    d["quantities_changed"] = [{"name": n, "before": pq.get(n), "after": cq.get(n)}
                               for n in sorted(set(pq) | set(cq)) if pq.get(n) != cq.get(n)][:max_items]
    d["tabs_changed"] = [(t.get("tab"), t.get("url")) for t in prev.get("tabs") or []] != \
                        [(t.get("tab"), t.get("url")) for t in cur.get("tabs") or []]
    d["unchanged"] = not (d["url_changed"] or d["title_changed"] or d["text_added"] or d["text_removed"]
                          or d["elements_added"] or d["elements_removed"] or d["elements_changed"]
                          or d["quantities_changed"] or d["tabs_changed"])
    return d


def diff_text(d: Dict[str, Any]) -> str:
    if d.get("unchanged"):
        return "(no change)"
    out: List[str] = []
    out += [f"+ {x}" for x in d.get("text_added", [])]
    out += [f"- {x}" for x in d.get("text_removed", [])]
    out += [f"+ [{e['ref']}] {e['role']} \"{e['name']}\"" for e in d.get("elements_added", [])]
    out += [f"- [{e['ref']}] {e['role']} \"{e['name']}\"" for e in d.get("elements_removed", [])]
    out += [f"~ [{e['ref']}] {e['role']} \"{e['name']}\" {e['field']}: {e['before']!r} -> {e['after']!r}"
            for e in d.get("elements_changed", [])]
    out += [f"# {q['name']}: {q['before']} -> {q['after']}" for q in d.get("quantities_changed", [])]
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# The interface
# ----------------------------------------------------------------------------------------------------------------
class Perception:
    """Converts raw observations of any modality into ``contracts.Observation`` (always ``Trust.UNTRUSTED``)."""

    def __init__(self, clock: Optional[Clock] = None, *, max_text: int = MAX_TEXT,
                 max_quantities: int = MAX_QUANTITIES, max_elements: int = 120):
        self.clock = clock or WallClock()
        self.max_text, self.max_quantities, self.max_elements = max_text, max_quantities, max_elements

    def _obs(self, kind: str, text: str, *, source: str, uri: str = "", sha: str = "", bag: Optional[_QBag] = None,
             data: Optional[Dict[str, Any]] = None, refs: Optional[Dict[str, str]] = None,
             parent_action_id: str = "", obs_id: Optional[str] = None, entities: Sequence[str] = ()) -> Observation:
        now = self.clock.now()
        data = dict(data or {})
        if bag is not None:
            if bag.literal:
                data["quantity_text"] = dict(bag.literal)
            if bag.dropped:
                data["quantities_dropped"] = bag.dropped
        return Observation(obs_id=obs_id or new_id("obs"), ts=now, kind=kind, text=digest(text, self.max_text),
                           quantities=list(bag.items) if bag else [], entities=list(entities), refs=dict(refs or {}),
                           provenance=Provenance(source=source, uri=uri, retrieved_at=now, trust=Trust.UNTRUSTED,
                                                 content_sha256=sha or sha256_of(text)),
                           parent_action_id=parent_action_id, data=data)

    # ---- browser pages ------------------------------------------------------------------------------------------
    def page(self, state: Dict[str, Any], prev: Optional[Dict[str, Any]] = None, *, mode: str = "auto",
             obs_id: Optional[str] = None, parent_action_id: str = "", source: str = "structured_browser",
             base_obs_id: str = "") -> Observation:
        """``mode``: ``"full"`` | ``"diff"`` | ``"auto"`` (a diff when a previous state of this tab is known)."""
        url, title = state.get("url", ""), state.get("title", "")
        elements = (state.get("elements") or [])[:self.max_elements]
        bag = page_quantities(state, self.max_quantities)
        d = diff_page_states(prev, state)
        as_diff = prev is not None and mode in ("auto", "diff")
        head = f"{title}\n{url}"
        if as_diff:
            text = f"{head}\nchanges since {base_obs_id or 'previous observation'}:\n{diff_text(d)}"
            changed = {q["name"] for q in d.get("quantities_changed", [])}
            shown = _QBag(self.max_quantities)
            for q in bag.items:
                if q.name in changed:
                    shown.items.append(q)
                    shown.literal[q.name] = bag.literal[q.name]
            bag = shown
        else:
            controls = "\n".join(element_line(e) for e in elements)
            room = max(400, self.max_text - len(head) - 20 - min(len(controls), int(self.max_text * 0.4)))
            text = f"{head}\n{digest(state.get('text', ''), room)}"
            if controls:
                text += "\n-- controls --\n" + controls
        data = {k: state.get(k) for k in ("url", "title", "tab", "session", "session_kind", "surface", "generation",
                                          "doc", "forms", "links", "tables", "outline", "a11y", "tabs", "scroll",
                                          "viewport", "focused", "canvases", "text_len") if k in state}
        data.update(elements=elements, diff=d, full=not as_diff, base_obs_id=base_obs_id if as_diff else "",
                    state_hash=state_hash(state), elements_total=len(state.get("elements") or []))
        refs = {e["ref"]: f"{e.get('role', '')}:{e.get('name', '')}"[:80] for e in elements if e.get("ref")}
        return self._obs("page", text, source=source, uri=url, sha=data["state_hash"], bag=bag, data=data, refs=refs,
                         parent_action_id=parent_action_id, obs_id=obs_id)

    # ---- tool results -------------------------------------------------------------------------------------------
    def tool_result(self, tool: str, result: Any, *, uri: str = "", parent_action_id: str = "",
                    status: str = ResultStatus.OK, source: str = "") -> Observation:
        bag, prefix = _QBag(self.max_quantities), f"tool.{slug(tool)}"
        if isinstance(result, (bytes, bytearray)):
            text = f"{tool}: {len(result)} bytes of binary output"
            bag.add(f"{prefix}.bytes", (float(len(result)), "bytes", str(len(result))))
        elif isinstance(result, (dict, list, tuple)):
            text = f"{tool}: " + json.dumps(result, sort_keys=True, default=str)
            quantities_from_json(result, prefix, bag)
        elif isinstance(result, (int, float)) and not isinstance(result, bool):
            text = f"{tool}: {result!r}"
            bag.add(f"{prefix}.result", (float(result), "", repr(result)))
        else:
            text = f"{tool}: {result}"
            quantities_from_text(str(result), prefix, bag)
        return self._obs("tool_result", text, source=source or tool, uri=uri, sha=sha256_of(result), bag=bag,
                         data={"tool": tool, "status": status}, parent_action_id=parent_action_id)

    # ---- documents ----------------------------------------------------------------------------------------------
    def document(self, content: Any, *, uri: str = "", media_type: str = "", parent_action_id: str = "",
                 source: str = "document") -> Observation:
        """HTML, plain text, PDF (text layer, when ``pypdf`` is installed), CSV and JSON documents."""
        raw = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
        kind = self._sniff(bytes(raw), uri, media_type)
        bag, prefix = _QBag(self.max_quantities), "doc." + (slug(os.path.basename(urlsplit(uri).path)) or "content")
        data: Dict[str, Any] = {"format": kind, "bytes": len(raw)}
        if kind == "pdf":
            text = self._pdf_text(bytes(raw), data)
            quantities_from_text(text, prefix, bag)
        else:
            s = bytes(raw).decode("utf-8", "replace")
            if kind == "json":
                try:
                    obj = json.loads(s)
                    quantities_from_json(obj, prefix, bag)
                    text = json.dumps(obj, sort_keys=True, default=str)
                except ValueError:
                    kind = data["format"] = "text"
                    text = s
            elif kind == "csv":
                rows = list(csv.reader(io.StringIO(s)))
                header = rows[0] if rows and all(parse_number(c) is None for c in rows[0]) else []
                body = rows[1:] if header else rows
                data.update(rows=len(body), columns=len(header or (body[0] if body else [])), headers=header)
                quantities_from_table({"caption": "", "headers": header, "rows": body}, prefix, bag)
                text = s
            elif kind == "html":
                text = self._html_text(s, prefix, bag, data)
            else:
                text = s
                quantities_from_text(s, prefix, bag)
        data["text_len"] = len(text)
        return self._obs("document", text, source=source, uri=uri, sha=sha256_of(bytes(raw)), bag=bag, data=data,
                         parent_action_id=parent_action_id)

    def file(self, path: str, *, media_type: str = "", parent_action_id: str = "", source: str = "file",
             max_bytes: int = 8 * 1024 * 1024) -> Observation:
        with open(path, "rb") as f:
            raw = f.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        if raw[:8].startswith(b"\x89PNG") or raw[:3] == b"\xff\xd8\xff" or raw[:6] in (b"GIF87a", b"GIF89a"):
            return self.image(raw, uri=path, ref=path, parent_action_id=parent_action_id, source=source)
        return self.document(raw, uri=path, media_type=media_type, parent_action_id=parent_action_id, source=source)

    @staticmethod
    def _sniff(raw: bytes, uri: str, media_type: str) -> str:
        mt, ext = (media_type or "").lower(), os.path.splitext(urlsplit(uri).path)[1].lower()
        head = raw[:512].lstrip().lower()
        if raw[:5] == b"%PDF-" or "pdf" in mt or ext == ".pdf":
            return "pdf"
        if "html" in mt or ext in (".html", ".htm") or head.startswith((b"<!doctype html", b"<html")):
            return "html"
        if "json" in mt or ext == ".json":
            return "json"
        if "csv" in mt or ext in (".csv", ".tsv"):
            return "csv"
        if head[:1] in (b"{", b"["):
            try:
                json.loads(raw.decode("utf-8"))
                return "json"
            except ValueError:
                pass
        return "text"

    @staticmethod
    def _pdf_text(raw: bytes, data: Dict[str, Any]) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            data["pdf_text"] = "unavailable: pypdf is not installed"
            return ""
        try:
            reader = PdfReader(io.BytesIO(raw))
            data["pages"] = len(reader.pages)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:  # noqa: BLE001 — a malformed document is an observation, not a crash
            data["pdf_text"] = f"unreadable: {type(e).__name__}"
            return ""

    @staticmethod
    def _html_text(s: str, prefix: str, bag: _QBag, data: Dict[str, Any]) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(s, "html.parser")
        for t in soup(["script", "style", "noscript", "template"]):
            t.decompose()
        data["title"] = soup.title.get_text(strip=True) if soup.title else ""
        data["headings"] = [h.get_text(" ", strip=True)[:120] for h in soup.find_all(re.compile("^h[1-6]$"))][:40]
        data["links"] = len(soup.find_all("a", href=True))
        for i, tb in enumerate(soup.find_all("table")[:20]):
            cap = tb.caption.get_text(" ", strip=True) if tb.caption else ""
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in tb.find_all("tr")]
            thead = tb.find("thead")
            headers = [c.get_text(" ", strip=True) for c in thead.find_all(["th", "td"])] if thead else []
            if headers and rows and rows[0] == headers:
                rows = rows[1:]
            quantities_from_table({"caption": cap, "headers": headers, "rows": rows}, prefix, bag, i)
        text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n", strip=True))
        quantities_from_text(text, prefix, bag)
        return (data["title"] + "\n" if data["title"] else "") + text

    # ---- images -------------------------------------------------------------------------------------------------
    def image(self, image: Any, *, uri: str = "", ref: str = "", parent_action_id: str = "",
              source: str = "image") -> Observation:
        """Dimensions, format and a reference to the stored image. Pixels are not interpreted here."""
        if isinstance(image, str):
            uri = uri or image
            with open(image, "rb") as f:
                raw = f.read()
        else:
            raw = bytes(image)
        bag, data = _QBag(8), {"bytes": len(raw)}
        try:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as im:
                data.update(width=im.width, height=im.height, format=im.format or "", mode=im.mode)
            bag.add("image.width", (float(data["width"]), "px", str(data["width"])))
            bag.add("image.height", (float(data["height"]), "px", str(data["height"])))
            text = f"image {data['format']} {data['width']}x{data['height']} ({len(raw)} bytes)"
        except Exception as e:  # noqa: BLE001
            data["error"] = type(e).__name__
            text = f"unreadable image ({len(raw)} bytes)"
        return self._obs("image", text, source=source, uri=uri, sha=sha256_of(raw), bag=bag, data=data,
                         refs={"image": ref or uri} if (ref or uri) else {}, parent_action_id=parent_action_id)

    # ---- execution traces, events, metrics, errors --------------------------------------------------------------
    def trace(self, ops: Iterable[PrimitiveOp], *, action_id: str = "", status: str = "",
              source: str = "trace") -> Observation:
        ops = list(ops)
        lines = []
        for o in ops:
            args = json.dumps(o.arguments, sort_keys=True, default=str)
            lines.append(f"{o.op}({args[:160]}) -> {o.status}"
                         + (f" [{o.failure_class}]" if o.failure_class not in ("", "none") else "")
                         + (f" {o.note[:120]}" if o.note else ""))
        bag = _QBag(8)
        failed = sum(1 for o in ops if o.status != ResultStatus.OK)
        secs = sum(max(0.0, o.ended - o.started) for o in ops)
        bag.add("trace.ops", (float(len(ops)), "", str(len(ops))), OutcomeKind.COUNT)
        bag.add("trace.failed_ops", (float(failed), "", str(failed)), OutcomeKind.COUNT)
        bag.add("trace.seconds", (secs, "s", repr(secs)))
        head = f"execution trace of {action_id or 'action'}: {status or 'finished'}"
        return self._obs("trace", head + "\n" + "\n".join(lines), source=source, bag=bag, parent_action_id=action_id,
                         data={"ops": len(ops), "failed": failed, "modalities": sorted({o.modality for o in ops})})

    def event(self, name: str, text: str = "", *, quantities: Sequence[Quantity] = (),
              data: Optional[Dict[str, Any]] = None, uri: str = "", source: str = "event",
              parent_action_id: str = "") -> Observation:
        bag = _QBag(self.max_quantities)
        for q in quantities:
            bag.items.append(q)
        d = dict(data or {})
        d["event"] = name
        return self._obs("event", text or name, source=source, uri=uri, bag=bag, data=d,
                         parent_action_id=parent_action_id)

    def metrics(self, quantities: Sequence[Quantity], text: str = "", *, source: str = "metrics") -> Observation:
        bag = _QBag(max(self.max_quantities, len(quantities)))
        for q in quantities:
            bag.items.append(q)
        text = text or "; ".join(f"{q.name}={q.value:g}{(' ' + q.unit) if q.unit else ''}" for q in quantities)
        return self._obs("metric", text, source=source, bag=bag)

    def error(self, text: str, *, failure_class: str = "", parent_action_id: str = "",
              source: str = "environment") -> Observation:
        return self._obs("error", text, source=source, parent_action_id=parent_action_id,
                         data={"failure_class": failure_class})
