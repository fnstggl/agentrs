"""Controlled computer-use pages for the grounding benchmark and the visual-executor tests (workstream EC).

Every page is one HTML document whose controls are *drawn on a canvas*: there is no DOM element, accessible name
or selector to fall back on, so the only way to operate it is by looking at pixels. A page is described by a
spec (controls with rectangles, one target per stage); the same script renders it, hit-tests clicks, handles
keyboard and dragging, and exposes what happened as ``window.__state`` so a test or the benchmark can read the
real effect of an action instead of trusting the actor.

Visible feedback that does not depend on where the target is: the strip at the top turns red when a control other
than the target was activated, and the i-th progress box at its right end turns green when stage i was completed.
``Escape`` resets the page.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

W, H = 1280, 800
BANNER_H = 28
GREEN, RED, NEUTRAL = (26, 154, 58), (198, 40, 40), (120, 124, 130)

_JS = r"""
(() => {
const SPEC = __SPEC__;
const cv = document.getElementById('c'); const g = cv.getContext('2d');
const S = {stage: 0, hits: [], done: false, wrong: 0, fields: {}, focus: null, values: {}, drag: null, resets: 0};
window.__state = S;
function stage() { return SPEC.stages[Math.min(S.stage, SPEC.stages.length - 1)]; }
window.__targets = () => { const o = {}; for (const c of stage().controls) if (c.kind !== 'label') o[c.id] = [c.x, c.y, c.w, c.h]; return o; };
window.__target = () => stage().target;
function rr(x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
const ICONS = {
  search(x, y, s) { g.beginPath(); g.arc(x - s * .12, y - s * .12, s * .26, 0, 7); g.stroke(); g.beginPath(); g.moveTo(x + s * .08, y + s * .08); g.lineTo(x + s * .36, y + s * .36); g.stroke(); },
  settings(x, y, s) { for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4; g.beginPath(); g.moveTo(x + Math.cos(a) * s * .26, y + Math.sin(a) * s * .26); g.lineTo(x + Math.cos(a) * s * .4, y + Math.sin(a) * s * .4); g.stroke(); } g.beginPath(); g.arc(x, y, s * .26, 0, 7); g.stroke(); g.beginPath(); g.arc(x, y, s * .1, 0, 7); g.stroke(); },
  trash(x, y, s) { g.beginPath(); g.moveTo(x - s * .32, y - s * .22); g.lineTo(x + s * .32, y - s * .22); g.stroke(); g.strokeRect(x - s * .1, y - s * .34, s * .2, s * .12); g.beginPath(); g.moveTo(x - s * .24, y - s * .22); g.lineTo(x - s * .18, y + s * .36); g.lineTo(x + s * .18, y + s * .36); g.lineTo(x + s * .24, y - s * .22); g.stroke(); for (const d of [-.08, .08]) { g.beginPath(); g.moveTo(x + s * d, y - s * .1); g.lineTo(x + s * d, y + s * .24); g.stroke(); } },
  home(x, y, s) { g.beginPath(); g.moveTo(x - s * .38, y); g.lineTo(x, y - s * .36); g.lineTo(x + s * .38, y); g.stroke(); g.strokeRect(x - s * .26, y, s * .52, s * .34); g.strokeRect(x - s * .07, y + s * .12, s * .14, s * .22); },
  mail(x, y, s) { g.strokeRect(x - s * .36, y - s * .24, s * .72, s * .48); g.beginPath(); g.moveTo(x - s * .36, y - s * .24); g.lineTo(x, y + s * .04); g.lineTo(x + s * .36, y - s * .24); g.stroke(); },
  bell(x, y, s) { g.beginPath(); g.moveTo(x - s * .3, y + s * .2); g.lineTo(x - s * .22, y + s * .08); g.lineTo(x - s * .22, y - s * .1); g.arc(x, y - s * .1, s * .22, Math.PI, 0); g.lineTo(x + s * .22, y + s * .08); g.lineTo(x + s * .3, y + s * .2); g.closePath(); g.stroke(); g.beginPath(); g.arc(x, y + s * .28, s * .07, 0, 7); g.stroke(); },
  heart(x, y, s) { g.beginPath(); g.moveTo(x, y + s * .32); g.bezierCurveTo(x - s * .6, y - s * .1, x - s * .22, y - s * .46, x, y - s * .14); g.bezierCurveTo(x + s * .22, y - s * .46, x + s * .6, y - s * .1, x, y + s * .32); g.stroke(); },
  star(x, y, s) { g.beginPath(); for (let i = 0; i < 10; i++) { const r = i % 2 ? s * .16 : s * .4, a = -Math.PI / 2 + i * Math.PI / 5; g.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r); } g.closePath(); g.stroke(); },
  plus(x, y, s) { g.beginPath(); g.moveTo(x - s * .32, y); g.lineTo(x + s * .32, y); g.moveTo(x, y - s * .32); g.lineTo(x, y + s * .32); g.stroke(); },
  menu(x, y, s) { for (const d of [-.24, 0, .24]) { g.beginPath(); g.moveTo(x - s * .34, y + s * d); g.lineTo(x + s * .34, y + s * d); g.stroke(); } },
  close(x, y, s) { g.beginPath(); g.moveTo(x - s * .28, y - s * .28); g.lineTo(x + s * .28, y + s * .28); g.moveTo(x + s * .28, y - s * .28); g.lineTo(x - s * .28, y + s * .28); g.stroke(); },
  download(x, y, s) { g.beginPath(); g.moveTo(x, y - s * .36); g.lineTo(x, y + s * .12); g.moveTo(x - s * .2, y - s * .06); g.lineTo(x, y + s * .14); g.lineTo(x + s * .2, y - s * .06); g.moveTo(x - s * .34, y + s * .32); g.lineTo(x + s * .34, y + s * .32); g.stroke(); },
  user(x, y, s) { g.beginPath(); g.arc(x, y - s * .16, s * .17, 0, 7); g.stroke(); g.beginPath(); g.arc(x, y + s * .42, s * .34, Math.PI * 1.12, Math.PI * 1.88); g.stroke(); },
  play(x, y, s) { g.beginPath(); g.moveTo(x - s * .22, y - s * .32); g.lineTo(x + s * .34, y); g.lineTo(x - s * .22, y + s * .32); g.closePath(); g.stroke(); },
};
function draw() {
  const st = stage();
  g.fillStyle = SPEC.background || '#f4f5f7'; g.fillRect(0, 0, cv.width, cv.height);
  g.fillStyle = S.wrong ? 'rgb(198,40,40)' : 'rgb(120,124,130)'; g.fillRect(0, 0, cv.width, SPEC.banner);
  for (let i = 0; i < SPEC.stages.length; i++) { g.fillStyle = i < S.stage || S.done ? 'rgb(26,154,58)' : 'rgb(70,72,76)'; g.fillRect(cv.width - 30 * (SPEC.stages.length - i), 4, 22, SPEC.banner - 8); }
  g.fillStyle = '#fff'; g.font = '13px "DejaVu Sans", sans-serif'; g.textBaseline = 'middle'; g.textAlign = 'left';
  g.fillText(SPEC.title + (S.done ? '  - completed' : ''), 10, SPEC.banner / 2);
  for (const c of st.controls) {
    const fp = c.font_px || 14; g.font = (c.bold ? 'bold ' : '') + fp + 'px "DejaVu Sans", sans-serif'; g.lineWidth = c.line || 2;
    if (c.kind === 'label') { g.fillStyle = c.color || '#222'; g.textAlign = 'left'; g.fillText(c.label, c.x, c.y + c.h / 2); continue; }
    if (c.kind === 'button' || c.kind === 'tile') { rr(c.x, c.y, c.w, c.h, c.kind === 'tile' ? 10 : 5); g.fillStyle = c.color || '#2a6edc'; g.fill(); if (c.label) { g.fillStyle = c.text_color || '#fff'; g.textAlign = 'center'; g.fillText(c.label, c.x + c.w / 2, c.y + c.h / 2 + 1); } }
    if (c.kind === 'icon') { rr(c.x, c.y, c.w, c.h, 6); g.fillStyle = c.color || '#ffffff'; g.fill(); g.strokeStyle = '#c3c7cf'; g.lineWidth = 1; g.stroke(); g.strokeStyle = c.text_color || '#2b2f36'; g.lineWidth = Math.max(1.5, c.w / 14); g.lineCap = 'round'; g.lineJoin = 'round'; ICONS[c.icon](c.x + c.w / 2, c.y + c.h / 2, Math.min(c.w, c.h) * .78); }
    if (c.kind === 'field') { g.fillStyle = '#fff'; g.fillRect(c.x, c.y, c.w, c.h); g.strokeStyle = S.focus === c.id ? '#2a6edc' : '#9aa0aa'; g.lineWidth = S.focus === c.id ? 2 : 1; g.strokeRect(c.x + .5, c.y + .5, c.w - 1, c.h - 1); g.fillStyle = '#111'; g.textAlign = 'left'; g.fillText((S.fields[c.id] || '') + (S.focus === c.id ? '|' : ''), c.x + 6, c.y + c.h / 2 + 1); }
    if (c.kind === 'checkbox') { g.fillStyle = '#fff'; g.fillRect(c.x, c.y, c.w, c.h); g.strokeStyle = '#555'; g.lineWidth = 1.5; g.strokeRect(c.x + .5, c.y + .5, c.w - 1, c.h - 1); if (S.values[c.id]) { g.strokeStyle = '#1a9a3a'; g.lineWidth = 2.5; g.beginPath(); g.moveTo(c.x + 3, c.y + c.h / 2); g.lineTo(c.x + c.w / 2 - 1, c.y + c.h - 4); g.lineTo(c.x + c.w - 3, c.y + 3); g.stroke(); } }
    if (c.kind === 'slider') { g.fillStyle = '#c9ced6'; g.fillRect(c.x, c.y + c.h / 2 - 3, c.w, 6); const v = S.values[c.id] || 0; g.fillStyle = '#2a6edc'; g.fillRect(c.x, c.y + c.h / 2 - 3, c.w * v, 6); g.beginPath(); g.arc(c.x + c.w * v, c.y + c.h / 2, c.h / 2, 0, 7); g.fill(); }
  }
}
function at(x, y) { const cs = stage().controls; for (let i = cs.length - 1; i >= 0; i--) { const c = cs[i]; if (c.kind === 'label') continue; const pad = c.kind === 'slider' ? c.h / 2 : 0; if (x >= c.x - pad && x < c.x + c.w + pad && y >= c.y && y < c.y + c.h) return c; } return null; }
function activate(c) {
  S.hits.push(c ? c.id : 'none'); if (!c) { S.focus = null; draw(); return; }
  if (c.kind === 'field') S.focus = c.id; else S.focus = null;
  if (c.kind === 'checkbox') S.values[c.id] = !S.values[c.id];
  if (c.id === stage().target) { S.wrong = 0; if (S.stage + 1 >= SPEC.stages.length) { S.done = true; S.stage = SPEC.stages.length; } else S.stage += 1; }
  else if (!S.done && c.kind !== 'slider' && !(c.kind === 'field' && SPEC.free_fields)) S.wrong += 1;
  draw();
}
cv.addEventListener('mousedown', e => { const c = at(e.offsetX, e.offsetY); if (c && c.kind === 'slider') S.drag = c; });
cv.addEventListener('mousemove', e => { if (S.drag) { const c = S.drag; S.values[c.id] = Math.max(0, Math.min(1, (e.offsetX - c.x) / c.w)); draw(); } });
window.addEventListener('mouseup', e => { if (S.drag) { const c = S.drag; S.drag = null; S.hits.push(c.id); if (c.id === stage().target && S.values[c.id] >= (c.goal || 0.8)) { S.done = S.stage + 1 >= SPEC.stages.length; S.stage += 1; } draw(); } });
cv.addEventListener('click', e => { const c = at(e.offsetX, e.offsetY); if (c && c.kind === 'slider') return; activate(c); });
cv.addEventListener('dblclick', e => { const c = at(e.offsetX, e.offsetY); S.dbl = (S.dbl || 0) + 1; S.dbl_on = c ? c.id : 'none'; });
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') { S.stage = 0; S.hits = []; S.done = false; S.wrong = 0; S.fields = {}; S.focus = null; S.values = {}; S.resets += 1; draw(); return; }
  S.last_key = (e.ctrlKey ? 'Control+' : '') + (e.shiftKey && e.key.length > 1 ? 'Shift+' : '') + e.key;
  if (!S.focus) return;
  if (e.ctrlKey && e.key.toLowerCase() === 'a') { S.selected = S.focus; e.preventDefault(); return; }
  if (e.key === 'Backspace') { S.fields[S.focus] = S.selected === S.focus ? '' : (S.fields[S.focus] || '').slice(0, -1); S.selected = null; }
  else if (e.key === 'Enter') { S.submitted = Object.assign({}, S.fields); }
  else if (e.key.length === 1 && !e.ctrlKey) { S.fields[S.focus] = (S.selected === S.focus ? '' : (S.fields[S.focus] || '')) + e.key; S.selected = null; }
  draw();
});
window.addEventListener('scroll', () => { S.scroll_y = window.scrollY; });
window.__draw = draw; window.__spec = SPEC;
draw();
})();
"""

_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;padding:0;background:#f4f5f7}canvas{display:block}.band{height:__BAND__px;font:20px "DejaVu Sans",sans-serif;
padding:20px;box-sizing:border-box;color:#fff}</style></head><body>
<canvas id="c" width="__W__" height="__H__"></canvas>__EXTRA__<script>__JS__</script></body></html>"""


@dataclass
class Task:
    """One benchmark task: a page and, per stage, the description of the control to operate."""
    task_id: str
    family: str
    html: str
    targets: List[str]                               # one description per stage
    spec: Dict[str, Any] = field(default_factory=dict)

    @property
    def stages(self) -> int:
        return len(self.targets)

    def progress_probe(self, stage: int) -> Tuple[int, int]:
        """Pixel inside the progress box of ``stage`` (it is green once that stage has been completed)."""
        return W - 30 * (self.stages - stage) + 11, BANNER_H // 2


def build_page(spec: Dict[str, Any], tall_bands: int = 0) -> str:
    spec = dict(spec)
    spec.setdefault("banner", BANNER_H)
    spec.setdefault("title", "task")
    extra = "".join(f'<div class="band" style="background:hsl({(i * 67) % 360},55%,{35 + (i * 7) % 20}%)">section {i + 1}</div>'
                    for i in range(tall_bands))
    return (_HTML.replace("__JS__", _JS.replace("__SPEC__", json.dumps(spec))).replace("__TITLE__", spec["title"])
            .replace("__W__", str(W)).replace("__H__", str(H)).replace("__EXTRA__", extra).replace("__BAND__", "420"))


def _place(rng: random.Random, n: int, w: int, h: int, margin: int = 24, top: int = BANNER_H + 40,
           right: int = W, bottom: int = H) -> List[Tuple[int, int]]:
    """``n`` non-overlapping top-left corners for boxes of ``w`` x ``h``."""
    out: List[Tuple[int, int]] = []
    for _ in range(4000):
        if len(out) == n:
            break
        x = rng.randrange(margin, right - w - margin)
        y = rng.randrange(top, bottom - h - margin)
        if all(abs(x - px) > w + margin or abs(y - py) > h + margin for px, py in out):
            out.append((x, y))
    if len(out) < n:
        raise ValueError("layout does not fit")
    return out


_WORDS = ["Export", "Archive", "Share", "Duplicate", "Rename", "Publish", "Preview", "Compare", "Restore", "Merge",
          "Upload", "Filter", "Refresh", "Approve", "Schedule", "Download", "Translate", "Validate", "Assign", "Convert"]
_ICON_NAMES = {"search": "search (magnifying glass) icon", "settings": "settings gear icon", "trash": "trash can (delete) icon",
               "home": "home (house) icon", "mail": "mail (envelope) icon", "bell": "notification bell icon",
               "heart": "heart (favourite) icon", "star": "star icon", "plus": "plus (add) icon",
               "menu": "menu icon with three horizontal lines", "close": "close (X) icon", "download": "download arrow icon",
               "user": "user profile (person) icon", "play": "play (triangle) icon"}
_COLOURS = ["#2a6edc", "#7b3fc4", "#0f8a7e", "#c2571a", "#3d4654", "#b0306a"]


def canvas_buttons(seed: int) -> Task:
    rng = random.Random(1000 + seed)
    labels = rng.sample(_WORDS, 7)
    pos = _place(rng, 7, 150, 44)
    controls = [{"id": f"b{i}", "kind": "button", "label": lab, "x": x, "y": y, "w": 150, "h": 44,
                 "color": rng.choice(_COLOURS), "font_px": 15} for i, (lab, (x, y)) in enumerate(zip(labels, pos))]
    t = rng.randrange(7)
    spec = {"title": "canvas buttons", "stages": [{"controls": controls, "target": f"b{t}"}]}
    return Task(f"canvas_buttons-{seed}", "canvas_buttons", build_page(spec), [f'"{labels[t]}" button'], spec)


def icon_only(seed: int) -> Task:
    rng = random.Random(2000 + seed)
    icons = rng.sample(sorted(_ICON_NAMES), 9)
    size = rng.choice([40, 48, 56])
    if rng.random() < 0.35:                          # a vertical side bar
        x0, y0 = rng.randrange(40, W - size - 40), rng.randrange(60, H - 9 * (size + 8) - 10)
        controls = [{"id": f"i{i}", "kind": "icon", "icon": ic, "x": x0, "y": y0 + i * (size + 8), "w": size, "h": size}
                    for i, ic in enumerate(icons)]
    else:                                            # a horizontal tool bar
        x0, y0 = rng.randrange(40, W - 9 * (size + 14) - 40), rng.randrange(80, H - size - 40)
        controls = [{"id": f"i{i}", "kind": "icon", "icon": ic, "x": x0 + i * (size + 14), "y": y0, "w": size, "h": size}
                    for i, ic in enumerate(icons)]
    t = rng.randrange(9)
    spec = {"title": "icon toolbar", "stages": [{"controls": controls, "target": f"i{t}"}]}
    return Task(f"icon_only-{seed}", "icon_only", build_page(spec), [_ICON_NAMES[icons[t]]], spec)


_FIELDS = ["First name", "Last name", "Email", "Phone", "Company", "Job title", "Street", "City", "Postal code",
           "Country", "Website", "Tax number", "Department", "Manager", "Start date", "Cost centre", "Badge number",
           "Office", "Time zone", "Language", "Emergency contact", "Bank account", "Reference", "Notes"]


def dense_form(seed: int) -> Task:
    rng = random.Random(3000 + seed)
    names = rng.sample(_FIELDS, 18)
    controls: List[Dict[str, Any]] = []
    cols, x0, y0, cw, rh = 3, 40, 70, 400, 36
    for i, name in enumerate(names):
        cx, cy = x0 + (i % cols) * cw, y0 + (i // cols) * rh
        controls.append({"id": f"l{i}", "kind": "label", "label": name, "x": cx, "y": cy, "w": 150, "h": 26, "font_px": 13})
        controls.append({"id": f"f{i}", "kind": "field", "x": cx + 150, "y": cy, "w": 220, "h": 26, "font_px": 13})
    opts = rng.sample(["Send me updates", "Remember this device", "Use billing address", "Mark as urgent",
                       "Share with team", "Require approval"], 4)
    for j, name in enumerate(opts):
        cy = y0 + 6 * rh + 24 + j * 30
        controls.append({"id": f"c{j}", "kind": "checkbox", "x": x0, "y": cy, "w": 18, "h": 18})
        controls.append({"id": f"cl{j}", "kind": "label", "label": name, "x": x0 + 28, "y": cy - 4, "w": 300, "h": 26, "font_px": 13})
    controls.append({"id": "save", "kind": "button", "label": "Save", "x": x0, "y": y0 + 6 * rh + 160, "w": 110, "h": 36})
    controls.append({"id": "cancel", "kind": "button", "label": "Cancel", "x": x0 + 130, "y": y0 + 6 * rh + 160, "w": 110,
                     "h": 36, "color": "#8a9099"})
    if rng.random() < 0.7:
        t = rng.randrange(18)
        target, desc = f"f{t}", f'text input field labelled "{names[t]}"'
    else:
        t = rng.randrange(4)
        target, desc = f"c{t}", f'checkbox for "{opts[t]}"'
    spec = {"title": "dense form", "stages": [{"controls": controls, "target": target}]}
    return Task(f"dense_form-{seed}", "dense_form", build_page(spec), [desc], spec)


def small_targets(seed: int) -> Task:
    rng = random.Random(4000 + seed)
    rows = rng.sample([f"Report {n}" for n in range(11, 60)], 9)
    size = rng.choice([12, 14, 16])
    x0, y0 = rng.randrange(60, 520), 80
    controls: List[Dict[str, Any]] = []
    for i, name in enumerate(rows):
        cy = y0 + i * 44
        controls.append({"id": f"l{i}", "kind": "label", "label": name, "x": x0, "y": cy, "w": 200, "h": 24, "font_px": 14})
        controls.append({"id": f"x{i}", "kind": "icon", "icon": "close", "x": x0 + 260, "y": cy + (24 - size) // 2,
                         "w": size, "h": size})
        controls.append({"id": f"s{i}", "kind": "icon", "icon": "star", "x": x0 + 260 + size + 10, "y": cy + (24 - size) // 2,
                         "w": size, "h": size})
    t = rng.randrange(9)
    which = rng.choice(["x", "s"])
    what = "small close (X) button" if which == "x" else "small star button"
    spec = {"title": "small targets", "stages": [{"controls": controls, "target": f"{which}{t}"}]}
    return Task(f"small_targets-{seed}", "small_targets", build_page(spec), [f'{what} in the row "{rows[t]}"'], spec)


def lookalike(seed: int) -> Task:
    rng = random.Random(5000 + seed)
    names = [f"invoice-{rng.randrange(1000, 9999)}" for _ in range(8)]
    verb = rng.choice(["Delete", "Open", "Send", "Edit"])
    x0, y0 = rng.randrange(60, 420), 80
    controls: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        cy = y0 + i * 52
        controls.append({"id": f"l{i}", "kind": "label", "label": name, "x": x0, "y": cy, "w": 220, "h": 34, "font_px": 15})
        controls.append({"id": f"a{i}", "kind": "button", "label": verb, "x": x0 + 260, "y": cy, "w": 96, "h": 34})
        controls.append({"id": f"b{i}", "kind": "button", "label": verb + " all", "x": x0 + 372, "y": cy, "w": 120, "h": 34,
                         "color": "#8a9099"})
    t = rng.randrange(8)
    spec = {"title": "look-alike rows", "stages": [{"controls": controls, "target": f"a{t}"}]}
    return Task(f"lookalike-{seed}", "lookalike", build_page(spec), [f'"{verb}" button in the row of "{names[t]}"'], spec)


def workflow(seed: int) -> Task:
    """Three screens in a row: a start button, a choice among tiles, a confirmation among similar buttons."""
    rng = random.Random(6000 + seed)
    start = rng.choice(["Begin setup", "Start import", "New project", "Create report"])
    (sx, sy), (dx, dy) = _place(rng, 2, 180, 48)
    s1 = [{"id": "start", "kind": "button", "label": start, "x": sx, "y": sy, "w": 180, "h": 48},
          {"id": "help", "kind": "button", "label": "Help", "x": dx, "y": dy, "w": 180, "h": 48, "color": "#8a9099"}]
    tiles = rng.sample(["Basic", "Team", "Business", "Enterprise", "Student", "Trial"], 4)
    pos = _place(rng, 4, 200, 120)
    s2 = [{"id": f"t{i}", "kind": "tile", "label": lab, "x": x, "y": y, "w": 200, "h": 120, "color": _COLOURS[i],
           "font_px": 18} for i, (lab, (x, y)) in enumerate(zip(tiles, pos))]
    pick = rng.randrange(4)
    finals = ["Confirm", "Cancel", "Back"]
    rng.shuffle(finals)
    fx, fy = rng.randrange(80, 700), rng.randrange(200, 640)
    s3 = [{"id": f"k{i}", "kind": "button", "label": lab, "x": fx + i * 150, "y": fy, "w": 130, "h": 42,
           "color": "#2a6edc" if lab == "Confirm" else "#8a9099"} for i, lab in enumerate(finals)]
    s3.insert(0, {"id": "sum", "kind": "label", "label": f"Selected plan: {tiles[pick]}", "x": fx, "y": fy - 60, "w": 400,
                  "h": 30, "font_px": 18})
    spec = {"title": "three-step workflow", "stages": [{"controls": s1, "target": "start"},
                                                        {"controls": s2, "target": f"t{pick}"},
                                                        {"controls": s3, "target": f"k{finals.index('Confirm')}"}]}
    return Task(f"workflow-{seed}", "workflow", build_page(spec),
                [f'"{start}" button', f'"{tiles[pick]}" tile', '"Confirm" button'], spec)


FAMILIES = {"canvas_buttons": canvas_buttons, "icon_only": icon_only, "dense_form": dense_form,
            "small_targets": small_targets, "lookalike": lookalike, "workflow": workflow}


def make_tasks(per_family: int = 10, families: Any = None) -> List[Task]:
    """The benchmark task set: ``per_family`` seeded layouts of every family (deterministic)."""
    out: List[Task] = []
    for name in (families or FAMILIES):
        out += [FAMILIES[name](seed) for seed in range(per_family)]
    return out


# ---- pages for the visual-executor tests ---------------------------------------------------------------------
def interaction_page() -> Tuple[str, Dict[str, Any]]:
    """One canvas with a button, two text fields, a checkbox and a slider (fields do not count as wrong clicks)."""
    controls = [
        {"id": "go", "kind": "button", "label": "Apply", "x": 100, "y": 100, "w": 160, "h": 48},
        {"id": "other", "kind": "button", "label": "Discard", "x": 320, "y": 100, "w": 160, "h": 48, "color": "#8a9099"},
        {"id": "name", "kind": "field", "x": 100, "y": 220, "w": 320, "h": 34},
        {"id": "city", "kind": "field", "x": 100, "y": 280, "w": 320, "h": 34},
        {"id": "agree", "kind": "checkbox", "x": 100, "y": 350, "w": 22, "h": 22},
        {"id": "level", "kind": "slider", "x": 100, "y": 440, "w": 400, "h": 24, "goal": 0.8},
    ]
    spec = {"title": "interaction", "free_fields": True, "stages": [{"controls": controls, "target": "go"}]}
    return build_page(spec), spec


def tall_page(bands: int = 6) -> Tuple[str, Dict[str, Any]]:
    """A canvas screen followed by coloured sections: the document is several screens high."""
    spec = {"title": "tall page", "stages": [{"controls": [
        {"id": "top", "kind": "button", "label": "Top action", "x": 100, "y": 100, "w": 180, "h": 48}], "target": "top"}]}
    return build_page(spec, tall_bands=bands), spec
