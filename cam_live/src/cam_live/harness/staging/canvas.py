"""cam_live.harness.staging.canvas — the record-an-item task rendered only as canvas-drawn controls (workstream EB).

The page contains one ``<canvas>`` and no DOM semantics: every label, field and button is painted, input is taken
from raw mouse and keyboard events. It is the target for screenshot-based (visual) computer use. ``LAYOUT`` is the
single source of truth for control rectangles (also injected into the page script) so tests and grounding benchmarks
have exact ground truth; it is served at ``/layout.json`` and is not linked from the page.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .server import Request, Response, Site, as_json, page
from .sites import CATEGORIES

WIDTH, HEIGHT = 900, 600

#: screen -> control name -> {"rect": [x, y, w, h], "label": text painted on / next to the control, "kind": ...}
LAYOUT: Dict[str, Dict[str, Dict[str, Any]]] = {
    "form": {
        "name_field": {"rect": [60, 120, 400, 40], "label": "Name", "kind": "field"},
        "qty_minus": {"rect": [60, 220, 40, 40], "label": "-", "kind": "button"},
        "qty_value": {"rect": [110, 220, 100, 40], "label": "Quantity", "kind": "field"},
        "qty_plus": {"rect": [220, 220, 40, 40], "label": "+", "kind": "button"},
        "cat_tools": {"rect": [60, 320, 110, 40], "label": "Tools", "kind": "choice"},
        "cat_parts": {"rect": [180, 320, 110, 40], "label": "Parts", "kind": "choice"},
        "cat_misc": {"rect": [300, 320, 110, 40], "label": "Misc", "kind": "choice"},
        "review": {"rect": [60, 420, 140, 48], "label": "Review", "kind": "button"},
    },
    "review": {
        "confirm": {"rect": [60, 420, 140, 48], "label": "Confirm", "kind": "button"},
        "back": {"rect": [220, 420, 140, 48], "label": "Back", "kind": "button"},
    },
    "done": {
        "new_record": {"rect": [60, 420, 180, 48], "label": "New record", "kind": "button"},
    },
}


def center(screen: str, control: str) -> List[int]:
    x, y, w, h = LAYOUT[screen][control]["rect"]
    return [x + w // 2, y + h // 2]


_JS = r"""
const L = __LAYOUT__;
const st = {screen:'form', name:'', qty:1, cat:'', focus:'', fresh:false, saved:0, last:'', error:''};
window.__panel_state = () => JSON.parse(JSON.stringify(st));
let cv, g;
function box(r, fill, stroke){ g.fillStyle=fill; g.fillRect(r[0],r[1],r[2],r[3]); g.strokeStyle=stroke; g.lineWidth=2; g.strokeRect(r[0],r[1],r[2],r[3]); }
function label(t,x,y,size,color){ g.fillStyle=color||'#111'; g.font=(size||18)+'px sans-serif'; g.textBaseline='middle'; g.textAlign='left'; g.fillText(t,x,y); }
function button(c, active){ const r=c.rect; box(r, active?'#2a5db0':'#e4e7ee', '#30405a'); g.fillStyle=active?'#fff':'#111'; g.font='18px sans-serif';
  g.textAlign='center'; g.textBaseline='middle'; g.fillText(c.label, r[0]+r[2]/2, r[1]+r[3]/2); }
function draw(){
  g.fillStyle='#fff'; g.fillRect(0,0,cv.width,cv.height);
  label('Record panel', 60, 50, 28);
  label('Saved records: '+st.saved, 600, 50, 18, '#333');
  if (st.screen==='form'){
    const F=L.form;
    label('Name', 60, 105, 16, '#333'); box(F.name_field.rect, '#fff', st.focus==='name'?'#2a5db0':'#777');
    label(st.name + (st.focus==='name'?'|':''), 70, 140, 18);
    label('Quantity', 60, 205, 16, '#333'); button(F.qty_minus); box(F.qty_value.rect,'#fff', st.focus==='qty'?'#2a5db0':'#777');
    label(String(st.qty) + (st.focus==='qty'?'|':''), 120, 240, 18); button(F.qty_plus);
    label('Category', 60, 305, 16, '#333');
    button(F.cat_tools, st.cat==='tools'); button(F.cat_parts, st.cat==='parts'); button(F.cat_misc, st.cat==='misc');
    button(F.review);
    if (st.error) label(st.error, 60, 500, 18, '#a00');
  } else if (st.screen==='review'){
    label('Review', 60, 110, 22);
    label('Name: '+st.name, 60, 170, 18); label('Quantity: '+st.qty, 60, 210, 18); label('Category: '+st.cat, 60, 250, 18);
    button(L.review.confirm); button(L.review.back);
    if (st.error) label(st.error, 60, 500, 18, '#a00');
  } else {
    label('Recorded '+st.last, 60, 170, 22, '#060'); button(L.done.new_record);
  }
}
function inside(r,x,y){ return x>=r[0] && x<=r[0]+r[2] && y>=r[1] && y<=r[1]+r[3]; }
async function click(x,y){
  st.error='';
  if (st.screen==='form'){
    const F=L.form; st.focus='';
    if (inside(F.name_field.rect,x,y)) st.focus='name';
    else if (inside(F.qty_value.rect,x,y)) { st.focus='qty'; st.fresh=true; }
    else if (inside(F.qty_minus.rect,x,y)) st.qty=Math.max(1,st.qty-1);
    else if (inside(F.qty_plus.rect,x,y)) st.qty=st.qty+1;
    else if (inside(F.cat_tools.rect,x,y)) st.cat='tools';
    else if (inside(F.cat_parts.rect,x,y)) st.cat='parts';
    else if (inside(F.cat_misc.rect,x,y)) st.cat='misc';
    else if (inside(F.review.rect,x,y)) { if (!st.name.trim()) st.error='A name is needed'; else if (!st.cat) st.error='Pick a category'; else st.screen='review'; }
  } else if (st.screen==='review'){
    if (inside(L.review.back.rect,x,y)) st.screen='form';
    else if (inside(L.review.confirm.rect,x,y)) {
      const r = await fetch('/api/items',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:st.name,category:st.cat,quantity:st.qty})});
      const j = await r.json();
      if (j.ok){ st.saved=j.count; st.last=st.name; st.screen='done'; } else st.error='Not saved: '+j.error;
    }
  } else if (inside(L.done.new_record.rect,x,y)) { st.name=''; st.qty=1; st.cat=''; st.screen='form'; }
  draw();
}
function key(e){
  if (st.screen!=='form' || !st.focus) return;
  if (st.focus==='name'){
    if (e.key==='Backspace') st.name=st.name.slice(0,-1); else if (e.key.length===1 && st.name.length<40) st.name+=e.key;
  } else if (st.focus==='qty'){
    let s=String(st.qty);
    if (e.key==='Backspace') s=s.slice(0,-1); else if (/^[0-9]$/.test(e.key) && s.length<6) s = st.fresh ? e.key : s+e.key;
    st.fresh=false; st.qty = Math.max(0, Number(s||'0'));
  }
  e.preventDefault(); draw();
}
window.addEventListener('DOMContentLoaded', ()=>{
  cv=document.getElementById('c'); g=cv.getContext('2d');
  cv.addEventListener('mousedown', (e)=>{ const b=cv.getBoundingClientRect(); click(e.clientX-b.left, e.clientY-b.top); });
  window.addEventListener('keydown', key);
  fetch('/api/state').then(r=>r.json()).then(j=>{ st.saved=j.items.length; draw(); });
  draw();
});
"""


class CanvasSite(Site):
    name = "canvas"
    description = "the record-an-item task painted on a canvas: no DOM semantics, mouse and keyboard only"

    def __init__(self) -> None:
        super().__init__()
        self.items: List[Dict[str, Any]] = []
        self.route("GET", "/", self.home)
        self.route("POST", "/api/items", self.add)
        self.route("GET", "/api/state", lambda q: as_json(self.state()))
        self.route("GET", "/layout.json", lambda q: as_json({"width": WIDTH, "height": HEIGHT, "screens": LAYOUT}))

    def state(self) -> Dict[str, Any]:
        return {"items": [dict(i) for i in self.items], "hits": dict(self.hits)}

    def home(self, req: Request) -> Response:
        head = ("<style>body{margin:0}canvas{display:block}</style><script>"
                + _JS.replace("__LAYOUT__", json.dumps(LAYOUT)) + "</script>")
        return page("Panel", f"<canvas id='c' width='{WIDTH}' height='{HEIGHT}'></canvas>", head=head)

    def add(self, req: Request) -> Response:
        j = req.json() or {}
        name = str(j.get("name", "")).strip()
        if not name or j.get("category") not in dict(CATEGORIES):
            return as_json({"ok": False, "error": "incomplete record"}, 400)
        try:
            qty = int(j.get("quantity"))
        except (TypeError, ValueError):
            return as_json({"ok": False, "error": "bad quantity"}, 400)
        if qty < 1:
            return as_json({"ok": False, "error": "quantity must be at least 1"}, 400)
        self.items.append({"name": name, "category": j["category"], "quantity": qty})
        return as_json({"ok": True, "count": len(self.items)})
