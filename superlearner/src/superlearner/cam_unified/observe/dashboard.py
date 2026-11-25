"""Static learning dashboard over a run directory (spec 18.4; manifest A20 with trace.py).

``build_dashboard(run_dir)`` writes ``<run_dir>/dashboard/index.html`` plus one drill-down page per lifetime
(``lifetime-<n>.html``). Every page is a single self-contained HTML file: inline CSS, inline SVG charts, a few lines
of inline JavaScript for hover tooltips, no external assets, readable offline and in light or dark mode. Each chart
has a table view.

Sections of the index (each states plainly when the run recorded no data for it — nothing is invented):

* active worlds and goals (actor heartbeats: state, weights version, environments, goal text, progress);
* checkpoints (LATEST, versions on disk with validation result and relative parameter change);
* experience counts (per actor, queue totals, dedup, gaps, policy-lag routing and lag histogram);
* learning curves (learner update metrics; per-actor episode scores over wall time);
* world-model calibration (predicted vs observed outcomes from decision traces: reliability diagram and Brier score
  for probabilities, interval coverage and error for continuous predictions);
* retention and transfer (``eval/*.jsonl`` rows ``{"metric", "value", "version"?, "t"?, "family"?}`` written by
  evaluation harnesses);
* real versus imagined outcomes (the imagined outcome of the selected candidate against the realised outcome);
* lifetimes (one row per environment lifetime, linked to its drill-down page: retrospective summary labelled as an
  interpretation, action timeline with predictions vs outcomes, per-decision table with update references);
  traces are streamed once into per-lifetime partitions (`trace.partition_lifetimes`) and processed one lifetime
  at a time, so memory is bounded by the largest lifetime, not the run (calibration and real-vs-imagined statistics
  accumulate incrementally);
* model updates (update log: version, data consumed per actor, lag, metrics, publication);
* compute (merged ComputeMeter snapshots per process and per role).
"""
from __future__ import annotations

import html
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from typing import Optional

from .accounting import merge_snapshots, read_snapshots
from .trace import INTERPRETATION_LABEL, load_partition, partition_lifetimes, retrospective_summary

# ============================================================================ style (reference palette, light + dark)

_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,0.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,0.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#0ca30c;
--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#c98500;--s5:#d55181;--s6:#0ca30c;--s7:#9085e9;--s8:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:14px;margin:16px 0 6px}
p.sub{color:var(--ink2);margin:0 0 16px}.note{color:var(--ink2);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:12px 0 8px;font-size:13px}nav a{color:var(--ink2)}
section{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px}
.tile .l{color:var(--ink2);font-size:12px}.tile .v{font-size:20px;font-weight:600}
.scroll{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num,th.num{text-align:right}
details{margin-top:6px}summary{cursor:pointer;color:var(--ink2);font-size:12.5px}
.chart{position:relative;margin:6px 0}.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin:2px 0 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:5px;vertical-align:-1px}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);
border-radius:6px;padding:4px 8px;font-size:12px;white-space:nowrap;display:none;box-shadow:0 2px 8px rgba(0,0,0,.12)}
.badge{display:inline-block;padding:0 6px;border-radius:8px;font-size:11.5px;border:1px solid var(--ring)}
.ok::before{content:"\\2714\\00a0";color:var(--good)}.bad::before{content:"\\2716\\00a0";color:var(--crit)}
.label{border-left:3px solid var(--axis);padding:4px 10px;color:var(--ink2);font-size:12.5px;margin:6px 0}
code{font-size:12px}a{color:var(--s1)}
"""

_JS = """
document.querySelectorAll('.chart[data-series]').forEach(function(el){
  var d=JSON.parse(el.getAttribute('data-series'));var svg=el.querySelector('svg');var tip=el.querySelector('.tip');
  var cross=svg.querySelector('.cross');
  svg.addEventListener('mousemove',function(ev){var r=svg.getBoundingClientRect();var sx=d.w/r.width;
    var px=(ev.clientX-r.left)*sx;if(px<d.x0||px>d.x1){tip.style.display='none';if(cross)cross.style.display='none';return}
    var xv=d.xmin+(px-d.x0)/(d.x1-d.x0)*(d.xmax-d.xmin);var rows=[];var bx=null;
    d.s.forEach(function(s){var best=null,bd=1e300;s.p.forEach(function(p){var dd=Math.abs(p[0]-xv);if(dd<bd){bd=dd;best=p}});
      if(best){rows.push(s.n+': '+(+best[1].toPrecision(4)));if(bx===null)bx=best[0]}});
    if(bx===null)return;var cx=d.x0+(bx-d.xmin)/((d.xmax-d.xmin)||1)*(d.x1-d.x0);
    if(cross){cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.style.display='block'}
    tip.innerHTML=d.xl+' '+(+bx.toPrecision(5))+'<br>'+rows.join('<br>');tip.style.display='block';
    var left=(cx/sx)+12;if(left>r.width-160)left=(cx/sx)-170;tip.style.left=left+'px';tip.style.top='8px'});
  svg.addEventListener('mouseleave',function(){tip.style.display='none';if(cross)cross.style.display='none'});
});
"""

_SERIES = [f"var(--s{i})" for i in range(1, 9)]


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if abs(v) >= 10_000:
            return f"{v:,.0f}"
        return f"{v:.{digits}g}" if abs(v) < 1 else f"{v:,.{max(0, digits - 1)}f}"
    return _e(v)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0]
    if hi == lo:
        hi = lo + (abs(lo) or 1.0)
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    out = []
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out


def _tick_label(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e4:
        return f"{v / 1e3:g}K"
    if a >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"


def _downsample(pts: list, k: int = 400) -> list:
    if len(pts) <= k:
        return pts
    step = len(pts) / k
    return [pts[int(i * step)] for i in range(k)] + [pts[-1]]


_chart_id = [0]


def line_chart(series: list, x_label: str, y_label: str, y_range: Optional[tuple] = None, w: int = 720,
               h: int = 240, diagonal: bool = False, markers: bool = False) -> str:
    """series: [(name, [(x, y), ...])] (<= 8; first 8 shown). Returns legend + SVG + table view."""
    series = [(n, [(float(x), float(y)) for x, y in p if y is not None and math.isfinite(float(y))])
              for n, p in series[:8]]
    series = [(n, p) for n, p in series if p]
    if not series:
        return '<p class="note">No data points.</p>'
    xs = [x for _, p in series for x, _ in p]
    ys = [y for _, p in series for _, y in p]
    xmin, xmax = min(xs), max(xs)
    ylo, yhi = (min(ys), max(ys)) if y_range is None else y_range
    if diagonal:
        ylo, yhi = min(ylo, xmin), max(yhi, xmax)
    yt = _nice_ticks(ylo, yhi)
    ylo, yhi = min(yt[0], ylo), max(yt[-1], yhi)
    xt = _nice_ticks(xmin, xmax, 6)
    L, R, T, B = 56, 16, 10, 34
    x0, x1, y0, y1 = L, w - R, h - B, T
    sx = (lambda x: x0 + (x - xmin) / ((xmax - xmin) or 1) * (x1 - x0))
    sy = (lambda y: y0 - (y - ylo) / ((yhi - ylo) or 1) * (y0 - y1))
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(y_label)} by {_e(x_label)}">']
    for t in yt:
        if ylo - 1e-12 <= t <= yhi + 1e-12:
            g.append(f'<line x1="{x0}" x2="{x1}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="var(--grid)" '
                     f'stroke-width="1"/><text x="{x0 - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="var(--muted)">{_tick_label(t)}</text>')
    for t in xt:
        if xmin - 1e-12 <= t <= xmax + 1e-12:
            g.append(f'<text x="{sx(t):.1f}" y="{y0 + 16}" text-anchor="middle" font-size="11" '
                     f'fill="var(--muted)">{_tick_label(t)}</text>')
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="var(--axis)" stroke-width="1"/>')
    g.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{h - 4}" text-anchor="middle" font-size="11" '
             f'fill="var(--ink2)">{_e(x_label)}</text>')
    g.append(f'<text x="12" y="{(y0 + y1) / 2:.0f}" text-anchor="middle" font-size="11" fill="var(--ink2)" '
             f'transform="rotate(-90 12 {(y0 + y1) / 2:.0f})">{_e(y_label)}</text>')
    if diagonal:
        lo, hi = max(xmin, ylo), min(xmax, yhi)
        g.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" '
                 f'stroke="var(--axis)" stroke-width="1"/>')
    for i, (n, p) in enumerate(series):
        col = _SERIES[i]
        p = sorted(_downsample(sorted(p)))
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(p))
        if len(p) > 1:
            g.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                     f'stroke-linecap="round"/>')
        pts = p if (markers or len(p) == 1) else [p[-1]]
        for x, y in pts:
            g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{col}" stroke="var(--surface)" '
                     f'stroke-width="2"><title>{_e(n)}: {_e(f"{x:g}")} → {_e(f"{y:.4g}")}</title></circle>')
    g.append(f'<line class="cross" x1="0" x2="0" y1="{y1}" y2="{y0}" stroke="var(--muted)" stroke-width="1" '
             f'style="display:none"/>')
    g.append("</svg>")
    data = {"w": w, "x0": x0, "x1": x1, "xmin": xmin, "xmax": xmax, "xl": x_label,
            "s": [{"n": n, "p": [[x, y] for x, y in _downsample(sorted(p))]} for n, p in series]}
    legend = ""
    if len(series) >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:{_SERIES[i]}"></i>{_e(n)}</span>' for i, (n, _) in enumerate(series)) + \
            "</div>"
    _chart_id[0] += 1
    rows = "".join(f"<tr><td>{_e(n)}</td><td class=num>{x:g}</td><td class=num>{y:.4g}</td></tr>"
                   for n, p in series for x, y in _downsample(sorted(p), 200))
    table = (f'<details><summary>Table view</summary><div class="scroll"><table><tr><th>series</th>'
             f'<th class=num>{_e(x_label)}</th><th class=num>{_e(y_label)}</th></tr>{rows}</table></div></details>')
    return (legend + f'<div class="chart" data-series="{_e(json.dumps(data))}">' + "".join(g) +
            '<div class="tip"></div></div>' + table)


def bar_chart(cats: list, vals: list, x_label: str, y_label: str, w: int = 720, h: int = 200) -> str:
    if not cats:
        return '<p class="note">No data.</p>'
    L, R, T, B = 56, 16, 10, 34
    x0, x1, y0, y1 = L, w - R, h - B, T
    vmax = max(max(vals), 1e-12)
    yt = _nice_ticks(0, vmax)
    top = max(yt[-1], vmax)
    band = (x1 - x0) / len(cats)
    bw = min(24.0, band * 0.7)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(y_label)} by {_e(x_label)}">']
    for t in yt:
        y = y0 - t / top * (y0 - y1)
        g.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
                 f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">'
                 f'{_tick_label(t)}</text>')
    every = max(1, len(cats) // 16)
    for i, (c, v) in enumerate(zip(cats, vals)):
        cx = x0 + band * (i + 0.5)
        bh = v / top * (y0 - y1)
        x = cx - bw / 2
        r = min(4.0, bw / 2, bh)
        if bh > 0:
            g.append(f'<path d="M{x:.1f},{y0} L{x:.1f},{y0 - bh + r:.1f} Q{x:.1f},{y0 - bh:.1f} {x + r:.1f},'
                     f'{y0 - bh:.1f} L{x + bw - r:.1f},{y0 - bh:.1f} Q{x + bw:.1f},{y0 - bh:.1f} {x + bw:.1f},'
                     f'{y0 - bh + r:.1f} L{x + bw:.1f},{y0} Z" fill="var(--s1)"><title>{_e(c)}: {_e(v)}</title>'
                     f'</path>')
        if i % every == 0:
            g.append(f'<text x="{cx:.1f}" y="{y0 + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">'
                     f'{_e(c)}</text>')
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke="var(--axis)"/>')
    g.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{h - 4}" text-anchor="middle" font-size="11" fill="var(--ink2)">'
             f'{_e(x_label)}</text></svg>')
    rows = "".join(f"<tr><td>{_e(c)}</td><td class=num>{_fmt(v)}</td></tr>" for c, v in zip(cats, vals))
    return ('<div class="chart">' + "".join(g) + '</div><details><summary>Table view</summary><div class="scroll">'
            f'<table><tr><th>{_e(x_label)}</th><th class=num>{_e(y_label)}</th></tr>{rows}</table></div></details>')


def _table(headers: list, rows: list, num_cols: Optional[set] = None, limit: Optional[int] = None) -> str:
    num_cols = num_cols or set()
    shown = rows if limit is None else rows[:limit]
    h = "".join(f'<th class="{"num" if i in num_cols else ""}">{_e(c)}</th>' for i, c in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in shown)
    more = f'<p class="note">Showing {len(shown)} of {len(rows)} rows.</p>' if limit and len(rows) > limit else ""
    return f'<div class="scroll"><table><tr>{h}</tr>{body}</table></div>{more}'


def _tiles(items: list) -> str:
    return '<div class="tiles">' + "".join(f'<div class="tile"><div class="l">{_e(l)}</div>'
                                           f'<div class="v">{v}</div></div>' for l, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str, nav: list) -> str:
    links = "".join(f'<a href="#{a}">{_e(t)}</a>' for a, t in nav)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>{_CSS}</style></head>'
            f'<body><main><h1>{_e(title)}</h1><p class="sub">{subtitle}</p><nav>{links}</nav>{body}</main>'
            f'<script>{_JS}</script></body></html>')


# ============================================================================ data collection

def _read_json(path: str, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: str) -> list:
    out = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return out


def collect(run_dir: str) -> dict:
    p = lambda *a: os.path.join(run_dir, *a)          # noqa: E731
    status = {}
    if os.path.isdir(p("status")):
        for fn in sorted(os.listdir(p("status"))):
            if fn.endswith(".json"):
                st = _read_json(p("status", fn))
                if st is not None:
                    status[fn[:-5]] = st
    ckpts = []
    if os.path.isdir(p("ckpt")):
        for fn in sorted(os.listdir(p("ckpt"))):
            if fn.startswith("v") and fn[1:].isdigit():
                m = _read_json(p("ckpt", fn, "meta.json"))
                if m:
                    ckpts.append(m)
    evals = []
    if os.path.isdir(p("eval")):
        for fn in sorted(os.listdir(p("eval"))):
            if fn.endswith(".jsonl"):
                for r in _read_jsonl(p("eval", fn)):
                    r.setdefault("source", fn[:-6])
                    evals.append(r)
    snaps = read_snapshots(p("compute"))
    return {"run": _read_json(p("run.json"), {}), "summary": _read_json(p("summary.json"), {}),
            "status": status, "latest": _read_json(p("ckpt", "LATEST")), "ckpts": ckpts,
            "updates": [u for u in _read_jsonl(p("learner", "updates.jsonl")) if "update_id" in u],
            "rollbacks": [u for u in _read_jsonl(p("learner", "updates.jsonl")) if u.get("rollback")],
            "series": _read_jsonl(p("series.jsonl")),
            "events": _read_jsonl(p("launcher.jsonl")), "evals": evals, "snaps": snaps,
            "compute": merge_snapshots(snaps) if snaps else {}, "traces_root": p("traces")}


# ============================================================================ analyses (pure, testable)

def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class Calibration:
    """Incremental world-model calibration over decision traces (`add` one lifetime at a time; `result`).

    Per outcome key present in both ``predicted`` and ``outcome``: probability-type predictions (all predicted in
    [0,1] and all observed in {0,1}) get a reliability table + Brier score; others get MAE and the coverage of the
    ±1σ / ±2σ intervals implied by ``uncertainty[key]`` (a calibrated Gaussian covers ~68% / ~95%)."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.k: dict = {}

    def add(self, traces) -> None:
        nb = self.n_bins
        for t in traces:
            for key, pv in (t.predicted or {}).items():
                p, o = _num(pv), _num((t.outcome or {}).get(key))
                if p is None or o is None:
                    continue
                u = _num((t.uncertainty or {}).get(key))
                s = self.k.get(key)
                if s is None:
                    s = self.k[key] = {"n": 0, "prob": True, "bins": [[0, 0.0, 0.0] for _ in range(nb)],
                                       "brier": 0.0, "obs": 0.0, "err": 0.0, "nu": 0, "c1": 0, "c2": 0}
                s["n"] += 1
                s["prob"] = s["prob"] and 0.0 <= p <= 1.0 and o in (0.0, 1.0)
                if 0.0 <= p <= 1.0:
                    bb = s["bins"][min(nb - 1, int(p * nb))]
                    bb[0] += 1
                    bb[1] += p
                    bb[2] += o
                s["brier"] += (p - o) ** 2
                s["obs"] += o
                s["err"] += abs(p - o)
                if u is not None and u > 0:
                    s["nu"] += 1
                    s["c1"] += abs(o - p) <= u
                    s["c2"] += abs(o - p) <= 2 * u

    def result(self) -> dict:
        out = {}
        for key, s in self.k.items():
            n = s["n"]
            if s["prob"]:
                rel = [(bb[1] / bb[0], bb[2] / bb[0], bb[0]) for bb in s["bins"] if bb[0] > 0]
                base = s["obs"] / n
                out[key] = {"type": "probability", "n": n, "brier": s["brier"] / n,
                            "brier_climatology": base * (1 - base), "reliability": rel,
                            "ece": sum(abs(mp - mo) * c for mp, mo, c in rel) / n}
            else:
                nu = s["nu"]
                out[key] = {"type": "continuous", "n": n, "mae": s["err"] / n,
                            "coverage_1sd": s["c1"] / nu if nu else None, "coverage_2sd": s["c2"] / nu if nu else None,
                            "n_with_uncertainty": nu}
        return out


def calibration(traces: list, n_bins: int = 10) -> dict:
    c = Calibration(n_bins)
    c.add(traces)
    return c.result()


class RealVsImagined:
    """Incremental comparison of the imagined outcome of the selected candidate (matched by tool and args) with the
    realised outcome, per (key, horizon): running sums for n / means / MAE, and a uniform reservoir of at most
    ``max_pairs`` (t_sim, imagined, real) points per key for charts."""

    def __init__(self, max_pairs: int = 20000, seed: int = 0):
        self.max_pairs = max_pairs
        self.rng = random.Random(seed)
        self.k: dict = {}

    def add(self, traces) -> None:
        for t in traces:
            if not t.imagined or not t.selected:
                continue
            idx = None
            for i, c in enumerate(t.candidates or []):
                a = c.get("action", {})
                if a.get("tool") == t.selected.get("tool") and (a.get("args") or {}) == (t.selected.get("args") or {}):
                    idx = i
                    break
            for im in t.imagined:
                if idx is not None and im.get("candidate") != idx:
                    continue
                for key, v in (im.get("outcomes") or {}).items():
                    iv, rv = _num(v), _num((t.outcome or {}).get(key))
                    if iv is None or rv is None:
                        continue
                    s = self.k.setdefault((key, im.get("horizon", 1)), {"n": 0, "si": 0.0, "sr": 0.0, "sa": 0.0,
                                                                        "pairs": []})
                    s["n"] += 1
                    s["si"] += iv
                    s["sr"] += rv
                    s["sa"] += abs(iv - rv)
                    if len(s["pairs"]) < self.max_pairs:
                        s["pairs"].append((t.t_sim, iv, rv))
                    else:
                        j = self.rng.randrange(s["n"])
                        if j < self.max_pairs:
                            s["pairs"][j] = (t.t_sim, iv, rv)

    def result(self) -> dict:
        return {f"{k}@h{hz}": {"n": s["n"], "mean_imagined": s["si"] / s["n"], "mean_real": s["sr"] / s["n"],
                               "mae": s["sa"] / s["n"], "pairs": s["pairs"]} for (k, hz), s in self.k.items()}


def real_vs_imagined(traces: list) -> dict:
    """For traces with imagined trajectories: the imagined outcome of the selected candidate (matched by tool and
    args) vs the realised outcome, per outcome key -> {"n", "mean_imagined", "mean_real", "mae", "pairs": [...]}."""
    r = RealVsImagined()
    r.add(traces)
    return r.result()


def _rolling(pts: list, k: int) -> list:
    if not pts:
        return []
    pts = sorted(pts)
    out, acc = [], 0.0
    q = []
    for x, y in pts:
        q.append(y)
        acc += y
        if len(q) > k:
            acc -= q.pop(0)
        out.append((x, acc / len(q)))
    return out


# ============================================================================ rendering

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def build_dashboard(run_dir: str, out_dir: Optional[str] = None, max_lifetime_pages: int = 64,
                    title: Optional[str] = None) -> str:
    """Render the dashboard; returns the path of index.html."""
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "dashboard")
    os.makedirs(out_dir, exist_ok=True)
    D = collect(run_dir)
    # one streaming pass partitions the traces by lifetime; each lifetime is then loaded, analysed and rendered alone
    part_dir = os.path.join(out_dir, ".partitions")
    shutil.rmtree(part_dir, ignore_errors=True)
    parts = partition_lifetimes(D["traces_root"], part_dir)
    untagged = os.path.join(part_dir, "_untagged.jsonl")
    cal_acc, rvi_acc = Calibration(), RealVsImagined()
    life_files, life_rows = {}, []
    for i, env_id in enumerate(sorted(parts)):
        ts = load_partition(parts[env_id], untagged)
        if not ts:
            continue
        cal_acc.add(ts)
        rvi_acc.add(ts)
        if len(life_files) < max_lifetime_pages:
            fn = life_files[env_id] = f"lifetime-{len(life_files):03d}-{_safe(env_id)}.html"
            with open(os.path.join(out_dir, fn), "w") as fh:
                fh.write(lifetime_page(env_id, ts, run_dir))
        sm = retrospective_summary(ts)
        ph = sm["phases"]
        link = (f'<a href="{_e(life_files[env_id])}">{_e(env_id)}</a>' if env_id in life_files else _e(env_id))
        life_rows.append([link, _fmt(len(ts)),
                          _e(f"{sm['versions'][0]}..{sm['versions'][-1]}" if sm["versions"] else ""),
                          _fmt(ph[0]["mean_reward"] if ph else None), _fmt(ph[-1]["mean_reward"] if ph else None),
                          _fmt(len({u for t in ts for u in t.update_refs}))])
        del ts
    shutil.rmtree(part_dir, ignore_errors=True)
    secs, nav = [], []

    def section(anchor: str, head: str, body: str):
        nav.append((anchor, head))
        secs.append(f'<section id="{anchor}"><h2>{_e(head)}</h2>{body}</section>')

    status = D["status"]
    actors = {k: v for k, v in status.items() if "actor_id" in v}
    learner = status.get("learner", {})
    comp = D["compute"]
    run = D["run"]
    t_start = run.get("t_start") or (min((s.get("t_start", time.time()) for s in status.values()), default=None))
    # ---- overview tiles
    tiles = [("Decisions", _fmt(int(sum(a.get("decisions", 0) for a in actors.values())))),
             ("Experience records", _fmt(int(sum(a.get("records", 0) for a in actors.values())))),
             ("Current checkpoint", _fmt(D["latest"]["version"]) if D["latest"] else "none"),
             ("Learner updates", _fmt(int(learner.get("counts", {}).get("updates", len(
                 [u for u in D["updates"] if u.get("trained")])))) ),
             ("Actors", _fmt(len(actors))),
             ("CPU seconds", _fmt(comp.get("cpu_s"))), ("GPU seconds", _fmt(comp.get("gpu_s")))]
    overview = _tiles(tiles)

    # ---- active worlds and goals
    rows = []
    for aid, a in sorted(actors.items()):
        age = time.time() - a.get("t", time.time())
        goals = "; ".join(sorted(set(a.get("goals", []))))
        rows.append([_e(aid), _e(a.get("state")), _fmt(a.get("version")), _e(", ".join(a.get("env_ids", [])[:6])),
                     _e(goals[:300]), _fmt(a.get("episodes")), _fmt(a.get("decisions")), _fmt(a.get("recent_score")),
                     _fmt(a.get("refreshes")), _fmt(a.get("rebuilds")), f"{age:.0f}s ago"])
    section("worlds", "Active worlds and goals",
            _table(["actor", "state", "weights version", "environments", "goal", "episodes", "decisions",
                    "recent score", "refreshes", "latent rebuilds", "heartbeat"], rows, {2, 5, 6, 7, 8, 9})
            if rows else '<p class="note">No actor heartbeats recorded (status/*.json).</p>')

    # ---- checkpoints
    body = ""
    if D["latest"]:
        lt = D["latest"]
        body += (f'<p>LATEST → version <b>{_fmt(lt["version"])}</b> (<code>{_e(lt["dir"])}</code>, sha256 '
                 f'<code>{_e(lt["sha256"][:16])}</code>…).</p>')
    if D["ckpts"]:
        body += _table(["version", "published", "validated", "checks / reason", "Δθ relative", "parameters",
                        "bytes"],
                       [[_fmt(m["version"]), time.strftime("%H:%M:%S", time.localtime(m["t"])),
                         f'<span class="{"ok" if m.get("validated") else "bad"}">{"yes" if m.get("validated") else "no"}'
                         f"</span>", _e((m.get("validation") or {}).get("reason")), _fmt(m.get("rel_change")),
                         _fmt(m.get("n_params")), _fmt(m.get("bytes"))] for m in D["ckpts"]], {0, 4, 5, 6})
        rc = [(m["version"], m["rel_change"]) for m in D["ckpts"] if m.get("rel_change") is not None]
        if len(rc) >= 2:
            body += "<h3>Relative parameter change per published version (versions still on disk)</h3>"
            body += line_chart([("‖Δθ‖/‖θ‖", rc)], "version", "relative change", markers=True)
    pub = [(u["version"], 1.0) for u in D["updates"] if u.get("published")]
    if pub:
        body += f'<p class="note">{len(pub)} validated publications recorded in the update log.</p>'
    section("checkpoints", "Checkpoints", body or '<p class="note">No checkpoints published.</p>')

    # ---- experience counts
    body = ""
    rows = []
    for aid, a in sorted(actors.items()):
        w = a.get("writer", {})
        rows.append([_e(aid), _fmt(a.get("decisions")), _fmt(a.get("delayed")), _fmt(a.get("records")),
                     _fmt(w.get("segments")), _fmt(w.get("bytes")), _fmt(w.get("blocked_s")),
                     _fmt(w.get("shed_items")), _fmt(a.get("env_failures")), _fmt(a.get("truncated_episodes"))])
    if rows:
        body += _table(["actor", "decisions", "delayed outcomes", "records", "segments", "bytes", "blocked s",
                        "shed records", "env failures", "truncated episodes"], rows, set(range(1, 10)))
    q = learner.get("queue", {})
    if q:
        tq = q.get("totals", {})
        body += "<h3>Learner ingestion</h3>" + _tiles([
            ("Ingested events", _fmt(tq.get("events"))), ("Ingested traces", _fmt(tq.get("traces"))),
            ("Duplicates dropped", _fmt(tq.get("dup"))), ("Delayed outcomes", _fmt(tq.get("delayed"))),
            ("Quarantined segments", _fmt(tq.get("quarantined"))), ("Pending segments", _fmt(q.get("pending"))),
            ("Archive records", _fmt(learner.get("archive_count")))])
        gaps = q.get("gaps") or {}
        body += (f'<p class="note">Sequence gaps (segments never ingested): '
                 f'{_e(json.dumps(gaps)) if gaps else "none"}.</p>')
    router = learner.get("router", {})
    if router:
        c = router.get("counts", {})
        body += (f"<h3>Policy-lag routing (max lag {router.get('max_lag')}, mode {_e(router.get('mode'))})</h3>"
                 + _tiles([("On-policy decisions", _fmt(c.get("on_policy", 0))),
                           ("Off-policy (V-trace)", _fmt(c.get("off_policy", 0))),
                           ("Discarded (too stale)", _fmt(c.get("discarded", 0)))]))
        hist = router.get("lag_hist", {})
        if hist:
            ks = sorted(hist, key=lambda x: int(x))
            body += "<h3>Policy lag of trained and discarded decisions</h3>" + bar_chart(
                ks, [hist[k] for k in ks], "lag (learner updates behind)", "decisions")
    section("experience", "Experience counts", body or '<p class="note">No experience recorded.</p>')

    # ---- learning curves
    body = ""
    ups = [u for u in D["updates"] if u.get("trained")]
    if ups:
        keys = [k for k in ("mean_reward", "loss", "v_loss", "entropy", "payoff_loss", "world", "actor")
                if any(_num(u.get("metrics", {}).get(k)) is not None for u in ups)]
        for k in keys:
            pts = [(u["version"], _num(u["metrics"].get(k))) for u in ups]
            body += f"<h3>{_e(k)} per learner update</h3>" + line_chart(
                [(k, [p for p in pts if p[1] is not None])], "learner version", k)
    ser = []
    for aid, a in sorted(actors.items()):
        log = a.get("episode_log") or []
        if log and t_start:
            ser.append((aid, _rolling([(r[4] - t_start, r[3]) for r in log], 10)))
    if ser:
        if len(ser) > 8:
            allp = sorted(p for _, s in ser for p in s)
            ser = [("mean over actors", _rolling(allp, 10 * len(ser)))]
        body += ("<h3>Episode score by actor (rolling mean of 10 episodes)</h3>" +
                 line_chart(ser, "wall seconds since start", "episode score"))
    if D["series"]:
        pts = [(r["t_rel"], r["decisions"]) for r in D["series"] if r.get("decisions") is not None]
        body += "<h3>Cumulative decisions (all actors)</h3>" + line_chart([("decisions", pts)], "wall seconds",
                                                                          "decisions")
    section("learning", "Learning curves", body or '<p class="note">No learner updates recorded yet.</p>')

    # ---- calibration
    cal = cal_acc.result()
    body = ""
    for k, c in sorted(cal.items()):
        if c["type"] == "probability":
            body += (f"<h3>{_e(k)}: reliability (n={c['n']:,}, Brier {c['brier']:.4f} vs climatology "
                     f"{c['brier_climatology']:.4f}, ECE {c['ece']:.4f})</h3>")
            body += line_chart([("observed frequency", [(mp, mo) for mp, mo, _ in c["reliability"]])],
                               "predicted probability", "observed frequency", y_range=(0, 1), diagonal=True,
                               markers=True)
        else:
            body += _table(["outcome", "n", "mean |error|", "±1σ coverage", "±2σ coverage"],
                           [[_e(k), _fmt(c["n"]), _fmt(c["mae"]), _fmt(c["coverage_1sd"]),
                             _fmt(c["coverage_2sd"])]], {1, 2, 3, 4})
    section("calibration", "World-model calibration",
            body or '<p class="note">No decision trace has both a prediction and an observed value for the same '
                    'outcome key.</p>')

    # ---- retention / transfer
    body = ""
    if D["evals"]:
        by = defaultdict(list)
        for r in D["evals"]:
            x = r.get("version", r.get("t"))
            if _num(r.get("value")) is not None and x is not None:
                by[(r.get("metric", "value"), r.get("family", ""))].append((x, r["value"]))
        metrics = sorted({m for m, _ in by})
        for m in metrics:
            ser = [(f or m, pts) for (mm, f), pts in sorted(by.items()) if mm == m]
            body += f"<h3>{_e(m)}</h3>" + line_chart(ser, "version", m, markers=True)
    section("retention", "Retention and transfer",
            body or '<p class="note">No retention or transfer evaluations recorded in this run '
                    '(<code>eval/*.jsonl</code>).</p>')

    # ---- real vs imagined
    rvi = rvi_acc.result()
    body = ""
    if rvi:
        body += _table(["outcome@horizon", "n", "mean imagined", "mean real", "mean |imagined − real|"],
                       [[_e(k), _fmt(v["n"]), _fmt(v["mean_imagined"]), _fmt(v["mean_real"]), _fmt(v["mae"])]
                        for k, v in sorted(rvi.items())], {1, 2, 3, 4})
        for k, v in sorted(rvi.items())[:3]:
            pts = sorted(v["pairs"])
            body += f"<h3>{_e(k)}: imagined vs real over simulated time (rolling mean of 50)</h3>" + line_chart(
                [("imagined", _rolling([(t, i) for t, i, _ in pts], 50)),
                 ("real", _rolling([(t, r) for t, _, r in pts], 50))], "simulated days", k)
    section("imagined", "Real versus imagined outcomes",
            body or '<p class="note">No imagined trajectories were recorded in the decision traces.</p>')

    # ---- lifetimes
    rows = life_rows
    section("lifetimes", "Lifetimes",
            (_table(["lifetime (environment)", "decisions", "policy versions", "mean reward, first quarter",
                     "mean reward, last quarter", "updates that used it"], rows, {1, 3, 4, 5}, limit=200)
             if rows else '<p class="note">No decision traces recorded (traces/).</p>'))

    # ---- updates
    rows = []
    for u in D["updates"][-200:][::-1]:
        lag = u.get("lag") or {}
        m = u.get("metrics") or {}
        rows.append([_e(u["update_id"]), _fmt(u["version"]), _fmt(u.get("t_rel")), _fmt(u.get("n_decisions")),
                     _e(", ".join(f"{a}:{n}" for a, n in sorted((u.get("by_actor") or {}).items()))),
                     _fmt(lag.get("n_on_policy")), _fmt(lag.get("n_off_policy")), _fmt(lag.get("n_discarded")),
                     _fmt(lag.get("lag_mean")), _fmt(m.get("loss")), _fmt(m.get("mean_reward")),
                     '<span class="ok">yes</span>' if u.get("published") else "no"])
    rb = D.get("rollbacks") or []
    rb_note = "".join(f'<p class="note">Learner restart resumed from version {_fmt(r.get("resumed_version"))}: '
                      f'updates {_e(r.get("rolled_back_versions"))} had not been published, so their '
                      f'{_fmt(r.get("n_records"))} consumed records were rolled back and re-trained.</p>' for r in rb)
    section("updates", "Model updates", rb_note +
            _table(["update", "version", "t (s)", "decisions", "by actor", "on-policy decisions",
                    "off-policy (V-trace) decisions", "not trained (lag/logp)", "mean lag", "loss", "mean reward",
                    "published"], rows,
                   {1, 2, 3, 5, 6, 7, 8, 9, 10}) if rows else '<p class="note">No updates recorded.</p>')

    # ---- compute
    body = ""
    if comp:
        body += _tiles([("Wall seconds", _fmt(comp.get("wall_s"))), ("CPU seconds", _fmt(comp.get("cpu_s"))),
                        ("GPU seconds", _fmt(comp.get("gpu_s"))), ("Env steps", _fmt(comp.get("env_steps"))),
                        ("Decisions", _fmt(comp.get("decisions"))), ("Simulated days", _fmt(comp.get("sim_days"))),
                        ("Prompt tokens", _fmt(comp.get("tokens_prompt"))),
                        ("Generated tokens", _fmt(comp.get("tokens_generated"))),
                        ("Peak RSS max (MB)", _fmt(comp.get("peak_rss_mb_max"))),
                        ("Peak GPU max (MB)", _fmt(comp.get("peak_gpu_mb_max"))),
                        ("Monetary cost (USD)", _fmt(comp.get("cost_usd")) if comp.get("cost_usd") is not None
                         else "no price set")])
        body += _table(["process", "role", "pid", "wall s", "CPU s", "GPU s", "decisions", "records", "updates",
                        "tokens", "peak RSS MB", "peak GPU MB"],
                       [[_e(s["name"]), _e(s.get("role")), _fmt(s.get("pid")), _fmt(s.get("wall_s")),
                         _fmt(s.get("cpu_s")), _fmt(s.get("gpu_s")), _fmt(s.get("decisions")), _fmt(s.get("records")),
                         _fmt(s.get("updates")), _fmt((s.get("tokens_prompt") or 0) + (s.get("tokens_generated") or 0)),
                         _fmt(s.get("peak_rss_mb")), _fmt(s.get("peak_gpu_mb"))] for s in D["snaps"]],
                       set(range(2, 12)))
    section("compute", "Compute", body or '<p class="note">No compute snapshots (compute/*.json).</p>')

    ev = D["events"]
    restarts = [e for e in ev if e.get("event") == "restart"]
    if ev:
        section("supervision", "Supervision events",
                (f"<p>{len(restarts)} restart(s).</p>" if restarts else "<p>No restarts.</p>") +
                _table(["time", "event", "process", "details"],
                       [[time.strftime("%H:%M:%S", time.localtime(e["t"])), _e(e["event"]), _e(e["name"]),
                         _e(json.dumps({k: v for k, v in e.items() if k not in ("t", "event", "name")}))]
                        for e in ev], limit=200))
    subtitle = (f"Run <code>{_e(run_dir)}</code>" + (f", started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}"
                                                     if t_start else "") +
                f". Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from the files in the run directory.")
    index = _page(title or "CAM learning dashboard", subtitle, overview + "".join(secs), nav)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as fh:
        fh.write(index)
    return path


def lifetime_page(env_id: str, traces: list, run_dir: str = "", max_rows: int = 1000) -> str:
    s = retrospective_summary(traces)
    body = [f'<section id="summary"><h2>Retrospective summary</h2><p class="label">{_e(INTERPRETATION_LABEL)}</p>'
            "<ul>" + "".join(f"<li>{_e(x)}</li>" for x in s["statements"]) + "</ul>"]
    if s["phases"]:
        body.append(_table(["from t", "to t", "decisions", "versions", "top actions", "success rate", "mean reward",
                            "updates"],
                           [[_fmt(p["from_t"]), _fmt(p["to_t"]), _fmt(p["n"]), _e(p["versions"]),
                             _e(", ".join(f"{a}×{c}" for a, c in p["top_actions"])), _fmt(p["ok_rate"]),
                             _fmt(p["mean_reward"]), _fmt(p["updates_consumed"])] for p in s["phases"]],
                           {0, 1, 2, 5, 6, 7}))
    body.append("</section>")
    # action timeline: action index over time; predicted vs observed for the most common key
    acts = sorted({(t.selected or {}).get("tool", "") for t in traces})
    if acts and len(acts) <= 40:
        aidx = {a: i for i, a in enumerate(acts)}
        pts = [(t.t_sim, aidx[(t.selected or {}).get("tool", "")]) for t in traces]
        body.append('<section id="timeline"><h2>Action timeline</h2><p class="note">y = action ('
                    + ", ".join(f"{i}: {_e(a)}" for a, i in aidx.items()) + ")</p>"
                    + line_chart([("action", pts)], "simulated days", "action index", markers=True) + "</section>")
    keys = Counter(k for t in traces for k in (t.predicted or {}) if _num((t.outcome or {}).get(k)) is not None)
    if keys:
        k = keys.most_common(1)[0][0]
        pp = [(t.t_sim, _num(t.predicted.get(k))) for t in traces if _num((t.outcome or {}).get(k)) is not None]
        oo = [(t.t_sim, _num(t.outcome.get(k))) for t in traces if _num((t.outcome or {}).get(k)) is not None]
        win = max(1, len(pp) // 40)
        body.append(f'<section id="predictions"><h2>Predicted vs observed: {_e(k)}</h2><p class="note">Rolling '
                    f"mean over {win} decisions.</p>" + line_chart([("predicted", _rolling(pp, win)),
                                                                    ("observed", _rolling(oo, win))],
                                                                   "simulated days", k) + "</section>")
    rows = []
    for t in traces[:max_rows]:
        sel = t.selected or {}
        delayed = (t.outcome or {}).get("delayed") or []
        rows.append([_fmt(t.t_sim), _fmt(t.policy_version), _e(sel.get("tool")),
                     _e(json.dumps(sel.get("args") or {})[:80]),
                     _e(", ".join(f"{k}={_fmt(_num(v))}" for k, v in (t.predicted or {}).items())),
                     _e(", ".join(f"{k}=±{_fmt(_num(v))}" for k, v in (t.uncertainty or {}).items())),
                     _e(", ".join(f"{k}={v}" for k, v in (t.outcome or {}).items()
                                  if k not in ("delayed",) and not isinstance(v, (list, dict)))[:160]),
                     _fmt(len(delayed)), _e(", ".join(t.update_refs[:3]) + ("…" if len(t.update_refs) > 3 else ""))])
    body.append('<section id="decisions"><h2>Decisions</h2>' + _table(
        ["t (days)", "version", "action", "args", "predicted", "uncertainty", "outcome", "delayed", "updates"],
        rows, {0, 1, 7}) + (f'<p class="note">First {max_rows} of {len(traces)} decisions.</p>'
                            if len(traces) > max_rows else "") + "</section>")
    nav = [("summary", "Summary"), ("timeline", "Action timeline"), ("predictions", "Predictions"),
           ("decisions", "Decisions")]
    return _page(f"Lifetime {env_id}", f'<a href="index.html">← dashboard</a> · {len(traces)} decisions · run '
                                        f"<code>{_e(run_dir)}</code>", "".join(body), nav)


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="cam_unified.observe.dashboard")
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    print(build_dashboard(a.run_dir, a.out))


if __name__ == "__main__":
    _main()
