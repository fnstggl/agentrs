"""Public-information reference for generated worlds with a wandering optimum (evaluator side; scripted, labelled as
such in evidence). MASTER_SPEC Part 7.4 "scripted positive controls"; T.4.

`play(view, env)` is the mechanism-specific reference of generated families whose structure has the driver
`wandering_optimum` (`compose._wandering`): the setting the market wants moves along a hidden, smooth path during the
whole episode and sales fall off away from it. The reference knows that much about the mechanism (as every family's
reference knows its family's mechanism) and nothing else: it acts only through the `PublicView` - the levers'
public ranges and defaults (from the tool schema), the observed net income of each tick - never the path, the
width of the fit, the centre or any other hidden parameter, and it never replays copies of the scored world.

Strategy (a scan, then a perturbation tracker that rescans when it has lost the optimum). The public default says
nothing about where the market's taste is (`compose._wandering` draws it independently of every hidden parameter,
DEF-N-8), so the reference first scans: it holds `SCAN` evenly spaced settings of each lever for one tick each
(the other levers at their defaults) and starts every lever at its best scanned setting. Then, one lever per
three-tick cycle: for lever j it holds m_j + d_j, then m_j - d_j, then m_j for one tick each, estimates the slope and
curvature of observed net income in that lever from the three ticks, and moves m_j to the vertex of the fitted
parabola when the response is concave there (clipped to two probe widths) or by one and a half probe widths uphill
when it is not. The probe width shrinks while consecutive moves keep their direction small and grows when the
tracker is far from the top (convex response). The next lever to probe is the one whose recent first-order gain per
cycle times the ticks since it was last probed is largest. When a cycle's net income falls below `LOST` of the best
cycle of the last `MEMORY` ticks for two cycles in a row (the optimum moved away faster than the tracker followed),
it scans again. It never stops probing, so it keeps following an optimum that keeps moving. Nothing hidden is read:
only the public lever ranges and the observed net income of each tick.
"""
from __future__ import annotations

import re

import numpy as np

from ...rapid.identify import _net

_DEFAULT = re.compile(r"default\s+(-?[0-9.eE+-]+)")


def _levers(view, agent) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    sp = view.action_space(agent)
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    d0 = []
    for n, a, b in zip(names, lo, hi):
        m = _DEFAULT.search(str(props[n].get("description", "")))
        d0.append(float(m.group(1)) if m else (a + b) / 2.0)
    return names, lo, hi, np.clip(np.array(d0), lo, hi)


SCAN = 6                                                # settings per lever in a scan (evenly spaced)
LOST = 0.35                                             # a cycle below this share of the recent best counts as lost
MEMORY = 30                                             # ticks over which the recent best cycle is remembered
PROBE = (0.05, 0.02, 0.1)                               # probe width: initial, smallest, largest (share of the range)


def _scan(view, agent, names, lo, hi, m) -> np.ndarray | None:
    """Hold SCAN evenly spaced settings of each lever for one tick (others at m); m with each lever at its best
    scanned setting (None when the episode ended during the scan)."""
    m = m.copy()
    for j in range(len(names)):
        best, bx = -np.inf, m[j]
        for x in lo[j] + (np.arange(SCAN) + 0.5) / SCAN * (hi[j] - lo[j]):
            if view.done:
                return None
            u = m.copy()
            u[j] = x
            r = view.step({"kind": "levers", "values": {n: float(v) for n, v in zip(names, u)}})
            obs = r.get("observation") or (r.get("observations") or {}).get(agent) or view.observe(agent)
            y = _net(obs)
            if y > best:
                best, bx = y, float(x)
        m[j] = bx
    return m


def play(view, env=None) -> None:
    agent = view.agents()[0]
    names, lo, hi, m = _levers(view, agent)
    span = hi - lo
    m = _scan(view, agent, names, lo, hi, m)
    if m is None:
        return
    d = PROBE[0] * span                                 # probe width per lever (lever units)
    j, phase, ys = 0, 0, [0.0, 0.0, 0.0]
    last = np.zeros(len(names))
    act = np.full(len(names), np.inf)                   # recent gain per probe cycle of each lever (untried: inf)
    since = np.zeros(len(names))                        # ticks since each lever was last probed
    recent: list[tuple[int, float]] = []                # (tick, mean net income) of recent cycles
    lost = 0
    while not view.done:
        u = m.copy()
        if phase == 0:
            u[j] = m[j] + d[j]
        elif phase == 1:
            u[j] = m[j] - d[j]
        u = np.clip(u, lo, hi)
        r = view.step({"kind": "levers", "values": {n: float(x) for n, x in zip(names, u)}})
        obs = r.get("observation") or (r.get("observations") or {}).get(agent) or view.observe(agent)
        ys[phase] = _net(obs)
        phase += 1
        if phase < 3:
            continue
        phase = 0
        yp, ym, y0 = ys
        g = (yp - ym) / (2.0 * d[j])
        h = (yp + ym - 2.0 * y0) / (d[j] * d[j])
        if h < 0:
            step = float(np.clip(-g / h, -2.0 * d[j], 2.0 * d[j]))
        else:                                           # convex or flat here: far from the top, climb faster
            step = float(np.sign(g)) * 1.5 * d[j]
            d[j] = min(d[j] * 1.25, PROBE[2] * span[j])
        if step * last[j] < 0:                          # direction flipped: close to the top, probe finer
            d[j] = max(d[j] * 0.8, PROBE[1] * span[j])
        last[j] = step
        m[j] = float(np.clip(m[j] + step, lo[j], hi[j]))
        gain = abs(g * step)
        act[j] = gain if not np.isfinite(act[j]) else 0.7 * act[j] + 0.3 * gain
        since += 3
        since[j] = 0
        j = int(np.argmax(np.where(np.isfinite(act), act, np.inf) * (since + 1)))
        # lost the optimum? (two cycles in a row far below the best recent cycle): scan again
        t = int(view.t)
        y = float(np.mean(ys))
        recent = [(k, v) for k, v in recent if t - k <= MEMORY] + [(t, y)]
        top = max(v for _, v in recent)
        lost = lost + 1 if top > 0 and y < LOST * top else 0
        if lost >= 2:
            m2 = _scan(view, agent, names, lo, hi, m)
            if m2 is None:
                return
            m, lost, recent = m2, 0, []
            d = PROBE[0] * span
            last[:] = 0.0
