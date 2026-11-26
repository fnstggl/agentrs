"""Public-information reference for generated worlds with a wandering optimum (evaluator side; scripted, labelled as
such in evidence). MASTER_SPEC Part 7.4 "scripted positive controls"; T.4.

`play(view, env)` is the mechanism-specific reference of generated families whose structure has the driver
`wandering_optimum` (`compose._wandering`): the setting the market wants moves along a hidden, smooth path during the
whole episode and sales fall off away from it. The reference knows that much about the mechanism (as every family's
reference knows its family's mechanism) and nothing else: it acts only through the `PublicView` - the levers'
public ranges and defaults (from the tool schema), the observed net income of each tick - never the path, the
width of the fit, the centre or any other hidden parameter, and it never replays copies of the scored world.

Strategy (a perturbation tracker, one lever per three-tick cycle, starting at the public defaults): for lever j it holds
m_j + d_j, then m_j - d_j, then m_j for one tick each, estimates the slope and curvature of observed net income in
that lever from the three ticks, and moves m_j to the vertex of the fitted parabola when the response is concave
there (clipped to two probe widths) or by one and a half probe widths uphill when it is not. The probe width shrinks
while consecutive moves keep their direction small and grows when the tracker is far from the top (convex
response). The next lever to probe is the one whose recent first-order gain per cycle times the ticks since it
was last probed is largest, so a lever that has settled at its best setting is probed rarely and one whose best
setting keeps moving often. It never stops probing, so it keeps following an optimum that keeps moving.
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


def play(view, env=None) -> None:
    agent = view.agents()[0]
    names, lo, hi, m = _levers(view, agent)
    span = hi - lo
    d = 0.05 * span                                     # probe width per lever (lever units)
    j, phase, ys = 0, 0, [0.0, 0.0, 0.0]
    last = np.zeros(len(names))
    act = np.full(len(names), np.inf)                   # recent gain per probe cycle of each lever (untried: inf)
    since = np.zeros(len(names))                        # ticks since each lever was last probed
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
            d[j] = min(d[j] * 1.25, 0.1 * span[j])
        if step * last[j] < 0:                          # direction flipped: close to the top, probe finer
            d[j] = max(d[j] * 0.8, 0.02 * span[j])
        last[j] = step
        m[j] = float(np.clip(m[j] + step, lo[j], hi[j]))
        # first-order gain of this cycle's move; the next lever is the one whose recent gain times the time since
        # it was last probed is largest (a lever that has settled is probed rarely, a moving one often)
        gain = abs(g * step)
        act[j] = gain if not np.isfinite(act[j]) else 0.7 * act[j] + 0.3 * gain
        since += 3
        since[j] = 0
        j = int(np.argmax(np.where(np.isfinite(act), act, np.inf) * (since + 1)))
